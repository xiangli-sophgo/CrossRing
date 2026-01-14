"""
单Die流量图渲染器 - 使用Plotly生成单Die NoC可交互流量图

专注于单Die网络的流量可视化：
- 节点、链路、IP方块可视化
- 三通道切换（请求/响应/数据）
- Hover交互显示详细信息
- HTML输出

依赖flow_render_helpers.py提供的工具类
"""

import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import os
from functools import lru_cache

from .analyzers import IP_COLOR_MAP, MAX_BANDWIDTH_NORMALIZATION, RN_TYPES, SN_TYPES
from .flow_render_helpers import ChannelSwitchManager, LinkDataProcessor, IPLayoutCalculator
from .base_flow_renderer import BaseFlowRenderer


@lru_cache(maxsize=8)
def _calculate_node_positions_cached(num_rows: int, num_cols: int, num_nodes: int, rotation: int, offset_x: float, offset_y: float, node_spacing: float = 3.0) -> tuple:
    """
    计算节点位置（带缓存优化）

    Args:
        num_rows: 原始拓扑行数
        num_cols: 原始拓扑列数
        num_nodes: 节点总数
        rotation: 旋转角度
        offset_x: X轴偏移
        offset_y: Y轴偏移
        node_spacing: 节点间距

    Returns:
        tuple: ((node_id, (x, y)), ...) 节点位置元组，可哈希
    """
    pos_list = []
    for node in range(num_nodes):
        # 计算原始坐标
        orig_row = node // num_cols
        orig_col = node % num_cols

        # 根据旋转角度变换坐标
        if rotation == 0 or abs(rotation) == 360:
            new_row = orig_row
            new_col = orig_col
        elif abs(rotation) == 90 or abs(rotation) == -270:
            new_row = orig_col
            new_col = num_rows - 1 - orig_row
        elif abs(rotation) == 180:
            new_row = num_rows - 1 - orig_row
            new_col = num_cols - 1 - orig_col
        elif abs(rotation) == 270 or abs(rotation) == -90:
            new_row = num_cols - 1 - orig_col
            new_col = orig_row
        else:
            new_row = orig_row
            new_col = orig_col

        # 计算实际位置
        x = new_col * node_spacing + offset_x
        y = -new_row * node_spacing + offset_y
        pos_list.append((node, (x, y)))

    return tuple(pos_list)


class SingleDieFlowRenderer(BaseFlowRenderer):
    """单Die流量图渲染器 - 专注于单Die NoC流量可视化"""

    def __init__(self):
        """初始化单Die流量图渲染器"""
        super().__init__()

    def draw_flow_graph(
        self,
        networks: Dict,
        ip_bandwidth_data: Dict = None,
        config=None,
        mode: str = "utilization",
        node_size: int = 2000,
        save_path: str = None,
        show_fig: bool = False,
        return_fig: bool = False,
        static_bandwidth=None,
    ):
        """
        绘制单Die网络流量图(交互式版本，支持多通道)

        Args:
            networks: 多通道networks字典 {"req": [...], "rsp": [...], "data": [...]}
                     每个值是一个列表，包含该网络类型的所有通道
            ip_bandwidth_data: IP带宽数据字典
            config: 配置对象
            mode: 可视化模式
            node_size: 节点大小
            save_path: 保存路径（如果为None则返回fig对象）
            show_fig: 是否在浏览器中显示图像
            return_fig: 是否返回Figure对象而不是保存文件
            static_bandwidth: 静态带宽数据字典

        Returns:
            Figure对象
        """
        import time

        # 从第一个可用的网络获取config
        if config is None:
            for net_type in ["data", "req", "rsp"]:
                if net_type in networks and len(networks[net_type]) > 0:
                    if hasattr(networks[net_type][0], "config"):
                        config = networks[net_type][0].config
                        break

        # 计算全局IP带宽范围（用于透明度归一化）
        max_ip_bandwidth = None
        min_ip_bandwidth = None
        if ip_bandwidth_data is not None and mode in ip_bandwidth_data:
            all_ip_bandwidths = []
            mode_data = ip_bandwidth_data[mode]
            for ip_type, data_matrix in mode_data.items():
                nonzero_bw = data_matrix[data_matrix > 0.001]
                if len(nonzero_bw) > 0:
                    all_ip_bandwidths.extend(nonzero_bw.tolist())
            if all_ip_bandwidths:
                max_ip_bandwidth = max(all_ip_bandwidths)
                min_ip_bandwidth = min(all_ip_bandwidths)

        # 创建图形
        fig = go.Figure()

        # 检测每个网络类型的通道数
        channel_counts = {}
        for net_type in ["req", "rsp", "data"]:
            if net_type in networks:
                channel_counts[net_type] = len(networks[net_type])
            else:
                channel_counts[net_type] = 0

        max_channels = max(channel_counts.values()) if channel_counts else 0

        # 使用data网络的第一个通道绘制基础结构（nodes和IP信息）
        base_trace_count = len(fig.data)
        base_network = networks["data"][0] if "data" in networks and len(networks["data"]) > 0 else None

        pos = self.draw_single_die_flow(
            fig=fig,
            network=base_network,
            config=config,
            ip_bandwidth_data=ip_bandwidth_data,
            mode=mode,
            node_size=node_size,
            max_ip_bandwidth=max_ip_bandwidth,
            min_ip_bandwidth=min_ip_bandwidth,
            draw_links=False,  # 不绘制links，只绘制nodes
        )

        # 基础traces（nodes等）的数量
        num_base_traces = len(fig.data) - base_trace_count

        # 记录所有traces和annotations的范围
        # trace_ranges: {(net_type, ch_idx): (start, end)}
        # annotation_ranges: {(net_type, ch_idx): (start, end)}
        trace_ranges = {}
        annotation_ranges = {}
        all_annotations = []

        # 为所有网络类型的所有通道绘制links
        for net_type in ["req", "rsp", "data"]:
            if net_type not in networks or len(networks[net_type]) == 0:
                continue

            for ch_idx, network in enumerate(networks[net_type]):
                trace_start_idx = len(fig.data)
                annotation_start_idx = len(all_annotations)

                # 只绘制该通道的links，返回annotations而不是直接添加
                channel_anns = self._draw_channel_links_only(
                    fig=fig,
                    network=network,
                    config=config,
                    pos=pos,
                    mode=mode,
                    node_size=node_size,
                    draw_self_loops=True,  # 所有通道都绘制自环
                    static_bandwidth=static_bandwidth,
                )

                # 收集annotations
                all_annotations.extend(channel_anns)

                trace_end_idx = len(fig.data)
                annotation_end_idx = len(all_annotations)

                # 记录trace和annotation范围
                trace_ranges[(net_type, ch_idx)] = (trace_start_idx, trace_end_idx)
                annotation_ranges[(net_type, ch_idx)] = (annotation_start_idx, annotation_end_idx)

                # 设置初始可见性：默认显示data网络的第一个通道
                is_visible = net_type == "data" and ch_idx == 0
                for i in range(trace_start_idx, trace_end_idx):
                    fig.data[i].visible = is_visible

        # 一次性添加所有annotations
        existing_anns = list(fig.layout.annotations) if fig.layout.annotations else []

        # 设置初始可见性：默认显示data网络的第一个通道
        for i, ann in enumerate(all_annotations):
            ann_key = None
            for key, (ann_start, ann_end) in annotation_ranges.items():
                if ann_start <= i < ann_end:
                    ann_key = key
                    break
            ann["visible"] = (ann_key == ("data", 0)) if ann_key else False

        fig.layout.annotations = existing_anns + all_annotations

        # 创建两层按钮：网络类型 + 通道
        network_buttons = []
        channel_buttons = []

        num_existing_anns = len(existing_anns)

        # 第一层：网络类型按钮（REQ/RSP/DATA）
        for net_type in ["req", "rsp", "data"]:
            if channel_counts[net_type] == 0:
                continue

            # 计算总trace数量（暂不包括legend/colorbar，稍后添加）
            total_traces = num_base_traces
            for key in trace_ranges.keys():
                s, e = trace_ranges[key]
                total_traces = max(total_traces, e)

            # 创建visibility数组：使用绝对索引
            visibility = [False] * total_traces
            # 基础traces始终可见
            for i in range(num_base_traces):
                visibility[i] = True
            # 只显示当前网络类型的第一个通道
            for key in trace_ranges.keys():
                s, e = trace_ranges[key]
                is_visible = key == (net_type, 0)
                for i in range(s, e):
                    visibility[i] = is_visible

            # 创建annotations visibility
            updated_annotations = []
            for i, ann in enumerate(fig.layout.annotations):
                ann_dict = ann.to_plotly_json() if hasattr(ann, "to_plotly_json") else dict(ann)
                if i < num_existing_anns:
                    # 原有annotations（IP信息等），始终可见
                    ann_dict["visible"] = True
                else:
                    # 新添加的channel annotations
                    channel_ann_idx = i - num_existing_anns
                    ann_key = None
                    for key, (ann_start, ann_end) in annotation_ranges.items():
                        if ann_start <= channel_ann_idx < ann_end:
                            ann_key = key
                            break
                    # 只显示当前网络类型的第一个通道
                    ann_dict["visible"] = (ann_key == (net_type, 0)) if ann_key else False
                updated_annotations.append(ann_dict)

            network_buttons.append(dict(label=net_type.upper(), method="update", args=[{"visible": visibility}, {"annotations": updated_annotations}]))

        # 第二层：通道按钮（Ch0/Ch1/Ch2...）- 如果有多通道
        if max_channels > 1:
            for ch_idx in range(max_channels):
                # 注意：这个按钮需要JavaScript动态处理，因为不同网络类型的通道数不同
                # 这里先创建占位按钮，实际切换逻辑由JavaScript处理
                channel_buttons.append(
                    dict(
                        label=f"Ch{ch_idx}",
                        method="skip",  # 使用skip方法，由JavaScript处理
                    )
                )

        # 创建updatemenus
        updatemenus = [
            dict(
                buttons=network_buttons,
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=True,
                active=2,  # 默认选中DATA
                x=0.5,
                y=1.12,
                xanchor="center",
                yanchor="top",
                bgcolor="#f0f0f0",
                bordercolor="#ccc",
                font=dict(size=12, color="#333"),
                type="buttons",
            )
        ]

        # 如果有多通道，添加通道按钮组
        if max_channels > 1:
            updatemenus.append(
                dict(
                    buttons=channel_buttons,
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    active=0,  # 默认选中Ch0
                    x=0.7,
                    y=1.12,
                    xanchor="center",
                    yanchor="top",
                    bgcolor="#f0f0f0",
                    bordercolor="#ccc",
                    font=dict(size=12, color="#333"),
                    type="buttons",
                )
            )

        # 收集使用的IP类型(用于legend)
        used_ip_types = set()
        if ip_bandwidth_data is not None and mode in ip_bandwidth_data:
            mode_data = ip_bandwidth_data[mode]
            for ip_type, data_matrix in mode_data.items():
                if data_matrix.sum() > 0.001:  # 只包含有数据的IP类型
                    used_ip_types.add(ip_type)

        # 记录添加legend/colorbar前的trace数量
        num_traces_before_legend = len(fig.data)

        # 添加IP类型Legend
        if used_ip_types:
            self._add_ip_legend_plotly(fig, used_ip_types)

        # 添加带宽Colorbar
        if max_ip_bandwidth and min_ip_bandwidth:
            self._add_bandwidth_colorbar_plotly(fig, min_ip_bandwidth, max_ip_bandwidth)

        # 记录legend/colorbar的trace数量
        num_legend_colorbar_traces = len(fig.data) - num_traces_before_legend

        # 修正网络类型按钮的visibility数组，确保legend/colorbar始终可见
        if updatemenus:
            for button in updatemenus[0]["buttons"]:
                # 在visibility数组末尾添加True，使legend/colorbar始终可见
                button["args"][0]["visible"] = button["args"][0]["visible"] + [True] * num_legend_colorbar_traces

        # 设置布局
        layout_config = dict(
            showlegend=True,
            hovermode="closest",
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, constrain="domain"),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1, constrain="domain"),
            margin=dict(l=50, r=50, t=80, b=50, autoexpand=True),
            width=1200,
            height=1000,
            autosize=True,
        )

        # 添加网络类型和通道切换按钮
        if updatemenus:
            layout_config["updatemenus"] = updatemenus

        fig.update_layout(**layout_config)

        # 生成JavaScript代码（如果有多通道）
        js_code = None
        if max_channels > 1:
            js_code = self._generate_flow_chart_javascript(
                trace_ranges=trace_ranges,
                annotation_ranges=annotation_ranges,
                channel_counts=channel_counts,
                num_base_traces=num_base_traces,
                num_existing_anns=num_existing_anns,
                num_legend_colorbar_traces=num_legend_colorbar_traces,
            )

        if return_fig:
            # 返回Figure对象和JavaScript代码（如果有多通道）
            if js_code:
                return fig, js_code
            return fig

        if save_path:
            # 生成HTML文件
            if not save_path.endswith(".html"):
                save_path = save_path.replace(".png", ".html").replace(".jpg", ".html")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if js_code:
                # 注入JavaScript到HTML
                html_string = fig.to_html(include_plotlyjs="cdn", config={"displayModeBar": True})
                html_string = html_string.replace("</body>", js_code + "</body>")
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(html_string)
            else:
                fig.write_html(save_path, include_plotlyjs="cdn", config={"displayModeBar": True})

        if show_fig:
            fig.show()

        if save_path:
            return save_path
        else:
            return fig

    def _generate_flow_chart_javascript(
        self,
        trace_ranges: Dict,
        annotation_ranges: Dict,
        channel_counts: Dict,
        num_base_traces: int,
        num_existing_anns: int,
        num_legend_colorbar_traces: int,
    ) -> str:
        """生成流量图的JavaScript代码，处理通道切换逻辑"""

        # 将trace_ranges转换为JavaScript可用的格式
        # trace_ranges_js: {net_type: {ch_idx: [start, end]}}
        trace_ranges_js = {}
        for (net_type, ch_idx), (start, end) in trace_ranges.items():
            if net_type not in trace_ranges_js:
                trace_ranges_js[net_type] = {}
            trace_ranges_js[net_type][ch_idx] = [start, end]

        annotation_ranges_js = {}
        for (net_type, ch_idx), (start, end) in annotation_ranges.items():
            if net_type not in annotation_ranges_js:
                annotation_ranges_js[net_type] = {}
            annotation_ranges_js[net_type][ch_idx] = [start, end]

        js_code = f"""
<script>
    // 流量图多通道切换逻辑
    const traceRanges = {str(trace_ranges_js).replace("'", '"')};
    const annotationRanges = {str(annotation_ranges_js).replace("'", '"')};
    const channelCounts = {str(channel_counts).replace("'", '"')};
    const numBaseTraces = {num_base_traces};
    const numExistingAnns = {num_existing_anns};
    const numLegendColorbarTraces = {num_legend_colorbar_traces};

    let currentNetworkType = "data";
    let currentChannelIdx = 0;

    document.addEventListener('DOMContentLoaded', function() {{
        setTimeout(function() {{
            const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            if (!plotDiv) return;

            const allButtons = plotDiv.querySelectorAll('.updatemenu-button');
            const networkButtons = Array.from(allButtons).slice(0, 3);
            const maxChannels = Math.max(...Object.values(channelCounts));
            const channelButtons = Array.from(allButtons).slice(3, 3 + maxChannels);

            // 监听通道按钮点击
            channelButtons.forEach((btn, idx) => {{
                btn.addEventListener('click', function(e) {{
                    setTimeout(() => {{
                        currentChannelIdx = idx;
                        switchToChannel(currentNetworkType, currentChannelIdx);
                        channelButtons.forEach(b => b.classList.remove('active'));
                        this.classList.add('active');
                    }}, 10);
                }});
            }});

            // 监听网络类型按钮点击
            networkButtons.forEach((btn, idx) => {{
                btn.addEventListener('click', function(e) {{
                    const networks = ['req', 'rsp', 'data'];
                    setTimeout(() => {{
                        currentNetworkType = networks[idx];
                        updateChannelButtonsVisibility();

                        // 检查新网络类型是否有当前选择的通道
                        let targetChannel = currentChannelIdx;
                        if (!traceRanges[currentNetworkType] || !traceRanges[currentNetworkType][targetChannel]) {{
                            // 如果当前通道不存在，切换到Ch0
                            targetChannel = 0;
                            currentChannelIdx = 0;
                            channelButtons.forEach(b => b.classList.remove('active'));
                            if (channelButtons[0]) channelButtons[0].classList.add('active');
                        }}

                        // 主动切换到目标通道（覆盖Plotly的默认行为）
                        switchToChannel(currentNetworkType, targetChannel);
                    }}, 100);
                }});
            }});

            function switchToChannel(netType, chIdx) {{
                if (!traceRanges[netType] || !traceRanges[netType][chIdx]) return;

                // 计算链路traces的最大结束索引
                let maxLinkTraceEnd = numBaseTraces;
                for (let net of ['req', 'rsp', 'data']) {{
                    if (!traceRanges[net]) continue;
                    for (let ch in traceRanges[net]) {{
                        const [start, end] = traceRanges[net][ch];
                        maxLinkTraceEnd = Math.max(maxLinkTraceEnd, end);
                    }}
                }}

                // 总trace数 = 链路traces结束位置 + legend/colorbar traces数量
                const totalTraces = maxLinkTraceEnd + numLegendColorbarTraces;

                // 初始化visibility数组，默认所有为false
                const visibility = new Array(totalTraces).fill(false);

                // 基础traces始终可见
                for (let i = 0; i < numBaseTraces; i++) {{
                    visibility[i] = true;
                }}

                // 设置当前通道的链路traces可见
                for (let net of ['req', 'rsp', 'data']) {{
                    if (!traceRanges[net]) continue;
                    for (let ch in traceRanges[net]) {{
                        const [start, end] = traceRanges[net][ch];
                        const isVisible = (net === netType && parseInt(ch) === chIdx);
                        for (let i = start; i < end; i++) {{
                            visibility[i] = isVisible;
                        }}
                    }}
                }}

                // Legend和colorbar traces始终可见（在链路traces之后）
                for (let i = maxLinkTraceEnd; i < totalTraces; i++) {{
                    visibility[i] = true;
                }}

                const updatedAnnotations = [];
                if (plotDiv.layout.annotations) {{
                    for (let i = 0; i < plotDiv.layout.annotations.length; i++) {{
                        const ann = plotDiv.layout.annotations[i];
                        let annVisible = true;
                        if (i >= numExistingAnns) {{
                            const channelAnnIdx = i - numExistingAnns;
                            annVisible = false;
                            if (annotationRanges[netType] && annotationRanges[netType][chIdx]) {{
                                const [start, end] = annotationRanges[netType][chIdx];
                                if (channelAnnIdx >= start && channelAnnIdx < end) annVisible = true;
                            }}
                        }}
                        updatedAnnotations.push({{...ann, visible: annVisible}});
                    }}
                }}

                Plotly.update(plotDiv, {{visible: visibility}}, {{annotations: updatedAnnotations}});
            }}

            function updateChannelButtonsVisibility() {{
                const currentChannels = channelCounts[currentNetworkType] || 1;
                channelButtons.forEach((btn, idx) => {{
                    btn.style.display = (idx < currentChannels) ? '' : 'none';
                }});
            }}

            updateChannelButtonsVisibility();
        }}, 500);
    }});
</script>
"""
        return js_code
