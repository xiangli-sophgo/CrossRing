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
        network,
        ip_bandwidth_data: Dict = None,
        config=None,
        mode: str = "utilization",
        node_size: int = 2000,
        save_path: str = None,
        show_fig: bool = False,
        return_fig: bool = False,
        req_network=None,
        rsp_network=None,
        static_bandwidth=None,
    ):
        """
        绘制单Die网络流量图(交互式版本)

        Args:
            network: Network对象（data_network，用于向后兼容）
            ip_bandwidth_data: IP带宽数据字典
            config: 配置对象
            mode: 可视化模式
            node_size: 节点大小
            save_path: 保存路径（如果为None则返回fig对象）
            show_fig: 是否在浏览器中显示图像
            return_fig: 是否返回Figure对象而不是保存文件
            req_network: 请求网络对象（可选，用于通道分离显示）
            rsp_network: 响应网络对象（可选，用于通道分离显示）
            static_bandwidth: 静态带宽数据字典

        Returns:
            str or Figure: 如果return_fig=True，返回Figure对象；否则返回保存路径或fig对象
        """
        import time

        # 如果没有传入config，尝试从network获取
        if config is None and hasattr(network, "config"):
            config = network.config

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

        # 判断是否启用三通道分离显示
        enable_channel_switch = req_network is not None and rsp_network is not None

        if enable_channel_switch:
            # 初始化通道管理器
            channel_manager = ChannelSwitchManager()
            networks_dict = channel_manager.setup_channels(req_network, rsp_network, network)

            # 三通道模式：先用data_network绘制基础结构（nodes、annotations等）
            base_trace_count = len(fig.data)

            # 使用data_network绘制基础结构（nodes和IP信息）
            pos = self.draw_single_die_flow(
                fig=fig,
                network=network,
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

            # 为三个通道分别绘制links
            for channel_name, net in networks_dict.items():
                trace_start_idx = len(fig.data)
                annotation_start_idx = len(channel_manager.all_annotations)

                # 只绘制该通道的links，返回annotations而不是直接添加
                channel_anns = self._draw_channel_links_only(
                    fig=fig,
                    network=net,
                    config=config,
                    pos=pos,
                    mode=mode,
                    node_size=node_size,
                    draw_self_loops=True,  # 所有通道都绘制自环
                    static_bandwidth=static_bandwidth,
                )

                # 收集annotations到管理器
                channel_manager.add_annotations(channel_anns)

                trace_end_idx = len(fig.data)
                annotation_end_idx = len(channel_manager.all_annotations)

                # 记录通道的trace和annotation范围
                channel_manager.record_trace_range(channel_name, trace_start_idx, trace_end_idx)
                channel_manager.record_annotation_range(channel_name, annotation_start_idx, annotation_end_idx)

                # 设置初始可见性：默认显示数据通道
                for i in range(trace_start_idx, trace_end_idx):
                    fig.data[i].visible = channel_name == "数据"

            # 一次性添加所有annotations
            existing_anns = list(fig.layout.annotations) if fig.layout.annotations else []

            # 设置初始可见性
            for i, ann in enumerate(channel_manager.all_annotations):
                ann_channel = None
                for ch_name in ChannelSwitchManager.CHANNEL_NAMES:
                    if ch_name not in channel_manager.annotation_indices:
                        continue
                    ann_start, ann_end = channel_manager.annotation_indices[ch_name]
                    if ann_start <= i < ann_end:
                        ann_channel = ch_name
                        break
                ann["visible"] = ann_channel == "数据"

            fig.layout.annotations = existing_anns + channel_manager.all_annotations

            # 创建通道切换按钮（自动处理visibility逻辑）
            buttons = []
            for channel_name in ChannelSwitchManager.CHANNEL_NAMES:
                if channel_name not in channel_manager.trace_indices:
                    continue

                # 创建visibility数组：基础traces始终可见，只切换link traces
                visibility = [True] * num_base_traces
                for ch_name in ChannelSwitchManager.CHANNEL_NAMES:
                    if ch_name not in channel_manager.trace_indices:
                        continue
                    s, e = channel_manager.trace_indices[ch_name]
                    visibility.extend([ch_name == channel_name] * (e - s))

                # 使用管理器创建annotation visibility（但需要手动调整因为有existing_anns）
                num_existing_anns = len(existing_anns)
                updated_annotations = []
                if fig.layout.annotations:
                    for i, ann in enumerate(fig.layout.annotations):
                        ann_dict = ann.to_plotly_json() if hasattr(ann, 'to_plotly_json') else dict(ann)
                        if i < num_existing_anns:
                            # 原有annotations（如IP信息），始终可见
                            ann_dict["visible"] = True
                        else:
                            # 新添加的channel annotations
                            channel_ann_idx = i - num_existing_anns
                            ann_channel = None
                            for ch_name in ChannelSwitchManager.CHANNEL_NAMES:
                                if ch_name not in channel_manager.annotation_indices:
                                    continue
                                ann_start, ann_end = channel_manager.annotation_indices[ch_name]
                                if ann_start <= channel_ann_idx < ann_end:
                                    ann_channel = ch_name
                                    break
                            ann_dict["visible"] = ann_channel == channel_name
                        updated_annotations.append(ann_dict)

                buttons.append(dict(
                    label=channel_name,
                    method="update",
                    args=[{"visible": visibility}, {"annotations": updated_annotations}]
                ))

            updatemenus = [
                dict(
                    buttons=buttons,
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    active=2,  # 默认选中"数据"通道
                    x=0.5,
                    y=1.12,
                    xanchor="center",
                    yanchor="top",
                    bgcolor="lightblue",
                    bordercolor="blue",
                    font=dict(size=12),
                    type="buttons",
                )
            ]
        else:
            # 单通道模式（向后兼容）
            self.draw_single_die_flow(
                fig=fig,
                network=network,
                config=config,
                ip_bandwidth_data=ip_bandwidth_data,
                mode=mode,
                node_size=node_size,
                max_ip_bandwidth=max_ip_bandwidth,
                min_ip_bandwidth=min_ip_bandwidth,
            )
            updatemenus = None

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

        # 修正三通道模式的visibility数组，确保legend/colorbar始终可见
        if enable_channel_switch and updatemenus:
            for button in updatemenus[0]["buttons"]:
                # 在visibility数组末尾添加True，使legend/colorbar始终可见
                button["args"][0]["visible"].extend([True] * num_legend_colorbar_traces)

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

        # 添加通道切换按钮
        if enable_channel_switch and updatemenus:
            layout_config["updatemenus"] = updatemenus

        fig.update_layout(**layout_config)

        if return_fig:
            return fig

        if save_path:
            # 生成HTML文件
            if not save_path.endswith(".html"):
                save_path = save_path.replace(".png", ".html").replace(".jpg", ".html")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path, include_plotlyjs="cdn", config={"displayModeBar": True})

        if show_fig:
            fig.show()

        if save_path:
            return save_path
        else:
            return fig


