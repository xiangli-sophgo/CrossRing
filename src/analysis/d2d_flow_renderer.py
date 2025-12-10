"""
D2D流量图渲染器 - 使用Plotly生成多Die可交互流量图

专注于D2D (Die-to-Die) 多Die网络的流量可视化：
- 多Die布局和对齐
- 跨Die连接可视化
- Die内部流量图
- 三通道切换
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


class D2DFlowRenderer(BaseFlowRenderer):
    """D2D流量图渲染器 - 专注于多Die流量可视化"""

    def __init__(self):
        """初始化D2D流量图渲染器"""
        super().__init__()

    def draw_d2d_flow_graph(
        self,
        die_networks: Dict = None,
        dies: Dict = None,
        config=None,
        die_ip_bandwidth_data: Dict = None,
        mode: str = "utilization",
        node_size: int = 2500,
        save_path: str = None,
        show_fig: bool = False,
        return_fig: bool = False,
        enable_channel_switch: bool = True,
        static_bandwidth: Dict = None,
    ):
        """
        绘制D2D系统流量图（多Die布局）- 交互式版本

        Args:
            die_networks: Die网络字典
            dies: Die模型字典
            config: 配置对象
            die_ip_bandwidth_data: D2D IP带宽数据
            mode: 可视化模式
            node_size: 节点大小
            save_path: 保存路径
            show_fig: 是否在浏览器中显示图像
            static_bandwidth: 静态带宽数据字典 {die_id: {link_key: bw}, d2d_key: bw}

        Returns:
            str: 保存的HTML文件路径，如果save_path为None则返回fig对象
        """
        # 兼容旧的调用方式
        if die_networks is not None and dies is None:
            dies = {}
            for die_id, network in die_networks.items():
                if hasattr(network, "die_model"):
                    dies[die_id] = network.die_model
                elif hasattr(network, "_die_model"):
                    dies[die_id] = network._die_model

            if not dies:
                die_networks_for_draw = die_networks
            else:
                die_networks_for_draw = {die_id: die_model.data_network for die_id, die_model in dies.items()}
        else:
            die_networks_for_draw = {die_id: die_model.data_network for die_id, die_model in dies.items()}

        # 获取布局配置
        die_layout = config.die_layout_positions
        die_layout_type = config.die_layout_type
        die_rotations = config.DIE_ROTATIONS

        # 计算Die尺寸和偏移量
        base_die_rows = 5
        base_die_cols = 4
        node_spacing = 3.0

        die_width = (base_die_cols - 1) * node_spacing
        die_height = (base_die_rows - 1) * node_spacing

        # 使用动态布局计算（复用原有逻辑）
        die_offsets, figsize = self._calculate_die_offsets_from_layout(die_layout, die_layout_type, die_width, die_height, dies=dies, config=config, die_rotations=die_rotations)

        # 创建plotly figure
        fig = go.Figure()

        # 收集所有IP类型和全局带宽范围
        all_ip_bandwidths = []
        ip_bandwidth_data_dict = {}
        used_ip_types = set()  # 收集所有使用的IP类型

        if die_ip_bandwidth_data:
            ip_bandwidth_data_dict = die_ip_bandwidth_data
            for die_id, die_data in die_ip_bandwidth_data.items():
                if mode in die_data:
                    mode_data = die_data[mode]
                    for ip_type, data_matrix in mode_data.items():
                        nonzero_bw = data_matrix[data_matrix > 0.001]
                        if len(nonzero_bw) > 0:
                            all_ip_bandwidths.extend(nonzero_bw.tolist())
                            used_ip_types.add(ip_type.upper())  # 记录使用的IP类型
        elif dies:
            for die_id, die_model in dies.items():
                if hasattr(die_model, "ip_bandwidth_data") and die_model.ip_bandwidth_data is not None:
                    ip_bandwidth_data_dict[die_id] = die_model.ip_bandwidth_data
                    if mode in die_model.ip_bandwidth_data:
                        mode_data = die_model.ip_bandwidth_data[mode]
                        for ip_type, data_matrix in mode_data.items():
                            nonzero_bw = data_matrix[data_matrix > 0.001]
                            if len(nonzero_bw) > 0:
                                all_ip_bandwidths.extend(nonzero_bw.tolist())
                                used_ip_types.add(ip_type.upper())  # 记录使用的IP类型

        # 计算全局IP带宽范围
        max_ip_bandwidth = max(all_ip_bandwidths) if all_ip_bandwidths else None
        min_ip_bandwidth = min(all_ip_bandwidths) if all_ip_bandwidths else None

        # 收集每个Die的三个网络（用于通道切换）
        die_req_networks = {}
        die_rsp_networks = {}
        die_data_networks = {}
        if enable_channel_switch and dies:
            for die_id, die_model in dies.items():
                if hasattr(die_model, "req_network") and die_model.req_network:
                    die_req_networks[die_id] = die_model.req_network
                if hasattr(die_model, "rsp_network") and die_model.rsp_network:
                    die_rsp_networks[die_id] = die_model.rsp_network
                if hasattr(die_model, "data_network") and die_model.data_network:
                    die_data_networks[die_id] = die_model.data_network

        # 为每个Die绘制流量图（节点和IP方块，暂不绘制links）
        die_node_positions = {}
        for die_id, network in die_networks_for_draw.items():
            offset_x, offset_y = die_offsets[die_id]
            die_model = dies.get(die_id) if dies else None
            die_rotation = die_rotations.get(die_id, 0)

            ip_bandwidth_data = ip_bandwidth_data_dict.get(die_id) if ip_bandwidth_data_dict else None

            node_positions = self.draw_single_die_flow(
                fig=fig,
                network=network,
                config=die_model.config if die_model else config,
                mode=mode,
                node_size=node_size,
                die_id=die_id,
                offset_x=offset_x,
                offset_y=offset_y,
                rotation=die_rotation,
                ip_bandwidth_data=ip_bandwidth_data_dict,
                max_ip_bandwidth=max_ip_bandwidth,
                min_ip_bandwidth=min_ip_bandwidth,
                is_d2d_scenario=True,
                show_ip_bandwidth_value=False,
                link_label_fontsize=8,
                draw_links=not enable_channel_switch,  # 如果启用通道切换，则不在这里绘制links
            )
            die_node_positions[die_id] = node_positions

        # 添加Die标签
        # 首先计算所有Die的中心X坐标，找出最左和最右的Die
        die_centers = {}
        for die_id in die_node_positions.keys():
            node_positions = die_node_positions[die_id]
            if node_positions:
                xs = [p[0] for p in node_positions.values()]
                die_center_x = (min(xs) + max(xs)) / 2
                die_centers[die_id] = (die_center_x, min(xs), max(xs))

        # 找出最左和最右的Die
        if die_centers:
            leftmost_die = min(die_centers.keys(), key=lambda d: die_centers[d][0])
            rightmost_die = max(die_centers.keys(), key=lambda d: die_centers[d][0])

        # 为每个Die添加标签
        for die_id in die_node_positions.keys():
            node_positions = die_node_positions[die_id]
            if node_positions:
                xs = [p[0] for p in node_positions.values()]
                ys = [p[1] for p in node_positions.values()]
                die_center_y = (min(ys) + max(ys)) / 2

                # 智能标签位置：最左侧Die放左边，最右侧Die放右边
                if die_id == leftmost_die:
                    label_x = min(xs) - 4
                elif die_id == rightmost_die:
                    label_x = max(xs) + 4
                else:
                    # 中间的Die根据位置决定
                    die_center_x = die_centers[die_id][0]
                    left_center = die_centers[leftmost_die][0]
                    right_center = die_centers[rightmost_die][0]
                    if abs(die_center_x - left_center) < abs(die_center_x - right_center):
                        label_x = min(xs) - 4
                    else:
                        label_x = max(xs) + 4

                label_y = die_center_y

                fig.add_annotation(
                    x=label_x,
                    y=label_y,
                    text=f"Die {die_id}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="lightblue",
                    borderpad=4,
                    opacity=0.7,
                )

        # 如果启用通道切换，分别为三个通道绘制links
        if enable_channel_switch and (die_req_networks or die_rsp_networks or die_data_networks):
            # 记录基础traces数量（节点等）
            num_base_traces = len(fig.data)

            # 初始化通道管理器（用于D2D多Die场景，传入die_networks字典）
            channel_manager = ChannelSwitchManager()
            # 为D2D场景准备网络字典 (每个通道包含多个Die的网络)
            networks_dict = {"请求": die_req_networks if die_req_networks else {}, "响应": die_rsp_networks if die_rsp_networks else {}, "数据": die_data_networks if die_data_networks else {}}

            # 保存原有annotations（Die标签等）
            existing_anns = list(fig.layout.annotations) if fig.layout.annotations else []

            for channel_name, die_networks in networks_dict.items():
                if not die_networks:
                    continue

                trace_start_idx = len(fig.data)
                annotation_start_idx = len(channel_manager.all_annotations)

                # 为每个Die绘制该通道的links
                for die_id, network in die_networks.items():
                    if die_id not in die_node_positions:
                        continue

                    node_positions = die_node_positions[die_id]
                    die_model = dies.get(die_id)

                    if not die_model:
                        continue

                    # 获取该Die的旋转角度
                    die_rotation = die_rotations.get(die_id, 0)

                    # 提取该Die的静态带宽数据
                    die_static_bw = None
                    if static_bandwidth and isinstance(static_bandwidth, dict):
                        die_static_bw = static_bandwidth.get(die_id, None)

                    # 使用_draw_channel_links_only绘制该通道的links
                    channel_anns = self._draw_channel_links_only(
                        fig=fig,
                        network=network,
                        config=die_model.config,
                        pos=node_positions,
                        mode=mode,
                        node_size=node_size,
                        draw_self_loops=True,
                        rotation=die_rotation,
                        is_d2d_scenario=True,
                        fontsize=8,
                        static_bandwidth=die_static_bw,
                    )
                    channel_manager.add_annotations(channel_anns)

                # 绘制该通道的跨Die连接（如果有多个Die）
                if dies and len(dies) > 1:
                    try:
                        d2d_bandwidth = self._calculate_d2d_sys_bandwidth(dies, config)
                        # 提取跨Die静态带宽数据
                        static_d2d_bw = None
                        if static_bandwidth and isinstance(static_bandwidth, dict):
                            static_d2d_bw = static_bandwidth.get('d2d', None)
                        cross_die_anns = self._draw_cross_die_connections(fig, d2d_bandwidth, die_node_positions, config, dies, die_offsets, channel_name=channel_name, static_d2d_bandwidth=static_d2d_bw)
                        if cross_die_anns:
                            channel_manager.add_annotations(cross_die_anns)
                    except Exception as e:
                        import traceback

                        traceback.print_exc()

                trace_end_idx = len(fig.data)
                annotation_end_idx = len(channel_manager.all_annotations)

                # 记录该通道的trace和annotation范围
                channel_manager.record_trace_range(channel_name, trace_start_idx, trace_end_idx)
                channel_manager.record_annotation_range(channel_name, annotation_start_idx, annotation_end_idx)

                # 设置初始可见性：默认显示数据通道
                for i in range(trace_start_idx, trace_end_idx):
                    fig.data[i].visible = channel_name == "数据"

            # 设置annotations初始可见性
            for i, ann in enumerate(channel_manager.all_annotations):
                ann_channel = None
                for ch_name in ["请求", "响应", "数据"]:
                    if ch_name not in channel_manager.annotation_indices:
                        continue
                    ann_start, ann_end = channel_manager.annotation_indices[ch_name]
                    if ann_start <= i < ann_end:
                        ann_channel = ch_name
                        break
                ann["visible"] = ann_channel == "数据"

            # 一次性添加所有annotations
            fig.layout.annotations = existing_anns + channel_manager.all_annotations

            # 创建通道切换按钮（传入基础traces数量和原有annotations数量）
            buttons = channel_manager.create_buttons(fig=fig, num_extra_traces=0, num_base_traces=num_base_traces, num_existing_annotations=len(existing_anns))  # 先不包含legend/colorbar，稍后更新

            # 暂不设置updatemenus，等添加完跨Die链路、legend、colorbar后再设置
            # （为了能在按钮的visibility数组中包含这些额外的traces）

        # 记录三通道traces结束位置（包括跨Die连接）
        num_traces_after_channels = len(fig.data) if enable_channel_switch and (die_req_networks or die_rsp_networks or die_data_networks) else 0

        # 如果没有启用三通道切换，绘制默认的跨Die连接
        if not (enable_channel_switch and (die_req_networks or die_rsp_networks or die_data_networks)):
            if dies and len(dies) > 1:
                try:
                    d2d_bandwidth = self._calculate_d2d_sys_bandwidth(dies, config)
                    # 提取跨Die静态带宽数据
                    static_d2d_bw = None
                    if static_bandwidth and isinstance(static_bandwidth, dict):
                        static_d2d_bw = static_bandwidth.get('d2d', None)
                    self._draw_cross_die_connections(fig, d2d_bandwidth, die_node_positions, config, dies, die_offsets, channel_name=None, static_d2d_bandwidth=static_d2d_bw)
                except Exception as e:
                    import traceback

                    traceback.print_exc()

        # 添加D2D节点的点击处理scatter点（D2D_RN和D2D_SN没有画出IP方块，需要单独添加点击区域）
        # 只在有实际带宽数据的D2D节点上添加点击区域
        d2d_pairs = getattr(config, 'D2D_PAIRS', [])
        if d2d_pairs and die_node_positions:
            d2d_click_x = []
            d2d_click_y = []
            d2d_click_text = []
            d2d_click_customdata = []

            # 计算square_size（与draw_single_die_flow保持一致）
            square_size = np.sqrt(node_size) / 50

            # 收集有带宽数据的D2D节点（去重）
            d2d_nodes_with_data = set()
            if die_ip_bandwidth_data:
                for die_id, die_data in die_ip_bandwidth_data.items():
                    if mode in die_data:
                        mode_data = die_data[mode]
                        # 检查d2d_rn和d2d_sn的带宽数据
                        for ip_type in ['d2d_rn', 'd2d_sn', 'D2D_RN', 'D2D_SN']:
                            if ip_type in mode_data:
                                data_matrix = mode_data[ip_type]
                                num_cols = data_matrix.shape[1]
                                # 找出有带宽的节点位置
                                for row in range(data_matrix.shape[0]):
                                    for col in range(num_cols):
                                        if data_matrix[row, col] > 0.001:
                                            node_pos = row * num_cols + col
                                            d2d_nodes_with_data.add((die_id, ip_type.lower(), node_pos))

            for die0_id, node0, die1_id, node1 in d2d_pairs:
                # Die0的D2D节点
                if die0_id in die_node_positions and node0 in die_node_positions[die0_id]:
                    x, y = die_node_positions[die0_id][node0]
                    # D2D_RN - 只在有数据时添加
                    if (die0_id, 'd2d_rn', node0) in d2d_nodes_with_data:
                        d2d_click_x.append(x - square_size * 0.35)
                        d2d_click_y.append(y + square_size * 0.15)
                        d2d_click_text.append(f"d2d_rn_0 @ Pos {node0}")
                        d2d_click_customdata.append([die0_id, 'd2d_rn_0', node0])
                    # D2D_SN - 只在有数据时添加
                    if (die0_id, 'd2d_sn', node0) in d2d_nodes_with_data:
                        d2d_click_x.append(x - square_size * 0.35)
                        d2d_click_y.append(y - square_size * 0.15)
                        d2d_click_text.append(f"d2d_sn_0 @ Pos {node0}")
                        d2d_click_customdata.append([die0_id, 'd2d_sn_0', node0])

                # Die1的D2D节点
                if die1_id in die_node_positions and node1 in die_node_positions[die1_id]:
                    x, y = die_node_positions[die1_id][node1]
                    # D2D_RN - 只在有数据时添加
                    if (die1_id, 'd2d_rn', node1) in d2d_nodes_with_data:
                        d2d_click_x.append(x - square_size * 0.35)
                        d2d_click_y.append(y + square_size * 0.15)
                        d2d_click_text.append(f"d2d_rn_0 @ Pos {node1}")
                        d2d_click_customdata.append([die1_id, 'd2d_rn_0', node1])
                    # D2D_SN - 只在有数据时添加
                    if (die1_id, 'd2d_sn', node1) in d2d_nodes_with_data:
                        d2d_click_x.append(x - square_size * 0.35)
                        d2d_click_y.append(y - square_size * 0.15)
                        d2d_click_text.append(f"d2d_sn_0 @ Pos {node1}")
                        d2d_click_customdata.append([die1_id, 'd2d_sn_0', node1])

            if d2d_click_x:
                fig.add_trace(
                    go.Scatter(
                        x=d2d_click_x,
                        y=d2d_click_y,
                        mode="markers",
                        marker=dict(
                            size=square_size * 12,
                            opacity=0,  # 完全透明，仅用于点击检测
                            color="rgba(0,0,0,0)",
                        ),
                        text=d2d_click_text,
                        customdata=d2d_click_customdata,
                        hoverinfo="text",
                        hoverlabel=dict(
                            bgcolor="white",
                            bordercolor="#333333",
                            font=dict(color="#333333"),
                        ),
                        showlegend=False,
                        name="D2D Click Handler",
                    )
                )

        # 添加IP类型Legend
        if used_ip_types:
            self._add_ip_legend_plotly(fig, used_ip_types)

        # 添加带宽Colorbar
        if all_ip_bandwidths and max_ip_bandwidth and min_ip_bandwidth:
            self._add_bandwidth_colorbar_plotly(fig, min_ip_bandwidth, max_ip_bandwidth)

        # 如果启用了三通道切换，现在更新按钮的visibility数组以包含跨Die链路、legend、colorbar
        if enable_channel_switch and (die_req_networks or die_rsp_networks or die_data_networks) and buttons:
            num_extra_traces = len(fig.data) - num_traces_after_channels
            if num_extra_traces > 0:
                # 为每个按钮的visibility数组添加额外的True值
                for button in buttons:
                    button["args"][0]["visible"].extend([True] * num_extra_traces)

            # 现在设置updatemenus
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=buttons,
                        pad={"r": 10, "t": 5},
                        showactive=True,
                        x=0.5,
                        xanchor="center",
                        y=1.05,  # 从1.15降到1.05，减小按钮与图表间距
                        yanchor="top",
                    )
                ]
            )

        # 设置布局
        canvas_width = int(figsize[0] * 100)
        canvas_height = int(figsize[1] * 100)

        layout_config = dict(
            showlegend=True,
            hovermode="closest",
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=50, t=40, b=20),  # 减小上边距t从50到40，减小左右边距更紧凑
            width=canvas_width,
            height=canvas_height,
            legend=dict(
                x=1.0,  # 放到右侧，与colorbar对齐
                y=0.8,  # 放到colorbar上方
                xanchor="left",
                yanchor="top",
            ),
        )

        # 只在非集成模式下显示标题
        if not return_fig:
            title = f"D2D Flow Graph - {mode.capitalize()}"
            layout_config["title"] = dict(text=title, font=dict(size=14))

        fig.update_layout(**layout_config)

        if return_fig:
            return fig

        if save_path:
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

    def _calculate_die_offsets_from_layout(self, die_layout, die_layout_type, die_width, die_height, dies=None, config=None, die_rotations=None):
        """
        复用原有的Die偏移量计算逻辑（包含智能对齐）
        """
        if not die_layout:
            raise ValueError("die_layout不能为空")

        if die_rotations is None:
            die_rotations = {}

        max_x = max(pos[0] for pos in die_layout.values()) if die_layout else 0
        max_y = max(pos[1] for pos in die_layout.values()) if die_layout else 0

        # 计算每个Die旋转后的实际尺寸
        die_sizes = {}
        for die_id in die_layout.keys():
            rotation = die_rotations.get(die_id, 0)
            if rotation in [90, 270]:
                die_sizes[die_id] = (die_height, die_width)
            else:
                die_sizes[die_id] = (die_width, die_height)

        # 计算每行每列的最大尺寸
        max_width_per_col = {}
        max_height_per_row = {}
        for die_id, (grid_x, grid_y) in die_layout.items():
            w, h = die_sizes[die_id]
            max_width_per_col[grid_x] = max(max_width_per_col.get(grid_x, 0), w)
            max_height_per_row[grid_y] = max(max_height_per_row.get(grid_y, 0), h)

        # 计算每个Die的偏移量
        die_offsets = {}
        gap_x = 7.0  # Die之间的横向间隙
        gap_y = 5.0 if len(die_layout.values()) == 2 else 1

        for die_id, (grid_x, grid_y) in die_layout.items():
            offset_x = sum(max_width_per_col.get(x, 0) + gap_x for x in range(grid_x))
            offset_y = sum(max_height_per_row.get(y, 0) + gap_y for y in range(grid_y))
            die_offsets[die_id] = (offset_x, offset_y)

        # 如果提供了dies和config，计算对齐偏移
        if dies and config:
            try:
                alignment_offsets = self._calculate_die_alignment_offsets(dies, config)

                # 应用对齐偏移
                for die_id, (base_x, base_y) in die_offsets.items():
                    if die_id in alignment_offsets:
                        align_x, align_y = alignment_offsets[die_id]
                        die_offsets[die_id] = (base_x + align_x, base_y + align_y)
            except Exception as e:
                # 对齐计算失败时使用默认布局
                print(f"[对齐优化] 对齐计算失败，使用默认布局: {e}")

        # 计算总的画布尺寸
        total_width = sum(max_width_per_col.get(x, 0) for x in range(max_x + 1)) + gap_x * max_x + 2
        total_height = sum(max_height_per_row.get(y, 0) for y in range(max_y + 1)) + gap_y * max_y + 2

        canvas_width = total_width * 0.3
        canvas_height = total_height * 0.3

        canvas_width = max(min(canvas_width, 20), 14)
        canvas_height = max(min(canvas_height, 16), 10)

        figsize = (canvas_width, canvas_height)

        return die_offsets, figsize

    def _calculate_d2d_sys_bandwidth(self, dies, config):
        """
        计算每个Die每个D2D节点的AXI通道带宽（复用原有逻辑）

        Args:
            dies: Dict[die_id, die_model] - Die模型字典
            config: 配置对象

        Returns:
            dict: D2D_Sys AXI通道带宽统计 {die_id: {node_pos: {channel: bandwidth_gbps}}}
        """
        d2d_sys_bandwidth = {}

        # 从dies中获取仿真周期
        sim_end_cycle = next(iter(dies.values())).cycle
        network_frequency = config.NETWORK_FREQUENCY
        flit_size = config.FLIT_SIZE
        time_ns = sim_end_cycle / network_frequency

        for die_id, die_model in dies.items():
            d2d_sys_bandwidth[die_id] = {}

            if hasattr(die_model, "d2d_systems"):
                for pos_key, d2d_sys in die_model.d2d_systems.items():
                    node_bandwidth = {"AR": 0.0, "R": 0.0, "AW": 0.0, "W": 0.0, "B": 0.0}

                    if hasattr(d2d_sys, "axi_channel_flit_count"):
                        for channel, flit_count in d2d_sys.axi_channel_flit_count.items():
                            if time_ns > 0:
                                bandwidth_gbps = (flit_count * flit_size) / time_ns
                                node_bandwidth[channel] = bandwidth_gbps

                    d2d_sys_bandwidth[die_id][pos_key] = node_bandwidth.copy()

        return d2d_sys_bandwidth

    def _get_connection_type(self, from_die_pos, to_die_pos):
        """
        判断D2D连接类型

        Args:
            from_die_pos: 源Die的网格位置 (x, y)
            to_die_pos: 目标Die的网格位置 (x, y)

        Returns:
            str: "vertical" | "horizontal" | "diagonal"
        """
        dx = abs(from_die_pos[0] - to_die_pos[0])
        dy = abs(from_die_pos[1] - to_die_pos[1])

        if dx == 0:
            return "vertical"
        elif dy == 0:
            return "horizontal"
        else:
            return "diagonal"

    def _draw_cross_die_connections(self, fig, d2d_bandwidth, die_node_positions, config, dies=None, die_offsets=None, channel_name=None, static_d2d_bandwidth=None):
        """
        绘制跨Die带宽连接 - Plotly版本

        Args:
            fig: plotly Figure对象
            d2d_bandwidth: D2D带宽数据（AXI通道统计）
            die_node_positions: 各Die的节点位置
            config: 配置对象
            dies: Die模型字典
            die_offsets: Die偏移量
            channel_name: 通道名称 ("请求"/"响应"/"数据"/None)
                - "请求": AR + AW（地址通道）
                - "数据": W + R（数据通道）
                - "响应": B（响应通道）
                - None: W + R（向后兼容）
            static_d2d_bandwidth: 静态D2D带宽数据 {(src_die, src_node, dst_die, dst_node): bandwidth_GB/s}

        Returns:
            list: 添加的annotation列表
        """
        added_annotations = []
        try:
            d2d_pairs = config.D2D_PAIRS
            if not d2d_pairs:
                return

            die_layout = config.die_layout_positions

            arrow_index = 0
            for die0_id, die0_node, die1_id, die1_node in d2d_pairs:
                # 双向检查流量并绘制
                directions = [
                    (die0_id, die0_node, die1_id, die1_node),  # Die0 -> Die1
                    (die1_id, die1_node, die0_id, die0_node),  # Die1 -> Die0
                ]

                for from_die, from_node, to_die, to_node in directions:
                    key = f"{from_node}_to_{to_die}_{to_node}"

                    # 获取所有AXI通道的带宽（用于hover显示）
                    all_channel_bw = {
                        "AR": d2d_bandwidth.get(from_die, {}).get(key, {}).get("AR", 0.0),
                        "AW": d2d_bandwidth.get(from_die, {}).get(key, {}).get("AW", 0.0),
                        "W": d2d_bandwidth.get(from_die, {}).get(key, {}).get("W", 0.0),
                        "R": d2d_bandwidth.get(from_die, {}).get(key, {}).get("R", 0.0),
                        "B": d2d_bandwidth.get(from_die, {}).get(key, {}).get("B", 0.0),
                    }

                    # 根据通道筛选带宽
                    if channel_name == "请求":
                        # 请求通道：AR + AW（地址通道）
                        ar_bw = all_channel_bw["AR"]
                        aw_bw = all_channel_bw["AW"]
                        total_bw = ar_bw + aw_bw
                        w_bw = aw_bw  # 写地址
                        r_bw = ar_bw  # 读地址
                    elif channel_name == "数据":
                        # 数据通道：W + R（数据通道）
                        w_bw = all_channel_bw["W"]
                        r_bw = all_channel_bw["R"]
                        total_bw = w_bw + r_bw
                    elif channel_name == "响应":
                        # 响应通道：B（响应通道）
                        b_bw = all_channel_bw["B"]
                        total_bw = b_bw
                        w_bw = b_bw  # 写响应
                        r_bw = 0.0
                    else:
                        # 默认（向后兼容）：W + R
                        w_bw = all_channel_bw["W"]
                        r_bw = all_channel_bw["R"]
                        total_bw = w_bw + r_bw

                    # 获取节点位置
                    from_die_positions = die_node_positions.get(from_die, {})
                    to_die_positions = die_node_positions.get(to_die, {})

                    if from_node not in from_die_positions or to_node not in to_die_positions:
                        continue

                    from_x, from_y = from_die_positions[from_node]
                    to_x, to_y = to_die_positions[to_node]

                    from_die_pos = die_layout.get(from_die, (0, 0))
                    to_die_pos = die_layout.get(to_die, (0, 0))
                    connection_type = self._get_connection_type(from_die_pos, to_die_pos)

                    # 对角连接：围绕连线中点顺时针旋转8度
                    if connection_type == "diagonal":
                        mid_x = (from_x + to_x) / 2
                        mid_y = (from_y + to_y) / 2

                        angle = -8 * np.pi / 180
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)

                        dx_from = from_x - mid_x
                        dy_from = from_y - mid_y
                        from_x = mid_x + dx_from * cos_a - dy_from * sin_a
                        from_y = mid_y + dx_from * sin_a + dy_from * cos_a

                        dx_to = to_x - mid_x
                        dy_to = to_y - mid_y
                        to_x = mid_x + dx_to * cos_a - dy_to * sin_a
                        to_y = mid_y + dx_to * sin_a + dy_to * cos_a

                    # 获取静态带宽（如果有）
                    static_bw = None
                    if static_d2d_bandwidth:
                        static_key = (from_die, from_node, to_die, to_node)
                        static_bw = static_d2d_bandwidth.get(static_key, None)

                    # 绘制D2D箭头（total_bw已经在上面根据通道计算好了）
                    anns = self._draw_single_d2d_arrow_plotly(
                        fig, from_x, from_y, to_x, to_y, total_bw, from_die, from_node, to_die, to_node, connection_type, w_bw, r_bw, channel_name=channel_name, all_channel_bw=all_channel_bw, static_bw=static_bw
                    )
                    if anns:
                        added_annotations.extend(anns)
                    arrow_index += 1

        except Exception as e:
            import traceback

            traceback.print_exc()

        return added_annotations

    def _draw_single_d2d_arrow_plotly(
        self, fig, start_x, start_y, end_x, end_y, total_bandwidth, from_die, from_node, to_die, to_node, connection_type=None, w_bw=0.0, r_bw=0.0, channel_name=None, all_channel_bw=None, static_bw=None
    ):
        """
        绘制单个D2D箭头（Plotly版本）

        Args:
            fig: plotly Figure对象
            start_x, start_y: 起始坐标
            end_x, end_y: 终点坐标
            total_bandwidth: 总带宽
            from_die, from_node: 源Die和节点
            to_die, to_node: 目标Die和节点
            connection_type: 连接类型
            w_bw: 写通道带宽
            r_bw: 读通道带宽
            channel_name: 通道名称（用于生成hover信息）
            all_channel_bw: 所有AXI通道的带宽字典 {"AR": xx, "AW": xx, "W": xx, "R": xx, "B": xx}
            static_bw: 静态带宽（GB/s）

        Returns:
            list: 添加的annotation列表（箭头annotation + 文本annotation）
        """
        added_annotations = []
        dx = end_x - start_x
        dy = end_y - start_y
        length = np.sqrt(dx * dx + dy * dy)

        if length == 0:
            return

        # 归一化方向向量
        ux, uy = dx / length, dy / length
        perpx, perpy = uy * 0.2, -ux * 0.2

        # 计算箭头起止坐标（留出节点空间）
        if connection_type == "diagonal":
            node_offset = 1.2
            perp_offset = 1.2
            arrow_start_x = start_x + ux * node_offset + perpx * perp_offset
            arrow_start_y = start_y + uy * node_offset + perpy * perp_offset
            arrow_end_x = end_x - ux * node_offset + perpx * perp_offset
            arrow_end_y = end_y - uy * node_offset + perpy * perp_offset
        else:
            arrow_start_x = start_x + ux * 1.2 + perpx
            arrow_start_y = start_y + uy * 1.2 + perpy
            arrow_end_x = end_x - ux * 1.2 + perpx
            arrow_end_y = end_y - uy * 1.2 + perpy

        # 确定颜色和样式
        if total_bandwidth > 0.001:
            intensity = min(total_bandwidth / MAX_BANDWIDTH_NORMALIZATION, 1.0)
            color_str = f"rgb({int(255*intensity)}, 0, 0)"
            label_text = f"{total_bandwidth:.1f}"
        else:
            color_str = "rgb(179, 179, 179)"  # 灰色
            label_text = None

        # 生成全面的hover文本（显示所有通道，然后突出当前通道）
        if all_channel_bw:
            hover_text = (
                f"D2D连接: Die{from_die}(节点{from_node}) → Die{to_die}(节点{to_node})<br>"
                f"<br>"
            )

            # 添加静态带宽信息（如果有）
            if static_bw is not None:
                hover_text += f"【静态带宽】: {static_bw:.2f} GB/s<br><br>"

            hover_text += (
                f"【所有AXI通道带宽】<br>"
                f"  AR (读地址): {all_channel_bw['AR']:.2f} GB/s<br>"
                f"  AW (写地址): {all_channel_bw['AW']:.2f} GB/s<br>"
                f"  W  (写数据): {all_channel_bw['W']:.2f} GB/s<br>"
                f"  R  (读数据): {all_channel_bw['R']:.2f} GB/s<br>"
                f"  B  (写响应): {all_channel_bw['B']:.2f} GB/s<br>"
            )

            # 根据当前通道添加高亮信息
            if channel_name == "请求":
                hover_text += f"<br>" f"【当前通道: 请求】<br>" f"  总带宽: {total_bandwidth:.2f} GB/s<br>" f"  写地址(AW): {w_bw:.2f} GB/s<br>" f"  读地址(AR): {r_bw:.2f} GB/s"
            elif channel_name == "响应":
                hover_text += f"<br>" f"【当前通道: 响应】<br>" f"  总带宽: {total_bandwidth:.2f} GB/s<br>" f"  写响应(B): {w_bw:.2f} GB/s"
            else:  # "数据"
                hover_text += f"<br>" f"【当前通道: 数据】<br>" f"  总带宽: {total_bandwidth:.2f} GB/s<br>" f"  写数据(W): {w_bw:.2f} GB/s<br>" f"  读数据(R): {r_bw:.2f} GB/s"
        else:
            raise ValueError("没有数据")

        arrow_ann = dict(
            x=arrow_end_x,
            y=arrow_end_y,
            ax=arrow_start_x,
            ay=arrow_start_y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.9,  # 从1.2缩小到0.9，箭头更精致
            arrowwidth=3.5,  # 从3增加到3.5，线条更粗
            arrowcolor=color_str,
            standoff=0,
            hovertext=hover_text,
        )
        added_annotations.append(arrow_ann)

        # 添加标签（只在有流量时）
        if label_text:
            if connection_type == "diagonal":
                label_x = arrow_start_x + (arrow_end_x - arrow_start_x) * 0.85
                label_y_base = arrow_start_y + (arrow_end_y - arrow_start_y) * 0.85
                if (dx > 0 and dy > 0) or (dx > 0 and dy < 0):
                    label_y = label_y_base - 0.6
                else:
                    label_y = label_y_base + 0.6
            else:
                mid_x = (arrow_start_x + arrow_end_x) / 2
                mid_y = (arrow_start_y + arrow_end_y) / 2
                is_horizontal = abs(dx) > abs(dy)

                if is_horizontal:
                    label_x = mid_x
                    label_y = mid_y + (-0.5 if dx > 0 else 0.5)
                else:
                    # 垂直链路：根据数字长度动态调整偏移量
                    label_x = mid_x + (-dy * 0.1 if dx > 0 else dy * 0.1)
                    # 根据数字位数调整偏移：数字越长，偏移越大
                    text_length = len(label_text)
                    if text_length <= 3:  # 例如 "1.0"
                        y_offset = 0.12
                    elif text_length <= 4:  # 例如 "10.0"
                        y_offset = 0.15
                    elif text_length <= 5:  # 例如 "100.0"
                        y_offset = 0.18
                    else:  # 更长的数字
                        y_offset = 0.22
                    label_y = mid_y - y_offset

            # 计算箭头角度（使用原始方向，与matplotlib版本保持一致）
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)

            # 只有对角线连接需要额外旋转90度
            if connection_type == "diagonal":
                # Plotly的textangle需要额外旋转90度才能与箭头平行
                angle_deg += 90

                # 确保文字不会倒置（在加90度后再判断）
                if angle_deg > 90:
                    angle_deg -= 180
                elif angle_deg < -90:
                    angle_deg += 180
            else:
                # 水平和垂直连接保持原角度
                # 确保文字不会倒置
                if angle_deg > 90:
                    angle_deg -= 180
                elif angle_deg < -90:
                    angle_deg += 180

            # 使用annotation添加旋转文本（支持textangle）
            text_ann = dict(
                x=label_x,
                y=label_y,
                text=label_text,
                showarrow=False,
                font=dict(size=12, color=color_str),  # 从10增大到12
                textangle=angle_deg,
                xref="x",
                yref="y",
            )
            added_annotations.append(text_ann)

            # 使用透明scatter提供hover功能（annotation的hover功能有限）
            fig.add_trace(
                go.Scatter(
                    x=[label_x],
                    y=[label_y],
                    mode="markers",
                    marker=dict(size=20, color="rgba(0,0,0,0)"),  # 完全透明的标记
                    hovertext=hover_text,
                    hoverinfo="text",
                    showlegend=False,
                    name="",
                )
            )

        return added_annotations

    def _apply_rotation(self, orig_row, orig_col, rows, cols, rotation):
        """
        根据旋转角度计算节点的旋转后行列位置

        Args:
            orig_row: 原始行号
            orig_col: 原始列号
            rows: 总行数
            cols: 总列数
            rotation: 旋转角度（0, 90, 180, 270）

        Returns:
            tuple: (new_row, new_col) 旋转后的行列位置
        """
        if rotation == 0 or abs(rotation) == 360:
            return orig_row, orig_col
        elif abs(rotation) == 90 or abs(rotation) == -270:
            # 顺时针90度
            return orig_col, rows - 1 - orig_row
        elif abs(rotation) == 180:
            # 180度
            return rows - 1 - orig_row, cols - 1 - orig_col
        elif abs(rotation) == 270 or abs(rotation) == -90:
            # 顺时针270度（逆时针90度）
            return cols - 1 - orig_col, orig_row
        else:
            return orig_row, orig_col

    def _calculate_die_alignment_offsets(self, dies, config):
        """
        根据D2D连接计算Die位置偏移，使连接线对齐

        Args:
            dies: Die模型字典 {die_id: die_model}
            config: 配置对象

        Returns:
            dict: {die_id: (offset_x, offset_y)} 额外的偏移量
        """
        d2d_pairs = config.D2D_PAIRS
        die_layout = config.die_layout_positions
        die_rotations = config.DIE_ROTATIONS

        if not d2d_pairs or not die_layout:
            return {}

        # 收集各Die对之间的偏移需求，每对Die只保留偏移量最大的连接
        alignment_constraints = {"vertical": {}, "horizontal": {}}

        for die0_id, die0_node, die1_id, die1_node in d2d_pairs:
            from_die_pos = die_layout.get(die0_id, (0, 0))
            to_die_pos = die_layout.get(die1_id, (0, 0))

            conn_type = self._get_connection_type(from_die_pos, to_die_pos)

            # 获取节点在各自Die内的物理位置
            die0_model = dies.get(die0_id)
            die1_model = dies.get(die1_id)

            if die0_model and die1_model:
                # 获取节点的原始行列位置和旋转角度
                die0_cols = die0_model.config.NUM_COL
                die0_rows = die0_model.config.NUM_ROW
                die1_cols = die1_model.config.NUM_COL
                die1_rows = die1_model.config.NUM_ROW

                die0_rotation = die_rotations.get(die0_id, 0)
                die1_rotation = die_rotations.get(die1_id, 0)

                # 计算原始行列位置
                die0_orig_row = die0_node // die0_cols
                die0_orig_col = die0_node % die0_cols
                die1_orig_row = die1_node // die1_cols
                die1_orig_col = die1_node % die1_cols

                # 计算旋转后的行列位置
                die0_row, die0_col = self._apply_rotation(die0_orig_row, die0_orig_col, die0_rows, die0_cols, die0_rotation)
                die1_row, die1_col = self._apply_rotation(die1_orig_row, die1_orig_col, die1_rows, die1_cols, die1_rotation)

                die_pair = (min(die0_id, die1_id), max(die0_id, die1_id))

                if conn_type == "vertical":
                    # 垂直连接：需要X对齐
                    die0_x = die0_col * 3
                    die1_x = die1_col * 3
                    offset_needed = abs(die0_x - die1_x)

                    # 只保留偏移量最大的连接
                    if die_pair not in alignment_constraints["vertical"] or offset_needed > alignment_constraints["vertical"][die_pair]["offset"]:
                        alignment_constraints["vertical"][die_pair] = {
                            "die0": die0_id,
                            "die1": die1_id,
                            "col0": die0_col,
                            "col1": die1_col,
                            "row0": die0_row,
                            "row1": die1_row,
                            "offset": offset_needed,
                        }

                elif conn_type == "horizontal":
                    # 水平连接：需要Y对齐
                    die0_y = die0_row * -3
                    die1_y = die1_row * -3
                    offset_needed = abs(die0_y - die1_y)

                    # 只保留偏移量最大的连接
                    if die_pair not in alignment_constraints["horizontal"] or offset_needed > alignment_constraints["horizontal"][die_pair]["offset"]:
                        alignment_constraints["horizontal"][die_pair] = {
                            "die0": die0_id,
                            "die1": die1_id,
                            "row0": die0_row,
                            "row1": die1_row,
                            "col0": die0_col,
                            "col1": die1_col,
                            "offset": offset_needed,
                        }

        # 计算最优偏移量，固定Die 0作为参考点
        die_offsets = {}
        for die_id in die_layout.keys():
            die_offsets[die_id] = [0.0, 0.0]  # [x_offset, y_offset]

        # 固定Die 0作为参考点
        reference_die = 0

        # 处理垂直对齐约束（X方向）
        for die_pair, constraint in alignment_constraints["vertical"].items():
            die0 = constraint["die0"]
            die1 = constraint["die1"]
            col0 = constraint["col0"]
            col1 = constraint["col1"]

            die0_x = col0 * 3
            die1_x = col1 * 3
            actual_x_diff = die0_x - die1_x

            # 只移动非参考Die
            if die0 == reference_die:
                die_offsets[die1][0] += actual_x_diff
            elif die1 == reference_die:
                die_offsets[die0][0] -= actual_x_diff
            else:
                # 两个都不是参考Die，选择ID较大的移动
                if die0 > die1:
                    die_offsets[die0][0] -= actual_x_diff
                else:
                    die_offsets[die1][0] += actual_x_diff

        # 处理水平对齐约束（Y方向）
        for die_pair, constraint in alignment_constraints["horizontal"].items():
            die0 = constraint["die0"]
            die1 = constraint["die1"]
            row0 = constraint["row0"]
            row1 = constraint["row1"]

            die0_y = row0 * -3
            die1_y = row1 * -3
            actual_y_diff = die0_y - die1_y

            # 只移动非参考Die
            if die0 == reference_die:
                die_offsets[die1][1] += actual_y_diff
            elif die1 == reference_die:
                die_offsets[die0][1] -= actual_y_diff
            else:
                # 两个都不是参考Die，选择ID较大的移动
                if die0 > die1:
                    die_offsets[die0][1] -= actual_y_diff
                else:
                    die_offsets[die1][1] += actual_y_diff

        # 转换为元组格式
        return {die_id: tuple(offsets) for die_id, offsets in die_offsets.items()}

    def _add_ip_legend_plotly(self, fig, used_ip_types):
        """添加IP类型颜色Legend（Plotly版本）"""
        # 过滤和归一化IP类型
        processed_types = set()
        for ip_type in used_ip_types:
            # 过滤掉 D2D_RN 和 D2D_SN
            if ip_type in ["D2D_RN", "D2D_SN"]:
                continue

            # 移除编号后缀（如 DDR_0 -> DDR, GDMA_1 -> GDMA），并转为大写
            base_type = ip_type.split("_")[0] if "_" in ip_type else ip_type
            base_type = base_type.upper()  # 确保大写以匹配IP_COLOR_MAP
            processed_types.add(base_type)

        # 为每个基础IP类型添加一个虚拟的scatter trace用于legend
        for ip_type in sorted(processed_types):
            color = IP_COLOR_MAP.get(ip_type, IP_COLOR_MAP["OTHER"])
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color, symbol="square", line=dict(color="black", width=1)),
                    name=ip_type,
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    def _add_bandwidth_colorbar_plotly(self, fig, min_bw, max_bw):
        """添加带宽透明度对应关系Colorbar（Plotly版本）"""
        import numpy as np

        if max_bw <= min_bw:
            return

        # 创建一个虚拟的scatter trace用于colorbar
        # 使用灰度渐变
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=0.1,
                    color=[min_bw, max_bw],  # 数值范围
                    colorscale=[[0, "#E0E0E0"], [0.25, "#B0B0B0"], [0.5, "#808080"], [0.75, "#505050"], [1, "#202020"]],  # 浅灰  # 深灰
                    cmin=min_bw,
                    cmax=max_bw,
                    colorbar=dict(
                        title=dict(text="IP BW<br>(GB/s)", side="right", font=dict(size=10)),
                        tickfont=dict(size=9),
                        len=0.3,  # colorbar长度
                        thickness=15,  # colorbar宽度
                        x=1.0,  # 位置：从1.02调整到1.0，更靠近图表
                        y=0.5,  # 垂直居中
                        xanchor="left",
                        yanchor="middle",
                    ),
                    showscale=True,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
