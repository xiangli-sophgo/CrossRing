"""
交互式流量图渲染器 - 使用Plotly生成可交互的HTML流量图

基于matplotlib版本的flow_graph_renderer.py重构，完全保留所有几何计算逻辑，
使用plotly替换matplotlib实现交互式可视化。

主要功能:
1. 单Die流量图 - 节点、链路、IP方块可视化
2. D2D多Die流量图 - 跨Die连接可视化
3. Hover交互 - 节点/链路/IP方块详细信息显示
4. HTML输出 - 可交互的独立HTML文件

技术要点:
- 使用plotly.graph_objects构建所有图形元素
- 复用原有的几何计算算法(旋转、偏移、对齐)
- shapes实现IP方块网格布局
- annotations实现链路箭头
- customdata和hovertemplate实现hover交互
"""

import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import os
import warnings

from .analyzers import IP_COLOR_MAP, MAX_BANDWIDTH_NORMALIZATION, RN_TYPES, SN_TYPES


class InteractiveFlowGraphRenderer:
    """交互式流量图渲染器 - 使用Plotly绘制可交互的网络拓扑和流量分布图"""

    def __init__(self):
        """初始化交互式流量图渲染器"""
        pass

    def draw_flow_graph(self, network, ip_bandwidth_data: Dict = None, config=None, mode: str = "utilization", node_size: int = 2000, save_path: str = None, show_fig: bool = False):
        """
        绘制单Die网络流量图(交互式版本)

        Args:
            network: Network对象
            ip_bandwidth_data: IP带宽数据字典
            config: 配置对象
            mode: 可视化模式
            node_size: 节点大小
            save_path: 保存路径（如果为None则返回fig对象）
            show_fig: 是否在浏览器中显示图像

        Returns:
            str: 保存的HTML文件路径，如果save_path为None则返回fig对象
        """
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

        # 调用单Die绘制方法
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

        # 收集使用的IP类型(用于legend)
        used_ip_types = set()
        if ip_bandwidth_data is not None and mode in ip_bandwidth_data:
            mode_data = ip_bandwidth_data[mode]
            for ip_type, data_matrix in mode_data.items():
                if data_matrix.sum() > 0.001:  # 只包含有数据的IP类型
                    used_ip_types.add(ip_type)

        # 添加IP类型Legend
        if used_ip_types:
            self._add_ip_legend_plotly(fig, used_ip_types)

        # 添加带宽Colorbar
        if max_ip_bandwidth and min_ip_bandwidth:
            self._add_bandwidth_colorbar_plotly(fig, min_ip_bandwidth, max_ip_bandwidth)

        # 设置布局
        title = f"Network Flow - {mode.capitalize()}"
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=True,
            hovermode="closest",
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, t=50, b=20),
            width=1200,
            height=1000,
        )

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

    def draw_single_die_flow(
        self,
        fig,
        network,
        config,
        ip_bandwidth_data: Dict = None,
        mode: str = "utilization",
        node_size: int = 2000,
        die_id: int = None,
        offset_x: float = 0,
        offset_y: float = 0,
        max_link_flow: float = None,
        min_link_flow: float = None,
        max_ip_bandwidth: float = None,
        min_ip_bandwidth: float = None,
        rotation: int = 0,
        is_d2d_scenario: bool = False,
        show_ip_bandwidth_value: bool = True,
        link_label_fontsize: int = 12,
    ):
        """
        在指定的figure上绘制单个Die的流量图(交互式版本)

        Args:
            fig: plotly Figure对象
            network: Network对象
            config: 配置对象
            ip_bandwidth_data: IP带宽数据字典
            mode: 可视化模式 ("utilization", "bandwidth", "count")
            node_size: 节点大小
            die_id: Die ID（用于D2D场景）
            offset_x: X轴偏移量
            offset_y: Y轴偏移量
            max_link_flow: 最大链路流量（用于归一化）
            min_link_flow: 最小链路流量
            max_ip_bandwidth: 最大IP带宽
            min_ip_bandwidth: 最小IP带宽
            rotation: Die旋转角度（0, 90, 180, 270）
            is_d2d_scenario: 是否为D2D场景（影响数据结构解析）
            show_ip_bandwidth_value: 是否在IP方块中显示带宽数值
            link_label_fontsize: 链路标签字体大小
        """
        # 获取链路统计数据
        links = {}
        if hasattr(network, "get_links_utilization_stats") and callable(network.get_links_utilization_stats):
            try:
                utilization_stats = network.get_links_utilization_stats()
                if mode == "utilization":
                    links = {link: stats["utilization"] for link, stats in utilization_stats.items()}
                elif mode == "ITag_ratio":
                    links = {link: stats["ITag_ratio"] for link, stats in utilization_stats.items()}
                elif mode == "total":
                    links = {}
                    for link, stats in utilization_stats.items():
                        total_flit = stats["total_flit"]
                        total_cycles = stats["total_cycles"]
                        if total_cycles > 0:
                            time_ns = total_cycles / config.NETWORK_FREQUENCY
                            bandwidth = total_flit * 128 / time_ns
                            links[link] = bandwidth
                        else:
                            links[link] = 0.0
                else:
                    links = {link: stats["utilization"] for link, stats in utilization_stats.items()}
            except Exception as e:
                links = {}

        # 获取网络节点
        if hasattr(network, "queues") and network.queues:
            actual_nodes = list(network.queues.keys())
        else:
            # 默认5x4拓扑
            actual_nodes = list(range(config.NUM_ROW * config.NUM_COL))

        # 计算节点位置
        pos = {}
        node_spacing = 3.0
        orig_rows = config.NUM_ROW
        orig_cols = config.NUM_COL

        for node in actual_nodes:
            # 计算原始坐标
            orig_row = node // orig_cols
            orig_col = node % orig_cols

            # 根据旋转角度变换坐标
            if rotation == 0 or abs(rotation) == 360:
                new_row = orig_row
                new_col = orig_col
            elif abs(rotation) == 90 or abs(rotation) == -270:
                new_row = orig_col
                new_col = orig_rows - 1 - orig_row
            elif abs(rotation) == 180:
                new_row = orig_rows - 1 - orig_row
                new_col = orig_cols - 1 - orig_col
            elif abs(rotation) == 270 or abs(rotation) == -90:
                new_row = orig_cols - 1 - orig_col
                new_col = orig_row
            else:
                new_row = orig_row
                new_col = orig_col

            # 计算实际位置
            x = new_col * node_spacing + offset_x
            y = -new_row * node_spacing + offset_y
            pos[node] = (x, y)

        # 计算节点大小
        square_size = np.sqrt(node_size) / 50

        # 绘制节点背景
        self._draw_nodes(fig, pos, square_size, actual_nodes, config=config, ip_bandwidth_data=ip_bandwidth_data, mode=mode, die_id=die_id, is_d2d_scenario=is_d2d_scenario)

        # 绘制链路箭头
        edge_labels = {}
        edge_colors = {}
        for link_key, value in links.items():
            # 处理link格式
            if len(link_key) == 2:
                i, j = link_key
                direction = None
            elif len(link_key) == 3:
                i, j, direction = link_key
            else:
                continue

            if i not in actual_nodes or j not in actual_nodes:
                continue

            # 计算显示值和颜色
            if mode in ["utilization", "T2_ratio", "T1_ratio", "T0_ratio", "ITag_ratio"]:
                display_value = float(value) if value else 0.0
                formatted_label = f"{display_value*100:.1f}%" if display_value > 0 else ""
                color_intensity = display_value
            elif mode == "total":
                display_value = float(value) if value else 0.0
                formatted_label = f"{display_value:.1f}" if display_value > 0 else ""
                color_intensity = min(display_value / 500.0, 1.0)
            else:
                display_value = float(value) if value else 0.0
                formatted_label = f"{display_value:.1f}" if display_value > 0 else ""
                color_intensity = min(display_value / 500.0, 1.0)

            if display_value > 0:
                color = (color_intensity, 0, 0)
            else:
                color = (0.8, 0.8, 0.8)

            # 跳过自环
            if i != j:
                edge_labels[(i, j)] = formatted_label
                edge_colors[(i, j)] = color

        # 绘制链路箭头（传递完整的统计数据用于hover）
        self._draw_link_arrows(fig, pos, edge_labels, edge_colors, links, config, square_size, rotation, link_label_fontsize, utilization_stats, is_d2d_scenario)

        # 批量收集所有节点的IP方块
        if ip_bandwidth_data is not None:
            all_ip_shapes = []
            all_ip_annotations = []

            for node, (x, y) in pos.items():
                ip_shapes, ip_annotations = self._collect_ip_info_shapes(
                    x,
                    y,
                    node,
                    config,
                    mode,
                    square_size,
                    ip_bandwidth_data,
                    max_ip_bandwidth,
                    min_ip_bandwidth,
                    die_id=die_id,
                    is_d2d_scenario=is_d2d_scenario,
                    show_bandwidth_value=show_ip_bandwidth_value,
                )
                all_ip_shapes.extend(ip_shapes)
                all_ip_annotations.extend(ip_annotations)

            # 批量添加IP方块shapes
            if all_ip_shapes:
                current_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
                fig.update_layout(shapes=current_shapes + all_ip_shapes)

            # 批量添加IP标签annotations
            for ann in all_ip_annotations:
                fig.add_annotation(**ann)

        return pos

    def _draw_nodes(self, fig, pos, square_size, actual_nodes, config=None, node_id_offset=None, ip_bandwidth_data=None, mode="total", die_id=None, is_d2d_scenario=False):
        """绘制节点背景方块并添加hover交互（批量优化版本）"""
        # 批量收集节点shapes和scatter数据
        node_shapes = []
        node_x = []
        node_y = []
        node_hover_text = []

        for node in actual_nodes:
            if node not in pos:
                continue

            x, y = pos[node]

            # 收集shape数据
            node_shapes.append(
                dict(
                    type="rect",
                    x0=x - square_size / 2,
                    y0=y - square_size / 2,
                    x1=x + square_size / 2,
                    y1=y + square_size / 2,
                    fillcolor="#E8F5E9",
                    line=dict(color="black", width=1),
                    layer="below",
                )
            )

            # 收集scatter数据
            node_x.append(x)
            node_y.append(y)

            # 计算hover文本 - 显示节点内IP的带宽信息
            hover_text = f"<b>节点 {node}</b><br>"

            # 获取节点内的IP带宽信息
            if ip_bandwidth_data and config:
                physical_row = node // config.NUM_COL
                physical_col = node % config.NUM_COL

                # 根据场景选择数据源
                if is_d2d_scenario and die_id is not None:
                    # D2D场景
                    if die_id in ip_bandwidth_data and mode in ip_bandwidth_data[die_id]:
                        mode_data = ip_bandwidth_data[die_id][mode]
                    else:
                        mode_data = {}
                else:
                    # 单Die场景
                    mode_data = ip_bandwidth_data.get(mode, {})

                # 收集该节点的IP带宽
                node_ips = []
                for ip_type, data_matrix in mode_data.items():
                    if physical_row < data_matrix.shape[0] and physical_col < data_matrix.shape[1]:
                        bandwidth = data_matrix[physical_row, physical_col]
                        if bandwidth > 0.001:
                            node_ips.append((ip_type.upper(), bandwidth))

                # 添加IP带宽信息到hover
                if node_ips:
                    # 先按RN/SN分类，再按带宽排序
                    def ip_sort_key(item):
                        ip_type, bw = item
                        base_type = ip_type.split("_")[0] if "_" in ip_type else ip_type
                        # RN类型排在前面(返回0)，SN类型排在后面(返回1)，其他类型排最后(返回2)
                        if base_type in RN_TYPES:
                            category = 0
                        elif base_type in SN_TYPES:
                            category = 1
                        else:
                            category = 2
                        # 在同一类别内按带宽降序排序
                        return (category, -bw)

                    for ip_type, bw in sorted(node_ips, key=ip_sort_key):
                        hover_text += f"{ip_type}: {bw:.1f} GB/s<br>"
                else:
                    hover_text += "无IP流量<br>"

            node_hover_text.append(hover_text)

        # 批量添加shapes
        if node_shapes:
            current_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
            fig.update_layout(shapes=current_shapes + node_shapes)

        # 添加scatter（一次性）
        if node_x:
            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers",
                    marker=dict(size=square_size * 10, color="rgba(0,0,0,0)", line=dict(width=0)),
                    text=node_hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                    name="节点",
                )
            )

    def _draw_link_arrows(self, fig, pos, edge_labels, edge_colors, links, config, square_size, rotation, fontsize, utilization_stats=None, is_d2d_scenario=False):
        """绘制链路箭头（批量优化版本，增强hover信息）

        Args:
            is_d2d_scenario: 是否为D2D场景，影响标签偏移量大小
        """
        import math

        # 预计算旋转矩阵（避免重复计算）
        angle_rad = math.radians(rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # 构建有向图结构用于判断双向链路
        graph_edges = set(edge_labels.keys())

        # 批量收集箭头annotations和标签数据
        arrow_annotations = []
        label_x_list = []
        label_y_list = []
        label_text_list = []
        label_color_list = []
        label_hover_list = []  # 新增：hover信息

        for (i, j), label in edge_labels.items():
            if i not in pos or j not in pos:
                continue

            color = edge_colors.get((i, j), (0.8, 0.8, 0.8))
            color_str = f"rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"

            x1, y1 = pos[i]
            x2, y2 = pos[j]

            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx * dx + dy * dy)

            if length > 0:
                dx, dy = dx / length, dy / length
                perp_dx, perp_dy = dy * 0.1, -dx * 0.1

                # 检查是否有反向边
                has_reverse = (j, i) in graph_edges

                if has_reverse:
                    # 双向链路：偏移
                    start_x = x1 + dx * square_size / 2 + perp_dx
                    start_y = y1 + dy * square_size / 2 + perp_dy
                    end_x = x2 - dx * square_size / 2 + perp_dx
                    end_y = y2 - dy * square_size / 2 + perp_dy
                else:
                    # 单向链路：不偏移
                    start_x = x1 + dx * square_size / 2
                    start_y = y1 + dy * square_size / 2
                    end_x = x2 - dx * square_size / 2
                    end_y = y2 - dy * square_size / 2

                # 收集箭头annotation（缩小箭头）
                arrow_annotations.append(
                    dict(
                        x=end_x,
                        y=end_y,
                        ax=start_x,
                        ay=start_y,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=0.8,  # 从1减小到0.8
                        arrowwidth=1.5,  # 从2减小到1.5
                        arrowcolor=color_str,
                        standoff=0,
                    )
                )

                # 收集标签数据（使用scatter代替annotation）
                if label:
                    mid_x = (start_x + end_x) / 2
                    mid_y = (start_y + end_y) / 2

                    # 双向链路需要偏移标签
                    if has_reverse:
                        # 计算Die内部方向
                        orig_i_row = i // config.NUM_COL
                        orig_j_row = j // config.NUM_COL
                        is_horizontal_in_die = orig_i_row == orig_j_row

                        # 根据场景和Die内部方向计算偏移量
                        # D2D场景使用较大偏移(0.70/0.35)，单Die场景使用较小偏移(0.25/0.15)
                        is_90_or_270 = abs(rotation) in [90, 270]

                        if is_d2d_scenario:
                            # D2D场景：使用原来的大偏移量
                            if is_horizontal_in_die:
                                offset_magnitude = 0.70 if is_90_or_270 else 0.35
                                if i < j:
                                    offset_x_die, offset_y_die = 0, offset_magnitude
                                else:
                                    offset_x_die, offset_y_die = 0, -offset_magnitude
                            else:
                                offset_magnitude = 0.35 if is_90_or_270 else 0.70
                                if i < j:
                                    offset_x_die, offset_y_die = -offset_magnitude, 0
                                else:
                                    offset_x_die, offset_y_die = offset_magnitude, 0
                        else:
                            # 单Die场景：使用较小偏移量
                            if is_horizontal_in_die:
                                offset_magnitude = 0.30 if is_90_or_270 else 0.20
                                if i < j:
                                    offset_x_die, offset_y_die = 0, offset_magnitude
                                else:
                                    offset_x_die, offset_y_die = 0, -offset_magnitude
                            else:
                                offset_magnitude = 0.20 if is_90_or_270 else 0.30
                                if i < j:
                                    offset_x_die, offset_y_die = -offset_magnitude, 0
                                else:
                                    offset_x_die, offset_y_die = offset_magnitude, 0

                        # 应用旋转变换（使用预计算的旋转矩阵）
                        offset_x_screen = offset_x_die * cos_a - offset_y_die * sin_a
                        offset_y_screen = offset_x_die * sin_a + offset_y_die * cos_a

                        label_x = mid_x + offset_x_screen
                        label_y = mid_y - offset_y_screen
                    else:
                        label_x = mid_x
                        label_y = mid_y

                    # 构建详细的hover信息
                    if utilization_stats and (i, j) in utilization_stats:
                        stats = utilization_stats[(i, j)]
                        utilization = stats.get("utilization", 0) * 100
                        empty_ratio = stats.get("empty_ratio", 0) * 100
                        effective_ratio = stats.get("effective_ratio", 0) * 100
                        total_flit = stats.get("total_flit", 0)
                        total_cycles = stats.get("total_cycles", 0)

                        # 计算带宽
                        if total_cycles > 0:
                            time_ns = total_cycles / config.NETWORK_FREQUENCY
                            bandwidth = total_flit * 128 / time_ns
                        else:
                            bandwidth = 0

                        # 获取eject_attempts分布
                        merged_ratios = stats.get("eject_attempts_merged_ratios", {"0": 0, "1": 0, "2": 0, ">2": 0})
                        attempts_0 = merged_ratios.get("0", 0) * 100
                        attempts_1 = merged_ratios.get("1", 0) * 100
                        attempts_2 = merged_ratios.get("2", 0) * 100
                        attempts_gt2 = merged_ratios.get(">2", 0) * 100

                        hover_text = (
                            f"<b>链路: {i} → {j}</b><br>"
                            f"带宽: {bandwidth:.2f} GB/s<br>"
                            f"flit数量: {total_flit}<br>"
                            f"有效利用率: {effective_ratio:.1f}%<br>"
                            f"总利用率: {utilization:.1f}%<br>"
                            # f"下环尝试次数0: {attempts_0:.1f}%<br>"
                            # f"下环尝试次数1: {attempts_1:.1f}%<br>"
                            # f"下环尝试次数2: {attempts_2:.1f}%<br>"
                            f"下环尝试次数大于2占比: {attempts_gt2:.1f}%<br>"
                            f"空闲率: {empty_ratio:.1f}%"
                        )
                    else:
                        hover_text = f"<b>链路: {i} → {j}</b><br>值: {label}"

                    label_x_list.append(label_x)
                    label_y_list.append(label_y)
                    label_text_list.append(label)
                    label_color_list.append(color_str)
                    label_hover_list.append(hover_text)

        # 批量添加箭头annotations
        for ann in arrow_annotations:
            fig.add_annotation(**ann)

        # 按颜色分组绘制标签（避免重复绘制）
        if label_x_list:
            # 按颜色分组（包含hover信息）
            color_groups = {}
            for x, y, text, color, hover in zip(label_x_list, label_y_list, label_text_list, label_color_list, label_hover_list):
                if color not in color_groups:
                    color_groups[color] = {"x": [], "y": [], "text": [], "hover": []}
                color_groups[color]["x"].append(x)
                color_groups[color]["y"].append(y)
                color_groups[color]["text"].append(text)
                color_groups[color]["hover"].append(hover)

            # 为每种颜色创建一个trace
            for color, data in color_groups.items():
                fig.add_trace(
                    go.Scatter(
                        x=data["x"],
                        y=data["y"],
                        mode="text",
                        text=data["text"],
                        textfont=dict(size=fontsize + 2, color=color),  # 增大2号字体
                        textposition="middle center",
                        showlegend=False,
                        hovertext=data["hover"],
                        hoverinfo="text",
                    )
                )

    def _collect_ip_info_shapes(
        self, x, y, node, config, mode, square_size, ip_bandwidth_data, max_ip_bandwidth=None, min_ip_bandwidth=None, die_id=None, is_d2d_scenario=False, show_bandwidth_value=True
    ):
        """
        收集节点内IP信息方块的shapes和annotations（批量优化版本）

        Args:
            x, y: 节点中心位置
            node: 节点ID
            config: 配置对象
            mode: 显示模式
            square_size: 节点方块大小
            ip_bandwidth_data: IP带宽数据字典
            max_ip_bandwidth: 全局最大IP带宽
            min_ip_bandwidth: 全局最小IP带宽
            die_id: Die ID（D2D场景必填）
            is_d2d_scenario: 是否为D2D场景
            show_bandwidth_value: 是否显示带宽数值

        Returns:
            tuple: (shapes列表, annotations列表)
        """
        MAX_ROWS = 4  # 最多显示4行IP

        # 计算物理位置
        physical_col = node % config.NUM_COL
        physical_row = node // config.NUM_COL

        # 收集该节点所有有流量的IP类型
        active_ips = []

        # 根据场景选择数据源
        if ip_bandwidth_data is not None:
            if is_d2d_scenario and die_id is not None:
                # D2D场景：使用 ip_bandwidth_data[die_id][mode]
                if die_id in ip_bandwidth_data and mode in ip_bandwidth_data[die_id]:
                    mode_data = ip_bandwidth_data[die_id][mode]
                else:
                    mode_data = {}
            else:
                # 单Die场景：使用 ip_bandwidth_data[mode]
                mode_data = ip_bandwidth_data.get(mode, {})

            for ip_type, data_matrix in mode_data.items():
                # D2D场景下过滤掉 d2d_rn 和 d2d_sn 类型
                if is_d2d_scenario and ip_type.lower() in ["d2d_rn", "d2d_sn"]:
                    continue

                matrix_row = physical_row
                if matrix_row < data_matrix.shape[0] and physical_col < data_matrix.shape[1]:
                    bandwidth = data_matrix[matrix_row, physical_col]
                    if bandwidth > 0.001:
                        active_ips.append((ip_type.upper(), bandwidth))

        # 如果没有活跃IP，返回空列表
        if not active_ips:
            return [], []

        # 按IP基本类型分组
        ip_type_count = defaultdict(list)
        for ip_type, bw in active_ips:
            base_type = ip_type.split("_")[0] if "_" in ip_type else ip_type
            ip_type_count[base_type].append(bw)

        # 按RN/SN分类排序
        rn_ips = [(k, v) for k, v in ip_type_count.items() if k.upper() in RN_TYPES]
        sn_ips = [(k, v) for k, v in ip_type_count.items() if k.upper() in SN_TYPES]
        other_ips = [(k, v) for k, v in ip_type_count.items() if k.upper() not in RN_TYPES + SN_TYPES]

        rn_ips.sort(key=lambda x: sum(x[1]), reverse=True)
        sn_ips.sort(key=lambda x: sum(x[1]), reverse=True)
        other_ips.sort(key=lambda x: sum(x[1]), reverse=True)

        # 构建最终显示列表
        display_rows = []
        display_rows.extend(rn_ips)
        display_rows.extend(sn_ips)

        if len(display_rows) + len(other_ips) > MAX_ROWS:
            display_rows = display_rows[:MAX_ROWS]
            for i, (ip_type, instances) in enumerate(other_ips):
                target_row = i % len(display_rows)
                display_rows[target_row] = (display_rows[target_row][0], display_rows[target_row][1] + instances)
        else:
            display_rows.extend(other_ips)
            if len(display_rows) > MAX_ROWS:
                display_rows = display_rows[:MAX_ROWS]

        ip_type_count = dict(display_rows)

        # 计算布局参数
        num_ip_types = len(ip_type_count)
        max_instances = max(len(instances) for instances in ip_type_count.values())

        available_width = square_size * 0.90
        available_height = square_size * 0.90
        grid_spacing = square_size * 0.10
        row_spacing = square_size * 0.1

        max_square_width = (available_width - (max_instances - 1) * grid_spacing) / max_instances
        max_square_height = (available_height - (num_ip_types - 1) * row_spacing) / num_ip_types
        grid_square_size = min(max_square_width, max_square_height, square_size * 0.5)

        total_content_height = num_ip_types * grid_square_size + (num_ip_types - 1) * row_spacing

        # 收集IP小方块的shapes和annotations
        ip_shapes = []
        ip_annotations = []
        row_idx = 0

        for ip_type, instances in ip_type_count.items():
            num_instances = len(instances)
            base_type = ip_type.upper()
            ip_color = IP_COLOR_MAP.get(base_type, IP_COLOR_MAP["OTHER"])

            row_width = num_instances * grid_square_size + (num_instances - 1) * grid_spacing
            row_start_x = x - row_width / 2
            row_y = y + total_content_height / 2 - row_idx * (grid_square_size + row_spacing) - grid_square_size / 2

            for col_idx, bandwidth in enumerate(instances):
                block_x = row_start_x + col_idx * (grid_square_size + grid_spacing) + grid_square_size / 2
                block_y = row_y

                # 计算透明度
                alpha = self._calculate_bandwidth_alpha(bandwidth, min_ip_bandwidth if min_ip_bandwidth is not None else 0, max_ip_bandwidth if max_ip_bandwidth is not None else 1)

                # 转换颜色为rgba格式
                rgba_color = self._hex_to_rgba(ip_color, alpha)

                # 收集小方块shape
                ip_shapes.append(
                    dict(
                        type="rect",
                        x0=block_x - grid_square_size / 2,
                        y0=block_y - grid_square_size / 2,
                        x1=block_x + grid_square_size / 2,
                        y1=block_y + grid_square_size / 2,
                        fillcolor=rgba_color,
                        line=dict(color="black", width=0.8),
                        layer="above",
                    )
                )

                # 收集带宽数值annotation（可选）
                if show_bandwidth_value and grid_square_size >= square_size * 0.4:
                    bw_text = f"{bandwidth:.0f}" if bandwidth >= 10 else f"{bandwidth:.1f}"
                    ip_annotations.append(
                        dict(
                            x=block_x,
                            y=block_y,
                            text=bw_text,
                            showarrow=False,
                            font=dict(size=10, color="black"),
                            xref="x",
                            yref="y",
                        )
                    )

            row_idx += 1

        return ip_shapes, ip_annotations

    def _calculate_bandwidth_alpha(self, bandwidth, min_bw, max_bw):
        """计算带宽透明度（复用原有逻辑）"""
        if max_bw is None or min_bw is None:
            return 0.8

        if max_bw <= min_bw or max_bw <= 0:
            return 0.8

        normalized = max(0, min(1, (bandwidth - min_bw) / (max_bw - min_bw)))
        return 0.3 + 0.7 * normalized

    def _hex_to_rgba(self, hex_color, alpha):
        """将十六进制颜色转换为rgba字符串"""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"

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

        # 为每个Die绘制流量图
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
            )
            die_node_positions[die_id] = node_positions

        # 添加Die标签
        for die_id in die_node_positions.keys():
            node_positions = die_node_positions[die_id]
            if node_positions:
                xs = [p[0] for p in node_positions.values()]
                ys = [p[1] for p in node_positions.values()]
                die_center_x = (min(xs) + max(xs)) / 2
                die_center_y = (min(ys) + max(ys)) / 2

                # 智能标签位置（复用原有逻辑）
                if die_id in die_layout:
                    grid_x, grid_y = die_layout[die_id]
                    other_dies = [did for did in die_layout.keys() if did != die_id]

                    if other_dies:
                        other_die_id = other_dies[0]
                        other_grid_x, other_grid_y = die_layout[other_die_id]

                        is_vertical_connection = grid_y != other_grid_y
                        is_horizontal_connection = grid_x != other_grid_x

                        if is_vertical_connection:
                            if grid_x == 0:
                                label_x = min(xs) - 3
                                label_y = die_center_y
                            else:
                                label_x = max(xs) + 3
                                label_y = die_center_y
                        elif is_horizontal_connection:
                            label_x = die_center_x
                            label_y = min(ys) - 2
                        else:
                            label_x = die_center_x
                            label_y = max(ys) + 2.5
                    else:
                        label_x = die_center_x
                        label_y = max(ys) + 2.5
                else:
                    label_x = die_center_x
                    label_y = max(ys) + 2.5

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

        # 绘制跨Die连接
        if dies and len(dies) > 1:
            try:
                d2d_bandwidth = self._calculate_d2d_sys_bandwidth(dies, config)
                self._draw_cross_die_connections(fig, d2d_bandwidth, die_node_positions, config, dies, die_offsets)
            except Exception as e:
                import traceback

                traceback.print_exc()

        # 添加IP类型Legend
        if used_ip_types:
            self._add_ip_legend_plotly(fig, used_ip_types)

        # 添加带宽Colorbar
        if all_ip_bandwidths and max_ip_bandwidth and min_ip_bandwidth:
            self._add_bandwidth_colorbar_plotly(fig, min_ip_bandwidth, max_ip_bandwidth)

        # 设置布局
        title = f"D2D Flow Graph - {mode.capitalize()}"
        canvas_width = int(figsize[0] * 100)
        canvas_height = int(figsize[1] * 100)

        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            showlegend=True,
            hovermode="closest",
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, t=50, b=20),
            width=canvas_width,
            height=canvas_height,
        )

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

    def _draw_cross_die_connections(self, fig, d2d_bandwidth, die_node_positions, config, dies=None, die_offsets=None):
        """
        绘制跨Die数据带宽连接（只显示R和W通道的数据流）- Plotly版本

        Args:
            fig: plotly Figure对象
            d2d_bandwidth: D2D带宽数据
            die_node_positions: 各Die的节点位置
            config: 配置对象
            dies: Die模型字典
            die_offsets: Die偏移量
        """
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

                    # 检查读写数据流量
                    w_bw = d2d_bandwidth.get(from_die, {}).get(key, {}).get("W", 0.0)
                    r_bw = d2d_bandwidth.get(from_die, {}).get(key, {}).get("R", 0.0)

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

                    # 合并读写通道带宽
                    total_bw = w_bw + r_bw

                    # 绘制D2D箭头
                    self._draw_single_d2d_arrow_plotly(fig, from_x, from_y, to_x, to_y, total_bw, from_die, from_node, to_die, to_node, connection_type, w_bw, r_bw)
                    arrow_index += 1

        except Exception as e:
            import traceback

            traceback.print_exc()

    def _draw_single_d2d_arrow_plotly(self, fig, start_x, start_y, end_x, end_y, total_bandwidth, from_die, from_node, to_die, to_node, connection_type=None, w_bw=0.0, r_bw=0.0):
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
        """
        dx = end_x - start_x
        dy = end_y - start_y
        length = np.sqrt(dx * dx + dy * dy)

        if length == 0:
            return

        # 归一化方向向量
        ux, uy = dx / length, dy / length
        perpx, perpy = -uy * 0.2, ux * 0.2

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

        # 绘制箭头
        hover_text = (
            f"D2D连接: Die{from_die}(节点{from_node}) → Die{to_die}(节点{to_node})<br>" f"总带宽: {total_bandwidth:.2f} GB/s<br>" f"写通道(W): {w_bw:.2f} GB/s<br>" f"读通道(R): {r_bw:.2f} GB/s"
        )

        fig.add_annotation(
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

        # 添加标签（只在有流量时）
        if label_text:
            if connection_type == "diagonal":
                label_x = arrow_start_x + (arrow_end_x - arrow_start_x) * 0.85
                label_y_base = arrow_start_y + (arrow_end_y - arrow_start_y) * 0.85
                if (dx > 0 and dy > 0) or (dx > 0 and dy < 0):
                    label_y = label_y_base + 0.6
                else:
                    label_y = label_y_base - 0.6
            else:
                mid_x = (arrow_start_x + arrow_end_x) / 2
                mid_y = (arrow_start_y + arrow_end_y) / 2
                is_horizontal = abs(dx) > abs(dy)

                if is_horizontal:
                    label_x = mid_x
                    label_y = mid_y + (0.5 if dx > 0 else -0.5)
                else:
                    label_x = mid_x + (dy * 0.1 if dx > 0 else -dy * 0.1)
                    label_y = mid_y - 0.15

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

            fig.add_annotation(
                x=label_x,
                y=label_y,
                text=label_text,
                showarrow=False,
                font=dict(size=12, color=color_str),
                xref="x",
                yref="y",
                textangle=angle_deg,
            )

            # 在标签位置添加透明的scatter点用于hover
            fig.add_trace(
                go.Scatter(
                    x=[label_x],
                    y=[label_y],
                    mode="markers",
                    marker=dict(size=20, color="rgba(0,0,0,0)"),  # 透明标记
                    hovertext=hover_text,
                    hoverinfo="text",
                    showlegend=False,
                    name="",
                )
            )

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
                        x=1.02,  # 位置：右侧
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
