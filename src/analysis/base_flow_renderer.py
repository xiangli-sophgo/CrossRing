"""
流量图渲染器基类 - 提供共享的绘制方法

包含所有单Die和D2D渲染器共享的基础绘制功能：
- 节点背景绘制
- 链路箭头绘制
- IP方块布局
- 图例和colorbar
"""

import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from functools import lru_cache

from .analyzers import IP_COLOR_MAP, MAX_BANDWIDTH_NORMALIZATION, RN_TYPES, SN_TYPES
from .flow_render_helpers import ChannelSwitchManager, LinkDataProcessor, IPLayoutCalculator


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


class BaseFlowRenderer:
    """流量图渲染器基类 - 提供共享的绘制方法"""

    def __init__(self):
        """初始化基类"""
        pass

    def _apply_rotation(self, orig_row, orig_col, rows, cols, rotation):
        """
        根据旋转角度变换行列坐标

        Args:
            orig_row: 原始行索引
            orig_col: 原始列索引
            rows: 总行数
            cols: 总列数
            rotation: 旋转角度 (0, 90, 180, 270)

        Returns:
            (new_row, new_col)
        """
        if rotation == 0 or abs(rotation) == 360:
            return orig_row, orig_col
        elif abs(rotation) == 90 or abs(rotation) == -270:
            # 90度顺时针：行列互换，行索引取反
            return orig_col, rows - 1 - orig_row
        elif abs(rotation) == 180:
            # 180度：行列都取反
            return rows - 1 - orig_row, cols - 1 - orig_col
        elif abs(rotation) == 270 or abs(rotation) == -90:
            # 270度顺时针(或-90度)：行列互换，列索引取反
            return cols - 1 - orig_col, orig_row
        else:
            return orig_row, orig_col

    def __init__(self):
        """初始化单Die流量图渲染器"""
        pass

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

    def _draw_link_arrows(
        self, fig, pos, edge_labels, edge_colors, links, config, square_size, rotation, fontsize, utilization_stats=None, is_d2d_scenario=False, show_labels=True, static_bandwidth=None
    ):
        """绘制链路箭头（批量优化版本，增强hover信息）

        Returns:
            list: annotations列表

        Args:
            is_d2d_scenario: 是否为D2D场景，影响标签偏移量大小
            show_labels: 是否显示链路文本标签
            static_bandwidth: 静态带宽数据字典 {((src_col, src_row), (dst_col, dst_row)): bandwidth}
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

            # 跳过自环（由_draw_self_loop_labels单独处理）
            if i == j:
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

                # 计算节点框的半宽度（从节点中心到边缘）
                # 节点框大小为 square_size，但实际需要考虑IP方块等元素，尾部需要更大偏移
                node_half_size_head = square_size / 2  # 头部：使用节点框边缘
                node_half_size_tail = square_size * (0.35 if abs(i - j) != 1 else 0.45)

                if has_reverse:
                    # 双向链路：偏移
                    start_x = x1 + dx * node_half_size_tail + perp_dx
                    start_y = y1 + dy * node_half_size_tail + perp_dy
                    end_x = x2 - dx * node_half_size_head + perp_dx
                    end_y = y2 - dy * node_half_size_head + perp_dy
                else:
                    # 单向链路：不偏移
                    start_x = x1 + dx * node_half_size_tail
                    start_y = y1 + dy * node_half_size_tail
                    end_x = x2 - dx * node_half_size_head
                    end_y = y2 - dy * node_half_size_head

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
                        arrowsize=0.8,
                        arrowwidth=1.5,
                        arrowcolor=color_str,
                        standoff=0,
                    )
                )

                # 收集标签数据（使用scatter代替annotation）
                if label:
                    start_x += dx * square_size * (0.15 if abs(i - j) != 1 else 0.05)
                    start_y += dy * square_size * (0.15 if abs(i - j) != 1 else 0.05)
                    mid_x = (start_x + end_x) / 2
                    mid_y = (start_y + end_y) / 2

                    # 双向链路需要偏移标签
                    if has_reverse:
                        # 计算Die内部方向
                        orig_i_row = i // config.NUM_COL
                        orig_j_row = j // config.NUM_COL
                        is_horizontal_in_die = orig_i_row == orig_j_row

                        # 根据label长度计算额外偏移
                        label_str = str(label)
                        text_length = len(label_str)
                        if text_length <= 3:
                            length_factor = 1.0
                        elif text_length <= 4:
                            length_factor = 1.15
                        elif text_length <= 5:
                            length_factor = 1.30
                        else:
                            length_factor = 1.50

                        # 根据场景和Die内部方向计算偏移量
                        # D2D场景使用较大偏移(0.70/0.35)，单Die场景使用较小偏移(0.25/0.15)
                        is_90_or_270 = abs(rotation) in [90, 270]

                        if is_d2d_scenario:
                            # D2D场景：使用原来的大偏移量
                            if is_horizontal_in_die:
                                offset_magnitude = (0.45 if is_90_or_270 else 0.25) * length_factor
                                if i < j:
                                    offset_x_die, offset_y_die = 0, offset_magnitude
                                else:
                                    offset_x_die, offset_y_die = 0, -offset_magnitude
                            else:
                                offset_magnitude = (0.25 if is_90_or_270 else 0.45) * length_factor
                                if i < j:
                                    offset_x_die, offset_y_die = -offset_magnitude, 0
                                else:
                                    offset_x_die, offset_y_die = offset_magnitude, 0
                        else:
                            # 单Die场景：使用较小偏移量
                            if is_horizontal_in_die:
                                offset_magnitude = (0.30 if is_90_or_270 else 0.20) * length_factor
                                if i < j:
                                    offset_x_die, offset_y_die = 0, offset_magnitude
                                else:
                                    offset_x_die, offset_y_die = 0, -offset_magnitude
                            else:
                                offset_magnitude = (0.20 if is_90_or_270 else 0.30) * length_factor
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

                        # 计算实际带宽
                        if total_cycles > 0:
                            time_ns = total_cycles / config.NETWORK_FREQUENCY
                            bandwidth = total_flit * 128 / time_ns
                        else:
                            bandwidth = 0

                        # 获取静态带宽（如果有）
                        static_bw = None
                        if static_bandwidth:
                            # 将节点ID转换为坐标
                            i_row, i_col = i // config.NUM_COL, i % config.NUM_COL
                            j_row, j_col = j // config.NUM_COL, j % config.NUM_COL
                            link_key = ((i_col, i_row), (j_col, j_row))
                            static_bw = static_bandwidth.get(link_key, None)

                        # 获取eject_attempts分布
                        merged_ratios = stats.get("eject_attempts_merged_ratios", {"0": 0, "1": 0, "2": 0, ">2": 0})
                        attempts_0 = merged_ratios.get("0", 0) * 100
                        attempts_1 = merged_ratios.get("1", 0) * 100
                        attempts_2 = merged_ratios.get("2", 0) * 100
                        attempts_gt2 = merged_ratios.get(">2", 0) * 100

                        hover_text = f"<b>链路: {i} → {j}</b><br>"
                        if static_bw is not None:
                            hover_text += f"静态带宽: {static_bw:.2f} GB/s<br>"
                        hover_text += (
                            f"实际带宽: {bandwidth:.2f} GB/s<br>"
                            f"flit数量: {total_flit}<br>"
                            f"有效利用率: {effective_ratio:.1f}%<br>"
                            f"总利用率: {utilization:.1f}%<br>"
                            # f"下环尝试次数0: {attempts_0:.1f}%<br>"
                            # f"下环尝试次数1: {attempts_1:.1f}%<br>"
                            # f"下环尝试次数2: {attempts_2:.1f}%<br>"
                            f"下环尝试次数大于2占比: {attempts_gt2:.1f}%<br>"
                            f"空闲率: {empty_ratio:.1f}%"
                        )
                        # 反方向上环统计（仅当功能开启且有数据时显示）
                        if config and getattr(config, "REVERSE_DIRECTION_ENABLED", False):
                            reverse_inject_total = stats.get("reverse_inject_total", 0)
                            reverse_inject_ratio = stats.get("reverse_inject_ratio", 0) * 100
                            if reverse_inject_total > 0:
                                hover_text += f"<br>反方向上环: {reverse_inject_total} ({reverse_inject_ratio:.1f}%)"
                    else:
                        hover_text = f"<b>链路: {i} → {j}</b><br>值: {label}"

                    label_x_list.append(label_x)
                    label_y_list.append(label_y)
                    label_text_list.append(label)
                    label_color_list.append(color_str)
                    label_hover_list.append(hover_text)

        # 按颜色分组绘制标签（避免重复绘制）
        if show_labels and label_x_list:
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

        # 返回箭头annotations
        return arrow_annotations

    def _draw_channel_links_only(self, fig, network, config, pos, mode, node_size, draw_self_loops=False, rotation=0, is_d2d_scenario=False, fontsize=12, static_bandwidth=None):
        """
        只绘制指定network的链路traces（不绘制nodes和annotations）

        Args:
            fig: plotly Figure对象
            network: Network对象
            config: 配置对象
            pos: 节点位置字典
            mode: 可视化模式
            node_size: 节点大小
            draw_self_loops: 是否绘制自环标签（避免重复）
            static_bandwidth: 静态带宽数据字典

        Returns:
            list: 该通道的annotations列表
        """
        channel_annotations = []  # 收集该通道的所有annotations

        # 使用LinkDataProcessor提取链路统计数据
        links, utilization_stats = LinkDataProcessor.extract_link_stats(network, mode, config)

        # 计算节点大小
        square_size = np.sqrt(node_size) / 50

        # 获取实际节点列表
        if hasattr(network, "queues") and network.queues:
            actual_nodes = list(network.queues.keys())
        else:
            actual_nodes = list(range(config.NUM_ROW * config.NUM_COL))

        # 使用LinkDataProcessor处理链路数据
        edge_labels, edge_colors, self_loop_labels = LinkDataProcessor.process_links_for_drawing(links, actual_nodes, mode)

        # 绘制链路箭头（带文本标签）- 收集返回的annotations
        arrow_anns = self._draw_link_arrows(
            fig,
            pos,
            edge_labels,
            edge_colors,
            links,
            config,
            square_size,
            rotation=rotation,
            fontsize=fontsize,
            utilization_stats=utilization_stats,
            is_d2d_scenario=is_d2d_scenario,
            show_labels=True,
            static_bandwidth=static_bandwidth,
        )
        channel_annotations.extend(arrow_anns)

        # 绘制自环标签（只在需要时绘制，避免重复）- 收集返回的annotations
        if draw_self_loops and self_loop_labels:
            selfloop_anns = self._draw_self_loop_labels(fig, pos, self_loop_labels, config, square_size, rotation=rotation, fontsize=fontsize, utilization_stats=utilization_stats)
            channel_annotations.extend(selfloop_anns)

        return channel_annotations

    def _draw_self_loop_labels(self, fig, pos, self_loop_labels, config, square_size, rotation, fontsize, utilization_stats):
        """
        绘制自环链路标签（Plotly版本，支持hover）

        Args:
            fig: Plotly图形对象
            pos: 节点位置字典
            self_loop_labels: 自环标签字典 {(node, direction): (label, color)}
            config: 配置对象
            square_size: 节点方块大小
            rotation: Die旋转角度
            fontsize: 字体大小
            utilization_stats: 链路统计数据字典

        Returns:
            list: annotations列表
        """
        selfloop_annotations = []  # 收集该方法的所有annotations
        # 按颜色和旋转角度分组收集标签数据
        label_groups = {}  # {(color_str, text_rotation): {"x": [], "y": [], "text": [], "hover": []}}

        for (node, direction), (label, color) in self_loop_labels.items():
            if node not in pos or not label:
                continue

            x, y = pos[node]
            original_row = node // config.NUM_COL
            original_col = node % config.NUM_COL

            # Step 1: 判断旋转后的屏幕方向
            if rotation in [90, 270, -270]:
                # 90/270度：横纵互换
                screen_direction = "v" if direction == "h" else "h"
            else:
                # 0/180度：方向不变
                screen_direction = direction

            # Step 2: 计算旋转后的行列（用于判断边界）
            orig_rows = config.NUM_ROW
            orig_cols = config.NUM_COL
            if abs(rotation) == 90 or abs(rotation) == -270:
                # 顺时针90度
                rotated_row = original_col
                rotated_col = orig_rows - 1 - original_row
                rotated_rows = orig_cols
                rotated_cols = orig_rows
            elif abs(rotation) == 180:
                # 180度
                rotated_row = orig_rows - 1 - original_row
                rotated_col = orig_cols - 1 - original_col
                rotated_rows = orig_rows
                rotated_cols = orig_cols
            elif abs(rotation) == 270 or abs(rotation) == -90:
                # 顺时针270度
                rotated_row = orig_cols - 1 - original_col
                rotated_col = original_row
                rotated_rows = orig_cols
                rotated_cols = orig_rows
            else:
                # 0度
                rotated_row = original_row
                rotated_col = original_col
                rotated_rows = orig_rows
                rotated_cols = orig_cols

            # Step 3: 根据屏幕方向和边界位置计算偏移量
            offset_dist = 0.3
            if screen_direction == "h":
                # 屏幕水平自环：放左右两边（水平显示，避免annotation不受visibility控制）
                if rotated_col == 0:
                    # 屏幕左边
                    offset_x_screen = -square_size / 2 - offset_dist
                    offset_y_screen = 0
                    text_rotation = 90
                else:
                    # 屏幕右边
                    offset_x_screen = square_size / 2 + offset_dist
                    offset_y_screen = 0
                    text_rotation = -90
            else:
                # 屏幕垂直自环：放上下两边
                if rotated_row == 0:
                    # 屏幕上边
                    offset_x_screen = 0
                    offset_y_screen = square_size / 2 + offset_dist
                    text_rotation = 0  # 水平
                else:
                    # 屏幕下边
                    offset_x_screen = 0
                    offset_y_screen = -square_size / 2 - offset_dist
                    text_rotation = 0  # 水平

            label_x = x + offset_x_screen
            label_y = y + offset_y_screen

            # 转换颜色格式
            color_str = f"rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"

            # 构建hover信息（自环链路 i == j）
            direction_label = "横向" if direction == "h" else "纵向"
            if utilization_stats and (node, node, direction) in utilization_stats:
                stats = utilization_stats[(node, node, direction)]
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

                # 获取eject_attempts分布（完整获取所有字段，与普通链路保持一致）
                merged_ratios = stats.get("eject_attempts_merged_ratios", {"0": 0, "1": 0, "2": 0, ">2": 0})
                attempts_0 = merged_ratios.get("0", 0) * 100
                attempts_1 = merged_ratios.get("1", 0) * 100
                attempts_2 = merged_ratios.get("2", 0) * 100
                attempts_gt2 = merged_ratios.get(">2", 0) * 100

                hover_text = (
                    f"<b>链路: {node} → {node} ({direction_label}自环)</b><br>"
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
                hover_text = f"<b>链路: {node} → {node} ({direction_label}自环)</b><br>值: {label}"

            # 按颜色和旋转角度分组
            group_key = (color_str, text_rotation)
            if group_key not in label_groups:
                label_groups[group_key] = {"x": [], "y": [], "text": [], "hover": []}

            label_groups[group_key]["x"].append(label_x)
            label_groups[group_key]["y"].append(label_y)
            label_groups[group_key]["text"].append(label)
            label_groups[group_key]["hover"].append(hover_text)

        # 为每个分组创建scatter trace或annotation（根据是否需要旋转）
        for (color_str, text_rotation), data in label_groups.items():
            if text_rotation == 0:
                # 水平文本：使用scatter trace支持hover
                fig.add_trace(
                    go.Scatter(
                        x=data["x"],
                        y=data["y"],
                        mode="text",
                        text=data["text"],
                        textfont=dict(size=fontsize + 2, color=color_str, family="Arial"),
                        textposition="middle center",
                        showlegend=False,
                        hovertext=data["hover"],
                        hoverinfo="text",
                        hoverlabel=dict(bgcolor="#333", font=dict(color="white")),  # 统一深色背景
                    )
                )
            else:
                # 旋转文本：使用annotation（但annotation不支持hover）
                # 为了支持hover，添加一个不可见的scatter点
                fig.add_trace(
                    go.Scatter(
                        x=data["x"],
                        y=data["y"],
                        mode="markers",
                        marker=dict(size=15, opacity=0),  # 不可见的标记点
                        showlegend=False,
                        hovertext=data["hover"],
                        hoverinfo="text",
                        hoverlabel=dict(bgcolor="#333", font=dict(color="white")),  # 统一深色背景
                    )
                )
                # 批量添加旋转文本annotation
                annotations = []
                for x, y, text in zip(data["x"], data["y"], data["text"]):
                    annotations.append(
                        dict(
                            x=x,
                            y=y,
                            text=text,
                            showarrow=False,
                            font=dict(size=fontsize + 2, color=color_str, family="Arial"),
                            xanchor="center",
                            yanchor="middle",
                            textangle=text_rotation,
                        )
                    )
                # 收集旋转文本annotations
                selfloop_annotations.extend(annotations)

        return selfloop_annotations

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
                alpha = IPLayoutCalculator.calculate_bandwidth_alpha(bandwidth, min_ip_bandwidth if min_ip_bandwidth is not None else 0, max_ip_bandwidth if max_ip_bandwidth is not None else 1)

                # 转换颜色为rgba格式
                rgba_color = IPLayoutCalculator.hex_to_rgba(ip_color, alpha)

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

    def _calculate_ip_block_positions(self, node_x, node_y, active_ips, square_size):
        """
        计算节点内每个IP方块的中心位置（复用_draw_ip_blocks_in_node的布局逻辑）

        Args:
            node_x: 节点中心X坐标
            node_y: 节点中心Y坐标
            active_ips: {ip_type: bandwidth} 字典
            square_size: 节点方块大小

        Returns:
            {ip_type: (ip_x, ip_y, ip_size)} 字典
        """
        # 与_draw_ip_blocks_in_node中的布局逻辑保持一致
        MAX_ROWS = 3
        ip_type_count = {}

        # 按IP类型分组统计实例
        for ip_type, bandwidth in active_ips.items():
            base_type = ip_type.rsplit("_", 1)[0]  # 去掉最后的数字
            if base_type not in ip_type_count:
                ip_type_count[base_type] = []
            ip_type_count[base_type].append(bandwidth)

        # 优先显示规则（与_draw_ip_blocks_in_node保持一致）
        priority_types = ["gdma", "ddr", "sdma", "cdma"]
        display_rows = []

        for ptype in priority_types:
            if ptype in ip_type_count:
                display_rows.append((ptype, ip_type_count[ptype]))

        other_ips = [(k, v) for k, v in ip_type_count.items() if k not in priority_types]
        if other_ips:
            if len(display_rows) + len(other_ips) > MAX_ROWS:
                remaining_slots = MAX_ROWS - len(display_rows)
                for i, (otype, instances) in enumerate(other_ips):
                    if i < remaining_slots:
                        display_rows.append((otype, instances))
                    else:
                        target_row = i % len(display_rows)
                        display_rows[target_row] = (display_rows[target_row][0], display_rows[target_row][1] + instances)
            else:
                display_rows.extend(other_ips)

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

        # 计算每个IP方块的位置
        ip_positions = {}
        row_idx = 0

        for base_type, instances in ip_type_count.items():
            num_instances = len(instances)
            row_width = num_instances * grid_square_size + (num_instances - 1) * grid_spacing
            row_start_x = node_x - row_width / 2
            row_y = node_y + total_content_height / 2 - row_idx * (grid_square_size + row_spacing) - grid_square_size / 2

            for col_idx in range(num_instances):
                block_x = row_start_x + col_idx * (grid_square_size + grid_spacing) + grid_square_size / 2
                block_y = row_y

                # 找到对应的完整ip_type名称
                ip_type_full = f"{base_type}_{col_idx}"
                if ip_type_full in active_ips:
                    ip_positions[ip_type_full] = (block_x, block_y, grid_square_size)

            row_idx += 1

        return ip_positions

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
        show_link_labels: bool = True,
        draw_links: bool = True,
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
        # 使用LinkDataProcessor提取链路统计数据
        links, utilization_stats = LinkDataProcessor.extract_link_stats(network, mode, config)

        # 获取网络节点
        if hasattr(network, "queues") and network.queues:
            actual_nodes = list(network.queues.keys())
        else:
            # 默认5x4拓扑
            actual_nodes = list(range(config.NUM_ROW * config.NUM_COL))

        # 计算节点位置（使用缓存优化）
        node_spacing = 3.0
        orig_rows = config.NUM_ROW
        orig_cols = config.NUM_COL
        num_nodes = orig_rows * orig_cols

        # 调用缓存函数获取位置
        pos_tuple = _calculate_node_positions_cached(orig_rows, orig_cols, num_nodes, rotation, offset_x, offset_y, node_spacing)
        # 转换为字典
        pos = dict(pos_tuple)

        # 计算节点大小
        square_size = np.sqrt(node_size) / 50

        # 绘制节点背景
        self._draw_nodes(fig, pos, square_size, actual_nodes, config=config, ip_bandwidth_data=ip_bandwidth_data, mode=mode, die_id=die_id, is_d2d_scenario=is_d2d_scenario)

        # 绘制链路箭头（可选）
        if draw_links:
            # 使用LinkDataProcessor处理链路数据
            edge_labels, edge_colors, self_loop_labels = LinkDataProcessor.process_links_for_drawing(links, actual_nodes, mode)

            # 绘制链路箭头（传递完整的统计数据用于hover）
            link_arrow_anns = self._draw_link_arrows(
                fig, pos, edge_labels, edge_colors, links, config, square_size, rotation, link_label_fontsize, utilization_stats, is_d2d_scenario, show_link_labels
            )
            # 将箭头annotations添加到figure
            if link_arrow_anns:
                existing_anns = list(fig.layout.annotations) if fig.layout.annotations else []
                fig.layout.annotations = existing_anns + link_arrow_anns

            # 绘制自环标签
            if self_loop_labels and show_link_labels:  # 只在需要显示标签时绘制
                selfloop_anns = self._draw_self_loop_labels(fig, pos, self_loop_labels, config, square_size, rotation, link_label_fontsize, utilization_stats)
                if selfloop_anns:
                    existing_anns = list(fig.layout.annotations) if fig.layout.annotations else []
                    fig.layout.annotations = existing_anns + selfloop_anns

        # 批量收集所有节点的IP方块
        if ip_bandwidth_data is not None:
            all_ip_shapes = []
            all_ip_annotations = []
            # 添加用于点击事件的不可见scatter点
            ip_click_x = []
            ip_click_y = []
            ip_click_text = []
            ip_click_customdata = []

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

                # 为每个活跃IP添加不可见的点用于捕获点击事件
                # 需要计算每个IP方块的实际位置，而不是节点中心
                if ip_bandwidth_data is not None:
                    if is_d2d_scenario and die_id is not None:
                        mode_data = ip_bandwidth_data.get(die_id, {}).get(mode, {})
                    else:
                        mode_data = ip_bandwidth_data.get(mode, {})

                    physical_row = node // config.NUM_COL
                    physical_col = node % config.NUM_COL

                    # 收集该节点的所有活跃IP及其带宽
                    active_ips = {}
                    for ip_type, data_matrix in mode_data.items():
                        if is_d2d_scenario and ip_type.lower() in ["d2d_rn", "d2d_sn"]:
                            continue
                        if physical_row < data_matrix.shape[0] and physical_col < data_matrix.shape[1]:
                            bandwidth = data_matrix[physical_row, physical_col]
                            if bandwidth > 0.001:
                                active_ips[ip_type] = bandwidth

                    # 为每个活跃IP创建scatter点，位置对应IP方块的实际位置
                    if active_ips:
                        ip_positions = self._calculate_ip_block_positions(x, y, active_ips, square_size)
                        for ip_type, (ip_x, ip_y, ip_size) in ip_positions.items():
                            ip_click_x.append(ip_x)
                            ip_click_y.append(ip_y)
                            ip_click_text.append(f"{ip_type} @ Pos {node}")
                            ip_click_customdata.append([die_id if die_id is not None else 0, ip_type, node])

            # 批量添加IP方块shapes
            if all_ip_shapes:
                current_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
                fig.update_layout(shapes=current_shapes + all_ip_shapes)

            # 批量添加IP标签annotations（使用循环，Plotly内部会优化）
            for ann in all_ip_annotations:
                fig.add_annotation(**ann)

            # 添加不可见的scatter点用于捕获点击事件（每个IP方块一个点）
            if ip_click_x:
                fig.add_trace(
                    go.Scatter(
                        x=ip_click_x,
                        y=ip_click_y,
                        mode="markers",
                        marker=dict(
                            size=square_size * 16,  # 约为IP方块大小，确保覆盖
                            opacity=0.0,  # 稍微可见，便于调试（生产环境可改为0.01）
                            color="rgba(0,0,0,0.05)",
                            line=dict(width=0),
                        ),
                        text=ip_click_text,
                        customdata=ip_click_customdata,
                        hoverinfo="text",
                        showlegend=False,
                        name="IP Click Handler",
                    )
                )

        return pos
