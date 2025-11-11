"""
可视化器模块 - 提供流量图、带宽曲线和热图绘制功能

包含:
1. FlowGraphRenderer - 流量图渲染类
2. BandwidthPlotter - 带宽曲线绘制类
3. HeatmapDrawer - 热图绘制类
4. IPInfoBoxDrawer - IP信息框绘制类
"""

import sys
import matplotlib

# 兼容不同系统的matplotlib backend配置
if sys.platform == "darwin":  # macOS
    try:
        matplotlib.use("macosx")
    except ImportError:
        matplotlib.use("Agg")


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from .analyzers import IP_COLOR_MAP, MAX_BANDWIDTH_NORMALIZATION
import warnings
import os
import math
import traceback




class FlowGraphRenderer:
    """流量图渲染器 - 绘制网络拓扑和流量分布图"""

    def __init__(self):
        """初始化流量图渲染器"""
        pass

    def draw_single_die_flow(
        self,
        ax,
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
        在指定的axes上绘制单个Die的流量图

        Args:
            ax: matplotlib axes对象
            network: Network对象
            config: 配置对象
            ip_bandwidth_data: IP带宽数据字典
                - 单Die场景: {mode: {ip_type: data_matrix}}
                - D2D场景: {die_id: {mode: {ip_type: data_matrix}}}
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
        """
        # 创建NetworkX图
        G = nx.DiGraph()

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
                traceback.print_exc()
                links = {}

        # 获取网络节点
        if hasattr(network, "queues") and network.queues:
            actual_nodes = list(network.queues.keys())
        else:
            # 默认5x4拓扑
            actual_nodes = list(range(config.NUM_ROW * config.NUM_COL))

        # 添加节点到图中
        G.add_nodes_from(actual_nodes)

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

        # 添加边和标签
        edge_labels = {}
        edge_colors = {}
        self_loop_labels = {}

        for link_key, value in links.items():
            # 处理新架构：link可能是(i, j)或(i, j, 'h'/'v')
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

            # 自环link单独存储
            if i == j:
                if direction:
                    self_loop_labels[(i, direction)] = (formatted_label, color)
                else:
                    self_loop_labels[(i, "unknown")] = (formatted_label, color)
            else:
                G.add_edge(i, j, weight=value)
                if display_value > 0:
                    edge_labels[(i, j)] = formatted_label
                    edge_colors[(i, j)] = color
                else:
                    edge_colors[(i, j)] = color

        # 计算节点大小
        square_size = np.sqrt(node_size) / 50

        # 绘制网络边
        for i, j in G.edges():
            if i not in pos or j not in pos:
                continue

            color = edge_colors.get((i, j), (0.8, 0.8, 0.8))
            x1, y1 = pos[i]
            x2, y2 = pos[j]

            if i != j:
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx * dx + dy * dy)
                if length > 0:
                    dx, dy = dx / length, dy / length

                    perp_dx, perp_dy = dy * 0.1, -dx * 0.1

                    has_reverse = G.has_edge(j, i)
                    if has_reverse:
                        start_x = x1 + dx * square_size / 2 + perp_dx
                        start_y = y1 + dy * square_size / 2 + perp_dy
                        end_x = x2 - dx * square_size / 2 + perp_dx
                        end_y = y2 - dy * square_size / 2 + perp_dy
                    else:
                        start_x = x1 + dx * square_size / 2
                        start_y = y1 + dy * square_size / 2
                        end_x = x2 - dx * square_size / 2
                        end_y = y2 - dy * square_size / 2

                    arrow = FancyArrowPatch(
                        (start_x, start_y),
                        (end_x, end_y),
                        arrowstyle="-|>",
                        mutation_scale=8,
                        color=color,
                        zorder=1,
                        linewidth=1,
                    )
                    ax.add_patch(arrow)

        # 绘制边标签
        if edge_labels:
            # 计算颜色映射范围
            link_values = [float(links.get((i, j), 0)) for (i, j) in edge_labels.keys()]
            link_mapping_max = max(link_values) if link_values else 0.0
            link_mapping_min = max(0.6 * link_mapping_max, 100) if mode == "total" else 0.0

            for (i, j), label in edge_labels.items():
                if i in pos and j in pos:
                    edge_value = float(links.get((i, j), 0))
                    if edge_value == 0.0:
                        continue

                    # 计算颜色
                    if mode == "total":
                        if edge_value <= link_mapping_min:
                            intensity = 0.0
                        else:
                            intensity = (edge_value - link_mapping_min) / (link_mapping_max - link_mapping_min)
                        intensity = min(max(intensity, 0.0), 1.0)
                        color = (intensity, 0, 0)
                    else:
                        color = (edge_value, 0, 0)

                    x1, y1 = pos[i]
                    x2, y2 = pos[j]
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    dx, dy = x2 - x1, y2 - y1

                    # 检测是否有反向边
                    has_reverse = G.has_edge(j, i)

                    # 双向link需要计算偏移量（考虑Die旋转）
                    if has_reverse:
                        # 计算Die内部方向（基于原始节点行列）
                        orig_i_row = i // config.NUM_COL
                        orig_i_col = i % config.NUM_COL
                        orig_j_row = j // config.NUM_COL
                        orig_j_col = j % config.NUM_COL
                        is_horizontal_in_die = orig_i_row == orig_j_row

                        # 计算屏幕方向
                        is_horizontal_on_screen = abs(dx) > abs(dy)

                        # 根据Die内部方向计算偏移量
                        is_90_or_270 = abs(rotation) in [90, 270]

                        if is_horizontal_in_die:
                            # Die内水平链路
                            offset_magnitude = 0.70 if is_90_or_270 else 0.35
                            if i < j:
                                offset_x_die, offset_y_die = 0, offset_magnitude
                            else:
                                offset_x_die, offset_y_die = 0, -offset_magnitude
                        else:
                            # Die内垂直链路
                            offset_magnitude = 0.35 if is_90_or_270 else 0.70
                            if i < j:
                                offset_x_die, offset_y_die = -offset_magnitude, 0
                            else:
                                offset_x_die, offset_y_die = offset_magnitude, 0

                        # 应用旋转矩阵变换
                        import math

                        angle_rad = math.radians(rotation)
                        cos_a = math.cos(angle_rad)
                        sin_a = math.sin(angle_rad)
                        offset_x_screen = offset_x_die * cos_a - offset_y_die * sin_a
                        offset_y_screen = offset_x_die * sin_a + offset_y_die * cos_a

                        label_x = mid_x + offset_x_screen
                        label_y = mid_y - offset_y_screen
                    else:
                        # 单向link：标签直接放在中间
                        label_x = mid_x
                        label_y = mid_y

                    ax.text(label_x, label_y, label, ha="center", va="center", fontsize=link_label_fontsize, fontweight="normal", color=color)

        # 绘制自环边标签
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
                # 屏幕水平自环：放左右两边
                if rotated_col == 0:
                    # 屏幕左边
                    offset_x_screen = -square_size / 2 - offset_dist
                    offset_y_screen = 0
                    text_rotation = 90  # 从下往上读
                else:
                    # 屏幕右边
                    offset_x_screen = square_size / 2 + offset_dist
                    offset_y_screen = 0
                    text_rotation = -90  # 从上往下读
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

            ax.text(label_x, label_y, f"{label}", ha="center", va="center", color=color, fontweight="normal", fontsize=link_label_fontsize, rotation=text_rotation)

        # 绘制节点
        for node, (x, y) in pos.items():
            rect = Rectangle(
                (x - square_size / 2, y - square_size / 2),
                width=square_size,
                height=square_size,
                color="#E8F5E9",
                ec="black",
                zorder=2,
            )
            ax.add_patch(rect)

            # 绘制IP信息
            if ip_bandwidth_data is not None:
                self._draw_ip_info_in_node(
                    ax,
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

        return pos

    def draw_flow_graph(self, network, ip_bandwidth_data: Dict = None, config=None, mode: str = "utilization", node_size: int = 2000, save_path: str = None):
        """
        绘制网络流量图

        Args:
            network: Network对象
            ip_bandwidth_data: IP带宽数据字典
            config: 配置对象
            mode: 可视化模式
            node_size: 节点大小
            save_path: 保存路径（如果为None则显示）
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
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect("equal")

        # 调用单Die绘制方法
        pos = self.draw_single_die_flow(
            ax=ax,
            network=network,
            config=config,
            ip_bandwidth_data=ip_bandwidth_data,
            mode=mode,
            node_size=node_size,
            max_ip_bandwidth=max_ip_bandwidth,
            min_ip_bandwidth=min_ip_bandwidth,
        )

        # 设置图表
        title = f"Network Flow - {mode.capitalize()}"
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.axis("equal")
        ax.margins(0.05)
        ax.axis("off")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout(pad=1.5)

        if save_path:
            plt.savefig(os.path.join(save_path), dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def _draw_ip_info_in_node(
        self, ax, x, y, node, config, mode, square_size, ip_bandwidth_data, max_ip_bandwidth=None, min_ip_bandwidth=None, die_id=None, is_d2d_scenario=False, show_bandwidth_value=True
    ):
        """
        在节点内绘制IP信息（小方块+透明度）

        Args:
            ax: matplotlib坐标轴
            x, y: 节点中心位置
            node: 节点ID
            config: 配置对象
            mode: 显示模式
            square_size: 节点方块大小
            ip_bandwidth_data: IP带宽数据字典
                - 单Die场景: {mode: {ip_type: data_matrix}}
                - D2D场景: {die_id: {mode: {ip_type: data_matrix}}}
            max_ip_bandwidth: 全局最大IP带宽
            min_ip_bandwidth: 全局最小IP带宽
            die_id: Die ID（D2D场景必填）
            is_d2d_scenario: 是否为D2D场景
            show_bandwidth_value: 是否显示带宽数值
        """
        from matplotlib.patches import Rectangle
        from collections import defaultdict
        from src.analysis.analyzers import IP_COLOR_MAP, RN_TYPES, SN_TYPES

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

        # 如果没有活跃IP，直接返回
        if not active_ips:
            return

        # 按IP基本类型分组,但保留每个实例的独立带宽值
        ip_type_count = defaultdict(list)
        for ip_type, bw in active_ips:
            # 提取基本类型(如ddr_0 -> DDR)
            base_type = ip_type.split("_")[0] if "_" in ip_type else ip_type
            ip_type_count[base_type].append(bw)

        # 按RN/SN分类排序
        rn_ips = [(k, v) for k, v in ip_type_count.items() if k.upper() in RN_TYPES]
        sn_ips = [(k, v) for k, v in ip_type_count.items() if k.upper() in SN_TYPES]
        other_ips = [(k, v) for k, v in ip_type_count.items() if k.upper() not in RN_TYPES + SN_TYPES]

        # 按带宽总和排序
        rn_ips.sort(key=lambda x: sum(x[1]), reverse=True)
        sn_ips.sort(key=lambda x: sum(x[1]), reverse=True)
        other_ips.sort(key=lambda x: sum(x[1]), reverse=True)

        # 构建最终显示列表（从上到下：RN -> SN -> Other）
        display_rows = []
        display_rows.extend(rn_ips)
        display_rows.extend(sn_ips)

        # 如果总行数超过MAX_ROWS，合并other_ips
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

        # 计算布局参数 - 在节点内部绘制
        num_ip_types = len(ip_type_count)
        max_instances = max(len(instances) for instances in ip_type_count.values())

        # 计算小方块大小和间距（使用节点内部空间）
        available_width = square_size * 0.90
        available_height = square_size * 0.90
        grid_spacing = square_size * 0.10
        row_spacing = square_size * 0.1

        max_square_width = (available_width - (max_instances - 1) * grid_spacing) / max_instances
        max_square_height = (available_height - (num_ip_types - 1) * row_spacing) / num_ip_types
        grid_square_size = min(max_square_width, max_square_height, square_size * 0.5)

        # 计算总内容高度
        total_content_height = num_ip_types * grid_square_size + (num_ip_types - 1) * row_spacing

        # 绘制IP小方块
        row_idx = 0
        for ip_type, instances in ip_type_count.items():
            num_instances = len(instances)
            # ip_type已经是基本类型,直接用于颜色映射
            base_type = ip_type.upper()
            ip_color = IP_COLOR_MAP.get(base_type, IP_COLOR_MAP["OTHER"])

            # 计算当前行的总宽度
            row_width = num_instances * grid_square_size + (num_instances - 1) * grid_spacing

            # 计算当前行的起始位置（水平居中在节点内部）
            row_start_x = x - row_width / 2

            # 计算当前行的垂直位置（垂直居中在节点内部）
            row_y = y + total_content_height / 2 - row_idx * (grid_square_size + row_spacing) - grid_square_size / 2

            # 绘制该类型的所有实例
            for col_idx, bandwidth in enumerate(instances):
                # 计算小方块位置
                block_x = row_start_x + col_idx * (grid_square_size + grid_spacing) + grid_square_size / 2
                block_y = row_y

                # 计算透明度（使用全局带宽范围）
                alpha = self._calculate_bandwidth_alpha(bandwidth, min_ip_bandwidth if min_ip_bandwidth is not None else 0, max_ip_bandwidth if max_ip_bandwidth is not None else 1)

                # 绘制小方块
                ip_block = Rectangle(
                    (block_x - grid_square_size / 2, block_y - grid_square_size / 2),
                    width=grid_square_size,
                    height=grid_square_size,
                    facecolor=ip_color,
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=alpha,
                    zorder=3,
                )
                ax.add_patch(ip_block)

                # 在小方块中显示带宽数值（可选）
                if show_bandwidth_value and grid_square_size >= square_size * 0.4:
                    bw_text = f"{bandwidth:.0f}" if bandwidth >= 10 else f"{bandwidth:.1f}"
                    ax.text(
                        block_x,
                        block_y,
                        bw_text,
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="normal",
                        color="black",
                        zorder=4,
                    )

            row_idx += 1

    def _calculate_bandwidth_alpha(self, bandwidth, min_bw, max_bw):
        """计算带宽透明度"""
        # 如果没有提供范围，使用默认透明度
        if max_bw is None or min_bw is None:
            return 0.8

        # 防止除零错误
        if max_bw <= min_bw or max_bw <= 0:
            return 0.8

        # 归一化到0-1范围
        normalized = max(0, min(1, (bandwidth - min_bw) / (max_bw - min_bw)))
        return 0.3 + 0.7 * normalized  # 透明度范围0.3-1.0

    def _calculate_die_offsets_from_layout(self, die_layout, die_layout_type, die_width, die_height, dies=None, config=None, die_rotations=None):
        """
        根据推断的 Die 布局计算绘图偏移量和画布大小，包含对齐优化

        Args:
            die_layout: Die 布局位置字典 {die_id: (x, y)}
            die_layout_type: 布局类型字符串，如 "2x2", "2x1" 等
            die_width: 基础Die的宽度（旋转前）
            die_height: 基础Die的高度（旋转前）
            dies: Die模型字典 {die_id: die_model}，用于对齐计算
            config: 配置对象，用于对齐计算
            die_rotations: Die旋转角度字典 {die_id: rotation}

        Returns:
            (die_offsets, figsize): Die偏移量字典和画布大小
        """
        if not die_layout:
            raise ValueError("die_layout不能为空")

        if die_rotations is None:
            die_rotations = {}

        # 计算布局尺寸
        max_x = max(pos[0] for pos in die_layout.values()) if die_layout else 0
        max_y = max(pos[1] for pos in die_layout.values()) if die_layout else 0

        # 计算每个Die旋转后的实际尺寸
        die_sizes = {}
        for die_id in die_layout.keys():
            rotation = die_rotations.get(die_id, 0)
            if rotation in [90, 270]:
                # 90度或270度旋转：宽高互换
                die_sizes[die_id] = (die_height, die_width)
            else:
                # 0度或180度旋转：宽高不变
                die_sizes[die_id] = (die_width, die_height)

        # 计算每行每列的最大尺寸
        max_width_per_col = {}
        max_height_per_row = {}
        for die_id, (grid_x, grid_y) in die_layout.items():
            w, h = die_sizes[die_id]
            max_width_per_col[grid_x] = max(max_width_per_col.get(grid_x, 0), w)
            max_height_per_row[grid_y] = max(max_height_per_row.get(grid_y, 0), h)

        # 计算每个Die的偏移量（累加前面所有Die的尺寸）
        die_offsets = {}
        gap_x = 7.0  # Die之间的横向间隙
        gap_y = 5.0 if len(die_layout.values()) == 2 else 1

        for die_id, (grid_x, grid_y) in die_layout.items():
            # X方向：累加左侧所有列的宽度 + 间隙
            offset_x = sum(max_width_per_col.get(x, 0) + gap_x for x in range(grid_x))
            # Y方向：累加下方所有行的高度 + 间隙
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

        # 计算总的画布尺寸（基于累加后的实际尺寸）
        total_width = sum(max_width_per_col.get(x, 0) for x in range(max_x + 1)) + gap_x * max_x + 2
        total_height = sum(max_height_per_row.get(y, 0) for y in range(max_y + 1)) + gap_y * max_y + 2

        # 转换为英寸尺寸（假设每个单位 = 0.3英寸）
        canvas_width = total_width * 0.3
        canvas_height = total_height * 0.3

        # 限制画布尺寸范围
        canvas_width = max(min(canvas_width, 20), 14)  # 14-20英寸
        canvas_height = max(min(canvas_height, 16), 10)  # 10-16英寸

        figsize = (canvas_width, canvas_height)

        return die_offsets, figsize

    def draw_d2d_flow_graph(
        self, die_networks: Dict = None, dies: Dict = None, config=None, die_ip_bandwidth_data: Dict = None, mode: str = "utilization", node_size: int = 2500, save_path: str = None
    ):
        """
        绘制D2D系统流量图（多Die布局）

        Args:
            die_networks: Die网络字典
            dies: Die模型字典
            config: 配置对象
            die_ip_bandwidth_data: D2D IP带宽数据 {die_id: {mode: {ip_type: bandwidth_matrix}}}
            mode: 可视化模式
            node_size: 节点大小
            save_path: 保存路径
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
        base_die_rows = 5  # 默认5x4拓扑
        base_die_cols = 4
        node_spacing = 3.0

        die_width = (base_die_cols - 1) * node_spacing
        die_height = (base_die_rows - 1) * node_spacing

        # 使用动态布局计算
        die_offsets, figsize = self._calculate_die_offsets_from_layout(die_layout, die_layout_type, die_width, die_height, dies=dies, config=config, die_rotations=die_rotations)

        # 创建画布
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        # 收集所有IP类型和全局带宽范围（用于归一化透明度）
        all_ip_bandwidths = []
        ip_bandwidth_data_dict = {}  # 存储每个die的IP带宽数据

        # 优先使用传入的die_ip_bandwidth_data参数
        if die_ip_bandwidth_data:
            ip_bandwidth_data_dict = die_ip_bandwidth_data
            for die_id, die_data in die_ip_bandwidth_data.items():
                if mode in die_data:
                    mode_data = die_data[mode]
                    for ip_type, data_matrix in mode_data.items():
                        nonzero_bw = data_matrix[data_matrix > 0.001]
                        if len(nonzero_bw) > 0:
                            all_ip_bandwidths.extend(nonzero_bw.tolist())
        # 否则从dies中获取IP带宽数据
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

        # 计算全局IP带宽范围
        max_ip_bandwidth = max(all_ip_bandwidths) if all_ip_bandwidths else None
        min_ip_bandwidth = min(all_ip_bandwidths) if all_ip_bandwidths else None

        # 为每个Die绘制流量图并收集节点位置
        die_node_positions = {}
        for die_id, network in die_networks_for_draw.items():
            offset_x, offset_y = die_offsets[die_id]
            die_model = dies.get(die_id) if dies else None
            die_rotation = die_rotations.get(die_id, 0)

            # 获取该Die的IP带宽数据
            ip_bandwidth_data = ip_bandwidth_data_dict.get(die_id) if ip_bandwidth_data_dict else None

            node_positions = self.draw_single_die_flow(
                ax=ax,
                network=network,
                config=die_model.config if die_model else config,
                mode=mode,
                node_size=node_size,
                die_id=die_id,
                offset_x=offset_x,
                offset_y=offset_y,
                rotation=die_rotation,
                ip_bandwidth_data=ip_bandwidth_data_dict,  # 传递完整的字典
                max_ip_bandwidth=max_ip_bandwidth,
                min_ip_bandwidth=min_ip_bandwidth,
                is_d2d_scenario=True,  # D2D场景
                show_ip_bandwidth_value=False,  # D2D flow图不显示IP带宽数值
                link_label_fontsize=8,  # D2D场景使用更小的字体
            )
            die_node_positions[die_id] = node_positions

        # 添加Die标签 - 根据连接方向智能放置
        for die_id in die_node_positions.keys():
            node_positions = die_node_positions[die_id]
            if node_positions:
                xs = [p[0] for p in node_positions.values()]
                ys = [p[1] for p in node_positions.values()]
                die_center_x = (min(xs) + max(xs)) / 2
                die_center_y = (min(ys) + max(ys)) / 2

                # 根据Die布局确定标签位置
                if die_id in die_layout:
                    grid_x, grid_y = die_layout[die_id]

                    # 判断连接方向
                    other_dies = [did for did in die_layout.keys() if did != die_id]
                    if other_dies:
                        other_die_id = other_dies[0]
                        other_grid_x, other_grid_y = die_layout[other_die_id]

                        # 判断连接方向
                        is_vertical_connection = grid_y != other_grid_y
                        is_horizontal_connection = grid_x != other_grid_x

                        if is_vertical_connection:
                            # 垂直连接：标题放在左边或右边
                            if grid_x == 0:  # 左边的Die，标题放在左边
                                label_x = min(xs) - 3
                                label_y = die_center_y
                            else:  # 右边的Die，标题放在右边
                                label_x = max(xs) + 3
                                label_y = die_center_y
                        elif is_horizontal_connection:
                            # 水平连接：标题放在下边
                            label_x = die_center_x
                            label_y = min(ys) - 2
                        else:
                            # 其他情况：默认放在上方
                            label_x = die_center_x
                            label_y = max(ys) + 2.5
                    else:
                        # 只有一个Die时：默认放在上方
                        label_x = die_center_x
                        label_y = max(ys) + 2.5
                else:
                    # 没有布局信息时，默认放在上方
                    label_x = die_center_x
                    label_y = max(ys) + 2.5

                ax.text(
                    label_x,
                    label_y,
                    f"Die {die_id}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7, edgecolor="none"),
                    rotation=0,  # Die标签统一水平显示
                )

        # 绘制跨Die连接（完整实现）
        if dies and len(dies) > 1:
            try:
                # 计算D2D带宽
                d2d_bandwidth = self._calculate_d2d_sys_bandwidth(dies, config)
                # 绘制跨Die连接
                self._draw_cross_die_connections(ax, d2d_bandwidth, die_node_positions, config, dies, die_offsets)
            except Exception as e:
                import traceback

                traceback.print_exc()

        # 收集所有使用的IP类型（用于图例，过滤掉d2d_rn和d2d_sn）
        used_ip_types = set()
        if die_ip_bandwidth_data:
            for die_id, die_data in die_ip_bandwidth_data.items():
                if mode in die_data:
                    for ip_type in die_data[mode].keys():
                        # 过滤掉D2D专用节点
                        if ip_type.lower() not in ["d2d_rn", "d2d_sn"]:
                            used_ip_types.add(ip_type.upper().split("_")[0])

        # 添加IP类型图例
        if used_ip_types:
            self._add_ip_type_legend(ax, used_ip_types)

        # 添加带宽colorbar（如果有IP带宽数据）
        if all_ip_bandwidths:
            self._add_bandwidth_range_legend(ax, min_ip_bandwidth, max_ip_bandwidth)

        # 设置图表
        title = f"D2D Flow Graph - {mode.capitalize()}"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("equal")
        ax.margins(0.1)
        ax.axis("off")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                plt.tight_layout(pad=0.5)
                plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
            plt.close()
            return save_path
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                plt.tight_layout(pad=0.5)
                plt.show()
            return None

    def _calculate_d2d_sys_bandwidth(self, dies, config):
        """
        计算每个Die每个D2D节点的AXI通道带宽

        Args:
            dies: Dict[die_id, die_model] - Die模型字典
            config: 配置对象

        Returns:
            dict: D2D_Sys AXI通道带宽统计 {die_id: {node_pos: {channel: bandwidth_gbps}}}
        """
        d2d_sys_bandwidth = {}

        # 从dies中获取仿真周期（从第一个die的cycle获取）
        sim_end_cycle = next(iter(dies.values())).cycle
        network_frequency = config.NETWORK_FREQUENCY
        flit_size = config.FLIT_SIZE
        time_ns = sim_end_cycle / network_frequency

        for die_id, die_model in dies.items():
            d2d_sys_bandwidth[die_id] = {}

            # 从该Die的所有d2d_systems分别计算每个节点的带宽
            if hasattr(die_model, "d2d_systems"):
                for pos_key, d2d_sys in die_model.d2d_systems.items():
                    # pos_key的格式: "节点_to_目标Die_目标节点" 或者 简单的节点位置
                    # 为每个节点单独计算带宽
                    node_bandwidth = {"AR": 0.0, "R": 0.0, "AW": 0.0, "W": 0.0, "B": 0.0}

                    if hasattr(d2d_sys, "axi_channel_flit_count"):
                        # 计算该节点各通道的带宽
                        for channel, flit_count in d2d_sys.axi_channel_flit_count.items():
                            # 计算带宽：(flit数 × flit大小) / 时间(ns) = bytes/ns = GB/s
                            if time_ns > 0:
                                bandwidth_gbps = (flit_count * flit_size) / time_ns
                                node_bandwidth[channel] = bandwidth_gbps

                    # 将pos_key和通道带宽存储在d2d_sys_bandwidth中
                    # 格式: {die_id: {pos_key: {channel: bandwidth}}}
                    d2d_sys_bandwidth[die_id][pos_key] = node_bandwidth.copy()

        return d2d_sys_bandwidth

    def _calculate_arrow_vectors(self, from_x, from_y, to_x, to_y):
        """
        计算箭头方向向量

        Args:
            from_x, from_y: 起始点坐标
            to_x, to_y: 终点坐标

        Returns:
            tuple: (ux, uy, perpx, perpy) 单位方向向量和垂直向量，如果长度为0则返回None
        """
        import numpy as np

        dx, dy = to_x - from_x, to_y - from_y
        length = np.sqrt(dx * dx + dy * dy)

        if length > 0:
            ux, uy = dx / length, dy / length
            perpx, perpy = -uy * 0.2, ux * 0.2
            return ux, uy, perpx, perpy
        else:
            return None

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

        if dx == 0:  # X坐标相同，垂直连接
            return "vertical"
        elif dy == 0:  # Y坐标相同，水平连接
            return "horizontal"
        else:  # 对角连接
            return "diagonal"

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

    def _draw_cross_die_connections(self, ax, d2d_bandwidth, die_node_positions, config, dies=None, die_offsets=None):
        """
        绘制跨Die数据带宽连接（只显示R和W通道的数据流）
        基于推断的布局和D2D_PAIRS配置绘制连接
        """
        import numpy as np

        try:
            # 使用推断的D2D连接对
            d2d_pairs = config.D2D_PAIRS

            if not d2d_pairs:
                return

            # 获取Die布局信息
            die_layout = config.die_layout_positions

            # 遍历所有D2D连接对
            arrow_index = 0
            for die0_id, die0_node, die1_id, die1_node in d2d_pairs:
                # 双向检查流量并绘制
                directions = [
                    (die0_id, die0_node, die1_id, die1_node),  # Die0 -> Die1
                    (die1_id, die1_node, die0_id, die0_node),  # Die1 -> Die0
                ]

                for from_die, from_node, to_die, to_node in directions:
                    # 构造复合键
                    key = f"{from_node}_to_{to_die}_{to_node}"

                    # 检查写数据流量 (W通道)
                    w_bw = d2d_bandwidth.get(from_die, {}).get(key, {}).get("W", 0.0)
                    # 检查读数据返回流量 (R通道)
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

                    # 对角连接：围绕连线中点顺时针旋转
                    if connection_type == "diagonal":
                        # 计算连线中点
                        mid_x = (from_x + to_x) / 2
                        mid_y = (from_y + to_y) / 2

                        # 顺时针旋转8度
                        angle = -8 * np.pi / 180
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)

                        # 旋转起点
                        dx_from = from_x - mid_x
                        dy_from = from_y - mid_y
                        from_x = mid_x + dx_from * cos_a - dy_from * sin_a
                        from_y = mid_y + dx_from * sin_a + dy_from * cos_a

                        # 旋转终点
                        dx_to = to_x - mid_x
                        dy_to = to_y - mid_y
                        to_x = mid_x + dx_to * cos_a - dy_to * sin_a
                        to_y = mid_y + dx_to * sin_a + dy_to * cos_a

                    # 计算箭头向量
                    arrow_vectors = self._calculate_arrow_vectors(from_x, from_y, to_x, to_y)
                    if arrow_vectors is None:
                        continue

                    ux, uy, perpx, perpy = arrow_vectors

                    # 合并读写通道带宽（同一AXI通道）
                    total_bw = w_bw + r_bw

                    # 绘制单条箭头，显示总带宽
                    self._draw_single_d2d_arrow(ax, from_x, from_y, to_x, to_y, ux, uy, perpx, perpy, total_bw, arrow_index, connection_type)
                    arrow_index += 1

        except Exception as e:
            import traceback

            traceback.print_exc()

    def _draw_single_d2d_arrow(self, ax, start_node_x, start_node_y, end_node_x, end_node_y, ux, uy, perpx, perpy, bandwidth, connection_index, connection_type=None):
        """绘制单个D2D箭头"""
        import numpy as np
        from matplotlib.patches import FancyArrowPatch

        # 计算箭头起止坐标（留出节点空间）
        if connection_type == "diagonal":
            node_offset = 1.2
            perp_offset = 1.2
            start_x = start_node_x + ux * node_offset + perpx * perp_offset
            start_y = start_node_y + uy * node_offset + perpy * perp_offset
            end_x = end_node_x - ux * node_offset + perpx * perp_offset
            end_y = end_node_y - uy * node_offset + perpy * perp_offset
        else:
            start_x = start_node_x + ux * 1.2 + perpx
            start_y = start_node_y + uy * 1.2 + perpy
            end_x = end_node_x - ux * 1.2 + perpx
            end_y = end_node_y - uy * 1.2 + perpy

        # 确定颜色和标签
        # MAX_BANDWIDTH_NORMALIZATION = 100.0  # 归一化基准
        if bandwidth > 0.001:
            # 有数据流量
            intensity = min(bandwidth / MAX_BANDWIDTH_NORMALIZATION, 1.0)
            color = (intensity, 0, 0)  # 红色
            label_text = f"{bandwidth:.1f}"
            linewidth = 2.5
            zorder = 5
        else:
            # 无数据流量 - 灰色实线
            color = (0.7, 0.7, 0.7)
            label_text = None
            linewidth = 2.5
            zorder = 4

        # 绘制箭头
        arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle="-|>", mutation_scale=10, color=color, linewidth=linewidth, zorder=zorder)
        ax.add_patch(arrow)

        # 只在有流量时添加标签
        if label_text:
            dx = end_x - start_x
            dy = end_y - start_y

            # 对角连接使用靠近终点的位置，其他连接使用中点
            if connection_type == "diagonal":
                label_x = start_x + dx * 0.85
                label_y_base = start_y + dy * 0.85

                if (dx > 0 and dy > 0) or (dx > 0 and dy < 0):
                    label_y = label_y_base + 0.6
                else:
                    label_y = label_y_base - 0.6
            else:
                # 垂直和水平连接：使用中点
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2

                is_horizontal = abs(dx) > abs(dy)

                if is_horizontal:
                    label_x = mid_x
                    label_y = mid_y + (0.5 if dx > 0 else -0.5)
                else:
                    label_x = mid_x + (dy * 0.1 if dx > 0 else -dy * 0.1)
                    label_y = mid_y - 0.15

            # 计算箭头角度
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)

            # 确保文字不会倒置
            if angle_deg > 90:
                angle_deg -= 180
            elif angle_deg < -90:
                angle_deg += 180

            # 绘制标签
            ax.text(label_x, label_y, label_text, ha="center", va="center", fontsize=8, fontweight="normal", color=color, rotation=angle_deg, rotation_mode="anchor")

    def draw_ip_bandwidth_heatmap(self, dies: Dict = None, config=None, die_ip_bandwidth_data: Dict = None, mode: str = "total", node_size: int = 4000, save_path: Optional[str] = None):
        """
        绘制IP带宽热力图（不显示链路，只显示节点和IP带宽）

        本质上是 draw_d2d_flow_graph 的简化版本：
        - 不绘制链路和箭头
        - 不显示节点编号
        - 使用特殊的节点背景色（有IP=浅黄，无IP=浅灰）
        - 节点尺寸更大
        - IP方块显示带宽数值

        Args:
            dies: Die模型字典 {die_id: die_model}
            config: D2D配置对象
            die_ip_bandwidth_data: Die级别的IP带宽数据 {die_id: {mode: {ip_type: matrix}}}
            mode: 显示模式 ('read', 'write', 'total')
            node_size: 节点大小
            save_path: 保存路径

        Returns:
            str: 保存的图片路径，如果没有保存则返回None
        """
        if dies is None or len(dies) == 0:
            print("警告: 没有提供Die数据")
            return None

        if not die_ip_bandwidth_data:
            print("警告: 没有die_ip_bandwidth_data数据，跳过IP带宽热力图绘制")
            return None

        # 获取Die布局配置
        die_layout = config.die_layout_positions
        die_layout_type = config.die_layout_type
        die_rotations = config.DIE_ROTATIONS

        # 计算Die尺寸
        node_spacing = 3.0
        first_die = list(dies.values())[0]
        base_die_rows = first_die.config.NUM_ROW
        base_die_cols = first_die.config.NUM_COL
        die_width = (base_die_cols - 1) * node_spacing
        die_height = (base_die_rows - 1) * node_spacing

        # 计算Die偏移量和画布大小
        die_offsets, figsize = self._calculate_die_offsets_from_layout(die_layout, die_layout_type, die_width, die_height, dies=dies, config=config, die_rotations=die_rotations)

        # 创建画布
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        # 收集全局带宽范围（过滤掉D2D专用节点）
        all_bandwidths = []
        used_ip_types = set()

        for die_id in dies.keys():
            if die_id in die_ip_bandwidth_data:
                die_data = die_ip_bandwidth_data[die_id]
                if mode in die_data:
                    for ip_type, data_matrix in die_data[mode].items():
                        # 过滤掉 d2d_rn 和 d2d_sn 类型
                        if ip_type.lower() in ["d2d_rn", "d2d_sn"]:
                            continue

                        nonzero_bw = data_matrix[data_matrix > 0.001]
                        if len(nonzero_bw) > 0:
                            all_bandwidths.extend(nonzero_bw.tolist())
                            used_ip_types.add(ip_type.upper().split("_")[0])

        max_bandwidth = max(all_bandwidths) if all_bandwidths else 1.0
        min_bandwidth = min(all_bandwidths) if all_bandwidths else 0.0

        # 为每个Die绘制节点（不绘制链路）
        die_positions = {}
        for die_id, die_model in dies.items():
            if die_id not in die_ip_bandwidth_data:
                continue

            offset_x, offset_y = die_offsets[die_id]
            die_config = die_model.config
            die_rotation = die_rotations.get(die_id, 0)

            # 获取所有节点
            physical_nodes = list(range(die_config.NUM_ROW * die_config.NUM_COL))
            orig_rows = die_config.NUM_ROW
            orig_cols = die_config.NUM_COL

            xs = []
            ys = []
            for node in physical_nodes:
                # 计算节点位置
                orig_row = node // orig_cols
                orig_col = node % orig_cols

                # 应用旋转
                new_row, new_col = self._apply_die_rotation(orig_row, orig_col, orig_rows, orig_cols, die_rotation)

                x = new_col * node_spacing + offset_x
                y = -new_row * node_spacing + offset_y
                xs.append(x)
                ys.append(y)

                # 绘制节点（热力图样式）
                self._draw_heatmap_node(ax, x, y, node, die_id, die_config, die_ip_bandwidth_data, mode, node_size, max_bandwidth, min_bandwidth)

            if xs and ys:
                die_positions[die_id] = {"xs": xs, "ys": ys, "offset_x": offset_x, "offset_y": offset_y}

        # 添加Die标签
        self._add_die_labels_for_heatmap(ax, die_positions, die_layout, die_rotations)

        # 设置标题
        title = f"IP Bandwidth Heatmap - {mode.capitalize()} Mode"
        ax.set_title(title, fontsize=14, fontweight="bold", y=0.96)

        # 添加IP类型图例
        self._add_ip_type_legend(ax, used_ip_types)

        # 添加带宽范围说明
        self._add_bandwidth_range_legend(ax, min_bandwidth, max_bandwidth)

        # 调整坐标轴
        ax.axis("equal")
        ax.margins(0.05)
        ax.axis("off")

        # 保存或显示
        if save_path:
            if os.path.isdir(save_path) or (not save_path.endswith(".png") and not save_path.endswith(".jpg")):
                filename = f"ip_bandwidth_heatmap_{mode}.png"
                save_path = os.path.join(save_path, filename)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
                plt.tight_layout(pad=0.3)
                plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
            plt.close()
            return save_path
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
                plt.tight_layout(pad=0.3)
                plt.show()
            return None

    def _draw_heatmap_node(self, ax, x, y, node, die_id, config, die_ip_bandwidth_data, mode, node_size, max_bandwidth, min_bandwidth):
        """绘制热力图样式的节点（特殊背景色 + IP方块）"""
        from matplotlib.patches import Rectangle
        from collections import defaultdict
        from src.analysis.analyzers import IP_COLOR_MAP, RN_TYPES, SN_TYPES

        # 获取节点物理位置
        physical_col = node % config.NUM_COL
        physical_row = node // config.NUM_COL

        # 收集该节点的IP带宽（过滤掉D2D专用节点）
        active_ips = []
        if die_id in die_ip_bandwidth_data:
            die_data = die_ip_bandwidth_data[die_id]
            if mode in die_data:
                for ip_type, data_matrix in die_data[mode].items():
                    # 过滤掉 d2d_rn 和 d2d_sn 类型
                    if ip_type.lower() in ["d2d_rn", "d2d_sn"]:
                        continue

                    if physical_row < data_matrix.shape[0] and physical_col < data_matrix.shape[1]:
                        bandwidth = data_matrix[physical_row, physical_col]
                        if bandwidth > 0.001:
                            active_ips.append((ip_type, bandwidth))

        # 计算节点框大小
        square_size = (node_size / 1000.0) * 0.3
        node_box_size = square_size * 3.98

        # 绘制节点填充（特殊背景色）
        bg_color = "#FFF9C4" if active_ips else "#F5F5F5"  # 有IP=浅黄，无IP=浅灰
        bg_alpha = 0.3 if active_ips else 1.0

        node_fill = Rectangle(
            (x - node_box_size / 2, y - node_box_size / 2),
            width=node_box_size,
            height=node_box_size,
            facecolor=bg_color,
            edgecolor="none",
            alpha=bg_alpha,
            zorder=1,
        )
        ax.add_patch(node_fill)

        # 绘制节点边框
        node_border = Rectangle(
            (x - node_box_size / 2, y - node_box_size / 2),
            width=node_box_size,
            height=node_box_size,
            facecolor="none",
            edgecolor="black",
            linewidth=0.8,
            zorder=1,
        )
        ax.add_patch(node_border)

        # 如果没有IP，直接返回
        if not active_ips:
            return

        # 按IP类型分组
        ip_type_dict = defaultdict(list)
        for ip_type, bw in active_ips:
            base_type = ip_type.upper().split("_")[0]
            ip_type_dict[base_type].append(bw)

        # RN/SN分类排序
        rn_ips = [(k, v) for k, v in ip_type_dict.items() if k.upper() in RN_TYPES]
        sn_ips = [(k, v) for k, v in ip_type_dict.items() if k.upper() in SN_TYPES]
        other_ips = [(k, v) for k, v in ip_type_dict.items() if k.upper() not in RN_TYPES + SN_TYPES]

        rn_ips.sort(key=lambda x: sum(x[1]), reverse=True)
        sn_ips.sort(key=lambda x: sum(x[1]), reverse=True)
        other_ips.sort(key=lambda x: sum(x[1]), reverse=True)

        sorted_ip_types = []
        sorted_ip_types.extend(rn_ips)
        sorted_ip_types.extend(sn_ips)
        sorted_ip_types.extend(other_ips)

        # 计算网格布局
        num_ip_types = len(sorted_ip_types)
        max_instances = max(len(instances) for instances in ip_type_dict.values())

        available_size = node_box_size * 1.0
        grid_spacing = square_size * 0.1

        ip_block_width = (available_size - (max_instances - 1) * grid_spacing) / max_instances
        ip_block_height = (available_size - (num_ip_types - 1) * grid_spacing) / num_ip_types
        ip_block_size = min(ip_block_width, ip_block_height, square_size * 1.5)

        total_height = num_ip_types * ip_block_size + (num_ip_types - 1) * grid_spacing

        # 绘制IP方块
        row_idx = 0
        for ip_type, bandwidths in sorted_ip_types:
            num_instances = len(bandwidths)
            ip_color = IP_COLOR_MAP.get(ip_type, "#808080")

            row_width = num_instances * ip_block_size + (num_instances - 1) * grid_spacing

            for col_idx, bandwidth in enumerate(bandwidths):
                alpha = self._calculate_bandwidth_alpha(bandwidth, min_bandwidth, max_bandwidth)

                ip_x = x - row_width / 2 + col_idx * (ip_block_size + grid_spacing)
                ip_y = y + total_height / 2 - row_idx * (ip_block_size + grid_spacing)

                # 绘制IP方块
                ip_rect = Rectangle(
                    (ip_x, ip_y - ip_block_size),
                    width=ip_block_size,
                    height=ip_block_size,
                    facecolor=ip_color,
                    edgecolor="black",
                    linewidth=1,
                    alpha=alpha,
                    zorder=3,
                )
                ax.add_patch(ip_rect)

                # 显示带宽数值
                bw_text = f"{bandwidth:.1f}" if bandwidth >= 0.1 else f"{bandwidth:.2f}"
                ax.text(
                    ip_x + ip_block_size / 2,
                    ip_y - ip_block_size / 2,
                    bw_text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="black",
                    zorder=4,
                )

            row_idx += 1

    def _apply_die_rotation(self, orig_row, orig_col, orig_rows, orig_cols, die_rotation):
        """应用Die旋转变换"""
        if die_rotation == 0 or abs(die_rotation) == 360:
            return orig_row, orig_col
        elif abs(die_rotation) == 90 or abs(die_rotation) == -270:
            return orig_col, orig_rows - 1 - orig_row
        elif abs(die_rotation) == 180:
            return orig_rows - 1 - orig_row, orig_cols - 1 - orig_col
        elif abs(die_rotation) == 270 or abs(die_rotation) == -90:
            return orig_cols - 1 - orig_col, orig_row
        else:
            return orig_row, orig_col

    def _add_die_labels_for_heatmap(self, ax, die_positions, die_layout, die_rotations):
        """为热力图添加Die标签"""
        for die_id in die_positions.keys():
            xs = die_positions[die_id]["xs"]
            ys = die_positions[die_id]["ys"]
            die_center_x = (min(xs) + max(xs)) / 2
            die_center_y = (min(ys) + max(ys)) / 2

            # 默认标签位置
            label_x = die_center_x
            label_y = max(ys) + 2.5

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

            ax.text(
                label_x,
                label_y,
                f"Die {die_id}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7, edgecolor="none"),
                rotation=0,
            )

    def _add_ip_type_legend(self, ax, used_ip_types):
        """添加IP类型颜色图例"""
        from matplotlib.patches import Patch
        from src.analysis.analyzers import IP_COLOR_MAP

        legend_elements = []
        for ip_type in sorted(used_ip_types):
            color = IP_COLOR_MAP.get(ip_type, "#808080")
            legend_elements.append(Patch(facecolor=color, edgecolor="black", label=ip_type))

        if legend_elements:
            ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9, title="IP Types")

    def _add_bandwidth_range_legend(self, ax, min_bw, max_bw):
        """添加带宽透明度对应关系colorbar"""
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import LinearSegmentedColormap
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import matplotlib.colors as mcolors
        import numpy as np

        # 如果范围为0，不显示
        if max_bw <= min_bw:
            return

        # 创建插入的colorbar坐标轴，放在右上角IP图例下方
        cax = inset_axes(
            ax,
            width="1.5%",  # colorbar宽度（缩小）
            height="15%",  # colorbar高度（稍微缩小）
            loc="upper right",
            bbox_to_anchor=(-0.05, -0.35, 1, 1),  # 位置：IP图例下方
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        # 创建灰度渐变colormap（从浅到深）
        # alpha值: 0.9(低带宽,浅色) -> 0.3(高带宽,深色)
        colors = ["#E0E0E0", "#B0B0B0", "#808080", "#505050", "#202020"]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list("bandwidth_alpha", colors, N=n_bins)

        # 创建归一化对象
        norm = mcolors.Normalize(vmin=min_bw, vmax=max_bw)

        # 创建colorbar
        cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")

        # 设置colorbar标签
        cb.set_label("IP BW (GB/s)", fontsize=7, labelpad=2)

        # 设置刻度
        cax.tick_params(labelsize=6)
        n_ticks = 4
        tick_values = np.linspace(min_bw, max_bw, n_ticks)
        cb.set_ticks(tick_values)
        cb.set_ticklabels([f"{v:.1f}" for v in tick_values])


