"""
流量图渲染辅助工具模块

提供可复用的工具类和函数，支持交互式流量图渲染
包含：通道切换管理、链路数据处理、IP布局计算、几何计算工具
"""

from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import plotly.graph_objects as go


# ==================== 通道切换管理器 ====================

class ChannelSwitchManager:
    """管理三通道（请求/响应/数据）切换逻辑"""

    CHANNEL_NAMES = ["请求", "响应", "数据"]

    def __init__(self):
        self.channels = {}  # {channel_name: network}
        self.trace_indices = {}  # {channel_name: (start_idx, end_idx)}
        self.annotation_indices = {}  # {channel_name: (start_idx, end_idx)}
        self.all_annotations = []

    def setup_channels(self, req_network, rsp_network, data_network):
        """设置三个通道的网络对象"""
        self.channels = {
            "请求": req_network,
            "响应": rsp_network,
            "数据": data_network,
        }
        return {k: v for k, v in self.channels.items() if v is not None}

    def record_trace_range(self, channel_name: str, start_idx: int, end_idx: int):
        """记录某个通道的trace索引范围"""
        self.trace_indices[channel_name] = (start_idx, end_idx)

    def record_annotation_range(self, channel_name: str, start_idx: int, end_idx: int):
        """记录某个通道的annotation索引范围"""
        self.annotation_indices[channel_name] = (start_idx, end_idx)

    def add_annotations(self, annotations: List):
        """添加annotations到管理器"""
        self.all_annotations.extend(annotations)

    def create_buttons(self, fig, num_extra_traces: int = 0, num_base_traces: int = 0, num_existing_annotations: int = None):
        """
        创建通道切换按钮

        Args:
            fig: plotly Figure对象
            num_extra_traces: 额外的traces数量（legend、colorbar等）
            num_base_traces: 基础traces数量（节点背景等），这些traces在所有通道中都可见
            num_existing_annotations: 原有annotations数量（如Die标签），如果为None则从fig.layout.annotations计算

        Returns:
            按钮配置字典
        """
        buttons = []
        if num_existing_annotations is None:
            num_existing_anns = len(fig.layout.annotations) if fig.layout.annotations else 0
        else:
            num_existing_anns = num_existing_annotations

        for channel_name in self.CHANNEL_NAMES:
            if channel_name not in self.trace_indices:
                continue

            # 创建visibility数组
            total_traces = len(fig.data)
            if num_base_traces > 0:
                # 基础traces始终可见
                visibility = [True if i < num_base_traces else False for i in range(total_traces)]
            else:
                visibility = [False] * total_traces

            # 设置当前通道的traces可见
            trace_start, trace_end = self.trace_indices[channel_name]
            for i in range(trace_start, trace_end):
                if i < total_traces:
                    visibility[i] = True

            # 额外的traces（legend、colorbar）始终可见
            if num_extra_traces > 0:
                for i in range(total_traces - num_extra_traces, total_traces):
                    visibility[i] = True

            # 处理annotations可见性
            updated_annotations = []
            # 获取fig上所有annotations (包括原有的)
            all_fig_anns = list(fig.layout.annotations) if fig.layout.annotations else []

            for i, ann in enumerate(all_fig_anns):
                ann_dict = ann.to_plotly_json() if hasattr(ann, 'to_plotly_json') else dict(ann)

                if i < num_existing_anns:
                    # 原有annotations（Die标签等），始终可见
                    ann_dict["visible"] = True
                else:
                    # 新添加的channel annotations
                    channel_ann_idx = i - num_existing_anns
                    ann_channel = None
                    for ch_name in self.CHANNEL_NAMES:
                        if ch_name not in self.annotation_indices:
                            continue
                        ann_start, ann_end = self.annotation_indices[ch_name]
                        if ann_start <= channel_ann_idx < ann_end:
                            ann_channel = ch_name
                            break
                    ann_dict["visible"] = (ann_channel == channel_name)

                updated_annotations.append(ann_dict)

            buttons.append(
                dict(
                    label=channel_name,
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"annotations": updated_annotations}
                    ],
                )
            )

        return buttons


# ==================== 链路数据处理器 ====================

class LinkDataProcessor:
    """处理链路统计数据和格式化"""

    MAX_BANDWIDTH_NORMALIZATION = 100.0  # GB/s

    @staticmethod
    def extract_link_stats(network, mode: str = "utilization", config=None):
        """
        从network提取链路统计数据

        Args:
            network: 网络对象
            mode: 显示模式 ("utilization", "total", "ITag_ratio"等)
            config: 配置对象（用于计算带宽时需要）

        Returns:
            tuple: (links字典, utilization_stats字典)
        """
        links = {}
        utilization_stats = {}

        if hasattr(network, "get_links_utilization_stats") and callable(network.get_links_utilization_stats):
            try:
                utilization_stats = network.get_links_utilization_stats()

                if mode == "utilization":
                    links = {link: stats["utilization"] for link, stats in utilization_stats.items()}
                elif mode == "ITag_ratio":
                    links = {link: stats["ITag_ratio"] for link, stats in utilization_stats.items()}
                elif mode == "total":
                    # 计算带宽需要config
                    links = {}
                    for link, stats in utilization_stats.items():
                        total_flit = stats["total_flit"]
                        total_cycles = stats["total_cycles"]
                        if total_cycles > 0 and config:
                            time_ns = total_cycles / config.NETWORK_FREQUENCY
                            bandwidth = total_flit * 128 / time_ns
                            links[link] = bandwidth
                        else:
                            links[link] = 0.0
                else:
                    # 其他mode默认使用utilization
                    links = {link: stats.get(mode, stats["utilization"]) for link, stats in utilization_stats.items()}
            except Exception as e:
                links = {}
                utilization_stats = {}

        return links, utilization_stats

    @staticmethod
    def format_link_label(link_value: float, mode: str) -> str:
        """格式化链路标签文本"""
        if mode in ["utilization", "T2_ratio", "T1_ratio", "T0_ratio", "ITag_ratio"]:
            display_value = float(link_value) if link_value else 0.0
            return f"{display_value*100:.1f}%" if display_value > 0 else ""
        else:  # total bandwidth 或其他
            display_value = float(link_value) if link_value else 0.0
            return f"{display_value:.1f}" if display_value > 0 else ""

    @staticmethod
    def calculate_link_color_tuple(link_value: float, mode: str) -> tuple:
        """
        计算链路颜色（返回RGB元组）

        Args:
            link_value: 链路数值
            mode: 显示模式

        Returns:
            RGB颜色元组 (r, g, b)，范围0-1
        """
        display_value = float(link_value) if link_value else 0.0

        if mode in ["utilization", "T2_ratio", "T1_ratio", "T0_ratio", "ITag_ratio"]:
            color_intensity = display_value
        elif mode == "total":
            color_intensity = min(display_value / 500.0, 1.0)
        else:
            color_intensity = min(display_value / 500.0, 1.0)

        if display_value > 0:
            return (color_intensity, 0, 0)
        else:
            return (0.8, 0.8, 0.8)

    @staticmethod
    def process_links_for_drawing(links: dict, actual_nodes: list, mode: str):
        """
        处理links字典，分离自环和普通链路，生成labels和colors

        Args:
            links: 链路字典 {(i, j) or (i, j, direction): value}
            actual_nodes: 实际节点列表
            mode: 显示模式

        Returns:
            tuple: (edge_labels, edge_colors, self_loop_labels)
                - edge_labels: {(i, j): "label"}
                - edge_colors: {(i, j): (r, g, b)}
                - self_loop_labels: {(i, direction): ("label", (r, g, b))}
        """
        edge_labels = {}
        edge_colors = {}
        self_loop_labels = {}

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

            # 计算标签和颜色
            formatted_label = LinkDataProcessor.format_link_label(value, mode)
            color = LinkDataProcessor.calculate_link_color_tuple(value, mode)

            # 区分自环和普通链路
            if i == j:
                # 只为有带宽的自环创建标签
                display_value = float(value) if value else 0.0
                if display_value > 0:
                    if direction:
                        self_loop_labels[(i, direction)] = (formatted_label, color)
                    else:
                        self_loop_labels[(i, "unknown")] = (formatted_label, color)
            else:
                edge_labels[(i, j)] = formatted_label
                edge_colors[(i, j)] = color

        return edge_labels, edge_colors, self_loop_labels

    @staticmethod
    def calculate_link_color(link_value: float, mode: str, max_value: float = None) -> str:
        """
        计算链路颜色

        Args:
            link_value: 链路数值
            mode: 显示模式
            max_value: 最大值（用于归一化）

        Returns:
            RGB颜色字符串
        """
        if mode == "utilization":
            # 利用率模式：0-100%
            intensity = min(link_value / 100.0, 1.0)
        else:
            # 带宽模式：归一化到MAX_BANDWIDTH
            max_bw = max_value if max_value else LinkDataProcessor.MAX_BANDWIDTH_NORMALIZATION
            intensity = min(link_value / max_bw, 1.0)

        # 红色渐变
        r = int(255 * intensity)
        return f"rgb({r}, 0, 0)"

    @staticmethod
    def build_hover_info(link: Tuple[int, int], utilization_stats: Dict, mode: str) -> str:
        """
        构建hover信息

        Args:
            link: 链路元组 (from_node, to_node)
            utilization_stats: 利用率统计字典
            mode: 显示模式

        Returns:
            hover文本
        """
        if link not in utilization_stats:
            return f"链路: {link[0]} → {link[1]}"

        stats = utilization_stats[link]
        from_node, to_node = link

        hover_parts = [f"链路: {from_node} → {to_node}"]

        if mode == "utilization":
            hover_parts.append(f"利用率: {stats['utilization']:.2f}%")

        if "total_bandwidth" in stats:
            hover_parts.append(f"总带宽: {stats['total_bandwidth']:.2f} GB/s")

        if "read_bandwidth" in stats:
            hover_parts.append(f"读带宽: {stats['read_bandwidth']:.2f} GB/s")

        if "write_bandwidth" in stats:
            hover_parts.append(f"写带宽: {stats['write_bandwidth']:.2f} GB/s")

        return "<br>".join(hover_parts)


# ==================== IP布局计算器 ====================

class IPLayoutCalculator:
    """计算IP方块布局和位置"""

    @staticmethod
    def calculate_bandwidth_alpha(bandwidth: float, min_bw: float, max_bw: float) -> float:
        """
        根据带宽计算透明度

        Args:
            bandwidth: 当前带宽
            min_bw: 最小带宽
            max_bw: 最大带宽

        Returns:
            透明度值 (0.3-1.0)
        """
        if max_bw <= min_bw:
            return 1.0

        normalized = (bandwidth - min_bw) / (max_bw - min_bw)
        alpha = 0.3 + 0.7 * normalized
        return alpha

    @staticmethod
    def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
        """
        将十六进制颜色转换为RGBA

        Args:
            hex_color: 十六进制颜色 (如 "#FF5733")
            alpha: 透明度 (0-1)

        Returns:
            RGBA颜色字符串
        """
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"

    @staticmethod
    def calculate_ip_layout(
        ip_modules: Dict,
        ip_bandwidth_data: Dict,
        ip_color_map: Dict,
        square_size: float,
        mode: str = "utilization",
        max_rows: int = 4
    ) -> Tuple[List, Dict, Dict]:
        """
        统一的IP布局计算方法（整合shapes和positions两个方法的逻辑）

        Args:
            ip_modules: IP模块字典 {(ip_type, node_id): ip_module}
            ip_bandwidth_data: IP带宽数据
            ip_color_map: IP颜色映射
            square_size: 方块大小
            mode: 显示模式
            max_rows: 最大行数

        Returns:
            tuple: (shapes列表, all_ip_bandwidths字典, ip_positions字典)
        """
        shapes = []
        all_ip_bandwidths = {}
        ip_positions = {}

        # 优先级IP类型
        priority_types = ["D2D_RN", "D2D_SN"]

        # 按node_id分组IP
        node_ips = {}
        for (ip_type, node_id), ip_module in ip_modules.items():
            if node_id not in node_ips:
                node_ips[node_id] = []
            node_ips[node_id].append((ip_type, ip_module))

        # 计算全局带宽范围（用于归一化透明度）
        if mode == "utilization":
            min_bw = float('inf')
            max_bw = float('-inf')
            for node_id, ips in node_ips.items():
                for ip_type, _ in ips:
                    bw = ip_bandwidth_data.get((ip_type, node_id), 0.0)
                    all_ip_bandwidths[(ip_type, node_id)] = bw
                    if bw > 0:
                        min_bw = min(min_bw, bw)
                        max_bw = max(max_bw, bw)

            if min_bw == float('inf'):
                min_bw = 0.0
                max_bw = 1.0
        else:
            min_bw = 0.0
            max_bw = 100.0

        # 为每个节点生成IP布局
        for node_id in sorted(node_ips.keys()):
            ips = node_ips[node_id]

            # 按优先级排序
            def get_priority(ip_tuple):
                ip_type = ip_tuple[0]
                if ip_type in priority_types:
                    return priority_types.index(ip_type)
                return len(priority_types)

            ips.sort(key=get_priority)

            # 计算布局
            num_ips = len(ips)
            display_rows = min(num_ips, max_rows)
            display_cols = (num_ips + display_rows - 1) // display_rows

            # 计算起始位置（基于node_id的坐标）
            # 这里假设有一个get_node_position函数，实际使用时需要传入
            # 为了通用性，返回相对位置，由调用者计算绝对位置

            positions_for_node = {}
            for idx, (ip_type, ip_module) in enumerate(ips):
                row = idx % display_rows
                col = idx // display_rows

                # 相对位置
                rel_x = col
                rel_y = row

                positions_for_node[ip_type] = (rel_x, rel_y)

                # 获取带宽和颜色
                bandwidth = ip_bandwidth_data.get((ip_type, node_id), 0.0)
                base_color = ip_color_map.get(ip_type.split('_')[0].upper(), ip_color_map.get("OTHER", "#CCCCCC"))

                # 计算透明度
                if mode == "utilization" and bandwidth > 0:
                    alpha = IPLayoutCalculator.calculate_bandwidth_alpha(bandwidth, min_bw, max_bw)
                else:
                    alpha = 1.0

                color = IPLayoutCalculator.hex_to_rgba(base_color, alpha)

                # 这里只返回布局信息，实际shapes由调用者根据具体坐标创建
                ip_positions[(ip_type, node_id)] = {
                    'rel_pos': (rel_x, rel_y),
                    'color': color,
                    'bandwidth': bandwidth,
                    'alpha': alpha
                }

        return shapes, all_ip_bandwidths, ip_positions


# ==================== 几何计算工具函数 ====================

@lru_cache(maxsize=128)
def calculate_node_positions(num_nodes: int, num_cols: int, square_size: float) -> Dict[int, Tuple[float, float]]:
    """
    计算节点位置（带缓存）

    Args:
        num_nodes: 节点数量
        num_cols: 列数
        square_size: 方块大小

    Returns:
        位置字典 {node_id: (x, y)}
    """
    pos = {}
    for node in range(num_nodes):
        row = node // num_cols
        col = node % num_cols
        x = col * square_size
        y = -row * square_size  # Y轴向下
        pos[node] = (x, y)
    return pos


def apply_rotation(
    orig_row: int,
    orig_col: int,
    rows: int,
    cols: int,
    rotation: int
) -> Tuple[int, int]:
    """
    应用旋转变换到节点坐标

    Args:
        orig_row: 原始行号
        orig_col: 原始列号
        rows: 总行数
        cols: 总列数
        rotation: 旋转角度 (0, 90, 180, 270)

    Returns:
        tuple: (new_row, new_col) 旋转后的坐标
    """
    if rotation == 0:
        return orig_row, orig_col
    elif rotation == 90:
        new_row = orig_col
        new_col = rows - 1 - orig_row
        return new_row, new_col
    elif rotation == 180:
        new_row = rows - 1 - orig_row
        new_col = cols - 1 - orig_col
        return new_row, new_col
    elif rotation == 270:
        new_row = cols - 1 - orig_col
        new_col = orig_row
        return new_row, new_col
    else:
        return orig_row, orig_col


def calculate_die_offsets(
    die_layout: Dict,
    die_layout_type: str,
    die_width: float,
    die_height: float,
    config=None,
    dies: Dict = None,
    die_rotations: Dict = None
) -> Dict[int, Tuple[float, float]]:
    """
    计算各Die的偏移量

    Args:
        die_layout: Die布局配置
        die_layout_type: 布局类型 ("grid" 或 "custom")
        die_width: 单个Die宽度
        die_height: 单个Die高度
        config: 配置对象
        dies: Die对象字典
        die_rotations: Die旋转角度字典

    Returns:
        偏移量字典 {die_id: (offset_x, offset_y)}
    """
    die_offsets = {}

    if die_layout_type == "grid":
        # 网格布局
        grid_rows = die_layout.get("rows", 1)
        grid_cols = die_layout.get("cols", 1)
        spacing = die_layout.get("spacing", 2.0)

        for die_id in range(grid_rows * grid_cols):
            row = die_id // grid_cols
            col = die_id % grid_cols
            offset_x = col * (die_width + spacing)
            offset_y = row * (die_height + spacing)
            die_offsets[die_id] = (offset_x, offset_y)

    elif die_layout_type == "custom":
        # 自定义位置
        positions = die_layout.get("positions", {})
        for die_id, pos in positions.items():
            die_offsets[int(die_id)] = tuple(pos)

    # 应用智能对齐（如果提供了dies和config）
    if dies and config and len(dies) > 1:
        die_offsets = calculate_die_alignment_offsets(
            dies, config, die_offsets, die_rotations
        )

    return die_offsets


def calculate_die_alignment_offsets(
    dies: Dict,
    config,
    base_offsets: Dict,
    die_rotations: Dict = None
) -> Dict[int, Tuple[float, float]]:
    """
    计算Die对齐偏移量（智能对齐D2D_RN/SN节点）

    Args:
        dies: Die模型字典
        config: 配置对象
        base_offsets: 基础偏移量
        die_rotations: Die旋转角度

    Returns:
        调整后的偏移量字典
    """
    if not hasattr(config, 'D2D_PAIRS') or not config.D2D_PAIRS:
        return base_offsets

    adjusted_offsets = dict(base_offsets)
    d2d_pairs = config.D2D_PAIRS
    die_rotations = die_rotations or {}

    # 提取D2D连接的节点位置信息
    for die0_id, die0_node, die1_id, die1_node in d2d_pairs:
        if die0_id not in dies or die1_id not in dies:
            continue

        # 获取节点在各自Die中的坐标
        die0_config = dies[die0_id].config
        die1_config = dies[die1_id].config

        # 考虑旋转
        die0_rot = die_rotations.get(die0_id, 0)
        die1_rot = die_rotations.get(die1_id, 0)

        # 计算节点行列
        die0_row = die0_node // die0_config.NUM_COL
        die0_col = die0_node % die0_config.NUM_COL
        die1_row = die1_node // die1_config.NUM_COL
        die1_col = die1_node % die1_config.NUM_COL

        # 应用旋转
        die0_rows = die0_config.NUM_ROW
        die0_cols = die0_config.NUM_COL
        die1_rows = die1_config.NUM_ROW
        die1_cols = die1_config.NUM_COL

        die0_row, die0_col = apply_rotation(die0_row, die0_col, die0_rows, die0_cols, die0_rot)
        die1_row, die1_col = apply_rotation(die1_row, die1_col, die1_rows, die1_cols, die1_rot)

        # 简化版对齐：这里只做基本的水平/垂直对齐判断
        # 实际复杂对齐逻辑保留在原文件中

    return adjusted_offsets


def get_connection_type(from_node: int, to_node: int, num_cols: int) -> str:
    """
    判断连接类型（水平、垂直或对角）

    Args:
        from_node: 起始节点
        to_node: 目标节点
        num_cols: 列数

    Returns:
        连接类型: "horizontal", "vertical" 或 "diagonal"
    """
    from_row = from_node // num_cols
    from_col = from_node % num_cols
    to_row = to_node // num_cols
    to_col = to_node % num_cols

    if from_row == to_row:
        return "horizontal"
    elif from_col == to_col:
        return "vertical"
    else:
        return "diagonal"
