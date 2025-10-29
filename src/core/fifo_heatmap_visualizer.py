"""
FIFO使用率热力图可视化模块

提供FIFO使用率数据收集和交互式热力图可视化功能
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class FIFOUtilizationCollector:
    """FIFO使用率数据收集器"""

    def __init__(self, config):
        """
        初始化FIFO使用率收集器

        Args:
            config: 配置对象
        """
        self.config = config
        self.fifo_utilization_data = {}

    def collect_from_network(self, network, die_id: int, total_cycles: int) -> Dict:
        """
        从单个Network对象收集FIFO使用率数据

        Args:
            network: Network对象
            die_id: Die ID
            total_cycles: 仿真总周期数

        Returns:
            Dict: FIFO使用率数据
                {
                    'IQ': {node_pos: {'TL': {'avg': 0.5, 'peak': 0.8, 'capacity': 8}, ...}},
                    'RB': {node_pos: {'TL': {'avg': 0.3, 'peak': 0.6, 'capacity': 16}, ...}},
                    'EQ': {node_pos: {'TU': {'avg': 0.4, 'peak': 0.7, 'capacity': 8}, ...}},
                    'IQ_CH': {node_pos: {'gdma': {'avg': 0.2, 'peak': 0.5, 'capacity': 4}, ...}},
                    'EQ_CH': {node_pos: {'ddr': {'avg': 0.3, 'peak': 0.6, 'capacity': 4}, ...}}
                }
        """
        die_data = {"IQ": {}, "RB": {}, "EQ": {}, "IQ_CH": {}, "EQ_CH": {}}

        if total_cycles <= 0:
            print(f"警告: Die {die_id} 总周期数无效: {total_cycles}")
            return die_data

        # 收集IQ方向队列数据 (inject_queues)
        for direction in ["TL", "TR", "TU", "TD", "EQ"]:
            for node_pos, depth_sum in network.fifo_depth_sum["IQ"].get(direction, {}).items():
                if node_pos not in die_data["IQ"]:
                    die_data["IQ"][node_pos] = {}

                max_depth = network.fifo_max_depth["IQ"][direction].get(node_pos, 0)
                capacity = self._get_fifo_capacity("IQ", direction, node_pos)

                avg_util = (depth_sum / total_cycles / capacity * 100) if capacity > 0 else 0
                peak_util = (max_depth / capacity * 100) if capacity > 0 else 0

                die_data["IQ"][node_pos][direction] = {
                    "avg": avg_util,
                    "peak": peak_util,
                    "capacity": capacity,
                    "avg_depth": depth_sum / total_cycles if total_cycles > 0 else 0,
                    "peak_depth": max_depth,
                }

        # 收集IQ通道缓冲数据 (IQ_channel_buffer)
        for node_pos, ip_types in network.fifo_depth_sum["IQ"].get("CH_buffer", {}).items():
            if node_pos not in die_data["IQ_CH"]:
                die_data["IQ_CH"][node_pos] = {}

            for ip_type, depth_sum in ip_types.items():
                max_depth = network.fifo_max_depth["IQ"]["CH_buffer"][node_pos].get(ip_type, 0)
                capacity = self._get_fifo_capacity("IQ_CH", ip_type, node_pos)

                avg_util = (depth_sum / total_cycles / capacity * 100) if capacity > 0 else 0
                peak_util = (max_depth / capacity * 100) if capacity > 0 else 0

                die_data["IQ_CH"][node_pos][ip_type] = {
                    "avg": avg_util,
                    "peak": peak_util,
                    "capacity": capacity,
                    "avg_depth": depth_sum / total_cycles if total_cycles > 0 else 0,
                    "peak_depth": max_depth,
                }

        # 收集RB数据 (ring_bridge)
        for direction in ["TL", "TR", "TU", "TD", "EQ"]:
            for node_pos, depth_sum in network.fifo_depth_sum["RB"].get(direction, {}).items():
                if node_pos not in die_data["RB"]:
                    die_data["RB"][node_pos] = {}

                max_depth = network.fifo_max_depth["RB"][direction].get(node_pos, 0)
                capacity = self._get_fifo_capacity("RB", direction, node_pos)

                avg_util = (depth_sum / total_cycles / capacity * 100) if capacity > 0 else 0
                peak_util = (max_depth / capacity * 100) if capacity > 0 else 0

                die_data["RB"][node_pos][direction] = {
                    "avg": avg_util,
                    "peak": peak_util,
                    "capacity": capacity,
                    "avg_depth": depth_sum / total_cycles if total_cycles > 0 else 0,
                    "peak_depth": max_depth,
                }

        # 收集EQ下环队列数据 (eject_queues)
        for direction in ["TU", "TD"]:
            for node_pos, depth_sum in network.fifo_depth_sum["EQ"].get(direction, {}).items():
                if node_pos not in die_data["EQ"]:
                    die_data["EQ"][node_pos] = {}

                max_depth = network.fifo_max_depth["EQ"][direction].get(node_pos, 0)
                capacity = self._get_fifo_capacity("EQ", direction, node_pos)

                avg_util = (depth_sum / total_cycles / capacity * 100) if capacity > 0 else 0
                peak_util = (max_depth / capacity * 100) if capacity > 0 else 0

                die_data["EQ"][node_pos][direction] = {
                    "avg": avg_util,
                    "peak": peak_util,
                    "capacity": capacity,
                    "avg_depth": depth_sum / total_cycles if total_cycles > 0 else 0,
                    "peak_depth": max_depth,
                }

        # 收集EQ通道缓冲数据 (EQ_channel_buffer)
        for node_pos, ip_types in network.fifo_depth_sum["EQ"].get("CH_buffer", {}).items():
            if node_pos not in die_data["EQ_CH"]:
                die_data["EQ_CH"][node_pos] = {}

            for ip_type, depth_sum in ip_types.items():
                max_depth = network.fifo_max_depth["EQ"]["CH_buffer"][node_pos].get(ip_type, 0)
                capacity = self._get_fifo_capacity("EQ_CH", ip_type, node_pos)

                avg_util = (depth_sum / total_cycles / capacity * 100) if capacity > 0 else 0
                peak_util = (max_depth / capacity * 100) if capacity > 0 else 0

                die_data["EQ_CH"][node_pos][ip_type] = {
                    "avg": avg_util,
                    "peak": peak_util,
                    "capacity": capacity,
                    "avg_depth": depth_sum / total_cycles if total_cycles > 0 else 0,
                    "peak_depth": max_depth,
                }

        return die_data

    def collect_from_dies(self, dies: Dict, total_cycles: int) -> Dict:
        """
        从所有Die收集FIFO使用率数据

        Args:
            dies: Die字典 {die_id: die_model}
            total_cycles: 仿真总周期数

        Returns:
            Dict: {die_id: die_data}
        """
        self.fifo_utilization_data = {}

        for die_id, die_model in dies.items():
            # 统一使用data_network来统计FIFO使用率
            # 因为数据网络承载最大流量，最能反映FIFO压力
            if hasattr(die_model, "data_network"):
                network = die_model.data_network
            else:
                print(f"警告: Die {die_id} 没有data_network属性")
                continue

            self.fifo_utilization_data[die_id] = self.collect_from_network(network, die_id, total_cycles)

        return self.fifo_utilization_data

    def _get_fifo_capacity(self, fifo_category: str, fifo_type: str, node_pos: int) -> int:
        """
        获取FIFO容量

        Args:
            fifo_category: FIFO类别 ('IQ', 'RB', 'EQ', 'IQ_CH', 'EQ_CH')
            fifo_type: FIFO类型 (如'TL', 'TR', 'gdma'等)
            node_pos: 节点位置

        Returns:
            int: FIFO容量
        """
        config = self.config

        if fifo_category == "IQ":
            if fifo_type in ["TL", "TR"]:
                return getattr(config, "IQ_OUT_FIFO_DEPTH_HORIZONTAL", 8)
            elif fifo_type in ["TU", "TD"]:
                return getattr(config, "IQ_OUT_FIFO_DEPTH_VERTICAL", 8)
            elif fifo_type == "EQ":
                return getattr(config, "IQ_OUT_FIFO_DEPTH_EQ", 8)

        elif fifo_category == "IQ_CH":
            return getattr(config, "IQ_CH_FIFO_DEPTH", 4)

        elif fifo_category == "RB":
            return getattr(config, "RB_IN_FIFO_DEPTH", 16)

        elif fifo_category == "EQ":
            return getattr(config, "EQ_IN_FIFO_DEPTH", 8)

        elif fifo_category == "EQ_CH":
            return getattr(config, "IQ_CH_FIFO_DEPTH", 4)  # EQ_CH和IQ_CH容量相同

        return 0


class FIFOHeatmapVisualizer:
    """FIFO使用率交互式热力图可视化器"""

    def __init__(self, config, fifo_data: Dict):
        """
        初始化可视化器

        Args:
            config: 配置对象
            fifo_data: FIFO使用率数据 {die_id: die_data}
        """
        self.config = config
        self.fifo_data = fifo_data

    def create_interactive_heatmap(self, dies: Dict, die_layout: Optional[Dict] = None, die_rotations: Optional[Dict] = None, save_path: Optional[str] = None) -> Optional[str]:
        """
        创建交互式FIFO使用率热力图

        Args:
            dies: Die字典
            die_layout: Die布局 {die_id: (grid_x, grid_y)}
            die_rotations: Die旋转角度 {die_id: rotation}
            save_path: HTML文件保存路径

        Returns:
            str: 保存路径（如果提供）
        """
        if not self.fifo_data:
            print("警告: 没有FIFO数据可供可视化")
            return None

        # 获取所有可用的FIFO类型选项
        fifo_options = self._get_available_fifo_types()

        if not fifo_options:
            print("警告: 没有找到可用的FIFO类型")
            return None

        # 创建图形
        fig = self._create_plotly_figure(dies, die_layout, die_rotations, fifo_options)

        # 保存或显示
        if save_path:
            # 生成HTML并注入JavaScript交互代码
            self._save_html_with_click_events(fig, save_path, fifo_options, len(dies))
            print(f"FIFO使用率热力图已保存到: {save_path}")
            return save_path
        else:
            fig.show()
            return None

    def _get_available_fifo_types(self) -> List[Tuple[str, str, str]]:
        """
        获取所有可用的FIFO类型

        Returns:
            List[Tuple]: [(显示名称, fifo_category, fifo_type), ...]
        """
        options = []

        for die_id, die_data in self.fifo_data.items():
            # IQ方向队列
            for node_pos, directions in die_data.get("IQ", {}).items():
                for direction in directions.keys():
                    option_name = f"IQ-{direction}"
                    if (option_name, "IQ", direction) not in options:
                        options.append((option_name, "IQ", direction))

            # IQ通道缓冲
            for node_pos, ip_types in die_data.get("IQ_CH", {}).items():
                for ip_type in ip_types.keys():
                    option_name = f"IQ_CH-{ip_type}"
                    if (option_name, "IQ_CH", ip_type) not in options:
                        options.append((option_name, "IQ_CH", ip_type))

            # RB
            for node_pos, directions in die_data.get("RB", {}).items():
                for direction in directions.keys():
                    option_name = f"RB-{direction}"
                    if (option_name, "RB", direction) not in options:
                        options.append((option_name, "RB", direction))

            # EQ下环队列
            for node_pos, directions in die_data.get("EQ", {}).items():
                for direction in directions.keys():
                    option_name = f"EQ-{direction}"
                    if (option_name, "EQ", direction) not in options:
                        options.append((option_name, "EQ", direction))

            # EQ通道缓冲
            for node_pos, ip_types in die_data.get("EQ_CH", {}).items():
                for ip_type in ip_types.keys():
                    option_name = f"EQ_CH-{ip_type}"
                    if (option_name, "EQ_CH", ip_type) not in options:
                        options.append((option_name, "EQ_CH", ip_type))

        # 排序：IQ -> RB -> EQ -> IQ_CH -> EQ_CH
        def sort_key(item):
            name, category, _ = item
            category_order = {"IQ": 0, "RB": 1, "EQ": 2, "IQ_CH": 3, "EQ_CH": 4}
            return (category_order.get(category, 999), name)

        options.sort(key=sort_key)
        return options

    def _create_plotly_figure(self, dies: Dict, die_layout: Optional[Dict], die_rotations: Optional[Dict], fifo_options: List[Tuple[str, str, str]]) -> go.Figure:
        """
        创建Plotly交互式图形 - 左侧热力图 + 右侧架构图

        Args:
            dies: Die字典
            die_layout: Die布局
            die_rotations: Die旋转
            fifo_options: FIFO选项列表

        Returns:
            go.Figure: Plotly图形对象
        """
        # 准备Die布局和旋转信息
        if die_layout is None:
            die_layout = {die_id: (die_id, 0) for die_id in dies.keys()}
        if die_rotations is None:
            die_rotations = {die_id: 0 for die_id in dies.keys()}

        # 创建子图布局: 1行2列 (50% + 50%)
        fig = make_subplots(
            rows=1, cols=2, column_widths=[0.5, 0.5], subplot_titles=("FIFO使用率热力图", "CrossRing架构图"), specs=[[{"type": "heatmap"}, {"type": "scatter"}]], horizontal_spacing=0.08
        )

        # 为每个FIFO类型和统计模式创建热力图trace
        traces_data = self._create_heatmap_traces(fifo_options, dies)

        # 将所有traces添加到左侧子图（初始时只显示第一个选项的峰值模式）
        default_option = fifo_options[0]
        default_mode = "peak"

        for option in fifo_options:
            for mode in ["avg", "peak"]:
                traces = traces_data[(option, mode)]
                for trace in traces:
                    # 只有默认选项的默认模式可见
                    trace.visible = option == default_option and mode == default_mode
                    fig.add_trace(trace, row=1, col=1)

        # 添加右侧架构图
        self._add_architecture_diagram(fig, fifo_options, dies)

        # 添加平均/峰值按钮
        self._add_mode_buttons(fig, fifo_options, traces_data)

        # 设置布局
        fig.update_layout(
            title=dict(text="FIFO使用率热力图", y=0.97, x=0.5, xanchor="center", yanchor="top"),
            hovermode="closest",
            width=1800,
            height=900,
            plot_bgcolor="white",
            margin=dict(t=150, b=50, l=80, r=80),  # 增加顶部边距
            showlegend=False,
        )

        # 隐藏左侧热力图的坐标轴，设置正方形比例
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, title="", row=1, col=1)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, title="", autorange="reversed", scaleanchor="x", scaleratio=1, row=1, col=1)

        # 隐藏右侧架构图的坐标轴
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, title="", row=1, col=2)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, title="", row=1, col=2)

        return fig

    def _calculate_node_positions(self, dies: Dict, die_layout: Dict, die_rotations: Dict) -> Dict:
        """
        计算所有Die的节点位置

        Returns:
            Dict: {die_id: {node_id: (x, y)}}
        """
        node_spacing = 3.0
        die_spacing = 5.0

        positions = {}

        for die_id, die_model in dies.items():
            die_config = die_model.config
            rows = die_config.NUM_ROW
            cols = die_config.NUM_COL
            rotation = die_rotations.get(die_id, 0)

            # 计算Die偏移
            grid_x, grid_y = die_layout.get(die_id, (die_id, 0))

            # 根据旋转计算Die实际尺寸
            if rotation in [90, 270, -90, -270]:
                die_width = (rows - 1) * node_spacing
                die_height = (cols - 1) * node_spacing
            else:
                die_width = (cols - 1) * node_spacing
                die_height = (rows - 1) * node_spacing

            offset_x = grid_x * (die_width + die_spacing)
            offset_y = -grid_y * (die_height + die_spacing)

            # 计算每个节点的位置
            positions[die_id] = {}
            for node_id in range(rows * cols):
                orig_row = node_id // cols
                orig_col = node_id % cols

                # 应用旋转
                if rotation == 0:
                    new_row, new_col = orig_row, orig_col
                elif rotation in [90, -270]:
                    new_row = orig_col
                    new_col = rows - 1 - orig_row
                elif rotation in [180, -180]:
                    new_row = rows - 1 - orig_row
                    new_col = cols - 1 - orig_col
                elif rotation in [270, -90]:
                    new_row = cols - 1 - orig_col
                    new_col = orig_row
                else:
                    new_row, new_col = orig_row, orig_col

                x = new_col * node_spacing + offset_x
                y = -new_row * node_spacing + offset_y
                positions[die_id][node_id] = (x, y)

        return positions

    def _create_heatmap_traces(self, fifo_options: List[Tuple[str, str, str]], dies: Dict) -> Dict:
        """
        为所有FIFO选项和统计模式创建热力图traces

        Returns:
            Dict: {(option_tuple, mode): [traces]}
        """
        traces_data = {}

        for option in fifo_options:
            option_name, fifo_category, fifo_type = option

            for mode in ["avg", "peak"]:
                traces = []

                # 为每个Die创建heatmap trace
                for die_id in sorted(self.fifo_data.keys()):
                    die_model = dies[die_id]
                    die_config = die_model.config
                    rows = die_config.NUM_ROW
                    cols = die_config.NUM_COL

                    die_data = self.fifo_data[die_id]
                    category_data = die_data.get(fifo_category, {})

                    # 创建使用率矩阵（行x列）
                    utilization_matrix = np.full((rows, cols), np.nan)
                    hover_matrix = [["" for _ in range(cols)] for _ in range(rows)]
                    text_matrix = [["" for _ in range(cols)] for _ in range(rows)]  # 用于显示在方块上的文本

                    for node_id in range(rows * cols):
                        row = node_id // cols
                        col = node_id % cols

                        node_data = category_data.get(node_id, {})
                        fifo_info = node_data.get(fifo_type)

                        if fifo_info is not None:
                            util_value = fifo_info[mode]
                            capacity = fifo_info["capacity"]
                            avg_depth = fifo_info["avg_depth"]
                            peak_depth = fifo_info["peak_depth"]

                            utilization_matrix[row, col] = util_value
                            text_matrix[row][col] = f"{util_value:.1f}%"  # 显示使用率百分比

                            hover_text = (
                                f"Die {die_id} - 节点 {node_id} ({row},{col})<br>"
                                f"FIFO: {option_name}<br>"
                                f"平均: {fifo_info['avg']:.1f}% ({avg_depth:.2f}/{capacity})<br>"
                                f"峰值: {fifo_info['peak']:.1f}% ({peak_depth}/{capacity})<br>"
                                f"容量: {capacity}"
                            )
                            hover_matrix[row][col] = hover_text
                        else:
                            hover_matrix[row][col] = f"Die {die_id} - 节点 {node_id}<br>无数据"
                            text_matrix[row][col] = ""  # 无数据时不显示文本

                    # 创建Heatmap trace
                    trace = go.Heatmap(
                        z=utilization_matrix,
                        x=list(range(cols)),
                        y=list(range(rows)),
                        text=text_matrix,  # 在方块上显示的文本
                        texttemplate="%{text}",  # 文本模板
                        textfont={"size": 16},  # 文本字体大小
                        colorscale="RdYlGn_r",  # 红-黄-绿反转（红色=高使用率）
                        zmin=0,
                        zmax=100,
                        colorbar=dict(title="使用率 (%)", x=-0.15, xanchor="left", len=0.7) if die_id == max(self.fifo_data.keys()) else None,  # 移到左边
                        showscale=die_id == max(self.fifo_data.keys()),
                        hovertext=hover_matrix,
                        hoverinfo="text",
                        name=f"Die {die_id}",
                        xgap=1,  # 网格间隙
                        ygap=1,
                    )
                    traces.append(trace)

                traces_data[(option, mode)] = traces

        return traces_data

    def _add_mode_buttons(self, fig: go.Figure, fifo_options: List[Tuple[str, str, str]], traces_data: Dict):
        """添加平均/峰值切换按钮"""

        # 创建平均/峰值按钮 (初始显示第一个FIFO的对应模式)
        default_option = fifo_options[0]

        # 计算热力图traces总数
        num_heatmap_traces = sum(len(traces_data.get((opt, mode), [])) for opt in fifo_options for mode in ["avg", "peak"])

        # 计算架构图traces总数（当前fig.data中的总数减去热力图traces）
        total_traces = len(fig.data)
        num_architecture_traces = total_traces - num_heatmap_traces

        mode_buttons = []
        for target_mode in ["avg", "peak"]:
            mode_label = "平均使用率" if target_mode == "avg" else "峰值使用率"

            # 按钮不执行任何操作，完全由JavaScript控制
            button = dict(label=mode_label, method="skip")
            mode_buttons.append(button)

        # 添加按钮到顶部（降低位置避免与标题重叠）
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=mode_buttons,
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.5,
                    xanchor="center",
                    y=1.08,  # 降低y值
                    yanchor="top",
                    bgcolor="lightblue",
                    bordercolor="blue",
                    font=dict(size=12),
                    type="buttons",
                )
            ]
        )

    def _add_architecture_diagram(self, fig: go.Figure, fifo_options: List[Tuple[str, str, str]], dies: Dict):
        """
        在右侧子图添加CrossRing架构示意图（参考可视化效果重新设计）

        布局：
        - 左侧：Inject Queue（纵向排列，通道在上，方向在下）
        - 右上：Eject Queue（横向排列通道，右侧纵向TU/TD）
        - 右下：Ring Bridge（左侧纵向TL/TR，右上角TU/TD/EQ）
        """
        # 获取第一个Die的配置
        first_die = list(dies.values())[0]
        config = first_die.config
        ch_names = config.CH_NAME_LIST if hasattr(config, "CH_NAME_LIST") else ["G0", "G1", "S0", "S1", "C0", "C1", "D0", "D1", "L0", "L1"]

        # === Inject Queue (左侧) ===
        iq_x, iq_y = 0, 0
        iq_w, iq_h = 12, 10
        self._draw_module_box(fig, iq_x, iq_y, iq_w, iq_h, "Inject Queue", "lightgreen")

        # 通道缓冲（水平排列在顶部，小长方形）
        ch_start_x = iq_x + 0.8
        ch_y = iq_y + iq_h - 2
        ch_width = 0.7
        ch_height = 1.5
        ch_spacing = 0.9
        for i, ch_name in enumerate(ch_names):
            x_pos = ch_start_x + i * ch_spacing
            self._draw_fifo_item(fig, x_pos, ch_y, ch_width, ch_height, ch_name, ("IQ_CH", ch_name), fifo_options, text_angle=90)

        # 方向队列TL/TR（底部左侧，竖向长方形）
        dir_y = iq_y + 1.5
        dir_spacing = 1.8
        directions = ["TL", "TR"]
        for i, direction in enumerate(directions):
            x_pos = iq_x + 1.5 + i * dir_spacing
            self._draw_fifo_item(fig, x_pos, dir_y, 1.0, 1.8, direction, ("IQ", direction), fifo_options)

        # EQ（右侧最上方，横向长方形）
        eq_x = iq_x + iq_w - 2.5
        eq_y = iq_y + iq_h - 3.5
        self._draw_fifo_item(fig, eq_x, eq_y, 1.8, 1, "EQ", ("IQ", "EQ"), fifo_options)

        # TU/TD（右侧下方，横向长方形）
        other_dirs = ["TU", "TD"]
        for i, direction in enumerate(other_dirs):
            x_pos = iq_x + iq_w - 2.5
            y_pos = iq_y + 2 + i * 2
            self._draw_fifo_item(fig, x_pos, y_pos, 1.8, 1, direction, ("IQ", direction), fifo_options)

        # === Ring Bridge (右下) ===
        rb_x, rb_y = 14, 0
        rb_w, rb_h = 10, 10
        self._draw_module_box(fig, rb_x, rb_y, rb_w, rb_h, "Ring Bridge", "lightyellow")

        # EQ (左上角，长方形)
        self._draw_fifo_item(fig, rb_x + 1, rb_y + rb_h - 2.5, 1, 1.8, "EQ", ("RB", "EQ"), fifo_options)

        # TL, TR (左侧下方水平排列，小长方形）
        for i, direction in enumerate(["TL", "TR"]):
            x_pos = rb_x + 1 + i * 2.2
            y_pos = rb_y + 1.5
            self._draw_fifo_item(fig, x_pos, y_pos, 1, 1.8, direction, ("RB", direction), fifo_options)

        # TU, TD (右上角垂直排列，小长方形)
        rb_right_dirs = ["TU", "TD"]
        for i, direction in enumerate(rb_right_dirs):
            x_pos = rb_x + rb_w - 2.5
            y_pos = rb_y + rb_h - 2.5 - i * 2.2
            self._draw_fifo_item(fig, x_pos, y_pos, 1.8, 1, direction, ("RB", direction), fifo_options)

        # === Eject Queue (右上) ===
        eq_x, eq_y = 14, 12
        eq_w, eq_h = 10, 8
        self._draw_module_box(fig, eq_x, eq_y, eq_w, eq_h, "Eject Queue", "lightcoral")

        # 通道缓冲（垂直排列在左侧，小长方形）
        eq_ch_start_y = eq_y + eq_h - 1.2
        eq_ch_x = eq_x + 1
        eq_ch_spacing = 0.65
        for i, ch_name in enumerate(ch_names):
            y_pos = eq_ch_start_y - i * eq_ch_spacing
            self._draw_fifo_item(fig, eq_ch_x, y_pos, 1.5, 0.5, ch_name, ("EQ_CH", ch_name), fifo_options)

        # TU, TD (右侧纵向，小长方形）
        for i, direction in enumerate(["TU", "TD"]):
            x_pos = eq_x + eq_w - 2.5
            y_pos = eq_y + eq_h - 2.5 - i * 2.2
            self._draw_fifo_item(fig, x_pos, y_pos, 1.8, 1, direction, ("EQ", direction), fifo_options)

        # 设置右侧子图的范围
        fig.update_xaxes(range=[-1, 26], row=1, col=2)
        fig.update_yaxes(range=[-1, 22], row=1, col=2)

    def _draw_module_box(self, fig: go.Figure, x: float, y: float, w: float, h: float, title: str, color: str):
        """绘制模块边框和标题"""
        # 绘制矩形框
        fig.add_shape(type="rect", x0=x, y0=y, x1=x + w, y1=y + h, line=dict(color="black", width=2), fillcolor=color, opacity=0.2, row=1, col=2)

        # 添加标题
        fig.add_annotation(x=x + w / 2, y=y + h + 0.3, text=title, showarrow=False, font=dict(size=14, color="black", family="Arial Black"), row=1, col=2)

    def _draw_fifo_item(self, fig: go.Figure, x: float, y: float, w: float, h: float, label: str, fifo_info: Tuple[str, str], fifo_options: List[Tuple], text_angle: int = 0):
        """
        绘制单个FIFO项（可点击的矩形）

        Args:
            x, y, w, h: 位置和大小
            label: 显示标签
            fifo_info: (category, type) 用于匹配fifo_options
            fifo_options: 所有可用的FIFO选项列表
            text_angle: 文本旋转角度（默认0度）
        """
        category, fifo_type = fifo_info

        # 检查这个FIFO是否在可用选项中
        is_available = any(opt[1] == category and opt[2] == fifo_type for opt in fifo_options)

        if not is_available:
            return

        # 先绘制矩形背景（使用shape）
        fig.add_shape(type="rect", x0=x, y0=y, x1=x + w, y1=y + h, line=dict(color="black", width=1), fillcolor="lightgray", opacity=0.5, row=1, col=2, name=f"{category}_{fifo_type}")

        # 添加文本标签（使用annotation支持旋转）
        fig.add_annotation(
            x=x + w / 2,
            y=y + h / 2,
            text=label,
            showarrow=False,
            font=dict(size=9),
            textangle=text_angle,
            row=1,
            col=2,
        )

        # 添加透明的scatter trace用于点击交互
        fig.add_trace(
            go.Scatter(
                x=[x + w / 2],
                y=[y + h / 2],
                mode="markers",
                marker=dict(size=1, opacity=0),  # 完全透明
                hoverinfo="text",
                hovertext=f"点击查看 {label} 使用率",
                hoverlabel=dict(bgcolor="white", font=dict(color="black")),  # 统一hover标签颜色
                customdata=[[category, fifo_type]],  # 存储FIFO信息用于点击事件
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    def _save_html_with_click_events(self, fig: go.Figure, save_path: str, fifo_options: List[Tuple[str, str, str]], num_dies: int):
        """
        保存HTML文件并注入JavaScript代码来处理FIFO点击事件

        Args:
            fig: Plotly图形对象
            save_path: 保存路径
            fifo_options: FIFO选项列表
            num_dies: Die数量
        """
        # 先生成基础HTML
        html_string = fig.to_html(include_plotlyjs="cdn")

        # 创建FIFO选项映射 (用于JavaScript查找)
        fifo_map = {}
        for idx, (name, category, fifo_type) in enumerate(fifo_options):
            key = f"{category}_{fifo_type}"
            fifo_map[key] = idx

        # 计算trace索引
        # 左侧热力图: len(fifo_options) * 2 (avg+peak) * num_dies 个traces
        num_heatmap_traces = len(fifo_options) * 2 * num_dies

        # 右侧架构图的traces从 num_heatmap_traces 开始

        # 生成JavaScript代码
        js_code = f"""
<script>
    // FIFO选项映射
    const fifoMap = {str(fifo_map).replace("'", '"')};
    const numFifoOptions = {len(fifo_options)};
    const numDies = {num_dies};
    const numHeatmapTraces = {num_heatmap_traces};

    // 当前选中的FIFO和模式
    let currentFifoIndex = 0;  // 默认第一个FIFO
    let currentMode = 'peak';  // 默认峰值模式

    // 等待Plotly加载完成
    document.addEventListener('DOMContentLoaded', function() {{
        setTimeout(function() {{
            const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            if (!plotDiv) return;

            // 监听点击事件
            plotDiv.on('plotly_click', function(data) {{
                // 只处理右侧架构图的点击 (trace index >= numHeatmapTraces)
                const clickedPoint = data.points[0];
                const traceIndex = clickedPoint.curveNumber;

                if (traceIndex >= numHeatmapTraces) {{
                    // 获取自定义数据 (category, fifo_type)
                    const customdata = clickedPoint.customdata;
                    if (customdata && customdata.length >= 2) {{
                        const category = customdata[0];
                        const fifoType = customdata[1];
                        const key = category + '_' + fifoType;

                        // 查找对应的FIFO索引
                        const fifoIndex = fifoMap[key];
                        if (fifoIndex !== undefined) {{
                            currentFifoIndex = fifoIndex;
                            updateHeatmapVisibility();
                        }}
                    }}
                }}
            }});

            // 更新热力图可见性和架构图高亮
            function updateHeatmapVisibility() {{
                const update = {{}};
                const visibility = [];

                // 计算哪些traces应该可见
                for (let i = 0; i < numFifoOptions; i++) {{
                    for (let mode of ['avg', 'peak']) {{
                        for (let die = 0; die < numDies; die++) {{
                            const shouldShow = (i === currentFifoIndex && mode === currentMode);
                            visibility.push(shouldShow);
                        }}
                    }}
                }}

                // 架构图的traces保持可见
                for (let i = numHeatmapTraces; i < plotDiv.data.length; i++) {{
                    visibility.push(true);
                }}

                update.visible = visibility;
                Plotly.restyle(plotDiv, update);

                // 更新架构图高亮
                updateArchitectureHighlight();
            }}

            // 更新架构图高亮
            function updateArchitectureHighlight() {{
                // 获取所有shapes
                const shapes = plotDiv.layout.shapes || [];
                const newShapes = shapes.map((shape, idx) => {{
                    // 跳过模块边框（前3个shape是模块边框）
                    if (idx < 3 || !shape.name) {{
                        return shape;
                    }}

                    // 检查是否为当前选中的FIFO
                    const shapeName = shape.name;
                    const fifoIndex = fifoMap[shapeName];
                    const isSelected = (fifoIndex === currentFifoIndex);

                    // 返回更新后的shape
                    return {{
                        ...shape,
                        line: {{
                            ...shape.line,
                            color: isSelected ? 'red' : 'black',
                            width: isSelected ? 3 : 1
                        }}
                    }};
                }});

                // 更新layout
                Plotly.relayout(plotDiv, {{'shapes': newShapes}});
            }}

            // 监听按钮点击（平均/峰值切换）
            const buttons = plotDiv.querySelectorAll('.updatemenu-button');
            if (buttons.length > 0) {{
                buttons.forEach((btn, idx) => {{
                    btn.addEventListener('click', function() {{
                        currentMode = (idx === 0) ? 'avg' : 'peak';
                        // 立即更新显示，不需要延迟
                        updateHeatmapVisibility();
                    }});
                }});
            }}

            // 初始化时高亮默认FIFO
            updateArchitectureHighlight();
        }}, 500);  // 延迟500ms确保Plotly完全加载
    }});
</script>
"""

        # 在</body>之前注入JavaScript
        html_string = html_string.replace("</body>", js_code + "</body>")

        # 保存修改后的HTML
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_string)


def create_fifo_heatmap(dies: Dict, config, total_cycles: int, die_layout: Optional[Dict] = None, die_rotations: Optional[Dict] = None, save_path: Optional[str] = None) -> Optional[str]:
    """
    便捷函数：一键创建FIFO使用率热力图

    Args:
        dies: Die字典
        config: 配置对象
        total_cycles: 仿真总周期数
        die_layout: Die布局
        die_rotations: Die旋转
        save_path: 保存路径

    Returns:
        str: 保存路径（如果提供）
    """
    # 收集数据
    collector = FIFOUtilizationCollector(config)
    fifo_data = collector.collect_from_dies(dies, total_cycles)

    # 创建可视化
    visualizer = FIFOHeatmapVisualizer(config, fifo_data)
    return visualizer.create_interactive_heatmap(dies, die_layout, die_rotations, save_path)
