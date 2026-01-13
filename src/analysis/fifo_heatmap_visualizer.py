"""
FIFO使用率热力图可视化模块

提供FIFO使用率数据收集和交互式热力图可视化功能
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from src.utils.statistical_fifo import StatisticalFIFO


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

    def collect_from_network(self, network, die_id: int, total_cycles: int, network_type: str) -> Dict:
        """
        从单个Network对象收集FIFO使用率数据

        使用 StatisticalFIFO 获取统计数据

        Args:
            network: Network对象
            die_id: Die ID
            total_cycles: 仿真总周期数
            network_type: 网络类型 ('req', 'rsp', 'data')

        Returns:
            Dict: FIFO使用率数据
        """
        die_data = {"IQ": {}, "RB": {}, "EQ": {}, "IQ_CH": {}, "EQ_CH": {},
                    "RS_IN": {}, "RS_OUT": {}}  # v2 新增类别

        if total_cycles <= 0:
            print(f"警告: Die {die_id} 总周期数无效: {total_cycles}")
            return die_data

        # 从 StatisticalFIFO 获取数据
        all_fifos = StatisticalFIFO.get_all_fifos()

        # FIFO名称后缀映射 (用于过滤当前网络的FIFO)
        # 注意: 后缀是从 network.name 提取的前3字母，如 "Response Network" -> "res"
        network_suffix = {"req": "_req", "rsp": "_res", "data": "_dat"}
        suffix = network_suffix.get(network_type, "")

        for name, fifo in all_fifos.items():
            # 过滤只属于当前网络的FIFO (名称以_req/_res/_dat结尾)
            if suffix and not name.endswith(suffix):
                continue

            node_pos = fifo.node_pos
            category = fifo.category
            fifo_type = fifo.fifo_type
            ip_type = fifo.ip_type

            # 计算统计数据
            capacity = fifo.maxlen
            avg_depth = fifo.get_avg_depth()
            peak_depth = fifo.max_depth
            flit_count = fifo.flit_count

            avg_util = (avg_depth / capacity * 100) if capacity > 0 else 0
            peak_util = (peak_depth / capacity * 100) if capacity > 0 else 0

            stat_data = {
                "avg": avg_util,
                "peak": peak_util,
                "flit_count": flit_count,
                "capacity": capacity,
                "avg_depth": avg_depth,
                "peak_depth": peak_depth,
                "itag_cumulative_count": 0,
                "itag_rate": 0.0,
                "etag_t0_cumulative": 0,
                "etag_t1_cumulative": 0,
                "etag_t2_cumulative": 0,
                "etag_t0_rate": 0.0,
                "etag_t1_rate": 0.0,
                "etag_t2_rate": 0.0,
                "reverse_inject_count": 0,
                "reverse_inject_rate": 0.0,
            }

            # 分类存储
            if category in ["IQ", "RB", "EQ"]:
                # v1 方向 FIFO
                if fifo_type in ["TL", "TR", "TU", "TD", "EQ"]:
                    if node_pos not in die_data[category]:
                        die_data[category][node_pos] = {}
                    die_data[category][node_pos][fifo_type] = stat_data
                # v1 CH buffer
                elif fifo_type == "CH" and ip_type:
                    ch_key = f"{category}_CH"
                    if node_pos not in die_data[ch_key]:
                        die_data[ch_key][node_pos] = {}
                    die_data[ch_key][node_pos][ip_type] = stat_data

            elif category in ["RS_IN", "RS_OUT"]:
                # v2 RingStation FIFO
                if node_pos not in die_data[category]:
                    die_data[category][node_pos] = {}
                key = ip_type if ip_type else fifo_type
                die_data[category][node_pos][key] = stat_data

        # 附加 ITag/ETag 统计 (从 network 获取)
        self._collect_itag_etag_stats(network, die_data, total_cycles)

        return die_data

    def _collect_itag_etag_stats(self, network, die_data: Dict, total_cycles: int):
        """收集 ITag/ETag 统计数据"""
        # IQ ITag统计 (TR/TL横向注入)
        for direction in ["TR", "TL"]:
            if hasattr(network, "fifo_itag_cumulative_count"):
                for node_pos in die_data.get("IQ", {}):
                    if direction in die_data["IQ"].get(node_pos, {}):
                        itag_cumulative = network.fifo_itag_cumulative_count.get("IQ", {}).get(direction, {}).get(node_pos, 0)
                        flit_count = die_data["IQ"][node_pos][direction].get("flit_count", 0)
                        itag_rate = (itag_cumulative / (total_cycles * flit_count) * 100) if flit_count > 0 and total_cycles > 0 else 0
                        die_data["IQ"][node_pos][direction]["itag_cumulative_count"] = itag_cumulative
                        die_data["IQ"][node_pos][direction]["itag_rate"] = itag_rate

        # RB ITag统计 (TU/TD纵向转向)
        for direction in ["TU", "TD"]:
            if hasattr(network, "fifo_itag_cumulative_count"):
                for node_pos in die_data.get("RB", {}):
                    if direction in die_data["RB"].get(node_pos, {}):
                        itag_cumulative = network.fifo_itag_cumulative_count.get("RB", {}).get(direction, {}).get(node_pos, 0)
                        flit_count = die_data["RB"][node_pos][direction].get("flit_count", 0)
                        itag_rate = (itag_cumulative / (total_cycles * flit_count) * 100) if flit_count > 0 and total_cycles > 0 else 0
                        die_data["RB"][node_pos][direction]["itag_cumulative_count"] = itag_cumulative
                        die_data["RB"][node_pos][direction]["itag_rate"] = itag_rate

        # RB ETag统计 (TL/TR横向下环)
        for direction in ["TL", "TR"]:
            if hasattr(network, "fifo_etag_entry_count"):
                for node_pos in die_data.get("RB", {}):
                    if direction in die_data["RB"].get(node_pos, {}):
                        etag_dist = network.fifo_etag_entry_count.get("RB", {}).get(direction, {}).get(node_pos, {})
                        if etag_dist:
                            t0, t1, t2 = etag_dist.get("T0", 0), etag_dist.get("T1", 0), etag_dist.get("T2", 0)
                            total_etag = t0 + t1 + t2
                            die_data["RB"][node_pos][direction]["etag_t0_cumulative"] = t0
                            die_data["RB"][node_pos][direction]["etag_t1_cumulative"] = t1
                            die_data["RB"][node_pos][direction]["etag_t2_cumulative"] = t2
                            if total_etag > 0:
                                die_data["RB"][node_pos][direction]["etag_t0_rate"] = t0 / total_etag * 100
                                die_data["RB"][node_pos][direction]["etag_t1_rate"] = t1 / total_etag * 100
                                die_data["RB"][node_pos][direction]["etag_t2_rate"] = t2 / total_etag * 100

        # EQ ETag统计 (TU/TD纵向下环)
        for direction in ["TU", "TD"]:
            if hasattr(network, "fifo_etag_entry_count"):
                for node_pos in die_data.get("EQ", {}):
                    if direction in die_data["EQ"].get(node_pos, {}):
                        etag_dist = network.fifo_etag_entry_count.get("EQ", {}).get(direction, {}).get(node_pos, {})
                        if etag_dist:
                            t0, t1, t2 = etag_dist.get("T0", 0), etag_dist.get("T1", 0), etag_dist.get("T2", 0)
                            total_etag = t0 + t1 + t2
                            die_data["EQ"][node_pos][direction]["etag_t0_cumulative"] = t0
                            die_data["EQ"][node_pos][direction]["etag_t1_cumulative"] = t1
                            die_data["EQ"][node_pos][direction]["etag_t2_cumulative"] = t2
                            if total_etag > 0:
                                die_data["EQ"][node_pos][direction]["etag_t0_rate"] = t0 / total_etag * 100
                                die_data["EQ"][node_pos][direction]["etag_t1_rate"] = t1 / total_etag * 100
                                die_data["EQ"][node_pos][direction]["etag_t2_rate"] = t2 / total_etag * 100

    def collect_from_dies(self, dies: Dict, total_cycles: int) -> Dict:
        """
        从所有Die收集FIFO使用率数据（支持三个网络）

        Args:
            dies: Die字典 {die_id: die_model}
            total_cycles: 仿真总周期数

        Returns:
            Dict: {die_id: {network_type: die_data}}
        """
        self.fifo_utilization_data = {}

        for die_id, die_model in dies.items():
            self.fifo_utilization_data[die_id] = {}

            # 收集三个网络的数据
            network_types = [("req", "req_network"), ("rsp", "rsp_network"), ("data", "data_network")]

            for network_type, network_attr in network_types:
                if hasattr(die_model, network_attr):
                    network = getattr(die_model, network_attr)
                    self.fifo_utilization_data[die_id][network_type] = self.collect_from_network(network, die_id, total_cycles, network_type)
                else:
                    print(f"警告: Die {die_id} 没有{network_attr}属性")

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
            if fifo_type in ["TL", "TR"]:
                return getattr(config, "RB_IN_FIFO_DEPTH", 16)  # 输入FIFO
            else:  # TU, TD, EQ
                return getattr(config, "RB_OUT_FIFO_DEPTH", 16)  # 输出FIFO

        elif fifo_category == "EQ":
            return getattr(config, "EQ_IN_FIFO_DEPTH", 8)

        elif fifo_category == "EQ_CH":
            return getattr(config, "EQ_CH_FIFO_DEPTH", 4)  # EQ_CH和IQ_CH容量相同

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

    def create_interactive_heatmap(self, dies: Dict, die_layout: Optional[Dict] = None, die_rotations: Optional[Dict] = None, save_path: Optional[str] = None, show_fig: bool = False, return_fig_and_js: bool = False):
        """
        创建交互式FIFO使用率热力图

        Args:
            dies: Die字典
            die_layout: Die布局 {die_id: (grid_x, grid_y)}
            die_rotations: Die旋转角度 {die_id: rotation}
            save_path: HTML文件保存路径
            show_fig: 是否在浏览器中显示
            return_fig_and_js: 是否返回(Figure, JavaScript)元组而不是保存文件

        Returns:
            str or tuple: 如果return_fig_and_js=True，返回(Figure对象, JavaScript字符串)；否则返回保存路径
        """
        if not self.fifo_data:
            print("警告: 没有FIFO数据可供可视化")
            return None

        # 获取所有可用的FIFO类型选项
        fifo_options = self._get_available_fifo_types(dies)

        if not fifo_options:
            print("警告: 没有找到可用的FIFO类型")
            return None

        # 创建图形
        fig = self._create_plotly_figure(dies, die_layout, die_rotations, fifo_options, return_fig_and_js)

        # 生成多通道按钮（如果是多通道情况）
        buttons_html = None
        buttons_js_extra = None

        # 检查是否是多通道
        num_channels = self._count_channels(dies)
        if num_channels > 1:
            buttons_html, buttons_js_extra = self._generate_fifo_multichannel_buttons(dies, num_channels)

        # 如果只返回Figure和JavaScript
        if return_fig_and_js:
            js_code = self._generate_custom_javascript(fifo_options, len(dies))
            # 如果有多通道按钮，添加到JavaScript
            if buttons_js_extra:
                js_code += "\n" + buttons_js_extra
            # 返回(fig, buttons_html, buttons_js)元组
            if buttons_html:
                return fig, buttons_html, js_code
            else:
                return fig, None, js_code

        # 保存或显示
        if save_path:
            # 生成HTML并注入JavaScript交互代码
            self._save_html_with_click_events(fig, save_path, fifo_options, len(dies))
            # print(f"FIFO使用率热力图已保存到: {save_path}")
            if show_fig:
                import webbrowser
                import os
                webbrowser.open('file://' + os.path.abspath(save_path))
            return save_path
        else:
            fig.show()
            return None

    def _get_available_fifo_types(self, dies: Dict) -> List[Tuple[str, str, str, str]]:
        """
        获取所有可用的FIFO类型（支持三网络）

        Args:
            dies: Die字典

        Returns:
            List[Tuple]: [(显示名称, fifo_category, fifo_type, network_type), ...]
        """
        options = []
        network_display = {"req": "请求", "rsp": "响应", "data": "数据"}

        # 获取配置中的CH_NAME_LIST（从任意一个Die获取）
        ch_name_list = []
        if dies:
            first_die = list(dies.values())[0]
            if hasattr(first_die, "config") and hasattr(first_die.config, "CH_NAME_LIST"):
                ch_name_list = first_die.config.CH_NAME_LIST

        for die_id, networks_data in self.fifo_data.items():
            for network_type, die_data in networks_data.items():
                net_label = network_display.get(network_type, network_type)

                # IQ方向队列
                for node_pos, directions in die_data.get("IQ", {}).items():
                    for direction in directions.keys():
                        option_name = f"IQ-{direction} ({net_label})"
                        option_tuple = (option_name, "IQ", direction, network_type)
                        if option_tuple not in options:
                            options.append(option_tuple)

                # IQ通道缓冲 - 使用配置中的CH_NAME_LIST
                if ch_name_list:
                    for ip_type in ch_name_list:
                        option_name = f"IQ_CH-{ip_type} ({net_label})"
                        option_tuple = (option_name, "IQ_CH", ip_type, network_type)
                        if option_tuple not in options:
                            options.append(option_tuple)
                else:
                    # 降级方案：从已收集的数据获取
                    for node_pos, ip_types in die_data.get("IQ_CH", {}).items():
                        for ip_type in ip_types.keys():
                            option_name = f"IQ_CH-{ip_type} ({net_label})"
                            option_tuple = (option_name, "IQ_CH", ip_type, network_type)
                            if option_tuple not in options:
                                options.append(option_tuple)

                # RB
                for node_pos, directions in die_data.get("RB", {}).items():
                    for direction in directions.keys():
                        option_name = f"RB-{direction} ({net_label})"
                        option_tuple = (option_name, "RB", direction, network_type)
                        if option_tuple not in options:
                            options.append(option_tuple)

                # EQ下环队列
                for node_pos, directions in die_data.get("EQ", {}).items():
                    for direction in directions.keys():
                        option_name = f"EQ-{direction} ({net_label})"
                        option_tuple = (option_name, "EQ", direction, network_type)
                        if option_tuple not in options:
                            options.append(option_tuple)

                # EQ通道缓冲 - 使用配置中的CH_NAME_LIST
                if ch_name_list:
                    for ip_type in ch_name_list:
                        option_name = f"EQ_CH-{ip_type} ({net_label})"
                        option_tuple = (option_name, "EQ_CH", ip_type, network_type)
                        if option_tuple not in options:
                            options.append(option_tuple)
                else:
                    # 降级方案：从已收集的数据获取
                    for node_pos, ip_types in die_data.get("EQ_CH", {}).items():
                        for ip_type in ip_types.keys():
                            option_name = f"EQ_CH-{ip_type} ({net_label})"
                            option_tuple = (option_name, "EQ_CH", ip_type, network_type)
                            if option_tuple not in options:
                                options.append(option_tuple)

                # RS_IN (v2 RingStation 输入端)
                for node_pos, fifo_types in die_data.get("RS_IN", {}).items():
                    for fifo_key in fifo_types.keys():
                        option_name = f"RS_IN-{fifo_key} ({net_label})"
                        option_tuple = (option_name, "RS_IN", fifo_key, network_type)
                        if option_tuple not in options:
                            options.append(option_tuple)

                # RS_OUT (v2 RingStation 输出端)
                for node_pos, fifo_types in die_data.get("RS_OUT", {}).items():
                    for fifo_key in fifo_types.keys():
                        option_name = f"RS_OUT-{fifo_key} ({net_label})"
                        option_tuple = (option_name, "RS_OUT", fifo_key, network_type)
                        if option_tuple not in options:
                            options.append(option_tuple)

        # 排序：先按category，再按network_type，最后按名称
        def sort_key(item):
            name, category, _, network_type = item
            category_order = {"IQ": 0, "RB": 1, "EQ": 2, "IQ_CH": 3, "EQ_CH": 4, "RS_IN": 5, "RS_OUT": 6}
            network_order = {"req": 0, "rsp": 1, "data": 2}
            return (category_order.get(category, 999), network_order.get(network_type, 999), name)

        options.sort(key=sort_key)
        return options

    def _create_plotly_figure(self, dies: Dict, die_layout: Optional[Dict], die_rotations: Optional[Dict], fifo_options: List[Tuple[str, str, str, str]], hide_subplot_titles: bool = False) -> go.Figure:
        """
        创建Plotly交互式图形 - 左侧热力图 + 右侧架构图

        Args:
            dies: Die字典
            die_layout: Die布局
            die_rotations: Die旋转
            fifo_options: FIFO选项列表
            hide_subplot_titles: 是否隐藏子图标题（用于集成报告）

        Returns:
            go.Figure: Plotly图形对象
        """
        # 准备Die布局和旋转信息
        if die_layout is None:
            die_layout = {die_id: (die_id, 0) for die_id in dies.keys()}
        if die_rotations is None:
            die_rotations = {die_id: 0 for die_id in dies.keys()}

        # 创建子图布局: 1行2列 (50% + 50%)
        subplot_titles = None if hide_subplot_titles else ("FIFO使用率热力图", "CrossRing架构图")
        fig = make_subplots(
            rows=1, cols=2, column_widths=[0.5, 0.5], subplot_titles=subplot_titles, specs=[[{"type": "scatter"}, {"type": "scatter"}]], horizontal_spacing=0.02
        )

        # 计算画布范围（用于坐标轴设置）
        die_offsets, max_x, max_y = self._calculate_die_offsets_from_layout(die_layout, die_rotations, dies)

        # 为每个FIFO类型和统计模式创建热力图trace
        traces_data = self._create_heatmap_traces(fifo_options, dies, die_layout, die_rotations)

        # 将所有traces添加到左侧子图（初始时显示第一个data网络选项的平均模式）
        # 查找第一个data网络的FIFO选项
        default_option = fifo_options[0]  # 默认值
        for option in fifo_options:
            if option[3] == "data":  # network_type是元组的第4个元素
                default_option = option
                break
        default_mode = "avg"

        # 批量收集所有traces并设置subplot坐标
        all_traces = []
        for option in fifo_options:
            for mode in ["avg", "peak", "flit_count"]:
                trace = traces_data[(option, mode)]
                # 只有默认选项的默认模式可见
                trace.visible = option == default_option and mode == default_mode
                # 设置subplot位置 (row=1, col=1对应xaxis, yaxis)
                trace.xaxis = "x"
                trace.yaxis = "y"
                all_traces.append(trace)

        # 批量添加所有traces（直接扩展fig.data）
        fig.add_traces(all_traces)

        # 添加Die边框和标签
        self._add_die_borders_and_labels(fig, dies, die_layout, die_rotations)

        # 添加右侧架构图
        self._add_architecture_diagram(fig, fifo_options, dies)

        # 添加平均/峰值按钮
        self._add_mode_buttons(fig, fifo_options, traces_data)

        # 设置布局
        fig.update_layout(
            hovermode="closest",
            width=1800,
            height=900,
            plot_bgcolor="white",
            margin=dict(t=80, b=20, l=20, r=20),  # 紧凑布局，减少边距
            showlegend=False,
        )

        # 左侧热力图坐标轴设置（自动缩放）
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="",
            scaleanchor="y1",
            scaleratio=1,
            autorange=True,
            automargin=True,
            row=1,
            col=1,
        )
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, title="", autorange=True, automargin=True, row=1, col=1)

        # 隐藏右侧架构图的坐标轴
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, title="", row=1, col=2)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, title="", row=1, col=2)

        return fig

    def _calculate_die_offsets_from_layout(self, die_layout: Dict, die_rotations: Dict, dies: Dict) -> Tuple[Dict, float, float]:
        """
        根据die_layout计算每个Die的偏移量（改进的对齐算法）

        Args:
            die_layout: Die布局 {die_id: (grid_x, grid_y)}
            die_rotations: Die旋转角度 {die_id: rotation}
            dies: Die字典 {die_id: die_model}

        Returns:
            (die_offsets, max_x, max_y): Die偏移量字典和画布最大坐标
        """
        if not die_layout:
            # 默认布局：所有Die水平排列
            die_layout = {die_id: (die_id, 0) for die_id in dies.keys()}

        if not die_rotations:
            die_rotations = {die_id: 0 for die_id in dies.keys()}

        # 节点间距和Die间距（根据Die数量自适应）
        if len(dies) == 1:
            node_spacing = 1  # 单Die时节点更紧密
        else:
            node_spacing = 2.15  # 多Die时节点稍微松散
        die_gap = 1.5  # Die之间的间距

        # 计算每个Die旋转后的实际尺寸（以像素计）
        die_sizes = {}
        for die_id, die_model in dies.items():
            rows = die_model.config.NUM_ROW
            cols = die_model.config.NUM_COL
            rotation = die_rotations.get(die_id, 0)

            if rotation in [90, 270, -90, -270]:
                # 90度或270度旋转：宽高互换
                width = (rows - 1) * node_spacing
                height = (cols - 1) * node_spacing
            else:
                # 0度或180度旋转：宽高不变
                width = (cols - 1) * node_spacing
                height = (rows - 1) * node_spacing

            die_sizes[die_id] = (width, height)

        # 构建网格结构：记录每个网格位置的Die
        grid_dies = {}
        for die_id, (grid_x, grid_y) in die_layout.items():
            grid_dies[(grid_x, grid_y)] = die_id

        # 获取网格范围
        all_grid_x = [pos[0] for pos in die_layout.values()]
        all_grid_y = [pos[1] for pos in die_layout.values()]
        min_grid_x, max_grid_x = min(all_grid_x), max(all_grid_x)
        min_grid_y, max_grid_y = min(all_grid_y), max(all_grid_y)

        # 两步对齐算法：
        # 步骤1: 计算每行每列的累积位置（基于左侧和下侧的Die）
        col_positions = {}  # grid_x -> 该列左边界的x坐标
        row_positions = {}  # grid_y -> 该行下边界的y坐标

        # 从左到右计算列位置
        current_x = 0
        for grid_x in range(min_grid_x, max_grid_x + 1):
            col_positions[grid_x] = current_x
            # 找出这一列中最宽的Die
            max_width = 0
            for grid_y in range(min_grid_y, max_grid_y + 1):
                if (grid_x, grid_y) in grid_dies:
                    die_id = grid_dies[(grid_x, grid_y)]
                    width, _ = die_sizes[die_id]
                    max_width = max(max_width, width)
            current_x += max_width + die_gap

        # 从下到上计算行位置
        current_y = 0
        for grid_y in range(min_grid_y, max_grid_y + 1):
            row_positions[grid_y] = current_y
            # 找出这一行中最高的Die
            max_height = 0
            for grid_x in range(min_grid_x, max_grid_x + 1):
                if (grid_x, grid_y) in grid_dies:
                    die_id = grid_dies[(grid_x, grid_y)]
                    _, height = die_sizes[die_id]
                    max_height = max(max_height, height)
            current_y += max_height + die_gap

        # 步骤2: 根据行列位置和对齐规则确定每个Die的偏移
        die_offsets = {}
        for die_id, (grid_x, grid_y) in die_layout.items():
            width, height = die_sizes[die_id]

            # 获取该格子的基准位置
            base_x = col_positions[grid_x]
            base_y = row_positions[grid_y]

            # 计算该列和该行的最大尺寸
            col_max_width = 0
            for gy in range(min_grid_y, max_grid_y + 1):
                if (grid_x, gy) in grid_dies:
                    w, _ = die_sizes[grid_dies[(grid_x, gy)]]
                    col_max_width = max(col_max_width, w)

            row_max_height = 0
            for gx in range(min_grid_x, max_grid_x + 1):
                if (gx, grid_y) in grid_dies:
                    _, h = die_sizes[grid_dies[(gx, grid_y)]]
                    row_max_height = max(row_max_height, h)

            # 对齐策略：
            # X方向：最左列左对齐，最右列右对齐，其他列左对齐
            if grid_x == min_grid_x:
                offset_x = base_x  # 左对齐
            elif grid_x == max_grid_x:
                offset_x = base_x + col_max_width - width  # 右对齐
            else:
                offset_x = base_x  # 左对齐

            # Y方向：最下行下对齐，最上行上对齐，其他行下对齐
            if grid_y == min_grid_y:
                offset_y = base_y  # 下对齐
            elif grid_y == max_grid_y:
                offset_y = base_y + row_max_height - height  # 上对齐
            else:
                offset_y = base_y  # 下对齐

            die_offsets[die_id] = (offset_x, offset_y)

        # 计算画布范围
        max_x = col_positions[max_grid_x]
        for grid_y in range(min_grid_y, max_grid_y + 1):
            if (max_grid_x, grid_y) in grid_dies:
                die_id = grid_dies[(max_grid_x, grid_y)]
                width, _ = die_sizes[die_id]
                max_x = max(max_x, col_positions[max_grid_x] + width)

        max_y = row_positions[max_grid_y]
        for grid_x in range(min_grid_x, max_grid_x + 1):
            if (grid_x, max_grid_y) in grid_dies:
                die_id = grid_dies[(grid_x, max_grid_y)]
                _, height = die_sizes[die_id]
                max_y = max(max_y, row_positions[max_grid_y] + height)

        return die_offsets, max_x, max_y

    def _apply_rotation(self, orig_row: int, orig_col: int, rows: int, cols: int, rotation: int) -> Tuple[int, int]:
        """
        应用Die旋转变换

        Args:
            orig_row: 原始行号
            orig_col: 原始列号
            rows: Die总行数
            cols: Die总列数
            rotation: 旋转角度（0/90/180/270）

        Returns:
            (new_row, new_col): 旋转后的行列号
        """
        if rotation == 0:
            return orig_row, orig_col
        elif rotation in [90, -270]:
            return orig_col, rows - 1 - orig_row
        elif rotation in [180, -180]:
            return rows - 1 - orig_row, cols - 1 - orig_col
        elif rotation in [270, -90]:
            return cols - 1 - orig_col, orig_row
        else:
            # 未知旋转角度，不应用旋转
            return orig_row, orig_col

    def _create_heatmap_traces(self, fifo_options: List[Tuple[str, str, str, str]], dies: Dict, die_layout: Dict, die_rotations: Dict) -> Dict:
        """
        为所有FIFO选项和统计模式创建热力图traces（使用Scatter支持多Die布局）

        Returns:
            Dict: {(option_tuple, mode): trace}
        """
        # 计算Die偏移量
        die_offsets, max_x, max_y = self._calculate_die_offsets_from_layout(die_layout, die_rotations, dies)

        # 节点间距（根据Die数量自适应）
        if len(dies) == 1:
            node_spacing = 0.8  # 单Die时节点更紧密
            size = 130
        else:
            node_spacing = 1.7  # 多Die时节点稍微松散
            size = 56

        traces_data = {}

        # 预扫描：计算所有FIFO的flit_count全局最大值（用于统一颜色范围）
        global_max_flit_count = 0
        for option in fifo_options:
            option_name, fifo_category, fifo_type, network_type = option
            for die_id in sorted(self.fifo_data.keys()):
                networks_data = self.fifo_data[die_id]
                die_data = networks_data.get(network_type, {})
                category_data = die_data.get(fifo_category, {})

                for node_data in category_data.values():
                    if isinstance(node_data, dict):
                        fifo_info = node_data.get(fifo_type)
                        if fifo_info and isinstance(fifo_info, dict):
                            flit_count = fifo_info.get("flit_count", 0)
                            global_max_flit_count = max(global_max_flit_count, flit_count)

        # 确保至少为1，避免除零
        if global_max_flit_count == 0:
            global_max_flit_count = 1

        for option in fifo_options:
            option_name, fifo_category, fifo_type, network_type = option

            for mode in ["avg", "peak", "flit_count"]:
                # 收集所有Die的所有节点数据
                all_x = []
                all_y = []
                all_colors = []
                all_hover_texts = []
                all_text_labels = []

                # 为每个Die收集数据
                for die_id in sorted(self.fifo_data.keys()):
                    die_model = dies[die_id]
                    die_config = die_model.config
                    rows = die_config.NUM_ROW
                    cols = die_config.NUM_COL
                    rotation = die_rotations.get(die_id, 0)
                    offset_x, offset_y = die_offsets[die_id]

                    networks_data = self.fifo_data[die_id]
                    die_data = networks_data.get(network_type, {})
                    category_data = die_data.get(fifo_category, {})

                    # === 先遍历一遍，记录本die所有节点旋转后的new_row最大值，用于底对齐 ===
                    rotated_rows = [self._apply_rotation(node_id // cols, node_id % cols, rows, cols, rotation)[0] for node_id in range(rows * cols)]
                    max_new_row = max(rotated_rows)

                    for node_id in range(rows * cols):
                        orig_row = node_id // cols
                        orig_col = node_id % cols
                        new_row, new_col = self._apply_rotation(orig_row, orig_col, rows, cols, rotation)

                        # 统一底对齐：y轴方向用 max_new_row-new_row
                        y = offset_y + (max_new_row - new_row) * node_spacing
                        x = offset_x + new_col * node_spacing

                        node_data = category_data.get(node_id, {})
                        fifo_info = node_data.get(fifo_type)

                        # 对于channel buffer (IQ_CH/EQ_CH)，即使没有配置也显示空白格子
                        is_channel_buffer = fifo_category in ["IQ_CH", "EQ_CH"]

                        if fifo_info is not None:
                            capacity = fifo_info["capacity"]
                            avg_depth = fifo_info["avg_depth"]
                            peak_depth = fifo_info["peak_depth"]
                            flit_count = fifo_info.get("flit_count", 0)

                            # 根据模式选择显示的值
                            if mode == "flit_count":
                                display_value = flit_count
                                text_label = f"{int(flit_count)}"  # 确保显示为整数
                            else:
                                display_value = fifo_info[mode]
                                text_label = f"{display_value:.1f}%"

                            all_x.append(x)
                            all_y.append(y)
                            all_colors.append(display_value)
                            all_text_labels.append(text_label)

                            network_display = {"req": "Request", "rsp": "Response", "data": "Data"}
                            net_label = network_display.get(network_type, network_type)

                            # 构建Tag统计信息
                            tag_info = []

                            # ITag统计（只有IQ_OUT TR/TL和RB_OUT TU/TD有ITag）
                            itag_cumulative = fifo_info.get("itag_cumulative_count", 0)
                            itag_rate = fifo_info.get("itag_rate", 0.0)
                            if itag_cumulative > 0:
                                tag_info.append(f"ITag: {itag_cumulative} ({itag_rate:.2f}%)")

                            # ETag统计（只有RB_IN TL/TR和EQ_IN TU/TD有ETag）
                            etag_t0 = fifo_info.get("etag_t0_cumulative", 0)
                            etag_t1 = fifo_info.get("etag_t1_cumulative", 0)
                            etag_t2 = fifo_info.get("etag_t2_cumulative", 0)
                            etag_t0_rate = fifo_info.get("etag_t0_rate", 0.0)
                            etag_t1_rate = fifo_info.get("etag_t1_rate", 0.0)
                            etag_t2_rate = fifo_info.get("etag_t2_rate", 0.0)

                            if etag_t0 > 0 or etag_t1 > 0 or etag_t2 > 0:
                                tag_info.append(f"ETag T0: {etag_t0} ({etag_t0_rate:.2f}%)")
                                tag_info.append(f"ETag T1: {etag_t1} ({etag_t1_rate:.2f}%)")
                                tag_info.append(f"ETag T2: {etag_t2} ({etag_t2_rate:.2f}%)")

                            # 反方向上环统计（只有功能开启且有数据时显示）
                            reverse_inject_count = fifo_info.get("reverse_inject_count", 0)
                            reverse_inject_rate = fifo_info.get("reverse_inject_rate", 0.0)
                            if getattr(self.config, "REVERSE_DIRECTION_ENABLED", False) and reverse_inject_count > 0:
                                tag_info.append(f"反方向上环: {reverse_inject_count} ({reverse_inject_rate:.2f}%)")

                            # 组装hover文本
                            hover_text = (
                                f"Die {die_id} - 节点 {node_id} ({orig_row},{orig_col})<br>"
                                f"网络: {net_label}<br>"
                                f"FIFO: {option_name}<br>"
                                f"平均: {fifo_info['avg']:.1f}% ({avg_depth:.2f}/{capacity})<br>"
                                f"峰值: {fifo_info['peak']:.1f}% ({peak_depth}/{capacity})<br>"
                                f"累计: {flit_count} flits<br>"
                                f"容量: {capacity}"
                            )

                            # 添加Tag信息（如果有）
                            if tag_info:
                                hover_text += "<br>" + "<br>".join(tag_info)

                            all_hover_texts.append(hover_text)

                        elif is_channel_buffer:
                            # 对于channel buffer，没有配置时显示空白（灰色）
                            all_x.append(x)
                            all_y.append(y)
                            all_colors.append(0)  # 设置为0显示灰色

                            # 根据模式显示不同格式
                            if mode == "flit_count":
                                text_label = "0"  # 数字模式显示0
                            else:
                                text_label = "0.0%"  # 百分比模式显示0.0%
                            all_text_labels.append(text_label)

                            network_display = {"req": "Request", "rsp": "Response", "data": "Data"}
                            net_label = network_display.get(network_type, network_type)

                            hover_text = (
                                f"Die {die_id} - 节点 {node_id} ({orig_row},{orig_col})<br>"
                                f"网络: {net_label}<br>"
                                f"FIFO: {option_name}<br>"
                                f"状态: 未配置"
                            )
                            all_hover_texts.append(hover_text)

                # 根据模式配置colorscale和colorbar
                if mode == "flit_count":
                    # flit_count模式：使用全局最大值和统一配色方案
                    colorscale = "RdYlGn_r"
                    cmin = 0
                    cmax = global_max_flit_count
                    colorbar_title = "累计Flit数"
                    colorbar_config = dict(title=colorbar_title, thickness=18, x=-0.02, tickformat="d")  # 整数格式，不显示小数点
                else:
                    # 使用率模式：0-100%
                    colorscale = "RdYlGn_r"
                    cmin = 0
                    cmax = 100
                    colorbar_title = "使用率 (%)"
                    colorbar_config = dict(title=colorbar_title, thickness=18, tickmode="array", x=-0.02)

                # 创建单个Scatter trace（包含所有Die的所有节点）
                trace = go.Scatter(
                    x=all_x,
                    y=all_y,
                    mode="markers+text",
                    marker=dict(
                        size=size,  # ← 增大节点尺寸
                        symbol="square",
                        color=all_colors,
                        colorscale=colorscale,
                        cmin=cmin,
                        cmax=cmax,
                        colorbar=colorbar_config,
                        line=dict(width=1, color="black"),
                    ),
                    text=all_text_labels,
                    textfont=dict(size=14, color="black"),
                    textposition="middle center",
                    hovertext=all_hover_texts,
                    hoverinfo="text",
                    showlegend=False,
                )

                traces_data[(option, mode)] = trace

        return traces_data

    def _add_die_borders_and_labels(self, fig: go.Figure, dies: Dict, die_layout: Dict, die_rotations: Dict):
        """
        添加Die边框和标签

        Args:
            fig: Plotly图形对象
            dies: Die字典
            die_layout: Die布局
            die_rotations: Die旋转角度
        """
        # 计算Die偏移量
        die_offsets, _, _ = self._calculate_die_offsets_from_layout(die_layout, die_rotations, dies)

        # 节点间距（根据Die数量自适应，与trace生成保持一致）
        if len(dies) == 1:
            node_spacing = 0.1  # 单Die时节点更紧密
        else:
            node_spacing = 1.6  # 多Die时节点稍微松散

        for die_id, die_model in dies.items():
            rows = die_model.config.NUM_ROW
            cols = die_model.config.NUM_COL
            rotation = die_rotations.get(die_id, 0)
            offset_x, offset_y = die_offsets[die_id]

            # 计算Die实际尺寸（考虑旋转）
            if rotation in [90, 270, -90, -270]:
                die_width = (rows - 1) * node_spacing
                die_height = (cols - 1) * node_spacing
            else:
                die_width = (cols - 1) * node_spacing
                die_height = (rows - 1) * node_spacing

    def _add_mode_buttons(self, fig: go.Figure, fifo_options: List[Tuple[str, str, str, str]], traces_data: Dict):
        """添加平均/峰值切换按钮和网络类型切换按钮"""

        # 创建平均/峰值/Flit计数按钮
        mode_buttons = []
        for target_mode in ["avg", "peak", "flit_count"]:
            if target_mode == "avg":
                mode_label = "平均使用率"
            elif target_mode == "peak":
                mode_label = "峰值使用率"
            else:  # flit_count
                mode_label = "Flit计数"
            button = dict(label=mode_label, method="skip")
            mode_buttons.append(button)

        # 创建网络类型切换按钮
        network_buttons = []
        for network_type in ["req", "rsp", "data"]:
            network_label = {"req": "请求", "rsp": "响应", "data": "数据"}[network_type]
            button = dict(label=network_label, method="skip")
            network_buttons.append(button)

        # 添加两组按钮
        fig.update_layout(
            updatemenus=[
                # 第一组：平均/峰值按钮（左侧）
                dict(
                    buttons=mode_buttons,
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    active=0,  # 初始高亮平均使用率
                    x=0.3,
                    xanchor="center",
                    y=1.12,  # 提高位置避免与标题重叠
                    yanchor="top",
                    bgcolor="lightblue",
                    bordercolor="blue",
                    font=dict(size=12),
                    type="buttons",
                ),
                # 第二组：网络类型按钮（右侧）
                dict(
                    buttons=network_buttons,
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    active=2,  # 初始高亮数据
                    x=0.7,
                    xanchor="center",
                    y=1.12,  # 提高位置避免与标题重叠
                    yanchor="top",
                    bgcolor="lightblue",  # 改为蓝色
                    bordercolor="blue",
                    font=dict(size=12),
                    type="buttons",
                ),
            ]
        )

    def _short_name(self, name: str) -> str:
        """将IP名称缩写: gdma_0 -> G0, ddr_0 -> D0"""
        if "_" in name:
            prefix, idx = name.rsplit("_", 1)
            short = prefix[0].upper()
            return f"{short}{idx}"
        return name

    def _add_architecture_diagram(self, fig: go.Figure, fifo_options: List[Tuple[str, str, str, str]], dies: Dict):
        """
        在右侧子图添加CrossRing架构示意图
        根据KCIN版本自动选择v1或v2布局
        """
        # 获取第一个Die的配置
        first_die = list(dies.values())[0]
        config = first_die.config

        # 根据KCIN版本选择布局
        kcin_version = getattr(config, "KCIN_VERSION", "v1")
        if kcin_version == "v2":
            self._add_architecture_diagram_v2(fig, fifo_options, dies)
        else:
            self._add_architecture_diagram_v1(fig, fifo_options, dies)

    def _add_architecture_diagram_v1(self, fig: go.Figure, fifo_options: List[Tuple[str, str, str, str]], dies: Dict):
        """
        v1架构示意图（IQ/RB/EQ三模块布局）

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
        ch_width = 0.8
        ch_height = 1.6
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
            self._draw_fifo_item(fig, x_pos, dir_y, 1.2, 2.0, direction, ("IQ", direction), fifo_options)

        # EQ（右侧最上方，横向长方形）
        eq_x = iq_x + iq_w - 2.5
        eq_y = iq_y + iq_h - 3.5
        self._draw_fifo_item(fig, eq_x, eq_y, 2.2, 1.0, "EQ", ("IQ", "EQ"), fifo_options)

        # TU/TD（右侧下方，横向长方形）
        other_dirs = ["TU", "TD"]
        for i, direction in enumerate(other_dirs):
            x_pos = iq_x + iq_w - 2.5
            y_pos = iq_y + 2 + i * 2
            self._draw_fifo_item(fig, x_pos, y_pos, 2.2, 1.0, direction, ("IQ", direction), fifo_options)

        # === Ring Bridge (右下) ===
        rb_x, rb_y = 14, 0
        rb_w, rb_h = 10, 10
        self._draw_module_box(fig, rb_x, rb_y, rb_w, rb_h, "Ring Bridge", "lightyellow")

        # EQ (左上角，长方形)
        self._draw_fifo_item(fig, rb_x + 1, rb_y + rb_h - 2.5, 1.2, 2.0, "EQ", ("RB", "EQ"), fifo_options)

        # TL, TR (左侧下方水平排列，小长方形）
        for i, direction in enumerate(["TL", "TR"]):
            x_pos = rb_x + 1 + i * 2.2
            y_pos = rb_y + 1.5
            self._draw_fifo_item(fig, x_pos, y_pos, 1.2, 2.0, direction, ("RB", direction), fifo_options)

        # TU, TD (右上角垂直排列，小长方形)
        rb_right_dirs = ["TU", "TD"]
        for i, direction in enumerate(rb_right_dirs):
            x_pos = rb_x + rb_w - 2.5
            y_pos = rb_y + rb_h - 2.5 - i * 2.2
            self._draw_fifo_item(fig, x_pos, y_pos, 2.2, 1.0, direction, ("RB", direction), fifo_options)

        # === Eject Queue (右上) ===
        eq_x, eq_y = 14, 12
        eq_w, eq_h = 10, 8
        self._draw_module_box(fig, eq_x, eq_y, eq_w, eq_h, "Eject Queue", "lightcoral")

        # 通道缓冲（垂直排列在左侧，小长方形）
        eq_ch_start_y = eq_y + eq_h - 1.2
        eq_ch_x = eq_x + 1
        eq_ch_spacing = 0.8
        for i, ch_name in enumerate(ch_names):
            y_pos = eq_ch_start_y - i * eq_ch_spacing
            self._draw_fifo_item(fig, eq_ch_x, y_pos, 1.6, 0.8, ch_name, ("EQ_CH", ch_name), fifo_options)

        # TU, TD (右侧纵向，小长方形）
        for i, direction in enumerate(["TU", "TD"]):
            x_pos = eq_x + eq_w - 2.5
            y_pos = eq_y + eq_h - 2.5 - i * 2.2
            self._draw_fifo_item(fig, x_pos, y_pos, 2.2, 1.0, direction, ("EQ", direction), fifo_options)

        # 设置右侧子图的范围
        fig.update_xaxes(range=[-1, 26], row=1, col=2)
        fig.update_yaxes(range=[-1, 22], row=1, col=2)

    def _add_architecture_diagram_v2(self, fig: go.Figure, fifo_options: List[Tuple[str, str, str, str]], dies: Dict):
        """
        v2架构示意图（RingStation单框布局）

        布局（单个RS框）：
        - 左上角: IP channels (IQ_CH/EQ_CH) - 每个IP的I/O竖向排列
        - 下方: TL/TR方向 (IQ/RB) - 每个方向的I/O竖向排列
        - 右侧: TU/TD方向 (IQ/RB) - 每个方向的I/O横向排列
        """
        # 获取第一个Die的配置
        first_die = list(dies.values())[0]
        config = first_die.config
        ch_names = config.CH_NAME_LIST if hasattr(config, "CH_NAME_LIST") else ["gdma_0", "gdma_1", "ddr_0"]

        # === RS 框参数 (紧凑布局) ===
        rs_x, rs_y = 0, 0
        rs_padding = 0.5
        fifo_gap = 1.2  # 同方向I/O之间的间距
        dir_gap = 1.8   # 不同方向之间的间距
        item_w, item_h = 1.2, 1.5  # FIFO项尺寸

        # === 1. 左上角: IP channels (竖向) ===
        ip_section_x = rs_x + rs_padding
        ip_section_y = rs_y + 6

        for i, ch_name in enumerate(ch_names):
            base_x = ip_section_x + i * (fifo_gap + dir_gap)
            short_name = self._short_name(ch_name)

            # I (Input from IP) - RS_IN
            label_i = f"{short_name}I"
            self._draw_fifo_item(fig, base_x, ip_section_y, item_w, item_h, label_i, ("RS_IN", ch_name), fifo_options)

            # O (Output to IP) - RS_OUT
            label_o = f"{short_name}O"
            self._draw_fifo_item(fig, base_x + fifo_gap, ip_section_y, item_w, item_h, label_o, ("RS_OUT", ch_name), fifo_options)

        # === 2. 下方: TL/TR (竖向) ===
        h_section_x = rs_x + rs_padding
        h_section_y = rs_y + rs_padding

        for i, direction in enumerate(["TL", "TR"]):
            base_x = h_section_x + i * (fifo_gap + dir_gap)

            # I (Input from ring) - RS_IN
            label_i = f"{direction}I"
            self._draw_fifo_item(fig, base_x, h_section_y, item_w, item_h, label_i, ("RS_IN", direction), fifo_options)

            # O (Output to ring) - RS_OUT
            label_o = f"{direction}O"
            self._draw_fifo_item(fig, base_x + fifo_gap, h_section_y, item_w, item_h, label_o, ("RS_OUT", direction), fifo_options)

        # === 3. 右侧: TU/TD (横向) ===
        v_section_x = rs_x + max(len(ch_names), 2) * (fifo_gap + dir_gap) + rs_padding
        v_section_y = rs_y + rs_padding + 0.5

        h_item_w, h_item_h = 1.8, 1.0  # 横向FIFO项尺寸

        for i, direction in enumerate(["TU", "TD"]):
            base_y = v_section_y + i * (fifo_gap + dir_gap)

            # I (Input from ring) - RS_IN
            label_i = f"{direction}I"
            self._draw_fifo_item(fig, v_section_x, base_y, h_item_w, h_item_h, label_i, ("RS_IN", direction), fifo_options)

            # O (Output to ring) - RS_OUT
            label_o = f"{direction}O"
            self._draw_fifo_item(fig, v_section_x, base_y + fifo_gap, h_item_w, h_item_h, label_o, ("RS_OUT", direction), fifo_options)

        # === RS 外轮廓框 ===
        rs_w = v_section_x + h_item_w + rs_padding
        rs_h = ip_section_y + item_h + rs_padding + 0.3
        self._draw_module_box(fig, rs_x, rs_y, rs_w, rs_h, "RingStation", "lightblue")

        # 设置右侧子图的范围
        fig.update_xaxes(range=[-0.5, rs_w + 1], row=1, col=2)
        fig.update_yaxes(range=[-0.5, rs_h + 1], row=1, col=2)

    def _draw_module_box(self, fig: go.Figure, x: float, y: float, w: float, h: float, title: str, color: str):
        """绘制模块边框和标题"""
        # 绘制矩形框
        fig.add_shape(type="rect", x0=x, y0=y, x1=x + w, y1=y + h, line=dict(color="black", width=2), fillcolor=color, opacity=0.2, row=1, col=2)

        # 添加标题
        fig.add_annotation(x=x + w / 2, y=y + h + 0.3, text=title, showarrow=False, font=dict(size=14, color="black", family="Arial Black"), row=1, col=2)

    def _draw_fifo_item(self, fig: go.Figure, x: float, y: float, w: float, h: float, label: str, fifo_info: Tuple[str, str], fifo_options: List[Tuple[str, str, str, str]], text_angle: int = 0):
        """
        绘制单个FIFO项（可点击的矩形）

        Args:
            x, y, w, h: 位置和大小
            label: 显示标签
            fifo_info: (category, type) 用于匹配fifo_options
            fifo_options: 所有可用的FIFO选项列表（四元组：name, category, fifo_type, network_type）
            text_angle: 文本旋转角度（默认0度）
        """
        category, fifo_type = fifo_info

        # 检查这个FIFO是否在可用选项中（任何网络类型都算）
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

    def _generate_custom_javascript(self, fifo_options: List[Tuple[str, str, str, str]], num_dies: int) -> str:
        """
        生成FIFO热力图的自定义JavaScript代码

        Args:
            fifo_options: FIFO选项列表（四元组）
            num_dies: Die数量

        Returns:
            str: JavaScript代码字符串
        """
        # 创建FIFO选项映射 (用于JavaScript查找)
        # key: "category_fifo_type_network_type" -> index
        fifo_map = {}
        for idx, (name, category, fifo_type, network_type) in enumerate(fifo_options):
            key = f"{category}_{fifo_type}_{network_type}"
            fifo_map[key] = idx

        # 计算trace索引
        num_heatmap_traces = len(fifo_options) * 3

        # 创建FIFO选项的详细信息（用于JavaScript）
        fifo_options_js = [[opt[0], opt[1], opt[2], opt[3]] for opt in fifo_options]

        # 生成JavaScript代码（包含CSS样式）
        js_code = f"""
<style>
    /* 自定义按钮样式 */
    .updatemenu-button.active {{
        background-color: #3b82f6 !important;
        color: white !important;
        border: 2px solid #1d4ed8 !important;
        font-weight: bold !important;
    }}
    .updatemenu-button {{
        transition: all 0.2s ease !important;
    }}
</style>
<script>
    // FIFO选项映射（包含网络类型）
    const fifoMap = {str(fifo_map).replace("'", '"')};
    const fifoOptionsData = {str(fifo_options_js).replace("'", '"')};
    const numFifoOptions = {len(fifo_options)};
    const numDies = {num_dies};
    const numHeatmapTraces = {num_heatmap_traces};

    // 当前选中的状态
    let currentFifoIndex = 0;  // 默认第一个FIFO
    let currentMode = 'avg';  // 默认平均模式
    let currentNetworkType = fifoOptionsData.length > 0 ? fifoOptionsData[0][3] : 'data';  // 从第一个选项获取网络类型
    let currentCategory = fifoOptionsData.length > 0 ? fifoOptionsData[0][1] : null;  // 从第一个选项获取类别
    let currentFifoType = fifoOptionsData.length > 0 ? fifoOptionsData[0][2] : null;  // 从第一个选项获取FIFO类型

    // 等待Plotly加载完成
    document.addEventListener('DOMContentLoaded', function() {{
        setTimeout(function() {{
            const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            if (!plotDiv) return;

            // 初始化当前FIFO的category和type
            updateCurrentFifoInfo();

            // 监听架构图点击事件
            plotDiv.on('plotly_click', function(data) {{
                const clickedPoint = data.points[0];
                const traceIndex = clickedPoint.curveNumber;
                console.log('点击trace索引:', traceIndex, '热力图trace数:', numHeatmapTraces);

                if (traceIndex >= numHeatmapTraces) {{
                    const customdata = clickedPoint.customdata;
                    console.log('customdata:', customdata);
                    if (customdata && customdata.length >= 2) {{
                        const category = customdata[0];
                        const fifoType = customdata[1];
                        console.log('点击FIFO:', category, fifoType);

                        // 使用当前选中的网络类型
                        const key = category + '_' + fifoType + '_' + currentNetworkType;
                        console.log('查找key:', key);

                        let fifoIndex = fifoMap[key];
                        if (fifoIndex === undefined) {{
                            // 如果当前网络类型不存在，尝试其他网络
                            for (let net of ['data', 'req', 'rsp']) {{
                                const tryKey = category + '_' + fifoType + '_' + net;
                                if (fifoMap[tryKey] !== undefined) {{
                                    fifoIndex = fifoMap[tryKey];
                                    currentNetworkType = net;
                                    updateNetworkButtonHighlight();
                                    console.log('在网络', net, '找到FIFO, index:', fifoIndex);
                                    break;
                                }}
                            }}
                        }} else {{
                            console.log('找到FIFO index:', fifoIndex);
                        }}
                        if (fifoIndex !== undefined) {{
                            currentFifoIndex = fifoIndex;
                            updateCurrentFifoInfo();
                            updateHeatmapVisibility();
                        }} else {{
                            console.warn('未找到FIFO:', category, fifoType);
                        }}
                    }}
                }}
            }});

            // 更新当前FIFO的信息
            function updateCurrentFifoInfo() {{
                if (currentFifoIndex >= 0 && currentFifoIndex < fifoOptionsData.length) {{
                    const option = fifoOptionsData[currentFifoIndex];
                    currentCategory = option[1];
                    currentFifoType = option[2];
                    currentNetworkType = option[3];
                }}
            }}

            // 更新热力图可见性和架构图高亮
            function updateHeatmapVisibility() {{
                const update = {{}};
                const visibility = [];

                // 计算哪些traces应该可见（每个FIFO选项+模式组合1个trace）
                for (let i = 0; i < numFifoOptions; i++) {{
                    for (let mode of ['avg', 'peak', 'flit_count']) {{
                        const shouldShow = (i === currentFifoIndex && mode === currentMode);
                        visibility.push(shouldShow);
                    }}
                }}

                // 架构图的traces保持可见
                for (let i = numHeatmapTraces; i < plotDiv.data.length; i++) {{
                    visibility.push(true);
                }}

                update.visible = visibility;
                console.log('更新trace可见性:', visibility.filter(v => v).length, '个可见,', 'currentFifoIndex:', currentFifoIndex, 'currentMode:', currentMode);
                Plotly.restyle(plotDiv, update);

                // 更新架构图高亮
                updateArchitectureHighlight();
            }}

            // 更新架构图高亮
            function updateArchitectureHighlight() {{
                const shapes = plotDiv.layout.shapes || [];
                const expectedName = currentCategory + '_' + currentFifoType;

                const newShapes = shapes.map((shape, idx) => {{
                    // 跳过没有name的shape（模块边框没有name属性）
                    if (!shape.name) {{
                        return shape;
                    }}

                    // 检查是否为当前选中的FIFO（只比较category和fifo_type）
                    const shapeName = shape.name;
                    const isSelected = (shapeName === expectedName);

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

            // 等待按钮渲染完成后绑定事件
            function setupButtonListeners() {{
                const allButtons = plotDiv.querySelectorAll('.updatemenu-button');
                console.log('找到按钮数量:', allButtons.length);

                if (allButtons.length < 6) {{
                    console.warn('按钮未完全渲染，重试...');
                    setTimeout(setupButtonListeners, 200);
                    return;
                }}

                // 第一组：平均/峰值/Flit计数按钮（前3个）
                const modeButtons = Array.from(allButtons).slice(0, 3);
                // 第二组：网络类型按钮（后3个）
                const networkButtons = Array.from(allButtons).slice(3, 6);
                console.log('模式按钮数量:', modeButtons.length, '网络按钮数量:', networkButtons.length);

                // 监听平均/峰值/Flit计数按钮点击
                modeButtons.forEach((btn, idx) => {{
                    btn.addEventListener('click', function(e) {{
                        const modeNames = ['avg', 'peak', 'flit_count'];
                        console.log('点击模式按钮:', modeNames[idx]);
                        setTimeout(() => {{
                            // 移除同组按钮的active类
                            modeButtons.forEach(b => b.classList.remove('active'));
                            // 添加到当前按钮
                            this.classList.add('active');

                            currentMode = modeNames[idx];
                            updateHeatmapVisibility();
                        }}, 10);
                    }});
                }});

                // 监听网络类型按钮点击
                networkButtons.forEach((btn, idx) => {{
                    btn.addEventListener('click', function(e) {{
                        const networks = ['req', 'rsp', 'data'];
                        console.log('点击网络按钮:', networks[idx]);
                        setTimeout(() => {{
                            // 移除同组按钮的active类
                            networkButtons.forEach(b => b.classList.remove('active'));
                            // 添加到当前按钮
                            this.classList.add('active');

                            currentNetworkType = networks[idx];

                            // 切换到当前FIFO在新网络中的对应项
                            switchToNetwork(currentNetworkType);
                        }}, 10);
                    }});
                }});

                // 初始化按钮高亮状态
                if (modeButtons.length > 0) {{
                    modeButtons[0].classList.add('active');  // 平均
                }}
                if (networkButtons.length > 0) {{
                    networkButtons[2].classList.add('active');  // Data
                }}
            }}

            // 启动按钮监听器设置
            setupButtonListeners();

            // 切换到指定网络类型
            function switchToNetwork(networkType) {{
                if (!currentCategory || !currentFifoType) return;

                const key = currentCategory + '_' + currentFifoType + '_' + networkType;
                const fifoIndex = fifoMap[key];

                if (fifoIndex !== undefined) {{
                    currentFifoIndex = fifoIndex;
                    updateCurrentFifoInfo();
                    updateHeatmapVisibility();
                }} else {{
                    console.warn('FIFO not found for network:', networkType);
                }}
            }}

            // 更新网络按钮高亮
            function updateNetworkButtonHighlight() {{
                const allButtons = plotDiv.querySelectorAll('.updatemenu-button');
                const networkButtons = Array.from(allButtons).slice(3, 6);  // 网络按钮是第4-6个(索引3-5)
                networkButtons.forEach(b => b.classList.remove('active'));
                const networks = ['req', 'rsp', 'data'];
                const netIdx = networks.indexOf(currentNetworkType);
                if (netIdx >= 0 && netIdx < networkButtons.length) {{
                    networkButtons[netIdx].classList.add('active');
                }}
            }}

            // 初始化时高亮默认FIFO
            updateArchitectureHighlight();
        }}, 500);  // 延迟500ms确保Plotly完全加载
    }});
</script>
"""
        return js_code

    def _save_html_with_click_events(self, fig: go.Figure, save_path: str, fifo_options: List[Tuple[str, str, str, str]], num_dies: int):
        """
        保存HTML文件并注入JavaScript代码来处理FIFO点击事件

        Args:
            fig: Plotly图形对象
            save_path: 保存路径
            fifo_options: FIFO选项列表（四元组）
            num_dies: Die数量
        """
        # 先生成基础HTML（使用本地内嵌Plotly，避免CDN加载失败）
        html_string = fig.to_html(include_plotlyjs=True)

        # 生成自定义JavaScript代码
        js_code = self._generate_custom_javascript(fifo_options, num_dies)

        # 在</body>之前注入JavaScript
        html_string = html_string.replace("</body>", js_code + "</body>")

        # 保存修改后的HTML
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_string)


def create_fifo_heatmap(dies: Dict, config, total_cycles: int, die_layout: Optional[Dict] = None, die_rotations: Optional[Dict] = None, save_path: Optional[str] = None, show_fig: bool = False, return_fig_and_js: bool = False):
    """
    便捷函数：一键创建FIFO使用率热力图

    Args:
        dies: Die字典
        config: 配置对象
        total_cycles: 仿真总周期数
        die_layout: Die布局
        die_rotations: Die旋转
        save_path: 保存路径
        show_fig: 是否在浏览器中显示
        return_fig_and_js: 是否返回(Figure, JavaScript)元组

    Returns:
        str or tuple: 如果return_fig_and_js=True，返回(Figure, JavaScript)；否则返回保存路径
    """
    # 收集数据
    collector = FIFOUtilizationCollector(config)
    fifo_data = collector.collect_from_dies(dies, total_cycles)

    # 创建可视化
    visualizer = FIFOHeatmapVisualizer(config, fifo_data)
    return visualizer.create_interactive_heatmap(dies, die_layout, die_rotations, save_path, show_fig, return_fig_and_js)
