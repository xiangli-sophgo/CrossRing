"""
拓扑可视化模块 - 基于Plotly的交互式拓扑网格

提供拓扑网格绘制和交互式节点选择功能
"""

import plotly.graph_objects as go
from typing import Dict, List, Set, Tuple


# IP颜色映射(复用analyzers.py的配色方案)
IP_COLOR_MAP = {
    "gdma": "#4472C4",  # 蓝色
    "sdma": "#ED7D31",  # 橙色
    "cdma": "#70AD47",  # 绿色
    "ddr": "#C00000",   # 红色
    "l2m": "#7030A0",   # 紫色
}

# 节点选择状态颜色
SELECT_COLOR = {
    "source": "#87CEEB",     # 天蓝色 - 源节点
    "destination": "#FFB6C1", # 粉色 - 目标节点
    "both": "#DDA0DD",       # 梅红色 - 既是源又是目标
    "default": "#F0F0F0",    # 浅灰色 - 未选中的普通节点
    "ip_default": "#FFFFFF",  # 白色 - 未选中的IP节点
}


class TopologyVisualizer:
    """拓扑可视化器"""

    def __init__(self, topo_type="5x4", ip_mappings=None):
        """
        :param topo_type: 拓扑类型 "5x4" 或 "4x4"
        :param ip_mappings: IP位置映射字典
        """
        self.topo_type = topo_type
        self.rows, self.cols = map(int, topo_type.split('x'))
        self.num_nodes = self.rows * self.cols
        self.ip_mappings = ip_mappings or {}

    def get_node_position(self, node_id: int) -> Tuple[float, float]:
        """
        计算节点在图中的位置坐标

        :param node_id: 节点ID
        :return: (x, y) 坐标
        """
        row = node_id // self.cols
        col = node_id % self.cols
        # y坐标翻转,使得节点0在左上角
        return (col, self.rows - 1 - row)

    def get_node_color(
        self,
        node_id: int,
        selected_src: Set[int],
        selected_dst: Set[int]
    ) -> str:
        """
        获取节点颜色(根据IP类型和选中状态)

        :param node_id: 节点ID
        :param selected_src: 选中的源节点集合
        :param selected_dst: 选中的目标节点集合
        :return: 颜色代码
        """
        # 检查是否是IP节点
        ip_type = self.get_node_ip_type(node_id)

        # 检查选中状态
        is_src = node_id in selected_src
        is_dst = node_id in selected_dst

        # 优先级: 选中状态 > IP类型
        if is_src and is_dst:
            return SELECT_COLOR["both"]
        elif is_src:
            return SELECT_COLOR["source"]
        elif is_dst:
            return SELECT_COLOR["destination"]
        elif ip_type:
            return IP_COLOR_MAP.get(ip_type, SELECT_COLOR["ip_default"])
        else:
            return SELECT_COLOR["default"]

    def get_node_ip_type(self, node_id: int) -> str:
        """
        获取节点的IP类型

        :param node_id: 节点ID
        :return: IP类型字符串,如果不是IP节点则返回空字符串
        """
        for ip_type, positions in self.ip_mappings.items():
            if node_id in positions:
                return ip_type
        return ""

    def get_node_label(self, node_id: int) -> str:
        """
        获取节点显示标签

        :param node_id: 节点ID
        :return: 标签字符串
        """
        ip_type = self.get_node_ip_type(node_id)
        if ip_type:
            return f"{node_id}<br>{ip_type.upper()}"
        return str(node_id)

    def draw_topology_grid(
        self,
        selected_src: Set[int] = None,
        selected_dst: Set[int] = None,
        show_legend: bool = True,
        node_ips: Dict[int, List[str]] = None
    ) -> go.Figure:
        """
        绘制交互式拓扑网格

        :param selected_src: 选中的源节点集合
        :param selected_dst: 选中的目标节点集合
        :param show_legend: 是否显示图例
        :param node_ips: 节点挂载的IP列表 {node_id: [ip_list]}
        :return: Plotly Figure对象
        """
        selected_src = selected_src or set()
        selected_dst = selected_dst or set()
        node_ips = node_ips or {}

        # 创建图形
        fig = go.Figure()

        # 绘制节点矩形（不使用scatter避免圆圈）
        for node_id in range(self.num_nodes):
            x, y = self.get_node_position(node_id)
            color = self.get_node_color(node_id, selected_src, selected_dst)

            # 绘制节点矩形框
            fig.add_shape(
                type="rect",
                x0=x - 0.35, y0=y - 0.35,
                x1=x + 0.35, y1=y + 0.35,
                fillcolor=color,
                line=dict(color="black", width=2),
                opacity=0.8
            )

            # 添加节点ID文本（黑色，无描边）
            fig.add_annotation(
                x=x,
                y=y,
                text=str(node_id),
                showarrow=False,
                font=dict(size=14, color="black", family="Arial Black"),
                xanchor="center",
                yanchor="middle"
            )

        # 添加透明scatter用于点击事件捕获
        node_x = [self.get_node_position(i)[0] for i in range(self.num_nodes)]
        node_y = [self.get_node_position(i)[1] for i in range(self.num_nodes)]

        # hover文本
        hover_texts = []
        for node_id in range(self.num_nodes):
            ips = node_ips.get(node_id, [])
            hover_text = f"节点 {node_id}"
            if ips:
                hover_text += f"<br>挂载IP: {', '.join(ips)}"
            hover_texts.append(hover_text)

        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(
                size=40,
                color='rgba(0,0,0,0)',  # 完全透明
                line=dict(width=0)
            ),
            hovertext=hover_texts,
            hoverinfo="text",
            customdata=list(range(self.num_nodes)),
            showlegend=False,
            name="节点"
        ))

        # 添加网格线
        self._add_grid_lines(fig)

        # 在节点内部添加IP小方块
        self._add_ip_markers(fig, node_ips)

        # 更新布局
        fig.update_layout(
            title=dict(
                text=f"{self.topo_type} NoC拓扑结构",
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                autorange=True
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x",
                scaleratio=1,
                autorange=True
            ),
            width=1000,
            height=800,
            hovermode='closest',
            showlegend=show_legend,
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=80)  # 下方留空给图例
        )

        # 添加图例说明
        if show_legend:
            self._add_legend_annotations(fig, node_ips)

        return fig

    def _add_grid_lines(self, fig: go.Figure):
        """添加网格线"""
        # 垂直线
        for i in range(self.cols + 1):
            fig.add_shape(
                type="line",
                x0=i - 0.5, y0=-0.5,
                x1=i - 0.5, y1=self.rows - 0.5,
                line=dict(color="lightgray", width=1)
            )

        # 水平线
        for i in range(self.rows + 1):
            fig.add_shape(
                type="line",
                x0=-0.5, y0=i - 0.5,
                x1=self.cols - 0.5, y1=i - 0.5,
                line=dict(color="lightgray", width=1)
            )

    def _add_ip_markers(self, fig: go.Figure, node_ips: Dict[int, List[str]]):
        """在节点内部添加IP小方块标记，分两行显示：第一行RN类型，第二行SN类型"""
        if not node_ips:
            return

        # RN类型（所有dma）
        rn_types = {'gdma', 'sdma', 'cdma'}
        # SN类型（ddr、l2m等）
        sn_types = {'ddr', 'l2m'}

        for node_id, ip_list in node_ips.items():
            if not ip_list:
                continue

            # 获取节点位置
            x, y = self.get_node_position(node_id)

            # 分类并排序IP
            rn_ips = []
            sn_ips = []

            for ip in ip_list:
                ip_type = ip.split('_')[0].lower() if '_' in ip else ip.lower()
                if ip_type in rn_types:
                    rn_ips.append(ip)
                elif ip_type in sn_types:
                    sn_ips.append(ip)

            # 按编号排序（提取_后面的数字）
            def get_ip_index(ip):
                parts = ip.split('_')
                return int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

            rn_ips.sort(key=get_ip_index)
            sn_ips.sort(key=get_ip_index)

            # IP方块大小
            marker_size = 0.22
            spacing = 0.25  # 方块之间的间距

            # 第一行：RN类型（上方）
            if rn_ips:
                row_y = y + 0.15  # 上方位置
                self._draw_ip_row(fig, rn_ips, x, row_y, marker_size, spacing)

            # 第二行：SN类型（下方）
            if sn_ips:
                row_y = y - 0.15  # 下方位置
                self._draw_ip_row(fig, sn_ips, x, row_y, marker_size, spacing)

    def _draw_ip_row(self, fig: go.Figure, ip_list: List[str], center_x: float, row_y: float, marker_size: float, spacing: float):
        """绘制一行IP标记"""
        num_ips = len(ip_list)
        if num_ips == 0:
            return

        # 计算起始X位置（居中对齐）
        total_width = (num_ips - 1) * spacing
        start_x = center_x - total_width / 2

        for i, ip in enumerate(ip_list):
            marker_x = start_x + i * spacing

            # 获取IP类型和颜色
            ip_type = ip.split('_')[0] if '_' in ip else ip
            ip_color = IP_COLOR_MAP.get(ip_type.lower(), "#808080")

            # 绘制小方块
            fig.add_shape(
                type="rect",
                x0=marker_x - marker_size/2,
                y0=row_y - marker_size/2,
                x1=marker_x + marker_size/2,
                y1=row_y + marker_size/2,
                fillcolor=ip_color,
                line=dict(color="black", width=2),
                opacity=0.9
            )

            # 添加IP标签
            ip_label = self._get_ip_short_label(ip)
            fig.add_annotation(
                x=marker_x,
                y=row_y,
                text=ip_label,
                showarrow=False,
                font=dict(size=7, color="white", family="Arial Black"),
                xanchor="center",
                yanchor="middle"
            )

    def _get_ip_short_label(self, ip: str) -> str:
        """获取IP的简短标签"""
        # 例如: "gdma_0" -> "G0", "ddr_1" -> "D1"
        parts = ip.split('_')
        if len(parts) == 2:
            type_name = parts[0]
            index = parts[1]
            return f"{type_name[0].upper()}{index}"
        return ip[:2].upper()

    def _add_legend_annotations(self, fig: go.Figure, node_ips: Dict[int, List[str]]):
        """添加图例标注，只显示已挂载的IP类型"""
        # 提取所有已挂载的IP类型
        mounted_ip_types = set()
        for ips in node_ips.values():
            for ip in ips:
                ip_type = ip.split('_')[0] if '_' in ip else ip
                mounted_ip_types.add(ip_type.lower())

        # 只显示已挂载的IP类型
        legend_items = []

        # IP类型标签映射
        ip_type_labels = {
            "gdma": "GDMA",
            "ddr": "DDR",
            "l2m": "L2M",
            "sdma": "SDMA",
            "cdma": "CDMA",
        }

        for ip_type in sorted(mounted_ip_types):
            if ip_type in IP_COLOR_MAP:
                label = ip_type_labels.get(ip_type, ip_type.upper())
                legend_items.append((label, IP_COLOR_MAP[ip_type]))

        # 在图的下方添加图例（水平排列）
        if not legend_items:
            return

        num_items = len(legend_items)
        item_width = 1.0  # 每个图例项的宽度
        total_width = num_items * item_width
        x_start = (self.cols - total_width) / 2  # 居中
        y_pos = -0.8  # 下方位置

        for i, (label, color) in enumerate(legend_items):
            x_pos = x_start + i * item_width

            # 添加颜色方块
            fig.add_shape(
                type="rect",
                x0=x_pos, y0=y_pos - 0.12,
                x1=x_pos + 0.24, y1=y_pos + 0.12,
                fillcolor=color,
                line=dict(color="black", width=1)
            )

            # 添加文字标签
            fig.add_annotation(
                x=x_pos + 0.35,
                y=y_pos,
                text=label,
                showarrow=False,
                xanchor="left",
                font=dict(size=10, color="black")
            )

    def parse_click_data(self, click_data: dict) -> int:
        """
        解析Plotly点击事件数据,提取节点ID

        :param click_data: Streamlit返回的click_data字典
        :return: 节点ID
        """
        if not click_data or 'points' not in click_data:
            return None

        points = click_data['points']
        if not points:
            return None

        # 从customdata获取节点ID
        node_id = points[0].get('customdata')
        return node_id

    def parse_node_ids(self, node_input: str) -> List[int]:
        """
        解析节点ID输入字符串

        支持格式:
        - 单个节点: "6"
        - 逗号分隔: "6,7,26,27"
        - 范围表达: "6-7,26-27"
        - 混合格式: "6,8-10,12"

        :param node_input: 节点ID输入字符串
        :return: 节点ID列表
        :raises ValueError: 如果输入格式错误或节点ID超出范围
        """
        if not node_input or not node_input.strip():
            return []

        node_ids = set()

        # 按逗号分隔
        parts = node_input.split(',')

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # 检查是否是范围表达式
            if '-' in part:
                # 范围: "6-7"
                try:
                    start, end = part.split('-')
                    start_id = int(start.strip())
                    end_id = int(end.strip())

                    if start_id > end_id:
                        raise ValueError(f"无效范围: {part} (起始值大于结束值)")

                    for nid in range(start_id, end_id + 1):
                        if nid < 0 or nid >= self.num_nodes:
                            raise ValueError(f"节点ID {nid} 超出范围 [0, {self.num_nodes - 1}]")
                        node_ids.add(nid)

                except ValueError as e:
                    if "invalid literal" in str(e):
                        raise ValueError(f"无效范围格式: {part}")
                    raise
            else:
                # 单个节点
                try:
                    nid = int(part)
                    if nid < 0 or nid >= self.num_nodes:
                        raise ValueError(f"节点ID {nid} 超出范围 [0, {self.num_nodes - 1}]")
                    node_ids.add(nid)
                except ValueError:
                    raise ValueError(f"无效节点ID: {part}")

        return sorted(list(node_ids))


def get_default_ip_mappings(topo_type="5x4") -> Dict[str, List[int]]:
    """
    获取默认IP位置映射

    :param topo_type: 拓扑类型 ("5x4" 或 "4x4")
    :return: IP映射字典
    """
    if topo_type == "5x4":
        return {
            "gdma": [6, 7, 26, 27],
            "ddr": [12, 13, 32, 33],
            "l2m": [18, 19, 38, 39],
            "sdma": [0, 1, 20, 21],
            "cdma": [14, 15, 34],
        }
    elif topo_type == "4x4":
        return {
            "gdma": [0, 1, 2, 3],
            "ddr": [12, 13, 14, 15],
            "l2m": [8, 9, 10, 11],
            "sdma": [4, 5, 6, 7],
            "cdma": [14, 15],
        }
    else:
        raise ValueError(f"不支持的拓扑类型: {topo_type}")
