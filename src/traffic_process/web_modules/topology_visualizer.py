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
        show_legend: bool = True
    ) -> go.Figure:
        """
        绘制交互式拓扑网格

        :param selected_src: 选中的源节点集合
        :param selected_dst: 选中的目标节点集合
        :param show_legend: 是否显示图例
        :return: Plotly Figure对象
        """
        selected_src = selected_src or set()
        selected_dst = selected_dst or set()

        # 准备节点数据
        node_x = []
        node_y = []
        node_colors = []
        node_labels = []
        node_texts = []  # hover显示文本

        for node_id in range(self.num_nodes):
            x, y = self.get_node_position(node_id)
            node_x.append(x)
            node_y.append(y)

            color = self.get_node_color(node_id, selected_src, selected_dst)
            node_colors.append(color)

            label = self.get_node_label(node_id)
            node_labels.append(label)

            # hover文本
            ip_type = self.get_node_ip_type(node_id)
            status = []
            if node_id in selected_src:
                status.append("源节点")
            if node_id in selected_dst:
                status.append("目标节点")

            hover_text = f"节点 {node_id}"
            if ip_type:
                hover_text += f"<br>类型: {ip_type.upper()}"
            if status:
                hover_text += f"<br>状态: {', '.join(status)}"

            node_texts.append(hover_text)

        # 创建节点散点图
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=40,
                color=node_colors,
                line=dict(width=2, color='black')
            ),
            text=node_labels,
            textposition="middle center",
            textfont=dict(size=10, color='black'),
            hovertext=node_texts,
            hoverinfo="text",
            customdata=list(range(self.num_nodes)),  # 存储节点ID用于点击事件
            name="节点"
        ))

        # 添加网格线
        self._add_grid_lines(fig)

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
                range=[-0.5, self.cols - 0.5]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.5, self.rows - 0.5],
                scaleanchor="x",
                scaleratio=1
            ),
            width=800,
            height=600,
            hovermode='closest',
            showlegend=show_legend,
            plot_bgcolor='white'
        )

        # 添加图例说明
        if show_legend:
            self._add_legend_annotations(fig)

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

    def _add_legend_annotations(self, fig: go.Figure):
        """添加图例标注"""
        legend_items = [
            ("源节点", SELECT_COLOR["source"]),
            ("目标节点", SELECT_COLOR["destination"]),
            ("GDMA", IP_COLOR_MAP["gdma"]),
            ("DDR", IP_COLOR_MAP["ddr"]),
            ("L2M", IP_COLOR_MAP["l2m"]),
            ("SDMA", IP_COLOR_MAP["sdma"]),
            ("CDMA", IP_COLOR_MAP["cdma"]),
        ]

        # 在图的右侧添加图例
        x_start = self.cols + 0.5
        y_start = self.rows - 1

        for i, (label, color) in enumerate(legend_items):
            y_pos = y_start - i * 0.6

            # 添加颜色方块
            fig.add_shape(
                type="rect",
                x0=x_start, y0=y_pos - 0.2,
                x1=x_start + 0.4, y1=y_pos + 0.2,
                fillcolor=color,
                line=dict(color="black", width=1)
            )

            # 添加文字标签
            fig.add_annotation(
                x=x_start + 0.6,
                y=y_pos,
                text=label,
                showarrow=False,
                xanchor="left",
                font=dict(size=10)
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
