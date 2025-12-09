"""
层级结构图渲染器

使用 Plotly 生成交互式层级结构可视化
"""

from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 层级颜色定义
LEVEL_COLORS = {
    "pod": "#1890ff",      # 蓝色
    "server": "#52c41a",   # 绿色
    "board": "#fa8c16",    # 橙色
    "chip": "#722ed1",     # 紫色
    "die": "#f5222d",      # 红色
}

# 连接类型颜色
CONNECTION_COLORS = {
    "d2d": "#13c2c2",  # 青色
    "c2c": "#722ed1",  # 紫色
    "b2b": "#fa8c16",  # 橙色
    "s2s": "#52c41a",  # 绿色
    "p2p": "#1890ff",  # 蓝色
}


class HierarchyGraphRenderer:
    """层级结构图渲染器"""

    def __init__(self):
        self.nodes = []
        self.edges = []

    def render_treemap(self, hierarchy_data: Dict) -> go.Figure:
        """
        渲染树状图 (Treemap)

        Args:
            hierarchy_data: 层级数据字典

        Returns:
            Plotly Figure
        """
        ids = []
        labels = []
        parents = []
        values = []
        colors = []

        def traverse(node: Dict, parent_id: str = ""):
            node_id = node["id"]
            level = node["level"]

            ids.append(node_id)
            labels.append(f"{level.upper()}\n{node_id.split('_')[-1]}")
            parents.append(parent_id)
            values.append(1)
            colors.append(LEVEL_COLORS.get(level, "#999999"))

            for child_id, child in node.get("children", {}).items():
                traverse(child, node_id)

        traverse(hierarchy_data)

        fig = go.Figure(go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            textinfo="label",
            hovertemplate="<b>%{label}</b><br>ID: %{id}<extra></extra>",
            branchvalues="total",
        ))

        fig.update_layout(
            title="Tier6+ 层级结构图 (Treemap)",
            margin=dict(t=50, l=25, r=25, b=25),
        )

        return fig

    def render_sunburst(self, hierarchy_data: Dict) -> go.Figure:
        """
        渲染旭日图 (Sunburst)

        Args:
            hierarchy_data: 层级数据字典

        Returns:
            Plotly Figure
        """
        ids = []
        labels = []
        parents = []
        values = []
        colors = []

        def traverse(node: Dict, parent_id: str = ""):
            node_id = node["id"]
            level = node["level"]

            ids.append(node_id)
            labels.append(f"{level}")
            parents.append(parent_id)
            values.append(1)
            colors.append(LEVEL_COLORS.get(level, "#999999"))

            for child_id, child in node.get("children", {}).items():
                traverse(child, node_id)

        traverse(hierarchy_data)

        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>ID: %{id}<extra></extra>",
        ))

        fig.update_layout(
            title="Tier6+ 层级结构图 (Sunburst)",
            margin=dict(t=50, l=25, r=25, b=25),
        )

        return fig

    def render_nested_boxes(self, hierarchy_data: Dict) -> go.Figure:
        """
        渲染嵌套盒子图

        Args:
            hierarchy_data: 层级数据字典

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        # 计算布局
        shapes = []
        annotations = []

        def calculate_layout(node: Dict, x: float, y: float, width: float, height: float, depth: int = 0):
            level = node["level"]
            color = LEVEL_COLORS.get(level, "#999999")

            # 添加矩形
            shapes.append(dict(
                type="rect",
                x0=x, y0=y,
                x1=x + width, y1=y + height,
                line=dict(color=color, width=2),
                fillcolor=color,
                opacity=0.1 + depth * 0.1,
                layer="below",
            ))

            # 添加标签
            annotations.append(dict(
                x=x + width / 2,
                y=y + height - 0.02,
                text=f"<b>{level.upper()}</b>",
                showarrow=False,
                font=dict(size=12 - depth, color=color),
            ))

            # 处理子节点
            children = list(node.get("children", {}).values())
            if children:
                padding = 0.05
                inner_x = x + padding * width
                inner_y = y + padding * height
                inner_width = width * (1 - 2 * padding)
                inner_height = height * (1 - 2 * padding) * 0.85  # 留空间给标签

                # 子节点网格布局
                n_children = len(children)
                cols = min(n_children, 4)
                rows = (n_children + cols - 1) // cols

                child_width = inner_width / cols * 0.9
                child_height = inner_height / rows * 0.9
                gap_x = inner_width / cols * 0.1
                gap_y = inner_height / rows * 0.1

                for i, child in enumerate(children):
                    row = i // cols
                    col = i % cols
                    cx = inner_x + col * (child_width + gap_x)
                    cy = inner_y + (rows - 1 - row) * (child_height + gap_y)
                    calculate_layout(child, cx, cy, child_width, child_height, depth + 1)

        calculate_layout(hierarchy_data, 0, 0, 1, 1)

        fig.update_layout(
            shapes=shapes,
            annotations=annotations,
            title="Tier6+ 层级结构图 (嵌套盒子)",
            xaxis=dict(visible=False, range=[-0.05, 1.05]),
            yaxis=dict(visible=False, range=[-0.05, 1.05], scaleanchor="x"),
            width=800,
            height=600,
            margin=dict(t=50, l=25, r=25, b=25),
        )

        # 添加图例
        for level, color in LEVEL_COLORS.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=15, color=color),
                name=level.upper(),
                showlegend=True,
            ))

        return fig

    def render_network_graph(self, hierarchy_data: Dict) -> go.Figure:
        """
        渲染网络拓扑图

        Args:
            hierarchy_data: 层级数据字典

        Returns:
            Plotly Figure
        """
        nodes = []
        edges = []
        node_positions = {}

        def collect_nodes(node: Dict, level_y: float, x_offset: float, x_range: float):
            node_id = node["id"]
            level = node["level"]

            # 计算位置
            children = list(node.get("children", {}).values())
            n_children = len(children) if children else 1

            x = x_offset + x_range / 2
            y = level_y

            node_positions[node_id] = (x, y)
            nodes.append({
                "id": node_id,
                "level": level,
                "x": x,
                "y": y,
            })

            # 递归子节点
            if children:
                child_range = x_range / n_children
                for i, child in enumerate(children):
                    child_x_offset = x_offset + i * child_range
                    collect_nodes(child, level_y - 1, child_x_offset, child_range)

                    # 添加边
                    edges.append({
                        "source": node_id,
                        "target": child["id"],
                    })

        collect_nodes(hierarchy_data, 5, 0, 10)

        # 创建图形
        fig = go.Figure()

        # 绘制边
        for edge in edges:
            x0, y0 = node_positions[edge["source"]]
            x1, y1 = node_positions[edge["target"]]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(color="#999999", width=1),
                hoverinfo="none",
                showlegend=False,
            ))

        # 绘制节点
        for level in LEVEL_COLORS.keys():
            level_nodes = [n for n in nodes if n["level"] == level]
            if level_nodes:
                fig.add_trace(go.Scatter(
                    x=[n["x"] for n in level_nodes],
                    y=[n["y"] for n in level_nodes],
                    mode="markers+text",
                    marker=dict(
                        size=30,
                        color=LEVEL_COLORS[level],
                        line=dict(width=2, color="white"),
                    ),
                    text=[level.upper() for _ in level_nodes],
                    textposition="middle center",
                    textfont=dict(color="white", size=10),
                    name=level.upper(),
                    hovertemplate="<b>%{text}</b><br>ID: %{customdata}<extra></extra>",
                    customdata=[n["id"] for n in level_nodes],
                ))

        fig.update_layout(
            title="Tier6+ 层级网络拓扑图",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=1000,
            height=700,
            margin=dict(t=50, l=25, r=25, b=25),
            showlegend=True,
        )

        return fig

    def render(
        self,
        hierarchy_data: Dict,
        style: str = "treemap"
    ) -> go.Figure:
        """
        渲染层级结构图

        Args:
            hierarchy_data: 层级数据字典
            style: 样式 ('treemap', 'sunburst', 'nested', 'network')

        Returns:
            Plotly Figure
        """
        if style == "treemap":
            return self.render_treemap(hierarchy_data)
        elif style == "sunburst":
            return self.render_sunburst(hierarchy_data)
        elif style == "nested":
            return self.render_nested_boxes(hierarchy_data)
        elif style == "network":
            return self.render_network_graph(hierarchy_data)
        else:
            raise ValueError(f"不支持的样式: {style}")

    def save_html(self, fig: go.Figure, path: str):
        """保存为 HTML 文件"""
        fig.write_html(path)
