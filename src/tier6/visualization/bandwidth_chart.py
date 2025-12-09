"""
带宽瓶颈图渲染器

使用 Plotly 生成带宽分析可视化
"""

from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 层级颜色
LEVEL_COLORS = {
    "die": "#f5222d",
    "chip": "#722ed1",
    "board": "#fa8c16",
    "server": "#52c41a",
    "pod": "#1890ff",
}


class BandwidthBottleneckChart:
    """带宽瓶颈图渲染器"""

    def render_utilization_bar(self, bandwidth_data: Dict) -> go.Figure:
        """
        渲染利用率条形图

        Args:
            bandwidth_data: 带宽数据
                {location: {theoretical_bandwidth_gbps, effective_bandwidth_gbps, utilization}}

        Returns:
            Plotly Figure
        """
        locations = list(bandwidth_data.keys())
        utilizations = [bandwidth_data[l].get("utilization", 0) * 100 for l in locations]
        theoretical = [bandwidth_data[l].get("theoretical_bandwidth_gbps", 0) for l in locations]
        effective = [bandwidth_data[l].get("effective_bandwidth_gbps", 0) for l in locations]

        # 根据利用率设置颜色
        colors = []
        for u in utilizations:
            if u >= 90:
                colors.append("#f5222d")  # 红色 - 临界
            elif u >= 70:
                colors.append("#fa8c16")  # 橙色 - 警告
            elif u >= 50:
                colors.append("#fadb14")  # 黄色 - 注意
            else:
                colors.append("#52c41a")  # 绿色 - 正常

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=locations,
            y=utilizations,
            marker_color=colors,
            text=[f"{u:.1f}%" for u in utilizations],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "利用率: %{y:.1f}%<br>"
                "理论带宽: %{customdata[0]:.1f} GB/s<br>"
                "有效带宽: %{customdata[1]:.1f} GB/s"
                "<extra></extra>"
            ),
            customdata=list(zip(theoretical, effective)),
        ))

        # 添加阈值线
        fig.add_hline(y=70, line_dash="dash", line_color="#fa8c16",
                      annotation_text="警告阈值 (70%)")
        fig.add_hline(y=90, line_dash="dash", line_color="#f5222d",
                      annotation_text="临界阈值 (90%)")

        fig.update_layout(
            title="Tier6+ 带宽利用率分析",
            xaxis_title="位置",
            yaxis_title="利用率 (%)",
            yaxis=dict(range=[0, 110]),
        )

        return fig

    def render_heatmap(self, bandwidth_matrix: Dict) -> go.Figure:
        """
        渲染带宽热力图

        Args:
            bandwidth_matrix: 带宽矩阵 {(src, dst): utilization}

        Returns:
            Plotly Figure
        """
        # 提取节点
        nodes = set()
        for src, dst in bandwidth_matrix.keys():
            nodes.add(src)
            nodes.add(dst)
        nodes = sorted(nodes)

        # 构建矩阵
        n = len(nodes)
        matrix = [[0] * n for _ in range(n)]
        node_idx = {node: i for i, node in enumerate(nodes)}

        for (src, dst), util in bandwidth_matrix.items():
            i = node_idx.get(src, -1)
            j = node_idx.get(dst, -1)
            if i >= 0 and j >= 0:
                matrix[i][j] = util * 100

        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=nodes,
            y=nodes,
            colorscale=[
                [0, "#52c41a"],      # 绿色 - 低利用率
                [0.5, "#fadb14"],    # 黄色 - 中等
                [0.7, "#fa8c16"],    # 橙色 - 警告
                [1, "#f5222d"],      # 红色 - 临界
            ],
            colorbar=dict(title="利用率 (%)"),
            hovertemplate="源: %{y}<br>目标: %{x}<br>利用率: %{z:.1f}%<extra></extra>",
        ))

        fig.update_layout(
            title="Tier6+ 带宽利用率热力图",
            xaxis_title="目标",
            yaxis_title="源",
        )

        return fig

    def render_sankey(self, flow_data: List[Dict]) -> go.Figure:
        """
        渲染桑基图

        Args:
            flow_data: 流量数据列表
                [{source, target, value, level}]

        Returns:
            Plotly Figure
        """
        # 收集节点
        node_set = set()
        for flow in flow_data:
            node_set.add(flow["source"])
            node_set.add(flow["target"])
        nodes = sorted(node_set)
        node_idx = {n: i for i, n in enumerate(nodes)}

        # 构建链接
        sources = [node_idx[f["source"]] for f in flow_data]
        targets = [node_idx[f["target"]] for f in flow_data]
        values = [f.get("value", 1) for f in flow_data]

        # 节点颜色
        node_colors = []
        for node in nodes:
            for level, color in LEVEL_COLORS.items():
                if level in node.lower():
                    node_colors.append(color)
                    break
            else:
                node_colors.append("#999999")

        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=node_colors,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(150, 150, 150, 0.4)",
            ),
        ))

        fig.update_layout(
            title="Tier6+ 带宽流向桑基图",
            font_size=12,
        )

        return fig

    def render_bottleneck_indicator(
        self,
        bottleneck: Optional[Dict],
        total_bandwidth: float
    ) -> go.Figure:
        """
        渲染瓶颈指示器

        Args:
            bottleneck: 瓶颈信息 {location, utilization, theoretical_bandwidth_gbps}
            total_bandwidth: 总带宽

        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=("瓶颈利用率", "有效带宽"),
        )

        if bottleneck:
            utilization = bottleneck.get("utilization", 0) * 100
            effective_bw = bottleneck.get("effective_bandwidth_gbps", 0)

            # 利用率仪表盘
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=utilization,
                    number={"suffix": "%"},
                    title={"text": f"瓶颈: {bottleneck.get('location', 'N/A')}"},
                    gauge=dict(
                        axis=dict(range=[0, 100]),
                        bar=dict(color="#1890ff"),
                        steps=[
                            {"range": [0, 50], "color": "#52c41a"},
                            {"range": [50, 70], "color": "#fadb14"},
                            {"range": [70, 90], "color": "#fa8c16"},
                            {"range": [90, 100], "color": "#f5222d"},
                        ],
                        threshold=dict(
                            line=dict(color="red", width=4),
                            thickness=0.75,
                            value=90,
                        ),
                    ),
                ),
                row=1, col=1,
            )

            # 带宽指示器
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=effective_bw,
                    number={"suffix": " GB/s"},
                    title={"text": "有效带宽"},
                    delta={
                        "reference": total_bandwidth,
                        "relative": True,
                        "valueformat": ".1%",
                    },
                ),
                row=1, col=2,
            )
        else:
            # 无瓶颈
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=0,
                    title={"text": "无瓶颈检测"},
                ),
                row=1, col=1,
            )

        fig.update_layout(
            title="Tier6+ 瓶颈状态指示器",
            height=400,
        )

        return fig

    def render(
        self,
        bandwidth_data: Dict,
        style: str = "bar",
        bottleneck: Optional[Dict] = None
    ) -> go.Figure:
        """
        渲染带宽分析图

        Args:
            bandwidth_data: 带宽数据
            style: 样式 ('bar', 'heatmap', 'indicator')
            bottleneck: 瓶颈信息

        Returns:
            Plotly Figure
        """
        if style == "bar":
            return self.render_utilization_bar(bandwidth_data)
        elif style == "heatmap":
            # 需要转换为矩阵格式
            matrix = {}
            for loc, data in bandwidth_data.items():
                parts = loc.split(":")
                if len(parts) == 2:
                    matrix[(parts[0], parts[1])] = data.get("utilization", 0)
            return self.render_heatmap(matrix)
        elif style == "indicator":
            total_bw = sum(d.get("theoretical_bandwidth_gbps", 0) for d in bandwidth_data.values())
            return self.render_bottleneck_indicator(bottleneck, total_bw)
        else:
            raise ValueError(f"不支持的样式: {style}")

    def save_html(self, fig: go.Figure, path: str):
        """保存为 HTML 文件"""
        fig.write_html(path)
