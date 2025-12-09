"""
延迟分解图渲染器

使用 Plotly 生成延迟分解可视化
"""

from typing import Dict, List, Optional
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

# 延迟类型颜色
LATENCY_TYPE_COLORS = {
    "propagation": "#1890ff",
    "queuing": "#fa8c16",
    "processing": "#52c41a",
    "transmission": "#722ed1",
}


class LatencyBreakdownChart:
    """延迟分解图渲染器"""

    def render_stacked_bar(self, latency_breakdown: Dict) -> go.Figure:
        """
        渲染堆叠条形图

        Args:
            latency_breakdown: 延迟分解数据
                {level: {propagation_ns, queuing_ns, processing_ns, transmission_ns, total_ns}}

        Returns:
            Plotly Figure
        """
        levels = list(latency_breakdown.keys())

        # 按层级顺序排序
        level_order = ["die", "chip", "board", "server", "pod"]
        levels = sorted(levels, key=lambda x: level_order.index(x) if x in level_order else 99)

        propagation = [latency_breakdown[l].get("propagation_ns", 0) for l in levels]
        queuing = [latency_breakdown[l].get("queuing_ns", 0) for l in levels]
        processing = [latency_breakdown[l].get("processing_ns", 0) for l in levels]
        transmission = [latency_breakdown[l].get("transmission_ns", 0) for l in levels]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="传播延迟",
            x=levels,
            y=propagation,
            marker_color=LATENCY_TYPE_COLORS["propagation"],
            hovertemplate="传播延迟: %{y:.2f} ns<extra></extra>",
        ))

        fig.add_trace(go.Bar(
            name="排队延迟",
            x=levels,
            y=queuing,
            marker_color=LATENCY_TYPE_COLORS["queuing"],
            hovertemplate="排队延迟: %{y:.2f} ns<extra></extra>",
        ))

        fig.add_trace(go.Bar(
            name="处理延迟",
            x=levels,
            y=processing,
            marker_color=LATENCY_TYPE_COLORS["processing"],
            hovertemplate="处理延迟: %{y:.2f} ns<extra></extra>",
        ))

        fig.add_trace(go.Bar(
            name="传输延迟",
            x=levels,
            y=transmission,
            marker_color=LATENCY_TYPE_COLORS["transmission"],
            hovertemplate="传输延迟: %{y:.2f} ns<extra></extra>",
        ))

        fig.update_layout(
            barmode="stack",
            title="Tier6+ 延迟分解图 (按层级)",
            xaxis_title="层级",
            yaxis_title="延迟 (ns)",
            legend_title="延迟类型",
            hovermode="x unified",
        )

        return fig

    def render_waterfall(self, latency_breakdown: Dict) -> go.Figure:
        """
        渲染瀑布图

        Args:
            latency_breakdown: 延迟分解数据

        Returns:
            Plotly Figure
        """
        # 按层级顺序排序
        level_order = ["die", "chip", "board", "server", "pod"]
        levels = sorted(
            latency_breakdown.keys(),
            key=lambda x: level_order.index(x) if x in level_order else 99
        )

        names = []
        values = []
        measures = []
        colors = []

        cumulative = 0
        for level in levels:
            total = latency_breakdown[level].get("total_ns", 0)
            if total > 0:
                names.append(level.upper())
                values.append(total)
                measures.append("relative")
                colors.append(LEVEL_COLORS.get(level, "#999999"))
                cumulative += total

        # 添加总计
        names.append("总计")
        values.append(cumulative)
        measures.append("total")
        colors.append("#333333")

        fig = go.Figure(go.Waterfall(
            name="延迟",
            orientation="v",
            measure=measures,
            x=names,
            y=values,
            connector={"line": {"color": "#999999"}},
            textposition="outside",
            text=[f"{v:.1f}" for v in values],
            decreasing={"marker": {"color": "#52c41a"}},
            increasing={"marker": {"color": "#f5222d"}},
            totals={"marker": {"color": "#1890ff"}},
        ))

        # 使用自定义颜色
        fig.update_traces(
            marker_color=colors,
        )

        fig.update_layout(
            title="Tier6+ 延迟瀑布图",
            xaxis_title="层级",
            yaxis_title="延迟 (ns)",
            showlegend=False,
        )

        return fig

    def render_pie(self, latency_breakdown: Dict) -> go.Figure:
        """
        渲染饼图

        Args:
            latency_breakdown: 延迟分解数据

        Returns:
            Plotly Figure
        """
        levels = []
        values = []
        colors = []

        for level, data in latency_breakdown.items():
            total = data.get("total_ns", 0)
            if total > 0:
                levels.append(level.upper())
                values.append(total)
                colors.append(LEVEL_COLORS.get(level, "#999999"))

        fig = go.Figure(go.Pie(
            labels=levels,
            values=values,
            marker=dict(colors=colors),
            hole=0.4,
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>延迟: %{value:.2f} ns<br>占比: %{percent}<extra></extra>",
        ))

        total_latency = sum(values)
        fig.update_layout(
            title=f"Tier6+ 延迟占比图 (总计: {total_latency:.2f} ns)",
            annotations=[dict(
                text=f"{total_latency:.0f} ns",
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False,
            )],
        )

        return fig

    def render_combined(self, latency_breakdown: Dict) -> go.Figure:
        """
        渲染组合图 (堆叠条形 + 饼图)

        Args:
            latency_breakdown: 延迟分解数据

        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            subplot_titles=("延迟分解 (按层级)", "延迟占比"),
            column_widths=[0.6, 0.4],
        )

        # 左侧：堆叠条形图
        level_order = ["die", "chip", "board", "server", "pod"]
        levels = sorted(
            latency_breakdown.keys(),
            key=lambda x: level_order.index(x) if x in level_order else 99
        )

        for latency_type, color in LATENCY_TYPE_COLORS.items():
            key = f"{latency_type}_ns"
            values = [latency_breakdown[l].get(key, 0) for l in levels]
            fig.add_trace(
                go.Bar(
                    name=latency_type,
                    x=levels,
                    y=values,
                    marker_color=color,
                ),
                row=1, col=1,
            )

        # 右侧：饼图
        pie_levels = []
        pie_values = []
        pie_colors = []
        for level in levels:
            total = latency_breakdown[level].get("total_ns", 0)
            if total > 0:
                pie_levels.append(level.upper())
                pie_values.append(total)
                pie_colors.append(LEVEL_COLORS.get(level, "#999999"))

        fig.add_trace(
            go.Pie(
                labels=pie_levels,
                values=pie_values,
                marker=dict(colors=pie_colors),
                hole=0.3,
                textinfo="percent",
            ),
            row=1, col=2,
        )

        fig.update_layout(
            barmode="stack",
            title="Tier6+ 延迟分析",
            height=500,
        )

        return fig

    def render(
        self,
        latency_breakdown: Dict,
        style: str = "stacked"
    ) -> go.Figure:
        """
        渲染延迟分解图

        Args:
            latency_breakdown: 延迟分解数据
            style: 样式 ('stacked', 'waterfall', 'pie', 'combined')

        Returns:
            Plotly Figure
        """
        if style == "stacked":
            return self.render_stacked_bar(latency_breakdown)
        elif style == "waterfall":
            return self.render_waterfall(latency_breakdown)
        elif style == "pie":
            return self.render_pie(latency_breakdown)
        elif style == "combined":
            return self.render_combined(latency_breakdown)
        else:
            raise ValueError(f"不支持的样式: {style}")

    def save_html(self, fig: go.Figure, path: str):
        """保存为 HTML 文件"""
        fig.write_html(path)
