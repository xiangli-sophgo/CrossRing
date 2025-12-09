"""
规模扩展曲线渲染器

使用 Plotly 生成规模扩展分析可视化
"""

from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ScalingAnalysisChart:
    """规模扩展曲线渲染器"""

    def render_latency_scaling(self, scaling_data: Dict) -> go.Figure:
        """
        渲染延迟扩展曲线

        Args:
            scaling_data: 扩展数据
                {scale_factors, latency: {ideal, predicted}}

        Returns:
            Plotly Figure
        """
        scale_factors = scaling_data["scale_factors"]
        ideal = scaling_data["latency"]["ideal"]
        predicted = scaling_data["latency"]["predicted"]

        fig = go.Figure()

        # 理想扩展 (延迟不变)
        fig.add_trace(go.Scatter(
            x=scale_factors,
            y=ideal,
            mode="lines",
            name="理想扩展",
            line=dict(dash="dash", color="#52c41a"),
            hovertemplate="扩展因子: %{x}<br>理想延迟: %{y:.2f} ns<extra></extra>",
        ))

        # 预测扩展
        fig.add_trace(go.Scatter(
            x=scale_factors,
            y=predicted,
            mode="lines+markers",
            name="模型预测",
            line=dict(color="#1890ff"),
            marker=dict(size=10),
            hovertemplate="扩展因子: %{x}<br>预测延迟: %{y:.2f} ns<extra></extra>",
        ))

        fig.update_layout(
            title="Tier6+ 延迟扩展曲线",
            xaxis_title="扩展因子",
            yaxis_title="延迟 (ns)",
            xaxis_type="log",
            legend=dict(x=0.02, y=0.98),
        )

        return fig

    def render_throughput_scaling(self, scaling_data: Dict) -> go.Figure:
        """
        渲染吞吐量扩展曲线

        Args:
            scaling_data: 扩展数据
                {scale_factors, throughput: {ideal, predicted}}

        Returns:
            Plotly Figure
        """
        scale_factors = scaling_data["scale_factors"]
        ideal = scaling_data["throughput"]["ideal"]
        predicted = scaling_data["throughput"]["predicted"]

        fig = go.Figure()

        # 理想线性扩展
        fig.add_trace(go.Scatter(
            x=scale_factors,
            y=ideal,
            mode="lines",
            name="理想线性扩展",
            line=dict(dash="dash", color="#52c41a"),
            hovertemplate="扩展因子: %{x}<br>理想吞吐: %{y:.2f} GB/s<extra></extra>",
        ))

        # 预测扩展
        fig.add_trace(go.Scatter(
            x=scale_factors,
            y=predicted,
            mode="lines+markers",
            name="模型预测",
            line=dict(color="#1890ff"),
            marker=dict(size=10),
            hovertemplate="扩展因子: %{x}<br>预测吞吐: %{y:.2f} GB/s<extra></extra>",
        ))

        fig.update_layout(
            title="Tier6+ 吞吐量扩展曲线",
            xaxis_title="扩展因子",
            yaxis_title="吞吐量 (GB/s)",
            xaxis_type="log",
            yaxis_type="log",
            legend=dict(x=0.02, y=0.98),
        )

        return fig

    def render_efficiency(self, scaling_data: Dict) -> go.Figure:
        """
        渲染扩展效率曲线

        Args:
            scaling_data: 扩展数据
                {scale_factors, efficiency}

        Returns:
            Plotly Figure
        """
        scale_factors = scaling_data["scale_factors"]
        efficiency = [e * 100 for e in scaling_data["efficiency"]]

        # 根据效率设置颜色
        colors = []
        for e in efficiency:
            if e >= 80:
                colors.append("#52c41a")  # 绿色 - 优秀
            elif e >= 60:
                colors.append("#fadb14")  # 黄色 - 良好
            elif e >= 40:
                colors.append("#fa8c16")  # 橙色 - 一般
            else:
                colors.append("#f5222d")  # 红色 - 差

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=[str(s) for s in scale_factors],
            y=efficiency,
            marker_color=colors,
            text=[f"{e:.1f}%" for e in efficiency],
            textposition="outside",
            hovertemplate="扩展因子: %{x}<br>扩展效率: %{y:.1f}%<extra></extra>",
        ))

        # 添加阈值线
        fig.add_hline(y=80, line_dash="dash", line_color="#52c41a",
                      annotation_text="优秀 (80%)")
        fig.add_hline(y=60, line_dash="dot", line_color="#fadb14",
                      annotation_text="良好 (60%)")

        fig.update_layout(
            title="Tier6+ 扩展效率分析",
            xaxis_title="扩展因子",
            yaxis_title="扩展效率 (%)",
            yaxis=dict(range=[0, 110]),
        )

        return fig

    def render_combined(self, scaling_data: Dict) -> go.Figure:
        """
        渲染组合图 (延迟 + 吞吐量 + 效率)

        Args:
            scaling_data: 扩展数据

        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar", "colspan": 2}, None],
            ],
            subplot_titles=(
                "延迟扩展",
                "吞吐量扩展",
                "扩展效率",
            ),
            vertical_spacing=0.15,
        )

        scale_factors = scaling_data["scale_factors"]

        # 延迟扩展 (左上)
        fig.add_trace(
            go.Scatter(
                x=scale_factors,
                y=scaling_data["latency"]["ideal"],
                mode="lines",
                name="理想延迟",
                line=dict(dash="dash", color="#52c41a"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=scale_factors,
                y=scaling_data["latency"]["predicted"],
                mode="lines+markers",
                name="预测延迟",
                line=dict(color="#1890ff"),
            ),
            row=1, col=1,
        )

        # 吞吐量扩展 (右上)
        fig.add_trace(
            go.Scatter(
                x=scale_factors,
                y=scaling_data["throughput"]["ideal"],
                mode="lines",
                name="理想吞吐",
                line=dict(dash="dash", color="#52c41a"),
                showlegend=False,
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=scale_factors,
                y=scaling_data["throughput"]["predicted"],
                mode="lines+markers",
                name="预测吞吐",
                line=dict(color="#fa8c16"),
                showlegend=False,
            ),
            row=1, col=2,
        )

        # 扩展效率 (下)
        efficiency = [e * 100 for e in scaling_data["efficiency"]]
        colors = ["#52c41a" if e >= 80 else "#fadb14" if e >= 60 else "#fa8c16" if e >= 40 else "#f5222d"
                  for e in efficiency]

        fig.add_trace(
            go.Bar(
                x=[str(s) for s in scale_factors],
                y=efficiency,
                marker_color=colors,
                text=[f"{e:.1f}%" for e in efficiency],
                textposition="outside",
                name="效率",
                showlegend=False,
            ),
            row=2, col=1,
        )

        # 更新布局
        fig.update_xaxes(title_text="扩展因子", row=1, col=1)
        fig.update_xaxes(title_text="扩展因子", row=1, col=2)
        fig.update_xaxes(title_text="扩展因子", row=2, col=1)

        fig.update_yaxes(title_text="延迟 (ns)", row=1, col=1)
        fig.update_yaxes(title_text="吞吐量 (GB/s)", row=1, col=2)
        fig.update_yaxes(title_text="效率 (%)", range=[0, 110], row=2, col=1)

        fig.update_layout(
            title="Tier6+ 规模扩展综合分析",
            height=700,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
        )

        return fig

    def render_amdahl_vs_gustafson(
        self,
        scale_factors: List[int],
        parallel_ratio: float = 0.9
    ) -> go.Figure:
        """
        渲染 Amdahl 定律 vs Gustafson 定律对比

        Args:
            scale_factors: 扩展因子列表
            parallel_ratio: 可并行比例

        Returns:
            Plotly Figure
        """
        from ..math_models import ScalingModel

        # 计算 Amdahl 加速比
        amdahl_speedup = []
        for n in scale_factors:
            # 加速比 = 1 / ((1-p) + p/n)
            speedup = 1 / ((1 - parallel_ratio) + parallel_ratio / n)
            amdahl_speedup.append(speedup)

        # 计算 Gustafson 加速比
        gustafson_speedup = []
        for n in scale_factors:
            # 加速比 = (1-p) + p*n
            speedup = (1 - parallel_ratio) + parallel_ratio * n
            gustafson_speedup.append(speedup)

        fig = go.Figure()

        # 理想线性
        fig.add_trace(go.Scatter(
            x=scale_factors,
            y=scale_factors,
            mode="lines",
            name="理想线性",
            line=dict(dash="dot", color="#999999"),
        ))

        # Amdahl
        fig.add_trace(go.Scatter(
            x=scale_factors,
            y=amdahl_speedup,
            mode="lines+markers",
            name=f"Amdahl (p={parallel_ratio})",
            line=dict(color="#f5222d"),
        ))

        # Gustafson
        fig.add_trace(go.Scatter(
            x=scale_factors,
            y=gustafson_speedup,
            mode="lines+markers",
            name=f"Gustafson (p={parallel_ratio})",
            line=dict(color="#1890ff"),
        ))

        fig.update_layout(
            title="Amdahl 定律 vs Gustafson 定律",
            xaxis_title="处理器数量",
            yaxis_title="加速比",
            xaxis_type="log",
            yaxis_type="log",
            legend=dict(x=0.02, y=0.98),
        )

        return fig

    def render(
        self,
        scaling_data: Dict,
        style: str = "combined"
    ) -> go.Figure:
        """
        渲染规模扩展分析图

        Args:
            scaling_data: 扩展数据
            style: 样式 ('latency', 'throughput', 'efficiency', 'combined')

        Returns:
            Plotly Figure
        """
        if style == "latency":
            return self.render_latency_scaling(scaling_data)
        elif style == "throughput":
            return self.render_throughput_scaling(scaling_data)
        elif style == "efficiency":
            return self.render_efficiency(scaling_data)
        elif style == "combined":
            return self.render_combined(scaling_data)
        else:
            raise ValueError(f"不支持的样式: {style}")

    def save_html(self, fig: go.Figure, path: str):
        """保存为 HTML 文件"""
        fig.write_html(path)
