"""
延迟分布可视化模块

提供延迟分布图表类型:
- 直方图+CDF组合图 (Histogram with CDF)
- 小提琴图 (Violin Plot)
"""

from typing import Dict, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class LatencyDistributionPlotter:
    """延迟分布绘图器"""

    # 颜色方案
    COLORS = {
        "read": "#1f77b4",  # 蓝色
        "write": "#d62728",  # 红色
        "mixed": "#2ca02c",  # 绿色
    }

    # 延迟类型的中文标签
    LATENCY_LABELS = {
        "cmd": "命令延迟",
        "data": "数据延迟",
        "trans": "事务总延迟",
    }

    # 请求类型的中文标签
    REQUEST_LABELS = {
        "read": "读请求",
        "write": "写请求",
        "mixed": "混合",
    }

    def __init__(self, latency_stats: Dict, title_prefix: str = "NoC"):
        """
        初始化延迟分布绘图器

        Args:
            latency_stats: 延迟统计字典,必须包含values字段
            title_prefix: 图表标题前缀,如"NoC"或"D2D"
        """
        self.latency_stats = latency_stats
        self.title_prefix = title_prefix

    def plot_histogram_with_cdf(self, return_fig: bool = True) -> Optional[go.Figure]:
        """
        绘制延迟直方图+CDF组合图(双Y轴) - 显示所有请求的整体分布

        Args:
            return_fig: 是否返回Figure对象

        Returns:
            Plotly Figure对象或None
        """
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[
                f"{self.LATENCY_LABELS['cmd']} (ns)",
                f"{self.LATENCY_LABELS['data']} (ns)",
                f"{self.LATENCY_LABELS['trans']} (ns)",
            ],
            specs=[[{}], [{}], [{}]],
            vertical_spacing=0.08,
        )

        categories = ["cmd", "data", "trans"]

        for row_idx, category in enumerate(categories, start=1):
            # 只使用mixed类型(所有请求的混合)
            values = self.latency_stats[category]["mixed"].get("values", [])
            if len(values) == 0:
                continue

            # 添加直方图
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name="直方图" if row_idx == 1 else None,
                    marker_color="#1f77b4",
                    opacity=0.7,
                    nbinsx=50,
                    showlegend=(row_idx == 1),
                    legendgroup="histogram",
                    hovertemplate="延迟: %{x:.1f} ns<br>计数: %{y}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )

            # 在当前子图上添加统计值的垂直线标注
            mean_val = np.mean(values)
            p50_val = np.percentile(values, 50)
            p95_val = self.latency_stats[category]["mixed"].get("p95", 0)
            p99_val = self.latency_stats[category]["mixed"].get("p99", 0)

            # 添加垂直线标注统计值位置
            # 使用不同的y高度来错开标注,避免重叠
            stats_lines = [
                (mean_val, "平均值", "green", "dash", 0.85),
                (p50_val, "P50", "blue", "dot", 0.70),
                (p95_val, "P95", "orange", "dashdot", 0.55),
                (p99_val, "P99", "red", "solid", 0.40),
            ]

            for stat_val, stat_name, color, dash_style, y_position in stats_lines:
                # 使用vline直接添加垂直线到指定的subplot
                fig.add_vline(
                    x=stat_val,
                    line_dash=dash_style,
                    line_color=color,
                    line_width=2,
                    row=row_idx,
                    col=1,
                )

                # 添加文本标注 - 必须明确指定subplot的坐标系
                xref = "x" if row_idx == 1 else f"x{row_idx}"
                yref = "y" if row_idx == 1 else f"y{row_idx}"

                fig.add_annotation(
                    x=stat_val,
                    y=y_position,
                    yref=f"{yref} domain",
                    xref=xref,
                    text=f"{stat_name}: {stat_val:.1f}ns",
                    showarrow=False,
                    font=dict(size=14, color=color),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=2,
                )

        # 设置所有子图的轴标签
        for row_idx in range(1, 4):
            fig.update_xaxes(title_text="延迟 (ns)", row=row_idx, col=1)
            fig.update_yaxes(title_text="频次", row=row_idx, col=1)

        fig.update_layout(
            height=1800,
            width=1600,
            hovermode="closest",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0.01),
            margin=dict(l=120, r=40, t=20, b=20),
        )

        # 调整子图标题位置,避免与垂直线标注重叠
        for annotation in fig["layout"]["annotations"]:
            if "text" in annotation and any(label in annotation["text"] for label in ["命令延迟", "数据延迟", "事务总延迟"]):
                annotation["y"] = annotation["y"] + 0.01  # 向上移动标题

        if return_fig:
            return fig
        else:
            fig.show()
            return None

    def plot_violin(self, return_fig: bool = True) -> Optional[go.Figure]:
        """
        绘制小提琴图 - 显示所有请求的整体分布

        Args:
            return_fig: 是否返回Figure对象

        Returns:
            Plotly Figure对象或None
        """
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                f"{self.LATENCY_LABELS['cmd']} (ns)",
                f"{self.LATENCY_LABELS['data']} (ns)",
                f"{self.LATENCY_LABELS['trans']} (ns)",
            ],
            horizontal_spacing=0.10,
        )

        categories = ["cmd", "data", "trans"]

        for col_idx, category in enumerate(categories, start=1):
            # 只使用mixed类型(所有请求的混合)
            values = self.latency_stats[category]["mixed"].get("values", [])
            if len(values) == 0:
                continue

            # 添加小提琴图
            fig.add_trace(
                go.Violin(
                    y=values,
                    name=self.LATENCY_LABELS[category],
                    fillcolor="#1f77b4",
                    line_color="#1f77b4",
                    opacity=0.6,
                    box_visible=True,  # 显示内部箱线图
                    meanline_visible=True,  # 显示均值线
                    showlegend=False,
                    hovertemplate=f"{self.LATENCY_LABELS[category]}<br>" + "延迟: %{y:.1f} ns<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )

            # 设置轴标签
            fig.update_yaxes(title_text="延迟 (ns)", row=1, col=col_idx)

        fig.update_layout(
            title_text=f"{self.title_prefix} 延迟分布小提琴图",
            height=600,
            showlegend=False,
        )

        if return_fig:
            return fig
        else:
            fig.show()
            return None
