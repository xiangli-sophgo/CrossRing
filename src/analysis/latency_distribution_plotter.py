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

    def plot_latency_vs_time_windowed(self, window_ns: int = 2000, return_fig: bool = True) -> Optional[go.Figure]:
        """
        绘制时间窗口平均延迟折线图（方案A）

        Args:
            window_ns: 时间窗口大小（ns）
            return_fig: 是否返回Figure对象

        Returns:
            Plotly Figure对象或None
        """
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[
                f"{self.LATENCY_LABELS['cmd']} vs 时间",
                f"{self.LATENCY_LABELS['data']} vs 时间",
                f"{self.LATENCY_LABELS['trans']} vs 时间",
            ],
            specs=[[{}], [{}], [{}]],
            vertical_spacing=0.08,
        )

        categories = ["cmd", "data", "trans"]

        for row_idx, category in enumerate(categories, start=1):
            # 获取时间-延迟配对数据
            time_value_pairs = self.latency_stats[category]["mixed"].get("time_value_pairs", [])
            if len(time_value_pairs) == 0:
                continue

            # 按时间排序
            time_value_pairs = sorted(time_value_pairs, key=lambda x: x[0])

            # 计算时间窗口统计
            window_data = self._compute_windowed_stats(time_value_pairs, window_ns)

            if len(window_data["times"]) == 0:
                continue

            # 添加平均值折线
            fig.add_trace(
                go.Scatter(
                    x=window_data["times"],
                    y=window_data["means"],
                    mode="lines+markers",
                    name="平均延迟" if row_idx == 1 else None,
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=4),
                    showlegend=(row_idx == 1),
                    legendgroup="mean",
                    hovertemplate="时间: %{x:.0f} ns<br>平均延迟: %{y:.1f} ns<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )

            # 添加min-max阴影带
            fig.add_trace(
                go.Scatter(
                    x=window_data["times"] + window_data["times"][::-1],
                    y=window_data["maxs"] + window_data["mins"][::-1],
                    fill="toself",
                    fillcolor="rgba(31, 119, 180, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=(row_idx == 1),
                    legendgroup="range",
                    name="Min-Max范围" if row_idx == 1 else None,
                    hoverinfo="skip",
                ),
                row=row_idx,
                col=1,
            )

        # 设置所有子图的轴标签
        for row_idx in range(1, 4):
            fig.update_xaxes(title_text="仿真时间 (ns)", row=row_idx, col=1)
            fig.update_yaxes(title_text="延迟 (ns)", row=row_idx, col=1)

        fig.update_layout(
            height=1800,
            width=1600,
            hovermode="closest",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0.01),
            margin=dict(l=120, r=40, t=20, b=20),
        )

        if return_fig:
            return fig
        else:
            fig.show()
            return None

    def _compute_windowed_stats(self, time_value_pairs, window_ns):
        """
        计算时间窗口统计

        Args:
            time_value_pairs: [(time, latency), ...] 列表
            window_ns: 窗口大小（ns）

        Returns:
            Dict: {"times": [中点时间], "means": [平均值], "mins": [最小值], "maxs": [最大值]}
        """
        if len(time_value_pairs) == 0:
            return {"times": [], "means": [], "mins": [], "maxs": []}

        # 确定时间范围
        min_time = time_value_pairs[0][0]
        max_time = time_value_pairs[-1][0]

        window_data = {"times": [], "means": [], "mins": [], "maxs": []}

        # 遍历时间窗口
        current_time = min_time
        while current_time <= max_time:
            window_end = current_time + window_ns

            # 收集该窗口内的延迟值
            window_values = [lat for t, lat in time_value_pairs if current_time <= t < window_end]

            if len(window_values) > 0:
                window_data["times"].append(current_time + window_ns / 2)  # 窗口中点
                window_data["means"].append(np.mean(window_values))
                window_data["mins"].append(np.min(window_values))
                window_data["maxs"].append(np.max(window_values))

            current_time += window_ns

        return window_data

    def plot_latency_vs_time_scatter(self, rolling_window: int = 100, return_fig: bool = True) -> Optional[go.Figure]:
        """
        绘制散点图 + 滑动窗口趋势线（方案B）

        Args:
            rolling_window: 滑动窗口大小（请求数）
            return_fig: 是否返回Figure对象

        Returns:
            Plotly Figure对象或None
        """
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[
                f"{self.LATENCY_LABELS['cmd']} vs 时间（散点图）",
                f"{self.LATENCY_LABELS['data']} vs 时间（散点图）",
                f"{self.LATENCY_LABELS['trans']} vs 时间（散点图）",
            ],
            specs=[[{}], [{}], [{}]],
            vertical_spacing=0.08,
        )

        categories = ["cmd", "data", "trans"]

        for row_idx, category in enumerate(categories, start=1):
            # 获取时间-延迟配对数据
            time_value_pairs = self.latency_stats[category]["mixed"].get("time_value_pairs", [])
            if len(time_value_pairs) == 0:
                continue

            # 按时间排序
            time_value_pairs = sorted(time_value_pairs, key=lambda x: x[0])
            times = [t for t, _ in time_value_pairs]
            latencies = [lat for _, lat in time_value_pairs]

            # 添加散点图（使用Scattergl提升性能）
            fig.add_trace(
                go.Scattergl(
                    x=times,
                    y=latencies,
                    mode="markers",
                    name="请求延迟" if row_idx == 1 else None,
                    marker=dict(size=2, color="#1f77b4", opacity=0.3),
                    showlegend=(row_idx == 1),
                    legendgroup="scatter",
                    hovertemplate="时间: %{x:.0f} ns<br>延迟: %{y:.1f} ns<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )

            # 计算滑动窗口平均值（用于趋势线）
            if len(latencies) >= rolling_window:
                rolling_times, rolling_means = self._compute_rolling_average(times, latencies, rolling_window)

                # 添加趋势线
                fig.add_trace(
                    go.Scatter(
                        x=rolling_times,
                        y=rolling_means,
                        mode="lines",
                        name="滑动平均" if row_idx == 1 else None,
                        line=dict(color="#d62728", width=2),
                        showlegend=(row_idx == 1),
                        legendgroup="trend",
                        hovertemplate="时间: %{x:.0f} ns<br>滑动平均: %{y:.1f} ns<extra></extra>",
                    ),
                    row=row_idx,
                    col=1,
                )

        # 设置所有子图的轴标签
        for row_idx in range(1, 4):
            fig.update_xaxes(title_text="仿真时间 (ns)", row=row_idx, col=1)
            fig.update_yaxes(title_text="延迟 (ns)", row=row_idx, col=1)

        fig.update_layout(
            height=1800,
            width=1600,
            hovermode="closest",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0.01),
            margin=dict(l=120, r=40, t=20, b=20),
        )

        if return_fig:
            return fig
        else:
            fig.show()
            return None

    def _compute_rolling_average(self, times, values, window_size):
        """
        计算滑动窗口平均值

        Args:
            times: 时间列表
            values: 值列表
            window_size: 窗口大小（元素数）

        Returns:
            (rolling_times, rolling_means): 滑动平均的时间和值
        """
        rolling_times = []
        rolling_means = []

        for i in range(len(values) - window_size + 1):
            window_vals = values[i : i + window_size]
            window_time = times[i + window_size // 2]  # 使用窗口中点的时间
            rolling_times.append(window_time)
            rolling_means.append(np.mean(window_vals))

        return rolling_times, rolling_means

    def plot_latency_vs_time_heatmap(self, time_bin_ns: int = 1000, latency_bin_ns: int = 50, return_fig: bool = True) -> Optional[go.Figure]:
        """
        绘制2D热力图（方案C）

        Args:
            time_bin_ns: 时间分箱大小（ns）
            latency_bin_ns: 延迟分箱大小（ns）
            return_fig: 是否返回Figure对象

        Returns:
            Plotly Figure对象或None
        """
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[
                f"{self.LATENCY_LABELS['cmd']} 时间-延迟热力图",
                f"{self.LATENCY_LABELS['data']} 时间-延迟热力图",
                f"{self.LATENCY_LABELS['trans']} 时间-延迟热力图",
            ],
            specs=[[{"type": "heatmap"}], [{"type": "heatmap"}], [{"type": "heatmap"}]],
            vertical_spacing=0.08,
        )

        categories = ["cmd", "data", "trans"]

        for row_idx, category in enumerate(categories, start=1):
            # 获取时间-延迟配对数据
            time_value_pairs = self.latency_stats[category]["mixed"].get("time_value_pairs", [])
            if len(time_value_pairs) == 0:
                continue

            # 按时间排序
            time_value_pairs = sorted(time_value_pairs, key=lambda x: x[0])
            times = [t for t, _ in time_value_pairs]
            latencies = [lat for _, lat in time_value_pairs]

            # 创建2D直方图
            heatmap_data = self._compute_2d_histogram(times, latencies, time_bin_ns, latency_bin_ns)

            if len(heatmap_data["time_bins"]) == 0 or len(heatmap_data["latency_bins"]) == 0:
                continue

            # 添加热力图
            fig.add_trace(
                go.Heatmap(
                    x=heatmap_data["time_bins"],
                    y=heatmap_data["latency_bins"],
                    z=heatmap_data["counts"],
                    colorscale="YlOrRd",
                    colorbar=dict(title="请求数", x=1.02 if row_idx == 2 else 1.02),
                    hovertemplate="时间: %{x:.0f} ns<br>延迟: %{y:.0f} ns<br>请求数: %{z}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )

        # 设置所有子图的轴标签
        for row_idx in range(1, 4):
            fig.update_xaxes(title_text="仿真时间 (ns)", row=row_idx, col=1)
            fig.update_yaxes(title_text="延迟 (ns)", row=row_idx, col=1)

        fig.update_layout(
            height=1800,
            width=1600,
            margin=dict(l=120, r=100, t=20, b=20),
        )

        if return_fig:
            return fig
        else:
            fig.show()
            return None

    def _compute_2d_histogram(self, times, latencies, time_bin_ns, latency_bin_ns):
        """
        计算2D直方图

        Args:
            times: 时间列表
            latencies: 延迟列表
            time_bin_ns: 时间分箱大小
            latency_bin_ns: 延迟分箱大小

        Returns:
            Dict: {"time_bins": [...], "latency_bins": [...], "counts": [[...]]}
        """
        if len(times) == 0:
            return {"time_bins": [], "latency_bins": [], "counts": []}

        # 确定时间和延迟的范围
        min_time = min(times)
        max_time = max(times)
        min_lat = min(latencies)
        max_lat = max(latencies)

        # 创建分箱边界
        time_edges = list(range(int(min_time), int(max_time) + time_bin_ns, time_bin_ns))
        lat_edges = list(range(int(min_lat), int(max_lat) + latency_bin_ns, latency_bin_ns))

        # 初始化计数矩阵
        counts = [[0 for _ in range(len(time_edges) - 1)] for _ in range(len(lat_edges) - 1)]

        # 填充计数矩阵
        for t, lat in zip(times, latencies):
            # 找到时间所属的箱
            time_idx = int((t - min_time) / time_bin_ns)
            time_idx = min(time_idx, len(time_edges) - 2)

            # 找到延迟所属的箱
            lat_idx = int((lat - min_lat) / latency_bin_ns)
            lat_idx = min(lat_idx, len(lat_edges) - 2)

            counts[lat_idx][time_idx] += 1

        # 生成箱中点标签
        time_bins = [(time_edges[i] + time_edges[i + 1]) / 2 for i in range(len(time_edges) - 1)]
        latency_bins = [(lat_edges[i] + lat_edges[i + 1]) / 2 for i in range(len(lat_edges) - 1)]

        return {"time_bins": time_bins, "latency_bins": latency_bins, "counts": counts}

    def plot_histogram_and_scatter_combined(self, rolling_window: int = 100, return_fig: bool = True) -> Optional[go.Figure]:
        """
        绘制直方图和散点图的组合视图（3行2列布局）

        左列：三种延迟类型的直方图
        右列：三种延迟类型的散点图（带滑动平均趋势线）

        Args:
            rolling_window: 滑动窗口大小（散点图趋势线）
            return_fig: 是否返回Figure对象

        Returns:
            Plotly Figure对象或None
        """
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                f"{self.LATENCY_LABELS['cmd']} - 分布",
                f"{self.LATENCY_LABELS['cmd']} - 时序",
                f"{self.LATENCY_LABELS['data']} - 分布",
                f"{self.LATENCY_LABELS['data']} - 时序",
                f"{self.LATENCY_LABELS['trans']} - 分布",
                f"{self.LATENCY_LABELS['trans']} - 时序",
            ],
            specs=[[{}, {}], [{}, {}], [{}, {}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
            column_widths=[0.5, 0.5],
        )

        categories = ["cmd", "data", "trans"]

        for row_idx, category in enumerate(categories, start=1):
            # === 左列：直方图 ===
            values = self.latency_stats[category]["mixed"].get("values", [])
            if len(values) > 0:
                # 添加直方图
                fig.add_trace(
                    go.Histogram(
                        x=values,
                        name="频次分布" if row_idx == 1 else None,
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

                # 添加统计线
                mean_val = np.mean(values)
                p95_val = self.latency_stats[category]["mixed"].get("p95", 0)
                p99_val = self.latency_stats[category]["mixed"].get("p99", 0)

                # P95线
                fig.add_vline(
                    x=p95_val,
                    line_dash="dashdot",
                    line_color="orange",
                    line_width=2,
                    row=row_idx,
                    col=1,
                )

                # P99线
                fig.add_vline(
                    x=p99_val,
                    line_dash="solid",
                    line_color="red",
                    line_width=2,
                    row=row_idx,
                    col=1,
                )

            # === 右列：散点图 ===
            time_value_pairs = self.latency_stats[category]["mixed"].get("time_value_pairs", [])
            if len(time_value_pairs) > 0:
                # 按时间排序
                time_value_pairs = sorted(time_value_pairs, key=lambda x: x[0])
                times = [t for t, _ in time_value_pairs]
                latencies = [lat for _, lat in time_value_pairs]

                # 添加散点
                fig.add_trace(
                    go.Scattergl(
                        x=times,
                        y=latencies,
                        mode="markers",
                        name="各请求延迟" if row_idx == 1 else None,
                        marker=dict(size=3, color="#1f77b4", opacity=0.4),
                        showlegend=(row_idx == 1),
                        legendgroup="scatter",
                        hovertemplate="时间: %{x:.0f} ns<br>延迟: %{y:.1f} ns<extra></extra>",
                    ),
                    row=row_idx,
                    col=2,
                )

                # 计算滑动平均
                if len(latencies) >= rolling_window:
                    rolling_times, rolling_means = self._compute_rolling_average(times, latencies, rolling_window)

                    fig.add_trace(
                        go.Scatter(
                            x=rolling_times,
                            y=rolling_means,
                            mode="lines",
                            name="滑动平均" if row_idx == 1 else None,
                            line=dict(color="#d62728", width=2),
                            showlegend=(row_idx == 1),
                            legendgroup="trend",
                            hovertemplate="时间: %{x:.0f} ns<br>平均延迟: %{y:.1f} ns<extra></extra>",
                        ),
                        row=row_idx,
                        col=2,
                    )

        # 设置所有子图的轴标签
        for row_idx in range(1, 4):
            # 左列（直方图）
            fig.update_xaxes(title_text="延迟 (ns)", row=row_idx, col=1)
            fig.update_yaxes(title_text="频次", row=row_idx, col=1)

            # 右列（散点图）
            fig.update_xaxes(title_text="仿真时间 (ns)", row=row_idx, col=2)
            fig.update_yaxes(title_text="延迟 (ns)", row=row_idx, col=2)

        fig.update_layout(
            height=1400,
            width=1800,
            hovermode="closest",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(l=80, r=40, t=80, b=60),
        )

        if return_fig:
            return fig
        else:
            fig.show()
            return None
