"""
流量分析模块 - 流量数据的可视化分析

提供流量文件解析、时间序列分析、热力图和统计功能
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Tuple
from pathlib import Path


class TrafficAnalyzer:
    """流量分析器"""

    def __init__(self):
        """初始化分析器"""
        self.df = None

    def load_traffic_file(self, file_path: str) -> pd.DataFrame:
        """
        加载流量文件

        :param file_path: 流量文件路径
        :return: DataFrame
        """
        # 检查文件是否存在
        if not Path(file_path).exists():
            raise FileNotFoundError(f"流量文件不存在: {file_path}")

        # 读取CSV文件
        self.df = pd.read_csv(
            file_path,
            names=["timestamp", "src_pos", "src_type", "dst_pos", "dst_type", "req_type", "burst"]
        )

        return self.df

    def load_dataframe(self, df: pd.DataFrame):
        """
        加载已有的DataFrame

        :param df: 流量数据DataFrame
        """
        self.df = df.copy()

    def get_statistics(self) -> Dict[str, any]:
        """
        获取流量统计信息

        :return: 统计字典
        """
        if self.df is None or len(self.df) == 0:
            return {
                "total_requests": 0,
                "read_requests": 0,
                "write_requests": 0,
                "time_range": "N/A",
                "unique_src_nodes": 0,
                "unique_dst_nodes": 0,
                "unique_pairs": 0,
            }

        total = len(self.df)
        read_count = len(self.df[self.df['req_type'] == 'R'])
        write_count = len(self.df[self.df['req_type'] == 'W'])

        return {
            "total_requests": total,
            "read_requests": read_count,
            "write_requests": write_count,
            "read_ratio": read_count / total if total > 0 else 0,
            "write_ratio": write_count / total if total > 0 else 0,
            "time_range": f"{self.df['timestamp'].min()} - {self.df['timestamp'].max()} ns",
            "unique_src_nodes": self.df['src_pos'].nunique(),
            "unique_dst_nodes": self.df['dst_pos'].nunique(),
            "unique_pairs": len(self.df.groupby(['src_pos', 'dst_pos'])),
            "avg_burst": self.df['burst'].mean(),
        }

    def plot_time_series(self, bins: int = 50) -> go.Figure:
        """
        绘制时间序列图

        :param bins: 时间分组数量
        :return: Plotly Figure对象
        """
        if self.df is None or len(self.df) == 0:
            # 返回空图表
            fig = go.Figure()
            fig.add_annotation(
                text="无流量数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig

        # 按时间窗口统计请求数
        time_bins = pd.cut(self.df['timestamp'], bins=bins)
        counts = self.df.groupby(time_bins, observed=False).size()

        # 获取时间窗口的中点
        bin_centers = [interval.mid for interval in counts.index]

        # 创建折线图
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=counts.values,
            mode='lines+markers',
            name='请求数',
            line=dict(color='#4472C4', width=2),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="流量时间分布",
            xaxis_title="时间 (ns)",
            yaxis_title="请求数",
            hovermode='x unified',
            height=400
        )

        return fig

    def plot_req_type_distribution(self) -> go.Figure:
        """
        绘制读写请求分布图

        :return: Plotly Figure对象
        """
        if self.df is None or len(self.df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="无流量数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig

        # 统计读写请求
        req_counts = self.df['req_type'].value_counts()

        # 创建饼图
        fig = go.Figure(data=[go.Pie(
            labels=['读请求' if x == 'R' else '写请求' for x in req_counts.index],
            values=req_counts.values,
            marker=dict(colors=['#4472C4', '#ED7D31']),
            hole=0.3
        )])

        fig.update_layout(
            title="读写请求分布",
            height=400
        )

        return fig

    def plot_heatmap(self, max_nodes: int = 50) -> go.Figure:
        """
        绘制源-目标节点流量热力图

        :param max_nodes: 最大显示节点数(避免图表过大)
        :return: Plotly Figure对象
        """
        if self.df is None or len(self.df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="无流量数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig

        # 统计源-目标对的请求数
        pair_counts = self.df.groupby(['src_pos', 'dst_pos']).size().reset_index(name='count')

        # 检查节点数量
        unique_nodes = set(pair_counts['src_pos']).union(set(pair_counts['dst_pos']))
        if len(unique_nodes) > max_nodes:
            # 如果节点太多,只显示请求数最多的节点对
            pair_counts = pair_counts.nlargest(max_nodes, 'count')

        # 创建透视表
        matrix = pair_counts.pivot(index='src_pos', columns='dst_pos', values='count').fillna(0)

        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale='Blues',
            colorbar=dict(title="请求数"),
            hovertemplate='源节点: %{y}<br>目标节点: %{x}<br>请求数: %{z}<extra></extra>'
        ))

        fig.update_layout(
            title="节点间流量热力图",
            xaxis_title="目标节点",
            yaxis_title="源节点",
            height=500
        )

        return fig

    def plot_bandwidth_distribution(self) -> go.Figure:
        """
        绘制带宽分布柱状图(按源-目标对)

        :return: Plotly Figure对象
        """
        if self.df is None or len(self.df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="无流量数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig

        # 统计源-目标对的请求数(取前20个)
        pair_counts = (
            self.df.groupby(['src_pos', 'dst_pos'])
            .size()
            .reset_index(name='count')
            .nlargest(20, 'count')
        )

        # 创建标签
        pair_counts['pair_label'] = (
            pair_counts['src_pos'].astype(str) + ' → ' + pair_counts['dst_pos'].astype(str)
        )

        # 创建柱状图
        fig = go.Figure(data=[go.Bar(
            x=pair_counts['pair_label'],
            y=pair_counts['count'],
            marker=dict(color='#4472C4')
        )])

        fig.update_layout(
            title="Top 20 节点对流量分布",
            xaxis_title="节点对 (源→目标)",
            yaxis_title="请求数",
            xaxis_tickangle=-45,
            height=400
        )

        return fig

    def get_preview_dataframe(self, n: int = 100) -> pd.DataFrame:
        """
        获取预览数据

        :param n: 预览行数
        :return: DataFrame
        """
        if self.df is None:
            return pd.DataFrame()

        return self.df.head(n)

    def export_statistics_table(self) -> pd.DataFrame:
        """
        导出统计表格

        :return: 统计DataFrame
        """
        stats = self.get_statistics()

        # 创建表格数据 - 所有值转为字符串以避免Arrow序列化错误
        data = {
            "指标": [
                "总请求数",
                "读请求数",
                "写请求数",
                "读请求占比",
                "写请求占比",
                "时间范围",
                "涉及源节点数",
                "涉及目标节点数",
                "唯一节点对数",
                "平均Burst长度"
            ],
            "值": [
                f"{stats['total_requests']:,}",
                f"{stats['read_requests']:,}",
                f"{stats['write_requests']:,}",
                f"{stats['read_ratio']:.1%}",
                f"{stats['write_ratio']:.1%}",
                str(stats['time_range']),
                str(stats['unique_src_nodes']),
                str(stats['unique_dst_nodes']),
                str(stats['unique_pairs']),
                f"{stats['avg_burst']:.2f}"
            ]
        }

        return pd.DataFrame(data)
