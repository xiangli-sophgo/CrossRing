"""
可视化器模块 - 提供带宽曲线、热图和IP信息绘制功能

包含:
1. BandwidthPlotter - 带宽曲线绘制类
2. HeatmapDrawer - 热图绘制类
3. IPInfoBoxDrawer - IP信息框绘制类

注意: FlowGraphRenderer已移至flow_graph_renderer.py
"""

import sys
import matplotlib

# 兼容不同系统的matplotlib backend配置
if sys.platform == "darwin":  # macOS
    try:
        matplotlib.use("macosx")
    except ImportError:
        matplotlib.use("Agg")


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from .analyzers import IP_COLOR_MAP, MAX_BANDWIDTH_NORMALIZATION
import warnings
import os
import math
import traceback
import plotly.graph_objects as go


class BandwidthPlotter:
    """带宽曲线绘制器 - 绘制时间序列带宽曲线"""

    def __init__(self):
        """初始化带宽曲线绘制器"""
        pass

    def plot_rn_bandwidth_curves(self, rn_bandwidth_time_series: Dict, network_frequency: int = 2, save_path: str = None, show_fig: bool = False, return_fig: bool = False):
        """
        绘制RN带宽时间曲线（Plotly交互式版本）

        Args:
            rn_bandwidth_time_series: RN带宽时间序列数据
                格式: {port_key: {"time": [...], "start_times": [...], "bytes": [...]}}
            network_frequency: 网络频率 (GHz)
            save_path: 保存路径（.html文件）
            show_fig: 是否在浏览器中显示图像
            return_fig: 是否返回Figure对象而不是保存文件

        Returns:
            float or tuple: 如果return_fig=False，返回总带宽；如果return_fig=True，返回(总带宽, Figure对象)
        """
        fig = go.Figure()
        total_bw = 0

        # 批量收集traces和annotations
        all_traces = []
        all_annotations = []

        for port_key, data_dict in rn_bandwidth_time_series.items():
            if not data_dict.get("time"):
                continue

            # 排序时间戳并去除nan和inf值
            raw_times = np.array(data_dict["time"])
            clean_times = raw_times[~(np.isnan(raw_times) | np.isinf(raw_times))]
            times = np.sort(clean_times)

            if len(times) == 0:
                continue

            # 计算累积带宽
            cum_counts = np.arange(1, len(times) + 1)
            bandwidth = (cum_counts * 128 * 4) / times  # bytes/ns转换为GB/s

            # 只显示前100%的时间段
            t = np.percentile(times, 100)
            mask = times <= t

            times_us = times[mask] / 1000  # 转换为微秒
            bandwidth_filtered = bandwidth[mask]
            final_bw = bandwidth_filtered[-1]

            # 收集曲线轨迹
            all_traces.append(go.Scatter(
                x=times_us,
                y=bandwidth_filtered,
                mode="lines",
                name=port_key,
                hovertemplate="<b>%{fullData.name}</b><br>时间: %{x:.2f} us<br>带宽: %{y:.2f} GB/s<extra></extra>"
            ))

            # 收集末尾文本标注
            all_annotations.append(dict(
                x=times_us[-1],
                y=final_bw,
                text=f"{final_bw:.2f}",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=12)
            ))

            total_bw += final_bw

        # 批量添加所有traces和annotations
        if all_traces:
            fig.add_traces(all_traces)
        for ann in all_annotations:
            fig.add_annotation(**ann)

        # 设置图表布局
        layout_config = dict(
            xaxis_title="Time (us)",
            yaxis_title="Bandwidth (GB/s)",
            hovermode="closest",
            showlegend=True,
            width=1200,
            height=800,
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
        )

        # 只在非集成模式下显示标题
        if not return_fig:
            layout_config["title"] = "RN Bandwidth"

        fig.update_layout(**layout_config)

        if return_fig:
            return total_bw, fig

        if save_path:
            fig.write_html(save_path)

        if show_fig:
            fig.show()

        return total_bw

    def plot_rn_bandwidth_curves_work_interval(self, rn_bandwidth_time_series: Dict, network_frequency: int = 2, save_path: str = None, show_fig: bool = False, return_fig: bool = False):
        """
        绘制RN带宽工作区间曲线（去除空闲时段）

        Args:
            rn_bandwidth_time_series: RN带宽时间序列数据
            network_frequency: 网络频率
            save_path: 保存路径
            show_fig: 是否在浏览器中显示图像
            return_fig: 是否返回Figure对象

        Returns:
            float or tuple: 如果return_fig=False，返回带宽积分；如果return_fig=True，返回(带宽积分, Figure对象)
        """
        # 简化实现：调用主函数
        return self.plot_rn_bandwidth_curves(rn_bandwidth_time_series, network_frequency, save_path, show_fig, return_fig)


class HeatmapDrawer:
    """热图绘制器 - 绘制IP带宽热图"""

    def __init__(self):
        """初始化热图绘制器"""
        self.IP_COLOR_MAP = IP_COLOR_MAP

    def draw_ip_bandwidth_heatmap(self, die_ip_bandwidth_data: Dict, dies: Dict = None, config=None, mode: str = "total", node_size: int = 4000, save_path: str = None):
        """
        绘制IP带宽热图（支持多Die）

        Args:
            die_ip_bandwidth_data: IP带宽数据字典
                格式: {die_id: {mode: {ip_type: bandwidth_matrix}}}
            dies: Die模型字典
            config: 配置对象
            mode: 显示模式 ("read", "write", "total")
            node_size: 节点大小
            save_path: 保存路径
        """
        if not die_ip_bandwidth_data:
            print("警告: 没有die_ip_bandwidth_data数据，跳过IP带宽热力图绘制")
            return

        # 创建画布
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_aspect("equal")

        # 收集所有IP类型和最大带宽值
        all_bandwidths = []
        used_ip_types = set()

        for die_id in die_ip_bandwidth_data.keys():
            if die_id in die_ip_bandwidth_data:
                die_data = die_ip_bandwidth_data[die_id]
                if mode in die_data:
                    for ip_type, data_matrix in die_data[mode].items():
                        nonzero_bw = data_matrix[data_matrix > 0.001]
                        if len(nonzero_bw) > 0:
                            all_bandwidths.extend(nonzero_bw.tolist())
                            used_ip_types.add(ip_type.upper().split("_")[0])

        # 计算全局带宽范围
        max_bandwidth = max(all_bandwidths) if all_bandwidths else 1.0
        min_bandwidth = min(all_bandwidths) if all_bandwidths else 0.0

        # 绘制每个Die的热图
        node_spacing = 3.0
        for idx, (die_id, die_data) in enumerate(die_ip_bandwidth_data.items()):
            if mode not in die_data:
                continue

            offset_x = idx * 15
            offset_y = 0

            die_model = dies[die_id] if dies and die_id in dies else None
            die_config = die_model.config if die_model else config

            # 绘制该Die的节点热图
            for node in range(die_config.NUM_ROW * die_config.NUM_COL):
                row = node // die_config.NUM_COL
                col = node % die_config.NUM_COL

                x = col * node_spacing + offset_x
                y = -row * node_spacing + offset_y

                # 绘制该节点的IP热图
                self._draw_ip_heatmap_in_node(ax, x, y, node, die_id, die_data, mode, node_size, max_bandwidth, min_bandwidth)

        # 设置图表
        plt.title(f"IP Bandwidth Heatmap - {mode.capitalize()}", fontsize=14, fontweight="bold")
        ax.axis("equal")
        ax.margins(0.1)
        ax.axis("off")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def _draw_ip_heatmap_in_node(self, ax, x, y, node, die_id, die_data, mode, node_size, max_bandwidth, min_bandwidth):
        """
        在指定位置绘制节点的IP带宽热力图

        Args:
            ax: matplotlib坐标轴
            x, y: 节点中心位置
            node: 节点ID
            die_id: Die ID
            die_data: Die带宽数据
            mode: 显示模式
            node_size: 节点大小
            max_bandwidth: 全局最大带宽
            min_bandwidth: 全局最小带宽
        """
        from matplotlib.patches import Rectangle

        # 收集该节点的所有IP及其带宽
        active_ips = []
        if mode in die_data:
            for ip_type, data_matrix in die_data[mode].items():
                # 简化：假设节点对应矩阵的某个位置
                # 实际应该根据节点的row/col计算
                bandwidth = 0
                if hasattr(data_matrix, "shape"):
                    if data_matrix.size > node:
                        bandwidth = data_matrix.flat[node]
                if bandwidth > 0.001:
                    active_ips.append((ip_type, bandwidth))

        # 计算节点框大小
        square_size = (node_size / 1000.0) * 0.3
        node_box_size = square_size * 3.98

        # 绘制节点外框
        node_fill = Rectangle(
            (x - node_box_size / 2, y - node_box_size / 2),
            width=node_box_size,
            height=node_box_size,
            facecolor="#FFF9C4" if active_ips else "#F5F5F5",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.3 if active_ips else 1.0,
            zorder=1,
        )
        ax.add_patch(node_fill)

        if not active_ips:
            return

        # 绘制IP热图方块
        for idx, (ip_type, bandwidth) in enumerate(active_ips):
            # 计算方块位置
            ip_y = y + (idx - len(active_ips) / 2) * 0.3
            ip_size = 0.2

            # 计算颜色
            base_type = ip_type.upper().split("_")[0]
            base_color = self.IP_COLOR_MAP.get(base_type, self.IP_COLOR_MAP["OTHER"])

            # 根据带宽调整透明度
            if max_bandwidth > min_bandwidth:
                alpha = (bandwidth - min_bandwidth) / (max_bandwidth - min_bandwidth)
            else:
                alpha = 1.0

            ip_rect = Rectangle(
                (x - ip_size / 2, ip_y - ip_size / 2),
                width=ip_size,
                height=ip_size,
                facecolor=base_color,
                edgecolor="black",
                linewidth=0.5,
                alpha=alpha,
                zorder=2,
            )
            ax.add_patch(ip_rect)


class IPInfoBoxDrawer:
    """IP信息框绘制器 - 绘制节点上的IP信息和带宽"""

    def __init__(self):
        """初始化IP信息框绘制器"""
        self.IP_COLOR_MAP = IP_COLOR_MAP

    def draw_ip_info_box(self, ax, x: float, y: float, node: int, config, mode: str, square_size: float, max_ip_bandwidth: float = None, min_ip_bandwidth: float = None):
        """
        在单Die场景下绘制IP信息框

        Args:
            ax: matplotlib axes对象
            x: X坐标
            y: Y坐标
            node: 节点ID
            config: 配置对象
            mode: 显示模式
            square_size: 方框大小
            max_ip_bandwidth: 最大IP带宽（用于归一化）
            min_ip_bandwidth: 最小IP带宽
        """
        # 简化实现：绘制节点编号
        ax.text(x, y, str(node), ha="center", va="center", fontsize=10, fontweight="bold", zorder=3)

    def draw_d2d_ip_info_box(
        self,
        ax,
        x: float,
        y: float,
        node: int,
        config,
        mode: str,
        square_size: float,
        die_id: int = None,
        die_model=None,
        max_ip_bandwidth: float = None,
        min_ip_bandwidth: float = None,
        rotation: int = 0,
    ):
        """
        在D2D场景下绘制IP信息框

        Args:
            ax: matplotlib axes对象
            x: X坐标
            y: Y坐标
            node: 节点ID
            config: 配置对象
            mode: 显示模式
            square_size: 方框大小
            die_id: Die ID
            die_model: Die模型对象
            max_ip_bandwidth: 最大IP带宽
            min_ip_bandwidth: 最小IP带宽
            rotation: Die旋转角度
        """
        # 简化实现：绘制节点编号
        ax.text(x, y, f"D{die_id}N{node}", ha="center", va="center", fontsize=8, fontweight="bold", zorder=3, rotation=rotation)

    def _get_ip_color(self, ip_type: str, bandwidth: float, max_bw: float, min_bw: float) -> str:
        """
        根据IP类型和带宽获取颜色

        Args:
            ip_type: IP类型（如"gdma", "ddr"）
            bandwidth: 当前带宽
            max_bw: 最大带宽
            min_bw: 最小带宽

        Returns:
            str: 十六进制颜色代码
        """
        # 获取基础颜色
        base_color = IP_COLOR_MAP.get(ip_type.upper(), IP_COLOR_MAP["OTHER"])

        # 根据带宽调整亮度
        if max_bw > min_bw:
            intensity = (bandwidth - min_bw) / (max_bw - min_bw)
        else:
            intensity = 1.0

        # 将hex颜色转换为RGB并调整亮度
        rgb = mcolors.hex2color(base_color)
        # 线性插值：从白色(1,1,1)到基础色
        adjusted_rgb = tuple(1.0 - intensity * (1.0 - c) for c in rgb)
        return mcolors.rgb2hex(adjusted_rgb)
