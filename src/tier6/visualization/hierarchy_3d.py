"""
3D交互式层级拓扑图渲染器

使用Plotly生成可交互的3D层级网络拓扑可视化
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import plotly.graph_objects as go

from .shapes_3d import (
    Shape3DFactory,
    TopologyGenerator,
    LayoutCalculator,
    LEVEL_COLORS,
    LEVEL_Z_POSITIONS,
)


@dataclass
class LevelConfig:
    """层级配置"""
    level: str           # 层级名称: die, chip, board, server, pod
    count: int           # 节点数量
    topology: str        # 拓扑类型: mesh, all_to_all, ring
    visible: bool = True # 是否可见


class Hierarchy3DRenderer:
    """3D交互式层级拓扑图渲染器"""

    # 层级顺序（从下到上）
    LEVEL_ORDER = ["die", "chip", "board", "server", "pod"]

    # 层级显示名称
    LEVEL_NAMES = {
        "die": "Die (晶粒)",
        "chip": "Chip (芯片)",
        "board": "Board (电路板)",
        "server": "Server (服务器)",
        "pod": "Pod (机柜)",
    }

    def __init__(self):
        self.shape_factory = Shape3DFactory()

    def render(self,
               level_configs: List[LevelConfig],
               show_inter_level_connections: bool = True,
               layout_type: str = "circular") -> go.Figure:
        """
        渲染3D拓扑图

        Args:
            level_configs: 每层配置列表
            show_inter_level_connections: 是否显示层间连接
            layout_type: 布局类型 (circular, grid)

        Returns:
            Plotly Figure对象
        """
        fig = go.Figure()

        # 存储每层的节点位置，用于层间连接
        level_positions = {}

        # 按层级顺序处理
        for config in level_configs:
            level = config.level
            z_pos = LEVEL_Z_POSITIONS.get(level, 0)

            # 根据层级调整布局半径
            level_index = self.LEVEL_ORDER.index(level) if level in self.LEVEL_ORDER else 0
            radius = 2.0 + level_index * 0.5

            # 计算节点位置
            if layout_type == "circular":
                positions = LayoutCalculator.circular_layout(config.count, radius, z_pos)
            else:
                positions = LayoutCalculator.grid_layout(config.count, 1.5, z_pos)

            level_positions[level] = positions

            # 添加节点
            self._add_level_nodes(fig, level, positions, config.visible)

            # 添加层内连接
            if config.count > 1:
                edges = TopologyGenerator.generate_edges(config.topology, config.count)
                self._add_level_edges(fig, level, positions, edges, config.visible)

        # 添加层间连接
        if show_inter_level_connections:
            self._add_inter_level_connections(fig, level_configs, level_positions)

        # 添加交互控件
        self._add_interactive_controls(fig, level_configs)

        # 配置3D场景
        self._configure_scene(fig)

        return fig

    def _add_level_nodes(self,
                         fig: go.Figure,
                         level: str,
                         positions: List[Tuple[float, float, float]],
                         visible: bool = True):
        """添加某一层级的所有节点"""
        color = LEVEL_COLORS.get(level, "#999999")

        for i, pos in enumerate(positions):
            # 获取形状数据
            shape_data = self.shape_factory.create_shape_for_level(level, pos)

            fig.add_trace(go.Mesh3d(
                x=shape_data["x"],
                y=shape_data["y"],
                z=shape_data["z"],
                i=shape_data["i"],
                j=shape_data["j"],
                k=shape_data["k"],
                color=color,
                opacity=0.8,
                name=f"{self.LEVEL_NAMES.get(level, level)} #{i}",
                hoverinfo="name",
                visible=visible,
                legendgroup=level,
                showlegend=(i == 0),  # 只有第一个节点显示图例
            ))

    def _add_level_edges(self,
                         fig: go.Figure,
                         level: str,
                         positions: List[Tuple[float, float, float]],
                         edges: List[Tuple[int, int]],
                         visible: bool = True):
        """添加某一层级的内部连接"""
        color = LEVEL_COLORS.get(level, "#999999")

        # 收集所有边的坐标
        x_coords = []
        y_coords = []
        z_coords = []

        for src, dst in edges:
            src_pos = positions[src]
            dst_pos = positions[dst]

            x_coords.extend([src_pos[0], dst_pos[0], None])
            y_coords.extend([src_pos[1], dst_pos[1], None])
            z_coords.extend([src_pos[2], dst_pos[2], None])

        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="lines",
            line=dict(color=color, width=3),
            name=f"{self.LEVEL_NAMES.get(level, level)} 连接",
            hoverinfo="none",
            visible=visible,
            legendgroup=f"{level}_edges",
            showlegend=True,
        ))

    def _add_inter_level_connections(self,
                                     fig: go.Figure,
                                     level_configs: List[LevelConfig],
                                     level_positions: Dict[str, List[Tuple]]):
        """添加层间连接（父子关系）"""
        # 按层级顺序排列
        sorted_levels = sorted(
            [c.level for c in level_configs],
            key=lambda x: self.LEVEL_ORDER.index(x) if x in self.LEVEL_ORDER else 99
        )

        x_coords = []
        y_coords = []
        z_coords = []

        # 连接相邻层级
        for i in range(len(sorted_levels) - 1):
            lower_level = sorted_levels[i]
            upper_level = sorted_levels[i + 1]

            lower_positions = level_positions.get(lower_level, [])
            upper_positions = level_positions.get(upper_level, [])

            if not lower_positions or not upper_positions:
                continue

            # 简单策略：每个上层节点连接若干下层节点
            lower_per_upper = max(1, len(lower_positions) // len(upper_positions))

            for u_idx, u_pos in enumerate(upper_positions):
                # 计算这个上层节点管理的下层节点范围
                start = u_idx * lower_per_upper
                end = min(start + lower_per_upper, len(lower_positions))

                for l_idx in range(start, end):
                    l_pos = lower_positions[l_idx]
                    x_coords.extend([l_pos[0], u_pos[0], None])
                    y_coords.extend([l_pos[1], u_pos[1], None])
                    z_coords.extend([l_pos[2], u_pos[2], None])

        if x_coords:
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode="lines",
                line=dict(color="#cccccc", width=1, dash="dash"),
                name="层间连接",
                hoverinfo="none",
                visible=True,
                legendgroup="inter_level",
                showlegend=True,
            ))

    def _add_interactive_controls(self,
                                  fig: go.Figure,
                                  level_configs: List[LevelConfig]):
        """添加交互控件"""
        # 创建层级可见性按钮
        buttons = []

        # "显示全部"按钮
        buttons.append(dict(
            label="显示全部",
            method="update",
            args=[{"visible": True}],
        ))

        # 每个层级的单独按钮
        for config in level_configs:
            level = config.level
            level_name = self.LEVEL_NAMES.get(level, level)

            # 计算哪些trace属于这个层级
            # 这里简化处理，实际需要根据trace的legendgroup来判断
            buttons.append(dict(
                label=f"仅 {level_name}",
                method="update",
                args=[{"visible": "legendonly"}],  # 需要更精细的控制
            ))

        # 添加dropdown菜单
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    x=0.02,
                    y=0.98,
                    xanchor="left",
                    yanchor="top",
                    buttons=buttons,
                    showactive=True,
                    bgcolor="white",
                    bordercolor="#cccccc",
                ),
            ],
        )

    def _configure_scene(self, fig: go.Figure):
        """配置3D场景"""
        fig.update_layout(
            title=dict(
                text="Tier6+ 3D 层级网络拓扑图",
                x=0.5,
                xanchor="center",
            ),
            scene=dict(
                xaxis=dict(
                    showgrid=True,
                    gridcolor="#eeeeee",
                    showticklabels=False,
                    title="",
                    showbackground=True,
                    backgroundcolor="#f8f8f8",
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#eeeeee",
                    showticklabels=False,
                    title="",
                    showbackground=True,
                    backgroundcolor="#f8f8f8",
                ),
                zaxis=dict(
                    showgrid=True,
                    gridcolor="#eeeeee",
                    showticklabels=False,
                    title="层级",
                    showbackground=True,
                    backgroundcolor="#f8f8f8",
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    up=dict(x=0, y=0, z=1),
                ),
                aspectmode="data",
            ),
            width=1000,
            height=800,
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(
                x=0.98,
                y=0.98,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#cccccc",
                borderwidth=1,
            ),
        )

    def render_default(self) -> go.Figure:
        """
        使用默认配置渲染

        Returns:
            Plotly Figure对象
        """
        default_configs = [
            LevelConfig(level="die", count=4, topology="mesh"),
            LevelConfig(level="chip", count=2, topology="mesh"),
            LevelConfig(level="board", count=2, topology="mesh"),
            LevelConfig(level="server", count=2, topology="mesh"),
            LevelConfig(level="pod", count=1, topology="mesh"),
        ]
        return self.render(default_configs)

    def save_html(self, fig: go.Figure, path: str):
        """保存为HTML文件"""
        fig.write_html(path)
