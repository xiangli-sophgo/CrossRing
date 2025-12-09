"""
3D形状生成器

为不同层级的网络组件生成3D形状
"""

from typing import Dict, List, Tuple
import numpy as np


# 层级颜色定义
LEVEL_COLORS = {
    "die": "#f5222d",      # 红色 - 芯片晶粒
    "chip": "#722ed1",     # 紫色 - 芯片封装
    "board": "#52c41a",    # 绿色 - 电路板
    "server": "#1890ff",   # 蓝色 - 服务器
    "pod": "#fa8c16",      # 橙色 - 机柜
}

# 层级Z轴高度
LEVEL_Z_POSITIONS = {
    "die": 0,
    "chip": 3,
    "board": 6,
    "server": 9,
    "pod": 12,
}


class Shape3DFactory:
    """3D形状工厂 - 生成Plotly Mesh3d所需的顶点和面数据"""

    @staticmethod
    def create_cube(center: Tuple[float, float, float],
                    size_x: float = 1.0,
                    size_y: float = 1.0,
                    size_z: float = 1.0) -> Dict:
        """
        创建立方体/长方体

        Args:
            center: 中心点坐标 (x, y, z)
            size_x, size_y, size_z: 三个方向的尺寸

        Returns:
            包含顶点和面索引的字典
        """
        cx, cy, cz = center
        hx, hy, hz = size_x / 2, size_y / 2, size_z / 2

        # 8个顶点
        vertices = np.array([
            [cx - hx, cy - hy, cz - hz],  # 0
            [cx + hx, cy - hy, cz - hz],  # 1
            [cx + hx, cy + hy, cz - hz],  # 2
            [cx - hx, cy + hy, cz - hz],  # 3
            [cx - hx, cy - hy, cz + hz],  # 4
            [cx + hx, cy - hy, cz + hz],  # 5
            [cx + hx, cy + hy, cz + hz],  # 6
            [cx - hx, cy + hy, cz + hz],  # 7
        ])

        # 12个三角面 (6个面，每面2个三角形)
        faces = np.array([
            # 底面
            [0, 1, 2], [0, 2, 3],
            # 顶面
            [4, 6, 5], [4, 7, 6],
            # 前面
            [0, 5, 1], [0, 4, 5],
            # 后面
            [3, 2, 6], [3, 6, 7],
            # 左面
            [0, 3, 7], [0, 7, 4],
            # 右面
            [1, 5, 6], [1, 6, 2],
        ])

        return {
            "x": vertices[:, 0].tolist(),
            "y": vertices[:, 1].tolist(),
            "z": vertices[:, 2].tolist(),
            "i": faces[:, 0].tolist(),
            "j": faces[:, 1].tolist(),
            "k": faces[:, 2].tolist(),
        }

    @staticmethod
    def create_die(center: Tuple[float, float, float], size: float = 0.4) -> Dict:
        """
        创建Die形状 - 小立方体（芯片晶粒）

        Args:
            center: 中心点坐标
            size: 边长

        Returns:
            Mesh3d数据
        """
        return Shape3DFactory.create_cube(center, size, size, size * 0.3)

    @staticmethod
    def create_chip(center: Tuple[float, float, float], size: float = 0.8) -> Dict:
        """
        创建Chip形状 - 扁平方块（BGA封装）

        Args:
            center: 中心点坐标
            size: 边长

        Returns:
            Mesh3d数据
        """
        return Shape3DFactory.create_cube(center, size, size, size * 0.2)

    @staticmethod
    def create_board(center: Tuple[float, float, float],
                     width: float = 1.5,
                     depth: float = 1.0) -> Dict:
        """
        创建Board形状 - 薄矩形板（PCB）

        Args:
            center: 中心点坐标
            width: 宽度
            depth: 深度

        Returns:
            Mesh3d数据
        """
        return Shape3DFactory.create_cube(center, width, depth, 0.1)

    @staticmethod
    def create_server(center: Tuple[float, float, float],
                      width: float = 2.0,
                      depth: float = 0.8) -> Dict:
        """
        创建Server形状 - 扁长方体（机架服务器）

        Args:
            center: 中心点坐标
            width: 宽度
            depth: 深度

        Returns:
            Mesh3d数据
        """
        return Shape3DFactory.create_cube(center, width, depth, 0.3)

    @staticmethod
    def create_pod(center: Tuple[float, float, float],
                   width: float = 1.2,
                   height: float = 2.5) -> Dict:
        """
        创建Pod形状 - 高方柱（机柜）

        Args:
            center: 中心点坐标
            width: 宽度
            height: 高度

        Returns:
            Mesh3d数据
        """
        return Shape3DFactory.create_cube(center, width, width, height)

    @classmethod
    def create_shape_for_level(cls, level: str,
                               center: Tuple[float, float, float]) -> Dict:
        """
        根据层级创建对应形状

        Args:
            level: 层级名称 (die, chip, board, server, pod)
            center: 中心点坐标

        Returns:
            Mesh3d数据
        """
        creators = {
            "die": cls.create_die,
            "chip": cls.create_chip,
            "board": cls.create_board,
            "server": cls.create_server,
            "pod": cls.create_pod,
        }

        creator = creators.get(level)
        if not creator:
            raise ValueError(f"未知层级: {level}")

        return creator(center)


class TopologyGenerator:
    """拓扑连接生成器"""

    @staticmethod
    def generate_mesh_edges(num_nodes: int) -> List[Tuple[int, int]]:
        """
        生成全连接(Mesh)拓扑的边

        Args:
            num_nodes: 节点数量

        Returns:
            边列表 [(源节点索引, 目标节点索引), ...]
        """
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edges.append((i, j))
        return edges

    @staticmethod
    def generate_all_to_all_edges(num_nodes: int,
                                   group_size: int = 2) -> List[Tuple[int, int]]:
        """
        生成分组全连接(All-to-All)拓扑的边

        Args:
            num_nodes: 节点数量
            group_size: 每组节点数

        Returns:
            边列表
        """
        edges = []
        num_groups = (num_nodes + group_size - 1) // group_size

        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, num_nodes)
            # 组内全连接
            for i in range(start, end):
                for j in range(i + 1, end):
                    edges.append((i, j))

        return edges

    @staticmethod
    def generate_ring_edges(num_nodes: int) -> List[Tuple[int, int]]:
        """
        生成环形(Ring)拓扑的边

        Args:
            num_nodes: 节点数量

        Returns:
            边列表
        """
        if num_nodes < 2:
            return []

        edges = []
        for i in range(num_nodes):
            edges.append((i, (i + 1) % num_nodes))
        return edges

    @classmethod
    def generate_edges(cls, topology_type: str,
                       num_nodes: int,
                       **kwargs) -> List[Tuple[int, int]]:
        """
        根据拓扑类型生成边

        Args:
            topology_type: 拓扑类型 (mesh, all_to_all, ring)
            num_nodes: 节点数量
            **kwargs: 额外参数

        Returns:
            边列表
        """
        generators = {
            "mesh": cls.generate_mesh_edges,
            "all_to_all": lambda n: cls.generate_all_to_all_edges(n, kwargs.get("group_size", 2)),
            "ring": cls.generate_ring_edges,
        }

        generator = generators.get(topology_type)
        if not generator:
            raise ValueError(f"未知拓扑类型: {topology_type}")

        return generator(num_nodes)


class LayoutCalculator:
    """节点布局计算器"""

    @staticmethod
    def circular_layout(num_nodes: int,
                        radius: float = 2.0,
                        z: float = 0.0) -> List[Tuple[float, float, float]]:
        """
        圆形布局

        Args:
            num_nodes: 节点数量
            radius: 半径
            z: Z轴高度

        Returns:
            节点坐标列表 [(x, y, z), ...]
        """
        if num_nodes == 0:
            return []
        if num_nodes == 1:
            return [(0.0, 0.0, z)]

        positions = []
        for i in range(num_nodes):
            angle = 2 * np.pi * i / num_nodes
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append((x, y, z))

        return positions

    @staticmethod
    def grid_layout(num_nodes: int,
                    spacing: float = 1.5,
                    z: float = 0.0) -> List[Tuple[float, float, float]]:
        """
        网格布局

        Args:
            num_nodes: 节点数量
            spacing: 节点间距
            z: Z轴高度

        Returns:
            节点坐标列表
        """
        if num_nodes == 0:
            return []

        cols = int(np.ceil(np.sqrt(num_nodes)))
        rows = int(np.ceil(num_nodes / cols))

        # 居中偏移
        offset_x = (cols - 1) * spacing / 2
        offset_y = (rows - 1) * spacing / 2

        positions = []
        for i in range(num_nodes):
            row = i // cols
            col = i % cols
            x = col * spacing - offset_x
            y = row * spacing - offset_y
            positions.append((x, y, z))

        return positions
