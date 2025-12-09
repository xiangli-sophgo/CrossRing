"""
拓扑生成器

生成3D拓扑的节点位置和连接数据
"""

from typing import List, Dict, Tuple
import math


# 层级颜色定义
LEVEL_COLORS = {
    "die": "#f5222d",      # 红色
    "chip": "#722ed1",     # 紫色
    "board": "#52c41a",    # 绿色
    "server": "#1890ff",   # 蓝色
    "pod": "#fa8c16",      # 橙色
}

# 层级Z轴高度
LEVEL_Z_POSITIONS = {
    "die": 0,
    "chip": 3,
    "board": 6,
    "server": 9,
    "pod": 12,
}

# 层级顺序
LEVEL_ORDER = ["die", "chip", "board", "server", "pod"]


class TopologyGenerator:
    """拓扑生成器"""

    def generate(self,
                 levels: List[Dict],
                 show_inter_level: bool = True,
                 layout: str = "circular") -> Dict:
        """
        生成拓扑数据

        Args:
            levels: 层级配置列表
            show_inter_level: 是否生成层间连接
            layout: 布局类型 (circular, grid)

        Returns:
            包含nodes和edges的字典
        """
        nodes = []
        edges = []
        level_nodes = {}  # 存储每层的节点ID，用于层间连接

        for level_config in levels:
            level = level_config.get("level") if isinstance(level_config, dict) else level_config.level
            count = level_config.get("count") if isinstance(level_config, dict) else level_config.count
            topology = level_config.get("topology", "mesh") if isinstance(level_config, dict) else level_config.topology
            visible = level_config.get("visible", True) if isinstance(level_config, dict) else level_config.visible

            if not visible:
                continue

            z_pos = LEVEL_Z_POSITIONS.get(level, 0)
            color = LEVEL_COLORS.get(level, "#999999")

            # 根据层级调整半径
            level_index = LEVEL_ORDER.index(level) if level in LEVEL_ORDER else 0
            radius = 2.0 + level_index * 1.5

            # 计算节点位置
            if layout == "circular":
                positions = self._circular_layout(count, radius, z_pos)
            else:
                positions = self._grid_layout(count, 1.5, z_pos)

            # 生成节点
            level_node_ids = []
            for i, pos in enumerate(positions):
                node_id = f"{level}_{i}"
                nodes.append({
                    "id": node_id,
                    "level": level,
                    "position": list(pos),
                    "color": color,
                })
                level_node_ids.append(node_id)

            level_nodes[level] = level_node_ids

            # 生成层内连接
            if count > 1:
                intra_edges = self._generate_edges(topology, count)
                for src_idx, dst_idx in intra_edges:
                    edges.append({
                        "source": f"{level}_{src_idx}",
                        "target": f"{level}_{dst_idx}",
                        "type": "intra_level",
                    })

        # 生成层间连接
        if show_inter_level:
            inter_edges = self._generate_inter_level_edges(levels, level_nodes)
            edges.extend(inter_edges)

        return {"nodes": nodes, "edges": edges}

    def _circular_layout(self,
                         num_nodes: int,
                         radius: float,
                         z: float) -> List[Tuple[float, float, float]]:
        """圆形布局"""
        if num_nodes == 0:
            return []
        if num_nodes == 1:
            return [(0.0, 0.0, z)]

        positions = []
        for i in range(num_nodes):
            angle = 2 * math.pi * i / num_nodes
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions.append((x, y, z))

        return positions

    def _grid_layout(self,
                     num_nodes: int,
                     spacing: float,
                     z: float) -> List[Tuple[float, float, float]]:
        """网格布局"""
        if num_nodes == 0:
            return []

        cols = int(math.ceil(math.sqrt(num_nodes)))
        rows = int(math.ceil(num_nodes / cols))

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

    def _generate_edges(self,
                        topology: str,
                        num_nodes: int) -> List[Tuple[int, int]]:
        """根据拓扑类型生成边"""
        if topology == "mesh":
            return self._mesh_edges(num_nodes)
        elif topology == "all_to_all":
            return self._all_to_all_edges(num_nodes, group_size=2)
        elif topology == "ring":
            return self._ring_edges(num_nodes)
        else:
            return self._mesh_edges(num_nodes)

    def _mesh_edges(self, num_nodes: int) -> List[Tuple[int, int]]:
        """全连接边"""
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edges.append((i, j))
        return edges

    def _all_to_all_edges(self,
                          num_nodes: int,
                          group_size: int = 2) -> List[Tuple[int, int]]:
        """分组全连接边"""
        edges = []
        num_groups = (num_nodes + group_size - 1) // group_size

        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, num_nodes)
            for i in range(start, end):
                for j in range(i + 1, end):
                    edges.append((i, j))

        return edges

    def _ring_edges(self, num_nodes: int) -> List[Tuple[int, int]]:
        """环形边"""
        if num_nodes < 2:
            return []

        edges = []
        for i in range(num_nodes):
            edges.append((i, (i + 1) % num_nodes))
        return edges

    def _generate_inter_level_edges(self,
                                    levels: List[Dict],
                                    level_nodes: Dict[str, List[str]]) -> List[Dict]:
        """生成层间连接"""
        edges = []

        # 获取可见层级并排序
        visible_levels = []
        for level_config in levels:
            level = level_config.get("level") if isinstance(level_config, dict) else level_config.level
            visible = level_config.get("visible", True) if isinstance(level_config, dict) else level_config.visible
            if visible and level in level_nodes:
                visible_levels.append(level)

        # 按层级顺序排序
        visible_levels = sorted(
            visible_levels,
            key=lambda x: LEVEL_ORDER.index(x) if x in LEVEL_ORDER else 99
        )

        # 连接相邻层级
        for i in range(len(visible_levels) - 1):
            lower_level = visible_levels[i]
            upper_level = visible_levels[i + 1]

            lower_nodes = level_nodes.get(lower_level, [])
            upper_nodes = level_nodes.get(upper_level, [])

            if not lower_nodes or not upper_nodes:
                continue

            # 每个上层节点连接若干下层节点
            lower_per_upper = max(1, len(lower_nodes) // len(upper_nodes))

            for u_idx, u_node in enumerate(upper_nodes):
                start = u_idx * lower_per_upper
                end = min(start + lower_per_upper, len(lower_nodes))

                for l_idx in range(start, end):
                    edges.append({
                        "source": lower_nodes[l_idx],
                        "target": u_node,
                        "type": "inter_level",
                    })

        return edges
