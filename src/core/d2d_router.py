# d2d_router.py
"""
D2D路由器模块

提供基于配置驱动的D2D节点选择算法，支持负载均衡和多拓扑。
"""

from typing import List, Dict, Tuple, Optional


class D2DRouter:
    """
    D2D路由器，负责选择跨Die通信的D2D节点

    主要功能：
    1. 基于D2D_PAIRS配置构建路由表
    2. 使用目标节点取模进行负载均衡
    3. 支持任意拓扑和Die数量
    """

    def __init__(self, config):
        """
        初始化D2D路由器

        Args:
            config: D2D配置对象，包含D2D_PAIRS
        """
        self.d2d_pairs = getattr(config, "D2D_PAIRS", [])
        if not self.d2d_pairs:
            raise ValueError("D2D_PAIRS配置为空，无法初始化D2D路由器")

        # 构建路由表：{(src_die, dst_die): [可用的D2D节点列表]}
        self.d2d_routing_table = self._build_routing_table()

    def _build_routing_table(self) -> Dict[Tuple[int, int], List[int]]:
        """
        构建路由表：从D2D_PAIRS中提取每个Die到其他Die的可用连接

        Returns:
            routing_table: {(src_die, dst_die): [可用的D2D节点列表]}
        """
        routing_table = {}

        # 遍历D2D_PAIRS: [(die0_id, node0, die1_id, node1), ...]
        for die0_id, node0, die1_id, node1 in self.d2d_pairs:
            # Die0 -> Die1 的连接
            key = (die0_id, die1_id)
            if key not in routing_table:
                routing_table[key] = []
            routing_table[key].append(node0)

            # Die1 -> Die0 的连接（双向）
            key = (die1_id, die0_id)
            if key not in routing_table:
                routing_table[key] = []
            routing_table[key].append(node1)

        # 对每个连接的节点列表排序，确保一致性
        for key in routing_table:
            routing_table[key].sort()

        return routing_table

    def select_d2d_node(self, src_die: int, dst_die: int, dst_node: int) -> Optional[int]:
        """
        选择D2D节点进行跨Die传输

        算法：
        1. 判断是否跨Die（src_die != dst_die）
        2. 获取可用的D2D节点列表
        3. 使用dst_node % len(可用节点) 进行负载均衡

        Args:
            src_die: 源Die ID
            dst_die: 目标Die ID
            dst_node: 目标节点编号（用于负载均衡）

        Returns:
            selected_d2d_node: 选中的D2D节点位置，如果不需要跨Die则返回None
        """
        # 检查是否跨Die
        if src_die == dst_die:
            return None  # 不需要D2D路由

        # 获取可用的D2D节点
        available_nodes = self.get_available_d2d_nodes(src_die, dst_die)

        if not available_nodes:
            raise ValueError(f"没有从Die{src_die}到Die{dst_die}的D2D连接")

        # 使用目标节点号进行负载均衡
        index = dst_node % len(available_nodes)
        selected_node = available_nodes[index]

        # print(f"[D2D路由] Die{src_die}->Die{dst_die}, 目标节点{dst_node}, "
        #   f"可用D2D节点{available_nodes}, 选择节点{selected_node}(索引{index})")

        return selected_node

    def get_available_d2d_nodes(self, src_die: int, dst_die: int) -> List[int]:
        """
        获取从src_die到dst_die的所有可用D2D节点

        Args:
            src_die: 源Die ID
            dst_die: 目标Die ID

        Returns:
            available_nodes: 可用D2D节点列表
        """
        key = (src_die, dst_die)
        return self.d2d_routing_table.get(key, [])

    def is_cross_die(self, src_die: int, dst_die: int) -> bool:
        """
        判断是否为跨Die通信

        Args:
            src_die: 源Die ID
            dst_die: 目标Die ID

        Returns:
            bool: True表示跨Die，False表示同Die内部
        """
        return src_die != dst_die

    def get_routing_stats(self) -> Dict:
        """
        获取路由统计信息

        Returns:
            stats: 包含连接数、最大负载等信息的字典
        """
        stats = {"total_connections": len(self.d2d_routing_table), "total_pairs": len(self.d2d_pairs), "connections_per_die_pair": {}}

        for (src_die, dst_die), nodes in self.d2d_routing_table.items():
            connection_count = len(nodes)
            stats["connections_per_die_pair"][f"Die{src_die}->Die{dst_die}"] = connection_count

        return stats
