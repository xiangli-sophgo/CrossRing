"""
静态链路带宽分析器

基于 StaticFlow 项目的静态带宽计算逻辑，为 CrossRing 流量生成工具提供链路带宽预估功能。
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


class StaticBandwidthAnalyzer:
    """静态链路带宽分析器"""

    def __init__(self, topo_type: str, node_ips: Dict[int, List[str]], configs: List):
        """
        初始化静态带宽分析器

        Args:
            topo_type: 拓扑类型，如 "5x4"
            node_ips: 节点IP映射，格式 {节点ID: [IP列表]}
            configs: TrafficConfig列表
        """
        self.topo_type = topo_type
        self.node_ips = node_ips
        self.configs = configs

        # 解析拓扑参数
        self.num_row, self.num_col = self._parse_topo(topo_type)

        # 链路带宽字典: {((src_x, src_y), (dst_x, dst_y)): bandwidth}
        self.link_bandwidth: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}

    def _parse_topo(self, topo_type: str) -> Tuple[int, int]:
        """解析拓扑类型字符串"""
        parts = topo_type.split("x")
        rows = int(parts[0])
        cols = int(parts[1])
        return rows, cols

    def _node_id_to_pos(self, node_id: int) -> Tuple[int, int]:
        """
        将节点ID转换为坐标

        CrossRing节点ID: row_idx * num_col + col_idx
        返回坐标: (col_idx, row_idx)
        """
        row_idx = node_id // self.num_col
        col_idx = node_id % self.num_col
        return (col_idx, row_idx)

    def _pos_to_node_id(self, pos: Tuple[int, int]) -> int:
        """坐标转换为节点ID"""
        col_idx, row_idx = pos
        return row_idx * self.num_col + col_idx

    def _convert_flow_config(self) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
        """
        转换TrafficConfig列表为flow字典

        Returns:
            {((src_x, src_y), (dst_x, dst_y)): bandwidth}
        """
        flow = {}

        for config in self.configs:
            # 提取源节点列表
            src_nodes = []
            for ip_type, nodes in config.src_map.items():
                src_nodes.extend(nodes)

            # 提取目标节点列表
            dst_nodes = []
            for ip_type, nodes in config.dst_map.items():
                dst_nodes.extend(nodes)

            # 计算每对源-目标的带宽
            if src_nodes and dst_nodes:
                total_bandwidth = config.speed  # GB/s
                bandwidth_per_pair = total_bandwidth / (len(src_nodes) * len(dst_nodes))

                # 生成流量字典
                for src_node in src_nodes:
                    for dst_node in dst_nodes:
                        src_pos = self._node_id_to_pos(src_node)
                        dst_pos = self._node_id_to_pos(dst_node)

                        flow_key = (src_pos, dst_pos)
                        if flow_key not in flow:
                            flow[flow_key] = 0.0
                        flow[flow_key] += bandwidth_per_pair

        return flow

    def _static_initial(self):
        """初始化所有可能的链路，带宽设为0"""
        self.link_bandwidth = {}
        for i in range(self.num_col):
            for j in range(self.num_row):
                pos = (i, j)
                # 东向链路
                if i + 1 < self.num_col:
                    self.link_bandwidth[(pos, (i + 1, j))] = 0
                # 西向链路
                if i - 1 >= 0:
                    self.link_bandwidth[(pos, (i - 1, j))] = 0
                # 北向链路 (y轴正方向)
                if j + 1 < self.num_row:
                    self.link_bandwidth[(pos, (i, j + 1))] = 0
                # 南向链路 (y轴负方向)
                if j - 1 >= 0:
                    self.link_bandwidth[(pos, (i, j - 1))] = 0

    def _link_bandwidth_xy(self, source: Tuple[int, int], target: Tuple[int, int], bandwidth: float):
        """
        使用XY路由计算链路带宽

        XY路由: 先沿X方向(水平)，再沿Y方向(垂直)
        """
        node = source
        while True:
            dx = target[0] - node[0]
            dy = target[1] - node[1]

            # 先处理X方向
            if dx > 0:
                next_node = (node[0] + 1, node[1])
            elif dx < 0:
                next_node = (node[0] - 1, node[1])
            # X方向完成，处理Y方向
            elif dy > 0:
                next_node = (node[0], node[1] + 1)
            elif dy < 0:
                next_node = (node[0], node[1] - 1)
            else:
                # 到达目标
                return

            link_key = (node, next_node)
            if link_key not in self.link_bandwidth:
                self.link_bandwidth[link_key] = 0
            self.link_bandwidth[link_key] += bandwidth
            node = next_node

    def _link_bandwidth_yx(self, source: Tuple[int, int], target: Tuple[int, int], bandwidth: float):
        """
        使用YX路由计算链路带宽

        YX路由: 先沿Y方向(垂直)，再沿X方向(水平)
        """
        node = source
        while True:
            dx = target[0] - node[0]
            dy = target[1] - node[1]

            # 先处理Y方向
            if dy > 0:
                next_node = (node[0], node[1] + 1)
            elif dy < 0:
                next_node = (node[0], node[1] - 1)
            # Y方向完成，处理X方向
            elif dx > 0:
                next_node = (node[0] + 1, node[1])
            elif dx < 0:
                next_node = (node[0] - 1, node[1])
            else:
                # 到达目标
                return

            link_key = (node, next_node)
            if link_key not in self.link_bandwidth:
                self.link_bandwidth[link_key] = 0
            self.link_bandwidth[link_key] += bandwidth
            node = next_node

    def _compute_bandwidth(self, routing_type: str):
        """根据路由类型计算所有流量的链路带宽"""
        flow = self._convert_flow_config()

        if routing_type == "XY":
            for (src, dst), bw in flow.items():
                self._link_bandwidth_xy(src, dst, bw)
        elif routing_type == "YX":
            for (src, dst), bw in flow.items():
                self._link_bandwidth_yx(src, dst, bw)
        else:
            raise ValueError(f"不支持的路由类型: {routing_type}. 必须是 'XY' 或 'YX'")

    def compute(self, routing_type: str = "XY") -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
        """
        计算静态链路带宽

        Args:
            routing_type: 路由算法类型，"XY" 或 "YX"

        Returns:
            链路带宽字典，格式: {((src_x, src_y), (dst_x, dst_y)): bandwidth_GB/s}
        """
        # 初始化链路
        self._static_initial()

        # 计算带宽
        self._compute_bandwidth(routing_type)

        return self.link_bandwidth

    def get_statistics(self) -> Dict[str, float]:
        """
        获取链路带宽统计信息

        Returns:
            统计信息字典，包含: max_bandwidth, sum_bandwidth, avg_bandwidth, num_active_links
        """
        if not self.link_bandwidth:
            return {
                "max_bandwidth": 0.0,
                "sum_bandwidth": 0.0,
                "avg_bandwidth": 0.0,
                "num_active_links": 0,
            }

        # 只统计有流量的链路
        active_bandwidths = [bw for bw in self.link_bandwidth.values() if bw > 0]

        if not active_bandwidths:
            return {
                "max_bandwidth": 0.0,
                "sum_bandwidth": 0.0,
                "avg_bandwidth": 0.0,
                "num_active_links": 0,
            }

        return {
            "max_bandwidth": max(active_bandwidths),
            "sum_bandwidth": sum(active_bandwidths),
            "avg_bandwidth": sum(active_bandwidths) / len(active_bandwidths),
            "num_active_links": len(active_bandwidths),
        }


def compute_link_bandwidth(
    topo_type: str,
    node_ips: Dict[int, List[str]],
    configs: List,
    routing_type: str = "XY",
) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
    """
    便捷函数: 计算静态链路带宽

    Args:
        topo_type: 拓扑类型，如 "5x4"
        node_ips: 节点IP映射，格式 {节点ID: [IP列表]}
        configs: TrafficConfig列表
        routing_type: 路由算法，"XY" 或 "YX"

    Returns:
        链路带宽字典: {((src_x, src_y), (dst_x, dst_y)): bandwidth_GB/s}
    """
    analyzer = StaticBandwidthAnalyzer(topo_type, node_ips, configs)
    return analyzer.compute(routing_type)
