"""
静态链路带宽分析器

基于 StaticFlow 项目的静态带宽计算逻辑，为 CrossRing 流量生成工具提供链路带宽预估功能。
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


class StaticBandwidthAnalyzer:
    """静态链路带宽分析器"""

    def __init__(self, topo_type: str, configs: List):
        """
        初始化静态带宽分析器

        Args:
            topo_type: 拓扑类型，如 "5x4"
            configs: TrafficConfig列表（包含src_map/dst_map）
        """
        self.topo_type = topo_type
        self.configs = configs

        # 解析拓扑参数
        self.num_row, self.num_col = self._parse_topo(topo_type)

        # 链路带宽字典: {((src_x, src_y), (dst_x, dst_y)): bandwidth}
        self.link_bandwidth: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}

        # 链路带宽组成: {link_key: [{src_node, src_ip, dst_node, dst_ip, bandwidth}, ...]}
        self.link_composition: Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[Dict]] = {}

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

    def _convert_flow_config(self) -> List[Dict]:
        """
        转换TrafficConfig列表为flow列表

        Returns:
            [{flow_src, flow_dst, src_node, src_ip, dst_node, dst_ip, bandwidth, req_type}, ...]
        """
        flows = []

        for config in self.configs:
            # 提取源节点和IP类型
            src_node_ips = []
            for ip_type, nodes in config.src_map.items():
                for node in nodes:
                    src_node_ips.append((node, ip_type))

            # 提取目标节点和IP类型
            dst_node_ips = []
            for ip_type, nodes in config.dst_map.items():
                for node in nodes:
                    dst_node_ips.append((node, ip_type))

            if src_node_ips and dst_node_ips:
                bandwidth_per_src = config.speed
                bandwidth_per_pair = bandwidth_per_src / len(dst_node_ips)
                req_type = getattr(config, 'req_type', 'W')

                for src_node, src_ip in src_node_ips:
                    for dst_node, dst_ip in dst_node_ips:
                        src_pos = self._node_id_to_pos(src_node)
                        dst_pos = self._node_id_to_pos(dst_node)

                        if req_type == 'R':
                            flow_src, flow_dst = dst_pos, src_pos
                        else:
                            flow_src, flow_dst = src_pos, dst_pos

                        flows.append({
                            'flow_src': flow_src,
                            'flow_dst': flow_dst,
                            'src_node': src_node,
                            'src_ip': src_ip,
                            'dst_node': dst_node,
                            'dst_ip': dst_ip,
                            'bandwidth': bandwidth_per_pair,
                            'req_type': req_type
                        })

        return flows

    def _static_initial(self):
        """初始化所有可能的链路，带宽设为0"""
        self.link_bandwidth = {}
        self.link_composition = {}
        for i in range(self.num_col):
            for j in range(self.num_row):
                pos = (i, j)
                # 东向链路
                if i + 1 < self.num_col:
                    self.link_bandwidth[(pos, (i + 1, j))] = 0
                    self.link_composition[(pos, (i + 1, j))] = []
                # 西向链路
                if i - 1 >= 0:
                    self.link_bandwidth[(pos, (i - 1, j))] = 0
                    self.link_composition[(pos, (i - 1, j))] = []
                # 北向链路 (y轴正方向)
                if j + 1 < self.num_row:
                    self.link_bandwidth[(pos, (i, j + 1))] = 0
                    self.link_composition[(pos, (i, j + 1))] = []
                # 南向链路 (y轴负方向)
                if j - 1 >= 0:
                    self.link_bandwidth[(pos, (i, j - 1))] = 0
                    self.link_composition[(pos, (i, j - 1))] = []

    def _link_bandwidth_xy(self, source: Tuple[int, int], target: Tuple[int, int], bandwidth: float, flow_info: Dict = None):
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
                self.link_composition[link_key] = []
            self.link_bandwidth[link_key] += bandwidth
            if flow_info:
                self.link_composition[link_key].append(flow_info)
            node = next_node

    def _link_bandwidth_yx(self, source: Tuple[int, int], target: Tuple[int, int], bandwidth: float, flow_info: Dict = None):
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
                self.link_composition[link_key] = []
            self.link_bandwidth[link_key] += bandwidth
            if flow_info:
                self.link_composition[link_key].append(flow_info)
            node = next_node

    def _compute_bandwidth(self, routing_type: str):
        """根据路由类型计算所有流量的链路带宽"""
        flows = self._convert_flow_config()

        if routing_type == "XY":
            for flow in flows:
                flow_info = {
                    'src_node': flow['src_node'],
                    'src_ip': flow['src_ip'],
                    'dst_node': flow['dst_node'],
                    'dst_ip': flow['dst_ip'],
                    'bandwidth': flow['bandwidth'],
                    'req_type': flow['req_type']
                }
                self._link_bandwidth_xy(flow['flow_src'], flow['flow_dst'], flow['bandwidth'], flow_info)
        elif routing_type == "YX":
            for flow in flows:
                flow_info = {
                    'src_node': flow['src_node'],
                    'src_ip': flow['src_ip'],
                    'dst_node': flow['dst_node'],
                    'dst_ip': flow['dst_ip'],
                    'bandwidth': flow['bandwidth'],
                    'req_type': flow['req_type']
                }
                self._link_bandwidth_yx(flow['flow_src'], flow['flow_dst'], flow['bandwidth'], flow_info)
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

    def get_link_composition(self) -> Dict[str, List[Dict]]:
        """
        获取链路带宽组成

        Returns:
            {link_key_str: [{src_node, src_ip, dst_node, dst_ip, bandwidth, req_type}, ...]}
        """
        result = {}
        for link_key, flows in self.link_composition.items():
            if flows:  # 只返回有流量的链路
                key_str = f"{link_key[0][0]},{link_key[0][1]}-{link_key[1][0]},{link_key[1][1]}"
                result[key_str] = flows
        return result

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
    configs: List,
    routing_type: str = "XY",
) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
    """
    便捷函数: 计算静态链路带宽

    Args:
        topo_type: 拓扑类型，如 "5x4"
        configs: TrafficConfig列表（包含src_map/dst_map）
        routing_type: 路由算法，"XY" 或 "YX"

    Returns:
        链路带宽字典: {((src_x, src_y), (dst_x, dst_y)): bandwidth_GB/s}
    """
    analyzer = StaticBandwidthAnalyzer(topo_type, configs)
    return analyzer.compute(routing_type)


class D2DStaticBandwidthAnalyzer:
    """D2D静态链路带宽分析器"""

    def __init__(
        self,
        topo_type: str,
        configs: List,
        d2d_pairs: List[Tuple[int, int, int, int]],
        num_dies: int = 2,
    ):
        """
        初始化D2D静态带宽分析器

        Args:
            topo_type: 拓扑类型，如 "5x4"
            configs: TrafficConfig列表（包含die_pairs和src_map/dst_map）
            d2d_pairs: D2D连接配对列表 [(src_die, src_node, dst_die, dst_node), ...]
            num_dies: Die数量
        """
        self.topo_type = topo_type
        self.configs = configs
        self.d2d_pairs = d2d_pairs
        self.num_dies = num_dies

        # 解析拓扑参数
        self.num_row, self.num_col = self._parse_topo(topo_type)

        # 每个Die的链路带宽字典: {die_id: {((src_x, src_y), (dst_x, dst_y)): bandwidth}}
        self.die_link_bandwidth: Dict[int, Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]] = {}

        # D2D跨Die链路带宽字典: {(src_die, src_node, dst_die, dst_node): bandwidth}
        self.d2d_link_bandwidth: Dict[Tuple[int, int, int, int], float] = {}

        # Die内部链路组成: {die_id: {((src_x, src_y), (dst_x, dst_y)): [flow_info, ...]}}
        self.die_link_composition: Dict[int, Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[Dict]]] = {}

        # D2D跨Die链路组成: {(src_die, src_node, dst_die, dst_node): [flow_info, ...]}
        self.d2d_link_composition: Dict[Tuple[int, int, int, int], List[Dict]] = {}

        # 构建D2D路由表
        self.d2d_routing_table = self._build_d2d_routing_table()

    def _parse_topo(self, topo_type: str) -> Tuple[int, int]:
        """解析拓扑类型字符串"""
        parts = topo_type.split("x")
        rows = int(parts[0])
        cols = int(parts[1])
        return rows, cols

    def _node_id_to_pos(self, node_id: int) -> Tuple[int, int]:
        """将节点ID转换为坐标"""
        row_idx = node_id // self.num_col
        col_idx = node_id % self.num_col
        return (col_idx, row_idx)

    def _build_d2d_routing_table(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        构建D2D路由表

        Returns:
            {(src_die, dst_die): [(src_node, dst_node), ...]}
        """
        routing_table = {}
        for src_die, src_node, dst_die, dst_node in self.d2d_pairs:
            key = (src_die, dst_die)
            if key not in routing_table:
                routing_table[key] = []
            routing_table[key].append((src_node, dst_node))

        # 排序确保一致性
        for key in routing_table:
            routing_table[key].sort(key=lambda x: x[0])

        return routing_table

    def _select_d2d_nodes(self, src_die: int, dst_die: int, dst_ip_id: int) -> Tuple[int, int]:
        """
        选择D2D节点（与运行时相同的算法）

        Args:
            src_die: 源Die ID
            dst_die: 目标Die ID
            dst_ip_id: 目标IP编号

        Returns:
            (d2d_sn_node, d2d_rn_node): D2D_SN节点和D2D_RN节点
        """
        key = (src_die, dst_die)
        available_pairs = self.d2d_routing_table.get(key, [])

        if not available_pairs:
            # 列出可用的Die连接
            available_connections = list(self.d2d_routing_table.keys())
            raise ValueError(
                f"D2D配置错误: 没有从Die{src_die}到Die{dst_die}的连接。"
                f"当前D2D配置仅支持以下连接: {available_connections}。"
                f"请检查流量配置中的die_pairs是否与D2D拓扑配置匹配。"
            )

        index = dst_ip_id % len(available_pairs)
        d2d_sn_node, d2d_rn_node = available_pairs[index]

        return d2d_sn_node, d2d_rn_node

    def _extract_ip_id(self, ip_type: str) -> int:
        """从IP类型中提取编号，如 'ddr_2' -> 2"""
        if '_' in ip_type:
            try:
                return int(ip_type.split('_')[1])
            except (IndexError, ValueError):
                return 0
        return 0

    def _static_initial(self, die_id: int):
        """初始化指定Die的所有可能链路"""
        self.die_link_bandwidth[die_id] = {}
        for i in range(self.num_col):
            for j in range(self.num_row):
                pos = (i, j)
                if i + 1 < self.num_col:
                    self.die_link_bandwidth[die_id][(pos, (i + 1, j))] = 0
                if i - 1 >= 0:
                    self.die_link_bandwidth[die_id][(pos, (i - 1, j))] = 0
                if j + 1 < self.num_row:
                    self.die_link_bandwidth[die_id][(pos, (i, j + 1))] = 0
                if j - 1 >= 0:
                    self.die_link_bandwidth[die_id][(pos, (i, j - 1))] = 0

    def _add_path_bandwidth_xy(
        self, die_id: int, source: Tuple[int, int], target: Tuple[int, int], bandwidth: float, flow_info: Dict = None
    ):
        """使用XY路由添加路径带宽到指定Die"""
        if die_id not in self.die_link_bandwidth:
            self._static_initial(die_id)
        if die_id not in self.die_link_composition:
            self.die_link_composition[die_id] = {}

        node = source
        while True:
            dx = target[0] - node[0]
            dy = target[1] - node[1]

            if dx > 0:
                next_node = (node[0] + 1, node[1])
            elif dx < 0:
                next_node = (node[0] - 1, node[1])
            elif dy > 0:
                next_node = (node[0], node[1] + 1)
            elif dy < 0:
                next_node = (node[0], node[1] - 1)
            else:
                return

            link_key = (node, next_node)
            if link_key not in self.die_link_bandwidth[die_id]:
                self.die_link_bandwidth[die_id][link_key] = 0
            self.die_link_bandwidth[die_id][link_key] += bandwidth

            # 记录链路组成
            if flow_info:
                if link_key not in self.die_link_composition[die_id]:
                    self.die_link_composition[die_id][link_key] = []
                self.die_link_composition[die_id][link_key].append(flow_info)

            node = next_node

    def _add_path_bandwidth_yx(
        self, die_id: int, source: Tuple[int, int], target: Tuple[int, int], bandwidth: float, flow_info: Dict = None
    ):
        """使用YX路由添加路径带宽到指定Die"""
        if die_id not in self.die_link_bandwidth:
            self._static_initial(die_id)
        if die_id not in self.die_link_composition:
            self.die_link_composition[die_id] = {}

        node = source
        while True:
            dx = target[0] - node[0]
            dy = target[1] - node[1]

            if dy > 0:
                next_node = (node[0], node[1] + 1)
            elif dy < 0:
                next_node = (node[0], node[1] - 1)
            elif dx > 0:
                next_node = (node[0] + 1, node[1])
            elif dx < 0:
                next_node = (node[0] - 1, node[1])
            else:
                return

            link_key = (node, next_node)
            if link_key not in self.die_link_bandwidth[die_id]:
                self.die_link_bandwidth[die_id][link_key] = 0
            self.die_link_bandwidth[die_id][link_key] += bandwidth

            # 记录链路组成
            if flow_info:
                if link_key not in self.die_link_composition[die_id]:
                    self.die_link_composition[die_id][link_key] = []
                self.die_link_composition[die_id][link_key].append(flow_info)

            node = next_node

    def compute(self, routing_type: str = "XY") -> Dict[int, Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]]:
        """
        计算D2D静态链路带宽

        Args:
            routing_type: 路由算法类型，"XY" 或 "YX"

        Returns:
            每个Die的链路带宽字典: {die_id: {((src_x, src_y), (dst_x, dst_y)): bandwidth_GB/s}}
        """
        # 初始化所有Die的链路
        for die_id in range(self.num_dies):
            self._static_initial(die_id)

        # 选择路由函数
        add_path_fn = self._add_path_bandwidth_xy if routing_type == "XY" else self._add_path_bandwidth_yx

        # 遍历每个配置
        for config in self.configs:
            # 提取源节点列表
            src_nodes = []
            src_ip_types = []
            for ip_type, nodes in config.src_map.items():
                for node in nodes:
                    src_nodes.append(node)
                    src_ip_types.append(ip_type)

            # 提取目标节点列表
            dst_nodes = []
            dst_ip_types = []
            for ip_type, nodes in config.dst_map.items():
                for node in nodes:
                    dst_nodes.append(node)
                    dst_ip_types.append(ip_type)

            if not src_nodes or not dst_nodes:
                continue

            # 获取die_pairs
            die_pairs = getattr(config, 'die_pairs', None)
            if not die_pairs:
                # 从 src_die/dst_die 构建 die_pairs
                src_die = getattr(config, 'src_die', 0)
                dst_die = getattr(config, 'dst_die', 0)
                die_pairs = [[src_die, dst_die]]

            req_type = getattr(config, 'req_type', 'W')
            bandwidth_per_src = config.speed

            # 计算总的目标数：考虑所有die_pairs中的目标Die数量
            num_target_dies = len(set([pair[1] for pair in die_pairs]))  # 去重统计目标Die数
            total_targets = len(dst_nodes) * num_target_dies
            bandwidth_per_pair = bandwidth_per_src / total_targets if total_targets > 0 else 0

            # 为每个源-目标对和每个Die对计算带宽
            for src_idx, src_node in enumerate(src_nodes):
                src_ip_type = src_ip_types[src_idx]
                for dst_idx, dst_node in enumerate(dst_nodes):
                    dst_ip_type = dst_ip_types[dst_idx]
                    dst_ip_id = self._extract_ip_id(dst_ip_type)

                    src_pos = self._node_id_to_pos(src_node)
                    dst_pos = self._node_id_to_pos(dst_node)

                    for die_pair in die_pairs:
                        src_die, dst_die = die_pair[0], die_pair[1]

                        # 创建流量信息
                        flow_info = {
                            'src_node': src_node,
                            'src_ip': src_ip_type,
                            'dst_node': dst_node,
                            'dst_ip': dst_ip_type,
                            'bandwidth': bandwidth_per_pair,
                            'req_type': req_type,
                            'src_die': src_die,
                            'dst_die': dst_die
                        }

                        if src_die == dst_die:
                            # Die内流量
                            if req_type == 'R':
                                # 读请求：数据从目标流向源
                                add_path_fn(src_die, dst_pos, src_pos, bandwidth_per_pair, flow_info)
                            else:
                                # 写请求：数据从源流向目标
                                add_path_fn(src_die, src_pos, dst_pos, bandwidth_per_pair, flow_info)
                        else:
                            # 跨Die流量
                            d2d_sn_node, d2d_rn_node = self._select_d2d_nodes(src_die, dst_die, dst_ip_id)
                            d2d_sn_pos = self._node_id_to_pos(d2d_sn_node)
                            d2d_rn_pos = self._node_id_to_pos(d2d_rn_node)

                            if req_type == 'R':
                                # 读请求：数据从目标Die返回源Die
                                # 目标Die: 目标IP → D2D_RN
                                add_path_fn(dst_die, dst_pos, d2d_rn_pos, bandwidth_per_pair, flow_info)
                                # 源Die: D2D_SN → 源IP
                                add_path_fn(src_die, d2d_sn_pos, src_pos, bandwidth_per_pair, flow_info)

                                # 累加D2D链路带宽（目标Die → 源Die）
                                d2d_key = (dst_die, d2d_rn_node, src_die, d2d_sn_node)
                                self.d2d_link_bandwidth[d2d_key] = self.d2d_link_bandwidth.get(d2d_key, 0.0) + bandwidth_per_pair
                                # 记录D2D链路组成
                                if d2d_key not in self.d2d_link_composition:
                                    self.d2d_link_composition[d2d_key] = []
                                self.d2d_link_composition[d2d_key].append(flow_info)
                            else:
                                # 写请求：数据从源Die发送到目标Die
                                # 源Die: 源IP → D2D_SN
                                add_path_fn(src_die, src_pos, d2d_sn_pos, bandwidth_per_pair, flow_info)
                                # 目标Die: D2D_RN → 目标IP
                                add_path_fn(dst_die, d2d_rn_pos, dst_pos, bandwidth_per_pair, flow_info)

                                # 累加D2D链路带宽（源Die → 目标Die）
                                d2d_key = (src_die, d2d_sn_node, dst_die, d2d_rn_node)
                                self.d2d_link_bandwidth[d2d_key] = self.d2d_link_bandwidth.get(d2d_key, 0.0) + bandwidth_per_pair
                                # 记录D2D链路组成
                                if d2d_key not in self.d2d_link_composition:
                                    self.d2d_link_composition[d2d_key] = []
                                self.d2d_link_composition[d2d_key].append(flow_info)

        return self.die_link_bandwidth

    def get_link_composition(self) -> Dict[str, List[Dict]]:
        """
        获取链路带宽组成信息（包括Die内部和跨Die链路）

        Returns:
            字典格式: {link_key: [flow_info, ...]}
            - Die内部链路: key="x1,y1-x2,y2"
            - D2D跨Die链路: key="src_die-src_node-dst_die-dst_node"
        """
        result = {}

        # Die内部链路
        for die_id, link_comp in self.die_link_composition.items():
            for link_key, flows in link_comp.items():
                # 转换为字符串格式，包含die_id以区分不同Die的相同位置链路
                src_pos, dst_pos = link_key
                key = f"{die_id}-{src_pos[0]},{src_pos[1]}-{dst_pos[0]},{dst_pos[1]}"
                result[key] = flows

        # D2D跨Die链路
        for d2d_key, flows in self.d2d_link_composition.items():
            src_die, src_node, dst_die, dst_node = d2d_key
            key = f"{src_die}-{src_node}-{dst_die}-{dst_node}"
            result[key] = flows

        return result

    def get_statistics(self) -> Dict[int, Dict[str, float]]:
        """获取每个Die的链路带宽统计信息"""
        statistics = {}
        for die_id, link_bw in self.die_link_bandwidth.items():
            active_bandwidths = [bw for bw in link_bw.values() if bw > 0]
            if active_bandwidths:
                statistics[die_id] = {
                    "max_bandwidth": max(active_bandwidths),
                    "sum_bandwidth": sum(active_bandwidths),
                    "avg_bandwidth": sum(active_bandwidths) / len(active_bandwidths),
                    "num_active_links": len(active_bandwidths),
                }
            else:
                statistics[die_id] = {
                    "max_bandwidth": 0.0,
                    "sum_bandwidth": 0.0,
                    "avg_bandwidth": 0.0,
                    "num_active_links": 0,
                }
        return statistics


def compute_d2d_link_bandwidth(
    topo_type: str,
    configs: List,
    d2d_pairs: List[Tuple[int, int, int, int]],
    routing_type: str = "XY",
    num_dies: int = 2,
) -> Tuple[Dict[int, Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]], Dict[Tuple[int, int, int, int], float], Dict[str, List[Dict]]]:
    """
    便捷函数: 计算D2D静态链路带宽

    Args:
        topo_type: 拓扑类型，如 "5x4"
        configs: TrafficConfig列表（包含die_pairs和src_map/dst_map）
        d2d_pairs: D2D连接配对列表
        routing_type: 路由算法，"XY" 或 "YX"
        num_dies: Die数量

    Returns:
        (die_link_bandwidth, d2d_link_bandwidth, link_composition)
        - die_link_bandwidth: {die_id: {((src_x, src_y), (dst_x, dst_y)): bandwidth_GB/s}}
        - d2d_link_bandwidth: {(src_die, src_node, dst_die, dst_node): bandwidth_GB/s}
        - link_composition: {link_key: [flow_info, ...]}
    """
    analyzer = D2DStaticBandwidthAnalyzer(topo_type, configs, d2d_pairs, num_dies)
    analyzer.compute(routing_type)
    return analyzer.die_link_bandwidth, analyzer.d2d_link_bandwidth, analyzer.get_link_composition()
