"""D2D (Die-to-Die) 配置管理器

本模块提供D2DConfig类，用于管理D2D特定的配置参数和自动节点位置计算。
"""

import json
import yaml
import logging
import os
from typing import List, Tuple, Optional, Dict, Any


class D2DConfig:
    """D2D配置管理器，独立管理D2D系统配置

    主要功能：
    1. 管理D2D布局配置和每个Die的拓扑信息
    2. 自动加载每个Die对应的拓扑配置文件
    3. 管理RN-SN配对关系
    4. 验证D2D配置的合理性
    """

    def __init__(self, d2d_config_file: str) -> None:
        """初始化D2D配置

        Args:
            d2d_config_file: D2D专用配置文件路径

        Raises:
            ValueError: 如果未指定d2d_config_file
        """
        self._init_d2d_config()

        if d2d_config_file:
            self._load_d2d_config_file(d2d_config_file)
        else:
            raise ValueError("必须指定d2d_config_file")

        self._generate_d2d_pairs()
        self._update_channel_spec_for_d2d()
        self._validate_d2d_layout()

    def _init_d2d_config(self):
        """初始化D2D基础配置"""
        self.D2D_DIE_POSITIONS: Dict[int, List[int]] = {}
        self.D2D_RN_POSITIONS: Dict[int, List[int]] = {}
        self.D2D_SN_POSITIONS: Dict[int, List[int]] = {}
        self.D2D_PAIRS: List[Tuple[int, int, int, int]] = []

        self.D2D_CONNECTION_MAP: Dict[str, Any] = {}
        self.D2D_MULTI_HOP_ENABLED: bool = False
        self.D2D_ROUTING_ALGORITHM: str = "shortest_path"

        self.die_layout_positions: Dict[int, Tuple[int, int]] = {}
        self.die_layout_type: str = ""

        self.CHANNEL_SPEC: Dict[str, int] = {}
        self.CH_NAME_LIST: List[str] = []
        self.DIE_TOPOLOGIES: Dict[int, str] = {}

        self.NETWORK_FREQUENCY: Optional[int] = None
        self.FLIT_SIZE: Optional[int] = None
        self.BURST: Optional[int] = None

    def _generate_d2d_pairs(self) -> None:
        """生成D2D配对关系

        Raises:
            ValueError: 如果Die数量少于2个
        """
        num_dies = getattr(self, "NUM_DIES", 2)
        if num_dies < 2:
            raise ValueError(f"D2D需要至少2个Die，当前配置: {num_dies}")

        self._setup_new_format()

    def _setup_new_format(self) -> None:
        """新格式：基于DIE_POSITIONS和D2D_CONNECTIONS的简化配置

        Raises:
            ValueError: 如枟缺少必需的配置项
        """

        if not hasattr(self, "DIE_POSITIONS"):
            raise ValueError("新格式配置缺少DIE_POSITIONS")
        if not hasattr(self, "D2D_CONNECTIONS"):
            raise ValueError("新格式配置缺少D2D_CONNECTIONS")

        self.die_layout_positions = getattr(self, "DIE_POSITIONS", {})
        die_topologies = getattr(self, "DIE_TOPOLOGIES", {})
        self.DIE_TOPOLOGIES = die_topologies

        connections = getattr(self, "D2D_CONNECTIONS", [])
        pairs = self._generate_pairs_from_connections(connections)

        self.D2D_PAIRS = pairs
        self._setup_die_positions_from_pairs(pairs)
        self._calculate_die_layout_type()

    def _generate_pairs_from_connections(self, connections: List[List[int]]) -> List[Tuple[int, int, int, int]]:
        """从D2D_CONNECTIONS生成配对关系

        Args:
            connections: D2D连接配置列表，每个元素为[src_die, src_node, dst_die, dst_node]

        Returns:
            配对关系列表，每个元素为(die0_id, node0, die1_id, node1)

        Raises:
            ValueError: 如果连接配置格式错误或Die/节点ID超出范围
        """
        pairs = []

        for conn in connections:
            if len(conn) != 4:
                raise ValueError(f"连接配置格式错误，应为[src_die, src_node, dst_die, dst_node]: {conn}")

            src_die, src_node, dst_die, dst_node = conn

            num_dies = getattr(self, "NUM_DIES", 2)
            if src_die >= num_dies or dst_die >= num_dies:
                raise ValueError(f"Die ID超出范围: src_die={src_die}, dst_die={dst_die}, NUM_DIES={num_dies}")

            src_topology = self.DIE_TOPOLOGIES.get(src_die, "5x4")
            dst_topology = self.DIE_TOPOLOGIES.get(dst_die, "5x4")

            src_num_row, src_num_col = self._parse_topology(src_topology)
            dst_num_row, dst_num_col = self._parse_topology(dst_topology)

            max_src_node = src_num_row * src_num_col - 1
            max_dst_node = dst_num_row * dst_num_col - 1

            if src_node > max_src_node:
                raise ValueError(f"源节点ID超出范围: {src_node} > {max_src_node} (Die{src_die}, 拓扑{src_topology})")
            if dst_node > max_dst_node:
                raise ValueError(f"目标节点ID超出范围: {dst_node} > {max_dst_node} (Die{dst_die}, 拓扑{dst_topology})")

            src_pos_rn = src_node % src_num_col + src_num_col + src_node // src_num_col * 2 * src_num_col
            src_pos_sn = src_node % src_num_col + src_node // src_num_col * 2 * src_num_col

            dst_pos_rn = dst_node % dst_num_col + dst_num_col + dst_node // dst_num_col * 2 * dst_num_col
            dst_pos_sn = dst_node % dst_num_col + dst_node // dst_num_col * 2 * dst_num_col

            pairs.append((src_die, src_pos_rn, dst_die, dst_pos_sn))
            pairs.append((dst_die, dst_pos_rn, src_die, src_pos_sn))

        return pairs

    def _setup_die_positions_from_pairs(self, pairs: List[Tuple[int, int, int, int]]) -> None:
        """从配对关系中设置各Die的D2D节点位置

        Args:
            pairs: D2D配对关系列表
        """
        num_dies = getattr(self, "NUM_DIES", 2)

        for die_id in range(num_dies):
            all_positions = []
            rn_positions = []
            sn_positions = []

            for pair in pairs:
                if pair[0] == die_id:
                    all_positions.append(pair[1])
                    node_pos = pair[1]
                    topology_str = self.DIE_TOPOLOGIES.get(die_id, "5x4")
                    _, num_col = self._parse_topology(topology_str)
                    network_row = node_pos // num_col
                    if network_row % 2 == 0:
                        sn_positions.append(node_pos)
                    else:
                        rn_positions.append(node_pos)

                if pair[2] == die_id:
                    all_positions.append(pair[3])
                    node_pos = pair[3]
                    topology_str = self.DIE_TOPOLOGIES.get(die_id, "5x4")
                    _, num_col = self._parse_topology(topology_str)
                    network_row = node_pos // num_col
                    if network_row % 2 == 0:
                        sn_positions.append(node_pos)
                    else:
                        rn_positions.append(node_pos)

            self.D2D_DIE_POSITIONS[die_id] = list(set(all_positions))
            self.D2D_RN_POSITIONS[die_id] = list(set(rn_positions))
            self.D2D_SN_POSITIONS[die_id] = list(set(sn_positions))

    def _parse_topology(self, topology_str: str) -> Tuple[int, int]:
        """解析拓扑字符串获取行列信息

        Args:
            topology_str: 拓扑字符串，如 "5x4", "4x5" 等

        Returns:
            (num_row, num_col) 逻辑行列数

        Raises:
            ValueError: 如果拓扑格式错误
        """
        try:
            parts = topology_str.split("x")
            if len(parts) != 2:
                raise ValueError(f"拓扑格式错误，应为 'AxB' 格式: {topology_str}")

            num_row = int(parts[0])
            num_col = int(parts[1])

            return num_row, num_col
        except Exception as e:
            raise ValueError(f"解析拓扑 '{topology_str}' 失败: {e}")

    def _get_edge_nodes(self, edge: str, num_row: int, num_col: int, interface_type: Optional[str] = None) -> List[int]:
        """获取指定边的所有节点

        Args:
            edge: 边界方向 ("top", "bottom", "left", "right")
            num_row: 逻辑行数
            num_col: 列数
            interface_type: 接口类型 ("rn" 或 "sn")，用于区分奇偶行

        Returns:
            边界节点列表
        """
        if edge == "top":
            if interface_type == "rn":
                return list(range(num_col, 2 * num_col))
            elif interface_type == "sn":
                return list(range(0, num_col))
            else:
                return list(range(0, num_col))
        elif edge == "bottom":
            if interface_type == "rn":
                return list(range((num_row - 1) * num_col, num_row * num_col))
            elif interface_type == "sn":
                return list(range((num_row - 2) * num_col, (num_row - 1) * num_col))
            else:
                return list(range((num_row - 1) * num_col, num_row * num_col))
        elif edge == "left":
            if interface_type == "rn":
                return [row * num_col for row in range(1, num_row, 2)]
            elif interface_type == "sn":
                return [row * num_col for row in range(0, num_row, 2)]
            else:
                return [row * num_col for row in range(num_row)]
        elif edge == "right":
            if interface_type == "rn":
                return [row * num_col + (num_col - 1) for row in range(1, num_row, 2)]
            elif interface_type == "sn":
                return [row * num_col + (num_col - 1) for row in range(0, num_row, 2)]
            else:
                return [row * num_col + (num_col - 1) for row in range(num_row)]
        else:
            return []

    def _get_opposite_edge(self, edge: str) -> str:
        """获取对边

        Args:
            edge: 边界方向

        Returns:
            对边方向
        """
        opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
        return opposite.get(edge, "")

    def _generate_d2d_pairs_from_config(self, config: Dict[int, Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
        """根据新配置格式生成D2D节点配对

        Args:
            config: Die配置字典

        Returns:
            配对关系列表 [(die0_id, node0, die1_id, node1), ...]
        """
        self._validate_d2d_config(config)

        pairs = []
        processed = set()

        if not hasattr(self, "DIE_TOPOLOGIES"):
            self.DIE_TOPOLOGIES = {}

        for die_id, die_config in config.items():
            topology_str = die_config["topology"]
            num_row, num_col = self._parse_topology(topology_str)
            self.DIE_TOPOLOGIES[die_id] = topology_str

            for edge, conn_info in die_config["connections"].items():
                d2d_nodes = conn_info["d2d_nodes"]
                node_mapping = conn_info["node_to_die_mapping"]

                edge_rn_nodes = self._get_edge_nodes(edge, num_row, num_col, "rn")
                edge_sn_nodes = self._get_edge_nodes(edge, num_row, num_col, "sn")

                die_groups = {}
                for node_idx in d2d_nodes:
                    target_die = node_mapping[node_idx]
                    if target_die not in die_groups:
                        die_groups[target_die] = []
                    die_groups[target_die].append(node_idx)

                for target_die, node_indices in die_groups.items():
                    pair_key = tuple(sorted([die_id, target_die]))
                    if pair_key in processed:
                        continue
                    processed.add(pair_key)

                    src_rn_nodes = [edge_rn_nodes[i] for i in node_indices]
                    src_sn_nodes = [edge_sn_nodes[i] for i in node_indices]

                    opposite_edge = self._get_opposite_edge(edge)
                    target_config = config[target_die]
                    target_conn = target_config["connections"][opposite_edge]

                    target_topology_str = target_config["topology"]
                    target_num_row, target_num_col = self._parse_topology(target_topology_str)

                    target_edge_rn_nodes = self._get_edge_nodes(opposite_edge, target_num_row, target_num_col, "rn")
                    target_edge_sn_nodes = self._get_edge_nodes(opposite_edge, target_num_row, target_num_col, "sn")

                    target_indices = []
                    for idx, t_die in target_conn["node_to_die_mapping"].items():
                        if t_die == die_id:
                            target_indices.append(idx)

                    target_rn_nodes = [target_edge_rn_nodes[i] for i in target_indices]
                    target_sn_nodes = [target_edge_sn_nodes[i] for i in target_indices]

                    for i in range(min(len(src_rn_nodes), len(target_sn_nodes))):
                        pairs.append((die_id, src_rn_nodes[i], target_die, target_sn_nodes[i]))

                    for i in range(min(len(src_sn_nodes), len(target_rn_nodes))):
                        pairs.append((die_id, src_sn_nodes[i], target_die, target_rn_nodes[i]))

        return pairs

    def _update_channel_spec_for_d2d(self) -> None:
        """更新 CHANNEL_SPEC 以包含 D2D 接口类型"""
        if hasattr(self, "D2D_ENABLED") and self.D2D_ENABLED:
            if "d2d_rn" not in self.CHANNEL_SPEC:
                self.CHANNEL_SPEC["d2d_rn"] = 1
            if "d2d_sn" not in self.CHANNEL_SPEC:
                self.CHANNEL_SPEC["d2d_sn"] = 1

            self.CH_NAME_LIST = []
            for key in self.CHANNEL_SPEC:
                for idx in range(self.CHANNEL_SPEC[key]):
                    self.CH_NAME_LIST.append(f"{key}_{idx}")

    def _load_d2d_config_file(self, d2d_config_file: str) -> None:
        """加载D2D专用配置文件

        Args:
            d2d_config_file: D2D配置文件路径

        Raises:
            FileNotFoundError: 如果配置文件不存在
            ValueError: 如果配置文件格式错误
        """
        try:
            with open(d2d_config_file, "r", encoding="utf-8") as f:
                if str(d2d_config_file).endswith((".yaml", ".yml")):
                    d2d_config = yaml.safe_load(f)
                else:
                    d2d_config = json.load(f)

            for key, value in d2d_config.items():
                if key.startswith("D2D_") or key in ["NUM_DIES", "D2D_DIE_CONFIG", "DIE_POSITIONS", "DIE_TOPOLOGIES", "NETWORK_FREQUENCY", "FLIT_SIZE", "BURST"]:
                    setattr(self, key, value)

            print(f"成功加载D2D配置文件: {d2d_config_file}")
            if hasattr(self, "D2D_DIE_CONFIG"):
                print(f"D2D_DIE_CONFIG包含 {len(self.D2D_DIE_CONFIG)} 个Die配置")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"D2D配置文件不存在: {d2d_config_file}")
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"D2D配置文件格式错误: {e}")

    def _validate_d2d_config(self, die_config: Dict[int, Dict[str, Any]]) -> bool:
        """验证D2D配置的正确性

        Args:
            die_config: Die配置字典

        Returns:
            验证结果

        Raises:
            ValueError: 如果配置验证失败
        """
        errors = []
        num_dies = getattr(self, "NUM_DIES", 2)

        for die_id in range(num_dies):
            if die_id not in die_config:
                errors.append(f"缺少Die {die_id}的配置")

        for die_id, config in die_config.items():
            if "connections" not in config:
                errors.append(f"Die {die_id}缺少connections配置")
                continue

            for edge, conn_info in config.get("connections", {}).items():
                if "connect_die" not in conn_info:
                    errors.append(f"Die {die_id} {edge}边缺少connect_die")
                if "d2d_nodes" not in conn_info:
                    errors.append(f"Die {die_id} {edge}边缺少d2d_nodes")
                if "node_to_die_mapping" not in conn_info:
                    errors.append(f"Die {die_id} {edge}边缺少node_to_die_mapping")
                    continue

                connect_die = conn_info["connect_die"]
                d2d_nodes = conn_info["d2d_nodes"]
                mapping = conn_info["node_to_die_mapping"]

                if connect_die >= num_dies:
                    errors.append(f"Die {die_id} {edge}边connect_die {connect_die}不存在")

                for node_idx in d2d_nodes:
                    if node_idx not in mapping:
                        errors.append(f"Die {die_id} {edge}边节点{node_idx}缺少映射")

                for node_idx, target_die in mapping.items():
                    if target_die >= num_dies:
                        errors.append(f"Die {die_id}节点{node_idx}映射到不存在的Die {target_die}")

                mapped_dies = set(mapping.values())
                if connect_die not in mapped_dies:
                    errors.append(f"Die {die_id} {edge}边connect_die {connect_die}不在node_to_die_mapping中")

                if "topology" in config:
                    num_row, num_col = self._parse_topology(config["topology"])
                    edge_nodes = self._get_edge_nodes(edge, num_row, num_col)
                    for node_idx in d2d_nodes:
                        if node_idx >= len(edge_nodes):
                            errors.append(f"Die {die_id} {edge}边节点索引{node_idx}超出范围")
                else:
                    errors.append(f"Die {die_id}缺少topology配置")

        errors.extend(self._validate_bidirectional_connections(die_config))

        if errors:
            raise ValueError("D2D配置验证失败:\n" + "\n".join(errors))

        return True

    def _validate_bidirectional_connections(self, die_config: Dict[int, Dict[str, Any]]) -> List[str]:
        """验证双向连接的一致性

        Args:
            die_config: Die配置字典

        Returns:
            错误信息列表
        """
        errors = []
        checked_pairs = set()

        for die_id, config in die_config.items():
            for edge, conn_info in config.get("connections", {}).items():
                connect_die = conn_info.get("connect_die")
                if connect_die is None:
                    continue

                pair_key = tuple(sorted([die_id, connect_die]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                # 查找对端配置
                opposite_edge = self._get_opposite_edge(edge)
                target_config = die_config.get(connect_die, {})
                target_conn = target_config.get("connections", {}).get(opposite_edge)

                if not target_conn:
                    errors.append(f"Die {connect_die}缺少对应的{opposite_edge}边连接到Die {die_id}")
                    continue

                # 验证对端的connect_die指向当前Die
                target_connect_die = target_conn.get("connect_die")
                if target_connect_die != die_id:
                    errors.append(f"Die {connect_die} {opposite_edge}边connect_die是{target_connect_die}，应该是{die_id}")

        return errors

    def _validate_d2d_layout(self):
        """验证D2D布局的合理性"""
        # 检查基本参数
        if not getattr(self, "D2D_ENABLED", False):
            return

        num_dies = getattr(self, "NUM_DIES", 2)
        if num_dies < 2:
            raise ValueError(f"D2D需要至少2个Die，当前配置: {num_dies}")

        if num_dies > 4:
            raise ValueError(f"当前最多支持4个Die，当前配置: {num_dies}")

        # 布局验证已不需要，所有布局信息都在D2D_DIE_CONFIG中定义

        # 检查节点位置有效性（映射后的网络坐标）
        for die_id in range(num_dies):
            # 获取该Die的拓扑信息
            topology_str = self.DIE_TOPOLOGIES.get(die_id)
            if topology_str:
                num_row, num_col = self._parse_topology(topology_str)
                # 网络坐标的最大值是: (num_row * 2 * num_col) - 1
                max_network_pos = (num_row * 2 * num_col) - 1

                positions = self.D2D_DIE_POSITIONS.get(die_id, [])
                for pos in positions:
                    if pos < 0 or pos > max_network_pos:
                        raise ValueError(f"Die{die_id}网络节点位置无效: {pos}, 有效范围: 0-{max_network_pos} (拓扑: {topology_str})")
            else:
                print(f"警告: Die{die_id}未指定拓扑，跳过节点位置验证")

        # 检查配对数量
        if len(self.D2D_PAIRS) == 0:
            raise ValueError("没有有效的D2D连接对")

        # 4-Die特定验证
        if num_dies == 4:
            self._validate_4die_config()

    def _validate_4die_config(self):
        """验证4-Die配置的特定要求"""
        # 验证每个Die都有D2D节点位置
        for die_id in range(4):
            if die_id not in self.D2D_DIE_POSITIONS or not self.D2D_DIE_POSITIONS[die_id]:
                raise ValueError(f"Die{die_id}没有配置D2D节点位置")

        # 验证配对的完整性：每个Die应该与其他Die有连接
        die_connections = {i: set() for i in range(4)}
        for pair in self.D2D_PAIRS:
            die_connections[pair[0]].add(pair[2])
            die_connections[pair[2]].add(pair[0])

        # 不限制每个Die连接的数量，只要双向连接一致即可

    def get_die_boundary_nodes(self, die_id: int) -> Dict[str, List[int]]:
        """获取指定Die的边界节点

        Args:
            die_id: Die ID

        Returns:
            边界节点字典 {"top": [...], "bottom": [...], "left": [...], "right": [...]}
        """
        boundary = {"top": [], "bottom": [], "left": [], "right": []}

        for row in range(self.NUM_ROW):
            for col in range(self.NUM_COL):
                node_id = row * self.NUM_COL + col

                if row == 0:
                    boundary["top"].append(node_id)
                if row == self.NUM_ROW - 1:
                    boundary["bottom"].append(node_id)
                if col == 0:
                    boundary["left"].append(node_id)
                if col == self.NUM_COL - 1:
                    boundary["right"].append(node_id)

        return boundary

    def print_d2d_layout(self) -> None:
        """可视化打印D2D布局信息"""
        print("\n=== D2D布局配置 ===")
        num_dies = getattr(self, "NUM_DIES", 2)
        print(f"Die数量: {num_dies}")

        for die_id in range(num_dies):
            positions = self.D2D_DIE_POSITIONS.get(die_id, [])
            print(f"Die{die_id} D2D节点位置: {positions}")

        print(f"D2D连接对数量: {len(self.D2D_PAIRS)}")
        print("D2D连接对详情:")
        for i, pair in enumerate(self.D2D_PAIRS):
            print(f"  {i+1:2d}. Die{pair[0]}:节点{pair[1]} ↔ Die{pair[2]}:节点{pair[3]}")

        print("=" * 30)

    def generate_d2d_traffic_example(self, num_requests: int = 10) -> List[str]:
        """生成D2D测试流量示例

        Args:
            num_requests: 生成的请求数量

        Returns:
            流量字符串列表
        """
        import random

        traffic_lines = []
        traffic_lines.append("# D2D Traffic Generated by D2DConfig")
        traffic_lines.append("# Format: inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length")

        # 生成随机流量数据
        for i in range(num_requests):
            if i % 2 == 0:
                src_die, dst_die = 0, 1
            else:
                src_die, dst_die = 1, 0

            # 使用基本节点数量（需要在子类中定义）
            max_nodes = 20  # 默认值，实际使用时应从拓扑信息获取
            src_node = random.choice(range(max_nodes))
            dst_node = random.choice(range(max_nodes))

            inject_time = i * 100 + 100
            src_ip = f"gdma_{random.randint(0, 1)}"
            dst_ip = f"ddr_{random.randint(0, 1)}"
            req_type = random.choice(["R", "W"])
            burst_length = random.choice([4, 8, 16])

            line = f"{inject_time}, {src_die}, {src_node}, {src_ip}, {dst_die}, {dst_node}, {dst_ip}, {req_type}, {burst_length}"
            traffic_lines.append(line)

        return traffic_lines

    def save_d2d_config(self, config_file: str) -> None:
        """保存D2D配置到文件

        Args:
            config_file: 配置文件路径
        """
        d2d_config = {
            "D2D_ENABLED": getattr(self, "D2D_ENABLED", True),
            "NUM_DIES": getattr(self, "NUM_DIES", 2),
            "D2D_DIE_POSITIONS": self.D2D_DIE_POSITIONS,
            "D2D_PAIRS": self.D2D_PAIRS,
            "D2D_AR_LATENCY": getattr(self, "D2D_AR_LATENCY", 10),
            "D2D_R_LATENCY": getattr(self, "D2D_R_LATENCY", 10),
            "D2D_AW_LATENCY": getattr(self, "D2D_AW_LATENCY", 10),
            "D2D_W_LATENCY": getattr(self, "D2D_W_LATENCY", 10),
            "D2D_B_LATENCY": getattr(self, "D2D_B_LATENCY", 10),
            "D2D_DBID_LATENCY": getattr(self, "D2D_DBID_LATENCY", 10),
            "D2D_MAX_OUTSTANDING": getattr(self, "D2D_MAX_OUTSTANDING", 16),
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(d2d_config, f, indent=4, ensure_ascii=False)

    def _calculate_die_layout_type(self):
        """计算Die布局类型"""
        positions = self.die_layout_positions
        if positions:
            max_x = max(pos[0] for pos in positions.values())
            max_y = max(pos[1] for pos in positions.values())
            num_cols = max_x + 1
            num_rows = max_y + 1
            self.die_layout_type = f"{num_rows}x{num_cols}"
            print(f"Die布局类型: {self.die_layout_type}")
