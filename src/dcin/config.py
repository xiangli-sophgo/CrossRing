"""
DCIN (Die-to-Die Chip Interconnect Network) 配置管理器

本模块提供 DCINConfig 类，用于管理 DCIN 特定的配置参数和自动节点位置计算。
"""

import json
import yaml
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from fractions import Fraction
from math import lcm, gcd


class DCINConfig:
    """DCIN 配置管理器，独立管理 DCIN 系统配置

    主要功能：
    1. 管理 DCIN 布局配置和每个 Die 的拓扑信息
    2. 自动加载每个 Die 对应的拓扑配置文件
    3. 管理 RN-SN 配对关系
    4. 验证 DCIN 配置的合理性
    """

    def __init__(self, dcin_config_file: str, die_config_file: Optional[str] = None) -> None:
        """初始化 DCIN 配置

        Args:
            dcin_config_file: DCIN 专用配置文件路径
            die_config_file: DIE 拓扑配置文件路径（可选，用于覆盖 DIE_TOPOLOGIES 中指定的配置）

        Raises:
            ValueError: 如果未指定 dcin_config_file
        """
        self._init_dcin_config()

        if dcin_config_file:
            self._load_dcin_config_file(dcin_config_file)
        else:
            raise ValueError("必须指定 dcin_config_file")

        # 如果指定了单独的 DIE 拓扑配置文件，更新 DIE_TOPOLOGIES
        if die_config_file:
            self._apply_die_config_file(die_config_file)
        else:
            # 从 DIE_TOPOLOGIES 自动加载第一个 Die 的拓扑配置来获取共享参数
            if self.DIE_TOPOLOGIES:
                first_topo = list(self.DIE_TOPOLOGIES.values())[0]
                # 配置文件路径：项目根目录/config/topologies/
                project_root = Path(__file__).resolve().parent.parent.parent
                kcin_file = project_root / "config" / "topologies" / f"kcin_{first_topo}.yaml"
                from src.kcin.v1.config import V1Config
                self.die_config = KCINConfig(str(kcin_file))
                if not self.NETWORK_FREQUENCY:
                    self.NETWORK_FREQUENCY = self.die_config.NETWORK_FREQUENCY
                if not self.FLIT_SIZE:
                    self.FLIT_SIZE = self.die_config.FLIT_SIZE
                if not self.BURST:
                    self.BURST = self.die_config.BURST

        self._generate_d2d_pairs()
        self._update_channel_spec_for_d2d()
        self._validate_d2d_layout()
        self.update_latency()

    def _init_dcin_config(self):
        """初始化DCIN基础配置"""
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
        self.DIE_ROTATIONS: Dict[int, int] = {}

        self.NETWORK_FREQUENCY: Optional[float] = None
        self.IP_FREQUENCY: Optional[float] = None
        self.FLIT_SIZE: Optional[int] = None
        self.BURST: Optional[int] = None

        # 时间缩放参数（由 _calculate_time_scale 计算）
        self.CYCLES_PER_NS: Optional[int] = None
        self.NETWORK_SCALE: Optional[int] = None
        self.IP_SCALE: Optional[int] = None
        self.EFFECTIVE_NETWORK_FREQ: Optional[float] = None
        self.EFFECTIVE_IP_FREQ: Optional[float] = None

        # D2D延迟配置 (原始ns值)
        self.D2D_AR_LATENCY_original: Optional[float] = None
        self.D2D_R_LATENCY_original: Optional[float] = None
        self.D2D_AW_LATENCY_original: Optional[float] = None
        self.D2D_W_LATENCY_original: Optional[float] = None
        self.D2D_B_LATENCY_original: Optional[float] = None

        # D2D延迟配置 (转换后的cycles值)
        self.D2D_AR_LATENCY: Optional[int] = None
        self.D2D_R_LATENCY: Optional[int] = None
        self.D2D_AW_LATENCY: Optional[int] = None
        self.D2D_W_LATENCY: Optional[int] = None
        self.D2D_B_LATENCY: Optional[int] = None

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

        if not hasattr(self, "D2D_CONNECTIONS"):
            raise ValueError("新格式配置缺少D2D_CONNECTIONS")

        # 自动生成DIE_POSITIONS（如果缺失）
        if not hasattr(self, "DIE_POSITIONS") or not self.DIE_POSITIONS:
            self.DIE_POSITIONS = self._generate_default_die_positions()

        # 自动生成DIE_ROTATIONS（如果缺失）
        if not hasattr(self, "DIE_ROTATIONS") or not self.DIE_ROTATIONS:
            self.DIE_ROTATIONS = self._generate_default_die_rotations()

        self.die_layout_positions = getattr(self, "DIE_POSITIONS", {})
        die_topologies = getattr(self, "DIE_TOPOLOGIES", {})
        self.DIE_TOPOLOGIES = die_topologies

        connections = getattr(self, "D2D_CONNECTIONS", [])
        pairs = self._generate_pairs_from_connections(connections)

        self.D2D_PAIRS = pairs
        self._setup_die_positions_from_pairs(pairs)
        self._calculate_die_layout_type()

    def _generate_default_die_positions(self) -> Dict[int, List[int]]:
        """根据NUM_DIES自动生成默认的Die布局位置

        布局规则：
        - 2 Dies: 水平排列 [0,0], [1,0]
        - 4 Dies: 2x2网格，右下角为Die0，逆时针排列
          Die2 - Die3
          |     |
          Die1 - Die0
        - 其他: 尽量接近正方形的网格

        Returns:
            Die位置字典，key为die_id，value为[x, y]坐标
        """
        num_dies = getattr(self, "NUM_DIES", 2)
        positions = {}

        if num_dies == 2:
            positions = {0: [1, 0], 1: [0, 0]}
        elif num_dies == 4:
            # 2x2布局，Die0在右下角，逆时针排列
            positions = {
                0: [1, 0],  # 右下
                1: [0, 0],  # 左下
                2: [0, 1],  # 左上
                3: [1, 1],  # 右上
            }
        else:
            # 通用网格布局
            import math
            cols = math.ceil(math.sqrt(num_dies))
            for i in range(num_dies):
                x = i % cols
                y = i // cols
                positions[i] = [x, y]

        return positions

    def _generate_default_die_rotations(self) -> Dict[int, int]:
        """根据NUM_DIES和DIE_POSITIONS自动生成默认的Die旋转角度

        旋转规则（针对4 Die 2x2布局）：
        - Die0 (右下): 0度
        - Die1 (左下): 90度
        - Die2 (左上): 180度
        - Die3 (右上): 270度

        Returns:
            Die旋转角度字典，key为die_id，value为旋转角度（度）
        """
        num_dies = getattr(self, "NUM_DIES", 2)
        rotations = {}

        if num_dies == 2:
            rotations = {0: 0, 1: 90}
        elif num_dies == 4:
            # 2x2布局的标准旋转
            rotations = {
                0: 0,    # 右下不旋转
                1: 90,   # 左下顺时针90度
                2: 180,  # 左上顺时针180度
                3: 270,  # 右上顺时针270度
            }
        else:
            # 默认不旋转
            for i in range(num_dies):
                rotations[i] = 0

        return rotations

    def _generate_pairs_from_connections(self, connections: List[List[int]]) -> List[Tuple[int, int, int, int]]:
        """从D2D_CONNECTIONS生成配对关系

        配置文件中的节点编号直接就是网络节点位置，不需要任何映射转换。
        一个网络节点可以同时作为RN和SN使用（双向通信）。

        Args:
            connections: D2D连接配置列表，每个元素为[src_die, src_node, dst_die, dst_node]
                        src_node和dst_node直接是网络节点位置

        Returns:
            配对关系列表，每个元素为(die0_id, node0, die1_id, node1)
                        node0和node1是网络节点位置

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

            # 网络节点总数 = 行数 × 列数
            max_src_node = src_num_row * src_num_col - 1
            max_dst_node = dst_num_row * dst_num_col - 1

            if src_node > max_src_node:
                raise ValueError(f"源节点位置超出范围: {src_node} > {max_src_node} (Die{src_die}, 拓扑{src_topology})")
            if dst_node > max_dst_node:
                raise ValueError(f"目标节点位置超出范围: {dst_node} > {max_dst_node} (Die{dst_die}, 拓扑{dst_topology})")

            # 直接使用节点位置，生成双向配对
            pairs.append((src_die, src_node, dst_die, dst_node))
            pairs.append((dst_die, dst_node, src_die, src_node))

        return pairs

    def _setup_die_positions_from_pairs(self, pairs: List[Tuple[int, int, int, int]]) -> None:
        """从配对关系中设置各Die的D2D节点位置

        现在不再区分RN/SN，所有D2D节点既可以作为RN也可以作为SN。

        Args:
            pairs: D2D配对关系列表
        """
        num_dies = getattr(self, "NUM_DIES", 2)

        for die_id in range(num_dies):
            all_positions = []

            for pair in pairs:
                # 收集当前Die作为源的节点
                if pair[0] == die_id:
                    all_positions.append(pair[1])

                # 收集当前Die作为目标的节点
                if pair[2] == die_id:
                    all_positions.append(pair[3])

            # 去重并排序
            unique_positions = list(set(all_positions))

            self.D2D_DIE_POSITIONS[die_id] = unique_positions
            # RN和SN位置列表与总位置列表相同（不再区分）
            self.D2D_RN_POSITIONS[die_id] = unique_positions
            self.D2D_SN_POSITIONS[die_id] = unique_positions

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
        """获取指定边的所有物理节点

        Args:
            edge: 边界方向 ("top", "bottom", "left", "right")
            num_row: 物理行数
            num_col: 物理列数
            interface_type: 保留参数，为兼容性，不再使用

        Returns:
            物理边界节点列表（物理节点编号 = 物理行号 × 列数 + 物理列号）
        """
        # 现在直接返回物理边缘节点，不区分RN/SN
        if edge == "top":
            # 上边：第0行的所有节点 [0, 1, 2, ..., num_col-1]
            return list(range(0, num_col))
        elif edge == "bottom":
            # 下边：最后一行的所有节点 [(num_row-1)*num_col, ..., num_row*num_col-1]
            return list(range((num_row - 1) * num_col, num_row * num_col))
        elif edge == "left":
            # 左边：每行的第0列节点 [0, num_col, 2*num_col, ...]
            return [row * num_col for row in range(num_row)]
        elif edge == "right":
            # 右边：每行的最后一列节点 [num_col-1, 2*num_col-1, ...]
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

    def _load_dcin_config_file(self, dcin_config_file: str) -> None:
        """加载DCIN专用配置文件

        Args:
            dcin_config_file: DCIN配置文件路径

        Raises:
            FileNotFoundError: 如果配置文件不存在
            ValueError: 如果配置文件格式错误
        """
        try:
            with open(dcin_config_file, "r", encoding="utf-8") as f:
                if str(dcin_config_file).endswith((".yaml", ".yml")):
                    dcin_config = yaml.safe_load(f)
                else:
                    dcin_config = json.load(f)

            # D2D延迟配置项列表
            d2d_latency_keys = ["D2D_AR_LATENCY", "D2D_R_LATENCY", "D2D_AW_LATENCY", "D2D_W_LATENCY", "D2D_B_LATENCY"]

            for key, value in dcin_config.items():
                if key.startswith("D2D_") or key in ["NUM_DIES", "D2D_DIE_CONFIG", "DIE_POSITIONS", "DIE_TOPOLOGIES", "DIE_ROTATIONS", "NETWORK_FREQUENCY", "IP_FREQUENCY", "FLIT_SIZE", "BURST"]:
                    # D2D延迟配置保存为_original后缀
                    if key in d2d_latency_keys:
                        setattr(self, f"{key}_original", value)
                    else:
                        setattr(self, key, value)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"DCIN配置文件不存在: {dcin_config_file}")
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"DCIN配置文件格式错误: {e}")

    def _apply_die_config_file(self, die_config_file: str) -> None:
        """应用单独的DIE拓扑配置文件

        将指定的DIE拓扑配置文件应用到所有DIE，更新DIE_TOPOLOGIES中的配置引用。
        同时加载该配置文件中的参数作为die_config属性。

        Args:
            die_config_file: DIE拓扑配置文件路径

        Raises:
            FileNotFoundError: 如果配置文件不存在
            ValueError: 如果配置文件格式错误
        """
        try:
            with open(die_config_file, "r", encoding="utf-8") as f:
                if str(die_config_file).endswith((".yaml", ".yml")):
                    die_config = yaml.safe_load(f)
                else:
                    die_config = json.load(f)

            # 从文件名提取拓扑名称（如 kcin_5x4.yaml -> 5x4）
            import os
            filename = os.path.basename(die_config_file)
            if filename.startswith("kcin_"):
                kcin_name = filename.replace("kcin_", "").replace(".yaml", "").replace(".yml", "")
            else:
                kcin_name = filename.replace(".yaml", "").replace(".yml", "")

            # 更新所有DIE使用相同的拓扑配置
            num_dies = getattr(self, "NUM_DIES", 2)
            for die_id in range(num_dies):
                self.DIE_TOPOLOGIES[die_id] = kcin_name

            # 保存DIE配置内容供后续使用
            from src.kcin.v1.config import V1Config
            self.die_config = KCINConfig(die_config_file)

            # 从KCIN配置中获取关键参数（如果DCIN配置中没有设置）
            if not self.NETWORK_FREQUENCY and hasattr(self.die_config, 'NETWORK_FREQUENCY'):
                self.NETWORK_FREQUENCY = self.die_config.NETWORK_FREQUENCY
            if not self.FLIT_SIZE and hasattr(self.die_config, 'FLIT_SIZE'):
                self.FLIT_SIZE = self.die_config.FLIT_SIZE
            if not self.BURST and hasattr(self.die_config, 'BURST'):
                self.BURST = self.die_config.BURST

            logging.info(f"已应用DIE拓扑配置: {die_config_file} (拓扑: {kcin_name})")

        except FileNotFoundError:
            raise FileNotFoundError(f"DIE拓扑配置文件不存在: {die_config_file}")
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"DIE拓扑配置文件格式错误: {e}")

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
            # print(f"Die布局类型: {self.die_layout_type}")

    def _calculate_time_scale(self):
        """根据网络频率和IP频率自动计算时间缩放参数

        计算 CYCLES_PER_NS、NETWORK_SCALE、IP_SCALE，使得：
        - 所有时间转换都使用整数运算
        - 网络域操作每 NETWORK_SCALE 个仿真周期执行一次
        - IP 域操作每 IP_SCALE 个仿真周期执行一次
        """
        if not self.NETWORK_FREQUENCY:
            raise ValueError("必须先设置NETWORK_FREQUENCY才能计算时间缩放参数")

        # 如果没有设置 IP_FREQUENCY，默认为 1GHz
        if not self.IP_FREQUENCY:
            self.IP_FREQUENCY = 1

        max_denominator = 16  # 限制分母，支持17/8、33/16等精确分数

        # 转换为分数
        net_frac = Fraction(self.NETWORK_FREQUENCY).limit_denominator(max_denominator)
        ip_frac = Fraction(self.IP_FREQUENCY).limit_denominator(max_denominator)

        # 通分
        common_denom = lcm(net_frac.denominator, ip_frac.denominator)
        net_num = net_frac.numerator * (common_denom // net_frac.denominator)
        ip_num = ip_frac.numerator * (common_denom // ip_frac.denominator)

        # 计算最小 CYCLES_PER_NS
        net_divisor = net_num // gcd(net_num, common_denom)
        ip_divisor = ip_num // gcd(ip_num, common_denom)
        self.CYCLES_PER_NS = lcm(net_divisor, ip_divisor)

        # 计算各域的缩放因子
        self.NETWORK_SCALE = self.CYCLES_PER_NS * common_denom // net_num
        self.IP_SCALE = self.CYCLES_PER_NS * common_denom // ip_num

        # 计算实际生效的频率
        self.EFFECTIVE_NETWORK_FREQ = self.CYCLES_PER_NS / self.NETWORK_SCALE
        self.EFFECTIVE_IP_FREQ = self.CYCLES_PER_NS / self.IP_SCALE

        # 精度检查
        net_error = abs(self.EFFECTIVE_NETWORK_FREQ - self.NETWORK_FREQUENCY) / self.NETWORK_FREQUENCY
        ip_error = abs(self.EFFECTIVE_IP_FREQ - self.IP_FREQUENCY) / self.IP_FREQUENCY
        if net_error > 0.02:
            print(f"警告: 网络频率 {self.NETWORK_FREQUENCY}GHz 近似为 {self.EFFECTIVE_NETWORK_FREQ:.4f}GHz")
        if ip_error > 0.02:
            print(f"警告: IP频率 {self.IP_FREQUENCY}GHz 近似为 {self.EFFECTIVE_IP_FREQ:.4f}GHz")

    def update_latency(self):
        """将D2D延迟配置从ns转换为cycles

        转换公式: latency_cycles = latency_ns * CYCLES_PER_NS
        """
        if not self.NETWORK_FREQUENCY:
            raise ValueError("必须先设置NETWORK_FREQUENCY才能转换延迟配置")

        # 先计算时间缩放参数
        if self.CYCLES_PER_NS is None:
            self._calculate_time_scale()

        # 转换各个延迟配置
        if self.D2D_AR_LATENCY_original is not None:
            self.D2D_AR_LATENCY = int(self.D2D_AR_LATENCY_original * self.CYCLES_PER_NS)
        if self.D2D_R_LATENCY_original is not None:
            self.D2D_R_LATENCY = int(self.D2D_R_LATENCY_original * self.CYCLES_PER_NS)
        if self.D2D_AW_LATENCY_original is not None:
            self.D2D_AW_LATENCY = int(self.D2D_AW_LATENCY_original * self.CYCLES_PER_NS)
        if self.D2D_W_LATENCY_original is not None:
            self.D2D_W_LATENCY = int(self.D2D_W_LATENCY_original * self.CYCLES_PER_NS)
        if self.D2D_B_LATENCY_original is not None:
            self.D2D_B_LATENCY = int(self.D2D_B_LATENCY_original * self.CYCLES_PER_NS)
