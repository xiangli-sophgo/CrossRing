# d2d_config.py
"""
D2D (Die-to-Die) Configuration Manager

This module provides D2DConfig class that extends CrossRingConfig with D2D-specific
configuration parameters and automatic node placement calculation.
"""

import json
import yaml
import logging
import os
from typing import List, Tuple, Optional, Dict


class D2DConfig:
    """
    D2D配置管理器，独立管理D2D系统配置

    主要功能：
    1. 管理D2D布局配置和每个Die的拓扑信息
    2. 自动加载每个Die对应的拓扑配置文件
    3. 管理RN-SN配对关系
    4. 验证D2D配置的合理性
    """

    def __init__(self, d2d_config_file):
        """
        初始化D2D配置

        Args:
            d2d_config_file: D2D专用配置文件路径
        """
        # 初始化D2D配置
        self._init_d2d_config()

        # 加载D2D特定配置
        if d2d_config_file:
            self._load_d2d_config_file(d2d_config_file)
        else:
            raise ValueError("必须指定d2d_config_file")

        # 生成D2D配对关系（需要在加载配置后）
        self._generate_d2d_pairs()

        # 更新 CHANNEL_SPEC 以包含 D2D 接口
        self._update_channel_spec_for_d2d()

        # 验证配置
        self._validate_d2d_layout()

    def _init_d2d_config(self):
        """初始化D2D基础配置"""
        # D2D基础参数
        self.D2D_DIE_POSITIONS = {}  # Die的D2D节点位置字典: {die_id: [node_positions]} (向后兼容)
        self.D2D_RN_POSITIONS = {}  # Die的D2D RN节点位置字典: {die_id: [node_positions]}
        self.D2D_SN_POSITIONS = {}  # Die的D2D SN节点位置字典: {die_id: [node_positions]}
        self.D2D_PAIRS = []  # D2D连接对：(die0_id, node0, die1_id, node1)

        # D2D扩展参数
        self.D2D_CONNECTION_MAP = {}  # Die间连接映射（兼容性保留）
        self.D2D_MULTI_HOP_ENABLED = False  # 多跳路由支持
        self.D2D_ROUTING_ALGORITHM = "shortest_path"  # 路由算法

        # Die 布局推断相关属性
        self.die_layout_positions = {}  # 推断的Die布局位置: {die_id: (x, y)}
        self.die_layout_type = ""  # 布局类型：如 "2x1", "1x2", "2x2" 等

        # 基础配置属性（原来由CrossRingConfig提供）
        self.CHANNEL_SPEC = {}  # 通道规格
        self.CH_NAME_LIST = []  # 通道名称列表
        self.DIE_TOPOLOGIES = {}  # Die拓扑信息存储

        # 网络基础参数（将从D2D配置文件中加载）
        self.NETWORK_FREQUENCY = None  # 从配置文件读取
        self.FLIT_SIZE = None  # 从配置文件读取
        self.BURST = None  # 从配置文件读取

    def _generate_d2d_pairs(self):
        """生成D2D配对关系"""
        num_dies = getattr(self, "NUM_DIES", 2)
        if num_dies < 2:
            raise ValueError(f"D2D需要至少2个Die，当前配置: {num_dies}")

        self._setup_config_based()

    def _setup_config_based(self):
        """基于配置的通用Die设置方式"""
        # 从配置中读取Die配置结构
        die_config = getattr(self, "D2D_DIE_CONFIG", None)

        if die_config is None:
            raise ValueError("未找到D2D_DIE_CONFIG配置，请在YAML配置文件中定义D2D_DIE_CONFIG")

        # 使用配置生成D2D配对
        pairs = self._generate_d2d_pairs_from_config(die_config)

        # 设置配对结果
        self.D2D_PAIRS = pairs

        # 设置各Die的D2D节点位置到字典中（分别处理RN和SN）
        num_dies = getattr(self, "NUM_DIES", 2)
        for die_id in range(num_dies):
            all_positions = []
            rn_positions = []
            sn_positions = []

            for pair in pairs:
                if pair[0] == die_id:
                    all_positions.append(pair[1])
                    # 根据节点奇偶行判断是RN还是SN
                    node_pos = pair[1]
                    # 获取Die的拓扑信息
                    topology_str = self.DIE_TOPOLOGIES.get(die_id)
                    if topology_str:
                        _, num_col = self._parse_topology(topology_str)
                        row = node_pos // num_col
                        if row % 2 == 0:  # 偶数行是SN
                            sn_positions.append(node_pos)
                        else:  # 奇数行是RN
                            rn_positions.append(node_pos)
                if pair[2] == die_id:
                    all_positions.append(pair[3])
                    # 根据节点奇偶行判断是RN还是SN
                    node_pos = pair[3]
                    # 获取Die的拓扑信息
                    topology_str = self.DIE_TOPOLOGIES.get(die_id)
                    if topology_str:
                        _, num_col = self._parse_topology(topology_str)
                        row = node_pos // num_col
                        if row % 2 == 0:  # 偶数行是SN
                            sn_positions.append(node_pos)
                        else:  # 奇数行是RN
                            rn_positions.append(node_pos)

            # 去重并设置到字典中
            self.D2D_DIE_POSITIONS[die_id] = list(set(all_positions))  # 向后兼容
            self.D2D_RN_POSITIONS[die_id] = list(set(rn_positions))
            self.D2D_SN_POSITIONS[die_id] = list(set(sn_positions))

        # 推断 Die 布局
        self._infer_die_layout()

    def _parse_topology(self, topology_str):
        """
        解析拓扑字符串并从配置文件中获取实际的行列信息

        Args:
            topology_str: 拓扑字符串，如 "5x4", "10x4" 等

        Returns:
            tuple: (num_row, num_col) 实际的逻辑行列数
        """
        try:
            # 加载对应的拓扑配置文件获取真实的行列信息
            topo_config = self._load_topology_config(topology_str)
            num_row = topo_config.get("NUM_ROW")
            num_col = topo_config.get("NUM_COL")

            if num_row is None or num_col is None:
                raise ValueError(f"拓扑配置文件缺少NUM_ROW或NUM_COL: {topology_str}")

            return num_row, num_col
        except Exception as e:
            raise ValueError(f"解析拓扑 '{topology_str}' 失败: {e}")

    def _load_topology_config(self, topology_str):
        """
        加载对应的拓扑配置文件

        Args:
            topology_str: 拓扑字符串，如 "5x4"

        Returns:
            dict: 拓扑配置内容
        """
        import yaml
        import os

        # 构建拓扑配置文件路径
        config_path = os.path.join("../config", "topologies", f"topo_{topology_str}.yaml")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                topo_config = yaml.safe_load(f)
                return topo_config
        except FileNotFoundError:
            raise FileNotFoundError(f"拓扑配置文件不存在: {config_path}")
        except Exception as e:
            raise RuntimeError(f"加载拓扑配置文件失败: {config_path}, 错误: {e}")

    def _get_edge_nodes(self, edge, num_row, num_col, interface_type=None):
        """
        获取指定边的所有节点

        Args:
            edge: 边界方向 ("top", "bottom", "left", "right")
            num_row: 逻辑行数
            num_col: 列数
            interface_type: 接口类型 ("rn" 或 "sn")，用于区分奇偶行
        """
        if edge == "top":
            # 第一行（row=0）和第二行（row=1）
            if interface_type == "rn":
                # RN在奇数行（第1行）
                return list(range(num_col, 2 * num_col))
            elif interface_type == "sn":
                # SN在偶数行（第0行）
                return list(range(0, num_col))
            else:
                # 默认返回第0行（向后兼容）
                return list(range(0, num_col))
        elif edge == "bottom":
            # 最后两行
            if interface_type == "rn":
                # RN在奇数行（最后一行）
                return list(range((num_row - 1) * num_col, num_row * num_col))
            elif interface_type == "sn":
                # SN在偶数行（倒数第二行）
                return list(range((num_row - 2) * num_col, (num_row - 1) * num_col))
            else:
                # 默认返回最后一行（向后兼容）
                return list(range((num_row - 1) * num_col, num_row * num_col))
        elif edge == "left":
            # 所有行的左边（col=0）
            if interface_type == "rn":
                # RN在奇数行
                return [row * num_col for row in range(1, num_row, 2)]
            elif interface_type == "sn":
                # SN在偶数行
                return [row * num_col for row in range(0, num_row, 2)]
            else:
                # 默认返回所有行
                return [row * num_col for row in range(num_row)]
        elif edge == "right":
            # 所有行的右边（col=num_col-1）
            if interface_type == "rn":
                # RN在奇数行
                return [row * num_col + (num_col - 1) for row in range(1, num_row, 2)]
            elif interface_type == "sn":
                # SN在偶数行
                return [row * num_col + (num_col - 1) for row in range(0, num_row, 2)]
            else:
                # 默认返回所有行
                return [row * num_col + (num_col - 1) for row in range(num_row)]
        else:
            return []

    def _get_opposite_edge(self, edge):
        """获取对边"""
        opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
        return opposite.get(edge, "")

    def _generate_d2d_pairs_from_config(self, config):
        """
        根据新配置格式生成D2D节点配对

        Args:
            config: Die配置字典

        Returns:
            pairs: [(die0_id, node0, die1_id, node1), ...]
        """
        # 先验证配置
        self._validate_d2d_config(config)

        pairs = []
        processed = set()

        # 初始化Die拓扑信息存储
        if not hasattr(self, "DIE_TOPOLOGIES"):
            self.DIE_TOPOLOGIES = {}

        for die_id, die_config in config.items():
            # 解析topology字符串获取行列数
            topology_str = die_config["topology"]
            num_row, num_col = self._parse_topology(topology_str)

            # 存储Die拓扑信息供后续使用
            self.DIE_TOPOLOGIES[die_id] = topology_str

            for edge, conn_info in die_config["connections"].items():
                d2d_nodes = conn_info["d2d_nodes"]
                node_mapping = conn_info["node_to_die_mapping"]

                # 分别获取RN和SN的边界节点
                edge_rn_nodes = self._get_edge_nodes(edge, num_row, num_col, "rn")
                edge_sn_nodes = self._get_edge_nodes(edge, num_row, num_col, "sn")

                # 按目标Die分组处理
                die_groups = {}
                for node_idx in d2d_nodes:
                    target_die = node_mapping[node_idx]
                    if target_die not in die_groups:
                        die_groups[target_die] = []
                    die_groups[target_die].append(node_idx)

                # 为每个目标Die生成配对
                for target_die, node_indices in die_groups.items():
                    pair_key = tuple(sorted([die_id, target_die]))
                    if pair_key in processed:
                        continue
                    processed.add(pair_key)

                    # 获取源节点（RN和SN）
                    src_rn_nodes = [edge_rn_nodes[i] for i in node_indices]
                    src_sn_nodes = [edge_sn_nodes[i] for i in node_indices]

                    # 获取目标节点（从对端配置）
                    opposite_edge = self._get_opposite_edge(edge)
                    target_config = config[target_die]
                    target_conn = target_config["connections"][opposite_edge]

                    # 解析目标Die的拓扑信息
                    target_topology_str = target_config["topology"]
                    target_num_row, target_num_col = self._parse_topology(target_topology_str)

                    target_edge_rn_nodes = self._get_edge_nodes(opposite_edge, target_num_row, target_num_col, "rn")
                    target_edge_sn_nodes = self._get_edge_nodes(opposite_edge, target_num_row, target_num_col, "sn")

                    # 找到对应的目标节点索引
                    target_indices = []
                    for idx, t_die in target_conn["node_to_die_mapping"].items():
                        if t_die == die_id:
                            target_indices.append(idx)

                    target_rn_nodes = [target_edge_rn_nodes[i] for i in target_indices]
                    target_sn_nodes = [target_edge_sn_nodes[i] for i in target_indices]

                    # 生成RN-SN配对（RN发送到SN接收）
                    for i in range(min(len(src_rn_nodes), len(target_sn_nodes))):
                        pairs.append((die_id, src_rn_nodes[i], target_die, target_sn_nodes[i]))

                    # 生成SN-RN配对（SN发送到RN接收）
                    for i in range(min(len(src_sn_nodes), len(target_rn_nodes))):
                        pairs.append((die_id, src_sn_nodes[i], target_die, target_rn_nodes[i]))

        return pairs

    def _update_channel_spec_for_d2d(self):
        """更新 CHANNEL_SPEC 以包含 D2D 接口类型"""
        if hasattr(self, "D2D_ENABLED") and self.D2D_ENABLED:
            # 为 D2D 接口添加通道规格
            if "d2d_rn" not in self.CHANNEL_SPEC:
                self.CHANNEL_SPEC["d2d_rn"] = 1  # 每个 D2D 节点一个 RN 接口
            if "d2d_sn" not in self.CHANNEL_SPEC:
                self.CHANNEL_SPEC["d2d_sn"] = 1  # 每个 D2D 节点一个 SN 接口

            # 重新构建 CH_NAME_LIST
            self.CH_NAME_LIST = []
            for key in self.CHANNEL_SPEC:
                for idx in range(self.CHANNEL_SPEC[key]):
                    self.CH_NAME_LIST.append(f"{key}_{idx}")

    def _load_d2d_config_file(self, d2d_config_file):
        """加载D2D专用配置文件"""
        try:
            with open(d2d_config_file, "r", encoding="utf-8") as f:
                if str(d2d_config_file).endswith((".yaml", ".yml")):
                    d2d_config = yaml.safe_load(f)
                else:
                    d2d_config = json.load(f)

            # 更新D2D特定参数和网络基础参数
            for key, value in d2d_config.items():
                if key.startswith("D2D_") or key in ["NUM_DIES", "D2D_DIE_CONFIG", "NETWORK_FREQUENCY", "FLIT_SIZE", "BURST"]:
                    setattr(self, key, value)

            print(f"成功加载D2D配置文件: {d2d_config_file}")
            if hasattr(self, "D2D_DIE_CONFIG"):
                print(f"D2D_DIE_CONFIG包含 {len(self.D2D_DIE_CONFIG)} 个Die配置")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"D2D配置文件不存在: {d2d_config_file}")
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"D2D配置文件格式错误: {e}")

    def _create_d2d_pairs(self):
        """创建D2D连接对关系（旧方法，已被新的配置驱动方式替代）"""
        # 现在所有Die配置都使用新的配置驱动方式，此方法不再使用
        # 保留此方法以防某些地方还有调用，但实际上已经不执行任何操作
        return

    def _get_die_positions(self, die_id: int) -> List[int]:
        """获取指定Die的D2D节点位置"""
        return self.D2D_DIE_POSITIONS.get(die_id, [])

    def _validate_d2d_config(self, die_config):
        """验证D2D配置的正确性"""
        errors = []
        num_dies = getattr(self, "NUM_DIES", 2)

        # 1. 验证所有Die都有配置
        for die_id in range(num_dies):
            if die_id not in die_config:
                errors.append(f"缺少Die {die_id}的配置")

        # 2. 验证每个连接配置
        for die_id, config in die_config.items():
            if "connections" not in config:
                errors.append(f"Die {die_id}缺少connections配置")
                continue

            for edge, conn_info in config.get("connections", {}).items():
                # 检查必需字段
                if "connect_die" not in conn_info:
                    errors.append(f"Die {die_id} {edge}边缺少connect_die")
                if "d2d_nodes" not in conn_info:
                    errors.append(f"Die {die_id} {edge}边缺少d2d_nodes")
                if "node_to_die_mapping" not in conn_info:
                    errors.append(f"Die {die_id} {edge}边缺少node_to_die_mapping")
                    continue

                # 验证映射完整性
                connect_die = conn_info["connect_die"]
                d2d_nodes = conn_info["d2d_nodes"]
                mapping = conn_info["node_to_die_mapping"]

                # 检查connect_die是否存在
                if connect_die >= num_dies:
                    errors.append(f"Die {die_id} {edge}边connect_die {connect_die}不存在")

                # 检查所有d2d_nodes都有映射
                for node_idx in d2d_nodes:
                    if node_idx not in mapping:
                        errors.append(f"Die {die_id} {edge}边节点{node_idx}缺少映射")

                # 检查目标Die是否存在
                for node_idx, target_die in mapping.items():
                    if target_die >= num_dies:
                        errors.append(f"Die {die_id}节点{node_idx}映射到不存在的Die {target_die}")

                # 验证connect_die与node_to_die_mapping的一致性（允许部分不同）
                mapped_dies = set(mapping.values())
                if connect_die not in mapped_dies:
                    errors.append(f"Die {die_id} {edge}边connect_die {connect_die}不在node_to_die_mapping中")

                # 验证节点索引有效性
                # 解析topology获取行列数
                if "topology" in config:
                    num_row, num_col = self._parse_topology(config["topology"])
                    edge_nodes = self._get_edge_nodes(edge, num_row, num_col)
                    for node_idx in d2d_nodes:
                        if node_idx >= len(edge_nodes):
                            errors.append(f"Die {die_id} {edge}边节点索引{node_idx}超出范围")
                else:
                    errors.append(f"Die {die_id}缺少topology配置")

        # 3. 验证双向连接一致性
        errors.extend(self._validate_bidirectional_connections(die_config))

        if errors:
            raise ValueError("D2D配置验证失败:\n" + "\n".join(errors))

        return True

    def _validate_bidirectional_connections(self, die_config):
        """验证双向连接的一致性"""
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

        # 检查节点位置有效性（现在每个Die有不同拓扑）
        for die_id in range(num_dies):
            # 获取该Die的拓扑信息
            topology_str = self.DIE_TOPOLOGIES.get(die_id)
            if topology_str:
                num_row, num_col = self._parse_topology(topology_str)
                max_node_id = (num_row * num_col) - 1

                positions = self.D2D_DIE_POSITIONS.get(die_id, [])
                for pos in positions:
                    if pos < 0 or pos > max_node_id:
                        raise ValueError(f"Die{die_id}节点位置无效: {pos}, 有效范围: 0-{max_node_id} (拓扑: {topology_str})")
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

        print(f"4-Die配置验证通过: {len(self.D2D_PAIRS)}个连接对")

    def get_die_boundary_nodes(self, die_id: int) -> Dict[str, List[int]]:
        """
        获取指定Die的边界节点

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

    def print_d2d_layout(self):
        """可视化打印D2D布局信息"""
        print("\n=== D2D布局配置 ===")
        print(f"拓扑规格: {self.NUM_ROW}x{self.NUM_COL} ({self.NUM_NODE}个节点)")
        print(f"Die数量: {getattr(self, 'NUM_DIES', 2)}")

        # 打印所有Die的D2D节点位置
        for die_id in range(getattr(self, "NUM_DIES", 2)):
            positions = self.D2D_DIE_POSITIONS.get(die_id, [])
            print(f"Die{die_id} D2D节点位置: {positions}")

        print(f"D2D连接对数量: {len(self.D2D_PAIRS)}")
        print("D2D连接对详情:")
        for i, pair in enumerate(self.D2D_PAIRS):
            print(f"  {i+1:2d}. Die{pair[0]}:节点{pair[1]} ↔ Die{pair[2]}:节点{pair[3]}")

        print("=" * 30)

    def generate_d2d_traffic_example(self, num_requests: int = 10) -> List[str]:
        """
        生成D2D测试流量示例

        Args:
            num_requests: 生成的请求数量

        Returns:
            流量字符串列表
        """
        import random

        traffic_lines = []
        traffic_lines.append("# D2D Traffic Generated by D2DConfig")
        traffic_lines.append("# Format: inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length")

        for i in range(num_requests):
            # 交替生成Die 0->1 和 Die 1->0 的流量
            if i % 2 == 0:
                src_die, dst_die = 0, 1
                src_node = random.choice(range(self.NUM_NODE))
                dst_node = random.choice(range(self.NUM_NODE))
            else:
                src_die, dst_die = 1, 0
                src_node = random.choice(range(self.NUM_NODE))
                dst_node = random.choice(range(self.NUM_NODE))

            inject_time = i * 100 + 100
            src_ip = f"gdma_{random.randint(0, 1)}"
            dst_ip = f"ddr_{random.randint(0, 1)}"
            req_type = random.choice(["R", "W"])
            burst_length = random.choice([4, 8, 16])

            line = f"{inject_time}, {src_die}, {src_node}, {src_ip}, {dst_die}, {dst_node}, {dst_ip}, {req_type}, {burst_length}"
            traffic_lines.append(line)

        return traffic_lines

    def save_d2d_config(self, config_file: str):
        """
        保存D2D配置到文件

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

    def _infer_die_layout(self):
        """
        根据 D2D_DIE_CONFIG 中的连接关系推断 Die 的 2D 布局位置

        算法：
        1. 选择 Die 0 作为参考点 (0, 0)
        2. 广度优先遍历，根据连接方向推断其他 Die 的位置
        3. 坐标归一化，使最小坐标为 (0, 0)
        """
        die_config = getattr(self, "D2D_DIE_CONFIG", None)
        if not die_config:
            # 没有配置时使用默认布局
            num_dies = getattr(self, "NUM_DIES", 2)
            if num_dies == 2:
                self.die_layout_positions = {0: (0, 0), 1: (1, 0)}
                self.die_layout_type = "2x1"
            return

        num_dies = getattr(self, "NUM_DIES", 2)

        # 方向到坐标偏移的映射 (标准数学坐标系：Y轴向上为正)
        direction_offsets = {"left": (-1, 0), "right": (1, 0), "top": (0, 1), "bottom": (0, -1)}  # 向左：x减1  # 向右：x加1  # 向上：y加1  # 向下：y减1

        # 存储每个 Die 的位置
        positions = {}
        visited = set()

        # BFS 队列：(die_id, x, y)
        from collections import deque

        queue = deque()

        # 从 Die 0 开始，设为参考点 (0, 0)
        queue.append((0, 0, 0))
        positions[0] = (0, 0)
        visited.add(0)

        while queue:
            current_die, current_x, current_y = queue.popleft()

            if current_die not in die_config:
                continue

            # 检查当前 Die 的所有连接
            connections = die_config[current_die].get("connections", {})

            for direction, conn_info in connections.items():
                # 使用connect_die字段确定目标Die
                target_die = conn_info.get("connect_die")

                if target_die is None or target_die in visited:
                    continue

                # 根据方向计算目标 Die 的位置
                dx, dy = direction_offsets.get(direction, (0, 0))
                target_x = current_x + dx
                target_y = current_y + dy

                positions[target_die] = (target_x, target_y)
                visited.add(target_die)
                queue.append((target_die, target_x, target_y))

        # 坐标归一化：使最小坐标为 (0, 0)
        if positions:
            min_x = min(pos[0] for pos in positions.values())
            min_y = min(pos[1] for pos in positions.values())

            normalized_positions = {}
            for die_id, (x, y) in positions.items():
                normalized_positions[die_id] = (x - min_x, y - min_y)

            self.die_layout_positions = normalized_positions

            # 推断布局类型 (行x列格式)
            max_x = max(pos[0] for pos in normalized_positions.values())  # 列数-1
            max_y = max(pos[1] for pos in normalized_positions.values())  # 行数-1
            num_cols = max_x + 1
            num_rows = max_y + 1
            self.die_layout_type = f"{num_rows}x{num_cols}"  # 行x列格式

            # 按行优先顺序显示（从上到下，从左到右）
            die_by_position = [(y, x, die_id) for die_id, (x, y) in normalized_positions.items()]
            die_by_position.sort()  # 按(row, col)排序

        else:
            raise ValueError
