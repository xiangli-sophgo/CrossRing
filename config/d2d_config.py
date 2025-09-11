# d2d_config.py
"""
D2D (Die-to-Die) Configuration Manager

This module provides D2DConfig class that extends CrossRingConfig with D2D-specific
configuration parameters and automatic node placement calculation.
"""

import json
import yaml
import logging
from typing import List, Tuple, Optional, Dict
from .config import CrossRingConfig


class D2DConfig(CrossRingConfig):
    """
    D2D配置管理器，继承CrossRingConfig并添加D2D特定功能

    主要功能：
    1. 管理D2D布局配置（水平/垂直摆放）
    2. 自动计算RN/SN节点位置
    3. 管理RN-SN配对关系
    4. 验证D2D配置的合理性
    """

    def __init__(self, die_config_file=None, d2d_config_file=None):
        """
        初始化D2D配置

        Args:
            die_config_file: Die拓扑配置文件路径
            d2d_config_file: D2D专用配置文件路径
        """
        # 初始化基础配置
        super().__init__(die_config_file)

        # 初始化D2D配置
        self._init_d2d_config()

        # 如果指定了D2D配置文件，则加载D2D特定配置
        if d2d_config_file:
            self._load_d2d_config_file(d2d_config_file)

        # 生成D2D配对关系（需要在加载配置后）
        self._generate_d2d_pairs()

        # 更新 CHANNEL_SPEC 以包含 D2D 接口
        self._update_channel_spec_for_d2d()

        # 验证配置
        self._validate_d2d_layout()

    def _init_d2d_config(self):
        """初始化D2D基础配置"""
        # D2D基础参数
        self.D2D_DIE_POSITIONS = {}  # Die的D2D节点位置字典: {die_id: [node_positions]}
        self.D2D_PAIRS = []  # D2D连接对：(die0_id, node0, die1_id, node1)

        # D2D扩展参数
        self.D2D_CONNECTION_MAP = {}  # Die间连接映射（兼容性保留）
        self.D2D_MULTI_HOP_ENABLED = False  # 多跳路由支持
        self.D2D_ROUTING_ALGORITHM = "shortest_path"  # 路由算法

        # Die 布局推断相关属性
        self.die_layout_positions = {}  # 推断的Die布局位置: {die_id: (x, y)}
        self.die_layout_type = ""  # 布局类型：如 "2x1", "1x2", "2x2" 等

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
        
        # 设置各Die的D2D节点位置到字典中
        num_dies = getattr(self, "NUM_DIES", 2)
        for die_id in range(num_dies):
            positions = []
            for pair in pairs:
                if pair[0] == die_id:
                    positions.append(pair[1])
                if pair[2] == die_id:
                    positions.append(pair[3])
            # 去重并设置到字典中
            self.D2D_DIE_POSITIONS[die_id] = list(set(positions))

        # 推断 Die 布局
        self._infer_die_layout()


    def _get_edge_nodes(self, edge, num_row, num_col):
        """获取指定边的所有节点"""
        if edge == "top":
            # 第二行（row=1）
            return list(range(num_col, 2 * num_col))
        elif edge == "bottom":
            # 最后一行（row=num_row-1）
            return list(range((num_row - 1) * num_col, num_row * num_col))
        elif edge == "left":
            # 奇数行的左边（col=0）
            return [row * num_col for row in range(1, num_row, 2)]
        elif edge == "right":
            # 奇数行的右边（col=num_col-1）
            return [row * num_col + (num_col - 1) for row in range(1, num_row, 2)]
        else:
            return []

    def _get_opposite_edge(self, edge):
        """获取对边"""
        opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
        return opposite.get(edge, "")

    def _generate_d2d_pairs_from_config(self, config):
        """
        根据配置生成D2D节点配对

        Args:
            config: Die配置字典

        Returns:
            pairs: [(die0_id, node0, die1_id, node1), ...]
        """
        pairs = []
        processed = set()  # 避免重复处理

        for die_id, die_config in config.items():
            # 获取Die的网络规模
            num_row = die_config["num_row"]
            num_col = die_config["num_col"]

            # 处理每个连接
            for edge, conn_info in die_config["connections"].items():
                target_die = conn_info["die"]
                d2d_node_indices = conn_info["d2d_nodes"]  # 相对位置索引列表

                # 避免重复处理
                pair_key = tuple(sorted([die_id, target_die]))
                if pair_key in processed:
                    continue
                processed.add(pair_key)

                # 获取边缘节点
                edge_nodes = self._get_edge_nodes(edge, num_row, num_col)

                # 获取对端信息
                opposite_edge = self._get_opposite_edge(edge)
                target_config = config[target_die]
                target_num_row = target_config["num_row"]
                target_num_col = target_config["num_col"]
                target_nodes = self._get_edge_nodes(opposite_edge, target_num_row, target_num_col)

                # 根据相对索引获取具体节点
                d2d_nodes = [edge_nodes[i] for i in d2d_node_indices if i < len(edge_nodes)]

                # 获取对端的相对索引（从对端配置中找到）
                target_d2d_indices = None
                for target_edge, target_conn_info in target_config["connections"].items():
                    if target_conn_info["die"] == die_id and target_edge == opposite_edge:
                        target_d2d_indices = target_conn_info["d2d_nodes"]
                        break

                if target_d2d_indices is None:
                    continue  # 找不到对端配置，跳过

                target_d2d_nodes = [target_nodes[i] for i in target_d2d_indices if i < len(target_nodes)]

                # 生成配对
                for i in range(min(len(d2d_nodes), len(target_d2d_nodes))):
                    pairs.append((die_id, d2d_nodes[i], target_die, target_d2d_nodes[i]))

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

            # 更新D2D特定参数
            for key, value in d2d_config.items():
                if key.startswith("D2D_") or key == "NUM_DIES" or key == "D2D_DIE_CONFIG":
                    setattr(self, key, value)

        except FileNotFoundError as e:
            pass  # D2D配置文件不存在，使用默认配置
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            print(f"警告: D2D配置文件格式错误: {e}")
            pass  # 配置文件格式错误，使用默认配置


    def _create_d2d_pairs(self):
        """创建D2D连接对关系（旧方法，已被新的配置驱动方式替代）"""
        # 现在所有Die配置都使用新的配置驱动方式，此方法不再使用
        # 保留此方法以防某些地方还有调用，但实际上已经不执行任何操作
        return

    def _get_die_positions(self, die_id: int) -> List[int]:
        """获取指定Die的D2D节点位置"""
        return self.D2D_DIE_POSITIONS.get(die_id, [])

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

        # 检查节点位置有效性
        max_node_id = self.NUM_NODE - 1

        for die_id in range(num_dies):
            positions = self.D2D_DIE_POSITIONS.get(die_id, [])
            for pos in positions:
                if pos < 0 or pos > max_node_id:
                    raise ValueError(f"Die{die_id}节点位置无效: {pos}, 有效范围: 0-{max_node_id}")

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

        # 每个Die应该至少连接到其他2个Die（对于2x2网格拓扑）
        for die_id, connected_dies in die_connections.items():
            if len(connected_dies) != 2:
                raise ValueError(f"Die{die_id} 应该连接到2个其他Die，当前连接到: {connected_dies}")

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
        
        # 方向到坐标偏移的映射
        direction_offsets = {
            "left": (-1, 0),
            "right": (1, 0),
            "top": (0, -1),
            "bottom": (0, 1)
        }
        
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
                target_die = conn_info["die"]
                
                if target_die in visited:
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
            
            # 推断布局类型
            max_x = max(pos[0] for pos in normalized_positions.values())
            max_y = max(pos[1] for pos in normalized_positions.values())
            self.die_layout_type = f"{max_x + 1}x{max_y + 1}"
            
            print(f"[D2D布局推断] 检测到 {self.die_layout_type} 布局:")
            for die_id in range(num_dies):
                if die_id in normalized_positions:
                    x, y = normalized_positions[die_id]
                    print(f"  Die{die_id}: ({x}, {y})")
        else:
            # 回退到默认布局
            self.die_layout_positions = {i: (i, 0) for i in range(num_dies)}
            self.die_layout_type = f"{num_dies}x1"
