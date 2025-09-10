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

        # D2D特定参数
        self.D2D_LAYOUT = "HORIZONTAL"  # 默认值，将从配置文件覆盖
        self.D2D_DIE0_POSITIONS = []  # Die0的D2D节点位置
        self.D2D_DIE1_POSITIONS = []  # Die1的D2D节点位置
        self.D2D_DIE2_POSITIONS = []  # Die2的D2D节点位置（4-Die支持）
        self.D2D_DIE3_POSITIONS = []  # Die3的D2D节点位置（4-Die支持）
        self.D2D_PAIRS = []  # D2D连接对：(die0_node, die1_node)
        
        # 4-Die特定参数
        self.D2D_CONNECTION_MAP = {}  # Die间连接映射
        self.D2D_MULTI_HOP_ENABLED = False  # 多跳路由支持
        self.D2D_ROUTING_ALGORITHM = "shortest_path"  # 路由算法

        # 如果指定了D2D配置文件，则加载D2D特定配置
        if d2d_config_file:
            self._load_d2d_config_file(d2d_config_file)

        # 根据拓扑和布局计算节点位置
        self._calculate_d2d_positions()

        # 创建D2D连接对关系
        self._create_d2d_pairs()

        # 更新 CHANNEL_SPEC 以包含 D2D 接口
        self._update_channel_spec_for_d2d()

        # 验证配置
        self._validate_d2d_layout()

    def _update_channel_spec_for_d2d(self):
        """更新 CHANNEL_SPEC 以包含 D2D 接口类型"""
        if hasattr(self, 'D2D_ENABLED') and self.D2D_ENABLED:
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
                if key.startswith("D2D_") or key == "NUM_DIES":
                    setattr(self, key, value)

        except FileNotFoundError:
            pass  # D2D配置文件不存在，使用默认配置
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            print(f"警告: D2D配置文件格式错误: {e}")
            pass  # 配置文件格式错误，使用默认配置

    def _calculate_d2d_positions(self):
        """根据拓扑类型和布局方式计算D2D节点位置"""
        layout = self.D2D_LAYOUT.upper()  # 转换为大写进行比较
        
        if layout == "HORIZONTAL":
            if not self.D2D_DIE0_POSITIONS or not self.D2D_DIE1_POSITIONS:
                self._calculate_horizontal_positions()
        elif layout == "VERTICAL":
            if not self.D2D_DIE0_POSITIONS or not self.D2D_DIE1_POSITIONS:
                self._calculate_vertical_positions()
        elif layout == "GRID_2X2":
            # 4-Die模式：检查所有Die的位置
            if (not self.D2D_DIE0_POSITIONS or not self.D2D_DIE1_POSITIONS or 
                not self.D2D_DIE2_POSITIONS or not self.D2D_DIE3_POSITIONS):
                self._calculate_grid_2x2_positions()
        else:
            raise ValueError(f"不支持的D2D布局方式: {self.D2D_LAYOUT}")

    def _calculate_horizontal_positions(self):
        """计算水平布局的D2D节点位置"""
        # 水平布局：左右摆放两个Die
        # 两个Die都使用右边界作为D2D节点，位置相同

        if not self.D2D_DIE0_POSITIONS:
            # Die 0使用右边界列
            right_col = self.NUM_COL - 1
            self.D2D_DIE0_POSITIONS = [row * self.NUM_COL + right_col for row in range(1, self.NUM_ROW, 2)]

        if not self.D2D_DIE1_POSITIONS:
            # Die 1也使用右边界列（相同位置）
            right_col = self.NUM_COL - 1
            self.D2D_DIE1_POSITIONS = [row * self.NUM_COL + right_col for row in range(1, self.NUM_ROW, 2)]

    def _calculate_vertical_positions(self):
        """计算垂直布局的D2D节点位置"""
        # 垂直布局：上下摆放两个Die
        # 两个Die都使用下边界作为D2D节点，位置相同

        if not self.D2D_DIE0_POSITIONS:
            # Die 0使用下边界行
            bottom_row = self.NUM_ROW - 1
            self.D2D_DIE0_POSITIONS = [bottom_row * self.NUM_COL + col for col in range(self.NUM_COL)]

        if not self.D2D_DIE1_POSITIONS:
            # Die 1也使用下边界行（相同位置）
            bottom_row = self.NUM_ROW - 1
            self.D2D_DIE1_POSITIONS = [bottom_row * self.NUM_COL + col for col in range(self.NUM_COL)]

    def _calculate_grid_2x2_positions(self):
        """计算2x2网格布局的D2D节点位置"""
        # 2x2网格布局：4个Die呈方形排列
        # 所有Die使用相同的内部配置，只是连接关系不同
        # 每个Die的D2D节点位置相同，连接通过D2D_CONNECTION_MAP管理
        
        # 使用标准的边缘位置作为D2D节点（类似水平/垂直布局）
        # 选择一些固定位置作为D2D接口，所有Die都使用相同位置
        
        # 为所有Die计算相同的D2D节点位置
        common_positions = []
        
        # 选择一些边缘节点作为D2D接口
        # 使用第1行和最后一行的中间列位置
        mid_col = self.NUM_COL // 2
        if mid_col > 0:
            # 上边缘
            common_positions.append(1 * self.NUM_COL + mid_col)
            # 下边缘
            common_positions.append((self.NUM_ROW - 1) * self.NUM_COL + mid_col)
        
        # 使用第1列和最后一列的中间行位置
        mid_row = self.NUM_ROW // 2
        if mid_row > 0:
            # 左边缘
            common_positions.append(mid_row * self.NUM_COL + 1)
            # 右边缘
            common_positions.append(mid_row * self.NUM_COL + (self.NUM_COL - 1))
        
        # 所有Die使用相同的D2D节点位置
        if not self.D2D_DIE0_POSITIONS:
            self.D2D_DIE0_POSITIONS = common_positions.copy()
        
        if not self.D2D_DIE1_POSITIONS:
            self.D2D_DIE1_POSITIONS = common_positions.copy()
        
        if not self.D2D_DIE2_POSITIONS:
            self.D2D_DIE2_POSITIONS = common_positions.copy()
        
        if not self.D2D_DIE3_POSITIONS:
            self.D2D_DIE3_POSITIONS = common_positions.copy()

    def _create_d2d_pairs(self):
        """创建D2D连接对关系"""
        if not self.D2D_PAIRS:
            layout = self.D2D_LAYOUT.upper()
            
            if layout in ["HORIZONTAL", "VERTICAL"]:
                # 2-Die模式：Die0和Die1之间的连接
                min_pairs = min(len(self.D2D_DIE0_POSITIONS), len(self.D2D_DIE1_POSITIONS))
                self.D2D_PAIRS = [(self.D2D_DIE0_POSITIONS[i], self.D2D_DIE1_POSITIONS[i]) for i in range(min_pairs)]
            
            elif layout == "GRID_2X2":
                # 4-Die模式：根据D2D_CONNECTION_MAP创建连接对
                self._create_4die_pairs()

    def _create_4die_pairs(self):
        """创建4-Die的D2D连接对"""
        pairs = []
        
        # 默认连接映射（如果没有配置D2D_CONNECTION_MAP）
        if not self.D2D_CONNECTION_MAP:
            self.D2D_CONNECTION_MAP = {
                0: {1: "horizontal", 2: "vertical"},
                1: {0: "horizontal", 3: "vertical"},
                2: {0: "vertical", 3: "horizontal"},
                3: {1: "vertical", 2: "horizontal"}
            }
        
        # 根据连接映射创建连接对
        processed_connections = set()  # 避免重复连接
        
        for src_die, connections in self.D2D_CONNECTION_MAP.items():
            src_die_id = int(src_die)
            src_positions = self._get_die_positions(src_die_id)
            
            for dst_die, connection_type in connections.items():
                dst_die_id = int(dst_die)
                dst_positions = self._get_die_positions(dst_die_id)
                
                # 避免重复处理同一对Die的连接
                connection_key = tuple(sorted([src_die_id, dst_die_id]))
                if connection_key in processed_connections:
                    continue
                processed_connections.add(connection_key)
                
                # 创建连接对（选择位置列表的前几个位置进行配对）
                min_positions = min(len(src_positions), len(dst_positions))
                for i in range(min_positions):
                    pairs.append((src_positions[i], dst_positions[i]))
        
        self.D2D_PAIRS = pairs

    def _get_die_positions(self, die_id: int) -> List[int]:
        """获取指定Die的D2D节点位置"""
        position_map = {
            0: self.D2D_DIE0_POSITIONS,
            1: self.D2D_DIE1_POSITIONS, 
            2: self.D2D_DIE2_POSITIONS,
            3: self.D2D_DIE3_POSITIONS
        }
        return position_map.get(die_id, [])

    def _validate_d2d_layout(self):
        """验证D2D布局的合理性"""
        # 检查基本参数
        if not getattr(self, 'D2D_ENABLED', False):
            return

        num_dies = getattr(self, 'NUM_DIES', 2)
        if num_dies < 2:
            raise ValueError(f"D2D需要至少2个Die，当前配置: {num_dies}")
        
        if num_dies > 4:
            raise ValueError(f"当前最多支持4个Die，当前配置: {num_dies}")

        # 检查布局有效性
        layout = self.D2D_LAYOUT.upper()
        if layout == "GRID_2X2" and num_dies != 4:
            raise ValueError(f"GRID_2X2布局需要4个Die，当前配置: {num_dies}")

        # 检查节点位置有效性
        max_node_id = self.NUM_NODE - 1
        die_positions_map = {
            0: self.D2D_DIE0_POSITIONS,
            1: self.D2D_DIE1_POSITIONS,
            2: self.D2D_DIE2_POSITIONS,
            3: self.D2D_DIE3_POSITIONS
        }
        
        for die_id in range(num_dies):
            positions = die_positions_map[die_id]
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
        # 检查D2D_CONNECTION_MAP
        if not self.D2D_CONNECTION_MAP:
            raise ValueError("4-Die模式需要配置D2D_CONNECTION_MAP")
        
        # 验证连接映射的完整性
        expected_dies = set(range(4))
        configured_dies = set(int(die_id) for die_id in self.D2D_CONNECTION_MAP.keys())
        
        if configured_dies != expected_dies:
            missing_dies = expected_dies - configured_dies
            raise ValueError(f"D2D_CONNECTION_MAP缺少Die配置: {missing_dies}")
        
        # 验证每个Die的连接数量（应该有2个连接）
        for die_id, connections in self.D2D_CONNECTION_MAP.items():
            if len(connections) != 2:
                raise ValueError(f"Die {die_id} 应该有2个连接，当前有{len(connections)}个")
        
        # 验证连接的对称性
        for src_die, connections in self.D2D_CONNECTION_MAP.items():
            src_die_id = int(src_die)
            for dst_die, connection_type in connections.items():
                dst_die_id = int(dst_die)
                
                # 检查反向连接是否存在
                reverse_connections = self.D2D_CONNECTION_MAP.get(str(dst_die_id), {})
                if str(src_die_id) not in reverse_connections:
                    raise ValueError(f"缺少反向连接: Die{dst_die_id} -> Die{src_die_id}")
                
                # 检查连接类型是否一致
                reverse_type = reverse_connections[str(src_die_id)]
                if reverse_type != connection_type:
                    raise ValueError(f"连接类型不匹配: Die{src_die_id}-Die{dst_die_id} ({connection_type}) vs Die{dst_die_id}-Die{src_die_id} ({reverse_type})")
        
        print(f"4-Die配置验证通过: {len(self.D2D_PAIRS)}个连接对")

    def get_d2d_pair_for_traffic(self, src_die: int, dst_die: int, traffic_type: str = "balanced") -> Tuple[int, int]:
        """
        根据流量特征选择合适的D2D连接对

        Args:
            src_die: 源Die ID
            dst_die: 目标Die ID
            traffic_type: 流量类型 ("balanced", "first", "random")

        Returns:
            (die0_position, die1_position) - D2D连接对的节点位置
        """
        if not self.D2D_PAIRS:
            raise ValueError("没有可用的D2D连接对")

        if traffic_type == "first":
            return self.D2D_PAIRS[0]
        elif traffic_type == "random":
            import random

            return random.choice(self.D2D_PAIRS)
        else:  # balanced
            # 简单的负载均衡：根据源目标Die ID选择
            pair_idx = (src_die + dst_die) % len(self.D2D_PAIRS)
            return self.D2D_PAIRS[pair_idx]

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
        print(f"布局方式: {self.D2D_LAYOUT}")
        print(f"拓扑规格: {self.NUM_ROW}x{self.NUM_COL} ({self.NUM_NODE}个节点)")
        print(f"Die数量: {self.NUM_DIES}")
        print(f"Die0 D2D节点位置: {self.D2D_DIE0_POSITIONS}")
        print(f"Die1 D2D节点位置: {self.D2D_DIE1_POSITIONS}")
        print(f"D2D连接对: {self.D2D_PAIRS}")
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
            "D2D_ENABLED": self.D2D_ENABLED,
            "NUM_DIES": self.NUM_DIES,
            "D2D_LAYOUT": self.D2D_LAYOUT,
            "D2D_DIE0_POSITIONS": self.D2D_DIE0_POSITIONS,
            "D2D_DIE1_POSITIONS": self.D2D_DIE1_POSITIONS,
            "D2D_PAIRS": self.D2D_PAIRS,
            "D2D_AR_LATENCY": self.D2D_AR_LATENCY,
            "D2D_R_LATENCY": self.D2D_R_LATENCY,
            "D2D_AW_LATENCY": self.D2D_AW_LATENCY,
            "D2D_W_LATENCY": self.D2D_W_LATENCY,
            "D2D_B_LATENCY": self.D2D_B_LATENCY,
            "D2D_DBID_LATENCY": self.D2D_DBID_LATENCY,
            "D2D_MAX_OUTSTANDING": self.D2D_MAX_OUTSTANDING,
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(d2d_config, f, indent=4, ensure_ascii=False)
