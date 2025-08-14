# d2d_config.py
"""
D2D (Die-to-Die) Configuration Manager

This module provides D2DConfig class that extends CrossRingConfig with D2D-specific
configuration parameters and automatic node placement calculation.
"""

import json
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
    
    def __init__(self, base_config_file=None, d2d_layout="horizontal", d2d_config_file=None):
        """
        初始化D2D配置
        
        Args:
            base_config_file: 基础配置文件路径
            d2d_layout: D2D布局方式 ("horizontal" 或 "vertical")
            d2d_config_file: D2D专用配置文件路径
        """
        # 初始化基础配置
        super().__init__(base_config_file)
        
        # D2D特定参数
        self.D2D_LAYOUT = d2d_layout
        self.D2D_RN_POSITIONS = []
        self.D2D_SN_POSITIONS = []
        self.D2D_RN_SN_PAIRS = []
        
        # 从原配置文件中提取D2D参数，如果存在的话
        self._load_d2d_base_config()
        
        # 如果指定了D2D配置文件，则加载D2D特定配置
        if d2d_config_file:
            self._load_d2d_config_file(d2d_config_file)
        
        # 根据拓扑和布局计算节点位置
        self._calculate_d2d_positions()
        
        # 创建RN-SN配对关系
        self._create_rn_sn_pairs()
        
        # 验证配置
        self._validate_d2d_layout()
        
        logging.info(f"D2D配置初始化完成: {self.D2D_LAYOUT}布局, {len(self.D2D_RN_SN_PAIRS)}对RN-SN节点")
    
    def _load_d2d_base_config(self):
        """从基础配置中加载D2D相关参数"""
        # D2D基础参数
        self.D2D_ENABLED = getattr(self, 'D2D_ENABLED', True)
        self.NUM_DIES = getattr(self, 'NUM_DIES', 2)
        
        # D2D延迟参数
        self.D2D_AR_LATENCY = getattr(self, 'D2D_AR_LATENCY', 10)
        self.D2D_R_LATENCY = getattr(self, 'D2D_R_LATENCY', 8)
        self.D2D_AW_LATENCY = getattr(self, 'D2D_AW_LATENCY', 10)
        self.D2D_W_LATENCY = getattr(self, 'D2D_W_LATENCY', 2)
        self.D2D_B_LATENCY = getattr(self, 'D2D_B_LATENCY', 8)
        self.D2D_DBID_LATENCY = getattr(self, 'D2D_DBID_LATENCY', 5)
        self.D2D_MAX_OUTSTANDING = getattr(self, 'D2D_MAX_OUTSTANDING', 16)
        
        # 兼容旧的单节点位置配置
        if hasattr(self, 'D2D_RN_POSITION'):
            self.D2D_RN_POSITIONS = [self.D2D_RN_POSITION]
        if hasattr(self, 'D2D_SN_POSITION'):
            self.D2D_SN_POSITIONS = [self.D2D_SN_POSITION]
    
    def _load_d2d_config_file(self, d2d_config_file):
        """加载D2D专用配置文件"""
        try:
            with open(d2d_config_file, 'r', encoding='utf-8') as f:
                d2d_config = json.load(f)
            
            # 更新D2D特定参数
            for key, value in d2d_config.items():
                if key.startswith('D2D_'):
                    setattr(self, key, value)
            
            logging.info(f"已加载D2D配置文件: {d2d_config_file}")
        except FileNotFoundError:
            logging.warning(f"D2D配置文件未找到: {d2d_config_file}")
        except json.JSONDecodeError as e:
            logging.error(f"D2D配置文件格式错误: {e}")
    
    def _calculate_d2d_positions(self):
        """根据拓扑类型和布局方式计算D2D节点位置"""
        if not self.D2D_RN_POSITIONS or not self.D2D_SN_POSITIONS:
            if self.D2D_LAYOUT == "horizontal":
                self._calculate_horizontal_positions()
            elif self.D2D_LAYOUT == "vertical":
                self._calculate_vertical_positions()
            else:
                raise ValueError(f"不支持的D2D布局方式: {self.D2D_LAYOUT}")
    
    def _calculate_horizontal_positions(self):
        """计算水平布局的D2D节点位置"""
        # 水平布局：左右摆放
        # Die 0: 最右列放置D2D节点
        # Die 1: 最左列放置D2D节点
        
        if not self.D2D_RN_POSITIONS:
            # Die 0的右边界列 (最后一列)
            die0_right_col = self.NUM_COL - 1
            self.D2D_RN_POSITIONS = [
                row * self.NUM_COL + die0_right_col 
                for row in range(self.NUM_ROW)
            ]
        
        if not self.D2D_SN_POSITIONS:
            # Die 1的左边界列 (第一列)
            die1_left_col = 0
            self.D2D_SN_POSITIONS = [
                row * self.NUM_COL + die1_left_col 
                for row in range(self.NUM_ROW)
            ]
        
        logging.info(f"水平布局计算完成: RN节点 {self.D2D_RN_POSITIONS}, SN节点 {self.D2D_SN_POSITIONS}")
    
    def _calculate_vertical_positions(self):
        """计算垂直布局的D2D节点位置"""
        # 垂直布局：上下摆放
        # Die 0: 最下行放置D2D节点
        # Die 1: 最上行放置D2D节点
        
        if not self.D2D_RN_POSITIONS:
            # Die 0的下边界行 (最后一行)
            die0_bottom_row = self.NUM_ROW - 1
            self.D2D_RN_POSITIONS = [
                die0_bottom_row * self.NUM_COL + col 
                for col in range(self.NUM_COL)
            ]
        
        if not self.D2D_SN_POSITIONS:
            # Die 1的上边界行 (第一行)
            die1_top_row = 0
            self.D2D_SN_POSITIONS = [
                die1_top_row * self.NUM_COL + col 
                for col in range(self.NUM_COL)
            ]
        
        logging.info(f"垂直布局计算完成: RN节点 {self.D2D_RN_POSITIONS}, SN节点 {self.D2D_SN_POSITIONS}")
    
    def _create_rn_sn_pairs(self):
        """创建RN-SN配对关系"""
        if not self.D2D_RN_SN_PAIRS:
            # 确保RN和SN节点数量一致
            min_pairs = min(len(self.D2D_RN_POSITIONS), len(self.D2D_SN_POSITIONS))
            self.D2D_RN_SN_PAIRS = [
                (self.D2D_RN_POSITIONS[i], self.D2D_SN_POSITIONS[i])
                for i in range(min_pairs)
            ]
        
        logging.info(f"创建了 {len(self.D2D_RN_SN_PAIRS)} 对RN-SN配对")
    
    def _validate_d2d_layout(self):
        """验证D2D布局的合理性"""
        # 检查基本参数
        if not self.D2D_ENABLED:
            logging.warning("D2D功能未启用")
            return
        
        if self.NUM_DIES < 2:
            raise ValueError(f"D2D需要至少2个Die，当前配置: {self.NUM_DIES}")
        
        # 检查节点位置有效性
        max_node_id = self.NUM_NODE - 1
        for pos in self.D2D_RN_POSITIONS:
            if pos < 0 or pos > max_node_id:
                raise ValueError(f"RN节点位置无效: {pos}, 有效范围: 0-{max_node_id}")
        
        for pos in self.D2D_SN_POSITIONS:
            if pos < 0 or pos > max_node_id:
                raise ValueError(f"SN节点位置无效: {pos}, 有效范围: 0-{max_node_id}")
        
        # 检查配对数量
        if len(self.D2D_RN_SN_PAIRS) == 0:
            raise ValueError("没有有效的RN-SN配对")
        
        logging.info("D2D布局验证通过")
    
    def get_rn_sn_pair_for_traffic(self, src_die: int, dst_die: int, traffic_type: str = "balanced") -> Tuple[int, int]:
        """
        根据流量特征选择合适的RN-SN配对
        
        Args:
            src_die: 源Die ID
            dst_die: 目标Die ID  
            traffic_type: 流量类型 ("balanced", "first", "random")
            
        Returns:
            (rn_position, sn_position)
        """
        if not self.D2D_RN_SN_PAIRS:
            raise ValueError("没有可用的RN-SN配对")
        
        if traffic_type == "first":
            return self.D2D_RN_SN_PAIRS[0]
        elif traffic_type == "random":
            import random
            return random.choice(self.D2D_RN_SN_PAIRS)
        else:  # balanced
            # 简单的负载均衡：根据源目标Die ID选择
            pair_idx = (src_die + dst_die) % len(self.D2D_RN_SN_PAIRS)
            return self.D2D_RN_SN_PAIRS[pair_idx]
    
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
        print(f"RN节点位置: {self.D2D_RN_POSITIONS}")
        print(f"SN节点位置: {self.D2D_SN_POSITIONS}")
        print(f"RN-SN配对: {self.D2D_RN_SN_PAIRS}")
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
            req_type = random.choice(['R', 'W'])
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
            "D2D_RN_POSITIONS": self.D2D_RN_POSITIONS,
            "D2D_SN_POSITIONS": self.D2D_SN_POSITIONS,
            "D2D_RN_SN_PAIRS": self.D2D_RN_SN_PAIRS,
            "D2D_AR_LATENCY": self.D2D_AR_LATENCY,
            "D2D_R_LATENCY": self.D2D_R_LATENCY,
            "D2D_AW_LATENCY": self.D2D_AW_LATENCY,
            "D2D_W_LATENCY": self.D2D_W_LATENCY,
            "D2D_B_LATENCY": self.D2D_B_LATENCY,
            "D2D_DBID_LATENCY": self.D2D_DBID_LATENCY,
            "D2D_MAX_OUTSTANDING": self.D2D_MAX_OUTSTANDING
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(d2d_config, f, indent=4, ensure_ascii=False)
        
        logging.info(f"D2D配置已保存到: {config_file}")