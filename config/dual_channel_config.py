"""
DualChannelConfig class for NoC simulation.
Extends CrossRingConfig class to support dual-channel data configuration.
"""

from .config import CrossRingConfig
from pathlib import Path
import json


class DualChannelConfig(CrossRingConfig):
    """双通道配置类，扩展基础配置以支持双通道数据传输"""
    
    def __init__(self, default_config=None):
        # 调用父类初始化
        super().__init__(default_config)
        
        # 初始化双通道相关参数
        self._init_dual_channel_params()
    
    def _init_dual_channel_params(self):
        """初始化双通道相关参数"""
        # 双通道启用开关
        self.DATA_DUAL_CHANNEL_ENABLED = True
        
        # 通道选择策略
        # 可选值: hash_based, size_based, type_based, round_robin, load_balanced, random
        self.DATA_CHANNEL_SELECT_STRATEGY = "hash_based"
        
        # 通道优先级设置
        self.DATA_CHANNEL_0_PRIORITY = "normal"  # normal, high, low
        self.DATA_CHANNEL_1_PRIORITY = "normal"
        
        # 双通道缓冲区深度配置
        self.DATA_CH0_IQ_FIFO_DEPTH = getattr(self, 'IQ_CH_FIFO_DEPTH', 8)
        self.DATA_CH1_IQ_FIFO_DEPTH = getattr(self, 'IQ_CH_FIFO_DEPTH', 8)
        self.DATA_CH0_EQ_FIFO_DEPTH = getattr(self, 'EQ_CH_FIFO_DEPTH', 8)
        self.DATA_CH1_EQ_FIFO_DEPTH = getattr(self, 'EQ_CH_FIFO_DEPTH', 8)
        
        # 双通道带宽分配配置
        self.DATA_CH0_BANDWIDTH_RATIO = 0.5  # 通道0的带宽比例
        self.DATA_CH1_BANDWIDTH_RATIO = 0.5  # 通道1的带宽比例
        
        # 双通道统计配置
        self.ENABLE_DUAL_CHANNEL_STATS = True
        self.DUAL_CHANNEL_STATS_INTERVAL = 1000  # 统计间隔周期数
        
        # 双通道调度参数
        self.DATA_CH0_WEIGHT = 1.0  # 通道0在负载均衡中的权重
        self.DATA_CH1_WEIGHT = 1.0  # 通道1在负载均衡中的权重
        
        # 双通道拥塞控制
        self.DUAL_CHANNEL_CONGESTION_THRESHOLD = 0.8  # 拥塞阈值
        self.ENABLE_ADAPTIVE_CHANNEL_SELECTION = True  # 自适应通道选择
        
        # 双通道专用的CHANNEL_SPEC
        self.DATA_DUAL_CHANNEL_SPEC = {
            "data_ch0": {
                "sdma": getattr(self.CHANNEL_SPEC, 'sdma', 2),
                "gdma": getattr(self.CHANNEL_SPEC, 'gdma', 2), 
                "cdma": getattr(self.CHANNEL_SPEC, 'cdma', 2),
                "ddr": getattr(self.CHANNEL_SPEC, 'ddr', 2),
                "l2m": getattr(self.CHANNEL_SPEC, 'l2m', 2)
            },
            "data_ch1": {
                "sdma": getattr(self.CHANNEL_SPEC, 'sdma', 2),
                "gdma": getattr(self.CHANNEL_SPEC, 'gdma', 2),
                "cdma": getattr(self.CHANNEL_SPEC, 'cdma', 2),
                "ddr": getattr(self.CHANNEL_SPEC, 'ddr', 2),
                "l2m": getattr(self.CHANNEL_SPEC, 'l2m', 2)
            }
        }
        
        # 更新通道名称列表以包含双通道
        self._update_dual_channel_names()
    
    def _update_dual_channel_names(self):
        """更新双通道名称列表"""
        self.DUAL_CH_NAME_LIST = []
        for ch in ["ch0", "ch1"]:
            for key in self.CHANNEL_SPEC:
                for idx in range(self.CHANNEL_SPEC[key]):
                    self.DUAL_CH_NAME_LIST.append(f"data_{ch}_{key}_{idx}")
    
    def set_channel_selection_strategy(self, strategy):
        """设置通道选择策略"""
        valid_strategies = [
            "hash_based", "size_based", "type_based", 
            "round_robin", "load_balanced", "random"
        ]
        if strategy in valid_strategies:
            self.DATA_CHANNEL_SELECT_STRATEGY = strategy
        else:
            raise ValueError(f"Invalid strategy: {strategy}. Valid options: {valid_strategies}")
    
    def set_channel_bandwidth_ratio(self, ch0_ratio, ch1_ratio=None):
        """设置通道带宽分配比例"""
        if ch1_ratio is None:
            ch1_ratio = 1.0 - ch0_ratio
            
        if ch0_ratio + ch1_ratio != 1.0:
            # 自动归一化
            total = ch0_ratio + ch1_ratio
            ch0_ratio /= total
            ch1_ratio /= total
            
        self.DATA_CH0_BANDWIDTH_RATIO = ch0_ratio
        self.DATA_CH1_BANDWIDTH_RATIO = ch1_ratio
    
    def set_channel_priorities(self, ch0_priority, ch1_priority):
        """设置通道优先级"""
        valid_priorities = ["high", "normal", "low"]
        
        if ch0_priority not in valid_priorities:
            raise ValueError(f"Invalid ch0_priority: {ch0_priority}. Valid options: {valid_priorities}")
        if ch1_priority not in valid_priorities:
            raise ValueError(f"Invalid ch1_priority: {ch1_priority}. Valid options: {valid_priorities}")
            
        self.DATA_CHANNEL_0_PRIORITY = ch0_priority
        self.DATA_CHANNEL_1_PRIORITY = ch1_priority
    
    def set_channel_fifo_depths(self, ch0_iq_depth=None, ch1_iq_depth=None, 
                               ch0_eq_depth=None, ch1_eq_depth=None):
        """设置通道FIFO深度"""
        if ch0_iq_depth is not None:
            self.DATA_CH0_IQ_FIFO_DEPTH = ch0_iq_depth
        if ch1_iq_depth is not None:
            self.DATA_CH1_IQ_FIFO_DEPTH = ch1_iq_depth
        if ch0_eq_depth is not None:
            self.DATA_CH0_EQ_FIFO_DEPTH = ch0_eq_depth
        if ch1_eq_depth is not None:
            self.DATA_CH1_EQ_FIFO_DEPTH = ch1_eq_depth
    
    def get_channel_config_summary(self):
        """获取双通道配置摘要"""
        return {
            "enabled": self.DATA_DUAL_CHANNEL_ENABLED,
            "strategy": self.DATA_CHANNEL_SELECT_STRATEGY,
            "ch0_priority": self.DATA_CHANNEL_0_PRIORITY,
            "ch1_priority": self.DATA_CHANNEL_1_PRIORITY,
            "ch0_bandwidth_ratio": self.DATA_CH0_BANDWIDTH_RATIO,
            "ch1_bandwidth_ratio": self.DATA_CH1_BANDWIDTH_RATIO,
            "ch0_iq_depth": self.DATA_CH0_IQ_FIFO_DEPTH,
            "ch1_iq_depth": self.DATA_CH1_IQ_FIFO_DEPTH,
            "adaptive_selection": self.ENABLE_ADAPTIVE_CHANNEL_SELECTION,
            "congestion_threshold": self.DUAL_CHANNEL_CONGESTION_THRESHOLD
        }
    
    def save_dual_channel_config(self, config_path):
        """保存双通道配置到JSON文件"""
        config_data = {
            "dual_channel": {
                "enabled": self.DATA_DUAL_CHANNEL_ENABLED,
                "strategy": self.DATA_CHANNEL_SELECT_STRATEGY,
                "priorities": {
                    "ch0": self.DATA_CHANNEL_0_PRIORITY,
                    "ch1": self.DATA_CHANNEL_1_PRIORITY
                },
                "bandwidth_ratios": {
                    "ch0": self.DATA_CH0_BANDWIDTH_RATIO,
                    "ch1": self.DATA_CH1_BANDWIDTH_RATIO
                },
                "fifo_depths": {
                    "ch0_iq": self.DATA_CH0_IQ_FIFO_DEPTH,
                    "ch1_iq": self.DATA_CH1_IQ_FIFO_DEPTH,
                    "ch0_eq": self.DATA_CH0_EQ_FIFO_DEPTH,
                    "ch1_eq": self.DATA_CH1_EQ_FIFO_DEPTH
                },
                "adaptive_selection": self.ENABLE_ADAPTIVE_CHANNEL_SELECTION,
                "congestion_threshold": self.DUAL_CHANNEL_CONGESTION_THRESHOLD,
                "stats_enabled": self.ENABLE_DUAL_CHANNEL_STATS,
                "stats_interval": self.DUAL_CHANNEL_STATS_INTERVAL
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_dual_channel_config(self, config_path):
        """从JSON文件加载双通道配置"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            if 'dual_channel' in config_data:
                dc_config = config_data['dual_channel']
                
                self.DATA_DUAL_CHANNEL_ENABLED = dc_config.get('enabled', True)
                self.DATA_CHANNEL_SELECT_STRATEGY = dc_config.get('strategy', 'hash_based')
                
                if 'priorities' in dc_config:
                    self.DATA_CHANNEL_0_PRIORITY = dc_config['priorities'].get('ch0', 'normal')
                    self.DATA_CHANNEL_1_PRIORITY = dc_config['priorities'].get('ch1', 'normal')
                
                if 'bandwidth_ratios' in dc_config:
                    self.DATA_CH0_BANDWIDTH_RATIO = dc_config['bandwidth_ratios'].get('ch0', 0.5)
                    self.DATA_CH1_BANDWIDTH_RATIO = dc_config['bandwidth_ratios'].get('ch1', 0.5)
                
                if 'fifo_depths' in dc_config:
                    self.DATA_CH0_IQ_FIFO_DEPTH = dc_config['fifo_depths'].get('ch0_iq', 8)
                    self.DATA_CH1_IQ_FIFO_DEPTH = dc_config['fifo_depths'].get('ch1_iq', 8)
                    self.DATA_CH0_EQ_FIFO_DEPTH = dc_config['fifo_depths'].get('ch0_eq', 8)
                    self.DATA_CH1_EQ_FIFO_DEPTH = dc_config['fifo_depths'].get('ch1_eq', 8)
                
                self.ENABLE_ADAPTIVE_CHANNEL_SELECTION = dc_config.get('adaptive_selection', True)
                self.DUAL_CHANNEL_CONGESTION_THRESHOLD = dc_config.get('congestion_threshold', 0.8)
                self.ENABLE_DUAL_CHANNEL_STATS = dc_config.get('stats_enabled', True)
                self.DUAL_CHANNEL_STATS_INTERVAL = dc_config.get('stats_interval', 1000)
                
        except Exception as e:
            print(f"Warning: Failed to load dual channel config from {config_path}: {e}")
            print("Using default dual channel configuration.")
    
    def print_dual_channel_config(self):
        """打印双通道配置信息"""
        print("\n=== Dual Channel Data Configuration ===")
        print(f"Enabled: {self.DATA_DUAL_CHANNEL_ENABLED}")
        print(f"Channel Selection Strategy: {self.DATA_CHANNEL_SELECT_STRATEGY}")
        print(f"Channel 0 Priority: {self.DATA_CHANNEL_0_PRIORITY}")
        print(f"Channel 1 Priority: {self.DATA_CHANNEL_1_PRIORITY}")
        print(f"Bandwidth Ratio - CH0: {self.DATA_CH0_BANDWIDTH_RATIO:.1f}, CH1: {self.DATA_CH1_BANDWIDTH_RATIO:.1f}")
        print(f"FIFO Depths - CH0 IQ: {self.DATA_CH0_IQ_FIFO_DEPTH}, CH1 IQ: {self.DATA_CH1_IQ_FIFO_DEPTH}")
        print(f"Adaptive Selection: {self.ENABLE_ADAPTIVE_CHANNEL_SELECTION}")
        print(f"Congestion Threshold: {self.DUAL_CHANNEL_CONGESTION_THRESHOLD}")
        print("========================================\n")


# 便捷的预设配置函数
def create_balanced_dual_channel_config(base_config_path=None):
    """创建均衡的双通道配置"""
    config = DualChannelConfig(base_config_path)
    config.set_channel_selection_strategy("load_balanced")
    config.set_channel_bandwidth_ratio(0.5, 0.5)
    config.set_channel_priorities("normal", "normal")
    return config


def create_read_write_separated_config(base_config_path=None):
    """创建读写分离的双通道配置"""
    config = DualChannelConfig(base_config_path)
    config.set_channel_selection_strategy("type_based")
    config.set_channel_priorities("high", "normal")  # 读通道高优先级
    config.set_channel_bandwidth_ratio(0.6, 0.4)     # 读通道更多带宽
    return config


def create_size_based_dual_channel_config(base_config_path=None):
    """创建基于包大小的双通道配置"""
    config = DualChannelConfig(base_config_path)
    config.set_channel_selection_strategy("size_based")
    config.set_channel_priorities("high", "normal")  # 小包通道高优先级
    config.set_channel_bandwidth_ratio(0.4, 0.6)     # 大包通道更多带宽
    return config