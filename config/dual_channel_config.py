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
        
        # 通道选择策略 - 实际使用的策略
        # 可选值: ip_id_based (当前代码中使用)
        self.DATA_CHANNEL_SELECT_STRATEGY = "ip_id_based"
    
    def get_channel_config_summary(self):
        """获取双通道配置摘要"""
        return {
            "enabled": self.DATA_DUAL_CHANNEL_ENABLED,
            "strategy": self.DATA_CHANNEL_SELECT_STRATEGY
        }
    
    def set_channel_selection_strategy(self, strategy):
        """设置通道选择策略"""
        valid_strategies = ["ip_id_based"]  # 目前实际支持的策略
        if strategy in valid_strategies:
            self.DATA_CHANNEL_SELECT_STRATEGY = strategy
        else:
            raise ValueError(f"Invalid strategy: {strategy}. Valid options: {valid_strategies}")
    
    def print_dual_channel_config(self):
        """打印双通道配置信息"""
        print("\n=== Dual Channel Data Configuration ===")
        print(f"Enabled: {self.DATA_DUAL_CHANNEL_ENABLED}")
        print(f"Channel Selection Strategy: {self.DATA_CHANNEL_SELECT_STRATEGY}")
        print("========================================\n")