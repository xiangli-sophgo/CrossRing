"""
Channel selector classes for dual-channel data transmission.
Provides three simple strategies for selecting data channels in dual-channel networks.
"""

from abc import ABC, abstractmethod


class ChannelSelector(ABC):
    """抽象基类，定义通道选择接口"""
    
    @abstractmethod
    def select_channel(self, flit):
        """选择数据通道
        
        Args:
            flit: 需要选择通道的flit对象
            
        Returns:
            int: 选择的通道ID (0 or 1)
        """
        pass


class DefaultChannelSelector(ChannelSelector):
    """默认通道选择器，支持从配置文件中读取的策略名称"""
    
    def __init__(self, strategy="ip_id_based", network=None, ip_id=None):
        # 映射配置文件中的策略名称到实际实现的策略
        self.strategy_mapping = {
            "hash_based": "ip_id_based",
            "size_based": "flit_id_based",
            "type_based": "target_node_based",
            "round_robin": "ip_id_based",
            "load_balanced": "ip_id_based",
            "random": "ip_id_based",
            "ip_id_based": "ip_id_based",
            "target_node_based": "target_node_based",
            "flit_id_based": "flit_id_based"
        }
        
        self.strategy = strategy
        self.network = network
        self.ip_id = ip_id if ip_id is not None else 0
        
        # 将配置中的策略名称映射到实际实现的策略
        self.actual_strategy = self.strategy_mapping.get(strategy, "ip_id_based")
        
    def select_channel(self, flit):
        """根据配置的策略选择通道"""
        if self.actual_strategy == "ip_id_based":
            return self._select_by_ip_id()
        elif self.actual_strategy == "target_node_based":
            return self._select_by_target_node(flit)
        elif self.actual_strategy == "flit_id_based":
            return self._select_by_flit_id(flit)
        else:
            # 默认使用ip_id策略
            return self._select_by_ip_id()
    
    def _select_by_ip_id(self):
        """基于IP的ID奇偶选择通道"""
        return (self.ip_id + 1) % 2
    
    def _select_by_target_node(self, flit):
        """基于目标节点ID奇偶选择通道"""
        destination = getattr(flit, 'destination', 0)
        return destination % 2
    
    def _select_by_flit_id(self, flit):
        """基于flit_id奇偶选择通道"""
        flit_id = getattr(flit, 'flit_id', 0)
        return flit_id % 2


class StaticChannelSelector(ChannelSelector):
    """静态通道选择器，总是选择固定通道"""
    
    def __init__(self, channel_id=0):
        self.channel_id = channel_id if channel_id in [0, 1] else 0
    
    def select_channel(self, flit):
        """总是返回固定的通道ID"""
        return self.channel_id