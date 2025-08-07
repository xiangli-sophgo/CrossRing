"""
Channel selector classes for dual-channel data transmission.
Provides different strategies for selecting data channels in dual-channel networks.
"""

import random
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
    """默认通道选择器，支持多种选择策略"""
    
    def __init__(self, strategy="hash_based", network=None):
        self.strategy = strategy
        self.network = network
        self.round_robin_counter = 0
        
    def select_channel(self, flit):
        """根据配置的策略选择通道"""
        if self.strategy == "hash_based":
            return self._select_by_hash(flit)
        elif self.strategy == "size_based":
            return self._select_by_size(flit)
        elif self.strategy == "type_based":
            return self._select_by_type(flit)
        elif self.strategy == "round_robin":
            return self._select_by_round_robin()
        elif self.strategy == "load_balanced":
            return self._select_by_load_balance()
        elif self.strategy == "random":
            return self._select_by_random()
        else:
            # 默认使用hash策略
            return self._select_by_hash(flit)
    
    def _select_by_hash(self, flit):
        """基于源目的地址哈希选择通道"""
        return (hash(str(flit.source) + str(flit.destination))) % 2
    
    def _select_by_size(self, flit):
        """基于包大小选择通道：小包走ch0，大包走ch1"""
        threshold = getattr(flit, 'burst_length', 4)
        return 0 if threshold <= 4 else 1
    
    def _select_by_type(self, flit):
        """基于读写类型选择通道：读数据走ch0，写数据走ch1"""
        if hasattr(flit, 'req_type') and flit.req_type:
            return 0 if flit.req_type == "read" else 1
        return 0  # 默认通道0
    
    def _select_by_round_robin(self):
        """轮询分配通道"""
        self.round_robin_counter = (self.round_robin_counter + 1) % 2
        return self.round_robin_counter
    
    def _select_by_load_balance(self):
        """基于负载均衡选择通道"""
        if self.network is None:
            # 没有网络信息时，使用轮询
            return self._select_by_round_robin()
        
        try:
            # 计算两个通道的负载
            ch0_load = self._calculate_channel_load(0)
            ch1_load = self._calculate_channel_load(1)
            
            # 选择负载较低的通道
            return 0 if ch0_load <= ch1_load else 1
        except:
            # 如果计算负载失败，使用轮询
            return self._select_by_round_robin()
    
    def _select_by_random(self):
        """随机选择通道"""
        return random.randint(0, 1)
    
    def _calculate_channel_load(self, channel_id):
        """计算指定通道的当前负载"""
        if not hasattr(self.network, 'get_inject_queues'):
            return 0
            
        total_load = 0
        try:
            inject_queues = self.network.get_inject_queues(channel_id)
            for direction in ["TL", "TR", "TU", "TD"]:
                for node_queues in inject_queues.get(direction, {}).values():
                    if hasattr(node_queues, '__len__'):
                        total_load += len(node_queues)
        except:
            pass
            
        return total_load


class StaticChannelSelector(ChannelSelector):
    """静态通道选择器，总是选择固定通道"""
    
    def __init__(self, channel_id=0):
        self.channel_id = channel_id if channel_id in [0, 1] else 0
    
    def select_channel(self, flit):
        """总是返回固定的通道ID"""
        return self.channel_id


class PriorityChannelSelector(ChannelSelector):
    """优先级通道选择器，基于flit的ETag优先级选择通道"""
    
    def __init__(self, high_priority_channel=0):
        self.high_priority_channel = high_priority_channel
        self.low_priority_channel = 1 - high_priority_channel
    
    def select_channel(self, flit):
        """基于ETag优先级选择通道"""
        if hasattr(flit, 'ETag_priority'):
            if flit.ETag_priority == "T0":  # 最高优先级
                return self.high_priority_channel
            elif flit.ETag_priority == "T1":  # 中等优先级
                return self.high_priority_channel
            else:  # T2或其他低优先级
                return self.low_priority_channel
        
        # 没有优先级信息时默认使用低优先级通道
        return self.low_priority_channel


class AdaptiveChannelSelector(ChannelSelector):
    """自适应通道选择器，根据网络状况动态调整选择策略"""
    
    def __init__(self, network=None):
        self.network = network
        self.base_selector = DefaultChannelSelector("hash_based", network)
        self.load_selector = DefaultChannelSelector("load_balanced", network)
        self.congestion_threshold = 0.8
        
    def select_channel(self, flit):
        """根据网络拥塞情况自适应选择通道"""
        try:
            congestion_level = self._calculate_network_congestion()
            
            if congestion_level > self.congestion_threshold:
                # 高拥塞时使用负载均衡
                return self.load_selector.select_channel(flit)
            else:
                # 低拥塞时使用哈希分配
                return self.base_selector.select_channel(flit)
        except:
            # 异常情况下使用基础选择器
            return self.base_selector.select_channel(flit)
    
    def _calculate_network_congestion(self):
        """计算网络拥塞程度 (0-1)"""
        if not self.network:
            return 0.0
            
        try:
            # 简单的拥塞计算：基于注入队列的平均占用率
            total_capacity = 0
            total_usage = 0
            
            for channel_id in [0, 1]:
                inject_queues = self.network.get_inject_queues(channel_id)
                for direction in ["TL", "TR", "TU", "TD"]:
                    for node_queues in inject_queues.get(direction, {}).values():
                        if hasattr(node_queues, '__len__') and hasattr(node_queues, 'maxlen'):
                            total_usage += len(node_queues)
                            total_capacity += node_queues.maxlen or 10  # 默认容量
            
            return total_usage / total_capacity if total_capacity > 0 else 0.0
        except:
            return 0.0


class SourceDestinationChannelSelector(ChannelSelector):
    """基于源目的节点位置的通道选择器"""
    
    def __init__(self, config=None):
        self.config = config
    
    def select_channel(self, flit):
        """基于源目的节点的位置关系选择通道"""
        try:
            source = getattr(flit, 'source', 0)
            destination = getattr(flit, 'destination', 0)
            
            # 基于源节点和目标节点的相对位置选择通道
            if self.config and hasattr(self.config, 'NUM_COL'):
                # 同行的数据包使用通道0，跨行的使用通道1
                source_row = source // self.config.NUM_COL
                dest_row = destination // self.config.NUM_COL
                return 0 if source_row == dest_row else 1
            else:
                # 没有配置信息时使用简单的奇偶分配
                return (source + destination) % 2
        except:
            return 0