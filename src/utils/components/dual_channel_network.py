"""
DualChannelDataNetwork class for NoC simulation.
Extends Network class to support dual-channel data transmission.
"""

from .network import Network
from collections import defaultdict, deque
import logging


class DualChannelDataNetwork(Network):
    """专门用于数据传输的双通道网络"""
    
    def __init__(self, config, adjacency_matrix, name="dual_channel_data_network"):
        super().__init__(config, adjacency_matrix, name)
        self._init_dual_channel_structures()
    
    def _init_dual_channel_structures(self):
        """初始化双通道数据结构"""
        # 双通道IQ缓冲区 - 使用ch0/ch1后缀
        self.IQ_channel_buffer_ch0 = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.IQ_channel_buffer_ch1 = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.IQ_channel_buffer_pre_ch0 = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.IQ_channel_buffer_pre_ch1 = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        
        # 双通道注入队列 - 四个方向都支持双通道
        self.inject_queues_ch0 = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.inject_queues_ch1 = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.inject_queues_pre_ch0 = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.inject_queues_pre_ch1 = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        
        # 双通道Ring Bridge
        self.ring_bridge_ch0 = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.ring_bridge_ch1 = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.ring_bridge_pre_ch0 = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.ring_bridge_pre_ch1 = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        
        # 双通道弹出队列
        self.eject_queues_ch0 = {"TU": {}, "TD": {}}
        self.eject_queues_ch1 = {"TU": {}, "TD": {}}
        self.eject_queues_in_pre_ch0 = {"TU": {}, "TD": {}}
        self.eject_queues_in_pre_ch1 = {"TU": {}, "TD": {}}
        
        # 双通道EQ缓冲区
        self.EQ_channel_buffer_ch0 = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.EQ_channel_buffer_ch1 = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.EQ_channel_buffer_pre_ch0 = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.EQ_channel_buffer_pre_ch1 = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        
        # 双通道round_robin仲裁计数器
        self.round_robin_ch0 = {"IQ": defaultdict(lambda: defaultdict(dict)), 
                                "RB": defaultdict(lambda: defaultdict(dict)), 
                                "EQ": defaultdict(lambda: defaultdict(dict))}
        self.round_robin_ch1 = {"IQ": defaultdict(lambda: defaultdict(dict)), 
                                "RB": defaultdict(lambda: defaultdict(dict)), 
                                "EQ": defaultdict(lambda: defaultdict(dict))}
        
        # 双通道统计信息
        self.dual_channel_stats = {
            "ch0": {
                "inject_count": 0,
                "eject_count": 0, 
                "latency": [],
                "utilization": [],
                "throughput": []
            },
            "ch1": {
                "inject_count": 0,
                "eject_count": 0,
                "latency": [],
                "utilization": [],
                "throughput": []
            }
        }
        
        # 双通道flit计数
        self.circuits_flit_ch0 = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.circuits_flit_ch1 = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        
        logging.info(f"DualChannelDataNetwork initialized with dual-channel support")
    
    def get_iq_channel_buffer(self, channel_id):
        """获取指定通道的IQ缓冲区"""
        if channel_id == 0:
            return self.IQ_channel_buffer_ch0
        elif channel_id == 1:
            return self.IQ_channel_buffer_ch1
        else:
            raise ValueError(f"Invalid channel_id: {channel_id}. Must be 0 or 1.")
    
    def get_iq_channel_buffer_pre(self, channel_id):
        """获取指定通道的IQ预缓冲区"""
        if channel_id == 0:
            return self.IQ_channel_buffer_pre_ch0
        elif channel_id == 1:
            return self.IQ_channel_buffer_pre_ch1
        else:
            raise ValueError(f"Invalid channel_id: {channel_id}. Must be 0 or 1.")
    
    def get_inject_queues(self, channel_id):
        """获取指定通道的注入队列"""
        if channel_id == 0:
            return self.inject_queues_ch0
        elif channel_id == 1:
            return self.inject_queues_ch1
        else:
            raise ValueError(f"Invalid channel_id: {channel_id}. Must be 0 or 1.")
    
    def get_ring_bridge(self, channel_id):
        """获取指定通道的Ring Bridge"""
        if channel_id == 0:
            return self.ring_bridge_ch0
        elif channel_id == 1:
            return self.ring_bridge_ch1
        else:
            raise ValueError(f"Invalid channel_id: {channel_id}. Must be 0 or 1.")
    
    def get_eject_queues(self, channel_id):
        """获取指定通道的弹出队列"""
        if channel_id == 0:
            return self.eject_queues_ch0
        elif channel_id == 1:
            return self.eject_queues_ch1
        else:
            raise ValueError(f"Invalid channel_id: {channel_id}. Must be 0 or 1.")
    
    def get_eq_channel_buffer(self, channel_id):
        """获取指定通道的EQ缓冲区"""
        if channel_id == 0:
            return self.EQ_channel_buffer_ch0
        elif channel_id == 1:
            return self.EQ_channel_buffer_ch1
        else:
            raise ValueError(f"Invalid channel_id: {channel_id}. Must be 0 or 1.")
    
    def get_eq_channel_buffer_pre(self, channel_id):
        """获取指定通道的EQ预缓冲区"""
        if channel_id == 0:
            return self.EQ_channel_buffer_pre_ch0
        elif channel_id == 1:
            return self.EQ_channel_buffer_pre_ch1
        else:
            raise ValueError(f"Invalid channel_id: {channel_id}. Must be 0 or 1.")
    
    def get_channel_stats(self):
        """获取双通道统计信息"""
        return self.dual_channel_stats
    
    def update_channel_stats(self, channel_id, stat_type, value):
        """更新通道统计信息"""
        if channel_id in [0, 1] and stat_type in self.dual_channel_stats[f"ch{channel_id}"]:
            if stat_type in ["latency", "utilization", "throughput"]:
                self.dual_channel_stats[f"ch{channel_id}"][stat_type].append(value)
            else:
                self.dual_channel_stats[f"ch{channel_id}"][stat_type] += value
    
    def reset_channel_stats(self):
        """重置双通道统计信息"""
        for channel in ["ch0", "ch1"]:
            self.dual_channel_stats[channel] = {
                "inject_count": 0,
                "eject_count": 0,
                "latency": [],
                "utilization": [],
                "throughput": []
            }