"""
Channel selector classes for multi-channel data transmission.
Provides strategies for selecting data channels in multi-channel networks.
通道选择器返回base_channel_id，IPInterface通过模运算映射到实际通道。
"""


class ChannelSelector:
    """通道选择器，支持任意通道数"""

    def __init__(self, strategy="ip_id_based", num_channels=2):
        """
        Args:
            strategy: 选择策略 ("ip_id_based", "target_node_based", "flit_id_based")
            num_channels: 最大通道数量（用于生成base_channel_id）
        """
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
            "flit_id_based": "flit_id_based",
        }

        self.strategy = strategy
        self.num_channels = num_channels

        # 将配置中的策略名称映射到实际实现的策略
        self.actual_strategy = self.strategy_mapping.get(strategy, "ip_id_based")

    def select_channel(self, flit, ip_id=None):
        """根据配置的策略选择通道

        Returns:
            int: base_channel_id，通过模运算映射到实际通道
        """
        if self.actual_strategy == "ip_id_based":
            return self._select_by_ip_id(ip_id)
        elif self.actual_strategy == "target_node_based":
            return self._select_by_target_node(flit)
        elif self.actual_strategy == "flit_id_based":
            return self._select_by_flit_id(flit)
        else:
            return self._select_by_ip_id(ip_id)

    def _select_by_ip_id(self, ip_id):
        """基于IP的ID选择通道"""
        if ip_id is None:
            ip_id = 0
        return ip_id % self.num_channels

    def _select_by_target_node(self, flit):
        """基于目标节点ID选择通道"""
        destination = getattr(flit, "destination", 0)
        return destination % self.num_channels

    def _select_by_flit_id(self, flit):
        """基于flit_id选择通道（保证同一请求的flit在同一通道）"""
        # 使用packet_id而非flit_id，保证同一packet的所有flit在同一通道
        packet_id = getattr(flit, "packet_id", 0)
        return packet_id % self.num_channels


# 保留兼容性别名
DefaultChannelSelector = ChannelSelector


class StaticChannelSelector(ChannelSelector):
    """静态通道选择器，总是选择固定通道"""

    def __init__(self, channel_id=0, num_channels=2):
        super().__init__("ip_id_based", num_channels)
        self.channel_id = channel_id % num_channels

    def select_channel(self, flit, ip_id=None):
        """总是返回固定的通道ID"""
        return self.channel_id
