"""
Channel selector classes for multi-channel data transmission.
Provides strategies for selecting data channels in multi-channel networks.
通道选择器返回base_channel_id，IPInterface通过模运算映射到实际通道。

支持的策略:
- coord_based: 基于节点坐标(XID/YID)选择通道
- node_id_based: 基于节点ID选择通道
- burst_interleave: 同一packet的不同flit分配到不同通道
- ip_type_id_based: 基于IP类型内的ID选择通道
- round_robin: 轮询选择通道（支持packet/flit/request三种粒度）
"""


class ChannelSelector:
    """通道选择器，支持任意通道数"""

    def __init__(self, strategy="ip_type_id_based", num_channels=2, num_cols=None, params=None):
        """
        Args:
            strategy: 选择策略
            num_channels: 最大通道数量（用于生成base_channel_id）
            num_cols: 网格列数（coord_based策略需要）
            params: 策略参数字典
                - basis: "source" | "destination" (用于coord_based, node_id_based)
                - dimension: "x" | "y" (用于coord_based)
                - granularity: "packet" | "flit" | "request" (用于round_robin)
        """
        self.strategy = strategy
        self.num_channels = num_channels
        self.num_cols = num_cols
        self.params = params or {}

        # 策略参数
        self.basis = self.params.get("basis", "destination")
        self.dimension = self.params.get("dimension", "x")
        self.granularity = self.params.get("granularity", "packet")

        # 轮询状态
        self.counter = 0
        self.seen_packets = set()

        # 支持的策略列表
        self.supported_strategies = {
            "coord_based",
            "node_id_based",
            "burst_interleave",
            "ip_type_id_based",
            "round_robin",
        }

        if strategy not in self.supported_strategies:
            raise ValueError(f"不支持的策略: {strategy}，支持的策略: {self.supported_strategies}")

    def select_channel(self, flit, ip_id=None):
        """根据配置的策略选择通道

        Returns:
            int: base_channel_id，通过模运算映射到实际通道
        """
        if self.strategy == "coord_based":
            return self._select_by_coord(flit)
        elif self.strategy == "node_id_based":
            return self._select_by_node_id(flit)
        elif self.strategy == "burst_interleave":
            return self._select_by_burst_interleave(flit)
        elif self.strategy == "ip_type_id_based":
            return self._select_by_ip_type_id(ip_id)
        elif self.strategy == "round_robin":
            return self._select_by_round_robin(flit)

    def _get_node_id(self, flit):
        """根据basis参数获取源或目标节点ID"""
        if self.basis == "source":
            return getattr(flit, "source", 0)
        else:
            return getattr(flit, "destination", 0)

    def _select_by_coord(self, flit):
        """基于节点坐标(XID/YID)选择通道"""
        if self.num_cols is None:
            raise ValueError("coord_based策略需要num_cols参数")
        node_id = self._get_node_id(flit)
        if self.dimension == "x":
            coord = node_id % self.num_cols  # XID = col
        else:
            coord = node_id // self.num_cols  # YID = row
        return coord % self.num_channels

    def _select_by_node_id(self, flit):
        """基于节点ID选择通道"""
        node_id = self._get_node_id(flit)
        return node_id % self.num_channels

    def _select_by_burst_interleave(self, flit):
        """同一packet的不同flit分配到不同通道（Per-Flit粒度）"""
        flit_position = getattr(flit, "flit_position", 0)
        return flit_position % self.num_channels

    def _select_by_ip_type_id(self, ip_id):
        """基于IP类型内的ID选择通道"""
        if ip_id is None:
            ip_id = 0
        return ip_id % self.num_channels

    def _select_by_round_robin(self, flit):
        """轮询选择通道，支持三种粒度"""
        if self.granularity == "flit":
            # 每个flit递增counter
            result = self.counter % self.num_channels
            self.counter += 1
            return result
        elif self.granularity == "packet":
            # 每个新packet递增counter
            packet_id = getattr(flit, "packet_id", 0)
            if packet_id not in self.seen_packets:
                self.seen_packets.add(packet_id)
                self.counter += 1
            return self.counter % self.num_channels
        elif self.granularity == "request":
            # DATA/RSP继承base_channel_id，新REQ递增counter
            if getattr(flit, "base_channel_id", None) is not None:
                return flit.base_channel_id
            result = self.counter % self.num_channels
            self.counter += 1
            return result
        else:
            return self.counter % self.num_channels


class StaticChannelSelector(ChannelSelector):
    """静态通道选择器，总是选择固定通道"""

    def __init__(self, channel_id=0, num_channels=2):
        super().__init__("ip_type_id_based", num_channels)
        self.fixed_channel_id = channel_id % num_channels

    def select_channel(self, flit, ip_id=None):
        """总是返回固定的通道ID"""
        return self.fixed_channel_id
