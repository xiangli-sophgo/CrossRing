"""
Tier6+ 基础类和数据结构定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class HierarchyLevel(Enum):
    """层级类型"""
    DIE = "die"
    CHIP = "chip"
    BOARD = "board"
    SERVER = "server"
    POD = "pod"


class ConnectionType(Enum):
    """跨层级连接类型"""
    D2D = "d2d"      # Die-to-Die
    C2C = "c2c"      # Chip-to-Chip
    B2B = "b2b"      # Board-to-Board
    S2S = "s2s"      # Server-to-Server
    P2P = "p2p"      # Pod-to-Pod


@dataclass
class LatencyResult:
    """延迟计算结果"""
    propagation_latency_ns: float = 0.0    # 传播延迟
    queuing_latency_ns: float = 0.0        # 排队延迟
    processing_latency_ns: float = 0.0     # 处理延迟
    transmission_latency_ns: float = 0.0   # 传输延迟

    @property
    def total_latency_ns(self) -> float:
        """总延迟"""
        return (
            self.propagation_latency_ns +
            self.queuing_latency_ns +
            self.processing_latency_ns +
            self.transmission_latency_ns
        )

    def __add__(self, other: "LatencyResult") -> "LatencyResult":
        """延迟累加"""
        return LatencyResult(
            propagation_latency_ns=self.propagation_latency_ns + other.propagation_latency_ns,
            queuing_latency_ns=self.queuing_latency_ns + other.queuing_latency_ns,
            processing_latency_ns=self.processing_latency_ns + other.processing_latency_ns,
            transmission_latency_ns=self.transmission_latency_ns + other.transmission_latency_ns,
        )


@dataclass
class BandwidthResult:
    """带宽计算结果"""
    theoretical_bandwidth_gbps: float = 0.0   # 理论带宽
    effective_bandwidth_gbps: float = 0.0     # 有效带宽
    utilization: float = 0.0                  # 链路利用率
    bottleneck_link: Optional[str] = None     # 瓶颈链路标识


@dataclass
class TrafficFlow:
    """流量流定义"""
    flow_id: str                              # 流量ID
    source_id: str                            # 源节点ID
    destination_id: str                       # 目标节点ID
    bandwidth_gbps: float                     # 带宽需求 (GB/s)
    request_rate_per_sec: float = 0.0         # 请求率 (req/s)
    burst_size_bytes: int = 512               # 突发大小 (bytes)
    read_ratio: float = 0.5                   # 读占比


@dataclass
class HierarchyNode:
    """层级节点"""
    id: str
    name: str
    level: HierarchyLevel
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # 性能参数
    internal_latency_ns: float = 0.0          # 内部延迟
    internal_bandwidth_gbps: float = float('inf')  # 内部带宽

    # 统计结果
    latency_result: Optional[LatencyResult] = None
    bandwidth_result: Optional[BandwidthResult] = None


@dataclass
class HierarchyConnection:
    """层级间连接"""
    id: str
    connection_type: ConnectionType
    source_id: str
    target_id: str

    # 连接参数
    latency_ns: float = 0.0
    bandwidth_gbps: float = 0.0

    # 运行时统计
    utilization: float = 0.0
    traffic_gbps: float = 0.0


class HierarchicalModel(ABC):
    """层级模型抽象基类"""

    def __init__(
        self,
        model_id: str,
        level: HierarchyLevel,
        config: Optional[Dict] = None
    ):
        self.model_id = model_id
        self.level = level
        self.config = config or {}

        # 层级结构
        self.children: Dict[str, "HierarchicalModel"] = {}
        self.parent: Optional["HierarchicalModel"] = None

        # 连接
        self.connections: Dict[str, HierarchyConnection] = {}

        # 默认参数
        self.propagation_latency_ns: float = 0.0
        self.bandwidth_limit_gbps: float = float('inf')

        # 结果缓存
        self._latency_result: Optional[LatencyResult] = None
        self._bandwidth_result: Optional[BandwidthResult] = None

        self._init_from_config()

    def _init_from_config(self):
        """从配置初始化参数"""
        if not self.config:
            return
        self.propagation_latency_ns = self.config.get('latency_ns', self.propagation_latency_ns)
        self.bandwidth_limit_gbps = self.config.get('bandwidth_gbps', self.bandwidth_limit_gbps)

    @abstractmethod
    def calculate_latency(self, traffic_flows: List[TrafficFlow]) -> LatencyResult:
        """计算本层级延迟"""
        pass

    @abstractmethod
    def calculate_bandwidth(self, traffic_flows: List[TrafficFlow]) -> BandwidthResult:
        """计算本层级带宽"""
        pass

    def add_child(self, child_id: str, child: "HierarchicalModel"):
        """添加子层级"""
        self.children[child_id] = child
        child.parent = self

    def add_connection(self, connection: HierarchyConnection):
        """添加连接"""
        self.connections[connection.id] = connection

    def get_total_latency(self, traffic_flows: List[TrafficFlow]) -> LatencyResult:
        """递归计算从本层到最底层的总延迟"""
        # 本层延迟
        local_latency = self.calculate_latency(traffic_flows)

        if not self.children:
            return local_latency

        # 子层级延迟取最大值（并行路径取最慢的）
        child_latencies = [
            child.get_total_latency(traffic_flows)
            for child in self.children.values()
        ]

        if child_latencies:
            max_child_latency = max(child_latencies, key=lambda x: x.total_latency_ns)
            return local_latency + max_child_latency

        return local_latency

    def get_latency_breakdown(self, traffic_flows: List[TrafficFlow]) -> Dict[str, LatencyResult]:
        """获取各层级延迟分解"""
        breakdown = {}

        # 本层延迟
        breakdown[self.level.value] = self.calculate_latency(traffic_flows)

        # 递归子层级
        for child in self.children.values():
            child_breakdown = child.get_latency_breakdown(traffic_flows)
            for level, latency in child_breakdown.items():
                if level in breakdown:
                    # 同层级取最大
                    if latency.total_latency_ns > breakdown[level].total_latency_ns:
                        breakdown[level] = latency
                else:
                    breakdown[level] = latency

        return breakdown

    def find_bandwidth_bottleneck(self, traffic_flows: List[TrafficFlow]) -> Optional[Tuple[str, BandwidthResult]]:
        """找出带宽瓶颈"""
        max_utilization = 0.0
        bottleneck = None

        # 检查本层
        bw_result = self.calculate_bandwidth(traffic_flows)
        if bw_result.utilization > max_utilization:
            max_utilization = bw_result.utilization
            bottleneck = (f"{self.level.value}:{self.model_id}", bw_result)

        # 检查连接
        for conn_id, conn in self.connections.items():
            if conn.utilization > max_utilization:
                max_utilization = conn.utilization
                bottleneck = (f"{conn.connection_type.value}:{conn_id}", BandwidthResult(
                    theoretical_bandwidth_gbps=conn.bandwidth_gbps,
                    effective_bandwidth_gbps=conn.traffic_gbps,
                    utilization=conn.utilization,
                    bottleneck_link=conn_id
                ))

        # 递归子层级
        for child in self.children.values():
            child_bottleneck = child.find_bandwidth_bottleneck(traffic_flows)
            if child_bottleneck:
                _, child_bw = child_bottleneck
                if child_bw.utilization > max_utilization:
                    max_utilization = child_bw.utilization
                    bottleneck = child_bottleneck

        return bottleneck

    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            "id": self.model_id,
            "level": self.level.value,
            "latency_ns": self.propagation_latency_ns,
            "bandwidth_gbps": self.bandwidth_limit_gbps,
            "children": {k: v.to_dict() for k, v in self.children.items()},
            "connections": {k: {
                "type": v.connection_type.value,
                "source": v.source_id,
                "target": v.target_id,
                "latency_ns": v.latency_ns,
                "bandwidth_gbps": v.bandwidth_gbps,
            } for k, v in self.connections.items()},
        }
