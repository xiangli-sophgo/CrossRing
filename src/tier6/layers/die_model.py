"""
Die 层级模型 - NoC 内部
"""

from typing import Dict, List, Optional

from ..base import (
    HierarchicalModel,
    HierarchyLevel,
    LatencyResult,
    BandwidthResult,
    TrafficFlow,
)
from ..math_models import QueuingModel, CongestionModel, BandwidthModel


# Die 层级默认参数
DEFAULT_DIE_CONFIG = {
    "num_nodes": 20,           # 节点数 (5x4)
    "num_cols": 4,             # 列数
    "noc_link_bw_gbps": 128.0, # NoC 链路带宽 (GB/s)
    "hop_latency_ns": 0.5,     # 单跳延迟 (ns)
    "ddr_read_latency_ns": 50, # DDR 读延迟 (ns)
    "l2m_read_latency_ns": 20, # L2M 读延迟 (ns)
}


class DieModel(HierarchicalModel):
    """
    Die 层级模型

    建模 NoC 内部通信延迟和带宽
    """

    def __init__(
        self,
        model_id: str,
        config: Optional[Dict] = None
    ):
        # 合并默认配置
        merged_config = {**DEFAULT_DIE_CONFIG, **(config or {})}
        super().__init__(model_id, HierarchyLevel.DIE, merged_config)

        # Die 特有参数
        self.num_nodes = self.config["num_nodes"]
        self.num_cols = self.config["num_cols"]
        self.num_rows = self.num_nodes // self.num_cols
        self.noc_link_bw_gbps = self.config["noc_link_bw_gbps"]
        self.hop_latency_ns = self.config["hop_latency_ns"]
        self.ddr_read_latency_ns = self.config["ddr_read_latency_ns"]
        self.l2m_read_latency_ns = self.config["l2m_read_latency_ns"]

    def _estimate_average_hops(self) -> float:
        """估算平均跳数 (曼哈顿距离)"""
        # 对于均匀随机流量，平均跳数约为 (rows + cols) / 3
        return (self.num_rows + self.num_cols) / 3

    def _estimate_hop_latency(self, num_hops: float) -> float:
        """估算跳数对应的传播延迟"""
        return num_hops * self.hop_latency_ns

    def calculate_latency(self, traffic_flows: List[TrafficFlow]) -> LatencyResult:
        """计算 Die 内延迟"""
        if not traffic_flows:
            return LatencyResult()

        # 传播延迟：基于平均跳数
        avg_hops = self._estimate_average_hops()
        propagation_latency = self._estimate_hop_latency(avg_hops)

        # 计算总流量需求
        total_traffic_gbps = sum(f.bandwidth_gbps for f in traffic_flows)
        total_request_rate = sum(f.request_rate_per_sec for f in traffic_flows)

        # 链路利用率
        # 简化模型：总流量 / (链路数 * 链路带宽)
        num_links = self.num_nodes * 2  # 简化估计
        total_capacity = num_links * self.noc_link_bw_gbps
        utilization = min(total_traffic_gbps / total_capacity, 0.99) if total_capacity > 0 else 0

        # 排队延迟：基于利用率的非线性增长
        queuing_latency = CongestionModel.nonlinear_latency_increase(
            base_latency_ns=self.hop_latency_ns,
            utilization=utilization
        ) - self.hop_latency_ns  # 减去基础延迟避免重复计算

        # 处理延迟：取 DDR 和 L2M 延迟的加权平均
        processing_latency = (self.ddr_read_latency_ns + self.l2m_read_latency_ns) / 2

        # 传输延迟：基于突发大小和带宽
        avg_burst = sum(f.burst_size_bytes for f in traffic_flows) / len(traffic_flows)
        transmission_latency = (avg_burst / (self.noc_link_bw_gbps * 1e9)) * 1e9  # ns

        return LatencyResult(
            propagation_latency_ns=propagation_latency,
            queuing_latency_ns=max(0, queuing_latency),
            processing_latency_ns=processing_latency,
            transmission_latency_ns=transmission_latency,
        )

    def calculate_bandwidth(self, traffic_flows: List[TrafficFlow]) -> BandwidthResult:
        """计算 Die 内带宽"""
        if not traffic_flows:
            return BandwidthResult()

        total_demand = sum(f.bandwidth_gbps for f in traffic_flows)

        # 理论最大带宽：对分带宽 (bisection bandwidth)
        # 简化为 num_cols * link_bw
        theoretical_bw = self.num_cols * self.noc_link_bw_gbps

        utilization = BandwidthModel.calculate_utilization(traffic_flows, theoretical_bw)

        # 有效带宽受拥塞影响
        effective_bw = BandwidthModel.effective_bandwidth(
            theoretical_bw, utilization
        )

        return BandwidthResult(
            theoretical_bandwidth_gbps=theoretical_bw,
            effective_bandwidth_gbps=min(effective_bw, total_demand),
            utilization=utilization,
            bottleneck_link="noc_bisection" if utilization > 0.8 else None,
        )
