"""
Chip 层级模型 - D2D (Die-to-Die)
"""

from typing import Dict, List, Optional

from ..base import (
    HierarchicalModel,
    HierarchyLevel,
    ConnectionType,
    HierarchyConnection,
    LatencyResult,
    BandwidthResult,
    TrafficFlow,
)
from ..math_models import CongestionModel, BandwidthModel
from .die_model import DieModel


# Chip 层级默认参数
DEFAULT_CHIP_CONFIG = {
    "num_dies": 2,
    "d2d_latency_ns": 20.0,       # D2D 传播延迟 (ns)
    "d2d_bandwidth_gbps": 192.0,  # D2D 带宽 (GB/s)
    "die_config": {},             # Die 层级配置
}


class ChipModel(HierarchicalModel):
    """
    Chip 层级模型

    建模 Die 间 (D2D) 通信，如 UCIe, CoWoS 等
    """

    def __init__(
        self,
        model_id: str,
        config: Optional[Dict] = None
    ):
        merged_config = {**DEFAULT_CHIP_CONFIG, **(config or {})}
        super().__init__(model_id, HierarchyLevel.CHIP, merged_config)

        # Chip 特有参数
        self.num_dies = self.config["num_dies"]
        self.d2d_latency_ns = self.config["d2d_latency_ns"]
        self.d2d_bandwidth_gbps = self.config["d2d_bandwidth_gbps"]
        self.die_config = self.config.get("die_config", {})

        # 创建子 Die 模型
        self._create_dies()
        # 创建 D2D 连接
        self._create_d2d_connections()

    def _create_dies(self):
        """创建子 Die 模型"""
        for i in range(self.num_dies):
            die_id = f"{self.model_id}_die_{i}"
            die = DieModel(die_id, self.die_config)
            self.add_child(die_id, die)

    def _create_d2d_connections(self):
        """创建 D2D 连接 (全连接拓扑)"""
        die_ids = list(self.children.keys())
        for i, src_id in enumerate(die_ids):
            for j, dst_id in enumerate(die_ids):
                if i < j:  # 避免重复
                    conn_id = f"d2d_{src_id}_{dst_id}"
                    conn = HierarchyConnection(
                        id=conn_id,
                        connection_type=ConnectionType.D2D,
                        source_id=src_id,
                        target_id=dst_id,
                        latency_ns=self.d2d_latency_ns,
                        bandwidth_gbps=self.d2d_bandwidth_gbps,
                    )
                    self.add_connection(conn)

    def _identify_cross_die_flows(
        self, traffic_flows: List[TrafficFlow]
    ) -> List[TrafficFlow]:
        """识别跨 Die 的流量"""
        # 简化模型：假设 flow_id 包含 "cross_die" 或源目标不同 Die
        cross_die_flows = []
        for flow in traffic_flows:
            # 检查是否跨 Die
            if "cross_die" in flow.source_id or "cross_die" in flow.destination_id:
                cross_die_flows.append(flow)
            elif flow.source_id != flow.destination_id:
                # 简化：不同 ID 可能跨 Die
                cross_die_flows.append(flow)
        return cross_die_flows

    def calculate_latency(self, traffic_flows: List[TrafficFlow]) -> LatencyResult:
        """计算 Chip 级 (D2D) 延迟"""
        cross_die_flows = self._identify_cross_die_flows(traffic_flows)

        if not cross_die_flows:
            return LatencyResult()

        # D2D 传播延迟 (往返)
        propagation_latency = self.d2d_latency_ns * 2

        # 计算 D2D 链路利用率
        cross_die_bw = sum(f.bandwidth_gbps for f in cross_die_flows)
        utilization = min(cross_die_bw / self.d2d_bandwidth_gbps, 0.99)

        # 排队延迟
        queuing_latency = CongestionModel.nonlinear_latency_increase(
            base_latency_ns=self.d2d_latency_ns,
            utilization=utilization
        ) - self.d2d_latency_ns

        return LatencyResult(
            propagation_latency_ns=propagation_latency,
            queuing_latency_ns=max(0, queuing_latency),
            processing_latency_ns=0,
            transmission_latency_ns=0,
        )

    def calculate_bandwidth(self, traffic_flows: List[TrafficFlow]) -> BandwidthResult:
        """计算 Chip 级 (D2D) 带宽"""
        cross_die_flows = self._identify_cross_die_flows(traffic_flows)

        if not cross_die_flows:
            return BandwidthResult(
                theoretical_bandwidth_gbps=self.d2d_bandwidth_gbps,
                utilization=0.0
            )

        cross_die_demand = sum(f.bandwidth_gbps for f in cross_die_flows)
        utilization = BandwidthModel.calculate_utilization(
            cross_die_flows, self.d2d_bandwidth_gbps
        )

        return BandwidthResult(
            theoretical_bandwidth_gbps=self.d2d_bandwidth_gbps,
            effective_bandwidth_gbps=min(self.d2d_bandwidth_gbps, cross_die_demand),
            utilization=utilization,
            bottleneck_link="d2d" if utilization > 0.8 else None,
        )

    def update_connection_stats(self, traffic_flows: List[TrafficFlow]):
        """更新连接统计"""
        cross_die_flows = self._identify_cross_die_flows(traffic_flows)
        total_traffic = sum(f.bandwidth_gbps for f in cross_die_flows)

        # 均分到各连接
        num_conns = len(self.connections)
        if num_conns > 0:
            traffic_per_conn = total_traffic / num_conns
            for conn in self.connections.values():
                conn.traffic_gbps = traffic_per_conn
                conn.utilization = traffic_per_conn / conn.bandwidth_gbps
