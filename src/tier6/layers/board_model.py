"""
Board 层级模型 - C2C (Chip-to-Chip)
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
from .chip_model import ChipModel


# Board 层级默认参数
DEFAULT_BOARD_CONFIG = {
    "num_chips": 2,
    "c2c_latency_ns": 100.0,      # C2C 传播延迟 (ns)
    "c2c_bandwidth_gbps": 64.0,   # C2C 带宽 (GB/s) - UCIe/PCIe
    "chip_config": {},            # Chip 层级配置
}


class BoardModel(HierarchicalModel):
    """
    Board 层级模型

    建模 Chip 间 (C2C) 通信，如 UCIe, PCIe 等
    """

    def __init__(
        self,
        model_id: str,
        config: Optional[Dict] = None
    ):
        merged_config = {**DEFAULT_BOARD_CONFIG, **(config or {})}
        super().__init__(model_id, HierarchyLevel.BOARD, merged_config)

        # Board 特有参数
        self.num_chips = self.config["num_chips"]
        self.c2c_latency_ns = self.config["c2c_latency_ns"]
        self.c2c_bandwidth_gbps = self.config["c2c_bandwidth_gbps"]
        self.chip_config = self.config.get("chip_config", {})

        # 创建子 Chip 模型
        self._create_chips()
        # 创建 C2C 连接
        self._create_c2c_connections()

    def _create_chips(self):
        """创建子 Chip 模型"""
        for i in range(self.num_chips):
            chip_id = f"{self.model_id}_chip_{i}"
            chip = ChipModel(chip_id, self.chip_config)
            self.add_child(chip_id, chip)

    def _create_c2c_connections(self):
        """创建 C2C 连接"""
        chip_ids = list(self.children.keys())
        for i, src_id in enumerate(chip_ids):
            for j, dst_id in enumerate(chip_ids):
                if i < j:
                    conn_id = f"c2c_{src_id}_{dst_id}"
                    conn = HierarchyConnection(
                        id=conn_id,
                        connection_type=ConnectionType.C2C,
                        source_id=src_id,
                        target_id=dst_id,
                        latency_ns=self.c2c_latency_ns,
                        bandwidth_gbps=self.c2c_bandwidth_gbps,
                    )
                    self.add_connection(conn)

    def _identify_cross_chip_flows(
        self, traffic_flows: List[TrafficFlow]
    ) -> List[TrafficFlow]:
        """识别跨 Chip 的流量"""
        return [f for f in traffic_flows if "cross_chip" in f.source_id or "cross_chip" in f.destination_id]

    def calculate_latency(self, traffic_flows: List[TrafficFlow]) -> LatencyResult:
        """计算 Board 级 (C2C) 延迟"""
        cross_chip_flows = self._identify_cross_chip_flows(traffic_flows)

        if not cross_chip_flows:
            return LatencyResult()

        # C2C 传播延迟 (往返)
        propagation_latency = self.c2c_latency_ns * 2

        # 利用率
        cross_chip_bw = sum(f.bandwidth_gbps for f in cross_chip_flows)
        utilization = min(cross_chip_bw / self.c2c_bandwidth_gbps, 0.99)

        # 排队延迟
        queuing_latency = CongestionModel.nonlinear_latency_increase(
            base_latency_ns=self.c2c_latency_ns,
            utilization=utilization
        ) - self.c2c_latency_ns

        return LatencyResult(
            propagation_latency_ns=propagation_latency,
            queuing_latency_ns=max(0, queuing_latency),
        )

    def calculate_bandwidth(self, traffic_flows: List[TrafficFlow]) -> BandwidthResult:
        """计算 Board 级 (C2C) 带宽"""
        cross_chip_flows = self._identify_cross_chip_flows(traffic_flows)

        if not cross_chip_flows:
            return BandwidthResult(
                theoretical_bandwidth_gbps=self.c2c_bandwidth_gbps,
                utilization=0.0
            )

        cross_chip_demand = sum(f.bandwidth_gbps for f in cross_chip_flows)
        utilization = BandwidthModel.calculate_utilization(
            cross_chip_flows, self.c2c_bandwidth_gbps
        )

        return BandwidthResult(
            theoretical_bandwidth_gbps=self.c2c_bandwidth_gbps,
            effective_bandwidth_gbps=min(self.c2c_bandwidth_gbps, cross_chip_demand),
            utilization=utilization,
            bottleneck_link="c2c" if utilization > 0.8 else None,
        )
