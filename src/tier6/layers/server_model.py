"""
Server 层级模型 - B2B (Board-to-Board)
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
from .board_model import BoardModel


# Server 层级默认参数
DEFAULT_SERVER_CONFIG = {
    "num_boards": 2,
    "b2b_latency_ns": 500.0,      # B2B 传播延迟 (ns)
    "b2b_bandwidth_gbps": 32.0,   # B2B 带宽 (GB/s) - 背板/线缆
    "board_config": {},           # Board 层级配置
}


class ServerModel(HierarchicalModel):
    """
    Server 层级模型

    建模 Board 间 (B2B) 通信，如背板、线缆等
    """

    def __init__(
        self,
        model_id: str,
        config: Optional[Dict] = None
    ):
        merged_config = {**DEFAULT_SERVER_CONFIG, **(config or {})}
        super().__init__(model_id, HierarchyLevel.SERVER, merged_config)

        # Server 特有参数
        self.num_boards = self.config["num_boards"]
        self.b2b_latency_ns = self.config["b2b_latency_ns"]
        self.b2b_bandwidth_gbps = self.config["b2b_bandwidth_gbps"]
        self.board_config = self.config.get("board_config", {})

        # 创建子 Board 模型
        self._create_boards()
        # 创建 B2B 连接
        self._create_b2b_connections()

    def _create_boards(self):
        """创建子 Board 模型"""
        for i in range(self.num_boards):
            board_id = f"{self.model_id}_board_{i}"
            board = BoardModel(board_id, self.board_config)
            self.add_child(board_id, board)

    def _create_b2b_connections(self):
        """创建 B2B 连接"""
        board_ids = list(self.children.keys())
        for i, src_id in enumerate(board_ids):
            for j, dst_id in enumerate(board_ids):
                if i < j:
                    conn_id = f"b2b_{src_id}_{dst_id}"
                    conn = HierarchyConnection(
                        id=conn_id,
                        connection_type=ConnectionType.B2B,
                        source_id=src_id,
                        target_id=dst_id,
                        latency_ns=self.b2b_latency_ns,
                        bandwidth_gbps=self.b2b_bandwidth_gbps,
                    )
                    self.add_connection(conn)

    def _identify_cross_board_flows(
        self, traffic_flows: List[TrafficFlow]
    ) -> List[TrafficFlow]:
        """识别跨 Board 的流量"""
        return [f for f in traffic_flows if "cross_board" in f.source_id or "cross_board" in f.destination_id]

    def calculate_latency(self, traffic_flows: List[TrafficFlow]) -> LatencyResult:
        """计算 Server 级 (B2B) 延迟"""
        cross_board_flows = self._identify_cross_board_flows(traffic_flows)

        if not cross_board_flows:
            return LatencyResult()

        # B2B 传播延迟 (往返)
        propagation_latency = self.b2b_latency_ns * 2

        # 利用率
        cross_board_bw = sum(f.bandwidth_gbps for f in cross_board_flows)
        utilization = min(cross_board_bw / self.b2b_bandwidth_gbps, 0.99)

        # 排队延迟
        queuing_latency = CongestionModel.nonlinear_latency_increase(
            base_latency_ns=self.b2b_latency_ns,
            utilization=utilization
        ) - self.b2b_latency_ns

        return LatencyResult(
            propagation_latency_ns=propagation_latency,
            queuing_latency_ns=max(0, queuing_latency),
        )

    def calculate_bandwidth(self, traffic_flows: List[TrafficFlow]) -> BandwidthResult:
        """计算 Server 级 (B2B) 带宽"""
        cross_board_flows = self._identify_cross_board_flows(traffic_flows)

        if not cross_board_flows:
            return BandwidthResult(
                theoretical_bandwidth_gbps=self.b2b_bandwidth_gbps,
                utilization=0.0
            )

        cross_board_demand = sum(f.bandwidth_gbps for f in cross_board_flows)
        utilization = BandwidthModel.calculate_utilization(
            cross_board_flows, self.b2b_bandwidth_gbps
        )

        return BandwidthResult(
            theoretical_bandwidth_gbps=self.b2b_bandwidth_gbps,
            effective_bandwidth_gbps=min(self.b2b_bandwidth_gbps, cross_board_demand),
            utilization=utilization,
            bottleneck_link="b2b" if utilization > 0.8 else None,
        )
