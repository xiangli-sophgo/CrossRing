"""
Pod 层级模型 - S2S (Server-to-Server)
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
from .server_model import ServerModel


# Pod 层级默认参数
DEFAULT_POD_CONFIG = {
    "num_servers": 4,
    "s2s_latency_ns": 2000.0,      # S2S 传播延迟 (ns) = 2μs
    "s2s_bandwidth_gbps": 100.0,   # S2S 带宽 (GB/s) - 100GbE
    "server_config": {},           # Server 层级配置
    "topology": "full_mesh",       # 拓扑类型: full_mesh, fat_tree, ring
}


class PodModel(HierarchicalModel):
    """
    Pod 层级模型

    建模 Server 间 (S2S) 通信，如数据中心网络交换
    """

    def __init__(
        self,
        model_id: str,
        config: Optional[Dict] = None
    ):
        merged_config = {**DEFAULT_POD_CONFIG, **(config or {})}
        super().__init__(model_id, HierarchyLevel.POD, merged_config)

        # Pod 特有参数
        self.num_servers = self.config["num_servers"]
        self.s2s_latency_ns = self.config["s2s_latency_ns"]
        self.s2s_bandwidth_gbps = self.config["s2s_bandwidth_gbps"]
        self.server_config = self.config.get("server_config", {})
        self.topology = self.config.get("topology", "full_mesh")

        # 创建子 Server 模型
        self._create_servers()
        # 创建 S2S 连接
        self._create_s2s_connections()

    def _create_servers(self):
        """创建子 Server 模型"""
        for i in range(self.num_servers):
            server_id = f"{self.model_id}_server_{i}"
            server = ServerModel(server_id, self.server_config)
            self.add_child(server_id, server)

    def _create_s2s_connections(self):
        """创建 S2S 连接"""
        server_ids = list(self.children.keys())

        if self.topology == "full_mesh":
            # 全连接拓扑
            for i, src_id in enumerate(server_ids):
                for j, dst_id in enumerate(server_ids):
                    if i < j:
                        conn_id = f"s2s_{src_id}_{dst_id}"
                        conn = HierarchyConnection(
                            id=conn_id,
                            connection_type=ConnectionType.S2S,
                            source_id=src_id,
                            target_id=dst_id,
                            latency_ns=self.s2s_latency_ns,
                            bandwidth_gbps=self.s2s_bandwidth_gbps,
                        )
                        self.add_connection(conn)

        elif self.topology == "ring":
            # 环形拓扑
            for i in range(len(server_ids)):
                src_id = server_ids[i]
                dst_id = server_ids[(i + 1) % len(server_ids)]
                conn_id = f"s2s_{src_id}_{dst_id}"
                conn = HierarchyConnection(
                    id=conn_id,
                    connection_type=ConnectionType.S2S,
                    source_id=src_id,
                    target_id=dst_id,
                    latency_ns=self.s2s_latency_ns,
                    bandwidth_gbps=self.s2s_bandwidth_gbps,
                )
                self.add_connection(conn)

        # fat_tree 等其他拓扑可后续扩展

    def _identify_cross_server_flows(
        self, traffic_flows: List[TrafficFlow]
    ) -> List[TrafficFlow]:
        """识别跨 Server 的流量"""
        return [f for f in traffic_flows if "cross_server" in f.source_id or "cross_server" in f.destination_id]

    def calculate_latency(self, traffic_flows: List[TrafficFlow]) -> LatencyResult:
        """计算 Pod 级 (S2S) 延迟"""
        cross_server_flows = self._identify_cross_server_flows(traffic_flows)

        if not cross_server_flows:
            return LatencyResult()

        # S2S 传播延迟 (往返)
        propagation_latency = self.s2s_latency_ns * 2

        # 利用率
        cross_server_bw = sum(f.bandwidth_gbps for f in cross_server_flows)

        # 多连接场景下的有效带宽
        num_conns = len(self.connections) if self.connections else 1
        effective_capacity = self.s2s_bandwidth_gbps * num_conns
        utilization = min(cross_server_bw / effective_capacity, 0.99)

        # 排队延迟 (网络交换引入)
        queuing_latency = CongestionModel.nonlinear_latency_increase(
            base_latency_ns=self.s2s_latency_ns * 0.5,  # 交换机排队
            utilization=utilization
        )

        return LatencyResult(
            propagation_latency_ns=propagation_latency,
            queuing_latency_ns=max(0, queuing_latency),
        )

    def calculate_bandwidth(self, traffic_flows: List[TrafficFlow]) -> BandwidthResult:
        """计算 Pod 级 (S2S) 带宽"""
        cross_server_flows = self._identify_cross_server_flows(traffic_flows)

        # 聚合带宽
        num_conns = len(self.connections) if self.connections else 1
        aggregate_bw = self.s2s_bandwidth_gbps * num_conns

        if not cross_server_flows:
            return BandwidthResult(
                theoretical_bandwidth_gbps=aggregate_bw,
                utilization=0.0
            )

        cross_server_demand = sum(f.bandwidth_gbps for f in cross_server_flows)
        utilization = cross_server_demand / aggregate_bw if aggregate_bw > 0 else 0

        return BandwidthResult(
            theoretical_bandwidth_gbps=aggregate_bw,
            effective_bandwidth_gbps=min(aggregate_bw, cross_server_demand),
            utilization=utilization,
            bottleneck_link="s2s_network" if utilization > 0.8 else None,
        )

    def get_topology_info(self) -> Dict:
        """获取拓扑信息"""
        return {
            "topology": self.topology,
            "num_servers": self.num_servers,
            "num_connections": len(self.connections),
            "total_bandwidth_gbps": self.s2s_bandwidth_gbps * len(self.connections),
        }
