"""
Ring拓扑建模类 - 基于BaseModel重新实现
复用CrossRing的大部分功能，添加Ring特有的路由和拓扑支持
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time
import logging

from src.core.base_model import BaseModel
from src.utils.routing_strategies import create_routing_strategy
from src.utils.component import Flit, Network, Node
from config.config import CrossRingConfig


class RingConfig(CrossRingConfig):
    """Ring拓扑配置 - 扩展CrossRingConfig"""

    def __init__(self, config_file_path=None):
        super().__init__(config_file_path)

        # Ring特有参数
        self.RING_NUM_NODES = 8
        self.TOPO_TYPE = f"Ring_{self.RING_NUM_NODES}"

        # 路由策略配置
        self.ROUTING_STRATEGY = "load_balanced"  # shortest/load_balanced/adaptive/custom
        self.LOAD_BALANCE_POLICY = "even_cw_odd_ccw"  # 负载均衡策略
        self.ADAPTIVE_THRESHOLD = 0.7  # 自适应路由阈值
        self.CUSTOM_ROUTES = {}  # 自定义路由表

        # Ring特有的缓冲配置
        self.RING_BUFFER_DEPTH = 8
        self.ENABLE_ADAPTIVE_ROUTING = False
        self.CONGESTION_THRESHOLD = 0.7

        # 重写IP分布以适配Ring拓扑
        self._setup_ring_ip_distribution()

    def _setup_ring_ip_distribution(self):
        """设置Ring拓扑的IP分布"""
        # 在Ring中，所有节点都可以连接IP
        all_positions = list(range(self.RING_NUM_NODES))

        # 根据IP数量分配到节点
        self.GDMA_SEND_POSITION_LIST = all_positions[: min(len(all_positions), 4)]
        self.SDMA_SEND_POSITION_LIST = all_positions[: min(len(all_positions), 4)]
        self.DDR_SEND_POSITION_LIST = all_positions
        self.L2M_SEND_POSITION_LIST = all_positions[: min(len(all_positions), 4)]
        self.CDMA_SEND_POSITION_LIST = []  # Ring暂不使用CDMA

    def update_config(self, topo_type):
        """重写配置更新以支持Ring拓扑"""
        if topo_type.startswith("Ring"):
            # 解析Ring节点数
            try:
                self.RING_NUM_NODES = int(topo_type.split("_")[1])
            except (IndexError, ValueError):
                self.RING_NUM_NODES = 8

            # 更新基本拓扑参数
            self.NUM_NODE = self.RING_NUM_NODES
            self.NUM_COL = 1  # Ring是1维拓扑
            self.NUM_ROW = self.RING_NUM_NODES

            # 重新设置IP分布
            self._setup_ring_ip_distribution()
        else:
            # 调用父类的CrossRing配置更新
            super().update_config(topo_type)


class RingModel(BaseModel):
    """Ring拓扑模型 - 继承BaseModel"""

    def __init__(self, model_type, config: RingConfig, topo_type, traffic_file_path, **kwargs):
        # 确保使用Ring拓扑类型
        if not topo_type.startswith("Ring"):
            topo_type = f"Ring_{config.RING_NUM_NODES}"

        super().__init__(model_type, config, topo_type, traffic_file_path, **kwargs)

        # 初始化Ring特有的路由策略
        self.routing_strategy = self._create_routing_strategy()

        # Ring特有的统计信息
        self.ring_stats = {"cw_usage": 0, "ccw_usage": 0, "diagonal_routes": 0, "adaptive_routes": 0, "routing_decisions": defaultdict(int)}

        logging.info(f"Ring topology initialized: {self.config.RING_NUM_NODES} nodes, " f"routing strategy: {self.config.ROUTING_STRATEGY}")
        self.initial()

    def _create_routing_strategy(self):
        """创建路由策略"""
        strategy_params = {}

        if self.config.ROUTING_STRATEGY == "load_balanced":
            strategy_params["policy"] = self.config.LOAD_BALANCE_POLICY
        elif self.config.ROUTING_STRATEGY == "adaptive":
            strategy_params["congestion_threshold"] = self.config.ADAPTIVE_THRESHOLD
        elif self.config.ROUTING_STRATEGY == "custom":
            strategy_params["custom_routes"] = self.config.CUSTOM_ROUTES

        return create_routing_strategy(self.config, self.config.ROUTING_STRATEGY, **strategy_params)

    def create_adjacency_matrix(self):
        """创建Ring拓扑邻接矩阵"""
        return self._create_ring_adjacency_matrix(self.config.RING_NUM_NODES)

    def _create_ring_adjacency_matrix(self, num_nodes: int):
        """创建Ring拓扑邻接矩阵"""
        matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for i in range(num_nodes):
            # 每个节点连接到相邻的两个节点
            next_node = (i + 1) % num_nodes
            prev_node = (i - 1) % num_nodes
            matrix[i][next_node] = 1  # 顺时针连接
            matrix[i][prev_node] = 1  # 逆时针连接
        return matrix

    def initial(self):
        """重写初始化方法以支持Ring拓扑"""
        # 设置Ring特有的拓扑类型
        self.topo_type_stat = self.config.TOPO_TYPE
        self.config.update_config(self.topo_type_stat)

        # 创建Ring拓扑的邻接矩阵
        self.adjacency_matrix = self._create_ring_adjacency_matrix(self.config.RING_NUM_NODES)

        # 初始化网络（复用BaseModel的Network）
        self.req_network = Network(self.config, self.adjacency_matrix, name="Request Network")
        self.rsp_network = Network(self.config, self.adjacency_matrix, name="Response Network")
        self.data_network = Network(self.config, self.adjacency_matrix, name="Data Network")

        # Ring特有的网络配置
        self._configure_ring_networks()

        # 复用BaseModel的其他初始化
        if hasattr(self, "result_processor"):
            from src.core.result_processor import BandwidthAnalyzer

            self.result_processor = BandwidthAnalyzer(self.config, min_gap_threshold=50, plot_rn_bw_fig=self.plot_RN_BW_fig, plot_flow_graph=self.plot_flow_fig)

        # 设置节点位置信息
        self.rn_positions = set(self.config.GDMA_SEND_POSITION_LIST + self.config.SDMA_SEND_POSITION_LIST + self.config.CDMA_SEND_POSITION_LIST)
        self.sn_positions = set(self.config.DDR_SEND_POSITION_LIST + self.config.L2M_SEND_POSITION_LIST)
        self.flit_positions = set(
            self.config.GDMA_SEND_POSITION_LIST + self.config.SDMA_SEND_POSITION_LIST + self.config.CDMA_SEND_POSITION_LIST + self.config.DDR_SEND_POSITION_LIST + self.config.L2M_SEND_POSITION_LIST
        )

        # 计算Ring路径
        self.routes = self._calculate_ring_routes()

        # 初始化Node和IP模块（复用BaseModel）
        self.node = Node(self.config)
        self.ip_modules = {}
        self._setup_ip_modules()

    def _configure_ring_networks(self):
        """配置Ring网络的特有属性"""
        for network in [self.req_network, self.rsp_network, self.data_network]:
            # Ring特有的网络配置
            network.ring_mode = True
            network.routing_strategy = self.routing_strategy

            # 配置ETag支持（复用CrossRing的ETag逻辑）
            if self.config.ETag_BOTHSIDE_UPGRADE:
                network.ETag_BOTHSIDE_UPGRADE = True

    def _calculate_ring_routes(self):
        """计算Ring拓扑的路由表"""
        routes = {}
        num_nodes = self.config.RING_NUM_NODES

        for source in range(num_nodes):
            routes[source] = {}
            for destination in range(num_nodes):
                routes[source][destination] = self._get_ring_path(source, destination)

        return routes

    def _get_ring_path(self, source: int, destination: int) -> List[int]:
        """获取Ring拓扑中两点间的路径"""
        if source == destination:
            return [source]

        # 使用路由策略决定方向
        direction = self.routing_strategy.get_route_direction(source, destination)

        path = [source]
        current = source

        if direction == "CW":
            # 顺时针路径
            while current != destination:
                current = (current + 1) % self.config.RING_NUM_NODES
                path.append(current)
        else:  # CCW
            # 逆时针路径
            while current != destination:
                current = (current - 1) % self.config.RING_NUM_NODES
                path.append(current)

        return path

    def _setup_ip_modules(self):
        """设置IP模块（复用BaseModel的逻辑）"""
        from src.utils.component import IPInterface, RingIPInterface

        for ip_pos in self.flit_positions:
            for ip_type in self.config.CH_NAME_LIST:
                self.ip_modules[(ip_type, ip_pos)] = IPInterface(ip_type, ip_pos, self.config, self.req_network, self.rsp_network, self.data_network, self.node, self.routes)

    def get_valid_directions(self, current_pos: int, target_pos: int, **context) -> List[str]:
        """
        获取Ring拓扑中的有效路由方向
        这是Ring特有的路由逻辑
        """
        if current_pos == target_pos:
            return ["LOCAL"]

        # 使用路由策略决定方向
        direction = self.routing_strategy.get_route_direction(current_pos, target_pos, **context)

        # 更新统计信息
        self.ring_stats["routing_decisions"][direction] += 1

        if direction == "CW":
            self.ring_stats["cw_usage"] += 1
            return ["TR"]  # CrossRing中TR表示向右（顺时针）
        elif direction == "CCW":
            self.ring_stats["ccw_usage"] += 1
            return ["TL"]  # CrossRing中TL表示向左（逆时针）
        else:
            return ["LOCAL"]

    def adaptive_routing_decision(self, flit: Flit, context: Dict = None) -> str:
        """
        自适应路由决策（如果启用）
        复用并扩展原有的自适应路由逻辑
        """
        if not self.config.ENABLE_ADAPTIVE_ROUTING:
            return self.routing_strategy.get_route_direction(flit.source, flit.destination)

        # 获取网络拥塞信息
        if context is None:
            context = self._get_network_congestion_info(flit.source)

        direction = self.routing_strategy.get_route_direction(flit.source, flit.destination, **context)

        if hasattr(self.routing_strategy, "get_path_info"):
            path_info = self.routing_strategy.get_path_info(flit.source, flit.destination, direction)
            # 可以在这里记录路径信息用于分析

        self.ring_stats["adaptive_routes"] += 1
        return direction

    def _get_network_congestion_info(self, node_id: int) -> Dict:
        """获取网络拥塞信息用于自适应路由"""
        # 这里可以实现具体的拥塞检测逻辑
        # 暂时返回模拟数据
        return {"cw_congestion": 0.0, "ccw_congestion": 0.0, "node_utilization": 0.0}

    def step_simulation(self):
        """执行一个仿真周期 - 复用BaseModel的大部分逻辑"""
        # 调用父类的仿真步骤
        super().step_simulation()

        # Ring特有的处理
        self._update_ring_statistics()

    def _update_ring_statistics(self):
        """更新Ring特有的统计信息"""
        total_decisions = sum(self.ring_stats["routing_decisions"].values())
        if total_decisions > 0:
            self.ring_stats["cw_utilization"] = self.ring_stats["cw_usage"] / total_decisions
            self.ring_stats["ccw_utilization"] = self.ring_stats["ccw_usage"] / total_decisions

    def get_ring_performance_stats(self) -> Dict:
        """获取Ring特有的性能统计"""
        base_stats = super().get_performance_stats() if hasattr(super(), "get_performance_stats") else {}

        ring_specific_stats = {
            "routing_strategy": self.config.ROUTING_STRATEGY,
            "cw_utilization": self.ring_stats.get("cw_utilization", 0),
            "ccw_utilization": self.ring_stats.get("ccw_utilization", 0),
            "diagonal_routes": self.ring_stats["diagonal_routes"],
            "adaptive_routes": self.ring_stats["adaptive_routes"],
            "routing_decisions": dict(self.ring_stats["routing_decisions"]),
        }

        return {**base_stats, **ring_specific_stats}

    def print_ring_summary(self):
        """打印Ring拓扑的性能摘要"""
        print(f"\n=== Ring Topology Performance Summary ===")
        print(f"Nodes: {self.config.RING_NUM_NODES}")
        print(f"Routing Strategy: {self.config.ROUTING_STRATEGY}")

        if self.config.ROUTING_STRATEGY == "load_balanced":
            print(f"Load Balance Policy: {self.config.LOAD_BALANCE_POLICY}")

        print(f"CW Utilization: {self.ring_stats.get('cw_utilization', 0):.3f}")
        print(f"CCW Utilization: {self.ring_stats.get('ccw_utilization', 0):.3f}")
        print(f"Adaptive Routes: {self.ring_stats['adaptive_routes']}")

        print("\nRouting Decisions:")
        for direction, count in self.ring_stats["routing_decisions"].items():
            print(f"  {direction}: {count}")
        print("==========================================")


# 便捷的工厂函数
def create_ring_model(num_nodes: int = 8, routing_strategy: str = "load_balanced", model_type: str = "REQ_RSP", **kwargs) -> RingModel:
    """
    创建Ring模型的便捷函数

    Args:
        num_nodes: Ring节点数量
        routing_strategy: 路由策略名称
        model_type: 模型类型
        **kwargs: 其他参数

    Returns:
        RingModel: 配置好的Ring模型实例
    """
    config = RingConfig(r"../../config/config2.json")
    config.RING_NUM_NODES = num_nodes
    config.ROUTING_STRATEGY = routing_strategy

    topo_type = f"Ring_{num_nodes}"

    # 设置默认参数
    default_params = {
        "traffic_file_path": "./test_data",
        "traffic_config": [["Read_burst4_2262HBM_v2.txt"]],
        "result_save_path": f"./results/Ring_{num_nodes}",
        "plot_flow_fig": True,
        "plot_RN_BW_fig": True,
        "verbose": True,
    }

    # 合并用户参数
    params = {**default_params, **kwargs}

    return RingModel(model_type, config, topo_type, **params)


# 使用示例
if __name__ == "__main__":
    # 创建8节点Ring，使用负载均衡路由
    ring_model = create_ring_model(num_nodes=8, routing_strategy="load_balanced", traffic_file_path="../../test_data")

    # 运行仿真
    print("Starting Ring simulation...")
    results = ring_model.run()

    # 打印结果
    ring_model.print_ring_summary()

    # 获取详细统计
    stats = ring_model.get_ring_performance_stats()
    print(f"\nDetailed stats: {stats}")
