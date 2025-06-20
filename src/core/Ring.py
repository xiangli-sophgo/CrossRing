"""
Ring拓扑建模类 - 复用现有Flit和traffic处理组件
主要复用：
1. Flit类：完全复用现有的Flit数据结构
2. TrafficScheduler：复用traffic文件解析和调度
3. IPInterface：复用IP接口的频率转换和缓冲机制
4. Network基础架构：复用网络基础组件
"""

from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.utils.component import Flit, Network, IPInterface, Node
from src.core.traffic_scheduler import TrafficScheduler
from config.config import CrossRingConfig


class RingConfig(CrossRingConfig):
    """Ring拓扑配置 - 继承并扩展CrossRingConfig"""

    def __init__(self):
        super().__init__(default_config=r"../../config/config2.json")

        # Ring特有参数
        self.NUM_NODE = 16  # Ring节点数量
        self.RING_BUFFER_DEPTH = 2  # 环路转发缓冲深度
        self.ENABLE_ADAPTIVE_ROUTING = False  # 自适应路由
        self.CONGESTION_THRESHOLD = 0.7  # 拥塞阈值

        # Ring路由表缓存
        self._ring_routes_cache = {}

    def calculate_ring_distance(self, src: int, dst: int) -> Tuple[str, int]:
        """计算Ring拓扑的最短路径方向和距离"""
        if src == dst:
            return "LOCAL", 0

        # 顺时针距离
        cw_dist = (dst - src) % self.NUM_NODE
        # 逆时针距离
        ccw_dist = (src - dst) % self.NUM_NODE

        if cw_dist <= ccw_dist:
            return "CW", cw_dist
        else:
            return "CCW", ccw_dist

    def get_ring_routes(self):
        """生成Ring拓扑的路由表"""
        if self._ring_routes_cache:
            return self._ring_routes_cache

        routes = {}
        for src in range(self.NUM_NODE):
            routes[src] = {}
            for dst in range(self.NUM_NODE):
                if src == dst:
                    routes[src][dst] = [src]
                else:
                    direction, distance = self.calculate_ring_distance(src, dst)
                    path = [src]
                    current = src

                    for _ in range(distance):
                        if direction == "CW":
                            current = (current + 1) % self.NUM_NODE
                        else:  # CCW
                            current = (current - 1) % self.NUM_NODE
                        path.append(current)

                    routes[src][dst] = path

        self._ring_routes_cache = routes
        return routes


class RingNode:
    """Ring节点 - 复用现有Network的FIFO结构"""

    def __init__(self, node_id: int, config: RingConfig):
        self.node_id = node_id
        self.config = config

        # 注入队列 - 两个方向
        self.inject_queues = {
            "CW": deque(maxlen=config.IQ_OUT_FIFO_DEPTH),
            "CCW": deque(maxlen=config.IQ_OUT_FIFO_DEPTH),
        }

        # 弹出队列 - 两个方向来的flit
        self.eject_queues = {
            "CW": deque(maxlen=config.EQ_IN_FIFO_DEPTH),
            "CCW": deque(maxlen=config.EQ_IN_FIFO_DEPTH),
        }

        # 统一的弹出缓冲
        self.eject_output_buffer = deque(maxlen=config.EQ_CH_FIFO_DEPTH)

        # 环路转发缓冲 - 这是Ring特有的
        self.ring_buffers = {
            "CW": deque(maxlen=config.RING_BUFFER_DEPTH),
            "CCW": deque(maxlen=config.RING_BUFFER_DEPTH),
        }

        # 拥塞统计
        self.congestion_stats = {
            "CW": 0,
            "CCW": 0,
        }

        # Round-robin仲裁状态
        self.rr_state = {
            "inject_priority": ["CW", "CCW"],
            "eject_priority": ["CW", "CCW"],
        }

        # IP连接信息
        self.connected_ip_type = None
        self.ip_interface = None


class RingNetwork(Network):
    """Ring网络 - 继承Network基类"""

    def __init__(self, config: RingConfig, name="ring_network"):
        # 创建Ring拓扑的邻接矩阵
        adjacency_matrix = self._create_ring_adjacency_matrix(config.NUM_NODE)
        super().__init__(config, adjacency_matrix, name)

        self.ring_nodes: List[RingNode] = []
        self.routes = config.get_ring_routes()

        # 初始化Ring节点
        for i in range(config.NUM_NODE):
            ring_node = RingNode(i, config)
            self.ring_nodes.append(ring_node)

        # 链路状态 - 每个方向每个链路的flit
        self.ring_links = {
            "CW": [None] * config.NUM_NODE,  # 顺时针链路
            "CCW": [None] * config.NUM_NODE,  # 逆时针链路
        }

    def _create_ring_adjacency_matrix(self, num_nodes: int):
        """创建Ring拓扑邻接矩阵"""
        matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for i in range(num_nodes):
            # 顺时针连接
            matrix[i][(i + 1) % num_nodes] = 1
            # 逆时针连接
            matrix[i][(i - 1) % num_nodes] = 1
        return matrix


class RingTopology:
    """Ring拓扑主类 - 复用现有架构"""

    def __init__(self, config: RingConfig, traffic_file_path: str):
        self.config = config
        self.traffic_file_path = traffic_file_path
        self.current_cycle = 0

        # 创建Ring网络 - 复用Network架构
        self.ring_network = RingNetwork(config, "ring_net")

        # 复用TrafficScheduler来处理traffic文件
        self.traffic_scheduler = TrafficScheduler(config, traffic_file_path)

        # 复用Node进行packet_id管理
        self.node = Node(config)

        # IP模块字典 - 复用IPInterface
        self.ip_modules = {}
        self._setup_ip_connections()

        # 统计信息
        self.stats = {
            "total_flits_injected": 0,
            "total_flits_ejected": 0,
            "total_hops": 0,
            "congestion_events": 0,
            "adaptive_routing_events": 0,
            "cw_usage": 0,
            "ccw_usage": 0,
        }

    def _setup_ip_connections(self):
        """设置IP连接 - 复用IPInterface"""
        # 根据配置设置IP位置
        ip_positions = {
            "ddr": [0, 4, 8, 12],
            "l2m": [2, 6, 10, 14],
            "sdma": [1, 5, 9, 13],
            "gdma": [3, 7, 11, 15],
        }

        for ip_type, positions in ip_positions.items():
            for pos in positions:
                if pos < self.config.NUM_NODE:
                    # 创建IPInterface - 复用现有实现
                    ip_interface = IPInterface(
                        ip_type=ip_type,
                        ip_pos=pos,
                        config=self.config,
                        req_network=self.ring_network,
                        rsp_network=self.ring_network,  # Ring共用一个网络
                        data_network=self.ring_network,
                        node=self.node,
                        routes=self.ring_network.routes,
                    )

                    self.ip_modules[(ip_type, pos)] = ip_interface
                    self.ring_network.ring_nodes[pos].connected_ip_type = ip_type
                    self.ring_network.ring_nodes[pos].ip_interface = ip_interface

    def setup_traffic(self, traffic_config):
        """设置traffic - 复用TrafficScheduler"""
        if isinstance(traffic_config, list):
            self.traffic_scheduler.setup_parallel_chains(traffic_config)
        else:
            self.traffic_scheduler.setup_single_chain([traffic_config])

        self.traffic_scheduler.start_initial_traffics()

    def adaptive_routing_decision(self, flit: Flit) -> str:
        """自适应路由决策"""
        if not self.config.ENABLE_ADAPTIVE_ROUTING:
            direction, _ = self.config.calculate_ring_distance(flit.source, flit.destination)
            return direction

        source_node = self.ring_network.ring_nodes[flit.source]

        # 计算两个方向的拥塞程度
        cw_congestion = self._calculate_direction_congestion(source_node, "CW")
        ccw_congestion = self._calculate_direction_congestion(source_node, "CCW")

        # 如果拥塞差异显著，选择较不拥塞的方向
        if abs(cw_congestion - ccw_congestion) > self.config.CONGESTION_THRESHOLD:
            if cw_congestion > ccw_congestion:
                self.stats["adaptive_routing_events"] += 1
                return "CCW"
            else:
                self.stats["adaptive_routing_events"] += 1
                return "CW"

        # 否则选择最短路径
        direction, _ = self.config.calculate_ring_distance(flit.source, flit.destination)
        return direction

    def _calculate_direction_congestion(self, node: RingNode, direction: str) -> float:
        """计算指定方向的拥塞程度"""
        inject_occupancy = len(node.inject_queues[direction]) / node.inject_queues[direction].maxlen
        ring_occupancy = len(node.ring_buffers[direction]) / node.ring_buffers[direction].maxlen
        return (inject_occupancy + ring_occupancy) / 2.0

    def step_simulation(self):
        """执行一个仿真周期"""
        self.current_cycle += 1
        self.traffic_scheduler.current_cycle = self.current_cycle

        # 1. 处理新请求注入 - 复用traffic处理流程
        self._process_new_requests()

        # 2. IP接口处理 - 复用IPInterface的周期处理
        self._process_ip_interfaces()

        # 3. Ring网络传输
        self._process_ring_transmission()

        # 4. 处理弹出队列
        self._process_eject_queues()

        # 5. 更新统计信息
        self._update_statistics()

    def _process_new_requests(self):
        """处理新请求 - 复用TrafficScheduler"""
        ready_requests = self.traffic_scheduler.get_ready_requests(self.current_cycle)

        for req_data in ready_requests:
            # 解析请求数据
            source = req_data[1]  # 直接使用Ring节点ID
            destination = req_data[3]
            traffic_id = req_data[7] if len(req_data) > 7 else "default"

            # 创建Flit - 复用现有Flit类
            path = self.ring_network.routes[source][destination]
            flit = Flit(source, destination, path)

            # 设置flit属性 - 复用现有字段
            flit.source_original = req_data[1]
            flit.destination_original = req_data[3]
            flit.flit_type = "req"
            flit.departure_cycle = req_data[0]
            flit.burst_length = req_data[6]
            flit.source_type = req_data[2]
            flit.destination_type = req_data[4]
            flit.req_type = "read" if req_data[5] == "R" else "write"
            flit.packet_id = Node.get_next_packet_id()
            flit.traffic_id = traffic_id

            # 自适应路由决策
            direction = self.adaptive_routing_decision(flit)
            flit.ring_direction = direction  # 添加Ring特有字段

            # 通过IPInterface注入
            ip_type = flit.source_type
            ip_pos = flit.source
            if (ip_type, ip_pos) in self.ip_modules:
                ip_interface = self.ip_modules[(ip_type, ip_pos)]
                ip_interface.enqueue(flit, "req")

            self.stats["total_flits_injected"] += 1

    def _process_ip_interfaces(self):
        """处理IP接口 - 复用IPInterface的周期处理"""
        for (ip_type, ip_pos), ip_interface in self.ip_modules.items():
            ip_interface.current_cycle = self.current_cycle

            # 处理各个网络类型的接口
            for net_type in ["req", "rsp", "data"]:
                ip_interface.inject_to_l2h(net_type)
                ip_interface.l2h_to_IQ_channel_buffer(net_type)
                ip_interface.EQ_to_h2l(net_type)
                ip_interface.h2l_to_eject(net_type)

    def _process_ring_transmission(self):
        """处理Ring传输"""
        # 1. 从IQ_channel_buffer注入到Ring
        self._inject_from_IQ_to_ring()

        # 2. Ring链路传输
        self._move_flits_on_ring()

        # 3. 从Ring弹出到EQ
        self._eject_from_ring_to_EQ()

    def _inject_from_IQ_to_ring(self):
        """从IQ注入到Ring"""
        for node in self.ring_network.ring_nodes:
            if not node.ip_interface:
                continue

            ip_interface = node.ip_interface
            ip_type = node.connected_ip_type

            # 检查IQ_channel_buffer中的flit
            buffer = self.ring_network.IQ_channel_buffer[ip_type][node.node_id]
            if not buffer:
                continue

            flit = buffer[0]

            # 根据路由方向选择注入队列
            direction = getattr(flit, "ring_direction", None)
            if not direction:
                direction = self.adaptive_routing_decision(flit)
                flit.ring_direction = direction

            # 尝试注入到对应方向的队列
            inject_queue = node.inject_queues[direction]
            if len(inject_queue) < inject_queue.maxlen:
                buffer.popleft()
                inject_queue.append(flit)
                flit.inject_ring_cycle = self.current_cycle

    def _move_flits_on_ring(self):
        """Ring链路上的flit移动"""
        new_links = {
            "CW": [None] * self.config.NUM_NODE,
            "CCW": [None] * self.config.NUM_NODE,
        }

        # 1. 从注入队列到链路
        for node in self.ring_network.ring_nodes:
            for direction in ["CW", "CCW"]:
                if not node.inject_queues[direction]:
                    continue

                # 获取下一个链路位置
                next_link_idx = self._get_next_link_index(node.node_id, direction)

                # 检查链路是否空闲
                if self.ring_network.ring_links[direction][next_link_idx] is None:
                    flit = node.inject_queues[direction].popleft()
                    new_links[direction][next_link_idx] = flit

                    if direction == "CW":
                        self.stats["cw_usage"] += 1
                    else:
                        self.stats["ccw_usage"] += 1

        # 2. 链路上的flit移动
        for direction in ["CW", "CCW"]:
            for i, flit in enumerate(self.ring_network.ring_links[direction]):
                if flit is None:
                    continue

                # 计算下一个位置
                next_idx = self._get_next_link_index(i, direction)

                # 检查是否到达目标节点
                target_node = self.ring_network.ring_nodes[next_idx]
                if flit.destination == target_node.node_id:
                    # 尝试弹出到目标节点
                    eject_queue = target_node.eject_queues[direction]
                    if len(eject_queue) < eject_queue.maxlen:
                        eject_queue.append(flit)
                        flit.eject_ring_cycle = self.current_cycle
                        self.stats["total_flits_ejected"] += 1
                        # 计算跳数
                        hops = abs(self.current_cycle - flit.inject_ring_cycle)
                        self.stats["total_hops"] += hops
                    else:
                        # 目标节点拥塞，flit留在原位
                        new_links[direction][i] = flit
                        target_node.congestion_stats[direction] += 1
                        self.stats["congestion_events"] += 1
                else:
                    # 继续传输
                    if new_links[direction][next_idx] is None:
                        new_links[direction][next_idx] = flit
                    else:
                        # 下游拥塞，flit留在原位
                        new_links[direction][i] = flit

        # 更新链路状态
        self.ring_network.ring_links = new_links

    def _eject_from_ring_to_EQ(self):
        """从Ring弹出到EQ_channel_buffer"""
        for node in self.ring_network.ring_nodes:
            if not node.ip_interface:
                continue

            ip_type = node.connected_ip_type

            # Round-robin仲裁两个方向的弹出队列
            for direction in node.rr_state["eject_priority"]:
                eject_queue = node.eject_queues[direction]
                if not eject_queue:
                    continue

                # 尝试移动到EQ_channel_buffer
                eq_buffer = self.ring_network.EQ_channel_buffer[ip_type][node.node_id]
                if len(eq_buffer) < self.config.EQ_CH_FIFO_DEPTH:
                    flit = eject_queue.popleft()
                    eq_buffer.append(flit)

                    # 更新round-robin优先级
                    node.rr_state["eject_priority"].remove(direction)
                    node.rr_state["eject_priority"].append(direction)
                    break

    def _process_eject_queues(self):
        """处理弹出队列 - 复用IPInterface的EQ处理"""
        # IPInterface会自动处理EQ_channel_buffer到eject_fifo的传输
        pass

    def _get_next_link_index(self, current_idx: int, direction: str) -> int:
        """获取下一个链路索引"""
        if direction == "CW":
            return (current_idx + 1) % self.config.NUM_NODE
        else:  # CCW
            return (current_idx - 1) % self.config.NUM_NODE

    def _update_statistics(self):
        """更新统计信息"""
        if self.stats["total_flits_ejected"] > 0:
            self.stats["average_latency"] = self.stats["total_hops"] / self.stats["total_flits_ejected"]

        if self.current_cycle > 0:
            self.stats["throughput"] = self.stats["total_flits_ejected"] / self.current_cycle
            self.stats["cw_utilization"] = self.stats["cw_usage"] / self.current_cycle
            self.stats["ccw_utilization"] = self.stats["ccw_usage"] / self.current_cycle

    def get_network_status(self) -> Dict:
        """获取网络状态"""
        return {
            "current_cycle": self.current_cycle,
            "statistics": self.stats.copy(),
            "node_status": [
                {
                    "node_id": node.node_id,
                    "connected_ip": node.connected_ip_type,
                    "inject_queues": {"CW": len(node.inject_queues["CW"]), "CCW": len(node.inject_queues["CCW"])},
                    "eject_queues": {"CW": len(node.eject_queues["CW"]), "CCW": len(node.eject_queues["CCW"])},
                    "congestion": node.congestion_stats.copy(),
                }
                for node in self.ring_network.ring_nodes
            ],
        }

    def run_simulation(self, max_cycles: int = 10000) -> Dict:
        """运行完整仿真"""
        for cycle in range(max_cycles):
            self.step_simulation()

            # 检查是否所有traffic都完成
            if self.traffic_scheduler.are_all_traffics_completed():
                break

            if cycle % 1000 == 0 and cycle > 0:
                print(f"Cycle {cycle}: Injected={self.stats['total_flits_injected']}, " f"Ejected={self.stats['total_flits_ejected']}")

        print(f"Simulation completed at cycle {self.current_cycle}")
        return self.get_network_status()


# 使用示例
if __name__ == "__main__":
    # 创建配置
    config = RingConfig()
    config.NUM_NODE = 16

    # 创建Ring拓扑
    ring = RingTopology(config, r"../../traffic/0617")

    # 设置traffic - 复用现有traffic文件格式
    traffic_config = [["Write_burst4_2262HBM_v2.txt"]]
    ring.setup_traffic(traffic_config)

    # 运行仿真
    results = ring.run_simulation(max_cycles=5000)

    print(f"最终统计:")
    print(f"  注入flit数: {results['statistics']['total_flits_injected']}")
    print(f"  弹出flit数: {results['statistics']['total_flits_ejected']}")
    print(f"  平均延迟: {results['statistics'].get('average_latency', 0):.2f}")
    print(f"  吞吐量: {results['statistics'].get('throughput', 0):.3f}")
    print(f"  自适应路由事件: {results['statistics']['adaptive_routing_events']}")
    print(f"  顺时针利用率: {results['statistics'].get('cw_utilization', 0):.3f}")
    print(f"  逆时针利用率: {results['statistics'].get('ccw_utilization', 0):.3f}")
