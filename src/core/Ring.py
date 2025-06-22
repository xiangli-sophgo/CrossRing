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
import time
from src.utils.component import Flit, Network, IPInterface, Node
from src.core.traffic_scheduler import TrafficScheduler
from src.core.result_processor import BandwidthAnalyzer
from config.config import CrossRingConfig


class RingConfig(CrossRingConfig):
    """Ring拓扑配置 - 继承并扩展CrossRingConfig"""

    def __init__(self, default_config=None):
        if default_config is None:
            # 尝试不同的相对路径
            import os

            possible_paths = ["../../config/config2.json", "../config/config2.json", "config/config2.json", "/Users/lixiang/Documents/工作/CrossRing/config/config2.json"]
            for path in possible_paths:
                if os.path.exists(path):
                    default_config = path
                    break
            if default_config is None:
                raise FileNotFoundError("Could not find config2.json in any expected location")
        super().__init__(default_config)

        # Ring特有参数
        self.NUM_RING_NODES = 8  # Ring节点数量
        self.NUM_NODE = self.NUM_RING_NODES  # 重写父类的NUM_NODE以匹配Ring节点数
        self.RING_BUFFER_DEPTH = 8  # 环路转发缓冲深度，增加以提高吞吐量
        self.ENABLE_ADAPTIVE_ROUTING = False  # 自适应路由
        self.CONGESTION_THRESHOLD = 0.7  # 拥塞阈值

        # 重写CHANNEL_SPEC以支持Ring拓扑中的多IP实例
        self.CHANNEL_SPEC = {
            "gdma": min(2, self.NUM_RING_NODES // 4),  # 每4个节点一个GDMA
            "sdma": min(2, self.NUM_RING_NODES // 8),  # 每8个节点一个SDMA
            "ddr": min(2, self.NUM_RING_NODES // 4),  # 每4个节点一个DDR
            "l2m": min(2, self.NUM_RING_NODES // 8),  # 每8个节点一个L2M
        }

        # 重新生成CH_NAME_LIST
        self.CH_NAME_LIST = []
        for key in self.CHANNEL_SPEC:
            for idx in range(self.CHANNEL_SPEC[key]):
                self.CH_NAME_LIST.append(f"{key}_{idx}")

        # 复用现有FIFO深度配置
        # IQ_CH_FIFO_DEPTH, IQ_OUT_FIFO_DEPTH 等已在父类定义

        # 重写IP位置列表以匹配Ring拓扑，分布到不同节点
        self.GDMA_SEND_POSITION_LIST = list(range(self.NUM_RING_NODES))

        self.DDR_SEND_POSITION_LIST = list(range(self.NUM_RING_NODES))

        self.SDMA_SEND_POSITION_LIST = list(range(self.NUM_RING_NODES))

        self.L2M_SEND_POSITION_LIST = list(range(self.NUM_RING_NODES))

        self.CDMA_SEND_POSITION_LIST = []  # Ring拓扑暂不使用CDMA

        # Ring路由表缓存
        self._ring_routes_cache = {}

        # 节点映射：将原始拓扑的节点映射到Ring拓扑
        self.node_mapping = self._create_node_mapping()

        # 新增：是否在到达目的地时本地弹出，否则绕环
        self.RING_LOCAL_EJECT = True

    def _create_node_mapping(self):
        """创建从原始拓扑节点到Ring拓扑节点的映射"""
        # 根据traffic文件中的IP类型，将节点映射到对应的Ring位置
        mapping = {}

        ddr_positions = list(range(0, self.NUM_RING_NODES))
        l2m_positions = list(range(0, self.NUM_RING_NODES))
        sdma_positions = list(range(0, self.NUM_RING_NODES))
        gdma_positions = list(range(0, self.NUM_RING_NODES))

        # 简单均匀分布映射：0-7 -> Ring positions
        all_positions = []
        all_positions.extend(ddr_positions)
        all_positions.extend(l2m_positions)
        all_positions.extend(sdma_positions)
        all_positions.extend(gdma_positions)
        all_positions.sort()

        # 映射0-7到前8个位置
        for i in range(self.NUM_RING_NODES):
            mapping[i] = all_positions[i]

        return mapping

    def map_node(self, original_node: int) -> int:
        """将原始拓扑节点映射到Ring拓扑节点"""
        return self.node_mapping.get(original_node, original_node % self.NUM_RING_NODES)

    def calculate_ring_distance(self, src: int, dst: int) -> Tuple[str, int]:
        """计算Ring拓扑的最短路径方向和距离"""
        if src == dst:
            return "LOCAL", 0

        # 顺时针距离
        cw_dist = (dst - src) % self.NUM_RING_NODES
        # 逆时针距离
        ccw_dist = (src - dst) % self.NUM_RING_NODES

        if cw_dist <= ccw_dist:
            return "CW", cw_dist
        else:
            return "CCW", ccw_dist

    def get_ring_link_slices(self, src: int, dst: int) -> int:
        """获取Ring拓扑中两个相邻节点之间的slice数

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            int: 链路上的slice数，与SLICE_PER_LINK相同
        """
        # 检查是否为相邻节点
        next_cw = (src + 1) % self.NUM_RING_NODES
        prev_ccw = (src - 1) % self.NUM_RING_NODES

        if dst == next_cw or dst == prev_ccw:
            return self.SLICE_PER_LINK
        else:
            # 非相邻节点，返回路径上所有链路的slice数总和
            direction, distance = self.calculate_ring_distance(src, dst)
            return distance * self.SLICE_PER_LINK

    def get_ring_routes(self):
        """生成Ring拓扑的路由表"""
        if self._ring_routes_cache:
            return self._ring_routes_cache

        routes = {}
        for src in range(self.NUM_RING_NODES):
            routes[src] = {}
            for dst in range(self.NUM_RING_NODES):
                if src == dst:
                    routes[src][dst] = [src]
                else:
                    direction, distance = self.calculate_ring_distance(src, dst)
                    path = [src]
                    current = src

                    for _ in range(distance):
                        if direction == "CW":
                            current = (current + 1) % self.NUM_RING_NODES
                        else:  # CCW
                            current = (current - 1) % self.NUM_RING_NODES
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

        # Round-robin仲裁状态 - 增强版本
        self.rr_state = {
            "inject_priority": ["CW", "CCW"],
            "eject_priority": ["CW", "CCW"],
        }

        # 新增的仲裁状态
        self.inject_arbitration_state = {"channel_priority": [], "last_served": {}}  # 将在_setup_ip_connections中初始化
        self.eject_arbitration_state = {"direction_priority": ["CW", "CCW"], "channel_assignment": {}, "last_served_direction": None}

        # IP连接信息
        self.connected_ip_type = None
        self.ip_interface = None


class RingNetwork(Network):
    """Ring网络 - 继承Network基类"""

    def __init__(self, config: RingConfig, name="ring_network"):
        # 创建Ring拓扑的邻接矩阵
        adjacency_matrix = self._create_ring_adjacency_matrix(config.NUM_RING_NODES)
        super().__init__(config, adjacency_matrix, name)

        self.ring_nodes: List[RingNode] = []
        self.routes = config.get_ring_routes()

        # 重新初始化IQ和EQ channel buffers以使用Ring的CH_NAME_LIST
        self._initialize_ring_channel_buffers(config)

        # 初始化Ring节点
        for i in range(config.NUM_RING_NODES):
            ring_node = RingNode(i, config)
            self.ring_nodes.append(ring_node)

        # 链路状态 - 每个方向每个链路的slice级flit存储
        # 每个链路有SLICE_PER_LINK个slice位置
        self.ring_links = {
            "CW": [[None] * config.SLICE_PER_LINK for _ in range(config.NUM_RING_NODES)],  # 顺时针链路
            "CCW": [[None] * config.SLICE_PER_LINK for _ in range(config.NUM_RING_NODES)],  # 逆时针链路
        }

        # 流量统计
        self.links_flow_stat = {
            "read": {},  # 读流量 {(src, dst): count}
            "write": {},  # 写流量 {(src, dst): count}
        }

    def _initialize_ring_channel_buffers(self, config: RingConfig):
        """为Ring拓扑重新初始化channel buffers"""
        from collections import deque

        # 清除父类创建的buffers
        self.IQ_channel_buffer = {}
        self.IQ_channel_buffer_pre = {}
        self.EQ_channel_buffer = {}
        self.EQ_channel_buffer_pre = {}

        # 使用Ring的CH_NAME_LIST初始化buffers
        for ip_type in config.CH_NAME_LIST:
            self.IQ_channel_buffer[ip_type] = {}
            self.IQ_channel_buffer_pre[ip_type] = {}
            self.EQ_channel_buffer[ip_type] = {}
            self.EQ_channel_buffer_pre[ip_type] = {}

            for node_id in range(config.NUM_RING_NODES):
                self.IQ_channel_buffer[ip_type][node_id] = deque(maxlen=config.IQ_CH_FIFO_DEPTH)
                self.IQ_channel_buffer_pre[ip_type][node_id] = None
                self.EQ_channel_buffer[ip_type][node_id] = deque(maxlen=config.EQ_CH_FIFO_DEPTH)
                self.EQ_channel_buffer_pre[ip_type][node_id] = None

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

        # 创建三个独立的 Ring 网络，分别对应 req、rsp、data 通道
        self.networks = {
            "req": RingNetwork(config, "req_ring"),
            "rsp": RingNetwork(config, "rsp_ring"),
            "data": RingNetwork(config, "data_ring"),
        }

        # 复用TrafficScheduler来处理traffic文件
        self.traffic_scheduler = TrafficScheduler(config, traffic_file_path)

        # 复用Node进行packet_id管理
        self.node = Node(config)

        # IP模块字典 - 复用IPInterface
        self.ip_modules = {}
        self._setup_ip_connections()

        # 添加send_flits缓存：按通道保存所有发送的flit
        self.send_flits = {"req": [], "rsp": [], "data": []}  # 请求通道  # 响应通道  # 数据通道

        # 统计信息
        self.stats = {
            "total_flits_injected": 0,
            "total_flits_ejected": 0,
            "total_requests_ejected": 0,  # 请求数量
            "total_data_flits_ejected": 0,  # 数据flit数量
            "total_hops": 0,
            "congestion_events": 0,
            "adaptive_routing_events": 0,
            "cw_usage": 0,
            "ccw_usage": 0,
        }

        # 结果处理器
        self.bandwidth_analyzer = None

        # Debug追踪功能 - 简化版本
        self.debug_packet_ids = set()  # 要追踪的packet_id集合
        self.debug_enabled = False
        self.real_time_debug = False  # 实时debug显示
        self.debug_delay = 0.5  # debug延迟(秒)

    def _setup_ip_connections(self):
        """
        为 **每个 Ring 节点** 挂接 `CH_NAME_LIST` 中的所有 IP 实例，并生成
        (ip_type_with_index, node_id) → IPInterface 映射。

        这意味着 *每个* node 都有 gdma_0…gdma_N, sdma_0… 等完整一套 IP。
        """
        for node_id in range(self.config.NUM_RING_NODES):
            # 由于我们有三个独立的网络，每个节点的RingNode对象应从req网络获取（因为节点对象是独立的，但结构上相同）
            ring_node = self.networks["req"].ring_nodes[node_id]

            # 初始化列表 / 字典
            if ring_node.connected_ip_type is None:
                ring_node.connected_ip_type = []
            if ring_node.ip_interface is None:
                ring_node.ip_interface = {}

            for ip_name in self.config.CH_NAME_LIST:
                ip_interface = IPInterface(
                    ip_type=ip_name,
                    ip_pos=node_id,
                    config=self.config,
                    req_network=self.networks["req"],
                    rsp_network=self.networks["rsp"],
                    data_network=self.networks["data"],
                    node=self.node,
                    routes=self.networks["req"].routes,
                )

                # 写入全局表
                self.ip_modules[(ip_name, node_id)] = ip_interface

                # 记录到节点
                ring_node.connected_ip_type.append(ip_name)
                ring_node.ip_interface[ip_name] = ip_interface

            # 初始化仲裁状态的 channel_priority（保持与 connected_ip_type 顺序一致）
            ring_node.inject_arbitration_state["channel_priority"] = list(ring_node.connected_ip_type)
            ring_node.eject_arbitration_state["channel_assignment"] = {}

    def _find_injection_point(self, preferred_node: int, ip_type: str) -> Optional[int]:
        """
        如果 (ip_type, preferred_node) 存在，则返回 preferred_node，
        否则返回 None（Ring 流量必须从正确的节点注入）。
        """
        return preferred_node if (ip_type, preferred_node) in self.ip_modules else None

    def _find_injection_point_by_ip_type(self, ip_type: str, node_id: int) -> Optional[int]:
        """
        检查指定 node_id 是否挂载了 ip_type 接口。
        只有当 (ip_type, node_id) 在 self.ip_modules 中时返回 node_id，否则返回 None。
        """
        return node_id if (ip_type, node_id) in self.ip_modules else None

    def setup_traffic(self, traffic_config):
        """设置traffic - 复用TrafficScheduler，并添加节点映射"""
        if isinstance(traffic_config, list):
            self.traffic_scheduler.setup_parallel_chains(traffic_config)
        else:
            self.traffic_scheduler.setup_single_chain([traffic_config])

        # 为TrafficScheduler添加节点映射功能
        self.traffic_scheduler.ring_config = self.config
        self.traffic_scheduler.use_ring_mapping = True

        self.traffic_scheduler.start_initial_traffics()

    def initialize_result_processor(self):
        """初始化结果处理器"""
        if self.bandwidth_analyzer is None:
            self.bandwidth_analyzer = BandwidthAnalyzer(self.config)
        # 让 analyzer 知道 base_model（用于计算 IP 带宽和其他属性）
        self.bandwidth_analyzer.base_model = self

    def process_results(self, plot_rn=True, plot_flow=True, save_path=None):
        """处理结果并生成可视化"""
        if self.bandwidth_analyzer is None:
            self.initialize_result_processor()

        # 设置结果处理器的数据
        self.bandwidth_analyzer.finish_cycle = self.current_cycle

        # 传递统计信息到result processor
        self.networks["data"].stats = self.stats

        total_bw = 0
        if plot_rn:
            total_bw = self.bandwidth_analyzer.plot_rn_bandwidth_curves()

        if plot_flow:
            self.bandwidth_analyzer.draw_flow_graph_ring(self.networks["req"], save_path=save_path)

        return total_bw

    def adaptive_routing_decision(self, flit: Flit) -> str:
        """自适应路由决策"""
        if not self.config.ENABLE_ADAPTIVE_ROUTING:
            direction, _ = self.config.calculate_ring_distance(flit.source, flit.destination)
            return direction

        source_node = self.networks["req"].ring_nodes[flit.source]

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

        # 5. 检查并推进traffic链状态
        self.traffic_scheduler.check_and_advance_chains(self.current_cycle)

        # 6. 更新统计信息
        self._update_statistics()

        # 7. Debug追踪
        self._trace_packet_locations()

        # Debug send_flits counts summary
        # if self.debug_enabled:
        #     print("Debug send_flits counts:", {ch: len(self.send_flits[ch]) for ch in self.send_flits})

    def _process_new_requests(self):
        """处理新请求 - 复用TrafficScheduler"""
        ready_requests = self.traffic_scheduler.get_ready_requests(self.current_cycle)

        for req_data in ready_requests:
            # 解析请求数据
            source = req_data[1]  # 原始节点ID
            destination = req_data[3]  # 原始节点ID
            source_type = req_data[2]  # e.g., "gdma_0"
            dest_type = req_data[4]  # e.g., "ddr_0"
            traffic_id = req_data[7] if len(req_data) > 7 else "default"

            injection_point = self._find_injection_point_by_ip_type(source_type, source)
            if injection_point is None:
                print(f"Warning: Cannot find injection point for {source_type} at node {source}")
                continue  # 跳过无法注入的请求

            # 查找目标IP的位置
            destination_point = self._find_injection_point_by_ip_type(dest_type, destination)
            if destination_point is None:
                print(f"Warning: Cannot find destination point for {dest_type} at node {destination}")
                continue

            # 创建Flit - 使用找到的injection_point和destination_point
            path = self.networks["req"].routes[injection_point][destination_point]
            flit = Flit(injection_point, destination_point, path)

            # 设置flit属性 - 复用现有字段
            flit.source_original = source
            flit.destination_original = destination
            flit.flit_type = "req"
            flit.departure_cycle = req_data[0]
            flit.burst_length = req_data[6]
            flit.source_type = source_type
            flit.destination_type = dest_type
            flit.original_destination_type = dest_type  # 添加缺失的属性
            flit.original_source_type = source_type  # 添加缺失的属性
            flit.req_type = "read" if req_data[5] == "R" else "write"
            flit.packet_id = Node.get_next_packet_id()
            flit.traffic_id = traffic_id

            # 自适应路由决策
            direction = self.adaptive_routing_decision(flit)
            flit.ring_direction = direction  # 添加Ring特有字段

            # 通过IPInterface注入
            ip_pos = injection_point
            # 查找injection_point的IP接口
            ip_interface = None

            ip_interface = self.ip_modules[(source_type, ip_pos)]

            if ip_interface:
                # 注入请求到网络
                success = ip_interface.enqueue(flit, "req")
                if success:
                    self.stats["total_flits_injected"] += 1
                    # 记录此次请求注入，用于debug追踪
                    self.send_flits["req"].append(flit)
                    # self.traffic_scheduler.update_traffic_stats(flit.traffic_id, "sent_flit")
                else:
                    # 注入失败，可能是队列满了
                    continue
            else:
                # 找不到合适的IP接口，跳过这个请求
                print(f"Warning: Cannot find IP interface for {source_type} at node {ip_pos}")

    def _process_ip_interfaces(self):
        """处理IP接口 - 复用IPInterface的频率处理逻辑"""
        # 1. 处理inject步骤 - 包含1GHz和2GHz操作
        for (ip_type, ip_pos), ip_interface in self.ip_modules.items():
            ip_interface.inject_step(self.current_cycle)

        # 2. 处理pre_to_fifo移动 - 每个周期都执行
        for (ip_type, ip_pos), ip_interface in self.ip_modules.items():
            ip_interface.move_pre_to_fifo()

        # 3. 处理eject步骤 - 让IPInterface自行处理请求、响应和数据
        self.ejected_flits = []
        for (ip_type, ip_pos), ip_interface in self.ip_modules.items():
            ejected_flits = ip_interface.eject_step(self.current_cycle)
            if ejected_flits:
                self.ejected_flits.extend(ejected_flits)

                # 更新traffic完成统计
                for flit in ejected_flits:
                    if hasattr(flit, "traffic_id") and flit.flit_type == "data":
                        self.traffic_scheduler.update_traffic_stats(flit.traffic_id, "received_flit")

                    # 区分请求和数据统计
                    if hasattr(flit, "flit_type"):
                        if flit.flit_type in ["req", "rsp"]:
                            self.stats["total_requests_ejected"] += 1
                        elif flit.flit_type == "data":
                            self.stats["total_data_flits_ejected"] += 1

    def _process_ring_transmission(self):
        """处理Ring传输"""
        # 1. 先移动环上已有 flit（清空 slice 0）
        self._move_flits_on_ring()

        # 2. 再从IQ_channel_buffer注入到Ring（确保首 slice 为空）
        self._inject_from_IQ_to_ring()

        # 3. 从Ring弹出到EQ
        self._eject_from_ring_to_EQ()

    def _move_ring_pre_to_queues(self, node: RingNode):
        """移动Ring特有的pre缓冲到正式队列"""
        # 这里可以添加Ring特有的pre缓冲移动逻辑
        # 目前Ring直接使用network的IQ_channel_buffer，所以主要依赖IPInterface的move_pre_to_fifo
        pass

    def _inject_from_IQ_to_ring(self):
        """从IQ_channel_buffer注入到Ring - 实现n to 2仲裁机制"""
        # 对每个channel分别处理注入
        for channel in ["req", "rsp", "data"]:
            ring_network = self.networks[channel]
            # 统一使用req网络的connected_ip_type遍历所有节点
            for node_id in range(self.config.NUM_RING_NODES):
                req_node = self.networks["req"].ring_nodes[node_id]
                ip_types = req_node.connected_ip_type or []
                if not ip_types:
                    continue
                ring_node = ring_network.ring_nodes[node_id]

                # 收集所有有数据的IP channel buffer
                available_channels = []
                if hasattr(ring_network, "IQ_channel_buffer"):
                    for ip_type in ip_types:
                        if ip_type in ring_network.IQ_channel_buffer:
                            buffer = ring_network.IQ_channel_buffer[ip_type][node_id]
                            if buffer:
                                available_channels.append((ip_type, buffer))

                if not available_channels:
                    continue

                # Round-robin仲裁：为两个方向(CW/CCW)各选择一个channel
                injected_count = 0
                max_inject_per_cycle = min(4, len(available_channels))  # 增加注入带宽，最多4个flit

                # 获取节点的仲裁状态，如果不存在则初始化
                if not hasattr(ring_node, "inject_arbitration_state"):
                    ring_node.inject_arbitration_state = {"channel_priority": [ip_type for ip_type in ip_types], "last_served": {}}

                # 轮询所有可用的channel，优先处理能注入的flit
                for _ in range(max_inject_per_cycle):
                    if injected_count >= max_inject_per_cycle:
                        break

                    # 查找下一个可以注入的flit
                    selected_channel = None
                    selected_direction = None

                    for ip_type, buffer in available_channels:
                        if not buffer:  # buffer可能在前面的处理中被清空
                            continue

                        flit = buffer[0]

                        # 确定flit的路由方向
                        flit_direction = getattr(flit, "ring_direction", None)
                        if not flit_direction:
                            flit_direction = self.adaptive_routing_decision(flit)
                            flit.ring_direction = flit_direction

                        # 本地传输处理
                        if flit_direction == "LOCAL":
                            target_node = ring_network.ring_nodes[flit.destination]
                            eject_queue = target_node.eject_queues["CW"]
                            if len(eject_queue) < eject_queue.maxlen:
                                buffer.popleft()
                                eject_queue.append(flit)
                                flit.eject_ring_cycle = self.current_cycle
                                injected_count += 1  # 已完成注入计数
                                selected_channel = (ip_type, buffer)  # 标记已处理
                                break
                            else:
                                continue

                        # 检查对应方向的注入队列是否有空间
                        inject_queue = ring_node.inject_queues[flit_direction]
                        if len(inject_queue) < inject_queue.maxlen:
                            selected_channel = (ip_type, buffer)
                            selected_direction = flit_direction
                            break

                    # 如果找到合适的channel，执行注入
                    if selected_channel and selected_direction:
                        ip_type, buffer = selected_channel
                        flit = buffer.popleft()
                        inject_queue = ring_node.inject_queues[selected_direction]
                        inject_queue.append(flit)
                        flit.inject_ring_cycle = self.current_cycle
                        injected_count += 1

                        # 更新round-robin仲裁状态
                        self._update_inject_arbitration_state(ring_node, ip_type, selected_direction)

                        # 统计使用情况
                        if selected_direction == "CW":
                            self.stats["cw_usage"] += 1
                        else:
                            self.stats["ccw_usage"] += 1
                    else:
                        # 没有找到可注入的flit，退出循环
                        break

    def _move_flits_on_ring(self):
        """Ring链路上的flit移动 - 实现pre缓冲机制避免slot冲突"""
        for channel in ["req", "rsp", "data"]:
            ring_network = self.networks[channel]

            # 初始化pre缓冲区 - 存储本周期待移动的flit
            ring_network.ring_links_pre = {
                "CW": [[None] * self.config.SLICE_PER_LINK for _ in range(self.config.NUM_RING_NODES)],
                "CCW": [[None] * self.config.SLICE_PER_LINK for _ in range(self.config.NUM_RING_NODES)],
            }

            # 阶段1: 准备移动到pre缓冲区 - 计算所有移动而不立即执行
            moves_to_execute = []  # [(flit, src_info, dst_info, action_type)]

            # 1.1 从注入队列到链路第一个slice的移动
            for node in ring_network.ring_nodes:
                for direction in ["CW", "CCW"]:
                    if not node.inject_queues[direction]:
                        continue

                    # 获取下一个链路位置
                    next_link_idx = self._get_next_link_index(node.node_id, direction)
                    flit = node.inject_queues[direction][0]  # 查看但不移除

                    # 检查链路第一个slice是否空闲
                    if ring_network.ring_links[direction][next_link_idx][0] is None:
                        moves_to_execute.append((flit, ("inject", node.node_id, direction), ("link", direction, next_link_idx, 0), "inject_to_link"))

            # 1.2 链路上的flit在slice间移动
            for direction in ["CW", "CCW"]:
                for link_idx in range(self.config.NUM_RING_NODES):
                    link_slices = ring_network.ring_links[direction][link_idx]

                    # 从后往前处理slice，避免依赖冲突
                    for slice_idx in range(self.config.SLICE_PER_LINK - 1, -1, -1):
                        flit = link_slices[slice_idx]
                        if flit is None:
                            continue

                        # 检查是否在最后一个slice
                        if slice_idx == self.config.SLICE_PER_LINK - 1:
                            # 到达最后slice时，根据配置决定本地弹出还是绕环
                            if flit.destination == link_idx and self.config.RING_LOCAL_EJECT:
                                # 弹出到目标节点
                                target_node = ring_network.ring_nodes[link_idx]
                                eject_queue = target_node.eject_queues[direction]
                                if len(eject_queue) < eject_queue.maxlen:
                                    moves_to_execute.append((flit, ("link", direction, link_idx, slice_idx), ("eject", link_idx, direction), "link_to_eject"))
                                # 如果无法弹出，保持原位
                            else:
                                # 绕环或未到达目的地：移动到下一个链路的第一个slice
                                next_link_idx = self._get_next_link_index(link_idx, direction)
                                moves_to_execute.append((flit, ("link", direction, link_idx, slice_idx), ("link", direction, next_link_idx, 0), "link_to_link"))
                        else:
                            # 移动到下一个slice
                            # 总是尝试移动，冲突在阶段2检查
                            if True:
                                moves_to_execute.append((flit, ("link", direction, link_idx, slice_idx), ("link", direction, link_idx, slice_idx + 1), "slice_advance"))

            # 阶段2: 检查冲突并执行可行的移动
            executed_moves = set()  # 跟踪已执行的移动，避免重复
            destination_occupied = set()  # 跟踪已被占用的目标位置

            for i, (flit, src_info, dst_info, action_type) in enumerate(moves_to_execute):
                # 生成目标位置的唯一键
                if dst_info[0] == "link":
                    dst_key = ("link", dst_info[1], dst_info[2], dst_info[3])  # (type, direction, link_idx, slice_idx)
                elif dst_info[0] == "eject":
                    dst_key = ("eject", dst_info[1], dst_info[2])  # (type, node_idx, direction)
                else:
                    continue

                # 检查目标位置是否已被占用
                if dst_key in destination_occupied:
                    continue  # 跳过这个移动，flit保持原位

                # 执行移动
                self._execute_flit_move(flit, src_info, dst_info, action_type, ring_network)
                executed_moves.add(i)
                destination_occupied.add(dst_key)

            # 阶段3: 处理未能移动的flit（保持原位）
            for direction in ["CW", "CCW"]:
                for link_idx in range(self.config.NUM_RING_NODES):
                    link_slices = ring_network.ring_links[direction][link_idx]
                    for slice_idx in range(self.config.SLICE_PER_LINK):
                        flit = link_slices[slice_idx]
                        if flit is not None:
                            # 检查这个flit是否已经被移动
                            flit_moved = False
                            for i, (move_flit, _, _, _) in enumerate(moves_to_execute):
                                if move_flit == flit and i in executed_moves:
                                    flit_moved = True
                                    break

                            # 如果未移动，将其复制到pre缓冲区（保持原位）
                            if not flit_moved:
                                ring_network.ring_links_pre[direction][link_idx][slice_idx] = flit

            # 阶段4: 将pre缓冲区的结果复制回主链路
            ring_network.ring_links = ring_network.ring_links_pre

    def _eject_from_ring_to_EQ(self):
        """从Ring弹出到EQ_channel_buffer - 实现2 to n仲裁机制"""
        for channel in ["req", "rsp", "data"]:
            ring_network = self.networks[channel]
            for node in ring_network.ring_nodes:
                if not node.connected_ip_type:
                    continue

                # 收集所有有数据的弹出队列
                available_directions = []
                for direction in ["CW", "CCW"]:
                    eject_queue = node.eject_queues[direction]
                    if eject_queue:
                        available_directions.append((direction, eject_queue))

                if not available_directions:
                    continue

                # 获取节点的弹出仲裁状态
                if not hasattr(node, "eject_arbitration_state"):
                    node.eject_arbitration_state = {"direction_priority": ["CW", "CCW"], "channel_assignment": {}, "last_served_direction": None}  # direction -> channel mapping

                ejected_count = 0
                max_eject_per_cycle = len(node.connected_ip_type)  # 每个周期最多弹出的flit数等于IP数量

                # Round-robin仲裁：依次处理每个方向的弹出队列
                for direction, eject_queue in available_directions:
                    if ejected_count >= max_eject_per_cycle:
                        break

                    flit = eject_queue[0]

                    # 确定目标IP channel - 基于flit的目标类型和节点位置
                    target_ip_type = flit.destination_type

                    if target_ip_type and target_ip_type in ring_network.EQ_channel_buffer:
                        eq_buffer = ring_network.EQ_channel_buffer[target_ip_type][node.node_id]

                        # 检查目标channel buffer是否有空间
                        if len(eq_buffer) < self.config.EQ_CH_FIFO_DEPTH:
                            flit = eject_queue.popleft()
                            eq_buffer.append(flit)
                            ejected_count += 1

                            # 更新仲裁状态
                            self._update_eject_arbitration_state(node, direction, target_ip_type)

                            # 更新统计
                            if flit.flit_type == "data":
                                self.stats["total_flits_ejected"] += 1

    def _process_eject_queues(self):
        """
        处理弹出队列 - 将 RingNode.eject_queues 中的 flit
        注入到各自通道的 EQ_channel_buffer，以便 IPInterface.eject_step 能取出。
        """
        # 对每个通道分别处理
        for channel in ["req", "rsp", "data"]:
            ring_network = self.networks[channel]
            # 遍历所有节点
            for node_id in range(self.config.NUM_RING_NODES):
                node = ring_network.ring_nodes[node_id]
                # 两个方向的 eject_queues
                for direction in ["CW", "CCW"]:
                    queue = node.eject_queues[direction]
                    # 将所有待弹出的 flit 转入 EQ_channel_buffer
                    while queue:
                        flit = queue.popleft()
                        # 目标 IP channel 名称应该与 flit.destination_type 一致
                        ip_type = getattr(flit, "destination_type", None)
                        if ip_type and ip_type in ring_network.EQ_channel_buffer:
                            ring_network.EQ_channel_buffer[ip_type][node_id].append(flit)

    def _execute_flit_move(self, flit, src_info, dst_info, action_type, ring_network):
        """执行单个flit的移动操作"""
        # 从源位置移除flit
        if src_info[0] == "inject":
            node_id, direction = src_info[1], src_info[2]
            node = ring_network.ring_nodes[node_id]
            if node.inject_queues[direction]:
                removed_flit = node.inject_queues[direction].popleft()
                assert removed_flit == flit, "Inject queue flit mismatch"
        elif src_info[0] == "link":
            direction, link_idx, slice_idx = src_info[1], src_info[2], src_info[3]
            ring_network.ring_links[direction][link_idx][slice_idx] = None

        # 移动到目标位置
        if dst_info[0] == "link":
            direction, link_idx, slice_idx = dst_info[1], dst_info[2], dst_info[3]
            ring_network.ring_links_pre[direction][link_idx][slice_idx] = flit

            # 更新flit位置信息
            flit.current_slice = slice_idx
            flit.flit_position = "Link"
            flit.current_link = (link_idx, self._get_next_link_index(link_idx, direction))
            flit.current_seat_index = slice_idx

            # 统计和流量记录
            if action_type == "inject_to_link":
                if direction == "CW":
                    self.stats["cw_usage"] += 1
                else:
                    self.stats["ccw_usage"] += 1

                # 记录流量统计
                src_node = src_info[1] if src_info[0] == "inject" else link_idx
                if direction == "CW":
                    dst_node = (src_node + 1) % self.config.NUM_RING_NODES
                else:
                    dst_node = (src_node - 1) % self.config.NUM_RING_NODES

                flow_key = (src_node, dst_node)
                req_type = getattr(flit, "req_type", "read")
                if flow_key not in ring_network.links_flow_stat[req_type]:
                    ring_network.links_flow_stat[req_type][flow_key] = 0
                ring_network.links_flow_stat[req_type][flow_key] += 1

        elif dst_info[0] == "eject":
            node_idx, direction = dst_info[1], dst_info[2]
            target_node = ring_network.ring_nodes[node_idx]
            eject_queue = target_node.eject_queues[direction]
            eject_queue.append(flit)

            # 更新flit信息
            flit.eject_ring_cycle = self.current_cycle
            flit.flit_position = f"Ring_Eject_{direction}"
            flit.current_position = target_node.node_id

            # 统计
            if flit.flit_type == "data":
                self.stats["total_flits_ejected"] += 1
            hops = abs(self.current_cycle - getattr(flit, "inject_ring_cycle", self.current_cycle))
            self.stats["total_hops"] += hops

    def _get_next_link_index(self, current_idx: int, direction: str) -> int:
        """获取下一个链路索引"""
        if direction == "CW":
            return (current_idx + 1) % self.config.NUM_RING_NODES
        else:  # CCW
            return (current_idx - 1) % self.config.NUM_RING_NODES

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
        # 以req网络的节点为主（节点结构一致）
        nodes = self.networks["req"].ring_nodes
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
                for node in nodes
            ],
        }

    def run_simulation(self, max_cycles: int = 10000, verbose: bool = True) -> Dict:
        """运行完整仿真 - 复用BaseModel的仿真逻辑"""
        print(f"Starting Ring simulation with {self.config.NUM_RING_NODES} nodes")
        start_time = time.time()

        for cycle in range(max_cycles):
            self.step_simulation()

            # 检查是否所有traffic都完成
            if self._are_all_traffics_completed():
                if verbose:
                    print(f"All traffics completed at cycle {self.current_cycle}")
                break

            if verbose and cycle % 1000 == 0 and cycle > 0:
                elapsed = time.time() - start_time
                print(f"Cycle {cycle}: Injected={self.stats['total_flits_injected']}, " f"Ejected={self.stats['total_flits_ejected']}, " f"Time={elapsed:.1f}s")

        total_time = time.time() - start_time
        final_results = self.get_network_status()

        if verbose:
            print(f"Ring simulation completed in {total_time:.2f}s")
            self._print_final_statistics(final_results)

        return final_results

    def _are_all_traffics_completed(self) -> bool:
        """
        判断所有 traffic 是否完成：当所有 active_traffics 中的 TrafficState
        的 is_completed() 返回 True 时，即可结束仿真。
        """
        # 仅依据 TrafficScheduler 的状态判断，而非网络内部缓冲
        for traffic_state in self.traffic_scheduler.active_traffics.values():
            if not traffic_state.is_completed():
                return False
        return True

    def _print_final_statistics(self, results: Dict):
        """打印最终统计信息"""
        stats = results["statistics"]
        print(f"\n=== Ring Topology Performance Summary ===")
        print(f"  Network Configuration: {self.config.NUM_RING_NODES} nodes")
        print(f"  Total Cycles: {self.current_cycle}")
        print(f"  Injected Flits: {stats['total_flits_injected']}")
        print(f"  Ejected Flits: {stats['total_flits_ejected']}")
        print(f"  Average Latency: {stats.get('average_latency', 0):.2f} cycles")
        print(f"  Throughput: {stats.get('throughput', 0):.3f} flits/cycle")
        print(f"  Congestion Events: {stats['congestion_events']}")
        print(f"  Adaptive Routing Events: {stats['adaptive_routing_events']}")
        print(f"  CW Utilization: {stats.get('cw_utilization', 0):.3f}")
        print(f"  CCW Utilization: {stats.get('ccw_utilization', 0):.3f}")

        # 打印节点状态摘要
        total_inject_occupancy = 0
        total_eject_occupancy = 0
        max_congestion = 0

        for node_status in results["node_status"]:
            inject_occ = sum(node_status["inject_queues"].values())
            eject_occ = sum(node_status["eject_queues"].values())
            total_inject_occupancy += inject_occ
            total_eject_occupancy += eject_occ
            node_congestion = sum(node_status["congestion"].values())
            max_congestion = max(max_congestion, node_congestion)

        print(f"  Final Queue Occupancy - Inject: {total_inject_occupancy}, Eject: {total_eject_occupancy}")
        print(f"  Max Node Congestion: {max_congestion}")
        print(f"===========================================")

    def enable_packet_debug(self, packet_ids, real_time=False, delay=0.5):
        """启用特定packet_id的追踪调试

        Args:
            packet_ids: 要追踪的packet_id列表或单个packet_id
            real_time: 是否启用实时debug显示
            delay: 实时debug的延迟时间(秒)
        """
        if isinstance(packet_ids, (list, tuple)):
            self.debug_packet_ids.update(packet_ids)
        else:
            self.debug_packet_ids.add(packet_ids)

        self.debug_enabled = True
        self.real_time_debug = real_time
        self.debug_delay = delay

        mode_str = "实时显示模式" if real_time else "静默追踪模式"
        print(f"已启用packet追踪调试 ({mode_str})，追踪packet_id: {list(self.debug_packet_ids)}")
        if real_time:
            print(f"实时debug延迟: {delay}秒")
            print("=" * 60)

    def _trace_packet_locations(self):
        """追踪调试包在当前cycle的位置"""
        if not self.debug_enabled or not self.debug_packet_ids:
            return

        import time

        for packet_id in self.debug_packet_ids:
            locations = self._find_packet_locations(packet_id)

            if locations:
                # 简化debug显示 - 只打印当前cycle状态，不记录历史
                if self.real_time_debug:
                    print(f"Cycle {self.current_cycle:4d}")
                    for location in locations:
                        ch = location.get("channel", "unknown")
                        flit = location.get("flit")
                        if flit:
                            print(f"[{ch}] {flit}")

        # 如果启用实时debug，添加延迟
        if self.real_time_debug and locations:
            time.sleep(self.debug_delay)

    def _find_packet_locations(self, packet_id):
        """
        基于各通道网络的 send_flits 缓存查找指定 packet_id 的所有 flit。
        返回列表，每项格式为 {"channel": 通道名, "flit": flit 对象}
        """
        locations = []
        for ch, net in self.networks.items():
            send_buf = getattr(net, "send_flits", None)
            if isinstance(send_buf, dict):
                flist = send_buf.get(packet_id, [])
            elif isinstance(send_buf, list):
                flist = [fl for fl in send_buf if getattr(fl, "packet_id", None) == packet_id]
            else:
                continue

            for fl in flist:
                locations.append({"channel": ch, "flit": fl})
        return locations

    def print_packet_trace(self, packet_id=None):
        """简化的debug状态打印"""
        if not self.debug_enabled:
            print("Debug追踪未启用")
            return

        print(f"当前追踪的包: {list(self.debug_packet_ids)}")
        print("注意: 现在使用实时debug模式，不再记录历史日志")

    def get_packet_summary(self, packet_id):
        """简化的包状态摘要"""
        if not self.debug_enabled:
            return {"error": "Debug未启用"}

        if packet_id not in self.debug_packet_ids:
            return {"error": f"packet_id {packet_id} 未被追踪"}

        # 只返回当前状态
        locations = self._find_packet_locations(packet_id)
        return {"packet_id": packet_id, "current_cycle": self.current_cycle, "current_locations": [loc["location"] for loc in locations], "status": "正在追踪中"}

    def _update_inject_arbitration_state(self, node, ip_type, direction):
        """更新注入仲裁状态"""
        if not hasattr(node, "inject_arbitration_state"):
            return

        # 更新channel优先级 (round-robin)
        if ip_type in node.inject_arbitration_state["channel_priority"]:
            node.inject_arbitration_state["channel_priority"].remove(ip_type)
            node.inject_arbitration_state["channel_priority"].append(ip_type)

        # 记录最后服务的channel
        node.inject_arbitration_state["last_served"][direction] = ip_type

        # 更新传统的round-robin状态以保持兼容性
        if direction in node.rr_state["inject_priority"]:
            node.rr_state["inject_priority"].remove(direction)
            node.rr_state["inject_priority"].append(direction)

    def _update_eject_arbitration_state(self, node, direction, target_ip_type):
        """更新弹出仲裁状态"""
        if not hasattr(node, "eject_arbitration_state"):
            return

        # 更新方向优先级
        if direction in node.eject_arbitration_state["direction_priority"]:
            node.eject_arbitration_state["direction_priority"].remove(direction)
            node.eject_arbitration_state["direction_priority"].append(direction)

        # 记录方向到channel的分配
        node.eject_arbitration_state["channel_assignment"][direction] = target_ip_type
        node.eject_arbitration_state["last_served_direction"] = direction

        # 更新传统的round-robin状态以保持兼容性
        if direction in node.rr_state["eject_priority"]:
            node.rr_state["eject_priority"].remove(direction)
            node.rr_state["eject_priority"].append(direction)

    def _determine_target_ip_channel(self, flit, node):
        """确定flit应该弹出到哪个IP channel"""
        # 基于flit的目标类型确定对应的IP channel
        dest_type = getattr(flit, "destination_type", None)
        if not dest_type:
            # 如果没有指定目标类型，使用第一个可用的IP channel
            return node.connected_ip_type[0] if node.connected_ip_type else None

        # 查找匹配的IP类型
        for ip_type in node.connected_ip_type:
            if ip_type.startswith(dest_type.split("_")[0]):
                return ip_type

        # 如果没有找到匹配的，返回第一个可用的IP channel
        return node.connected_ip_type[0] if node.connected_ip_type else None

    def show_ring_topology_info(self):
        """显示Ring拓扑的详细信息"""
        print(f"=== Ring拓扑信息 ===")
        print(f"节点数: {self.config.NUM_RING_NODES}")
        print(f"每链路slice数: {self.config.SLICE_PER_LINK}")
        print(f"网络频率: {self.config.NETWORK_FREQUENCY} GHz")
        print()

        print("相邻节点间的slice数:")
        for i in range(min(4, self.config.NUM_RING_NODES)):  # 只显示前4个节点的例子
            next_node = (i + 1) % self.config.NUM_RING_NODES
            prev_node = (i - 1) % self.config.NUM_RING_NODES
            print(f"  Node {i} -> Node {next_node}: {self.config.get_ring_link_slices(i, next_node)} slices")
            print(f"  Node {i} -> Node {prev_node}: {self.config.get_ring_link_slices(i, prev_node)} slices")

        print()
        print("IP连接信息:")
        for (ip_type, ip_pos), ip_interface in self.ip_modules.items():
            print(f"  {ip_type} 连接到 Node {ip_pos}")

        print()
        print("队列深度配置:")
        print(f"  注入队列深度: {self.config.IQ_OUT_FIFO_DEPTH}")
        print(f"  弹出队列深度: {self.config.EQ_IN_FIFO_DEPTH}")
        print(f"  IQ通道缓冲深度: {self.config.IQ_CH_FIFO_DEPTH}")
        print(f"  EQ通道缓冲深度: {self.config.EQ_CH_FIFO_DEPTH}")
        print(f"  Ring缓冲深度: {self.config.RING_BUFFER_DEPTH}")
        print("==================")


def debug_example():
    """展示如何使用packet追踪调试功能的示例"""
    print("=== Ring拓扑Packet追踪调试功能使用示例 ===")

    # 创建配置和Ring拓扑
    config = RingConfig()
    config.NUM_RING_NODES = 8
    ring = RingTopology(config, r"../../test_data")

    # 设置traffic
    traffic_config = [["Read_burst4_2262HBM_v2.txt"]]
    ring.setup_traffic(traffic_config)

    # 1. 启用特定packet的追踪
    target_packet_ids = [10]
    ring.enable_packet_debug(target_packet_ids, real_time=0, delay=1.0)

    # 2. 运行几个周期的仿真
    print("运行仿真...")
    for cycle in range(100):
        ring.step_simulation()

        # 可以在特定cycle检查packet状态
        if cycle == 50:
            ring.debug_packet_in_cycle(10, 50)

    # 3. 打印完整的追踪日志
    ring.print_packet_trace(10)

    # 4. 获取生命周期摘要
    summary = ring.get_packet_summary(10)
    if "error" not in summary:
        print(f"Packet {summary['packet_id']} 摘要:")
        print(f"  端到端延迟: {summary.get('end_to_end_latency', 'N/A')} cycles")
        print(f"  访问的位置数: {summary['locations_visited']}")

    print("调试示例完成!")


# 使用示例
if __name__ == "__main__":
    # 创建配置
    config = RingConfig()
    config.NUM_RING_NODES = 8

    # 创建Ring拓扑
    ring = RingTopology(config, r"../../test_data")

    # 设置traffic - 复用现有traffic文件格式
    traffic_config = [["Read_burst4_2262HBM_v2.txt"]]
    ring.setup_traffic(traffic_config)

    # 显示Ring拓扑信息
    # ring.show_ring_topology_info()

    # 启用packet追踪调试（示例：追踪packet_id为0的请求，启用实时显示）
    ring.enable_packet_debug([1], real_time=0, delay=0.3)  # 实时显示，延迟1秒

    # 运行仿真
    results = ring.run_simulation(max_cycles=100000)

    # # 获取生命周期摘要
    # summary = ring.get_packet_summary(0)
    # if "error" not in summary:
    #     print(f"\nPacket {summary['packet_id']} 生命周期摘要:")
    #     print(f"  首次出现: Cycle {summary['first_seen_cycle']}")
    #     print(f"  最后出现: Cycle {summary['last_seen_cycle']}")
    #     print(f"  总周期数: {summary['total_cycles']}")
    #     print(f"  访问位置数: {summary['locations_visited']}")
    #     if summary.get("end_to_end_latency"):
    #         print(f"  端到端延迟: {summary['end_to_end_latency']} cycles")
    #     print(f"  生命周期阶段: {[stage['stage'] for stage in summary['lifecycle_stages']]}")
    # else:
    #     print(f"追踪摘要错误: {summary['error']}")

    print(f"最终统计:")
    print(f"  注入flit数: {results['statistics']['total_flits_injected']}")
    print(f"  弹出flit数: {results['statistics']['total_flits_ejected']}")
    print(f"  平均延迟: {results['statistics'].get('average_latency', 0):.2f}")
    print(f"  吞吐量: {results['statistics'].get('throughput', 0):.3f}")
    print(f"  自适应路由事件: {results['statistics']['adaptive_routing_events']}")
    print(f"  顺时针利用率: {results['statistics'].get('cw_utilization', 0):.3f}")
    print(f"  逆时针利用率: {results['statistics'].get('ccw_utilization', 0):.3f}")

    # 使用result_processor进行结果分析和可视化
    print("\n=== 生成Ring拓扑分析结果 ===")
    try:
        # 初始化result processor
        ring.initialize_result_processor()

        # 生成可视化图表
        total_bw = ring.process_results(plot_rn=False, plot_flow=True, save_path="ring_topology_flow.png")
        print(f"Ring拓扑可视化图已保存: ring_topology_flow.png")

        if total_bw > 0:
            print(f"总带宽: {total_bw:.2f} GB/s")

    except Exception as e:
        print(f"结果分析过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


def _ip_prefix(ip_type: str) -> str:
    """返回不带编号后缀的 ip_type 前缀 (如 'ddr_0' → 'ddr')."""
    return ip_type.split("_")[0]
