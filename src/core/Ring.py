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

# from src.utils.routing_strategies import create_routing_strategy  # 暂时注释掉
from src.utils.component import Flit, Network, Node, RingNetwork
from config.config import CrossRingConfig


class RingConfig(CrossRingConfig):
    """Ring拓扑配置 - 扩展CrossRingConfig"""

    def __init__(self):
        # 使用默认配置文件路径如果没有提供
        super().__init__()

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

        # 确保CH_NAME_LIST存在，Ring拓扑不使用CDMA
        self.CHANNEL_SPEC = {
            "gdma": 2,
            "sdma": 2,
            "ddr": 2,
            "l2m": 2,
        }
        self.CH_NAME_LIST = []
        for key in self.CHANNEL_SPEC:
            for idx in range(self.CHANNEL_SPEC[key]):
                self.CH_NAME_LIST.append(f"{key}_{idx}")

        # 重写IP分布以适配Ring拓扑
        self._setup_ring_ip_distribution()

    def _setup_ring_ip_distribution(self):
        """设置Ring拓扑的IP分布"""
        # 在Ring中，所有节点都可以连接IP
        all_positions = list(range(self.RING_NUM_NODES))

        # 根据IP数量分配到节点
        self.GDMA_SEND_POSITION_LIST = all_positions
        self.SDMA_SEND_POSITION_LIST = all_positions
        self.DDR_SEND_POSITION_LIST = all_positions
        self.L2M_SEND_POSITION_LIST = all_positions
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
            self.NUM_COL = 2  # Ring是1维拓扑
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

        # 提取BaseModel需要的参数
        base_params = {
            "result_save_path": kwargs.get("result_save_path", ""),
            "traffic_config": kwargs.get("traffic_config", []),
            "results_fig_save_path": kwargs.get("results_fig_save_path", ""),
            "plot_flow_fig": kwargs.get("plot_flow_fig", False),
            "flow_fig_show_CDMA": kwargs.get("flow_fig_show_CDMA", False),
            "plot_RN_BW_fig": kwargs.get("plot_RN_BW_fig", False),
            "plot_link_state": kwargs.get("plot_link_state", False),
            "plot_start_time": kwargs.get("plot_start_time", -1),
            "print_trace": kwargs.get("print_trace", False),
            "show_trace_id": kwargs.get("show_trace_id", 0),
            "show_node_id": kwargs.get("show_node_id", 3),
            "verbose": kwargs.get("verbose", 0),
        }

        super().__init__(model_type, config, topo_type, traffic_file_path, **base_params)

        # 初始化Ring特有的路由策略
        self.routing_strategy = self._create_simple_routing_strategy()

        # Ring特有的统计信息
        self.ring_stats = {"cw_usage": 0, "ccw_usage": 0, "diagonal_routes": 0, "adaptive_routes": 0, "routing_decisions": defaultdict(int)}

        logging.info(f"Ring topology initialized: {self.config.RING_NUM_NODES} nodes, " f"routing strategy: {self.config.ROUTING_STRATEGY}")
        self.initial()

    def node_map(self, node, is_source=True):
        return node

    def _create_simple_routing_strategy(self):
        """创建简单的路由策略"""

        class SimpleRingRouter:
            def __init__(self, config):
                self.config = config
                self.num_nodes = config.RING_NUM_NODES

            def get_route_direction(self, source, destination, **context):
                if source == destination:
                    return "LOCAL"

                # 计算顺时针和逆时针距离
                cw_distance = (destination - source) % self.num_nodes
                ccw_distance = (source - destination) % self.num_nodes

                # 选择更短的路径
                if cw_distance <= ccw_distance:
                    return "CW"  # 顺时针
                else:
                    return "CCW"  # 逆时针

        return SimpleRingRouter(self.config)

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
        # Call BaseModel.initial to set up common fields
        super().initial()
        # Ensure Ring config is applied
        self.topo_type_stat = self.config.TOPO_TYPE
        self.config.update_config(self.topo_type_stat)
        # Create Ring-specific adjacency matrix
        self.adjacency_matrix = self.create_adjacency_matrix()
        # Reinitialize networks with Ring topology
        self.req_network = RingNetwork(self.config, self.adjacency_matrix, name="Request Network")
        self.rsp_network = RingNetwork(self.config, self.adjacency_matrix, name="Response Network")
        self.data_network = RingNetwork(self.config, self.adjacency_matrix, name="Data Network")
        # Apply Ring-specific network configurations
        self._configure_ring_networks()
        # Recompute routing paths for Ring topology
        from src.utils.optimal_placement import find_shortest_paths

        self.routes = find_shortest_paths(self.adjacency_matrix)
        # Reinitialize IP modules with new networks and routes
        # from src.utils.component import IPInterface
        from src.utils.component import create_ring_ip_interface

        self.ip_modules = {}
        for ip_pos in range(self.config.RING_NUM_NODES):
            for ip_type in self.config.CH_NAME_LIST:
                ip_interface = create_ring_ip_interface(ip_type, ip_pos, self.config, self.req_network, self.rsp_network, self.data_network, self.node, self.routes)
                # Preserve ring_model reference if supported
                if hasattr(ip_interface, "ring_model"):
                    ip_interface.ring_model = self
                self.ip_modules[(ip_type, ip_pos)] = ip_interface

        # Override IQ directions for pure Ring topology
        self.IQ_directions = ["TL", "TR", "EQ"]
        self.IQ_direction_conditions = {
            "TL": lambda flit: len(flit.path) > 1 and flit.path[1] - flit.path[0] == -1,
            "TR": lambda flit: len(flit.path) > 1 and flit.path[1] - flit.path[0] == 1,
            "EQ": lambda flit: len(flit.path) == 1,
        }

    def _configure_ring_networks(self):
        """配置Ring网络的特有属性"""
        for network in [self.req_network, self.rsp_network, self.data_network]:
            # Ring特有的网络配置 - 只设置存在的属性
            if hasattr(network, "ring_mode"):
                network.ring_mode = True
            if hasattr(network, "routing_strategy"):
                network.routing_strategy = self.routing_strategy

            # 配置ETag支持（复用CrossRing的ETag逻辑）
            if self.config.ETag_BOTHSIDE_UPGRADE:
                if hasattr(network, "ETag_BOTHSIDE_UPGRADE"):
                    network.ETag_BOTHSIDE_UPGRADE = True

    def _inject_queue_arbitration(self, network, ip_positions, network_type):
        """
        Parameters
        ----------
        network : Network
            要操作的网络实例 (req / rsp / data)
        ip_positions : Iterable[int]
            需要遍历的 IP 物理位置集合
        network_type : str
            "req" | "rsp" | "data"
        """
        for ip_pos in ip_positions:
            for direction in self.IQ_directions:
                rr_queue = network.round_robin["IQ"][direction][ip_pos]
                queue_pre = network.inject_queues_pre[direction]
                if queue_pre[ip_pos]:
                    continue  # pre 槽占用
                queue = network.inject_queues[direction]
                if len(queue[ip_pos]) >= self.config.IQ_OUT_FIFO_DEPTH:
                    continue  # FIFO 满

                for ip_type in list(rr_queue):
                    # —— 网络‑特定 ip_type 过滤 ——
                    if network_type == "req" and not (ip_type.startswith("sdma") or ip_type.startswith("gdma") or ip_type.startswith("cdma")):
                        continue
                    if network_type == "rsp" and not (ip_type.startswith("ddr") or ip_type.startswith("l2m")):
                        continue
                    # data 网络不筛选 ip_type

                    if not network.IQ_channel_buffer[ip_type][ip_pos]:
                        continue  # channel‑buffer 空

                    flit = network.IQ_channel_buffer[ip_type][ip_pos][0]
                    if not self.IQ_direction_conditions[direction](flit):
                        continue  # 方向不匹配

                    # —— 网络‑特定前置检查 / 统计 ——
                    if network_type == "req":
                        max_gap = self.config.GDMA_RW_GAP if ip_type.startswith("gdma") else self.config.SDMA_RW_GAP
                        counts = self.dma_rw_counts[ip_type][ip_pos]
                        rd, wr = counts["read"], counts["write"]
                        if flit.req_type == "read" and abs(rd + 1 - wr) >= max_gap:
                            continue
                        if flit.req_type == "write" and abs(wr + 1 - rd) >= max_gap:
                            continue
                        # 使用现有函数做资源检查 + 注入
                        if not self._try_inject_to_direction(flit, ip_type, ip_pos, direction, counts):
                            continue
                        # _try_inject_to_direction 已经做了 popleft & pre‑缓冲写入，故直接 break
                        rr_queue.remove(ip_type)
                        rr_queue.append(ip_type)
                        break

                    else:
                        # —— rsp / data 网络：直接移动到 pre‑缓冲 ——
                        network.IQ_channel_buffer[ip_type][ip_pos].popleft()
                        queue_pre[ip_pos] = flit

                        if network_type == "rsp":
                            flit.rsp_entry_network_cycle = self.cycle
                        elif network_type == "data":
                            req = self.req_network.send_flits[flit.packet_id][0]
                            flit.sync_latency_record(req)
                            flit.data_entry_network_cycle = self.cycle
                            self.send_flits_num += 1
                            self.trans_flits_num += 1
                            if hasattr(flit, "traffic_id"):
                                self.traffic_scheduler.update_traffic_stats(flit.traffic_id, "sent_flit")

                        rr_queue.remove(ip_type)
                        rr_queue.append(ip_type)
                        break

    def _tag_move(self, network):
        """
        Override tag movement for Ring topology: shift tags clockwise along the ring.
        """
        num_nodes = self.config.RING_NUM_NODES
        slice_count = self.config.SLICE_PER_LINK
        # Build list of CW links: each node to its next neighbor
        links = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
        # Flatten existing tags
        saved_tags = []
        for link in links:
            saved_tags.extend(network.links_tag[link])
        # Rotate tags by one link (slice_count elements)
        rotated = saved_tags[-slice_count:] + saved_tags[:-slice_count]
        # Write back rotated tags
        idx = 0
        for link in links:
            for j in range(slice_count):
                network.links_tag[link][j] = rotated[idx]
                idx += 1

    def _calculate_ring_routes(self):
        """计算Ring拓扑的路由表 - 使用最短路径算法"""
        from src.utils.optimal_placement import find_shortest_paths

        return find_shortest_paths(self.adjacency_matrix)

    def _get_ring_path(self, source: int, destination: int) -> List[int]:
        """获取Ring拓扑中两点间的路径 - 简化版本"""
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
        from src.utils.component import IPInterface

        for ip_pos in self.flit_positions:
            for ip_type in self.config.CH_NAME_LIST:
                # 传入Ring特有的属性给IPInterface
                ip_interface = IPInterface(ip_type, ip_pos, self.config, self.req_network, self.rsp_network, self.data_network, self.node, self.routes)
                # 如果支持，设置Ring特有属性
                if hasattr(ip_interface, "ring_model"):
                    ip_interface.ring_model = self
                self.ip_modules[(ip_type, ip_pos)] = ip_interface

    def get_ring_directions(self, current_pos: int, target_pos: int, **context) -> str:
        """
        获取Ring拓扑中的路由方向 - Ring特有方法
        """
        if current_pos == target_pos:
            return "LOCAL"

        # 使用路由策略决定方向
        direction = self.routing_strategy.get_route_direction(current_pos, target_pos, **context)

        # 更新统计信息
        self.ring_stats["routing_decisions"][direction] += 1

        if direction == "CW":
            self.ring_stats["cw_usage"] += 1
            return "CW"  # 顺时针
        elif direction == "CCW":
            self.ring_stats["ccw_usage"] += 1
            return "CCW"  # 逆时针
        else:
            return "LOCAL"

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
        # BaseModel没有step_simulation方法，直接实现Ring特有的处理
        self._update_ring_statistics()

    def _update_ring_statistics(self):
        """更新Ring特有的统计信息"""
        total_decisions = sum(self.ring_stats["routing_decisions"].values())
        if total_decisions > 0:
            self.ring_stats["cw_utilization"] = self.ring_stats["cw_usage"] / total_decisions
            self.ring_stats["ccw_utilization"] = self.ring_stats["ccw_usage"] / total_decisions

    def get_ring_performance_stats(self) -> Dict:
        """获取Ring特有的性能统计"""
        # BaseModel没有get_performance_stats方法，直接返回Ring特有的统计
        ring_specific_stats = {
            "routing_strategy": self.config.ROUTING_STRATEGY,
            "cw_utilization": self.ring_stats.get("cw_utilization", 0),
            "ccw_utilization": self.ring_stats.get("ccw_utilization", 0),
            "diagonal_routes": self.ring_stats["diagonal_routes"],
            "adaptive_routes": self.ring_stats["adaptive_routes"],
            "routing_decisions": dict(self.ring_stats["routing_decisions"]),
        }

        return ring_specific_stats

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

    def Ring_Bridge_arbitration(self, network):
        """No-op for Ring topology: no cross-ring bridges."""
        return

    def RB_inject_vertical(self, network):
        """No-op for Ring topology: no vertical ring injection."""
        return []

    def _process_ring_bridge_inject(self, network, *args, **kwargs):
        """No-op for Ring topology: no ring_bridge injection."""
        return None

    def Eject_Queue_arbitration(self, network, flit_type):
        """Override eject logic: Ring topology uses only EQ and IQ injections."""
        # Only handle IQ->EQ eject; skip TU and TD
        # Use BaseModel EQ eject: inject directly from inject_queues
        eject_count = 0
        for ip_pos in range(self.config.RING_NUM_NODES):
            # 构造eject_flits
            eject_flits = [network.eject_queues[fifo_pos][ip_pos][0] if network.eject_queues[fifo_pos][ip_pos] else None for fifo_pos in ["TL", "TR"]] + [
                network.inject_queues[fifo_pos][ip_pos][0] if network.inject_queues[fifo_pos][ip_pos] else None for fifo_pos in ["EQ"]
            ]
            if not any(eject_flits):
                continue
            self._move_to_eject_queues_pre(network, eject_flits, ip_pos)
            eject_count += 1

    def _move_to_eject_queues_pre(self, network: Network, eject_flits, ip_pos):
        for ip_type in network.EQ_channel_buffer.keys():
            # Ring topology: use ip_pos directly for round_robin indexing
            rr_queue = network.round_robin["EQ"][ip_type][ip_pos]
            for i in rr_queue:
                if i >= len(eject_flits) or eject_flits[i] is None:
                    continue
                if eject_flits[i].destination_type == ip_type and len(network.EQ_channel_buffer[ip_type][ip_pos]) < network.config.EQ_CH_FIFO_DEPTH:
                    network.EQ_channel_buffer_pre[ip_type][ip_pos] = eject_flits[i]
                    eject_flits[i].is_arrive = True
                    eject_flits[i].arrival_eject_cycle = self.cycle
                    eject_flits[i] = None
                    if i == 0:
                        flit = network.eject_queues["TL"][ip_pos].popleft()
                        if flit.used_entry_level == "T0":
                            network.EQ_UE_Counters["TL"][ip_pos]["T0"] -= 1
                        elif flit.used_entry_level == "T1":
                            network.EQ_UE_Counters["TL"][ip_pos]["T1"] -= 1
                        elif flit.used_entry_level == "T2":
                            network.EQ_UE_Counters["TL"][ip_pos]["T2"] -= 1
                    elif i == 1:
                        flit = network.eject_queues["TR"][ip_pos].popleft()
                        if flit.used_entry_level == "T1":
                            network.EQ_UE_Counters["TR"][ip_pos]["T1"] -= 1
                        elif flit.used_entry_level == "T2":
                            network.EQ_UE_Counters["TR"][ip_pos]["T2"] -= 1
                    elif i == 2:
                        flit = network.inject_queues["EQ"][ip_pos].popleft()

                    if flit.ETag_priority == "T1":
                        self.EQ_ETag_T1_num_stat += 1
                    elif flit.ETag_priority == "T0":
                        self.EQ_ETag_T0_num_stat += 1
                    flit.ETag_priority = "T2"

                    rr_queue.remove(i)
                    rr_queue.append(i)
                    return eject_flits
        return eject_flits

    def _move_pre_to_queues(self, network, ip_pos):
        """重写Ring拓扑的pre-to-queues移动逻辑"""
        # Ring拓扑简化版本，只处理基本的IQ和EQ方向
        for direction in self.IQ_directions:
            queue_pre = network.inject_queues_pre[direction]
            queue = network.inject_queues[direction]
            if queue_pre[ip_pos] and len(queue[ip_pos]) < self.config.IQ_OUT_FIFO_DEPTH:
                flit = queue_pre[ip_pos]
                flit.departure_inject_cycle = self.cycle
                flit.flit_position = f"IQ_{direction}"
                queue[ip_pos].append(flit)
                queue_pre[ip_pos] = None

        # EQ_IN_PRE → EQ_IN
        for fifo_pos in ("TR", "TL"):
            queue_pre = network.eject_queues_in_pre[fifo_pos]
            queue = network.eject_queues[fifo_pos]
            if queue_pre[ip_pos] and len(queue[ip_pos]) < self.config.EQ_IN_FIFO_DEPTH:
                flit = queue_pre[ip_pos]
                flit.is_arrive = fifo_pos == "EQ"
                flit.flit_position = f"EQ_{fifo_pos}"
                queue[ip_pos].append(flit)
                queue_pre[ip_pos] = None

        # EQ_channel_buffer_pre → EQ_channel_buffer
        for ip_type in network.EQ_channel_buffer_pre.keys():
            queue_pre = network.EQ_channel_buffer_pre[ip_type]
            queue = network.EQ_channel_buffer[ip_type]
            if queue_pre[ip_pos] and len(queue[ip_pos]) < self.config.EQ_CH_FIFO_DEPTH:
                flit = queue_pre[ip_pos]
                flit.flit_position = "EQ_CH"
                queue[ip_pos].append(flit)
                queue_pre[ip_pos] = None

    def _flit_move(self, network, flits, flit_type):
        """重写Ring拓扑的flit移动逻辑，去除不需要的CrossRing特性"""
        # 在Ring拓扑中，只需要处理基本的链路移动和注入队列

        # 1. 处理链路上flit移动 - 先plan所有移动，再execute
        link_flits = []
        for flit in flits:
            if flit.flit_position == "Link":
                link_flits.append(flit)

        # 分阶段处理避免冲突：先计划移动，再执行移动
        for flit in link_flits:
            network.plan_move(flit, self.cycle)

        # 执行移动，如果成功则从flits列表中移除
        flits_to_remove = []
        for flit in link_flits:
            if network.execute_moves(flit, self.cycle):
                flits_to_remove.append(flit)

        for flit in flits_to_remove:
            flits.remove(flit)

        # 2. 处理eject仲裁（简化版本）
        self.Eject_Queue_arbitration(network, flit_type)

        # 3. 处理注入队列
        for direction, inject_queues in network.inject_queues.items():
            if direction not in self.IQ_directions:
                continue

            num, IQ_inject_flits = self.process_inject_queues(network, inject_queues, direction)
            if num == 0 and not IQ_inject_flits:
                continue

            if flit_type == "req":
                self.req_num += num
            elif flit_type == "rsp":
                self.rsp_num += num
            elif flit_type == "data":
                self.flit_num += num

            for flit in IQ_inject_flits:
                if flit not in flits:
                    flits.append(flit)

        network.update_excess_ITag()
        # network.update_cross_point()

        return flits

    def move_pre_to_queues_all(self):
        """重写Ring拓扑的pre-to-queues移动，兼容IP接口和网络级别"""
        # 先处理IPInterface的pre->FIFO
        for ip_pos in range(self.config.RING_NUM_NODES):
            for ip_type in self.config.CH_NAME_LIST:
                self.ip_modules[(ip_type, ip_pos)].move_pre_to_fifo()
        # 再处理网络级别的pre->FIFO
        for ip_pos in range(self.config.RING_NUM_NODES):
            self._move_pre_to_queues(self.req_network, ip_pos)
            self._move_pre_to_queues(self.rsp_network, ip_pos)
            self._move_pre_to_queues(self.data_network, ip_pos)

    def process_comprehensive_results(self):
        """Ring拓扑的综合结果处理 - 重写以使用Ring特有的画图功能"""
        if not self.result_save_path:
            return

        # 调用基础的结果处理
        super().process_comprehensive_results()

        # Ring特有的画图功能
        if self.plot_flow_fig:
            try:
                # 使用Ring特有的画图函数
                self.result_processor.draw_flow_graph_ring_rectangular(
                    self.data_network,  # 使用data network的链路统计
                    save_path=self.results_fig_save_path or self.result_save_path
                )
                print("Ring flow graph generated successfully")
            except Exception as e:
                print(f"Error generating Ring flow graph: {e}")

        # Ring特有的统计输出
        self.print_ring_summary()


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
    config = RingConfig()  # 使用默认配置，避免文件路径问题
    config.RING_NUM_NODES = num_nodes
    config.ROUTING_STRATEGY = routing_strategy

    topo_type = f"Ring_{num_nodes}"

    # 设置默认参数
    default_params = {
        "traffic_file_path": "../../test_data",
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
    ring_model = create_ring_model(
        num_nodes=8,
        routing_strategy="load_balanced",
        traffic_file_path="../../test_data",
        print_trace=0,
        show_trace_id=0,
    )

    # 运行仿真
    print("Starting Ring simulation...")
    results = ring_model.run()

    # 打印结果
    ring_model.print_ring_summary()

    # 获取详细统计
    stats = ring_model.get_ring_performance_stats()
    print(f"\nDetailed stats: {stats}")
