#!/usr/bin/env python3
"""
事件驱动理论带宽计算工具 - 基于CrossRing网络拓扑的精确事务模拟
支持三通道独立性、Token Bucket带宽限制和事件驱动调度
相比顺序处理版本，提供更真实的并发仿真能力
"""

import math
import csv
import yaml
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
import sys
import heapq
from collections import deque

if sys.platform == "darwin":  # macOS
    try:
        matplotlib.use("macosx")
    except ImportError:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import argparse
import os

# 添加配置路径
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from src.utils.optimal_placement import find_shortest_paths, create_adjacency_matrix_adv
except ImportError:
    print("警告: 无法导入optimal_placement模块，将使用简化的路径计算")
    find_shortest_paths = None
    create_adjacency_matrix_adv = None


@dataclass
class TrafficRequest:
    """流量请求数据结构"""

    start_time: int  # ns
    source_node: int
    source_ip_type: str  # 如 "gdma_0"，其中数字0是相对于节点的IP编号
    dest_node: int
    dest_ip_type: str  # 如 "ddr_0"，其中数字0是相对于节点的IP编号
    req_type: str  # 'R' or 'W'
    burst_length: int

    packet_id: int = 0

    def __post_init__(self):
        self.total_bytes = self.burst_length * 128  # 128 bytes per flit


@dataclass
class Event:
    """事件数据结构"""

    time: float  # 事件时间
    priority: int  # 优先级（同时间内的处理顺序）
    event_type: str  # 事件类型
    data: Dict[str, Any]  # 事件数据

    def __lt__(self, other):
        """用于heapq比较"""
        if self.time != other.time:
            return self.time < other.time
        return self.priority < other.priority


@dataclass
class TransactionState:
    """事务状态跟踪"""

    request: TrafficRequest
    start_time: float
    current_stage: str  # 当前阶段
    tracker_acquired: bool = False
    cmd_send_time: float = 0
    cmd_recv_time: float = 0
    data_send_times: List[float] = None
    data_recv_times: List[float] = None
    ack_send_time: float = 0  # 写事务用
    ack_recv_time: float = 0  # 写事务用
    completion_time: float = 0
    path_hops: int = 0
    path_delay: int = 0

    def __post_init__(self):
        if self.data_send_times is None:
            self.data_send_times = []
        if self.data_recv_times is None:
            self.data_recv_times = []


@dataclass
class WorkingInterval:
    """工作区间数据结构"""

    start_time: int
    end_time: int
    duration: int
    flit_count: int
    total_bytes: int
    request_count: int

    @property
    def bandwidth_bytes_per_ns(self) -> float:
        """计算该区间的带宽 (bytes/ns)"""
        return self.total_bytes / self.duration if self.duration > 0 else 0.0


@dataclass
class TransactionResult:
    """事务处理结果"""

    packet_id: int
    start_time: int
    completion_time: int
    source_node: int
    dest_node: int
    source_ip_type: str  # 如 "gdma_0"
    dest_ip_type: str  # 如 "ddr_0"
    req_type: str
    burst_length: int
    total_bytes: int

    # 时延统计
    path_hops: int
    path_delay: int  # ns
    cmd_latency: int  # ns
    data_latency: int  # ns
    transaction_latency: int  # ns


class Tracker:
    """Outstanding credit计数器"""

    def __init__(self, limit):
        self.limit = limit
        self.count = 0

    def can_accept(self):
        return self.count < self.limit

    def inc(self):
        if not self.can_accept():
            raise RuntimeError("Exceeded limit")
        self.count += 1

    def dec(self):
        if self.count == 0:
            raise RuntimeError("Underflow")
        self.count -= 1


class TokenBucket:
    """Token Bucket带宽限制器"""

    def __init__(self, rate_gb_per_s: float):
        self.rate = rate_gb_per_s  # GB/s
        self.tokens = rate_gb_per_s  # 初始token
        self.last_update = 0.0

    def consume(self, size_bytes: int, current_time: float) -> float:
        """
        消耗tokens并返回实际可用时间

        Args:
            size_bytes: 数据大小（字节）
            current_time: 当前时间（ns）

        Returns:
            实际可用时间（ns）
        """
        # 更新tokens（基于时间流逝）
        elapsed_s = (current_time - self.last_update) / 1e9  # 转换为秒
        self.tokens = min(self.tokens + elapsed_s * self.rate, self.rate)
        self.last_update = current_time

        # 计算需要的tokens（GB）
        required_tokens = size_bytes / 1e9

        # 如果tokens不足，计算等待时间
        if self.tokens < required_tokens:
            wait_time_s = (required_tokens - self.tokens) / self.rate
            wait_time_ns = wait_time_s * 1e9
            actual_time = math.ceil(current_time + wait_time_ns)
            self.tokens = 0  # 消耗完所有tokens
            self.last_update = actual_time
            return actual_time
        else:
            self.tokens -= required_tokens
            return current_time


@dataclass
class Config:
    """简化的配置类"""

    TOPO_TYPE: str = "5x4"
    NUM_NODE: int = 40
    NUM_COL: int = 4
    NUM_ROW: int = 10
    NUM_IP: int = 32
    FLIT_SIZE: int = 128
    SLICE_PER_LINK: int = 8
    BURST: int = 4
    NETWORK_FREQUENCY: float = 2.0  # GHz

    # 延迟配置 (ns)
    DDR_R_LATENCY: int = 40  # 40 * 2
    DDR_W_LATENCY: int = 0
    L2M_R_LATENCY: int = 12  # 12 * 2
    L2M_W_LATENCY: int = 16  # 16 * 2
    SN_TRACKER_RELEASE_LATENCY: int = 80  # 40 * 2

    # Tracker配置 (Outstanding限制)
    RN_R_TRACKER_OSTD: int = 64
    RN_W_TRACKER_OSTD: int = 64
    SN_DDR_R_TRACKER_OSTD: int = 64
    SN_DDR_W_TRACKER_OSTD: int = 64
    SN_L2M_R_TRACKER_OSTD: int = 64
    SN_L2M_W_TRACKER_OSTD: int = 64

    # 工作区间配置
    MIN_GAP_THRESHOLD: int = 50  # ns，小于此值的间隔被视为同一工作区间

    # 带宽限制 (GB/s)
    GDMA_BW_LIMIT: float = 128.0
    SDMA_BW_LIMIT: float = 128.0
    CDMA_BW_LIMIT: float = 128.0
    DDR_BW_LIMIT: float = 128.0
    L2M_BW_LIMIT: float = 128.0

    # 节点位置列表（简化版）
    GDMA_SEND_POSITION_LIST: List[int] = None
    SDMA_SEND_POSITION_LIST: List[int] = None
    DDR_SEND_POSITION_LIST: List[int] = None
    L2M_SEND_POSITION_LIST: List[int] = None

    def __post_init__(self):
        if self.GDMA_SEND_POSITION_LIST is None:
            # 为5x4拓扑生成默认位置
            self.GDMA_SEND_POSITION_LIST = [i * 2 for i in range(self.NUM_IP // 4)]
            self.SDMA_SEND_POSITION_LIST = [i * 2 + 1 for i in range(self.NUM_IP // 4)]
            self.DDR_SEND_POSITION_LIST = [self.NUM_COL + i * 2 for i in range(self.NUM_IP // 4)]
            self.L2M_SEND_POSITION_LIST = [self.NUM_COL + i * 2 + 1 for i in range(self.NUM_IP // 4)]


class IPState:
    """单个IP实例的状态管理"""

    def __init__(self, ip_id: str, ip_type: str, config: Config):
        self.ip_id = ip_id
        self.ip_type = ip_type

        # 三通道独立管理（下次可用时间）
        self.next_send_time = {"REQ": 0.0, "RSP": 0.0, "DATA": 0.0}
        self.next_recv_time = {"REQ": 0.0, "RSP": 0.0, "DATA": 0.0}

        # Token Bucket带宽限制（每个IP实例独立）
        self.token_bucket = self._create_token_bucket(ip_type, config)

        # Tracker Outstanding限制（每个IP实例独立）
        self.tracker_rd, self.tracker_wr = self._create_trackers(ip_type, config)

    def _create_token_bucket(self, ip_type: str, config: Config) -> TokenBucket:
        """根据IP类型创建对应的Token Bucket"""
        if ip_type.startswith("gdma"):
            return TokenBucket(config.GDMA_BW_LIMIT)
        elif ip_type.startswith("sdma"):
            return TokenBucket(config.SDMA_BW_LIMIT)
        elif ip_type.startswith("cdma"):
            return TokenBucket(config.CDMA_BW_LIMIT)
        elif ip_type.startswith("ddr"):
            return TokenBucket(config.DDR_BW_LIMIT)
        elif ip_type.startswith("l2m"):
            return TokenBucket(config.L2M_BW_LIMIT)
        else:
            # 默认无限制
            return TokenBucket(float("inf"))

    def _create_trackers(self, ip_type: str, config: Config) -> tuple[Tracker, Tracker]:
        """根据IP类型创建对应的Tracker"""
        if ip_type.startswith("gdma") or ip_type.startswith("sdma") or ip_type.startswith("cdma"):
            # RN端IP：GDMA/SDMA/CDMA
            return Tracker(config.RN_R_TRACKER_OSTD), Tracker(config.RN_W_TRACKER_OSTD)
        elif ip_type.startswith("ddr"):
            # SN端DDR
            return Tracker(config.SN_DDR_R_TRACKER_OSTD), Tracker(config.SN_DDR_W_TRACKER_OSTD)
        elif ip_type.startswith("l2m"):
            # SN端L2M
            return Tracker(config.SN_L2M_R_TRACKER_OSTD), Tracker(config.SN_L2M_W_TRACKER_OSTD)
        else:
            # 默认配置
            return Tracker(64), Tracker(64)


class NodeState:
    """节点状态管理 - 可能包含多个IP"""

    def __init__(self, node_id: int):
        self.node_id = node_id
        self.ip_instances = {}  # ip_type -> IPState

    def add_ip_instance(self, ip_type: str, config: Config):
        """添加IP实例到节点"""
        if ip_type not in self.ip_instances:
            self.ip_instances[ip_type] = IPState(f"{ip_type}@{self.node_id}", ip_type, config)

    def get_ip_instance(self, ip_type: str) -> Optional[IPState]:
        """获取指定类型的IP实例"""
        return self.ip_instances.get(ip_type)


class EventDrivenCalculator:
    """事件驱动理论带宽计算器"""

    # 网络延迟参数 (ns) - 可直接在此处调整
    INJECT_LATENCY = 2  # 注入延迟
    EJECT_LATENCY = 2  # 弹出延迟
    DIMENSION_CHANGE_LATENCY = 2  # 维度转换延迟（XY路由最多1次）

    # 事件优先级定义
    PRIORITY_RETRY_CHECK = 0  # 重试检查最高优先级
    PRIORITY_TRACKER_RELEASE = 1  # tracker释放
    PRIORITY_DATA_RECV = 2  # 数据接收
    PRIORITY_DATA_SEND = 3  # 数据发送
    PRIORITY_ACK_RECV = 4  # ACK接收
    PRIORITY_ACK_SEND = 5  # ACK发送
    PRIORITY_CMD_RECV = 6  # 命令接收
    PRIORITY_CMD_SEND = 7  # 命令发送
    PRIORITY_REQUEST_START = 8  # 请求开始最低优先级

    def __init__(self, config_file: str, traffic_file: str, debug=False, progress_interval=1000):
        self.config = self._load_config(config_file)
        self.traffic_requests = self._load_traffic(traffic_file)
        self.debug = debug
        self.progress_interval = progress_interval  # 进度报告间隔

        # 构建网络拓扑
        self.adjacency_matrix = self._build_topology()
        if find_shortest_paths is not None:
            self.shortest_paths = find_shortest_paths(self.adjacency_matrix)
        else:
            self.shortest_paths = self._compute_simple_shortest_paths()

        # 初始化节点状态
        self.nodes = self._initialize_nodes()

        # 事件驱动系统
        self.event_queue = []  # heapq最小堆
        self.current_time = 0.0

        # 事务状态管理
        self.active_transactions: Dict[int, TransactionState] = {}
        self.completed_transactions: List[TransactionResult] = []

        # 重试队列管理
        self.retry_queue = deque()  # 等待重试的请求

        # 进度跟踪
        self.total_requests = len(self.traffic_requests)
        self.last_progress_time = 0

        # 事件处理器映射
        self.event_handlers = {
            "REQUEST_START": self._handle_request_start,
            "CMD_SEND": self._handle_cmd_send,
            "CMD_RECV": self._handle_cmd_recv,
            "ACK_SEND": self._handle_ack_send,
            "ACK_RECV": self._handle_ack_recv,
            "DATA_SEND": self._handle_data_send,
            "DATA_RECV": self._handle_data_recv,
            "TRACKER_RELEASE": self._handle_tracker_release,
            "RETRY_CHECK": self._handle_retry_check,
        }

        if self.debug:
            print(f"初始化完成，共{len(self.traffic_requests)}个请求待处理")

    def _load_config(self, config_file: str) -> Config:
        """加载配置文件"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        # 从YAML文件创建配置
        with open(config_file, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)

        # 创建Config实例
        config = Config()

        # 更新配置参数
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # 触发__post_init__来设置默认位置列表
        config.__post_init__()

        return config

    def _load_traffic(self, traffic_file: str) -> List[TrafficRequest]:
        """加载流量文件"""
        if not os.path.exists(traffic_file):
            raise FileNotFoundError(f"流量文件不存在: {traffic_file}")

        requests = []
        packet_id = 1

        with open(traffic_file, "r") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                try:
                    parts = line.split(",")
                    if len(parts) >= 7:
                        start_time = int(parts[0])
                        source_node = int(parts[1])
                        source_ip_type = parts[2]  # 如 "gdma_0"
                        dest_node = int(parts[3])
                        dest_ip_type = parts[4]  # 如 "ddr_0"
                        req_type = parts[5]
                        burst_length = int(parts[6])

                        request = TrafficRequest(
                            start_time=start_time,
                            source_node=source_node,
                            source_ip_type=source_ip_type,
                            dest_node=dest_node,
                            dest_ip_type=dest_ip_type,
                            req_type=req_type,
                            burst_length=burst_length,
                            packet_id=packet_id,
                        )
                        requests.append(request)
                        packet_id += 1
                except (ValueError, IndexError) as e:
                    print(f"警告: 流量文件第{line_no}行格式错误: {line}")
                    continue

        print(f"成功加载 {len(requests)} 个流量请求")
        return requests

    def _build_topology(self) -> np.ndarray:
        """构建网络拓扑邻接矩阵"""
        if create_adjacency_matrix_adv is not None:
            try:
                return create_adjacency_matrix_adv(self.config.TOPO_TYPE, self.config.NUM_NODE, self.config.NUM_COL)
            except Exception as e:
                print(f"警告: 使用optimal_placement模块失败: {e}")
                print("回退到简化拓扑创建")
                return self._create_simple_grid_topology()
        else:
            # 简化的邻接矩阵创建（Grid拓扑）
            return self._create_simple_grid_topology()

    def _create_simple_grid_topology(self) -> np.ndarray:
        """创建简化的Grid拓扑邻接矩阵"""
        num_nodes = self.config.NUM_NODE
        num_cols = self.config.NUM_COL
        num_rows = self.config.NUM_ROW

        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        for node in range(num_nodes):
            row = node // num_cols
            col = node % num_cols

            # 水平连接
            if col < num_cols - 1:
                adjacency_matrix[node][node + 1] = 1
                adjacency_matrix[node + 1][node] = 1

            # 垂直连接
            if row < num_rows - 1:
                adjacency_matrix[node][node + num_cols] = 1
                adjacency_matrix[node + num_cols][node] = 1

        return adjacency_matrix

    def _compute_simple_shortest_paths(self) -> Dict[int, Dict[int, List[int]]]:
        """使用BFS计算简化的最短路径"""
        from collections import deque

        num_nodes = self.config.NUM_NODE
        paths = {}

        for src in range(num_nodes):
            paths[src] = {}
            # BFS
            queue = deque([(src, [src])])
            visited = {src}

            while queue:
                current, path = queue.popleft()

                if current not in paths[src]:
                    paths[src][current] = path

                # 探索邻居
                for next_node in range(num_nodes):
                    if self.adjacency_matrix[current][next_node] == 1 and next_node not in visited:
                        visited.add(next_node)
                        queue.append((next_node, path + [next_node]))

        return paths

    def _initialize_nodes(self) -> Dict[int, NodeState]:
        """初始化所有节点状态"""
        nodes = {}

        # 为每个节点创建NodeState
        for node_id in range(self.config.NUM_NODE):
            nodes[node_id] = NodeState(node_id)

        # 根据配置为每个节点添加相应的IP实例
        self._add_ip_instances_to_nodes(nodes)

        return nodes

    def _add_ip_instances_to_nodes(self, nodes: Dict[int, NodeState]):
        """为节点添加IP实例"""
        # 每个节点可能有多种类型的IP，每种类型可能有多个实例
        # 这里假设每个节点的IP类型分布是固定的（基于配置）

        for node_id in range(self.config.NUM_NODE):
            node = nodes[node_id]

            # 为每个节点添加可能的IP类型（每种类型最多4个实例，编号0-3）
            ip_types = ["gdma", "sdma", "ddr", "l2m"]
            for ip_type in ip_types:
                for ip_idx in range(4):  # 假设每种类型最多4个实例
                    ip_type_with_idx = f"{ip_type}_{ip_idx}"
                    node.add_ip_instance(ip_type_with_idx, self.config)

    def calculate_path_delay(self, src: int, dest: int) -> Tuple[int, int]:
        """
        计算路径延迟，包括inject、eject和维度转换延迟

        Returns:
            (跳数, 总延迟ns)
        """
        if src == dest:
            # 即使是同一节点，也需要inject和eject延迟
            same_node_delay = self.INJECT_LATENCY + self.EJECT_LATENCY
            return 0, same_node_delay

        if src in self.shortest_paths and dest in self.shortest_paths[src]:
            path = self.shortest_paths[src][dest]
            hops = len(path) - 1
        else:
            raise ValueError(f"找不到从节点{src}到节点{dest}的路径")

        # 计算基础路径延迟：跳数 × slice数量 / 网络频率，向上取整
        path_delay_cycles = hops * self.config.SLICE_PER_LINK
        base_path_delay_ns = math.ceil(path_delay_cycles / self.config.NETWORK_FREQUENCY)

        # 计算维度转换次数（XY路由）
        src_row, src_col = divmod(src, self.config.NUM_COL)
        dest_row, dest_col = divmod(dest, self.config.NUM_COL)

        # 如果需要同时水平和垂直移动，则有1次维度转换
        dimension_changes = 1 if (src_row != dest_row and src_col != dest_col) else 0

        # 总延迟 = 基础路径延迟 + inject + eject + 维度转换延迟
        total_delay_ns = base_path_delay_ns + self.INJECT_LATENCY + self.EJECT_LATENCY + (dimension_changes * self.DIMENSION_CHANGE_LATENCY)

        return hops, total_delay_ns

    def schedule_event(self, time: float, event_type: str, data: Dict[str, Any], priority: Optional[int] = None):
        """调度一个新事件"""
        if priority is None:
            # 根据事件类型设置默认优先级
            priority_map = {
                "RETRY_CHECK": self.PRIORITY_RETRY_CHECK,
                "TRACKER_RELEASE": self.PRIORITY_TRACKER_RELEASE,
                "DATA_RECV": self.PRIORITY_DATA_RECV,
                "DATA_SEND": self.PRIORITY_DATA_SEND,
                "ACK_RECV": self.PRIORITY_ACK_RECV,
                "ACK_SEND": self.PRIORITY_ACK_SEND,
                "CMD_RECV": self.PRIORITY_CMD_RECV,
                "CMD_SEND": self.PRIORITY_CMD_SEND,
                "REQUEST_START": self.PRIORITY_REQUEST_START,
            }
            priority = priority_map.get(event_type, 999)

        event = Event(time=time, priority=priority, event_type=event_type, data=data)
        heapq.heappush(self.event_queue, event)

        if self.debug:
            print(f"[{time:.1f}ns] 调度事件: {event_type} (优先级{priority})")

    def _can_process_request(self, request: TrafficRequest) -> bool:
        """检查是否可以处理请求（两端tracker都有空闲）"""
        src_node = self.nodes[request.source_node]
        dest_node = self.nodes[request.dest_node]

        src_ip = src_node.get_ip_instance(request.source_ip_type)
        dest_ip = dest_node.get_ip_instance(request.dest_ip_type)

        if src_ip is None or dest_ip is None:
            return False

        if request.req_type == "R":
            return src_ip.tracker_rd.can_accept() and dest_ip.tracker_rd.can_accept()
        else:  # "W"
            return src_ip.tracker_wr.can_accept() and dest_ip.tracker_wr.can_accept()

    def _acquire_trackers(self, request: TrafficRequest):
        """获取tracker资源"""
        src_node = self.nodes[request.source_node]
        dest_node = self.nodes[request.dest_node]

        src_ip = src_node.get_ip_instance(request.source_ip_type)
        dest_ip = dest_node.get_ip_instance(request.dest_ip_type)

        if request.req_type == "R":
            src_ip.tracker_rd.inc()
            dest_ip.tracker_rd.inc()
        else:  # "W"
            src_ip.tracker_wr.inc()
            dest_ip.tracker_wr.inc()

    def _release_trackers(self, request: TrafficRequest):
        """释放tracker资源"""
        src_node = self.nodes[request.source_node]
        dest_node = self.nodes[request.dest_node]

        src_ip = src_node.get_ip_instance(request.source_ip_type)
        dest_ip = dest_node.get_ip_instance(request.dest_ip_type)

        if request.req_type == "R":
            src_ip.tracker_rd.dec()
            dest_ip.tracker_rd.dec()
        else:  # "W"
            src_ip.tracker_wr.dec()
            dest_ip.tracker_wr.dec()

    def _get_ip_instances(self, request: TrafficRequest) -> Tuple[IPState, IPState]:
        """获取请求涉及的IP实例"""
        src_node = self.nodes[request.source_node]
        dest_node = self.nodes[request.dest_node]

        src_ip = src_node.get_ip_instance(request.source_ip_type)
        dest_ip = dest_node.get_ip_instance(request.dest_ip_type)

        if src_ip is None:
            raise ValueError(f"源节点{request.source_node}没有IP类型{request.source_ip_type}")
        if dest_ip is None:
            raise ValueError(f"目标节点{request.dest_node}没有IP类型{request.dest_ip_type}")

        return src_ip, dest_ip

    def _print_progress(self):
        """打印仿真进度"""
        completed_count = len(self.completed_transactions)

        # 只在达到间隔时打印
        if completed_count % self.progress_interval == 0 or completed_count == self.total_requests:
            progress_percent = (completed_count / self.total_requests) * 100
            active_count = len(self.active_transactions)
            retry_count = len(self.retry_queue)

            print(f"进度: {completed_count}/{self.total_requests} ({progress_percent:.1f}%) " f"- 活跃事务: {active_count}, 等待重试: {retry_count}, " f"当前时间: {self.current_time:.1f}ns")

    # ==================== 事件处理器 ====================

    def _handle_request_start(self, event: Event):
        """处理请求开始事件"""
        request = event.data["request"]

        if self.debug:
            print(f"[{event.time:.1f}ns] 开始处理请求 {request.packet_id} ({request.req_type})")

        # 检查是否可以处理（tracker可用性）
        if not self._can_process_request(request):
            # 加入重试队列
            self.retry_queue.append(request)
            if self.debug:
                print(f"[{event.time:.1f}ns] 请求 {request.packet_id} tracker不可用，加入重试队列")
            return

        # 获取tracker资源
        self._acquire_trackers(request)

        # 计算路径延迟
        path_hops, path_delay = self.calculate_path_delay(request.source_node, request.dest_node)

        # 创建事务状态
        transaction_state = TransactionState(request=request, start_time=event.time, current_stage="CMD_SEND", tracker_acquired=True, path_hops=path_hops, path_delay=path_delay)

        self.active_transactions[request.packet_id] = transaction_state

        # 调度CMD发送事件
        self.schedule_event(time=event.time, event_type="CMD_SEND", data={"packet_id": request.packet_id})

    def _handle_cmd_send(self, event: Event):
        """处理CMD发送事件"""
        packet_id = event.data["packet_id"]
        transaction = self.active_transactions[packet_id]
        request = transaction.request

        src_ip, dest_ip = self._get_ip_instances(request)

        # 计算实际发送时间（考虑通道可用性）
        actual_send_time = max(event.time, src_ip.next_send_time["REQ"])
        src_ip.next_send_time["REQ"] = actual_send_time + 1

        transaction.cmd_send_time = actual_send_time
        transaction.current_stage = "CMD_RECV"

        if self.debug:
            print(f"[{actual_send_time:.1f}ns] CMD发送完成 - 请求 {packet_id}")

        # 调度CMD接收事件
        cmd_recv_time = actual_send_time + transaction.path_delay
        self.schedule_event(time=cmd_recv_time, event_type="CMD_RECV", data={"packet_id": packet_id})

    def _handle_cmd_recv(self, event: Event):
        """处理CMD接收事件"""
        packet_id = event.data["packet_id"]
        transaction = self.active_transactions[packet_id]
        request = transaction.request

        src_ip, dest_ip = self._get_ip_instances(request)

        # 计算实际接收时间（考虑通道可用性）
        actual_recv_time = max(event.time, dest_ip.next_recv_time["REQ"])
        dest_ip.next_recv_time["REQ"] = actual_recv_time + 1

        transaction.cmd_recv_time = actual_recv_time

        if self.debug:
            print(f"[{actual_recv_time:.1f}ns] CMD接收完成 - 请求 {packet_id}")

        if request.req_type == "R":
            # 读事务：直接处理数据发送
            self._schedule_read_data_processing(transaction, actual_recv_time)
        else:
            # 写事务：发送ACK
            transaction.current_stage = "ACK_SEND"
            self.schedule_event(time=actual_recv_time, event_type="ACK_SEND", data={"packet_id": packet_id})

    def _handle_ack_send(self, event: Event):
        """处理ACK发送事件（写事务）"""
        packet_id = event.data["packet_id"]
        transaction = self.active_transactions[packet_id]
        request = transaction.request

        src_ip, dest_ip = self._get_ip_instances(request)

        # 计算实际发送时间
        actual_send_time = max(event.time, dest_ip.next_send_time["RSP"])
        dest_ip.next_send_time["RSP"] = actual_send_time + 1

        transaction.ack_send_time = actual_send_time
        transaction.current_stage = "ACK_RECV"

        if self.debug:
            print(f"[{actual_send_time:.1f}ns] ACK发送完成 - 请求 {packet_id}")

        # 调度ACK接收事件
        ack_recv_time = actual_send_time + transaction.path_delay
        self.schedule_event(time=ack_recv_time, event_type="ACK_RECV", data={"packet_id": packet_id})

    def _handle_ack_recv(self, event: Event):
        """处理ACK接收事件（写事务）"""
        packet_id = event.data["packet_id"]
        transaction = self.active_transactions[packet_id]
        request = transaction.request

        src_ip, dest_ip = self._get_ip_instances(request)

        # 计算实际接收时间
        actual_recv_time = max(event.time, src_ip.next_recv_time["RSP"])
        src_ip.next_recv_time["RSP"] = actual_recv_time + 1

        transaction.ack_recv_time = actual_recv_time

        if self.debug:
            print(f"[{actual_recv_time:.1f}ns] ACK接收完成 - 请求 {packet_id}")

        # 开始数据发送
        self._schedule_write_data_sending(transaction, actual_recv_time)

    def _schedule_read_data_processing(self, transaction: TransactionState, start_time: float):
        """调度读事务的数据处理"""
        request = transaction.request

        # 计算存储器处理延迟
        if request.dest_ip_type.startswith("ddr"):
            process_latency = self.config.DDR_R_LATENCY
        else:  # l2m
            process_latency = self.config.L2M_R_LATENCY

        process_done_time = start_time + process_latency
        transaction.current_stage = "DATA_SEND"

        # 调度第一个数据flit发送
        self.schedule_event(time=process_done_time, event_type="DATA_SEND", data={"packet_id": request.packet_id, "flit_index": 0, "is_source_sending": False})  # dest_ip发送数据

    def _schedule_write_data_sending(self, transaction: TransactionState, start_time: float):
        """调度写事务的数据发送"""
        request = transaction.request
        transaction.current_stage = "DATA_SEND"

        # 调度第一个数据flit发送
        self.schedule_event(time=start_time, event_type="DATA_SEND", data={"packet_id": request.packet_id, "flit_index": 0, "is_source_sending": True})  # src_ip发送数据

    def _handle_data_send(self, event: Event):
        """处理数据发送事件"""
        packet_id = event.data["packet_id"]
        flit_index = event.data["flit_index"]
        is_source_sending = event.data["is_source_sending"]

        transaction = self.active_transactions[packet_id]
        request = transaction.request

        src_ip, dest_ip = self._get_ip_instances(request)

        # 确定发送方IP
        sender_ip = src_ip if is_source_sending else dest_ip

        # 计算理想发送时间
        if flit_index == 0:
            ideal_send_time = event.time
        else:
            ideal_send_time = max(event.time, transaction.data_send_times[-1] + 1)

        # 考虑通道可用性
        actual_send_time = max(ideal_send_time, sender_ip.next_send_time["DATA"])

        # 应用带宽限制
        bandwidth_limited_time = sender_ip.token_bucket.consume(128, actual_send_time)
        final_send_time = bandwidth_limited_time

        # 更新发送方状态
        sender_ip.next_send_time["DATA"] = final_send_time + 1
        transaction.data_send_times.append(final_send_time)

        if self.debug:
            print(f"[{final_send_time:.1f}ns] 数据发送 flit {flit_index} - 请求 {packet_id}")

        # 调度对应的接收事件
        data_recv_time = final_send_time + transaction.path_delay
        self.schedule_event(time=data_recv_time, event_type="DATA_RECV", data={"packet_id": packet_id, "flit_index": flit_index, "is_source_sending": is_source_sending})

        # 如果还有更多flit需要发送，调度下一个
        if flit_index + 1 < request.burst_length:
            next_send_time = final_send_time + 1
            self.schedule_event(time=next_send_time, event_type="DATA_SEND", data={"packet_id": packet_id, "flit_index": flit_index + 1, "is_source_sending": is_source_sending})

    def _handle_data_recv(self, event: Event):
        """处理数据接收事件"""
        packet_id = event.data["packet_id"]
        flit_index = event.data["flit_index"]
        is_source_sending = event.data["is_source_sending"]

        transaction = self.active_transactions[packet_id]
        request = transaction.request

        src_ip, dest_ip = self._get_ip_instances(request)

        # 确定接收方IP
        receiver_ip = dest_ip if is_source_sending else src_ip

        # 计算实际接收时间
        actual_recv_time = max(event.time, receiver_ip.next_recv_time["DATA"])
        receiver_ip.next_recv_time["DATA"] = actual_recv_time + 1

        transaction.data_recv_times.append(actual_recv_time)

        if self.debug:
            print(f"[{actual_recv_time:.1f}ns] 数据接收 flit {flit_index} - 请求 {packet_id}")

        # 检查是否所有数据都已接收
        if len(transaction.data_recv_times) == request.burst_length:
            # 所有数据接收完成，调度事务完成
            self._schedule_transaction_completion(transaction, actual_recv_time)

    def _schedule_transaction_completion(self, transaction: TransactionState, data_completion_time: float):
        """调度事务完成"""
        request = transaction.request

        if request.req_type == "W":
            # 写事务需要考虑写入延迟
            if request.dest_ip_type.startswith("ddr"):
                write_latency = self.config.DDR_W_LATENCY
            else:
                write_latency = self.config.L2M_W_LATENCY

            # 加上tracker释放延迟
            completion_time = data_completion_time + write_latency + self.config.SN_TRACKER_RELEASE_LATENCY
        else:
            # 读事务数据接收完成就算完成
            completion_time = data_completion_time

        transaction.completion_time = completion_time
        transaction.current_stage = "COMPLETE"

        # 调度tracker释放事件
        self.schedule_event(time=completion_time, event_type="TRACKER_RELEASE", data={"packet_id": request.packet_id})

    def _handle_tracker_release(self, event: Event):
        """处理tracker释放事件"""
        packet_id = event.data["packet_id"]
        transaction = self.active_transactions[packet_id]
        request = transaction.request

        # 释放tracker资源
        self._release_trackers(request)

        # 计算各种延迟
        cmd_latency = transaction.cmd_recv_time - transaction.cmd_send_time
        if request.req_type == "W":
            cmd_latency = transaction.ack_recv_time - transaction.cmd_send_time  # 包含ACK往返

        data_latency = transaction.data_recv_times[-1] - transaction.data_send_times[0]
        transaction_latency = transaction.completion_time - transaction.start_time

        # 创建结果记录
        result = TransactionResult(
            packet_id=packet_id,
            start_time=int(transaction.start_time),
            completion_time=int(transaction.completion_time),
            source_node=request.source_node,
            dest_node=request.dest_node,
            source_ip_type=request.source_ip_type,
            dest_ip_type=request.dest_ip_type,
            req_type=request.req_type,
            burst_length=request.burst_length,
            total_bytes=request.total_bytes,
            path_hops=transaction.path_hops,
            path_delay=transaction.path_delay,
            cmd_latency=int(cmd_latency),
            data_latency=int(data_latency),
            transaction_latency=int(transaction_latency),
        )

        self.completed_transactions.append(result)
        del self.active_transactions[packet_id]

        if self.debug:
            print(f"[{event.time:.1f}ns] 事务完成 - 请求 {packet_id} (延迟: {transaction_latency:.1f}ns)")

        # 进度报告
        self._print_progress()

        # 调度重试检查
        self.schedule_event(time=event.time, event_type="RETRY_CHECK", data={})

    def _handle_retry_check(self, event: Event):
        """处理重试检查事件"""
        if not self.retry_queue:
            return

        # 尝试处理等待队列中的请求
        processed_count = 0
        max_retry_per_check = 10  # 限制每次检查处理的重试数量，避免无限循环

        while self.retry_queue and processed_count < max_retry_per_check:
            request = self.retry_queue.popleft()

            if self._can_process_request(request):
                # 可以处理，重新调度为REQUEST_START事件
                self.schedule_event(time=event.time, event_type="REQUEST_START", data={"request": request})
                processed_count += 1
                if self.debug:
                    print(f"[{event.time:.1f}ns] 重试成功 - 请求 {request.packet_id}")
            else:
                # 还是不能处理，重新加回队列末尾
                self.retry_queue.append(request)
                break  # 避免无限循环

        # 如果还有等待的请求，调度下一次重试检查
        if self.retry_queue:
            self.schedule_event(time=event.time + 1, event_type="RETRY_CHECK", data={})  # 1ns后再次检查

    def handle_event(self, event: Event):
        """处理单个事件"""
        self.current_time = event.time

        handler = self.event_handlers.get(event.event_type)
        if handler:
            handler(event)
        else:
            print(f"警告: 未知事件类型 {event.event_type}")

    def simulate(self) -> List[TransactionResult]:
        """执行事件驱动仿真"""
        print(f"开始事件驱动仿真 {len(self.traffic_requests)} 个事务...")

        # 初始化：将所有请求转换为REQUEST_START事件
        for request in self.traffic_requests:
            self.schedule_event(time=request.start_time, event_type="REQUEST_START", data={"request": request})

        # 主事件循环
        event_count = 0
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.handle_event(event)

            event_count += 1
            if event_count % 10000 == 0 and self.debug:
                print(f"已处理 {event_count} 个事件，当前时间: {self.current_time:.1f}ns")

        print(f"事件驱动仿真完成:")
        print(f"  - 总事件数: {event_count}")
        print(f"  - 完成事务: {len(self.completed_transactions)}")
        print(f"  - 未完成事务: {len(self.active_transactions)}")
        print(f"  - 等待重试: {len(self.retry_queue)}")

        return self.completed_transactions

    def calculate_working_intervals(self, results: List[TransactionResult]) -> List[WorkingInterval]:
        """计算工作区间（与原版本兼容）"""
        if not results:
            return []

        # 创建事件列表：(时间点, 事件类型, 事务ID)
        events = []
        for result in results:
            events.append((result.start_time, "start", result.packet_id))
            events.append((result.completion_time, "end", result.packet_id))

        # 按时间排序
        events.sort()

        # 识别工作区间
        raw_intervals = []
        active_requests = set()
        current_start = None

        for time_point, event_type, req_id in events:
            if event_type == "start":
                if not active_requests:  # 开始新的工作区间
                    current_start = time_point
                active_requests.add(req_id)
            else:  # end
                active_requests.discard(req_id)
                if not active_requests and current_start is not None:
                    # 工作区间结束
                    raw_intervals.append((current_start, time_point))
                    current_start = None

        # 合并相近的区间
        merged_intervals = self._merge_close_intervals(raw_intervals)

        # 构建WorkingInterval对象
        working_intervals = []
        for start, end in merged_intervals:
            # 找到该区间内的所有请求
            interval_results = [r for r in results if not (r.completion_time < start or r.start_time > end)]

            flit_count = sum(r.burst_length for r in interval_results)
            total_bytes = sum(r.total_bytes for r in interval_results)

            interval = WorkingInterval(start_time=start, end_time=end, duration=end - start, flit_count=flit_count, total_bytes=total_bytes, request_count=len(interval_results))
            working_intervals.append(interval)

        return working_intervals

    def _merge_close_intervals(self, intervals: List[tuple]) -> List[tuple]:
        """合并相近的工作区间"""
        if not intervals:
            return []

        merged = [intervals[0]]
        for current_start, current_end in intervals[1:]:
            last_start, last_end = merged[-1]

            # 如果当前区间与上一个区间间隔小于阈值，则合并
            if current_start - last_end <= self.config.MIN_GAP_THRESHOLD:
                merged[-1] = (last_start, current_end)
            else:
                merged.append((current_start, current_end))

        return merged


# ==================== 辅助函数（与原版本保持兼容） ====================


def _save_csv_results(results, output_path, verbose=True):
    """保存CSV结果文件"""
    if verbose:
        print(f"保存CSV结果到: {output_path}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 写入头部
        headers = [
            "数据包ID",
            "开始时间",
            "完成时间",
            "源节点",
            "目标节点",
            "源IP类型",
            "目标IP类型",
            "请求类型",
            "突发长度",
            "总字节数",
            "路径跳数",
            "路径延迟",
            "命令延迟",
            "数据延迟",
            "事务延迟",
        ]
        writer.writerow(headers)

        # 写入数据
        for result in results:
            writer.writerow(
                [
                    result.packet_id,
                    result.start_time,
                    result.completion_time,
                    result.source_node,
                    result.dest_node,
                    result.source_ip_type,
                    result.dest_ip_type,
                    result.req_type,
                    result.burst_length,
                    result.total_bytes,
                    result.path_hops,
                    result.path_delay,
                    result.cmd_latency,
                    result.data_latency,
                    result.transaction_latency,
                ]
            )


def _print_statistics(results, calculator=None, verbose=True, print_latency_stats=True):
    """打印统计信息"""
    if not verbose:
        return

    print("\n=== 事件驱动仿真带宽统计 ===")

    # 按IP类型分组统计（不区分读写）
    type_groups = defaultdict(list)
    rn_ip_count = defaultdict(set)  # 统计每种类型的RN IP实例数量

    for result in results:
        ip_type = result.source_ip_type.split("_")[0].upper()
        type_groups[ip_type].append(result)

        # 统计RN IP实例（节点+IP类型组合）
        rn_ip_instance = f"{result.source_node}-{result.source_ip_type}"
        rn_ip_count[ip_type].add(rn_ip_instance)

    weighted_bandwidth_info = []

    for ip_type, group_results in type_groups.items():
        if not group_results:
            continue

        # 使用工作区间计算带宽
        if calculator is not None:
            working_intervals = calculator.calculate_working_intervals(group_results)

            if working_intervals:
                # 计算加权带宽：各区间带宽按flit数量加权平均
                total_weighted_bandwidth = 0.0
                total_weight = 0

                for interval in working_intervals:
                    weight = interval.flit_count  # 权重是工作区间的flit数量
                    bandwidth = interval.bandwidth_bytes_per_ns  # bytes/ns
                    total_weighted_bandwidth += bandwidth * weight
                    total_weight += weight

                weighted_bandwidth = (total_weighted_bandwidth / total_weight) if total_weight > 0 else 0.0
            else:
                weighted_bandwidth = 0.0
        else:
            # 回退到简单计算
            total_bytes = sum(r.total_bytes for r in group_results)
            min_start = min(r.start_time for r in group_results)
            max_completion = max(r.completion_time for r in group_results)
            time_span = max_completion - min_start
            weighted_bandwidth = (total_bytes / time_span) if time_span > 0 else 0.0

        # 获取该IP类型的RN IP实例数量
        num_rn_ips = len(rn_ip_count[ip_type])

        # 计算每个IP实例的平均带宽
        avg_bandwidth_per_ip = weighted_bandwidth / num_rn_ips if num_rn_ips > 0 else 0

        weighted_bandwidth_info.append((ip_type, weighted_bandwidth, avg_bandwidth_per_ip, num_rn_ips))

        print(f"{ip_type:15s}: {weighted_bandwidth:.2f} GB/s ({num_rn_ips}个实例，平均每实例 {avg_bandwidth_per_ip:.2f} GB/s)")

    # 计算总的加权带宽
    total_bandwidth = sum(info[1] for info in weighted_bandwidth_info)
    total_avg_bandwidth = sum(info[2] for info in weighted_bandwidth_info)

    print(f"{'总加权带宽':15s}: {total_bandwidth:.2f} GB/s")
    print(f"{'平均加权带宽':15s}: {total_avg_bandwidth:.2f} GB/s")

    # 统计工作区间数量
    if calculator is not None:
        total_working_intervals = calculator.calculate_working_intervals(results)
        print(f"{'工作区间数':15s}: {len(total_working_intervals)}")

    # 延迟统计
    if print_latency_stats:
        print("\n=== 延迟统计 ===")
        all_cmd_latencies = [r.cmd_latency for r in results]
        all_data_latencies = [r.data_latency for r in results]
        all_trans_latencies = [r.transaction_latency for r in results]

        # CMD延迟统计
        print(f"CMD延迟  - 平均: {np.mean(all_cmd_latencies):.1f} ns, " f"最小: {np.min(all_cmd_latencies):.1f} ns, " f"最大: {np.max(all_cmd_latencies):.1f} ns")

        # 数据延迟统计
        print(f"数据延迟 - 平均: {np.mean(all_data_latencies):.1f} ns, " f"最小: {np.min(all_data_latencies):.1f} ns, " f"最大: {np.max(all_data_latencies):.1f} ns")

        # 事务延迟统计
        print(f"事务延迟 - 平均: {np.mean(all_trans_latencies):.1f} ns, " f"最小: {np.min(all_trans_latencies):.1f} ns, " f"最大: {np.max(all_trans_latencies):.1f} ns")


def _plot_bandwidth_curve(results, base_name, topo_type, save_path, calculator=None):
    """绘制带宽曲线（基于工作区间）"""
    try:
        # 设置中文字体
        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # 按IP类型分组（与统计函数保持一致，不区分读写）
        type_groups = defaultdict(list)
        for result in results:
            ip_type = result.source_ip_type.split("_")[0].upper()
            type_groups[ip_type].append(result)

        plt.figure(figsize=(12, 8))

        for group_key, group_results in type_groups.items():
            if not group_results or calculator is None:
                continue

            # 计算该分组的工作区间
            working_intervals = calculator.calculate_working_intervals(group_results)

            if not working_intervals:
                continue

            # 为每个工作区间绘制独立的带宽曲线段
            for i, interval in enumerate(working_intervals):
                # 获取该区间内的事务
                interval_results = [r for r in group_results if interval.start_time <= r.completion_time <= interval.end_time]

                if not interval_results:
                    continue

                # 按完成时间排序
                interval_results.sort(key=lambda x: x.completion_time)

                # 计算相对时间（从该区间的start_time开始）
                rel_times = np.array([r.completion_time - interval.start_time for r in interval_results])
                rel_times_us = rel_times / 1000.0  # 转换为μs

                # 计算累积字节数
                cumulative_bytes = np.cumsum([r.total_bytes for r in interval_results])

                # 计算带宽（使用相对时间，避免除零）
                bandwidth = np.zeros_like(rel_times_us)
                for j in range(len(rel_times_us)):
                    if rel_times[j] > 0:  # 使用ns时间避免精度问题
                        bandwidth[j] = cumulative_bytes[j] / (rel_times[j])  # GB/s
                    else:
                        bandwidth[j] = 0.0

                # 转换回绝对时间用于绘图
                abs_times_us = (rel_times + interval.start_time) / 1000.0

                # 为每个工作区间绘制独立的曲线段
                # 只有第一个区间显示标签，避免重复
                label = f"{group_key} (事件驱动)" if i == 0 else None
                plt.plot(abs_times_us, bandwidth, label=label, drawstyle="steps-post")

        plt.xlabel("时间 (μs)")
        plt.ylabel("带宽 (GB/s)")
        plt.title(f"事件驱动理论带宽 - {base_name} ({topo_type})")
        plt.legend()
        plt.grid(True)

        # 保存图片或显示
        if save_path:
            output_fig = os.path.join(save_path, f"{base_name}_{topo_type}_event_driven_bw.png")
            plt.savefig(output_fig, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"事件驱动带宽曲线图已保存: {output_fig}")
        else:
            plt.show()
            print("事件驱动带宽曲线图已显示")

    except Exception as e:
        print(f"绘制带宽曲线时发生错误: {e}")


def main():
    """事件驱动理论带宽计算主函数"""

    # ==================== 流量配置 ====================
    # traffic_file_path = r"../../test_data/"
    # traffic_file_path = r"../../../C2C/traffic_data"
    # traffic_file_path = r"../../traffic/traffic0730"
    # traffic_file_path = r"../../example/"
    traffic_file_path = r"../../traffic/0617/"
    # traffic_file_path = r"../../traffic/DeepSeek_0616/step6_ch_map_all/"
    # traffic_file_path = r"../../traffic/RW_4x2_4x4/"
    # traffic_file_path = r"../../traffic/nxn_traffics"

    traffic_config = [
        [
            # r"R_5x2.txt",
            # r"W_5x2.txt",
            # r"Read_burst4_2262HBM_v2.txt",
            # r"Write_burst4_2262HBM_v2.txt",
            # r"MLP_MoE.txt",
        ]
        * 1,
        [
            # r"All2All_Combine.txt",
            # r"All2All_Dispatch.txt",
            # r"full_bw_R_4x5.txt"
            # "LLama2_AllReduce.txt"
            # "test_data.txt"
            # "traffic_2260E_case1.txt",
            "LLama2_AttentionFC.txt"
            # "W_8x8.txt"
            # "MLA_B32.txt"
            # "Add.txt"
            # "attn_fc.txt"
            # "moe_gate.txt"
        ],
    ]

    # ==================== 输出配置 ====================
    result_save_path = f"../../Result/TheoreticalBandwidth_EventDriven/"
    results_fig_save_path = None

    # ==================== 拓扑配置 ====================
    topo_config_map = {
        "3x3": r"../../config/topologies/topo_3x3.yaml",
        "4x4": r"../../config/topologies/topo_4x4.yaml",
        "5x2": r"../../config/topologies/topo_5x2.yaml",
        "5x4": r"../../config/topologies/topo_5x4.yaml",
        "6x5": r"../../config/topologies/topo_6x5.yaml",
        "8x8": r"../../config/topologies/topo_8x8.yaml",
    }

    topo_type = "5x4"  # SG2262

    # ==================== 仿真参数 ====================
    verbose = 1  # 详细输出
    save_csv_results = 1  # 保存CSV结果
    plot_bandwidth_curve = 1  # 绘制带宽曲线
    print_latency_stats = 1  # 打印延迟统计
    debug_mode = 0  # 调试模式

    # ==================== 执行仿真 ====================
    config_path = topo_config_map.get(topo_type, r"../config/default.yaml")

    # 确保输出路径存在
    os.makedirs(result_save_path, exist_ok=True)
    if results_fig_save_path:
        os.makedirs(results_fig_save_path, exist_ok=True)

    # 处理所有流量配置
    for config_group in traffic_config:
        for traffic_file in config_group:
            if verbose:
                print(f"\n{'='*60}")
                print(f"处理流量文件: {traffic_file}")
                print(f"拓扑类型: {topo_type}")
                print(f"配置文件: {config_path}")
                print(f"仿真模式: 事件驱动")
                print(f"{'='*60}")

            # 构建完整的流量文件路径
            full_traffic_path = os.path.join(traffic_file_path, traffic_file)

            if not os.path.exists(full_traffic_path):
                print(f"警告: 流量文件不存在: {full_traffic_path}")
                continue

            try:
                # 创建事件驱动理论带宽计算器
                calculator = EventDrivenCalculator(config_file=config_path, traffic_file=full_traffic_path, debug=debug_mode)

                # 执行事件驱动仿真
                results = calculator.simulate()

                if not results:
                    print("警告: 没有有效的仿真结果")
                    continue

                # 生成输出文件名
                base_name = os.path.splitext(traffic_file)[0]
                output_csv = os.path.join(result_save_path, f"{base_name}_{topo_type}_event_driven.csv")

                # 保存CSV结果
                if save_csv_results:
                    _save_csv_results(results, output_csv, verbose)

                # 计算并输出统计信息
                _print_statistics(results, calculator, verbose, print_latency_stats)

                # 绘制带宽曲线（如果启用）
                if plot_bandwidth_curve:
                    _plot_bandwidth_curve(results, base_name, topo_type, results_fig_save_path, calculator)

            except Exception as e:
                print(f"错误: 处理文件 {traffic_file} 时发生异常: {e}")
                import traceback

                traceback.print_exc()
                continue

    if verbose:
        print(f"\n{'='*60}")
        print("事件驱动理论带宽计算完成!")
        print(f"结果保存路径: {result_save_path}")
        if results_fig_save_path:
            print(f"图表保存路径: {results_fig_save_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    exit(main())
