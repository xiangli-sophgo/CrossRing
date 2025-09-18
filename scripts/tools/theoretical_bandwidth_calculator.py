#!/usr/bin/env python3
"""
理论带宽计算工具 - 基于CrossRing网络拓扑的精确事务模拟
支持三通道独立性、Token Bucket带宽限制和精确延迟计算
"""

import math
import csv
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
import sys

if sys.platform == "darwin":  # macOS
    try:
        matplotlib.use("macosx")
    except ImportError:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import argparse
import sys
import os
from datetime import datetime

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
    DDR_R_LATENCY: int = 40
    DDR_W_LATENCY: int = 0
    L2M_R_LATENCY: int = 12
    L2M_W_LATENCY: int = 16
    SN_TRACKER_RELEASE_LATENCY: int = 40

    # Tracker配置 (Outstanding限制)
    RN_R_TRACKER_OSTD: int = 64
    RN_W_TRACKER_OSTD: int = 64
    SN_DDR_R_TRACKER_OSTD: int = 64
    SN_DDR_W_TRACKER_OSTD: int = 64
    SN_L2M_R_TRACKER_OSTD: int = 64
    SN_L2M_W_TRACKER_OSTD: int = 64

    # 工作区间配置
    MIN_GAP_THRESHOLD: int = 20  # ns，小于此值的间隔被视为同一工作区间

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


class TheoreticalBandwidthCalculator:
    """理论带宽计算器"""

    # 网络延迟参数 (ns) - 可直接在此处调整
    INJECT_LATENCY = 0  # 注入延迟
    EJECT_LATENCY = 0  # 弹出延迟
    DIMENSION_CHANGE_LATENCY = 0  # 维度转换延迟（XY路由最多1次）

    def __init__(self, traffic_file: str):
        self.config = Config()
        self.config.__post_init__()
        self.traffic_requests = self._load_traffic(traffic_file)

        # 构建网络拓扑
        self.adjacency_matrix = self._build_topology()
        if find_shortest_paths is not None:
            self.shortest_paths = find_shortest_paths(self.adjacency_matrix)
        else:
            self.shortest_paths = self._compute_simple_shortest_paths()

        # 初始化节点状态
        self.nodes = self._initialize_nodes()

        # 结果存储
        self.results: List[TransactionResult] = []

        # 事务队列（用于tracker满时的重试）
        self.pending_requests = []

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

        # 为简化起见，我们假设：
        # - 每个节点可能有GDMA、SDMA、DDR、L2M等不同类型的IP
        # - 每种类型的IP在节点内部编号从0开始

        for node_id in range(self.config.NUM_NODE):
            node = nodes[node_id]

            # 根据节点位置和配置确定该节点有哪些IP类型
            # 这里简化处理：假设所有节点都可能有所有类型的IP
            # 实际项目中应该根据具体的拓扑配置来确定

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
            raise ValueError

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

    def _manhattan_distance(self, src: int, dest: int) -> int:
        """计算曼哈顿距离（用于估算跳数）"""
        src_row, src_col = divmod(src, self.config.NUM_COL)
        dest_row, dest_col = divmod(dest, self.config.NUM_COL)
        return abs(src_row - dest_row) + abs(src_col - dest_col)

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

    def calculate_working_intervals(self, results: List[TransactionResult]) -> List[WorkingInterval]:
        """
        计算工作区间，去除空闲时间段

        Args:
            results: 事务结果列表

        Returns:
            工作区间列表
        """
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

    def process_read_transaction(self, request: TrafficRequest) -> TransactionResult:
        """处理读事务"""
        # 检查tracker可用性，如果不可用则抛出异常用于重试
        if not self._can_process_request(request):
            raise RuntimeError("Tracker not available")

        # 获取tracker资源
        self._acquire_trackers(request)

        # 获取源节点和目标节点
        src_node = self.nodes[request.source_node]
        dest_node = self.nodes[request.dest_node]

        # 获取对应的IP实例
        src_ip = src_node.get_ip_instance(request.source_ip_type)
        dest_ip = dest_node.get_ip_instance(request.dest_ip_type)

        if src_ip is None:
            raise ValueError(f"源节点{request.source_node}没有IP类型{request.source_ip_type}")
        if dest_ip is None:
            raise ValueError(f"目标节点{request.dest_node}没有IP类型{request.dest_ip_type}")

        path_hops, path_delay = self.calculate_path_delay(request.source_node, request.dest_node)

        # 1. 发送CMD请求 (REQ通道)
        cmd_send_time = max(request.start_time, src_ip.next_send_time["REQ"])
        src_ip.next_send_time["REQ"] = cmd_send_time + 1

        # 2. 接收CMD (REQ通道)
        cmd_recv_time = cmd_send_time + path_delay
        cmd_recv_time = max(cmd_recv_time, dest_ip.next_recv_time["REQ"])
        dest_ip.next_recv_time["REQ"] = cmd_recv_time + 1

        # 3. 存储器处理延迟
        if request.dest_ip_type.startswith("ddr"):
            process_latency = self.config.DDR_R_LATENCY
        else:  # l2m
            process_latency = self.config.L2M_R_LATENCY
        process_done_time = cmd_recv_time + process_latency

        # 4. 发送数据 (DATA通道) - BURST个flit
        data_send_times = []

        for i in range(request.burst_length):
            # 计算理想发送时间
            ideal_send_time = max(process_done_time if i == 0 else data_send_times[-1] + 1, dest_ip.next_send_time["DATA"])

            # 应用带宽限制（128字节per flit）
            actual_send_time = dest_ip.token_bucket.consume(128, ideal_send_time)

            dest_ip.next_send_time["DATA"] = actual_send_time + 1
            data_send_times.append(actual_send_time)

        # 5. 接收数据 (DATA通道)
        data_recv_times = []
        for i in range(request.burst_length):
            recv_time = data_send_times[i] + path_delay
            recv_time = max(recv_time, src_ip.next_recv_time["DATA"])
            src_ip.next_recv_time["DATA"] = recv_time + 1
            data_recv_times.append(recv_time)

        # 6. 数据接收完成后释放tracker
        completion_time = data_recv_times[-1]
        self._release_trackers(request)

        # 计算各种延迟
        cmd_latency = cmd_recv_time - cmd_send_time
        # 数据延迟：从第一个数据flit发送开始到最后一个数据flit接收完成
        data_latency = data_recv_times[-1] - data_send_times[0]
        transaction_latency = completion_time - request.start_time

        return TransactionResult(
            packet_id=request.packet_id,
            start_time=request.start_time,
            completion_time=completion_time,
            source_node=request.source_node,
            dest_node=request.dest_node,
            source_ip_type=request.source_ip_type,
            dest_ip_type=request.dest_ip_type,
            req_type=request.req_type,
            burst_length=request.burst_length,
            total_bytes=request.total_bytes,
            path_hops=path_hops,
            path_delay=path_delay,
            cmd_latency=cmd_latency,
            data_latency=data_latency,
            transaction_latency=transaction_latency,
        )

    def process_write_transaction(self, request: TrafficRequest) -> TransactionResult:
        """处理写事务"""
        # 检查tracker可用性，如果不可用则抛出异常用于重试
        if not self._can_process_request(request):
            raise RuntimeError("Tracker not available")

        # 获取tracker资源
        self._acquire_trackers(request)

        # 获取源节点和目标节点
        src_node = self.nodes[request.source_node]
        dest_node = self.nodes[request.dest_node]

        # 获取对应的IP实例
        src_ip = src_node.get_ip_instance(request.source_ip_type)
        dest_ip = dest_node.get_ip_instance(request.dest_ip_type)

        if src_ip is None:
            raise ValueError(f"源节点{request.source_node}没有IP类型{request.source_ip_type}")
        if dest_ip is None:
            raise ValueError(f"目标节点{request.dest_node}没有IP类型{request.dest_ip_type}")

        path_hops, path_delay = self.calculate_path_delay(request.source_node, request.dest_node)

        # 1. 发送CMD请求 (REQ通道)
        cmd_send_time = max(request.start_time, src_ip.next_send_time["REQ"])
        src_ip.next_send_time["REQ"] = cmd_send_time + 1

        # 2. 接收CMD (REQ通道)
        cmd_recv_time = cmd_send_time + path_delay
        cmd_recv_time = max(cmd_recv_time, dest_ip.next_recv_time["REQ"])
        dest_ip.next_recv_time["REQ"] = cmd_recv_time + 1

        # 3. 发送ACK响应 (RSP通道)
        ack_send_time = max(cmd_recv_time, dest_ip.next_send_time["RSP"])
        dest_ip.next_send_time["RSP"] = ack_send_time + 1

        # 4. 接收ACK (RSP通道)
        ack_recv_time = ack_send_time + path_delay
        ack_recv_time = max(ack_recv_time, src_ip.next_recv_time["RSP"])
        src_ip.next_recv_time["RSP"] = ack_recv_time + 1

        # 5. 发送数据 (DATA通道) - BURST个flit
        data_send_times = []

        for i in range(request.burst_length):
            # 计算理想发送时间
            ideal_send_time = max(ack_recv_time if i == 0 else data_send_times[-1] + 1, src_ip.next_send_time["DATA"])

            # 应用带宽限制（128字节per flit）
            actual_send_time = src_ip.token_bucket.consume(128, ideal_send_time)

            src_ip.next_send_time["DATA"] = actual_send_time + 1
            data_send_times.append(actual_send_time)

        # 6. 接收数据 (DATA通道)
        data_recv_times = []
        for i in range(request.burst_length):
            recv_time = data_send_times[i] + path_delay
            recv_time = max(recv_time, dest_ip.next_recv_time["DATA"])
            dest_ip.next_recv_time["DATA"] = recv_time + 1
            data_recv_times.append(recv_time)

        # 7. 存储器写入 + Tracker释放延迟
        if request.dest_ip_type.startswith("ddr"):
            write_latency = self.config.DDR_W_LATENCY
        else:
            write_latency = self.config.L2M_W_LATENCY

        # 8. 数据接收完成后经过release_latency释放tracker
        tracker_release_time = data_recv_times[-1] + write_latency + self.config.SN_TRACKER_RELEASE_LATENCY
        completion_time = tracker_release_time
        self._release_trackers(request)

        # 计算各种延迟
        cmd_latency = ack_recv_time - cmd_send_time  # 包含ACK往返
        data_latency = data_recv_times[-1] - data_send_times[0]
        transaction_latency = completion_time - request.start_time

        return TransactionResult(
            packet_id=request.packet_id,
            start_time=request.start_time,
            completion_time=completion_time,
            source_node=request.source_node,
            dest_node=request.dest_node,
            source_ip_type=request.source_ip_type,
            dest_ip_type=request.dest_ip_type,
            req_type=request.req_type,
            burst_length=request.burst_length,
            total_bytes=request.total_bytes,
            path_hops=path_hops,
            path_delay=path_delay,
            cmd_latency=cmd_latency,
            data_latency=data_latency,
            transaction_latency=transaction_latency,
        )

    def simulate(self, print_interval=1000, simulation_end_time=None) -> List[TransactionResult]:
        """执行仿真

        Args:
            print_interval: 每处理多少个事务打印一次进度
            simulation_end_time: 仿真结束时间(ns)，None表示处理所有事务
        """
        print(f"开始仿真 {len(self.traffic_requests)} 个事务...")
        if simulation_end_time is not None:
            print(f"仿真结束时间: {simulation_end_time} ns")

        # 按开始时间排序处理
        sorted_requests = sorted(self.traffic_requests, key=lambda x: x.start_time)

        # 如果设置了结束时间，过滤掉超出时间的请求
        if simulation_end_time is not None:
            sorted_requests = [req for req in sorted_requests if req.start_time <= simulation_end_time]
            print(f"在结束时间内的事务数量: {len(sorted_requests)}")

        results = []
        retry_queue = []

        for i, request in enumerate(sorted_requests):
            retry_count = 0
            current_request = request

            while True:
                try:
                    if current_request.req_type == "R":
                        result = self.process_read_transaction(current_request)
                    elif current_request.req_type == "W":
                        result = self.process_write_transaction(current_request)
                    else:
                        print(f"警告: 未知请求类型 {current_request.req_type}")
                        break

                    results.append(result)
                    break

                except RuntimeError as e:
                    if "Tracker not available" in str(e):
                        # Tracker满，延迟1ns后重试
                        retry_count += 1
                        if retry_count > 1000:  # 防止无限重试
                            print(f"警告: 事务 {current_request.packet_id} 重试超过1000次，跳过")
                            break

                        # 创建延迟后的请求
                        current_request = TrafficRequest(
                            start_time=current_request.start_time + 1,
                            source_node=current_request.source_node,
                            source_ip_type=current_request.source_ip_type,
                            dest_node=current_request.dest_node,
                            dest_ip_type=current_request.dest_ip_type,
                            req_type=current_request.req_type,
                            burst_length=current_request.burst_length,
                            packet_id=current_request.packet_id,
                        )
                        continue
                    else:
                        # 其他错误，重新抛出
                        raise

            if (i + 1) % print_interval == 0:
                print(f"已处理 {i + 1}/{len(sorted_requests)} 个事务")

        self.results = results
        print(f"仿真完成，共处理 {len(results)} 个事务")
        return results


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


def _save_combined_csv_results(all_results_with_info, output_path, verbose=True):
    """保存合并的CSV结果文件，包含流量文件信息"""
    if verbose:
        print(f"保存合并CSV结果到: {output_path}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 写入头部（增加流量文件名列）
        headers = [
            "流量文件名",
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
        for traffic_file, results in all_results_with_info:
            for result in results:
                writer.writerow(
                    [
                        traffic_file,
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


def calculate_statistics_summary(results, calculator=None):
    """
    计算统计汇总结果

    Args:
        results: 事务结果列表
        calculator: 计算器实例

    Returns:
        dict: 统计汇总字典
    """
    if not results:
        return {}

    # 按IP类型分组统计
    type_groups = defaultdict(list)
    rn_ip_count = defaultdict(set)

    for result in results:
        ip_type = result.source_ip_type.split("_")[0].upper()
        type_groups[ip_type].append(result)

        # 统计RN IP实例
        rn_ip_instance = f"{result.source_node}-{result.source_ip_type}"
        rn_ip_count[ip_type].add(rn_ip_instance)

    # 计算带宽统计
    bandwidth_stats = {}
    total_bandwidth = 0.0
    total_avg_bandwidth = 0.0

    for ip_type, group_results in type_groups.items():
        if not group_results:
            continue

        # 使用工作区间计算带宽
        if calculator is not None:
            working_intervals = calculator.calculate_working_intervals(group_results)

            if working_intervals:
                # 计算加权带宽：Σ(区间数据量 × 区间带宽) / Σ(区间数据量)
                total_weighted_bandwidth = 0.0
                total_data_bytes = 0.0

                for interval in working_intervals:
                    interval_bandwidth = interval.bandwidth_bytes_per_ns  # 区间带宽 (bytes/ns)
                    interval_data_bytes = interval.total_bytes  # 区间数据量 (bytes)

                    total_weighted_bandwidth += interval_data_bytes * interval_bandwidth
                    total_data_bytes += interval_data_bytes

                weighted_bandwidth = (total_weighted_bandwidth / total_data_bytes) if total_data_bytes > 0 else 0.0
            else:
                weighted_bandwidth = 0.0
        else:
            # calculator为None时，无法计算加权带宽
            weighted_bandwidth = 0.0

        # 获取IP实例数量
        num_rn_ips = len(rn_ip_count[ip_type])
        avg_bandwidth_per_ip = weighted_bandwidth / num_rn_ips if num_rn_ips > 0 else 0

        bandwidth_stats[ip_type] = {"weighted_bandwidth": weighted_bandwidth, "avg_bandwidth_per_ip": avg_bandwidth_per_ip, "num_instances": num_rn_ips}

        total_bandwidth += weighted_bandwidth
        total_avg_bandwidth += avg_bandwidth_per_ip

    # 计算延迟统计
    all_cmd_latencies = [r.cmd_latency for r in results]
    all_data_latencies = [r.data_latency for r in results]
    all_trans_latencies = [r.transaction_latency for r in results]

    # 计算工作区间数量
    working_intervals_count = 0
    if calculator is not None:
        total_working_intervals = calculator.calculate_working_intervals(results)
        working_intervals_count = len(total_working_intervals)

    return {
        "bandwidth_stats": bandwidth_stats,
        "total_bandwidth": total_bandwidth,
        "total_avg_bandwidth": total_avg_bandwidth,
        "working_intervals_count": working_intervals_count,
        "latency_stats": {
            "cmd_latency": {"avg": np.mean(all_cmd_latencies), "min": np.min(all_cmd_latencies), "max": np.max(all_cmd_latencies)},
            "data_latency": {"avg": np.mean(all_data_latencies), "min": np.min(all_data_latencies), "max": np.max(all_data_latencies)},
            "trans_latency": {"avg": np.mean(all_trans_latencies), "min": np.min(all_trans_latencies), "max": np.max(all_trans_latencies)},
        },
        "total_transactions": len(results),
    }


def _save_statistics_summary(all_results_with_info, output_path, topo_type, verbose=True):
    """保存统计汇总结果到标准CSV格式"""
    if verbose:
        print(f"保存统计汇总结果到: {output_path}")

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        # 写入CSV头部
        headers = [
            "流量文件名",
            "拓扑类型",
            "生成时间",
            "IP类型",
            "加权带宽(GB/s)",
            "平均每实例带宽(GB/s)",
            "IP实例数量",
            "总加权带宽(GB/s)",
            "平均加权带宽(GB/s)",
            "工作区间数",
            "总事务数",
            "CMD延迟_平均(ns)",
            "CMD延迟_最小(ns)",
            "CMD延迟_最大(ns)",
            "数据延迟_平均(ns)",
            "数据延迟_最小(ns)",
            "数据延迟_最大(ns)",
            "事务延迟_平均(ns)",
            "事务延迟_最小(ns)",
            "事务延迟_最大(ns)",
        ]
        writer.writerow(headers)

        # 当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for traffic_file, results, calculator in all_results_with_info:
            # 使用保存的calculator实例来计算准确的工作区间
            stats = calculate_statistics_summary(results, calculator)

            if not stats:
                continue

            latency_stats = stats["latency_stats"]

            # 为每个IP类型写入一行数据
            for ip_type, bandwidth_info in stats["bandwidth_stats"].items():
                row = [
                    traffic_file,  # 流量文件名
                    topo_type,  # 拓扑类型
                    current_time,  # 生成时间
                    ip_type,  # IP类型
                    f"{bandwidth_info['weighted_bandwidth']:.2f}",  # 加权带宽
                    f"{bandwidth_info['avg_bandwidth_per_ip']:.2f}",  # 平均每实例带宽
                    bandwidth_info["num_instances"],  # IP实例数量
                    f"{stats['total_bandwidth']:.2f}",  # 总加权带宽
                    f"{stats['total_avg_bandwidth']:.2f}",  # 平均加权带宽
                    stats["working_intervals_count"],  # 工作区间数
                    stats["total_transactions"],  # 总事务数
                    f"{latency_stats['cmd_latency']['avg']:.1f}",  # CMD延迟_平均
                    f"{latency_stats['cmd_latency']['min']:.1f}",  # CMD延迟_最小
                    f"{latency_stats['cmd_latency']['max']:.1f}",  # CMD延迟_最大
                    f"{latency_stats['data_latency']['avg']:.1f}",  # 数据延迟_平均
                    f"{latency_stats['data_latency']['min']:.1f}",  # 数据延迟_最小
                    f"{latency_stats['data_latency']['max']:.1f}",  # 数据延迟_最大
                    f"{latency_stats['trans_latency']['avg']:.1f}",  # 事务延迟_平均
                    f"{latency_stats['trans_latency']['min']:.1f}",  # 事务延迟_最小
                    f"{latency_stats['trans_latency']['max']:.1f}",  # 事务延迟_最大
                ]
                writer.writerow(row)

            # 如果一个文件没有IP类型数据，至少写入文件级的汇总信息
            if not stats["bandwidth_stats"]:
                row = [
                    traffic_file,  # 流量文件名
                    topo_type,  # 拓扑类型
                    current_time,  # 生成时间
                    "N/A",  # IP类型
                    "0.00",  # 加权带宽
                    "0.00",  # 平均每实例带宽
                    0,  # IP实例数量
                    f"{stats['total_bandwidth']:.2f}",  # 总加权带宽
                    f"{stats['total_avg_bandwidth']:.2f}",  # 平均加权带宽
                    stats["working_intervals_count"],  # 工作区间数
                    stats["total_transactions"],  # 总事务数
                    f"{latency_stats['cmd_latency']['avg']:.1f}",  # CMD延迟_平均
                    f"{latency_stats['cmd_latency']['min']:.1f}",  # CMD延迟_最小
                    f"{latency_stats['cmd_latency']['max']:.1f}",  # CMD延迟_最大
                    f"{latency_stats['data_latency']['avg']:.1f}",  # 数据延迟_平均
                    f"{latency_stats['data_latency']['min']:.1f}",  # 数据延迟_最小
                    f"{latency_stats['data_latency']['max']:.1f}",  # 数据延迟_最大
                    f"{latency_stats['trans_latency']['avg']:.1f}",  # 事务延迟_平均
                    f"{latency_stats['trans_latency']['min']:.1f}",  # 事务延迟_最小
                    f"{latency_stats['trans_latency']['max']:.1f}",  # 事务延迟_最大
                ]
                writer.writerow(row)


def process_traffic_folder(
    traffic_folder_path,
    topo_type,
    result_save_path,
    print_interval=1000,
    simulation_end_time=None,
    verbose=True,
    save_individual_csv=True,
    save_combined_csv=True,
    plot_bandwidth_curve=False,
    print_latency_stats=True,
):
    """
    遍历文件夹中的所有流量文件并处理

    Args:
        traffic_folder_path: 流量文件夹路径
        topo_type: 拓扑类型
        result_save_path: 结果保存路径
        print_interval: 打印间隔
        simulation_end_time: 仿真结束时间
        verbose: 是否详细输出
        save_individual_csv: 是否保存单个CSV文件
        save_combined_csv: 是否保存合并的CSV文件
        plot_bandwidth_curve: 是否绘制带宽曲线
        print_latency_stats: 是否打印延迟统计

    Returns:
        List[Tuple[str, List[TransactionResult], TheoreticalBandwidthCalculator]]: 所有结果列表
    """
    if not os.path.exists(traffic_folder_path):
        raise FileNotFoundError(f"流量文件夹不存在: {traffic_folder_path}")

    # 获取所有.txt文件
    traffic_files = []
    for file in os.listdir(traffic_folder_path):
        if file.endswith(".txt") and not file.startswith("#"):
            traffic_files.append(file)

    if not traffic_files:
        print(f"警告: 在文件夹 {traffic_folder_path} 中没有找到.txt流量文件")
        return []

    traffic_files.sort()  # 按文件名排序
    print(f"找到 {len(traffic_files)} 个流量文件: {traffic_files}")

    all_results_with_info = []  # List[Tuple[str, List[TransactionResult], TheoreticalBandwidthCalculator]]

    for i, traffic_file in enumerate(traffic_files):
        if verbose:
            print(f"\n{'='*60}")
            print(f"处理流量文件 {i+1}/{len(traffic_files)}: {traffic_file}")
            print(f"拓扑类型: {topo_type}")
            print(f"{'='*60}")

        # 构建完整的流量文件路径
        full_traffic_path = os.path.join(traffic_folder_path, traffic_file)

        try:
            # 创建理论带宽计算器
            calculator = TheoreticalBandwidthCalculator(traffic_file=full_traffic_path)

            # 执行仿真
            results = calculator.simulate(print_interval=print_interval, simulation_end_time=simulation_end_time)

            if not results:
                print("警告: 没有有效的仿真结果")
                continue

            # 存储结果和calculator实例
            all_results_with_info.append((traffic_file, results, calculator))

            # 生成输出文件名
            base_name = os.path.splitext(traffic_file)[0]

            # 保存单个CSV结果
            if save_individual_csv:
                output_csv = os.path.join(result_save_path, f"{base_name}_{topo_type}_theoretical.csv")
                _save_csv_results(results, output_csv, verbose)

            # 计算并输出统计信息
            if verbose:
                _print_statistics(results, calculator, verbose, print_latency_stats)

            # 绘制带宽曲线（如果启用）
            if plot_bandwidth_curve:
                _plot_bandwidth_curve(results, base_name, topo_type, result_save_path, calculator)

        except Exception as e:
            print(f"错误: 处理文件 {traffic_file} 时发生异常: {e}")
            import traceback

            traceback.print_exc()
            continue

    # 保存统计汇总结果
    if save_combined_csv and all_results_with_info:
        statistics_csv_path = os.path.join(result_save_path, f"statistics_summary_{topo_type}.csv")
        _save_statistics_summary(all_results_with_info, statistics_csv_path, topo_type, verbose)

        # 打印汇总统计信息
        if verbose:
            total_transactions = sum(len(results) for _, results, _ in all_results_with_info)
            print(f"\n{'='*60}")
            print(f"统计汇总信息:")
            print(f"处理文件数: {len(all_results_with_info)}")
            print(f"总事务数: {total_transactions}")
            print(f"统计汇总结果保存到: {statistics_csv_path}")
            print(f"{'='*60}")

    return all_results_with_info


def _print_statistics(results, calculator=None, verbose=True, print_latency_stats=True):
    """打印统计信息"""
    if not verbose:
        return

    print("\n=== 带宽统计 ===")

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
                # 计算加权带宽：Σ(区间数据量 × 区间带宽) / Σ(区间数据量)
                total_weighted_bandwidth = 0.0
                total_data_bytes = 0.0

                for interval in working_intervals:
                    interval_bandwidth = interval.bandwidth_bytes_per_ns  # 区间带宽 (bytes/ns)
                    interval_data_bytes = interval.total_bytes  # 区间数据量 (bytes)

                    total_weighted_bandwidth += interval_data_bytes * interval_bandwidth
                    total_data_bytes += interval_data_bytes

                weighted_bandwidth = (total_weighted_bandwidth / total_data_bytes) if total_data_bytes > 0 else 0.0
            else:
                weighted_bandwidth = 0.0

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

        # 定义IP类型到颜色的映射
        ip_color_map = {
            "GDMA": "#1f77b4",  # 蓝色
            "SDMA": "#ff7f0e",  # 橙色
            "CDMA": "#2ca02c",  # 绿色
            "DDR": "#d62728",  # 红色
            "L2M": "#9467bd",  # 紫色
            "DMA": "#8c564b",  # 棕色
            "HBM": "#e377c2",  # 粉色
            "SRAM": "#7f7f7f",  # 灰色
            "L3": "#bcbd22",  # 橄榄色
            "PCIe": "#17becf",  # 青色
        }

        # 按IP类型分组（与统计函数保持一致，不区分读写）
        type_groups = defaultdict(list)
        for result in results:
            ip_type = result.source_ip_type.split("_")[0].upper()
            type_groups[ip_type].append(result)

        plt.figure(figsize=(12, 8))

        for group_key, group_results in type_groups.items():
            if not group_results or calculator is None:
                continue

            # 获取该IP类型对应的颜色，如果没有定义则使用默认颜色
            color = ip_color_map.get(group_key, "#000000")  # 默认黑色

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

                # 为每个工作区间绘制独立的曲线段，使用固定颜色
                # 只有第一个区间显示标签，避免重复
                label = group_key if i == 0 else None
                plt.plot(abs_times_us, bandwidth, label=label, color=color, drawstyle="steps-post")

        plt.xlabel("时间 (μs)")
        plt.ylabel("带宽 (GB/s)")
        plt.title(f"理论带宽 - {base_name} ({topo_type})")
        plt.legend()
        plt.grid(True)

        # 保存图片或显示
        if save_path:
            output_fig = os.path.join(save_path, f"{base_name}_{topo_type}_theoretical_bw.png")
            plt.savefig(output_fig, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"带宽曲线图已保存: {output_fig}")
        else:
            plt.show()
            print("带宽曲线图已显示")

    except Exception as e:
        print(f"绘制带宽曲线时发生错误: {e}")


def main():
    """理论带宽计算主函数"""

    # ==================== 流量配置 ====================
    # traffic_file_path = r"../../test_data/"
    # traffic_file_path = r"../../../C2C/traffic_data"
    # traffic_file_path = r"../../traffic/traffic0730"
    # traffic_file_path = r"../../example/"
    # traffic_file_path = r"../../traffic/0617/"
    traffic_file_path = r"../../traffic/DeepSeek_0616/step6_ch_map/"
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
            # "LLama2_AttentionFC.txt"
            # "W_8x8.txt"
            # "MLA_B32.txt"
            "Add.txt",
            # "attn_fc.txt"
        ],
    ]

    # ==================== 输出配置 ====================
    # 生成时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_save_path = f"../../Result/TheoreticalBandwidth/{timestamp}/"
    results_fig_save_path = None
    # results_fig_save_path = f"../../Result/Plt_Theoretical_BW/{timestamp}/"

    # ==================== 拓扑配置 ====================
    # 默认拓扑类型（现在仅用于文件命名）
    topo_type = "5x4"  # SG2262

    # ==================== 仿真参数 ====================
    verbose = 1  # 详细输出
    save_csv_results = 1  # 保存CSV结果
    plot_bandwidth_curve = 1  # 绘制带宽曲线
    print_latency_stats = 1  # 打印延迟统计

    # 进度打印和仿真控制参数
    print_interval = 5000  # 每处理多少个事务打印一次进度
    simulation_end_time = 10000  # 仿真结束时间(ns)，None表示处理所有事务

    # 处理模式配置
    process_entire_folder = 0  # True: 处理整个文件夹，False: 处理指定文件列表
    save_individual_csv = 1  # 是否保存单个CSV文件
    save_combined_csv = 1  # 是否保存合并的CSV文件

    # ==================== 执行仿真 ====================
    # 确保输出路径存在
    os.makedirs(result_save_path, exist_ok=True)
    if results_fig_save_path:
        os.makedirs(results_fig_save_path, exist_ok=True)

    # 根据处理模式选择执行方式
    if process_entire_folder:
        # 处理整个文件夹模式
        if verbose:
            print(f"\n{'='*60}")
            print(f"文件夹处理模式")
            print(f"流量文件夹: {traffic_file_path}")
            print(f"拓扑类型: {topo_type}")
            print(f"{'='*60}")

        try:
            all_results = process_traffic_folder(
                traffic_folder_path=traffic_file_path,
                topo_type=topo_type,
                result_save_path=result_save_path,
                print_interval=print_interval,
                simulation_end_time=simulation_end_time,
                verbose=verbose,
                save_individual_csv=save_individual_csv,
                save_combined_csv=save_combined_csv,
                plot_bandwidth_curve=plot_bandwidth_curve,
                print_latency_stats=print_latency_stats,
            )

            if verbose and all_results:
                print(f"\n文件夹处理完成！共处理 {len(all_results)} 个流量文件")

        except Exception as e:
            print(f"错误: 处理文件夹时发生异常: {e}")
            import traceback

            traceback.print_exc()

    else:
        # 处理指定文件列表模式
        for config_group in traffic_config:
            for traffic_file in config_group:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"处理流量文件: {traffic_file}")
                    print(f"拓扑类型: {topo_type}")
                    print(f"{'='*60}")

                # 构建完整的流量文件路径
                full_traffic_path = os.path.join(traffic_file_path, traffic_file)

                if not os.path.exists(full_traffic_path):
                    print(f"警告: 流量文件不存在: {full_traffic_path}")
                    continue

                try:
                    # 创建理论带宽计算器
                    calculator = TheoreticalBandwidthCalculator(traffic_file=full_traffic_path)

                    # 执行仿真
                    results = calculator.simulate(print_interval=print_interval, simulation_end_time=simulation_end_time)

                    if not results:
                        print("警告: 没有有效的仿真结果")
                        continue

                    # 生成输出文件名
                    base_name = os.path.splitext(traffic_file)[0]
                    output_csv = os.path.join(result_save_path, f"{base_name}_{topo_type}_theoretical.csv")

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
        print("理论带宽计算完成!")
        print(f"结果保存路径: {result_save_path}")
        if results_fig_save_path:
            print(f"图表保存路径: {results_fig_save_path}")
        print(f"使用内置配置: Config类")
        print(f"{'='*60}")


if __name__ == "__main__":
    exit(main())
