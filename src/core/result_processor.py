import numpy as np
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
from src.utils.component import *
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from functools import lru_cache
from src.core.result_processor import *
import time


@dataclass
class RequestInfo:
    """请求信息数据结构"""

    packet_id: int
    start_time: int  # ns
    end_time: int  # ns (整体网络结束时间)
    rn_end_time: int  # ns (RN端口结束时间)
    sn_end_time: int  # ns (SN端口结束时间)
    req_type: str  # 'read' or 'write'
    source_node: int
    dest_node: int
    source_type: str
    dest_type: str
    burst_length: int
    total_bytes: int
    cmd_latency: int
    data_latency: int
    transaction_latency: int


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
        """区间内平均带宽 (bytes/ns)"""
        return self.total_bytes / self.duration if self.duration > 0 else 0.0


@dataclass
class BandwidthMetrics:
    """带宽指标数据结构"""

    unweighted_bandwidth: float  # GB/s
    weighted_bandwidth: float  # GB/s
    working_intervals: List[WorkingInterval]
    total_working_time: int  # ns
    network_start_time: int  # ns
    network_end_time: int  # ns
    total_bytes: int
    total_requests: int


@dataclass
class PortBandwidthMetrics:
    """端口带宽指标"""

    port_id: str
    read_metrics: BandwidthMetrics
    write_metrics: BandwidthMetrics
    mixed_metrics: BandwidthMetrics  # 新增：混合读写指标


class BandwidthAnalyzer:
    def _calculate_latency_stats(self):
        """计算并返回延迟统计数据字典，过滤掉无穷大"""
        import math

        stats = {
            "cmd": {"read": {"sum": 0, "max": 0, "count": 0}, "write": {"sum": 0, "max": 0, "count": 0}, "mixed": {"sum": 0, "max": 0, "count": 0}},
            "data": {"read": {"sum": 0, "max": 0, "count": 0}, "write": {"sum": 0, "max": 0, "count": 0}, "mixed": {"sum": 0, "max": 0, "count": 0}},
            "trans": {"read": {"sum": 0, "max": 0, "count": 0}, "write": {"sum": 0, "max": 0, "count": 0}, "mixed": {"sum": 0, "max": 0, "count": 0}},
        }
        for r in self.requests:
            # CMD
            if math.isfinite(r.cmd_latency):
                group = stats["cmd"][r.req_type]
                group["sum"] += r.cmd_latency
                group["count"] += 1
                group["max"] = max(group["max"], r.cmd_latency)
                mixed = stats["cmd"]["mixed"]
                mixed["sum"] += r.cmd_latency
                mixed["count"] += 1
                mixed["max"] = max(mixed["max"], r.cmd_latency)
            # Data
            if math.isfinite(r.data_latency):
                group = stats["data"][r.req_type]
                group["sum"] += r.data_latency
                group["count"] += 1
                group["max"] = max(group["max"], r.data_latency)
                mixed = stats["data"]["mixed"]
                mixed["sum"] += r.data_latency
                mixed["count"] += 1
                mixed["max"] = max(mixed["max"], r.data_latency)
            # Transaction
            if math.isfinite(r.transaction_latency):
                group = stats["trans"][r.req_type]
                group["sum"] += r.transaction_latency
                group["count"] += 1
                group["max"] = max(group["max"], r.transaction_latency)
                mixed = stats["trans"]["mixed"]
                mixed["sum"] += r.transaction_latency
                mixed["count"] += 1
                mixed["max"] = max(mixed["max"], r.transaction_latency)
        return stats

    """
    带宽分析器 - 统一的带宽统计分析类

    功能：
    1. 统计工作区间（去除空闲时间段）
    2. 计算非加权和加权带宽
    3. 分别统计读写操作
    4. 统计混合读写操作（不区分读写类型）
    5. 网络整体和RN端口带宽统计
    6. 生成统一报告
    """

    def __init__(
        self,
        config,
        min_gap_threshold: int = 50,
        plot_rn_bw_fig: bool = False,
        plot_flow_graph: bool = False,
    ):
        """
        初始化带宽分析器

        Args:
            config: 网络配置对象
            min_gap_threshold: 工作区间合并阈值(ns)，小于此值的间隔被视为同一工作区间
        """
        self.config = config
        self.min_gap_threshold = min_gap_threshold
        self.network_frequency = config.NETWORK_FREQUENCY  # GHz
        self.plot_rn_bw_fig = plot_rn_bw_fig
        self.plot_flow_graph = plot_flow_graph
        self.finish_cycle = -np.inf

        # 数据存储
        self.requests: List[RequestInfo] = []
        self.rn_positions = set()
        self.sn_positions = set()
        self.rn_bandwidth_time_series = defaultdict(lambda: {"time": [], "bytes": []})
        self.ip_bandwidth_data = None
        self.read_ip_intervals = defaultdict(list)
        self.write_ip_intervals = defaultdict(list)

        # 初始化节点位置
        self._initialize_node_positions()

    def _initialize_node_positions(self):
        """初始化RN和SN节点位置"""
        if hasattr(self.config, "GDMA_SEND_POSITION_LIST"):
            self.rn_positions.update(self.config.GDMA_SEND_POSITION_LIST)
        if hasattr(self.config, "SDMA_SEND_POSITION_LIST"):
            self.rn_positions.update(self.config.SDMA_SEND_POSITION_LIST)
        if hasattr(self.config, "DDR_SEND_POSITION_LIST"):
            self.sn_positions.update(pos - self.config.NUM_COL for pos in self.config.DDR_SEND_POSITION_LIST)
        if hasattr(self.config, "L2M_SEND_POSITION_LIST"):
            self.sn_positions.update(pos - self.config.NUM_COL for pos in self.config.L2M_SEND_POSITION_LIST)

    def collect_requests_data(self, base_model) -> None:
        """从base_model收集请求数据"""
        self.requests.clear()
        self.base_model = base_model

        for packet_id, flits in base_model.data_network.arrive_flits.items():
            if not flits or len(flits) != flits[0].burst_length:
                continue

            representative_flit: Flit = flits[-1]

            # 计算不同角度的结束时间
            network_end_time = representative_flit.data_received_complete_cycle // self.network_frequency

            if representative_flit.req_type == "read":
                # 读请求：RN在收到数据时结束，SN在发出数据时结束
                rn_end_time = representative_flit.data_received_complete_cycle // self.network_frequency  # RN收到数据
                sn_end_time = representative_flit.data_entry_noc_from_cake1_cycle // self.network_frequency  # SN发出数据
            else:  # write
                # 写请求：RN在发出数据时结束，SN在收到数据时结束
                rn_end_time = representative_flit.data_entry_noc_from_cake0_cycle // self.network_frequency  # RN发出数据
                sn_end_time = representative_flit.data_received_complete_cycle // self.network_frequency  # SN收到数据

            request_info = RequestInfo(
                packet_id=packet_id,
                start_time=representative_flit.cmd_entry_cake0_cycle // self.network_frequency,
                end_time=network_end_time,  # 整体网络结束时间
                rn_end_time=rn_end_time,
                sn_end_time=sn_end_time,
                req_type=representative_flit.req_type,
                source_node=representative_flit.source,
                dest_node=representative_flit.destination,
                source_type=representative_flit.original_source_type,
                dest_type=representative_flit.original_destination_type,
                burst_length=representative_flit.burst_length,
                total_bytes=representative_flit.burst_length * 128,
                cmd_latency=representative_flit.cmd_latency // self.network_frequency,
                data_latency=representative_flit.data_latency // self.network_frequency,
                transaction_latency=representative_flit.transaction_latency // self.network_frequency,
            )

            # 收集RN带宽时间序列数据
            port_key = f"{representative_flit.original_source_type[:-3].upper()} {representative_flit.req_type} {representative_flit.original_destination_type[:3].upper()}"

            if representative_flit.req_type == "read":
                completion_time = rn_end_time
            else:  # write
                completion_time = rn_end_time

            self.rn_bandwidth_time_series[port_key]["time"].append(completion_time)
            self.rn_bandwidth_time_series[port_key]["bytes"].append(representative_flit.burst_length * 128)

            # 更新finish_cycle
            if representative_flit.data_received_complete_cycle != float("inf"):
                self.finish_cycle = max(self.finish_cycle, representative_flit.data_received_complete_cycle)

            self.requests.append(request_info)

        # 按开始时间排序
        self.requests.sort(key=lambda x: x.start_time)

    def calculate_ip_bandwidth_data(self):
        """计算IP带宽数据矩阵"""
        rows = self.config.NUM_ROW
        cols = self.config.NUM_COL
        if getattr(self.base_model, "topo_type_stat", None) != "4x5":
            rows -= 1

        # 初始化数据结构
        self.ip_bandwidth_data = {
            "read": {
                "sdma": np.zeros((rows, cols)),
                "gdma": np.zeros((rows, cols)),
                "ddr": np.zeros((rows, cols)),
                "l2m": np.zeros((rows, cols)),
            },
            "write": {
                "sdma": np.zeros((rows, cols)),
                "gdma": np.zeros((rows, cols)),
                "ddr": np.zeros((rows, cols)),
                "l2m": np.zeros((rows, cols)),
            },
            "total": {
                "sdma": np.zeros((rows, cols)),
                "gdma": np.zeros((rows, cols)),
                "ddr": np.zeros((rows, cols)),
                "l2m": np.zeros((rows, cols)),
            },
        }

        # 处理RN端口（按照source_node分组，避免重复计算）
        rn_requests_by_source = defaultdict(list)

        for req in self.requests:
            if req.source_node in self.rn_positions:
                rn_requests_by_source[req.source_node].append(req)

        # 计算每个RN源节点的带宽
        for source_node, node_requests in rn_requests_by_source.items():
            # 按source_type分组
            by_type = defaultdict(list)
            for req in node_requests:
                # 提取source_type的前缀（去掉_ip后缀）
                source_type = req.source_type.lower()[:-2]
                by_type[source_type].append(req)

            # 计算物理位置
            physical_col = source_node % cols
            physical_row = source_node // cols // 2

            # 为每种类型计算带宽
            for source_type, type_requests in by_type.items():
                # 分别计算读写带宽
                read_requests = [req for req in type_requests if req.req_type == "read"]
                write_requests = [req for req in type_requests if req.req_type == "write"]

                # 计算读带宽
                if read_requests:
                    read_metrics = self.calculate_bandwidth_metrics(read_requests, "read")
                    self.ip_bandwidth_data["read"][source_type][physical_row, physical_col] = read_metrics.weighted_bandwidth

                # 计算写带宽
                if write_requests:
                    write_metrics = self.calculate_bandwidth_metrics(write_requests, "write")
                    self.ip_bandwidth_data["write"][source_type][physical_row, physical_col] = write_metrics.weighted_bandwidth

                # 计算总带宽
                if type_requests:
                    total_metrics = self.calculate_bandwidth_metrics(type_requests, operation_type=None)
                    self.ip_bandwidth_data["total"][source_type][physical_row, physical_col] = total_metrics.weighted_bandwidth

        # 处理SN端口
        sn_requests_by_dest = defaultdict(list)

        for req in self.requests:
            if req.dest_node in self.sn_positions:
                sn_requests_by_dest[req.dest_node].append(req)

        # 计算每个SN目标节点的带宽
        for dest_node, node_requests in sn_requests_by_dest.items():
            # 按dest_type分组
            by_type = defaultdict(list)
            for req in node_requests:
                # 提取dest_type的前缀（去掉_ip后缀）
                dest_type = req.dest_type.lower()[:-2]
                by_type[dest_type].append(req)

            # 计算物理位置
            physical_col = dest_node % cols
            physical_row = dest_node // cols // 2

            # 为每种类型计算带宽
            for dest_type, type_requests in by_type.items():
                # 分别计算读写带宽
                read_requests = [req for req in type_requests if req.req_type == "read"]
                write_requests = [req for req in type_requests if req.req_type == "write"]

                # 计算读带宽
                if read_requests:
                    read_metrics = self.calculate_bandwidth_metrics(read_requests, "read")
                    self.ip_bandwidth_data["read"][dest_type][physical_row, physical_col] = read_metrics.weighted_bandwidth

                # 计算写带宽
                if write_requests:
                    write_metrics = self.calculate_bandwidth_metrics(write_requests, "write")
                    self.ip_bandwidth_data["write"][dest_type][physical_row, physical_col] = write_metrics.weighted_bandwidth

                # 计算总带宽
                if type_requests:
                    total_metrics = self.calculate_bandwidth_metrics(type_requests, operation_type=None)
                    self.ip_bandwidth_data["total"][dest_type][physical_row, physical_col] = total_metrics.weighted_bandwidth

    def plot_rn_bandwidth_curves(self) -> float:
        """
        绘制RN端带宽曲线图，使用累积和计算带宽

        Returns:
            总带宽 (GB/s)
        """
        if self.plot_rn_bw_fig:
            fig = plt.figure(figsize=(12, 8))

        total_bw = 0

        for port_key, data_dict in self.rn_bandwidth_time_series.items():
            if not data_dict["time"]:
                continue

            # 排序时间戳并去除nan值
            raw_times = np.array(data_dict["time"])
            clean_times = raw_times[~np.isnan(raw_times)]
            times = np.sort(clean_times)

            if len(times) == 0:
                continue

            # 计算累积带宽
            cum_counts = np.arange(1, len(times) + 1)
            bandwidth = (cum_counts * 128 * self.config.BURST) / times  # bytes/ns转换为GB/s

            # 只显示前100%的时间段
            t = np.percentile(times, 100)
            mask = times <= t

            if self.plot_rn_bw_fig:
                (line,) = plt.plot(times[mask] / 1000, bandwidth[mask], drawstyle="default", label=f"{port_key}")
                plt.text(times[mask][-1] / 1000, bandwidth[mask][-1], f"{bandwidth[mask][-1]:.2f}", va="center", color=line.get_color(), fontsize=12)

            # 打印最终带宽值
            if hasattr(self, "base_model") and self.base_model and hasattr(self.base_model, "verbose") and self.base_model.verbose:
                print(f"{port_key} Final Bandwidth: {bandwidth[mask][-1]:.2f} GB/s")

            total_bw += bandwidth[mask][-1]

        if hasattr(self, "base_model") and self.base_model and hasattr(self.base_model, "verbose") and self.base_model.verbose:
            print(f"Total Bandwidth: {total_bw:.2f} GB/s")
            print("=" * 60)

        if self.plot_rn_bw_fig:
            plt.xlabel("Time (us)")
            plt.ylabel("Bandwidth (GB/s)")
            plt.title("RN Bandwidth")
            plt.legend()
            plt.grid(True)
            # 自动保存RN带宽曲线到结果文件夹
            if self.plot_rn_bw_fig and hasattr(self, "base_model") and getattr(self.base_model, "results_fig_save_path", None):
                rn_save_path = os.path.join(self.base_model.results_fig_save_path, f"rn_bandwidth_{self.config.TOPO_TYPE}_{self.base_model.file_name}.png")
                fig.savefig(rn_save_path, bbox_inches="tight")
            else:
                plt.show()

        return total_bw

    def calculate_working_intervals(self, requests: List[RequestInfo]) -> List[WorkingInterval]:
        """
        计算工作区间，去除空闲时间段

        Args:
            requests: 请求列表

        Returns:
            工作区间列表
        """
        if not requests:
            return []

        # 构建时间轴事件
        events = []
        for req in requests:
            events.append((req.start_time, "start", req))
            events.append((req.end_time, "end", req))
        events.sort(key=lambda x: (x[0], x[1]))  # 按时间排序，相同时间时'end'在'start'前面

        # 识别连续工作时段
        active_requests = set()
        raw_intervals = []
        current_start = None

        for time_point, event_type, req in events:
            # 检查时间点是否有效（非nan且非None）
            if time_point is not None and not (isinstance(time_point, float) and np.isnan(time_point)):

                if event_type == "start":
                    if not active_requests:  # 开始新的工作区间
                        current_start = time_point
                    active_requests.add(req.packet_id)
                else:  # 'end'
                    active_requests.discard(req.packet_id)
                    if not active_requests and current_start is not None:
                        # 工作区间结束
                        raw_intervals.append((current_start, time_point))
                        current_start = None

        # 处理最后未结束的区间
        if active_requests and current_start is not None:
            last_end = max(req.end_time for req in requests)
            raw_intervals.append((current_start, last_end))

        # 合并相近区间（间隔小于阈值）
        merged_intervals = self._merge_close_intervals(raw_intervals)

        # 构建WorkingInterval对象
        working_intervals = []
        for start, end in merged_intervals:
            # 找到该区间内的所有请求
            interval_requests = [req for req in requests if req.start_time < end and req.end_time > start]

            if not interval_requests:
                continue

            # 计算区间统计
            total_bytes = sum(req.total_bytes for req in interval_requests)
            flit_count = sum(req.burst_length for req in interval_requests)

            interval = WorkingInterval(start_time=start, end_time=end, duration=end - start, flit_count=flit_count, total_bytes=total_bytes, request_count=len(interval_requests))
            working_intervals.append(interval)

        return working_intervals

    def _merge_close_intervals(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """合并相近的时间区间"""
        if not intervals:
            return []

        # 按开始时间排序
        sorted_intervals = sorted(intervals)
        merged = [sorted_intervals[0]]

        for current_start, current_end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]

            # 如果间隙小于阈值，则合并
            if current_start - last_end <= self.min_gap_threshold:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        return merged

    def calculate_bandwidth_metrics(self, requests: List[RequestInfo], operation_type: str = None, endpoint_type: str = "network") -> BandwidthMetrics:
        """
        计算指定操作类型的带宽指标

        Args:
            requests: 所有请求列表
            operation_type: 'read'、'write' 或 None（表示混合读写）
            endpoint_type: 'network'(整体网络)、'rn'(RN端口)、'sn'(SN端口)
        """
        # 筛选请求并创建临时请求列表（使用正确的结束时间）
        filtered_requests = []

        for req in requests:
            if operation_type is not None and req.req_type != operation_type:
                continue

            # 根据endpoint_type选择正确的结束时间
            if endpoint_type == "rn":
                end_time = req.rn_end_time
            elif endpoint_type == "sn":
                end_time = req.sn_end_time
            else:  # network
                end_time = req.end_time

            # 创建临时请求对象，使用正确的结束时间
            temp_req = RequestInfo(
                packet_id=req.packet_id,
                start_time=req.start_time,
                end_time=end_time,
                rn_end_time=req.rn_end_time,
                sn_end_time=req.sn_end_time,
                req_type=req.req_type,
                source_node=req.source_node,
                dest_node=req.dest_node,
                source_type=req.source_type,
                dest_type=req.dest_type,
                burst_length=req.burst_length,
                total_bytes=req.total_bytes,
                cmd_latency=req.cmd_latency,
                data_latency=req.data_latency,
                transaction_latency=req.transaction_latency,
            )
            filtered_requests.append(temp_req)

        if not filtered_requests:
            return BandwidthMetrics(
                unweighted_bandwidth=0.0,
                weighted_bandwidth=0.0,
                working_intervals=[],
                total_working_time=0,
                network_start_time=0,
                network_end_time=0,
                total_bytes=0,
                total_requests=0,
            )

        # 计算工作区间
        working_intervals = self.calculate_working_intervals(filtered_requests)

        # 网络工作时间窗口
        network_start = min(req.start_time for req in filtered_requests)
        network_end = max(req.end_time for req in filtered_requests)
        total_network_time = network_end - network_start

        # 总工作时间和总字节数
        total_working_time = sum(interval.duration for interval in working_intervals)
        total_bytes = sum(req.total_bytes for req in filtered_requests)

        # 计算非加权带宽：总数据量 / 网络总时间 / IP数量
        unweighted_bandwidth = (total_bytes / total_network_time) if total_network_time > 0 else 0.0

        # 计算加权带宽：各区间带宽按flit数量加权平均
        if working_intervals:
            total_weighted_bandwidth = 0.0
            total_weight = 0

            for interval in working_intervals:
                weight = interval.flit_count  # 权重是工作时间段的flit数量
                bandwidth = interval.bandwidth_bytes_per_ns  # bytes
                total_weighted_bandwidth += bandwidth * weight
                total_weight += weight

            weighted_bandwidth = (total_weighted_bandwidth / total_weight) if total_weight > 0 else 0.0
        else:
            weighted_bandwidth = 0.0

        return BandwidthMetrics(
            unweighted_bandwidth=unweighted_bandwidth,
            weighted_bandwidth=weighted_bandwidth,
            working_intervals=working_intervals,
            total_working_time=total_working_time,
            network_start_time=network_start,
            network_end_time=network_end,
            total_bytes=total_bytes,
            total_requests=len(filtered_requests),
        )

    def calculate_network_overall_bandwidth(self) -> Dict[str, BandwidthMetrics]:
        """
        计算网络整体带宽（读、写和混合）

        Returns:
            {'read': BandwidthMetrics, 'write': BandwidthMetrics, 'mixed': BandwidthMetrics}
        """
        results = {}

        # 分别计算读写带宽
        for operation in ["read", "write"]:
            results[operation] = self.calculate_bandwidth_metrics(self.requests, operation)

        # 计算混合读写带宽（不区分请求类型）
        results["mixed"] = self.calculate_bandwidth_metrics(self.requests, operation_type=None)

        return results

    def calculate_rn_port_bandwidth(self) -> Dict[str, PortBandwidthMetrics]:
        """计算每个RN端口的带宽"""
        port_metrics = {}
        rn_requests_by_port = defaultdict(list)

        for req in self.requests:
            if req.source_node in self.rn_positions:
                port_id = f"{req.source_type}_{req.source_node}"
                rn_requests_by_port[port_id].append(req)

        for port_id, port_requests in rn_requests_by_port.items():
            # 使用RN端点类型计算带宽
            read_metrics = self.calculate_bandwidth_metrics(port_requests, "read", "rn")
            write_metrics = self.calculate_bandwidth_metrics(port_requests, "write", "rn")
            mixed_metrics = self.calculate_bandwidth_metrics(port_requests, None, "rn")

            port_metrics[port_id] = PortBandwidthMetrics(port_id=port_id, read_metrics=read_metrics, write_metrics=write_metrics, mixed_metrics=mixed_metrics)

        return port_metrics

    def calculate_sn_port_bandwidth(self) -> Dict[str, PortBandwidthMetrics]:
        """计算每个SN端口的带宽"""
        port_metrics = {}
        sn_requests_by_port = defaultdict(list)

        for req in self.requests:
            if req.dest_node in self.sn_positions:
                port_id = f"{req.dest_type}_{req.dest_node}"
                sn_requests_by_port[port_id].append(req)

        for port_id, port_requests in sn_requests_by_port.items():
            # 使用SN端点类型计算带宽
            read_metrics = self.calculate_bandwidth_metrics(port_requests, "read", "sn")
            write_metrics = self.calculate_bandwidth_metrics(port_requests, "write", "sn")
            mixed_metrics = self.calculate_bandwidth_metrics(port_requests, None, "sn")

            port_metrics[port_id] = PortBandwidthMetrics(port_id=port_id, read_metrics=read_metrics, write_metrics=write_metrics, mixed_metrics=mixed_metrics)

        return port_metrics

    def precalculate_ip_bandwidth_data(self):
        """预计算IP带宽数据，供绘图使用"""
        if self.ip_bandwidth_data is not None:
            return  # 已经计算过了

        # 使用现有的calculate_ip_bandwidth_data逻辑
        # 但基于已有的端口带宽计算结果
        self.calculate_ip_bandwidth_data()

    def analyze_all_bandwidth(self) -> Dict:
        """
        执行完整的带宽分析

        Returns:
            包含所有分析结果的字典
        """
        if not self.requests:
            raise ValueError("没有请求数据，请先调用 collect_requests_data()")

        # 网络整体带宽分析
        network_overall = self.calculate_network_overall_bandwidth()

        # RN端口带宽分析
        rn_port_metrics = self.calculate_rn_port_bandwidth()

        # 汇总统计
        total_read_requests = len([r for r in self.requests if r.req_type == "read"])
        total_write_requests = len([r for r in self.requests if r.req_type == "write"])
        total_read_flits = sum(req.burst_length for req in self.requests if req.req_type == "read")
        total_write_flits = sum(req.burst_length for req in self.requests if req.req_type == "write")

        # 获取Circuit统计数据（如果存在）
        circuit_stats = {}
        if hasattr(self, "base_model") and self.base_model:
            circuit_stats = {
                "req_circuits_h": getattr(self.base_model, "req_cir_h_num_stat", 0),
                "req_circuits_v": getattr(self.base_model, "req_cir_v_num_stat", 0),
                "rsp_circuits_h": getattr(self.base_model, "rsp_cir_h_num_stat", 0),
                "rsp_circuits_v": getattr(self.base_model, "rsp_cir_v_num_stat", 0),
                "data_circuits_h": getattr(self.base_model, "data_cir_h_num_stat", 0),
                "data_circuits_v": getattr(self.base_model, "data_cir_v_num_stat", 0),
                "req_wait_cycles_h": getattr(self.base_model, "req_wait_cycle_h_num_stat", 0),
                "req_wait_cycles_v": getattr(self.base_model, "req_wait_cycle_v_num_stat", 0),
                "rsp_wait_cycles_h": getattr(self.base_model, "rsp_wait_cycle_h_num_stat", 0),
                "rsp_wait_cycles_v": getattr(self.base_model, "rsp_wait_cycle_v_num_stat", 0),
                "data_wait_cycles_h": getattr(self.base_model, "data_wait_cycle_h_num_stat", 0),
                "data_wait_cycles_v": getattr(self.base_model, "data_wait_cycle_v_num_stat", 0),
                "read_retry_num": getattr(self.base_model, "read_retry_num_stat", 0),
                "write_retry_num": getattr(self.base_model, "write_retry_num_stat", 0),
                "EQ_ETag_T1_num": getattr(self.base_model, "EQ_ETag_T1_num_stat", 0),
                "EQ_ETag_T0_num": getattr(self.base_model, "EQ_ETag_T0_num_stat", 0),
                "RB_ETag_T1_num": getattr(self.base_model, "RB_ETag_T1_num_stat", 0),
                "RB_ETag_T0_num": getattr(self.base_model, "RB_ETag_T0_num_stat", 0),
                "ITag_h_num": getattr(self.base_model, "ITag_h_num_stat", 0),
                "ITag_v_num": getattr(self.base_model, "ITag_v_num_stat", 0),
            }

        results = {
            "network_overall": network_overall,
            "rn_ports": rn_port_metrics,
            "summary": {
                "total_requests": len(self.requests),
                "read_requests": total_read_requests,
                "write_requests": total_write_requests,
                "total_read_flits": total_read_flits,
                "total_write_flits": total_write_flits,
                "analysis_config": {"min_gap_threshold_ns": self.min_gap_threshold, "network_frequency_ghz": self.network_frequency},
                "circuit_stats": circuit_stats,
            },
        }

        # 控制台输出重要数据
        if hasattr(self, "base_model") and self.base_model and hasattr(self.base_model, "verbose") and self.base_model.verbose:
            self._print_summary_to_console(results)

        # 绘制RN带宽曲线
        total_bandwidth = self.plot_rn_bandwidth_curves()
        results["summary"]["Total_sum_BW"] = total_bandwidth
        results["Total_sum_BW"] = total_bandwidth

        if self.plot_flow_graph and hasattr(self, "base_model") and self.base_model and getattr(self.base_model, "results_fig_save_path", None):
            # 自动保存流量图到结果文件夹
            flow_fname = f"flow_graph_{self.config.TOPO_TYPE}_{self.base_model.file_name}.png"
            flow_save_path = os.path.join(self.base_model.results_fig_save_path, flow_fname)
            self.draw_flow_graph(self.base_model.data_network, mode="total", save_path=flow_save_path)

        return results

    def draw_flow_graph(self, network: Network, mode="total", node_size=2000, save_path=None):
        """
        绘制合并的网络流图和IP

        :param network: 网络对象
        :param mode: 显示模式，可以是'read', 'write'或'total'
        :param node_size: 节点大小
        :param save_path: 图片保存路径
        """
        # 确保IP带宽数据已计算
        self.precalculate_ip_bandwidth_data()

        # 准备网络流数据
        G = nx.DiGraph()

        # 处理不同模式的网络流数据
        if mode == "read":
            links = network.links_flow_stat.get("read", {})
        elif mode == "write":
            links = network.links_flow_stat.get("write", {})
        else:  # total模式，需要合并读和写的数据
            read_links = network.links_flow_stat.get("read", {})
            write_links = network.links_flow_stat.get("write", {})

            # 合并读和写的流量
            all_keys = set(read_links.keys()) | set(write_links.keys())
            links = {}
            for key in all_keys:
                read_val = read_links.get(key, 0)
                write_val = write_links.get(key, 0)
                links[key] = read_val + write_val

        link_values = []
        for (i, j), value in links.items():
            link_value = value * 128 / (self.finish_cycle // self.config.NETWORK_FREQUENCY) if value else 0
            link_values.append(link_value)
            formatted_label = f"{link_value:.1f}"
            G.add_edge(i, j, label=formatted_label)

        # 链路颜色映射范围：动态最大值的60%阈值起红
        link_mapping_max = max(link_values) if link_values else 0.0
        link_mapping_min = max(0.6 * link_mapping_max, 100)
        # IP颜色映射范围：动态最大值的60%阈值起红
        # 从 ip_bandwidth_data 提取所有带宽值，计算当前最大
        all_ip_vals = []
        for svc in ["sdma", "gdma", "ddr", "l2m"]:
            all_ip_vals.extend(self.ip_bandwidth_data[mode][svc].flatten())
        ip_mapping_max = max(all_ip_vals) if all_ip_vals else 0.0
        ip_mapping_min = max(0.6 * ip_mapping_max, 80)

        # 计算节点位置
        pos = {}
        for node in G.nodes():
            x = node % self.config.NUM_COL
            y = node // self.config.NUM_COL
            if y % 2 == 1:  # 奇数行左移
                x -= 0.25
                y -= 0.6
            pos[node] = (x * 3, -y * 1.5)

        # 动态计算字体大小，添加最大值上限以避免过大
        node_count = len(G.nodes())
        base_font = 9
        dynamic_font = max(4, base_font * (65 / node_count) ** 0.5)
        max_font = 14  # 最大字号上限，可以根据需要调整
        dynamic_font = min(dynamic_font, max_font)

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect("equal")

        # 调整方形节点大小
        square_size = np.sqrt(node_size) / 100

        # 绘制网络流图
        nx.draw_networkx_nodes(G, pos, node_size=square_size, node_shape="s", ax=ax)

        # 绘制方形节点并添加IP信息
        for node, (x, y) in pos.items():
            # 绘制主节点方框
            rect = Rectangle(
                (x - square_size / 2, y - square_size / 2),
                width=square_size,
                height=square_size,
                color="lightblue",
                ec="black",
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(x, y, str(node), ha="center", va="center", fontsize=dynamic_font)

            # 在节点左侧添加IP信息
            physical_row = node // self.config.NUM_COL
            physical_col = node % self.config.NUM_COL

            if physical_row % 2 == 0:
                # 田字格位置和大小
                ip_width = square_size * 2.6
                ip_height = square_size * 2.6
                ip_x = x - square_size - ip_width / 2.8
                ip_y = y + 0.26

                # 绘制田字格外框
                ip_rect = Rectangle(
                    (ip_x - ip_width / 2, ip_y - ip_height / 2),
                    width=ip_width,
                    height=ip_height,
                    color="white",
                    ec="black",
                    linewidth=1,
                    zorder=2,
                )
                ax.add_patch(ip_rect)

                # 绘制田字格内部线条
                ax.plot(
                    [ip_x - ip_width / 2, ip_x + ip_width / 2],
                    [ip_y, ip_y],
                    color="black",
                    linewidth=1,
                    zorder=3,
                )
                ax.plot(
                    [ip_x, ip_x],
                    [ip_y - ip_height / 2, ip_y + ip_height / 2],
                    color="black",
                    linewidth=1,
                    zorder=3,
                )

                # 为左列和右列填充不同颜色（DMA和DDR区分）
                left_color = "honeydew"  # 左列颜色（DMA）
                right_color = "aliceblue"  # 右列颜色（GDMA）
                # 左列矩形（DMA区域）
                left_rect = Rectangle(
                    (ip_x - ip_width / 2, ip_y - ip_height / 2),
                    width=ip_width / 2,
                    height=ip_height,
                    color=left_color,
                    ec="none",
                    zorder=2,
                )
                ax.add_patch(left_rect)

                # 右列矩形（DDR区域）
                right_rect = Rectangle(
                    (ip_x, ip_y - ip_height / 2),
                    width=ip_width / 2,
                    height=ip_height,
                    color=right_color,
                    ec="none",
                    zorder=2,
                )
                ax.add_patch(right_rect)

                # 添加IP信息
                if mode == "read":
                    sdma_value = self.ip_bandwidth_data["read"]["sdma"][physical_row // 2, physical_col]
                    gdma_value = self.ip_bandwidth_data["read"]["gdma"][physical_row // 2, physical_col]
                    ddr_value = self.ip_bandwidth_data["read"]["ddr"][physical_row // 2, physical_col]
                    l2m_value = self.ip_bandwidth_data["read"]["l2m"][physical_row // 2, physical_col]

                    # 收集当前模式下每个IP的所有值
                    all_sdma = self.ip_bandwidth_data["read"]["sdma"].flatten()
                    all_gdma = self.ip_bandwidth_data["read"]["gdma"].flatten()
                    all_ddr = self.ip_bandwidth_data["read"]["ddr"].flatten()
                    all_l2m = self.ip_bandwidth_data["read"]["l2m"].flatten()

                elif mode == "write":
                    sdma_value = self.ip_bandwidth_data["write"]["sdma"][physical_row // 2, physical_col]
                    gdma_value = self.ip_bandwidth_data["write"]["gdma"][physical_row // 2, physical_col]
                    ddr_value = self.ip_bandwidth_data["write"]["ddr"][physical_row // 2, physical_col]
                    l2m_value = self.ip_bandwidth_data["write"]["l2m"][physical_row // 2, physical_col]

                    all_sdma = self.ip_bandwidth_data["write"]["sdma"].flatten()
                    all_gdma = self.ip_bandwidth_data["write"]["gdma"].flatten()
                    all_ddr = self.ip_bandwidth_data["write"]["ddr"].flatten()
                    all_l2m = self.ip_bandwidth_data["write"]["l2m"].flatten()

                else:  # total
                    sdma_value = self.ip_bandwidth_data["total"]["sdma"][physical_row // 2, physical_col]
                    gdma_value = self.ip_bandwidth_data["total"]["gdma"][physical_row // 2, physical_col]
                    ddr_value = self.ip_bandwidth_data["total"]["ddr"][physical_row // 2, physical_col]
                    l2m_value = self.ip_bandwidth_data["total"]["l2m"][physical_row // 2, physical_col]

                    all_sdma = self.ip_bandwidth_data["total"]["sdma"].flatten()
                    all_gdma = self.ip_bandwidth_data["total"]["gdma"].flatten()
                    all_ddr = self.ip_bandwidth_data["total"]["ddr"].flatten()
                    all_l2m = self.ip_bandwidth_data["total"]["l2m"].flatten()

                # 计算每个IP的阈值（例如取前20%的分位数）
                sdma_threshold = np.percentile(all_sdma, 90)
                gdma_threshold = np.percentile(all_gdma, 90)
                ddr_threshold = np.percentile(all_ddr, 90)
                l2m_threshold = np.percentile(all_l2m, 90)

                # SDMA在左上半部分
                if sdma_value <= ip_mapping_min:
                    intensity = 0.0
                else:
                    intensity = (sdma_value - ip_mapping_min) / (ip_mapping_max - ip_mapping_min)
                intensity = min(max(intensity, 0.0), 1.0)
                sdma_color = (intensity, 0, 0)
                ax.text(
                    ip_x - ip_width / 4,
                    ip_y + ip_height / 4,
                    f"{sdma_value:.1f}",
                    fontweight="bold",
                    ha="center",
                    va="center",
                    fontsize=dynamic_font * 0.6,
                    color=sdma_color,
                )

                # GDMA在左下半部分
                if gdma_value <= ip_mapping_min:
                    intensity = 0.0
                else:
                    intensity = (gdma_value - ip_mapping_min) / (ip_mapping_max - ip_mapping_min)
                intensity = min(max(intensity, 0.0), 1.0)
                gdma_color = (intensity, 0, 0)
                ax.text(
                    ip_x - ip_width / 4,
                    ip_y - ip_height / 4,
                    f"{gdma_value:.1f}",
                    fontweight="bold",
                    ha="center",
                    va="center",
                    fontsize=dynamic_font * 0.6,
                    color=gdma_color,
                )

                # l2m在右上半部分
                if l2m_value <= ip_mapping_min:
                    intensity = 0.0
                else:
                    intensity = (l2m_value - ip_mapping_min) / (ip_mapping_max - ip_mapping_min)
                intensity = min(max(intensity, 0.0), 1.0)
                l2m_color = (intensity, 0, 0)
                ax.text(
                    ip_x + ip_width / 4,
                    ip_y + ip_height / 4,
                    f"{l2m_value:.1f}",
                    fontweight="bold",
                    ha="center",
                    va="center",
                    fontsize=dynamic_font * 0.6,
                    color=l2m_color,
                )

                # ddr在右下半部分
                if ddr_value <= ip_mapping_min:
                    intensity = 0.0
                else:
                    intensity = (ddr_value - ip_mapping_min) / (ip_mapping_max - ip_mapping_min)
                intensity = min(max(intensity, 0.0), 1.0)
                ddr_color = (intensity, 0, 0)
                ax.text(
                    ip_x + ip_width / 4,
                    ip_y - ip_height / 4,
                    f"{ddr_value:.1f}",
                    fontweight="bold",
                    ha="center",
                    va="center",
                    fontsize=dynamic_font * 0.6,
                    color=ddr_color,
                )

        # 绘制边和边标签
        edge_value_threshold = np.percentile(link_values, 90)
        # 链路带宽热力图颜色映射
        # norm = mcolors.Normalize(vmin=min(link_values), vmax=max(link_values))

        for i, j, data in G.edges(data=True):
            x1, y1 = pos[i]
            x2, y2 = pos[j]
            # 根据流量值映射颜色
            value = float(data["label"])
            # 链路60%阈值起红
            if value <= link_mapping_min:
                intensity = 0.0
            else:
                intensity = (value - link_mapping_min) / (link_mapping_max - link_mapping_min)
            # clamp to [0,1]
            intensity = min(max(intensity, 0.0), 1.0)
            color = (intensity, 0, 0)

            if i != j:  # 普通边
                dx, dy = x2 - x1, y2 - y1
                dist = np.hypot(dx, dy)
                if dist > 0:
                    dx, dy = dx / dist, dy / dist
                    perp_dx, perp_dy = -dy * 0.1, dx * 0.1

                    has_reverse = G.has_edge(j, i)
                    if has_reverse:
                        start_x = x1 + dx * square_size / 2 + perp_dx
                        start_y = y1 + dy * square_size / 2 + perp_dy
                        end_x = x2 - dx * square_size / 2 + perp_dx
                        end_y = y2 - dy * square_size / 2 + perp_dy
                    else:
                        start_x = x1 + dx * square_size / 2
                        start_y = y1 + dy * square_size / 2
                        end_x = x2 - dx * square_size / 2
                        end_y = y2 - dy * square_size / 2

                    arrow = FancyArrowPatch(
                        (start_x, start_y),
                        (end_x, end_y),
                        arrowstyle="-|>",
                        mutation_scale=10,
                        color=color,
                        zorder=1,
                        linewidth=1,
                    )
                    ax.add_patch(arrow)

        # 绘制边标签
        edge_labels = nx.get_edge_attributes(G, "label")
        for edge, label in edge_labels.items():
            i, j = edge
            if float(label) == 0.0:
                continue
            # 根据带宽值映射标签颜色
            value = float(label)
            # 链路60%阈值起红
            if value <= link_mapping_min:
                intensity = 0.0
            else:
                intensity = (value - link_mapping_min) / (link_mapping_max - link_mapping_min)
            # clamp to [0,1]
            intensity = min(max(intensity, 0.0), 1.0)
            color = (intensity, 0, 0)
            if i == j:
                # 计算标签位置
                original_row = i // self.config.NUM_COL
                original_col = i % self.config.NUM_COL
                x, y = pos[i]

                offset = 0.17  # 标签偏移量
                if original_row == 0:
                    label_pos = (x, y + square_size / 2 + offset)
                    angle = 0
                elif original_row == self.config.NUM_ROW - 2:
                    label_pos = (x, y - square_size / 2 - offset)
                    angle = 0
                elif original_col == 0:
                    label_pos = (x - square_size / 2 - offset, y)
                    angle = -90
                elif original_col == self.config.NUM_COL - 1:
                    label_pos = (x + square_size / 2 + offset, y)
                    angle = 90
                else:
                    label_pos = (x, y + square_size / 2 + offset)
                    angle = 0

                ax.text(
                    *label_pos,
                    str(label),
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                    fontsize=dynamic_font * 0.9,
                    rotation=angle,
                )

            if i != j:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                angle = np.degrees(np.arctan2(dy, dx))

                has_reverse = G.has_edge(j, i)
                is_horizontal = abs(dx) > abs(dy)

                if has_reverse:
                    if is_horizontal:
                        perp_dx, perp_dy = -dy * 0.1 + 0.2, dx * 0.1
                    else:
                        perp_dx, perp_dy = -dy * 0.18, dx * 0.18 - 0.3
                    label_x = mid_x + perp_dx
                    label_y = mid_y + perp_dy
                else:
                    if is_horizontal:
                        label_x = mid_x + dx * 0.1
                        label_y = mid_y + dy * 0.1
                    else:
                        label_x = mid_x + (-dy * 0.1 if dx > 0 else dy * 0.1)
                        label_y = mid_y - 0.1

                ax.text(
                    label_x,
                    label_y,
                    str(label),
                    ha="center",
                    va="center",
                    fontsize=dynamic_font,
                    fontweight="bold",
                    color=color,
                )

        plt.axis("off")
        title = f"{network.name} - {mode.capitalize()} Bandwidth"
        if self.config.SPARE_CORE_ROW != -1:
            title += f"\nRow: {self.config.SPARE_CORE_ROW}, Failed cores: {self.config.FAIL_CORE_POS}, Spare cores: {self.config.spare_core_pos}"
        plt.title(title, fontsize=20)

        # # 添加图例说明
        # legend_text = f"IP {mode.capitalize()} Bandwidth (GB/s):\n" "SDMA: Top half of square\n" "GDMA: Bottom half of square"
        # plt.figtext(0.02, 0.98, legend_text, ha="TL", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

        plt.tight_layout()

        if save_path:
            plt.savefig(
                os.path.join(
                    save_path,
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

    def generate_unified_report(self, results: Dict, output_path: str) -> None:
        """
        生成统一的带宽分析报告

        Args:
            results: analyze_all_bandwidth()的返回结果
            output_path: 输出目录路径
        """
        os.makedirs(output_path, exist_ok=True)

        report_file = os.path.join(output_path, "bandwidth_analysis_report.txt")

        with open(report_file, "w", encoding="utf-8") as f:
            self._write_report_header(f, results)
            self._write_network_overall_section(f, results["network_overall"])
            self._generate_rn_ports_csv(results["rn_ports"], output_path)

        # 生成详细请求记录的CSV文件（读写分开）
        self._generate_detailed_request_csv(output_path)
        if self.base_model.verbose:
            print(f"带宽分析报告： {report_file}")
            print(f"具体RN端口的统计CSV： {output_path}rn_ports_bandwidth.csv")

    def _print_summary_to_console(self, results):
        """输出重要数据到控制台"""
        print("\n" + "=" * 60)
        print("网络带宽分析结果摘要")
        print("=" * 60)

        # 网络整体带宽
        read_metrics = results["network_overall"]["read"]
        write_metrics = results["network_overall"]["write"]
        mixed_metrics = results["network_overall"]["mixed"]

        print(f"网络整体带宽:")
        print(f"  读带宽  - 非加权: {read_metrics.unweighted_bandwidth:.3f} GB/s, 加权: {read_metrics.weighted_bandwidth:.3f} GB/s")
        print(f"  写带宽  - 非加权: {write_metrics.unweighted_bandwidth:.3f} GB/s, 加权: {write_metrics.weighted_bandwidth:.3f} GB/s")
        print(f"  混合带宽 - 非加权: {mixed_metrics.unweighted_bandwidth:.3f} GB/s, 加权: {mixed_metrics.weighted_bandwidth:.3f} GB/s")
        print(
            f"  总带宽  - 非加权: {read_metrics.unweighted_bandwidth + write_metrics.unweighted_bandwidth:.3f} GB/s, 加权: {read_metrics.weighted_bandwidth + write_metrics.weighted_bandwidth:.3f} GB/s"
        )
        print(f"  读带宽  - 平均非加权: {read_metrics.unweighted_bandwidth / self.config.NUM_IP:.3f} GB/s, 平均加权: {read_metrics.weighted_bandwidth / self.config.NUM_IP:.3f} GB/s")
        print(f"  写带宽  - 平均非加权: {write_metrics.unweighted_bandwidth / self.config.NUM_IP:.3f} GB/s, 平均加权: {write_metrics.weighted_bandwidth / self.config.NUM_IP:.3f} GB/s")
        print(f"  混合带宽 - 平均非加权: {mixed_metrics.unweighted_bandwidth / self.config.NUM_IP:.3f} GB/s, 平均加权: {mixed_metrics.weighted_bandwidth / self.config.NUM_IP:.3f} GB/s")
        print(
            f"  总带宽  - 平均非加权: {(read_metrics.unweighted_bandwidth + write_metrics.unweighted_bandwidth) / self.config.NUM_IP:.3f} GB/s, 平均加权: {(read_metrics.weighted_bandwidth + write_metrics.weighted_bandwidth) / self.config.NUM_IP:.3f} GB/s"
        )

        # 请求统计
        summary = results["summary"]
        print(f"\n请求统计:")
        print(f"  总请求数: {summary['total_requests']} (读: {summary['read_requests']}, 写: {summary['write_requests']})")
        print(f"  总flit数: {summary['total_read_flits'] + summary['total_write_flits']} (读: {summary['total_read_flits']}, 写: {summary['total_write_flits']})")

        # Circuit统计
        circuit_stats = summary.get("circuit_stats", {})
        if circuit_stats:
            print(f"\n绕环与Tag统计:")
            print(f"  Circuits req  - h: {circuit_stats.get('req_circuits_h', 0)}, v: {circuit_stats.get('req_circuits_v', 0)}")
            print(f"  Circuits rsp  - h: {circuit_stats.get('rsp_circuits_h', 0)}, v: {circuit_stats.get('rsp_circuits_v', 0)}")
            print(f"  Circuits data - h: {circuit_stats.get('data_circuits_h', 0)}, v: {circuit_stats.get('data_circuits_v', 0)}")
            print(f"  Wait cycle req  - h: {circuit_stats.get('req_wait_cycles_h', 0)}, v: {circuit_stats.get('req_wait_cycles_v', 0)}")
            print(f"  Wait cycle rsp  - h: {circuit_stats.get('rsp_wait_cycles_h', 0)}, v: {circuit_stats.get('rsp_wait_cycles_v', 0)}")
            print(f"  Wait cycle data - h: {circuit_stats.get('data_wait_cycles_h', 0)}, v: {circuit_stats.get('data_wait_cycles_v', 0)}")
            print(f"  RB ETag - T1: {circuit_stats.get('RB_ETag_T1_num', 0)}, T0: {circuit_stats.get('RB_ETag_T0_num', 0)}")
            print(f"  EQ ETag - T1: {circuit_stats.get('EQ_ETag_T1_num', 0)}, T0: {circuit_stats.get('EQ_ETag_T0_num', 0)}")
            print(f"  ITag - h: {circuit_stats.get('ITag_h_num', 0)}, v: {circuit_stats.get('ITag_v_num', 0)}")
            print(f"  Retry - read: {circuit_stats.get('read_retry_num', 0)}, write: {circuit_stats.get('write_retry_num', 0)}")

        # 工作区间统计
        print(f"\n工作区间统计:")
        print(f"  读操作工作区间: {len(read_metrics.working_intervals)}")
        print(f"  写操作工作区间: {len(write_metrics.working_intervals)}")
        print(f"  混合操作工作区间: {len(mixed_metrics.working_intervals)}")

        print("=" * 60)

        # 延迟统计 (cmd, data, transaction) - 用新helper
        stats = self._calculate_latency_stats()

        def _avg(cat, op):
            s = stats[cat][op]
            return s["sum"] / s["count"] if s["count"] else 0.0

        print("\n延迟统计 (单位: cycle)")
        for key, label in [("cmd", "CMD"), ("data", "Data"), ("trans", "Trans")]:
            print(
                f"  {label} 延迟  - "
                f"读: avg {_avg(key,'read'):.2f}, max {stats[key]['read']['max']}；"
                f"写: avg {_avg(key,'write'):.2f}, max {stats[key]['write']['max']}；"
                f"混合: avg {_avg(key,'mixed'):.2f}, max {stats[key]['mixed']['max']}"
            )

    def _write_report_header(self, f, results):
        """写入报告头部"""
        summary = results["summary"]
        f.write("=" * 80 + "\n")
        f.write("网络带宽分析报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("分析概览:\n")
        f.write(f"  总请求数: {summary['total_requests']}\n")
        f.write(f"  读请求数: {summary['read_requests']}\n")
        f.write(f"  写请求数: {summary['write_requests']}\n")
        f.write(f"  读取总flit数: {summary['total_read_flits']}\n")
        f.write(f"  写入总flit数: {summary['total_write_flits']}\n")
        f.write(f"  工作区间阈值: {summary['analysis_config']['min_gap_threshold_ns']} ns\n")

        # 添加Circuit统计信息
        circuit_stats = summary.get("circuit_stats", {})
        if circuit_stats:
            f.write("绕环与Tag统计:\n")
            f.write(f"  Circuits req  - h: {circuit_stats.get('req_circuits_h', 0)}, v: {circuit_stats.get('req_circuits_v', 0)}\n")
            f.write(f"  Circuits rsp  - h: {circuit_stats.get('rsp_circuits_h', 0)}, v: {circuit_stats.get('rsp_circuits_v', 0)}\n")
            f.write(f"  Circuits data - h: {circuit_stats.get('data_circuits_h', 0)}, v: {circuit_stats.get('data_circuits_v', 0)}\n")
            f.write(f"  Wait cycle req  - h: {circuit_stats.get('req_wait_cycles_h', 0)}, v: {circuit_stats.get('req_wait_cycles_v', 0)}\n")
            f.write(f"  Wait cycle rsp  - h: {circuit_stats.get('rsp_wait_cycles_h', 0)}, v: {circuit_stats.get('rsp_wait_cycles_v', 0)}\n")
            f.write(f"  Wait cycle data - h: {circuit_stats.get('data_wait_cycles_h', 0)}, v: {circuit_stats.get('data_wait_cycles_v', 0)}\n")
            f.write(f"  RB ETag - T1: {circuit_stats.get('RB_ETag_T1_num', 0)}, T0: {circuit_stats.get('RB_ETag_T0_num', 0)}\n")
            f.write(f"  EQ ETag - T1: {circuit_stats.get('EQ_ETag_T1_num', 0)}, T0: {circuit_stats.get('EQ_ETag_T0_num', 0)}\n")
            f.write(f"  ITag - h: {circuit_stats.get('ITag_h_num', 0)}, v: {circuit_stats.get('ITag_v_num', 0)}\n")
            f.write(f"  Retry - read: {circuit_stats.get('read_retry_num', 0)}, write: {circuit_stats.get('write_retry_num', 0)}\n")

        # 延迟统计 (cmd, data, transaction)
        stats = self._calculate_latency_stats()

        def _avg(cat, op):
            s = stats[cat][op]
            return s["sum"] / s["count"] if s["count"] else 0.0

        f.write("延迟统计 (cycle)：\n")
        for cat, label in [("cmd", "CMD"), ("data", "Data"), ("trans", "Trans")]:
            rl = stats[cat]
            f.write(
                f"  {label} 延迟 - "
                f"读 avg {_avg(cat,'read'):.2f}, max {rl['read']['max']}; "
                f"写 avg {_avg(cat,'write'):.2f}, max {rl['write']['max']}; "
                f"混合 avg {_avg(cat,'mixed'):.2f}, max {rl['mixed']['max']}\n"
            )

        f.write("\n")

    def _write_network_overall_section(self, f, network_overall):
        """写入网络整体带宽统计部分"""
        f.write("=" * 50 + "\n")
        f.write("网络带宽统计\n")
        f.write("=" * 50 + "\n\n")

        for operation in ["read", "write", "mixed"]:
            metrics = network_overall[operation]
            f.write(f"{operation.upper()} 操作带宽:\n")
            f.write(f"  非加权带宽: {metrics.unweighted_bandwidth:.3f} GB/s, 平均：{metrics.unweighted_bandwidth/self.config.NUM_IP:.3f}\n")
            f.write(f"  加权带宽:   {metrics.weighted_bandwidth:.3f} GB/s, 平均：{metrics.weighted_bandwidth/self.config.NUM_IP:.3f}\n\n")
            f.write(f"  网络工作时间: {metrics.network_start_time} - {metrics.network_end_time} ns (总计 {metrics.network_end_time - metrics.network_start_time} ns)\n")
            f.write(f"  实际工作时间: {metrics.total_working_time} ns\n")
            f.write(f"  工作区间数: {len(metrics.working_intervals)}\n")
            f.write(f"  请求总数: {metrics.total_requests}\n")
            f.write(f"  flit总数: {sum(interval.flit_count for interval in metrics.working_intervals)}\n")

            if metrics.working_intervals:
                f.write(f"  工作区间详情:\n")
                for i, interval in enumerate(metrics.working_intervals):
                    f.write(
                        f"    区间{i+1}: {interval.start_time}-{interval.end_time}ns, "
                        f"持续{interval.duration}ns, {interval.flit_count}个flit, "
                        f"平均带宽{interval.bandwidth_bytes_per_ns / self.config.NUM_IP:.3f}GB/s\n"
                    )
            f.write("\n")

    def _generate_rn_ports_csv(self, rn_ports, output_path: str):
        """
        生成RN端口详细统计的CSV文件

        Args:
            rn_ports: RN端口统计数据
            output_path: 输出目录路径
        """
        if not rn_ports:
            if hasattr(self, "base_model") and self.base_model and hasattr(self.base_model, "verbose") and self.base_model.verbose:
                print("没有RN端口数据，跳过CSV生成")
            return

        # CSV文件头部 - 增加混合带宽字段
        csv_header = [
            "port_id",
            "coordinate",
            "read_unweighted_bandwidth_gbps",
            "read_weighted_bandwidth_gbps",
            "write_unweighted_bandwidth_gbps",
            "write_weighted_bandwidth_gbps",
            "mixed_unweighted_bandwidth_gbps",
            "mixed_weighted_bandwidth_gbps",
            "read_requests_count",
            "write_requests_count",
            "total_requests_count",
            "read_flits_count",
            "write_flits_count",
            "total_flits_count",
            "read_working_intervals_count",
            "write_working_intervals_count",
            "mixed_working_intervals_count",
            "read_total_working_time_ns",
            "write_total_working_time_ns",
            "mixed_total_working_time_ns",
            "read_network_start_time_ns",
            "read_network_end_time_ns",
            "write_network_start_time_ns",
            "write_network_end_time_ns",
            "mixed_network_start_time_ns",
            "mixed_network_end_time_ns",
        ]

        # 生成RN端口CSV
        rn_csv_file = os.path.join(output_path, "rn_ports_bandwidth.csv")
        with open(rn_csv_file, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(csv_header) + "\n")

            # 按照特定顺序排序（先按类型，再按数字ID）
            sorted_ports = sorted(rn_ports.items(), key=lambda x: (x[0].split("_")[0], int(x[0].split("_")[-1])))

            for port_id, port_metrics in sorted_ports:
                # 计算坐标
                idx = int(port_id.rsplit("_", 1)[1])
                row = 4 - idx // self.config.NUM_COL // 2
                col = idx % self.config.NUM_COL
                coordinate = f"x{col}_y{row}"

                # 计算flit数量
                read_flits = sum(interval.flit_count for interval in port_metrics.read_metrics.working_intervals) if port_metrics.read_metrics.working_intervals else 0
                write_flits = sum(interval.flit_count for interval in port_metrics.write_metrics.working_intervals) if port_metrics.write_metrics.working_intervals else 0
                mixed_flits = sum(interval.flit_count for interval in port_metrics.mixed_metrics.working_intervals) if port_metrics.mixed_metrics.working_intervals else 0

                row_data = [
                    port_id,
                    coordinate,
                    port_metrics.read_metrics.unweighted_bandwidth,
                    port_metrics.read_metrics.weighted_bandwidth,
                    port_metrics.write_metrics.unweighted_bandwidth,
                    port_metrics.write_metrics.weighted_bandwidth,
                    port_metrics.mixed_metrics.unweighted_bandwidth,
                    port_metrics.mixed_metrics.weighted_bandwidth,
                    port_metrics.read_metrics.total_requests,
                    port_metrics.write_metrics.total_requests,
                    port_metrics.mixed_metrics.total_requests,
                    read_flits,
                    write_flits,
                    mixed_flits,
                    len(port_metrics.read_metrics.working_intervals),
                    len(port_metrics.write_metrics.working_intervals),
                    len(port_metrics.mixed_metrics.working_intervals),
                    port_metrics.read_metrics.total_working_time,
                    port_metrics.write_metrics.total_working_time,
                    port_metrics.mixed_metrics.total_working_time,
                    port_metrics.read_metrics.network_start_time,
                    port_metrics.read_metrics.network_end_time,
                    port_metrics.write_metrics.network_start_time,
                    port_metrics.write_metrics.network_end_time,
                    port_metrics.mixed_metrics.network_start_time,
                    port_metrics.mixed_metrics.network_end_time,
                ]

                f.write(",".join(map(str, row_data)) + "\n")

    def _write_rn_ports_section(self, f, rn_ports):
        """写入RN端口带宽统计部分"""
        f.write("=" * 50 + "\n")
        f.write("二、RN端口带宽统计\n")
        f.write("=" * 50 + "\n\n")

        if not rn_ports:
            f.write("没有RN端口数据\n\n")
            return

        f.write(f"RN端口统计概览:\n")
        f.write(f"  总端口数: {len(rn_ports)}\n")

        # 统计各类型端口数量
        port_types = {}
        for port_id in rn_ports.keys():
            port_type = port_id.split("_")[0]
            port_types[port_type] = port_types.get(port_type, 0) + 1

        for port_type, count in sorted(port_types.items()):
            f.write(f"  {port_type.upper()}端口: {count}个\n")

    def _write_usage_instructions(self, f):
        """写入使用说明"""
        f.write("=" * 50 + "\n")
        f.write("四、使用说明\n")
        f.write("=" * 50 + "\n\n")

        f.write("使用方法:\n")
        f.write("```python\n")
        f.write("# 1. 创建分析器\n")
        f.write("analyzer = BandwidthAnalyzer(config, min_gap_threshold=50)\n\n")
        f.write("# 2. 收集数据\n")
        f.write("analyzer.collect_requests_data(base_model)\n\n")
        f.write("# 3. 执行分析\n")
        f.write("results = analyzer.analyze_all_bandwidth()\n\n")
        f.write("# 4. 生成报告\n")
        f.write("analyzer.generate_unified_report(results, './output')\n\n")
        f.write("# 5. 获取特定结果\n")
        f.write("read_unweighted_bw = results['network_overall']['read'].unweighted_bandwidth\n")
        f.write("read_weighted_bw = results['network_overall']['read'].weighted_bandwidth\n")
        f.write("mixed_unweighted_bw = results['network_overall']['mixed'].unweighted_bandwidth\n")
        f.write("mixed_weighted_bw = results['network_overall']['mixed'].weighted_bandwidth\n")
        f.write("```\n\n")

        f.write("参数说明:\n")
        f.write("- min_gap_threshold: 工作区间合并阈值(ns)，默认50ns\n")
        f.write("- 非加权带宽: 总数据量 / 网络总时间\n")
        f.write("- 加权带宽: 各工作区间带宽按flit数量加权平均\n")
        f.write("- 混合带宽: 不区分读写请求类型的带宽统计\n")
        f.write("- 网络整体工作区间: 从第一笔请求发出到最后一笔数据完成传输\n")
        f.write("- RN端口读请求: 从第一笔请求到收到最后一笔读数据\n")
        f.write("- RN端口写请求: 从第一笔请求到发出最后一笔写数据\n")

    def _generate_detailed_request_csv(self, output_path: str):
        """
        生成详细请求记录的CSV文件，读写分开

        Args:
            output_path: 输出目录路径
        """
        # 分离读写请求
        read_requests = [req for req in self.requests if req.req_type == "read"]
        write_requests = [req for req in self.requests if req.req_type == "write"]

        # CSV文件头部
        csv_header = [
            "packet_id",
            "start_time_ns",
            "end_time_ns",
            "source_node",
            "source_type",
            "dest_node",
            "dest_type",
            "burst_length",
            # "total_bytes",
            "cmd_latency_ns",
            "data_latency_ns",
            "transaction_latency_ns",
        ]

        # 生成读请求CSV
        read_csv_file = os.path.join(output_path, "read_requests.csv")
        with open(read_csv_file, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(csv_header) + "\n")
            for req in read_requests:
                row = [
                    req.packet_id,
                    req.start_time,
                    req.end_time,
                    req.source_node,
                    req.source_type,
                    req.dest_node,
                    req.dest_type,
                    req.burst_length,
                    # req.total_bytes,
                    req.cmd_latency,
                    req.data_latency,
                    req.transaction_latency,
                ]
                f.write(",".join(map(str, row)) + "\n")

        # 生成写请求CSV
        write_csv_file = os.path.join(output_path, "write_requests.csv")
        with open(write_csv_file, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(csv_header) + "\n")
            for req in write_requests:
                row = [
                    req.packet_id,
                    req.start_time,
                    req.end_time,
                    req.source_node,
                    req.source_type,
                    req.dest_node,
                    req.dest_type,
                    req.burst_length,
                    # req.total_bytes,
                    req.cmd_latency,
                    req.data_latency,
                    req.transaction_latency,
                ]
                f.write(",".join(map(str, row)) + "\n")

        # 输出统计信息
        if hasattr(self, "base_model") and self.base_model and hasattr(self.base_model, "verbose") and self.base_model.verbose:
            print(f"详细请求记录统计:")
            print(f"  读请求CSV, {len(read_requests)} 条记录:  {read_csv_file}")
            print(f"  写请求CSV, {len(write_requests)} 条记录:  {write_csv_file}")

    def _generate_json_report(self, results: Dict, json_file: str):
        """生成JSON格式的详细数据报告"""
        # 转换为可序列化的格式
        serializable_results = {}

        # 网络整体数据 - 包含混合带宽
        serializable_results["network_overall"] = {}
        for op_type, metrics in results["network_overall"].items():
            serializable_results["network_overall"][op_type] = {
                "unweighted_bandwidth_gbps": metrics.unweighted_bandwidth,
                "weighted_bandwidth_gbps": metrics.weighted_bandwidth,
                "total_working_time_ns": metrics.total_working_time,
                "network_start_time_ns": metrics.network_start_time,
                "network_end_time_ns": metrics.network_end_time,
                "total_bytes": metrics.total_bytes,
                "total_requests": metrics.total_requests,
                "working_intervals": [
                    {
                        "start_time_ns": interval.start_time,
                        "end_time_ns": interval.end_time,
                        "duration_ns": interval.duration,
                        "flit_count": interval.flit_count,
                        "total_bytes": interval.total_bytes,
                        "request_count": interval.request_count,
                        "bandwidth_bytes_per_ns": interval.bandwidth_bytes_per_ns,
                    }
                    for interval in metrics.working_intervals
                ],
            }

        # RN端口数据 - 包含混合带宽
        serializable_results["rn_ports"] = {}
        for port_id, port_metrics in results["rn_ports"].items():
            serializable_results["rn_ports"][port_id] = {
                "read": {
                    "unweighted_bandwidth_gbps": port_metrics.read_metrics.unweighted_bandwidth,
                    "weighted_bandwidth_gbps": port_metrics.read_metrics.weighted_bandwidth,
                    "total_requests": port_metrics.read_metrics.total_requests,
                    "total_bytes": port_metrics.read_metrics.total_bytes,
                },
                "write": {
                    "unweighted_bandwidth_gbps": port_metrics.write_metrics.unweighted_bandwidth,
                    "weighted_bandwidth_gbps": port_metrics.write_metrics.weighted_bandwidth,
                    "total_requests": port_metrics.write_metrics.total_requests,
                    "total_bytes": port_metrics.write_metrics.total_bytes,
                },
                "mixed": {
                    "unweighted_bandwidth_gbps": port_metrics.mixed_metrics.unweighted_bandwidth,
                    "weighted_bandwidth_gbps": port_metrics.mixed_metrics.weighted_bandwidth,
                    "total_requests": port_metrics.mixed_metrics.total_requests,
                    "total_bytes": port_metrics.mixed_metrics.total_bytes,
                },
            }

        # 汇总数据
        serializable_results["summary"] = results["summary"]

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)


# 便捷使用函数
def analyze_bandwidth(base_model, config, output_path: str = "./bandwidth_analysis", min_gap_threshold: int = 50, plot_rn_bw_fig: bool = False, plot_flow_graph: bool = False) -> Dict:
    """
    便捷的带宽分析函数

    Args:
        base_model: BaseModel实例
        config: 配置对象
        output_path: 输出路径
        min_gap_threshold: 工作区间合并阈值(ns)

    Returns:
        分析结果字典
    """
    # 创建分析器
    analyzer = BandwidthAnalyzer(config, min_gap_threshold, plot_rn_bw_fig)

    # 收集数据
    analyzer.collect_requests_data(base_model)

    # 执行分析
    results = analyzer.analyze_all_bandwidth()

    # 生成报告
    analyzer.generate_unified_report(results, output_path)

    return results


# 使用示例
def main():
    """使用示例"""
    print("BandwidthAnalyzer 使用示例:")
    print()
    print("# 基本使用方法:")
    print("analyzer = BandwidthAnalyzer(config, min_gap_threshold=50)")
    print("analyzer.collect_requests_data(base_model)")
    print("results = analyzer.analyze_all_bandwidth()")
    print()
    print("# 获取混合读写带宽:")
    print("mixed_unweighted_bw = results['network_overall']['mixed'].unweighted_bandwidth")
    print("mixed_weighted_bw = results['network_overall']['mixed'].weighted_bandwidth")
    print()
    print("# 获取RN端口混合带宽:")
    print("for port_id, port_metrics in results['rn_ports'].items():")
    print("    print(f'{port_id} 混合带宽: {port_metrics.mixed_metrics.unweighted_bandwidth:.3f} GB/s')")


if __name__ == "__main__":
    main()
