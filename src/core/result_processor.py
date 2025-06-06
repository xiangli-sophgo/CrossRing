import numpy as np
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
from src.utils.component import *


@dataclass
class RequestInfo:
    """请求信息数据结构"""

    packet_id: int
    start_time: int  # ns
    end_time: int  # ns
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


class BandwidthAnalyzer:
    """
    带宽分析器 - 统一的带宽统计分析类

    功能：
    1. 统计工作区间（去除空闲时间段）
    2. 计算非加权和加权带宽
    3. 分别统计读写操作
    4. 网络整体和RN端口带宽统计
    5. 生成统一报告
    """

    def __init__(self, config, min_gap_threshold: int = 50):
        """
        初始化带宽分析器

        Args:
            config: 网络配置对象
            min_gap_threshold: 工作区间合并阈值(ns)，小于此值的间隔被视为同一工作区间
        """
        self.config = config
        self.min_gap_threshold = min_gap_threshold
        self.network_frequency = config.NETWORK_FREQUENCY  # GHz

        # 数据存储
        self.requests: List[RequestInfo] = []
        self.rn_positions = set()
        self.sn_positions = set()

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
        """
        从base_model收集请求数据

        Args:
            base_model: BaseModel实例
        """
        self.requests.clear()
        self.base_model = base_model  # 保存base_model引用以获取统计数据

        # 从数据网络的arrive_flits中提取请求信息
        for packet_id, flits in base_model.data_network.arrive_flits.items():
            if not flits or len(flits) != flits[0].burst_length:
                continue

            # 使用最后一个flit作为代表（包含完整统计信息）
            representative_flit: Flit = flits[-1]

            request_info = RequestInfo(
                packet_id=packet_id,
                start_time=representative_flit.cmd_entry_cake0_cycle // self.network_frequency,
                end_time=representative_flit.data_received_complete_cycle // self.network_frequency,
                req_type=representative_flit.req_type,
                source_node=representative_flit.source,
                dest_node=representative_flit.destination,
                source_type=representative_flit.original_source_type,
                dest_type=representative_flit.original_destination_type,
                burst_length=representative_flit.burst_length,
                total_bytes=representative_flit.burst_length * 128,  # 每个flit 128字节
                cmd_latency=representative_flit.cmd_latency // self.network_frequency,
                data_latency=representative_flit.data_latency // self.network_frequency,
                transaction_latency=representative_flit.transaction_latency // self.network_frequency,
            )

            self.requests.append(request_info)

        # 按开始时间排序
        self.requests.sort(key=lambda x: x.start_time)

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

    def calculate_bandwidth_metrics(self, requests: List[RequestInfo], operation_type: str) -> BandwidthMetrics:
        """
        计算指定操作类型的带宽指标

        Args:
            requests: 所有请求列表
            operation_type: 'read' 或 'write'

        Returns:
            BandwidthMetrics对象
        """
        # 筛选指定类型的请求
        filtered_requests = [req for req in requests if req.req_type == operation_type]

        if not filtered_requests:
            return BandwidthMetrics(
                unweighted_bandwidth=0.0, weighted_bandwidth=0.0, working_intervals=[], total_working_time=0, network_start_time=0, network_end_time=0, total_bytes=0, total_requests=0
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
        计算网络整体带宽（读和写）

        Returns:
            {'read': BandwidthMetrics, 'write': BandwidthMetrics}
        """
        results = {}

        for operation in ["read", "write"]:
            results[operation] = self.calculate_bandwidth_metrics(self.requests, operation)

        return results

    def calculate_rn_port_bandwidth(self) -> Dict[str, PortBandwidthMetrics]:
        """
        计算每个RN端口的带宽

        Returns:
            {port_id: PortBandwidthMetrics}
        """
        port_metrics = {}

        # 按源节点分组RN端口的请求
        rn_requests_by_port = defaultdict(list)

        for req in self.requests:
            if req.source_node in self.rn_positions:
                # 对于读请求：从第一笔请求到收到最后一笔读数据
                # 对于写请求：从第一笔请求到发出最后一笔写数据
                port_id = f"{req.source_type}_{req.source_node}"
                rn_requests_by_port[port_id].append(req)

        # 计算每个端口的读写带宽
        for port_id, port_requests in rn_requests_by_port.items():
            read_metrics = self.calculate_bandwidth_metrics(port_requests, "read")
            write_metrics = self.calculate_bandwidth_metrics(port_requests, "write")

            port_metrics[port_id] = PortBandwidthMetrics(port_id=port_id, read_metrics=read_metrics, write_metrics=write_metrics)

        return port_metrics

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

        return results

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

        print(f"网络整体带宽:")
        print(f"  读带宽  - 非加权: {read_metrics.unweighted_bandwidth:.3f} GB/s, 加权: {read_metrics.weighted_bandwidth:.3f} GB/s")
        print(f"  写带宽  - 非加权: {write_metrics.unweighted_bandwidth:.3f} GB/s, 加权: {write_metrics.weighted_bandwidth:.3f} GB/s")
        print(
            f"  总带宽  - 非加权: {read_metrics.unweighted_bandwidth + write_metrics.unweighted_bandwidth:.3f} GB/s, 加权: {read_metrics.weighted_bandwidth + write_metrics.weighted_bandwidth:.3f} GB/s"
        )
        print(f"  读带宽  - 平均非加权: {read_metrics.unweighted_bandwidth / self.config.NUM_IP:.3f} GB/s, 平均加权: {read_metrics.weighted_bandwidth / self.config.NUM_IP:.3f} GB/s")
        print(f"  写带宽  - 平均非加权: {write_metrics.unweighted_bandwidth / self.config.NUM_IP:.3f} GB/s, 平均加权: {write_metrics.weighted_bandwidth / self.config.NUM_IP:.3f} GB/s")
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

        print("=" * 60)

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

        f.write("\n")

    def _write_network_overall_section(self, f, network_overall):
        """写入网络整体带宽统计部分"""
        f.write("=" * 50 + "\n")
        f.write("网络带宽统计\n")
        f.write("=" * 50 + "\n\n")

        for operation in ["read", "write"]:
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

        # CSV文件头部
        csv_header = [
            "port_id",
            "coordinate",
            "read_unweighted_bandwidth_gbps",
            "read_weighted_bandwidth_gbps",
            "write_unweighted_bandwidth_gbps",
            "write_weighted_bandwidth_gbps",
            "read_requests_count",
            "write_requests_count",
            "read_flits_count",
            "write_flits_count",
            "read_working_intervals_count",
            "write_working_intervals_count",
            "read_total_working_time_ns",
            "write_total_working_time_ns",
            "read_network_start_time_ns",
            "read_network_end_time_ns",
            "write_network_start_time_ns",
            "write_network_end_time_ns",
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

                row_data = [
                    port_id,
                    coordinate,
                    port_metrics.read_metrics.unweighted_bandwidth,
                    port_metrics.read_metrics.weighted_bandwidth,
                    port_metrics.write_metrics.unweighted_bandwidth,
                    port_metrics.write_metrics.weighted_bandwidth,
                    port_metrics.read_metrics.total_requests,
                    port_metrics.write_metrics.total_requests,
                    read_flits,
                    write_flits,
                    len(port_metrics.read_metrics.working_intervals),
                    len(port_metrics.write_metrics.working_intervals),
                    port_metrics.read_metrics.total_working_time,
                    port_metrics.write_metrics.total_working_time,
                    port_metrics.read_metrics.network_start_time,
                    port_metrics.read_metrics.network_end_time,
                    port_metrics.write_metrics.network_start_time,
                    port_metrics.write_metrics.network_end_time,
                ]

                f.write(",".join(map(str, row_data)) + "\n")

        # 输出统计信息
        # if hasattr(self, "base_model") and self.base_model and hasattr(self.base_model, "verbose") and self.base_model.verbose:
        #     print(f"RN端口统计:")
        #     print(f"  RN端口: {len(rn_ports)} 个端口 -> {rn_csv_file}")

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
        f.write("```\n\n")

        f.write("参数说明:\n")
        f.write("- min_gap_threshold: 工作区间合并阈值(ns)，默认50ns\n")
        f.write("- 非加权带宽: 总数据量 / 网络总时间\n")
        f.write("- 加权带宽: 各工作区间带宽按flit数量加权平均\n")
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

        # 网络整体数据
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

        # RN端口数据
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
            }

        # 汇总数据
        serializable_results["summary"] = results["summary"]

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)


# 便捷使用函数
def analyze_bandwidth(base_model, config, output_path: str = "./bandwidth_analysis", min_gap_threshold: int = 50) -> Dict:
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
    analyzer = BandwidthAnalyzer(config, min_gap_threshold)

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
