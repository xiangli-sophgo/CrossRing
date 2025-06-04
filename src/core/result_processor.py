import numpy as np
import json
import os
from collections import defaultdict, namedtuple
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd


class WeightStrategy(Enum):
    """权重策略枚举"""

    REQUEST_COUNT = "request_count"
    FLIT_COUNT = "flit_count"
    BYTE_COUNT = "byte_count"
    UNIFORM = "uniform"


@dataclass
class RequestInfo:
    """请求信息数据结构"""

    packet_id: int
    start_time: int
    end_time: int
    req_type: str  # 'read' or 'write'
    source_node: int
    dest_node: int
    source_type: str  # 'gdma', 'sdma', etc.
    dest_type: str  # 'ddr', 'l2m', etc.
    burst_length: int
    total_bytes: int
    cmd_latency: int
    rsp_latency: int
    dat_latency: int
    total_latency: int


@dataclass
class WorkingInterval:
    """工作区间数据结构"""

    start_time: int
    end_time: int
    duration: int
    request_count: int
    flit_count: int
    total_bytes: int
    operation_types: Set[str]
    node_types: Set[str]
    rn_requests: int  # RN端请求数
    sn_requests: int  # SN端请求数

    @property
    def weight_by_requests(self) -> int:
        return self.request_count

    @property
    def weight_by_flits(self) -> int:
        return self.flit_count

    @property
    def weight_by_bytes(self) -> int:
        return self.total_bytes


@dataclass
class WeightedBandwidthResult:
    """加权带宽结果数据结构"""

    overall: float
    rn_bandwidth: float
    sn_bandwidth: float
    by_operation: Dict[str, float]
    by_node_type: Dict[str, float]
    by_network: Dict[str, float]
    working_intervals: List[WorkingInterval]
    total_working_time: int
    total_idle_time: int
    efficiency_ratio: float  # 工作时间占比


@dataclass
class TraditionalBandwidthResult:
    """传统带宽统计结果"""

    read_bandwidth: float
    write_bandwidth: float
    total_bandwidth: float
    read_latency_stats: Dict[str, float]
    write_latency_stats: Dict[str, float]
    finish_times: Dict[str, int]
    ip_bandwidth_stats: Dict[str, Dict[str, float]]


class ResultStatisticsProcessor:
    """完整的结果统计处理类 - 包含传统统计和加权带宽统计"""

    def __init__(self, config):
        """
        初始化结果统计处理器

        Args:
            config: 网络配置对象
            network_frequency: 网络频率
        """
        self.config = config
        self.network_frequency = config.NETWORK_FREQUENCY  # GHz

        # 加权带宽统计配置参数
        self.min_interval_duration = 10  # 最小区间长度(ns)
        self.max_merge_gap = 50  # 区间合并最大间隙(ns)
        self.idle_threshold = 100  # 空闲时间阈值(ns)

        # 数据存储
        self.requests: List[RequestInfo] = []
        self.working_intervals: List[WorkingInterval] = []
        self.base_model = None  # 存储base_model引用

        # 传统统计数据存储
        self.read_latency = {
            "total_latency": [],
            "cmd_latency": [],
            "rsp_latency": [],
            "dat_latency": [],
        }
        self.write_latency = {
            "total_latency": [],
            "cmd_latency": [],
            "rsp_latency": [],
            "dat_latency": [],
        }
        self.read_merged_intervals = [(0, 0, 0)]
        self.write_merged_intervals = [(0, 0, 0)]

        # IP带宽统计
        self.read_ip_intervals = defaultdict(list)
        self.write_ip_intervals = defaultdict(list)

        # 节点分类
        self.rn_positions = set()
        self.sn_positions = set()
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

    def collect_data_from_base_model(self, base_model) -> None:
        """
        从base_model收集统计所需的数据

        Args:
            base_model: BaseModel实例
        """
        self.base_model = base_model
        self.requests.clear()

        # 重置传统统计数据
        self.read_latency = {
            "total_latency": [],
            "cmd_latency": [],
            "rsp_latency": [],
            "dat_latency": [],
        }
        self.write_latency = {
            "total_latency": [],
            "cmd_latency": [],
            "rsp_latency": [],
            "dat_latency": [],
        }
        self.read_merged_intervals = [(0, 0, 0)]
        self.write_merged_intervals = [(0, 0, 0)]
        self.read_ip_intervals.clear()
        self.write_ip_intervals.clear()

        # 从数据网络的arrive_flits中提取请求信息
        for packet_id, flits in base_model.data_network.arrive_flits.items():
            if not flits or len(flits) != flits[0].burst_length:
                continue

            # 使用最后一个flit作为代表（完整的统计信息）
            representative_flit = flits[-1]

            request_info = RequestInfo(
                packet_id=packet_id,
                start_time=representative_flit.cmd_entry_cmd_table_cycle // self.network_frequency,
                end_time=representative_flit.arrival_cycle // self.network_frequency,
                req_type=representative_flit.req_type,
                source_node=representative_flit.source,
                dest_node=representative_flit.destination,
                source_type=representative_flit.original_source_type,
                dest_type=representative_flit.original_destination_type,
                burst_length=representative_flit.burst_length,
                total_bytes=representative_flit.burst_length * 128,  # 假设每个flit 128字节
                cmd_latency=representative_flit.cmd_latency // self.network_frequency,
                rsp_latency=representative_flit.rsp_latency // self.network_frequency,
                dat_latency=representative_flit.dat_latency // self.network_frequency,
                total_latency=representative_flit.total_latency // self.network_frequency,
            )

            self.requests.append(request_info)

        # 按开始时间排序
        self.requests.sort(key=lambda x: x.start_time)

    def evaluate_all_results(self, network) -> Tuple[TraditionalBandwidthResult, WeightedBandwidthResult]:
        """
        完整的结果评估，包含传统统计和加权带宽统计

        Args:
            network: 数据网络对象

        Returns:
            传统带宽结果和加权带宽结果的元组
        """
        if not self.base_model:
            raise ValueError("需要先调用 collect_data_from_base_model")

        # 执行传统的结果评估
        traditional_result = self._evaluate_traditional_results(network)

        # 执行加权带宽统计
        weighted_result = self.calculate_weighted_bandwidth(WeightStrategy.REQUEST_COUNT)

        return traditional_result, weighted_result

    def _evaluate_traditional_results(self, network) -> TraditionalBandwidthResult:
        """
        执行传统的结果评估逻辑（从原base_model的evaluate_results移植）

        Args:
            network: 网络对象

        Returns:
            传统带宽统计结果
        """
        if not self.base_model.result_save_path:
            return self._create_empty_traditional_result()

        # 保存配置
        with open(os.path.join(self.base_model.result_save_path, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=4)

        # 创建结果文件
        with open(
            os.path.join(self.base_model.result_save_path, f"Result_{self.base_model.file_name[10:-9]}R.txt"),
            "w",
        ) as f1, open(
            os.path.join(self.base_model.result_save_path, f"Result_{self.base_model.file_name[10:-9]}W.txt"),
            "w",
        ) as f2:

            # 打印表头
            header = (
                "tx_time(ns), src_id, src_type, des_id, des_type, R/W, burst_len, rx_time(ns), "
                "path, total_latency, cmd_latency, rsp_latency, dat_latency, circuits_completed_v, circuits_completed_h"
            )
            print(header, file=f1)
            print(header, file=f2)

            # 处理每个flit
            self._reset_special_stats()

            for flits in network.arrive_flits.values():
                if len(flits) != flits[0].burst_length:
                    continue
                for flit in flits:
                    self.base_model.data_cir_h_num_stat += flit.circuits_completed_h
                    self.base_model.data_cir_v_num_stat += flit.circuits_completed_v
                    self.base_model.data_wait_cycle_h_num_stat += flit.wait_cycle_h
                    self.base_model.data_wait_cycle_v_num_stat += flit.wait_cycle_v

                self._process_flits_traditional(flits[-1], network, f1, f2)

        # 计算并输出结果
        return self._calculate_and_output_traditional_results(network)

    def _reset_special_stats(self):
        """重置特殊统计变量"""
        self.base_model.sdma_R_ddr_finish_time = 0
        self.base_model.sdma_W_l2m_finish_time = 0
        self.base_model.gdma_R_l2m_finish_time = 0
        self.base_model.sdma_R_ddr_flit_num = 0
        self.base_model.sdma_W_l2m_flit_num = 0
        self.base_model.gdma_R_l2m_flit_num = 0
        self.base_model.sdma_R_ddr_latency = []
        self.base_model.sdma_W_l2m_latency = []
        self.base_model.gdma_R_l2m_latency = []

    def _process_flits_traditional(self, flit, network, f1, f2):
        """处理单个flit的传统统计（从原base_model移植）"""
        # 计算延迟
        flit.total_latency = (flit.arrival_cycle - flit.cmd_entry_cmd_table_cycle) // self.network_frequency
        flit.cmd_latency = (flit.sn_receive_req_cycle - flit.req_entry_network_cycle) // self.network_frequency

        if flit.req_type == "read":
            flit.rsp_latency = 0
            flit.dat_latency = (flit.rn_data_collection_complete_cycle - flit.sn_receive_req_cycle) // self.network_frequency
            self.base_model.rn_bandwidth[f"{flit.original_source_type[:-2].upper()} {flit.req_type} {flit.original_destination_type[:3].upper()}"]["time"].append(
                flit.rn_data_collection_complete_cycle // self.network_frequency
            )
        elif flit.req_type == "write":
            flit.rsp_latency = (flit.rn_receive_rsp_cycle - flit.sn_receive_req_cycle) // self.network_frequency
            flit.dat_latency = (flit.sn_data_collection_complete_cycle - flit.data_entry_network_cycle) // self.network_frequency
            self.base_model.rn_bandwidth[f"{flit.original_source_type[:-2].upper()} {flit.req_type} {flit.original_destination_type[:3].upper()}"]["time"].append(
                flit.data_entry_network_cycle // self.network_frequency
            )

        # 更新合并区间和延迟
        if flit.req_type == "read":
            self._update_intervals_traditional(flit, self.read_merged_intervals, self.read_latency, f1, "R")
        elif flit.req_type == "write":
            self._update_intervals_traditional(flit, self.write_merged_intervals, self.write_latency, f2, "W")

    def _update_intervals_traditional(self, flit, merged_intervals, latency, file, req_type):
        """更新传统的区间和延迟统计"""
        # 根据请求类型更新对应的IP区间
        if req_type == "R":
            dma_id = f"{str(flit.original_source_type)}_{str(flit.destination + self.config.NUM_COL)}"
            ddr_id = f"{str(flit.original_destination_type)}_{str(flit.source)}"
            dma_intervals = self.read_ip_intervals[dma_id]
            ddr_intervals = self.read_ip_intervals[ddr_id]
        elif req_type == "W":
            dma_id = f"{str(flit.original_source_type)}_{str(flit.source)}"
            ddr_id = f"{str(flit.original_destination_type)}_{str(flit.destination + self.config.NUM_COL)}"
            dma_intervals = self.write_ip_intervals[dma_id]
            ddr_intervals = self.write_ip_intervals[ddr_id]

        # 合并区间逻辑
        current_start = flit.req_departure_cycle // self.network_frequency
        current_end = flit.arrival_cycle // self.network_frequency
        current_count = flit.burst_length

        # 更新 dma_intervals
        self._merge_ip_intervals(dma_intervals, current_start, current_end, current_count)
        self._merge_ip_intervals(ddr_intervals, current_start, current_end, current_count)

        # 更新总体区间
        new_interval = (current_start, current_end, current_count)
        self._merge_global_intervals(merged_intervals, new_interval)

        # 更新特殊统计
        self._update_special_stats(flit, req_type)

        # 更新延迟统计
        latency["total_latency"].append(flit.total_latency // self.network_frequency)
        latency["cmd_latency"].append(flit.cmd_latency // self.network_frequency)
        latency["rsp_latency"].append(flit.rsp_latency // self.network_frequency)
        latency["dat_latency"].append(flit.dat_latency // self.network_frequency)

        # 写入文件
        print(
            f"{flit.req_departure_cycle // self.network_frequency},{flit.source_original},"
            f"{flit.original_source_type},{flit.destination_original},{flit.original_destination_type},"
            f"{req_type},{flit.burst_length},{flit.arrival_cycle // self.network_frequency},"
            f"{flit.path},{flit.total_latency},{flit.cmd_latency},{flit.rsp_latency},"
            f"{flit.dat_latency},{flit.circuits_completed_v},{flit.circuits_completed_h}",
            file=file,
        )

    def _merge_ip_intervals(self, intervals, start, end, count):
        """合并IP区间"""
        if not intervals:
            intervals.append((start, end, count))
        else:
            while intervals and start <= intervals[-1][1]:
                last_start, last_end, last_count = intervals.pop()
                start = min(last_start, start)
                end = max(last_end, end)
                count += last_count
            intervals.append((start, end, count))

    def _merge_global_intervals(self, merged_intervals, new_interval):
        """合并全局区间"""
        if not merged_intervals:
            merged_intervals.append(new_interval)
        else:
            while merged_intervals and (new_interval[0] <= merged_intervals[-1][1]):
                last_start, last_end, last_count = merged_intervals.pop()
                merged_start = min(last_start, new_interval[0])
                merged_end = max(last_end, new_interval[1])
                merged_count = last_count + new_interval[2]
                new_interval = (merged_start, merged_end, merged_count)
            merged_intervals.append(new_interval)

    def _update_special_stats(self, flit, req_type):
        """更新特殊统计数据"""
        if flit.source_type == "ddr" and flit.destination_type == "sdma" and req_type == "R":
            self.base_model.sdma_R_ddr_finish_time = max(
                self.base_model.sdma_R_ddr_finish_time,
                flit.arrival_cycle // self.network_frequency,
            )
            self.base_model.sdma_R_ddr_flit_num += flit.burst_length
            if flit.leave_db_cycle is None:
                flit.leave_db_cycle = flit.arrival_cycle
            self.base_model.sdma_R_ddr_latency.append(flit.leave_db_cycle - flit.entry_db_cycle)
        elif flit.source_type == "ddr" and flit.destination_type == "sdma" and req_type == "W":
            self.base_model.sdma_W_l2m_finish_time = max(
                self.base_model.sdma_W_l2m_finish_time,
                flit.arrival_cycle // self.network_frequency,
            )
            self.base_model.sdma_W_l2m_flit_num += flit.burst_length
            if flit.leave_db_cycle is None:
                flit.leave_db_cycle = flit.arrival_cycle
            self.base_model.sdma_W_l2m_latency.append(flit.leave_db_cycle - flit.entry_db_cycle)
        elif flit.source_type == "l2m" and flit.destination_type == "gdma" and req_type == "R":
            self.base_model.gdma_R_l2m_finish_time = max(
                self.base_model.gdma_R_l2m_finish_time,
                flit.arrival_cycle // self.network_frequency,
            )
            self.base_model.gdma_R_l2m_flit_num += flit.burst_length
            if flit.leave_db_cycle is None:
                flit.leave_db_cycle = flit.arrival_cycle
            self.base_model.gdma_R_l2m_latency.append(flit.leave_db_cycle - flit.entry_db_cycle)

    def _calculate_and_output_traditional_results(self, network) -> TraditionalBandwidthResult:
        """计算并输出传统统计结果"""
        # 计算平均延迟
        for source in self.base_model.flit_positions:
            destination = source - self.config.NUM_COL
            if network.inject_time[source]:
                network.avg_inject_time[source] = sum(network.inject_time[source]) / len(network.inject_time[source])
            if network.eject_time[destination]:
                network.avg_eject_time[destination] = sum(network.eject_time[destination]) / len(network.eject_time[destination])

        # 计算电路统计
        network.avg_circuits_h = sum(network.circuits_h) / len(network.circuits_h) / 2 if network.circuits_h else None
        network.max_circuits_h = max(network.circuits_h) / 2 if network.circuits_h else None
        network.avg_circuits_v = sum(network.circuits_v) / len(network.circuits_v) / 2 if network.circuits_v else None
        network.max_circuits_v = max(network.circuits_v) / 2 if network.circuits_v else None

        # 输出总结果
        if self.base_model.verbose:
            print("=" * 50)

        total_result_path = os.path.join(self.base_model.result_save_path, "total_result.txt")
        with open(total_result_path, "w", encoding="utf-8") as f3:
            if self.base_model.verbose:
                print(f"Topology: {self.base_model.topo_type_stat}, file_name: {self.base_model.file_name}")
            print(f"Topology: {self.base_model.topo_type_stat}, file_name: {self.base_model.file_name}", file=f3)

            # 处理读写带宽统计
            read_bandwidth = 0
            write_bandwidth = 0
            read_latency_stats = {}
            write_latency_stats = {}

            if self.read_latency:
                read_bandwidth, read_latency_stats = self._output_intervals_traditional(f3, self.read_merged_intervals, "Read", self.read_latency)

            if self.write_latency:
                write_bandwidth, write_latency_stats = self._output_intervals_traditional(f3, self.write_merged_intervals, "Write", self.write_latency)

            # 输出IP带宽统计
            ip_bandwidth_stats = self._calculate_ip_bandwidth_stats(f3)

        # 计算RN带宽
        self.base_model.cal_rn_bandwidth()

        # 更新base_model统计
        self._update_base_model_stats(read_bandwidth, write_bandwidth, read_latency_stats, write_latency_stats)

        # 绘制图表
        if self.base_model.plot_flow_fig:
            self.base_model.draw_flow_graph(network, save_path=self.base_model.results_fig_save_path)

        return TraditionalBandwidthResult(
            read_bandwidth=read_bandwidth,
            write_bandwidth=write_bandwidth,
            total_bandwidth=read_bandwidth + write_bandwidth,
            read_latency_stats=read_latency_stats,
            write_latency_stats=write_latency_stats,
            finish_times={"read": self.base_model.R_finish_time_stat, "write": self.base_model.W_finish_time_stat},
            ip_bandwidth_stats=ip_bandwidth_stats,
        )

    def _output_intervals_traditional(self, f3, merged_intervals, req_type, latency):
        """输出传统区间统计"""
        print(f"{req_type} intervals:", file=f3)
        if self.base_model.verbose:
            print(f"{req_type} results:")

        total_count = 0
        finish_time = 0
        total_interval_time = 0

        for start, end, count in merged_intervals:
            if start == end:
                continue
            interval_bandwidth = count * 128 / (end - start) / self.config.NUM_IP
            interval_time = end - start
            total_interval_time += interval_time
            total_count += count
            print(f"Interval: {start} to {end}, count: {count}, bandwidth: {interval_bandwidth:.1f}", file=f3)
            finish_time = max(finish_time, end)

        # 带宽计算
        if total_interval_time > 0:
            total_bandwidth = total_count * 128 / total_interval_time / (self.config.NUM_RN if req_type == "Read" else self.config.NUM_SN)
        else:
            total_bandwidth = 0

        # 更新finish time和tail latency
        if req_type == "Read":
            self.base_model.R_finish_time_stat = finish_time
            self.base_model.R_tail_latency_stat = finish_time - self.base_model.R_tail_latency_stat // self.network_frequency
        elif req_type == "Write":
            self.base_model.W_finish_time_stat = finish_time
            self.base_model.W_tail_latency_stat = finish_time - self.base_model.W_tail_latency_stat // self.network_frequency

        # 计算延迟统计
        latency_stats = {
            "total_avg": np.average(latency["total_latency"]) if latency["total_latency"] else 0,
            "total_max": max(latency["total_latency"]) if latency["total_latency"] else 0,
            "cmd_avg": np.average(latency["cmd_latency"]) if latency["cmd_latency"] else 0,
            "cmd_max": max(latency["cmd_latency"]) if latency["cmd_latency"] else 0,
            "rsp_avg": np.average(latency["rsp_latency"]) if latency["rsp_latency"] else 0,
            "rsp_max": max(latency["rsp_latency"]) if latency["rsp_latency"] else 0,
            "dat_avg": np.average(latency["dat_latency"]) if latency["dat_latency"] else 0,
            "dat_max": max(latency["dat_latency"]) if latency["dat_latency"] else 0,
        }

        # 输出统计信息
        print(
            f"Bandwidth: {total_bandwidth:.1f}; \nTotal latency: Avg: {latency_stats['total_avg']:.1f}, Max: {latency_stats['total_max']}; "
            f"cmd_latency: Avg: {latency_stats['cmd_avg']:.1f}, Max: {latency_stats['cmd_max']}; "
            f"rsp_latency: Avg: {latency_stats['rsp_avg']:.1f}, Max: {latency_stats['rsp_max']}; "
            f"dat_latency: Avg: {latency_stats['dat_avg']:.1f}, Max: {latency_stats['dat_max']}",
            file=f3,
        )

        if self.base_model.verbose:
            print(
                f"Bandwidth: {total_bandwidth:.1f}; \nTotal latency: Avg: {latency_stats['total_avg']:.1f}, Max: {latency_stats['total_max']}; "
                f"cmd_latency: Avg: {latency_stats['cmd_avg']:.1f}, Max: {latency_stats['cmd_max']}; "
                f"rsp_latency: Avg: {latency_stats['rsp_avg']:.1f}, Max: {latency_stats['rsp_max']}; "
                f"dat_latency: Avg: {latency_stats['dat_avg']:.1f}, Max: {latency_stats['dat_max']}"
            )

        return total_bandwidth, latency_stats

    def _calculate_ip_bandwidth_stats(self, f3):
        """计算IP带宽统计"""
        print("\nPer-IP Weighted Bandwidth:", file=f3)

        ip_stats = {}

        # 处理读带宽
        print("\nRead Bandwidth per IP:", file=f3)
        rn_read_bws = []
        sn_read_bws = []

        for ip_id in sorted(self.read_ip_intervals.keys(), key=lambda k: (k.split("_")[0], int(k.split("_")[-1]))):
            idx = int(ip_id.rsplit("_", 1)[1])
            row = 4 - idx // self.config.NUM_COL // 2
            col = idx % self.config.NUM_COL
            intervals = self.read_ip_intervals[ip_id]
            bw = self._calculate_ip_bandwidth(intervals)
            ip_name = f"{ip_id.rsplit('_', 1)[0]}_x{col}_y{row}"
            print(f"{ip_id} {ip_name}: {bw:.1f} GB/s", file=f3)

            if ip_id.startswith(("gdma", "sdma")):
                rn_read_bws.append(bw)
            elif ip_id.startswith(("ddr", "l2m")):
                sn_read_bws.append(bw)

            ip_stats[f"{ip_id}_read"] = bw

        # 处理写带宽
        print("\nWrite Bandwidth per IP:", file=f3)
        rn_write_bws = []
        sn_write_bws = []

        for ip_id in sorted(self.write_ip_intervals.keys(), key=lambda k: (k.split("_")[0], int(k.split("_")[-1]))):
            idx = int(ip_id.rsplit("_", 1)[1])
            row = 4 - idx // self.config.NUM_COL // 2
            col = idx % self.config.NUM_COL
            intervals = self.write_ip_intervals[ip_id]
            bw = self._calculate_ip_bandwidth(intervals)
            ip_name = f"{ip_id.rsplit('_', 1)[0]}_x{col}_y{row}"
            print(f"{ip_id} {ip_name}: {bw:.1f} GB/s", file=f3)

            if ip_id.startswith(("gdma", "sdma")):
                rn_write_bws.append(bw)
            elif ip_id.startswith(("ddr", "l2m")):
                sn_write_bws.append(bw)

            ip_stats[f"{ip_id}_write"] = bw

        # 输出统计信息
        print("")  # 屏幕输出空行分隔
        self._print_stats(rn_read_bws, "RN", "Read", f3)
        self._print_stats(sn_read_bws, "SN", "Read", f3)

        print("")  # 屏幕输出空行分隔
        self._print_stats(rn_write_bws, "RN", "Write", f3)
        self._print_stats(sn_write_bws, "SN", "Write", f3)

        return ip_stats

    def _print_stats(self, bw_list, name, operation, file):
        """打印带宽统计信息"""
        if bw_list:
            avg = sum(bw_list) / getattr(self.config, f"NUM_{name}")
            min_bw = min(bw_list)
            max_bw = max(bw_list)

            if name == "RN":
                self.base_model.RN_BW_avg_stat = avg
                self.base_model.RN_BW_min_stat = min_bw
                self.base_model.RN_BW_max_stat = max_bw
            elif name == "SN":
                self.base_model.SN_BW_avg_stat = avg
                self.base_model.SN_BW_min_stat = min_bw
                self.base_model.SN_BW_max_stat = max_bw

            print(f"\n{name} {operation} Bandwidth Stats:", file=file)
            print(f" Sum: {sum(bw_list)}, Average: {avg:.1f} GB/s", file=file)
            print(f"  Range: {min_bw:.1f} - {max_bw:.1f} GB/s", file=file)

            # 屏幕输出
            if self.base_model.verbose:
                print(f"{name} {operation}: Sum: {sum(bw_list):.1f}, Avg: {avg:.1f} GB/s, Range: {min_bw:.1f}-{max_bw:.1f} GB/s")

    def _calculate_ip_bandwidth(self, intervals):
        """计算给定区间的加权带宽"""
        total_count = 0
        total_interval_time = 0

        for start, end, count in intervals:
            if start >= end:
                continue  # 跳过无效区间
            interval_time = end - start
            total_count += count
            total_interval_time += interval_time

        return total_count * 128 / total_interval_time if total_interval_time > 0 else 0.0

    def _update_base_model_stats(self, read_bandwidth, write_bandwidth, read_latency_stats, write_latency_stats):
        """更新base_model的统计变量"""
        # 带宽统计
        self.base_model.read_BW_stat = read_bandwidth
        self.base_model.write_BW_stat = write_bandwidth
        self.base_model.Total_BW_stat = read_bandwidth + write_bandwidth

        # 读延迟统计
        self.base_model.read_total_latency_avg_stat = read_latency_stats.get("total_avg", 0)
        self.base_model.read_total_latency_max_stat = read_latency_stats.get("total_max", 0)
        self.base_model.read_cmd_latency_avg_stat = read_latency_stats.get("cmd_avg", 0)
        self.base_model.read_cmd_latency_max_stat = read_latency_stats.get("cmd_max", 0)
        self.base_model.read_rsp_latency_avg_stat = read_latency_stats.get("rsp_avg", 0)
        self.base_model.read_rsp_latency_max_stat = read_latency_stats.get("rsp_max", 0)
        self.base_model.read_dat_latency_avg_stat = read_latency_stats.get("dat_avg", 0)
        self.base_model.read_dat_latency_max_stat = read_latency_stats.get("dat_max", 0)

        # 写延迟统计
        self.base_model.write_total_latency_avg_stat = write_latency_stats.get("total_avg", 0)
        self.base_model.write_total_latency_max_stat = write_latency_stats.get("total_max", 0)
        self.base_model.write_cmd_latency_avg_stat = write_latency_stats.get("cmd_avg", 0)
        self.base_model.write_cmd_latency_max_stat = write_latency_stats.get("cmd_max", 0)
        self.base_model.write_rsp_latency_avg_stat = write_latency_stats.get("rsp_avg", 0)
        self.base_model.write_rsp_latency_max_stat = write_latency_stats.get("rsp_max", 0)
        self.base_model.write_dat_latency_avg_stat = write_latency_stats.get("dat_avg", 0)
        self.base_model.write_dat_latency_max_stat = write_latency_stats.get("dat_max", 0)

        # 输出总带宽
        if self.base_model.verbose:
            print(f"Read + Write Bandwidth: {self.base_model.Total_BW_stat:.1f}")
            print("=" * 50)
            print(f"Total Circuits req h: {self.base_model.req_cir_h_num_stat}, v: {self.base_model.req_cir_v_num_stat}")
            print(f"Total Circuits rsp h: {self.base_model.rsp_cir_h_num_stat}, v: {self.base_model.rsp_cir_v_num_stat}")
            print(f"Total Circuits data h: {self.base_model.data_cir_h_num_stat}, v: {self.base_model.data_cir_v_num_stat}")
            print(f"Total wait cycle req h: {self.base_model.req_wait_cycle_h_num_stat}, v: {self.base_model.req_wait_cycle_v_num_stat}")
            print(f"Total wait cycle rsp h: {self.base_model.rsp_wait_cycle_h_num_stat}, v: {self.base_model.rsp_wait_cycle_v_num_stat}")
            print(f"Total wait cycle data h: {self.base_model.data_wait_cycle_h_num_stat}, v: {self.base_model.data_wait_cycle_v_num_stat}")
            print(
                f"Total RB ETag: T1: {self.base_model.RB_ETag_T1_num_stat}, T0: {self.base_model.RB_ETag_T0_num_stat}; EQ ETag: T1: {self.base_model.EQ_ETag_T1_num_stat}, T0: {self.base_model.EQ_ETag_T0_num_stat}"
            )
            print(f"Total ITag: h: {self.base_model.ITag_h_num_stat}, v: {self.base_model.ITag_v_num_stat}")

            if self.base_model.model_type_stat == "REQ_RSP":
                for ip_pos in self.base_model.flit_positions:
                    for ip_type in self.config.CH_NAME_LIST:
                        ip_interface = self.base_model.ip_modules[(ip_type, ip_pos)]
                        self.base_model.read_retry_num_stat += ip_interface.read_retry_num_stat
                        self.base_model.write_retry_num_stat += ip_interface.write_retry_num_stat
                print(f"Retry num: R: {self.base_model.read_retry_num_stat}, W: {self.base_model.write_retry_num_stat}")

    def _create_empty_traditional_result(self) -> TraditionalBandwidthResult:
        """创建空的传统结果对象"""
        return TraditionalBandwidthResult(read_bandwidth=0.0, write_bandwidth=0.0, total_bandwidth=0.0, read_latency_stats={}, write_latency_stats={}, finish_times={}, ip_bandwidth_stats={})

    # ==================== 加权带宽统计部分 ====================

    def identify_working_intervals(self) -> List[WorkingInterval]:
        """
        识别工作区间，去除空闲时间段

        Returns:
            工作区间列表
        """
        if not self.requests:
            return []

        # 收集所有时间点
        time_events = []
        for req in self.requests:
            time_events.append((req.start_time, "start", req))
            time_events.append((req.end_time, "end", req))

        # 按时间排序
        time_events.sort(key=lambda x: x[0])

        # 构建初始活跃区间
        raw_intervals = []
        active_requests = set()
        current_start = None

        for time_point, event_type, req in time_events:
            if event_type == "start":
                if not active_requests:  # 从空闲状态变为活跃状态
                    current_start = time_point
                active_requests.add(req.packet_id)
            else:  # event_type == 'end'
                active_requests.discard(req.packet_id)
                if not active_requests and current_start is not None:  # 从活跃状态变为空闲状态
                    if time_point - current_start >= self.min_interval_duration:
                        raw_intervals.append((current_start, time_point))
                    current_start = None

        # 如果最后还有活跃请求，添加最后一个区间
        if active_requests and current_start is not None:
            last_end = max(req.end_time for req in self.requests)
            if last_end - current_start >= self.min_interval_duration:
                raw_intervals.append((current_start, last_end))

        # 合并相邻的区间
        merged_intervals = self._merge_intervals(raw_intervals)

        # 构建WorkingInterval对象
        working_intervals = []
        for start, end in merged_intervals:
            interval_requests = [req for req in self.requests if req.start_time < end and req.end_time > start]

            if not interval_requests:
                continue

            # 计算区间统计信息
            total_bytes = sum(req.total_bytes for req in interval_requests)
            flit_count = sum(req.burst_length for req in interval_requests)
            operation_types = set(req.req_type for req in interval_requests)
            node_types = set()
            for req in interval_requests:
                node_types.add(req.source_type)
                node_types.add(req.dest_type)

            # 统计RN和SN端请求数
            rn_requests = sum(1 for req in interval_requests if req.source_node in self.rn_positions)
            sn_requests = sum(1 for req in interval_requests if req.dest_node in self.sn_positions)

            interval = WorkingInterval(
                start_time=start,
                end_time=end,
                duration=end - start,
                request_count=len(interval_requests),
                flit_count=flit_count,
                total_bytes=total_bytes,
                operation_types=operation_types,
                node_types=node_types,
                rn_requests=rn_requests,
                sn_requests=sn_requests,
            )

            working_intervals.append(interval)

        self.working_intervals = working_intervals
        return working_intervals

    def _merge_intervals(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        合并相邻或重叠的区间

        Args:
            intervals: 原始区间列表

        Returns:
            合并后的区间列表
        """
        if not intervals:
            return []

        # 按开始时间排序
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]

        for current_start, current_end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]

            # 如果当前区间与上一个区间的间隙小于阈值，则合并
            if current_start - last_end <= self.max_merge_gap:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        return merged

    def calculate_weighted_bandwidth(self, weight_strategy: WeightStrategy = WeightStrategy.REQUEST_COUNT) -> WeightedBandwidthResult:
        """
        计算加权带宽

        Args:
            weight_strategy: 权重策略

        Returns:
            加权带宽结果
        """
        if not self.working_intervals:
            self.identify_working_intervals()

        if not self.working_intervals:
            return self._create_empty_weighted_result()

        # 计算总体加权带宽
        overall_bandwidth = self._calculate_overall_weighted_bandwidth(weight_strategy)

        # 计算RN端加权带宽
        rn_bandwidth = self._calculate_rn_weighted_bandwidth(weight_strategy)

        # 计算SN端加权带宽
        sn_bandwidth = self._calculate_sn_weighted_bandwidth(weight_strategy)

        # 按操作类型分类计算
        by_operation = self._calculate_bandwidth_by_operation(weight_strategy)

        # 按节点类型分类计算
        by_node_type = self._calculate_bandwidth_by_node_type(weight_strategy)

        # 按网络类型分类计算（简化版本）
        by_network = {"request": overall_bandwidth * 0.3, "response": overall_bandwidth * 0.2, "data": overall_bandwidth * 0.5}  # 估算

        # 计算时间统计
        total_working_time = sum(interval.duration for interval in self.working_intervals)
        total_time = max(req.end_time for req in self.requests) - min(req.start_time for req in self.requests)
        total_idle_time = total_time - total_working_time
        efficiency_ratio = total_working_time / total_time if total_time > 0 else 0

        return WeightedBandwidthResult(
            overall=overall_bandwidth,
            rn_bandwidth=rn_bandwidth,
            sn_bandwidth=sn_bandwidth,
            by_operation=by_operation,
            by_node_type=by_node_type,
            by_network=by_network,
            working_intervals=self.working_intervals,
            total_working_time=total_working_time,
            total_idle_time=total_idle_time,
            efficiency_ratio=efficiency_ratio,
        )

    def _get_interval_weight(self, interval: WorkingInterval, strategy: WeightStrategy) -> float:
        """获取区间权重"""
        if strategy == WeightStrategy.REQUEST_COUNT:
            return interval.weight_by_requests
        elif strategy == WeightStrategy.FLIT_COUNT:
            return interval.weight_by_flits
        elif strategy == WeightStrategy.BYTE_COUNT:
            return interval.weight_by_bytes
        else:  # UNIFORM
            return 1.0

    def _calculate_overall_weighted_bandwidth(self, weight_strategy: WeightStrategy) -> float:
        """计算整体加权带宽"""
        total_weighted_bytes = 0
        total_weighted_time = 0

        for interval in self.working_intervals:
            weight = self._get_interval_weight(interval, weight_strategy)
            total_weighted_bytes += interval.total_bytes * weight
            total_weighted_time += interval.duration * weight

        if total_weighted_time == 0:
            return 0.0

        return total_weighted_bytes / total_weighted_time  # bytes per ns

    def _calculate_rn_weighted_bandwidth(self, weight_strategy: WeightStrategy) -> float:
        """计算RN端加权带宽"""
        total_weighted_bytes = 0
        total_weighted_time = 0

        for interval in self.working_intervals:
            weight = self._get_interval_weight(interval, weight_strategy)
            # 计算RN相关的字节数（从RN发出的请求）
            rn_requests = [req for req in self.requests if (req.start_time < interval.end_time and req.end_time > interval.start_time and req.source_node in self.rn_positions)]
            rn_bytes = sum(req.total_bytes for req in rn_requests)

            total_weighted_bytes += rn_bytes * weight
            total_weighted_time += interval.duration * weight

        if total_weighted_time == 0:
            return 0.0

        return total_weighted_bytes / total_weighted_time

    def _calculate_sn_weighted_bandwidth(self, weight_strategy: WeightStrategy) -> float:
        """计算SN端加权带宽"""
        total_weighted_bytes = 0
        total_weighted_time = 0

        for interval in self.working_intervals:
            weight = self._get_interval_weight(interval, weight_strategy)
            # 计算SN相关的字节数（发送到SN的请求）
            sn_requests = [req for req in self.requests if (req.start_time < interval.end_time and req.end_time > interval.start_time and req.dest_node in self.sn_positions)]
            sn_bytes = sum(req.total_bytes for req in sn_requests)

            total_weighted_bytes += sn_bytes * weight
            total_weighted_time += interval.duration * weight

        if total_weighted_time == 0:
            return 0.0

        return total_weighted_bytes / total_weighted_time

    def _calculate_bandwidth_by_operation(self, weight_strategy: WeightStrategy) -> Dict[str, float]:
        """按操作类型计算加权带宽"""
        result = {}

        for op_type in ["read", "write"]:
            total_weighted_bytes = 0
            total_weighted_time = 0

            for interval in self.working_intervals:
                if op_type not in interval.operation_types:
                    continue

                weight = self._get_interval_weight(interval, weight_strategy)
                # 计算该操作类型的字节数
                op_requests = [req for req in self.requests if (req.start_time < interval.end_time and req.end_time > interval.start_time and req.req_type == op_type)]
                op_bytes = sum(req.total_bytes for req in op_requests)

                total_weighted_bytes += op_bytes * weight
                total_weighted_time += interval.duration * weight

            result[op_type] = total_weighted_bytes / total_weighted_time if total_weighted_time > 0 else 0.0

        return result

    def _calculate_bandwidth_by_node_type(self, weight_strategy: WeightStrategy) -> Dict[str, float]:
        """按节点类型计算加权带宽"""
        result = {}
        node_types = ["gdma", "sdma", "ddr", "l2m"]

        for node_type in node_types:
            total_weighted_bytes = 0
            total_weighted_time = 0

            for interval in self.working_intervals:
                if node_type not in interval.node_types:
                    continue

                weight = self._get_interval_weight(interval, weight_strategy)
                # 计算涉及该节点类型的字节数
                node_requests = [
                    req
                    for req in self.requests
                    if (req.start_time < interval.end_time and req.end_time > interval.start_time and (req.source_type.startswith(node_type) or req.dest_type.startswith(node_type)))
                ]
                node_bytes = sum(req.total_bytes for req in node_requests)

                total_weighted_bytes += node_bytes * weight
                total_weighted_time += interval.duration * weight

            result[node_type] = total_weighted_bytes / total_weighted_time if total_weighted_time > 0 else 0.0

        return result

    def _create_empty_weighted_result(self) -> WeightedBandwidthResult:
        """创建空的加权带宽结果对象"""
        return WeightedBandwidthResult(
            overall=0.0, rn_bandwidth=0.0, sn_bandwidth=0.0, by_operation={}, by_node_type={}, by_network={}, working_intervals=[], total_working_time=0, total_idle_time=0, efficiency_ratio=0.0
        )

    def generate_comprehensive_report(self, traditional_result: TraditionalBandwidthResult, weighted_result: WeightedBandwidthResult, output_path: str) -> None:
        """
        生成综合统计报告

        Args:
            traditional_result: 传统带宽结果
            weighted_result: 加权带宽结果
            output_path: 输出路径
        """
        os.makedirs(output_path, exist_ok=True)

        # 生成综合文本报告
        report_file = os.path.join(output_path, "comprehensive_statistics_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=== 综合统计报告 ===\n\n")

            # 传统统计结果
            f.write("=== 传统带宽统计 ===\n")
            f.write(f"读带宽: {traditional_result.read_bandwidth:.2f} GB/s\n")
            f.write(f"写带宽: {traditional_result.write_bandwidth:.2f} GB/s\n")
            f.write(f"总带宽: {traditional_result.total_bandwidth:.2f} GB/s\n\n")

            # 加权带宽统计结果
            f.write("=== 加权带宽统计 ===\n")
            f.write(f"整体加权带宽: {weighted_result.overall:.2f} GB/s\n")
            f.write(f"RN端加权带宽: {weighted_result.rn_bandwidth:.2f} GB/s\n")
            f.write(f"SN端加权带宽: {weighted_result.sn_bandwidth:.2f} GB/s\n")
            f.write(f"网络工作效率: {weighted_result.efficiency_ratio:.2%}\n")
            f.write(f"工作区间数量: {len(weighted_result.working_intervals)}\n\n")

            # 对比分析
            f.write("=== 对比分析 ===\n")
            if traditional_result.total_bandwidth > 0:
                efficiency_gain = (weighted_result.overall - traditional_result.total_bandwidth) / traditional_result.total_bandwidth * 100
                f.write(f"加权带宽相对传统带宽提升: {efficiency_gain:.1f}%\n")
            f.write(f"时间利用率: {weighted_result.efficiency_ratio:.2%}\n")
            f.write(f"平均区间持续时间: {weighted_result.total_working_time / len(weighted_result.working_intervals) if weighted_result.working_intervals else 0:.1f} ns\n")

        # 生成JSON数据
        json_file = os.path.join(output_path, "comprehensive_statistics_data.json")
        json_data = {
            "traditional_statistics": {
                "read_bandwidth": traditional_result.read_bandwidth,
                "write_bandwidth": traditional_result.write_bandwidth,
                "total_bandwidth": traditional_result.total_bandwidth,
                "read_latency_stats": traditional_result.read_latency_stats,
                "write_latency_stats": traditional_result.write_latency_stats,
            },
            "weighted_statistics": {
                "overall_bandwidth": weighted_result.overall,
                "rn_bandwidth": weighted_result.rn_bandwidth,
                "sn_bandwidth": weighted_result.sn_bandwidth,
                "by_operation": weighted_result.by_operation,
                "by_node_type": weighted_result.by_node_type,
                "efficiency_ratio": weighted_result.efficiency_ratio,
                "working_intervals_count": len(weighted_result.working_intervals),
            },
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"综合统计报告已生成: {report_file}")
        print(f"数据文件已生成: {json_file}")

    def plot_comprehensive_analysis(self, traditional_result: TraditionalBandwidthResult, weighted_result: WeightedBandwidthResult, output_path: str) -> None:
        """
        绘制综合分析图表

        Args:
            traditional_result: 传统带宽结果
            weighted_result: 加权带宽结果
            output_path: 输出路径
        """
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 传统 vs 加权带宽对比
        categories = ["传统总带宽", "加权总带宽", "RN端加权", "SN端加权"]
        values = [traditional_result.total_bandwidth, weighted_result.overall, weighted_result.rn_bandwidth, weighted_result.sn_bandwidth]
        colors = ["lightblue", "lightcoral", "lightgreen", "gold"]

        bars = ax1.bar(categories, values, color=colors)
        ax1.set_ylabel("带宽 (GB/s)")
        ax1.set_title("传统 vs 加权带宽对比")
        ax1.tick_params(axis="x", rotation=45)

        # 在柱状图上添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{value:.2f}", ha="center", va="bottom")

        # 2. 工作区间时间线
        if weighted_result.working_intervals:
            intervals_start = [interval.start_time for interval in weighted_result.working_intervals]
            intervals_duration = [interval.duration for interval in weighted_result.working_intervals]
            intervals_bytes = [interval.total_bytes for interval in weighted_result.working_intervals]

            # 使用颜色表示字节数
            if max(intervals_bytes) > 0:
                colors = plt.cm.viridis(np.array(intervals_bytes) / max(intervals_bytes))
            else:
                colors = "blue"

            ax2.barh(range(len(weighted_result.working_intervals)), intervals_duration, left=intervals_start, color=colors, alpha=0.7)
            ax2.set_xlabel("时间 (ns)")
            ax2.set_ylabel("工作区间")
            ax2.set_title("工作区间时间线分析")
            ax2.grid(True, alpha=0.3)

        # 3. 按操作类型的加权带宽
        if weighted_result.by_operation:
            operations = list(weighted_result.by_operation.keys())
            bandwidths = list(weighted_result.by_operation.values())
            ax3.pie(bandwidths, labels=operations, autopct="%1.1f%%", startangle=90)
            ax3.set_title("按操作类型的加权带宽分布")

        # 4. 效率分析
        efficiency_data = {"工作时间占比": weighted_result.efficiency_ratio, "空闲时间占比": 1 - weighted_result.efficiency_ratio}

        labels = list(efficiency_data.keys())
        sizes = list(efficiency_data.values())
        colors = ["green", "red"]

        ax4.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax4.set_title("网络时间利用率分析")

        plt.tight_layout()

        # 保存图表
        plot_file = os.path.join(output_path, "comprehensive_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"综合分析图表已保存: {plot_file}")

    def get_comprehensive_summary(self, traditional_result: TraditionalBandwidthResult, weighted_result: WeightedBandwidthResult) -> Dict[str, Any]:
        """
        获取综合摘要指标

        Args:
            traditional_result: 传统带宽结果
            weighted_result: 加权带宽结果

        Returns:
            综合摘要指标字典
        """
        return {
            # 传统统计指标
            "traditional_read_bandwidth_gbps": round(traditional_result.read_bandwidth, 2),
            "traditional_write_bandwidth_gbps": round(traditional_result.write_bandwidth, 2),
            "traditional_total_bandwidth_gbps": round(traditional_result.total_bandwidth, 2),
            # 加权统计指标
            "weighted_overall_bandwidth_gbps": round(weighted_result.overall, 2),
            "weighted_rn_bandwidth_gbps": round(weighted_result.rn_bandwidth, 2),
            "weighted_sn_bandwidth_gbps": round(weighted_result.sn_bandwidth, 2),
            "network_efficiency_ratio": round(weighted_result.efficiency_ratio, 4),
            "working_intervals_count": len(weighted_result.working_intervals),
            # 对比指标
            "bandwidth_improvement_ratio": round(
                (weighted_result.overall - traditional_result.total_bandwidth) / traditional_result.total_bandwidth if traditional_result.total_bandwidth > 0 else 0, 4
            ),
            "avg_interval_duration_ns": round(weighted_result.total_working_time / len(weighted_result.working_intervals) if weighted_result.working_intervals else 0, 2),
            # 基础统计
            "total_requests": len(self.requests),
            "total_working_time_ns": weighted_result.total_working_time,
            "total_idle_time_ns": weighted_result.total_idle_time,
        }

    def update_base_model_weighted_stats(self, weighted_result: WeightedBandwidthResult, weight_strategy: WeightStrategy) -> None:
        """
        将加权带宽统计结果更新到base_model的统计变量中

        Args:
            weighted_result: 加权带宽结果
            weight_strategy: 使用的权重策略
        """
        if not self.base_model:
            return

        # 根据权重策略设置不同的变量名后缀
        strategy_suffix = {WeightStrategy.REQUEST_COUNT: "by_requests", WeightStrategy.FLIT_COUNT: "by_flits", WeightStrategy.BYTE_COUNT: "by_bytes", WeightStrategy.UNIFORM: "uniform"}.get(
            weight_strategy, "unknown"
        )

        # 更新加权带宽统计变量
        setattr(self.base_model, f"weighted_overall_BW_{strategy_suffix}_stat", weighted_result.overall)
        setattr(self.base_model, f"weighted_rn_BW_{strategy_suffix}_stat", weighted_result.rn_bandwidth)
        setattr(self.base_model, f"weighted_sn_BW_{strategy_suffix}_stat", weighted_result.sn_bandwidth)
        setattr(self.base_model, f"network_efficiency_ratio_{strategy_suffix}_stat", weighted_result.efficiency_ratio)

        # 更新按操作类型的带宽
        for op_type, bandwidth in weighted_result.by_operation.items():
            setattr(self.base_model, f"weighted_{op_type}_BW_{strategy_suffix}_stat", bandwidth)

        # 更新按节点类型的带宽
        for node_type, bandwidth in weighted_result.by_node_type.items():
            setattr(self.base_model, f"weighted_{node_type}_BW_{strategy_suffix}_stat", bandwidth)

        # 更新时间统计
        setattr(self.base_model, f"total_working_time_{strategy_suffix}_stat", weighted_result.total_working_time)
        setattr(self.base_model, f"total_idle_time_{strategy_suffix}_stat", weighted_result.total_idle_time)
        setattr(self.base_model, f"working_intervals_count_{strategy_suffix}_stat", len(weighted_result.working_intervals))

        if self.base_model.verbose:
            print(f"加权带宽统计 ({strategy_suffix}): 整体={weighted_result.overall:.2f} GB/s, " f"RN={weighted_result.rn_bandwidth:.2f} GB/s, " f"SN={weighted_result.sn_bandwidth:.2f} GB/s")
            print(f"网络工作效率: {weighted_result.efficiency_ratio:.2%}")

    def validate_comprehensive_results(self, traditional_result: TraditionalBandwidthResult, weighted_result: WeightedBandwidthResult) -> Dict[str, bool]:
        """
        验证综合结果的一致性

        Args:
            traditional_result: 传统带宽结果
            weighted_result: 加权带宽结果

        Returns:
            验证结果字典
        """
        validations = {}

        # 检查带宽值是否为正数
        validations["positive_bandwidth"] = all(
            [
                traditional_result.read_bandwidth >= 0,
                traditional_result.write_bandwidth >= 0,
                traditional_result.total_bandwidth >= 0,
                weighted_result.overall >= 0,
                weighted_result.rn_bandwidth >= 0,
                weighted_result.sn_bandwidth >= 0,
            ]
        )

        # 检查带宽值的合理性关系
        validations["bandwidth_relationship"] = abs(traditional_result.total_bandwidth - (traditional_result.read_bandwidth + traditional_result.write_bandwidth)) < 0.01

        # 检查加权带宽的合理性
        validations["weighted_bandwidth_reasonable"] = (
            weighted_result.overall > 0 and weighted_result.rn_bandwidth <= weighted_result.overall * 1.1 and weighted_result.sn_bandwidth <= weighted_result.overall * 1.1  # 允许10%的误差
        )

        # 检查时间统计的一致性
        validations["time_consistency"] = weighted_result.total_working_time >= 0 and weighted_result.total_idle_time >= 0 and 0 <= weighted_result.efficiency_ratio <= 1

        # 检查工作区间的合理性
        validations["intervals_valid"] = all(interval.duration > 0 and interval.start_time < interval.end_time for interval in weighted_result.working_intervals)

        # 检查请求数据的一致性
        validations["request_data_consistent"] = len(self.requests) > 0 and all(req.total_bytes > 0 and req.burst_length > 0 for req in self.requests)

        return validations


# 使用示例和辅助函数
def create_result_processor(config, network_frequency: int = 1000) -> ResultStatisticsProcessor:
    """
    工厂函数：创建结果统计处理器实例

    Args:
        config: 网络配置对象
        network_frequency: 网络频率

    Returns:
        ResultStatisticsProcessor实例
    """
    return ResultStatisticsProcessor(config, network_frequency)


def process_simulation_results(base_model, processor: ResultStatisticsProcessor = None) -> Tuple[TraditionalBandwidthResult, WeightedBandwidthResult]:
    """
    处理仿真结果的便捷函数

    Args:
        base_model: BaseModel实例
        processor: 可选的结果处理器实例

    Returns:
        传统带宽结果和加权带宽结果的元组
    """
    if processor is None:
        processor = create_result_processor(base_model.config, base_model.config.NETWORK_FREQUENCY)

    # 收集数据
    processor.collect_data_from_base_model(base_model)

    # 评估结果
    traditional_result, weighted_result = processor.evaluate_all_results(base_model.data_network)

    return traditional_result, weighted_result


# 使用示例主函数
def main():
    """使用示例"""
    # 这里只是示例，实际使用时需要传入真实的config和base_model
    from config.config import CrossRingConfig

    config_path = r"../../config/config2.json"
    config = CrossRingConfig(config_path)
    processor = ResultStatisticsProcessor(config)

    # 模拟使用流程：
    # processor.collect_data_from_base_model(base_model)
    # traditional_result, weighted_result = processor.evaluate_all_results(network)
    # processor.generate_comprehensive_report(traditional_result, weighted_result, "./output")
    # processor.plot_comprehensive_analysis(traditional_result, weighted_result, "./output")

    print("ResultStatisticsProcessor 类已创建完成")
    print("主要功能包括：")
    print("1. 传统带宽统计（从原base_model移植）")
    print("2. 加权带宽统计（新增功能）")
    print("3. 综合报告生成")
    print("4. 可视化分析图表")
    print("5. 结果验证和一致性检查")


if __name__ == "__main__":
    main()
