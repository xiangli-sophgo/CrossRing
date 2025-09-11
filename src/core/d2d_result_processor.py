"""
D2D系统专用结果处理器

用于处理跨Die通信的带宽统计和请求记录
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle, FancyArrowPatch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .result_processor import BandwidthAnalyzer, RequestInfo, BandwidthMetrics, WorkingInterval
from src.utils.components.flit import Flit


@dataclass
class D2DRequestInfo:
    """D2D请求信息数据结构"""

    packet_id: int
    source_die: int
    target_die: int
    source_node: int
    target_node: int
    source_type: str
    target_type: str
    req_type: str  # 'read' or 'write'
    burst_length: int
    data_bytes: int
    start_time_ns: int
    end_time_ns: int
    latency_ns: int


@dataclass
class D2DBandwidthStats:
    """D2D带宽统计数据结构"""

    die0_to_die1_read_bw: float = 0.0
    die0_to_die1_read_bw_weighted: float = 0.0
    die0_to_die1_write_bw: float = 0.0
    die0_to_die1_write_bw_weighted: float = 0.0

    die1_to_die0_read_bw: float = 0.0
    die1_to_die0_read_bw_weighted: float = 0.0
    die1_to_die0_write_bw: float = 0.0
    die1_to_die0_write_bw_weighted: float = 0.0

    total_read_requests: int = 0
    total_write_requests: int = 0
    total_bytes_transferred: int = 0


class D2DResultProcessor(BandwidthAnalyzer):
    """D2D系统专用的结果处理器，继承自BandwidthAnalyzer"""

    FLIT_SIZE_BYTES = 128  # 每个flit的字节数
    MIN_SIMULATION_TIME_NS = 1000  # 最小仿真时间（纳秒），避免除零
    MAX_BANDWIDTH_NORMALIZATION = 256.0  # 最大带宽归一化值（GB/s）

    def __init__(self, config, min_gap_threshold: int = 50):
        super().__init__(config, min_gap_threshold)
        self.d2d_requests: List[D2DRequestInfo] = []
        self.d2d_stats = D2DBandwidthStats()
        # 修复网络频率属性问题
        self.network_frequency = getattr(config, "NETWORK_FREQUENCY", 2)

    def collect_cross_die_requests(self, dies: Dict):
        """
        从两个Die的网络中收集跨Die请求数据

        Args:
            dies: Dict[die_id, die_model] - Die模型字典
        """
        self.d2d_requests.clear()

        for die_id, die_model in dies.items():
            # 检查数据网络中的arrive_flits
            if hasattr(die_model, "data_network") and hasattr(die_model.data_network, "arrive_flits"):
                self._collect_requests_from_network(die_model.data_network, die_id)

        # print(f"[D2D结果处理] 收集到 {len(self.d2d_requests)} 个跨Die请求")

    def _collect_requests_from_network(self, network, die_id: int):
        """从单个网络中收集跨Die请求"""
        for packet_id, flits in network.arrive_flits.items():
            if not flits:
                continue

            first_flit = flits[0]
            # 改进数据验证
            if not hasattr(first_flit, "burst_length") or len(flits) != first_flit.burst_length:
                continue

            last_flit = flits[-1]

            # 检查是否为跨Die请求
            if not self._is_cross_die_request(first_flit):
                continue

            # 只记录请求发起方Die的数据，避免重复记录
            if hasattr(first_flit, "d2d_origin_die") and first_flit.d2d_origin_die != die_id:
                continue

            # 提取D2D信息
            d2d_info = self._extract_d2d_info(first_flit, last_flit, packet_id)
            if d2d_info:
                self.d2d_requests.append(d2d_info)

    def _is_cross_die_request(self, flit: Flit) -> bool:
        """检查flit是否为跨Die请求"""
        return (
            hasattr(flit, "d2d_origin_die") and hasattr(flit, "d2d_target_die") and flit.d2d_origin_die is not None and flit.d2d_target_die is not None and flit.d2d_origin_die != flit.d2d_target_die
        )

    def _extract_d2d_info(self, first_flit: Flit, last_flit: Flit, packet_id: int) -> Optional[D2DRequestInfo]:
        """从flit中提取D2D请求信息"""
        try:
            # 计算开始时间 - 优先使用req_start_cycle（tracker消耗开始）
            if hasattr(first_flit, "req_start_cycle") and first_flit.req_start_cycle < float("inf"):
                start_time_ns = first_flit.req_start_cycle // self.network_frequency
            elif hasattr(first_flit, "cmd_entry_noc_from_cake0_cycle") and first_flit.cmd_entry_noc_from_cake0_cycle < float("inf"):
                start_time_ns = first_flit.cmd_entry_noc_from_cake0_cycle // self.network_frequency
            else:
                start_time_ns = 0

            # 计算结束时间 - 根据请求类型选择合适的时间戳
            req_type = getattr(first_flit, "req_type", "unknown")
            if req_type == "read":
                # 读请求：使用data_received_complete_cycle（读数据到达时tracker释放）
                if hasattr(last_flit, "data_received_complete_cycle") and last_flit.data_received_complete_cycle < float("inf"):
                    end_time_ns = last_flit.data_received_complete_cycle // self.network_frequency
                else:
                    end_time_ns = start_time_ns
            elif req_type == "write":
                # 写请求：使用write_complete_received_cycle（写完成响应到达时tracker释放）
                if hasattr(first_flit, "write_complete_received_cycle") and first_flit.write_complete_received_cycle < float("inf"):
                    end_time_ns = first_flit.write_complete_received_cycle // self.network_frequency
                else:
                    end_time_ns = start_time_ns
            else:
                end_time_ns = start_time_ns

            latency_ns = end_time_ns - start_time_ns if end_time_ns > start_time_ns else 0

            # 计算数据量
            burst_length = getattr(first_flit, "burst_length", 1)
            data_bytes = burst_length * self.FLIT_SIZE_BYTES

            return D2DRequestInfo(
                packet_id=packet_id,
                source_die=getattr(first_flit, "d2d_origin_die", 0),
                target_die=getattr(first_flit, "d2d_target_die", 1),
                source_node=getattr(first_flit, "d2d_origin_node", 0),
                target_node=getattr(first_flit, "d2d_target_node", 0),
                source_type=getattr(first_flit, "d2d_origin_type", ""),
                target_type=getattr(first_flit, "d2d_target_type", ""),
                req_type=getattr(first_flit, "req_type", "unknown"),
                burst_length=burst_length,
                data_bytes=data_bytes,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                latency_ns=latency_ns,
            )
        except (AttributeError, KeyError, ValueError) as e:
            # print(f"[D2D结果处理] 提取请求信息失败 packet_id={packet_id}: {e}")
            return None
        except Exception as e:
            # print(f"[D2D结果处理] 未预期的错误 packet_id={packet_id}: {e}")
            raise

    def save_d2d_requests_csv(self, output_path: str):
        """
        保存D2D请求到CSV文件

        Args:
            output_path: 输出目录路径
        """
        os.makedirs(output_path, exist_ok=True)

        # 分别保存读请求和写请求
        read_requests = [req for req in self.d2d_requests if req.req_type == "read"]
        write_requests = [req for req in self.d2d_requests if req.req_type == "write"]

        # CSV文件头
        csv_header = ["packet_id", "source_die", "target_die", "source_node", "target_node", "source_type", "target_type", "burst_length", "start_time_ns", "end_time_ns", "latency_ns", "data_bytes"]

        # 只有存在请求时才保存对应的CSV文件
        if read_requests:
            read_csv_path = os.path.join(output_path, "d2d_read_requests.csv")
            self._save_requests_to_csv(read_requests, read_csv_path, csv_header)
            print(f"[D2D结果处理] 已保存 {len(read_requests)} 个读请求到 {read_csv_path}")
        else:
            print(f"[D2D结果处理] 无读请求数据，跳过读请求CSV文件生成")

        if write_requests:
            write_csv_path = os.path.join(output_path, "d2d_write_requests.csv")
            self._save_requests_to_csv(write_requests, write_csv_path, csv_header)
            print(f"[D2D结果处理] 已保存 {len(write_requests)} 个写请求到 {write_csv_path}")
        else:
            print(f"[D2D结果处理] 无写请求数据，跳过写请求CSV文件生成")

    def _save_requests_to_csv(self, requests: List[D2DRequestInfo], file_path: str, header: List[str]):
        """保存请求列表到CSV文件"""
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)

                for req in requests:
                    writer.writerow(
                        [
                            req.packet_id,
                            req.source_die,
                            req.target_die,
                            req.source_node,
                            req.target_node,
                            req.source_type,
                            req.target_type,
                            req.burst_length,
                            req.start_time_ns,
                            req.end_time_ns,
                            req.latency_ns,
                            req.data_bytes,
                        ]
                    )
        except (IOError, OSError) as e:
            # print(f"[D2D结果处理] 保存CSV文件失败 {file_path}: {e}")
            raise

    def calculate_d2d_bandwidth(self) -> D2DBandwidthStats:
        """计算D2D带宽统计"""
        stats = D2DBandwidthStats()

        # 按方向和类型分组请求
        groups = {("0to1", "read"): [], ("0to1", "write"): [], ("1to0", "read"): [], ("1to0", "write"): []}

        for req in self.d2d_requests:
            direction = "0to1" if req.source_die == 0 else "1to0"
            key = (direction, req.req_type)
            if key in groups:
                groups[key].append(req)

        # 计算各组的带宽
        stats.die0_to_die1_read_bw, stats.die0_to_die1_read_bw_weighted = self._calculate_bandwidth_for_group(groups[("0to1", "read")])
        stats.die0_to_die1_write_bw, stats.die0_to_die1_write_bw_weighted = self._calculate_bandwidth_for_group(groups[("0to1", "write")])
        stats.die1_to_die0_read_bw, stats.die1_to_die0_read_bw_weighted = self._calculate_bandwidth_for_group(groups[("1to0", "read")])
        stats.die1_to_die0_write_bw, stats.die1_to_die0_write_bw_weighted = self._calculate_bandwidth_for_group(groups[("1to0", "write")])

        # 统计总数
        stats.total_read_requests = len(groups[("0to1", "read")]) + len(groups[("1to0", "read")])
        stats.total_write_requests = len(groups[("0to1", "write")]) + len(groups[("1to0", "write")])
        stats.total_bytes_transferred = sum(req.data_bytes for req in self.d2d_requests)

        self.d2d_stats = stats
        return stats

    def _calculate_bandwidth_for_group(self, requests: List[D2DRequestInfo]) -> Tuple[float, float]:
        """计算一组请求的带宽（非加权和加权）"""
        if not requests:
            return 0.0, 0.0

        # 计算总时间和总字节数
        if len(requests) == 1:
            # 单个请求，使用其延迟
            total_time_ns = max(requests[0].latency_ns, 1)  # 避免除零
        else:
            # 多个请求，计算整体时间跨度
            start_time = min(req.start_time_ns for req in requests)
            end_time = max(req.end_time_ns for req in requests)
            total_time_ns = max(end_time - start_time, 1)

        total_bytes = sum(req.data_bytes for req in requests)

        # 非加权带宽 (GB/s)
        unweighted_bw = (total_bytes / total_time_ns) if total_time_ns > 0 else 0.0

        # 加权带宽计算：使用工作区间方法
        working_intervals = self.calculate_d2d_working_intervals(requests)

        if working_intervals:
            total_weighted_bw = 0.0
            total_weight = 0

            for interval in working_intervals:
                weight = interval.flit_count  # 权重是区间的flit数量
                bandwidth = interval.bandwidth_bytes_per_ns  # 区间带宽 (bytes/ns)
                total_weighted_bw += bandwidth * weight
                total_weight += weight

            weighted_bw = (total_weighted_bw / total_weight) if total_weight > 0 else unweighted_bw

            # 调试信息：显示工作区间统计
            # print(f"[D2D带宽] 工作区间数量: {len(working_intervals)}, 总权重: {total_weight}")
        else:
            weighted_bw = unweighted_bw
            # print(f"[D2D带宽] 无有效工作区间，使用非加权带宽")

        return unweighted_bw, weighted_bw

    def calculate_d2d_working_intervals(self, requests: List[D2DRequestInfo]) -> List[WorkingInterval]:
        """
        计算D2D请求的工作区间

        Args:
            requests: D2D请求列表

        Returns:
            工作区间列表
        """
        if not requests:
            return []

        # 构建时间轴事件
        events = []
        for req in requests:
            # 使用D2DRequestInfo的字段
            events.append((req.start_time_ns, "start", req))
            events.append((req.end_time_ns, "end", req))
        events.sort(key=lambda x: (x[0], x[1]))  # 按时间排序，相同时间时'end'在'start'前面

        # 识别连续工作时段
        active_requests = set()
        raw_intervals = []
        current_start = None

        for time_point, event_type, req in events:
            # 检查时间点是否有效
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
            last_end = max(req.end_time_ns for req in requests)
            raw_intervals.append((current_start, last_end))

        # 合并相近区间（复用基类方法）
        merged_intervals = self._merge_close_intervals(raw_intervals)

        # 构建WorkingInterval对象
        working_intervals = []
        for start, end in merged_intervals:
            # 找到该区间内的所有D2D请求
            interval_requests = [req for req in requests if req.start_time_ns < end and req.end_time_ns > start]

            if interval_requests:
                total_bytes = sum(req.data_bytes for req in interval_requests)
                flit_count = sum(req.burst_length for req in interval_requests)

                interval = WorkingInterval(start_time=start, end_time=end, duration=end - start, flit_count=flit_count, total_bytes=total_bytes, request_count=len(interval_requests))
                working_intervals.append(interval)

        return working_intervals

    def generate_d2d_bandwidth_report(self, output_path: str):
        """生成D2D带宽报告，打印到屏幕并保存到txt文件"""
        stats = self.calculate_d2d_bandwidth()

        # 生成报告内容
        report_lines = [
            "=" * 50,
            "D2D带宽统计报告",
            "=" * 50,
            "",
            "Die0 → Die1:",
            f"  读带宽: {stats.die0_to_die1_read_bw:.2f} GB/s (加权: {stats.die0_to_die1_read_bw_weighted:.2f} GB/s)",
            f"  写带宽: {stats.die0_to_die1_write_bw:.2f} GB/s (加权: {stats.die0_to_die1_write_bw_weighted:.2f} GB/s)",
            "",
            "Die1 → Die0:",
            f"  读带宽: {stats.die1_to_die0_read_bw:.2f} GB/s (加权: {stats.die1_to_die0_read_bw_weighted:.2f} GB/s)",
            f"  写带宽: {stats.die1_to_die0_write_bw:.2f} GB/s (加权: {stats.die1_to_die0_write_bw_weighted:.2f} GB/s)",
            "",
            "总计:",
            f"  跨Die总带宽: {stats.die0_to_die1_read_bw + stats.die0_to_die1_write_bw + stats.die1_to_die0_read_bw + stats.die1_to_die0_write_bw:.2f} GB/s",
            f"  跨Die加权总带宽: {stats.die0_to_die1_read_bw_weighted + stats.die0_to_die1_write_bw_weighted + stats.die1_to_die0_read_bw_weighted + stats.die1_to_die0_write_bw_weighted:.2f} GB/s",
            f"  读请求数: {stats.total_read_requests}",
            f"  写请求数: {stats.total_write_requests}",
            f"  总传输字节数: {stats.total_bytes_transferred:,} bytes",
            "=" * 50,
        ]

        # 打印到屏幕
        for line in report_lines:
            print(line)

        # 保存到文件
        os.makedirs(output_path, exist_ok=True)
        report_file = os.path.join(output_path, "d2d_bandwidth_summary.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            for line in report_lines:
                f.write(line + "\n")

        # print(f"\n[D2D结果处理] 带宽报告已保存到 {report_file}")

    def process_d2d_results(self, dies: Dict, output_path: str):
        """
        完整的D2D结果处理流程

        Args:
            dies: Die模型字典
            output_path: 输出目录路径
        """
        # print("\n[D2D结果处理] 开始处理D2D系统结果...")

        # 1. 收集跨Die请求数据
        self.collect_cross_die_requests(dies)

        # 2. 计算D2D节点IP带宽统计
        self.calculate_d2d_ip_bandwidth_data(dies)

        # 3. 保存请求到CSV文件
        self.save_d2d_requests_csv(output_path)

        # 4. 计算并输出带宽报告
        self.generate_d2d_bandwidth_report(output_path)

        # 5. 计算D2D_Sys AXI通道带宽统计
        d2d_bandwidth = self._calculate_d2d_sys_bandwidth(dies)

        # 6. 保存AXI通道统计到文件
        self.save_d2d_axi_channel_statistics(output_path, d2d_bandwidth, dies, self.config)

        # print("[D2D结果处理] D2D结果处理完成!")

    def calculate_d2d_ip_bandwidth_data(self, dies: Dict):
        """
        基于D2D请求计算IP带宽数据

        Args:
            dies: Die模型字典
        """
        # 初始化每个Die独立的ip_bandwidth_data结构
        rows = self.config.NUM_ROW
        cols = self.config.NUM_COL
        # D2D系统使用5x4拓扑，调整行数
        if getattr(self, "topo_type_stat", "5x4") != "4x5":
            rows -= 1

        # 为每个Die创建独立的IP带宽数据结构
        self.die_ip_bandwidth_data = {}

        for die_id in dies.keys():
            self.die_ip_bandwidth_data[die_id] = {
                "read": {
                    "sdma": np.zeros((rows, cols)),
                    "gdma": np.zeros((rows, cols)),
                    "cdma": np.zeros((rows, cols)),
                    "ddr": np.zeros((rows, cols)),
                    "l2m": np.zeros((rows, cols)),
                    "d2d_rn": np.zeros((rows, cols)),
                    "d2d_sn": np.zeros((rows, cols)),
                },
                "write": {
                    "sdma": np.zeros((rows, cols)),
                    "gdma": np.zeros((rows, cols)),
                    "cdma": np.zeros((rows, cols)),
                    "ddr": np.zeros((rows, cols)),
                    "l2m": np.zeros((rows, cols)),
                    "d2d_rn": np.zeros((rows, cols)),
                    "d2d_sn": np.zeros((rows, cols)),
                },
                "total": {
                    "sdma": np.zeros((rows, cols)),
                    "gdma": np.zeros((rows, cols)),
                    "cdma": np.zeros((rows, cols)),
                    "ddr": np.zeros((rows, cols)),
                    "l2m": np.zeros((rows, cols)),
                    "d2d_rn": np.zeros((rows, cols)),
                    "d2d_sn": np.zeros((rows, cols)),
                },
            }

        # 基于D2D请求计算带宽
        self._calculate_bandwidth_from_d2d_requests(dies)

        # print(f"[D2D结果处理] D2D IP带宽统计完成")

    def _calculate_bandwidth_from_d2d_requests(self, dies: Dict):
        """基于D2D请求计算各Die的IP带宽"""
        # print(f"[D2D调试] 开始基于{len(self.d2d_requests)}个D2D请求计算IP带宽")

        # 计算仿真总时间（纳秒）
        if self.d2d_requests:
            max_end_time = max(req.end_time_ns for req in self.d2d_requests)
            sim_time_ns = max(max_end_time, 1)  # 避免除0
        else:
            sim_time_ns = 1

        # 遍历每个D2D请求
        for request in self.d2d_requests:
            # 对于源Die：记录source_type的发送带宽
            if request.source_die in dies:
                self._record_source_bandwidth(request, dies[request.source_die], sim_time_ns)

            # 对于目标Die：记录target_type的接收带宽
            if request.target_die in dies:
                self._record_target_bandwidth(request, dies[request.target_die], sim_time_ns)

        # print(f"[D2D调试] D2D请求带宽计算完成")

    def _record_source_bandwidth(self, request: "D2DRequestInfo", die_model, sim_time_ns: int):
        """记录源IP的发送带宽"""
        try:
            # 标准化IP类型
            source_type_normalized = self._normalize_d2d_ip_type(request.source_type)

            # 计算物理位置
            row, col = self._get_physical_position(request.source_node, die_model)

            # 计算带宽 (GB/s)
            bandwidth_gbps = (request.data_bytes * 1e9) / (sim_time_ns * 1e9)  # GB/s

            # 记录到对应Die的数据结构
            die_data = self.die_ip_bandwidth_data[request.source_die]
            if source_type_normalized in die_data[request.req_type]:
                die_data[request.req_type][source_type_normalized][row, col] += bandwidth_gbps
                die_data["total"][source_type_normalized][row, col] += bandwidth_gbps

            # print(f"[D2D调试] 源IP带宽: Die{request.source_die} {source_type_normalized}({row},{col}) {request.req_type} = {bandwidth_gbps:.3f} GB/s")

        except Exception as e:
            # print(f"[D2D调试] 记录源带宽失败: {e}")
            pass

    def _record_target_bandwidth(self, request: "D2DRequestInfo", die_model, sim_time_ns: int):
        """记录目标IP的接收带宽"""
        try:
            # 标准化IP类型
            target_type_normalized = self._normalize_d2d_ip_type(request.target_type)

            # 计算物理位置
            row, col = self._get_physical_position(request.target_node, die_model)

            # 计算带宽 (GB/s)
            bandwidth_gbps = (request.data_bytes * 1e9) / (sim_time_ns * 1e9)  # GB/s

            # 记录到对应Die的数据结构
            die_data = self.die_ip_bandwidth_data[request.target_die]
            if target_type_normalized in die_data[request.req_type]:
                die_data[request.req_type][target_type_normalized][row, col] += bandwidth_gbps
                die_data["total"][target_type_normalized][row, col] += bandwidth_gbps

            # print(f"[D2D调试] 目标IP带宽: Die{request.target_die} {target_type_normalized}({row},{col}) {request.req_type} = {bandwidth_gbps:.3f} GB/s")

        except Exception as e:
            # print(f"[D2D调试] 记录目标带宽失败: {e}")
            pass

    def _normalize_d2d_ip_type(self, ip_type: str) -> str:
        """标准化D2D IP类型"""
        if not ip_type:
            return "l2m"

        # 转换为小写并去除数字后缀
        ip_type = ip_type.lower()
        if "_" in ip_type:
            base_type = ip_type.split("_")[0]
        else:
            base_type = ip_type

        # 支持的类型
        supported_types = ["sdma", "gdma", "cdma", "ddr", "l2m", "d2d_rn", "d2d_sn"]
        if base_type in supported_types:
            return base_type
        else:
            return "l2m"

    def _get_physical_position(self, node_id: int, die_model) -> tuple:
        """获取节点的物理位置(row, col)"""
        cols = self.config.NUM_COL

        # 简单映射：node_id直接映射到物理位置
        row = node_id // cols
        col = node_id % cols

        # 如果是奇数行，调整到偶数行显示（因为奇偶数行是同一个物理节点）
        if row % 2 == 1:
            row = row - 1  # 奇数行变成偶数行

        return row, col

    def draw_d2d_flow_graph(self, die_networks=None, dies=None, config=None, mode="utilization", node_size=2000, save_path=None, show_cdma=True):
        """
        绘制D2D双Die流量图，根据D2D_LAYOUT配置动态调整Die排列

        Args:
            die_networks: 字典 {die_id: network_object}，包含两个Die的网络对象（兼容旧调用）
            dies: 字典 {die_id: die_model}，包含两个Die的模型对象（推荐使用）
            config: D2D配置对象
            mode: 显示模式，支持 'utilization', 'total', 'ITag_ratio' 等
            node_size: 节点大小
            save_path: 图片保存路径
            show_cdma: 是否显示CDMA
        """

        # 兼容旧的调用方式
        if die_networks is not None and dies is None:
            dies = {}
            # 尝试从die_networks中提取die模型
            for die_id, network in die_networks.items():
                if hasattr(network, "die_model"):
                    dies[die_id] = network.die_model
                elif hasattr(network, "_die_model"):
                    dies[die_id] = network._die_model

            # 如果无法提取，使用网络对象作为替代
            if not dies:
                die_networks_for_draw = die_networks
            else:
                die_networks_for_draw = {die_id: die_model.data_network for die_id, die_model in dies.items()}
        else:
            # 新的调用方式：直接传入die模型
            die_networks_for_draw = {die_id: die_model.data_network for die_id, die_model in dies.items()}

        # 获取D2D布局配置
        d2d_layout = getattr(config, "D2D_LAYOUT", "HORIZONTAL").upper()

        # 根据布局设置画布大小和Die偏移
        die_width = 16
        die_height = 10

        if d2d_layout == "HORIZONTAL":
            die_spacing_x = die_width + 10  # 水平间距：Die宽度+额外间隙
            die_spacing_y = 0
            die_offsets = {0: (0, 0), 1: (die_spacing_x, die_spacing_y)}  # Die0在左  # Die1在右
            figsize = (24, 14)  # 增大图像尺寸以提高清晰度
        else:  # VERTICAL
            die_spacing_x = 0
            die_spacing_y = -(die_height + 6)  # 垂直间距：Die高度+额外间隙
            die_offsets = {0: (0, -1.5), 1: (die_spacing_x, die_spacing_y)}  # Die0在上  # Die1在下
            figsize = (10, 20)  # 增大图像尺寸以提高清晰度

        # 创建画布
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        # 为每个Die绘制流量图并收集节点位置
        die_node_positions = {}
        for die_id, network in die_networks_for_draw.items():
            offset_x, offset_y = die_offsets[die_id]

            # 绘制单个Die的流量图并获取节点位置
            node_positions = self._draw_single_die_flow(ax, network, config, die_id, offset_x, offset_y, mode, node_size, show_cdma)
            die_node_positions[die_id] = node_positions

        # 绘制跨Die数据带宽连接
        try:
            if dies:
                # print(f"[D2D流量图] 获取到{len(dies)}个Die模型")
                # 计算D2D_Sys带宽统计
                d2d_bandwidth = self._calculate_d2d_sys_bandwidth(dies)
                # print(f"[D2D流量图] 计算得到的D2D带宽统计: {d2d_bandwidth}")
                # 绘制跨Die连接，传入实际节点位置
                self._draw_cross_die_connections(ax, d2d_bandwidth, die_node_positions, config)
            # else:
            # print("[D2D流量图] 警告：无法获取die模型，跳过跨Die连接绘制")
        except Exception as e:
            # print(f"[D2D流量图] 绘制跨Die连接时出错: {e}")
            import traceback

            traceback.print_exc()

        # 设置图表标题和坐标轴 - 缩小标题字体并调整位置
        title = f"D2D Flow Graph"
        ax.set_title(title, fontsize=14, fontweight="bold", y=0.96)  # 降低标题位置避免被裁剪

        # 自动调整坐标轴范围以确保所有内容都显示
        ax.axis("equal")  # 保持纵横比
        ax.margins(0.05)  # 恢复边距以确保内容显示完整
        ax.axis("off")  # 隐藏坐标轴

        # 保存或显示图片
        if save_path:
            plt.tight_layout(pad=0.3)  # 减少边距以节省空间
            plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
            plt.close()
        else:
            plt.tight_layout(pad=0.3)  # 减少边距以节省空间
            plt.show()

    def _draw_single_die_flow(self, ax, network, config, die_id, offset_x, offset_y, mode="utilization", node_size=2000, show_cdma=True):
        """绘制单个Die的流量图，复用原有draw_flow_graph的核心逻辑"""

        # 创建NetworkX图
        G = nx.DiGraph()

        # 获取链路统计数据 - 只使用当前格式
        links = {}
        if hasattr(network, "get_links_utilization_stats") and callable(network.get_links_utilization_stats):
            try:
                utilization_stats = network.get_links_utilization_stats()
                if mode == "utilization":
                    links = {link: stats["utilization"] for link, stats in utilization_stats.items()}
                elif mode == "ITag_ratio":
                    links = {link: stats["ITag_ratio"] for link, stats in utilization_stats.items()}
                elif mode == "total":
                    time_cycles = getattr(self, "simulation_end_cycle", 1000) // config.NETWORK_FREQUENCY
                    links = {}
                    for link, stats in utilization_stats.items():
                        total_flit = stats.get("total_flit", 0)
                        if time_cycles > 0:
                            bandwidth = total_flit * 128 / time_cycles
                            links[link] = bandwidth
                        else:
                            links[link] = 0.0
                else:
                    links = {link: stats["utilization"] for link, stats in utilization_stats.items()}

                # 统计有流量的链路
                active_links = [l for l, v in links.items() if v > 0]
                if len(active_links) > 0:
                    # print(f"[信息] Die {die_id}: 绘制 {len(active_links)} 条有流量的链路")
                    pass
            except Exception as e:
                # print(f"[D2D流量图] Die {die_id}: 获取链路统计数据失败: {e}")
                import traceback

                traceback.print_exc()
                links = {}
        else:
            print(f"[警告] Die {die_id}: 网络对象缺少 get_links_utilization_stats 方法")

        # 获取网络节点
        if hasattr(network, "queues") and network.queues:
            actual_nodes = list(network.queues.keys())
        else:
            # 默认5x4拓扑
            actual_nodes = list(range(config.NUM_ROW * config.NUM_COL))

        # 添加节点到图中
        G.add_nodes_from(actual_nodes)

        # 计算节点位置（应用偏移）
        pos = {}
        for node in actual_nodes:
            x = node % config.NUM_COL
            y = node // config.NUM_COL
            if y % 2 == 1:  # 奇数行左移
                x -= 0.25
                y -= 0.6
            pos[node] = (x * 3 + offset_x, -y * 1.5 + offset_y)  # 调整节点间距，更紧凑

        # 添加有权重的边
        edge_labels = {}
        edge_colors = []
        for (i, j), value in links.items():
            if i not in actual_nodes or j not in actual_nodes:
                continue

            # 过滤自环链路（自己到自己）
            if i == j:
                continue

            G.add_edge(i, j, weight=value)

            # 计算显示值和颜色
            if mode in ["utilization", "T2_ratio", "T1_ratio", "T0_ratio", "ITag_ratio"]:
                display_value = float(value) if value else 0.0
                formatted_label = f"{display_value*100:.1f}%" if display_value > 0 else ""
                color_intensity = display_value
            elif mode == "total":
                display_value = float(value) if value else 0.0
                formatted_label = f"{display_value:.1f}" if display_value > 0 else ""
                color_intensity = min(display_value / 500.0, 1.0)  # 归一化到0-1

                # 为total模式添加使用率信息（参考原始实现）
                if display_value > 0 and hasattr(network, "get_links_utilization_stats"):
                    try:
                        utilization_stats = network.get_links_utilization_stats()
                        if (i, j) in utilization_stats:
                            stats = utilization_stats[(i, j)]

                            # 获取已经计算好的比例数据
                            eject_attempts_h_ratios = stats.get("eject_attempts_h_ratios", {"0": 0.0, "1": 0.0, "2": 0.0, ">2": 0.0})
                            eject_attempts_v_ratios = stats.get("eject_attempts_v_ratios", {"0": 0.0, "1": 0.0, "2": 0.0, ">2": 0.0})

                            # 根据链路方向选择显示哪个方向的0次尝试比例
                            if abs(i - j) == 1:  # 横向链路
                                zero_attempts_ratio = eject_attempts_h_ratios.get("0", 0.0)
                            else:  # 纵向链路
                                zero_attempts_ratio = eject_attempts_v_ratios.get("0", 0.0)

                            empty_ratio = stats.get("empty_ratio", 0.0)
                            # 添加0次尝试比例和空闲比例到标签
                            formatted_label += f"\n{zero_attempts_ratio*100:.0f}% {empty_ratio*100:.0f}%"
                    except:
                        # 如果获取统计数据失败，保持原标签
                        pass
            else:
                display_value = float(value) if value else 0.0
                formatted_label = f"{display_value:.1f}" if display_value > 0 else ""
                color_intensity = min(display_value / 500.0, 1.0)

            if display_value > 0:
                edge_labels[(i, j)] = formatted_label
                edge_colors.append((color_intensity, 0, 0))  # 红色渐变
            else:
                edge_colors.append((0.8, 0.8, 0.8))  # 灰色用于零值

        # 计算节点大小
        square_size = np.sqrt(node_size) / 100

        # 绘制网络边 - 按照原始flow图的方式绘制双向箭头
        for (i, j), color in zip(G.edges(), edge_colors):
            if i not in pos or j not in pos:
                continue

            x1, y1 = pos[i]
            x2, y2 = pos[j]

            if i != j:  # 非自环边
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx * dx + dy * dy)
                if length > 0:
                    dx, dy = dx / length, dy / length

                    # 计算垂直偏移
                    perp_dx, perp_dy = -dy * 0.1, dx * 0.1

                    # 检查是否有反向边
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
                        mutation_scale=6,  # 较小的箭头
                        color=color,
                        zorder=1,
                        linewidth=1,
                    )
                    ax.add_patch(arrow)

        # 绘制边标签 - 使用正确的显示方式（无背景框，偏移位置，颜色映射）
        if edge_labels:
            # 计算颜色映射范围
            link_values = [float(links.get((i, j), 0)) for (i, j) in edge_labels.keys()]
            link_mapping_max = max(link_values) if link_values else 0.0
            link_mapping_min = max(0.6 * link_mapping_max, 100) if mode == "total" else 0.0

            for (i, j), label in edge_labels.items():
                if i in pos and j in pos:
                    # 获取边的值用于颜色映射
                    edge_value = float(links.get((i, j), 0))

                    # 计算颜色强度
                    if edge_value == 0.0:
                        continue  # 跳过零值边

                    if mode == "total":
                        # 带宽模式：使用红色渐变
                        if edge_value <= link_mapping_min:
                            intensity = 0.0
                        else:
                            intensity = (edge_value - link_mapping_min) / (link_mapping_max - link_mapping_min)
                        intensity = min(max(intensity, 0.0), 1.0)
                        color = (intensity, 0, 0)  # 红色渐变
                    else:
                        # 利用率模式：同样使用红色渐变
                        color = (edge_value, 0, 0)  # 直接使用利用率作为红色强度

                    x1, y1 = pos[i]
                    x2, y2 = pos[j]
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    dx, dy = x2 - x1, y2 - y1

                    # 检查是否有反向边
                    has_reverse = G.has_edge(j, i)
                    is_horizontal = abs(dx) > abs(dy)

                    # 计算标签位置偏移（参考原始实现）
                    if has_reverse:
                        if is_horizontal:
                            perp_dx, perp_dy = -dy * 0.15 + 0.25, dx * 0.15
                        else:
                            perp_dx, perp_dy = -dy * 0.23, dx * 0.23 - 0.35
                        label_x = mid_x + perp_dx
                        label_y = mid_y + perp_dy
                    else:
                        if is_horizontal:
                            label_x = mid_x + dx * 0.15
                            label_y = mid_y + dy * 0.15
                        else:
                            label_x = mid_x + (-dy * 0.15 if dx > 0 else dy * 0.15)
                            label_y = mid_y - 0.15

                    # 绘制标签文本（无背景框）- 支持多行显示
                    # 检查是否为多行标签（包含比例信息）
                    label_str = str(label)
                    if "\n" in label_str:
                        # 分别绘制带宽和比例（参考原始实现）
                        lines = label_str.split("\n")
                        bandwidth_text = lines[0]  # 带宽值
                        ratio_text = lines[1] if len(lines) > 1 else ""  # 比例信息

                        # 绘制带宽文本（较大字体）
                        ax.text(
                            label_x,
                            label_y + 0.12,  # 向上偏移
                            bandwidth_text,
                            ha="center",
                            va="center",
                            fontsize=8,  # 带宽用较大字体
                            fontweight="normal",
                            color=color,
                        )

                        # 绘制比例文本（较小字体）
                        if ratio_text:
                            ax.text(
                                label_x,
                                label_y - 0.2,  # 向下偏移
                                ratio_text,
                                ha="center",
                                va="center",
                                fontsize=6,  # 比例用较小字体
                                fontweight="normal",
                                color=color,
                            )
                    else:
                        # 单行标签
                        ax.text(label_x, label_y, label, ha="center", va="center", fontsize=8, fontweight="normal", color=color)

        # 绘制方形节点和IP信息
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
            ax.text(x, y, str(node), ha="center", va="center", fontsize=7)  # 减小字体

            # 为D2D流量图添加简化的IP信息显示 - 仅对偶数行节点显示，并区分Die
            physical_row = node // config.NUM_COL
            if physical_row % 2 == 0:
                self._draw_d2d_ip_info_box(ax, x, y, node, config, mode, square_size, die_id)

        # 添加Die标签 - 根据布局调整位置，去掉黄色框
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            die_center_x = (min(xs) + max(xs)) / 2
            die_center_y = (min(ys) + max(ys)) / 2

            # 根据Die ID和布局确定标签位置
            # 获取D2D布局配置
            d2d_layout = getattr(config, "D2D_LAYOUT", "HORIZONTAL").upper()

            if d2d_layout == "HORIZONTAL":
                # 水平布局：标签放在上方，增加间距避免重叠
                label_x = die_center_x
                label_y = max(ys) + 2.5
            else:
                # 垂直布局：标签放在左边，增加间距避免重叠
                label_x = min(xs) - 4
                label_y = die_center_y

            ax.text(
                label_x, label_y, f"Die {die_id}", ha="center", va="center", fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7, edgecolor="none")
            )

        # 返回节点位置信息供跨Die连接使用
        return pos

    def _draw_d2d_ip_info_box(self, ax, x, y, node, config, mode, square_size, die_id=None):
        """为D2D流量图绘制简化的IP信息框 - 在一个框内显示所有有流量的IP"""
        # 计算物理位置
        physical_col = node % config.NUM_COL
        physical_row = node // config.NUM_COL

        # 收集该节点所有有流量的IP类型
        active_ips = []

        # 首先尝试从die_processors获取数据
        if hasattr(self, "die_processors") and die_id is not None and die_id in self.die_processors:
            die_processor = self.die_processors[die_id]
            if hasattr(die_processor, "ip_bandwidth_data") and die_processor.ip_bandwidth_data is not None:
                # 检查是否有对应模式的数据
                if mode in die_processor.ip_bandwidth_data:
                    mode_data = die_processor.ip_bandwidth_data[mode]
                    for ip_type, data_matrix in mode_data.items():
                        if physical_row < data_matrix.shape[0] and physical_col < data_matrix.shape[1]:
                            bandwidth = data_matrix[physical_row, physical_col]
                            # 临时降低阈值并添加调试信息
                            if bandwidth > 0.001:  # 降低阈值以便看到更多数据
                                active_ips.append((ip_type.upper(), bandwidth))
                                # if node % 10 == 0:  # 减少调试输出频率
                                # print(f"[IP调试] Die {die_id} Node {node}: {ip_type.upper()} = {bandwidth:.3f}")
                # else:
                # 只为第一个节点打印一次，避免重复输出
                # if node % 20 == 0:
                # print(f"[信息] Die {die_id}: 模式'{mode}'暂无IP数据，可用模式: {list(die_processor.ip_bandwidth_data.keys())}")
            # else:
            # if node % 20 == 0:
            # print(f"[信息] Die {die_id}: die_processor没有ip_bandwidth_data")

        # 如果没有从die_processors获取到数据，尝试从自身的ip_bandwidth_data获取，但要根据Die区分
        if not active_ips and hasattr(self, "ip_bandwidth_data") and self.ip_bandwidth_data is not None:
            # 根据D2D请求过滤，只显示属于该Die的IP带宽
            die_specific_ips = self._get_die_specific_ips(node, die_id)

            for ip_type, data_matrix in self.ip_bandwidth_data.get(mode, {}).items():
                if physical_row < data_matrix.shape[0] and physical_col < data_matrix.shape[1]:
                    bandwidth = data_matrix[physical_row, physical_col]
                    # 只显示属于该Die且有流量的IP
                    if bandwidth > 0.001 and ip_type.lower() in die_specific_ips:
                        active_ips.append((ip_type.upper(), bandwidth))
                        print(f"[IP调试] Die {die_id} Node {node}: {ip_type.upper()} = {bandwidth:.3f}")

        # 始终绘制信息框 - 改为正方形
        # IP信息框位置和大小（正方形）
        ip_size = square_size * 2.0  # 正方形大小
        ip_x = x - square_size - ip_size / 2
        ip_y = y + 0.2

        # 绘制IP信息框
        ip_rect = Rectangle(
            (ip_x - ip_size / 2, ip_y - ip_size / 2),
            width=ip_size,
            height=ip_size,
            facecolor="lightyellow" if active_ips else "lightcyan",
            edgecolor="black",
            linewidth=1,
            zorder=2,
        )
        ax.add_patch(ip_rect)

        # 在框内显示内容
        if active_ips:
            # 有活跃IP时显示流量信息，使用简化名称
            if len(active_ips) <= 3:
                # 少于3个IP时，每行显示一个
                ip_text = "\n".join([f"{self._shorten_ip_name(ip)}: {bw:.1f}" for ip, bw in active_ips])
            else:
                # 多于3个IP时，压缩显示
                ip_text = "\n".join([f"{self._shorten_ip_name(ip)}:{bw:.1f}" for ip, bw in active_ips])

            ax.text(ip_x, ip_y, ip_text, ha="center", va="center", fontsize=6, color="darkblue", fontweight="normal")
        # 没有活跃IP时不显示任何文字，保持空白

    def _get_die_specific_ips(self, node, die_id):
        """根据D2D请求确定该Die该节点的IP类型"""
        if not hasattr(self, "d2d_requests") or not self.d2d_requests:
            return []

        die_specific_ips = []

        # 计算当前绘制节点的物理位置
        cols = self.config.NUM_COL
        node_row = node // cols
        node_col = node % cols
        # 调整到偶数行（与IP带宽计算保持一致）
        if node_row % 2 == 1:
            node_row = node_row - 1

        for request in self.d2d_requests:
            # 计算请求中节点的物理位置
            source_row = request.source_node // cols
            source_col = request.source_node % cols
            if source_row % 2 == 1:
                source_row = source_row - 1

            target_row = request.target_node // cols
            target_col = request.target_node % cols
            if target_row % 2 == 1:
                target_row = target_row - 1

            # 检查源Die和物理位置匹配
            if request.source_die == die_id and source_row == node_row and source_col == node_col:
                ip_type = request.source_type.lower().split("_")[0] if "_" in request.source_type else request.source_type.lower()
                if ip_type not in die_specific_ips:
                    die_specific_ips.append(ip_type)

            # 检查目标Die和物理位置匹配
            if request.target_die == die_id and target_row == node_row and target_col == node_col:
                ip_type = request.target_type.lower().split("_")[0] if "_" in request.target_type else request.target_type.lower()
                if ip_type not in die_specific_ips:
                    die_specific_ips.append(ip_type)

        return die_specific_ips

    def _shorten_ip_name(self, ip_name):
        """将IP类型名称缩短为单字母"""
        ip_name_upper = ip_name.upper()
        if ip_name_upper.startswith("GDMA"):
            return "G"
        elif ip_name_upper.startswith("SDMA"):
            return "S"
        elif ip_name_upper.startswith("CDMA"):
            return "C"
        elif ip_name_upper.startswith("DDR"):
            return "D"
        elif ip_name_upper.startswith("L2M"):
            return "L"
        elif ip_name_upper.startswith("D2D_RN"):
            return "DR"
        elif ip_name_upper.startswith("D2D_SN"):
            return "DS"
        else:
            return ip_name_upper[:2]  # 其他情况取前两个字母

    def _calculate_d2d_sys_bandwidth(self, dies):
        """
        计算每个Die每个D2D节点的AXI通道带宽

        Args:
            dies: Dict[die_id, die_model] - Die模型字典

        Returns:
            dict: D2D_Sys AXI通道带宽统计 {die_id: {node_pos: {channel: bandwidth_gbps}}}
        """
        d2d_sys_bandwidth = {}

        # 使用simulation_end_cycle来计算时间，与其他带宽计算保持一致
        sim_end_cycle = getattr(self, "simulation_end_cycle", 1000)
        network_frequency = getattr(self.config, "NETWORK_FREQUENCY", 2)
        time_cycles = sim_end_cycle // network_frequency

        # print(f"[D2D_Sys带宽] 仿真周期: {sim_end_cycle} cycles, 网络频率: {network_frequency}, 时间周期: {time_cycles}")

        for die_id, die_model in dies.items():
            d2d_sys_bandwidth[die_id] = {}

            # 从该Die的所有d2d_systems分别计算每个节点的带宽
            if hasattr(die_model, "d2d_systems"):
                # print(f"[D2D_Sys调试] Die{die_id}有{len(die_model.d2d_systems)}个D2D系统")

                for pos, d2d_sys in die_model.d2d_systems.items():
                    # 为每个节点单独计算带宽
                    node_bandwidth = {"AR": 0.0, "R": 0.0, "AW": 0.0, "W": 0.0, "B": 0.0}  # 读地址通道  # 读数据通道  # 写地址通道  # 写数据通道  # 写响应通道

                    if hasattr(d2d_sys, "axi_channel_flit_count"):
                        # print(f"[D2D_Sys调试] Die{die_id}位置{pos}的AXI通道flit统计: {d2d_sys.axi_channel_flit_count}")

                        # 计算该节点各通道的带宽，与其他地方保持一致
                        for channel, flit_count in d2d_sys.axi_channel_flit_count.items():
                            if channel in node_bandwidth and flit_count > 0 and time_cycles > 0:
                                bandwidth = flit_count * 128 / time_cycles  # 与第652行的计算方式一致
                                node_bandwidth[channel] = bandwidth

                                # print(f"[D2D_Sys带宽] Die{die_id}节点{pos} {channel}通道: {flit_count} flits -> {bandwidth:.3f} 单位")
                    # else:
                    # print(f"[D2D_Sys调试] Die{die_id}位置{pos}的D2D_Sys没有axi_channel_flit_count属性")

                    d2d_sys_bandwidth[die_id][pos] = node_bandwidth

        return d2d_sys_bandwidth

    def _draw_cross_die_connections(self, ax, d2d_bandwidth, die_node_positions, config):
        """
        绘制跨Die数据带宽连接（只显示R和W通道的数据流）
        基于实际流量数据和实际节点位置确定连接关系

        Args:
            ax: matplotlib轴对象
            d2d_bandwidth: D2D_Sys带宽统计 {die_id: {node_pos: {channel: bandwidth}}}
            die_node_positions: 实际的Die节点位置 {die_id: {node: (x, y)}}
            config: 配置对象
        """
        try:
            # 获取D2D节点位置配置
            die0_positions = getattr(config, "D2D_DIE0_POSITIONS", [36, 37, 38, 39])
            die1_positions = getattr(config, "D2D_DIE1_POSITIONS", [4, 5, 6, 7])

            if not die0_positions or not die1_positions:
                print("[D2D连接] 警告：D2D节点位置配置缺失")
                return

            # 收集所有有流量的连接
            active_connections = []

            # 检查Die0的写数据流量 (W通道) - 表示从Die0发送到Die1
            for die0_node in die0_positions:
                w_bw = d2d_bandwidth.get(0, {}).get(die0_node, {}).get("W", 0.0)
                if w_bw > 0.001:
                    # print(f"[D2D连接分析] Die0节点{die0_node}有写数据流量: {w_bw:.3f} GB/s")
                    # 找到对应的Die1目标节点（从D2D请求中推断）
                    target_die1_node = self._find_target_node_for_write(die0_node, die1_positions)
                    if target_die1_node:
                        active_connections.append({"type": "write", "from_die": 0, "from_node": die0_node, "to_die": 1, "to_node": target_die1_node, "bandwidth": w_bw})

            # 检查Die1的读数据返回流量 (R通道) - 表示从Die1返回到Die0
            for die1_node in die1_positions:
                r_bw = d2d_bandwidth.get(1, {}).get(die1_node, {}).get("R", 0.0)
                if r_bw > 0.001:
                    # print(f"[D2D连接分析] Die1节点{die1_node}有读数据返回流量: {r_bw:.3f} GB/s")
                    # 找到对应的Die0目标节点（从D2D请求中推断）
                    target_die0_node = self._find_target_node_for_read(die1_node, die0_positions)
                    if target_die0_node:
                        active_connections.append({"type": "read", "from_die": 1, "from_node": die1_node, "to_die": 0, "to_node": target_die0_node, "bandwidth": r_bw})

            # 绘制所有活跃连接
            for i, conn in enumerate(active_connections):
                # 使用实际节点位置
                from_die_positions = die_node_positions.get(conn["from_die"], {})
                to_die_positions = die_node_positions.get(conn["to_die"], {})

                if conn["from_node"] not in from_die_positions or conn["to_node"] not in to_die_positions:
                    print(f"[D2D连接] 警告：找不到节点位置 - From: Die{conn['from_die']}节点{conn['from_node']}, To: Die{conn['to_die']}节点{conn['to_node']}")
                    continue

                from_x, from_y = from_die_positions[conn["from_node"] - config.NUM_COL]
                to_x, to_y = to_die_positions[conn["to_node"] - config.NUM_COL]

                # print(f"[D2D连接] 绘制{conn['type']}连接: Die{conn['from_die']}节点{conn['from_node']} -> Die{conn['to_die']}节点{conn['to_node']}, 带宽={conn['bandwidth']:.3f} GB/s")

                # 计算箭头方向
                dx, dy = to_x - from_x, to_y - from_y
                length = np.sqrt(dx * dx + dy * dy)

                if length > 0:
                    ux, uy = dx / length, dy / length
                    perpx, perpy = -uy * 0.2, ux * 0.2

                    self._draw_single_d2d_arrow(ax, from_x, from_y, to_x, to_y, ux, uy, perpx, perpy, conn["bandwidth"], conn["type"], i)

            # 为没有流量的D2D节点对绘制灰色连接线（显示潜在连接）
            self._draw_inactive_d2d_connections(ax, die0_positions, die1_positions, active_connections, die_node_positions, config)

        except Exception as e:
            print(f"[D2D连接] 绘制跨Die连接失败: {e}")
            import traceback

            traceback.print_exc()

    def _find_target_node_for_write(self, die0_node, die1_positions):
        """根据D2D请求推断写数据的目标节点"""
        # 简化逻辑：按索引对应
        die0_positions = getattr(self.config, "D2D_DIE0_POSITIONS", [36, 37, 38, 39])
        try:
            index = die0_positions.index(die0_node)
            if index < len(die1_positions):
                return die1_positions[index]
        except (ValueError, IndexError):
            pass
        return die1_positions[0] if die1_positions else None

    def _find_target_node_for_read(self, die1_node, die0_positions):
        """根据D2D请求推断读数据返回的目标节点"""
        # 简化逻辑：按索引对应
        die1_positions = getattr(self.config, "D2D_DIE1_POSITIONS", [4, 5, 6, 7])
        try:
            index = die1_positions.index(die1_node)
            if index < len(die0_positions):
                return die0_positions[index]
        except (ValueError, IndexError):
            pass
        return die0_positions[0] if die0_positions else None

    def _draw_inactive_d2d_connections(self, ax, die0_positions, die1_positions, active_connections, die_node_positions, config):
        """为没有流量的节点对绘制灰色连接线"""
        # 获取已经绘制的活跃连接（只记录实际有流量的方向）
        active_pairs = set()
        for conn in active_connections:
            active_pairs.add((conn["from_node"], conn["to_node"]))

        # 为每对D2D节点绘制缺失方向的灰色连接
        num_pairs = min(len(die0_positions), len(die1_positions))
        for i in range(num_pairs):
            die0_node = die0_positions[i]
            # 根据垂直对应关系找到目标节点
            target_die1_node = self._find_target_node_for_write(die0_node, die1_positions)
            if not target_die1_node:
                continue

            # 使用实际节点位置
            die0_positions_map = die_node_positions.get(0, {})
            die1_positions_map = die_node_positions.get(1, {})

            if die0_node not in die0_positions_map or target_die1_node not in die1_positions_map:
                continue

            die0_x, die0_y = die0_positions_map[die0_node - config.NUM_COL]
            die1_x, die1_y = die1_positions_map[target_die1_node - config.NUM_COL]

            dx, dy = die1_x - die0_x, die1_y - die0_y
            length = np.sqrt(dx * dx + dy * dy)

            if length > 0:
                ux, uy = dx / length, dy / length
                perpx, perpy = -uy * 0.1, ux * 0.1

                # 检查两个方向是否有活跃连接
                has_write_connection = (die0_node, target_die1_node) in active_pairs
                has_read_connection = (target_die1_node, die0_node) in active_pairs

                # 如果写方向没有活跃连接，绘制写方向的灰色箭头
                if not has_write_connection:
                    self._draw_single_d2d_arrow(ax, die0_x, die0_y, die1_x, die1_y, ux, uy, perpx, perpy, 0.0, "write", f"inactive_{i}_w")

                # 如果读方向没有活跃连接，绘制读方向的灰色箭头
                if not has_read_connection:
                    self._draw_single_d2d_arrow(ax, die1_x, die1_y, die0_x, die0_y, -ux, -uy, -perpx, -perpy, 0.0, "read", f"inactive_{i}_r")

    def _draw_single_d2d_arrow(self, ax, start_node_x, start_node_y, end_node_x, end_node_y, ux, uy, perpx, perpy, bandwidth, arrow_type, connection_index):
        """
        绘制单个D2D箭头

        Args:
            ax: matplotlib轴对象
            start_node_x, start_node_y: 起始节点坐标
            end_node_x, end_node_y: 结束节点坐标
            ux, uy: 单位方向向量
            perpx, perpy: 垂直方向向量
            bandwidth: 带宽值
            arrow_type: 箭头类型 ("write" 或 "read")
            connection_index: 连接索引（用于调试）
        """
        # 计算箭头起止坐标（留出节点空间）
        start_x = start_node_x + ux * 0.5 + perpx
        start_y = start_node_y + uy * 0.5 + perpy
        end_x = end_node_x - ux * 0.5 + perpx
        end_y = end_node_y - uy * 0.5 + perpy

        # 确定颜色和标签
        if bandwidth > 0.001:
            # 有数据流量
            intensity = min(bandwidth / self.MAX_BANDWIDTH_NORMALIZATION, 1.0)
            color = (intensity, 0, 0)  # 红色
            label_text = f"{bandwidth:.1f}"  # 只显示数值，不加GB/s后缀
            linewidth = 3
        else:
            # 无数据流量 - 灰色实线
            color = (0.7, 0.7, 0.7)
            label_text = None  # 不显示0
            linewidth = 2

        # 绘制箭头
        arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle="-|>", mutation_scale=10, color=color, linewidth=linewidth, zorder=5)  # 稍小的箭头
        ax.add_patch(arrow)

        # 只在有流量时添加标签，参考Die内部链路的标记方式
        if label_text:
            # 计算箭头中点和方向
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            dx = end_x - start_x
            dy = end_y - start_y

            # 判断是否为水平方向（跨Die连接通常是垂直的）
            is_horizontal = abs(dx) > abs(dy)

            # 根据方向计算标签偏移，参考Die内部链路的逻辑
            if is_horizontal:
                # 水平方向：标签稍微向上/下偏移
                label_x = mid_x + dx * 0.15
                label_y = mid_y + dy * 0.15
            else:
                # 垂直方向：标签向左/右偏移
                label_x = mid_x + (dy * 0.3 if dx > 0 else -dy * 0.3)
                label_y = mid_y - 0.15

            # 绘制单个标签（与Die内部链路一致）
            ax.text(label_x, label_y, label_text, ha="center", va="center", fontsize=8, fontweight="normal", color=color)

            # print(f"[D2D连接] 连接{connection_index}: {arrow_type}箭头, 带宽={bandwidth:.3f} GB/s")

    def save_d2d_axi_channel_statistics(self, output_path, d2d_bandwidth, dies, config):
        """
        保存所有AXI通道的带宽统计到文件

        Args:
            output_path: 输出目录路径
            d2d_bandwidth: D2D_Sys带宽统计
            dies: Die模型字典
            config: 配置对象
        """
        try:
            os.makedirs(output_path, exist_ok=True)

            # 1. 保存详细的AXI通道带宽统计到CSV
            csv_path = os.path.join(output_path, "d2d_axi_channel_bandwidth.csv")

        except (OSError, PermissionError) as e:
            print(f"[D2D AXI统计] 无法创建输出目录: {e}")
            return

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)

                # CSV文件头
                writer.writerow(["Die_ID", "Channel", "Direction", "Bandwidth_GB/s", "Flit_Count", "Channel_Description"])

                # 写入各通道数据 - 适应新的数据结构
                for die_id, node_data in d2d_bandwidth.items():
                    # 遍历该Die的每个D2D节点
                    for node_pos, channels in node_data.items():
                        # 从die模型获取该节点的原始flit计数
                        die_model = dies.get(die_id)
                        flit_counts = {channel: 0 for channel in ["AR", "R", "AW", "W", "B"]}

                        if die_model and hasattr(die_model, "d2d_systems"):
                            d2d_sys = die_model.d2d_systems.get(node_pos)
                            if d2d_sys and hasattr(d2d_sys, "axi_channel_flit_count"):
                                flit_counts = d2d_sys.axi_channel_flit_count.copy()

                        # 写入各通道数据
                        channel_descriptions = {
                            "AR": "读地址通道 (Address Read)",
                            "R": "读数据通道 (Read Data)",
                            "AW": "写地址通道 (Address Write)",
                            "W": "写数据通道 (Write Data)",
                            "B": "写响应通道 (Write Response)",
                        }

                        direction_mapping = {
                            "AR": f"Die{die_id}->Die{1-die_id}",  # 地址通道：发起方->目标方
                            "R": f"Die{1-die_id}->Die{die_id}",  # 读数据：目标方->发起方
                            "AW": f"Die{die_id}->Die{1-die_id}",  # 写地址：发起方->目标方
                            "W": f"Die{die_id}->Die{1-die_id}",  # 写数据：发起方->目标方
                            "B": f"Die{1-die_id}->Die{die_id}",  # 写响应：目标方->发起方
                        }

                        for channel, bandwidth in channels.items():
                            flit_count = flit_counts.get(channel, 0)
                            direction = direction_mapping.get(channel, f"Die{die_id}")
                            description = channel_descriptions.get(channel, f"{channel} Channel")

                            # 添加节点位置信息到CSV
                            writer.writerow([f"Die{die_id}_Node{node_pos}", channel, direction, f"{bandwidth:.6f}", flit_count, description])

            print(f"[D2D AXI统计] 已保存AXI通道带宽统计到 {csv_path}")

        except (IOError, PermissionError) as e:
            print(f"[D2D AXI统计] 保存CSV文件失败: {e}")
            return

        # 2. 生成汇总报告
        try:
            summary_path = os.path.join(output_path, "d2d_axi_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("D2D AXI通道带宽统计汇总报告\n")
                f.write("=" * 60 + "\n\n")

                # 各Die的通道统计
                for die_id, node_data in d2d_bandwidth.items():
                    f.write(f"Die {die_id} AXI通道带宽:\n")
                    f.write("-" * 30 + "\n")

                    die_total_bandwidth = 0.0
                    die_data_bandwidth = 0.0  # 数据通道（R+W）

                    for node_pos, channels in node_data.items():
                        f.write(f"  节点{node_pos}:\n")
                        node_total = 0.0
                        node_data_bw = 0.0

                        for channel, bandwidth in channels.items():
                            if bandwidth > 0:
                                f.write(f"    {channel}通道: {bandwidth:.3f} GB/s\n")
                                node_total += bandwidth
                                die_total_bandwidth += bandwidth
                                if channel in ["R", "W"]:
                                    node_data_bw += bandwidth
                                    die_data_bandwidth += bandwidth

                        if node_total > 0:
                            f.write(f"    节点总带宽: {node_total:.3f} GB/s (数据: {node_data_bw:.3f} GB/s)\n")
                        f.write("\n")

                    f.write(f"  Die{die_id}总带宽: {die_total_bandwidth:.3f} GB/s\n")
                    f.write(f"  Die{die_id}数据带宽 (R+W): {die_data_bandwidth:.3f} GB/s\n\n")

                # 跨Die数据流汇总
                f.write("跨Die数据流汇总:\n")
                f.write("-" * 30 + "\n")

                if 0 in d2d_bandwidth and 1 in d2d_bandwidth:
                    # 计算各Die的总数据带宽
                    die0_total_w = sum(channels.get("W", 0.0) for channels in d2d_bandwidth[0].values())
                    die1_total_r = sum(channels.get("R", 0.0) for channels in d2d_bandwidth[1].values())

                    f.write(f"Die0 -> Die1 总写数据带宽: {die0_total_w:.3f} GB/s\n")
                    f.write(f"Die1 -> Die0 总读数据带宽: {die1_total_r:.3f} GB/s\n")
                    f.write(f"跨Die总数据带宽: {die0_total_w + die1_total_r:.3f} GB/s\n\n")

                    # 详细的节点对连接统计
                    f.write("详细节点对连接:\n")
                    d2d_pairs = getattr(config, "D2D_PAIRS", [])
                    
                    if d2d_pairs:
                        for i, (die0_id, die0_node, die1_id, die1_node) in enumerate(d2d_pairs):
                            die0_w = d2d_bandwidth.get(die0_id, {}).get(die0_node, {}).get("W", 0.0)
                            die1_r = d2d_bandwidth.get(die1_id, {}).get(die1_node, {}).get("R", 0.0)
                            if die0_w > 0 or die1_r > 0:
                                f.write(f"  连接{i}: Die{die0_id}节点{die0_node} <-> Die{die1_id}节点{die1_node}\n")
                                f.write(f"    写数据: {die0_w:.3f} GB/s, 读数据: {die1_r:.3f} GB/s\n")
                    else:
                        # 向后兼容
                        die0_positions = getattr(config, "D2D_DIE0_POSITIONS", [])
                        die1_positions = getattr(config, "D2D_DIE1_POSITIONS", [])
                        if die0_positions and die1_positions:
                            for i, (die0_pos, die1_pos) in enumerate(zip(die0_positions, die1_positions)):
                                die0_w = d2d_bandwidth.get(0, {}).get(die0_pos, {}).get("W", 0.0)
                                die1_r = d2d_bandwidth.get(1, {}).get(die1_pos, {}).get("R", 0.0)
                                if die0_w > 0 or die1_r > 0:
                                    f.write(f"  连接{i}: Die0节点{die0_pos} <-> Die1节点{die1_pos}\n")
                                    f.write(f"    写数据: {die0_w:.3f} GB/s, 读数据: {die1_r:.3f} GB/s\n")
                    f.write("\n")

                # 通道利用率分析
                f.write("AXI通道功能说明:\n")
                f.write("-" * 30 + "\n")
                f.write("AR (Address Read): 读地址通道 - 发送读请求地址\n")
                f.write("R  (Read Data):    读数据通道 - 返回读取的数据\n")
                f.write("AW (Address Write): 写地址通道 - 发送写请求地址\n")
                f.write("W  (Write Data):   写数据通道 - 发送写入的数据\n")
                f.write("B  (Write Response): 写响应通道 - 返回写操作完成确认\n\n")

                f.write("注意: 流量图中只显示数据通道(R+W)的带宽，\n")
                f.write("      完整的AXI通道统计请参考CSV文件。\n")

            print(f"[D2D AXI统计] 已保存AXI汇总报告到 {summary_path}")

        except (IOError, PermissionError) as e:
            print(f"[D2D AXI统计] 保存汇总报告失败: {e}")
        except Exception as e:
            print(f"[D2D AXI统计] 生成报告时发生未预期的错误: {e}")
            import traceback

            traceback.print_exc()

    def collect_requests_data(self, sim_model, simulation_end_cycle=None) -> None:
        """
        重写基类方法，增加D2D特殊处理
        修复D2D跨Die请求的original_source_type和original_destination_type问题
        """
        # 调用基类方法收集基本数据
        super().collect_requests_data(sim_model, simulation_end_cycle)

        # print(f"[D2D调试] 开始修复D2D请求的original类型信息，总请求数: {len(self.requests)}")

        # D2D特殊处理：修复丢失的original类型信息
        fixed_count = 0

        for i, request in enumerate(self.requests):
            # 检查是否需要修复
            need_fix = False
            if not hasattr(request, "original_source_type") or not request.original_source_type:
                need_fix = True
            if not hasattr(request, "original_destination_type") or not request.original_destination_type:
                need_fix = True

            if need_fix:
                # 尝试从相关的arrive_flits中找到对应的flit并恢复d2d属性
                fixed = self._fix_request_original_types(request, sim_model)
                if fixed:
                    fixed_count += 1

        # print(f"[D2D调试] 修复了 {fixed_count} 个D2D请求的类型信息")

    def _fix_request_original_types(self, request: RequestInfo, sim_model) -> bool:
        """
        尝试修复单个请求的original类型信息
        通过查找网络中对应的flit来恢复d2d属性
        """
        try:
            # 根据请求的开始和结束时间、节点位置等信息查找对应的flit
            target_networks = []

            # 检查数据网络
            if hasattr(sim_model, "data_network") and hasattr(sim_model.data_network, "arrive_flits"):
                target_networks.append(sim_model.data_network)

            # 检查请求网络
            if hasattr(sim_model, "request_network") and hasattr(sim_model.request_network, "arrive_flits"):
                target_networks.append(sim_model.request_network)

            for network in target_networks:
                for packet_id, flits in network.arrive_flits.items():
                    if not flits:
                        continue

                    first_flit = flits[0]

                    # 匹配条件：时间窗口、节点位置、请求类型
                    if self._is_matching_flit(request, first_flit):
                        # 找到匹配的flit，尝试从d2d属性恢复原始类型
                        if hasattr(first_flit, "d2d_origin_type") and first_flit.d2d_origin_type:
                            request.original_source_type = first_flit.d2d_origin_type

                        if hasattr(first_flit, "d2d_target_type") and first_flit.d2d_target_type:
                            request.original_destination_type = first_flit.d2d_target_type

                        # print(f"[D2D调试] 为请求修复类型: {request.original_source_type} -> {request.original_destination_type}")
                        return True

        except Exception as e:
            print(f"[D2D调试] 修复请求类型失败: {e}")

        return False

    def _is_matching_flit(self, request: RequestInfo, flit) -> bool:
        """检查flit是否与request匹配"""
        try:
            # 检查时间窗口（允许一定误差）
            flit_start_time = getattr(flit, "cmd_entry_noc_from_cake0_cycle", 0) // self.network_frequency
            time_diff = abs(flit_start_time - request.start_time)
            if time_diff > 10:  # 允许10ns误差
                return False

            # 检查请求类型
            if hasattr(flit, "req_type") and request.req_type != flit.req_type:
                return False

            # 检查D2D属性存在性
            if not (hasattr(flit, "d2d_origin_type") or hasattr(flit, "d2d_target_type")):
                return False

            return True

        except Exception:
            return False
