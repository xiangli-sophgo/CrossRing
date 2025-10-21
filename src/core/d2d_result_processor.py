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
    """D2D带宽统计数据结构（支持任意数量的Die间组合）"""

    # 按 (src_die, dst_die) 记录读写带宽 (unweighted, weighted)
    pair_read_bw: Dict[Tuple[int, int], Tuple[float, float]] = None
    pair_write_bw: Dict[Tuple[int, int], Tuple[float, float]] = None

    total_read_requests: int = 0
    total_write_requests: int = 0
    total_bytes_transferred: int = 0


class D2DResultProcessor(BandwidthAnalyzer):
    """D2D系统专用的结果处理器，继承自BandwidthAnalyzer"""

    FLIT_SIZE_BYTES = 128  # 每个flit的字节数
    MAX_BANDWIDTH_NORMALIZATION = 256.0  # 最大带宽归一化值（GB/s）

    # AXI通道常量定义（避免重复定义）
    AXI_CHANNEL_DESCRIPTIONS = {
        "AR": "读地址通道 (Address Read)",
        "R": "读数据通道 (Read Data)",
        "AW": "写地址通道 (Address Write)",
        "W": "写数据通道 (Write Data)",
        "B": "写响应通道 (Write Response)",
    }

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

            # 检查是否为D2D请求（包括Die内和跨Die）
            if not self._is_d2d_request(first_flit):
                continue

            # 只记录请求发起方Die的数据，避免重复记录
            if hasattr(first_flit, "d2d_origin_die") and first_flit.d2d_origin_die != die_id:
                continue

            # 提取D2D信息
            d2d_info = self._extract_d2d_info(first_flit, last_flit, packet_id)
            if d2d_info:
                self.d2d_requests.append(d2d_info)

    def _is_d2d_request(self, flit: Flit) -> bool:
        """检查flit是否为D2D请求（包括Die内和跨Die请求）"""
        return (
            hasattr(flit, "d2d_origin_die") and hasattr(flit, "d2d_target_die") and flit.d2d_origin_die is not None and flit.d2d_target_die is not None
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
            return None
        except Exception as e:
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
        csv_header = [
            "packet_id",
            "source_die",
            "target_die",
            "source_node",
            "target_node",
            "source_type",
            "target_type",
            "burst_length",
            "start_time_ns",
            "end_time_ns",
            "latency_ns",
            "data_bytes",
        ]

        # 只有存在请求时才保存对应的CSV文件
        if read_requests:
            read_csv_path = os.path.join(output_path, "d2d_read_requests.csv")
            self._save_requests_to_csv(read_requests, read_csv_path, csv_header)
            print(f"  - 读请求: {read_csv_path} ({len(read_requests)} 条记录)")

        if write_requests:
            write_csv_path = os.path.join(output_path, "d2d_write_requests.csv")
            self._save_requests_to_csv(write_requests, write_csv_path, csv_header)
            print(f"  - 写请求: {write_csv_path} ({len(write_requests)} 条记录)")

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
            raise

    def save_ip_bandwidth_to_csv(self, output_path: str):
        """
        保存所有Die的IP带宽数据到单个CSV文件（类似ports_bandwidth.csv格式）

        Args:
            output_path: 输出目录路径
        """
        os.makedirs(output_path, exist_ok=True)

        # 使用die_ip_bandwidth_data (D2D专用)
        if not hasattr(self, "die_ip_bandwidth_data") or not self.die_ip_bandwidth_data:
            print("警告: 没有die_ip_bandwidth_data数据，跳过IP带宽CSV导出")
            return

        csv_path = os.path.join(output_path, "ip_bandwidth.csv")

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)

                # 写入CSV头：简洁明了的格式
                writer.writerow(
                    [
                        "ip_instance",  # IP实例名 (如gdma_0, ddr_1)
                        "die_id",  # Die ID
                        "node_id",  # 节点ID
                        "ip_type",  # IP类型 (如gdma, ddr)
                        "read_bandwidth_gbps",  # 读带宽
                        "write_bandwidth_gbps",  # 写带宽
                        "total_bandwidth_gbps",  # 总带宽
                    ]
                )

                # 收集所有数据行
                all_rows = []

                for die_id, die_data in self.die_ip_bandwidth_data.items():
                    # 获取该Die的配置信息
                    num_col = getattr(self.config, "NUM_COL", 4)
                    num_row = getattr(self.config, "NUM_ROW", 5)

                    # 获取三种模式的数据
                    read_data = die_data.get("read", {})
                    write_data = die_data.get("write", {})
                    total_data = die_data.get("total", {})

                    # 收集所有IP实例
                    all_ip_instances = set(read_data.keys()) | set(write_data.keys()) | set(total_data.keys())

                    for ip_instance in all_ip_instances:
                        # 提取IP基本类型
                        if ip_instance.lower().startswith("d2d"):
                            parts = ip_instance.lower().split("_")
                            ip_type = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
                        else:
                            ip_type = ip_instance.split("_")[0]

                        # 获取矩阵
                        read_matrix = read_data.get(ip_instance)
                        write_matrix = write_data.get(ip_instance)
                        total_matrix = total_data.get(ip_instance)

                        # 确定矩阵形状
                        if read_matrix is not None:
                            rows, cols = read_matrix.shape
                        elif write_matrix is not None:
                            rows, cols = write_matrix.shape
                        elif total_matrix is not None:
                            rows, cols = total_matrix.shape
                        else:
                            continue

                        # 遍历矩阵中的所有位置
                        for matrix_row in range(rows):
                            for matrix_col in range(cols):
                                # 获取三种带宽值
                                read_bw = read_matrix[matrix_row, matrix_col] if read_matrix is not None else 0.0
                                write_bw = write_matrix[matrix_row, matrix_col] if write_matrix is not None else 0.0
                                total_bw = total_matrix[matrix_row, matrix_col] if total_matrix is not None else 0.0

                                # 只保存有带宽的数据（任一模式大于阈值）
                                if read_bw > 0.001 or write_bw > 0.001 or total_bw > 0.001:
                                    # 计算节点ID
                                    physical_row = matrix_row * 2  # 偶数行
                                    node_id = physical_row * num_col + matrix_col

                                    all_rows.append([ip_instance, die_id, node_id, ip_type, f"{read_bw:.6f}", f"{write_bw:.6f}", f"{total_bw:.6f}"])  # IP实例名  # Die ID  # 节点ID  # IP类型

                # 排序：先按die_id，再按node_id，最后按ip_instance
                all_rows.sort(key=lambda x: (int(x[1]), int(x[2]), x[0]))

                # 写入所有数据行
                for row in all_rows:
                    writer.writerow(row)

            print(f"  - IP带宽: {csv_path} ({len(all_rows)} 条记录)")

        except (IOError, OSError) as e:
            print(f"警告: 保存IP带宽CSV失败 ({csv_path}): {e}")

    def calculate_d2d_bandwidth(self) -> D2DBandwidthStats:
        """计算D2D带宽统计（支持多Die）"""
        stats = D2DBandwidthStats(pair_read_bw={}, pair_write_bw={}, total_read_requests=0, total_write_requests=0, total_bytes_transferred=0)

        # 按 (src_die, dst_die) + 类型 分组
        grouped: Dict[Tuple[int, int, str], List[D2DRequestInfo]] = {}
        for req in self.d2d_requests:
            if req.source_die is None or req.target_die is None:
                continue
            key = (int(req.source_die), int(req.target_die), req.req_type)
            grouped.setdefault(key, []).append(req)

        # 计算所有组合的带宽
        for (src_die, dst_die, req_type), reqs in grouped.items():
            unweighted, weighted = self._calculate_bandwidth_for_group(reqs)
            if req_type == "read":
                stats.pair_read_bw[(src_die, dst_die)] = (unweighted, weighted)
                stats.total_read_requests += len(reqs)
            elif req_type == "write":
                stats.pair_write_bw[(src_die, dst_die)] = (unweighted, weighted)
                stats.total_write_requests += len(reqs)

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
        else:
            weighted_bw = unweighted_bw

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
        """生成D2D带宽报告（按任意Die组合逐项列出）"""
        stats = self.calculate_d2d_bandwidth()

        # 汇总总带宽
        total_unweighted = 0.0
        total_weighted = 0.0

        # 生成报告内容
        report_lines = [
            "=" * 50,
            "D2D带宽统计报告",
            "=" * 50,
            "",
            "按Die组合的读带宽 (GB/s):",
        ]

        # 读带宽
        for (src, dst), (uw, wt) in sorted(stats.pair_read_bw.items()):
            report_lines.append(f"Die{src} → Die{dst} (Read):  {uw:.2f} (加权: {wt:.2f})")
            total_unweighted += uw
            total_weighted += wt

        report_lines.extend(["", "按Die组合的写带宽 (GB/s):"])

        # 写带宽
        for (src, dst), (uw, wt) in sorted(stats.pair_write_bw.items()):
            report_lines.append(f"Die{src} → Die{dst} (Write): {uw:.2f} (加权: {wt:.2f})")
            total_unweighted += uw
            total_weighted += wt

        report_lines.extend(
            [
                "",
                "总计:",
                f"  跨Die总带宽: {total_unweighted:.2f} GB/s",
                f"  跨Die加权总带宽: {total_weighted:.2f} GB/s",
                f"  读请求数: {stats.total_read_requests}",
                f"  写请求数: {stats.total_write_requests}",
                f"  总传输字节数: {stats.total_bytes_transferred:,} bytes",
                "=" * 50,
            ]
        )

        # 打印到屏幕
        for line in report_lines:
            print(line)

        # 保存到文件
        os.makedirs(output_path, exist_ok=True)
        report_file = os.path.join(output_path, "d2d_bandwidth_summary.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            for line in report_lines:
                f.write(line + "\n")

        print(f"\n报告已保存: {report_file}")

    def process_d2d_results(self, dies: Dict, output_path: str):
        """
        完整的D2D结果处理流程

        Args:
            dies: Die模型字典
            output_path: 输出目录路径
        """

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

    def calculate_d2d_ip_bandwidth_data(self, dies: Dict):
        """
        基于D2D请求计算IP带宽数据 - 动态收集IP实例

        Args:
            dies: Die模型字典
        """
        # 第一步：收集所有IP实例名称（按Die分组）
        die_ip_instances = {}
        for die_id in dies.keys():
            die_ip_instances[die_id] = set()

        for request in self.d2d_requests:
            # 收集source_type
            if request.source_die in dies and request.source_type:
                raw_source = request.source_type
                if raw_source.endswith("_ip"):
                    raw_source = raw_source[:-3]
                source_type = self._normalize_d2d_ip_type(raw_source)
                die_ip_instances[request.source_die].add(source_type)

            # 收集target_type
            if request.target_die in dies and request.target_type:
                raw_target = request.target_type
                if raw_target.endswith("_ip"):
                    raw_target = raw_target[:-3]
                target_type = self._normalize_d2d_ip_type(raw_target)
                die_ip_instances[request.target_die].add(target_type)

        # 第二步：为每个Die动态创建IP带宽数据结构
        self.die_ip_bandwidth_data = {}

        for die_id, die_model in dies.items():
            # 从每个Die的config获取其拓扑配置
            rows = die_model.config.NUM_ROW
            cols = die_model.config.NUM_COL
            # D2D系统使用5x4拓扑，调整行数
            if getattr(self, "topo_type_stat", "5x4") != "4x5":
                rows -= 1

            # 为该Die的每个IP实例创建矩阵
            self.die_ip_bandwidth_data[die_id] = {
                "read": {},
                "write": {},
                "total": {},
            }

            for ip_instance in die_ip_instances[die_id]:
                self.die_ip_bandwidth_data[die_id]["read"][ip_instance] = np.zeros((rows, cols))
                self.die_ip_bandwidth_data[die_id]["write"][ip_instance] = np.zeros((rows, cols))
                self.die_ip_bandwidth_data[die_id]["total"][ip_instance] = np.zeros((rows, cols))

        # 基于D2D请求计算带宽
        self._calculate_bandwidth_from_d2d_requests(dies)

    def _calculate_bandwidth_from_d2d_requests(self, dies: Dict):
        """基于D2D请求计算各Die的IP带宽"""

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

        except Exception as e:
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

        except Exception as e:
            pass

    def _normalize_d2d_ip_type(self, ip_type: str) -> str:
        """标准化D2D IP类型，保留实例编号（如gdma_0, ddr_1）"""
        if not ip_type:
            return "l2m"

        # 转换为小写
        ip_type = ip_type.lower()

        # 提取基本类型以验证
        if "_" in ip_type:
            base_type = ip_type.split("_")[0]
        else:
            base_type = ip_type

        # 支持的基本类型
        supported_types = ["sdma", "gdma", "cdma", "ddr", "l2m", "d2d"]

        # 检查基本类型是否支持
        if base_type in supported_types or base_type.startswith("d2d"):
            return ip_type  # 返回完整的实例名（包含编号）
        else:
            return "l2m"

    def _get_physical_position(self, node_id: int, die_model) -> tuple:
        """获取节点的物理位置(row, col)"""
        cols = die_model.config.NUM_COL

        # 简单映射：node_id直接映射到物理位置
        row = node_id // cols
        col = node_id % cols

        # 如果是奇数行，调整到偶数行显示（因为奇偶数行是同一个物理节点）
        if row % 2 == 1:
            row = row - 1  # 奇数行变成偶数行

        return row, col

    def draw_d2d_flow_graph(self, die_networks=None, dies=None, config=None, mode="utilization", node_size=1200, save_path=None, show_cdma=True):
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

        # 获取推断的 Die 布局
        die_layout = getattr(config, "die_layout_positions", {})
        die_layout_type = getattr(config, "die_layout_type", "2x1")

        # 根据布局设置画布大小和Die偏移
        die_width = 16
        die_height = 10

        # 使用推断的布局，传入dies和config进行对齐优化
        die_offsets, figsize = self._calculate_die_offsets_from_layout(die_layout, die_layout_type, die_width, die_height, dies=dies, config=config)

        # 创建画布
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        # 为每个Die绘制流量图并收集节点位置
        die_node_positions = {}
        used_ip_types = set()  # 收集实际使用的IP类型
        for die_id, network in die_networks_for_draw.items():
            offset_x, offset_y = die_offsets[die_id]
            die_model = dies.get(die_id) if dies else None

            # 绘制单个Die的流量图并获取节点位置
            node_positions = self._draw_single_die_flow(ax, network, die_model.config if die_model else config, die_id, offset_x, offset_y, mode, node_size, show_cdma, die_model, d2d_config=config)
            die_node_positions[die_id] = node_positions

            # 收集该Die使用的IP类型（只收集有实际带宽的IP）
            if hasattr(self, "die_processors") and die_id in self.die_processors:
                die_processor = self.die_processors[die_id]
                if hasattr(die_processor, "ip_bandwidth_data") and die_processor.ip_bandwidth_data is not None:
                    if mode in die_processor.ip_bandwidth_data:
                        mode_data = die_processor.ip_bandwidth_data[mode]
                        for ip_type, data_matrix in mode_data.items():
                            # 检查该IP类型在任意位置是否有带宽 > 0.001
                            if (data_matrix > 0.001).any():
                                used_ip_types.add(ip_type.upper())

        # 如果没有从die_processors获取到IP类型，尝试从self.ip_bandwidth_data获取
        if not used_ip_types and hasattr(self, "ip_bandwidth_data") and self.ip_bandwidth_data is not None:
            if mode in self.ip_bandwidth_data:
                mode_data = self.ip_bandwidth_data[mode]
                for ip_type, data_matrix in mode_data.items():
                    # 检查该IP类型在任意位置是否有带宽 > 0.001
                    if (data_matrix > 0.001).any():
                        used_ip_types.add(ip_type.upper())

        # 绘制跨Die数据带宽连接
        try:
            if dies:
                # 计算D2D_Sys带宽统计
                d2d_bandwidth = self._calculate_d2d_sys_bandwidth(dies)
                # 绘制跨Die连接，传入实际节点位置和Die偏移
                self._draw_cross_die_connections(ax, d2d_bandwidth, die_node_positions, config, dies, die_offsets)
            # else:
        except Exception as e:
            import traceback

            traceback.print_exc()

        # 设置图表标题和坐标轴 - 缩小标题字体并调整位置
        title = f"D2D Flow Graph"
        ax.set_title(title, fontsize=14, fontweight="bold", y=0.96)  # 降低标题位置避免被裁剪

        # 添加IP类型颜色图例（只显示实际使用的IP类型）
        self._add_ip_legend(ax, fig, used_ip_types)

        # 添加IP带宽热力条图例
        self._add_flow_graph_bandwidth_colorbar(ax, fig, dies, mode)

        # 自动调整坐标轴范围以确保所有内容都显示
        ax.axis("equal")  # 保持纵横比
        ax.margins(0.05)  # 恢复边距以确保内容显示完整
        ax.axis("off")  # 隐藏坐标轴

        # 保存或显示图片
        import warnings

        if save_path:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
                plt.tight_layout(pad=0.3)
                plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
            plt.close()
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
                plt.tight_layout(pad=0.3)
                plt.show()

    def draw_ip_bandwidth_heatmap(self, dies=None, config=None, mode="total", node_size=4000, save_path=None):
        """
        绘制IP带宽热力图，不显示链路带宽，只显示物理节点（偶数行）和IP带宽

        Args:
            dies: 字典 {die_id: die_model}，包含所有Die的模型对象
            config: D2D配置对象
            mode: 显示模式，支持 'read', 'write', 'total'
            node_size: 节点大小（热力图中节点会更大）
            save_path: 图片保存路径
        """
        if dies is None or len(dies) == 0:
            print("警告: 没有提供Die数据")
            return

        if not hasattr(self, "die_ip_bandwidth_data") or not self.die_ip_bandwidth_data:
            print("警告: 没有die_ip_bandwidth_data数据，跳过IP带宽热力图绘制")
            return

        # 获取推断的 Die 布局
        die_layout = getattr(config, "die_layout_positions", {})
        die_layout_type = getattr(config, "die_layout_type", "2x1")

        # 根据布局设置画布大小和Die偏移
        die_width = 16
        die_height = 10

        # 使用推断的布局
        die_offsets, figsize = self._calculate_die_offsets_from_layout(die_layout, die_layout_type, die_width, die_height, dies=dies, config=config)

        # 创建画布
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        # 收集所有IP类型和最大带宽值（用于归一化透明度）
        all_bandwidths = []
        used_ip_types = set()

        for die_id in dies.keys():
            if die_id in self.die_ip_bandwidth_data:
                die_data = self.die_ip_bandwidth_data[die_id]
                if mode in die_data:
                    for ip_type, data_matrix in die_data[mode].items():
                        # 收集所有非零带宽值
                        nonzero_bw = data_matrix[data_matrix > 0.001]
                        if len(nonzero_bw) > 0:
                            all_bandwidths.extend(nonzero_bw.tolist())
                            used_ip_types.add(ip_type.upper().split("_")[0])

        # 计算全局带宽范围（用于透明度归一化）
        max_bandwidth = max(all_bandwidths) if all_bandwidths else 1.0
        min_bandwidth = min(all_bandwidths) if all_bandwidths else 0.0

        # 为每个Die绘制IP热力图并收集位置信息
        die_positions = {}  # 存储每个Die的节点位置范围
        for die_id, die_model in dies.items():
            if die_id not in self.die_ip_bandwidth_data:
                continue

            offset_x, offset_y = die_offsets[die_id]
            die_config = die_model.config

            # 获取该Die的物理节点（只处理偶数行）
            physical_nodes = []
            for node in range(die_config.NUM_ROW * die_config.NUM_COL):
                row = node // die_config.NUM_COL
                if row % 2 == 0:  # 只保留偶数行
                    physical_nodes.append(node)

            # 为每个物理节点绘制IP热力图，并收集位置信息
            xs = []
            ys = []
            for node in physical_nodes:
                col = node % die_config.NUM_COL
                row = node // die_config.NUM_COL

                # 计算节点中心位置
                x = col * 3 + offset_x
                y = -row * 1.5 + offset_y
                xs.append(x)
                ys.append(y)

                # 绘制该节点的IP热力图
                self._draw_ip_heatmap_in_node(ax, x, y, node, die_id, die_config, mode, node_size, max_bandwidth, min_bandwidth)

            # 存储该Die的位置范围
            if xs and ys:
                die_positions[die_id] = {"xs": xs, "ys": ys, "offset_x": offset_x, "offset_y": offset_y}

        # 添加Die标签 - 根据连接方向智能放置（参考flow图的逻辑）
        for die_id in die_positions.keys():
            xs = die_positions[die_id]["xs"]
            ys = die_positions[die_id]["ys"]
            die_center_x = (min(xs) + max(xs)) / 2
            die_center_y = (min(ys) + max(ys)) / 2

            # 根据Die布局确定标签位置
            if die_id in die_layout:
                grid_x, grid_y = die_layout[die_id]

                # 判断连接方向
                other_dies = [did for did in die_layout.keys() if did != die_id]
                if other_dies:
                    other_die_id = other_dies[0]
                    other_grid_x, other_grid_y = die_layout[other_die_id]

                    # 判断连接方向
                    is_vertical_connection = grid_y != other_grid_y
                    is_horizontal_connection = grid_x != other_grid_x

                    if is_vertical_connection:
                        # 垂直连接：标题放在左边或右边
                        if grid_x == 0:  # 左边的Die，标题放在左边
                            label_x = min(xs) - 3
                            label_y = die_center_y
                        else:  # 右边的Die，标题放在右边
                            label_x = max(xs) + 3
                            label_y = die_center_y
                    elif is_horizontal_connection:
                        # 水平连接：标题放在下边
                        label_x = die_center_x
                        label_y = min(ys) - 2
                    else:
                        # 其他情况：默认放在上方
                        label_x = die_center_x
                        label_y = max(ys) + 2.5
                else:
                    # 只有一个Die时：默认放在上方
                    label_x = die_center_x
                    label_y = max(ys) + 2.5
            else:
                # 没有布局信息时，默认放在上方
                label_x = die_center_x
                label_y = max(ys) + 2.5

            ax.text(
                label_x,
                label_y,
                f"Die {die_id}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7, edgecolor="none"),
            )

        # 设置图表标题
        title = f"IP Bandwidth Heatmap - {mode.capitalize()} Mode"
        ax.set_title(title, fontsize=14, fontweight="bold", y=0.96)

        # 添加IP类型颜色图例
        self._add_ip_legend(ax, fig, used_ip_types)

        # 添加透明度-带宽对应关系说明
        self._add_bandwidth_alpha_legend(ax, fig, min_bandwidth, max_bandwidth)

        # 自动调整坐标轴范围
        ax.axis("equal")
        ax.margins(0.05)
        ax.axis("off")

        # 保存或显示图片
        import warnings

        if save_path:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
                plt.tight_layout(pad=0.3)
                plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
            plt.close()
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
                plt.tight_layout(pad=0.3)
                plt.show()

    def _draw_single_die_flow(self, ax, network, config, die_id, offset_x, offset_y, mode="utilization", node_size=2000, show_cdma=True, die_model=None, d2d_config=None):
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
                    pass
            except Exception as e:
                import traceback

                traceback.print_exc()
                links = {}
        # else:

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
                x -= 0.24
                y -= 0.5
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
                            # formatted_label += f"\n{zero_attempts_ratio*100:.0f}% {empty_ratio*100:.0f}%"
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
                color="#E8F5E9", 
                ec="black",
                zorder=2,
            )
            ax.add_patch(rect)

            # 为D2D流量图添加简化的IP信息显示 - 仅对偶数行节点显示，并区分Die
            physical_row = node // config.NUM_COL
            if physical_row % 2 == 0:
                self._draw_d2d_ip_info_box(ax, x, y, node, config, mode, square_size, die_id, die_model)

        # 添加Die标签 - 根据连接方向智能放置
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            die_center_x = (min(xs) + max(xs)) / 2
            die_center_y = (min(ys) + max(ys)) / 2

            # 根据推断的Die布局确定标签位置
            if d2d_config:
                die_layout = getattr(d2d_config, "die_layout_positions", {})
            else:
                die_layout = getattr(config, "die_layout_positions", {})

            if not die_layout:
                raise ValueError("无法获取die_layout_positions布局信息，请检查D2D配置")

            if die_id in die_layout:
                grid_x, grid_y = die_layout[die_id]

                # 判断连接方向：检查是否有其他Die以及它们的相对位置
                other_dies = [did for did in die_layout.keys() if did != die_id]
                if other_dies:
                    other_die_id = other_dies[0]  # 假设只有两个Die
                    other_grid_x, other_grid_y = die_layout[other_die_id]

                    # 判断连接方向
                    is_vertical_connection = grid_y != other_grid_y  # y坐标不同为垂直连接
                    is_horizontal_connection = grid_x != other_grid_x  # x坐标不同为水平连接

                    if is_vertical_connection:
                        # 垂直连接：标题放在左边或右边
                        if grid_x == 0:  # 左边的Die，标题放在左边
                            label_x = min(xs) - 3
                            label_y = die_center_y
                        else:  # 右边的Die，标题放在右边
                            label_x = max(xs) + 3
                            label_y = die_center_y
                    elif is_horizontal_connection:
                        # 水平连接：标题放在下边
                        label_x = die_center_x
                        label_y = min(ys) - 2
                    else:
                        # 其他情况：默认放在上方
                        label_x = die_center_x
                        label_y = max(ys) + 2.5
                else:
                    # 只有一个Die时：默认放在上方
                    label_x = die_center_x
                    label_y = max(ys) + 2.5
            else:
                # 没有布局信息时，默认放在上方
                label_x = die_center_x
                label_y = max(ys) + 2.5

            ax.text(
                label_x,
                label_y,
                f"Die {die_id}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7, edgecolor="none"),
            )

        # 返回节点位置信息供跨Die连接使用
        return pos

    def _draw_d2d_ip_info_box(self, ax, x, y, node, config, mode, square_size, die_id=None, die_model=None):
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

        # 如果没有从die_processors获取到数据，尝试从自身的ip_bandwidth_data获取，但要根据Die区分
        if not active_ips and hasattr(self, "ip_bandwidth_data") and self.ip_bandwidth_data is not None:
            # 根据D2D请求过滤，只显示属于该Die的IP带宽
            die_specific_ips = self._get_die_specific_ips(node, die_id, die_model)

            for ip_type, data_matrix in self.ip_bandwidth_data.get(mode, {}).items():
                if physical_row < data_matrix.shape[0] and physical_col < data_matrix.shape[1]:
                    bandwidth = data_matrix[physical_row, physical_col]
                    # 只显示属于该Die且有流量的IP
                    if bandwidth > 0.001 and ip_type.lower() in die_specific_ips:
                        active_ips.append((ip_type.upper(), bandwidth))

        # 始终绘制信息框 - 改为正方形
        # IP信息框位置和大小（正方形）
        ip_size = square_size * 3  # 正方形大小
        ip_x = x - square_size - ip_size * 0.4
        ip_y = y + 0.04

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
            # IP类型颜色映射
            ip_color_map = {
                "GDMA": "#4472C4",  # 蓝色
                "SDMA": "#ED7D31",  # 橙色
                "CDMA": "#70AD47",  # 绿色
                "DDR": "#C00000",  # 红色
                "L2M": "#7030A0",  # 紫色
                "D2D_RN": "#00B0F0",  # 青色
                "D2D_SN": "#FFC000",  # 黄色
                "OTHER": "#808080",  # 灰色（合并类型）
            }

            # 按IP基本类型分组（去除实例编号），统计每种类型的数量
            from collections import defaultdict

            ip_type_count = defaultdict(list)
            for ip_type, bw in active_ips:
                # 提取基本类型：gdma_0 -> GDMA, ddr_1 -> DDR
                base_type = ip_type.split("_")[0] if "_" in ip_type else ip_type
                ip_type_count[base_type].append(bw)

            # 按RN/SN分类排序，最多显示3行
            MAX_ROWS = 3

            # RN类型(GDMA/SDMA/CDMA)和SN类型(DDR)分组
            rn_types = ["GDMA", "SDMA", "CDMA"]
            sn_types = ["DDR"]

            rn_ips = [(k, v) for k, v in ip_type_count.items() if k.upper() in rn_types]
            sn_ips = [(k, v) for k, v in ip_type_count.items() if k.upper() in sn_types]
            other_ips = [(k, v) for k, v in ip_type_count.items() if k.upper() not in rn_types + sn_types]

            # 按带宽总和排序
            rn_ips.sort(key=lambda x: sum(x[1]), reverse=True)
            sn_ips.sort(key=lambda x: sum(x[1]), reverse=True)
            other_ips.sort(key=lambda x: sum(x[1]), reverse=True)

            # 构建最终显示列表(从上到下:RN -> SN)
            display_rows = []
            display_rows.extend(rn_ips)
            display_rows.extend(sn_ips)

            # 如果总行数超过MAX_ROWS，将other_ips均匀分配到前面的行
            if len(display_rows) + len(other_ips) > MAX_ROWS:
                # 只保留前MAX_ROWS行，将other_ips均匀分配
                display_rows = display_rows[:MAX_ROWS]
                # 将other类型的实例均匀添加到现有行
                for i, (ip_type, instances) in enumerate(other_ips):
                    target_row = i % len(display_rows)
                    display_rows[target_row] = (display_rows[target_row][0], display_rows[target_row][1] + instances)
            else:
                # 如果不超过MAX_ROWS，直接添加other类型
                display_rows.extend(other_ips)
                if len(display_rows) > MAX_ROWS:
                    display_rows = display_rows[:MAX_ROWS]

            ip_type_count = dict(display_rows)

            # 计算布局参数
            num_ip_types = len(ip_type_count)  # 行数 = IP类型数（最多3行）
            max_instances = max(len(instances) for instances in ip_type_count.values())  # 最大列数 = 最多实例数

            # 计算每个小方块的大小和间距，确保适应ip_size大框
            available_width = ip_size * 0.98  # 可用宽度
            available_height = ip_size * 0.98  # 可用高度

            # 根据可用空间和实例数计算小方块大小
            grid_spacing = square_size * 0.10  # 小方块之间的间距
            row_spacing = square_size * 0.3  # 行之间的间距

            # 计算小方块大小（确保不超出大框）
            max_square_width = (available_width - (max_instances - 1) * grid_spacing) / max_instances
            max_square_height = (available_height - (num_ip_types - 1) * row_spacing) / num_ip_types
            grid_square_size = min(max_square_width, max_square_height, square_size * 1)

            # 计算所有行的总高度
            total_content_height = num_ip_types * grid_square_size + (num_ip_types - 1) * row_spacing

            # 按行绘制小方块（每行一种IP类型），垂直居中
            row_idx = 0
            for ip_type, instances in ip_type_count.items():
                num_instances = len(instances)
                # 提取基本类型以获取颜色
                base_type = ip_type.upper()
                ip_color = ip_color_map.get(base_type, "#808080")  # 默认灰色

                # 计算当前行内容的总宽度
                row_width = num_instances * grid_square_size + (num_instances - 1) * grid_spacing

                # 计算当前行的起始位置（水平居中）
                row_start_x = ip_x - row_width / 2

                # 计算当前行的垂直位置（垂直居中）
                row_y = ip_y + total_content_height / 2 - row_idx * (grid_square_size + row_spacing) - grid_square_size / 2

                # 在当前行绘制该类型的所有实例
                for col_idx, bandwidth in enumerate(instances):
                    # 计算小方块位置
                    block_x = row_start_x + col_idx * (grid_square_size + grid_spacing) + grid_square_size / 2
                    block_y = row_y

                    # 计算透明度（需要全局最大/最小带宽）
                    # 从所有active_ips中提取带宽范围
                    all_bw_values = [bw for _, bw in active_ips]
                    min_bw = min(all_bw_values) if all_bw_values else 0
                    max_bw = max(all_bw_values) if all_bw_values else 1

                    # 计算当前IP的透明度
                    alpha = self._calculate_bandwidth_alpha(bandwidth, min_bw, max_bw)

                    # 绘制小方块（使用透明度表示带宽大小）
                    ip_block = Rectangle(
                        (block_x - grid_square_size / 2, block_y - grid_square_size / 2),
                        width=grid_square_size,
                        height=grid_square_size,
                        facecolor=ip_color,
                        edgecolor="black",
                        linewidth=0.8,
                        alpha=alpha,  # 使用透明度
                        zorder=3,
                    )
                    ax.add_patch(ip_block)

                row_idx += 1
        # 没有活跃IP时不显示任何文字，保持空白

    def _get_die_specific_ips(self, node, die_id, die_model=None):
        """根据D2D请求确定该Die该节点的IP类型"""
        if not hasattr(self, "d2d_requests") or not self.d2d_requests:
            return []

        die_specific_ips = []

        # 计算当前绘制节点的物理位置
        if die_model is None:
            raise ValueError("die_model参数不能为None")
        cols = die_model.config.NUM_COL
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

    def _draw_ip_heatmap_in_node(self, ax, x, y, node, die_id, config, mode, node_size, max_bandwidth, min_bandwidth):
        """
        在指定位置绘制节点的IP带宽热力图

        Args:
            ax: matplotlib坐标轴
            x, y: 节点中心位置
            node: 节点ID
            die_id: Die ID
            config: Die配置
            mode: 显示模式
            node_size: 节点大小
            max_bandwidth: 全局最大带宽（用于归一化）
            min_bandwidth: 全局最小带宽（用于归一化）
        """
        from matplotlib.patches import Rectangle
        from matplotlib import colors as mcolors

        # IP类型颜色映射
        ip_color_map = {
            "GDMA": "#4472C4",  # 蓝色
            "SDMA": "#ED7D31",  # 橙色
            "CDMA": "#70AD47",  # 绿色
            "DDR": "#C00000",  # 红色
            "L2M": "#7030A0",  # 紫色
            "D2D_RN": "#00B0F0",  # 青色
            "D2D_SN": "#FFC000",  # 黄色
        }

        # 获取该节点的物理位置
        physical_col = node % config.NUM_COL
        physical_row = node // config.NUM_COL

        # 收集该节点的所有IP及其带宽
        active_ips = []
        if die_id in self.die_ip_bandwidth_data:
            die_data = self.die_ip_bandwidth_data[die_id]
            if mode in die_data:
                for ip_type, data_matrix in die_data[mode].items():
                    if physical_row < data_matrix.shape[0] and physical_col < data_matrix.shape[1]:
                        bandwidth = data_matrix[physical_row, physical_col]
                        if bandwidth > 0.001:  # 阈值过滤
                            active_ips.append((ip_type, bandwidth))

        # 计算节点框大小
        square_size = (node_size / 1000.0) * 0.3  # 根据node_size调整
        node_box_size = square_size * 3.98  # 节点框大小（减小以增加节点间距）

        # 始终绘制节点外框（即使没有IP）
        # 先绘制带透明度的填充
        node_fill = Rectangle(
            (x - node_box_size / 2, y - node_box_size / 2),
            width=node_box_size,
            height=node_box_size,
            facecolor="#FFF9C4" if active_ips else "#F5F5F5",  # 有IP时用浅黄色，没有IP时用非常浅的灰色
            edgecolor="none",
            alpha=0.3 if active_ips else 1.0,  # 有IP时降低透明度，让背景更淡
            zorder=1,
        )
        ax.add_patch(node_fill)

        # 再绘制不透明的黑色边框
        node_border = Rectangle(
            (x - node_box_size / 2, y - node_box_size / 2),
            width=node_box_size,
            height=node_box_size,
            facecolor="none",
            edgecolor="black",
            linewidth=0.8,
            zorder=1,
        )
        ax.add_patch(node_border)

        # IP带宽热力图不显示节点编号

        # 如果没有活跃的IP，只绘制空框就返回
        if not active_ips:
            return

        # 按IP类型分组（去除实例编号）
        from collections import defaultdict

        ip_type_dict = defaultdict(list)
        for ip_type, bw in active_ips:
            base_type = ip_type.upper().split("_")[0]
            ip_type_dict[base_type].append(bw)

        # 按RN/SN分类排序（与flow图保持一致：RN在上，SN在下）
        rn_types = ["GDMA", "SDMA", "CDMA"]
        sn_types = ["DDR"]

        rn_ips = [(k, v) for k, v in ip_type_dict.items() if k.upper() in rn_types]
        sn_ips = [(k, v) for k, v in ip_type_dict.items() if k.upper() in sn_types]
        other_ips = [(k, v) for k, v in ip_type_dict.items() if k.upper() not in rn_types + sn_types]

        # 按带宽总和排序
        rn_ips.sort(key=lambda x: sum(x[1]), reverse=True)
        sn_ips.sort(key=lambda x: sum(x[1]), reverse=True)
        other_ips.sort(key=lambda x: sum(x[1]), reverse=True)

        # 构建最终显示列表(从上到下:RN -> SN -> Other)
        sorted_ip_types = []
        sorted_ip_types.extend(rn_ips)
        sorted_ip_types.extend(sn_ips)
        sorted_ip_types.extend(other_ips)

        # 计算网格布局
        num_ip_types = len(sorted_ip_types)
        max_instances = max(len(instances) for instances in ip_type_dict.values())

        # 计算IP方块大小和间距
        available_size = node_box_size * 0.92  # 增加可用空间比例
        grid_spacing = square_size * 0.1  # 减小间距

        ip_block_width = (available_size - (max_instances - 1) * grid_spacing) / max_instances
        ip_block_height = (available_size - (num_ip_types - 1) * grid_spacing) / num_ip_types
        ip_block_size = min(ip_block_width, ip_block_height, square_size * 1.2)  # 增大IP方块最大尺寸

        # 计算总内容高度（用于垂直居中）
        total_height = num_ip_types * ip_block_size + (num_ip_types - 1) * grid_spacing

        # 绘制IP方块
        row_idx = 0
        for ip_type, bandwidths in sorted_ip_types:
            num_instances = len(bandwidths)
            ip_color = ip_color_map.get(ip_type, "#808080")

            # 计算当前行的总宽度（用于水平居中）
            row_width = num_instances * ip_block_size + (num_instances - 1) * grid_spacing

            # 绘制该类型的所有实例
            for col_idx, bandwidth in enumerate(bandwidths):
                # 计算透明度：带宽越大，alpha越小（颜色越深）
                alpha = self._calculate_bandwidth_alpha(bandwidth, min_bandwidth, max_bandwidth)

                # 计算IP方块位置（水平和垂直居中）
                ip_x = x - row_width / 2 + col_idx * (ip_block_size + grid_spacing)
                ip_y = y + total_height / 2 - row_idx * (ip_block_size + grid_spacing)

                # 绘制IP方块
                ip_rect = Rectangle((ip_x, ip_y - ip_block_size), width=ip_block_size, height=ip_block_size, facecolor=ip_color, edgecolor="black", linewidth=1, alpha=alpha, zorder=3)
                ax.add_patch(ip_rect)

                # 在方块上显示带宽数值
                bandwidth_text = f"{bandwidth:.1f}" if bandwidth >= 0.1 else f"{bandwidth:.2f}"
                ax.text(
                    ip_x + ip_block_size / 2,
                    ip_y - ip_block_size / 2,
                    bandwidth_text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="black",  # 统一使用黑色字体
                    zorder=4,
                )

            row_idx += 1

    def _calculate_bandwidth_alpha(self, bandwidth, min_bandwidth, max_bandwidth):
        """
        根据带宽值计算透明度
        带宽越大，alpha越大（不透明度越高，颜色越深）

        Args:
            bandwidth: 当前带宽值
            min_bandwidth: 最小带宽
            max_bandwidth: 最大带宽

        Returns:
            alpha值 (0.2-1.0)
        """
        if max_bandwidth <= min_bandwidth:
            return 0.6  # 默认中等透明度

        # 归一化到 0-1
        normalized = (bandwidth - min_bandwidth) / (max_bandwidth - min_bandwidth)

        # 映射到 alpha 范围 (0.2-1.0)，带宽越大alpha越大
        alpha = 0.2 + normalized * 0.8
        return max(0.2, min(1.0, alpha))

    def _add_ip_legend(self, ax, fig, used_ip_types=None):
        """在图表右上角添加IP类型颜色图例（竖着显示，仅显示实际使用的IP）"""
        # IP类型颜色映射（与_draw_d2d_ip_info_box中一致）
        all_ip_legend_data = [
            ("GDMA", "#4472C4"),
            ("SDMA", "#ED7D31"),
            ("CDMA", "#70AD47"),
            ("DDR", "#C00000"),
            ("L2M", "#7030A0"),
            ("D2D_RN", "#00B0F0"),
            ("D2D_SN", "#FFC000"),
        ]

        # 如果提供了used_ip_types，只显示实际使用的IP类型
        if used_ip_types:
            # 将实例名归并为基本类型（gdma_0, gdma_1 -> GDMA）
            base_types = set()
            for ip_instance in used_ip_types:
                # 提取基本类型：gdma_0 -> gdma, d2d_rn_0 -> d2d_rn
                if ip_instance.lower().startswith("d2d"):
                    # D2D类型特殊处理：d2d_rn_0 -> d2d_rn
                    parts = ip_instance.lower().split("_")
                    if len(parts) >= 2:
                        base_type = "_".join(parts[:2])  # d2d_rn
                    else:
                        base_type = parts[0]
                else:
                    base_type = ip_instance.split("_")[0]
                base_types.add(base_type.upper())

            # 只显示实际使用的基本类型
            ip_legend_data = [(label, color) for label, color in all_ip_legend_data if label.upper() in base_types]
        else:
            ip_legend_data = all_ip_legend_data

        # 如果没有要显示的IP类型，直接返回
        if not ip_legend_data:
            return

        # 创建正方形图例标记
        from matplotlib.lines import Line2D

        legend_elements = []
        for label, color in ip_legend_data:
            # 使用正方形marker
            legend_elements.append(Line2D([0], [0], marker="s", color="w", markerfacecolor=color, markeredgecolor="black", markersize=8, label=label, linewidth=0))

        # 添加图例到右上角，竖着显示
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=1,  # 单列显示（竖着排列）
            frameon=True,
            fontsize=9,
            title="IP Types",
            title_fontsize=10,
            edgecolor="black",
            fancybox=False,
        )

    def _shorten_ip_name(self, ip_name):
        """将IP类型名称缩短为单字母"""
        ip_name_upper = ip_name.upper()

        # 使用字典映射简化条件判断
        ip_mapping = {"GDMA": "G", "SDMA": "S", "CDMA": "C", "DDR": "D", "L2M": "L", "D2D_RN": "DR", "D2D_SN": "DS"}

        for prefix, short_name in ip_mapping.items():
            if ip_name_upper.startswith(prefix):
                return short_name

        return ip_name_upper[:2]  # 其他情况取前两个字母

    def _add_bandwidth_alpha_legend(self, ax, fig, min_bandwidth, max_bandwidth):
        """
        添加热力条形式的带宽图例

        Args:
            ax: matplotlib坐标轴
            fig: matplotlib图形对象
            min_bandwidth: 最小带宽值
            max_bandwidth: 最大带宽值
        """
        from matplotlib.patches import Rectangle
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # 如果带宽范围为0，不显示图例
        if max_bandwidth <= min_bandwidth:
            return

        # 创建插入的colorbar坐标轴，放在右上角IP图例下方
        # 位置: [left, bottom, width, height] (相对于主坐标轴)
        cax = inset_axes(
            ax,
            width="2%",  # 宽度，减小
            height="20%",  # 高度，减小
            loc="upper right",  # 位置改为右上
            bbox_to_anchor=(-0.05, -0.35, 1, 1),  # 调整到IP图例下方
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        # 创建自定义colormap（从浅到深，模拟alpha效果）
        # 使用灰度渐变，从浅灰到深灰
        colors = ["#E0E0E0", "#B0B0B0", "#808080", "#505050", "#202020"]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list("bandwidth", colors, N=n_bins)

        # 创建归一化对象
        import matplotlib.colors as mcolors

        norm = mcolors.Normalize(vmin=min_bandwidth, vmax=max_bandwidth)

        # 创建colorbar
        cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")


        # 设置colorbar标签
        cb.set_label("IP BW (GB/s)", fontsize=8, labelpad=3)  # 缩短标签，减小字号

        # 设置刻度标签字体大小
        cax.tick_params(labelsize=7)  # 减小刻度字号

        # 设置刻度数量
        import numpy as np

        n_ticks = 4  # 减少刻度数量
        tick_values = np.linspace(min_bandwidth, max_bandwidth, n_ticks)
        cb.set_ticks(tick_values)
        cb.set_ticklabels([f"{v:.1f}" for v in tick_values])

    def _add_flow_graph_bandwidth_colorbar(self, ax, fig, dies, mode):
        """
        为流量图添加IP带宽热力条图例

        Args:
            ax: matplotlib坐标轴
            fig: matplotlib图形对象
            dies: Die模型字典
            mode: 显示模式
        """
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import LinearSegmentedColormap
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import numpy as np

        # 收集所有IP带宽数据以确定范围
        all_bandwidths = []

        if hasattr(self, "die_processors") and self.die_processors:
            for die_id, die_processor in self.die_processors.items():
                if hasattr(die_processor, "ip_bandwidth_data") and die_processor.ip_bandwidth_data:
                    if mode in die_processor.ip_bandwidth_data:
                        for ip_type, data_matrix in die_processor.ip_bandwidth_data[mode].items():
                            nonzero_bw = data_matrix[data_matrix > 0.001]
                            if len(nonzero_bw) > 0:
                                all_bandwidths.extend(nonzero_bw.tolist())

        # 如果没有带宽数据，不显示colorbar
        if not all_bandwidths:
            return

        min_bandwidth = min(all_bandwidths)
        max_bandwidth = max(all_bandwidths)

        # 如果范围为0，不显示
        if max_bandwidth <= min_bandwidth:
            return

        # 创建插入的colorbar坐标轴，放在右上角IP图例下方
        cax = inset_axes(ax, width="2%", height="18%", loc="upper right", bbox_to_anchor=(-0.05, -0.35, 1, 1), bbox_transform=ax.transAxes, borderpad=0)  # 减小宽度  # 减小高度  # 调整到IP图例下方

        # 创建灰度渐变colormap
        colors = ["#E0E0E0", "#B0B0B0", "#808080", "#505050", "#202020"]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list("bandwidth", colors, N=n_bins)

        # 创建归一化对象
        import matplotlib.colors as mcolors

        norm = mcolors.Normalize(vmin=min_bandwidth, vmax=max_bandwidth)

        # 创建colorbar
        cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")


        # 设置colorbar标签
        cb.set_label("IP BW (GB/s)", fontsize=7, labelpad=2)  # 减小字号和间距

        # 设置刻度
        cax.tick_params(labelsize=6)  # 减小刻度字号
        n_ticks = 4
        tick_values = np.linspace(min_bandwidth, max_bandwidth, n_ticks)
        cb.set_ticks(tick_values)
        cb.set_ticklabels([f"{v:.1f}" for v in tick_values])

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

        for die_id, die_model in dies.items():
            d2d_sys_bandwidth[die_id] = {}

            # 从该Die的所有d2d_systems分别计算每个节点的带宽
            if hasattr(die_model, "d2d_systems"):

                for pos, d2d_sys in die_model.d2d_systems.items():
                    # 为每个节点单独计算带宽
                    node_bandwidth = {"AR": 0.0, "R": 0.0, "AW": 0.0, "W": 0.0, "B": 0.0}  # 读地址通道  # 读数据通道  # 写地址通道  # 写数据通道  # 写响应通道

                    if hasattr(d2d_sys, "axi_channel_flit_count"):

                        # 计算该节点各通道的带宽，与其他地方保持一致
                        for channel, flit_count in d2d_sys.axi_channel_flit_count.items():
                            if channel in node_bandwidth and flit_count > 0 and time_cycles > 0:
                                bandwidth = flit_count * 128 / time_cycles  # 与第652行的计算方式一致
                                node_bandwidth[channel] = bandwidth

                    # else:

                    d2d_sys_bandwidth[die_id][pos] = node_bandwidth

        return d2d_sys_bandwidth

    def _calculate_d2d_node_positions(self, from_die_id, from_node, to_die_id, to_node, dies, config):
        """
        计算D2D连接的节点位置

        Args:
            from_die_id: 源Die ID
            from_node: 源节点位置
            to_die_id: 目标Die ID
            to_node: 目标节点位置
            dies: Die对象字典
            config: 配置对象

        Returns:
            tuple: (from_node_pos, to_node_pos) 计算后的节点位置

        Raises:
            ValueError: 当连接类型未知或缺少dies参数时
        """
        # 获取Die布局位置
        die_layout = getattr(config, "die_layout_positions", {})
        from_die_pos = die_layout.get(from_die_id, (0, 0))
        to_die_pos = die_layout.get(to_die_id, (0, 0))

        # 判断连接方向：使用已有的连接类型判断方法
        connection_type = self._get_connection_type(from_die_pos, to_die_pos)

        if dies:
            from_die_cols = dies[from_die_id].config.NUM_COL if from_die_id in dies else 4
            to_die_cols = dies[to_die_id].config.NUM_COL if to_die_id in dies else 4
        else:
            # 需要dies参数来获取各Die的NUM_COL配置
            raise ValueError(f"连接类型为{connection_type}时需要dies参数来获取各Die的NUM_COL配置")

        if connection_type == "vertical":
            # 垂直连接
            from_node_pos = from_node - from_die_cols
            to_node_pos = to_node
        elif connection_type == "horizontal":
            # 水平连接
            from_node_pos = from_node + (from_die_cols if from_node in self.config.D2D_SN_POSITIONS[from_die_id] else 0)
            to_node_pos = to_node + (to_die_cols if to_node in self.config.D2D_SN_POSITIONS[to_die_id] else 0)
        elif connection_type == "diagonal":
            # 对角连接
            from_node_pos = from_node + (from_die_cols if from_node in self.config.D2D_SN_POSITIONS[from_die_id] else 0)
            to_node_pos = to_node + (to_die_cols if to_node in self.config.D2D_SN_POSITIONS[to_die_id] else 0)
        else:
            raise ValueError(f"未知的连接类型: {connection_type}")

        return from_node_pos, to_node_pos

    def _calculate_die_boundary(self, die_offset_x, die_offset_y, num_col, num_row):
        """
        计算Die的边界范围

        Args:
            die_offset_x, die_offset_y: Die的偏移量
            num_col, num_row: Die的列数和行数

        Returns:
            dict: {"left": x, "right": x, "top": y, "bottom": y}
        """
        # 根据节点绘图规则计算边界
        # x坐标: x * 3 + offset_x，其中x范围[0, NUM_COL-1]
        # y坐标: -y * 1.5 + offset_y，其中y范围[0, NUM_ROW-1]
        return {
            "left": die_offset_x - 0.5,  # 左边界稍微延伸
            "right": die_offset_x + (num_col - 1) * 3 + 0.5,  # 右边界稍微延伸
            "top": die_offset_y + 0.5,  # 上边界（y较大）
            "bottom": die_offset_y - (num_row - 1) * 1.5 - 0.5,  # 下边界（y较小）
        }

    def _project_point_to_die_edge(self, node_x, node_y, boundary, direction):
        """
        将节点中心投影到Die边缘

        Args:
            node_x, node_y: 节点中心坐标
            boundary: Die边界字典 {"left", "right", "top", "bottom"}
            direction: 投影方向 "right", "left", "top", "bottom"

        Returns:
            tuple: (edge_x, edge_y) 边缘坐标
        """
        if direction == "right":
            return boundary["right"], node_y
        elif direction == "left":
            return boundary["left"], node_y
        elif direction == "top":
            return node_x, boundary["top"]
        elif direction == "bottom":
            return node_x, boundary["bottom"]
        else:
            return node_x, node_y

    def _project_diagonal_to_edge(self, from_x, from_y, to_x, to_y, from_die_pos, to_die_pos, from_die_boundary, to_die_boundary):
        """
        将对角连接的起止点投影到Die边缘

        Args:
            from_x, from_y: 源节点坐标
            to_x, to_y: 目标节点坐标
            from_die_pos: 源Die的布局位置 (x, y)
            to_die_pos: 目标Die的布局位置 (x, y)
            from_die_boundary: 源Die边界
            to_die_boundary: 目标Die边界

        Returns:
            tuple: (from_x, from_y, to_x, to_y) 投影后的坐标
        """
        # 计算Die之间的相对方向
        dx_die = to_die_pos[0] - from_die_pos[0]
        dy_die = to_die_pos[1] - from_die_pos[1]

        # 推断Die ID（基于标准2x2布局）
        # Die布局: Die2(0,1) - Die3(1,1)
        #          Die1(0,0) - Die0(1,0)
        die_pos_to_id = {
            (1, 0): 0,  # Die 0: 右下
            (0, 0): 1,  # Die 1: 左下
            (0, 1): 2,  # Die 2: 左上
            (1, 1): 3,  # Die 3: 右上
        }

        from_die_id = die_pos_to_id.get(tuple(from_die_pos))
        to_die_id = die_pos_to_id.get(tuple(to_die_pos))

        # 确定投影方向：根据较小Die ID的奇偶性
        # Die 0 ↔ Die 2 (偶数ID): 使用水平投影（左右边）
        # Die 1 ↔ Die 3 (奇数ID): 使用垂直投影（上下边）
        if from_die_id is not None and to_die_id is not None:
            min_die_id = min(from_die_id, to_die_id)
            use_horizontal = min_die_id % 2 == 0
        else:
            # 如果无法推断Die ID，默认使用水平投影
            use_horizontal = True

        if use_horizontal:
            # 水平投影（左右边）
            if dx_die > 0:  # 目标在右侧
                from_edge_x, from_edge_y = self._project_point_to_die_edge(from_x, from_y, from_die_boundary, "right")
            else:  # 目标在左侧
                from_edge_x, from_edge_y = self._project_point_to_die_edge(from_x, from_y, from_die_boundary, "left")

            if dx_die > 0:  # 源在左侧
                to_edge_x, to_edge_y = self._project_point_to_die_edge(to_x, to_y, to_die_boundary, "left")
            else:  # 源在右侧
                to_edge_x, to_edge_y = self._project_point_to_die_edge(to_x, to_y, to_die_boundary, "right")
        else:
            # 垂直投影（上下边）
            if dy_die > 0:  # 目标在上侧
                from_edge_x, from_edge_y = self._project_point_to_die_edge(from_x, from_y, from_die_boundary, "top")
            else:  # 目标在下侧
                from_edge_x, from_edge_y = self._project_point_to_die_edge(from_x, from_y, from_die_boundary, "bottom")

            if dy_die > 0:  # 源在下侧
                to_edge_x, to_edge_y = self._project_point_to_die_edge(to_x, to_y, to_die_boundary, "bottom")
            else:  # 源在上侧
                to_edge_x, to_edge_y = self._project_point_to_die_edge(to_x, to_y, to_die_boundary, "top")

        return from_edge_x, from_edge_y, to_edge_x, to_edge_y

    def _calculate_arrow_vectors(self, from_x, from_y, to_x, to_y):
        """
        计算箭头方向向量

        Args:
            from_x, from_y: 起始点坐标
            to_x, to_y: 终点坐标

        Returns:
            tuple: (ux, uy, perpx, perpy) 单位方向向量和垂直向量，如果长度为0则返回None
        """
        dx, dy = to_x - from_x, to_y - from_y
        length = np.sqrt(dx * dx + dy * dy)

        if length > 0:
            ux, uy = dx / length, dy / length
            perpx, perpy = -uy * 0.2, ux * 0.2
            return ux, uy, perpx, perpy
        else:
            return None

    def _draw_cross_die_connections(self, ax, d2d_bandwidth, die_node_positions, config, dies=None, die_offsets=None):
        """
        绘制跨Die数据带宽连接（只显示R和W通道的数据流）
        基于推断的布局和D2D_PAIRS配置绘制连接

        Args:
            ax: matplotlib轴对象
            d2d_bandwidth: D2D_Sys带宽统计 {die_id: {node_pos: {channel: bandwidth}}}
            die_node_positions: 实际的Die节点位置 {die_id: {node: (x, y)}}
            config: 配置对象
            dies: Die模型字典
            die_offsets: Die偏移量字典 {die_id: (offset_x, offset_y)}
        """
        try:
            # 使用推断的D2D连接对
            d2d_pairs = getattr(config, "D2D_PAIRS", [])

            if not d2d_pairs:
                return

            # 收集所有有流量的连接
            active_connections = []

            # 遍历所有D2D连接对
            for die0_id, die0_node, die1_id, die1_node in d2d_pairs:
                # D2D带宽数据使用复合键格式：'源节点_to_目标Die_目标节点'
                # 例如：'5_to_1_37' 表示节点5到Die1节点37的连接

                # 构造复合键
                key_0to1 = f"{die0_node}_to_{die1_id}_{die1_node}"  # Die0 -> Die1
                key_1to0 = f"{die1_node}_to_{die0_id}_{die0_node}"  # Die1 -> Die0

                # 检查写数据流量 (W通道) - 双向都要检查
                w_bw_0to1 = d2d_bandwidth.get(die0_id, {}).get(key_0to1, {}).get("W", 0.0)
                w_bw_1to0 = d2d_bandwidth.get(die1_id, {}).get(key_1to0, {}).get("W", 0.0)

                # 检查读数据返回流量 (R通道) - 双向都要检查
                r_bw_0to1 = d2d_bandwidth.get(die0_id, {}).get(key_0to1, {}).get("R", 0.0)
                r_bw_1to0 = d2d_bandwidth.get(die1_id, {}).get(key_1to0, {}).get("R", 0.0)

                # 添加有流量的连接
                if w_bw_0to1 > 0.001:
                    active_connections.append({"type": "write", "from_die": die0_id, "from_node": die0_node, "to_die": die1_id, "to_node": die1_node, "bandwidth": w_bw_0to1})

                if w_bw_1to0 > 0.001:
                    active_connections.append({"type": "write", "from_die": die1_id, "from_node": die1_node, "to_die": die0_id, "to_node": die0_node, "bandwidth": w_bw_1to0})

                if r_bw_0to1 > 0.001:
                    active_connections.append({"type": "read", "from_die": die0_id, "from_node": die0_node, "to_die": die1_id, "to_node": die1_node, "bandwidth": r_bw_0to1})

                if r_bw_1to0 > 0.001:
                    active_connections.append({"type": "read", "from_die": die1_id, "from_node": die1_node, "to_die": die0_id, "to_node": die0_node, "bandwidth": r_bw_1to0})

            # 绘制所有活跃连接
            for i, conn in enumerate(active_connections):
                # 使用实际节点位置
                from_die_positions = die_node_positions.get(conn["from_die"], {})
                to_die_positions = die_node_positions.get(conn["to_die"], {})

                from_node = conn["from_node"]
                to_node = conn["to_node"]

                # 获取连接类型
                die_layout = getattr(config, "die_layout_positions", {})
                from_die_pos = die_layout.get(conn["from_die"], (0, 0))
                to_die_pos = die_layout.get(conn["to_die"], (0, 0))
                connection_type = self._get_connection_type(from_die_pos, to_die_pos)

                # 计算节点位置
                from_node_pos, to_node_pos = self._calculate_d2d_node_positions(conn["from_die"], from_node, conn["to_die"], to_node, dies, config)

                if from_node_pos not in from_die_positions or to_node_pos not in to_die_positions:
                    print(f"[D2D连接] 警告：找不到节点位置 - From: Die{conn['from_die']}节点{from_node}(pos:{from_node_pos}), To: Die{conn['to_die']}节点{to_node}(pos:{to_node_pos})")
                    continue

                from_x, from_y = from_die_positions[from_node_pos]
                to_x, to_y = to_die_positions[to_node_pos]

                # 计算箭头向量
                arrow_vectors = self._calculate_arrow_vectors(from_x, from_y, to_x, to_y)
                if arrow_vectors is not None:
                    ux, uy, perpx, perpy = arrow_vectors
                    self._draw_single_d2d_arrow(ax, from_x, from_y, to_x, to_y, ux, uy, perpx, perpy, conn["bandwidth"], conn["type"], i, connection_type)

            # 为没有流量的D2D节点对绘制灰色连接线（显示潜在连接）
            self._draw_inactive_d2d_connections(ax, d2d_pairs, active_connections, die_node_positions, dies)

        except Exception as e:
            import traceback

            traceback.print_exc()

    def _draw_single_d2d_arrow(self, ax, start_node_x, start_node_y, end_node_x, end_node_y, ux, uy, perpx, perpy, bandwidth, arrow_type, connection_index, connection_type=None):
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
            connection_type: 连接类型 ("vertical", "horizontal", "diagonal")
        """
        # 计算箭头起止坐标（留出节点空间）
        # 对角连接需要调整偏移策略
        if connection_type == "diagonal":
            # 对角连接：坐标已经投影到边缘，只需要很小的偏移
            node_offset = 1.4  # 从边缘稍微延伸出来
            perp_offset = 1.4  # 垂直方向的小偏移
            start_x = start_node_x + ux * node_offset + perpx * perp_offset + 0.4
            start_y = start_node_y + uy * node_offset + perpy * perp_offset + 0.6
            end_x = end_node_x - ux * node_offset + perpx * perp_offset + 0.4
            end_y = end_node_y - uy * node_offset + perpy * perp_offset + 0.6
        else:
            # 水平/垂直连接：保持原有逻辑
            start_x = start_node_x + ux * 0.8 + perpx
            start_y = start_node_y + uy * 0.8 + perpy
            end_x = end_node_x - ux * 0.8 + perpx
            end_y = end_node_y - uy * 0.8 + perpy

        # 确定颜色和标签
        if bandwidth > 0.001:
            # 有数据流量
            intensity = min(bandwidth / self.MAX_BANDWIDTH_NORMALIZATION, 1.0)
            color = (intensity, 0, 0)  # 红色
            label_text = f"{bandwidth:.1f}"  # 只显示数值，不加GB/s后缀
            linewidth = 2.5
            zorder = 5
        else:
            # 无数据流量 - 灰色实线
            color = (0.7, 0.7, 0.7)
            label_text = None  # 不显示0
            linewidth = 2.5
            zorder = 4

        # 绘制箭头
        arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle="-|>", mutation_scale=10, color=color, linewidth=linewidth, zorder=zorder)  # 稍小的箭头
        ax.add_patch(arrow)

        # 只在有流量时添加标签，参考Die内部链路的标记方式
        if label_text:
            dx = end_x - start_x
            dy = end_y - start_y

            # 对角连接使用靠近终点的位置，其他连接使用中点
            if connection_type == "diagonal":
                # 对角连接: 在靠近终点的位置
                label_x = start_x + dx * 0.85
                label_y_base = start_y + dy * 0.85

                # 根据方向决定标签位置
                # 右下→左上(dx>0,dy>0) → 上方; 左上→右下(dx<0,dy<0) → 下方
                # 右上→左下(dx>0,dy<0) → 上方; 左下→右上(dx<0,dy>0) → 上方
                if (dx > 0 and dy > 0) or (dx > 0 and dy < 0):
                    # 右下→左上 或 右上→左下 → 放上方
                    label_y = label_y_base + 0.5
                else:
                    # 左上→右下 或 左下→右上 → 放下方
                    label_y = label_y_base - 0.5
            else:
                # 垂直和水平连接：使用中点
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2

                # 判断是否为水平方向
                is_horizontal = abs(dx) > abs(dy)

                # 根据方向计算标签偏移
                if is_horizontal:
                    # 水平方向：向右的箭头标签放上面,向左的箭头标签放下面
                    label_x = mid_x
                    label_y = mid_y + (0.5 if dx > 0 else -0.5)
                else:
                    # 垂直方向：标签根据方向向量向左/右偏移，远离箭头
                    label_x = mid_x + (dy * 0.1 if dx > 0 else -dy * 0.1)
                    label_y = mid_y - 0.15

            # 计算箭头角度(以度为单位)
            import numpy as np

            angle_rad = np.arctan2(dy, dx)  # 计算弧度
            angle_deg = np.degrees(angle_rad)  # 转换为角度

            # 确保文字不会倒置(角度在-90到90度之间)
            if angle_deg > 90:
                angle_deg -= 180
            elif angle_deg < -90:
                angle_deg += 180

            # 垂直方向(±90度)的文字也跟随箭头方向旋转
            # 绘制单个标签,文字方向跟随箭头角度
            ax.text(label_x, label_y, label_text, ha="center", va="center", fontsize=8, fontweight="normal", color=color, rotation=angle_deg, rotation_mode="anchor")

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
            #             print(f"[D2D AXI统计] 无法创建输出目录: {e}")
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
                            description = self.AXI_CHANNEL_DESCRIPTIONS.get(channel, f"{channel} Channel")

                            # 添加节点位置信息到CSV
                            writer.writerow([f"Die{die_id}_Node{node_pos}", channel, direction, f"{bandwidth:.6f}", flit_count, description])

        except (IOError, PermissionError) as e:
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

    def _calculate_die_offsets_from_layout(self, die_layout, die_layout_type, die_width, die_height, dies=None, config=None):
        """
        根据推断的 Die 布局计算绘图偏移量和画布大小，包含对齐优化

        Args:
            die_layout: Die 布局位置字典 {die_id: (x, y)}
            die_layout_type: 布局类型字符串，如 "2x2", "2x1" 等
            die_width: 单个 Die 的宽度
            die_height: 单个 Die 的高度
            dies: Die模型字典 {die_id: die_model}，用于对齐计算
            config: 配置对象，用于对齐计算

        Returns:
            (die_offsets, figsize): Die偏移量字典和画布大小
        """
        if not die_layout:
            raise ValueError

        # 计算布局尺寸
        max_x = max(pos[0] for pos in die_layout.values()) if die_layout else 0
        max_y = max(pos[1] for pos in die_layout.values()) if die_layout else 0

        # 计算 Die 间距
        die_spacing_x = die_width + 0.8
        die_spacing_y = die_height + (5.8 if len(die_layout.values()) == 2 else 0.8)

        # 计算每个 Die 的基础绘图偏移量
        die_offsets = {}
        for die_id, (grid_x, grid_y) in die_layout.items():
            offset_x = grid_x * die_spacing_x
            # 修正Y坐标：保持数学坐标系，grid_y越大显示越上方
            offset_y = grid_y * die_spacing_y  # Y 坐标向上为正，与推理坐标系一致
            die_offsets[die_id] = (offset_x, offset_y)

        # 如果提供了dies和config，计算对齐偏移
        if dies and config:
            try:
                alignment_offsets = self._calculate_die_alignment_offsets(dies, config)

                # 应用对齐偏移
                for die_id, (base_x, base_y) in die_offsets.items():
                    if die_id in alignment_offsets:
                        align_x, align_y = alignment_offsets[die_id]
                        die_offsets[die_id] = (base_x + align_x, base_y + align_y)
            except Exception as e:
                # 对齐计算失败时使用默认布局
                print(f"[对齐优化] 对齐计算失败，使用默认布局: {e}")

        # 根据布局大小自动调整画布尺寸 (单位转换为英寸)
        canvas_width = ((max_x + 1) * die_spacing_x + 8) / 10  # 转换为合理的英寸尺寸
        canvas_height = ((max_y + 1) * die_spacing_y + 8) / 10  # 转换为合理的英寸尺寸

        # 限制画布尺寸范围
        canvas_width = max(min(canvas_width, 16), 10)  # 10-16英寸
        canvas_height = max(min(canvas_height, 12), 8)  # 8-12英寸

        figsize = (canvas_width, canvas_height)

        for die_id, offset in die_offsets.items():
            grid_pos = die_layout.get(die_id, (0, 0))

        return die_offsets, figsize

    def _get_connection_type(self, from_die_pos, to_die_pos):
        """
        判断D2D连接类型

        Args:
            from_die_pos: 源Die的网格位置 (x, y)
            to_die_pos: 目标Die的网格位置 (x, y)

        Returns:
            str: "vertical" | "horizontal" | "diagonal"
        """
        dx = abs(from_die_pos[0] - to_die_pos[0])
        dy = abs(from_die_pos[1] - to_die_pos[1])

        if dx == 0:  # X坐标相同，垂直连接
            return "vertical"
        elif dy == 0:  # Y坐标相同，水平连接
            return "horizontal"
        else:  # 对角连接
            return "diagonal"

    def _calculate_die_alignment_offsets(self, dies, config):
        """
        根据D2D连接计算Die位置偏移，使连接线对齐

        Args:
            dies: Die模型字典 {die_id: die_model}
            config: 配置对象

        Returns:
            dict: {die_id: (offset_x, offset_y)} 额外的偏移量
        """
        d2d_pairs = getattr(config, "D2D_PAIRS", [])
        die_layout = getattr(config, "die_layout_positions", {})

        if not d2d_pairs or not die_layout:
            return {}

        # 收集各Die对之间的偏移需求，每对Die只保留偏移量最大的连接
        alignment_constraints = {"vertical": {}, "horizontal": {}}  # {(die0, die1): 最大偏移需求}

        for die0_id, die0_node, die1_id, die1_node in d2d_pairs:
            from_die_pos = die_layout.get(die0_id, (0, 0))
            to_die_pos = die_layout.get(die1_id, (0, 0))

            conn_type = self._get_connection_type(from_die_pos, to_die_pos)

            # 获取节点在各自Die内的物理位置
            die0_model = dies.get(die0_id)
            die1_model = dies.get(die1_id)

            if die0_model and die1_model:
                # 获取节点的行列位置
                die0_cols = die0_model.config.NUM_COL
                die1_cols = die1_model.config.NUM_COL
                die1_node += die1_cols

                # d2d_pairs中存储的已经是网络节点位置（在配置解析时已转换）
                # 直接计算网络节点在绘图中的位置（与_draw_single_die_flow中的计算一致）
                die0_row = die0_node // die0_cols
                die0_col = die0_node % die0_cols
                die1_row = die1_node // die1_cols
                die1_col = die1_node % die1_cols

                die_pair = (min(die0_id, die1_id), max(die0_id, die1_id))

                if conn_type == "vertical":
                    # 垂直连接：需要X对齐，计算实际偏移量
                    die0_x = die0_col
                    die1_x = die1_col

                    die0_x *= 3
                    die1_x *= 3

                    offset_needed = abs(die0_x - die1_x)

                    # 只保留偏移量最大的连接
                    if die_pair not in alignment_constraints["vertical"] or offset_needed > alignment_constraints["vertical"][die_pair]["offset"]:
                        alignment_constraints["vertical"][die_pair] = {
                            "die0": die0_id,
                            "die1": die1_id,
                            "col0": die0_col,
                            "col1": die1_col,
                            "row0": die0_row,
                            "row1": die1_row,
                            "offset": offset_needed,
                        }

                elif conn_type == "horizontal":
                    # 水平连接：需要Y对齐，计算实际偏移量
                    die0_y = die0_row
                    die1_y = die1_row

                    die0_y *= -1.5
                    die1_y *= -1.5

                    offset_needed = abs(die0_y - die1_y)

                    # 只保留偏移量最大的连接
                    if die_pair not in alignment_constraints["horizontal"] or offset_needed > alignment_constraints["horizontal"][die_pair]["offset"]:
                        alignment_constraints["horizontal"][die_pair] = {
                            "die0": die0_id,
                            "die1": die1_id,
                            "row0": die0_row,
                            "row1": die1_row,
                            "col0": die0_col,
                            "col1": die1_col,
                            "offset": offset_needed,
                        }

        # 计算最优偏移量，固定Die 0作为参考点
        die_offsets = {}
        for die_id in die_layout.keys():
            die_offsets[die_id] = [0.0, 0.0]  # [x_offset, y_offset]

        # 固定Die 0作为参考点（偏移量保持为0）
        reference_die = 0

        # 处理垂直对齐约束（X方向）- 每对Die只处理一次
        for die_pair, constraint in alignment_constraints["vertical"].items():
            die0 = constraint["die0"]
            die1 = constraint["die1"]
            col0 = constraint["col0"]
            col1 = constraint["col1"]
            row0 = constraint["row0"]
            row1 = constraint["row1"]

            # 计算实际的X坐标差异（与_draw_single_die_flow中的坐标计算完全一致）
            die0_x = col0
            die1_x = col1

            die0_x *= 3
            die1_x *= 3

            actual_x_diff = die0_x - die1_x

            # 只移动非参考Die
            if die0 == reference_die:
                # 固定die0，移动die1，使用实际位置差异
                die_offsets[die1][0] += actual_x_diff
            elif die1 == reference_die:
                # 固定die1，移动die0
                die_offsets[die0][0] -= actual_x_diff
            else:
                # 两个都不是参考Die，选择ID较大的移动
                if die0 > die1:
                    die_offsets[die0][0] -= actual_x_diff
                else:
                    die_offsets[die1][0] += actual_x_diff

        # 处理水平对齐约束（Y方向）- 每对Die只处理一次
        for die_pair, constraint in alignment_constraints["horizontal"].items():
            die0 = constraint["die0"]
            die1 = constraint["die1"]
            row0 = constraint["row0"]
            row1 = constraint["row1"]
            col0 = constraint["col0"]
            col1 = constraint["col1"]

            # 计算实际的Y坐标差异（与_draw_single_die_flow中的坐标计算完全一致）
            die0_y = row0
            die1_y = row1

            die0_y *= -1.5
            die1_y *= -1.5

            actual_y_diff = die0_y - die1_y

            # 只移动非参考Die
            if die0 == reference_die:
                # 固定die0，移动die1，使用实际位置差异
                die_offsets[die1][1] += actual_y_diff
            elif die1 == reference_die:
                # 固定die1，移动die0
                die_offsets[die0][1] -= actual_y_diff
            else:
                # 两个都不是参考Die，选择ID较大的移动
                if die0 > die1:
                    die_offsets[die0][1] -= actual_y_diff
                else:
                    die_offsets[die1][1] += actual_y_diff

        # 转换为元组格式
        return {die_id: tuple(offsets) for die_id, offsets in die_offsets.items()}

    def _draw_inactive_d2d_connections(self, ax, d2d_pairs, active_connections, die_node_positions, dies=None, die_offsets=None):
        """
        为没有流量的D2D连接对绘制灰色连接线

        Args:
            ax: matplotlib轴对象
            d2d_pairs: D2D连接对列表 [(die0_id, die0_node, die1_id, die1_node), ...]
            active_connections: 活跃连接列表
            die_node_positions: Die节点位置 {die_id: {node: (x, y)}}
            dies: Die对象字典，用于获取各Die的NUM_COL配置
            die_offsets: Die偏移量字典 {die_id: (offset_x, offset_y)}
        """
        # 获取所有活跃连接的节点对
        active_pairs = set()
        for conn in active_connections:
            active_pairs.add((conn["from_die"], conn["from_node"], conn["to_die"], conn["to_node"]))

        for die0_id, die0_node, die1_id, die1_node in d2d_pairs:
            from_die_positions = die_node_positions.get(die0_id, {})
            to_die_positions = die_node_positions.get(die1_id, {})

            # 获取连接类型
            die_layout = getattr(self.config, "die_layout_positions", {})
            from_die_pos = die_layout.get(die0_id, (0, 0))
            to_die_pos = die_layout.get(die1_id, (0, 0))
            connection_type = self._get_connection_type(from_die_pos, to_die_pos)

            # 计算节点位置
            from_node_pos, to_node_pos = self._calculate_d2d_node_positions(die0_id, die0_node, die1_id, die1_node, dies, self.config)

            if from_node_pos not in from_die_positions or to_node_pos not in to_die_positions:
                continue

            from_x, from_y = from_die_positions[from_node_pos]
            to_x, to_y = to_die_positions[to_node_pos]

            # 对角连接需要将起止点投影到Die边缘
            if connection_type == "diagonal" and die_offsets and dies:
                # 获取Die配置和偏移
                from_die = dies.get(die0_id)
                to_die = dies.get(die1_id)
                if from_die and to_die:
                    from_offset_x, from_offset_y = die_offsets.get(die0_id, (0, 0))
                    to_offset_x, to_offset_y = die_offsets.get(die1_id, (0, 0))

                    # 计算Die边界
                    from_boundary = self._calculate_die_boundary(from_offset_x, from_offset_y, from_die.config.NUM_COL, from_die.config.NUM_ROW)
                    to_boundary = self._calculate_die_boundary(to_offset_x, to_offset_y, to_die.config.NUM_COL, to_die.config.NUM_ROW)

                    # 投影到边缘
                    from_x, from_y, to_x, to_y = self._project_diagonal_to_edge(from_x, from_y, to_x, to_y, from_die_pos, to_die_pos, from_boundary, to_boundary)

            # 计算箭头向量
            arrow_vectors = self._calculate_arrow_vectors(from_x, from_y, to_x, to_y)
            if arrow_vectors is not None:
                ux, uy, perpx, perpy = arrow_vectors

                # 检查两个方向是否有活跃连接
                has_forward = (die0_id, die0_node, die1_id, die1_node) in active_pairs

                # 如果正向没有活跃连接，绘制正向的灰色箭头
                if not has_forward:
                    self._draw_single_d2d_arrow(ax, from_x, from_y, to_x, to_y, ux, uy, perpx, perpy, 0.0, "inactive", f"inactive_forward_{die0_id}_{die1_id}", connection_type)
