"""
D2D系统专用结果处理器

用于处理跨Die通信的带宽统计和请求记录
"""

import os
import csv
import math
import warnings
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from matplotlib.patches import Rectangle, FancyArrowPatch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .result_processor import BandwidthAnalyzer, RequestInfo, BandwidthMetrics, WorkingInterval
from src.utils.components.flit import Flit, get_original_source_type, get_original_destination_type


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
    cmd_latency_ns: int
    data_latency_ns: int
    transaction_latency_ns: int
    d2d_sn_node: int = None  # 经过的D2D_SN节点物理ID
    d2d_rn_node: int = None  # 经过的D2D_RN节点物理ID


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

    # 统计数据结构模板
    _LATENCY_STAT_TEMPLATE = {"sum": 0, "max": 0, "count": 0}
    _CIRCLING_STAT_TEMPLATE = {"circling_flits": 0, "total_data_flits": 0, "circling_ratio": 0.0}

    def __init__(self, config, min_gap_threshold: int = 50):
        super().__init__(config, min_gap_threshold)
        self.d2d_requests: List[D2DRequestInfo] = []
        self.d2d_stats = D2DBandwidthStats()
        # 修复网络频率属性问题
        self.network_frequency = getattr(config, "NETWORK_FREQUENCY", 2)

    @staticmethod
    def _create_latency_stats() -> Dict:
        """创建延迟统计数据结构"""
        template = D2DResultProcessor._LATENCY_STAT_TEMPLATE
        return {
            "cmd": {"read": template.copy(), "write": template.copy(), "mixed": template.copy()},
            "data": {"read": template.copy(), "write": template.copy(), "mixed": template.copy()},
            "trans": {"read": template.copy(), "write": template.copy(), "mixed": template.copy()},
        }

    @staticmethod
    def _create_circling_stats() -> Dict:
        """创建绕环统计数据结构"""
        template = D2DResultProcessor._CIRCLING_STAT_TEMPLATE
        return {
            "horizontal": template.copy(),
            "vertical": template.copy(),
            "overall": template.copy(),
        }

    @staticmethod
    def _update_latency_stat(stat_dict: Dict, req_type: str, latency_cycle: float) -> None:
        """更新延迟统计数据"""
        group = stat_dict[req_type]
        group["sum"] += latency_cycle
        group["count"] += 1
        group["max"] = max(group["max"], latency_cycle)

        mixed = stat_dict["mixed"]
        mixed["sum"] += latency_cycle
        mixed["count"] += 1
        mixed["max"] = max(mixed["max"], latency_cycle)

    @staticmethod
    def get_rotated_node_mapping(rows: int = 5, cols: int = 4, rotation: int = 0) -> Dict[int, int]:
        """
        计算Die旋转后每个原始节点对应的新节点ID

        Args:
            rows: 原始布局的行数
            cols: 原始布局的列数
            rotation: 旋转角度，可选值：0, 90, 180, 270（顺时针）

        Returns:
            Dict[int, int]: {原始节点ID: 旋转后节点ID} 的映射字典

        Example:
            # Die1顺时针旋转90°后，原始节点0对应旋转后的哪个节点
            mapping = get_rotated_node_mapping(5, 4, 90)
            rotated_node = mapping[0]
        """
        mapping = {}
        total_nodes = rows * cols

        for node_id in range(total_nodes):
            row = node_id // cols
            col = node_id % cols

            if rotation == 0:
                # 无旋转
                new_row, new_col = row, col
                new_cols = cols
            elif rotation == 90:
                # 顺时针90°：5行4列 → 4行5列
                new_row = col
                new_col = rows - 1 - row
                new_cols = rows
            elif rotation == 180:
                # 180°：5行4列 → 5行4列
                new_row = rows - 1 - row
                new_col = cols - 1 - col
                new_cols = cols
            elif rotation == 270:
                # 顺时针270°：5行4列 → 4行5列
                new_row = cols - 1 - col
                new_col = row
                new_cols = rows
            else:
                raise ValueError(f"不支持的旋转角度: {rotation}，只支持0/90/180/270")

            new_node_id = new_row * new_cols + new_col
            mapping[node_id] = new_node_id

        return mapping

    @staticmethod
    def _format_circuit_stats(stats: Dict, prefix: str = "  ") -> List[str]:
        """
        格式化绕环统计数据为文本行

        Args:
            stats: 统计数据字典
            prefix: 行前缀（用于缩进）

        Returns:
            格式化的文本行列表
        """
        lines = [
            f"{prefix}等待时间统计:",
            f"{prefix}  REQ: 横向={stats['wait_cycles']['req_h']}, 纵向={stats['wait_cycles']['req_v']}",
            f"{prefix}  RSP: 横向={stats['wait_cycles']['rsp_h']}, 纵向={stats['wait_cycles']['rsp_v']}",
            f"{prefix}  DATA: 横向={stats['wait_cycles']['data_h']}, 纵向={stats['wait_cycles']['data_v']}",
            f"{prefix}RB ETag统计:",
            f"{prefix}  T1={stats['etag']['RB']['T1']}, T0={stats['etag']['RB']['T0']}",
            f"{prefix}EQ ETag统计:",
            f"{prefix}  T1={stats['etag']['EQ']['T1']}, T0={stats['etag']['EQ']['T0']}",
            f"{prefix}ITag统计:",
            f"{prefix}  H={stats['itag']['h']}, V={stats['itag']['v']}",
            f"{prefix}Retry数量:",
            f"{prefix}  读: {stats['retry']['read']}, 写: {stats['retry']['write']}",
            f"{prefix}下环尝试统计:",
            f"{prefix}  REQ: 横向={stats['circuits']['req_h']}, 纵向={stats['circuits']['req_v']}",
            f"{prefix}  RSP: 横向={stats['circuits']['rsp_h']}, 纵向={stats['circuits']['rsp_v']}",
            f"{prefix}  DATA: 横向={stats['circuits']['data_h']}, 纵向={stats['circuits']['data_v']}",
            f"{prefix}数据绕环比例:",
            f"{prefix}  横向: {stats['circling_ratio']['horizontal']['circling_ratio']*100:.2f}% ({stats['circling_ratio']['horizontal']['circling_flits']}/{stats['circling_ratio']['horizontal']['total_data_flits']})",
            f"{prefix}  纵向: {stats['circling_ratio']['vertical']['circling_ratio']*100:.2f}% ({stats['circling_ratio']['vertical']['circling_flits']}/{stats['circling_ratio']['vertical']['total_data_flits']})",
            f"{prefix}  总体: {stats['circling_ratio']['overall']['circling_ratio']*100:.2f}% ({stats['circling_ratio']['overall']['circling_flits']}/{stats['circling_ratio']['overall']['total_data_flits']})",
        ]
        return lines

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
        return hasattr(flit, "d2d_origin_die") and hasattr(flit, "d2d_target_die") and flit.d2d_origin_die is not None and flit.d2d_target_die is not None

    def _extract_d2d_info(self, first_flit: Flit, last_flit: Flit, packet_id: int) -> Optional[D2DRequestInfo]:
        """从flit中提取D2D请求信息"""
        try:
            # 计算开始时间 - 使用cmd_entry_cake0_cycle（tracker消耗开始）
            if hasattr(first_flit, "cmd_entry_cake0_cycle") and first_flit.cmd_entry_cake0_cycle < float("inf"):
                start_time_ns = first_flit.cmd_entry_cake0_cycle // self.network_frequency
            else:
                start_time_ns = 0

            # 计算结束时间 - 根据请求类型选择合适的时间戳
            req_type = getattr(first_flit, "req_type", "unknown")
            if req_type == "read":
                # 读请求：使用data_received_complete_cycle
                if hasattr(last_flit, "data_received_complete_cycle") and last_flit.data_received_complete_cycle < float("inf"):
                    end_time_ns = last_flit.data_received_complete_cycle // self.network_frequency
                else:
                    end_time_ns = start_time_ns
            elif req_type == "write":
                # 写请求：使用write_complete_received_cycle
                if hasattr(first_flit, "write_complete_received_cycle") and first_flit.write_complete_received_cycle < float("inf"):
                    end_time_ns = first_flit.write_complete_received_cycle // self.network_frequency
                else:
                    end_time_ns = start_time_ns
            else:
                end_time_ns = start_time_ns

            # 从flit读取已计算的延迟值并转换为ns
            cmd_latency_ns = 0
            data_latency_ns = 0
            transaction_latency_ns = 0

            if hasattr(first_flit, "cmd_latency") and first_flit.cmd_latency < float("inf"):
                cmd_latency_ns = int(first_flit.cmd_latency // self.network_frequency)

            if hasattr(first_flit, "data_latency") and first_flit.data_latency < float("inf"):
                data_latency_ns = int(first_flit.data_latency // self.network_frequency)

            if hasattr(first_flit, "transaction_latency") and first_flit.transaction_latency < float("inf"):
                transaction_latency_ns = int(first_flit.transaction_latency // self.network_frequency)

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
                cmd_latency_ns=cmd_latency_ns,
                data_latency_ns=data_latency_ns,
                transaction_latency_ns=transaction_latency_ns,
                d2d_sn_node=getattr(first_flit, "d2d_sn_node", None),
                d2d_rn_node=getattr(first_flit, "d2d_rn_node", None),
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
            "cmd_latency_ns",
            "data_latency_ns",
            "transaction_latency_ns",
            "data_bytes",
            "d2d_sn_node",
            "d2d_rn_node",
        ]

        # 只有存在请求时才保存对应的CSV文件
        if read_requests:
            read_csv_path = os.path.join(output_path, "d2d_read_requests.csv")
            self._save_requests_to_csv(read_requests, read_csv_path, csv_header)

        if write_requests:
            write_csv_path = os.path.join(output_path, "d2d_write_requests.csv")
            self._save_requests_to_csv(write_requests, write_csv_path, csv_header)

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
                            req.cmd_latency_ns,
                            req.data_latency_ns,
                            req.transaction_latency_ns,
                            req.data_bytes,
                            req.d2d_sn_node if req.d2d_sn_node is not None else "",
                            req.d2d_rn_node if req.d2d_rn_node is not None else "",
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

                # 计算并添加平均带宽统计
                from collections import defaultdict

                ip_type_groups = defaultdict(lambda: {"read": [], "write": [], "total": []})

                # 按IP类型分组（去掉实例编号）
                for row in all_rows:
                    ip_type = row[3]  # IP类型列
                    read_bw = float(row[4])  # 读带宽
                    write_bw = float(row[5])  # 写带宽
                    total_bw = float(row[6])  # 总带宽

                    ip_type_groups[ip_type]["read"].append(read_bw)
                    ip_type_groups[ip_type]["write"].append(write_bw)
                    ip_type_groups[ip_type]["total"].append(total_bw)

                # 添加空行分隔
                writer.writerow([])
                writer.writerow(["# 平均带宽统计（按IP类型）"])
                writer.writerow(["ip_type", "avg_read_bandwidth_gbps", "avg_write_bandwidth_gbps", "avg_total_bandwidth_gbps", "instance_count"])

                # 计算并写入平均值
                for ip_type in sorted(ip_type_groups.keys()):
                    group = ip_type_groups[ip_type]
                    count = len(group["read"])

                    avg_read = sum(group["read"]) / count if count > 0 else 0.0
                    avg_write = sum(group["write"]) / count if count > 0 else 0.0
                    avg_total = sum(group["total"]) / count if count > 0 else 0.0

                    writer.writerow([ip_type, f"{avg_read:.6f}", f"{avg_write:.6f}", f"{avg_total:.6f}", count])

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
        start_time = min(req.start_time_ns for req in requests)
        end_time = max(req.end_time_ns for req in requests)
        total_time_ns = max(end_time - start_time, 1)
        total_bytes = sum(req.data_bytes for req in requests)

        # 非加权带宽 (bytes/ns)
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

    def _calculate_d2d_latency_stats(self):
        """计算D2D请求的延迟统计数据（cmd/data/transaction）"""
        stats = self._create_latency_stats()

        # 定义延迟字段映射
        latency_fields = [("cmd", "cmd_latency_ns"), ("data", "data_latency_ns"), ("trans", "transaction_latency_ns")]

        for req in self.d2d_requests:
            for category, field_name in latency_fields:
                latency_ns = getattr(req, field_name, float("inf"))
                latency_cycle = latency_ns * self.network_frequency if latency_ns < float("inf") else float("inf")

                if math.isfinite(latency_cycle):
                    self._update_latency_stat(stats[category], req.req_type, latency_cycle)

        return stats

    def _collect_d2d_circuit_stats(self, dies: Dict):
        """
        从各Die收集绕环和Tag统计数据

        Args:
            dies: Die模型字典

        Returns:
            dict: 包含per_die和summary的统计数据
        """
        per_die_stats = {}

        # 遍历每个Die收集数据
        for die_id, die_model in dies.items():
            # 基础统计
            die_stats = {
                "circuits": {
                    "req_h": getattr(die_model, "req_cir_h_num_stat", 0),
                    "req_v": getattr(die_model, "req_cir_v_num_stat", 0),
                    "rsp_h": getattr(die_model, "rsp_cir_h_num_stat", 0),
                    "rsp_v": getattr(die_model, "rsp_cir_v_num_stat", 0),
                    "data_h": getattr(die_model, "data_cir_h_num_stat", 0),
                    "data_v": getattr(die_model, "data_cir_v_num_stat", 0),
                },
                "wait_cycles": {
                    "req_h": getattr(die_model, "req_wait_cycle_h_num_stat", 0),
                    "req_v": getattr(die_model, "req_wait_cycle_v_num_stat", 0),
                    "rsp_h": getattr(die_model, "rsp_wait_cycle_h_num_stat", 0),
                    "rsp_v": getattr(die_model, "rsp_wait_cycle_v_num_stat", 0),
                    "data_h": getattr(die_model, "data_wait_cycle_h_num_stat", 0),
                    "data_v": getattr(die_model, "data_wait_cycle_v_num_stat", 0),
                },
                "retry": {
                    "read": getattr(die_model, "read_retry_num_stat", 0),
                    "write": getattr(die_model, "write_retry_num_stat", 0),
                },
                "etag": {
                    "RB": {
                        "T1": getattr(die_model, "RB_ETag_T1_num_stat", 0),
                        "T0": getattr(die_model, "RB_ETag_T0_num_stat", 0),
                    },
                    "EQ": {
                        "T1": getattr(die_model, "EQ_ETag_T1_num_stat", 0),
                        "T0": getattr(die_model, "EQ_ETag_T0_num_stat", 0),
                    },
                },
                "itag": {
                    "h": getattr(die_model, "ITag_h_num_stat", 0),
                    "v": getattr(die_model, "ITag_v_num_stat", 0),
                },
            }

            # 计算绕环比例（如果Die有result_processor）
            circling_stats = self._create_circling_stats()

            if hasattr(die_model, "result_processor") and die_model.result_processor:
                try:
                    circling_stats = die_model.result_processor.calculate_circling_eject_stats()
                except Exception:
                    pass

            die_stats["circling_ratio"] = circling_stats

            per_die_stats[die_id] = die_stats

        # 计算汇总统计
        summary_stats = {
            "circuits": {"req_h": 0, "req_v": 0, "rsp_h": 0, "rsp_v": 0, "data_h": 0, "data_v": 0},
            "wait_cycles": {"req_h": 0, "req_v": 0, "rsp_h": 0, "rsp_v": 0, "data_h": 0, "data_v": 0},
            "retry": {"read": 0, "write": 0},
            "etag": {"RB": {"T1": 0, "T0": 0}, "EQ": {"T1": 0, "T0": 0}},
            "itag": {"h": 0, "v": 0},
            "circling_ratio": self._create_circling_stats(),
        }

        for die_stats in per_die_stats.values():
            for key in summary_stats["circuits"]:
                summary_stats["circuits"][key] += die_stats["circuits"][key]
            for key in summary_stats["wait_cycles"]:
                summary_stats["wait_cycles"][key] += die_stats["wait_cycles"][key]
            for key in summary_stats["retry"]:
                summary_stats["retry"][key] += die_stats["retry"][key]
            for tag_type in ["RB", "EQ"]:
                for t in ["T1", "T0"]:
                    summary_stats["etag"][tag_type][t] += die_stats["etag"][tag_type][t]
            for key in summary_stats["itag"]:
                summary_stats["itag"][key] += die_stats["itag"][key]

            # 汇总绕环统计
            for direction in ["horizontal", "vertical", "overall"]:
                summary_stats["circling_ratio"][direction]["circling_flits"] += die_stats["circling_ratio"][direction]["circling_flits"]
                summary_stats["circling_ratio"][direction]["total_data_flits"] += die_stats["circling_ratio"][direction]["total_data_flits"]

        # 计算汇总绕环比例
        for direction in ["horizontal", "vertical", "overall"]:
            total = summary_stats["circling_ratio"][direction]["total_data_flits"]
            circling = summary_stats["circling_ratio"][direction]["circling_flits"]
            summary_stats["circling_ratio"][direction]["circling_ratio"] = circling / total if total > 0 else 0.0

        return {"per_die": per_die_stats, "summary": summary_stats}

    def generate_d2d_bandwidth_report(self, output_path: str, dies: Dict = None):
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

        # 计算延迟统计
        latency_stats = self._calculate_d2d_latency_stats()

        def _avg(cat, op):
            s = latency_stats[cat][op]
            return s["sum"] / s["count"] if s["count"] else 0.0

        report_lines.extend(["", "延迟统计 (cycle):"])
        for cat, label in [("cmd", "CMD"), ("data", "Data"), ("trans", "Trans")]:
            line = (
                f"  {label} 延迟  - "
                f"读: avg {_avg(cat,'read'):.2f}, max {int(latency_stats[cat]['read']['max'])}; "
                f"写: avg {_avg(cat,'write'):.2f}, max {int(latency_stats[cat]['write']['max'])}; "
                f"混合: avg {_avg(cat,'mixed'):.2f}, max {int(latency_stats[cat]['mixed']['max'])}"
            )
            report_lines.append(line)

        # 添加工作区间统计
        read_requests = [r for r in self.d2d_requests if r.req_type == "read"]
        write_requests = [r for r in self.d2d_requests if r.req_type == "write"]

        read_intervals = self.calculate_d2d_working_intervals(read_requests)
        write_intervals = self.calculate_d2d_working_intervals(write_requests)
        mixed_intervals = self.calculate_d2d_working_intervals(self.d2d_requests)

        report_lines.extend(
            [
                "",
                "工作区间统计:",
                f"  读操作工作区间: {len(read_intervals)}",
                f"  写操作工作区间: {len(write_intervals)}",
                f"  混合操作工作区间: {len(mixed_intervals)}",
            ]
        )

        # 添加绕环统计
        circuit_stats_data = None
        if dies:
            circuit_stats_data = self._collect_d2d_circuit_stats(dies)
            summary = circuit_stats_data["summary"]

            report_lines.extend(["", "-" * 60, "绕环与Tag统计（汇总）", "-" * 60])
            report_lines.extend(self._format_circuit_stats(summary, prefix="  "))

        report_lines.append("-" * 60)

        # 打印到屏幕
        for line in report_lines:
            print(line)

        # 保存到文件（包含每个Die的详细统计）
        os.makedirs(output_path, exist_ok=True)
        report_file = os.path.join(output_path, "d2d_bandwidth_summary.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            for line in report_lines:
                f.write(line + "\n")

            # 添加每个Die的详细统计
            if dies and circuit_stats_data:
                f.write("\n\n")
                f.write("=" * 60 + "\n")
                f.write("各Die详细统计\n")
                f.write("=" * 60 + "\n\n")

                for die_id in sorted(circuit_stats_data["per_die"].keys()):
                    die_stats = circuit_stats_data["per_die"][die_id]
                    f.write(f"Die {die_id}:\n")
                    f.write("-" * 30 + "\n")
                    for line in self._format_circuit_stats(die_stats, prefix="  "):
                        f.write(line + "\n")
                    f.write("\n")

        return report_file

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
        self.generate_d2d_bandwidth_report(output_path, dies)

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

        # 添加D2D节点的带宽统计
        self._calculate_d2d_node_bandwidth(dies)

    def _calculate_bandwidth_from_d2d_requests(self, dies: Dict):
        """基于D2D请求计算各Die的IP带宽"""
        from collections import defaultdict

        # 第一步：按(die_id, source_node, source_type)分组source请求
        source_groups = defaultdict(list)
        for request in self.d2d_requests:
            if request.source_die in dies:
                source_type_normalized = self._normalize_d2d_ip_type(request.source_type)
                key = (request.source_die, request.source_node, source_type_normalized)
                source_groups[key].append(request)

        # 第二步：按(die_id, target_node, target_type)分组target请求
        target_groups = defaultdict(list)
        for request in self.d2d_requests:
            if request.target_die in dies:
                target_type_normalized = self._normalize_d2d_ip_type(request.target_type)
                key = (request.target_die, request.target_node, target_type_normalized)
                target_groups[key].append(request)

        # 第三步：处理source带宽
        for (die_id, node, ip_type), requests in source_groups.items():
            row, col = self._get_physical_position(node, dies[die_id])

            # 按req_type分组并计算带宽
            read_reqs = [r for r in requests if r.req_type == "read"]
            write_reqs = [r for r in requests if r.req_type == "write"]

            if read_reqs:
                _, weighted_bw = self._calculate_bandwidth_for_group(read_reqs)
                self.die_ip_bandwidth_data[die_id]["read"][ip_type][row, col] += weighted_bw

            if write_reqs:
                _, weighted_bw = self._calculate_bandwidth_for_group(write_reqs)
                self.die_ip_bandwidth_data[die_id]["write"][ip_type][row, col] += weighted_bw

            if requests:
                _, weighted_bw = self._calculate_bandwidth_for_group(requests)
                self.die_ip_bandwidth_data[die_id]["total"][ip_type][row, col] += weighted_bw

        # 第四步：处理target带宽
        for (die_id, node, ip_type), requests in target_groups.items():
            row, col = self._get_physical_position(node, dies[die_id])

            # 按req_type分组并计算带宽
            read_reqs = [r for r in requests if r.req_type == "read"]
            write_reqs = [r for r in requests if r.req_type == "write"]

            if read_reqs:
                _, weighted_bw = self._calculate_bandwidth_for_group(read_reqs)
                self.die_ip_bandwidth_data[die_id]["read"][ip_type][row, col] += weighted_bw

            if write_reqs:
                _, weighted_bw = self._calculate_bandwidth_for_group(write_reqs)
                self.die_ip_bandwidth_data[die_id]["write"][ip_type][row, col] += weighted_bw

            if requests:
                _, weighted_bw = self._calculate_bandwidth_for_group(requests)
                self.die_ip_bandwidth_data[die_id]["total"][ip_type][row, col] += weighted_bw

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

        return row, col

    def draw_d2d_flow_graph(self, die_networks=None, dies=None, config=None, mode="utilization", node_size=2500, save_path=None):
        """
        绘制D2D双Die流量图，根据D2D_LAYOUT配置动态调整Die排列

        Args:
            die_networks: 字典 {die_id: network_object}，包含两个Die的网络对象（兼容旧调用）
            dies: 字典 {die_id: die_model}，包含两个Die的模型对象（推荐使用）
            config: D2D配置对象
            mode: 显示模式，支持 'utilization', 'total', 'ITag_ratio' 等
            node_size: 节点大小
            save_path: 图片保存路径
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
        die_rotations = getattr(config, "DIE_ROTATIONS", {})

        # 根据布局设置画布大小和Die偏移
        # 注意：所有Die使用相同的基础拓扑（5x4），但旋转会影响实际尺寸
        base_die_rows = 5
        base_die_cols = 4
        node_spacing = 3.0  # 与_draw_single_die_flow中的node_spacing保持一致

        die_width = (base_die_cols - 1) * node_spacing
        die_height = (base_die_rows - 1) * node_spacing

        # 使用推断的布局，传入dies和config进行对齐优化
        # 传入die_rotations以便计算时考虑旋转后的实际尺寸
        die_offsets, figsize = self._calculate_die_offsets_from_layout(die_layout, die_layout_type, die_width, die_height, dies=dies, config=config, die_rotations=die_rotations)

        # 创建画布
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        # 收集所有IP类型和全局带宽范围（用于归一化透明度，与热力图保持一致）
        all_ip_bandwidths = []
        if hasattr(self, "die_processors"):
            for die_id in dies.keys():
                if die_id in self.die_processors:
                    die_processor = self.die_processors[die_id]
                    if hasattr(die_processor, "ip_bandwidth_data") and die_processor.ip_bandwidth_data is not None:
                        if mode in die_processor.ip_bandwidth_data:
                            mode_data = die_processor.ip_bandwidth_data[mode]
                            for ip_type, data_matrix in mode_data.items():
                                # 过滤D2D节点，不在flow图统计中
                                if ip_type.startswith("d2d_sn") or ip_type.startswith("d2d_rn"):
                                    continue
                                nonzero_bw = data_matrix[data_matrix > 0.001]
                                if len(nonzero_bw) > 0:
                                    all_ip_bandwidths.extend(nonzero_bw.tolist())

        # 如果没有从die_processors获取到带宽数据，尝试从self.ip_bandwidth_data获取
        if not all_ip_bandwidths and hasattr(self, "ip_bandwidth_data") and self.ip_bandwidth_data is not None:
            if mode in self.ip_bandwidth_data:
                mode_data = self.ip_bandwidth_data[mode]
                for ip_type, data_matrix in mode_data.items():
                    # 过滤D2D节点，不在flow图统计中
                    if ip_type.startswith("d2d_sn") or ip_type.startswith("d2d_rn"):
                        continue
                    nonzero_bw = data_matrix[data_matrix > 0.001]
                    if len(nonzero_bw) > 0:
                        all_ip_bandwidths.extend(nonzero_bw.tolist())

        # 计算全局IP带宽范围（用于透明度归一化，与热力图保持一致）
        max_ip_bandwidth = max(all_ip_bandwidths) if all_ip_bandwidths else 1.0
        min_ip_bandwidth = min(all_ip_bandwidths) if all_ip_bandwidths else 0.0

        # 获取Die旋转配置
        die_rotations = getattr(config, "DIE_ROTATIONS", {})

        # 为每个Die绘制流量图并收集节点位置
        die_node_positions = {}
        used_ip_types = set()  # 收集实际使用的IP类型
        for die_id, network in die_networks_for_draw.items():
            offset_x, offset_y = die_offsets[die_id]
            die_model = dies.get(die_id) if dies else None
            die_rotation = die_rotations.get(die_id, 0)  # 获取该Die的旋转角度，默认0度
            # 文本需要反方向旋转以保持正向可读
            # 特殊处理180度旋转：-180度和180度都是倒置，应该用0度保持水平
            text_rotation = -die_rotation
            if abs(text_rotation) == 180:
                text_rotation = 0

            # 绘制单个Die的流量图并获取节点位置
            node_positions = self._draw_single_die_flow(
                ax,
                network,
                die_model.config if die_model else config,
                die_id,
                offset_x,
                offset_y,
                mode,
                node_size,
                die_model,
                d2d_config=config,
                max_ip_bandwidth=max_ip_bandwidth,
                min_ip_bandwidth=min_ip_bandwidth,
                rotation=text_rotation,
                die_rotation=die_rotation,
            )
            die_node_positions[die_id] = node_positions

            # 收集该Die使用的IP类型（只收集有实际带宽的IP）
            if hasattr(self, "die_processors") and die_id in self.die_processors:
                die_processor = self.die_processors[die_id]
                if hasattr(die_processor, "ip_bandwidth_data") and die_processor.ip_bandwidth_data is not None:
                    if mode in die_processor.ip_bandwidth_data:
                        mode_data = die_processor.ip_bandwidth_data[mode]
                        for ip_type, data_matrix in mode_data.items():
                            # 过滤D2D节点，不在图例中显示
                            if ip_type.startswith("d2d_sn") or ip_type.startswith("d2d_rn"):
                                continue
                            # 检查该IP类型在任意位置是否有带宽 > 0.001
                            if (data_matrix > 0.001).any():
                                used_ip_types.add(ip_type.upper())

        # 如果没有从die_processors获取到IP类型，尝试从self.ip_bandwidth_data获取
        if not used_ip_types and hasattr(self, "ip_bandwidth_data") and self.ip_bandwidth_data is not None:
            if mode in self.ip_bandwidth_data:
                mode_data = self.ip_bandwidth_data[mode]
                for ip_type, data_matrix in mode_data.items():
                    # 过滤D2D节点，不在图例中显示
                    if ip_type.startswith("d2d_sn") or ip_type.startswith("d2d_rn"):
                        continue
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
            traceback.print_exc()

        # 设置图表标题和坐标轴
        title = f"D2D Flow Graph"
        ax.set_title(title, fontsize=14, fontweight="bold", y=0.96)  # 标题位置稍微远离图表

        # 添加IP类型颜色图例（只有存在IP类型时才绘制）
        if used_ip_types:
            self._add_ip_legend(ax, fig, used_ip_types)

        # 添加IP带宽热力条图例
        self._add_flow_graph_bandwidth_colorbar(ax, fig, dies, mode)

        # 自动调整坐标轴范围以确保所有内容都显示
        ax.axis("equal")  # 保持纵横比
        ax.margins(0.1)  # 增大边距以确保上下都显示完整
        ax.axis("off")  # 隐藏坐标轴

        # 保存或显示图片
        if save_path:
            # 如果save_path是文件夹，自动生成文件名
            if os.path.isdir(save_path) or (not save_path.endswith(".png") and not save_path.endswith(".jpg")):
                filename = f"d2d_flow_graph_{mode}.png"
                save_path = os.path.join(save_path, filename)

            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
                plt.tight_layout(pad=0.5)  # 增大padding留出更多空间
                plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
            plt.close()
            return save_path
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
                plt.tight_layout(pad=0.5)
                plt.show()
            return None

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
        die_rotations = getattr(config, "DIE_ROTATIONS", {})

        # 根据布局设置画布大小和Die偏移
        # 使用与flow图一致的计算方式
        node_spacing = 3.0  # 与节点绘制中的node_spacing保持一致
        # 从第一个Die获取拓扑尺寸
        first_die = list(dies.values())[0]
        base_die_rows = first_die.config.NUM_ROW
        base_die_cols = first_die.config.NUM_COL
        die_width = (base_die_cols - 1) * node_spacing
        die_height = (base_die_rows - 1) * node_spacing

        # 使用推断的布局
        die_offsets, figsize = self._calculate_die_offsets_from_layout(die_layout, die_layout_type, die_width, die_height, dies=dies, config=config, die_rotations=die_rotations)

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
                        # 过滤D2D节点，不在热力图中显示
                        if ip_type.startswith("d2d_sn") or ip_type.startswith("d2d_rn"):
                            continue
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

            # 获取该Die的旋转角度
            die_rotation = die_rotations.get(die_id, 0)

            # 获取该Die的所有节点
            physical_nodes = list(range(die_config.NUM_ROW * die_config.NUM_COL))

            # 原始拓扑尺寸
            orig_rows = die_config.NUM_ROW
            orig_cols = die_config.NUM_COL

            # 为每个物理节点绘制IP热力图，并收集位置信息
            xs = []
            ys = []
            node_spacing = 3.0  # 统一的节点间距
            for node in physical_nodes:
                # 计算原始坐标
                orig_row = node // orig_cols
                orig_col = node % orig_cols

                # 根据Die旋转角度变换坐标
                if die_rotation == 0 or abs(die_rotation) == 360:
                    # 0度：不旋转
                    new_row = orig_row
                    new_col = orig_col
                elif abs(die_rotation) == 90 or abs(die_rotation) == -270:
                    # 顺时针90度：(row, col) → (col, rows-1-row)
                    new_row = orig_col
                    new_col = orig_rows - 1 - orig_row
                elif abs(die_rotation) == 180:
                    # 180度：(row, col) → (rows-1-row, cols-1-col)
                    new_row = orig_rows - 1 - orig_row
                    new_col = orig_cols - 1 - orig_col
                elif abs(die_rotation) == 270 or abs(die_rotation) == -90:
                    # 顺时针270度（逆时针90度）：(row, col) → (cols-1-col, row)
                    new_row = orig_cols - 1 - orig_col
                    new_col = orig_row
                else:
                    # 其他角度：保持原样
                    new_row = orig_row
                    new_col = orig_col

                # 计算实际位置
                x = new_col * node_spacing + offset_x
                y = -new_row * node_spacing + offset_y
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

            # 获取Die旋转角度（文字反向旋转以保持可读）
            die_rotation = die_rotations.get(die_id, 0)
            rotation = -die_rotation
            # 特殊处理180度旋转：-180度和180度都是倒置，应该用0度保持水平
            if abs(rotation) == 180:
                rotation = 0

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
                rotation=0,  # Die标签统一水平显示
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
        if save_path:
            # 如果save_path是文件夹，自动生成文件名
            if os.path.isdir(save_path) or (not save_path.endswith(".png") and not save_path.endswith(".jpg")):
                filename = f"ip_bandwidth_heatmap_{mode}.png"
                save_path = os.path.join(save_path, filename)

            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
                plt.tight_layout(pad=0.3)
                plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
            plt.close()
            return save_path
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
                plt.tight_layout(pad=0.3)
                plt.show()
            return None

    def _draw_single_die_flow(
        self,
        ax,
        network,
        config,
        die_id,
        offset_x,
        offset_y,
        mode="utilization",
        node_size=2000,
        die_model=None,
        d2d_config=None,
        max_ip_bandwidth=None,
        min_ip_bandwidth=None,
        rotation=0,
        die_rotation=0,
    ):
        """
        绘制单个Die的流量图，复用原有draw_flow_graph的核心逻辑

        Args:
            rotation: 文字旋转角度（用于保持文字可读）
            die_rotation: Die整体旋转角度（用于计算节点位置）
        """
        import math

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
                    time_cycles = getattr(self, "simulation_end_cycle", 1000) / config.NETWORK_FREQUENCY
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
                traceback.print_exc()
                links = {}

        # 获取网络节点
        if hasattr(network, "queues") and network.queues:
            actual_nodes = list(network.queues.keys())
        else:
            # 默认5x4拓扑
            actual_nodes = list(range(config.NUM_ROW * config.NUM_COL))

        # 添加节点到图中
        G.add_nodes_from(actual_nodes)

        # 计算节点位置（应用偏移和旋转）
        # 现在不区分RN/SN行，所有节点在标准网格位置，横纵间距相同
        pos = {}
        node_spacing = 3.0  # 统一的节点间距

        # 原始拓扑尺寸
        orig_rows = config.NUM_ROW
        orig_cols = config.NUM_COL

        for node in actual_nodes:
            # 计算原始坐标
            orig_row = node // orig_cols
            orig_col = node % orig_cols

            # 根据Die旋转角度变换坐标（使用die_rotation而不是rotation）
            if die_rotation == 0 or abs(die_rotation) == 360:
                # 0度：不旋转
                new_row = orig_row
                new_col = orig_col
            elif abs(die_rotation) == 90 or abs(die_rotation) == -270:
                # 顺时针90度：(row, col) → (col, rows-1-row)
                new_row = orig_col
                new_col = orig_rows - 1 - orig_row
            elif abs(die_rotation) == 180:
                # 180度：(row, col) → (rows-1-row, cols-1-col)
                new_row = orig_rows - 1 - orig_row
                new_col = orig_cols - 1 - orig_col
            elif abs(die_rotation) == 270 or abs(die_rotation) == -90:
                # 顺时针270度（逆时针90度）：(row, col) → (cols-1-col, row)
                new_row = orig_cols - 1 - orig_col
                new_col = orig_row
            else:
                # 其他角度：保持原样
                new_row = orig_row
                new_col = orig_col

            # 计算实际位置
            x = new_col * node_spacing + offset_x
            y = -new_row * node_spacing + offset_y
            pos[node] = (x, y)

        # 添加有权重的边
        edge_labels = {}
        edge_colors = {}
        self_loop_labels = {}  # 单独存储自环link: {(node, direction): (label, color)}
        for link_key, value in links.items():
            # 处理新架构：link可能是(i, j)或(i, j, 'h'/'v')
            if len(link_key) == 2:
                i, j = link_key
                direction = None
            elif len(link_key) == 3:
                i, j, direction = link_key  # 保留方向标识
            else:
                continue

            if i not in actual_nodes or j not in actual_nodes:
                continue

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
                color = (color_intensity, 0, 0)
            else:
                color = (0.8, 0.8, 0.8)

            # 自环link单独存储，保留方向信息
            if i == j:
                if direction:  # 有方向标识
                    self_loop_labels[(i, direction)] = (formatted_label, color)
                else:  # 无方向标识的自环
                    self_loop_labels[(i, "unknown")] = (formatted_label, color)
            else:
                # 非自环边添加到图中
                G.add_edge(i, j, weight=value)
                if display_value > 0:
                    edge_labels[(i, j)] = formatted_label
                    edge_colors[(i, j)] = color
                else:
                    edge_colors[(i, j)] = color

        # 计算节点大小 - 新架构：增大节点以容纳内部IP方块
        square_size = np.sqrt(node_size) / 50

        # 绘制网络边 - 按照原始flow图的方式绘制双向箭头
        for i, j in G.edges():
            if i not in pos or j not in pos:
                continue

            color = edge_colors.get((i, j), (0.8, 0.8, 0.8))

            x1, y1 = pos[i]
            x2, y2 = pos[j]

            if i != j:  # 非自环边
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx * dx + dy * dy)
                if length > 0:
                    dx, dy = dx / length, dy / length

                    # 计算垂直偏移
                    perp_dx, perp_dy = dy * 0.1, -dx * 0.1

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
                        mutation_scale=8,
                        color=color,
                        zorder=1,
                        linewidth=1,
                    )
                    ax.add_patch(arrow)

        # 绘制非自环边标签
        if edge_labels:
            link_values = [float(links.get((i, j), 0)) for (i, j) in edge_labels.keys()]
            link_mapping_max = max(link_values) if link_values else 0.0
            link_mapping_min = max(0.6 * link_mapping_max, 100) if mode == "total" else 0.0

            for (i, j), label in edge_labels.items():
                if i in pos and j in pos:
                    edge_value = float(links.get((i, j), 0))
                    if edge_value == 0.0:
                        continue

                    if mode == "total":
                        if edge_value <= link_mapping_min:
                            intensity = 0.0
                        else:
                            intensity = (edge_value - link_mapping_min) / (link_mapping_max - link_mapping_min)
                        intensity = min(max(intensity, 0.0), 1.0)
                        color = (intensity, 0, 0)
                    else:
                        color = (edge_value, 0, 0)

                    x1, y1 = pos[i]
                    x2, y2 = pos[j]
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    dx, dy = x2 - x1, y2 - y1

                    has_reverse = G.has_edge(j, i)

                    # 判断链路方向：
                    # 1. Die内部方向：用于计算标签偏移位置
                    # 2. 屏幕方向：用于判断标签是否需要旋转90度
                    orig_i_row = i // config.NUM_COL
                    orig_i_col = i % config.NUM_COL
                    orig_j_row = j // config.NUM_COL
                    orig_j_col = j % config.NUM_COL
                    is_horizontal_in_die = orig_i_row == orig_j_row  # Die内部：同行 = 水平链路
                    is_horizontal_on_screen = abs(dx) > abs(dy)  # 屏幕上：基于屏幕坐标判断

                    # 新架构：标签放在link中间，双向link根据方向放在不同侧
                    # 偏移需要根据Die旋转角度进行变换
                    if has_reverse:
                        # 根据旋转角度决定偏移量大小
                        # 90/270度旋转时，水平和垂直互换，偏移量大小也需要互换
                        is_90_or_270 = abs(die_rotation) in [90, 270]

                        # 计算Die内部坐标系的偏移
                        # 基于Die内部节点编号关系（i和j），保证旋转后相对位置一致
                        if is_horizontal_in_die:
                            # Die内水平link
                            offset_magnitude = 0.70 if is_90_or_270 else 0.35

                            # 基于Die内部节点编号：i<j表示向右，标签放下方（Die内坐标系）
                            if i < j:
                                offset_x_die, offset_y_die = 0, offset_magnitude  # Die内下方
                            else:
                                offset_x_die, offset_y_die = 0, -offset_magnitude  # Die内上方
                        else:
                            # Die内垂直link
                            offset_magnitude = 0.35 if is_90_or_270 else 0.70

                            # 基于Die内部节点编号：i<j表示向下，标签放右侧（Die内坐标系）
                            if i < j:
                                offset_x_die, offset_y_die = -offset_magnitude, 0  # Die内右侧
                            else:
                                offset_x_die, offset_y_die = offset_magnitude, 0  # Die内左侧

                        # 将Die内部偏移旋转到屏幕坐标系
                        angle_rad = math.radians(die_rotation)
                        cos_a = math.cos(angle_rad)
                        sin_a = math.sin(angle_rad)
                        offset_x_screen = offset_x_die * cos_a - offset_y_die * sin_a
                        offset_y_screen = offset_x_die * sin_a + offset_y_die * cos_a

                        label_x = mid_x + offset_x_screen
                        label_y = mid_y - offset_y_screen  # Y轴翻转（屏幕坐标）
                    else:
                        # 单向link：标签直接放在中间
                        label_x = mid_x
                        label_y = mid_y

                    # 所有link标记文字都保持水平可读（rotation=0）
                    # 不根据link方向或Die旋转改变文字方向
                    ax.text(label_x, label_y, label, ha="center", va="center", fontsize=8, fontweight="normal", color=color, rotation=0)

        # 绘制自环边标签
        for (node, direction), (label, color) in self_loop_labels.items():
            if node not in pos or not label:
                continue

            x, y = pos[node]
            original_row = node // config.NUM_COL
            original_col = node % config.NUM_COL

            # Step 1: 判断旋转后的屏幕方向
            if die_rotation in [90, 270]:
                # 90/270度：横纵互换
                screen_direction = "v" if direction == "h" else "h"
            else:
                # 0/180度：方向不变
                screen_direction = direction

            # Step 2: 计算旋转后的行列（用于判断边界）
            orig_rows = config.NUM_ROW
            orig_cols = config.NUM_COL
            if abs(die_rotation) == 90 or abs(die_rotation) == -270:
                # 顺时针90度
                rotated_row = original_col
                rotated_col = orig_rows - 1 - original_row
                rotated_rows = orig_cols
                rotated_cols = orig_rows
            elif abs(die_rotation) == 180:
                # 180度
                rotated_row = orig_rows - 1 - original_row
                rotated_col = orig_cols - 1 - original_col
                rotated_rows = orig_rows
                rotated_cols = orig_cols
            elif abs(die_rotation) == 270 or abs(die_rotation) == -90:
                # 顺时针270度
                rotated_row = orig_cols - 1 - original_col
                rotated_col = original_row
                rotated_rows = orig_cols
                rotated_cols = orig_rows
            else:
                # 0度
                rotated_row = original_row
                rotated_col = original_col
                rotated_rows = orig_rows
                rotated_cols = orig_cols

            # Step 3: 根据屏幕方向和边界位置计算Die内部偏移
            # Step 4: 直接在屏幕坐标系定义偏移量（不需要旋转变换）
            offset_dist = 0.3
            if screen_direction == "h":
                # 屏幕水平自环：放左右两边
                if rotated_col == 0:
                    # 屏幕左边
                    offset_x_screen = -square_size / 2 - offset_dist
                    offset_y_screen = 0
                    text_rotation = 90  # 从下往上读
                else:
                    # 屏幕右边
                    offset_x_screen = square_size / 2 + offset_dist
                    offset_y_screen = 0
                    text_rotation = -90  # 从上往下读
            else:
                # 屏幕垂直自环：放上下两边
                if rotated_row == 0:
                    # 屏幕上边（Y轴向上为正）
                    offset_x_screen = 0
                    offset_y_screen = square_size / 2 + offset_dist
                    text_rotation = 0  # 水平
                else:
                    # 屏幕下边
                    offset_x_screen = 0
                    offset_y_screen = -square_size / 2 - offset_dist
                    text_rotation = 0  # 水平

            label_x = x + offset_x_screen
            label_y = y + offset_y_screen  # 直接使用屏幕坐标系偏移

            # Step 5: 绘制标签
            ax.text(label_x, label_y, f"{label}", ha="center", va="center", color=color, fontweight="normal", fontsize=8, rotation=text_rotation)

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

            # 绘制IP信息 - 新架构：所有节点都显示IP信息
            self._draw_d2d_ip_info_box(ax, x, y, node, config, mode, square_size, die_id, die_model, max_ip_bandwidth, min_ip_bandwidth, rotation=rotation)

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
                rotation=0,  # Die标签统一水平显示
            )

        # 返回节点位置信息供跨Die连接使用
        return pos

    def _draw_d2d_ip_info_box(self, ax, x, y, node, config, mode, square_size, die_id=None, die_model=None, max_ip_bandwidth=None, min_ip_bandwidth=None, rotation=0):
        """
        绘制IP信息框 - 在节点内部显示所有有流量的IP实例（使用小方块+透明度）
        与基类_draw_ip_info_box保持一致

        Args:
            ax: matplotlib坐标轴
            x, y: 节点中心位置
            node: 节点ID
            config: 配置对象
            mode: 显示模式
            square_size: 节点方块大小
            die_id: Die ID
            die_model: Die模型
            max_ip_bandwidth: 全局最大IP带宽
            min_ip_bandwidth: 全局最小IP带宽
        """
        from matplotlib.patches import Rectangle
        from collections import defaultdict

        # 计算物理位置
        physical_col = node % config.NUM_COL
        physical_row = node // config.NUM_COL

        # 收集该节点所有有流量的IP类型
        active_ips = []

        # 从die_ip_bandwidth_data获取数据（D2D的带宽数据结构）
        if hasattr(self, "die_ip_bandwidth_data") and die_id is not None and die_id in self.die_ip_bandwidth_data:
            if mode in self.die_ip_bandwidth_data[die_id]:
                mode_data = self.die_ip_bandwidth_data[die_id][mode]
                for ip_type, data_matrix in mode_data.items():
                    # 过滤D2D节点，不在flow图中显示
                    if ip_type.startswith("d2d_sn") or ip_type.startswith("d2d_rn"):
                        continue
                    matrix_row = physical_row
                    if matrix_row < data_matrix.shape[0] and physical_col < data_matrix.shape[1]:
                        bandwidth = data_matrix[matrix_row, physical_col]
                        if bandwidth > 0.001:
                            active_ips.append((ip_type.upper(), bandwidth))

        # 如果没有活跃IP，直接返回
        if not active_ips:
            return

        # 按IP基本类型分组（去除实例编号）
        ip_type_count = defaultdict(list)
        for ip_type, bw in active_ips:
            # 提取基本类型
            base_type = ip_type.split("_")[0] if "_" in ip_type else ip_type
            ip_type_count[base_type].append(bw)

        # 按RN/SN分类排序，使用类常量
        rn_ips = [(k, v) for k, v in ip_type_count.items() if k.upper() in self.RN_TYPES]
        sn_ips = [(k, v) for k, v in ip_type_count.items() if k.upper() in self.SN_TYPES]
        other_ips = [(k, v) for k, v in ip_type_count.items() if k.upper() not in self.RN_TYPES + self.SN_TYPES]

        # 按带宽总和排序
        rn_ips.sort(key=lambda x: sum(x[1]), reverse=True)
        sn_ips.sort(key=lambda x: sum(x[1]), reverse=True)
        other_ips.sort(key=lambda x: sum(x[1]), reverse=True)

        # 构建最终显示列表（从上到下：RN -> SN -> Other）
        display_rows = []
        display_rows.extend(rn_ips)
        display_rows.extend(sn_ips)

        # 如果总行数超过self.MAX_ROWS，合并other_ips
        if len(display_rows) + len(other_ips) > self.MAX_ROWS:
            display_rows = display_rows[: self.MAX_ROWS]
            for i, (ip_type, instances) in enumerate(other_ips):
                target_row = i % len(display_rows)
                display_rows[target_row] = (display_rows[target_row][0], display_rows[target_row][1] + instances)
        else:
            display_rows.extend(other_ips)
            if len(display_rows) > self.MAX_ROWS:
                display_rows = display_rows[: self.MAX_ROWS]

        ip_type_count = dict(display_rows)

        # 计算布局参数 - 新架构：在节点内部绘制
        num_ip_types = len(ip_type_count)
        max_instances = max(len(instances) for instances in ip_type_count.values())

        # 计算小方块大小和间距（使用节点内部空间）
        available_width = square_size * 0.85
        available_height = square_size * 0.85
        grid_spacing = square_size * 0.05
        row_spacing = square_size * 0.05

        max_square_width = (available_width - (max_instances - 1) * grid_spacing) / max_instances
        max_square_height = (available_height - (num_ip_types - 1) * row_spacing) / num_ip_types
        grid_square_size = min(max_square_width, max_square_height, square_size * 0.5)

        # 计算总内容高度
        total_content_height = num_ip_types * grid_square_size + (num_ip_types - 1) * row_spacing

        # 绘制IP小方块
        row_idx = 0
        for ip_type, instances in ip_type_count.items():
            num_instances = len(instances)
            base_type = ip_type.upper()
            ip_color = self.IP_COLOR_MAP.get(base_type, self.IP_COLOR_MAP["OTHER"])

            # 计算当前行的总宽度
            row_width = num_instances * grid_square_size + (num_instances - 1) * grid_spacing

            # 计算当前行的起始位置（水平居中在节点内部）
            row_start_x = x - row_width / 2

            # 计算当前行的垂直位置（垂直居中在节点内部）
            row_y = y + total_content_height / 2 - row_idx * (grid_square_size + row_spacing) - grid_square_size / 2

            # 绘制该类型的所有实例
            for col_idx, bandwidth in enumerate(instances):
                # 计算小方块位置
                block_x = row_start_x + col_idx * (grid_square_size + grid_spacing) + grid_square_size / 2
                block_y = row_y

                # 计算透明度（使用全局带宽范围）
                alpha = self._calculate_bandwidth_alpha(bandwidth, min_ip_bandwidth if min_ip_bandwidth is not None else 0, max_ip_bandwidth if max_ip_bandwidth is not None else 1)

                # 绘制小方块
                ip_block = Rectangle(
                    (block_x - grid_square_size / 2, block_y - grid_square_size / 2),
                    width=grid_square_size,
                    height=grid_square_size,
                    facecolor=ip_color,
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=alpha,
                    zorder=3,
                )
                ax.add_patch(ip_block)

            row_idx += 1

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

        # 获取该节点的物理位置
        physical_col = node % config.NUM_COL
        physical_row = node // config.NUM_COL

        # 收集该节点的所有IP及其带宽
        active_ips = []
        if die_id in self.die_ip_bandwidth_data:
            die_data = self.die_ip_bandwidth_data[die_id]
            if mode in die_data:
                for ip_type, data_matrix in die_data[mode].items():
                    # 过滤D2D节点，不在热力图中显示
                    if ip_type.startswith("d2d_sn") or ip_type.startswith("d2d_rn"):
                        continue
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
        rn_ips = [(k, v) for k, v in ip_type_dict.items() if k.upper() in self.RN_TYPES]
        sn_ips = [(k, v) for k, v in ip_type_dict.items() if k.upper() in self.SN_TYPES]
        other_ips = [(k, v) for k, v in ip_type_dict.items() if k.upper() not in self.RN_TYPES + self.SN_TYPES]

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
        available_size = node_box_size * 1  # 增加可用空间比例
        grid_spacing = square_size * 0.1  # 减小间距

        ip_block_width = (available_size - (max_instances - 1) * grid_spacing) / max_instances
        ip_block_height = (available_size - (num_ip_types - 1) * grid_spacing) / num_ip_types
        ip_block_size = min(ip_block_width, ip_block_height, square_size * 1.5)  # 增大IP方块最大尺寸

        # 计算总内容高度（用于垂直居中）
        total_height = num_ip_types * ip_block_size + (num_ip_types - 1) * grid_spacing

        # 绘制IP方块
        row_idx = 0
        for ip_type, bandwidths in sorted_ip_types:
            num_instances = len(bandwidths)
            ip_color = self.IP_COLOR_MAP.get(ip_type, "#808080")

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

        # 新架构：节点编号就是物理节点编号，不需要额外的映射转换
        from_node_pos = from_node
        to_node_pos = to_node

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
        将对角连接的起止点投影到Die边缘（基于节点在Die内的相对位置）

        Args:
            from_x, from_y: 源节点坐标
            to_x, to_y: 目标节点坐标
            from_die_pos: 源Die的布局位置 (x, y) [未使用，保留接口兼容性]
            to_die_pos: 目标Die的布局位置 (x, y) [未使用，保留接口兼容性]
            from_die_boundary: 源Die边界
            to_die_boundary: 目标Die边界

        Returns:
            tuple: (from_x, from_y, to_x, to_y) 投影后的坐标
        """
        # 计算源节点到各边界的距离
        from_dist_left = abs(from_x - from_die_boundary["left"])
        from_dist_right = abs(from_x - from_die_boundary["right"])
        from_dist_top = abs(from_y - from_die_boundary["top"])
        from_dist_bottom = abs(from_y - from_die_boundary["bottom"])

        # 计算目标节点到各边界的距离
        to_dist_left = abs(to_x - to_die_boundary["left"])
        to_dist_right = abs(to_x - to_die_boundary["right"])
        to_dist_top = abs(to_y - to_die_boundary["top"])
        to_dist_bottom = abs(to_y - to_die_boundary["bottom"])

        # 源节点：选择最近的边进行投影
        from_h_edge = "left" if from_dist_left < from_dist_right else "right"
        from_v_edge = "top" if from_dist_top < from_dist_bottom else "bottom"

        # 目标节点：选择最近的边进行投影
        to_h_edge = "left" if to_dist_left < to_dist_right else "right"
        to_v_edge = "top" if to_dist_top < to_dist_bottom else "bottom"

        # 投影到水平边（X坐标取边界值，Y坐标保持节点坐标）
        from_edge_x = from_die_boundary[from_h_edge]
        to_edge_x = to_die_boundary[to_h_edge]

        # 投影到垂直边（Y坐标取边界值，X坐标保持节点坐标）
        from_edge_y = from_die_boundary[from_v_edge]
        to_edge_y = to_die_boundary[to_v_edge]

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

            # 获取Die布局信息
            die_layout = getattr(config, "die_layout_positions", {})

            # 遍历所有D2D连接对，统一绘制所有连接（活跃+非活跃）
            arrow_index = 0
            for die0_id, die0_node, die1_id, die1_node in d2d_pairs:
                # D2D带宽数据使用复合键格式：'源节点_to_目标Die_目标节点'
                # 例如：'5_to_1_37' 表示节点5到Die1节点37的连接

                # 双向检查流量并绘制
                directions = [
                    (die0_id, die0_node, die1_id, die1_node),  # Die0 -> Die1
                    (die1_id, die1_node, die0_id, die0_node),  # Die1 -> Die0
                ]

                for from_die, from_node, to_die, to_node in directions:
                    # 构造复合键
                    key = f"{from_node}_to_{to_die}_{to_node}"

                    # 检查写数据流量 (W通道)
                    w_bw = d2d_bandwidth.get(from_die, {}).get(key, {}).get("W", 0.0)
                    # 检查读数据返回流量 (R通道)
                    r_bw = d2d_bandwidth.get(from_die, {}).get(key, {}).get("R", 0.0)

                    # 获取节点位置
                    from_die_positions = die_node_positions.get(from_die, {})
                    to_die_positions = die_node_positions.get(to_die, {})

                    if from_node not in from_die_positions or to_node not in to_die_positions:
                        continue

                    from_x, from_y = from_die_positions[from_node]
                    to_x, to_y = to_die_positions[to_node]

                    from_die_pos = die_layout.get(from_die, (0, 0))
                    to_die_pos = die_layout.get(to_die, (0, 0))
                    connection_type = self._get_connection_type(from_die_pos, to_die_pos)

                    # 对角连接：围绕连线中点顺时针旋转
                    if connection_type == "diagonal":
                        # 计算连线中点
                        mid_x = (from_x + to_x) / 2
                        mid_y = (from_y + to_y) / 2

                        # 顺时针旋转10度（角度转弧度）
                        angle = -8 * np.pi / 180  # 负号表示顺时针
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)

                        # 旋转起点
                        dx_from = from_x - mid_x
                        dy_from = from_y - mid_y
                        from_x = mid_x + dx_from * cos_a - dy_from * sin_a
                        from_y = mid_y + dx_from * sin_a + dy_from * cos_a

                        # 旋转终点
                        dx_to = to_x - mid_x
                        dy_to = to_y - mid_y
                        to_x = mid_x + dx_to * cos_a - dy_to * sin_a
                        to_y = mid_y + dx_to * sin_a + dy_to * cos_a

                    # 计算箭头向量
                    arrow_vectors = self._calculate_arrow_vectors(from_x, from_y, to_x, to_y)
                    if arrow_vectors is None:
                        continue

                    ux, uy, perpx, perpy = arrow_vectors

                    # 合并读写通道带宽（同一AXI通道）
                    total_bw = w_bw + r_bw

                    # 绘制单条箭头，显示总带宽
                    self._draw_single_d2d_arrow(ax, from_x, from_y, to_x, to_y, ux, uy, perpx, perpy, total_bw, arrow_index, connection_type)
                    arrow_index += 1

        except Exception as e:
            import traceback

            traceback.print_exc()

    def _draw_single_d2d_arrow(self, ax, start_node_x, start_node_y, end_node_x, end_node_y, ux, uy, perpx, perpy, bandwidth, connection_index, connection_type=None):
        """
        绘制单个D2D箭头

        Args:
            ax: matplotlib轴对象
            start_node_x, start_node_y: 起始节点坐标
            end_node_x, end_node_y: 结束节点坐标
            ux, uy: 单位方向向量
            perpx, perpy: 垂直方向向量
            bandwidth: 带宽值（读写合并后的总带宽）
            connection_index: 连接索引（用于调试）
            connection_type: 连接类型 ("vertical", "horizontal", "diagonal")
        """
        # 计算箭头起止坐标（留出节点空间）
        # 对角连接需要调整偏移策略
        if connection_type == "diagonal":
            node_offset = 1.2  # 从边缘稍微延伸出来
            perp_offset = 1.2  # 垂直方向的小偏移
            start_x = start_node_x + ux * node_offset + perpx * perp_offset
            start_y = start_node_y + uy * node_offset + perpy * perp_offset
            end_x = end_node_x - ux * node_offset + perpx * perp_offset
            end_y = end_node_y - uy * node_offset + perpy * perp_offset
        else:
            # 水平/垂直连接：保持原有逻辑
            start_x = start_node_x + ux * 1.2 + perpx
            start_y = start_node_y + uy * 1.2 + perpy
            end_x = end_node_x - ux * 1.2 + perpx
            end_y = end_node_y - uy * 1.2 + perpy

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
                    label_y = label_y_base + 0.6
                else:
                    # 左上→右下 或 左下→右上 → 放下方
                    label_y = label_y_base - 0.6
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
            traceback.print_exc()

    def collect_requests_data(self, sim_model, simulation_end_cycle=None) -> None:
        """
        重写基类方法，增加D2D特殊处理
        不再需要修复original_*属性，辅助函数会自动从d2d_*属性推断
        """
        # 调用基类方法收集基本数据（内部使用辅助函数读取original_*属性）
        super().collect_requests_data(sim_model, simulation_end_cycle)

    def _calculate_die_offsets_from_layout(self, die_layout, die_layout_type, die_width, die_height, dies=None, config=None, die_rotations=None):
        """
        根据推断的 Die 布局计算绘图偏移量和画布大小，包含对齐优化

        Args:
            die_layout: Die 布局位置字典 {die_id: (x, y)}
            die_layout_type: 布局类型字符串，如 "2x2", "2x1" 等
            die_width: 基础Die的宽度（旋转前）
            die_height: 基础Die的高度（旋转前）
            dies: Die模型字典 {die_id: die_model}，用于对齐计算
            config: 配置对象，用于对齐计算
            die_rotations: Die旋转角度字典 {die_id: rotation}

        Returns:
            (die_offsets, figsize): Die偏移量字典和画布大小
        """
        if not die_layout:
            raise ValueError

        if die_rotations is None:
            die_rotations = {}

        # 计算布局尺寸
        max_x = max(pos[0] for pos in die_layout.values()) if die_layout else 0
        max_y = max(pos[1] for pos in die_layout.values()) if die_layout else 0

        # 计算每个Die旋转后的实际尺寸
        die_sizes = {}
        for die_id in die_layout.keys():
            rotation = die_rotations.get(die_id, 0)
            if rotation in [90, 270]:
                # 90度或270度旋转：宽高互换
                die_sizes[die_id] = (die_height, die_width)
            else:
                # 0度或180度旋转：宽高不变
                die_sizes[die_id] = (die_width, die_height)

        # 计算每行每列的最大尺寸
        max_width_per_col = {}
        max_height_per_row = {}
        for die_id, (grid_x, grid_y) in die_layout.items():
            w, h = die_sizes[die_id]
            max_width_per_col[grid_x] = max(max_width_per_col.get(grid_x, 0), w)
            max_height_per_row[grid_y] = max(max_height_per_row.get(grid_y, 0), h)

        # 计算每个Die的偏移量（累加前面所有Die的尺寸）
        die_offsets = {}
        gap_x = 7.0  # Die之间的横向间隙
        gap_y = 5.0 if len(die_layout.values()) == 2 else 1

        for die_id, (grid_x, grid_y) in die_layout.items():
            # X方向：累加左侧所有列的宽度 + 间隙
            offset_x = sum(max_width_per_col.get(x, 0) + gap_x for x in range(grid_x))
            # Y方向：累加下方所有行的高度 + 间隙
            offset_y = sum(max_height_per_row.get(y, 0) + gap_y for y in range(grid_y))
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

        # 计算总的画布尺寸（基于累加后的实际尺寸）
        total_width = sum(max_width_per_col.get(x, 0) for x in range(max_x + 1)) + gap_x * max_x + 2
        total_height = sum(max_height_per_row.get(y, 0) for y in range(max_y + 1)) + gap_y * max_y + 2

        # 转换为英寸尺寸（假设每个单位 = 0.3英寸）
        canvas_width = total_width * 0.3
        canvas_height = total_height * 0.3

        # 限制画布尺寸范围
        canvas_width = max(min(canvas_width, 20), 14)  # 10-20英寸
        canvas_height = max(min(canvas_height, 16), 10)  # 8-16英寸

        figsize = (canvas_width, canvas_height)

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

    def _apply_rotation(self, orig_row, orig_col, rows, cols, rotation):
        """
        根据旋转角度计算节点的旋转后行列位置

        Args:
            orig_row: 原始行号
            orig_col: 原始列号
            rows: 总行数
            cols: 总列数
            rotation: 旋转角度（0, 90, 180, 270）

        Returns:
            tuple: (new_row, new_col) 旋转后的行列位置
        """
        if rotation == 0 or abs(rotation) == 360:
            return orig_row, orig_col
        elif abs(rotation) == 90 or abs(rotation) == -270:
            # 顺时针90度：(row, col) → (col, rows-1-row)
            return orig_col, rows - 1 - orig_row
        elif abs(rotation) == 180:
            # 180度：(row, col) → (rows-1-row, cols-1-col)
            return rows - 1 - orig_row, cols - 1 - orig_col
        elif abs(rotation) == 270 or abs(rotation) == -90:
            # 顺时针270度：(row, col) → (cols-1-col, row)
            return cols - 1 - orig_col, orig_row
        else:
            return orig_row, orig_col

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
        die_rotations = getattr(config, "DIE_ROTATIONS", {})

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
                # 获取节点的原始行列位置和旋转角度
                die0_cols = die0_model.config.NUM_COL
                die0_rows = die0_model.config.NUM_ROW
                die1_cols = die1_model.config.NUM_COL
                die1_rows = die1_model.config.NUM_ROW

                die0_rotation = die_rotations.get(die0_id, 0)
                die1_rotation = die_rotations.get(die1_id, 0)

                # 计算原始行列位置
                die0_orig_row = die0_node // die0_cols
                die0_orig_col = die0_node % die0_cols
                die1_orig_row = die1_node // die1_cols
                die1_orig_col = die1_node % die1_cols

                # 计算旋转后的行列位置（与_draw_single_die_flow中的逻辑一致）
                die0_row, die0_col = self._apply_rotation(die0_orig_row, die0_orig_col, die0_rows, die0_cols, die0_rotation)
                die1_row, die1_col = self._apply_rotation(die1_orig_row, die1_orig_col, die1_rows, die1_cols, die1_rotation)

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

                    # 使用统一的节点间距3.0
                    die0_y *= -3
                    die1_y *= -3

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

            # 使用统一的节点间距3.0
            die0_y *= -3
            die1_y *= -3

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

    def _calculate_d2d_node_bandwidth(self, dies: Dict):
        """
        计算D2D_SN和D2D_RN节点的NoC端口带宽
        基于d2d_requests中记录的准确D2D节点ID（d2d_sn_node和d2d_rn_node）
        """
        from collections import defaultdict

        # 按D2D节点分组请求 - key: (die_id, node_id, ip_type)
        d2d_node_groups = defaultdict(lambda: {"read": [], "write": []})

        # 调试：检查请求中的D2D节点ID
        debug_d2d_nodes = {"sn": set(), "rn": set()}
        for request in self.d2d_requests:
            if request.d2d_sn_node is not None:
                debug_d2d_nodes["sn"].add((request.source_die, request.d2d_sn_node))
            if request.d2d_rn_node is not None:
                debug_d2d_nodes["rn"].add((request.target_die, request.d2d_rn_node))

        if debug_d2d_nodes["sn"] or debug_d2d_nodes["rn"]:
            print(f"\n=== D2D节点带宽计算调试 ===")
            print(f"D2D_SN节点: {sorted(debug_d2d_nodes['sn'])}")
            print(f"D2D_RN节点: {sorted(debug_d2d_nodes['rn'])}")
            print(f"总请求数: {len(self.d2d_requests)}")

        for request in self.d2d_requests:
            # 使用记录的准确D2D节点ID
            if request.d2d_sn_node is not None and request.source_die in dies:
                die_model = dies[request.source_die]
                # 查找对应的D2D_SN类型
                for (ip_type, ip_pos), ip_interface in die_model.ip_modules.items():
                    if ip_type.startswith("d2d_sn") and ip_pos == request.d2d_sn_node:
                        key = (request.source_die, ip_pos, ip_type)
                        d2d_node_groups[key][request.req_type].append(request)
                        break

            if request.d2d_rn_node is not None and request.target_die in dies:
                die_model = dies[request.target_die]
                # 查找对应的D2D_RN类型
                for (ip_type, ip_pos), ip_interface in die_model.ip_modules.items():
                    if ip_type.startswith("d2d_rn") and ip_pos == request.d2d_rn_node:
                        key = (request.target_die, ip_pos, ip_type)
                        d2d_node_groups[key][request.req_type].append(request)
                        break

        # 计算每个D2D节点的带宽
        for (die_id, node_id, ip_type), req_groups in d2d_node_groups.items():
            die_model = dies[die_id]
            row, col = self._get_physical_position(node_id, die_model)

            # 确保该D2D节点在数据结构中存在
            for mode in ["read", "write", "total"]:
                if ip_type not in self.die_ip_bandwidth_data[die_id][mode]:
                    rows = die_model.config.NUM_ROW
                    cols = die_model.config.NUM_COL
                    self.die_ip_bandwidth_data[die_id][mode][ip_type] = np.zeros((rows, cols))

            # 计算读带宽
            if req_groups["read"]:
                _, weighted_bw = self._calculate_bandwidth_for_group(req_groups["read"])
                self.die_ip_bandwidth_data[die_id]["read"][ip_type][row, col] += weighted_bw

            # 计算写带宽
            if req_groups["write"]:
                _, weighted_bw = self._calculate_bandwidth_for_group(req_groups["write"])
                self.die_ip_bandwidth_data[die_id]["write"][ip_type][row, col] += weighted_bw

            # 计算总带宽
            all_requests = req_groups["read"] + req_groups["write"]
            if all_requests:
                _, weighted_bw = self._calculate_bandwidth_for_group(all_requests)
                self.die_ip_bandwidth_data[die_id]["total"][ip_type][row, col] += weighted_bw
