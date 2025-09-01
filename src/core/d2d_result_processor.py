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

        print(f"[D2D结果处理] 收集到 {len(self.d2d_requests)} 个跨Die请求")

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
            print(f"[D2D结果处理] 提取请求信息失败 packet_id={packet_id}: {e}")
            return None
        except Exception as e:
            print(f"[D2D结果处理] 未预期的错误 packet_id={packet_id}: {e}")
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
            print(f"[D2D结果处理] 保存CSV文件失败 {file_path}: {e}")
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

        # 加权带宽计算
        if len(requests) > 1:
            total_weighted_bw = 0.0
            total_weight = 0

            for req in requests:
                if req.latency_ns > 0:
                    weight = req.burst_length  # 使用burst_length作为权重
                    bandwidth = req.data_bytes / req.latency_ns if req.latency_ns > 0 else 0.0  # 修复零除错误
                    total_weighted_bw += bandwidth * weight
                    total_weight += weight

            weighted_bw = (total_weighted_bw / total_weight) if total_weight > 0 else unweighted_bw
        else:
            weighted_bw = unweighted_bw

        return unweighted_bw, weighted_bw

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
            f"  跨Die总带宽: {self._calculate_total_bandwidth(stats):.2f} GB/s",
            f"  跨Die加权总带宽: {self._calculate_total_weighted_bandwidth(stats):.2f} GB/s",
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

        print(f"\n[D2D结果处理] 带宽报告已保存到 {report_file}")

    def _calculate_total_bandwidth(self, stats: D2DBandwidthStats) -> float:
        """计算总带宽"""
        return stats.die0_to_die1_read_bw + stats.die0_to_die1_write_bw + stats.die1_to_die0_read_bw + stats.die1_to_die0_write_bw

    def _calculate_total_weighted_bandwidth(self, stats: D2DBandwidthStats) -> float:
        """计算加权总带宽"""
        return stats.die0_to_die1_read_bw_weighted + stats.die0_to_die1_write_bw_weighted + stats.die1_to_die0_read_bw_weighted + stats.die1_to_die0_write_bw_weighted

    def process_d2d_results(self, dies: Dict, output_path: str):
        """
        完整的D2D结果处理流程

        Args:
            dies: Die模型字典
            output_path: 输出目录路径
        """
        print("\n[D2D结果处理] 开始处理D2D系统结果...")

        # 1. 收集跨Die请求数据
        self.collect_cross_die_requests(dies)

        # 2. 计算D2D节点IP带宽统计
        self.calculate_d2d_ip_bandwidth_data(dies)

        # 3. 保存请求到CSV文件
        self.save_d2d_requests_csv(output_path)

        # 4. 计算并输出带宽报告
        self.generate_d2d_bandwidth_report(output_path)

        print("[D2D结果处理] D2D结果处理完成!")

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

        print(f"[D2D结果处理] D2D IP带宽统计完成")

    def _calculate_bandwidth_from_d2d_requests(self, dies: Dict):
        """基于D2D请求计算各Die的IP带宽"""
        print(f"[D2D调试] 开始基于{len(self.d2d_requests)}个D2D请求计算IP带宽")

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

        print(f"[D2D调试] D2D请求带宽计算完成")

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

            print(f"[D2D调试] 源IP带宽: Die{request.source_die} {source_type_normalized}({row},{col}) {request.req_type} = {bandwidth_gbps:.3f} GB/s")

        except Exception as e:
            print(f"[D2D调试] 记录源带宽失败: {e}")

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

            print(f"[D2D调试] 目标IP带宽: Die{request.target_die} {target_type_normalized}({row},{col}) {request.req_type} = {bandwidth_gbps:.3f} GB/s")

        except Exception as e:
            print(f"[D2D调试] 记录目标带宽失败: {e}")

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

    def draw_d2d_flow_graph(self, die_networks, config, mode="utilization", node_size=2000, save_path=None, show_cdma=True):
        """
        绘制D2D双Die流量图，根据D2D_LAYOUT配置动态调整Die排列

        Args:
            die_networks: 字典 {die_id: network_object}，包含两个Die的网络对象
            config: D2D配置对象
            mode: 显示模式，支持 'utilization', 'total', 'ITag_ratio' 等
            node_size: 节点大小
            save_path: 图片保存路径
            show_cdma: 是否显示CDMA
        """

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

        # 为每个Die绘制流量图
        for die_id, network in die_networks.items():
            offset_x, offset_y = die_offsets[die_id]

            # 绘制单个Die的流量图
            self._draw_single_die_flow(ax, network, config, die_id, offset_x, offset_y, mode, node_size, show_cdma)

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
                    print(f"[信息] Die {die_id}: 绘制 {len(active_links)} 条有流量的链路")
            except Exception as e:
                print(f"[D2D流量图] Die {die_id}: 获取链路统计数据失败: {e}")
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

    def _draw_ip_info_box(self, ax, x, y, node, config, mode, show_cdma, square_size):
        """绘制IP信息框"""
        # IP信息框位置和大小
        ip_width = square_size * 3.2
        ip_height = square_size * 2.6
        ip_x = x - square_size - ip_width / 2.5
        ip_y = y + 0.26

        # 绘制IP信息框
        ip_rect = Rectangle(
            (ip_x - ip_width / 2, ip_y - ip_height / 2),
            width=ip_width,
            height=ip_height,
            facecolor="lightcyan",
            edgecolor="black",
            linewidth=1,
            zorder=2,
        )
        ax.add_patch(ip_rect)

        # 添加分割线
        ax.plot([ip_x, ip_x], [ip_y - ip_height / 2, ip_y + ip_height / 2], color="black", linewidth=1, zorder=3)

        # 简化的IP标签（如果没有具体的带宽数据）
        ax.text(ip_x - ip_width / 4, ip_y + ip_height / 4, "SDMA\nGDMA", ha="center", va="center", fontsize=6, color="blue")
        ax.text(ip_x + ip_width / 4, ip_y, "DDR\nL2M", ha="center", va="center", fontsize=6, color="green")

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
                                if node % 10 == 0:  # 减少调试输出频率
                                    print(f"[IP调试] Die {die_id} Node {node}: {ip_type.upper()} = {bandwidth:.3f}")
                else:
                    # 只为第一个节点打印一次，避免重复输出
                    if node % 20 == 0:
                        print(f"[信息] Die {die_id}: 模式'{mode}'暂无IP数据，可用模式: {list(die_processor.ip_bandwidth_data.keys())}")
            else:
                if node % 20 == 0:
                    print(f"[信息] Die {die_id}: die_processor没有ip_bandwidth_data")

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
            return "RN"
        elif ip_name_upper.startswith("D2D_SN"):
            return "SN"
        else:
            return ip_name_upper[:2]  # 其他情况取前两个字母

    def collect_requests_data(self, sim_model, simulation_end_cycle=None) -> None:
        """
        重写基类方法，增加D2D特殊处理
        修复D2D跨Die请求的original_source_type和original_destination_type问题
        """
        # 调用基类方法收集基本数据
        super().collect_requests_data(sim_model, simulation_end_cycle)

        print(f"[D2D调试] 开始修复D2D请求的original类型信息，总请求数: {len(self.requests)}")

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

        print(f"[D2D调试] 修复了 {fixed_count} 个D2D请求的类型信息")

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

                        print(f"[D2D调试] 为请求修复类型: {request.original_source_type} -> {request.original_destination_type}")
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
