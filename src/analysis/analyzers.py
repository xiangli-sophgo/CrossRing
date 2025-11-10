"""
分析器模块 - 包含所有数据类定义、常量和分析器框架

提供:
1. 数据类定义 (RequestInfo, D2DRequestInfo, WorkingInterval, BandwidthMetrics等)
2. 常量定义 (IP_COLOR_MAP, RN_TYPES, SN_TYPES, FLIT_SIZE等)
3. SingleDieAnalyzer - 单Die分析器框架
4. D2DAnalyzer - D2D跨Die分析器框架
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


# ==================== 数据类定义 ====================

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
    # 保序相关字段
    src_dest_order_id: int = -1  # src-dest对的保序ID
    packet_category: str = ""  # 包类型分类 (REQ/RSP/DATA)
    # 所有cycle数据字段
    cmd_entry_cake0_cycle: int = -1
    cmd_entry_noc_from_cake0_cycle: int = -1
    cmd_entry_noc_from_cake1_cycle: int = -1
    cmd_received_by_cake0_cycle: int = -1
    cmd_received_by_cake1_cycle: int = -1
    data_entry_noc_from_cake0_cycle: int = -1
    data_entry_noc_from_cake1_cycle: int = -1
    data_received_complete_cycle: int = -1
    rsp_entry_network_cycle: int = -1
    # 数据flit的尝试下环次数列表
    data_eject_attempts_h_list: List[int] = None
    data_eject_attempts_v_list: List[int] = None
    # 数据flit因保序被阻止的下环次数列表
    data_ordering_blocked_h_list: List[int] = None
    data_ordering_blocked_v_list: List[int] = None

    def __post_init__(self):
        # 初始化列表，避免None值
        if self.data_eject_attempts_h_list is None:
            self.data_eject_attempts_h_list = []
        if self.data_eject_attempts_v_list is None:
            self.data_eject_attempts_v_list = []
        if self.data_ordering_blocked_h_list is None:
            self.data_ordering_blocked_h_list = []
        if self.data_ordering_blocked_v_list is None:
            self.data_ordering_blocked_v_list = []


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
    d2d_sn_node: int = None  # D2D_SN节点物理ID
    d2d_rn_node: int = None  # D2D_RN节点物理ID


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
    mixed_metrics: BandwidthMetrics  # 混合读写指标


@dataclass
class D2DBandwidthStats:
    """D2D带宽统计数据结构（支持任意数量的Die间组合）"""

    # 按 (src_die, dst_die) 记录读写带宽 (unweighted, weighted)
    pair_read_bw: Dict[Tuple[int, int], Tuple[float, float]] = None
    pair_write_bw: Dict[Tuple[int, int], Tuple[float, float]] = None

    total_read_requests: int = 0
    total_write_requests: int = 0
    total_bytes_transferred: int = 0

    def __post_init__(self):
        if self.pair_read_bw is None:
            self.pair_read_bw = {}
        if self.pair_write_bw is None:
            self.pair_write_bw = {}


# ==================== 常量定义 ====================

# IP类型颜色映射（统一定义，所有分析器共享）
IP_COLOR_MAP = {
    "GDMA": "#4472C4",  # 蓝色
    "SDMA": "#ED7D31",  # 橙色
    "CDMA": "#70AD47",  # 绿色
    "DDR": "#C00000",   # 红色
    "L2M": "#7030A0",   # 紫色
    "D2D_RN": "#00B0F0",  # 青色
    "D2D_SN": "#FFC000",  # 黄色
    "OTHER": "#808080",   # 灰色
}

# IP类型分类
RN_TYPES = ["GDMA", "SDMA", "CDMA"]
SN_TYPES = ["DDR", "L2M"]

# Flit大小常量
FLIT_SIZE_BYTES = 128  # 每个flit的字节数

# 图表显示配置
MAX_ROWS = 3  # IP信息框最大行数
MAX_BANDWIDTH_NORMALIZATION = 256.0  # 最大带宽归一化值（GB/s）

# AXI通道常量定义
AXI_CHANNEL_DESCRIPTIONS = {
    "AR": "读地址通道 (Address Read)",
    "R": "读数据通道 (Read Data)",
    "AW": "写地址通道 (Address Write)",
    "W": "写数据通道 (Write Data)",
    "B": "写响应通道 (Write Response)",
}


# ==================== 分析器类框架 ====================

class SingleDieAnalyzer:
    """
    单Die带宽分析器

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
        min_gap_threshold: int = 200,
        plot_rn_bw_fig: bool = False,
        plot_flow_graph: bool = False,
    ):
        """
        初始化单Die分析器

        Args:
            config: 网络配置对象
            min_gap_threshold: 工作区间合并阈值(ns)，小于此值的间隔被视为同一工作区间
            plot_rn_bw_fig: 是否绘制RN带宽曲线图
            plot_flow_graph: 是否绘制流量图
        """
        from collections import defaultdict
        from .core_calculators import DataValidator, TimeIntervalCalculator, BandwidthCalculator
        from .data_collectors import RequestCollector, LatencyStatsCollector, CircuitStatsCollector
        from .visualizers import BandwidthPlotter, FlowGraphRenderer
        from .exporters import CSVExporter, ReportGenerator

        self.config = config
        self.min_gap_threshold = min_gap_threshold
        self.network_frequency = getattr(config, "NETWORK_FREQUENCY", 2)  # GHz
        self.plot_rn_bw_fig = plot_rn_bw_fig
        self.plot_flow_graph = plot_flow_graph
        self.finish_cycle = 0
        self.sim_model = None  # 添加sim_model引用

        # 数据存储
        self.requests: List[RequestInfo] = []
        self.rn_positions = set()
        self.sn_positions = set()
        self.rn_bandwidth_time_series = defaultdict(lambda: {"time": [], "start_times": [], "bytes": []})
        self.ip_bandwidth_data = None
        self.read_ip_intervals = defaultdict(list)
        self.write_ip_intervals = defaultdict(list)

        # 动态IP统计
        self.unique_rn_ips = set()  # 存储 (node, type) 对
        self.unique_sn_ips = set()  # 存储 (node, type) 对
        self.ip_count_by_type = defaultdict(set)  # 按类型统计IP数量
        self.actual_num_ip = 0  # 实际IP数量

        # 初始化组件（组合模式）
        self.validator = DataValidator()
        self.interval_calculator = TimeIntervalCalculator(min_gap_threshold)
        self.calculator = BandwidthCalculator(self.interval_calculator)
        self.request_collector = RequestCollector(self.network_frequency)
        self.latency_collector = LatencyStatsCollector()
        self.circuit_collector = CircuitStatsCollector()
        self.visualizer = BandwidthPlotter()
        self.flow_visualizer = FlowGraphRenderer()
        self.exporter = CSVExporter()
        self.report_generator = ReportGenerator()

        # 初始化节点位置
        self._initialize_node_positions()

    def _initialize_node_positions(self):
        """初始化RN和SN节点位置"""
        # 从配置文件初始化（向后兼容）
        if hasattr(self.config, "GDMA_SEND_POSITION_LIST"):
            self.rn_positions.update(self.config.GDMA_SEND_POSITION_LIST)
        if hasattr(self.config, "SDMA_SEND_POSITION_LIST"):
            self.rn_positions.update(self.config.SDMA_SEND_POSITION_LIST)
        if hasattr(self.config, "CDMA_SEND_POSITION_LIST"):
            self.rn_positions.update(self.config.CDMA_SEND_POSITION_LIST)
        if hasattr(self.config, "DDR_SEND_POSITION_LIST"):
            self.sn_positions.update(pos - self.config.NUM_COL for pos in self.config.DDR_SEND_POSITION_LIST)
            self.sn_positions.update(self.config.DDR_SEND_POSITION_LIST)
        if hasattr(self.config, "L2M_SEND_POSITION_LIST"):
            self.sn_positions.update(pos - self.config.NUM_COL for pos in self.config.L2M_SEND_POSITION_LIST)
            self.sn_positions.update(self.config.L2M_SEND_POSITION_LIST)

    def collect_ip_statistics(self):
        """从请求数据中动态统计IP信息"""
        self.unique_rn_ips.clear()
        self.unique_sn_ips.clear()
        self.ip_count_by_type.clear()

        for req in self.requests:
            # RN端：source_node + source_type
            self.unique_rn_ips.add((req.source_node, req.source_type))
            # SN端：dest_node + dest_type
            self.unique_sn_ips.add((req.dest_node, req.dest_type))

        # 统计各类型IP数量
        for node, ip_type in self.unique_rn_ips.union(self.unique_sn_ips):
            if ip_type is None:
                continue
            base_type = ip_type.split("_")[0] if "_" in ip_type else ip_type
            ip_id = ip_type.split("_")[1] if "_" in ip_type else "0"
            self.ip_count_by_type[base_type].add((node, ip_id))

        # 计算总的unique IP数量
        self.actual_num_ip = len(self.unique_rn_ips)

        # 更新rn_positions和sn_positions为实际使用的节点
        self.rn_positions = set(node for node, _ in self.unique_rn_ips)
        self.sn_positions = set(node for node, _ in self.unique_sn_ips)

    def normalize_ip_type(self, ip_type):
        """标准化IP类型名称，保留实例编号（如gdma_0, gdma_1）"""
        if not ip_type:
            return "l2m"

        ip_type = ip_type.lower()

        if ip_type == "unknown" or ip_type.startswith("unknown"):
            return "l2m"

        if ip_type.endswith("_ip"):
            ip_type = ip_type[:-3]

        base_type = ip_type.split("_")[0] if "_" in ip_type else ip_type
        supported_types = ["sdma", "gdma", "cdma", "ddr", "l2m", "d2d"]

        if base_type in supported_types or base_type.startswith("d2d"):
            return ip_type
        else:
            return "l2m"

    def collect_requests_data(self, sim_model, simulation_end_cycle=None) -> None:
        """从sim_model收集请求数据 - 委托给RequestCollector"""
        self.sim_model = sim_model
        self.requests = self.request_collector.collect_requests_data(sim_model, simulation_end_cycle)

        # 更新本地状态
        self.finish_cycle = max((req.data_received_complete_cycle for req in self.requests
                                if hasattr(req, 'data_received_complete_cycle')), default=0)

        # 收集RN带宽时间序列数据
        for req in self.requests:
            # 构建port_key
            if req.source_type and req.dest_type:
                port_key = f"{req.source_type[:-2].upper()} {req.req_type} {req.dest_type[:3].upper()}"
            else:
                source_backup = getattr(req, 'source_type', 'UNKNOWN') or "UNKNOWN"
                dest_backup = getattr(req, 'dest_type', 'UNKNOWN') or "UNKNOWN"
                port_key = f"{source_backup[:-2].upper() if len(source_backup) > 2 else source_backup.upper()} {req.req_type} {dest_backup[:3].upper()}"

            completion_time = req.rn_end_time
            self.rn_bandwidth_time_series[port_key]["time"].append(completion_time)
            self.rn_bandwidth_time_series[port_key]["start_times"].append(req.start_time)
            self.rn_bandwidth_time_series[port_key]["bytes"].append(req.burst_length * 128)

        # 动态统计IP信息
        self.collect_ip_statistics()

    def analyze_all_bandwidth(self) -> Dict:
        """执行完整的带宽分析"""
        if not self.requests:
            raise ValueError("没有请求数据，请先调用 collect_requests_data()")

        # 网络整体带宽分析
        network_overall = self.calculate_network_overall_bandwidth()

        # RN端口带宽分析
        rn_port_metrics = self.calculate_rn_port_bandwidth()

        # SN端口带宽分析
        sn_port_metrics = self.calculate_sn_port_bandwidth()

        # 计算端口平均带宽
        all_ports = {**rn_port_metrics, **sn_port_metrics}
        port_averages = self._calculate_port_bandwidth_averages(all_ports)

        # 汇总统计
        total_read_requests = len([r for r in self.requests if r.req_type == "read"])
        total_write_requests = len([r for r in self.requests if r.req_type == "write"])
        total_read_flits = sum(req.burst_length for req in self.requests if req.req_type == "read")
        total_write_flits = sum(req.burst_length for req in self.requests if req.req_type == "write")

        # 获取Circuit统计数据
        circuit_stats = {}
        if self.sim_model:
            circuit_stats = {
                "req_circuits_h": getattr(self.sim_model, "req_cir_h_num_stat", 0),
                "req_circuits_v": getattr(self.sim_model, "req_cir_v_num_stat", 0),
                "rsp_circuits_h": getattr(self.sim_model, "rsp_cir_h_num_stat", 0),
                "rsp_circuits_v": getattr(self.sim_model, "rsp_cir_v_num_stat", 0),
                "data_circuits_h": getattr(self.sim_model, "data_cir_h_num_stat", 0),
                "data_circuits_v": getattr(self.sim_model, "data_cir_v_num_stat", 0),
            }

        results = {
            "network_overall": network_overall,
            "rn_ports": rn_port_metrics,
            "sn_ports": sn_port_metrics,
            "port_averages": port_averages,
            "summary": {
                "total_requests": len(self.requests),
                "read_requests": total_read_requests,
                "write_requests": total_write_requests,
                "total_read_flits": total_read_flits,
                "total_write_flits": total_write_flits,
                "analysis_config": {
                    "min_gap_threshold_ns": self.min_gap_threshold,
                    "network_frequency_ghz": self.network_frequency
                },
                "circuit_stats": circuit_stats,
            },
        }

        # 计算绕环统计
        circling_eject_stats = self.circuit_collector.calculate_circling_eject_stats(self.requests)
        results["circling_eject_stats"] = circling_eject_stats

        # 计算保序绕环统计
        ordering_blocked_stats = self.circuit_collector.calculate_ordering_blocked_stats(self.requests)
        results["ordering_blocked_stats"] = ordering_blocked_stats

        # 计算延迟统计
        latency_stats = self.latency_collector.calculate_latency_stats(self.requests)
        results["latency_stats"] = latency_stats

        # 保存的图片路径列表
        saved_figures = []

        # 绘制RN带宽曲线并计算Total_sum_BW
        if self.plot_rn_bw_fig and self.sim_model:
            if self.sim_model.results_fig_save_path:
                import time, os
                rn_save_path = os.path.join(self.sim_model.results_fig_save_path,
                                          f"rn_bandwidth_{self.config.TOPO_TYPE}_{self.sim_model.file_name}_{time.time_ns()}.png")
            else:
                rn_save_path = None

            total_bandwidth = self.visualizer.plot_rn_bandwidth_curves_work_interval(
                rn_bandwidth_time_series=self.rn_bandwidth_time_series,
                network_frequency=self.network_frequency,
                save_path=rn_save_path
            )

            if rn_save_path:
                saved_figures.append(("RN带宽曲线", rn_save_path))
        else:
            total_bandwidth = network_overall.get("mixed", self._empty_metrics()).weighted_bandwidth

        results["Total_sum_BW"] = total_bandwidth
        results["summary"]["Total_sum_BW"] = total_bandwidth

        # 绘制流图
        if self.plot_flow_graph and self.sim_model:
            # 先计算IP带宽数据
            self.calculate_ip_bandwidth_data()

            if self.sim_model.results_fig_save_path:
                import time
                flow_fname = f"flow_graph_{self.config.TOPO_TYPE}_{self.sim_model.file_name}_{time.time_ns()}.png"
                flow_save_path = os.path.join(self.sim_model.results_fig_save_path, flow_fname)
            else:
                flow_save_path = None

            if self.sim_model.topo_type_stat.startswith("Ring"):
                self.flow_visualizer.draw_ring_flow_graph(
                    network=self.sim_model.data_network,
                    ip_bandwidth_data=self.ip_bandwidth_data,
                    config=self.config,
                    save_path=flow_save_path
                )
            else:
                self.flow_visualizer.draw_flow_graph(
                    network=self.sim_model.data_network,
                    ip_bandwidth_data=self.ip_bandwidth_data,
                    config=self.config,
                    mode="total",
                    save_path=flow_save_path
                )

            if flow_save_path:
                saved_figures.append(("流图", flow_save_path))

        # 保存图片路径到results中,供后续打印使用
        results["saved_figures"] = saved_figures

        # 控制台打印摘要信息
        if self.sim_model and hasattr(self.sim_model, "verbose") and self.sim_model.verbose:
            self._print_summary_to_console(results)

        return results

    def calculate_network_overall_bandwidth(self) -> Dict[str, BandwidthMetrics]:
        """计算网络整体带宽"""
        read_requests = [r for r in self.requests if r.req_type == "read"]
        write_requests = [r for r in self.requests if r.req_type == "write"]

        result = {
            "read": self.calculator.calculate_bandwidth_metrics(read_requests, "read") if read_requests else self._empty_metrics(),
            "write": self.calculator.calculate_bandwidth_metrics(write_requests, "write") if write_requests else self._empty_metrics(),
            "mixed": self.calculator.calculate_bandwidth_metrics(self.requests, None) if self.requests else self._empty_metrics()
        }

        return result

    def calculate_rn_port_bandwidth(self) -> Dict[str, PortBandwidthMetrics]:
        """计算RN端口带宽"""
        port_metrics = {}

        for source_node in self.rn_positions:
            node_requests = [r for r in self.requests if r.source_node == source_node]
            if not node_requests:
                continue

            # 按source_type分组
            from collections import defaultdict
            by_type = defaultdict(list)
            for req in node_requests:
                source_type = self.normalize_ip_type(req.source_type)
                by_type[source_type].append(req)

            for source_type, type_requests in by_type.items():
                port_id = f"{source_type}_{source_node}"

                read_reqs = [r for r in type_requests if r.req_type == "read"]
                write_reqs = [r for r in type_requests if r.req_type == "write"]

                read_metrics = self.calculator.calculate_bandwidth_metrics(read_reqs, "read") if read_reqs else self._empty_metrics()
                write_metrics = self.calculator.calculate_bandwidth_metrics(write_reqs, "write") if write_reqs else self._empty_metrics()
                mixed_metrics = self.calculator.calculate_bandwidth_metrics(type_requests, None)

                port_metrics[port_id] = PortBandwidthMetrics(
                    port_id=port_id,
                    read_metrics=read_metrics,
                    write_metrics=write_metrics,
                    mixed_metrics=mixed_metrics
                )

        return port_metrics

    def calculate_sn_port_bandwidth(self) -> Dict[str, PortBandwidthMetrics]:
        """计算SN端口带宽"""
        port_metrics = {}

        for dest_node in self.sn_positions:
            node_requests = [r for r in self.requests if r.dest_node == dest_node]
            if not node_requests:
                continue

            # 按dest_type分组
            from collections import defaultdict
            by_type = defaultdict(list)
            for req in node_requests:
                dest_type = self.normalize_ip_type(req.dest_type)
                by_type[dest_type].append(req)

            for dest_type, type_requests in by_type.items():
                port_id = f"{dest_type}_{dest_node}"

                read_reqs = [r for r in type_requests if r.req_type == "read"]
                write_reqs = [r for r in type_requests if r.req_type == "write"]

                read_metrics = self.calculator.calculate_bandwidth_metrics(read_reqs, "read") if read_reqs else self._empty_metrics()
                write_metrics = self.calculator.calculate_bandwidth_metrics(write_reqs, "write") if write_reqs else self._empty_metrics()
                mixed_metrics = self.calculator.calculate_bandwidth_metrics(type_requests, None)

                port_metrics[port_id] = PortBandwidthMetrics(
                    port_id=port_id,
                    read_metrics=read_metrics,
                    write_metrics=write_metrics,
                    mixed_metrics=mixed_metrics
                )

        return port_metrics

    def _empty_metrics(self) -> BandwidthMetrics:
        """返回空的BandwidthMetrics"""
        return BandwidthMetrics(
            unweighted_bandwidth=0.0,
            weighted_bandwidth=0.0,
            working_intervals=[],
            total_working_time=0,
            network_start_time=0,
            network_end_time=0,
            total_bytes=0,
            total_requests=0
        )

    def _calculate_port_bandwidth_averages(self, all_ports: Dict[str, PortBandwidthMetrics]) -> Dict[str, float]:
        """计算每种端口类型的平均带宽"""
        from collections import defaultdict
        port_bw_groups = defaultdict(lambda: {"read": [], "write": [], "mixed": []})

        for port_id, metrics in all_ports.items():
            port_type = port_id.split("_")[0]
            port_bw_groups[port_type]["read"].append(metrics.read_metrics.weighted_bandwidth)
            port_bw_groups[port_type]["write"].append(metrics.write_metrics.weighted_bandwidth)
            port_bw_groups[port_type]["mixed"].append(metrics.mixed_metrics.weighted_bandwidth)

        avg_port_metrics = {}
        for port_type, bw_dict in port_bw_groups.items():
            if bw_dict["read"]:
                avg_port_metrics[f"avg_{port_type}_read_bw"] = sum(bw_dict["read"]) / len(bw_dict["read"])
            if bw_dict["write"]:
                avg_port_metrics[f"avg_{port_type}_write_bw"] = sum(bw_dict["write"]) / len(bw_dict["write"])
            if bw_dict["mixed"]:
                avg_port_metrics[f"avg_{port_type}_bw"] = sum(bw_dict["mixed"]) / len(bw_dict["mixed"])

        return avg_port_metrics

    def calculate_ip_bandwidth_data(self):
        """计算IP带宽数据矩阵 - 支持区分IP实例"""
        import numpy as np
        from collections import defaultdict

        rows = self.config.NUM_ROW
        cols = self.config.NUM_COL
        if hasattr(self, "sim_model") and self.sim_model and getattr(self.sim_model, "topo_type_stat", "").startswith("Ring"):
            rows = self.config.RING_NUM_NODE // 2
            cols = 2

        # 收集所有IP实例名称
        all_ip_instances = set()
        for req in self.requests:
            if req.source_type:
                source_type = self.normalize_ip_type(req.source_type)
                all_ip_instances.add(source_type)
            if req.dest_type:
                dest_type = self.normalize_ip_type(req.dest_type)
                all_ip_instances.add(dest_type)

        # 初始化数据结构
        self.ip_bandwidth_data = {
            "read": {},
            "write": {},
            "total": {},
        }

        for ip_instance in all_ip_instances:
            self.ip_bandwidth_data["read"][ip_instance] = np.zeros((rows, cols))
            self.ip_bandwidth_data["write"][ip_instance] = np.zeros((rows, cols))
            self.ip_bandwidth_data["total"][ip_instance] = np.zeros((rows, cols))

        # 处理RN端口
        rn_requests_by_source = defaultdict(list)
        for req in self.requests:
            if req.source_node in self.rn_positions:
                rn_requests_by_source[req.source_node].append(req)

        for source_node, node_requests in rn_requests_by_source.items():
            by_type = defaultdict(list)
            for req in node_requests:
                source_type = self.normalize_ip_type(req.source_type)
                by_type[source_type].append(req)

            # 计算物理位置
            if hasattr(self, "sim_model") and self.sim_model and getattr(self.sim_model, "topo_type_stat", "").startswith("Ring"):
                if source_node < rows:
                    physical_col = 0
                    physical_row = source_node
                else:
                    physical_col = 1
                    physical_row = self.config.RING_NUM_NODE - 1 - source_node
            else:
                physical_col = source_node % cols
                physical_row = source_node // cols

            for source_type, type_requests in by_type.items():
                read_requests = [req for req in type_requests if req.req_type == "read"]
                write_requests = [req for req in type_requests if req.req_type == "write"]

                if read_requests:
                    read_metrics = self.calculator.calculate_bandwidth_metrics(read_requests, "read")
                    self.ip_bandwidth_data["read"][source_type][physical_row, physical_col] += read_metrics.weighted_bandwidth

                if write_requests:
                    write_metrics = self.calculator.calculate_bandwidth_metrics(write_requests, "write")
                    self.ip_bandwidth_data["write"][source_type][physical_row, physical_col] += write_metrics.weighted_bandwidth

                if type_requests:
                    total_metrics = self.calculator.calculate_bandwidth_metrics(type_requests, None)
                    self.ip_bandwidth_data["total"][source_type][physical_row, physical_col] += total_metrics.weighted_bandwidth

        # 处理SN端口
        sn_requests_by_dest = defaultdict(list)
        for req in self.requests:
            if req.dest_node in self.sn_positions:
                sn_requests_by_dest[req.dest_node].append(req)

        for dest_node, node_requests in sn_requests_by_dest.items():
            by_type = defaultdict(list)
            for req in node_requests:
                dest_type = self.normalize_ip_type(req.dest_type)
                by_type[dest_type].append(req)

            # 计算物理位置
            if hasattr(self, "sim_model") and self.sim_model and getattr(self.sim_model, "topo_type_stat", "").startswith("Ring"):
                if dest_node < rows:
                    physical_col = 0
                    physical_row = dest_node
                else:
                    physical_col = 1
                    physical_row = self.config.RING_NUM_NODE - 1 - dest_node
            else:
                physical_col = dest_node % cols
                physical_row = dest_node // cols

            for dest_type, type_requests in by_type.items():
                read_requests = [req for req in type_requests if req.req_type == "read"]
                write_requests = [req for req in type_requests if req.req_type == "write"]

                if read_requests:
                    read_metrics = self.calculator.calculate_bandwidth_metrics(read_requests, "read")
                    self.ip_bandwidth_data["read"][dest_type][physical_row, physical_col] += read_metrics.weighted_bandwidth

                if write_requests:
                    write_metrics = self.calculator.calculate_bandwidth_metrics(write_requests, "write")
                    self.ip_bandwidth_data["write"][dest_type][physical_row, physical_col] += write_metrics.weighted_bandwidth

                if type_requests:
                    total_metrics = self.calculator.calculate_bandwidth_metrics(type_requests, None)
                    self.ip_bandwidth_data["total"][dest_type][physical_row, physical_col] += total_metrics.weighted_bandwidth

    def generate_unified_report(self, results: Dict, output_path: str) -> None:
        """
        生成统一的带宽分析报告 (委托给ReportGenerator和CSVExporter)

        Args:
            results: analyze_all_bandwidth()的返回结果
            output_path: 输出目录路径
        """
        # 生成文本报告
        self.report_generator.generate_unified_report(
            results=results,
            output_path=output_path,
            num_ip=self.actual_num_ip if hasattr(self, 'actual_num_ip') else 1
        )

        # 生成详细请求CSV
        self.exporter.generate_detailed_request_csv(
            requests=self.requests,
            output_path=output_path
        )

        # 生成端口CSV
        self.exporter.generate_ports_csv(
            rn_ports=results.get("rn_ports"),
            sn_ports=results.get("sn_ports"),
            output_path=output_path,
            config=self.config
        )

        # 打印文件生成提示
        if self.sim_model and hasattr(self.sim_model, "verbose") and self.sim_model.verbose:
            import os
            report_file = os.path.join(output_path, "bandwidth_analysis_report.txt")
            print(f"带宽分析报告： {report_file}")
            print(f"具体端口的统计CSV： {output_path}ports_bandwidth.csv")
            if hasattr(self.sim_model, "data_network") and self.sim_model.data_network:
                print(f"链路统计CSV： {output_path}link_statistics.csv")

            # 打印保存的图片路径
            saved_figures = results.get("saved_figures", [])
            for fig_name, fig_path in saved_figures:
                print(f"{fig_name}已保存: {fig_path}")

    def generate_fifo_usage_csv(self, sim_model):
        """
        生成FIFO使用率CSV (委托给CircuitStatsCollector)

        Args:
            sim_model: 仿真模型对象
        """
        if hasattr(self.circuit_collector, 'generate_fifo_usage_csv'):
            # output_path可以是None,让generate_fifo_usage_csv自己构建路径
            self.circuit_collector.generate_fifo_usage_csv(
                model=sim_model,
                output_path=None  # 让方法自己构建文件路径
            )

    def _calculate_latency_stats(self):
        """
        计算延迟统计 (委托给LatencyStatsCollector)

        Returns:
            延迟统计字典
        """
        return self.latency_collector.calculate_latency_stats(self.requests)

    def _print_summary_to_console(self, results: Dict) -> None:
        """输出重要数据到控制台"""
        print("\n" + "=" * 60)
        print("网络带宽分析结果摘要")
        print("=" * 60)

        # 网络整体带宽
        read_metrics = results["network_overall"]["read"]
        write_metrics = results["network_overall"]["write"]
        mixed_metrics = results["network_overall"]["mixed"]

        # 使用实际IP数量
        num_ip_for_avg = self.actual_num_ip or 1

        print(f"网络带宽:")
        print(f"  读带宽:    {read_metrics.weighted_bandwidth:.3f} GB/s (平均: {read_metrics.weighted_bandwidth / num_ip_for_avg:.3f} GB/s)")
        print(f"  写带宽:    {write_metrics.weighted_bandwidth:.3f} GB/s (平均: {write_metrics.weighted_bandwidth / num_ip_for_avg:.3f} GB/s)")
        print(f"  混合带宽:  {mixed_metrics.weighted_bandwidth:.3f} GB/s (平均: {mixed_metrics.weighted_bandwidth / num_ip_for_avg:.3f} GB/s)")

        # 请求统计
        summary = results["summary"]
        print(f"\n请求统计:")
        print(f"  总请求数: {summary['total_requests']} (读: {summary['read_requests']}, 写: {summary['write_requests']})")
        print(f"  总flit数: {summary['total_read_flits'] + summary['total_write_flits']} (读: {summary['total_read_flits']}, 写: {summary['total_write_flits']})")

        # Circuit统计
        circuit_stats = summary.get("circuit_stats", {})
        print(f"\n绕环与Tag统计:")
        if circuit_stats:
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

        # 绕环比例统计
        circling_stats = results.get("circling_eject_stats", {})
        if circling_stats:
            h_ratio = circling_stats["horizontal"]["circling_ratio"]
            v_ratio = circling_stats["vertical"]["circling_ratio"]
            overall_ratio = circling_stats["overall"]["circling_ratio"]
            print(f"  绕环比例: H: {h_ratio*100:.2f}%, V: {v_ratio*100:.2f}%, Overall: {overall_ratio*100:.2f}%")

        # 保序导致的绕环比例统计
        ordering_blocked_stats = results.get("ordering_blocked_stats", {})
        if ordering_blocked_stats:
            h_ratio = ordering_blocked_stats["horizontal"]["ordering_blocked_ratio"]
            v_ratio = ordering_blocked_stats["vertical"]["ordering_blocked_ratio"]
            overall_ratio = ordering_blocked_stats["overall"]["ordering_blocked_ratio"]
            print(f"  保序导致绕环比例: H: {h_ratio*100:.2f}%, V: {v_ratio*100:.2f}%, Overall: {overall_ratio*100:.2f}%")

        # 工作区间统计
        print(f"\n工作区间统计:")
        print(f"  读操作工作区间: {len(read_metrics.working_intervals)}")
        print(f"  写操作工作区间: {len(write_metrics.working_intervals)}")
        print(f"  混合操作工作区间: {len(mixed_metrics.working_intervals)}")

        print("=" * 60)

        # 延迟统计
        latency_stats = results.get("latency_stats", {})
        if latency_stats:
            def _avg(cat, op):
                s = latency_stats[cat][op]
                return s["sum"] / s["count"] if s["count"] else 0.0

            print("\n延迟统计 (单位: cycle)")
            for key, label in [("cmd", "CMD"), ("data", "Data"), ("trans", "Trans")]:
                if key in latency_stats:
                    print(
                        f"  {label} 延迟  - "
                        f"读: avg {_avg(key,'read'):.2f}, max {latency_stats[key]['read']['max']}；"
                        f"写: avg {_avg(key,'write'):.2f}, max {latency_stats[key]['write']['max']}；"
                        f"混合: avg {_avg(key,'mixed'):.2f}, max {latency_stats[key]['mixed']['max']}"
                    )


class D2DAnalyzer:
    """
    D2D跨Die分析器

    功能：
    1. 收集跨Die请求数据
    2. 统计Die间带宽
    3. 分析D2D延迟
    4. 生成D2D报告
    """

    def __init__(
        self,
        config,
        min_gap_threshold: int = 50,
    ):
        """
        初始化D2D分析器

        Args:
            config: 网络配置对象
            min_gap_threshold: 工作区间合并阈值(ns)
        """
        from collections import defaultdict
        from .core_calculators import DataValidator, TimeIntervalCalculator, BandwidthCalculator
        from .data_collectors import RequestCollector, LatencyStatsCollector, CircuitStatsCollector
        from .visualizers import BandwidthPlotter, FlowGraphRenderer
        from .exporters import CSVExporter, ReportGenerator, JSONExporter

        self.config = config
        self.min_gap_threshold = min_gap_threshold
        self.network_frequency = getattr(config, "NETWORK_FREQUENCY", 2)

        # D2D数据存储
        self.d2d_requests: List[D2DRequestInfo] = []
        self.d2d_stats = D2DBandwidthStats()

        # 初始化组件
        self.validator = DataValidator()
        self.interval_calculator = TimeIntervalCalculator(min_gap_threshold)
        self.calculator = BandwidthCalculator(self.interval_calculator)
        self.request_collector = RequestCollector(self.network_frequency)
        self.latency_collector = LatencyStatsCollector()
        self.circuit_collector = CircuitStatsCollector()
        self.visualizer = BandwidthPlotter()
        self.flow_visualizer = FlowGraphRenderer()
        self.exporter = CSVExporter()
        self.report_generator = ReportGenerator()
        self.json_exporter = JSONExporter()

    def collect_cross_die_requests(self, dies: Dict) -> None:
        """从多Die系统收集跨Die请求数据"""
        self.d2d_requests = self.request_collector.collect_cross_die_requests(dies)

    def calculate_d2d_bandwidth(self) -> D2DBandwidthStats:
        """计算D2D带宽统计"""
        if not self.d2d_requests:
            return self.d2d_stats

        # 按Die对分组
        from collections import defaultdict
        read_by_pair = defaultdict(list)
        write_by_pair = defaultdict(list)

        for req in self.d2d_requests:
            pair_key = (req.source_die, req.target_die)
            if req.req_type == "read":
                read_by_pair[pair_key].append(req)
            else:
                write_by_pair[pair_key].append(req)

        # 计算每对Die之间的带宽
        for pair_key, requests in read_by_pair.items():
            if requests:
                # 将D2DRequestInfo转换为RequestInfo以兼容带宽计算器
                request_infos = self._convert_d2d_to_request_info(requests)
                metrics = self.calculator.calculate_bandwidth_metrics(request_infos, "read")
                self.d2d_stats.pair_read_bw[pair_key] = (
                    metrics.unweighted_bandwidth,
                    metrics.weighted_bandwidth
                )
                self.d2d_stats.total_read_requests += len(requests)

        for pair_key, requests in write_by_pair.items():
            if requests:
                request_infos = self._convert_d2d_to_request_info(requests)
                metrics = self.calculator.calculate_bandwidth_metrics(request_infos, "write")
                self.d2d_stats.pair_write_bw[pair_key] = (
                    metrics.unweighted_bandwidth,
                    metrics.weighted_bandwidth
                )
                self.d2d_stats.total_write_requests += len(requests)

        # 计算总字节数
        self.d2d_stats.total_bytes_transferred = sum(req.data_bytes for req in self.d2d_requests)

        return self.d2d_stats

    def _convert_d2d_to_request_info(self, d2d_requests: List[D2DRequestInfo]) -> List[RequestInfo]:
        """将D2DRequestInfo转换为RequestInfo以兼容带宽计算器"""
        request_infos = []
        for d2d_req in d2d_requests:
            req_info = RequestInfo(
                packet_id=d2d_req.packet_id,
                start_time=d2d_req.start_time_ns,
                end_time=d2d_req.end_time_ns,
                rn_end_time=d2d_req.end_time_ns,
                sn_end_time=d2d_req.end_time_ns,
                req_type=d2d_req.req_type,
                source_node=d2d_req.source_node,
                dest_node=d2d_req.target_node,
                source_type=d2d_req.source_type,
                dest_type=d2d_req.target_type,
                burst_length=d2d_req.burst_length,
                total_bytes=d2d_req.data_bytes,
                cmd_latency=d2d_req.cmd_latency_ns,
                data_latency=d2d_req.data_latency_ns,
                transaction_latency=d2d_req.transaction_latency_ns,
            )
            request_infos.append(req_info)
        return request_infos

    def process_d2d_results(self, dies: Dict, output_path: str):
        """
        处理D2D结果的主入口函数

        Args:
            dies: Die字典
            output_path: 输出路径
        """
        # 收集跨Die请求
        self.collect_cross_die_requests(dies)

        if not self.d2d_requests:
            print("没有检测到跨Die流量")
            return

        # 计算D2D带宽
        self.d2d_stats = self.calculate_d2d_bandwidth()

        # 计算延迟统计
        latency_stats = self.latency_collector.calculate_latency_stats(
            self._convert_d2d_to_request_info(self.d2d_requests)
        )

        # 导出结果
        if output_path:
            import os
            import json

            # 导出D2D带宽统计
            bw_stats = {
                "pair_read_bw": {f"{k[0]}->{k[1]}": v for k, v in self.d2d_stats.pair_read_bw.items()},
                "pair_write_bw": {f"{k[0]}->{k[1]}": v for k, v in self.d2d_stats.pair_write_bw.items()},
                "total_read_requests": self.d2d_stats.total_read_requests,
                "total_write_requests": self.d2d_stats.total_write_requests,
                "total_bytes_transferred": self.d2d_stats.total_bytes_transferred,
            }

            bw_file = os.path.join(output_path, "d2d_bandwidth_stats.json")
            with open(bw_file, 'w', encoding='utf-8') as f:
                json.dump(bw_stats, f, indent=2, ensure_ascii=False)

            # 导出延迟统计
            latency_file = os.path.join(output_path, "d2d_latency_stats.json")
            with open(latency_file, 'w', encoding='utf-8') as f:
                json.dump(latency_stats, f, indent=2, ensure_ascii=False)

            print(f"D2D分析结果已保存到: {output_path}")

    def calculate_d2d_working_intervals(self, requests: List[D2DRequestInfo]) -> List[WorkingInterval]:
        """计算D2D工作区间"""
        request_infos = self._convert_d2d_to_request_info(requests)
        return self.interval_calculator.calculate_working_intervals(request_infos)

    def calculate_d2d_ip_bandwidth_data(self, dies: Dict):
        """
        基于D2D请求计算IP带宽数据 - 动态收集IP实例

        Args:
            dies: Die模型字典
        """
        import numpy as np
        from collections import defaultdict

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

            # 收集D2D_SN（跨Die请求）
            if request.source_die != request.target_die and request.d2d_sn_node is not None:
                if request.source_die in dies:
                    die_ip_instances[request.source_die].add("d2d_sn")

            # 收集D2D_RN（跨Die请求）
            if request.source_die != request.target_die and request.d2d_rn_node is not None:
                if request.target_die in dies:
                    die_ip_instances[request.target_die].add("d2d_rn")

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

        # 第五步：处理D2D_SN节点带宽（所有跨Die请求都经过D2D_SN）
        d2d_sn_groups = defaultdict(list)
        for request in self.d2d_requests:
            # 只处理跨Die请求，且d2d_sn_node不为None
            if request.source_die != request.target_die and request.d2d_sn_node is not None:
                # D2D_SN节点在源Die上
                key = (request.source_die, request.d2d_sn_node, "d2d_sn")
                d2d_sn_groups[key].append(request)

        for (die_id, node, ip_type), requests in d2d_sn_groups.items():
            if die_id not in dies:
                continue
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

        # 第六步：处理D2D_RN节点带宽（所有跨Die请求都经过D2D_RN）
        d2d_rn_groups = defaultdict(list)
        for request in self.d2d_requests:
            # 只处理跨Die请求，且d2d_rn_node不为None
            if request.source_die != request.target_die and request.d2d_rn_node is not None:
                # D2D_RN节点在目标Die上
                key = (request.target_die, request.d2d_rn_node, "d2d_rn")
                d2d_rn_groups[key].append(request)

        for (die_id, node, ip_type), requests in d2d_rn_groups.items():
            if die_id not in dies:
                continue
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

    def _get_physical_position(self, node: int, die_model) -> tuple:
        """获取节点的物理行列位置"""
        cols = die_model.config.NUM_COL
        row = node // cols
        col = node % cols
        return row, col

    def _calculate_bandwidth_for_group(self, requests: list) -> tuple:
        """计算请求组的带宽（非加权和加权）"""
        if not requests:
            return 0.0, 0.0

        # 转换D2DRequestInfo到RequestInfo以便使用interval_calculator
        converted_requests = []
        for r in requests:
            # 检查是否是D2DRequestInfo
            if hasattr(r, 'start_time_ns'):
                # 创建一个简单的对象模拟RequestInfo
                class _TempRequest:
                    def __init__(self, d2d_req):
                        self.packet_id = d2d_req.packet_id
                        self.start_time = d2d_req.start_time_ns
                        self.end_time = d2d_req.end_time_ns
                        self.total_bytes = d2d_req.data_bytes
                        self.burst_length = d2d_req.burst_length
                converted_requests.append(_TempRequest(r))
            else:
                converted_requests.append(r)

        # 计算工作区间
        working_intervals = self.interval_calculator.calculate_working_intervals(converted_requests)

        # 计算总字节数 (使用原始requests)
        total_bytes = sum(getattr(r, 'data_bytes', 0) or getattr(r, 'total_bytes', 0) for r in requests)

        # 计算总工作时间
        total_working_time = sum((interval.end_time - interval.start_time) for interval in working_intervals)

        # 非加权带宽
        if total_working_time > 0:
            unweighted_bandwidth = total_bytes / total_working_time  # GB/s
        else:
            unweighted_bandwidth = 0.0

        # 加权带宽
        if working_intervals:
            network_start = min(interval.start_time for interval in working_intervals)
            network_end = max(interval.end_time for interval in working_intervals)
            total_time = network_end - network_start
            if total_time > 0:
                weighted_bandwidth = total_bytes / total_time
            else:
                weighted_bandwidth = 0.0
        else:
            weighted_bandwidth = 0.0

        return unweighted_bandwidth, weighted_bandwidth

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
        """
        mapping = {}
        total_nodes = rows * cols

        for node_id in range(total_nodes):
            row = node_id // cols
            col = node_id % cols

            if rotation == 0:
                new_row, new_col = row, col
                new_cols = cols
            elif rotation == 90:
                new_row = col
                new_col = rows - 1 - row
                new_cols = rows
            elif rotation == 180:
                new_row = rows - 1 - row
                new_col = cols - 1 - col
                new_cols = cols
            elif rotation == 270:
                new_row = cols - 1 - col
                new_col = row
                new_cols = rows
            else:
                raise ValueError(f"不支持的旋转角度: {rotation}，只支持0/90/180/270")

            new_node_id = new_row * new_cols + new_col
            mapping[node_id] = new_node_id

        return mapping

    def _normalize_d2d_ip_type(self, ip_type: str) -> str:
        """规范化D2D IP类型名称"""
        if not ip_type:
            return "unknown"

        ip_type = ip_type.lower()

        if ip_type.endswith("_ip"):
            ip_type = ip_type[:-3]

        # D2D特殊处理
        if ip_type.startswith("d2d"):
            return ip_type

        base_type = ip_type.split("_")[0] if "_" in ip_type else ip_type
        supported_types = ["sdma", "gdma", "cdma", "ddr", "l2m"]

        return base_type if base_type in supported_types else "other"

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
        import os
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

    def _calculate_d2d_latency_stats(self):
        """计算D2D请求的延迟统计数据（cmd/data/transaction）"""
        import numpy as np

        stats = {
            "cmd": {"read": {"sum": 0, "max": 0, "count": 0}, "write": {"sum": 0, "max": 0, "count": 0}, "mixed": {"sum": 0, "max": 0, "count": 0}},
            "data": {"read": {"sum": 0, "max": 0, "count": 0}, "write": {"sum": 0, "max": 0, "count": 0}, "mixed": {"sum": 0, "max": 0, "count": 0}},
            "trans": {"read": {"sum": 0, "max": 0, "count": 0}, "write": {"sum": 0, "max": 0, "count": 0}, "mixed": {"sum": 0, "max": 0, "count": 0}},
        }

        for req in self.d2d_requests:
            req_type = req.req_type
            for lat_type in ["cmd", "data", "trans"]:
                lat_attr = f"{lat_type}_latency"
                if hasattr(req, lat_attr):
                    lat_val = getattr(req, lat_attr)
                    if not np.isinf(lat_val):
                        stats[lat_type][req_type]["sum"] += lat_val
                        stats[lat_type][req_type]["max"] = max(stats[lat_type][req_type]["max"], lat_val)
                        stats[lat_type][req_type]["count"] += 1
                        stats[lat_type]["mixed"]["sum"] += lat_val
                        stats[lat_type]["mixed"]["max"] = max(stats[lat_type]["mixed"]["max"], lat_val)
                        stats[lat_type]["mixed"]["count"] += 1

        return stats

    def _collect_d2d_circuit_stats(self, dies: Dict):
        """
        从各Die收集绕环和Tag统计数据

        Args:
            dies: Die字典 {die_id: BaseModel}

        Returns:
            {"per_die": {die_id: stats_dict}, "summary": aggregated_stats}
        """
        per_die_stats = {}
        summary_stats = {
            "circuits_req_h": 0,
            "circuits_req_v": 0,
            "circuits_rsp_h": 0,
            "circuits_rsp_v": 0,
            "circuits_data_h": 0,
            "circuits_data_v": 0,
            "wait_cycle_req_h": 0,
            "wait_cycle_req_v": 0,
            "wait_cycle_rsp_h": 0,
            "wait_cycle_rsp_v": 0,
            "wait_cycle_data_h": 0,
            "wait_cycle_data_v": 0,
            "RB_ETag_T1_num": 0,
            "RB_ETag_T0_num": 0,
            "EQ_ETag_T1_num": 0,
            "EQ_ETag_T0_num": 0,
            "ITag_h_num": 0,
            "ITag_v_num": 0,
            "read_retry_num": 0,
            "write_retry_num": 0,
            "circling_ratio": {
                "horizontal": {"circling_flits": 0, "total_flits": 0, "circling_ratio": 0.0},
                "vertical": {"circling_flits": 0, "total_flits": 0, "circling_ratio": 0.0},
                "overall": {"circling_flits": 0, "total_flits": 0, "circling_ratio": 0.0},
            },
        }

        # 从每个Die收集统计数据
        for die_id, die_model in dies.items():
            if hasattr(die_model, "circuit_stats"):
                die_stats = die_model.circuit_stats.copy()
                per_die_stats[die_id] = die_stats

                # 汇总到summary
                for key in summary_stats.keys():
                    if key == "circling_ratio":
                        for direction in ["horizontal", "vertical", "overall"]:
                            if direction in die_stats.get("circling_ratio", {}):
                                summary_stats["circling_ratio"][direction]["circling_flits"] += die_stats["circling_ratio"][direction]["circling_flits"]
                                summary_stats["circling_ratio"][direction]["total_flits"] += die_stats["circling_ratio"][direction]["total_flits"]
                    elif key in die_stats:
                        summary_stats[key] += die_stats[key]

        # 计算汇总的绕环比例
        for direction in ["horizontal", "vertical", "overall"]:
            total = summary_stats["circling_ratio"][direction]["total_flits"]
            circling = summary_stats["circling_ratio"][direction]["circling_flits"]
            summary_stats["circling_ratio"][direction]["circling_ratio"] = circling / total if total > 0 else 0.0

        return {"per_die": per_die_stats, "summary": summary_stats}

    @staticmethod
    def _format_circuit_stats(stats: Dict, prefix: str = "  ") -> List[str]:
        """
        格式化绕环统计数据为文本行

        Args:
            stats: 统计数据字典
            prefix: 每行的前缀

        Returns:
            格式化的文本行列表
        """
        lines = []
        lines.append(f"{prefix}Circuits req  - h: {stats.get('circuits_req_h', 0)}, v: {stats.get('circuits_req_v', 0)}")
        lines.append(f"{prefix}Circuits rsp  - h: {stats.get('circuits_rsp_h', 0)}, v: {stats.get('circuits_rsp_v', 0)}")
        lines.append(f"{prefix}Circuits data - h: {stats.get('circuits_data_h', 0)}, v: {stats.get('circuits_data_v', 0)}")
        lines.append(f"{prefix}Wait cycle req  - h: {stats.get('wait_cycle_req_h', 0)}, v: {stats.get('wait_cycle_req_v', 0)}")
        lines.append(f"{prefix}Wait cycle rsp  - h: {stats.get('wait_cycle_rsp_h', 0)}, v: {stats.get('wait_cycle_rsp_v', 0)}")
        lines.append(f"{prefix}Wait cycle data - h: {stats.get('wait_cycle_data_h', 0)}, v: {stats.get('wait_cycle_data_v', 0)}")
        lines.append(f"{prefix}RB ETag - T1: {stats.get('RB_ETag_T1_num', 0)}, T0: {stats.get('RB_ETag_T0_num', 0)}")
        lines.append(f"{prefix}EQ ETag - T1: {stats.get('EQ_ETag_T1_num', 0)}, T0: {stats.get('EQ_ETag_T0_num', 0)}")
        lines.append(f"{prefix}ITag - h: {stats.get('ITag_h_num', 0)}, v: {stats.get('ITag_v_num', 0)}")
        lines.append(f"{prefix}Retry - read: {stats.get('read_retry_num', 0)}, write: {stats.get('write_retry_num', 0)}")

        if "circling_ratio" in stats:
            h_ratio = stats["circling_ratio"]["horizontal"]["circling_ratio"]
            v_ratio = stats["circling_ratio"]["vertical"]["circling_ratio"]
            overall_ratio = stats["circling_ratio"]["overall"]["circling_ratio"]
            lines.append(f"{prefix}绕环比例: H: {h_ratio*100:.2f}%, V: {v_ratio*100:.2f}%, Overall: {overall_ratio*100:.2f}%")

        return lines

    def save_d2d_requests_csv(self, output_path: str):
        """
        保存D2D请求到CSV文件（委托给CSVExporter）

        Args:
            output_path: 输出目录路径
        """
        self.exporter.save_d2d_requests_csv(
            d2d_requests=self.d2d_requests,
            output_path=output_path
        )

    def save_ip_bandwidth_to_csv(self, output_path: str, die_ip_bandwidth_data: Dict = None, config=None):
        """
        保存IP带宽数据到CSV（委托给CSVExporter）

        Args:
            output_path: 输出目录路径
            die_ip_bandwidth_data: Die IP带宽数据（可选，如果不提供则使用self.die_ip_bandwidth_data）
            config: 配置对象（可选，如果不提供则使用self.config）
        """
        # 如果没有提供数据，尝试从实例属性获取
        if die_ip_bandwidth_data is None:
            die_ip_bandwidth_data = getattr(self, 'die_ip_bandwidth_data', None)

        if config is None:
            config = self.config

        # 如果还是没有数据，直接返回
        if not die_ip_bandwidth_data:
            return

        self.exporter.save_ip_bandwidth_to_csv(
            die_ip_bandwidth_data=die_ip_bandwidth_data,
            output_path=output_path,
            config=config
        )

    def draw_d2d_flow_graph(
        self,
        die_networks: Dict = None,
        dies: Dict = None,
        config = None,
        die_ip_bandwidth_data: Dict = None,
        mode: str = "total",
        save_path: str = None
    ):
        """
        绘制D2D流图（委托给FlowGraphRenderer）

        Args:
            die_networks: Die网络字典
            dies: Die模型字典
            config: 配置对象
            die_ip_bandwidth_data: Die IP带宽数据
            mode: 显示模式
            save_path: 保存路径
        """
        self.flow_visualizer.draw_d2d_flow_graph(
            die_networks=die_networks,
            dies=dies,
            config=config,
            die_ip_bandwidth_data=die_ip_bandwidth_data if die_ip_bandwidth_data else self.die_ip_bandwidth_data if hasattr(self, 'die_ip_bandwidth_data') else None,
            mode=mode,
            save_path=save_path
        )

    def draw_ip_bandwidth_heatmap(self, dies=None, config=None, mode="total", node_size=4000, save_path=None):
        """
        绘制IP带宽热力图(委托给HeatmapVisualizer)

        Args:
            dies: Die模型字典
            config: 配置对象
            mode: 显示模式 (read/write/total)
            node_size: 节点大小
            save_path: 保存路径
        """
        # 检查是否有die_ip_bandwidth_data属性
        if not hasattr(self, "die_ip_bandwidth_data") or not self.die_ip_bandwidth_data:
            print("警告: 没有die_ip_bandwidth_data数据，跳过IP带宽热力图绘制")
            return None

        # 委托给HeatmapVisualizer处理
        if hasattr(self, 'heatmap_visualizer'):
            return self.heatmap_visualizer.draw_d2d_ip_bandwidth_heatmap(
                die_ip_bandwidth_data=self.die_ip_bandwidth_data,
                dies=dies,
                config=config,
                mode=mode,
                node_size=node_size,
                save_path=save_path
            )
        else:
            print("警告: HeatmapVisualizer未初始化，跳过IP带宽热力图绘制")
            return None
