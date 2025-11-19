"""
分析器模块 - 包含单Die分析器和基础数据类

提供:
1. 数据类定义 (RequestInfo, WorkingInterval, BandwidthMetrics等)
2. 常量定义 (IP_COLOR_MAP, RN_TYPES, SN_TYPES, FLIT_SIZE等)
3. SingleDieAnalyzer - 单Die分析器框架

注意: D2D相关类已移至d2d_analyzer.py模块
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from enum import Enum
import os
import psutil

# 避免循环导入,只在类型检查时导入
if TYPE_CHECKING:
    from .d2d_analyzer import D2DRequestInfo, D2DBandwidthStats, D2DAnalyzer

__all__ = [
    # 基础数据类
    "RequestInfo",
    "WorkingInterval",
    "BandwidthMetrics",
    "PortBandwidthMetrics",
    # 分析器
    "SingleDieAnalyzer",
    # 常量
    "IP_COLOR_MAP",
    "RN_TYPES",
    "SN_TYPES",
    "FLIT_SIZE_BYTES",
]


# ==================== 数据类定义 ====================


@dataclass
class RequestInfo:
    """请求信息数据结构"""

    packet_id: int
    start_time: int  # ns
    end_time: int  # ns (整体网络结束时间)
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


# ==================== 常量定义 ====================

# IP类型颜色映射（统一定义，所有分析器共享）
IP_COLOR_MAP = {
    "GDMA": "#4472C4",  # 蓝色
    "SDMA": "#ED7D31",  # 橙色
    "CDMA": "#70AD47",  # 绿色
    "DDR": "#C00000",  # 红色
    "L2M": "#7030A0",  # 紫色
    "D2D_RN": "#00B0F0",  # 青色
    "D2D_SN": "#FFC000",  # 黄色
    "OTHER": "#808080",  # 灰色
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
        flow_graph_interactive: bool = False,
        show_result_analysis: bool = False,
        verbose: int = 0,
    ):
        """
        初始化单Die分析器

        Args:
            config: 网络配置对象
            min_gap_threshold: 工作区间合并阈值(ns)，小于此值的间隔被视为同一工作区间
            plot_rn_bw_fig: 是否绘制RN带宽曲线图
            plot_flow_graph: 是否绘制静态流量图（PNG）
            flow_graph_interactive: 是否绘制交互式流量图（HTML）
            show_fig: 是否在浏览器中显示图像
            verbose: 详细程度（0=静默，1=正常）
        """
        from collections import defaultdict
        from .core_calculators import DataValidator, TimeIntervalCalculator, BandwidthCalculator
        from .data_collectors import RequestCollector, LatencyStatsCollector, CircuitStatsCollector
        from .result_visualizers import BandwidthPlotter

        # from .flow_graph_renderer import FlowGraphRenderer  # 已弃用
        from .single_die_flow_renderer import SingleDieFlowRenderer
        from .exporters import CSVExporter, ReportGenerator
        from .latency_distribution_plotter import LatencyDistributionPlotter

        self.config = config
        self.min_gap_threshold = min_gap_threshold
        self.network_frequency = config.NETWORK_FREQUENCY  # GHz
        self.plot_rn_bw_fig = plot_rn_bw_fig
        self.plot_flow_graph = plot_flow_graph
        self.flow_graph_interactive = flow_graph_interactive
        self.show_fig = show_result_analysis
        self.verbose = verbose
        self.finish_cycle = 0
        self.sim_model = None  # 添加sim_model引用

        # 数据存储
        self.requests: List[RequestInfo] = []
        self.rn_positions = set()
        self.sn_positions = set()
        self.rn_bandwidth_time_series = defaultdict(lambda: {"time": [], "start_times": [], "bytes": []})
        self.ip_bandwidth_data = None
        self._ip_bandwidth_data_cached = None  # IP带宽数据缓存
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
        self.flow_visualizer = SingleDieFlowRenderer()  # 用于静态PNG流图
        self.interactive_flow_visualizer = SingleDieFlowRenderer()
        self.exporter = CSVExporter(verbose=self.verbose)
        self.report_generator = ReportGenerator()

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

    def normalize_ip_type(self, ip_type, default_fallback="l2m"):
        """标准化IP类型名称，保留实例编号（如gdma_0, gdma_1）

        Args:
            ip_type: IP类型字符串
            default_fallback: 空值或不支持类型的默认返回值

        Returns:
            标准化后的IP类型名称(保留实例编号)
        """
        if not ip_type:
            return default_fallback

        ip_type = ip_type.lower()

        if ip_type == "unknown" or ip_type.startswith("unknown"):
            return default_fallback

        if ip_type.endswith("_ip"):
            ip_type = ip_type[:-3]

        base_type = ip_type.split("_")[0] if "_" in ip_type else ip_type
        supported_types = ["sdma", "gdma", "cdma", "ddr", "l2m", "d2d"]

        if base_type in supported_types or base_type.startswith("d2d"):
            return ip_type
        else:
            return default_fallback

    def collect_requests_data(self, sim_model, simulation_end_cycle=None) -> None:
        """从sim_model收集请求数据 - 委托给RequestCollector"""
        self.sim_model = sim_model
        self.requests = self.request_collector.collect_requests_data(sim_model, simulation_end_cycle)

        # 更新本地状态
        self.finish_cycle = max((req.data_received_complete_cycle for req in self.requests if hasattr(req, "data_received_complete_cycle")), default=0)

        # 收集RN带宽时间序列数据
        for req in self.requests:
            # 构建port_key
            if req.source_type and req.dest_type:
                port_key = f"{req.source_type[:-2].upper()} {req.req_type} {req.dest_type[:3].upper()}"
            else:
                source_backup = getattr(req, "source_type", "UNKNOWN") or "UNKNOWN"
                dest_backup = getattr(req, "dest_type", "UNKNOWN") or "UNKNOWN"
                port_key = f"{source_backup[:-2].upper() if len(source_backup) > 2 else source_backup.upper()} {req.req_type} {dest_backup[:3].upper()}"

            completion_time = req.end_time
            self.rn_bandwidth_time_series[port_key]["time"].append(completion_time)
            self.rn_bandwidth_time_series[port_key]["start_times"].append(req.start_time)
            self.rn_bandwidth_time_series[port_key]["bytes"].append(req.burst_length * 128)

        # 动态统计IP信息
        self.collect_ip_statistics()

    def analyze_all_bandwidth(self) -> Dict:
        """执行完整的带宽分析"""
        import time

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
                # Wait cycle统计
                "req_wait_cycles_h": getattr(self.sim_model, "req_wait_cycle_h_num_stat", 0),
                "req_wait_cycles_v": getattr(self.sim_model, "req_wait_cycle_v_num_stat", 0),
                "rsp_wait_cycles_h": getattr(self.sim_model, "rsp_wait_cycle_h_num_stat", 0),
                "rsp_wait_cycles_v": getattr(self.sim_model, "rsp_wait_cycle_v_num_stat", 0),
                "data_wait_cycles_h": getattr(self.sim_model, "data_wait_cycle_h_num_stat", 0),
                "data_wait_cycles_v": getattr(self.sim_model, "data_wait_cycle_v_num_stat", 0),
                # Retry统计
                "read_retry_num": getattr(self.sim_model, "read_retry_num_stat", 0),
                "write_retry_num": getattr(self.sim_model, "write_retry_num_stat", 0),
                # ETag统计
                "RB_ETag_T1_num": getattr(self.sim_model, "RB_ETag_T1_num_stat", 0),
                "RB_ETag_T0_num": getattr(self.sim_model, "RB_ETag_T0_num_stat", 0),
                "EQ_ETag_T1_num": getattr(self.sim_model, "EQ_ETag_T1_num_stat", 0),
                "EQ_ETag_T0_num": getattr(self.sim_model, "EQ_ETag_T0_num_stat", 0),
                # ITag统计
                "ITag_h_num": getattr(self.sim_model, "ITag_h_num_stat", 0),
                "ITag_v_num": getattr(self.sim_model, "ITag_v_num_stat", 0),
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
                "analysis_config": {"min_gap_threshold_ns": self.min_gap_threshold, "network_frequency_ghz": self.network_frequency},
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

        # 收集需要合并的图表
        charts_to_merge = []  # [(title, fig, custom_js), ...]

        # 生成延迟分布图
        if latency_stats and any(latency_stats.get(cat, {}).get(req_type, {}).get("values", []) for cat in ["cmd", "data", "trans"] for req_type in ["read", "write", "mixed"]):
            from .latency_distribution_plotter import LatencyDistributionPlotter

            latency_plotter = LatencyDistributionPlotter(latency_stats, title_prefix="NoC")

            # 生成直方图+CDF组合图
            hist_cdf_fig = latency_plotter.plot_histogram_with_cdf(return_fig=True)

            # 添加到图表列表
            charts_to_merge.append(("延迟分布", hist_cdf_fig, None))
            # charts_to_merge.append(("NoC延迟分布-小提琴图", violin_fig, None))  # 暂时隐藏

        # 绘制RN带宽曲线并计算Total_sum_BW
        rn_fig = None
        if self.plot_rn_bw_fig and self.sim_model:
            t_start = time.time()
            total_bandwidth, rn_fig = self.visualizer.plot_rn_bandwidth_curves_work_interval(
                rn_bandwidth_time_series=self.rn_bandwidth_time_series,
                network_frequency=self.network_frequency,
                save_path=None,  # 不保存独立文件
                show_fig=False,  # 不显示
                return_fig=True,  # 返回Figure对象
            )
            charts_to_merge.append(("RN带宽曲线", rn_fig, None))
        else:
            total_bandwidth = network_overall.get("mixed", self._empty_metrics()).weighted_bandwidth

        results["Total_sum_BW"] = total_bandwidth
        results["summary"]["Total_sum_BW"] = total_bandwidth

        # 绘制静态流图（PNG）
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
                self.flow_visualizer.draw_ring_flow_graph(network=self.sim_model.data_network, ip_bandwidth_data=self.ip_bandwidth_data, config=self.config, save_path=flow_save_path)
            else:
                self.flow_visualizer.draw_flow_graph(network=self.sim_model.data_network, ip_bandwidth_data=self.ip_bandwidth_data, config=self.config, mode="total", save_path=flow_save_path)

            if flow_save_path:
                saved_figures.append(("流图", flow_save_path))

        # 绘制交互式流图（HTML）
        flow_fig = None
        if self.flow_graph_interactive and self.sim_model:
            # 如果之前没有计算IP带宽数据，则计算
            if self.ip_bandwidth_data is None:
                self.calculate_ip_bandwidth_data()

            # 目前单Die只支持网格拓扑的交互式流图
            if not self.sim_model.topo_type_stat.startswith("Ring"):
                flow_fig = self.interactive_flow_visualizer.draw_flow_graph(
                    network=self.sim_model.data_network,
                    ip_bandwidth_data=self.ip_bandwidth_data,
                    config=self.config,
                    mode="total",
                    save_path=None,  # 不保存独立文件
                    show_fig=False,  # 不显示
                    return_fig=True,  # 返回Figure对象
                    req_network=self.sim_model.req_network,  # 传入请求网络
                    rsp_network=self.sim_model.rsp_network,  # 传入响应网络
                )
                # 流量图放在最前面（按用户要求顺序）
                charts_to_merge.insert(0, ("流量图", flow_fig, None))

        # 将收集的图表保存到result_processor中，供base_model后续合并
        if self.sim_model and hasattr(self.sim_model, "result_processor"):
            if not hasattr(self.sim_model.result_processor, "charts_to_merge"):
                self.sim_model.result_processor.charts_to_merge = []
            # 将当前收集的图表合并到result_processor
            self.sim_model.result_processor.charts_to_merge.extend(charts_to_merge)

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
            "mixed": self.calculator.calculate_bandwidth_metrics(self.requests, None) if self.requests else self._empty_metrics(),
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

                port_metrics[port_id] = PortBandwidthMetrics(port_id=port_id, read_metrics=read_metrics, write_metrics=write_metrics, mixed_metrics=mixed_metrics)

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

                port_metrics[port_id] = PortBandwidthMetrics(port_id=port_id, read_metrics=read_metrics, write_metrics=write_metrics, mixed_metrics=mixed_metrics)

        return port_metrics

    def _empty_metrics(self) -> BandwidthMetrics:
        """返回空的BandwidthMetrics"""
        return BandwidthMetrics(unweighted_bandwidth=0.0, weighted_bandwidth=0.0, working_intervals=[], total_working_time=0, network_start_time=0, network_end_time=0, total_bytes=0, total_requests=0)

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
        """计算IP带宽数据矩阵 - 支持区分IP实例（带缓存优化）"""
        # 如果缓存存在，直接返回
        if self._ip_bandwidth_data_cached is not None:
            self.ip_bandwidth_data = self._ip_bandwidth_data_cached
            return

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

        # 保存到缓存
        self._ip_bandwidth_data_cached = self.ip_bandwidth_data

    def generate_unified_report(self, results: Dict, output_path: str) -> None:
        """
        生成统一的带宽分析报告 (委托给ReportGenerator和CSVExporter)

        Args:
            results: analyze_all_bandwidth()的返回结果
            output_path: 输出目录路径
        """
        # 生成文本报告
        self.report_generator.generate_unified_report(results=results, output_path=output_path, num_ip=self.actual_num_ip if hasattr(self, "actual_num_ip") else 1)

        # 生成详细请求CSV
        self.exporter.generate_detailed_request_csv(requests=self.requests, output_path=output_path)

        # 生成端口CSV
        self.exporter.generate_ports_csv(rn_ports=results.get("rn_ports"), sn_ports=results.get("sn_ports"), output_path=output_path, config=self.config)

        # 导出链路统计数据到CSV
        if hasattr(self.sim_model, "data_network") and self.sim_model.data_network:
            import os

            link_stats_csv = os.path.join(output_path, "link_statistics.csv")
            self.exporter.export_link_statistics_csv(self.sim_model.data_network, link_stats_csv)

        # 生成HTML格式的结果摘要，添加到集成报告中
        if self.sim_model and hasattr(self.sim_model, "result_processor"):
            # 使用actual_num_ip
            num_ip = self.actual_num_ip if hasattr(self, "actual_num_ip") else 1

            # 获取circuit_stats
            circuit_stats = results.get("summary", {}).get("circuit_stats", {})

            # 生成HTML版本
            report_html = self.report_generator.generate_summary_report_html(results=results, num_ip=num_ip, circuit_stats=circuit_stats)

            # 添加到图表列表末尾
            if not hasattr(self.sim_model.result_processor, "charts_to_merge"):
                self.sim_model.result_processor.charts_to_merge = []
            self.sim_model.result_processor.charts_to_merge.append(("结果分析", None, report_html))

        # 打印文件生成提示
        if self.sim_model and hasattr(self.sim_model, "verbose") and self.sim_model.verbose:
            import os

            print()
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
        if hasattr(self.circuit_collector, "generate_fifo_usage_csv"):
            # output_path可以是None,让generate_fifo_usage_csv自己构建路径
            self.circuit_collector.generate_fifo_usage_csv(model=sim_model, output_path=None)  # 让方法自己构建文件路径

    def _calculate_latency_stats(self):
        """
        计算延迟统计 (委托给LatencyStatsCollector)

        Returns:
            延迟统计字典
        """
        return self.latency_collector.calculate_latency_stats(self.requests)

    def _print_summary_to_console(self, results: Dict) -> None:
        """输出重要数据到控制台（已禁用）"""
        pass

    def _generate_integrated_html(self, results: Dict):
        """生成集成的HTML可视化报告"""
        if not self.sim_model or not hasattr(self.sim_model, "result_processor"):
            return

        # 获取所有收集的图表
        all_charts = getattr(self.sim_model.result_processor, "charts_to_merge", [])
        if not all_charts:
            return

        # 确保顺序：流量图 → FIFO热力图 → RN带宽曲线
        # 重新排序图表
        ordered_charts = []
        flow_chart = None
        fifo_chart = None
        rn_chart = None

        for title, fig, custom_js in all_charts:
            if "流量图" in title:
                flow_chart = (title, fig, custom_js)
            elif "FIFO" in title:
                fifo_chart = (title, fig, custom_js)
            elif "RN" in title or "带宽" in title:
                rn_chart = (title, fig, custom_js)

        # 按顺序添加
        if flow_chart:
            ordered_charts.append(flow_chart)
        if fifo_chart:
            ordered_charts.append(fifo_chart)
        if rn_chart:
            ordered_charts.append(rn_chart)

        # 如果没有图表，直接返回
        if not ordered_charts:
            return

        try:
            from src.analysis.integrated_visualizer import create_integrated_report

            # 确定保存路径
            if self.sim_model.results_fig_save_path:
                import os

                save_path = os.path.join(self.sim_model.results_fig_save_path, "result_analysis.html")
            else:
                return

            # 生成集成HTML
            integrated_path = create_integrated_report(charts_config=ordered_charts, save_path=save_path, show_result_analysis=self.show_fig)

        except Exception as e:
            pass
