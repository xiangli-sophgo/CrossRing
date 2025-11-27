"""
D2D分析器模块 - 专门处理跨Die请求分析

提供:
1. D2DRequestInfo - D2D请求信息数据结构
2. D2DBandwidthStats - D2D带宽统计数据结构
3. D2DAnalyzer - D2D跨Die分析器框架
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# 导入基础数据类
from .analyzers import RequestInfo, WorkingInterval, FLIT_SIZE_BYTES


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
        from .result_visualizers import BandwidthPlotter

        # from .flow_graph_renderer import FlowGraphRenderer  # 已弃用
        from .d2d_flow_renderer import D2DFlowRenderer
        from .exporters import CSVExporter, ReportGenerator, JSONExporter
        from .latency_distribution_plotter import LatencyDistributionPlotter

        self.config = config
        self.min_gap_threshold = min_gap_threshold
        self.network_frequency = config.NETWORK_FREQUENCY

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
        self.flow_visualizer = D2DFlowRenderer()  # 用于静态PNG流图
        self.interactive_flow_visualizer = D2DFlowRenderer()  # D2D交互式渲染器
        self.exporter = CSVExporter()
        self.report_generator = ReportGenerator()
        self.json_exporter = JSONExporter()

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
                self.d2d_stats.pair_read_bw[pair_key] = (metrics.unweighted_bandwidth, metrics.weighted_bandwidth)
                self.d2d_stats.total_read_requests += len(requests)

        for pair_key, requests in write_by_pair.items():
            if requests:
                request_infos = self._convert_d2d_to_request_info(requests)
                metrics = self.calculator.calculate_bandwidth_metrics(request_infos, "write")
                self.d2d_stats.pair_write_bw[pair_key] = (metrics.unweighted_bandwidth, metrics.weighted_bandwidth)
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

    def analyze_d2d_results(self) -> Dict:
        """
        执行完整的D2D分析（参考SingleDieAnalyzer.analyze_all_bandwidth）

        Returns:
            Dict: 包含所有D2D分析结果的字典
        """
        # 计算D2D带宽统计
        self.d2d_stats = self.calculate_d2d_bandwidth()

        # 计算延迟统计（包括跨Die和Die内部请求）
        self.latency_stats = self._calculate_d2d_latency_stats()
        latency_stats = self.latency_stats

        # 检查是否有任何请求（跨Die或Die内部）
        total_all_requests = len(self.d2d_requests)
        if hasattr(self, 'all_die_requests'):
            total_all_requests += sum(len(reqs) for reqs in self.all_die_requests.values())

        if total_all_requests == 0:
            return {
                "d2d_stats": self.d2d_stats,
                "latency_stats": {},
                "total_requests": 0,
                "read_requests": 0,
                "write_requests": 0,
                "latency_distribution_figs": [],
            }

        # 统计请求数量
        read_requests = [r for r in self.d2d_requests if r.req_type == "read"]
        write_requests = [r for r in self.d2d_requests if r.req_type == "write"]

        # 生成延迟分布图
        latency_distribution_figs = []

        # 检查是否有延迟数据（只检查mixed类型）
        has_latency_values = False
        if latency_stats:
            for cat in ["cmd", "data", "trans"]:
                mixed_values = latency_stats.get(cat, {}).get("mixed", {}).get("values", [])
                if len(mixed_values) > 0:
                    has_latency_values = True
                    break

        if has_latency_values:
            from .latency_distribution_plotter import LatencyDistributionPlotter

            latency_plotter = LatencyDistributionPlotter(latency_stats, title_prefix="D2D")

            # 生成直方图
            hist_fig = latency_plotter.plot_histogram_with_cdf(return_fig=True)

            # 添加到图表列表
            latency_distribution_figs = [
                ("延迟分布", hist_fig),
            ]

        return {
            "d2d_stats": self.d2d_stats,
            "latency_stats": latency_stats,
            "total_requests": len(self.d2d_requests),
            "read_requests": len(read_requests),
            "write_requests": len(write_requests),
            "latency_distribution_figs": latency_distribution_figs,
        }

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
        latency_stats = self.latency_collector.calculate_latency_stats(self._convert_d2d_to_request_info(self.d2d_requests))

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
            with open(bw_file, "w", encoding="utf-8") as f:
                json.dump(bw_stats, f, indent=2, ensure_ascii=False)

            # 导出延迟统计
            latency_file = os.path.join(output_path, "d2d_latency_stats.json")
            with open(latency_file, "w", encoding="utf-8") as f:
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
        # 同时收集所有请求（包括Die内部和跨Die请求）
        die_ip_instances = {}
        all_die_requests = {}  # {die_id: [requests]}

        for die_id in dies.keys():
            die_ip_instances[die_id] = set()
            all_die_requests[die_id] = []

        # 从每个Die的RequestTracker收集Die内部请求
        for die_id, die_model in dies.items():
            if hasattr(die_model, 'request_tracker') and die_model.request_tracker:
                completed_requests = die_model.request_tracker.get_completed_requests()

                for packet_id, lifecycle in completed_requests.items():
                    # 只收集Die内部请求（非跨Die）
                    if lifecycle.is_cross_die:
                        continue

                    # 只收集属于该Die的请求（通过origin_die判断）
                    if hasattr(lifecycle, 'origin_die') and lifecycle.origin_die != die_id:
                        continue

                    # 计算延迟
                    timestamps = lifecycle.timestamps
                    if not timestamps:
                        timestamps = die_model.request_tracker.collect_timestamps_from_flits(packet_id)

                    # 使用data_collectors的延迟计算方法
                    from src.analysis.data_collectors import RequestCollector
                    temp_collector = RequestCollector(network_frequency=self.network_frequency)
                    cmd_latency, data_latency, transaction_latency = temp_collector._calculate_latencies(lifecycle, timestamps)

                    # 从data_flits收集绕环统计数据
                    data_eject_attempts_h_list = [f.eject_attempts_h for f in lifecycle.data_flits]
                    data_eject_attempts_v_list = [f.eject_attempts_v for f in lifecycle.data_flits]
                    data_ordering_blocked_h_list = [f.ordering_blocked_eject_h for f in lifecycle.data_flits]
                    data_ordering_blocked_v_list = [f.ordering_blocked_eject_v for f in lifecycle.data_flits]

                    # 转换为RequestInfo格式以便后续计算
                    from src.analysis.analyzers import RequestInfo
                    req_info = RequestInfo(
                        packet_id=packet_id,
                        start_time=int(lifecycle.created_cycle / self.network_frequency),
                        end_time=int(lifecycle.completed_cycle / self.network_frequency),
                        req_type=lifecycle.op_type,
                        burst_length=lifecycle.burst_size,
                        total_bytes=lifecycle.burst_size * 128,
                        source_node=lifecycle.source,
                        dest_node=lifecycle.destination,
                        source_type=lifecycle.source_type,
                        dest_type=lifecycle.dest_type,
                        cmd_latency=cmd_latency if cmd_latency >= 0 else 0,
                        data_latency=data_latency if data_latency >= 0 else 0,
                        transaction_latency=transaction_latency if transaction_latency >= 0 else 0,
                        data_eject_attempts_h_list=data_eject_attempts_h_list,
                        data_eject_attempts_v_list=data_eject_attempts_v_list,
                        data_ordering_blocked_h_list=data_ordering_blocked_h_list,
                        data_ordering_blocked_v_list=data_ordering_blocked_v_list,
                    )
                    all_die_requests[die_id].append(req_info)

                    # 收集IP类型
                    if lifecycle.source_type:
                        source_type = self.normalize_ip_type(lifecycle.source_type, default_fallback="other")
                        die_ip_instances[die_id].add(source_type)
                    if lifecycle.dest_type:
                        dest_type = self.normalize_ip_type(lifecycle.dest_type, default_fallback="other")
                        die_ip_instances[die_id].add(dest_type)

        # 从跨Die请求中收集IP实例
        for request in self.d2d_requests:
            # 收集source_type
            if request.source_die in dies and request.source_type:
                raw_source = request.source_type
                if raw_source.endswith("_ip"):
                    raw_source = raw_source[:-3]
                source_type = self.normalize_ip_type(raw_source, default_fallback="other")
                die_ip_instances[request.source_die].add(source_type)

            # 收集target_type
            if request.target_die in dies and request.target_type:
                raw_target = request.target_type
                if raw_target.endswith("_ip"):
                    raw_target = raw_target[:-3]
                target_type = self.normalize_ip_type(raw_target, default_fallback="other")
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

        # 保存all_die_requests以便延迟统计使用
        self.all_die_requests = all_die_requests

        # 计算跨Die请求带宽
        self._calculate_bandwidth_from_d2d_requests(dies)

        # 计算Die内部请求带宽
        self._calculate_bandwidth_from_die_internal_requests(dies, all_die_requests)

    def _calculate_bandwidth_from_d2d_requests(self, dies: Dict):
        """基于跨Die请求计算各Die的IP带宽（只处理跨Die请求，Die内请求由_calculate_bandwidth_from_die_internal_requests处理）"""
        from collections import defaultdict

        # 过滤出跨Die请求
        cross_die_requests = [r for r in self.d2d_requests if r.source_die != r.target_die]

        # 第一步：按(die_id, source_node, source_type)分组source请求
        source_groups = defaultdict(list)
        for request in cross_die_requests:
            if request.source_die in dies:
                source_type_normalized = self.normalize_ip_type(request.source_type, default_fallback="other")
                key = (request.source_die, request.source_node, source_type_normalized)
                source_groups[key].append(request)

        # 第二步：按(die_id, target_node, target_type)分组target请求
        target_groups = defaultdict(list)
        for request in cross_die_requests:
            if request.target_die in dies:
                target_type_normalized = self.normalize_ip_type(request.target_type, default_fallback="other")
                key = (request.target_die, request.target_node, target_type_normalized)
                target_groups[key].append(request)

        # 第三步：处理source带宽
        for (die_id, node, ip_type), requests in source_groups.items():
            row, col = self._get_physical_position(node, dies[die_id])

            # 按req_type分组并计算带宽(使用RN端时间戳)
            read_reqs = [r for r in requests if r.req_type == "read"]
            write_reqs = [r for r in requests if r.req_type == "write"]

            if read_reqs:
                _, weighted_bw = self._calculate_bandwidth_for_group(read_reqs, endpoint_type="rn")
                self.die_ip_bandwidth_data[die_id]["read"][ip_type][row, col] += weighted_bw

            if write_reqs:
                _, weighted_bw = self._calculate_bandwidth_for_group(write_reqs, endpoint_type="rn")
                self.die_ip_bandwidth_data[die_id]["write"][ip_type][row, col] += weighted_bw

            if requests:
                _, weighted_bw = self._calculate_bandwidth_for_group(requests, endpoint_type="rn")
                self.die_ip_bandwidth_data[die_id]["total"][ip_type][row, col] += weighted_bw

        # 第四步：处理target带宽
        for (die_id, node, ip_type), requests in target_groups.items():
            row, col = self._get_physical_position(node, dies[die_id])

            # 按req_type分组并计算带宽(使用SN端时间戳)
            read_reqs = [r for r in requests if r.req_type == "read"]
            write_reqs = [r for r in requests if r.req_type == "write"]

            if read_reqs:
                _, weighted_bw = self._calculate_bandwidth_for_group(read_reqs, endpoint_type="sn")
                self.die_ip_bandwidth_data[die_id]["read"][ip_type][row, col] += weighted_bw

            if write_reqs:
                _, weighted_bw = self._calculate_bandwidth_for_group(write_reqs, endpoint_type="sn")
                self.die_ip_bandwidth_data[die_id]["write"][ip_type][row, col] += weighted_bw

            if requests:
                _, weighted_bw = self._calculate_bandwidth_for_group(requests, endpoint_type="sn")
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

    def _calculate_bandwidth_from_die_internal_requests(self, dies: Dict, all_die_requests: Dict):
        """基于Die内部请求计算各Die的IP带宽"""
        from collections import defaultdict


        for die_id, requests in all_die_requests.items():
            if die_id not in dies or not requests:
                continue


            # 按(source_node, source_type)分组source请求
            source_groups = defaultdict(list)
            for req in requests:
                if req.source_type:
                    source_type = self.normalize_ip_type(req.source_type, default_fallback="other")
                    key = (req.source_node, source_type)
                    source_groups[key].append(req)

            # 按(dest_node, dest_type)分组target请求
            target_groups = defaultdict(list)
            for req in requests:
                if req.dest_type:
                    dest_type = self.normalize_ip_type(req.dest_type, default_fallback="other")
                    key = (req.dest_node, dest_type)
                    target_groups[key].append(req)

            # 处理source带宽
            for (node, ip_type), reqs in source_groups.items():
                row, col = self._get_physical_position(node, dies[die_id])

                read_reqs = [r for r in reqs if r.req_type == "read"]
                write_reqs = [r for r in reqs if r.req_type == "write"]

                if read_reqs:
                    _, weighted_bw = self._calculate_bandwidth_for_group(read_reqs, endpoint_type="rn")
                    self.die_ip_bandwidth_data[die_id]["read"][ip_type][row, col] += weighted_bw

                if write_reqs:
                    _, weighted_bw = self._calculate_bandwidth_for_group(write_reqs, endpoint_type="rn")
                    self.die_ip_bandwidth_data[die_id]["write"][ip_type][row, col] += weighted_bw

                if reqs:
                    _, weighted_bw = self._calculate_bandwidth_for_group(reqs, endpoint_type="rn")
                    self.die_ip_bandwidth_data[die_id]["total"][ip_type][row, col] += weighted_bw

            # 处理target带宽
            for (node, ip_type), reqs in target_groups.items():
                row, col = self._get_physical_position(node, dies[die_id])

                read_reqs = [r for r in reqs if r.req_type == "read"]
                write_reqs = [r for r in reqs if r.req_type == "write"]

                if read_reqs:
                    _, weighted_bw = self._calculate_bandwidth_for_group(read_reqs, endpoint_type="sn")
                    self.die_ip_bandwidth_data[die_id]["read"][ip_type][row, col] += weighted_bw

                if write_reqs:
                    _, weighted_bw = self._calculate_bandwidth_for_group(write_reqs, endpoint_type="sn")
                    self.die_ip_bandwidth_data[die_id]["write"][ip_type][row, col] += weighted_bw

                if reqs:
                    _, weighted_bw = self._calculate_bandwidth_for_group(reqs, endpoint_type="sn")
                    self.die_ip_bandwidth_data[die_id]["total"][ip_type][row, col] += weighted_bw

    def _get_physical_position(self, node: int, die_model) -> tuple:
        """获取节点的物理行列位置"""
        cols = die_model.config.NUM_COL
        row = node // cols
        col = node % cols
        return row, col

    def _calculate_bandwidth_for_group(self, requests: list, endpoint_type: str = None) -> tuple:
        """计算请求组的带宽（非加权和加权）

        Args:
            requests: 请求列表
            endpoint_type: 端点类型 ("rn", "sn", None)
                - "rn": 使用RN端时间戳(不跨DIE请求),或end_time(跨DIE请求)
                - "sn": 使用SN端时间戳(不跨DIE请求),或end_time(跨DIE请求)
                - None: 使用end_time

        Returns:
            (unweighted_bandwidth, weighted_bandwidth)
        """
        if not requests:
            return 0.0, 0.0

        # 转换D2DRequestInfo到RequestInfo以便使用interval_calculator
        converted_requests = []
        for r in requests:
            # 检查是否是D2DRequestInfo
            if hasattr(r, "start_time_ns"):
                # 创建一个简单的对象模拟RequestInfo
                class _TempRequest:
                    def __init__(self, d2d_req, endpoint_type):
                        self.packet_id = d2d_req.packet_id
                        self.start_time = d2d_req.start_time_ns

                        # 统一使用整体end_time
                        self.end_time = d2d_req.end_time_ns

                        self.total_bytes = d2d_req.data_bytes
                        self.burst_length = d2d_req.burst_length

                converted_requests.append(_TempRequest(r, endpoint_type))
            else:
                converted_requests.append(r)

        # 计算工作区间
        working_intervals = self.interval_calculator.calculate_working_intervals(converted_requests)

        # 计算总字节数 (使用原始requests)
        total_bytes = sum(getattr(r, "data_bytes", 0) or getattr(r, "total_bytes", 0) for r in requests)

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

    def generate_d2d_bandwidth_report(self, output_path: str, dies: Dict = None):
        """生成D2D带宽报告（按任意Die组合逐项列出）"""
        stats = self.d2d_stats  # 使用已计算的结果

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

        report_lines.extend(["", "延迟统计 (ns):"])
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

        # 保存到文件（包含每个Die的详细统计）
        import os

        os.makedirs(output_path, exist_ok=True)
        report_file = os.path.join(output_path, "d2d_bandwidth_summary.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            for line in report_lines:
                f.write(line + "\n")

            # 添加每个Die的详细统计（不包含绕环比例）
            if dies and circuit_stats_data:
                f.write("\n\n")
                f.write("=" * 60 + "\n")
                f.write("各Die详细统计（D2D绕环比例见上方汇总）\n")
                f.write("=" * 60 + "\n\n")

                for die_id in sorted(circuit_stats_data["per_die"].keys()):
                    die_stats = circuit_stats_data["per_die"][die_id]
                    f.write(f"Die {die_id}:\n")
                    f.write("-" * 30 + "\n")
                    for line in self._format_circuit_stats(die_stats, prefix="  ", skip_circling=True):
                        f.write(line + "\n")
                    f.write("\n")

        return report_file

    def generate_d2d_summary_report_html(self, dies=None):
        """
        生成D2D HTML格式的统计摘要

        Args:
            dies: Die模型字典（用于绕环统计）

        Returns:
            str: HTML格式的报告内容
        """
        from .exporters import ReportGenerator

        # 收集延迟统计（并存储为实例属性）
        self.latency_stats = self._calculate_d2d_latency_stats()
        latency_stats = self.latency_stats

        # 收集绕环统计（并存储为实例属性）
        self.circuit_stats = None
        if dies:
            self.circuit_stats = self._collect_d2d_circuit_stats(dies)
        circuit_stats_data = self.circuit_stats

        # 调用ReportGenerator生成HTML
        report_generator = ReportGenerator()
        html_content = report_generator.generate_d2d_summary_report_html(
            d2d_stats=self.d2d_stats,
            d2d_requests=self.d2d_requests,
            latency_stats=latency_stats,
            circuit_stats=circuit_stats_data,
            die_ip_bandwidth_data=self.die_ip_bandwidth_data
        )

        return html_content

    def _calculate_d2d_latency_stats(self):
        """计算D2D请求和Die内部请求的延迟统计数据（cmd/data/transaction）"""
        import numpy as np

        stats = {
            "cmd": {"read": {"sum": 0, "max": 0, "count": 0, "values": []}, "write": {"sum": 0, "max": 0, "count": 0, "values": []}, "mixed": {"sum": 0, "max": 0, "count": 0, "values": []}},
            "data": {"read": {"sum": 0, "max": 0, "count": 0, "values": []}, "write": {"sum": 0, "max": 0, "count": 0, "values": []}, "mixed": {"sum": 0, "max": 0, "count": 0, "values": []}},
            "trans": {"read": {"sum": 0, "max": 0, "count": 0, "values": []}, "write": {"sum": 0, "max": 0, "count": 0, "values": []}, "mixed": {"sum": 0, "max": 0, "count": 0, "values": []}},
        }

        # 统计跨Die请求（D2DRequestInfo）
        for req in self.d2d_requests:
            req_type = req.req_type
            # 延迟字段映射: D2DRequestInfo使用_ns后缀
            latency_fields = [("cmd", "cmd_latency_ns"), ("data", "data_latency_ns"), ("trans", "transaction_latency_ns")]
            for lat_type, lat_attr in latency_fields:
                if hasattr(req, lat_attr):
                    lat_val = getattr(req, lat_attr)
                    if not np.isinf(lat_val) and lat_val > 0:
                        stats[lat_type][req_type]["sum"] += lat_val
                        stats[lat_type][req_type]["max"] = max(stats[lat_type][req_type]["max"], lat_val)
                        stats[lat_type][req_type]["count"] += 1
                        stats[lat_type][req_type]["values"].append(lat_val)
                        stats[lat_type]["mixed"]["sum"] += lat_val
                        stats[lat_type]["mixed"]["max"] = max(stats[lat_type]["mixed"]["max"], lat_val)
                        stats[lat_type]["mixed"]["count"] += 1
                        stats[lat_type]["mixed"]["values"].append(lat_val)

        # 统计Die内部请求（RequestInfo）
        if hasattr(self, 'all_die_requests'):
            for die_id, requests in self.all_die_requests.items():
                for req in requests:
                    req_type = req.req_type
                    # 延迟字段映射: RequestInfo不使用_ns后缀
                    latency_fields = [("cmd", "cmd_latency"), ("data", "data_latency"), ("trans", "transaction_latency")]
                    for lat_type, lat_attr in latency_fields:
                        if hasattr(req, lat_attr):
                            lat_val = getattr(req, lat_attr)
                            if not np.isinf(lat_val) and lat_val > 0:
                                stats[lat_type][req_type]["sum"] += lat_val
                                stats[lat_type][req_type]["max"] = max(stats[lat_type][req_type]["max"], lat_val)
                                stats[lat_type][req_type]["count"] += 1
                                stats[lat_type][req_type]["values"].append(lat_val)
                                stats[lat_type]["mixed"]["sum"] += lat_val
                                stats[lat_type]["mixed"]["max"] = max(stats[lat_type]["mixed"]["max"], lat_val)
                                stats[lat_type]["mixed"]["count"] += 1
                                stats[lat_type]["mixed"]["values"].append(lat_val)

        # 计算百分位数（p95, p99）
        for cat in ["cmd", "data", "trans"]:
            for req_type in ["read", "write", "mixed"]:
                values = stats[cat][req_type]["values"]
                if len(values) > 0:
                    stats[cat][req_type]["p95"] = np.percentile(values, 95)
                    stats[cat][req_type]["p99"] = np.percentile(values, 99)
                else:
                    stats[cat][req_type]["p95"] = 0.0
                    stats[cat][req_type]["p99"] = 0.0

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

        # 从每个Die收集统计数据(直接从die_model属性读取，参考旧版d2d_result_processor)
        for die_id, die_model in dies.items():
            die_stats = {
                "circuits_req_h": die_model.req_cir_h_num_stat,
                "circuits_req_v": die_model.req_cir_v_num_stat,
                "circuits_rsp_h": die_model.rsp_cir_h_num_stat,
                "circuits_rsp_v": die_model.rsp_cir_v_num_stat,
                "circuits_data_h": die_model.data_cir_h_num_stat,
                "circuits_data_v": die_model.data_cir_v_num_stat,
                "wait_cycle_req_h": die_model.req_wait_cycle_h_num_stat,
                "wait_cycle_req_v": die_model.req_wait_cycle_v_num_stat,
                "wait_cycle_rsp_h": die_model.rsp_wait_cycle_h_num_stat,
                "wait_cycle_rsp_v": die_model.rsp_wait_cycle_v_num_stat,
                "wait_cycle_data_h": die_model.data_wait_cycle_h_num_stat,
                "wait_cycle_data_v": die_model.data_wait_cycle_v_num_stat,
                "read_retry_num": die_model.read_retry_num_stat,
                "write_retry_num": die_model.write_retry_num_stat,
                "RB_ETag_T1_num": die_model.RB_ETag_T1_num_stat,
                "RB_ETag_T0_num": die_model.RB_ETag_T0_num_stat,
                "EQ_ETag_T1_num": die_model.EQ_ETag_T1_num_stat,
                "EQ_ETag_T0_num": die_model.EQ_ETag_T0_num_stat,
                "ITag_h_num": die_model.ITag_h_num_stat,
                "ITag_v_num": die_model.ITag_v_num_stat,
            }

            per_die_stats[die_id] = die_stats

            # 汇总到summary (不包括绕环比例)
            for key in summary_stats.keys():
                if key == "circling_ratio":
                    continue  # 跳过绕环比例，稍后使用全局请求计算
                elif key in die_stats:
                    summary_stats[key] += die_stats[key]

        # 使用全局D2D请求计算绕环比例
        from .data_collectors import CircuitStatsCollector
        circuit_collector = CircuitStatsCollector()

        # 将all_die_requests字典合并为请求列表
        all_requests = []
        if isinstance(self.all_die_requests, dict):
            for die_requests in self.all_die_requests.values():
                all_requests.extend(die_requests)
        else:
            all_requests = self.all_die_requests

        global_circling_stats = circuit_collector.calculate_circling_eject_stats(all_requests)

        # 更新summary中的绕环统计
        summary_stats["circling_ratio"]["horizontal"]["circling_flits"] = global_circling_stats["horizontal"]["circling_flits"]
        summary_stats["circling_ratio"]["horizontal"]["total_flits"] = global_circling_stats["horizontal"]["total_data_flits"]
        summary_stats["circling_ratio"]["horizontal"]["circling_ratio"] = global_circling_stats["horizontal"]["circling_ratio"]

        summary_stats["circling_ratio"]["vertical"]["circling_flits"] = global_circling_stats["vertical"]["circling_flits"]
        summary_stats["circling_ratio"]["vertical"]["total_flits"] = global_circling_stats["vertical"]["total_data_flits"]
        summary_stats["circling_ratio"]["vertical"]["circling_ratio"] = global_circling_stats["vertical"]["circling_ratio"]

        summary_stats["circling_ratio"]["overall"]["circling_flits"] = global_circling_stats["overall"]["circling_flits"]
        summary_stats["circling_ratio"]["overall"]["total_flits"] = global_circling_stats["overall"]["total_data_flits"]
        summary_stats["circling_ratio"]["overall"]["circling_ratio"] = global_circling_stats["overall"]["circling_ratio"]

        return {"per_die": per_die_stats, "summary": summary_stats}

    @staticmethod
    def _format_circuit_stats(stats: Dict, prefix: str = "  ", skip_circling: bool = False) -> List[str]:
        """
        格式化绕环统计数据为文本行

        Args:
            stats: 统计数据字典
            prefix: 每行的前缀
            skip_circling: 是否跳过绕环比例显示

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

        if not skip_circling and "circling_ratio" in stats:
            h_ratio = stats["circling_ratio"]["horizontal"]["circling_ratio"]
            v_ratio = stats["circling_ratio"]["vertical"]["circling_ratio"]
            overall_ratio = stats["circling_ratio"]["overall"]["circling_ratio"]
            lines.append(f"{prefix}绕环比例: H: {h_ratio*100:.2f}%, V: {v_ratio*100:.2f}%, Overall: {overall_ratio*100:.2f}%")

        return lines

    def save_d2d_requests_csv(self, output_path: str = None, return_content: bool = False):
        """
        保存D2D请求到CSV文件（委托给CSVExporter）

        Args:
            output_path: 输出目录路径（return_content=True时可为None）
            return_content: 如果True，返回{filename: content}字典，不写文件

        Returns:
            如果return_content=True: Dict[str, str] 文件名到内容的映射
            否则: None
        """
        return self.exporter.save_d2d_requests_csv(d2d_requests=self.d2d_requests, output_path=output_path, return_content=return_content)

    def save_ip_bandwidth_to_csv(self, output_path: str = None, die_ip_bandwidth_data: Dict = None, config=None, return_content: bool = False):
        """
        保存IP带宽数据到CSV（委托给CSVExporter）

        Args:
            output_path: 输出目录路径（return_content=True时可为None）
            die_ip_bandwidth_data: Die IP带宽数据（可选，如果不提供则使用self.die_ip_bandwidth_data）
            config: 配置对象（可选，如果不提供则使用self.config）
            return_content: 如果True，返回CSV内容字符串，不写文件

        Returns:
            如果return_content=True: str CSV内容
            否则: None
        """
        # 如果没有提供数据，尝试从实例属性获取
        if die_ip_bandwidth_data is None:
            die_ip_bandwidth_data = getattr(self, "die_ip_bandwidth_data", None)

        if config is None:
            config = self.config

        # 如果还是没有数据，直接返回
        if not die_ip_bandwidth_data:
            return "" if return_content else None

        return self.exporter.save_ip_bandwidth_to_csv(die_ip_bandwidth_data=die_ip_bandwidth_data, output_path=output_path, config=config, return_content=return_content)

    def draw_d2d_flow_graph(self, die_networks: Dict = None, dies: Dict = None, config=None, die_ip_bandwidth_data: Dict = None, mode: str = "total", save_path: str = None):
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
            die_ip_bandwidth_data=die_ip_bandwidth_data if die_ip_bandwidth_data else self.die_ip_bandwidth_data if hasattr(self, "die_ip_bandwidth_data") else None,
            mode=mode,
            save_path=save_path,
        )

    def draw_d2d_flow_graph_interactive(
        self,
        die_networks: Dict = None,
        dies: Dict = None,
        config=None,
        die_ip_bandwidth_data: Dict = None,
        mode: str = "total",
        save_path: str = None,
        show_fig: bool = False,
        return_fig: bool = False,
        static_bandwidth: Dict = None,
    ):
        """
        绘制D2D流图（交互式版本，生成HTML文件）

        Args:
            die_networks: Die网络字典
            dies: Die模型字典
            config: 配置对象
            die_ip_bandwidth_data: Die IP带宽数据
            mode: 显示模式
            save_path: 保存路径（会自动转换为.html后缀）
            show_fig: 是否在浏览器中显示图像
            return_fig: 是否返回Figure对象（用于集成报告）
            static_bandwidth: 静态带宽数据

        Returns:
            str or Figure: 如果return_fig=True返回Figure对象，否则返回HTML文件路径
        """
        return self.interactive_flow_visualizer.draw_d2d_flow_graph(
            die_networks=die_networks,
            dies=dies,
            config=config,
            die_ip_bandwidth_data=die_ip_bandwidth_data if die_ip_bandwidth_data else self.die_ip_bandwidth_data if hasattr(self, "die_ip_bandwidth_data") else None,
            mode=mode,
            save_path=save_path,
            show_fig=show_fig,
            return_fig=return_fig,
            static_bandwidth=static_bandwidth,
        )

    def draw_ip_bandwidth_heatmap(self, dies=None, config=None, mode="total", node_size=4000, save_path=None):
        """
        绘制IP带宽热力图(委托给FlowGraphRenderer)

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

        # 委托给FlowGraphRenderer处理（使用draw_ip_bandwidth_heatmap方法）
        return self.flow_visualizer.draw_ip_bandwidth_heatmap(dies=dies, config=config, die_ip_bandwidth_data=self.die_ip_bandwidth_data, mode=mode, node_size=node_size, save_path=save_path)
