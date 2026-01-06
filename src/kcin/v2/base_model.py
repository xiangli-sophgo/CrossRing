from collections import deque, defaultdict

from src.kcin.base.topology_utils import create_adjacency_matrix, find_shortest_paths
from src.kcin.base.config import KCINConfigBase
from src.kcin.v2.config import V2Config
from src.utils.flit import Flit, TokenBucket
from src.utils.request_tracker import RequestTracker
from src.kcin.v2.components import Network, IPInterface

from src.analysis.Link_State_Visualizer import NetworkLinkVisualizer
import matplotlib.pyplot as plt
import os
import sys, time
import inspect, logging
import numpy as np
from functools import wraps, lru_cache
from src.analysis.analyzers import SingleDieAnalyzer, RequestInfo, BandwidthMetrics, WorkingInterval
from src.kcin.base.traffic_scheduler import TrafficScheduler
from src.utils.arbitration import create_arbiter_from_config
import threading

from src.kcin.v2.mixins import StatsMixin, DataflowMixin


class BaseModel(StatsMixin, DataflowMixin):
    # 全局packet_id生成器（从Node类迁移）
    _global_packet_id = 0

    @classmethod
    def get_next_packet_id(cls):
        """获取下一个packet_id"""
        cls._global_packet_id += 1
        return cls._global_packet_id

    @classmethod
    def reset_packet_id(cls):
        """重置packet_id计数器"""
        cls._global_packet_id = 0

    def __init__(self, model_type, config: KCINConfigBase, topo_type, verbose: int = 0):
        """
        初始化BaseModel - 仅设置核心属性

        Args:
            model_type: 模型类型（REQ_RSP, Packet_Base, Feature）
            config: KCINConfigBase配置对象
            topo_type: 拓扑类型（如"5x4", "4x4"等）
            verbose: 详细程度（0=静默，1=正常）
        """
        self.model_type_stat = model_type
        self.config = config
        self.topo_type_stat = topo_type
        self.verbose = verbose

        # Traffic相关 - 通过setup_traffic_scheduler设置
        self.traffic_file_path = None
        self.traffic_scheduler = None
        self.file_name = "unknown.txt"

        # 结果保存路径 - 通过setup_result_analysis设置
        self.result_save_path = None
        self.results_fig_save_path = None

        # 可视化配置 - 通过setup_result_analysis设置
        self.plot_flow_fig = False
        self.flow_graph_interactive = False

        # 链路状态可视化 - 通过setup_visualization设置
        self.plot_link_state = False
        self.plot_start_cycle = -1

        # 调试配置 - 通过setup_debug设置
        self.print_trace = False
        self.show_trace_id = []
        self.show_node_id = 3
        self.update_interval = 0.0

        # 内部状态标志
        self._done_flags = {
            "req": False,
            "rsp": False,
            "flit": False,
        }

        # 请求追踪器
        network_freq = getattr(config, "NETWORK_FREQUENCY", 2.0)
        self.request_tracker = RequestTracker(network_frequency=network_freq)

        # 取消标志（用于外部中断仿真）
        self._cancelled = False

    def setup_traffic_scheduler(self, traffic_file_path: str, traffic_chains: list) -> None:
        """
        配置流量调度器并提取IP需求

        Args:
            traffic_file_path: 流量文件路径
            traffic_chains: 流量链配置，每个链包含文件名列表
                           例如: [["file1.txt", "file2.txt"], ["file3.txt"]]
        """
        self.traffic_file_path = traffic_file_path

        # 1. 提取traffic文件中的IP需求
        self._extract_ip_requirements_from_traffic(traffic_file_path, traffic_chains)

        # 2. 初始化TrafficScheduler
        self.traffic_scheduler = TrafficScheduler(self.config, traffic_file_path)
        self.traffic_scheduler.set_verbose(self.verbose > 0)

        # 3. 处理traffic配置
        if isinstance(traffic_chains, str):
            # 单个文件字符串
            self.file_name = traffic_chains
            self.traffic_scheduler.setup_single_chain([traffic_chains])
        elif isinstance(traffic_chains, list):
            # 多traffic链配置
            self.traffic_scheduler.setup_parallel_chains(traffic_chains)
            self.file_name = self.traffic_scheduler.get_save_filename() + ".txt"
        else:
            raise ValueError("traffic_chains必须是字符串(单文件)或列表(多链)")

        # 4. 加载traffic元数据并计算静态带宽（如果有）
        meta_data = self._load_traffic_metadata(traffic_file_path, traffic_chains)
        if meta_data and self.verbose:
            self._compute_static_bandwidth(meta_data)

    def _extract_ip_requirements_from_traffic(self, traffic_file_path: str, traffic_chains):
        """
        从traffic文件中提取IP接口需求

        Args:
            traffic_file_path: traffic文件基础路径
            traffic_chains: traffic文件链配置
        """
        from src.utils.traffic_ip_extractor import TrafficIPExtractor
        import os

        extractor = TrafficIPExtractor()

        # 收集所有要解析的traffic文件
        traffic_files = []
        if isinstance(traffic_chains, str):
            traffic_files = [os.path.join(traffic_file_path, traffic_chains)]
        elif isinstance(traffic_chains, list):
            for chain in traffic_chains:
                if isinstance(chain, list):
                    for file_name in chain:
                        traffic_files.append(os.path.join(traffic_file_path, file_name))
                else:
                    traffic_files.append(os.path.join(traffic_file_path, chain))

        # 解析所有traffic文件
        result = extractor.extract_from_multiple_files(traffic_files)
        self._required_ips = result["required_ips"]
        self._has_cross_die_traffic = result["has_cross_die"]

        # 提取唯一的IP类型
        ip_types = TrafficIPExtractor.get_unique_ip_types(self._required_ips)

        # 更新config的CH_NAME_LIST和CHANNEL_SPEC
        self.config.update_channel_list_from_ips(ip_types)
        self.config.infer_channel_spec_from_ips(ip_types)

    def _load_traffic_metadata(self, traffic_file_path: str, traffic_chains):
        """
        从traffic文件第一行读取元数据

        Args:
            traffic_file_path: traffic文件基础路径
            traffic_chains: traffic文件链配置

        Returns:
            元数据字典，如果没有元数据则返回None
        """
        import os
        import json

        try:
            # 获取第一个traffic文件
            first_file = None
            if isinstance(traffic_chains, str):
                first_file = traffic_chains
            elif isinstance(traffic_chains, list) and len(traffic_chains) > 0:
                first_chain = traffic_chains[0]
                if isinstance(first_chain, list) and len(first_chain) > 0:
                    first_file = first_chain[0]
                elif isinstance(first_chain, str):
                    first_file = first_chain

            if not first_file:
                return None

            file_path = os.path.join(traffic_file_path, first_file)

            # 读取第一行
            with open(file_path, "r") as f:
                first_line = f.readline().strip()

            # 检查是否是元数据行
            if first_line.startswith("# TRAFFIC_META:"):
                meta_json = first_line.replace("# TRAFFIC_META:", "").strip()
                return json.loads(meta_json)
        except Exception as e:
            if self.verbose:
                print(f"[INFO] 无法加载traffic元数据: {e}")

        return None

    def _compute_static_bandwidth(self, meta_data: dict):
        """
        基于元数据计算静态链路带宽（NoC单Die模式）

        Args:
            meta_data: 从traffic文件加载的元数据（只包含configs）
        """
        try:
            from src.traffic_process.traffic_gene.static_bandwidth_analyzer import compute_link_bandwidth

            # 重建TrafficConfig对象（简化版本）
            class SimpleConfig:
                def __init__(self, data):
                    self.__dict__.update(data)

            configs = [SimpleConfig(cfg) for cfg in meta_data["configs"]]

            # 从仿真配置获取topo_type
            topo_type = self.config.TOPO_TYPE

            # 计算静态带宽
            self.static_link_bandwidth = compute_link_bandwidth(topo_type=topo_type, configs=configs, routing_type="XY")  # 默认XY路由

            # 保存configs供可视化使用
            self.static_bw_configs = configs

            if self.verbose:
                active_links = sum(1 for bw in self.static_link_bandwidth.values() if bw > 0)
                print(f"[INFO] 静态带宽分析完成（基于traffic元数据）: {active_links} 条活跃链路")
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] 静态带宽计算失败: {e}")
            self.static_link_bandwidth = None
            self.static_bw_configs = None

    def _extract_node_ips_from_configs(self, configs) -> dict:
        """
        从configs中提取node_ips映射

        Args:
            configs: TrafficConfig列表

        Returns:
            {node_id: [ip_type_list]}
        """
        node_ips = {}
        for config in configs:
            # 从src_map提取
            if hasattr(config, "src_map"):
                for ip_type, nodes in config.src_map.items():
                    for node in nodes:
                        if node not in node_ips:
                            node_ips[node] = []
                        if ip_type not in node_ips[node]:
                            node_ips[node].append(ip_type)
            # 从dst_map提取
            if hasattr(config, "dst_map"):
                for ip_type, nodes in config.dst_map.items():
                    for node in nodes:
                        if node not in node_ips:
                            node_ips[node] = []
                        if ip_type not in node_ips[node]:
                            node_ips[node].append(ip_type)
        return node_ips

    def _generate_static_bandwidth_figure(self):
        """
        生成静态带宽拓扑图（使用Cytoscape.js交互式可视化）

        Returns:
            (None, html_snippet)元组，其中html_snippet是Cytoscape的HTML代码，
            如果无法生成则返回None
        """
        if not hasattr(self, "static_link_bandwidth") or not self.static_link_bandwidth:
            return None

        if not hasattr(self, "static_bw_configs") or not self.static_bw_configs:
            return None

        try:
            from src.traffic_process.traffic_gene.cytoscape_bandwidth_visualizer import CytoscapeBandwidthVisualizer

            # 从configs中提取node_ips
            node_ips = self._extract_node_ips_from_configs(self.static_bw_configs)

            # 从仿真配置获取topo_type
            topo_type = self.config.TOPO_TYPE

            # 创建Cytoscape可视化器（NoC模式）
            viz = CytoscapeBandwidthVisualizer(topo_type=topo_type, mode="noc", link_bandwidth=self.static_link_bandwidth, node_ips=node_ips, d2d_config=None)

            # 生成HTML片段
            html_snippet = viz.generate_html_snippet()

            # 返回(None, html)表示自定义HTML内容
            return (None, html_snippet)

        except Exception as e:
            if self.verbose:
                print(f"[WARNING] 静态带宽图生成失败: {e}")
            return None

    def setup_result_analysis(
        self,
        result_save_path: str = "",
        results_fig_save_path: str = "",
        plot_flow_fig: bool = False,
        flow_graph_interactive: bool = False,
        plot_RN_BW_fig: bool = False,
        fifo_utilization_heatmap: bool = False,
        show_result_analysis: bool = False,
    ) -> None:
        """
        配置结果分析选项

        Args:
            result_save_path: 结果保存路径
            results_fig_save_path: 图表保存路径（已弃用，使用show_fig控制图像显示）
            plot_flow_fig: 是否绘制静态流量图（PNG）
            flow_graph_interactive: 是否绘制交互式流量图（HTML）
            plot_RN_BW_fig: 是否绘制RN带宽图
            fifo_utilization_heatmap: 是否绘制FIFO使用率热力图
            show_result_analysis: 是否显示结果分析（在浏览器中打开），默认True
        """
        self.plot_flow_fig = plot_flow_fig
        self.flow_graph_interactive = flow_graph_interactive
        self.plot_RN_BW_fig = plot_RN_BW_fig
        self.fifo_utilization_heatmap = fifo_utilization_heatmap

        # 创建结果保存路径
        if result_save_path:
            self.result_save_path = f"{result_save_path}{self.topo_type_stat}/{self.file_name[:-4]}/"
            os.makedirs(self.result_save_path, exist_ok=True)
            # 交互式流图始终保存到result_save_path
            self.results_fig_save_path = self.result_save_path
        elif results_fig_save_path:
            self.results_fig_save_path = results_fig_save_path
            os.makedirs(self.results_fig_save_path, exist_ok=True)
        else:
            self.results_fig_save_path = None

        # show_fig参数控制是否在浏览器中显示图像
        self.show_result_analysis = show_result_analysis

    def setup_debug(
        self,
        print_trace: bool = False,
        show_trace_id: list = None,
        update_interval: float = 0.0,
    ) -> None:
        """
        配置调试选项

        Args:
            print_trace: 是否打印trace信息
            show_trace_id: 要跟踪的packet ID列表
            update_interval: 每个周期的暂停时间（秒），用于实时观察
        """
        self.print_trace = print_trace
        self.show_trace_id = show_trace_id if show_trace_id is not None else []
        self.update_interval = update_interval

    def setup_visualization(
        self,
        plot_link_state: bool = False,
        plot_start_cycle: int = -1,
        show_node_id: int = 3,
    ) -> None:
        """
        配置实时可视化选项

        Args:
            plot_link_state: 是否启用链路状态可视化
            plot_start_cycle: 开始可视化的周期
        """
        self.plot_link_state = plot_link_state
        self.plot_start_cycle = plot_start_cycle
        self.show_node_id = show_node_id

    def run_simulation(
        self,
        max_time: int = 10000,
        print_interval: int = 1000,
    ) -> None:
        """
        运行仿真 - 统一入口，封装 initial() + run()

        Args:
            max_cycles: 最大仿真周期数
            print_interval: 打印进度的间隔周期数
        """
        # 初始化
        self.initial()

        # 设置仿真参数（在initial()之后，避免被覆盖）
        self.end_time = max_time
        self.print_interval = print_interval

        # 运行仿真
        # print("\n提示: 按 Ctrl+C 可以随时中断仿真并查看当前结果\n")
        self.run()

    def initial(self):
        self.topo_type_stat = self.config.TOPO_TYPE
        self.config.update_config(self.topo_type_stat)
        self.adjacency_matrix = create_adjacency_matrix("CrossRing", self.config.NUM_NODE, self.config.NUM_COL)
        self.req_network = Network(self.config, self.adjacency_matrix, name="Request Network")
        self.rsp_network = Network(self.config, self.adjacency_matrix, name="Response Network")
        self.data_network = Network(self.config, self.adjacency_matrix, name="Data Network")
        self.result_processor = SingleDieAnalyzer(
            self.config,
            min_gap_threshold=200,
            plot_rn_bw_fig=self.plot_RN_BW_fig,
            plot_flow_graph=self.plot_flow_fig,
            flow_graph_interactive=self.flow_graph_interactive,
            show_result_analysis=self.show_result_analysis,
            verbose=self.verbose,
        )
        if self.plot_link_state:
            self.link_state_vis = NetworkLinkVisualizer(self.data_network)

        # 智能设置各network的双侧升级：全局配置 OR (双侧下环/动态方向 AND 在保序列表中)
        self.req_network.ETAG_BOTHSIDE_UPGRADE = self.config.ETAG_BOTHSIDE_UPGRADE or (self.config.ORDERING_PRESERVATION_MODE in [2, 3] and "REQ" in self.config.IN_ORDER_PACKET_CATEGORIES)
        self.rsp_network.ETAG_BOTHSIDE_UPGRADE = self.config.ETAG_BOTHSIDE_UPGRADE or (self.config.ORDERING_PRESERVATION_MODE in [2, 3] and "RSP" in self.config.IN_ORDER_PACKET_CATEGORIES)
        self.data_network.ETAG_BOTHSIDE_UPGRADE = self.config.ETAG_BOTHSIDE_UPGRADE or (self.config.ORDERING_PRESERVATION_MODE in [2, 3] and "DATA" in self.config.IN_ORDER_PACKET_CATEGORIES)

        # Initialize arbiters based on configuration
        arbitration_config = getattr(self.config, "arbitration", {})

        # Create arbiters for different queue types
        default_config = arbitration_config.get("default", {"type": "round_robin"})
        self.iq_arbiter = create_arbiter_from_config(arbitration_config.get("iq", default_config))
        self.eq_arbiter = create_arbiter_from_config(arbitration_config.get("eq", default_config))
        self.rb_arbiter = create_arbiter_from_config(arbitration_config.get("rb", default_config))

        # 缓存网络类型到IP类型的映射
        self.network_ip_types = {
            "req": [ip_type for ip_type in self.config.CH_NAME_LIST if ip_type.startswith("gdma") or ip_type.startswith("sdma") or ip_type.startswith("cdma")],
            "rsp": [ip_type for ip_type in self.config.CH_NAME_LIST if ip_type.startswith("ddr") or ip_type.startswith("l2m")],
            "data": self.config.CH_NAME_LIST,  # data网络不筛选
        }
        # 使用新的XY/YX确定性路由替代networkx最短路径
        self.routes = self._build_routing_table()

        # 动态创建IP接口
        self.ip_modules = {}
        self._create_dynamic_ip_interfaces()

        # 为所有IP接口设置request_tracker
        for ip_interface in self.ip_modules.values():
            ip_interface.request_tracker = self.request_tracker

        self.flits = []
        self.throughput_time = []
        self.data_count = 0
        self.req_count = 0
        self.flit_id_count = 0
        self.send_flits_num = 0
        self.send_reqs_num = 0
        self.trans_flits_num = 0
        self.end_time = np.inf
        self.print_interval = 5000
        self._last_printed_cycle = -1
        self.flit_num, self.req_num, self.rsp_num = 0, 0, 0
        self.new_write_req = []
        # v2架构: IQ_directions和IQ_direction_conditions已移至RingStation仲裁
        self.read_ip_intervals = defaultdict(list)  # 存储每个IP的读请求时间区间
        self.write_ip_intervals = defaultdict(list)  # 存储每个IP的写请求时间区间

        # 新架构: 所有节点都可以作为IP节点
        self.dma_rw_counts = self.config._make_channels(
            ("gdma", "sdma", "cdma"),
            {ip: {"read": 0, "write": 0} for ip in range(self.config.NUM_NODE)},
        )

        self.rn_bandwidth = {
            "SDMA read DDR": {"time": [], "bandwidth": []},
            "SDMA read L2M": {"time": [], "bandwidth": []},
            "GDMA read DDR": {"time": [], "bandwidth": []},
            "GDMA read L2M": {"time": [], "bandwidth": []},
            "SDMA write DDR": {"time": [], "bandwidth": []},
            "SDMA write L2M": {"time": [], "bandwidth": []},
            "GDMA write DDR": {"time": [], "bandwidth": []},
            "GDMA write L2M": {"time": [], "bandwidth": []},
            "total": {"time": [], "bandwidth": []},
        }

        # statistical data
        self.send_read_flits_num_stat = 0
        self.send_write_flits_num_stat = 0
        self.file_name_stat = self.file_name[:-4]
        self.R_finish_time_stat, self.W_finish_time_stat = 0, 0
        self.Total_finish_time_stat = 0
        self.R_tail_latency_stat, self.W_tail_latency_stat = 0, 0
        self.req_cir_h_num_stat, self.req_cir_v_num_stat = 0, 0
        self.rsp_cir_h_num_stat, self.rsp_cir_v_num_stat = 0, 0
        self.data_cir_h_num_stat, self.data_cir_v_num_stat = 0, 0
        # 反方向上环统计
        self.req_reverse_h_num_stat, self.req_reverse_v_num_stat = 0, 0
        self.rsp_reverse_h_num_stat, self.rsp_reverse_v_num_stat = 0, 0
        self.data_reverse_h_num_stat, self.data_reverse_v_num_stat = 0, 0
        self.req_wait_cycle_h_num_stat, self.req_wait_cycle_v_num_stat = 0, 0
        self.rsp_wait_cycle_h_num_stat, self.rsp_wait_cycle_v_num_stat = 0, 0
        self.data_wait_cycle_h_num_stat, self.data_wait_cycle_v_num_stat = 0, 0
        self.read_retry_num_stat, self.write_retry_num_stat = 0, 0
        self.EQ_ETag_T1_num_stat, self.EQ_ETag_T0_num_stat = 0, 0
        self.RB_ETag_T1_num_stat, self.RB_ETag_T0_num_stat = 0, 0

        # Per-node FIFO ETag statistics
        self.EQ_ETag_T1_per_node_fifo = {}  # {node_id: {"TU": count, "TD": count}}
        self.EQ_ETag_T0_per_node_fifo = {}  # {node_id: {"TU": count, "TD": count}}
        self.RB_ETag_T1_per_node_fifo = {}  # {node_id: {"TU": count, "TD": count, "TL": count, "TR": count}}
        self.RB_ETag_T0_per_node_fifo = {}  # {node_id: {"TU": count, "TD": count, "TL": count, "TR": count}}

        # 总数据量统计 (每个节点下环到RB和EQ的flit总数)
        self.EQ_total_flits_per_node = {}  # {node_id: {"TU": count, "TD": count}}
        self.RB_total_flits_per_node = {}  # {node_id: {"TU": count, "TD": count, "TL": count, "TR": count}}

        # 按通道类型分开的ETag统计
        self.EQ_ETag_T1_per_channel = {"req": {}, "rsp": {}, "data": {}}  # {channel: {node_id: {"TU": count, "TD": count}}}
        self.EQ_ETag_T0_per_channel = {"req": {}, "rsp": {}, "data": {}}
        self.RB_ETag_T1_per_channel = {"req": {}, "rsp": {}, "data": {}}  # {channel: {node_id: {"TU": count, "TD": count, "TL": count, "TR": count}}}
        self.RB_ETag_T0_per_channel = {"req": {}, "rsp": {}, "data": {}}

        # 按通道类型分开的总数据量统计
        self.EQ_total_flits_per_channel = {"req": {}, "rsp": {}, "data": {}}  # {channel: {node_id: {"TU": count, "TD": count}}}
        self.RB_total_flits_per_channel = {"req": {}, "rsp": {}, "data": {}}  # {channel: {node_id: {"TU": count, "TD": count, "TL": count, "TR": count}}}
        self.ITag_h_num_stat, self.ITag_v_num_stat = 0, 0
        self.Total_sum_BW_stat = 0

        # Mixed (total) bandwidth/latency stats initialization
        self.mixed_unweighted_bw_stat = 0
        self.mixed_weighted_bw_stat = 0
        self.cmd_mixed_avg_latency_stat = 0
        self.cmd_mixed_max_latency_stat = 0
        self.data_mixed_avg_latency_stat = 0
        self.data_mixed_max_latency_stat = 0
        self.trans_mixed_avg_latency_stat = 0
        self.trans_mixed_max_latency_stat = 0
        # Overall average bandwidth stats (unweighted and weighted)
        self.total_unweighted_bw_stat = 0
        self.total_weighted_bw_stat = 0

        # Performance monitoring - simple simulation time tracking
        self.simulation_start_time = None

        # Initialize per-node FIFO ETag statistics after networks are created
        self._initialize_per_node_etag_stats()

        # 初始化Network的channel buffer(延迟初始化)
        self.req_network.initialize_buffers()
        self.rsp_network.initialize_buffers()
        self.data_network.initialize_buffers()

    def _create_dynamic_ip_interfaces(self):
        """
        动态创建IP接口 - 只创建traffic中实际使用的IP

        根据traffic文件解析结果和D2D配置创建IP接口:
        1. 为traffic中出现的(node_id, ip_type)创建普通IP接口
        2. 根据D2D_CONNECTIONS配置创建d2d_rn和d2d_sn接口
        """
        # 1. 收集需要创建的IP列表
        required_ips = set()

        # 从traffic中提取的IP需求
        if hasattr(self, "_required_ips"):
            required_ips.update(self._required_ips)

        # 2. 添加D2D接口(根据D2D_CONNECTIONS配置)
        if hasattr(self.config, "D2D_CONNECTIONS") and self.config.D2D_CONNECTIONS:
            d2d_ips = self._get_d2d_ips_from_config()
            required_ips.update(d2d_ips)

        # 3. 创建IP接口
        for node_id, ip_type in required_ips:
            self._create_single_ip_interface(ip_type, node_id)

    def _get_d2d_ips_from_config(self):
        """
        从D2D_CONNECTIONS配置中提取需要创建的D2D IP接口

        Returns:
            Set[(node_id, ip_type)]: D2D IP接口集合
        """
        d2d_ips = set()

        if not hasattr(self.config, "D2D_CONNECTIONS"):
            return d2d_ips

        # D2D_CONNECTIONS格式: [[src_die, src_node, dst_die, dst_node], ...]
        for connection in self.config.D2D_CONNECTIONS:
            src_die, src_node, dst_die, dst_node = connection

            # 为源节点和目标节点都创建d2d_rn_0和d2d_sn_0
            d2d_ips.add((src_node, "d2d_rn_0"))
            d2d_ips.add((src_node, "d2d_sn_0"))
            d2d_ips.add((dst_node, "d2d_rn_0"))
            d2d_ips.add((dst_node, "d2d_sn_0"))

        return d2d_ips

    def _create_single_ip_interface(self, ip_type, node_id):
        """
        创建单个IP接口

        Args:
            ip_type: IP类型 (如"gdma_0", "d2d_rn_0")
            node_id: 节点ID
        """
        # 避免重复创建
        if (ip_type, node_id) in self.ip_modules:
            return

        # 根据IP类型创建相应的接口
        if ip_type == "d2d_rn_0":
            from src.dcin.components import D2D_RN_Interface

            self.ip_modules[(ip_type, node_id)] = D2D_RN_Interface(
                ip_type,
                node_id,
                self.config,
                self.req_network,
                self.rsp_network,
                self.data_network,
                self.routes,
            )
        elif ip_type == "d2d_sn_0":
            from src.dcin.components import D2D_SN_Interface

            self.ip_modules[(ip_type, node_id)] = D2D_SN_Interface(
                ip_type,
                node_id,
                self.config,
                self.req_network,
                self.rsp_network,
                self.data_network,
                self.routes,
            )
        else:
            # 普通IP接口
            self.ip_modules[(ip_type, node_id)] = IPInterface(
                ip_type,
                node_id,
                self.config,
                self.req_network,
                self.rsp_network,
                self.data_network,
                self.routes,
            )

    def _initialize_per_node_etag_stats(self):
        """Initialize per-node FIFO ETag statistics dictionaries."""
        # Get all IP positions from the request network
        all_nodes = self.req_network.all_ip_positions

        # Initialize EQ ETag statistics (only TU and TD directions)
        for node_id in all_nodes:
            self.EQ_ETag_T1_per_node_fifo[node_id] = {"TU": 0, "TD": 0}
            self.EQ_ETag_T0_per_node_fifo[node_id] = {"TU": 0, "TD": 0}
            self.EQ_total_flits_per_node[node_id] = {"TU": 0, "TD": 0}

            # Initialize per-channel EQ statistics
            for channel in ["req", "rsp", "data"]:
                self.EQ_ETag_T1_per_channel[channel][node_id] = {"TU": 0, "TD": 0}
                self.EQ_ETag_T0_per_channel[channel][node_id] = {"TU": 0, "TD": 0}
                self.EQ_total_flits_per_channel[channel][node_id] = {"TU": 0, "TD": 0}

        # Initialize RB ETag statistics (all four directions)
        for node_id in all_nodes:
            self.RB_ETag_T1_per_node_fifo[node_id] = {"TU": 0, "TD": 0, "TL": 0, "TR": 0}
            self.RB_ETag_T0_per_node_fifo[node_id] = {"TU": 0, "TD": 0, "TL": 0, "TR": 0}
            self.RB_total_flits_per_node[node_id] = {"TU": 0, "TD": 0, "TL": 0, "TR": 0}

            # Initialize per-channel RB statistics
            for channel in ["req", "rsp", "data"]:
                self.RB_ETag_T1_per_channel[channel][node_id] = {"TU": 0, "TD": 0, "TL": 0, "TR": 0}
                self.RB_ETag_T0_per_channel[channel][node_id] = {"TU": 0, "TD": 0, "TL": 0, "TR": 0}
                self.RB_total_flits_per_channel[channel][node_id] = {"TU": 0, "TD": 0, "TL": 0, "TR": 0}

    def step(self):
        """Execute one simulation cycle step."""
        # Tag moves
        self.release_completed_sn_tracker()

        self.process_new_request()

        self.tag_move_all_networks()

        self.ip_inject_to_network()

        # Network arbitration and movement
        self._step_reqs = self.move_flits_in_network(self.req_network, self._step_reqs, "req")
        self._step_rsps = self.move_flits_in_network(self.rsp_network, self._step_rsps, "rsp")
        self._step_flits = self.move_flits_in_network(self.data_network, self._step_flits, "data")

        self.network_to_ip_eject()

        self.move_pre_to_queues_all()

        # Collect statistics
        self.req_network.collect_cycle_end_link_statistics(self.cycle)
        self.rsp_network.collect_cycle_end_link_statistics(self.cycle)
        self.data_network.collect_cycle_end_link_statistics(self.cycle)

        self.debug_func()

        # Evaluate throughput time
        self.update_throughput_metrics(self._step_flits)

        # 检查traffic完成情况并推进链
        completed_traffics = self.traffic_scheduler.check_and_advance_chains(self.cycle)
        if completed_traffics and self.verbose:
            print(f"Completed traffics: {completed_traffics}")

        # 处理延迟释放的Entry
        self._process_pending_entry_release(self.req_network)
        self._process_pending_entry_release(self.rsp_network)
        self._process_pending_entry_release(self.data_network)

    def is_completed(self):
        """Check if this die's simulation is completed."""
        traffic_completed = self.traffic_scheduler.is_all_completed()
        recv_completed = self.data_network.recv_flits_num >= (self.read_flit + self.write_flit)
        trans_completed = self.trans_flits_num == 0
        write_completed = not self.new_write_req

        return traffic_completed and recv_completed and trans_completed and write_completed

    def run(self):
        """Main simulation loop."""
        simulation_start = time.perf_counter()
        self.load_request_stream()
        # Initialize step variables
        self._step_flits, self._step_reqs, self._step_rsps = [], [], []
        self.cycle = 0
        tail_time = 6

        try:
            while True:
                # 检查取消标志
                if self._cancelled:
                    if self.verbose:
                        print("\n仿真被取消，正在保存结果...")
                    break

                self.cycle += 1
                self.cycle_mod = self.cycle % self.config.CYCLES_PER_NS

                # Execute one step
                self.step()

                if self.cycle / self.config.CYCLES_PER_NS % self.print_interval == 0:
                    self.log_summary()

                if self.is_completed() or self.cycle > self.end_time * self.config.CYCLES_PER_NS:
                    if tail_time == 0:
                        if self.verbose:
                            print("Finish!")
                        break
                    else:
                        tail_time -= 1

        except KeyboardInterrupt:
            print("\n仿真中断 (Ctrl+C)，正在退出...")
            # 不重新抛出异常，继续执行结果分析
        except Exception as e:
            print(f"\n仿真过程中出现错误: {e}")
            raise
        finally:
            # 确保仿真结束状态被正确设置
            pass

        # Performance evaluation
        self.print_data_statistic()
        self.log_summary()
        self.syn_IP_stat()

        # 更新结束时间统计
        self.update_finish_time_stats()

        # 结果统计
        self.process_comprehensive_results()

        # Record simulation performance
        simulation_end = time.perf_counter()
        simulation_time = simulation_end - simulation_start
        self.simulation_total_time = simulation_time

        if self.verbose:
            print()
            print(f"Simulation completed in {simulation_time:.2f} seconds, Performance: {self.cycle / simulation_time:.0f} cycles/second")

    def print_data_statistic(self):
        if self.verbose:
            print(f"Data statistic: Read: {self.read_req, self.read_flit}, " f"Write: {self.write_req, self.write_flit}, " f"Total: {self.read_req + self.write_req, self.read_flit + self.write_flit}")

    def log_summary(self):
        current_time = self.cycle // self.config.CYCLES_PER_NS
        total_req = getattr(self, 'read_req', 0) + getattr(self, 'write_req', 0)
        total_flits = getattr(self, 'read_flit', 0) + getattr(self, 'write_flit', 0)
        summary_data = {
            "current_time": current_time,
            "max_time": self.end_time,
            "progress": min(100, int(current_time / self.end_time * 100)) if self.end_time > 0 else 0,
            "req_count": self.req_count,
            "total_req": total_req,
            "in_req": self.req_num,
            "rsp": self.rsp_num,
            "read_flits": self.send_read_flits_num_stat,
            "write_flits": self.send_write_flits_num_stat,
            "trans_flits": self.trans_flits_num,
            "recv_flits": self.data_network.recv_flits_num,
            "total_flits": total_flits,
        }

        # 调用进度回调（如果设置了）
        if hasattr(self, 'progress_callback') and self.progress_callback:
            self.progress_callback(summary_data)

        if self.verbose:
            print(
                f"T: {current_time}, Req_cnt: {self.req_count} In_Req: {self.req_num}, Rsp: {self.rsp_num},"
                f" R_fn: {self.send_read_flits_num_stat}, W_fn: {self.send_write_flits_num_stat}, "
                f"Trans_fn: {self.trans_flits_num}, Recv_fn: {self.data_network.recv_flits_num}"
            )

    def update_throughput_metrics(self, flits):
        """Update throughput metrics based on flit counts."""
        self.trans_flits_num = len(flits)

    def load_request_stream(self):
        """修改：使用TrafficScheduler来管理请求流"""
        # 启动初始的traffic
        self.traffic_scheduler.start_initial_traffics()

        # 从TrafficScheduler获取统计信息
        total_stats = self._get_total_traffic_stats()
        self.read_req = total_stats["read_req"]
        self.write_req = total_stats["write_req"]
        self.read_flit = total_stats["read_flit"]
        self.write_flit = total_stats["write_flit"]

        # 统计输出保持原行为
        self.print_data_statistic()

    def _get_total_traffic_stats(self):
        """从所有链中统计总的请求和flit数量"""
        total_read_req = total_write_req = total_read_flit = total_write_flit = 0

        for chain in self.traffic_scheduler.parallel_chains:
            for traffic_file in chain.traffic_files:
                abs_path = os.path.join(self.traffic_file_path, traffic_file)
                with open(abs_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        # 跳过空行和注释行（元数据行）
                        if not line or line.startswith("#"):
                            continue

                        parts = line.split(",")
                        if len(parts) >= 7:
                            op, burst = parts[5], int(parts[6])
                            if op == "R":

                                total_read_req += 1
                                total_read_flit += burst
                            else:
                                total_write_req += 1
                                total_write_flit += burst

        return {"read_req": total_read_req, "write_req": total_write_req, "read_flit": total_read_flit, "write_flit": total_write_flit}

    def process_new_request(self):
        """修改：使用TrafficScheduler获取多个准备就绪的请求"""
        # 从TrafficScheduler获取所有准备就绪的请求
        ready_requests = self.traffic_scheduler.get_ready_requests(self.cycle)

        if not ready_requests:
            return  # 没有待处理的请求

        # 处理所有准备就绪的请求
        for req_data in ready_requests:
            self._process_single_request(req_data)

    def _process_single_request(self, req_data):
        """处理单个请求"""
        # 解析请求数据（注意最后一个元素是traffic_id）
        source = req_data[1]  # 物理直接映射,无需node_map转换
        destination = req_data[3]  # 物理直接映射,无需node_map转换
        path = self.routes[source][destination]
        traffic_id = req_data[7]  # 最后一个元素是traffic_id

        # 创建flit对象 (使用对象池)
        req = Flit.create_flit(source, destination, path)
        req.flit_type = "req"
        # 保序信息将在inject_fifo出队时分配（inject_to_l2h_pre）
        req.departure_cycle = req_data[0]
        req.burst_length = req_data[6]
        req.source_type = f"{req_data[2]}_0" if "_" not in req_data[2] else req_data[2]
        req.destination_type = f"{req_data[4]}_0" if "_" not in req_data[4] else req_data[4]
        # Die内请求：source/destination即为原始节点，无需设置_original属性
        # 辅助函数会自动使用source/destination作为回退值
        req.traffic_id = traffic_id  # 添加traffic_id标记

        req.packet_id = BaseModel.get_next_packet_id()
        req.req_type = "read" if req_data[5] == "R" else "write"
        req.req_attr = "new"
        # req.cmd_entry_cake0_cycle = self.cycle

        # 在RequestTracker中开始追踪请求
        if hasattr(self, "request_tracker") and self.request_tracker:
            self.request_tracker.start_request(
                packet_id=req.packet_id,
                source=source,
                destination=destination,
                source_type=req.source_type,
                dest_type=req.destination_type,
                op_type=req.req_type,
                burst_size=req_data[6],
                cycle=self.cycle,
                is_cross_die=False,  # NoC内部请求不跨Die
                origin_die=None,
                target_die=None,
            )

        try:
            # 通过IPInterface处理请求
            node_id = req.source
            ip_type = req.source_type

            ip_interface: IPInterface = self.ip_modules[(ip_type, node_id)]
            if ip_interface is None:
                raise ValueError(f"IP module setup error for ({ip_type}, {node_id})!")

            # 检查IP接口是否能接受新请求（可选的流控机制）
            if hasattr(ip_interface, "can_accept_request") and not ip_interface.can_accept_request():
                if self.traffic_scheduler.verbose:
                    print(f"Warning: IP interface ({ip_type}, {node_id}) is busy, request may be delayed")

            ip_interface.enqueue(req, "req")

            # 更新统计信息
            if req.req_type == "read":
                self.R_tail_latency_stat = req_data[0]
            elif req.req_type == "write":
                self.W_tail_latency_stat = req_data[0]

            # 更新请求计数
            self.req_count += 1

        except Exception as e:
            logging.exception(f"Error processing request {req_data} in chain {traffic_id}")

    def error_log(self, flit, target_id, flit_id):
        if flit and flit.packet_id == target_id and flit.flit_id == flit_id:
            print(inspect.currentframe().f_back.f_code.co_name, self.cycle, flit)

    def _calculate_path_xy(self, src, dst):
        """
        XY路由: 先水平后垂直 (参考C2C实现)

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            路径列表 [src, ..., dst]
        """
        if src == dst:
            return [src]

        path = [src]
        current = src
        src_row, src_col = divmod(src, self.config.NUM_COL)
        dst_row, dst_col = divmod(dst, self.config.NUM_COL)

        # 1. 水平移动
        while src_col != dst_col:
            src_col += 1 if dst_col > src_col else -1
            current = src_row * self.config.NUM_COL + src_col
            path.append(current)

        # 2. 垂直移动
        while src_row != dst_row:
            src_row += 1 if dst_row > src_row else -1
            current = src_row * self.config.NUM_COL + src_col
            path.append(current)

        return path

    def _calculate_path_yx(self, src, dst):
        """
        YX路由: 先垂直后水平 (参考C2C实现)

        Args:
            src: 源节点ID
            dst: 目标节点ID

        Returns:
            路径列表 [src, ..., dst]
        """
        if src == dst:
            return [src]

        path = [src]
        current = src
        src_row, src_col = divmod(src, self.config.NUM_COL)
        dst_row, dst_col = divmod(dst, self.config.NUM_COL)

        # 1. 垂直移动
        while src_row != dst_row:
            src_row += 1 if dst_row > src_row else -1
            current = src_row * self.config.NUM_COL + src_col
            path.append(current)

        # 2. 水平移动
        while src_col != dst_col:
            src_col += 1 if dst_col > src_col else -1
            current = src_row * self.config.NUM_COL + src_col
            path.append(current)

        return path

    def _build_routing_table(self):
        """
        构建路由表: 预计算所有节点对的路径

        新架构: 使用XY/YX确定性路由替代networkx最短路径
        """
        routing_strategy = getattr(self.config, "ROUTING_STRATEGY", "XY")

        routes = {}
        for src in range(self.config.NUM_NODE):
            routes[src] = {}
            for dst in range(self.config.NUM_NODE):
                if routing_strategy == "YX":
                    routes[src][dst] = self._calculate_path_yx(src, dst)
                else:  # 默认XY路由
                    routes[src][dst] = self._calculate_path_xy(src, dst)

        return routes
