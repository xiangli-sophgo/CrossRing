from collections import deque, defaultdict

from src.utils.optimal_placement import create_adjacency_matrix, find_shortest_paths
from config.config import CrossRingConfig
from src.utils.components import Flit, Network, TokenBucket, IPInterface

from src.core.Link_State_Visualizer import NetworkLinkVisualizer
import matplotlib.pyplot as plt
import os
import sys, time
import inspect, logging
from functools import wraps, lru_cache
from src.core.result_processor import *
from src.core.traffic_scheduler import TrafficScheduler
from src.utils.arbitration import create_arbiter_from_config
import threading


class BaseModel:
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

    def __init__(self, model_type, config: CrossRingConfig, topo_type, verbose: int = 0):
        """
        初始化BaseModel - 仅设置核心属性

        Args:
            model_type: 模型类型（REQ_RSP, Packet_Base, Feature）
            config: CrossRingConfig配置对象
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
        self.result_save_path_original = None
        self.results_fig_save_path = None

        # 可视化配置 - 通过setup_result_analysis设置
        self.plot_flow_fig = False

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

    def setup_traffic_scheduler(self, traffic_file_path: str, traffic_chains: list) -> None:
        """
        配置流量调度器

        Args:
            traffic_file_path: 流量文件路径
            traffic_chains: 流量链配置，每个链包含文件名列表
                           例如: [["file1.txt", "file2.txt"], ["file3.txt"]]
        """
        self.traffic_file_path = traffic_file_path

        # 初始化TrafficScheduler
        self.traffic_scheduler = TrafficScheduler(self.config, traffic_file_path)
        self.traffic_scheduler.set_verbose(self.verbose > 0)

        # 处理traffic配置
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

    def setup_result_analysis(
        self,
        result_save_path: str = "",
        results_fig_save_path: str = "",
        plot_flow_fig: bool = False,
        plot_RN_BW_fig: bool = False,
        fifo_utilization_heatmap: bool = False,
        save_fig: bool = True,
    ) -> None:
        """
        配置结果分析选项

        Args:
            result_save_path: 结果保存路径
            results_fig_save_path: 图表保存路径（已弃用，使用save_fig控制图像保存）
            plot_flow_fig: 是否绘制流量图
            plot_RN_BW_fig: 是否绘制RN带宽图
            fifo_utilization_heatmap: 是否绘制FIFO使用率热力图
            save_fig: 是否保存图像到数据流保存目录，默认True
        """
        self.result_save_path_original = result_save_path
        self.plot_flow_fig = plot_flow_fig
        self.plot_RN_BW_fig = plot_RN_BW_fig
        self.fifo_utilization_heatmap = fifo_utilization_heatmap

        # 创建结果保存路径
        if result_save_path:
            self.result_save_path = f"{result_save_path}{self.topo_type_stat}/{self.file_name[:-4]}/"
            os.makedirs(self.result_save_path, exist_ok=True)

        # 根据save_fig参数决定图像保存位置
        if save_fig and result_save_path:
            self.results_fig_save_path = self.result_save_path
        elif results_fig_save_path:
            self.results_fig_save_path = results_fig_save_path
            os.makedirs(self.results_fig_save_path, exist_ok=True)
        else:
            self.results_fig_save_path = None

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
        self.result_processor = BandwidthAnalyzer(self.config, min_gap_threshold=200, plot_rn_bw_fig=self.plot_RN_BW_fig, plot_flow_graph=self.plot_flow_fig)
        if self.plot_link_state:
            self.link_state_vis = NetworkLinkVisualizer(self.data_network)

        # 智能设置各network的双侧升级：全局配置 OR (双侧下环 AND 在保序列表中)
        self.req_network.ETag_BOTHSIDE_UPGRADE = self.config.ETag_BOTHSIDE_UPGRADE or (self.config.ORDERING_PRESERVATION_MODE == 2 and "REQ" in self.config.IN_ORDER_PACKET_CATEGORIES)
        self.rsp_network.ETag_BOTHSIDE_UPGRADE = self.config.ETag_BOTHSIDE_UPGRADE or (self.config.ORDERING_PRESERVATION_MODE == 2 and "RSP" in self.config.IN_ORDER_PACKET_CATEGORIES)
        self.data_network.ETag_BOTHSIDE_UPGRADE = self.config.ETag_BOTHSIDE_UPGRADE or (self.config.ORDERING_PRESERVATION_MODE == 2 and "DATA" in self.config.IN_ORDER_PACKET_CATEGORIES)

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
        self.ip_modules = {}
        for node_id in range(self.config.NUM_NODE):
            for ip_type in self.config.CH_NAME_LIST:
                # 检查是否是D2D接口类型
                if ip_type == "d2d_rn_0":
                    from src.utils.components.d2d_rn_interface import D2D_RN_Interface

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
                    from src.utils.components.d2d_sn_interface import D2D_SN_Interface

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
        self.IQ_directions = ["TR", "TL", "TU", "TD", "EQ"]
        # 新架构: 物理直接映射, 相邻节点差值简化
        # TR: +1 (向右), TL: -1 (向左), TU: -NUM_COL (向上), TD: +NUM_COL (向下)
        self.IQ_direction_conditions = {
            "TR": lambda flit: len(flit.path) > 1 and flit.path[1] - flit.path[0] == 1,
            "TL": lambda flit: len(flit.path) > 1 and flit.path[1] - flit.path[0] == -1,
            "TU": lambda flit: len(flit.path) > 1 and flit.path[1] - flit.path[0] == -self.config.NUM_COL,
            "TD": lambda flit: len(flit.path) > 1 and flit.path[1] - flit.path[0] == self.config.NUM_COL,
            "EQ": lambda flit: len(flit.path) == 1,  # 源=目标,直接下环
        }
        # 如果只有1列，禁用横向注入，仅保留垂直和EQ方向
        if self.config.NUM_COL <= 1:
            self.IQ_directions = ["EQ", "TU", "TD"]
            self.IQ_direction_conditions = {
                "TU": lambda flit: len(flit.path) > 1 and flit.path[1] - flit.path[0] == -self.config.NUM_COL,
                "TD": lambda flit: len(flit.path) > 1 and flit.path[1] - flit.path[0] == self.config.NUM_COL,
                "EQ": lambda flit: len(flit.path) == 1,
            }
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
                self.cycle += 1
                self.cycle_mod = self.cycle % self.config.NETWORK_FREQUENCY

                # Execute one step
                self.step()

                if self.cycle / self.config.NETWORK_FREQUENCY % self.print_interval == 0:
                    self.log_summary()

                if self.is_completed() or self.cycle > self.end_time * self.config.NETWORK_FREQUENCY:
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
            print(f"Simulation completed in {simulation_time:.2f} seconds")
            print(f"Processed {self.cycle} cycles")
            print(f"Performance: {self.cycle / simulation_time:.0f} cycles/second")

    def update_finish_time_stats(self):
        """从traffic_scheduler和result_processor获取结束时间并更新统计"""
        read_end_times = []
        write_end_times = []
        all_end_times = []

        # 从traffic_scheduler获取结束时间统计
        try:
            finish_stats = self.traffic_scheduler.get_finish_time_stats()
            if finish_stats["R_finish_time"] > 0:
                read_end_times.append(finish_stats["R_finish_time"])
                all_end_times.append(finish_stats["R_finish_time"])
            if finish_stats["W_finish_time"] > 0:
                write_end_times.append(finish_stats["W_finish_time"])
                all_end_times.append(finish_stats["W_finish_time"])
            if finish_stats["Total_finish_time"] > 0:
                all_end_times.append(finish_stats["Total_finish_time"])
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not get finish time stats from traffic_scheduler: {e}")

        # 从result_processor获取请求的结束时间
        try:
            if hasattr(self, "result_processor") and hasattr(self.result_processor, "requests"):
                for req_info in self.result_processor.requests:
                    end_time_ns = req_info.end_time // self.config.NETWORK_FREQUENCY
                    all_end_times.append(end_time_ns)
                    if req_info.req_type == "read":
                        read_end_times.append(end_time_ns)
                    elif req_info.req_type == "write":
                        write_end_times.append(end_time_ns)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not get finish time stats from result_processor: {e}")

        # 更新统计数据，使用当前cycle作为备选
        current_time_ns = self.cycle // self.config.NETWORK_FREQUENCY

        if read_end_times:
            self.R_finish_time_stat = max(read_end_times)
        else:
            self.R_finish_time_stat = current_time_ns

        if write_end_times:
            self.W_finish_time_stat = max(write_end_times)
        else:
            self.W_finish_time_stat = current_time_ns

        if all_end_times:
            self.Total_finish_time_stat = max(all_end_times)
        else:
            self.Total_finish_time_stat = current_time_ns

        if self.verbose:
            print(f"Updated finish times - Read: {self.R_finish_time_stat}ns, Write: {self.W_finish_time_stat}ns, Total: {self.Total_finish_time_stat}ns")

    def update_traffic_completion_stats(self, flit):
        """在flit完成时更新TrafficScheduler的统计"""
        # 只有当 flit 真正到达 IP_eject 状态时才更新统计
        if hasattr(flit, "traffic_id") and flit.flit_position.startswith("IP_eject"):
            self.traffic_scheduler.update_traffic_stats(flit.traffic_id, "received_flit")

    def syn_IP_stat(self):
        for node_id in range(self.config.NUM_NODE):
            for ip_type in self.config.CH_NAME_LIST:
                ip_interface: IPInterface = self.ip_modules[(ip_type, node_id)]
                if self.model_type_stat == "REQ_RSP":
                    self.read_retry_num_stat += ip_interface.read_retry_num_stat
                    self.write_retry_num_stat += ip_interface.write_retry_num_stat
                self.req_cir_h_num_stat += ip_interface.req_cir_h_num
                self.req_cir_v_num_stat += ip_interface.req_cir_v_num
                self.rsp_cir_h_num_stat += ip_interface.rsp_cir_h_num
                self.rsp_cir_v_num_stat += ip_interface.rsp_cir_v_num
                self.data_cir_h_num_stat += ip_interface.data_cir_h_num
                self.data_cir_v_num_stat += ip_interface.data_cir_v_num
                self.req_wait_cycle_h_num_stat += ip_interface.req_wait_cycles_h
                self.req_wait_cycle_v_num_stat += ip_interface.req_wait_cycles_v
                self.rsp_wait_cycle_h_num_stat += ip_interface.rsp_wait_cycles_h
                self.rsp_wait_cycle_v_num_stat += ip_interface.rsp_wait_cycles_v
                self.data_wait_cycle_h_num_stat += ip_interface.data_wait_cycles_h
                self.data_wait_cycle_v_num_stat += ip_interface.data_wait_cycles_v

    def debug_func(self):
        if self.print_trace:
            self.flit_trace(self.show_trace_id)
        if self.plot_link_state:
            while self.link_state_vis.paused and not self.link_state_vis.should_stop:
                plt.pause(0.05)
            if self.link_state_vis.should_stop:
                return
            if self.cycle < self.plot_start_cycle:
                return

            self.link_state_vis.update([self.req_network, self.rsp_network, self.data_network], self.cycle)

    def ip_inject_to_network(self):
        for node_id in range(self.config.NUM_NODE):
            for ip_type in self.config.CH_NAME_LIST:
                # 检查IP接口是否存在，避免KeyError
                ip_key = (ip_type, node_id)
                if ip_key in self.ip_modules:
                    ip_interface: IPInterface = self.ip_modules[ip_key]
                    ip_interface.inject_step(self.cycle)

    def network_to_ip_eject(self):
        """从网络到IP的eject步骤，并更新received_flit统计"""
        for node_id in range(self.config.NUM_NODE):
            for ip_type in self.config.CH_NAME_LIST:
                ip_interface: IPInterface = self.ip_modules[(ip_type, node_id)]
                # 执行eject，获取已到达目的IP的flit列表
                ejected_flits = ip_interface.eject_step(self.cycle)
                # 更新TrafficScheduler中的received_flit统计
                if ejected_flits:
                    for flit in ejected_flits:
                        self.update_traffic_completion_stats(flit)

    def release_completed_sn_tracker(self):
        """Check if any trackers can be released based on the current cycle."""
        # 遍历所有IP模块，检查各自的tracker释放队列
        for (ip_type, node_id), ip_interface in self.ip_modules.items():
            for release_time in sorted(ip_interface.sn_tracker_release_time.keys()):
                if release_time > self.cycle:
                    continue
                tracker_list = ip_interface.sn_tracker_release_time.pop(release_time)
                for req in tracker_list:
                    # 检查 tracker 是否还在列表中（避免重复释放）
                    if req in ip_interface.sn_tracker:
                        ip_interface.release_completed_sn_tracker(req)

    def _move_pre_to_queues(self, network: Network, node_id):
        """Move all items from pre-injection queues to injection queues for a given network."""
        # ===  注入队列 *_pre → *_FIFO ===

        # IQ_channel_buffer_pre → IQ_channel_buffer
        for ip_type in network.IQ_channel_buffer_pre.keys():
            queue_pre = network.IQ_channel_buffer_pre[ip_type]
            queue = network.IQ_channel_buffer[ip_type]
            if queue_pre[node_id] and len(queue[node_id]) < self.config.IQ_CH_FIFO_DEPTH:
                flit = queue_pre[node_id]
                flit.flit_position = "IQ_CH"
                queue[node_id].append(flit)
                queue_pre[node_id] = None

        # IQ_pre → IQ_OUT
        for direction in self.IQ_directions:
            queue_pre = network.inject_queues_pre[direction]
            queue = network.inject_queues[direction]
            if queue_pre[node_id] and len(queue[node_id]) < self.config.RB_OUT_FIFO_DEPTH:
                flit = queue_pre[node_id]
                flit.departure_inject_cycle = self.cycle
                flit.flit_position = f"IQ_{direction}"
                queue[node_id].append(flit)
                queue_pre[node_id] = None

        # RB_IN_PRE → RB_IN
        for direction in ["TL", "TR"]:
            queue_pre = network.ring_bridge_pre[direction]
            queue = network.ring_bridge[direction]
            if queue_pre[node_id] and len(queue[node_id]) < self.config.RB_IN_FIFO_DEPTH:
                flit = queue_pre[node_id]
                flit.flit_position = f"RB_{direction}"
                queue[node_id].append(flit)
                queue_pre[node_id] = None

        # RB_OUT_PRE → RB_OUT
        for fifo_pos in ("EQ", "TU", "TD"):
            queue_pre = network.ring_bridge_pre[fifo_pos]
            queue = network.ring_bridge[fifo_pos]
            if queue_pre[node_id] and len(queue[node_id]) < self.config.RB_OUT_FIFO_DEPTH:
                flit = queue_pre[node_id]
                flit.is_arrive = fifo_pos == "EQ"
                flit.flit_position = f"RB_{fifo_pos}"
                queue[node_id].append(flit)
                queue_pre[node_id] = None

        # EQ_IN_PRE → EQ_IN
        for fifo_pos in ("TU", "TD"):
            queue_pre = network.eject_queues_in_pre[fifo_pos]
            queue = network.eject_queues[fifo_pos]
            if queue_pre[node_id] and len(queue[node_id]) < self.config.EQ_IN_FIFO_DEPTH:
                flit = queue_pre[node_id]
                flit.is_arrive = fifo_pos == "EQ"
                flit.flit_position = f"EQ_{fifo_pos}"
                queue[node_id].append(flit)
                queue_pre[node_id] = None

        # EQ_channel_buffer_pre → EQ_channel_buffer
        for ip_type in network.EQ_channel_buffer_pre.keys():
            queue_pre = network.EQ_channel_buffer_pre[ip_type]
            queue = network.EQ_channel_buffer[ip_type]
            if queue_pre[node_id] and len(queue[node_id]) < self.config.EQ_CH_FIFO_DEPTH:
                flit = queue_pre[node_id]
                flit.flit_position = "EQ_CH"
                queue[node_id].append(flit)
                queue_pre[node_id] = None

        # 更新FIFO统计
        network.update_fifo_stats_after_move(node_id)

    def print_data_statistic(self):
        if self.verbose:
            print(f"Data statistic: Read: {self.read_req, self.read_flit}, " f"Write: {self.write_req, self.write_flit}, " f"Total: {self.read_req + self.write_req, self.read_flit + self.write_flit}")

    def log_summary(self):
        if self.verbose:
            print(
                f"T: {self.cycle // self.config.NETWORK_FREQUENCY}, Req_cnt: {self.req_count} In_Req: {self.req_num}, Rsp: {self.rsp_num},"
                f" R_fn: {self.send_read_flits_num_stat}, W_fn: {self.send_write_flits_num_stat}, "
                f"Trans_fn: {self.trans_flits_num}, Recv_fn: {self.data_network.recv_flits_num}"
            )

    def move_flits_in_network(self, network, flits, flit_type):
        """Process injection queues and move flits."""
        flits = self._network_cycle_process(network, flits, flit_type)
        return flits

    def _try_inject_to_direction(self, req: Flit, ip_type, node_id, direction, counts):
        """检查tracker空间并尝试注入到指定direction的pre缓冲"""
        # 设置flit的允许下环方向（仅在第一次注入时设置）
        if not hasattr(req, "allowed_eject_directions") or req.allowed_eject_directions is None:
            req.allowed_eject_directions = self.req_network.determine_allowed_eject_directions(req)

        # 直接注入到指定direction的pre缓冲
        queue_pre = self.req_network.inject_queues_pre[direction]
        queue_pre[node_id] = req

        # 从channel buffer移除
        self.req_network.IQ_channel_buffer[ip_type][node_id].popleft()

        # 更新计数和状态
        if req.req_attr == "new":  # 只有新请求才更新计数器和tracker
            if req.req_type == "read":
                counts["read"] += 1

            elif req.req_type == "write":
                counts["write"] += 1

        # req.cmd_entry_noc_from_cake0_cycle = self.cycle
        return True

    # ------------------------------------------------------------------
    # IQ仲裁：把 channel‑buffer 的 flit/req/rsp 放到 inject_queues_pre
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 模块化处理函数：IQ, Link, RB, EQ, CP
    # ------------------------------------------------------------------
    def _IQ_process(self, network: Network, flit_type: str):
        """IQ模块：IQ仲裁处理

        从channel buffer移动到inject_queues_pre

        Args:
            network: 网络实例 (req_network / rsp_network / data_network)
            flit_type: flit类型 ("req" / "rsp" / "data")
        """
        # 所有节点都可以作为IP节点
        for node_id in range(self.config.NUM_NODE):
            # 1. 收集所有可能的 ip_types 和 directions
            all_ip_types = set()
            for direction in self.IQ_directions:
                rr_queue = network.round_robin["IQ"][direction][node_id]
                all_ip_types.update(rr_queue)

            if not all_ip_types:
                continue

            ip_types_list = sorted(list(all_ip_types))
            directions_list = list(self.IQ_directions)

            # 2. 构建请求矩阵 (ip_types × directions)
            request_matrix = []
            ip_type_to_flit = {}  # 缓存每个ip_type的flit

            for ip_type in ip_types_list:
                row = []
                for direction in directions_list:
                    # 检查是否可以注入到这个方向
                    can_inject = self._check_iq_injection_conditions(network, node_id, ip_type, direction, flit_type, ip_type_to_flit)
                    row.append(can_inject)
                request_matrix.append(row)

            # 3. 执行匹配
            if not any(any(row) for row in request_matrix):
                continue  # 没有有效请求

            queue_id = f"IQ_pos{node_id}_{flit_type}"
            matches = self.iq_arbiter.match(request_matrix, queue_id=queue_id)

            # 4. 根据匹配结果处理注入
            for ip_idx, dir_idx in matches:
                ip_type = ip_types_list[ip_idx]
                direction = directions_list[dir_idx]

                flit = ip_type_to_flit.get((ip_type, direction))
                if not flit:
                    continue

                # 执行注入
                if flit_type == "req":
                    counts = None
                    if not ip_type.startswith("d2d_rn"):
                        counts = self.dma_rw_counts[ip_type][node_id]
                    else:
                        counts = self.dma_rw_counts.get(ip_type, {}).get(node_id, {"read": 0, "write": 0})

                    self._try_inject_to_direction(flit, ip_type, node_id, direction, counts)
                else:
                    # rsp / data 网络：直接移动到 pre‑缓冲
                    # 设置flit的允许下环方向（仅在第一次注入时设置）
                    if not hasattr(flit, "allowed_eject_directions") or flit.allowed_eject_directions is None:
                        flit.allowed_eject_directions = network.determine_allowed_eject_directions(flit)

                    network.IQ_channel_buffer[ip_type][node_id].popleft()
                    queue_pre = network.inject_queues_pre[direction]
                    queue_pre[node_id] = flit

                    if flit_type == "rsp":
                        flit.rsp_entry_network_cycle = self.cycle
                    elif flit_type == "data":
                        req = self.req_network.send_flits[flit.packet_id][0]
                        flit.sync_latency_record(req)
                        self.send_flits_num += 1
                        self.trans_flits_num += 1

                        if hasattr(req, "req_type"):
                            if req.req_type == "read":
                                self.send_read_flits_num_stat += 1
                            elif req.req_type == "write":
                                self.send_write_flits_num_stat += 1

                        if hasattr(flit, "traffic_id"):
                            self.traffic_scheduler.update_traffic_stats(flit.traffic_id, "sent_flit")

    def _Link_process(self, network: Network, flits):
        """Link模块：Link传输处理

        处理Link上的flit移动

        Args:
            network: 网络实例
            flits: 当前网络中的flit列表

        Returns:
            list: 更新后的flits列表
        """
        # 第一步：对Link上的flit执行plan_move
        for flit in flits:
            if flit.flit_position == "Link":
                network.plan_move(flit, self.cycle)

        # 第二步：执行execute_moves并收集需要移除的flit
        executed_flits = set()
        for flit in flits:
            if network.execute_moves(flit, self.cycle):
                executed_flits.add(id(flit))

        # 第三步：一次过滤重建列表（O(n)，比多次remove快）
        if executed_flits:
            flits[:] = [flit for flit in flits if id(flit) not in executed_flits]

        return flits

    def _RB_process(self, network: Network):
        """RB模块：Ring Bridge仲裁处理

        使用多对多匹配的Ring Bridge仲裁（全局最优）

        Args:
            network: 网络实例
        """
        # 遍历所有节点作为Ring Bridge
        for pos in range(self.config.NUM_NODE):
            # 新架构: Ring Bridge在本节点,键直接使用节点号
            next_pos = pos  # 保留用于兼容性（queue_id等使用）

            # 1. 获取各输入槽位的flit
            station_flits = [
                network.ring_bridge["TL"][pos][0] if network.ring_bridge["TL"][pos] else None,
                network.ring_bridge["TR"][pos][0] if network.ring_bridge["TR"][pos] else None,
                network.inject_queues["TU"][pos][0] if pos in network.inject_queues["TU"] and network.inject_queues["TU"][pos] else None,
                network.inject_queues["TD"][pos][0] if pos in network.inject_queues["TD"] and network.inject_queues["TD"][pos] else None,
            ]

            if not any(station_flits):
                continue

            # 2. 构建请求矩阵 (input_slots × output_directions)
            # input_slots: 0=TL, 1=TR, 2=TU, 3=TD
            # output_directions: 0=EQ, 1=TU, 2=TD
            slot_names = ["TL", "TR", "TU", "TD"]
            output_dirs = ["EQ", "TU", "TD"]
            direction_conditions = {"EQ": lambda d, n: d == n, "TU": lambda d, n: d < n, "TD": lambda d, n: d > n}

            request_matrix = []
            for slot_idx, flit in enumerate(station_flits):
                row = []
                for out_dir in output_dirs:
                    # 检查是否可以从这个slot转发到这个输出方向
                    can_forward = self._check_rb_forward_conditions(network, flit, pos, next_pos, out_dir, direction_conditions[out_dir])
                    row.append(can_forward)
                request_matrix.append(row)

            # 3. 执行匹配
            if not any(any(row) for row in request_matrix):
                continue

            queue_id = f"RB_pos{pos}_{next_pos}"
            matches = self.rb_arbiter.match(request_matrix, queue_id=queue_id)

            # 4. 根据匹配结果处理转发
            for slot_idx, out_dir_idx in matches:
                out_dir = output_dirs[out_dir_idx]
                flit = station_flits[slot_idx]

                if flit:
                    # 新架构: ring_bridge_pre键直接使用节点号
                    network.ring_bridge_pre[out_dir][pos] = flit
                    station_flits[slot_idx] = None  # 标记为已使用
                    self._update_ring_bridge(network, pos, out_dir, slot_idx)

    def _EQ_process(self, network: Network, flit_type: str):
        """EQ模块：Eject Queue仲裁处理

        处理eject的仲裁逻辑，根据flit类型处理不同的eject队列

        Args:
            network: 网络实例
            flit_type: flit类型 ("req" / "rsp" / "data")
        """
        # 遍历所有节点处理eject_queues和ring_bridge
        for node_id in range(self.config.NUM_NODE):
            # 构造eject_flits
            eject_flits = (
                [network.eject_queues[fifo_pos][node_id][0] if network.eject_queues[fifo_pos][node_id] else None for fifo_pos in ["TU", "TD"]]
                + [network.inject_queues[fifo_pos][node_id][0] if network.inject_queues[fifo_pos][node_id] else None for fifo_pos in ["EQ"]]
                + [network.ring_bridge["EQ"][node_id][0] if network.ring_bridge["EQ"][node_id] else None]
            )
            if not any(eject_flits):
                continue
            self._move_to_eject_queues_pre(network, eject_flits, node_id)

    def _CP_process(self, network: Network, flits, flit_type: str):
        """CP模块：CrossPoint处理

        处理CrossPoint的上环逻辑

        Args:
            network: 网络实例
            flits: 当前网络中的flit列表
            flit_type: flit类型 ("req" / "rsp" / "data")

        Returns:
            list: 更新后的flits列表
        """
        # CrossPoint注入（TL/TR/TU/TD四个方向）
        for direction in ["TL", "TR", "TU", "TD"]:
            # 获取对应的队列（数据结构都是dict[node_pos] -> deque）
            if direction in ["TL", "TR"]:
                queues = network.inject_queues[direction]
            else:  # TU/TD
                queues = network.ring_bridge[direction]

            num, injected_flits = self.process_inject_queues(network, queues, direction)

            # 横向注入需要更新统计
            if direction in ["TL", "TR"] and num > 0:
                if flit_type == "req":
                    self.req_num += num
                elif flit_type == "rsp":
                    self.rsp_num += num
                elif flit_type == "data":
                    self.flit_num += num

            # 添加注入的flit到列表
            for flit in injected_flits:
                if flit not in flits:
                    flits.append(flit)

        # 4. 更新ITag和CrossPoint状态
        network.update_excess_ITag()
        network.update_cross_point()

        return flits

    def _network_cycle_process(self, network: Network, flits, flit_type: str):
        """网络周期处理：协调各模块完成一个完整的网络周期

        处理顺序：IQ注入 -> Link传输 -> RB仲裁 -> EQ下环 -> CP处理

        Args:
            network: 网络实例
            flits: 当前网络中的flit列表
            flit_type: flit类型 ("req" / "rsp" / "data")

        Returns:
            list: 更新后的flits列表
        """
        # 1. IQ模块：IQ仲裁
        self._IQ_process(network, flit_type)

        # 2. Link模块：Link传输
        flits = self._Link_process(network, flits)

        # 3. RB模块：Ring Bridge仲裁
        self._RB_process(network)

        # 4. EQ模块：Eject Queue仲裁
        self._EQ_process(network, flit_type)

        # 5. CP模块：CrossPoint处理（包含上环和下环）
        flits = self._CP_process(network, flits, flit_type)

        return flits

    def _check_iq_injection_conditions(self, network, node_id, ip_type, direction, network_type, flit_cache):
        """
        检查是否可以从ip_type注入到direction

        Returns:
            bool: 是否可以注入
        """
        # 检查round_robin队列中是否有这个ip_type
        rr_queue = network.round_robin["IQ"][direction][node_id]
        if ip_type not in rr_queue:
            return False

        # 检查pre槽是否占用
        queue_pre = network.inject_queues_pre[direction]
        if queue_pre[node_id]:
            return False

        # 检查FIFO是否满
        queue = network.inject_queues[direction]
        if direction in ["TR", "TL"]:
            fifo_depth = self.config.IQ_OUT_FIFO_DEPTH_HORIZONTAL
        elif direction in ["TU", "TD"]:
            fifo_depth = self.config.IQ_OUT_FIFO_DEPTH_VERTICAL
        else:  # EQ
            fifo_depth = self.config.IQ_OUT_FIFO_DEPTH_EQ

        if len(queue[node_id]) >= fifo_depth:
            return False

        # 网络特定 ip_type 过滤
        if network_type == "req" and not (ip_type.startswith("sdma") or ip_type.startswith("gdma") or ip_type.startswith("cdma") or ip_type.startswith("d2d_rn")):
            return False
        if network_type == "rsp" and not (ip_type.startswith("ddr") or ip_type.startswith("l2m") or ip_type.startswith("d2d_sn")):
            return False

        # 检查channel‑buffer是否为空
        if not network.IQ_channel_buffer[ip_type][node_id]:
            return False

        flit = network.IQ_channel_buffer[ip_type][node_id][0]

        # 缓存flit供后续使用
        flit_cache[(ip_type, direction)] = flit

        # 检查方向条件
        if not self.IQ_direction_conditions[direction](flit):
            return False

        # 网络特定前置检查
        if network_type == "req":
            if not ip_type.startswith("d2d_rn"):
                max_gap = self.config.GDMA_RW_GAP if ip_type.startswith("gdma") else self.config.SDMA_RW_GAP
                counts = self.dma_rw_counts[ip_type][node_id]
                rd, wr = counts["read"], counts["write"]
                if flit.req_type == "read" and abs(rd + 1 - wr) >= max_gap:
                    return False
                if flit.req_type == "write" and abs(wr + 1 - rd) >= max_gap:
                    return False

        return True

    def move_pre_to_queues_all(self):
        #  所有 IPInterface 的 *_pre → FIFO
        for node_id in range(self.config.NUM_NODE):
            for ip_type in self.config.CH_NAME_LIST:
                self.ip_modules[(ip_type, node_id)].move_pre_to_fifo()

        # 所有网络的 *_pre → FIFO
        for node_id in range(self.config.NUM_NODE):
            self._move_pre_to_queues(self.req_network, node_id)
            self._move_pre_to_queues(self.rsp_network, node_id)
            self._move_pre_to_queues(self.data_network, node_id)

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
                        parts = line.strip().split(",")
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
        req.source_original = req_data[1]
        req.destination_original = req_data[3]
        req.flit_type = "req"
        # 保序信息将在inject_fifo出队时分配（inject_to_l2h_pre）
        req.departure_cycle = req_data[0]
        req.burst_length = req_data[6]
        req.source_type = f"{req_data[2]}_0" if "_" not in req_data[2] else req_data[2]
        req.destination_type = f"{req_data[4]}_0" if "_" not in req_data[4] else req_data[4]
        req.original_source_type = f"{req_data[2]}_0" if "_" not in req_data[2] else req_data[2]
        req.original_destination_type = f"{req_data[4]}_0" if "_" not in req_data[4] else req_data[4]
        req.traffic_id = traffic_id  # 添加traffic_id标记

        req.packet_id = BaseModel.get_next_packet_id()
        req.req_type = "read" if req_data[5] == "R" else "write"
        req.req_attr = "new"
        # req.cmd_entry_cake0_cycle = self.cycle

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

    def flit_trace(self, packet_id):
        """打印指定 packet_id 或 packet_id 列表的调试信息"""
        if self.plot_link_state and self.link_state_vis.should_stop:
            return
        # 统一处理 packet_id（兼容单个值或列表）
        packet_ids = [packet_id] if isinstance(packet_id, (int, str)) else packet_id

        for pid in packet_ids:
            self._debug_print(self.req_network, "req", pid)
            self._debug_print(self.rsp_network, "rsp", pid)
            self._debug_print(self.data_network, "flit", pid)

    def _should_skip_waiting_flit(self, flit) -> bool:
        """判断flit是否在等待状态，不需要打印"""
        if hasattr(flit, "flit_position"):
            # IP_inject 状态算等待状态
            if flit.flit_position == "IP_inject":
                return True
            # L2H状态且还未到departure时间 = 等待状态
            if flit.flit_position == "L2H" and hasattr(flit, "departure_cycle") and flit.departure_cycle > self.cycle:
                return True
            # IP_eject状态且位置没有变化，也算等待状态
            if flit.flit_position == "IP_eject":
                # 使用外部字典跟踪flit的稳定周期（避免修改Flit类的__slots__）
                if not hasattr(self, "_flit_stable_cycles"):
                    self._flit_stable_cycles = {}

                flit_key = f"{flit.packet_id}_{flit.flit_id}"
                if flit_key in self._flit_stable_cycles:
                    if self.cycle - self._flit_stable_cycles[flit_key] > 2:  # 在IP_eject超过2个周期就跳过
                        return True
                else:
                    self._flit_stable_cycles[flit_key] = self.cycle
        return False

    def _debug_print(self, net, net_type, packet_id):
        # 取出所有 flit
        flits = net.send_flits.get(packet_id)
        if not flits:
            return

        # 如果这个 packet_id 已经标记完成，直接跳过
        packet_done_key = f"{packet_id}_{net_type}"
        if self._done_flags.get(packet_done_key, False):
            return

        # 检查是否有活跃的flit（非等待状态的flit）
        has_active_flit = any(not self._should_skip_waiting_flit(flit) for flit in flits)

        # 对于单 flit 的 negative rsp，到达后不打印也不更新状态
        if net_type == "rsp":
            last_flit = flits[-1]
            if last_flit.rsp_type == "negative" and len(flits) == 1 and last_flit.is_finish:
                return

        # 只有当有活跃flit时才打印
        if has_active_flit:
            # —— 到这里，说明需要打印调试信息 ——
            if self.cycle != self._last_printed_cycle:
                print(f"Cycle {self.cycle}:")  # 醒目标记当前 cycle
                self._last_printed_cycle = self.cycle  # 更新记录

            # 收集所有flit并格式化打印
            all_flits = []

            # REQ网络的flit
            req_flits = self.req_network.send_flits.get(packet_id, [])
            for flit in req_flits:
                all_flits.append(f"REQ,{flit}")

            # RSP网络的flit
            rsp_flits = self.rsp_network.send_flits.get(packet_id, [])
            for flit in rsp_flits:
                all_flits.append(f"RSP,{flit}")

            # DATA网络的flit
            data_flits = self.data_network.send_flits.get(packet_id, [])
            for flit in data_flits:
                all_flits.append(f"DATA,{flit}")

            # 打印所有flit，用 | 分隔
            if all_flits:
                print(" | ".join(all_flits) + " |")

        # —— 更新完成标记 ——
        # 检查所有 flit 是否都已到达 IP_eject 状态
        all_at_ip_eject = all(f.flit_position == "IP_eject" for f in flits)

        if net_type == "rsp":
            # 只有最后一个 datasend 到达 IP_eject 时才算完成
            last_flit = flits[-1]
            if last_flit.rsp_type == "datasend" and last_flit.flit_position == "IP_eject":
                self._done_flags[packet_done_key] = True
        else:
            # 其他网络类型，所有 flit 都到达 IP_eject 才算完成
            if all_at_ip_eject:
                self._done_flags[packet_done_key] = True

        # 只有在实际打印了信息时才执行sleep
        if has_active_flit and self.update_interval > 0:
            time.sleep(self.update_interval)

    def _check_rb_forward_conditions(self, network, flit, pos, next_pos, out_dir, cmp_func):
        """
        检查是否可以从slot转发到输出方向

        Args:
            network: 网络实例
            flit: 待转发的flit
            pos: 当前位置
            next_pos: 下一个位置
            out_dir: 输出方向 ("EQ", "TU", "TD")
            cmp_func: 目的地比较函数

        Returns:
            bool: 是否可以转发
        """
        if not flit:
            return False

        # 检查输出FIFO是否满
        # 新架构: ring_bridge键直接使用节点号
        if len(network.ring_bridge[out_dir][pos]) >= self.config.RB_OUT_FIFO_DEPTH:
            return False

        # 基于路径的下一跳判断，避免在XY路由的横向阶段提前下竖向环
        next_hop = self._get_next_hop_for_node(flit, pos)

        if out_dir == "EQ":
            final_dest = flit.destination_original if getattr(flit, "destination_original", -1) != -1 else flit.destination
            return final_dest == pos

        if next_hop is None:
            return False

        # 只允许在确实需要向上/向下移动时进入TU/TD
        diff = next_hop - pos
        if out_dir == "TU":
            return diff < 0 and diff % self.config.NUM_COL == 0
        if out_dir == "TD":
            return diff > 0 and diff % self.config.NUM_COL == 0

        return False

    def _get_next_hop_for_node(self, flit, current_node):
        """
        根据flit的路径与path_index定位当前节点的下一跳。

        Returns:
            int | None: 下一跳节点ID，若不存在或无法确定则返回None。
        """
        path = getattr(flit, "path", None)
        if not path:
            return None

        path_len = len(path)
        if path_len <= 1:
            return None

        path_index = getattr(flit, "path_index", None)
        candidate_idx = None

        if isinstance(path_index, int):
            for offset in (0, -1, 1):
                idx = path_index + offset
                if 0 <= idx < path_len and path[idx] == current_node:
                    candidate_idx = idx
                    break

        if candidate_idx is None:
            # Fallback: 从后向前搜索当前节点
            try:
                reverse_idx = path[::-1].index(current_node)
                candidate_idx = path_len - 1 - reverse_idx
            except ValueError:
                return None

        if candidate_idx + 1 < path_len:
            next_hop = path[candidate_idx + 1]
            if next_hop != current_node:
                return next_hop
        return None

    def _update_ring_bridge(self, network: Network, pos, direction, index):
        """更新transfer stations

        新架构: ring_bridge键直接使用节点号pos，next_pos参数保留用于兼容性
        TU/TD方向由CrossPoint处理，flit作为参数传入，不再从队列pop
        """
        if index == 0:
            flit = network.ring_bridge["TL"][pos].popleft()
            if flit.used_entry_level == "T0":
                network.RB_UE_Counters["TL"][pos]["T0"] -= 1
            elif flit.used_entry_level == "T1":
                network.RB_UE_Counters["TL"][pos]["T1"] -= 1
            elif flit.used_entry_level == "T2":
                network.RB_UE_Counters["TL"][pos]["T2"] -= 1
        elif index == 1:
            flit = network.ring_bridge["TR"][pos].popleft()
            if flit.used_entry_level == "T1":
                network.RB_UE_Counters["TR"][pos]["T1"] -= 1
            elif flit.used_entry_level == "T2":
                network.RB_UE_Counters["TR"][pos]["T2"] -= 1
        elif index == 2:
            flit = network.inject_queues["TU"][pos].popleft()
        elif index == 3:
            flit = network.inject_queues["TD"][pos].popleft()

        # 获取通道类型
        channel_type = getattr(flit, "flit_type", "req")  # 默认为req

        # 更新RB总数据量统计（所有经过的flit，无论ETag等级）
        if direction != "EQ":
            if pos in self.RB_total_flits_per_node:
                self.RB_total_flits_per_node[pos][direction] += 1

            # 更新按通道分类的RB总数据量统计
            if pos in self.RB_total_flits_per_channel.get(channel_type, {}):
                self.RB_total_flits_per_channel[channel_type][pos][direction] += 1

            if flit.ETag_priority == "T1":
                self.RB_ETag_T1_num_stat += 1
                # Update per-node FIFO statistics
                if pos in self.RB_ETag_T1_per_node_fifo:
                    self.RB_ETag_T1_per_node_fifo[pos][direction] += 1

                # Update per-channel statistics
                if pos in self.RB_ETag_T1_per_channel.get(channel_type, {}):
                    self.RB_ETag_T1_per_channel[channel_type][pos][direction] += 1

            elif flit.ETag_priority == "T0":
                self.RB_ETag_T0_num_stat += 1
                # Update per-node FIFO statistics
                if pos in self.RB_ETag_T0_per_node_fifo:
                    self.RB_ETag_T0_per_node_fifo[pos][direction] += 1

                # Update per-channel statistics
                if pos in self.RB_ETag_T0_per_channel.get(channel_type, {}):
                    self.RB_ETag_T0_per_channel[channel_type][pos][direction] += 1

        flit.ETag_priority = "T2"

    def tag_move_all_networks(self):
        self._tag_move(self.req_network)
        self._tag_move(self.rsp_network)
        self._tag_move(self.data_network)

    def _tag_move(self, network: Network):
        # 第一部分：纵向环处理
        for col_start in range(self.config.NUM_COL):
            interval = self.config.NUM_COL  # 新架构: 直接使用NUM_COL
            col_end = col_start + interval * (self.config.NUM_ROW - 1)  # 最后一行节点

            # 保存起始位置的tag（使用垂直自环键）
            v_key_start = (col_start, col_start, "v")
            last_position = network.links_tag[v_key_start][0]

            # 前向传递：从起点到终点
            network.links_tag[v_key_start][0] = network.links_tag[(col_start + interval, col_start)][-1]

            for i in range(1, self.config.NUM_ROW):  # 新架构: 遍历所有行
                current_node = col_start + i * interval
                next_node = col_start + (i - 1) * interval

                for j in range(self.config.SLICE_PER_LINK_VERTICAL - 1, -1, -1):
                    if j == 0 and current_node == col_end:
                        v_key_current = (current_node, current_node, "v")
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[v_key_current][-1]
                    elif j == 0:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node + interval, current_node)][-1]
                    else:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]

            # 终点自环处理（使用垂直自环键）
            v_key_end = (col_end, col_end, "v")
            network.links_tag[v_key_end][-1] = network.links_tag[v_key_end][0]
            network.links_tag[v_key_end][0] = network.links_tag[(col_end - interval, col_end)][-1]

            # 回程传递：从终点回到起点
            # 修复：确保处理所有回程连接
            for i in range(1, self.config.NUM_ROW):  # 新架构: 遍历所有行
                current_node = col_end - i * interval
                next_node = col_end - (i - 1) * interval

                for j in range(self.config.SLICE_PER_LINK_VERTICAL - 1, -1, -1):
                    if j == 0 and current_node == col_start:
                        v_key_current = (current_node, current_node, "v")
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[v_key_current][-1]
                    elif j == 0:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node - interval, current_node)][-1]
                    else:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]

            # 起点自环处理（使用垂直自环键）
            network.links_tag[v_key_start][-1] = last_position

        # 第二部分：横向环处理
        # Skip horizontal tag movement if only one column or links_tag missing
        if self.config.NUM_COL <= 1:
            return
        # 新架构: 遍历所有行 (包括第一行row=0)
        for row_start in range(0, self.config.NUM_NODE, self.config.NUM_COL):
            row_end = row_start + self.config.NUM_COL - 1
            # 使用水平自环键
            h_key_start = (row_start, row_start, "h")
            if h_key_start not in network.links_tag:
                continue
            last_position = network.links_tag[h_key_start][0]
            if (row_start + 1, row_start) in network.links_tag:
                network.links_tag[h_key_start][0] = network.links_tag[(row_start + 1, row_start)][-1]
            else:
                network.links_tag[h_key_start][0] = last_position

            for i in range(1, self.config.NUM_COL):
                current_node, next_node = row_start + i, row_start + i - 1
                for j in range(self.config.SLICE_PER_LINK_HORIZONTAL - 1, -1, -1):
                    if j == 0 and current_node == row_end:
                        h_key_end = (current_node, current_node, "h")
                        if h_key_end in network.links_tag and (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[h_key_end][-1]
                    elif j == 0:
                        if (current_node + 1, current_node) in network.links_tag and (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node + 1, current_node)][-1]
                    else:
                        if (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]

            h_key_end = (row_end, row_end, "h")
            if h_key_end in network.links_tag:
                network.links_tag[h_key_end][-1] = network.links_tag[h_key_end][0]
                if (row_end - 1, row_end) in network.links_tag:
                    network.links_tag[h_key_end][0] = network.links_tag[(row_end - 1, row_end)][-1]
                else:
                    network.links_tag[h_key_end][0] = last_position

            for i in range(1, self.config.NUM_COL):
                current_node, next_node = row_end - i, row_end - i + 1
                for j in range(self.config.SLICE_PER_LINK_HORIZONTAL - 1, -1, -1):
                    if j == 0 and current_node == row_start:
                        h_key_current = (current_node, current_node, "h")
                        if h_key_current in network.links_tag and (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[h_key_current][-1]
                    elif j == 0:
                        if (current_node - 1, current_node) in network.links_tag and (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node - 1, current_node)][-1]
                    else:
                        if (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]

            if h_key_start in network.links_tag:
                network.links_tag[h_key_start][-1] = last_position

    def _move_to_eject_queues_pre(self, network: Network, eject_flits, node_id):
        """
        使用多对多匹配的EQ仲裁（全局最优）

        构建 input_ports × ip_types 的请求矩阵，
        使用匹配算法进行全局优化，确保每个端口只弹出到一个IP类型。
        """
        # 1. 收集所有IP类型
        ip_types_list = list(network.EQ_channel_buffer.keys())
        if not ip_types_list:
            return

        # 2. 构建请求矩阵 (input_ports × ip_types)
        # input_ports: 0=TU, 1=TD, 2=IQ, 3=RB
        port_names = ["TU", "TD", "IQ", "RB"]
        request_matrix = []

        for port_idx, flit in enumerate(eject_flits):
            row = []
            for ip_type in ip_types_list:
                # 检查是否可以从这个端口弹出到这个IP类型
                can_eject = self._check_eq_eject_conditions(network, flit, node_id, port_idx, ip_type)
                row.append(can_eject)
            request_matrix.append(row)

        # 3. 执行匹配
        if not any(any(row) for row in request_matrix):
            return

        queue_id = f"EQ_pos{node_id}"
        matches = self.eq_arbiter.match(request_matrix, queue_id=queue_id)

        # 4. 根据匹配结果处理弹出
        for port_idx, ip_type_idx in matches:
            ip_type = ip_types_list[ip_type_idx]
            flit = eject_flits[port_idx]

            if flit:
                network.EQ_channel_buffer_pre[ip_type][node_id] = flit
                flit.is_arrive = True
                flit.arrival_eject_cycle = self.cycle
                eject_flits[port_idx] = None

                # 从对应的队列中移除flit
                if port_idx == 0:  # TU
                    removed_flit = network.eject_queues["TU"][node_id].popleft()
                    if removed_flit.used_entry_level == "T0":
                        network.EQ_UE_Counters["TU"][node_id]["T0"] -= 1
                    elif removed_flit.used_entry_level == "T1":
                        network.EQ_UE_Counters["TU"][node_id]["T1"] -= 1
                    elif removed_flit.used_entry_level == "T2":
                        network.EQ_UE_Counters["TU"][node_id]["T2"] -= 1
                elif port_idx == 1:  # TD
                    removed_flit = network.eject_queues["TD"][node_id].popleft()
                    if removed_flit.used_entry_level == "T1":
                        network.EQ_UE_Counters["TD"][node_id]["T1"] -= 1
                    elif removed_flit.used_entry_level == "T2":
                        network.EQ_UE_Counters["TD"][node_id]["T2"] -= 1
                elif port_idx == 2:  # IQ
                    removed_flit = network.inject_queues["EQ"][node_id].popleft()
                elif port_idx == 3:  # RB
                    removed_flit = network.ring_bridge["EQ"][node_id].popleft()

                # 获取通道类型
                flit_channel_type = getattr(flit, "flit_type", "req")

                # 更新总数据量统计
                if node_id in self.EQ_total_flits_per_node:
                    if port_idx == 0:  # TU direction
                        self.EQ_total_flits_per_node[node_id]["TU"] += 1
                    elif port_idx == 1:  # TD direction
                        self.EQ_total_flits_per_node[node_id]["TD"] += 1

                # 更新按通道分类的总数据量统计
                if node_id in self.EQ_total_flits_per_channel.get(flit_channel_type, {}):
                    if port_idx == 0:  # TU direction
                        self.EQ_total_flits_per_channel[flit_channel_type][node_id]["TU"] += 1
                    elif port_idx == 1:  # TD direction
                        self.EQ_total_flits_per_channel[flit_channel_type][node_id]["TD"] += 1

                if flit.ETag_priority == "T1":
                    self.EQ_ETag_T1_num_stat += 1
                    # Update per-node FIFO statistics (only for TU and TD directions)
                    if node_id in self.EQ_ETag_T1_per_node_fifo:
                        if port_idx == 0:  # TU direction
                            self.EQ_ETag_T1_per_node_fifo[node_id]["TU"] += 1
                        elif port_idx == 1:  # TD direction
                            self.EQ_ETag_T1_per_node_fifo[node_id]["TD"] += 1

                    # Update per-channel statistics
                    if node_id in self.EQ_ETag_T1_per_channel.get(flit_channel_type, {}):
                        if port_idx == 0:  # TU direction
                            self.EQ_ETag_T1_per_channel[flit_channel_type][node_id]["TU"] += 1
                        elif port_idx == 1:  # TD direction
                            self.EQ_ETag_T1_per_channel[flit_channel_type][node_id]["TD"] += 1

                elif flit.ETag_priority == "T0":
                    self.EQ_ETag_T0_num_stat += 1
                    # Update per-node FIFO statistics (only for TU and TD directions)
                    if node_id in self.EQ_ETag_T0_per_node_fifo:
                        if port_idx == 0:  # TU direction
                            self.EQ_ETag_T0_per_node_fifo[node_id]["TU"] += 1
                        elif port_idx == 1:  # TD direction
                            self.EQ_ETag_T0_per_node_fifo[node_id]["TD"] += 1

                    # Update per-channel statistics
                    if node_id in self.EQ_ETag_T0_per_channel.get(flit_channel_type, {}):
                        if port_idx == 0:  # TU direction
                            self.EQ_ETag_T0_per_channel[flit_channel_type][node_id]["TU"] += 1
                        elif port_idx == 1:  # TD direction
                            self.EQ_ETag_T0_per_channel[flit_channel_type][node_id]["TD"] += 1

                flit.ETag_priority = "T2"

    def _check_eq_eject_conditions(self, network, flit, node_id, port_idx, ip_type):
        """
        检查是否可以从端口弹出到IP类型

        Args:
            network: 网络实例
            flit: 待弹出的flit
            node_id: 节点ID
            port_idx: 端口索引 (0=TU, 1=TD, 2=IQ, 3=RB)
            ip_type: IP类型

        Returns:
            bool: 是否可以弹出
        """
        if flit is None:
            return False

        # 检查目的地类型是否匹配
        if flit.destination_type != ip_type:
            return False

        # 检查EQ channel buffer是否满
        if len(network.EQ_channel_buffer[ip_type][node_id]) >= network.config.EQ_CH_FIFO_DEPTH:
            return False

        return True

    def process_inject_queues(self, network: Network, inject_queues, direction):
        """统一的CrossPoint注入处理（支持TL/TR/TU/TD四个方向）

        Args:
            inject_queues: 对于TL/TR是network.inject_queues[direction]
                          对于TU/TD是network.ring_bridge[direction]
            direction: TL/TR/TU/TD
        """
        flit_num = 0
        flits = []

        # 判断是横向还是纵向
        is_horizontal = direction in ["TL", "TR"]
        cp_type = "horizontal" if is_horizontal else "vertical"
        wait_attr = "wait_cycle_h" if is_horizontal else "wait_cycle_v"
        itag_attr = "itag_h" if is_horizontal else "itag_v"
        threshold = self.config.ITag_TRIGGER_Th_H if is_horizontal else self.config.ITag_TRIGGER_Th_V

        for node_id, queue in inject_queues.items():
            if not queue or not queue[0]:
                continue

            # 1. 检查是否需要生成Buffer_Reach_Th信号
            flit = queue[0]
            if getattr(flit, wait_attr) == threshold:
                network.itag_req_counter[direction][node_id] += 1

            # 2. 获取CrossPoint并调用统一的注入方法
            crosspoint = network.crosspoints[node_id][cp_type]
            injected_flit = crosspoint.process_inject(node_id, queue, direction, self.cycle)

            if injected_flit:
                # 3. 首次上环时分配order_id
                if injected_flit.src_dest_order_id == -1:
                    src_node = injected_flit.source_original if injected_flit.source_original != -1 else injected_flit.source
                    dest_node = injected_flit.destination_original if injected_flit.destination_original != -1 else injected_flit.destination
                    src_type = injected_flit.original_source_type if injected_flit.original_source_type else injected_flit.source_type
                    dest_type = injected_flit.original_destination_type if injected_flit.original_destination_type else injected_flit.destination_type
                    injected_flit.src_dest_order_id = Flit.get_next_order_id(
                        src_node, src_type, dest_node, dest_type,
                        injected_flit.flit_type.upper(), self.config.ORDERING_GRANULARITY
                    )

                # 4. 横向注入需要更新inject_num统计
                if is_horizontal:
                    network.inject_num += 1
                    flit_num += 1

                # 5. 纵向注入需要更新flit状态
                if not is_horizontal:
                    injected_flit.current_position = node_id
                    injected_flit.path_index += 1

                # 6. 设置is_new_on_network（CrossPoint已经注入到link）
                injected_flit.is_new_on_network = False
                flits.append(injected_flit)

                # 7. ITag释放处理（统一逻辑）
                if getattr(injected_flit, wait_attr) >= threshold:
                    network.itag_req_counter[direction][node_id] -= 1
                    excess = network.tagged_counter[direction][node_id] - network.itag_req_counter[direction][node_id]
                    if excess > 0:
                        network.excess_ITag_to_remove[direction][node_id] += excess

            # 8. ITag统计
            if queue and queue[0] and getattr(queue[0], itag_attr, False):
                if is_horizontal:
                    self.ITag_h_num_stat += 1
                else:
                    self.ITag_v_num_stat += 1

        return flit_num, flits

    def process_comprehensive_results(self):
        """处理综合统计结果"""
        if not self.result_save_path:
            return

        self.result_processor.collect_requests_data(self, self.cycle)
        results = self.result_processor.analyze_all_bandwidth()
        self.result_processor.generate_unified_report(results, self.result_save_path)
        self.Total_sum_BW_stat = results["Total_sum_BW"]

        # 额外带宽统计
        read_metrics = results["network_overall"]["read"]
        write_metrics = results["network_overall"]["write"]
        # 非加权 / 加权 带宽
        self.read_unweighted_bw_stat = read_metrics.unweighted_bandwidth
        self.read_weighted_bw_stat = read_metrics.weighted_bandwidth
        self.write_unweighted_bw_stat = write_metrics.unweighted_bandwidth
        self.write_weighted_bw_stat = write_metrics.weighted_bandwidth

        # 延迟统计
        latency_stats = self.result_processor._calculate_latency_stats()

        # FIFO使用率统计
        self.result_processor.generate_fifo_usage_csv(self)
        # CMD 延迟
        self.cmd_read_avg_latency_stat = (latency_stats["cmd"]["read"]["sum"] / latency_stats["cmd"]["read"]["count"]) if latency_stats["cmd"]["read"]["count"] else 0.0
        self.cmd_read_max_latency_stat = latency_stats["cmd"]["read"]["max"]
        self.cmd_write_avg_latency_stat = (latency_stats["cmd"]["write"]["sum"] / latency_stats["cmd"]["write"]["count"]) if latency_stats["cmd"]["write"]["count"] else 0.0
        self.cmd_write_max_latency_stat = latency_stats["cmd"]["write"]["max"]
        # Data 延迟
        self.data_read_avg_latency_stat = (latency_stats["data"]["read"]["sum"] / latency_stats["data"]["read"]["count"]) if latency_stats["data"]["read"]["count"] else 0.0
        self.data_read_max_latency_stat = latency_stats["data"]["read"]["max"]
        self.data_write_avg_latency_stat = (latency_stats["data"]["write"]["sum"] / latency_stats["data"]["write"]["count"]) if latency_stats["data"]["write"]["count"] else 0.0
        self.data_write_max_latency_stat = latency_stats["data"]["write"]["max"]
        # Transaction 延迟
        self.trans_read_avg_latency_stat = (latency_stats["trans"]["read"]["sum"] / latency_stats["trans"]["read"]["count"]) if latency_stats["trans"]["read"]["count"] else 0.0
        self.trans_read_max_latency_stat = latency_stats["trans"]["read"]["max"]
        self.trans_write_avg_latency_stat = (latency_stats["trans"]["write"]["sum"] / latency_stats["trans"]["write"]["count"]) if latency_stats["trans"]["write"]["count"] else 0.0
        self.trans_write_max_latency_stat = latency_stats["trans"]["write"]["max"]

        # Mixed 带宽统计
        mixed_metrics = results["network_overall"]["mixed"]
        self.mixed_unweighted_bw_stat = mixed_metrics.unweighted_bandwidth
        self.mixed_weighted_bw_stat = mixed_metrics.weighted_bandwidth
        # Total average bandwidth stats (unweighted and weighted)
        # 使用result_processor中动态计算的实际IP数量
        actual_num_ip = self.result_processor.actual_num_ip or 1  # 避免除零错误
        self.mixed_avg_unweighted_bw_stat = mixed_metrics.unweighted_bandwidth / actual_num_ip
        self.mixed_avg_weighted_bw_stat = mixed_metrics.weighted_bandwidth / actual_num_ip

        # Mixed 延迟统计
        # CMD 混合
        self.cmd_mixed_avg_latency_stat = (latency_stats["cmd"]["mixed"]["sum"] / latency_stats["cmd"]["mixed"]["count"]) if latency_stats["cmd"]["mixed"]["count"] else 0.0
        self.cmd_mixed_max_latency_stat = latency_stats["cmd"]["mixed"]["max"]
        # Data 混合
        self.data_mixed_avg_latency_stat = (latency_stats["data"]["mixed"]["sum"] / latency_stats["data"]["mixed"]["count"]) if latency_stats["data"]["mixed"]["count"] else 0.0
        self.data_mixed_max_latency_stat = latency_stats["data"]["mixed"]["max"]
        # Trans 混合
        self.trans_mixed_avg_latency_stat = (latency_stats["trans"]["mixed"]["sum"] / latency_stats["trans"]["mixed"]["count"]) if latency_stats["trans"]["mixed"]["count"] else 0.0
        self.trans_mixed_max_latency_stat = latency_stats["trans"]["mixed"]["max"]

        # FIFO使用率热力图生成
        if getattr(self, "fifo_utilization_heatmap", False):
            try:
                from src.core.fifo_heatmap_visualizer import create_fifo_heatmap

                # 计算总周期数
                total_cycles = self.cycle // self.config.NETWORK_FREQUENCY

                # 构造dies字典（单Die情况）
                dies = {0: self}

                # 确定保存路径
                if self.result_save_path:
                    fifo_save_path = f"{self.result_save_path}fifo_utilization_heatmap.html"
                else:
                    fifo_save_path = None

                # 生成FIFO热力图
                fifo_heatmap_path = create_fifo_heatmap(dies=dies, config=self.config, total_cycles=total_cycles, die_layout=None, die_rotations=None, save_path=fifo_save_path)

                if fifo_heatmap_path and self.verbose:
                    print(f"FIFO使用率热力图: {fifo_heatmap_path}")
            except Exception as e:
                if self.verbose:
                    print(f"警告: FIFO使用率热力图生成失败: {e}")

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

    def get_results(self):
        """
        Extract simulation statistics and configuration variables.

        Returns:
            dict: A combined dictionary of configuration variables and statistics.
        """
        # Get all variables from the sim instance

        sim_vars = vars(self)

        # Extract statistics (ending with "_stat") and translate to Chinese
        stat_name_map = {
            "model_type": "模型类型",
            "topo_type": "拓扑类型",
            "file_name": "数据流名称",
            "send_read_flits_num": "发送读flit数",
            "send_write_flits_num": "发送写flit数",
            "R_finish_time": "读完成时间",
            "W_finish_time": "写完成时间",
            "Total_finish_time": "总完成时间",
            "R_tail_latency": "读尾延迟",
            "W_tail_latency": "写尾延迟",
            # CMD延迟
            "cmd_read_avg_latency": "命令延迟_读_平均",
            "cmd_read_max_latency": "命令延迟_读_最大",
            "cmd_write_avg_latency": "命令延迟_写_平均",
            "cmd_write_max_latency": "命令延迟_写_最大",
            "cmd_mixed_avg_latency": "命令延迟_混合_平均",
            "cmd_mixed_max_latency": "命令延迟_混合_最大",
            # Data延迟
            "data_read_avg_latency": "数据延迟_读_平均",
            "data_read_max_latency": "数据延迟_读_最大",
            "data_write_avg_latency": "数据延迟_写_平均",
            "data_write_max_latency": "数据延迟_写_最大",
            "data_mixed_avg_latency": "数据延迟_混合_平均",
            "data_mixed_max_latency": "数据延迟_混合_最大",
            # Transaction延迟
            "trans_read_avg_latency": "事务延迟_读_平均",
            "trans_read_max_latency": "事务延迟_读_最大",
            "trans_write_avg_latency": "事务延迟_写_平均",
            "trans_write_max_latency": "事务延迟_写_最大",
            "trans_mixed_avg_latency": "事务延迟_混合_平均",
            "trans_mixed_max_latency": "事务延迟_混合_最大",
            # Circling统计
            "req_cir_h_num": "请求横向环次数",
            "req_cir_v_num": "请求纵向环次数",
            "rsp_cir_h_num": "响应横向环次数",
            "rsp_cir_v_num": "响应纵向环次数",
            "data_cir_h_num": "数据横向环次数",
            "data_cir_v_num": "数据纵向环次数",
            # Wait Cycle统计
            "req_wait_cycle_h_num": "请求横向等待周期",
            "req_wait_cycle_v_num": "请求纵向等待周期",
            "rsp_wait_cycle_h_num": "响应横向等待周期",
            "rsp_wait_cycle_v_num": "响应纵向等待周期",
            "data_wait_cycle_h_num": "数据横向等待周期",
            "data_wait_cycle_v_num": "数据纵向等待周期",
            # Retry统计
            "read_retry_num": "读重试次数",
            "write_retry_num": "写重试次数",
            # ETag统计
            "EQ_ETag_T1_num": "EQ ETag_T1数量",
            "EQ_ETag_T0_num": "EQ ETag_T0数量",
            "RB_ETag_T1_num": "RB ETag_T1数量",
            "RB_ETag_T0_num": "RB ETag_T0数量",
            # ITag统计
            "ITag_h_num": "ITag横向数量",
            "ITag_v_num": "ITag纵向数量",
            # 带宽统计
            "Total_sum_BW": "总和带宽",
            "read_unweighted_bw": "带宽_读_非加权",
            "read_weighted_bw": "带宽_读_加权",
            "write_unweighted_bw": "带宽_写_非加权",
            "write_weighted_bw": "带宽_写_加权",
            "mixed_unweighted_bw": "带宽_混合_非加权",
            "mixed_weighted_bw": "带宽_混合_加权",
            "mixed_avg_unweighted_bw": "带宽_混合_平均非加权",
            "mixed_avg_weighted_bw": "带宽_混合_平均加权",
            "total_unweighted_bw": "带宽_总_非加权",
            "total_weighted_bw": "带宽_总_加权",
        }

        results = {}
        for key, value in sim_vars.items():
            if key.endswith("_stat"):
                base_key = key.rsplit("_stat", 1)[0]
                chinese_key = stat_name_map.get(base_key, base_key)
                results[chinese_key] = value

        # Define config whitelist (only YAML defined parameters)
        config_whitelist = [
            # Basic parameters
            "TOPO_TYPE",
            "FLIT_SIZE",
            "SLICE_PER_LINK_HORIZONTAL",
            "SLICE_PER_LINK_VERTICAL",
            "BURST",
            "NETWORK_FREQUENCY",
            # Resource configuration
            "RN_RDB_SIZE",
            "RN_WDB_SIZE",
            "SN_DDR_RDB_SIZE",
            "SN_DDR_WDB_SIZE",
            "SN_L2M_RDB_SIZE",
            "SN_L2M_WDB_SIZE",
            "UNIFIED_RW_TRACKER",
            # Latency configuration (using original values)
            "DDR_R_LATENCY_original",
            "DDR_R_LATENCY_VAR_original",
            "DDR_W_LATENCY_original",
            "L2M_R_LATENCY_original",
            "L2M_W_LATENCY_original",
            "SN_TRACKER_RELEASE_LATENCY_original",
            # FIFO depths
            "IQ_CH_FIFO_DEPTH",
            "EQ_CH_FIFO_DEPTH",
            "IQ_OUT_FIFO_DEPTH_HORIZONTAL",
            "IQ_OUT_FIFO_DEPTH_VERTICAL",
            "IQ_OUT_FIFO_DEPTH_EQ",
            "RB_OUT_FIFO_DEPTH",
            "RB_IN_FIFO_DEPTH",
            "EQ_IN_FIFO_DEPTH",
            # ETag configuration
            "TL_Etag_T1_UE_MAX",
            "TL_Etag_T2_UE_MAX",
            "TR_Etag_T2_UE_MAX",
            "TU_Etag_T1_UE_MAX",
            "TU_Etag_T2_UE_MAX",
            "TD_Etag_T2_UE_MAX",
            "ETag_BOTHSIDE_UPGRADE",
            # ITag configuration
            "ITag_TRIGGER_Th_H",
            "ITag_TRIGGER_Th_V",
            "ITag_MAX_NUM_H",
            "ITag_MAX_NUM_V",
            # Feature switches
            "ENABLE_CROSSPOINT_CONFLICT_CHECK",
            "ORDERING_PRESERVATION_MODE",
            "CROSSRING_VERSION",
            # Bandwidth limits
            "GDMA_BW_LIMIT",
            "SDMA_BW_LIMIT",
            "CDMA_BW_LIMIT",
            "DDR_BW_LIMIT",
            "L2M_BW_LIMIT",
            # Other configurations
            "GDMA_RW_GAP",
            "SDMA_RW_GAP",
            "IN_ORDER_EJECTION_PAIRS",
            "IN_ORDER_PACKET_CATEGORIES",
            # Tag configuration
            "RB_ONLY_TAG_NUM_HORIZONTAL",
            "RB_ONLY_TAG_NUM_VERTICAL",
            # IP frequency transformation FIFO depths
            "IP_L2H_FIFO_DEPTH",
            "IP_H2L_H_FIFO_DEPTH",
            "IP_H2L_L_FIFO_DEPTH",
        ]

        # Add selected configuration variables
        for key in config_whitelist:
            if hasattr(self.config, key):
                results[key] = getattr(self.config, key)

        # Clear flit and packet IDs (assuming these are class methods)
        Flit.clear_flit_id()
        BaseModel.reset_packet_id()

        # Add result processor analysis for port bandwidth data
        try:
            if hasattr(self, "result_processor") and self.result_processor:
                # Collect request data and analyze bandwidth
                self.result_processor.collect_requests_data(self, self.cycle)
                bandwidth_analysis = self.result_processor.analyze_all_bandwidth()

                # Include port averages in results (both original dict and expanded fields)
                if "port_averages" in bandwidth_analysis:
                    port_avg = bandwidth_analysis["port_averages"]
                    results["port_averages"] = port_avg  # Keep original dict for compatibility

                    # Expand port_averages dictionary into individual fields with Chinese names
                    # Pattern: avg_{port}_{op}_bw -> 平均带宽_{端口}_{操作}
                    port_name_map = {
                        "avg_gdma_read_bw": "平均带宽_GDMA_读",
                        "avg_gdma_write_bw": "平均带宽_GDMA_写",
                        "avg_gdma_bw": "平均带宽_GDMA_混合",
                        "avg_sdma_read_bw": "平均带宽_SDMA_读",
                        "avg_sdma_write_bw": "平均带宽_SDMA_写",
                        "avg_sdma_bw": "平均带宽_SDMA_混合",
                        "avg_cdma_read_bw": "平均带宽_CDMA_读",
                        "avg_cdma_write_bw": "平均带宽_CDMA_写",
                        "avg_cdma_bw": "平均带宽_CDMA_混合",
                        "avg_ddr_read_bw": "平均带宽_DDR_读",
                        "avg_ddr_write_bw": "平均带宽_DDR_写",
                        "avg_ddr_bw": "平均带宽_DDR_混合",
                        "avg_l2m_read_bw": "平均带宽_L2M_读",
                        "avg_l2m_write_bw": "平均带宽_L2M_写",
                        "avg_l2m_bw": "平均带宽_L2M_混合",
                    }
                    for key, value in port_avg.items():
                        chinese_key = port_name_map.get(key, key)
                        results[chinese_key] = value

                # Include other useful bandwidth metrics
                if "Total_sum_BW" in bandwidth_analysis:
                    results["总和带宽"] = bandwidth_analysis["Total_sum_BW"]

                # Include circling eject stats (both original dict and expanded fields)
                if "circling_eject_stats" in bandwidth_analysis:
                    circling_stats = bandwidth_analysis["circling_eject_stats"]
                    results["circling_eject_stats"] = circling_stats  # Keep original dict for compatibility

                    # Expand circling_eject_stats dictionary into individual fields with Chinese names
                    if "horizontal" in circling_stats:
                        results["绕环_横向_总flit数"] = circling_stats["horizontal"]["total_data_flits"]
                        results["绕环_横向_绕环flit数"] = circling_stats["horizontal"]["circling_flits"]
                        results["绕环_横向_比例"] = circling_stats["horizontal"]["circling_ratio"]

                    if "vertical" in circling_stats:
                        results["绕环_纵向_总flit数"] = circling_stats["vertical"]["total_data_flits"]
                        results["绕环_纵向_绕环flit数"] = circling_stats["vertical"]["circling_flits"]
                        results["绕环_纵向_比例"] = circling_stats["vertical"]["circling_ratio"]

                    if "overall" in circling_stats:
                        results["绕环_整体_总flit数"] = circling_stats["overall"]["total_data_flits"]
                        results["绕环_整体_绕环flit数"] = circling_stats["overall"]["circling_flits"]
                        results["绕环_整体_比例"] = circling_stats["overall"]["circling_ratio"]

                # Include ordering blocked stats (both original dict and expanded fields)
                if "ordering_blocked_stats" in bandwidth_analysis:
                    ordering_stats = bandwidth_analysis["ordering_blocked_stats"]
                    results["ordering_blocked_stats"] = ordering_stats  # Keep original dict for compatibility

                    # Expand ordering_blocked_stats dictionary into individual fields
                    if "horizontal" in ordering_stats:
                        results["保序阻止_横向_总flit数"] = ordering_stats["horizontal"]["total_data_flits"]
                        results["保序阻止_横向_被阻止flit数"] = ordering_stats["horizontal"]["ordering_blocked_flits"]
                        results["保序阻止_横向_比例"] = ordering_stats["horizontal"]["ordering_blocked_ratio"]

                    if "vertical" in ordering_stats:
                        results["保序阻止_纵向_总flit数"] = ordering_stats["vertical"]["total_data_flits"]
                        results["保序阻止_纵向_被阻止flit数"] = ordering_stats["vertical"]["ordering_blocked_flits"]
                        results["保序阻止_纵向_比例"] = ordering_stats["vertical"]["ordering_blocked_ratio"]

                    if "overall" in ordering_stats:
                        results["保序阻止_整体_总flit数"] = ordering_stats["overall"]["total_data_flits"]
                        results["保序阻止_整体_被阻止flit数"] = ordering_stats["overall"]["ordering_blocked_flits"]
                        results["保序阻止_整体_比例"] = ordering_stats["overall"]["ordering_blocked_ratio"]

        except Exception as e:
            if hasattr(self, "verbose") and self.verbose:
                print(f"Warning: Could not get port bandwidth analysis: {e}")
            # Set empty port_averages to avoid errors in downstream code
            results["port_averages"] = {}

        return results

    def get_performance_stats(self):
        """Get performance optimization statistics"""
        stats = {
            "simulation_cycles": self.cycle,
            "total_flits_processed": self.trans_flits_num,
            "flit_pool_stats": Flit.get_pool_stats(),
        }

        # Add I/O performance stats if available
        if hasattr(self.traffic_scheduler, "get_io_stats"):
            stats["io_performance"] = self.traffic_scheduler.get_io_stats()

        return stats
