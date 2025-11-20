"""
D2D_Model class for Die-to-Die simulation.
Manages multiple die instances and coordinates cross-die communication.
"""

import copy
import time
import logging
import os
from typing import Dict, List, Optional, Set, Tuple

from src.noc.base_model import BaseModel
from .d2d_traffic_scheduler import D2DTrafficScheduler
from src.analysis.d2d_analyzer import D2DAnalyzer
from src.d2d.components import D2D_RN_Interface, D2D_SN_Interface
from config.config import CrossRingConfig
from src.utils.request_tracker import RequestTracker


class D2D_Model:
    """
    D2D仿真主类 - 管理多Die协调
    每个Die是独立的BaseModel实例，D2D_Model负责：
    1. 创建和管理多个Die实例
    2. 设置Die间的连接关系
    3. 协调多Die的时钟同步
    """

    def __init__(self, config: CrossRingConfig, traffic_config: list = None, **kwargs):
        self.config = config
        self.traffic_config = traffic_config or [[]]
        self.kwargs = kwargs
        self.current_cycle = 0

        # 保存结果路径
        self.result_save_path = kwargs.get("result_save_path", "../Result/")
        self.results_fig_save_path = kwargs.get("results_fig_save_path", "../Result/figures/")

        # 获取Die数量，默认为2
        self.num_dies = getattr(config, "NUM_DIES", 2)

        # 存储各Die实例
        self.dies: Dict[int, BaseModel] = {}

        # 仿真参数
        self.end_time = getattr(config, "END_TIME", 10000)
        self.print_interval = getattr(config, "PRINT_INTERVAL", 1000)

        # 流量图控制参数
        self.enable_flow_graph = False
        self.flow_graph_mode = "total"

        # 结果分析配置（通过setup_result_analysis设置）
        self._result_analysis_config = {
            "flow_graph": False,
            "ip_bandwidth_heatmap": False,
            "plot_rn_bw_fig": False,  # RN带宽曲线图
            "show_fig": False,
            "export_d2d_requests_csv": True,
            "export_ip_bandwidth_csv": True,
            "heatmap_mode": "total",
        }

        # 调试配置（通过setup_debug设置）
        self._debug_config = {
            "enabled": False,
            "trace_packets": [],
            "update_interval": 0.1,
        }

        # 可视化配置（通过setup_visualization设置）
        self._visualization_config = {
            "enabled": False,
            "update_interval": 1,
            "start_cycle": 0,
        }

        # 统计信息
        self.total_cross_die_requests = 0
        self.total_cross_die_responses = 0

        # D2D跨Die事务简化统计
        self.d2d_expected_flits = {i: 0 for i in range(self.num_dies)}  # 每个Die期望接收的跨Die数据包数
        self.d2d_received_flits = {i: 0 for i in range(self.num_dies)}  # 每个Die实际接收的跨Die数据包数
        self.d2d_requests_sent = {i: 0 for i in range(self.num_dies)}  # 每个Die发出的跨Die请求数
        self.d2d_requests_completed = {i: 0 for i in range(self.num_dies)}  # 每个Die完成的跨Die请求数

        # 创建全局RequestTracker（所有Die共享）
        network_freq = getattr(config, "NETWORK_FREQUENCY", 2.0)
        self.request_tracker = RequestTracker(network_frequency=network_freq)
        print(f"[D2D RequestTracker] 已初始化全局请求追踪器，网络频率={network_freq} GHz")

        # 创建D2D专用的traffic调度器（如果提供了traffic_config）
        self.d2d_traffic_scheduler = None
        if traffic_config:
            self.d2d_traffic_scheduler = D2DTrafficScheduler(traffic_config, self.kwargs.get("traffic_file_path", "../traffic/"), config)

        # 创建Die实例
        self._create_die_instances()

        # 设置跨Die连接
        self._setup_cross_die_connections()

        # 初始化D2D路由器
        from .d2d_router import D2DRouter

        self.d2d_router = D2DRouter(self.config)

        # 初始化D2D链路状态可视化器
        self.d2d_link_state_vis = None
        if self.kwargs.get("plot_link_state", 0):
            from src.analysis.D2D_Link_State_Visualizer import D2D_Link_State_Visualizer

            # 获取第一个Die的第一个网络作为初始网络（用于配置信息）
            initial_network = self.dies[0].req_network
            self.d2d_link_state_vis = D2D_Link_State_Visualizer(self.num_dies, initial_network)

    # ==================== Setup Methods ====================

    def setup_traffic_scheduler(self, traffic_file_path: str, traffic_chains: List[List[str]]) -> None:
        """
        配置流量调度器

        Args:
            traffic_file_path: 流量文件路径
            traffic_chains: 流量链配置，每个链包含文件名列表
        """
        # 重新创建D2D traffic调度器
        self.d2d_traffic_scheduler = D2DTrafficScheduler(traffic_chains, traffic_file_path, self.config)

        # 更新kwargs中的traffic配置，以便Die实例使用
        self.kwargs["traffic_file_path"] = traffic_file_path
        self.traffic_config = traffic_chains

        # 分析全局traffic，为每个Die提取IP需求
        die_ip_requirements = self._extract_die_ip_requirements(traffic_file_path, traffic_chains)

        # 为每个Die动态创建IP接口
        for die_id, die_model in self.dies.items():
            # 为Die设置空的traffic scheduler（D2D场景下不使用，但需要避免None）
            from src.noc.traffic_scheduler import TrafficScheduler
            die_model.traffic_scheduler = TrafficScheduler(die_model.config, traffic_file_path)
            die_model.traffic_scheduler.set_verbose(False)

            # 获取该Die需要的IP列表
            required_ips = die_ip_requirements.get(die_id, set())

            # 添加D2D接口需求
            if hasattr(self.config, 'D2D_CONNECTIONS') and self.config.D2D_CONNECTIONS:
                d2d_ips = self._get_d2d_ips_for_die(die_id)
                required_ips.update(d2d_ips)

            # 更新config的CH_NAME_LIST和CHANNEL_SPEC
            from src.utils.traffic_ip_extractor import TrafficIPExtractor
            ip_types = TrafficIPExtractor.get_unique_ip_types(required_ips)
            die_model.config.update_channel_list_from_ips(ip_types)
            die_model.config.infer_channel_spec_from_ips(ip_types)

            # 创建IP接口
            die_model.ip_modules = {}
            for node_id, ip_type in required_ips:
                self._create_ip_interface_for_die(die_model, ip_type, node_id)

            # 为所有IP接口设置request_tracker
            for ip_interface in die_model.ip_modules.values():
                ip_interface.request_tracker = die_model.request_tracker

            # 重新初始化Network buffers（包括round_robin队列）
            die_model.req_network.initialize_buffers()
            die_model.rsp_network.initialize_buffers()
            die_model.data_network.initialize_buffers()

            # 重新关联D2D_Sys和接口
            self._add_d2d_nodes_to_die(die_model, die_id)

    def setup_debug(self, trace_packets: List[int] = None, update_interval: float = 0.0) -> None:
        """
        配置调试模式

        Args:
            trace_packets: 要跟踪的packet ID列表
            update_interval: 每个周期的暂停时间（用于实时观察）
        """
        self._debug_config["enabled"] = trace_packets is not None or update_interval > 0

        if trace_packets:
            self._debug_config["trace_packets"] = trace_packets
            # 向后兼容：同步到kwargs
            self.kwargs["show_d2d_trace_id"] = trace_packets
            self.kwargs["print_d2d_trace"] = 1

        if update_interval is not None:
            self._debug_config["update_interval"] = update_interval
            # 向后兼容：同步到kwargs
            self.kwargs["d2d_trace_sleep"] = update_interval

        if self._debug_config["enabled"]:
            trace_info = f"跟踪packets={trace_packets}" if trace_packets else ""
            interval_info = f"更新间隔={update_interval}s" if update_interval > 0 else ""
            print(f"- 调试模式已启用: {trace_info} {interval_info}".strip())
        else:
            print("- 调试模式已禁用")

    def setup_result_analysis(
        self,
        # 图片生成控制
        flow_graph: bool = False,
        flow_graph_interactive: bool = False,  # 新增：交互式flow图
        ip_bandwidth_heatmap: bool = False,
        fifo_utilization_heatmap: bool = False,
        plot_rn_bw_fig: bool = False,  # 新增：RN带宽曲线图
        show_result_analysis: bool = False,
        # CSV文件导出控制
        export_d2d_requests_csv: bool = True,
        export_ip_bandwidth_csv: bool = True,
        # 通用设置
        save_dir: str = "",
        heatmap_mode: str = "total",
    ) -> None:
        """
        配置结果分析

        图片生成控制:
            flow_graph: 是否生成流量图（PNG/静态）
            flow_graph_interactive: 是否生成交互式流量图（HTML）
            ip_bandwidth_heatmap: 是否生成IP带宽热力图
            fifo_utilization_heatmap: 是否生成FIFO使用率热力图
            plot_rn_bw_fig: 是否生成RN带宽曲线图（IP tracker曲线）
            show_result_analysis: 是否在浏览器中显示图像

        CSV文件导出控制:
            export_d2d_requests_csv: 是否导出跨Die请求记录
            export_ip_bandwidth_csv: 是否导出IP带宽统计

        通用设置:
            save_dir: 结果保存目录
            heatmap_mode: 热力图模式 ("total", "read", "write")
        """
        self._result_analysis_config.update(
            {
                "flow_graph": flow_graph,
                "flow_graph_interactive": flow_graph_interactive,  # 新增
                "ip_bandwidth_heatmap": ip_bandwidth_heatmap,
                "fifo_utilization_heatmap": fifo_utilization_heatmap,
                "plot_rn_bw_fig": plot_rn_bw_fig,  # 新增
                "show_fig": show_result_analysis,
                "export_d2d_requests_csv": export_d2d_requests_csv,
                "export_ip_bandwidth_csv": export_ip_bandwidth_csv,
                "heatmap_mode": heatmap_mode,
            }
        )

        # 向后兼容：同步到实例变量
        self.enable_flow_graph = flow_graph
        self.kwargs["enable_flow_graph"] = flow_graph
        self.fifo_utilization_heatmap = fifo_utilization_heatmap

        # 更新结果保存路径 - 默认使用result_save_path
        if not save_dir:
            save_dir = self.result_save_path
        self.kwargs["results_fig_save_path"] = save_dir

    def setup_visualization(self, enable: bool = True, update_interval: float = 1.0, start_cycle: int = 0) -> None:
        """
        配置实时可视化

        Args:
            enable: 是否启用链路状态可视化
            update_interval: 更新间隔（秒）
            start_cycle: 开始可视化的周期
        """
        self._visualization_config.update(
            {
                "enabled": enable,
                "update_interval": update_interval,
                "start_cycle": start_cycle,
            }
        )

        # 向后兼容：同步到kwargs
        self.kwargs["plot_link_state"] = 1 if enable else 0
        self.kwargs["plot_start_cycle"] = start_cycle

        # 如果启用且尚未创建可视化器，则创建
        if enable and self.d2d_link_state_vis is None and len(self.dies) > 0:
            from src.analysis.D2D_Link_State_Visualizer import D2D_Link_State_Visualizer

            initial_network = self.dies[0].req_network
            self.d2d_link_state_vis = D2D_Link_State_Visualizer(self.num_dies, initial_network)

        if enable:
            print(f"- 实时可视化已启用: 更新间隔={update_interval}s, 开始周期={start_cycle}")
            print("   提示: 可视化窗口将在仿真开始后自动打开")
        else:
            print("- 实时可视化已禁用")

    def run_simulation(
        self,
        max_time: int = None,
        print_interval: int = None,
        results_analysis: bool = True,
        verbose: int = 1,
    ) -> None:
        """
        运行D2D仿真并可选地处理结果

        Args:
            max_time: 最大仿真时间（ns）（如果为None，使用配置中的END_TIME）
            print_interval: 打印进度的间隔时间（ns）（如果为None，使用配置中的PRINT_INTERVAL）
            results_analysis: 仿真完成后是否自动处理结果
            verbose: 详细程度（0=静默，1=正常，2=详细）
        """
        # 获取网络频率（GHz）
        network_frequency = self.config.NETWORK_FREQUENCY
        if not network_frequency:
            raise ValueError("必须在配置中设置NETWORK_FREQUENCY才能运行仿真")

        # 设置仿真参数 - 将ns转换为cycles
        if max_time is not None:
            self.end_time = int(max_time * network_frequency)
        if print_interval is not None:
            self.print_interval = int(print_interval * network_frequency)

        # 初始化仿真
        self.initial()

        # 运行仿真
        # print("\n提示: 按 Ctrl+C 可以随时中断仿真并查看当前结果\n")

        self.run()

        # 处理结果（如果启用）
        if results_analysis:
            try:
                self.process_d2d_comprehensive_results()
            except Exception as e:
                print(f"D2D结果处理失败: {e}")
                import traceback

                traceback.print_exc()

    # ==================== Internal Methods ====================

    def _create_die_instances(self):
        """为每个Die创建独立的BaseModel实例"""
        for die_id in range(self.num_dies):
            die_config = self._create_die_config(die_id)

            # 创建BaseModel实例 - 使用新的简化构造函数
            # 只在第一个Die时启用verbose，避免重复打印
            die_verbose = self.kwargs.get("verbose", 1) if die_id == 0 else 0
            die_model = BaseModel(
                model_type=self.kwargs.get("model_type", "REQ_RSP"),
                config=die_config,
                topo_type=self.kwargs.get("topo_type", "5x4"),
                verbose=die_verbose,
            )

            # 设置Die ID
            die_model.die_id = die_id

            # 共享全局RequestTracker（关键！所有Die使用同一个tracker）
            die_model.request_tracker = self.request_tracker

            # 使用新的setup方法配置结果分析（D2D模式下禁用单个Die的结果保存）
            die_model.setup_result_analysis(
                result_save_path="",  # 禁用单个Die的结果保存，避免生成Die_0/Die_1文件夹
                results_fig_save_path="",  # 禁用单个Die的图片保存
                plot_flow_fig=False,  # 禁用单个Die的流图生成
                plot_RN_BW_fig=True,  # 启用RN带宽数据收集（不保存图片，用于D2D集成）
            )

            # 初始化Die
            die_model.initial()

            # D2D模式下手动设置traffic统计属性，避免AttributeError
            die_model.read_flit = 0
            die_model.write_flit = 0
            die_model.read_req = 0
            die_model.write_req = 0

            # 初始化step变量
            die_model._step_flits, die_model._step_reqs, die_model._step_rsps = [], [], []

            # 初始化d2d_systems字典
            die_model.d2d_systems = {}

            # 设置D2D模型引用
            die_model.d2d_model = self

            # 为网络设置D2D模型引用，方便IP接口访问
            die_model.req_network.d2d_model = self
            die_model.rsp_network.d2d_model = self
            die_model.data_network.d2d_model = self

            # 替换或添加D2D节点
            self._add_d2d_nodes_to_die(die_model, die_id)

            self.dies[die_id] = die_model

    def _create_die_config(self, die_id: int) -> CrossRingConfig:
        """为每个Die创建独立的配置，根据topology加载对应的拓扑配置文件"""

        # 获取Die的拓扑类型
        die_topologies = getattr(self.config, "DIE_TOPOLOGIES", {})
        topology_str = die_topologies.get(die_id)

        if topology_str:
            # 根据拓扑类型加载对应的配置文件
            topo_config_path = f"../../config/topologies/topo_{topology_str}.yaml"
            # print(f"为Die {die_id}加载拓扑配置: {topology_str} (文件: {topo_config_path})")

            try:
                from config.config import CrossRingConfig

                die_config = CrossRingConfig(default_config=topo_config_path)
            except Exception as e:
                raise RuntimeError(f"加载Die {die_id}拓扑配置失败 ({topo_config_path}): {e}")
        else:
            # 没有指定拓扑是配置错误
            raise ValueError(f"Die {die_id}未指定topology配置，每个Die必须指定拓扑类型")

        # 设置Die ID
        die_config.DIE_ID = die_id

        # 获取当前Die的D2D节点位置（分别获取RN和SN位置）
        d2d_rn_positions = getattr(self.config, "D2D_RN_POSITIONS", {})
        d2d_sn_positions = getattr(self.config, "D2D_SN_POSITIONS", {})
        rn_positions = d2d_rn_positions.get(die_id, [])
        sn_positions = d2d_sn_positions.get(die_id, [])

        # 添加D2D节点到CHANNEL_SPEC和CH_NAME_LIST
        # if hasattr(die_config, "CHANNEL_SPEC"):
        # D2D IP接口现在由BaseModel根据D2D_CONNECTIONS动态创建,不再需要手动添加
        # CHANNEL_SPEC和CH_NAME_LIST将在traffic解析后自动推断和更新

        # 设置D2D节点的发送位置列表（分别设置RN和SN）
        die_config.D2D_RN_SEND_POSITION_LIST = rn_positions
        die_config.D2D_SN_SEND_POSITION_LIST = sn_positions

        # 复制D2D配置中的网络基础参数到Die配置中
        if hasattr(self.config, "NETWORK_FREQUENCY"):
            die_config.NETWORK_FREQUENCY = self.config.NETWORK_FREQUENCY
        if hasattr(self.config, "FLIT_SIZE"):
            die_config.FLIT_SIZE = self.config.FLIT_SIZE
        if hasattr(self.config, "BURST"):
            die_config.BURST = self.config.BURST

        # 复制D2D延迟和带宽配置
        d2d_config_keys = [
            "D2D_AR_LATENCY",
            "D2D_R_LATENCY",
            "D2D_AW_LATENCY",
            "D2D_W_LATENCY",
            "D2D_B_LATENCY",
            "D2D_DBID_LATENCY",
            "D2D_MAX_OUTSTANDING",
            "D2D_DATA_BW_LIMIT",
            "D2D_RN_BW_LIMIT",
            "D2D_SN_BW_LIMIT",
            "D2D_RN_R_TRACKER_OSTD",
            "D2D_RN_W_TRACKER_OSTD",
            "D2D_RN_RDB_SIZE",
            "D2D_RN_WDB_SIZE",
            "D2D_SN_R_TRACKER_OSTD",
            "D2D_SN_W_TRACKER_OSTD",
            "D2D_SN_RDB_SIZE",
            "D2D_SN_WDB_SIZE",
        ]
        for key in d2d_config_keys:
            if hasattr(self.config, key):
                setattr(die_config, key, getattr(self.config, key))

        # 规范化 D2D tracker 配置：支持将 "auto" 解析为 databuffer_size // BURST
        def _is_auto(value):
            return isinstance(value, str) and value.lower() == "auto"

        burst = getattr(die_config, "BURST", getattr(self.config, "BURST", 4))

        # D2D_RN trackers
        if hasattr(die_config, "D2D_RN_R_TRACKER_OSTD") and _is_auto(die_config.D2D_RN_R_TRACKER_OSTD):
            rn_rdb = int(getattr(die_config, "D2D_RN_RDB_SIZE", getattr(self.config, "D2D_RN_RDB_SIZE", 0)))
            die_config.D2D_RN_R_TRACKER_OSTD = rn_rdb // burst if burst else rn_rdb

        if hasattr(die_config, "D2D_RN_W_TRACKER_OSTD") and _is_auto(die_config.D2D_RN_W_TRACKER_OSTD):
            rn_wdb = int(getattr(die_config, "D2D_RN_WDB_SIZE", getattr(self.config, "D2D_RN_WDB_SIZE", 0)))
            die_config.D2D_RN_W_TRACKER_OSTD = rn_wdb // burst if burst else rn_wdb

        # D2D_SN trackers
        if hasattr(die_config, "D2D_SN_R_TRACKER_OSTD") and _is_auto(die_config.D2D_SN_R_TRACKER_OSTD):
            sn_rdb = int(getattr(die_config, "D2D_SN_RDB_SIZE", getattr(self.config, "D2D_SN_RDB_SIZE", 0)))
            die_config.D2D_SN_R_TRACKER_OSTD = sn_rdb // burst if burst else sn_rdb

        if hasattr(die_config, "D2D_SN_W_TRACKER_OSTD") and _is_auto(die_config.D2D_SN_W_TRACKER_OSTD):
            sn_wdb = int(getattr(die_config, "D2D_SN_WDB_SIZE", getattr(self.config, "D2D_SN_WDB_SIZE", 0)))
            die_config.D2D_SN_W_TRACKER_OSTD = sn_wdb // burst if burst else sn_wdb

        return die_config

    def _add_d2d_nodes_to_die(self, die_model: BaseModel, die_id: int):
        """向Die添加D2D节点，基于D2D_PAIRS配置创建D2D_Sys实例"""
        config = die_model.config

        # 从D2D_PAIRS获取当前Die的连接配对
        d2d_pairs = getattr(self.config, "D2D_PAIRS", [])
        if not d2d_pairs:
            print(f"警告: 没有找到D2D连接对配置，跳过Die{die_id}的D2D系统创建")
            return

        # 创建D2D_Sys并关联已有的D2D接口
        from src.d2d.components import D2D_Sys

        # 为当前Die的每个连接配对创建D2D_Sys实例
        for pair in d2d_pairs:
            die0_id, die0_node, die1_id, die1_node = pair

            # 检查这个配对是否涉及当前Die
            if die0_id == die_id:
                # 当前Die是源Die，创建到目标Die的D2D_Sys
                node_pos = die0_node
                target_die_id = die1_id
                target_node_pos = die1_node
            elif die1_id == die_id:
                # 当前Die是目标Die，创建到源Die的D2D_Sys
                node_pos = die1_node
                target_die_id = die0_id
                target_node_pos = die0_node
            else:
                # 这个配对不涉及当前Die
                continue

            # 创建D2D_Sys实例，每个实例管理一个点对点连接
            d2d_sys = D2D_Sys(node_pos, die_id, target_die_id, target_node_pos, config)

            # 获取BaseModel已创建的D2D接口
            d2d_rn = die_model.ip_modules.get(("d2d_rn_0", node_pos))
            d2d_sn = die_model.ip_modules.get(("d2d_sn_0", node_pos))

            # 关联D2D_Sys和接口
            if d2d_rn:
                d2d_rn.d2d_sys = d2d_sys
                d2d_sys.rn_interface = d2d_rn
            if d2d_sn:
                d2d_sn.d2d_sys = d2d_sys
                d2d_sys.sn_interface = d2d_sn

            # 存储D2D_Sys，使用组合key来支持一个节点位置有多个连接
            d2d_sys_key = f"{node_pos}_to_{target_die_id}_{target_node_pos}"
            die_model.d2d_systems[d2d_sys_key] = d2d_sys

        # 获取当前Die的所有D2D节点位置（从配对中提取）
        d2d_positions_set = set()
        for pair in d2d_pairs:
            die0_id, die0_node, die1_id, die1_node = pair
            if die0_id == die_id:
                d2d_positions_set.add(die0_node)
            elif die1_id == die_id:
                d2d_positions_set.add(die1_node)

        d2d_positions_list = list(d2d_positions_set)

        # 设置D2D配置（保持向后兼容性）
        config.D2D_RN_POSITIONS = d2d_positions_list
        config.D2D_SN_POSITIONS = d2d_positions_list  # RN和SN通常在同一位置

    def _setup_cross_die_connections(self):
        """建立Die间的连接关系，基于D2D_PAIRS配置"""
        # 从配置中获取D2D连接对
        d2d_pairs = getattr(self.config, "D2D_PAIRS", [])

        if not d2d_pairs:
            print("警告: 没有找到D2D连接对配置")
            return

        # 为每个D2D连接对建立双向连接
        for i, (die0_id, die0_node, die1_id, die1_node) in enumerate(d2d_pairs):
            self._setup_single_pair_connection(die0_id, die0_node, die1_id, die1_node)

    def _setup_single_pair_connection(self, die0_id: int, die0_node: int, die1_id: int, die1_node: int):
        """为单个配对建立双向连接"""

        # 建立Die0 -> Die1的连接
        self._setup_directional_connection(src_die_id=die0_id, src_node=die0_node, dst_die_id=die1_id, dst_node=die1_node)

        # 建立Die1 -> Die0的连接
        self._setup_directional_connection(src_die_id=die1_id, src_node=die1_node, dst_die_id=die0_id, dst_node=die0_node)

    def _setup_directional_connection(self, src_die_id: int, src_node: int, dst_die_id: int, dst_node: int):
        """建立单向连接

        节点编号直接就是网络位置，不需要映射转换
        """

        # 获取IP接口 - 节点编号直接作为网络位置使用
        src_rn_key = ("d2d_rn_0", src_node)
        src_sn_key = ("d2d_sn_0", src_node)
        dst_sn_key = ("d2d_sn_0", dst_node)
        dst_rn_key = ("d2d_rn_0", dst_node)

        src_die = self.dies[src_die_id]
        dst_die = self.dies[dst_die_id]

        # 检查接口是否存在
        if src_rn_key not in src_die.ip_modules or dst_sn_key not in dst_die.ip_modules:
            return

        src_rn = src_die.ip_modules[src_rn_key]
        dst_sn = dst_die.ip_modules[dst_sn_key]
        dst_rn = dst_die.ip_modules.get(dst_rn_key)
        src_sn = src_die.ip_modules.get(src_sn_key)

        # 设置RN接口的目标Die连接（向后兼容）
        if dst_die_id not in src_rn.target_die_interfaces:
            src_rn.target_die_interfaces[dst_die_id] = []
        src_rn.target_die_interfaces[dst_die_id].append(dst_sn)

        # 设置D2D_Sys的目标接口
        d2d_sys_key = f"{src_node}_to_{dst_die_id}_{dst_node}"
        if d2d_sys_key in src_die.d2d_systems:
            d2d_sys = src_die.d2d_systems[d2d_sys_key]
            d2d_sys.target_die_interfaces[dst_die_id] = {"sn": dst_sn, "rn": dst_rn}

        # 设置SN接口的目标RN连接
        if src_sn and dst_rn:
            src_sn.target_die_interfaces[dst_die_id] = dst_rn

    def initial(self):
        """初始化D2D仿真"""
        # Dies已在构造函数中初始化

    def run(self):
        """运行D2D仿真主循环"""
        simulation_start = time.perf_counter()
        tail_time = 6  # 类似BaseModel的tail_time逻辑

        try:
            # 主仿真循环
            while True:
                self.current_cycle += 1

                # 更新所有Die的当前周期和cycle_mod
                for die_id, die_model in self.dies.items():
                    die_model.cycle = self.current_cycle
                    die_model.cycle_mod = self.current_cycle % die_model.config.NETWORK_FREQUENCY

                    # 更新所有IP模块的当前周期
                    for ip_module in die_model.ip_modules.values():
                        ip_module.current_cycle = self.current_cycle

                # 执行各Die的单周期步进
                for die_id, die_model in self.dies.items():
                    # 调用Die的step方法
                    self._step_die(die_model)

                # D2D链路状态可视化更新
                if self.d2d_link_state_vis and self.current_cycle >= self.kwargs.get("plot_start_cycle", 0):
                    # 检查停止信号 - 如果已停止则完全跳过可视化
                    if not self.d2d_link_state_vis.should_stop:
                        # 收集所有Die的所有网络数据
                        all_die_networks = []
                        for die_id in range(self.num_dies):
                            die_model = self.dies[die_id]
                            die_networks = [die_model.req_network, die_model.rsp_network, die_model.data_network]
                            all_die_networks.append(die_networks)

                        # 更新可视化器
                        try:
                            self.d2d_link_state_vis.update(all_die_networks, self.current_cycle)
                        except Exception as e:
                            # 窗口已关闭，设置停止标志
                            self.d2d_link_state_vis.should_stop = True

                        # 暂停阻塞机制 - 与基类保持一致
                        import matplotlib.pyplot as plt

                        while self.d2d_link_state_vis.paused and not self.d2d_link_state_vis.should_stop:
                            plt.pause(0.05)  # 阻塞在这里，不推进仿真

                # D2D Trace调试（如果启用）
                if self.kwargs.get("print_d2d_trace", False):
                    trace_ids = self.kwargs.get("show_d2d_trace_id", None)
                    if trace_ids is not None:
                        self.d2d_trace(trace_ids)
                    else:
                        self.debug_d2d_trace()

                # 打印进度
                if self.current_cycle % self.print_interval == 0 and self.current_cycle > 0:
                    self._print_progress()

                # 检查结束条件
                if self._check_all_completed() or self.current_cycle > self.end_time:
                    if tail_time == 0:
                        if self.kwargs.get("verbose", 1):
                            print("D2D仿真完成！")
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

        # 仿真结束
        simulation_end = time.perf_counter()
        simulation_time = simulation_end - simulation_start

        if self.kwargs.get("verbose", 1):
            print(f"D2D仿真耗时: {simulation_time:.2f} 秒")
            print(f"处理周期数: {self.current_cycle} 周期")
            print(f"仿真性能: {self.current_cycle / simulation_time:.0f} 周期/秒")

            # 打印最终状态（使用print_interval的格式）
            print("\n仿真结束时的最终状态:")
            self._print_progress()

        # 收集每个Die的统计数据（从IPInterface累加到die_model层级）
        for die_id, die_model in self.dies.items():
            die_model.syn_IP_stat()
            # 收集请求数据用于绕环统计
            if hasattr(die_model, "result_processor") and die_model.result_processor:
                die_model.result_processor.collect_requests_data(die_model, self.current_cycle)

        # 创建并缓存 D2D 结果处理器（供后续结果分析使用）
        if not hasattr(self, "_cached_d2d_processor") or not self._cached_d2d_processor:
            d2d_processor = D2DAnalyzer(self.config)
            d2d_processor.simulation_end_cycle = self.current_cycle
            d2d_processor.finish_cycle = self.current_cycle

            # 收集数据
            d2d_processor.collect_cross_die_requests(self.dies)
            d2d_processor.calculate_d2d_ip_bandwidth_data(self.dies)
            d2d_processor.analyze_d2d_results()

            # 缓存处理器
            self._cached_d2d_processor = d2d_processor

    def _step_die(self, die_model: BaseModel):
        """执行单个Die的单周期步进"""
        # 处理D2D Traffic注入
        self._process_d2d_traffic(die_model)

        # 执行D2D_Sys的步进处理
        for pos, d2d_sys in die_model.d2d_systems.items():
            d2d_sys.step(self.current_cycle)

        # 调用BaseModel的step方法来执行标准的单周期步进
        die_model.step()

    def _check_all_completed(self):
        """检查所有Die是否都已完成仿真"""
        # 首先检查D2D traffic调度器是否完成
        if not self.d2d_traffic_scheduler.is_all_completed():
            return False

        # 检查是否有任何请求被发出
        total_requests_sent = sum(self.d2d_requests_sent.values())
        if total_requests_sent == 0:
            return False  # 没有请求被发出，继续等待

        # 使用基于读数据接收和写响应完成的判断逻辑
        burst_length = 4  # 默认burst长度

        # 初始化统计变量（如果还没有初始化）
        if not hasattr(self, "_local_read_requests"):
            self._local_read_requests = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_local_write_requests"):
            self._local_write_requests = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_cross_die_read_requests"):
            self._cross_die_read_requests = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_cross_die_write_requests"):
            self._cross_die_write_requests = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_actual_read_flits_received"):
            self._actual_read_flits_received = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_write_complete_received"):
            self._write_complete_received = {i: 0 for i in range(self.num_dies)}

        # 检查每个Die的事务完成情况
        for die_id in range(self.num_dies):
            # 读事务完成判断：检查读数据接收
            expected_read_flits = (self._local_read_requests[die_id] + self._cross_die_read_requests[die_id]) * burst_length
            actual_read_flits = self._actual_read_flits_received[die_id]

            if actual_read_flits < expected_read_flits:
                return False  # 读数据接收未完成

            # 写事务完成判断：检查write_complete响应接收
            # 注意：只有跨Die写请求才有write_complete响应，Die内写请求通过延迟释放tracker结束
            expected_write_complete = self._cross_die_write_requests[die_id]  # 只统计跨Die写请求
            actual_write_complete = self._write_complete_received[die_id]

            if actual_write_complete < expected_write_complete:
                return False  # 写完成响应未接收完

        # 检查网络是否空闲和没有待处理的事务
        for die_id, die_model in self.dies.items():
            # 检查网络中是否还有传输中的flits
            trans_completed = die_model.trans_flits_num == 0
            # 检查是否有新的写请求待处理
            write_completed = not die_model.new_write_req

            # 检查D2D_Sys的AXI通道是否空闲
            d2d_sys_idle = True
            if hasattr(die_model, "d2d_systems"):
                for pos, d2d_sys in die_model.d2d_systems.items():
                    if d2d_sys.send_flits:  # 如果AXI通道中有flits
                        d2d_sys_idle = False
                        break

            if not (trans_completed and write_completed and d2d_sys_idle):
                return False

        return True

    def _process_d2d_traffic(self, die_model: BaseModel):
        """处理D2D traffic注入"""
        # 获取当前周期的D2D请求（使用缓存避免重复获取）
        if not hasattr(self, "_current_cycle_cache") or self._current_cycle_cache != self.current_cycle:
            # 新周期，重新获取请求
            self._current_cycle_cache = self.current_cycle
            self._cached_pending_requests = self.d2d_traffic_scheduler.get_pending_requests(self.current_cycle)

        pending_requests = self._cached_pending_requests

        # 统计当前Die的请求数量
        die_requests = []
        for req_data in pending_requests:
            # 检查这个请求是否属于当前Die
            src_die = req_data[1]  # src_die字段
            if src_die != die_model.die_id:
                continue  # 不是当前Die的请求，跳过
            die_requests.append(req_data)

        # 处理属于当前Die的请求
        for req_data in die_requests:
            # 处理单个D2D请求
            self._process_single_d2d_request(die_model, req_data)

    def _process_single_d2d_request(self, die_model: BaseModel, req_data):
        """处理单个D2D请求，参考BaseModel._process_single_request"""
        if len(req_data) < 10:
            raise ValueError(f"Invalid D2D request format: {req_data}")

        # 解析D2D请求数据
        (inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length, traffic_id) = req_data

        # 新架构：直接使用节点编号，无需映射
        source_physical = src_node

        # 根据是否跨Die决定路由策略
        if src_die != dst_die:
            # 跨Die：使用D2D路由器选择节点，根据目标IP编号选择d2d_pair
            # 从dst_ip提取IP编号（如"ddr_2"→2，"ddr"→0）
            dst_ip_id = int(dst_ip.split("_")[1]) if "_" in dst_ip else 0
            d2d_node = self.d2d_router.select_d2d_node(src_die, dst_die, dst_ip_id)
            if d2d_node is None:
                raise ValueError(f"D2D路由器返回None，但这是跨Die请求 Die{src_die}->Die{dst_die}")

            # 节点编号直接就是网络位置，不需要RN/SN转换
            # D2D节点既可以作为RN（发起方）也可以作为SN（接收方）
            intermediate_dest = d2d_node
            destination_type = f"d2d_sn_0"  # 跨Die时目标是D2D_SN，包含正确的编号
        else:
            # 新架构：本地路由直接使用目标节点，无需映射
            intermediate_dest = dst_node
            destination_type = dst_ip

        # 创建flit（参考BaseModel._process_single_request）
        from src.utils.flit import Flit

        path = die_model.routes[source_physical][intermediate_dest]
        req = Flit.create_flit(source_physical, intermediate_dest, path)

        # 设置D2D统一属性（新设计）
        req.d2d_origin_die = src_die  # 发起Die ID
        req.d2d_origin_node = source_physical  # 发起节点源映射位置
        req.d2d_origin_type = src_ip  # 发起IP类型

        req.d2d_target_die = dst_die  # 目标Die ID
        # 新架构：目标节点直接使用物理编号，无需映射
        req.d2d_target_node = dst_node
        req.d2d_target_type = dst_ip  # 目标IP类型

        # 设置标准属性（与BaseModel一致）
        # D2D传输不设置_original属性，辅助函数会从d2d_*属性推断
        req.flit_type = "req"
        req.departure_cycle = inject_time
        req.burst_length = burst_length
        req.source_type = f"{src_ip}_0" if "_" not in src_ip else src_ip
        req.destination_type = destination_type  # 已经包含了正确的编号
        # D2D传输不设置original_*属性
        req.req_type = "read" if req_type == "R" else "write"
        req.req_attr = "new"
        req.traffic_id = traffic_id
        req.packet_id = BaseModel.get_next_packet_id()

        # 在RequestTracker中开始追踪请求
        if hasattr(self, "request_tracker") and self.request_tracker:
            is_cross_die = src_die != dst_die
            self.request_tracker.start_request(
                packet_id=req.packet_id,
                source=source_physical,
                destination=dst_node,
                source_type=src_ip,
                dest_type=dst_ip,
                op_type=req.req_type,
                burst_size=burst_length,
                cycle=self.current_cycle,
                is_cross_die=is_cross_die,
                origin_die=src_die,
                target_die=dst_die,
            )

        # 保序信息将在inject_fifo出队时分配（inject_to_l2h_pre）

        # 记录跨Die统计（只对跨Die请求记录）
        if src_die != dst_die:
            self.d2d_requests_sent[src_die] += 1
            self.d2d_expected_flits[src_die] += burst_length

        try:
            # 通过IP接口注入请求
            ip_pos = req.source
            ip_type = req.source_type

            ip_interface = die_model.ip_modules.get((ip_type, ip_pos))
            if ip_interface is None:
                raise ValueError(f"IP module setup error for ({ip_type}, {ip_pos})!")

            # 注入请求到IP接口
            ip_interface.enqueue(req, "req")

        except Exception as e:
            print(f"Error processing D2D request: {e}")
            print(f"Request data: {req_data}")
            raise

    def _get_die_d2d_positions(self, die_id):
        """获取指定Die的D2D节点位置"""
        d2d_die_positions = getattr(self.config, "D2D_DIE_POSITIONS", {})
        return d2d_die_positions.get(die_id, [])

    # 旧的_select_d2d_node_for_target方法已被D2DRouter替代

    def _print_progress(self):
        """打印仿真进度"""
        cycle_time = self.current_cycle // getattr(self.config, "NETWORK_FREQUENCY", 2)

        # 先打印总体时间信息
        print(f"T: {cycle_time}")

        # 然后打印各Die的详细统计信息
        for die_id, die_model in self.dies.items():
            # 获取基本网络统计
            req_cnt = getattr(die_model, "req_count", 0)
            in_req = getattr(die_model, "req_num", 0)
            rsp = getattr(die_model, "rsp_num", 0)
            r_fn = getattr(die_model, "send_read_flits_num_stat", 0)
            w_fn = getattr(die_model, "send_write_flits_num_stat", 0)
            trans_fn = getattr(die_model, "trans_flits_num", 0)
            recv_fn = die_model.data_network.recv_flits_num if hasattr(die_model, "data_network") else 0

            # 使用D2D完成数量统计
            d2d_completed = self.d2d_requests_completed.get(die_id, 0)  # 完成的D2D请求数

            # 获取新的D2D统计信息
            local_read_reqs = getattr(self, "_local_read_requests", {}).get(die_id, 0)
            local_write_reqs = getattr(self, "_local_write_requests", {}).get(die_id, 0)
            cross_read_reqs = getattr(self, "_cross_die_read_requests", {}).get(die_id, 0)
            cross_write_reqs = getattr(self, "_cross_die_write_requests", {}).get(die_id, 0)
            actual_read_flits = getattr(self, "_actual_read_flits_received", {}).get(die_id, 0)
            actual_write_flits = getattr(self, "_actual_write_flits_received", {}).get(die_id, 0)

            # 计算期望接收的flit数量
            # 读期望：本Die本地读请求 + 本Die发起的跨Die读请求的返回数据
            expected_read_flits = (local_read_reqs + cross_read_reqs) * 4

            # 写期望：本Die本地写请求 + 所有其他Die向本Die发起的跨Die写请求
            cross_write_to_this_die = 0
            for other_die_id in range(self.num_dies):
                if other_die_id != die_id:
                    # 统计其他Die向本Die发起的跨Die写请求
                    # 这里需要检查目标Die是否为当前die_id（暂时简化为所有跨Die写）
                    cross_write_to_this_die += getattr(self, "_cross_die_write_requests", {}).get(other_die_id, 0)
            expected_write_flits = (local_write_reqs + cross_write_to_this_die) * 4

            # print(f"  Die{die_id}: Req_cnt: {req_cnt}, In_Req: {in_req}, Rsp: {rsp}, " f"R_fn: {r_fn}, W_fn: {w_fn}, Trans_fn: {trans_fn}, Recv_fn: {recv_fn}, " f"D2D_done: {d2d_completed}")
            print(f"  Die{die_id}:")
            # 获取按local/cross分类的数据统计
            local_read_data = getattr(self, "_local_read_flits_received", {}).get(die_id, 0)
            cross_read_data = getattr(self, "_cross_read_flits_received", {}).get(die_id, 0)
            local_write_data = getattr(self, "_local_write_flits_received", {}).get(die_id, 0)
            cross_write_data = getattr(self, "_cross_write_flits_received", {}).get(die_id, 0)

            # 计算完成情况
            burst_length = 4
            local_read_completed = local_read_data // burst_length if burst_length > 0 else 0
            cross_read_completed = cross_read_data // burst_length if burst_length > 0 else 0
            local_write_completed = local_write_data // burst_length if burst_length > 0 else 0
            cross_write_completed = getattr(self, "_write_complete_received", {}).get(die_id, 0)

            # 读请求统计(显示完成数/总数)
            print(f"    Read - Requests: Local={local_read_completed}/{local_read_reqs}, Cross={cross_read_completed}/{cross_read_reqs} | Data: Local={local_read_data}, Cross={cross_read_data}")

            # 写请求统计(显示完成数/总数)
            print(
                f"    Write - Requests: Local={local_write_completed}/{local_write_reqs}, Cross={cross_write_completed}/{cross_write_reqs} | Data: Local={local_write_data}, Cross={cross_write_data}"
            )

    def generate_combined_flow_graph(self, mode="total", save_path=None):
        """
        生成D2D双Die组合流量图

        Args:
            mode: 显示模式，支持 'utilization', 'total', 'ITag_ratio' 等
            save_path: 图片保存路径
        """
        # 收集网络对象
        die_networks = {}
        for die_id, die_model in self.dies.items():
            # 根据mode选择合适的网络
            if mode == "total":
                network = die_model.data_network  # 使用data_network显示总带宽

                # 获取网络统计数据
                stats = network.get_links_utilization_stats()
                active_links = sum(1 for link_stats in stats.values() if link_stats.get("total_flit", 0) > 0)

            else:
                network = die_model.req_network  # 默认使用req_network

            die_networks[die_id] = network

        if len(die_networks) < 2:
            print("D2D组合流量图需要至少2个Die的网络数据")
            return

        # 设置保存路径（如果save_path为None则直接显示，不保存）
        if save_path == "auto":  # 只有明确指定"auto"时才自动生成路径
            import time

            timestamp = int(time.time())
            save_path = f"../Result/d2d_combined_flow_{mode}_{timestamp}.png"

        # 使用已有的结果处理器或创建新的并传递Die数据
        if hasattr(self, "result_processor") and self.result_processor:
            # 使用已有的结果处理器
            d2d_processor = self.result_processor
        else:
            # 创建新的D2D结果处理器并传递Die数据
            d2d_processor = D2DAnalyzer(self.config)
            d2d_processor.simulation_end_cycle = self.current_cycle

            # 检查是否已经有D2D处理器实例（避免重复计算）
            if hasattr(self, "_cached_d2d_processor") and self._cached_d2d_processor:
                d2d_processor = self._cached_d2d_processor
            else:
                # 第一次计算：收集D2D请求数据并计算正确的IP带宽
                d2d_processor.collect_cross_die_requests(self.dies)
                d2d_processor.calculate_d2d_ip_bandwidth_data(self.dies)
                d2d_processor.analyze_d2d_results()
                # 缓存处理器供后续使用
                self._cached_d2d_processor = d2d_processor

            # 传递每个Die的结果处理器数据，但使用D2D计算的正确IP带宽数据
            d2d_processor.die_processors = {}
            for die_id, die_model in self.dies.items():
                if hasattr(die_model, "result_processor") and die_model.result_processor:
                    die_processor = die_model.result_processor
                    # print(f"[信息] 为Die {die_id} 准备IP带宽数据")

                    # 确保die_processor有sim_model引用
                    if not hasattr(die_processor, "sim_model"):
                        die_processor.sim_model = die_model

                    # 确保die_model有topo_type_stat属性
                    if not hasattr(die_model, "topo_type_stat"):
                        die_model.topo_type_stat = self.kwargs.get("topo_type", "5x4")

                    # 使用D2D处理器计算的该Die特定的IP带宽数据
                    if hasattr(d2d_processor, "die_ip_bandwidth_data") and die_id in d2d_processor.die_ip_bandwidth_data:
                        die_processor.ip_bandwidth_data = d2d_processor.die_ip_bandwidth_data[die_id]
                        # print(f"[信息] Die {die_id}: 使用D2D计算的该Die特定IP带宽数据")

                        # 检查IP带宽数据的内容
                        total_values = 0
                        for mode_data in die_processor.ip_bandwidth_data.values():
                            for ip_data in mode_data.values():
                                total_values += (ip_data > 0.001).sum()
                        # print(f"[信息] Die {die_id} IP带宽数据计算完成，非零值数量: {total_values}")
                    else:
                        print(f"[警告] Die {die_id} 无法获取D2D计算的IP带宽数据")

                    d2d_processor.die_processors[die_id] = die_processor

        try:
            # 调用D2D专用的流量图绘制方法，传入die模型以支持跨Die带宽绘制
            saved_path = d2d_processor.draw_d2d_flow_graph(dies=self.dies, config=self.config, mode=mode, save_path=save_path)
            return saved_path

        except Exception as e:
            print(f"生成D2D组合流量图失败: {e}")
            import traceback

            traceback.print_exc()
            return None

    def run_with_flow_visualization(self, enable_flow_graph=True, flow_mode="total"):
        """
        运行D2D仿真并生成流量可视化

        Args:
            enable_flow_graph: 是否启用流量图生成
            flow_mode: 流量图显示模式
        """
        # 先运行正常的仿真
        self.run()

        # 仿真完成后进行综合结果处理
        combined_flow_path = None
        if enable_flow_graph:
            combined_flow_path = self.generate_combined_flow_graph(mode=flow_mode, save_path="auto")

        # 处理D2D综合结果分析
        self.process_d2d_comprehensive_results(combined_flow_path=combined_flow_path)

    def process_d2d_comprehensive_results(self, combined_flow_path=None):
        """
        处理D2D综合结果分析，复用现有的结果处理方法

        Args:
            combined_flow_path: generate_combined_flow_graph保存的文件路径（如果有）
        """
        # D2D专用结果处理，并收集保存的文件路径
        saved_files = self._process_d2d_specific_results()

        # 3. 添加组合流量图路径（如果有）
        if combined_flow_path:
            saved_files.insert(0, {"type": "D2D组合流量图", "path": combined_flow_path})

        # 4. 简化文件保存提示，与单Die风格保持一致
        if saved_files:
            print()
            for file_info in saved_files:
                if "count" in file_info:
                    print(f"{file_info['type']}CSV: {file_info['path']}")
                else:
                    print(f"{file_info['type']}已保存: {file_info['path']}")

    def _process_d2d_specific_results(self):
        """处理D2D专有的结果分析（跨Die请求记录和带宽统计）"""
        saved_files = []

        try:
            # 使用缓存的D2D处理器（如果存在），避免重复计算
            if hasattr(self, "_cached_d2d_processor") and self._cached_d2d_processor:
                d2d_processor = self._cached_d2d_processor
            else:
                # 如果没有缓存，创建新的处理器
                print("\n[警告] 没有找到缓存的D2D处理器，创建新实例")
                d2d_processor = D2DAnalyzer(self.config)
                d2d_processor.simulation_end_cycle = self.current_cycle
                d2d_processor.finish_cycle = self.current_cycle

                # 直接设置所需属性，避免创建临时sim_model
                d2d_processor.topo_type_stat = "5x4"  # D2D系统使用5x4拓扑
                d2d_processor.results_fig_save_path = self.kwargs.get("results_fig_save_path", "../Result/")
                d2d_processor.file_name = "d2d_system"
                d2d_processor.verbose = self.kwargs.get("verbose", 1)

                # 收集数据（如果没有缓存的话）
                d2d_processor.collect_cross_die_requests(self.dies)
                d2d_processor.calculate_d2d_ip_bandwidth_data(self.dies)
                d2d_processor.analyze_d2d_results()

            # 获取结果保存路径，使用流量文件名
            result_save_path = self.kwargs.get("result_save_path", "../Result/")
            traffic_name = self.d2d_traffic_scheduler.get_save_filename()
            d2d_result_path = os.path.join(result_save_path, f"{self.num_dies}die", traffic_name)

            # 步骤1: 生成带宽分析报告（txt文件）
            report_file = d2d_processor.generate_d2d_bandwidth_report(d2d_result_path, self.dies)
            if report_file:
                saved_files.append({"type": "D2D带宽报告", "path": report_file})

            # 步骤1.5: 生成D2D统计摘要HTML（用于集成报告）
            d2d_summary_html = d2d_processor.generate_d2d_summary_report_html(dies=self.dies)

            # 步骤2: 保存数据文件（根据配置）
            if self._result_analysis_config.get("export_d2d_requests_csv"):
                read_requests = [req for req in d2d_processor.d2d_requests if req.req_type == "read"]
                write_requests = [req for req in d2d_processor.d2d_requests if req.req_type == "write"]

                if read_requests:
                    read_csv_path = os.path.join(d2d_result_path, "d2d_read_requests.csv")
                    saved_files.append({"type": "读请求", "path": read_csv_path, "count": len(read_requests)})

                if write_requests:
                    write_csv_path = os.path.join(d2d_result_path, "d2d_write_requests.csv")
                    saved_files.append({"type": "写请求", "path": write_csv_path, "count": len(write_requests)})

                d2d_processor.save_d2d_requests_csv(d2d_result_path)

            if self._result_analysis_config.get("export_ip_bandwidth_csv"):
                csv_path = os.path.join(d2d_result_path, "ip_bandwidth.csv")
                d2d_processor.save_ip_bandwidth_to_csv(d2d_result_path)

                # 计算保存的记录数
                if hasattr(d2d_processor, "die_ip_bandwidth_data") and d2d_processor.die_ip_bandwidth_data:
                    total_records = 0
                    for die_data in d2d_processor.die_ip_bandwidth_data.values():
                        all_ip_instances = set()
                        for mode_data in die_data.values():
                            all_ip_instances.update(mode_data.keys())
                        total_records += len(all_ip_instances)
                    saved_files.append({"type": "IP带宽", "path": csv_path, "count": total_records})

            # 步骤3: 生成流量图（如果启用）
            if self._result_analysis_config.get("flow_graph"):
                # 设置die_processors以便流量图显示IP信息
                d2d_processor.die_processors = {}
                for die_id, die_model in self.dies.items():
                    if hasattr(die_model, "result_processor") and die_model.result_processor:
                        die_processor = die_model.result_processor

                        # 确保die_processor有sim_model引用
                        if not hasattr(die_processor, "sim_model"):
                            die_processor.sim_model = die_model

                        # 确保die_model有topo_type_stat属性
                        if not hasattr(die_model, "topo_type_stat"):
                            die_model.topo_type_stat = self.kwargs.get("topo_type", "5x4")

                        # 使用D2D处理器计算的该Die特定的IP带宽数据
                        if hasattr(d2d_processor, "die_ip_bandwidth_data") and die_id in d2d_processor.die_ip_bandwidth_data:
                            die_processor.ip_bandwidth_data = d2d_processor.die_ip_bandwidth_data[die_id]

                        d2d_processor.die_processors[die_id] = die_processor

                # 统一使用d2d_result_path保存图片
                save_path = d2d_result_path
                flow_path = d2d_processor.draw_d2d_flow_graph(dies=self.dies, config=self.config, mode=self.flow_graph_mode, save_path=save_path)
                if flow_path:
                    saved_files.append({"type": "D2D流量图(PNG)", "path": flow_path})

            # 收集D2D集成报告的图表
            d2d_charts_to_merge = []

            # 步骤3.5: 生成交互式流量图（如果启用）
            if self._result_analysis_config.get("flow_graph_interactive"):
                # 设置die_processors（与静态版本相同）
                d2d_processor.die_processors = {}
                for die_id, die_model in self.dies.items():
                    if hasattr(die_model, "result_processor") and die_model.result_processor:
                        die_processor = die_model.result_processor

                        if not hasattr(die_processor, "sim_model"):
                            die_processor.sim_model = die_model

                        if not hasattr(die_model, "topo_type_stat"):
                            die_model.topo_type_stat = self.kwargs.get("topo_type", "5x4")

                        if hasattr(d2d_processor, "die_ip_bandwidth_data") and die_id in d2d_processor.die_ip_bandwidth_data:
                            die_processor.ip_bandwidth_data = d2d_processor.die_ip_bandwidth_data[die_id]

                        d2d_processor.die_processors[die_id] = die_processor

                # 获取Figure对象用于集成报告
                flow_fig = d2d_processor.draw_d2d_flow_graph_interactive(dies=self.dies, config=self.config, mode=self.flow_graph_mode, return_fig=True)
                if flow_fig:
                    d2d_charts_to_merge.append(("D2D流量图", flow_fig, None))

            # 步骤4: 生成IP带宽热力图（如果启用）
            if self._result_analysis_config.get("ip_bandwidth_heatmap"):
                # 统一使用d2d_result_path保存图片
                save_path = d2d_result_path
                heatmap_mode = self._result_analysis_config.get("heatmap_mode", "total")

                heatmap_path = d2d_processor.draw_ip_bandwidth_heatmap(dies=self.dies, config=self.config, mode=heatmap_mode, node_size=2500, save_path=save_path)
                if heatmap_path:
                    saved_files.append({"type": f"IP带宽热力图({heatmap_mode})", "path": heatmap_path})

            # 步骤5: 生成FIFO使用率CSV文件（汇总所有Die的数据）
            should_export_fifo_csv = self._result_analysis_config.get("export_fifo_usage_csv", True)
            if should_export_fifo_csv:
                fifo_csv_path = os.path.join(d2d_result_path, "fifo_usage_statistics.csv")
                self._generate_d2d_fifo_usage_csv(fifo_csv_path)
                saved_files.append({"type": "FIFO使用率统计", "path": fifo_csv_path})

            # 步骤6: 生成FIFO使用率热力图（如果启用）- 只集成到HTML报告，不单独保存
            should_plot_fifo = self._result_analysis_config.get("fifo_utilization_heatmap") or getattr(self, "fifo_utilization_heatmap", False)
            if should_plot_fifo:
                from src.analysis.fifo_heatmap_visualizer import create_fifo_heatmap

                # 计算总周期数(使用物理周期数,因为depth_sum在每个物理周期累加)
                total_cycles = self.current_cycle

                # 生成FIFO热力图（只用于集成报告，不保存单独文件）
                fifo_fig, fifo_js = create_fifo_heatmap(
                    dies=self.dies,
                    config=self.dies[0].config,
                    total_cycles=total_cycles,
                    die_layout=getattr(self.config, "die_layout_positions", None),
                    die_rotations=getattr(self.config, "DIE_ROTATIONS", None),
                    save_path=None,  # 不保存独立文件
                    show_fig=False,
                    return_fig_and_js=True,  # 返回Figure和JS
                )
                if fifo_fig:
                    d2d_charts_to_merge.append(("FIFO使用率热力图", fifo_fig, fifo_js))

            # 步骤6.5: 生成RN带宽曲线图（如果启用）
            rn_chart = None
            if self._result_analysis_config.get("plot_rn_bw_fig"):
                from src.analysis.result_visualizers import plot_rn_bandwidth_curves_work_interval

                # 收集所有Die的RN带宽时序数据
                all_die_rn_data = {}
                for die_id, die_model in self.dies.items():
                    if hasattr(die_model, "result_processor") and die_model.result_processor:
                        die_processor = die_model.result_processor
                        if hasattr(die_processor, "rn_bandwidth_time_series") and die_processor.rn_bandwidth_time_series:
                            all_die_rn_data[f"Die{die_id}"] = die_processor.rn_bandwidth_time_series

                # 如果有数据则生成曲线
                if all_die_rn_data:
                    try:
                        rn_fig = plot_rn_bandwidth_curves_work_interval(
                            rn_bandwidth_time_series=all_die_rn_data, show_fig=False, save_path=None, return_fig=True  # 不直接显示  # 不保存独立文件  # 返回Figure对象
                        )
                        if rn_fig:
                            rn_chart = ("IP Tracker曲线", rn_fig, None)
                            saved_files.append({"type": "IP Tracker曲线", "path": "集成到HTML报告中"})
                    except Exception as e:
                        print(f"警告: 生成RN带宽曲线失败: {e}")

            # 步骤6.7: 生成D2D延迟分布图
            d2d_analysis_results = d2d_processor.analyze_d2d_results()
            latency_distribution_figs = d2d_analysis_results.get("latency_distribution_figs", [])

            # 步骤7: 按顺序排列图表并添加统计摘要
            ordered_charts = []
            flow_chart = None
            fifo_chart = None

            for title, fig, custom_js in d2d_charts_to_merge:
                if "流量图" in title:
                    flow_chart = (title, fig, custom_js)
                elif "FIFO" in title:
                    fifo_chart = (title, fig, custom_js)

            # 按顺序添加：流量图 → RN曲线 → FIFO热力图 → 延迟分布图 → 统计摘要
            if flow_chart:
                ordered_charts.append(flow_chart)
            if rn_chart:
                ordered_charts.append(rn_chart)
            if fifo_chart:
                ordered_charts.append(fifo_chart)
            # 添加D2D延迟分布图
            for latency_chart in latency_distribution_figs:
                title, fig = latency_chart
                ordered_charts.append((title, fig, None))
            # 添加统计摘要HTML（最后）
            if d2d_summary_html:
                ordered_charts.append(("D2D结果分析", None, d2d_summary_html))

            # 步骤7.5: 收集所有Die的tracker使用数据
            from src.analysis.data_collectors import TrackerDataCollector

            tracker_collector = TrackerDataCollector()
            # 第一个die调用collect_tracker_data会clear，后续die需要手动添加数据
            first_die = True
            for die_id, die_model in self.dies.items():
                if first_die:
                    tracker_data = tracker_collector.collect_tracker_data(die_model)
                    first_die = False
                else:
                    # 对于后续die，手动收集数据而不clear
                    die_id_from_model = getattr(die_model.config, "DIE_ID", die_id)
                    if die_id_from_model not in tracker_collector.tracker_data:
                        tracker_collector.tracker_data[die_id_from_model] = {}

                    for (ip_type, ip_pos), ip_module in die_model.ip_modules.items():
                        if ip_type not in tracker_collector.tracker_data[die_id_from_model]:
                            tracker_collector.tracker_data[die_id_from_model][ip_type] = {}

                        tracker_usage_data = ip_module.get_tracker_usage_data()
                        if tracker_collector._has_tracker_data(tracker_usage_data):
                            tracker_collector.tracker_data[die_id_from_model][ip_type][ip_pos] = tracker_usage_data

            tracker_json_path = tracker_collector.save_to_json(d2d_result_path, "tracker_data.json")

            # 步骤8: 生成D2D集成可视化报告（合并所有图表）
            if ordered_charts:
                from src.analysis.integrated_visualizer import create_integrated_report

                integrated_save_path = os.path.join(d2d_result_path, "result_analysis.html")
                integrated_path = create_integrated_report(charts_config=ordered_charts, save_path=integrated_save_path, show_result_analysis=self._result_analysis_config.get("show_fig", False))

                # 步骤8.5: 注入tracker功能到HTML
                if integrated_path and tracker_json_path:
                    from src.analysis.tracker_html_injector import inject_tracker_functionality

                    inject_tracker_functionality(integrated_path, tracker_json_path)
                if integrated_path:
                    saved_files.append({"type": "集成可视化报告", "path": integrated_path})

        except Exception as e:
            import traceback

            print(f"警告: D2D专用结果处理失败: {e}")
            print("详细错误信息:")
            traceback.print_exc()

        return saved_files

    def _generate_d2d_fifo_usage_csv(self, output_path: str):
        """生成D2D系统的FIFO使用率CSV文件（包含所有Die的数据和Die ID列）"""
        import csv
        from src.analysis.data_collectors import CircuitStatsCollector

        # 准备CSV数据
        rows = []
        headers = ["Die ID", "网络", "类别", "FIFO类型", "位置", "平均使用率(%)", "最大使用率(%)", "平均深度", "最大深度", "累计flit数", "平均吞吐量(flit/cycle)"]

        # 创建CircuitStatsCollector实例
        circuit_collector = CircuitStatsCollector()

        # 遍历每个Die并收集FIFO统计
        for die_id, die_model in self.dies.items():
            # 直接使用CircuitStatsCollector处理FIFO统计
            try:
                fifo_stats = circuit_collector.process_fifo_usage_statistics(die_model)
            except Exception as e:
                print(f"警告: Die {die_id} FIFO统计处理失败: {e}")
                continue

            # 只处理data网络的数据
            if "data" in fifo_stats:
                net_data = fifo_stats["data"]
                for category, category_data in net_data.items():
                    for fifo_type, fifo_data in category_data.items():
                        for pos, stats in fifo_data.items():
                            row = {
                                "Die ID": die_id,
                                "网络": "data",
                                "类别": category,
                                "FIFO类型": fifo_type,
                                "位置": pos,
                                "平均使用率(%)": f"{stats['avg_utilization']:.2f}",
                                "最大使用率(%)": f"{stats['max_utilization']:.2f}",
                                "平均深度": f"{stats['avg_depth']:.2f}",
                                "最大深度": stats["max_depth"],
                                "累计flit数": stats.get("flit_count", 0),
                                "平均吞吐量(flit/cycle)": f"{stats.get('avg_throughput', 0):.4f}",
                            }
                            rows.append(row)

        # 写入CSV文件（使用UTF-8 with BOM编码，防止Excel打开乱码）
        with open(output_path, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        # if self.kwargs.get("verbose", 1):
        #     print(f"FIFO使用率统计CSV已保存: {output_path}")

    def _collect_d2d_statistics(self):
        """收集D2D专有统计信息"""
        d2d_stats = {"cross_die_requests": 0, "cross_die_responses": 0, "die_stats": {}}

        for die_id, die_model in self.dies.items():
            die_stat = {
                "read_req": die_model.read_req,
                "write_req": die_model.write_req,
                "read_flit": die_model.read_flit,
                "write_flit": die_model.write_flit,
                "total_cycles": die_model.cycle,
            }

            # D2D专有统计
            # 获取当前Die的D2D_RN和D2D_SN位置
            d2d_rn_positions = getattr(die_model.config, "D2D_RN_POSITIONS", [])
            d2d_sn_positions = getattr(die_model.config, "D2D_SN_POSITIONS", [])

            # 对于每个位置，尝试获取统计信息
            d2d_rn_key = None
            d2d_sn_key = None

            if d2d_rn_positions and len(d2d_rn_positions) > die_id:
                d2d_rn_key = ("d2d_rn_0", d2d_rn_positions[die_id])
            if d2d_sn_positions and len(d2d_sn_positions) > die_id:
                d2d_sn_key = ("d2d_sn_0", d2d_sn_positions[die_id])

            if d2d_rn_key and d2d_rn_key in die_model.ip_modules:
                rn_stats = die_model.ip_modules[d2d_rn_key].get_statistics()
                die_stat["d2d_rn_sent"] = rn_stats.get("cross_die_requests_sent", 0)
                die_stat["d2d_rn_received"] = rn_stats.get("cross_die_responses_received", 0)
                d2d_stats["cross_die_requests"] += rn_stats.get("cross_die_requests_sent", 0)

            if d2d_sn_key and d2d_sn_key in die_model.ip_modules:
                sn_stats = die_model.ip_modules[d2d_sn_key].get_statistics()
                die_stat["d2d_sn_received"] = sn_stats.get("cross_die_requests_received", 0)
                die_stat["d2d_sn_forwarded"] = sn_stats.get("cross_die_requests_forwarded", 0)
                die_stat["d2d_sn_responses"] = sn_stats.get("cross_die_responses_sent", 0)
                d2d_stats["cross_die_responses"] += sn_stats.get("cross_die_responses_sent", 0)

            # 收集D2D_Sys的AXI通道统计
            if hasattr(die_model, "d2d_systems"):
                die_stat["axi_systems"] = []
                for pos, d2d_sys in die_model.d2d_systems.items():
                    sys_stats = d2d_sys.get_statistics()
                    die_stat["axi_systems"].append({"position": pos, "target_die": d2d_sys.target_die_id, "stats": sys_stats})

            d2d_stats["die_stats"][die_id] = die_stat

        return d2d_stats

    def _print_d2d_statistics(self, d2d_stats):
        """输出D2D专有统计信息"""
        print(f"\nD2D专有统计:")
        print(f"  跨Die请求总数: {d2d_stats['cross_die_requests']}")
        print(f"  跨Die响应总数: {d2d_stats['cross_die_responses']}")

        for die_id, stat in d2d_stats["die_stats"].items():
            print(f"\nDie {die_id} D2D统计:")
            if "d2d_rn_sent" in stat:
                print(f"  D2D_RN: 发送={stat['d2d_rn_sent']}, 接收={stat['d2d_rn_received']}")
            if "d2d_sn_received" in stat:
                print(f"  D2D_SN: 接收={stat['d2d_sn_received']}, 转发={stat['d2d_sn_forwarded']}, 响应={stat['d2d_sn_responses']}")

            # 打印AXI通道详细统计
            if "axi_systems" in stat:
                for axi_sys in stat["axi_systems"]:
                    print(f"\n  D2D_Sys (Die{die_id} → Die{axi_sys['target_die']}):")
                    sys_stats = axi_sys["stats"]
                    print(f"    总周期: {sys_stats.get('total_cycles', 0)}")
                    print(f"    仲裁统计: RN={sys_stats.get('rn_transmit_count', 0)}, " + f"SN={sys_stats.get('sn_transmit_count', 0)}, " + f"总计={sys_stats.get('total_transmit_count', 0)}")

                    # AXI通道统计
                    axi_stats = sys_stats.get("axi_channel_stats", {})
                    axi_util = sys_stats.get("axi_utilization_pct", {})
                    print(f"    AXI通道统计:")
                    for ch in ["AW", "W", "B", "AR", "R"]:
                        if ch in axi_stats:
                            ch_stat = axi_stats[ch]
                            util = axi_util.get(ch, 0)
                            print(f"      {ch:3s}: Inj={ch_stat['injected']:5d}, " + f"Eje={ch_stat['ejected']:5d}, " + f"Thr={ch_stat['throttled']:5d}, " + f"Util={util:5.1f}%")

                    # 队列状态
                    print(f"    队列: RN={sys_stats.get('rn_queue_length', 0)}, " + f"SN={sys_stats.get('sn_queue_length', 0)}")

    def _generate_d2d_combined_report(self, die_results, d2d_stats):
        """生成D2D组合报告，基于现有的结果分析"""
        if not self.kwargs.get("result_save_path"):
            return

        result_path = self.kwargs.get("result_save_path", "../Result/")

        # 生成D2D组合统计报告
        combined_report_file = os.path.join(result_path, f"d2d_combined_report_{int(time.time())}.txt")

        with open(combined_report_file, "w", encoding="utf-8") as f:
            f.write("D2D双Die组合仿真报告\n")
            f.write("=" * 50 + "\n\n")

            # 配置信息
            f.write("仿真配置:\n")
            f.write(f"  Die数量: {self.num_dies}\n")
            f.write(f"  拓扑类型: 5x4 (每个Die)\n")
            f.write(f"  仿真周期: {self.current_cycle}\n")
            f.write(f"  网络频率: {self.config.NETWORK_FREQUENCY}\n\n")

            # D2D专有统计
            f.write("D2D通信统计:\n")
            f.write(f"  跨Die请求: {d2d_stats['cross_die_requests']}\n")
            f.write(f"  跨Die响应: {d2d_stats['cross_die_responses']}\n\n")

            # 各Die的带宽分析结果摘要
            f.write("各Die带宽分析摘要:\n")
            for die_id, results in die_results.items():
                if results:
                    f.write(f"  Die {die_id}:\n")
                    if "Total_sum_BW" in results:
                        f.write(f"    总带宽: {results['Total_sum_BW']:.2f} GB/s\n")
                    if "network_overall" in results:
                        read_bw = results["network_overall"]["read"].unweighted_bandwidth
                        write_bw = results["network_overall"]["write"].unweighted_bandwidth
                        f.write(f"    读带宽: {read_bw:.2f} GB/s, 写带宽: {write_bw:.2f} GB/s\n")
                    f.write("\n")
                else:
                    f.write(f"  Die {die_id}: 结果处理失败\n\n")

            # 分Die的D2D统计
            for die_id, stat in d2d_stats["die_stats"].items():
                f.write(f"Die {die_id} D2D详细统计:\n")
                f.write(f"  基本流量: 读请求={stat['read_req']}, 写请求={stat['write_req']}\n")
                f.write(f"  基本流量: 读flit={stat['read_flit']}, 写flit={stat['write_flit']}\n")
                if "d2d_rn_sent" in stat:
                    f.write(f"  D2D_RN: 发送={stat['d2d_rn_sent']}, 接收={stat['d2d_rn_received']}\n")
                if "d2d_sn_received" in stat:
                    f.write(f"  D2D_SN: 接收={stat['d2d_sn_received']}, 转发={stat['d2d_sn_forwarded']}, 响应={stat['d2d_sn_responses']}\n")
                f.write("\n")

        print(f"\nD2D组合报告已保存: {combined_report_file}")

    def _print_cycle_header_once(self, has_active_flits=True):
        """只在每个周期打印一次周期标题"""
        if has_active_flits and self.current_cycle != self._d2d_last_printed_cycle:
            print(f"\nCycle {self.current_cycle}:")
            self._d2d_last_printed_cycle = self.current_cycle
            return True
        return False

    def d2d_trace(self, packet_id):
        """
        D2D专用的flit trace功能
        跟踪跨die请求的运动过程，包括在不同die之间的传递

        Args:
            packet_id: 要跟踪的packet ID，可以是单个值或列表
        """
        # 统一处理 packet_id（兼容单个值或列表）
        packet_ids = [packet_id] if isinstance(packet_id, (int, str)) else packet_id

        # 初始化跟踪状态
        if not hasattr(self, "_d2d_last_printed_cycle"):
            self._d2d_last_printed_cycle = -1
        if not hasattr(self, "_d2d_done_flags"):
            self._d2d_done_flags = {}
        if not hasattr(self, "_d2d_flit_stable_cycles"):
            self._d2d_flit_stable_cycles = {}
        if not hasattr(self, "_d2d_write_complete_count"):
            self._d2d_write_complete_count = {}
        if not hasattr(self, "_d2d_write_burst_info"):
            self._d2d_write_burst_info = {}  # {packet_id: {"burst_length": int, "received_count": {die_id: int}}}
        if not hasattr(self, "_d2d_write_data_received"):
            self._d2d_write_data_received = {}  # {packet_id: {die_id: received_count}}

        # 添加请求统计追踪
        if not hasattr(self, "_local_read_requests"):
            self._local_read_requests = {i: 0 for i in range(self.num_dies)}  # 每个Die的本地读请求数
        if not hasattr(self, "_local_write_requests"):
            self._local_write_requests = {i: 0 for i in range(self.num_dies)}  # 每个Die的本地写请求数
        if not hasattr(self, "_cross_die_read_requests"):
            self._cross_die_read_requests = {i: 0 for i in range(self.num_dies)}  # 每个Die发起的跨Die读请求数
        if not hasattr(self, "_cross_die_write_requests"):
            self._cross_die_write_requests = {i: 0 for i in range(self.num_dies)}  # 每个Die发起的跨Die写请求数

        # 实际接收的数据flit统计
        if not hasattr(self, "_actual_read_flits_received"):
            self._actual_read_flits_received = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_actual_write_flits_received"):
            self._actual_write_flits_received = {i: 0 for i in range(self.num_dies)}

        # 按local/cross分类的数据接收统计
        if not hasattr(self, "_local_read_flits_received"):
            self._local_read_flits_received = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_local_write_flits_received"):
            self._local_write_flits_received = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_cross_read_flits_received"):
            self._cross_read_flits_received = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_cross_write_flits_received"):
            self._cross_write_flits_received = {i: 0 for i in range(self.num_dies)}

        # 检查是否有任何活跃的flit
        has_any_active = False
        for pid in packet_ids:
            if not self._d2d_done_flags.get(pid, False):
                # 检查是否有活跃flit
                for die_id, die_model in self.dies.items():
                    die_flits = self._collect_die_flits(die_model, pid, die_id)
                    for flit_info in die_flits:
                        if not self._should_skip_d2d_flit(flit_info["flit"]):
                            has_any_active = True
                            break
                    if has_any_active:
                        break
                if has_any_active:
                    break

        # 使用统一的周期标题打印方法
        self._print_cycle_header_once(has_any_active)

        # 打印每个packet的阶段信息
        for pid in packet_ids:
            self._debug_print_d2d_packet(pid)

        # 如果打印了信息且设置了trace sleep时间，则暂停
        if has_any_active:
            trace_sleep_time = self.kwargs.get("d2d_trace_sleep", 0)
            if trace_sleep_time > 0:
                time.sleep(trace_sleep_time)

    def _debug_print_d2d_packet(self, packet_id):
        """打印指定packet在整个D2D系统中的状态"""
        # 检查是否已经完成跟踪
        if self._d2d_done_flags.get(packet_id, False):
            return

        # 收集所有die中的相关flit
        all_flits_info = []
        has_active_flit = False

        for die_id, die_model in self.dies.items():
            die_flits = self._collect_die_flits(die_model, packet_id, die_id)
            if die_flits:
                all_flits_info.extend(die_flits)
                # 检查是否有活跃的flit
                for flit_info in die_flits:
                    if not self._should_skip_d2d_flit(flit_info["flit"]):
                        has_active_flit = True

        # 只有当有活跃flit时才打印
        if has_active_flit and all_flits_info:
            # 按die分组并打印
            die_groups = {}
            for flit_info in all_flits_info:
                die_id = flit_info["die_id"]
                if die_id not in die_groups:
                    die_groups[die_id] = []
                die_groups[die_id].append(flit_info)

            # 格式化打印每个die的flit状态（只显示阶段信息）
            for die_id in sorted(die_groups.keys()):
                die_flits = die_groups[die_id]
                flit_strs = []
                for flit_info in die_flits:
                    flit = flit_info["flit"]
                    net_type = flit_info["network"]

                    # 构建简化的flit阶段状态字符串
                    status_str = self._format_d2d_flit_stage_only(flit, net_type)
                    flit_strs.append(status_str)

                if flit_strs:
                    print(f"  Packet{packet_id} Die{die_id}: {' | '.join(flit_strs)}")

        # 检查是否完成
        self._check_d2d_packet_completion(packet_id)

    def _collect_die_flits(self, die_model, packet_id, die_id):
        """收集指定die中指定packet的所有flit"""
        flits_info = []

        # 检查三个网络
        networks = [(die_model.req_network, "REQ"), (die_model.rsp_network, "RSP"), (die_model.data_network, "DATA")]

        for network, net_type in networks:
            flits = network.send_flits.get(packet_id, [])
            for flit in flits:
                flits_info.append({"flit": flit, "network": net_type, "die_id": die_id})

        # 检查D2D_Sys的send_flits（更简洁的debug结构）
        for pos, d2d_sys in die_model.d2d_systems.items():
            d2d_flits = d2d_sys.send_flits.get(packet_id, [])
            for flit in d2d_flits:
                # 根据flit的flit_position确定AXI通道类型
                axi_channel = flit.flit_position if hasattr(flit, "flit_position") and flit.flit_position.startswith("AXI_") else "AXI_UNKNOWN"
                flits_info.append({"flit": flit, "network": axi_channel, "die_id": die_id})

        return flits_info

    def _should_skip_d2d_flit(self, flit):
        """判断D2D flit是否在等待状态，不需要打印"""
        if hasattr(flit, "flit_position"):
            # IP_inject 状态算等待状态
            if flit.flit_position == "IP_inject":
                return True
            # L2H状态且还未到departure时间 = 等待状态
            if flit.flit_position == "L2H" and hasattr(flit, "departure_cycle") and flit.departure_cycle > self.current_cycle:
                return True
            # IP_eject状态且位置没有变化，也算等待状态
            if flit.flit_position == "IP_eject":
                flit_key = f"{flit.packet_id}_{flit.flit_id}"
                if flit_key in self._d2d_flit_stable_cycles:
                    if self.current_cycle - self._d2d_flit_stable_cycles[flit_key] > 2:
                        return True
                else:
                    self._d2d_flit_stable_cycles[flit_key] = self.current_cycle
        return False

    def _format_d2d_flit_stage_only(self, flit, net_type):
        """格式化D2D flit的阶段状态字符串，只显示阶段信息"""
        try:
            # 网络类型简写
            if net_type.startswith("AXI_"):
                net = net_type  # AXI_AR, AXI_R等直接使用完整名称
            else:
                net = net_type[:3]  # REQ->REQ, RSP->RSP, DATA->DAT

            # DATA类型和AXI_R/AXI_W通道需要显示flit_id
            if net == "DAT" or net_type == "DATA" or net_type == "AXI_R" or net_type == "AXI_W":
                id_part = f":{getattr(flit, 'flit_id', '?')}"
            else:
                id_part = ""

            # 位置信息（阶段信息）
            if hasattr(flit, "flit_position"):
                if flit.flit_position == "Link" and hasattr(flit, "current_link"):
                    seat_info = f":{flit.current_seat_index}" if hasattr(flit, "current_seat_index") else ""
                    pos_info = f"Link:{flit.current_link[0]}->{flit.current_link[1]}{seat_info}"
                elif flit.flit_position.startswith("AXI_"):
                    # AXI通道传输状态
                    axi_end_cycle = getattr(flit, "axi_end_cycle", self.current_cycle)
                    remaining_cycles = max(0, axi_end_cycle - self.current_cycle)
                    pos_info = f"AXI:{remaining_cycles}cyc"
                else:
                    pos_info = flit.flit_position
            else:
                pos_info = "Unknown"

            # 响应类型优先（如果是RSP网络或AXI响应通道的flit），否则显示请求类型
            if hasattr(flit, "rsp_type") and flit.rsp_type and (net_type == "RSP" or net_type in ["AXI_R", "AXI_B"]):
                # 响应类型映射到清晰的缩写
                rsp_type_map = {
                    "positive": "pos",  # positive response
                    "negative": "neg",  # negative response
                    "datasend": "dat",  # datasend is positive response
                    "write_complete": "ack",  # write acknowledgment
                    "write_ack": "ack",  # write acknowledgment
                    "write_response": "ack",  # write response
                }
                type_info = rsp_type_map.get(flit.rsp_type, flit.rsp_type[:3])
            elif hasattr(flit, "req_type") and flit.req_type:
                type_info = flit.req_type[0].upper()  # read->R, write->W
            else:
                type_info = "?"

            # 状态标记
            status = ""
            if getattr(flit, "is_finish", False):
                status += "F"

            # 添加当前阶段的路由信息
            def get_type_abbreviation(type_str):
                if not type_str:
                    return "??"
                if type_str.startswith("d2d_sn"):
                    return "ds"
                elif type_str.startswith("d2d_rn"):
                    return "dr"
                else:
                    return type_str[0] + type_str[-1]

            # 当前阶段的源和目标信息
            current_src_type = get_type_abbreviation(getattr(flit, "source_type", None))
            current_dst_type = get_type_abbreviation(getattr(flit, "destination_type", None))
            route_info = f"{flit.source}.{current_src_type}->{flit.destination}.{current_dst_type}"

            # 简化输出：只有DATA类型显示flit_id，添加路由信息
            return f"{net}{id_part}:{route_info}:{pos_info},{type_info}" + (f"[{status}]" if status else "")

        except Exception as e:
            # 出错时返回基本信息
            return f"{net_type}:ERROR({str(e)[:20]})"

    def _get_flit_current_die(self, flit, network_die_id):
        """
        确定flit当前所在的Die

        Args:
            flit: 要检查的flit
            network_die_id: flit所在网络的Die ID

        Returns:
            int: 当前Die ID
        """
        # 如果flit在AXI通道中，需要特殊处理
        if hasattr(flit, "flit_position") and flit.flit_position.startswith("AXI_"):
            # 在AXI传输中，考虑传输方向
            if hasattr(flit, "axi_target_die_id"):
                return flit.axi_target_die_id  # 正在传输到目标Die
            else:
                return network_die_id  # 默认返回当前网络所在Die
        else:
            # 在普通网络中，返回网络所在的Die
            return network_die_id

    def _check_d2d_packet_completion(self, packet_id):
        """检查D2D packet是否完成传输

        对于跨Die读请求，完成条件是：
        1. 请求阶段：原始请求从源到目标Die的DDR
        2. 响应阶段：数据从目标Die的DDR返回到源Die的源节点

        只有当整个往返流程完成才算真正完成
        """
        # 首先检查是否有任何flit存在
        has_any_flit = False
        all_completed = True
        has_cross_die_flit = False
        has_response_returned = False  # 是否有响应返回到源

        for die_id, die_model in self.dies.items():
            # 检查所有网络中的flit
            for network in [die_model.req_network, die_model.rsp_network, die_model.data_network]:
                # 检查send_flits中的flit
                flits = network.send_flits.get(packet_id, [])
                if flits:  # 如果有flit存在
                    has_any_flit = True

                # 处理send_flits中的flit
                if flits:
                    for flit in flits:
                        # 检查是否为跨Die请求
                        if hasattr(flit, "d2d_origin_die") and hasattr(flit, "d2d_target_die") and flit.d2d_origin_die != flit.d2d_target_die:
                            has_cross_die_flit = True

                            # 对于跨Die请求，需要检查完整的流程
                            # 读流程：需要回到源Die的源节点
                            # 写流程：需要收到写响应
                            if hasattr(flit, "req_type"):
                                # 请求flit：检查是否回到了源Die的源节点
                                if flit.flit_type == "rsp" or flit.flit_type == "data":
                                    # 响应/数据：需要回到源Die
                                    current_die = self._get_flit_current_die(flit, die_id)
                                    if current_die != flit.d2d_origin_die or not hasattr(flit, "flit_position") or flit.flit_position != "IP_eject":
                                        all_completed = False
                                        break
                                else:
                                    # 请求：需要到达目标Die的目标节点
                                    if not hasattr(flit, "flit_position") or flit.flit_position != "IP_eject":
                                        all_completed = False
                                        break
                            else:
                                # 响应或数据flit
                                if not hasattr(flit, "flit_position") or flit.flit_position != "IP_eject":
                                    all_completed = False
                                    break
                        else:
                            # 本地请求，只需要检查是否到达IP_eject
                            if not hasattr(flit, "flit_position") or flit.flit_position != "IP_eject":
                                all_completed = False
                                break
                if not all_completed:
                    break
            if not all_completed:
                break

        # 对于跨Die请求，还需要检查D2D_Sys的队列状态
        if has_cross_die_flit and all_completed:
            for die_id, die_model in self.dies.items():
                for pos, d2d_sys in die_model.d2d_systems.items():
                    # 检查是否还有该packet的pending传输
                    for item in d2d_sys.rn_pending_queue:
                        if item["flit"].packet_id == packet_id:
                            all_completed = False
                            break
                    if not all_completed:
                        break

                    for item in d2d_sys.sn_pending_queue:
                        if item["flit"].packet_id == packet_id:
                            all_completed = False
                            break
                    if not all_completed:
                        break

                    # 检查AXI通道中是否还有该packet的flit
                    for channel_type, channel in d2d_sys.axi_channels.items():
                        if packet_id in channel["send_flits"]:
                            # AXI通道中还有该packet的flit，未完成
                            all_completed = False
                            break
                    if not all_completed:
                        break

                if not all_completed:
                    break

        # 使用新的完成判断逻辑：基于每个Die接收的数据flit数量是否达到期望值
        completion_check_passed = True

        # 确保统计变量已初始化
        if not hasattr(self, "_actual_read_flits_received"):
            self._actual_read_flits_received = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_actual_write_flits_received"):
            self._actual_write_flits_received = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_local_read_requests"):
            self._local_read_requests = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_local_write_requests"):
            self._local_write_requests = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_cross_die_read_requests"):
            self._cross_die_read_requests = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_cross_die_write_requests"):
            self._cross_die_write_requests = {i: 0 for i in range(self.num_dies)}

        burst_length = 4  # 默认burst长度

        # 检查每个Die的数据接收完成情况
        for die_id in range(self.num_dies):
            # 计算该Die期望接收的读数据flit数量
            # 读期望：本Die本地读请求 + 本Die发起的跨Die读请求的返回数据
            expected_read_flits = (self._local_read_requests[die_id] + self._cross_die_read_requests[die_id]) * burst_length

            # 计算该Die期望接收的写数据flit数量
            # 写期望：本Die本地写请求 + 所有其他Die向本Die发起的跨Die写请求
            cross_write_to_this_die = 0
            for other_die_id in range(self.num_dies):
                if other_die_id != die_id:
                    # 统计其他Die向本Die发起的跨Die写请求
                    # 这里需要检查目标Die是否为当前die_id（暂时简化为所有跨Die写）
                    cross_write_to_this_die += self._cross_die_write_requests[other_die_id]
            expected_write_flits = (self._local_write_requests[die_id] + cross_write_to_this_die) * burst_length

            # 检查实际接收数量是否达到期望
            actual_read = self._actual_read_flits_received[die_id]
            actual_write = self._actual_write_flits_received[die_id]

            if actual_read < expected_read_flits or actual_write < expected_write_flits:
                completion_check_passed = False
                break

        # 只有当存在flit、所有传输阶段都完成、且所有Die的数据接收完成时才标记为完成
        if has_any_flit and all_completed and completion_check_passed:
            self._d2d_done_flags[packet_id] = True
            if has_cross_die_flit:
                print(f"[D2D Trace] Packet {packet_id} cross-die transmission completed!")
            else:
                print(f"[D2D Trace] Packet {packet_id} local transmission completed!")

    def record_write_complete(self, packet_id, die_id):
        """
        记录收到write_complete响应

        Args:
            packet_id: 完成的写事务的packet ID
            die_id: 接收write_complete响应的Die ID（即发起写请求的Die）
        """
        if not hasattr(self, "_d2d_write_complete_count"):
            self._d2d_write_complete_count = {}
        if not hasattr(self, "_write_complete_received"):
            self._write_complete_received = {i: 0 for i in range(self.num_dies)}

        self._d2d_write_complete_count[packet_id] = True
        self._write_complete_received[die_id] += 1

        # print(f"[D2D Write Complete] Packet {packet_id} received write_complete response on Die{die_id} "
        #   f"(total: {self._write_complete_received[die_id]})")

    def record_write_data_received(self, packet_id, die_id, burst_length=None, is_cross_die=False):
        """
        记录写数据接收

        Args:
            packet_id: packet ID
            die_id: 接收数据的die ID
            burst_length: 如果是第一次记录，设置burst长度
            is_cross_die: 是否为跨Die数据
        """
        if not hasattr(self, "_d2d_write_data_received"):
            self._d2d_write_data_received = {}
        if not hasattr(self, "_actual_write_flits_received"):
            self._actual_write_flits_received = {i: 0 for i in range(self.num_dies)}

        if packet_id not in self._d2d_write_data_received:
            self._d2d_write_data_received[packet_id] = {}

        if die_id not in self._d2d_write_data_received[packet_id]:
            self._d2d_write_data_received[packet_id][die_id] = {"received": 0, "burst_length": burst_length or 4}

        self._d2d_write_data_received[packet_id][die_id]["received"] += 1
        self._actual_write_flits_received[die_id] += 1  # 累计该Die接收的所有写数据flit

        # 按local/cross分类统计
        if is_cross_die:
            if not hasattr(self, "_cross_write_flits_received"):
                self._cross_write_flits_received = {i: 0 for i in range(self.num_dies)}
            self._cross_write_flits_received[die_id] += 1
        else:
            if not hasattr(self, "_local_write_flits_received"):
                self._local_write_flits_received = {i: 0 for i in range(self.num_dies)}
            self._local_write_flits_received[die_id] += 1

        if burst_length:
            self._d2d_write_data_received[packet_id][die_id]["burst_length"] = burst_length

        # print(
        # f"[D2D Data Received] Packet {packet_id} Die{die_id}: {self._d2d_write_data_received[packet_id][die_id]['received']}/{self._d2d_write_data_received[packet_id][die_id]['burst_length']} write data received (total: {self._actual_write_flits_received[die_id]})"
        # )

    def record_read_data_received(self, packet_id, die_id, burst_length=None, is_cross_die=False):
        """
        记录读数据接收

        Args:
            packet_id: packet ID
            die_id: 接收数据的die ID
            burst_length: 如果是第一次记录，设置burst长度
            is_cross_die: 是否为跨Die数据
        """
        if not hasattr(self, "_d2d_read_data_received"):
            self._d2d_read_data_received = {}
        if not hasattr(self, "_actual_read_flits_received"):
            self._actual_read_flits_received = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_local_read_flits_received"):
            self._local_read_flits_received = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_cross_read_flits_received"):
            self._cross_read_flits_received = {i: 0 for i in range(self.num_dies)}

        if packet_id not in self._d2d_read_data_received:
            self._d2d_read_data_received[packet_id] = {}

        if die_id not in self._d2d_read_data_received[packet_id]:
            self._d2d_read_data_received[packet_id][die_id] = {"received": 0, "burst_length": burst_length or 4}

        self._d2d_read_data_received[packet_id][die_id]["received"] += 1
        self._actual_read_flits_received[die_id] += 1  # 累计该Die接收的所有读数据flit

        # 按local/cross分类统计
        if is_cross_die:
            if not hasattr(self, "_cross_read_flits_received"):
                self._cross_read_flits_received = {i: 0 for i in range(self.num_dies)}
            self._cross_read_flits_received[die_id] += 1
        else:
            if not hasattr(self, "_local_read_flits_received"):
                self._local_read_flits_received = {i: 0 for i in range(self.num_dies)}
            self._local_read_flits_received[die_id] += 1

        if burst_length:
            self._d2d_read_data_received[packet_id][die_id]["burst_length"] = burst_length

        # print(
        # f"[D2D Data Received] Packet {packet_id} Die{die_id}: {self._d2d_read_data_received[packet_id][die_id]['received']}/{self._d2d_read_data_received[packet_id][die_id]['burst_length']} read data received (total: {self._actual_read_flits_received[die_id]})"
        # )

    def record_request_issued(self, packet_id, die_id, req_type, is_cross_die):
        """
        记录请求发起

        Args:
            packet_id: packet ID
            die_id: 发起请求的die ID
            req_type: 请求类型 ("read" 或 "write")
            is_cross_die: 是否为跨Die请求
        """
        # 确保统计变量已初始化
        if not hasattr(self, "_local_read_requests"):
            self._local_read_requests = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_local_write_requests"):
            self._local_write_requests = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_cross_die_read_requests"):
            self._cross_die_read_requests = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_cross_die_write_requests"):
            self._cross_die_write_requests = {i: 0 for i in range(self.num_dies)}
        if not hasattr(self, "_recorded_requests"):
            self._recorded_requests = set()

        # 防止重复记录同一个packet_id
        if packet_id in self._recorded_requests:
            return

        self._recorded_requests.add(packet_id)

        if is_cross_die:
            if req_type == "read":
                self._cross_die_read_requests[die_id] += 1
                # print(f"[D2D Request] Die{die_id} issued cross-die read request {packet_id} (total: {self._cross_die_read_requests[die_id]})")
            elif req_type == "write":
                self._cross_die_write_requests[die_id] += 1
                # print(f"[D2D Request] Die{die_id} issued cross-die write request {packet_id} (total: {self._cross_die_write_requests[die_id]})")
        else:
            if req_type == "read":
                self._local_read_requests[die_id] += 1
                # print(f"[D2D Request] Die{die_id} issued local read request {packet_id} (total: {self._local_read_requests[die_id]})")
            elif req_type == "write":
                self._local_write_requests[die_id] += 1
                # print(f"[D2D Request] Die{die_id} issued local write request {packet_id} (total: {self._local_write_requests[die_id]})")

    def debug_d2d_trace(self, trace_packet_ids=None):
        """
        D2D调试函数，在每个周期调用以跟踪指定的packet

        Args:
            trace_packet_ids: 要跟踪的packet ID列表，如果为None则跟踪所有活跃的packet
        """
        if trace_packet_ids is None:
            # 自动发现活跃的packet
            active_packets = set()
            for die_model in self.dies.values():
                for network in [die_model.req_network, die_model.rsp_network, die_model.data_network]:
                    active_packets.update(network.send_flits.keys())
                # 检查D2D_Sys的send_flits
                for pos, d2d_sys in die_model.d2d_systems.items():
                    active_packets.update(d2d_sys.send_flits.keys())
            trace_packet_ids = list(active_packets)

        if trace_packet_ids:
            # 检查是否有任何活跃的flit
            has_any_active = False
            for pid in trace_packet_ids:
                if not self._d2d_done_flags.get(pid, False):
                    # 检查是否有活跃flit
                    for die_id, die_model in self.dies.items():
                        die_flits = self._collect_die_flits(die_model, pid, die_id)
                        for flit_info in die_flits:
                            if not self._should_skip_d2d_flit(flit_info["flit"]):
                                has_any_active = True
                                break
                        if has_any_active:
                            break
                    if has_any_active:
                        break

            # 使用统一的周期标题打印方法
            self._print_cycle_header_once(has_any_active)

            # 打印每个packet的阶段信息
            for pid in trace_packet_ids:
                self._debug_print_d2d_packet(pid)

            # 如果打印了信息且设置了trace sleep时间，则暂停
            if has_any_active:
                trace_sleep_time = self.kwargs.get("d2d_trace_sleep", 0)
                if trace_sleep_time > 0:
                    time.sleep(trace_sleep_time)

    # ==================== IP Management Helpers ====================

    def _extract_die_ip_requirements(self, traffic_file_path: str, traffic_chains: List[List[str]]) -> Dict[int, Set[Tuple[int, str]]]:
        """
        从全局traffic文件中提取各Die的IP需求

        Args:
            traffic_file_path: traffic文件基础路径
            traffic_chains: traffic文件链配置

        Returns:
            字典 {die_id: Set[(node_id, ip_type)]}
        """
        import os
        from src.utils.traffic_ip_extractor import TrafficIPExtractor

        die_ip_requirements = {i: set() for i in range(self.num_dies)}

        # 收集所有traffic文件
        traffic_files = []
        for chain in traffic_chains:
            if isinstance(chain, list):
                for file_name in chain:
                    traffic_files.append(os.path.join(traffic_file_path, file_name))
            else:
                traffic_files.append(os.path.join(traffic_file_path, chain))

        # 解析每个traffic文件
        for file_path in traffic_files:
            if not os.path.exists(file_path):
                continue

            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = [p.strip() for p in line.split(',')]

                    # D2D格式: inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst
                    if len(parts) >= 9:
                        try:
                            src_die = int(parts[1])
                            src_node = int(parts[2])
                            src_ip = parts[3].strip()
                            dst_die = int(parts[4])
                            dst_node = int(parts[5])
                            dst_ip = parts[6].strip()

                            # 为源Die添加源IP
                            if src_die < self.num_dies:
                                die_ip_requirements[src_die].add((src_node, src_ip))

                            # 为目标Die添加目标IP
                            if dst_die < self.num_dies:
                                die_ip_requirements[dst_die].add((dst_node, dst_ip))

                        except (ValueError, IndexError):
                            continue

        return die_ip_requirements

    def _get_d2d_ips_for_die(self, die_id: int) -> Set[Tuple[int, str]]:
        """
        获取指定Die需要的D2D接口

        Args:
            die_id: Die ID

        Returns:
            Set[(node_id, ip_type)]
        """
        d2d_ips = set()

        if not hasattr(self.config, 'D2D_CONNECTIONS'):
            return d2d_ips

        # D2D_CONNECTIONS格式: [[src_die, src_node, dst_die, dst_node], ...]
        for connection in self.config.D2D_CONNECTIONS:
            src_die, src_node, dst_die, dst_node = connection

            # 如果该Die参与此连接，添加D2D接口
            if src_die == die_id:
                d2d_ips.add((src_node, "d2d_rn_0"))
                d2d_ips.add((src_node, "d2d_sn_0"))

            if dst_die == die_id:
                d2d_ips.add((dst_node, "d2d_rn_0"))
                d2d_ips.add((dst_node, "d2d_sn_0"))

        return d2d_ips

    def _create_ip_interface_for_die(self, die_model, ip_type: str, node_id: int):
        """
        为指定Die创建单个IP接口

        Args:
            die_model: Die的BaseModel实例
            ip_type: IP类型 (如"gdma_0", "d2d_rn_0")
            node_id: 节点ID
        """
        from src.noc.components.ip_interface import IPInterface

        # 避免重复创建
        if (ip_type, node_id) in die_model.ip_modules:
            return

        # 根据IP类型创建相应的接口
        if ip_type == "d2d_rn_0":
            from src.d2d.components import D2D_RN_Interface

            die_model.ip_modules[(ip_type, node_id)] = D2D_RN_Interface(
                ip_type,
                node_id,
                die_model.config,
                die_model.req_network,
                die_model.rsp_network,
                die_model.data_network,
                die_model.routes,
            )
        elif ip_type == "d2d_sn_0":
            from src.d2d.components import D2D_SN_Interface

            die_model.ip_modules[(ip_type, node_id)] = D2D_SN_Interface(
                ip_type,
                node_id,
                die_model.config,
                die_model.req_network,
                die_model.rsp_network,
                die_model.data_network,
                die_model.routes,
            )
        else:
            # 普通IP接口
            die_model.ip_modules[(ip_type, node_id)] = IPInterface(
                ip_type,
                node_id,
                die_model.config,
                die_model.req_network,
                die_model.rsp_network,
                die_model.data_network,
                die_model.routes,
            )

    def _initialize_network_buffers_for_ips(self, die_model, required_ips: Set[Tuple[int, str]]):
        """
        为新创建的IP接口初始化Network缓冲区

        Args:
            die_model: Die的BaseModel实例
            required_ips: IP需求集合 Set[(node_id, ip_type)]
        """
        from collections import deque, defaultdict

        # 提取所有IP类型
        ip_types = set(ip_type for _, ip_type in required_ips)

        # 为每个network初始化buffer
        for network in [die_model.req_network, die_model.rsp_network, die_model.data_network]:
            # 为每个IP类型创建buffer结构
            for ip_type in ip_types:
                # 如果该IP类型还没有buffer，创建完整的buffer结构
                if ip_type not in network.IQ_channel_buffer:
                    network.IQ_channel_buffer[ip_type] = defaultdict(
                        lambda: deque(maxlen=network.config.IQ_CH_FIFO_DEPTH)
                    )
                    network.IQ_channel_buffer_pre[ip_type] = {}
                    network.IQ_arbiter_input_fifo[ip_type] = defaultdict(lambda: deque(maxlen=2))
                    network.IQ_arbiter_input_fifo_pre[ip_type] = {}

                if ip_type not in network.EQ_channel_buffer:
                    network.EQ_channel_buffer[ip_type] = defaultdict(
                        lambda: deque(maxlen=network.config.EQ_CH_FIFO_DEPTH)
                    )
                    network.EQ_channel_buffer_pre[ip_type] = {}

            # 为所有节点初始化pre buffer和仲裁FIFO（模仿initialize_buffers的行为）
            for ip_type in ip_types:
                for node_pos in range(network.config.NUM_NODE):
                    if node_pos not in network.IQ_channel_buffer_pre[ip_type]:
                        network.IQ_channel_buffer_pre[ip_type][node_pos] = None
                    if node_pos not in network.EQ_channel_buffer_pre[ip_type]:
                        network.EQ_channel_buffer_pre[ip_type][node_pos] = None
                    if node_pos not in network.IQ_arbiter_input_fifo_pre[ip_type]:
                        network.IQ_arbiter_input_fifo_pre[ip_type][node_pos] = None
