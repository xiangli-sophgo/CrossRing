"""
D2D_Model class for Die-to-Die simulation.
Manages multiple die instances and coordinates cross-die communication.
"""

import copy
import time
import logging
import os
from typing import Dict, List, Optional

from .base_model import BaseModel
from .d2d_traffic_scheduler import D2DTrafficScheduler
from .d2d_result_processor import D2DResultProcessor
from src.utils.components.d2d_rn_interface import D2D_RN_Interface
from src.utils.components.d2d_sn_interface import D2D_SN_Interface
from config.config import CrossRingConfig

from src.core import base_model


class D2D_Model:
    """
    D2D仿真主类 - 管理多Die协调
    每个Die是独立的BaseModel实例，D2D_Model负责：
    1. 创建和管理多个Die实例
    2. 设置Die间的连接关系
    3. 协调多Die的时钟同步
    """

    def __init__(self, config: CrossRingConfig, traffic_config: list, **kwargs):
        self.config = config
        self.traffic_config = traffic_config
        self.kwargs = kwargs
        self.current_cycle = 0

        # 获取Die数量，默认为2
        self.num_dies = getattr(config, "NUM_DIES", 2)

        # 存储各Die实例
        self.dies: Dict[int, BaseModel] = {}

        # 仿真参数
        self.end_time = getattr(config, "END_TIME", 10000)
        self.print_interval = getattr(config, "PRINT_INTERVAL", 1000)

        # 流量图控制参数
        self.enable_flow_graph = kwargs.get("enable_flow_graph", True)  # 是否启用流量图绘制
        self.flow_graph_mode = kwargs.get("flow_graph_mode", "total")  # 流量图模式：total, utilization, ITag_ratio

        # 统计信息
        self.total_cross_die_requests = 0
        self.total_cross_die_responses = 0

        # D2D跨Die事务简化统计
        self.d2d_expected_flits = {i: 0 for i in range(self.num_dies)}  # 每个Die期望接收的跨Die数据包数
        self.d2d_received_flits = {i: 0 for i in range(self.num_dies)}  # 每个Die实际接收的跨Die数据包数
        self.d2d_requests_sent = {i: 0 for i in range(self.num_dies)}  # 每个Die发出的跨Die请求数
        self.d2d_requests_completed = {i: 0 for i in range(self.num_dies)}  # 每个Die完成的跨Die请求数

        # 创建D2D专用的traffic调度器
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
            from .D2D_Link_State_Visualizer import D2D_Link_State_Visualizer

            # 获取第一个Die的第一个网络作为初始网络（用于配置信息）
            initial_network = self.dies[0].req_network
            self.d2d_link_state_vis = D2D_Link_State_Visualizer(self.num_dies, initial_network)

    def _create_die_instances(self):
        """为每个Die创建独立的BaseModel实例"""
        for die_id in range(self.num_dies):
            die_config = self._create_die_config(die_id)

            # 创建BaseModel实例（D2D模式下禁用单个Die的结果保存路径）
            # 只在第一个Die时启用verbose，避免重复打印
            die_verbose = self.kwargs.get("verbose", 1) if die_id == 0 else 0
            die_model = BaseModel(
                model_type=self.kwargs.get("model_type", "REQ_RSP"),
                config=die_config,
                topo_type=self.kwargs.get("topo_type", "5x4"),
                traffic_file_path=self.kwargs.get("traffic_file_path", "../traffic/"),
                traffic_config=self.traffic_config,
                result_save_path="",  # 禁用单个Die的结果保存，避免生成Die_0/Die_1文件夹
                results_fig_save_path="",  # 禁用单个Die的图片保存，避免生成figure文件夹
                plot_flow_fig=0,  # 禁用单个Die的流图生成
                plot_RN_BW_fig=0,  # 禁用单个Die的带宽图生成
                plot_link_state=0,  # 禁用单个Die的可视化，使用D2D统一可视化
                plot_start_cycle=self.kwargs.get("plot_start_cycle", 0),
                print_trace=self.kwargs.get("print_trace", 0),
                show_trace_id=self.kwargs.get("show_trace_id", 0),
                verbose=die_verbose,  # 只在第一个Die时启用verbose
            )

            # 设置Die ID
            die_model.die_id = die_id

            # 调试：检查Die配置中的CHANNEL_SPEC和CH_NAME_LIST是否包含D2D

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
        if hasattr(die_config, "CHANNEL_SPEC"):
            # 更新CHANNEL_SPEC
            if "d2d_rn" not in die_config.CHANNEL_SPEC:
                die_config.CHANNEL_SPEC["d2d_rn"] = 1
            if "d2d_sn" not in die_config.CHANNEL_SPEC:
                die_config.CHANNEL_SPEC["d2d_sn"] = 1

            # 更新CH_NAME_LIST
            if hasattr(die_config, "CH_NAME_LIST"):
                if "d2d_rn_0" not in die_config.CH_NAME_LIST:
                    die_config.CH_NAME_LIST.append("d2d_rn_0")
                if "d2d_sn_0" not in die_config.CH_NAME_LIST:
                    die_config.CH_NAME_LIST.append("d2d_sn_0")

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
        from src.utils.components.d2d_sys import D2D_Sys

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
        """建立单向连接"""

        # 获取IP接口
        src_rn_key = ("d2d_rn_0", src_node)
        src_sn_key = ("d2d_sn_0", src_node)
        dst_sn_key = ("d2d_sn_0", dst_node + self.dies[dst_die_id].config.NUM_COL)
        dst_rn_key = ("d2d_rn_0", dst_node + self.dies[dst_die_id].config.NUM_COL)

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
                # 收集所有Die的所有网络数据
                all_die_networks = []
                for die_id in range(self.num_dies):
                    die_model = self.dies[die_id]
                    die_networks = [die_model.req_network, die_model.rsp_network, die_model.data_network]
                    all_die_networks.append(die_networks)

                # 更新可视化器
                self.d2d_link_state_vis.update(all_die_networks, self.current_cycle)

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

        # 根据参数决定是否生成流量图
        if self.enable_flow_graph:
            if self.kwargs.get("verbose", 1):
                # print(f"\n生成D2D流量图 (模式: {self.flow_graph_mode})")
                pass
            try:
                self.generate_combined_flow_graph(mode=self.flow_graph_mode, save_path=None, show_cdma=True)
            except Exception as e:
                print(f"流量图生成失败: {e}")

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
            expected_write_complete = self._local_write_requests[die_id] + self._cross_die_write_requests[die_id]  # 该Die发起的所有写请求
            actual_write_complete = self._write_complete_received[die_id]

            if actual_write_complete < expected_write_complete:
                return False  # 写完成响应未接收完

        # 检查网络是否空闲和没有待处理的事务
        for die_id, die_model in self.dies.items():
            # 检查网络中是否还有传输中的flits
            trans_completed = die_model.trans_flits_num == 0
            # 检查是否有新的写请求待处理
            write_completed = not die_model.new_write_req

            if not (trans_completed and write_completed):
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

        # 使用die_model的node_map进行节点映射
        source_physical = die_model.node_map(src_node, True)

        # 根据是否跨Die决定路由策略
        if src_die != dst_die:
            # 跨Die：使用D2D路由器选择节点，但需要根据请求类型选择正确的RN/SN节点
            base_d2d_node = self.d2d_router.select_d2d_node(src_die, dst_die, dst_node)
            if base_d2d_node is None:
                raise ValueError(f"D2D路由器返回None，但这是跨Die请求 Die{src_die}->Die{dst_die}")

            # 根据请求类型选择正确的D2D节点：
            # 请求需要发送到D2D_SN（偶数行），响应从D2D_RN发出（奇数行）
            d2d_sn_positions = getattr(self.config, "D2D_SN_POSITIONS", {}).get(src_die, [])
            base_d2d_node -= die_model.config.NUM_COL

            # 找到对应的D2D_SN节点位置和索引
            if base_d2d_node in d2d_sn_positions:
                intermediate_dest = base_d2d_node
            else:
                raise ValueError("未找到d2d_sn")

            destination_type = f"d2d_sn_0"  # 跨Die时目标是D2D_SN，包含正确的编号
        else:
            # 本地：直接路由到目标
            intermediate_dest = die_model.node_map(dst_node, False)
            destination_type = dst_ip

        # 创建flit（参考BaseModel._process_single_request）
        from src.utils.components.flit import Flit
        from src.utils.components.node import Node

        path = die_model.routes[source_physical][intermediate_dest]
        req = Flit.create_flit(source_physical, intermediate_dest, path)

        # 设置D2D统一属性（新设计）
        req.d2d_origin_die = src_die  # 发起Die ID
        req.d2d_origin_node = source_physical  # 发起节点源映射位置
        req.d2d_origin_type = src_ip  # 发起IP类型

        req.d2d_target_die = dst_die  # 目标Die ID
        # 目标节点源映射位置（统一保存源映射）
        # bug: 应该是目标Die的node_map，而不是源Die的node_map
        req.d2d_target_node = die_model.node_map(dst_node, True)  # 目标节点的源映射
        req.d2d_target_type = dst_ip  # 目标IP类型

        # 设置标准属性（与BaseModel一致）
        req.source_original = src_node
        req.destination_original = intermediate_dest
        req.flit_type = "req"
        req.departure_cycle = inject_time
        req.burst_length = burst_length
        req.source_type = f"{src_ip}_0" if "_" not in src_ip else src_ip
        req.destination_type = destination_type  # 已经包含了正确的编号
        req.original_source_type = f"{src_ip}_0" if "_" not in src_ip else src_ip
        req.original_destination_type = f"{dst_ip}_0" if "_" not in dst_ip else dst_ip
        req.req_type = "read" if req_type == "R" else "write"
        req.req_attr = "new"
        req.traffic_id = traffic_id
        req.packet_id = Node.get_next_packet_id()

        # 设置保序信息
        req.set_packet_category_and_order_id()

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

            print(f"    Read - Requests: Local={local_read_reqs}, Cross={cross_read_reqs} | Data: Local={local_read_data}, Cross={cross_read_data}")
            print(f"    Write - Requests: Local={local_write_reqs}, Cross={cross_write_reqs} | Data: Local={local_write_data}, Cross={cross_write_data}")

    def generate_combined_flow_graph(self, mode="total", save_path=None, show_cdma=True):
        """
        生成D2D双Die组合流量图

        Args:
            mode: 显示模式，支持 'utilization', 'total', 'ITag_ratio' 等
            save_path: 图片保存路径
            show_cdma: 是否显示CDMA
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
            d2d_processor = D2DResultProcessor(self.config)
            d2d_processor.simulation_end_cycle = self.current_cycle

            # 检查是否已经有D2D处理器实例（避免重复计算）
            if hasattr(self, "_cached_d2d_processor") and self._cached_d2d_processor:
                d2d_processor = self._cached_d2d_processor
            else:
                # 第一次计算：收集D2D请求数据并计算正确的IP带宽
                d2d_processor.collect_cross_die_requests(self.dies)
                d2d_processor.calculate_d2d_ip_bandwidth_data(self.dies)
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
            d2d_processor.draw_d2d_flow_graph(dies=self.dies, config=self.config, mode=mode, save_path=save_path, show_cdma=show_cdma)

            if save_path:
                print(f"D2D组合流量图已保存: {save_path}")
            else:
                print("D2D组合流量图已显示")

        except Exception as e:
            print(f"生成D2D组合流量图失败: {e}")
            import traceback

            traceback.print_exc()

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
        if enable_flow_graph:
            self.generate_combined_flow_graph(mode=flow_mode)

        # 处理D2D综合结果分析
        self.process_d2d_comprehensive_results()

    def process_d2d_comprehensive_results(self):
        """
        处理D2D综合结果分析，复用现有的结果处理方法
        """
        print("\n" + "=" * 60)
        print("D2D仿真综合结果分析")
        print("=" * 60)

        # 收集D2D专有统计信息
        d2d_stats = self._collect_d2d_statistics()

        # 1. 跳过Die内部结果分析（D2D系统中Die内部没有数据流）
        die_results = {}

        # 2. D2D专用结果处理
        self._process_d2d_specific_results()

        # 3. 输出D2D专有统计信息（可选，现在主要依靠D2D专用报告）
        # self._print_d2d_statistics(d2d_stats)

        # 4. 不再生成重复的组合报告，D2D专用报告已足够

        print("=" * 60)
        print("D2D综合结果分析完成")
        print("=" * 60)

    def _process_d2d_specific_results(self):
        """处理D2D专有的结果分析（跨Die请求记录和带宽统计）"""
        try:
            print("\n处理D2D专用结果分析...")

            # 使用缓存的D2D处理器（如果存在），避免重复计算
            if hasattr(self, "_cached_d2d_processor") and self._cached_d2d_processor:
                # print("[信息] 复用缓存的D2D处理器，避免重复计算")
                d2d_processor = self._cached_d2d_processor
            else:
                # 如果没有缓存，创建新的处理器
                print("[警告] 没有找到缓存的D2D处理器，创建新实例")
                d2d_processor = D2DResultProcessor(self.config)
                d2d_processor.simulation_end_cycle = self.current_cycle
                d2d_processor.finish_cycle = self.current_cycle

                # 直接设置所需属性，避免创建临时sim_model
                d2d_processor.topo_type_stat = "5x4"  # D2D系统使用5x4拓扑
                d2d_processor.results_fig_save_path = self.kwargs.get("results_fig_save_path", "../Result/")
                d2d_processor.file_name = "d2d_system"
                d2d_processor.verbose = self.kwargs.get("verbose", 1)
                d2d_processor.flow_fig_show_CDMA = True

                # 收集数据（如果没有缓存的话）
                d2d_processor.collect_cross_die_requests(self.dies)
                d2d_processor.calculate_d2d_ip_bandwidth_data(self.dies)

            # 获取结果保存路径，添加时间戳
            import time

            timestamp = int(time.time())
            result_save_path = self.kwargs.get("result_save_path", "../Result/")
            d2d_result_path = os.path.join(result_save_path, f"D2D_{timestamp}")

            # 只执行保存和报告生成部分，避免重复计算
            # print("[信息] 保存D2D请求到CSV并生成带宽报告")
            d2d_processor.save_d2d_requests_csv(d2d_result_path)
            d2d_processor.generate_d2d_bandwidth_report(d2d_result_path)

        except Exception as e:
            import traceback

            print(f"警告: D2D专用结果处理失败: {e}")
            print("详细错误信息:")
            traceback.print_exc()

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
                die_stat["axi_channel_flit_count"] = {}
                total_axi_flits = 0
                for pos, d2d_sys in die_model.d2d_systems.items():
                    if hasattr(d2d_sys, "axi_channel_flit_count"):
                        for channel, count in d2d_sys.axi_channel_flit_count.items():
                            if channel not in die_stat["axi_channel_flit_count"]:
                                die_stat["axi_channel_flit_count"][channel] = 0
                            die_stat["axi_channel_flit_count"][channel] += count
                            total_axi_flits += count
                die_stat["total_axi_flits"] = total_axi_flits

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
            if "axi_channel_flit_count" in stat and stat["axi_channel_flit_count"]:
                print(f"  AXI通道传输统计 (总计: {stat.get('total_axi_flits', 0)} flits):")
                for channel, count in stat["axi_channel_flit_count"].items():
                    print(f"    {channel}: {count} flits")

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
