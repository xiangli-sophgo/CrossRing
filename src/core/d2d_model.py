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

    def _create_die_instances(self):
        """为每个Die创建独立的BaseModel实例"""
        for die_id in range(self.num_dies):
            die_config = self._create_die_config(die_id)

            # 创建BaseModel实例
            die_model = BaseModel(
                model_type=self.kwargs.get("model_type", "REQ_RSP"),
                config=die_config,
                topo_type=self.kwargs.get("topo_type", "8x9"),
                traffic_file_path=self.kwargs.get("traffic_file_path", "../traffic/"),
                traffic_config=self.traffic_config,
                result_save_path=self.kwargs.get("result_save_path", "../Result/"),
                results_fig_save_path=self.kwargs.get("results_fig_save_path", ""),
                plot_flow_fig=self.kwargs.get("plot_flow_fig", 0),
                plot_RN_BW_fig=self.kwargs.get("plot_RN_BW_fig", 0),
                plot_link_state=self.kwargs.get("plot_link_state", 0),
                plot_start_cycle=self.kwargs.get("plot_start_cycle", 0),
                print_trace=self.kwargs.get("print_trace", 0),
                show_trace_id=self.kwargs.get("show_trace_id", 0),
                verbose=self.kwargs.get("verbose", 1),
            )

            # 设置Die ID
            die_model.die_id = die_id

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
        """为每个Die创建独立的配置"""
        die_config = copy.deepcopy(self.config)

        # 设置Die ID
        die_config.DIE_ID = die_id

        # 获取当前Die的D2D节点位置
        if die_id == 0:
            d2d_positions = getattr(self.config, "D2D_DIE0_POSITIONS", [])
        else:
            d2d_positions = getattr(self.config, "D2D_DIE1_POSITIONS", [])

        # 添加D2D节点到IP列表 (使用BaseModel期望的_0后缀格式)
        if hasattr(die_config, "CH_NAME_LIST"):
            if "d2d_rn_0" not in die_config.CH_NAME_LIST:
                die_config.CH_NAME_LIST.append("d2d_rn_0")
            if "d2d_sn_0" not in die_config.CH_NAME_LIST:
                die_config.CH_NAME_LIST.append("d2d_sn_0")

        # 设置D2D节点的发送位置列表
        die_config.D2D_RN_SEND_POSITION_LIST = d2d_positions
        die_config.D2D_SN_SEND_POSITION_LIST = d2d_positions

        return die_config

    def _add_d2d_nodes_to_die(self, die_model: BaseModel, die_id: int):
        """向Die添加D2D节点"""
        config = die_model.config

        # 获取当前Die的D2D节点位置
        if die_id == 0:
            d2d_positions_list = getattr(self.config, "D2D_DIE0_POSITIONS", [])
        else:
            d2d_positions_list = getattr(self.config, "D2D_DIE1_POSITIONS", [])

        # 创建D2D_Sys并关联已有的D2D接口
        from src.utils.components.d2d_sys import D2D_Sys

        for i, rn_pos in enumerate(d2d_positions_list):
            # 计算对应的SN位置（根据CrossRing拓扑规则）
            sn_pos = rn_pos  # - config.NUM_COL

            # 创建D2D_Sys（使用RN位置作为物理节点标识）
            d2d_sys = D2D_Sys(rn_pos, die_id, config)

            # 获取BaseModel已创建的D2D接口
            d2d_rn = die_model.ip_modules.get(("d2d_rn_0", rn_pos))
            d2d_sn = die_model.ip_modules.get(("d2d_sn_0", sn_pos))

            # 关联D2D_Sys和接口
            if d2d_rn:
                d2d_rn.d2d_sys = d2d_sys
                d2d_sys.rn_interface = d2d_rn
            if d2d_sn:
                d2d_sn.d2d_sys = d2d_sys
                d2d_sys.sn_interface = d2d_sn

            # 存储D2D_Sys
            die_model.d2d_systems[rn_pos] = d2d_sys  # 使用RN位置作为key

        # 设置D2D配置
        d2d_sn_positions = [pos - config.NUM_COL for pos in d2d_positions_list]
        config.D2D_RN_POSITIONS = d2d_positions_list
        config.D2D_SN_POSITIONS = d2d_sn_positions

    def _setup_cross_die_connections(self):
        """建立Die间的连接关系"""
        # 从配置中获取D2D连接对
        d2d_pairs = getattr(self.config, "D2D_PAIRS", [])

        if not d2d_pairs:
            print("警告: 没有找到D2D连接对配置")
            return

        # 为每个D2D连接对建立Die间连接
        for die0_node, die1_node in d2d_pairs:
            # Die0 -> Die1 连接 (使用正确的_0后缀格式)
            die0_rn_key = ("d2d_rn_0", die0_node)
            die1_sn_key = ("d2d_sn_0", die1_node)
            die1_rn_key = ("d2d_rn_0", die1_node)

            if die0_rn_key in self.dies[0].ip_modules and die1_sn_key in self.dies[1].ip_modules:

                d2d_rn = self.dies[0].ip_modules[die0_rn_key]
                d2d_sn = self.dies[1].ip_modules[die1_sn_key]
                target_d2d_rn = self.dies[1].ip_modules[die1_rn_key] if die1_rn_key in self.dies[1].ip_modules else None

                # 建立Die0到Die1的连接（保持向后兼容）
                if 1 not in d2d_rn.target_die_interfaces:
                    d2d_rn.target_die_interfaces[1] = []
                d2d_rn.target_die_interfaces[1].append(d2d_sn)

                # 为D2D_Sys设置目标接口
                # D2D_Sys使用RN位置作为key
                if die0_node in self.dies[0].d2d_systems:
                    d2d_sys = self.dies[0].d2d_systems[die0_node]
                    d2d_sys.target_die_interfaces[1] = {"sn": d2d_sn, "rn": target_d2d_rn}

                # 为Die0的D2D_SN设置目标Die1的D2D_RN接口
                die0_sn_key = ("d2d_sn_0", die0_node)
                if die0_sn_key in self.dies[0].ip_modules:
                    die0_d2d_sn = self.dies[0].ip_modules[die0_sn_key]
                    die0_d2d_sn.target_die_interfaces[1] = target_d2d_rn

            # Die1 -> Die0 连接 (使用正确的_0后缀格式)
            die1_rn_key = ("d2d_rn_0", die1_node)
            die0_sn_key = ("d2d_sn_0", die0_node)
            die0_rn_key = ("d2d_rn_0", die0_node)

            if die1_rn_key in self.dies[1].ip_modules and die0_sn_key in self.dies[0].ip_modules:

                d2d_rn = self.dies[1].ip_modules[die1_rn_key]
                d2d_sn = self.dies[0].ip_modules[die0_sn_key]
                target_d2d_rn = self.dies[0].ip_modules[die0_rn_key] if die0_rn_key in self.dies[0].ip_modules else None

                # 建立Die1到Die0的连接（保持向后兼容）
                if 0 not in d2d_rn.target_die_interfaces:
                    d2d_rn.target_die_interfaces[0] = []
                d2d_rn.target_die_interfaces[0].append(d2d_sn)

                # 为D2D_Sys设置目标接口
                # D2D_Sys使用RN位置作为key
                if die1_node in self.dies[1].d2d_systems:
                    d2d_sys = self.dies[1].d2d_systems[die1_node]
                    d2d_sys.target_die_interfaces[0] = {"sn": d2d_sn, "rn": target_d2d_rn}

                # 为Die1的D2D_SN设置目标Die0的D2D_RN接口
                die1_sn_key = ("d2d_sn_0", die1_node)
                if die1_sn_key in self.dies[1].ip_modules:
                    die1_d2d_sn = self.dies[1].ip_modules[die1_sn_key]
                    die1_d2d_sn.target_die_interfaces[0] = target_d2d_rn

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

        # 检查跨Die事务是否完成：每个Die是否收到了所有期望的跨Die响应
        for die_id in range(self.num_dies):
            expected = self.d2d_expected_flits[die_id]
            received = self.d2d_received_flits[die_id]
            if expected > received:
                return False

        # # 检查每个Die的本地完成状态
        # for die_id, die_model in self.dies.items():
        #     if not die_model.is_completed():
        #         print(f"[DEBUG] Die{die_id}本地事务未完成")
        #         return False

        # print(f"[DEBUG] 所有条件满足，仿真应该结束")
        return True

    def _process_d2d_traffic(self, die_model: BaseModel):
        """处理D2D traffic注入"""
        # 获取当前周期的D2D请求
        pending_requests = self.d2d_traffic_scheduler.get_pending_requests(self.current_cycle)

        for req_data in pending_requests:
            # 检查这个请求是否属于当前Die
            src_die = req_data[1]  # src_die字段
            if src_die != die_model.die_id:
                continue  # 不是当前Die的请求，跳过

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
            # 跨Die：根据目标位置选择D2D节点
            d2d_positions = self._get_die_d2d_positions(src_die)
            intermediate_dest = self._select_d2d_node_for_target(dst_node, dst_ip, d2d_positions, die_model, dst_die)
            # 找到对应的D2D_SN channel编号
            d2d_index = d2d_positions.index(intermediate_dest) if intermediate_dest in d2d_positions else 0
            destination_type = f"d2d_sn_{d2d_index}"  # 跨Die时目标是D2D_SN，包含正确的编号
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
        if die_id == 0:
            return getattr(self.config, "D2D_DIE0_POSITIONS", [])
        else:
            return getattr(self.config, "D2D_DIE1_POSITIONS", [])

    def _select_d2d_node_for_target(self, dst_node, dst_ip, d2d_positions, die_model: base_model, dst_die):
        """根据目标DDR位置选择对应的D2D节点（VERTICAL布局）"""

        # 只处理目标是DDR的情况
        if not dst_ip.startswith("ddr"):
            # 非DDR目标，返回第一个D2D节点
            return d2d_positions[0] if d2d_positions else None

        # 获取拓扑参数
        num_col = die_model.config.NUM_COL  # 4列
        dst_node = die_model.node_map(dst_node)

        # 计算目标节点的行列位置
        dst_row = dst_node // num_col
        dst_col = dst_node % num_col

        # Die0（上方）的映射规则
        if dst_die == 0:
            if dst_col == 0:  # 第一列
                if dst_row == 3:  # 第4行
                    d2d_index = 0
                elif dst_row == 5:  # 第6行
                    d2d_index = 1
                elif dst_row == 7:  # 第8行
                    d2d_index = 0
                elif dst_row == 9:  # 第10行
                    d2d_index = 1
                else:
                    d2d_index = 0  # 默认
            elif dst_col == num_col - 1:  # 最后一列
                if dst_row == 3:  # 第4行
                    d2d_index = 3
                elif dst_row == 5:  # 第6行
                    d2d_index = 2
                elif dst_row == 7:  # 第8行
                    d2d_index = 3
                elif dst_row == 9:  # 第10行
                    d2d_index = 2
                else:
                    d2d_index = 2  # 默认
            else:
                d2d_index = 0  # 非边界列，使用默认

        # Die1（下方）的映射规则
        else:  # src_die == 1
            if dst_col == 0:  # 第一列
                if dst_row == 1:  # 第2行
                    d2d_index = 0
                elif dst_row == 3:  # 第4行
                    d2d_index = 1
                elif dst_row == 5:  # 第6行
                    d2d_index = 0
                elif dst_row == 7:  # 第8行
                    d2d_index = 1
                else:
                    d2d_index = 0  # 默认
            elif dst_col == num_col - 1:  # 最后一列
                if dst_row == 1:  # 第2行
                    d2d_index = 3
                elif dst_row == 3:  # 第4行
                    d2d_index = 2
                elif dst_row == 5:  # 第6行
                    d2d_index = 3
                elif dst_row == 7:  # 第8行
                    d2d_index = 2
                else:
                    d2d_index = 2  # 默认
            else:
                d2d_index = 0  # 非边界列，使用默认

        # 确保索引在有效范围内
        if d2d_index >= len(d2d_positions):
            d2d_index = 0

        return d2d_positions[d2d_index] - num_col

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

            # 使用新的D2D统计计数器
            d2d_sent = self.d2d_requests_sent.get(die_id, 0)
            d2d_recv = self.d2d_requests_completed.get(die_id, 0)  # 使用完成的请求数

            print(
                f"  Die{die_id}: Req_cnt: {req_cnt}, In_Req: {in_req}, Rsp: {rsp}, "
                f"R_fn: {r_fn}, W_fn: {w_fn}, Trans_fn: {trans_fn}, Recv_fn: {recv_fn}, "
                f"D2D_send: {d2d_sent}, D2D_recv: {d2d_recv}"
            )

    def generate_combined_flow_graph(self, mode="total", save_path=None, show_cdma=True):
        """
        生成D2D双Die组合流量图

        Args:
            mode: 显示模式，支持 'utilization', 'total', 'ITag_ratio' 等
            save_path: 图片保存路径
            show_cdma: 是否显示CDMA
        """
        # 检查是否有BandwidthAnalyzer实例
        analyzers = {}
        die_networks = {}

        for die_id, die_model in self.dies.items():
            if hasattr(die_model, "result_processor") and die_model.result_processor:
                analyzers[die_id] = die_model.result_processor
                # 收集网络对象 - 根据mode选择合适的网络
                if mode == "total":
                    die_networks[die_id] = die_model.data_network  # 使用data_network显示总带宽
                else:
                    die_networks[die_id] = die_model.req_network  # 默认使用req_network

            else:
                return

        if len(analyzers) < 2:
            return

        # 使用第一个analyzer来绘制组合图
        primary_analyzer = list(analyzers.values())[0]

        # 设置保存路径
        if save_path is None:
            import time

            timestamp = int(time.time())
            save_path = f"../Result/d2d_combined_flow_{mode}_{timestamp}.png"

        # 确保primary_analyzer有必要的属性
        if not hasattr(primary_analyzer, "simulation_end_cycle"):
            primary_analyzer.simulation_end_cycle = self.current_cycle

        try:
            # 调用新的draw_d2d_flow_graph方法
            primary_analyzer.draw_d2d_flow_graph(die_networks=die_networks, config=self.config, mode=mode, save_path=save_path, show_cdma=show_cdma)

            print(f"D2D组合流量图已保存: {save_path}")

        except Exception as e:
            import traceback

            print(f"生成D2D组合流量图失败: {e}")

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

        # 1. 对每个Die调用现有的结果处理方法
        die_results = {}
        for die_id, die_model in self.dies.items():
            print(f"\n处理Die {die_id}的结果...")

            # 调用现有的综合结果处理方法
            if hasattr(die_model, "result_processor") and die_model.result_processor:
                try:
                    # 收集请求数据
                    die_model.result_processor.collect_requests_data(die_model, die_model.cycle)
                    # 分析带宽
                    die_results[die_id] = die_model.result_processor.analyze_all_bandwidth()

                    # 为每个Die生成独立的报告（保存到子目录）
                    die_result_path = os.path.join(self.kwargs.get("result_save_path", "../Result/"), f"Die_{die_id}")
                    os.makedirs(die_result_path, exist_ok=True)
                    die_model.result_processor.generate_unified_report(die_results[die_id], die_result_path)

                    # 生成FIFO使用率报告
                    die_model.result_processor.generate_fifo_usage_csv(die_model, die_result_path)

                    print(f"  Die {die_id}结果已保存到: {die_result_path}")

                except Exception as e:
                    print(f"  警告: Die {die_id}结果处理失败: {e}")
                    die_results[die_id] = None

        # 2. 输出D2D专有统计信息
        self._print_d2d_statistics(d2d_stats)

        # 3. 生成D2D组合报告（基于现有结果）
        self._generate_d2d_combined_report(die_results, d2d_stats)

        print("=" * 60)
        print("D2D综合结果分析完成")
        print("=" * 60)

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
            d2d_rn_key = ("d2d_rn", die_model.config.D2D_RN_POSITION)
            d2d_sn_key = ("d2d_sn", die_model.config.D2D_SN_POSITION)

            if d2d_rn_key in die_model.ip_modules:
                rn_stats = die_model.ip_modules[d2d_rn_key].get_statistics()
                die_stat["d2d_rn_sent"] = rn_stats["cross_die_requests_sent"]
                die_stat["d2d_rn_received"] = rn_stats["cross_die_responses_received"]
                d2d_stats["cross_die_requests"] += rn_stats["cross_die_requests_sent"]

            if d2d_sn_key in die_model.ip_modules:
                sn_stats = die_model.ip_modules[d2d_sn_key].get_statistics()
                die_stat["d2d_sn_received"] = sn_stats["cross_die_requests_received"]
                die_stat["d2d_sn_forwarded"] = sn_stats["cross_die_requests_forwarded"]
                die_stat["d2d_sn_responses"] = sn_stats["cross_die_responses_sent"]
                d2d_stats["cross_die_responses"] += sn_stats["cross_die_responses_sent"]

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
                    "datasend": "pos",  # datasend is positive response
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
                        if hasattr(flit, "source_die_id") and hasattr(flit, "target_die_id") and flit.source_die_id != flit.target_die_id:
                            has_cross_die_flit = True

                            # 对于跨Die请求，需要检查完整的流程
                            # 读流程：需要回到源Die的源节点
                            # 写流程：需要收到写响应
                            if hasattr(flit, "req_type"):
                                # 请求flit：检查是否回到了源Die的源节点
                                if flit.flit_type == "rsp" or flit.flit_type == "data":
                                    # 响应/数据：需要回到源Die
                                    current_die = self._get_flit_current_die(flit, die_id)
                                    if current_die != flit.source_die_id or not hasattr(flit, "flit_position") or flit.flit_position != "IP_eject":
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

        # 只有当存在flit且所有阶段都完成时才标记为完成
        # 暂时禁用完成检查，让第三阶段能够显示
        # TODO: 需要实现完整的6阶段完成检查逻辑
        if False and has_any_flit and all_completed:
            self._d2d_done_flags[packet_id] = True
            if has_cross_die_flit:
                print(f"[D2D Trace] Packet {packet_id} cross-die transmission completed!")
            else:
                print(f"[D2D Trace] Packet {packet_id} local transmission completed!")

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
