"""
DualChannelBaseModel class for NoC simulation.
Extends BaseModel class to support dual-channel data transmission with independent arbitration.
简化版本：使用两个独立的Network实例实现双通道
"""

import time
import matplotlib.pyplot as plt

from .base_model import BaseModel
from src.kcin.v1.components import Network, DualChannelIPInterface
from src.kcin.base.channel_selector import DefaultChannelSelector
from src.analysis.Dual_Channel_Link_State_Visualizer import DualChannelNetworkLinkVisualizer
import logging


class DualChannelBaseModel(BaseModel):
    """双通道基础模型，支持数据双通道传输 - 简化版本"""

    def __init__(self, *args, **kwargs):
        # 调用父类初始化
        super().__init__(*args, **kwargs)

        # 初始化IP类型的ID计数器
        self.ip_id_counters = {"gdma": 0, "sdma": 0, "cdma": 0, "ddr": 0, "l2m": 0}

        # 设置通道选择器（在父类初始化后，config已可用）
        self.channel_selector = self._create_channel_selector()


    def _create_channel_selector(self):
        """创建通道选择器"""
        strategy = getattr(self.config, "DATA_CHANNEL_SELECT_STRATEGY", "ip_id_based")
        return DefaultChannelSelector(strategy)

    def _create_dual_channel_data_network(self):
        """创建双通道数据网络 - 使用两个独立的Network实例"""
        # 创建两个独立的数据网络实例
        self.data_network_ch0 = Network(self.config, self.adjacency_matrix, "Data Channel 0")
        self.data_network_ch1 = Network(self.config, self.adjacency_matrix, "Data Channel 1")

        # 更新通道选择器的网络引用
        self.channel_selector.network = self.data_network_ch0

    def initial(self):
        """重写初始化方法，使用双通道IP接口"""
        # 临时禁用可视化器创建，避免父类创建单通道可视化器
        original_plot_link_state = self.plot_link_state
        self.plot_link_state = False
        
        # 调用父类初始化但不让它创建可视化器
        super().initial()
        
        # 恢复可视化器设置
        self.plot_link_state = original_plot_link_state

        # 创建双通道数据网络
        self._create_dual_channel_data_network()

        # 重新创建IP模块为双通道版本
        self.ip_modules = {}

        for ip_pos in self.flit_positions:
            for ip_type in self.config.CH_NAME_LIST:
                # 提取基础IP类型名（去掉数字后缀）
                base_ip_type = ip_type.split("_")[0]

                # 如果基础类型不在计数器中，使用默认的计数器
                if base_ip_type not in self.ip_id_counters:
                    # 动态添加计数器
                    self.ip_id_counters[base_ip_type] = 0

                # 为当前IP类型分配IP_ID并递增计数器
                ip_id = self.ip_id_counters[base_ip_type]
                self.ip_id_counters[base_ip_type] += 1

                self.ip_modules[(ip_type, ip_pos)] = DualChannelIPInterface(
                    ip_type,
                    ip_pos,
                    self.config,
                    self.req_network,
                    self.rsp_network,
                    self.data_network_ch0,
                    self.data_network_ch1,
                    self.node,
                    self.routes,
                    self.channel_selector,
                    ip_id,
                )

        # 初始化双通道可视化器（在所有网络创建完成后）
        if self.plot_link_state:
            self.link_state_vis = DualChannelNetworkLinkVisualizer(self.config, self.data_network_ch0)

    def run(self):
        """重写父类的仿真主循环以支持双通道"""
        simulation_start = time.perf_counter()
        self.load_request_stream()
        reqs, rsps = [], []
        flits_ch0, flits_ch1 = [], []  # 为每个数据通道创建独立的flit列表
        self.cycle = 0
        tail_time = 6

        while True:
            self.cycle += 1
            self.cycle_mod = self.cycle % self.config.NETWORK_FREQUENCY

            self.release_completed_sn_tracker()
            self.process_new_request()
            self.tag_move_all_networks()
            self.ip_inject_to_network()

            # REQ and RSP networks (no change)
            self._inject_queue_arbitration(self.req_network, self.rn_positions_list, "req")
            reqs = self.move_flits_in_network(self.req_network, reqs, "req")
            self._inject_queue_arbitration(self.rsp_network, self.sn_positions_list, "rsp")
            rsps = self.move_flits_in_network(self.rsp_network, rsps, "rsp")

            # Data networks (dual channel handling)
            self._inject_queue_arbitration(self.data_network_ch0, self.flit_positions_list, "data")
            self._inject_queue_arbitration(self.data_network_ch1, self.flit_positions_list, "data")
            flits_ch0 = self.move_flits_in_network(self.data_network_ch0, flits_ch0, "data")
            flits_ch1 = self.move_flits_in_network(self.data_network_ch1, flits_ch1, "data")

            self.network_to_ip_eject()
            self.move_pre_to_queues_all()

            # Link statistics for all networks
            self.req_network.collect_cycle_end_link_statistics(self.cycle)
            self.rsp_network.collect_cycle_end_link_statistics(self.cycle)
            self.data_network_ch0.collect_cycle_end_link_statistics(self.cycle)
            self.data_network_ch1.collect_cycle_end_link_statistics(self.cycle)

            self.debug_func()

            # Update throughput metrics with combined flits
            self.update_throughput_metrics(flits_ch0 + flits_ch1)

            if self.cycle / self.config.NETWORK_FREQUENCY % self.print_interval == 0:
                self.log_summary()

            # Check for traffic completion and advance chains
            completed_traffics = self.traffic_scheduler.check_and_advance_chains(self.cycle)
            if completed_traffics and self.verbose:
                print(f"Completed traffics: {completed_traffics}")

            # Termination condition with dual channel support
            total_recv_flits = self.data_network_ch0.recv_flits_num + self.data_network_ch1.recv_flits_num
            if (
                self.traffic_scheduler.is_all_completed() and total_recv_flits >= (self.read_flit + self.write_flit) and self.trans_flits_num == 0 and not self.new_write_req
            ) or self.cycle > self.end_time * self.config.NETWORK_FREQUENCY:
                if tail_time == 0:
                    if self.verbose:
                        print("Finish!")
                    break
                else:
                    tail_time -= 1

        # Final statistics and reporting
        self.print_data_statistic()
        self.log_summary()
        self.syn_IP_stat()
        self.update_finish_time_stats()
        # 在处理综合结果之前合并双通道数据
        self.collect_dual_channel_data()
        # 先打印双通道负载分布
        self.print_dual_channel_summary()
        self.process_comprehensive_results()

        simulation_end = time.perf_counter()
        simulation_time = simulation_end - simulation_start
        self.simulation_total_time = simulation_time

        if self.verbose:
            print(f"Simulation completed in {simulation_time:.2f} seconds")
            print(f"Processed {self.cycle} cycles")
            print(f"Performance: {self.cycle / simulation_time:.0f} cycles/second")

    def log_summary(self):
        """重写日志方法以显示双通道统计信息"""
        if self.verbose:
            total_recv = self.data_network_ch0.recv_flits_num + self.data_network_ch1.recv_flits_num
            print(
                f"T: {self.cycle // self.config.NETWORK_FREQUENCY}, Req_cnt: {self.req_count} In_Req: {self.req_num}, Rsp: {self.rsp_num},"
                f" R_fn: {self.send_read_flits_num_stat}, W_fn: {self.send_write_flits_num_stat}, "
                f"Trans_fn: {self.trans_flits_num}, Recv_fn_ch0: {self.data_network_ch0.recv_flits_num}, Recv_fn_ch1: {self.data_network_ch1.recv_flits_num}, Recv_fn_total: {total_recv}"
            )

    def debug_func(self):
        """重写调试方法以支持双通道可视化"""
        if self.print_trace:
            self.flit_trace(self.show_trace_id)
        if self.plot_link_state:
            while self.link_state_vis.paused and not self.link_state_vis.should_stop:
                plt.pause(0.05)
            if self.link_state_vis.should_stop:
                return
            if self.cycle < self.plot_start_cycle:
                return
            # 可视化所有网络，包括两个数据通道
            self.link_state_vis.update([self.req_network, self.rsp_network, self.data_network_ch0, self.data_network_ch1], self.cycle)

    def flit_trace(self, packet_id):
        """重写追踪方法以显示双通道flit"""
        if self.plot_link_state and self.link_state_vis.should_stop:
            return

        packet_ids = [packet_id] if isinstance(packet_id, (int, str)) else packet_id

        for pid in packet_ids:
            # 收集所有flit信息
            all_flits_str = []
            has_active_flit = False
            
            # REQ
            req_flits = self.req_network.send_flits.get(pid, [])
            for flit in req_flits:
                if not self._should_skip_waiting_flit(flit):
                    has_active_flit = True
                all_flits_str.append(f"REQ,{flit}")
                
            # RSP
            rsp_flits = self.rsp_network.send_flits.get(pid, [])
            for flit in rsp_flits:
                if not self._should_skip_waiting_flit(flit):
                    has_active_flit = True
                all_flits_str.append(f"RSP,{flit}")
                
            # DATA CH0
            ch0_flits = self.data_network_ch0.send_flits.get(pid, [])
            for flit in ch0_flits:
                if not self._should_skip_waiting_flit(flit):
                    has_active_flit = True
                all_flits_str.append(f"DATA_CH0,{flit}")
                
            # DATA CH1
            ch1_flits = self.data_network_ch1.send_flits.get(pid, [])
            for flit in ch1_flits:
                if not self._should_skip_waiting_flit(flit):
                    has_active_flit = True
                all_flits_str.append(f"DATA_CH1,{flit}")

            # 只有当有活跃的flit时才打印
            if has_active_flit and all_flits_str:
                if self.cycle != self._last_printed_cycle:
                    print(f"Cycle {self.cycle}:")
                    self._last_printed_cycle = self.cycle
                print(" | ".join(all_flits_str) + " |")
                time.sleep(0.3)

    def _inject_queue_arbitration(self, network=None, positions_list=None, network_type=None):
        """重写注入队列仲裁以支持双通道数据网络"""
        if network_type == "data":
            # 数据网络使用双通道仲裁：对每个通道独立调用原有仲裁
            super()._inject_queue_arbitration(self.data_network_ch0, self.flit_positions_list, "data")
            super()._inject_queue_arbitration(self.data_network_ch1, self.flit_positions_list, "data")
        elif network is not None and positions_list is not None and network_type is not None:
            # 有参数的调用，使用父类方法
            super()._inject_queue_arbitration(network, positions_list, network_type)
        else:
            # 无参数调用，执行双通道仲裁
            super()._inject_queue_arbitration(self.data_network_ch0, self.flit_positions_list, "data")
            super()._inject_queue_arbitration(self.data_network_ch1, self.flit_positions_list, "data")

    def move_pre_to_queues_all(self):
        """重写pre到queue移动，支持双通道数据网络"""
        # 所有IPInterface的*_pre → FIFO
        for ip_pos in self.flit_positions_list:
            for ip_type in self.config.CH_NAME_LIST:
                self.ip_modules[(ip_type, ip_pos)].move_pre_to_fifo()

        # req和rsp网络使用原有逻辑
        for in_pos in self.flit_positions_list:
            self._move_pre_to_queues(self.req_network, in_pos)
            self._move_pre_to_queues(self.rsp_network, in_pos)
            self._move_pre_to_queues(self.data_network_ch0, in_pos)
            self._move_pre_to_queues(self.data_network_ch1, in_pos)

    def collect_dual_channel_data(self):
        """合并双通道数据用于结果处理"""
        # 为了让父类的统计和分析工具能工作，我们需要临时创建一个合并后的data_network
        # 注意：这只是为了兼容旧的分析工具，核心的run循环已经分离了
        if not hasattr(self, "data_network") or self.data_network is None:
            self.data_network = Network(self.config, self.adjacency_matrix, "Data Channel Merged")

        # 合并arrive_flits - 需要合并同一packet_id的flits
        merged_arrive_flits = {}

        # 先添加通道0的数据
        for packet_id, flits in self.data_network_ch0.arrive_flits.items():
            merged_arrive_flits[packet_id] = flits[:]  # 复制列表

        # 再添加通道1的数据，如果有相同packet_id则合并
        for packet_id, flits in self.data_network_ch1.arrive_flits.items():
            if packet_id in merged_arrive_flits:
                merged_arrive_flits[packet_id].extend(flits)
            else:
                merged_arrive_flits[packet_id] = flits[:]

        self.data_network.arrive_flits = merged_arrive_flits

        # 合并统计数据
        self.data_network.recv_flits_num = self.data_network_ch0.recv_flits_num + self.data_network_ch1.recv_flits_num
        self.data_network.inject_num = self.data_network_ch0.inject_num + self.data_network_ch1.inject_num
        self.data_network.eject_num = self.data_network_ch0.eject_num + self.data_network_ch1.eject_num

        # 合并发送数据统计
        merged_send_flits = {}
        for packet_id, flits in self.data_network_ch0.send_flits.items():
            merged_send_flits[packet_id] = flits
        for packet_id, flits in self.data_network_ch1.send_flits.items():
            if packet_id in merged_send_flits:
                merged_send_flits[packet_id].extend(flits)
            else:
                merged_send_flits[packet_id] = flits
        self.data_network.send_flits = merged_send_flits

        # 合并links_flow_stat数据（这是带宽计算的关键）
        self._merge_links_flow_stat()


    def result_statistic(self):
        """重写结果统计，合并双通道数据"""
        # 先合并双通道数据以兼容父类分析工具
        self.collect_dual_channel_data()

        # 调用父类的结果统计
        super().result_statistic()

    def print_dual_channel_summary(self):
        """打印双通道仿真总结"""
        # 获取统计数据
        ch0_recv = self.data_network_ch0.recv_flits_num
        ch1_recv = self.data_network_ch1.recv_flits_num
        total_recv = ch0_recv + ch1_recv

        if total_recv > 0:
            ch0_ratio = ch0_recv / total_recv * 100
            ch1_ratio = ch1_recv / total_recv * 100
            print(f"\n双通道负载分布: {ch0_ratio:.0f}%:{ch1_ratio:.0f}%, {ch0_recv}:{ch1_recv}")
        else:
            print("双通道负载分布: 无数据")

    def _merge_links_flow_stat(self):
        """合并两个数据通道的链路流量统计"""
        # 初始化合并后的links_flow_stat
        self.data_network.links_flow_stat = {}
        
        # 获取两个通道的links_flow_stat
        ch0_stats = self.data_network_ch0.links_flow_stat
        ch1_stats = self.data_network_ch1.links_flow_stat
        
        # 合并所有链路的统计数据
        all_links = set(ch0_stats.keys()) | set(ch1_stats.keys())
        
        for link in all_links:
            # 初始化合并后的链路统计
            merged_stat = {}
            
            # 从通道0获取数据（如果存在）
            ch0_data = ch0_stats.get(link, {})
            ch1_data = ch1_stats.get(link, {})
            
            # 合并数值型统计（相加）
            numeric_keys = ["ITag_count", "empty_count", "total_cycles"]
            for key in numeric_keys:
                merged_stat[key] = ch0_data.get(key, 0) + ch1_data.get(key, 0)
            
            # 合并字典型统计
            dict_keys = ["eject_attempts_h", "eject_attempts_v", "inject_attempts_h", "inject_attempts_v"]
            for key in dict_keys:
                merged_stat[key] = {}
                # 合并通道0的数据
                if key in ch0_data:
                    for sub_key, value in ch0_data[key].items():
                        merged_stat[key][sub_key] = value
                # 合并通道1的数据
                if key in ch1_data:
                    for sub_key, value in ch1_data[key].items():
                        if sub_key in merged_stat[key]:
                            merged_stat[key][sub_key] += value
                        else:
                            merged_stat[key][sub_key] = value
            
            self.data_network.links_flow_stat[link] = merged_stat

    def tag_move_all_networks(self):
        """重写tag_move方法以支持双通道数据网络"""
        self._tag_move(self.req_network)
        self._tag_move(self.rsp_network)
        self._tag_move(self.data_network_ch0)  # 处理通道0
        self._tag_move(self.data_network_ch1)  # 处理通道1
