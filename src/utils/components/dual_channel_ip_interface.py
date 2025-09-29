"""
DualChannelIPInterface class for NoC simulation.
Extends IPInterface class to support dual-channel data transmission.
简化版本：使用两个独立的Network实例，在IP接口层进行通道选择和合并。
"""

from .ip_interface import IPInterface
from .flit import Flit
from ..channel_selector import DefaultChannelSelector
from ..arbitration import create_arbiter_from_config
from collections import deque
import logging


class DualChannelIPInterface(IPInterface):
    """双通道IP接口，支持数据双通道传输 - 简化版本

    架构说明：
    - 发送路径：inject_fifo → l2h_fifo(单通道) → 通道选择 → 选择网络的IQ_buffer
    - 接收路径：两个网络的EQ_buffer → 2to1轮询仲裁 → h2l_fifo_h(单通道) → h2l_fifo_l(单通道)
    """

    def __init__(self, ip_type, ip_pos, config, req_network, rsp_network, data_network_ch0, data_network_ch1, node, routes, channel_selector=None, ip_id=None):
        # 存储双通道网络引用
        self.data_network_ch0 = data_network_ch0
        self.data_network_ch1 = data_network_ch1

        # 首先调用父类初始化（使用ch0作为默认data_network）
        super().__init__(ip_type, ip_pos, config, req_network, rsp_network, data_network_ch0, node, routes, ip_id)

        # 保存对父类data网络配置的引用，以便在双通道方法中使用
        self.data_network_config = self.networks["data"]

        # 设置通道选择器，传递ip_id信息
        # 如果没有提供channel_selector，则使用从配置中获取的策略
        if channel_selector is None:
            # 在DualChannelBaseModel中会传递正确的channel_selector
            self.channel_selector = DefaultChannelSelector("ip_id_based", self.data_network_ch0)
        else:
            self.channel_selector = channel_selector

        # 创建双通道仲裁器（基于配置）
        arbitration_config = getattr(config, 'arbitration', {})
        dual_channel_arbiter_config = arbitration_config.get('dual_channel', {'type': 'round_robin'})
        self.dual_channel_arbiter = create_arbiter_from_config(dual_channel_arbiter_config)


    def enqueue(self, flit, network_type, retry=False):
        """重写enqueue方法，对data网络进行通道区分存储"""
        if network_type == "data":
            # 数据网络需要特殊处理
            self._enqueue_data_with_channel_selection(flit, retry)
        else:
            # req和rsp使用原有逻辑
            super().enqueue(flit, network_type, retry)

    def _enqueue_data_with_channel_selection(self, flit, retry=False):
        """数据网络的enqueue，根据通道选择存储到对应网络"""
        # 选择通道（如果还没有选择的话）
        flit.data_channel_id = self.channel_selector.select_channel(flit, self.ip_id)

        channel_id = flit.data_channel_id

        # 选择对应的网络
        target_network = self.data_network_ch0 if channel_id == 0 else self.data_network_ch1

        # 将flit添加到父类data网络配置的inject_fifo（因为inject_fifo在IPInterface中管理）
        net_info = self.data_network_config  # 使用父类的data网络配置
        if retry:
            net_info["inject_fifo"].appendleft(flit)
        else:
            net_info["inject_fifo"].append(flit)

        flit.flit_position = "IP_inject"

        # 将flit添加到对应网络的send_flits进行追踪
        if flit.packet_id not in target_network.send_flits:
            target_network.send_flits[flit.packet_id] = []
        target_network.send_flits[flit.packet_id].append(flit)

    def l2h_to_IQ_channel_buffer(self, network_type):
        """重写L2H到IQ方法，在此处进行通道选择"""
        if network_type == "data":
            # 数据网络在此处选择通道
            self._l2h_to_IQ_with_channel_selection()
        else:
            # req和rsp使用原有逻辑
            super().l2h_to_IQ_channel_buffer(network_type)

    def _l2h_to_IQ_with_channel_selection(self):
        """从单通道L2H FIFO取出flit，选择通道后放入对应网络的IQ buffer"""
        net_info = self.data_network_config  # 使用父类的data网络配置

        if not net_info["l2h_fifo"]:
            return

        flit = net_info["l2h_fifo"][0]  # 查看队首元素

        # 选择数据通道
        # if not hasattr(flit, 'data_channel_id'):
        flit.data_channel_id = self.channel_selector.select_channel(flit, self.ip_id)

        channel_id = flit.data_channel_id

        # 选择对应的网络
        target_network = self.data_network_ch0 if channel_id == 0 else self.data_network_ch1

        # 获取对应网络的IQ缓冲区
        try:
            iq_buffer = target_network.IQ_channel_buffer[self.ip_type][self.ip_pos]
            iq_buffer_pre = target_network.IQ_channel_buffer_pre[self.ip_type][self.ip_pos]
        except (KeyError, AttributeError) as e:
            return

        iq_depth = getattr(self.config, "IQ_CH_FIFO_DEPTH", 8)
        if len(iq_buffer) >= iq_depth or iq_buffer_pre is not None:
            return  # 目标通道满，等待

        # 可以传输，从l2h_fifo取出
        flit = net_info["l2h_fifo"].popleft()
        flit.flit_position = f"IQ_CH_ch{channel_id}"

        # 将flit放入对应网络的预缓冲区
        target_network.IQ_channel_buffer_pre[self.ip_type][self.ip_pos] = flit

        # 更新时间戳
        if flit.req_type == "read" and flit.flit_id == 0:
            flit.data_entry_noc_from_cake1_cycle = self.current_cycle
        elif flit.req_type == "write" and flit.flit_id == 0:
            flit.data_entry_noc_from_cake0_cycle = self.current_cycle

    def EQ_channel_buffer_to_h2l_pre(self, network_type):
        """重写EQ到H2L方法，实现2to1轮询仲裁"""
        if network_type == "data":
            # 数据网络使用2to1轮询仲裁
            self._eq_dual_channel_to_h2l_arbitration()
        else:
            # req和rsp使用原有逻辑
            super().EQ_channel_buffer_to_h2l_pre(network_type)

    def _eq_dual_channel_to_h2l_arbitration(self):
        """双通道EQ到单通道H2L的2to1轮询仲裁"""
        net_info = self.data_network_config  # 使用父类的data网络配置

        # 检查h2l_fifo_h是否可以接收
        if net_info["h2l_fifo_h_pre"] is not None:
            return  # 预缓冲已占用

        if len(net_info["h2l_fifo_h"]) >= net_info["h2l_fifo_h"].maxlen:
            return  # H2L FIFO已满

        # 确定弹出位置索引
        if hasattr(self.config, "RING_NUM_NODE") and self.config.RING_NUM_NODE > 0:
            pos_index = self.ip_pos
        else:
            pos_index = self.ip_pos - self.config.NUM_COL

        # 使用统一仲裁器选择通道
        channel_networks = [self.data_network_ch0, self.data_network_ch1]

        def is_channel_valid(network):
            """检查通道是否有待处理的数据"""
            try:
                eq_buf = network.EQ_channel_buffer[self.ip_type][pos_index]
                return len(eq_buf) > 0
            except (KeyError, AttributeError):
                return False

        # 使用仲裁器选择通道
        selected_network, channel_idx = self.dual_channel_arbiter.select(
            candidates=channel_networks,
            queue_id=f"dual_channel_{self.ip_type}_{self.ip_pos}",
            is_valid=is_channel_valid
        )

        selected_flit = None
        if selected_network is not None:
            # 从选中的通道获取flit
            try:
                eq_buf = selected_network.EQ_channel_buffer[self.ip_type][pos_index]
                if eq_buf:
                    selected_flit = eq_buf.popleft()
                    # 记录来源通道，用于统计
                    selected_flit.ejected_from_channel = channel_idx
                    # 确保data_channel_id与ejected_from_channel一致
                    if not hasattr(selected_flit, "data_channel_id"):
                        selected_flit.data_channel_id = channel_idx
            except (KeyError, AttributeError):
                pass  # 该通道没有数据

        # 如果选中了flit，放入h2l_fifo_h_pre
        if selected_flit:
            selected_flit.is_arrive = True
            # 确保ejected_from_channel属性存在
            if hasattr(selected_flit, "ejected_from_channel"):
                selected_flit.flit_position = f"H2L_H_ch{selected_flit.ejected_from_channel}"
            else:
                selected_flit.flit_position = "H2L_H_ch0"  # 默认通道0
            net_info["h2l_fifo_h_pre"] = selected_flit

    def h2l_l_to_eject_fifo(self, network_type):
        """重写h2l_l到eject的处理，确保数据网络的统计正确"""
        if network_type == "data":
            net_info = self.data_network_config  # 使用父类的data网络配置

            if not net_info["h2l_fifo_l"]:
                return None

            current_cycle = getattr(self, "current_cycle", 0)

            # 带宽控制
            if self.rx_token_bucket:
                self.rx_token_bucket.refill(current_cycle)
                if not self.rx_token_bucket.consume():
                    return None

            flit = net_info["h2l_fifo_l"].popleft()

            # 确定使用哪个通道ID进行统计
            channel_for_stats = None
            if hasattr(flit, "data_channel_id"):
                # 优先使用原始通道ID
                channel_for_stats = flit.data_channel_id
            elif hasattr(flit, "ejected_from_channel"):
                # 如果没有原始通道ID，使用弹出通道
                channel_for_stats = flit.ejected_from_channel

            if channel_for_stats is not None:
                flit.flit_position = f"IP_eject_ch{channel_for_stats}"
                # 更新对应网络的统计
                target_network = self.data_network_ch0 if channel_for_stats == 0 else self.data_network_ch1
                if flit.packet_id not in target_network.arrive_flits:
                    target_network.arrive_flits[flit.packet_id] = []
                target_network.arrive_flits[flit.packet_id].append(flit)
                target_network.recv_flits_num += 1
            else:
                flit.flit_position = "IP_eject"
                # 默认添加到ch0统计
                if flit.packet_id not in self.data_network_ch0.arrive_flits:
                    self.data_network_ch0.arrive_flits[flit.packet_id] = []
                self.data_network_ch0.arrive_flits[flit.packet_id].append(flit)
                self.data_network_ch0.recv_flits_num += 1

            flit.is_finish = True

            # 处理接收到的数据
            self._handle_received_data(flit)

            return flit
        else:
            # req和rsp使用父类逻辑
            return super().h2l_l_to_eject_fifo(network_type)

    def _handle_received_data(self, flit: Flit):
        """重写处理接收到的数据方法，支持双通道时间戳同步"""
        flit.arrival_cycle = getattr(self, "current_cycle", 0)
        self.data_wait_cycles_h += flit.wait_cycle_h
        self.data_wait_cycles_v += flit.wait_cycle_v
        self.data_cir_h_num += flit.eject_attempts_h
        self.data_cir_v_num += flit.eject_attempts_v

        if flit.req_type == "read":
            # 读数据到达RN端，需要收集到data buffer中
            self.node.rn_rdb[self.ip_type][self.ip_pos][flit.packet_id].append(flit)
            # 检查是否收集完整个burst
            if len(self.node.rn_rdb[self.ip_type][self.ip_pos][flit.packet_id]) == flit.burst_length:
                req = next((req for req in self.node.rn_tracker["read"][self.ip_type][self.ip_pos] if req.packet_id == flit.packet_id), None)
                if req:
                    # 立即释放tracker和更新计数
                    self.node.rn_tracker["read"][self.ip_type][self.ip_pos].remove(req)
                    self.node.rn_tracker_count["read"][self.ip_type][self.ip_pos]["count"] += 1
                    self.node.rn_tracker_pointer["read"][self.ip_type][self.ip_pos] -= 1
                    self.node.rn_rdb_count[self.ip_type][self.ip_pos]["count"] += req.burst_length

                    # 双通道时间戳同步：处理分散在两个通道的flit
                    self._sync_dual_channel_timestamps(flit.packet_id, req, "read", None)

                    # 清理data buffer（数据已经收集完成）
                    self.node.rn_rdb[self.ip_type][self.ip_pos].pop(flit.packet_id)

        elif flit.req_type == "write":
            # 写数据到达SN端，需要收集到data buffer中
            self.node.sn_wdb[self.ip_type][self.ip_pos][flit.packet_id].append(flit)
            # 检查是否收集完整个burst
            if len(self.node.sn_wdb[self.ip_type][self.ip_pos][flit.packet_id]) == flit.burst_length:
                req = next((req for req in self.node.sn_tracker[self.ip_type][self.ip_pos] if req.packet_id == flit.packet_id), None)
                if req:
                    # 设置tracker延迟释放时间
                    release_time = self.current_cycle + self.config.SN_TRACKER_RELEASE_LATENCY

                    # 初始化释放时间字典（如果不存在）
                    if not hasattr(self.node, "sn_tracker_release_time"):
                        from collections import defaultdict
                        self.node.sn_tracker_release_time = defaultdict(list)

                    # 双通道时间戳同步：处理分散在两个通道的flit
                    self._sync_dual_channel_timestamps(flit.packet_id, req, "write", release_time)

                    # 清理data buffer（数据已经收集完成）
                    self.node.sn_wdb[self.ip_type][self.ip_pos].pop(flit.packet_id)

                    # 添加到延迟释放队列
                    self.node.sn_tracker_release_time[release_time].append((self.ip_type, self.ip_pos, req))

    def _sync_dual_channel_timestamps(self, packet_id, req, req_type, release_time=None):
        """同步双通道中同一packet的flit时间戳"""
        complete_cycle = self.current_cycle
        
        # 收集两个通道中的所有flit
        all_flits = []
        
        # 从通道0收集flit
        if packet_id in self.data_network_ch0.send_flits:
            all_flits.extend(self.data_network_ch0.send_flits[packet_id])
            
        # 从通道1收集flit
        if packet_id in self.data_network_ch1.send_flits:
            all_flits.extend(self.data_network_ch1.send_flits[packet_id])
            
        if not all_flits:
            return
            
        # 找到第一个flit（用于延迟计算）
        first_flit = min(all_flits, key=lambda f: f.flit_id)
        
        # 为所有flit设置时间戳和计算延迟
        for f in all_flits:
            if req_type == "read":
                f.leave_db_cycle = self.current_cycle
            else:  # write
                f.leave_db_cycle = release_time if release_time else self.current_cycle
                
            f.sync_latency_record(req)
            # 为所有flit设置receive时间戳，确保后续处理能获得正确值
            f.data_received_complete_cycle = complete_cycle
            
            # 计算延迟
            if req_type == "read":
                f.cmd_latency = f.cmd_received_by_cake1_cycle - f.cmd_entry_noc_from_cake0_cycle
                f.data_latency = complete_cycle - first_flit.data_entry_noc_from_cake1_cycle
                f.transaction_latency = complete_cycle - f.cmd_entry_cake0_cycle
            else:  # write
                f.cmd_latency = f.cmd_received_by_cake0_cycle - f.cmd_entry_noc_from_cake0_cycle
                f.data_latency = complete_cycle - first_flit.data_entry_noc_from_cake0_cycle
                f.transaction_latency = complete_cycle + self.config.SN_TRACKER_RELEASE_LATENCY - f.cmd_entry_cake0_cycle
