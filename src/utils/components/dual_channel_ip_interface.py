"""
DualChannelIPInterface class for NoC simulation.
Extends IPInterface class to support dual-channel data transmission.
简化版本：使用两个独立的Network实例，在IP接口层进行通道选择和合并。
"""

from .ip_interface import IPInterface
from .flit import Flit
from ..channel_selector import DefaultChannelSelector
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
            self.channel_selector = DefaultChannelSelector("ip_id_based", self.data_network_ch0, self.ip_id)
        else:
            self.channel_selector = channel_selector

        # 2to1轮询仲裁计数器
        self.eq_round_robin_counter = 0

        logging.info(f"DualChannelIPInterface created for {ip_type}_{ip_pos}")

    # def move_pre_to_fifo(self):
    #     """重写move_pre_to_fifo方法，特殊处理数据网络的双通道"""
    #     # 先调用父类方法处理req和rsp网络
    #     super().move_pre_to_fifo()

    #     # 特殊处理data网络的双通道
    #     # 对于双通道数据网络，IQ_channel_buffer_pre的移动在_l2h_to_IQ_with_channel_selection方法中处理
    #     # 这里只需要处理本地FIFO的移动
    #     net_info = self.data_network_config  # 使用父类的data网络配置

    #     # 处理l2h_fifo_pre → l2h_fifo（与父类相同）
    #     if net_info["l2h_fifo_pre"] is not None and len(net_info["l2h_fifo"]) < net_info["l2h_fifo"].maxlen:
    #         net_info["l2h_fifo"].append(net_info["l2h_fifo_pre"])
    #         net_info["l2h_fifo_pre"] = None

    #     # 处理h2l_fifo_h_pre → h2l_fifo_h（与父类相同）
    #     if net_info["h2l_fifo_h_pre"] is not None and len(net_info["h2l_fifo_h"]) < net_info["h2l_fifo_h"].maxlen:
    #         net_info["h2l_fifo_h"].append(net_info["h2l_fifo_h_pre"])
    #         net_info["h2l_fifo_h_pre"] = None

    #     # 处理h2l_fifo_l_pre → h2l_fifo_l（与父类相同）
    #     if net_info["h2l_fifo_l_pre"] is not None and len(net_info["h2l_fifo_l"]) < net_info["h2l_fifo_l"].maxlen:
    #         net_info["h2l_fifo_l"].append(net_info["h2l_fifo_l_pre"])
    #         net_info["h2l_fifo_l_pre"] = None

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
        flit.data_channel_id = self.channel_selector.select_channel(flit)

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

        # 轮询选择通道
        selected_flit = None
        attempts = 0

        while selected_flit is None and attempts < 2:
            current_channel = self.eq_round_robin_counter % 2

            # 选择对应的网络
            target_network = self.data_network_ch0 if current_channel == 0 else self.data_network_ch1

            # 尝试从当前通道的EQ buffer获取
            try:
                eq_buf = target_network.EQ_channel_buffer[self.ip_type][pos_index]
                if eq_buf:
                    selected_flit = eq_buf.popleft()
                    # 记录来源通道，用于统计
                    selected_flit.ejected_from_channel = current_channel
                    # 确保data_channel_id与ejected_from_channel一致
                    if not hasattr(selected_flit, 'data_channel_id'):
                        selected_flit.data_channel_id = current_channel
            except (KeyError, AttributeError) as e:
                pass  # 该通道没有数据

            if selected_flit:
                # 成功选中，更新轮询计数器
                self.eq_round_robin_counter = (self.eq_round_robin_counter + 1) % 2
                break

            # 当前通道没有数据，尝试另一个通道
            self.eq_round_robin_counter = (self.eq_round_robin_counter + 1) % 2
            attempts += 1

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
