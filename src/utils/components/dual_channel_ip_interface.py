"""
DualChannelIPInterface class for NoC simulation.
Extends IPInterface class to support dual-channel data transmission.
"""

from .ip_interface import IPInterface
from .flit import Flit
from ..channel_selector import DefaultChannelSelector
from collections import deque
import logging


class DualChannelIPInterface(IPInterface):
    """双通道IP接口，支持数据双通道传输"""
    
    def __init__(self, ip_type, ip_pos, config, req_network, rsp_network, 
                 data_network, node, routes, channel_selector=None):
        # 首先调用父类初始化
        super().__init__(ip_type, ip_pos, config, req_network, rsp_network, 
                        data_network, node, routes)
        
        # 设置通道选择器
        self.channel_selector = channel_selector or DefaultChannelSelector("hash_based", data_network)
        
        # 修改data网络结构为双通道
        self._modify_data_network_structure()
        
        logging.info(f"DualChannelIPInterface created for {ip_type}_{ip_pos}")
    
    def _modify_data_network_structure(self):
        """将data网络结构修改为双通道"""
        l2h_depth = self.config.IP_L2H_FIFO_DEPTH
        h2l_h_depth = self.config.IP_H2L_H_FIFO_DEPTH
        h2l_l_depth = self.config.IP_H2L_L_FIFO_DEPTH
        
        # 替换原有data网络结构为双通道结构
        self.networks["data"] = {
            "network": self.data_network,
            "send_flits": self.data_network.send_flits,
            "ch0": {
                "inject_fifo": deque(),
                "l2h_fifo_pre": None,
                "h2l_fifo_h_pre": None,
                "h2l_fifo_l_pre": None,
                "l2h_fifo": deque(maxlen=l2h_depth),
                "h2l_fifo_h": deque(maxlen=h2l_h_depth),
                "h2l_fifo_l": deque(maxlen=h2l_l_depth),
            },
            "ch1": {
                "inject_fifo": deque(),
                "l2h_fifo_pre": None,
                "h2l_fifo_h_pre": None,
                "h2l_fifo_l_pre": None,
                "l2h_fifo": deque(maxlen=l2h_depth),
                "h2l_fifo_h": deque(maxlen=h2l_h_depth),
                "h2l_fifo_l": deque(maxlen=h2l_l_depth),
            }
        }
    
    def enqueue(self, flit: Flit, network_type: str, retry=False):
        """重写enqueue方法，data包使用双通道"""
        if network_type == "data":
            return self._enqueue_data_dual_channel(flit, retry)
        else:
            # req和rsp使用原有单通道逻辑
            return super().enqueue(flit, network_type, retry)
    
    def _enqueue_data_dual_channel(self, flit: Flit, retry=False):
        """数据包双通道入队"""
        # 选择数据通道
        flit.data_channel_id = self.channel_selector.select_channel(flit)
        channel_key = f"ch{flit.data_channel_id}"
        
        if retry:
            self.networks["data"][channel_key]["inject_fifo"].appendleft(flit)
        else:
            flit.cmd_entry_cake0_cycle = self.current_cycle
            self.networks["data"][channel_key]["inject_fifo"].append(flit)
        
        flit.flit_position = f"IP_inject_data_ch{flit.data_channel_id}"
        
        # 确保send_flits存在该packet_id的列表
        if flit.packet_id not in self.networks["data"]["send_flits"]:
            self.networks["data"]["send_flits"][flit.packet_id] = []
        self.networks["data"]["send_flits"][flit.packet_id].append(flit)
        
        return True
    
    def inject_to_l2h_pre(self, network_type):
        """重写inject方法，支持数据双通道"""
        if network_type == "data":
            # 数据网络使用双通道处理
            for channel_id in [0, 1]:
                self._inject_to_l2h_pre_data_channel(channel_id)
        else:
            # req和rsp使用原有逻辑
            super().inject_to_l2h_pre(network_type)
    
    def _inject_to_l2h_pre_data_channel(self, channel_id):
        """数据包双通道L2H处理"""
        channel_key = f"ch{channel_id}"
        net_info = self.networks["data"][channel_key]
        
        if (not net_info["inject_fifo"] or 
            len(net_info["l2h_fifo"]) >= net_info["l2h_fifo"].maxlen or 
            net_info["l2h_fifo_pre"] is not None):
            return
            
        flit = net_info["inject_fifo"][0]
        current_cycle = getattr(self, "current_cycle", 0)
        
        # 检查发送时间
        if hasattr(flit, "departure_cycle") and flit.departure_cycle > current_cycle:
            return
            
        # 带宽控制
        if self.tx_token_bucket:
            self.tx_token_bucket.refill(current_cycle)
            if not self.tx_token_bucket.consume():
                return
                
        flit = net_info["inject_fifo"].popleft()
        flit.flit_position = f"L2H_data_ch{channel_id}"
        flit.start_inject = True
        net_info["l2h_fifo_pre"] = flit
    
    def l2h_to_IQ_channel_buffer(self, network_type):
        """重写L2H到IQ方法，支持数据双通道"""
        if network_type == "data":
            # 数据网络使用双通道处理
            for channel_id in [0, 1]:
                self._l2h_to_IQ_channel_buffer_data_channel(channel_id)
        else:
            # req和rsp使用原有逻辑
            super().l2h_to_IQ_channel_buffer(network_type)
    
    def _l2h_to_IQ_channel_buffer_data_channel(self, channel_id):
        """数据包双通道L2H到IQ处理"""
        channel_key = f"ch{channel_id}"
        net_info = self.networks["data"][channel_key]
        
        if not net_info["l2h_fifo"]:
            return
            
        # 使用双通道数据IQ缓冲区
        iq_buffer = self.data_network.get_iq_channel_buffer(channel_id)[self.ip_type][self.ip_pos]
        iq_buffer_pre = self.data_network.get_iq_channel_buffer_pre(channel_id)[self.ip_type][self.ip_pos]
        
        if len(iq_buffer) >= getattr(self.config, "IQ_CH_FIFO_DEPTH", 8) or iq_buffer_pre is not None:
            return
            
        flit = net_info["l2h_fifo"].popleft()
        flit.flit_position = f"IQ_CH_data_ch{channel_id}"
        
        # 将flit放入预缓冲区
        self.data_network.get_iq_channel_buffer_pre(channel_id)[self.ip_type][self.ip_pos] = flit
        
        # 更新时间戳
        if flit.req_type == "read" and flit.flit_id == 0:
            flit.data_entry_noc_from_cake1_cycle = self.current_cycle
        elif flit.req_type == "write" and flit.flit_id == 0:
            flit.data_entry_noc_from_cake0_cycle = self.current_cycle
    
    def EQ_channel_buffer_to_h2l_pre(self, network_type):
        """重写EQ到H2L方法，支持数据双通道"""
        if network_type == "data":
            # 数据网络使用双通道处理
            for channel_id in [0, 1]:
                self._EQ_channel_buffer_to_h2l_pre_data_channel(channel_id)
        else:
            # req和rsp使用原有逻辑
            super().EQ_channel_buffer_to_h2l_pre(network_type)
    
    def _EQ_channel_buffer_to_h2l_pre_data_channel(self, channel_id):
        """数据包双通道EQ到H2L处理"""
        channel_key = f"ch{channel_id}"
        net_info = self.networks["data"][channel_key]
        
        if net_info["h2l_fifo_h_pre"] is not None:
            return
            
        try:
            # 确定弹出位置索引
            if hasattr(self.config, "RING_NUM_NODE") and self.config.RING_NUM_NODE > 0:
                pos_index = self.ip_pos
            else:
                pos_index = self.ip_pos - self.config.NUM_COL
                
            eq_buf = self.data_network.get_eq_channel_buffer(channel_id)[self.ip_type][pos_index]
            if not eq_buf:
                return
                
            if len(net_info["h2l_fifo_h"]) >= net_info["h2l_fifo_h"].maxlen:
                return
                
            flit = eq_buf.popleft()
            flit.is_arrive = True
            flit.flit_position = f"H2L_H_data_ch{channel_id}"
            net_info["h2l_fifo_h_pre"] = flit
            
            # 更新到达统计
            if flit.packet_id not in self.data_network.arrive_flits:
                self.data_network.arrive_flits[flit.packet_id] = []
            self.data_network.arrive_flits[flit.packet_id].append(flit)
            self.data_network.recv_flits_num += 1
            
        except (KeyError, AttributeError) as e:
            logging.warning(f"Data EQ to h2l_h transfer failed for ch{channel_id}: {e}")
    
    def move_pre_to_fifo(self):
        """重写pre到fifo的移动，支持数据双通道"""
        # 处理req和rsp网络的pre移动
        for net_type in ["req", "rsp"]:
            if net_type not in self.networks:
                continue
                
            net_info = self.networks[net_type]
            net = net_info["network"]
            
            # l2h_fifo_pre → l2h_fifo
            if net_info["l2h_fifo_pre"] is not None and len(net_info["l2h_fifo"]) < net_info["l2h_fifo"].maxlen:
                net_info["l2h_fifo"].append(net_info["l2h_fifo_pre"])
                net_info["l2h_fifo_pre"] = None
            
            # IQ buffer pre移动
            if net.IQ_channel_buffer_pre[self.ip_type][self.ip_pos] is not None:
                if len(net.IQ_channel_buffer[self.ip_type][self.ip_pos]) < getattr(self.config, "IQ_CH_FIFO_DEPTH", 8):
                    net.IQ_channel_buffer[self.ip_type][self.ip_pos].append(net.IQ_channel_buffer_pre[self.ip_type][self.ip_pos])
                    net.IQ_channel_buffer_pre[self.ip_type][self.ip_pos] = None
            
            # h2l_fifo_h_pre → h2l_fifo_h
            if net_info["h2l_fifo_h_pre"] is not None and len(net_info["h2l_fifo_h"]) < net_info["h2l_fifo_h"].maxlen:
                net_info["h2l_fifo_h"].append(net_info["h2l_fifo_h_pre"])
                net_info["h2l_fifo_h_pre"] = None
            
            # h2l_fifo_l_pre → h2l_fifo_l
            if net_info["h2l_fifo_l_pre"] is not None and len(net_info["h2l_fifo_l"]) < net_info["h2l_fifo_l"].maxlen:
                net_info["h2l_fifo_l"].append(net_info["h2l_fifo_l_pre"])
                net_info["h2l_fifo_l_pre"] = None
        
        # 处理数据双通道的pre移动
        for channel_id in [0, 1]:
            channel_key = f"ch{channel_id}"
            net_info = self.networks["data"][channel_key]
            
            # l2h_fifo_pre → l2h_fifo
            if net_info["l2h_fifo_pre"] is not None and len(net_info["l2h_fifo"]) < net_info["l2h_fifo"].maxlen:
                net_info["l2h_fifo"].append(net_info["l2h_fifo_pre"])
                net_info["l2h_fifo_pre"] = None
            
            # IQ buffer pre移动
            iq_buffer = self.data_network.get_iq_channel_buffer(channel_id)[self.ip_type][self.ip_pos]
            iq_buffer_pre_ref = self.data_network.get_iq_channel_buffer_pre(channel_id)
            if iq_buffer_pre_ref[self.ip_type][self.ip_pos] is not None:
                if len(iq_buffer) < getattr(self.config, "IQ_CH_FIFO_DEPTH", 8):
                    iq_buffer.append(iq_buffer_pre_ref[self.ip_type][self.ip_pos])
                    iq_buffer_pre_ref[self.ip_type][self.ip_pos] = None
            
            # h2l_fifo_h_pre → h2l_fifo_h
            if net_info["h2l_fifo_h_pre"] is not None and len(net_info["h2l_fifo_h"]) < net_info["h2l_fifo_h"].maxlen:
                net_info["h2l_fifo_h"].append(net_info["h2l_fifo_h_pre"])
                net_info["h2l_fifo_h_pre"] = None
            
            # h2l_fifo_l_pre → h2l_fifo_l
            if net_info["h2l_fifo_l_pre"] is not None and len(net_info["h2l_fifo_l"]) < net_info["h2l_fifo_l"].maxlen:
                net_info["h2l_fifo_l"].append(net_info["h2l_fifo_l_pre"])
                net_info["h2l_fifo_l_pre"] = None
    
    def inject_step(self, cycle):
        """重写注入步骤，支持数据双通道"""
        self.current_cycle = cycle
        cycle_mod = cycle % self.config.NETWORK_FREQUENCY
        
        # 1GHz操作（每个网络周期执行一次）
        if cycle_mod == 0:
            # 对req和rsp网络执行原有逻辑
            for net_type in ["req", "rsp"]:
                self.inject_to_l2h_pre(net_type)
            
            # 对data网络执行双通道处理
            self.inject_to_l2h_pre("data")
        
        # 2GHz操作（每半个网络周期执行一次）
        for net_type in ["req", "rsp"]:
            self.l2h_to_IQ_channel_buffer(net_type)
        
        # 数据双通道处理
        self.l2h_to_IQ_channel_buffer("data")
    
    def eject_step(self, cycle):
        """重写弹出步骤，支持数据双通道"""
        self.current_cycle = cycle
        cycle_mod = cycle % self.config.NETWORK_FREQUENCY
        
        # 收集弹出的flits
        ejected_flits = []
        
        # 2GHz操作（每半个网络周期执行一次）
        for net_type in ["req", "rsp"]:
            self.EQ_channel_buffer_to_h2l_pre(net_type)
            self.h2l_h_to_h2l_l_pre(net_type)
        
        # 数据双通道EQ到H2L处理
        self.EQ_channel_buffer_to_h2l_pre("data")
        
        # 数据双通道H2L处理
        for channel_id in [0, 1]:
            self._h2l_h_to_h2l_l_pre_data_channel(channel_id)
        
        # 1GHz操作（每个网络周期执行一次）
        if cycle_mod == 0:
            # 对req和rsp网络执行原有h2l_to_eject_fifo
            for net_type in ["req", "rsp"]:
                flit = self.h2l_to_eject_fifo(net_type)
                if flit:
                    ejected_flits.append(flit)
            
            # 对数据双通道执行弹出处理
            for channel_id in [0, 1]:
                flit = self._h2l_to_eject_fifo_data_channel(channel_id)
                if flit:
                    ejected_flits.append(flit)
        
        return ejected_flits
    
    def _h2l_h_to_h2l_l_pre_data_channel(self, channel_id):
        """数据双通道H2L_H到H2L_L预缓冲"""
        channel_key = f"ch{channel_id}"
        net_info = self.networks["data"][channel_key]
        
        if net_info["h2l_fifo_l_pre"] is not None:
            return  # L级预缓冲已占用
            
        if not net_info["h2l_fifo_h"]:
            return  # H级FIFO为空
            
        # 检查L级FIFO是否有空间
        if len(net_info["h2l_fifo_l"]) >= net_info["h2l_fifo_l"].maxlen:
            return
            
        # 从H级FIFO传输到L级预缓冲
        flit = net_info["h2l_fifo_h"].popleft()
        flit.flit_position = f"H2L_L_data_ch{channel_id}"
        net_info["h2l_fifo_l_pre"] = flit
    
    def _h2l_to_eject_fifo_data_channel(self, channel_id):
        """数据双通道H2L到eject处理"""
        channel_key = f"ch{channel_id}"
        net_info = self.networks["data"][channel_key]
        
        if not net_info["h2l_fifo_l"]:
            return None
            
        current_cycle = getattr(self, "current_cycle", 0)
        
        # 带宽控制
        if self.rx_token_bucket:
            self.rx_token_bucket.refill(current_cycle)
            if not self.rx_token_bucket.consume():
                return None
                
        flit = net_info["h2l_fifo_l"].popleft()
        flit.flit_position = f"IP_eject_data_ch{channel_id}"
        flit.is_finish = True
        
        # 处理接收到的数据
        self._handle_received_data(flit)
        
        return flit