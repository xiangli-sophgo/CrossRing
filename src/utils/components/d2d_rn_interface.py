"""
D2D_RN_Interface class for Die-to-Die communication.
Handles cross-die request initiation with AXI channel delays.
"""

from __future__ import annotations
import heapq
from collections import deque
from .ip_interface import IPInterface
from .flit import Flit
import logging


class D2D_RN_Interface(IPInterface):
    """
    Die间请求节点 - 发起跨Die请求
    继承自IPInterface，复用所有现有功能
    """
    
    def __init__(self, ip_type: str, ip_pos: int, config, req_network, rsp_network, data_network, node, routes, ip_id: int = None):
        # 调用父类初始化
        super().__init__(ip_type, ip_pos, config, req_network, rsp_network, data_network, node, routes, ip_id)
        
        # D2D特有属性
        self.die_id = getattr(config, 'DIE_ID', 0)  # 当前Die的ID
        self.cross_die_delay_queue = []  # 使用heapq管理的延迟队列 [(arrival_cycle, flit)]
        self.target_die_interfaces = {}  # 将由D2D_Model设置 {die_id: d2d_sn_interface}
        
        # 获取D2D延迟配置
        self.d2d_ar_latency = getattr(config, 'D2D_AR_LATENCY', 10)
        self.d2d_r_latency = getattr(config, 'D2D_R_LATENCY', 8)
        self.d2d_aw_latency = getattr(config, 'D2D_AW_LATENCY', 10)
        self.d2d_w_latency = getattr(config, 'D2D_W_LATENCY', 2)
        self.d2d_b_latency = getattr(config, 'D2D_B_LATENCY', 8)
        
        # 跨Die请求统计
        self.cross_die_requests_sent = 0
        self.cross_die_responses_received = 0
        
        logging.info(f"D2D_RN_Interface initialized at position {ip_pos} for Die {self.die_id}")
    
    def is_cross_die_request(self, flit: Flit) -> bool:
        """检查是否为跨Die请求"""
        target_die_id = getattr(flit, 'target_die_id', None)
        if target_die_id is None:
            # 如果没有target_die_id属性，默认为本地请求
            return False
        return target_die_id != self.die_id
    
    def handle_cross_die_request(self, flit: Flit):
        """
        处理跨Die请求 - 添加AR/AW延迟并发送到目标Die的D2D_SN
        """
        if not self.is_cross_die_request(flit):
            # 本地请求，走正常流程
            return False
        
        target_die_id = flit.target_die_id
        if target_die_id not in self.target_die_interfaces:
            logging.error(f"D2D_RN: No interface found for target Die {target_die_id}")
            return False
        
        # 根据请求类型选择延迟
        if hasattr(flit, 'req_type'):
            if flit.req_type == 'read':
                delay = self.d2d_ar_latency
            else:  # write
                delay = self.d2d_aw_latency
        else:
            # 默认使用AR延迟
            delay = self.d2d_ar_latency
        
        # 计算到达时间
        arrival_cycle = self.current_cycle + delay
        
        # 获取目标Die的D2D_SN接口
        target_d2d_sn = self.target_die_interfaces[target_die_id]
        
        # 调度跨Die接收
        target_d2d_sn.schedule_cross_die_receive(flit, arrival_cycle)
        
        self.cross_die_requests_sent += 1
        
        logging.debug(f"D2D_RN Die{self.die_id}: Sent cross-die request to Die{target_die_id}, arrival at cycle {arrival_cycle}")
        
        return True
    
    def handle_cross_die_response(self, flit: Flit):
        """
        处理跨Die响应 - 添加R/B延迟并发送回源Die
        """
        source_die_id = getattr(flit, 'source_die_id', None)
        if source_die_id is None or source_die_id == self.die_id:
            # 本地响应，走正常流程
            return False
        
        if source_die_id not in self.target_die_interfaces:
            logging.error(f"D2D_RN: No interface found for source Die {source_die_id}")
            return False
        
        # 根据响应类型选择延迟
        if hasattr(flit, 'rsp_type'):
            if flit.rsp_type == 'read_data':
                delay = self.d2d_r_latency
            else:  # write_response
                delay = self.d2d_b_latency
        else:
            # 默认使用R延迟
            delay = self.d2d_r_latency
        
        # 计算到达时间
        arrival_cycle = self.current_cycle + delay
        
        # 获取源Die的D2D_SN接口
        source_d2d_sn = self.target_die_interfaces[source_die_id]
        
        # 调度跨Die接收
        source_d2d_sn.schedule_cross_die_receive(flit, arrival_cycle)
        
        self.cross_die_responses_received += 1
        
        logging.debug(f"D2D_RN Die{self.die_id}: Sent cross-die response to Die{source_die_id}, arrival at cycle {arrival_cycle}")
        
        return True
    
    def process_inject_request(self, flit: Flit, network_type: str):
        """
        重写父类方法，拦截跨Die请求
        """
        if network_type == 'req' and self.is_cross_die_request(flit):
            # 处理跨Die请求
            if self.handle_cross_die_request(flit):
                return  # 已处理，不走正常网络流程
        
        # 非跨Die请求或其他网络类型，调用父类方法
        super().process_inject_request(flit, network_type)
    
    def get_statistics(self) -> dict:
        """获取D2D_RN统计信息"""
        stats = super().get_statistics()
        stats.update({
            'cross_die_requests_sent': self.cross_die_requests_sent,
            'cross_die_responses_received': self.cross_die_responses_received,
        })
        return stats