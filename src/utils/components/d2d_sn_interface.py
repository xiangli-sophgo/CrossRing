"""
D2D_SN_Interface class for Die-to-Die communication.
Handles cross-die request reception and forwarding within the die.
"""

from __future__ import annotations
import heapq
from collections import deque
from .ip_interface import IPInterface
from .flit import Flit
import logging


class D2D_SN_Interface(IPInterface):
    """
    Die间响应节点 - 接收跨Die请求并转发到Die内目标节点
    继承自IPInterface，复用所有现有功能
    """
    
    def __init__(self, ip_type: str, ip_pos: int, config, req_network, rsp_network, data_network, node, routes, ip_id: int = None):
        # 调用父类初始化
        super().__init__(ip_type, ip_pos, config, req_network, rsp_network, data_network, node, routes, ip_id)
        
        # D2D特有属性
        self.die_id = getattr(config, 'DIE_ID', 0)  # 当前Die的ID
        self.cross_die_receive_queue = []  # 使用heapq管理的接收队列 [(arrival_cycle, flit)]
        
        # 获取D2D延迟配置
        self.d2d_ar_latency = getattr(config, 'D2D_AR_LATENCY', 10)
        self.d2d_r_latency = getattr(config, 'D2D_R_LATENCY', 8)
        self.d2d_aw_latency = getattr(config, 'D2D_AW_LATENCY', 10)
        self.d2d_w_latency = getattr(config, 'D2D_W_LATENCY', 2)
        self.d2d_b_latency = getattr(config, 'D2D_B_LATENCY', 8)
        
        # 跨Die请求统计
        self.cross_die_requests_received = 0
        self.cross_die_requests_forwarded = 0
        self.cross_die_responses_sent = 0
        
        logging.info(f"D2D_SN_Interface initialized at position {ip_pos} for Die {self.die_id}")
    
    def schedule_cross_die_receive(self, flit: Flit, arrival_cycle: int):
        """
        调度跨Die接收 - 由对方Die的D2D_RN调用
        """
        heapq.heappush(self.cross_die_receive_queue, (arrival_cycle, flit))
        self.cross_die_requests_received += 1
        
        logging.debug(f"D2D_SN Die{self.die_id}: Scheduled cross-die receive at cycle {arrival_cycle}")
    
    def process_cross_die_receives(self):
        """
        处理到期的跨Die接收 - 在每个周期调用
        """
        while self.cross_die_receive_queue and self.cross_die_receive_queue[0][0] <= self.current_cycle:
            arrival_cycle, flit = heapq.heappop(self.cross_die_receive_queue)
            self.handle_received_cross_die_flit(flit)
    
    def handle_received_cross_die_flit(self, flit: Flit):
        """
        处理接收到的跨Die flit
        根据flit类型进行相应处理
        """
        logging.debug(f"D2D_SN Die{self.die_id}: Processing received cross-die flit at cycle {self.current_cycle}")
        
        # 检查是否为读请求
        if hasattr(flit, 'req_type'):
            if flit.req_type == 'read':
                # 读请求：转发到Die内目标SN节点
                self.forward_read_request_to_local_sn(flit)
            elif flit.req_type == 'write':
                # 写请求：转发到Die内目标SN节点
                self.forward_write_request_to_local_sn(flit)
        elif hasattr(flit, 'rsp_type'):
            # 响应：转发回Die内原始请求节点
            self.forward_response_to_local_rn(flit)
        else:
            logging.warning(f"D2D_SN: Unknown flit type received")
    
    def forward_read_request_to_local_sn(self, flit: Flit):
        """
        将跨Die读请求转发到本地目标SN节点
        """
        # 更新flit的源和目标信息
        # 保存原始信息供响应时使用
        flit.original_source_die_id = flit.source_die_id
        flit.original_source_node_id = flit.source_node_id
        
        # 设置新的源为D2D_SN节点
        flit.source = self.ip_pos
        flit.destination = flit.target_node_id  # Die内目标节点
        
        # 通过请求网络发送
        self.networks['req']['inject_fifo'].append(flit)
        self.cross_die_requests_forwarded += 1
        
        logging.debug(f"D2D_SN Die{self.die_id}: Forwarded read request to local node {flit.destination}")
    
    def forward_write_request_to_local_sn(self, flit: Flit):
        """
        将跨Die写请求转发到本地目标SN节点
        """
        # 类似读请求处理
        flit.original_source_die_id = flit.source_die_id
        flit.original_source_node_id = flit.source_node_id
        
        flit.source = self.ip_pos
        flit.destination = flit.target_node_id
        
        # 通过请求网络发送
        self.networks['req']['inject_fifo'].append(flit)
        self.cross_die_requests_forwarded += 1
        
        logging.debug(f"D2D_SN Die{self.die_id}: Forwarded write request to local node {flit.destination}")
    
    def forward_response_to_local_rn(self, flit: Flit):
        """
        将跨Die响应转发回本地原始请求节点
        """
        # 恢复原始目标信息
        if hasattr(flit, 'original_source_node_id'):
            flit.destination = flit.original_source_node_id
        else:
            flit.destination = flit.source_node_id
        
        flit.source = self.ip_pos
        
        # 根据响应类型选择网络
        if hasattr(flit, 'rsp_type') and flit.rsp_type == 'read_data':
            # 读数据通过数据网络
            self.networks['data']['inject_fifo'].append(flit)
        else:
            # 写响应通过响应网络
            self.networks['rsp']['inject_fifo'].append(flit)
        
        self.cross_die_responses_sent += 1
        
        logging.debug(f"D2D_SN Die{self.die_id}: Forwarded response to local node {flit.destination}")
    
    def update(self):
        """
        重写update方法，在每个周期处理跨Die接收
        """
        # 首先处理跨Die接收队列
        self.process_cross_die_receives()
        
        # 调用父类的update方法
        super().update()
    
    def handle_local_response_for_cross_die(self, flit: Flit):
        """
        处理本地响应，检查是否需要发送回其他Die
        """
        # 检查是否是对跨Die请求的响应
        if hasattr(flit, 'original_source_die_id') and flit.original_source_die_id != self.die_id:
            # 需要发送回源Die
            # 这应该由连接的D2D_RN处理
            return True
        return False
    
    def get_statistics(self) -> dict:
        """获取D2D_SN统计信息"""
        stats = super().get_statistics()
        stats.update({
            'cross_die_requests_received': self.cross_die_requests_received,
            'cross_die_requests_forwarded': self.cross_die_requests_forwarded,
            'cross_die_responses_sent': self.cross_die_responses_sent,
            'pending_receives': len(self.cross_die_receive_queue),
        })
        return stats