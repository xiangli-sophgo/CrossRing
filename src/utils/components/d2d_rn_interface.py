"""
D2D_RN_Interface class for Die-to-Die communication.
Handles cross-die request initiation with AXI channel delays.
"""

from __future__ import annotations
import heapq
from collections import deque
from .ip_interface import IPInterface
from .flit import Flit, TokenBucket
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
        self.die_id = getattr(config, "DIE_ID", 0)  # 当前Die的ID
        self.cross_die_receive_queue = []  # 使用heapq管理的接收队列 [(arrival_cycle, flit)]
        self.target_die_interfaces = {}  # 将由D2D_Model设置 {die_id: d2d_sn_interface}
        
        # 添加D2D_RN的带宽限制（在父类初始化后）
        if not self.tx_token_bucket and not self.rx_token_bucket:
            # 如果父类没有设置带宽限制，使用D2D_RN专用配置
            d2d_rn_bw_limit = getattr(config, "D2D_RN_BW_LIMIT", 128)
            self.tx_token_bucket = TokenBucket(
                rate=d2d_rn_bw_limit / config.NETWORK_FREQUENCY / config.FLIT_SIZE,
                bucket_size=d2d_rn_bw_limit,
            )
            self.rx_token_bucket = TokenBucket(
                rate=d2d_rn_bw_limit / config.NETWORK_FREQUENCY / config.FLIT_SIZE,
                bucket_size=d2d_rn_bw_limit,
            )

        # 获取D2D延迟配置
        self.d2d_ar_latency = getattr(config, "D2D_AR_LATENCY", 10)
        self.d2d_r_latency = getattr(config, "D2D_R_LATENCY", 8)
        self.d2d_aw_latency = getattr(config, "D2D_AW_LATENCY", 10)
        self.d2d_w_latency = getattr(config, "D2D_W_LATENCY", 2)
        self.d2d_b_latency = getattr(config, "D2D_B_LATENCY", 8)

        # 跨Die请求统计
        self.cross_die_requests_sent = 0
        self.cross_die_responses_received = 0
        self.cross_die_requests_received = 0
        self.cross_die_requests_forwarded = 0
        
        # D2D_Sys引用（由D2DModel设置）
        self.d2d_sys = None

    def schedule_cross_die_receive(self, flit: Flit, arrival_cycle: int):
        """
        调度跨Die接收 - 由对方Die的D2D_SN调用
        """
        heapq.heappush(self.cross_die_receive_queue, (arrival_cycle, flit))
        self.cross_die_requests_received += 1

    def process_cross_die_receives(self):
        """
        处理到期的跨Die接收 - 在每个周期调用
        """
        while self.cross_die_receive_queue and self.cross_die_receive_queue[0][0] <= self.current_cycle:
            arrival_cycle, flit = heapq.heappop(self.cross_die_receive_queue)
            self.handle_received_cross_die_flit(flit)

    def handle_received_cross_die_flit(self, flit: Flit):
        """
        处理接收到的跨Die flit - 第三阶段路由（D2D_RN → 目标节点）
        创建新的内部请求并消耗tracker/databuffer资源
        """
        # 创建新的请求flit，避免修改AXI传输的flit
        from .flit import Flit
        target_pos = getattr(flit, "final_destination_physical", flit.destination)
        path = self.routes[self.ip_pos][target_pos] if target_pos in self.routes[self.ip_pos] else []
        
        new_flit = Flit(
            source=self.ip_pos,  # 新请求的源为D2D_RN
            destination=target_pos,
            path=path
        )
        
        # 复制关键属性（包括ID，但不包括Die ID，因为第三阶段是本地请求）
        for attr in ['packet_id', 'flit_id', 'req_type', 'rsp_type', 'flit_type',
                     'target_node_id', 'source_node_id', 'burst_length']:
            if hasattr(flit, attr):
                setattr(new_flit, attr, getattr(flit, attr))
        
        # 确保burst_length有默认值
        if not hasattr(new_flit, 'burst_length') or new_flit.burst_length is None or new_flit.burst_length < 0:
            new_flit.burst_length = 4  # 默认burst长度
        
        # 设置第三阶段的类型信息
        new_flit.source_type = self.ip_type      # D2D_RN的类型
        new_flit.destination_type = getattr(flit, "final_destination_type", "ddr_0")  # 目标类型
        
        # 设置原始类型信息，用于创建返回路径
        new_flit.original_source_type = getattr(flit, "source_type", None)  # 原始源类型
        new_flit.original_destination_type = new_flit.destination_type  # 当前目标类型
        new_flit.destination_original = new_flit.destination  # 当前目标位置  
        new_flit.source_original = getattr(flit, "source_physical", getattr(flit, "source", flit.source))  # 原始源位置
        
        # 第三阶段是Die1内部的本地请求，更新Die ID
        new_flit.source_die_id = self.die_id  # 当前Die ID (Die1)
        new_flit.target_die_id = self.die_id  # 目标也是当前Die ID (Die1)
        
        # 保存原始源信息供响应时使用
        # 如果是第一次跨Die（阶段2），source_die_id就是原始的die
        new_flit.source_physical = getattr(flit, 'source_physical', flit.source)
        new_flit.source_die_id_physical = flit.source_die_id  # 这里应该是Die0
        new_flit.source_node_id_physical = getattr(flit, 'source_node_id_physical', flit.source_node_id)
        
        new_flit.final_destination_physical = getattr(flit, "final_destination_physical", flit.destination)
        new_flit.final_destination_type = getattr(flit, "final_destination_type", flit.destination_type)

        # 设置网络状态
        new_flit.path_index = 0
        new_flit.current_position = self.ip_pos
        new_flit.is_injected = False
        new_flit.is_new_on_network = True
        new_flit.req_attr = "new"  # 标记为新请求，消耗资源

        # 根据请求类型选择网络，使用enqueue方法而不是直接append
        if hasattr(new_flit, "req_type"):
            if new_flit.req_type in ["read", "write"]:
                self.enqueue(new_flit, "req")
            else:
                self.enqueue(new_flit, "req")
        elif hasattr(new_flit, "rsp_type"):
            if new_flit.rsp_type == "read_data":
                self.enqueue(new_flit, "data")
            else:
                self.enqueue(new_flit, "rsp")
        else:
            self.enqueue(new_flit, "req")

        self.cross_die_requests_forwarded += 1

    def is_cross_die_request(self, flit: Flit) -> bool:
        """检查是否为跨Die请求"""
        target_die_id = getattr(flit, "target_die_id", None)
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
            return False

        # 根据请求类型选择延迟
        if hasattr(flit, "req_type"):
            if flit.req_type == "read":
                delay = self.d2d_ar_latency
            else:  # write
                delay = self.d2d_aw_latency
        else:
            # 默认使用AR延迟
            delay = self.d2d_ar_latency

        # 使用D2D_Sys进行仲裁和AXI传输
        self.d2d_sys.enqueue_rn(flit, target_die_id, delay)

        self.cross_die_requests_sent += 1

        return True

    def handle_cross_die_response(self, flit: Flit):
        """
        处理跨Die响应 - 添加R/B延迟并发送回源Die
        """
        source_die_id = getattr(flit, "source_die_id", None)
        if source_die_id is None or source_die_id == self.die_id:
            # 本地响应，走正常流程
            return False

        if source_die_id not in self.target_die_interfaces:
            return False

        # 根据响应类型选择延迟
        if hasattr(flit, "rsp_type"):
            if flit.rsp_type == "read_data":
                delay = self.d2d_r_latency
            else:  # write_response
                delay = self.d2d_b_latency
        else:
            # 默认使用R延迟
            delay = self.d2d_r_latency

        # 使用D2D_Sys进行仲裁和AXI传输
        self.d2d_sys.enqueue_rn(flit, source_die_id, delay)

        self.cross_die_responses_received += 1

        return True

    def process_inject_request(self, flit: Flit, network_type: str):
        """
        重写父类方法，拦截跨Die请求
        """
        if network_type == "req" and self.is_cross_die_request(flit):
            # 处理跨Die请求
            if self.handle_cross_die_request(flit):
                return  # 已处理，不走正常网络流程

        # 非跨Die请求或其他网络类型，调用父类方法
        super().process_inject_request(flit, network_type)

    def inject_step(self, cycle):
        """
        重写inject_step方法，在inject阶段处理跨Die接收
        """
        # 首先处理跨Die接收队列
        self.process_cross_die_receives()
        

        # 调用父类的inject_step方法
        super().inject_step(cycle)
    
    def _handle_received_data(self, flit: Flit):
        """
        重写数据接收处理，支持跨Die数据返回
        """
        # 对于跨Die数据，我们需要特殊处理，不能让父类立即释放tracker
        is_cross_die_data = (hasattr(flit, "source_die_id_physical") and 
                            flit.source_die_id_physical is not None and
                            flit.source_die_id_physical != self.die_id)
        
        
        if is_cross_die_data and getattr(flit, "req_type", "read") == "read":
            # 跨Die读数据，手动处理而不调用父类
            flit.arrival_cycle = getattr(self, "current_cycle", 0)
            self.data_wait_cycles_h += getattr(flit, "wait_cycle_h", 0)
            self.data_wait_cycles_v += getattr(flit, "wait_cycle_v", 0)
            self.data_cir_h_num += getattr(flit, "eject_attempts_h", 0)
            self.data_cir_v_num += getattr(flit, "eject_attempts_v", 0)
            
            # 收集到data buffer中
            if flit.packet_id not in self.node.rn_rdb[self.ip_type][self.ip_pos]:
                self.node.rn_rdb[self.ip_type][self.ip_pos][flit.packet_id] = []
            self.node.rn_rdb[self.ip_type][self.ip_pos][flit.packet_id].append(flit)
            
            # 检查是否收集完整个burst
            collected_flits = self.node.rn_rdb[self.ip_type][self.ip_pos][flit.packet_id]
            if len(collected_flits) == flit.burst_length:
                # 找到对应的tracker但不释放
                req = next((req for req in self.node.rn_tracker["read"][self.ip_type][self.ip_pos] 
                           if req.packet_id == flit.packet_id), None)
                
                if req:
                    # 不能动态添加属性到Flit，使用其他方式标记
                    
                    # 设置完成时间戳等信息
                    complete_cycle = self.current_cycle
                    for f in collected_flits:
                        f.leave_db_cycle = self.current_cycle
                        if hasattr(f, "sync_latency_record"):
                            f.sync_latency_record(req)
                        f.data_received_complete_cycle = complete_cycle
                    
                    # 处理跨Die数据返回
                    self.handle_cross_die_data_response(collected_flits, req)
                else:
                    print(f"[D2D_RN] 错误：找不到packet_id={flit.packet_id}的tracker")
        else:
            # 非跨Die数据或写数据，调用父类正常处理
            super()._handle_received_data(flit)
    
    def handle_cross_die_data_response(self, data_flits: list, tracker_req):
        """
        处理跨Die数据响应，发送回源Die
        """
        if not data_flits:
            return
        
        # 获取第一个flit的信息作为参考
        first_flit = data_flits[0]
        source_die_id = getattr(first_flit, "source_die_id_physical", None)
        
        if source_die_id is None or source_die_id == self.die_id:
            # 不需要跨Die返回
            return
        
        # 为每个数据flit准备跨Die传输
        for flit in data_flits:
            # 不能添加新属性到Flit，使用现有属性标记跨Die返回
            # 使用flit_type或其他方式识别
            
            # 保持原始请求者信息，用于第六阶段
            # 这些属性在create_read_packet中应该已经设置了
            
            # 使用D2D_Sys进行AXI R通道传输
            if self.d2d_sys:
                self.d2d_sys.enqueue_rn(flit, source_die_id, self.d2d_r_latency, channel="R")
        
        # 记录tracker的packet_id，用于后续释放
        if not hasattr(self, "pending_cross_die_transfers"):
            self.pending_cross_die_transfers = {}
        self.pending_cross_die_transfers[tracker_req.packet_id] = {
            "tracker": tracker_req,
            "flits_count": len(data_flits),
            "completed_count": 0
        }
        
        self.cross_die_data_responses_sent = getattr(self, "cross_die_data_responses_sent", 0) + len(data_flits)
        print(f"[D2D_RN] 发送{len(data_flits)}个数据包返回Die{source_die_id}")
    
    def release_cross_die_tracker(self, packet_id: int):
        """
        释放跨Die传输完成的tracker
        在D2D_Sys确认传输完成后调用
        """
        if hasattr(self, "pending_cross_die_transfers") and packet_id in self.pending_cross_die_transfers:
            transfer_info = self.pending_cross_die_transfers[packet_id]
            tracker = transfer_info["tracker"]
            
            # 从tracker列表中移除
            if tracker in self.node.rn_tracker["read"][self.ip_type][self.ip_pos]:
                self.node.rn_tracker["read"][self.ip_type][self.ip_pos].remove(tracker)
                self.node.rn_tracker_count["read"][self.ip_type][self.ip_pos] += 1
                self.node.rn_tracker_pointer["read"][self.ip_type][self.ip_pos] -= 1
                self.node.rn_rdb_count[self.ip_type][self.ip_pos] += tracker.burst_length
                
                print(f"[D2D_RN] 释放跨Die传输完成的tracker，packet_id={packet_id}")
            
            # 清理pending记录
            del self.pending_cross_die_transfers[packet_id]
    
    def notify_cross_die_transfer_complete(self, flit: Flit):
        """
        通知跨Die传输完成（由D2D_Sys调用）
        """
        packet_id = getattr(flit, "packet_id", None)
        if packet_id is None:
            return
        
        if hasattr(self, "pending_cross_die_transfers") and packet_id in self.pending_cross_die_transfers:
            transfer_info = self.pending_cross_die_transfers[packet_id]
            transfer_info["completed_count"] += 1
            
            # 检查是否所有flit都已传输完成
            if transfer_info["completed_count"] >= transfer_info["flits_count"]:
                # 所有数据包传输完成，释放tracker
                self.release_cross_die_tracker(packet_id)

    def get_statistics(self) -> dict:
        """获取D2D_RN统计信息"""
        # 由于父类IPInterface没有get_statistics方法，直接返回D2D统计信息
        stats = {
            "cross_die_requests_sent": self.cross_die_requests_sent,
            "cross_die_responses_received": self.cross_die_responses_received,
            "cross_die_requests_received": self.cross_die_requests_received,
            "cross_die_requests_forwarded": self.cross_die_requests_forwarded,
            "pending_receives": len(self.cross_die_receive_queue),
        }
        return stats
