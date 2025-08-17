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
import traceback


class D2D_SN_Interface(IPInterface):
    """
    Die间响应节点 - 接收跨Die请求并转发到Die内目标节点
    继承自IPInterface，复用所有现有功能
    """

    def __init__(self, ip_type: str, ip_pos: int, config, req_network, rsp_network, data_network, node, routes, ip_id: int = None):
        # 调用父类初始化
        super().__init__(ip_type, ip_pos, config, req_network, rsp_network, data_network, node, routes, ip_id)

        # D2D特有属性
        self.die_id = getattr(config, "DIE_ID", 0)  # 当前Die的ID
        self.cross_die_receive_queue = []  # 使用heapq管理的接收队列 [(arrival_cycle, flit)]
        
        # 添加D2D_SN的带宽限制（在父类初始化后）
        if not self.tx_token_bucket and not self.rx_token_bucket:
            # 如果父类没有设置带宽限制，使用D2D_SN专用配置
            d2d_sn_bw_limit = getattr(config, "D2D_SN_BW_LIMIT", 128)
            from .flit import TokenBucket
            self.tx_token_bucket = TokenBucket(
                rate=d2d_sn_bw_limit / config.NETWORK_FREQUENCY / config.FLIT_SIZE,
                bucket_size=d2d_sn_bw_limit,
            )
            self.rx_token_bucket = TokenBucket(
                rate=d2d_sn_bw_limit / config.NETWORK_FREQUENCY / config.FLIT_SIZE,
                bucket_size=d2d_sn_bw_limit,
            )

        # 获取D2D延迟配置
        self.d2d_ar_latency = getattr(config, "D2D_AR_LATENCY", 10)
        self.d2d_r_latency = getattr(config, "D2D_R_LATENCY", 8)
        self.d2d_aw_latency = getattr(config, "D2D_AW_LATENCY", 10)
        self.d2d_w_latency = getattr(config, "D2D_W_LATENCY", 2)
        self.d2d_b_latency = getattr(config, "D2D_B_LATENCY", 8)

        # 跨Die请求统计
        self.cross_die_requests_received = 0
        self.cross_die_requests_forwarded = 0
        self.cross_die_responses_sent = 0
        
        # D2D_Sys引用（由D2DModel设置）
        self.d2d_sys = None

    def schedule_cross_die_receive(self, flit: Flit, arrival_cycle: int):
        """
        调度跨Die接收 - 由对方Die的D2D_RN调用
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
        处理接收到的跨Die flit
        根据flit类型进行相应处理
        """

        # 检查是否为读请求
        if hasattr(flit, "req_type"):
            if flit.req_type == "read":
                # 读请求：转发到Die内目标SN节点
                self.forward_read_request_to_local_sn(flit)
            elif flit.req_type == "write":
                # 写请求：转发到Die内目标SN节点
                self.forward_write_request_to_local_sn(flit)
        elif hasattr(flit, "rsp_type"):
            # 响应：转发回Die内原始请求节点
            self.forward_response_to_local_rn(flit)

    def forward_read_request_to_local_sn(self, flit: Flit):
        """
        将跨Die读请求转发到本地目标SN节点
        """
        # 更新flit的源和目标信息
        # 保存原始信息供响应时使用（使用_physical命名一致性）
        flit.source_physical = flit.source
        flit.source_die_id_physical = flit.source_die_id
        flit.source_node_id_physical = flit.source_node_id

        # 设置新的源为D2D_SN节点
        flit.source = self.ip_pos
        flit.destination = flit.target_node_id  # Die内目标节点

        # 通过请求网络发送
        self.networks["req"]["inject_fifo"].append(flit)
        self.cross_die_requests_forwarded += 1

    def forward_write_request_to_local_sn(self, flit: Flit):
        """
        将跨Die写请求转发到本地目标SN节点
        """
        # 类似读请求处理（使用_physical命名一致性）
        flit.source_physical = flit.source
        flit.source_die_id_physical = flit.source_die_id
        flit.source_node_id_physical = flit.source_node_id

        flit.source = self.ip_pos
        flit.destination = flit.target_node_id

        # 通过请求网络发送
        self.networks["req"]["inject_fifo"].append(flit)
        self.cross_die_requests_forwarded += 1

    def forward_response_to_local_rn(self, flit: Flit):
        """
        将跨Die响应转发回本地原始请求节点
        """
        # 恢复原始目标信息（使用_physical命名一致性）
        if hasattr(flit, "source_physical"):
            flit.destination = flit.source_physical
        elif hasattr(flit, "source_node_id_physical"):
            flit.destination = flit.source_node_id_physical
        else:
            flit.destination = flit.source_node_id

        flit.source = self.ip_pos

        # 根据响应类型选择网络
        if hasattr(flit, "rsp_type") and flit.rsp_type == "read_data":
            # 读数据通过数据网络
            self.networks["data"]["inject_fifo"].append(flit)
        else:
            # 写响应通过响应网络
            self.networks["rsp"]["inject_fifo"].append(flit)

        self.cross_die_responses_sent += 1

    def inject_step(self, cycle):
        """
        重写inject_step方法，在inject阶段处理跨Die接收
        """
        # 首先处理跨Die接收队列
        self.process_cross_die_receives()

        # 调用父类的inject_step方法
        super().inject_step(cycle)

    def eject_step(self, cycle):
        """
        重写eject_step方法，在eject阶段检查跨Die请求并转发
        """
        # 调用父类的eject_step方法
        ejected_flits = super().eject_step(cycle)
        
        # 检查ejected的flit是否需要跨Die转发
        if ejected_flits:
            for flit in ejected_flits:
                if self._is_cross_die_flit(flit):
                    self._handle_cross_die_transfer(flit)

        return ejected_flits

    def _is_cross_die_flit(self, flit):
        """检查flit是否需要跨Die转发"""
        return hasattr(flit, "source_die_id") and hasattr(flit, "target_die_id") and flit.source_die_id is not None and flit.target_die_id is not None and flit.source_die_id != flit.target_die_id

    def _handle_cross_die_transfer(self, flit):
        """处理跨Die转发（第二阶段：Die0_D2D_SN → Die1_D2D_RN）
        可选择重新生成AXI专用flit，保持packet_id不变
        """
        try:
            target_die_id = flit.target_die_id

            # 根据请求类型选择AXI通道
            if hasattr(flit, "req_type"):
                if flit.req_type == "read":
                    channel = "AR"
                else:  # write
                    channel = "AW"
            else:
                channel = "AR"

            # 可选：重新生成AXI专用flit（保持packet_id和关键信息）
            # 当前实现：直接使用原始flit进行AXI传输
            # 如果需要，可以在这里创建新的AXI专用flit：
            # axi_flit = self._create_axi_flit(flit)
            # 但为了简化，当前直接使用原始flit
            
            # 使用D2D_Sys进行仲裁和AXI传输
            if hasattr(self, 'd2d_sys') and self.d2d_sys:
                self.d2d_sys.enqueue_sn(flit, target_die_id, channel)
                self.cross_die_requests_forwarded += 1

        except Exception as e:
            import traceback
            print(f"D2D_SN跨Die请求转发错误 [周期{self.current_cycle}, 位置{self.position}]:")
            print(f"  错误类型: {type(e).__name__}")
            print(f"  错误信息: {str(e)}")
            print(f"  Flit信息: source={flit.source}, dest={flit.destination}, dst_die={flit.target_die_id}")
            print(f"  目标Die接口: {getattr(self, 'target_die_d2d_rn', {})}")
            if hasattr(e, '__traceback__'):
                tb_lines = traceback.format_exc().split('\n')
                print(f"  错误位置: {tb_lines[-3].strip() if len(tb_lines) >= 3 else 'N/A'}")

    def handle_local_response_for_cross_die(self, flit: Flit):
        """
        处理本地响应，检查是否需要发送回其他Die
        """
        # 检查是否是对跨Die请求的响应
        if hasattr(flit, "source_die_id_physical") and flit.source_die_id_physical != self.die_id:
            # 需要发送回源Die
            # 这应该由连接的D2D_RN处理
            return True
        return False

    def _handle_received_data(self, flit: Flit):
        """
        重写数据接收处理，支持跨Die数据转发到原始请求者
        """
        # 检查是否是跨Die返回的数据
        # 通过检查是否有final_destination_physical属性来判断
        if hasattr(flit, "final_destination_physical") and flit.final_destination_physical is not None:
            # 这是从其他Die返回的数据，需要转发到原始请求者
            self.forward_cross_die_data_to_requester(flit)
        else:
            # 本地数据，调用父类正常处理
            super()._handle_received_data(flit)
    
    def forward_cross_die_data_to_requester(self, flit: Flit):
        """
        将跨Die返回的数据转发到原始请求者
        """
        # 获取最终目标信息
        final_destination = getattr(flit, "final_destination_physical", None)
        final_destination_type = getattr(flit, "final_destination_type", None)
        
        if final_destination is None:
            print(f"[D2D_SN] 错误：数据flit缺少final_destination_physical信息")
            return
        
        # 设置路由信息
        flit.source = self.ip_pos  # 当前D2D_SN
        flit.destination = final_destination  # 原始请求者（如GDMA）
        
        # 更新路径
        if hasattr(self, "routes") and self.ip_pos in self.routes and final_destination in self.routes[self.ip_pos]:
            flit.path = self.routes[self.ip_pos][final_destination]
            flit.path_index = 0
        
        # 标记为新的网络传输
        flit.is_injected = False
        flit.is_new_on_network = True
        flit.current_position = self.ip_pos
        
        # 记录D2D_SN处理的数据包
        if not hasattr(self, "sn_data_tracker"):
            self.sn_data_tracker = {}
        
        packet_id = getattr(flit, "packet_id", -1)
        if packet_id not in self.sn_data_tracker:
            self.sn_data_tracker[packet_id] = {
                "expected_count": getattr(flit, "burst_length", 4),
                "forwarded_count": 0,
                "start_cycle": self.current_cycle
            }
        
        # 通过数据网络发送到最终目标
        self.enqueue(flit, "data")
        self.sn_data_tracker[packet_id]["forwarded_count"] += 1
        
        # 统计
        self.cross_die_data_forwarded = getattr(self, "cross_die_data_forwarded", 0) + 1
        
        # 检查是否所有数据都已转发
        tracker = self.sn_data_tracker[packet_id]
        if tracker["forwarded_count"] >= tracker["expected_count"]:
            # 所有数据包已转发，可以清理tracker
            print(f"[D2D_SN] 完成packet_id={packet_id}的{tracker['expected_count']}个数据包转发")
            del self.sn_data_tracker[packet_id]
    
    def get_statistics(self) -> dict:
        """获取D2D_SN统计信息"""
        # 由于父类IPInterface没有get_statistics方法，直接返回D2D统计信息
        stats = {
            "cross_die_requests_received": self.cross_die_requests_received,
            "cross_die_requests_forwarded": self.cross_die_requests_forwarded,
            "cross_die_responses_sent": self.cross_die_responses_sent,
            "pending_receives": len(self.cross_die_receive_queue),
        }
        return stats
