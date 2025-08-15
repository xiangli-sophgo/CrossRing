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
        self.die_id = getattr(config, "DIE_ID", 0)  # 当前Die的ID
        self.cross_die_receive_queue = []  # 使用heapq管理的接收队列 [(arrival_cycle, flit)]
        self.target_die_interfaces = {}  # 将由D2D_Model设置 {die_id: d2d_sn_interface}

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
        """

        # 保存原始源信息（使用_physical命名一致性）
        flit.source_physical = flit.source
        flit.source_die_id_physical = flit.source_die_id
        flit.source_node_id_physical = flit.source_node_id

        # 创建第三阶段路由：D2D_RN → Die内目标节点
        if hasattr(flit, "final_destination_physical") and flit.final_destination_physical is not None:
            # 设置新的源为D2D_RN节点
            flit.source = self.ip_pos
            flit.destination = flit.final_destination_physical

            # 更新路径为从D2D_RN到最终目标
            if flit.final_destination_physical in self.routes[self.ip_pos]:
                flit.path = self.routes[self.ip_pos][flit.final_destination_physical]
                flit.path_index = 0  # 重置路径索引
                flit.current_position = self.ip_pos

                # 重置网络状态
                flit.is_injected = False
                flit.is_new_on_network = True

                # 根据请求类型选择网络
                if hasattr(flit, "req_type"):
                    if flit.req_type == "read":
                        # 读请求通过请求网络
                        self.networks["req"]["inject_fifo"].append(flit)
                    elif flit.req_type == "write":
                        # 写请求通过请求网络
                        self.networks["req"]["inject_fifo"].append(flit)
                    else:
                        # 默认请求网络
                        self.networks["req"]["inject_fifo"].append(flit)
                elif hasattr(flit, "rsp_type"):
                    if flit.rsp_type == "read_data":
                        # 读数据通过数据网络
                        self.networks["data"]["inject_fifo"].append(flit)
                    else:
                        # 写响应通过响应网络
                        self.networks["rsp"]["inject_fifo"].append(flit)
                else:
                    # 默认请求网络
                    self.networks["req"]["inject_fifo"].append(flit)

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

        # 计算到达时间
        arrival_cycle = self.current_cycle + delay

        # 获取目标Die的D2D_SN接口
        target_d2d_sn = self.target_die_interfaces[target_die_id]

        # 调度跨Die接收
        target_d2d_sn.schedule_cross_die_receive(flit, arrival_cycle)

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

        # 计算到达时间
        arrival_cycle = self.current_cycle + delay

        # 获取源Die的D2D_SN接口
        source_d2d_sn = self.target_die_interfaces[source_die_id]

        # 调度跨Die接收
        source_d2d_sn.schedule_cross_die_receive(flit, arrival_cycle)

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
