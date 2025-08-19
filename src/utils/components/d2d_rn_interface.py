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

        # 防止重复发送write_complete响应的记录 {packet_id: True}
        self.sent_write_complete_responses = set()

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

        # 跨Die写数据缓存 {packet_id: [data_flits]}
        self.cross_die_write_data_cache = {}

        # 跨Die写请求追踪 {packet_id: write_request_flit}
        self.cross_die_write_requests = {}

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
        处理接收到的跨Die flit
        区分写请求、写数据和其他类型
        """
        # print(f"[D2D RN Debug] 接收到跨Die flit: packet_id={getattr(flit, 'packet_id', 'None')}, "
        #       f"req_type={getattr(flit, 'req_type', 'None')}, rsp_type={getattr(flit, 'rsp_type', 'None')}, "
        #       f"flit_type={getattr(flit, 'flit_type', 'None')}, position={getattr(flit, 'flit_position', 'None')}")

        # 检查是否为AXI_W写数据
        if hasattr(flit, "flit_position") and flit.flit_position and "AXI_W" in str(flit.flit_position):
            # AXI_W通道的写数据
            # print(f"[D2D RN Debug] 处理AXI_W写数据")
            self.handle_cross_die_write_data(flit)
        elif hasattr(flit, "req_type") and flit.req_type == "write" and not (hasattr(flit, "flit_type") and flit.flit_type == "data"):
            # 跨Die写请求（非数据）
            # print(f"[D2D RN Debug] 处理跨Die写请求")
            self.handle_cross_die_write_request(flit)
        elif hasattr(flit, "flit_type") and flit.flit_type == "data" and hasattr(flit, "req_type") and flit.req_type == "write":
            # 传统写数据flit
            # print(f"[D2D RN Debug] 处理传统写数据")
            self.handle_cross_die_write_data(flit)
        else:
            # 其他类型（读请求等），使用原有逻辑
            # print(f"[D2D RN Debug] 处理其他类型flit（读数据等）")
            self.handle_other_cross_die_flit(flit)

    def handle_cross_die_write_request(self, flit: Flit):
        """
        处理跨Die写请求 - 消耗tracker资源并缓存请求
        """
        packet_id = flit.packet_id

        # 检查D2D_RN的写资源
        has_tracker = self.node.rn_tracker_count["write"][self.ip_type][self.ip_pos] > 0
        has_databuffer = self.node.rn_wdb_count[self.ip_type][self.ip_pos] >= flit.burst_length

        if has_tracker and has_databuffer:
            # 消耗资源
            self.node.rn_tracker_count["write"][self.ip_type][self.ip_pos] -= 1
            self.node.rn_wdb_count[self.ip_type][self.ip_pos] -= flit.burst_length

            # 缓存写请求等待写数据
            self.cross_die_write_requests[packet_id] = flit
            self.cross_die_write_data_cache[packet_id] = []

    def handle_cross_die_write_data(self, flit: Flit):
        """
        处理跨Die写数据 - 缓存数据直到完整接收
        """
        packet_id = flit.packet_id

        if packet_id not in self.cross_die_write_data_cache:
            return

        # 缓存写数据
        self.cross_die_write_data_cache[packet_id].append(flit)

        # 注意：不在D2D_RN中记录写数据接收，因为这里只是中转节点
        # 写数据接收应该在真正的目标IP（如DDR）中记录

        # 检查是否收集完所有写数据
        collected_flits = self.cross_die_write_data_cache[packet_id]
        expected_length = flit.burst_length

        if len(collected_flits) >= expected_length:
            # 所有写数据已收集完成，转发到本地SN
            write_req = self.cross_die_write_requests.get(packet_id)
            if write_req:
                self.forward_write_to_local_sn(write_req, collected_flits)

    def forward_write_to_local_sn(self, write_req: Flit, data_flits: list):
        """
        将写请求和写数据转发到本地SN
        """
        # 使用D2D统一属性获取目标节点位置
        target_pos = write_req.d2d_target_node - self.config.NUM_COL
        if target_pos < 0:
            return

        # 创建本地写请求
        source_mapped = self.ip_pos
        path = self.routes[source_mapped][target_pos] if target_pos in self.routes[source_mapped] else []

        from .flit import Flit

        local_write_req = Flit(source=source_mapped, destination=target_pos, path=path)

        # 复制关键属性
        for attr in ["packet_id", "req_type", "burst_length", "d2d_origin_die", "d2d_origin_node", "d2d_origin_type", "d2d_target_die", "d2d_target_node", "d2d_target_type"]:
            if hasattr(write_req, attr):
                setattr(local_write_req, attr, getattr(write_req, attr))

        local_write_req.source_type = self.ip_type
        local_write_req.destination_type = write_req.d2d_target_type
        local_write_req.req_attr = "new"

        # 设置路径信息
        local_write_req.path_index = 0
        local_write_req.current_position = self.ip_pos
        local_write_req.is_injected = False
        local_write_req.is_new_on_network = True

        # 发送写请求
        self.enqueue(local_write_req, "req")

        # 准备写数据（将在收到data_send响应后发送）
        self.node.rn_wdb[self.ip_type][self.ip_pos][write_req.packet_id] = data_flits

        self.cross_die_requests_forwarded += 1

    def handle_other_cross_die_flit(self, flit: Flit):
        """
        处理其他类型的跨Die flit（读请求等）- 修复版本
        在转发时检查并分配D2D_RN的tracker和RDB资源
        """
        # 确保burst_length有效
        burst_length = getattr(flit, "burst_length", 4)
        if burst_length is None or burst_length <= 0:
            burst_length = 4

        # 检查D2D_RN的资源可用性（只对读请求检查读资源）
        req_type = getattr(flit, "req_type", "read")
        if req_type == "read":
            has_tracker = self.node.rn_tracker_count["read"][self.ip_type][self.ip_pos] > 0
            has_databuffer = self.node.rn_rdb_count[self.ip_type][self.ip_pos] >= burst_length
            
            if not (has_tracker and has_databuffer):
                # TODO: 实现资源不足时的等待队列机制，而不是直接丢弃请求
                # 当前版本：资源不足时直接返回失败，可能导致请求丢失
                print(f"[D2D_RN] 资源不足，无法转发packet {getattr(flit, 'packet_id', '?')}: "
                      f"tracker={has_tracker}, databuffer={has_databuffer} (需要{burst_length})")
                return False
                
            # 分配D2D_RN的读资源
            self.node.rn_tracker_count["read"][self.ip_type][self.ip_pos] -= 1
            self.node.rn_rdb_count[self.ip_type][self.ip_pos] -= burst_length
            self.node.rn_tracker_pointer["read"][self.ip_type][self.ip_pos] += 1

        # 使用D2D统一属性获取目标节点位置
        # d2d_target_node存储的是源映射位置，转为目标映射需要减去NUM_COL
        target_pos = flit.d2d_target_node - self.config.NUM_COL
        if target_pos is None or target_pos < 0:
            # 资源分配失败，需要回滚
            if req_type == "read":
                self.node.rn_tracker_count["read"][self.ip_type][self.ip_pos] += 1
                self.node.rn_rdb_count[self.ip_type][self.ip_pos] += burst_length
                self.node.rn_tracker_pointer["read"][self.ip_type][self.ip_pos] -= 1
            raise ValueError(f"flit的d2d_target_node转换错误: d2d_target_node={flit.d2d_target_node}, target_pos={target_pos}, packet_id={getattr(flit, 'packet_id', 'None')}")

        # 计算路径（使用映射后的source和destination）
        source_mapped = self.ip_pos  # D2D_RN的位置已经是映射后的
        path = self.routes[source_mapped][target_pos] if target_pos in self.routes[source_mapped] else []

        # 创建新的请求flit，避免修改AXI传输的flit
        from .flit import Flit

        new_flit = Flit(source=source_mapped, destination=target_pos, path=path)  # D2D_RN的位置  # 映射后的目标节点位置

        # 复制关键属性（使用D2D统一属性）
        for attr in [
            "packet_id",
            "flit_id",
            "req_type",
            "rsp_type",
            "flit_type",
            "d2d_origin_die",
            "d2d_origin_node",
            "d2d_origin_type",
            "d2d_target_die",
            "d2d_target_node",
            "d2d_target_type",
            "burst_length",
        ]:
            if hasattr(flit, attr):
                setattr(new_flit, attr, getattr(flit, attr))

        # 确保burst_length有默认值
        if not hasattr(new_flit, "burst_length") or new_flit.burst_length is None or new_flit.burst_length < 0:
            new_flit.burst_length = 4  # 默认burst长度

        # 设置第三阶段的类型信息
        new_flit.source_type = self.ip_type  # D2D_RN的类型
        new_flit.destination_type = flit.d2d_target_type or "ddr_0"  # 使用D2D统一属性，提供默认值

        # 设置原始类型信息，用于创建返回路径
        new_flit.original_source_type = flit.d2d_origin_type or "gdma_0"  # 原始源类型，提供默认值
        new_flit.original_destination_type = new_flit.destination_type  # 当前目标类型
        new_flit.destination_original = new_flit.destination  # 当前目标位置
        new_flit.source_original = flit.d2d_origin_node  # 原始源位置（源映射）

        # 设置网络状态
        new_flit.path_index = 0
        new_flit.current_position = self.ip_pos
        new_flit.is_injected = False
        new_flit.is_new_on_network = True
        new_flit.req_attr = "new"  # 标记为新请求，消耗资源

        # 记录D2D_RN的tracker信息（用于后续数据返回时释放）
        if req_type == "read":
            self.node.rn_tracker["read"][self.ip_type][self.ip_pos].append(new_flit)
            # 设置burst_length确保释放时正确计算databuffer
            new_flit.burst_length = burst_length

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
        target_die_id = getattr(flit, "d2d_target_die", None)
        if target_die_id is None:
            return False
        return target_die_id != self.die_id

    def handle_cross_die_request(self, flit: Flit):
        """
        处理跨Die请求 - 添加AR/AW延迟并发送到目标Die的D2D_SN
        """
        if not self.is_cross_die_request(flit):
            # 本地请求，走正常流程
            return False

        target_die_id = flit.d2d_target_die
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
        # 使用D2D统一属性获取原始请求者Die
        source_die_id = getattr(flit, "d2d_origin_die", None)
        if source_die_id is None or source_die_id == self.die_id:
            # 本地响应，走正常流程
            return False

        if source_die_id not in self.target_die_interfaces:
            return False

        # 根据响应类型选择延迟
        if hasattr(flit, "rsp_type"):
            if flit.rsp_type == "read_data":
                delay = self.d2d_r_latency
            elif flit.rsp_type in ["write_complete", "negative"]:
                delay = self.d2d_b_latency
            else:
                delay = self.d2d_r_latency
        else:
            # 默认使用R延迟
            delay = self.d2d_r_latency

        # 使用D2D_Sys进行仲裁和AXI传输
        self.d2d_sys.enqueue_rn(flit, source_die_id, delay)

        self.cross_die_responses_received += 1

        return True

    def _handle_received_response(self, rsp: Flit):
        """
        重写响应处理，支持跨Die写响应的特殊处理
        """
        packet_id = rsp.packet_id

        # 检查是否为跨Die写请求的datasend响应
        if hasattr(rsp, "rsp_type") and rsp.rsp_type == "datasend" and packet_id in self.cross_die_write_requests:
            # 这是跨Die写请求的datasend响应，发送写数据后需要发送write_complete

            # 调用父类处理（会发送写数据）
            super()._handle_received_response(rsp)

            # 防重复：检查是否已经发送过write_complete响应
            if packet_id not in self.sent_write_complete_responses:
                # 发送写数据完成后，立即发送write_complete响应到AXI_B通道
                self.send_cross_die_write_complete(packet_id)
                # 标记已发送
                self.sent_write_complete_responses.add(packet_id)
            return

        # 检查是否为跨Die写完成响应
        elif hasattr(rsp, "rsp_type") and rsp.rsp_type == "write_complete" and packet_id in self.cross_die_write_requests:
            # 这是跨Die写请求的本地完成响应，需要通过B通道返回给源Die

            # 防重复：检查是否已经发送过write_complete响应
            if packet_id not in self.sent_write_complete_responses:
                # 清理跨Die写相关缓存
                if packet_id in self.cross_die_write_requests:
                    write_req = self.cross_die_write_requests.pop(packet_id)
                    self.cross_die_write_data_cache.pop(packet_id, None)

                    # 释放D2D_RN的tracker
                    self.node.rn_tracker_count["write"][self.ip_type][self.ip_pos] += 1

                    # 修改响应类型为write_complete并通过B通道返回
                    rsp.rsp_type = "write_complete"
                    self.handle_cross_die_response(rsp)

                    # 标记已发送
                    self.sent_write_complete_responses.add(packet_id)
            return

        # 其他响应类型，调用父类处理
        super()._handle_received_response(rsp)

    def send_cross_die_write_complete(self, packet_id: int):
        """
        发送跨Die写完成响应到AXI_B通道
        """
        if packet_id not in self.cross_die_write_requests:
            return

        write_req = self.cross_die_write_requests[packet_id]

        # 创建write_complete响应 - 简化创建，让D2D_Sys处理AXI传输细节
        from .flit import Flit

        write_complete_rsp = Flit(source=self.ip_pos, destination=0, path=[self.ip_pos])  # D2D_Sys会重新设置  # D2D_Sys会重新设置  # 提供有效路径避免初始化错误，D2D_Sys会重新设置

        # 设置基本属性
        write_complete_rsp.packet_id = packet_id
        write_complete_rsp.rsp_type = "write_complete"
        write_complete_rsp.req_type = "write"  # 设置req_type为write，让D2D_SN能正确识别
        write_complete_rsp.source_type = write_req.d2d_target_type  # 使用目标类型（DDR）作为响应源
        write_complete_rsp.destination_type = write_req.d2d_origin_type

        # 复制D2D属性 - 这是关键信息
        for attr in ["d2d_origin_die", "d2d_origin_node", "d2d_origin_type", "d2d_target_die", "d2d_target_node", "d2d_target_type"]:
            if hasattr(write_req, attr):
                setattr(write_complete_rsp, attr, getattr(write_req, attr))

        # 通过AXI_B通道发送回源Die，明确指定B通道
        source_die_id = write_req.d2d_origin_die
        if self.d2d_sys and source_die_id is not None:
            self.d2d_sys.enqueue_rn(write_complete_rsp, source_die_id, self.d2d_b_latency, channel="B")

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

    def eject_step(self, cycle):
        """
        重写eject_step方法，处理从本地网络接收到的跨Die读数据并发送回原始请求者
        """
        # 调用父类的eject_step方法
        ejected_flits = super().eject_step(cycle)

        # 跨Die读数据的处理已经在_handle_received_data中的handle_cross_die_data_response方法中统一处理
        # 这里不需要再单独处理，避免重复发送

        return ejected_flits

    def _handle_received_data(self, flit: Flit):
        """
        重写数据接收处理，支持跨Die数据返回
        """
        # 对于跨Die数据，我们需要特殊处理，不能让父类立即释放tracker
        # 使用D2D统一属性判断跨Die数据
        is_cross_die_data = hasattr(flit, "d2d_origin_die") and flit.d2d_origin_die is not None and flit.d2d_origin_die != self.die_id

        if is_cross_die_data and getattr(flit, "req_type", "read") == "read":
            # 跨Die读数据，手动处理而不调用父类，避免计入recv_flits_num
            flit.arrival_cycle = getattr(self, "current_cycle", 0)
            self.data_wait_cycles_h += getattr(flit, "wait_cycle_h", 0)
            self.data_wait_cycles_v += getattr(flit, "wait_cycle_v", 0)
            self.data_cir_h_num += getattr(flit, "eject_attempts_h", 0)
            self.data_cir_v_num += getattr(flit, "eject_attempts_v", 0)

            # 记录读数据接收到D2D模型
            d2d_model = getattr(self.req_network, "d2d_model", None)
            if d2d_model:
                burst_length = getattr(flit, "burst_length", 4)
                # 这是跨Die读数据接收
                d2d_model.record_read_data_received(flit.packet_id, self.die_id, burst_length, is_cross_die=True)

            # 收集到data buffer中，但不更新网络的recv_flits_num
            if flit.packet_id not in self.node.rn_rdb[self.ip_type][self.ip_pos]:
                self.node.rn_rdb[self.ip_type][self.ip_pos][flit.packet_id] = []
            self.node.rn_rdb[self.ip_type][self.ip_pos][flit.packet_id].append(flit)

            # 注意：这里不调用父类的网络统计更新，避免跨Die数据被计入本Die的recv_flits_num

            # 检查是否收集完整个burst - 只有在刚好收齐时才处理一次
            collected_flits = self.node.rn_rdb[self.ip_type][self.ip_pos][flit.packet_id]
            if len(collected_flits) == flit.burst_length:
                # 找到对应的tracker但不释放
                req = next((req for req in self.node.rn_tracker["read"][self.ip_type][self.ip_pos] if req.packet_id == flit.packet_id), None)

                if req:
                    # 设置D2D_RN节点的处理时间戳（但不设置最终完成时间）
                    complete_cycle = self.current_cycle
                    for f in collected_flits:
                        f.leave_db_cycle = self.current_cycle
                        if hasattr(f, "sync_latency_record"):
                            f.sync_latency_record(req)
                        # 注意：data_received_complete_cycle应该在最终请求方IP设置，而不是在D2D_RN中转节点

                    # 处理跨Die数据返回 - 只调用一次
                    self.handle_cross_die_data_response(collected_flits, req)
        else:
            # 非跨Die数据或写数据，调用父类正常处理
            super()._handle_received_data(flit)

    def handle_cross_die_data_response(self, data_flits: list, tracker_req):
        """
        处理跨Die数据响应，发送回源Die
        修复版本：数据发送后立即释放tracker资源
        """
        if not data_flits:
            return

        # 获取第一个flit的信息作为参考
        first_flit = data_flits[0]
        source_die_id = getattr(first_flit, "d2d_origin_die", None)

        if source_die_id is None or source_die_id == self.die_id:
            # 不需要跨Die返回
            return

        # 为每个数据flit准备跨Die传输
        for i, flit in enumerate(data_flits):
            # 修正flit_id：数据包应该有递增的flit_id (0, 1, 2, 3)
            flit.flit_id = i
            # 设置最后一个flit标记
            flit.is_last_flit = i == len(data_flits) - 1

            # 使用D2D_Sys进行AXI R通道传输
            if self.d2d_sys:
                self.d2d_sys.enqueue_rn(flit, source_die_id, self.d2d_r_latency, channel="R")

        # 数据发送到AXI通道后，立即释放D2D_RN的tracker和RDB资源
        if tracker_req in self.node.rn_tracker["read"][self.ip_type][self.ip_pos]:
            self.node.rn_tracker["read"][self.ip_type][self.ip_pos].remove(tracker_req)
            self.node.rn_tracker_count["read"][self.ip_type][self.ip_pos] += 1
            self.node.rn_tracker_pointer["read"][self.ip_type][self.ip_pos] -= 1
            self.node.rn_rdb_count[self.ip_type][self.ip_pos] += tracker_req.burst_length
            
            print(f"[D2D_RN] 立即释放tracker资源 packet {getattr(first_flit, 'packet_id', '?')}: "
                  f"tracker_count={self.node.rn_tracker_count['read'][self.ip_type][self.ip_pos]}, "
                  f"rdb_count={self.node.rn_rdb_count[self.ip_type][self.ip_pos]}")

        self.cross_die_data_responses_sent = getattr(self, "cross_die_data_responses_sent", 0) + len(data_flits)


    def get_statistics(self) -> dict:
        """获取D2D_RN统计信息"""
        # 由于父类IPInterface没有get_statistics方法，直接返回D2D统计信息
        stats = {
            "cross_die_requests_sent": self.cross_die_requests_sent,
            "cross_die_responses_received": self.cross_die_responses_received,
            "cross_die_requests_received": self.cross_die_requests_received,
            "cross_die_requests_forwarded": self.cross_die_requests_forwarded,
            "pending_receives": len(self.cross_die_receive_queue),
            "d2d_rn_tracker_count": self.node.rn_tracker_count["read"][self.ip_type][self.ip_pos],
            "d2d_rn_rdb_count": self.node.rn_rdb_count[self.ip_type][self.ip_pos],
        }
        return stats
