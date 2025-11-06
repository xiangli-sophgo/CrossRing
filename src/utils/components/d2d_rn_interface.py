"""
D2D_RN_Interface class for Die-to-Die communication.
Handles cross-die request initiation with AXI channel delays.
"""

from __future__ import annotations
import heapq
from collections import deque
from .ip_interface import IPInterface
from .flit import Flit, TokenBucket, D2D_ORIGIN_TARGET_ATTRS
import logging


class D2D_RN_Interface(IPInterface):
    """
    Die间请求节点 - 发起跨Die请求
    继承自IPInterface，复用所有现有功能
    """

    def __init__(self, ip_type: str, ip_pos: int, config, req_network, rsp_network, data_network, routes, ip_id: int = None):
        # 调用父类初始化
        super().__init__(ip_type, ip_pos, config, req_network, rsp_network, data_network, routes, ip_id)

        # D2D特有属性
        self.die_id = getattr(config, "DIE_ID", 0)  # 当前Die的ID

        # 每个AXI通道独立的接收FIFO队列 {channel_type: deque([(arrival_cycle, flit)])}
        self.cross_die_receive_queues = {"AR": deque(), "R": deque(), "AW": deque(), "W": deque(), "B": deque()}

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

        # 获取D2D延迟配置（已转换为cycles）
        self.d2d_ar_latency = config.D2D_AR_LATENCY
        self.d2d_r_latency = config.D2D_R_LATENCY
        self.d2d_aw_latency = config.D2D_AW_LATENCY
        self.d2d_w_latency = config.D2D_W_LATENCY
        self.d2d_b_latency = config.D2D_B_LATENCY

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

    def _get_flit_channel_type(self, flit: Flit) -> str:
        """
        从flit属性判断AXI通道类型

        Args:
            flit: 待判断的flit

        Returns:
            str: AXI通道类型 ("AR", "R", "AW", "W", "B")
        """
        # 优先检查flit_position (AXI flit有明确标记)
        if hasattr(flit, "flit_position") and flit.flit_position:
            flit_pos = str(flit.flit_position)
            if "AXI_AR" in flit_pos:
                return "AR"
            if "AXI_R" in flit_pos:
                return "R"
            if "AXI_AW" in flit_pos:
                return "AW"
            if "AXI_W" in flit_pos:
                return "W"
            if "AXI_B" in flit_pos:
                return "B"

        # 备用判断：根据flit类型和属性
        if hasattr(flit, "flit_type") and flit.flit_type == "data":
            if hasattr(flit, "write_data") or (hasattr(flit, "req_type") and flit.req_type == "write"):
                return "W"
            else:
                return "R"

        if hasattr(flit, "req_type"):
            if flit.req_type == "read":
                return "AR"
            elif flit.req_type == "write":
                return "AW"

        if hasattr(flit, "rsp_type"):
            if flit.rsp_type in ["datasend", "read_data"]:
                return "R"
            elif flit.rsp_type in ["write_ack", "write_response"]:
                return "B"

        # 默认返回AR
        return "AR"

    def schedule_cross_die_receive(self, flit: Flit, arrival_cycle: int):
        """
        调度跨Die接收 - 由对方Die的D2D_SN调用
        """
        channel_type = self._get_flit_channel_type(flit)
        self.cross_die_receive_queues[channel_type].append((arrival_cycle, flit))
        self.cross_die_requests_received += 1

    def process_cross_die_receives(self):
        """
        处理到期的跨Die接收 - 在每个周期调用
        遍历5个AXI通道的FIFO队列，处理到期的flit
        """
        for channel_type, queue in self.cross_die_receive_queues.items():
            while queue and queue[0][0] <= self.current_cycle:
                arrival_cycle, flit = queue.popleft()
                self.handle_received_cross_die_flit(flit)

    def handle_received_cross_die_flit(self, flit: Flit):
        """
        处理接收到的跨Die flit (AXI flit)
        区分写请求、写数据和其他类型，处理完后回收AXI flit
        """
        # print(f"[D2D RN Debug] 接收到跨Die flit: packet_id={getattr(flit, 'packet_id', 'None')}, "
        #       f"req_type={getattr(flit, 'req_type', 'None')}, rsp_type={getattr(flit, 'rsp_type', 'None')}, "
        #       f"flit_type={getattr(flit, 'flit_type', 'None')}, position={getattr(flit, 'flit_position', 'None')}")

        # 检查是否为AXI_W写数据
        if hasattr(flit, "flit_position") and flit.flit_position and "AXI_W" in str(flit.flit_position):
            # AXI_W通道的写数据
            # print(f"[D2D RN Debug] 处理AXI_W写数据")
            self.handle_cross_die_write_data(flit)
            # 回收AXI flit
            from .flit import _flit_pool

            _flit_pool.return_flit(flit)
        elif hasattr(flit, "req_type") and flit.req_type == "write" and not (hasattr(flit, "flit_type") and flit.flit_type == "data"):
            # 跨Die写请求（非数据）
            # print(f"[D2D RN Debug] 处理跨Die写请求")
            self.handle_cross_die_write_request(flit)
            # 回收AXI flit
            from .flit import _flit_pool

            _flit_pool.return_flit(flit)
        elif hasattr(flit, "flit_type") and flit.flit_type == "data" and hasattr(flit, "req_type") and flit.req_type == "write":
            # 传统写数据flit
            # print(f"[D2D RN Debug] 处理传统写数据")
            self.handle_cross_die_write_data(flit)
            # 回收AXI flit
            from .flit import _flit_pool

            _flit_pool.return_flit(flit)
        else:
            # 其他类型（读请求等），使用原有逻辑
            # print(f"[D2D RN Debug] 处理其他类型flit（读数据等）")
            self.handle_other_cross_die_flit(flit)
            # 回收AXI flit
            from .flit import _flit_pool

            _flit_pool.return_flit(flit)

    def handle_cross_die_write_request(self, flit: Flit):
        """
        处理跨Die写请求 - 消耗tracker资源并缓存请求
        """
        packet_id = flit.packet_id

        # 检查D2D_RN的写资源
        has_tracker = self.rn_tracker_count["write"]["count"] > 0
        has_databuffer = self.rn_wdb_count["count"] >= flit.burst_length

        if has_tracker and has_databuffer:
            # 消耗资源
            self.rn_tracker_count["write"]["count"] -= 1
            self.rn_wdb_count["count"] -= flit.burst_length

            # 创建flit副本保存写请求（因为AXI flit会被回收）
            from .flit import create_d2d_flit_copy

            write_req_copy = create_d2d_flit_copy(flit, source=0, destination=0, path=[0], attr_preset="request")

            # 缓存写请求副本等待写数据
            self.cross_die_write_requests[packet_id] = write_req_copy
            self.cross_die_write_data_cache[packet_id] = []

            # 添加到tracker list和更新pointer（修复：确保datasend响应处理器能找到请求）
            self.rn_tracker["write"].append(write_req_copy)
            self.rn_tracker_pointer["write"] += 1
        else:
            # 资源不足：这违反了AXI协议！应该在D2D_SN预留资源
            logging.warning(f"[D2D_RN] packet_id={packet_id} 资源不足被丢弃! tracker={self.rn_tracker_count['write']['count']}, wdb={self.rn_wdb_count['count']}")

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
        target_pos = write_req.d2d_target_node
        if target_pos < 0:
            return

        # 创建本地写请求
        source_mapped = self.ip_pos
        path = self.routes[source_mapped][target_pos] if target_pos in self.routes[source_mapped] else []

        from .flit import create_d2d_flit_copy

        # 创建本地写请求flit（使用basic预设，因为不需要req_attr）
        local_write_req = create_d2d_flit_copy(write_req, source=source_mapped, destination=target_pos, path=path, attr_preset="basic")

        local_write_req.source_type = self.ip_type
        local_write_req.destination_type = write_req.d2d_target_type

        # 记录经过的D2D_RN节点
        local_write_req.d2d_rn_node = self.ip_pos

        # 关键：设置为"new"让DDR进行SN资源检查（支持Die1的独立retry）
        # D2D_RN的RN资源已在handle_cross_die_write_request中分配
        local_write_req.req_attr = "new"

        # 设置路径信息
        local_write_req.path_index = 0
        local_write_req.current_position = self.ip_pos
        local_write_req.is_injected = False

        # 发送写请求（D2D_RN的enqueue会跳过RN资源检查）
        self.enqueue(local_write_req, "req")

        # 替换tracker list中的跨Die请求为本地写请求，确保create_write_packet使用正确的source/destination
        tracker_list = self.rn_tracker["write"]
        for i, req in enumerate(tracker_list):
            if req.packet_id == write_req.packet_id:
                tracker_list[i] = local_write_req
                break

        # 注意：不预填充wdb，让create_write_packet生成正确的Die内写数据
        # 跨Die写数据已缓存在cross_die_write_data_cache中

        self.cross_die_requests_forwarded += 1

    def handle_other_cross_die_flit(self, flit: Flit):
        """
        处理其他类型的跨Die flit（读请求等）
        注意：这里预先分配D2D_RN的RN资源（第296行加入tracker）
        转发时设置req_attr="new"让DDR进行SN资源检查（支持Die1的独立retry）
        """
        packet_id = getattr(flit, "packet_id", None)
        req_type = getattr(flit, "req_type", "read")

        # 确保burst_length有效（后面的代码需要用到）
        burst_length = getattr(flit, "burst_length", 4)
        if burst_length is None or burst_length <= 0:
            burst_length = 4

        # print(f"[D2D_RN] 转发跨Die请求 packet_id={packet_id}, req_type={req_type} (tracker已在基类分配)")

        # 使用D2D统一属性获取目标节点位置
        # d2d_target_node存储的是源映射位置，转为目标映射需要减去NUM_COL
        target_pos = flit.d2d_target_node
        if target_pos is None or target_pos < 0:
            raise ValueError(f"flit的d2d_target_node转换错误: d2d_target_node={flit.d2d_target_node}, target_pos={target_pos}, packet_id={getattr(flit, 'packet_id', 'None')}")

        # 计算路径（使用映射后的source和destination）
        source_mapped = self.ip_pos  # D2D_RN的位置已经是映射后的
        path = self.routes[source_mapped][target_pos] if target_pos in self.routes[source_mapped] else []

        # 创建新的请求flit，避免修改AXI传输的flit
        from .flit import create_d2d_flit_copy

        # 使用basic预设创建flit副本
        new_flit = create_d2d_flit_copy(flit, source=source_mapped, destination=target_pos, path=path, attr_preset="basic")

        # 确保burst_length有默认值
        if not hasattr(new_flit, "burst_length") or new_flit.burst_length is None or new_flit.burst_length < 0:
            new_flit.burst_length = 4  # 默认burst长度

        # 设置第三阶段的类型信息
        new_flit.source_type = self.ip_type  # D2D_RN的类型
        new_flit.destination_type = flit.d2d_target_type or "ddr_0"  # 使用D2D统一属性，提供默认值

        # D2D传输不设置original_*属性，辅助函数会从d2d_*属性推断

        # 保持D2D属性传递，确保后续处理能够正确识别
        if hasattr(flit, "d2d_origin_die"):
            new_flit.d2d_origin_die = flit.d2d_origin_die
        if hasattr(flit, "d2d_origin_node"):
            new_flit.d2d_origin_node = flit.d2d_origin_node
        if hasattr(flit, "d2d_origin_type"):
            new_flit.d2d_origin_type = flit.d2d_origin_type
        if hasattr(flit, "d2d_target_die"):
            new_flit.d2d_target_die = flit.d2d_target_die
        if hasattr(flit, "d2d_target_node"):
            new_flit.d2d_target_node = flit.d2d_target_node
        if hasattr(flit, "d2d_target_type"):
            new_flit.d2d_target_type = flit.d2d_target_type

        # 记录经过的D2D_RN节点
        new_flit.d2d_rn_node = self.ip_pos

        # 设置网络状态
        new_flit.path_index = 0
        new_flit.current_position = self.ip_pos
        new_flit.is_injected = False
        new_flit.req_attr = "new"  # 标记为新请求，消耗资源

        # 记录D2D_RN的tracker信息（用于后续数据返回时释放）
        if req_type == "read":
            self.rn_tracker["read"].append(new_flit)
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
            # 这是跨Die写请求的datasend响应，需要发送跨Die写数据到DDR

            # 获取tracker中的请求（已被替换为local_write_req）
            req = next((r for r in self.rn_tracker["write"] if r.packet_id == packet_id), None)

            if req and packet_id in self.cross_die_write_data_cache:
                # 根据跨Die写数据重新创建Die内写数据（参考create_write_packet）
                cross_die_data_flits = self.cross_die_write_data_cache[packet_id]
                local_data_flits = []

                for i, cross_die_flit in enumerate(cross_die_data_flits):
                    # 创建新的Die内写数据flit
                    from .flit import Flit

                    local_flit = Flit(req.source, req.destination, req.path)

                    # 复制关键属性
                    local_flit.sync_latency_record(req)
                    # D2D传输不设置_original属性，辅助函数会从d2d_*属性推断
                    local_flit.flit_type = "data"
                    # 保序信息将在inject_fifo出队时分配（inject_to_l2h_pre）
                    local_flit.departure_cycle = self.current_cycle + i * self.config.NETWORK_FREQUENCY
                    local_flit.req_departure_cycle = req.departure_cycle if hasattr(req, "departure_cycle") else self.current_cycle
                    local_flit.entry_db_cycle = req.entry_db_cycle if hasattr(req, "entry_db_cycle") else self.current_cycle
                    local_flit.source_type = req.source_type
                    local_flit.destination_type = req.destination_type
                    # D2D传输不设置original_*属性
                    local_flit.req_type = req.req_type
                    local_flit.packet_id = req.packet_id
                    local_flit.flit_id = i
                    local_flit.burst_length = req.burst_length
                    local_flit.traffic_id = getattr(req, "traffic_id", 0)
                    local_flit.is_last_flit = i == len(cross_die_data_flits) - 1

                    # 继承D2D属性
                    for attr in ["d2d_origin_die", "d2d_origin_node", "d2d_origin_type", "d2d_target_die", "d2d_target_node", "d2d_target_type"]:
                        if hasattr(req, attr):
                            setattr(local_flit, attr, getattr(req, attr))

                    local_data_flits.append(local_flit)

                # 发送新创建的Die内写数据
                for flit in local_data_flits:
                    self.enqueue(flit, "data")

                # 清理wdb（不需要放到wdb，直接发送）
                self.rn_wdb.pop(packet_id, None)

            # 防重复：检查是否已经发送过write_complete响应
            if packet_id not in self.sent_write_complete_responses:
                # 发送写数据完成后，立即发送write_complete响应到AXI_B通道
                # tracker会在send_cross_die_write_complete中释放
                self.send_cross_die_write_complete(packet_id)
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

        # 从tracker获取请求（已被替换为包含d2d_rn_node的local_write_req）
        tracker_req = next((r for r in self.rn_tracker["write"] if r.packet_id == packet_id), None)
        if not tracker_req:
            return

        # 创建write_complete响应
        from .flit import _flit_pool
        from .flit import copy_flit_attributes

        write_complete_rsp = _flit_pool.get_flit(source=self.ip_pos, destination=0, path=[self.ip_pos])

        # 设置基本属性
        write_complete_rsp.packet_id = packet_id
        write_complete_rsp.rsp_type = "write_complete"
        write_complete_rsp.req_type = "write"  # 设置req_type为write，让D2D_SN能正确识别
        write_complete_rsp.source_type = tracker_req.d2d_target_type  # 使用目标类型（DDR）作为响应源
        write_complete_rsp.destination_type = tracker_req.d2d_origin_type

        # 复制D2D属性（从tracker_req复制，包含完整的D2D属性包括d2d_rn_node）
        copy_flit_attributes(
            tracker_req,
            write_complete_rsp,
            D2D_ORIGIN_TARGET_ATTRS,
        )

        # 通过AXI_B通道发送回源Die，明确指定B通道
        source_die_id = tracker_req.d2d_origin_die
        if self.d2d_sys and source_die_id is not None:
            self.d2d_sys.enqueue_rn(write_complete_rsp, source_die_id, self.d2d_b_latency, channel="B")

        # 发送后立即释放D2D_RN的tracker和WDB资源（符合设计文档3.3节要求）
        self.cross_die_write_requests.pop(packet_id)
        self.cross_die_write_data_cache.pop(packet_id, None)
        self.rn_tracker_count["write"]["count"] += 1
        self.rn_wdb_count["count"] += tracker_req.burst_length

        # 从tracker list中移除并更新pointer（通过packet_id查找，因为tracker可能已被替换为local_write_req）
        tracker_list = self.rn_tracker["write"]
        req_to_remove = next((r for r in tracker_list if r.packet_id == packet_id), None)
        if req_to_remove:
            tracker_list.remove(req_to_remove)
            self.rn_tracker_pointer["write"] -= 1

    def _check_and_reserve_resources(self, req: Flit) -> bool:
        """
        重写RN资源检查：对于跨Die请求，资源已在接收时预先分配

        - 跨Die读请求：资源在handle_other_cross_die_flit中分配（第296行）
        - 跨Die写请求：资源在handle_cross_die_write_request中分配
        """
        # 检查是否为已预分配资源的跨Die请求
        is_cross_die_read = req.req_type == "read" and req in self.rn_tracker["read"]
        is_cross_die_write = req.req_type == "write" and req.packet_id in self.cross_die_write_requests

        if is_cross_die_read or is_cross_die_write:
            # 资源已预先分配，跳过重复检查
            return True

        # 其他请求使用基类逻辑
        return super()._check_and_reserve_resources(req)

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
        重写inject_step方法
        在inject阶段处理跨Die接收，inject_to_l2h_pre以1GHz频率运行
        """
        self.current_cycle = cycle
        cycle_mod = cycle % self.config.NETWORK_FREQUENCY

        # 首先处理跨Die接收队列
        self.process_cross_die_receives()

        # 1GHz inject操作（每个网络周期执行一次）
        if cycle_mod == 0:
            for net_type in ["req", "rsp", "data"]:
                self.inject_to_l2h_pre(net_type)

        # 2GHz操作：l2h_pre到IQ_channel_buffer
        for net_type in ["req", "rsp", "data"]:
            self.l2h_to_IQ_channel_buffer(net_type)

    def eject_step(self, cycle):
        """
        重写eject_step方法，处理从本地网络接收到的跨Die读数据并发送回原始请求者
        D2D_RN的h2l_l FIFO运行在1GHz频率
        """
        self.current_cycle = cycle
        cycle_mod = cycle % self.config.NETWORK_FREQUENCY

        ejected_flits = []

        # 2GHz 操作(每半个网络周期执行一次)
        for net_type in ["req", "rsp", "data"]:
            self.EQ_channel_buffer_to_h2l_pre(net_type)
            self.h2l_h_to_h2l_l_pre(net_type)

        # 1GHz 操作（每个网络周期执行一次）
        if cycle_mod == 0:
            for net_type in ["req", "rsp", "data"]:
                flit = self.h2l_l_to_eject_fifo(net_type)
                if flit:
                    ejected_flits.append(flit)

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

            # 收集到data buffer中，但不更新网络的recv_flits_num
            if flit.packet_id not in self.rn_rdb:
                self.rn_rdb[flit.packet_id] = []
            self.rn_rdb[flit.packet_id].append(flit)

            # 注意：这里不调用父类的网络统计更新，避免跨Die数据被计入本Die的recv_flits_num

            # 检查是否收集完整个burst - 只有在刚好收齐时才处理一次
            collected_flits = self.rn_rdb[flit.packet_id]
            if len(collected_flits) == flit.burst_length:
                # 找到对应的tracker但不释放
                req = next((req for req in self.rn_tracker["read"] if req.packet_id == flit.packet_id), None)

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
        根据DDR返回的数据重新创建跨Die路由的读数据flit（类似写请求处理）
        """
        if not data_flits:
            return

        # 获取第一个flit的信息作为参考
        first_flit = data_flits[0]
        source_die_id = getattr(first_flit, "d2d_origin_die", None)

        if source_die_id is None or source_die_id == self.die_id:
            # 不需要跨Die返回
            return

        # 根据DDR返回的数据重新创建跨Die读数据flit
        cross_die_data_flits = []

        for i, local_flit in enumerate(data_flits):
            # 创建新的跨Die读数据flit（从D2D_RN返回到D2D_SN）
            from .flit import _flit_pool, copy_flit_attributes

            # 计算跨Die返回路径：D2D_RN → D2D_SN
            source = self.ip_pos  # D2D_RN的位置
            destination = first_flit.d2d_origin_node  # 原始请求者的源映射位置
            path = [source]  # AXI传输不需要NoC路径

            cross_die_flit = _flit_pool.get_flit(source, destination, path)

            # 设置基本属性
            cross_die_flit.sync_latency_record(tracker_req)
            # D2D传输不设置_original属性，辅助函数会从d2d_*属性推断
            cross_die_flit.flit_type = "data"
            cross_die_flit.req_type = "read"
            cross_die_flit.rsp_type = "read_data"
            cross_die_flit.packet_id = first_flit.packet_id
            cross_die_flit.flit_id = i
            cross_die_flit.burst_length = first_flit.burst_length
            cross_die_flit.traffic_id = getattr(first_flit, "traffic_id", 0)
            cross_die_flit.is_last_flit = i == len(data_flits) - 1
            cross_die_flit.source_type = self.ip_type
            # D2D传输不设置original_*属性

            # 继承D2D属性（从tracker_req复制，因为DDR返回的数据flit不包含这些属性）
            # tracker_req是D2D_RN保存的请求flit，包含完整的D2D跨Die属性
            copy_flit_attributes(tracker_req, cross_die_flit, ["d2d_origin_die", "d2d_origin_node", "d2d_origin_type", "d2d_target_die", "d2d_target_node", "d2d_target_type", "d2d_sn_node"])

            # 设置destination_type（必须在copy_flit_attributes之后，使用已复制的d2d_origin_type）
            cross_die_flit.destination_type = cross_die_flit.d2d_origin_type

            # 记录经过的D2D_RN节点
            cross_die_flit.d2d_rn_node = self.ip_pos

            # 继承时间戳信息
            copy_flit_attributes(local_flit, cross_die_flit, ["departure_cycle", "entry_db_cycle", "req_departure_cycle", "leave_db_cycle"])

            cross_die_data_flits.append(cross_die_flit)

        # 通过AXI R通道发送新创建的跨Die读数据
        for flit in cross_die_data_flits:
            if self.d2d_sys:
                self.d2d_sys.enqueue_rn(flit, source_die_id, self.d2d_r_latency, channel="R")

        # 数据发送到AXI通道后，立即释放D2D_RN的tracker和RDB资源
        if tracker_req in self.rn_tracker["read"]:
            self.rn_tracker["read"].remove(tracker_req)
            self.rn_tracker_count["read"]["count"] += 1
            self.rn_tracker_pointer["read"] -= 1
            self.rn_rdb_count["count"] += tracker_req.burst_length

        self.cross_die_data_responses_sent = getattr(self, "cross_die_data_responses_sent", 0) + len(cross_die_data_flits)

    def get_statistics(self) -> dict:
        """获取D2D_RN统计信息"""
        # 由于父类IPInterface没有get_statistics方法，直接返回D2D统计信息
        stats = {
            "cross_die_requests_sent": self.cross_die_requests_sent,
            "cross_die_responses_received": self.cross_die_responses_received,
            "cross_die_requests_received": self.cross_die_requests_received,
            "cross_die_requests_forwarded": self.cross_die_requests_forwarded,
            "pending_receives": sum(len(q) for q in self.cross_die_receive_queues.values()),
            "d2d_rn_tracker_count": self.rn_tracker_count["read"],
            "d2d_rn_rdb_count": self.rn_rdb_count,
        }
        return stats
