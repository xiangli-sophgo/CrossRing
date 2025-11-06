"""
D2D_SN_Interface class for Die-to-Die communication.
Handles cross-die request reception and forwarding within the die.
"""

from __future__ import annotations
import heapq
from collections import deque
from .ip_interface import IPInterface
from .flit import Flit
import traceback


class D2D_SN_Interface(IPInterface):
    """
    Die间响应节点 - 接收跨Die请求并转发到Die内目标节点
    继承自IPInterface，复用所有现有功能
    """

    def __init__(self, ip_type: str, ip_pos: int, config, req_network, rsp_network, data_network, routes, ip_id: int = None):
        # 调用父类初始化
        super().__init__(ip_type, ip_pos, config, req_network, rsp_network, data_network, routes, ip_id)

        # D2D特有属性
        self.die_id = getattr(config, "DIE_ID", 0)  # 当前Die的ID

        # 每个AXI通道独立的接收FIFO队列 {channel_type: deque([(arrival_cycle, flit)])}
        self.cross_die_receive_queues = {
            "AR": deque(),
            "R": deque(),
            "AW": deque(),
            "W": deque(),
            "B": deque()
        }

        self.target_die_interfaces = {}  # 将由D2D_Model设置 {die_id: d2d_rn_interface}

        # 防止重复处理AXI_B响应的记录 {(packet_id, cycle): True}
        self.processed_write_complete_responses = {}

        # 防止重复处理写请求 {packet_id: True}
        self.processed_write_requests = set()

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

        # 获取D2D延迟配置（已转换为cycles）
        self.d2d_ar_latency = config.D2D_AR_LATENCY
        self.d2d_r_latency = config.D2D_R_LATENCY
        self.d2d_aw_latency = config.D2D_AW_LATENCY
        self.d2d_w_latency = config.D2D_W_LATENCY
        self.d2d_b_latency = config.D2D_B_LATENCY

        # 跨Die请求统计
        self.cross_die_requests_received = 0
        self.cross_die_requests_forwarded = 0
        self.cross_die_responses_sent = 0

        # D2D_Sys引用（由D2DModel设置）
        self.d2d_sys = None

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

        # 默认返回R (SN主要处理响应)
        return "R"

    def _create_response_flit(self, req_flit: Flit, rsp_type: str, destination_pos: int = None) -> Flit:
        """创建响应flit的通用方法"""
        from .flit import Flit

        # 节点编号直接就是网络位置，不需要映射转换
        dest_pos = destination_pos if destination_pos is not None else req_flit.source

        # 检查路由表中是否存在该路径
        if dest_pos not in self.routes.get(self.ip_pos, {}):
            raise ValueError(f"路由表中找不到从{self.ip_pos}到{dest_pos}的路径")

        path = self.routes[self.ip_pos][dest_pos]

        response = Flit(source=self.ip_pos, destination=dest_pos, path=path)
        response.packet_id = req_flit.packet_id
        response.rsp_type = rsp_type
        response.req_type = getattr(req_flit, "req_type", "write")
        response.source_type = self.ip_type
        response.destination_type = getattr(req_flit, "source_type", "gdma_0")

        return response

    def schedule_cross_die_receive(self, flit: Flit, arrival_cycle: int):
        """
        调度跨Die接收 - 由对方Die的D2D_RN调用
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
        根据flit类型进行相应处理，处理完后回收AXI flit
        """
        # 优先检查是否为AXI_B的write_complete响应
        if hasattr(flit, "flit_position") and flit.flit_position and "AXI_B" in str(flit.flit_position):
            # AXI_B通道的写完成响应
            self.handle_cross_die_write_complete_response(flit)
            # 回收AXI flit（forward方法会创建新NoC flit）
            from .flit import _flit_pool

            _flit_pool.return_flit(flit)
            return  # 已处理，直接返回
        elif hasattr(flit, "flit_type") and flit.flit_type == "data":
            # 数据flit：检查是否为跨Die写数据
            if hasattr(flit, "req_type") and flit.req_type == "write":
                self.handle_cross_die_write_data(flit)
                # 回收AXI flit
                from .flit import _flit_pool

                _flit_pool.return_flit(flit)
            else:
                # 其他数据（如读数据），调用父类处理
                self._handle_received_data(flit)
                # 回收AXI flit
                from .flit import _flit_pool

                _flit_pool.return_flit(flit)
        elif hasattr(flit, "req_type") and flit.req_type:
            # 请求flit：需要资源检查
            if flit.req_type == "read":
                # 读请求：转发到Die内目标SN节点
                self.forward_read_request_to_local_sn(flit)
                # 回收AXI flit
                from .flit import _flit_pool

                _flit_pool.return_flit(flit)
            elif flit.req_type == "write":
                # 写请求：需要先检查资源并返回data_send响应
                self.handle_local_cross_die_write_request(flit)
                # 回收AXI flit
                from .flit import _flit_pool

                _flit_pool.return_flit(flit)
        elif hasattr(flit, "rsp_type") and flit.rsp_type:
            # 响应：转发回Die内原始请求节点
            self.forward_response_to_local_rn(flit)
            # 回收AXI flit
            from .flit import _flit_pool

            _flit_pool.return_flit(flit)

    def forward_read_request_to_local_sn(self, flit: Flit):
        """
        将跨Die读请求转发到本地目标SN节点
        """
        # 使用统一方法创建新的NoC flit
        from .flit import create_d2d_flit_copy

        # 设置新的源为D2D_SN节点
        source = self.ip_pos
        # 使用D2D统一属性
        if hasattr(flit, "d2d_target_node"):
            destination = flit.d2d_target_node  # Die内目标节点源映射位置
        else:
            destination = getattr(flit, "target_node_id", self.ip_pos)  # 兼容旧属性

        # 计算路径
        path = self.routes[source][destination] if destination in self.routes[source] else [source]

        # 创建请求flit副本
        new_flit = create_d2d_flit_copy(flit, source=source, destination=destination, path=path, attr_preset="request")

        # 保持已有的d2d_sn_node(在eject阶段已设置)，不要覆盖
        # create_d2d_flit_copy会复制d2d_sn_node属性

        # 设置网络状态
        new_flit.path_index = 0
        new_flit.is_injected = False
        new_flit.current_position = self.ip_pos

        # 通过请求网络发送
        self.enqueue(new_flit, "req")
        self.cross_die_requests_forwarded += 1

    def forward_response_to_local_rn(self, flit: Flit):
        """
        将跨Die响应转发回本地原始请求节点
        """
        # 使用统一方法创建新的NoC flit
        from .flit import create_d2d_flit_copy

        # 使用D2D统一属性恢复原始目标信息
        # d2d_origin_node是源映射位置，转为目标映射位置（减去NUM_COL）
        destination = flit.d2d_origin_node
        source = self.ip_pos

        # 计算正确的路径
        if destination in self.routes[source]:
            path = self.routes[source][destination]
        else:
            path = [source]  # 如果没有路由，使用源节点作为路径

        # 创建响应flit副本
        new_flit = create_d2d_flit_copy(flit, source=source, destination=destination, path=path, attr_preset="response")

        # D2D节点追踪：
        # - 如果是跨Die响应返回(已有d2d_sn_node)，保持原值
        # - 如果是Die内响应(没有d2d_sn_node)，设置当前节点
        if not hasattr(flit, "d2d_sn_node") or flit.d2d_sn_node is None:
            new_flit.d2d_sn_node = self.ip_pos

        new_flit.path_index = 0

        # 为write_complete响应设置正确的属性
        if hasattr(new_flit, "rsp_type") and new_flit.rsp_type == "write_complete":
            new_flit.req_type = "write"  # 写完成响应需要标记为write类型
            # 设置正确的source_type为D2D_SN自己（响应从D2D_SN发出）
            new_flit.source_type = self.ip_type  # d2d_sn_0
            # 设置destination_type为原始请求者类型
            new_flit.destination_type = new_flit.d2d_origin_type  # 原始请求者类型 (gdma_0)

        # 重置网络状态属性以确保能被正确inject
        new_flit.is_injected = False
        new_flit.current_position = self.ip_pos

        # 根据响应类型选择网络
        if hasattr(new_flit, "rsp_type") and new_flit.rsp_type == "read_data":
            # 读数据通过数据网络
            self.enqueue(new_flit, "data")
        else:
            # 写响应通过响应网络
            self.enqueue(new_flit, "rsp")

        self.cross_die_responses_sent += 1

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
        重写eject_step方法，在eject阶段处理写请求的资源检查和跨Die转发
        D2D_SN的h2l_l FIFO运行在1GHz频率
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

        # 检查ejected的flit
        if ejected_flits:
            for flit in ejected_flits:
                if (
                    hasattr(flit, "d2d_target_die")
                    and hasattr(flit, "d2d_origin_die")
                    and flit.d2d_target_die is not None
                    and flit.d2d_origin_die is not None
                    and flit.d2d_target_die != flit.d2d_origin_die
                ):
                    # 记录跨Die请求/数据经过的D2D_SN节点
                    if not hasattr(flit, "d2d_sn_node") or flit.d2d_sn_node is None:
                        flit.d2d_sn_node = self.ip_pos

                        # 同时设置对应的d2d_rn_node（从配置或d2d_sys获取）
                        if not hasattr(flit, "d2d_rn_node") or flit.d2d_rn_node is None:
                            # 从d2d_sys获取目标Die的D2D_RN节点位置
                            target_die = flit.d2d_target_die
                            if hasattr(self, "d2d_sys") and self.d2d_sys and target_die is not None:
                                # d2d_sys.target_die_rn_pos 存储的是目标Die的D2D_RN物理位置
                                flit.d2d_rn_node = self.d2d_sys.target_die_rn_pos

                    # 检查flit类型：数据flit vs 请求flit
                    if hasattr(flit, "flit_type") and flit.flit_type == "data":
                        # 检查是否是从AXI跨Die传输来的数据（已经在handle_received_cross_die_flit中处理过）
                        if hasattr(flit, "flit_position") and flit.flit_position and "AXI" in str(flit.flit_position):
                            pass  # 跳过AXI跨Die数据flit（已在跨Die接收中处理）
                        else:
                            # 本地网络的跨Die写数据：收集数据，最后一笔数据到达时发送AXI AW+W
                            self.handle_cross_die_write_data(flit)
                    elif hasattr(flit, "req_type") and flit.req_type == "write":
                        # 跨Die写请求：先进行资源检查并返回data_send响应
                        self.handle_local_cross_die_write_request(flit)
                    elif hasattr(flit, "req_type") and flit.req_type == "read":
                        # 跨Die读请求：使用修改后的SN处理机制
                        # print(f"[D2D_SN] 处理跨Die读请求 packet_id={flit.packet_id}")
                        # 标记为新请求，使用D2D专用的处理方法
                        flit.req_attr = "new"
                        self._handle_cross_die_read_request(flit)
                    else:
                        # 其他类型：直接跨Die转发
                        self._handle_cross_die_transfer(flit)

        return ejected_flits

    def handle_local_cross_die_write_request(self, flit: Flit):
        """
        处理本地接收的跨Die写请求
        完全遵循基类ip_interface的retry逻辑

        每个Die独立处理，不继承前一个Die的retry状态
        """
        packet_id = flit.packet_id

        # 遵循基类逻辑：根据req_attr区分新请求和retry请求
        if getattr(flit, "req_attr", "new") == "new":
            # 新请求：检查资源
            has_tracker = self.sn_tracker_count["share"]["count"] > 0
            has_databuffer = self.sn_wdb_count["count"] >= flit.burst_length

            if has_tracker and has_databuffer:
                # 分配资源并加入tracker（与基类一致）
                flit.sn_tracker_type = "share"
                self.sn_tracker.append(flit)
                self.sn_tracker_count["share"]["count"] -= 1
                self.sn_wdb[flit.packet_id] = []
                self.sn_wdb_count["count"] -= flit.burst_length

                # 发送datasend响应
                data_send_rsp = self._create_response_flit(flit, "datasend")
                self.enqueue(data_send_rsp, "rsp")
            else:
                # 资源不足：发送negative并加入等待队列（与基类一致）
                negative_rsp = self._create_response_flit(flit, "negative")
                self.enqueue(negative_rsp, "rsp")
                self.sn_req_wait["write"].append(flit)
        else:
            # retry请求（req_attr="old"）：直接发送datasend（与基类一致）
            # flit已在等待队列处理时加入tracker，这里不需要再次分配资源
            data_send_rsp = self._create_response_flit(flit, "datasend")
            self.enqueue(data_send_rsp, "rsp")

    def _handle_cross_die_read_request(self, flit: Flit):
        """
        处理跨Die读请求：使用基类的tracker分配逻辑，但转发到D2D_RN而不是直接生成读数据
        """
        packet_id = flit.packet_id

        if flit.req_attr == "new":
            # 使用基类的SN tracker分配逻辑
            if self.sn_tracker_count["ro"]["count"] > 0:
                flit.sn_tracker_type = "ro"
                self.sn_tracker.append(flit)
                self.sn_tracker_count["ro"]["count"] -= 1

                # print(f"[D2D_SN] 分配RO tracker并转发跨Die读请求 packet_id={packet_id}, "
                #       f"剩余tracker={self.node.sn_tracker_count[self.ip_type]['ro'][self.ip_pos]}")

                # 转发到D2D_RN，而不是直接生成读数据包
                self._handle_cross_die_transfer(flit)

                # 注意：不在这里释放tracker！tracker会在数据返回时释放
            else:
                # 资源不足，返回negative响应
                # print(f"[D2D_SN] RO tracker不足，读请求 packet_id={packet_id} 返回negative响应")
                self.create_rsp(flit, "negative")
                self.sn_req_wait[flit.req_type].append(flit)
        else:
            # 重试请求：直接转发
            print(f"[D2D_SN] 转发重试的跨Die读请求 packet_id={packet_id}")
            self._handle_cross_die_transfer(flit)

    def handle_local_cross_die_read_request(self, flit: Flit):
        """
        已废弃：现在使用_handle_cross_die_read_request方法
        """
        print(f"[D2D_SN] 警告：调用了已废弃的handle_local_cross_die_read_request方法")
        # 转到新的处理方法
        self._handle_cross_die_read_request(flit)

    def _handle_cross_die_transfer(self, flit):
        """处理跨Die转发（第二阶段：Die0_D2D_SN → Die1_D2D_RN）"""
        target_die_id = getattr(flit, "d2d_target_die", getattr(flit, "target_die_id", None))
        if target_die_id is None or not self.d2d_sys:
            return

        # 根据flit类型和请求类型选择AXI通道
        req_type = getattr(flit, "req_type", None)
        flit_type = getattr(flit, "flit_type", None)

        if req_type == "read" and flit_type != "data":
            channel = "AR"  # 读请求使用地址读通道
        elif req_type == "read" and flit_type == "data":
            channel = "R"  # 读数据使用读数据通道
        elif req_type == "write" and flit_type != "data":
            channel = "AW"  # 写请求使用地址写通道
        elif req_type == "write" and flit_type == "data":
            channel = "W"  # 写数据使用写数据通道
        else:
            channel = "AW"  # 默认使用写地址通道

        # print(f"[D2D SN Debug] 跨Die转发: packet_id={getattr(flit, 'packet_id', 'None')}, "
        #   f"req_type={req_type}, flit_type={flit_type}, channel={channel}, target_die={target_die_id}")

        # 使用D2D_Sys进行仲裁和AXI传输
        self.d2d_sys.enqueue_sn(flit, target_die_id, channel)
        self.cross_die_requests_forwarded += 1

    def _handle_received_data(self, flit: Flit):
        """
        重写数据接收处理，支持跨Die数据转发和写数据接收
        """
        # 检查是否为跨Die写数据
        if hasattr(flit, "req_type") and flit.req_type == "write" and hasattr(flit, "flit_type") and flit.flit_type == "data":
            # 这是跨Die写数据，需要处理
            self.handle_cross_die_write_data(flit)
            return

        # 检查是否是跨Die返回的数据
        flit_pos = getattr(flit, "flit_position", "")
        is_cross_die_data = (flit_pos and "AXI" in flit_pos) or (hasattr(flit, "d2d_target_die") and flit.d2d_target_die != self.die_id)

        if is_cross_die_data:
            # 这是从其他Die返回的数据，收集后批量转发到原始请求者
            self.collect_cross_die_read_data(flit)
        else:
            # 本地数据，调用父类正常处理
            super()._handle_received_data(flit)

    def collect_cross_die_read_data(self, flit: Flit):
        """
        收集跨Die返回的读数据到sn_rdb，收齐burst后批量转发
        注意: flit是AXI flit,需要复制到新flit后存储
        """
        packet_id = getattr(flit, "packet_id", -1)
        if packet_id == -1:
            return

        # 使用现有的sn_rdb结构，但需要转换为字典结构
        if not isinstance(self.sn_rdb, dict):
            # 第一次使用时转换为字典
            self.sn_rdb = {}

        if packet_id not in self.sn_rdb:
            self.sn_rdb[packet_id] = []

        # 创建新flit并复制AXI flit的属性（不直接存储AXI flit，因为它会被回收）
        from .flit import create_d2d_flit_copy

        # 创建临时flit用于存储（包含时间戳）
        temp_flit = create_d2d_flit_copy(flit, source=0, destination=0, path=[0], attr_preset="with_timestamp")

        # 添加到sn_rdb
        self.sn_rdb[packet_id].append(temp_flit)

        # 检查是否收齐完整burst
        collected_flits = self.sn_rdb[packet_id]
        expected_length = getattr(flit, "burst_length", 4)

        if len(collected_flits) >= expected_length:
            # 收齐了，批量转发
            self.forward_collected_cross_die_data(packet_id, collected_flits)
            # 清理sn_rdb
            del self.sn_rdb[packet_id]

    def forward_collected_cross_die_data(self, packet_id: int, data_flits: list):
        """
        批量转发收齐的跨Die读数据到原始请求者
        注意: data_flits是从collect_cross_die_read_data创建的temp_flit
        """
        if not data_flits:
            return

        # 使用第一个flit的信息作为参考
        first_flit = data_flits[0]

        # 获取原始请求节点的源映射位置
        original_source_mapped = getattr(first_flit, "d2d_origin_node", None)
        if original_source_mapped is None:
            print(f"[D2D_SN] 警告: flit缺少d2d_origin_node信息: packet_id={packet_id}")
            return

        # 计算目标映射位置
        source_mapped = self.ip_pos  # D2D_SN的位置(36)
        destination_mapped = original_source_mapped

        # 计算路径
        path = self.routes[source_mapped][destination_mapped] if destination_mapped in self.routes[source_mapped] else []

        # 批量转发所有数据flits（已经是独立的flit对象，直接更新属性）
        for i, flit in enumerate(data_flits):
            # 更新路由信息
            flit.source = source_mapped
            flit.destination = destination_mapped
            flit.path = path.copy()
            flit.path_index = 0

            # 设置类型信息
            flit.source_type = self.ip_type  # D2D_SN的类型
            flit.destination_type = getattr(first_flit, "d2d_origin_type", "gdma_0")

            # 标记为新的网络传输
            flit.is_injected = False
            flit.current_position = self.ip_pos

            # 设置发送时间
            flit.departure_cycle = self.current_cycle

            # 通过数据网络发送
            self.enqueue(flit, "data")

        # print(f"[D2D_SN] 批量转发packet {packet_id}的{len(data_flits)}个数据flits到Die{getattr(first_flit, 'd2d_origin_die', '?')}")

        # 发送完成后释放D2D_SN的tracker
        self.release_sn_tracker_for_cross_die_data(packet_id)

    def release_sn_tracker_for_cross_die_data(self, packet_id: int):
        """
        释放D2D_SN用于跨Die数据转发的tracker
        实现正确的tracker释放和等待队列处理
        """
        # 查找对应的tracker
        tracker = next((req for req in self.sn_tracker if req.packet_id == packet_id), None)

        if tracker:
            # 获取正确的tracker类型
            tracker_type = getattr(tracker, "sn_tracker_type", "ro")  # 默认为RO类型

            # 释放tracker资源
            self.sn_tracker.remove(tracker)
            self.sn_tracker_count[tracker_type]["count"] += 1

            # 对于读请求，通常不需要释放RDB（读缓冲由RN管理）
            # 对于写请求，需要释放WDB
            if hasattr(tracker, "req_type") and tracker.req_type == "write":
                self.sn_wdb_count["count"] += tracker.burst_length

            # print(f"[D2D_SN] 释放packet {packet_id}的{tracker_type} tracker资源")

            # 实现retry机制：检查等待队列
            self._process_waiting_requests_after_release(tracker)

    def _process_waiting_requests_after_release(self, completed_req: "Flit"):
        """
        tracker释放后处理等待队列中的请求
        """
        req_type = getattr(completed_req, "req_type", "read")
        wait_list = self.sn_req_wait[req_type]

        if not wait_list:
            return

        # 检查是否有足够资源处理等待的请求
        if req_type == "read":
            # 读请求只需要RO tracker
            if self.sn_tracker_count["ro"]["count"] > 0:
                new_req = wait_list.pop(0)

                # 分配资源并处理
                self.sn_tracker_count["ro"]["count"] -= 1
                new_req.sn_tracker_type = "ro"
                self.sn_tracker.append(new_req)

                # 直接处理请求（不发送positive，直接转发）
                self._handle_cross_die_transfer(new_req)

                # print(f"[D2D_SN] 处理等待队列中的读请求 packet_id={new_req.packet_id}")

        elif req_type == "write":
            # 写请求需要share tracker和WDB
            if self.sn_tracker_count["share"]["count"] > 0 and self.sn_wdb_count["count"] >= wait_list[0].burst_length:

                new_req = wait_list.pop(0)
                new_req.sn_tracker_type = "share"

                # 完全遵循基类逻辑：分配资源并加入tracker
                self.sn_tracker.append(new_req)
                self.sn_tracker_count["share"]["count"] -= 1
                self.sn_wdb_count["count"] -= new_req.burst_length

                # 发送positive响应触发GDMA retry
                positive_rsp = self._create_response_flit(new_req, "positive")
                self.enqueue(positive_rsp, "rsp")

                # print(f"[D2D_SN] 发送positive响应触发写请求retry packet_id={new_req.packet_id}")

    def forward_cross_die_data_to_requester(self, flit: Flit):
        """
        将跨Die返回的数据转发到原始请求者（阶段6）
        """
        # 获取原始请求节点的源映射位置
        original_source_mapped = flit.d2d_origin_node
        if original_source_mapped is None:
            raise ValueError(f"flit缺少d2d_origin_node信息: packet_id={getattr(flit, 'packet_id', 'None')}")

        # 对于D2D跨Die数据返回，需要通过node_map重新映射目标
        # D2D_SN接收到跨Die数据后，应该转发给同位置的GDMA
        source_mapped = self.ip_pos  # D2D_SN的位置(36)

        # d2d_origin_node存储的是源映射位置，转为目标映射位置（减去NUM_COL）
        destination_mapped = original_source_mapped

        # 计算从源映射到目标映射的路径
        # 即使是同一个物理节点，源映射和目标映射也应该不同
        path = self.routes[source_mapped][destination_mapped] if destination_mapped in self.routes[source_mapped] else []

        # 设置路由信息
        flit.source = source_mapped
        flit.destination = destination_mapped  # 同位置的GDMA
        flit.path = path
        flit.path_index = 0

        # 设置第6阶段的类型信息：从D2D_SN发送到GDMA
        flit.source_type = self.ip_type  # D2D_SN的类型 "d2d_sn_0"

        # 设置目标类型为原始请求者的类型
        flit.destination_type = flit.d2d_origin_type

        # 标记为新的网络传输
        flit.is_injected = False
        flit.current_position = self.ip_pos

        # 清除可能影响inject的旧属性
        if hasattr(flit, "departure_cycle"):
            # 立即可发送，不需要等待
            flit.departure_cycle = self.current_cycle

        # 记录D2D_SN处理的数据包
        if not hasattr(self, "sn_data_tracker"):
            self.sn_data_tracker = {}

        packet_id = getattr(flit, "packet_id", -1)
        if packet_id not in self.sn_data_tracker:
            self.sn_data_tracker[packet_id] = {"expected_count": getattr(flit, "burst_length", 4), "forwarded_count": 0, "start_cycle": self.current_cycle}

        # 通过数据网络发送到最终目标
        self.enqueue(flit, "data")
        self.sn_data_tracker[packet_id]["forwarded_count"] += 1

        # 统计
        self.cross_die_data_forwarded = getattr(self, "cross_die_data_forwarded", 0) + 1

        # 检查是否所有数据都已转发
        tracker = self.sn_data_tracker[packet_id]
        if tracker["forwarded_count"] >= tracker["expected_count"]:
            # 所有数据包已转发，清理tracker
            del self.sn_data_tracker[packet_id]

    def handle_cross_die_write_data(self, flit: Flit):
        """
        处理跨Die写数据 - 接收数据并通过AW+W通道转发
        """
        packet_id = flit.packet_id

        # 将数据存储到缓冲区
        if packet_id not in self.sn_wdb:
            self.sn_wdb[packet_id] = []

        # 检查是否已经添加过这个flit（避免重复处理）
        flit_id = getattr(flit, "flit_id", -1)
        existing_flit_ids = [getattr(f, "flit_id", -1) for f in self.sn_wdb[packet_id]]

        if flit_id in existing_flit_ids:
            return  # 跳过重复处理

        self.sn_wdb[packet_id].append(flit)

        # 注意：不在D2D_SN中记录写数据接收，因为这里只是源Die的中转节点
        # 写数据接收应该在真正的目标IP（如目标Die的DDR）中记录

        # 检查是否收集完所有写数据
        collected_flits = self.sn_wdb[packet_id]
        expected_length = flit.burst_length

        if len(collected_flits) >= expected_length:
            # 所有写数据已收集完成，找到对应的写请求
            write_req = next((req for req in self.sn_tracker if req.packet_id == packet_id), None)

            if write_req:
                # 通过AW通道发送写请求，W通道发送写数据
                self.forward_write_request_and_data_cross_die(write_req, collected_flits)

    def forward_write_request_and_data_cross_die(self, write_req: Flit, data_flits: list):
        """
        通过AW和W通道将写请求和写数据转发到目标Die的D2D_RN
        """
        target_die_id = getattr(write_req, "d2d_target_die", None)

        if target_die_id is None or not self.d2d_sys:
            return

        from .flit import create_d2d_flit_copy

        # 创建新的AXI写请求flit
        axi_write_req = create_d2d_flit_copy(write_req, source=self.ip_pos, destination=0, path=[0], attr_preset="request")

        # 保持已有的d2d_sn_node(在eject阶段已设置)
        # create_d2d_flit_copy会复制d2d_sn_node属性

        # 通过AW通道发送写请求（第三个参数传0，让系统自动判断channel）
        self.d2d_sys.enqueue_sn(axi_write_req, target_die_id, 0)

        # 为每个写数据flit创建AXI flit
        for flit in data_flits:
            axi_data_flit = create_d2d_flit_copy(flit, source=self.ip_pos, destination=0, path=[0], attr_preset="data")
            # 确保标记为写数据
            axi_data_flit.flit_type = "data"
            axi_data_flit.req_type = "write"

            # 记录经过的D2D_SN节点
            axi_data_flit.d2d_sn_node = self.ip_pos

            # 通过W通道发送写数据
            self.d2d_sys.enqueue_sn(axi_data_flit, target_die_id, 0)

    def handle_cross_die_write_complete_response(self, flit: Flit):
        """
        处理从B通道返回的写完成响应
        释放D2D_SN的tracker并转发响应给原始RN
        """
        packet_id = flit.packet_id

        # 查找对应的写请求tracker
        write_req = next((req for req in self.sn_tracker if req.packet_id == packet_id), None)

        if write_req:
            # 释放D2D_SN的tracker和资源
            self.sn_tracker.remove(write_req)
            self.sn_tracker_count["share"]["count"] += 1
            self.sn_wdb_count["count"] += write_req.burst_length

            # 清理写数据缓冲
            if packet_id in self.sn_wdb:
                del self.sn_wdb[packet_id]

            # 转发写完成响应给原始RN
            self.forward_response_to_local_rn(flit)

            # 检查等待队列，处理等待的写请求
            self._process_waiting_requests_after_release(write_req)
        else:
            # 没有找到tracker，可能是重复响应，尝试转发一次
            if not hasattr(self, "forwarded_responses"):
                self.forwarded_responses = set()

            if packet_id not in self.forwarded_responses:
                self.forwarded_responses.add(packet_id)
                self.forward_response_to_local_rn(flit)

    def _handle_received_response(self, rsp: Flit):
        """
        重写响应处理，支持B通道写完成响应
        """
        packet_id = getattr(rsp, "packet_id", "?")
        req_type = getattr(rsp, "req_type", "None")
        rsp_type = getattr(rsp, "rsp_type", "None")
        flit_position = getattr(rsp, "flit_position", "None")

        # 只处理从AXI_B通道来的跨Die写完成响应（D2D_SN自己的业务）
        if rsp_type == "write_complete" and flit_position == "AXI_B":
            # 防重复处理
            response_key = (packet_id, self.current_cycle)
            if response_key in self.processed_write_complete_responses:
                return

            # AXI_B通道的原始写完成响应，需要释放D2D_SN的tracker
            self.processed_write_complete_responses[response_key] = True
            self.handle_cross_die_write_complete_response(rsp)
            return

        # 检查req_type有效性
        if req_type in [None, "None"]:
            return

        # 其他所有响应（包括已转发的write_complete），调用父类处理
        super()._handle_received_response(rsp)

    def get_statistics(self) -> dict:
        """获取D2D_SN统计信息"""
        stats = {
            "cross_die_requests_received": self.cross_die_requests_received,
            "cross_die_requests_forwarded": self.cross_die_requests_forwarded,
            "cross_die_responses_sent": self.cross_die_responses_sent,
            "cross_die_data_forwarded": getattr(self, "cross_die_data_forwarded", 0),
            "processed_write_requests_count": len(self.processed_write_requests),
            "processed_write_complete_responses_count": len(self.processed_write_complete_responses),
            "cross_die_receive_queue_size": sum(len(q) for q in self.cross_die_receive_queues.values()),
        }
        return stats
