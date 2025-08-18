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
        self.target_die_interfaces = {}  # 将由D2D_Model设置 {die_id: d2d_rn_interface}

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
        flit_info = f"packet_id={getattr(flit, 'packet_id', '?')}, req_type={getattr(flit, 'req_type', 'None')}, flit_type={getattr(flit, 'flit_type', 'None')}, rsp_type={getattr(flit, 'rsp_type', 'None')}, flit_position={getattr(flit, 'flit_position', 'None')}"
        print(f"[D2D_SN] 接收到跨Die flit: {flit_info}")

        # 优先检查是否为AXI_B的write_complete响应
        if hasattr(flit, "flit_position") and flit.flit_position and "AXI_B" in str(flit.flit_position):
            # AXI_B通道的写完成响应
            print(f"[D2D_SN] 识别为AXI_B写完成响应")
            self.forward_response_to_local_rn(flit)
        elif hasattr(flit, "flit_type") and flit.flit_type == "data":
            # 数据flit：直接处理，不需要资源检查
            print(f"[D2D_SN] 处理跨Die数据flit packet_id={flit.packet_id}")
            self._handle_received_data(flit)
        elif hasattr(flit, "req_type") and flit.req_type:
            # 请求flit：需要资源检查
            if flit.req_type == "read":
                # 读请求：转发到Die内目标SN节点
                print(f"[D2D_SN] 处理跨Die读请求 packet_id={flit.packet_id}")
                self.forward_read_request_to_local_sn(flit)
            elif flit.req_type == "write":
                # 写请求：需要先检查资源并返回data_send响应
                print(f"[D2D_SN] 处理跨Die写请求 packet_id={flit.packet_id}")
                self.handle_cross_die_write_request(flit)
        elif hasattr(flit, "rsp_type") and flit.rsp_type:
            # 响应：转发回Die内原始请求节点
            print(f"[D2D_SN] 处理跨Die响应 packet_id={flit.packet_id}, rsp_type={flit.rsp_type}")
            self.forward_response_to_local_rn(flit)
        else:
            print(f"[D2D_SN] 未知跨Die flit类型，无法处理")

    def forward_read_request_to_local_sn(self, flit: Flit):
        """
        将跨Die读请求转发到本地目标SN节点
        """
        # 更新flit的源和目标信息
        # D2D属性已经包含原始信息，不需要单独保存

        # 设置新的源为D2D_SN节点
        flit.source = self.ip_pos
        # 使用D2D统一属性
        if hasattr(flit, "d2d_target_node"):
            flit.destination = flit.d2d_target_node  # Die内目标节点源映射位置
        else:
            flit.destination = getattr(flit, "target_node_id", self.ip_pos)  # 兼容旧属性

        # 通过请求网络发送
        self.networks["req"]["inject_fifo"].append(flit)
        self.cross_die_requests_forwarded += 1

    def forward_write_request_to_local_sn(self, flit: Flit):
        """
        将跨Die写请求转发到本地目标SN节点
        """
        # 类似读请求处理，使用D2D统一属性

        flit.source = self.ip_pos
        # 使用D2D统一属性
        if hasattr(flit, "d2d_target_node"):
            flit.destination = flit.d2d_target_node  # Die内目标节点源映射位置
        else:
            flit.destination = getattr(flit, "target_node_id", self.ip_pos)  # 兼容旧属性

        # 通过请求网络发送
        self.networks["req"]["inject_fifo"].append(flit)
        self.cross_die_requests_forwarded += 1

    def forward_response_to_local_rn(self, flit: Flit):
        """
        将跨Die响应转发回本地原始请求节点
        """
        # 使用D2D统一属性恢复原始目标信息
        # d2d_origin_node是源映射位置，转为目标映射位置（减去NUM_COL）
        destination = flit.d2d_origin_node - self.config.NUM_COL
        source = self.ip_pos

        # 计算正确的路径
        if destination in self.routes[source]:
            path = self.routes[source][destination]
        else:
            path = [source]  # 如果没有路由，使用源节点作为路径

        # 更新flit信息
        flit.source = source
        flit.destination = destination
        flit.path = path
        flit.path_index = 0

        # 为write_complete响应设置正确的属性
        if hasattr(flit, "rsp_type") and flit.rsp_type == "write_complete":
            flit.req_type = "write"  # 写完成响应需要标记为write类型
            # 设置正确的source_type为原始请求者类型，让RN能找到正确的tracker
            flit.source_type = flit.d2d_origin_type  # 原始请求者类型 (gdma_0)
            
            # 不清除D2D属性，保持完整信息

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
        # 注意：数据包在process_cross_die_receives中已经直接处理，不会进入父类的inject流程
        super().inject_step(cycle)

    def eject_step(self, cycle):
        """
        重写eject_step方法，在eject阶段处理写请求的资源检查和跨Die转发
        """
        # 调用父类的eject_step方法
        ejected_flits = super().eject_step(cycle)

        # 检查ejected的flit
        if ejected_flits:
            for flit in ejected_flits:
                print(f"[D2D_SN] eject flit: packet_id={getattr(flit, 'packet_id', '?')}, req_type={getattr(flit, 'req_type', '?')}, is_cross_die={self._is_cross_die_flit(flit)}")

                if self._is_cross_die_flit(flit):
                    # 检查flit类型：数据flit vs 请求flit
                    if hasattr(flit, "flit_type") and flit.flit_type == "data":
                        # 跨Die写数据：收集数据，最后一笔数据到达时发送AXI AW+W
                        print(f"[D2D_SN] 收到跨Die写数据flit packet_id={flit.packet_id}, flit_id={getattr(flit, 'flit_id', '?')}")
                        self.handle_cross_die_write_data(flit)
                    elif hasattr(flit, "req_type") and flit.req_type == "write":
                        # 跨Die写请求：先进行资源检查并返回data_send响应
                        print(f"[D2D_SN] 处理本地跨Die写请求 packet_id={flit.packet_id}")
                        self.handle_local_cross_die_write_request(flit)
                    else:
                        # 其他类型（如读请求）：直接跨Die转发
                        self._handle_cross_die_transfer(flit)

        return ejected_flits

    def handle_local_cross_die_write_request(self, flit: Flit):
        """
        处理本地接收的跨Die写请求
        先返回data_send响应，收到数据后再跨Die传输
        """
        # 检查D2D_SN的资源
        has_tracker = self.node.sn_tracker_count[self.ip_type]["share"][self.ip_pos] > 0
        has_databuffer = self.node.sn_wdb_count[self.ip_type][self.ip_pos] >= flit.burst_length

        if has_tracker and has_databuffer:
            # 消耗资源
            self.node.sn_tracker_count[self.ip_type]["share"][self.ip_pos] -= 1
            self.node.sn_wdb_count[self.ip_type][self.ip_pos] -= flit.burst_length

            # 记录tracker信息
            flit.sn_tracker_type = "write"
            self.node.sn_tracker[self.ip_type][self.ip_pos].append(flit)

            # 创建data_send响应
            from .flit import Flit

            # 计算正确的目标位置：源映射位置转为目标映射位置
            response_destination = flit.source - self.config.NUM_COL  # 源映射转目标映射

            data_send_rsp = Flit(
                source=self.ip_pos,  # D2D_SN位置(32)
                destination=response_destination,  # GDMA目标映射位置(32)
                path=self.routes[self.ip_pos][response_destination] if response_destination in self.routes[self.ip_pos] else [],
            )
            data_send_rsp.packet_id = flit.packet_id
            data_send_rsp.rsp_type = "datasend"
            data_send_rsp.req_type = "write"  # 设置req_type为write
            data_send_rsp.source_type = self.ip_type
            data_send_rsp.destination_type = getattr(flit, "source_type", "gdma_0")

            # 发送data_send响应
            self.enqueue(data_send_rsp, "rsp")

            # 准备接收写数据的数据结构
            self.node.sn_wdb[self.ip_type][self.ip_pos][flit.packet_id] = []

            print(f"[D2D_SN] 本地跨Die写请求 packet_id={flit.packet_id}，资源检查通过，已发送data_send响应")
        else:
            # 资源不足，发送negative响应
            from .flit import Flit

            # 计算正确的目标位置：源映射位置转为目标映射位置
            response_destination = flit.source - self.config.NUM_COL  # 源映射转目标映射

            negative_rsp = Flit(source=self.ip_pos, destination=response_destination, path=self.routes[self.ip_pos][response_destination] if response_destination in self.routes[self.ip_pos] else [])
            negative_rsp.packet_id = flit.packet_id
            negative_rsp.rsp_type = "negative"
            negative_rsp.req_type = "write"  # 设置req_type为write
            negative_rsp.source_type = self.ip_type
            negative_rsp.destination_type = getattr(flit, "source_type", "gdma_0")

            self.enqueue(negative_rsp, "rsp")
            print(f"[D2D_SN] 本地跨Die写请求 packet_id={flit.packet_id}，资源不足，已发送negative响应")

    def _is_cross_die_flit(self, flit):
        """检查flit是否需要跨Die转发"""
        return (
            hasattr(flit, "d2d_target_die") and hasattr(flit, "d2d_origin_die") and flit.d2d_target_die is not None and flit.d2d_origin_die is not None and flit.d2d_target_die != flit.d2d_origin_die
        )

    def _handle_cross_die_transfer(self, flit):
        """处理跨Die转发（第二阶段：Die0_D2D_SN → Die1_D2D_RN）
        可选择重新生成AXI专用flit，保持packet_id不变
        """
        try:
            # 优先使用新的D2D属性
            if hasattr(flit, "d2d_target_die"):
                target_die_id = flit.d2d_target_die
            else:
                target_die_id = getattr(flit, "target_die_id", None)

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
            if hasattr(self, "d2d_sys") and self.d2d_sys:
                self.d2d_sys.enqueue_sn(flit, target_die_id, channel)
                self.cross_die_requests_forwarded += 1

        except Exception as e:
            import traceback

            print(f"D2D_SN跨Die请求转发错误 [周期{self.current_cycle}, 位置{self.position}]:")
            print(f"  错误类型: {type(e).__name__}")
            print(f"  错误信息: {str(e)}")
            print(f"  Flit信息: source={flit.source}, dest={flit.destination}, dst_die={flit.target_die_id}")
            print(f"  目标Die接口: {getattr(self, 'target_die_d2d_rn', {})}")
            if hasattr(e, "__traceback__"):
                tb_lines = traceback.format_exc().split("\n")
                print(f"  错误位置: {tb_lines[-3].strip() if len(tb_lines) >= 3 else 'N/A'}")

    def handle_local_response_for_cross_die(self, flit: Flit):
        """
        处理本地响应，检查是否需要发送回其他Die
        """
        # 检查是否是对跨Die请求的响应
        # 优先使用新的D2D属性
        if hasattr(flit, "d2d_origin_die") and flit.d2d_origin_die != self.die_id:
            return True
        # 兼容旧属性
        elif hasattr(flit, "source_die_id_physical") and flit.source_die_id_physical != self.die_id:
            return True
        return False

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
        # 判断标准：
        # 1. 有d2d_origin_die属性且与target_die不同，或
        # 2. 来自AXI传输（有axi相关属性）
        is_cross_die_data = False

        # 对于AXI_R数据，应该返回到origin_die
        if hasattr(flit, "flit_position") and flit.flit_position and "AXI_R" in flit.flit_position:
            # AXI_R数据，需要转发到原始请求者
            is_cross_die_data = True
        elif hasattr(flit, "d2d_target_die") and flit.d2d_target_die is not None:
            if flit.d2d_target_die != self.die_id:
                is_cross_die_data = True
        elif hasattr(flit, "flit_position") and flit.flit_position and "AXI" in flit.flit_position:
            # 其他AXI传输数据
            is_cross_die_data = True

        if is_cross_die_data:
            # 这是从其他Die返回的数据，需要转发到原始请求者
            self.forward_cross_die_data_to_requester(flit)
        else:
            # 本地数据，调用父类正常处理
            super()._handle_received_data(flit)

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

        # d2d_origin_node存储的是源映射位置，转为目标映射需要减去NUM_COL
        # 36 (源映射) -> 32 (目标映射)
        destination_mapped = original_source_mapped - self.config.NUM_COL

        # 计算从源映射到目标映射的路径
        # 即使是同一个物理节点，源映射和目标映射也应该不同
        path = self.routes[source_mapped][destination_mapped] if destination_mapped in self.routes[source_mapped] else []

        if source_mapped == destination_mapped:
            # 如果源和目标映射相同，说明映射逻辑有问题
            print(f"[WARNING] D2D_SN路由错误: source_mapped={source_mapped}, destination_mapped={destination_mapped} 相同")
            print(f"  d2d_origin_node={getattr(flit, 'd2d_origin_node', 'None')}, original_source_mapped={original_source_mapped}")

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
        flit.is_new_on_network = True
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

    def handle_cross_die_write_request(self, flit: Flit):
        """
        处理跨Die写请求 - 检查资源并返回data_send响应
        根据D2D写流程文档，D2D_SN需要：
        1. 检查自身资源（tracker/databuffer）
        2. 如果有资源，消耗tracker并返回data_send响应
        3. 接收写数据后通过AW和W通道转发
        """
        # 检查D2D_SN的资源
        has_tracker = self.node.sn_tracker_count[self.ip_type]["share"][self.ip_pos] > 0
        has_databuffer = self.node.sn_wdb_count[self.ip_type][self.ip_pos] >= flit.burst_length

        if has_tracker and has_databuffer:
            # 消耗资源
            self.node.sn_tracker_count[self.ip_type]["share"][self.ip_pos] -= 1
            self.node.sn_wdb_count[self.ip_type][self.ip_pos] -= flit.burst_length

            # 记录tracker信息
            flit.sn_tracker_type = "write"
            self.node.sn_tracker[self.ip_type][self.ip_pos].append(flit)

            # 创建data_send响应
            from .flit import Flit

            data_send_rsp = Flit(source=self.ip_pos, destination=flit.source, path=self.routes[self.ip_pos][flit.source] if flit.source in self.routes[self.ip_pos] else [])  # 返回给请求来源
            data_send_rsp.packet_id = flit.packet_id
            data_send_rsp.rsp_type = "datasend"
            data_send_rsp.source_type = self.ip_type
            data_send_rsp.destination_type = getattr(flit, "source_type", "d2d_rn")

            # 复制D2D属性以便后续处理
            for attr in ["d2d_origin_die", "d2d_origin_node", "d2d_origin_type", "d2d_target_die", "d2d_target_node", "d2d_target_type"]:
                if hasattr(flit, attr):
                    setattr(data_send_rsp, attr, getattr(flit, attr))

            # 发送data_send响应
            self.enqueue(data_send_rsp, "rsp")

            # 准备接收写数据的数据结构
            self.node.sn_wdb[self.ip_type][self.ip_pos][flit.packet_id] = []

            print(f"[D2D_SN] 跨Die写请求 packet_id={flit.packet_id}，资源检查通过，已发送data_send响应")
        else:
            # 资源不足，发送negative响应
            from .flit import Flit

            negative_rsp = Flit(source=self.ip_pos, destination=flit.source, path=self.routes[self.ip_pos][flit.source] if flit.source in self.routes[self.ip_pos] else [])
            negative_rsp.packet_id = flit.packet_id
            negative_rsp.rsp_type = "negative"
            negative_rsp.source_type = self.ip_type
            negative_rsp.destination_type = getattr(flit, "source_type", "d2d_rn")

            self.enqueue(negative_rsp, "rsp")
            print(f"[D2D_SN] 跨Die写请求 packet_id={flit.packet_id}，资源不足，已发送negative响应")

    def handle_cross_die_write_data(self, flit: Flit):
        """
        处理跨Die写数据 - 接收数据并通过AW+W通道转发
        """
        packet_id = flit.packet_id

        # 将数据存储到缓冲区
        if packet_id not in self.node.sn_wdb[self.ip_type][self.ip_pos]:
            self.node.sn_wdb[self.ip_type][self.ip_pos][packet_id] = []

        # 检查是否已经添加过这个flit（避免重复处理）
        flit_id = getattr(flit, "flit_id", -1)
        existing_flit_ids = [getattr(f, "flit_id", -1) for f in self.node.sn_wdb[self.ip_type][self.ip_pos][packet_id]]

        if flit_id not in existing_flit_ids:
            self.node.sn_wdb[self.ip_type][self.ip_pos][packet_id].append(flit)
            print(f"[D2D_SN] 添加数据flit packet_id={packet_id}, flit_id={flit_id}, 当前缓冲区大小={len(self.node.sn_wdb[self.ip_type][self.ip_pos][packet_id])}")
        else:
            print(f"[D2D_SN] 跳过重复的数据flit packet_id={packet_id}, flit_id={flit_id}")
            return  # 跳过重复处理

        # 检查是否收集完所有写数据
        collected_flits = self.node.sn_wdb[self.ip_type][self.ip_pos][packet_id]
        expected_length = flit.burst_length

        if len(collected_flits) >= expected_length:
            # 所有写数据已收集完成，找到对应的写请求
            write_req = next((req for req in self.node.sn_tracker[self.ip_type][self.ip_pos] if req.packet_id == packet_id), None)

            if write_req:
                # 通过AW通道发送写请求，W通道发送写数据
                self.forward_write_request_and_data_cross_die(write_req, collected_flits)
                print(f"[D2D_SN] 跨Die写数据 packet_id={packet_id}，{len(collected_flits)}个flit已通过AW+W通道转发")

    def forward_write_request_and_data_cross_die(self, write_req: Flit, data_flits: list):
        """
        通过AW和W通道将写请求和写数据转发到目标Die的D2D_RN
        """
        target_die_id = getattr(write_req, "d2d_target_die", None)

        if target_die_id is None or target_die_id not in self.target_die_interfaces:
            print(f"[D2D_SN] 错误：无法找到目标Die {target_die_id}的接口")
            return

        # 1. 通过AW通道发送写请求
        if self.d2d_sys:
            self.d2d_sys.enqueue_sn(write_req, target_die_id, self.d2d_aw_latency)

            # 2. 通过W通道发送每个数据flit
            for flit in data_flits:
                self.d2d_sys.enqueue_sn(flit, target_die_id, self.d2d_w_latency)
        else:
            print(f"[D2D_SN] 错误：D2D_Sys未设置")

    def handle_cross_die_write_complete_response(self, flit: Flit):
        """
        处理从B通道返回的写完成响应
        释放D2D_SN的tracker并转发响应给原始RN
        """
        packet_id = flit.packet_id

        # 查找对应的写请求tracker
        write_req = next((req for req in self.node.sn_tracker[self.ip_type][self.ip_pos] if req.packet_id == packet_id), None)

        if write_req:
            # 释放D2D_SN的tracker和资源
            self.node.sn_tracker[self.ip_type][self.ip_pos].remove(write_req)
            self.node.sn_tracker_count[self.ip_type]["share"][self.ip_pos] += 1
            self.node.sn_wdb_count[self.ip_type][self.ip_pos] += write_req.burst_length

            # 清理写数据缓冲
            self.node.sn_wdb[self.ip_type][self.ip_pos].pop(packet_id, None)

            print(f"[D2D_SN] B通道写完成响应 packet_id={packet_id}，释放tracker并转发给原始RN")

            # 转发写完成响应给原始RN
            self.forward_response_to_local_rn(flit)
        else:
            # 没有找到tracker说明已经处理过了，避免重复处理
            print(f"[D2D_SN] 跳过重复的write_complete响应 packet_id={packet_id}")
            return

    def _handle_received_response(self, rsp: Flit):
        """
        重写响应处理，支持B通道写完成响应
        """
        # 添加调试信息
        print(f"[D2D_SN] _handle_received_response 调试: packet_id={getattr(rsp, 'packet_id', '?')}, req_type={getattr(rsp, 'req_type', 'None')}, rsp_type={getattr(rsp, 'rsp_type', 'None')}, source_type={getattr(rsp, 'source_type', 'None')}")
        
        # 检查是否为需要处理的跨Die写完成响应
        is_write_complete = hasattr(rsp, "rsp_type") and rsp.rsp_type == "write_complete"
        is_cross_die = (hasattr(rsp, "d2d_origin_die") and hasattr(rsp, "d2d_target_die") and 
                       rsp.d2d_origin_die is not None and rsp.d2d_target_die is not None)
        # 关键判断：只有source_type不是原始请求者类型的write_complete才需要D2D_SN处理
        # 如果source_type已经是原始请求者类型，说明是转发的响应，应该跳过
        is_from_d2d_rn = (hasattr(rsp, "source_type") and hasattr(rsp, "d2d_origin_type") and 
                         rsp.source_type != rsp.d2d_origin_type)
        
        if is_write_complete and is_cross_die and is_from_d2d_rn:
            # 这是从B通道返回的写完成响应（不是转发的）
            # 首先处理D2D_SN自己的SN tracker
            self.handle_cross_die_write_complete_response(rsp)
            return
        elif is_write_complete and is_cross_die and not is_from_d2d_rn:
            # 这是转发的write_complete响应，不需要D2D_SN处理，直接跳过
            print(f"[D2D_SN] 跳过转发的write_complete响应 packet_id={getattr(rsp, 'packet_id', '?')}")
            return

        # 检查req_type是否为None，避免KeyError
        if rsp.req_type is None:
            print(f"[D2D_SN] 警告：响应req_type为None，跳过处理 rsp_type={getattr(rsp, 'rsp_type', 'None')}")
            return

        # 其他响应类型，调用父类处理
        print(f"[D2D_SN] 调用父类处理响应: req_type={rsp.req_type}, self.ip_type={self.ip_type}")
        super()._handle_received_response(rsp)
