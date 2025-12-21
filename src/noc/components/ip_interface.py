"""
IPInterface class for NoC simulation.
Handles IP interface functionality including frequency conversion, OSTD/Data-buffer behavior,
and network interaction through inject/eject FIFOs.
"""

from __future__ import annotations
import numpy as np
from collections import deque, defaultdict
from config.config import CrossRingConfig
from src.utils.flit import (
    Flit,
    TokenBucket,
    get_original_source_node,
    get_original_destination_node,
    get_original_source_type,
    get_original_destination_type,
)
import logging
import inspect

# Response优先级顺序（列表顺序即优先级，靠前优先级高）
RSP_PRIORITY_ORDER = [
    "datasend",
    "negative",
    "write_complete",
    "positive",
]


# IPInterface的Ring支持扩展
class RingIPInterface:
    """
    为IPInterface添加Ring支持的Mixin类
    """

    def __init__(self, ip_type, ip_pos, config, req_network, rsp_network, data_network, routes, ring_mode=False):
        # 调用原有的IPInterface初始化
        super().__init__(ip_type, ip_pos, config, req_network, rsp_network, data_network, routes)

        self.ring_mode = ring_mode
        self.ring_model = None  # 将由Ring模型设置
        if ring_mode:
            self._init_ring_specific_features()

    def _init_ring_specific_features(self):
        """初始化Ring特有的IPInterface功能"""
        # Ring模式下的特殊配置
        self.ring_config = {"adaptive_injection": True, "load_balancing": True, "etag_enabled": True, "itag_enabled": True}


class IPInterface:
    """
    模拟一个 IP 的频率转化和 OSTD/Data-buffer 行为:
      - inject_fifo: 1GHz 入队
      - l2h_fifo: 1→2GHz 上转换 FIFO（深度可调）
      - 和 network.IQ_channel_buffer 对接（2GHz）
      - network.EQ_channel_buffer → h2l_fifo → eject_fifo → IP（1GHz）
    """

    def __init__(
        self,
        ip_type: str,
        ip_pos: int,
        config: CrossRingConfig,
        req_network,
        rsp_network,
        data_network,
        routes: dict,
        ip_id: int = None,
    ):
        self.current_cycle = 0
        self.ip_type = ip_type
        self.ip_pos = ip_pos
        self.ip_id = ip_id if ip_id is not None else 0  # IP在其类型中的唯一ID
        self.config = config
        self.req_network = req_network
        self.rsp_network = rsp_network
        self.data_network = data_network
        self.routes = routes
        self.read_retry_num_stat = 0
        self.write_retry_num_stat = 0
        self.req_wait_cycles_h = 0
        self.req_wait_cycles_v = 0
        self.rsp_wait_cycles_h = 0
        self.rsp_wait_cycles_v = 0
        self.data_wait_cycles_h = 0
        self.data_wait_cycles_v = 0
        self.req_cir_h_num, self.req_cir_v_num = 0, 0
        self.rsp_cir_h_num, self.rsp_cir_v_num = 0, 0
        self.data_cir_h_num, self.data_cir_v_num = 0, 0
        # 反方向上环统计
        self.req_reverse_h_num, self.req_reverse_v_num = 0, 0
        self.rsp_reverse_h_num, self.rsp_reverse_v_num = 0, 0
        self.data_reverse_h_num, self.data_reverse_v_num = 0, 0

        # 验证FIFO深度配置
        l2h_depth = config.IP_L2H_FIFO_DEPTH
        h2l_h_depth = config.IP_H2L_H_FIFO_DEPTH
        h2l_l_depth = config.IP_H2L_L_FIFO_DEPTH

        # 处理延迟配置
        self.sn_processing_latency = getattr(config, "SN_PROCESSING_LATENCY", 0)
        self.rn_processing_latency = getattr(config, "RN_PROCESSING_LATENCY", 0)

        self.networks = {
            "req": {
                "network": req_network,
                "send_flits": req_network.send_flits,
                "inject_fifo": deque(),  # 1GHz域 - IP核产生的请求
                "l2h_fifo_pre": None,  # 1GHz → 2GHz 预缓冲
                "h2l_fifo_h_pre": None,  # 2GHz → H2L_H 预缓冲
                "h2l_fifo_l_pre": None,  # H2L_H → H2L_L 预缓冲
                "l2h_fifo": deque(maxlen=l2h_depth),  # 1GHz → 2GHz 转换FIFO
                "h2l_fifo_h": deque(maxlen=h2l_h_depth),  # 2GHz域 高频FIFO
                "h2l_fifo_l": deque(maxlen=h2l_l_depth),  # 1GHz域 低频FIFO
            },
            "rsp": {
                "network": rsp_network,
                "send_flits": rsp_network.send_flits,
                "inject_fifo": deque(),
                "l2h_fifo_pre": None,
                "h2l_fifo_h_pre": None,
                "h2l_fifo_l_pre": None,
                "l2h_fifo": deque(maxlen=l2h_depth),
                "h2l_fifo_h": deque(maxlen=h2l_h_depth),
                "h2l_fifo_l": deque(maxlen=h2l_l_depth),
            },
            "data": {
                "network": data_network,
                "send_flits": data_network.send_flits,
                "inject_fifo": deque(),
                "l2h_fifo_pre": None,
                "h2l_fifo_h_pre": None,
                "h2l_fifo_l_pre": None,
                "l2h_fifo": deque(maxlen=l2h_depth),
                "h2l_fifo_h": deque(maxlen=h2l_h_depth),
                "h2l_fifo_l": deque(maxlen=h2l_l_depth),
            },
        }

        # Response优先级队列（用于rsp网络，按优先级分别存储）
        self.rsp_inject_queues = {rsp_type: deque() for rsp_type in RSP_PRIORITY_ORDER}

        # 根据IP类型设置双向带宽限制令牌桶
        if ip_type.startswith("ddr"):
            # DDR通道限速
            tx_limit = rx_limit = self.config.DDR_BW_LIMIT
            self.tx_token_bucket = TokenBucket(
                rate=tx_limit / self.config.NETWORK_FREQUENCY / self.config.FLIT_SIZE,
                bucket_size=tx_limit,
            )
            self.rx_token_bucket = TokenBucket(
                rate=rx_limit / self.config.NETWORK_FREQUENCY / self.config.FLIT_SIZE,
                bucket_size=rx_limit,
            )
        elif ip_type.startswith("l2m"):
            # L2M通道限速
            tx_limit = rx_limit = self.config.L2M_BW_LIMIT
            self.tx_token_bucket = TokenBucket(
                rate=tx_limit / self.config.NETWORK_FREQUENCY / self.config.FLIT_SIZE,
                bucket_size=tx_limit,
            )
            self.rx_token_bucket = TokenBucket(
                rate=rx_limit / self.config.NETWORK_FREQUENCY / self.config.FLIT_SIZE,
                bucket_size=rx_limit,
            )
        elif ip_type[:4] in ("sdma", "gdma", "cdma"):
            # DMA通道（SDMA/GDMA/CDMA）限速
            ip_prefix = ip_type[:4].upper()
            tx_limit = rx_limit = getattr(self.config, f"{ip_prefix}_BW_LIMIT", 128)
            self.tx_token_bucket = TokenBucket(
                rate=tx_limit / self.config.FLIT_SIZE,
                bucket_size=tx_limit,
            )
            self.rx_token_bucket = TokenBucket(
                rate=rx_limit / self.config.FLIT_SIZE,
                bucket_size=rx_limit,
            )
        else:
            # 默认不限速
            self.tx_token_bucket = None
            self.rx_token_bucket = None

        # ========== RN资源管理（每个IP独立管理） ==========
        # RN Tracker
        self.rn_tracker = {"read": [], "write": []}
        self.rn_tracker_count = {"read": {"count": config.RN_R_TRACKER_OSTD}, "write": {"count": config.RN_W_TRACKER_OSTD}}
        self.rn_tracker_pointer = {"read": -1, "write": -1}
        self.rn_tracker_wait = {"read": [], "write": []}

        # RN Data Buffer
        self.rn_rdb = {}  # 读数据缓冲 {packet_id: [flits]}
        self.rn_wdb = {}  # 写数据缓冲 {packet_id: [flits]}

        # 统一tracker模式下，读写共享同一个字典对象
        if config.UNIFIED_RW_TRACKER:
            shared_tracker = {"count": config.RN_R_TRACKER_OSTD}
            shared_db = {"count": config.RN_RDB_SIZE}
            self.rn_tracker_count["read"] = shared_tracker
            self.rn_tracker_count["write"] = shared_tracker
            self.rn_rdb_count = shared_db
            self.rn_wdb_count = shared_db
        else:
            self.rn_rdb_count = {"count": config.RN_RDB_SIZE}
            self.rn_wdb_count = {"count": config.RN_WDB_SIZE}

        self.rn_rdb_reserve = 0
        self.rn_wdb_reserve = 0
        self.rn_rdb_recv = []
        self.rn_wdb_send = []

        # ========== SN资源管理（每个IP独立管理） ==========
        self.sn_tracker = []
        self.sn_req_wait = {"read": [], "write": []}
        self.sn_rdb = []
        self.sn_wdb = {}  # {packet_id: [flits]}
        self.sn_wdb_recv = []
        self.sn_rsp_queue = []
        self.sn_tracker_release_time = defaultdict(list)

        # 根据IP类型设置SN tracker数量
        if ip_type.startswith("ddr"):
            self.sn_tracker_count = {"ro": {"count": config.SN_DDR_R_TRACKER_OSTD}, "share": {"count": config.SN_DDR_W_TRACKER_OSTD}}
            self.sn_wdb_count = {"count": config.SN_DDR_WDB_SIZE}
        elif ip_type.startswith("l2m"):
            self.sn_tracker_count = {"ro": {"count": config.SN_L2M_R_TRACKER_OSTD}, "share": {"count": config.SN_L2M_W_TRACKER_OSTD}}
            self.sn_wdb_count = {"count": config.SN_L2M_WDB_SIZE}
        elif ip_type.startswith("d2d_sn"):
            # D2D_SN使用专用配置
            self.sn_tracker_count = {
                "ro": {"count": getattr(config, "D2D_SN_R_TRACKER_OSTD", config.SN_DDR_R_TRACKER_OSTD)},
                "share": {"count": getattr(config, "D2D_SN_W_TRACKER_OSTD", config.SN_DDR_W_TRACKER_OSTD)},
            }
            self.sn_wdb_count = {"count": getattr(config, "D2D_SN_WDB_SIZE", config.SN_DDR_WDB_SIZE)}
        else:
            # DMA类IP通常不作为SN
            self.sn_tracker_count = {"ro": {"count": 0}, "share": {"count": 0}}
            self.sn_wdb_count = {"count": 0}

        # 统一tracker模式下，SN的读写tracker共享
        if config.UNIFIED_RW_TRACKER and ip_type.startswith(("ddr", "l2m", "d2d_sn")):
            shared_sn_tracker = {"count": self.sn_tracker_count["ro"]["count"]}
            shared_sn_wdb = self.sn_wdb_count
            self.sn_tracker_count["ro"] = shared_sn_tracker
            self.sn_tracker_count["share"] = shared_sn_tracker
            self.sn_wdb_count = shared_sn_wdb

        # D2D_RN资源（如果是d2d_rn类型）
        if ip_type.startswith("d2d_rn"):
            # 覆盖RN资源配置为D2D_RN专用
            rdb_size = getattr(config, "D2D_RN_RDB_SIZE", config.RN_RDB_SIZE)
            wdb_size = getattr(config, "D2D_RN_WDB_SIZE", config.RN_WDB_SIZE)
            r_tracker_count = getattr(config, "D2D_RN_R_TRACKER_OSTD", config.RN_R_TRACKER_OSTD)
            w_tracker_count = getattr(config, "D2D_RN_W_TRACKER_OSTD", config.RN_W_TRACKER_OSTD)

            if config.UNIFIED_RW_TRACKER:
                shared_tracker = {"count": r_tracker_count}
                shared_db = {"count": rdb_size}
                self.rn_tracker_count["read"] = shared_tracker
                self.rn_tracker_count["write"] = shared_tracker
                self.rn_rdb_count = shared_db
                self.rn_wdb_count = shared_db
            else:
                self.rn_tracker_count["read"]["count"] = r_tracker_count
                self.rn_tracker_count["write"]["count"] = w_tracker_count
                self.rn_rdb_count["count"] = rdb_size
                self.rn_wdb_count["count"] = wdb_size

        # ========== Tracker使用情况记录（事件驱动） ==========
        self.tracker_events = {
            "rn_read": {"allocations": [], "releases": []},
            "rn_write": {"allocations": [], "releases": []},
            "rn_rdb": {"allocations": [], "releases": []},
            "rn_wdb": {"allocations": [], "releases": []},
            "sn_ro": {"allocations": [], "releases": []},
            "sn_share": {"allocations": [], "releases": []},
            "sn_wdb": {"allocations": [], "releases": []},
            "read_retry": {"allocations": [], "releases": []},
            "write_retry": {"allocations": [], "releases": []},
        }

        self.tracker_total_allocated = {"rn_read": 0, "rn_write": 0, "rn_rdb": 0, "rn_wdb": 0, "sn_ro": 0, "sn_share": 0, "sn_wdb": 0, "read_retry": 0, "write_retry": 0}

        self.tracker_block_events = []

        self.tracker_total_config = {
            "rn_read": self.rn_tracker_count["read"]["count"],
            "rn_write": self.rn_tracker_count["write"]["count"],
            "rn_rdb": self.rn_rdb_count["count"],
            "rn_wdb": self.rn_wdb_count["count"],
            "sn_ro": self.sn_tracker_count["ro"]["count"],
            "sn_share": self.sn_tracker_count["share"]["count"],
            "sn_wdb": self.sn_wdb_count["count"],
            "read_retry": None,
            "write_retry": None,
        }

    def enqueue(self, flit: Flit, network_type: str, retry=False):
        """IP 核把flit丢进对应网络的 inject_fifo"""
        if network_type == "rsp":
            # Response按优先级分流到对应队列
            rsp_type = getattr(flit, "rsp_type", "positive")
            if rsp_type not in self.rsp_inject_queues:
                rsp_type = "positive"  # 未知类型默认最低优先级
            if retry:
                self.rsp_inject_queues[rsp_type].appendleft(flit)
            else:
                self.rsp_inject_queues[rsp_type].append(flit)
        else:
            if retry:
                self.networks[network_type]["inject_fifo"].appendleft(flit)
            else:
                self.networks[network_type]["inject_fifo"].append(flit)
        flit.set_position("IP_inject", self.current_cycle)

        # 检查是否为新请求并记录统计
        # 新请求必须满足：1) 请求网络 2) 本IP首次见到此packet_id 3) 非重试 4) 当前IP是原始发起IP
        is_new_request = network_type == "req" and not self.networks[network_type]["send_flits"][flit.packet_id] and not retry and hasattr(flit, "source_type") and flit.source_type == self.ip_type

        if is_new_request and hasattr(flit, "req_type"):
            # 记录请求开始时间（tracker消耗开始）
            flit.req_start_cycle = self.current_cycle

            # 判断是否为跨Die请求
            is_cross_die = hasattr(flit, "d2d_target_die") and hasattr(flit, "d2d_origin_die") and flit.d2d_target_die != flit.d2d_origin_die

            # 记录请求统计
            d2d_model = getattr(self.req_network, "d2d_model", None)
            if d2d_model:
                die_id = getattr(self.config, "DIE_ID", 0)
                d2d_model.record_request_issued(flit.packet_id, die_id, flit.req_type, is_cross_die)

        if network_type == "req" and self.networks[network_type]["send_flits"][flit.packet_id]:
            return True
        self.networks[network_type]["send_flits"][flit.packet_id].append(flit)

        # 注：flit添加到RequestTracker的操作移至接收端（避免重复添加）
        # request_flit在process_req_from_network时添加
        # data_flit在process_data_from_network时添加

        return True

    def inject_to_l2h_pre(self, network_type):
        """1GHz: inject_fifo → l2h_fifo_pre（分网络处理）"""
        net_info = self.networks[network_type]

        # 检查l2h_fifo是否有空间
        if len(net_info["l2h_fifo"]) >= net_info["l2h_fifo"].maxlen or net_info["l2h_fifo_pre"] is not None:
            return

        # rsp网络使用优先级队列，其他网络使用inject_fifo
        if network_type == "rsp":
            # 响应网络：按优先级顺序检查各队列
            flit = None
            current_cycle = getattr(self, "current_cycle", 0)
            for rsp_type in RSP_PRIORITY_ORDER:
                queue = self.rsp_inject_queues[rsp_type]
                if queue:
                    candidate = queue[0]
                    if hasattr(candidate, "departure_cycle") and candidate.departure_cycle > current_cycle:
                        continue  # 还没到发送时间，检查下一优先级
                    flit = queue.popleft()
                    break

            if flit is None:
                return  # 所有队列都空或都未到发送时间

            flit.set_position("L2H", self.current_cycle)
            flit.start_inject = True
            net_info["l2h_fifo_pre"] = flit
            return

        # req和data网络检查inject_fifo
        if not net_info["inject_fifo"]:
            return

        flit = net_info["inject_fifo"][0]

        # 根据网络类型进行不同的处理
        if network_type == "req":
            if flit.req_attr == "new" and not self._check_and_reserve_resources(flit):
                return  # 资源不足，保持在inject_fifo中
            flit.set_position("L2H", self.current_cycle)
            flit.start_inject = True
            net_info["l2h_fifo_pre"] = net_info["inject_fifo"].popleft()

        elif network_type == "data":
            # 数据网络：检查departure_cycle
            current_cycle = getattr(self, "current_cycle", 0)
            if hasattr(flit, "departure_cycle") and flit.departure_cycle > current_cycle:
                return  # 还没到发送时间
            if self.tx_token_bucket:
                self.tx_token_bucket.refill(current_cycle)
                if not self.tx_token_bucket.consume():
                    return
            flit: Flit = net_info["inject_fifo"].popleft()
            flit.set_position("L2H", current_cycle)
            flit.start_inject = True
            net_info["l2h_fifo_pre"] = flit

            # 读数据最后一个flit进入l2h时释放SN tracker
            if flit.req_type == "read" and getattr(flit, "is_last_flit", False):
                self._release_sn_read_tracker(flit.packet_id)

    def l2h_to_IQ_channel_buffer(self, network_type):
        """2GHz: l2h_fifo → network.IQ_channel_buffer"""
        net_info = self.networks[network_type]
        network = net_info["network"]

        if not net_info["l2h_fifo"]:
            return

        # 检查目标缓冲区是否已满（只能在 *pre* 缓冲区为空且正式 FIFO 未满时移动）
        fifo = network.IQ_channel_buffer[self.ip_type][self.ip_pos]
        fifo_pre = network.IQ_channel_buffer_pre[self.ip_type][self.ip_pos]
        if len(fifo) >= getattr(self.config, "IQ_CH_FIFO_DEPTH", 8) or fifo_pre is not None:
            return  # 没空间，或 pre 槽已占用

        # 从 l2h_fifo 弹出一个 flit，先放到 *pre* 槽
        flit: Flit = net_info["l2h_fifo"].popleft()
        flit.set_position("IQ_CH", self.current_cycle)
        network.IQ_channel_buffer_pre[self.ip_type][self.ip_pos] = flit

        # 更新cycle统计
        if network_type == "req" and flit.req_attr == "new":
            flit.cmd_entry_noc_from_cake0_cycle = self.current_cycle
            # 更新RequestTracker
            if hasattr(self, "request_tracker") and self.request_tracker:
                self.request_tracker.update_timestamp(flit.packet_id, "cmd_entry_noc_from_cake0_cycle", self.current_cycle)
        elif network_type == "rsp":
            flit.cmd_entry_noc_from_cake1_cycle = self.current_cycle
            # 更新RequestTracker
            if hasattr(self, "request_tracker") and self.request_tracker:
                self.request_tracker.update_timestamp(flit.packet_id, "cmd_entry_noc_from_cake1_cycle", self.current_cycle)
        elif network_type == "data":
            # 只为第一个data flit设置entry时间戳
            if flit.req_type == "read" and flit.flit_id == 0:
                flit.data_entry_noc_from_cake1_cycle = self.current_cycle
                # 更新RequestTracker
                if hasattr(self, "request_tracker") and self.request_tracker:
                    self.request_tracker.update_timestamp(flit.packet_id, "data_entry_noc_from_cake1_cycle", self.current_cycle)
            elif flit.req_type == "write" and flit.flit_id == 0:
                flit.data_entry_noc_from_cake0_cycle = self.current_cycle
                # 更新RequestTracker
                if hasattr(self, "request_tracker") and self.request_tracker:
                    self.request_tracker.update_timestamp(flit.packet_id, "data_entry_noc_from_cake0_cycle", self.current_cycle)

    def _check_and_reserve_resources(self, req):
        """检查并预占新请求所需的资源"""
        ip_pos, ip_type = self.ip_pos, self.ip_type

        try:
            if req.req_type == "read":
                # 如果是retry请求（req_attr="old"），资源已在第一次分配，直接返回
                if req.req_attr == "old" and req.packet_id in self.rn_rdb:
                    # Retry请求：tracker和databuffer已在第一次分配
                    return True

                # 检查读请求资源（仅新请求）
                rdb_available = self.rn_rdb_count["count"] >= req.burst_length
                tracker_available = self.rn_tracker_count["read"]["count"] > 0
                reserve_ok = self.rn_rdb_count["count"] > self.rn_rdb_reserve * req.burst_length

                if not (rdb_available and tracker_available and reserve_ok):
                    return False

                # 预占资源（仅新请求）
                self.rn_rdb_count["count"] -= req.burst_length
                self.rn_tracker_count["read"]["count"] -= 1
                self._record_tracker_allocation("rn_read")
                self._record_tracker_allocation("rn_rdb")
                self.rn_rdb[req.packet_id] = []
                req.cmd_entry_cake0_cycle = self.current_cycle  # 这里记录cycle

                # 更新RequestTracker的timestamp
                if hasattr(self, "request_tracker") and self.request_tracker:
                    self.request_tracker.update_timestamp(req.packet_id, "cmd_entry_cake0_cycle", self.current_cycle)

                self.rn_tracker["read"].append(req)
                self.rn_tracker_pointer["read"] += 1

            elif req.req_type == "write":
                # 如果是retry请求（req_attr="old"），资源已在第一次分配，直接返回
                if req.req_attr == "old" and req.packet_id in self.rn_wdb:
                    # Retry请求：tracker和databuffer已在第一次分配
                    return True

                # 检查写请求资源（仅新请求）
                wdb_available = self.rn_wdb_count["count"] >= req.burst_length
                tracker_available = self.rn_tracker_count["write"]["count"] > 0

                if not (wdb_available and tracker_available):
                    return False

                # 预占资源（仅新请求）
                self.rn_wdb_count["count"] -= req.burst_length
                self.rn_tracker_count["write"]["count"] -= 1
                self._record_tracker_allocation("rn_write")
                self._record_tracker_allocation("rn_wdb")
                self.rn_wdb[req.packet_id] = []
                req.cmd_entry_cake0_cycle = self.current_cycle  # 这里记录cycle

                # 更新RequestTracker的timestamp
                if hasattr(self, "request_tracker") and self.request_tracker:
                    self.request_tracker.update_timestamp(req.packet_id, "cmd_entry_cake0_cycle", self.current_cycle)

                self.rn_tracker["write"].append(req)
                self.rn_tracker_pointer["write"] += 1

                # 写数据包将在收到 datasend 响应时创建

            return True

        except (KeyError, AttributeError) as e:
            logging.warning(f"Resource check failed for {req}: {e}")
            return False

    def EQ_channel_buffer_to_h2l_pre(self, network_type):
        """2GHz: network.EQ_channel_buffer → h2l_fifo_h_pre"""
        net_info = self.networks[network_type]
        network = net_info["network"]

        if net_info["h2l_fifo_h_pre"] is not None:
            return  # 预缓冲已占用

        try:
            # 新架构：直接使用 ip_pos
            pos_index = self.ip_pos

            eq_buf = network.EQ_channel_buffer[self.ip_type][pos_index]
            if not eq_buf:
                return

            # 检查h2l_fifo_h是否有空间
            if len(net_info["h2l_fifo_h"]) >= net_info["h2l_fifo_h"].maxlen:
                return

            flit = eq_buf.popleft()
            flit.is_arrive = True
            flit.set_position("H2L_H", self.current_cycle)
            net_info["h2l_fifo_h_pre"] = flit
            # arrive_flits添加移动到IP_eject阶段，确保只记录真正完成的flit

        except (KeyError, AttributeError) as e:
            logging.warning(f"EQ to h2l_h transfer failed for {network_type}: {e}")

    def h2l_h_to_h2l_l_pre(self, network_type):
        """网络频率: h2l_fifo_h → h2l_fifo_l_pre"""
        net_info = self.networks[network_type]

        if net_info["h2l_fifo_l_pre"] is not None:
            return  # L级预缓冲已占用

        if not net_info["h2l_fifo_h"]:
            return  # H级FIFO为空

        # 检查L级FIFO是否有空间
        if len(net_info["h2l_fifo_l"]) >= net_info["h2l_fifo_l"].maxlen:
            return

        # 从H级FIFO传输到L级预缓冲（不设置延迟，直接传输）
        flit = net_info["h2l_fifo_h"].popleft()
        flit.set_position("H2L_L", self.current_cycle)
        net_info["h2l_fifo_l_pre"] = flit

    def _handle_received_request(self, req: Flit):
        """处理接收到的请求（SN端）"""
        # 只有SN-side IP类型可以处理接收到的请求
        if not (self.ip_type.startswith("ddr") or self.ip_type.startswith("l2m")):
            # RN-side IP类型不应该接收请求，直接忽略
            return

        req.cmd_received_by_cake1_cycle = getattr(self, "current_cycle", 0)
        self.req_wait_cycles_h += req.wait_cycle_h
        self.req_wait_cycles_v += req.wait_cycle_v
        self.req_cir_h_num += req.eject_attempts_h
        self.req_cir_v_num += req.eject_attempts_v
        self.req_reverse_h_num += req.reverse_inject_h
        self.req_reverse_v_num += req.reverse_inject_v
        req.cmd_received_by_cake1_cycle = self.current_cycle

        # 添加到RequestTracker
        if hasattr(self, "request_tracker") and self.request_tracker:
            self.request_tracker.add_request_flit(req.packet_id, req)
            self.request_tracker.update_timestamp(req.packet_id, "cmd_received_by_cake1_cycle", self.current_cycle)

        if req.req_type == "read":
            if req.req_attr == "new":
                if self.sn_tracker_count["ro"]["count"] > 0:
                    req.sn_tracker_type = "ro"
                    self.sn_tracker.append(req)
                    self.sn_tracker_count["ro"]["count"] -= 1
                    self._record_tracker_allocation("sn_ro")
                    self.create_read_packet(req)
                    # tracker释放移至inject_to_l2h_pre，在最后一个data flit进入l2h时释放
                else:
                    self.create_rsp(req, "negative")
                    self._record_tracker_block("sn_ro")
                    self.sn_req_wait[req.req_type].append(req)
            else:
                self.create_read_packet(req)
                # tracker释放移至inject_to_l2h_pre，在最后一个data flit进入l2h时释放

        elif req.req_type == "write":
            if req.req_attr == "new":
                if self.sn_tracker_count["share"]["count"] > 0 and self.sn_wdb_count["count"] >= req.burst_length:
                    req.sn_tracker_type = "share"
                    self.sn_tracker.append(req)
                    self.sn_tracker_count["share"]["count"] -= 1
                    self.sn_wdb[req.packet_id] = []
                    self.sn_wdb_count["count"] -= req.burst_length
                    self._record_tracker_allocation("sn_share")
                    self._record_tracker_allocation("sn_wdb")
                    self.create_rsp(req, "datasend")
                else:
                    self.create_rsp(req, "negative")
                    self._record_tracker_block("sn_share")
                    self.sn_req_wait[req.req_type].append(req)
            else:
                self.create_rsp(req, "datasend")

    def _handle_received_response(self, rsp: Flit):
        """处理接收到的响应（RN端）"""
        rsp.cmd_received_by_cake1_cycle = getattr(self, "current_cycle", 0)
        self.rsp_wait_cycles_h += rsp.wait_cycle_h
        self.rsp_wait_cycles_v += rsp.wait_cycle_v
        self.rsp_cir_h_num += rsp.eject_attempts_h
        self.rsp_cir_v_num += rsp.eject_attempts_v
        self.rsp_reverse_h_num += rsp.reverse_inject_h
        self.rsp_reverse_v_num += rsp.reverse_inject_v
        rsp.cmd_received_by_cake0_cycle = self.current_cycle

        # 添加到RequestTracker
        if hasattr(self, "request_tracker") and self.request_tracker:
            self.request_tracker.add_response_flit(rsp.packet_id, rsp)
            self.request_tracker.update_timestamp(rsp.packet_id, "cmd_received_by_cake0_cycle", self.current_cycle)
        if rsp.req_type == "read" and rsp.rsp_type == "negative":
            self.read_retry_num_stat += 1
        elif rsp.req_type == "write" and rsp.rsp_type == "negative":
            self.write_retry_num_stat += 1

        req = next((req for req in self.rn_tracker[rsp.req_type] if req.packet_id == rsp.packet_id), None)

        if not req:
            return
        req.sync_latency_record(rsp)

        if rsp.req_type == "read":
            if rsp.rsp_type == "negative":
                # 读请求retry逻辑
                if req.req_attr == "old":
                    # 该请求已经被重新注入网络，不需要再次修改。直接返回
                    return
                req.req_state = "invalid"
                req.is_injected = False
                req.path_index = 0
                req.req_attr = "old"
                # 保留rn_rdb和databuffer资源，等待positive后重试
                # databuffer已在第一次分配，retry时不需要重复分配和释放

            elif rsp.rsp_type == "positive":
                # 处理读重试：资源已在第一次分配，直接重新发送请求
                req.req_state = "valid"
                req.req_attr = "old"
                req.is_injected = False
                req.path_index = 0
                req.is_arrive = False
                # 重置order_id，让网络层重新分配
                req.src_dest_order_id = -1
                # 放入请求网络的inject_fifo
                self.enqueue(req, "req", retry=True)

        elif rsp.req_type == "write":
            if rsp.rsp_type == "negative":
                # 写请求retry逻辑
                if req.req_attr == "old":
                    # 该请求已经被重新注入网络，不需要再次修改。直接返回
                    return
                req.req_state = "invalid"
                req.req_attr = "old"
                req.is_injected = False
                req.path_index = 0

            elif rsp.rsp_type == "positive":
                # 处理写重试
                req.req_state = "valid"
                req.is_injected = False
                req.path_index = 0
                req.req_attr = "old"
                req.is_arrive = False
                # 重置order_id，让网络层重新分配
                req.src_dest_order_id = -1
                # 放入请求网络的inject_fifo
                self.enqueue(req, "req", retry=True)

            elif rsp.rsp_type == "datasend":
                # 收到写数据发送响应，现在创建并发送写数据包
                req: Flit = next((r for r in self.rn_tracker["write"] if r.packet_id == rsp.packet_id), None)
                if req:
                    # 创建写数据包
                    self.create_write_packet(req)
                    # 将写数据包放入数据网络inject_fifo
                    for flit in self.rn_wdb[rsp.packet_id]:
                        self.enqueue(flit, "data")

                    # 检查是否为跨Die写请求
                    is_cross_die_write = self._is_cross_die_write_request(req)

                    if is_cross_die_write:
                        # 跨Die写请求：发送数据但不释放tracker，等待B通道响应
                        pass
                    else:
                        # 普通Die内写请求：发送数据并立即释放tracker
                        self.rn_tracker["write"].remove(req)
                        self.rn_wdb_count["count"] += req.burst_length
                        self.rn_tracker_count["write"]["count"] += 1
                        self._record_tracker_release("rn_write")
                        self._record_tracker_release("rn_wdb")
                        self.rn_tracker_pointer["write"] -= 1

                # 同时清理写缓冲
                self.rn_wdb.pop(rsp.packet_id, None)

            elif rsp.rsp_type == "write_complete":
                # 收到写完成响应，查找对应的写请求（包括本地和跨Die写请求）
                req: Flit = next((r for r in self.rn_tracker["write"] if r.packet_id == rsp.packet_id), None)
                if req:
                    # 记录写完成响应接收时间（所有写请求都需要记录）
                    req.write_complete_received_cycle = self.current_cycle
                    req.data_received_complete_cycle = self.current_cycle  # 写完成即数据接收完成

                    # 标记写请求完成
                    if hasattr(self, "request_tracker") and self.request_tracker:
                        self.request_tracker.update_timestamp(rsp.packet_id, "write_complete_received_cycle", self.current_cycle)
                        self.request_tracker.update_timestamp(rsp.packet_id, "data_received_complete_cycle", self.current_cycle)
                        self.request_tracker.mark_request_completed(rsp.packet_id, self.current_cycle)

                    # 同时更新arrive_flits中对应packet的所有flit的时间戳
                    if req.packet_id in self.req_network.arrive_flits:
                        for flit in self.req_network.arrive_flits[req.packet_id]:
                            flit.write_complete_received_cycle = self.current_cycle
                            flit.data_received_complete_cycle = self.current_cycle

                    # 同时更新数据网络中对应packet的所有flit的时间戳（用于结果统计）
                    if req.packet_id in self.data_network.arrive_flits:
                        for flit in self.data_network.arrive_flits[req.packet_id]:
                            flit.write_complete_received_cycle = self.current_cycle
                            flit.data_received_complete_cycle = self.current_cycle

                # 对于跨Die写请求，需要特殊处理tracker释放
                if req and self._is_cross_die_write_request(req):
                    # 计算跨Die写请求的延迟（在释放tracker之前）
                    if req.packet_id in self.data_network.send_flits:
                        first_flit = self.data_network.send_flits[req.packet_id][0]
                        complete_cycle = self.current_cycle

                        for f in self.data_network.send_flits[req.packet_id]:
                            f.sync_latency_record(req)
                            # 计算延迟
                            f.cmd_latency = f.cmd_received_by_cake0_cycle - f.cmd_entry_noc_from_cake0_cycle
                            f.data_latency = f.data_received_complete_cycle - first_flit.data_entry_noc_from_cake0_cycle
                            f.transaction_latency = complete_cycle - f.cmd_entry_cake0_cycle

                    # 这是跨Die写请求的最终B通道响应，现在可以释放tracker
                    self.rn_tracker["write"].remove(req)
                    self.rn_wdb_count["count"] += req.burst_length
                    self.rn_tracker_count["write"]["count"] += 1
                    self._record_tracker_release("rn_write")
                    self._record_tracker_release("rn_wdb")
                    self.rn_tracker_pointer["write"] -= 1

                    # 更新D2D请求完成计数（只在源IP收到write_complete时记录，不在D2D_SN记录）
                    if hasattr(req, "d2d_origin_die") and hasattr(req, "d2d_target_die") and hasattr(req, "d2d_origin_type") and req.d2d_origin_type == self.ip_type:
                        die_id = getattr(self.config, "DIE_ID", None)
                        if die_id is not None and req.d2d_origin_die == die_id:
                            d2d_model = getattr(self.req_network, "d2d_model", None)
                            if d2d_model:
                                d2d_model.d2d_requests_completed[die_id] += 1
                                # 记录write_complete响应的接收（只在真正的源IP记录）
                                d2d_model.record_write_complete(req.packet_id, die_id)

    def _is_cross_die_write_request(self, req: Flit) -> bool:
        """检查是否为跨Die写请求"""
        # 检查是否有D2D目标Die ID，且与当前Die ID不同
        if hasattr(req, "d2d_target_die") and req.d2d_target_die is not None:
            current_die_id = getattr(self.config, "DIE_ID", 0)
            return req.d2d_target_die != current_die_id
        return False

    def _should_record_request_issued(self, flit: Flit, is_cross_die: bool) -> bool:
        """判断是否应该记录请求发出统计"""
        # 对于跨Die请求，只在源Die的原始发起IP记录
        if is_cross_die:
            current_die_id = getattr(self.config, "DIE_ID", 0)
            origin_die_id = getattr(flit, "d2d_origin_die", None)
            origin_type = getattr(flit, "d2d_origin_type", None)

            # 只有在源Die且当前IP是原始发起IP时才记录
            return current_die_id == origin_die_id and origin_type == self.ip_type
        else:
            # 本地请求直接记录
            return True

    def release_completed_sn_tracker(self, req: Flit):
        # —— 1) 移除已完成的 tracker ——
        self.sn_tracker.remove(req)
        # 释放一个 tracker 槽
        self.sn_tracker_count[req.sn_tracker_type]["count"] += 1
        self._record_tracker_release(f"sn_{req.sn_tracker_type}")

        # —— 2) 对于写请求，还要释放写缓冲额度 ——
        if req.req_type == "write":
            self.sn_wdb_count["count"] += req.burst_length
            self._record_tracker_release("sn_wdb")

        # —— 3) 尝试给等待队列里的请求重新分配资源 ——
        wait_list = self.sn_req_wait[req.req_type]
        if not wait_list:
            return

        if req.req_type == "write":
            # 写：既要有空 tracker，也要有足够 wdb_count
            if self.sn_tracker_count[req.sn_tracker_type]["count"] > 0 and self.sn_wdb_count["count"] >= wait_list[0].burst_length:
                new_req = wait_list.pop(0)
                new_req.sn_tracker_type = req.sn_tracker_type

                # 分配 tracker + wdb
                self.sn_tracker.append(new_req)
                self.sn_tracker_count[new_req.sn_tracker_type]["count"] -= 1
                self.sn_wdb_count["count"] -= new_req.burst_length
                self._record_tracker_allocation("sn_share")
                self._record_tracker_allocation("sn_wdb")

                # 记录retry释放（写请求成功处理，发送positive前记录）
                self._record_tracker_release("write_retry")

                # 发送 positive 响应
                self.create_rsp(new_req, "positive")

        elif req.req_type == "read":
            # 读：只要有空 tracker 即可
            if self.sn_tracker_count[req.sn_tracker_type]["count"] > 0:
                new_req = wait_list.pop(0)
                new_req.sn_tracker_type = req.sn_tracker_type

                # 分配 tracker
                self.sn_tracker.append(new_req)
                self.sn_tracker_count[new_req.sn_tracker_type]["count"] -= 1
                self._record_tracker_allocation("sn_ro")

                # 记录retry释放（读请求成功处理，发送positive前记录）
                self._record_tracker_release("read_retry")

                # 发送 positive 响应，让RN重新发送请求
                self.create_rsp(new_req, "positive")

    def _handle_received_data(self, flit: Flit):
        """处理接收到的数据"""
        flit.arrival_cycle = getattr(self, "current_cycle", 0)
        self.data_wait_cycles_h += flit.wait_cycle_h
        self.data_wait_cycles_v += flit.wait_cycle_v
        self.data_cir_h_num += flit.eject_attempts_h
        self.data_cir_v_num += flit.eject_attempts_v
        self.data_reverse_h_num += flit.reverse_inject_h
        self.data_reverse_v_num += flit.reverse_inject_v
        if flit.req_type == "read":
            # 检查是否为跨Die返回的数据，更新D2D统计
            die_id = getattr(self.config, "DIE_ID", None)
            d2d_model = getattr(self.req_network, "d2d_model", None)

            if hasattr(flit, "d2d_origin_die") and hasattr(flit, "d2d_target_die"):
                if flit.d2d_origin_die != flit.d2d_target_die:
                    # 这是跨Die请求的返回数据
                    if die_id is not None and flit.d2d_origin_die == die_id:
                        if d2d_model:
                            burst_length = getattr(flit, "burst_length", 4)
                            # 记录跨Die读数据接收
                            d2d_model.record_read_data_received(flit.packet_id, die_id, burst_length, is_cross_die=True)
                else:
                    # d2d属性存在但origin_die == target_die，这是Die内本地请求
                    if die_id is not None and d2d_model:
                        burst_length = getattr(flit, "burst_length", 4)
                        d2d_model.record_read_data_received(flit.packet_id, die_id, burst_length, is_cross_die=False)
            else:
                # 没有d2d属性，这是本地读请求的返回数据
                if die_id is not None and d2d_model:
                    burst_length = getattr(flit, "burst_length", 4)
                    d2d_model.record_read_data_received(flit.packet_id, die_id, burst_length, is_cross_die=False)

            # 读数据到达RN端，需要收集到data buffer中
            if flit.packet_id not in self.rn_rdb:
                self.rn_rdb[flit.packet_id] = []
            self.rn_rdb[flit.packet_id].append(flit)

            # 添加到RequestTracker
            if hasattr(self, "request_tracker") and self.request_tracker:
                self.request_tracker.add_data_flit(flit.packet_id, flit)

            # 如果是跨Die数据且完成了整个burst，更新请求完成计数
            if hasattr(flit, "d2d_origin_die") and hasattr(flit, "d2d_target_die"):
                if flit.d2d_origin_die != flit.d2d_target_die:
                    die_id = getattr(self.config, "DIE_ID", None)
                    if die_id is not None and flit.d2d_origin_die == die_id:
                        if len(self.rn_rdb[flit.packet_id]) == flit.burst_length:
                            d2d_model = getattr(self.req_network, "d2d_model", None)
                            if d2d_model:
                                d2d_model.d2d_requests_completed[die_id] += 1
            # 检查是否收集完整个burst
            collected = len(self.rn_rdb[flit.packet_id])
            expected = flit.burst_length
            if collected == expected:
                # 标记读请求完成
                if hasattr(self, "request_tracker") and self.request_tracker:
                    self.request_tracker.mark_request_completed(flit.packet_id, self.current_cycle)

                req = next((req for req in self.rn_tracker["read"] if req.packet_id == flit.packet_id), None)
                if req:
                    # 立即释放tracker和更新计数
                    self.rn_tracker["read"].remove(req)
                    self.rn_tracker_count["read"]["count"] += 1
                    self._record_tracker_release("rn_read")
                    self._record_tracker_release("rn_rdb")
                    self.rn_tracker_pointer["read"] -= 1
                    self.rn_rdb_count["count"] += req.burst_length
                    # 为所有flit设置完成时间戳和计算延迟
                    first_flit = self.data_network.send_flits[flit.packet_id][0]
                    # 记录最后到达时间
                    complete_cycle = self.current_cycle

                    # 同时更新tracker中的data_flits和network中的send_flits
                    for f in self.data_network.send_flits[flit.packet_id]:
                        f.leave_db_cycle = self.current_cycle
                        f.sync_latency_record(req)
                        # 为所有flit设置receive时间戳，确保后续处理能获得正确值
                        f.data_received_complete_cycle = complete_cycle
                        # 计算延迟
                        f.cmd_latency = f.cmd_received_by_cake1_cycle - f.cmd_entry_noc_from_cake0_cycle
                        f.data_latency = complete_cycle - first_flit.data_entry_noc_from_cake1_cycle
                        f.transaction_latency = complete_cycle - f.cmd_entry_cake0_cycle

                    # 同步更新tracker中的data_flits时间戳并重新收集
                    if hasattr(self, "request_tracker") and self.request_tracker:
                        lifecycle = self.request_tracker.active_requests.get(flit.packet_id) or self.request_tracker.completed_requests.get(flit.packet_id)
                        if lifecycle:
                            for f in lifecycle.data_flits:
                                f.data_received_complete_cycle = complete_cycle
                            # 重新收集时间戳以确保timestamps字典被更新
                            self.request_tracker.collect_timestamps_from_flits(flit.packet_id)

                    # 清理data buffer（数据已经收集完成）
                    self.rn_rdb.pop(flit.packet_id)

                else:
                    print(f"Warning: No RN tracker found for packet_id {flit.packet_id}")

        elif flit.req_type == "write":
            # D2D写数据统计（包括跨Die和Die内）
            die_id = getattr(self.config, "DIE_ID", None)
            d2d_model = getattr(self.req_network, "d2d_model", None)

            if hasattr(flit, "d2d_origin_die") and hasattr(flit, "d2d_target_die"):
                if die_id is not None and d2d_model:
                    burst_length = getattr(flit, "burst_length", 4)
                    is_cross_die = flit.d2d_origin_die != flit.d2d_target_die

                    # 记录写数据接收：跨Die或Die内
                    if is_cross_die and flit.d2d_target_die == die_id:
                        # 跨Die写数据接收
                        d2d_model.record_write_data_received(flit.packet_id, die_id, burst_length, is_cross_die=True)
                    elif not is_cross_die:
                        # Die内写数据接收
                        d2d_model.record_write_data_received(flit.packet_id, die_id, burst_length, is_cross_die=False)
            else:
                # 没有d2d属性，这是本地写数据
                if die_id is not None and d2d_model:
                    burst_length = getattr(flit, "burst_length", 4)
                    d2d_model.record_write_data_received(flit.packet_id, die_id, burst_length, is_cross_die=False)

            # 确保sn_wdb中存在packet_id的列表（跨Die写数据可能没有预先创建）
            if flit.packet_id not in self.sn_wdb:
                self.sn_wdb[flit.packet_id] = []

            self.sn_wdb[flit.packet_id].append(flit)

            # 添加到RequestTracker（写请求的data flit）
            if hasattr(self, "request_tracker") and self.request_tracker:
                self.request_tracker.add_data_flit(flit.packet_id, flit)

            # 检查是否收集完整个burst
            if len(self.sn_wdb[flit.packet_id]) == flit.burst_length:

                # 检查是否为跨Die写数据（tracker由D2D_RN管理，不在本地SN中）
                is_cross_die_write = (
                    hasattr(flit, "d2d_origin_die")
                    and hasattr(flit, "d2d_target_die")
                    and flit.d2d_origin_die is not None
                    and flit.d2d_target_die is not None
                    and flit.d2d_origin_die != flit.d2d_target_die
                )

                # 找到对应的请求（跨Die和Die内都需要释放SN tracker）
                req = next((req for req in self.sn_tracker if req.packet_id == flit.packet_id), None)
                if req:
                    # 设置tracker延迟释放时间
                    release_time = self.current_cycle + self.config.SN_TRACKER_RELEASE_LATENCY

                    # 释放时间字典已在__init__中初始化

                    # 更新flit时间戳
                    first_flit = next((flit for flit in self.data_network.send_flits[flit.packet_id] if flit.flit_id == 0), self.data_network.send_flits[flit.packet_id][0])
                    # 记录最后到达时间
                    complete_cycle = self.current_cycle

                    for f in self.data_network.send_flits[flit.packet_id]:
                        f.leave_db_cycle = release_time
                        f.sync_latency_record(req)
                        # 为所有flit设置receive时间戳，确保后续处理能获得正确值
                        f.data_received_complete_cycle = complete_cycle
                        # 计算延迟
                        f.cmd_latency = f.cmd_received_by_cake0_cycle - f.cmd_entry_noc_from_cake0_cycle
                        f.data_latency = complete_cycle - first_flit.data_entry_noc_from_cake0_cycle
                        f.transaction_latency = complete_cycle + self.config.SN_TRACKER_RELEASE_LATENCY - f.cmd_entry_cake0_cycle

                    # 更新RequestTracker的timestamp
                    if hasattr(self, "request_tracker") and self.request_tracker:
                        self.request_tracker.update_timestamp(flit.packet_id, "data_received_complete_cycle", complete_cycle)
                        self.request_tracker.update_timestamp(flit.packet_id, "data_entry_noc_from_cake0_cycle", first_flit.data_entry_noc_from_cake0_cycle)

                        # 对于NoC内部写请求，数据到达SN端即完成（不需要write_complete响应）
                        # 但是对于跨Die写请求，需要等待write_complete响应（在RN端处理）
                        if not is_cross_die_write:
                            self.request_tracker.mark_request_completed(flit.packet_id, complete_cycle)

                    # 清理data buffer（数据已经收集完成）
                    self.sn_wdb.pop(flit.packet_id)

                    # 添加到延迟释放队列
                    self.sn_tracker_release_time[release_time].append(req)
                else:
                    print(f"Warning: No SN tracker found for packet_id {flit.packet_id}")

    def h2l_l_to_eject_fifo(self, network_type):
        """1GHz: h2l_fifo_l → 处理完成"""
        net_info = self.networks[network_type]
        if not net_info["h2l_fifo_l"]:
            return

        current_cycle = getattr(self, "current_cycle", 0)

        # 带宽限制检查（仅数据网络）
        if network_type == "data" and self.rx_token_bucket:
            self.rx_token_bucket.refill(current_cycle)
            if not self.rx_token_bucket.consume():
                return

        # 出队flit
        flit = net_info["h2l_fifo_l"].popleft()
        flit.set_position("IP_eject", current_cycle)
        flit.is_finish = True

        # 在IP_eject阶段添加到arrive_flits，确保只记录真正完成的flit
        if flit.packet_id not in net_info["network"].arrive_flits:
            net_info["network"].arrive_flits[flit.packet_id] = []
        net_info["network"].arrive_flits[flit.packet_id].append(flit)
        net_info["network"].recv_flits_num += 1

        # 根据网络类型进行特殊处理
        if network_type == "req":
            self._handle_received_request(flit)
        elif network_type == "rsp":
            self._handle_received_response(flit)
        elif network_type == "data":
            self._handle_received_data(flit)
        return flit
        # Ensure method always returns None if no flit ejected
        return None

    def inject_step(self, cycle):
        """根据周期和频率调用inject相应的方法"""
        self.current_cycle = cycle
        cycle_mod = cycle % self.config.NETWORK_FREQUENCY

        # 1GHz 操作（每个网络周期执行一次）
        if cycle_mod == 0:
            # 对三个网络分别执行inject_to_l2h_pre
            for net_type in ["req", "rsp", "data"]:
                self.inject_to_l2h_pre(net_type)

        # 2GHz 操作（每半个网络周期执行一次）
        for net_type in ["req", "rsp", "data"]:
            self.l2h_to_IQ_channel_buffer(net_type)

    def move_pre_to_fifo(self):
        # pre → fifo 的移动（每个周期都执行）
        for net_type, net_info in self.networks.items():
            net = net_info["network"]
            # l2h_fifo_pre → l2h_fifo
            if net_info["l2h_fifo_pre"] is not None and len(net_info["l2h_fifo"]) < net_info["l2h_fifo"].maxlen:
                net_info["l2h_fifo"].append(net_info["l2h_fifo_pre"])
                net_info["l2h_fifo_pre"] = None

            # h2l_fifo_h_pre → h2l_fifo_h
            if net_info["h2l_fifo_h_pre"] is not None and len(net_info["h2l_fifo_h"]) < net_info["h2l_fifo_h"].maxlen:
                net_info["h2l_fifo_h"].append(net_info["h2l_fifo_h_pre"])
                net_info["h2l_fifo_h_pre"] = None

            # h2l_fifo_l_pre → h2l_fifo_l (直接移动)
            if net_info["h2l_fifo_l_pre"] is not None and len(net_info["h2l_fifo_l"]) < net_info["h2l_fifo_l"].maxlen:
                net_info["h2l_fifo_l"].append(net_info["h2l_fifo_l_pre"])
                net_info["h2l_fifo_l_pre"] = None

    def eject_step(self, cycle):
        """根据周期和频率调用eject相应的方法"""
        self.current_cycle = cycle
        cycle_mod = cycle % self.config.NETWORK_FREQUENCY

        # Initialize list to collect ejected flits
        ejected_flits = []

        # 2GHz 操作（每半个网络周期执行一次）
        for net_type in ["req", "rsp", "data"]:
            self.EQ_channel_buffer_to_h2l_pre(net_type)
            self.h2l_h_to_h2l_l_pre(net_type)

        # 1GHz 操作（每个网络周期执行一次）
        if cycle_mod == 0:
            # 对三个网络分别执行h2l_to_eject_fifo，收集被eject的flit
            for net_type in ["req", "rsp", "data"]:
                flit = self.h2l_l_to_eject_fifo(net_type)
                if flit:
                    ejected_flits.append(flit)

        return ejected_flits

    def create_write_packet(self, req: Flit):
        """创建写数据包并放入数据网络inject_fifo"""
        cycle = getattr(self, "current_cycle", 0)
        for i in range(req.burst_length):
            source = req.source
            destination = req.destination
            path = self.routes[source][destination]
            flit = Flit(source, destination, path)
            flit.sync_latency_record(req)
            flit.flit_type = "data"
            # 保序信息将在inject_fifo出队时分配（inject_to_l2h_pre）
            # 根据目标IP类型选择延迟
            dest_type = get_original_destination_type(req)
            flit.departure_cycle = (
                cycle + self.config.DDR_W_LATENCY + i * self.config.NETWORK_FREQUENCY + self.rn_processing_latency
                if dest_type and dest_type.startswith("ddr")
                else cycle + self.config.L2M_W_LATENCY + i * self.config.NETWORK_FREQUENCY + self.rn_processing_latency
            )
            flit.req_departure_cycle = req.departure_cycle
            flit.entry_db_cycle = req.entry_db_cycle
            flit.source_type = req.source_type
            flit.destination_type = req.destination_type
            # Die内传输：不需要设置_original属性，辅助函数会使用source/destination
            flit.req_type = req.req_type
            flit.packet_id = req.packet_id
            flit.flit_id = i
            flit.burst_length = req.burst_length
            flit.traffic_id = req.traffic_id
            if i == req.burst_length - 1:
                flit.is_last_flit = True

            # 继承D2D属性
            if hasattr(req, "d2d_origin_die"):
                flit.d2d_origin_die = req.d2d_origin_die  # 发起Die ID
                flit.d2d_origin_node = req.d2d_origin_node  # 发起节点源映射位置
                flit.d2d_origin_type = req.d2d_origin_type  # 发起IP类型
                flit.d2d_target_die = req.d2d_target_die  # 目标Die ID
                flit.d2d_target_node = req.d2d_target_node  # 目标节点源映射位置
                flit.d2d_target_type = req.d2d_target_type  # 目标IP类型

            # 将写数据包放入wdb中
            self.rn_wdb[flit.packet_id].append(flit)

    def create_read_packet(self, req: Flit):
        """创建读数据包并放入数据网络inject_fifo"""
        cycle = getattr(self, "current_cycle", 0)
        for i in range(req.burst_length):
            # 新架构：数据包直接反向，无需偏移
            source = req.destination
            destination = req.source

            path = self.routes[source][destination]
            flit = Flit(source, destination, path)
            flit.sync_latency_record(req)
            # Die内传输不设置_original属性，D2D传输继承d2d_*属性即可
            flit.req_type = req.req_type
            flit.flit_type = "data"
            # 保序信息将在inject_fifo出队时分配（inject_to_l2h_pre）
            if get_original_destination_type(req).startswith("ddr"):
                latency = np.random.uniform(low=self.config.DDR_R_LATENCY - self.config.DDR_R_LATENCY_VAR, high=self.config.DDR_R_LATENCY + self.config.DDR_R_LATENCY_VAR, size=None)
            elif hasattr(req, "destination_type") and req.destination_type and req.destination_type.startswith("ddr"):
                latency = np.random.uniform(low=self.config.DDR_R_LATENCY - self.config.DDR_R_LATENCY_VAR, high=self.config.DDR_R_LATENCY + self.config.DDR_R_LATENCY_VAR, size=None)
            else:
                latency = self.config.L2M_R_LATENCY
            flit.departure_cycle = cycle + latency + i * self.config.NETWORK_FREQUENCY + self.sn_processing_latency
            flit.entry_db_cycle = cycle
            flit.req_departure_cycle = req.departure_cycle
            flit.source_type = getattr(req, "destination_type", None)
            flit.destination_type = getattr(req, "source_type", None)
            # Die内传输不设置original_*属性，辅助函数会自动从d2d_*属性推断
            flit.packet_id = req.packet_id
            flit.flit_id = i
            flit.burst_length = req.burst_length
            flit.traffic_id = req.traffic_id
            if i == req.burst_length - 1:
                flit.is_last_flit = True

            # 继承D2D属性
            if hasattr(req, "d2d_origin_die"):
                flit.d2d_origin_die = req.d2d_origin_die  # 发起Die ID
                flit.d2d_origin_node = req.d2d_origin_node  # 发起节点源映射位置
                flit.d2d_origin_type = req.d2d_origin_type  # 发起IP类型
                flit.d2d_target_die = req.d2d_target_die  # 目标Die ID
                flit.d2d_target_node = req.d2d_target_node  # 目标节点源映射位置
                flit.d2d_target_type = req.d2d_target_type  # 目标IP类型

            # 将读数据包放入数据网络的inject_fifo
            self.enqueue(flit, "data")

    def create_rsp(self, req: Flit, rsp_type):
        """创建响应并放入响应网络inject_fifo"""
        cycle = getattr(self, "current_cycle", 0)

        # 检查是否为Ring拓扑
        # 新架构：响应直接反向，无需偏移
        source = req.destination
        destination = req.source

        path = self.routes[source][destination]
        rsp = Flit(source, destination, path)
        rsp.flit_type = "rsp"
        rsp.rsp_type = rsp_type
        rsp.req_type = req.req_type

        # Die内传输不设置_original属性，D2D传输继承d2d_*属性即可

        # 保序信息将在inject_fifo出队时分配（inject_to_l2h_pre）
        rsp.packet_id = req.packet_id
        rsp.departure_cycle = cycle + self.config.NETWORK_FREQUENCY + self.sn_processing_latency
        rsp.req_departure_cycle = req.departure_cycle
        rsp.source_type = req.destination_type
        rsp.destination_type = req.source_type
        rsp.sync_latency_record(req)
        rsp.sn_rsp_generate_cycle = cycle

        # 记录negative响应（retry分配）
        if rsp_type == "negative":
            retry_type = "read_retry" if req.req_type == "read" else "write_retry"
            self._record_tracker_allocation(retry_type)

        # 将响应放入响应网络的inject_fifo
        self.enqueue(rsp, "rsp")

    def error_log(self, flit, target_id, flit_id):
        if flit and flit.packet_id == target_id and flit.flit_id == flit_id:
            print(inspect.currentframe().f_back.f_code.co_name, flit)

    # ========== Tracker记录方法 ==========
    def _record_tracker_allocation(self, tracker_type: str):
        """记录tracker分配事件"""
        if self.tracker_total_config.get(tracker_type, 0) == 0:
            return
        cycle = getattr(self, "current_cycle", 0)
        self.tracker_events[tracker_type]["allocations"].append(cycle)
        self.tracker_total_allocated[tracker_type] += 1

    def _record_tracker_release(self, tracker_type: str):
        """记录tracker释放事件"""
        if self.tracker_total_config.get(tracker_type, 0) == 0:
            return
        cycle = getattr(self, "current_cycle", 0)
        self.tracker_events[tracker_type]["releases"].append(cycle)

    def _record_tracker_block(self, tracker_type: str):
        """记录tracker阻塞事件"""
        cycle = getattr(self, "current_cycle", 0)
        self.tracker_block_events.append({"cycle": cycle, "tracker_type": tracker_type})

    def _release_sn_read_tracker(self, packet_id: int):
        """读数据最后一个flit进入l2h时释放SN tracker"""
        tracker = next((req for req in self.sn_tracker if req.packet_id == packet_id), None)
        if tracker:
            self.sn_tracker.remove(tracker)
            self.sn_tracker_count[tracker.sn_tracker_type]["count"] += 1
            self._record_tracker_release(f"sn_{tracker.sn_tracker_type}")
            # 处理等待队列
            self._process_sn_read_wait_queue(tracker.sn_tracker_type)

    def _process_sn_read_wait_queue(self, tracker_type: str):
        """处理SN读请求等待队列"""
        wait_list = self.sn_req_wait["read"]
        if not wait_list:
            return

        if self.sn_tracker_count[tracker_type]["count"] > 0:
            new_req = wait_list.pop(0)
            new_req.sn_tracker_type = tracker_type

            # 分配tracker
            self.sn_tracker.append(new_req)
            self.sn_tracker_count[new_req.sn_tracker_type]["count"] -= 1
            self._record_tracker_allocation("sn_ro")

            # 记录retry释放（发送positive前记录）
            self._record_tracker_release("read_retry")

            # 发送 positive 响应，让RN重新发送请求
            self.create_rsp(new_req, "positive")

    def get_tracker_usage_data(self):
        """获取tracker使用数据"""
        return {"events": self.tracker_events, "total_allocated": self.tracker_total_allocated, "block_events": self.tracker_block_events, "total_config": self.tracker_total_config}


def create_ring_ip_interface(ip_type, ip_pos, config, req_network, rsp_network, data_network, node, routes):
    """创建支持Ring的IPInterface实例"""
    from .ip_interface import IPInterface

    # 创建混合类
    class RingIPInterfaceImpl(RingIPInterface, IPInterface):
        pass

    return RingIPInterfaceImpl(ip_type, ip_pos, config, req_network, rsp_network, data_network, node, routes, ring_mode=True)
