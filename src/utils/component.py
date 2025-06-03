from __future__ import annotations
import numpy as np
from collections import deque, defaultdict
from config.config import CrossRingConfig
import logging
import inspect


class TokenBucket:
    """Simple token bucket for rate limiting."""

    def __init__(self, rate=1, bucket_size=10):
        self.rate = rate
        self.bucket_size = bucket_size
        self.tokens = bucket_size
        self.last_cycle = 0

    def consume(self):
        if self.tokens > 0:
            self.tokens -= 1
            return True
        return False

    def refill(self, cycle):
        dt = cycle - self.last_cycle
        if dt <= 0:
            return
        add = dt * self.rate
        self.last_cycle = cycle
        self.tokens = min(self.tokens + add, self.bucket_size)


class Flit:
    last_id = 0

    def __init__(self, source, destination, path):
        self.source = source
        self.source_original = -1
        self.destination = destination
        self.destination_original = -1
        self.source_type = None
        self.destination_type = None
        self.burst_length = -1
        self.path = path
        self.flit_position = ""
        self.is_finish = False
        Flit.last_id += 1
        self.packet_id = None
        self.moving_direction = self.calculate_direction(path)
        self.moving_direction_v = 1 if source < destination else -1
        self.flit_type = "flit"
        self.req_type = None
        self.req_attr = "new"
        self.req_state = "valid"
        self.id = Flit.last_id
        self.flit_id = -1
        self.is_last_flit = False
        self.circuits_completed_v = 0
        self.circuits_completed_h = 0
        self.wait_cycle_h = 0
        self.wait_cycle_v = 0
        self.path_index = 0
        self.current_seat_index = -1
        self.current_link = None
        self.rsp_type = None
        self.rn_tracker_type = None
        self.sn_tracker_type = None
        self.init_param()

    def init_param(self):
        self.early_rsp = False
        self.current_position = self.path[0]
        self.station_position = -1
        self.departure_cycle = np.inf
        self.req_departure_cycle = np.inf
        self.departure_network_cycle = np.inf
        self.departure_inject_cycle = np.inf
        self.arrival_cycle = np.inf
        self.arrival_network_cycle = np.inf
        self.arrival_eject_cycle = np.inf
        self.entry_db_cycle = np.inf
        self.leave_db_cycle = np.inf
        self.start_inject = False
        self.is_injected = False
        self.is_ejected = False
        self.is_new_on_network = True
        self.is_on_station = False
        self.is_delay = False
        self.is_arrive = False
        self.predicted_duration = 0
        self.actual_duration = 0
        self.actual_ject_duration = 0
        self.actual_network_duration = 0
        self.itag_v = False
        self.itag_h = False
        self.is_tagged = False
        self.ETag_priority = "T2"  # 默认优先级为 T2
        # 记录下环 / 弹出时实际占用的是哪一级 entry（"T0" / "T1" / "T2"）
        self.used_entry_level = None
        # Latency record
        self.cmd_entry_cmd_table_cycle = np.inf
        self.req_entry_network_cycle = np.inf
        self.sn_receive_req_cycle = np.inf
        self.sn_data_generated_cycle = np.inf
        self.data_entry_network_cycle = np.inf
        self.rn_data_collection_complete_cycle = np.inf
        self.sn_rsp_generate_cycle = np.inf
        self.rsp_entry_network_cycle = np.inf
        self.rn_receive_rsp_cycle = np.inf
        self.rn_data_generated_cycle = np.inf
        self.sn_data_collection_complete_cycle = np.inf
        self.total_latency = -1
        self.cmd_latency = -1
        self.rsp_latency = -1
        self.dat_latency = -1

    def sync_latency_record(self, flit):
        if flit.req_type == "read":
            self.cmd_entry_cmd_table_cycle = flit.cmd_entry_cmd_table_cycle
            self.req_entry_network_cycle = flit.req_entry_network_cycle
            self.sn_receive_req_cycle = flit.sn_receive_req_cycle
            self.sn_data_generated_cycle = flit.sn_data_generated_cycle
            self.data_entry_network_cycle = flit.data_entry_network_cycle
            self.rn_data_collection_complete_cycle = flit.rn_data_collection_complete_cycle
        elif flit.req_type == "write":
            self.cmd_entry_cmd_table_cycle = flit.cmd_entry_cmd_table_cycle
            self.req_entry_network_cycle = flit.req_entry_network_cycle
            self.sn_receive_req_cycle = flit.sn_receive_req_cycle
            self.sn_rsp_generate_cycle = flit.sn_rsp_generate_cycle
            self.rsp_entry_network_cycle = flit.rsp_entry_network_cycle
            self.rn_receive_rsp_cycle = flit.rn_receive_rsp_cycle
            self.rn_data_generated_cycle = flit.rn_data_generated_cycle
            self.data_entry_network_cycle = flit.data_entry_network_cycle
            self.sn_data_collection_complete_cycle = flit.sn_data_collection_complete_cycle

    def calculate_direction(self, path):
        if len(path) < 2:
            return 0  # Or handle this case appropriately
        return 1 if path[1] - path[0] == 1 else -1 if path[1] - path[0] == -1 else 0

    def inject(self, network: "Network"):
        if self.path_index == 0 and not self.is_injected:
            if len(self.path) > 1:  # Ensure there is a next position
                next_position = self.path[self.path_index + 1]
                if network.can_move_to_next(self, self.source, next_position):
                    self.current_position = self.source
                    self.is_injected = True
                    self.is_new_on_network = True
                    self.current_link = None
                    return True
        return False

    def __repr__(self):
        req_attr = "O" if self.req_attr == "old" else "N"
        type_display = self.rsp_type[:3] if self.rsp_type else self.req_type[0]
        flit_position = (
            f"{self.current_position}:{self.flit_position}"
            if self.flit_position != "Link"
            else f"({self.current_position}: {self.current_link[0]}->{self.current_link[1]}).{self.current_seat_index}, "
        )
        finish_status = "F" if self.is_finish else ""
        eject_status = "E" if self.is_ejected else ""

        return (
            f"{self.packet_id}.{self.flit_id} {self.source}.{self.source_type[0]}{self.source_type[-1]}->{self.destination}.{self.destination_type[0]}{self.destination_type[-1]}: "
            f"{flit_position}, "
            f"{req_attr}, {self.flit_type}, {type_display}, "
            f"{finish_status}{eject_status}, "
            f"{self.ETag_priority};"
        )

    @classmethod
    def clear_flit_id(cls):
        cls.last_id = 0


class Node:
    global_packet_id = -1

    def __init__(self, config: CrossRingConfig):
        self.config = config
        self.initialize_data_structures()
        self.initialize_rn()
        self.initialize_sn()

    @classmethod
    def get_next_packet_id(cls):
        cls.global_packet_id += 1
        return cls.global_packet_id

    @classmethod
    def clear_packet_id(cls):
        cls.global_packet_id = -1

    def initialize_data_structures(self):
        """Initialize the data structures for read and write databases."""

        self.rn_rdb = self.config._make_channels(("sdma", "gdma"))
        self.rn_rdb_reserve = self.config._make_channels(("sdma", "gdma"))
        self.rn_rdb_recv = self.config._make_channels(("sdma", "gdma"))
        self.rn_rdb_count = self.config._make_channels(("sdma", "gdma"))
        self.rn_wdb = self.config._make_channels(("sdma", "gdma"))
        self.rn_wdb_reserve = self.config._make_channels(("sdma", "gdma"))
        self.rn_wdb_count = self.config._make_channels(("sdma", "gdma"))
        self.rn_wdb_send = self.config._make_channels(("sdma", "gdma"))
        self.rn_tracker = {"read": self.config._make_channels(("sdma", "gdma")), "write": self.config._make_channels(("sdma", "gdma"))}
        self.rn_tracker_wait = {"read": self.config._make_channels(("sdma", "gdma")), "write": self.config._make_channels(("sdma", "gdma"))}
        self.rn_tracker_count = {"read": self.config._make_channels(("sdma", "gdma")), "write": self.config._make_channels(("sdma", "gdma"))}
        self.rn_tracker_pointer = {"read": self.config._make_channels(("sdma", "gdma")), "write": self.config._make_channels(("sdma", "gdma"))}
        self.sn_rdb = self.config._make_channels(("ddr", "l2m"))
        self.sn_rsp_queue = self.config._make_channels(("ddr", "l2m"))
        self.sn_req_wait = {"read": self.config._make_channels(("ddr", "l2m")), "write": self.config._make_channels(("ddr", "l2m"))}
        self.sn_tracker = self.config._make_channels(("ddr", "l2m"))
        self.sn_tracker_count = self.config._make_channels(("ddr", "l2m"), value_factory={"ro": {}, "share": {}})
        self.sn_wdb = self.config._make_channels(("ddr", "l2m"))
        self.sn_wdb_recv = self.config._make_channels(("ddr", "l2m"))
        self.sn_wdb_count = self.config._make_channels(("ddr", "l2m"))
        self.rn_wait_to_inject = []

    def initialize_rn(self):
        """Initialize RN structures."""
        for ip_type in self.rn_rdb.keys():
            for ip_pos in getattr(self.config, f"{ip_type[:-2].upper()}_SEND_POSITION_LIST"):
                self.rn_rdb[ip_type][ip_pos] = defaultdict(list)
                self.rn_wdb[ip_type][ip_pos] = defaultdict(list)
                self.setup_rn_trackers(ip_type, ip_pos)

    def setup_rn_trackers(self, ip_type, ip_pos):
        """Setup read and write trackers for RN."""
        self.rn_rdb_recv[ip_type][ip_pos] = []
        self.rn_rdb_count[ip_type][ip_pos] = self.config.RN_RDB_SIZE
        self.rn_rdb_reserve[ip_type][ip_pos] = 0
        self.rn_wdb_count[ip_type][ip_pos] = self.config.RN_WDB_SIZE
        self.rn_wdb_send[ip_type][ip_pos] = []
        self.rn_wdb_reserve[ip_type][ip_pos] = 0

        for req_type in ["read", "write"]:
            self.rn_tracker[req_type][ip_type][ip_pos] = []
            self.rn_tracker_wait[req_type][ip_type][ip_pos] = []
            self.rn_tracker_count[req_type][ip_type][ip_pos] = self.config.RN_R_TRACKER_OSTD if req_type == "read" else self.config.RN_W_TRACKER_OSTD
            self.rn_tracker_pointer[req_type][ip_type][ip_pos] = -1

    def initialize_sn(self):
        """Initialize SN structures."""
        self.sn_tracker_release_time = defaultdict(list)
        for ip_pos in set(self.config.DDR_SEND_POSITION_LIST + self.config.L2M_SEND_POSITION_LIST):
            for key in self.sn_tracker:
                self.sn_rdb[key][ip_pos] = []
                self.sn_wdb[key][ip_pos] = defaultdict(list)
                self.setup_sn_trackers(key, ip_pos)

    def setup_sn_trackers(self, key, ip_pos):
        """Setup trackers for SN."""
        self.sn_rsp_queue[key][ip_pos] = []
        for req_type in ["read", "write"]:
            self.sn_req_wait[req_type][key][ip_pos] = []
        self.sn_wdb_recv[key][ip_pos] = []
        self.sn_tracker[key][ip_pos] = []
        # if self.config.topo_type != "3x3":
        if key.startswith("ddr"):
            self.sn_wdb_count[key][ip_pos] = self.config.SN_DDR_WDB_SIZE
            self.sn_tracker_count[key]["ro"][ip_pos] = self.config.SN_DDR_R_TRACKER_OSTD
            self.sn_tracker_count[key]["share"][ip_pos] = self.config.SN_DDR_W_TRACKER_OSTD
        elif key.startswith("l2m"):
            self.sn_wdb_count[key][ip_pos] = self.config.SN_L2M_WDB_SIZE
            self.sn_tracker_count[key]["ro"][ip_pos] = self.config.SN_L2M_R_TRACKER_OSTD
            self.sn_tracker_count[key]["share"][ip_pos] = self.config.SN_L2M_W_TRACKER_OSTD


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
        req_network: Network,
        rsp_network: Network,
        data_network: Network,
        node: Node,
        routes: dict,
    ):
        self.ip_type = ip_type
        self.ip_pos = ip_pos
        self.config = config
        self.req_network = req_network
        self.rsp_network = rsp_network
        self.data_network = data_network
        self.node = node
        self.routes = routes

        # 验证FIFO深度配置
        l2h_depth = getattr(config, "IP_L2H_FIFO_DEPTH", 4)
        h2l_depth = getattr(config, "IP_H2L_FIFO_DEPTH", 4)

        assert l2h_depth >= 2, f"L2H FIFO depth ({l2h_depth}) should be at least 2"
        assert h2l_depth >= 2, f"H2L FIFO depth ({h2l_depth}) should be at least 2"

        self.networks = {
            "req": {
                "network": req_network,
                "send_flits": req_network.send_flits,
                "inject_fifo": deque(),  # 1GHz域 - IP核产生的请求
                "l2h_fifo_pre": None,  # 1GHz → 2GHz 预缓冲
                "h2l_fifo_pre": None,  # 2GHz → 1GHz 预缓冲
                "l2h_fifo": deque(maxlen=l2h_depth),  # 1GHz → 2GHz 转换FIFO
                "h2l_fifo": deque(maxlen=h2l_depth),  # 2GHz → 1GHz 转换FIFO
            },
            "rsp": {
                "network": rsp_network,
                "send_flits": rsp_network.send_flits,
                "inject_fifo": deque(),
                "l2h_fifo_pre": None,
                "h2l_fifo_pre": None,
                "l2h_fifo": deque(maxlen=l2h_depth),
                "h2l_fifo": deque(maxlen=h2l_depth),
            },
            "data": {
                "network": data_network,
                "send_flits": data_network.send_flits,
                "inject_fifo": deque(),
                "l2h_fifo_pre": None,
                "h2l_fifo_pre": None,
                "l2h_fifo": deque(maxlen=l2h_depth),
                "h2l_fifo": deque(maxlen=h2l_depth),
            },
        }

        if ip_type.startswith("ddr"):
            self.token_bucket = TokenBucket(
                rate=self.config.DDR_BW_LIMIT / self.config.NETWORK_FREQUENCY / self.config.FLIT_SIZE,
                bucket_size=self.config.DDR_BW_LIMIT,
            )
        elif ip_type.startswith("l2m"):
            self.token_bucket = TokenBucket(
                rate=self.config.L2M_BW_LIMIT / self.config.NETWORK_FREQUENCY / self.config.FLIT_SIZE,
                bucket_size=self.config.L2M_BW_LIMIT,
            )
        else:
            self.token_bucket = None

    def enqueue(self, flit: Flit, network_type: str, retry=False):
        """IP 核把flit丢进对应网络的 inject_fifo"""
        if retry:
            self.networks[network_type]["inject_fifo"].appendleft(flit)
        else:
            self.networks[network_type]["inject_fifo"].append(flit)
        flit.flit_position = "IP_inject"
        if network_type == "req" and self.networks[network_type]["send_flits"][flit.packet_id]:
            return True
        self.networks[network_type]["send_flits"][flit.packet_id].append(flit)
        return True

    def inject_to_l2h_pre(self, network_type):
        """1GHz: inject_fifo → l2h_fifo_pre（分网络处理）"""
        net_info = self.networks[network_type]

        if not net_info["inject_fifo"] or len(net_info["l2h_fifo"]) >= net_info["l2h_fifo"].maxlen or net_info["l2h_fifo_pre"] is not None:
            return

        flit = net_info["inject_fifo"][0]

        # 根据网络类型进行不同的处理
        if network_type == "req":
            if flit.req_attr == "new" and not self._check_and_reserve_resources(flit):
                return  # 资源不足，保持在inject_fifo中
            flit.flit_position = "L2H_FIFO"
            flit.start_inject = True
            net_info["l2h_fifo_pre"] = net_info["inject_fifo"].popleft()

        elif network_type == "rsp":
            # 响应网络：直接移动
            flit.flit_position = "L2H_FIFO"
            flit.start_inject = True
            net_info["l2h_fifo_pre"] = net_info["inject_fifo"].popleft()

        elif network_type == "data":
            # 数据网络：检查departure_cycle
            current_cycle = getattr(self, "current_cycle", 0)
            if hasattr(flit, "departure_cycle") and flit.departure_cycle > current_cycle:
                return  # 还没到发送时间
            if self.token_bucket:
                self.token_bucket.refill(current_cycle)
                if not self.token_bucket.consume():
                    return
            flit: Flit = net_info["inject_fifo"].popleft()
            flit.flit_position = "L2H_FIFO"
            flit.start_inject = True
            net_info["l2h_fifo_pre"] = flit

    def l2h_to_IQ_channel_buffer(self, network_type):
        """2GHz: l2h_fifo → network.IQ_channel_buffer"""
        net_info = self.networks[network_type]
        network: Network = net_info["network"]

        if not net_info["l2h_fifo"]:
            return

        # 检查目标缓冲区是否已满（只能在 *pre* 缓冲区为空且正式 FIFO 未满时移动）
        fifo = network.IQ_channel_buffer[self.ip_type][self.ip_pos]
        fifo_pre = network.IQ_channel_buffer_pre[self.ip_type][self.ip_pos]
        if len(fifo) >= getattr(self.config, "IQ_CH_FIFO_DEPTH", 8) or fifo_pre is not None:
            return  # 没空间，或 pre 槽已占用

        # 从 l2h_fifo 弹出一个 flit，先放到 *pre* 槽
        flit = net_info["l2h_fifo"].popleft()
        flit.flit_position = "IQ_CH"
        network.IQ_channel_buffer_pre[self.ip_type][self.ip_pos] = flit

    def _check_and_reserve_resources(self, req):
        """检查并预占新请求所需的资源"""
        ip_pos, ip_type = self.ip_pos, self.ip_type

        try:
            if req.req_type == "read":
                # 检查读请求资源
                rdb_available = self.node.rn_rdb_count[ip_type][ip_pos] >= req.burst_length
                tracker_available = self.node.rn_tracker_count["read"][ip_type][ip_pos] > 0
                reserve_ok = self.node.rn_rdb_count[ip_type][ip_pos] > self.node.rn_rdb_reserve[ip_type][ip_pos] * req.burst_length

                if not (rdb_available and tracker_available and reserve_ok):
                    return False

                # 预占资源
                self.node.rn_rdb_count[ip_type][ip_pos] -= req.burst_length
                self.node.rn_tracker_count["read"][ip_type][ip_pos] -= 1
                self.node.rn_rdb[ip_type][ip_pos][req.packet_id] = []
                self.node.rn_tracker["read"][ip_type][ip_pos].append(req)
                self.node.rn_tracker_pointer["read"][ip_type][ip_pos] += 1

            elif req.req_type == "write":
                # 检查写请求资源
                wdb_available = self.node.rn_wdb_count[ip_type][ip_pos] >= req.burst_length
                tracker_available = self.node.rn_tracker_count["write"][ip_type][ip_pos] > 0

                if not (wdb_available and tracker_available):
                    return False

                # 预占资源
                self.node.rn_wdb_count[ip_type][ip_pos] -= req.burst_length
                self.node.rn_tracker_count["write"][ip_type][ip_pos] -= 1
                self.node.rn_wdb[ip_type][ip_pos][req.packet_id] = []
                self.node.rn_tracker["write"][ip_type][ip_pos].append(req)
                self.node.rn_tracker_pointer["write"][ip_type][ip_pos] += 1

                # 创建写数据包
                self.create_write_packet(req)

            return True

        except (KeyError, AttributeError) as e:
            logging.warning(f"Resource check failed for {req}: {e}")
            return False

    def EQ_channel_buffer_to_h2l_pre(self, network_type):
        """2GHz: network.EQ_channel_buffer → h2l_fifo_pre"""
        net_info = self.networks[network_type]
        network = net_info["network"]

        if net_info["h2l_fifo_pre"] is not None:
            return  # 预缓冲已占用

        try:
            # 接收，使用 ip_pos - NUM_COL
            pos_index = self.ip_pos - self.config.NUM_COL

            eq_buf = network.EQ_channel_buffer[self.ip_type][pos_index]
            if not eq_buf:
                return

            # 检查h2l_fifo是否有空间
            if len(net_info["h2l_fifo"]) >= net_info["h2l_fifo"].maxlen:
                return

            flit = eq_buf.popleft()
            flit.is_arrive = True
            flit.flit_position = "H2L_FIFO"
            net_info["h2l_fifo_pre"] = flit
            net_info["network"].arrive_flits[flit.packet_id].append(flit)
            net_info["network"].recv_flits_num += 1

            # # 根据网络类型进行特殊处理
            # if network_type == "req":
            #     self._handle_received_request(flit)
            # elif network_type == "rsp":
            #     self._handle_received_response(flit)
            # elif network_type == "data":
            #     self._handle_received_data(flit)

        except (KeyError, AttributeError) as e:
            logging.warning(f"EQ to h2l transfer failed for {network_type}: {e}")

    def _handle_received_request(self, req: Flit):
        """处理接收到的请求（SN端）"""
        req.sn_receive_req_cycle = getattr(self, "current_cycle", 0)

        if req.req_type == "read":
            if req.req_attr == "new":
                if self.node.sn_tracker_count[self.ip_type]["ro"][self.ip_pos] > 0:
                    req.sn_tracker_type = "ro"
                    self.node.sn_tracker[self.ip_type][self.ip_pos].append(req)
                    self.node.sn_tracker_count[self.ip_type]["ro"][self.ip_pos] -= 1
                    self.create_read_packet(req)
                    self.release_completed_sn_tracker(req)
                else:
                    self.create_rsp(req, "negative")
                    self.node.sn_req_wait[req.req_type][self.ip_type][self.ip_pos].append(req)
            else:
                self.create_read_packet(req)
                self.release_completed_sn_tracker(req)

        elif req.req_type == "write":
            if req.req_attr == "new":
                if self.node.sn_tracker_count[self.ip_type]["share"][self.ip_pos] > 0 and self.node.sn_wdb_count[self.ip_type][self.ip_pos] >= req.burst_length:
                    req.sn_tracker_type = "share"
                    self.node.sn_tracker[self.ip_type][self.ip_pos].append(req)
                    self.node.sn_tracker_count[self.ip_type]["share"][self.ip_pos] -= 1
                    self.node.sn_wdb[self.ip_type][self.ip_pos][req.packet_id] = []
                    self.node.sn_wdb_count[self.ip_type][self.ip_pos] -= req.burst_length
                    self.create_rsp(req, "datasend")
                else:
                    self.create_rsp(req, "negative")
                    self.node.sn_req_wait[req.req_type][self.ip_type][self.ip_pos].append(req)
            else:
                self.create_rsp(req, "datasend")

    def _handle_received_response(self, rsp: Flit):
        """处理接收到的响应（RN端）"""
        rsp.sn_receive_req_cycle = getattr(self, "current_cycle", 0)

        req = next((req for req in self.node.rn_tracker[rsp.req_type][self.ip_type][self.ip_pos] if req.packet_id == rsp.packet_id), None)
        if not req:
            logging.warning(f"RSP {rsp} do not have REQ")
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
                self.node.rn_rdb_count[self.ip_type][self.ip_pos] += req.burst_length
                if req.packet_id in self.node.rn_rdb[self.ip_type][self.ip_pos]:
                    self.node.rn_rdb[self.ip_type][self.ip_pos].pop(req.packet_id)
                self.node.rn_rdb_reserve[self.ip_type][self.ip_pos] += 1

            elif rsp.rsp_type == "positive":
                # 处理读重试
                if req.req_attr == "new":
                    self.node.rn_rdb_count[self.ip_type][self.ip_pos] += req.burst_length
                    if req.packet_id in self.node.rn_rdb[self.ip_type][self.ip_pos]:
                        self.node.rn_rdb[self.ip_type][self.ip_pos].pop(req.packet_id)
                    self.node.rn_rdb_reserve[self.ip_type][self.ip_pos] += 1
                req.req_state = "valid"
                req.req_attr = "old"
                req.is_injected = False
                req.path_index = 0
                req.is_new_on_network = True
                req.is_arrive = False
                # 放入请求网络的inject_fifo
                self.enqueue(req, "req", retry=True)
                self.node.rn_rdb_reserve[self.ip_type][self.ip_pos] -= 1

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
                req.is_new_on_network = True
                req.is_arrive = False
                # 放入请求网络的inject_fifo
                self.enqueue(req, "req", retry=True)

            elif rsp.rsp_type == "datasend":
                # 写数据发送，将写数据包放入数据网络inject_fifo
                for flit in self.node.rn_wdb[self.ip_type][self.ip_pos][rsp.packet_id]:
                    self.enqueue(flit, "data")
                # Enqueue 完所有写数据后 —— 释放 RN write‐tracker
                req: Flit = next((r for r in self.node.rn_tracker["write"][self.ip_type][self.ip_pos] if r.packet_id == rsp.packet_id), None)
                if req:
                    self.node.rn_tracker["write"][self.ip_type][self.ip_pos].remove(req)
                    self.node.rn_wdb_count[self.ip_type][self.ip_pos] += req.burst_length
                    self.node.rn_tracker_count["write"][self.ip_type][self.ip_pos] += 1
                    self.node.rn_tracker_pointer["write"][self.ip_type][self.ip_pos] -= 1
                # 同时清理写缓冲
                self.node.rn_wdb[self.ip_type][self.ip_pos].pop(rsp.packet_id, None)

    def release_completed_sn_tracker(self, req: Flit):
        # —— 1) 移除已完成的 tracker ——
        self.node.sn_tracker[self.ip_type][self.ip_pos].remove(req)
        # 释放一个 tracker 槽
        self.node.sn_tracker_count[self.ip_type][req.sn_tracker_type][self.ip_pos] += 1

        # —— 2) 对于写请求，还要释放写缓冲额度 ——
        if req.req_type == "write":
            self.node.sn_wdb_count[self.ip_type][self.ip_pos] += req.burst_length

        # —— 3) 尝试给等待队列里的请求重新分配资源 ——
        wait_list = self.node.sn_req_wait[req.req_type][self.ip_type][self.ip_pos]
        if not wait_list:
            return

        if req.req_type == "write":
            # 写：既要有空 tracker，也要有足够 wdb_count
            if self.node.sn_tracker_count[self.ip_type][req.sn_tracker_type][self.ip_pos] > 0 and self.node.sn_wdb_count[self.ip_type][self.ip_pos] > 0:
                new_req = wait_list.pop(0)
                new_req.sn_tracker_type = req.sn_tracker_type

                # 分配 tracker + wdb
                self.node.sn_tracker[self.ip_type][self.ip_pos].append(new_req)
                self.node.sn_tracker_count[self.ip_type][new_req.sn_tracker_type][self.ip_pos] -= 1

                self.node.sn_wdb_count[self.ip_type][self.ip_pos] -= new_req.burst_length

                # 发送 positive 响应
                self.create_rsp(new_req, "positive")

        elif req.req_type == "read":
            # 读：只要有空 tracker 即可
            if self.node.sn_tracker_count[self.ip_type][req.sn_tracker_type][self.ip_pos] > 0:
                new_req = wait_list.pop(0)
                new_req.sn_tracker_type = req.sn_tracker_type

                # 分配 tracker
                self.node.sn_tracker[self.ip_type][self.ip_pos].append(new_req)
                self.node.sn_tracker_count[self.ip_type][new_req.sn_tracker_type][self.ip_pos] -= 1

                # 直接生成并发送读数据包
                self.create_read_packet(new_req)

    def _handle_received_data(self, flit: Flit):
        """处理接收到的数据"""
        flit.arrival_cycle = getattr(self, "current_cycle", 0)
        if flit.req_type == "read":
            # 读数据到达RN端，需要收集到data buffer中
            self.node.rn_rdb[self.ip_type][self.ip_pos][flit.packet_id].append(flit)
            # 检查是否收集完整个burst
            if len(self.node.rn_rdb[self.ip_type][self.ip_pos][flit.packet_id]) == flit.burst_length:
                # 更新统计
                flit.rn_data_collection_complete_cycle = self.current_cycle

                req = next((req for req in self.node.rn_tracker["read"][self.ip_type][self.ip_pos] if req.packet_id == flit.packet_id), None)
                if req:
                    # 立即释放tracker和更新计数
                    self.node.rn_tracker["read"][self.ip_type][self.ip_pos].remove(req)
                    self.node.rn_tracker_count["read"][self.ip_type][self.ip_pos] += 1
                    self.node.rn_tracker_pointer["read"][self.ip_type][self.ip_pos] -= 1
                    self.node.rn_rdb_count[self.ip_type][self.ip_pos] += req.burst_length
                    # 为所有flit设置完成时间戳
                    for f in self.node.rn_rdb[self.ip_type][self.ip_pos][flit.packet_id]:
                        f.leave_db_cycle = self.current_cycle

                    # 清理data buffer（数据已经收集完成）
                    self.node.rn_rdb[self.ip_type][self.ip_pos].pop(flit.packet_id)

                else:
                    print(f"Warning: No RN tracker found for packet_id {flit.packet_id}")

        elif flit.req_type == "write":
            self.node.sn_wdb[self.ip_type][self.ip_pos][flit.packet_id].append(flit)

            # 检查是否收集完整个burst
            if len(self.node.sn_wdb[self.ip_type][self.ip_pos][flit.packet_id]) == flit.burst_length:
                # 完整burst到达，设置完成时间戳
                for f in self.node.sn_wdb[self.ip_type][self.ip_pos][flit.packet_id]:
                    f.sn_data_collection_complete_cycle = self.current_cycle

                # 找到对应的请求
                req = next((req for req in self.node.sn_tracker[self.ip_type][self.ip_pos] if req.packet_id == flit.packet_id), None)
                if req:
                    # 设置tracker延迟释放时间
                    release_time = self.current_cycle + self.config.SN_TRACKER_RELEASE_LATENCY

                    # 初始化释放时间字典（如果不存在）
                    if not hasattr(self.node, "sn_tracker_release_time"):
                        self.node.sn_tracker_release_time = defaultdict(list)

                    # 更新flit时间戳
                    for f in self.node.sn_wdb[self.ip_type][self.ip_pos][flit.packet_id]:
                        f.leave_db_cycle = release_time

                    # 清理data buffer（数据已经收集完成）
                    self.node.sn_wdb[self.ip_type][self.ip_pos].pop(flit.packet_id)

                    # 添加到延迟释放队列
                    self.node.sn_tracker_release_time[release_time].append((self.ip_type, self.ip_pos, req))
                else:
                    print(f"Warning: No SN tracker found for packet_id {flit.packet_id}")

    def h2l_to_eject_fifo(self, network_type):
        """1GHz: h2l_fifo → 处理完成"""
        net_info = self.networks[network_type]
        if not net_info["h2l_fifo"]:
            return

        current_cycle = getattr(self, "current_cycle", 0)
        if network_type == "data" and self.token_bucket:
            self.token_bucket.refill(current_cycle)
            if not self.token_bucket.consume():
                return

        flit = net_info["h2l_fifo"].popleft()
        flit.flit_position = "IP_eject"
        flit.is_finish = True
        # 根据网络类型进行特殊处理
        if network_type == "req":
            self._handle_received_request(flit)
        elif network_type == "rsp":
            self._handle_received_response(flit)
        elif network_type == "data":
            self._handle_received_data(flit)

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
            net: Network = net_info["network"]
            # l2h_fifo_pre → l2h_fifo
            if net_info["l2h_fifo_pre"] is not None and len(net_info["l2h_fifo"]) < net_info["l2h_fifo"].maxlen:
                net_info["l2h_fifo"].append(net_info["l2h_fifo_pre"])
                net_info["l2h_fifo_pre"] = None

            if net.IQ_channel_buffer_pre[self.ip_type][self.ip_pos] is not None and len(net.IQ_channel_buffer[self.ip_type][self.ip_pos]) < net.IQ_channel_buffer[self.ip_type][self.ip_pos].maxlen:
                net.IQ_channel_buffer[self.ip_type][self.ip_pos].append(net.IQ_channel_buffer_pre[self.ip_type][self.ip_pos])
                net.IQ_channel_buffer_pre[self.ip_type][self.ip_pos] = None

            if net_info["h2l_fifo_pre"] is not None and len(net_info["h2l_fifo"]) < net_info["h2l_fifo"].maxlen:
                net_info["h2l_fifo"].append(net_info["h2l_fifo_pre"])
                net_info["h2l_fifo_pre"] = None

    def eject_step(self, cycle):
        """根据周期和频率调用eject相应的方法"""
        self.current_cycle = cycle
        cycle_mod = cycle % self.config.NETWORK_FREQUENCY

        # 2GHz 操作（每半个网络周期执行一次）
        for net_type in ["req", "rsp", "data"]:
            self.EQ_channel_buffer_to_h2l_pre(net_type)

        # 1GHz 操作（每个网络周期执行一次）
        if cycle_mod == 0:
            # 对三个网络分别执行h2l_to_eject_fifo
            for net_type in ["req", "rsp", "data"]:
                self.h2l_to_eject_fifo(net_type)

    def create_write_packet(self, req):
        """创建写数据包并放入数据网络inject_fifo"""
        cycle = getattr(self, "current_cycle", 0)
        for i in range(req.burst_length):
            source = req.source
            destination = req.destination
            path = self.routes[source][destination]
            flit = Flit(source, destination, path)
            flit.source_original = req.source_original
            flit.destination_original = req.destination_original
            flit.flit_type = "data"
            flit.departure_cycle = (
                cycle + self.config.DDR_W_LATENCY + i * self.config.NETWORK_FREQUENCY
                if req.original_destination_type.startswith("ddr")
                else cycle + self.config.L2M_W_LATENCY + i * self.config.NETWORK_FREQUENCY
            )
            flit.req_departure_cycle = req.departure_cycle
            flit.entry_db_cycle = req.entry_db_cycle
            flit.source_type = req.source_type
            flit.destination_type = req.destination_type
            flit.original_source_type = req.original_source_type
            flit.original_destination_type = req.original_destination_type
            flit.req_type = req.req_type
            flit.packet_id = req.packet_id
            flit.flit_id = i
            flit.burst_length = req.burst_length
            if i == req.burst_length - 1:
                flit.is_last_flit = True
            flit.rn_data_generated_cycle = cycle

            # 将写数据包放入wdb中
            self.node.rn_wdb[self.ip_type][self.ip_pos][flit.packet_id].append(flit)

    def create_read_packet(self, req):
        """创建读数据包并放入数据网络inject_fifo"""
        cycle = getattr(self, "current_cycle", 0)
        for i in range(req.burst_length):
            source = req.destination + self.config.NUM_COL
            destination = req.source - self.config.NUM_COL
            path = self.routes[source][destination]
            flit = Flit(source, destination, path)
            flit.source_original = req.destination_original
            flit.destination_original = req.source_original
            flit.req_type = req.req_type
            flit.flit_type = "data"
            if req.original_destination_type.startswith("ddr"):
                latency = np.random.uniform(low=self.config.DDR_R_LATENCY - self.config.DDR_R_LATENCY_VAR, high=self.config.DDR_R_LATENCY + self.config.DDR_R_LATENCY_VAR, size=None)
            else:
                latency = self.config.L2M_R_LATENCY
            flit.departure_cycle = cycle + latency + i * self.config.NETWORK_FREQUENCY
            flit.entry_db_cycle = cycle
            flit.req_departure_cycle = req.departure_cycle
            flit.source_type = req.destination_type
            flit.destination_type = req.source_type
            flit.original_source_type = req.original_source_type
            flit.original_destination_type = req.original_destination_type
            flit.packet_id = req.packet_id
            flit.flit_id = i
            flit.burst_length = req.burst_length
            if i == req.burst_length - 1:
                flit.is_last_flit = True
            flit.sync_latency_record(req)
            flit.sn_data_generated_cycle = cycle

            # 将读数据包放入数据网络的inject_fifo
            self.enqueue(flit, "data")

    def create_rsp(self, req, rsp_type):
        """创建响应并放入响应网络inject_fifo"""
        cycle = getattr(self, "current_cycle", 0)
        source = req.destination + self.config.NUM_COL
        destination = req.source - self.config.NUM_COL
        path = self.routes[source][destination]
        rsp = Flit(source, destination, path)
        rsp.flit_type = "rsp"
        rsp.rsp_type = rsp_type
        rsp.req_type = req.req_type
        rsp.packet_id = req.packet_id
        rsp.departure_cycle = cycle + self.config.NETWORK_FREQUENCY
        rsp.req_departure_cycle = req.departure_cycle
        rsp.source_type = req.destination_type
        rsp.destination_type = req.source_type
        rsp.sync_latency_record(req)
        rsp.sn_rsp_generate_cycle = cycle

        # 将响应放入响应网络的inject_fifo
        self.enqueue(rsp, "rsp")


class Network:
    def __init__(self, config: CrossRingConfig, adjacency_matrix, name="network"):
        self.config = config
        self.name = name
        self.current_cycle = []
        self.flits_num = []
        self.schedules = {"sdma": None}
        self.inject_num = 0
        self.eject_num = 0
        self.inject_queues = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.inject_queues_pre = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.eject_queues = {"TU": {}, "TD": {}}
        self.eject_queues_in_pre = {"TU": {}, "TD": {}}
        self.arrive_node_pre = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.IQ_channel_buffer = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.EQ_channel_buffer = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.IQ_channel_buffer_pre = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.EQ_channel_buffer_pre = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.links = {}
        self.cross_point = {"horizontal": defaultdict(lambda: defaultdict(list)), "vertical": defaultdict(lambda: defaultdict(list))}
        self.links_flow_stat = {"read": {}, "write": {}}
        # ITag setup
        self.links_tag = {}
        self.remain_tag = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}
        self.tagged_counter = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}  # 环上已标记ITag数
        self.itag_req_counter = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}  # FIFO中ITag需求数
        self.excess_ITag_to_remove = {"TL": {}, "TR": {}, "TD": {}, "TU": {}}

        # 每个FIFO Entry的等待计数器
        self.fifo_counters = {"TL": {}, "TR": {}}
        self.ring_bridge = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.ring_bridge_pre = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.round_robin = {"IQ": defaultdict(lambda: defaultdict(dict)), "RB": defaultdict(lambda: defaultdict(dict)), "EQ": defaultdict(lambda: defaultdict(dict))}
        self.round_robin_counter = 0

        self.recv_flits_num = 0
        self.send_flits = defaultdict(list)
        self.arrive_flits = defaultdict(list)
        self.all_latency = []
        self.ject_latency = []
        self.network_latency = []
        self.predicted_recv_time = []
        self.inject_time = {}
        self.eject_time = {}
        self.avg_inject_time = {}
        self.avg_eject_time = {}
        self.predicted_avg_latency = None
        self.predicted_max_latency = None
        self.actual_avg_latency = None
        self.actual_max_latency = None
        self.actual_avg_ject_latency = None
        self.actual_max_ject_latency = None
        self.actual_avg_net_latency = None
        self.actual_max_net_latency = None
        #
        self.circuits_h = []
        self.circuits_v = []
        self.avg_circuits_h = None
        self.max_circuits_h = None
        self.avg_circuits_v = None
        self.max_circuits_v = None
        self.circuits_flit_h = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.circuits_flit_v = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.gdma_recv = 0
        self.gdma_remainder = 0
        self.gdma_count = 512
        self.l2m_recv = 0
        self.l2m_remainder = 0
        self.sdma_send = []
        self.num_send = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.num_recv = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.per_send_throughput = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.per_recv_throughput = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.send_throughput = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.recv_throughput = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.last_select = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.throughput = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))

        # # channel buffer setup

        self.ring_bridge_map = {
            0: ("TL", self.config.RB_IN_FIFO_DEPTH),
            1: ("TR", self.config.RB_IN_FIFO_DEPTH),
            -1: ("IQ_TU", self.config.IQ_OUT_FIFO_DEPTH),
            -2: ("IQ_TD", self.config.IQ_OUT_FIFO_DEPTH),
        }

        self.token_bucket = defaultdict(dict)
        self.flit_size_bytes = 128
        for ch_name in self.IQ_channel_buffer.keys():
            for ip_pos in set(self.config.DDR_SEND_POSITION_LIST + self.config.L2M_SEND_POSITION_LIST):

                if ch_name.startswith("ddr"):
                    self.token_bucket[ip_pos][ch_name] = TokenBucket(
                        rate=self.config.DDR_BW_LIMIT / self.config.NETWORK_FREQUENCY / self.flit_size_bytes,
                        bucket_size=self.config.DDR_BW_LIMIT,
                    )
                    self.token_bucket[ip_pos - self.config.NUM_COL][ch_name] = TokenBucket(
                        rate=self.config.DDR_BW_LIMIT / self.config.NETWORK_FREQUENCY / self.flit_size_bytes,
                        bucket_size=self.config.DDR_BW_LIMIT,
                    )
                elif ch_name.startswith("l2m"):
                    self.token_bucket[ip_pos][ch_name] = TokenBucket(
                        rate=self.config.L2M_BW_LIMIT / self.config.NETWORK_FREQUENCY / self.flit_size_bytes,
                        bucket_size=self.config.L2M_BW_LIMIT,
                    )

        # ETag setup
        self.T0_Etag_Order_FIFO = deque()  # 用于轮询选择 T0 Flit 的 Order FIFO
        self.RB_UE_Counters = {"TL": {}, "TR": {}}
        self.EQ_UE_Counters = {"TU": {}, "TD": {}}
        self.ETag_BOTHSIDE_UPGRADE = False

        for ip_pos in set(config.DDR_SEND_POSITION_LIST + config.SDMA_SEND_POSITION_LIST + config.L2M_SEND_POSITION_LIST + config.GDMA_SEND_POSITION_LIST):
            self.cross_point["horizontal"][ip_pos]["TL"] = [None] * 2
            self.cross_point["horizontal"][ip_pos]["TR"] = [None] * 2
            self.cross_point["vertical"][ip_pos]["TU"] = [None] * 2
            self.cross_point["vertical"][ip_pos]["TD"] = [None] * 2
            self.inject_queues["TL"][ip_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH)
            self.inject_queues["TR"][ip_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH)
            self.inject_queues["TU"][ip_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH)
            self.inject_queues["TD"][ip_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH)
            self.inject_queues["EQ"][ip_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH)
            self.inject_queues_pre["TL"][ip_pos] = None
            self.inject_queues_pre["TR"][ip_pos] = None
            self.inject_queues_pre["TU"][ip_pos] = None
            self.inject_queues_pre["TD"][ip_pos] = None
            self.inject_queues_pre["EQ"][ip_pos] = None
            for key in self.config.CH_NAME_LIST:
                self.IQ_channel_buffer_pre[key][ip_pos] = None
                self.EQ_channel_buffer_pre[key][ip_pos - config.NUM_COL] = None
            for key in self.arrive_node_pre:
                self.arrive_node_pre[key][ip_pos - config.NUM_COL] = None
            self.eject_queues["TU"][ip_pos - config.NUM_COL] = deque(maxlen=config.EQ_IN_FIFO_DEPTH)
            self.eject_queues["TD"][ip_pos - config.NUM_COL] = deque(maxlen=config.EQ_IN_FIFO_DEPTH)
            self.eject_queues_in_pre["TU"][ip_pos - config.NUM_COL] = None
            self.eject_queues_in_pre["TD"][ip_pos - config.NUM_COL] = None
            self.EQ_UE_Counters["TU"][ip_pos - config.NUM_COL] = {"T2": 0, "T1": 0, "T0": 0}
            self.EQ_UE_Counters["TD"][ip_pos - config.NUM_COL] = {"T2": 0, "T1": 0}

            for key in self.round_robin.keys():
                if key == "IQ":
                    for fifo_name in ["TR", "TL", "TU", "TD", "EQ"]:
                        self.round_robin[key][fifo_name][ip_pos - config.NUM_COL] = deque()
                        for ch_name in self.IQ_channel_buffer.keys():
                            self.round_robin[key][fifo_name][ip_pos - config.NUM_COL].append(ch_name)
                elif key == "EQ":
                    for ch_name in self.IQ_channel_buffer.keys():
                        self.round_robin[key][ch_name][ip_pos - config.NUM_COL] = deque([0, 1, 2, 3])
                else:
                    for fifo_name in ["TU", "TD", "EQ"]:
                        self.round_robin[key][fifo_name][ip_pos - config.NUM_COL] = deque([0, 1, 2, 3])

            self.inject_time[ip_pos] = []
            self.eject_time[ip_pos - config.NUM_COL] = []
            self.avg_inject_time[ip_pos] = 0
            self.avg_eject_time[ip_pos - config.NUM_COL] = 1

        for i in range(config.NUM_NODE):
            for j in range(config.NUM_NODE):
                if adjacency_matrix[i][j] == 1 and i - j != config.NUM_COL:
                    self.links[(i, j)] = [None] * config.SLICE_PER_LINK
                    self.links_flow_stat["read"][(i, j)] = 0
                    self.links_flow_stat["write"][(i, j)] = 0
                    self.links_tag[(i, j)] = [None] * config.SLICE_PER_LINK
            if i in range(0, config.NUM_COL):
                self.links[(i, i)] = [None] * 2
                self.links[(i + config.NUM_NODE - config.NUM_COL * 2, i + config.NUM_NODE - config.NUM_COL * 2)] = [None] * 2
                self.links_flow_stat["read"][(i, i)] = 0
                self.links_flow_stat["write"][(i, i)] = 0
                self.links_flow_stat["read"][(i + config.NUM_NODE - config.NUM_COL * 2, i + config.NUM_NODE - config.NUM_COL * 2)] = 0
                self.links_flow_stat["write"][(i + config.NUM_NODE - config.NUM_COL * 2, i + config.NUM_NODE - config.NUM_COL * 2)] = 0
                self.links_tag[(i, i)] = [None] * 2
                self.links_tag[(i + config.NUM_NODE - config.NUM_COL * 2, i + config.NUM_NODE - config.NUM_COL * 2)] = [None] * 2
            if i % config.NUM_COL == 0 and (i // config.NUM_COL) % 2 != 0:
                self.links[(i, i)] = [None] * 2
                self.links[(i + config.NUM_COL - 1, i + config.NUM_COL - 1)] = [None] * 2
                self.links_flow_stat["read"][(i, i)] = 0
                self.links_flow_stat["write"][(i, i)] = 0
                self.links_flow_stat["read"][(i + config.NUM_COL - 1, i + config.NUM_COL - 1)] = 0
                self.links_flow_stat["write"][(i + config.NUM_COL - 1, i + config.NUM_COL - 1)] = 0
                self.links_tag[(i, i)] = [None] * 2
                self.links_tag[(i + config.NUM_COL - 1, i + config.NUM_COL - 1)] = [None] * 2

        for row in range(1, config.NUM_ROW, 2):
            for col in range(config.NUM_COL):
                pos = row * config.NUM_COL + col
                next_pos = pos - config.NUM_COL
                self.ring_bridge["TL"][(pos, next_pos)] = deque(maxlen=config.RB_IN_FIFO_DEPTH)
                self.ring_bridge["TR"][(pos, next_pos)] = deque(maxlen=config.RB_IN_FIFO_DEPTH)
                self.ring_bridge["TU"][(pos, next_pos)] = deque(maxlen=config.RB_OUT_FIFO_DEPTH)
                self.ring_bridge["TD"][(pos, next_pos)] = deque(maxlen=config.RB_OUT_FIFO_DEPTH)
                self.ring_bridge["EQ"][(pos, next_pos)] = deque(maxlen=config.RB_OUT_FIFO_DEPTH)

                self.ring_bridge_pre["TL"][(pos, next_pos)] = None
                self.ring_bridge_pre["TR"][(pos, next_pos)] = None
                self.ring_bridge_pre["TU"][(pos, next_pos)] = None
                self.ring_bridge_pre["TD"][(pos, next_pos)] = None
                self.ring_bridge_pre["EQ"][(pos, next_pos)] = None

                self.RB_UE_Counters["TL"][(pos, next_pos)] = {"T2": 0, "T1": 0, "T0": 0}
                self.RB_UE_Counters["TR"][(pos, next_pos)] = {"T2": 0, "T1": 0}
                # self.round_robin["TU"][next_pos] = deque([0, 1, 2])
                # self.round_robin["TD"][next_pos] = deque([0, 1, 2])
                # self.round_robin["RB"][next_pos] = deque([0, 1, 2])
                for direction in ["TL", "TR"]:
                    self.remain_tag[direction][pos] = config.ITag_MAX_NUM_H
                    self.itag_req_counter[direction][pos] = 0
                    self.tagged_counter[direction][pos] = 0
                    self.excess_ITag_to_remove[direction][pos] = 0
                for direction in ["TU", "TD"]:
                    self.remain_tag[direction][pos] = config.ITag_MAX_NUM_V
                    self.itag_req_counter[direction][pos] = 0
                    self.tagged_counter[direction][pos] = 0
                    self.excess_ITag_to_remove[direction][pos] = 0

        for ip_type in self.num_recv.keys():
            source_positions = getattr(config, f"{ip_type[:-2].upper()}_SEND_POSITION_LIST")
            for source in source_positions:
                destination = source - config.NUM_COL
                self.num_send[ip_type][source] = 0
                self.num_recv[ip_type][destination] = 0
                self.per_send_throughput[ip_type][source] = 0
                self.per_recv_throughput[ip_type][destination] = 0

        for ip_type in self.IQ_channel_buffer.keys():
            for ip_index in getattr(config, f"{ip_type[:-2].upper()}_SEND_POSITION_LIST"):
                ip_recv_index = ip_index - config.NUM_COL
                self.IQ_channel_buffer[ip_type][ip_index] = deque(maxlen=config.IQ_CH_FIFO_DEPTH)
                self.EQ_channel_buffer[ip_type][ip_recv_index] = deque(maxlen=config.EQ_CH_FIFO_DEPTH)
        for ip_type in self.last_select.keys():
            for ip_index in getattr(config, f"{ip_type[:-2].upper()}_SEND_POSITION_LIST"):
                self.last_select[ip_type][ip_index] = "write"
        for ip_type in self.throughput.keys():
            for ip_index in getattr(config, f"{ip_type[:-2].upper()}_SEND_POSITION_LIST"):
                self.throughput[ip_type][ip_index] = [0, 0, 10000000, 0]

        self.RB_CAPACITY = {"TL": {}, "TR": {}}
        self.EQ_CAPACITY = {"TU": {}, "TD": {}}

        # TL capacity
        def _cap_tl(lvl):
            if lvl == "T2":
                return self.config.TL_Etag_T2_UE_MAX
            if lvl == "T1":
                return self.config.TL_Etag_T1_UE_MAX - self.config.TL_Etag_T2_UE_MAX
            if lvl == "T0":
                return self.config.RB_IN_FIFO_DEPTH - self.config.TL_Etag_T1_UE_MAX

        # TR capacity
        def _cap_tr(lvl):
            if lvl == "T2":
                return self.config.TR_Etag_T2_UE_MAX
            if lvl == "T1":
                return self.config.RB_IN_FIFO_DEPTH - self.config.TR_Etag_T2_UE_MAX
            return 0  # TR 无 T0

        # TU capacity
        def _cap_tu(lvl):
            if lvl == "T2":
                return self.config.TU_Etag_T2_UE_MAX
            if lvl == "T1":
                return self.config.TU_Etag_T1_UE_MAX - self.config.TU_Etag_T2_UE_MAX
            if lvl == "T0":
                return self.config.EQ_IN_FIFO_DEPTH - self.config.TU_Etag_T1_UE_MAX

        # TD capacity
        def _cap_td(lvl):
            if lvl == "T2":
                return self.config.TD_Etag_T2_UE_MAX
            if lvl == "T1":
                return self.config.EQ_IN_FIFO_DEPTH - self.config.TD_Etag_T2_UE_MAX
            return 0  # TD 无 T0

        for pair in self.RB_UE_Counters["TL"]:
            self.RB_CAPACITY["TL"][pair] = {lvl: _cap_tl(lvl) for lvl in ("T0", "T1", "T2")}
        for pair in self.RB_UE_Counters["TR"]:
            self.RB_CAPACITY["TR"][pair] = {lvl: _cap_tr(lvl) for lvl in ("T1", "T2")}

        for pos in self.EQ_UE_Counters["TU"]:
            self.EQ_CAPACITY["TU"][pos] = {lvl: _cap_tu(lvl) for lvl in ("T0", "T1", "T2")}
        for pos in self.EQ_UE_Counters["TD"]:
            self.EQ_CAPACITY["TD"][pos] = {lvl: _cap_td(lvl) for lvl in ("T1", "T2")}

    def _entry_available(self, dir_type, key, level):
        if dir_type in ("TL", "TR"):
            cap = self.RB_CAPACITY[dir_type][key][level]
            occ = self.RB_UE_Counters[dir_type][key][level]
        else:
            cap = self.EQ_CAPACITY[dir_type][key][level]
            occ = self.EQ_UE_Counters[dir_type][key][level]
        return occ < cap

    # ------------------------------------------------------------------
    def _occupy_entry(self, dir_type, key, level, flit):
        """
        统一处理占用计数器，并记录 flit.used_entry_level
        dir_type : "TL"|"TR"|"TU"|"TD"
        key      : (cur,next) for RB  or  dest_pos for EQ
        level    : "T0"/"T1"/"T2"
        """
        if dir_type in ("TL", "TR"):
            self.RB_UE_Counters[dir_type][key][level] += 1
            # flit.flit_position = f"RB_{dir_type}"
        else:
            self.EQ_UE_Counters[dir_type][key][level] += 1
            # flit.flit_position = f"EQ_{dir_type}"
        flit.used_entry_level = level

    def error_log(self, flit, target_id, flit_id):
        if flit and flit.packet_id == target_id and flit.flit_id == flit_id:
            print(inspect.currentframe().f_back.f_code.co_name, self.cycle, flit)

    def set_link_slice(self, link: tuple[int, int], slice_index: int, flit: "Flit", cycle, *, override: bool = False):
        """
        Safely assign a flit to a given slice on a link.

        Parameters
        ----------
        link : tuple[int, int]
            (u, v) node indices of the directed link.
        index : int
            Slice index on that link (0 == head).
        flit : Flit
            The flit object to place.
        override : bool, optional
            If True, forcibly override the existing flit (logging a warning).
            If False (default), raise RuntimeError when the slot is occupied.

        Raises
        ------
        RuntimeError
            When the target slice is already occupied and override == False.
        """
        try:
            current = self.links[link][slice_index]
        except KeyError as e:
            raise KeyError(f"Link {link} does not exist in Network '{self.name}'") from e
        except IndexError as e:
            raise IndexError(f"Slice index {slice_index} out of range for link {link}") from e

        if current is not None and not override:
            raise RuntimeError(f"[Cycle {cycle}] " f"Attempt to assign flit {flit} to occupied " f"link {link}[{slice_index}] already holding flit {current}")

        if current is not None and override:
            logging.warning(f"[Cycle {cycle}] " f"Overriding link {link}[{slice_index}] flit {current.packet_id}.{current.flit_id} " f"with {flit.packet_id}.{flit.flit_id}")

        self.links[link][slice_index] = flit

    def can_move_to_next(self, flit, current, next_node):
        # 1. flit不进入Cross Poing
        if flit.source - flit.destination == self.config.NUM_COL:
            return len(self.inject_queues["EQ"]) < self.config.IQ_OUT_FIFO_DEPTH
        elif current - next_node == self.config.NUM_COL:
            # 向 Ring Bridge 移动. v1.3 在IQ中分TU和TD两个FIFO
            if len(flit.path) > 2 and flit.path[2] - flit.path[1] == self.config.NUM_COL * 2:
                return len(self.inject_queues["TD"][current]) < self.config.IQ_OUT_FIFO_DEPTH
            elif len(flit.path) > 2 and flit.path[2] - flit.path[1] == -self.config.NUM_COL * 2:
                return len(self.inject_queues["TU"][current]) < self.config.IQ_OUT_FIFO_DEPTH

        direction = "TR" if next_node == current + 1 else "TL"
        link = (current, next_node)

        # 横向环ITag处理
        if self.links[link][0] is not None:  # Link被占用
            # 检查是否需要标记ITag（内联所有检查逻辑）
            if (
                self.links_tag[link][0] is None
                and flit.wait_cycle_h > self.config.ITag_TRIGGER_Th_H
                and self.tagged_counter[direction][current] < self.config.ITag_MAX_NUM_H
                and self.itag_req_counter[direction][current] > 0
                and self.remain_tag[direction][current] > 0
            ):

                # 创建ITag标记（内联逻辑）
                self.remain_tag[direction][current] -= 1
                self.tagged_counter[direction][current] += 1
                self.links_tag[link][0] = [current, direction]
                flit.itag_h = True
            return False

        else:  # Link空闲
            if self.links_tag[link][0] is None:  # 无预约
                return True  # 直接上环
            else:  # 有预约
                if self.links_tag[link][0] == [current, direction]:  # 是自己的预约
                    # 使用预约（内联逻辑）
                    self.links_tag[link][0] = None
                    self.remain_tag[direction][current] += 1  # 修复：使用direction
                    self.tagged_counter[direction][current] -= 1
                    return True
        return False

    def update_excess_ITag(self):
        """在主循环中调用，处理多余ITag释放"""
        # 处理多余ITag释放（简化版）
        for direction in ["TL", "TR"]:
            for node_id in set(self.config.DDR_SEND_POSITION_LIST + self.config.SDMA_SEND_POSITION_LIST + self.config.L2M_SEND_POSITION_LIST + self.config.GDMA_SEND_POSITION_LIST):
                if self.excess_ITag_to_remove[direction][node_id] > 0:
                    # 寻找该节点创建的ITag并释放
                    for link, tag_info in self.links_tag.items():
                        if tag_info[0] is not None and tag_info[0] == [node_id, direction] and link[0] == node_id:  # ITag回到创建节点
                            # 释放多余ITag
                            self.links_tag[link][0] = None
                            self.tagged_counter[direction][node_id] -= 1
                            self.remain_tag[direction][node_id] += 1
                            self.excess_ITag_to_remove[direction][node_id] -= 1
                            break  # 一次只释放一个

    def update_cross_point(self):
        for ip_pos in set(self.config.DDR_SEND_POSITION_LIST + self.config.SDMA_SEND_POSITION_LIST + self.config.L2M_SEND_POSITION_LIST + self.config.GDMA_SEND_POSITION_LIST):
            left_pos = ip_pos - 1 if ip_pos % self.config.NUM_COL != 0 else ip_pos
            right_pos = ip_pos + 1 if ip_pos % self.config.NUM_COL != self.config.NUM_COL - 1 else ip_pos
            up_pos = ip_pos - self.config.NUM_COL * 3 if ip_pos // self.config.NUM_COL != 1 else ip_pos - self.config.NUM_COL
            down_pos = ip_pos + self.config.NUM_COL * 1 if ip_pos // self.config.NUM_COL != self.config.NUM_ROW - 1 else ip_pos - self.config.NUM_COL
            self.cross_point["horizontal"][ip_pos]["TR"] = [self.links[(left_pos, ip_pos)][-1], self.links[(ip_pos, right_pos)][0]]
            self.cross_point["horizontal"][ip_pos]["TL"] = [self.links[(ip_pos, left_pos)][0], self.links[(right_pos, ip_pos)][-1]]
            self.cross_point["vertical"][ip_pos]["TU"] = [self.links[(down_pos, ip_pos - self.config.NUM_COL)][-1], self.links[(ip_pos - self.config.NUM_COL, up_pos)][0]]
            self.cross_point["vertical"][ip_pos]["TD"] = [self.links[(ip_pos - self.config.NUM_COL, down_pos)][0], self.links[(up_pos, ip_pos - self.config.NUM_COL)][-1]]

    def plan_move(self, flit, cycle):
        self.cycle = cycle
        if flit.is_new_on_network:
            # current = flit.source
            current = flit.path[flit.path_index]
            next_node = flit.path[flit.path_index + 1]
            flit.current_position = current
            flit.is_new_on_network = False
            flit.flit_position = "Link"
            flit.is_arrive = False
            flit.is_on_station = False
            flit.current_link = (current, next_node)
            if flit.source - flit.destination == self.config.NUM_COL:
                flit.is_arrive = True
            elif current - next_node == self.config.NUM_COL:
                if len(flit.path) > 2 and flit.path[flit.path_index + 2] - next_node == 2 * self.config.NUM_COL:
                    flit.current_seat_index = -1
                elif len(flit.path) > 2 and flit.path[flit.path_index + 2] - next_node == -2 * self.config.NUM_COL:
                    flit.current_seat_index = -2
            else:
                flit.current_seat_index = 0

            return

        # 计算行和列的起始和结束点
        current, next_node = flit.current_link
        if current - next_node != self.config.NUM_COL:
            row_start = (current // self.config.NUM_COL) * self.config.NUM_COL
            row_start = row_start if (row_start // self.config.NUM_COL) % 2 != 0 else -1
            row_end = row_start + self.config.NUM_COL - 1 if row_start > 0 else -1
            col_start = current % (self.config.NUM_COL * 2)
            col_start = col_start if col_start < self.config.NUM_COL else -1
            col_end = col_start + self.config.NUM_NODE - self.config.NUM_COL * 2 if col_start >= 0 else -1

            link = self.links.get(flit.current_link)
            # self.error_log(flit, 6491, -1)

            # Plan non ring bridge moves
            # Handling delay flits
            if flit.is_delay:
                return self._handle_delay_flit(flit, link, current, next_node, row_start, row_end, col_start, col_end)
            # Handling regular flits
            else:
                return self._handle_regular_flit(flit, link, current, next_node, row_start, row_end, col_start, col_end)

    def _handle_delay_flit(self, flit: Flit, link, current, next_node, row_start, row_end, col_start, col_end):
        # 1. 非链路末端
        if flit.current_seat_index < len(link) - 1:
            link[flit.current_seat_index] = None
            flit.current_seat_index += 1
            return
        # self.error_log(flit, 144, -1)
        # 2. 到达链路末端，此时flit在next_node节点
        target_eject_node_id = flit.path[flit.path_index + 1] if flit.path_index + 1 < len(flit.path) else flit.path[flit.path_index]  # delay情况下path_index不更新
        # A. 处理横边界情况
        if current == next_node:
            # A1. 左边界情况
            if next_node == row_start:
                if next_node == flit.current_position:
                    # Flit已经绕横向环一圈
                    flit.circuits_completed_h += 1
                    link_station = self.ring_bridge["TR"].get((next_node, target_eject_node_id))
                    can_use_T1 = self._entry_available("TR", (next_node, target_eject_node_id), "T1")
                    can_use_T2 = self._entry_available("TR", (next_node, target_eject_node_id), "T2")
                    # TR方向尝试下环
                    if len(link_station) < self.config.RB_IN_FIFO_DEPTH and (
                        (flit.ETag_priority == "T1" and can_use_T1)
                        or (flit.ETag_priority == "T2" and can_use_T2)
                        or (flit.ETag_priority == "T1" and not can_use_T1 and can_use_T2)  # T1使用T2 entry
                        or (flit.ETag_priority == "T0" and can_use_T1)  # T0使用T1 entry
                        or (flit.ETag_priority == "T0" and not can_use_T1 and can_use_T2)  # T0使用T2 entry
                    ):
                        flit.is_delay = False
                        flit.current_link = (next_node, target_eject_node_id)
                        link[flit.current_seat_index] = None
                        flit.current_seat_index = 1
                        if flit.ETag_priority == "T0":
                            # 若升级到T0则需要从T0队列中移除flit
                            self.T0_Etag_Order_FIFO.remove((next_node, flit))
                            if can_use_T1:
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T1", flit)
                            else:
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T2", flit)
                        elif flit.ETag_priority == "T2":
                            self._occupy_entry("TR", (next_node, target_eject_node_id), "T2", flit)
                        elif flit.ETag_priority == "T1":
                            if can_use_T1:
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T1", flit)
                            else:
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T2", flit)
                    else:
                        # 无法下环,TR方向的flit不能升级T0
                        link[flit.current_seat_index] = None
                        next_pos = next_node + 1
                        flit.current_link = (next_node, next_pos)
                        flit.current_seat_index = 0
                        if self.ETag_BOTHSIDE_UPGRADE and flit.ETag_priority == "T2":
                            flit.ETag_priority = "T1"
                else:
                    # Flit未绕回下环点，向右绕环
                    link[flit.current_seat_index] = None
                    next_pos = next_node + 1
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0

            # A2. 右边界情况:
            elif next_node == row_end:
                if next_node == flit.current_position:
                    flit.circuits_completed_h += 1
                    link_station = self.ring_bridge["TL"].get((next_node, target_eject_node_id))
                    can_use_T0 = self._entry_available("TL", (next_node, target_eject_node_id), "T0")
                    can_use_T1 = self._entry_available("TL", (next_node, target_eject_node_id), "T1")
                    can_use_T2 = self._entry_available("TL", (next_node, target_eject_node_id), "T2")
                    # 尝试TL下环，非T0情况
                    if flit.ETag_priority in ["T1", "T2"]:
                        if len(link_station) < self.config.RB_IN_FIFO_DEPTH and (
                            (flit.ETag_priority == "T1" and can_use_T1)
                            or (flit.ETag_priority == "T2" and can_use_T2)
                            or (flit.ETag_priority == "T1" and not can_use_T1 and can_use_T2)  # T1使用T2 entry
                        ):
                            flit.is_delay = False
                            flit.current_link = (next_node, target_eject_node_id)
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0
                            if flit.ETag_priority == "T2":
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T2", flit)
                            elif flit.ETag_priority == "T1":
                                if can_use_T1:
                                    # T1使用T1 entry
                                    self._occupy_entry("TL", (next_node, target_eject_node_id), "T1", flit)
                                else:
                                    # T1使用T2 entry
                                    self._occupy_entry("TL", (next_node, target_eject_node_id), "T2", flit)

                        else:
                            # 无法下环,升级ETag并记录
                            if flit.ETag_priority == "T2":
                                flit.ETag_priority = "T1"
                            elif flit.ETag_priority == "T1":
                                flit.ETag_priority = "T0"
                                self.T0_Etag_Order_FIFO.append((next_node, flit))
                            link[flit.current_seat_index] = None
                            next_pos = next_node - 1
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                    # 尝试TL以T0下环
                    elif flit.ETag_priority == "T0":
                        if len(link_station) < self.config.RB_IN_FIFO_DEPTH:
                            # 按优先级尝试: T0专用 > T1 > T2
                            if self.T0_Etag_Order_FIFO[0] == (next_node, flit) and can_use_T0:
                                # 使用T0专用entry
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T0", flit)
                                flit.is_delay = False
                                flit.current_link = (next_node, target_eject_node_id)
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                                self.T0_Etag_Order_FIFO.popleft()
                            elif can_use_T1:
                                # 使用T1 entry
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T1", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.current_link = (next_node, target_eject_node_id)
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            elif can_use_T2:
                                # 使用T2 entry
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T2", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.current_link = (next_node, target_eject_node_id)
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            else:
                                # 无法下环，继续绕环
                                link[flit.current_seat_index] = None
                                next_pos = next_node - 1
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                        else:
                            # link_station满，无法下环
                            link[flit.current_seat_index] = None
                            next_pos = next_node - 1
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                # 未到下环节点，继续向左绕环
                else:
                    link[flit.current_seat_index] = None
                    next_pos = next_node - 1
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0
            # A3. 上边界情况:
            elif next_node == col_start:
                if next_node == flit.current_position:
                    flit.circuits_completed_v += 1
                    link_eject = self.eject_queues["TD"][next_node]
                    can_use_T1 = self._entry_available("TD", next_node, "T1")
                    can_use_T2 = self._entry_available("TD", next_node, "T2")

                    if len(link_eject) < self.config.EQ_IN_FIFO_DEPTH and (
                        (flit.ETag_priority == "T1" and can_use_T1) or (flit.ETag_priority == "T2" and can_use_T2) or (flit.ETag_priority == "T0" and (can_use_T1 or can_use_T2))
                    ):
                        flit.is_delay = False
                        flit.is_arrive = True
                        link[flit.current_seat_index] = None
                        flit.current_seat_index = 0

                        if flit.ETag_priority == "T0":
                            self.T0_Etag_Order_FIFO.remove((next_node, flit))
                            if can_use_T1:
                                # T0使用T1 entry
                                self._occupy_entry("TD", next_node, "T1", flit)
                            else:
                                # T0使用T2 entry
                                self._occupy_entry("TD", next_node, "T2", flit)
                        elif flit.ETag_priority == "T2":
                            self._occupy_entry("TD", next_node, "T2", flit)
                        elif flit.ETag_priority == "T1":
                            if can_use_T1:
                                # T1使用T1 entry
                                self._occupy_entry("TD", next_node, "T1", flit)
                            else:
                                # T1使用T2 entry
                                self._occupy_entry("TD", next_node, "T2", flit)
                    else:
                        # 无法下环,TD方向的flit不能升级T0
                        if self.ETag_BOTHSIDE_UPGRADE and flit.ETag_priority == "T2":
                            flit.ETag_priority = "T1"
                        link[flit.current_seat_index] = None
                        next_pos = next_node + self.config.NUM_COL * 2
                        flit.current_link = (next_node, next_pos)
                        flit.current_seat_index = 0
                else:
                    link[flit.current_seat_index] = None
                    next_pos = next_node + self.config.NUM_COL * 2
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0
            # A4. 下边界情况:
            elif next_node == col_end:
                if next_node == flit.current_position:
                    flit.circuits_completed_v += 1
                    link_eject = self.eject_queues["TU"][next_node]
                    can_use_T0 = self._entry_available("TU", next_node, "T0")
                    can_use_T1 = self._entry_available("TU", next_node, "T1")
                    can_use_T2 = self._entry_available("TU", next_node, "T2")

                    if flit.ETag_priority in ["T1", "T2"]:
                        if len(link_eject) < self.config.EQ_IN_FIFO_DEPTH and (
                            (flit.ETag_priority == "T1" and can_use_T1)
                            or (flit.ETag_priority == "T2" and can_use_T2)
                            or (flit.ETag_priority == "T1" and not can_use_T1 and can_use_T2)  # T1使用T2 entry
                        ):
                            flit.is_delay = False
                            flit.is_arrive = True
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0

                            if flit.ETag_priority == "T2":
                                self._occupy_entry("TU", next_node, "T2", flit)
                            elif flit.ETag_priority == "T1":
                                if can_use_T1:
                                    # T1使用T1 entry
                                    self._occupy_entry("TU", next_node, "T1", flit)
                                else:
                                    # T1使用T2 entry
                                    self._occupy_entry("TU", next_node, "T2", flit)
                        else:
                            # 无法下环,升级ETag并记录
                            if flit.ETag_priority == "T2":
                                flit.ETag_priority = "T1"
                            elif flit.ETag_priority == "T1":
                                self.T0_Etag_Order_FIFO.append((next_node, flit))
                                flit.ETag_priority = "T0"
                            link[flit.current_seat_index] = None
                            next_pos = next_node - self.config.NUM_COL * 2
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0

                    elif flit.ETag_priority == "T0":
                        if len(link_eject) < self.config.EQ_IN_FIFO_DEPTH:
                            # 按优先级尝试: T0专用 > T1 > T2
                            if self.T0_Etag_Order_FIFO[0] == (next_node, flit) and can_use_T0:
                                # 使用T0专用entry
                                self._occupy_entry("TU", next_node, "T0", flit)
                                flit.is_delay = False
                                flit.is_arrive = True
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                                self.T0_Etag_Order_FIFO.popleft()
                            elif can_use_T1:
                                # 使用T1 entry
                                self._occupy_entry("TU", next_node, "T1", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.is_arrive = True
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            elif can_use_T2:
                                # 使用T2 entry
                                self._occupy_entry("TU", next_node, "T2", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.is_arrive = True
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            else:
                                # 无法下环，继续绕环
                                link[flit.current_seat_index] = None
                                next_pos = next_node - self.config.NUM_COL * 2
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                        else:
                            # link_eject满，无法下环
                            link[flit.current_seat_index] = None
                            next_pos = next_node - self.config.NUM_COL * 2
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                else:
                    link[flit.current_seat_index] = None
                    next_pos = next_node - self.config.NUM_COL * 2
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0
        # B. 非边界横向环情况
        elif abs(current - next_node) == 1:
            if next_node == flit.current_position:
                flit.circuits_completed_h += 1
                if current - next_node == 1:
                    link_station = self.ring_bridge["TL"].get((next_node, target_eject_node_id))
                    can_use_T0 = self._entry_available("TL", (next_node, target_eject_node_id), "T0")
                    can_use_T1 = self._entry_available("TL", (next_node, target_eject_node_id), "T1")
                    can_use_T2 = self._entry_available("TL", (next_node, target_eject_node_id), "T2")

                    if flit.ETag_priority in ["T1", "T2"]:
                        if len(link_station) < self.config.RB_IN_FIFO_DEPTH and (
                            (flit.ETag_priority == "T1" and can_use_T1)
                            or (flit.ETag_priority == "T2" and can_use_T2)
                            or (flit.ETag_priority == "T1" and not can_use_T1 and can_use_T2)  # T1使用T2 entry
                        ):
                            flit.is_delay = False
                            flit.current_link = (next_node, target_eject_node_id)
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0

                            if flit.ETag_priority == "T2":
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T2", flit)
                            elif flit.ETag_priority == "T1":
                                if can_use_T1:
                                    # T1使用T1 entry
                                    self._occupy_entry("TL", (next_node, target_eject_node_id), "T1", flit)
                                else:
                                    # T1使用T2 entry
                                    self._occupy_entry("TL", (next_node, target_eject_node_id), "T2", flit)

                        else:
                            if flit.ETag_priority == "T2":
                                flit.ETag_priority = "T1"
                            elif flit.ETag_priority == "T1":
                                self.T0_Etag_Order_FIFO.append((next_node, flit))
                                flit.ETag_priority = "T0"
                            link[flit.current_seat_index] = None
                            next_pos = max(next_node - 1, row_start)
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0

                    elif flit.ETag_priority == "T0":
                        if len(link_station) < self.config.RB_IN_FIFO_DEPTH:
                            # 按优先级尝试: T0专用 > T1 > T2
                            if self.T0_Etag_Order_FIFO[0] == (next_node, flit) and can_use_T0:
                                # 使用T0专用entry
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T0", flit)
                                flit.is_delay = False
                                flit.current_link = (next_node, target_eject_node_id)
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                                self.T0_Etag_Order_FIFO.popleft()
                            elif can_use_T1:
                                # 使用T1 entry
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T1", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.current_link = (next_node, target_eject_node_id)
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            elif can_use_T2:
                                # 使用T2 entry
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T2", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.current_link = (next_node, target_eject_node_id)
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            else:
                                # 无法下环，继续绕环
                                link[flit.current_seat_index] = None
                                next_pos = max(next_node - 1, row_start)
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                        else:
                            # link_station满，无法下环
                            link[flit.current_seat_index] = None
                            next_pos = max(next_node - 1, row_start)
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                else:
                    # 横向环TR尝试下环
                    link_station = self.ring_bridge["TR"].get((next_node, target_eject_node_id))
                    can_use_T1 = self._entry_available("TR", (next_node, target_eject_node_id), "T1")
                    can_use_T2 = self._entry_available("TR", (next_node, target_eject_node_id), "T2")

                    if len(link_station) < self.config.RB_IN_FIFO_DEPTH and (
                        (flit.ETag_priority == "T1" and can_use_T1) or (flit.ETag_priority == "T2" and can_use_T2) or (flit.ETag_priority == "T0" and (can_use_T1 or can_use_T2))
                    ):
                        flit.is_delay = False
                        flit.current_link = (next_node, target_eject_node_id)
                        link[flit.current_seat_index] = None
                        flit.current_seat_index = 1

                        # 根据使用的entry类型更新计数器
                        if flit.ETag_priority == "T0":
                            self.T0_Etag_Order_FIFO.remove((next_node, flit))
                            if can_use_T1:
                                # T0使用T1 entry
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T1", flit)
                            else:
                                # T0使用T2 entry
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T2", flit)
                        elif flit.ETag_priority == "T2":
                            self._occupy_entry("TR", (next_node, target_eject_node_id), "T2", flit)
                        elif flit.ETag_priority == "T1":
                            if can_use_T1:
                                # T1使用T1 entry
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T1", flit)
                            else:
                                # T1使用T2 entry
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T2", flit)
                    else:
                        link[flit.current_seat_index] = None
                        next_pos = min(next_node + 1, row_end)
                        flit.current_link = (next_node, next_pos)
                        flit.current_seat_index = 0
                        if self.ETag_BOTHSIDE_UPGRADE and flit.ETag_priority == "T2":
                            flit.ETag_priority = "T1"
            else:
                link[flit.current_seat_index] = None
                if current - next_node == 1:
                    next_pos = max(next_node - 1, row_start)
                else:
                    next_pos = min(next_node + 1, row_end)
                flit.current_link = (next_node, next_pos)
                flit.current_seat_index = 0
        # C. 非边界纵向环情况
        else:
            if next_node == flit.current_position:
                flit.circuits_completed_v += 1
                if current - next_node == self.config.NUM_COL * 2:
                    link_eject = self.eject_queues["TU"][next_node]
                    can_use_T0 = self._entry_available("TU", next_node, "T0")
                    can_use_T1 = self._entry_available("TU", next_node, "T1")
                    can_use_T2 = self._entry_available("TU", next_node, "T2")
                    if flit.ETag_priority in ["T1", "T2"]:
                        # up move
                        if len(link_eject) < self.config.EQ_IN_FIFO_DEPTH and (
                            (flit.ETag_priority == "T1" and can_use_T1)
                            or (flit.ETag_priority == "T2" and can_use_T2)
                            or (flit.ETag_priority == "T1" and not can_use_T1 and can_use_T2)  # T1使用T2 entry
                        ):
                            flit.is_delay = False
                            flit.is_arrive = True
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0
                            if flit.ETag_priority == "T2":
                                self._occupy_entry("TU", next_node, "T2", flit)
                            elif flit.ETag_priority == "T1":
                                if can_use_T1:
                                    # T1使用T1 entry
                                    self._occupy_entry("TU", next_node, "T1", flit)
                                else:
                                    # T1使用T2 entry
                                    self._occupy_entry("TU", next_node, "T2", flit)
                        else:
                            if flit.ETag_priority == "T2":
                                flit.ETag_priority = "T1"
                            elif flit.ETag_priority == "T1":
                                self.T0_Etag_Order_FIFO.append((next_node, flit))
                                flit.ETag_priority = "T0"
                            link[flit.current_seat_index] = None
                            next_pos = next_node - self.config.NUM_COL * 2 if next_node - self.config.NUM_COL * 2 >= col_start else col_start
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                    elif flit.ETag_priority == "T0":
                        if len(link_eject) < self.config.EQ_IN_FIFO_DEPTH:
                            # 按优先级尝试: T0专用 > T1 > T2
                            if self.T0_Etag_Order_FIFO[0] == (next_node, flit) and can_use_T0:
                                # 使用T0专用entry
                                self._occupy_entry("TU", next_node, "T0", flit)
                                flit.is_delay = False
                                flit.is_arrive = True
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                                self.T0_Etag_Order_FIFO.popleft()
                            elif can_use_T1:
                                # 使用T1 entry
                                self._occupy_entry("TU", next_node, "T1", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.is_arrive = True
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            elif can_use_T2:
                                # 使用T2 entry
                                self._occupy_entry("TU", next_node, "T2", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.is_arrive = True
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            else:
                                # 无法下环，继续绕环
                                link[flit.current_seat_index] = None
                                next_pos = max(next_node - self.config.NUM_COL * 2, col_start)
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                        else:
                            # link_eject满，无法下环
                            link[flit.current_seat_index] = None
                            next_pos = max(next_node - self.config.NUM_COL * 2, col_start)
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                else:
                    # down move
                    link_eject = self.eject_queues["TD"][next_node]
                    can_use_T1 = self._entry_available("TD", next_node, "T1")
                    can_use_T2 = self._entry_available("TD", next_node, "T2")

                    if len(link_eject) < self.config.EQ_IN_FIFO_DEPTH and (
                        (flit.ETag_priority == "T1" and can_use_T1) or (flit.ETag_priority == "T2" and can_use_T2) or (flit.ETag_priority == "T0" and (can_use_T1 or can_use_T2))
                    ):
                        flit.is_delay = False
                        flit.is_arrive = True
                        link[flit.current_seat_index] = None
                        flit.current_seat_index = 0
                        # 根据使用的entry类型更新计数器
                        if flit.ETag_priority == "T0":
                            self.T0_Etag_Order_FIFO.remove((next_node, flit))
                            if can_use_T1:
                                # T0使用T1 entry
                                self._occupy_entry("TD", next_node, "T1", flit)
                            else:
                                # T0使用T2 entry
                                self._occupy_entry("TD", next_node, "T2", flit)
                        elif flit.ETag_priority == "T2":
                            self.EQ_UE_Counters["TD"][next_node]["T2"] += 1
                        elif flit.ETag_priority == "T1":
                            if can_use_T1:
                                # T1使用T1 entry
                                self._occupy_entry("TD", next_node, "T1", flit)
                            else:
                                # T1使用T2 entry
                                self._occupy_entry("TD", next_node, "T2", flit)
                    else:
                        link[flit.current_seat_index] = None
                        next_pos = min(next_node + self.config.NUM_COL * 2, col_end)
                        flit.current_link = (next_node, next_pos)
                        flit.current_seat_index = 0
            else:
                link[flit.current_seat_index] = None
                if current - next_node == self.config.NUM_COL * 2:
                    next_pos = max(next_node - self.config.NUM_COL * 2, col_start)
                else:
                    next_pos = min(next_node + self.config.NUM_COL * 2, col_end)
                flit.current_link = (next_node, next_pos)
                flit.current_seat_index = 0
        return

    def _handle_regular_flit(self, flit: Flit, link, current, next_node, row_start, row_end, col_start, col_end):
        # 1. 非链路末端: 在当前链路上前进一步
        if flit.current_seat_index < len(link) - 1:
            link[flit.current_seat_index] = None
            flit.current_seat_index += 1
            return
        # self.error_log(flit, 7, -1)

        # 2. 已经到达
        flit.current_position = next_node
        flit.path_index += 1
        # 检查是否还有后续路径
        if flit.path_index + 1 < len(flit.path):
            new_current, new_next_node = next_node, flit.path[flit.path_index + 1]

            # A. 处理横边界情况（非自环）
            if current == next_node and new_next_node != new_current:
                # 这里可以添加特殊处理逻辑
                pass

            # 2a. 正常绕环
            if new_current - new_next_node != self.config.NUM_COL:
                flit.current_link = (new_current, new_next_node)
                link[flit.current_seat_index] = None
                flit.current_seat_index = 0

            # 2b. 横向环向左进入Ring Bridge
            elif current - next_node == 1:
                station = self.ring_bridge["TL"].get((new_current, new_next_node))
                # TL有空位
                if self.config.RB_IN_FIFO_DEPTH > len(station) and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T2"] < self.config.TL_Etag_T2_UE_MAX:
                    flit.current_link = (new_current, new_next_node)
                    link[flit.current_seat_index] = None
                    flit.current_seat_index = 0
                    # 更新计数器
                    self._occupy_entry("TL", (new_current, new_next_node), "T2", flit)
                else:
                    # TL无空位: 预留到右侧等待队列，设置延迟标志，ETag升级
                    flit.ETag_priority = "T1"
                    next_pos = next_node - 1 if next_node - 1 >= row_start else row_start
                    flit.is_delay = True
                    link[flit.current_seat_index] = None
                    flit.current_link = (new_current, next_pos)
                    flit.current_seat_index = 0

            # 2c. 横向环向右进入Ring Bridge
            elif current - next_node == -1:
                station = self.ring_bridge["TR"].get((new_current, new_next_node))
                if self.config.RB_IN_FIFO_DEPTH > len(station) and self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] < self.config.TR_Etag_T2_UE_MAX:
                    flit.current_link = (new_current, new_next_node)
                    link[flit.current_seat_index] = None
                    flit.current_seat_index = 1
                    self._occupy_entry("TR", (new_current, new_next_node), "T2", flit)
                else:
                    # TR无空位: 设置延迟标志，如果双边ETag升级，则升级ETag。
                    if self.ETag_BOTHSIDE_UPGRADE:
                        flit.ETag_priority = "T1"
                    next_pos = next_node + 1 if next_node + 1 <= row_end else row_end
                    flit.is_delay = True
                    link[flit.current_seat_index] = None
                    flit.current_link = (new_current, next_pos)
                    flit.current_seat_index = 0
        else:
            # 3. 已经到达目的地，执行eject逻辑
            if current - next_node == self.config.NUM_COL * 2:  # 纵向环向上TU
                eject_queue = self.eject_queues["TU"][next_node]
                if self.config.EQ_IN_FIFO_DEPTH > len(eject_queue) and self.EQ_UE_Counters["TU"][next_node]["T2"] < self.config.TU_Etag_T2_UE_MAX:
                    link[flit.current_seat_index] = None
                    flit.current_seat_index = 0
                    flit.is_arrive = True
                    self._occupy_entry("TU", next_node, "T2", flit)
                else:
                    flit.ETag_priority = "T1"
                    next_pos = next_node - self.config.NUM_COL * 2 if next_node - self.config.NUM_COL * 2 >= col_start else col_start
                    flit.is_delay = True
                    link[flit.current_seat_index] = None
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0
            elif current - next_node == -self.config.NUM_COL * 2:  # 纵向环向下TD
                eject_queue = self.eject_queues["TD"][next_node]
                if self.config.EQ_IN_FIFO_DEPTH > len(eject_queue) and self.EQ_UE_Counters["TD"][next_node]["T2"] < self.config.TD_Etag_T2_UE_MAX:
                    link[flit.current_seat_index] = None
                    flit.current_seat_index = 0
                    flit.is_arrive = True
                    self._occupy_entry("TD", next_node, "T2", flit)
                else:
                    if self.ETag_BOTHSIDE_UPGRADE:
                        flit.ETag_priority = "T1"
                    # TODO: next_pos 名称不好
                    next_pos = next_node + self.config.NUM_COL * 2 if next_node + self.config.NUM_COL * 2 <= col_end else col_end
                    flit.is_delay = True
                    link[flit.current_seat_index] = None
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0

    def execute_moves(self, flit: Flit, cycle):
        if not flit.is_arrive:
            current, next_node = flit.current_link
            if current - next_node != self.config.NUM_COL:
                link = self.links.get(flit.current_link)
                self.set_link_slice(flit.current_link, flit.current_seat_index, flit, cycle)
                # link[flit.current_seat_index] = flit
                if (flit.current_seat_index == len(link) - 1 and len(link) > 2) or (flit.current_seat_index == 1 and len(link) == 2):
                    self.links_flow_stat[flit.req_type][flit.current_link] += 1
            else:
                # 将 flit 放入 ring_bridge 的相应方向
                if not flit.is_on_station:
                    # 使用字典映射 seat_index 到 ring_bridge 的方向和深度限制
                    direction, max_depth = self.ring_bridge_map.get(flit.current_seat_index, (None, None))
                    if direction is None:
                        return False
                    if direction in self.ring_bridge.keys() and len(self.ring_bridge[direction][flit.current_link]) < max_depth and self.ring_bridge_pre[direction][flit.current_link] is None:
                        # flit.flit_position = f"RB_{direction}"
                        # self.ring_bridge[direction][flit.current_link].append(flit)
                        self.ring_bridge_pre[direction][flit.current_link] = flit
                        flit.is_on_station = True
            return False
        else:
            if flit.current_link is not None:
                current, next_node = flit.current_link
            flit.arrival_network_cycle = cycle

            if flit.source - flit.destination == self.config.NUM_COL:
                flit.flit_position = f"IQ_EQ"
                flit.is_arrived = True

                return True
            elif current - next_node == self.config.NUM_COL * 2 or (current == next_node and current not in range(0, self.config.NUM_COL)):
                direction = "TU"
                queue = self.eject_queues["TU"]
                queue_pre = self.eject_queues_in_pre["TU"]
            else:
                direction = "TD"
                queue = self.eject_queues["TD"]
                queue_pre = self.eject_queues_in_pre["TD"]

            # flit.flit_position = f"EQ_{direction}"
            # queue[next_node].append(flit)
            if queue_pre[next_node]:
                return False
            else:
                queue_pre[next_node] = flit
                return True
