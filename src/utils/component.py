import numpy as np
from collections import deque, defaultdict
from config.config import CrossRingConfig


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
        self.current_position = None
        self.station_position = None
        self.departure_cycle = None
        self.req_departure_cycle = None
        self.departure_network_cycle = None
        self.departure_inject_cycle = None
        self.arrival_cycle = None
        self.arrival_network_cycle = None
        self.arrival_eject_cycle = None
        self.entry_db_cycle = None
        self.leave_db_cycle = None
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
        # Latency record
        self.cmd_entry_cmd_table_cycle = None
        self.req_entry_network_cycle = None
        self.sn_receive_req_cycle = None
        self.sn_data_generated_cycle = None
        self.data_entry_network_cycle = None
        self.rn_data_collection_complete_cycle = None
        self.sn_rsp_generate_cycle = None
        self.rsp_entry_network_cycle = None
        self.rn_receive_rsp_cycle = None
        self.rn_data_generated_cycle = None
        self.sn_data_collection_complete_cycle = None
        self.total_latency = None
        self.cmd_latency = None
        self.rsp_latency = None
        self.dat_latency = None

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
        arrive_status = "A" if self.is_arrive else ""
        eject_status = "E" if self.is_ejected else ""

        return (
            f"{self.packet_id}.{self.flit_id} {self.source}.{self.source_type[0]}{self.source_type[-1]}->{self.destination}.{self.destination_type[0]}{self.destination_type[-1]}: "
            f"{self.current_link} -> {self.current_seat_index}, "
            f"{self.current_position}, "
            f"{req_attr}, {self.flit_type}, {type_display}, "
            f"{arrive_status}{eject_status}, "
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

    def initialize_rn(self):
        """Initialize RN structures."""
        for ip_type in self.rn_rdb.keys():
            for ip_pos in getattr(self.config, f"{ip_type[:-2]}_send_positions"):
                self.rn_rdb[ip_type][ip_pos] = defaultdict(list)
                self.rn_wdb[ip_type][ip_pos] = defaultdict(list)
                self.setup_rn_trackers(ip_type, ip_pos)

    def setup_rn_trackers(self, ip_type, ip_pos):
        """Setup read and write trackers for RN."""
        self.rn_rdb_recv[ip_type][ip_pos] = []
        self.rn_rdb_count[ip_type][ip_pos] = self.config.rn_rdb_size
        self.rn_rdb_reserve[ip_type][ip_pos] = 0
        self.rn_wdb_count[ip_type][ip_pos] = self.config.rn_wdb_size
        self.rn_wdb_send[ip_type][ip_pos] = []
        self.rn_wdb_reserve[ip_type][ip_pos] = 0

        for req_type in ["read", "write"]:
            self.rn_tracker[req_type][ip_type][ip_pos] = []
            self.rn_tracker_wait[req_type][ip_type][ip_pos] = []
            self.rn_tracker_count[req_type][ip_type][ip_pos] = self.config.rn_read_tracker_ostd if req_type == "read" else self.config.rn_write_tracker_ostd
            self.rn_tracker_pointer[req_type][ip_type][ip_pos] = -1

    def initialize_sn(self):
        """Initialize SN structures."""
        self.sn_tracker_release_time = defaultdict(list)
        for ip_pos in set(self.config.ddr_send_positions + self.config.l2m_send_positions):
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
            self.sn_wdb_count[key][ip_pos] = self.config.sn_ddr_wdb_size
            self.sn_tracker_count[key]["ro"][ip_pos] = self.config.sn_ddr_read_tracker_ostd
            self.sn_tracker_count[key]["share"][ip_pos] = self.config.sn_ddr_write_tracker_ostd
        elif key.startswith("l2m"):
            self.sn_wdb_count[key][ip_pos] = self.config.sn_l2m_wdb_size
            self.sn_tracker_count[key]["ro"][ip_pos] = self.config.sn_l2m_read_tracker_ostd
            self.sn_tracker_count[key]["share"][ip_pos] = self.config.sn_l2m_write_tracker_ostd
        # else:
        # if ip_pos in self.config.ddr_real_positions:
        #     self.sn_wdb_count[key][ip_pos] = self.config.sn_ddr_wdb_size
        #     self.sn_tracker_count[key]["ro"][ip_pos] = self.config.sn_ddr_read_tracker_ostd
        #     self.sn_tracker_count[key]["share"][ip_pos] = self.config.sn_ddr_write_tracker_ostd
        # elif ip_pos in self.config.l2m_real_positions:
        #     self.sn_wdb_count[key][ip_pos] = self.config.sn_l2m_wdb_size
        #     self.sn_tracker_count[key]["ro"][ip_pos] = self.config.sn_l2m_read_tracker_ostd
        #     self.sn_tracker_count[key]["share"][ip_pos] = self.config.sn_l2m_write_tracker_ostd


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
        self.EQ_channel_buffer_pre = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.eject_queues = {"TU": {}, "TD": {}}
        self.arrive_node_pre = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.IQ_channel_buffer = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.EQ_channel_buffer = self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))
        self.links = {}
        self.links_flow_stat = {"read": {}, "write": {}}
        self.links_tag = {}
        self.remain_tag = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}
        self.ring_bridge = {"TL": {}, "TR": {}, "ft": {}, "TU": {}, "TD": {}, "EQ": {}}
        # self.inject_queue_rr = {"TL": {0: {}, 1: {}}, "TR": {0: {}, 1: {}}, "TU": {0: {}, 1: {}}, "EQ": {0: {}, 1: {}}}
        # self.inject_rr = {"TL": {}, "TR": {}, "TU": {}, "EQ": {}}
        # self.round_robin = {**{"TU": {}, "TD": {}, "RB": {}}, **self.config._make_channels(("sdma", "gdma", "ddr", "l2m"))}
        self.round_robin = {"IQ": {}, "RB": {}, "EQ": {}}

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
            -3: ("ft", self.config.ft_len),
        }

        self.token_bucket = defaultdict(dict)
        self.flit_size_bytes = 128
        for ch_name in self.IQ_channel_buffer.keys():
            for ip_pos in set(self.config.ddr_send_positions + self.config.l2m_send_positions):

                if ch_name.startswith("ddr"):
                    self.token_bucket[ip_pos][ch_name] = TokenBucket(
                        rate=self.config.ddr_bandwidth_limit / self.config.network_frequency / self.flit_size_bytes,
                        bucket_size=self.config.ddr_bandwidth_limit,
                    )
                    self.token_bucket[ip_pos - self.config.cols][ch_name] = TokenBucket(
                        rate=self.config.ddr_bandwidth_limit / self.config.network_frequency / self.flit_size_bytes,
                        bucket_size=self.config.ddr_bandwidth_limit,
                    )
                elif ch_name.startswith("l2m"):
                    self.token_bucket[ip_pos][ch_name] = TokenBucket(
                        rate=self.config.l2m_bandwidth_limit / self.config.network_frequency / self.flit_size_bytes,
                        bucket_size=self.config.l2m_bandwidth_limit,
                    )

        # ETag setup
        self.T0_Etag_Order_FIFO = deque()  # 用于轮询选择 T0 Flit 的 Order FIFO
        self.RB_UE_Counters = {"TL": {}, "TR": {}}
        self.EQ_UE_Counters = {"TU": {}, "TD": {}}
        self.Both_side_ETag_upgrade = False

        for ip_pos in set(config.ddr_send_positions + config.sdma_send_positions + config.l2m_send_positions + config.gdma_send_positions):
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
            for key in self.EQ_channel_buffer_pre:
                self.EQ_channel_buffer_pre[key][ip_pos - config.cols] = None
            for key in self.arrive_node_pre:
                self.arrive_node_pre[key][ip_pos - config.cols] = None
            self.eject_queues["TU"][ip_pos - config.cols] = deque(maxlen=config.EQ_IN_FIFO_DEPTH)
            self.eject_queues["TD"][ip_pos - config.cols] = deque(maxlen=config.EQ_IN_FIFO_DEPTH)
            self.EQ_UE_Counters["TU"][ip_pos - config.cols] = {"T2": 0, "T1": 0, "T0": 0}
            self.EQ_UE_Counters["TD"][ip_pos - config.cols] = {"T2": 0, "T1": 0}
            # for key in self.inject_queue_rr:
            #     self.inject_queue_rr[key][0][ip_pos] = deque([0, 1])
            #     self.inject_queue_rr[key][1][ip_pos] = deque([0, 1])
            # self.inject_rr["TR"][ip_pos] = deque([0, 1, 2])
            # self.inject_rr["TL"][ip_pos] = deque([0, 1, 2])
            # self.inject_rr["TU"][ip_pos] = deque([0, 1, 2])
            # self.inject_rr["EQ"][ip_pos] = deque([0, 1, 2])
            for key in self.round_robin.keys():
                if key == "IQ":
                    self.round_robin[key][ip_pos - config.cols] = deque()
                    for ch_name in self.IQ_channel_buffer.keys():
                        self.round_robin[key][ip_pos - config.cols].append(ch_name)
                elif key == "EQ":
                    self.round_robin[key][ip_pos - config.cols] = deque([0, 1, 2, 3])
                else:
                    self.round_robin[key][ip_pos - config.cols] = deque([0, 0, 0, 1, 1, 1, 2, 3, 4])

            self.inject_time[ip_pos] = []
            self.eject_time[ip_pos - config.cols] = []
            self.avg_inject_time[ip_pos] = 0
            self.avg_eject_time[ip_pos - config.cols] = 1

        for i in range(config.num_nodes):
            for j in range(config.num_nodes):
                if adjacency_matrix[i][j] == 1 and i - j != config.cols:
                    self.links[(i, j)] = [None] * config.seats_per_link
                    self.links_flow_stat["read"][(i, j)] = 0
                    self.links_flow_stat["write"][(i, j)] = 0
                    self.links_tag[(i, j)] = [None] * config.seats_per_link
            if i in range(0, config.cols):
                self.links[(i, i)] = [None] * 2
                self.links[(i + config.num_nodes - config.cols * 2, i + config.num_nodes - config.cols * 2)] = [None] * 2
                self.links_flow_stat["read"][(i, i)] = 0
                self.links_flow_stat["write"][(i, i)] = 0
                self.links_flow_stat["read"][(i + config.num_nodes - config.cols * 2, i + config.num_nodes - config.cols * 2)] = 0
                self.links_flow_stat["write"][(i + config.num_nodes - config.cols * 2, i + config.num_nodes - config.cols * 2)] = 0
                self.links_tag[(i, i)] = [None] * 2
                self.links_tag[(i + config.num_nodes - config.cols * 2, i + config.num_nodes - config.cols * 2)] = [None] * 2
            if i % config.cols == 0 and (i // config.cols) % 2 != 0:
                self.links[(i, i)] = [None] * 2
                self.links[(i + config.cols - 1, i + config.cols - 1)] = [None] * 2
                self.links_flow_stat["read"][(i, i)] = 0
                self.links_flow_stat["write"][(i, i)] = 0
                self.links_flow_stat["read"][(i + config.cols - 1, i + config.cols - 1)] = 0
                self.links_flow_stat["write"][(i + config.cols - 1, i + config.cols - 1)] = 0
                self.links_tag[(i, i)] = [None] * 2
                self.links_tag[(i + config.cols - 1, i + config.cols - 1)] = [None] * 2

        for row in range(1, config.rows, 2):
            for col in range(config.cols):
                pos = row * config.cols + col
                next_pos = pos - config.cols
                self.ring_bridge["TL"][(pos, next_pos)] = deque(maxlen=config.RB_IN_FIFO_DEPTH)
                self.ring_bridge["TR"][(pos, next_pos)] = deque(maxlen=config.RB_IN_FIFO_DEPTH)
                self.ring_bridge["ft"][(pos, next_pos)] = deque(maxlen=config.ft_len)
                self.ring_bridge["TU"][(pos, next_pos)] = deque(maxlen=config.RB_OUT_FIFO_DEPTH)
                self.ring_bridge["TD"][(pos, next_pos)] = deque(maxlen=config.RB_OUT_FIFO_DEPTH)
                self.ring_bridge["EQ"][(pos, next_pos)] = deque(maxlen=config.RB_OUT_FIFO_DEPTH)
                self.RB_UE_Counters["TL"][(pos, next_pos)] = {"T2": 0, "T1": 0, "T0": 0}
                self.RB_UE_Counters["TR"][(pos, next_pos)] = {"T2": 0, "T1": 0}
                # self.round_robin["TU"][next_pos] = deque([0, 1, 2])
                # self.round_robin["TD"][next_pos] = deque([0, 1, 2])
                # self.round_robin["RB"][next_pos] = deque([0, 1, 2])
                for direction in ["TL", "TR"]:
                    self.remain_tag[direction][pos] = config.ITag_Max_Num_H
                for direction in ["TU", "TD"]:
                    self.remain_tag[direction][next_pos] = config.ITag_Max_Num_V

        for ip_type in self.num_recv.keys():
            source_positions = getattr(config, f"{ip_type[:-2]}_send_positions")
            for source in source_positions:
                destination = source - config.cols
                self.num_send[ip_type][source] = 0
                self.num_recv[ip_type][destination] = 0
                self.per_send_throughput[ip_type][source] = 0
                self.per_recv_throughput[ip_type][destination] = 0

        for ip_type in self.IQ_channel_buffer.keys():
            for ip_index in getattr(config, f"{ip_type[:-2]}_send_positions"):
                ip_recv_index = ip_index - config.cols
                self.IQ_channel_buffer[ip_type][ip_index] = deque()
                self.EQ_channel_buffer[ip_type][ip_recv_index] = deque(maxlen=config.EQ_CH_FIFO_DEPTH)
        for ip_type in self.last_select.keys():
            for ip_index in getattr(config, f"{ip_type[:-2]}_send_positions"):
                self.last_select[ip_type][ip_index] = "write"
        for ip_type in self.throughput.keys():
            for ip_index in getattr(config, f"{ip_type[:-2]}_send_positions"):
                self.throughput[ip_type][ip_index] = [0, 0, 10000000, 0]

    def can_move_to_next(self, flit, current, next_node):
        # flit inject的时候判断是否可以将flit放到本地出口队列。
        if flit.source - flit.destination == self.config.cols:
            # return len(self.eject_queues["IQ"][flit.destination]) < self.config.EQ_IN_FIFO_DEPTH
            return True

        # 获取当前节点所在列的索引
        current_column_index = current % self.config.cols

        if current_column_index == 0:
            if next_node == current + 1:
                if self.links[(current, current)][-1] is not None:
                    if self.links_tag[(current, current)][-1] is None:
                        flit_l = self.links[(current, current)][-1]
                        if flit_l.current_link[1] == flit_l.current_position:
                            # flit_exist_right = any(flit_r.id == flit_l.id for flit_r in self.station_reservations["TR"][(flit_l.current_link[1], flit_l.path[flit_l.path_index])])
                            link_station = self.ring_bridge["TR"].get((flit_l.current_link[1], flit_l.path[flit_l.path_index]))
                            new_current = flit_l.current_link[1]
                            new_next_node = flit_l.path[flit_l.path_index]
                            # if len(link_station) < self.config.RB_IN_FIFO_DEPTH and flit_exist_right:
                            # if len(link_station) < self.config.RB_IN_FIFO_DEPTH:
                            if self.config.RB_IN_FIFO_DEPTH > len(link_station) and (
                                (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T1"] < self.config.RB_IN_FIFO_DEPTH and flit_l.ETag_priority in ["T0", "T1"])
                                or (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] < self.config.TR_Etag_T2_UE_MAX and flit_l.ETag_priority == "T2")
                            ):
                                return True
                        if flit.wait_cycle_h > self.config.ITag_Trigger_Th_H and not flit.itag_h:
                            if self.remain_tag["TR"][current] > 0:
                                self.remain_tag["TR"][current] -= 1
                                self.links_tag[(current, current)][-1] = [current, "TR"]
                                flit.itag_h = True
                    else:
                        flit_l = self.links[(current, current)][-1]
                        if flit_l.current_link[1] == flit_l.current_position:
                            # flit_exist_right = any(flit_r.id == flit_l.id for flit_r in self.station_reservations["TR"][(flit_l.current_link[1], flit_l.path[flit_l.path_index])])
                            link_station = self.ring_bridge["TR"].get((flit_l.current_link[1], flit_l.path[flit_l.path_index]))
                            new_current = flit_l.current_link[1]
                            new_next_node = flit_l.path[flit_l.path_index]
                            # if len(link_station) < self.config.RB_IN_FIFO_DEPTH and flit_exist_right:
                            # if len(link_station) < self.config.RB_IN_FIFO_DEPTH:
                            if self.config.RB_IN_FIFO_DEPTH > len(link_station) and (
                                (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T1"] < self.config.RB_IN_FIFO_DEPTH and flit_l.ETag_priority in ["T0", "T1"])
                                or (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] < self.config.TR_Etag_T2_UE_MAX and flit_l.ETag_priority == "T2")
                            ):
                                if self.links_tag[(current, current)][-1] == [current, "TR"]:
                                    self.links_tag[(current, current)][-1] = None
                                    self.remain_tag["TR"][current] += 1
                                    return True
                else:
                    if self.links_tag[(current, current)][-1] is None:
                        return True
                    else:
                        if self.links_tag[(current, current)][-1] == [current, "TR"]:
                            self.links_tag[(current, current)][-1] = None
                            self.remain_tag["TR"][current] += 1
                            return True
                return False
        elif current_column_index == self.config.cols - 1:
            # 处理右边界向左移动的逻辑
            if next_node == current - 1:
                if self.links[(current, current)][-1] is not None:
                    if self.links_tag[(current, current)][-1] is None:
                        flit_l = self.links[(current, current)][-1]
                        if flit_l.current_link[1] == flit_l.current_position:
                            # flit_exist_left = any(flit_r.id == flit_l.id for flit_r in self.station_reservations["TL"][(flit_l.current_link[1], flit_l.path[flit_l.path_index])])
                            link_station = self.ring_bridge["TL"].get((flit_l.current_link[1], flit_l.path[flit_l.path_index]))
                            new_current = flit_l.current_link[1]
                            new_next_node = flit_l.path[flit_l.path_index]
                            # if len(link_station) < self.config.RB_IN_FIFO_DEPTH and flit_exist_left:
                            if self.config.RB_IN_FIFO_DEPTH > len(link_station) and (
                                (flit_l.ETag_priority == "T2" and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T2"] < self.config.TL_Etag_T2_UE_MAX)
                                or (flit_l.ETag_priority == "T1" and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T1"] < self.config.TL_Etag_T2_UE_MAX)
                                or (
                                    flit_l.ETag_priority == "T0"
                                    and self.T0_Etag_Order_FIFO[0] == (new_current, flit_l)
                                    and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T0"] < self.config.RB_IN_FIFO_DEPTH
                                )
                            ):
                                return True
                        if flit.wait_cycle_h > self.config.ITag_Trigger_Th_H and not flit.itag_h:
                            if self.remain_tag["TL"][current] > 0:
                                self.remain_tag["TL"][current] -= 1
                                self.links_tag[(current, current)][-1] = [current, "TL"]
                                flit.itag_h = True
                    else:
                        flit_l = self.links[(current, current)][-1]
                        if flit_l.current_link[1] == flit_l.current_position:
                            # flit_exist_left = any(flit_r.id == flit_l.id for flit_r in self.station_reservations["TL"][(flit_l.current_link[1], flit_l.path[flit_l.path_index])])
                            link_station = self.ring_bridge["TL"].get((flit_l.current_link[1], flit_l.path[flit_l.path_index]))
                            new_current = flit_l.current_link[1]
                            new_next_node = flit_l.path[flit_l.path_index]
                            # if len(link_station) < self.config.RB_IN_FIFO_DEPTH and flit_exist_left:
                            # if len(link_station) < self.config.RB_IN_FIFO_DEPTH:
                            if self.config.RB_IN_FIFO_DEPTH > len(link_station) and (
                                (flit_l.ETag_priority == "T2" and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T2"] < self.config.TL_Etag_T2_UE_MAX)
                                or (flit_l.ETag_priority == "T1" and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T1"] < self.config.TL_Etag_T2_UE_MAX)
                                or (
                                    flit_l.ETag_priority == "T0"
                                    and self.T0_Etag_Order_FIFO[0] == (new_current, flit_l)
                                    and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T0"] < self.config.RB_IN_FIFO_DEPTH
                                )
                            ):
                                if self.links_tag[(current, current)][-1] == [current, "TL"]:
                                    self.links_tag[(current, current)][-1] = None
                                    self.remain_tag["TL"][current] += 1
                                    return True
                else:
                    if self.links_tag[(current, current)][-1] is None:
                        return True
                    else:
                        if self.links_tag[(current, current)][-1] == [current, "TL"]:
                            self.links_tag[(current, current)][-1] = None
                            self.remain_tag["TL"][current] += 1
                            return True
                return False

        if current - next_node == self.config.cols:
            # 向 Ring Bridge 移动
            # return len(self.ring_bridge["TU"][(current, next_node)]) < self.config.RB_IN_FIFO_DEPTH
            # v1.3 在IQ中分TU和TD两个FIFO
            if len(flit.path) > 2 and flit.path[2] - flit.path[1] == self.config.cols * 2:
                return len(self.inject_queues["TD"][current]) < self.config.IQ_OUT_FIFO_DEPTH
            elif len(flit.path) > 2 and flit.path[2] - flit.path[1] == -self.config.cols * 2:
                return len(self.inject_queues["TU"][current]) < self.config.IQ_OUT_FIFO_DEPTH
        elif next_node - current == 1:
            # 向右移动
            if self.links[(current - 1, current)][-1] is not None:
                if self.links_tag[(current - 1, current)][-1] is None:
                    flit_l = self.links[(current - 1, current)][-1]
                    if flit_l.path_index + 1 < len(flit_l.path) and flit_l.current_link[1] - flit_l.path[flit_l.path_index + 1] == self.config.cols:
                        new_current = flit_l.current_link[1]
                        new_next_node = flit_l.path[flit_l.path_index + 1]
                        link_station = self.ring_bridge["TR"].get((new_current, new_next_node))
                        # if self.config.RB_IN_FIFO_DEPTH - len(station_right) > len(self.station_reservations["TR"][(new_current, new_next_node)]):
                        if self.config.RB_IN_FIFO_DEPTH > len(link_station) and (
                            (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T1"] < self.config.RB_IN_FIFO_DEPTH and flit_l.ETag_priority in ["T0", "T1"])
                            or (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] < self.config.TR_Etag_T2_UE_MAX and flit_l.ETag_priority == "T2")
                        ):
                            return True
                    if flit.wait_cycle_h > self.config.ITag_Trigger_Th_H and not flit.itag_h:
                        if self.remain_tag["TR"][current] > 0:
                            self.remain_tag["TR"][current] -= 1
                            self.links_tag[(current - 1, current)][-1] = [current, "TR"]
                            flit.itag_h = True
                else:
                    flit_l = self.links[(current - 1, current)][-1]
                    if flit_l.path_index + 1 < len(flit_l.path) and flit_l.current_link[1] - flit_l.path[flit_l.path_index + 1] == self.config.cols:
                        new_current = flit_l.current_link[1]
                        new_next_node = flit_l.path[flit_l.path_index + 1]
                        link_station = self.ring_bridge["TR"].get((new_current, new_next_node))
                        # if self.config.RB_IN_FIFO_DEPTH - len(station_right) > len(self.station_reservations["TR"][(new_current, new_next_node)]):
                        if self.config.RB_IN_FIFO_DEPTH > len(link_station) and (
                            (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T1"] < self.config.RB_IN_FIFO_DEPTH and flit_l.ETag_priority in ["T0", "T1"])
                            or (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] < self.config.TR_Etag_T2_UE_MAX and flit_l.ETag_priority == "T2")
                        ):
                            if self.links_tag[(current - 1, current)][-1] == [current, "TR"]:
                                self.links_tag[(current - 1, current)][-1] = None
                                self.remain_tag["TR"][current] += 1
                                return True
            else:
                if self.links_tag[(current - 1, current)][-1] is None:
                    return True
                else:
                    if self.links_tag[(current - 1, current)][-1] == [current, "TR"]:
                        self.links_tag[(current - 1, current)][-1] = None
                        self.remain_tag["TR"][current] += 1
                        return True
            return False
        elif current - next_node == 1:
            # 向左移动
            if self.links[(current + 1, current)][-1] is not None:
                if self.links_tag[current + 1, current][-1] is None:
                    flit_l = self.links[(current + 1, current)][-1]
                    if flit_l.path_index + 1 < len(flit_l.path) and flit_l.current_link[1] - flit_l.path[flit_l.path_index + 1] == self.config.cols:
                        new_current = flit_l.current_link[1]
                        new_next_node = flit_l.path[flit_l.path_index + 1]
                        station_left = self.ring_bridge["TL"].get((new_current, new_next_node))
                        # if self.config.RB_IN_FIFO_DEPTH - len(station_left) > len(self.station_reservations["TL"][(new_current, new_next_node)]):
                        if self.config.RB_IN_FIFO_DEPTH > len(station_left) and (
                            (flit_l.ETag_priority == "T2" and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T2"] < self.config.TL_Etag_T2_UE_MAX)
                            or (flit_l.ETag_priority == "T1" and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T1"] < self.config.TL_Etag_T2_UE_MAX)
                            or (
                                flit_l.ETag_priority == "T0"
                                and self.T0_Etag_Order_FIFO[0] == (new_current, flit_l)
                                and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T0"] < self.config.RB_IN_FIFO_DEPTH
                            )
                        ):
                            return True
                    if flit.wait_cycle_h > self.config.ITag_Trigger_Th_H and not flit.itag_h:
                        if self.remain_tag["TL"][current] > 0:
                            self.remain_tag["TL"][current] -= 1
                            self.links_tag[(current + 1, current)][-1] = [current, "TL"]
                            flit.itag_h = True
                else:
                    flit_l = self.links[(current + 1, current)][-1]
                    if flit_l.path_index + 1 < len(flit_l.path) and flit_l.current_link[1] - flit_l.path[flit_l.path_index + 1] == self.config.cols:
                        new_current = flit_l.current_link[1]
                        new_next_node = flit_l.path[flit_l.path_index + 1]
                        station_left = self.ring_bridge["TL"].get((new_current, new_next_node))
                        # if self.config.RB_IN_FIFO_DEPTH - len(station_left) > len(self.station_reservations["TL"][(new_current, new_next_node)]):
                        if self.config.RB_IN_FIFO_DEPTH > len(station_left) and (
                            (flit_l.ETag_priority == "T2" and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T2"] < self.config.TL_Etag_T2_UE_MAX)
                            or (flit_l.ETag_priority == "T1" and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T1"] < self.config.TL_Etag_T2_UE_MAX)
                            or (
                                flit_l.ETag_priority == "T0"
                                and self.T0_Etag_Order_FIFO[0] == (new_current, flit_l)
                                and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T0"] < self.config.RB_IN_FIFO_DEPTH
                            )
                        ):
                            if self.links_tag[(current + 1, current)][-1] == [current, "TL"]:
                                self.links_tag[(current + 1, current)][-1] = None
                                self.remain_tag["TL"][current] += 1
                                return True
            else:
                if self.links_tag[(current + 1, current)][-1] is None:
                    return True
                else:
                    if self.links_tag[(current + 1, current)][-1] == [current, "TL"]:
                        self.links_tag[(current + 1, current)][-1] = None
                        self.remain_tag["TL"][current] += 1
                        return True
            return False

    def plan_move(self, flit):
        # if flit.packet_id == 7 and flit.flit_id == -1:
        # print(flit)
        if flit.is_new_on_network:
            current = flit.source
            next_node = flit.path[flit.path_index + 1]
            flit.current_position = current
            flit.is_new_on_network = False
            flit.is_arrive = False
            flit.is_on_station = False
            flit.current_link = (current, next_node)
            # flit.current_seat_index = 2 if (current - next_node == self.config.cols) else 0
            if current - next_node == self.config.cols:
                if len(flit.path) > 2 and flit.path[flit.path_index + 2] - next_node == 2 * self.config.cols:
                    flit.current_seat_index = -1
                elif len(flit.path) > 2 and flit.path[flit.path_index + 2] - next_node == -2 * self.config.cols:
                    flit.current_seat_index = -2
            else:
                flit.current_seat_index = 0

            flit.path_index += 1
            return

        # 计算行和列的起始和结束点
        current, next_node = flit.current_link
        row_start = (current // self.config.cols) * self.config.cols
        row_start = row_start if (row_start // self.config.cols) % 2 != 0 else -1
        row_end = row_start + self.config.cols - 1 if row_start > 0 else -1
        col_start = current % (self.config.cols * 2)
        col_start = col_start if col_start < self.config.cols else -1
        col_end = col_start + self.config.num_nodes - self.config.cols * 2 if col_start >= 0 else -1

        link = self.links.get(flit.current_link)
        # if link and flit.current_seat_index == len(link) - 1:
        #     print(self.name, flit.current_link, flit.packet_id, flit.current_seat_index, flit.flit_id)
        #     self.links_flow_stat[flit.req_type][flit.current_link] += 1
        # Plan non ring bridge moves
        if current - next_node != self.config.cols:
            # Handling delay flits
            if flit.is_delay:
                return self._handle_delay_flit(flit, link, current, next_node, row_start, row_end, col_start, col_end)
            # Handling regular flits
            else:
                return self._handle_regular_flit(flit, link, current, next_node, row_start, row_end, col_start, col_end)

    def _handle_delay_flit(self, flit, link, current, next_node, row_start, row_end, col_start, col_end):
        # 1. 非链路末端
        if flit.current_seat_index < len(link) - 1:
            link[flit.current_seat_index] = None
            flit.current_seat_index += 1
            return

        # 2. 到达链路末端
        new_current, new_next_node = next_node, flit.path[flit.path_index]  # delay情况下path_index不更新
        # A. 处理横边界情况
        if current == next_node:
            # A1. 左边界情况
            if current == row_start:
                if current == flit.current_position:
                    # Flit已经绕横向环一圈
                    flit.circuits_completed_h += 1
                    # flit_exist_left = any(flit_l.id == flit.id for flit_l in self.station_reservations["TL"][(new_current, new_next_node)])
                    # flit_exist_right = any(flit_r.id == flit.id for flit_r in self.station_reservations["TR"][(new_current, new_next_node)])
                    link_station = self.ring_bridge["TR"].get((next_node, flit.path[flit.path_index]))
                    # TR方向尝试下环
                    if (
                        len(link_station) < self.config.RB_IN_FIFO_DEPTH
                        # and flit_exist_right
                        and (
                            (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T1"] < self.config.RB_IN_FIFO_DEPTH and flit.ETag_priority in ["T0", "T1"])
                            or (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] < self.config.TR_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                        )
                    ):
                        flit.is_delay = False
                        flit.current_link = (next_node, flit.path[flit.path_index])
                        link[flit.current_seat_index] = None
                        flit.current_seat_index = 1
                        # self.station_reservations["TR"][(next_node, flit.path[flit.path_index])].remove(flit)
                        if flit.ETag_priority == "T2":
                            self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] += 1
                        self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T1"] += 1
                        if flit.ETag_priority == "T0":
                            # 若升级到T0则需要从T0队列中移除flit
                            self.T0_Etag_Order_FIFO.remove((next_node, flit))
                    # elif (
                    #     not flit_exist_right
                    #     and self.config.RB_IN_FIFO_DEPTH - len(link_station) > len(self.station_reservations["TR"][(new_current, new_next_node)])
                    #     and (
                    #         (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T1"] < self.config.RB_IN_FIFO_DEPTH and flit.ETag_priority in ["T0", "T1"])
                    #         or (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] < self.config.TR_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                    #     )
                    # ):
                    #     flit.is_delay = False
                    #     flit.current_link = (next_node, flit.path[flit.path_index])
                    #     link[flit.current_seat_index] = None
                    #     flit.current_seat_index = 1
                    #     if flit.ETag_priority == "T2":
                    #         self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] += 1
                    #     self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T1"] += 1
                    #     if flit.ETag_priority == "T0":
                    #         self.T0_Etag_Order_FIFO.remove((next_node, flit))
                    #     if flit_exist_left:
                    #         self.station_reservations["TL"][(next_node, flit.path[flit.path_index])].remove(flit)
                    else:
                        # 无法下环,TR方向的flit不能升级T0
                        # if not flit_exist_left and not flit_exist_right:
                        # if len(self.station_reservations["TL"][(new_current, new_next_node)]) < self.config.reservation_num:
                        # self.station_reservations["TL"][(new_current, new_next_node)].append(flit)
                        link[flit.current_seat_index] = None
                        next_pos = next_node + 1
                        flit.current_link = (next_node, next_pos)
                        flit.current_seat_index = 0
                        if not self.Both_side_ETag_upgrade and flit.ETag_priority == "T2":
                            flit.ETag_priority = "T1"
                else:
                    # Flit未绕回下环点，向右绕环
                    link[flit.current_seat_index] = None
                    next_pos = next_node + 1
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0

            # A2. 右边界情况：
            elif current == row_end:
                if current == flit.current_position:
                    flit.circuits_completed_h += 1
                    # flit_exist_left = any(flit_l.id == flit.id for flit_l in self.station_reservations["TL"][(new_current, new_next_node)])
                    # flit_exist_right = any(flit_r.id == flit.id for flit_r in self.station_reservations["TR"][(new_current, new_next_node)])
                    # 绕环超过阈值，通过FT下环
                    if flit.circuits_completed_h > self.config.ft_count:
                        link_station = self.ring_bridge["ft"].get((next_node, flit.path[flit.path_index]))
                        if len(link_station) < self.config.ft_len:
                            flit.is_delay = False
                            flit.current_link = (new_current, new_next_node)
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = -3
                            if flit.ETag_priority == "T0":
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                            # if flit_exist_left:
                            #     self.station_reservations["TL"][(new_current, new_next_node)].remove(flit)
                            # elif flit_exist_right:
                            #     self.station_reservations["TR"][(new_current, new_next_node)].remove(flit)
                        else:
                            # FT无法下环，向左绕环
                            # if not flit_exist_left and not flit_exist_right:
                            # if len(self.station_reservations["TR"][(new_current, new_next_node)]) < self.config.reservation_num:
                            # self.station_reservations["TR"][(new_current, new_next_node)].append(flit)
                            link[flit.current_seat_index] = None
                            next_pos = next_node - 1
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                    # 尝试TL下环，非T0情况
                    elif flit.ETag_priority in ["T1", "T2"]:
                        link_station = self.ring_bridge["TL"].get((new_current, new_next_node))
                        if (
                            len(link_station) < self.config.RB_IN_FIFO_DEPTH
                            # and flit_exist_left
                            and (
                                (self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T1"] < self.config.TL_Etag_T1_UE_MAX and flit.ETag_priority == "T1")
                                or (self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T2"] < self.config.TL_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                            )
                        ):
                            flit.is_delay = False
                            flit.current_link = (new_current, new_next_node)
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0
                            # self.station_reservations["TL"][(new_current, new_next_node)].remove(flit)
                            if flit.ETag_priority == "T2":
                                self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T2"] += 1
                            self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T1"] += 1
                            self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T0"] += 1
                        # elif (
                        #     not flit_exist_left
                        #     and self.config.RB_IN_FIFO_DEPTH - len(link_station) > len(self.station_reservations["TL"][(new_current, new_next_node)])
                        #     and (
                        #         (self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T1"] < self.config.TL_Etag_T1_UE_MAX and flit.ETag_priority == "T1")
                        #         or (self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T2"] < self.config.TL_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                        #     )
                        # ):
                        #     flit.is_delay = False
                        #     flit.current_link = (next_node, flit.path[flit.path_index])
                        #     link[flit.current_seat_index] = None
                        #     flit.current_seat_index = 0
                        #     if flit.ETag_priority == "T2":
                        #         self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T2"] += 1
                        #     self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T1"] += 1
                        #     self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T0"] += 1
                        #     if flit_exist_right:
                        #         self.station_reservations["TR"][(next_node, flit.path[flit.path_index])].remove(flit)
                        else:
                            # 无法下环,升级ETag并记录
                            if flit.ETag_priority == "T2":
                                flit.ETag_priority = "T1"
                            elif flit.ETag_priority == "T1":
                                flit.ETag_priority = "T0"
                                self.T0_Etag_Order_FIFO.append((next_node, flit))
                            # if not flit_exist_left and not flit_exist_right:
                            #     if len(self.station_reservations["TR"][(new_current, new_next_node)]) < self.config.reservation_num:
                            #         self.station_reservations["TR"][(new_current, new_next_node)].append(flit)
                            link[flit.current_seat_index] = None
                            next_pos = next_node - 1
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                    # 尝试TL以T0下环
                    elif flit.ETag_priority == "T0":
                        if self.T0_Etag_Order_FIFO[0] == (next_node, flit) and self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T0"] < self.config.RB_IN_FIFO_DEPTH:
                            self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T0"] += 1
                            flit.is_delay = False
                            flit.current_link = (new_current, new_next_node)
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0
                            self.T0_Etag_Order_FIFO.popleft()
                        else:
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
            # A3. 上边界情况：
            elif current == col_start:
                if next_node == flit.destination:
                    flit.circuits_completed_v += 1
                    # flit_exist_up = any(flit_u.id == flit.id for flit_u in self.eject_reservations["TU"][next_node])
                    # flit_exist_down = any(flit_r.id == flit.id for flit_r in self.eject_reservations["TD"][next_node])
                    link_eject = self.eject_queues["TD"][next_node]
                    if (
                        len(link_eject) < self.config.EQ_IN_FIFO_DEPTH
                        # and flit_exist_down
                        and (
                            (self.EQ_UE_Counters["TD"][next_node]["T1"] < self.config.EQ_IN_FIFO_DEPTH and flit.ETag_priority in ["T1", "T0"])
                            or (self.EQ_UE_Counters["TD"][next_node]["T2"] < self.config.TD_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                        )
                    ):
                        flit.is_delay = False
                        flit.is_arrive = True
                        link[flit.current_seat_index] = None
                        flit.current_seat_index = 0
                        # self.eject_reservations["TD"][next_node].remove(flit)
                        if flit.ETag_priority == "T2":
                            self.EQ_UE_Counters["TD"][next_node]["T2"] += 1
                        self.EQ_UE_Counters["TD"][next_node]["T1"] += 1
                        if flit.ETag_priority == "T0":
                            self.T0_Etag_Order_FIFO.remove((next_node, flit))
                    # elif (
                    #     not flit_exist_down
                    #     and self.config.EQ_IN_FIFO_DEPTH - len(link_eject) > len(self.eject_reservations["TD"][next_node])
                    #     and (
                    #         (self.EQ_UE_Counters["TD"][next_node]["T1"] < self.config.EQ_IN_FIFO_DEPTH and flit.ETag_priority in ["T1", "T0"])
                    #         or (self.EQ_UE_Counters["TD"][next_node]["T2"] < self.config.TD_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                    #     )
                    # ):
                    #     flit.is_delay = False
                    #     flit.is_arrive = True
                    #     link[flit.current_seat_index] = None
                    #     flit.current_seat_index = 0
                    #     if flit.ETag_priority == "T2":
                    #         self.EQ_UE_Counters["TD"][next_node]["T2"] += 1
                    #     self.EQ_UE_Counters["TD"][next_node]["T1"] += 1
                    #     if flit.ETag_priority == "T0":
                    #         self.T0_Etag_Order_FIFO.remove((next_node, flit))
                    #     if flit_exist_up:
                    #         self.eject_reservations["TU"][next_node].remove(flit)
                    else:
                        # 无法下环,TD方向的flit不能升级T0
                        if not self.Both_side_ETag_upgrade and flit.ETag_priority == "T2":
                            flit.ETag_priority = "T1"
                        # if not flit_exist_up and not flit_exist_down:
                        # if len(self.eject_reservations["TU"][next_node]) < self.config.reservation_num:
                        # self.eject_reservations["TU"][next_node].append(flit)
                        link[flit.current_seat_index] = None
                        next_pos = next_node + self.config.cols * 2
                        flit.current_link = (next_node, next_pos)
                        flit.current_seat_index = 0
                else:
                    link[flit.current_seat_index] = None
                    next_pos = next_node + self.config.cols * 2
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0
            # A4. 下边界情况：
            elif current == col_end:
                if next_node == flit.destination:
                    flit.circuits_completed_v += 1
                    # flit_exist_up = any(flit_u.id == flit.id for flit_u in self.eject_reservations["TU"][next_node])
                    # flit_exist_down = any(flit_r.id == flit.id for flit_r in self.eject_reservations["TD"][next_node])
                    link_eject = self.eject_queues["TU"][next_node]
                    if flit.ETag_priority in ["T1", "T2"]:
                        if (
                            len(link_eject) < self.config.EQ_IN_FIFO_DEPTH
                            # and flit_exist_up
                            and (
                                (self.EQ_UE_Counters["TU"][next_node]["T1"] < self.config.TU_Etag_T1_UE_MAX and flit.ETag_priority == "T1")
                                or (self.EQ_UE_Counters["TU"][next_node]["T2"] < self.config.TU_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                            )
                        ):
                            flit.is_delay = False
                            flit.is_arrive = True
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0
                            # self.eject_reservations["TU"][next_node].remove(flit)
                            if flit.ETag_priority == "T2":
                                self.EQ_UE_Counters["TU"][next_node]["T2"] += 1
                            self.EQ_UE_Counters["TU"][next_node]["T1"] += 1
                            self.EQ_UE_Counters["TU"][next_node]["T0"] += 1
                        # elif (
                        #     not flit_exist_up
                        #     and self.config.EQ_IN_FIFO_DEPTH - len(link_eject) > len(self.eject_reservations["TU"][next_node])
                        #     and (
                        #         (self.EQ_UE_Counters["TU"][next_node]["T1"] < self.config.TU_Etag_T1_UE_MAX and flit.ETag_priority == "T1")
                        #         or (self.EQ_UE_Counters["TU"][next_node]["T2"] < self.config.TU_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                        #     )
                        # ):
                        #     flit.is_delay = False
                        #     flit.is_arrive = True
                        #     link[flit.current_seat_index] = None
                        #     flit.current_seat_index = 0
                        #     if flit.ETag_priority == "T2":
                        #         self.EQ_UE_Counters["TU"][next_node]["T2"] += 1
                        #     self.EQ_UE_Counters["TU"][next_node]["T1"] += 1
                        #     self.EQ_UE_Counters["TU"][next_node]["T0"] += 1
                        #     if flit_exist_down:
                        #         self.eject_reservations["TD"][next_node].remove(flit)
                        else:
                            # 无法下环,升级ETag并记录
                            if flit.ETag_priority == "T2":
                                flit.ETag_priority = "T1"
                            elif flit.ETag_priority == "T1":
                                self.T0_Etag_Order_FIFO.append((next_node, flit))
                                flit.ETag_priority = "T0"
                            # if not flit_exist_up and not flit_exist_down:
                            #     if len(self.eject_reservations["TD"][next_node]) < self.config.reservation_num:
                            #         self.eject_reservations["TD"][next_node].append(flit)
                            link[flit.current_seat_index] = None
                            next_pos = next_node - self.config.cols * 2
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                    elif flit.ETag_priority == "T0":
                        if self.T0_Etag_Order_FIFO[0] == (next_node, flit) and self.EQ_UE_Counters["TU"][next_node]["T0"] < self.config.EQ_IN_FIFO_DEPTH:
                            self.EQ_UE_Counters["TU"][next_node]["T0"] += 1
                            flit.is_delay = False
                            flit.is_arrive = True
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0
                            self.T0_Etag_Order_FIFO.popleft()
                        else:
                            link[flit.current_seat_index] = None
                            next_pos = next_node - self.config.cols * 2
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                else:
                    link[flit.current_seat_index] = None
                    next_pos = next_node - self.config.cols * 2
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0
        # B. 非边界横向环情况
        elif abs(current - next_node) == 1:
            if next_node == flit.current_position:
                flit.circuits_completed_h += 1
                # flit_exist_left = any(flit_l.id == flit.id for flit_l in self.station_reservations["TL"][(next_node, flit.path[flit.path_index])])
                # flit_exist_right = any(flit_r.id == flit.id for flit_r in self.station_reservations["TR"][(next_node, flit.path[flit.path_index])])
                if flit.circuits_completed_h > self.config.ft_count and current - next_node == 1:
                    link_station = self.ring_bridge["ft"].get((next_node, flit.path[flit.path_index]))
                    if len(link_station) < self.config.ft_len:
                        flit.is_delay = False
                        flit.current_link = (next_node, flit.path[flit.path_index])
                        link[flit.current_seat_index] = None
                        flit.current_seat_index = -3
                        if flit.ETag_priority == "T0":
                            self.T0_Etag_Order_FIFO.remove((next_node, flit))
                        # if flit_exist_left:
                        #     self.station_reservations["TL"][(new_current, new_next_node)].remove(flit)
                        # elif flit_exist_right:
                        #     self.station_reservations["TR"][(new_current, new_next_node)].remove(flit)
                    else:
                        # if not flit_exist_left and not flit_exist_right:
                        #     if len(self.station_reservations["TR"][(new_current, new_next_node)]) < self.config.reservation_num:
                        #         self.station_reservations["TR"][(new_current, new_next_node)].append(flit)
                        link[flit.current_seat_index] = None
                        if current - next_node == 1:
                            next_pos = max(next_node - 1, row_start)
                        else:
                            next_pos = min(next_node + 1, row_end)
                        flit.current_link = (next_node, next_pos)
                        flit.current_seat_index = 0
                else:
                    if current - next_node == 1:
                        if flit.ETag_priority in ["T1", "T2"]:
                            link_station = self.ring_bridge["TL"].get((new_current, new_next_node))
                            if (
                                len(link_station) < self.config.RB_IN_FIFO_DEPTH
                                # and flit_exist_left
                                and (
                                    (self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T1"] < self.config.TL_Etag_T1_UE_MAX and flit.ETag_priority == "T1")
                                    or (self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T2"] < self.config.TL_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                                )
                            ):
                                flit.is_delay = False
                                flit.current_link = (new_current, new_next_node)
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                                # self.station_reservations["TL"][(new_current, new_next_node)].remove(flit)
                                if flit.ETag_priority == "T2":
                                    self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T2"] += 1
                                self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T1"] += 1
                                self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T0"] += 1
                                # elif (
                                #     not flit_exist_left
                                #     and self.config.RB_IN_FIFO_DEPTH - len(link_station) > len(self.station_reservations["TL"][(new_current, new_next_node)])
                                #     and (
                                #         (self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T1"] < self.config.TL_Etag_T1_UE_MAX and flit.ETag_priority == "T1")
                                #         or (self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T2"] < self.config.TL_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                                #     )
                                # ):
                                #     flit.is_delay = False
                                #     flit.current_link = (next_node, flit.path[flit.path_index])
                                #     link[flit.current_seat_index] = None
                                #     flit.current_seat_index = 0
                                #     if flit.ETag_priority == "T2":
                                #         self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T2"] += 1
                                #     self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T1"] += 1
                                #     self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T0"] += 1
                                #     if flit_exist_right:
                                #         self.station_reservations["TR"][(next_node, flit.path[flit.path_index])].remove(flit)
                            else:
                                if flit.ETag_priority == "T2":
                                    flit.ETag_priority = "T1"
                                elif flit.ETag_priority == "T1":
                                    self.T0_Etag_Order_FIFO.append((next_node, flit))
                                    flit.ETag_priority = "T0"
                                # if not flit_exist_left and not flit_exist_right:
                                #     if len(self.station_reservations["TR"][(new_current, new_next_node)]) < self.config.reservation_num:
                                #         self.station_reservations["TR"][(new_current, new_next_node)].append(flit)
                                link[flit.current_seat_index] = None
                                next_pos = max(next_node - 1, row_start)
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                        elif (
                            flit.ETag_priority == "T0"
                            and self.T0_Etag_Order_FIFO[0] == (next_node, flit)
                            and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T0"] < self.config.RB_IN_FIFO_DEPTH
                        ):
                            self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T0"] += 1
                            flit.is_delay = False
                            flit.current_link = (next_node, flit.path[flit.path_index])
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0
                            self.T0_Etag_Order_FIFO.popleft()
                        else:
                            link[flit.current_seat_index] = None
                            if current - next_node == 1:
                                next_pos = max(next_node - 1, row_start)
                            else:
                                next_pos = min(next_node + 1, row_end)
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                    else:
                        # 横向环TR尝试下环
                        link_station = self.ring_bridge["TR"].get((new_current, new_next_node))
                        if (
                            len(link_station) < self.config.RB_IN_FIFO_DEPTH
                            # and flit_exist_right
                            and (
                                (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T1"] < self.config.RB_IN_FIFO_DEPTH and flit.ETag_priority in ["T1", "T0"])
                                or (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] < self.config.TR_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                            )
                        ):
                            flit.is_delay = False
                            flit.current_link = (new_current, new_next_node)
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 1
                            # self.station_reservations["TR"][(new_current, new_next_node)].remove(flit)
                            if flit.ETag_priority == "T2":
                                self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] += 1
                            self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T1"] += 1
                            if flit.ETag_priority == "T0":
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                        # elif (
                        #     not flit_exist_right
                        #     and self.config.RB_IN_FIFO_DEPTH - len(link_station) > len(self.station_reservations["TR"][(new_current, new_next_node)])
                        #     and (
                        #         (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T1"] < self.config.RB_IN_FIFO_DEPTH and flit.ETag_priority in ["T1", "T0"])
                        #         or (self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] < self.config.TR_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                        #     )
                        # ):
                        #     flit.is_delay = False
                        #     flit.current_link = (next_node, flit.path[flit.path_index])
                        #     link[flit.current_seat_index] = None
                        #     flit.current_seat_index = 1
                        #     if flit.ETag_priority == "T2":
                        #         self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] += 1
                        #     self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T1"] += 1
                        #     if flit.ETag_priority == "T0":
                        #         self.T0_Etag_Order_FIFO.remove((next_node, flit))
                        #     if flit_exist_left:
                        #         self.station_reservations["TL"][(next_node, flit.path[flit.path_index])].remove(flit)
                        else:
                            # if not flit_exist_left and not flit_exist_right:
                            #     if len(self.station_reservations["TL"][(new_current, new_next_node)]) < self.config.reservation_num:
                            #         self.station_reservations["TL"][(new_current, new_next_node)].append(flit)
                            link[flit.current_seat_index] = None
                            next_pos = min(next_node + 1, row_end)
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                            if not self.Both_side_ETag_upgrade and flit.ETag_priority == "T2":
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
            if next_node == flit.destination:
                flit.circuits_completed_v += 1
                # flit_exist_up = any(flit_u.id == flit.id for flit_u in self.eject_reservations["TU"][next_node])
                # flit_exist_down = any(flit_r.id == flit.id for flit_r in self.eject_reservations["TD"][next_node])
                if current - next_node == self.config.cols * 2:
                    if flit.ETag_priority in ["T1", "T2"]:
                        # up move
                        link_eject = self.eject_queues["TU"][next_node]
                        if (
                            len(link_eject) < self.config.EQ_IN_FIFO_DEPTH
                            # and flit_exist_up
                            and (
                                (self.EQ_UE_Counters["TU"][next_node]["T1"] < self.config.TU_Etag_T1_UE_MAX and flit.ETag_priority == "T1")
                                or (self.EQ_UE_Counters["TU"][next_node]["T2"] < self.config.TU_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                            )
                        ):
                            flit.is_delay = False
                            flit.is_arrive = True
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0
                            # self.eject_reservations["TU"][next_node].remove(flit)
                            if flit.ETag_priority == "T2":
                                self.EQ_UE_Counters["TU"][next_node]["T2"] += 1
                            self.EQ_UE_Counters["TU"][next_node]["T1"] += 1
                            self.EQ_UE_Counters["TU"][next_node]["T0"] += 1
                        # elif (
                        #     not flit_exist_up
                        #     and self.config.EQ_IN_FIFO_DEPTH - len(link_eject) > len(self.eject_reservations["TU"][next_node])
                        #     and (
                        #         (self.EQ_UE_Counters["TU"][next_node]["T1"] < self.config.TU_Etag_T1_UE_MAX and flit.ETag_priority == "T1")
                        #         or (self.EQ_UE_Counters["TU"][next_node]["T2"] < self.config.TU_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                        #     )
                        # ):
                        #     flit.is_delay = False
                        #     flit.is_arrive = True
                        #     link[flit.current_seat_index] = None
                        #     flit.current_seat_index = 0
                        #     self.EQ_UE_Counters["TU"][next_node]["T1"] += 1
                        #     self.EQ_UE_Counters["TU"][next_node]["T0"] += 1
                        #     if flit_exist_down:
                        #         self.eject_reservations["TD"][next_node].remove(flit)
                        else:
                            if flit.ETag_priority == "T2":
                                flit.ETag_priority = "T1"
                            elif flit.ETag_priority == "T1":
                                self.T0_Etag_Order_FIFO.append((next_node, flit))
                                flit.ETag_priority = "T0"
                            # if not flit_exist_up and not flit_exist_down:
                            #     if len(self.eject_reservations["TD"][next_node]) < self.config.reservation_num:
                            #         self.eject_reservations["TD"][next_node].append(flit)
                            link[flit.current_seat_index] = None
                            next_pos = next_node - self.config.cols * 2 if next_node - self.config.cols * 2 >= col_start else col_start
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                    elif flit.ETag_priority == "T0" and self.T0_Etag_Order_FIFO[0] == (next_node, flit) and self.EQ_UE_Counters["TU"][next_node]["T0"] < self.config.EQ_IN_FIFO_DEPTH:
                        self.EQ_UE_Counters["TU"][next_node]["T0"] += 1
                        flit.is_delay = False
                        flit.is_arrive = True
                        link[flit.current_seat_index] = None
                        flit.current_seat_index = 0
                        self.T0_Etag_Order_FIFO.popleft()
                    else:
                        link[flit.current_seat_index] = None
                        if current - next_node == self.config.cols * 2:
                            next_pos = max(next_node - self.config.cols * 2, col_start)
                        else:
                            next_pos = min(next_node + self.config.cols * 2, col_end)
                        flit.current_link = (next_node, next_pos)
                        flit.current_seat_index = 0
                else:
                    # down move
                    link_eject = self.eject_queues["TD"][next_node]
                    if (
                        len(link_eject) < self.config.EQ_IN_FIFO_DEPTH
                        # and flit_exist_down
                        and (
                            (self.EQ_UE_Counters["TD"][next_node]["T1"] < self.config.EQ_IN_FIFO_DEPTH and flit.ETag_priority in ["T1", "T0"])
                            or (self.EQ_UE_Counters["TD"][next_node]["T2"] < self.config.TD_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                        )
                    ):
                        flit.is_delay = False
                        flit.is_arrive = True
                        link[flit.current_seat_index] = None
                        flit.current_seat_index = 0
                        if flit.ETag_priority == "T2":
                            self.EQ_UE_Counters["TD"][next_node]["T2"] += 1
                        self.EQ_UE_Counters["TD"][next_node]["T1"] += 1
                        if flit.ETag_priority == "T0":
                            self.T0_Etag_Order_FIFO.remove((next_node, flit))
                        # self.eject_reservations["TD"][next_node].remove(flit)
                    # elif (
                    #     not flit_exist_down
                    #     and self.config.EQ_IN_FIFO_DEPTH - len(link_eject) > len(self.eject_reservations["TD"][next_node])
                    #     and (
                    #         (self.EQ_UE_Counters["TD"][next_node]["T1"] < self.config.EQ_IN_FIFO_DEPTH and flit.ETag_priority in ["T1", "T0"])
                    #         or (self.EQ_UE_Counters["TD"][next_node]["T2"] < self.config.TD_Etag_T2_UE_MAX and flit.ETag_priority == "T2")
                    #     )
                    # ):
                    #     flit.is_delay = False
                    #     flit.is_arrive = True
                    #     link[flit.current_seat_index] = None
                    #     flit.current_seat_index = 0
                    #     if flit.ETag_priority == "T2":
                    #         self.EQ_UE_Counters["TD"][next_node]["T2"] += 1
                    #     self.EQ_UE_Counters["TD"][next_node]["T1"] += 1
                    #     if flit.ETag_priority == "T0":
                    #         self.T0_Etag_Order_FIFO.remove((next_node, flit))
                    #     if flit_exist_up:
                    #         self.eject_reservations["TU"][next_node].remove(flit)
                    else:
                        # if not flit_exist_up and not flit_exist_down:
                        #     if len(self.eject_reservations["TU"][next_node]) < self.config.reservation_num:
                        #         self.eject_reservations["TU"][next_node].append(flit)
                        link[flit.current_seat_index] = None
                        next_pos = min(next_node + self.config.cols * 2, col_end)
                        flit.current_link = (next_node, next_pos)
                        flit.current_seat_index = 0
            else:
                link[flit.current_seat_index] = None
                if current - next_node == self.config.cols * 2:
                    next_pos = max(next_node - self.config.cols * 2, col_start)
                else:
                    next_pos = min(next_node + self.config.cols * 2, col_end)
                flit.current_link = (next_node, next_pos)
                flit.current_seat_index = 0
        return

    def _handle_regular_flit(self, flit, link, current, next_node, row_start, row_end, col_start, col_end):
        # 1. 非链路末端：在当前链路上前进一步
        if flit.current_seat_index < len(link) - 1:
            link[flit.current_seat_index] = None
            flit.current_seat_index += 1
            return

        # 2. 已经到达
        flit.current_position = next_node
        # 检查是否还有后续路径
        if flit.path_index + 1 < len(flit.path):
            flit.path_index += 1
            new_current, new_next_node = next_node, flit.path[flit.path_index]

            # 2a. 正常绕环
            if new_current - new_next_node != self.config.cols:
                flit.current_link = (new_current, new_next_node)
                link[flit.current_seat_index] = None
                flit.current_seat_index = 0

            # 2b. 横向环向左进入Ring Bridge
            elif current - next_node == 1:
                station = self.ring_bridge["TL"].get((new_current, new_next_node))
                # reservations = self.station_reservations["TL"][(new_current, new_next_node)]
                # TL有空位
                # if self.config.RB_IN_FIFO_DEPTH - len(station) > len(reservations) and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T2"] < self.config.TL_Etag_T2_UE_MAX:
                if self.config.RB_IN_FIFO_DEPTH > len(station) and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T2"] < self.config.TL_Etag_T2_UE_MAX:
                    flit.current_link = (new_current, new_next_node)
                    link[flit.current_seat_index] = None
                    flit.current_seat_index = 0
                    # 更新计数器
                    self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T2"] += 1
                    self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T1"] += 1
                    self.RB_UE_Counters["TL"][(new_current, new_next_node)]["T0"] += 1
                else:
                    # TL无空位：预留到右侧等待队列，设置延迟标志，ETag升级
                    # if len(self.station_reservations["TR"][(new_current, new_next_node)]) < self.config.reservation_num:
                    # self.station_reservations["TR"][(new_current, new_next_node)].append(flit)
                    flit.ETag_priority = "T1"
                    next_pos = next_node - 1 if next_node - 1 >= row_start else row_start
                    flit.is_delay = True
                    link[flit.current_seat_index] = None
                    flit.current_link = (new_current, next_pos)
                    flit.current_seat_index = 0

            # 2c. 横向环向右进入Ring Bridge
            elif current - next_node == -1:
                station = self.ring_bridge["TR"].get((new_current, new_next_node))
                # reservations = self.station_reservations["TR"][(new_current, new_next_node)]
                # if self.config.RB_IN_FIFO_DEPTH - len(station) > len(reservations) and self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] < self.config.TR_Etag_T2_UE_MAX:
                if self.config.RB_IN_FIFO_DEPTH > len(station) and self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] < self.config.TR_Etag_T2_UE_MAX:
                    flit.current_link = (new_current, new_next_node)
                    link[flit.current_seat_index] = None
                    flit.current_seat_index = 1
                    self.RB_UE_Counters["TR"][(new_current, new_next_node)]["T2"] += 1
                    self.RB_UE_Counters["TR"][(new_current, new_next_node)]["T1"] += 1
                else:
                    # TR无空位：设置延迟标志，如果双边ETag升级，则升级ETag。
                    # if len(self.station_reservations["TL"][(new_current, new_next_node)]) < self.config.reservation_num:
                    # self.station_reservations["TL"][(new_current, new_next_node)].append(flit)
                    if self.Both_side_ETag_upgrade:
                        flit.ETag_priority = "T1"
                    next_pos = next_node + 1 if next_node + 1 <= row_end else row_end
                    flit.is_delay = True
                    link[flit.current_seat_index] = None
                    flit.current_link = (new_current, next_pos)
                    flit.current_seat_index = 0
        else:
            # 3. 已经到达目的地，执行eject逻辑
            if current - next_node == self.config.cols * 2:  # 纵向环向上TU
                eject_queue = self.eject_queues["TU"][next_node]
                # if self.config.EQ_IN_FIFO_DEPTH - len(eject_queue) > len(self.eject_reservations["TU"][next_node]) and self.EQ_UE_Counters["TU"][next_node]["T2"] < self.config.TU_Etag_T2_UE_MAX:
                if self.config.EQ_IN_FIFO_DEPTH > len(eject_queue) and self.EQ_UE_Counters["TU"][next_node]["T2"] < self.config.TU_Etag_T2_UE_MAX:
                    link[flit.current_seat_index] = None
                    flit.current_seat_index = 0
                    flit.is_arrive = True
                    self.EQ_UE_Counters["TU"][next_node]["T2"] += 1
                    self.EQ_UE_Counters["TU"][next_node]["T1"] += 1
                    self.EQ_UE_Counters["TU"][next_node]["T0"] += 1
                else:
                    # if len(self.eject_reservations["TD"][next_node]) < self.config.reservation_num:
                    # self.eject_reservations["TD"][next_node].append(flit)
                    flit.ETag_priority = "T1"
                    next_pos = next_node - self.config.cols * 2 if next_node - self.config.cols * 2 >= col_start else col_start
                    flit.is_delay = True
                    link[flit.current_seat_index] = None
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0
            elif current - next_node == -self.config.cols * 2:  # 纵向环向下TD
                eject_queue = self.eject_queues["TD"][next_node]
                # if self.config.EQ_IN_FIFO_DEPTH - len(eject_queue) > len(self.eject_reservations["TD"][next_node]) and self.EQ_UE_Counters["TD"][next_node]["T2"] < self.config.TD_Etag_T2_UE_MAX:
                if self.config.EQ_IN_FIFO_DEPTH > len(eject_queue) and self.EQ_UE_Counters["TD"][next_node]["T2"] < self.config.TD_Etag_T2_UE_MAX:
                    link[flit.current_seat_index] = None
                    flit.current_seat_index = 0
                    flit.is_arrive = True
                    self.EQ_UE_Counters["TD"][next_node]["T2"] += 1
                    self.EQ_UE_Counters["TD"][next_node]["T1"] += 1
                else:
                    # if len(self.eject_reservations["TU"][next_node]) < self.config.reservation_num:
                    # self.eject_reservations["TU"][next_node].append(flit)
                    if self.Both_side_ETag_upgrade:
                        flit.ETag_priority = "T1"
                    next_pos = next_node + self.config.cols * 2 if next_node + self.config.cols * 2 <= col_end else col_end
                    flit.is_delay = True
                    link[flit.current_seat_index] = None
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0

    def execute_moves(self, flit, cycle):
        if not flit.is_arrive:
            current, next_node = flit.current_link
            if current - next_node != self.config.cols:
                link = self.links.get(flit.current_link)
                link[flit.current_seat_index] = flit
                if (flit.current_seat_index == 6 and len(link) == 7) or (flit.current_seat_index == 1 and len(link) == 2):
                    self.links_flow_stat[flit.req_type][flit.current_link] += 1
            else:
                # 将 flit 放入 ring_bridge 的相应方向
                if not flit.is_on_station:
                    # 使用字典映射 seat_index 到 ring_bridge 的方向和深度限制
                    direction, max_depth = self.ring_bridge_map.get(flit.current_seat_index, (None, None))
                    if direction is None:
                        return False
                    if direction in self.ring_bridge.keys() and len(self.ring_bridge[direction][flit.current_link]) < max_depth:
                        self.ring_bridge[direction][flit.current_link].append(flit)
                        flit.is_on_station = True
            return False
        else:
            if flit.current_link is not None:
                current, next_node = flit.current_link
            flit.arrival_network_cycle = cycle

            if flit.source - flit.destination == self.config.cols:
                # queue = self.eject_queues["IQ"][flit.destination]
                flit.is_arrived = True

                return True
            elif current - next_node == self.config.cols * 2 or (current == next_node and current not in range(0, self.config.cols)):
                queue = self.eject_queues["TU"][next_node]
            else:
                queue = self.eject_queues["TD"][next_node]

            if len(queue) < self.config.EQ_IN_FIFO_DEPTH:
                queue.append(flit)
                return True
            flit.is_arrive = False
            flit.is_delay = True
            return False
