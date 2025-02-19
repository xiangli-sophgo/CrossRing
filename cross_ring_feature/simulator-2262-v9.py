import numpy as np
from collections import deque
from optimal_placement import create_adjacency_matrix
from optimal_placement import find_shortest_paths
import matplotlib.pyplot as plt
import random


class SimulationConfig:
    def __init__(
        self,
        num_nodes,
        rows,
        num_cycles_total,
        num_cycles_send,
        num_round_cycles,
        ddr_send_rate,
        sdma_send_rate,
        l2m_send_rate,
        gdma_send_rate,
        num_ddr,
        num_sdma,
        num_l2m,
        num_gdma,
        ddr_send_positions,
        ft_count,
        ft_len,
        tags_num,
        seats_per_vstation,
        sdma_send_positions,
        l2m_send_positions,
        gdma_send_positions,
        wait_cycle_h,
        wait_cycle_v,
        reservation_num,
        packet_size,
        flit_size,
        seats_per_link,
        seats_per_station,
        inject_queues_len,
        eject_queues_len,
        ip_eject_len,
        rn_read_tracker_len,
        rn_write_tracker_len,
        rn_rdb_len,
        rn_wdb_len,
        sn_wdb_len,
        ro_trak_len,
        share_trak_len,
        network_frequency,
        ddr_latency,
    ):
        self.num_nodes = num_nodes
        self.rows = rows
        self.num_cycles_total = num_cycles_total
        self.num_cycles_send = num_cycles_send
        self.num_round_cycles = num_round_cycles
        self.ddr_send_rate = ddr_send_rate
        self.sdma_send_rate = sdma_send_rate
        self.l2m_send_rate = l2m_send_rate
        self.gdma_send_rate = gdma_send_rate
        self.num_ddr = num_ddr
        self.num_sdma = num_sdma
        self.num_l2m = num_l2m
        self.num_gdma = num_gdma
        self.ddr_send_positions = ddr_send_positions
        self.sdma_send_positions = sdma_send_positions
        self.l2m_send_positions = l2m_send_positions
        self.gdma_send_positions = gdma_send_positions
        self.packet_size = packet_size
        self.flit_size = flit_size
        self.seats_per_link = seats_per_link
        self.seats_per_station = seats_per_station
        self.seats_per_vstation = seats_per_vstation
        self.inject_queues_len = inject_queues_len
        self.eject_queues_len = eject_queues_len
        self.ip_eject_len = ip_eject_len
        self.wait_cycle_h = wait_cycle_h
        self.wait_cycle_v = wait_cycle_v
        self.ft_count = ft_count
        self.ft_len = ft_len
        self.tags_num = tags_num
        self.reservation_num = reservation_num
        self.rn_read_tracker_len = rn_read_tracker_len
        self.rn_write_tracker_len = rn_write_tracker_len
        self.rn_rdb_len = rn_rdb_len
        self.rn_wdb_len = rn_wdb_len
        self.sn_wdb_len = sn_wdb_len
        self.ro_trak_len = ro_trak_len
        self.share_trak_len = share_trak_len
        self.network_frequency = network_frequency
        self.ddr_latency = ddr_latency


class DestinationQueueManager:
    def __init__(self, config):
        self.config = config
        self.queues = {"ddr": {}, "sdma": {}, "l2m": {}, "gdma": {}}

        for ip_type in ["ddr", "sdma", "l2m", "gdma"]:
            for ip_index in range(getattr(self.config, f"num_{ip_type}")):
                self.queues[ip_type][ip_index] = []

    def refill_queue(self, ip_type, ip_index):
        num_repeats = 1
        destination_type = self.get_destination_type(ip_type)
        send_positions = self.config.ddr_send_positions
        destination_positions = [x - self.config.rows for x in send_positions]
        # random_positions = np.random.permutation(destination_positions)
        random_positions = destination_positions
        queue = []
        for pos in random_positions:
            queue.extend([pos] * num_repeats)
        self.queues[ip_type][ip_index] = queue

    def get_destination_type(self, ip_type):
        if ip_type == "gdma":
            return "ddr1"
        elif ip_type == "sdma":
            return "ddr1"

    def get_destination(self, ip_type, ip_index):
        if not self.queues[ip_type][ip_index]:
            self.refill_queue(ip_type, ip_index)
        return self.queues[ip_type][ip_index].pop(0)


class Flit:
    last_id = 0

    def __init__(self, config, source, destination, path):
        self.flit_type = "flit"
        self.req_type = None
        self.req_attr = "new"
        self.early_rsp = False
        self.rn_tracker_type = None
        self.sn_tracker_type = None
        self.req_state = "valid"
        self.rsp_type = None
        # self.id = None
        self.id = Flit.last_id
        Flit.last_id += 1
        self.packet_id = None
        self.config = config
        self.source = source
        self.destination = destination
        self.source_type = None
        self.destination_type = None
        self.path = path
        self.current_position = None
        self.station_position = None
        self.path_index = 0
        self.current_seat_index = -1
        self.current_link = None
        self.departure_cycle = None
        self.departure_network_cycle = None
        self.departure_inject_cycle = None
        self.arrival_cycle = None
        self.arrival_network_cycle = None
        self.arrival_eject_cycle = None
        self.is_injected = False
        self.is_new_on_network = True
        self.is_on_station = False
        self.is_delay = False
        self.is_arrive = False
        self.predicted_duration = 0
        self.actual_duration = 0
        self.actual_ject_duration = 0
        self.actual_network_duration = 0
        self.circuits_completed_v = 0
        self.circuits_completed_h = 0
        self.wait_cycle = 0
        self.wait_cycle_v = 0
        self.is_tag_v = False
        self.is_tag_h = False

        self.moving_direction = 0  # Initial direction, 1 represents right, -1 represents left
        if self.path[1] - self.path[0] == 1:
            self.moving_direction = 1
        elif self.path[1] - self.path[0] == -1:
            self.moving_direction = -1
        self.moving_direction_v = 0  # Initial direction, 1 represents right, -1 represents left
        if self.source < self.destination:
            self.moving_direction_v = 1
        else:
            self.moving_direction_v = -1

    def inject(self, network):
        if self.path_index == 0 and not self.is_injected:
            next_position = self.path[self.path_index + 1]
            if network.can_move_to_next(self, self.source, next_position):
                self.current_position = self.source
                self.is_injected = True
                return True
        return False


class Node:
    global_packet_id = -1

    def __init__(self, config):
        self.config = config
        self.rn_rdb = {"sdma": {}, "gdma": {}}
        self.rn_rdb_reserve = {"sdma": {}, "gdma": {}}
        self.rn_rdb_recv = {"sdma": {}, "gdma": {}}
        self.rn_rdb_count = {"sdma": {}, "gdma": {}}
        self.rn_wdb = {"sdma": {}, "gdma": {}}
        self.rn_wdb_reserve = {"sdma": {}, "gdma": {}}
        self.rn_wdb_count = {"sdma": {}, "gdma": {}}
        self.rn_wdb_send = {"sdma": {}, "gdma": {}}
        self.rn_tracker = {"read": {"sdma": {}, "gdma": {}}, "write": {"sdma": {}, "gdma": {}}}
        self.rn_tracker_wait = {"read": {"sdma": {}, "gdma": {}}, "write": {"sdma": {}, "gdma": {}}}
        self.rn_tracker_count = {"read": {"sdma": {}, "gdma": {}}, "write": {"sdma": {}, "gdma": {}}}
        self.rn_tracker_pointer = {"read": {"sdma": {}, "gdma": {}}, "write": {"sdma": {}, "gdma": {}}}
        self.sn_rdb = {"ddr1": {}, "ddr2": {}}
        self.sn_rsp_queue = {"ddr1": {}, "ddr2": {}}
        self.sn_req_wait = {"read": {"ddr1": {}, "ddr2": {}}, "write": {"ddr1": {}, "ddr2": {}}}
        self.sn_tracker = {"ddr1": {}, "ddr2": {}}
        self.sn_tracker_count = {"ddr1": {"ro": {}, "share": {}}, "ddr2": {"ro": {}, "share": {}}}
        self.sn_wdb = {"ddr1": {}, "ddr2": {}}
        self.sn_wdb_recv = {"ddr1": {}, "ddr2": {}}
        self.sn_wdb_count = {"ddr1": {}, "ddr2": {}}

        for ip_type in ["sdma", "gdma"]:
            for ip_pos in getattr(config, f"{ip_type}_send_positions"):
                self.rn_rdb[ip_type][ip_pos] = {}
                self.rn_rdb_recv[ip_type][ip_pos] = []
                self.rn_rdb_count[ip_type][ip_pos] = config.rn_rdb_len
                self.rn_rdb_reserve[ip_type][ip_pos] = 0
                self.rn_wdb[ip_type][ip_pos] = {}
                self.rn_wdb_count[ip_type][ip_pos] = config.rn_wdb_len
                self.rn_wdb_send[ip_type][ip_pos] = []
                self.rn_wdb_reserve[ip_type][ip_pos] = 0
                for req_type in ["read", "write"]:
                    self.rn_tracker[req_type][ip_type][ip_pos] = []
                    self.rn_tracker_wait[req_type][ip_type][ip_pos] = []
                    self.rn_tracker_count[req_type][ip_type][ip_pos] = (
                        config.rn_read_tracker_len if req_type == "read" else config.rn_write_tracker_len
                    )
                    self.rn_tracker_pointer[req_type][ip_type][ip_pos] = -1

        for ip_pos in config.ddr_send_positions:
            for key in self.sn_tracker:
                self.sn_rdb[key][ip_pos] = []
                self.sn_rsp_queue[key][ip_pos] = []
                for req_type in ["read", "write"]:
                    self.sn_req_wait[req_type][key][ip_pos] = []
                self.sn_wdb[key][ip_pos] = {}
                self.sn_wdb_recv[key][ip_pos] = []
                self.sn_wdb_count[key][ip_pos] = config.sn_wdb_len
                self.sn_tracker[key][ip_pos] = []
                self.sn_tracker_count[key]["ro"][ip_pos] = config.ro_trak_len
                self.sn_tracker_count[key]["share"][ip_pos] = config.share_trak_len

    @classmethod
    def get_next_packet_id(cls):
        cls.global_packet_id += 1
        return cls.global_packet_id


class Network:
    def __init__(self, config, adjacency_matrix):
        self.config = config
        self.current_cycle = []
        self.flits_num = []
        self.schedules = {"sdma": None}
        self.inject_num = 0
        self.eject_num = 0
        self.inject_queues_left = {}
        self.inject_queues_right = {}
        self.inject_queues_up = {}
        self.inject_queues_local = {}
        self.inject_queues_left_pre = {}
        self.inject_queues_right_pre = {}
        self.inject_queues_up_pre = {}
        self.inject_queues_local_pre = {}
        self.eject_queues_pre = {"ddr": {}, "ddr1": {}, "ddr2": {}, "sdma": {}, "gdma": {}, "l2m": {}}
        self.eject_queues_up = {}
        self.eject_queues_down = {}
        self.eject_queues_mid = {}
        self.eject_queues_local = {}
        self.eject_reservations_down = {}
        self.eject_reservations_up = {}
        self.arrive_node_pre = {"ddr1": {}, "ddr2": {}, "sdma": {}, "gdma": {}}
        self.ip_inject = {"ddr": {}, "ddr1": {}, "ddr2": {}, "sdma": {}, "l2m": {}, "gdma": {}}
        self.ip_eject = {"ddr": {}, "ddr1": {}, "ddr2": {}, "sdma": {}, "l2m": {}, "gdma": {}}
        self.ip_read = {"sdma": {}, "gdma": {}}
        self.ip_write = {"sdma": {}, "gdma": {}}
        self.links = {}
        self.links_tag = {}
        self.remain_tag = {"left": {}, "right": {}, "up": {}, "down": {}}
        self.transfer_stations_left = {}
        self.transfer_stations_right = {}
        self.transfer_stations_up = {}
        self.transfer_stations_ft = {}
        self.transfer_stations_vup = {}
        self.transfer_stations_vdown = {}
        self.transfer_stations_eject = {}
        self.station_reservations_left = {}
        self.station_reservations_right = {}
        self.inject_queue_rr = {"left": {0: {}, 1: {}}, "right": {0: {}, 1: {}}, "up": {0: {}, 1: {}}, "local": {0: {}, 1: {}}}
        self.inject_right_rr = {}
        self.inject_left_rr = {}
        self.inject_up_rr = {}
        self.inject_local_rr = {}
        self.ddr_rr = {}
        self.ddr1_rr = {}
        self.ddr2_rr = {}
        self.sdma_rr = {}
        self.l2m_rr = {}
        self.gdma_rr = {}
        self.up_rr = {}
        self.down_rr = {}
        self.mid_rr = {}
        self.round_robin_counter = 0
        self.recv_flits_num = 0
        self.send_flits = []
        self.arrive_flits = []
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
        self.circuits_h = []
        self.circuits_v = []
        self.avg_circuits_h = None
        self.max_circuits_h = None
        self.avg_circuits_v = None
        self.max_circuits_v = None
        self.circuits_flit_h = {"ddr": 0, "ddr1": 0, "ddr2": 0, "sdma": 0, "l2m": 0, "gdma": 0}
        self.circuits_flit_v = {"ddr": 0, "ddr1": 0, "ddr2": 0, "sdma": 0, "l2m": 0, "gdma": 0}
        self.gdma_recv = 0
        self.gdma_remainder = 0
        self.gdma_count = 512
        self.l2m_recv = 0
        self.l2m_remainder = 0
        self.sdma_send = []
        self.num_send = {"ddr": {}, "ddr1": {}, "ddr2": {}, "sdma": {}, "l2m": {}, "gdma": {}}
        self.num_recv = {"ddr": {}, "ddr1": {}, "ddr2": {}, "sdma": {}, "l2m": {}, "gdma": {}}
        self.per_send_throughput = {"ddr": {}, "ddr1": {}, "ddr2": {}, "sdma": {}, "l2m": {}, "gdma": {}}
        self.per_recv_throughput = {"ddr": {}, "ddr1": {}, "ddr2": {}, "sdma": {}, "l2m": {}, "gdma": {}}
        self.send_throughput = {"ddr": 0, "ddr1": 0, "ddr2": 0, "sdma": 0, "l2m": 0, "gdma": 0}
        self.recv_throughput = {"ddr": 0, "ddr1": 0, "ddr2": 0, "sdma": 0, "l2m": 0, "gdma": 0}
        self.last_select = {"sdma": {}, "gdma": {}}
        self.throughput = {"sdma": {}, "ddr": {}, "ddr1": {}, "gdma": {}, "l2m": {}}

        for ip_pos in set(config.ddr_send_positions + config.sdma_send_positions + config.l2m_send_positions + config.gdma_send_positions):
            self.inject_queues_left[ip_pos] = deque(maxlen=config.inject_queues_len)
            self.inject_queues_right[ip_pos] = deque(maxlen=config.inject_queues_len)
            self.inject_queues_up[ip_pos] = deque(maxlen=config.inject_queues_len)
            self.inject_queues_local[ip_pos] = deque(maxlen=config.inject_queues_len)
            self.inject_queues_left_pre[ip_pos] = None
            self.inject_queues_right_pre[ip_pos] = None
            self.inject_queues_up_pre[ip_pos] = None
            self.inject_queues_local_pre[ip_pos] = None
            for key in self.eject_queues_pre:
                self.eject_queues_pre[key][ip_pos - config.rows] = None
            for key in self.arrive_node_pre:
                self.arrive_node_pre[key][ip_pos - config.rows] = None
            self.eject_queues_up[ip_pos - config.rows] = deque(maxlen=config.eject_queues_len)
            self.eject_queues_down[ip_pos - config.rows] = deque(maxlen=config.eject_queues_len)
            self.eject_queues_mid[ip_pos - config.rows] = deque(maxlen=config.eject_queues_len)
            self.eject_queues_local[ip_pos - config.rows] = deque(maxlen=config.eject_queues_len)
            self.eject_reservations_down[ip_pos - config.rows] = deque(maxlen=config.reservation_num)
            self.eject_reservations_up[ip_pos - config.rows] = deque(maxlen=config.reservation_num)
            for key in self.inject_queue_rr:
                self.inject_queue_rr[key][0][ip_pos] = deque([0, 1])
                self.inject_queue_rr[key][1][ip_pos] = deque([0, 1])
            self.inject_right_rr[ip_pos] = deque([0, 1, 2])
            self.inject_left_rr[ip_pos] = deque([0, 1, 2])
            self.inject_up_rr[ip_pos] = deque([0, 1, 2])
            self.inject_local_rr[ip_pos] = deque([0, 1, 2])
            self.ddr_rr[ip_pos - config.rows] = deque([0, 1, 2, 3])
            self.ddr1_rr[ip_pos - config.rows] = deque([0, 1, 2, 3])
            self.ddr2_rr[ip_pos - config.rows] = deque([0, 1, 2, 3])
            self.sdma_rr[ip_pos - config.rows] = deque([0, 1, 2, 3])
            self.l2m_rr[ip_pos - config.rows] = deque([0, 1, 2, 3])
            self.gdma_rr[ip_pos - config.rows] = deque([0, 1, 2, 3])
            self.inject_time[ip_pos] = []
            self.eject_time[ip_pos - config.rows] = []
            self.avg_inject_time[ip_pos] = 0
            self.avg_eject_time[ip_pos - config.rows] = 1

        for i in range(config.num_nodes):
            for j in range(config.num_nodes):
                if adjacency_matrix[i][j] == 1 and i - j != config.rows:
                    self.links[(i, j)] = [None] * config.seats_per_link
                    self.links_tag[(i, j)] = [None] * config.seats_per_link
            if i in range(0, config.rows):
                self.links[(i, i)] = [None] * 2
                self.links[(i + config.num_nodes - config.rows * 2, i + config.num_nodes - config.rows * 2)] = [None] * 2
                self.links_tag[(i, i)] = [None] * 2
                self.links_tag[(i + config.num_nodes - config.rows * 2, i + config.num_nodes - config.rows * 2)] = [None] * 2
            if i % config.rows == 0 and (i // config.rows) % 2 != 0:
                self.links[(i, i)] = [None] * 2
                self.links[(i + config.rows - 1, i + config.rows - 1)] = [None] * 2
                self.links_tag[(i, i)] = [None] * 2
                self.links_tag[(i + config.rows - 1, i + config.rows - 1)] = [None] * 2

        for row in range(1, config.num_nodes // config.rows, 2):
            for col in range(config.rows):
                pos = row * config.rows + col
                next_pos = pos - config.rows
                self.transfer_stations_left[(pos, next_pos)] = deque(maxlen=config.seats_per_station)
                self.transfer_stations_right[(pos, next_pos)] = deque(maxlen=config.seats_per_station)
                self.transfer_stations_up[(pos, next_pos)] = deque(maxlen=config.seats_per_station)
                self.transfer_stations_ft[(pos, next_pos)] = deque(maxlen=config.ft_len)
                self.transfer_stations_vup[(pos, next_pos)] = deque(maxlen=config.seats_per_vstation)
                self.transfer_stations_vdown[(pos, next_pos)] = deque(maxlen=config.seats_per_vstation)
                self.transfer_stations_eject[(pos, next_pos)] = deque(maxlen=config.seats_per_vstation)
                self.station_reservations_left[(pos, next_pos)] = deque(maxlen=config.reservation_num)
                self.station_reservations_right[(pos, next_pos)] = deque(maxlen=config.reservation_num)
                self.up_rr[next_pos] = deque([0, 1, 2])
                self.down_rr[next_pos] = deque([0, 1, 2])
                self.mid_rr[next_pos] = deque([0, 1, 2])
                for direction in ["left", "right"]:
                    self.remain_tag[direction][pos] = config.tags_num
                for direction in ["up", "down"]:
                    self.remain_tag[direction][next_pos] = config.tags_num

        for ip_type in self.num_recv:
            if ip_type == "ddr1" or ip_type == "ddr2":
                ip_type = "ddr"
            source_positions = getattr(config, f"{ip_type}_send_positions")
            for source in source_positions:
                destination = source - config.rows
                self.num_send[ip_type][source] = 0
                self.num_recv[ip_type][destination] = 0
                self.per_send_throughput[ip_type][source] = 0
                self.per_recv_throughput[ip_type][destination] = 0

        for ip_type in ["ddr", "ddr1", "ddr2", "sdma", "l2m", "gdma"]:
            ip = "ddr" if ip_type == "ddr1" or ip_type == "ddr2" else ip_type
            for ip_index in getattr(config, f"{ip}_send_positions"):
                ip_recv_index = ip_index - config.rows
                self.ip_inject[ip_type][ip_index] = deque()
                self.ip_eject[ip_type][ip_recv_index] = deque(maxlen=config.ip_eject_len)
        for ip_type in ["sdma", "gdma"]:
            for ip_index in getattr(config, f"{ip_type}_send_positions"):
                self.ip_read[ip_type][ip_index] = deque()
                self.ip_write[ip_type][ip_index] = deque()
                self.last_select[ip_type][ip_index] = "write"
        for ip_type in ["sdma", "ddr1", "ddr", "gdma"]:
            ip = "ddr" if ip_type == "ddr1" or ip_type == "ddr2" else ip_type
            for ip_index in getattr(config, f"{ip}_send_positions"):
                self.throughput[ip_type][ip_index] = [0, 0, 10000000, 0]

    def generate_flit_schedule(self, ip_type, num_round_cycles, reqs_per_round, num_ips, cycle):
        schedule = np.zeros((num_ips, num_round_cycles), dtype=bool)
        for ip in range(num_ips):
            cycles = np.random.choice(np.arange(0, num_round_cycles), size=reqs_per_round, replace=False)
            schedule[ip, cycles] = True
            # schedule[ip, ip] = False
        return schedule

    def can_move_to_next(self, flit, current, next_node):
        if flit.source - flit.destination == self.config.rows:
            return len(self.eject_queues_local[flit.destination]) < self.config.eject_queues_len
        current_column_index = current % self.config.rows
        if current_column_index == 0:
            if next_node == current + 1:
                if self.links[(current, current)][-1] is not None:
                    if self.links_tag[(current, current)][-1] is None:
                        flit_l = self.links[(current, current)][-1]
                        if flit_l.current_link[1] == flit_l.current_position:
                            flit_exist_right = any(
                                flit_r.id == flit_l.id
                                for flit_r in self.station_reservations_right[(flit_l.current_link[1], flit_l.path[flit_l.path_index])]
                            )
                            link_station = self.transfer_stations_right.get((flit_l.current_link[1], flit_l.path[flit_l.path_index]))
                            if len(link_station) < self.config.seats_per_station and flit_exist_right:
                                return True
                        if flit.wait_cycle > self.config.wait_cycle_h and not flit.is_tag_h:
                            if self.remain_tag["right"][current] > 0:
                                self.remain_tag["right"][current] -= 1
                                self.links_tag[(current, current)][-1] = [current, "right"]
                                flit.is_tag_h = True
                    else:
                        flit_l = self.links[(current, current)][-1]
                        if flit_l.current_link[1] == flit_l.current_position:
                            flit_exist_right = any(
                                flit_r.id == flit_l.id
                                for flit_r in self.station_reservations_right[(flit_l.current_link[1], flit_l.path[flit_l.path_index])]
                            )
                            link_station = self.transfer_stations_right.get((flit_l.current_link[1], flit_l.path[flit_l.path_index]))
                            if len(link_station) < self.config.seats_per_station and flit_exist_right:
                                if self.links_tag[(current, current)][-1] == [current, "right"]:
                                    self.links_tag[(current, current)][-1] = None
                                    self.remain_tag["right"][current] += 1
                                    return True
                else:
                    if self.links_tag[(current, current)][-1] is None:
                        return True
                    else:
                        if self.links_tag[(current, current)][-1] == [current, "right"]:
                            self.links_tag[(current, current)][-1] = None
                            self.remain_tag["right"][current] += 1
                            return True
                return False
        elif current_column_index == self.config.rows - 1:
            if next_node == current - 1:
                if self.links[(current, current)][-1] is not None:
                    if self.links_tag[(current, current)][-1] is None:
                        flit_l = self.links[(current, current)][-1]
                        if flit_l.current_link[1] == flit_l.current_position:
                            flit_exist_left = any(
                                flit_r.id == flit_l.id
                                for flit_r in self.station_reservations_left[(flit_l.current_link[1], flit_l.path[flit_l.path_index])]
                            )
                            link_station = self.transfer_stations_left.get((flit_l.current_link[1], flit_l.path[flit_l.path_index]))
                            if len(link_station) < self.config.seats_per_station and flit_exist_left:
                                return True
                        if flit.wait_cycle > self.config.wait_cycle_h and not flit.is_tag_h:
                            if self.remain_tag["left"][current] > 0:
                                self.remain_tag["left"][current] -= 1
                                self.links_tag[(current, current)][-1] = [current, "left"]
                                flit.is_tag_h = True
                    else:
                        flit_l = self.links[(current, current)][-1]
                        if flit_l.current_link[1] == flit_l.current_position:
                            flit_exist_left = any(
                                flit_r.id == flit_l.id
                                for flit_r in self.station_reservations_left[(flit_l.current_link[1], flit_l.path[flit_l.path_index])]
                            )
                            link_station = self.transfer_stations_left.get((flit_l.current_link[1], flit_l.path[flit_l.path_index]))
                            if len(link_station) < self.config.seats_per_station and flit_exist_left:
                                if self.links_tag[(current, current)][-1] == [current, "left"]:
                                    self.links_tag[(current, current)][-1] = None
                                    self.remain_tag["left"][current] += 1
                                    return True
                else:
                    if self.links_tag[(current, current)][-1] is None:
                        return True
                    else:
                        if self.links_tag[(current, current)][-1] == [current, "left"]:
                            self.links_tag[(current, current)][-1] = None
                            self.remain_tag["left"][current] += 1
                            return True
                return False

        if current - next_node == self.config.rows:
            return len(self.transfer_stations_up[(current, next_node)]) < self.config.seats_per_station
        elif next_node - current == 1:
            if self.links[(current - 1, current)][-1] is not None:
                if self.links_tag[(current - 1, current)][-1] is None:
                    flit_l = self.links[(current - 1, current)][-1]
                    if flit_l.path_index + 1 < len(flit_l.path) and flit_l.current_link[1] - flit_l.path[flit_l.path_index + 1] == self.config.rows:
                        new_current = flit_l.current_link[1]
                        new_next_node = flit_l.path[flit_l.path_index + 1]
                        station_right = self.transfer_stations_right.get((new_current, new_next_node))
                        if self.config.seats_per_station - len(station_right) > len(self.station_reservations_right[(new_current, new_next_node)]):
                            return True
                    if flit.wait_cycle > self.config.wait_cycle_h and not flit.is_tag_h:
                        if self.remain_tag["right"][current] > 0:
                            self.remain_tag["right"][current] -= 1
                            self.links_tag[(current - 1, current)][-1] = [current, "right"]
                            flit.is_tag_h = True
                else:
                    flit_l = self.links[(current - 1, current)][-1]
                    if flit_l.path_index + 1 < len(flit_l.path) and flit_l.current_link[1] - flit_l.path[flit_l.path_index + 1] == self.config.rows:
                        new_current = flit_l.current_link[1]
                        new_next_node = flit_l.path[flit_l.path_index + 1]
                        station_right = self.transfer_stations_right.get((new_current, new_next_node))
                        if self.config.seats_per_station - len(station_right) > len(self.station_reservations_right[(new_current, new_next_node)]):
                            if self.links_tag[(current - 1, current)][-1] == [current, "right"]:
                                self.links_tag[(current - 1, current)][-1] = None
                                self.remain_tag["right"][current] += 1
                                return True
            else:
                if self.links_tag[(current - 1, current)][-1] is None:
                    return True
                else:
                    if self.links_tag[(current - 1, current)][-1] == [current, "right"]:
                        self.links_tag[(current - 1, current)][-1] = None
                        self.remain_tag["right"][current] += 1
                        return True
            return False
        elif current - next_node == 1:
            if self.links[(current + 1, current)][-1] is not None:
                if self.links_tag[current + 1, current][-1] is None:
                    flit_l = self.links[(current + 1, current)][-1]
                    if flit_l.path_index + 1 < len(flit_l.path) and flit_l.current_link[1] - flit_l.path[flit_l.path_index + 1] == self.config.rows:
                        new_current = flit_l.current_link[1]
                        new_next_node = flit_l.path[flit_l.path_index + 1]
                        station_left = self.transfer_stations_left.get((new_current, new_next_node))
                        if self.config.seats_per_station - len(station_left) > len(self.station_reservations_left[(new_current, new_next_node)]):
                            return True
                    if flit.wait_cycle > self.config.wait_cycle_h and not flit.is_tag_h:
                        if self.remain_tag["left"][current] > 0:
                            self.remain_tag["left"][current] -= 1
                            self.links_tag[(current + 1, current)][-1] = [current, "left"]
                            flit.is_tag_h = True
                else:
                    flit_l = self.links[(current + 1, current)][-1]
                    if flit_l.path_index + 1 < len(flit_l.path) and flit_l.current_link[1] - flit_l.path[flit_l.path_index + 1] == self.config.rows:
                        new_current = flit_l.current_link[1]
                        new_next_node = flit_l.path[flit_l.path_index + 1]
                        station_left = self.transfer_stations_left.get((new_current, new_next_node))
                        if self.config.seats_per_station - len(station_left) > len(self.station_reservations_left[(new_current, new_next_node)]):
                            if self.links_tag[(current + 1, current)][-1] == [current, "left"]:
                                self.links_tag[(current + 1, current)][-1] = None
                                self.remain_tag["left"][current] += 1
                                return True
            else:
                if self.links_tag[(current + 1, current)][-1] is None:
                    return True
                else:
                    if self.links_tag[(current + 1, current)][-1] == [current, "left"]:
                        self.links_tag[(current + 1, current)][-1] = None
                        self.remain_tag["left"][current] += 1
                        return True
            return False

    def plan_move(self, flit):
        if flit.is_new_on_network:
            current, next_node = flit.source, flit.path[flit.path_index + 1]
            flit.current_position = current
            flit.is_new_on_network = False
            if current - next_node == self.config.rows:
                flit.path_index += 1
                flit.current_link = (current, next_node)
                flit.current_seat_index = 2
            else:
                flit.path_index += 1
                flit.current_link = (current, next_node)
                flit.current_seat_index = 0
            return
        current, next_node = flit.current_link
        row_start = (current // self.config.rows) * self.config.rows
        row_start = row_start if row_start % self.config.rows == 0 and (row_start // self.config.rows) % 2 != 0 else -1
        row_end = row_start + self.config.rows - 1 if row_start > 0 else -1
        col_start = current % (self.config.rows * 2) if current % (self.config.rows * 2) in range(0, self.config.rows) else -1
        col_end = col_start + self.config.num_nodes - self.config.rows * 2 if col_start >= 0 else -1
        if current - next_node != self.config.rows:
            # Handling delay flits
            if flit.is_delay:
                link = self.links.get(flit.current_link)
                if flit.current_seat_index != len(link) - 1:
                    link[flit.current_seat_index] = None
                    flit.current_seat_index += 1
                else:
                    if current == next_node:
                        if current == row_start:
                            if current == flit.current_position:
                                flit.circuits_completed_h += 1
                                flit_exist_left = any(
                                    flit_r.id == flit.id for flit_r in self.station_reservations_left[(next_node, flit.path[flit.path_index])]
                                )
                                flit_exist_right = any(
                                    flit_r.id == flit.id for flit_r in self.station_reservations_right[(next_node, flit.path[flit.path_index])]
                                )
                                link_station = self.transfer_stations_right.get((next_node, flit.path[flit.path_index]))
                                if len(link_station) < self.config.seats_per_station and flit_exist_right:
                                    flit.is_delay = False
                                    flit.current_link = (next_node, flit.path[flit.path_index])
                                    link[flit.current_seat_index] = None
                                    flit.current_seat_index = 1
                                    self.station_reservations_right[(next_node, flit.path[flit.path_index])].remove(flit)
                                elif not flit_exist_right and self.config.seats_per_station - len(link_station) > len(
                                    self.station_reservations_right[(next_node, flit.path[flit.path_index])]
                                ):
                                    flit.is_delay = False
                                    flit.current_link = (next_node, flit.path[flit.path_index])
                                    link[flit.current_seat_index] = None
                                    flit.current_seat_index = 1
                                    if flit_exist_left:
                                        self.station_reservations_left[(next_node, flit.path[flit.path_index])].remove(flit)
                                else:
                                    if not flit_exist_left and not flit_exist_right:
                                        if len(self.station_reservations_left[(next_node, flit.path[flit.path_index])]) < self.config.reservation_num:
                                            self.station_reservations_left[(next_node, flit.path[flit.path_index])].append(flit)
                                    link[flit.current_seat_index] = None
                                    next_pos = next_node + 1
                                    flit.current_link = (next_node, next_pos)
                                    flit.current_seat_index = 0
                            else:
                                link[flit.current_seat_index] = None
                                next_pos = next_node + 1
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                        elif current == row_end:
                            if current == flit.current_position:
                                flit.circuits_completed_h += 1
                                flit_exist_left = any(
                                    flit_r.id == flit.id for flit_r in self.station_reservations_left[(next_node, flit.path[flit.path_index])]
                                )
                                flit_exist_right = any(
                                    flit_r.id == flit.id for flit_r in self.station_reservations_right[(next_node, flit.path[flit.path_index])]
                                )
                                if flit.circuits_completed_h > self.config.ft_count:
                                    link_station = self.transfer_stations_ft.get((next_node, flit.path[flit.path_index]))
                                    if len(link_station) < self.config.ft_len:
                                        flit.is_delay = False
                                        flit.current_link = (next_node, flit.path[flit.path_index])
                                        link[flit.current_seat_index] = None
                                        flit.current_seat_index = -2
                                        if flit_exist_left:
                                            self.station_reservations_left[(next_node, flit.path[flit.path_index])].remove(flit)
                                        elif flit_exist_right:
                                            self.station_reservations_right[(next_node, flit.path[flit.path_index])].remove(flit)
                                    else:
                                        if not flit_exist_left and not flit_exist_right:
                                            if (
                                                len(self.station_reservations_right[(next_node, flit.path[flit.path_index])])
                                                < self.config.reservation_num
                                            ):
                                                self.station_reservations_right[(next_node, flit.path[flit.path_index])].append(flit)
                                        link[flit.current_seat_index] = None
                                        next_pos = next_node - 1
                                        flit.current_link = (next_node, next_pos)
                                        flit.current_seat_index = 0
                                else:
                                    link_station = self.transfer_stations_left.get((next_node, flit.path[flit.path_index]))
                                    if len(link_station) < self.config.seats_per_station and flit_exist_left:
                                        flit.is_delay = False
                                        flit.current_link = (next_node, flit.path[flit.path_index])
                                        link[flit.current_seat_index] = None
                                        flit.current_seat_index = 0
                                        self.station_reservations_left[(next_node, flit.path[flit.path_index])].remove(flit)
                                    elif not flit_exist_left and self.config.seats_per_station - len(link_station) > len(
                                        self.station_reservations_left[(next_node, flit.path[flit.path_index])]
                                    ):
                                        flit.is_delay = False
                                        flit.current_link = (next_node, flit.path[flit.path_index])
                                        link[flit.current_seat_index] = None
                                        flit.current_seat_index = 0
                                        if flit_exist_right:
                                            self.station_reservations_right[(next_node, flit.path[flit.path_index])].remove(flit)
                                    else:
                                        if not flit_exist_left and not flit_exist_right:
                                            if (
                                                len(self.station_reservations_right[(next_node, flit.path[flit.path_index])])
                                                < self.config.reservation_num
                                            ):
                                                self.station_reservations_right[(next_node, flit.path[flit.path_index])].append(flit)
                                        link[flit.current_seat_index] = None
                                        next_pos = next_node - 1
                                        flit.current_link = (next_node, next_pos)
                                        flit.current_seat_index = 0
                            else:
                                link[flit.current_seat_index] = None
                                next_pos = next_node - 1
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                        elif current == col_start:
                            if next_node == flit.destination:
                                flit.circuits_completed_v += 1
                                flit_exist_up = any(flit_r.id == flit.id for flit_r in self.eject_reservations_up[next_node])
                                flit_exist_down = any(flit_r.id == flit.id for flit_r in self.eject_reservations_down[next_node])
                                link_eject = self.eject_queues_down[next_node]
                                if len(link_eject) < self.config.eject_queues_len and flit_exist_down:
                                    flit.is_delay = False
                                    flit.is_arrive = True
                                    link[flit.current_seat_index] = None
                                    flit.current_seat_index = 0
                                    self.eject_reservations_down[next_node].remove(flit)
                                elif not flit_exist_down and self.config.eject_queues_len - len(link_eject) > len(
                                    self.eject_reservations_down[next_node]
                                ):
                                    flit.is_delay = False
                                    flit.is_arrive = True
                                    link[flit.current_seat_index] = None
                                    flit.current_seat_index = 0
                                    if flit_exist_up:
                                        self.eject_reservations_up[next_node].remove(flit)
                                else:
                                    if not flit_exist_up and not flit_exist_down:
                                        if len(self.eject_reservations_up[next_node]) < self.config.reservation_num:
                                            self.eject_reservations_up[next_node].append(flit)
                                    link[flit.current_seat_index] = None
                                    next_pos = next_node + self.config.rows * 2
                                    flit.current_link = (next_node, next_pos)
                                    flit.current_seat_index = 0
                            else:
                                link[flit.current_seat_index] = None
                                next_pos = next_node + self.config.rows * 2
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                        elif current == col_end:
                            if next_node == flit.destination:
                                flit.circuits_completed_v += 1
                                flit_exist_up = any(flit_r.id == flit.id for flit_r in self.eject_reservations_up[next_node])
                                flit_exist_down = any(flit_r.id == flit.id for flit_r in self.eject_reservations_down[next_node])
                                link_eject = self.eject_queues_up[next_node]
                                if len(link_eject) < self.config.eject_queues_len and flit_exist_up:
                                    flit.is_delay = False
                                    flit.is_arrive = True
                                    link[flit.current_seat_index] = None
                                    flit.current_seat_index = 0
                                    self.eject_reservations_up[next_node].remove(flit)
                                elif not flit_exist_up and self.config.eject_queues_len - len(link_eject) > len(
                                    self.eject_reservations_up[next_node]
                                ):
                                    flit.is_delay = False
                                    flit.is_arrive = True
                                    link[flit.current_seat_index] = None
                                    flit.current_seat_index = 0
                                    if flit_exist_down:
                                        self.eject_reservations_down[next_node].remove(flit)
                                else:
                                    if not flit_exist_up and not flit_exist_down:
                                        if len(self.eject_reservations_down[next_node]) < self.config.reservation_num:
                                            self.eject_reservations_down[next_node].append(flit)
                                    link[flit.current_seat_index] = None
                                    next_pos = next_node - self.config.rows * 2
                                    flit.current_link = (next_node, next_pos)
                                    flit.current_seat_index = 0
                            else:
                                link[flit.current_seat_index] = None
                                next_pos = next_node - self.config.rows * 2
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                    elif current - next_node == 1 or current - next_node == -1:
                        if next_node == flit.current_position:
                            flit.circuits_completed_h += 1
                            flit_exist_left = any(
                                flit_r.id == flit.id for flit_r in self.station_reservations_left[(next_node, flit.path[flit.path_index])]
                            )
                            flit_exist_right = any(
                                flit_r.id == flit.id for flit_r in self.station_reservations_right[(next_node, flit.path[flit.path_index])]
                            )
                            if flit.circuits_completed_h > self.config.ft_count and current - next_node == 1:
                                link_station = self.transfer_stations_ft.get((next_node, flit.path[flit.path_index]))
                                if len(link_station) < self.config.ft_len:
                                    flit.is_delay = False
                                    flit.current_link = (next_node, flit.path[flit.path_index])
                                    link[flit.current_seat_index] = None
                                    flit.current_seat_index = -2
                                    if flit_exist_left:
                                        self.station_reservations_left[(next_node, flit.path[flit.path_index])].remove(flit)
                                    elif flit_exist_right:
                                        self.station_reservations_right[(next_node, flit.path[flit.path_index])].remove(flit)
                                else:
                                    if not flit_exist_left and not flit_exist_right:
                                        if (
                                            len(self.station_reservations_right[(next_node, flit.path[flit.path_index])])
                                            < self.config.reservation_num
                                        ):
                                            self.station_reservations_right[(next_node, flit.path[flit.path_index])].append(flit)
                                    link[flit.current_seat_index] = None
                                    if current - next_node == 1:
                                        next_pos = next_node - 1 if next_node - 1 >= row_start else row_start
                                    else:
                                        next_pos = next_node + 1 if next_node + 1 <= row_end else row_end
                                    flit.current_link = (next_node, next_pos)
                                    flit.current_seat_index = 0
                            else:
                                if current - next_node == 1:
                                    link_station = self.transfer_stations_left.get((next_node, flit.path[flit.path_index]))
                                    if len(link_station) < self.config.seats_per_station and flit_exist_left:
                                        flit.is_delay = False
                                        flit.current_link = (next_node, flit.path[flit.path_index])
                                        link[flit.current_seat_index] = None
                                        flit.current_seat_index = 0
                                        self.station_reservations_left[(next_node, flit.path[flit.path_index])].remove(flit)
                                    elif not flit_exist_left and self.config.seats_per_station - len(link_station) > len(
                                        self.station_reservations_left[(next_node, flit.path[flit.path_index])]
                                    ):
                                        flit.is_delay = False
                                        flit.current_link = (next_node, flit.path[flit.path_index])
                                        link[flit.current_seat_index] = None
                                        flit.current_seat_index = 0
                                        if flit_exist_right:
                                            self.station_reservations_right[(next_node, flit.path[flit.path_index])].remove(flit)
                                    else:
                                        if not flit_exist_left and not flit_exist_right:
                                            if (
                                                len(self.station_reservations_right[(next_node, flit.path[flit.path_index])])
                                                < self.config.reservation_num
                                            ):
                                                self.station_reservations_right[(next_node, flit.path[flit.path_index])].append(flit)
                                        link[flit.current_seat_index] = None
                                        next_pos = next_node - 1 if next_node - 1 >= row_start else row_start
                                        flit.current_link = (next_node, next_pos)
                                        flit.current_seat_index = 0
                                else:
                                    link_station = self.transfer_stations_right.get((next_node, flit.path[flit.path_index]))
                                    if len(link_station) < self.config.seats_per_station and flit_exist_right:
                                        flit.is_delay = False
                                        flit.current_link = (next_node, flit.path[flit.path_index])
                                        link[flit.current_seat_index] = None
                                        flit.current_seat_index = 1
                                        self.station_reservations_right[(next_node, flit.path[flit.path_index])].remove(flit)
                                    elif not flit_exist_right and self.config.seats_per_station - len(link_station) > len(
                                        self.station_reservations_right[(next_node, flit.path[flit.path_index])]
                                    ):
                                        flit.is_delay = False
                                        flit.current_link = (next_node, flit.path[flit.path_index])
                                        link[flit.current_seat_index] = None
                                        flit.current_seat_index = 1
                                        if flit_exist_left:
                                            self.station_reservations_left[(next_node, flit.path[flit.path_index])].remove(flit)
                                    else:
                                        if not flit_exist_left and not flit_exist_right:
                                            if (
                                                len(self.station_reservations_left[(next_node, flit.path[flit.path_index])])
                                                < self.config.reservation_num
                                            ):
                                                self.station_reservations_left[(next_node, flit.path[flit.path_index])].append(flit)
                                        link[flit.current_seat_index] = None
                                        next_pos = next_node + 1 if next_node + 1 <= row_end else row_end
                                        flit.current_link = (next_node, next_pos)
                                        flit.current_seat_index = 0
                        else:
                            link[flit.current_seat_index] = None
                            if current - next_node == 1:
                                next_pos = next_node - 1 if next_node - 1 >= row_start else row_start
                            else:
                                next_pos = next_node + 1 if next_node + 1 <= row_end else row_end
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                    else:
                        if next_node == flit.destination:
                            flit.circuits_completed_v += 1
                            flit_exist_up = any(flit_r.id == flit.id for flit_r in self.eject_reservations_up[next_node])
                            flit_exist_down = any(flit_r.id == flit.id for flit_r in self.eject_reservations_down[next_node])
                            if current - next_node == self.config.rows * 2:
                                link_eject = self.eject_queues_up[next_node]
                                if len(link_eject) < self.config.eject_queues_len and flit_exist_up:
                                    flit.is_delay = False
                                    flit.is_arrive = True
                                    link[flit.current_seat_index] = None
                                    flit.current_seat_index = 0
                                    self.eject_reservations_up[next_node].remove(flit)
                                elif not flit_exist_up and self.config.eject_queues_len - len(link_eject) > len(
                                    self.eject_reservations_up[next_node]
                                ):
                                    flit.is_delay = False
                                    flit.is_arrive = True
                                    link[flit.current_seat_index] = None
                                    flit.current_seat_index = 0
                                    if flit_exist_down:
                                        self.eject_reservations_down[next_node].remove(flit)
                                else:
                                    if not flit_exist_up and not flit_exist_down:
                                        if len(self.eject_reservations_down[next_node]) < self.config.reservation_num:
                                            self.eject_reservations_down[next_node].append(flit)
                                    link[flit.current_seat_index] = None
                                    next_pos = next_node - self.config.rows * 2 if next_node - self.config.rows * 2 >= col_start else col_start
                                    flit.current_link = (next_node, next_pos)
                                    flit.current_seat_index = 0
                            else:
                                link_eject = self.eject_queues_down[next_node]
                                if len(link_eject) < self.config.eject_queues_len and flit_exist_down:
                                    flit.is_delay = False
                                    flit.is_arrive = True
                                    link[flit.current_seat_index] = None
                                    flit.current_seat_index = 0
                                    self.eject_reservations_down[next_node].remove(flit)
                                elif not flit_exist_down and self.config.eject_queues_len - len(link_eject) > len(
                                    self.eject_reservations_down[next_node]
                                ):
                                    flit.is_delay = False
                                    flit.is_arrive = True
                                    link[flit.current_seat_index] = None
                                    flit.current_seat_index = 0
                                    if flit_exist_up:
                                        self.eject_reservations_up[next_node].remove(flit)
                                else:
                                    if not flit_exist_up and not flit_exist_down:
                                        if len(self.eject_reservations_up[next_node]) < self.config.reservation_num:
                                            self.eject_reservations_up[next_node].append(flit)
                                    link[flit.current_seat_index] = None
                                    next_pos = next_node + self.config.rows * 2 if next_node + self.config.rows * 2 <= col_end else col_end
                                    flit.current_link = (next_node, next_pos)
                                    flit.current_seat_index = 0
                        else:
                            link[flit.current_seat_index] = None
                            if current - next_node == self.config.rows * 2:
                                next_pos = next_node - self.config.rows * 2 if next_node - self.config.rows * 2 >= col_start else col_start
                            else:
                                next_pos = next_node + self.config.rows * 2 if next_node + self.config.rows * 2 <= col_end else col_end
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                return
            # Handling regular flits
            else:
                link = self.links.get(flit.current_link)
                if flit.current_seat_index != len(link) - 1:
                    link[flit.current_seat_index] = None
                    flit.current_seat_index += 1
                else:
                    flit.current_position = next_node
                    if flit.path_index + 1 < len(flit.path):
                        flit.path_index += 1
                        new_current, new_next_node = next_node, flit.path[flit.path_index]
                        if new_current - new_next_node != self.config.rows:
                            flit.current_link = (new_current, new_next_node)
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0
                        else:
                            if current - next_node == 1:
                                station_left = self.transfer_stations_left.get((new_current, new_next_node))
                                if self.config.seats_per_station - len(station_left) > len(
                                    self.station_reservations_left[(new_current, new_next_node)]
                                ):
                                    flit.current_link = (new_current, new_next_node)
                                    link[flit.current_seat_index] = None
                                    flit.current_seat_index = 0
                                else:
                                    if len(self.station_reservations_right[(new_current, new_next_node)]) < self.config.reservation_num:
                                        self.station_reservations_right[(new_current, new_next_node)].append(flit)
                                    flit.is_delay = True
                                    link[flit.current_seat_index] = None
                                    next_pos = next_node - 1 if next_node - 1 >= row_start else row_start
                                    flit.current_link = (new_current, next_pos)
                                    flit.current_seat_index = 0
                            elif current - next_node == -1:
                                station_right = self.transfer_stations_right.get((new_current, new_next_node))
                                if self.config.seats_per_station - len(station_right) > len(
                                    self.station_reservations_right[(new_current, new_next_node)]
                                ):
                                    flit.current_link = (new_current, new_next_node)
                                    link[flit.current_seat_index] = None
                                    flit.current_seat_index = 1
                                else:
                                    if len(self.station_reservations_left[(new_current, new_next_node)]) < self.config.reservation_num:
                                        self.station_reservations_left[(new_current, new_next_node)].append(flit)
                                    flit.is_delay = True
                                    link[flit.current_seat_index] = None
                                    next_pos = new_current + 1 if next_node + 1 <= row_end else row_end
                                    flit.current_link = (new_current, next_pos)
                                    flit.current_seat_index = 0
                    else:
                        if current - next_node == self.config.rows * 2:
                            eject_up = self.eject_queues_up[next_node]
                            if self.config.eject_queues_len - len(eject_up) > len(self.eject_reservations_up[next_node]):
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                                flit.is_arrive = True
                            else:
                                if len(self.eject_reservations_down[next_node]) < self.config.reservation_num:
                                    self.eject_reservations_down[next_node].append(flit)
                                flit.is_delay = True
                                link[flit.current_seat_index] = None
                                next_pos = next_node - self.config.rows * 2 if next_node - self.config.rows * 2 >= col_start else col_start
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                        elif current - next_node == -self.config.rows * 2:
                            eject_down = self.eject_queues_down[next_node]
                            if self.config.eject_queues_len - len(eject_down) > len(self.eject_reservations_down[next_node]):
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                                flit.is_arrive = True
                            else:
                                if len(self.eject_reservations_up[next_node]) < self.config.reservation_num:
                                    self.eject_reservations_up[next_node].append(flit)
                                flit.is_delay = True
                                link[flit.current_seat_index] = None
                                next_pos = next_node + self.config.rows * 2 if next_node + self.config.rows * 2 <= col_end else col_end
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                return

    def execute_moves(self, flit, cycle):
        if not flit.is_arrive:
            current, next_node = flit.current_link
            if current - next_node != self.config.rows:
                link = self.links.get(flit.current_link)
                link[flit.current_seat_index] = flit
            else:
                if not flit.is_on_station:
                    if flit.current_seat_index == 0:
                        self.transfer_stations_left[flit.current_link].append(flit)
                    elif flit.current_seat_index == 1:
                        self.transfer_stations_right[flit.current_link].append(flit)
                    elif flit.current_seat_index == -2:
                        self.transfer_stations_ft[flit.current_link].append(flit)
                    else:
                        self.transfer_stations_up[flit.current_link].append(flit)
                    flit.is_on_station = True
            return False
        else:
            if flit.current_link is not None:
                current, next_node = flit.current_link
            flit.arrival_network_cycle = cycle
            if flit.source - flit.destination == self.config.rows:
                self.eject_queues_local[flit.destination].append(flit)
            elif current - next_node == self.config.rows * 2 or (current == next_node and current not in range(0, self.config.rows)):
                self.eject_queues_up[next_node].append(flit)
            else:
                self.eject_queues_down[next_node].append(flit)
            return True


def process_eject_queues(network, eject_flits, rr_queue, destination_type, ip_pos, cycle):
    for i in rr_queue:
        if (
            eject_flits[i] is not None
            and eject_flits[i].destination_type == destination_type
            and len(network.ip_eject[destination_type][ip_pos]) < network.config.ip_eject_len
        ):
            # network.ip_eject[destination_type][ip_pos].append(eject_flits[i])
            network.eject_queues_pre[destination_type][ip_pos] = eject_flits[i]
            eject_flits[i].arrival_eject_cycle = cycle
            eject_flits[i] = None
            if i == 0:
                network.eject_queues_up[ip_pos].popleft()
            elif i == 1:
                network.eject_queues_mid[ip_pos].popleft()
            elif i == 2:
                network.eject_queues_down[ip_pos].popleft()
            elif i == 3:
                network.eject_queues_local[ip_pos].popleft()
            rr_queue.remove(i)
            rr_queue.append(i)
            break
    return eject_flits


def create_read_packet(config, node, req, cycle, routes):
    for i in range(config.packet_size // config.flit_size):
        source = req.destination + config.rows
        destination = req.source - config.rows
        path = routes[source][destination]
        flit = Flit(config, source, destination, path)
        flit.flit_type = "data"
        flit.departure_cycle = cycle + config.ddr_latency + i
        flit.source_type = req.destination_type
        flit.destination_type = req.source_type
        flit.packet_id = req.packet_id
        flit.id = req.packet_id * (config.packet_size // config.flit_size) + i
        node.sn_rdb[flit.source_type][source].append(flit)


def create_write_packet(config, node, req, cycle, routes):
    for i in range(config.packet_size // config.flit_size):
        source = req.source
        destination = req.destination
        path = routes[source][destination]
        flit = Flit(config, source, destination, path)
        flit.flit_type = "data"
        flit.departure_cycle = cycle
        flit.source_type = req.source_type
        flit.destination_type = req.destination_type
        flit.packet_id = req.packet_id
        flit.id = req.packet_id * (config.packet_size // config.flit_size) + i
        node.rn_wdb[flit.source_type][source][flit.packet_id].append(flit)


def create_rsp(config, node, req, cycle, routes, rsp_type):
    source = req.destination + config.rows
    destination = req.source - config.rows
    path = routes[source][destination]
    rsp = Flit(config, source, destination, path)
    rsp.flit_type = "rsp"
    rsp.rsp_type = rsp_type
    rsp.req_type = req.req_type
    rsp.packet_id = req.packet_id
    rsp.id = Flit.last_id
    rsp.departure_cycle = cycle
    rsp.source_type = req.destination_type
    rsp.destination_type = req.source_type
    node.sn_rsp_queue[rsp.source_type][source].append(rsp)


def flit_move(config, network, node, flits, cycle, flit_type, routes):
    transfer_station_flits = []
    vertical_flits = []
    horizontal_flits = []
    new_flits = []
    local_flits = []
    cycle_mod = cycle % config.network_frequency
    type_mapping = {0: ("sdma", "ddr2"), 1: ("gdma", "ddr1")}
    rn_type, sn_type = type_mapping.get(cycle_mod, ("Idle", "Idle"))
    data_num = config.packet_size // config.flit_size
    for flit in flits:
        if flit.source - flit.destination == network.config.rows:
            flit.is_new_on_network = False
            flit.is_arrive = True
            local_flits.append(flit)
        elif not flit.current_link:
            new_flits.append(flit)
        elif flit.current_link[0] - flit.current_link[1] == config.rows:
            transfer_station_flits.append(flit)
        elif flit.current_link[0] - flit.current_link[1] == 1 or flit.current_link[0] - flit.current_link[1] == -1:
            horizontal_flits.append(flit)
        else:
            vertical_flits.append(flit)
    for flit in new_flits:
        network.plan_move(flit)
    for flit in horizontal_flits:
        network.plan_move(flit)
    # transfer stations arbitrate
    for row in range(1, config.num_nodes // config.rows, 2):
        for col in range(config.rows):
            pos = row * config.rows + col
            next_pos = pos - config.rows
            eject_flit, vup_flit, vdown_flit = None, None, None
            flit0 = network.transfer_stations_up[(pos, next_pos)][0] if network.transfer_stations_up[(pos, next_pos)] else None
            flit1 = network.transfer_stations_left[(pos, next_pos)][0] if network.transfer_stations_left[(pos, next_pos)] else None
            flit2 = network.transfer_stations_right[(pos, next_pos)][0] if network.transfer_stations_right[(pos, next_pos)] else None
            flit3 = network.transfer_stations_ft[(pos, next_pos)][0] if network.transfer_stations_ft[(pos, next_pos)] else None
            station_flits = [flit0, flit1, flit2]
            if len(network.transfer_stations_eject[(pos, next_pos)]) < config.seats_per_vstation:
                if flit3 is not None and flit3.destination == next_pos:
                    eject_flit = flit3
                    network.transfer_stations_ft[(pos, next_pos)].popleft()
                else:
                    index = network.mid_rr[next_pos]
                    for i in index:
                        if station_flits[i] is not None and station_flits[i].destination == next_pos:
                            eject_flit = station_flits[i]
                            station_flits[i] = None
                            if i == 0:
                                network.transfer_stations_up[(pos, next_pos)].popleft()
                            elif i == 1:
                                network.transfer_stations_left[(pos, next_pos)].popleft()
                            elif i == 2:
                                network.transfer_stations_right[(pos, next_pos)].popleft()
                            network.mid_rr[next_pos].remove(i)
                            network.mid_rr[next_pos].append(i)
                            break
            if len(network.transfer_stations_vup[(pos, next_pos)]) < config.seats_per_vstation:
                if flit3 is not None and flit3.destination < next_pos:
                    vup_flit = flit3
                    network.transfer_stations_ft[(pos, next_pos)].popleft()
                else:
                    index = network.up_rr[next_pos]
                    for i in index:
                        if station_flits[i] is not None and station_flits[i].destination < next_pos:
                            vup_flit = station_flits[i]
                            station_flits[i] = None
                            if i == 0:
                                network.transfer_stations_up[(pos, next_pos)].popleft()
                            elif i == 1:
                                network.transfer_stations_left[(pos, next_pos)].popleft()
                            elif i == 2:
                                network.transfer_stations_right[(pos, next_pos)].popleft()
                            network.mid_rr[next_pos].remove(i)
                            network.mid_rr[next_pos].append(i)
                            break
            if len(network.transfer_stations_vdown[(pos, next_pos)]) < config.seats_per_vstation:
                if flit3 is not None and flit3.destination > next_pos:
                    vdown_flit = flit3
                    network.transfer_stations_ft[(pos, next_pos)].popleft()
                else:
                    index = network.down_rr[next_pos]
                    for i in index:
                        if station_flits[i] is not None and station_flits[i].destination > next_pos:
                            vdown_flit = station_flits[i]
                            station_flits[i] = None
                            if i == 0:
                                network.transfer_stations_up[(pos, next_pos)].popleft()
                            elif i == 1:
                                network.transfer_stations_left[(pos, next_pos)].popleft()
                            elif i == 2:
                                network.transfer_stations_right[(pos, next_pos)].popleft()
                            network.mid_rr[next_pos].remove(i)
                            network.mid_rr[next_pos].append(i)
                            break
            # transfer_eject
            if len(network.eject_queues_mid[next_pos]) < config.eject_queues_len:
                if network.transfer_stations_eject[(pos, next_pos)]:
                    flit = network.transfer_stations_eject[(pos, next_pos)].popleft()
                    flit.is_arrive = True
            up_node, down_node = next_pos - config.rows * 2, next_pos + config.rows * 2
            if up_node < 0:
                up_node = next_pos
            if down_node >= config.num_nodes:
                down_node = next_pos

            if network.transfer_stations_vup[(pos, next_pos)]:
                if network.links[(down_node, next_pos)][-1] is not None:
                    if network.links_tag[(down_node, next_pos)][-1] is None:
                        flit_l = network.links[(down_node, next_pos)][-1]
                        if flit_l.destination == next_pos:
                            eject_up = network.eject_queues_up[next_pos]
                            if network.config.eject_queues_len - len(eject_up) > len(network.eject_reservations_up[next_pos]):
                                flit = network.transfer_stations_vup[(pos, next_pos)].popleft()
                                flit.current_position = next_pos
                                flit.path_index += 1
                                flit.current_link = (next_pos, up_node)
                                flit.current_seat_index = 0
                                network.links[(next_pos, up_node)][0] = flit
                            else:
                                if (
                                    network.transfer_stations_vup[(pos, next_pos)][0].wait_cycle_v > config.wait_cycle_v
                                    and not network.transfer_stations_vup[(pos, next_pos)][0].is_tag_v
                                ):
                                    if network.remain_tag["up"][next_pos] > 0:
                                        network.remain_tag["up"][next_pos] -= 1
                                        network.links_tag[(down_node, next_pos)][-1] = [next_pos, "up"]
                                        network.transfer_stations_vup[(pos, next_pos)][0].is_tag_v = True
                                else:
                                    for flit in network.transfer_stations_vup[(pos, next_pos)]:
                                        flit.wait_cycle_v += 1
                        else:
                            if (
                                network.transfer_stations_vup[(pos, next_pos)][0].wait_cycle_v > config.wait_cycle_v
                                and not network.transfer_stations_vup[(pos, next_pos)][0].is_tag_v
                            ):
                                if network.remain_tag["up"][next_pos] > 0:
                                    network.remain_tag["up"][next_pos] -= 1
                                    network.links_tag[(down_node, next_pos)][-1] = [next_pos, "up"]
                                    network.transfer_stations_vup[(pos, next_pos)][0].is_tag_v = True
                            else:
                                for flit in network.transfer_stations_vup[(pos, next_pos)]:
                                    flit.wait_cycle_v += 1
                    else:
                        flit_l = network.links[(down_node, next_pos)][-1]
                        if flit_l.destination == next_pos:
                            eject_up = network.eject_queues_up[next_pos]
                            if network.config.eject_queues_len - len(eject_up) > len(network.eject_reservations_up[next_pos]):
                                if network.links_tag[(down_node, next_pos)][-1] == [next_pos, "up"]:
                                    network.remain_tag["up"][next_pos] += 1
                                    network.links_tag[(down_node, next_pos)][-1] = None
                                    flit = network.transfer_stations_vup[(pos, next_pos)].popleft()
                                    flit.current_position = next_pos
                                    flit.path_index += 1
                                    flit.current_link = (next_pos, up_node)
                                    flit.current_seat_index = 0
                                    network.links[(next_pos, up_node)][0] = flit
                else:
                    if network.links_tag[(down_node, next_pos)][-1] is None:
                        flit = network.transfer_stations_vup[(pos, next_pos)].popleft()
                        flit.current_position = next_pos
                        flit.path_index += 1
                        flit.current_link = (next_pos, up_node)
                        flit.current_seat_index = 0
                        network.links[(next_pos, up_node)][0] = flit
                    else:
                        if network.links_tag[(down_node, next_pos)][-1] == [next_pos, "up"]:
                            network.remain_tag["up"][next_pos] += 1
                            network.links_tag[(down_node, next_pos)][-1] = None
                            flit = network.transfer_stations_vup[(pos, next_pos)].popleft()
                            flit.current_position = next_pos
                            flit.path_index += 1
                            flit.current_link = (next_pos, up_node)
                            flit.current_seat_index = 0
                            network.links[(next_pos, up_node)][0] = flit

            # if network.transfer_stations_vup[(pos, next_pos)]:
            #     if network.links[(down_node, next_pos)][-1] is not None:
            #         if network.transfer_stations_vup[(pos, next_pos)][0].wait_cycle_v > config.wait_cycle_v:
            #             if network.links_tag[(down_node, next_pos)][-1] is None and not network.transfer_stations_vup[(pos, next_pos)][0].is_tag_v:
            #                 if network.remain_tag['up'][next_pos] > 0:
            #                     network.remain_tag['up'][next_pos] -= 1
            #                     network.links_tag[(down_node, next_pos)][-1] = [next_pos, 'up']
            #                     network.transfer_stations_vup[(pos, next_pos)][0].is_tag_v = True
            #         else:
            #             for flit in network.transfer_stations_vup[(pos, next_pos)]:
            #                 flit.wait_cycle_v += 1
            #     elif network.links[(down_node, next_pos)][-1] is None and network.links_tag[(down_node, next_pos)][-1] is None:
            #         flit = network.transfer_stations_vup[(pos, next_pos)].popleft()
            #         flit.current_position = next_pos
            #         flit.path_index += 1
            #         flit.current_link = (next_pos, up_node)
            #         flit.current_seat_index = 0
            #         network.links[(next_pos, up_node)][0] = flit
            #     elif network.links[(down_node, next_pos)][-1] is None and network.links_tag[(down_node, next_pos)][-1] is not None:
            #         if network.links_tag[(down_node, next_pos)][-1] == [next_pos, 'up']:
            #             network.remain_tag['up'][next_pos] += 1
            #             network.links_tag[(down_node, next_pos)][-1] = None
            #             flit = network.transfer_stations_vup[(pos, next_pos)].popleft()
            #             flit.current_position = next_pos
            #             flit.path_index += 1
            #             flit.current_link = (next_pos, up_node)
            #             flit.current_seat_index = 0
            #             network.links[(next_pos, up_node)][0] = flit

            if network.transfer_stations_vdown[(pos, next_pos)]:
                if network.links[(up_node, next_pos)][-1] is not None:
                    if network.links_tag[(up_node, next_pos)][-1] is None:
                        flit_l = network.links[(up_node, next_pos)][-1]
                        if flit_l.destination == next_pos:
                            eject_down = network.eject_queues_down[next_pos]
                            if network.config.eject_queues_len - len(eject_down) > len(network.eject_reservations_down[next_pos]):
                                flit = network.transfer_stations_vdown[(pos, next_pos)].popleft()
                                flit.current_position = next_pos
                                flit.path_index += 1
                                flit.current_link = (next_pos, down_node)
                                flit.current_seat_index = 0
                                network.links[(next_pos, down_node)][0] = flit
                            else:
                                if (
                                    network.transfer_stations_vdown[(pos, next_pos)][0].wait_cycle_v > config.wait_cycle_v
                                    and not network.transfer_stations_vdown[(pos, next_pos)][0].is_tag_v
                                ):
                                    if network.remain_tag["down"][next_pos] > 0:
                                        network.remain_tag["down"][next_pos] -= 1
                                        network.links_tag[(up_node, next_pos)][-1] = [next_pos, "down"]
                                        network.transfer_stations_vdown[(pos, next_pos)][0].is_tag_v = True
                                else:
                                    for flit in network.transfer_stations_vdown[(pos, next_pos)]:
                                        flit.wait_cycle_v += 1
                        else:
                            if (
                                network.transfer_stations_vdown[(pos, next_pos)][0].wait_cycle_v > config.wait_cycle_v
                                and not network.transfer_stations_vdown[(pos, next_pos)][0].is_tag_v
                            ):
                                if network.remain_tag["down"][next_pos] > 0:
                                    network.remain_tag["down"][next_pos] -= 1
                                    network.links_tag[(up_node, next_pos)][-1] = [next_pos, "down"]
                                    network.transfer_stations_vdown[(pos, next_pos)][0].is_tag_v = True
                            else:
                                for flit in network.transfer_stations_vdown[(pos, next_pos)]:
                                    flit.wait_cycle_v += 1
                    else:
                        flit_l = network.links[(up_node, next_pos)][-1]
                        if flit_l.destination == next_pos:
                            eject_down = network.eject_queues_down[next_pos]
                            if network.config.eject_queues_len - len(eject_down) > len(network.eject_reservations_down[next_pos]):
                                if network.links_tag[(up_node, next_pos)][-1] == [next_pos, "down"]:
                                    network.remain_tag["down"][next_pos] += 1
                                    network.links_tag[(up_node, next_pos)][-1] = None
                                    flit = network.transfer_stations_vdown[(pos, next_pos)].popleft()
                                    flit.current_position = next_pos
                                    flit.path_index += 1
                                    flit.current_link = (next_pos, down_node)
                                    flit.current_seat_index = 0
                                    network.links[(next_pos, down_node)][0] = flit
                else:
                    if network.links_tag[(up_node, next_pos)][-1] is None:
                        flit = network.transfer_stations_vdown[(pos, next_pos)].popleft()
                        flit.current_position = next_pos
                        flit.path_index += 1
                        flit.current_link = (next_pos, down_node)
                        flit.current_seat_index = 0
                        network.links[(next_pos, down_node)][0] = flit
                    else:
                        if network.links_tag[(up_node, next_pos)][-1] == [next_pos, "down"]:
                            network.remain_tag["down"][next_pos] += 1
                            network.links_tag[(up_node, next_pos)][-1] = None
                            flit = network.transfer_stations_vdown[(pos, next_pos)].popleft()
                            flit.current_position = next_pos
                            flit.path_index += 1
                            flit.current_link = (next_pos, down_node)
                            flit.current_seat_index = 0
                            network.links[(next_pos, down_node)][0] = flit

            # if network.transfer_stations_vdown[(pos, next_pos)]:
            #     if network.links[(up_node, next_pos)][-1] is not None:
            #         if network.transfer_stations_vdown[(pos, next_pos)][0].wait_cycle_v > config.wait_cycle_v:
            #             if network.links_tag[(up_node, next_pos)][-1] is None and not network.transfer_stations_vdown[(pos, next_pos)][0].is_tag_v:
            #                 if network.remain_tag['down'][next_pos] > 0:
            #                     network.remain_tag['down'][next_pos] -= 1
            #                     network.links_tag[(up_node, next_pos)][-1] = [next_pos, 'down']
            #                     network.transfer_stations_vdown[(pos, next_pos)][0].is_tag_v = True
            #         else:
            #             for flit in network.transfer_stations_vdown[(pos, next_pos)]:
            #                 flit.wait_cycle_v += 1
            #     elif network.links[(up_node, next_pos)][-1] is None and network.links_tag[(up_node, next_pos)][-1] is None:
            #         flit = network.transfer_stations_vdown[(pos, next_pos)].popleft()
            #         flit.current_position = next_pos
            #         flit.path_index += 1
            #         flit.current_link = (next_pos, down_node)
            #         flit.current_seat_index = 0
            #         network.links[(next_pos, down_node)][0] = flit
            #     elif network.links[(up_node, next_pos)][-1] is None and network.links_tag[(up_node, next_pos)][-1] is not None:
            #         if network.links_tag[(up_node, next_pos)][-1] == [next_pos, 'down']:
            #             network.remain_tag['down'][next_pos] += 1
            #             network.links_tag[(up_node, next_pos)][-1] = None
            #             flit = network.transfer_stations_vdown[(pos, next_pos)].popleft()
            #             flit.current_position = next_pos
            #             flit.path_index += 1
            #             flit.current_link = (next_pos, down_node)
            #             flit.current_seat_index = 0
            #             network.links[(next_pos, down_node)][0] = flit
            if eject_flit:
                network.transfer_stations_eject[(pos, next_pos)].append(eject_flit)
            if vup_flit:
                network.transfer_stations_vup[(pos, next_pos)].append(vup_flit)
            if vdown_flit:
                network.transfer_stations_vdown[(pos, next_pos)].append(vdown_flit)

    for flit in vertical_flits:
        network.plan_move(flit)

    # eject arbitrate
    if flit_type == "flit":
        for in_pos in set(config.ddr_send_positions + config.sdma_send_positions + config.l2m_send_positions + config.gdma_send_positions):
            ip_pos = in_pos - config.rows
            eject_flits = [
                network.eject_queues_up[ip_pos][0] if network.eject_queues_up[ip_pos] else None,
                network.eject_queues_mid[ip_pos][0] if network.eject_queues_mid[ip_pos] else None,
                network.eject_queues_down[ip_pos][0] if network.eject_queues_down[ip_pos] else None,
                network.eject_queues_local[ip_pos][0] if network.eject_queues_local[ip_pos] else None,
            ]
            eject_flits = process_eject_queues(network, eject_flits, network.ddr_rr[ip_pos], "ddr", ip_pos, cycle)
            eject_flits = process_eject_queues(network, eject_flits, network.sdma_rr[ip_pos], "sdma", ip_pos, cycle)
            eject_flits = process_eject_queues(network, eject_flits, network.l2m_rr[ip_pos], "l2m", ip_pos, cycle)
            eject_flits = process_eject_queues(network, eject_flits, network.gdma_rr[ip_pos], "gdma", ip_pos, cycle)
        if cycle % 2 == 1:
            for ip_type in ["ddr"]:
                for ip_index in getattr(config, f"{ip_type}_send_positions"):
                    ip_recv_index = ip_index - config.rows
                    if network.ip_eject[ip_type][ip_recv_index]:
                        flit = network.ip_eject[ip_type][ip_recv_index].popleft()
                        flit.arrival_cycle = cycle
                        network.eject_num += 1
                        network.arrive_flits.append(flit)
                        network.recv_flits_num += 1
                        if cycle >= 10 and cycle <= 828:
                            network.num_recv[ip_type][ip_recv_index] += 1
        if cycle % 2 == 0:
            for ip_type in ["sdma", "l2m", "gdma"]:
                for ip_index in getattr(config, f"{ip_type}_send_positions"):
                    ip_recv_index = ip_index - config.rows
                    if network.ip_eject[ip_type][ip_recv_index]:
                        flit = network.ip_eject[ip_type][ip_recv_index].popleft()
                        flit.arrival_cycle = cycle
                        network.eject_num += 1
                        network.arrive_flits.append(flit)
                        network.recv_flits_num += 1
                        if cycle >= 10 and cycle <= 828:
                            network.num_recv[ip_type][ip_recv_index] += 1
        for in_pos in set(config.ddr_send_positions + config.sdma_send_positions + config.l2m_send_positions + config.gdma_send_positions):
            ip_pos = in_pos - config.rows
            for ip_type in network.eject_queues_pre:
                if network.eject_queues_pre[ip_type][ip_pos]:
                    network.ip_eject[ip_type][ip_pos].append(network.eject_queues_pre[ip_type][ip_pos])
                    network.eject_queues_pre[ip_type][ip_pos] = None

    if flit_type == "req":
        for in_pos in config.ddr_send_positions:
            ip_pos = in_pos - config.rows
            eject_flits = [
                network.eject_queues_up[ip_pos][0] if network.eject_queues_up[ip_pos] else None,
                network.eject_queues_mid[ip_pos][0] if network.eject_queues_mid[ip_pos] else None,
                network.eject_queues_down[ip_pos][0] if network.eject_queues_down[ip_pos] else None,
                network.eject_queues_local[ip_pos][0] if network.eject_queues_local[ip_pos] else None,
            ]
            eject_flits = process_eject_queues(network, eject_flits, network.ddr1_rr[ip_pos], "ddr1", ip_pos, cycle)
            eject_flits = process_eject_queues(network, eject_flits, network.ddr2_rr[ip_pos], "ddr2", ip_pos, cycle)
        if sn_type != "Idle":
            for in_pos in config.ddr_send_positions:
                ip_pos = in_pos - config.rows
                if network.ip_eject[sn_type][ip_pos]:
                    req = network.ip_eject[sn_type][ip_pos].popleft()
                    if req.req_type == "read":
                        if req.req_attr == "new":
                            if node.sn_tracker_count[sn_type]["ro"][in_pos] > 0:
                                req.sn_tracker_type = "ro"
                                node.sn_tracker[sn_type][in_pos].append(req)
                                node.sn_tracker_count[sn_type]["ro"][in_pos] -= 1
                                create_read_packet(config, node, req, cycle, routes)
                            elif node.sn_tracker_count[sn_type]["share"][in_pos] > 0:
                                req.sn_tracker_type = "share"
                                node.sn_tracker[sn_type][in_pos].append(req)
                                node.sn_tracker_count[sn_type]["share"][in_pos] -= 1
                                create_read_packet(config, node, req, cycle, routes)
                            else:
                                create_rsp(config, node, req, cycle, routes, "negative")
                                node.sn_req_wait[req.req_type][sn_type][in_pos].append(req)
                        else:
                            create_read_packet(config, node, req, cycle, routes)
                    elif req.req_type == "write":
                        if req.req_attr == "new":
                            if node.sn_tracker_count[sn_type]["share"][in_pos] > 0 and node.sn_wdb_count[sn_type][in_pos] >= data_num:
                                req.sn_tracker_type = "share"
                                node.sn_tracker[sn_type][in_pos].append(req)
                                node.sn_tracker_count[sn_type]["share"][in_pos] -= 1
                                node.sn_wdb[sn_type][in_pos][req.packet_id] = []
                                node.sn_wdb_count[sn_type][in_pos] -= data_num
                                create_rsp(config, node, req, cycle, routes, "datasend")
                            else:
                                create_rsp(config, node, req, cycle, routes, "negative")
                                node.sn_req_wait[req.req_type][sn_type][in_pos].append(req)
                        else:
                            create_rsp(config, node, req, cycle, routes, "datasend")
        for in_pos in config.ddr_send_positions:
            ip_pos = in_pos - config.rows
            for ip_type in network.eject_queues_pre:
                if network.eject_queues_pre[ip_type][ip_pos]:
                    network.ip_eject[ip_type][ip_pos].append(network.eject_queues_pre[ip_type][ip_pos])
                    network.eject_queues_pre[ip_type][ip_pos] = None
    if flit_type == "rsp":
        for in_pos in set(config.sdma_send_positions + config.gdma_send_positions):
            ip_pos = in_pos - config.rows
            eject_flits = [
                network.eject_queues_up[ip_pos][0] if network.eject_queues_up[ip_pos] else None,
                network.eject_queues_mid[ip_pos][0] if network.eject_queues_mid[ip_pos] else None,
                network.eject_queues_down[ip_pos][0] if network.eject_queues_down[ip_pos] else None,
                network.eject_queues_local[ip_pos][0] if network.eject_queues_local[ip_pos] else None,
            ]
            eject_flits = process_eject_queues(network, eject_flits, network.sdma_rr[ip_pos], "sdma", ip_pos, cycle)
            eject_flits = process_eject_queues(network, eject_flits, network.gdma_rr[ip_pos], "gdma", ip_pos, cycle)
        if rn_type != "Idle":
            for in_pos in getattr(config, f"{rn_type}_send_positions"):
                ip_pos = in_pos - config.rows
                if network.ip_eject[rn_type][ip_pos]:
                    rsp = network.ip_eject[rn_type][ip_pos].popleft()
                    if rsp.req_type == "read":
                        if rsp.rsp_type == "negative":
                            req = next((req for req in node.rn_tracker["read"][rn_type][in_pos] if req.packet_id == rsp.packet_id), None)
                            if req and not req.early_rsp:
                                req.req_state = "invalid"
                                req.is_injected = False
                                req.path_index = 0
                                node.rn_rdb_count[rn_type][in_pos] += data_num
                                node.rn_rdb[rn_type][in_pos].pop(req.packet_id)
                                node.rn_tracker_wait["read"][rn_type][in_pos].append(req)
                        else:
                            req = next((req for req in node.rn_tracker["read"][rn_type][in_pos] if req.packet_id == rsp.packet_id), None)
                            req.req_state = "valid"
                            node.rn_rdb_reserve[rn_type][in_pos] += 1
                            if req not in node.rn_tracker_wait["read"][rn_type][in_pos]:
                                req.is_injected = False
                                req.path_index = 0
                                req.early_rsp = True
                                node.rn_tracker_wait["read"][rn_type][in_pos].append(req)
                    elif rsp.req_type == "write":
                        if rsp.rsp_type == "negative":
                            req = next((req for req in node.rn_tracker["write"][rn_type][in_pos] if req.packet_id == rsp.packet_id), None)
                            if req and not req.early_rsp:
                                req.req_state = "invalid"
                                req.is_injected = False
                                req.path_index = 0
                                node.rn_tracker_wait["write"][rn_type][in_pos].append(req)
                        elif rsp.rsp_type == "positive":
                            req = next((req for req in node.rn_tracker["write"][rn_type][in_pos] if req.packet_id == rsp.packet_id), None)
                            req.req_state = "valid"
                            node.rn_wdb_reserve[rn_type][in_pos] += 1
                            if req not in node.rn_tracker_wait["write"][rn_type][in_pos]:
                                req.is_injected = False
                                req.path_index = 0
                                req.early_rsp = True
                                node.rn_tracker_wait["write"][rn_type][in_pos].append(req)
                        else:
                            node.rn_wdb_send[rn_type][in_pos].append(rsp.packet_id)
        for in_pos in set(config.sdma_send_positions + config.gdma_send_positions):
            ip_pos = in_pos - config.rows
            for ip_type in network.eject_queues_pre:
                if network.eject_queues_pre[ip_type][ip_pos]:
                    network.ip_eject[ip_type][ip_pos].append(network.eject_queues_pre[ip_type][ip_pos])
                    network.eject_queues_pre[ip_type][ip_pos] = None
    if flit_type == "data":
        for in_pos in set(config.ddr_send_positions + config.sdma_send_positions + config.l2m_send_positions + config.gdma_send_positions):
            ip_pos = in_pos - config.rows
            eject_flits = [
                network.eject_queues_up[ip_pos][0] if network.eject_queues_up[ip_pos] else None,
                network.eject_queues_mid[ip_pos][0] if network.eject_queues_mid[ip_pos] else None,
                network.eject_queues_down[ip_pos][0] if network.eject_queues_down[ip_pos] else None,
                network.eject_queues_local[ip_pos][0] if network.eject_queues_local[ip_pos] else None,
            ]
            eject_flits = process_eject_queues(network, eject_flits, network.ddr1_rr[ip_pos], "ddr1", ip_pos, cycle)
            eject_flits = process_eject_queues(network, eject_flits, network.ddr2_rr[ip_pos], "ddr2", ip_pos, cycle)
            eject_flits = process_eject_queues(network, eject_flits, network.sdma_rr[ip_pos], "sdma", ip_pos, cycle)
            eject_flits = process_eject_queues(network, eject_flits, network.l2m_rr[ip_pos], "l2m", ip_pos, cycle)
            eject_flits = process_eject_queues(network, eject_flits, network.gdma_rr[ip_pos], "gdma", ip_pos, cycle)
        if rn_type != "Idle":
            for in_pos in set(config.ddr_send_positions + config.sdma_send_positions + config.l2m_send_positions + config.gdma_send_positions):
                for ip_type in [rn_type, sn_type]:
                    ip_pos = in_pos - config.rows
                    if network.ip_eject[ip_type][ip_pos]:
                        flit = network.ip_eject[ip_type][ip_pos].popleft()
                        flit.arrival_cycle = cycle
                        network.arrive_node_pre[ip_type][ip_pos] = flit
                        network.eject_num += 1
                        network.arrive_flits.append(flit)
                        network.recv_flits_num += 1
        for in_pos in set(config.ddr_send_positions + config.sdma_send_positions + config.l2m_send_positions + config.gdma_send_positions):
            ip_pos = in_pos - config.rows
            for ip_type in network.eject_queues_pre:
                if network.eject_queues_pre[ip_type][ip_pos]:
                    network.ip_eject[ip_type][ip_pos].append(network.eject_queues_pre[ip_type][ip_pos])
                    network.eject_queues_pre[ip_type][ip_pos] = None
            if rn_type != "Idle":
                if network.arrive_node_pre[rn_type][ip_pos]:
                    node.rn_rdb[rn_type][in_pos][network.arrive_node_pre[rn_type][ip_pos].packet_id].append(network.arrive_node_pre[rn_type][ip_pos])
                    if len(node.rn_rdb[rn_type][in_pos][network.arrive_node_pre[rn_type][ip_pos].packet_id]) == data_num:
                        node.rn_rdb_recv[rn_type][in_pos].append(network.arrive_node_pre[rn_type][ip_pos].packet_id)
                    network.arrive_node_pre[rn_type][ip_pos] = None
                if network.arrive_node_pre[sn_type][ip_pos]:
                    node.sn_wdb[sn_type][in_pos][network.arrive_node_pre[sn_type][ip_pos].packet_id].append(network.arrive_node_pre[sn_type][ip_pos])
                    if len(node.sn_wdb[sn_type][in_pos][network.arrive_node_pre[sn_type][ip_pos].packet_id]) == data_num:
                        node.sn_wdb_recv[sn_type][in_pos].append(network.arrive_node_pre[sn_type][ip_pos].packet_id)
                    network.arrive_node_pre[sn_type][ip_pos] = None

    for flit in vertical_flits + horizontal_flits + new_flits + local_flits:
        if network.execute_moves(flit, cycle):
            flits.remove(flit)
    for flit in transfer_station_flits:
        if flit.is_arrive:
            flit.arrival_network_cycle = cycle
            network.eject_queues_mid[flit.destination].append(flit)
            flits.remove(flit)
    return flits


def tag_move(config, network, cycle):
    if cycle % (config.seats_per_link * (config.rows - 1) * 2 + 4) == 0:
        for i, j in network.links:
            if i - j == 1 or (i == j and (i % config.rows == config.rows - 1 and (i // config.rows) % 2 != 0)):
                if network.links_tag[(i, j)][-1] == [j, "left"] and network.links[(i, j)][-1] is None:
                    network.links_tag[(i, j)][-1] = None
                    network.remain_tag["left"][j] += 1
            elif i - j == -1 or (i == j and (i % config.rows == 0 and (i // config.rows) % 2 != 0)):
                if network.links_tag[(i, j)][-1] == [j, "right"] and network.links[(i, j)][-1] is None:
                    network.links_tag[(i, j)][-1] = None
                    network.remain_tag["right"][j] += 1
            elif i - j == config.rows * 2 or (
                i == j and i in range(config.num_nodes - config.rows * 2, config.rows + config.num_nodes - config.rows * 2)
            ):
                if network.links_tag[(i, j)][-1] == [j, "up"] and network.links[(i, j)][-1] is None:
                    network.links_tag[(i, j)][-1] = None
                    network.remain_tag["up"][j] += 1
            elif i - j == -config.rows * 2 or (i == j and i in range(0, config.rows)):
                if network.links_tag[(i, j)][-1] == [j, "down"] and network.links[(i, j)][-1] is None:
                    network.links_tag[(i, j)][-1] = None
                    network.remain_tag["down"][j] += 1

    for col_start in range(config.rows):
        interval = config.num_nodes // config.rows
        col_end = col_start + interval * (config.rows - 1)
        last_position = network.links_tag[(col_start, col_start)][0]
        network.links_tag[(col_start, col_start)][0] = network.links_tag[(col_start + interval, col_start)][-1]
        for i in range(1, config.rows):
            current_node, next_node = col_start + i * interval, col_start + (i - 1) * interval
            for j in range(config.seats_per_link - 6 - 1, -1, -1):
                if j == 0 and current_node == col_end:
                    network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, current_node)][-1]
                elif j == 0:
                    network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node + interval, current_node)][-1]
                else:
                    network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]
        network.links_tag[(col_end, col_end)][-1] = network.links_tag[(col_end, col_end)][0]
        network.links_tag[(col_end, col_end)][0] = network.links_tag[(col_end - interval, col_end)][-1]
        for i in range(1, config.rows):
            current_node, next_node = col_end - i * interval, col_end - (i - 1) * interval
            for j in range(config.seats_per_link - 1, -1, -1):
                if j == 0 and current_node == col_start:
                    network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, current_node)][-1]
                elif j == 0:
                    network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node - interval, current_node)][-1]
                else:
                    network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]
        network.links_tag[(col_start, col_start)][-1] = last_position

    for row_start in range(config.rows, config.num_nodes, config.rows * 2):
        row_end = row_start + config.rows - 1
        last_position = network.links_tag[(row_start, row_start)][0]
        network.links_tag[(row_start, row_start)][0] = network.links_tag[(row_start + 1, row_start)][-1]
        for i in range(1, config.rows):
            current_node, next_node = row_start + i, row_start + i - 1
            for j in range(config.seats_per_link - 1, -1, -1):
                if j == 0 and current_node == row_end:
                    network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, current_node)][-1]
                elif j == 0:
                    network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node + 1, current_node)][-1]
                else:
                    network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]
        network.links_tag[(row_end, row_end)][-1] = network.links_tag[(row_end, row_end)][0]
        network.links_tag[(row_end, row_end)][0] = network.links_tag[(row_end - 1, row_end)][-1]
        for i in range(1, config.rows):
            current_node, next_node = row_end - i, row_end - i + 1
            for j in range(config.seats_per_link - 1, -1, -1):
                if j == 0 and current_node == row_start:
                    network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, current_node)][-1]
                elif j == 0:
                    network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node - 1, current_node)][-1]
                else:
                    network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]
        network.links_tag[(row_start, row_start)][-1] = last_position


def has_duplicate_ids(flit_list):
    seen_ids = set()
    for flit in flit_list:
        if flit.id in seen_ids:
            return True
        seen_ids.add(flit.id)
    return False


def node_change(node, is_source=True):
    if is_source:
        node = 16 * (node // 8) + 8 + node % 8
    else:
        node = 16 * (node // 8) + 8 + node % 8 - 8
    return int(node)


def all_queues_empty(network):
    return (
        all(len(queue) == 0 for queue in network.eject_queues_down.values())
        and all(len(queue) == 0 for queue in network.eject_queues_up.values())
        and all(len(queue) == 0 for queue in network.eject_queues_mid.values())
        and all(len(queue) == 0 for queue in network.inject_queues_left.values())
        and all(len(queue) == 0 for queue in network.inject_queues_right.values())
        and all(len(queue) == 0 for queue in network.inject_queues_up.values())
    )


def is_ip_eject_empty(network):
    for ip_type in ["ddr", "sdma", "l2m", "gdma"]:
        if all(len(deque) == 0 for deque in network.ip_eject[ip_type].values()) is not True:
            return False
    return True


def process_inject_queues(network, inject_queues, cycle):
    flit_num = 0
    flits = []
    # item = list(inject_queues.items())
    # random.shuffle(item)
    # for ip_pos in config.ddr_send_positions:
    for source, queue in inject_queues.items():
        if queue and queue[0] is not None:
            flit = queue.popleft()
            if flit.inject(network):
                network.inject_num += 1
                flit_num += 1
                flit.departure_network_cycle = cycle
                flits.append(flit)
            else:
                queue.appendleft(flit)
                for flit in queue:
                    flit.wait_cycle += 1
    return flit_num, flits


def move_to_inject_queue(network, queue_pre, queue, ip_pos, cycle):
    if queue_pre[ip_pos]:
        queue_pre[ip_pos].departure_inject_cycle = cycle
        queue[ip_pos].append(queue_pre[ip_pos])
        queue_pre[ip_pos] = None


def select_inject_network(config, req_network, node, directions, direction_conditions, rn_type, ip_pos):
    data_num = config.packet_size // config.flit_size
    read_old = True if node.rn_rdb_reserve[rn_type][ip_pos] > 0 and node.rn_rdb_count[rn_type][ip_pos] >= data_num else False
    read_new = True if len(node.rn_tracker["read"][rn_type][ip_pos]) - 1 > node.rn_tracker_pointer["read"][rn_type][ip_pos] else False
    write_old = True if node.rn_wdb_reserve[rn_type][ip_pos] > 0 else False
    write_new = True if len(node.rn_tracker["write"][rn_type][ip_pos]) - 1 > node.rn_tracker_pointer["write"][rn_type][ip_pos] else False
    read_valid = read_old or read_new
    write_valid = write_old or write_new
    if read_valid and write_valid:
        if req_network.last_select[rn_type][ip_pos] == "write":
            if read_old:
                req = next((req for req in node.rn_tracker_wait["read"][rn_type][ip_pos] if req.req_state == "valid"), None)
                for direction in directions:
                    queue = getattr(req_network, f"inject_queues_{direction}")
                    queue_pre = getattr(req_network, f"inject_queues_{direction}_pre")
                    if direction_conditions[direction](req) and len(queue[ip_pos]) < config.inject_queues_len:
                        queue_pre[ip_pos] = req
                        node.rn_tracker_wait["read"][rn_type][ip_pos].remove(req)
                        node.rn_rdb_reserve[rn_type][ip_pos] -= 1
                        node.rn_rdb_count[rn_type][ip_pos] -= data_num
                        node.rn_rdb[rn_type][ip_pos][req.packet_id] = []
                        req_network.last_select[rn_type][ip_pos] = "read"
            elif read_new:
                rn_tracker_pointer = node.rn_tracker_pointer["read"][rn_type][ip_pos] + 1
                req = node.rn_tracker["read"][rn_type][ip_pos][rn_tracker_pointer]
                for direction in directions:
                    queue = getattr(req_network, f"inject_queues_{direction}")
                    queue_pre = getattr(req_network, f"inject_queues_{direction}_pre")
                    if direction_conditions[direction](req) and len(queue[ip_pos]) < config.inject_queues_len:
                        queue_pre[ip_pos] = req
                        node.rn_tracker_pointer["read"][rn_type][ip_pos] += 1
                        req_network.last_select[rn_type][ip_pos] = "read"
        else:
            if write_old:
                req = next((req for req in node.rn_tracker_wait["write"][rn_type][ip_pos] if req.req_state == "valid"), None)
                for direction in directions:
                    queue = getattr(req_network, f"inject_queues_{direction}")
                    queue_pre = getattr(req_network, f"inject_queues_{direction}_pre")
                    if direction_conditions[direction](req) and len(queue[ip_pos]) < config.inject_queues_len:
                        queue_pre[ip_pos] = req
                        node.rn_tracker_wait["write"][rn_type][ip_pos].remove(req)
                        node.rn_wdb_reserve[rn_type][ip_pos] -= 1
                        req_network.last_select[rn_type][ip_pos] = "write"
            elif write_new:
                rn_tracker_pointer = node.rn_tracker_pointer["write"][rn_type][ip_pos] + 1
                req = node.rn_tracker["write"][rn_type][ip_pos][rn_tracker_pointer]
                for direction in directions:
                    queue = getattr(req_network, f"inject_queues_{direction}")
                    queue_pre = getattr(req_network, f"inject_queues_{direction}_pre")
                    if direction_conditions[direction](req) and len(queue[ip_pos]) < config.inject_queues_len:
                        queue_pre[ip_pos] = req
                        node.rn_tracker_pointer["write"][rn_type][ip_pos] += 1
                        req_network.last_select[rn_type][ip_pos] = "write"
    elif read_valid:
        if read_old:
            req = next((req for req in node.rn_tracker_wait["read"][rn_type][ip_pos] if req.req_state == "valid"), None)
            for direction in directions:
                queue = getattr(req_network, f"inject_queues_{direction}")
                queue_pre = getattr(req_network, f"inject_queues_{direction}_pre")
                if direction_conditions[direction](req) and len(queue[ip_pos]) < config.inject_queues_len:
                    queue_pre[ip_pos] = req
                    node.rn_tracker_wait["read"][rn_type][ip_pos].remove(req)
                    node.rn_rdb_reserve[rn_type][ip_pos] -= 1
                    node.rn_rdb_count[rn_type][ip_pos] -= data_num
                    node.rn_rdb[rn_type][ip_pos][req.packet_id] = []
                    req_network.last_select[rn_type][ip_pos] = "read"
        elif read_new:
            rn_tracker_pointer = node.rn_tracker_pointer["read"][rn_type][ip_pos] + 1
            req = node.rn_tracker["read"][rn_type][ip_pos][rn_tracker_pointer]
            for direction in directions:
                queue = getattr(req_network, f"inject_queues_{direction}")
                queue_pre = getattr(req_network, f"inject_queues_{direction}_pre")
                if direction_conditions[direction](req) and len(queue[ip_pos]) < config.inject_queues_len:
                    queue_pre[ip_pos] = req
                    node.rn_tracker_pointer["read"][rn_type][ip_pos] += 1
                    req_network.last_select[rn_type][ip_pos] = "read"
    elif write_valid:
        if write_old:
            req = next((req for req in node.rn_tracker_wait["write"][rn_type][ip_pos] if req.req_state == "valid"), None)
            for direction in directions:
                queue = getattr(req_network, f"inject_queues_{direction}")
                queue_pre = getattr(req_network, f"inject_queues_{direction}_pre")
                if direction_conditions[direction](req) and len(queue[ip_pos]) < config.inject_queues_len:
                    queue_pre[ip_pos] = req
                    node.rn_tracker_wait["write"][rn_type][ip_pos].remove(req)
                    node.rn_wdb_reserve[rn_type][ip_pos] -= 1
                    req_network.last_select[rn_type][ip_pos] = "write"
        elif write_new:
            rn_tracker_pointer = node.rn_tracker_pointer["write"][rn_type][ip_pos] + 1
            req = node.rn_tracker["write"][rn_type][ip_pos][rn_tracker_pointer]
            for direction in directions:
                queue = getattr(req_network, f"inject_queues_{direction}")
                queue_pre = getattr(req_network, f"inject_queues_{direction}_pre")
                if direction_conditions[direction](req) and len(queue[ip_pos]) < config.inject_queues_len:
                    queue_pre[ip_pos] = req
                    node.rn_tracker_pointer["write"][rn_type][ip_pos] += 1
                    req_network.last_select[rn_type][ip_pos] = "write"


def analysis(config, packets):
    diff_Latency = {}
    all_predicted_latency, all_actual_latency, all_actual_ject_latency, all_actual_net_latency = 0, 0, 0, 0
    predicted_max_latency, actual_max_latency, actual_max_ject_latency, actual_max_net_latency = 0, 0, 0, 0
    len_packets = 0
    for packet_id in packets:
        predicted_latency = 0
        actual_ject_duration = 0
        for flit in packets[packet_id]:
            predicted_latency += flit.predicted_duration
            actual_ject_duration += flit.actual_ject_duration
            all_predicted_latency += flit.predicted_duration
            predicted_max_latency = max(flit.predicted_duration, predicted_max_latency)
            all_actual_latency += flit.actual_duration
            actual_max_latency = max(flit.actual_duration, actual_max_latency)
            all_actual_ject_latency += flit.actual_ject_duration
            actual_max_ject_latency = max(flit.actual_ject_duration, actual_max_ject_latency)
            all_actual_net_latency += flit.actual_network_duration
            actual_max_net_latency = max(flit.actual_network_duration, actual_max_net_latency)
        predicted_latency /= len(packets[packet_id])
        actual_ject_duration /= len(packets[packet_id])
        diff = actual_ject_duration - predicted_latency
        diff_Latency[packet_id] = diff / config.network_frequency
        len_packets += len(packets[packet_id])
    if not len_packets:
        len_packets = 1
    predicted_avg_latency = all_predicted_latency / len_packets / config.network_frequency
    predicted_max_latency = predicted_max_latency / config.network_frequency
    actual_avg_latency = all_actual_latency / len_packets / config.network_frequency
    actual_max_latency = actual_max_latency / config.network_frequency
    actual_avg_ject_latency = all_actual_ject_latency / len_packets / config.network_frequency
    actual_max_ject_latency = actual_max_ject_latency / config.network_frequency
    actual_avg_net_latency = all_actual_net_latency / len_packets / config.network_frequency
    actual_max_net_latency = actual_max_net_latency / config.network_frequency

    analysis_result = [
        predicted_avg_latency,
        predicted_max_latency,
        actual_avg_latency,
        actual_max_latency,
        actual_avg_ject_latency,
        actual_max_ject_latency,
        actual_avg_net_latency,
        actual_max_net_latency,
    ]

    # keys = list(diff_Latency.keys())
    # values = list(diff_Latency.values())
    # plt.figure(figsize=(10, 6))
    # # plt.plot(keys, values, marker='o')
    # plt.scatter(keys, values)
    # plt.xlabel('packet number')
    # plt.ylabel('Diff')
    # plt.title('512B 3.0')
    # plt.grid(True)
    # plt.show(block=True)

    return analysis_result


def performance_evaluate(config, network, cycle, throughput_time):
    # receive confirm
    send_flit_ids = {flit.id for flit in network.send_flits}
    arrive_flit_ids = {flit.id for flit in network.arrive_flits}
    unreceived_flit_ids = send_flit_ids - arrive_flit_ids
    unreceived_flits = [flit for flit in network.send_flits if flit.id in unreceived_flit_ids]

    # Latency confirm
    for flit in network.arrive_flits:
        for i in range(1, len(flit.path)):
            if flit.path[i] - flit.path[i - 1] == -config.rows:
                flit.predicted_duration += 2
            else:
                flit.predicted_duration += config.seats_per_link
        if flit.path[1] - flit.path[0] == -config.rows:
            flit.predicted_duration += 2
        else:
            flit.predicted_duration += 3
        if len(flit.path) == 2:
            flit.predicted_duration = 0

        flit.actual_duration = flit.arrival_cycle - flit.departure_cycle
        flit.actual_ject_duration = flit.arrival_eject_cycle - flit.departure_inject_cycle
        flit.actual_network_duration = flit.arrival_network_cycle - flit.departure_network_cycle
        network.inject_time[flit.source].append(flit.departure_inject_cycle - flit.departure_cycle)
        network.eject_time[flit.destination].append(flit.arrival_cycle - flit.arrival_network_cycle)
        network.all_latency.append(flit.actual_duration)
        network.ject_latency.append(flit.actual_ject_duration)
        network.network_latency.append(flit.actual_network_duration)
        network.predicted_recv_time.append(flit.predicted_duration)
        network.circuits_h.append(flit.circuits_completed_h) if flit.circuits_completed_h != 0 else None
        network.circuits_v.append(flit.circuits_completed_v) if flit.circuits_completed_v != 0 else None
        if flit.circuits_completed_h != 0:
            network.circuits_flit_h[flit.destination_type] += 1
        if flit.circuits_completed_v != 0:
            network.circuits_flit_v[flit.destination_type] += 1

    for source in set(config.ddr_send_positions + config.sdma_send_positions + config.l2m_send_positions + config.gdma_send_positions):
        destination = source - config.rows
        if network.inject_time[source]:
            network.avg_inject_time[source] = sum(network.inject_time[source]) / len(network.inject_time[source])
        if len(network.eject_time[destination]) > 0:
            network.avg_eject_time[destination] = sum(network.eject_time[destination]) / len(network.eject_time[destination])
    # network.predicted_avg_latency = sum(network.predicted_recv_time) / len(network.predicted_recv_time) / 2
    # network.predicted_max_latency = max(network.predicted_recv_time) / 2
    # network.actual_avg_latency = sum(network.all_latency) / len(network.all_latency) / 2
    # network.actual_max_latency = max(network.all_latency) / 2
    # network.actual_avg_ject_latency = sum(network.ject_latency) / len(network.ject_latency) / 2
    # network.actual_max_ject_latency = max(network.ject_latency) / 2
    # network.actual_avg_net_latency = sum(network.network_latency) / len(network.network_latency) / 2
    # network.actual_max_net_latency = max(network.network_latency) / 2
    # network.avg_circuits_h = sum(network.circuits_h) / len(network.circuits_h) / 2 if len(network.circuits_h) > 0 else None
    # network.max_circuits_h = max(network.circuits_h) / 2 if len(network.circuits_h) > 0 else None
    # network.avg_circuits_v = sum(network.circuits_v) / len(network.circuits_v) / 2 if len(network.circuits_v) > 0 else None
    # network.max_circuits_v = max(network.circuits_v) / 2 if len(network.circuits_v) > 0 else None

    # throughput confirm
    cycle = 828 - 10
    for ip_type in network.per_send_throughput:
        for source in network.per_send_throughput[ip_type]:
            destination = source - config.rows
            network.per_send_throughput[ip_type][source] = 256 * network.num_send[ip_type][source] / (cycle)
            network.per_recv_throughput[ip_type][destination] = 256 * network.num_recv[ip_type][destination] / (cycle)
            network.send_throughput[ip_type] += network.per_send_throughput[ip_type][source]
            network.recv_throughput[ip_type] += network.per_recv_throughput[ip_type][destination]
        # network.send_throughput[ip_type] = network.send_throughput[ip_type] / len(network.per_send_throughput[ip_type]) if len(network.per_send_throughput[ip_type]) > 0 else None
        # network.recv_throughput[ip_type] = network.recv_throughput[ip_type] / len(network.per_send_throughput[ip_type]) if len(network.per_send_throughput[ip_type]) > 0 else None

    sorted_flits = sorted(network.arrive_flits, key=lambda flit: flit.id)
    read_flits, write_flits = {}, {}
    for flit in sorted_flits:
        if flit.source_type == "ddr" or flit.source_type == "ddr1" or flit.source_type == "ddr2":
            if flit.packet_id in read_flits:
                read_flits[flit.packet_id].append(flit)
            else:
                read_flits[flit.packet_id] = [flit]
        else:
            if flit.packet_id in read_flits:
                write_flits[flit.packet_id].append(flit)
            else:
                write_flits[flit.packet_id] = [flit]
    read_result = analysis(config, read_flits)
    write_result = analysis(config, write_flits)
    sdma_throughput, ddr_throughput = 0, 0
    for flit in sorted_flits:
        network.throughput[flit.destination_type][flit.destination + config.rows][1] += 1
        first_time = network.throughput[flit.destination_type][flit.destination + config.rows][2]
        network.throughput[flit.destination_type][flit.destination + config.rows][2] = min(flit.departure_inject_cycle, first_time)
        last_time = network.throughput[flit.destination_type][flit.destination + config.rows][3]
        network.throughput[flit.destination_type][flit.destination + config.rows][3] = max(flit.arrival_cycle, last_time)
    for source in config.sdma_send_positions:
        network.throughput["sdma"][source][0] = (
            network.throughput["sdma"][source][1]
            * config.flit_size
            * config.network_frequency
            / (network.throughput["sdma"][source][3] - network.throughput["sdma"][source][2])
        )
        sdma_throughput += network.throughput["sdma"][source][0]
    for source in config.ddr_send_positions:
        network.throughput["ddr1"][source][0] = (
            network.throughput["ddr1"][source][1]
            * config.flit_size
            * config.network_frequency
            / (network.throughput["ddr1"][source][3] - network.throughput["ddr1"][source][2])
        )
        ddr_throughput += network.throughput["ddr1"][source][0]
    for source in config.ddr_send_positions:
        network.throughput["ddr"][source][0] = (
            network.throughput["ddr"][source][1]
            * config.flit_size
            * config.network_frequency
            / (network.throughput["ddr"][source][3] - network.throughput["ddr"][source][2])
        )
        ddr_throughput += network.throughput["ddr"][source][0]
    sdma_throughput /= 56
    ddr_throughput /= 64
    throughput_data = {
        "ddr": {"ddr_throughput": 0, "first_send": 0, "recv_num": 0, "last_recv": 0, "num": 64},
        "ddr1": {"ddr1_throughput": 0, "first_send": 0, "recv_num": 0, "last_recv": 0, "num": 64},
        "ddr2": {"ddr2_throughput": 0, "first_send": 0, "recv_num": 0, "last_recv": 0, "num": 64},
        "sdma": {"sdma_throughput": 0, "first_send": 0, "recv_num": 0, "last_recv": 0, "num": 64},
        "gdma": {"gdma_throughput": 0, "first_send": 0, "recv_num": 0, "last_recv": 0, "num": 64},
    }
    begin_id = 0

    for i in range(len(throughput_time)):
        end_id = throughput_time[i] + begin_id
        for j in range(begin_id, end_id):
            source = sorted_flits[j].source_type
            destination = sorted_flits[j].destination_type
            if throughput_data[source]["first_send"] == 0:
                throughput_data[source]["first_send"] = sorted_flits[j].departure_inject_cycle
            if destination in throughput_data:
                throughput_data[destination]["recv_num"] += 1
                throughput_data[destination]["last_recv"] = max(throughput_data[destination]["last_recv"], sorted_flits[j].arrival_cycle)
        for key in throughput_data:
            if throughput_data[key]["last_recv"] > 0:
                if key == "sdma":
                    throughput_data[key][f"{key}_throughput"] = (
                        throughput_data[key]["recv_num"]
                        * 256
                        / (throughput_data["sdma"]["last_recv"] - throughput_data["ddr1"]["first_send"])
                        / throughput_data[key]["num"]
                    )
                elif key == "ddr":
                    throughput_data[key][f"{key}_throughput"] = (
                        throughput_data[key]["recv_num"]
                        * 256
                        / (throughput_data["ddr"]["last_recv"] - throughput_data["sdma"]["first_send"])
                        / throughput_data[key]["num"]
                    )
                elif key == "ddr1":
                    throughput_data[key][f"{key}_throughput"] = (
                        throughput_data[key]["recv_num"]
                        * 256
                        / (throughput_data["ddr1"]["last_recv"] - throughput_data["sdma"]["first_send"])
                        / throughput_data[key]["num"]
                    )
                elif key == "ddr2":
                    throughput_data[key][f"{key}_throughput"] = (
                        throughput_data[key]["recv_num"]
                        * 256
                        / (throughput_data["ddr2"]["last_recv"] - throughput_data["sdma"]["first_send"])
                        / throughput_data[key]["num"]
                    )
                elif key == "gdma":
                    throughput_data[key][f"{key}_throughput"] = (
                        throughput_data[key]["recv_num"]
                        * 256
                        / (throughput_data["gdma"]["last_recv"] - throughput_data["ddr"]["first_send"])
                        / throughput_data[key]["num"]
                    )
        begin_id = end_id
    return


def draw_figure(config, flit_network):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    ax1.plot(flit_network.current_cycle, flit_network.flits_num, linestyle="-", color="b", label="Sample Data")
    ax1.set_title("Figure 1")
    ax1.set_xlabel("cycle")
    ax1.set_ylabel("packet num on flit_network")
    ax1.grid(True)
    ax1.legend()

    x_values = list(range(64))
    y_values = []
    for ip_pos in set(config.ddr_send_positions + config.sdma_send_positions + config.l2m_send_positions + config.gdma_send_positions):
        y_values.append(flit_network.avg_inject_time[ip_pos])
    ax2.bar(x_values, y_values, color="blue")
    ax2.set_title("Average Inject Time")
    ax2.set_xlabel("Node")
    ax2.set_ylabel("Latency")
    ax2.set_xticks(list(range(64)))
    ax2.set_xticklabels(list(range(64)), rotation=90)

    x_values = list(range(64))
    y_values = []
    for in_pos in set(config.ddr_send_positions + config.sdma_send_positions + config.l2m_send_positions + config.gdma_send_positions):
        ip_pos = in_pos - config.rows
        y_values.append(flit_network.avg_eject_time[ip_pos])
    ax3.bar(x_values, y_values, color="red")
    ax3.set_title("Average Eject Time")
    ax3.set_xlabel("Node")
    ax3.set_ylabel("Latency")
    ax3.set_xticks(list(range(64)))
    ax3.set_xticklabels(list(range(64)), rotation=90)

    plt.tight_layout()
    plt.show(block=True)


def choose_positions(sn_type, node):
    array = []
    sorted_dict = dict(sorted(node.sn_rdb[sn_type].items(), key=lambda item: len(item[1]), reverse=True))
    key = list(sorted_dict.keys())
    for i in range(len(sorted_dict) // 2):
        if len(sorted_dict[key[i]]) - len(sorted_dict[key[len(sorted_dict) - i - 1]]) > 128:
            array.append(key[i])
        else:
            array.append(key[i])
            array.append(key[len(sorted_dict) - i - 1])
    return array


def specific_dataflow(config, routes, flit_network):
    node = Node(config)
    flits = []
    send_flits_num, trans_flits_num = 0, 0
    flit_num = 0
    begin, end = None, None
    throughput_time = []

    # file_path = "/home/samzhang/work/program/model/topology/data/v6_data_trace/AddPacketId_Ring_all_reduce_8cluster_all2all_Trace.txt"
    file_path = "burst1.txt"
    data_stream = []
    data_count = 0
    with open(file_path, "r") as file:
        for line in file:
            split_line = [x for x in line.strip().split(",")]
            # split_line = [int(split_line[0]), int(split_line[1]), split_line[2], int(split_line[3]), split_line[4], int(split_line[5])]
            split_line = [int(split_line[0]), int(split_line[1]), split_line[2], int(split_line[3]), split_line[4], split_line[5]]
            data_stream.append(split_line)

    for cycle in range(config.num_cycles_total):
        while data_count < len(data_stream):
            if data_stream[data_count][0] == cycle:
                send_flits_num += 1
                source = node_change(data_stream[data_count][1])
                destination = node_change(data_stream[data_count][3], False)
                trans_flits_num += 1
                path = routes[source][destination]
                flit = Flit(config, source, destination, path)
                flit.departure_cycle = cycle
                flit.source_type = data_stream[data_count][2]
                flit.destination_type = data_stream[data_count][4]
                # flit.packet_id = data_stream[data_count][5]
                flit.packet_id = data_count + 1
                flit_network.send_flits.append(flit)
                flit_network.ip_inject[flit.source_type][source].append(flit)
                if cycle >= 10 and cycle <= 828:
                    flit_network.num_send[data_stream[data_count][2]][source] += 1
                data_count += 1
            else:
                break

        if cycle % 2 == 0:
            for ip_index in set(config.ddr_send_positions + config.sdma_send_positions + config.l2m_send_positions + config.gdma_send_positions):
                # flit0 = flit_network.ip_inject['ddr'][ip_index][0] if flit_network.ip_inject['ddr'][ip_index] else None
                flit1 = flit_network.ip_inject["sdma"][ip_index][0] if flit_network.ip_inject["sdma"][ip_index] else None
                flit2 = flit_network.ip_inject["l2m"][ip_index][0] if flit_network.ip_inject["l2m"][ip_index] else None
                flit3 = flit_network.ip_inject["gdma"][ip_index][0] if flit_network.ip_inject["gdma"][ip_index] else None
                inject_flits = [flit1, flit2, flit3]
                rr_index = flit_network.inject_right_rr[ip_index]
                for i in rr_index:
                    flit = inject_flits[i]
                    if flit is not None:
                        if flit.path[1] - flit.path[0] == 1 and len(flit_network.inject_queues_right[ip_index]) < config.inject_queues_len:
                            flit_network.inject_queues_right_pre[flit.source] = flit
                            flit_network.ip_inject[flit.source_type][ip_index].popleft()
                            inject_flits[i] = None
                            flit_network.inject_right_rr[ip_index].remove(i)
                            flit_network.inject_right_rr[ip_index].append(i)
                            break
                rr_index = flit_network.inject_left_rr[ip_index]
                for i in rr_index:
                    flit = inject_flits[i]
                    if flit is not None:
                        if flit.path[1] - flit.path[0] == -1 and len(flit_network.inject_queues_left[ip_index]) < config.inject_queues_len:
                            flit_network.inject_queues_left_pre[flit.source] = flit
                            flit_network.ip_inject[flit.source_type][ip_index].popleft()
                            inject_flits[i] = None
                            flit_network.inject_left_rr[ip_index].remove(i)
                            flit_network.inject_left_rr[ip_index].append(i)
                            break
                rr_index = flit_network.inject_up_rr[ip_index]
                for i in rr_index:
                    flit = inject_flits[i]
                    if flit is not None:
                        if (
                            flit.path[1] - flit.path[0] == -config.rows
                            and flit.source - flit.destination != config.rows
                            and len(flit_network.inject_queues_up[ip_index]) < config.inject_queues_len
                        ):
                            flit_network.inject_queues_up_pre[flit.source] = flit
                            flit_network.ip_inject[flit.source_type][ip_index].popleft()
                            inject_flits[i] = None
                            flit_network.inject_up_rr[ip_index].remove(i)
                            flit_network.inject_up_rr[ip_index].append(i)
                            break
                rr_index = flit_network.inject_local_rr[ip_index]
                for i in rr_index:
                    flit = inject_flits[i]
                    if flit is not None:
                        if (
                            flit.source - flit.destination == config.rows
                            and len(flit_network.inject_queues_local[ip_index]) < config.inject_queues_len
                        ):
                            flit_network.inject_queues_local_pre[flit.source] = flit
                            flit_network.ip_inject[flit.source_type][ip_index].popleft()
                            inject_flits[i] = None
                            flit_network.inject_local_rr[ip_index].remove(i)
                            flit_network.inject_local_rr[ip_index].append(i)
                            break
        else:
            for ip_index in config.ddr_send_positions:
                if flit_network.ip_inject["ddr"][ip_index]:
                    flit = flit_network.ip_inject["ddr"][ip_index][0]
                    if flit.source - flit.destination == config.rows and len(flit_network.inject_queues_local[ip_index]) < config.inject_queues_len:
                        flit_network.inject_queues_local_pre[flit.source] = flit
                        flit_network.ip_inject["ddr"][ip_index].popleft()
                    elif flit.path[1] - flit.path[0] == 1 and len(flit_network.inject_queues_right[ip_index]) < config.inject_queues_len:
                        flit_network.inject_queues_right_pre[flit.source] = flit
                        flit_network.ip_inject["ddr"][ip_index].popleft()
                    elif flit.path[1] - flit.path[0] == -1 and len(flit_network.inject_queues_left[ip_index]) < config.inject_queues_len:
                        flit_network.inject_queues_left_pre[flit.source] = flit
                        flit_network.ip_inject["ddr"][ip_index].popleft()
                    elif (
                        flit.path[1] - flit.path[0] == -config.rows
                        and flit.source - flit.destination != config.rows
                        and len(flit_network.inject_queues_up[ip_index]) < config.inject_queues_len
                    ):
                        flit_network.inject_queues_up_pre[flit.source] = flit
                        flit_network.ip_inject["ddr"][ip_index].popleft()

        for inject_queues in [
            flit_network.inject_queues_right,
            flit_network.inject_queues_left,
            flit_network.inject_queues_up,
            flit_network.inject_queues_local,
        ]:
            num, list = process_inject_queues(flit_network, inject_queues, cycle)
            flit_num += num
            flits.extend(list)

        flits = flit_move(config, flit_network, node, flits, cycle, "flit", routes)

        for ip_pos in set(config.ddr_send_positions + config.sdma_send_positions + config.l2m_send_positions + config.gdma_send_positions):
            move_to_inject_queue(flit_network, flit_network.inject_queues_left_pre, flit_network.inject_queues_left, ip_pos, cycle)
            move_to_inject_queue(flit_network, flit_network.inject_queues_right_pre, flit_network.inject_queues_right, ip_pos, cycle)
            move_to_inject_queue(flit_network, flit_network.inject_queues_up_pre, flit_network.inject_queues_up, ip_pos, cycle)
            move_to_inject_queue(flit_network, flit_network.inject_queues_local_pre, flit_network.inject_queues_local, ip_pos, cycle)

        tag_move(config, flit_network, cycle)
        flit_network.current_cycle.append(cycle)
        flit_network.flits_num.append(len(flits))
        if len(flits) > 0 and begin == None:
            begin = cycle
        if len(flits) == 0 and begin != None and end == None:
            end = cycle
            throughput_time.append(trans_flits_num)
            trans_flits_num = 0
            begin, end = None, None
        # print(cycle, len(flits))
        if cycle > 0 and cycle % 1000 == 0:
            print(
                f"{cycle}, Flit count: {flit_num},  Sent flits: {send_flits_num}, Transferred flits: {trans_flits_num}, Received flits: {flit_network.recv_flits_num}"
            )
        if len(flits) == 0 and all_queues_empty(flit_network) and cycle >= data_stream[-1][0] and is_ip_eject_empty(flit_network):
            break
    print(has_duplicate_ids(flit_network.arrive_flits))
    performance_evaluate(config, flit_network, cycle, throughput_time)
    print(f"predicted_recv_time: {flit_network.predicted_avg_latency}, actual_recv_time: {flit_network.actual_avg_latency}")
    print(f"flit_num: {flit_num}")
    print(f"send_flits_num: {send_flits_num}, trans_flits_num: {trans_flits_num}, recv_flits_num: {flit_network.recv_flits_num}")


def read_dataflow(config, routes, req_network, rsp_network, flit_network):
    node = Node(config)
    flits, reqs, rsps = [], [], []
    queue_manager = DestinationQueueManager(config)
    send_flits_num, trans_flits_num = 0, 0
    send_reqs_num, trans_flits_num = 0, 0
    flit_num, req_num, rsp_num = 0, 0, 0
    data_num = config.packet_size // config.flit_size
    read_req, write_req = 0, 0
    begin, end = None, None
    throughput_time = []
    directions = ["right", "left", "up", "local"]
    direction_conditions = {
        "right": lambda flit: flit.path[1] - flit.path[0] == 1,
        "left": lambda flit: flit.path[1] - flit.path[0] == -1,
        "up": lambda flit: flit.path[1] - flit.path[0] == -config.rows and flit.source - flit.destination != config.rows,
        "local": lambda flit: flit.source - flit.destination == config.rows,
    }

    # file_path = r"../mesh/demo3.txt"
    file_path = r"demo3.txt"
    # file_path = r"../traffic/output-v7-32/step5_data_merge/LLama2_MM_QKV_Trace.txt"
    req_stream = []
    req_count = 0
    with open(file_path, "r") as file:
        for line in file:
            split_line = [x for x in line.strip().split(",")]
            split_line = [int(split_line[0]) * 2, int(split_line[1]), split_line[2], int(split_line[3]), split_line[4], split_line[5]]
            if split_line[5] == "R":
                read_req += 1
            elif split_line[5] == "W":
                write_req += 1
            req_stream.append(split_line)
    cycle = -1
    # for cycle in range(config.num_cycles_total):
    while True:
        cycle += 1
        cycle_mod = cycle % config.network_frequency
        type_mapping = {0: ("sdma", "ddr2"), 1: ("gdma", "ddr1")}
        rn_type, sn_type = type_mapping.get(cycle_mod, ("Idle", "Idle"))
        # req process
        # specific stream
        while req_count < len(req_stream):
            if req_stream[req_count][0] == cycle:
                send_reqs_num += 1
                source = node_change(req_stream[req_count][1])
                destination = node_change(req_stream[req_count][3], False)
                # if source % 2 ==  0:
                path = routes[source][destination]
                req = Flit(config, source, destination, path)
                req.flit_type = "req"
                req.departure_cycle = cycle
                req.source_type = req_stream[req_count][2]
                req.destination_type = "ddr1" if req_stream[req_count][4] == "ddr" else None
                req.packet_id = Node.get_next_packet_id()
                req.id = Flit.last_id
                req.req_type = "read" if req_stream[req_count][5] == "R" else "write"
                req_network.send_flits.append(req)
                if req.req_type == "read":
                    req_network.ip_read[req.source_type][source].append(req)
                if req.req_type == "write":
                    req_network.ip_write[req.source_type][source].append(req)
                req_count += 1
                # else:
                #     req_count += 1
            else:
                break
        # #random stream
        # if cycle < config.num_cycles_send:
        #     if cycle % config.num_round_cycles == 0:
        #         for ip_type in req_network.schedules:
        #             req_network.schedules[ip_type] = req_network.generate_flit_schedule(ip_type, config.num_round_cycles,
        #                                              getattr(config, f"{ip_type}_send_rate"), getattr(config, f"num_{ip_type}"), cycle)
        # for ip_type in req_network.schedules:
        #     for ip_index in range(getattr(config, f"num_{ip_type}")):
        #         if req_network.schedules[ip_type][ip_index, cycle % config.num_round_cycles]:
        #             source = getattr(config, f"{ip_type}_send_positions")[ip_index]
        #             destination = queue_manager.get_destination(ip_type, ip_index)
        #             path = routes[source][destination]
        #             req = Flit(config, source, destination, path)
        #             req.packet_id = Node.get_next_packet_id()
        #             req.flit_type = 'req'
        #             req.source_type = ip_type
        #             req.destination_type = queue_manager.get_destination_type(ip_type)
        #             req.departure_cycle = cycle
        #             req.req_type = 'read'
        #             req_network.send_flits.append(req)
        #             req_network.ip_inject[req.source_type][source].append(req)
        if rn_type != "Idle":
            for ip_pos in getattr(config, f"{rn_type}_send_positions"):
                for req_type in ["read", "write"]:
                    if req_type == "read":
                        if req_network.ip_read[rn_type][ip_pos]:
                            req = req_network.ip_read[rn_type][ip_pos][0]
                            if node.rn_rdb_count[rn_type][ip_pos] > node.rn_rdb_reserve[rn_type][ip_pos] * data_num:
                                if node.rn_tracker_count[req_type][rn_type][ip_pos] > 0:
                                    req_network.ip_read[rn_type][ip_pos].popleft()
                                    node.rn_tracker[req_type][rn_type][ip_pos].append(req)
                                    node.rn_tracker_count[req_type][rn_type][ip_pos] -= 1
                                    node.rn_rdb_count[rn_type][ip_pos] -= data_num
                                    node.rn_rdb[rn_type][ip_pos][req.packet_id] = []
                    elif req_type == "write":
                        if req_network.ip_write[rn_type][ip_pos]:
                            req = req_network.ip_write[rn_type][ip_pos][0]
                            if node.rn_wdb_count[rn_type][ip_pos] >= data_num:
                                if node.rn_tracker_count[req_type][rn_type][ip_pos] > 0:
                                    req_network.ip_write[rn_type][ip_pos].popleft()
                                    node.rn_tracker[req_type][rn_type][ip_pos].append(req)
                                    node.rn_tracker_count[req_type][rn_type][ip_pos] -= 1
                                    node.rn_wdb_count[rn_type][ip_pos] -= data_num
                                    node.rn_wdb[rn_type][ip_pos][req.packet_id] = []
                                    create_write_packet(config, node, req, cycle, routes)
                select_inject_network(config, req_network, node, directions, direction_conditions, rn_type, ip_pos)

        for inject_queues in [
            req_network.inject_queues_right,
            req_network.inject_queues_left,
            req_network.inject_queues_up,
            req_network.inject_queues_local,
        ]:
            num, list = process_inject_queues(req_network, inject_queues, cycle)
            req_num += num
            reqs.extend(list)
        reqs = flit_move(config, req_network, node, reqs, cycle, "req", routes)
        if rn_type != "Idle":
            for ip_pos in getattr(config, f"{rn_type}_send_positions"):
                move_to_inject_queue(req_network, req_network.inject_queues_left_pre, req_network.inject_queues_left, ip_pos, cycle)
                move_to_inject_queue(req_network, req_network.inject_queues_right_pre, req_network.inject_queues_right, ip_pos, cycle)
                move_to_inject_queue(req_network, req_network.inject_queues_up_pre, req_network.inject_queues_up, ip_pos, cycle)
                move_to_inject_queue(req_network, req_network.inject_queues_local_pre, req_network.inject_queues_local, ip_pos, cycle)
        # rsp process
        if sn_type != "Idle":
            for ip_pos in config.ddr_send_positions:
                if node.sn_rsp_queue[sn_type][ip_pos]:
                    rsp = node.sn_rsp_queue[sn_type][ip_pos][0]
                    for direction in directions:
                        queue = getattr(rsp_network, f"inject_queues_{direction}")
                        queue_pre = getattr(rsp_network, f"inject_queues_{direction}_pre")
                        if direction_conditions[direction](rsp) and len(queue[ip_pos]) < config.inject_queues_len:
                            queue_pre[ip_pos] = rsp
                            node.sn_rsp_queue[sn_type][ip_pos].pop(0)
        for inject_queues in [
            rsp_network.inject_queues_right,
            rsp_network.inject_queues_left,
            rsp_network.inject_queues_up,
            rsp_network.inject_queues_local,
        ]:
            num, list = process_inject_queues(rsp_network, inject_queues, cycle)
            rsp_num += num
            rsps.extend(list)
        rsps = flit_move(config, rsp_network, node, rsps, cycle, "rsp", routes)
        if sn_type != "Idle":
            for ip_pos in config.ddr_send_positions:
                move_to_inject_queue(rsp_network, rsp_network.inject_queues_left_pre, rsp_network.inject_queues_left, ip_pos, cycle)
                move_to_inject_queue(rsp_network, rsp_network.inject_queues_right_pre, rsp_network.inject_queues_right, ip_pos, cycle)
                move_to_inject_queue(rsp_network, rsp_network.inject_queues_up_pre, rsp_network.inject_queues_up, ip_pos, cycle)
                move_to_inject_queue(rsp_network, rsp_network.inject_queues_local_pre, rsp_network.inject_queues_local, ip_pos, cycle)
        # data process
        if sn_type != "Idle":
            for ip_pos in set(config.ddr_send_positions + config.sdma_send_positions + config.gdma_send_positions):
                inject_flits = [
                    (
                        node.sn_rdb[sn_type][ip_pos][0]
                        if node.sn_rdb[sn_type][ip_pos] and node.sn_rdb[sn_type][ip_pos][0].departure_cycle <= cycle
                        else None
                    ),
                    node.rn_wdb[rn_type][ip_pos][node.rn_wdb_send[rn_type][ip_pos][0]][0] if len(node.rn_wdb_send[rn_type][ip_pos]) > 0 else None,
                ]
                for direction in directions:
                    rr_index = flit_network.inject_queue_rr[direction][cycle_mod][ip_pos]
                    for i in rr_index:
                        flit = inject_flits[i]
                        if flit is not None:
                            queue = getattr(flit_network, f"inject_queues_{direction}")
                            queue_pre = getattr(flit_network, f"inject_queues_{direction}_pre")
                            if direction_conditions[direction](flit) and len(queue[ip_pos]) < config.inject_queues_len:
                                queue_pre[flit.source] = flit
                                if i == 0:
                                    send_flits_num += 1
                                    trans_flits_num += 1
                                    node.sn_rdb[sn_type][ip_pos].pop(0)
                                    if flit.id % data_num == data_num - 1:
                                        req = next((req for req in node.sn_tracker[sn_type][ip_pos] if req.packet_id == flit.packet_id), None)
                                        node.sn_tracker[sn_type][ip_pos].remove(req)
                                        node.sn_tracker_count[sn_type][req.sn_tracker_type][ip_pos] += 1
                                        if node.sn_req_wait["read"][sn_type][ip_pos]:
                                            new_req = node.sn_req_wait["read"][sn_type][ip_pos].pop(0)
                                            new_req.sn_tracker_type = req.sn_tracker_type
                                            new_req.req_attr = "old"
                                            node.sn_tracker[sn_type][ip_pos].append(new_req)
                                            node.sn_tracker_count[sn_type][req.sn_tracker_type][ip_pos] -= 1
                                            create_rsp(config, node, new_req, cycle, routes, "positive")
                                else:
                                    send_flits_num += 1
                                    trans_flits_num += 1
                                    node.rn_wdb[rn_type][ip_pos][node.rn_wdb_send[rn_type][ip_pos][0]].pop(0)
                                    if flit.id % data_num == data_num - 1:
                                        req = next(
                                            (req for req in node.rn_tracker["write"][rn_type][ip_pos] if req.packet_id == flit.packet_id), None
                                        )
                                        node.rn_tracker["write"][rn_type][ip_pos].remove(req)
                                        node.rn_tracker_count["write"][rn_type][ip_pos] += 1
                                        node.rn_tracker_pointer["write"][rn_type][ip_pos] -= 1
                                        node.rn_wdb_send[rn_type][ip_pos].pop(0)
                                        node.rn_wdb[rn_type][ip_pos].pop(req.packet_id)
                                        node.rn_wdb_count[rn_type][ip_pos] += data_num
                                inject_flits[i] = None
                                # rr_index.remove(i)
                                # rr_index.append(i)
                                break
        for inject_queues in [
            flit_network.inject_queues_right,
            flit_network.inject_queues_left,
            flit_network.inject_queues_up,
            flit_network.inject_queues_local,
        ]:
            num, list = process_inject_queues(flit_network, inject_queues, cycle)
            flit_num += num
            flits.extend(list)
        flits = flit_move(config, flit_network, node, flits, cycle, "data", routes)
        for ip_pos in set(config.ddr_send_positions + config.sdma_send_positions + config.l2m_send_positions + config.gdma_send_positions):
            move_to_inject_queue(flit_network, flit_network.inject_queues_left_pre, flit_network.inject_queues_left, ip_pos, cycle)
            move_to_inject_queue(flit_network, flit_network.inject_queues_right_pre, flit_network.inject_queues_right, ip_pos, cycle)
            move_to_inject_queue(flit_network, flit_network.inject_queues_up_pre, flit_network.inject_queues_up, ip_pos, cycle)
            move_to_inject_queue(flit_network, flit_network.inject_queues_local_pre, flit_network.inject_queues_local, ip_pos, cycle)
        tag_move(config, flit_network, cycle)
        if rn_type != "Idle":
            for in_pos in set(config.ddr_send_positions + config.sdma_send_positions + config.l2m_send_positions + config.gdma_send_positions):
                if len(node.rn_rdb_recv[rn_type][in_pos]) > 0:
                    packet_id = node.rn_rdb_recv[rn_type][in_pos][0]
                    node.rn_rdb[rn_type][in_pos][packet_id].pop(0)
                    if len(node.rn_rdb[rn_type][in_pos][packet_id]) == 0:
                        node.rn_rdb[rn_type][in_pos].pop(packet_id)
                        node.rn_rdb_recv[rn_type][in_pos].pop(0)
                        node.rn_rdb_count[rn_type][in_pos] += data_num
                        req = next((req for req in node.rn_tracker["read"][rn_type][in_pos] if req.packet_id == packet_id), None)
                        node.rn_tracker["read"][rn_type][in_pos].remove(req)
                        node.rn_tracker_count["read"][rn_type][in_pos] += 1
                        node.rn_tracker_pointer["read"][rn_type][in_pos] -= 1
                if len(node.sn_wdb_recv[sn_type][in_pos]) > 0:
                    packet_id = node.sn_wdb_recv[sn_type][in_pos][0]
                    node.sn_wdb[sn_type][in_pos][packet_id].pop(0)
                    if len(node.sn_wdb[sn_type][in_pos][packet_id]) == 0:
                        node.sn_wdb[sn_type][in_pos].pop(packet_id)
                        node.sn_wdb_recv[sn_type][in_pos].pop(0)
                        node.sn_wdb_count[sn_type][in_pos] += data_num
                        req = next((req for req in node.sn_tracker[sn_type][in_pos] if req.packet_id == packet_id), None)
                        node.sn_tracker[sn_type][in_pos].remove(req)
                        node.sn_tracker_count[sn_type][req.sn_tracker_type][in_pos] += 1
                        if node.sn_req_wait["write"][sn_type][in_pos]:
                            new_req = node.sn_req_wait["write"][sn_type][in_pos].pop(0)
                            new_req.sn_tracker_type = req.sn_tracker_type
                            new_req.req_attr = "old"
                            node.sn_tracker[sn_type][in_pos].append(new_req)
                            node.sn_tracker_count[sn_type][new_req.sn_tracker_type][in_pos] -= 1
                            node.sn_wdb[sn_type][in_pos][new_req.packet_id] = []
                            node.sn_wdb_count[sn_type][in_pos] -= data_num
                            create_rsp(config, node, new_req, cycle, routes, "positive")

        if len(flits) > 0 and begin == None:
            begin = cycle
        if len(flits) == 0 and begin != None and end == None:
            end = cycle
            throughput_time.append(trans_flits_num)
            trans_flits_num = 0
            begin, end = None, None
        if cycle > 0 and cycle % 1000 == 0:
            print(
                f"{cycle}, Flit count: {flit_num}, Request Count: {req_count}, Request Count: {req_num}, Response Count: {rsp_num}, Sent reqs: {send_reqs_num}, Sent flits: {send_flits_num}, Transferred flits: {trans_flits_num}, Received flits: {flit_network.recv_flits_num}"
            )
        # if req_num == read_req + write_req and send_flits_num == flit_network.recv_flits_num:
        #     break
        if (
            len(flits) == 0
            and all_queues_empty(flit_network)
            # and cycle >= req_stream[-1][0] * 2
            and is_ip_eject_empty(flit_network)
            and cycle > 100000
        ):
            break

    print(
        f"{cycle}, Flit count: {flit_num}, Request Count: {req_count}, Request Count: {req_num}, Response Count: {rsp_num}, Sent reqs: {send_reqs_num}, Sent flits: {send_flits_num}, Transferred flits: {trans_flits_num}, Received flits: {flit_network.recv_flits_num}"
    )
    performance_evaluate(config, flit_network, cycle, throughput_time)
    print("finish!")


def simulate_transmission(config, routes, flit_network, req_network, rsp_network):
    specific_dataflow(config, routes, flit_network)
    # draw_figure(config, flit_network)
    # read_dataflow(config, routes, req_network, rsp_network, flit_network)
    # draw_figure(config, flit_network)


def main():
    # 参数设置
    config = SimulationConfig(
        num_nodes=128,
        rows=8,
        num_cycles_total=8000,
        num_cycles_send=64,
        num_round_cycles=64,
        ddr_send_rate=76,
        sdma_send_rate=64,
        l2m_send_rate=128,
        gdma_send_rate=64,
        num_ddr=64,
        num_sdma=64,
        num_l2m=64,
        num_gdma=64,
        ddr_send_positions=[int(16 * (x // 8) + 8 + x % 8) for x in range(64)],
        sdma_send_positions=[int(16 * (x // 8) + 8 + x % 8) for x in range(64)],
        l2m_send_positions=[int(16 * (x // 8) + 8 + x % 8) for x in range(64)],
        gdma_send_positions=[int(16 * (x // 8) + 8 + x % 8) for x in range(64)],
        packet_size=128,
        flit_size=128,
        seats_per_link=6,
        seats_per_station=15,
        seats_per_vstation=5,
        inject_queues_len=2,
        eject_queues_len=8,
        ip_eject_len=10,
        wait_cycle_h=100,
        wait_cycle_v=100,
        ft_count=4000000,
        ft_len=2,
        tags_num=1,
        reservation_num=5,
        rn_read_tracker_len=48,
        rn_rdb_len=192,
        rn_write_tracker_len=48,
        rn_wdb_len=192,
        ro_trak_len=48,
        share_trak_len=48,
        sn_wdb_len=192,
        network_frequency=2,
        ddr_latency=300,
    )

    adjacency_matrix = create_adjacency_matrix("CrossRing", 128, 8)
    routes = find_shortest_paths(adjacency_matrix)
    flit_network = Network(config, adjacency_matrix)
    req_network = Network(config, adjacency_matrix)
    rsp_network = Network(config, adjacency_matrix)
    simulate_transmission(config, routes, flit_network, req_network, rsp_network)


if __name__ == "__main__":
    main()
