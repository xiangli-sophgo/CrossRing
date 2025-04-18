import numpy as np
from collections import deque, defaultdict

from src.utils.optimal_placement import create_adjacency_matrix, find_shortest_paths
from config.config import SimulationConfig
from src.utils.component import Flit, Network, Node
import matplotlib.pyplot as plt
import random
import json
import os
import sys, time
import inspect

import cProfile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
import networkx as nx
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.colors as mcolors
import matplotlib.cm as cm


class BaseModel:
    def __init__(self, model_type, config, topo_type, traffic_file_path, file_name, result_save_path=None, results_fig_save_path=None, plot_flow_fig=False):
        self.model_type_stat = model_type
        self.config = config
        self.topo_type_stat = topo_type
        self.traffic_file_path = traffic_file_path
        self.file_name = file_name
        print(f"\nModel Type: {model_type}, Topology: {self.topo_type_stat}, file_name: {self.file_name[:-4]}")

        self.result_save_path_original = result_save_path
        self.plot_flow_fig = plot_flow_fig
        self.results_fig_save_path = None
        if result_save_path:
            self.result_save_path = self.result_save_path_original + str(topo_type) + "/" + self.file_name[:-4] + "/"
            if not os.path.exists(self.result_save_path):
                os.makedirs(self.result_save_path)
        if results_fig_save_path:
            self.results_fig_save_path = results_fig_save_path
            if not os.path.exists(self.results_fig_save_path):
                os.makedirs(self.results_fig_save_path)
        self.config.topology_select(self.topo_type_stat)
        self.config.update_config()
        # self.initial()

    def initial(self):
        self.adjacency_matrix = create_adjacency_matrix("CrossRing", self.config.num_nodes, self.config.cols)
        # plot_adjacency_matrix(self.adjacency_matrix)
        self.req_network = Network(self.config, self.adjacency_matrix, name="Request Network")
        self.rsp_network = Network(self.config, self.adjacency_matrix, name="Response Network")
        self.flit_network = Network(self.config, self.adjacency_matrix, name="Data Network")
        if self.config.Both_side_ETag_upgrade:
            self.req_network.Both_side_ETag_upgrade = self.rsp_network.Both_side_ETag_upgrade = self.flit_network.Both_side_ETag_upgrade = True
        self.routes = find_shortest_paths(self.adjacency_matrix)
        self.node = Node(self.config)
        self.flits = []
        self.throughput_time = []
        self.data_count = 0
        self.req_count = 0
        self.flit_id_count = 0
        self.send_flits_num = 0
        self.send_reqs_num = 0
        self.trans_flits_num = 0
        self.end_time = np.inf
        self.print_interval = 5000
        self.flit_num, self.req_num, self.rsp_num = 0, 0, 0
        self.new_write_req = []
        self.directions = ["right", "left", "up", "local"]
        self.direction_conditions = {
            "right": lambda flit: flit.path[1] - flit.path[0] == 1,
            "left": lambda flit: flit.path[1] - flit.path[0] == -1,
            "up": lambda flit: flit.path[1] - flit.path[0] == -self.config.cols and flit.source - flit.destination != self.config.cols,
            "local": lambda flit: flit.source - flit.destination == self.config.cols,
        }
        self.flit_position = set(self.config.ddr_send_positions + self.config.sdma_send_positions + self.config.l2m_send_positions + self.config.gdma_send_positions)
        self.read_ip_intervals = defaultdict(list)  # 存储每个IP的读请求时间区间
        self.write_ip_intervals = defaultdict(list)  # 存储每个IP的写请求时间区间

        # statistical data
        self.send_read_flits_num_stat = 0
        self.send_write_flits_num_stat = 0
        self.rn_send_num_stat = 0
        self.file_name_stat = self.file_name[:-4]
        self.negative_rsp_num_stat, self.positive_rsp_num_stat = 0, 0
        self.R_finish_time_stat, self.W_finish_time_stat = 0, 0
        self.R_tail_latency_stat, self.W_tail_latency_stat = 0, 0
        self.req_cir_h_num_stat, self.req_cir_v_num_stat = 0, 0
        self.rsp_cir_h_num_stat, self.rsp_cir_v_num_stat = 0, 0
        self.data_cir_h_num_stat, self.data_cir_v_num_stat = 0, 0
        self.req_wait_cycle_h_num_stat, self.req_wait_cycle_v_num_stat = 0, 0
        self.rsp_wait_cycle_h_num_stat, self.rsp_wait_cycle_v_num_stat = 0, 0
        self.data_wait_cycle_h_num_stat, self.data_wait_cycle_v_num_stat = 0, 0
        self.read_retry_num_stat, self.write_retry_num_stat = 0, 0
        self.EQ_ETag_T1_num_stat, self.EQ_ETag_T0_num_stat = 0, 0
        self.RB_ETag_T1_num_stat, self.RB_ETag_T0_num_stat = 0, 0
        self.ITag_h_num_stat, self.ITag_v_num_stat = 0, 0
        self.read_BW_stat, self.read_latency_avg_stat, self.read_latency_max_stat = 0, 0, 0
        self.write_BW_stat, self.write_latency_avg_stat, self.write_latency_max_stat = 0, 0, 0
        self.Total_BW_stat = 0

    def run(self):
        """Main simulation loop."""
        self.load_request_stream()
        flits, reqs, rsps = [], [], []
        self.cycle = 0
        tail_time = 0

        while True:
            self.cycle += 1
            self.cycle_mod = self.cycle % self.config.network_frequency
            self.rn_type, self.sn_type = self.get_network_types()

            self.check_and_release_sn_tracker()

            # Process requests
            self.process_requests()

            # Inject and process flits for requests
            if self.rn_type != "Idle":
                self.handle_request_injection()

            reqs = self.process_and_move_flits(self.req_network, reqs, "req")

            if self.rn_type != "Idle":
                self.move_all_to_inject_queue(self.req_network, "req")

                # Inject and process responses
                self.handle_response_injection(self.cycle, self.sn_type)

            rsps = self.process_and_move_flits(self.rsp_network, rsps, "rsp")

            if self.sn_type != "Idle":
                self.move_all_to_inject_queue(self.rsp_network, "rsp")

                # Inject and process data flits
                self.handle_data_injection()

            flits = self.process_and_move_flits(self.flit_network, flits, "data")
            # if flits:
            # print(flits)

            self.move_all_to_inject_queue(self.flit_network, "data")

            # Tag moves
            self.tag_move(self.flit_network)

            if self.rn_type != "Idle":
                self.process_received_data()

            # Evaluate throughput time
            self.update_throughput_metrics(flits)

            if self.cycle / self.config.network_frequency % self.print_interval == 0:
                self.log_summary()

            # if len(flits) == 0 and self.all_flit_queues_empty() and self.cycle >= self.req_stream[-1][0] * 2 and self.is_all_ip_eject_empty():
            #     print("Finish!")
            #     break

            if (
                self.req_count >= self.read_req + self.write_req
                and self.send_flits_num == self.flit_network.recv_flits_num >= self.read_flit + self.write_flit
                and self.trans_flits_num == 0
                and not self.new_write_req
                or self.cycle > self.end_time * self.config.network_frequency
                # or self.cycle > 60000 * self.config.network_frequency
            ):
                if tail_time == 0:
                    print("Finish!")
                    break
                else:
                    tail_time -= 1

        # Performance evaluation
        self.print_data_statistic()
        self.log_summary()
        self.evaluate_results(self.flit_network)

    def all_flit_queues_empty(self):
        return (
            all(len(queue) == 0 for queue in self.flit_network.eject_queues["down"].values())
            and all(len(queue) == 0 for queue in self.flit_network.eject_queues["up"].values())
            and all(len(queue) == 0 for queue in self.flit_network.eject_queues["ring_bridge"].values())
            and all(len(queue) == 0 for queue in self.flit_network.inject_queues["left"].values())
            and all(len(queue) == 0 for queue in self.flit_network.inject_queues["right"].values())
            and all(len(queue) == 0 for queue in self.flit_network.inject_queues["up"].values())
        )

    def is_all_ip_eject_empty(self):
        return not any(any(len(deque) != 0 for deque in self.flit_network.ip_eject[ip_type].values()) for ip_type in ["ddr", "sdma", "l2m", "gdma"])

    def process_received_data(self):
        """Process received data in RN and SN networks."""
        positions = self.flit_position
        for in_pos in positions:
            self.process_rn_received_data(in_pos)
            self.process_sn_received_data(in_pos)

    def process_rn_received_data(self, in_pos):
        """Handle received data in the RN network."""
        if in_pos in self.node.rn_rdb_recv[self.rn_type] and len(self.node.rn_rdb_recv[self.rn_type][in_pos]) > 0:
            packet_id = self.node.rn_rdb_recv[self.rn_type][in_pos][0]
            self.node.rn_rdb[self.rn_type][in_pos][packet_id].pop(0)
            if len(self.node.rn_rdb[self.rn_type][in_pos][packet_id]) == 0:
                self.node.rn_rdb[self.rn_type][in_pos].pop(packet_id)
                self.node.rn_rdb_recv[self.rn_type][in_pos].pop(0)
                self.node.rn_rdb_count[self.rn_type][in_pos] += self.req_network.send_flits[packet_id][0].burst_length
                req = next(
                    (req for req in self.node.rn_tracker["read"][self.rn_type][in_pos] if req.packet_id == packet_id),
                    None,
                )
                self.req_cir_h_num_stat += req.circuits_completed_h
                self.req_cir_v_num_stat += req.circuits_completed_v
                self.req_wait_cycle_h_num_stat += req.wait_cycle_h
                self.req_wait_cycle_v_num_stat += req.wait_cycle_v
                for flit in self.flit_network.arrive_flits[packet_id]:
                    flit.leave_db_cycle = self.cycle
                self.node.rn_tracker["read"][self.rn_type][in_pos].remove(req)
                self.node.rn_tracker_count["read"][self.rn_type][in_pos] += 1
                self.node.rn_tracker_pointer["read"][self.rn_type][in_pos] -= 1

    def process_sn_received_data(self, in_pos):
        """Handle received data in the SN network."""
        if in_pos in self.node.sn_wdb_recv[self.sn_type] and len(self.node.sn_wdb_recv[self.sn_type][in_pos]) > 0:
            packet_id = self.node.sn_wdb_recv[self.sn_type][in_pos][0]
            self.node.sn_wdb[self.sn_type][in_pos][packet_id].pop(0)
            if len(self.node.sn_wdb[self.sn_type][in_pos][packet_id]) == 0:
                self.node.sn_wdb[self.sn_type][in_pos].pop(packet_id)
                self.node.sn_wdb_recv[self.sn_type][in_pos].pop(0)
                self.node.sn_wdb_count[self.sn_type][in_pos] += self.req_network.send_flits[packet_id][0].burst_length
                req = next(
                    (req for req in self.node.sn_tracker[self.sn_type][in_pos] if req.packet_id == packet_id),
                    None,
                )
                self.req_cir_h_num_stat += req.circuits_completed_h
                self.req_cir_v_num_stat += req.circuits_completed_v
                for flit in self.flit_network.arrive_flits[packet_id]:
                    flit.leave_db_cycle = self.cycle + self.config.sn_tracker_release_latency
                # 释放tracker 增加40ns
                release_time = self.cycle + self.config.sn_tracker_release_latency
                self.node.sn_tracker_release_time[release_time].append((self.sn_type, in_pos, req))
                # self.node.sn_tracker[self.sn_type][in_pos].remove(req)
                # self.node.sn_tracker_count[self.sn_type][req.sn_tracker_type][in_pos] += 1
                if self.node.sn_wdb_count[self.sn_type][in_pos] > 0 and self.node.sn_req_wait["write"][self.sn_type][in_pos]:
                    new_req = self.node.sn_req_wait["write"][self.sn_type][in_pos].pop(0)
                    new_req.sn_tracker_type = req.sn_tracker_type
                    new_req.req_attr = "old"
                    self.node.sn_tracker[self.sn_type][in_pos].append(new_req)
                    self.node.sn_tracker_count[self.sn_type][new_req.sn_tracker_type][in_pos] -= 1
                    self.node.sn_wdb[self.sn_type][in_pos][new_req.packet_id] = []
                    self.node.sn_wdb_count[self.sn_type][in_pos] -= new_req.burst_length
                    self.create_rsp(new_req, "positive")

    def check_and_release_sn_tracker(self):
        """Check if any trackers can be released based on the current cycle."""
        for release_time in sorted(self.node.sn_tracker_release_time.keys()):
            if release_time > self.cycle:
                return
            tracker_list = self.node.sn_tracker_release_time.pop(release_time)
            for sn_type, in_pos, req in tracker_list:
                self.node.sn_tracker[sn_type][in_pos].remove(req)
                self.node.sn_tracker_count[sn_type][req.sn_tracker_type][in_pos] += 1

    def move_all_to_inject_queue(self, network, network_type):
        """Move all items from pre-injection queues to injection queues for a given network."""
        if network_type == "req":
            positions = getattr(self.config, f"{self.rn_type}_send_positions")
        elif network_type == "rsp":
            positions = getattr(self.config, f"{self.sn_type}_send_positions")
        elif network_type == "data":
            positions = set(getattr(self.config, f"{self.rn_type}_send_positions") + getattr(self.config, f"{self.sn_type}_send_positions"))

        for ip_pos in positions:
            for direction in self.directions:
                pre_queue = network.inject_queues_pre[direction]
                queue = network.inject_queues[direction]
                self.move_to_inject_queue(network, pre_queue, queue, ip_pos)

    def print_data_statistic(self):
        print(f"Data statistic: Read: {self.read_req, self.read_flit}, " f"Write: {self.write_req, self.write_flit}, " f"Total: {self.read_req + self.write_req, self.read_flit + self.write_flit}")

    def log_summary(self):
        print(
            f"T: {self.cycle // self.config.network_frequency}, Req_cnt: {self.req_count} In_Req: {self.req_num}, Rsp: {self.rsp_num},"
            # f"" Sent flits: {self.send_flits_num}, "
            f" R_fn: {self.send_read_flits_num_stat}, W_fn: {self.send_write_flits_num_stat}, "
            f"Trans_fn: {self.trans_flits_num}, Recv_fn: {self.flit_network.recv_flits_num}"
        )

    def handle_request_injection(self):
        """Inject requests into the network."""
        for ip_pos in getattr(self.config, f"{self.rn_type}_send_positions"):
            for req_type in ["read", "write"]:
                if req_type == "read":
                    if self.req_network.ip_read[self.rn_type][ip_pos]:
                        req = self.req_network.ip_read[self.rn_type][ip_pos][0]
                        if (
                            self.node.rn_rdb_count[self.rn_type][ip_pos] > self.node.rn_rdb_reserve[self.rn_type][ip_pos] * req.burst_length
                            and self.node.rn_tracker_count[req_type][self.rn_type][ip_pos] > 0
                        ):
                            # req.entry_db_cycle = self.cycle
                            self.req_network.ip_read[self.rn_type][ip_pos].popleft()
                            self.node.rn_tracker[req_type][self.rn_type][ip_pos].append(req)
                            self.node.rn_tracker_count[req_type][self.rn_type][ip_pos] -= 1
                            self.node.rn_rdb_count[self.rn_type][ip_pos] -= req.burst_length
                            self.node.rn_rdb[self.rn_type][ip_pos][req.packet_id] = []
                elif req_type == "write":
                    if self.req_network.ip_write[self.rn_type][ip_pos]:
                        req = self.req_network.ip_write[self.rn_type][ip_pos][0]
                        if self.node.rn_wdb_count[self.rn_type][ip_pos] >= req.burst_length and self.node.rn_tracker_count[req_type][self.rn_type][ip_pos] > 0:
                            self.req_network.ip_write[self.rn_type][ip_pos].popleft()
                            self.node.rn_tracker[req_type][self.rn_type][ip_pos].append(req)
                            self.node.rn_tracker_count[req_type][self.rn_type][ip_pos] -= 1
                            self.node.rn_wdb_count[self.rn_type][ip_pos] -= req.burst_length
                            self.node.rn_wdb[self.rn_type][ip_pos][req.packet_id] = []
                            self.create_write_packet(req)
            self.select_inject_network(ip_pos)

    def process_and_move_flits(self, network, flits, flit_type):
        """Process injection queues and move flits."""
        for inject_queues in network.inject_queues.values():
            num, moved_flits = self.process_inject_queues(network, inject_queues)
            if num == 0 and not moved_flits:
                continue
            if flit_type == "req":
                self.req_num += num
            elif flit_type == "rsp":
                self.rsp_num += num
            elif flit_type == "data":
                self.flit_num += num
            flits.extend(moved_flits)
        flits = self.flit_move(network, flits, flit_type)
        return flits

    def handle_response_injection(self, cycle, sn_type):
        """Inject responses into the network."""
        for ip_pos in getattr(self.config, f"{self.sn_type}_send_positions"):
            if self.node.sn_rsp_queue[sn_type][ip_pos]:
                rsp = self.node.sn_rsp_queue[sn_type][ip_pos][0]
                for direction in self.directions:
                    queue = self.rsp_network.inject_queues[direction]
                    queue_pre = self.rsp_network.inject_queues_pre[direction]
                    if self.direction_conditions[direction](rsp) and len(queue[ip_pos]) < self.config.IQ_OUT_FIFO_DEPTH:
                        queue_pre[ip_pos] = rsp
                        self.node.sn_rsp_queue[sn_type][ip_pos].pop(0)

    def handle_data_injection(self):
        """
        Inject data flits into the network.
        """
        for ip_pos in self.flit_position:
            inject_flits = [
                (self.node.sn_rdb[self.sn_type][ip_pos][0] if self.node.sn_rdb[self.sn_type][ip_pos] and self.node.sn_rdb[self.sn_type][ip_pos][0].departure_cycle <= self.cycle else None),
                (self.node.rn_wdb[self.rn_type][ip_pos][self.node.rn_wdb_send[self.rn_type][ip_pos][0]][0] if len(self.node.rn_wdb_send[self.rn_type][ip_pos]) > 0 else None),
            ]
            for direction in self.directions:
                rr_index = self.flit_network.inject_queue_rr[direction][self.cycle_mod][ip_pos]
                for i in rr_index:
                    if flit := inject_flits[i]:
                        queue = self.flit_network.inject_queues[direction]
                        queue_pre = self.flit_network.inject_queues_pre[direction]
                        if self.direction_conditions[direction](flit) and len(queue[ip_pos]) < self.config.IQ_OUT_FIFO_DEPTH:
                            queue_pre[flit.source] = flit
                            self.send_flits_num += 1
                            self.trans_flits_num += 1
                            if i == 0:
                                self.send_read_flits_num_stat += 1
                                self.node.sn_rdb[self.sn_type][ip_pos].pop(0)
                                if len(self.flit_network.arrive_flits[flit.packet_id]) == flit.burst_length:
                                    # finish current req injection
                                    req = next(
                                        (req for req in self.node.sn_tracker[self.sn_type][ip_pos] if req.packet_id == flit.packet_id),
                                        None,
                                    )
                                    self.node.sn_tracker[self.sn_type][ip_pos].remove(req)
                                    self.node.sn_tracker_count[self.sn_type][req.sn_tracker_type][ip_pos] += 1
                                    if self.node.sn_req_wait["read"][self.sn_type][ip_pos]:
                                        # If there is a waiting request, inject it
                                        new_req = self.node.sn_req_wait["read"][self.sn_type][ip_pos].pop(0)
                                        new_req.sn_tracker_type = req.sn_tracker_type
                                        new_req.req_attr = "old"
                                        self.node.sn_tracker[self.sn_type][ip_pos].append(new_req)
                                        self.node.sn_tracker_count[self.sn_type][req.sn_tracker_type][ip_pos] -= 1
                                        self.create_rsp(new_req, "positive")
                            else:
                                self.send_write_flits_num_stat += 1
                                if flit.flit_id_in_packet == 0:
                                    for f in self.node.rn_wdb[self.rn_type][ip_pos][flit.packet_id]:
                                        f.entry_db_cycle = self.cycle
                                self.node.rn_wdb[self.rn_type][ip_pos][flit.packet_id].pop(0)
                                # if flit.is_last_flit:
                                if len(self.flit_network.arrive_flits[flit.packet_id]) == flit.burst_length:
                                    # finish current req injection
                                    req = next(
                                        (req for req in self.node.rn_tracker["write"][self.rn_type][ip_pos] if req.packet_id == flit.packet_id),
                                        None,
                                    )
                                    self.node.rn_tracker["write"][self.rn_type][ip_pos].remove(req)
                                    self.node.rn_tracker_count["write"][self.rn_type][ip_pos] += 1
                                    self.node.rn_tracker_pointer["write"][self.rn_type][ip_pos] -= 1
                                    self.node.rn_wdb_send[self.rn_type][ip_pos].pop(0)
                                    self.node.rn_wdb[self.rn_type][ip_pos].pop(req.packet_id)
                                    self.node.rn_wdb_count[self.rn_type][ip_pos] += req.burst_length
                            inject_flits[i] = None
                            break

    def update_throughput_metrics(self, flits):
        """Update throughput metrics based on flit counts."""
        # if len(flits) > 0 and self.begin is None:
        #     self.begin = self.cycle
        self.trans_flits_num = len(flits)
        # if len(flits) == 0:
        #     self.throughput_time.append(self.trans_flits_num)
        #     self.trans_flits_num = 0
        # self.end = self.cycle
        # self.begin, self.end = None, None

    def load_request_stream(self):
        # self.req_stream = []
        self.read_req, self.write_req = 0, 0
        self.read_flit, self.write_flit = 0, 0
        with open(self.traffic_file_path + self.file_name, "r") as file:
            for line in file:
                split_line = list(line.strip().split(","))
                # TODO: network frequence change
                split_line = [
                    # request cycle
                    int(split_line[0]) * self.config.network_frequency,
                    int(split_line[1]),  # source id
                    split_line[2],  # source type
                    int(split_line[3]),  # destination id
                    split_line[4],  # destination type
                    split_line[5],  # request type
                    int(split_line[6]),  # burst length
                ]
                if split_line[5] == "R":
                    self.read_req += 1
                    self.read_flit += split_line[6]
                elif split_line[5] == "W":
                    self.write_req += 1
                    self.write_flit += split_line[6]
                    # self.req_stream.append(split_line)
        self.print_data_statistic()
        self.req_stream = self._load_requests_stream()
        self.next_req = None  # 缓存未处理的请求

    def _load_requests_stream(self):
        """从文件生成请求流（按时间排序）"""
        with open(self.traffic_file_path + self.file_name, "r") as f:
            for line in f:
                # 解析每行数据为元组（根据实际格式调整）
                split_line = line.strip().split(",")
                yield (
                    # request cycle
                    int(split_line[0]) * self.config.network_frequency,
                    int(split_line[1]),  # source id
                    split_line[2],  # source type
                    int(split_line[3]),  # destination id
                    split_line[4],  # destination type
                    split_line[5],  # request type
                    int(split_line[6]),  # burst length
                )

    def get_network_types(self):
        type_mapping = {0: ("sdma", "ddr"), 1: ("gdma", "l2m")}
        return type_mapping.get(self.cycle_mod, ("Idle", "Idle"))

    def error_log(self, flit, target_id):
        if flit and flit.packet_id == target_id:
            print(
                # inspect.currentframe().f_back.f_code.co_name,  # 调用函数名称
                self.cycle,
                flit,
            )

    def flit_trace(self, packet_id):
        # if self.cycle % 1 == 0 and self.flit_network.send_flits[packet_id] and self.flit_network.send_flits[packet_id][0].current_link is not None:
        if self.cycle % 1 == 0 and self.req_network.send_flits[packet_id]:
            # print(self.cycle, self.req_network.send_flits[packet_id], self.rsp_network.send_flits[packet_id], len(self.flit_network.arrive_flits[packet_id]))
            print(self.cycle, self.req_network.send_flits[packet_id], self.rsp_network.send_flits[packet_id], self.flit_network.send_flits[packet_id])
            time.sleep(0.3)

    def process_requests(self):
        while self.new_write_req and self.new_write_req[0].departure_cycle <= self.cycle:
            req = self.new_write_req[0]
            self.req_network.send_flits[req.packet_id].append(req)
            self.req_network.ip_write[req.source_type][req.source].append(req)
            self.req_count += 1
            self.write_req += 1
            self.new_write_req.pop(0)
            self.write_flit += req.burst_length

        while True:
            # 获取下一个请求（如果缓存中没有）
            if self.next_req is None:
                try:
                    self.next_req = next(self.req_stream)
                except StopIteration:
                    break  # 无更多请求

            # 检查请求时间是否已到
            if self.next_req[0] > self.cycle:
                break  # 等待下一周期

            req_data = self.next_req
            source = self.node_map(req_data[1])
            destination = self.node_map(req_data[3], False)
            path = self.routes[source][destination]
            req = Flit(source, destination, path)
            req.source_original = req_data[1]
            req.destination_original = req_data[3]
            req.flit_type = "req"
            req.departure_cycle = req_data[0]
            req.burst_length = req_data[6]
            req.source_type = req_data[2]
            req.destination_type = req_data[4][:3]
            req.original_source_type = req_data[2]
            req.original_destination_type = req_data[4]
            if self.topo_type_stat in ["5x4", "4x5"]:
                req.source_type = "gdma" if req_data[1] < 16 else "sdma"
                req.destination_type = "ddr" if req_data[3] < 16 else "l2m"
            req.packet_id = Node.get_next_packet_id()
            req.req_type = "read" if req_data[5] == "R" else "write"
            self.req_network.send_flits[req.packet_id].append(req)
            if req.req_type == "read":
                self.req_network.ip_read[req.source_type][req.source].append(req)
                self.R_tail_latency_stat = req_data[0]
            if req.req_type == "write":
                self.req_network.ip_write[req.source_type][req.source].append(req)
                self.W_tail_latency_stat = req_data[0]

            # 重置缓存并更新计数
            self.next_req = None
            self.req_count += 1

    def select_inject_network(self, ip_pos):
        read_old = self.node.rn_rdb_reserve[self.rn_type][ip_pos] > 0 and self.node.rn_rdb_count[self.rn_type][ip_pos] > self.config.burst
        read_new = len(self.node.rn_tracker["read"][self.rn_type][ip_pos]) - 1 > self.node.rn_tracker_pointer["read"][self.rn_type][ip_pos]
        write_old = self.node.rn_wdb_reserve[self.rn_type][ip_pos] > 0
        write_new = len(self.node.rn_tracker["write"][self.rn_type][ip_pos]) - 1 > self.node.rn_tracker_pointer["write"][self.rn_type][ip_pos]
        read_valid = read_old or read_new
        write_valid = write_old or write_new
        if read_valid and write_valid:
            if self.req_network.last_select[self.rn_type][ip_pos] == "write":
                if read_old:
                    if req := next(
                        (req for req in self.node.rn_tracker_wait["read"][self.rn_type][ip_pos] if req.req_state == "valid"),
                        None,
                    ):
                        for direction in self.directions:
                            queue_pre = self.req_network.inject_queues_pre[direction]
                            queue = self.req_network.inject_queues[direction]
                            if self.direction_conditions[direction](req) and len(queue[ip_pos]) < self.config.IQ_OUT_FIFO_DEPTH:
                                queue_pre[ip_pos] = req
                                self.node.rn_tracker_wait["read"][self.rn_type][ip_pos].remove(req)
                                self.node.rn_rdb_reserve[self.rn_type][ip_pos] -= 1
                                self.node.rn_rdb_count[self.rn_type][ip_pos] -= req.burst_length
                                self.node.rn_rdb[self.rn_type][ip_pos][req.packet_id] = []
                                self.req_network.last_select[self.rn_type][ip_pos] = "read"
                else:
                    rn_tracker_pointer = self.node.rn_tracker_pointer["read"][self.rn_type][ip_pos] + 1
                    if req := self.node.rn_tracker["read"][self.rn_type][ip_pos][rn_tracker_pointer]:
                        for direction in self.directions:
                            queue = self.req_network.inject_queues[direction]
                            queue_pre = self.req_network.inject_queues_pre[direction]
                            if self.direction_conditions[direction](req) and len(queue[ip_pos]) < self.config.IQ_OUT_FIFO_DEPTH:
                                queue_pre[ip_pos] = req
                                self.node.rn_tracker_pointer["read"][self.rn_type][ip_pos] += 1
                                self.req_network.last_select[self.rn_type][ip_pos] = "read"
            elif write_old:
                if req := next(
                    (req for req in self.node.rn_tracker_wait["write"][self.rn_type][ip_pos] if req.req_state == "valid"),
                    None,
                ):
                    for direction in self.directions:
                        queue = self.req_network.inject_queues[direction]
                        queue_pre = self.req_network.inject_queues_pre[direction]
                        if self.direction_conditions[direction](req) and len(queue[ip_pos]) < self.config.IQ_OUT_FIFO_DEPTH:
                            queue_pre[ip_pos] = req
                            self.node.rn_tracker_wait["write"][self.rn_type][ip_pos].remove(req)
                            self.node.rn_wdb_reserve[self.rn_type][ip_pos] -= 1
                            self.req_network.last_select[self.rn_type][ip_pos] = "write"
            else:
                rn_tracker_pointer = self.node.rn_tracker_pointer["write"][self.rn_type][ip_pos] + 1
                if req := self.node.rn_tracker["write"][self.rn_type][ip_pos][rn_tracker_pointer]:
                    for direction in self.directions:
                        queue = self.req_network.inject_queues[direction]
                        queue_pre = self.req_network.inject_queues_pre[direction]
                        if self.direction_conditions[direction](req) and len(queue[ip_pos]) < self.config.IQ_OUT_FIFO_DEPTH:
                            queue_pre[ip_pos] = req
                            self.node.rn_tracker_pointer["write"][self.rn_type][ip_pos] += 1
                            self.req_network.last_select[self.rn_type][ip_pos] = "write"
        elif read_valid:
            if read_old:
                if req := next(
                    (req for req in self.node.rn_tracker_wait["read"][self.rn_type][ip_pos] if req.req_state == "valid"),
                    None,
                ):
                    for direction in self.directions:
                        queue = self.req_network.inject_queues[direction]
                        queue_pre = self.req_network.inject_queues_pre[direction]
                        if self.direction_conditions[direction](req) and len(queue[ip_pos]) < self.config.IQ_OUT_FIFO_DEPTH:
                            queue_pre[ip_pos] = req
                            self.node.rn_tracker_wait["read"][self.rn_type][ip_pos].remove(req)
                            self.node.rn_rdb_reserve[self.rn_type][ip_pos] -= 1
                            self.node.rn_rdb_count[self.rn_type][ip_pos] -= req.burst_length
                            self.node.rn_rdb[self.rn_type][ip_pos][req.packet_id] = []
                            self.req_network.last_select[self.rn_type][ip_pos] = "read"
            else:
                rn_tracker_pointer = self.node.rn_tracker_pointer["read"][self.rn_type][ip_pos] + 1
                if req := self.node.rn_tracker["read"][self.rn_type][ip_pos][rn_tracker_pointer]:
                    for direction in self.directions:
                        queue = self.req_network.inject_queues[direction]
                        queue_pre = self.req_network.inject_queues_pre[direction]
                        if self.direction_conditions[direction](req) and len(queue[ip_pos]) < self.config.IQ_OUT_FIFO_DEPTH:
                            queue_pre[ip_pos] = req
                            self.node.rn_tracker_pointer["read"][self.rn_type][ip_pos] += 1
                            self.req_network.last_select[self.rn_type][ip_pos] = "read"
        elif write_valid:
            if write_old:
                if req := next(
                    (req for req in self.node.rn_tracker_wait["write"][self.rn_type][ip_pos] if req.req_state == "valid"),
                    None,
                ):
                    for direction in self.directions:
                        queue = self.req_network.inject_queues[direction]
                        queue_pre = self.req_network.inject_queues_pre[direction]
                        if self.direction_conditions[direction](req) and len(queue[ip_pos]) < self.config.IQ_OUT_FIFO_DEPTH:
                            queue_pre[ip_pos] = req
                            self.node.rn_tracker_wait["write"][self.rn_type][ip_pos].remove(req)
                            self.node.rn_wdb_reserve[self.rn_type][ip_pos] -= 1
                            self.req_network.last_select[self.rn_type][ip_pos] = "write"
            else:
                rn_tracker_pointer = self.node.rn_tracker_pointer["write"][self.rn_type][ip_pos] + 1
                if req := self.node.rn_tracker["write"][self.rn_type][ip_pos][rn_tracker_pointer]:
                    for direction in self.directions:
                        queue = self.req_network.inject_queues[direction]
                        queue_pre = self.req_network.inject_queues_pre[direction]
                        if self.direction_conditions[direction](req) and len(queue[ip_pos]) < self.config.IQ_OUT_FIFO_DEPTH:
                            queue_pre[ip_pos] = req
                            self.node.rn_tracker_pointer["write"][self.rn_type][ip_pos] += 1
                            self.req_network.last_select[self.rn_type][ip_pos] = "write"

    def move_to_inject_queue(self, network, queue_pre, queue, ip_pos):
        if queue_pre[ip_pos]:
            queue_pre[ip_pos].departure_inject_cycle = self.cycle
            queue[ip_pos].append(queue_pre[ip_pos])
            queue_pre[ip_pos] = None

    def classify_flits(self, flits):
        ring_bridge_EQ_flits, vertical_flits, horizontal_flits, new_flits, local_flits = [], [], [], [], []
        for flit in flits:
            # if flit.packet_id == 102 and flit.flit_id_in_packet == 0:
            # print(flit, "1")
            if flit.source - flit.destination == self.config.cols:
                flit.is_new_on_network = False
                flit.is_arrive = True
                local_flits.append(flit)
            elif not flit.current_link:
                new_flits.append(flit)
            elif flit.current_link[0] - flit.current_link[1] == self.config.cols and flit.current_link[1] == flit.destination:
                ring_bridge_EQ_flits.append(flit)
            elif abs(flit.current_link[0] - flit.current_link[1]) == 1:
                # 横向环
                horizontal_flits.append(flit)
            else:
                # 纵向环
                vertical_flits.append(flit)
        return ring_bridge_EQ_flits, vertical_flits, horizontal_flits, new_flits, local_flits

    def flit_move(self, network, flits, flit_type):
        # 分类不同类型的flits
        ring_bridge_EQ_flits, vertical_flits, horizontal_flits, new_flits, local_flits = self.classify_flits(flits)

        # 处理新到达的flits
        for flit in new_flits + horizontal_flits:
            network.plan_move(flit)
            if network.execute_moves(flit, self.cycle):
                flits.remove(flit)

        # 处理transfer station的flits
        for col in range(1, self.config.rows, 2):
            for row in range(self.config.cols):
                pos = col * self.config.cols + row
                next_pos = pos - self.config.cols
                eject_flit, vup_flit, vdown_flit = None, None, None

                # 获取各方向的flit
                # station_flits = [
                #     (network.ring_bridge["up"][(pos, next_pos)][0] if network.ring_bridge["up"][(pos, next_pos)] else None),
                #     (network.ring_bridge["left"][(pos, next_pos)][0] if network.ring_bridge["left"][(pos, next_pos)] else None),
                #     (network.ring_bridge["right"][(pos, next_pos)][0] if network.ring_bridge["right"][(pos, next_pos)] else None),
                #     (network.ring_bridge["ft"][(pos, next_pos)][0] if network.ring_bridge["ft"][(pos, next_pos)] else None),
                # ]
                station_flits = [network.ring_bridge[fifo_pos][(pos, next_pos)][0] if network.ring_bridge[fifo_pos][(pos, next_pos)] else None for fifo_pos in ["up", "left", "right", "ft"]]
                # if not all(flit is None for flit in station_flits):
                #     print(station_flits)

                # 处理eject操作
                if len(network.ring_bridge["eject"][(pos, next_pos)]) < self.config.RB_OUT_FIFO_DEPTH:
                    eject_flit = self._process_eject_flit(network, station_flits, pos, next_pos)

                # 处理vup操作
                if len(network.ring_bridge["vup"][(pos, next_pos)]) < self.config.RB_OUT_FIFO_DEPTH:
                    vup_flit = self._process_vup_flit(network, station_flits, pos, next_pos)

                # 处理vdown操作
                if len(network.ring_bridge["vdown"][(pos, next_pos)]) < self.config.RB_OUT_FIFO_DEPTH:
                    vdown_flit = self._process_vdown_flit(network, station_flits, pos, next_pos)

                # transfer_eject
                # 处理eject队列
                if (
                    next_pos in network.eject_queues["ring_bridge"]
                    and len(network.eject_queues["ring_bridge"][next_pos]) < self.config.EQ_IN_FIFO_DEPTH
                    and network.ring_bridge["eject"][(pos, next_pos)]
                ):
                    flit = network.ring_bridge["eject"][(pos, next_pos)].popleft()
                    flit.is_arrive = True

                up_node, down_node = next_pos - self.config.cols * 2, next_pos + self.config.cols * 2
                if up_node < 0:
                    up_node = next_pos
                if down_node >= self.config.num_nodes:
                    down_node = next_pos
                # 处理vup方向
                self._process_ring_bridge(network, "up", pos, next_pos, down_node, up_node)

                # 处理vdown方向
                self._process_ring_bridge(network, "down", pos, next_pos, up_node, down_node)

                if eject_flit:
                    network.ring_bridge["eject"][(pos, next_pos)].append(eject_flit)
                if vup_flit:
                    network.ring_bridge["vup"][(pos, next_pos)].append(vup_flit)
                if vdown_flit:
                    network.ring_bridge["vdown"][(pos, next_pos)].append(vdown_flit)

        # 处理纵向flits的移动
        for flit in vertical_flits:
            network.plan_move(flit)
            if network.execute_moves(flit, self.cycle):
                flits.remove(flit)

        # eject arbitration
        if flit_type in ["req", "rsp", "data"]:
            self._handle_eject_arbitration(network, flit_type)

        # 执行所有flit的移动
        for flit in local_flits:
            if network.execute_moves(flit, self.cycle):
                flits.remove(flit)

        # 处理transfer station的flits
        for flit in ring_bridge_EQ_flits:
            if flit.is_arrive:
                flit.arrival_network_cycle = self.cycle
                if len(network.eject_queues["ring_bridge"][flit.destination]) < self.config.EQ_IN_FIFO_DEPTH:
                    network.eject_queues["ring_bridge"][flit.destination].append(flit)
                    flits.remove(flit)
                else:
                    flit.is_arrive = False
            else:
                network.execute_moves(flit, self.cycle)

        return flits

    def _process_eject_flit(self, network, station_flits, pos, next_pos):
        """处理eject操作"""
        eject_flit = None

        if station_flits[3] and station_flits[3].destination == next_pos:
            eject_flit = station_flits[3]
            station_flits[3] = None
            network.ring_bridge["ft"][(pos, next_pos)].popleft()
        else:
            index = network.round_robin["ring_bridge"][next_pos]
            for i in index:
                if station_flits[i] and station_flits[i].destination == next_pos:
                    eject_flit = station_flits[i]
                    station_flits[i] = None
                    self._update_ring_bridge(network, pos, next_pos, i)
                    break

        return eject_flit

    def _process_vup_flit(self, network, station_flits, pos, next_pos):
        """处理vup操作"""
        vup_flit = None

        if station_flits[3] and station_flits[3].destination < next_pos:
            vup_flit = station_flits[3]
            network.ring_bridge["ft"][(pos, next_pos)].popleft()
        else:
            index = network.round_robin["up"][next_pos]
            for i in index:
                if station_flits[i] and station_flits[i].destination < next_pos:
                    vup_flit = station_flits[i]
                    station_flits[i] = None
                    self._update_ring_bridge(network, pos, next_pos, i)
                    break

        return vup_flit

    def _process_vdown_flit(self, network, station_flits, pos, next_pos):
        """处理vdown操作"""
        vdown_flit = None

        if station_flits[3] and station_flits[3].destination > next_pos:
            vdown_flit = station_flits[3]
            network.ring_bridge["ft"][(pos, next_pos)].popleft()
        else:
            index = network.round_robin["down"][next_pos]
            for i in index:
                if station_flits[i] and station_flits[i].destination > next_pos:
                    vdown_flit = station_flits[i]
                    station_flits[i] = None
                    self._update_ring_bridge(network, pos, next_pos, i)
                    break

        return vdown_flit

    def _update_ring_bridge(self, network, pos, next_pos, index):
        """更新transfer stations"""
        if index == 0:
            flit = network.ring_bridge["up"][(pos, next_pos)].popleft()
        elif index == 1:
            flit = network.ring_bridge["left"][(pos, next_pos)].popleft()
            if flit.ETag_priority == "T0":
                network.RB_UE_Counters["left"][(pos, next_pos)]["T0"] -= 1
            elif flit.ETag_priority == "T1":
                network.RB_UE_Counters["left"][(pos, next_pos)]["T0"] -= 1
                network.RB_UE_Counters["left"][(pos, next_pos)]["T1"] -= 1
            else:
                network.RB_UE_Counters["left"][(pos, next_pos)]["T0"] -= 1
                network.RB_UE_Counters["left"][(pos, next_pos)]["T1"] -= 1
                network.RB_UE_Counters["left"][(pos, next_pos)]["T2"] -= 1
        elif index == 2:
            flit = network.ring_bridge["right"][(pos, next_pos)].popleft()
            if flit.ETag_priority == "T1" or flit.ETag_priority == "T0":
                network.RB_UE_Counters["right"][(pos, next_pos)]["T1"] -= 1
            else:
                network.RB_UE_Counters["right"][(pos, next_pos)]["T1"] -= 1
                network.RB_UE_Counters["right"][(pos, next_pos)]["T2"] -= 1

        if flit.ETag_priority == "T1":
            self.RB_ETag_T1_num_stat += 1
        elif flit.ETag_priority == "T0":
            self.RB_ETag_T0_num_stat += 1

        flit.ETag_priority = "T2"
        network.round_robin["ring_bridge"][next_pos].remove(index)
        network.round_robin["ring_bridge"][next_pos].append(index)

    def _handle_eject_arbitration(self, network, flit_type):
        """处理eject的仲裁逻辑,根据flit类型处理不同的eject队列"""
        if flit_type == "req":
            for in_pos in getattr(self.config, f"{self.rn_type}_send_positions"):
                ip_pos = in_pos - self.config.cols
                # eject_flits = [
                #     network.eject_queues["up"][ip_pos][0] if network.eject_queues["up"][ip_pos] else None,
                #     network.eject_queues["ring_bridge"][ip_pos][0] if network.eject_queues["ring_bridge"][ip_pos] else None,
                #     network.eject_queues["down"][ip_pos][0] if network.eject_queues["down"][ip_pos] else None,
                #     network.eject_queues["local"][ip_pos][0] if network.eject_queues["local"][ip_pos] else None,
                # ]
                eject_flits = [network.eject_queues[fifo_pos][ip_pos][0] if network.eject_queues[fifo_pos][ip_pos] else None for fifo_pos in ["up", "ring_bridge", "down", "local"]]

                # if not all(eject_flit is None for eject_flit in eject_flits):
                #     print(eject_flits)
                eject_flits = self.process_eject_queues(network, eject_flits, network.round_robin["ddr"][ip_pos], "ddr", ip_pos)
                eject_flits = self.process_eject_queues(network, eject_flits, network.round_robin["l2m"][ip_pos], "l2m", ip_pos)

            if self.sn_type != "Idle":
                for in_pos in getattr(self.config, f"{self.sn_type}_send_positions"):
                    ip_pos = in_pos - self.config.cols
                    if network.ip_eject[self.sn_type][ip_pos]:
                        req = network.ip_eject[self.sn_type][ip_pos].popleft()
                        self._handle_request(req, in_pos)

        elif flit_type == "rsp":
            for in_pos in getattr(self.config, f"{self.sn_type}_send_positions"):
                ip_pos = in_pos - self.config.cols
                # eject_flits = [
                #     network.eject_queues["up"][ip_pos][0] if network.eject_queues["up"][ip_pos] else None,
                #     network.eject_queues["ring_bridge"][ip_pos][0] if network.eject_queues["ring_bridge"][ip_pos] else None,
                #     network.eject_queues["down"][ip_pos][0] if network.eject_queues["down"][ip_pos] else None,
                #     network.eject_queues["local"][ip_pos][0] if network.eject_queues["local"][ip_pos] else None,
                # ]
                eject_flits = [network.eject_queues[fifo_pos][ip_pos][0] if network.eject_queues[fifo_pos][ip_pos] else None for fifo_pos in ["up", "ring_bridge", "down", "local"]]

                # if not all(eject_flit is None for eject_flit in eject_flits):
                #     print(eject_flits)
                eject_flits = self.process_eject_queues(network, eject_flits, network.round_robin["sdma"][ip_pos], "sdma", ip_pos)
                eject_flits = self.process_eject_queues(network, eject_flits, network.round_robin["gdma"][ip_pos], "gdma", ip_pos)

            if self.rn_type != "Idle":
                for in_pos in getattr(self.config, f"{self.rn_type}_send_positions"):
                    ip_pos = in_pos - self.config.cols
                    if network.ip_eject[self.rn_type][ip_pos]:
                        rsp = network.ip_eject[self.rn_type][ip_pos].popleft()
                        self._handle_response(rsp, in_pos)

        elif flit_type == "data":
            for in_pos in self.flit_position:
                ip_pos = in_pos - self.config.cols
                # eject_flits = [
                #     network.eject_queues["up"][ip_pos][0] if network.eject_queues["up"][ip_pos] else None,
                #     network.eject_queues["ring_bridge"][ip_pos][0] if network.eject_queues["ring_bridge"][ip_pos] else None,
                #     network.eject_queues["down"][ip_pos][0] if network.eject_queues["down"][ip_pos] else None,
                #     network.eject_queues["local"][ip_pos][0] if network.eject_queues["local"][ip_pos] else None,
                # ]
                eject_flits = [network.eject_queues[fifo_pos][ip_pos][0] if network.eject_queues[fifo_pos][ip_pos] else None for fifo_pos in ["up", "ring_bridge", "down", "local"]]

                # if not all(eject_flit is None for eject_flit in eject_flits):
                #     print(eject_flits)
                eject_flits = self.process_eject_queues(network, eject_flits, network.round_robin["ddr"][ip_pos], "ddr", ip_pos)
                eject_flits = self.process_eject_queues(network, eject_flits, network.round_robin["l2m"][ip_pos], "l2m", ip_pos)
                eject_flits = self.process_eject_queues(network, eject_flits, network.round_robin["sdma"][ip_pos], "sdma", ip_pos)
                eject_flits = self.process_eject_queues(network, eject_flits, network.round_robin["gdma"][ip_pos], "gdma", ip_pos)

            if self.rn_type != "Idle":
                for in_pos in self.flit_position:
                    for ip_type in [self.rn_type, self.sn_type]:
                        ip_pos = in_pos - self.config.cols
                        if network.ip_eject[ip_type][ip_pos]:
                            flit = network.ip_eject[ip_type][ip_pos].popleft()
                            flit.arrival_cycle = self.cycle
                            network.arrive_node_pre[ip_type][ip_pos] = flit
                            network.eject_num += 1
                            network.arrive_flits[flit.packet_id].append(flit)
                            network.recv_flits_num += 1
                            # if flit.req_type == "read" and flit.is_last_flit:
                            # self.create_write_req_after_read(flit)
            for in_pos in self.flit_position:
                ip_pos = in_pos - self.config.cols
                for ip_type in network.eject_queues_pre:
                    if req := network.eject_queues_pre[ip_type][ip_pos]:
                        network.ip_eject[ip_type][ip_pos].append(network.eject_queues_pre[ip_type][ip_pos])
                        network.eject_queues_pre[ip_type][ip_pos] = None

        # 最后,更新预先排队的eject队列
        if flit_type == "req":
            in_pos_position = set(self.config.ddr_send_positions + self.config.l2m_send_positions)
        elif flit_type == "rsp":
            in_pos_position = set(self.config.sdma_send_positions + self.config.gdma_send_positions)
        elif flit_type == "data":
            in_pos_position = self.flit_position

        for in_pos in in_pos_position:
            ip_pos = in_pos - self.config.cols
            for ip_type in network.eject_queues_pre:
                if network.eject_queues_pre[ip_type][ip_pos]:
                    network.ip_eject[ip_type][ip_pos].append(network.eject_queues_pre[ip_type][ip_pos])
                    network.eject_queues_pre[ip_type][ip_pos] = None
            if flit_type == "data" and self.rn_type != "Idle":
                if network.arrive_node_pre[self.rn_type][ip_pos]:
                    self.node.rn_rdb[self.rn_type][in_pos][network.arrive_node_pre[self.rn_type][ip_pos].packet_id].append(network.arrive_node_pre[self.rn_type][ip_pos])
                    if (
                        len(self.node.rn_rdb[self.rn_type][in_pos][network.arrive_node_pre[self.rn_type][ip_pos].packet_id])
                        == self.node.rn_rdb[self.rn_type][in_pos][network.arrive_node_pre[self.rn_type][ip_pos].packet_id][0].burst_length
                    ):
                        self.node.rn_rdb_recv[self.rn_type][in_pos].append(network.arrive_node_pre[self.rn_type][ip_pos].packet_id)
                    network.arrive_node_pre[self.rn_type][ip_pos] = None
                if network.arrive_node_pre[self.sn_type][ip_pos]:
                    self.node.sn_wdb[self.sn_type][in_pos][network.arrive_node_pre[self.sn_type][ip_pos].packet_id].append(network.arrive_node_pre[self.sn_type][ip_pos])
                    if (
                        len(self.node.sn_wdb[self.sn_type][in_pos][network.arrive_node_pre[self.sn_type][ip_pos].packet_id])
                        == self.node.sn_wdb[self.sn_type][in_pos][network.arrive_node_pre[self.sn_type][ip_pos].packet_id][0].burst_length
                    ):
                        self.node.sn_wdb_recv[self.sn_type][in_pos].append(network.arrive_node_pre[self.sn_type][ip_pos].packet_id)
                    network.arrive_node_pre[self.sn_type][ip_pos] = None

    def _handle_request(self, req, in_pos):
        """处理request类型的eject"""
        if req.req_type == "read":
            if req.req_attr == "new":
                if self.node.sn_tracker_count[self.sn_type]["ro"][in_pos] > 0:
                    req.sn_tracker_type = "ro"
                    self.node.sn_tracker[self.sn_type][in_pos].append(req)
                    self.node.sn_tracker_count[self.sn_type]["ro"][in_pos] -= 1
                    self.create_read_packet(req)
                # elif self.node.sn_tracker_count[self.sn_type]["share"][in_pos] > 0:
                #     req.sn_tracker_type = "share"
                #     self.node.sn_tracker[self.sn_type][in_pos].append(req)
                #     self.node.sn_tracker_count[self.sn_type]["share"][in_pos] -= 1
                #     self.create_read_packet(req)
                else:
                    self.create_rsp(req, "negative")
                    self.node.sn_req_wait[req.req_type][self.sn_type][in_pos].append(req)
            else:
                self.create_read_packet(req)
        elif req.req_type == "write":
            if req.req_attr == "new":
                if self.node.sn_tracker_count[self.sn_type]["share"][in_pos] > 0 and self.node.sn_wdb_count[self.sn_type][in_pos] >= req.burst_length:
                    req.sn_tracker_type = "share"
                    self.node.sn_tracker[self.sn_type][in_pos].append(req)
                    self.node.sn_tracker_count[self.sn_type]["share"][in_pos] -= 1
                    self.node.sn_wdb[self.sn_type][in_pos][req.packet_id] = []
                    self.node.sn_wdb_count[self.sn_type][in_pos] -= req.burst_length
                    self.create_rsp(req, "datasend")
                else:
                    # retry
                    self.create_rsp(req, "negative")
                    self.node.sn_req_wait[req.req_type][self.sn_type][in_pos].append(req)
            else:
                self.create_rsp(req, "datasend")
        if req.packet_id == 1784:
            print(req)

    def _process_ring_bridge(self, network, direction, pos, next_pos, curr_node, opposite_node):
        dir_key = f"v{direction}"

        if network.ring_bridge[dir_key][(pos, next_pos)]:
            link = (curr_node, next_pos)
            if network.links[link][-1]:
                flit_l = network.links[link][-1]
                if network.links_tag[link][-1]:
                    if flit_l.destination == next_pos:
                        eject_queue = network.eject_queues[direction][next_pos]
                        reservations = network.eject_reservations[direction][next_pos]
                        if network.links_tag[link][-1] == [next_pos, direction] and network.config.EQ_IN_FIFO_DEPTH - len(eject_queue) > len(reservations):
                            network.remain_tag[direction][next_pos] += 1
                            network.links_tag[link][-1] = None
                            return self._update_flit_state(network, dir_key, pos, next_pos, opposite_node, direction)
                elif flit_l.destination == next_pos:
                    eject_queue = network.eject_queues[direction][next_pos]
                    reservations = network.eject_reservations[direction][next_pos]
                    return (
                        self._update_flit_state(network, dir_key, pos, next_pos, opposite_node, direction)
                        if network.config.EQ_IN_FIFO_DEPTH - len(eject_queue) > len(reservations)
                        else self._handle_wait_cycles(network, dir_key, pos, next_pos, direction, link)
                    )
                else:
                    return self._handle_wait_cycles(network, dir_key, pos, next_pos, direction, link)
            else:
                if network.links_tag[link][-1] is None:
                    return self._update_flit_state(network, dir_key, pos, next_pos, opposite_node, direction)
                if network.links_tag[link][-1] == [next_pos, direction]:
                    network.remain_tag[direction][next_pos] += 1
                    network.links_tag[link][-1] = None
                    return self._update_flit_state(network, dir_key, pos, next_pos, opposite_node, direction)

    def _update_flit_state(self, network, ts_key, pos, next_pos, target_node, direction):
        flit = network.ring_bridge[ts_key][(pos, next_pos)].popleft()
        flit.current_position = next_pos
        flit.path_index += 1
        flit.current_link = (next_pos, target_node)
        flit.current_seat_index = 0
        network.links[(next_pos, target_node)][0] = flit
        return True

    def _handle_wait_cycles(self, network, ts_key, pos, next_pos, direction, link):
        if network.ring_bridge[ts_key][(pos, next_pos)][0].wait_cycle_v > self.config.ITag_Trigger_Th_V and not network.ring_bridge[ts_key][(pos, next_pos)][0].itag_v:
            if network.remain_tag[direction][next_pos] > 0:
                network.remain_tag[direction][next_pos] -= 1
                network.links_tag[link][-1] = [next_pos, direction]
                network.ring_bridge[ts_key][(pos, next_pos)][0].itag_v = True
                self.ITag_v_num_stat += 1
        else:
            for flit in network.ring_bridge[ts_key][(pos, next_pos)]:
                flit.wait_cycle_v += 1
        return False

    def _handle_response(self, rsp, in_pos):
        """处理response的eject"""
        req = next(
            (req for req in self.node.rn_tracker[rsp.req_type][self.rn_type][in_pos] if req.packet_id == rsp.packet_id),
            None,
        )
        self.rsp_cir_h_num_stat += rsp.circuits_completed_h
        self.rsp_cir_v_num_stat += rsp.circuits_completed_v
        self.rsp_wait_cycle_h_num_stat += rsp.wait_cycle_h
        self.rsp_wait_cycle_v_num_stat += rsp.wait_cycle_v
        if not req:
            print("RSP do not have REQ")
            return
        if rsp.req_type == "read":
            if rsp.rsp_type == "negative":
                if not req.early_rsp:
                    req.req_state = "invalid"
                    req.is_injected = False
                    req.path_index = 0
                    self.node.rn_rdb_count[self.rn_type][in_pos] += req.burst_length
                    self.node.rn_rdb[self.rn_type][in_pos].pop(req.packet_id)
                    self.node.rn_tracker_wait["read"][self.rn_type][in_pos].append(req)
            else:
                req.req_state = "valid"
                self.node.rn_rdb_reserve[self.rn_type][in_pos] += 1
                if req not in self.node.rn_tracker_wait["read"][self.rn_type][in_pos]:
                    req.is_injected = False
                    req.path_index = 0
                    req.early_rsp = True
                    self.node.rn_tracker_wait["read"][self.rn_type][in_pos].append(req)
        elif rsp.req_type == "write":
            if rsp.rsp_type == "negative":
                if not req.early_rsp:
                    req.req_state = "invalid"
                    req.is_injected = False
                    req.path_index = 0
                    self.node.rn_tracker_wait["write"][self.rn_type][in_pos].append(req)
            elif rsp.rsp_type == "positive":
                req.req_state = "valid"
                self.node.rn_wdb_reserve[self.rn_type][in_pos] += 1
                if req not in self.node.rn_tracker_wait["write"][self.rn_type][in_pos]:
                    req.is_injected = False
                    req.path_index = 0
                    req.early_rsp = True
                    self.node.rn_tracker_wait["write"][self.rn_type][in_pos].append(req)
            else:
                self.node.rn_wdb_send[self.rn_type][in_pos].append(rsp.packet_id)

    def tag_move(self, network):
        if self.cycle % (self.config.seats_per_link * (self.config.cols - 1) * 2 + 4) == 0:
            for i, j in network.links:
                if i - j == 1 or (i == j and (i % self.config.cols == self.config.cols - 1 and (i // self.config.cols) % 2 != 0)):
                    if network.links_tag[(i, j)][-1] == [j, "left"] and network.links[(i, j)][-1] is None:
                        network.links_tag[(i, j)][-1] = None
                        network.remain_tag["left"][j] += 1
                elif i - j == -1 or (i == j and (i % self.config.cols == 0 and (i // self.config.cols) % 2 != 0)):
                    if network.links_tag[(i, j)][-1] == [j, "right"] and network.links[(i, j)][-1] is None:
                        network.links_tag[(i, j)][-1] = None
                        network.remain_tag["right"][j] += 1
                elif i - j == self.config.cols * 2 or (i == j and i in range(self.config.num_nodes - self.config.cols * 2, self.config.cols + self.config.num_nodes - self.config.cols * 2)):
                    if network.links_tag[(i, j)][-1] == [j, "up"] and network.links[(i, j)][-1] is None:
                        network.links_tag[(i, j)][-1] = None
                        network.remain_tag["up"][j] += 1
                elif i - j == -self.config.cols * 2 or (i == j and i in range(0, self.config.cols)):
                    if network.links_tag[(i, j)][-1] == [j, "down"] and network.links[(i, j)][-1] is None:
                        network.links_tag[(i, j)][-1] = None
                        network.remain_tag["down"][j] += 1

        for col_start in range(self.config.cols):
            interval = self.config.cols * 2
            col_end = col_start + interval * (self.config.rows // 2 - 1)
            last_position = network.links_tag[(col_start, col_start)][0]
            network.links_tag[(col_start, col_start)][0] = network.links_tag[(col_start + interval, col_start)][-1]
            for i in range(1, self.config.cols):
                current_node, next_node = col_start + i * interval, col_start + (i - 1) * interval
                for j in range(self.config.seats_per_link - 7 - 1, -1, -1):
                    if j == 0 and current_node == col_end:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, current_node)][-1]
                    elif j == 0:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node + interval, current_node)][-1]
                    else:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]
            network.links_tag[(col_end, col_end)][-1] = network.links_tag[(col_end, col_end)][0]
            network.links_tag[(col_end, col_end)][0] = network.links_tag[(col_end - interval, col_end)][-1]
            for i in range(1, self.config.rows // 2):
                current_node, next_node = col_end - i * interval, col_end - (i - 1) * interval
                for j in range(self.config.seats_per_link - 1, -1, -1):
                    if j == 0 and current_node == col_start:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, current_node)][-1]
                    elif j == 0:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node - interval, current_node)][-1]
                    else:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]
            network.links_tag[(col_start, col_start)][-1] = last_position

        for row_start in range(self.config.cols, self.config.num_nodes, self.config.cols * 2):
            row_end = row_start + self.config.cols - 1
            last_position = network.links_tag[(row_start, row_start)][0]
            network.links_tag[(row_start, row_start)][0] = network.links_tag[(row_start + 1, row_start)][-1]
            for i in range(1, self.config.cols):
                current_node, next_node = row_start + i, row_start + i - 1
                for j in range(self.config.seats_per_link - 1, -1, -1):
                    if j == 0 and current_node == row_end:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, current_node)][-1]
                    elif j == 0:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node + 1, current_node)][-1]
                    else:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]
            network.links_tag[(row_end, row_end)][-1] = network.links_tag[(row_end, row_end)][0]
            network.links_tag[(row_end, row_end)][0] = network.links_tag[(row_end - 1, row_end)][-1]
            for i in range(1, self.config.cols):
                current_node, next_node = row_end - i, row_end - i + 1
                for j in range(self.config.seats_per_link - 1, -1, -1):
                    if j == 0 and current_node == row_start:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, current_node)][-1]
                    elif j == 0:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node - 1, current_node)][-1]
                    else:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]
            network.links_tag[(row_start, row_start)][-1] = last_position

    def process_eject_queues(self, network, eject_flits, rr_queue, destination_type, ip_pos):
        for i in rr_queue:
            if (
                ip_pos in network.ip_eject[destination_type]
                and eject_flits[i] is not None
                and eject_flits[i].destination_type == destination_type
                and len(network.ip_eject[destination_type][ip_pos]) < network.config.EQ_CH_FIFO_DEPTH
            ):
                # network.ip_eject[destination_type][ip_pos].append(eject_flits[i])
                # if eject_flits[i].packet_id == 64:
                # print(eject_flits[i])

                network.eject_queues_pre[destination_type][ip_pos] = eject_flits[i]
                eject_flits[i].arrival_eject_cycle = self.cycle
                eject_flits[i] = None
                if i == 0:
                    flit = network.eject_queues["up"][ip_pos].popleft()
                    if flit.ETag_priority == "T0":
                        network.EQ_UE_Counters["up"][ip_pos]["T0"] -= 1
                    elif flit.ETag_priority == "T1":
                        network.EQ_UE_Counters["up"][ip_pos]["T0"] -= 1
                        network.EQ_UE_Counters["up"][ip_pos]["T1"] -= 1
                    else:
                        network.EQ_UE_Counters["up"][ip_pos]["T0"] -= 1
                        network.EQ_UE_Counters["up"][ip_pos]["T1"] -= 1
                        network.EQ_UE_Counters["up"][ip_pos]["T2"] -= 1
                elif i == 1:
                    flit = network.eject_queues["ring_bridge"][ip_pos].popleft()
                elif i == 2:
                    flit = network.eject_queues["down"][ip_pos].popleft()
                    if flit.ETag_priority == "T1" or flit.ETag_priority == "T0":
                        network.EQ_UE_Counters["down"][ip_pos]["T1"] -= 1
                    else:
                        network.EQ_UE_Counters["down"][ip_pos]["T1"] -= 1
                        network.EQ_UE_Counters["down"][ip_pos]["T2"] -= 1
                elif i == 3:
                    flit = network.eject_queues["local"][ip_pos].popleft()

                if flit.ETag_priority == "T1":
                    self.EQ_ETag_T1_num_stat += 1
                elif flit.ETag_priority == "T0":
                    self.EQ_ETag_T0_num_stat += 1
                flit.ETag_priority = "T2"

                rr_queue.remove(i)
                rr_queue.append(i)
                break
        return eject_flits

    def create_write_req_after_read(self, flit):
        source = self.node_map(flit.destination_original)
        destination = self.node_map(flit.source_original, False)
        path = self.routes[source][destination]
        req = Flit(source, destination, path)
        req.source_original = flit.destination + self.config.cols
        req.destination_original = flit.source - self.config.cols
        req.flit_type = "req"
        req.departure_cycle = self.cycle + 1
        req.burst_length = flit.burst_length
        req.source_type = flit.destination_type
        req.destination_type = flit.source_type
        req.original_source_type = flit.original_destination_type
        req.original_destination_type = flit.original_source_type
        if self.topo_type_stat in ["5x4", "4x5"]:
            req.source_type = "gdma" if req.source_original < 16 else "sdma"
            req.destination_type = "ddr" if req.destination_original < 16 else "l2m"
        req.packet_id = Node.get_next_packet_id()
        req.req_type = "write"
        self.new_write_req.append(req)

    def create_write_packet(self, req):
        # if req.packet_id == 1785:
        #     print(req)
        for i in range(req.burst_length):
            source = req.source
            destination = req.destination
            path = self.routes[source][destination]
            flit = Flit(source, destination, path)
            flit.source_original = req.source_original
            flit.destination_original = req.destination_original
            flit.flit_type = "data"
            flit.departure_cycle = self.cycle
            flit.req_departure_cycle = req.departure_cycle
            flit.entry_db_cycle = req.entry_db_cycle
            flit.source_type = req.source_type
            flit.destination_type = req.destination_type
            flit.original_source_type = req.original_source_type
            flit.original_destination_type = req.original_destination_type
            flit.req_type = req.req_type
            flit.packet_id = req.packet_id
            flit.flit_id_in_packet = i
            flit.burst_length = req.burst_length
            if i == req.burst_length - 1:
                flit.is_last_flit = True
            self.node.rn_wdb[flit.source_type][flit.source][flit.packet_id].append(flit)
            self.flit_network.send_flits[flit.packet_id].append(flit)

    def create_read_packet(self, req):
        # if req.packet_id == 64:
        # print(req)
        for i in range(req.burst_length):
            source = req.destination + self.config.cols
            destination = req.source - self.config.cols
            path = self.routes[source][destination]
            flit = Flit(source, destination, path)
            flit.source_original = req.destination_original
            flit.destination_original = req.source_original
            flit.req_type = req.req_type
            flit.flit_type = "data"
            flit.departure_cycle = self.cycle + self.config.ddr_latency + i  # if req.destination_type == "ddr" else self.cycle + i
            flit.entry_db_cycle = self.cycle
            # flit.entry_db_cycle = req.entry_db_cycle
            flit.req_departure_cycle = req.departure_cycle
            flit.source_type = req.destination_type
            flit.destination_type = req.source_type
            flit.original_source_type = req.original_source_type
            flit.original_destination_type = req.original_destination_type
            flit.packet_id = req.packet_id
            flit.flit_id_in_packet = i
            flit.burst_length = req.burst_length
            if i == req.burst_length - 1:
                flit.is_last_flit = True
            self.node.sn_rdb[flit.source_type][flit.source].append(flit)
            self.flit_network.send_flits[flit.packet_id].append(flit)

    def create_rsp(self, req, rsp_type):
        if rsp_type == "negative":
            if req.req_type == "read":
                self.read_retry_num_stat += 1
            elif req.req_type == "write":
                self.write_retry_num_stat += 1
        source = req.destination + self.config.cols
        destination = req.source - self.config.cols
        path = self.routes[source][destination]
        rsp = Flit(source, destination, path)
        rsp.flit_type = "rsp"
        rsp.rsp_type = rsp_type
        rsp.req_type = req.req_type
        rsp.packet_id = req.packet_id
        rsp.departure_cycle = self.cycle
        rsp.req_departure_cycle = req.departure_cycle
        rsp.source_type = req.destination_type
        rsp.destination_type = req.source_type
        self.rsp_network.send_flits[rsp.packet_id].append(rsp)
        self.node.sn_rsp_queue[rsp.source_type][source].append(rsp)

    def process_inject_queues(self, network, inject_queues):
        flit_num = 0
        flits = []
        for source, queue in inject_queues.items():
            if queue and queue[0]:
                flit = queue.popleft()
                if flit.inject(network):
                    network.inject_num += 1
                    flit_num += 1
                    flit.departure_network_cycle = self.cycle
                    flits.append(flit)
                else:
                    queue.appendleft(flit)
                    for flit in queue:
                        flit.wait_cycle_h += 1
                if flit.itag_h:
                    self.ITag_h_num_stat += 1
        return flit_num, flits

    def evaluate_results(self, network):
        """
        Evaluate the results of the simulation.

        :param network: The network object
        :return: None
        """
        if not self.result_save_path:
            return

        # Save configuration
        with open(os.path.join(self.result_save_path, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=4)

        read_latency, write_latency = [], []
        read_merged_intervals, write_merged_intervals = [(0, 0, 0)], [(0, 0, 0)]

        with open(os.path.join(self.result_save_path, f"Result_{self.file_name[10:-9]}R.txt"), "w") as f1, open(os.path.join(self.result_save_path, f"Result_{self.file_name[10:-9]}W.txt"), "w") as f2:

            # Print headers
            print(
                "tx_time(ns), src_id, src_type, des_id, des_type, R/W, burst_len, rx_time(ns), path, circuits_completed_v, circuits_completed_h",
                file=f1,
            )
            print(
                "tx_time(ns), src_id, src_type, des_id, des_type, R/W, burst_len, rx_time(ns), path, circuits_completed_v, circuits_completed_h",
                file=f2,
            )

            # Process each flit
            self.sdma_R_ddr_finish_time, self.sdma_W_l2m_finish_time, self.gdma_R_l2m_finish_time = 0, 0, 0
            self.sdma_R_ddr_flit_num, self.sdma_W_l2m_flit_num, self.gdma_R_l2m_flit_num = 0, 0, 0
            self.sdma_R_ddr_latency, self.sdma_W_l2m_latency, self.gdma_R_l2m_latency = [], [], []
            for flits in network.arrive_flits.values():
                for flit in flits:
                    self.data_cir_h_num_stat += flit.circuits_completed_h
                    self.data_cir_v_num_stat += flit.circuits_completed_v
                    self.data_wait_cycle_h_num_stat += flit.wait_cycle_h
                    self.data_wait_cycle_v_num_stat += flit.wait_cycle_v
                self.process_flits(
                    # flits[-1],
                    next((flit for flit in flits if flit.is_last_flit), flits[-1]),
                    network,
                    read_latency,
                    write_latency,
                    read_merged_intervals,
                    write_merged_intervals,
                    f1,
                    f2,
                )

        # Calculate and output results
        self.calculate_and_output_results(network, read_latency, write_latency, read_merged_intervals, write_merged_intervals)

    def calculate_predicted_duration(self, flit):
        """Calculate the predicted duration based on the flit's path."""
        duration = sum((2 if flit.path[i] - flit.path[i - 1] == -self.config.cols else self.config.seats_per_link) for i in range(1, len(flit.path)))
        duration += 2 if flit.path[1] - flit.path[0] == -self.config.cols else 3
        return 0 if len(flit.path) == 2 else duration

    def process_flits(self, flit, network, read_latency, write_latency, read_merged_intervals, write_merged_intervals, f1, f2):
        """Process a single flit and update the network and latency data."""
        # Calculate predicted and actual durations
        flit.predicted_duration = self.calculate_predicted_duration(flit)
        flit.actual_duration = flit.arrival_cycle - flit.departure_cycle
        flit.actual_ject_duration = flit.arrival_eject_cycle - flit.departure_inject_cycle
        flit.actual_network_duration = flit.arrival_network_cycle - flit.departure_network_cycle

        # # Update network statistics
        # network.inject_time[flit.source].append(flit.departure_inject_cycle - flit.departure_cycle)
        # network.eject_time[flit.destination].append(flit.arrival_cycle - flit.arrival_network_cycle)
        # network.all_latency.append(flit.actual_duration)
        # network.ject_latency.append(flit.actual_ject_duration)
        # network.network_latency.append(flit.actual_network_duration)
        # network.predicted_recv_time.append(flit.predicted_duration)

        # if flit.circuits_completed_h != 0:
        #     network.circuits_h.append(flit.circuits_completed_h)
        #     network.circuits_flit_h[flit.destination_type] += 1
        # if flit.circuits_completed_v != 0:
        #     network.circuits_v.append(flit.circuits_completed_v)
        #     network.circuits_flit_v[flit.destination_type] += 1

        # Skip if not the last flit or if arrival/departure cycles are invalid
        if not flit.is_last_flit or not flit.arrival_cycle or not flit.req_departure_cycle:
            return

        # Update merged intervals and latencies
        if flit.req_type == "read":
            self.update_intervals(flit, read_merged_intervals, read_latency, f1, "R")
        elif flit.req_type == "write":
            self.update_intervals(flit, write_merged_intervals, write_latency, f2, "W")

    def calculate_and_output_results(self, network, read_latency, write_latency, read_merged_intervals, write_merged_intervals):
        """Calculate average latencies and output total results."""
        # Calculate average latencies
        for source in self.flit_position:
            destination = source - self.config.cols
            if network.inject_time[source]:
                network.avg_inject_time[source] = sum(network.inject_time[source]) / len(network.inject_time[source])
            if network.eject_time[destination]:
                network.avg_eject_time[destination] = sum(network.eject_time[destination]) / len(network.eject_time[destination])

        network.avg_circuits_h = sum(network.circuits_h) / len(network.circuits_h) / 2 if network.circuits_h else None
        network.max_circuits_h = max(network.circuits_h) / 2 if network.circuits_h else None
        network.avg_circuits_v = sum(network.circuits_v) / len(network.circuits_v) / 2 if network.circuits_v else None
        network.max_circuits_v = max(network.circuits_v) / 2 if network.circuits_v else None

        # Output total results
        print("=" * 50)
        total_result = os.path.join(self.result_save_path, "total_result.txt")
        with open(total_result, "w", encoding="utf-8") as f3:
            print(f"Topology: {self.topo_type_stat }, file_name: {self.file_name}")
            print(f"Topology: {self.topo_type_stat }, file_name: {self.file_name}", file=f3)
            if read_latency:
                self.read_BW_stat, self.read_latency_avg_stat, self.read_latency_max_stat = self.output_intervals(f3, read_merged_intervals, "Read", read_latency)
            if write_latency:
                self.write_BW_stat, self.write_latency_avg_stat, self.write_latency_max_stat = self.output_intervals(f3, write_merged_intervals, "Write", write_latency)

            print("\nPer-IP Weighted Bandwidth:", file=f3)
            # 处理读带宽
            print("\nRead Bandwidth per IP:", file=f3)
            for ip_id in sorted(self.read_ip_intervals.keys()):
                intervals = self.read_ip_intervals[ip_id]
                bw = self.calculate_ip_bandwidth(intervals)
                print(f"{ip_id}: {bw:.1f} GB/s", file=f3)

            # 处理写带宽
            print("\nWrite Bandwidth per IP:", file=f3)
            for ip_id in sorted(self.write_ip_intervals.keys()):
                intervals = self.write_ip_intervals[ip_id]
                bw = self.calculate_ip_bandwidth(intervals)
                print(f"{ip_id}: {bw:.1f} GB/s", file=f3)


        self.Total_BW_stat = self.read_BW_stat + self.write_BW_stat
        print(f"Read + Write Bandwidth: {self.Total_BW_stat:.1f}")
        print("=" * 50)
        print(f"Total Circuits req h: {self.req_cir_h_num_stat}, v: {self.req_cir_v_num_stat}")
        print(f"Total Circuits rsp h: {self.rsp_cir_h_num_stat}, v: {self.rsp_cir_v_num_stat}")
        print(f"Total Circuits data h: {self.data_cir_h_num_stat}, v: {self.data_cir_v_num_stat}")
        print(f"Total wait cycle req h: {self.req_wait_cycle_h_num_stat}, v: {self.req_wait_cycle_v_num_stat}")
        print(f"Total wait cycle rsp h: {self.rsp_wait_cycle_h_num_stat}, v: {self.rsp_wait_cycle_v_num_stat}")
        print(f"Total wait cycle data h: {self.data_wait_cycle_h_num_stat}, v: {self.data_wait_cycle_v_num_stat}")
        print(f"Total RB ETag: T1: {self.RB_ETag_T1_num_stat}, T0: {self.RB_ETag_T0_num_stat}; EQ ETag: T1: {self.EQ_ETag_T1_num_stat}, T0: {self.EQ_ETag_T0_num_stat}")
        print(f"Total ITag: h: {self.ITag_h_num_stat}, v: {self.ITag_v_num_stat}")
        if self.model_type_stat == "REQ_RSP":
            print(f"Retry num: R: {self.read_retry_num_stat}, W: {self.write_retry_num_stat}")
        if self.plot_flow_fig:
            self.draw_flow_graph(self.flit_network, save_path=self.results_fig_save_path)
        # print("=" * 50)
        # print(
        #     f"Throughput: sdma-R-DDR: {((self.sdma_R_ddr_flit_num * 128/self.sdma_R_ddr_finish_time/4) if self.sdma_R_ddr_finish_time>0 else 0):.1f}, "
        #     f"sdma-W-l2m: {(self.sdma_W_l2m_flit_num* 128/self.sdma_W_l2m_finish_time/4 if self.sdma_W_l2m_finish_time>0 else 0):.1f}, "
        #     f"gdma-R-L2M: {(self.gdma_R_l2m_flit_num* 128/self.gdma_R_l2m_finish_time/4 if self.gdma_R_l2m_finish_time>0 else 0):.1f}"
        # )
        # print(
        #     f"Finish Cycle: sdma-R-DDR: {self.sdma_R_ddr_finish_time * self.config.network_frequency}, "
        #     f"sdma-W-l2m: {self.sdma_W_l2m_finish_time* self.config.network_frequency}, "
        #     f"gdma-R-L2M: {self.gdma_R_l2m_finish_time* self.config.network_frequency}"
        # )
        # print(
        #     f"Avg Latency: sdma-R-DDR: {(np.average(self.sdma_R_ddr_latency) if self.sdma_R_ddr_latency else 0):.1f}, "
        #     f"sdma-W-l2m: {(np.average(self.sdma_W_l2m_latency) if self.sdma_W_l2m_latency else 0):.1f}, "
        #     f"gdma-R-L2M: {(np.average(self.gdma_R_l2m_latency)if self.gdma_R_l2m_latency else 0):.1f}"
        # )
        # print("=" * 50)

    def update_intervals(self, flit, merged_intervals, latency, file, req_type):
        """Update the merged intervals and latency for the given request type."""
        last_start, last_end, count = merged_intervals[-1]

        # 根据请求类型更新对应的IP区间
        if req_type == "R":
            dma_id = f"{str(flit.destination_type)}_{str(flit.destination + self.config.cols)}"
            ddr_id = f"{str(flit.source_type)}_{str(flit.source)}"
            dma_intervals = self.read_ip_intervals[dma_id]
            ddr_intervals = self.read_ip_intervals[ddr_id]
        elif req_type == "W":
            dma_id = f"{str(flit.source_type)}_{str(flit.source)}"
            ddr_id = f"{str(flit.destination_type)}_{str(flit.destination+ self.config.cols)}"
            dma_intervals = self.write_ip_intervals[dma_id]
            ddr_intervals = self.write_ip_intervals[ddr_id]

        # 合并区间逻辑（与全局merged_intervals相同）
        current_start = flit.req_departure_cycle // self.config.network_frequency
        current_end = flit.arrival_cycle // self.config.network_frequency
        current_count = flit.burst_length

        if not dma_intervals:
            dma_intervals.append((current_start, current_end, current_count))
        else:
            last_start, last_end, last_count = dma_intervals[-1]
            if current_start <= last_end:
                merged = (last_start, max(last_end, current_end), last_count + current_count)
                dma_intervals[-1] = merged
            else:
                dma_intervals.append((current_start, current_end, current_count))
        if not ddr_intervals:
            ddr_intervals.append((current_start, current_end, current_count))
        else:
            last_start, last_end, last_count = ddr_intervals[-1]
            if current_start <= last_end:
                merged = (last_start, max(last_end, current_end), last_count + current_count)
                ddr_intervals[-1] = merged
            else:
                ddr_intervals.append((current_start, current_end, current_count))

        # 更新字典
        if req_type == "R":
            self.read_ip_intervals[dma_id] = dma_intervals
            self.read_ip_intervals[ddr_id] = ddr_intervals
        elif req_type == "W":
            self.write_ip_intervals[dma_id] = dma_intervals
            self.write_ip_intervals[ddr_id] = ddr_intervals

        if flit.req_departure_cycle // self.config.network_frequency <= last_end:
            merged_intervals[-1] = (
                last_start,
                max(last_end, flit.arrival_cycle // self.config.network_frequency),
                count + flit.burst_length,
            )
        else:
            merged_intervals.append(
                (
                    flit.req_departure_cycle // self.config.network_frequency,
                    flit.arrival_cycle // self.config.network_frequency,
                    flit.burst_length,
                )
            )

        if flit.source_type == "ddr" and flit.destination_type == "sdma" and req_type == "R":
            self.sdma_R_ddr_finish_time = max(self.sdma_R_ddr_finish_time, flit.arrival_cycle // self.config.network_frequency)
            self.sdma_R_ddr_flit_num += flit.burst_length
            if flit.leave_db_cycle is None:
                flit.leave_db_cycle = flit.arrival_cycle
            self.sdma_R_ddr_latency.append(flit.leave_db_cycle - flit.entry_db_cycle)
        elif flit.source_type == "ddr" and flit.destination_type == "sdma" and req_type == "W":
            self.sdma_W_l2m_finish_time = max(self.sdma_W_l2m_finish_time, flit.arrival_cycle // self.config.network_frequency)
            self.sdma_W_l2m_flit_num += flit.burst_length
            if flit.leave_db_cycle is None:
                flit.leave_db_cycle = flit.arrival_cycle
            self.sdma_W_l2m_latency.append(flit.leave_db_cycle - flit.entry_db_cycle)
        elif flit.source_type == "l2m" and flit.destination_type == "gdma" and req_type == "R":
            self.gdma_R_l2m_finish_time = max(self.gdma_R_l2m_finish_time, flit.arrival_cycle // self.config.network_frequency)
            self.gdma_R_l2m_flit_num += flit.burst_length
            if flit.leave_db_cycle is None:
                flit.leave_db_cycle = flit.arrival_cycle
            self.gdma_R_l2m_latency.append(flit.leave_db_cycle - flit.entry_db_cycle)

        latency.append(flit.arrival_cycle // self.config.network_frequency - flit.req_departure_cycle // self.config.network_frequency)
        print(
            f"{flit.req_departure_cycle // self.config.network_frequency},{flit.source_original},{flit.original_source_type},{flit.destination_original},{flit.original_destination_type},"
            f"{req_type},{flit.burst_length},{flit.arrival_cycle // self.config.network_frequency},{flit.path},{flit.circuits_completed_v},{flit.circuits_completed_h}",
            file=file,
        )

    def output_intervals(self, f3, merged_intervals, req_type, latency):
        """Output the intervals and calculate bandwidth for the given request type."""
        print(f"{req_type} intervals:", file=f3)
        print(f"{req_type} results:")
        # print(f"{req_type} results:", file=f3)
        # weighted_bandwidth_sum, total_count, finish_time, total_bandwidth = 0, 0, 0, [np.inf, -np.inf]
        weighted_bandwidth_sum, total_count, finish_time, total_bandwidth = 0, 0, self.cycle // self.config.network_frequency, [np.inf, -np.inf]

        for start, end, count in merged_intervals:
            if start == end:
                continue
            bandwidth = count * 128 / (end - start) / self.config.num_ips
            total_bandwidth[0] = min(total_bandwidth[0], bandwidth)
            total_bandwidth[1] = max(total_bandwidth[1], bandwidth)
            weighted_bandwidth_sum += bandwidth * count
            total_count += count
            print(f"Interval: {start} to {end}, count: {count}, bandwidth: {bandwidth:.1f}", file=f3)
            finish_time = max(finish_time, end)

        # weighted_bandwidth = weighted_bandwidth_sum / total_count if total_count > 0 else 0
        weighted_bandwidth = total_count * 128 / finish_time / self.config.num_ips

        if req_type == "Read":
            self.R_finish_time_stat = finish_time
            self.R_tail_latency_stat = finish_time - self.R_tail_latency_stat // self.config.network_frequency
            print(f"Finish Time: {self.R_finish_time_stat}, Tail latency: {self.R_tail_latency_stat}")
        elif req_type == "Write":
            self.W_finish_time_stat = finish_time
            self.W_tail_latency_stat = finish_time - self.W_tail_latency_stat // self.config.network_frequency
            print(f"Finish Time: {self.W_finish_time_stat}, Tail latency: {self.W_tail_latency_stat}")

        latency_avg = np.average(latency)
        latency_max = max(latency)
        print(
            f"Weighted bandwidth: {weighted_bandwidth:.1f} AvgLatency: {latency_avg:.1f}, MaxLatency: {latency_max}",
            file=f3,
        )
        print(f"Weighted bandwidth: {weighted_bandwidth:.1f} AvgLatency: {latency_avg:.1f}, MaxLatency: {latency_max}")
        return weighted_bandwidth, latency_avg, latency_max

    def calculate_ip_bandwidth(self, intervals):
        """计算给定区间的加权带宽"""
        weighted_sum = 0.0
        total_count = 0
        finish_time = 0
        # finish_time = self.cycle // self.config.network_frequency
        for start, end, count in intervals:
            if start >= end:
                continue  # 跳过无效区间
            duration = end - start
            bandwidth = (count * 128) / duration  # 计算该区间的带宽（不除以IP总数）
            weighted_sum += bandwidth * count  # 加权求和
            total_count += count
            finish_time = max(finish_time, end)

        # return weighted_sum / total_count if total_count > 0 else 0.0
        return total_count * 128 / finish_time if finish_time > 0 else 0.0

    def calculate_ip_bandwidth_data(self):
        rows = self.config.rows
        cols = self.config.cols
        if self.topo_type_stat != "4x5":
            rows -= 1

        # 初始化数据结构
        self.ip_bandwidth_data = {
            "read": {"sdma": np.zeros((rows, cols)), "gdma": np.zeros((rows, cols)), "ddr": np.zeros((rows, cols)), "l2m": np.zeros((rows, cols))},
            "write": {"sdma": np.zeros((rows, cols)), "gdma": np.zeros((rows, cols)), "ddr": np.zeros((rows, cols)), "l2m": np.zeros((rows, cols))},
            "total": {"sdma": np.zeros((rows, cols)), "gdma": np.zeros((rows, cols)), "ddr": np.zeros((rows, cols)), "l2m": np.zeros((rows, cols))},
        }

        # 填充数据
        for ip_id in set(self.read_ip_intervals) | set(self.write_ip_intervals):
            source_type, source = ip_id.split("_")
            source = int(source)

            physical_col = source % cols
            physical_row = source // cols // 2

            # 计算带宽
            read_bw = self.calculate_ip_bandwidth(self.read_ip_intervals.get(ip_id, []))
            write_bw = self.calculate_ip_bandwidth(self.write_ip_intervals.get(ip_id, []))
            total_bw = read_bw + write_bw

            # 存储数据
            self.ip_bandwidth_data["read"][source_type][physical_row, physical_col] = read_bw
            self.ip_bandwidth_data["write"][source_type][physical_row, physical_col] = write_bw
            self.ip_bandwidth_data["total"][source_type][physical_row, physical_col] = total_bw

    def draw_flow_graph(self, network, mode="total", node_size=2000, save_path=None):
        """
        绘制合并的网络流图和IP

        :param network: 网络对象
        :param mode: 显示模式，可以是'read', 'write'或'total'
        :param node_size: 节点大小
        :param save_path: 图片保存路径
        """
        # 确保IP带宽数据已计算
        if not hasattr(self, "ip_bandwidth_data"):
            self.calculate_ip_bandwidth_data()

        # 准备网络流数据
        G = nx.DiGraph()

        # 处理不同模式的网络流数据
        if mode == "read":
            links = network.links_flow_stat.get("read", {})
        elif mode == "write":
            links = network.links_flow_stat.get("write", {})
        else:  # total模式，需要合并读和写的数据
            read_links = network.links_flow_stat.get("read", {})
            write_links = network.links_flow_stat.get("write", {})

            # 合并读和写的流量
            all_keys = set(read_links.keys()) | set(write_links.keys())
            links = {}
            for key in all_keys:
                read_val = read_links.get(key, 0)
                write_val = write_links.get(key, 0)
                links[key] = read_val + write_val

        link_values = []
        for (i, j), value in links.items():
            link_value = value * 128 / (self.cycle // self.config.network_frequency)
            link_values.append(link_value)
            formatted_label = f"{link_value:.1f}"
            G.add_edge(i, j, label=formatted_label)

        # 计算节点位置
        pos = {}
        for node in G.nodes():
            x = node % self.config.cols
            y = node // self.config.cols
            if y % 2 == 1:  # 奇数行左移
                x -= 0.25
                y -= 0.6
            pos[node] = (x * 3, -y * 1.5)

        # 创建图形
        fig, ax = plt.subplots(figsize=(16, 14))
        ax.set_aspect("equal")

        # 调整方形节点大小
        square_size = np.sqrt(node_size) / 100

        # 绘制网络流图
        nx.draw_networkx_nodes(G, pos, node_size=square_size, node_shape="s", ax=ax)

        # 绘制方形节点并添加IP信息
        for node, (x, y) in pos.items():
            # 绘制主节点方框
            rect = Rectangle((x - square_size / 2, y - square_size / 2), width=square_size, height=square_size, color="lightblue", ec="black", zorder=2)
            ax.add_patch(rect)
            ax.text(x, y, str(node), ha="center", va="center", fontsize=10)

            # 在节点左侧添加IP信息
            physical_row = node // self.config.cols
            physical_col = node % self.config.cols

            if physical_row % 2 == 0:
                # 田字格位置和大小
                ip_width = square_size * 2.6
                ip_height = square_size * 2.6
                ip_x = x - square_size - ip_width / 2.8
                ip_y = y + 0.26

                # 绘制田字格外框
                ip_rect = Rectangle((ip_x - ip_width / 2, ip_y - ip_height / 2), width=ip_width, height=ip_height, color="white", ec="black", linewidth=1, zorder=2)
                ax.add_patch(ip_rect)

                # 绘制田字格内部线条
                ax.plot([ip_x - ip_width / 2, ip_x + ip_width / 2], [ip_y, ip_y], color="black", linewidth=1, zorder=3)
                ax.plot([ip_x, ip_x], [ip_y - ip_height / 2, ip_y + ip_height / 2], color="black", linewidth=1, zorder=3)

                # 为左列和右列填充不同颜色（DMA和DDR区分）
                left_color = "honeydew"  # 左列颜色（DMA）
                right_color = "aliceblue"  # 右列颜色（GDMA）
                # 左列矩形（DMA区域）
                left_rect = Rectangle((ip_x - ip_width / 2, ip_y - ip_height / 2), width=ip_width / 2, height=ip_height, color=left_color, ec="none", zorder=2)
                ax.add_patch(left_rect)

                # 右列矩形（DDR区域）
                right_rect = Rectangle((ip_x, ip_y - ip_height / 2), width=ip_width / 2, height=ip_height, color=right_color, ec="none", zorder=2)
                ax.add_patch(right_rect)

                # 添加IP信息
                if mode == "read":
                    sdma_value = self.ip_bandwidth_data["read"]["sdma"][physical_row // 2, physical_col]
                    gdma_value = self.ip_bandwidth_data["read"]["gdma"][physical_row // 2, physical_col]
                    ddr_value = self.ip_bandwidth_data["read"]["ddr"][physical_row // 2, physical_col]
                    l2m_value = self.ip_bandwidth_data["read"]["l2m"][physical_row // 2, physical_col]

                    # 收集当前模式下每个IP的所有值
                    all_sdma = self.ip_bandwidth_data["read"]["sdma"].flatten()
                    all_gdma = self.ip_bandwidth_data["read"]["gdma"].flatten()
                    all_ddr = self.ip_bandwidth_data["read"]["ddr"].flatten()
                    all_l2m = self.ip_bandwidth_data["read"]["l2m"].flatten()

                elif mode == "write":
                    sdma_value = self.ip_bandwidth_data["write"]["sdma"][physical_row // 2, physical_col]
                    gdma_value = self.ip_bandwidth_data["write"]["gdma"][physical_row // 2, physical_col]
                    ddr_value = self.ip_bandwidth_data["write"]["ddr"][physical_row // 2, physical_col]
                    l2m_value = self.ip_bandwidth_data["write"]["l2m"][physical_row // 2, physical_col]

                    all_sdma = self.ip_bandwidth_data["write"]["sdma"].flatten()
                    all_gdma = self.ip_bandwidth_data["write"]["gdma"].flatten()
                    all_ddr = self.ip_bandwidth_data["write"]["ddr"].flatten()
                    all_l2m = self.ip_bandwidth_data["write"]["l2m"].flatten()

                else:  # total
                    sdma_value = self.ip_bandwidth_data["total"]["sdma"][physical_row // 2, physical_col]
                    gdma_value = self.ip_bandwidth_data["total"]["gdma"][physical_row // 2, physical_col]
                    ddr_value = self.ip_bandwidth_data["total"]["ddr"][physical_row // 2, physical_col]
                    l2m_value = self.ip_bandwidth_data["total"]["l2m"][physical_row // 2, physical_col]

                    all_sdma = self.ip_bandwidth_data["total"]["sdma"].flatten()
                    all_gdma = self.ip_bandwidth_data["total"]["gdma"].flatten()
                    all_ddr = self.ip_bandwidth_data["total"]["ddr"].flatten()
                    all_l2m = self.ip_bandwidth_data["total"]["l2m"].flatten()

                # 计算每个IP的阈值（例如取前20%的分位数）
                sdma_threshold = np.percentile(all_sdma, 90)  # 大于80%的值会被标红
                gdma_threshold = np.percentile(all_gdma, 90)
                ddr_threshold = np.percentile(all_ddr, 90)
                l2m_threshold = np.percentile(all_l2m, 90)

                # SDMA在左上半部分（大于阈值则红色）
                sdma_color = "red" if sdma_value > sdma_threshold else "black"
                ax.text(ip_x - ip_width / 4, ip_y + ip_height / 4, f"{sdma_value:.1f}", fontweight="bold", ha="center", va="center", fontsize=9.5, color=sdma_color)

                # GDMA在左下半部分
                gdma_color = "red" if gdma_value > gdma_threshold else "black"
                ax.text(ip_x - ip_width / 4, ip_y - ip_height / 4, f"{gdma_value:.1f}", fontweight="bold", ha="center", va="center", fontsize=9.5, color=gdma_color)

                # l2m在右上半部分
                l2m_color = "red" if l2m_value > l2m_threshold else "black"
                ax.text(ip_x + ip_width / 4, ip_y + ip_height / 4, f"{l2m_value:.1f}", fontweight="bold", ha="center", va="center", fontsize=9.5, color=l2m_color)

                # ddr在右下半部分
                ddr_color = "red" if ddr_value > ddr_threshold else "black"
                ax.text(ip_x + ip_width / 4, ip_y - ip_height / 4, f"{ddr_value:.1f}", fontweight="bold", ha="center", va="center", fontsize=9.5, color=ddr_color)

        # 绘制边和边标签
        edge_value_threshold = np.percentile(link_values, 90)

        for i, j, data in G.edges(data=True):
            x1, y1 = pos[i]
            x2, y2 = pos[j]
            if float(data["label"]) > edge_value_threshold:
                color = "red"
            else:
                color = "black"

            if i != j:  # 普通边
                dx, dy = x2 - x1, y2 - y1
                dist = np.hypot(dx, dy)
                if dist > 0:
                    dx, dy = dx / dist, dy / dist
                    perp_dx, perp_dy = -dy * 0.1, dx * 0.1

                    has_reverse = G.has_edge(j, i)
                    if has_reverse:
                        start_x = x1 + dx * square_size / 2 + perp_dx
                        start_y = y1 + dy * square_size / 2 + perp_dy
                        end_x = x2 - dx * square_size / 2 + perp_dx
                        end_y = y2 - dy * square_size / 2 + perp_dy
                    else:
                        start_x = x1 + dx * square_size / 2
                        start_y = y1 + dy * square_size / 2
                        end_x = x2 - dx * square_size / 2
                        end_y = y2 - dy * square_size / 2

                    arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle="-|>", mutation_scale=12, color=color, zorder=1, linewidth=1)
                    ax.add_patch(arrow)

        # 绘制边标签
        edge_labels = nx.get_edge_attributes(G, "label")
        for edge, label in edge_labels.items():
            i, j = edge
            if float(label) == 0.0:
                continue
            if float(label) > edge_value_threshold:
                color = "red"
            else:
                color = "black"
            if i == j:
                # 计算标签位置
                original_row = i // self.config.cols
                original_col = i % self.config.cols
                x, y = pos[i]

                offset = 0.17  # 标签偏移量
                if original_row == 0:
                    label_pos = (x, y + square_size / 2 + offset)
                    angle = 0
                elif original_row == self.config.rows - 2:
                    label_pos = (x, y - square_size / 2 - offset)
                    angle = 0
                elif original_col == 0:
                    label_pos = (x - square_size / 2 - offset, y)
                    angle = -90
                elif original_col == self.config.cols - 1:
                    label_pos = (x + square_size / 2 + offset, y)
                    angle = 90
                else:
                    label_pos = (x, y + square_size / 2 + offset)
                    angle = 0

                ax.text(*label_pos, str(label), ha="center", va="center", color=color, fontweight="bold", fontsize=11, rotation=angle)

            if i != j:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                angle = np.degrees(np.arctan2(dy, dx))

                has_reverse = G.has_edge(j, i)
                is_horizontal = abs(dx) > abs(dy)

                if has_reverse:
                    if is_horizontal:
                        perp_dx, perp_dy = -dy * 0.1 + 0.2, dx * 0.1
                    else:
                        perp_dx, perp_dy = -dy * 0.18, dx * 0.18 - 0.3
                    label_x = mid_x + perp_dx
                    label_y = mid_y + perp_dy
                else:
                    if is_horizontal:
                        label_x = mid_x + dx * 0.1
                        label_y = mid_y + dy * 0.1
                    else:
                        label_x = mid_x + (-dy * 0.1 if dx > 0 else dy * 0.1)
                        label_y = mid_y - 0.1

                ax.text(label_x, label_y, str(label), ha="center", va="center", fontsize=14, fontweight="bold", color=color)

        plt.axis("off")
        title = f"{network.name} - {mode.capitalize()} Bandwidth"
        if self.config.spare_core_row != -1:
            title += f"\nRow: {self.config.spare_core_row}, Failed cores: {self.config.fail_core_pos}, Spare cores: {self.config.spare_core_pos}"
        plt.title(title, fontsize=20)

        # # 添加图例说明
        # legend_text = f"IP {mode.capitalize()} Bandwidth (GB/s):\n" "SDMA: Top half of square\n" "GDMA: Bottom half of square"
        # plt.figtext(0.02, 0.98, legend_text, ha="left", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

        plt.tight_layout()

        if save_path:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{str(self.topo_type_stat)}_{self.file_name[:-4]}_combined_{mode}_{network.name}_{self.config.fail_core_pos}_{self.config.spare_core_row}_{str(time.time_ns())[-3:]}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

    def evaluate_performance(self, network):
        # receive confirm
        send_flit_ids = {flit.id for flit in network.send_flits}
        arrive_flit_ids = {flit.id for flit in network.arrive_flits}
        unreceived_flit_ids = send_flit_ids - arrive_flit_ids
        unreceived_flits = [flit for flit in network.send_flits if flit.id in unreceived_flit_ids]

        # Latency confirm
        for flit in network.arrive_flits:
            for i in range(1, len(flit.path)):
                if flit.path[i] - flit.path[i - 1] == -self.config.cols:
                    flit.predicted_duration += 2
                else:
                    flit.predicted_duration += self.config.seats_per_link
            if flit.path[1] - flit.path[0] == -self.config.cols:
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

        for source in self.flit_position:
            destination = source - self.config.cols
            if network.inject_time[source]:
                network.avg_inject_time[source] = sum(network.inject_time[source]) / len(network.inject_time[source])
            if network.eject_time[destination]:
                network.avg_eject_time[destination] = sum(network.eject_time[destination]) / len(network.eject_time[destination])
        network.predicted_avg_latency = sum(network.predicted_recv_time) / len(network.predicted_recv_time) / 2
        network.predicted_max_latency = max(network.predicted_recv_time) / 2
        network.actual_avg_latency = sum(network.all_latency) / len(network.all_latency) / 2
        network.actual_max_latency = max(network.all_latency) / 2
        network.actual_avg_ject_latency = sum(network.ject_latency) / len(network.ject_latency) / 2
        network.actual_max_ject_latency = max(network.ject_latency) / 2
        network.actual_avg_net_latency = sum(network.network_latency) / len(network.network_latency) / 2
        network.actual_max_net_latency = max(network.network_latency) / 2
        network.avg_circuits_h = sum(network.circuits_h) / len(network.circuits_h) / 2 if len(network.circuits_h) > 0 else None
        network.max_circuits_h = max(network.circuits_h) / 2 if len(network.circuits_h) > 0 else None
        network.avg_circuits_v = sum(network.circuits_v) / len(network.circuits_v) / 2 if len(network.circuits_v) > 0 else None
        network.max_circuits_v = max(network.circuits_v) / 2 if len(network.circuits_v) > 0 else None

        # throughput confirm
        # self.cycle = 828 - 10
        for ip_type in network.per_send_throughput:
            for source in network.per_send_throughput[ip_type]:
                destination = source - self.config.cols
                network.per_send_throughput[ip_type][source] = 256 * network.num_send[ip_type][source] / (self.cycle)
                network.per_recv_throughput[ip_type][destination] = 256 * network.num_recv[ip_type][destination] / (self.cycle)
                network.send_throughput[ip_type] += network.per_send_throughput[ip_type][source]
                network.recv_throughput[ip_type] += network.per_recv_throughput[ip_type][destination]
            # network.send_throughput[ip_type] = network.send_throughput[ip_type] / len(network.per_send_throughput[ip_type]) if len(network.per_send_throughput[ip_type]) > 0 else None
            # network.recv_throughput[ip_type] = network.recv_throughput[ip_type] / len(network.per_send_throughput[ip_type]) if len(network.per_send_throughput[ip_type]) > 0 else None

        sorted_flits = sorted(network.arrive_flits, key=lambda flit: flit.id)
        read_flits, write_flits = {}, {}
        for flit in sorted_flits:
            if flit.source_type in ["ddr", "l2m"]:
                if flit.packet_id in read_flits:
                    read_flits[flit.packet_id].append(flit)
                else:
                    read_flits[flit.packet_id] = [flit]
            elif flit.packet_id in read_flits:
                write_flits[flit.packet_id].append(flit)
            else:
                write_flits[flit.packet_id] = [flit]
        read_result = self.analysis(read_flits)
        write_result = self.analysis(write_flits)
        gdma_throughput, sdma_throughput, ddr_throughput, ddr1_throughput, ddr2_throughput = 0, 0, 0, 0, 0
        for flit in sorted_flits:
            destination_key = flit.destination + self.config.cols
            if flit.destination_type not in network.throughput:
                print(f"Warning: destination_type '{flit.destination_type}' is not initialized.")
                continue
            network.throughput[flit.destination_type][destination_key][1] += 1
            first_time = network.throughput[flit.destination_type][flit.destination + self.config.cols][2]
            network.throughput[flit.destination_type][destination_key][2] = min(flit.departure_inject_cycle, first_time)
            last_time = network.throughput[flit.destination_type][flit.destination + self.config.cols][3]
            network.throughput[flit.destination_type][destination_key][3] = max(flit.arrival_cycle, last_time)
        for source in self.config.gdma_send_positions:
            network.throughput["gdma"][source][0] = (
                network.throughput["gdma"][source][1] * self.config.flit_size * self.config.network_frequency / (network.throughput["gdma"][source][3] - network.throughput["gdma"][source][2])
            )
            gdma_throughput += network.throughput["sdma"][source][0]
        for source in self.config.sdma_send_positions:
            network.throughput["sdma"][source][0] = (
                network.throughput["sdma"][source][1] * self.config.flit_size * self.config.network_frequency / (network.throughput["sdma"][source][3] - network.throughput["sdma"][source][2])
            )
            sdma_throughput += network.throughput["sdma"][source][0]
        # for source in self.config.ddr_send_positions:
        #     network.throughput["ddr"][source][0] = (
        #         network.throughput["ddr"][source][1]
        #         * self.config.flit_size
        #         * self.config.network_frequency
        #         / (network.throughput["ddr"][source][3] - network.throughput["ddr"][source][2])
        #     )
        #     ddr_throughput += network.throughput["ddr"][source][0]
        for source in self.config.ddr_send_positions:
            network.throughput["ddr"][source][0] = (
                network.throughput["ddr"][source][1] * self.config.flit_size * self.config.network_frequency / (network.throughput["ddr"][source][3] - network.throughput["ddr"][source][2])
            )
            ddr1_throughput += network.throughput["ddr"][source][0]
        for source in self.config.l2m_send_positions:
            network.throughput["l2m"][source][0] = (
                network.throughput["l2m"][source][1] * self.config.flit_size * self.config.network_frequency / (network.throughput["l2m"][source][3] - network.throughput["l2m"][source][2])
            )
            ddr2_throughput += network.throughput["l2m"][source][0]
        gdma_throughput /= self.config.num_ips
        sdma_throughput /= self.config.num_ips
        ddr1_throughput /= self.config.num_ips
        ddr2_throughput /= self.config.num_ips

        throughput_data = {
            "ddr": {"ddr_throughput": 0, "first_send": 0, "recv_num": 0, "last_recv": 0, "num": self.config.num_ips},
            "l2m": {"l2m_throughput": 0, "first_send": 0, "recv_num": 0, "last_recv": 0, "num": self.config.num_ips},
            "sdma": {"sdma_throughput": 0, "first_send": 0, "recv_num": 0, "last_recv": 0, "num": self.config.num_ips},
            "gdma": {"gdma_throughput": 0, "first_send": 0, "recv_num": 0, "last_recv": 0, "num": self.config.num_ips},
        }
        begin_id = 0

        for i in range(len(self.throughput_time)):
            end_id = self.throughput_time[i] + begin_id
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
                            throughput_data[key]["recv_num"] * 256 / (throughput_data["sdma"]["last_recv"] - throughput_data["ddr"]["first_send"]) / throughput_data[key]["num"]
                        )
                    # elif key == "ddr":
                    #     throughput_data[key][f"{key}_throughput"] = (
                    #         throughput_data[key]["recv_num"]
                    #         * 256
                    #         / (throughput_data["ddr"]["last_recv"] - throughput_data["sdma"]["first_send"])
                    #         / throughput_data[key]["num"]
                    #     )
                    elif key == "ddr":
                        throughput_data[key][f"{key}_throughput"] = (
                            throughput_data[key]["recv_num"] * 256 / (throughput_data["ddr"]["last_recv"] - throughput_data["sdma"]["first_send"]) / throughput_data[key]["num"]
                        )
                    elif key == "l2m":
                        throughput_data[key][f"{key}_throughput"] = (
                            throughput_data[key]["recv_num"] * 256 / (throughput_data["l2m"]["last_recv"] - throughput_data["gdma"]["first_send"]) / throughput_data[key]["num"]
                        )
                    elif key == "gdma":
                        throughput_data[key][f"{key}_throughput"] = (
                            throughput_data[key]["recv_num"] * 256 / (throughput_data["gdma"]["last_recv"] - throughput_data["l2m"]["first_send"]) / throughput_data[key]["num"]
                        )
            begin_id = end_id
        # print(throughput_data)
        return

    def analysis(self, packets):
        if not packets:
            return [0] * 8
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
            diff_Latency[packet_id] = diff / self.config.network_frequency
            len_packets += len(packets[packet_id])
        predicted_avg_latency = all_predicted_latency / len_packets / self.config.network_frequency
        predicted_max_latency = predicted_max_latency / self.config.network_frequency
        actual_avg_latency = all_actual_latency / len_packets / self.config.network_frequency
        actual_max_latency = actual_max_latency / self.config.network_frequency
        actual_avg_ject_latency = all_actual_ject_latency / len_packets / self.config.network_frequency
        actual_max_ject_latency = actual_max_ject_latency / self.config.network_frequency
        actual_avg_net_latency = all_actual_net_latency / len_packets / self.config.network_frequency
        actual_max_net_latency = actual_max_net_latency / self.config.network_frequency

        return [
            predicted_avg_latency,
            predicted_max_latency,
            actual_avg_latency,
            actual_max_latency,
            actual_avg_ject_latency,
            actual_max_ject_latency,
            actual_avg_net_latency,
            actual_max_net_latency,
        ]

    def node_map(self, node, is_source=True):
        if is_source:
            if self.topo_type_stat in ["5x4", "4x5"]:
                return self.config.gdma_send_positions[node] if node < 16 else self.config.sdma_send_positions[node % 16]
            elif self.topo_type_stat == "6x5":
                return node % self.config.cols + self.config.cols + node // self.config.cols * 2 * self.config.cols
            return self.config.gdma_send_positions[node]
        else:
            if self.topo_type_stat in ["5x4", "4x5"]:
                return self.config.ddr_send_positions[node] - self.config.cols if node < 16 else self.config.l2m_send_positions[node % 16] - self.config.cols
            elif self.topo_type_stat == "6x5":
                return node % self.config.cols + node // self.config.cols * 2 * self.config.cols
            return self.config.ddr_send_positions[node] - self.config.cols

    def draw_figure(self):
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        # ax1.plot(self.flit_network.current_cycle, self.flit_network.flits_num, linestyle="-", color="b", label="Sample Data")
        # ax1.set_title("Figure 1")
        # ax1.set_xlabel("cycle")
        # ax1.set_ylabel("packet num on flit_network")
        # ax1.grid(True)
        # ax1.legend()

        fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(8, 10))

        x_values = list(range(self.config.num_ips))
        y_values = []
        y_values.extend(self.flit_network.avg_inject_time[ip_pos] for ip_pos in self.flit_position)
        ax2.bar(x_values, y_values, color="blue")
        ax2.set_title("Average Inject Time")
        ax2.set_xlabel("Node")
        ax2.set_ylabel("Latency")
        ax2.set_xticks(list(range(self.config.num_ips)))
        ax2.set_xticklabels(list(range(self.config.num_ips)), rotation=90)

        x_values = list(range(self.config.num_ips))
        y_values = [self.flit_network.avg_eject_time[in_pos - self.config.cols] for in_pos in self.flit_position]
        ax3.bar(x_values, y_values, color="red")
        ax3.set_title("Average Eject Time")
        ax3.set_xlabel("Node")
        ax3.set_ylabel("Latency")
        ax3.set_xticks(list(range(self.config.num_ips)))
        ax3.set_xticklabels(list(range(self.config.num_ips)), rotation=90)

        plt.tight_layout()
        plt.show(block=True)

    def get_results(self):
        """
        Extract simulation statistics and configuration variables.

        Returns:
            dict: A combined dictionary of configuration variables and statistics.
        """
        # Get all variables from the sim instance
        self.config.finish_del()

        sim_vars = vars(self)

        # Extract statistics (ending with "_stat")
        results = {key.rsplit("_stat", 1)[0]: value for key, value in sim_vars.items() if key.endswith("_stat")}

        # Add configuration variables
        config_var = {key: value for key, value in vars(self.config).items()}
        results = {**config_var, **results}

        # Clear flit and packet IDs (assuming these are class methods)
        Flit.clear_flit_id()
        Node.clear_packet_id()

        return results
