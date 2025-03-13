import itertools
import numpy as np
from collections import deque

from src.utils.optimal_placement import create_adjacency_matrix, find_shortest_paths
from config.config import SimulationConfig
from src.utils.component import Flit, Network, Node
from src.core.base_model import BaseModel
import matplotlib.pyplot as plt
import random
import json
import os
import sys

import cProfile


class Packet_Base_model(BaseModel):

    def run(self):
        """Main simulation loop."""
        self.load_request_stream()
        flits, reqs, rsps = [], [], []
        self.cycle = 0
        tail_time = 10

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
                and self.send_flits_num == self.flit_network.recv_flits_num >= self.read_flit + self.write_flit  # - 200
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

    def process_received_data(self):
        """Process received data in RN and SN networks."""
        positions = self.flit_position
        for in_pos in positions:
            self.process_rn_received_data(in_pos)
            self.process_sn_received_data(in_pos)

    def process_rn_received_data(self, in_pos):
        """Handle received data in the RN network."""
        if len(self.node.rn_rdb_recv[self.rn_type][in_pos]) > 0:
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
                self.req_cir_h_total += req.circuits_completed_h
                self.req_cir_v_total += req.circuits_completed_v
                for flit in self.flit_network.arrive_flits[packet_id]:
                    flit.leave_db_cycle = self.cycle
                req.leave_db_cycle = self.cycle
                self.node.rn_tracker["read"][self.rn_type][in_pos].remove(req)
                self.node.rn_tracker_count["read"][self.rn_type][in_pos] += 1
                self.node.rn_tracker_pointer["read"][self.rn_type][in_pos] -= 1

    def process_sn_received_data(self, in_pos):
        """Handle received data in the SN network."""
        if len(self.node.sn_wdb_recv[self.sn_type][in_pos]) > 0:
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
                self.req_cir_h_total += req.circuits_completed_h
                self.req_cir_v_total += req.circuits_completed_v
                for flit in self.flit_network.arrive_flits[packet_id]:
                    flit.leave_db_cycle = self.cycle + self.config.sn_tracker_release_latency
                # 释放tracker 增加40ns延迟
                release_time = self.cycle + self.config.sn_tracker_release_latency
                self.node.sn_tracker_release_time[release_time].append((self.sn_type, in_pos, req))

                # 没有延迟
                # self.node.sn_tracker[self.sn_type][in_pos].remove(req)
                # self.node.sn_tracker_count[self.sn_type][req.sn_tracker_type][in_pos] += 1

                # packet base: 收完数据发送rsp
                self.create_rsp(req, "finish")

                # if self.node.sn_wdb_count[self.sn_type][in_pos] > 0 and self.node.sn_req_wait["write"][self.sn_type][in_pos]:
                #     new_req = self.node.sn_req_wait["write"][self.sn_type][in_pos].pop(0)
                #     new_req.sn_tracker_type = req.sn_tracker_type
                #     new_req.req_attr = "old"
                #     self.node.sn_tracker[self.sn_type][in_pos].append(new_req)
                #     self.node.sn_tracker_count[self.sn_type][new_req.sn_tracker_type][in_pos] -= 1
                #     self.node.sn_wdb[self.sn_type][in_pos][new_req.packet_id] = []
                #     self.node.sn_wdb_count[self.sn_type][in_pos] -= new_req.burst_length
                #     self.create_rsp(new_req, "positive")

    def check_and_release_sn_tracker(self):
        """Check if any trackers can be released based on the current cycle."""
        for release_time in sorted(self.node.sn_tracker_release_time.keys()):
            if release_time <= self.cycle:
                tracker_list = self.node.sn_tracker_release_time.pop(release_time)
                for sn_type, in_pos, req in tracker_list:
                    self.node.sn_tracker[sn_type][in_pos].remove(req)
                    self.node.sn_tracker_count[sn_type][req.sn_tracker_type][in_pos] += 1
            else:
                break

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
                            req.entry_db_cycle = self.cycle
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
                            req.entry_db_cycle = self.cycle
                            self.create_write_packet(req)
                            # req.sn_tracker_type = "share"
                            self.node.rn_wdb_send[self.rn_type][ip_pos].append(req.packet_id)
            # self.select_inject_network(ip_pos)
            self.select_inject_network_packet_base(ip_pos)

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
        for ip_pos in self.config.ddr_send_positions:
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
        for ip_pos in set(self.config.ddr_send_positions + self.config.l2m_send_positions + self.config.sdma_send_positions + self.config.gdma_send_positions):
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
                            if i == 0:
                                self.send_read_flits_num += 1
                                self.node.sn_rdb[self.sn_type][ip_pos].pop(0)
                                if flit.is_last_flit:
                                    # finish current req injection
                                    req = next((req for req in self.node.sn_tracker[self.sn_type][ip_pos] if req.packet_id == flit.packet_id), None)
                                    self.node.sn_tracker[self.sn_type][ip_pos].remove(req)
                                    self.node.sn_tracker_count[self.sn_type][req.sn_tracker_type][ip_pos] += 1
                                    # if self.node.sn_req_wait["read"][self.sn_type][ip_pos]:
                                    #     # If there is a waiting request, inject it
                                    #     new_req = self.node.sn_req_wait["read"][self.sn_type][ip_pos].pop(0)
                                    #     new_req.sn_tracker_type = req.sn_tracker_type
                                    #     new_req.req_attr = "old"
                                    #     self.node.sn_tracker[self.sn_type][ip_pos].append(new_req)
                                    #     self.node.sn_tracker_count[self.sn_type][req.sn_tracker_type][ip_pos] -= 1
                                    #     self.create_rsp(new_req, "positive")
                            else:
                                self.send_write_flits_num += 1
                                self.node.rn_wdb[self.rn_type][ip_pos][self.node.rn_wdb_send[self.rn_type][ip_pos][0]].pop(0)

                                # packet base: 发完数据不更新rn tracker
                                if flit.is_last_flit:
                                    #     # finish current req injection
                                    req = next((req for req in self.node.rn_tracker["write"][self.rn_type][ip_pos] if req.packet_id == flit.packet_id), None)
                                    #     self.node.rn_tracker["write"][self.rn_type][ip_pos].remove(req)
                                    #     self.node.rn_tracker_count["write"][self.rn_type][ip_pos] += 1
                                    #     self.node.rn_tracker_pointer["write"][self.rn_type][ip_pos] -= 1
                                    self.node.rn_wdb_send[self.rn_type][ip_pos].pop(0)
                                    self.node.rn_wdb[self.rn_type][ip_pos].pop(req.packet_id)
                                    self.node.rn_wdb_count[self.rn_type][ip_pos] += req.burst_length
                            self.trans_flits_num += 1
                            self.send_flits_num += 1
                            inject_flits[i] = None
                            break

    def select_inject_network_packet_base(self, ip_pos):
        read_valid = len(self.node.rn_tracker["read"][self.rn_type][ip_pos]) - 1 > self.node.rn_tracker_pointer["read"][self.rn_type][ip_pos]
        if read_valid:
            rn_tracker_pointer = self.node.rn_tracker_pointer["read"][self.rn_type][ip_pos] + 1
            req = self.node.rn_tracker["read"][self.rn_type][ip_pos][rn_tracker_pointer]
            if req:
                for direction in self.directions:
                    queue = self.req_network.inject_queues[direction]
                    queue_pre = self.req_network.inject_queues_pre[direction]
                    if self.direction_conditions[direction](req) and len(queue[ip_pos]) < self.config.IQ_OUT_FIFO_DEPTH:
                        queue_pre[ip_pos] = req
                        self.node.rn_tracker_pointer["read"][self.rn_type][ip_pos] += 1

        # write_valid = len(self.node.rn_tracker["write"][self.rn_type][ip_pos]) - 1 > self.node.rn_tracker_pointer["write"][self.rn_type][ip_pos]

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
                elif read_new:
                    rn_tracker_pointer = self.node.rn_tracker_pointer["read"][self.rn_type][ip_pos] + 1
                    req = self.node.rn_tracker["read"][self.rn_type][ip_pos][rn_tracker_pointer]
                    if req:
                        for direction in self.directions:
                            queue = self.req_network.inject_queues[direction]
                            queue_pre = self.req_network.inject_queues_pre[direction]
                            if self.direction_conditions[direction](req) and len(queue[ip_pos]) < self.config.IQ_OUT_FIFO_DEPTH:
                                queue_pre[ip_pos] = req
                                self.node.rn_tracker_pointer["read"][self.rn_type][ip_pos] += 1
                                self.req_network.last_select[self.rn_type][ip_pos] = "read"
            elif write_old:
                req = next(
                    (req for req in self.node.rn_tracker_wait["write"][self.rn_type][ip_pos] if req.req_state == "valid"),
                    None,
                )
                if req:
                    for direction in self.directions:
                        queue = self.req_network.inject_queues[direction]
                        queue_pre = self.req_network.inject_queues_pre[direction]
                        if self.direction_conditions[direction](req) and len(queue[ip_pos]) < self.config.IQ_OUT_FIFO_DEPTH:
                            queue_pre[ip_pos] = req
                            self.node.rn_tracker_wait["write"][self.rn_type][ip_pos].remove(req)
                            self.node.rn_wdb_reserve[self.rn_type][ip_pos] -= 1
                            self.req_network.last_select[self.rn_type][ip_pos] = "write"
            elif write_new:
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
            elif read_new:
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
            elif write_new:
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
        ring_bridge_flits, vertical_flits, horizontal_flits, new_flits, local_flits = [], [], [], [], []
        for flit in flits:
            if flit.source - flit.destination == self.config.cols:
                flit.is_new_on_network = False
                flit.is_arrive = True
                local_flits.append(flit)
            elif not flit.current_link:
                new_flits.append(flit)
            elif flit.current_link[0] - flit.current_link[1] == self.config.cols:
                # Ring bridge: 横向环到纵向环
                ring_bridge_flits.append(flit)
            elif abs(flit.current_link[0] - flit.current_link[1]) == 1:
                # 横向环
                horizontal_flits.append(flit)
            else:
                # 纵向环
                vertical_flits.append(flit)
        return ring_bridge_flits, vertical_flits, horizontal_flits, new_flits, local_flits

    def flit_move(self, network, flits, flit_type):
        # 分类不同类型的flits
        ring_bridge_flits, vertical_flits, horizontal_flits, new_flits, local_flits = self.classify_flits(flits)

        # 处理新到达的flits
        for flit in new_flits + horizontal_flits:
            network.plan_move(flit)

        # 处理transfer station的flits
        for col in range(1, self.config.rows, 2):
            for row in range(self.config.cols):
                pos = col * self.config.cols + row
                next_pos = pos - self.config.cols
                eject_flit, vup_flit, vdown_flit = None, None, None

                # 获取各方向的flit
                # station_flits = [
                #     network.ring_bridge["up"][(pos, next_pos)][0] if network.ring_bridge["up"][(pos, next_pos)] else None,
                #     network.ring_bridge["left"][(pos, next_pos)][0] if network.ring_bridge["left"][(pos, next_pos)] else None,
                #     network.ring_bridge["right"][(pos, next_pos)][0] if network.ring_bridge["right"][(pos, next_pos)] else None,
                #     network.ring_bridge["ft"][(pos, next_pos)][0] if network.ring_bridge["ft"][(pos, next_pos)] else None,
                # ]
                station_flits = [network.ring_bridge[fifo_pos][(pos, next_pos)][0] if network.ring_bridge[fifo_pos][(pos, next_pos)] else None for fifo_pos in ["up", "left", "right", "ft"]]
                # if not all(flit is None for flit in station_flits):
                #     print(station_flits)

                # 处理eject操作
                if len(network.ring_bridge["eject"][(pos, next_pos)]) < self.config.RB_OUT_FIFO_DEPTH:
                    eject_flit = self._process_eject_flit(network, station_flits, pos, next_pos)

                # 处理vup操作
                if len(network.ring_bridge["vup"][(pos, next_pos)]) < self.config.RB_OUT_FIFO_DEPTH:
                    if vup_flit:
                        print(vup_flit)
                    vup_flit = self._process_vup_flit(network, station_flits, pos, next_pos)

                # 处理vdown操作
                if len(network.ring_bridge["vdown"][(pos, next_pos)]) < self.config.RB_OUT_FIFO_DEPTH:
                    if vdown_flit:
                        print(vdown_flit)
                    vdown_flit = self._process_vdown_flit(network, station_flits, pos, next_pos)

                # transfer_eject
                # 处理eject队列
                # TODO: eject_queue -> ETag
                if next_pos in network.eject_queues["mid"] and len(network.eject_queues["mid"][next_pos]) < self.config.EQ_IN_FIFO_DEPTH and network.ring_bridge["eject"][(pos, next_pos)]:
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

        # eject arbitration
        if flit_type in ["req", "rsp", "data"]:
            self._handle_eject_arbitration(network, flit_type)

        # 执行所有flit的移动
        for flit in vertical_flits + horizontal_flits + new_flits + local_flits:
            if network.execute_moves(flit, self.cycle):
                flits.remove(flit)

        # 处理transfer station的flits
        for flit in ring_bridge_flits:
            if flit.is_arrive:
                flit.arrival_network_cycle = self.cycle
                network.eject_queues["mid"][flit.destination].append(flit)
                flits.remove(flit)

        return flits

    def _process_eject_flit(self, network, station_flits, pos, next_pos):
        """处理eject操作"""
        eject_flit = None

        if station_flits[3] and station_flits[3].destination == next_pos:
            eject_flit = station_flits[3]
            station_flits[3] = None
            network.ring_bridge["ft"][(pos, next_pos)].popleft()
        else:
            index = network.round_robin["mid"][next_pos]
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
            network.ring_bridge["up"][(pos, next_pos)].popleft()
        elif index == 1:
            network.ring_bridge["left"][(pos, next_pos)].popleft()
        elif index == 2:
            network.ring_bridge["right"][(pos, next_pos)].popleft()
        network.round_robin["mid"][next_pos].remove(index)
        network.round_robin["mid"][next_pos].append(index)

    def _handle_eject_arbitration(self, network, flit_type):
        """处理eject的仲裁逻辑,根据flit类型处理不同的eject队列"""
        if flit_type == "req":
            for in_pos in set(self.config.ddr_send_positions + self.config.l2m_send_positions):
                ip_pos = in_pos - self.config.cols
                # eject_flits = [
                #     network.eject_queues["up"][ip_pos][0] if network.eject_queues["up"][ip_pos] else None,
                #     network.eject_queues["mid"][ip_pos][0] if network.eject_queues["mid"][ip_pos] else None,
                #     network.eject_queues["down"][ip_pos][0] if network.eject_queues["down"][ip_pos] else None,
                #     network.eject_queues["local"][ip_pos][0] if network.eject_queues["local"][ip_pos] else None,
                # ]
                eject_flits = [network.eject_queues[fifo_pos][ip_pos][0] if network.eject_queues[fifo_pos][ip_pos] else None for fifo_pos in ["up", "mid", "down", "local"]]

                # if not all(eject_flit is None for eject_flit in eject_flits):
                #     print(eject_flits)
                eject_flits = self.process_eject_queues(network, eject_flits, network.round_robin["ddr"][ip_pos], "ddr", ip_pos)
                eject_flits = self.process_eject_queues(network, eject_flits, network.round_robin["l2m"][ip_pos], "l2m", ip_pos)

            if self.sn_type != "Idle":
                for in_pos in self.config.ddr_send_positions:
                    ip_pos = in_pos - self.config.cols
                    if network.ip_eject[self.sn_type][ip_pos]:
                        req = network.ip_eject[self.sn_type][ip_pos][0]
                        if self._handle_request(req, in_pos):
                            network.ip_eject[self.sn_type][ip_pos].popleft()

        elif flit_type == "rsp":
            for in_pos in set(self.config.sdma_send_positions + self.config.gdma_send_positions):
                ip_pos = in_pos - self.config.cols
                # eject_flits = [
                # network.eject_queues["up"][ip_pos][0] if network.eject_queues["up"][ip_pos] else None,
                # network.eject_queues["mid"][ip_pos][0] if network.eject_queues["mid"][ip_pos] else None,
                # network.eject_queues["down"][ip_pos][0] if network.eject_queues["down"][ip_pos] else None,
                # network.eject_queues["local"][ip_pos][0] if network.eject_queues["local"][ip_pos] else None,
                # ]
                eject_flits = [network.eject_queues[fifo_pos][ip_pos][0] if network.eject_queues[fifo_pos][ip_pos] else None for fifo_pos in ["up", "mid", "down", "local"]]
                # if not all(eject_flit is None for eject_flit in eject_flits):
                #     print(eject_flits)
                eject_flits = self.process_eject_queues(network, eject_flits, network.round_robin["sdma"][ip_pos], "sdma", ip_pos)
                eject_flits = self.process_eject_queues(network, eject_flits, network.round_robin["gdma"][ip_pos], "gdma", ip_pos)

            if self.rn_type != "Idle":
                for in_pos in getattr(self.config, f"{self.rn_type}_send_positions"):
                    ip_pos = in_pos - self.config.cols
                    if network.ip_eject[self.rn_type][ip_pos]:
                        rsp = network.ip_eject[self.rn_type][ip_pos].popleft()
                        # self._handle_response_packet_base(rsp, in_pos)
                        self._handle_response(rsp, in_pos)

        elif flit_type == "data":
            for in_pos in self.flit_position:
                ip_pos = in_pos - self.config.cols
                # eject_flits = [
                # network.eject_queues["up"][ip_pos][0] if network.eject_queues["up"][ip_pos] else None,
                # network.eject_queues["mid"][ip_pos][0] if network.eject_queues["mid"][ip_pos] else None,
                # network.eject_queues["down"][ip_pos][0] if network.eject_queues["down"][ip_pos] else None,
                # network.eject_queues["local"][ip_pos][0] if network.eject_queues["local"][ip_pos] else None,
                # ]
                eject_flits = [network.eject_queues[fifo_pos][ip_pos][0] if network.eject_queues[fifo_pos][ip_pos] else None for fifo_pos in ["up", "mid", "down", "local"]]
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
                            flit = network.ip_eject[ip_type][ip_pos][0]
                            if flit.req_type == "read":
                                network.ip_eject[ip_type][ip_pos].popleft()
                                flit.arrival_cycle = self.cycle
                                network.arrive_node_pre[ip_type][ip_pos] = flit
                                network.eject_num += 1
                                network.arrive_flits[flit.packet_id].append(flit)
                                network.recv_flits_num += 1
                            elif flit.req_type == "write":
                                # packet base: 到达的flit
                                if flit.flit_id_in_packet == 0:
                                    if self.node.sn_wdb_count[self.sn_type][in_pos] > 0:
                                        flit.sn_tracker_type = "share"
                                        req = next(
                                            (req for req in self.node.rn_tracker[flit.req_type][flit.source_type][flit.source] if req.packet_id == flit.packet_id),
                                            None,
                                        )
                                        req.sn_tracker_type = "share"
                                        self.node.sn_tracker[self.sn_type][in_pos].append(req)
                                        self.node.sn_tracker_count[self.sn_type]["share"][in_pos] -= 1
                                        # self.node.sn_wdb[self.sn_type][in_pos][req.packet_id] = []
                                        self.node.sn_wdb_count[self.sn_type][in_pos] -= 1

                                        network.ip_eject[ip_type][ip_pos].popleft()
                                        flit.arrival_cycle = self.cycle
                                        network.arrive_node_pre[ip_type][ip_pos] = flit
                                        network.eject_num += 1
                                        network.arrive_flits[flit.packet_id].append(flit)
                                        network.recv_flits_num += 1

                                else:
                                    if self.node.sn_wdb_count[self.sn_type][in_pos] > 0:
                                        # self.node.sn_wdb[self.sn_type][in_pos][req.packet_id] = []
                                        self.node.sn_wdb_count[self.sn_type][in_pos] -= 1
                                        network.ip_eject[ip_type][ip_pos].popleft()
                                        flit.arrival_cycle = self.cycle
                                        network.arrive_node_pre[ip_type][ip_pos] = flit
                                        network.eject_num += 1
                                        network.arrive_flits[flit.packet_id].append(flit)
                                        network.recv_flits_num += 1

                            # if flit.req_type == "read" and flit.is_last_flit:
                            #     self.create_write_req_after_read(flit)
            for in_pos in self.flit_position:
                ip_pos = in_pos - self.config.cols
                for ip_type in network.eject_queues_pre:
                    if network.eject_queues_pre[ip_type][ip_pos]:
                        network.ip_eject[ip_type][ip_pos].append(network.eject_queues_pre[ip_type][ip_pos])
                        network.eject_queues_pre[ip_type][ip_pos] = None

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
                if flit := network.arrive_node_pre[self.rn_type][ip_pos]:
                    self.node.rn_rdb[self.rn_type][in_pos][flit.packet_id].append(flit)
                    if len(self.node.rn_rdb[self.rn_type][in_pos][flit.packet_id]) == self.node.rn_rdb[self.rn_type][in_pos][flit.packet_id][0].burst_length:
                        self.node.rn_rdb_recv[self.rn_type][in_pos].append(flit.packet_id)
                    network.arrive_node_pre[self.rn_type][ip_pos] = None
                if flit := network.arrive_node_pre[self.sn_type][ip_pos]:
                    self.node.sn_wdb[self.sn_type][in_pos][flit.packet_id].append(flit)
                    if len(self.node.sn_wdb[self.sn_type][in_pos][flit.packet_id]) == self.node.sn_wdb[self.sn_type][in_pos][flit.packet_id][0].burst_length:
                        self.node.sn_wdb_recv[self.sn_type][in_pos].append(flit.packet_id)
                    network.arrive_node_pre[self.sn_type][ip_pos] = None

    def _handle_request(self, req, in_pos):
        """处理request类型的eject"""
        if req.req_type == "read":
            if req.req_attr == "new" and self.node.sn_tracker_count[self.sn_type]["ro"][in_pos] > 0:
                req.sn_tracker_type = "ro"
                self.node.sn_tracker[self.sn_type][in_pos].append(req)
                self.node.sn_tracker_count[self.sn_type]["ro"][in_pos] -= 1
                self.create_read_packet(req)
                return True
        return False

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
                        if network.links_tag[link][-1] == [next_pos, direction]:
                            if network.config.EQ_IN_FIFO_DEPTH - len(eject_queue) > len(reservations):
                                network.remain_tag[direction][next_pos] += 1
                                network.links_tag[link][-1] = None
                                return self._update_flit_state(network, dir_key, pos, next_pos, opposite_node, direction)
                elif flit_l.destination == next_pos:
                    eject_queue = network.eject_queues[direction][next_pos]
                    reservations = network.eject_reservations[direction][next_pos]
                    # TODO: EQ_IN_FIFO_DEPTH -> ETag
                    return (
                        self._update_flit_state(
                            network,
                            dir_key,
                            pos,
                            next_pos,
                            opposite_node,
                            direction,
                        )
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
        if network.ring_bridge[ts_key][(pos, next_pos)][0].wait_cycle_v > self.config.wait_cycle_v and not network.ring_bridge[ts_key][(pos, next_pos)][0].is_tag_v:
            if network.remain_tag[direction][next_pos] > 0:
                network.remain_tag[direction][next_pos] -= 1
                network.links_tag[link][-1] = [next_pos, direction]
                network.ring_bridge[ts_key][(pos, next_pos)][0].is_tag_v = True
        else:
            for flit in network.ring_bridge[ts_key][(pos, next_pos)]:
                flit.wait_cycle_v += 1
        return False

    def _handle_response_packet_base(self, rsp, in_pos):
        if rsp.req_type == "read":
            req = next((req for req in self.node.rn_tracker["read"][self.rn_type][in_pos] if req.packet_id == rsp.packet_id), None)
            req.req_state = "valid"
            self.node.rn_rdb_reserve[self.rn_type][in_pos] += 1
            if req not in self.node.rn_tracker_wait["read"][self.rn_type][in_pos]:
                req.is_injected = False
                req.path_index = 0
                req.early_rsp = True
                self.node.rn_tracker_wait["read"][self.rn_type][in_pos].append(req)
        elif rsp.req_type == "write":
            req = next((req for req in self.node.rn_tracker["write"][self.rn_type][in_pos] if req.packet_id == rsp.packet_id), None)
            req.req_state = "valid"
            self.node.rn_wdb_reserve[self.rn_type][in_pos] += 1
            if req not in self.node.rn_tracker_wait["write"][self.rn_type][in_pos]:
                req.is_injected = False
                req.path_index = 0
                req.early_rsp = True
                self.node.rn_tracker_wait["write"][self.rn_type][in_pos].append(req)

    def _handle_response(self, rsp, in_pos):
        """处理response的eject"""
        req = next(
            (req for req in self.node.rn_tracker[rsp.req_type][self.rn_type][in_pos] if req.packet_id == rsp.packet_id),
            None,
        )
        self.rsp_cir_h_total += rsp.circuits_completed_h
        self.rsp_cir_v_total += rsp.circuits_completed_v
        if not req:
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
            elif rsp.rsp_type == "finish":
                self.node.rn_tracker["write"][self.rn_type][in_pos].remove(req)
                self.node.rn_tracker_count["write"][self.rn_type][in_pos] += 1
                self.node.rn_tracker_pointer["write"][self.rn_type][in_pos] -= 1
            # else:
            #     self.node.rn_wdb_send[self.rn_type][in_pos].append(rsp.packet_id)

    def process_eject_queues(self, network, eject_flits, rr_queue, destination_type, ip_pos):
        for i in rr_queue:
            if eject_flits[i] is not None and eject_flits[i].destination_type == destination_type and len(network.ip_eject[destination_type][ip_pos]) < network.config.ip_eject_len:
                # network.ip_eject[destination_type][ip_pos].append(eject_flits[i])
                network.eject_queues_pre[destination_type][ip_pos] = eject_flits[i]
                eject_flits[i].arrival_eject_cycle = self.cycle
                eject_flits[i] = None
                if i == 0:
                    network.eject_queues["up"][ip_pos].popleft()
                elif i == 1:
                    network.eject_queues["mid"][ip_pos].popleft()
                elif i == 2:
                    network.eject_queues["down"][ip_pos].popleft()
                elif i == 3:
                    network.eject_queues["local"][ip_pos].popleft()
                rr_queue.remove(i)
                rr_queue.append(i)
                break
        return eject_flits

    # def create_write_req_after_read(self, flit):
    #     source = self.node_change(flit.destination_original)
    #     destination = self.node_change(flit.source_original, False)
    #     path = self.routes[source][destination]
    #     req = Flit(source, destination, path)
    #     req.source_original = flit.destination + self.config.cols
    #     req.destination_original = flit.source - self.config.cols
    #     req.flit_type = "req"
    #     req.departure_cycle = self.cycle + 1
    #     req.burst_length = flit.burst_length
    #     req.source_type = flit.destination_type
    #     req.destination_type = flit.source_type
    #     req.original_source_type = flit.original_destination_type
    #     req.original_destination_type = flit.original_source_type
    #     if self.topo_type in ["5x4", "4x5"]:
    #         req.source_type = "sdma" if req.source_original > 15 else "gdma"
    #         req.destination_type = "ddr" if req.destination_original > 15 else "l2m"
    #     req.packet_id = Node.get_next_packet_id()
    #     req.req_type = "write"
    #     self.new_write_req.append(req)

    # def create_write_packet(self, req):
    #     for i in range(req.burst_length):
    #         source = req.source
    #         destination = req.destination
    #         path = self.routes[source][destination]
    #         flit = Flit(source, destination, path)
    #         flit.source_original = req.source_original
    #         flit.destination_original = req.destination_original
    #         flit.flit_type = "data"
    #         flit.departure_cycle = self.cycle
    #         flit.req_departure_cycle = req.departure_cycle
    #         flit.entry_db_cycle = req.entry_db_cycle
    #         flit.sn_tracker_type = "share"
    #         flit.source_type = req.source_type
    #         flit.destination_type = req.destination_type
    #         flit.original_source_type = req.original_source_type
    #         flit.original_destination_type = req.original_destination_type
    #         flit.req_type = req.req_type
    #         flit.packet_id = req.packet_id
    #         flit.flit_id_in_packet = i
    #         flit.burst_length = req.burst_length
    #         if i == req.burst_length - 1:
    #             flit.is_last_flit = True
    #         self.node.rn_wdb[flit.source_type][flit.source][flit.packet_id].append(flit)

    # def create_read_packet(self, req):
    #     for i in range(req.burst_length):
    #         source = req.destination + self.config.cols
    #         destination = req.source - self.config.cols
    #         path = self.routes[source][destination]
    #         flit = Flit(source, destination, path)
    #         flit.source_original = req.destination_original
    #         flit.destination_original = req.source_original
    #         flit.req_type = req.req_type
    #         flit.flit_type = "data"
    #         flit.departure_cycle = self.cycle + self.config.ddr_latency + i if req.destination_type == "ddr" else self.cycle + i
    #         flit.req_departure_cycle = req.departure_cycle
    #         flit.entry_db_cycle = req.entry_db_cycle
    #         flit.source_type = req.destination_type
    #         flit.destination_type = req.source_type
    #         flit.original_source_type = req.original_source_type
    #         flit.original_destination_type = req.original_destination_type
    #         flit.packet_id = req.packet_id
    #         flit.flit_id_in_packet = i
    #         flit.burst_length = req.burst_length
    #         if i == req.burst_length - 1:
    #             flit.is_last_flit = True
    #         self.node.sn_rdb[flit.source_type][flit.source].append(flit)

    # def process_inject_queues(self, network, inject_queues):
    #     flit_num = 0
    #     flits = []
    #     for source, queue in inject_queues.items():
    #         if queue and queue[0]:
    #             flit = queue.popleft()
    #             if flit.inject(network):
    #                 network.inject_num += 1
    #                 flit_num += 1
    #                 flit.departure_network_cycle = self.cycle
    #                 flits.append(flit)
    #             else:
    #                 queue.appendleft(flit)
    #                 for flit in queue:
    #                     flit.wait_cycle += 1
    #     return flit_num, flits
