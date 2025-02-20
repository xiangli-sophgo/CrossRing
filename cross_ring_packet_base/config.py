import json
import os

import argparse


class SimulationConfig:
    def __init__(self, default_config):
        args = self.parse_args(default_config)
        self.num_nodes = args.num_nodes
        self.cols = args.cols
        self.num_ips = args.num_ips
        self.rows = self.num_nodes // self.cols
        self.num_cycles_send = args.num_cycles_send
        self.num_round_cycles = args.num_round_cycles
        self.ddr_send_rate = args.ddr_send_rate
        self.sdma_send_rate = args.sdma_send_rate
        self.l2m_send_rate = args.l2m_send_rate
        self.gdma_send_rate = args.gdma_send_rate
        self.num_ddr = args.num_ddr
        self.num_sdma = args.num_sdma
        self.num_l2m = args.num_l2m
        self.num_gdma = args.num_gdma
        self.flit_size = args.flit_size
        self.seats_per_link = args.seats_per_link
        self.seats_per_station = args.seats_per_station
        self.seats_per_vstation = args.seats_per_vstation
        self.inject_queues_len = args.inject_queues_len
        self.eject_queues_len = args.eject_queues_len
        self.ip_eject_len = args.ip_eject_len
        self.wait_cycle_h = args.wait_cycle_h
        self.wait_cycle_v = args.wait_cycle_v
        self.ft_count = args.ft_count
        self.ft_len = args.ft_len
        self.tags_num = args.tags_num
        self.reservation_num = args.reservation_num
        self.burst = args.burst
        self.network_frequency = args.network_frequency
        self.rn_rdb_size = args.rn_rdb_size
        self.rn_wdb_size = args.rn_wdb_size
        self.sn_wdb_size = args.sn_wdb_size
        self.ddr_latency = args.ddr_latency
        self.sn_tracker_release_latency = args.sn_tracker_release_latency
        self.update_config()

    def update_config(self):
        self.ddr_latency = self.ddr_latency * self.network_frequency
        self.sn_tracker_release_latency = self.sn_tracker_release_latency * self.network_frequency
        self.rn_read_tracker_ostd = self.rn_rdb_size // self.burst
        self.rn_write_tracker_ostd = self.rn_wdb_size // self.burst
        self.ro_tracker_ostd = self.sn_wdb_size // self.burst
        self.share_tracker_ostd = self.sn_wdb_size // self.burst

    def topology_select(self, topo_type="default"):
        if topo_type == "default":
            self.ddr_send_positions = [self.cols * 2 * (x // self.cols) + self.cols + x % self.cols for x in range(self.num_ips)]
            self.sdma_send_positions = [self.cols * 2 * (x // self.cols) + self.cols + x % self.cols for x in range(self.num_ips)]
            self.l2m_send_positions = [self.cols * 2 * (x // self.cols) + self.cols + x % self.cols for x in range(self.num_ips)]
            self.gdma_send_positions = [self.cols * 2 * (x // self.cols) + self.cols + x % self.cols for x in range(self.num_ips)]

        elif topo_type == "4x9":
            self.num_nodes = 72
            self.cols = 9
            self.num_ips = 32
            self.rows = self.num_nodes // self.cols
            self.ddr_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [0])
            self.sdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [0])
            self.l2m_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [0])
            self.gdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [0])

        elif topo_type == "9x4":
            self.num_nodes = 72
            self.cols = 4
            self.num_ips = 32
            self.rows = self.num_nodes // self.cols
            self.ddr_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0] + [17], [])
            self.sdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0] + [17], [])
            self.l2m_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0] + [17], [])
            self.gdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0] + [17], [])
        elif topo_type == "5x4":
            self.num_nodes = 40
            self.cols = 4
            self.num_ips = 32
            self.rows = self.num_nodes // self.cols
            self.ddr_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0] + [9], [])
            self.sdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0] + [9], [])
            self.l2m_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0] + [9], [])
            self.gdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0] + [9], [])
        elif topo_type == "6x5":
            self.num_nodes = 60
            self.cols = 5
            self.num_ips = 8
            self.rows = self.num_nodes // self.cols
            self.ddr_send_positions = [6, 8, 15, 19, 25, 29, 45, 49]
            self.sdma_send_positions = [15, 18, 25, 28, 35, 38, 45, 48]
            self.l2m_send_positions = [16, 17, 26, 27, 36, 37, 46, 47]
            self.gdma_send_positions = [16, 17, 26, 27, 36, 37, 46, 47]
            # self.ddr_send_positions = [self.cols * 2 * (x // self.cols) + self.cols + x % self.cols for x in range(self.num_ips)]
            # self.sdma_send_positions = [self.cols * 2 * (x // self.cols) + self.cols + x % self.cols for x in range(self.num_ips)]
            # self.l2m_send_positions = [self.cols * 2 * (x // self.cols) + self.cols + x % self.cols for x in range(self.num_ips)]
            # self.gdma_send_positions = [self.cols * 2 * (x // self.cols) + self.cols + x % self.cols for x in range(self.num_ips)]
        elif topo_type == "4x5":
            self.num_nodes = 40
            self.cols = 5
            self.num_ips = 8
            self.rows = self.num_nodes // self.cols
            self.ddr_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [0])
            self.sdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [0])
            self.l2m_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [0])
            self.gdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [0])
        elif topo_type == "3x3":
            self.num_nodes = 18
            self.cols = 3
            self.num_ips = 4
            self.rows = self.num_nodes // self.cols
            self.ddr_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [])
            self.sdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [])
            self.l2m_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [])
            self.gdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [])
        else:
            raise ValueError("Error topology type: ", topo_type)

    def generate_ip_positions(self, zero_rows=None, zero_cols=None):
        # 创建一个矩阵，初始值为1
        matrix = [[1 for _ in range(self.cols)] for _ in range(self.rows)]

        # 将指定的行设置为0
        if zero_rows:
            for row in zero_rows:
                if 0 <= row < self.rows:
                    for col in range(self.cols):
                        matrix[row][col] = 0

        # 将指定的列设置为0
        if zero_cols:
            for col in zero_cols:
                if 0 <= col < self.cols:
                    for row in range(self.rows):
                        matrix[row][col] = 0

        # 收集所有元素为1的编号
        indices = []
        for r in range(self.rows):
            for c in range(self.cols):
                if matrix[r][c] == 1:
                    index = r * self.cols + c
                    indices.append(index)
        # assert len(indices) == self.num_ips, f"Expected {self.num_ips} indices, but got {len(indices)}."
        return indices

    def parse_args(self, default_config):
        if os.path.exists(default_config):
            with open(default_config, "r") as f:
                default_config = json.load(f)
        else:
            raise FileNotFoundError(f"{default_config} not found.")

        parser = argparse.ArgumentParser(description="Process simulation configuration parameters.")

        # 将 JSON 配置作为默认值
        parser.add_argument("--num_nodes", type=int, default=default_config["num_nodes"], help="Number of nodes")
        parser.add_argument("--cols", type=int, default=default_config["cols"], help="Number of columns")
        parser.add_argument("--num_ips", type=int, default=default_config["num_ips"], help="Number of IPs")
        parser.add_argument("--num_cycles_send", type=int, default=default_config["num_cycles_send"], help="Number of cycles to send")
        parser.add_argument("--num_round_cycles", type=int, default=default_config["num_round_cycles"], help="Number of round cycles")
        parser.add_argument("--ddr_send_rate", type=float, default=default_config["ddr_send_rate"], help="DDR send rate")
        parser.add_argument("--sdma_send_rate", type=float, default=default_config["sdma_send_rate"], help="SDMA send rate")
        parser.add_argument("--l2m_send_rate", type=float, default=default_config["l2m_send_rate"], help="L2M send rate")
        parser.add_argument("--gdma_send_rate", type=float, default=default_config["gdma_send_rate"], help="GDMA send rate")
        parser.add_argument("--num_ddr", type=int, default=default_config["num_ddr"], help="Number of DDRs")
        parser.add_argument("--num_sdma", type=int, default=default_config["num_sdma"], help="Number of SDMAs")
        parser.add_argument("--num_l2m", type=int, default=default_config["num_l2m"], help="Number of L2Ms")
        parser.add_argument("--num_gdma", type=int, default=default_config["num_gdma"], help="Number of GDMA")
        parser.add_argument("--flit_size", type=int, default=default_config["flit_size"], help="Flit size")
        parser.add_argument("--seats_per_link", type=int, default=default_config["seats_per_link"], help="Seats per link")
        parser.add_argument("--seats_per_station", type=int, default=default_config["seats_per_station"], help="Seats per station")
        parser.add_argument("--seats_per_vstation", type=int, default=default_config["seats_per_vstation"], help="Seats per virtual station")
        parser.add_argument("--inject_queues_len", type=int, default=default_config["inject_queues_len"], help="Length of inject queues")
        parser.add_argument("--eject_queues_len", type=int, default=default_config["eject_queues_len"], help="Length of eject queues")
        parser.add_argument("--ip_eject_len", type=int, default=default_config["ip_eject_len"], help="Length of IP eject queues")
        parser.add_argument("--wait_cycle_h", type=int, default=default_config["wait_cycle_h"], help="Horizontal wait cycles")
        parser.add_argument("--wait_cycle_v", type=int, default=default_config["wait_cycle_v"], help="Vertical wait cycles")
        parser.add_argument("--ft_count", type=int, default=default_config["ft_count"], help="FT count")
        parser.add_argument("--ft_len", type=int, default=default_config["ft_len"], help="FT length")
        parser.add_argument("--tags_num", type=int, default=default_config["tags_num"], help="Number of tags")
        parser.add_argument("--reservation_num", type=int, default=default_config["reservation_num"], help="Reservation number")
        parser.add_argument("--ddr_latency", type=int, default=default_config["ddr_latency"], help="Reservation number")
        parser.add_argument("--sn_tracker_release_latency", type=int, default=default_config["sn_tracker_release_latency"], help="Reservation number")
        parser.add_argument("--burst", type=int, default=default_config["burst"], help="Burst size")
        parser.add_argument("--network_frequency", type=float, default=default_config["network_frequency"], help="Network frequency")
        parser.add_argument("--rn_rdb_size", type=int, default=default_config["rn_rdb_size"], help="RN read buffer size")
        parser.add_argument("--rn_wdb_size", type=int, default=default_config["rn_wdb_size"], help="RN write buffer size")
        parser.add_argument("--sn_wdb_size", type=int, default=default_config["sn_wdb_size"], help="SN write buffer size")

        return parser.parse_args()
