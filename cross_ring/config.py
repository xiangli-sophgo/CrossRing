# config.py

import json


class SimulationConfig:
    def __init__(self, config_file):
        with open(config_file, "r") as f:
            self.config = json.load(f)

        self.num_nodes = self.config["num_nodes"]
        self.cols = self.config["cols"]
        self.num_ips = self.config["num_ips"]
        self.rows = self.num_nodes // self.cols
        self.num_cycles_send = self.config["num_cycles_send"]
        self.num_round_cycles = self.config["num_round_cycles"]
        self.ddr_send_rate = self.config["ddr_send_rate"]
        self.sdma_send_rate = self.config["sdma_send_rate"]
        self.l2m_send_rate = self.config["l2m_send_rate"]
        self.gdma_send_rate = self.config["gdma_send_rate"]
        self.num_ddr = self.config["num_ddr"]
        self.num_sdma = self.config["num_sdma"]
        self.num_l2m = self.config["num_l2m"]
        self.num_gdma = self.config["num_gdma"]
        # self.topology_select()

        # self.packet_size = config["packet_size"]
        self.flit_size = self.config["flit_size"]
        # self.data_num = self.packet_size // self.flit_size
        self.seats_per_link = self.config["seats_per_link"]
        self.seats_per_station = self.config["seats_per_station"]
        self.seats_per_vstation = self.config["seats_per_vstation"]
        self.inject_queues_len = self.config["inject_queues_len"]
        self.eject_queues_len = self.config["eject_queues_len"]
        self.ip_eject_len = self.config["ip_eject_len"]
        self.wait_cycle_h = self.config["wait_cycle_h"]
        self.wait_cycle_v = self.config["wait_cycle_v"]
        self.ft_count = self.config["ft_count"]
        self.ft_len = self.config["ft_len"]
        self.tags_num = self.config["tags_num"]
        self.reservation_num = self.config["reservation_num"]
        self.burst = self.config["burst"]
        self.network_frequency = self.config["network_frequency"]
        self.rn_rdb_size = self.config["rn_rdb_size"]
        self.rn_wdb_size = self.config["rn_wdb_size"]
        self.sn_wdb_size = self.config["sn_wdb_size"]
        self.update_config()

    def update_config(self):
        self.ddr_latency = self.config["ddr_latency"] * self.network_frequency
        self.sn_tracker_release_latency = self.config["sn_tracker_release_latency"] * self.network_frequency
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
            self.num_ips = 32
            self.rows = self.num_nodes // self.cols
            self.ddr_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [0])
            self.sdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [0])
            self.l2m_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [0])
            self.gdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [0])
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
