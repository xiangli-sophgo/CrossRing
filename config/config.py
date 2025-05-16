# config.py

import json
import os
import numpy as np
import argparse
from scipy.optimize import linear_sum_assignment
from collections import deque, defaultdict
from typing import Callable, Iterable, Dict, Any
import copy


class CrossRingConfig:
    def __init__(self, default_config):
        args = self.parse_args(default_config)
        self.topo_type = args.topo_type
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
        self.spare_core_row = -1
        self.fail_core_pos = []
        self.seats_per_link = args.seats_per_link
        self.RB_IN_FIFO_DEPTH = args.RB_IN_FIFO_DEPTH
        self.RB_OUT_FIFO_DEPTH = args.RB_OUT_FIFO_DEPTH
        self.IQ_OUT_FIFO_DEPTH = args.IQ_OUT_FIFO_DEPTH
        self.EQ_IN_FIFO_DEPTH = args.EQ_IN_FIFO_DEPTH
        self.IQ_CH_FIFO_DEPTH = args.IQ_CH_FIFO_DEPTH
        self.EQ_CH_FIFO_DEPTH = args.EQ_CH_FIFO_DEPTH
        self.ITag_Trigger_Th_H = args.ITag_Trigger_Th_H
        self.ITag_Trigger_Th_V = args.ITag_Trigger_Th_V
        self.ft_count = args.ft_count
        self.ft_len = args.ft_len
        self.ITag_Max_Num_H = args.ITag_Max_Num_H
        self.ITag_Max_Num_V = args.ITag_Max_Num_V
        self.reservation_num = args.reservation_num
        self.burst = args.burst
        self.network_frequency = args.network_frequency
        self.rn_rdb_size = args.rn_rdb_size
        self.rn_wdb_size = args.rn_wdb_size
        self.sn_ddr_wdb_size = args.sn_ddr_wdb_size
        self.sn_l2m_wdb_size = args.sn_l2m_wdb_size
        self.ddr_R_latency_original = args.ddr_R_latency
        self.ddr_R_latency_var_original = args.ddr_R_latency_var
        self.ddr_W_latency_original = args.ddr_W_latency
        self.l2m_R_latency_original = args.l2m_R_latency
        self.l2m_W_latency_original = args.l2m_W_latency
        self.sn_tracker_release_latency_original = args.sn_tracker_release_latency
        self.TL_Etag_T1_UE_MAX = args.TL_Etag_T1_UE_MAX
        self.TL_Etag_T2_UE_MAX = args.TL_Etag_T2_UE_MAX
        self.TR_Etag_T2_UE_MAX = args.TR_Etag_T2_UE_MAX
        self.TU_Etag_T1_UE_MAX = args.TU_Etag_T1_UE_MAX
        self.TU_Etag_T2_UE_MAX = args.TU_Etag_T2_UE_MAX
        self.TD_Etag_T2_UE_MAX = args.TD_Etag_T2_UE_MAX
        self.Both_side_ETag_upgrade = args.Both_side_ETag_upgrade
        self.CHANNEL_SPEC = {
            "gdma": 1,  # → RN 侧
            "sdma": 1,  # → RN 侧
            "ddr": 1,  # → SN 侧
            "l2m": 1,  # → SN 侧
        }
        self.channel_names = []
        for key in self.CHANNEL_SPEC:
            for idx in range(self.CHANNEL_SPEC[key]):
                self.channel_names.append(f"{key}_{idx}")
        assert (
            self.TL_Etag_T2_UE_MAX < self.TL_Etag_T1_UE_MAX < self.RB_IN_FIFO_DEPTH
            and self.TL_Etag_T2_UE_MAX < self.RB_IN_FIFO_DEPTH - 2
            and self.TR_Etag_T2_UE_MAX < self.RB_IN_FIFO_DEPTH - 1
            and self.TU_Etag_T2_UE_MAX < self.TU_Etag_T1_UE_MAX < self.EQ_IN_FIFO_DEPTH
            and self.TU_Etag_T2_UE_MAX < self.EQ_IN_FIFO_DEPTH - 2
            and self.TD_Etag_T2_UE_MAX < self.EQ_IN_FIFO_DEPTH - 1
        ), "ETag parameter conditions are not met."

        self.update_config()

    def _make_channels(self, key_types, value_factory=lambda: defaultdict(list)):  # 允许 None / callable / 静态对象
        # 把非 callable 的默认值包装成 deepcopy，可避免共享引用
        if not callable(value_factory):
            static_value = copy.deepcopy(value_factory)
            value_factory = lambda v=static_value: copy.deepcopy(v)

        ports = {}
        for key in key_types:
            for idx in range(self.CHANNEL_SPEC.get(key, 0)):
                ports[f"{key}_{idx}"] = value_factory() if value_factory else None
        return ports

    def update_config(self):
        self.update_latency()
        self.sn_tracker_release_latency = self.sn_tracker_release_latency_original * self.network_frequency
        self.rn_read_tracker_ostd = self.rn_rdb_size // self.burst
        self.rn_write_tracker_ostd = self.rn_wdb_size // self.burst
        self.sn_ddr_read_tracker_ostd = self.sn_ddr_wdb_size // self.burst
        self.sn_ddr_write_tracker_ostd = self.sn_ddr_wdb_size // self.burst
        self.sn_l2m_read_tracker_ostd = self.sn_l2m_wdb_size // self.burst
        self.sn_l2m_write_tracker_ostd = self.sn_l2m_wdb_size // self.burst
        self.channel_names = []
        for key in self.CHANNEL_SPEC:
            for idx in range(self.CHANNEL_SPEC[key]):
                self.channel_names.append(f"{key}_{idx}")

    def update_latency(self):
        self.ddr_R_latency = self.ddr_R_latency_original * self.network_frequency
        self.ddr_R_latency_var = self.ddr_R_latency_var_original * self.network_frequency
        self.ddr_W_latency = self.ddr_W_latency_original * self.network_frequency
        self.l2m_R_latency = self.l2m_R_latency_original * self.network_frequency
        self.l2m_W_latency = self.l2m_W_latency_original * self.network_frequency

    def topology_select(self, topo_type="default"):
        if topo_type == "default":
            self.ddr_send_positions = [self.cols * 2 * (x // self.cols) + self.cols + x % self.cols for x in range(self.num_ips)]
            self.sdma_send_positions = [self.cols * 2 * (x // self.cols) + self.cols + x % self.cols for x in range(self.num_ips)]
            self.l2m_send_positions = [self.cols * 2 * (x // self.cols) + self.cols + x % self.cols for x in range(self.num_ips)]
            self.gdma_send_positions = [self.cols * 2 * (x // self.cols) + self.cols + x % self.cols for x in range(self.num_ips)]

        elif topo_type == "8x8":
            self.num_nodes = 128
            self.cols = 8
            self.num_ips = 32
            self.rows = self.num_nodes // self.cols
            self.ddr_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [])
            self.sdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [])
            self.l2m_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [])
            self.gdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [])

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
        elif topo_type == "3x3":
            self.num_nodes = 18
            self.cols = 3
            self.num_ips = 4
            self.rows = self.num_nodes // self.cols
            self.ddr_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [])
            self.sdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [])
            self.l2m_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [])
            self.gdma_send_positions = self.generate_ip_positions([i for i in range(self.rows) if i % 2 == 0], [])
            # self.ddr_real_positions = [3, 5, 9, 10, 11, 15, 17]
            # self.l2m_real_positions = [4, 16]
        else:
            raise ValueError("Error topology type: ", topo_type)

    def generate_ip_positions(self, zero_rows=None, zero_cols=None):
        # 创建一个矩阵,初始值为1
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

    def distance(self, p1, p2):
        # return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])

    def assign_nearest_spare(self, failed_gdma, spare_cores):
        """
        为损坏核心分配备用核心,优先级为：
        1. 同列备用核心优先
        2. 同列中更靠近网络中心的优先
        3. 非同列时选择最靠近中心的备用核心
        """
        num_failed = len(failed_gdma)
        num_spare = len(spare_cores)

        if num_spare < num_failed:
            return []

        def decode(code):
            row = code // self.cols // 2
            col = code % self.cols
            return (col, self.cols - row)

        original_spare_cores = spare_cores.copy()
        failed_gdma = [decode(code) for code in failed_gdma]
        spare_cores = [decode(code) for code in spare_cores]

        # 计算每个备用核心的中心性分数（曼哈顿距离到中心点）
        network_center = (1.5, 2)  # 5x4 Mesh的中心坐标近似
        center_scores = {spare: abs(spare[0] - network_center[0]) + abs(spare[1] - network_center[1]) for spare in spare_cores}

        # 构造优先级矩阵
        cost_matrix = np.zeros((num_failed, num_spare))
        for i, gdma in enumerate(failed_gdma):
            for j, spare in enumerate(spare_cores):
                cost_matrix[i][j] = center_scores[spare] + self.distance(gdma, spare) * 1000

        # 匈牙利算法寻找最小总成本分配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return [original_spare_cores[j] for _, j in sorted(zip(row_ind, col_ind))]

    def spare_core_change(self, spare_core_row, fail_core_num, failed_core_poses=None):
        # print(spare_core_row)
        if spare_core_row > self.rows:
            return

        self.spare_core_row = spare_core_row
        spare_core_row = (spare_core_row + 1) // 2
        self.spare_core = [i for i in range(self.cols * (self.rows - 1 - spare_core_row * 2), self.cols * (self.rows - spare_core_row * 2))]
        add_core = [i for i in range(self.cols * (self.rows - 1), self.cols * (self.rows))]
        if spare_core_row != 0:
            for i, j in zip(self.spare_core, add_core):
                self.ddr_send_positions.remove(i)
                self.ddr_send_positions.append(j)
                self.gdma_send_positions.remove(i)
                self.gdma_send_positions.append(j)

        self.fail_core_num = fail_core_num

        if failed_core_poses is None:
            self.fail_core_pos = np.random.choice(self.gdma_send_positions, fail_core_num, replace=False).tolist()
        else:
            self.fail_core_pos = [self.gdma_send_positions[i] for i in failed_core_poses]

        self.spare_core_pos = self.assign_nearest_spare(self.fail_core_pos, self.spare_core)

        if len(self.fail_core_pos) != len(self.spare_core_pos):
            raise ValueError("fail_core_pos and spare_core_pos must have the same length")

        pos_mapping = dict(zip(self.fail_core_pos, self.spare_core_pos))

        # 执行批量替换
        self.gdma_send_positions = [pos_mapping.get(pos, pos) for pos in self.gdma_send_positions]  # 如果在映射表中则替换,否则保持原位置
        print(f"spare core: row: {self.spare_core_row}, id: {self.spare_core}, fail core: {self.fail_core_pos}, used spare core: {self.spare_core_pos}")

    def finish_del(self):
        del self.ddr_send_positions, self.l2m_send_positions, self.gdma_send_positions, self.sdma_send_positions

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
        parser.add_argument("--num_l2m", type=int, default=default_config["num_l2m"], help="Number of L2Ms")
        parser.add_argument("--num_sdma", type=int, default=default_config["num_sdma"], help="Number of SDMAs")
        parser.add_argument("--num_gdma", type=int, default=default_config["num_gdma"], help="Number of GDMA")
        parser.add_argument("--flit_size", type=int, default=default_config["flit_size"], help="Flit size")
        parser.add_argument("--seats_per_link", type=int, default=default_config["seats_per_link"], help="Seats per link")
        parser.add_argument("--RB_IN_FIFO_DEPTH", type=int, default=default_config["RB_IN_FIFO_DEPTH"], help="Depth of IN FIFOs in Ring Bridge")
        parser.add_argument("--RB_OUT_FIFO_DEPTH", type=int, default=default_config["RB_OUT_FIFO_DEPTH"], help="Depth of OUT FIFOs in Ring Bridge")
        parser.add_argument("--IQ_OUT_FIFO_DEPTH", type=int, default=default_config["IQ_OUT_FIFO_DEPTH"], help="Depth of IQ FIFOs in inject queues")
        parser.add_argument("--EQ_IN_FIFO_DEPTH", type=int, default=default_config["EQ_IN_FIFO_DEPTH"], help="Depth of EQ FIFOs in inject queues")
        parser.add_argument("--IQ_CH_FIFO_DEPTH", type=int, default=default_config["IQ_CH_FIFO_DEPTH"], help="Length of IP inject queues")
        parser.add_argument("--EQ_CH_FIFO_DEPTH", type=int, default=default_config["EQ_CH_FIFO_DEPTH"], help="Length of IP eject queues")
        parser.add_argument("--ITag_Trigger_Th_H", type=int, default=default_config["ITag_Trigger_Th_H"], help="Horizontal ring I-Tag trigger threshold")
        parser.add_argument("--ITag_Trigger_Th_V", type=int, default=default_config["ITag_Trigger_Th_V"], help="Vertical ring I-Tag trigger threshold")
        parser.add_argument("--ft_count", type=int, default=default_config["ft_count"], help="FT count")
        parser.add_argument("--ft_len", type=int, default=default_config["ft_len"], help="FT length")
        parser.add_argument("--ITag_Max_Num_H", type=int, default=default_config["ITag_Max_Num_H"], help="Maximum number of I-Tag reservations for horizontal ring XY nodes")
        parser.add_argument("--ITag_Max_Num_V", type=int, default=default_config["ITag_Max_Num_V"], help="Maximum number of I-Tag reservations for vertical ring XY nodes")
        parser.add_argument("--reservation_num", type=int, default=default_config["reservation_num"], help="Reservation number")
        parser.add_argument("--ddr_bandwidth_limit", type=int, default=default_config["ddr_bandwidth_limit"], help="DDR Bandwidth limit.")
        parser.add_argument("--ddr_R_latency", type=int, default=default_config["ddr_R_latency"], help="DDR latency")
        parser.add_argument("--ddr_R_latency_var", type=int, default=default_config["ddr_R_latency_var"], help="DDR latency")
        parser.add_argument("--ddr_W_latency", type=int, default=default_config["ddr_W_latency"], help="DDR latency")
        parser.add_argument("--l2m_R_latency", type=int, default=default_config["l2m_R_latency"], help="DDR latency")
        parser.add_argument("--l2m_W_latency", type=int, default=default_config["l2m_W_latency"], help="DDR latency")
        parser.add_argument("--sn_tracker_release_latency", type=int, default=default_config["sn_tracker_release_latency"], help="SN tracker release latency")

        parser.add_argument("--burst", type=int, default=default_config["burst"], help="Burst length")
        parser.add_argument("--network_frequency", type=float, default=default_config["network_frequency"], help="Network frequency")
        parser.add_argument("--rn_rdb_size", type=int, default=default_config["rn_rdb_size"], help="RN read buffer size")
        parser.add_argument("--rn_wdb_size", type=int, default=default_config["rn_wdb_size"], help="RN write buffer size")
        parser.add_argument("--sn_ddr_wdb_size", type=int, default=default_config["sn_ddr_wdb_size"], help="SN write buffer size")
        parser.add_argument("--sn_l2m_wdb_size", type=int, default=default_config["sn_l2m_wdb_size"], help="SN write buffer size")
        parser.add_argument("-tt", "--topo_type", type=str, default="", help="Choose topology type id from [4x9, 4x5, 5x4, 9x4, 3x3]")
        parser.add_argument("--TL_Etag_T1_UE_MAX", type=int, default=default_config["TL_Etag_T1_UE_MAX"], help="Horizontal cross point towards left T1 ETag FIFO Entry number")
        parser.add_argument("--TL_Etag_T2_UE_MAX", type=int, default=default_config["TL_Etag_T2_UE_MAX"], help="Horizontal cross point towards left T2 ETag FIFO Entry number")
        parser.add_argument("--TR_Etag_T2_UE_MAX", type=int, default=default_config["TR_Etag_T2_UE_MAX"], help="Horizontal cross point towards right T2 ETag FIFO Entry number")
        parser.add_argument("--TU_Etag_T1_UE_MAX", type=int, default=default_config["TU_Etag_T1_UE_MAX"], help="Vertival cross point towards up T1 ETag FIFO Entry number")
        parser.add_argument("--TU_Etag_T2_UE_MAX", type=int, default=default_config["TU_Etag_T2_UE_MAX"], help="Vertival cross point towards up T2 ETag FIFO Entry number")
        parser.add_argument("--TD_Etag_T2_UE_MAX", type=int, default=default_config["TD_Etag_T2_UE_MAX"], help="Vertival cross point towards down T2 ETag FIFO Entry number")
        parser.add_argument("--Both_side_ETag_upgrade", type=int, default=default_config["Both_side_ETag_upgrade"], help="ETag upgrade method")

        return parser.parse_args()
