# config.py

import json
import os
import numpy as np
import argparse
from scipy.optimize import linear_sum_assignment
from collections import deque, defaultdict
from typing import Callable, Iterable, Dict, Any
import copy
from pathlib import Path


class CrossRingConfig:
    def __init__(self, default_config=None):
        if default_config is None:
            # __file__ 是当前脚本所在路径，.parent.parent 假设脚本在项目根的子目录中
            project_root = Path(__file__).resolve().parent.parent
            default_config = project_root / "config" / "config2.json"

        args = self.parse_args(default_config)
        self.TOPO_TYPE = args.TOPO_TYPE
        self.NUM_NODE = args.NUM_NODE
        self.NUM_COL = args.NUM_COL
        self.NUM_IP = args.NUM_IP
        self.NUM_RN = args.NUM_RN
        self.NUM_SN = args.NUM_SN
        self.NUM_ROW = self.NUM_NODE // self.NUM_COL
        self.NUM_SDMA = args.NUM_SDMA
        self.NUM_GDMA = args.NUM_GDMA
        self.NUM_CDMA = args.NUM_CDMA
        self.NUM_DDR = args.NUM_DDR
        self.NUM_L2M = args.NUM_L2M
        self.FLIT_SIZE = args.FLIT_SIZE
        self.SPARE_CORE_ROW = -1
        self.FAIL_CORE_POS = []
        self.SLICE_PER_LINK = args.SLICE_PER_LINK
        self.RB_IN_FIFO_DEPTH = args.RB_IN_FIFO_DEPTH
        self.RB_OUT_FIFO_DEPTH = args.RB_OUT_FIFO_DEPTH
        self.IQ_OUT_FIFO_DEPTH_HORIZONTAL = args.IQ_OUT_FIFO_DEPTH_HORIZONTAL
        self.IQ_OUT_FIFO_DEPTH_VERTICAL = args.IQ_OUT_FIFO_DEPTH_VERTICAL
        self.IQ_OUT_FIFO_DEPTH_EQ = args.IQ_OUT_FIFO_DEPTH_EQ
        # 保留 IQ_OUT_FIFO_DEPTH 用于向后兼容
        self.IQ_OUT_FIFO_DEPTH = args.IQ_OUT_FIFO_DEPTH_HORIZONTAL
        self.EQ_IN_FIFO_DEPTH = args.EQ_IN_FIFO_DEPTH
        self.IQ_CH_FIFO_DEPTH = args.IQ_CH_FIFO_DEPTH
        self.EQ_CH_FIFO_DEPTH = args.EQ_CH_FIFO_DEPTH
        self.ITag_TRIGGER_Th_H = args.ITag_TRIGGER_Th_H
        self.ITag_TRIGGER_Th_V = args.ITag_TRIGGER_Th_V
        self.ITag_MAX_NUM_H = args.ITag_MAX_Num_H
        self.ITag_MAX_NUM_V = args.ITag_MAX_Num_V
        self.RB_ONLY_TAG_NUM_HORIZONTAL = args.RB_ONLY_TAG_NUM_HORIZONTAL
        self.RB_ONLY_TAG_NUM_VERTICAL = args.RB_ONLY_TAG_NUM_VERTICAL
        # 验证标签数量参数在有效范围内
        if self.RB_ONLY_TAG_NUM_HORIZONTAL < 0:
            raise ValueError(f"RB_ONLY_TAG_NUM_HORIZONTAL must be non-negative, got {self.RB_ONLY_TAG_NUM_HORIZONTAL}")
        if self.RB_ONLY_TAG_NUM_VERTICAL < 0:
            raise ValueError(f"RB_ONLY_TAG_NUM_VERTICAL must be non-negative, got {self.RB_ONLY_TAG_NUM_VERTICAL}")
        # self.reservation_num = args.reservation_num
        self.BURST = args.BURST
        self.NETWORK_FREQUENCY = args.NETWORK_FREQUENCY
        self.RN_R_TRACKER_OSTD = args.RN_R_TRACKER_OSTD
        self.RN_W_TRACKER_OSTD = args.RN_W_TRACKER_OSTD
        self.RN_RDB_SIZE = args.RN_RDB_SIZE
        self.RN_WDB_SIZE = args.RN_WDB_SIZE
        self.SN_DDR_R_TRACKER_OSTD = args.SN_DDR_R_TRACKER_OSTD
        self.SN_DDR_W_TRACKER_OSTD = args.SN_DDR_W_TRACKER_OSTD
        self.SN_L2M_R_TRACKER_OSTD = args.SN_L2M_R_TRACKER_OSTD
        self.SN_L2M_W_TRACKER_OSTD = args.SN_L2M_W_TRACKER_OSTD
        self.SN_DDR_RDB_SIZE = args.SN_DDR_RDB_SIZE
        self.SN_DDR_WDB_SIZE = args.SN_DDR_WDB_SIZE
        self.SN_L2M_RDB_SIZE = args.SN_L2M_RDB_SIZE
        self.SN_L2M_WDB_SIZE = args.SN_L2M_WDB_SIZE
        # 新的双向带宽限制参数
        self.GDMA_TX_BW_LIMIT = args.GDMA_TX_BW_LIMIT
        self.GDMA_RX_BW_LIMIT = args.GDMA_RX_BW_LIMIT
        self.SDMA_TX_BW_LIMIT = args.SDMA_TX_BW_LIMIT
        self.SDMA_RX_BW_LIMIT = args.SDMA_RX_BW_LIMIT
        self.CDMA_TX_BW_LIMIT = args.CDMA_TX_BW_LIMIT
        self.CDMA_RX_BW_LIMIT = args.CDMA_RX_BW_LIMIT
        self.DDR_TX_BW_LIMIT = args.DDR_TX_BW_LIMIT
        self.DDR_RX_BW_LIMIT = args.DDR_RX_BW_LIMIT
        self.L2M_TX_BW_LIMIT = args.L2M_TX_BW_LIMIT
        self.L2M_RX_BW_LIMIT = args.L2M_RX_BW_LIMIT
        self.DDR_R_LATENCY_original = args.DDR_R_LATENCY
        self.DDR_R_LATENCY_VAR_original = args.DDR_R_LATENCY_VAR
        self.DDR_W_LATENCY_original = args.DDR_W_LATENCY
        self.L2M_R_LATENCY_original = args.L2M_R_LATENCY
        self.L2M_W_LATENCY_original = args.L2M_W_LATENCY
        self.SN_TRACKER_RELEASE_LATENCY_original = args.SN_TRACKER_RELEASE_LATENCY
        self.TL_Etag_T1_UE_MAX = args.TL_Etag_T1_UE_MAX
        self.TL_Etag_T2_UE_MAX = args.TL_Etag_T2_UE_MAX
        self.TR_Etag_T2_UE_MAX = args.TR_Etag_T2_UE_MAX
        self.TU_Etag_T1_UE_MAX = args.TU_Etag_T1_UE_MAX
        self.TU_Etag_T2_UE_MAX = args.TU_Etag_T2_UE_MAX
        self.TD_Etag_T2_UE_MAX = args.TD_Etag_T2_UE_MAX
        self.ETag_BOTHSIDE_UPGRADE = args.ETag_BOTHSIDE_UPGRADE
        self.GDMA_RW_GAP = args.GDMA_RW_GAP
        self.SDMA_RW_GAP = args.SDMA_RW_GAP
        self.ENABLE_CROSSPOINT_CONFLICT_CHECK = args.ENABLE_CROSSPOINT_CONFLICT_CHECK
        self.CROSSRING_VERSION = args.CROSSRING_VERSION
        self.ENABLE_IN_ORDER_EJECTION = args.ENABLE_IN_ORDER_EJECTION
        self.CHANNEL_SPEC = {
            "gdma": 2,  # → RN 侧
            "sdma": 2,  # → RN 侧
            "cdma": 2,  # → RN 侧
            "ddr": 2,  # → SN 侧
            "l2m": 2,  # → SN 侧
        }
        self.CH_NAME_LIST = []
        for key in self.CHANNEL_SPEC:
            for idx in range(self.CHANNEL_SPEC[key]):
                self.CH_NAME_LIST.append(f"{key}_{idx}")
        assert (
            self.TL_Etag_T2_UE_MAX < self.TL_Etag_T1_UE_MAX < self.RB_IN_FIFO_DEPTH
            and self.TL_Etag_T2_UE_MAX < self.RB_IN_FIFO_DEPTH - 2
            and self.TR_Etag_T2_UE_MAX < self.RB_IN_FIFO_DEPTH - 1
            and self.TU_Etag_T2_UE_MAX < self.TU_Etag_T1_UE_MAX < self.EQ_IN_FIFO_DEPTH
            and self.TU_Etag_T2_UE_MAX < self.EQ_IN_FIFO_DEPTH - 2
            and self.TD_Etag_T2_UE_MAX < self.EQ_IN_FIFO_DEPTH - 1
        ), "ETag parameter conditions are not met."

        self.update_config(topo_type="default")

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

    def update_config(self, topo_type="default"):
        self.update_latency()
        self.topology_select(topo_type)
        self.SN_TRACKER_RELEASE_LATENCY = self.SN_TRACKER_RELEASE_LATENCY_original * self.NETWORK_FREQUENCY
        self.CH_NAME_LIST = []
        for key in self.CHANNEL_SPEC:
            for idx in range(self.CHANNEL_SPEC[key]):
                self.CH_NAME_LIST.append(f"{key}_{idx}")

    def update_latency(self):
        self.DDR_R_LATENCY = self.DDR_R_LATENCY_original * self.NETWORK_FREQUENCY
        self.DDR_R_LATENCY_VAR = self.DDR_R_LATENCY_VAR_original * self.NETWORK_FREQUENCY
        self.DDR_W_LATENCY = self.DDR_W_LATENCY_original * self.NETWORK_FREQUENCY
        self.L2M_R_LATENCY = self.L2M_R_LATENCY_original * self.NETWORK_FREQUENCY
        self.L2M_W_LATENCY = self.L2M_W_LATENCY_original * self.NETWORK_FREQUENCY

    def topology_select(self, topo_type="default"):
        if topo_type == "default":
            self.DDR_SEND_POSITION_LIST = [self.NUM_COL * 2 * (x // self.NUM_COL) + self.NUM_COL + x % self.NUM_COL for x in range(self.NUM_IP)]
            self.L2M_SEND_POSITION_LIST = [self.NUM_COL * 2 * (x // self.NUM_COL) + self.NUM_COL + x % self.NUM_COL for x in range(self.NUM_IP)]
            self.SDMA_SEND_POSITION_LIST = [self.NUM_COL * 2 * (x // self.NUM_COL) + self.NUM_COL + x % self.NUM_COL for x in range(self.NUM_IP)]
            self.GDMA_SEND_POSITION_LIST = [self.NUM_COL * 2 * (x // self.NUM_COL) + self.NUM_COL + x % self.NUM_COL for x in range(self.NUM_IP)]
            self.CDMA_SEND_POSITION_LIST = [self.NUM_COL * 2 * (x // self.NUM_COL) + self.NUM_COL + x % self.NUM_COL for x in range(self.NUM_IP)]

        elif topo_type == "5x2":
            self.NUM_NODE = 20
            self.NUM_COL = 2
            self.NUM_IP = 16
            self.NUM_ROW = self.NUM_NODE // self.NUM_COL
            self.DDR_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.L2M_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.SDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.GDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.CDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
        elif topo_type == "4x2":
            self.NUM_ROW = self.NUM_NODE // self.NUM_COL
            self.DDR_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.L2M_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.SDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.GDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.CDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
        elif topo_type == "5x4":
            self.NUM_NODE = 40
            self.NUM_COL = 4
            self.NUM_IP = 32
            self.NUM_ROW = self.NUM_NODE // self.NUM_COL
            self.DDR_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0] + [9], [])
            self.L2M_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0] + [9], [])
            self.SDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0] + [9], [])
            self.GDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0] + [9], [])
            self.CDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW - 1)], [])
        elif topo_type == "6x5":
            self.NUM_NODE = 60
            self.NUM_COL = 5
            self.NUM_IP = 8
            self.NUM_ROW = self.NUM_NODE // self.NUM_COL
            self.DDR_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.L2M_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.SDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.GDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.CDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW - 1)], [])

        elif topo_type == "4x5":
            self.NUM_NODE = 40
            self.NUM_COL = 5
            self.NUM_IP = 32
            self.NUM_ROW = self.NUM_NODE // self.NUM_COL
            self.DDR_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.L2M_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.SDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.GDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.CDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])

        else:
            self.DDR_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.L2M_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.SDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.GDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])
            self.CDMA_SEND_POSITION_LIST = self.generate_ip_positions([i for i in range(self.NUM_ROW) if i % 2 == 0], [])

    def generate_ip_positions(self, zero_rows=None, zero_cols=None):
        # 创建一个矩阵,初始值为1
        matrix = [[1 for _ in range(self.NUM_COL)] for _ in range(self.NUM_ROW)]

        # 将指定的行设置为0
        if zero_rows:
            for row in zero_rows:
                if 0 <= row < self.NUM_ROW:
                    for col in range(self.NUM_COL):
                        matrix[row][col] = 0

        # 将指定的列设置为0
        if zero_cols:
            for col in zero_cols:
                if 0 <= col < self.NUM_COL:
                    for row in range(self.NUM_ROW):
                        matrix[row][col] = 0

        # 收集所有元素为1的编号
        indices = []
        for r in range(self.NUM_ROW):
            for c in range(self.NUM_COL):
                if matrix[r][c] == 1:
                    index = r * self.NUM_COL + c
                    indices.append(index)
        # assert len(indices) == self.NUM_IP, f"Expected {self.NUM_IP} indices, but got {len(indices)}."
        return indices

    def distance(self, p1, p2):
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
            row = code // self.NUM_COL // 2
            col = code % self.NUM_COL
            return (col, self.NUM_COL - row)

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
        if spare_core_row > self.NUM_ROW:
            return

        self.SPARE_CORE_ROW = spare_core_row
        spare_core_row = (spare_core_row + 1) // 2
        self.spare_core = [i for i in range(self.NUM_COL * (self.NUM_ROW - 1 - spare_core_row * 2), self.NUM_COL * (self.NUM_ROW - spare_core_row * 2))]
        add_core = [i for i in range(self.NUM_COL * (self.NUM_ROW - 1), self.NUM_COL * (self.NUM_ROW))]
        if spare_core_row != 0:
            for i, j in zip(self.spare_core, add_core):
                self.DDR_SEND_POSITION_LIST.remove(i)
                self.DDR_SEND_POSITION_LIST.append(j)
                self.GDMA_SEND_POSITION_LIST.remove(i)
                self.GDMA_SEND_POSITION_LIST.append(j)

        self.fail_core_num = fail_core_num

        if failed_core_poses is None:
            self.FAIL_CORE_POS = np.random.choice(self.GDMA_SEND_POSITION_LIST, fail_core_num, replace=False).tolist()
        else:
            self.FAIL_CORE_POS = [self.GDMA_SEND_POSITION_LIST[i] for i in failed_core_poses]

        self.spare_core_pos = self.assign_nearest_spare(self.FAIL_CORE_POS, self.spare_core)

        if len(self.FAIL_CORE_POS) != len(self.spare_core_pos):
            raise ValueError("fail_core_pos and spare_core_pos must have the same length")

        pos_mapping = dict(zip(self.FAIL_CORE_POS, self.spare_core_pos))

        # 执行批量替换
        self.GDMA_SEND_POSITION_LIST = [pos_mapping.get(pos, pos) for pos in self.GDMA_SEND_POSITION_LIST]  # 如果在映射表中则替换,否则保持原位置
        print(f"spare core: row: {self.SPARE_CORE_ROW}, id: {self.spare_core}, fail core: {self.FAIL_CORE_POS}, used spare core: {self.spare_core_pos}")

    def finish_del(self):
        del self.DDR_SEND_POSITION_LIST, self.L2M_SEND_POSITION_LIST, self.GDMA_SEND_POSITION_LIST, self.SDMA_SEND_POSITION_LIST

    def parse_args(self, default_config):
        # Handle relative paths by resolving from the caller's directory
        if not os.path.isabs(default_config):
            # Get the caller's frame to determine the correct base directory
            import inspect

            caller_frame = inspect.currentframe().f_back.f_back
            caller_file = caller_frame.f_globals["__file__"]
            caller_dir = os.path.dirname(os.path.abspath(caller_file))
            default_config = os.path.join(caller_dir, default_config)

        if os.path.exists(default_config):
            with open(default_config, "r") as f:
                default_config = json.load(f)
        else:
            raise FileNotFoundError(f"{default_config} not found.")

        parser = argparse.ArgumentParser(description="Process simulation configuration parameters.")

        # 将 JSON 配置作为默认值
        parser.add_argument("--NUM_NODE", type=int, default=default_config["NUM_NODE"], help="Number of nodes")
        parser.add_argument("--NUM_COL", type=int, default=default_config["NUM_COL"], help="Number of columns")
        parser.add_argument("--NUM_IP", type=int, default=default_config["NUM_IP"], help="Number of IP")
        parser.add_argument("--NUM_RN", type=int, default=default_config["NUM_RN"], help="Number of RN")
        parser.add_argument("--NUM_SN", type=int, default=default_config["NUM_SN"], help="Number of SN")
        parser.add_argument("--NUM_DDR", type=int, default=default_config["NUM_DDR"], help="Number of DDRs")
        parser.add_argument("--NUM_L2M", type=int, default=default_config["NUM_L2M"], help="Number of L2Ms")
        parser.add_argument("--NUM_SDMA", type=int, default=default_config["NUM_SDMA"], help="Number of SDMAs")
        parser.add_argument("--NUM_GDMA", type=int, default=default_config["NUM_GDMA"], help="Number of GDMA")
        parser.add_argument("--NUM_CDMA", type=int, default=default_config["NUM_CDMA"], help="Number of GDMA")
        parser.add_argument("--FLIT_SIZE", type=int, default=default_config["FLIT_SIZE"], help="Flit size")
        parser.add_argument("--SLICE_PER_LINK", type=int, default=default_config["SLICE_PER_LINK"], help="Slice num per link, (num -2) equals to RTL slice num")
        parser.add_argument("--RB_IN_FIFO_DEPTH", type=int, default=default_config["RB_IN_FIFO_DEPTH"], help="Depth of IN FIFOs in Ring Bridge")
        parser.add_argument("--RB_OUT_FIFO_DEPTH", type=int, default=default_config["RB_OUT_FIFO_DEPTH"], help="Depth of OUT FIFOs in Ring Bridge")
        # 处理向后兼容性：如果只有 IQ_OUT_FIFO_DEPTH，使用它来初始化三个新参数
        if "IQ_OUT_FIFO_DEPTH" in default_config and "IQ_OUT_FIFO_DEPTH_HORIZONTAL" not in default_config:
            default_iq_depth = default_config["IQ_OUT_FIFO_DEPTH"]
            default_config["IQ_OUT_FIFO_DEPTH_HORIZONTAL"] = default_iq_depth
            default_config["IQ_OUT_FIFO_DEPTH_VERTICAL"] = default_iq_depth
            default_config["IQ_OUT_FIFO_DEPTH_EQ"] = default_iq_depth
        
        parser.add_argument("--IQ_OUT_FIFO_DEPTH_HORIZONTAL", type=int, default=default_config.get("IQ_OUT_FIFO_DEPTH_HORIZONTAL", 6), help="Depth of IQ FIFOs for TR/TL in inject queues")
        parser.add_argument("--IQ_OUT_FIFO_DEPTH_VERTICAL", type=int, default=default_config.get("IQ_OUT_FIFO_DEPTH_VERTICAL", 6), help="Depth of IQ FIFOs for TU/TD in inject queues")
        parser.add_argument("--IQ_OUT_FIFO_DEPTH_EQ", type=int, default=default_config.get("IQ_OUT_FIFO_DEPTH_EQ", 6), help="Depth of IQ FIFOs for EQ in inject queues")
        parser.add_argument("--EQ_IN_FIFO_DEPTH", type=int, default=default_config["EQ_IN_FIFO_DEPTH"], help="Depth of EQ FIFOs in inject queues")
        parser.add_argument("--IQ_CH_FIFO_DEPTH", type=int, default=default_config["IQ_CH_FIFO_DEPTH"], help="Length of IP inject queues")
        parser.add_argument("--EQ_CH_FIFO_DEPTH", type=int, default=default_config["EQ_CH_FIFO_DEPTH"], help="Length of IP eject queues")
        parser.add_argument("--ITag_TRIGGER_Th_H", type=int, default=default_config["ITag_TRIGGER_Th_H"], help="Horizontal ring I-Tag trigger threshold")
        parser.add_argument("--ITag_TRIGGER_Th_V", type=int, default=default_config["ITag_TRIGGER_Th_V"], help="Vertical ring I-Tag trigger threshold")
        parser.add_argument("--ITag_MAX_Num_H", type=int, default=default_config["ITag_MAX_Num_H"], help="Maximum number of I-Tag reservations for horizontal ring XY nodes")
        parser.add_argument("--ITag_MAX_Num_V", type=int, default=default_config["ITag_MAX_Num_V"], help="Maximum number of I-Tag reservations for vertical ring XY nodes")
        parser.add_argument("--RB_ONLY_TAG_NUM_HORIZONTAL", type=int, default=default_config["RB_ONLY_TAG_NUM_HORIZONTAL"], help="Number of RB only tags per horizontal ring")
        parser.add_argument("--RB_ONLY_TAG_NUM_VERTICAL", type=int, default=default_config["RB_ONLY_TAG_NUM_VERTICAL"], help="Number of RB only tags per vertical ring")
        # 新的双向带宽限制参数
        parser.add_argument("--GDMA_TX_BW_LIMIT", type=int, default=default_config["GDMA_TX_BW_LIMIT"], help="GDMA TX Bandwidth limit.")
        parser.add_argument("--GDMA_RX_BW_LIMIT", type=int, default=default_config["GDMA_RX_BW_LIMIT"], help="GDMA RX Bandwidth limit.")
        parser.add_argument("--SDMA_TX_BW_LIMIT", type=int, default=default_config["SDMA_TX_BW_LIMIT"], help="SDMA TX Bandwidth limit.")
        parser.add_argument("--SDMA_RX_BW_LIMIT", type=int, default=default_config["SDMA_RX_BW_LIMIT"], help="SDMA RX Bandwidth limit.")
        parser.add_argument("--CDMA_TX_BW_LIMIT", type=int, default=default_config["CDMA_TX_BW_LIMIT"], help="CDMA TX Bandwidth limit.")
        parser.add_argument("--CDMA_RX_BW_LIMIT", type=int, default=default_config["CDMA_RX_BW_LIMIT"], help="CDMA RX Bandwidth limit.")
        parser.add_argument("--DDR_TX_BW_LIMIT", type=int, default=default_config["DDR_TX_BW_LIMIT"], help="DDR TX Bandwidth limit.")
        parser.add_argument("--DDR_RX_BW_LIMIT", type=int, default=default_config["DDR_RX_BW_LIMIT"], help="DDR RX Bandwidth limit.")
        parser.add_argument("--L2M_TX_BW_LIMIT", type=int, default=default_config["L2M_TX_BW_LIMIT"], help="L2M TX Bandwidth limit.")
        parser.add_argument("--L2M_RX_BW_LIMIT", type=int, default=default_config["L2M_RX_BW_LIMIT"], help="L2M RX Bandwidth limit.")
        parser.add_argument("--DDR_R_LATENCY", type=int, default=default_config["DDR_R_LATENCY"], help="DDR latency")
        parser.add_argument("--DDR_R_LATENCY_VAR", type=int, default=default_config["DDR_R_LATENCY_VAR"], help="DDR latency")
        parser.add_argument("--DDR_W_LATENCY", type=int, default=default_config["DDR_W_LATENCY"], help="DDR latency")
        parser.add_argument("--L2M_R_LATENCY", type=int, default=default_config["L2M_R_LATENCY"], help="DDR latency")
        parser.add_argument("--L2M_W_LATENCY", type=int, default=default_config["L2M_W_LATENCY"], help="DDR latency")
        parser.add_argument("--SN_TRACKER_RELEASE_LATENCY", type=int, default=default_config["SN_TRACKER_RELEASE_LATENCY"], help="SN tracker release latency")

        parser.add_argument("--BURST", type=int, default=default_config["BURST"], help="Burst length")
        parser.add_argument("--NETWORK_FREQUENCY", type=float, default=default_config["NETWORK_FREQUENCY"], help="Network frequency")
        parser.add_argument("--RN_R_TRACKER_OSTD", type=int, default=default_config["RN_R_TRACKER_OSTD"], help="RN read tracker outstanding")
        parser.add_argument("--RN_W_TRACKER_OSTD", type=int, default=default_config["RN_W_TRACKER_OSTD"], help="RN write tracker outstanding")
        parser.add_argument("--RN_RDB_SIZE", type=int, default=default_config["RN_RDB_SIZE"], help="RN read buffer size")
        parser.add_argument("--RN_WDB_SIZE", type=int, default=default_config["RN_WDB_SIZE"], help="RN write buffer size")
        parser.add_argument("--SN_DDR_R_TRACKER_OSTD", type=int, default=default_config["SN_DDR_R_TRACKER_OSTD"], help="SN ddr read tracker outstanding")
        parser.add_argument("--SN_DDR_W_TRACKER_OSTD", type=int, default=default_config["SN_DDR_W_TRACKER_OSTD"], help="SN ddr write tracker outstanding")
        parser.add_argument("--SN_L2M_R_TRACKER_OSTD", type=int, default=default_config["SN_L2M_R_TRACKER_OSTD"], help="SN l2m read tracker outstanding")
        parser.add_argument("--SN_L2M_W_TRACKER_OSTD", type=int, default=default_config["SN_L2M_W_TRACKER_OSTD"], help="SN l2m write tracker outstanding")
        parser.add_argument("--SN_DDR_RDB_SIZE", type=int, default=default_config["SN_DDR_RDB_SIZE"], help="SN ddr read buffer size")
        parser.add_argument("--SN_DDR_WDB_SIZE", type=int, default=default_config["SN_DDR_WDB_SIZE"], help="SN ddr write buffer size")
        parser.add_argument("--SN_L2M_RDB_SIZE", type=int, default=default_config["SN_L2M_RDB_SIZE"], help="SN l2m read buffer size")
        parser.add_argument("--SN_L2M_WDB_SIZE", type=int, default=default_config["SN_L2M_WDB_SIZE"], help="SN l2m write buffer size")
        parser.add_argument("-tt", "--TOPO_TYPE", type=str, default="", help="Choose topology type id from [4x9, 4x5, 5x4, 9x4, 3x3]")
        parser.add_argument("--TL_Etag_T1_UE_MAX", type=int, default=default_config["TL_Etag_T1_UE_MAX"], help="Horizontal cross point towards left T1 ETag FIFO Entry number")
        parser.add_argument("--TL_Etag_T2_UE_MAX", type=int, default=default_config["TL_Etag_T2_UE_MAX"], help="Horizontal cross point towards left T2 ETag FIFO Entry number")
        parser.add_argument("--TR_Etag_T2_UE_MAX", type=int, default=default_config["TR_Etag_T2_UE_MAX"], help="Horizontal cross point towards right T2 ETag FIFO Entry number")
        parser.add_argument("--TU_Etag_T1_UE_MAX", type=int, default=default_config["TU_Etag_T1_UE_MAX"], help="Vertical cross point towards up T1 ETag FIFO Entry number")
        parser.add_argument("--TU_Etag_T2_UE_MAX", type=int, default=default_config["TU_Etag_T2_UE_MAX"], help="Vertical cross point towards up T2 ETag FIFO Entry number")
        parser.add_argument("--TD_Etag_T2_UE_MAX", type=int, default=default_config["TD_Etag_T2_UE_MAX"], help="Vertical cross point towards down T2 ETag FIFO Entry number")
        parser.add_argument("--ETag_BOTHSIDE_UPGRADE", type=int, default=default_config["ETag_BOTHSIDE_UPGRADE"], help="ETag upgrade method")
        parser.add_argument("--IP_L2H_FIFO_DEPTH", type=int, default=default_config["IP_L2H_FIFO_DEPTH"], help="IP frequency change l2h fifo depth")
        parser.add_argument("--IP_H2L_H_FIFO_DEPTH", type=int, default=default_config["IP_H2L_H_FIFO_DEPTH"], help="IP frequency change h2l high frequency fifo depth")
        parser.add_argument("--IP_H2L_L_FIFO_DEPTH", type=int, default=default_config["IP_H2L_L_FIFO_DEPTH"], help="IP frequency change h2l low frequency fifo depth")
        parser.add_argument("--GDMA_RW_GAP", type=int, default=default_config["GDMA_RW_GAP"], help="GDMA read and write cmd num gap")
        parser.add_argument("--SDMA_RW_GAP", type=int, default=default_config["SDMA_RW_GAP"], help="SDMA read and write cmd num gap")
        parser.add_argument("--ENABLE_CROSSPOINT_CONFLICT_CHECK", type=bool, default=default_config["ENABLE_CROSSPOINT_CONFLICT_CHECK"], help="Enable crosspoint conflict checking for inject queue")
        parser.add_argument("--CROSSRING_VERSION", type=str, default=default_config["CROSSRING_VERSION"], help="CrossRing version (V1 or V2)")
        parser.add_argument("--ENABLE_IN_ORDER_EJECTION", type=bool, default=default_config["ENABLE_IN_ORDER_EJECTION"], help="Enable in-order ejection for src-dest pairs")

        return parser.parse_args()
