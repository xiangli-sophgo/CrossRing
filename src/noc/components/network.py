"""
Network class for NoC simulation.
Contains the core network implementation with routing and flow control mechanisms.
Enhanced with integrated route table support for flexible routing decisions.
"""

from __future__ import annotations
import numpy as np
from collections import deque, defaultdict
from typing import Optional, Dict, List, Any, Tuple
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


class LinkSlot:
    """
    链路Slot对象 - 封装slot_id和tag信息

    Slot是链路上的基本传输单元，每个seat持有一个Slot对象。
    Slot携带slot_id（全局唯一标识）和ITag信息。
    """

    def __init__(self, slot_id: int):
        """
        初始化LinkSlot

        Args:
            slot_id: 全局唯一的slot标识符
        """
        self.slot_id = slot_id

        # ITag信息 (注入预约机制)
        self.itag_reserved = False
        self.itag_reserver_id = None
        self.itag_direction = None

    def reserve_itag(self, reserver_id: int, direction: str) -> bool:
        """
        预约ITag

        Args:
            reserver_id: 预约者节点ID
            direction: 预约方向

        Returns:
            是否成功预约
        """
        if self.itag_reserved:
            return False

        self.itag_reserved = True
        self.itag_reserver_id = reserver_id
        self.itag_direction = direction
        return True

    def clear_itag(self) -> None:
        """清除ITag预约"""
        self.itag_reserved = False
        self.itag_reserver_id = None
        self.itag_direction = None

    def check_itag_match(self, reserver_id: int, direction: str) -> bool:
        """
        检查ITag是否匹配

        Args:
            reserver_id: 预约者节点ID
            direction: 预约方向

        Returns:
            是否匹配
        """
        return self.itag_reserved and self.itag_reserver_id == reserver_id and self.itag_direction == direction


class Network:
    def __init__(self, config: CrossRingConfig, adjacency_matrix, name="network"):
        self.config = config
        self.name = name

        # Pre-calculate frequently used position sets for performance
        self._all_ip_positions = None
        self._rn_positions = None
        self._sn_positions = None
        self._positions_cache_lock = None  # For thread safety if needed

        # 保序跟踪表: {(src, dest, direction): last_ejected_id}
        # 每个network独立维护自己的tracking_table，不需要区分packet_category
        self.order_tracking_table = defaultdict(int)

        self.current_cycle = []
        self.flits_num = []
        self.schedules = {"sdma": None}
        self.inject_num = 0
        self.eject_num = 0
        self.inject_queues = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.inject_queues_pre = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.eject_queues = {"TU": {}, "TD": {}}
        self.eject_queues_in_pre = {"TU": {}, "TD": {}}
        self.arrive_node_pre = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.IQ_channel_buffer = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m", "d2d_rn", "d2d_sn"))
        self.EQ_channel_buffer = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m", "d2d_rn", "d2d_sn"))
        self.IQ_channel_buffer_pre = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m", "d2d_rn", "d2d_sn"))
        self.EQ_channel_buffer_pre = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m", "d2d_rn", "d2d_sn"))
        # IQ仲裁输入FIFO (深度2 + pre缓冲，按IP类型分类)
        self.IQ_arbiter_input_fifo = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m", "d2d_rn", "d2d_sn"))
        self.IQ_arbiter_input_fifo_pre = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m", "d2d_rn", "d2d_sn"))
        # EQ仲裁输入FIFO (深度2 + pre缓冲，4个输入端口，每个端口一个FIFO)
        self.EQ_arbiter_input_fifo = {"TU": {}, "TD": {}, "IQ": {}, "RB": {}}
        self.EQ_arbiter_input_fifo_pre = {"TU": {}, "TD": {}, "IQ": {}, "RB": {}}
        self.links = {}
        self.cross_point = {"horizontal": defaultdict(lambda: defaultdict(list)), "vertical": defaultdict(lambda: defaultdict(list))}
        # Crosspoint conflict status: maintains pipeline queue [current_cycle, previous_cycle]
        self.crosspoint_conflict = {"horizontal": defaultdict(lambda: defaultdict(lambda: [False, False])), "vertical": defaultdict(lambda: defaultdict(lambda: [False, False]))}
        # 新的链路状态统计 - 记录各种ETag/ITag状态的计数
        self.links_flow_stat = {}
        # 每个周期的瞬时状态统计
        self.links_state_snapshots = []
        # ITag setup
        self.links_tag = {}

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
        self.circuits_flit_h = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.circuits_flit_v = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.gdma_recv = 0
        self.gdma_remainder = 0
        self.gdma_count = 512
        self.l2m_recv = 0
        self.l2m_remainder = 0
        self.sdma_send = []
        self.num_send = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.num_recv = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.per_send_throughput = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.per_recv_throughput = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.send_throughput = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.recv_throughput = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.last_select = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.throughput = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))

        # FIFO使用率统计
        self.fifo_depth_sum = {
            "IQ": {"CH_buffer": {}, "TR": {}, "TL": {}, "TU": {}, "TD": {}, "EQ": {}},
            "RB": {"TR": {}, "TL": {}, "TU": {}, "TD": {}, "EQ": {}},
            "EQ": {"TU": {}, "TD": {}, "CH_buffer": {}},
        }
        self.fifo_max_depth = {
            "IQ": {"CH_buffer": {}, "TR": {}, "TL": {}, "TU": {}, "TD": {}, "EQ": {}},
            "RB": {"TR": {}, "TL": {}, "TU": {}, "TD": {}, "EQ": {}},
            "EQ": {"TU": {}, "TD": {}, "CH_buffer": {}},
        }
        self.fifo_flit_count = {
            "IQ": {"CH_buffer": {}, "TR": {}, "TL": {}, "TU": {}, "TD": {}, "EQ": {}},
            "RB": {"TR": {}, "TL": {}, "TU": {}, "TD": {}, "EQ": {}},
            "EQ": {"TU": {}, "TD": {}, "CH_buffer": {}},
        }

        # ITag累计次数统计（每周期累加FIFO中带ITag的flit数）
        self.fifo_itag_cumulative_count = {
            "IQ": {"TR": {}, "TL": {}},  # IQ_OUT (横向注入)
            "RB": {"TU": {}, "TD": {}},  # RB_OUT (纵向转向)
        }

        # ETag入队统计（flit进入pre缓冲区时的ETag等级快照）
        # 存储累计次数（每次flit入队时+1）
        self.fifo_etag_entry_count = {
            "RB": {
                "TR": {},  # {node_pos: {"T0": count, "T1": count, "T2": count}}
                "TL": {},
            },
            "EQ": {
                "TU": {},
                "TD": {},
            },
        }

        # Slot ID全局计数器（用于为每个seat分配唯一ID）
        self.global_slot_id_counter = 0

        # ETag setup (这些数据结构由Network管理，CrossPoint共享引用)
        # T0 Slot ID轮询队列：每个方向独立的FIFO（改为存储slot_id而非(node, flit)）
        self.T0_Etag_Order_FIFO = {"TL": deque(), "TR": deque(), "TU": deque(), "TD": deque()}
        self.RB_UE_Counters = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}  # 所有CrossPoint共享
        self.EQ_UE_Counters = {"TU": {}, "TD": {}}  # 所有CrossPoint共享
        self.ETag_BOTHSIDE_UPGRADE = False

        # ITag setup (这些数据结构由Network管理，CrossPoint共享引用)
        self.remain_tag = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}
        self.tagged_counter = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}
        self.itag_req_counter = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}
        self.excess_ITag_to_remove = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}

        # 方向控制初始化 - 将物理节点ID转换为IP位置集合
        self._init_direction_control()

        # 新架构: 为所有节点初始化所有资源（不区分IP节点和非IP节点）
        for node_pos in range(config.NUM_NODE):
            # CrossPoint
            self.cross_point["horizontal"][node_pos]["TL"] = [None] * 2
            self.cross_point["horizontal"][node_pos]["TR"] = [None] * 2
            self.cross_point["vertical"][node_pos]["TU"] = [None] * 2
            self.cross_point["vertical"][node_pos]["TD"] = [None] * 2

            # Inject/Eject queues
            self.inject_queues["TL"][node_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH_HORIZONTAL)
            self.inject_queues["TR"][node_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH_HORIZONTAL)
            self.inject_queues["TU"][node_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH_VERTICAL)
            self.inject_queues["TD"][node_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH_VERTICAL)
            self.inject_queues["EQ"][node_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH_EQ)
            self.inject_queues_pre["TL"][node_pos] = None
            self.inject_queues_pre["TR"][node_pos] = None
            self.inject_queues_pre["TU"][node_pos] = None
            self.inject_queues_pre["TD"][node_pos] = None
            self.inject_queues_pre["EQ"][node_pos] = None

            self.eject_queues["TU"][node_pos] = deque(maxlen=config.EQ_IN_FIFO_DEPTH)
            self.eject_queues["TD"][node_pos] = deque(maxlen=config.EQ_IN_FIFO_DEPTH)
            self.eject_queues_in_pre["TU"][node_pos] = None
            self.eject_queues_in_pre["TD"][node_pos] = None

            # EQ UE Counters
            self.EQ_UE_Counters["TU"][node_pos] = {"T2": 0, "T1": 0, "T0": 0}
            self.EQ_UE_Counters["TD"][node_pos] = {"T2": 0, "T1": 0, "T0": 0}

            # Channel buffers
            for key in self.config.CH_NAME_LIST:
                self.IQ_channel_buffer_pre[key][node_pos] = None
                self.EQ_channel_buffer_pre[key][node_pos] = None
                # IQ仲裁输入FIFO初始化（按IP类型分类）
                self.IQ_arbiter_input_fifo[key][node_pos] = deque(maxlen=2)
                self.IQ_arbiter_input_fifo_pre[key][node_pos] = None

            # EQ仲裁输入FIFO初始化（4个输入端口，每个端口一个FIFO）
            for input_port in ["TU", "TD", "IQ", "RB"]:
                self.EQ_arbiter_input_fifo[input_port][node_pos] = deque(maxlen=2)
                self.EQ_arbiter_input_fifo_pre[input_port][node_pos] = None
            for key in self.arrive_node_pre:
                self.arrive_node_pre[key][node_pos] = None

            # Round robin
            for key in self.round_robin.keys():
                if key == "IQ":
                    for fifo_name in ["TR", "TL", "TU", "TD", "EQ"]:
                        self.round_robin[key][fifo_name][node_pos] = deque()
                        for ch_name in self.IQ_channel_buffer.keys():
                            self.round_robin[key][fifo_name][node_pos].append(ch_name)
                elif key == "EQ":
                    for ch_name in self.IQ_channel_buffer.keys():
                        self.round_robin[key][ch_name][node_pos] = deque([0, 1, 2, 3])
                else:
                    for fifo_name in ["TU", "TD", "EQ"]:
                        self.round_robin[key][fifo_name][node_pos] = deque([0, 1, 2, 3])

            # Timing statistics
            self.inject_time[node_pos] = []
            self.eject_time[node_pos] = []
            self.avg_inject_time[node_pos] = 0
            self.avg_eject_time[node_pos] = 1

        # 新架构: 基于C2C的link初始化逻辑
        # 1. 创建普通链路（根据邻接矩阵）
        for i in range(config.NUM_NODE):
            for j in range(config.NUM_NODE):
                if adjacency_matrix[i][j] == 1:
                    # 判断链路类型：纵向链路还是横向链路
                    if abs(i - j) == config.NUM_COL:
                        # 纵向链路
                        slice_count = config.SLICE_PER_LINK_VERTICAL
                    else:
                        # 横向链路
                        slice_count = config.SLICE_PER_LINK_HORIZONTAL

                    self.links[(i, j)] = [None] * slice_count
                    self.links_flow_stat[(i, j)] = {
                        "ITag_count": 0,
                        "empty_count": 0,
                        "total_cycles": 0,
                        "eject_attempts_h": {"0": 0, "1": 0, "2": 0, ">2": 0},
                        "eject_attempts_v": {"0": 0, "1": 0, "2": 0, ">2": 0},
                    }
                    self.links_tag[(i, j)] = [LinkSlot(slot_id=self.global_slot_id_counter + idx) for idx in range(slice_count)]
                    self.global_slot_id_counter += slice_count

        # 2. 为边缘节点创建自环链路（用于Tag循环）
        for node_id in range(config.NUM_NODE):
            row = node_id // config.NUM_COL
            col = node_id % config.NUM_COL

            # 检查是否需要水平自环（左右边缘）
            if col == 0 or col == config.NUM_COL - 1:
                key = (node_id, node_id, "h")  # 水平自环
                self.links[key] = [None] * 2
                self.links_flow_stat[key] = {
                    "ITag_count": 0,
                    "empty_count": 0,
                    "total_cycles": 0,
                    "eject_attempts_h": {"0": 0, "1": 0, "2": 0, ">2": 0},
                    "eject_attempts_v": {"0": 0, "1": 0, "2": 0, ">2": 0},
                }
                self.links_tag[key] = [LinkSlot(slot_id=self.global_slot_id_counter + idx) for idx in range(2)]
                self.global_slot_id_counter += 2

            # 检查是否需要垂直自环（上下边缘）
            if row == 0 or row == config.NUM_ROW - 1:
                key = (node_id, node_id, "v")  # 垂直自环
                self.links[key] = [None] * 2
                self.links_flow_stat[key] = {
                    "ITag_count": 0,
                    "empty_count": 0,
                    "total_cycles": 0,
                    "eject_attempts_h": {"0": 0, "1": 0, "2": 0, ">2": 0},
                    "eject_attempts_v": {"0": 0, "1": 0, "2": 0, ">2": 0},
                }
                self.links_tag[key] = [LinkSlot(slot_id=self.global_slot_id_counter + idx) for idx in range(2)]
                self.global_slot_id_counter += 2

        for pos in range(config.NUM_NODE):
            # 新架构: Ring Bridge在同一节点，键直接使用节点号
            self.ring_bridge["TL"][pos] = deque(maxlen=config.RB_IN_FIFO_DEPTH)
            self.ring_bridge["TR"][pos] = deque(maxlen=config.RB_IN_FIFO_DEPTH)
            self.ring_bridge["TU"][pos] = deque(maxlen=config.RB_OUT_FIFO_DEPTH)
            self.ring_bridge["TD"][pos] = deque(maxlen=config.RB_OUT_FIFO_DEPTH)
            self.ring_bridge["EQ"][pos] = deque(maxlen=config.RB_OUT_FIFO_DEPTH)

            self.ring_bridge_pre["TL"][pos] = None
            self.ring_bridge_pre["TR"][pos] = None
            self.ring_bridge_pre["TU"][pos] = None
            self.ring_bridge_pre["TD"][pos] = None
            self.ring_bridge_pre["EQ"][pos] = None

            self.RB_UE_Counters["TL"][pos] = {"T2": 0, "T1": 0, "T0": 0}
            self.RB_UE_Counters["TR"][pos] = {"T2": 0, "T1": 0, "T0": 0}

        # 新架构: 所有节点都可以作为IP节点
        for ip_type in self.num_recv.keys():
            for source in range(config.NUM_NODE):
                destination = source  # 新架构: source和destination是同一节点
                self.num_send[ip_type][source] = 0
                self.num_recv[ip_type][destination] = 0
                self.per_send_throughput[ip_type][source] = 0
                self.per_recv_throughput[ip_type][destination] = 0

        for ip_type in self.IQ_channel_buffer.keys():
            for ip_index in range(config.NUM_NODE):
                self.IQ_channel_buffer[ip_type][ip_index] = deque(maxlen=config.IQ_CH_FIFO_DEPTH)
                self.EQ_channel_buffer[ip_type][ip_index] = deque(maxlen=config.EQ_CH_FIFO_DEPTH)
        for ip_type in self.last_select.keys():
            for ip_index in range(config.NUM_NODE):
                self.last_select[ip_type][ip_index] = "write"
        for ip_type in self.throughput.keys():
            for ip_index in range(config.NUM_NODE):
                self.throughput[ip_type][ip_index] = [0, 0, 10000000, 0]

        # 初始化RB_CAPACITY和EQ_CAPACITY (所有CrossPoint共享)
        self.RB_CAPACITY = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}
        self.EQ_CAPACITY = {"TU": {}, "TD": {}}

        # 容量计算辅助函数
        def _cap_tl(lvl):
            if lvl == "T2":
                return config.TL_Etag_T2_UE_MAX
            if lvl == "T1":
                return config.TL_Etag_T1_UE_MAX - config.TL_Etag_T2_UE_MAX
            if lvl == "T0":
                return config.RB_IN_FIFO_DEPTH - config.TL_Etag_T1_UE_MAX

        def _cap_tr(lvl):
            # 双侧下环保序模式：TR完全复用TL的配置
            if config.ORDERING_PRESERVATION_MODE == 2:
                return _cap_tl(lvl)
            # 其他模式：TR使用独立配置
            if lvl == "T2":
                return config.TR_Etag_T2_UE_MAX
            if lvl == "T1":
                return config.RB_IN_FIFO_DEPTH - config.TR_Etag_T2_UE_MAX
            return 0

        def _cap_tu(lvl):
            if lvl == "T2":
                return config.TU_Etag_T2_UE_MAX
            if lvl == "T1":
                return config.TU_Etag_T1_UE_MAX - config.TU_Etag_T2_UE_MAX
            if lvl == "T0":
                return config.EQ_IN_FIFO_DEPTH - config.TU_Etag_T1_UE_MAX

        def _cap_td(lvl):
            # 双侧下环保序模式：TD完全复用TU的配置
            if config.ORDERING_PRESERVATION_MODE == 2:
                return _cap_tu(lvl)
            # 其他模式：TD使用独立配置
            if lvl == "T2":
                return config.TD_Etag_T2_UE_MAX
            if lvl == "T1":
                return config.EQ_IN_FIFO_DEPTH - config.TD_Etag_T2_UE_MAX
            return 0

        # 为所有RB pair设置容量
        for pair in self.RB_UE_Counters["TL"]:
            self.RB_CAPACITY["TL"][pair] = {lvl: _cap_tl(lvl) for lvl in ("T0", "T1", "T2")}
        for pair in self.RB_UE_Counters["TR"]:
            self.RB_CAPACITY["TR"][pair] = {lvl: _cap_tr(lvl) for lvl in ("T0", "T1", "T2")}

        # 为所有EQ位置设置容量
        for pos in self.EQ_UE_Counters["TU"]:
            self.EQ_CAPACITY["TU"][pos] = {lvl: _cap_tu(lvl) for lvl in ("T0", "T1", "T2")}
        for pos in self.EQ_UE_Counters["TD"]:
            self.EQ_CAPACITY["TD"][pos] = {lvl: _cap_td(lvl) for lvl in ("T0", "T1", "T2")}

        # 创建CrossPoint对象
        from .cross_point import CrossPoint

        self.crosspoints = {}
        # 新架构: 所有节点都有CrossPoint
        for ip_pos in range(config.NUM_NODE):
            # 创建horizontal和vertical CrossPoint
            cp_h = CrossPoint(ip_pos, "horizontal", config, network_ref=self)
            cp_v = CrossPoint(ip_pos, "vertical", config, network_ref=self)

            # 设置ITag配置
            cp_h.setup_itag("TL", ip_pos, config.ITag_MAX_NUM_H)
            cp_h.setup_itag("TR", ip_pos, config.ITag_MAX_NUM_H)
            cp_v.setup_itag("TU", ip_pos, config.ITag_MAX_NUM_V)
            cp_v.setup_itag("TD", ip_pos, config.ITag_MAX_NUM_V)

            self.crosspoints[ip_pos] = {"horizontal": cp_h, "vertical": cp_v}

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
        """
        简化版本：只检查FIFO深度，I-Tag逻辑已移至CrossPoint

        用于EQ/TU/TD本地注入的FIFO深度检查
        TR/TL方向的注入已由CrossPoint管理，不再使用此方法
        """
        if flit.source == flit.destination or len(flit.path) <= 1:
            # 本地注入到EQ
            return len(self.inject_queues["EQ"][current]) < self.config.IQ_OUT_FIFO_DEPTH_EQ

        # 纵向环移动（上下方向） - 本地注入到TU/TD
        if abs(current - next_node) == self.config.NUM_COL:
            # 判断向上还是向下
            if next_node > current:
                # 向下移动
                return len(self.inject_queues["TD"][current]) < self.config.IQ_OUT_FIFO_DEPTH_VERTICAL
            else:
                # 向上移动
                return len(self.inject_queues["TU"][current]) < self.config.IQ_OUT_FIFO_DEPTH_VERTICAL

        # 横向环移动（TR/TL）已由CrossPoint处理，此路径不应到达
        # 保留兼容性，返回False
        return False

    def update_excess_ITag(self):
        """在主循环中调用，处理多余ITag释放"""
        # 调用所有CrossPoint的update_excess_ITag方法
        for ip_pos in self.crosspoints:
            self.crosspoints[ip_pos]["horizontal"].update_excess_ITag()
            self.crosspoints[ip_pos]["vertical"].update_excess_ITag()

    def update_cross_point(self):
        """新架构: 基于物理直接映射的CrossPoint更新"""
        # 新架构: 所有节点都有CrossPoint
        for ip_pos in range(self.config.NUM_NODE):
            row = ip_pos // self.config.NUM_COL
            col = ip_pos % self.config.NUM_COL

            # 计算物理邻居位置
            left_pos = ip_pos - 1 if col > 0 else None
            right_pos = ip_pos + 1 if col < self.config.NUM_COL - 1 else None
            up_pos = ip_pos - self.config.NUM_COL if row > 0 else None
            down_pos = ip_pos + self.config.NUM_COL if row < self.config.NUM_ROW - 1 else None

            # 水平CrossPoint连接 - TR方向 (向右)
            if right_pos is not None:
                # 有右邻居: arrival来自左邻居, departure去往右邻居
                arrival_link = (left_pos, ip_pos) if left_pos is not None else (ip_pos, ip_pos, "h")
                departure_link = (ip_pos, right_pos)
            else:
                # 右边界: 使用水平自环
                arrival_link = (left_pos, ip_pos) if left_pos is not None else (ip_pos, ip_pos, "h")
                departure_link = (ip_pos, ip_pos, "h")

            arrival_slice = self.links[arrival_link][-1] if arrival_link in self.links else None
            departure_slice = self.links[departure_link][0] if departure_link in self.links else None
            self.cross_point["horizontal"][ip_pos]["TR"] = [arrival_slice, departure_slice]

            # 水平CrossPoint连接 - TL方向 (向左)
            if left_pos is not None:
                # 有左邻居: arrival来自右邻居, departure去往左邻居
                arrival_link = (right_pos, ip_pos) if right_pos is not None else (ip_pos, ip_pos, "h")
                departure_link = (ip_pos, left_pos)
            else:
                # 左边界: 使用水平自环
                arrival_link = (right_pos, ip_pos) if right_pos is not None else (ip_pos, ip_pos, "h")
                departure_link = (ip_pos, ip_pos, "h")

            arrival_slice = self.links[arrival_link][-1] if arrival_link in self.links else None
            departure_slice = self.links[departure_link][0] if departure_link in self.links else None
            self.cross_point["horizontal"][ip_pos]["TL"] = [arrival_slice, departure_slice]

            # 垂直CrossPoint连接 - TU方向 (向上)
            if up_pos is not None:
                # 有上邻居: arrival来自下邻居, departure去往上邻居
                arrival_link = (down_pos, ip_pos) if down_pos is not None else (ip_pos, ip_pos, "v")
                departure_link = (ip_pos, up_pos)
            else:
                # 上边界: 使用垂直自环
                arrival_link = (down_pos, ip_pos) if down_pos is not None else (ip_pos, ip_pos, "v")
                departure_link = (ip_pos, ip_pos, "v")

            arrival_slice = self.links[arrival_link][-1] if arrival_link in self.links else None
            departure_slice = self.links[departure_link][0] if departure_link in self.links else None
            self.cross_point["vertical"][ip_pos]["TU"] = [arrival_slice, departure_slice]

            # 垂直CrossPoint连接 - TD方向 (向下)
            if down_pos is not None:
                # 有下邻居: arrival来自上邻居, departure去往下邻居
                arrival_link = (up_pos, ip_pos) if up_pos is not None else (ip_pos, ip_pos, "v")
                departure_link = (ip_pos, down_pos)
            else:
                # 下边界: 使用垂直自环
                arrival_link = (up_pos, ip_pos) if up_pos is not None else (ip_pos, ip_pos, "v")
                departure_link = (ip_pos, ip_pos, "v")

            arrival_slice = self.links[arrival_link][-1] if arrival_link in self.links else None
            departure_slice = self.links[departure_link][0] if departure_link in self.links else None
            self.cross_point["vertical"][ip_pos]["TD"] = [arrival_slice, departure_slice]

            # 更新CrossPoint冲突状态 (基于arrival slice是否有flit)
            # 水平冲突
            tr_arrival_link = (left_pos, ip_pos) if left_pos is not None else (ip_pos, ip_pos, "h")
            tl_arrival_link = (right_pos, ip_pos) if right_pos is not None else (ip_pos, ip_pos, "h")
            new_tr_conflict = self.links[tr_arrival_link][-1] is not None if tr_arrival_link in self.links else False
            new_tl_conflict = self.links[tl_arrival_link][-1] is not None if tl_arrival_link in self.links else False

            self.crosspoint_conflict["horizontal"][ip_pos]["TR"].insert(0, new_tr_conflict)
            self.crosspoint_conflict["horizontal"][ip_pos]["TR"] = self.crosspoint_conflict["horizontal"][ip_pos]["TR"][:2]
            self.crosspoint_conflict["horizontal"][ip_pos]["TL"].insert(0, new_tl_conflict)
            self.crosspoint_conflict["horizontal"][ip_pos]["TL"] = self.crosspoint_conflict["horizontal"][ip_pos]["TL"][:2]

            # 垂直冲突
            tu_arrival_link = (down_pos, ip_pos) if down_pos is not None else (ip_pos, ip_pos, "v")
            td_arrival_link = (up_pos, ip_pos) if up_pos is not None else (ip_pos, ip_pos, "v")
            new_tu_conflict = self.links[tu_arrival_link][-1] is not None if tu_arrival_link in self.links else False
            new_td_conflict = self.links[td_arrival_link][-1] is not None if td_arrival_link in self.links else False

            self.crosspoint_conflict["vertical"][ip_pos]["TU"].insert(0, new_tu_conflict)
            self.crosspoint_conflict["vertical"][ip_pos]["TU"] = self.crosspoint_conflict["vertical"][ip_pos]["TU"][:2]
            self.crosspoint_conflict["vertical"][ip_pos]["TD"].insert(0, new_td_conflict)
            self.crosspoint_conflict["vertical"][ip_pos]["TD"] = self.crosspoint_conflict["vertical"][ip_pos]["TD"][:2]

    def plan_move(self, flit, cycle):
        self.cycle = cycle

        # 计算行和列的起始和结束点
        current, next_node = flit.current_link[:2] if len(flit.current_link) == 3 else flit.current_link

        # 处理所有link（包括普通link和自环）
        link = self.links.get(flit.current_link)
        if link is not None:
            # Plan moves for all links including self-loops
            return self._handle_flit(flit, link, current, next_node)

    def _position_to_physical_node(self, position):
        """新架构: position就是physical node ID，无需映射"""
        return position

    def _can_eject_in_order(self, flit: Flit, target_eject_node, direction=None):
        """检查flit是否可以按序下环（包含方向检查）

        Args:
            target_eject_node: 目标下环节点的position（映射后的位置），可能是中间节点
        """
        # 先判断是否需要保序
        if not self._need_in_order_check(flit):
            return True

        # 方向检查：如果指定了方向，检查是否在允许的方向列表中
        if direction is not None and hasattr(flit, "allowed_eject_directions") and flit.allowed_eject_directions is not None:
            if direction not in flit.allowed_eject_directions:
                return False  # 方向不允许，不能下环

        # 确保flit已设置保序信息
        if not hasattr(flit, "src_dest_order_id"):
            return True

        if flit.src_dest_order_id == -1:
            return True

        # 获取原始的src（物理节点ID）
        src = get_original_source_node(flit)

        # 使用最终目的地（而不是中间下环节点）作为key的dest
        # 使用辅助函数统一获取原始节点ID
        dest = get_original_destination_node(flit)

        # 根据保序粒度构造key
        if self.config.ORDERING_GRANULARITY == 0:  # IP层级
            # 获取IP类型信息
            src_type = get_original_source_type(flit)
            dest_type = get_original_destination_type(flit)
            key = (src, src_type, dest, dest_type, direction)
        else:  # 节点层级（granularity == 1）
            key = (src, dest, direction)

        # 检查是否是期望的下一个顺序ID
        # 每个network只记录自己的order_id，不需要区分packet_category
        expected_order_id = self.order_tracking_table[key] + 1

        can_eject = flit.src_dest_order_id == expected_order_id
        return can_eject

    def _need_in_order_check(self, flit: Flit):
        """判断该flit是否需要保序检查"""
        if self.config.ORDERING_PRESERVATION_MODE == 0:
            return False

        # 检查通道类型是否需要保序
        packet_category = self._get_flit_packet_category(flit)
        if hasattr(self.config, "IN_ORDER_PACKET_CATEGORIES"):
            if packet_category not in self.config.IN_ORDER_PACKET_CATEGORIES:
                return False

        # 获取真实的src和dest
        src = get_original_source_node(flit)
        dest = get_original_destination_node(flit)

        # 如果未配置特定对或配置为空，则全部保序
        if not hasattr(self.config, "IN_ORDER_EJECTION_PAIRS") or len(self.config.IN_ORDER_EJECTION_PAIRS) == 0:
            return True

        # 检查是否在配置的保序对列表中
        return [src, dest] in self.config.IN_ORDER_EJECTION_PAIRS

    def _get_flit_packet_category(self, flit: Flit):
        """获取flit的包类型分类"""
        # 优先判断是否为数据包
        if hasattr(flit, "flit_type") and flit.flit_type == "data":
            return "DATA"
        elif flit.rsp_type is not None:
            return "RSP"
        elif flit.req_type is not None:
            return "REQ"
        else:
            return "REQ"  # 默认为REQ

    def _can_upgrade_to_T0_in_order(self, flit: Flit, node, direction=None):
        """检查flit是否可以按序升级到T0（包含方向检查）

        Args:
            node: 目标节点的position（映射后的位置），可能是中间节点
        """
        # 先判断是否需要保序
        if not self._need_in_order_check(flit):
            return True

        # 方向检查：如果指定了方向，检查是否在允许的方向列表中
        if direction is not None and hasattr(flit, "allowed_eject_directions") and flit.allowed_eject_directions is not None:
            if direction not in flit.allowed_eject_directions:
                return False  # 方向不允许，不能升级到T0

        # 确保flit已设置保序信息
        if not hasattr(flit, "src_dest_order_id"):
            return True

        # 获取原始的src（物理节点ID）
        src = get_original_source_node(flit)

        # 使用最终目的地（而不是中间下环节点）
        dest = get_original_destination_node(flit)

        # 使用 (src物理ID, final_dest物理ID, direction) 作为键
        key = (src, dest, direction)

        # 检查是否是期望的下一个顺序ID
        # 每个network只记录自己的order_id，不需要区分packet_category
        expected_order_id = self.order_tracking_table[key] + 1
        return flit.src_dest_order_id == expected_order_id

    def _update_order_tracking_table(self, flit: Flit, target_node: int, direction: str):
        """更新保序跟踪表

        Args:
            target_node: 目标节点的position（映射后的位置），可能是中间节点
        """
        # 先判断是否需要保序
        if not self._need_in_order_check(flit):
            return

        # 确保flit已设置保序信息
        if not hasattr(flit, "src_dest_order_id") or not hasattr(flit, "packet_category"):
            return

        if flit.src_dest_order_id == -1:
            return

        # 获取原始的src（物理节点ID）
        src = get_original_source_node(flit)

        # 使用最终目的地（而不是中间下环节点）
        dest = get_original_destination_node(flit)

        # 根据保序粒度构造key（与_can_eject_in_order保持一致）
        if self.config.ORDERING_GRANULARITY == 0:  # IP层级
            # 获取IP类型信息
            src_type = get_original_source_type(flit)
            dest_type = get_original_destination_type(flit)
            key = (src, src_type, dest, dest_type, direction)
        else:  # 节点层级（granularity == 1）
            key = (src, dest, direction)

        # 更新保序跟踪表
        # 每个network只记录自己的order_id，不需要区分packet_category
        self.order_tracking_table[key] = flit.src_dest_order_id

    def _init_direction_control(self):
        """初始化方向控制 - 使用物理节点ID集合（与配置文件和原始源节点相同的编号）"""
        # 为每个方向构建允许的源节点集合（物理节点ID）
        self.allowed_source_nodes = {
            "TL": set(self.config.TL_ALLOWED_SOURCE_NODES),
            "TR": set(self.config.TR_ALLOWED_SOURCE_NODES),
            "TU": set(self.config.TU_ALLOWED_SOURCE_NODES),
            "TD": set(self.config.TD_ALLOWED_SOURCE_NODES),
        }

    def determine_allowed_eject_directions(self, flit: Flit):
        """确定flit允许的下环方向"""
        mode = self.config.ORDERING_PRESERVATION_MODE

        # Mode 0: 不保序，所有方向都允许
        if mode == 0:
            return None

        # Mode 1: 单侧下环，固定只允许TL和TU方向
        if mode == 1:
            return ["TL", "TU"]

        # Mode 2: 双侧下环，根据方向配置决定
        if mode == 2:
            # 获取原始源节点编号（物理节点ID，未经node_map映射）
            src_node = get_original_source_node(flit)
            # self.error_log(flit, 1, 3)

            # 检查各方向是否允许
            allowed_dirs = []
            for direction in ["TL", "TR", "TU", "TD"]:
                # 空列表表示所有节点都允许
                if len(self.allowed_source_nodes[direction]) == 0 or src_node in self.allowed_source_nodes[direction]:
                    allowed_dirs.append(direction)

            return allowed_dirs if allowed_dirs else None

        # 未知模式，默认不保序
        return None

    def execute_moves(self, flit: Flit, cycle):
        # 情况1：is_arrive=True（已在IP模块）
        # 这种情况不应该在flits列表中，如果出现说明有bug
        if flit.is_arrive:
            flit.arrival_network_cycle = cycle
            return True  # 异常移除

        # 情况2：current_link=None（已下环到pre缓冲）
        # flit已离开Link系统，应该从flits列表移除
        if flit.current_link is None:
            return True  # 正常移除

        # 情况3：在Link上传输（包括普通link和自环）
        # 自环link使用3元组格式: (node, node, "h"/"v")
        # 普通link使用2元组格式: (u, v)
        self.set_link_slice(flit.current_link, flit.current_seat_index, flit, cycle)

        return False  # 保留在flits列表

    @property
    def rn_positions(self):
        """Cached property for RN positions - 新架构: 所有节点"""
        if self._rn_positions is None:
            self._rn_positions = list(range(self.config.NUM_NODE))
        return self._rn_positions

    @property
    def sn_positions(self):
        """Cached property for SN positions - 新架构: 所有节点"""
        if self._sn_positions is None:
            self._sn_positions = list(range(self.config.NUM_NODE))
        return self._sn_positions

    def clear_position_cache(self):
        """Clear position cache when network configuration changes"""
        self._all_ip_positions = None
        self._rn_positions = None
        self._sn_positions = None

    @property
    def all_ip_positions(self):
        """Cached property for all IP positions - 新架构: 所有节点"""
        if self._all_ip_positions is None:
            self._all_ip_positions = list(range(self.config.NUM_NODE))
        return self._all_ip_positions

    @property
    def rn_positions(self):
        """Cached property for RN positions - 新架构: 所有节点"""
        if self._rn_positions is None:
            self._rn_positions = list(range(self.config.NUM_NODE))
        return self._rn_positions

    @property
    def sn_positions(self):
        """Cached property for SN positions - 新架构: 所有节点"""
        if self._sn_positions is None:
            self._sn_positions = list(range(self.config.NUM_NODE))
        return self._sn_positions

    # ==================== 新的handle flit辅助函数 ====================

    def _continue_looping(self, flit, link, next_pos):
        """
        继续绕环

        新架构: 纵向link是直接连接，不应该创建自环
        只有边界节点的自环才是合法的（用于Tag circulation）

        Args:
            flit: 当前flit
            link: 当前链路
            next_pos: 下一个绕环位置
        """
        link[flit.current_seat_index] = None
        current_node = flit.current_link[1]

        # 判断当前link是横向还是纵向（用于确定自环类型）
        if len(flit.current_link) == 3:
            # 当前已经是自环
            link_type = flit.current_link[2]
        else:
            # 普通link：根据距离判断
            u, v = flit.current_link[:2]
            if abs(u - v) == self.config.NUM_COL:
                link_type = "v"  # 纵向
            else:
                link_type = "h"  # 横向

        # 构造新link
        if next_pos == current_node:
            # 自环：使用3元组格式
            new_link = (current_node, next_pos, link_type)
        else:
            # 普通link：使用2元组格式
            new_link = (current_node, next_pos)

        # 检查link是否存在
        if new_link not in self.links:
            raise ValueError(new_link)

        # 设置新link和seat_index
        flit.current_link = new_link
        flit.current_seat_index = 0  # 总是从头开始

        # 验证seat_index合法性
        new_link_length = len(self.links[new_link])
        if flit.current_seat_index >= new_link_length:
            raise RuntimeError(f"[Cycle {self.cycle}] Invalid seat_index {flit.current_seat_index} " f"for link {new_link} with length {new_link_length}. " f"Flit: {flit.packet_id}.{flit.flit_id}")

    def _analyze_flit_state(self, flit, current, next_node):
        """
        分析flit当前状态，判断是否应该尝试下环

        Args:
            flit: 当前flit
            current: 当前链路起点
            next_node: 当前链路终点

        Returns:
            dict: {
                'should_eject': bool,  # 是否应该尝试下环
                'direction': str,      # 下环方向（TL/TR/TU/TD）
                'next_pos': int        # 下次绕环的位置
            }
        """
        # 计算边界条件
        row = next_node // self.config.NUM_COL
        col = next_node % self.config.NUM_COL
        num_rows = self.config.NUM_NODE // self.config.NUM_COL

        row_start = row * self.config.NUM_COL
        row_end = row_start + self.config.NUM_COL - 1
        col_start = col
        col_end = col + (num_rows - 1) * self.config.NUM_COL

        # 判断方向和计算下一个绕环位置
        if current == next_node:
            # flit在自环上，确定离开自环后的反向移动
            if len(flit.current_link) == 3 and flit.current_link[2] == "v":
                # 纵向自环
                if row == 0:
                    # 上边界 → 向下
                    direction = "TD"
                    next_pos = next_node + self.config.NUM_COL
                else:
                    # 下边界 → 向上
                    direction = "TU"
                    next_pos = next_node - self.config.NUM_COL
            elif len(flit.current_link) == 3 and flit.current_link[2] == "h":
                # 横向自环
                if col == 0:
                    # 左边界 → 向右
                    direction = "TR"
                    next_pos = next_node + 1
                else:
                    # 右边界 → 向左
                    direction = "TL"
                    next_pos = next_node - 1
            else:
                # fallback: 非法情况
                return {"should_eject": False, "direction": "", "next_pos": next_node}

            return {"should_eject": False, "direction": direction, "next_pos": next_pos}
        elif abs(current - next_node) == 1:
            # 非边界横向环
            if current - next_node == 1:
                # 向左
                direction = "TL"
                # 检查是否已经到达左边界
                if next_node == row_start:
                    # 已到左边界，进入自环
                    next_pos = next_node
                else:
                    next_pos = next_node - 1
            else:
                # 向右
                direction = "TR"
                # 检查是否已经到达右边界
                if next_node == row_end:
                    # 已到右边界，进入自环
                    next_pos = next_node
                else:
                    next_pos = next_node + 1
        else:
            # 非边界纵向环
            if current - next_node == self.config.NUM_COL:
                # 向上
                direction = "TU"
                # 检查是否已经到达上边界
                if next_node == col_start:
                    # 已到上边界，进入自环
                    next_pos = next_node
                else:
                    next_pos = next_node - self.config.NUM_COL
            else:
                # 向下
                direction = "TD"
                # 检查是否已经到达下边界
                if next_node == col_end:
                    # 已到下边界，进入自环
                    next_pos = next_node
                else:
                    next_pos = next_node + self.config.NUM_COL

        # 判断是否应该尝试下环：只有绕回到起始位置才尝试
        should_eject = next_node == flit.current_position

        return {"should_eject": should_eject, "direction": direction, "next_pos": next_pos}

    def _handle_flit(self, flit: Flit, link, current, next_node):
        """
        处理flit在链路末端的行为（统一版本）

        Args:
            flit: 当前flit
            link: 当前链路
            current: 当前链路起点
            next_node: 当前链路终点
        """
        # 1. 非链路末端：继续前进
        if flit.current_seat_index < len(link) - 1:
            link[flit.current_seat_index] = None
            flit.current_seat_index += 1
            return

        # 2. 更新current_position(不使用path_index)
        if not flit.is_delay:
            flit.current_position = next_node

        # 3. 判断是否需要下环（使用CrossPoint的新方法）
        final_destination = flit.path[-1]

        # 先尝试水平CrossPoint判断
        should_eject_h = False
        eject_target_h = ""
        eject_direction_h = ""
        if next_node in self.crosspoints and "horizontal" in self.crosspoints[next_node]:
            cp_h = self.crosspoints[next_node]["horizontal"]
            should_eject_h, eject_target_h, eject_direction_h = cp_h.should_eject_flit(flit, next_node)

        # 再尝试垂直CrossPoint判断
        should_eject_v = False
        eject_target_v = ""
        eject_direction_v = ""
        if next_node in self.crosspoints and "vertical" in self.crosspoints[next_node]:
            cp_v = self.crosspoints[next_node]["vertical"]
            should_eject_v, eject_target_v, eject_direction_v = cp_v.should_eject_flit(flit, next_node)

        # 4. 处理下环
        if should_eject_h or should_eject_v:
            # 确定使用哪个CrossPoint
            if should_eject_h:
                crosspoint = self.crosspoints[next_node]["horizontal"]
                eject_direction = eject_direction_h
                eject_target = eject_target_h
            else:
                crosspoint = self.crosspoints[next_node]["vertical"]
                eject_direction = eject_direction_v
                eject_target = eject_target_v

            # 统计下环尝试次数
            if eject_direction in ["TL", "TR"]:
                flit.eject_attempts_h += 1
            else:
                flit.eject_attempts_v += 1

            # 保序检查：统一检查方向和order_id
            # _can_eject_in_order内部会先判断是否需要保序检查
            if not self._can_eject_in_order(flit, final_destination, eject_direction):
                # 保序检查失败（方向不对或order_id不对），继续绕环
                # 仅在首次尝试下环时记录为"保序导致绕环"
                if eject_direction in ["TL", "TR"]:
                    if flit.eject_attempts_h <= 1:
                        flit.ordering_blocked_eject_h += 1
                else:
                    if flit.eject_attempts_v <= 1:
                        flit.ordering_blocked_eject_v += 1

                # 模式1：保序失败时也升级ETag
                if self.config.ORDERING_ETAG_UPGRADE_MODE == 1:
                    if not flit.is_delay:
                        flit.is_delay = True

                    upgrade_to = crosspoint._determine_etag_upgrade(flit, eject_direction)
                    if upgrade_to:
                        flit.ETag_priority = upgrade_to
                        if upgrade_to == "T0":
                            crosspoint._register_T0_slot(flit, eject_direction)

                state = self._analyze_flit_state(flit, current, next_node)
                self._continue_looping(flit, link, state["next_pos"])
                return

            # 尝试下环（只检查资源：队列容量和entry）
            success, fail_reason = crosspoint._try_eject(flit, eject_direction, final_destination, link, ring_bridge=self.ring_bridge, eject_queues=self.eject_queues)

            if success:
                # 下环成功
                return

            # 下环失败：根据失败原因决定是否继续绕环
            # fail_reason可能是: "order" (保序), "capacity" (容量), "entry" (Entry不足)
            # 所有情况下都继续绕环，避免卡在自环上
            should_continue_loop = True

            # E-Tag升级
            if not flit.is_delay:
                flit.is_delay = True

            upgrade_to = crosspoint._determine_etag_upgrade(flit, eject_direction)
            if upgrade_to:
                flit.ETag_priority = upgrade_to
                if upgrade_to == "T0":
                    crosspoint._register_T0_slot(flit, eject_direction)

            # 继续绕环
            if should_continue_loop:
                state = self._analyze_flit_state(flit, current, next_node)
                self._continue_looping(flit, link, state["next_pos"])
            return

        # 5. 不需要下环：继续移动
        # 先分析flit状态，判断是否需要通过自环绕过边界
        state = self._analyze_flit_state(flit, current, next_node)

        # 如果需要进入自环（next_pos == next_node），则进入自环绕过边界
        if state["next_pos"] == next_node:
            # 到达边界，需要进入自环
            self._continue_looping(flit, link, state["next_pos"])
            return

        # 否则，尝试按path正常移动
        try:
            current_path_index = flit.path.index(next_node)
            if current_path_index + 1 < len(flit.path):
                next_hop = flit.path[current_path_index + 1]
                # 检查path的下一跳是否与分析得到的next_pos一致
                if next_hop == state["next_pos"]:
                    # 一致，正常移动
                    link[flit.current_seat_index] = None
                    flit.current_link = (next_node, next_hop)
                    flit.current_seat_index = 0
                    return
                else:
                    # 不一致，说明需要绕环（path可能穿过边界）
                    self._continue_looping(flit, link, state["next_pos"])
                    return
        except ValueError:
            # next_node不在path中,继续绕环
            pass

        # 6. 继续绕环
        self._continue_looping(flit, link, state["next_pos"])

    # ==================== 原有的辅助函数 ====================

    def _update_link_statistics_on_set(self, link, slice_index, new_flit, old_flit, cycle):
        """
        在设置链路slice时增量更新统计数据

        Args:
            link: 链路tuple (src, dst)
            slice_index: slice索引
            new_flit: 新设置的flit（可以是None表示清空）
            old_flit: 被替换的flit（可以是None表示原本为空）
            cycle: 当前周期
        """
        if link not in self.links_flow_stat:
            return

        # 根据新flit状态增加对应计数（每次设置slice时统计一次）

        # 首先检查ITag，无论是否有flit都要检查
        if link in self.links_tag and slice_index < len(self.links_tag[link]):
            tag_info = self.links_tag[link][slice_index]
            if tag_info is not None:
                # 有ITag标记
                self.links_flow_stat[link]["ITag_count"] += 1

        # 然后处理flit统计
        if new_flit is None:
            # slice为空，且没有ITag标记，才是真正的空闲
            if link not in self.links_tag or slice_index >= len(self.links_tag[link]) or self.links_tag[link][slice_index] is None:
                self.links_flow_stat[link]["empty_count"] += 1
        else:
            # slice被flit占用，按下环尝试次数分组统计
            self._update_eject_attempts_stats(link, new_flit)

    def _update_eject_attempts_stats(self, link, flit):
        """
        根据flit的下环尝试次数更新链路统计

        Args:
            link: 链路标识 (i, j) 或 (i, j, type)
            flit: flit对象
        """
        if not hasattr(flit, "eject_attempts_h") or not hasattr(flit, "eject_attempts_v"):
            return

        # 判断链路方向并更新相应的统计
        # 处理3元组自环和2元组普通link
        if len(link) == 3:
            i, j, link_direction = link
        else:
            i, j = link
            link_direction = None

        is_self_loop = i == j  # 自环链路
        is_horizontal = abs(i - j) == 1  # 横向链路
        is_vertical = abs(i - j) > 1  # 纵向链路

        if is_self_loop:
            # 自环链路：根据行号判断属于哪个环
            # 偶数行是纵向环，奇数行是横向环
            row = i // self.config.NUM_COL
            if row % 2 == 0:
                # 偶数行 → 纵向环 → 只统计纵向
                attempts = flit.eject_attempts_v
                if attempts == 0:
                    self.links_flow_stat[link]["eject_attempts_v"]["0"] += 1
                elif attempts == 1:
                    self.links_flow_stat[link]["eject_attempts_v"]["1"] += 1
                elif attempts == 2:
                    self.links_flow_stat[link]["eject_attempts_v"]["2"] += 1
                else:
                    self.links_flow_stat[link]["eject_attempts_v"][">2"] += 1
            else:
                # 奇数行 → 横向环 → 只统计横向
                attempts = flit.eject_attempts_h
                if attempts == 0:
                    self.links_flow_stat[link]["eject_attempts_h"]["0"] += 1
                elif attempts == 1:
                    self.links_flow_stat[link]["eject_attempts_h"]["1"] += 1
                elif attempts == 2:
                    self.links_flow_stat[link]["eject_attempts_h"]["2"] += 1
                else:
                    self.links_flow_stat[link]["eject_attempts_h"][">2"] += 1
        elif is_horizontal:
            # 横向链路，统计横向下环尝试次数
            attempts = flit.eject_attempts_h
            if attempts == 0:
                self.links_flow_stat[link]["eject_attempts_h"]["0"] += 1
            elif attempts == 1:
                self.links_flow_stat[link]["eject_attempts_h"]["1"] += 1
            elif attempts == 2:
                self.links_flow_stat[link]["eject_attempts_h"]["2"] += 1
            else:
                self.links_flow_stat[link]["eject_attempts_h"][">2"] += 1
        elif is_vertical:
            # 纵向链路，统计纵向下环尝试次数
            attempts = flit.eject_attempts_v
            if attempts == 0:
                self.links_flow_stat[link]["eject_attempts_v"]["0"] += 1
            elif attempts == 1:
                self.links_flow_stat[link]["eject_attempts_v"]["1"] += 1
            elif attempts == 2:
                self.links_flow_stat[link]["eject_attempts_v"]["2"] += 1
            else:
                self.links_flow_stat[link]["eject_attempts_v"][">2"] += 1

    def collect_cycle_end_link_statistics(self, cycle):
        """
        在每个周期结束时统计所有链路第一个slice位置的使用情况

        Args:
            cycle: 当前周期
        """
        for link in self.links_flow_stat:
            if link not in self.links:
                continue

            # 只统计第一个slice位置(索引0)的使用情况
            slice_index = 0
            if slice_index >= len(self.links[link]):
                continue

            flit = self.links[link][slice_index]
            # 首先检查ITag，无论是否有flit都要检查
            if link in self.links_tag and slice_index < len(self.links_tag[link]):
                slot = self.links_tag[link][slice_index]
                if slot.itag_reserved:
                    # 有ITag标记
                    self.links_flow_stat[link]["ITag_count"] += 1

            if flit is None:
                # slice为空，且没有ITag标记，才是真正的空闲
                if link not in self.links_tag or slice_index >= len(self.links_tag[link]) or not self.links_tag[link][slice_index].itag_reserved:
                    self.links_flow_stat[link]["empty_count"] += 1
            else:
                # slice被flit占用，按下环尝试次数分组统计
                self._update_eject_attempts_stats(link, flit)

            # 更新总周期计数
            self.links_flow_stat[link]["total_cycles"] += 1

    def update_fifo_stats_after_move(self, in_pos):
        """在move操作后批量更新所有FIFO统计"""
        ip_pos = in_pos  # 新架构: in_pos和ip_pos是同一节点

        # IQ统计 - inject_queues
        for direction in ["TR", "TL", "TU", "TD", "EQ"]:
            if in_pos in self.inject_queues.get(direction, {}):
                depth = len(self.inject_queues[direction][in_pos])
                if in_pos not in self.fifo_depth_sum["IQ"][direction]:
                    self.fifo_depth_sum["IQ"][direction][in_pos] = 0
                    self.fifo_max_depth["IQ"][direction][in_pos] = 0
                self.fifo_depth_sum["IQ"][direction][in_pos] += depth
                self.fifo_max_depth["IQ"][direction][in_pos] = max(self.fifo_max_depth["IQ"][direction][in_pos], depth)

        # IQ CH_buffer统计
        for ip_type in self.IQ_channel_buffer:
            if in_pos in self.IQ_channel_buffer[ip_type]:
                depth = len(self.IQ_channel_buffer[ip_type][in_pos])
                if in_pos not in self.fifo_depth_sum["IQ"]["CH_buffer"]:
                    self.fifo_depth_sum["IQ"]["CH_buffer"][in_pos] = {}
                    self.fifo_max_depth["IQ"]["CH_buffer"][in_pos] = {}
                if ip_type not in self.fifo_depth_sum["IQ"]["CH_buffer"][in_pos]:
                    self.fifo_depth_sum["IQ"]["CH_buffer"][in_pos][ip_type] = 0
                    self.fifo_max_depth["IQ"]["CH_buffer"][in_pos][ip_type] = 0
                self.fifo_depth_sum["IQ"]["CH_buffer"][in_pos][ip_type] += depth
                self.fifo_max_depth["IQ"]["CH_buffer"][in_pos][ip_type] = max(self.fifo_max_depth["IQ"]["CH_buffer"][in_pos][ip_type], depth)

        # RB统计 - ring_bridge
        # 新架构: ring_bridge键直接使用节点号
        for direction in ["TR", "TL", "TU", "TD", "EQ"]:
            if in_pos in self.ring_bridge.get(direction, {}):
                depth = len(self.ring_bridge[direction][in_pos])
                if in_pos not in self.fifo_depth_sum["RB"][direction]:
                    self.fifo_depth_sum["RB"][direction][in_pos] = 0
                    self.fifo_max_depth["RB"][direction][in_pos] = 0
                self.fifo_depth_sum["RB"][direction][in_pos] += depth
                self.fifo_max_depth["RB"][direction][in_pos] = max(self.fifo_max_depth["RB"][direction][in_pos], depth)

        # EQ统计 - eject_queues
        for direction in ["TU", "TD"]:
            if ip_pos in self.eject_queues.get(direction, {}):
                depth = len(self.eject_queues[direction][ip_pos])
                if ip_pos not in self.fifo_depth_sum["EQ"][direction]:
                    self.fifo_depth_sum["EQ"][direction][ip_pos] = 0
                    self.fifo_max_depth["EQ"][direction][ip_pos] = 0
                self.fifo_depth_sum["EQ"][direction][ip_pos] += depth
                self.fifo_max_depth["EQ"][direction][ip_pos] = max(self.fifo_max_depth["EQ"][direction][ip_pos], depth)

        # EQ CH_buffer统计
        for ip_type in self.EQ_channel_buffer:
            if ip_pos in self.EQ_channel_buffer[ip_type]:
                depth = len(self.EQ_channel_buffer[ip_type][ip_pos])
                if ip_pos not in self.fifo_depth_sum["EQ"]["CH_buffer"]:
                    self.fifo_depth_sum["EQ"]["CH_buffer"][ip_pos] = {}
                    self.fifo_max_depth["EQ"]["CH_buffer"][ip_pos] = {}
                if ip_type not in self.fifo_depth_sum["EQ"]["CH_buffer"][ip_pos]:
                    self.fifo_depth_sum["EQ"]["CH_buffer"][ip_pos][ip_type] = 0
                    self.fifo_max_depth["EQ"]["CH_buffer"][ip_pos][ip_type] = 0
                self.fifo_depth_sum["EQ"]["CH_buffer"][ip_pos][ip_type] += depth
                self.fifo_max_depth["EQ"]["CH_buffer"][ip_pos][ip_type] = max(self.fifo_max_depth["EQ"]["CH_buffer"][ip_pos][ip_type], depth)

        # === ITag累计统计 ===
        # IQ_OUT统计 (只统计TR/TL横向注入)
        for direction in ["TR", "TL"]:
            if direction in self.inject_queues and in_pos in self.inject_queues[direction]:
                itag_count = sum(1 for flit in self.inject_queues[direction][in_pos] if getattr(flit, "itag_h", False))

                if in_pos not in self.fifo_itag_cumulative_count["IQ"][direction]:
                    self.fifo_itag_cumulative_count["IQ"][direction][in_pos] = 0
                self.fifo_itag_cumulative_count["IQ"][direction][in_pos] += itag_count

        # RB_OUT统计 (TU/TD纵向转向)
        for direction in ["TU", "TD"]:
            if direction in self.ring_bridge and in_pos in self.ring_bridge[direction]:
                itag_count = sum(1 for flit in self.ring_bridge[direction][in_pos] if getattr(flit, "itag_v", False))

                if in_pos not in self.fifo_itag_cumulative_count["RB"][direction]:
                    self.fifo_itag_cumulative_count["RB"][direction][in_pos] = 0
                self.fifo_itag_cumulative_count["RB"][direction][in_pos] += itag_count

    def increment_fifo_flit_count(self, category: str, fifo_type: str, pos: int, ip_type: str = None):
        """更新FIFO的flit计数

        Args:
            category: FIFO类别 ("IQ"/"RB"/"EQ")
            fifo_type: FIFO类型 ("CH_buffer"/"TR"/"TL"/"TU"/"TD"/"EQ")
            pos: 节点位置
            ip_type: IP类型（仅CH_buffer需要）
        """
        if fifo_type == "CH_buffer":
            if pos not in self.fifo_flit_count[category][fifo_type]:
                self.fifo_flit_count[category][fifo_type][pos] = {}
            if ip_type not in self.fifo_flit_count[category][fifo_type][pos]:
                self.fifo_flit_count[category][fifo_type][pos][ip_type] = 0
            self.fifo_flit_count[category][fifo_type][pos][ip_type] += 1
        else:
            if pos not in self.fifo_flit_count[category][fifo_type]:
                self.fifo_flit_count[category][fifo_type][pos] = 0
            self.fifo_flit_count[category][fifo_type][pos] += 1

    def get_links_utilization_stats(self):
        """
        获取链路利用率统计信息
        返回每个链路各状态的比例

        利用率计算说明：
        - 总slice-周期数 = 周期数 × 每个链路的slice数
        - 各状态比例 = 该状态的slice-周期数 / 总slice-周期数
        - 链路利用率 = (T2 + T1 + T0) / 总slice-周期数
        """
        stats = {}
        for link, link_stats in self.links_flow_stat.items():
            total_cycles = link_stats["total_cycles"]

            if total_cycles > 0:
                # 获取下环尝试次数统计
                eject_attempts_h = link_stats.get("eject_attempts_h", {"0": 0, "1": 0, "2": 0, ">2": 0})
                eject_attempts_v = link_stats.get("eject_attempts_v", {"0": 0, "1": 0, "2": 0, ">2": 0})

                # 计算flit总数
                total_flit_h = sum(eject_attempts_h.values())
                total_flit_v = sum(eject_attempts_v.values())
                total_flit = total_flit_h + total_flit_v

                # 计算合并的eject_attempts分布(横向+纵向)，基于total_cycles
                merged_attempts = {
                    "0": (eject_attempts_h["0"] + eject_attempts_v["0"]) / total_cycles,
                    "1": (eject_attempts_h["1"] + eject_attempts_v["1"]) / total_cycles,
                    "2": (eject_attempts_h["2"] + eject_attempts_v["2"]) / total_cycles,
                    ">2": (eject_attempts_h[">2"] + eject_attempts_v[">2"]) / total_cycles,
                }

                stats[link] = {
                    # 主要比例（基于total_cycles）
                    "utilization": total_flit / total_cycles,
                    "ITag_ratio": link_stats["ITag_count"] / total_cycles,
                    "empty_ratio": link_stats["empty_count"] / total_cycles,
                    # 详细flit分布（相对于total_cycles）
                    "eject_attempts_h_ratios": {k: v / total_cycles if total_cycles > 0 else 0.0 for k, v in eject_attempts_h.items()},
                    "eject_attempts_v_ratios": {k: v / total_cycles if total_cycles > 0 else 0.0 for k, v in eject_attempts_v.items()},
                    # 合并的eject_attempts分布（基于total_cycles）
                    "eject_attempts_merged_ratios": merged_attempts,
                    # 有效利用率（eject_attempts=0的flit占总周期的比例）
                    "effective_ratio": merged_attempts["0"],
                    # 原始计数
                    "total_cycles": total_cycles,
                    "total_flit": total_flit,
                    "eject_attempts_h": eject_attempts_h,
                    "eject_attempts_v": eject_attempts_v,
                }

        return stats
