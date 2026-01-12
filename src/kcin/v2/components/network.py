"""
Network class for NoC simulation.
Contains the core network implementation with routing and flow control mechanisms.
Enhanced with integrated route table support for flexible routing decisions.
"""

from __future__ import annotations
import numpy as np
from collections import deque, defaultdict
from typing import Optional, Dict, List, Any, Tuple
from src.kcin.base.config import KCINConfigBase
from src.utils.flit import Flit, TokenBucket
from src.utils.ring_slice import RingSlice
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
    def __init__(self, config: KCINConfigBase, adjacency_matrix, name="network"):
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
        # v2架构: 使用RingStation替代旧的inject_queues/eject_queues/channel_buffer
        self.arrive_node_pre = {}
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

        # 环形链表结构（新架构）
        self.horizontal_rings = {}  # {row: [RingSlice列表]}
        self.vertical_rings = {}    # {col: [RingSlice列表]}
        self.cp_in_slices = {}      # {(node_id, direction): RingSlice}
        self.cp_out_slices = {}     # {(node_id, direction): RingSlice}

        # 每个FIFO Entry的等待计数器
        self.fifo_counters = {"TL": {}, "TR": {}}

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
        # Circuit统计 - 延迟初始化
        self.circuits_flit_h = {}
        self.circuits_flit_v = {}
        self.gdma_recv = 0
        self.gdma_remainder = 0
        self.gdma_count = 512
        self.l2m_recv = 0
        self.l2m_remainder = 0
        self.sdma_send = []
        # 吞吐量统计 - 延迟初始化
        self.num_send = {}
        self.num_recv = {}
        self.per_send_throughput = {}
        self.per_recv_throughput = {}
        self.send_throughput = {}
        self.recv_throughput = {}
        self.last_select = {}
        self.throughput = {}

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

        # 反方向上环flit计数统计
        self.fifo_reverse_inject_count = {
            "IQ": {"TR": {}, "TL": {}},  # IQ_OUT 横向反方向上环
            "RB": {"TU": {}, "TD": {}},  # RB_OUT 纵向反方向上环
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

        # v2 FIFO统计 - 使用 RingStation 语义命名
        self.fifo_depth_sum_v2 = {
            "RS_IN_CH": {},   # {node_pos: {ip_type: sum}} 从IP接收
            "RS_OUT_CH": {},  # {node_pos: {ip_type: sum}} 输出到IP
            "RS_IN_DIR": {"TL": {}, "TR": {}, "TU": {}, "TD": {}},   # 从环接收
            "RS_OUT_DIR": {"TL": {}, "TR": {}, "TU": {}, "TD": {}},  # 输出到环
        }
        self.fifo_max_depth_v2 = {
            "RS_IN_CH": {},
            "RS_OUT_CH": {},
            "RS_IN_DIR": {"TL": {}, "TR": {}, "TU": {}, "TD": {}},
            "RS_OUT_DIR": {"TL": {}, "TR": {}, "TU": {}, "TD": {}},
        }
        self.fifo_flit_count_v2 = {
            "RS_IN_CH": {},
            "RS_OUT_CH": {},
            "RS_IN_DIR": {"TL": {}, "TR": {}, "TU": {}, "TD": {}},
            "RS_OUT_DIR": {"TL": {}, "TR": {}, "TU": {}, "TD": {}},
        }

        # Slot ID全局计数器（用于为每个seat分配唯一ID）
        self.global_slot_id_counter = 0

        # ETag setup (这些数据结构由Network管理，CrossPoint共享引用)
        # T0轮询机制：环slot列表、T0_table和仲裁指针
        self.horizontal_ring_slots = {}  # {node_id: [slot_id, ...]} 横向环的slot列表
        self.vertical_ring_slots = {}    # {node_id: [slot_id, ...]} 纵向环的slot列表
        self.T0_table_h = {}  # {node_id: set(slot_id, ...)} 横向环T0 flit记录表
        self.T0_table_v = {}  # {node_id: set(slot_id, ...)} 纵向环T0 flit记录表
        self.T0_arb_pointer_h = {}  # {node_id: index} 横向环仲裁指针
        self.T0_arb_pointer_v = {}  # {node_id: index} 纵向环仲裁指针
        # v2统一Entry管理：RS_UE_Counters 替代 RB_UE_Counters 和 EQ_UE_Counters
        self.RS_UE_Counters = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}  # 所有CrossPoint共享
        self.ETAG_BOTHSIDE_UPGRADE = False

        # 延迟释放Entry机制：存储待释放的Entry信息 {node_id: [(level, release_cycle), ...]}
        self.RS_pending_entry_release = {"TL": defaultdict(list), "TR": defaultdict(list), "TU": defaultdict(list), "TD": defaultdict(list)}

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

            # v2架构: inject_queues/eject_queues已合并到RingStation中

            # RS UE Counters (TU/TD)
            self.RS_UE_Counters["TU"][node_pos] = {"T2": 0, "T1": 0, "T0": 0}
            self.RS_UE_Counters["TD"][node_pos] = {"T2": 0, "T1": 0, "T0": 0}

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
                        "reverse_inject_h": 0,  # 横向反方向上环flit数
                        "reverse_inject_v": 0,  # 纵向反方向上环flit数
                    }
                    self.links_tag[(i, j)] = [LinkSlot(slot_id=self.global_slot_id_counter + idx) for idx in range(slice_count)]
                    self.global_slot_id_counter += slice_count

        # 注意：self-link已移除，边缘节点的方向转换由CP内部环回处理

        # CP slice tag初始化
        # 结构：cp_slices_tag[node_id][direction] = [LinkSlot, ...]
        self.cp_slices_tag = {}
        cp_slice_count = config.CP_SLICE_COUNT
        for node_id in range(config.NUM_NODE):
            self.cp_slices_tag[node_id] = {}
            for direction in ["TL", "TR", "TU", "TD"]:
                self.cp_slices_tag[node_id][direction] = [
                    LinkSlot(slot_id=self.global_slot_id_counter + idx)
                    for idx in range(cp_slice_count)
                ]
                self.global_slot_id_counter += cp_slice_count

        for pos in range(config.NUM_NODE):
            # v2架构: ring_bridge已合并到RingStation中

            # RS UE Counters (TL/TR)
            self.RS_UE_Counters["TL"][pos] = {"T2": 0, "T1": 0, "T0": 0}
            self.RS_UE_Counters["TR"][pos] = {"T2": 0, "T1": 0, "T0": 0}

        # 新架构: 所有节点都可以作为IP节点
        for ip_type in self.num_recv.keys():
            for source in range(config.NUM_NODE):
                destination = source  # 新架构: source和destination是同一节点
                self.num_send[ip_type][source] = 0
                self.num_recv[ip_type][destination] = 0
                self.per_send_throughput[ip_type][source] = 0
                self.per_recv_throughput[ip_type][destination] = 0

        # v2架构: IQ_channel_buffer/EQ_channel_buffer已合并到RingStation中
        for ip_type in self.last_select.keys():
            for ip_index in range(config.NUM_NODE):
                self.last_select[ip_type][ip_index] = "write"
        for ip_type in self.throughput.keys():
            for ip_index in range(config.NUM_NODE):
                self.throughput[ip_type][ip_index] = [0, 0, 10000000, 0]

        # 初始化RS_CAPACITY (所有CrossPoint共享)
        self.RS_CAPACITY = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}

        # 容量计算辅助函数
        t1_enabled = config.ETAG_T1_ENABLED

        def _cap_tl(lvl):
            if lvl == "T2":
                return config.TL_Etag_T2_UE_MAX
            if lvl == "T1":
                if not t1_enabled:
                    return 0
                return config.TL_Etag_T1_UE_MAX - config.TL_Etag_T2_UE_MAX
            if lvl == "T0":
                if not t1_enabled:
                    return config.RS_IN_FIFO_DEPTH - config.TL_Etag_T2_UE_MAX
                return config.RS_IN_FIFO_DEPTH - config.TL_Etag_T1_UE_MAX

        def _cap_tr(lvl):
            # 双侧下环保序模式：TR完全复用TL的配置
            if config.ORDERING_PRESERVATION_MODE == 2:
                return _cap_tl(lvl)
            # 其他模式：TR使用独立配置
            if lvl == "T2":
                return config.TR_Etag_T2_UE_MAX
            if lvl == "T1":
                if not t1_enabled:
                    return 0
                return config.RS_IN_FIFO_DEPTH - config.TR_Etag_T2_UE_MAX
            return 0

        def _cap_tu(lvl):
            if lvl == "T2":
                return config.TU_Etag_T2_UE_MAX
            if lvl == "T1":
                if not t1_enabled:
                    return 0
                return config.TU_Etag_T1_UE_MAX - config.TU_Etag_T2_UE_MAX
            if lvl == "T0":
                if not t1_enabled:
                    return config.RS_IN_FIFO_DEPTH - config.TU_Etag_T2_UE_MAX
                return config.RS_IN_FIFO_DEPTH - config.TU_Etag_T1_UE_MAX

        def _cap_td(lvl):
            # 双侧下环保序模式：TD完全复用TU的配置
            if config.ORDERING_PRESERVATION_MODE == 2:
                return _cap_tu(lvl)
            # 其他模式：TD使用独立配置
            if lvl == "T2":
                return config.TD_Etag_T2_UE_MAX
            if lvl == "T1":
                if not t1_enabled:
                    return 0
                return config.RS_IN_FIFO_DEPTH - config.TD_Etag_T2_UE_MAX
            return 0

        # 为所有节点设置RS_CAPACITY
        for pos in self.RS_UE_Counters["TL"]:
            self.RS_CAPACITY["TL"][pos] = {lvl: _cap_tl(lvl) for lvl in ("T0", "T1", "T2")}
        for pos in self.RS_UE_Counters["TR"]:
            self.RS_CAPACITY["TR"][pos] = {lvl: _cap_tr(lvl) for lvl in ("T0", "T1", "T2")}
        for pos in self.RS_UE_Counters["TU"]:
            self.RS_CAPACITY["TU"][pos] = {lvl: _cap_tu(lvl) for lvl in ("T0", "T1", "T2")}
        for pos in self.RS_UE_Counters["TD"]:
            self.RS_CAPACITY["TD"][pos] = {lvl: _cap_td(lvl) for lvl in ("T0", "T1", "T2")}

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

        # 创建RingStation对象 (v2架构核心组件)
        from .ring_station import RingStation

        self.ring_stations = {}
        for node_id in range(config.NUM_NODE):
            self.ring_stations[node_id] = RingStation(node_id, config, network_ref=self)

        # 构建环slot列表和初始化T0仲裁数据结构
        self._build_ring_slots()

        # 构建环形链表（新架构）
        self._build_ring_linked_lists()

    def initialize_buffers(self):
        """
        延迟初始化 - 在IP接口创建后调用

        根据config.CH_NAME_LIST动态创建统计结构
        v2架构: channel buffer已合并到RingStation中
        """
        from collections import defaultdict, deque

        # 1. 为每个IP类型创建统计结构(排除d2d)
        for ip_type in self.config.CH_NAME_LIST:
            if not ip_type.startswith("d2d"):
                base_type = ip_type.split('_')[0]
                if base_type not in ["d2d"]:
                    # 使用ip_type作为key而不是base_type
                    self.circuits_flit_h[ip_type] = defaultdict(lambda: deque(maxlen=self.config.RS_IN_CH_BUFFER))
                    self.circuits_flit_v[ip_type] = defaultdict(lambda: deque(maxlen=self.config.RS_IN_CH_BUFFER))
                    self.num_send[ip_type] = {ip: 0 for ip in range(self.config.NUM_NODE)}
                    self.num_recv[ip_type] = {ip: 0 for ip in range(self.config.NUM_NODE)}
                    self.per_send_throughput[ip_type] = {ip: 0 for ip in range(self.config.NUM_NODE)}
                    self.per_recv_throughput[ip_type] = {ip: 0 for ip in range(self.config.NUM_NODE)}
                    self.send_throughput[ip_type] = {ip: 0 for ip in range(self.config.NUM_NODE)}
                    self.recv_throughput[ip_type] = {ip: 0 for ip in range(self.config.NUM_NODE)}
                    self.last_select[ip_type] = {ip: None for ip in range(self.config.NUM_NODE)}
                    self.throughput[ip_type] = {ip: 0 for ip in range(self.config.NUM_NODE)}

                    # arrive_node_pre (不包含d2d)
                    if ip_type not in self.arrive_node_pre:
                        self.arrive_node_pre[ip_type] = defaultdict(lambda: deque(maxlen=self.config.RS_IN_CH_BUFFER))

        # 2. 为每个节点初始化arrive_node_pre
        for node_pos in range(self.config.NUM_NODE):
            for key in self.arrive_node_pre:
                self.arrive_node_pre[key][node_pos] = None

    # ------------------------------------------------------------------
    # RingStation 处理方法 (v2 架构核心)
    # ------------------------------------------------------------------

    def process_ring_stations(self, cycle: int):
        """
        处理所有 RingStation 的仲裁逻辑

        只执行仲裁 + 数据转移到 output_fifos_pre。
        pre → fifos 的移动统一在 _move_pre_to_queues() 中处理。

        Args:
            cycle: 当前周期
        """
        for node_id, rs in self.ring_stations.items():
            rs.process_cycle(cycle)

    def process_cp_slices(self, cycle: int):
        """
        处理所有 CP slice 的 flit 移动

        包括：
        1. CP slice 内部移动（从 slice_0 到 out_slice）
        2. 边界环回（边缘节点的 out_slice → 另一方向的 slice_0）
        3. 非边缘 CP 的输出（out_slice → 下游 Link[0]）

        Args:
            cycle: 当前周期
        """
        for node_id in range(self.config.NUM_NODE):
            for cp_type in ["horizontal", "vertical"]:
                cp = self.crosspoints[node_id][cp_type]
                for direction in cp.managed_directions:
                    # 1. 处理边界环回
                    if direction in cp._edge_loop_map:
                        self._process_cp_edge_loop(cp, direction, cycle)

                    # 2. CP slice 内部移动
                    self._move_cp_slices(cp, direction, cycle)

                    # 3. 非边缘 CP 输出到 Link
                    if direction not in cp._edge_loop_map:
                        self._transfer_cp_to_link(cp, node_id, direction, cycle)

    def _process_cp_edge_loop(self, cp, direction: str, cycle: int):
        """
        处理边缘节点的 CP 环回

        将当前方向的 out_slice 移动到另一方向的 slice_0

        Args:
            cp: CrossPoint 实例
            direction: 当前方向
            cycle: 当前周期
        """
        target_direction = cp._edge_loop_map.get(direction)
        if target_direction is None:
            return

        # 检查 out_slice 是否有 flit
        flit = cp.cp_slices[direction][-1]
        if flit is None:
            return

        # 检查目标方向的 slice_0 是否空闲
        if cp.cp_slices[target_direction][0] is not None:
            return

        # 执行环回
        cp.cp_slices[target_direction][0] = flit
        cp.cp_slices[direction][-1] = None
        flit.set_position("CP_LOOP", cycle)

    def _move_cp_slices(self, cp, direction: str, cycle: int):
        """
        CP slice 内部移动

        从后向前移动，避免覆盖

        Args:
            cp: CrossPoint 实例
            direction: 方向
            cycle: 当前周期
        """
        slices = cp.cp_slices[direction]
        # 从后向前移动（不包括 out_slice，因为它可能需要特殊处理）
        for i in range(len(slices) - 1, 0, -1):
            if slices[i] is None and slices[i - 1] is not None:
                slices[i] = slices[i - 1]
                slices[i - 1] = None

    def _transfer_cp_to_link(self, cp, node_id: int, direction: str, cycle: int):
        """
        将 CP 的 out_slice 传输到下游 Link

        Args:
            cp: CrossPoint 实例
            node_id: 节点 ID
            direction: 方向
            cycle: 当前周期
        """
        flit = cp.cp_slices[direction][-1]
        if flit is None:
            return

        # 计算下游 Link
        link = cp._calculate_inject_link(node_id, direction)
        if link is None:
            return  # 边缘节点，不应该走这个分支

        # 检查下游 Link[0] 是否空闲
        if self.links[link][0] is not None:
            return

        # 传输到下游 Link
        self.links[link][0] = flit
        cp.cp_slices[direction][-1] = None
        flit.current_link = link
        flit.current_seat_index = 0
        flit.set_position("Link", cycle)

    def rs_enqueue_from_local(self, node_id: int, flit) -> bool:
        """
        从本地 IP 注入 flit 到 RingStation

        Args:
            node_id: 节点 ID
            flit: 要注入的 flit

        Returns:
            bool: 是否成功入队
        """
        if node_id not in self.ring_stations:
            return False
        return self.ring_stations[node_id].enqueue_from_local(flit)

    def rs_enqueue_from_ring(self, node_id: int, flit, direction: str) -> bool:
        """
        从环上下环的 flit 进入 RingStation

        Args:
            node_id: 节点 ID
            flit: 下环的 flit
            direction: 来源方向 (TL/TR/TU/TD)

        Returns:
            bool: 是否成功入队
        """
        if node_id not in self.ring_stations:
            return False
        return self.ring_stations[node_id].enqueue_from_ring(flit, direction)

    def rs_dequeue_to_ring(self, node_id: int, direction: str):
        """
        从 RingStation 取出准备上环的 flit

        Args:
            node_id: 节点 ID
            direction: 目标方向 (TL/TR/TU/TD)

        Returns:
            Flit or None
        """
        if node_id not in self.ring_stations:
            return None
        return self.ring_stations[node_id].dequeue_to_ring(direction)

    def rs_dequeue_to_local(self, node_id: int):
        """
        从 RingStation 取出准备弹出到本地 IP 的 flit

        Args:
            node_id: 节点 ID

        Returns:
            Flit or None
        """
        if node_id not in self.ring_stations:
            return None
        return self.ring_stations[node_id].dequeue_to_local()

    def rs_can_accept_input(self, node_id: int, port: str) -> bool:
        """
        检查 RingStation 输入端口是否可以接受新 flit

        Args:
            node_id: 节点 ID
            port: 端口名 (ch_buffer/TL/TR/TU/TD)

        Returns:
            bool: 是否可以接受
        """
        if node_id not in self.ring_stations:
            return False
        return self.ring_stations[node_id].can_accept_input(port)

    def rs_has_output(self, node_id: int, port: str) -> bool:
        """
        检查 RingStation 输出端口是否有 flit

        Args:
            node_id: 节点 ID
            port: 端口名 (ch_buffer/TL/TR/TU/TD)

        Returns:
            bool: 是否有 flit
        """
        if node_id not in self.ring_stations:
            return False
        return self.ring_stations[node_id].has_output(port)

    def rs_peek_output(self, node_id: int, port: str):
        """
        查看 RingStation 输出端口队首的 flit（不移除）

        Args:
            node_id: 节点 ID
            port: 端口名 (ch_buffer/TL/TR/TU/TD)

        Returns:
            Flit or None
        """
        if node_id not in self.ring_stations:
            return None
        return self.ring_stations[node_id].peek_output(port)

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
        v2统一架构: 使用 RingStation.output_fifos 替代 inject_queues
        """
        rs = self.ring_stations[current]
        if flit.source == flit.destination or len(flit.path) <= 1:
            # 本地注入到对应IP类型的channel
            dest_type = getattr(flit, 'destination_type', None)
            if dest_type and dest_type in rs.output_fifos:
                return len(rs.output_fifos[dest_type]) < self.config.RS_OUT_CH_BUFFER
            return False

        # 纵向环移动（上下方向） - 本地注入到TU/TD
        if abs(current - next_node) == self.config.NUM_COL:
            # 判断向上还是向下
            if next_node > current:
                # 向下移动
                return len(rs.output_fifos["TD"]) < self.config.RS_OUT_FIFO_DEPTH
            else:
                # 向上移动
                return len(rs.output_fifos["TU"]) < self.config.RS_OUT_FIFO_DEPTH

        # 横向环移动（TR/TL）已由CrossPoint处理，此路径不应到达
        # 保留兼容性，返回False
        return False

    # ------------------------------------------------------------------
    # T0轮询机制：环slot列表构建
    # ------------------------------------------------------------------

    def _build_ring_slots(self):
        """构建每个节点所在环的slot列表，并初始化T0仲裁数据结构"""
        num_col = self.config.NUM_COL
        num_row = self.config.NUM_ROW

        # 横向环
        for row in range(num_row):
            row_nodes = [row * num_col + col for col in range(num_col)]
            slot_ids = self._collect_horizontal_ring_slots(row_nodes)
            for node in row_nodes:
                self.horizontal_ring_slots[node] = slot_ids
                self.T0_table_h[node] = set()
                self.T0_arb_pointer_h[node] = 0

        # 纵向环
        for col in range(num_col):
            col_nodes = [row * num_col + col for row in range(num_row)]
            slot_ids = self._collect_vertical_ring_slots(col_nodes)
            for node in col_nodes:
                self.vertical_ring_slots[node] = slot_ids
                self.T0_table_v[node] = set()
                self.T0_arb_pointer_v[node] = 0

    def _collect_horizontal_ring_slots(self, row_nodes):
        """
        收集横向环的slot_id列表（按流动顺序）

        横向环结构（以节点0,1,2,3为例）：
        TL方向: 3 -> 2 -> 1 -> 0 （边缘由CP内部环回）
        TR方向: 0 -> 1 -> 2 -> 3 （边缘由CP内部环回）

        链路顺序: (3,2) -> (2,1) -> (1,0) -> (0,1) -> (1,2) -> (2,3)
        注：self-link已移除，边缘节点的方向转换由CP内部环回处理
        """
        slot_ids = []

        # TL方向: 右->左
        for i in range(len(row_nodes) - 1, 0, -1):
            link = (row_nodes[i], row_nodes[i - 1])
            if link in self.links_tag:
                slot_ids.extend(slot.slot_id for slot in self.links_tag[link])

        # TR方向: 左->右
        for i in range(len(row_nodes) - 1):
            link = (row_nodes[i], row_nodes[i + 1])
            if link in self.links_tag:
                slot_ids.extend(slot.slot_id for slot in self.links_tag[link])

        return slot_ids

    def _collect_vertical_ring_slots(self, col_nodes):
        """
        收集纵向环的slot_id列表（按流动顺序）

        纵向环结构（以节点0,4,8,12为例）：
        TU方向: 12 -> 8 -> 4 -> 0 （边缘由CP内部环回）
        TD方向: 0 -> 4 -> 8 -> 12 （边缘由CP内部环回）

        链路顺序: (12,8) -> (8,4) -> (4,0) -> (0,4) -> (4,8) -> (8,12)
        注：self-link已移除，边缘节点的方向转换由CP内部环回处理
        """
        slot_ids = []

        # TU方向: 下->上
        for i in range(len(col_nodes) - 1, 0, -1):
            link = (col_nodes[i], col_nodes[i - 1])
            if link in self.links_tag:
                slot_ids.extend(slot.slot_id for slot in self.links_tag[link])

        # TD方向: 上->下
        for i in range(len(col_nodes) - 1):
            link = (col_nodes[i], col_nodes[i + 1])
            if link in self.links_tag:
                slot_ids.extend(slot.slot_id for slot in self.links_tag[link])

        return slot_ids

    # ------------------------------------------------------------------
    # 环形链表构建（新架构）
    # ------------------------------------------------------------------

    def _build_ring_linked_lists(self):
        """构建所有环形链表"""
        num_col = self.config.NUM_COL
        num_row = self.config.NUM_ROW

        # 构建横向环
        for row in range(num_row):
            self.horizontal_rings[row] = self._build_horizontal_ring(row)

        # 构建纵向环
        for col in range(num_col):
            self.vertical_rings[col] = self._build_vertical_ring(col)

    def _build_horizontal_ring(self, row: int) -> list:
        """构建一行的横向环形链表"""
        num_col = self.config.NUM_COL
        nodes = [row * num_col + col for col in range(num_col)]
        all_slices = []

        cp_slice_count = self.config.CP_SLICE_COUNT
        cp_out_is_link = (cp_slice_count == 1)
        cp_internal_count = max(0, cp_slice_count - 2)
        link_slice_count = self.config.SLICE_PER_LINK_HORIZONTAL

        # === TR方向：从左到右 ===
        for i, node in enumerate(nodes):
            for _ in range(cp_internal_count):
                all_slices.append(RingSlice(RingSlice.CP_INTERNAL, node, "TR"))

            in_slice = RingSlice(RingSlice.CP_IN, node, "TR")
            all_slices.append(in_slice)
            self.cp_in_slices[(node, "TR")] = in_slice

            if i < len(nodes) - 1:
                if cp_out_is_link:
                    out_slice = RingSlice(RingSlice.LINK, node, "TR")
                    out_slice.is_cp_out = True
                    out_slice.link_index = 0
                    all_slices.append(out_slice)
                    self.cp_out_slices[(node, "TR")] = out_slice
                    for link_idx in range(1, link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TR")
                        s.link_index = link_idx
                        all_slices.append(s)
                else:
                    out_slice = RingSlice(RingSlice.CP_OUT, node, "TR")
                    all_slices.append(out_slice)
                    self.cp_out_slices[(node, "TR")] = out_slice
                    for link_idx in range(link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TR")
                        s.link_index = link_idx
                        all_slices.append(s)

        tl_start_index = len(all_slices)

        # === TL方向：从右到左 ===
        for i, node in enumerate(reversed(nodes)):
            for _ in range(cp_internal_count):
                all_slices.append(RingSlice(RingSlice.CP_INTERNAL, node, "TL"))

            in_slice = RingSlice(RingSlice.CP_IN, node, "TL")
            all_slices.append(in_slice)
            self.cp_in_slices[(node, "TL")] = in_slice

            if i < len(nodes) - 1:
                if cp_out_is_link:
                    out_slice = RingSlice(RingSlice.LINK, node, "TL")
                    out_slice.is_cp_out = True
                    out_slice.link_index = 0
                    all_slices.append(out_slice)
                    self.cp_out_slices[(node, "TL")] = out_slice
                    for link_idx in range(1, link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TL")
                        s.link_index = link_idx
                        all_slices.append(s)
                else:
                    out_slice = RingSlice(RingSlice.CP_OUT, node, "TL")
                    all_slices.append(out_slice)
                    self.cp_out_slices[(node, "TL")] = out_slice
                    for link_idx in range(link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TL")
                        s.link_index = link_idx
                        all_slices.append(s)

        for i in range(len(all_slices)):
            all_slices[i].next = all_slices[(i + 1) % len(all_slices)]

        right_edge_node = nodes[-1]
        left_edge_node = nodes[0]
        self.cp_out_slices[(right_edge_node, "TR")] = all_slices[tl_start_index]
        self.cp_out_slices[(left_edge_node, "TL")] = all_slices[0]

        for i, s in enumerate(all_slices):
            s.slot_id = self.global_slot_id_counter
            self.global_slot_id_counter += 1

        return all_slices

    def _build_vertical_ring(self, col: int) -> list:
        """构建一列的纵向环形链表"""
        num_col = self.config.NUM_COL
        num_row = self.config.NUM_ROW
        nodes = [row * num_col + col for row in range(num_row)]
        all_slices = []

        cp_slice_count = self.config.CP_SLICE_COUNT
        cp_out_is_link = (cp_slice_count == 1)
        cp_internal_count = max(0, cp_slice_count - 2)
        link_slice_count = self.config.SLICE_PER_LINK_VERTICAL

        # === TD方向：从上到下 ===
        for i, node in enumerate(nodes):
            for _ in range(cp_internal_count):
                all_slices.append(RingSlice(RingSlice.CP_INTERNAL, node, "TD"))

            in_slice = RingSlice(RingSlice.CP_IN, node, "TD")
            all_slices.append(in_slice)
            self.cp_in_slices[(node, "TD")] = in_slice

            if i < len(nodes) - 1:
                if cp_out_is_link:
                    out_slice = RingSlice(RingSlice.LINK, node, "TD")
                    out_slice.is_cp_out = True
                    out_slice.link_index = 0
                    all_slices.append(out_slice)
                    self.cp_out_slices[(node, "TD")] = out_slice
                    for link_idx in range(1, link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TD")
                        s.link_index = link_idx
                        all_slices.append(s)
                else:
                    out_slice = RingSlice(RingSlice.CP_OUT, node, "TD")
                    all_slices.append(out_slice)
                    self.cp_out_slices[(node, "TD")] = out_slice
                    for link_idx in range(link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TD")
                        s.link_index = link_idx
                        all_slices.append(s)

        tu_start_index = len(all_slices)

        # === TU方向：从下到上 ===
        for i, node in enumerate(reversed(nodes)):
            for _ in range(cp_internal_count):
                all_slices.append(RingSlice(RingSlice.CP_INTERNAL, node, "TU"))

            in_slice = RingSlice(RingSlice.CP_IN, node, "TU")
            all_slices.append(in_slice)
            self.cp_in_slices[(node, "TU")] = in_slice

            if i < len(nodes) - 1:
                if cp_out_is_link:
                    out_slice = RingSlice(RingSlice.LINK, node, "TU")
                    out_slice.is_cp_out = True
                    out_slice.link_index = 0
                    all_slices.append(out_slice)
                    self.cp_out_slices[(node, "TU")] = out_slice
                    for link_idx in range(1, link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TU")
                        s.link_index = link_idx
                        all_slices.append(s)
                else:
                    out_slice = RingSlice(RingSlice.CP_OUT, node, "TU")
                    all_slices.append(out_slice)
                    self.cp_out_slices[(node, "TU")] = out_slice
                    for link_idx in range(link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TU")
                        s.link_index = link_idx
                        all_slices.append(s)

        for i in range(len(all_slices)):
            all_slices[i].next = all_slices[(i + 1) % len(all_slices)]

        bottom_edge_node = nodes[-1]
        top_edge_node = nodes[0]
        self.cp_out_slices[(bottom_edge_node, "TD")] = all_slices[tu_start_index]
        self.cp_out_slices[(top_edge_node, "TU")] = all_slices[0]

        for i, s in enumerate(all_slices):
            s.slot_id = self.global_slot_id_counter
            self.global_slot_id_counter += 1

        return all_slices

    def update_excess_ITag(self):
        """在主循环中调用，处理多余ITag释放"""
        # 调用所有CrossPoint的update_excess_ITag方法
        for ip_pos in self.crosspoints:
            self.crosspoints[ip_pos]["horizontal"].update_excess_ITag()
            self.crosspoints[ip_pos]["vertical"].update_excess_ITag()

    # ------------------------------------------------------------------
    # 环形链表处理（新架构）
    # ------------------------------------------------------------------

    def process_ring_movement(self, ring_slices: list, cycle: int):
        """统一处理一个环上所有flit的移动"""
        for i in range(len(ring_slices) - 1, -1, -1):
            s = ring_slices[i]
            flit = s.flit

            if flit is None:
                continue

            next_slice = s.next

            if s.slice_type == RingSlice.CP_IN:
                should_eject, eject_target, eject_dir = self._should_eject(flit, s.node_id, s.direction)
                if should_eject:
                    if self._try_eject(flit, s.node_id, eject_dir, eject_target, cycle):
                        s.flit = None
                        continue

            if next_slice.flit is None:
                next_slice.flit = flit
                s.flit = None
                self._update_flit_position(flit, next_slice, cycle)

    def process_ring_injection(self, ring_slices: list, cycle: int):
        """处理上环注入"""
        for s in ring_slices:
            if not s.is_inject_point():
                continue
            if s.flit is not None:
                continue

            flit = self._get_pending_inject_flit(s.node_id, s.direction)
            if flit and self._can_inject(flit, s, cycle):
                self._execute_inject(flit, s, cycle)

    def process_all_rings(self, cycle: int):
        """处理所有环的移动和注入"""
        for row, ring in self.horizontal_rings.items():
            self.process_ring_movement(ring, cycle)
            self.process_ring_injection(ring, cycle)

        for col, ring in self.vertical_rings.items():
            self.process_ring_movement(ring, cycle)
            self.process_ring_injection(ring, cycle)

        # 同步RingSlice数据到links字典（用于可视化）
        self._sync_links_from_ring_slices()

    def _sync_links_from_ring_slices(self):
        """从RingSlice同步数据到links字典和cp_slices（用于可视化兼容）"""
        # 清空所有links
        for link_key in self.links:
            for i in range(len(self.links[link_key])):
                self.links[link_key][i] = None

        # 清空所有crosspoints的cp_slices
        for node_id in self.crosspoints:
            for cp_type in ["horizontal", "vertical"]:
                cp = self.crosspoints[node_id][cp_type]
                for direction in cp.cp_slices:
                    for i in range(len(cp.cp_slices[direction])):
                        cp.cp_slices[direction][i] = None

        # 从横向环同步
        for row, ring in self.horizontal_rings.items():
            for s in ring:
                node_id = s.node_id
                direction = s.direction
                flit = s.flit

                if s.slice_type == RingSlice.LINK and flit is not None:
                    # 同步到links
                    if direction == "TR":
                        next_node = node_id + 1
                    else:  # TL
                        next_node = node_id - 1
                    link_key = (node_id, next_node)
                    if link_key in self.links:
                        link_idx = s.link_index
                        if link_idx < len(self.links[link_key]):
                            self.links[link_key][link_idx] = flit

                elif s.slice_type in [RingSlice.CP_IN, RingSlice.CP_OUT, RingSlice.CP_INTERNAL] and flit is not None:
                    # 同步到cp_slices
                    cp = self.crosspoints[node_id]["horizontal"]
                    # 计算slice索引：CP_INTERNAL在前，CP_IN在中间，CP_OUT在后
                    cp_slices = cp.cp_slices.get(direction, [])
                    if s.slice_type == RingSlice.CP_IN:
                        # CP_IN 在倒数第二个位置
                        idx = len(cp_slices) - 2 if len(cp_slices) >= 2 else 0
                    elif s.slice_type == RingSlice.CP_OUT:
                        # CP_OUT 在最后一个位置
                        idx = len(cp_slices) - 1 if len(cp_slices) >= 1 else 0
                    else:
                        # CP_INTERNAL 在开头
                        idx = 0
                    if 0 <= idx < len(cp_slices):
                        cp_slices[idx] = flit

        # 从纵向环同步
        for col, ring in self.vertical_rings.items():
            for s in ring:
                node_id = s.node_id
                direction = s.direction
                flit = s.flit
                num_col = self.config.NUM_COL

                if s.slice_type == RingSlice.LINK and flit is not None:
                    # 同步到links
                    if direction == "TD":
                        next_node = node_id + num_col
                    else:  # TU
                        next_node = node_id - num_col
                    link_key = (node_id, next_node)
                    if link_key in self.links:
                        link_idx = s.link_index
                        if link_idx < len(self.links[link_key]):
                            self.links[link_key][link_idx] = flit

                elif s.slice_type in [RingSlice.CP_IN, RingSlice.CP_OUT, RingSlice.CP_INTERNAL] and flit is not None:
                    # 同步到cp_slices
                    cp = self.crosspoints[node_id]["vertical"]
                    cp_slices = cp.cp_slices.get(direction, [])
                    if s.slice_type == RingSlice.CP_IN:
                        idx = len(cp_slices) - 2 if len(cp_slices) >= 2 else 0
                    elif s.slice_type == RingSlice.CP_OUT:
                        idx = len(cp_slices) - 1 if len(cp_slices) >= 1 else 0
                    else:
                        idx = 0
                    if 0 <= idx < len(cp_slices):
                        cp_slices[idx] = flit

    def _should_eject(self, flit, node_id: int, direction: str) -> tuple:
        """
        判断flit是否需要在当前节点下环（简化版本，不依赖flit.current_link）

        Returns:
            tuple: (是否下环, 下环目标, 下环方向)
        """
        is_horizontal = direction in ["TL", "TR"]
        cp_type = "horizontal" if is_horizontal else "vertical"
        cp = self.crosspoints[node_id][cp_type]

        # 获取flit的最终目标和路径中的下一跳
        try:
            path = flit.path
            final_dest = path[-1]
            current_idx = path.index(node_id)
            next_node = path[current_idx + 1] if current_idx + 1 < len(path) else final_dest
        except (ValueError, IndexError):
            # 当前节点不在路径中或路径异常
            if node_id == flit.path[-1]:
                # 到达最终目标，需要下环
                return True, "RB" if is_horizontal else "EQ", direction
            return False, "", ""

        # 判断是否需要下环
        if is_horizontal:
            # 水平CP：检查是否需要转到纵向环
            if cp._needs_vertical_move(node_id, next_node):
                return True, "RB", direction
        else:
            # 纵向CP：检查是否到达最终目标
            if node_id == final_dest:
                return True, "EQ", direction

        return False, "", ""

    def _try_eject(self, flit, node_id: int, direction: str, eject_target: str, cycle: int) -> bool:
        """尝试下环到RingStation"""
        # 获取RingStation
        rs = self.ring_stations.get(node_id)
        if rs is None:
            return False

        # 检查RingStation输入FIFO是否有空间
        input_fifo = rs.input_fifos.get(direction)
        if input_fifo is None or len(input_fifo) >= input_fifo.maxlen:
            return False

        # 放入RingStation的输入FIFO
        input_fifo.append(flit)
        flit.set_position(f"RS_IN_{direction}", cycle)
        return True

    def _get_pending_inject_flit(self, node_id: int, direction: str):
        """获取等待上环的flit（从RingStation的输出FIFO）"""
        rs = self.ring_stations.get(node_id)
        if rs is None:
            return None

        output_fifo = rs.output_fifos.get(direction)
        if output_fifo and len(output_fifo) > 0:
            return output_fifo[0]
        return None

    def _can_inject(self, flit, ring_slice, cycle: int) -> bool:
        """检查是否可以上环"""
        if ring_slice.flit is not None:
            return False

        node_id = ring_slice.node_id
        direction = ring_slice.direction

        cp_type = "horizontal" if direction in ["TL", "TR"] else "vertical"
        cp = self.crosspoints[node_id][cp_type]

        # 构造虚拟link用于ITag检查
        num_col = self.config.NUM_COL
        if direction == "TR":
            next_node = node_id + 1 if (node_id + 1) % num_col != 0 else node_id
        elif direction == "TL":
            next_node = node_id - 1 if node_id % num_col != 0 else node_id
        elif direction == "TD":
            next_node = node_id + num_col if node_id + num_col < self.config.NUM_NODE else node_id
        else:  # TU
            next_node = node_id - num_col if node_id >= num_col else node_id

        if next_node == node_id:
            link = None
        else:
            link = (node_id, next_node)

        return cp._can_inject_to_link(flit, link, direction, cycle)

    def _execute_inject(self, flit, ring_slice, cycle: int):
        """执行上环操作"""
        node_id = ring_slice.node_id
        direction = ring_slice.direction
        is_horizontal = direction in ["TL", "TR"]

        # 1. 从RingStation输出FIFO弹出
        rs = self.ring_stations.get(node_id)
        if rs:
            output_fifo = rs.output_fifos.get(direction)
            if output_fifo and len(output_fifo) > 0 and output_fifo[0] is flit:
                output_fifo.popleft()

        # 2. 放入RingSlice
        ring_slice.flit = flit

        # 3. 统计更新
        if is_horizontal:
            self.inject_num += 1

        # 4. 首次上环分配order_id
        if flit.src_dest_order_id == -1:
            flit.src_dest_order_id = Flit.get_next_order_id(
                flit.source, flit.source_type,
                flit.destination, flit.destination_type,
                flit.flit_type.upper(),
                self.config.ORDERING_GRANULARITY,
                getattr(self.config, "DIE_ID", None)
            )

        # 5. 纵向上环更新位置
        if not is_horizontal:
            flit.current_position = node_id
            flit.path_index += 1

        # 6. 处理ITag释放
        cp_type = "horizontal" if is_horizontal else "vertical"
        cp = self.crosspoints[node_id][cp_type]

        num_col = self.config.NUM_COL
        if direction == "TR":
            next_node = node_id + 1 if (node_id + 1) % num_col != 0 else node_id
        elif direction == "TL":
            next_node = node_id - 1 if node_id % num_col != 0 else node_id
        elif direction == "TD":
            next_node = node_id + num_col if node_id + num_col < self.config.NUM_NODE else node_id
        else:  # TU
            next_node = node_id - num_col if node_id >= num_col else node_id

        if next_node != node_id:
            link = (node_id, next_node)
            slot = self.links_tag.get(link, [None])[0]
            if slot and slot.itag_reserved and slot.check_itag_match(node_id, direction):
                slot.clear_itag()
                cp.remain_tag[direction][node_id] += 1
                cp.tagged_counter[direction][node_id] -= 1

        # 清除I-Tag标记
        if is_horizontal and hasattr(flit, 'itag_h') and flit.itag_h:
            flit.itag_h = False
        elif not is_horizontal and hasattr(flit, 'itag_v') and flit.itag_v:
            flit.itag_v = False

        flit.ETag_priority = "T2"

        # 7. 更新位置
        self._update_flit_position(flit, ring_slice, cycle)

    def _update_flit_position(self, flit, ring_slice, cycle: int):
        """更新flit位置显示（与v1格式一致）"""
        from src.utils.ring_slice import RingSlice

        node_id = ring_slice.node_id
        direction = ring_slice.direction
        slice_type = ring_slice.slice_type

        if slice_type == RingSlice.LINK:
            # Link格式：node_id->next_node:link_index
            num_col = self.config.NUM_COL
            if direction == "TR":
                next_node = node_id + 1
            elif direction == "TL":
                next_node = node_id - 1
            elif direction == "TD":
                next_node = node_id + num_col
            else:  # TU
                next_node = node_id - num_col
            link_idx = ring_slice.link_index
            pos = f"{node_id}->{next_node}:{link_idx}"
        elif slice_type in [RingSlice.CP_IN, RingSlice.CP_OUT, RingSlice.CP_INTERNAL]:
            # CP格式：CP_H 或 CP_V
            if direction in ["TL", "TR"]:
                pos = "CP_H"
            else:
                pos = "CP_V"
        else:
            pos = ring_slice.get_position_str()

        flit.set_position(pos, cycle)

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
            # 边缘节点无下游Link，由CP内部环回处理
            if right_pos is not None:
                arrival_link = (left_pos, ip_pos) if left_pos is not None else None
                departure_link = (ip_pos, right_pos)
            else:
                # 右边界: 无下游Link
                arrival_link = (left_pos, ip_pos) if left_pos is not None else None
                departure_link = None

            arrival_slice = self.links[arrival_link][-1] if arrival_link and arrival_link in self.links else None
            departure_slice = self.links[departure_link][0] if departure_link and departure_link in self.links else None
            self.cross_point["horizontal"][ip_pos]["TR"] = [arrival_slice, departure_slice]

            # 水平CrossPoint连接 - TL方向 (向左)
            if left_pos is not None:
                arrival_link = (right_pos, ip_pos) if right_pos is not None else None
                departure_link = (ip_pos, left_pos)
            else:
                # 左边界: 无下游Link
                arrival_link = (right_pos, ip_pos) if right_pos is not None else None
                departure_link = None

            arrival_slice = self.links[arrival_link][-1] if arrival_link and arrival_link in self.links else None
            departure_slice = self.links[departure_link][0] if departure_link and departure_link in self.links else None
            self.cross_point["horizontal"][ip_pos]["TL"] = [arrival_slice, departure_slice]

            # 垂直CrossPoint连接 - TU方向 (向上)
            if up_pos is not None:
                arrival_link = (down_pos, ip_pos) if down_pos is not None else None
                departure_link = (ip_pos, up_pos)
            else:
                # 上边界: 无下游Link
                arrival_link = (down_pos, ip_pos) if down_pos is not None else None
                departure_link = None

            arrival_slice = self.links[arrival_link][-1] if arrival_link and arrival_link in self.links else None
            departure_slice = self.links[departure_link][0] if departure_link and departure_link in self.links else None
            self.cross_point["vertical"][ip_pos]["TU"] = [arrival_slice, departure_slice]

            # 垂直CrossPoint连接 - TD方向 (向下)
            if down_pos is not None:
                arrival_link = (up_pos, ip_pos) if up_pos is not None else None
                departure_link = (ip_pos, down_pos)
            else:
                # 下边界: 无下游Link
                arrival_link = (up_pos, ip_pos) if up_pos is not None else None
                departure_link = None

            arrival_slice = self.links[arrival_link][-1] if arrival_link and arrival_link in self.links else None
            departure_slice = self.links[departure_link][0] if departure_link and departure_link in self.links else None
            self.cross_point["vertical"][ip_pos]["TD"] = [arrival_slice, departure_slice]

            # 更新CrossPoint冲突状态 (基于arrival slice是否有flit)
            # 水平冲突
            tr_arrival_link = (left_pos, ip_pos) if left_pos is not None else None
            tl_arrival_link = (right_pos, ip_pos) if right_pos is not None else None
            new_tr_conflict = self.links[tr_arrival_link][-1] is not None if tr_arrival_link and tr_arrival_link in self.links else False
            new_tl_conflict = self.links[tl_arrival_link][-1] is not None if tl_arrival_link and tl_arrival_link in self.links else False

            self.crosspoint_conflict["horizontal"][ip_pos]["TR"].insert(0, new_tr_conflict)
            self.crosspoint_conflict["horizontal"][ip_pos]["TR"] = self.crosspoint_conflict["horizontal"][ip_pos]["TR"][:2]
            self.crosspoint_conflict["horizontal"][ip_pos]["TL"].insert(0, new_tl_conflict)
            self.crosspoint_conflict["horizontal"][ip_pos]["TL"] = self.crosspoint_conflict["horizontal"][ip_pos]["TL"][:2]

            # 垂直冲突
            tu_arrival_link = (down_pos, ip_pos) if down_pos is not None else None
            td_arrival_link = (up_pos, ip_pos) if up_pos is not None else None
            new_tu_conflict = self.links[tu_arrival_link][-1] is not None if tu_arrival_link and tu_arrival_link in self.links else False
            new_td_conflict = self.links[td_arrival_link][-1] is not None if td_arrival_link and td_arrival_link in self.links else False

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

        设计说明：
            每个Die的保序是独立的，只关心Die内部的传输对。
            key使用flit.source和flit.destination，而不是跨Die的原始源/目标。
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

        # 使用flit当前Die内的实际source/destination构建key
        src = flit.source
        dest = flit.destination
        die_id = getattr(self.config, "DIE_ID", None)

        # 根据保序粒度构造key（包含die_id确保不同Die独立保序）
        if self.config.ORDERING_GRANULARITY == 0:  # IP层级
            src_type = flit.source_type
            dest_type = flit.destination_type
            key = (die_id, src, src_type, dest, dest_type, direction)
        else:  # 节点层级（granularity == 1）
            key = (die_id, src, dest, direction)

        # 检查是否是期望的下一个顺序ID
        expected_order_id = self.order_tracking_table[key] + 1

        can_eject = flit.src_dest_order_id == expected_order_id

        return can_eject

    def _need_in_order_check(self, flit: Flit):
        """判断该flit是否需要保序检查

        设计说明：
            使用flit的实际source/destination进行判断，与保序key构建保持一致。
        """
        if self.config.ORDERING_PRESERVATION_MODE == 0:
            return False

        # 检查通道类型是否需要保序
        packet_category = self._get_flit_packet_category(flit)
        if hasattr(self.config, "IN_ORDER_PACKET_CATEGORIES"):
            if packet_category not in self.config.IN_ORDER_PACKET_CATEGORIES:
                return False

        # 使用flit的实际source/destination
        src = flit.source
        dest = flit.destination

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

        设计说明：
            每个Die的保序是独立的，只关心Die内部的传输对。
            key使用flit.source和flit.destination，而不是跨Die的原始源/目标。
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

        # 使用flit当前Die内的实际source/destination构建key
        src = flit.source
        dest = flit.destination

        # 使用 (src, dest, direction) 作为键（节点层级）
        key = (src, dest, direction)

        # 检查是否是期望的下一个顺序ID
        expected_order_id = self.order_tracking_table[key] + 1
        return flit.src_dest_order_id == expected_order_id

    def _update_order_tracking_table(self, flit: Flit, target_node: int, direction: str):
        """更新保序跟踪表

        Args:
            target_node: 目标节点的position（映射后的位置），可能是中间节点

        设计说明：
            每个Die的保序是独立的，只关心Die内部的传输对。
            key使用flit.source和flit.destination，而不是跨Die的原始源/目标。
        """
        # 先判断是否需要保序
        if not self._need_in_order_check(flit):
            return

        # 确保flit已设置保序信息
        if not hasattr(flit, "src_dest_order_id") or not hasattr(flit, "packet_category"):
            return

        if flit.src_dest_order_id == -1:
            return

        # 使用flit当前Die内的实际source/destination构建key
        src = flit.source
        dest = flit.destination
        die_id = getattr(self.config, "DIE_ID", None)

        # 根据保序粒度构造key（与_can_eject_in_order保持一致，包含die_id）
        if self.config.ORDERING_GRANULARITY == 0:  # IP层级
            src_type = flit.source_type
            dest_type = flit.destination_type
            key = (die_id, src, src_type, dest, dest_type, direction)
        else:  # 节点层级（granularity == 1）
            key = (die_id, src, dest, direction)

        # 更新保序跟踪表
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
        """确定flit允许的下环方向

        设计说明：
            Mode 0: 不保序，所有方向都允许
            Mode 1: 单侧下环，固定只允许TL和TU方向
            Mode 2: 双侧下环，基于源节点白名单配置
            Mode 3: 动态方向，基于(src, dest)相对位置计算
        """
        mode = self.config.ORDERING_PRESERVATION_MODE

        # Mode 0: 不保序，所有方向都允许
        if mode == 0:
            return None

        # Mode 1: 单侧下环，固定只允许TL和TU方向
        if mode == 1:
            return ["TL", "TU"]

        # Mode 2: 双侧下环，根据方向配置决定
        if mode == 2:
            # 使用flit的实际source节点（本Die内的源节点）
            src_node = flit.source

            # 检查各方向是否允许
            allowed_dirs = []
            for direction in ["TL", "TR", "TU", "TD"]:
                # 空列表表示所有节点都允许
                if len(self.allowed_source_nodes[direction]) == 0 or src_node in self.allowed_source_nodes[direction]:
                    allowed_dirs.append(direction)

            return allowed_dirs if allowed_dirs else None

        # Mode 3: 动态方向，基于(src, dest)相对位置计算
        if mode == 3:
            return self._calculate_dynamic_eject_directions(flit)

        # 未知模式，默认不保序
        return None

    def _calculate_dynamic_eject_directions(self, flit: Flit):
        """基于(src, dest)相对位置计算允许的下环方向

        横向：dest在src左边 → TL，dest在src右边 → TR
        纵向：dest在src上方 → TU，dest在src下方 → TD
        同行/同列时不分配该维度的方向（由EQ直接处理）
        """
        src_node = flit.source
        dest_node = flit.destination

        num_cols = self.config.NUM_COL

        src_row = src_node // num_cols
        src_col = src_node % num_cols
        dest_row = dest_node // num_cols
        dest_col = dest_node % num_cols

        allowed_dirs = []

        # 横向方向判断
        if dest_col < src_col:
            allowed_dirs.append("TL")  # 目标在左边，允许向左下环
        elif dest_col > src_col:
            allowed_dirs.append("TR")  # 目标在右边，允许向右下环
        # 同列时不需要横向下环，EQ直接处理

        # 纵向方向判断
        if dest_row < src_row:
            allowed_dirs.append("TU")  # 目标在上方，允许向上下环
        elif dest_row > src_row:
            allowed_dirs.append("TD")  # 目标在下方，允许向下下环
        # 同行时不需要纵向下环，EQ直接处理

        return allowed_dirs if allowed_dirs else None

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
            # 边缘情况：进入CP的slice进行环回
            # 确定方向
            u, v = flit.current_link[:2] if len(flit.current_link) >= 2 else (current_node, current_node)
            if link_type == "h":
                cp = self.crosspoints[current_node]["horizontal"]
                # 从左到右是TR，从右到左是TL
                direction = "TR" if u < v else "TL"
            else:
                cp = self.crosspoints[current_node]["vertical"]
                # 从上到下是TD，从下到上是TU
                direction = "TD" if u < v else "TU"

            # 将flit放入CP的slice_0
            if cp.cp_slices[direction][0] is None:
                cp.cp_slices[direction][0] = flit
                flit.current_link = None  # 在CP内部，不在Link上
                flit.current_seat_index = -1
                flit.set_position("CP_EDGE", self.cycle)
            else:
                # CP入口被占用，flit需要等待（放回原位置）
                # 这种情况不应该发生，因为flit已经从link中移除
                raise RuntimeError(f"[Cycle {self.cycle}] CP slice[0] occupied at node {current_node} direction {direction}")
            return

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
                        if upgrade_to == "T0" and eject_direction in ["TL", "TU"]:
                            crosspoint.T0_table_record(flit, eject_direction)

                state = self._analyze_flit_state(flit, current, next_node)
                self._continue_looping(flit, link, state["next_pos"])
                return

            # 尝试下环（v2统一架构：使用RingStation）
            success, fail_reason = crosspoint._try_eject(flit, eject_direction, final_destination, link)

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
                if upgrade_to == "T0" and eject_direction in ["TL", "TU"]:
                    crosspoint.T0_table_record(flit, eject_direction)

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
            # 统计横向反方向上环
            if hasattr(flit, "reverse_inject_h") and flit.reverse_inject_h > 0:
                self.links_flow_stat[link]["reverse_inject_h"] += 1
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
            # 统计纵向反方向上环
            if hasattr(flit, "reverse_inject_v") and flit.reverse_inject_v > 0:
                self.links_flow_stat[link]["reverse_inject_v"] += 1

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
        """在move操作后批量更新该节点所有FIFO统计

        v2架构使用RingStation，FIFO使用StatisticalFIFO自动记录统计
        """
        rs = self.ring_stations.get(in_pos)
        if rs is None:
            return

        # 采样所有 RingStation 的 FIFO
        # 输入端 FIFO (从IP接收 + 从环接收)
        for fifo in rs.input_fifos.values():
            fifo.sample()

        # 输出端 FIFO (输出到IP + 输出到环)
        for fifo in rs.output_fifos.values():
            fifo.sample()

        # === ITag累计统计 ===
        # IQ_OUT统计 (只统计TR/TL横向注入)
        for direction in ["TR", "TL"]:
            itag_count = sum(1 for flit in rs.output_fifos[direction] if getattr(flit, "itag_h", False))
            if in_pos not in self.fifo_itag_cumulative_count["IQ"][direction]:
                self.fifo_itag_cumulative_count["IQ"][direction][in_pos] = 0
            self.fifo_itag_cumulative_count["IQ"][direction][in_pos] += itag_count

        # RB_OUT统计 (TU/TD纵向转向)
        for direction in ["TU", "TD"]:
            itag_count = sum(1 for flit in rs.output_fifos[direction] if getattr(flit, "itag_v", False))
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

    def increment_fifo_reverse_inject_count(self, category: str, fifo_type: str, pos: int):
        """更新FIFO的反方向上环flit计数

        Args:
            category: FIFO类别 ("IQ"/"RB")
            fifo_type: FIFO类型 ("TR"/"TL"/"TU"/"TD")
            pos: 节点位置
        """
        if category not in self.fifo_reverse_inject_count:
            return
        if fifo_type not in self.fifo_reverse_inject_count[category]:
            return
        if pos not in self.fifo_reverse_inject_count[category][fifo_type]:
            self.fifo_reverse_inject_count[category][fifo_type][pos] = 0
        self.fifo_reverse_inject_count[category][fifo_type][pos] += 1

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

                # 获取反方向上环统计
                reverse_inject_h = link_stats.get("reverse_inject_h", 0)
                reverse_inject_v = link_stats.get("reverse_inject_v", 0)
                total_reverse_inject = reverse_inject_h + reverse_inject_v

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
                    # 反方向上环统计
                    "reverse_inject_h": reverse_inject_h,
                    "reverse_inject_v": reverse_inject_v,
                    "reverse_inject_total": total_reverse_inject,
                    "reverse_inject_ratio": total_reverse_inject / total_flit if total_flit > 0 else 0.0,
                }

        return stats
