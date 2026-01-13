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
from src.utils.ring import Ring
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
        self.cross_point = {"horizontal": defaultdict(lambda: defaultdict(list)), "vertical": defaultdict(lambda: defaultdict(list))}
        # Crosspoint conflict status: maintains pipeline queue [current_cycle, previous_cycle]
        self.crosspoint_conflict = {"horizontal": defaultdict(lambda: defaultdict(lambda: [False, False])), "vertical": defaultdict(lambda: defaultdict(lambda: [False, False]))}
        # 新的链路状态统计 - 记录各种ETag/ITag状态的计数
        self.links_flow_stat = {}
        # 每个周期的瞬时状态统计
        self.links_state_snapshots = []

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

        # ETag统计（下环时记录）
        self.RB_ETag_T1_num_stat = 0  # 横向下环T1级别的flit计数
        self.RB_ETag_T0_num_stat = 0  # 横向下环T0级别的flit计数
        self.RB_ETag_T1_per_node_fifo = {}  # {node_id: {"TL": count, "TR": count}}
        self.RB_ETag_T0_per_node_fifo = {}  # {node_id: {"TL": count, "TR": count}}

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
                    self.links_flow_stat[(i, j)] = {
                        "ITag_count": 0,
                        "empty_count": 0,
                        "total_cycles": 0,
                        "total_flit_moved": 0,  # 实际通过link的flit数（用于带宽计算）
                        "eject_attempts_h": {"0": 0, "1": 0, "2": 0, ">2": 0},
                        "eject_attempts_v": {"0": 0, "1": 0, "2": 0, ">2": 0},
                        "reverse_inject_h": 0,  # 横向反方向上环flit数
                        "reverse_inject_v": 0,  # 纵向反方向上环flit数
                    }

        # 注意：self-link已移除，边缘节点的方向转换由CP内部环回处理

        for pos in range(config.NUM_NODE):
            # v2架构: ring_bridge已合并到RingStation中

            # RS UE Counters (TL/TR)
            self.RS_UE_Counters["TL"][pos] = {"T2": 0, "T1": 0, "T0": 0}
            self.RS_UE_Counters["TR"][pos] = {"T2": 0, "T1": 0, "T0": 0}

            # ETag per-node 统计初始化
            self.RB_ETag_T1_per_node_fifo[pos] = {"TL": 0, "TR": 0}
            self.RB_ETag_T0_per_node_fifo[pos] = {"TL": 0, "TR": 0}

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

        # 构建Ring对象（slices列表+offset）
        self._build_ring_linked_lists()

        # 构建环slot列表和初始化T0仲裁数据结构（依赖环形链表）
        self._build_ring_slots()

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
        """
        构建每个节点所在环的slot列表，并初始化T0仲裁数据结构

        从已构建的Ring对象获取slot_ids列表（使用Ring.get_all_slot_ids()）
        """
        num_col = self.config.NUM_COL
        num_row = self.config.NUM_ROW

        # 横向环：从Ring对象获取slot_ids
        for row in range(num_row):
            row_nodes = [row * num_col + col for col in range(num_col)]
            ring = self.horizontal_rings[row]
            slot_ids = ring.get_all_slot_ids()
            for node in row_nodes:
                self.horizontal_ring_slots[node] = slot_ids
                self.T0_table_h[node] = set()
                self.T0_arb_pointer_h[node] = 0

        # 纵向环：从Ring对象获取slot_ids
        for col in range(num_col):
            col_nodes = [row * num_col + col for row in range(num_row)]
            ring = self.vertical_rings[col]
            slot_ids = ring.get_all_slot_ids()
            for node in col_nodes:
                self.vertical_ring_slots[node] = slot_ids
                self.T0_table_v[node] = set()
                self.T0_arb_pointer_v[node] = 0

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

    def _build_horizontal_ring(self, row: int) -> Ring:
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

        return Ring(all_slices)

    def _build_vertical_ring(self, col: int) -> Ring:
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

        return Ring(all_slices)

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

            # === CP_IN: 下环检查点 ===
            if s.slice_type == RingSlice.CP_IN:
                should_eject, eject_target, eject_dir = self._should_eject(flit, s.node_id, s.direction, cycle)
                if should_eject:
                    # 统计下环尝试次数
                    if s.direction in ["TL", "TR"]:
                        flit.eject_attempts_h += 1
                    else:
                        flit.eject_attempts_v += 1

                    # 先检查保序
                    if self._can_eject_in_order(flit, flit.path[-1], s.direction):
                        if self._try_eject(flit, s.node_id, eject_dir, eject_target, cycle):
                            s.flit = None  # 下环成功
                            continue
                        else:
                            # 下环失败（Entry不足或FIFO满），升级ETag
                            cp_type = "horizontal" if s.direction in ["TL", "TR"] else "vertical"
                            cp = self.crosspoints[s.node_id][cp_type]
                            cp.upgrade_etag_on_failure(flit, s.direction)
                    else:
                        # 保序失败，升级ETag
                        cp_type = "horizontal" if s.direction in ["TL", "TR"] else "vertical"
                        cp = self.crosspoints[s.node_id][cp_type]
                        cp.upgrade_etag_on_failure(flit, s.direction)

            # === 移动到下一个slice ===
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
            self.process_ring_movement(ring.slices, cycle)
            self.process_ring_injection(ring.slices, cycle)

        for col, ring in self.vertical_rings.items():
            self.process_ring_movement(ring.slices, cycle)
            self.process_ring_injection(ring.slices, cycle)

    def _should_eject(self, flit, node_id: int, direction: str, cycle: int = 0) -> tuple:
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
            # 水平CP：检查是否到达最终目标或需要转到纵向环
            if node_id == final_dest:
                # 到达最终目标，横向下环到RB，然后再转到纵向EQ
                return True, "RB", direction
            if cp._needs_vertical_move(node_id, next_node):
                return True, "RB", direction
        else:
            # 纵向CP：检查是否到达最终目标
            if node_id == final_dest:
                return True, "EQ", direction

        return False, "", ""

    def _try_eject(self, flit, node_id: int, direction: str, eject_target: str, cycle: int) -> bool:
        """尝试下环到RingStation

        检查顺序：
        1. RingStation输入端口容量
        2. Entry可用性
        3. 占用Entry并记录到flit
        """
        # 获取RingStation
        rs = self.ring_stations.get(node_id)
        if rs is None:
            return False

        # 1. 检查RingStation输入FIFO是否有空间
        input_fifo = rs.input_fifos.get(direction)
        if input_fifo is None or len(input_fifo) >= input_fifo.maxlen:
            return False

        # 2. 检查Entry可用性并占用最佳Entry
        cp_type = "horizontal" if direction in ["TL", "TR"] else "vertical"
        cp = self.crosspoints[node_id][cp_type]

        # 根据ETag优先级决定需要的Entry等级
        etag = getattr(flit, 'ETag_priority', 'T2')
        entry_level = None

        if etag == "T0":
            # T0可以用任何等级的Entry
            for level in ["T0", "T1", "T2"]:
                if cp._entry_available(direction, node_id, level):
                    entry_level = level
                    break
        elif etag == "T1":
            # T1可以用T1或T2的Entry
            for level in ["T1", "T2"]:
                if cp._entry_available(direction, node_id, level):
                    entry_level = level
                    break
        else:  # T2
            # T2只能用T2的Entry
            if cp._entry_available(direction, node_id, "T2"):
                entry_level = "T2"

        if entry_level is None:
            return False  # 没有可用Entry

        # 3. 占用Entry
        self.RS_UE_Counters[direction][node_id][entry_level] += 1
        flit.used_entry_level = entry_level
        flit.eject_direction = direction

        # 4. 放入RingStation的输入FIFO
        input_fifo.append(flit)
        flit.set_position(f"RS_IN_{direction}", cycle)

        # 5. 添加延迟释放Entry（下一个cycle释放）
        self.RS_pending_entry_release[direction][node_id].append(
            (entry_level, cycle + 1)
        )

        # 6. 横向下环时更新ETag统计（与v1的RB统计兼容）
        if direction in ["TL", "TR"]:
            if etag == "T1":
                self.RB_ETag_T1_num_stat += 1
                self.RB_ETag_T1_per_node_fifo[node_id][direction] += 1
            elif etag == "T0":
                self.RB_ETag_T0_num_stat += 1
                self.RB_ETag_T0_per_node_fifo[node_id][direction] += 1
            # 统计完成后重置ETag
            flit.ETag_priority = "T2"

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

        # 6. 处理ITag释放（使用Ring.itag）
        cp_type = "horizontal" if is_horizontal else "vertical"
        cp = self.crosspoints[node_id][cp_type]

        # 新架构：使用Ring.itag处理ITag释放
        ring = ring_slice.ring
        slot_id = ring_slice.slot_id
        if ring and slot_id is not None:
            if ring.check_itag(slot_id, node_id, direction):
                # 使用了预约，释放
                ring.release_itag(slot_id)
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

        # 始终更新current_position为当前所在节点
        flit.current_position = node_id

        if slice_type == RingSlice.LINK:
            # Link上flit：设置为"Link"并记录link信息
            num_col = self.config.NUM_COL
            if direction == "TR":
                next_node = node_id + 1
            elif direction == "TL":
                next_node = node_id - 1
            elif direction == "TD":
                next_node = node_id + num_col
            else:  # TU
                next_node = node_id - num_col

            flit.set_position("Link", cycle)
            flit.current_link = (node_id, next_node)
            flit.current_seat_index = ring_slice.link_index

            # 只在flit进入link时统计一次（link_index=0）
            if ring_slice.link_index == 0:
                link = (node_id, next_node)
                if link in self.links_flow_stat:
                    # 统计实际通过link的flit数（用于带宽计算）
                    self.links_flow_stat[link]["total_flit_moved"] += 1
        elif slice_type in [RingSlice.CP_IN, RingSlice.CP_OUT, RingSlice.CP_INTERNAL]:
            # CP格式：CP_H 或 CP_V
            if direction in ["TL", "TR"]:
                pos = "CP_H"
            else:
                pos = "CP_V"
            flit.set_position(pos, cycle)
        else:
            pos = ring_slice.get_position_str()
            flit.set_position(pos, cycle)

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
        在每个周期结束时统计所有链路第一个LINK slice的使用情况

        Args:
            cycle: 当前周期
        """
        for link in self.links_flow_stat:
            src = link[0]
            dest = link[1] if len(link) > 1 else link[0]

            # 根据link确定direction和Ring
            if src == dest:
                continue
            num_col = self.config.NUM_COL
            is_horizontal = abs(src - dest) == 1 or (src // num_col == dest // num_col)
            if is_horizontal:
                direction = "TR" if dest > src else "TL"
                row = src // num_col
                ring = self.horizontal_rings.get(row)
            else:
                direction = "TD" if dest > src else "TU"
                col = src % num_col
                ring = self.vertical_rings.get(col)

            if ring is None:
                continue

            # 找到link的第一个LINK slice (link_index=0) 并获取flit
            flit = None
            for s in ring.slices:
                if s.slice_type == RingSlice.LINK and s.node_id == src and s.direction == direction:
                    if hasattr(s, 'link_index') and s.link_index == 0:
                        flit = s.flit
                        break

            if flit is None:
                self.links_flow_stat[link]["empty_count"] += 1
            else:
                self._update_eject_attempts_stats(link, flit)

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

                # 获取实际通过link的flit数（在flit移动时统计，每个flit只统计一次）
                total_flit_moved = link_stats.get("total_flit_moved", 0)

                # 计算带宽 (GB/s)
                # bandwidth = (total_flit_moved * FLIT_SIZE) / (total_cycles / CYCLES_PER_NS)
                #           = total_flit_moved * FLIT_SIZE * CYCLES_PER_NS / total_cycles
                bandwidth_GB_s = (total_flit_moved * self.config.FLIT_SIZE * self.config.CYCLES_PER_NS) / total_cycles

                stats[link] = {
                    # 带宽 (GB/s)
                    "bandwidth_GB_s": bandwidth_GB_s,
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
                    "total_flit_moved": total_flit_moved,  # 实际通过link的flit数
                    "eject_attempts_h": eject_attempts_h,
                    "eject_attempts_v": eject_attempts_v,
                    # 反方向上环统计
                    "reverse_inject_h": reverse_inject_h,
                    "reverse_inject_v": reverse_inject_v,
                    "reverse_inject_total": total_reverse_inject,
                    "reverse_inject_ratio": total_reverse_inject / total_flit if total_flit > 0 else 0.0,
                }

        return stats
