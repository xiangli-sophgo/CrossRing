"""
Flit and TokenBucket classes for NoC simulation.
Contains the basic data unit (Flit) and rate limiting mechanism (TokenBucket).
"""

from __future__ import annotations
from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     from .network import Network  # 仅用于 IDE 类型提示

import numpy as np
from collections import deque
from typing import List, Optional, Union
import threading


class TokenBucket:
    """Simple token bucket for rate limiting with fractional tokens."""

    def __init__(self, rate: float = 1.0, bucket_size: float = 10.0):
        # rate: tokens added per cycle; bucket_size: max tokens stored
        self.rate = float(rate)
        self.bucket_size = float(bucket_size)
        self.tokens = self.bucket_size
        self.last_cycle = 0

    def consume(self, num: float = 1.0) -> bool:
        """Attempt to consume `num` tokens. Returns True if sufficient tokens."""
        # print(self.tokens)
        if self.tokens + 1e-8 >= num:
            self.tokens -= num
            return True
        return False

    def refill(self, cycle: int):
        """Refill tokens based on cycles elapsed since last refill."""
        # Calculate elapsed cycles
        dt = cycle - self.last_cycle
        if dt <= 0:
            return
        # Add fractional tokens
        added = dt * self.rate
        # Update last refill time
        self.last_cycle = cycle
        # Cap tokens to bucket size
        self.tokens = min(self.tokens + added, self.bucket_size)


class Flit:
    __slots__ = [
        "source",
        "destination",
        "source_type",
        "destination_type",
        "burst_length",
        "path",
        "flit_position",
        "is_finish",
        "packet_id",
        "traffic_id",
        "moving_direction",
        "flit_type",
        "req_type",
        "req_attr",
        "req_state",
        "id",
        "flit_id",
        "is_last_flit",
        "eject_attempts_v",
        "eject_attempts_h",
        "ordering_blocked_eject_h",  # 因保序被阻止的横向下环次数
        "ordering_blocked_eject_v",  # 因保序被阻止的纵向下环次数
        "reverse_inject_h",  # 横向反方向上环标记 (0=正常, 1=反方向)
        "reverse_inject_v",  # 纵向反方向上环标记 (0=正常, 1=反方向)
        "wait_cycle_h",
        "wait_cycle_v",
        "path_index",
        "current_seat_index",
        "current_link",
        "rsp_type",
        "rn_tracker_type",
        "sn_tracker_type",
        "sn_rsp_generate_cycle",
        "early_rsp",
        "current_position",
        "station_position",
        "departure_cycle",
        "req_departure_cycle",
        "departure_network_cycle",
        "departure_inject_cycle",
        "arrival_cycle",
        "arrival_network_cycle",
        "arrival_eject_cycle",
        "entry_db_cycle",
        "leave_db_cycle",
        "start_inject",
        "is_injected",
        "is_ejected",
        "is_on_station",
        "is_delay",
        "is_arrive",
        "predicted_duration",
        "actual_duration",
        "actual_ject_duration",
        "actual_network_duration",
        "itag_v",
        "itag_h",
        "is_tagged",
        "ETag_priority",
        "used_entry_level",
        "eject_direction",  # 下环方向（TL/TR/TU/TD），用于释放 entry
        "T0_slot_id",  # T0轮询机制的slot ID
        "T0_fifo_direction",  # T0 slot注册的FIFO方向（"TL"/"TR"/"TU"/"TD"）
        "cmd_entry_cake0_cycle",
        "cmd_entry_noc_from_cake0_cycle",
        "cmd_entry_noc_from_cake1_cycle",
        "cmd_received_by_cake0_cycle",
        "cmd_received_by_cake1_cycle",
        "data_entry_noc_from_cake0_cycle",
        "data_entry_noc_from_cake1_cycle",
        "data_received_complete_cycle",
        "write_complete_received_cycle",  # 写完成响应接收时间
        "req_start_cycle",  # 请求开始处理时间（tracker消耗开始）
        "rsp_entry_network_cycle",
        "transaction_latency",
        "cmd_latency",
        "data_latency",
        "src_dest_order_id",
        "packet_category",
        "allowed_eject_directions",  # 允许下环的方向列表
        "data_channel_id",
        # D2D相关属性
        "d2d_origin_die",  # 发起Die ID
        "d2d_origin_node",  # 发起节点物理ID
        "d2d_origin_type",  # 发起IP类型
        "d2d_target_die",  # 目标Die ID
        "d2d_target_node",  # 目标节点物理ID
        "d2d_target_type",  # 目标IP类型
        "d2d_sn_node",  # 经过的D2D_SN节点物理ID
        "d2d_rn_node",  # 经过的D2D_RN节点物理ID
        "inject_time",
        # AXI传输相关属性
        "axi_end_cycle",
        "axi_start_cycle",
        # D2D NoC端口时间戳
        "d2d_noc_inject_cycle",  # D2D节点注入NoC的时间
        "d2d_noc_eject_cycle",  # D2D节点从NoC弹出的时间
        # CP位置属性（用于统一两阶段处理）
        "cp_node_id",  # CP所在节点ID
        "cp_direction",  # CP方向 (TL/TR/TU/TD)
        "cp_slice_index",  # CP slice索引 (-1表示在pre中)
        "_next_pos",  # 计划的下一位置
        # 位置时间戳字典（自动记录）
        "position_timestamps",  # {position: cycle}
        "position_timestamps_backup",  # retry请求的第一次失败时间戳备份
    ]

    last_id = 0

    # 全局order_id分配器（从Node类迁移）
    _global_order_id_allocator = {}  # {(src, dest): {"REQ": next_id, "RSP": next_id, "DATA": next_id}}

    @classmethod
    def get_next_order_id(cls, src_node, src_type, dest_node, dest_type, packet_category, granularity, die_id=None):
        """
        获取下一个顺序ID

        Args:
            src_node: 源节点ID（flit当前Die内的source）
            src_type: 源IP类型（如"gdma_0"）
            dest_node: 目标节点ID（flit当前Die内的destination）
            dest_type: 目标IP类型（如"ddr_1"）
            packet_category: 包类型（"REQ"/"RSP"/"DATA"）
            granularity: 保序粒度（0=IP层级, 1=节点层级）
            die_id: Die ID，用于区分不同Die的保序（可选，None表示单Die模式）

        Returns:
            int: 分配的顺序ID
        """
        # 根据粒度构造key - 每个Die独立保序，使用flit的实际source/destination
        # die_id加入key确保不同Die之间保序独立
        if granularity == 0:  # IP层级
            key = (die_id, src_node, src_type, dest_node, dest_type)
        else:  # 节点层级（granularity == 1）
            key = (die_id, src_node, dest_node)

        if key not in cls._global_order_id_allocator:
            cls._global_order_id_allocator[key] = {"REQ": 1, "RSP": 1, "DATA": 1, "FLIT": 1}

        # 如果packet_category不在字典中，则添加它
        if packet_category not in cls._global_order_id_allocator[key]:
            cls._global_order_id_allocator[key][packet_category] = 1

        current_id = cls._global_order_id_allocator[key][packet_category]
        cls._global_order_id_allocator[key][packet_category] += 1
        return current_id

    @classmethod
    def reset_order_ids(cls):
        """重置所有顺序ID"""
        cls._global_order_id_allocator.clear()

    def __init__(self, source, destination, path):
        self.source = source
        self.destination = destination
        self.source_type = None
        self.destination_type = None
        self.burst_length = -1
        self.path = path
        self.flit_position = ""
        self.position_timestamps = {}  # {position: cycle} 位置时间戳自动记录
        self.position_timestamps_backup = None  # retry请求的第一次失败时间戳备份
        self.is_finish = False
        Flit.last_id += 1
        self.packet_id = None
        # 标记该flit所属的traffic
        self.traffic_id = None
        self.flit_type = "flit"
        self.req_type = None
        self.req_attr = "new"
        self.req_state = "valid"
        self.id = Flit.last_id
        self.flit_id = -1
        self.is_last_flit = False
        self.eject_attempts_v = 0
        self.eject_attempts_h = 0
        self.ordering_blocked_eject_h = 0
        self.ordering_blocked_eject_v = 0
        self.reverse_inject_h = 0
        self.reverse_inject_v = 0
        self.wait_cycle_h = 0
        self.wait_cycle_v = 0
        self.path_index = 0
        self.current_seat_index = -1
        self.current_link = None
        self.rsp_type = None
        self.rn_tracker_type = None
        self.sn_tracker_type = None
        # D2D新属性初始化
        self.d2d_origin_die = None
        self.d2d_origin_node = None
        self.d2d_origin_type = None
        self.d2d_target_die = None
        self.d2d_target_node = None
        self.d2d_target_type = None
        self.d2d_sn_node = None
        self.d2d_rn_node = None
        self.inject_time = None
        # CP位置属性初始化
        self.cp_node_id = None
        self.cp_direction = None
        self.cp_slice_index = None
        self._next_pos = None
        self.init_param()

    def init_param(self):
        self.early_rsp = False
        self.current_position = self.path[0]
        self.station_position = -1
        self.departure_cycle = np.inf
        self.req_departure_cycle = np.inf
        self.departure_network_cycle = np.inf
        self.departure_inject_cycle = np.inf
        self.arrival_cycle = np.inf
        self.arrival_network_cycle = np.inf
        self.arrival_eject_cycle = np.inf
        self.entry_db_cycle = np.inf
        self.leave_db_cycle = np.inf
        self.start_inject = False
        self.is_injected = False
        self.is_ejected = False
        self.is_on_station = False
        self.is_delay = False
        self.is_arrive = False
        self.predicted_duration = 0
        self.actual_duration = 0
        self.actual_ject_duration = 0
        self.actual_network_duration = 0
        self.itag_v = False
        self.itag_h = False
        self.is_tagged = False
        self.ETag_priority = "T2"  # 默认优先级为 T2
        # 记录下环 / 弹出时实际占用的是哪一级 entry（"T0" / "T1" / "T2"）
        self.used_entry_level = None
        self.T0_slot_id = None  # T0轮询机制的slot ID
        self.T0_fifo_direction = None  # T0 slot注册的FIFO方向（"TL"/"TR"/"TU"/"TD"）
        # Latency record
        self.req_start_cycle = np.inf  # 请求开始处理时间（tracker消耗开始）
        self.cmd_entry_cake0_cycle = np.inf
        self.cmd_entry_noc_from_cake0_cycle = np.inf
        self.cmd_entry_noc_from_cake1_cycle = np.inf
        self.cmd_received_by_cake0_cycle = np.inf
        self.cmd_received_by_cake1_cycle = np.inf
        self.data_entry_noc_from_cake0_cycle = np.inf
        self.data_entry_noc_from_cake1_cycle = np.inf
        self.data_received_complete_cycle = np.inf
        self.write_complete_received_cycle = np.inf  # 写完成响应接收时间
        self.rsp_entry_network_cycle = np.inf
        self.transaction_latency = np.inf
        self.cmd_latency = np.inf
        self.data_latency = np.inf
        self.sn_rsp_generate_cycle = np.inf
        self.src_dest_order_id = -1
        self.packet_category = None
        self.allowed_eject_directions = None  # 允许下环的方向列表
        self.data_channel_id = 0  # 默认数据通道0
        self.d2d_noc_inject_cycle = np.inf  # D2D NoC注入时间
        self.d2d_noc_eject_cycle = np.inf  # D2D NoC弹出时间

    def set_position(self, position: str, cycle: int):
        """设置位置并记录时间戳

        Args:
            position: 位置名称 (如 "IP_TX", "L2H", "IQ_CH", "Link", "RB", "EQ", "IP_RX")
            cycle: 当前时间周期
        """
        self.flit_position = position
        self.position_timestamps[position] = cycle

    def sync_latency_record(self, flit):
        if flit.req_type == "read":
            self.cmd_entry_cake0_cycle = min(flit.cmd_entry_cake0_cycle, self.cmd_entry_cake0_cycle)
            self.cmd_entry_noc_from_cake0_cycle = min(flit.cmd_entry_noc_from_cake0_cycle, self.cmd_entry_noc_from_cake0_cycle)
            self.cmd_received_by_cake1_cycle = min(flit.cmd_received_by_cake1_cycle, self.cmd_received_by_cake1_cycle)
            self.data_entry_noc_from_cake1_cycle = min(flit.data_entry_noc_from_cake1_cycle, self.data_entry_noc_from_cake1_cycle)
            self.data_received_complete_cycle = min(flit.data_received_complete_cycle, self.data_received_complete_cycle)
        elif flit.req_type == "write":
            self.cmd_entry_cake0_cycle = min(flit.cmd_entry_cake0_cycle, self.cmd_entry_cake0_cycle)
            self.cmd_entry_noc_from_cake0_cycle = min(flit.cmd_entry_noc_from_cake0_cycle, self.cmd_entry_noc_from_cake0_cycle)
            self.cmd_received_by_cake1_cycle = min(flit.cmd_received_by_cake1_cycle, self.cmd_received_by_cake1_cycle)
            self.cmd_entry_noc_from_cake1_cycle = min(flit.cmd_entry_noc_from_cake1_cycle, self.cmd_entry_noc_from_cake1_cycle)
            self.cmd_received_by_cake0_cycle = min(flit.cmd_received_by_cake0_cycle, self.cmd_received_by_cake0_cycle)
            self.data_entry_noc_from_cake0_cycle = min(flit.data_entry_noc_from_cake0_cycle, self.data_entry_noc_from_cake0_cycle)
            self.data_received_complete_cycle = min(flit.data_received_complete_cycle, self.data_received_complete_cycle)

    def set_packet_category_and_order_id(self):
        """根据flit的类型信息设置包类型和顺序ID"""
        # 确定包类型分类
        if self.flit_type == "req":
            self.packet_category = "REQ"
        elif self.flit_type == "rsp":
            self.packet_category = "RSP"
        elif self.flit_type == "data":
            self.packet_category = "DATA"
        else:
            raise ValueError(self.flit_type)

    def inject(self, network):  # 使用字符串类型标注
        if self.path_index == 0 and not self.is_injected:
            if len(self.path) > 1:  # Ensure there is a next position
                next_position = self.path[self.path_index + 1]
                if network.can_move_to_next(self, self.source, next_position):
                    self.current_position = self.source
                    self.is_injected = True
                    self.current_link = None
                    return True
        return False

    def __repr__(self):
        req_attr = "O" if self.req_attr == "old" else "N"
        type_display = self.rsp_type[:3] if self.rsp_type else self.req_type[0]

        # 处理flit位置显示
        if self.flit_position != "Link":
            flit_position = f"{self.current_position}:{self.flit_position}"
        elif self.current_link is None:
            flit_position = f"{self.current_position}:None"
        else:
            # 处理3元组（Ring Bridge）和2元组（普通link）
            link_start = self.current_link[0]
            link_end = self.current_link[1]
            flit_position = f"{link_start}->{link_end}:{self.current_seat_index}"

        finish_status = "F" if self.is_finish else ""
        eject_status = "E" if self.is_ejected else ""
        ITag_H = "H" if self.itag_h else ""
        ITag_V = "V" if self.itag_v else ""

        return (
            f"{self.packet_id}.{self.flit_id} {self.source}.{self.source_type[0]}{self.source_type[-1]}->{self.destination}.{self.destination_type[0]}{self.destination_type[-1]}: "
            f"{flit_position}, "
            f"{req_attr}, {self.flit_type}, {type_display}, "
            f"{finish_status}{eject_status}, "
            f"{self.ETag_priority}, {ITag_H}, {ITag_V}"
        )

    def _reset_for_reuse(self):
        """Reset flit to clean state for object pool reuse"""
        # Reset critical fields but keep __slots__ structure
        self.is_finish = False
        self.is_arrive = False
        self.is_injected = False
        self.is_ejected = False
        self.is_on_station = False
        self.is_delay = False
        self.wait_cycle_h = 0
        self.wait_cycle_v = 0
        self.eject_attempts_h = 0
        self.eject_attempts_v = 0
        self.ordering_blocked_eject_h = 0
        self.ordering_blocked_eject_v = 0
        self.reverse_inject_h = 0
        self.reverse_inject_v = 0
        self.ETag_priority = "T2"
        self.T0_slot_id = None
        self.path_index = 0
        self.current_seat_index = -1
        self.current_link = None
        self.traffic_id = None
        self.src_dest_order_id = -1
        self.packet_category = None
        self.allowed_eject_directions = None
        self.data_channel_id = 0
        # 重置D2D属性
        self.d2d_origin_die = None
        self.d2d_origin_node = None
        self.d2d_origin_type = None
        self.d2d_target_die = None
        self.d2d_target_node = None
        self.d2d_target_type = None
        self.d2d_sn_node = None
        self.d2d_rn_node = None
        self.inject_time = None

        # Reset timing fields
        self.req_start_cycle = np.inf
        self.write_complete_received_cycle = np.inf
        self.departure_cycle = np.inf
        self.req_departure_cycle = np.inf
        self.departure_network_cycle = np.inf
        self.departure_inject_cycle = np.inf
        self.arrival_cycle = np.inf
        self.arrival_network_cycle = np.inf
        self.arrival_eject_cycle = np.inf

        # Reset latency fields
        self.transaction_latency = np.inf
        self.cmd_latency = np.inf
        self.data_latency = np.inf
        # Reset position timestamps
        self.position_timestamps = {}
        self.position_timestamps_backup = None

    @classmethod
    def create_flit(cls, source, destination, path):
        """Factory method to create Flit using object pool"""
        return _flit_pool.get_flit(source, destination, path)

    @classmethod
    def return_to_pool(cls, flit):
        """Return flit to object pool"""
        _flit_pool.return_flit(flit)

    @classmethod
    def clear_flit_id(cls):
        cls.last_id = 0

    @classmethod
    def get_pool_stats(cls):
        """Get object pool statistics"""
        return _flit_pool.get_stats()


class FlitPool:
    """Object pool for Flit instances to reduce GC pressure"""

    def __init__(self, initial_size=1000):
        self._pool = deque()
        self._lock = threading.Lock()
        self._created_count = 0

        # Pre-populate the pool
        for _ in range(initial_size):
            self._pool.append(self._create_new_flit())

    def _create_new_flit(self):
        """Create a new Flit instance"""
        self._created_count += 1
        return Flit.__new__(Flit)

    def get_flit(self, source, destination, path):
        """Get a Flit from the pool or create a new one"""
        with self._lock:
            if self._pool:
                flit = self._pool.popleft()
            else:
                flit = self._create_new_flit()

        # Initialize the flit
        flit.__init__(source, destination, path)
        return flit

    def return_flit(self, flit):
        """Return a Flit to the pool after use"""
        if flit is None:
            return

        # Reset the flit to a clean state
        flit._reset_for_reuse()

        with self._lock:
            if len(self._pool) < 2000:  # Limit pool size
                self._pool.append(flit)

    def get_stats(self):
        """Get pool statistics"""
        with self._lock:
            return {"pool_size": len(self._pool), "created_count": self._created_count}


# Global flit pool instance
_flit_pool = FlitPool()


def copy_flit_attributes(src_flit: Flit, dst_flit: Flit, attr_list: list):
    """
    复制flit属性的工具函数

    Args:
        src_flit: 源flit对象
        dst_flit: 目标flit对象
        attr_list: 要复制的属性名列表
    """
    for attr in attr_list:
        if hasattr(src_flit, attr):
            setattr(dst_flit, attr, getattr(src_flit, attr))


# D2D属性预定义集合
D2D_BASIC_ATTRS = ["packet_id", "flit_id", "req_type", "burst_length", "traffic_id"]

D2D_ORIGIN_TARGET_ATTRS = ["d2d_origin_die", "d2d_origin_node", "d2d_origin_type", "d2d_target_die", "d2d_target_node", "d2d_target_type", "d2d_sn_node", "d2d_rn_node"]

D2D_REQUEST_ATTRS = D2D_BASIC_ATTRS + ["req_attr"] + D2D_ORIGIN_TARGET_ATTRS

D2D_RESPONSE_ATTRS = D2D_BASIC_ATTRS + ["rsp_type", "flit_type", "is_last_flit"] + D2D_ORIGIN_TARGET_ATTRS

D2D_LATENCY_TIMESTAMP_ATTRS = [
    "cmd_entry_cake0_cycle",
    "cmd_entry_noc_from_cake0_cycle",
    "cmd_received_by_cake0_cycle",
    "cmd_received_by_cake1_cycle",
    "data_entry_noc_from_cake0_cycle",
    "data_entry_noc_from_cake1_cycle",
    "data_received_complete_cycle",
    "write_complete_received_cycle",
]

D2D_DATA_ATTRS = D2D_BASIC_ATTRS + ["flit_type"] + D2D_ORIGIN_TARGET_ATTRS + D2D_LATENCY_TIMESTAMP_ATTRS

D2D_TIMESTAMP_ATTRS = ["departure_cycle", "entry_db_cycle", "req_departure_cycle", "leave_db_cycle"]


def create_d2d_flit_copy(src_flit: Flit, source: int = 0, destination: int = 0, path: list = None, attr_preset: str = "request") -> Flit:
    """
    创建D2D flit副本的统一方法

    Args:
        src_flit: 源flit（通常是AXI flit）
        source: 新flit的源节点位置
        destination: 新flit的目标节点位置
        path: 新flit的路径
        attr_preset: 属性预设类型，可选值：
            - "request": 请求flit（包含req_attr）
            - "response": 响应flit（包含rsp_type, flit_type, is_last_flit）
            - "data": 数据flit（包含flit_type）
            - "basic": 仅基础属性
            - "with_timestamp": 基础属性 + 时间戳

    Returns:
        Flit: 新创建的flit副本

    Examples:
        # 创建跨Die读请求副本
        new_flit = create_d2d_flit_copy(axi_flit, source=36, destination=4,
                                        path=[36, 4], attr_preset="request")

        # 创建跨Die写数据副本
        data_flit = create_d2d_flit_copy(axi_flit, source=36, destination=4,
                                         path=[36, 4], attr_preset="data")

        # 创建响应flit副本
        rsp_flit = create_d2d_flit_copy(axi_flit, source=4, destination=36,
                                        path=[4, 36], attr_preset="response")
    """
    if path is None:
        path = [source]

    # 从对象池获取新flit
    new_flit = _flit_pool.get_flit(source=source, destination=destination, path=path)

    # 根据预设类型选择要复制的属性
    if attr_preset == "request":
        attr_list = D2D_REQUEST_ATTRS
    elif attr_preset == "response":
        attr_list = D2D_RESPONSE_ATTRS
    elif attr_preset == "data":
        attr_list = D2D_DATA_ATTRS
    elif attr_preset == "with_timestamp":
        attr_list = D2D_BASIC_ATTRS + D2D_ORIGIN_TARGET_ATTRS + D2D_TIMESTAMP_ATTRS
    elif attr_preset == "basic":
        attr_list = D2D_BASIC_ATTRS + D2D_ORIGIN_TARGET_ATTRS
    else:
        raise ValueError(f"未知的attr_preset类型: {attr_preset}")

    # 复制属性
    copy_flit_attributes(src_flit, new_flit, attr_list)

    return new_flit


# ============================================================================
# 原始节点信息获取辅助函数
# 用于统一获取flit的原始源/目标节点信息，替代已废弃的_original属性
# ============================================================================


def get_original_source_node(flit: Flit) -> int:
    """
    获取flit的原始源节点位置（物理ID）

    优先返回D2D跨Die请求的原始发起者节点，否则返回当前源节点。
    用于替代已废弃的 source_original 属性。

    Args:
        flit: 要查询的flit对象

    Returns:
        int: 原始源节点位置（源映射）
    """
    if hasattr(flit, "d2d_origin_node") and flit.d2d_origin_node is not None:
        return flit.d2d_origin_node
    return flit.source


def get_original_destination_node(flit: Flit) -> int:
    """
    获取flit的原始目标节点位置（物理ID）

    优先返回D2D跨Die请求的最终目标节点，否则返回当前目标节点。
    用于替代已废弃的 destination_original 属性。

    Args:
        flit: 要查询的flit对象

    Returns:
        int: 原始目标节点位置（源映射）
    """
    if hasattr(flit, "d2d_target_node") and flit.d2d_target_node is not None:
        return flit.d2d_target_node
    return flit.destination


def get_original_source_type(flit: Flit) -> Optional[str]:
    """
    获取flit的原始源IP类型

    优先返回D2D跨Die请求的原始发起者IP类型，否则返回当前源IP类型。
    用于替代已废弃的 original_source_type 属性。

    Args:
        flit: 要查询的flit对象

    Returns:
        Optional[str]: 原始源IP类型（如 "gdma_0", "ddr_0" 等），无则返回None
    """
    if hasattr(flit, "d2d_origin_type") and flit.d2d_origin_type:
        return flit.d2d_origin_type
    return getattr(flit, "source_type", None)


def get_original_destination_type(flit: Flit) -> Optional[str]:
    """
    获取flit的原始目标IP类型

    优先返回D2D跨Die请求的最终目标IP类型，否则返回当前目标IP类型。
    用于替代已废弃的 original_destination_type 属性。

    Args:
        flit: 要查询的flit对象

    Returns:
        Optional[str]: 原始目标IP类型（如 "gdma_0", "ddr_0" 等），无则返回None
    """
    if hasattr(flit, "d2d_target_type") and flit.d2d_target_type:
        return flit.d2d_target_type
    return getattr(flit, "destination_type", None)
