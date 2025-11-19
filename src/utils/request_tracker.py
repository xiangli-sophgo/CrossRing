"""
请求追踪器模块 - 提供独立于arrive_flits的请求生命周期追踪

设计目标:
1. 解决跨DIE场景下时间戳无法同步的问题
2. 统一管理请求的完整生命周期状态
3. 从所有flit中收集时间戳到统一的timestamps字典
4. 提供请求完成状态查询，过滤未完成的请求

基于C2C仓库的RequestTracker设计，适配CrossRing的需求。
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


class RequestState(Enum):
    """请求状态枚举"""
    CREATED = "created"          # 请求创建
    INJECTED = "injected"        # 已注入网络
    IN_NETWORK = "in_network"    # 在网络中传输
    ARRIVED = "arrived"          # 已到达目标
    COMPLETED = "completed"      # 请求完成


@dataclass
class RequestLifecycle:
    """请求生命周期记录"""
    # 基本信息
    packet_id: int
    source: int
    destination: int
    source_type: str
    dest_type: str
    op_type: str                # 'read' or 'write'
    burst_size: int

    # 时间戳（cycle）
    created_cycle: int = 0
    injected_cycle: int = 0
    completed_cycle: int = 0

    # 状态管理
    current_state: RequestState = RequestState.CREATED

    # D2D属性（默认为0表示单Die场景）
    is_cross_die: bool = False
    origin_die: int = 0
    target_die: int = 0

    # Flit追踪（核心功能）
    request_flits: List[Any] = field(default_factory=list)
    response_flits: List[Any] = field(default_factory=list)
    data_flits: List[Any] = field(default_factory=list)

    # 统一的时间戳字典（从所有flit中收集）
    timestamps: Dict[str, int] = field(default_factory=dict)

    # 调试控制
    debug_started: bool = False


class RequestTracker:
    """请求追踪器 - 管理请求的完整生命周期"""

    def __init__(self, network_frequency: float = 2.0):
        """
        初始化请求追踪器

        Args:
            network_frequency: 网络频率 (GHz)，用于cycle到ns的转换
        """
        self.active_requests: Dict[int, RequestLifecycle] = {}      # 活跃请求
        self.completed_requests: Dict[int, RequestLifecycle] = {}  # 已完成请求
        self.network_frequency = network_frequency

    # ===== 请求生命周期管理 =====

    def start_request(self, packet_id: int, source: int, destination: int,
                     source_type: str, dest_type: str, op_type: str,
                     burst_size: int, cycle: int, **kwargs):
        """
        开始追踪请求

        Args:
            packet_id: 包ID
            source: 源节点ID
            destination: 目标节点ID
            source_type: 源IP类型
            dest_type: 目标IP类型
            op_type: 操作类型 ('read' or 'write')
            burst_size: burst长度
            cycle: 创建周期
            **kwargs: 可选参数（is_cross_die, origin_die, target_die等）
        """
        # 获取die信息，如果未指定则默认为0（单Die场景）
        origin_die = kwargs.get('origin_die', 0)
        target_die = kwargs.get('target_die', 0)
        is_cross_die = kwargs.get('is_cross_die', False) or (origin_die != target_die)

        lifecycle = RequestLifecycle(
            packet_id=packet_id,
            source=source,
            destination=destination,
            source_type=source_type,
            dest_type=dest_type,
            op_type=op_type,
            burst_size=burst_size,
            created_cycle=cycle,
            is_cross_die=is_cross_die,
            origin_die=origin_die,
            target_die=target_die
        )
        self.active_requests[packet_id] = lifecycle

    def update_request_state(self, packet_id: int, new_state: RequestState, cycle: int):
        """
        更新请求状态

        Args:
            packet_id: 包ID
            new_state: 新状态
            cycle: 当前周期
        """
        if packet_id in self.active_requests:
            lifecycle = self.active_requests[packet_id]
            lifecycle.current_state = new_state

            if new_state == RequestState.INJECTED:
                lifecycle.injected_cycle = cycle
            elif new_state == RequestState.COMPLETED:
                lifecycle.completed_cycle = cycle
                # 移动到完成列表
                self.completed_requests[packet_id] = lifecycle
                del self.active_requests[packet_id]

    def mark_request_injected(self, packet_id: int, cycle: int):
        """标记请求已注入网络"""
        self.update_request_state(packet_id, RequestState.INJECTED, cycle)

    def mark_request_completed(self, packet_id: int, cycle: int):
        """标记请求已完成"""
        self.update_request_state(packet_id, RequestState.COMPLETED, cycle)

    # ===== Flit追踪 =====

    def add_request_flit(self, packet_id: int, flit: Any):
        """添加请求flit"""
        if packet_id in self.active_requests:
            self.active_requests[packet_id].request_flits.append(flit)
        elif packet_id in self.completed_requests:
            self.completed_requests[packet_id].request_flits.append(flit)

    def add_response_flit(self, packet_id: int, flit: Any):
        """添加响应flit"""
        if packet_id in self.active_requests:
            self.active_requests[packet_id].response_flits.append(flit)
        elif packet_id in self.completed_requests:
            self.completed_requests[packet_id].response_flits.append(flit)

    def add_data_flit(self, packet_id: int, flit: Any):
        """添加数据flit"""
        if packet_id in self.active_requests:
            self.active_requests[packet_id].data_flits.append(flit)
        elif packet_id in self.completed_requests:
            self.completed_requests[packet_id].data_flits.append(flit)

    # ===== 时间戳管理（关键创新）=====

    def update_timestamp(self, packet_id: int, field_name: str, value: int):
        """
        更新请求的特定时间戳字段

        Args:
            packet_id: 包ID
            field_name: 时间戳字段名（如 'write_complete_received_cycle'）
            value: 时间戳值（cycle数）
        """
        lifecycle = self.active_requests.get(packet_id) or self.completed_requests.get(packet_id)
        if lifecycle:
            lifecycle.timestamps[field_name] = value

    def collect_timestamps_from_flits(self, packet_id: int) -> Dict[str, int]:
        """
        从所有flit中收集时间戳到统一字典（核心功能）

        这个方法解决了跨DIE场景下时间戳分散在不同flit上的问题。

        Args:
            packet_id: 包ID

        Returns:
            时间戳字典 {field_name: cycle_value}
        """
        lifecycle = self.active_requests.get(packet_id) or self.completed_requests.get(packet_id)
        if not lifecycle:
            return {}

        timestamps = {}
        all_flits = lifecycle.request_flits + lifecycle.response_flits + lifecycle.data_flits

        # 定义需要收集的时间戳字段
        timestamp_fields = [
            'cmd_entry_cake0_cycle',
            'cmd_entry_noc_from_cake0_cycle',
            'cmd_entry_noc_from_cake1_cycle',
            'cmd_received_by_cake0_cycle',
            'cmd_received_by_cake1_cycle',
            'data_entry_noc_from_cake0_cycle',
            'data_entry_noc_from_cake1_cycle',
            'data_received_complete_cycle',
            'write_complete_received_cycle',
            'rsp_entry_network_cycle',
        ]

        for field in timestamp_fields:
            values = []
            for f in all_flits:
                if hasattr(f, field):
                    val = getattr(f, field)
                    if val is not None and val < float('inf'):
                        values.append(val)

            if values:
                # 对于complete/received字段，使用最大值（最后完成的）
                # 对于entry字段，使用最小值（最早发生的）
                if 'complete' in field or 'received' in field:
                    timestamps[field] = max(values)
                else:
                    timestamps[field] = min(values)

        # 更新lifecycle的timestamps
        lifecycle.timestamps.update(timestamps)
        return timestamps

    # ===== 查询接口 =====

    def get_request_status(self, packet_id: int) -> Optional[RequestLifecycle]:
        """获取请求状态"""
        if packet_id in self.active_requests:
            return self.active_requests[packet_id]
        elif packet_id in self.completed_requests:
            return self.completed_requests[packet_id]
        return None

    def is_request_complete(self, packet_id: int) -> bool:
        """检查请求是否完成"""
        return packet_id in self.completed_requests

    def get_completed_requests(self) -> Dict[int, RequestLifecycle]:
        """获取所有已完成的请求"""
        return self.completed_requests

    def get_active_requests(self) -> Dict[int, RequestLifecycle]:
        """获取所有活跃请求"""
        return self.active_requests

    def get_timestamps(self, packet_id: int) -> Dict[str, int]:
        """
        获取请求的所有时间戳（懒加载）

        Args:
            packet_id: 包ID

        Returns:
            时间戳字典
        """
        lifecycle = self.get_request_status(packet_id)
        if not lifecycle:
            return {}

        # 如果还没有收集时间戳，现在收集
        if not lifecycle.timestamps:
            self.collect_timestamps_from_flits(packet_id)

        return lifecycle.timestamps

    # ===== 统计和调试 =====

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计字典，包含总请求数、活跃请求数、完成请求数、完成率等
        """
        total = len(self.active_requests) + len(self.completed_requests)
        completed = len(self.completed_requests)
        active = len(self.active_requests)

        stats = {
            "total_requests": total,
            "active_requests": active,
            "completed_requests": completed,
            "completion_rate": (completed / total * 100) if total > 0 else 0.0,
        }

        # 按请求类型统计
        read_completed = sum(1 for lc in self.completed_requests.values() if lc.op_type == "read")
        write_completed = sum(1 for lc in self.completed_requests.values() if lc.op_type == "write")
        cross_die_completed = sum(1 for lc in self.completed_requests.values() if lc.is_cross_die)

        stats.update({
            "read_completed": read_completed,
            "write_completed": write_completed,
            "cross_die_completed": cross_die_completed,
        })

        return stats

    def print_final_report(self) -> None:
        """打印最终报告"""
        stats = self.get_statistics()

        print(f"\n=== RequestTracker 最终报告 ===")
        print(f"总请求数: {stats['total_requests']}")
        print(f"活跃请求: {stats['active_requests']}")
        print(f"已完成请求: {stats['completed_requests']}")
        print(f"完成率: {stats['completion_rate']:.2f}%")
        print(f"  读请求完成: {stats['read_completed']}")
        print(f"  写请求完成: {stats['write_completed']}")
        print(f"  跨DIE完成: {stats['cross_die_completed']}")

        # 显示未完成请求（如果有）
        if self.active_requests:
            print(f"\n未完成的请求 ({len(self.active_requests)}个):")
            for packet_id, lifecycle in list(self.active_requests.items())[:10]:  # 最多显示10个
                print(f"  Packet {packet_id}: "
                      f"{lifecycle.source}→{lifecycle.destination}, "
                      f"类型={lifecycle.op_type}, "
                      f"状态={lifecycle.current_state.value}, "
                      f"flit数={len(lifecycle.request_flits + lifecycle.response_flits + lifecycle.data_flits)}")

            if len(self.active_requests) > 10:
                print(f"  ... 还有{len(self.active_requests) - 10}个未完成请求")

        # 显示已完成请求示例
        if self.completed_requests:
            print(f"\n已完成请求示例 (前5个):")
            for packet_id, lifecycle in list(self.completed_requests.items())[:5]:
                latency = lifecycle.completed_cycle - lifecycle.created_cycle
                print(f"  Packet {packet_id}: "
                      f"{lifecycle.source}→{lifecycle.destination}, "
                      f"延迟={latency}周期, "
                      f"类型={lifecycle.op_type}")

    def reset(self) -> None:
        """重置跟踪器"""
        self.active_requests.clear()
        self.completed_requests.clear()

    def should_print_debug(self, packet_id: int) -> bool:
        """
        判断是否应该打印调试信息

        Args:
            packet_id: 包ID

        Returns:
            是否应该打印调试信息
        """
        lifecycle = self.get_request_status(packet_id)
        if not lifecycle:
            return False

        # 已完成则停止打印
        if lifecycle.current_state == RequestState.COMPLETED:
            return False

        # 检查是否应该开始打印
        if not lifecycle.debug_started:
            # 已注入且有flit
            if lifecycle.current_state == RequestState.INJECTED and len(lifecycle.request_flits) > 0:
                lifecycle.debug_started = True
                return True

            # 或者检查flit位置
            all_flits = lifecycle.request_flits + lifecycle.response_flits + lifecycle.data_flits
            for flit in all_flits:
                if hasattr(flit, "flit_position"):
                    if flit.flit_position in ["channel", "l2h_fifo", "h2l_fifo", "pending", "IP_CH", "N"]:
                        lifecycle.debug_started = True
                        return True
            return False

        return True
