"""
核心计算器模块 - 提供数据验证、时间区间计算和带宽计算功能

包含:
1. DataValidator - 数据验证类
2. TimeIntervalCalculator - 时间区间计算类
3. BandwidthCalculator - 带宽计算类
"""

import numpy as np
from typing import List, Tuple, Optional
from .analyzers import RequestInfo, WorkingInterval, BandwidthMetrics


class DataValidator:
    """数据验证器 - 提供静态方法用于验证数据有效性"""

    @staticmethod
    def is_valid_number(value) -> bool:
        """
        检查数值是否有效（非 NaN/Inf）

        Args:
            value: 待检查的数值

        Returns:
            bool: 数值有效返回True，否则返回False
        """
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return not (np.isnan(value) or np.isinf(value))
        return False

    @staticmethod
    def sanitize_value(value, default=0.0):
        """
        清理无效值，替换为默认值

        Args:
            value: 待清理的值
            default: 默认值

        Returns:
            清理后的值或默认值
        """
        if DataValidator.is_valid_number(value):
            return value
        return default

    @staticmethod
    def validate_request(req: RequestInfo) -> bool:
        """
        验证请求数据完整性

        Args:
            req: RequestInfo对象

        Returns:
            bool: 验证通过返回True，否则返回False
        """
        # 检查关键时间字段
        if not DataValidator.is_valid_number(req.start_time):
            return False
        if not DataValidator.is_valid_number(req.end_time):
            return False

        # 检查数据大小字段
        if not DataValidator.is_valid_number(req.total_bytes) or req.total_bytes <= 0:
            return False
        if not DataValidator.is_valid_number(req.burst_length) or req.burst_length <= 0:
            return False

        # 检查时间逻辑
        if req.start_time > req.end_time:
            return False

        return True


class TimeIntervalCalculator:
    """时间区间计算器 - 计算工作区间并合并相近区间"""

    def __init__(self, min_gap_threshold: int = 200):
        """
        初始化时间区间计算器

        Args:
            min_gap_threshold: 工作区间合并阈值(ns)，小于此值的间隔被视为同一工作区间
        """
        self.min_gap_threshold = min_gap_threshold

    def calculate_working_intervals(self, requests: List[RequestInfo]) -> List[WorkingInterval]:
        """
        计算工作区间，去除空闲时间段

        Args:
            requests: 请求列表

        Returns:
            工作区间列表
        """
        if not requests:
            return []

        # 构建时间轴事件
        events = []
        for req in requests:
            events.append((req.start_time, "start", req))
            events.append((req.end_time, "end", req))
        events.sort(key=lambda x: (x[0], x[1]))  # 按时间排序，相同时间时'end'在'start'前面

        # 识别连续工作时段
        active_requests = set()
        raw_intervals = []
        current_start = None

        for time_point, event_type, req in events:
            # 检查时间点是否有效（非nan且非None）
            if time_point is not None and not (isinstance(time_point, float) and np.isnan(time_point)):

                if event_type == "start":
                    if not active_requests:  # 开始新的工作区间
                        current_start = time_point
                    active_requests.add(req.packet_id)
                else:  # 'end'
                    active_requests.discard(req.packet_id)
                    if not active_requests and current_start is not None:
                        # 工作区间结束
                        raw_intervals.append((current_start, time_point))
                        current_start = None

        # 处理最后未结束的区间
        if active_requests and current_start is not None:
            last_end = max(req.end_time for req in requests)
            raw_intervals.append((current_start, last_end))

        # 合并相近区间（间隔小于阈值）
        merged_intervals = self._merge_close_intervals(raw_intervals)

        # 构建WorkingInterval对象
        working_intervals = []
        for start, end in merged_intervals:
            # 找到该区间内的所有请求
            interval_requests = [req for req in requests if req.start_time < end and req.end_time > start]

            if not interval_requests:
                continue

            # 计算区间统计
            total_bytes = sum(req.total_bytes for req in interval_requests)
            flit_count = sum(req.burst_length for req in interval_requests)

            interval = WorkingInterval(start_time=start, end_time=end, duration=end - start, flit_count=flit_count, total_bytes=total_bytes, request_count=len(interval_requests))
            working_intervals.append(interval)

        return working_intervals

    def _merge_close_intervals(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        合并相近的时间区间

        Args:
            intervals: 时间区间列表 [(start, end), ...]

        Returns:
            合并后的时间区间列表
        """
        if not intervals:
            return []

        # 按开始时间排序
        sorted_intervals = sorted(intervals)
        merged = [sorted_intervals[0]]

        for current_start, current_end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]

            # 如果间隙小于阈值，则合并
            if current_start - last_end <= self.min_gap_threshold:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        return merged


class BandwidthCalculator:
    """带宽计算器 - 计算各种带宽指标"""

    def __init__(self, time_interval_calculator: TimeIntervalCalculator):
        """
        初始化带宽计算器

        Args:
            time_interval_calculator: 时间区间计算器实例
        """
        self.time_interval_calculator = time_interval_calculator

    def calculate_bandwidth_metrics(self, requests: List[RequestInfo], operation_type: str = None, endpoint_type: str = "network") -> BandwidthMetrics:
        """
        计算指定操作类型的带宽指标

        Args:
            requests: 所有请求列表
            operation_type: 'read'、'write' 或 None（表示混合读写）
            endpoint_type: 'network'(整体网络)、'rn'(RN端口)、'sn'(SN端口)

        Returns:
            BandwidthMetrics对象
        """
        # 筛选请求并创建临时请求列表（使用正确的结束时间）
        filtered_requests = []

        for req in requests:
            if operation_type is not None and req.req_type != operation_type:
                continue

            # 统一使用整体end_time
            end_time = req.end_time

            # 创建临时请求对象
            temp_req = RequestInfo(
                packet_id=req.packet_id,
                start_time=req.start_time,
                end_time=end_time,
                req_type=req.req_type,
                source_node=req.source_node,
                dest_node=req.dest_node,
                source_type=req.source_type,
                dest_type=req.dest_type,
                burst_length=req.burst_length,
                total_bytes=req.total_bytes,
                cmd_latency=req.cmd_latency,
                data_latency=req.data_latency,
                transaction_latency=req.transaction_latency,
                # 保序字段
                src_dest_order_id=req.src_dest_order_id,
                packet_category=req.packet_category,
                # cycle数据字段
                cmd_entry_cake0_cycle=req.cmd_entry_cake0_cycle,
                cmd_entry_noc_from_cake0_cycle=req.cmd_entry_noc_from_cake0_cycle,
                cmd_entry_noc_from_cake1_cycle=req.cmd_entry_noc_from_cake1_cycle,
                cmd_received_by_cake0_cycle=req.cmd_received_by_cake0_cycle,
                cmd_received_by_cake1_cycle=req.cmd_received_by_cake1_cycle,
                data_entry_noc_from_cake0_cycle=req.data_entry_noc_from_cake0_cycle,
                data_entry_noc_from_cake1_cycle=req.data_entry_noc_from_cake1_cycle,
                data_received_complete_cycle=req.data_received_complete_cycle,
                rsp_entry_network_cycle=req.rsp_entry_network_cycle,
            )
            filtered_requests.append(temp_req)

        if not filtered_requests:
            return BandwidthMetrics(
                unweighted_bandwidth=0.0,
                weighted_bandwidth=0.0,
                working_intervals=[],
                total_working_time=0,
                network_start_time=0,
                network_end_time=0,
                total_bytes=0,
                total_requests=0,
            )

        # 计算工作区间
        working_intervals = self.time_interval_calculator.calculate_working_intervals(filtered_requests)

        # 网络工作时间窗口
        network_start = min(req.start_time for req in filtered_requests)
        network_end = max(req.end_time for req in filtered_requests)
        total_network_time = network_end - network_start

        # 总工作时间和总字节数
        total_working_time = sum(interval.duration for interval in working_intervals)
        total_bytes = sum(req.total_bytes for req in filtered_requests)

        # 计算非加权带宽：总数据量 / 网络总时间
        if total_network_time > 0 and DataValidator.is_valid_number(total_bytes) and total_bytes > 0:
            unweighted_bandwidth = total_bytes / total_network_time
        else:
            unweighted_bandwidth = 0.0

        # 计算加权带宽：各区间带宽按flit数量加权平均
        if working_intervals:
            total_weighted_bandwidth = 0.0
            total_weight = 0

            for interval in working_intervals:
                weight = interval.flit_count  # 权重是工作时间段的flit数量
                bandwidth = interval.bandwidth_bytes_per_ns  # bytes

                # 验证带宽值有效性
                if DataValidator.is_valid_number(bandwidth) and DataValidator.is_valid_number(weight) and weight > 0:
                    total_weighted_bandwidth += bandwidth * weight
                    total_weight += weight

            weighted_bandwidth = (total_weighted_bandwidth / total_weight) if total_weight > 0 else 0.0

        else:
            weighted_bandwidth = 0.0

        # 最终验证计算结果
        unweighted_bandwidth = DataValidator.sanitize_value(unweighted_bandwidth, 0.0)
        weighted_bandwidth = DataValidator.sanitize_value(weighted_bandwidth, 0.0)

        return BandwidthMetrics(
            unweighted_bandwidth=unweighted_bandwidth,
            weighted_bandwidth=weighted_bandwidth,
            working_intervals=working_intervals,
            total_working_time=total_working_time,
            network_start_time=network_start,
            network_end_time=network_end,
            total_bytes=total_bytes,
            total_requests=len(filtered_requests),
        )
