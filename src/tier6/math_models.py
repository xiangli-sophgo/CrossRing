"""
Tier6+ 数学模型

包含:
- QueuingModel: 排队论模型 (M/M/1, M/G/1, M/M/c)
- CongestionModel: 拥塞延迟模型
- BandwidthModel: 带宽瓶颈分析
"""

import math
from typing import List, Optional, Tuple

from .base import TrafficFlow, BandwidthResult


class QueuingModel:
    """排队论模型工具类"""

    @staticmethod
    def mm1_latency(arrival_rate: float, service_rate: float) -> float:
        """
        M/M/1 队列平均等待时间

        Args:
            arrival_rate: 到达率 λ (请求/秒)
            service_rate: 服务率 μ (请求/秒)

        Returns:
            平均等待时间 (秒)，系统不稳定时返回 inf
        """
        if service_rate <= 0:
            raise ValueError("service_rate 必须大于 0")
        if arrival_rate >= service_rate:
            return float('inf')  # 不稳定系统
        return 1.0 / (service_rate - arrival_rate)

    @staticmethod
    def mm1_queue_length(arrival_rate: float, service_rate: float) -> float:
        """
        M/M/1 队列平均队列长度

        Args:
            arrival_rate: 到达率 λ
            service_rate: 服务率 μ

        Returns:
            平均队列长度
        """
        if service_rate <= 0:
            raise ValueError("service_rate 必须大于 0")
        rho = arrival_rate / service_rate
        if rho >= 1:
            return float('inf')
        return rho / (1 - rho)

    @staticmethod
    def mg1_latency(
        arrival_rate: float,
        service_rate: float,
        service_cv: float = 1.0
    ) -> float:
        """
        M/G/1 队列平均等待时间 (Pollaczek-Khinchin 公式)

        Args:
            arrival_rate: 到达率 λ
            service_rate: 服务率 μ
            service_cv: 服务时间变异系数 Cs = std/mean (默认1.0对应指数分布)

        Returns:
            平均等待时间 (秒)
        """
        if service_rate <= 0:
            raise ValueError("service_rate 必须大于 0")
        rho = arrival_rate / service_rate
        if rho >= 1:
            return float('inf')
        # Pollaczek-Khinchin: W = ρ / (2μ(1-ρ)) × (1 + Cs²)
        return (rho / (2 * service_rate * (1 - rho))) * (1 + service_cv ** 2)

    @staticmethod
    def mmc_latency(
        arrival_rate: float,
        service_rate: float,
        num_servers: int
    ) -> float:
        """
        M/M/c 队列平均等待时间 (多服务器)

        Args:
            arrival_rate: 总到达率 λ
            service_rate: 单服务器服务率 μ
            num_servers: 服务器数量 c

        Returns:
            平均等待时间 (秒)
        """
        if service_rate <= 0 or num_servers <= 0:
            raise ValueError("service_rate 和 num_servers 必须大于 0")

        c = num_servers
        rho = arrival_rate / (c * service_rate)  # 系统利用率

        if rho >= 1:
            return float('inf')

        a = arrival_rate / service_rate  # 提供负载

        # 计算 P0 (空闲概率)
        sum_term = sum((a ** k) / math.factorial(k) for k in range(c))
        last_term = (a ** c) / (math.factorial(c) * (1 - rho))
        p0 = 1.0 / (sum_term + last_term)

        # Erlang-C: 排队概率
        pq = ((a ** c) / (math.factorial(c) * (1 - rho))) * p0

        # 平均等待时间
        return pq / (c * service_rate - arrival_rate) + 1 / service_rate

    @staticmethod
    def utilization(arrival_rate: float, service_rate: float) -> float:
        """计算利用率 ρ = λ/μ"""
        if service_rate <= 0:
            raise ValueError("service_rate 必须大于 0")
        return arrival_rate / service_rate


class CongestionModel:
    """拥塞延迟模型"""

    @staticmethod
    def linear_degradation(
        utilization: float,
        threshold: float = 0.7,
        alpha: float = 2.0
    ) -> float:
        """
        线性性能退化模型

        当利用率超过阈值时，有效带宽线性下降

        Args:
            utilization: 当前利用率 (0-1)
            threshold: 开始退化的阈值 (默认0.7)
            alpha: 退化系数 (默认2.0)

        Returns:
            有效带宽比例 (0-1)
        """
        if utilization <= threshold:
            return 1.0
        degradation = alpha * (utilization - threshold)
        return max(0.1, 1.0 - degradation)

    @staticmethod
    def nonlinear_latency_increase(
        base_latency_ns: float,
        utilization: float
    ) -> float:
        """
        非线性延迟增长模型

        延迟随利用率按 1/(1-ρ) 增长

        Args:
            base_latency_ns: 基础延迟 (ns)
            utilization: 利用率 (0-1)

        Returns:
            实际延迟 (ns)
        """
        if utilization >= 1.0:
            return float('inf')
        if utilization < 0:
            utilization = 0
        return base_latency_ns / (1 - utilization)

    @staticmethod
    def exponential_latency(
        base_latency_ns: float,
        utilization: float,
        sensitivity: float = 5.0
    ) -> float:
        """
        指数延迟增长模型

        Args:
            base_latency_ns: 基础延迟 (ns)
            utilization: 利用率 (0-1)
            sensitivity: 敏感度系数

        Returns:
            实际延迟 (ns)
        """
        if utilization >= 1.0:
            return float('inf')
        return base_latency_ns * math.exp(sensitivity * utilization)


class BandwidthModel:
    """带宽计算模型"""

    @staticmethod
    def effective_bandwidth(
        theoretical_bw_gbps: float,
        utilization: float,
        overhead_ratio: float = 0.0
    ) -> float:
        """
        计算有效带宽

        Args:
            theoretical_bw_gbps: 理论带宽 (GB/s)
            utilization: 当前利用率 (0-1)
            overhead_ratio: 协议开销比例 (0-1)

        Returns:
            有效带宽 (GB/s)
        """
        usable_bw = theoretical_bw_gbps * (1 - overhead_ratio)
        return usable_bw * CongestionModel.linear_degradation(utilization)

    @staticmethod
    def calculate_utilization(
        traffic_flows: List[TrafficFlow],
        capacity_gbps: float
    ) -> float:
        """
        计算链路利用率

        Args:
            traffic_flows: 流量流列表
            capacity_gbps: 链路容量 (GB/s)

        Returns:
            利用率 (0-1)
        """
        if capacity_gbps <= 0:
            raise ValueError("capacity_gbps 必须大于 0")
        total_demand = sum(f.bandwidth_gbps for f in traffic_flows)
        return min(total_demand / capacity_gbps, 1.0)

    @staticmethod
    def find_bottleneck(
        link_utilizations: dict
    ) -> Optional[Tuple[str, float]]:
        """
        找出瓶颈链路

        Args:
            link_utilizations: {链路ID: 利用率} 字典

        Returns:
            (瓶颈链路ID, 利用率) 或 None
        """
        if not link_utilizations:
            return None

        max_link = max(link_utilizations.items(), key=lambda x: x[1])
        if max_link[1] > 0.7:  # 利用率超过70%认为是瓶颈
            return max_link
        return None

    @staticmethod
    def aggregate_bandwidth(
        traffic_flows: List[TrafficFlow],
        path_links: List[str],
        link_capacities: dict
    ) -> BandwidthResult:
        """
        聚合带宽分析

        Args:
            traffic_flows: 流量流列表
            path_links: 路径经过的链路列表
            link_capacities: {链路ID: 容量(GB/s)} 字典

        Returns:
            带宽分析结果
        """
        total_demand = sum(f.bandwidth_gbps for f in traffic_flows)

        # 计算各链路利用率
        link_utils = {}
        for link in path_links:
            capacity = link_capacities.get(link, float('inf'))
            link_utils[link] = total_demand / capacity if capacity > 0 else 1.0

        # 找瓶颈
        bottleneck = BandwidthModel.find_bottleneck(link_utils)

        # 有效带宽受限于最小容量链路
        min_capacity = min(link_capacities.get(l, float('inf')) for l in path_links)
        effective_bw = min(total_demand, min_capacity)

        return BandwidthResult(
            theoretical_bandwidth_gbps=min_capacity,
            effective_bandwidth_gbps=effective_bw,
            utilization=max(link_utils.values()) if link_utils else 0.0,
            bottleneck_link=bottleneck[0] if bottleneck else None
        )


class ScalingModel:
    """规模扩展分析模型"""

    @staticmethod
    def ideal_scaling(base_value: float, scale_factor: int) -> float:
        """理想线性扩展"""
        return base_value * scale_factor

    @staticmethod
    def amdahl_scaling(
        base_latency: float,
        parallel_ratio: float,
        scale_factor: int
    ) -> float:
        """
        Amdahl 定律扩展

        Args:
            base_latency: 基础延迟
            parallel_ratio: 可并行部分比例 (0-1)
            scale_factor: 扩展因子

        Returns:
            扩展后延迟
        """
        serial_part = base_latency * (1 - parallel_ratio)
        parallel_part = base_latency * parallel_ratio / scale_factor
        return serial_part + parallel_part

    @staticmethod
    def gustafson_scaling(
        base_throughput: float,
        parallel_ratio: float,
        scale_factor: int
    ) -> float:
        """
        Gustafson 定律扩展

        Args:
            base_throughput: 基础吞吐量
            parallel_ratio: 可并行部分比例 (0-1)
            scale_factor: 扩展因子

        Returns:
            扩展后吞吐量
        """
        return base_throughput * (
            (1 - parallel_ratio) + parallel_ratio * scale_factor
        )

    @staticmethod
    def scaling_efficiency(
        actual_speedup: float,
        scale_factor: int
    ) -> float:
        """
        计算扩展效率

        Args:
            actual_speedup: 实际加速比
            scale_factor: 扩展因子

        Returns:
            扩展效率 (0-1, 1为理想)
        """
        ideal_speedup = scale_factor
        return actual_speedup / ideal_speedup if ideal_speedup > 0 else 0.0
