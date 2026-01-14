"""
KCIN 基类模块

提供 KCIN v1/v2 共享的基类组件和工具。
"""

from .config import KCINConfigBase
from .routing_strategies import create_routing_strategy, RoutingStrategy
from .channel_selector import ChannelSelector
from .topology_utils import (
    create_adjacency_matrix,
    find_shortest_paths,
    throughput_cal,
)
from .traffic_scheduler import TrafficScheduler

__all__ = [
    # 配置基类
    "KCINConfigBase",
    # 路由策略
    "create_routing_strategy",
    "RoutingStrategy",
    # 通道选择
    "ChannelSelector",
    # 拓扑工具
    "create_adjacency_matrix",
    "find_shortest_paths",
    "throughput_cal",
    # 流量调度
    "TrafficScheduler",
]
