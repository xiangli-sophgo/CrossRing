"""
KCIN 模块入口 - 支持 v1/v2 版本选择

使用方式：
1. 指定版本导入：
   from src.kcin.v1 import V1Config, BaseModel, Network
   from src.kcin.v2 import V2Config, BaseModel, Network

2. 通过配置选择版本：
   from src.kcin.v1.config import V1Config
   config = V1Config("path/to/config.yaml")
"""

# 基类模块（包含共享工具）
from .base import (
    KCINConfigBase,
    create_routing_strategy,
    RoutingStrategy,
    ChannelSelector,
    create_adjacency_matrix,
    find_shortest_paths,
    throughput_cal,
)


def get_kcin_version(config):
    """
    根据配置返回对应版本的 KCIN 模块

    Args:
        config: 配置对象，需包含 KCIN_VERSION 属性

    Returns:
        对应版本的 kcin 模块
    """
    version = getattr(config, "KCIN_VERSION", "v1")
    if version == "v2":
        from src.kcin import v2
        return v2
    else:
        from src.kcin import v1
        return v1


__all__ = [
    # 基类
    "KCINConfigBase",
    # 共享工具
    "create_routing_strategy",
    "RoutingStrategy",
    "ChannelSelector",
    "create_adjacency_matrix",
    "find_shortest_paths",
    "throughput_cal",
    # 版本选择函数
    "get_kcin_version",
]
