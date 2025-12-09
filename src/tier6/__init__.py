"""
Tier6+ 多层级网络数学建模框架

支持从 Die 到 Pod 的多层级网络性能分析:
- Die: NoC 内部
- Chip: D2D (Die-to-Die)
- Board: C2C (Chip-to-Chip)
- Server: B2B (Board-to-Board)
- Pod: S2S (Server-to-Server)
"""

from .base import (
    HierarchyLevel,
    ConnectionType,
    LatencyResult,
    BandwidthResult,
    TrafficFlow,
    HierarchicalModel,
)
from .math_models import QueuingModel, CongestionModel, BandwidthModel
from .analyzer import Tier6Analyzer

__all__ = [
    "HierarchyLevel",
    "ConnectionType",
    "LatencyResult",
    "BandwidthResult",
    "TrafficFlow",
    "HierarchicalModel",
    "QueuingModel",
    "CongestionModel",
    "BandwidthModel",
    "Tier6Analyzer",
]
