"""NoC v1 模块 - IQ/RB/EQ 架构"""

from .config import V1Config
from .base_model import BaseModel
from .REQ_RSP import REQ_RSP_model

# NoC组件
from .components import (
    Network,
    LinkSlot,
    IPInterface,
    RingIPInterface,
    CrossPoint,
    DualChannelIPInterface,
)

__all__ = [
    # 配置
    "V1Config",
    # 模型
    "BaseModel",
    "REQ_RSP_model",
    # 组件
    "Network",
    "LinkSlot",
    "IPInterface",
    "RingIPInterface",
    "CrossPoint",
    "DualChannelIPInterface",
]
