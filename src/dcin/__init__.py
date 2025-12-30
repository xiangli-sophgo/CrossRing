"""
DCIN (Die-to-Die Chip Interconnect Network) 模块

提供 Die-to-Die 通信的模拟功能。
"""

# 配置模块
from .config import DCINConfig

# 模型
from .d2d_model import D2D_Model

# 组件
from .components import (
    D2D_SN_Interface,
    D2D_RN_Interface,
    D2D_Sys,
)

__all__ = [
    # 配置
    "DCINConfig",
    # 模型
    "D2D_Model",
    # 组件
    "D2D_SN_Interface",
    "D2D_RN_Interface",
    "D2D_Sys",
]
