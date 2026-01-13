"""NoC组件模块"""

from .network import Network
from .ip_interface import IPInterface, RingIPInterface
from .cross_point import CrossPoint

__all__ = [
    "Network",
    "IPInterface",
    "RingIPInterface",
    "CrossPoint",
]
