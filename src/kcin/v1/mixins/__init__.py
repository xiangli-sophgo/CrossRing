"""
BaseModel Mixins模块
将BaseModel的功能拆分为多个Mixin类，便于维护和扩展
"""

from src.kcin.v1.mixins.stats_mixin import StatsMixin
from src.kcin.v1.mixins.dataflow_mixin import DataflowMixin

__all__ = ["StatsMixin", "DataflowMixin"]
