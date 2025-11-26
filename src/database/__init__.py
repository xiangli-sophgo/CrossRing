"""
仿真结果数据库模块

支持 KCIN 和 DCIN 两种仿真类型，提供：
- SQLite 数据库存储
- 实时记录仿真结果
- CSV 导入历史数据
- 统计分析和敏感性分析
"""

from .models import (
    Base,
    Experiment,
    KcinResult,
    DcinResult,
)
from .database import DatabaseManager, DEFAULT_DB_PATH
from .manager import ResultManager


__all__ = [
    # ORM模型
    "Base",
    "Experiment",
    "KcinResult",
    "DcinResult",
    # 数据库管理
    "DatabaseManager",
    "DEFAULT_DB_PATH",
    # 业务管理器
    "ResultManager",
]
