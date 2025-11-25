"""
仿真结果数据库 - SQLAlchemy ORM模型定义

支持 NoC 和 D2D 两种仿真类型，使用 JSON 存储配置参数
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
    Index,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Experiment(Base):
    """实验元数据表"""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)  # 用户输入的实验名称
    experiment_type = Column(String(20), default="noc")  # "noc" 或 "d2d"
    created_at = Column(DateTime, default=datetime.now)
    description = Column(Text)  # 实验描述/调试信息

    # 仿真配置
    config_path = Column(String(512))  # 配置文件路径
    topo_type = Column(String(50))  # 拓扑类型 (如 "5x4")
    traffic_files = Column(Text)  # traffic文件列表 (JSON数组)
    traffic_weights = Column(Text)  # traffic权重 (JSON数组)
    simulation_time = Column(Integer)  # 仿真时间
    n_repeats = Column(Integer, default=1)  # 重复次数
    n_jobs = Column(Integer)  # 并行作业数

    # 状态追踪
    status = Column(String(50), default="running")  # running/completed/failed/interrupted
    total_combinations = Column(Integer)  # 总参数组合数
    completed_combinations = Column(Integer, default=0)  # 已完成组合数
    best_performance = Column(Float)  # 最佳性能 (GB/s)

    # 元数据
    git_commit = Column(String(50))  # Git commit hash (可选)
    notes = Column(Text)  # 额外备注

    # 关系 - NoC 结果
    noc_results = relationship(
        "NocResult", back_populates="experiment", cascade="all, delete-orphan"
    )

    # 关系 - D2D 结果
    d2d_results = relationship(
        "D2DResult", back_populates="experiment", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_experiments_status", "status"),
        Index("idx_experiments_created", "created_at"),
        Index("idx_experiments_type", "experiment_type"),
    )


class NocResult(Base):
    """NoC 仿真结果表"""

    __tablename__ = "noc_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    # 配置参数 - JSON格式（FIFO/ETag/延迟等所有配置）
    config_params = Column(JSON)

    # 核心结果指标
    performance = Column(Float, nullable=False)  # 主要性能指标 (加权带宽 GB/s)

    # 详细结果 - JSON格式
    result_details = Column(JSON)  # 带宽、延迟等统计数据

    # 错误信息
    error = Column(Text)

    # 关系
    experiment = relationship("Experiment", back_populates="noc_results")

    __table_args__ = (
        Index("idx_noc_results_experiment", "experiment_id"),
        Index("idx_noc_results_performance", "performance"),
    )


class D2DResult(Base):
    """D2D 仿真结果表"""

    __tablename__ = "d2d_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    # 配置参数 - JSON格式（D2D延迟/Tracker/资源等所有配置）
    config_params = Column(JSON)

    # 核心结果指标
    performance = Column(Float, nullable=False)  # 主要性能指标 (加权带宽 GB/s)

    # 详细结果 - JSON格式
    result_details = Column(JSON)  # 带宽、延迟等统计数据

    # 错误信息
    error = Column(Text)

    # 关系
    experiment = relationship("Experiment", back_populates="d2d_results")

    __table_args__ = (
        Index("idx_d2d_results_experiment", "experiment_id"),
        Index("idx_d2d_results_performance", "performance"),
    )


# 为了向后兼容，保留旧的别名
OptimizationResult = NocResult
