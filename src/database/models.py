"""
仿真结果数据库 - SQLAlchemy ORM模型定义

支持 KCIN 和 DCIN 两种仿真类型，使用 JSON 存储配置参数
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
    LargeBinary,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Experiment(Base):
    """实验元数据表"""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)  # 用户输入的实验名称
    experiment_type = Column(String(20), default="kcin")  # "kcin" 或 "dcin"
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

    # 关系 - KCIN 结果
    kcin_results = relationship(
        "KcinResult", back_populates="experiment", cascade="all, delete-orphan"
    )

    # 关系 - DCIN 结果
    dcin_results = relationship(
        "DcinResult", back_populates="experiment", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_experiments_status", "status"),
        Index("idx_experiments_created", "created_at"),
        Index("idx_experiments_type", "experiment_type"),
    )


class KcinResult(Base):
    """KCIN 仿真结果表"""

    __tablename__ = "kcin_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    # 配置参数 - JSON格式（FIFO/ETag/延迟等所有配置）
    config_params = Column(JSON)

    # 核心结果指标
    performance = Column(Float, nullable=False)  # 主要性能指标 (加权带宽 GB/s)

    # 详细结果 - JSON格式
    result_details = Column(JSON)  # 带宽、延迟等统计数据

    # 结果文件
    result_html = Column(Text)  # HTML报告内容
    result_files = Column(Text)  # 结果文件路径列表 (JSON数组)

    # 错误信息
    error = Column(Text)

    # 关系
    experiment = relationship("Experiment", back_populates="kcin_results")

    __table_args__ = (
        Index("idx_kcin_results_experiment", "experiment_id"),
        Index("idx_kcin_results_performance", "performance"),
    )


class DcinResult(Base):
    """DCIN 仿真结果表"""

    __tablename__ = "dcin_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    # 配置参数 - JSON格式（DCIN延迟/Tracker/资源等所有配置）
    config_params = Column(JSON)

    # 核心结果指标
    performance = Column(Float, nullable=False)  # 主要性能指标 (加权带宽 GB/s)

    # 详细结果 - JSON格式
    result_details = Column(JSON)  # 带宽、延迟等统计数据

    # 结果文件
    result_html = Column(Text)  # HTML报告内容
    result_files = Column(Text)  # 结果文件路径列表 (JSON数组)

    # 错误信息
    error = Column(Text)

    # 关系
    experiment = relationship("Experiment", back_populates="dcin_results")

    __table_args__ = (
        Index("idx_dcin_results_experiment", "experiment_id"),
        Index("idx_dcin_results_performance", "performance"),
    )


class ResultFile(Base):
    """结果文件存储表 - 存储文件内容到数据库"""

    __tablename__ = "result_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(Integer, nullable=False)  # 对应的结果ID
    result_type = Column(String(20), nullable=False)  # "kcin" 或 "dcin"
    file_name = Column(String(255), nullable=False)  # 文件名
    file_path = Column(String(512))  # 原始文件路径（用于参考）
    mime_type = Column(String(100))  # MIME类型
    file_size = Column(Integer)  # 文件大小（字节）
    file_content = Column(LargeBinary)  # 文件二进制内容
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_result_files_result", "result_id", "result_type"),
    )


class AnalysisChart(Base):
    """分析图表配置表 - 保存用户配置的分析图表"""

    __tablename__ = "analysis_charts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # 图表基本信息
    name = Column(String(255), nullable=False)  # 图表名称
    chart_type = Column(String(50), nullable=False)  # 图表类型: line/heatmap/sensitivity

    # 图表配置 - JSON格式
    config = Column(JSON, nullable=False)  # 存储图表配置（参数、指标等）

    # 排序
    sort_order = Column(Integer, default=0)  # 显示顺序

    __table_args__ = (
        Index("idx_analysis_charts_experiment", "experiment_id"),
    )
