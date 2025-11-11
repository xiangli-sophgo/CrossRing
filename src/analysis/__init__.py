"""
CrossRing 分析模块

提供模块化的结果分析功能,包括:
- 数据类定义和常量
- 核心计算器(数据验证、时间区间计算、带宽计算)
- 数据收集器(请求收集、延迟统计、绕环统计)
- 导出器(CSV、JSON、报告生成)
- 可视化器(流量图、带宽曲线、热图)
"""

# 数据类和常量
from .analyzers import (
    RequestInfo,
    WorkingInterval,
    BandwidthMetrics,
    PortBandwidthMetrics,
    IP_COLOR_MAP,
    RN_TYPES,
    SN_TYPES,
    FLIT_SIZE_BYTES,
    MAX_ROWS,
    MAX_BANDWIDTH_NORMALIZATION,
    AXI_CHANNEL_DESCRIPTIONS,
    SingleDieAnalyzer,
)

# D2D相关类从d2d_analyzer导入
from .d2d_analyzer import (
    D2DRequestInfo,
    D2DBandwidthStats,
    D2DAnalyzer,
)

# 核心计算器
from .core_calculators import (
    DataValidator,
    TimeIntervalCalculator,
    BandwidthCalculator,
)

# 数据收集器
from .data_collectors import (
    RequestCollector,
    LatencyStatsCollector,
    CircuitStatsCollector,
)

# 导出器
from .exporters import (
    CSVExporter,
    ReportGenerator,
    JSONExporter,
)

# 可视化器
from .visualizers import (
    FlowGraphRenderer,
    BandwidthPlotter,
    HeatmapDrawer,
    IPInfoBoxDrawer,
)

__all__ = [
    # 数据类
    "RequestInfo",
    "D2DRequestInfo",
    "WorkingInterval",
    "BandwidthMetrics",
    "PortBandwidthMetrics",
    "D2DBandwidthStats",
    # 常量
    "IP_COLOR_MAP",
    "RN_TYPES",
    "SN_TYPES",
    "FLIT_SIZE_BYTES",
    "MAX_ROWS",
    "MAX_BANDWIDTH_NORMALIZATION",
    "AXI_CHANNEL_DESCRIPTIONS",
    # 分析器框架
    "SingleDieAnalyzer",
    "D2DAnalyzer",
    # 核心计算器
    "DataValidator",
    "TimeIntervalCalculator",
    "BandwidthCalculator",
    # 数据收集器
    "RequestCollector",
    "LatencyStatsCollector",
    "CircuitStatsCollector",
    # 导出器
    "CSVExporter",
    "ReportGenerator",
    "JSONExporter",
    # 可视化器
    "FlowGraphRenderer",
    "BandwidthPlotter",
    "HeatmapDrawer",
    "IPInfoBoxDrawer",
]
