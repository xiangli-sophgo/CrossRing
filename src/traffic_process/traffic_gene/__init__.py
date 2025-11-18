"""
traffic_gene模块 - CrossRing流量生成工具包

模块说明:
- topology_visualizer: 拓扑可视化模块(含节点ID解析)
- config_manager: 配置管理模块(支持D2D配置)
- traffic_analyzer: 流量分析模块
- generation_engine: 流量生成引擎(单Die/D2D/拆分功能)

功能集成:
- 单Die流量生成(7字段格式)
- D2D流量生成(9字段格式)
- 流量拆分(按源IP拆分)
- 节点ID解析(支持范围表达式)
- 实时统计预估
- 结果可视化分析
"""

__all__ = [
    'topology_visualizer',
    'config_manager',
    'traffic_analyzer',
    'generation_engine'
]
