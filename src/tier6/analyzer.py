"""
Tier6+ 分析器

提供多层级网络的统一分析接口
"""

from typing import Dict, List, Optional, Any

from .base import (
    HierarchyLevel,
    HierarchicalModel,
    LatencyResult,
    BandwidthResult,
    TrafficFlow,
)
from .layers import DieModel, ChipModel, BoardModel, ServerModel, PodModel
from .math_models import ScalingModel


class Tier6Analyzer:
    """
    Tier6+ 多层级分析器

    提供:
    - 层级构建
    - 延迟分析
    - 带宽瓶颈识别
    - 规模扩展分析
    """

    # 层级类映射
    LEVEL_CLASSES = {
        HierarchyLevel.DIE: DieModel,
        HierarchyLevel.CHIP: ChipModel,
        HierarchyLevel.BOARD: BoardModel,
        HierarchyLevel.SERVER: ServerModel,
        HierarchyLevel.POD: PodModel,
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化分析器

        Args:
            config: 配置字典，可包含各层级配置
        """
        self.config = config or {}
        self.root_model: Optional[HierarchicalModel] = None

    def build_hierarchy(
        self,
        top_level: str = "pod",
        model_id: str = "tier6_root"
    ) -> HierarchicalModel:
        """
        构建层级结构

        Args:
            top_level: 顶层类型 ('die', 'chip', 'board', 'server', 'pod')
            model_id: 根模型ID

        Returns:
            构建的层级模型
        """
        level = HierarchyLevel(top_level)
        level_class = self.LEVEL_CLASSES.get(level)

        if not level_class:
            raise ValueError(f"不支持的层级类型: {top_level}")

        # 获取该层级的配置
        level_config = self.config.get(top_level, {})
        self.root_model = level_class(model_id, level_config)

        return self.root_model

    def analyze(self, traffic_flows: List[TrafficFlow]) -> Dict[str, Any]:
        """
        执行完整分析

        Args:
            traffic_flows: 流量流列表

        Returns:
            分析结果字典
        """
        if not self.root_model:
            raise RuntimeError("请先调用 build_hierarchy() 构建层级")

        # 计算总延迟
        total_latency = self.root_model.get_total_latency(traffic_flows)

        # 各层级延迟分解
        latency_breakdown = self.root_model.get_latency_breakdown(traffic_flows)

        # 带宽瓶颈
        bottleneck = self.root_model.find_bandwidth_bottleneck(traffic_flows)

        # 汇总结果
        return {
            "total_latency_ns": total_latency.total_latency_ns,
            "latency_breakdown": {
                level: {
                    "propagation_ns": result.propagation_latency_ns,
                    "queuing_ns": result.queuing_latency_ns,
                    "processing_ns": result.processing_latency_ns,
                    "transmission_ns": result.transmission_latency_ns,
                    "total_ns": result.total_latency_ns,
                }
                for level, result in latency_breakdown.items()
            },
            "bottleneck": {
                "location": bottleneck[0] if bottleneck else None,
                "utilization": bottleneck[1].utilization if bottleneck else 0.0,
                "bandwidth_gbps": bottleneck[1].theoretical_bandwidth_gbps if bottleneck else 0.0,
            } if bottleneck else None,
            "traffic_summary": self._summarize_traffic(traffic_flows),
        }

    def analyze_latency(self, traffic_flows: List[TrafficFlow]) -> Dict[str, LatencyResult]:
        """
        延迟分析

        Args:
            traffic_flows: 流量流列表

        Returns:
            各层级延迟结果
        """
        if not self.root_model:
            raise RuntimeError("请先调用 build_hierarchy() 构建层级")

        return self.root_model.get_latency_breakdown(traffic_flows)

    def analyze_bandwidth(self, traffic_flows: List[TrafficFlow]) -> Dict[str, BandwidthResult]:
        """
        带宽分析

        Args:
            traffic_flows: 流量流列表

        Returns:
            各层级带宽结果
        """
        if not self.root_model:
            raise RuntimeError("请先调用 build_hierarchy() 构建层级")

        results = {}
        self._collect_bandwidth_results(self.root_model, traffic_flows, results)
        return results

    def _collect_bandwidth_results(
        self,
        model: HierarchicalModel,
        flows: List[TrafficFlow],
        results: Dict
    ):
        """递归收集带宽结果"""
        results[f"{model.level.value}:{model.model_id}"] = model.calculate_bandwidth(flows)
        for child in model.children.values():
            self._collect_bandwidth_results(child, flows, results)

    def find_bottleneck(self, traffic_flows: List[TrafficFlow]) -> Optional[Dict]:
        """
        找出系统瓶颈

        Args:
            traffic_flows: 流量流列表

        Returns:
            瓶颈信息字典
        """
        if not self.root_model:
            raise RuntimeError("请先调用 build_hierarchy() 构建层级")

        bottleneck = self.root_model.find_bandwidth_bottleneck(traffic_flows)
        if not bottleneck:
            return None

        location, bw_result = bottleneck
        return {
            "location": location,
            "utilization": bw_result.utilization,
            "theoretical_bandwidth_gbps": bw_result.theoretical_bandwidth_gbps,
            "effective_bandwidth_gbps": bw_result.effective_bandwidth_gbps,
            "is_critical": bw_result.utilization > 0.9,
        }

    def analyze_scaling(
        self,
        base_traffic_flows: List[TrafficFlow],
        scale_factors: List[int] = None
    ) -> Dict[str, Any]:
        """
        规模扩展分析

        Args:
            base_traffic_flows: 基准流量流
            scale_factors: 扩展因子列表

        Returns:
            扩展分析结果
        """
        if not self.root_model:
            raise RuntimeError("请先调用 build_hierarchy() 构建层级")

        if scale_factors is None:
            scale_factors = [1, 2, 4, 8, 16, 32]

        # 基准测量
        base_result = self.analyze(base_traffic_flows)
        base_latency = base_result["total_latency_ns"]
        base_throughput = sum(f.bandwidth_gbps for f in base_traffic_flows)

        results = {
            "scale_factors": scale_factors,
            "latency": {"ideal": [], "predicted": []},
            "throughput": {"ideal": [], "predicted": []},
            "efficiency": [],
        }

        for factor in scale_factors:
            # 理想扩展
            ideal_latency = base_latency  # 延迟理想不变
            ideal_throughput = ScalingModel.ideal_scaling(base_throughput, factor)

            # 预测扩展 (使用 Amdahl 定律，假设 90% 可并行)
            predicted_latency = ScalingModel.amdahl_scaling(
                base_latency, parallel_ratio=0.9, scale_factor=factor
            )
            predicted_throughput = ScalingModel.gustafson_scaling(
                base_throughput, parallel_ratio=0.9, scale_factor=factor
            )

            results["latency"]["ideal"].append(ideal_latency)
            results["latency"]["predicted"].append(predicted_latency)
            results["throughput"]["ideal"].append(ideal_throughput)
            results["throughput"]["predicted"].append(predicted_throughput)

            # 扩展效率
            efficiency = ScalingModel.scaling_efficiency(
                predicted_throughput / base_throughput, factor
            )
            results["efficiency"].append(efficiency)

        return results

    def _summarize_traffic(self, flows: List[TrafficFlow]) -> Dict:
        """汇总流量信息"""
        if not flows:
            return {"total_flows": 0}

        return {
            "total_flows": len(flows),
            "total_bandwidth_gbps": sum(f.bandwidth_gbps for f in flows),
            "avg_burst_size_bytes": sum(f.burst_size_bytes for f in flows) / len(flows),
            "total_request_rate": sum(f.request_rate_per_sec for f in flows),
        }

    def get_hierarchy_structure(self) -> Dict:
        """获取层级结构"""
        if not self.root_model:
            return {}
        return self.root_model.to_dict()

    def print_summary(self, traffic_flows: List[TrafficFlow]):
        """打印分析摘要"""
        results = self.analyze(traffic_flows)

        print("=" * 60)
        print("Tier6+ 多层级网络分析报告")
        print("=" * 60)

        print(f"\n总延迟: {results['total_latency_ns']:.2f} ns")

        print("\n延迟分解:")
        print("-" * 40)
        for level, breakdown in results["latency_breakdown"].items():
            pct = breakdown["total_ns"] / results["total_latency_ns"] * 100 if results["total_latency_ns"] > 0 else 0
            print(f"  {level:10s}: {breakdown['total_ns']:8.2f} ns ({pct:5.1f}%)")

        if results["bottleneck"]:
            print(f"\n瓶颈位置: {results['bottleneck']['location']}")
            print(f"  利用率: {results['bottleneck']['utilization']*100:.1f}%")
            print(f"  带宽: {results['bottleneck']['bandwidth_gbps']:.1f} GB/s")

        print("\n流量统计:")
        print("-" * 40)
        ts = results["traffic_summary"]
        print(f"  流量数: {ts['total_flows']}")
        print(f"  总带宽: {ts['total_bandwidth_gbps']:.2f} GB/s")

        print("=" * 60)
