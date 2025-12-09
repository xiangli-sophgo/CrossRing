"""
Tier6+ 多层级网络建模示例

演示如何使用 Tier6+ 框架进行多层级网络性能分析
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tier6 import Tier6Analyzer, TrafficFlow
from src.tier6.config_loader import Tier6Config


def example_basic():
    """基础示例：使用默认配置"""
    print("\n" + "=" * 60)
    print("示例 1: 基础用法 - 默认配置")
    print("=" * 60)

    # 创建分析器
    analyzer = Tier6Analyzer()

    # 构建 Pod 级层级结构
    analyzer.build_hierarchy(top_level="pod", model_id="example_pod")

    # 定义流量
    traffic_flows = [
        TrafficFlow(
            flow_id="flow_1",
            source_id="server_0",
            destination_id="server_1",
            bandwidth_gbps=10.0,
            request_rate_per_sec=1000,
            burst_size_bytes=512,
        ),
        TrafficFlow(
            flow_id="flow_2",
            source_id="cross_server_0",
            destination_id="cross_server_1",
            bandwidth_gbps=20.0,
            request_rate_per_sec=2000,
        ),
    ]

    # 执行分析
    analyzer.print_summary(traffic_flows)


def example_with_config():
    """使用配置文件"""
    print("\n" + "=" * 60)
    print("示例 2: 使用配置文件")
    print("=" * 60)

    # 加载配置
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "tier6", "default.yaml"
    )

    config = Tier6Config(config_path)
    print(f"加载配置: {config_path}")

    # 使用配置创建分析器
    analyzer = Tier6Analyzer(config.to_dict())
    analyzer.build_hierarchy(top_level="chip", model_id="example_chip")

    # 定义跨 Die 流量
    traffic_flows = [
        TrafficFlow(
            flow_id="d2d_flow_1",
            source_id="cross_die_0",
            destination_id="cross_die_1",
            bandwidth_gbps=50.0,
            request_rate_per_sec=5000,
        ),
    ]

    analyzer.print_summary(traffic_flows)


def example_scaling_analysis():
    """规模扩展分析"""
    print("\n" + "=" * 60)
    print("示例 3: 规模扩展分析")
    print("=" * 60)

    analyzer = Tier6Analyzer()
    analyzer.build_hierarchy(top_level="server")

    # 基准流量
    base_flows = [
        TrafficFlow(
            flow_id="base_flow",
            source_id="cross_board_0",
            destination_id="cross_board_1",
            bandwidth_gbps=10.0,
        ),
    ]

    # 规模扩展分析
    scaling_results = analyzer.analyze_scaling(
        base_flows,
        scale_factors=[1, 2, 4, 8, 16]
    )

    print("\n规模扩展分析结果:")
    print("-" * 40)
    print(f"{'扩展因子':^10} {'预测延迟(ns)':^15} {'预测吞吐(GB/s)':^15} {'效率':^10}")
    print("-" * 40)

    for i, factor in enumerate(scaling_results["scale_factors"]):
        latency = scaling_results["latency"]["predicted"][i]
        throughput = scaling_results["throughput"]["predicted"][i]
        efficiency = scaling_results["efficiency"][i]
        print(f"{factor:^10} {latency:^15.2f} {throughput:^15.2f} {efficiency:^10.2%}")


def example_bottleneck_analysis():
    """瓶颈分析"""
    print("\n" + "=" * 60)
    print("示例 4: 瓶颈分析")
    print("=" * 60)

    # 自定义高负载配置
    config = {
        "pod": {
            "num_servers": 8,
            "s2s_bandwidth_gbps": 50.0,  # 降低带宽制造瓶颈
        },
        "server": {
            "num_boards": 4,
            "b2b_bandwidth_gbps": 20.0,
        },
    }

    analyzer = Tier6Analyzer(config)
    analyzer.build_hierarchy(top_level="pod")

    # 高负载流量
    traffic_flows = [
        TrafficFlow(
            flow_id=f"heavy_flow_{i}",
            source_id=f"cross_server_{i}",
            destination_id=f"cross_server_{i+1}",
            bandwidth_gbps=15.0,
        )
        for i in range(5)
    ]

    # 分析
    results = analyzer.analyze(traffic_flows)
    bottleneck = analyzer.find_bottleneck(traffic_flows)

    print(f"\n总流量: {sum(f.bandwidth_gbps for f in traffic_flows):.1f} GB/s")
    print(f"总延迟: {results['total_latency_ns']:.2f} ns")

    if bottleneck:
        print(f"\n瓶颈检测:")
        print(f"  位置: {bottleneck['location']}")
        print(f"  利用率: {bottleneck['utilization']*100:.1f}%")
        print(f"  理论带宽: {bottleneck['theoretical_bandwidth_gbps']:.1f} GB/s")
        print(f"  是否临界: {'是' if bottleneck['is_critical'] else '否'}")


def example_hierarchy_structure():
    """查看层级结构"""
    print("\n" + "=" * 60)
    print("示例 5: 层级结构")
    print("=" * 60)

    analyzer = Tier6Analyzer()
    analyzer.build_hierarchy(top_level="board")

    structure = analyzer.get_hierarchy_structure()

    def print_structure(s, indent=0):
        prefix = "  " * indent
        print(f"{prefix}[{s['level']}] {s['id']}")
        print(f"{prefix}  延迟: {s['latency_ns']} ns, 带宽: {s['bandwidth_gbps']} GB/s")

        if s.get('connections'):
            print(f"{prefix}  连接: {len(s['connections'])} 个")

        for child_id, child in s.get('children', {}).items():
            print_structure(child, indent + 1)

    print_structure(structure)


if __name__ == "__main__":
    example_basic()
    example_with_config()
    example_scaling_analysis()
    example_bottleneck_analysis()
    example_hierarchy_structure()

    print("\n" + "=" * 60)
    print("所有示例执行完成!")
    print("=" * 60)
