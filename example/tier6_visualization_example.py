"""
Tier6+ 可视化示例

演示如何使用 Tier6+ 可视化模块生成交互式分析报告
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tier6 import Tier6Analyzer, TrafficFlow
from src.tier6.visualization import (
    HierarchyGraphRenderer,
    LatencyBreakdownChart,
    BandwidthBottleneckChart,
    ScalingAnalysisChart,
    Tier6ReportGenerator,
)


def example_hierarchy_visualization():
    """层级结构图示例"""
    print("\n" + "=" * 60)
    print("示例 1: 层级结构图可视化")
    print("=" * 60)

    # 创建分析器并构建层级
    analyzer = Tier6Analyzer()
    analyzer.build_hierarchy(top_level="board", model_id="example_board")

    # 获取层级结构
    hierarchy_data = analyzer.get_hierarchy_structure()

    # 渲染不同样式的层级图
    renderer = HierarchyGraphRenderer()

    # Treemap
    fig_treemap = renderer.render(hierarchy_data, style="treemap")
    fig_treemap.write_html("output/hierarchy_treemap.html")
    print("  生成: output/hierarchy_treemap.html")

    # Sunburst
    fig_sunburst = renderer.render(hierarchy_data, style="sunburst")
    fig_sunburst.write_html("output/hierarchy_sunburst.html")
    print("  生成: output/hierarchy_sunburst.html")

    # 嵌套盒子
    fig_nested = renderer.render(hierarchy_data, style="nested")
    fig_nested.write_html("output/hierarchy_nested.html")
    print("  生成: output/hierarchy_nested.html")

    # 网络拓扑
    fig_network = renderer.render(hierarchy_data, style="network")
    fig_network.write_html("output/hierarchy_network.html")
    print("  生成: output/hierarchy_network.html")


def example_latency_visualization():
    """延迟分解图示例"""
    print("\n" + "=" * 60)
    print("示例 2: 延迟分解图可视化")
    print("=" * 60)

    # 创建分析器
    analyzer = Tier6Analyzer()
    analyzer.build_hierarchy(top_level="pod")

    # 定义流量
    traffic_flows = [
        TrafficFlow(
            flow_id="flow_1",
            source_id="cross_server_0",
            destination_id="cross_server_1",
            bandwidth_gbps=20.0,
            request_rate_per_sec=2000,
        ),
        TrafficFlow(
            flow_id="flow_2",
            source_id="cross_board_0",
            destination_id="cross_board_1",
            bandwidth_gbps=30.0,
        ),
        TrafficFlow(
            flow_id="flow_3",
            source_id="cross_chip_0",
            destination_id="cross_chip_1",
            bandwidth_gbps=50.0,
        ),
        TrafficFlow(
            flow_id="flow_4",
            source_id="cross_die_0",
            destination_id="cross_die_1",
            bandwidth_gbps=80.0,
        ),
    ]

    # 执行分析
    results = analyzer.analyze(traffic_flows)
    latency_breakdown = results["latency_breakdown"]

    # 渲染延迟图表
    chart = LatencyBreakdownChart()

    # 堆叠条形图
    fig_stacked = chart.render(latency_breakdown, style="stacked")
    fig_stacked.write_html("output/latency_stacked.html")
    print("  生成: output/latency_stacked.html")

    # 瀑布图
    fig_waterfall = chart.render(latency_breakdown, style="waterfall")
    fig_waterfall.write_html("output/latency_waterfall.html")
    print("  生成: output/latency_waterfall.html")

    # 饼图
    fig_pie = chart.render(latency_breakdown, style="pie")
    fig_pie.write_html("output/latency_pie.html")
    print("  生成: output/latency_pie.html")

    # 组合图
    fig_combined = chart.render(latency_breakdown, style="combined")
    fig_combined.write_html("output/latency_combined.html")
    print("  生成: output/latency_combined.html")


def example_bandwidth_visualization():
    """带宽瓶颈图示例"""
    print("\n" + "=" * 60)
    print("示例 3: 带宽瓶颈图可视化")
    print("=" * 60)

    # 模拟带宽数据
    bandwidth_data = {
        "die:noc": {
            "utilization": 0.45,
            "theoretical_bandwidth_gbps": 512,
            "effective_bandwidth_gbps": 480,
        },
        "chip:d2d": {
            "utilization": 0.72,
            "theoretical_bandwidth_gbps": 192,
            "effective_bandwidth_gbps": 170,
        },
        "board:c2c": {
            "utilization": 0.88,  # 瓶颈
            "theoretical_bandwidth_gbps": 64,
            "effective_bandwidth_gbps": 55,
        },
        "server:b2b": {
            "utilization": 0.55,
            "theoretical_bandwidth_gbps": 32,
            "effective_bandwidth_gbps": 30,
        },
        "pod:s2s": {
            "utilization": 0.35,
            "theoretical_bandwidth_gbps": 100,
            "effective_bandwidth_gbps": 95,
        },
    }

    chart = BandwidthBottleneckChart()

    # 利用率条形图
    fig_bar = chart.render(bandwidth_data, style="bar")
    fig_bar.write_html("output/bandwidth_bar.html")
    print("  生成: output/bandwidth_bar.html")

    # 瓶颈指示器
    bottleneck = {
        "location": "board:c2c",
        "utilization": 0.88,
        "theoretical_bandwidth_gbps": 64,
        "effective_bandwidth_gbps": 55,
    }
    fig_indicator = chart.render(bandwidth_data, style="indicator", bottleneck=bottleneck)
    fig_indicator.write_html("output/bandwidth_indicator.html")
    print("  生成: output/bandwidth_indicator.html")


def example_scaling_visualization():
    """规模扩展曲线示例"""
    print("\n" + "=" * 60)
    print("示例 4: 规模扩展曲线可视化")
    print("=" * 60)

    # 创建分析器
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
    scaling_data = analyzer.analyze_scaling(base_flows, scale_factors=[1, 2, 4, 8, 16, 32])

    chart = ScalingAnalysisChart()

    # 延迟扩展
    fig_latency = chart.render(scaling_data, style="latency")
    fig_latency.write_html("output/scaling_latency.html")
    print("  生成: output/scaling_latency.html")

    # 吞吐量扩展
    fig_throughput = chart.render(scaling_data, style="throughput")
    fig_throughput.write_html("output/scaling_throughput.html")
    print("  生成: output/scaling_throughput.html")

    # 扩展效率
    fig_efficiency = chart.render(scaling_data, style="efficiency")
    fig_efficiency.write_html("output/scaling_efficiency.html")
    print("  生成: output/scaling_efficiency.html")

    # 组合图
    fig_combined = chart.render(scaling_data, style="combined")
    fig_combined.write_html("output/scaling_combined.html")
    print("  生成: output/scaling_combined.html")

    # Amdahl vs Gustafson
    fig_laws = chart.render_amdahl_vs_gustafson([1, 2, 4, 8, 16, 32, 64], parallel_ratio=0.9)
    fig_laws.write_html("output/scaling_laws.html")
    print("  生成: output/scaling_laws.html")


def example_full_report():
    """完整报告生成示例"""
    print("\n" + "=" * 60)
    print("示例 5: 完整报告生成")
    print("=" * 60)

    # 创建分析器
    analyzer = Tier6Analyzer({
        "pod": {"num_servers": 8, "s2s_bandwidth_gbps": 100},
        "server": {"num_boards": 4, "b2b_bandwidth_gbps": 32},
        "board": {"num_chips": 2, "c2c_bandwidth_gbps": 64},
        "chip": {"num_dies": 2, "d2d_bandwidth_gbps": 192},
    })
    analyzer.build_hierarchy(top_level="pod", model_id="datacenter_pod")

    # 定义多种流量
    traffic_flows = [
        TrafficFlow("flow_local", "die_0", "die_0", 100.0),
        TrafficFlow("flow_d2d", "cross_die_0", "cross_die_1", 80.0),
        TrafficFlow("flow_c2c", "cross_chip_0", "cross_chip_1", 50.0),
        TrafficFlow("flow_b2b", "cross_board_0", "cross_board_1", 25.0),
        TrafficFlow("flow_s2s_1", "cross_server_0", "cross_server_1", 15.0),
        TrafficFlow("flow_s2s_2", "cross_server_2", "cross_server_3", 15.0),
    ]

    # 生成完整报告
    report_generator = Tier6ReportGenerator()
    report_path = report_generator.generate_quick_report(
        analyzer=analyzer,
        traffic_flows=traffic_flows,
        output_path="output/tier6_full_report.html",
    )

    print(f"  完整报告已生成: {report_path}")


if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    # 运行所有示例
    example_hierarchy_visualization()
    example_latency_visualization()
    example_bandwidth_visualization()
    example_scaling_visualization()
    example_full_report()

    print("\n" + "=" * 60)
    print("所有可视化示例执行完成!")
    print("请查看 output/ 目录中的 HTML 文件")
    print("=" * 60)
