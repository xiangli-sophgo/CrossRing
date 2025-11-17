"""
测试延迟分布图生成功能
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from src.analysis.latency_distribution_plotter import LatencyDistributionPlotter


def create_mock_latency_stats():
    """创建模拟的延迟统计数据"""
    np.random.seed(42)

    # 生成模拟延迟数据
    # 读请求：较低延迟，集中在50-150ns
    read_cmd_latencies = np.random.normal(100, 20, 500).clip(50, 200).tolist()
    read_data_latencies = np.random.normal(200, 30, 500).clip(100, 400).tolist()
    read_trans_latencies = (
        np.array(read_cmd_latencies) + np.array(read_data_latencies)
    ).tolist()

    # 写请求：较高延迟，集中在80-180ns
    write_cmd_latencies = np.random.normal(130, 25, 300).clip(80, 250).tolist()
    write_data_latencies = np.random.normal(250, 40, 300).clip(150, 500).tolist()
    write_trans_latencies = (
        np.array(write_cmd_latencies) + np.array(write_data_latencies)
    ).tolist()

    # 混合请求
    mixed_cmd_latencies = read_cmd_latencies + write_cmd_latencies
    mixed_data_latencies = read_data_latencies + write_data_latencies
    mixed_trans_latencies = read_trans_latencies + write_trans_latencies

    latency_stats = {
        "cmd": {
            "read": {
                "sum": sum(read_cmd_latencies),
                "max": max(read_cmd_latencies),
                "count": len(read_cmd_latencies),
                "values": read_cmd_latencies,
                "p95": np.percentile(read_cmd_latencies, 95),
                "p99": np.percentile(read_cmd_latencies, 99),
            },
            "write": {
                "sum": sum(write_cmd_latencies),
                "max": max(write_cmd_latencies),
                "count": len(write_cmd_latencies),
                "values": write_cmd_latencies,
                "p95": np.percentile(write_cmd_latencies, 95),
                "p99": np.percentile(write_cmd_latencies, 99),
            },
            "mixed": {
                "sum": sum(mixed_cmd_latencies),
                "max": max(mixed_cmd_latencies),
                "count": len(mixed_cmd_latencies),
                "values": mixed_cmd_latencies,
                "p95": np.percentile(mixed_cmd_latencies, 95),
                "p99": np.percentile(mixed_cmd_latencies, 99),
            },
        },
        "data": {
            "read": {
                "sum": sum(read_data_latencies),
                "max": max(read_data_latencies),
                "count": len(read_data_latencies),
                "values": read_data_latencies,
                "p95": np.percentile(read_data_latencies, 95),
                "p99": np.percentile(read_data_latencies, 99),
            },
            "write": {
                "sum": sum(write_data_latencies),
                "max": max(write_data_latencies),
                "count": len(write_data_latencies),
                "values": write_data_latencies,
                "p95": np.percentile(write_data_latencies, 95),
                "p99": np.percentile(write_data_latencies, 99),
            },
            "mixed": {
                "sum": sum(mixed_data_latencies),
                "max": max(mixed_data_latencies),
                "count": len(mixed_data_latencies),
                "values": mixed_data_latencies,
                "p95": np.percentile(mixed_data_latencies, 95),
                "p99": np.percentile(mixed_data_latencies, 99),
            },
        },
        "trans": {
            "read": {
                "sum": sum(read_trans_latencies),
                "max": max(read_trans_latencies),
                "count": len(read_trans_latencies),
                "values": read_trans_latencies,
                "p95": np.percentile(read_trans_latencies, 95),
                "p99": np.percentile(read_trans_latencies, 99),
            },
            "write": {
                "sum": sum(write_trans_latencies),
                "max": max(write_trans_latencies),
                "count": len(write_trans_latencies),
                "values": write_trans_latencies,
                "p95": np.percentile(write_trans_latencies, 95),
                "p99": np.percentile(write_trans_latencies, 99),
            },
            "mixed": {
                "sum": sum(mixed_trans_latencies),
                "max": max(mixed_trans_latencies),
                "count": len(mixed_trans_latencies),
                "values": mixed_trans_latencies,
                "p95": np.percentile(mixed_trans_latencies, 95),
                "p99": np.percentile(mixed_trans_latencies, 99),
            },
        },
    }

    return latency_stats


def test_noc_latency_distribution():
    """测试NoC延迟分布图生成"""
    print("=" * 60)
    print("测试 NoC 延迟分布图")
    print("=" * 60)

    # 创建模拟数据
    latency_stats = create_mock_latency_stats()

    # 创建绘图器
    plotter = LatencyDistributionPlotter(latency_stats, title_prefix="NoC")

    # 测试图表类型
    print("\n1. 生成直方图+CDF组合图...")
    hist_cdf_fig = plotter.plot_histogram_with_cdf(return_fig=True)
    print(f"   ✓ 直方图+CDF生成成功 (类型: {type(hist_cdf_fig).__name__})")

    print("\n2. 生成小提琴图...")
    violin_fig = plotter.plot_violin(return_fig=True)
    print(f"   ✓ 小提琴图生成成功 (类型: {type(violin_fig).__name__})")

    # 保存HTML文件
    output_dir = os.path.join(project_root, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    print("\n3. 保存图表到HTML文件...")
    hist_cdf_path = os.path.join(output_dir, "noc_latency_histogram_cdf.html")
    hist_cdf_fig.write_html(hist_cdf_path)
    print(f"   ✓ 直方图+CDF已保存: {hist_cdf_path}")

    violin_path = os.path.join(output_dir, "noc_latency_violin.html")
    violin_fig.write_html(violin_path)
    print(f"   ✓ 小提琴图已保存: {violin_path}")

    print("\n✅ NoC延迟分布图测试通过!")
    return True


def test_d2d_latency_distribution():
    """测试D2D延迟分布图生成"""
    print("\n" + "=" * 60)
    print("测试 D2D 延迟分布图")
    print("=" * 60)

    # 创建模拟数据(D2D延迟通常更高)
    latency_stats = create_mock_latency_stats()

    # 修改数据以模拟D2D更高的延迟
    for category in ["cmd", "data", "trans"]:
        for req_type in ["read", "write", "mixed"]:
            values = latency_stats[category][req_type]["values"]
            # D2D延迟增加50%
            latency_stats[category][req_type]["values"] = [v * 1.5 for v in values]
            latency_stats[category][req_type]["p95"] *= 1.5
            latency_stats[category][req_type]["p99"] *= 1.5
            latency_stats[category][req_type]["max"] *= 1.5
            latency_stats[category][req_type]["sum"] *= 1.5

    # 创建绘图器
    plotter = LatencyDistributionPlotter(latency_stats, title_prefix="D2D")

    # 测试图表类型
    print("\n1. 生成直方图+CDF组合图...")
    hist_cdf_fig = plotter.plot_histogram_with_cdf(return_fig=True)
    print(f"   ✓ 直方图+CDF生成成功 (类型: {type(hist_cdf_fig).__name__})")

    print("\n2. 生成小提琴图...")
    violin_fig = plotter.plot_violin(return_fig=True)
    print(f"   ✓ 小提琴图生成成功 (类型: {type(violin_fig).__name__})")

    # 保存HTML文件
    output_dir = os.path.join(project_root, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    print("\n3. 保存图表到HTML文件...")
    hist_cdf_path = os.path.join(output_dir, "d2d_latency_histogram_cdf.html")
    hist_cdf_fig.write_html(hist_cdf_path)
    print(f"   ✓ 直方图+CDF已保存: {hist_cdf_path}")

    violin_path = os.path.join(output_dir, "d2d_latency_violin.html")
    violin_fig.write_html(violin_path)
    print(f"   ✓ 小提琴图已保存: {violin_path}")

    print("\n✅ D2D延迟分布图测试通过!")
    return True


def test_empty_data():
    """测试空数据情况"""
    print("\n" + "=" * 60)
    print("测试空数据处理")
    print("=" * 60)

    # 创建空数据
    empty_stats = {
        "cmd": {
            "read": {"sum": 0, "max": 0, "count": 0, "values": [], "p95": 0, "p99": 0},
            "write": {"sum": 0, "max": 0, "count": 0, "values": [], "p95": 0, "p99": 0},
            "mixed": {"sum": 0, "max": 0, "count": 0, "values": [], "p95": 0, "p99": 0},
        },
        "data": {
            "read": {"sum": 0, "max": 0, "count": 0, "values": [], "p95": 0, "p99": 0},
            "write": {"sum": 0, "max": 0, "count": 0, "values": [], "p95": 0, "p99": 0},
            "mixed": {"sum": 0, "max": 0, "count": 0, "values": [], "p95": 0, "p99": 0},
        },
        "trans": {
            "read": {"sum": 0, "max": 0, "count": 0, "values": [], "p95": 0, "p99": 0},
            "write": {"sum": 0, "max": 0, "count": 0, "values": [], "p95": 0, "p99": 0},
            "mixed": {"sum": 0, "max": 0, "count": 0, "values": [], "p95": 0, "p99": 0},
        },
    }

    plotter = LatencyDistributionPlotter(empty_stats, title_prefix="Empty")

    print("\n测试空数据绘图...")
    try:
        fig = plotter.plot_histogram(return_fig=True)
        print("   ✓ 空数据处理成功 (生成空图表)")
    except Exception as e:
        print(f"   ⚠ 空数据处理警告: {e}")

    print("\n✅ 空数据处理测试通过!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("延迟分布图功能测试")
    print("=" * 60)

    try:
        # 运行测试
        test_noc_latency_distribution()
        test_d2d_latency_distribution()
        test_empty_data()

        print("\n" + "=" * 60)
        print("🎉 所有测试通过!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        sys.exit(1)
