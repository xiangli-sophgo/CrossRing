"""
测试FIFO使用率热力图功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.fifo_heatmap_visualizer import FIFOUtilizationCollector, FIFOHeatmapVisualizer, create_fifo_heatmap
from config.config import CrossRingConfig
import json


def test_with_mock_data():
    """使用模拟数据测试FIFO热力图"""

    print("=" * 60)
    print("测试FIFO使用率热力图功能")
    print("=" * 60)

    # 加载配置
    config_path = "../config/topologies/kcin_5x4.yaml"
    if not os.path.exists(config_path):
        config_path = "../config/config.json"

    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return

    config = CrossRingConfig(config_path)

    print(f"\n配置信息:")
    print(f"  拓扑: {config.NUM_ROW}x{config.NUM_COL}")
    print(f"  IQ_OUT_FIFO_DEPTH_HORIZONTAL: {config.IQ_OUT_FIFO_DEPTH_HORIZONTAL}")
    print(f"  RB_IN_FIFO_DEPTH: {config.RB_IN_FIFO_DEPTH}")

    # 创建模拟的Network对象
    class MockNetwork:
        def __init__(self):
            self.fifo_depth_sum = {
                "IQ": {"TR": {0: 1000, 1: 1500, 2: 800}, "TL": {0: 1200, 1: 900}, "TU": {0: 600}, "TD": {1: 700}, "EQ": {2: 500}, "CH_buffer": {0: {"gdma": 400, "sdma": 300}, 1: {"ddr": 500}}},
                "RB": {"TR": {0: 2000, 1: 2500}, "TL": {0: 1800, 1: 2200}, "TU": {2: 1000}, "TD": {2: 1100}, "EQ": {}},
                "EQ": {"TU": {0: 300, 1: 400}, "TD": {0: 350, 1: 450}, "CH_buffer": {0: {"gdma": 200}, 1: {"ddr": 250}}},
            }

            self.fifo_max_depth = {
                "IQ": {"TR": {0: 5, 1: 7, 2: 4}, "TL": {0: 6, 1: 5}, "TU": {0: 4}, "TD": {1: 5}, "EQ": {2: 3}, "CH_buffer": {0: {"gdma": 3, "sdma": 2}, 1: {"ddr": 4}}},
                "RB": {"TR": {0: 12, 1: 15}, "TL": {0: 10, 1: 14}, "TU": {2: 8}, "TD": {2: 9}, "EQ": {}},
                "EQ": {"TU": {0: 4, 1: 5}, "TD": {0: 4, 1: 6}, "CH_buffer": {0: {"gdma": 2}, 1: {"ddr": 3}}},
            }

    # 创建模拟的Die对象
    class MockDieModel:
        def __init__(self, die_id, config):
            self.die_id = die_id
            self.config = config
            self.network = MockNetwork()

    # 创建模拟的dies字典
    dies = {
        0: MockDieModel(0, config),
        # 1: MockDieModel(1, config)
    }

    total_cycles = 1000  # 模拟1000个周期

    print(f"\n开始收集FIFO使用率数据...")
    print(f"  总周期数: {total_cycles}")

    # 使用便捷函数创建热力图
    output_path = "../Result/test_fifo_heatmap.html"

    try:
        result = create_fifo_heatmap(dies=dies, config=config, total_cycles=total_cycles, die_layout={0: (0, 0), 1: (1, 0)}, die_rotations={0: 0, 1: 0}, save_path=output_path)

        if result:
            print(f"\n测试成功!")
            print(f"  热力图已保存到: {result}")
            print(f"\n请用浏览器打开以下文件查看交互式热力图:")
            print(f"  {os.path.abspath(result)}")
        else:
            print("\n测试失败: 未能生成热力图")

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)


def test_collector_directly():
    """直接测试Collector类"""

    print("\n" + "=" * 60)
    print("测试FIFOUtilizationCollector")
    print("=" * 60)

    # 使用配置文件
    config_path = "../config/topologies/kcin_5x4.yaml"
    if not os.path.exists(config_path):
        config_path = "../config/config.json"

    config = CrossRingConfig(config_path)
    collector = FIFOUtilizationCollector(config)

    # 测试容量获取
    print("\n测试FIFO容量获取:")
    print(f"  IQ-TR容量: {collector._get_fifo_capacity('IQ', 'TR', 0)}")
    print(f"  IQ-TU容量: {collector._get_fifo_capacity('IQ', 'TU', 0)}")
    print(f"  RB-TR容量: {collector._get_fifo_capacity('RB', 'TR', 0)}")
    print(f"  IQ_CH-gdma容量: {collector._get_fifo_capacity('IQ_CH', 'gdma', 0)}")

    print("\nCollector测试完成")


if __name__ == "__main__":
    # 先测试Collector
    test_collector_directly()

    # 再测试完整流程
    test_with_mock_data()
