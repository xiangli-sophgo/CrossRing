"""
D2D (Die-to-Die) 5x4拓扑演示
包含完整的结果处理和可视化功能。
"""

import sys

sys.path.append("..")
from src.core.d2d_model import D2D_Model
from config.d2d_config import D2DConfig
import numpy as np
import os


def main():
    """
    D2D 仿真演示 - 包含完整的结果处理和可视化
    """
    # 使用D2DConfig替代CrossRingConfig，获得D2D特定配置功能
    die_topo_type = "5x4"
    config = D2DConfig(die_config_file="../config/topologies/topo_5x4.yaml", d2d_config_file="../config/topologies/d2d_config.yaml")  # Die拓扑配置  # D2D专用配置

    # 定义拓扑结构

    print(f"配置信息:")
    print(f"  Die数量: {getattr(config, 'NUM_DIES', 2)}")
    print(f"  每个Die: {die_topo_type} (5行4列)")
    print()

    # 初始化D2D仿真模型 - 启用完整功能和D2D Trace
    sim = D2D_Model(
        config=config,
        traffic_file_path=r"../traffic/d2d_test/",
        traffic_config=[["d2d_test_traffic.txt"]],
        model_type="REQ_RSP",
        topo_type=die_topo_type,
        result_save_path="../Result/d2d_demo/",
        results_fig_save_path="../Result/d2d_demo/figures/",
        verbose=1,
        print_d2d_trace=1,  # 启用D2D trace功能
        show_d2d_trace_id=0,  # 自动跟踪所有活跃packet，也可以指定特定ID如[1, 2]
        d2d_trace_sleep=0.0,  # 不暂停，加快调试
    )

    # 初始化仿真
    sim.initial()

    # 设置仿真参数
    sim.end_time = 500  # 缩短测试周期以便调试
    sim.print_interval = 100

    print("-" * 40)
    print("开始D2D仿真...")
    print("期望看到:")
    print("1. 第1个cycle: 请求注入到Die 0")
    print("2. 请求传输到D2D_SN")
    print("3. AXI跨die传输")
    print("4. 完整的D2D trace输出")
    print("-" * 40)
    sim.run()


if __name__ == "__main__":
    main()
