#!/usr/bin/env python3
"""
CrossRing NoC Bidirectional Ring Bridge Demo
演示纵向环到横向环的双向转换功能
使用base_model_v2和network_v2实现
"""

import os
import sys
import numpy as np
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.kcin.config import KCINConfig
from src.kcin.base_model_v2 import BaseModel


def run_CR_v2_demo():
    # 创建配置 - 使用5x2拓扑的YAML配置
    print("\n初始化配置...")
    kcin_type = "5x2"
    config_path = os.path.join(project_root, "config", "topologies", f"kcin_{kcin_type}.yaml")
    config = KCINConfig(config_path)

    # 创建仿真实例（使用v2版本）
    result_dir = os.path.join(project_root, "Result", "CrossRing_v2")
    os.makedirs(result_dir, exist_ok=True)

    sim = BaseModel(
        model_type="REQ_RSP",
        config=config,
        kcin_type="5x2",
    )

    # 配置流量调度器
    sim.setup_traffic_scheduler(
        traffic_file_path=f"../traffic/0617",
        traffic_chains=[
            # ["Read_burst4_2262HBM_v2.txt"],
            [
                # "Read_burst4_2262HBM_v2.txt",
                # "Write_burst4_2262HBM_v2.txt",
                "W_5x2.txt"
            ],
        ],
    )

    # 配置结果分析
    sim.setup_result_analysis(
        result_save_path=result_dir + "/",
        plot_flow_fig=True,
        plot_RN_BW_fig=True,
    )

    # 配置调试
    sim.setup_debug(
        print_trace=False,
        show_trace_id=[1014],
        verbose=1,
    )

    # 配置可视化
    sim.setup_visualization(
        plot_link_state=True,
        plot_start_cycle=2000,
    )

    # 运行仿真
    start_time = time.time()
    sim.run_simulation(max_cycles=6000, print_interval=1000)
    end_time = time.time()

    print(f"\n✓ 仿真完成! 用时: {end_time - start_time:.2f}秒")


if __name__ == "__main__":
    np.random.seed(716)
    print("启动CrossRing v2.0:")
    results = run_CR_v2_demo()
