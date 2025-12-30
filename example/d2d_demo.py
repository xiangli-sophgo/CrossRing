#!/usr/bin/env python3
"""
D2D (Die-to-Die) 仿真示例
使用新的配置分离接口
"""

import sys
from pathlib import Path

sys.path.append("..")
from src.dcin.d2d_model import D2D_Model
from src.dcin.config import DCINConfig


def main():
    """D2D仿真演示 - 使用新的简化接口"""

    # 创建配置和模型
    config = DCINConfig(
        d2d_config_file="../config/topologies/d2d_config.yaml",
    )

    # 显示配置信息
    print(f"配置信息:")
    print(f"  Die数量: {getattr(config, 'NUM_DIES', 2)}")
    print(f"  Die拓扑配置:")
    die_topologies = getattr(config, "DIE_TOPOLOGIES", {})
    for die_id, topology in die_topologies.items():
        print(f"    Die{die_id}: {topology}")
    print()

    # 创建模型
    model = D2D_Model(
        config=config,
        model_type="REQ_RSP",
        result_save_path="../Result/d2d_demo/",
        results_fig_save_path="../Result/d2d_demo/figures/",
        verbose=1,
    )

    # 配置数据流
    traffic_file_path = str(Path(__file__).parent.parent / "test_data")
    traffic_chains = [
        [
            "d2d_test_traffic.txt",
        ]
    ]
    model.setup_traffic_scheduler(traffic_file_path=traffic_file_path, traffic_chains=traffic_chains)

    # 配置调试模式
    # model.setup_debug(trace_packets=[1, 2], update_interval=0.0)

    # 配置结果处理
    model.setup_result_analysis(
        # 图片生成控制
        flow_graph=True,
        ip_bandwidth_heatmap=True,
        save_figures=True,
        # CSV文件导出控制
        export_d2d_requests_csv=True,
        export_ip_bandwidth_csv=True,
        # 通用设置
        save_dir="../Result/d2d_demo/",
        heatmap_mode="total",
    )

    # 配置实时可视化
    # model.setup_visualization(enable=True, update_interval=1.0, start_cycle=0)

    # 运行仿真
    print("\n 开始D2D仿真演示")
    model.run_simulation(max_cycles=500, print_interval=100, results_analysis=True, verbose=1)

    print("\n D2D仿真演示完成!")


if __name__ == "__main__":
    main()
    # 保持程序运行，让matplotlib图表持续显示
    try:
        import matplotlib.pyplot as plt

        if plt.get_fignums():  # 如果有打开的图形
            print("\n图表已显示，按Ctrl+C或关闭图形窗口退出程序")
            plt.show(block=True)
    except KeyboardInterrupt:
        print("\n程序已退出")
    except Exception as e:
        print(f"显示图表时出错: {e}")
