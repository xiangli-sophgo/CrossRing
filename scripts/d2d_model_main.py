#!/usr/bin/env python3
"""
D2D (Die-to-Die) 仿真主脚本
使用新的配置分离接口，参考 CrossRingModel 设计模式
"""

from pathlib import Path
from src.d2d.d2d_model import D2D_Model
from config.d2d_config import D2DConfig


def main():
    """运行D2D仿真 - 使用新的简化接口"""

    # 创建配置和模型
    config = D2DConfig(
        # d2d_config_file="../config/topologies/d2d_2die_config.yaml",
        d2d_config_file="../config/topologies/d2d_4die_config.yaml",
    )

    # 创建模型
    model = D2D_Model(
        config=config,
        model_type="REQ_RSP",
        result_save_path="../Result/d2d_demo/",
        results_fig_save_path="../Result/d2d_demo/figures/",
        verbose=1,
    )

    # 配置各种选项
    # 配置数据流
    traffic_file_path = str(Path(__file__).parent.parent / "test_data")
    traffic_chains = [
        [
            # "d2d_data_simple_example.txt",
            # "d2d_16_share_D2D_R_1104.txt",
            # "d2d_16_share_D2D_W_1104.txt"
            # "d2d_64_share_D2D_R_1104.txt",
            # "d2d_64_share_D2D_W_1104.txt",
            # "d2d_16_share_W_1104.txt"
            # "d2d_16_share_R_1104.txt"
            # "data_sim_16_share_W_1110.txt"
            "2261_c2c_16_share_W_1118.txt"
        ]
    ]
    model.setup_traffic_scheduler(traffic_file_path=traffic_file_path, traffic_chains=traffic_chains)
    # model.setup_debug(trace_packets=[1], update_interval=0.1)
    # model.setup_visualization(enable=1, update_interval=0.5, start_cycle=200)

    model.setup_result_analysis(
        # 图片生成控制
        flow_graph_interactive=1,  # 生成HTML交互式流量图
        ip_bandwidth_heatmap=0,
        plot_rn_bw_fig=0,
        fifo_utilization_heatmap=1,
        show_result_analysis=1,  # 在浏览器中显示图像
        # CSV文件导出控制
        export_d2d_requests_csv=1,
        export_ip_bandwidth_csv=1,
        # 通用设置
        heatmap_mode="total",  # 可选: "total", "read", "write"
    )

    # 运行仿真
    print("开始仿真")
    model.run_simulation(
        max_time=200,
        print_interval=200,
        verbose=1,
    )


if __name__ == "__main__":
    main()
    # 保持程序运行，让matplotlib图表持续显示
    try:
        import matplotlib.pyplot as plt

        if plt.get_fignums():  # 如果有打开的图形
            print("\n图表已显示，按Ctrl+C或关闭图形窗口退出程序")
            plt.show(block=True)  # 阻塞显示，直到用户关闭所有图形窗口
    except KeyboardInterrupt:
        print("\n程序已退出")
    except Exception as e:
        print(f"显示图表时出错: {e}")
        pass
