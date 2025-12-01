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
        # d2d_config_file="../config/topologies/dcin_2die_config.yaml",
        d2d_config_file="../config/topologies/dcin_4die_config.yaml",
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
    # traffic_file_path = str(Path(__file__).parent.parent / "test_data")
    traffic_file_path = str(Path(__file__).parent.parent / "traffic")
    # traffic_file_path = str(Path(__file__).parent.parent / "traffic" / "2DIE")
    # traffic_file_path = str(Path(__file__).parent.parent / "traffic" / "2261")
    traffic_chains = [
        [
            # "d2d_data_simple_example.txt",
            # "d2d_16_share_D2D_R_1104.txt",
            # "d2d_16_share_D2D_W_1104.txt"
            # "d2d_64_share_D2D_R_1104.txt",
            # "d2d_64_share_D2D_W_1104.txt",
            # "2261_64share_d2d_W.txt"
            # "d2d_16_share_R_1104.txt"
            # "data_sim_16_share_W_1110.txt"
            # "2261_c2c_16share_R.txt"
            # "2261_c2c_64share_d2d_R.txt"
            # "2261_16share_R.txt"
            # "DIE0_2_16share_d2d_R.txt"
            "test_d2d.txt"
        ]
    ]
    model.setup_traffic_scheduler(traffic_file_path=traffic_file_path, traffic_chains=traffic_chains)
    # model.setup_debug(trace_packets=[1], update_interval=0.01)
    # model.setup_visualization(enable=1, update_interval=0.2, start_cycle=500)

    model.setup_result_analysis(
        # 图片生成控制
        flow_graph_interactive=1,  # 生成HTML交互式流量图
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
        max_time=1200,
        print_interval=200,
        verbose=1,
    )
    # model.save_to_database(experiment_name="DCIN 仿真")


if __name__ == "__main__":
    main()
