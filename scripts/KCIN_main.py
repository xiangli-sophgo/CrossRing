import matplotlib
import numpy as np
import sys

if sys.platform == "darwin":  # macOS 的系统标识是 'darwin'
    matplotlib.use("macosx")  # 仅在 macOS 上使用该后端


def main():
    """运行CrossRing仿真 - 使用新的简化配置接口"""

    # ==================== 流量配置 ====================
    # traffic_file_path = r"../traffic/DeepSeek/step6_ch_map/"
    # traffic_file_path = r"../test_data"
    traffic_file_path = r"../traffic"
    traffic_config = [
        [
            # "LLama2_AllReduce.txt"
            # "data_sim_16_share_R_1104.txt"
            # "data_sim_16_share_W_1104.txt"
            # "data_sim_16_share_d2d_W_1104.txt"
            # "data_sim_16_share_d2d_W_1104.txt"
            # "data_sim_64_share_d2d_R_1104.txt"
            # "data_sim_64_share_d2d_W_1104.txt"
            # "data_burst4_W_1111.txt"
            # "test.txt"
            # "simple_case_W.txt"
            "simple_case_R.txt"
            # "traffic_20251119_152813.txt"
        ],
    ]

    # ==================== 模型类型 ====================
    # model_type = "Feature"
    model_type = "REQ_RSP"
    # model_type = "Packet_Base"

    # ==================== 拓扑配置 ====================
    kcin_config_map = {
        "3x3": r"../config/topologies/kcin_3x3.yaml",
        "4x4": r"../config/topologies/kcin_4x4.yaml",
        "5x2": r"../config/topologies/kcin_5x2.yaml",
        "5x4": r"../config/topologies/kcin_5x4.yaml",
        "5x4_v2": r"../config/topologies/kcin_5x4_v2.yaml",  # v2 RingStation 架构
        "6x5": r"../config/topologies/kcin_6x5.yaml",
        "8x8": r"../config/topologies/kcin_8x8.yaml",
    }

    # kcin_type = "5x4"  # SG2262 v1 架构
    kcin_type = "5x4_v2"  # SG2262 v2 RingStation 架构
    # kcin_type = "4x4"
    # kcin_type = "5x2"
    # kcin_type = "3x3"
    # kcin_type = "6x5"  # SG2260
    # kcin_type = "8x8"  # SG2260E

    # ==================== 创建配置和模型 ====================
    config_path = kcin_config_map.get(kcin_type, r"../config/default.yaml")

    # 根据 kcin_type 选择对应版本
    if "_v2" in kcin_type:
        from src.kcin import v2 as kcin_module
        from src.kcin.v2.config import V2Config

        config = V2Config(config_path)
    else:
        from src.kcin import v1 as kcin_module
        from src.kcin.v1.config import V1Config

        config = V1Config(config_path)

    print(f"使用 KCIN 版本: {config.KCIN_VERSION}")

    # 从配置文件获取拓扑类型，如果没有则使用默认值
    topo_type = config.TOPO_TYPE if config.TOPO_TYPE else kcin_type.replace("_v2", "")

    # 创建模型实例（使用版本对应的模块）
    model_class = getattr(kcin_module, f"{model_type}_model")
    sim = model_class(
        model_type=model_type,
        config=config,
        topo_type=topo_type,
        verbose=1,
    )

    sim.setup_traffic_scheduler(traffic_file_path=traffic_file_path, traffic_chains=traffic_config)

    sim.setup_result_analysis(
        plot_RN_BW_fig=0,
        flow_graph_interactive=1,  # 生成交互式流量图
        fifo_utilization_heatmap=1,
        result_save_path=f"../Result/CrossRing/{model_type}/",
        show_result_analysis=1,
    )
    # sim.setup_debug(print_trace=1, show_trace_id=[3], update_interval=0.1)
    # sim.setup_visualization(plot_link_state=1, plot_start_cycle=1, show_node_id=1)
    # np.random.seed(801)

    sim.run_simulation(max_time=500, print_interval=200)

    # ==================== 保存结果到数据库 ====================
    # sim.save_to_database(experiment_name="KCIN 仿真")


if __name__ == "__main__":
    import traceback, logging

    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.error("Unhandled exception:\n%s", traceback.format_exc())
        import sys

        sys.exit(1)
