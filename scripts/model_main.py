from src.core import *
import os
from src.utils.components import *
from config.config import CrossRingConfig
import matplotlib
import numpy as np
import sys
import tracemalloc

if sys.platform == "darwin":  # macOS 的系统标识是 'darwin'
    matplotlib.use("macosx")  # 仅在 macOS 上使用该后端


def main():
    # traffic_file_path = r"../test_data/"
    # traffic_file_path = r"../../C2C/traffic_data"
    # traffic_file_path = r"../traffic/traffic0730"
    # traffic_file_path = r"../traffic/DeepSeek_0922/hashed/"
    # traffic_file_path = r"../traffic/data_0922/"
    traffic_file_path = r"../traffic/0617/"
    # traffic_file_path = r"../traffic/DeepSeek_0616/step6_ch_map/"
    # traffic_file_path = r"../traffic/RW_4x2_4x4/"
    # traffic_file_path = r"../traffic/nxn_traffics"

    traffic_config = [
        [
            # r"R_5x2.txt",
            # r"W_5x2.txt",
            # r"Read_burst4_2262HBM_v2.txt",
            # r"Write_burst4_2262HBM_v2.txt",
            # r"MLP_MoE.txt",
        ]
        * 1,
        [
            # r"All2All_Combine.txt",
            # r"All2All_Dispatch.txt",
            # r"full_bw_R_4x5.txt"
            "LLama2_AllReduce.txt"
            # "data_0924_R.txt"
            # "c2c_16_shared_burst2_W.txt"
            # "traffic_2260E_case1.txt",
            # "LLama2_AttentionFC.txt"
            # "W_8x8.txt"
            # "MLA_B32.txt"
            # "Add.txt"
            # "mm_q_lora_a.txt"
            # "MLP_MoE.txt"
            # "output_embedding.txt"
        ],
    ]

    # model_type = "Feature"
    model_type = "REQ_RSP"
    # model_type = "Packet_Base"

    results_fig_save_path = None

    result_save_path = f"../Result/CrossRing/{model_type}/"
    # results_fig_save_path = f"../Result/Plt_IP_BW/{model_type}/"

    # 拓扑类型到配置文件的映射
    topo_config_map = {
        "3x3": r"../config/topologies/topo_3x3.yaml",
        "4x4": r"../config/topologies/topo_4x4.yaml",
        "5x2": r"../config/topologies/topo_5x2.yaml",
        "5x4": r"../config/topologies/topo_5x4.yaml",
        "6x5": r"../config/topologies/topo_6x5.yaml",
        "8x8": r"../config/topologies/topo_8x8.yaml",
    }

    # 默认拓扑类型
    # topo_type = "4x9"
    # topo_type = "9x4"
    topo_type = "5x4"  # SG2262
    # topo_type = "4x4"
    # topo_type = "5x2"
    # topo_type = "3x3"
    # topo_type = "6x5"  # SG2260
    # topo_type = "8x8"  # SG2260E

    # 根据拓扑类型选择配置文件
    config_path = topo_config_map.get(topo_type, r"../config/default.yaml")
    config = CrossRingConfig(config_path)
    config.CROSSRING_VERSION = "V1"

    # 从配置文件获取拓扑类型，如果没有则使用默认值
    topo_type = config.TOPO_TYPE if config.TOPO_TYPE else topo_type

    sim: BaseModel = eval(f"{model_type}_model")(
        model_type=model_type,
        config=config,
        topo_type=topo_type,
        traffic_file_path=traffic_file_path,
        traffic_config=traffic_config,
        result_save_path=result_save_path,
        results_fig_save_path=results_fig_save_path,
        plot_flow_fig=1,
        flow_fig_show_CDMA=0,
        plot_RN_BW_fig=1,
        plot_link_state=0,
        plot_start_cycle=1500,
        print_trace=0,
        show_trace_id=[71],
        show_node_id=1,
        verbose=1,
    )
    np.random.seed(801)

    sim.initial()
    sim.end_time = 2000
    sim.print_interval = 2000
    sim.run()


if __name__ == "__main__":
    main()
