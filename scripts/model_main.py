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
    """运行CrossRing仿真 - 使用新的简化配置接口"""

    # ==================== 流量配置 ====================
    traffic_file_path = r"../traffic/DeepSeek_0616/step6_ch_map/"
    # traffic_file_path = r"../test_data"
    traffic_config = [
        [
            "LLama2_AllReduce.txt"
            # "data_1022_W.txt"
            # "data_1022_R.txt"
            # "data_1030_W.txt"
            # "MLP.txt"
        ],
    ]

    # ==================== 模型类型 ====================
    # model_type = "Feature"
    model_type = "REQ_RSP"
    # model_type = "Packet_Base"

    # ==================== 拓扑配置 ====================
    topo_config_map = {
        "3x3": r"../config/topologies/topo_3x3.yaml",
        "4x4": r"../config/topologies/topo_4x4.yaml",
        "5x2": r"../config/topologies/topo_5x2.yaml",
        "5x4": r"../config/topologies/topo_5x4.yaml",
        "6x5": r"../config/topologies/topo_6x5.yaml",
        "8x8": r"../config/topologies/topo_8x8.yaml",
    }

    topo_type = "5x4"  # SG2262
    # topo_type = "4x4"
    # topo_type = "5x2"
    # topo_type = "3x3"
    # topo_type = "6x5"  # SG2260
    # topo_type = "8x8"  # SG2260E

    # ==================== 创建配置和模型 ====================
    config_path = topo_config_map.get(topo_type, r"../config/default.yaml")
    config = CrossRingConfig(config_path)
    config.CROSSRING_VERSION = "V1"

    # 从配置文件获取拓扑类型，如果没有则使用默认值
    topo_type = config.TOPO_TYPE if config.TOPO_TYPE else topo_type

    # 创建模型实例
    sim: BaseModel = eval(f"{model_type}_model")(
        model_type=model_type,
        config=config,
        topo_type=topo_type,
        verbose=1,
    )

    sim.setup_traffic_scheduler(traffic_file_path=traffic_file_path, traffic_chains=traffic_config)

    sim.setup_result_analysis(
        plot_flow_fig=1,
        plot_RN_BW_fig=1,
        fifo_utilization_heatmap=1,
        result_save_path=f"../Result/CrossRing/{model_type}/",
        save_fig=0,
    )
    # sim.setup_debug(print_trace=1, show_trace_id=[1], update_interval=0.1)
    # sim.setup_visualization(plot_link_state=1, plot_start_cycle=3000, show_node_id=1)
    np.random.seed(801)

    sim.run_simulation(max_time=6000, print_interval=200)


if __name__ == "__main__":
    main()
