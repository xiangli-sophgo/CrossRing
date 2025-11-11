from src.noc import *
import os
from src.utils.component import Flit, Network, Node
from config.config import CrossRingConfig
import numpy as np
import csv


def main():
    import tracemalloc

    traffic_file_path = r"../test_data/"
    file_name = r"traffic_5x4_8_shared_0427.txt"
    # file_name = r"testcase-v1.1.1.txt"
    # file_name = r"burst2_large.txt"
    # file_name = r"burst4_common.txt"
    # file_name = r"3x3_burst2.txt"
    # file_name = r"demo_3x3.txt"
    # file_name = r"demo_459.txt"

    traffic_file_path = r"../traffic/"
    # traffic_file_path = r"../traffic/output_v8_new/step5_data_merge/"
    # traffic_file_path = r"../traffic/output_v8_All_reduce/step5_data_merge/"
    # traffic_file_path = r"../traffic/output-v8-32/2M/step5_data_merge/"
    # file_name = r"LLama2_Attention_FC_Trace.txt"
    # file_name = r"LLama2_Attention_QKV_Decode_Trace.txt"
    file_name = r"MLP_Trace.txt"
    # file_name = r"LLama2_MM_QKV_Trace.txt"
    # file_name = r"TPS009-Llama2-70B-S4K-O1-W8A8-B128-LMEM2M-AllReduce_Trace.txt"

    # model_type = "Feature"
    model_type = "REQ_RSP"
    # model_type = "Packet_Base"

    config_path = r"../config/topologies/topo_5x4.yaml"
    config = CrossRingConfig(config_path)
    if not config.TOPO_TYPE:
        # topo_type = "4x9"
        # topo_type = "9x4"
        topo_type = "5x4"
        # topo_type = "4x5"

        # topo_type = "6x5"

        # topo_type = "3x3"
    else:
        topo_type = config.TOPO_TYPE
    config.TOPO_TYPE = topo_type
    results_file_name = "Spare_core_MLP_0427"
    results_fig_save_path = None
    result_root_save_path = f"../Result/CrossRing/SCM/{model_type}/{results_file_name}/"
    # results_fig_save_path = f"../Result/Plt_IP_BW/SCM/{model_type}/{results_file_name}/"

    os.makedirs(result_root_save_path, exist_ok=True)  # 确保根目录存在

    output_csv = os.path.join(r"../Result/Params_csv/", f"{results_file_name}.csv")
    os.makedirs(result_root_save_path, exist_ok=True)

    np.random.seed(427)
    param = 0
    if param == 0:
        repeat_time_all = 1
    else:
        repeat_time_all = min(4 * param, 10)

    for repeat_time in range(repeat_time_all):
        for failed_core_num in range(param, param + 1):
            failed_core_poses = np.random.choice(list(i for i in range(16)), failed_core_num, replace=False)
            # failed_core_poses = [5]
            for spare_core_row in range(0, 9, 2):
                result_part_save_path = f"{failed_core_num}_{spare_core_row}_{repeat_time}/"

                # 创建模型实例
                sim = eval(f"{model_type}_model")(
                    model_type=model_type,
                    config=config,
                    topo_type=topo_type,
                )

                # 配置流量调度器
                sim.setup_traffic_scheduler(
                    traffic_file_path=traffic_file_path,
                    traffic_chains=file_name,
                )

                # 配置结果分析
                sim.setup_result_analysis(
                    result_save_path=result_root_save_path + result_part_save_path,
                    results_fig_save_path=results_fig_save_path,
                    plot_flow_fig=True,
                )

                # profiler = cProfile.Profile()
                # profiler.enable()

                # tracemalloc.start()

                # sim.end_time = 10000
                sim.config.BURST = 4
                sim.config.NUM_IP = 32
                if topo_type == "3x3":
                    sim.config.NUM_DDR = 8
                    sim.config.NUM_L2M = 4
                    sim.config.NUM_GDMA = 4
                    sim.config.NUM_SDMA = 4
                    sim.config.num_RN = 4
                    sim.config.num_SN = 8
                elif topo_type in ["5x4", "4x5"]:
                    sim.config.NUM_DDR = 32
                    sim.config.NUM_L2M = 32
                    sim.config.NUM_GDMA = 32
                    sim.config.NUM_SDMA = 32
                    sim.config.num_RN = 32
                    sim.config.num_SN = 32
                sim.config.RN_R_TRACKER_OSTD = 64
                sim.config.RN_W_TRACKER_OSTD = 64
                sim.config.RN_RDB_SIZE = sim.config.RN_R_TRACKER_OSTD * sim.config.BURST
                sim.config.RN_WDB_SIZE = sim.config.RN_W_TRACKER_OSTD * sim.config.BURST
                sim.config.sn_read_tracker_ostd = 128
                sim.config.sn_write_tracker_ostd = 64
                sim.config.sn_wdb_size = sim.config.sn_write_tracker_ostd * sim.config.BURST
                sim.config.SEAT_PRE_LINK = 7
                sim.config.RB_IN_FIFO_DEPTH = 8
                sim.config.TL_Etag_T2_UE_MAX = 5
                sim.config.TL_Etag_T1_UE_MAX = 7
                sim.config.TR_Etag_T2_UE_MAX = 6
                sim.config.TU_Etag_T2_UE_MAX = 4
                sim.config.TU_Etag_T1_UE_MAX = 7
                sim.config.TD_Etag_T2_UE_MAX = 5
                sim.config.ETag_BOTHSIDE_UPGRADE = 1
                sim.config.DDR_R_LATENCY_original = 150
                sim.config.DDR_R_LATENCY_VAR_original = 0
                sim.config.DDR_W_LATENCY_original = 16
                sim.config.L2M_R_LATENCY_original = 12
                sim.config.L2M_W_LATENCY_original = 16
                sim.config.spare_core_change(spare_core_row, failed_core_num, failed_core_poses)

                # 运行仿真
                sim.run_simulation(max_cycles=sim.end_time, print_interval=1000)

                results = sim.get_results()

                # 写入 CSV 文件
                csv_file_exists = os.path.isfile(output_csv)
                with open(output_csv, mode="a", newline="") as output_csv_file:
                    writer = csv.DictWriter(output_csv_file, fieldnames=results.keys())
                    if not csv_file_exists:
                        writer.writeheader()  # 写入表头
                    writer.writerow(results)  # 写入结果行


if __name__ == "__main__":
    main()
