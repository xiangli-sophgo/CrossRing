from src.core import *
import os
from src.utils.component import Flit, Network, Node
from config.config import SimulationConfig
import numpy as np
import csv


def main():
    import tracemalloc

    traffic_file_path = r"../test_data/"
    file_name = r"demo45.txt"
    # file_name = r"testcase-v1.1.1.txt"
    # file_name = r"burst2_large.txt"
    # file_name = r"burst4_common.txt"
    # file_name = r"3x3_burst2.txt"
    # file_name = r"demo_3x3.txt"
    # file_name = r"demo_459.txt"

    # traffic_file_path = r"../../traffic/"R
    # traffic_file_path = r"../traffic/output_v8_All_reduce/step5_data_merge/"
    # traffic_file_path = r"../traffic/output_v8-32/step5_data_merge/"
    # file_name = r"LLama2_Attention_FC_Trace.txt"
    # file_name = r"LLama2_Attention_QKV_Decode_Trace.txt"
    # file_name = r"LLama2_MLP_Trace.txt"
    # file_name = r"LLama2_MM_QKV_Trace.txt"
    # file_name = r"TPS009-Llama2-70B-S4K-O1-W8A8-B128-LMEM2M-AllReduce_Trace.txt"

    # model_type = "Feature"
    model_type = "REQ_RSP"
    # model_type = "Packet_Base"

    config_path = r"../config/config2.json"
    config = SimulationConfig(config_path)
    if not config.topo_type:
        # topo_type = "4x9"
        # topo_type = "9x4"
        topo_type = "5x4"
        # topo_type = "4x5"

        # topo_type = "6x5"

        # topo_type = "3x3"
    else:
        topo_type = config.topo_type

    results_file_name = "Spare_core_0401"
    result_root_save_path = f"../Result/CrossRing/SCM/{model_type}/{results_file_name}"
    os.makedirs(result_root_save_path, exist_ok=True)  # 确保根目录存在

    output_csv = os.path.join(r"../Result/Params_csv/", f"{results_file_name}.csv")
    os.makedirs(result_root_save_path, exist_ok=True)

    # result_save_path = None
    # config_path = r"config.json"
    np.random.seed(401)

    for failed_core_num in range(0, 4):
        for spare_core_row in range(0, 9):
            for repeat_time in range(1):
                result_part_save_path = f"{failed_core_num}_{spare_core_row}_{repeat_time}/"

                if model_type == "REQ_RSP":
                    sim = REQ_RSP_model(
                        model_type=model_type,
                        config=config,
                        topo_type=topo_type,
                        traffic_file_path=traffic_file_path,
                        file_name=file_name,
                        result_save_path=result_root_save_path + result_part_save_path,
                    )
                elif model_type == "Packet_Base":
                    sim = Packet_Base_model(
                        model_type=model_type,
                        config=config,
                        topo_type=topo_type,
                        traffic_file_path=traffic_file_path,
                        file_name=file_name,
                        result_save_path=result_root_save_path + result_part_save_path,
                    )

                # profiler = cProfile.Profile()
                # profiler.enable()

                # tracemalloc.start()

                # sim.end_time = 10000
                sim.config.burst = 4
                sim.config.rn_read_tracker_ostd = 64
                sim.config.rn_write_tracker_ostd = 64
                sim.config.rn_rdb_size = sim.config.rn_read_tracker_ostd * sim.config.burst
                sim.config.rn_wdb_size = sim.config.rn_write_tracker_ostd * sim.config.burst
                sim.config.ro_tracker_ostd = 128
                sim.config.share_tracker_ostd = 64
                sim.config.sn_wdb_size = sim.config.share_tracker_ostd * sim.config.burst
                sim.config.seats_per_link = 7
                sim.config.TL_Etag_T2_UE_MAX = 4
                sim.config.TL_Etag_T1_UE_MAX = 7
                sim.config.TR_Etag_T2_UE_MAX = 6
                sim.config.TU_Etag_T2_UE_MAX = 4
                sim.config.TU_Etag_T1_UE_MAX = 7
                sim.config.TD_Etag_T2_UE_MAX = 5
                sim.config.Both_side_ETag_upgrade = 1
                sim.config.spare_core_change(spare_core_row, failed_core_num)

                # sim.config.update_config()
                sim.initial()
                # sim.end_time = 20000
                sim.print_interval = 1000
                sim.run()
                sim.config.finish_del()

                sim_vars = vars(sim)
                results = {key[:-5]: value for key, value in sim_vars.items() if key.endswith("_stat")}

                config_var = {key: value for key, value in vars(sim.config).items()}
                results = {**results, **config_var}

                # 写入 CSV 文件
                csv_file_exists = os.path.isfile(output_csv)
                with open(output_csv, mode="a", newline="") as output_csv_file:
                    writer = csv.DictWriter(output_csv_file, fieldnames=results.keys())
                    if not csv_file_exists:
                        writer.writeheader()  # 写入表头
                    writer.writerow(results)  # 写入结果行

                Flit.clear_flit_id()
                Node.clear_packet_id()


if __name__ == "__main__":
    main()
