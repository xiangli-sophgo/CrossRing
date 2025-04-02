from src.core import *
import os
from src.utils.component import Flit, Network, Node
from config.config import SimulationConfig
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

    # traffic_file_path = r"../../traffic/"
    # traffic_file_path = r"../traffic/output_All_reduce/step5_data_merge/"
    traffic_file_path = r"../traffic/output_DeepSeek/step5_data_merge/"
    all_files = os.listdir(traffic_file_path)

    # 筛选出 .txt 文件
    file_names = [file for file in all_files if file.endswith(".txt")]

    # file_name = r"LLama2_Attention_FC_Trace.txt"
    # file_name = r"output_Trace.txt"
    # file_name = r"LLama2_Attention_QKV_Decode_Trace.txt"
    # file_name = r"LLama2_MLP_Trace.txt"
    # file_name = r"LLama2_MM_QKV_Trace.txt"
    # file_name = r"TPS009-Llama2-70B-S4K-O1-W8A8-B128-LMEM2M-AllReduce_Trace.txt"

    # model_type = "Feature"
    model_type = "REQ_RSP"
    # model_type = "Packet_Base"

    results_file_name = "DeepSeek_0402"

    result_save_path = f"../Result/CrossRing/{model_type}/{traffic_file_path.split('/')[-2]}/"
    output_csv = os.path.join(r"../Result/Traffic_result_csv/", f"{results_file_name}.csv")
    os.makedirs(result_save_path, exist_ok=True)

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
    config.topo_type = topo_type

    # result_save_path = None
    # config_path = r"config.json"
    for file_name in file_names:
        sim = eval(f"{model_type}_model")(
            model_type=model_type,
            config=config,
            topo_type=topo_type,
            traffic_file_path=traffic_file_path,
            file_name=file_name,
            result_save_path=result_save_path + file_name[:-4] + "/",
        )

        # profiler = cProfile.Profile()
        # profiler.enable()

        # tracemalloc.start()

        # sim.end_time = 10000
        sim.config.burst = 4
        sim.config.rn_read_tracker_ostd = 128
        sim.config.rn_write_tracker_ostd = 64
        sim.config.rn_rdb_size = sim.config.rn_read_tracker_ostd * sim.config.burst
        sim.config.rn_wdb_size = sim.config.rn_write_tracker_ostd * sim.config.burst
        sim.config.ro_tracker_ostd = 128
        sim.config.share_tracker_ostd = 64
        sim.config.sn_wdb_size = sim.config.share_tracker_ostd * sim.config.burst
        sim.config.seats_per_link = 7

        # sim.config.update_config()
        sim.initial()
        # sim.end_time = 1000
        sim.print_interval = 5000
        sim.run()

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
