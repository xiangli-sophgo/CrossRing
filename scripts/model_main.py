from src.core import *
import os
from src.utils.component import Flit, Network, Node
from config.config import SimulationConfig


def main():
    import tracemalloc

    traffic_file_path = r"../test_data/"
    file_name = r"Traffic_R_W_64GB.txt"
    # file_name = r"testcase-v1.1.1.txt"
    # file_name = r"burst2_large.txt"
    # file_name = r"burst4_common.txt"
    # file_name = r"3x3_burst2.txt"
    # file_name = r"demo_3x3.txt"
    # file_name = r"demo_459.txt"

    # traffic_file_path = r"../../traffic/"
    # traffic_file_path = r"../traffic/output_DeepSeek/step5_data_merge/"
    # traffic_file_path = r"../traffic/output_v8_new/step5_data_merge/"
    # traffic_file_path = r"../traffic/output_v8_All_reduce/step5_data_merge/"
    # file_name = r"output_embedding_Trace.txt"
    # file_name = r"LLama2_Attention_FC_Trace.txt"
    # file_name = r"output_Trace.txt"
    # file_name = r"LLama2_Attention_QKV_Decode_Trace.txt"
    # file_name = r"LLama2_MLP_Trace.txt"
    # file_name = r"LLama2_MM_QKV_Trace.txt"
    # file_name = r"TPS009-Llama2-70B-S4K-O1-W8A8-B128-LMEM2M-AllReduce_Trace.txt"

    # model_type = "Feature"
    model_type = "REQ_RSP"
    # model_type = "Packet_Base"

    p1 = 128
    p2 = 64

    results_fig_save_path = None

    result_save_path = f"../Result/CrossRing/{model_type}/"
    # results_fig_save_path = f"../Result/Plt_IP_BW/{model_type}/"

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
    sim = eval(f"{model_type}_model")(
        model_type=model_type,
        config=config,
        topo_type=topo_type,
        traffic_file_path=traffic_file_path,
        file_name=file_name,
        result_save_path=result_save_path,
        results_fig_save_path=results_fig_save_path,
        plot_flow_fig=1,
    )

    # profiler = cProfile.Profile()
    # profiler.enable()

    # tracemalloc.start()

    # sim.end_time = 10000
    sim.config.burst = 4
    sim.config.num_ips = 4
    sim.config.rn_read_tracker_ostd = 128
    sim.config.rn_write_tracker_ostd = 64
    sim.config.rn_rdb_size = sim.config.rn_read_tracker_ostd * sim.config.burst
    sim.config.rn_wdb_size = sim.config.rn_write_tracker_ostd * sim.config.burst
    sim.config.ro_tracker_ostd = 128
    sim.config.share_tracker_ostd = 64
    sim.config.sn_wdb_size = sim.config.share_tracker_ostd * sim.config.burst
    sim.config.ITag_Trigger_Th_H = 16
    sim.config.ITag_Trigger_Th_V = 16
    sim.config.ITag_Max_Num_H = sim.config.ITag_Max_Num_V = 3
    sim.config.seats_per_link = 7
    sim.config.Both_side_ETag_upgrade = 1

    # sim.config.update_config()
    sim.initial()
    # sim.end_time = 500
    sim.print_interval = 20000
    sim.run()
    # print(f"rn_r_tracker_ostd: {sim.config.rn_read_tracker_ostd}: rn_w_tracker_ostd: {sim.config.rn_write_tracker_ostd}")
    # print(f"ro_tracker_ostd: {p1}: share_tracker_ostd: {p2}\n")

    # # 获取当前的内存快照
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics("lineno")
    # print("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #     print(stat)

    # profiler.disable()
    # profiler.print_stats()

    # sim.draw_figure()


if __name__ == "__main__":
    main()
