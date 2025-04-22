from src.core import *
import os
from src.utils.component import Flit, Network, Node
from config.config import SimulationConfig


def main():
    import tracemalloc

    traffic_file_path = r"../test_data/"
    file_name = r"traffic_32_shared_0422.txt"
    # file_name = r"burst2_0417_2.txt"
    # file_name = r"burst2_large.txt"
    # file_name = r"burst4_common.txt"
    # file_name = r"3x3_burst2.txt"
    # file_name = r"demo_3x3.txt"
    # file_name = r"demo_459.txt"

    # traffic_file_path = r"../../traffic/"
    # traffic_file_path = r"../traffic/output_DeepSeek_part1/step5_data_merge/"
    traffic_file_path = r"../traffic/output_v8_new/step5_data_merge/"
    # traffic_file_path = r"../traffic/output_v8_All_reduce/step5_data_merge/"
    # file_name = r"output_embedding_Trace.txt"
    # file_name = r"LLama2_Attention_FC_Trace.txt"
    # file_name = r"output_Trace.txt"
    # file_name = r"LLama2_Attention_QKV_Decode_Trace.txt"
    # file_name = r"MLP_MoE_Trace.txt"
    # file_name = r"LLama2_MM_QKV_Trace.txt"
    file_name = r"TPS009-Llama2-70B-S4K-O1-W8A8-B128-LMEM2M-AllReduce_Trace.txt"

    # model_type = "Feature"
    model_type = "REQ_RSP"
    # model_type = "Packet_Base"

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
    sim.config.num_ips = 32
    sim.config.rn_read_tracker_ostd = 64
    sim.config.rn_write_tracker_ostd = 64
    sim.config.rn_rdb_size = sim.config.rn_read_tracker_ostd * sim.config.burst
    sim.config.rn_wdb_size = sim.config.rn_write_tracker_ostd * sim.config.burst
    sim.config.ro_tracker_ostd = 128
    sim.config.share_tracker_ostd = 64
    sim.config.sn_wdb_size = sim.config.share_tracker_ostd * sim.config.burst
    sim.config.IQ_OUT_FIFO_DEPTH = 8
    sim.config.EQ_IN_FIFO_DEPTH = 8
    sim.config.RB_IN_FIFO_DEPTH = 8
    sim.config.RB_OUT_FIFO_DEPTH = 8
    sim.config.TL_Etag_T2_UE_MAX = 4
    sim.config.TL_Etag_T1_UE_MAX = 7
    sim.config.TR_Etag_T2_UE_MAX = 6
    sim.config.TU_Etag_T2_UE_MAX = 4
    sim.config.TU_Etag_T1_UE_MAX = 7
    sim.config.TD_Etag_T3_UE_MAX = 5
    sim.config.ITag_Trigger_Th_H = 80
    sim.config.ITag_Trigger_Th_V = 80
    sim.config.ITag_Max_Num_H = sim.config.ITag_Max_Num_V = 1
    sim.config.seats_per_link = 7
    sim.config.Both_side_ETag_upgrade = 1

    # sim.config.update_config()
    sim.initial()
    # sim.end_time = 50000
    sim.print_interval = 2000
    sim.run()
    # print(f"rn_r_tracker_ostd: {sim.config.rn_read_tracker_ostd}: rn_w_tracker_ostd: {sim.config.rn_write_tracker_ostd}")
    print(f"ITag_Trigger_Th_H: {sim.config.ITag_Trigger_Th_H}: ITag_Max_Num: {sim.config.ITag_Max_Num_V}, {sim.config.seats_per_link}\n")

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
