from src.core import *
import os
from src.utils.component import Flit, Network, Node


def main():
    import tracemalloc

    traffic_file_path = r"../test_data/"
    file_name = r"demo3.txt"
    # file_name = r"testcase-v1.1.1.txt"
    # file_name = r"burst2_large.txt"
    # file_name = r"3x3_burst2.txt"

    # traffic_file_path = r"../../traffic/"
    # traffic_file_path = r"../traffic/output-v8-32/3M/step6_32core_map/"
    # file_name = r"LLama2_Attention_FC_Trace.txt"
    # file_name = r"LLama2_Attention_QKV_Decode_Trace.txt"
    # file_name = r"LLama2_MLP_Trace.txt"
    # file_name = r"LLama2_MM_QKV_Trace.txt"

    model_type = "REQ_RSP"
    # model_type = "Packet_Base"

    p1 = 64
    p2 = 64

    result_save_path = f"../Result/CrossRing/{model_type}/{p1}-{p2}/"

    # topo_type = "4x9"
    # topo_type = "9x4"
    # topo_type = "5x4"
    topo_type = "4x5"

    # topo_type = "6x5"

    # topo_type = "3x3"

    # result_save_path = None
    config_path = r"../config/config2.json"
    # config_path = r"config.json"
    sim = eval(f"{model_type}_model")(
        model_type=model_type,
        config_path=config_path,
        topo_type=topo_type,
        traffic_file_path=traffic_file_path,
        file_name=file_name,
        result_save_path=result_save_path,
    )

    # profiler = cProfile.Profile()
    # profiler.enable()

    # tracemalloc.start()

    # sim.end_time = 10000
    sim.config.rn_read_tracker_ostd = 64
    sim.config.rn_write_tracker_ostd = 32
    sim.config.rn_rdb_size = sim.config.rn_read_tracker_ostd * 2
    sim.config.rn_wdb_size = sim.config.rn_write_tracker_ostd * 2
    sim.config.ro_tracker_ostd = p1
    sim.config.share_tracker_ostd = p2
    sim.config.sn_wdb_size = sim.config.share_tracker_ostd * 2
    sim.config.seats_per_link = 7

    # sim.config.update_config()
    sim.initial()
    # sim.end_time = 1000
    sim.print_interval = 10000
    sim.run()
    print(f"rn_r_tracker_ostd: {sim.config.rn_read_tracker_ostd}: rn_w_tracker_ostd: {sim.config.rn_write_tracker_ostd}")
    print(f"ro_tracker_ostd: {p1}: share_tracker_ostd: {p2}\n")

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
