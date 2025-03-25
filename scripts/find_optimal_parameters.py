from src.core import *
import os
from src.utils.component import Flit, Network, Node
from config.config import SimulationConfig
import numpy as np


def find_optimal_parameters():
    import csv

    # 定义流量文件路径和文件名
    traffic_file_path = r"../test_data/"
    # file_name = r"demo3.txt"
    # file_name = r"demo_3x3_large.txt"
    file_name = r"demo45.txt"

    # traffic_file_path = r"../traffic/"
    # traffic_file_path = r"../traffic/output-v7-32/step6_mesh_32core_map/"
    # file_name = r"LLama2_Attention_FC_Trace.txt"
    # file_name = r"LLama2_Attention_QKV_Decode_Trace.txt"
    # file_name = r"LLama2_MLP_Trace.txt"
    # file_name = r"LLama2_MM_QKV_Trace.txt"

    config_path = r"../config/config2.json"
    config = SimulationConfig(config_path)

    # 定义拓扑类型
    if not config.topo_type:
        # topo_type = "4x9"
        # topo_type = "9x4"
        topo_type = "5x4"
        # topo_type = "4x5"

        # topo_type = "6x5"

        # topo_type = "3x3"
    else:
        topo_type = config.topo_type

    # result_save_path = None

    # model_type = "REQ_RSP"
    # model_type = "Packet_Base"
    model_type = "Feature"

    results_file_name = "ETag_EQ_single_0321"
    # results_file_name = "inject_eject_queue_length"

    # 创建结果保存路径
    result_root_save_path = f"../Result/CrossRing/{model_type}/FOP/{results_file_name}/"
    os.makedirs(result_root_save_path, exist_ok=True)  # 确保根目录存在

    output_csv = os.path.join(r"../Result/Params_csv/", f"{results_file_name}.csv")
    os.makedirs(result_root_save_path, exist_ok=True)

    # 定义参数范围
    parm1_start, parm1_end, parm1_step = (2, 5, 1)
    parm2_start, parm2_end, parm2_step = (2, 7, 1)
    parm3_start, parm3_end, parm3_step = (2, 6, 1)

    # 遍历参数组合
    # for parm1 in range(parm1_start, parm1_end + 1, parm1_step):
    # for parm2 in range(parm2_start, parm2_end + 1, parm2_step):
    for parm1 in range(parm1_start, parm1_end + 1, parm1_step):
        for parm2 in range(parm1 + 1, parm2_end + 1, parm2_step):
            for parm3 in range(parm3_start, parm3_end + 1, parm3_step):

                result_part_save_path = f"{parm1}_{parm2}/"

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
                # elif model_type == "Feature":
                #     sim = Feature_model(
                #         model_type=model_type,
                #         config=config,
                #         topo_type=topo_type,
                #         traffic_file_path=traffic_file_path,
                #         file_name=file_name,
                #         result_save_path=result_root_save_path + result_part_save_path,
                #     )

                if topo_type in ["4x9", "4x5", "4x5", "5x4"]:
                    sim.config.rn_read_tracker_ostd = 64
                    sim.config.rn_write_tracker_ostd = 64
                    sim.config.rn_rdb_size = sim.config.rn_read_tracker_ostd * 4
                    sim.config.rn_wdb_size = sim.config.rn_write_tracker_ostd * 4
                    sim.config.ro_tracker_ostd = 128
                    sim.config.share_tracker_ostd = 64
                    sim.config.sn_wdb_size = sim.config.share_tracker_ostd * 4
                    sim.config.seats_per_link = 7
                    sim.config.IQ_OUT_FIFO_DEPTH = 6
                    sim.config.EQ_IN_FIFO_DEPTH = 8
                    sim.config.RB_IN_FIFO_DEPTH = 8
                    sim.config.RB_OUT_FIFO_DEPTH = 8
                    sim.config.TL_Etag_T2_UE_MAX = 4
                    sim.config.TL_Etag_T1_UE_MAX = 7
                    sim.config.TR_Etag_T2_UE_MAX = 6
                    sim.config.TU_Etag_T2_UE_MAX = parm1
                    sim.config.TU_Etag_T1_UE_MAX = parm2
                    sim.config.TD_Etag_T2_UE_MAX = parm3
                    sim.config.Both_side_ETag_upgrade = 1

                elif topo_type in ["3x3"]:
                    sim.config.rn_read_tracker_ostd = 64
                    sim.config.rn_write_tracker_ostd = 32
                    sim.config.rn_rdb_size = sim.config.rn_read_tracker_ostd * 4
                    sim.config.rn_wdb_size = sim.config.rn_write_tracker_ostd * 4
                    sim.config.ro_tracker_ostd = 64
                    sim.config.share_tracker_ostd = 64
                    sim.config.sn_wdb_size = sim.config.share_tracker_ostd * 4
                    sim.config.seats_per_link = 7
                    sim.config.IQ_OUT_FIFO_DEPTH = 6
                    sim.config.EQ_IN_FIFO_DEPTH = 8
                    sim.config.RB_IN_FIFO_DEPTH = parm1
                    sim.config.RB_OUT_FIFO_DEPTH = parm2

                # sim.config.update_config()
                sim.initial()
                sim.end_time = 20000
                sim.print_interval = 10000
                print(f"Parm1: {parm1}, Parm2: {parm2}, Parm3: {parm3}")

                # 运行模拟
                sim.run()
                sim.config.finish_del()

                sim_vars = vars(sim)
                results = {key[:-5]: value for key, value in sim_vars.items() if key.endswith()("_stat")}

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
    find_optimal_parameters()
