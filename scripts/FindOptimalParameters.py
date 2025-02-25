from src.core import *
import os
from src.utils.component import Flit, Network, Node
import numpy as np


def find_optimal_parameters():
    import csv

    # 定义流量文件路径和文件名
    traffic_file_path = r"../test_data/"
    # file_name = r"demo3.txt"
    # file_name = r"demo_3x3.txt"
    file_name = r"demo_459.txt"

    # traffic_file_path = r"../traffic/"
    # traffic_file_path = r"../traffic/output-v7-32/step6_mesh_32core_map/"
    # file_name = r"LLama2_Attention_FC_Trace.txt"
    # file_name = r"LLama2_Attention_QKV_Decode_Trace.txt"
    # file_name = r"LLama2_MLP_Trace.txt"
    # file_name = r"LLama2_MM_QKV_Trace.txt"

    # 定义拓扑类型
    # topo_type = "4x9"
    # topo_type = "9x4"
    # topo_type = "5x4"
    topo_type = "4x5"

    # topo_type = "6x5"

    # topo_type = "3x3"

    # result_save_path = None
    config_path = r"../config/config2.json"

    model_type = "REQ_RSP"
    # model_type = "Packet_Base"

    results_file_name = "SN_Tracker_OSTD_Results_459"

    # 创建结果保存路径
    result_root_save_path = f"../Result/CrossRing/{model_type}/FOP/{results_file_name}/"
    os.makedirs(result_root_save_path, exist_ok=True)  # 确保根目录存在

    output_csv = os.path.join(r"../Result/Params_csv/", f"{results_file_name}.csv")
    os.makedirs(result_root_save_path, exist_ok=True)

    if topo_type in ["4x9", "5x4", "4x5"]:
        rn_r_tracker_ostd = 56
        rn_w_tracker_ostd = 64
    elif topo_type in ["9x4"]:
        rn_r_tracker_ostd = 48
        rn_w_tracker_ostd = 48

    # 定义参数范围
    parm1_start, parm1_end, parm1_step = (24, 96, 8)
    parm2_start, parm2_end, parm2_step = (24, 96, 8)

    # 遍历参数组合
    for parm1 in range(parm1_start, parm1_end + 1, parm1_step):  # 使用 parm1_end + 1 以包含结束值
        for parm2 in range(parm2_start, parm2_end + 1, parm2_step):  # 使用 parm2_end + 1 以包含结束值
            # 创建特定结果保存路径
            result_part_save_path = f"{parm1}_{parm2}/"  # 使用下划线分隔参数
            # 初始化模拟实例
            sim = eval(f"{model_type}_model")(
                model_type=model_type,
                config_path=config_path,
                topo_type=topo_type,
                traffic_file_path=traffic_file_path,
                file_name=file_name,
                result_save_path=result_root_save_path + result_part_save_path,
            )
            sim.config.rn_read_tracker_ostd = rn_r_tracker_ostd
            sim.config.rn_write_tracker_ostd = rn_w_tracker_ostd
            sim.config.rn_rdb_size = sim.config.rn_read_tracker_ostd * 4
            sim.config.rn_wdb_size = sim.config.rn_write_tracker_ostd * 4
            sim.config.ro_tracker_ostd = parm1
            sim.config.share_tracker_ostd = parm2
            sim.config.sn_wdb_size = sim.config.share_tracker_ostd * 4
            # sim.config.seats_per_link = parm2
            # sim.config.inject_queues_len = parm1
            # sim.config.eject_queues_len = parm2

            # sim.config.update_config()
            sim.initial()
            sim.end_time = 60000
            sim.print_interval = 10000
            print(f"Parm1: {parm1}, Parm2: {parm2}")

            # 运行模拟
            sim.run()

            # 计算并保存总结果
            # output_csv = os.path.join(r"../Result/Params_csv/", f"{results_file_name}.csv")
            # os.makedirs(result_root_save_path, exist_ok=True)
            # csv_file_exists = os.path.isfile(output_csv)

            # 准备写入结果
            results = {
                # "network_frequency": sim.config.network_frequency,
                "rn_w_tracker_ostd": sim.config.rn_write_tracker_ostd,
                "rn_r_tracker_ostd": sim.config.rn_read_tracker_ostd,
                "WriteBandWidth": sim.write_BW,
                "WriteAvgLatency": sim.write_latency_avg,
                "WriteMaxLatency": sim.write_latency_max,
                "share_tracker_ostd": sim.config.share_tracker_ostd,
                "ro_tracker_ostd": sim.config.ro_tracker_ostd,
                "ReadBandWidth": sim.read_BW,
                "ReadAvgLatency": sim.read_latency_avg,
                "ReadMaxLatency": sim.read_latency_max,
                "TotalBandWidth": sim.read_BW + sim.write_BW,
                "FinishCycle": sim.finish_time * sim.config.network_frequency,
                # "LBN": sim.config.seats_per_link,
                "eject_queues_len": sim.config.eject_queues_len,
                "inject_queues_len": sim.config.inject_queues_len,
                "sdma-R-DDR_thoughput": sim.sdma_R_ddr_flit_num * 128 / sim.sdma_R_ddr_finish_time / 4 if sim.sdma_R_ddr_finish_time > 0 else 0,
                "sdma-W-L2M_thoughput": sim.sdma_W_l2m_flit_num * 128 / sim.sdma_W_l2m_finish_time / 4 if sim.sdma_W_l2m_finish_time > 0 else 0,
                "gdma-R-L2M_thoughput": sim.gdma_R_l2m_flit_num * 128 / sim.gdma_R_l2m_finish_time / 4 if sim.gdma_R_l2m_finish_time > 0 else 0,
                "sdma-R-DDR_finish_cycle": sim.sdma_R_ddr_finish_time,
                "sdma-W-l2m_finish_cycle": sim.sdma_W_l2m_finish_time,
                "gdma-R-L2M_finish_cycle": sim.gdma_R_l2m_finish_time,
                "sdma-R-DDR_avg_latency": np.average(sim.sdma_R_ddr_latency) if sim.sdma_R_ddr_latency else 0,
                "sdma-W-l2m_avg_latency": np.average(sim.sdma_W_l2m_latency) if sim.sdma_W_l2m_latency else 0,
                "gdma-R-L2M_avg_latency": np.average(sim.gdma_R_l2m_latency) if sim.gdma_R_l2m_latency else 0,
                "req_cir_h_total": sim.req_cir_h_total,
                "req_cir_v_total": sim.req_cir_v_total,
                "rsp_cir_h_total": sim.rsp_cir_h_total,
                "rsp_cir_v_total": sim.rsp_cir_v_total,
                "data_cir_h_total": sim.data_cir_h_total,
                "data_cir_v_total": sim.data_cir_v_total,
                "read_retry_num": sim.read_retry_num,
                "write_retry_num": sim.write_retry_num,
                "model_type": sim.model_type,
                "Topo": topo_type,
            }

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
