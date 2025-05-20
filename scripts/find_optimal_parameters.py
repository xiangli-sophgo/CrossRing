from src.core import *
import os
from src.utils.component import Flit, Network, Node
from config.config import CrossRingConfig
import numpy as np
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm


def find_optimal_parameters():
    import csv

    # 定义流量文件路径和文件名
    traffic_file_path = r"../test_data/"
    file_name = r"traffic_2260E_case2.txt"

    # traffic_file_path = r"../traffic/"
    # traffic_file_path = r"../traffic/output-v7-32/step6_mesh_32core_map/"
    # file_name = r"LLama2_Attention_FC_Trace.txt"
    # file_name = r"LLama2_Attention_QKV_Decode_Trace.txt"
    # file_name = r"LLama2_MLP_Trace.txt"
    # file_name = r"LLama2_MM_QKV_Trace.txt"

    config_path = r"../config/config2.json"
    config = CrossRingConfig(config_path)

    # 定义拓扑类型
    if not config.topo_type:
        # topo_type = "4x9"
        # topo_type = "9x4"
        # topo_type = "5x4"
        # topo_type = "4x5"

        # topo_type = "6x5"

        topo_type = "3x3"
    else:
        topo_type = config.topo_type

    config.topo_type = topo_type

    # result_save_path = None

    model_type = "REQ_RSP"
    # model_type = "Packet_Base"
    # model_type = "Feature"

    results_file_name = "2260E_ETag_case1_0520"
    # results_file_name = "inject_eject_queue_length"

    # 创建结果保存路径
    result_root_save_path = f"../Result/CrossRing/{model_type}/FOP/{results_file_name}/"
    os.makedirs(result_root_save_path, exist_ok=True)  # 确保根目录存在

    output_csv = os.path.join(r"../Result/Params_csv/", f"{results_file_name}.csv")
    os.makedirs(result_root_save_path, exist_ok=True)

    # 定义参数范围
    param1_start, param1_end, param1_step = (2, 20, 1)
    param2_start, param2_end, param2_step = (2, 20, 1)
    param3_start, param3_end, param3_step = (2, 20, 1)
    param4_start, param4_end, param3_step = (8, 20, 1)

    def _run_one(param1, param2, param3, param4):
        """在给定参数下运行一次模拟，返回结果字典和参数元组。"""
        # 重新加载配置
        cfg = CrossRingConfig(config_path)
        cfg.topo_type = topo_type
        # 构造模拟对象
        sim = REQ_RSP_model(
            model_type=model_type,
            config=cfg,
            topo_type=topo_type,
            traffic_file_path=traffic_file_path,
            file_name=file_name,
            result_save_path=result_root_save_path + f"{param1}_{param2}/",
        )
        if topo_type == "3x3":
            sim.config.burst = 2
            sim.config.num_ips = 4
            sim.config.num_ddr = 8
            sim.config.num_l2m = 4
            sim.config.num_gdma = 4
            sim.config.num_sdma = 4
            sim.config.num_RN = 4
            sim.config.num_SN = 8
            sim.config.rn_read_tracker_ostd = 128
            sim.config.rn_write_tracker_ostd = 32
            sim.config.rn_rdb_size = sim.config.rn_read_tracker_ostd * sim.config.burst
            sim.config.rn_wdb_size = sim.config.rn_write_tracker_ostd * sim.config.burst
            sim.config.sn_ddr_read_tracker_ostd = 32
            sim.config.sn_ddr_write_tracker_ostd = 16
            sim.config.sn_l2m_read_tracker_ostd = 64
            sim.config.sn_l2m_write_tracker_ostd = 64
            sim.config.sn_ddr_wdb_size = sim.config.sn_ddr_write_tracker_ostd * sim.config.burst
            sim.config.sn_l2m_wdb_size = sim.config.sn_l2m_write_tracker_ostd * sim.config.burst
            sim.config.ddr_R_latency_original = 155
            sim.config.ddr_R_latency_var_original = 25
            # sim.config.ddr_R_latency_original = 0
            # sim.config.ddr_R_latency_var_original = 0
            sim.config.ddr_W_latency_original = 16
            sim.config.l2m_R_latency_original = 12
            sim.config.l2m_W_latency_original = 16
            sim.config.ddr_bandwidth_limit = 76.8 / 4
            # sim.config.ddr_bandwidth_limit = 10
            sim.config.l2m_bandwidth_limit = np.inf
            sim.config.IQ_CH_FIFO_DEPTH = 10
            sim.config.EQ_CH_FIFO_DEPTH = 16
            sim.config.IQ_OUT_FIFO_DEPTH = 8
            sim.config.RB_OUT_FIFO_DEPTH = 8

            # sim.config.EQ_IN_FIFO_DEPTH = 8
            # sim.config.RB_IN_FIFO_DEPTH = 8
            # sim.config.TL_Etag_T2_UE_MAX = 4
            # sim.config.TL_Etag_T1_UE_MAX = 7
            # sim.config.TR_Etag_T2_UE_MAX = 5
            # sim.config.TU_Etag_T2_UE_MAX = 4
            # sim.config.TU_Etag_T1_UE_MAX = 7
            # sim.config.TD_Etag_T3_UE_MAX = 6

            sim.config.EQ_IN_FIFO_DEPTH = 16
            sim.config.RB_IN_FIFO_DEPTH = 16
            sim.config.TL_Etag_T2_UE_MAX = 8
            sim.config.TL_Etag_T1_UE_MAX = 14
            sim.config.TR_Etag_T2_UE_MAX = 9
            sim.config.TU_Etag_T2_UE_MAX = 8
            sim.config.TU_Etag_T1_UE_MAX = 14
            sim.config.TD_Etag_T3_UE_MAX = 9

            sim.config.gdma_rw_gap = np.inf
            # sim.config.sdma_rw_gap = np.inf
            sim.config.sdma_rw_gap = 200
            sim.config.CHANNEL_SPEC = {
                "gdma": 1,
                "sdma": 1,
                "ddr": 4,
                "l2m": 2,
            }

        # 应用参数到 cfg
        sim.config.TL_Etag_T2_UE_MAX = param1
        sim.config.TL_Etag_T1_UE_MAX = param2
        sim.config.TR_Etag_T3_UE_MAX = param3
        sim.config.RB_IN_FIFO_DEPTH = param4
        # sim.config.TU_Etag_T2_UE_MAX = param1
        # sim.config.TU_Etag_T1_UE_MAX = param2
        # sim.config.TD_Etag_T2_UE_MAX = param3
        # 其他固定设置保持不变（可按原逻辑补充）
        sim.initial()
        sim.end_time = 10000
        sim.print_interval = 10000
        sim.run()
        results = sim.get_results()
        results.update({"param1": param1, "param2": param2, "param3": param3})
        return results

    # 构造参数组合列表，满足 param2 > param1
    param1_vals = list(range(param1_start, param1_end + 1, param1_step))
    param2_vals = list(range(param2_start, param2_end + 1, param2_step))
    param3_vals = list(range(param3_start, param3_end + 1, param3_step))
    param4_vals = list(range(param4_start, param4_end + 1, param3_step))
    combos = [(p1, p2, p3, p4) for p1 in param1_vals for p2 in param2_vals for p3 in param3_vals for p4 in param4_vals if (p4 > p2 > p1) and (p4 > p3)]
    # 并行执行
    all_results = Parallel(n_jobs=4)(delayed(_run_one)(p1, p2, p3, p4) for (p1, p2, p3, p4) in tqdm(combos, desc="Searching"))
    # 将所有结果写入 CSV
    csv_file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode="a", newline="") as output_csv_file:
        writer = csv.DictWriter(output_csv_file, fieldnames=all_results[0].keys())
        if not csv_file_exists:
            writer.writeheader()
        for res in all_results:
            writer.writerow(res)


if __name__ == "__main__":
    find_optimal_parameters()
