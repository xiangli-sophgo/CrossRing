from src.core import *
import os
from src.utils.component import Flit, Network, Node
from config.config import CrossRingConfig
import numpy as np
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
import optuna  # Bayesian optimization library

# 使用的 CPU 核心数；-1 表示全部核心
N_JOBS = 1


def find_optimal_parameters():
    import csv

    traffic_file_path = r"../test_data/"
    file_name = r"traffic_2260E_case2.txt"
    config_path = r"../config/config2.json"
    config = CrossRingConfig(config_path)

    topo_type = config.topo_type or "3x3"
    config.topo_type = topo_type

    model_type = "REQ_RSP"
    results_file_name = "2260E_ETag_case2_0521"
    result_root_save_path = f"../Result/CrossRing/{model_type}/FOP/{results_file_name}/"
    os.makedirs(result_root_save_path, exist_ok=True)
    output_csv = os.path.join(r"../Result/Params_csv/", f"{results_file_name}.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 参数范围
    param1_start, param1_end, param1_step = (2, 20, 1)
    param2_start, param2_end, param2_step = (2, 20, 1)
    param3_start, param3_end, param3_step = (2, 20, 1)
    param4_start, param4_end, param4_step = (8, 20, 1)

    def _run_one(param1, param2, param3, param4):
        cfg = CrossRingConfig(config_path)
        cfg.topo_type = topo_type
        sim = REQ_RSP_model(
            model_type=model_type,
            config=cfg,
            topo_type=topo_type,
            traffic_file_path=traffic_file_path,
            file_name=file_name,
            result_save_path=result_root_save_path + f"{param1}_{param2}_{param3}_{param4}/",
        )
        # ...（此处省略参数设置，与你原来一致）...
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
            sim.config.ddr_W_latency_original = 16
            sim.config.l2m_R_latency_original = 12
            sim.config.l2m_W_latency_original = 16
            sim.config.ddr_bandwidth_limit = 76.8 / 4
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
            sim.config.sdma_rw_gap = 50
            sim.config.CHANNEL_SPEC = {
                "gdma": 1,
                "sdma": 1,
                "ddr": 4,
                "l2m": 2,
            }

        # 应用参数
        sim.config.TL_Etag_T2_UE_MAX = param1
        sim.config.TL_Etag_T1_UE_MAX = param2
        sim.config.TR_Etag_T3_UE_MAX = param3
        sim.config.RB_IN_FIFO_DEPTH = param4

        try:
            sim.initial()
            sim.end_time = 1000
            sim.print_interval = 10000
            sim.run()
            results = sim.get_results()
        except Exception as e:
            print(f"Sim failed for params: {param1}, {param2}, {param3}, {param4}, error: {str(e)}")
            results = {}

        results.update({"param1": param1, "param2": param2, "param3": param3, "param4": param4})
        # 确保有 avg_latency 字段
        if "Total_sum_BW" not in results:
            results["Total_sum_BW"] = results.get("Total_sum_BW", -1e9)
        return results

    def objective(trial):
        # 采样参数
        p1 = trial.suggest_int("TL_Etag_T2_UE_MAX", param1_start, param1_end)
        # 保证 p2 > p1
        p2_low = p1 + 1
        if p2_low > param2_end:
            trial.set_user_attr("skip", True)
            return -1e9
        p2 = trial.suggest_int("TL_Etag_T1_UE_MAX", p2_low, param2_end)
        p3 = trial.suggest_int("TR_Etag_T3_UE_MAX", param3_start, param3_end)
        p4_low = max(p2 + 1, param4_start)
        if p4_low > param4_end:
            trial.set_user_attr("skip", True)
            return -1e9
        p4 = trial.suggest_int("RB_IN_FIFO_DEPTH", p4_low, param4_end)

        results = _run_one(p1, p2, p3, p4)
        score = results.get("Total_sum_BW", -1e9)
        for k, v in results.items():
            trial.set_user_attr(k, v)
        return score

    return objective, output_csv


if __name__ == "__main__":
    objective, output_csv = find_optimal_parameters()
    study = optuna.create_study(
        study_name="CrossRing_BO",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=200, n_jobs=N_JOBS, show_progress_bar=True)
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    df.to_csv(output_csv, index=False)
    print("最佳指标:", study.best_value)
    print("最佳参数:", study.best_params)
