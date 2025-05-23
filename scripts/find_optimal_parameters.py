from src.core import *
import os
from src.utils.component import Flit, Network, Node
from config.config import CrossRingConfig
import numpy as np
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
import optuna  # Bayesian optimization library
import pandas as pd
import csv
import traceback

from optuna.exceptions import TrialPruned
from optuna.trial import TrialState

# 使用的 CPU 核心数；-1 表示全部核心
N_JOBS = -1
# 每个参数组合重复仿真次数，用于平滑随机 latency 影响
N_REPEATS = 3  # 可根据需求调大
# 参数规模惩罚系数（与 BW 同量级，Total_sum_BW ≈ 1000）
ALPHA = 0.5  # 惩罚权重，可调


def find_optimal_parameters():

    traffic_file_path = r"../test_data/"
    file_name = r"traffic_2260E_case1.txt"
    config_path = r"../config/config2.json"
    config = CrossRingConfig(config_path)

    topo_type = config.TOPO_TYPE or "3x3"
    config.TOPO_TYPE = topo_type

    model_type = "REQ_RSP"
    results_file_name = "2260E_ETag_case1_0522"
    result_root_save_path = f"../Result/CrossRing/{model_type}/FOP/{results_file_name}/"
    os.makedirs(result_root_save_path, exist_ok=True)
    output_csv = os.path.join(r"../Result/Params_csv/", f"{results_file_name}.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 参数范围
    param1_start, param1_end = 2, 20
    param2_start, param2_end = 2, 20
    param3_start, param3_end = 2, 20
    param4_start, param4_end = 4, 20
    param5_start, param5_end = 2, 20
    param6_start, param6_end = 2, 20
    param7_start, param7_end = 2, 20
    param8_start, param8_end = 4, 20

    def _run_one(param1, param2, param3, param4, param5, param6, param7, param8):
        tot_bw_list = []
        for rpt in range(N_REPEATS):
            cfg = CrossRingConfig(config_path)
            cfg.TOPO_TYPE = topo_type
            sim = REQ_RSP_model(
                model_type=model_type,
                config=cfg,
                topo_type=topo_type,
                traffic_file_path=traffic_file_path,
                file_name=file_name,
                result_save_path=result_root_save_path,
                verbose=0,
            )

            # --- 固定平台参数 ------------------------------
            if topo_type == "3x3":
                sim.config.BURST = 2
                sim.config.NUM_IP = 4
                sim.config.NUM_DDR = 8
                sim.config.NUM_L2M = 4
                sim.config.NUM_GDMA = 4
                sim.config.NUM_SDMA = 4
                sim.config.num_RN = 4
                sim.config.num_SN = 8
                sim.config.rn_read_tracker_ostd = 128
                sim.config.rn_write_tracker_ostd = 32
                sim.config.RN_RDB_SIZE = sim.config.rn_read_tracker_ostd * sim.config.BURST
                sim.config.RN_WDB_SIZE = sim.config.rn_write_tracker_ostd * sim.config.BURST
                sim.config.sn_ddr_read_tracker_ostd = 32
                sim.config.sn_ddr_write_tracker_ostd = 16
                sim.config.sn_l2m_read_tracker_ostd = 64
                sim.config.sn_l2m_write_tracker_ostd = 64
                sim.config.SN_DDR_WDB_SIZE = sim.config.sn_ddr_write_tracker_ostd * sim.config.BURST
                sim.config.SN_L2M_WDB_SIZE = sim.config.sn_l2m_write_tracker_ostd * sim.config.BURST
                sim.config.DDR_R_LATENCY_original = 155
                sim.config.DDR_R_LATENCY_VAR_original = 25
                sim.config.DDR_W_LATENCY_original = 16
                sim.config.L2M_R_LATENCY_original = 12
                sim.config.L2M_W_LATENCY_original = 16
                sim.config.ddr_bandwidth_limit = 76.8 / 4
                sim.config.l2m_bandwidth_limit = np.inf
                sim.config.IQ_CH_FIFO_DEPTH = 10
                sim.config.EQ_CH_FIFO_DEPTH = 16
                sim.config.IQ_OUT_FIFO_DEPTH = 8
                sim.config.RB_OUT_FIFO_DEPTH = 8
                sim.config.EQ_IN_FIFO_DEPTH = 16
                sim.config.RB_IN_FIFO_DEPTH = 16
                sim.config.TL_Etag_T2_UE_MAX = 8
                sim.config.TL_Etag_T1_UE_MAX = 14
                sim.config.TR_Etag_T2_UE_MAX = 9
                sim.config.TU_Etag_T2_UE_MAX = 8
                sim.config.TU_Etag_T1_UE_MAX = 14
                sim.config.TD_Etag_T2_UE_MAX = 9
                sim.config.gdma_rw_gap = np.inf
                sim.config.sdma_rw_gap = 50
                sim.config.CHANNEL_SPEC = {
                    "gdma": 1,
                    "sdma": 1,
                    "ddr": 4,
                    "l2m": 2,
                }

            # --- 覆盖待优化参数 ----------------------------
            sim.config.TL_Etag_T2_UE_MAX = param1
            sim.config.TL_Etag_T1_UE_MAX = param2
            sim.config.TR_Etag_T2_UE_MAX = param3
            sim.config.RB_IN_FIFO_DEPTH = param4
            sim.config.TU_Etag_T2_UE_MAX = param5
            sim.config.TU_Etag_T1_UE_MAX = param6
            sim.config.TD_Etag_T2_UE_MAX = param7
            sim.config.EQ_IN_FIFO_DEPTH = param8

            try:
                sim.initial()
                sim.end_time = 10000
                sim.print_interval = 10000
                sim.run()
                bw = sim.get_results().get("Total_sum_BW", 0)
            except Exception as e:
                print(f"[RPT {rpt}] Sim failed for params: {param1}, {param2}, {param3}, {param4}, {param5}, {param6}, {param7}, {param8}")
                print("Exception stack trace:")
                traceback.print_exc()
                bw = 0
            tot_bw_list.append(bw)

        bw_mean = float(np.mean(tot_bw_list))
        bw_std = float(np.std(tot_bw_list))

        results = {
            "Total_sum_BW_mean": bw_mean,
            "Total_sum_BW_std": bw_std,
            "param1": param1,
            "param2": param2,
            "param3": param3,
            "param4": param4,
            "param5": param5,
            "param6": param6,
            "param7": param7,
            "param8": param8,
        }
        return results

    def objective(trial):
        # 采样参数
        p1 = trial.suggest_int("TL_Etag_T2_UE_MAX", param1_start, param1_end)
        # 保证 p2 > p1
        p2_low = p1 + 1
        if p2_low > param2_end:
            raise TrialPruned()  # 参数非法，直接剪枝，不计入有效 trial
        p2 = trial.suggest_int("TL_Etag_T1_UE_MAX", p2_low, param2_end)
        p3 = trial.suggest_int("TR_Etag_T2_UE_MAX", param3_start, param3_end)
        # 保证 p4 > max(p2, p3)
        p4_low = max(p2, p3) + 1
        if p4_low > param4_end:
            raise TrialPruned()  # 参数非法，直接剪枝，不计入有效 trial
        p4 = trial.suggest_int("RB_IN_FIFO_DEPTH", p4_low, param4_end)

        # 新增参数
        p5 = trial.suggest_int("TU_Etag_T2_UE_MAX", param5_start, param5_end)
        # 保证 p6 > p5
        p6_low = p5 + 1
        if p6_low > param6_end:
            raise TrialPruned()  # 参数非法，直接剪枝，不计入有效 trial
        p6 = trial.suggest_int("TU_Etag_T1_UE_MAX", p6_low, param6_end)
        p7 = trial.suggest_int("TD_Etag_T2_UE_MAX", param7_start, param7_end)
        # 保证 p8 > max(p6, p7)
        p8_low = max(p6, p7) + 1
        if p8_low > param8_end:
            raise TrialPruned()  # 参数非法，直接剪枝，不计入有效 trial
        p8 = trial.suggest_int("EQ_IN_FIFO_DEPTH", p8_low, param8_end)

        results = _run_one(p1, p2, p3, p4, p5, p6, p7, p8)
        # 归一化参数规模惩罚（越小越好）
        param_norm = (
            # (p1 - param1_start) / (param1_end - param1_start) +
            # (p2 - param2_start) / (param2_end - param2_start) +
            # (p3 - param3_start) / (param3_end - param3_start) +
            # (p5 - param5_start) / (param5_end - param5_start) +
            # (p6 - param6_start) / (param6_end - param6_start) +
            # (p7 - param7_start) / (param7_end - param7_start) +
            (p4 - param4_start) / (param4_end - param4_start)
            + (p8 - param8_start) / (param8_end - param8_start)
        ) / 2.0  # 平均化到 0~1
        score = results["Total_sum_BW_mean"] - ALPHA * param_norm * 200
        for k, v in results.items():
            trial.set_user_attr(k, v)
        return score

    return objective, output_csv


def save_intermediate_result(study, trial):
    """保存已完成 (COMPLETE) 的 trial 到 CSV。"""
    records = []
    for t in study.trials:
        if t.state != TrialState.COMPLETE:
            continue
        rec = {
            "number": t.number,
            "value": t.value,
            "state": t.state.name,
        }
        rec.update(t.params)
        rec.update(t.user_attrs)
        records.append(rec)
    pd.DataFrame(records).to_csv(output_csv, index=False)


if __name__ == "__main__":
    objective, output_csv = find_optimal_parameters()
    study = optuna.create_study(
        study_name="CrossRing_BO",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        objective,
        n_trials=500,
        n_jobs=N_JOBS,
        show_progress_bar=True,
        callbacks=[save_intermediate_result],
    )
    # 再存一次完整扁平化结果
    final_records = []
    for t in study.trials:
        if t.state != TrialState.COMPLETE:
            continue
        rec = {
            "number": t.number,
            "value": t.value,
            "state": t.state.name,
        }
        rec.update(t.params)
        rec.update(t.user_attrs)
        final_records.append(rec)
    pd.DataFrame(final_records).to_csv(output_csv, index=False)
    print("最佳指标:", study.best_value)
    print("最佳参数:", study.best_params)
