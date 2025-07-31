import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from find_optimal_parameters import (
    # 导入所有可视化和分析函数
    enhanced_create_visualization_plots,
    create_summary_report,
    save_progress_callback,
    create_intermediate_visualization,
    visualization_data,
    N_JOBS,
    N_REPEATS,
    N_TRIALS,
    # 导入2D热力图等可视化函数
    create_2d_param_bw_heatmaps,
    create_parameter_importance,
    create_optimization_history,
    create_parameter_correlation,
)
from src.core import *
from src.utils.component import Flit, Network, Node
from config.config import CrossRingConfig
import numpy as np
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
import optuna
import pandas as pd
import csv
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from optuna.exceptions import TrialPruned
from optuna.trial import TrialState

# ========== FIFO参数配置（全局统一） ==========
FIFO_PARAM_RANGES = {"IQ_CH_FIFO_DEPTH": {"start": 2, "end": 15}, "EQ_CH_FIFO_DEPTH": {"start": 2, "end": 15}}


def save_intermediate_result(study, trial, output_csv_path):
    """保存已完成 (COMPLETE) 的 trial 到 CSV，并创建实时可视化"""
    records = []
    for t in study.trials:
        if t.state != TrialState.COMPLETE:
            continue
        rec = {
            "number": t.number,
            "values": t.values,
            "state": t.state.name,
        }
        rec.update(t.params)
        rec.update(t.user_attrs)
        records.append(rec)

    # 保存CSV
    pd.DataFrame(records).to_csv(output_csv_path, index=False)

    # 保存进度并创建实时可视化
    save_progress_callback_local(study, trial, output_csv_path)


def save_progress_callback_local(study, trial, output_csv_path):
    """本地版本的save_progress_callback，避免全局变量问题"""
    global visualization_data

    # 更新可视化数据
    if trial.state == TrialState.COMPLETE and trial.values is not None:
        trial_data = {"number": trial.number, "values": trial.values, "params": trial.params.copy(), "user_attrs": trial.user_attrs.copy(), "timestamp": datetime.now().isoformat()}
        visualization_data["trials"].append(trial_data)

        # 保存进度数据
        progress_data = {
            "trial_number": trial.number,
            "best_values": study.best_trials[0].values if study.best_trials else None,
            "current_values": trial.values,
            "timestamp": datetime.now().isoformat(),
        }
        visualization_data["progress"].append(progress_data)

    # 每10个trial保存一次中间结果
    if trial.number % 10 == 0:
        try:
            # 保存到JSON文件
            progress_file = output_csv_path.replace(".csv", "_progress.json")
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(visualization_data, f, indent=2, ensure_ascii=False)

            # 如果有足够的数据，生成中间可视化
            if len(visualization_data["trials"]) >= 20:
                create_intermediate_visualization(study)

        except Exception as e:
            print(f"保存进度数据失败: {e}")


def find_optimal_fifo_depth():
    """专门用于寻找IQ_CH_FIFO_DEPTH和EQ_CH_FIFO_DEPTH最优参数的函数"""
    global output_csv

    # traffic_file_path = r"../test_data/"
    traffic_file_path = r"../traffic/0617/"

    # ===== 多个traffic文件配置 =====
    traffic_files = [
        r"LLama2_AllReduce.txt",
        # r"LLama2_AttentionFC.txt",
        # r"MLP_MoE.txt",
        # r"MLP.txt",
    ]

    # 每个traffic的权重（用于加权平均）
    # traffic_weights = [0.4, 0.2, 0.2, 0.2]
    traffic_weights = [1]

    assert len(traffic_files) == len(traffic_weights), "traffic文件数量和权重数量必须一致"
    assert abs(sum(traffic_weights) - 1.0) < 1e-6, "权重总和必须等于1"

    config_path = r"../config/config2.json"
    config = CrossRingConfig(config_path)

    # topo_type = "3x3"
    topo_type = "5x4"
    config.TOPO_TYPE = topo_type

    model_type = "REQ_RSP"
    results_file_name = f"FIFO_DEPTH_optimization_{datetime.now().strftime('%m%d_%H%M')}"
    result_root_save_path = f"../Result/CrossRing/{model_type}/FIFO_OPT/{results_file_name}/"
    os.makedirs(result_root_save_path, exist_ok=True)
    output_csv = os.path.join(r"../Result/Params_csv/", f"{results_file_name}.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # FIFO深度参数范围（使用统一配置）
    iq_ch_fifo_start, iq_ch_fifo_end = FIFO_PARAM_RANGES["IQ_CH_FIFO_DEPTH"]["start"], FIFO_PARAM_RANGES["IQ_CH_FIFO_DEPTH"]["end"]
    eq_ch_fifo_start, eq_ch_fifo_end = FIFO_PARAM_RANGES["EQ_CH_FIFO_DEPTH"]["start"], FIFO_PARAM_RANGES["EQ_CH_FIFO_DEPTH"]["end"]

    def _run_one_traffic(traffic_file, iq_ch_fifo_depth, eq_ch_fifo_depth):
        """运行单个traffic文件的仿真"""
        tot_bw_list = []
        for rpt in range(N_REPEATS):
            cfg = CrossRingConfig(config_path)
            cfg.TOPO_TYPE = topo_type
            sim = REQ_RSP_model(
                model_type=model_type,
                config=cfg,
                topo_type=topo_type,
                traffic_file_path=traffic_file_path,
                traffic_config=traffic_file,
                result_save_path=result_root_save_path,
                verbose=1,
            )

            # --- 固定平台参数 ------------------------------
            if topo_type == "3x3":
                sim.config.BURST = 2
                sim.config.NUM_IP = 4
                sim.config.NUM_DDR = 8
                sim.config.NUM_L2M = 4
                sim.config.NUM_GDMA = 4
                sim.config.NUM_SDMA = 4
                sim.config.NUM_RN = 4
                sim.config.NUM_SN = 8
                sim.config.RN_R_TRACKER_OSTD = 128
                sim.config.RN_W_TRACKER_OSTD = 32
                sim.config.RN_RDB_SIZE = sim.config.RN_R_TRACKER_OSTD * sim.config.BURST
                sim.config.RN_WDB_SIZE = sim.config.RN_W_TRACKER_OSTD * sim.config.BURST
                sim.config.SN_DDR_R_TRACKER_OSTD = 32
                sim.config.SN_DDR_W_TRACKER_OSTD = 16
                sim.config.SN_L2M_R_TRACKER_OSTD = 64
                sim.config.SN_L2M_W_TRACKER_OSTD = 64
                sim.config.SN_DDR_WDB_SIZE = sim.config.SN_DDR_W_TRACKER_OSTD * sim.config.BURST
                sim.config.SN_L2M_WDB_SIZE = sim.config.SN_L2M_W_TRACKER_OSTD * sim.config.BURST
                sim.config.DDR_R_LATENCY_original = 155
                sim.config.DDR_R_LATENCY_VAR_original = 25
                sim.config.DDR_W_LATENCY_original = 16
                sim.config.L2M_R_LATENCY_original = 12
                sim.config.L2M_W_LATENCY_original = 16
                sim.config.DDR_BW_LIMIT = 76.8 / 4
                sim.config.L2M_BW_LIMIT = np.inf
                sim.config.IQ_OUT_FIFO_DEPTH_HORIZONTAL = 8
                sim.config.IQ_OUT_FIFO_DEPTH_VERTICAL = 8
                sim.config.IQ_OUT_FIFO_DEPTH_EQ = 8
                sim.config.RB_OUT_FIFO_DEPTH = 8
                sim.config.EQ_IN_FIFO_DEPTH = 16
                sim.config.RB_IN_FIFO_DEPTH = 16
                sim.config.TL_Etag_T2_UE_MAX = 8
                sim.config.TL_Etag_T1_UE_MAX = 14
                sim.config.TR_Etag_T2_UE_MAX = 9
                sim.config.TU_Etag_T2_UE_MAX = 8
                sim.config.TU_Etag_T1_UE_MAX = 14
                sim.config.TD_Etag_T2_UE_MAX = 9
                sim.config.GDMA_RW_GAP = np.inf
                sim.config.SDMA_RW_GAP = 50
                sim.config.ETag_BOTHSIDE_UPGRADE = 0
                sim.config.CHANNEL_SPEC = {
                    "gdma": 1,
                    "sdma": 1,
                    "ddr": 4,
                    "l2m": 2,
                }
            elif topo_type in ["5x4", "4x5"]:
                sim.config.BURST = 4
                sim.config.NUM_IP = 32
                sim.config.NUM_DDR = 32
                sim.config.NUM_L2M = 32
                sim.config.NUM_GDMA = 32
                sim.config.NUM_SDMA = 32
                sim.config.NUM_RN = 32
                sim.config.NUM_SN = 32
                sim.config.RN_R_TRACKER_OSTD = 64
                sim.config.RN_W_TRACKER_OSTD = 64
                sim.config.RN_RDB_SIZE = sim.config.RN_R_TRACKER_OSTD * sim.config.BURST
                sim.config.RN_WDB_SIZE = sim.config.RN_W_TRACKER_OSTD * sim.config.BURST
                sim.config.SN_DDR_R_TRACKER_OSTD = 64
                sim.config.SN_DDR_W_TRACKER_OSTD = 64
                sim.config.SN_L2M_R_TRACKER_OSTD = 64
                sim.config.SN_L2M_W_TRACKER_OSTD = 64
                sim.config.SN_DDR_WDB_SIZE = sim.config.SN_DDR_W_TRACKER_OSTD * sim.config.BURST
                sim.config.SN_L2M_WDB_SIZE = sim.config.SN_L2M_W_TRACKER_OSTD * sim.config.BURST
                sim.config.DDR_R_LATENCY_original = 40
                sim.config.DDR_R_LATENCY_VAR_original = 0
                sim.config.DDR_W_LATENCY_original = 0
                sim.config.L2M_R_LATENCY_original = 12
                sim.config.L2M_W_LATENCY_original = 16
                sim.config.IQ_OUT_FIFO_DEPTH_HORIZONTAL = 8
                sim.config.IQ_OUT_FIFO_DEPTH_VERTICAL = 8
                sim.config.IQ_OUT_FIFO_DEPTH_EQ = 8
                sim.config.RB_OUT_FIFO_DEPTH = 8
                sim.config.SN_TRACKER_RELEASE_LATENCY = 40

                # 使用之前优化的最佳参数
                sim.config.TL_Etag_T2_UE_MAX = 8
                sim.config.TL_Etag_T1_UE_MAX = 15
                sim.config.TR_Etag_T2_UE_MAX = 12
                sim.config.RB_IN_FIFO_DEPTH = 16
                sim.config.TU_Etag_T2_UE_MAX = 8
                sim.config.TU_Etag_T1_UE_MAX = 15
                sim.config.TD_Etag_T2_UE_MAX = 12
                sim.config.EQ_IN_FIFO_DEPTH = 16

                sim.config.ITag_TRIGGER_Th_H = sim.config.ITag_TRIGGER_Th_V = 80
                sim.config.ITag_MAX_NUM_H = sim.config.ITag_MAX_NUM_V = 1
                sim.config.ETag_BOTHSIDE_UPGRADE = 0
                sim.config.SLICE_PER_LINK = 8

                sim.config.GDMA_RW_GAP = np.inf
                sim.config.SDMA_RW_GAP = np.inf
                sim.config.CHANNEL_SPEC = {
                    "gdma": 2,
                    "sdma": 2,
                    "ddr": 2,
                    "l2m": 2,
                }

            # --- 覆盖待优化的FIFO深度参数 ----------------------------
            sim.config.IQ_CH_FIFO_DEPTH = iq_ch_fifo_depth
            sim.config.EQ_CH_FIFO_DEPTH = eq_ch_fifo_depth

            try:
                sim.initial()
                sim.end_time = 5000
                sim.print_interval = 5000
                sim.run()
                bw = sim.get_results().get("mixed_avg_weighted_bw", 0)
            except Exception as e:
                print(f"[{traffic_file}][RPT {rpt}] Sim failed for FIFO params: IQ_CH={iq_ch_fifo_depth}, EQ_CH={eq_ch_fifo_depth}")
                print("Exception details (full traceback):")
                traceback.print_exc()
                bw = 0
            tot_bw_list.append(bw)

        bw_mean = float(np.mean(tot_bw_list))
        bw_std = float(np.std(tot_bw_list))

        return {
            f"mixed_avg_weighted_bw_mean_{traffic_file[:-4]}": bw_mean,
            f"mixed_avg_weighted_bw_std_{traffic_file[:-4]}": bw_std,
        }

    def _run_one(iq_ch_fifo_depth, eq_ch_fifo_depth):
        """运行所有traffic文件并综合结果"""
        all_results = {}
        all_bw_means = []

        for traffic_file in traffic_files:
            try:
                result = _run_one_traffic(traffic_file, iq_ch_fifo_depth, eq_ch_fifo_depth)
                all_results.update(result)
                bw_mean = result[f"mixed_avg_weighted_bw_mean_{traffic_file[:-4]}"]
                all_bw_means.append(bw_mean)
            except Exception as e:
                print(f"Error processing {traffic_file}: {e}")
                all_bw_means.append(0)
                all_results[f"mixed_avg_weighted_bw_mean_{traffic_file[:-4]}"] = 0
                all_results[f"mixed_avg_weighted_bw_std_{traffic_file[:-4]}"] = 0

        # 计算加权平均带宽
        weighted_bw_mean = sum(bw * weight for bw, weight in zip(all_bw_means, traffic_weights))

        # 计算最小带宽（保证所有traffic都有合理性能）
        min_bw_mean = min(all_bw_means) if all_bw_means else 0

        # 计算带宽方差（衡量不同traffic间的一致性）
        bw_variance = np.var(all_bw_means) if len(all_bw_means) > 1 else 0

        # 添加综合指标
        all_results.update(
            {
                "mixed_avg_weighted_bw_weighted_mean": weighted_bw_mean,
                "mixed_avg_weighted_bw_min": min_bw_mean,
                "mixed_avg_weighted_bw_variance": bw_variance,
                "IQ_CH_FIFO_DEPTH": iq_ch_fifo_depth,
                "EQ_CH_FIFO_DEPTH": eq_ch_fifo_depth,
            }
        )

        return all_results

    def objective(trial):
        # 采样FIFO深度参数
        iq_ch_fifo = trial.suggest_int("IQ_CH_FIFO_DEPTH", iq_ch_fifo_start, iq_ch_fifo_end)
        eq_ch_fifo = trial.suggest_int("EQ_CH_FIFO_DEPTH", eq_ch_fifo_start, eq_ch_fifo_end)

        results = _run_one(iq_ch_fifo, eq_ch_fifo)

        # 获取加权平均带宽
        weighted_bw = results["mixed_avg_weighted_bw_weighted_mean"]

        # FIFO深度资源消耗惩罚（更大的FIFO消耗更多资源）
        fifo_penalty = ((iq_ch_fifo - iq_ch_fifo_start) / (iq_ch_fifo_end - iq_ch_fifo_start) + (eq_ch_fifo - eq_ch_fifo_start) / (eq_ch_fifo_end - eq_ch_fifo_start)) / 2.0

        # 综合指标 = 加权带宽 - α * FIFO惩罚
        # 完全取消FIFO惩罚，只优化带宽
        penalty_weight = 0  # 完全取消惩罚
        composite_metric = weighted_bw - penalty_weight * fifo_penalty

        # 保存到 trial.user_attrs，便于后期分析 / CSV
        for k, v in results.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("fifo_penalty", fifo_penalty)
        trial.set_user_attr("penalty_weight", penalty_weight)
        trial.set_user_attr("composite_metric", composite_metric)

        return composite_metric

    return objective, output_csv, traffic_files, traffic_weights, result_root_save_path


def analyze_existing_results(result_path):
    """重新分析已有结果"""
    import joblib

    if not result_path:
        print("请提供结果路径")
        return

    # 加载Study对象
    study_file = os.path.join(result_path, "optuna_study.pkl")
    if not os.path.exists(study_file):
        print(f"未找到Study文件: {study_file}")
        return

    print(f"正在加载Study对象: {study_file}")
    study = joblib.load(study_file)

    # 加载配置
    config_file = os.path.join(result_path, "optimization_config.json")
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        traffic_files = config.get("traffic_files", [])
        traffic_weights = config.get("traffic_weights", [])
    else:
        traffic_files = ["LLama2_AllReduce.txt"]
        traffic_weights = [1.0]

    print("=" * 60)
    print(f"重新分析FIFO深度优化结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果路径: {result_path}")
    print(f"Traffic文件: {traffic_files}")
    print("=" * 60)

    # 重新生成所有可视化
    try:
        vis_dir = os.path.join(result_path, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 生成FIFO参数的2D热力图
        create_2d_param_bw_heatmaps(study.trials, metric_key="mixed_avg_weighted_bw_weighted_mean", param_pairs=[("IQ_CH_FIFO_DEPTH", "EQ_CH_FIFO_DEPTH")], save_dir=vis_dir)

        # 添加其他可视化函数
        create_parameter_importance(study, vis_dir)
        create_optimization_history(study.trials, vis_dir)
        create_parameter_correlation(study.trials, vis_dir)

        print(f"重新分析完成，可视化结果保存到: {vis_dir}/")

    except Exception as e:
        print(f"重新分析失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # 检查是否为重分析模式
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze-only":
        result_path = sys.argv[2] if len(sys.argv) > 2 else None
        analyze_existing_results(result_path)
    else:
        # 正常优化模式
        objective, output_csv, traffic_files, traffic_weights, result_root_save_path = find_optimal_fifo_depth()

        print("=" * 60)
        print(f"开始FIFO深度优化 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"优化参数: IQ_CH_FIFO_DEPTH, EQ_CH_FIFO_DEPTH")
        print(f"Traffic文件: {traffic_files}")
        print(f"权重: {traffic_weights}")
        print(f"结果保存路径: {result_root_save_path}")
        print("=" * 60)

        n_trials = 300

    study = optuna.create_study(
        study_name="CrossRing_FIFO_DEPTH_BO",
        direction="maximize",
        sampler=optuna.samplers.NSGAIISampler(),
    )

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=N_JOBS,
            show_progress_bar=True,
            callbacks=[lambda study, trial: save_intermediate_result(study, trial, output_csv)],
        )
    except KeyboardInterrupt:
        print("优化被用户中断")

    # 保存最终结果
    final_records = []
    for t in study.trials:
        if t.state != TrialState.COMPLETE:
            continue
        rec = {
            "number": t.number,
            "value": t.values,
            "state": t.state.name,
        }
        rec.update(t.params)
        rec.update(t.user_attrs)
        final_records.append(rec)

    final_df = pd.DataFrame(final_records)
    final_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 60)
    print("FIFO深度优化完成!")
    if study.best_trials:
        print("最佳指标:", study.best_trials[0].values)
        print("最佳FIFO深度参数:", study.best_trials[0].params)

    # 显示最佳结果的详细信息
    if study.best_trials:
        best_trial = study.best_trials[0]
        print("\n最佳配置的详细结果:")
        for traffic_file in traffic_files:
            traffic_name = traffic_file[:-4]
            if f"mixed_avg_weighted_bw_mean_{traffic_name}" in best_trial.user_attrs:
                print(f"  {traffic_name}: {best_trial.user_attrs[f'mixed_avg_weighted_bw_mean_{traffic_name}']:.2f} GB/s")
        print(f"  加权平均: {best_trial.user_attrs.get('mixed_avg_weighted_bw_weighted_mean', 0):.2f} GB/s")
        print(f"  最小值: {best_trial.user_attrs.get('mixed_avg_weighted_bw_min', 0):.2f} GB/s")
        print(f"  方差: {best_trial.user_attrs.get('mixed_avg_weighted_bw_variance', 0):.2f}")
        print(f"  IQ_CH_FIFO_DEPTH: {best_trial.params.get('IQ_CH_FIFO_DEPTH', 0)}")
        print(f"  EQ_CH_FIFO_DEPTH: {best_trial.params.get('EQ_CH_FIFO_DEPTH', 0)}")

    # 创建最终可视化
    print("\n正在生成最终可视化报告...")
    try:
        # 不使用通用的enhanced_create_visualization_plots，因为它包含了不适用的参数
        # enhanced_create_visualization_plots(study, traffic_files, traffic_weights, result_root_save_path)

        # 添加2D热力图和其他可视化函数
        vis_dir = os.path.join(result_root_save_path, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 生成FIFO参数的2D热力图（只针对FIFO参数）
        create_2d_param_bw_heatmaps(study.trials, metric_key="mixed_avg_weighted_bw_weighted_mean", param_pairs=[("IQ_CH_FIFO_DEPTH", "EQ_CH_FIFO_DEPTH")], save_dir=vis_dir)

        # 添加其他有用的可视化函数（只针对FIFO参数）
        create_parameter_importance(study, vis_dir)
        create_optimization_history(study.trials, vis_dir)

        # 跳过correlation分析，因为只有2个参数且可能常数输入
        # create_parameter_correlation(study.trials, vis_dir)

        print(f"可视化报告已生成: {vis_dir}/")

        # 创建总结报告
        create_summary_report(study, traffic_files, traffic_weights, result_root_save_path)

    except Exception as e:
        print(f"生成可视化失败: {e}")
        traceback.print_exc()

    print("=" * 60)

    # 保存Study对象和配置
    study_file = os.path.join(result_root_save_path, "optuna_study.pkl")
    import joblib

    joblib.dump(study, study_file)
    print(f"Study对象已保存: {study_file}")

    # 保存优化配置
    config_data = {
        "optimization_target": "IQ_CH_FIFO_DEPTH and EQ_CH_FIFO_DEPTH",
        "traffic_files": traffic_files,
        "traffic_weights": traffic_weights,
        "param_ranges": {
            "IQ_CH_FIFO_DEPTH": [FIFO_PARAM_RANGES["IQ_CH_FIFO_DEPTH"]["start"], FIFO_PARAM_RANGES["IQ_CH_FIFO_DEPTH"]["end"]],
            "EQ_CH_FIFO_DEPTH": [FIFO_PARAM_RANGES["EQ_CH_FIFO_DEPTH"]["start"], FIFO_PARAM_RANGES["EQ_CH_FIFO_DEPTH"]["end"]],
        },
        "n_trials": n_trials,
        "n_repeats": N_REPEATS,
        "timestamp": datetime.now().isoformat(),
        "result_root_save_path": result_root_save_path,
    }

    config_file = os.path.join(result_root_save_path, "optimization_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"优化配置已保存: {config_file}")

    print("\n📁 已保存以下文件用于后续分析:")
    print(f"  • Study对象: {study_file}")
    print(f"  • 配置文件: {config_file}")
    print(f"  • CSV数据: {output_csv}")
    print(f"  • HTML报告: {result_root_save_path}/optimization_report.html")
    print(f"  • 可视化: {result_root_save_path}/visualizations/")
