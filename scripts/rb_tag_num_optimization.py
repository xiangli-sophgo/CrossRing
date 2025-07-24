#!/usr/bin/env python3
"""
RB_ONLY_TAG数量优化脚本
使用Optuna优化框架寻找横向环和纵向环RB_ONLY标签数量的最优组合
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import csv
import json
import traceback
import pandas as pd
from datetime import datetime
from typing import List, Tuple

# 设置matplotlib中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import CrossRingConfig
from src.core.base_model_v2 import BaseModel
import optuna
from optuna.exceptions import TrialPruned
from optuna.trial import TrialState


# 配置参数
N_REPEATS = 1  # 每次仿真重复次数
N_TRIALS = 100  # Optuna优化试验次数

# RB_ONLY标签数量参数范围
RB_TAG_PARAM_RANGES = {"RB_ONLY_TAG_NUM_HORIZONTAL": {"start": 0, "end": 14}, "RB_ONLY_TAG_NUM_VERTICAL": {"start": 0, "end": 56}}


def save_intermediate_result(study, trial, output_csv_path):
    """保存已完成的trial到CSV文件"""
    records = []
    for t in study.trials:
        if t.state != TrialState.COMPLETE:
            continue
        rec = {
            "number": t.number,
            "value": t.values[0] if t.values else 0,
            "state": t.state.name,
        }
        rec.update(t.params)
        rec.update(t.user_attrs)
        records.append(rec)

    # 保存CSV
    if records:
        pd.DataFrame(records).to_csv(output_csv_path, index=False)
        if trial.number % 10 == 0:
            print(f"已完成 {len(records)} 个试验，中间结果已保存到: {output_csv_path}")


def create_2d_heatmap(study_trials, save_dir):
    """创建RB_ONLY标签数量的2D热力图"""
    if not study_trials:
        return

    # 提取完成的试验数据
    h_tags = []
    v_tags = []
    bandwidths = []

    for trial in study_trials:
        if trial.state == TrialState.COMPLETE and trial.values:
            h_tags.append(trial.params.get("RB_ONLY_TAG_NUM_HORIZONTAL", 0))
            v_tags.append(trial.params.get("RB_ONLY_TAG_NUM_VERTICAL", 0))
            # 从user_attrs中获取带宽数据
            bandwidth = trial.user_attrs.get("mixed_avg_weighted_bw_mean", trial.values[0] if trial.values else 0)
            bandwidths.append(bandwidth)

    if len(bandwidths) < 3:
        print("数据点太少，无法生成热力图")
        return

    # 创建热力图
    plt.figure(figsize=(12, 10))

    # 创建网格插值
    from scipy.interpolate import griddata

    h_range = np.arange(min(h_tags), max(h_tags) + 1)
    v_range = np.arange(min(v_tags), max(v_tags) + 1)
    H, V = np.meshgrid(h_range, v_range)

    # 插值到网格
    Z = griddata((h_tags, v_tags), bandwidths, (H, V), method="cubic", fill_value=0)

    # 绘制热力图
    im = plt.imshow(Z, cmap="YlGnBu", origin="lower", aspect="auto", extent=[min(h_tags), max(h_tags), min(v_tags), max(v_tags)])

    # 添加散点显示实际数据点
    scatter = plt.scatter(h_tags, v_tags, c=bandwidths, cmap="YlGnBu", s=50, edgecolors="black", alpha=0.8)

    plt.xlabel("横向环RB_ONLY标签数量", fontsize=14)
    plt.ylabel("纵向环RB_ONLY标签数量", fontsize=14)
    plt.title("RB_ONLY标签数量优化热力图", fontsize=16)

    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label("带宽指标 (GB/s)", fontsize=12)

    # 标注最优点
    if bandwidths:
        best_idx = np.argmax(bandwidths)
        best_h = h_tags[best_idx]
        best_v = v_tags[best_idx]
        best_bw = bandwidths[best_idx]
        plt.scatter(best_h, best_v, c="blue", s=200, marker="*", label=f"最优点({best_h},{best_v}): {best_bw:.3f}")
        plt.legend()

    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, "rb_tag_optimization_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"热力图已保存: {heatmap_path}")


def run_single_traffic(traffic_file, h_tags, v_tags, result_save_path):
    """运行单个traffic文件的仿真"""
    tot_bw_list = []

    for rpt in range(N_REPEATS):
        config = CrossRingConfig("../config/config2.json")
        config.TOPO_TYPE = "5x2"

        # 设置要测试的标签数量
        config.RB_ONLY_TAG_NUM_HORIZONTAL = h_tags
        config.RB_ONLY_TAG_NUM_VERTICAL = v_tags

        # 创建仿真实例
        sim = BaseModel(
            model_type="REQ_RSP",
            config=config,
            topo_type="5x2",
            traffic_file_path="../traffic/0617",
            traffic_config=traffic_file,
            result_save_path=result_save_path,
            verbose=0,  # 关闭详细输出
            print_trace=0,
            plot_link_state=0,
            plot_flow_fig=0,
            plot_RN_BW_fig=0,
        )

        try:
            sim.initial()
            sim.end_time = 6000
            sim.print_interval = 2000
            sim.run()
            bw = sim.get_results().get("mixed_avg_weighted_bw", 0)
        except Exception as e:
            print(f"[{traffic_file}][RPT {rpt}] 仿真失败 H:{h_tags}, V:{v_tags}")
            print(f"错误详情: {str(e)}")
            bw = 0

        tot_bw_list.append(bw)

    bw_mean = float(np.mean(tot_bw_list))
    bw_std = float(np.std(tot_bw_list))

    return {
        f"mixed_avg_weighted_bw_mean_{traffic_file[:-4]}": bw_mean,
        f"mixed_avg_weighted_bw_std_{traffic_file[:-4]}": bw_std,
    }


def run_simulation_with_tag_nums(h_tags: int, v_tags: int, traffic_files: List[str], traffic_weights: List[float], result_save_path: str) -> dict:
    """运行所有traffic文件并综合结果"""
    all_results = {}
    all_bw_means = []

    for traffic_file in traffic_files:
        try:
            result = run_single_traffic(traffic_file, h_tags, v_tags, os.path.join(result_save_path, f"h{h_tags}_v{v_tags}"))
            all_results.update(result)
            bw_mean = result[f"mixed_avg_weighted_bw_mean_{traffic_file[:-4]}"]
            all_bw_means.append(bw_mean)
        except Exception as e:
            print(f"处理{traffic_file}时出错: {e}")
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
            "RB_ONLY_TAG_NUM_HORIZONTAL": h_tags,
            "RB_ONLY_TAG_NUM_VERTICAL": v_tags,
        }
    )

    return all_results


def find_optimal_rb_tag_nums():
    """专门用于寻找RB_ONLY标签数量最优参数的函数"""

    # Traffic文件配置
    traffic_files = ["W_5x2.txt"]  # 可以添加更多traffic文件
    traffic_weights = [1.0]  # 对应权重

    assert len(traffic_files) == len(traffic_weights), "traffic文件数量和权重数量必须一致"
    assert abs(sum(traffic_weights) - 1.0) < 1e-6, "权重总和必须等于1"

    # 结果保存路径
    results_file_name = f"RB_TAG_NUM_optimization_{datetime.now().strftime('%m%d_%H%M')}"
    result_root_save_path = f"../Result/RB_Tag_Num_Optimization/{results_file_name}/"
    os.makedirs(result_root_save_path, exist_ok=True)
    output_csv = os.path.join(result_root_save_path, f"{results_file_name}.csv")

    # 参数范围
    h_start, h_end = RB_TAG_PARAM_RANGES["RB_ONLY_TAG_NUM_HORIZONTAL"]["start"], RB_TAG_PARAM_RANGES["RB_ONLY_TAG_NUM_HORIZONTAL"]["end"]
    v_start, v_end = RB_TAG_PARAM_RANGES["RB_ONLY_TAG_NUM_VERTICAL"]["start"], RB_TAG_PARAM_RANGES["RB_ONLY_TAG_NUM_VERTICAL"]["end"]

    def objective(trial):
        # 采样RB_ONLY标签数量参数
        h_tags = trial.suggest_int("RB_ONLY_TAG_NUM_HORIZONTAL", h_start, h_end)
        v_tags = trial.suggest_int("RB_ONLY_TAG_NUM_VERTICAL", v_start, v_end)

        results = run_simulation_with_tag_nums(h_tags, v_tags, traffic_files, traffic_weights, result_root_save_path)

        # 获取加权平均带宽
        weighted_bw = results["mixed_avg_weighted_bw_weighted_mean"]

        # 保存到trial.user_attrs，便于后期分析
        for k, v in results.items():
            trial.set_user_attr(k, v)

        return weighted_bw

    return objective, output_csv, traffic_files, traffic_weights, result_root_save_path


if __name__ == "__main__":
    print("=" * 60)
    print(f"开始RB_ONLY标签数量优化 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 获取优化函数和配置
    objective, output_csv, traffic_files, traffic_weights, result_root_save_path = find_optimal_rb_tag_nums()

    print(f"优化参数: RB_ONLY_TAG_NUM_HORIZONTAL, RB_ONLY_TAG_NUM_VERTICAL")
    print(
        f"参数范围: H:[{RB_TAG_PARAM_RANGES['RB_ONLY_TAG_NUM_HORIZONTAL']['start']}-{RB_TAG_PARAM_RANGES['RB_ONLY_TAG_NUM_HORIZONTAL']['end']}], V:[{RB_TAG_PARAM_RANGES['RB_ONLY_TAG_NUM_VERTICAL']['start']}-{RB_TAG_PARAM_RANGES['RB_ONLY_TAG_NUM_VERTICAL']['end']}]"
    )
    print(f"Traffic文件: {traffic_files}")
    print(f"权重: {traffic_weights}")
    print(f"试验次数: {N_TRIALS}")
    print(f"结果保存路径: {result_root_save_path}")
    print("=" * 60)

    # 创建Optuna研究
    study = optuna.create_study(
        study_name="CrossRing_RB_TAG_NUM_BO",
        direction="maximize",
        sampler=optuna.samplers.NSGAIISampler(),
    )

    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            n_jobs=1,  # RB标签优化通常串行执行比较稳定
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
            "value": t.values[0] if t.values else 0,
            "state": t.state.name,
        }
        rec.update(t.params)
        rec.update(t.user_attrs)
        final_records.append(rec)

    if final_records:
        final_df = pd.DataFrame(final_records)
        final_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 60)
    print("RB_ONLY标签数量优化完成!")
    if study.best_trials:
        best_trial = study.best_trials[0]
        print("最佳指标:", best_trial.values)
        print("最佳标签数量参数:", best_trial.params)

        # 显示最佳结果的详细信息
        print("\n最佳配置的详细结果:")
        for traffic_file in traffic_files:
            traffic_name = traffic_file[:-4]
            if f"mixed_avg_weighted_bw_mean_{traffic_name}" in best_trial.user_attrs:
                print(f"  {traffic_name}: {best_trial.user_attrs[f'mixed_avg_weighted_bw_mean_{traffic_name}']:.2f} GB/s")
        print(f"  加权平均: {best_trial.user_attrs.get('mixed_avg_weighted_bw_weighted_mean', 0):.2f} GB/s")
        print(f"  最小值: {best_trial.user_attrs.get('mixed_avg_weighted_bw_min', 0):.2f} GB/s")
        print(f"  方差: {best_trial.user_attrs.get('mixed_avg_weighted_bw_variance', 0):.2f}")
        print(f"  横向环标签数量: {best_trial.params.get('RB_ONLY_TAG_NUM_HORIZONTAL', 0)}")
        print(f"  纵向环标签数量: {best_trial.params.get('RB_ONLY_TAG_NUM_VERTICAL', 0)}")

    # 创建可视化
    print("\n正在生成可视化...")
    try:
        vis_dir = os.path.join(result_root_save_path, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 生成2D热力图
        create_2d_heatmap(study.trials, vis_dir)

        print(f"可视化已生成: {vis_dir}/")

    except Exception as e:
        print(f"生成可视化失败: {e}")
        traceback.print_exc()

    # 保存配置和研究对象
    config_data = {
        "optimization_target": "RB_ONLY_TAG_NUM_HORIZONTAL and RB_ONLY_TAG_NUM_VERTICAL",
        "traffic_files": traffic_files,
        "traffic_weights": traffic_weights,
        "param_ranges": RB_TAG_PARAM_RANGES,
        "n_trials": N_TRIALS,
        "n_repeats": N_REPEATS,
        "timestamp": datetime.now().isoformat(),
        "result_root_save_path": result_root_save_path,
    }

    config_file = os.path.join(result_root_save_path, "optimization_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"📁 结果文件:")
    print(f"  • CSV数据: {output_csv}")
    print(f"  • 配置文件: {config_file}")
    print(f"  • 可视化: {result_root_save_path}/visualizations/")
    print("=" * 60)
