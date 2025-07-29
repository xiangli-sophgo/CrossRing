#!/usr/bin/env python3
"""
RB_ONLY_TAG数量优化脚本
使用Optuna优化框架寻找横向环和纵向环RB_ONLY标签数量的最优组合

使用方法:
1. 默认Optuna优化模式:
   python rb_tag_num_optimization.py

2. 参数范围分析模式:
   python rb_tag_num_optimization.py range

参数范围分析模式会遍历指定的参数组合范围，系统性地测试每个组合的性能，
并生成详细的性能对比和热力图可视化。
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
import gc
import logging
import threading
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
N_TRIALS = 200  # Optuna优化试验次数
SIMULATION_TIMEOUT = 300  # 单次仿真超时时间（秒）

# RB_ONLY标签数量参数范围
RB_TAG_PARAM_RANGES = {"RB_ONLY_TAG_NUM_HORIZONTAL": {"start": -1, "end": 14}, "RB_ONLY_TAG_NUM_VERTICAL": {"start": -1, "end": 20}}

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler("rb_tag_optimization.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


# 超时异常处理
class SimulationTimeoutError(Exception):
    pass


class SimulationRunner:
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.result = None
        self.exception = None
        self.completed = False

    def run_with_timeout(self, target_func, *args, **kwargs):
        """使用线程实现跨平台超时控制"""

        def target():
            try:
                self.result = target_func(*args, **kwargs)
                self.completed = True
            except Exception as e:
                self.exception = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            # 超时了，线程仍在运行
            raise SimulationTimeoutError(f"仿真超时 ({self.timeout_seconds}秒)")

        if self.exception:
            raise self.exception

        if not self.completed:
            raise SimulationTimeoutError("仿真异常终止")

        return self.result


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


def run_parameter_range_analysis(h_range, v_range, traffic_files, traffic_weights, result_save_path):
    """
    遍历指定的参数范围，分析每个参数组合的性能表现

    Args:
        h_range: 横向环范围，可以是 (start, end) 或 [val1, val2, ...]
        v_range: 纵向环范围，可以是 (start, end) 或 [val1, val2, ...]
        traffic_files: traffic文件列表
        traffic_weights: 对应权重
        result_save_path: 结果保存路径

    Returns:
        DataFrame: 包含所有参数组合结果的数据框
    """
    logger.info("开始参数范围分析...")

    # 处理参数范围
    if isinstance(h_range, tuple) and len(h_range) == 2:
        h_values = list(range(h_range[0], h_range[1] + 1))
    else:
        h_values = list(h_range)

    if isinstance(v_range, tuple) and len(v_range) == 2:
        v_values = list(range(v_range[0], v_range[1] + 1))
    else:
        v_values = list(v_range)

    logger.info(f"横向环测试值: {h_values}")
    logger.info(f"纵向环测试值: {v_values}")
    logger.info(f"总计测试组合数: {len(h_values) * len(v_values)}")

    all_results = []
    total_combinations = len(h_values) * len(v_values)
    current_combination = 0

    for h_val in h_values:
        for v_val in v_values:
            current_combination += 1
            logger.info(f"测试组合 {current_combination}/{total_combinations}: H={h_val}, V={v_val}")

            try:
                result = run_simulation_with_tag_nums(h_val, v_val, traffic_files, traffic_weights, result_save_path)
                result["combination_index"] = current_combination
                all_results.append(result)

                # 实时输出当前结果
                weighted_bw = result.get("mixed_avg_weighted_bw_weighted_mean", 0)
                logger.info(f"组合 H={h_val}, V={v_val} 完成，加权带宽: {weighted_bw:.3f}")

            except Exception as e:
                logger.error(f"组合 H={h_val}, V={v_val} 失败: {e}")
                # 创建失败记录
                failed_result = {
                    "RB_ONLY_TAG_NUM_HORIZONTAL": h_val,
                    "RB_ONLY_TAG_NUM_VERTICAL": v_val,
                    "mixed_avg_weighted_bw_weighted_mean": 0,
                    "combination_index": current_combination,
                    "error": str(e),
                }
                for traffic_file in traffic_files:
                    traffic_name = traffic_file[:-4]
                    failed_result[f"mixed_avg_weighted_bw_mean_{traffic_name}"] = 0
                    failed_result[f"mixed_avg_weighted_bw_std_{traffic_name}"] = 0
                all_results.append(failed_result)

    # 转换为DataFrame
    results_df = pd.DataFrame(all_results)

    # 保存结果
    csv_path = os.path.join(result_save_path, f"parameter_range_analysis_{datetime.now().strftime('%m%d_%H%M')}.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"参数范围分析结果已保存: {csv_path}")

    # 输出分析摘要
    print_range_analysis_summary(results_df, h_values, v_values, traffic_files)

    return results_df


def print_range_analysis_summary(results_df, h_values, v_values, traffic_files):
    """打印参数范围分析摘要"""
    print("\n" + "=" * 80)
    print("参数范围分析摘要")
    print("=" * 80)

    print(f"横向环测试范围: {h_values}")
    print(f"纵向环测试范围: {v_values}")
    print(f"总测试组合数: {len(results_df)}")

    # 找出最优配置
    if not results_df.empty and "mixed_avg_weighted_bw_weighted_mean" in results_df.columns:
        best_idx = results_df["mixed_avg_weighted_bw_weighted_mean"].idxmax()
        best_result = results_df.loc[best_idx]

        print(f"\n最优配置:")
        print(f"  横向环: {best_result['RB_ONLY_TAG_NUM_HORIZONTAL']}")
        print(f"  纵向环: {best_result['RB_ONLY_TAG_NUM_VERTICAL']}")
        print(f"  加权平均带宽: {best_result['mixed_avg_weighted_bw_weighted_mean']:.3f}")

        print(f"\n各traffic性能:")
        for traffic_file in traffic_files:
            traffic_name = traffic_file[:-4]
            bw_key = f"mixed_avg_weighted_bw_mean_{traffic_name}"
            if bw_key in best_result:
                print(f"  {traffic_name}: {best_result[bw_key]:.3f}")

        # 性能分布统计
        print(f"\n性能分布统计:")
        bw_col = "mixed_avg_weighted_bw_weighted_mean"
        print(f"  加权带宽范围: {results_df[bw_col].min():.3f} - {results_df[bw_col].max():.3f}")
        print(f"  加权带宽平均: {results_df[bw_col].mean():.3f}")
        print(f"  加权带宽标准差: {results_df[bw_col].std():.3f}")

        # Top 5 配置
        print(f"\nTop 5 配置:")
        top5 = results_df.nlargest(5, bw_col)
        for i, (idx, row) in enumerate(top5.iterrows(), 1):
            print(f"  {i}. H={row['RB_ONLY_TAG_NUM_HORIZONTAL']}, V={row['RB_ONLY_TAG_NUM_VERTICAL']}, BW={row[bw_col]:.3f}")

    print("=" * 80)


def create_enhanced_heatmap(results_df, save_dir):
    """创建增强的热力图，支持参数范围分析结果"""
    if results_df.empty:
        print("无数据可用于生成热力图")
        return

    # 提取数据
    h_values = results_df["RB_ONLY_TAG_NUM_HORIZONTAL"].values
    v_values = results_df["RB_ONLY_TAG_NUM_VERTICAL"].values
    bw_values = results_df["mixed_avg_weighted_bw_weighted_mean"].values

    # 创建热力图数据
    h_unique = sorted(results_df["RB_ONLY_TAG_NUM_HORIZONTAL"].unique())
    v_unique = sorted(results_df["RB_ONLY_TAG_NUM_VERTICAL"].unique())

    # 创建网格数据
    heatmap_data = np.full((len(v_unique), len(h_unique)), np.nan)

    for _, row in results_df.iterrows():
        h_idx = h_unique.index(row["RB_ONLY_TAG_NUM_HORIZONTAL"])
        v_idx = v_unique.index(row["RB_ONLY_TAG_NUM_VERTICAL"])
        heatmap_data[v_idx, h_idx] = row["mixed_avg_weighted_bw_weighted_mean"]

    # 创建图表
    plt.figure(figsize=(14, 10))

    # 绘制热力图
    im = plt.imshow(heatmap_data, cmap="YlGnBu", aspect="auto", origin="lower")

    # 设置坐标轴
    plt.xticks(range(len(h_unique)), h_unique)
    plt.yticks(range(len(v_unique)), v_unique)
    plt.xlabel("横向环 Bubble Slot数量", fontsize=14)
    plt.ylabel("纵向环 Bubble Slot数量", fontsize=14)
    plt.title("参数范围分析热力图 (加权平均带宽)", fontsize=16, fontweight="bold")

    # 添加颜色条
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label("加权平均带宽 (GB/s)", fontsize=12)

    # 在网格上标注数值（只在高性能区域）
    max_bw = np.nanmax(heatmap_data)
    threshold = max_bw * 0.85  # 只在85%以上的区域显示数值

    for i in range(len(v_unique)):
        for j in range(len(h_unique)):
            if not np.isnan(heatmap_data[i, j]) and heatmap_data[i, j] >= threshold:
                text_color = "white" if heatmap_data[i, j] < max_bw * 0.95 else "black"
                plt.text(j, i, f"{heatmap_data[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    # 标记最优点
    if not results_df.empty:
        best_idx = results_df["mixed_avg_weighted_bw_weighted_mean"].idxmax()
        best_row = results_df.loc[best_idx]
        best_h = best_row["RB_ONLY_TAG_NUM_HORIZONTAL"]
        best_v = best_row["RB_ONLY_TAG_NUM_VERTICAL"]
        best_bw = best_row["mixed_avg_weighted_bw_weighted_mean"]

        h_pos = h_unique.index(best_h)
        v_pos = v_unique.index(best_v)

        plt.scatter(h_pos, v_pos, c="red", s=300, marker="*", edgecolors="white", linewidth=2, label=f"最优点({best_h},{best_v}): {best_bw:.3f}")
        plt.legend(loc="upper right")

    plt.tight_layout()

    # 保存图片
    heatmap_path = os.path.join(save_dir, f"parameter_range_heatmap_{datetime.now().strftime('%m%d_%H%M')}.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"增强热力图已保存: {heatmap_path}")

    return heatmap_path


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

    plt.xlabel("横向环Bubble slot数量", fontsize=14)
    plt.ylabel("纵向环Bubble slot数量", fontsize=14)
    plt.title("Bubble slot优化热力图", fontsize=16)

    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label("带宽 (GB/s)", fontsize=12)

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
        logger.info(f"开始仿真: {traffic_file}, H:{h_tags}, V:{v_tags}, 重复:{rpt+1}/{N_REPEATS}")

        sim = None
        try:
            # 验证文件路径
            config_path = os.path.abspath("../config/config2.json")
            traffic_path = os.path.abspath("../traffic/0617")

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            if not os.path.exists(traffic_path):
                raise FileNotFoundError(f"流量文件目录不存在: {traffic_path}")

            config = CrossRingConfig(config_path)
            config.TOPO_TYPE = "5x2"

            # 设置要测试的标签数量
            config.RB_ONLY_TAG_NUM_HORIZONTAL = h_tags
            config.RB_ONLY_TAG_NUM_VERTICAL = v_tags

            # 创建仿真实例
            sim = BaseModel(
                model_type="REQ_RSP",
                config=config,
                topo_type="5x2",
                traffic_file_path=traffic_path,
                traffic_config=traffic_file,
                result_save_path=result_save_path,
                verbose=0,  # 关闭详细输出
                print_trace=0,
                plot_link_state=0,
                plot_flow_fig=0,
                plot_RN_BW_fig=0,
            )

            # 使用跨平台超时控制
            def run_simulation():
                sim.initial()
                sim.end_time = 6000
                sim.print_interval = 2000
                sim.run()
                return sim.get_results().get("mixed_avg_weighted_bw", 0)

            runner = SimulationRunner(SIMULATION_TIMEOUT)
            bw = runner.run_with_timeout(run_simulation)
            logger.info(f"仿真完成: {traffic_file}, H:{h_tags}, V:{v_tags}, BW:{bw:.3f}")

        except SimulationTimeoutError as e:
            logger.error(f"仿真超时: {traffic_file}, H:{h_tags}, V:{v_tags}, 重复:{rpt+1} - {str(e)}")
            bw = 0
        except Exception as e:
            logger.error(f"仿真失败: {traffic_file}, H:{h_tags}, V:{v_tags}, 重复:{rpt+1}")
            logger.error(f"错误详情: {str(e)}")
            logger.error(f"堆栈跟踪: {traceback.format_exc()}")
            bw = 0
        finally:
            # 强制垃圾回收
            if sim is not None:
                del sim
            gc.collect()

        tot_bw_list.append(bw)

    bw_mean = float(np.mean(tot_bw_list))
    bw_std = float(np.std(tot_bw_list))

    logger.info(f"Traffic {traffic_file} 完成: 平均BW={bw_mean:.3f}, 标准差={bw_std:.3f}")

    return {
        f"mixed_avg_weighted_bw_mean_{traffic_file[:-4]}": bw_mean,
        f"mixed_avg_weighted_bw_std_{traffic_file[:-4]}": bw_std,
    }


def run_simulation_with_tag_nums(h_tags: int, v_tags: int, traffic_files: List[str], traffic_weights: List[float], result_save_path: str) -> dict:
    """运行所有traffic文件并综合结果"""
    logger.info(f"开始参数组合测试: H={h_tags}, V={v_tags}")

    all_results = {}
    all_bw_means = []

    for traffic_file in traffic_files:
        try:
            result = run_single_traffic(traffic_file, h_tags, v_tags, os.path.join(result_save_path, f"h{h_tags}_v{v_tags}"))
            all_results.update(result)
            bw_mean = result[f"mixed_avg_weighted_bw_mean_{traffic_file[:-4]}"]
            all_bw_means.append(bw_mean)
        except Exception as e:
            logger.error(f"处理{traffic_file}时出错: {e}")
            logger.error(f"堆栈跟踪: {traceback.format_exc()}")
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

    logger.info(f"参数组合完成: H={h_tags}, V={v_tags}, 加权平均BW={weighted_bw_mean:.3f}")
    return all_results


def find_optimal_rb_tag_nums():
    """专门用于寻找RB_ONLY标签数量最优参数的函数"""

    # Traffic文件配置
    traffic_files = [
        "R_5x2.txt",
        "W_5x2.txt",
    ]
    traffic_weights = [0.5, 0.5]  # 对应权重

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


def run_range_analysis_mode():
    """运行参数范围分析模式"""
    logger.info("=" * 60)
    logger.info("参数范围分析模式")
    logger.info("=" * 60)

    # 配置参数范围
    # 可以根据需要修改这些范围
    h_range = (0, 14)  # 横向环范围 1-10
    v_range = (0, 30)  # 纵向环范围 1-15

    # 也可以使用列表形式指定特定值
    # h_range = [1, 3, 5, 7, 9]
    # v_range = [2, 5, 8, 12, 15]

    traffic_files = ["R_5x2.txt", "W_5x2.txt"]
    traffic_weights = [0.5, 0.5]

    # 结果保存路径
    results_file_name = f"RB_TAG_NUM_range_analysis_{datetime.now().strftime('%m%d_%H%M')}"
    result_root_save_path = f"../Result/RB_Tag_Num_Optimization/{results_file_name}/"
    os.makedirs(result_root_save_path, exist_ok=True)

    logger.info(f"横向环范围: {h_range}")
    logger.info(f"纵向环范围: {v_range}")
    logger.info(f"Traffic文件: {traffic_files}")
    logger.info(f"权重: {traffic_weights}")
    logger.info(f"结果保存路径: {result_root_save_path}")

    # 运行参数范围分析
    results_df = run_parameter_range_analysis(h_range, v_range, traffic_files, traffic_weights, result_root_save_path)

    # 生成增强热力图
    if not results_df.empty:
        vis_dir = os.path.join(result_root_save_path, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        create_enhanced_heatmap(results_df, vis_dir)

    return results_df, result_root_save_path


if __name__ == "__main__":
    # 选择运行模式
    import sys

    # mode = "optuna"  # 默认使用optuna优化
    mode = "range"
    # if len(sys.argv) > 1:
    # mode = sys.argv[1].lower()

    if mode == "range":
        # 参数范围分析模式
        try:
            logger.info("启动参数范围分析模式...")
            results_df, result_save_path = run_range_analysis_mode()
            logger.info("参数范围分析完成!")
        except Exception as e:
            logger.error(f"参数范围分析失败: {e}")
            logger.error(f"堆栈跟踪: {traceback.format_exc()}")
            sys.exit(1)
    else:
        # 默认的Optuna优化模式
        try:
            logger.info("=" * 60)
            logger.info(f"开始RB_ONLY标签数量优化 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 60)

            # 获取优化函数和配置
            objective, output_csv, traffic_files, traffic_weights, result_root_save_path = find_optimal_rb_tag_nums()

            logger.info(f"优化参数: RB_ONLY_TAG_NUM_HORIZONTAL, RB_ONLY_TAG_NUM_VERTICAL")
            logger.info(
                f"参数范围: H:[{RB_TAG_PARAM_RANGES['RB_ONLY_TAG_NUM_HORIZONTAL']['start']}-{RB_TAG_PARAM_RANGES['RB_ONLY_TAG_NUM_HORIZONTAL']['end']}], V:[{RB_TAG_PARAM_RANGES['RB_ONLY_TAG_NUM_VERTICAL']['start']}-{RB_TAG_PARAM_RANGES['RB_ONLY_TAG_NUM_VERTICAL']['end']}]"
            )
            logger.info(f"Traffic文件: {traffic_files}")
            logger.info(f"权重: {traffic_weights}")
            logger.info(f"试验次数: {N_TRIALS}")
            logger.info(f"仿真超时: {SIMULATION_TIMEOUT}秒")
            logger.info(f"结果保存路径: {result_root_save_path}")
            logger.info("=" * 60)

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
                logger.warning("优化被用户中断")
            except Exception as e:
                logger.error(f"优化过程中发生错误: {e}")
                logger.error(f"堆栈跟踪: {traceback.format_exc()}")
        except Exception as e:
            logger.error(f"程序启动失败: {e}")
            logger.error(f"堆栈跟踪: {traceback.format_exc()}")
            sys.exit(1)

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
