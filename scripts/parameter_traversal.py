#!/usr/bin/env python3
"""
CrossRing 参数优化工具 - 高效版
支持多参数并行遍历，自动生成性能曲线分析

使用方法:
1. 修改main()函数中的参数配置区域
2. 运行: python parameter_traversal.py

参数配置:
- param_configs: 参数列表，每个包含name和range
- range格式:
  * 连续: "0,10" → [0,1,2,...,10]
  * 步长: "0,20,5" → [0,5,10,15,20]
  * 离散: "[1,3,5,7]" → [1,3,5,7]

性能特性:
- 并行处理: max_workers控制并行进程数
- YAML配置: 自动加载拓扑配置文件
- 智能可视化: 每参数独立曲线，显示最优组合
- 无冗余: 单次运行，去除重复统计
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import time
import argparse
import json
import pandas as pd
import gc
from datetime import datetime
from typing import List, Tuple, Dict, Any, Union
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import CrossRingConfig
from src.core.base_model import BaseModel

# 移除日志配置


# 设置matplotlib中文字体支持
def setup_font():
    import matplotlib.font_manager as fm
    
    # 尝试查找系统中的中文字体
    font_paths = []
    for font_path in fm.fontManager.ttflist:
        if any(name in font_path.name for name in ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']):
            font_paths.append(font_path.name)
    
    if font_paths:
        plt.rcParams["font.sans-serif"] = font_paths + ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"使用中文字体: {font_paths[0]}")
    else:
        # 如果没有中文字体，使用英文标签
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False
        print("未找到中文字体，将使用英文标签")
        return False
    return True

# 检查字体设置结果
chinese_font_available = setup_font()


def parse_range(range_str: str) -> List[Union[int, float]]:
    """
    解析范围字符串
    支持格式:
    - "start,end": 生成[start, start+1, ..., end]
    - "start,end,step": 生成[start, start+step, ..., end]
    - "[val1,val2,val3]": 直接使用提供的值列表
    """
    range_str = range_str.strip()

    # 检查是否是列表格式
    if range_str.startswith("[") and range_str.endswith("]"):
        # 解析列表
        values_str = range_str[1:-1]
        values = []
        for val in values_str.split(","):
            val = val.strip()
            try:
                # 尝试解析为整数
                values.append(int(val))
            except ValueError:
                try:
                    # 尝试解析为浮点数
                    values.append(float(val))
                except ValueError:
                    raise ValueError(f"无法解析值: {val}")
        return values
    else:
        # 解析范围格式
        parts = range_str.split(",")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            return list(range(start, end + 1))
        elif len(parts) == 3:
            start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
            return list(range(start, end + 1, step))
        else:
            raise ValueError(f"无效的范围格式: {range_str}")


def run_single_simulation(config_params: Dict[str, Any], traffic_file: str, base_config_path: str, topo_type, traffic_path: str, result_save_path: str) -> Dict[str, Any]:
    """运行单次仿真"""
    sim = None
    try:
        # 使用YAML配置文件加载配置
        config_file = f"../config/topologies/topo_{topo_type}.yaml"
        config = CrossRingConfig(config_file)

        # 更新参数
        for param_name, param_value in config_params.items():
            # 特殊处理 IN_FIFO_DEPTH 参数：同时设置 RB_IN_FIFO_DEPTH 和 EQ_IN_FIFO_DEPTH，并按比例调整相关参数
            if param_name == "IN_FIFO_DEPTH":
                if isinstance(param_value, (int, float)) and param_value > 0:
                    # 设置两个FIFO深度参数为相同值
                    config.RB_IN_FIFO_DEPTH = int(param_value)
                    config.EQ_IN_FIFO_DEPTH = int(param_value)

                    # 计算比例乘数（基准值为16）
                    ratio_multiplier = param_value / 16.0

                    # 按比例调整相关参数，确保最小值为1，并且T1比对应的T2至少大1
                    config.TL_Etag_T2_UE_MAX = max(1, int(8 * ratio_multiplier))
                    config.TL_Etag_T1_UE_MAX = max(config.TL_Etag_T2_UE_MAX + 1, int(15 * ratio_multiplier))
                    config.TR_Etag_T2_UE_MAX = max(1, int(12 * ratio_multiplier))
                    config.TU_Etag_T2_UE_MAX = max(1, int(8 * ratio_multiplier))
                    config.TU_Etag_T1_UE_MAX = max(config.TU_Etag_T2_UE_MAX + 1, int(15 * ratio_multiplier))
                    config.TD_Etag_T2_UE_MAX = max(1, int(12 * ratio_multiplier))

                    # IN_FIFO_DEPTH参数调整完成
                else:
                    print(f"IN_FIFO_DEPTH 参数值无效: {param_value}")
            elif hasattr(config, param_name):
                # 普通参数的处理
                if isinstance(param_value, (int, float)):
                    setattr(config, param_name, param_value)
                else:
                    print(f"参数 {param_name} 的值 {param_value} 不是数值类型")
            else:
                print(f"配置中不存在参数: {param_name}")

        # 创建仿真实例
        sim = BaseModel(
            model_type="REQ_RSP",
            config=config,
            topo_type=config.TOPO_TYPE,
            traffic_file_path=traffic_path,
            traffic_config=traffic_file,
            result_save_path=result_save_path,
            verbose=1,
            print_trace=0,
            plot_link_state=0,
            plot_flow_fig=0,
            plot_RN_BW_fig=0,
        )

        # 运行仿真
        sim.initial()
        sim.end_time = 6000
        sim.print_interval = 2000
        sim.run()
        return sim.get_results()  # 返回完整的结果字典

    except Exception as e:
        print(f"仿真失败: {e}")
        return {}  # 返回空字典而不是0
    finally:
        if sim is not None:
            del sim
        gc.collect()


def run_parameter_combination(
    params_dict: Dict[str, Any],
    traffic_files: List[str],
    traffic_weights: List[float],
    base_config_path: str,
    topo_type,
    traffic_path: str,
    result_save_path: str,
) -> Dict[str, Any]:
    """运行一个参数组合的所有仿真"""
    # 开始测试参数组合

    results = params_dict.copy()
    traffic_bw_list = []

    for traffic_file in traffic_files:
        traffic_name = traffic_file[:-4]

        # 单次仿真
        sim_results = run_single_simulation(params_dict, traffic_file, base_config_path, topo_type, traffic_path, result_save_path)

        if isinstance(sim_results, dict) and sim_results:
            bw = sim_results.get("mixed_avg_weighted_bw", 0)

            # 单个traffic文件时不添加后缀
            if len(traffic_files) == 1:
                results["bw"] = bw
                traffic_bw_list.append(bw)

                # 保存其他仿真指标（不添加后缀）
                for key, value in sim_results.items():
                    if key != "mixed_avg_weighted_bw" and isinstance(value, (int, float)):
                        results[key] = value
            else:
                # 多个traffic文件时保留原逻辑
                results[f"bw_{traffic_name}"] = bw
                traffic_bw_list.append(bw)

                # 保存其他仿真指标（添加后缀）
                for key, value in sim_results.items():
                    if key != "mixed_avg_weighted_bw" and isinstance(value, (int, float)):
                        results[f"{key}_{traffic_name}"] = value
        else:
            traffic_bw_list.append(0)
            if len(traffic_files) == 1:
                results["bw"] = 0
            else:
                results[f"bw_{traffic_name}"] = 0

    # 计算综合指标
    weighted_bw = sum(bw * weight for bw, weight in zip(traffic_bw_list, traffic_weights))
    min_bw = min(traffic_bw_list) if traffic_bw_list else 0

    results.update(
        {
            "weighted_bw": weighted_bw,
            "min_bw": min_bw,
        }
    )

    return results


def run_single_combination(args):
    """用于并行执行的包装函数"""
    params_dict, traffic_files, traffic_weights, config_path, topo_type, traffic_path, output_dir = args
    return run_parameter_combination(params_dict, traffic_files, traffic_weights, config_path, topo_type, traffic_path, output_dir)


def save_plot(save_dir: str, filename: str) -> str:
    """通用图表保存函数"""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plot_path = os.path.join(save_dir, f"{filename}_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    return plot_path


def add_stats_box(ax, data, x_pos=0.02, y_pos=0.98):
    """添加统计信息框"""
    stats_text = f"""性能统计:
最大值: {data.max():.3f} GB/s
平均值: {data.mean():.3f} GB/s
最小值: {data.min():.3f} GB/s"""
    ax.text(
        x_pos,
        y_pos,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.9, edgecolor="navy"),
        fontsize=11,
        fontweight="bold",
    )


def create_visualizations(results_df: pd.DataFrame, param_names: List[str], save_dir: str):
    """创建可视化图表"""
    os.makedirs(save_dir, exist_ok=True)

    if results_df.empty:
        print("结果数据为空，无法生成可视化图表")
        return

    try:
        if len(param_names) == 1:
            create_1d_plot(results_df, param_names[0], save_dir)
        else:
            create_param_curves(results_df, param_names, save_dir)
    except Exception as e:
        print(f"生成可视化图表时出错: {e}")


def create_1d_plot(df: pd.DataFrame, param_name: str, save_dir: str):
    """创建单参数折线图，参考RB_slot_curve_plot.py风格"""
    setup_font()
    sns.set_style("whitegrid")

    plt.figure(figsize=(14, 8))

    # 按参数值排序并计算统计数据
    param_grouped = df.groupby(param_name)["weighted_bw"].agg(["mean", "std", "max", "min", "count"])

    # 绘制主曲线 - 使用与RB_slot_curve_plot.py相同的风格
    mean_label = "平均性能" if chinese_font_available else "Average Performance"
    plt.plot(param_grouped.index, param_grouped["mean"], marker="o", linewidth=3, markersize=8, color="#2E86AB", label=mean_label)

    # 如果有多个数据点，添加标准差区域
    if param_grouped["count"].max() > 1:
        std_label = "±1标准差" if chinese_font_available else "±1 Std Dev"
        plt.fill_between(param_grouped.index, param_grouped["mean"] - param_grouped["std"], param_grouped["mean"] + param_grouped["std"], alpha=0.3, color="#2E86AB", label=std_label)

    # 绘制最大值和最小值曲线
    if not param_grouped["max"].equals(param_grouped["mean"]):
        max_label = "最大值" if chinese_font_available else "Maximum"
        plt.plot(param_grouped.index, param_grouped["max"], "g--", linewidth=2, marker="s", markersize=6, label=max_label)
    if not param_grouped["min"].equals(param_grouped["mean"]):
        min_label = "最小值" if chinese_font_available else "Minimum"
        plt.plot(param_grouped.index, param_grouped["min"], "r--", linewidth=2, marker="^", markersize=6, label=min_label)

    # 设置标签和标题 - 与RB_slot_curve_plot.py风格一致
    plt.xlabel(f"{param_name}", fontsize=14, fontweight="bold")
    ylabel = "加权带宽 (GB/s)" if chinese_font_available else "Weighted Bandwidth (GB/s)"
    title = f"{param_name} 性能关系曲线" if chinese_font_available else f"{param_name} Performance Curve"
    plt.ylabel(ylabel, fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.7)
    plt.legend(fontsize=12)

    # 标注最优点
    if not param_grouped.empty and len(param_grouped) > 0:
        best_param_val = param_grouped["mean"].idxmax()  # idxmax直接返回索引值
        best_performance = param_grouped.loc[best_param_val, "mean"]

        plt.scatter(best_param_val, best_performance, c="red", s=300, marker="*", zorder=10, edgecolors="white", linewidth=2)

        # 添加最优点标注
        if chinese_font_available:
            annotation_text = f"最优配置\n{param_name}={best_param_val}\n性能: {best_performance:.3f} GB/s"
        else:
            annotation_text = f"Optimal Config\n{param_name}={best_param_val}\nPerf: {best_performance:.3f} GB/s"
        plt.annotate(
            annotation_text,
            xy=(best_param_val, best_performance),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8, edgecolor="red"),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
            fontsize=12,
            fontweight="bold",
            color="red",
        )

    # 添加性能统计信息
    if chinese_font_available:
        stats_text = f"""性能统计:
最大值: {param_grouped['max'].max():.3f} GB/s
平均值: {param_grouped['mean'].mean():.3f} GB/s
最小值: {param_grouped['min'].min():.3f} GB/s
测试范围: {param_grouped.index.min()} - {param_grouped.index.max()}"""
    else:
        stats_text = f"""Performance Stats:
Max: {param_grouped['max'].max():.3f} GB/s
Mean: {param_grouped['mean'].mean():.3f} GB/s
Min: {param_grouped['min'].min():.3f} GB/s
Range: {param_grouped.index.min()} - {param_grouped.index.max()}"""

    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.9, edgecolor="navy"),
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()

    plot_path = save_plot(save_dir, f"{param_name}_curve")
    return plot_path


def create_param_curves(df: pd.DataFrame, param_names: List[str], save_dir: str):
    """为每个参数创建性能曲线，曲线值为该参数在其他参数组合下的最大带宽"""
    setup_font()
    sns.set_style("whitegrid")

    # 计算每个参数的最大性能曲线
    param_data = {}
    colors = ["#2E86AB", "#F18F01", "#A23B72", "#F18701", "#C73E1D"]

    for i, param in enumerate(param_names):
        param_stats = []
        param_values = sorted(df[param].unique())

        for p_val in param_values:
            # 找出该参数值下的所有数据
            p_data = df[df[param] == p_val]
            if len(p_data) > 0:
                max_bw = p_data["weighted_bw"].max()
                best_config = p_data.loc[p_data["weighted_bw"].idxmax()]

                # 记录最佳配对参数
                other_params = {}
                for other_param in param_names:
                    if other_param != param:
                        other_params[f"best_{other_param}"] = best_config[other_param]

                param_stats.append({param: p_val, "max_weighted_bw": max_bw, **other_params})

        param_data[param] = pd.DataFrame(param_stats)

    # 为每个参数创建独立的图表
    curve_paths = []

    for i, param in enumerate(param_names):
        data = param_data[param]
        color = colors[i % len(colors)]

        if not data.empty:
            # 创建单独的图表
            plt.figure(figsize=(12, 8))

            # 绘制主曲线
            label_text = "最大性能" if chinese_font_available else "Max Performance"
            plt.plot(data[param], data["max_weighted_bw"], marker="o", linewidth=3, markersize=8, color=color, label=label_text)

            # 标注最优点
            best_idx = data["max_weighted_bw"].idxmax()
            best_row = data.loc[best_idx]
            plt.scatter(best_row[param], best_row["max_weighted_bw"], c="red", s=300, marker="*", zorder=10, edgecolors="white", linewidth=2)

            # 添加最优点标注
            optimal_text = f'最优: {best_row[param]}\n{best_row["max_weighted_bw"]:.3f} GB/s' if chinese_font_available else f'Optimal: {best_row[param]}\n{best_row["max_weighted_bw"]:.3f} GB/s'
            plt.annotate(
                optimal_text,
                xy=(best_row[param], best_row["max_weighted_bw"]),
                xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8, edgecolor="red"),
                arrowprops=dict(arrowstyle="->", color="red", lw=2),
                fontsize=12,
                fontweight="bold",
                color="red",
            )

            # 设置标签和样式
            plt.xlabel(f"{param}", fontsize=14, fontweight="bold")
            ylabel = "加权带宽 (GB/s)" if chinese_font_available else "Weighted Bandwidth (GB/s)"
            title = f"{param} 性能关系曲线" if chinese_font_available else f"{param} Performance Curve"
            plt.ylabel(ylabel, fontsize=14, fontweight="bold")
            plt.title(title, fontsize=16, fontweight="bold")
            plt.grid(True, alpha=0.7)
            plt.legend(fontsize=12)

            # 添加统计信息
            if chinese_font_available:
                stats_text = f"""性能统计:
最大值: {data["max_weighted_bw"].max():.3f} GB/s
平均值: {data["max_weighted_bw"].mean():.3f} GB/s
最小值: {data["max_weighted_bw"].min():.3f} GB/s
测试范围: {data[param].min()} - {data[param].max()}"""
            else:
                stats_text = f"""Performance Stats:
Max: {data["max_weighted_bw"].max():.3f} GB/s
Mean: {data["max_weighted_bw"].mean():.3f} GB/s
Min: {data["max_weighted_bw"].min():.3f} GB/s
Range: {data[param].min()} - {data[param].max()}"""

            plt.text(
                0.02,
                0.98,
                stats_text,
                transform=plt.gca().transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.9, edgecolor="navy"),
                fontsize=11,
                fontweight="bold",
            )

            plt.tight_layout()

            # 保存图片
            curve_path = save_plot(save_dir, f"{param}_curve")
            curve_paths.append(curve_path)

    # 简化的分析摘要
    for param in param_names:
        data = param_data[param]
        if not data.empty:
            best_config = data.loc[data["max_weighted_bw"].idxmax()]
            if chinese_font_available:
                print(f"{param}最优值: {best_config[param]} (性能: {best_config['max_weighted_bw']:.3f})")
            else:
                print(f"{param} optimal: {best_config[param]} (perf: {best_config['max_weighted_bw']:.3f})")

    return curve_paths


def main():
    """
    主函数 - 在这里设置要遍历的参数
    """
    # ==================== 参数配置区域 ====================

    # 参数设置：可以配置1-3个参数
    param_configs = [
        # 示例：使用 IN_FIFO_DEPTH 同时遍历 RB_IN_FIFO_DEPTH 和 EQ_IN_FIFO_DEPTH，并按比例调整相关参数
        # {"name": "IN_FIFO_DEPTH", "range": "8,32,4"},  # 从8到32，步长为4
        {"name": "IN_FIFO_DEPTH", "range": "[20, 18, 16, 14, 12, 10, 8, 6, 4, 2]"},
        # {"name": "IN_FIFO_DEPTH", "range": "[8, 16]"},
        # {"name": "SLICE_PER_LINK", "range": "5, 20"},
        # {"name": "SLICE_PER_LINK", "range": "[20]"},
        # 其他参数配置示例：
        # {"name": "IQ_OUT_FIFO_DEPTH_VERTICAL", "range": "1,8"},
        # {"name": "SLICE_PER_LINK", "range": "17,20"},
    ]

    # 文件路径配置
    config_path = "../config/topologies/topo_5x4.yaml"
    # traffic_path = "../traffic/traffic0730"
    traffic_path = "../traffic/0617"
    # 仿真配置
    traffic_files = [
        # "W_4x4.txt",
        # "W_8x8.txt",
        # "W_12x12.txt",
        # "W_5x4_CR_v1.0.2.txt",
        "LLama2_AllReduce.txt",
    ]
    # topo_type = "4x4"
    # topo_type = "8x8"
    # topo_type = "12x12"
    topo_type = "5x4"
    traffic_weights = [1]
    max_workers = 4  # 并行运行的最大进程数，根据CPU核心数调整

    # 输出配置
    custom_output_name = None  # 设置为None则自动生成，或指定名称如 "my_test"

    # ==================== 参数配置区域结束 ====================

    # 解析参数配置
    param_names = []
    param_ranges = []

    for param_config in param_configs:
        if param_config["name"] and param_config["range"]:
            param_names.append(param_config["name"])
            param_ranges.append(parse_range(param_config["range"]))

    if not param_names:
        print("错误：至少需要配置一个参数")
        return

    # 验证traffic配置
    if len(traffic_files) != len(traffic_weights):
        print("错误：Traffic文件数量和权重数量不匹配")
        return

    if abs(sum(traffic_weights) - 1.0) > 1e-6:
        print("错误：权重总和必须为1")
        return

    # 设置输出目录
    if custom_output_name:
        output_dir = f"../Result/ParameterTraversal/{custom_output_name}"
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        params_str = "_".join(param_names)
        output_dir = f"../Result/ParameterTraversal/{params_str}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # 记录配置
    config_info = {
        "parameters": param_names,
        "ranges": {name: range_vals for name, range_vals in zip(param_names, param_ranges)},
        "traffic_files": traffic_files,
        "traffic_weights": traffic_weights,
        "topo_type": topo_type,
        "max_workers": max_workers,
        "config_path": config_path,
        "traffic_path": traffic_path,
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("参数遍历开始")
    print("=" * 60)
    print(f"参数: {param_names}")
    print(f"总组合数: {np.prod([len(r) for r in param_ranges])}")
    print(f"并行进程数: {max_workers}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 生成所有参数组合
    all_combinations = list(product(*param_ranges))
    total_combinations = len(all_combinations)

    # 准备并行执行的参数
    task_args = []
    for combination in all_combinations:
        params_dict = {name: value for name, value in zip(param_names, combination)}
        task_args.append((params_dict, traffic_files, traffic_weights, config_path, topo_type, traffic_path, output_dir))

    # 并行运行所有组合
    completed_count = 0
    csv_path = os.path.join(output_dir, "results.csv")

    if max_workers > 1:
        print(f"使用 {max_workers} 个进程并行运行...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_params = {executor.submit(run_single_combination, args): args[0] for args in task_args}

            # 处理完成的任务
            for future in as_completed(future_to_params):
                params_dict = future_to_params[future]
                try:
                    result = future.result()
                    completed_count += 1
                    print(f"完成: {completed_count}/{total_combinations}")

                    # 使用追加模式保存结果
                    df_single = pd.DataFrame([result])
                    if completed_count == 1:
                        # 第一次写入，包含表头
                        df_single.to_csv(csv_path, index=False, mode='w', encoding='utf-8-sig')
                    else:
                        # 后续追加，不包含表头
                        df_single.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')

                except Exception as e:
                    print(f"参数组合 {params_dict} 运行失败: {e}")
                    # 添加失败的结果
                    failed_result = params_dict.copy()
                    failed_result.update({"weighted_bw": 0, "min_bw": 0, "error": str(e)})

                    completed_count += 1
                    # 同样追加失败的结果
                    df_single = pd.DataFrame([failed_result])
                    if completed_count == 1:
                        df_single.to_csv(csv_path, index=False, mode='w', encoding='utf-8-sig')
                    else:
                        df_single.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')
    else:
        print("使用单进程串行运行...")
        for i, args in enumerate(task_args):
            params_dict = args[0]
            print(f"进度: {i+1}/{total_combinations}")

            try:
                result = run_single_combination(args)
                completed_count = i + 1

                # 使用追加模式保存结果
                df_single = pd.DataFrame([result])
                if completed_count == 1:
                    # 第一次写入，包含表头
                    df_single.to_csv(csv_path, index=False, mode='w', encoding='utf-8-sig')
                else:
                    # 后续追加，不包含表头
                    df_single.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')

            except Exception as e:
                print(f"参数组合 {params_dict} 运行失败: {e}")
                failed_result = params_dict.copy()
                failed_result.update({"weighted_bw": 0, "min_bw": 0, "error": str(e)})

                completed_count = i + 1
                # 同样追加失败的结果
                df_single = pd.DataFrame([failed_result])
                if completed_count == 1:
                    df_single.to_csv(csv_path, index=False, mode='w', encoding='utf-8-sig')
                else:
                    df_single.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')

    # 最终结果分析 - 从CSV文件读取所有结果
    print(f"从CSV文件读取结果: {csv_path}")
    try:
        results_df = pd.read_csv(csv_path)
        print(f"成功读取 {len(results_df)} 条结果")
    except Exception as e:
        print(f"读取结果文件失败: {e}")
        results_df = pd.DataFrame()  # 创建空DataFrame

    # 找出最优配置
    if not results_df.empty and "weighted_bw" in results_df.columns:
        best_idx = results_df["weighted_bw"].idxmax()
        best_config = results_df.loc[best_idx]

        print("=" * 60)
        if chinese_font_available:
            print("最优配置:")
            for param in param_names:
                print(f"  {param}: {best_config[param]}")
            print(f"  加权带宽: {best_config['weighted_bw']:.3f}")
        else:
            print("Optimal Configuration:")
            for param in param_names:
                print(f"  {param}: {best_config[param]}")
            print(f"  Weighted BW: {best_config['weighted_bw']:.3f}")

        # 创建可视化
        create_visualizations(results_df, param_names, os.path.join(output_dir, "visualizations"))

        # 生成报告
        report_path = os.path.join(output_dir, "report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("参数遍历报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"参数: {param_names}\n")
            f.write(f"总测试组合数: {total_combinations}\n")
            f.write("\n参数范围:\n")
            for i, param in enumerate(param_names):
                f.write(f"  {param}: {param_ranges[i]}\n")
            f.write("\n最优配置:\n")
            for param in param_names:
                f.write(f"  {param}: {best_config[param]}\n")
            f.write(f"  加权带宽: {best_config['weighted_bw']:.3f}\n")

            # 添加最优配置的详细指标
            f.write("\n最优配置详细性能指标:\n")

            # 定义要输出的关键性能指标
            perf_metrics = [
                "bw", "weighted_bw", "min_bw",
                "mixed_weighted_bw", "mixed_unweighted_bw",
                "read_weighted_bw", "read_unweighted_bw",
                "write_weighted_bw", "write_unweighted_bw",
                "Total_sum_BW",
                "cmd_mixed_avg_latency", "cmd_mixed_max_latency",
                "data_mixed_avg_latency", "data_mixed_max_latency",
                "trans_mixed_avg_latency", "trans_mixed_max_latency",
                "cmd_read_avg_latency", "cmd_read_max_latency",
                "cmd_write_avg_latency", "cmd_write_max_latency",
                "data_read_avg_latency", "data_read_max_latency",
                "data_write_avg_latency", "data_write_max_latency",
                "trans_read_avg_latency", "trans_read_max_latency",
                "trans_write_avg_latency", "trans_write_max_latency"
            ]

            # 输出存在的性能指标
            for metric in perf_metrics:
                if metric in best_config and isinstance(best_config[metric], (int, float)):
                    f.write(f"  {metric}: {best_config[metric]:.4f}\n")

            # 如果有多个traffic文件，也输出带后缀的指标
            if len(traffic_files) > 1:
                for col in sorted(results_df.columns):
                    if any(metric in col for metric in ["_bw", "_latency", "Total_sum_BW"]) and isinstance(best_config[col], (int, float)):
                        f.write(f"  {col}: {best_config[col]:.4f}\n")

            f.write("\nTop 10 配置:\n")
            top10 = results_df.nlargest(10, "weighted_bw")
            for i, (idx, row) in enumerate(top10.iterrows(), 1):
                f.write(f"{i}. ")
                for param in param_names:
                    f.write(f"{param}={row[param]} ")
                f.write(f"BW={row['weighted_bw']:.3f}\n")

        # 所有结果已在处理过程中追加保存到CSV文件
        print(f"所有结果已保存到: {csv_path}")

        print(f"报告已生成: {report_path}")
    else:
        print("无有效结果数据")

    print("参数遍历完成!")


if __name__ == "__main__":
    main()
