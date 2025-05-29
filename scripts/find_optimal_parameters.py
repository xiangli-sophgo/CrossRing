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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from optuna.exceptions import TrialPruned
from optuna.trial import TrialState

# 使用的 CPU 核心数；-1 表示全部核心
N_JOBS = -1
# 每个参数组合重复仿真次数，用于平滑随机 latency 影响
N_REPEATS = 5  # 减少重复次数，因为要测试多个traffic

# 全局变量用于存储可视化数据
visualization_data = {"trials": [], "progress": [], "pareto_data": [], "param_importance": {}, "convergence": []}


def create_visualization_plots(study, traffic_files, traffic_weights, save_dir):
    """创建全面的可视化图表"""
    print("生成可视化图表...")

    # 创建可视化目录
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 获取完成的trials
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete_trials:
        print("没有完成的trials，跳过可视化")
        return

    # 1. 优化历史图
    create_optimization_history(complete_trials, vis_dir)

    # 2. 参数重要性图
    create_parameter_importance(study, vis_dir)

    # 3. Pareto前沿图
    create_pareto_front(complete_trials, traffic_files, vis_dir)

    # 4. 参数相关性热力图
    create_parameter_correlation(complete_trials, vis_dir)

    # 5. 多Traffic性能对比
    create_multi_traffic_comparison(complete_trials, traffic_files, traffic_weights, vis_dir)

    # 6. 参数分布图
    create_parameter_distribution(complete_trials, vis_dir)

    # 7. 收敛图
    create_convergence_plot(complete_trials, vis_dir)

    # 8. 3D参数空间图
    create_3d_parameter_space(complete_trials, vis_dir)

    print(f"可视化图表已保存到: {vis_dir}")


def create_optimization_history(trials, save_dir):
    """创建优化历史图"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("优化历史", "最佳值历史", "Trial状态分布", "目标函数分布"),
        specs=[[{"secondary_y": True}, {"secondary_y": False}], [{"type": "pie"}, {"type": "histogram"}]],
    )

    # 优化历史
    trial_numbers = [t.number for t in trials]
    objective_values = [t.values[0] for t in trials]
    best_values = []
    best_so_far = float("-inf")
    for val in objective_values:
        if val > best_so_far:
            best_so_far = val
        best_values.append(best_so_far)

    fig.add_trace(go.Scatter(x=trial_numbers, y=objective_values, mode="markers", name="Trial值", opacity=0.6, marker=dict(color="lightblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=trial_numbers, y=best_values, mode="lines", name="最佳值", line=dict(color="red", width=2)), row=1, col=1, secondary_y=False)

    # 最佳值历史（放大）
    fig.add_trace(go.Scatter(x=trial_numbers, y=best_values, mode="lines+markers", name="最佳值历史", line=dict(color="green")), row=1, col=2)

    # Trial状态分布
    states = [t.state.name for t in trials]
    state_counts = pd.Series(states).value_counts()
    fig.add_trace(go.Pie(labels=state_counts.index, values=state_counts.values, name="状态"), row=2, col=1)

    # 目标函数分布
    fig.add_trace(go.Histogram(x=objective_values, name="目标函数分布", nbinsx=30), row=2, col=2)

    fig.update_layout(height=800, title_text="优化过程分析")
    fig.write_html(os.path.join(save_dir, "optimization_history.html"))


def create_parameter_importance(study, save_dir):
    """创建参数重要性图"""
    try:
        # 针对多目标 Study 需指定 target；默认使用第 1 个指标
        if len(study.directions) > 1:
            importance = optuna.importance.get_param_importances(study, target=lambda t: t.values[0])
        else:
            importance = optuna.importance.get_param_importances(study)

        params = list(importance.keys())
        values = list(importance.values())

        fig = go.Figure(data=[go.Bar(x=values, y=params, orientation="h", marker=dict(color=values, colorscale="Viridis"))])

        fig.update_layout(title="参数重要性分析", xaxis_title="重要性分数", yaxis_title="参数", height=600)

        fig.write_html(os.path.join(save_dir, "parameter_importance.html"))
    except Exception as e:
        print("参数重要性分析失败，详细错误信息：")
        traceback.print_exc()


def create_pareto_front(trials, traffic_files, save_dir):
    """创建Pareto前沿图"""
    if len(traffic_files) < 2:
        return

    # 提取两个主要指标
    x_vals = []
    y_vals = []
    colors = []
    texts = []

    traffic1_name = traffic_files[0][:-4]
    traffic2_name = traffic_files[1][:-4]

    for trial in trials:
        if trial.values is not None:
            x_key = f"Total_sum_BW_mean_{traffic1_name}"
            y_key = f"Total_sum_BW_mean_{traffic2_name}"

            if x_key in trial.user_attrs and y_key in trial.user_attrs:
                x_vals.append(trial.user_attrs[x_key])
                y_vals.append(trial.user_attrs[y_key])
                colors.append(trial.user_attrs.get("Total_sum_BW_weighted_mean", trial.values[0]))
                texts.append(f"Trial {trial.number}<br>" f"{traffic1_name}: {x_vals[-1]:.2f}<br>" f"{traffic2_name}: {y_vals[-1]:.2f}")

    if x_vals and y_vals:
        fig = go.Figure(
            data=go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(size=8, color=colors, colorscale="Viridis", colorbar=dict(title="加权得分"), line=dict(width=1, color="black")),
                text=texts,
                hovertemplate="%{text}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Pareto前沿: {traffic1_name} vs {traffic2_name}", xaxis_title=f"{traffic1_name} 带宽 (GB/s)", yaxis_title=f"{traffic2_name} 带宽 (GB/s)", height=600
        )

        fig.write_html(os.path.join(save_dir, "pareto_front.html"))


def create_parameter_correlation(trials, save_dir):
    """创建参数相关性热力图"""
    # 构建数据矩阵
    data = []
    for trial in trials:
        if trial.values is not None:
            row = {}
            for param_name, param_value in trial.params.items():
                row[param_name] = param_value
            row["objective"] = trial.user_attrs.get("Total_sum_BW_weighted_mean", trial.values[0])
            data.append(row)

    if data:
        df = pd.DataFrame(data)
        corr_matrix = df.corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
            )
        )

        fig.update_layout(title="参数相关性矩阵", height=600, width=800)

        fig.write_html(os.path.join(save_dir, "parameter_correlation.html"))


def create_multi_traffic_comparison(trials, traffic_files, traffic_weights, save_dir):
    """创建多Traffic性能对比图"""
    # 准备数据
    traffic_data = {f"{tf[:-4]}": [] for tf in traffic_files}
    weighted_data = []
    trial_numbers = []

    for trial in trials:
        if trial.values is not None:
            trial_numbers.append(trial.number)
            weighted_data.append(trial.user_attrs.get("Total_sum_BW_weighted_mean", 0))

            for tf in traffic_files:
                traffic_name = tf[:-4]
                key = f"Total_sum_BW_mean_{traffic_name}"
                traffic_data[traffic_name].append(trial.user_attrs.get(key, 0))

    # 创建子图
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("各Traffic性能对比", "加权平均vs最小值", "Top 10 Trials对比", "权重影响分析"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": False}, {"type": "bar"}]],
    )

    # 1. 各Traffic性能对比
    colors = px.colors.qualitative.Set1
    for i, (traffic_name, values) in enumerate(traffic_data.items()):
        fig.add_trace(go.Scatter(x=trial_numbers, y=values, mode="markers", name=traffic_name, marker=dict(color=colors[i % len(colors)])), row=1, col=1)

    # 2. 加权平均vs最小值
    min_values = [trial.user_attrs.get("Total_sum_BW_min", 0) for trial in trials if trial.values is not None]
    fig.add_trace(go.Scatter(x=weighted_data, y=min_values, mode="markers", name="加权平均vs最小值", marker=dict(color="red")), row=1, col=2)

    # 3. Top 10 Trials对比
    top_trials = sorted(trials, key=lambda t: t.values[0] if t.values else float("-inf"), reverse=True)[:10]
    top_data = {}
    for tf in traffic_files:
        traffic_name = tf[:-4]
        key = f"Total_sum_BW_mean_{traffic_name}"
        top_data[traffic_name] = [t.user_attrs.get(key, 0) for t in top_trials]

    x_pos = list(range(len(top_trials)))
    for i, (traffic_name, values) in enumerate(top_data.items()):
        fig.add_trace(go.Bar(x=x_pos, y=values, name=f"Top10-{traffic_name}", marker=dict(color=colors[i % len(colors)]), opacity=0.7), row=2, col=1)

    # 4. 权重影响分析
    weight_labels = [f"{tf[:-4]}<br>({w:.1%})" for tf, w in zip(traffic_files, traffic_weights)]
    fig.add_trace(go.Bar(x=weight_labels, y=traffic_weights, name="Traffic权重", marker=dict(color="lightgreen")), row=2, col=2)

    fig.update_layout(height=1000, title_text="多Traffic性能分析")
    fig.write_html(os.path.join(save_dir, "multi_traffic_comparison.html"))


def create_parameter_distribution(trials, save_dir):
    """创建参数分布图"""
    # 获取所有参数
    param_names = list(trials[0].params.keys()) if trials else []
    if not param_names:
        return

    # 创建子图
    n_params = len(param_names)
    cols = 3
    rows = (n_params + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=param_names, specs=[[{"type": "histogram"}] * cols for _ in range(rows)])

    for i, param_name in enumerate(param_names):
        row = i // cols + 1
        col = i % cols + 1

        values = [trial.params[param_name] for trial in trials if trial.values is not None]

        fig.add_trace(go.Histogram(x=values, name=param_name, nbinsx=20), row=row, col=col)

    fig.update_layout(height=300 * rows, title_text="参数分布分析")
    fig.write_html(os.path.join(save_dir, "parameter_distribution.html"))


def create_convergence_plot(trials, save_dir):
    """创建收敛图"""
    if len(trials) < 10:
        return

    # 计算移动平均
    window_sizes = [10, 20, 50]
    trial_numbers = [t.number for t in trials]
    objective_values = [t.values[0] for t in trials]

    fig = go.Figure()

    # 原始数据
    fig.add_trace(go.Scatter(x=trial_numbers, y=objective_values, mode="markers", name="原始值", marker=dict(color="lightblue", size=4, opacity=0.6)))

    # 移动平均
    colors = ["red", "green", "purple"]
    for i, window in enumerate(window_sizes):
        if len(objective_values) >= window:
            moving_avg = pd.Series(objective_values).rolling(window=window, center=True).mean()
            fig.add_trace(go.Scatter(x=trial_numbers, y=moving_avg, mode="lines", name=f"{window}点移动平均", line=dict(color=colors[i], width=2)))

    # 最佳值线
    best_values = []
    best_so_far = float("-inf")
    for val in objective_values:
        if val > best_so_far:
            best_so_far = val
        best_values.append(best_so_far)

    fig.add_trace(go.Scatter(x=trial_numbers, y=best_values, mode="lines", name="最佳值", line=dict(color="orange", width=3, dash="dash")))

    fig.update_layout(title="优化收敛分析", xaxis_title="Trial编号", yaxis_title="目标函数值", height=600)

    fig.write_html(os.path.join(save_dir, "convergence_analysis.html"))


def create_3d_parameter_space(trials, save_dir):
    """创建3D参数空间图"""
    if len(trials) < 20:
        return

    # 选择最重要的3个参数（基于方差）
    param_data = {}
    for trial in trials:
        if trial.values is not None:
            for param_name, param_value in trial.params.items():
                if param_name not in param_data:
                    param_data[param_name] = []
                param_data[param_name].append(param_value)

    # 计算方差选择参数
    param_vars = {name: np.var(values) for name, values in param_data.items()}
    top_params = sorted(param_vars.items(), key=lambda x: x[1], reverse=True)[:3]

    if len(top_params) >= 3:
        param_names = [p[0] for p in top_params]

        x_vals = [trial.params[param_names[0]] for trial in trials if trial.values is not None]
        y_vals = [trial.params[param_names[1]] for trial in trials if trial.values is not None]
        z_vals = [trial.params[param_names[2]] for trial in trials if trial.values is not None]
        colors = [t.user_attrs.get("Total_sum_BW_weighted_mean", t.values[0]) for t in trials if t.values is not None]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode="markers",
                    marker=dict(size=5, color=colors, colorscale="Viridis", colorbar=dict(title="目标函数值"), line=dict(width=0.5, color="black")),
                    text=[f"Trial {t.number}: {t.values}" for t in trials if t.values is not None],
                    hovertemplate="%{text}<extra></extra>",
                )
            ]
        )

        fig.update_layout(title="3D参数空间探索", scene=dict(xaxis_title=param_names[0], yaxis_title=param_names[1], zaxis_title=param_names[2]), height=700)

        fig.write_html(os.path.join(save_dir, "3d_parameter_space.html"))


def save_progress_callback(study, trial):
    """实时保存进度和中间结果"""
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
            progress_file = os.path.join(output_csv.replace(".csv", "_progress.json"))
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(visualization_data, f, indent=2, ensure_ascii=False)

            # 如果有足够的数据，生成中间可视化
            if len(visualization_data["trials"]) >= 20:
                create_intermediate_visualization(study)

        except Exception as e:
            print(f"保存进度数据失败: {e}")


def create_intermediate_visualization(study):
    """创建中间过程可视化"""
    try:
        vis_dir = os.path.join(os.path.dirname(output_csv), "intermediate_vis")
        os.makedirs(vis_dir, exist_ok=True)

        # 创建简化的实时图表
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

        if len(complete_trials) >= 10:
            # 1. 实时优化历史
            trial_numbers = [t.number for t in complete_trials]
            objective_values = [t.values[0] for t in complete_trials]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trial_numbers, y=objective_values, mode="markers+lines", name="目标函数值"))

            # 最佳值线
            best_values = []
            best_so_far = float("-inf")
            for val in objective_values:
                if val > best_so_far:
                    best_so_far = val
                best_values.append(best_so_far)

            fig.add_trace(go.Scatter(x=trial_numbers, y=best_values, mode="lines", name="最佳值", line=dict(color="red", width=2)))

            fig.update_layout(title=f"实时优化进度 (已完成{len(complete_trials)}个试验)", xaxis_title="Trial编号", yaxis_title="目标函数值")

            fig.write_html(os.path.join(vis_dir, "realtime_progress.html"))

            print(f"中间可视化已更新: {vis_dir}/realtime_progress.html")

    except Exception as e:
        print(f"创建中间可视化失败: {e}")


def find_optimal_parameters():
    global output_csv

    traffic_file_path = r"../test_data/"

    # ===== 多个traffic文件配置 =====
    traffic_files = [
        r"traffic_2260E_case1.txt",
        r"traffic_2260E_case2.txt",  # 添加你的第二个traffic文件
        # r"traffic_2260E_case3.txt",  # 可以继续添加更多
    ]

    # 每个traffic的权重（用于加权平均）
    traffic_weights = [0.5, 0.5]  # 第一个traffic权重0.6，第二个0.4

    assert len(traffic_files) == len(traffic_weights), "traffic文件数量和权重数量必须一致"
    assert abs(sum(traffic_weights) - 1.0) < 1e-6, "权重总和必须等于1"

    config_path = r"../config/config2.json"
    config = CrossRingConfig(config_path)

    topo_type = config.TOPO_TYPE or "3x3"
    config.TOPO_TYPE = topo_type

    model_type = "REQ_RSP"
    results_file_name = f"2260E_ETag_multi_traffic_{datetime.now().strftime('%m%d_%H%M')}"
    result_root_save_path = f"../Result/CrossRing/{model_type}/FOP/{results_file_name}/"
    os.makedirs(result_root_save_path, exist_ok=True)
    output_csv = os.path.join(r"../Result/Params_csv/", f"{results_file_name}.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 参数范围
    param1_start, param1_end = 2, 16
    param2_start, param2_end = 2, 16
    param3_start, param3_end = 2, 16
    param4_start, param4_end = 4, 16
    param5_start, param5_end = 2, 16
    param6_start, param6_end = 2, 16
    param7_start, param7_end = 2, 16
    param8_start, param8_end = 4, 16

    def _run_one_traffic(traffic_file, param1, param2, param3, param4, param5, param6, param7, param8):
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
                file_name=traffic_file,
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
                sim.config.NUM_RN = 4
                sim.config.NUM_SN = 8
                sim.config.RN_R_TRACKER_OSTD = 128
                sim.config.RN_W_TRacker_OSTD = 32
                sim.config.RN_RDB_SIZE = sim.config.RN_R_TRACKER_OSTD * sim.config.BURST
                sim.config.RN_WDB_SIZE = sim.config.RN_W_TRacker_OSTD * sim.config.BURST
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
                sim.config.IQ_CH_FIFO_DEPTH = 10
                sim.config.EQ_CH_FIFO_DEPTH = 12
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
                sim.config.GDMA_RW_GAP = np.inf
                sim.config.SDMA_RW_GAP = 50
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
                print(f"[{traffic_file}][RPT {rpt}] Sim failed for params: {param1}, {param2}, {param3}, {param4}, {param5}, {param6}, {param7}, {param8}")
                print("Exception details (full traceback):")
                traceback.print_exc()
                bw = 0
            tot_bw_list.append(bw)

        bw_mean = float(np.mean(tot_bw_list))
        bw_std = float(np.std(tot_bw_list))

        return {
            f"Total_sum_BW_mean_{traffic_file[:-4]}": bw_mean,
            f"Total_sum_BW_std_{traffic_file[:-4]}": bw_std,
        }

    def _run_one(param1, param2, param3, param4, param5, param6, param7, param8):
        """运行所有traffic文件并综合结果"""
        all_results = {}
        all_bw_means = []

        for traffic_file in traffic_files:
            try:
                result = _run_one_traffic(traffic_file, param1, param2, param3, param4, param5, param6, param7, param8)
                all_results.update(result)
                bw_mean = result[f"Total_sum_BW_mean_{traffic_file[:-4]}"]
                all_bw_means.append(bw_mean)
            except Exception as e:
                print(f"Error processing {traffic_file}: {e}")
                all_bw_means.append(0)
                all_results[f"Total_sum_BW_mean_{traffic_file[:-4]}"] = 0
                all_results[f"Total_sum_BW_std_{traffic_file[:-4]}"] = 0

        # 计算加权平均带宽
        weighted_bw_mean = sum(bw * weight for bw, weight in zip(all_bw_means, traffic_weights))

        # 计算最小带宽（保证所有traffic都有合理性能）
        min_bw_mean = min(all_bw_means) if all_bw_means else 0

        # 计算带宽方差（衡量不同traffic间的一致性）
        bw_variance = np.var(all_bw_means) if len(all_bw_means) > 1 else 0

        # 添加综合指标
        all_results.update(
            {
                "Total_sum_BW_weighted_mean": weighted_bw_mean,
                "Total_sum_BW_min": min_bw_mean,
                "Total_sum_BW_variance": bw_variance,
                "param1": param1,
                "param2": param2,
                "param3": param3,
                "param4": param4,
                "param5": param5,
                "param6": param6,
                "param7": param7,
                "param8": param8,
            }
        )

        return all_results

    def objective(trial):
        # 采样参数
        p1 = trial.suggest_int("TL_Etag_T2_UE_MAX", param1_start, param1_end)
        # 保证 p2 > p1
        p2_low = p1 + 1
        if p2_low > param2_end:
            raise TrialPruned()
        p2 = trial.suggest_int("TL_Etag_T1_UE_MAX", p2_low, param2_end)
        p3 = trial.suggest_int("TR_Etag_T2_UE_MAX", param3_start, param3_end)
        # 保证 p4 > max(p2, p3)
        p4_low = max(p2, p3) + 1
        if p4_low > param4_end:
            raise TrialPruned()
        p4 = trial.suggest_int("RB_IN_FIFO_DEPTH", p4_low, param4_end)

        # 新增参数
        p5 = trial.suggest_int("TU_Etag_T2_UE_MAX", param5_start, param5_end)
        # 保证 p6 > p5
        p6_low = p5 + 1
        if p6_low > param6_end:
            raise TrialPruned()
        p6 = trial.suggest_int("TU_Etag_T1_UE_MAX", p6_low, param6_end)
        p7 = trial.suggest_int("TD_Etag_T2_UE_MAX", param7_start, param7_end)
        # 保证 p8 > max(p6, p7)
        p8_low = max(p6, p7) + 1
        if p8_low > param8_end:
            raise TrialPruned()
        p8 = trial.suggest_int("EQ_IN_FIFO_DEPTH", p8_low, param8_end)

        results = _run_one(p1, p2, p3, p4, p5, p6, p7, p8)

        # ─── 两个 traffic 的带宽均值 ──────────────────────────
        bw1_mean = results[f"Total_sum_BW_mean_{traffic_files[0][:-4]}"]
        bw2_mean = results[f"Total_sum_BW_mean_{traffic_files[1][:-4]}"] if len(traffic_files) > 1 else 0.0

        # ─── 参数规模归一化（越小越好）───────────────────────
        param_norm = (
            # (p1 - param1_start) / (param1_end - param1_start)
            # + (p2 - param2_start) / (param2_end - param2_start)
            # + (p3 - param3_start) / (param3_end - param3_start)
            +(p4 - param4_start) / (param4_end - param4_start)
            # + (p5 - param5_start) / (param5_end - param5_start)
            # + (p6 - param6_start) / (param6_end - param6_start)
            # + (p7 - param7_start) / (param7_end - param7_start)
            + (p8 - param8_start) / (param8_end - param8_start)
        ) / 2.0

        # 保存到 trial.user_attrs，便于后期分析 / CSV
        for k, v in results.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("param_norm", param_norm)

        # ─── 多目标返回： (maximize, maximize, minimize) ────
        return bw1_mean, bw2_mean, param_norm

    return objective, output_csv, traffic_files, traffic_weights, result_root_save_path


def save_intermediate_result(study, trial):
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
    pd.DataFrame(records).to_csv(output_csv, index=False)

    # 保存进度并创建实时可视化
    save_progress_callback(study, trial)


def create_summary_report(study, traffic_files, traffic_weights, save_dir):
    """创建HTML总结报告"""
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if not complete_trials:
        return

    # 获取最佳试验
    best_trial = study.best_trials[0] if study.best_trials else None

    # 获取Top 10试验
    top_trials = sorted(complete_trials, key=lambda t: t.values[0] if t.values else -np.inf, reverse=True)[:10]

    # 生成HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>多Traffic参数优化报告</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .best-params {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; }}
            .table {{ border-collapse: collapse; width: 100%; }}
            .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .table th {{ background-color: #f2f2f2; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>多Traffic参数优化报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>完成试验数: {len(complete_trials)} / {len(study.trials)}</p>
        </div>
        
        <div class="section">
            <h2>优化配置</h2>
            <div class="metric">
                <strong>Traffic文件:</strong><br>
                {', '.join(traffic_files)}
            </div>
            <div class="metric">
                <strong>权重配置:</strong><br>
                {', '.join([f'{tf[:-4]}: {w:.1%}' for tf, w in zip(traffic_files, traffic_weights)])}
            </div>
        </div>
        
        <div class="section best-params">
            <h2>最佳配置</h2>
            <p><strong>最佳指标(BW1, BW2, Norm):</strong> {best_trial.values}</p>
            <p><strong>最佳参数:</strong></p>
            <ul>
    """

    for param, value in best_trial.params.items():
        html_content += f"<li>{param}: {value}</li>"

    html_content += f"""
            </ul>
            <p><strong>性能详情:</strong></p>
            <ul>
    """

    for traffic_file in traffic_files:
        traffic_name = traffic_file[:-4]
        if f"Total_sum_BW_mean_{traffic_name}" in best_trial.user_attrs:
            bw = best_trial.user_attrs[f"Total_sum_BW_mean_{traffic_name}"]
            html_content += f"<li>{traffic_name}: {bw:.2f} GB/s</li>"

    weighted_bw = best_trial.user_attrs.get("Total_sum_BW_weighted_mean", 0)
    min_bw = best_trial.user_attrs.get("Total_sum_BW_min", 0)
    variance = best_trial.user_attrs.get("Total_sum_BW_variance", 0)

    html_content += f"""
                <li>加权平均: {weighted_bw:.2f} GB/s</li>
                <li>最小值: {min_bw:.2f} GB/s</li>
                <li>方差: {variance:.2f}</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>Top 10 配置</h2>
            <table class="table">
                <tr>
                    <th>排名</th>
                    <th>Trial</th>
                    <th>指标(BW1, BW2, Norm)</th>
                    <th>加权平均</th>
                    <th>最小值</th>
                    <th>方差</th>
    """

    for param in best_trial.params.keys():
        html_content += f"<th>{param}</th>"

    html_content += "</tr>"

    for i, trial in enumerate(top_trials, 1):
        weighted = trial.user_attrs.get("Total_sum_BW_weighted_mean", 0)
        min_val = trial.user_attrs.get("Total_sum_BW_min", 0)
        var_val = trial.user_attrs.get("Total_sum_BW_variance", 0)

        html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{trial.number}</td>
                    <td>{trial.values}</td>
                    <td>{weighted:.2f}</td>
                    <td>{min_val:.2f}</td>
                    <td>{var_val:.2f}</td>
        """

        for param in best_trial.params.keys():
            html_content += f"<td>{trial.params.get(param, '-')}</td>"

        html_content += "</tr>"

    html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>可视化图表</h2>
            <p>详细的可视化分析请查看以下文件:</p>
            <ul>
                <li><a href="visualizations/optimization_history.html">优化历史</a></li>
                <li><a href="visualizations/parameter_importance.html">参数重要性</a></li>
                <li><a href="visualizations/pareto_front.html">Pareto前沿</a></li>
                <li><a href="visualizations/multi_traffic_comparison.html">多Traffic对比</a></li>
                <li><a href="visualizations/convergence_analysis.html">收敛分析</a></li>
                <li><a href="visualizations/3d_parameter_space.html">3D参数空间</a></li>
            </ul>
        </div>
        
        <div class="section">
            <h2>实时监控</h2>
            <p>运行过程中的实时图表:</p>
            <ul>
                <li><a href="intermediate_vis/realtime_progress.html">实时进度</a></li>
            </ul>
        </div>
    </body>
    </html>
    """

    # 保存HTML报告
    with open(os.path.join(save_dir, "optimization_report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"总结报告已生成: {save_dir}/optimization_report.html")


if __name__ == "__main__":
    objective, output_csv, traffic_files, traffic_weights, result_root_save_path = find_optimal_parameters()

    print("=" * 60)
    print(f"开始多Traffic优化 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Traffic文件: {traffic_files}")
    print(f"权重: {traffic_weights}")
    print(f"结果保存路径: {result_root_save_path}")
    print("=" * 60)

    study = optuna.create_study(
        study_name="CrossRing_Multi_Traffic_BO",
        directions=["maximize", "maximize", "minimize"],  # bw1 ↑  bw2 ↑  param_norm ↓
        sampler=optuna.samplers.NSGAIISampler(seed=527),
    )

    try:
        study.optimize(objective, n_trials=300, n_jobs=N_JOBS, show_progress_bar=True, callbacks=[save_intermediate_result], catch=(KeyboardInterrupt,))  # 可能需要增加trial数量
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
    print("优化完成!")
    if study.best_trials:
        print("最佳指标(BW1, BW2, Norm):", study.best_trials[0].values)
        print("最佳参数:", study.best_trials[0].params)

    # 显示最佳结果的详细信息
    if study.best_trials:
        best_trial = study.best_trials[0]
        print("\n最佳配置的详细结果:")
        for traffic_file in traffic_files:
            traffic_name = traffic_file[:-4]
            if f"Total_sum_BW_mean_{traffic_name}" in best_trial.user_attrs:
                print(f"  {traffic_name}: {best_trial.user_attrs[f'Total_sum_BW_mean_{traffic_name}']:.2f} GB/s")
        print(f"  加权平均: {best_trial.user_attrs.get('Total_sum_BW_weighted_mean', 0):.2f} GB/s")
        print(f"  最小值: {best_trial.user_attrs.get('Total_sum_BW_min', 0):.2f} GB/s")
        print(f"  方差: {best_trial.user_attrs.get('Total_sum_BW_variance', 0):.2f}")

    # 创建最终可视化
    print("\n正在生成最终可视化报告...")
    try:
        create_visualization_plots(study, traffic_files, traffic_weights, result_root_save_path)
        print(f"可视化报告已生成: {result_root_save_path}/visualizations/")

        # 创建总结报告
        create_summary_report(study, traffic_files, traffic_weights, result_root_save_path)

    except Exception as e:
        print(f"生成可视化失败: {e}")
        traceback.print_exc()

    print("=" * 60)
