import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.noc import *
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
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

# 使用的 CPU 核心数；-1 表示全部核心
N_JOBS = 1
# 每个参数组合重复仿真次数，用于平滑随机 latency 影响
N_REPEATS = 1  # 减少重复次数，因为要测试多个traffic
N_TRIALS = 300

# 全局变量用于存储可视化数据
visualization_data = {"trials": [], "progress": [], "pareto_data": [], "param_importance": {}, "convergence": []}


def create_parameter_impact_plot(trials, traffic_files, save_dir):
    """增强版参数影响分析图"""
    if not trials:
        return

    # 获取参数和性能数据
    param_names = list(trials[0].params.keys())
    traffic_names = [tf[:-4] for tf in traffic_files]

    # 准备数据
    param_data = {param: [] for param in param_names}
    performance_data = {traffic: [] for traffic in traffic_names}
    weighted_performance = []

    for trial in trials:
        if trial.values is not None:
            # 参数数据
            for param in param_names:
                param_data[param].append(trial.params[param])

            # 性能数据
            for traffic in traffic_names:
                key = f"Total_sum_BW_mean_{traffic}"
                performance_data[traffic].append(trial.user_attrs.get(key, 0))

            weighted_performance.append(trial.user_attrs.get("Total_sum_BW_weighted_mean", 0))

    # 1. 参数敏感性分析热力图
    fig1 = create_parameter_sensitivity_heatmap(param_data, performance_data, traffic_names, param_names)
    fig1.write_html(os.path.join(save_dir, "parameter_sensitivity_heatmap.html"))

    # 2. 参数影响力散点图矩阵
    fig2 = create_parameter_scatter_matrix(param_data, weighted_performance, param_names)
    fig2.write_html(os.path.join(save_dir, "parameter_scatter_matrix.html"))

    # 3. 参数协同效应分析
    fig3 = create_parameter_synergy_analysis(param_data, weighted_performance, param_names)
    fig3.write_html(os.path.join(save_dir, "parameter_synergy_analysis.html"))


def create_parameter_sensitivity_heatmap(param_data, performance_data, traffic_names, param_names):
    """创建参数敏感性热力图"""
    from scipy import stats

    # 计算相关系数矩阵
    correlation_matrix: np.ndarray = np.zeros((len(param_names), len(traffic_names)))

    for i, param in enumerate(param_names):
        for j, traffic in enumerate(traffic_names):
            if len(param_data[param]) > 1 and len(performance_data[traffic]) > 1:
                # 检查是否为常数输入
                param_std = np.std(param_data[param])
                perf_std = np.std(performance_data[traffic])
                if param_std > 1e-10 and perf_std > 1e-10:  # 避免常数输入
                    try:
                        corr, _ = stats.pearsonr(param_data[param], performance_data[traffic])
                        correlation_matrix[i, j] = corr if not np.isnan(corr) else 0
                    except:
                        correlation_matrix[i, j] = 0
                else:
                    correlation_matrix[i, j] = 0

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix,
            x=traffic_names,
            y=param_names,
            colorscale="RdBu",
            zmid=0,
            text=np.round(correlation_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
            colorbar=dict(title="相关系数"),
        )
    )

    fig.update_layout(title="参数对不同Traffic性能的敏感性分析", xaxis_title="Traffic类型", yaxis_title="参数", height=600, width=800)

    return fig


def create_parameter_scatter_matrix(param_data, performance, param_names):
    """创建参数影响散点图矩阵"""
    n_params = len(param_names)
    cols = min(3, n_params)
    rows = (n_params + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"{param} vs 性能" for param in param_names], vertical_spacing=0.08, horizontal_spacing=0.08)

    for i, param in enumerate(param_names):
        row = i // cols + 1
        col = i % cols + 1

        x_vals = param_data[param]
        y_vals = performance

        # 添加散点图
        fig.add_trace(
            go.Scatter(x=x_vals, y=y_vals, mode="markers", marker=dict(size=6, color=y_vals, colorscale="Viridis", opacity=0.7, line=dict(width=1, color="black")), name=param, showlegend=False),
            row=row,
            col=col,
        )

        # 添加趋势线
        if len(x_vals) > 2:
            try:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(x_vals), max(x_vals), 100)
                y_trend = p(x_trend)

                fig.add_trace(go.Scatter(x=x_trend, y=y_trend, mode="lines", line=dict(color="red", width=2), name=f"{param}_trend", showlegend=False), row=row, col=col)
            except:
                pass

    fig.update_layout(title="参数对性能的个体影响分析", height=300 * rows, width=1200)

    return fig


def create_parameter_synergy_analysis(param_data, performance, param_names):
    """创建参数协同效应分析"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression

    # 使用随机森林分析参数重要性和交互效应
    X = np.array([param_data[param] for param in param_names]).T
    y = np.array(performance)

    if len(X) < 10:
        # 数据太少，创建简单的相关性分析
        return create_simple_correlation_plot(param_data, performance, param_names)

    # 随机森林特征重要性
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importance = rf.feature_importances_

    # 创建子图
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=("参数重要性排名", "参数交互强度", "参数聚类分析", "协同效应网络"), specs=[[{"type": "bar"}, {"type": "heatmap"}], [{"type": "scatter"}, {"type": "scatter"}]]
    )

    # 1. 参数重要性排名
    sorted_idx = np.argsort(feature_importance)[::-1]
    fig.add_trace(
        go.Bar(x=[param_names[i] for i in sorted_idx], y=[feature_importance[i] for i in sorted_idx], marker=dict(color=feature_importance[sorted_idx], colorscale="Viridis"), name="重要性"),
        row=1,
        col=1,
    )

    # 2. 参数间相关性热力图
    param_corr = np.corrcoef(X.T)
    fig.add_trace(go.Heatmap(z=param_corr, x=param_names, y=param_names, colorscale="RdBu", zmid=0, showscale=False), row=1, col=2)

    fig.update_layout(title="参数协同效应深度分析", height=800, width=1200)

    return fig


def create_simple_correlation_plot(param_data, performance, param_names):
    """数据较少时的简单相关性分析"""
    from scipy import stats

    correlations = []
    for param in param_names:
        if len(param_data[param]) > 1:
            corr, _ = stats.pearsonr(param_data[param], performance)
            correlations.append(corr if not np.isnan(corr) else 0)
        else:
            correlations.append(0)

    fig = go.Figure(data=[go.Bar(x=param_names, y=correlations, marker=dict(color=correlations, colorscale="RdBu", colorbar=dict(title="相关系数")))])

    fig.update_layout(title="参数与性能的相关性分析", xaxis_title="参数", yaxis_title="相关系数", height=500)

    return fig


def create_enhanced_optimization_insight(study, vis_dir):
    """增强版优化过程深度洞察"""
    complete_trials = [t for t in study.trials if t.state.name == "COMPLETE"]

    if len(complete_trials) < 10:
        return

    # 准备数据
    trial_numbers = [t.number for t in complete_trials]
    objective_values = [t.values[0] if t.values else 0 for t in complete_trials]
    param_names = list(complete_trials[0].params.keys()) if complete_trials else []

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("探索与利用平衡分析", "参数进化趋势", "性能突破点识别", "优化效率分析"),
        specs=[[{"secondary_y": True}, {"secondary_y": False}], [{"secondary_y": False}, {"secondary_y": False}]],
    )

    # 1. 探索与利用平衡分析
    exploration_scores, exploitation_scores = analyze_exploration_exploitation(complete_trials)

    fig.add_trace(go.Scatter(x=trial_numbers, y=exploration_scores, mode="lines+markers", name="探索度", line=dict(color="blue", width=2), marker=dict(size=4)), row=1, col=1)

    fig.add_trace(go.Scatter(x=trial_numbers, y=exploitation_scores, mode="lines+markers", name="利用度", line=dict(color="red", width=2), marker=dict(size=4)), row=1, col=1, secondary_y=True)

    # 2. 参数进化趋势
    if param_names:
        param_evolution = [trial.params.get(param_names[0], 0) for trial in complete_trials]
        fig.add_trace(go.Scatter(x=trial_numbers, y=param_evolution, mode="lines+markers", name=f"{param_names[0]} 进化", line=dict(color="green", width=2), marker=dict(size=4)), row=1, col=2)

    # 3. 性能突破点识别
    breakthrough_points = identify_breakthrough_points(objective_values, trial_numbers)

    fig.add_trace(go.Scatter(x=trial_numbers, y=objective_values, mode="lines+markers", name="性能轨迹", line=dict(color="gray", width=1), marker=dict(size=3, opacity=0.6)), row=2, col=1)

    if breakthrough_points:
        breakthrough_x, breakthrough_y = zip(*breakthrough_points)
        fig.add_trace(go.Scatter(x=breakthrough_x, y=breakthrough_y, mode="markers", name="突破点", marker=dict(size=10, color="red", symbol="star", line=dict(width=2, color="black"))), row=2, col=1)

    # 4. 优化效率分析
    efficiency_scores = calculate_optimization_efficiency(objective_values)
    fig.add_trace(
        go.Scatter(x=trial_numbers[1:], y=efficiency_scores, mode="lines+markers", name="优化效率", line=dict(color="purple", width=2), marker=dict(size=4)), row=2, col=2  # 从第二个trial开始
    )

    fig.update_layout(title="优化过程深度洞察分析", height=800, width=1200, showlegend=True)

    fig.write_html(os.path.join(vis_dir, "enhanced_optimization_insight.html"))


def analyze_exploration_exploitation(trials):
    """分析探索与利用的平衡"""
    exploration_scores = []
    exploitation_scores = []

    for i, trial in enumerate(trials):
        if i == 0:
            exploration_scores.append(1.0)
            exploitation_scores.append(0.0)
            continue

        # 计算当前参数配置与历史最佳配置的差异（探索度）
        best_so_far = max(trials[:i], key=lambda x: x.values[0] if x.values else -np.inf)

        param_diff = 0
        param_count = 0
        for param_name in trial.params:
            if param_name in best_so_far.params:
                param_diff += abs(trial.params[param_name] - best_so_far.params[param_name])
                param_count += 1

        if param_count > 0:
            exploration = param_diff / param_count
            exploration_scores.append(min(exploration / 5.0, 1.0))  # 归一化
        else:
            exploration_scores.append(0.0)

        # 计算利用度（基于与历史最佳的相似性）
        exploitation_scores.append(1.0 - exploration_scores[-1])

    return exploration_scores, exploitation_scores


def identify_breakthrough_points(objective_values, trial_numbers, threshold=0.05):
    """识别性能突破点"""
    breakthrough_points = []

    if len(objective_values) < 3:
        return breakthrough_points

    # 计算移动平均
    window = min(5, len(objective_values) // 3)
    moving_avg = pd.Series(objective_values).rolling(window=window, center=True).mean()

    # 识别显著提升点
    for i in range(window, len(objective_values) - window):
        if i > 0:
            improvement = (moving_avg.iloc[i] - moving_avg.iloc[i - 1]) / abs(moving_avg.iloc[i - 1])
            if improvement > threshold:
                breakthrough_points.append((trial_numbers[i], objective_values[i]))

    return breakthrough_points


def calculate_optimization_efficiency(objective_values):
    """计算优化效率"""
    efficiency_scores = []

    for i in range(1, len(objective_values)):
        # 计算改进率
        if i == 1:
            improvement = 1.0 if objective_values[i] > objective_values[i - 1] else 0.0
        else:
            # 基于最近几次试验的平均改进
            recent_window = min(5, i)
            recent_best = max(objective_values[max(0, i - recent_window) : i])
            current_value = objective_values[i]

            if recent_best > 0:
                improvement = max(0, (current_value - recent_best) / recent_best)
            else:
                improvement = 1.0 if current_value > recent_best else 0.0

        efficiency_scores.append(improvement)

    return efficiency_scores


def create_optimization_guidance_report(study, traffic_files, save_dir):
    """创建参数优化指导报告"""
    complete_trials = [t for t in study.trials if t.state.name == "COMPLETE"]

    if len(complete_trials) < 10:
        return

    # 分析每个参数的优化策略
    param_names = list(complete_trials[0].params.keys()) if complete_trials else []
    optimization_insights = {}

    for param in param_names:
        insights = analyze_parameter_optimization_strategy(complete_trials, param)
        optimization_insights[param] = insights

    # 生成优化指导文档
    guidance_html = generate_optimization_guidance_html(optimization_insights, complete_trials, traffic_files)

    with open(os.path.join(save_dir, "optimization_guidance.html"), "w", encoding="utf-8") as f:
        f.write(guidance_html)


def analyze_parameter_optimization_strategy(trials, param_name):
    """分析单个参数的优化策略"""
    from scipy import stats

    param_values = [t.params[param_name] for t in trials]
    performance_values = [t.values[0] for t in trials]

    # 计算相关性
    correlation, p_value = stats.pearsonr(param_values, performance_values)

    # 找出最佳值范围
    top_trials = sorted(trials, key=lambda x: x.values[0], reverse=True)[: len(trials) // 4]
    best_param_values = [t.params[param_name] for t in top_trials]

    optimal_range = (min(best_param_values), max(best_param_values))
    optimal_mean = np.mean(best_param_values)

    # 分析参数敏感性
    sensitivity = analyze_parameter_sensitivity(param_values, performance_values)

    # 生成优化建议
    recommendations = generate_parameter_recommendations(param_name, correlation, optimal_range, optimal_mean, sensitivity)

    return {"correlation": correlation, "p_value": p_value, "optimal_range": optimal_range, "optimal_mean": optimal_mean, "sensitivity": sensitivity, "recommendations": recommendations}


def analyze_parameter_sensitivity(param_values, performance_values):
    """分析参数敏感性"""
    if len(param_values) < 5:
        return "数据不足"

    # 计算参数变化对性能的影响
    param_changes = np.diff(param_values)
    perf_changes = np.diff(performance_values)

    # 避免除零错误
    non_zero_changes = param_changes[param_changes != 0]
    corresponding_perf_changes = perf_changes[param_changes != 0]

    if len(non_zero_changes) == 0:
        return "低敏感性"

    # 计算敏感性分数
    sensitivity_scores = np.abs(corresponding_perf_changes / non_zero_changes)
    avg_sensitivity = np.mean(sensitivity_scores)

    if avg_sensitivity > 1.0:
        return "高敏感性"
    elif avg_sensitivity > 0.5:
        return "中等敏感性"
    else:
        return "低敏感性"


def generate_parameter_recommendations(param_name, correlation, optimal_range, optimal_mean, sensitivity):
    """生成参数优化建议"""
    recommendations = []

    # 基于相关性的建议
    if abs(correlation) > 0.7:
        if correlation > 0:
            recommendations.append(f"📈 {param_name} 与性能呈强正相关，建议适当增大此参数")
        else:
            recommendations.append(f"📉 {param_name} 与性能呈强负相关，建议适当减小此参数")
    elif abs(correlation) > 0.3:
        recommendations.append(f"📊 {param_name} 与性能存在中等相关性，需要结合其他参数综合考虑")
    else:
        recommendations.append(f"🔄 {param_name} 与性能相关性较弱，可能存在非线性关系或参数冗余")

    # 基于最优范围的建议
    range_size = optimal_range[1] - optimal_range[0]
    if range_size <= 2:
        recommendations.append(f"🎯 最佳值集中在 {optimal_range[0]}-{optimal_range[1]} 范围内，建议精确调优")
    else:
        recommendations.append(f"🔍 最佳值分布在 {optimal_range[0]}-{optimal_range[1]} 范围内，建议在此范围内探索")

    # 基于敏感性的建议
    if sensitivity == "高敏感性":
        recommendations.append(f"⚠️ {param_name} 高度敏感，小幅调整即可显著影响性能，需要精细调优")
    elif sensitivity == "中等敏感性":
        recommendations.append(f"⚖️ {param_name} 中等敏感，可以适度调整来优化性能")
    else:
        recommendations.append(f"🔧 {param_name} 敏感性较低，可以大幅调整或考虑固定此参数")

    # 推荐起始值
    recommendations.append(f"💡 建议起始值: {int(optimal_mean):.1f} (基于最佳试验的平均值)")

    return recommendations


def generate_optimization_guidance_html(optimization_insights, trials, traffic_files):
    """生成优化指导HTML文档"""

    # 获取最佳试验
    best_trial = max(trials, key=lambda x: x.values[0])

    # 计算整体统计信息
    performance_values = [t.values[0] for t in trials]
    performance_improvement = (max(performance_values) - min(performance_values)) / min(performance_values) * 100

    # 找出最重要的参数
    param_importance = {param: abs(insights.get("correlation", 0)) for param, insights in optimization_insights.items()}
    most_important_param = max(param_importance.keys(), key=lambda x: param_importance[x])

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>参数优化策略指导</title>
        <meta charset="utf-8">
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
                margin: 40px; 
                line-height: 1.6;
                color: #333;
            }}
            .header {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px; 
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .section {{ 
                margin: 25px 0; 
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #007bff;
            }}
            .param-analysis {{
                background: white;
                padding: 20px;
                margin: 15px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric {{ 
                display: inline-block; 
                margin: 10px 15px; 
                padding: 15px 20px; 
                background: white;
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                min-width: 200px;
            }}
            .recommendations {{
                background: #e8f5e8;
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
            }}
            .recommendations ul {{
                margin: 10px 0;
                padding-left: 0;
                list-style: none;
            }}
            .recommendations li {{
                margin: 8px 0;
                padding: 8px 12px;
                background: white;
                border-radius: 5px;
                border-left: 3px solid #28a745;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 参数优化策略指导报告</h1>
            <p>基于 {len(trials)} 次试验的深度分析 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>性能提升: <strong>{performance_improvement:.1f}%</strong> | 最佳性能: <strong>{best_trial.values[0]:.2f}</strong></p>
        </div>
        
        <div class="section">
            <h2>🏆 关键发现</h2>
            <ul>
                <li><strong>最关键参数</strong>: {most_important_param}</li>
                <li><strong>优化潜力</strong>: {performance_improvement:.1f}% 性能提升空间</li>
                <li><strong>稳定性</strong>: {'高' if performance_improvement < 50 else '中等'}</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>📋 详细参数分析</h2>
    """

    # 为每个参数生成详细分析
    for param_name, insights in optimization_insights.items():
        correlation = insights.get("correlation", 0)
        optimal_range = insights.get("optimal_range", (0, 0))
        optimal_mean = insights.get("optimal_mean", 0)
        sensitivity = insights.get("sensitivity", "未知")
        recommendations = insights.get("recommendations", [])

        html_content += f"""
            <div class="param-analysis">
                <h3>🔧 {param_name}</h3>
                <p><strong>相关性</strong>: {correlation:.3f} | <strong>最优范围</strong>: {optimal_range[0]:.0f}-{optimal_range[1]:.0f} | <strong>推荐值</strong>: {int(optimal_mean)}</p>
                
                <div class="recommendations">
                    <h4>🎯 优化建议:</h4>
                    <ul>
        """

        for rec in recommendations:
            html_content += f"<li>{rec}</li>"

        html_content += """
                    </ul>
                </div>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    return html_content


# ============= 修改主函数中的调用 =============
def enhanced_create_visualization_plots(study, traffic_files, traffic_weights, save_dir):
    """增强版可视化图表创建函数 - 在原有基础上添加新功能"""
    print("生成增强版可视化图表...")

    # 创建可视化目录
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 获取完成的trials
    complete_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not complete_trials:
        print("没有完成的trials，跳过可视化")
        return

    # ========== 保留所有原有的可视化函数 ==========
    create_optimization_history(complete_trials, vis_dir)
    create_parameter_importance(study, vis_dir)
    create_pareto_front(complete_trials, traffic_files, vis_dir)
    create_parameter_correlation(complete_trials, vis_dir)
    create_multi_traffic_comparison(complete_trials, traffic_files, traffic_weights, vis_dir)
    create_parameter_distribution(complete_trials, vis_dir)
    create_convergence_plot(complete_trials, vis_dir)
    create_3d_parameter_space(complete_trials, vis_dir)
    create_single_parameter_line_plots(complete_trials, save_dir=vis_dir)

    # ========== 新增增强版可视化函数 ==========
    create_parameter_impact_plot(complete_trials, traffic_files, vis_dir)
    create_enhanced_optimization_insight(study, vis_dir)
    create_optimization_guidance_report(study, traffic_files, vis_dir)
    # 新增 2D 参数×带宽热力图
    create_2d_param_bw_heatmaps(
        complete_trials,
        metric_key="Total_sum_BW_weighted_mean",  # 你也可以用单个 traffic 的 key
        # 手动指定感兴趣的参数对；留空则所有两两组合都会画
        param_pairs=[
            ("TL_Etag_T2_UE_MAX", "TL_Etag_T1_UE_MAX"),
            ("TU_Etag_T2_UE_MAX", "TU_Etag_T1_UE_MAX"),
            ("RB_IN_FIFO_DEPTH", "EQ_IN_FIFO_DEPTH"),
        ],
        save_dir=vis_dir,
    )

    print(f"增强版可视化图表已保存到: {vis_dir}")


def create_single_parameter_line_plots(trials, metric_key="Total_sum_BW_weighted_mean", save_dir="."):
    """为每个参数绘制一维性能曲线"""
    import pandas as pd

    if not trials:
        return

    rows = []
    for t in trials:
        if t.state.name != "COMPLETE" or t.values is None:
            continue
        row = {k: v for k, v in t.params.items()}
        row[metric_key] = t.user_attrs.get(metric_key, t.values[0])
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return

    os.makedirs(save_dir, exist_ok=True)

    for param in [c for c in df.columns if c != metric_key]:
        agg = df.groupby(param)[metric_key].mean().reset_index().sort_values(param)
        fig = go.Figure(data=go.Scatter(x=agg[param], y=agg[metric_key], mode="lines+markers"))
        fig.update_layout(title=f"{param} 对性能影响", xaxis_title=param, yaxis_title=metric_key, height=400)
        fig.write_html(os.path.join(save_dir, f"{param}_line.html"))


def create_2d_param_bw_heatmaps(trials, metric_key="Total_sum_BW_weighted_mean", param_pairs=None, aggfunc="mean", save_dir="."):
    """
    绘制所有指定参数对的 2D 热力图
    ─────────────────────────────────────────────
    • trials        : List[optuna.Trial]，只传 COMPLETE 的就行
    • metric_key    : 作为色值的指标；默认用加权整体带宽
    • param_pairs   : [('paramA','paramB'), ...]；为空时自动取所有两两组合
    • aggfunc       : 'mean' | 'max' | 'min' 等，透视表聚合方式
    • save_dir      : html/png 输出目录
    """
    import plotly.graph_objects as go
    import pandas as pd, numpy as np, os, itertools, seaborn as sns, matplotlib.pyplot as plt

    # ------- 整理 DataFrame -------
    rows = []
    for t in trials:
        if t.state.name != "COMPLETE" or t.values is None:
            continue
        row = {k: v for k, v in t.params.items()}
        row[metric_key] = t.user_attrs.get(metric_key, t.values[0])
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print("没有可用 Trial 数据，跳过 2D 热力图绘制")
        return

    # ------- 自动组合参数对 -------
    if not param_pairs:
        cols = list(trials[0].params.keys())
        param_pairs = list(itertools.combinations(cols, 2))

    os.makedirs(save_dir, exist_ok=True)

    # ------- 逐对绘制 -------
    for x, y in param_pairs:
        pivot = df.pivot_table(index=y, columns=x, values=metric_key, aggfunc=aggfunc)
        title = f"{y} vs {x} — {metric_key}"

        # ——— 方法 A：Plotly（交互式，默认输出 HTML） ———
        fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Viridis", colorbar=dict(title=metric_key)))
        fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=600, width=750)
        fig.write_html(os.path.join(save_dir, f"{y}_vs_{x}.html"))

        # ——— 方法 B：Seaborn（静态 PNG，可选） ———
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".1f", linewidths=0.5, linecolor="white")
        ax.invert_yaxis()
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{y}_vs_{x}.png"), dpi=200)
        plt.close()


def create_optimization_history(trials, save_dir):
    """创建优化历史图"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("优化历史", "最佳值历史", "Trial状态分布", "目标函数分布"),
        specs=[[{"secondary_y": True}, {"secondary_y": False}], [{"type": "pie"}, {"type": "histogram"}]],
    )

    # 优化历史（过滤掉values为None的trial）
    valid_trials = [t for t in trials if t.values is not None]
    trial_numbers = [t.number for t in valid_trials]
    objective_values = [t.values[0] for t in valid_trials]
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
    states = [t.state.name if hasattr(t.state, "name") else str(t.state) for t in trials]
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
            importance = optuna.importance.get_param_importances(study, target=lambda t: t.values[0] if t.values else 0)
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

        fig.update_layout(title=f"Pareto前沿: {traffic1_name} vs {traffic2_name}", xaxis_title=f"{traffic1_name} 带宽 (GB/s)", yaxis_title=f"{traffic2_name} 带宽 (GB/s)", height=600)

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

    # traffic_file_path = r"../test_data/"
    traffic_file_path = r"../traffic/0617/"

    # ===== 多个traffic文件配置 =====
    traffic_files = [
        r"LLama2_AllReduce.txt",
        r"LLama2_AttentionFC.txt",
        r"MLP_MoE.txt",
        r"MLP.txt",
    ]

    # 每个traffic的权重（用于加权平均）
    traffic_weights = [0.4, 0.2, 0.2, 0.2]  # 第一个traffic权重0.6，第二个0.4
    # traffic_weights = [1]  # 第一个traffic权重0.6，第二个0.4

    assert len(traffic_files) == len(traffic_weights), "traffic文件数量和权重数量必须一致"
    assert abs(sum(traffic_weights) - 1.0) < 1e-6, "权重总和必须等于1"

    config_path = r"../config/topologies/topo_5x4.yaml"
    config = CrossRingConfig(config_path)

    # topo_type = "3x3"
    topo_type = "5x4"
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
    param4_start, param4_end = 4, 20
    param5_start, param5_end = 2, 16
    param6_start, param6_end = 2, 16
    param7_start, param7_end = 2, 16
    param8_start, param8_end = 4, 20
    param9_start, param9_end = 0, 1

    def _run_one_traffic(traffic_file, param1, param2, param3, param4, param5, param6, param7, param8, param9):
        """运行单个traffic文件的仿真"""
        tot_bw_list = []
        for rpt in range(N_REPEATS):
            cfg = CrossRingConfig(config_path)
            cfg.TOPO_TYPE = topo_type

            sim = REQ_RSP_model(
                model_type=model_type,
                config=cfg,
                topo_type=topo_type,
                verbose=0,
            )

            sim.setup_traffic_scheduler(
                traffic_file_path=traffic_file_path,
                traffic_chains=traffic_file,
            )

            sim.setup_result_analysis(
                result_save_path=result_root_save_path,
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
                sim.config.IQ_CH_FIFO_DEPTH = 8
                sim.config.EQ_CH_FIFO_DEPTH = 8
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
                sim.config.IQ_CH_FIFO_DEPTH = 10
                sim.config.EQ_CH_FIFO_DEPTH = 10
                sim.config.IQ_OUT_FIFO_DEPTH_HORIZONTAL = 8
                sim.config.IQ_OUT_FIFO_DEPTH_VERTICAL = 8
                sim.config.IQ_OUT_FIFO_DEPTH_EQ = 8
                sim.config.RB_OUT_FIFO_DEPTH = 8
                sim.config.SN_TRACKER_RELEASE_LATENCY = 40

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
                sim.config.SLICE_PER_LINK_HORIZONTAL = 8
                sim.config.SLICE_PER_LINK_VERTICAL = 8

                sim.config.GDMA_RW_GAP = np.inf
                sim.config.SDMA_RW_GAP = np.inf
                sim.config.CHANNEL_SPEC = {
                    "gdma": 2,
                    "sdma": 2,
                    "ddr": 2,
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
            sim.config.ETag_BOTHSIDE_UPGRADE = param9

            try:
                sim.run_simulation(max_time=10000, print_interval=10000)
                bw = sim.get_results().get("Total_sum_BW", 0)
            except Exception as e:
                print(f"[{traffic_file}][RPT {rpt}] Sim failed for params: {param1}, {param2}, {param3}, {param4}, {param5}, {param6}, {param7}, {param8}, {param9}")
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

    def _run_one(param1, param2, param3, param4, param5, param6, param7, param8, param9):
        """运行所有traffic文件并综合结果"""
        all_results = {}
        all_bw_means = []

        for traffic_file in traffic_files:
            try:
                result = _run_one_traffic(traffic_file, param1, param2, param3, param4, param5, param6, param7, param8, param9)
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
                "param9": param9,
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
        p9 = trial.suggest_int("ETag_BOTHSIDE_UPGRADE", param9_start, param9_end)

        results = _run_one(p1, p2, p3, p4, p5, p6, p7, p8, p9)

        # ─── 两个 traffic 的带宽均值 ──────────────────────────
        weighted_bw = results["Total_sum_BW_weighted_mean"]

        # ─── 参数规模归一化（越小越好）───────────────────────
        param_penalty = (
            # (p1 - param1_start) / (param1_end - param1_start)
            # + (p2 - param2_start) / (param2_end - param2_start)
            # + (p3 - param3_start) / (param3_end - param3_start)
            +(p4 - param4_start) / (param4_end - param4_start)
            # + (p5 - param5_start) / (param5_end - param5_start)
            # + (p6 - param6_start) / (param6_end - param6_start)
            + (p8 - param8_start) / (param8_end - param8_start)
        ) / 2.0

        # 综合指标 = 加权带宽 - α * 参数惩罚
        # 调整α值平衡性能和资源消耗
        # composite_metric = weighted_bw - 50 * param_penalty
        # α 根据试验进度自适应，前期约束更强，后期逐渐减弱
        progress = min(trial.number / N_TRIALS, 1.0)
        penalty_weight = 50 * (1 - progress)
        composite_metric = weighted_bw - penalty_weight * param_penalty

        # 保存到 trial.user_attrs，便于后期分析 / CSV
        for k, v in results.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("param_penalty", param_penalty)
        trial.set_user_attr("penalty_weight", penalty_weight)
        trial.set_user_attr("composite_metric", composite_metric)

        # ─── 多目标返回： (maximize, maximize, minimize) ────
        return composite_metric

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
    pd.DataFrame(records).to_csv(output_csv, index=False, encoding='utf-8-sig')

    # 保存进度并创建实时可视化
    save_progress_callback(study, trial)


def create_summary_report(study, traffic_files, traffic_weights, save_dir):
    """创建HTML总结报告"""
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if not complete_trials:
        return

    # 获取最佳试验
    best_trial = study.best_trials[0]
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
        html_content += f"<li>{param}: {int(value)}</li>"

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

    n_trials = N_TRIALS

    study = optuna.create_study(
        study_name="CrossRing_Single_Traffic_BO",
        direction="maximize",
        sampler=optuna.samplers.NSGAIISampler(),
    )

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=N_JOBS,
            show_progress_bar=True,
            callbacks=[save_intermediate_result],
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
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

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
        enhanced_create_visualization_plots(study, traffic_files, traffic_weights, result_root_save_path)
        print(f"可视化报告已生成: {result_root_save_path}/visualizations/")

        # 创建总结报告
        create_summary_report(study, traffic_files, traffic_weights, result_root_save_path)

    except Exception as e:
        print(f"生成可视化失败: {e}")
        traceback.print_exc()

    print("=" * 60)

    # 1. 保存Study对象
    study_file = os.path.join(result_root_save_path, "optuna_study.pkl")
    import joblib

    joblib.dump(study, study_file)
    print(f"Study对象已保存: {study_file}")

    # 2. 保存优化配置
    config_data = {
        "traffic_files": traffic_files,
        "traffic_weights": traffic_weights,
        "param_ranges": {
            "TL_Etag_T2_UE_MAX": [2, 16],
            "TL_Etag_T1_UE_MAX": [2, 16],
            "TR_Etag_T2_UE_MAX": [2, 16],
            "RB_IN_FIFO_DEPTH": [4, 20],
            "TU_Etag_T2_UE_MAX": [2, 16],
            "TU_Etag_T1_UE_MAX": [2, 16],
            "TD_Etag_T2_UE_MAX": [2, 16],
            "EQ_IN_FIFO_DEPTH": [4, 20],
            "ETag_BOTHSIDE_UPGRADE": [0, 1],
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

    print(f"\n🔄 重新生成分析请运行:")
    print(f"python regenerate_analysis.py ../{result_root_save_path}")
