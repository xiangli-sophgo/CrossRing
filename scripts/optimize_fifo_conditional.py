"""
FIFO深度参数条件优化脚本

核心思路：
1. 分层条件优化：固定一个参数，优化其他参数，得到该参数的"瓶颈曲线"
2. 贝叶斯优化：使用Optuna进行智能搜索
3. 真实影响分析：展示每个参数作为瓶颈时的性能限制

优化策略：
- Phase 1: 快速全局探索，识别高影响参数
- Phase 2: 条件优化，生成单参数瓶颈曲线
- Phase 3: 联合精调，寻找全局最优
- Phase 4: 结果验证和可视化
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import *
from config.config import CrossRingConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import optuna
from joblib import Parallel, delayed
from tqdm import tqdm
import traceback
from typing import Dict, List, Tuple, Any, Optional
import joblib
from scipy import stats

# ==================== 全局配置 ====================
N_REPEATS = 1  # 每个配置的重复次数（无随机性，只需1次）
N_JOBS = 8  # 并行作业数
SIMULATION_TIME = 5000  # 仿真时间
OPTUNA_VERBOSITY = 1  # Optuna日志级别

# FIFO参数配置（统一范围2-16）
FIFO_PARAMS = {
    "IQ_CH_FIFO_DEPTH": {"range": [2, 16], "default": 4},
    "EQ_CH_FIFO_DEPTH": {"range": [2, 16], "default": 4},
    "IQ_OUT_FIFO_DEPTH_HORIZONTAL": {"range": [2, 16], "default": 8},
    "IQ_OUT_FIFO_DEPTH_VERTICAL": {"range": [2, 16], "default": 8},
    "IQ_OUT_FIFO_DEPTH_EQ": {"range": [2, 16], "default": 8},
    "RB_OUT_FIFO_DEPTH": {"range": [2, 16], "default": 8},
    "RB_IN_FIFO_DEPTH": {"range": [2, 16], "default": 16},
    "EQ_IN_FIFO_DEPTH": {"range": [2, 16], "default": 16},
}

# ETag参数配置（也需要优化，与FIFO参数有依赖关系）
# 由于FIFO参数最大为16，ETag参数必须小于对应的FIFO参数
ETAG_PARAMS = {
    "TL_Etag_T2_UE_MAX": {"range": [1, 15], "default": 8, "related_fifo": "RB_IN_FIFO_DEPTH"},
    "TL_Etag_T1_UE_MAX": {"range": [2, 15], "default": 15, "related_fifo": "RB_IN_FIFO_DEPTH", "constraint": "must_be_greater_than_T2"},
    "TR_Etag_T2_UE_MAX": {"range": [1, 15], "default": 12, "related_fifo": "RB_IN_FIFO_DEPTH"},
    "TU_Etag_T2_UE_MAX": {"range": [1, 15], "default": 8, "related_fifo": "EQ_IN_FIFO_DEPTH"},
    "TU_Etag_T1_UE_MAX": {"range": [2, 15], "default": 15, "related_fifo": "EQ_IN_FIFO_DEPTH", "constraint": "must_be_greater_than_T2"},
    "TD_Etag_T2_UE_MAX": {"range": [1, 15], "default": 12, "related_fifo": "EQ_IN_FIFO_DEPTH"},
}

# 合并所有需要优化的参数
ALL_PARAMS = {**FIFO_PARAMS, **ETAG_PARAMS}


class FIFOConditionalOptimizer:
    """FIFO条件优化器"""

    def __init__(self, config_path: str, topo_type: str, traffic_files: List[str], traffic_weights: List[float], traffic_path: str = "../traffic/0617/"):
        self.config_path = config_path
        self.topo_type = topo_type
        self.traffic_files = traffic_files
        self.traffic_weights = traffic_weights
        self.traffic_path = traffic_path

        # 创建结果目录
        timestamp = datetime.now().strftime("%m%d_%H%M")
        self.result_dir = f"../Result/FIFO_Conditional_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)

        # 结果存储
        self.cache = {}  # 仿真结果缓存
        self.global_results = []  # 全局探索结果
        self.conditional_results = {}  # 条件优化结果

    def _get_cache_key(self, params: Dict) -> str:
        """生成参数配置的缓存key"""
        # 转换numpy int64为普通int，确保JSON序列化兼容
        clean_params = {}
        for key, value in params.items():
            if hasattr(value, 'item'):  # numpy类型
                clean_params[key] = int(value.item())
            else:
                clean_params[key] = int(value)
        return json.dumps(clean_params, sort_keys=True)

    def _set_platform_config(self, sim):
        """设置平台相关配置"""
        if self.topo_type == "5x4":
            # 5x4拓扑的固定参数
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
            sim.config.DDR_R_LATENCY_original = 40
            sim.config.L2M_R_LATENCY_original = 12
            sim.config.L2M_W_LATENCY_original = 16
            sim.config.GDMA_RW_GAP = np.inf
            sim.config.SDMA_RW_GAP = np.inf
            sim.config.CHANNEL_SPEC = {"gdma": 2, "sdma": 2, "ddr": 2, "l2m": 2}

            # 使用之前优化得到的最佳ETag参数
            sim.config.TL_Etag_T2_UE_MAX = 8
            sim.config.TL_Etag_T1_UE_MAX = 15
            sim.config.TR_Etag_T2_UE_MAX = 12
            sim.config.TU_Etag_T2_UE_MAX = 8
            sim.config.TU_Etag_T1_UE_MAX = 15
            sim.config.TD_Etag_T2_UE_MAX = 12

    def run_simulation(self, fifo_params: Dict) -> float:
        """
        运行仿真并返回加权性能

        Args:
            fifo_params: FIFO参数字典

        Returns:
            加权平均带宽
        """
        # 检查缓存
        cache_key = self._get_cache_key(fifo_params)
        if cache_key in self.cache:
            return self.cache[cache_key]

        total_weighted_bw = 0

        for traffic_file, weight in zip(self.traffic_files, self.traffic_weights):
            bw_list = []

            for repeat in range(N_REPEATS):
                try:
                    # 创建仿真实例
                    cfg = CrossRingConfig(self.config_path)
                    cfg.TOPO_TYPE = self.topo_type

                    sim = REQ_RSP_model(
                        model_type="REQ_RSP",
                        config=cfg,
                        topo_type=self.topo_type,
                        traffic_file_path=self.traffic_path,
                        traffic_config=traffic_file,
                        result_save_path=self.result_dir,
                        verbose=0,
                    )

                    # 设置平台参数
                    self._set_platform_config(sim)

                    # 设置FIFO参数
                    for param_name, param_value in fifo_params.items():
                        setattr(sim.config, param_name, param_value)

                    # 运行仿真
                    sim.initial()
                    sim.end_time = SIMULATION_TIME
                    sim.print_interval = SIMULATION_TIME  # 减少输出
                    sim.run()

                    # 获取结果
                    bw = sim.get_results().get("mixed_avg_weighted_bw", 0)
                    bw_list.append(bw)

                except Exception as e:
                    print(f"仿真失败: {e}")
                    bw_list.append(0)

            # 计算该traffic的平均带宽
            avg_bw = np.mean(bw_list)
            total_weighted_bw += avg_bw * weight

        # 缓存结果
        self.cache[cache_key] = total_weighted_bw
        return total_weighted_bw

    def global_exploration(self, n_trials: int = 200) -> Dict:
        """
        Phase 1: 全局探索，识别参数重要性

        Args:
            n_trials: 探索试验次数

        Returns:
            探索结果统计
        """
        print("\n" + "=" * 60)
        print(f"[Phase 1] 全局探索 - {n_trials} trials")
        print("=" * 60)

        def objective(trial):
            # 采样所有参数（FIFO + ETag）
            all_params = {}

            # 先采样FIFO参数
            for param_name, param_info in FIFO_PARAMS.items():
                all_params[param_name] = trial.suggest_int(param_name, param_info["range"][0], param_info["range"][1])

            # 采样ETag参数（处理复杂的依赖约束）
            # 1. 先采样T2参数（调整为新的范围）
            all_params["TL_Etag_T2_UE_MAX"] = trial.suggest_int("TL_Etag_T2_UE_MAX", 2, 15)
            all_params["TU_Etag_T2_UE_MAX"] = trial.suggest_int("TU_Etag_T2_UE_MAX", 2, 15)
            all_params["TR_Etag_T2_UE_MAX"] = trial.suggest_int("TR_Etag_T2_UE_MAX", 2, 15)
            all_params["TD_Etag_T2_UE_MAX"] = trial.suggest_int("TD_Etag_T2_UE_MAX", 2, 15)

            # 2. 采样T1参数（必须大于对应的T2参数）
            tl_t1_min = all_params["TL_Etag_T2_UE_MAX"] + 1
            if tl_t1_min <= 15:
                all_params["TL_Etag_T1_UE_MAX"] = trial.suggest_int("TL_Etag_T1_UE_MAX", tl_t1_min, 15)
            else:
                # 如果约束无法满足，跳过这个trial
                from optuna.exceptions import TrialPruned

                raise TrialPruned()

            tu_t1_min = all_params["TU_Etag_T2_UE_MAX"] + 1
            if tu_t1_min <= 15:
                all_params["TU_Etag_T1_UE_MAX"] = trial.suggest_int("TU_Etag_T1_UE_MAX", tu_t1_min, 15)
            else:
                # 如果约束无法满足，跳过这个trial
                from optuna.exceptions import TrialPruned

                raise TrialPruned()

            # 3. 更新FIFO参数约束（FIFO必须大于相关ETag参数的最大值）
            # RB_IN_FIFO_DEPTH > max(TL_Etag_T1_UE_MAX, TR_Etag_T2_UE_MAX)
            rb_min = max(all_params["TL_Etag_T1_UE_MAX"], all_params["TR_Etag_T2_UE_MAX"]) + 1
            if "RB_IN_FIFO_DEPTH" in all_params and all_params["RB_IN_FIFO_DEPTH"] < rb_min:
                if rb_min <= 16:
                    all_params["RB_IN_FIFO_DEPTH"] = max(all_params["RB_IN_FIFO_DEPTH"], rb_min)
                else:
                    # 如果约束无法满足，跳过这个trial
                    from optuna.exceptions import TrialPruned

                    raise TrialPruned()

            # EQ_IN_FIFO_DEPTH > max(TU_Etag_T1_UE_MAX, TD_Etag_T2_UE_MAX)
            eq_min = max(all_params["TU_Etag_T1_UE_MAX"], all_params["TD_Etag_T2_UE_MAX"]) + 1
            if "EQ_IN_FIFO_DEPTH" in all_params and all_params["EQ_IN_FIFO_DEPTH"] < eq_min:
                if eq_min <= 16:
                    all_params["EQ_IN_FIFO_DEPTH"] = max(all_params["EQ_IN_FIFO_DEPTH"], eq_min)
                else:
                    # 如果约束无法满足，跳过这个trial
                    from optuna.exceptions import TrialPruned

                    raise TrialPruned()

            # 运行仿真
            performance = self.run_simulation(all_params)

            # 保存详细结果
            trial.set_user_attr("performance", performance)
            for param, value in all_params.items():
                trial.set_user_attr(param, value)

            return performance

        # 创建Study
        study = optuna.create_study(study_name="FIFO_Global_Exploration", direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        # 设置Optuna日志级别
        optuna.logging.set_verbosity(OPTUNA_VERBOSITY)

        # 运行优化
        study.optimize(objective, n_trials=n_trials, n_jobs=N_JOBS, show_progress_bar=True)

        # 保存全局结果
        self.global_results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = {"performance": trial.value}
                result.update(trial.params)
                self.global_results.append(result)

        # 分析参数重要性
        param_importance = self._analyze_parameter_importance()

        print(f"\n全局探索完成:")
        print(f"  最佳性能: {study.best_value:.2f} GB/s")
        print(f"  完成试验: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

        return {"study": study, "param_importance": param_importance, "best_params": study.best_params, "best_performance": study.best_value}

    def conditional_optimization(self, target_param: str, sample_points: int = 8, n_trials_per_point: int = 50) -> List[Dict]:
        """
        Phase 2: 条件优化 - 固定目标参数，优化其他参数

        Args:
            target_param: 目标参数名
            sample_points: 目标参数的采样点数
            n_trials_per_point: 每个采样点的优化试验次数

        Returns:
            条件优化结果列表
        """
        print(f"\n[Phase 2] 条件优化参数: {target_param}")

        # 获取目标参数信息
        if target_param in FIFO_PARAMS:
            param_info = FIFO_PARAMS[target_param]
        elif target_param in ETAG_PARAMS:
            param_info = ETAG_PARAMS[target_param]
        else:
            raise ValueError(f"Unknown parameter: {target_param}")

        param_range = param_info["range"]

        # 生成采样点
        sample_values = np.linspace(param_range[0], param_range[1], sample_points, dtype=int)

        results = []

        for i, target_value in enumerate(sample_values):
            print(f"  采样点 {i+1}/{len(sample_values)}: {target_param} = {target_value}")

            def objective(trial):
                # 固定目标参数
                all_params = {target_param: target_value}

                # 优化其他FIFO参数
                for param_name, param_info_inner in FIFO_PARAMS.items():
                    if param_name != target_param:
                        all_params[param_name] = trial.suggest_int(param_name, param_info_inner["range"][0], param_info_inner["range"][1])

                # 优化其他ETag参数（处理约束）
                for param_name, param_info_inner in ETAG_PARAMS.items():
                    if param_name != target_param:
                        if "constraint" in param_info_inner and param_info_inner["constraint"] == "must_be_greater_than_T2":
                            # T1参数必须大于对应的T2参数
                            if "TL_Etag_T1" in param_name:
                                t2_value = all_params.get("TL_Etag_T2_UE_MAX", param_info_inner["range"][0])
                                min_value = max(param_info_inner["range"][0], t2_value + 1)
                            elif "TU_Etag_T1" in param_name:
                                t2_value = all_params.get("TU_Etag_T2_UE_MAX", param_info_inner["range"][0])
                                min_value = max(param_info_inner["range"][0], t2_value + 1)
                            else:
                                min_value = param_info_inner["range"][0]

                            if min_value <= param_info_inner["range"][1]:
                                all_params[param_name] = trial.suggest_int(param_name, min_value, param_info_inner["range"][1])
                            else:
                                all_params[param_name] = param_info_inner["default"]
                        else:
                            all_params[param_name] = trial.suggest_int(param_name, param_info_inner["range"][0], param_info_inner["range"][1])

                # 运行仿真
                performance = self.run_simulation(all_params)

                # 保存结果
                for param, value in all_params.items():
                    trial.set_user_attr(param, value)

                return performance

            # 创建条件优化Study
            study = optuna.create_study(study_name=f"Conditional_{target_param}_{target_value}", direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

            # 运行优化
            study.optimize(objective, n_trials=n_trials_per_point, n_jobs=N_JOBS, show_progress_bar=False)

            # 记录结果
            result = {
                "target_param": target_param,
                "target_value": int(target_value),
                "best_performance": study.best_value if study.best_value is not None else 0,
                "best_other_params": study.best_params.copy() if study.best_params else {},
                "n_trials": len(study.trials),
                "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            }

            # 移除目标参数（因为它是固定的）
            if target_param in result["best_other_params"]:
                del result["best_other_params"][target_param]

            results.append(result)

            print(f"    最佳性能: {result['best_performance']:.2f} GB/s")

        return results

    def run_all_conditional_optimizations(self, high_priority_only: bool = True):
        """运行所有参数的条件优化"""
        print("\n" + "=" * 60)
        print("[Phase 2] 条件优化所有参数")
        print("=" * 60)

        # 确定要优化的参数（选择所有FIFO参数和部分ETag参数）
        if high_priority_only:
            # 只优化FIFO参数，ETag参数依赖关系太复杂
            target_params = list(FIFO_PARAMS.keys())
        else:
            target_params = list(ALL_PARAMS.keys())

        for param_name in target_params:
            print(f"\n正在优化参数: {param_name}")
            results = self.conditional_optimization(param_name)
            self.conditional_results[param_name] = results

    def _analyze_parameter_importance(self) -> Dict:
        """分析参数重要性"""
        if not self.global_results:
            return {}

        df = pd.DataFrame(self.global_results)
        importance = {}

        for param in ALL_PARAMS.keys():
            if param in df.columns:
                # 使用相关系数作为重要性指标
                correlation = df[param].corr(df["performance"])
                # 使用方差分析
                param_values = df[param].values
                performance_values = df["performance"].values

                # 计算不同参数值对应的性能方差
                unique_values = np.unique(param_values)
                if len(unique_values) > 1:
                    group_variances = []
                    for val in unique_values:
                        mask = param_values == val
                        if np.sum(mask) > 1:
                            group_var = np.var(performance_values[mask])
                            group_variances.append(group_var)

                    avg_variance = np.mean(group_variances) if group_variances else 0
                else:
                    avg_variance = 0

                importance[param] = {
                    "correlation": abs(correlation) if not np.isnan(correlation) else 0,
                    "variance_impact": avg_variance,
                    "combined_score": abs(correlation) * avg_variance if not np.isnan(correlation) else 0,
                }

        return importance

    def visualize_conditional_results(self):
        """生成条件优化结果的可视化"""
        if not self.conditional_results:
            print("没有条件优化结果可视化")
            return

        print("\n生成条件优化可视化...")

        # 创建可视化目录
        vis_dir = os.path.join(self.result_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 1. 单参数瓶颈曲线图
        n_params = len(self.conditional_results)
        cols = 3
        rows = (n_params + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if n_params == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (param_name, results) in enumerate(self.conditional_results.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # 提取数据
            x_values = [r["target_value"] for r in results]
            y_values = [r["best_performance"] for r in results]

            # 绘制曲线
            ax.plot(x_values, y_values, "b-o", linewidth=2, markersize=6)

            # 标记最优点
            if y_values:
                best_idx = np.argmax(y_values)
                ax.plot(x_values[best_idx], y_values[best_idx], "r*", markersize=12, label=f"最优: {x_values[best_idx]}")

                # 计算性能变化范围
                perf_range = max(y_values) - min(y_values)
                perf_change_pct = (perf_range / min(y_values)) * 100 if min(y_values) > 0 else 0

                ax.text(0.05, 0.95, f"变化: {perf_change_pct:.1f}%", transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

            ax.set_xlabel(param_name)
            ax.set_ylabel("最优性能 (GB/s)")
            ax.set_title(f"{param_name} 瓶颈分析")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 隐藏多余的子图
        for idx in range(len(self.conditional_results), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "conditional_optimization_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 2. 参数影响力对比图
        param_impacts = []
        param_names = []

        for param_name, results in self.conditional_results.items():
            performances = [r["best_performance"] for r in results]
            if performances:
                impact = (max(performances) - min(performances)) / min(performances) * 100
                param_impacts.append(impact)
                param_names.append(param_name)

        if param_impacts:
            # 按影响力排序
            sorted_data = sorted(zip(param_names, param_impacts), key=lambda x: x[1], reverse=True)
            param_names, param_impacts = zip(*sorted_data)

            plt.figure(figsize=(12, 6))
            colors = ["red" if impact > 5 else "orange" if impact > 2 else "gray" for impact in param_impacts]

            bars = plt.barh(range(len(param_names)), param_impacts, color=colors)
            plt.yticks(range(len(param_names)), param_names)
            plt.xlabel("性能影响 (%)")
            plt.title("FIFO参数瓶颈影响力排序")
            plt.grid(True, alpha=0.3, axis="x")

            # 添加数值标注
            for bar, impact in zip(bars, param_impacts):
                plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"{impact:.1f}%", ha="left", va="center")

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "parameter_bottleneck_ranking.png"), dpi=150, bbox_inches="tight")
            plt.close()

        print(f"可视化结果保存至: {vis_dir}")

    def save_results(self):
        """保存所有结果"""
        # 1. 保存条件优化结果
        results_path = os.path.join(self.result_dir, "conditional_optimization_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.conditional_results, f, indent=2, ensure_ascii=False)

        # 2. 保存全局探索结果
        if self.global_results:
            global_df = pd.DataFrame(self.global_results)
            global_df.to_csv(os.path.join(self.result_dir, "global_exploration.csv"), index=False)

        # 3. 生成摘要报告
        self._generate_summary_report()

        print(f"\n结果已保存至: {self.result_dir}")

    def _generate_summary_report(self):
        """生成摘要报告"""
        report = []
        report.append("=" * 60)
        report.append("FIFO深度条件优化报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)

        # 全局探索摘要
        if self.global_results:
            global_df = pd.DataFrame(self.global_results)
            best_global = global_df.loc[global_df["performance"].idxmax()]

            report.append(f"\n## 全局探索结果")
            report.append(f"- 试验次数: {len(self.global_results)}")
            report.append(f"- 最佳性能: {best_global['performance']:.2f} GB/s")
            report.append(f"- 最佳配置:")
            # 显示所有参数（FIFO + ETag）
            for param in ALL_PARAMS.keys():
                if param in best_global:
                    report.append(f"  - {param}: {int(best_global[param])}")

        # 条件优化摘要
        if self.conditional_results:
            report.append(f"\n## 条件优化结果")

            # 计算每个参数的影响力
            param_impacts = {}
            for param_name, results in self.conditional_results.items():
                performances = [r["best_performance"] for r in results]
                if performances and min(performances) > 0:
                    impact = (max(performances) - min(performances)) / min(performances) * 100
                    optimal_value = results[np.argmax(performances)]["target_value"]
                    param_impacts[param_name] = {"impact_pct": impact, "optimal_value": optimal_value, "max_performance": max(performances)}

            # 按影响力排序
            sorted_params = sorted(param_impacts.items(), key=lambda x: x[1]["impact_pct"], reverse=True)

            report.append(f"\n### 参数影响力排序:")
            for rank, (param, info) in enumerate(sorted_params, 1):
                report.append(f"{rank}. {param}:")
                report.append(f"   - 影响力: {info['impact_pct']:.1f}%")
                report.append(f"   - 最优值: {info['optimal_value']}")
                report.append(f"   - 最佳性能: {info['max_performance']:.2f} GB/s")

        # 保存报告
        report_path = os.path.join(self.result_dir, "optimization_summary.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

        print(f"摘要报告: {report_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("FIFO深度条件优化")
    print("=" * 60)

    # 配置
    config_path = "../config/config2.json"
    topo_type = "5x4"
    traffic_files = ["LLama2_AllReduce.txt"]
    traffic_weights = [1.0]

    # 创建优化器
    optimizer = FIFOConditionalOptimizer(config_path=config_path, topo_type=topo_type, traffic_files=traffic_files, traffic_weights=traffic_weights)

    print(f"配置信息:")
    print(f"  拓扑: {topo_type}")
    print(f"  Traffic: {traffic_files}")
    print(f"  结果目录: {optimizer.result_dir}")

    try:
        # Phase 1: 全局探索
        global_results = optimizer.global_exploration(n_trials=100)  # 减少试验次数用于测试

        # Phase 2: 条件优化
        optimizer.run_all_conditional_optimizations(high_priority_only=True)

        # Phase 3: 可视化和保存
        optimizer.visualize_conditional_results()
        optimizer.save_results()

        print("\n" + "=" * 60)
        print("优化完成！")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n用户中断优化")
        optimizer.save_results()
    except Exception as e:
        print(f"\n优化过程出错: {e}")
        traceback.print_exc()
        optimizer.save_results()


if __name__ == "__main__":
    main()
