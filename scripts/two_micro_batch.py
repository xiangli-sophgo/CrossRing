#!/usr/bin/env python3
"""
CDMA带宽并行行为分析脚本 - 修复版本

遍历不同的CDMA带宽设置，分析与上层计算模块的并行性能表现。
所有结果保存到CSV文件供后续分析。

主要修复：
1. 使用正确的性能指标进行分析
2. 添加内存清理和资源管理
3. 优化进程管理避免内存溢出

Usage:
    python scripts/two_micro_batch.py \
        --config ../config/topologies/topo_5x4.yaml \
        --traffic_path ../traffic/0617/ \
        --output_dir ../Result/cdma_analysis \
        --bandwidths 4 8 12 16 20 24 28 32 \
        --repeat 1 --topo 5x4 --max_workers 2
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import traceback
from pathlib import Path
import time
from datetime import datetime
import csv
import logging
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc  # 添加垃圾回收
import psutil  # 添加内存监控

# 添加项目路径
sys.path.append("../")
sys.path.append("../src")
sys.path.append("../config")

from config.config import CrossRingConfig
from src.noc import REQ_RSP_model


class CDMABandwidthAnalyzer:
    def __init__(self, config_path="../config/topologies/topo_5x4.yaml", traffic_file_path="../traffic/0617/", output_dir="../Result/cdma_analysis/", cdma_bw_ranges=[4, 16, 32], run_timestamp=None):
        """
        初始化CDMA带宽分析器

        Args:
            config_path: 配置文件路径
            traffic_file_path: traffic文件路径
            output_dir: 结果输出目录
        """
        self.config_path = config_path
        self.traffic_file_path = traffic_file_path
        self.run_timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CDMA带宽范围 (GB/s)：单通道1-32，4通道总带宽4-128
        self.cdma_bw_ranges = cdma_bw_ranges

        # 设置CSV输出文件
        self.output_csv = self.output_dir / f"cdma_bandwidth_analysis_{self.run_timestamp}.csv"
        self.csv_initialized = False

        print(f"CDMA带宽分析器初始化完成")
        print(f"配置文件: {config_path}")
        print(f"Traffic路径: {traffic_file_path}")
        print(f"输出目录: {output_dir}")
        print(f"结果CSV: {self.output_csv}")
        print(f"CDMA带宽测试范围: {self.cdma_bw_ranges} GB/s (单通道)")

    def save_result_to_csv(self, result_data):
        """保存单次仿真结果到CSV文件，使用文件锁保证并发安全"""

        if not result_data:
            return

        csv_file_exists = self.output_csv.exists()

        try:
            import portalocker

            use_portalocker = True
        except ImportError:
            print("Warning: portalocker not available, using threading.Lock instead")
            use_portalocker = False

        if use_portalocker:
            with open(self.output_csv, mode="a", newline="", encoding="utf-8") as output_csv_file:
                portalocker.lock(output_csv_file, portalocker.LOCK_EX)
                writer = csv.DictWriter(output_csv_file, fieldnames=result_data.keys())
                if not csv_file_exists:
                    writer.writeheader()
                writer.writerow(result_data)
        else:
            import threading

            if not hasattr(self, "_lock"):
                self._lock = threading.Lock()
            with self._lock:
                with open(self.output_csv, mode="a", newline="", encoding="utf-8") as output_csv_file:
                    writer = csv.DictWriter(output_csv_file, fieldnames=result_data.keys())
                    if not csv_file_exists:
                        writer.writeheader()
                    writer.writerow(result_data)

        print(f"结果已保存到CSV: CDMA带宽={result_data.get('cdma_bw_limit', 'N/A')}GB/s")

    def monitor_memory_usage(self):
        """监控当前进程及其所有子进程的总内存使用情况"""
        parent = psutil.Process()
        total_memory = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total_memory += child.memory_info().rss
            except psutil.NoSuchProcess:
                # Child process might have terminated
                continue
        memory_mb = total_memory / 1024 / 1024
        return memory_mb

    def run_bandwidth_sweep(self, traffic_files, topo_type="5x4", repeat=1, max_workers=None):
        """
        运行带宽扫描分析 - 优化内存管理

        Args:
            traffic_files: traffic文件列表
            topo_type: 拓扑类型
            repeat: 每个配置重复次数
        """
        print(f"\n开始CDMA带宽扫描分析...")
        print(f"拓扑类型: {topo_type}")
        print(f"重复次数: {repeat}")
        print(f"Traffic文件: {traffic_files}")

        total_runs = len(self.cdma_bw_ranges) * repeat
        runs = [(bw, rep) for rep in range(repeat) for bw in self.cdma_bw_ranges]

        # 更保守的并行设置
        if max_workers is None:
            max_workers = min(2, multiprocessing.cpu_count() // 2)  # 更保守的设置
        max_workers = max(1, max_workers)  # 至少1个worker

        print(f"使用 {max_workers} 个并行进程")

        start_time = time.time()
        completed = 0

        # 分批处理，避免一次性创建太多进程
        batch_size = max_workers * 2

        for batch_start in range(0, len(runs), batch_size):
            batch_runs = runs[batch_start : batch_start + batch_size]
            print(f"\n处理批次: {batch_start//batch_size + 1}/{(len(runs) + batch_size - 1)//batch_size}")
            print(f"当前内存使用: {self.monitor_memory_usage():.1f} MB")

            with ProcessPoolExecutor(max_workers=min(max_workers, len(batch_runs))) as executor:
                future_to_params = {executor.submit(self.run_single_simulation, bw, traffic_files, topo_type): (bw, rep) for bw, rep in batch_runs}

                for future in as_completed(future_to_params):
                    bw, rep = future_to_params[future]
                    try:
                        result = future.result(timeout=1800)
                        result["repeat_id"] = rep
                    except Exception as e:
                        logging.exception(f"带宽 {bw}GB/s 仿真失败: {e}")
                        result = {
                            "cdma_bw_limit": bw,
                            "repeat_id": rep,
                            "completion_status": "failed",
                            "error_message": str(e),
                        }

                    self.save_result_to_csv(result)
                    completed += 1
                    elapsed_time = time.time() - start_time
                    eta = (elapsed_time / completed) * (total_runs - completed) if completed else 0
                    print(f"  完成 {completed}/{total_runs} - ETA: {eta/60:.1f}分钟 - 内存: {self.monitor_memory_usage():.1f}MB")

            # 批次间清理内存
            gc.collect()
            time.sleep(1)  # 短暂休息

        print(f"\n带宽扫描完成! 总运行时间: {(time.time() - start_time)/60:.1f}分钟")
        print(f"所有结果已保存到: {self.output_csv}")

    def generate_summary_from_csv(self):
        """
        从CSV文件生成更详细、更具洞察力的Markdown分析摘要，使用正确的性能指标
        """
        try:
            if not self.output_csv.exists():
                print("CSV文件不存在，无法生成摘要")
                return

            # 1. 读取并清理数据
            df = pd.read_csv(self.output_csv)
            if df.empty:
                print("CSV文件为空，无法生成摘要")
                return

            # 处理NaN值，使用正确的指标列名
            performance_columns = ["trans_mixed_avg_latency", "data_mixed_avg_latency", "mixed_avg_weighted_bw", "Total_finish_time", "Total_sum_BW"]  # 保留原来的指标作为备选

            for col in performance_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            if "error_message" in df.columns:
                df["error_message"] = df["error_message"].fillna("N/A")

            # 2. 准备摘要文件和基本信息
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = self.output_dir / f"cdma_analysis_summary_{timestamp}.md"

            with open(summary_file, "w", encoding="utf-8") as f:
                f.write("# CDMA带宽性能分析报告\n\n")
                f.write(f"**分析时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**数据来源:** `{self.output_csv.name}`\n\n")

                # 提取实验配置信息 (从成功的运行中)
                success_df = df[df["completion_status"] == "success"]
                if not success_df.empty:
                    first_run = success_df.iloc[0]
                    f.write("## 1. 实验配置\n")
                    f.write(f"- **拓扑结构:** `{first_run.get('topo_type', 'N/A')}`\n")
                    f.write(f"- **流量文件:** `{first_run.get('traffic_files', 'N/A')}`\n")
                    f.write(f"- **测试带宽 (GB/s):** `{sorted(df['cdma_bw_limit'].unique())}`\n\n")

                # 3. 整体结果概览
                total_runs = len(df)
                success_count = len(success_df)
                success_rate = (success_count / total_runs * 100) if total_runs > 0 else 0
                f.write("## 2. 总体结果\n")
                f.write(f"- **总仿真次数:** {total_runs}\n")
                f.write(f"- **成功运行次数:** {success_count}\n")
                f.write(f"- **成功率:** `{success_rate:.1f}%`\n\n")

                if success_df.empty:
                    f.write("**没有成功的运行可供分析。**\n")
                    # 分析失败案例
                    failed_df = df[df["completion_status"] == "failed"]
                    if not failed_df.empty:
                        f.write("## 5. 失败分析\n\n")
                        f.write(f"共有 `{len(failed_df)}` 次仿真运行失败。\n\n")
                        failure_summary = failed_df.groupby(["cdma_bw_limit", "error_message"]).size().reset_index(name="count")
                        f.write("### 失败原因汇总\n\n")
                        f.write(failure_summary.to_markdown(index=False))
                        f.write("\n")
                    return

                # 4. 详细性能分析 - 使用正确的指标
                f.write("## 3. 详细性能分析\n\n")

                # 确定可用的性能指标
                available_metrics = {}
                if "mixed_avg_weighted_bw" in success_df.columns:
                    available_metrics["带宽性能"] = "mixed_avg_weighted_bw"
                elif "Total_sum_BW" in success_df.columns:
                    available_metrics["带宽性能"] = "Total_sum_BW"

                if "trans_mixed_avg_latency" in success_df.columns:
                    available_metrics["传输延迟"] = "trans_mixed_avg_latency"

                if "data_mixed_avg_latency" in success_df.columns:
                    available_metrics["数据延迟"] = "data_mixed_avg_latency"

                if "Total_finish_time" in success_df.columns:
                    available_metrics["完成时间"] = "Total_finish_time"

                if not available_metrics:
                    f.write("**警告：未找到可用的性能指标进行分析**\n\n")
                    return

                f.write("### 可用性能指标\n\n")
                for metric_name, column_name in available_metrics.items():
                    f.write(f"- **{metric_name}:** `{column_name}`\n")
                f.write("\n")

                # 计算统计数据
                agg_dict = {}
                for metric_name, column_name in available_metrics.items():
                    agg_dict[column_name] = ["mean", "std", "max", "min"]

                stats_df = success_df.groupby("cdma_bw_limit").agg(agg_dict).round(3)

                # 重命名列为中文
                new_columns = []
                for col in stats_df.columns:
                    column_name = col[0]
                    agg_func = col[1]
                    # 找到对应的中文名称
                    metric_name = None
                    for m_name, c_name in available_metrics.items():
                        if c_name == column_name:
                            metric_name = m_name
                            break

                    if metric_name:
                        if agg_func == "mean":
                            new_columns.append(f"{metric_name}_平均")
                        elif agg_func == "std":
                            new_columns.append(f"{metric_name}_标准差")
                        elif agg_func == "max":
                            new_columns.append(f"{metric_name}_最大")
                        elif agg_func == "min":
                            new_columns.append(f"{metric_name}_最小")
                        else:
                            new_columns.append(f"{metric_name}_{agg_func}")
                    else:
                        new_columns.append(f"{column_name}_{agg_func}")

                stats_df.columns = new_columns

                # 处理std为NaN的情况
                for col in stats_df.columns:
                    if "标准差" in col:
                        stats_df[col] = stats_df[col].fillna(0)

                # 计算性能效率（如果有带宽指标）
                if "带宽性能" in available_metrics:
                    bw_mean_col = "带宽性能_平均"
                    if bw_mean_col in stats_df.columns:
                        stats_df["效率_每GB带宽性能"] = (stats_df[bw_mean_col] / stats_df.index).round(3)
                        stats_df["效率_性能增益%"] = stats_df[bw_mean_col].pct_change().fillna(0) * 100
                        stats_df["效率_性能增益%"] = stats_df["效率_性能增益%"].round(1)

                f.write("### 性能统计表\n\n")
                f.write(stats_df.to_markdown(index=True))
                f.write("\n\n")

                # 5. 生成性能曲线图
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
                    plt.rcParams["axes.unicode_minus"] = False

                    # 创建多子图
                    n_metrics = len(available_metrics)
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    axes = axes.flatten()

                    plot_idx = 0

                    # 绘制每个可用指标
                    for metric_name, column_name in available_metrics.items():
                        if plot_idx < 4:  # 最多4个子图
                            ax = axes[plot_idx]

                            mean_col = f"{metric_name}_平均"
                            std_col = f"{metric_name}_标准差"

                            if mean_col in stats_df.columns:
                                x_vals = stats_df.index
                                y_vals = stats_df[mean_col]
                                y_err = stats_df[std_col] if std_col in stats_df.columns else None

                                ax.errorbar(x_vals, y_vals, yerr=y_err, marker="o", linestyle="-", capsize=5, label=f"{metric_name}")
                                ax.set_xlabel("CDMA带宽限制 (GB/s)")
                                ax.set_ylabel(f"{metric_name}")
                                ax.grid(True, alpha=0.3)
                                ax.legend()

                            plot_idx += 1

                    # 如果有效率分析，绘制效率曲线
                    if "效率_每GB带宽性能" in stats_df.columns and plot_idx < 4:
                        ax = axes[plot_idx]
                        ax.plot(stats_df.index, stats_df["效率_每GB带宽性能"], marker="s", linestyle="--", color="green", label="每GB带宽性能")
                        ax.set_xlabel("CDMA带宽限制 (GB/s)")
                        ax.set_ylabel("效率 (性能/带宽)")
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        plot_idx += 1

                    # 隐藏未使用的子图
                    for i in range(plot_idx, 4):
                        axes[i].set_visible(False)

                    plt.suptitle("CDMA带宽性能分析", fontsize=16)
                    plt.tight_layout()

                    plot_filename = f"performance_analysis_{timestamp}.png"
                    plot_path = self.output_dir / plot_filename
                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)

                    f.write("### 性能分析图表\n\n")
                    f.write(f"![性能分析图表]({plot_filename})\n\n")
                    print(f"性能分析图已保存到: {plot_path}")

                except Exception as plot_e:
                    f.write("### 性能分析图表\n\n")
                    f.write(f"*(生成图表时出错: {plot_e})*\n\n")

                # 6. 配置推荐
                f.write("## 4. 配置推荐\n\n")

                # 基于可用指标给出推荐
                if "带宽性能" in available_metrics and "带宽性能_平均" in stats_df.columns:
                    bw_mean_col = "带宽性能_平均"

                    # 最大性能点
                    max_perf_idx = stats_df[bw_mean_col].idxmax()
                    max_perf_val = stats_df.loc[max_perf_idx, bw_mean_col]
                    f.write(f"### 最大性能配置\n")
                    f.write(f"- **CDMA带宽:** `{max_perf_idx} GB/s`\n")
                    f.write(f"- **实现性能:** `{max_perf_val:.3f}`\n\n")

                    # 最佳效率点
                    if "效率_每GB带宽性能" in stats_df.columns:
                        eff_col = "效率_每GB带宽性能"
                        best_eff_idx = stats_df[eff_col].idxmax()
                        best_eff_val = stats_df.loc[best_eff_idx, eff_col]
                        f.write(f"### 最佳效率配置\n")
                        f.write(f"- **CDMA带宽:** `{best_eff_idx} GB/s`\n")
                        f.write(f"- **效率值:** `{best_eff_val:.3f}`\n\n")

                # 7. 失败案例分析
                failed_df = df[df["completion_status"] == "failed"]
                if not failed_df.empty:
                    f.write("## 5. 失败分析\n\n")
                    f.write(f"共有 `{len(failed_df)}` 次仿真运行失败。\n\n")
                    failure_summary = failed_df.groupby(["cdma_bw_limit", "error_message"]).size().reset_index(name="count")
                    f.write("### 失败原因汇总\n\n")
                    f.write(failure_summary.to_markdown(index=False))
                    f.write("\n")

            print(f"分析摘要已保存到: {summary_file}")

        except Exception as e:
            print(f"生成摘要时出错: {e}")
            traceback.print_exc()

    def run_single_simulation(self, cdma_bw_limit, traffic_files, topo_type="5x4"):
        """
        运行单次仿真 - 添加资源清理

        Args:
            cdma_bw_limit: CDMA带宽限制 (GB/s)
            traffic_files: traffic文件列表
            topo_type: 拓扑类型

        Returns:
            dict: 仿真结果
        """
        sim = None
        try:
            print(f"开始仿真: CDMA带宽={cdma_bw_limit}GB/s, PID={os.getpid()}")

            # 加载配置
            cfg = CrossRingConfig(self.config_path)
            cfg.TOPO_TYPE = topo_type

            # 创建仿真模型
            sim = REQ_RSP_model(
                model_type="REQ_RSP",
                config=cfg,
                topo_type=topo_type,
                traffic_file_path=self.traffic_file_path,
                traffic_config=traffic_files,
                result_save_path=f"../Result/CrossRing/TMB/{self.run_timestamp}/bw_{cdma_bw_limit}/",
                results_fig_save_path=f"{self.output_dir}/figs/",
                plot_flow_fig=1,
                plot_RN_BW_fig=1,
                verbose=1,
            )

            # 设置CDMA带宽限制
            sim.config.CDMA_BW_LIMIT = cdma_bw_limit

            # 配置仿真参数（保持原有配置）
            sim.config.BURST = 4
            sim.config.NUM_IP = 36
            sim.config.NUM_DDR = 32
            sim.config.NUM_L2M = 32
            sim.config.NUM_GDMA = 32
            sim.config.NUM_SDMA = 32
            sim.config.NUM_CDMA = 4
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

            # 延迟配置
            sim.config.DDR_R_LATENCY_original = 40
            sim.config.DDR_R_LATENCY_VAR_original = 0
            sim.config.DDR_W_LATENCY_original = 0
            sim.config.L2M_R_LATENCY_original = 12
            sim.config.L2M_W_LATENCY_original = 16

            # FIFO配置
            sim.config.IQ_CH_FIFO_DEPTH = 10
            sim.config.EQ_CH_FIFO_DEPTH = 10
            sim.config.IQ_OUT_FIFO_DEPTH_HORIZONTAL = 8
            sim.config.IQ_OUT_FIFO_DEPTH_VERTICAL = 8
            sim.config.IQ_OUT_FIFO_DEPTH_EQ = 8
            sim.config.RB_IN_FIFO_DEPTH = 16
            sim.config.RB_OUT_FIFO_DEPTH = 8
            sim.config.EQ_IN_FIFO_DEPTH = 16

            # Tag配置
            sim.config.TL_Etag_T2_UE_MAX = 8
            sim.config.TL_Etag_T1_UE_MAX = 15
            sim.config.TR_Etag_T2_UE_MAX = 12
            sim.config.TU_Etag_T2_UE_MAX = 8
            sim.config.TU_Etag_T1_UE_MAX = 15
            sim.config.TD_Etag_T2_UE_MAX = 12

            sim.config.ITag_TRIGGER_Th_H = sim.config.ITag_TRIGGER_Th_V = 80
            sim.config.ITag_MAX_NUM_H = sim.config.ITag_MAX_NUM_V = 1
            sim.config.ETag_BOTHSIDE_UPGRADE = 0

            # DMA配置
            sim.config.GDMA_RW_GAP = np.inf
            sim.config.SDMA_RW_GAP = np.inf

            # 通道配置
            sim.config.CHANNEL_SPEC = {
                "gdma": 2,
                "sdma": 2,
                "cdma": 1,
                "ddr": 2,
                "l2m": 2,
            }

            # 初始化并运行仿真
            sim.initial()
            # sim.end_time = 1000
            sim.print_interval = 50000  # 减少打印频率
            sim.run()

            # 收集结果
            results = sim.get_results()

            # 添加CDMA特定指标
            cdma_results = {
                "cdma_bw_limit": cdma_bw_limit,
                "total_cdma_bw_limit": cdma_bw_limit * 4,
                "traffic_files": str(traffic_files),
                "topo_type": topo_type,
                "completion_status": "success",
                "run_timestamp": datetime.now().isoformat(),
            }

            # 合并结果
            results.update(cdma_results)

            # 提取关键性能指标
            key_metrics = ["trans_mixed_avg_latency", "data_mixed_avg_latency", "mixed_avg_weighted_bw", "Total_finish_time", "Total_sum_BW"]

            # 提取端口平均带宽
            if "port_averages" in results:
                results.update(results["port_averages"])

            found_metrics = []
            for metric in key_metrics:
                if metric in results:
                    found_metrics.append(metric)

            print(f"仿真成功: CDMA={cdma_bw_limit}GB/s, 找到指标: {found_metrics}")

            return results

        except Exception as e:
            error_msg = f"仿真失败 - CDMA带宽: {cdma_bw_limit}GB/s, 错误: {str(e)}"
            print(error_msg)

            return {
                "cdma_bw_limit": cdma_bw_limit,
                "total_cdma_bw_limit": cdma_bw_limit * 4,
                "traffic_files": str(traffic_files),
                "topo_type": topo_type,
                "completion_status": "failed",
                "error_message": str(e),
                "run_timestamp": datetime.now().isoformat(),
            }

        finally:
            # 清理资源
            if sim is not None:
                try:
                    del sim
                except:
                    pass
            gc.collect()  # 强制垃圾回收


def main():
    """命令行接口，执行CDMA带宽分析"""

    parser = argparse.ArgumentParser(description="CDMA bandwidth sweep - 优化版本")
    parser.add_argument("--config", default="../config/topologies/topo_5x4.yaml", help="配置文件路径")
    parser.add_argument("--traffic_path", default="../traffic/0617/", help="traffic文件目录")
    parser.add_argument("--output_dir", default=None, help="结果输出目录")
    parser.add_argument("--bandwidths", nargs="+", type=int, default=list(range(32, 3, -1)), help="待测试的CDMA带宽列表")
    parser.add_argument("--repeat", type=int, default=1, help="每个带宽的重复次数")
    parser.add_argument("--topo", default="5x4", help="拓扑类型")
    parser.add_argument("--max_workers", type=int, default=1, help="并行进程数(推荐1-2个避免内存问题)")
    parser.add_argument("--memory_limit", type=float, default=10000, help="内存使用限制(MB)")
    args = parser.parse_args()

    # 检查系统资源
    total_memory = psutil.virtual_memory().total / 1024 / 1024  # MB
    available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB

    print(f"系统内存信息:")
    print(f"  总内存: {total_memory:.1f} MB")
    print(f"  可用内存: {available_memory:.1f} MB")
    print(f"  设定限制: {args.memory_limit} MB")

    if available_memory < args.memory_limit:
        print(f"警告: 可用内存({available_memory:.1f}MB) 低于设定限制({args.memory_limit}MB)")
        print("建议减少并行进程数或调整内存限制")

    # 根据内存情况调整worker数量
    # if args.max_workers > 2 and available_memory < 12000:
    #     suggested_workers = max(1, int(available_memory / 6000))  # 每个worker大约需要6GB
    #     print(f"基于可用内存，建议使用 {suggested_workers} 个worker")
    #     args.max_workers = min(args.max_workers, suggested_workers)

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"../Result/cdma_analysis/{time_stamp}"

    analyzer = CDMABandwidthAnalyzer(
        config_path=args.config,
        traffic_file_path=args.traffic_path,
        output_dir=output_dir,
        cdma_bw_ranges=args.bandwidths,
        run_timestamp=time_stamp,
    )

    # 流量文件配置
    traffic_files = [
        [
            # "All2All_Combine.txt",
            "All2All_Dispatch.txt",
        ],
        [
            "MLP_MoE.txt",
        ]
        * 9,
    ]

    print(f"\n开始分析，配置如下:")
    print(f"  CDMA带宽范围: {args.bandwidths}")
    print(f"  重复次数: {args.repeat}")
    print(f"  并行进程数: {args.max_workers}")
    print(f"  拓扑类型: {args.topo}")

    try:
        analyzer.run_bandwidth_sweep(
            traffic_files=traffic_files,
            topo_type=args.topo,
            repeat=args.repeat,
            max_workers=args.max_workers,
        )

        print(f"\n开始生成分析摘要...")
        analyzer.generate_summary_from_csv()

        print(f"\n分析完成!")
        print(f"结果文件: {analyzer.output_csv}")
        print(f"输出目录: {analyzer.output_dir}")

        # 显示最终内存使用情况
        final_memory = analyzer.monitor_memory_usage()
        print(f"最终内存使用: {final_memory:.1f} MB")

    except KeyboardInterrupt:
        print("\n用户中断，当前结果已保存在CSV文件中")
        try:
            analyzer.generate_summary_from_csv()
            print("已生成部分结果的分析摘要")
        except:
            pass
    except Exception as e:
        print(f"\n分析过程中出现错误: {e}")
        traceback.print_exc()
        try:
            analyzer.generate_summary_from_csv()
            print("尝试基于已有结果生成摘要")
        except:
            pass


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 设置进程启动方法（对某些系统有帮助）
    if hasattr(multiprocessing, "set_start_method"):
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # 已经设置过了

    main()
