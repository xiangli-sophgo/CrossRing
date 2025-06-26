#!/usr/bin/env python3
"""
CDMA带宽并行行为分析脚本

遍历不同的CDMA带宽设置，分析与上层计算模块的并行性能表现。
所有结果保存到CSV文件供后续分析。

Usage:
    python scripts/two_micro_batch.py \
        --config ../config/config2.json \
        --traffic_path ../traffic/0617/ \
        --output_dir ../Result/cdma_analysis \
        --bandwidths 4 8 12 16 20 24 28 32 \
        --repeat 1 --topo 5x4 --max_workers 4

每个带宽值在独立进程中运行仿真，结果保存在
``../Result/CrossRing/TMB/<bandwidth>/<topo>/<traffic...>/`` 目录下。
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

# 添加项目路径
sys.path.append("../")
sys.path.append("../src")
sys.path.append("../config")

from config.config import CrossRingConfig
from src.core import REQ_RSP_model


class CDMABandwidthAnalyzer:
    def __init__(self, config_path="../config/config2.json", traffic_file_path="../traffic/0617/", output_dir="../Result/cdma_analysis/", cdma_bw_ranges=[4, 16, 32]):
        """
        初始化CDMA带宽分析器

        Args:
            config_path: 配置文件路径
            traffic_file_path: traffic文件路径
            output_dir: 结果输出目录
        """
        self.config_path = config_path
        self.traffic_file_path = traffic_file_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CDMA带宽范围 (GB/s)：单通道1-32，4通道总带宽4-128
        self.cdma_bw_ranges = cdma_bw_ranges

        # 设置CSV输出文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_csv = self.output_dir / f"cdma_bandwidth_analysis_{timestamp}.csv"
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

    def run_bandwidth_sweep(self, traffic_files, topo_type="5x4", repeat=1, max_workers=None):
        """
        运行带宽扫描分析

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

        if max_workers is None:
            max_workers = min(len(self.cdma_bw_ranges), multiprocessing.cpu_count())

        start_time = time.time()
        completed = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {executor.submit(self.run_single_simulation, bw, traffic_files, topo_type): (bw, rep) for bw, rep in runs}

            for future in as_completed(future_to_params):
                bw, rep = future_to_params[future]
                try:
                    result = future.result()
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
                print(f"  完成 {completed}/{total_runs} - ETA: {eta/60:.1f}分钟")

        print(f"\n带宽扫描完成! 总运行时间: {(time.time() - start_time)/60:.1f}分钟")
        print(f"所有结果已保存到: {self.output_csv}")

    def generate_summary_from_csv(self):
        """
        从CSV文件生成更详细、更具洞察力的Markdown分析摘要，并包含性能曲线图。
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

            # 处理NaN值，对于数值列用0填充，对于错误信息用'N/A'填充
            for col in ["Total_sum_BW", "simulation_time"]:
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
                    # 即使没有成功案例，也要分析失败案例
                    failed_df = df[df["completion_status"] == "failed"]
                    if not failed_df.empty:
                        f.write("## 5. 失败分析\n\n")
                        f.write(f"共有 `{len(failed_df)}` 次仿真运行失败。\n\n")

                        # 汇总失败原因
                        failure_summary = failed_df.groupby(["cdma_bw_limit", "error_message"]).size().reset_index(name="count")
                        f.write("### 失败原因汇总\n\n")
                        f.write(failure_summary.to_markdown(index=False))
                        f.write("\n")
                    return

                # 4. 详细性能分析
                f.write("## 3. 详细性能分析\n\n")

                # 计算统计数据
                stats_df = (
                    success_df.groupby("cdma_bw_limit")
                    .agg(mean_bw=("Total_sum_BW", "mean"), std_bw=("Total_sum_BW", "std"), max_bw=("Total_sum_BW", "max"), mean_time=("simulation_time", "mean"))
                    .round(3)
                )

                # 处理std为NaN的情况 (当每个组只有一个样本时)
                stats_df["std_bw"] = stats_df["std_bw"].fillna(0)

                # 计算性价比和性能增益
                stats_df["perf_per_gb"] = (stats_df["mean_bw"] / stats_df.index).round(3)
                stats_df["perf_gain_pct"] = stats_df["mean_bw"].pct_change().fillna(0) * 100
                stats_df["perf_gain_pct"] = stats_df["perf_gain_pct"].round(1)

                f.write("### 性能统计 vs. CDMA带宽\n\n")
                f.write(stats_df.to_markdown(index=True))
                f.write("\n\n* **perf_per_gb:** 每GB/s带宽的性能 (平均总带宽 / CDMA带宽限制) - 值越高越好。\n")
                f.write("* **perf_gain_pct:** 相较于前一带宽设置的平均性能百分比增益。\n\n")

                # 5. 生成并嵌入图表
                try:
                    import matplotlib
                    matplotlib.use("Agg")  # Use non-interactive backend
                    import matplotlib.pyplot as plt
                    import matplotlib.font_manager as fm

                    # 设置中文字体
                    plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者其他支持中文的字体，如 'WenQuanYi Micro Hei'
                    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

                    fig, ax1 = plt.subplots(figsize=(12, 7))

                    # 绘制性能曲线和误差棒 (主Y轴)
                    ax1.errorbar(stats_df.index, stats_df["mean_bw"], yerr=stats_df["std_bw"], marker="o", linestyle="-", capsize=5, label="平均总带宽", color="b")
                    ax1.set_xlabel("CDMA带宽限制 (GB/s 每通道)")
                    ax1.set_ylabel("平均系统总带宽 (GB/s)", color="b")
                    ax1.tick_params(axis="y", labelcolor="b")
                    ax1.grid(True)

                    # 创建第二个Y轴绘制性价比
                    ax2 = ax1.twinx()
                    ax2.plot(stats_df.index, stats_df["perf_per_gb"], marker="s", linestyle="--", label="每GB/s性能", color="g")
                    ax2.set_ylabel("每GB/s性能 (平均带宽 / CDMA带宽)", color="g")
                    ax2.tick_params(axis="y", labelcolor="g")

                    # 创建第三个Y轴绘制仿真时间 (延迟)
                    ax3 = ax1.twinx()
                    # 调整ax3的位置，使其不与ax2重叠
                    ax3.spines["right"].set_position(("outward", 60))  # 60 points outward
                    ax3.plot(stats_df.index, stats_df["mean_time"], marker="^", linestyle=":", label="平均仿真时间 (延迟)", color="r")
                    ax3.set_ylabel("平均仿真时间 (ns)", color="r")
                    ax3.tick_params(axis="y", labelcolor="r")

                    # 合并图例
                    lines, labels = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    lines3, labels3 = ax3.get_legend_handles_labels()
                    ax3.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc="upper left")

                    plt.title("性能、效率与延迟 vs. CDMA带宽")
                    fig.tight_layout()

                    plot_filename = f"performance_summary_{timestamp}.png"
                    plot_path = self.output_dir / plot_filename
                    plt.savefig(plot_path)
                    plt.close(fig)

                    f.write("### 性能曲线图\n\n")
                    f.write(f"![性能曲线图]({plot_filename})\n\n")
                    print(f"性能分析图已保存到: {plot_path}")

                except ImportError:
                    f.write("### 性能曲线图\n\n")
                    f.write("*(未安装Matplotlib，跳过图表生成。)*\n\n")
                except Exception as plot_e:
                    f.write("### 性能曲线图\n\n")
                    f.write(f"*(生成图表时出错: {plot_e})*\n\n")

                # 6. 配置推荐
                f.write("## 4. 配置推荐\n\n")

                # 找到最大性能点
                max_perf_row = stats_df.loc[stats_df["mean_bw"].idxmax()]
                f.write(f"### 最大性能配置\n")
                f.write(f"- **CDMA带宽:** `{max_perf_row.name} GB/s`\n")
                f.write(f"- **实现平均带宽:** `{max_perf_row['mean_bw']:.3f} GB/s`\n")
                f.write(f"- *注意: 此设置提供最高的绝对性能，但可能不是最具成本效益的。*\n\n")

                # 找到最佳性价比点
                best_value_row = stats_df.loc[stats_df["perf_per_gb"].idxmax()]
                f.write(f"### 最佳性价比配置\n")
                f.write(f"- **CDMA带宽:** `{best_value_row.name} GB/s`\n")
                f.write(f"- **每GB/s性能:** `{best_value_row['perf_per_gb']:.3f}`\n")
                f.write(f"- **实现平均带宽:** `{best_value_row['mean_bw']:.3f} GB/s`\n")
                f.write(f"- *注意: 此设置提供了最佳的“投入产出比”。*\n\n")

                # 找到性能饱和点
                # 寻找性能增益首次低于5%的点的前一个点作为平衡点
                saturation_candidates = stats_df[stats_df["perf_gain_pct"] < 5]
                if not saturation_candidates.empty:
                    saturation_point_bw = saturation_candidates.index[0]
                    # 获取饱和点之前的一个带宽设置作为推荐
                    prev_bw_index = stats_df.index.get_loc(saturation_point_bw) - 1
                    if prev_bw_index >= 0:
                        balanced_bw = stats_df.index[prev_bw_index]
                        sat_row = stats_df.loc[saturation_point_bw]
                        f.write(f"### 平衡点 (性能饱和点) 配置\n")
                        f.write(f"- **CDMA带宽:** `{balanced_bw} GB/s`\n")
                        f.write(f"- *原因: 将带宽增加到 `{sat_row.name} GB/s` 之后，性能增益仅为 `{sat_row['perf_gain_pct']}%`，表明收益递减。*\n\n")

                # 7. 失败案例分析
                failed_df = df[df["completion_status"] == "failed"]
                if not failed_df.empty:
                    f.write("## 5. 失败分析\n\n")
                    f.write(f"共有 `{len(failed_df)}` 次仿真运行失败。\n\n")

                    # 汇总失败原因
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
        运行单次仿真

        Args:
            cdma_bw_limit: CDMA带宽限制 (GB/s)
            traffic_files: traffic文件列表
            topo_type: 拓扑类型

        Returns:
            dict: 仿真结果
        """
        try:
            print(f"开始仿真: CDMA带宽={cdma_bw_limit}GB/s, Traffic文件={traffic_files}")

            # 加载配置
            cfg = CrossRingConfig(self.config_path)
            cfg.TOPO_TYPE = topo_type

            # 创建仿真模型 - 参考traffic_sim_main.py的参数设置
            sim = REQ_RSP_model(
                model_type="REQ_RSP",
                config=cfg,
                topo_type=topo_type,
                traffic_file_path=self.traffic_file_path,
                traffic_config=traffic_files,  # 直接传入文件列表
                result_save_path=f"../Result/CrossRing/TMB/bw_{cdma_bw_limit}/",
                results_fig_save_path=f"{self.output_dir}/figs/",
                flow_fig_show_CDMA=1,
                plot_flow_fig=1,  # 不生成图像
                plot_RN_BW_fig=1,
                verbose=0,  # 减少输出
            )

            # 设置CDMA带宽限制
            sim.config.CDMA_BW_LIMIT = cdma_bw_limit

            # 配置仿真参数 - 完全参考traffic_sim_main.py
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

            # 延迟配置 - 使用traffic_sim_main.py的数值
            sim.config.DDR_R_LATENCY_original = 40
            sim.config.DDR_R_LATENCY_VAR_original = 0
            sim.config.DDR_W_LATENCY_original = 0
            sim.config.L2M_R_LATENCY_original = 12
            sim.config.L2M_W_LATENCY_original = 16

            # FIFO配置
            sim.config.IQ_CH_FIFO_DEPTH = 10
            sim.config.EQ_CH_FIFO_DEPTH = 10
            sim.config.IQ_OUT_FIFO_DEPTH = 8
            sim.config.RB_IN_FIFO_DEPTH = 16  # 添加这个配置
            sim.config.RB_OUT_FIFO_DEPTH = 8
            sim.config.EQ_IN_FIFO_DEPTH = 16  # 添加这个配置

            # 标签配置
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

            # 通道配置 - 修改为traffic_sim_main.py的设置
            sim.config.CHANNEL_SPEC = {
                "gdma": 2,
                "sdma": 2,
                "cdma": 1,  # 保持原来的CDMA配置
                "ddr": 4,  # 修改为4
                "l2m": 2,
            }

            # 初始化并运行仿真
            sim.initial()
            sim.end_time = 1000  # 足够的仿真时间
            sim.print_interval = 30000  # 减少打印频率
            sim.run()

            # 收集结果
            results = sim.get_results()

            # 添加CDMA特定指标
            cdma_results = {
                "cdma_bw_limit": cdma_bw_limit,
                "total_cdma_bw_limit": cdma_bw_limit * 4,  # 4通道总带宽
                "traffic_files": str(traffic_files),
                "topo_type": topo_type,
                "simulation_time": results.get("simulation_time", 0),
                "completion_status": "success",
                "run_timestamp": datetime.now().isoformat(),
            }

            # 合并结果
            results.update(cdma_results)

            print(f"仿真成功: CDMA带宽={cdma_bw_limit}GB/s, 总带宽={results.get('Total_sum_BW', 0):.3f}GB/s")
            return results

        except Exception as e:
            error_msg = f"仿真失败 - CDMA带宽: {cdma_bw_limit}GB/s, 错误: {str(e)}"
            print(error_msg)
            traceback.print_exc()

            return {
                "cdma_bw_limit": cdma_bw_limit,
                "total_cdma_bw_limit": cdma_bw_limit * 4,
                "traffic_files": str(traffic_files),
                "topo_type": topo_type,
                "completion_status": "failed",
                "error_message": str(e),
                "run_timestamp": datetime.now().isoformat(),
                "Total_sum_BW": 0,
                "simulation_time": 0,
            }


def main():
    """命令行接口，执行CDMA带宽分析"""

    parser = argparse.ArgumentParser(description="CDMA bandwidth sweep")
    parser.add_argument("--config", default="../config/config2.json", help="配置文件路径")
    parser.add_argument("--traffic_path", default="../traffic/0617/", help="traffic文件目录")
    parser.add_argument("--output_dir", default=None, help="结果输出目录")
    parser.add_argument("--bandwidths", nargs="+", type=int, default=list(range(30, 32)), help="待测试的CDMA带宽列表")
    parser.add_argument("--repeat", type=int, default=1, help="每个带宽的重复次数")
    parser.add_argument("--topo", default="5x4", help="拓扑类型")
    parser.add_argument("--max_workers", type=int, default=2, help="并行进程数")
    args = parser.parse_args()

    output_dir = args.output_dir or f"../Result/cdma_analysis/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    analyzer = CDMABandwidthAnalyzer(
        config_path=args.config,
        traffic_file_path=args.traffic_path,
        output_dir=output_dir,
        cdma_bw_ranges=args.bandwidths,
    )

    traffic_files = [
        [
            "All2All_Dispatch.txt",
        ],
        [
            "MLP_MoE.txt",
        ]
        * 9,
    ]

    try:
        analyzer.run_bandwidth_sweep(
            traffic_files=traffic_files,
            topo_type=args.topo,
            repeat=args.repeat,
            max_workers=args.max_workers,
        )
        analyzer.generate_summary_from_csv()

        print(f"\n分析完成!")
        print(f"结果文件: {analyzer.output_csv}")
        print(f"请查看输出目录: {analyzer.output_dir}")

    except KeyboardInterrupt:
        print("\n用户中断，当前结果已保存在CSV文件中")
    except Exception as e:
        print(f"\n分析过程中出现错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
