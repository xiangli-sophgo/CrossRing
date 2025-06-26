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
        从CSV文件生成分析摘要
        """
        try:
            if not self.output_csv.exists():
                print("CSV文件不存在，无法生成摘要")
                return

            # 读取CSV结果
            df = pd.read_csv(self.output_csv)

            if df.empty:
                print("CSV文件为空，无法生成摘要")
                return

            # 生成摘要文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = self.output_dir / f"cdma_analysis_summary_{timestamp}.txt"

            with open(summary_file, "w", encoding="utf-8") as f:
                f.write("CDMA带宽并行行为分析摘要\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总仿真次数: {len(df)}\n")

                success_count = len(df[df["completion_status"] == "success"])
                f.write(f"成功率: {success_count/len(df)*100:.1f}%\n\n")

                # CDMA带宽范围分析
                f.write("CDMA带宽范围分析:\n")
                f.write("-" * 30 + "\n")

                success_df = df[df["completion_status"] == "success"]
                if not success_df.empty:
                    # 按CDMA带宽分组统计
                    if "Total_sum_BW" in success_df.columns:
                        cdma_stats = success_df.groupby("cdma_bw_limit").agg({"Total_sum_BW": ["mean", "std", "max", "min"], "simulation_time": ["mean", "std"]}).round(3)
                        f.write(cdma_stats.to_string())
                        f.write("\n\n")

                        # 最优带宽设置
                        best_bw_idx = success_df["Total_sum_BW"].idxmax()
                        best_bw_row = success_df.loc[best_bw_idx]

                        f.write("最优性能配置:\n")
                        f.write("-" * 20 + "\n")
                        f.write(f"CDMA带宽: {best_bw_row['cdma_bw_limit']}GB/s\n")
                        f.write(f"总带宽: {best_bw_row['Total_sum_BW']:.3f}GB/s\n")
                        f.write(f"仿真时间: {best_bw_row['simulation_time']:.1f}ns\n\n")

                # 失败案例分析
                failed_df = df[df["completion_status"] == "failed"]
                if not failed_df.empty:
                    f.write("失败案例分析:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"失败次数: {len(failed_df)}\n")
                    f.write("失败的CDMA带宽设置:\n")
                    failed_bw = failed_df["cdma_bw_limit"].value_counts()
                    f.write(failed_bw.to_string())
                    f.write("\n\n")

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
                verbose=1,  # 减少输出
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
            # sim.end_time = 1000  # 足够的仿真时间
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
    parser.add_argument("--bandwidths", nargs="+", type=int, default=list(range(4, 32)), help="待测试的CDMA带宽列表")
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
            "MLP_MoE.txt",
        ]
        * 9,
        [
            "All2All_Dispatch.txt",
        ],
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
