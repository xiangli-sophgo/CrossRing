#!/usr/bin/env python3
"""
CDMA带宽并行行为分析脚本

遍历不同的CDMA带宽设置，分析与上层计算模块的并行性能表现
将所有结果保存到CSV文件供后续分析
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

# 添加项目路径
sys.path.append("../")
sys.path.append("../src")
sys.path.append("../config")

from config.config import CrossRingConfig
from src.core import REQ_RSP_model


class CDMABandwidthAnalyzer:
    def __init__(self, config_path="../config/config2.json", traffic_file_path="../examples/traffic/", output_dir="../results/cdma_analysis/"):
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
        self.cdma_bw_ranges = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]

        # 结果收集
        self.results = []

        print(f"CDMA带宽分析器初始化完成")
        print(f"配置文件: {config_path}")
        print(f"Traffic路径: {traffic_file_path}")
        print(f"输出目录: {output_dir}")
        print(f"CDMA带宽测试范围: {self.cdma_bw_ranges} GB/s (单通道)")

    def run_single_simulation(self, cdma_bw_limit, traffic_config, topo_type="5x4"):
        """
        运行单次仿真

        Args:
            cdma_bw_limit: CDMA带宽限制 (GB/s)
            traffic_config: traffic配置
            topo_type: 拓扑类型

        Returns:
            dict: 仿真结果
        """
        try:
            # 加载配置
            cfg = CrossRingConfig(self.config_path)
            cfg.TOPO_TYPE = topo_type

            # 创建仿真模型
            sim = REQ_RSP_model(
                model_type="REQ_RSP",
                config=cfg,
                topo_type=topo_type,
                traffic_file_path=self.traffic_file_path,
                traffic_config=traffic_config,
                result_save_path=None,  # 不保存中间结果
                plot_flow_fig=0,  # 不生成图像
                plot_RN_BW_fig=0,
                verbose=1,  # 静默模式
            )

            # 设置CDMA带宽限制
            if not hasattr(sim.config, "CDMA_BW_LIMIT"):
                # 如果配置中没有CDMA_BW_LIMIT，添加该属性
                sim.config.CDMA_BW_LIMIT = cdma_bw_limit
            else:
                sim.config.CDMA_BW_LIMIT = cdma_bw_limit

            # 配置仿真参数

            sim.config.NUM_IP = 32
            sim.config.NUM_DDR = 32
            sim.config.NUM_L2M = 16
            sim.config.NUM_GDMA = 16
            sim.config.NUM_SDMA = 16
            sim.config.NUM_CDMA = 16  # 4个CDMA通道，每个通道可以有多个IP

            sim.config.NUM_COL = 4
            sim.config.NUM_NODE = 40
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
            sim.config.IQ_OUT_FIFO_DEPTH = 8
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
            sim.config.SLICE_PER_LINK = 8

            sim.config.GDMA_RW_GAP = np.inf
            sim.config.SDMA_RW_GAP = np.inf
            sim.config.CHANNEL_SPEC = {
                "gdma": 2,
                "sdma": 2,
                "cdma": 1,
                "ddr": 2,
                "l2m": 2,
            }

            # 运行仿真
            sim.initial()
            # sim.end_time = 10000  # 足够的仿真时间
            sim.print_interval = 10000
            sim.run()

            # 收集结果
            results = sim.get_results()

            # 添加CDMA特定指标
            cdma_results = {
                "cdma_bw_limit": cdma_bw_limit,
                "total_cdma_bw_limit": cdma_bw_limit * 4,  # 4通道总带宽
                "traffic_config": str(traffic_config),
                "topo_type": topo_type,
                "simulation_time": sim.cycle / sim.config.NETWORK_FREQUENCY if hasattr(sim, "cycle") else 0,
                "completion_status": "success",
            }

            # 合并结果
            results.update(cdma_results)

            # 提取关键性能指标
            if hasattr(sim, "result_processor"):
                # 获取CDMA相关的带宽统计
                processor = sim.result_processor
                processor.collect_requests_data(sim)

                # 计算CDMA实际使用带宽
                cdma_actual_bw = 0
                if hasattr(processor, "ip_bandwidth_data") and processor.ip_bandwidth_data:
                    if "cdma" in processor.ip_bandwidth_data.get("total", {}):
                        cdma_actual_bw = np.sum(processor.ip_bandwidth_data["total"]["cdma"])

                results["cdma_actual_bandwidth"] = cdma_actual_bw
                results["cdma_utilization"] = cdma_actual_bw / (cdma_bw_limit * 4) if cdma_bw_limit > 0 else 0

            return results

        except Exception as e:
            print(f"仿真失败 - CDMA带宽: {cdma_bw_limit}GB/s, 错误: {str(e)}")
            return {
                "cdma_bw_limit": cdma_bw_limit,
                "total_cdma_bw_limit": cdma_bw_limit * 4,
                "traffic_config": str(traffic_config),
                "topo_type": topo_type,
                "completion_status": "failed",
                "error_message": str(e),
                "Total_sum_BW": 0,
                "cdma_actual_bandwidth": 0,
                "cdma_utilization": 0,
            }

    def run_bandwidth_sweep(self, traffic_config, topo_type="5x4", repeat=1):
        """
        运行带宽扫描分析

        Args:
            traffic_configs: traffic配置列表
            topo_type: 拓扑类型
            repeat: 每个配置重复次数
        """
        print(f"\n开始CDMA带宽扫描分析...")
        print(f"拓扑类型: {topo_type}")
        print(f"重复次数: {repeat}")
        print(f"Traffic配置数量: {len(traffic_config)}")

        total_runs = len(self.cdma_bw_ranges) * len(traffic_config) * repeat
        current_run = 0

        start_time = time.time()

        for cdma_bw in self.cdma_bw_ranges:
            print(f"  测试CDMA带宽: {cdma_bw}GB/s (4通道总计: {cdma_bw*4}GB/s)")

            for rep in range(repeat):
                current_run += 1
                elapsed_time = time.time() - start_time
                eta = (elapsed_time / current_run) * (total_runs - current_run) if current_run > 0 else 0

                print(f"    第{rep+1}次运行 [{current_run}/{total_runs}] - ETA: {eta/60:.1f}分钟")

                # 运行仿真
                result = self.run_single_simulation(cdma_bw, traffic_config, topo_type)
                result["repeat_id"] = rep
                result["run_timestamp"] = datetime.now().isoformat()

                self.results.append(result)

                # 实时保存结果（防止数据丢失）
                if current_run % 10 == 0:  # 每10次运行保存一次
                    self.save_results(suffix=f"_partial_{current_run}")

        print(f"\n带宽扫描完成! 总运行时间: {(time.time() - start_time)/60:.1f}分钟")

    def save_results(self, suffix=""):
        """
        保存结果到CSV文件

        Args:
            suffix: 文件名后缀
        """
        if not self.results:
            print("没有结果可保存")
            return

        # 转换为DataFrame
        df = pd.DataFrame(self.results)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cdma_bandwidth_analysis_{timestamp}{suffix}.csv"
        filepath = self.output_dir / filename

        # 保存CSV
        df.to_csv(filepath, index=False)
        print(f"结果已保存到: {filepath}")

        # 生成摘要统计
        self.generate_summary(df, suffix)

        return filepath

    def generate_summary(self, df, suffix=""):
        """
        生成分析摘要

        Args:
            df: 结果DataFrame
            suffix: 文件名后缀
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = self.output_dir / f"cdma_analysis_summary_{timestamp}{suffix}.txt"

            with open(summary_file, "w", encoding="utf-8") as f:
                f.write("CDMA带宽并行行为分析摘要\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总仿真次数: {len(df)}\n")
                f.write(f"成功率: {len(df[df['completion_status']=='success'])/len(df)*100:.1f}%\n\n")

                # CDMA带宽范围分析
                f.write("CDMA带宽范围分析:\n")
                f.write("-" * 30 + "\n")

                success_df = df[df["completion_status"] == "success"]
                if not success_df.empty:
                    # 按CDMA带宽分组统计
                    cdma_stats = (
                        success_df.groupby("cdma_bw_limit")
                        .agg({"Total_sum_BW": ["mean", "std", "max", "min"], "cdma_actual_bandwidth": ["mean", "std"], "cdma_utilization": ["mean", "std"], "simulation_time": ["mean", "std"]})
                        .round(3)
                    )

                    f.write(cdma_stats.to_string())
                    f.write("\n\n")

                    # 最优带宽设置
                    best_bw_idx = success_df["Total_sum_BW"].idxmax()
                    best_bw_row = success_df.loc[best_bw_idx]

                    f.write("最优性能配置:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"CDMA带宽: {best_bw_row['cdma_bw_limit']}GB/s\n")
                    f.write(f"总带宽: {best_bw_row['Total_sum_BW']:.3f}GB/s\n")
                    f.write(f"CDMA利用率: {best_bw_row['cdma_utilization']:.3f}\n")
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

    def analyze_results(self):
        """
        分析已收集的结果
        """
        if not self.results:
            print("没有结果可分析")
            return

        df = pd.DataFrame(self.results)

        print("\n=== 分析结果 ===")
        print(f"总仿真次数: {len(df)}")
        print(f"成功次数: {len(df[df['completion_status']=='success'])}")
        print(f"失败次数: {len(df[df['completion_status']=='failed'])}")

        success_df = df[df["completion_status"] == "success"]
        if not success_df.empty:
            print(f"\n性能指标范围:")
            print(f"总带宽: {success_df['Total_sum_BW'].min():.2f} - {success_df['Total_sum_BW'].max():.2f} GB/s")
            if "cdma_actual_bandwidth" in success_df.columns:
                print(f"CDMA实际带宽: {success_df['cdma_actual_bandwidth'].min():.2f} - {success_df['cdma_actual_bandwidth'].max():.2f} GB/s")
                print(f"CDMA利用率: {success_df['cdma_utilization'].min():.3f} - {success_df['cdma_utilization'].max():.3f}")


def main():
    """
    主函数 - 执行CDMA带宽分析
    """
    print("CDMA带宽并行行为分析脚本")
    print("=" * 50)

    # 初始化分析器
    traffic_file_path = r"../traffic/0617/"
    analyzer = CDMABandwidthAnalyzer(traffic_file_path=traffic_file_path)

    # 配置traffic文件 - 根据实际情况修改
    traffic_config = [
        [
            # r"Read_burst4_2262HBM_v2.txt",
            r"MLP_MoE.txt",
        ]
        * 2,
        [
            r"All2All_Combine.txt",
        ],
    ]

    try:
        # 运行带宽扫描
        analyzer.run_bandwidth_sweep(traffic_config=traffic_config, topo_type="5x4", repeat=1)  # 根据需要修改拓扑类型  # 每个配置重复1次，可以增加以提高统计可靠性

        # 分析结果
        analyzer.analyze_results()

        # 保存最终结果
        final_file = analyzer.save_results(suffix="_final")

        print(f"\n分析完成!")
        print(f"结果文件: {final_file}")
        print(f"请查看输出目录: {analyzer.output_dir}")

    except KeyboardInterrupt:
        print("\n用户中断，保存当前结果...")
        analyzer.save_results(suffix="_interrupted")
    except Exception as e:
        print(f"\n分析过程中出现错误: {e}")
        traceback.print_exc()
        analyzer.save_results(suffix="_error")


if __name__ == "__main__":
    main()
