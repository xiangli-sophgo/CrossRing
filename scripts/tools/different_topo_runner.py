from src.core import *
import os
import pandas as pd
import numpy as np
import sys
import tracemalloc
from src.utils.component import Flit, Network, Node
from config.config import CrossRingConfig
import matplotlib
import time
from datetime import datetime

if sys.platform == "darwin":  # macOS 的系统标识是 'darwin'
    matplotlib.use("macosx")  # 仅在 macOS 上使用该后端


class CrossRingBatchRunner:
    def __init__(self):
        self.results = []
        self.traffic_base_path = r"../../traffic/nxn_traffics/"
        self.config_path = r"../../config/config2.json"
        self.result_save_path = f"../../Result/CrossRing_different_topo/"
        self.csv_output_path = f"../../Result/CrossRing_different_topo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # 确保输出目录存在
        os.makedirs(self.result_save_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_output_path), exist_ok=True)

    def get_topology_configs(self):
        """定义不同拓扑的配置参数"""
        base_config = {
            "BURST": 4,
            "RN_R_TRACKER_OSTD": 64,
            "RN_W_TRACKER_OSTD": 64,
            "SN_DDR_R_TRACKER_OSTD": 64,
            "SN_DDR_W_TRACKER_OSTD": 64,
            "SN_L2M_R_TRACKER_OSTD": 64,
            "SN_L2M_W_TRACKER_OSTD": 64,
            "DDR_R_LATENCY_original": 40,
            "DDR_R_LATENCY_VAR_original": 0,
            "DDR_W_LATENCY_original": 0,
            "L2M_R_LATENCY_original": 12,
            "L2M_W_LATENCY_original": 16,
            "IQ_CH_FIFO_DEPTH": 10,
            "EQ_CH_FIFO_DEPTH": 10,
            "IQ_OUT_FIFO_DEPTH": 8,
            "RB_OUT_FIFO_DEPTH": 8,
            "SN_TRACKER_RELEASE_LATENCY": 40,
            "TL_Etag_T2_UE_MAX": 8,
            "TL_Etag_T1_UE_MAX": 15,
            "TR_Etag_T2_UE_MAX": 12,
            "RB_IN_FIFO_DEPTH": 16,
            "TU_Etag_T2_UE_MAX": 8,
            "TU_Etag_T1_UE_MAX": 15,
            "TD_Etag_T2_UE_MAX": 12,
            "EQ_IN_FIFO_DEPTH": 16,
            "ITag_TRIGGER_Th_H": 80,
            "ITag_TRIGGER_Th_V": 80,
            "ITag_MAX_NUM_H": 1,
            "ITag_MAX_NUM_V": 1,
            "ETag_BOTHSIDE_UPGRADE": 0,
            "SLICE_PER_LINK": 8,
            "GDMA_RW_GAP": np.inf,
            "SDMA_RW_GAP": np.inf,
            "CHANNEL_SPEC": {
                "gdma": 2,
                "sdma": 2,
                "ddr": 2,
                "l2m": 2,
            },
        }

        # 为不同拓扑生成配置
        topo_configs = {}
        for rows in range(3, 11):  # 3x3 到 10x10
            cols = rows  # 只生成方形拓扑
            topo_name = f"{rows}x{cols}"
            num_nodes = rows * cols

            config = base_config.copy()
            config["NUM_NODE"] = num_nodes**2
            config["NUM_COL"] = cols
            config["NUM_ROW"] = num_nodes // cols * 2
            config["NUM_IP"] = num_nodes
            config["NUM_DDR"] = num_nodes
            config["NUM_L2M"] = num_nodes
            config["NUM_GDMA"] = num_nodes
            config["NUM_SDMA"] = num_nodes
            config["NUM_RN"] = num_nodes
            config["NUM_SN"] = num_nodes

            topo_configs[topo_name] = config

        return topo_configs

    def is_matching_topology(self, filename, topo_name):
        """检查文件名是否与指定拓扑匹配"""
        filename_lower = filename.lower()
        topo_lower = topo_name.lower()

        # 直接包含拓扑名称 (如: traffic_3x3_case1.txt)
        if topo_lower in filename_lower:
            return True

        # 提取拓扑的行列数
        rows, cols = map(int, topo_name.split("x"))

        # 检查各种可能的格式
        patterns = [
            f"{rows}x{cols}",  # 3x3
            f"{rows}X{cols}",  # 3X3 (大写X)
            f"{rows}_{cols}",  # 3_3 (下划线)
            f"{rows}-{cols}",  # 3-3 (连字符)
            f"_{rows}x{cols}_",  # _3x3_
            f"_{rows}X{cols}_",  # _3X3_
            f"-{rows}x{cols}-",  # -3x3-
            f"-{rows}X{cols}-",  # -3X3-
        ]

        for pattern in patterns:
            if pattern in filename_lower:
                return True

        return False

    def find_traffic_files(self, topo_name):
        """查找指定拓扑的traffic文件"""
        if not os.path.exists(self.traffic_base_path):
            print(f"Warning: Traffic path not found: {self.traffic_base_path}")
            return []

        traffic_files = []
        try:
            for file in os.listdir(self.traffic_base_path):
                if file.endswith(".txt"):
                    if self.is_matching_topology(file, topo_name):
                        traffic_files.append(file)
        except Exception as e:
            print(f"Error reading traffic directory: {e}")
            return []

        if traffic_files:
            print(f"Found {len(traffic_files)} traffic files for {topo_name}: {traffic_files}")

        return traffic_files

    def run_single_simulation(self, topo_name, traffic_file, model_type="REQ_RSP"):
        """运行单个仿真"""
        try:
            print(f"Running simulation: {topo_name} with {traffic_file}")

            config = CrossRingConfig(self.config_path)
            config.TOPO_TYPE = topo_name
            num_col = int(topo_name[0])
            num_core = num_col**2
            config.NUM_NODE = num_core * 2
            config.NUM_COL = num_col
            config.NUM_ROW = num_col * 2
            config.NUM_IP = num_core
            config.NUM_RN = num_core
            config.NUM_SN = num_core
            config.NUM_GDMA = num_core
            config.NUM_SDMA = num_core
            config.NUM_DDR = num_core
            config.NUM_L2M = num_core

            config.BURST = 4
            config.NUM_DDR = 32
            config.NUM_L2M = 32
            config.NUM_GDMA = 32
            config.NUM_SDMA = 32
            config.NUM_RN = 32
            config.NUM_SN = 32
            config.RN_R_TRACKER_OSTD = 64
            config.RN_W_TRACKER_OSTD = 64
            config.RN_RDB_SIZE = config.RN_R_TRACKER_OSTD * config.BURST
            config.RN_WDB_SIZE = config.RN_W_TRACKER_OSTD * config.BURST
            config.SN_DDR_R_TRACKER_OSTD = 64
            config.SN_DDR_W_TRACKER_OSTD = 64
            config.SN_L2M_R_TRACKER_OSTD = 64
            config.SN_L2M_W_TRACKER_OSTD = 64
            config.SN_DDR_WDB_SIZE = config.SN_DDR_W_TRACKER_OSTD * config.BURST
            config.SN_L2M_WDB_SIZE = config.SN_L2M_W_TRACKER_OSTD * config.BURST
            config.DDR_R_LATENCY_original = 40
            config.DDR_R_LATENCY_VAR_original = 0
            config.DDR_W_LATENCY_original = 0
            config.L2M_R_LATENCY_original = 12
            config.L2M_W_LATENCY_original = 16
            config.IQ_CH_FIFO_DEPTH = 10
            config.EQ_CH_FIFO_DEPTH = 10
            config.IQ_OUT_FIFO_DEPTH = 8
            config.RB_OUT_FIFO_DEPTH = 8
            config.SN_TRACKER_RELEASE_LATENCY = 40

            # 创建仿真器
            sim: BaseModel = eval(f"{model_type}_model")(
                model_type=model_type,
                config=config,
                topo_type=topo_name,
                traffic_file_path=self.traffic_base_path,
                file_name=traffic_file,
                result_save_path=self.result_save_path,
                results_fig_save_path=None,  # 批量运行时不保存图片
                plot_flow_fig=1,
                plot_RN_BW_fig=1,
                plot_link_state=0,
                plot_start_time=0,
                print_trace=0,
                show_trace_id=0,
                show_node_id=0,
                verbose=1,
            )

            # 运行仿真
            np.random.seed(609)
            sim.initial()
            sim.end_time = 6000
            sim.print_interval = 6000  # 减少打印频率

            sim.run()

            # 收集结果
            result = {}

            # 添加更多性能指标（如果仿真器提供的话）
            if hasattr(sim, "get_performance_metrics"):
                metrics = sim.get_results()
                result.update(metrics)

            print(f"Completed: {topo_name} with {traffic_file}")
            del sim
            return result

        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\nTraceback:\n{traceback.format_exc()}"
            print(f"Error running {topo_name} with {traffic_file}: {error_msg}")
            return {"topology": topo_name, "traffic_file": traffic_file, "model_type": model_type, "status": f"error: {str(e)}"}  # CSV中只保存简短错误信息

    def run_batch(self, model_types=["REQ_RSP"], max_topology_size=None):
        """运行批量仿真"""
        print("Starting CrossRing batch simulation...")
        print(f"Traffic base path: {self.traffic_base_path}")
        print(f"Results will be saved to: {self.csv_output_path}")
        print()

        total_runs = 0
        completed_runs = 0

        # 获取所有拓扑配置
        topo_configs = self.get_topology_configs()

        # 首先扫描所有traffic文件，按拓扑分组
        print("Scanning traffic files by topology...")
        all_traffic_files = {}
        for topo_name in sorted(topo_configs.keys()):
            # 如果设置了最大拓扑大小限制
            if max_topology_size:
                rows, cols = map(int, topo_name.split("x"))
                if rows > max_topology_size or cols > max_topology_size:
                    continue

            traffic_files = self.find_traffic_files(topo_name)
            if traffic_files:
                all_traffic_files[topo_name] = traffic_files

        print(f"\nFound traffic files for {len(all_traffic_files)} topologies:")
        for topo, files in all_traffic_files.items():
            print(f"  {topo}: {len(files)} files")
        print()

        # 开始仿真
        for model_type in model_types:
            print(f"Running simulations with model type: {model_type}")

            for topo_name in sorted(all_traffic_files.keys()):
                traffic_files = all_traffic_files[topo_name]

                for traffic_file in traffic_files:
                    total_runs += 1
                    result = self.run_single_simulation(topo_name, traffic_file, model_type)
                    self.results.append(result)

                    completed_runs += 1

                    # 定期保存结果
                    if len(self.results) % 5 == 0:
                        self.save_results()

        # 最终保存结果
        self.save_results()

        print(f"\nBatch simulation completed!")
        print(f"Total runs: {total_runs}")

    def save_results(self):
        """保存结果到CSV文件"""
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        df.to_csv(self.csv_output_path, index=False)
        print(f"Results saved to {self.csv_output_path} ({len(self.results)} records)")

    def generate_summary_report(self):
        """生成汇总报告"""
        if not self.results:
            print("No results to summarize")
            return

        df = pd.DataFrame(self.results)

        print("\n" + "=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)

        # 按拓扑统计
        print("\nResults by Topology:")
        topo_summary = df.groupby("topology").agg({"status": lambda x: (x == "completed").sum(), "simulation_time": "mean", "total_cycles": "mean"}).round(2)
        topo_summary.columns = ["Completed_Runs", "Avg_SimTime(s)", "Avg_Cycles"]
        print(topo_summary)

        # 按模型类型统计
        print("\nResults by Model Type:")
        model_summary = df.groupby("model_type").agg({"status": lambda x: (x == "completed").sum(), "simulation_time": "mean"}).round(2)
        model_summary.columns = ["Completed_Runs", "Avg_SimTime(s)"]
        print(model_summary)

        # 失败的运行
        failed_runs = df[df["status"] != "completed"]
        if not failed_runs.empty:
            print(f"\nFailed Runs ({len(failed_runs)}):")
            for _, row in failed_runs.iterrows():
                print(f"  {row['topology']} - {row['traffic_file']}: {row['status']}")
        else:
            print("\nNo failed runs!")


def main():
    """主函数"""
    runner = CrossRingBatchRunner()

    # 配置运行参数
    model_types = ["REQ_RSP"]  # 可以添加其他模型类型，如 ["REQ_RSP", "Feature", "Packet_Base"]
    max_topology_size = 10  # 设置最大拓扑大小限制，None表示无限制

    print("=" * 60)
    print("CrossRing Topology-based Traffic Matching Runner")
    print("=" * 60)
    print(f"Traffic path: {runner.traffic_base_path}")
    print(f"Max topology size: {max_topology_size if max_topology_size else 'No limit'}")
    print(f"Model types: {model_types}")
    print("=" * 60)
    print()

    # 运行批量仿真
    runner.run_batch(model_types=model_types, max_topology_size=max_topology_size)


if __name__ == "__main__":
    main()
