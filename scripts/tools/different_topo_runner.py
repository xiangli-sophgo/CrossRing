from src.noc import *
import os
import pandas as pd
import numpy as np
import sys
import tracemalloc
import traceback
from src.utils.component import Flit, Network, Node
from config.config import CrossRingConfig
import matplotlib
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

if sys.platform == "darwin":  # macOS 的系统标识是 'darwin'
    matplotlib.use("macosx")  # 仅在 macOS 上使用该后端


class CrossRingBatchRunner:
    def __init__(self):
        self.results = []
        self.traffic_base_path = r"../../traffic/nxn_MLP_MoE"
        self.config_path = r"../../config/topologies/topo_5x4.yaml"
        self.result_save_path = f"../../Result/CrossRing_different_topo_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
        self.csv_output_path = f"../../Result/CrossRing_different_topo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # 确保输出目录存在
        os.makedirs(self.result_save_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_output_path), exist_ok=True)

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
        # try:
        print(f"Running simulation: {topo_name} with {traffic_file}")

        config = CrossRingConfig(self.config_path)
        config.TOPO_TYPE = topo_name
        rows, cols = map(int, topo_name.split("x"))
        # cols *= 2
        # 解析行列并计算核心数量
        num_cores = rows * cols
        # 每个核心对应两个网络节点（RN & SN）
        config.NUM_NODE = num_cores * 2
        # 网络行数是核心行数的两倍，列数保持核心列数
        config.NUM_ROW = rows * 2
        config.NUM_COL = cols
        # IP、RN、SN、DMA 等统计基于核心数量
        config.NUM_IP = num_cores
        config.NUM_RN = num_cores
        config.NUM_SN = num_cores
        config.NUM_GDMA = num_cores
        config.NUM_SDMA = num_cores
        config.NUM_DDR = num_cores
        config.NUM_L2M = num_cores

        config.BURST = 4
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
        config.IQ_OUT_FIFO_DEPTH_HORIZONTAL = 8
        config.IQ_OUT_FIFO_DEPTH_VERTICAL = 8
        config.IQ_OUT_FIFO_DEPTH_EQ = 8
        config.RB_OUT_FIFO_DEPTH = 8
        config.SN_TRACKER_RELEASE_LATENCY = 4
        # config.DDR_BW_LIMIT = 64

        # 创建仿真器
        sim: BaseModel = eval(f"{model_type}_model")(
            model_type=model_type,
            config=config,
            topo_type=topo_name,
            traffic_file_path=self.traffic_base_path,
            traffic_config=traffic_file,
            result_save_path=self.result_save_path,
            results_fig_save_path=self.result_save_path,  # 批量运行时不保存图片
            plot_flow_fig=1,
            plot_RN_BW_fig=1,
            plot_link_state=0,
            plot_start_cycle=0,
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

        metrics = sim.get_results()
        result.update(metrics)

        print(f"Completed: {topo_name} with {traffic_file}")
        del sim
        return result

        # except Exception as e:
        #     import traceback

        #     error_msg = f"{str(e)}\nTraceback:\n{traceback.format_exc()}"
        #     print(f"Error running {topo_name} with {traffic_file}: {error_msg}")
        #     return {"topology": topo_name, "traffic_file": traffic_file, "model_type": model_type, "status": f"error: {str(e)}"}  # CSV中只保存简短错误信息

    def run_batch(self, model_types=["REQ_RSP"], max_topology_size=None, max_workers=None):
        """运行批量仿真"""
        print("Starting CrossRing batch simulation...")
        print(f"Traffic base path: {self.traffic_base_path}")
        print(f"Results will be saved to: {self.csv_output_path}")
        print()

        total_runs = 0
        completed_runs = 0

        # 扫描 traffic 目录下的 .txt 文件，按拓扑名称分组
        import re

        pattern = re.compile(r"(\d+)[xX_ \-_](\d+)")
        all_traffic_files = {}
        try:
            files = [f for f in os.listdir(self.traffic_base_path) if f.endswith(".txt")]
            print(f"Debug: Found {len(files)} files in {self.traffic_base_path}")
            for file in files:
                print(f"Debug: Examining file {file}")
                m = pattern.search(file)
                if not m:
                    continue
                topo_name = f"{m.group(1)}x{m.group(2)}"
                # 可选：仅保留在 max_topology_size 范围内的拓扑
                if max_topology_size:
                    r, c = int(m.group(1)), int(m.group(2))
                    if r > max_topology_size or c > max_topology_size:
                        continue
                all_traffic_files.setdefault(topo_name, []).append(file)
        except Exception:
            print("Error scanning traffic files, full traceback:")
            traceback.print_exc()
            all_traffic_files = {}

        print(f"\nFound traffic files for {len(all_traffic_files)} topologies:")
        for topo, files in all_traffic_files.items():
            print(f"  {topo}: {len(files)} files")
        print()

        # 开始仿真
        for model_type in model_types:
            print(f"Running simulations with model type: {model_type}")

            # 准备并行执行任务
            tasks = []
            for topo_name in sorted(all_traffic_files.keys()):
                for traffic_file in all_traffic_files[topo_name]:
                    tasks.append((topo_name, traffic_file, model_type))

            total_runs += len(tasks)
            # 使用多进程并行执行
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(self.run_single_simulation, topo, tf, mt): (topo, tf) for topo, tf, mt in tasks}
                for future in as_completed(future_to_task):
                    result = future.result()
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
        df.to_csv(self.csv_output_path, index=False, encoding='utf-8-sig')
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


def main():
    """主函数"""
    runner = CrossRingBatchRunner()

    # 配置运行参数
    model_types = ["REQ_RSP"]  # 可以添加其他模型类型，如 ["REQ_RSP", "Feature", "Packet_Base"]
    max_topology_size = 20  # 设置最大拓扑大小限制，None表示无限制
    num_parallel_jobs = 8  # 并行作业数

    print("=" * 60)
    print("CrossRing Topology-based Traffic Matching Runner")
    print("=" * 60)
    print(f"Traffic path: {runner.traffic_base_path}")
    print(f"Max topology size: {max_topology_size if max_topology_size else 'No limit'}")
    print(f"Model types: {model_types}")
    print(f"Parallel jobs: {num_parallel_jobs}")
    print("=" * 60)
    print()

    # 运行批量仿真
    runner.run_batch(model_types=model_types, max_topology_size=max_topology_size, max_workers=num_parallel_jobs)


if __name__ == "__main__":
    main()
