"""
FIFO和ETag参数全遍历优化脚本

核心思路：
1. 对所有参数组合进行全遍历（笛卡尔积）
2. 处理ETag参数的约束关系：
   - TL/TR相关ETag参数必须小于RB_IN_FIFO_DEPTH
   - TU/TD相关ETag参数必须小于EQ_IN_FIFO_DEPTH
   - T1参数必须大于对应的T2参数
3. 为每个参数生成独立的性能变化曲线图
4. 只对满足约束的参数组合运行仿真

优化策略：
- 全组合遍历：生成所有参数的笛卡尔积
- 约束检查：不满足约束的组合直接跳过，不运行仿真
- 独立可视化：每个参数生成单独的性能变化曲线图（显示该参数在所有有效组合中的表现）
- 结果保存：完整数据、约束日志、可视化图表
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import *
from config.config import CrossRingConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import threading
import signal
import sys


# 配置中文字体显示
def setup_chinese_fonts():
    """设置中文字体显示"""
    try:
        # 检测系统可用的中文字体
        import matplotlib.font_manager as fm

        font_names = [f.name for f in fm.fontManager.ttflist]

        # 按优先级尝试中文字体
        chinese_fonts = [
            "Microsoft YaHei",  # 微软雅黑
            "SimHei",  # 黑体
            "KaiTi",  # 楷体
            "FangSong",  # 仿宋
            "STSong",  # 华文宋体
            "WenQuanYi Micro Hei",  # 文泉驿微米黑
            "Noto Sans CJK SC",  # 思源黑体简体中文
            "Arial Unicode MS",  # Arial Unicode MS
        ]

        # 找到第一个可用的中文字体
        available_font = None
        for font in chinese_fonts:
            if font in font_names:
                available_font = font
                break

        if available_font:
            matplotlib.rcParams["font.sans-serif"] = [available_font] + ["DejaVu Sans", "Arial"]
            # print(f"使用中文字体: {available_font}")
        else:
            # 如果没有找到中文字体，至少设置负号显示
            matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
            print("警告: 未找到中文字体，中文可能显示为方块")

        matplotlib.rcParams["axes.unicode_minus"] = False
        matplotlib.rcParams["font.size"] = 10

    except Exception as e:
        print(f"字体配置失败: {e}")
        matplotlib.rcParams["axes.unicode_minus"] = False


# 执行字体配置
setup_chinese_fonts()
import seaborn as sns
from datetime import datetime
import time
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import traceback
from typing import Dict, List, Tuple, Any, Optional
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor

# ==================== 全局配置 ====================
N_REPEATS = 1  # 每个配置的重复次数（无随机性，只需1次）
N_JOBS = 7  # 并行作业数
SIMULATION_TIME = 5000  # 仿真时间
VERBOSE = 0
# 内存优化配置
MAX_COMBINATIONS_IN_MEMORY = 100000  # 内存中最大组合数
BATCH_PROCESSING_SIZE = 1000  # 批处理大小

# Windows兼容性设置
import platform

WINDOWS_COMPATIBLE = platform.system() == "Windows"

# Windows并行处理设置
if WINDOWS_COMPATIBLE:
    try:
        import multiprocessing as mp

        # 尝试设置启动方法为spawn（Windows默认）
        mp.set_start_method("spawn", force=True)
        # 配置joblib使用loky后端（Windows兼容）
        import os

        os.environ["JOBLIB_START_METHOD"] = "spawn"
        USE_MULTIPROCESSING = True
    except (ImportError, RuntimeError, ModuleNotFoundError) as e:
        print(f"多进程不可用，将使用线程并行: {e}")
        USE_MULTIPROCESSING = False
else:
    USE_MULTIPROCESSING = True

# FIFO参数配置
FIFO_PARAMS = {
    # "IQ_CH_FIFO_DEPTH": {"range": [2, 16], "default": 4},
    # "EQ_CH_FIFO_DEPTH": {"range": [2, 16], "default": 4},
    # "IQ_OUT_FIFO_DEPTH_HORIZONTAL": {"range": [2, 8], "default": 8},
    # "IQ_OUT_FIFO_DEPTH_VERTICAL": {"range": [2, 8], "default": 8},
    # "IQ_OUT_FIFO_DEPTH_EQ": {"range": [2, 8], "default": 8},
    # "RB_OUT_FIFO_DEPTH": {"range": [2, 8], "default": 8},
    "RB_IN_FIFO_DEPTH": {"range": [2, 12], "default": 16},
    "EQ_IN_FIFO_DEPTH": {"range": [2, 12], "default": 16},
}

# ETag参数配置（包含约束关系）
ETAG_PARAMS = {
    "TL_Etag_T2_UE_MAX": {"range": [1, 11], "default": 8, "related_fifo": "RB_IN_FIFO_DEPTH", "constraint": "less_than_fifo"},
    "TL_Etag_T1_UE_MAX": {"range": [2, 11], "default": 15, "related_fifo": "RB_IN_FIFO_DEPTH", "constraint": "less_than_fifo_and_greater_than_t2", "corresponding_t2": "TL_Etag_T2_UE_MAX"},
    "TR_Etag_T2_UE_MAX": {"range": [1, 11], "default": 12, "related_fifo": "RB_IN_FIFO_DEPTH", "constraint": "less_than_fifo"},
    "TU_Etag_T2_UE_MAX": {"range": [1, 11], "default": 8, "related_fifo": "EQ_IN_FIFO_DEPTH", "constraint": "less_than_fifo"},
    "TU_Etag_T1_UE_MAX": {"range": [2, 11], "default": 15, "related_fifo": "EQ_IN_FIFO_DEPTH", "constraint": "less_than_fifo_and_greater_than_t2", "corresponding_t2": "TU_Etag_T2_UE_MAX"},
    "TD_Etag_T2_UE_MAX": {"range": [1, 11], "default": 12, "related_fifo": "EQ_IN_FIFO_DEPTH", "constraint": "less_than_fifo"},
}

# 合并所有参数
ALL_PARAMS = {**FIFO_PARAMS, **ETAG_PARAMS}


# 独立的并行仿真函数
def run_single_simulation_optimized(sim_params):
    """
    优化的独立仿真函数，参考traffic_sim_main.py的设计

    Args:
        sim_params: (combination, config_path, topo_type, traffic_files, traffic_weights, traffic_path)

    Returns:
        {"combination": combination, "performance": total_weighted_bw}
    """
    combination, config_path, topo_type, traffic_files, traffic_weights, traffic_path = sim_params

    try:
        print(f"Starting simulation for combination on process {os.getpid()}")

        # 创建仿真实例
        cfg = CrossRingConfig(config_path)
        cfg.TOPO_TYPE = topo_type

        total_weighted_bw = 0

        for traffic_file, weight in zip(traffic_files, traffic_weights):
            bw_list = []

            for repeat in range(N_REPEATS):
                sim = REQ_RSP_model(
                    model_type="REQ_RSP",
                    config=cfg,
                    topo_type=topo_type,
                    traffic_file_path=traffic_path,
                    traffic_config=traffic_file,
                    result_save_path="../Result/temp",
                    verbose=VERBOSE,
                )

                # 设置平台参数（5x4拓扑）
                if topo_type == "5x4":
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

                # 设置参数
                for param_name, param_value in combination.items():
                    if hasattr(sim.config, param_name):
                        setattr(sim.config, param_name, int(param_value))

                # 运行仿真
                sim.initial()
                sim.end_time = SIMULATION_TIME
                sim.print_interval = SIMULATION_TIME
                sim.run()

                # 获取完整的仿真结果（配置+统计）
                sim_results = sim.get_results()
                bw = sim_results.get("mixed_avg_weighted_bw", 0)

                # 调试：如果带宽为0，打印可用的结果字段
                if bw == 0:
                    print(f"Warning: bandwidth is 0 for combination {combination}")
                    print(f"Available result fields: {list(sim_results.keys())}")
                    # 尝试其他可能的带宽字段
                    for key in sim_results.keys():
                        if "bw" in key.lower() or "bandwidth" in key.lower():
                            print(f"  {key}: {sim_results[key]}")

                bw_list.append(bw)

                # 只在第一次重复时保存完整结果，避免重复
                if repeat == 0:
                    full_results = sim_results

            # 计算该traffic的平均带宽
            avg_bw = np.mean(bw_list)
            total_weighted_bw += avg_bw * weight

        # 使用完整的仿真结果，包含所有配置和统计信息
        # 这样生成的CSV与traffic_sim_main.py格式一致
        result_dict = full_results.copy()

        # 添加优化相关的元数据
        result_dict["performance"] = total_weighted_bw
        result_dict["optimization_performance"] = total_weighted_bw
        result_dict["optimization_simulation_time"] = SIMULATION_TIME
        result_dict["optimization_n_repeats"] = N_REPEATS
        result_dict["optimization_traffic_files"] = "_".join(traffic_files)
        result_dict["optimization_traffic_weights"] = "_".join(map(str, traffic_weights))

        return result_dict

    except Exception as e:
        print(f"仿真失败: {e}")
        # 失败时返回基础格式，包含错误信息
        result_dict = {
            "optimization_performance": 0,
            "optimization_simulation_time": SIMULATION_TIME,
            "optimization_n_repeats": N_REPEATS,
            "optimization_traffic_files": "_".join(traffic_files),
            "optimization_traffic_weights": "_".join(map(str, traffic_weights)),
            "optimization_error": str(e),
        }

        # 添加所有参数到结果中
        for param_name, param_value in combination.items():
            result_dict[param_name] = param_value

        return result_dict


def save_optimization_results_to_csv(result_data, output_csv):
    """保存优化结果到CSV文件，参考traffic_sim_main.py的实现"""
    if result_data is None:
        print("跳过CSV写入，由于仿真错误")
        return

    # 使用threading.Lock确保线程安全
    if not hasattr(save_optimization_results_to_csv, "_lock"):
        save_optimization_results_to_csv._lock = threading.Lock()

    csv_file_exists = os.path.isfile(output_csv)

    with save_optimization_results_to_csv._lock:
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)

            with open(output_csv, mode="a", newline="", encoding="utf-8") as output_csv_file:
                writer = csv.DictWriter(output_csv_file, fieldnames=result_data.keys())
                if not csv_file_exists:
                    writer.writeheader()
                writer.writerow(result_data)

            # 显示简要信息，使用优化后的性能指标
            combination_info = {k: v for k, v in result_data.items() if k in ALL_PARAMS.keys()}
            param_str = ", ".join([f"{k}:{v}" for k, v in list(combination_info.items())[:3]])
            performance = result_data.get("optimization_performance", result_data.get("performance", 0))
            # print(f"✓ CSV保存成功: {os.path.basename(output_csv)}")
            # print(f"  参数: {param_str}... 性能={performance:.2f}")

        except Exception as e:
            print(f"CSV保存失败: {e}")
            print(f"  输出路径: {output_csv}")
            print(f"  目录是否存在: {os.path.exists(os.path.dirname(output_csv))}")
            print(f"  结果数据键数: {len(result_data.keys()) if result_data else 0}")
            import traceback

            traceback.print_exc()


class FIFOExhaustiveOptimizer:
    """FIFO和ETag参数全遍历优化器"""

    def __init__(self, config_path: str, topo_type: str, traffic_files: List[str], traffic_weights: List[float], traffic_path: str = "../traffic/0617/"):
        self.config_path = config_path
        self.topo_type = topo_type
        self.traffic_files = traffic_files
        self.traffic_weights = traffic_weights
        self.traffic_path = traffic_path

        # 创建结果目录（参考traffic_sim_main.py的路径结构）
        timestamp = datetime.now().strftime("%m%d_%H%M")
        base_result_dir = "../Result"
        main_result_dir = f"{base_result_dir}/FIFO_Exhaustive"
        self.result_dir = f"{main_result_dir}/FIFO_Exhaustive_{timestamp}/"

        # 创建目录结构
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, "raw_data"), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, "visualizations"), exist_ok=True)

        # 同时确保基础目录存在
        os.makedirs(base_result_dir, exist_ok=True)
        os.makedirs(main_result_dir, exist_ok=True)

        # CSV输出路径
        self.csv_output_path = os.path.join(self.result_dir, "optimization_results.csv")

        # 结果存储
        self.cache = {}  # 仿真结果缓存
        self.param_results = {}  # 每个参数的遍历结果
        self.all_results = []  # 所有仿真结果

        # 周期性保存设置
        self.save_interval = 100  # 每100个仿真结果保存一次
        self.last_save_count = 0  # 上次保存时的结果数量

        # 约束优化相关
        self._constraint_cache = {}  # 约束验证缓存
        self._precomputed_ranges = {}  # 预计算的有效参数范围

        # 默认参数配置
        self.default_params = {}
        for param_name, param_info in ALL_PARAMS.items():
            self.default_params[param_name] = param_info["default"]

        # 中断处理
        self._interrupted = False
        self._setup_signal_handlers()

    def _check_constraints(self, param_name: str, param_value: int, all_params: Dict) -> bool:
        """
        检查参数约束是否满足（带缓存）

        Args:
            param_name: 参数名
            param_value: 参数值
            all_params: 所有参数的当前值

        Returns:
            True如果满足约束，False否则
        """
        if param_name not in ETAG_PARAMS:
            return True  # FIFO参数没有约束

        # 创建缓存键
        etag_info = ETAG_PARAMS[param_name]
        relevant_params = {param_name: param_value}

        if "less_than_fifo" in etag_info["constraint"]:
            related_fifo = etag_info["related_fifo"]
            relevant_params[related_fifo] = all_params.get(related_fifo, FIFO_PARAMS[related_fifo]["default"])

        if "greater_than_t2" in etag_info["constraint"]:
            corresponding_t2 = etag_info["corresponding_t2"]
            relevant_params[corresponding_t2] = all_params.get(corresponding_t2, ETAG_PARAMS[corresponding_t2]["default"])

        cache_key = json.dumps(relevant_params, sort_keys=True)

        # 检查缓存
        if cache_key in self._constraint_cache:
            return self._constraint_cache[cache_key]

        # 执行实际约束检查
        param_info = ETAG_PARAMS[param_name]
        result = True

        # 检查是否小于相关FIFO参数
        if "less_than_fifo" in param_info["constraint"]:
            related_fifo = param_info["related_fifo"]
            fifo_value = all_params.get(related_fifo, FIFO_PARAMS[related_fifo]["default"])
            if param_value >= fifo_value:
                result = False

        # 检查T1是否大于T2
        if result and "greater_than_t2" in param_info["constraint"]:
            corresponding_t2 = param_info["corresponding_t2"]
            t2_value = all_params.get(corresponding_t2, ETAG_PARAMS[corresponding_t2]["default"])
            if param_value <= t2_value:
                result = False

        # 缓存结果
        self._constraint_cache[cache_key] = result
        return result

    def _setup_signal_handlers(self):
        """设置信号处理器，支持优雅中断"""

        def signal_handler(signum, frame):
            print(f"\n收到中断信号 ({signum})，正在优雅停止...")
            print("正在保存当前结果...")
            self._interrupted = True

            # 立即保存当前结果
            if self.all_results:
                try:
                    print(f"已完成 {len(self.all_results)} 个仿真，正在生成报告...")
                    self._analyze_results()
                    self._print_best_result()
                    self.save_results()
                    print(f"中断保存完成，结果已保存至: {self.result_dir}")
                except Exception as e:
                    print(f"中断保存过程中出错: {e}")
                    # 至少保存原始数据
                    try:
                        import json

                        emergency_save = os.path.join(self.result_dir, "emergency_save.json")
                        with open(emergency_save, "w", encoding="utf-8") as f:
                            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
                        print(f"紧急保存完成: {emergency_save}")
                    except:
                        print("紧急保存也失败了")
            else:
                print("没有结果需要保存")

            print("程序已停止")
            sys.exit(0)

        # 注册信号处理器
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

    def generate_smart_combinations(self) -> List[Dict]:
        """
        智能生成参数组合，按依赖顺序生成，减少无效组合

        Returns:
            有效的参数组合列表
        """
        print("使用智能策略生成参数组合...")

        valid_combinations = []

        # Step 1: 生成FIFO参数的所有组合
        fifo_param_names = list(FIFO_PARAMS.keys())
        fifo_param_values = {}

        for param_name in fifo_param_names:
            param_range = FIFO_PARAMS[param_name]["range"]
            fifo_param_values[param_name] = list(range(param_range[0], param_range[1] + 1))

        fifo_combinations = list(itertools.product(*[fifo_param_values[name] for name in fifo_param_names]))
        print(f"FIFO参数组合数: {len(fifo_combinations):,}")

        # Step 2: 为每个FIFO组合生成对应的有效ETag组合
        total_valid = 0
        total_invalid = 0

        for fifo_values in tqdm(fifo_combinations, desc="生成ETag组合"):
            # 检查中断
            if self._interrupted:
                print("生成过程被中断")
                break

            fifo_dict = dict(zip(fifo_param_names, fifo_values))

            # 根据当前FIFO值动态计算ETag参数的有效范围
            etag_ranges = {}
            for etag_name, etag_info in ETAG_PARAMS.items():
                base_range = etag_info["range"]

                if "less_than_fifo" in etag_info["constraint"]:
                    related_fifo = etag_info["related_fifo"]
                    fifo_value = fifo_dict[related_fifo]
                    # ETag必须小于FIFO深度
                    max_val = min(base_range[1], fifo_value - 1)
                    if max_val >= base_range[0]:
                        etag_ranges[etag_name] = list(range(base_range[0], max_val + 1))
                    else:
                        etag_ranges[etag_name] = []  # 无有效值
                else:
                    etag_ranges[etag_name] = list(range(base_range[0], base_range[1] + 1))

            # 检查是否所有ETag参数都有有效值
            if any(len(ranges) == 0 for ranges in etag_ranges.values()):
                total_invalid += 1
                continue

            # 生成ETag参数的笛卡尔积
            etag_param_names = list(ETAG_PARAMS.keys())
            etag_combinations = itertools.product(*[etag_ranges[name] for name in etag_param_names])

            # 检查T1/T2约束
            for etag_values in etag_combinations:
                etag_dict = dict(zip(etag_param_names, etag_values))

                # 验证T1 > T2约束
                t1_t2_valid = True
                for etag_name, etag_info in ETAG_PARAMS.items():
                    if "greater_than_t2" in etag_info["constraint"]:
                        corresponding_t2 = etag_info["corresponding_t2"]
                        t1_value = etag_dict[etag_name]
                        t2_value = etag_dict[corresponding_t2]
                        if t1_value <= t2_value:
                            t1_t2_valid = False
                            break

                if t1_t2_valid:
                    # 合并FIFO和ETag参数
                    full_combination = {**fifo_dict, **etag_dict}
                    valid_combinations.append(full_combination)
                    total_valid += 1
                else:
                    total_invalid += 1

        print(f"智能生成完成:")
        print(f"  有效组合: {total_valid:,}")
        print(f"  无效组合: {total_invalid:,}")
        print(f"  有效率: {total_valid/(total_valid+total_invalid)*100:.1f}%")

        return valid_combinations

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

    def run_simulation(self, params: Dict) -> float:
        """
        运行仿真并返回加权性能

        Args:
            params: 参数字典

        Returns:
            加权平均带宽
        """
        # 检查缓存
        clean_params = {}
        for key, value in params.items():
            if hasattr(value, "item"):  # numpy类型
                clean_params[key] = int(value.item())
            else:
                clean_params[key] = int(value)
        cache_key = json.dumps(clean_params, sort_keys=True)

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
                        verbose=VERBOSE,  # 在并行模式下关闭详细输出
                    )

                    # 设置平台参数
                    self._set_platform_config(sim)

                    # 设置参数
                    for param_name, param_value in params.items():
                        if hasattr(sim.config, param_name):
                            setattr(sim.config, param_name, int(param_value))
                        else:
                            print(f"配置中不存在参数: {param_name}")
                            raise AttributeError(f"Configuration does not have parameter: {param_name}")

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

    def generate_all_combinations(self) -> List[Dict]:
        """
        生成所有参数的笛卡尔积组合

        Returns:
            所有可能的参数组合列表
        """
        print("生成所有参数组合...")

        # 为每个参数生成所有可能的值
        param_values = {}
        for param_name, param_info in ALL_PARAMS.items():
            param_range = param_info["range"]
            param_values[param_name] = list(range(param_range[0], param_range[1] + 1))

        # 生成笛卡尔积
        param_names = list(ALL_PARAMS.keys())
        value_combinations = itertools.product(*[param_values[name] for name in param_names])

        # 转换为字典形式
        all_combinations = []
        for values in value_combinations:
            combination = dict(zip(param_names, values))
            all_combinations.append(combination)

        total_combinations = len(all_combinations)
        print(f"生成了 {total_combinations:,} 个参数组合")

        # 估算内存和时间消耗
        estimated_memory_mb = total_combinations * 0.001  # 粗略估计每个组合1KB
        estimated_time_hours = total_combinations * SIMULATION_TIME / 1000 / 3600  # 粗略估计

        print(f"预估内存消耗: {estimated_memory_mb:.1f} MB")
        print(f"预估运行时间: {estimated_time_hours:.1f} 小时（单核）")
        print(f"使用 {N_JOBS} 核并行，预估时间: {estimated_time_hours/N_JOBS:.1f} 小时")

        return all_combinations

    def generate_smart_combinations_batches(self, batch_size: int = BATCH_PROCESSING_SIZE):
        """
        智能生成参数组合批次（生成器模式，节省内存）

        Args:
            batch_size: 每批的大小

        Yields:
            批量的有效参数组合
        """
        print(f"使用智能生成器模式分批生成有效组合，批大小: {batch_size}")

        # Step 1: 生成FIFO参数的所有组合
        fifo_param_names = list(FIFO_PARAMS.keys())
        fifo_param_values = {}

        for param_name in fifo_param_names:
            param_range = FIFO_PARAMS[param_name]["range"]
            fifo_param_values[param_name] = list(range(param_range[0], param_range[1] + 1))

        fifo_combinations = list(itertools.product(*[fifo_param_values[name] for name in fifo_param_names]))
        print(f"FIFO参数组合数: {len(fifo_combinations):,}")

        # Step 2: 为每个FIFO组合生成对应的有效ETag组合
        batch = []
        total_valid = 0
        total_invalid = 0
        batch_count = 0

        for fifo_values in fifo_combinations:
            fifo_dict = dict(zip(fifo_param_names, fifo_values))

            # 根据当前FIFO值动态计算ETag参数的有效范围
            etag_ranges = {}
            for etag_name, etag_info in ETAG_PARAMS.items():
                base_range = etag_info["range"]

                if "less_than_fifo" in etag_info["constraint"]:
                    related_fifo = etag_info["related_fifo"]
                    fifo_value = fifo_dict[related_fifo]
                    # ETag必须小于FIFO深度
                    max_val = min(base_range[1], fifo_value - 1)
                    if max_val >= base_range[0]:
                        etag_ranges[etag_name] = list(range(base_range[0], max_val + 1))
                    else:
                        etag_ranges[etag_name] = []  # 无有效值
                else:
                    etag_ranges[etag_name] = list(range(base_range[0], base_range[1] + 1))

            # 检查是否所有ETag参数都有有效值
            if any(len(ranges) == 0 for ranges in etag_ranges.values()):
                total_invalid += 1
                continue

            # 生成ETag参数的笛卡尔积
            etag_param_names = list(ETAG_PARAMS.keys())
            etag_combinations = itertools.product(*[etag_ranges[name] for name in etag_param_names])

            # 检查T1/T2约束
            for etag_values in etag_combinations:
                etag_dict = dict(zip(etag_param_names, etag_values))

                # 验证T1 > T2约束
                t1_t2_valid = True
                for etag_name, etag_info in ETAG_PARAMS.items():
                    if "greater_than_t2" in etag_info["constraint"]:
                        corresponding_t2 = etag_info["corresponding_t2"]
                        t1_value = etag_dict[etag_name]
                        t2_value = etag_dict[corresponding_t2]
                        if t1_value <= t2_value:
                            t1_t2_valid = False
                            break

                if t1_t2_valid:
                    # 合并FIFO和ETag参数
                    full_combination = {**fifo_dict, **etag_dict}
                    batch.append(full_combination)
                    total_valid += 1

                    # 当批次满了就返回
                    if len(batch) >= batch_size:
                        batch_count += 1
                        print(f"  智能生成批次 {batch_count}: {len(batch)} 个组合 (累计有效: {total_valid:,}, 无效: {total_invalid:,})")
                        yield batch
                        batch = []
                else:
                    total_invalid += 1

        # 返回最后一批（如果有的话）
        if batch:
            batch_count += 1
            print(f"  最后智能批次 {batch_count}: {len(batch)} 个组合")
            yield batch

        print(f"智能生成完成: 总计有效组合 {total_valid:,}, 无效组合 {total_invalid:,}")
        print(f"有效率: {total_valid/(total_valid+total_invalid)*100:.1f}%")
        return

    def generate_valid_combinations_batches(self, batch_size: int = BATCH_PROCESSING_SIZE):
        """
        批量生成有效的参数组合（生成器模式，节省内存）

        Args:
            batch_size: 每批的大小

        Yields:
            批量的有效参数组合
        """
        print(f"使用生成器模式分批生成有效组合，批大小: {batch_size}")

        # 为每个参数生成所有可能的值
        param_values = {}
        for param_name, param_info in ALL_PARAMS.items():
            param_range = param_info["range"]
            param_values[param_name] = list(range(param_range[0], param_range[1] + 1))

        param_names = list(ALL_PARAMS.keys())

        # 使用生成器生成笛卡尔积
        combination_generator = itertools.product(*[param_values[name] for name in param_names])

        batch = []
        total_generated = 0
        valid_count = 0
        invalid_count = 0

        for values in combination_generator:
            combination = dict(zip(param_names, values))

            # 检查约束
            is_valid = True
            for param_name, param_value in combination.items():
                if not self._check_constraints(param_name, param_value, combination):
                    is_valid = False
                    invalid_count += 1
                    break

            if is_valid:
                batch.append(combination)
                valid_count += 1

                # 当批次满了就返回
                if len(batch) >= batch_size:
                    total_generated += len(batch)
                    print(f"  生成批次: {len(batch)} 个组合 (累计有效: {valid_count:,}, 无效: {invalid_count:,})")
                    yield batch
                    batch = []

        # 返回最后一批（如果有的话）
        if batch:
            total_generated += len(batch)
            print(f"  最后批次: {len(batch)} 个组合")
            yield batch

        print(f"生成完成: 总计有效组合 {valid_count:,}, 无效组合 {invalid_count:,}")
        return

    def filter_valid_combinations(self, all_combinations: List[Dict]) -> List[Dict]:
        """
        过滤出满足约束的参数组合

        Args:
            all_combinations: 所有参数组合

        Returns:
            满足约束的参数组合列表
        """
        print("过滤满足约束的参数组合...")

        valid_combinations = []
        total_count = len(all_combinations)

        for combination in tqdm(all_combinations, desc="约束检查"):
            is_valid = True

            # 检查所有ETag参数的约束
            for param_name, param_value in combination.items():
                if not self._check_constraints(param_name, param_value, combination):
                    is_valid = False
                    break

            if is_valid:
                valid_combinations.append(combination)

        valid_count = len(valid_combinations)
        invalid_count = total_count - valid_count

        print(f"约束检查完成:")
        print(f"  总组合数: {total_count:,}")
        print(f"  有效组合: {valid_count:,} ({valid_count/total_count*100:.2f}%)")
        print(f"  无效组合: {invalid_count:,} ({invalid_count/total_count*100:.2f}%)")

        return valid_combinations

    def run_exhaustive_search_batch(self, combinations_batch: List[Dict]) -> List[Dict]:
        """
        批量运行仿真

        Args:
            combinations_batch: 参数组合批次

        Returns:
            仿真结果列表
        """
        results = []

        for combination in combinations_batch:
            performance = self.run_simulation(combination)
            result = {"combination": combination, "performance": performance}
            results.append(result)

        return results

    def run_all_exhaustive_searches(self):
        """运行所有参数组合的全遍历（内存优化版本）"""
        print("\n" + "=" * 60)
        print("开始全遍历所有FIFO和ETag参数组合（内存优化模式）")
        print("=" * 60)

        # 先估算组合数量
        total_combinations = 1
        for param_name, param_info in ALL_PARAMS.items():
            param_range = param_info["range"]
            range_size = param_range[1] - param_range[0] + 1
            total_combinations *= range_size

        print(f"理论组合总数: {total_combinations:,}")

        # 估算内存消耗
        estimated_memory_gb = total_combinations * 0.000001  # 粗略估计每个组合1KB，转换为GB
        if estimated_memory_gb > 8:  # 如果预估超过8GB内存
            print(f"预估内存消耗: {estimated_memory_gb:.1f} GB，使用分批处理模式")
            use_batch_mode = True
        else:
            print(f"预估内存消耗: {estimated_memory_gb:.3f} GB，可以使用常规模式")
            use_batch_mode = False

        if use_batch_mode:
            # 使用分批处理模式
            self._run_batch_mode()
        else:
            # 使用常规模式
            self._run_regular_mode()

    def _run_batch_mode(self):
        """分批处理模式"""
        print("使用分批处理模式...")

        batch_count = 0
        total_processed = 0

        # 分批处理有效组合
        for batch in self.generate_smart_combinations_batches(BATCH_PROCESSING_SIZE):
            batch_count += 1
            print(f"\n处理第 {batch_count} 批次，包含 {len(batch)} 个组合...")

            # 处理当前批次（简化的并行模式）
            n_jobs = N_JOBS  # 使用局部变量
            if n_jobs > 1 and len(batch) > 1:
                try:
                    print(f"  使用ProcessPoolExecutor处理批次 ({n_jobs}核)")

                    # 准备仿真参数列表
                    sim_params_list = []
                    for combination in batch:
                        sim_params = (combination, self.config_path, self.topo_type, self.traffic_files, self.traffic_weights, self.traffic_path)
                        sim_params_list.append(sim_params)

                    start_time = time.time()

                    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                        # 提交所有仿真任务
                        future_to_combination = {executor.submit(run_single_simulation_optimized, params): params[0] for params in sim_params_list}

                        # 收集结果
                        batch_results = []
                        for future in as_completed(future_to_combination):
                            combination = future_to_combination[future]
                            try:
                                result = future.result()
                                batch_results.append(result)
                                # 立即保存到CSV
                                save_optimization_results_to_csv(result, self.csv_output_path)
                            except Exception as e:
                                print(f"    仿真失败: {e}")
                                error_result = {"combination": combination, "performance": 0, "error": str(e)}
                                batch_results.append(error_result)
                                save_optimization_results_to_csv(error_result, self.csv_output_path)

                        self.all_results.extend(batch_results)

                    elapsed_time = time.time() - start_time
                    print(f"  批次完成，耗时: {elapsed_time:.1f}秒")

                except Exception as e:
                    print(f"  批次并行处理失败: {e}")
                    print(f"  使用串行处理批次{batch_count}")
                    for combination in tqdm(batch, desc=f"批次{batch_count}串行"):
                        sim_params = (combination, self.config_path, self.topo_type, self.traffic_files, self.traffic_weights, self.traffic_path)
                        try:
                            result = run_single_simulation_optimized(sim_params)
                            self.all_results.append(result)
                            save_optimization_results_to_csv(result, self.csv_output_path)
                        except Exception as e:
                            error_result = {"combination": combination, "optimization_performance": 0, "error": str(e)}
                            self.all_results.append(error_result)
                            save_optimization_results_to_csv(error_result, self.csv_output_path)
            else:
                # 串行处理小批次
                for combination in tqdm(batch, desc=f"批次{batch_count}仿真"):
                    sim_params = (combination, self.config_path, self.topo_type, self.traffic_files, self.traffic_weights, self.traffic_path)
                    try:
                        result = run_single_simulation_optimized(sim_params)
                        self.all_results.append(result)
                        save_optimization_results_to_csv(result, self.csv_output_path)
                    except Exception as e:
                        error_result = {"combination": combination, "optimization_performance": 0, "error": str(e)}
                        self.all_results.append(error_result)
                        save_optimization_results_to_csv(error_result, self.csv_output_path)

            total_processed += len(batch)

            # 定期保存中间结果
            if batch_count % 10 == 0:
                self._save_intermediate_results(batch_count, -1)  # -1表示未知总批次数

        print(f"\n分批处理完成，总计处理 {total_processed:,} 个有效组合")

        # 分析结果
        self._analyze_results()
        self._print_best_result()

    def _run_regular_mode(self):
        """常规处理模式（内存充足时使用）"""
        print("使用常规处理模式...")

        # Step 1: 使用智能生成策略直接生成有效组合
        valid_combinations = self.generate_smart_combinations()

        if not valid_combinations:
            print("没有找到满足约束的有效组合！")
            return

        print(f"\n开始运行 {len(valid_combinations):,} 个有效组合的仿真...")

        # Step 3: 批量处理有效组合
        # 使用局部变量避免全局变量作用域问题
        n_jobs = N_JOBS
        batch_size = max(1, len(valid_combinations) // n_jobs)
        batches = [valid_combinations[i : i + batch_size] for i in range(0, len(valid_combinations), batch_size)]

        print(f"将分为 {len(batches)} 个批次并行处理")

        # Step 4: 运行仿真（简化的并行处理）
        print(f"开始运行 {len(valid_combinations):,} 个有效组合的仿真...")

        # 准备仿真参数列表
        sim_params_list = []
        for combination in valid_combinations:
            sim_params = (combination, self.config_path, self.topo_type, self.traffic_files, self.traffic_weights, self.traffic_path)
            sim_params_list.append(sim_params)

        # 使用ProcessPoolExecutor进行并行处理
        if n_jobs > 1:
            try:
                print(f"使用ProcessPoolExecutor并行处理 ({n_jobs}核)")

                start_time = time.time()

                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    # 提交所有仿真任务
                    future_to_combination = {executor.submit(run_single_simulation_optimized, params): params[0] for params in sim_params_list}

                    print(f"已提交 {len(sim_params_list)} 个仿真任务到进程池")
                    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

                    # 收集结果
                    completed = 0
                    for future in as_completed(future_to_combination):
                        # 检查中断
                        if self._interrupted:
                            print("仿真过程被中断")
                            break

                        combination = future_to_combination[future]
                        try:
                            result = future.result()
                            self.all_results.append(result)
                            # 立即保存到CSV
                            save_optimization_results_to_csv(result, self.csv_output_path)
                            completed += 1

                            if completed % 50 == 0 or completed == len(sim_params_list):
                                elapsed_time = time.time() - start_time
                                avg_time = elapsed_time / completed if completed > 0 else 0
                                remaining_time = avg_time * (len(sim_params_list) - completed)
                                print(f"进度: {completed}/{len(sim_params_list)} ({completed/len(sim_params_list)*100:.1f}%) - " f"耗时: {elapsed_time:.1f}s - 预计剩余: {remaining_time:.1f}s")

                                # 检查中断（在进度报告时）
                                if self._interrupted:
                                    print("仿真过程被中断")
                                    break

                        except Exception as e:
                            print(f"仿真失败: {e}")
                            error_result = {"combination": combination, "performance": 0, "error": str(e)}
                            self.all_results.append(error_result)
                            save_optimization_results_to_csv(error_result, self.csv_output_path)
                            completed += 1

                end_time = time.time()
                print(f"并行仿真完成，总耗时: {end_time - start_time:.2f} 秒")

            except Exception as e:
                print(f"并行处理失败: {e}")
                print("使用单核处理模式")
                n_jobs = 1

        if n_jobs == 1:
            # 单核处理（并行失败时的备用方案）
            print("使用单核处理模式")

            # 准备仿真参数列表
            sim_params_list = []
            for combination in valid_combinations:
                sim_params = (combination, self.config_path, self.topo_type, self.traffic_files, self.traffic_weights, self.traffic_path)
                sim_params_list.append(sim_params)

            for i, sim_params in enumerate(tqdm(sim_params_list, desc="仿真进度")):
                try:
                    result = run_single_simulation_optimized(sim_params)
                    self.all_results.append(result)
                    save_optimization_results_to_csv(result, self.csv_output_path)
                except Exception as e:
                    combination = sim_params[0]
                    error_result = {"combination": combination, "optimization_performance": 0, "error": str(e)}
                    self.all_results.append(error_result)
                    save_optimization_results_to_csv(error_result, self.csv_output_path)

                # 每完成100个组合保存一次中间结果
                if (i + 1) % 100 == 0:
                    self._save_intermediate_results_single(i + 1, len(valid_combinations))

        # 分析结果
        self._analyze_results()
        self._print_best_result()

    def _print_best_result(self):
        """打印最佳结果"""
        if not self.all_results:
            print("没有有效的仿真结果")
            return

        best_result = max(self.all_results, key=lambda x: x.get("performance", 0))
        print(f"\n最佳配置性能: {best_result.get('performance', 0):.2f} GB/s")
        print("最佳参数组合:")
        for param_name in ALL_PARAMS.keys():
            if param_name in best_result:
                print(f"  {param_name}: {best_result[param_name]}")

    def _analyze_results(self):
        """分析所有结果，为每个参数生成统计数据"""
        print("\n分析结果数据...")

        # 为每个参数收集数据
        # 只分析实际有数据的参数
        actual_params = set()
        if self.all_results:
            # 从第一个结果中获取实际存在的参数
            first_result = self.all_results[0]
            actual_params = {k for k in first_result.keys() if k in ALL_PARAMS.keys()}
            print(f"实际参数: {list(actual_params)}")

        for param_name in actual_params:
            param_data = []

            for result in self.all_results:
                # 新格式：result直接包含所有参数和性能
                if isinstance(result, dict) and param_name in result:
                    performance = result.get("performance", 0)
                    param_value = result[param_name]

                    # 构造组合字典，保持兼容性
                    combination = {k: v for k, v in result.items() if k in actual_params}

                    param_data.append({"param_name": param_name, "param_value": param_value, "performance": performance, "full_combination": combination})

            self.param_results[param_name] = param_data

            # 统计信息
            performances = [d["performance"] for d in param_data]
            unique_values = set(d["param_value"] for d in param_data)

            print(f"{param_name}:")
            if unique_values and performances:
                print(f"  取值范围: {min(unique_values)} - {max(unique_values)}")
                print(f"  数据点数: {len(param_data)}")
                print(f"  性能范围: {min(performances):.2f} - {max(performances):.2f} GB/s")
            else:
                print(f"  无数据点")
                print(f"  数据点数: 0")

    def visualize_parameter_results(self):
        """为每个参数生成独立的性能分布图"""
        print("\n生成参数性能可视化...")

        vis_dir = os.path.join(self.result_dir, "visualizations")

        for param_name, param_data in self.param_results.items():
            if not param_data:
                continue

            # 按参数值分组，计算统计量
            df = pd.DataFrame(param_data)
            grouped = df.groupby("param_value")["performance"].agg(["mean", "min", "max", "std", "count"]).reset_index()

            # 创建图表
            plt.figure(figsize=(12, 8))

            # 绘制平均性能曲线
            plt.plot(grouped["param_value"], grouped["mean"], "b-o", linewidth=2, markersize=6, label="平均性能")

            # 绘制最大最小性能范围
            plt.fill_between(grouped["param_value"], grouped["min"], grouped["max"], alpha=0.2, color="blue", label="性能范围")

            # 标记全局最优点
            best_performance = df["performance"].max()
            best_row = df[df["performance"] == best_performance].iloc[0]
            best_value = best_row["param_value"]

            plt.plot(best_value, best_performance, "r*", markersize=15, label=f"全局最优: {best_value}")

            # 计算统计信息
            perf_range = df["performance"].max() - df["performance"].min()
            perf_change_pct = (perf_range / df["performance"].min()) * 100 if df["performance"].min() > 0 else 0
            unique_values = df["param_value"].nunique()
            total_combinations = len(param_data)

            # 添加统计信息文本框
            stats_text = (
                f"性能变化: {perf_change_pct:.1f}%\n" f"全局最优值: {best_value}\n" f"全局最优性能: {best_performance:.2f} GB/s\n" f"参数取值数: {unique_values}\n" f"总组合数: {total_combinations:,}"
            )

            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))

            plt.xlabel(f"{param_name}")
            plt.ylabel("性能 (GB/s)")
            plt.title(f"{param_name} 性能分布（基于所有有效组合）")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # 保存图表
            plt.savefig(os.path.join(vis_dir, f"{param_name}.png"), dpi=150, bbox_inches="tight")
            plt.close()

            # 额外生成箱线图显示分布
            plt.figure(figsize=(12, 6))

            # 准备箱线图数据
            box_data = []
            box_labels = []
            for value in sorted(df["param_value"].unique()):
                value_data = df[df["param_value"] == value]["performance"].values
                if len(value_data) > 0:
                    box_data.append(value_data)
                    box_labels.append(str(value))

            plt.boxplot(box_data, labels=box_labels)
            plt.xlabel(f"{param_name}")
            plt.ylabel("性能分布 (GB/s)")
            plt.title(f"{param_name} 性能分布箱线图")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45 if len(box_labels) > 10 else 0)

            # 保存箱线图
            plt.savefig(os.path.join(vis_dir, f"{param_name}_boxplot.png"), dpi=150, bbox_inches="tight")
            plt.close()

        print(f"所有参数的可视化图表已保存到: {vis_dir}")
        print(f"每个参数生成了2张图：性能曲线图和箱线图")

    def save_results(self):
        """保存所有结果"""
        print("\n保存结果...")

        # 1. 保存参数遍历结果
        all_results = []
        for param_name, results in self.param_results.items():
            all_results.extend(results)

        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(os.path.join(self.result_dir, "raw_data", "parameter_performance.csv"), index=False)

        # 3. 保存缓存
        cache_path = os.path.join(self.result_dir, "raw_data", "cache.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

        # 4. 生成摘要报告
        self._generate_summary_report()

        print(f"结果已保存至: {self.result_dir}")

    def _generate_summary_report(self):
        """生成摘要报告"""
        report = []
        report.append("=" * 60)
        report.append("FIFO和ETag参数全遍历优化报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)

        # 总体统计
        total_simulations = sum(len(results) for results in self.param_results.values())

        report.append(f"\n## 总体统计")
        report.append(f"- 参数总数: {len(ALL_PARAMS)}")
        report.append(f"- FIFO参数: {len(FIFO_PARAMS)}")
        report.append(f"- ETag参数: {len(ETAG_PARAMS)}")
        report.append(f"- 有效仿真次数: {total_simulations}")

        # 每个参数的最佳结果
        report.append(f"\n## 各参数最佳配置")

        param_best_results = []
        for param_name, results in self.param_results.items():
            if results:
                best_result = max(results, key=lambda x: x["performance"])
                param_best_results.append((param_name, best_result))

        # 按性能排序
        param_best_results.sort(key=lambda x: x[1]["performance"], reverse=True)

        for rank, (param_name, best_result) in enumerate(param_best_results, 1):
            report.append(f"{rank:2d}. {param_name}:")
            report.append(f"    最优值: {best_result['param_value']}")
            report.append(f"    最佳性能: {best_result['performance']:.2f} GB/s")

            # 计算性能影响
            param_results = self.param_results[param_name]
            performances = [r["performance"] for r in param_results]
            if len(performances) > 1:
                perf_range = max(performances) - min(performances)
                impact_pct = (perf_range / min(performances)) * 100 if min(performances) > 0 else 0
                report.append(f"    性能影响: {impact_pct:.1f}%")
                report.append(f"    有效配置数: {len(param_results)}")

        # 保存报告
        report_path = os.path.join(self.result_dir, "summary_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

        print(f"摘要报告: {report_path}")
        print(f"CSV结果文件: {self.csv_output_path}")

    def _save_intermediate_results(self, completed_batches: int, total_batches: int):
        """保存中间结果，防止长时间运行后丢失数据"""
        try:
            # 保存已完成的结果
            if self.all_results:
                intermediate_file = os.path.join(self.result_dir, "raw_data", f"intermediate_results_batch_{completed_batches}.csv")
                results_df = pd.DataFrame(self.all_results)
                results_df.to_csv(intermediate_file, index=False)

            # 保存缓存
            cache_file = os.path.join(self.result_dir, "raw_data", f"intermediate_cache_batch_{completed_batches}.json")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)

            # 生成中间进度报告
            progress_file = os.path.join(self.result_dir, "raw_data", f"progress_batch_{completed_batches}.txt")
            with open(progress_file, "w", encoding="utf-8") as f:
                f.write(f"中间保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"已完成批次: {completed_batches}/{total_batches}\n")
                f.write(f"已完成组合数: {len(self.all_results):,}\n")
                if self.all_results:
                    performances = [r["performance"] for r in self.all_results]
                    f.write(f"当前最佳性能: {max(performances):.4f}\n")
                    f.write(f"当前平均性能: {sum(performances)/len(performances):.4f}\n")

            print(f"\n[保存] 中间结果已保存 (批次 {completed_batches}/{total_batches})")

        except Exception as e:
            print(f"[警告] 中间结果保存失败: {e}")

    def _save_intermediate_results_single(self, completed_combinations: int, total_combinations: int):
        """保存单核模式的中间结果"""
        try:
            # 保存已完成的结果
            if self.all_results:
                intermediate_file = os.path.join(self.result_dir, "raw_data", f"intermediate_results_combo_{completed_combinations}.csv")
                results_df = pd.DataFrame(self.all_results)
                results_df.to_csv(intermediate_file, index=False)

            # 保存缓存
            cache_file = os.path.join(self.result_dir, "raw_data", f"intermediate_cache_combo_{completed_combinations}.json")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)

            # 生成中间进度报告
            progress_file = os.path.join(self.result_dir, "raw_data", f"progress_combo_{completed_combinations}.txt")
            with open(progress_file, "w", encoding="utf-8") as f:
                f.write(f"中间保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"已完成组合: {completed_combinations}/{total_combinations}\n")
                completion_rate = (completed_combinations / total_combinations) * 100
                f.write(f"完成率: {completion_rate:.2f}%\n")
                if self.all_results:
                    performances = [r["performance"] for r in self.all_results]
                    f.write(f"当前最佳性能: {max(performances):.4f}\n")
                    f.write(f"当前平均性能: {sum(performances)/len(performances):.4f}\n")

            print(f"\n[保存] 中间结果已保存 ({completed_combinations}/{total_combinations} 组合)")

        except Exception as e:
            print(f"[警告] 单核中间结果保存失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("FIFO和ETag参数全遍历优化")
    print("=" * 60)

    # 配置
    config_path = "../config/topologies/topo_5x4.yaml"
    topo_type = "5x4"
    traffic_files = ["LLama2_AllReduce.txt"]
    traffic_weights = [1.0]
    traffic_path = "../traffic/0617/"  # 使用绝对路径

    # 创建优化器
    optimizer = FIFOExhaustiveOptimizer(config_path=config_path, topo_type=topo_type, traffic_files=traffic_files, traffic_weights=traffic_weights, traffic_path=traffic_path)

    print(f"配置信息:")
    print(f"  拓扑: {topo_type}")
    print(f"  Traffic: {traffic_files}")
    print(f"  结果目录: {optimizer.result_dir}")
    print(f"  参数总数: {len(ALL_PARAMS)} (FIFO: {len(FIFO_PARAMS)}, ETag: {len(ETAG_PARAMS)})")

    try:
        # 运行全遍历
        optimizer.run_all_exhaustive_searches()

        # 生成可视化
        optimizer.visualize_parameter_results()

        # 保存结果
        optimizer.save_results()

        print("\n" + "=" * 60)
        print("全遍历优化完成！")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n用户中断优化")
        optimizer.save_results()
    except Exception as e:
        print(f"\n优化过程出错: {e}")
        traceback.print_exc()
        optimizer.save_results()


if __name__ == "__main__":
    # Windows多进程需要这个保护
    if WINDOWS_COMPATIBLE and USE_MULTIPROCESSING:
        try:
            import multiprocessing as mp

            mp.freeze_support()
        except:
            pass

    main()
