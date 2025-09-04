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
            print(f"使用中文字体: {available_font}")
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ==================== 全局配置 ====================
N_REPEATS = 1  # 每个配置的重复次数（无随机性，只需1次）
N_JOBS = 10  # 并行作业数
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
    # "IQ_OUT_FIFO_DEPTH_HORIZONTAL": {"range": [2, 16], "default": 8},
    # "IQ_OUT_FIFO_DEPTH_VERTICAL": {"range": [2, 16], "default": 8},
    # "IQ_OUT_FIFO_DEPTH_EQ": {"range": [2, 16], "default": 8},
    # "RB_OUT_FIFO_DEPTH": {"range": [2, 16], "default": 8},
    "RB_IN_FIFO_DEPTH": {"range": [3, 12], "default": 16},
    # "EQ_IN_FIFO_DEPTH": {"range": [2, 16], "default": 16},
}

# ETag参数配置（包含约束关系）
ETAG_PARAMS = {
    "TL_Etag_T2_UE_MAX": {"range": [1, 6], "default": 8, "related_fifo": "RB_IN_FIFO_DEPTH", "constraint": "less_than_fifo"},
    "TL_Etag_T1_UE_MAX": {"range": [2, 7], "default": 15, "related_fifo": "RB_IN_FIFO_DEPTH", "constraint": "less_than_fifo_and_greater_than_t2", "corresponding_t2": "TL_Etag_T2_UE_MAX"},
    "TR_Etag_T2_UE_MAX": {"range": [1, 6], "default": 12, "related_fifo": "RB_IN_FIFO_DEPTH", "constraint": "less_than_fifo"},
    # "TU_Etag_T2_UE_MAX": {"range": [1, 15], "default": 8, "related_fifo": "EQ_IN_FIFO_DEPTH", "constraint": "less_than_fifo"},
    # "TU_Etag_T1_UE_MAX": {"range": [2, 15], "default": 15, "related_fifo": "EQ_IN_FIFO_DEPTH", "constraint": "less_than_fifo_and_greater_than_t2", "corresponding_t2": "TU_Etag_T2_UE_MAX"},
    # "TD_Etag_T2_UE_MAX": {"range": [1, 15], "default": 12, "related_fifo": "EQ_IN_FIFO_DEPTH", "constraint": "less_than_fifo"},
}

# 合并所有参数
ALL_PARAMS = {**FIFO_PARAMS, **ETAG_PARAMS}


# Windows并行处理辅助函数
def run_single_simulation(combination, config_path, topo_type, traffic_files, traffic_weights, traffic_path):
    """
    并行处理的包装函数，用于Windows兼容性
    """
    try:
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

                # 获取结果
                bw = sim.get_results().get("mixed_avg_weighted_bw", 0)
                bw_list.append(bw)

            # 计算该traffic的平均带宽
            avg_bw = np.mean(bw_list)
            total_weighted_bw += avg_bw * weight

        return {"combination": combination, "performance": total_weighted_bw}

    except Exception as e:
        print(f"仿真失败: {e}")
        return {"combination": combination, "performance": 0}


def run_batch_simulation_wrapper(combinations_batch, config_path, topo_type, traffic_files, traffic_weights, traffic_path):
    """
    批量仿真的包装函数，用于并行处理
    """
    import os
    import time

    process_id = os.getpid()
    start_time = time.time()
    batch_size = len(combinations_batch)

    # 在子进程中输出状态（虽然主进程看不到，但可以确认在运行）
    print(f"[进程 {process_id}] 开始处理批次: {batch_size} 个组合")

    results = []
    for i, combination in enumerate(combinations_batch):
        if i % 5 == 0:  # 每5个组合输出一次状态
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1) if i > 0 else 0
            remaining_time = avg_time * (batch_size - i - 1)
            print(f"[进程 {process_id}] 进度: {i}/{batch_size} ({i/batch_size*100:.1f}%) - 耗时: {elapsed:.1f}s - 预计剩余: {remaining_time:.1f}s")

        # 运行单个仿真前的状态
        if i < 3:  # 只在前3个组合显示详细信息，避免过多输出
            param_info = ", ".join([f"{k}:{v}" for k, v in list(combination.items())[:3]])  # 只显示前3个参数
            print(f"[进程 {process_id}] 开始仿真组合 {i+1}: {param_info}...")

        result = run_single_simulation(combination, config_path, topo_type, traffic_files, traffic_weights, traffic_path)
        results.append(result)

        # 仿真完成的状态
        if i < 3:
            performance = result.get("performance", 0)
            print(f"[进程 {process_id}] 完成仿真组合 {i+1}: 性能 = {performance:.2f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[进程 {process_id}] 批次完成: {batch_size} 个组合，总耗时: {total_time:.2f}s，平均: {total_time/batch_size:.2f}s/组合")

    return results


def run_parallel_with_threads(combinations_batch_list, config_path, topo_type, traffic_files, traffic_weights, traffic_path, n_jobs=N_JOBS):
    """
    使用线程池进行并行处理（Windows兼容）
    """
    results = []

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # 提交所有任务
        future_to_batch = {executor.submit(run_batch_simulation_wrapper, batch, config_path, topo_type, traffic_files, traffic_weights, traffic_path): batch for batch in combinations_batch_list}

        # 收集结果
        for future in as_completed(future_to_batch):
            try:
                batch_result = future.result()
                results.extend(batch_result)
            except Exception as e:
                print(f"批次处理失败: {e}")
                # 对失败的批次返回空性能
                batch = future_to_batch[future]
                for combination in batch:
                    results.append({"combination": combination, "performance": 0})

    return results


def run_parallel_with_processes(combinations_batch_list, config_path, topo_type, traffic_files, traffic_weights, traffic_path, n_jobs=N_JOBS):
    """
    使用进程池进行并行处理（参考traffic_sim_main.py）
    """
    results = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # 提交所有任务
        future_to_batch = {
            executor.submit(run_batch_simulation_wrapper, batch, config_path, topo_type, traffic_files, traffic_weights, traffic_path): i for i, batch in enumerate(combinations_batch_list)
        }

        # 收集结果
        completed = 0
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_result = future.result()
                results.extend(batch_result)
                completed += 1
                print(f"进程并行进度: {completed}/{len(combinations_batch_list)} 批次完成")
            except Exception as e:
                print(f"批次 {batch_idx} 处理失败: {e}")
                # 对失败的批次返回空性能
                batch = combinations_batch_list[batch_idx]
                for combination in batch:
                    results.append({"combination": combination, "performance": 0})
                completed += 1

    return results


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

        # 结果存储
        self.cache = {}  # 仿真结果缓存
        self.param_results = {}  # 每个参数的遍历结果
        self.all_results = []  # 所有仿真结果

        # 默认参数配置
        self.default_params = {}
        for param_name, param_info in ALL_PARAMS.items():
            self.default_params[param_name] = param_info["default"]

    def _get_cache_key(self, params: Dict) -> str:
        """生成参数配置的缓存key"""
        clean_params = {}
        for key, value in params.items():
            if hasattr(value, "item"):  # numpy类型
                clean_params[key] = int(value.item())
            else:
                clean_params[key] = int(value)
        return json.dumps(clean_params, sort_keys=True)

    def _check_constraints(self, param_name: str, param_value: int, all_params: Dict) -> bool:
        """
        检查参数约束是否满足

        Args:
            param_name: 参数名
            param_value: 参数值
            all_params: 所有参数的当前值

        Returns:
            True如果满足约束，False否则
        """
        if param_name not in ETAG_PARAMS:
            return True  # FIFO参数没有约束

        param_info = ETAG_PARAMS[param_name]

        # 检查是否小于相关FIFO参数
        if "less_than_fifo" in param_info["constraint"]:
            related_fifo = param_info["related_fifo"]
            fifo_value = all_params.get(related_fifo, FIFO_PARAMS[related_fifo]["default"])
            if param_value >= fifo_value:
                return False

        # 检查T1是否大于T2
        if "greater_than_t2" in param_info["constraint"]:
            corresponding_t2 = param_info["corresponding_t2"]
            t2_value = all_params.get(corresponding_t2, ETAG_PARAMS[corresponding_t2]["default"])
            if param_value <= t2_value:
                return False

        return True

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
        cache_key = self._get_cache_key(params)
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
        for batch in self.generate_valid_combinations_batches(BATCH_PROCESSING_SIZE):
            batch_count += 1
            print(f"\n处理第 {batch_count} 批次，包含 {len(batch)} 个组合...")

            # 处理当前批次（智能并行模式）
            if N_JOBS > 1 and len(batch) > N_JOBS:
                parallel_success = False
                sub_batch_size = max(1, len(batch) // N_JOBS)
                sub_batches = [batch[i : i + sub_batch_size] for i in range(0, len(batch), sub_batch_size)]

                # 优先尝试多进程并行
                if USE_MULTIPROCESSING:
                    try:
                        if WINDOWS_COMPATIBLE:
                            # Windows系统：使用loky后端和包装函数
                            from joblib import parallel_backend

                            with parallel_backend("loky", n_jobs=N_JOBS):
                                batch_results = Parallel(n_jobs=N_JOBS)(
                                    delayed(run_batch_simulation_wrapper)(sub_batch, self.config_path, self.topo_type, self.traffic_files, self.traffic_weights, self.traffic_path)
                                    for sub_batch in sub_batches
                                )
                        else:
                            # 非Windows系统：使用默认后端
                            batch_results = Parallel(n_jobs=N_JOBS)(delayed(self.run_exhaustive_search_batch)(sub_batch) for sub_batch in sub_batches)

                        # 合并子批次结果
                        for batch_result in batch_results:
                            self.all_results.extend(batch_result)
                        parallel_success = True

                    except Exception as e:
                        print(f"  多进程并行失败: {e}")

                # 如果多进程失败，尝试线程并行
                if not parallel_success:
                    try:
                        with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
                            future_to_batch = {
                                executor.submit(run_batch_simulation_wrapper, sub_batch, self.config_path, self.topo_type, self.traffic_files, self.traffic_weights, self.traffic_path): sub_batch
                                for sub_batch in sub_batches
                            }

                            for future in as_completed(future_to_batch):
                                try:
                                    batch_result = future.result()
                                    self.all_results.extend(batch_result)
                                except Exception as e:
                                    print(f"  子批次处理失败: {e}")
                                    # 对失败的子批次返回空性能
                                    sub_batch = future_to_batch[future]
                                    for combination in sub_batch:
                                        self.all_results.append({"combination": combination, "performance": 0})

                        parallel_success = True

                    except Exception as e:
                        print(f"  线程并行也失败: {e}")

                # 如果所有并行方式都失败，使用串行处理
                if not parallel_success:
                    print(f"  所有并行方式失败，使用串行处理批次{batch_count}")
                    for combination in tqdm(batch, desc=f"批次{batch_count}串行"):
                        performance = self.run_simulation(combination)
                        result = {"combination": combination, "performance": performance}
                        self.all_results.append(result)
            else:
                # 串行处理小批次
                for combination in tqdm(batch, desc=f"批次{batch_count}仿真"):
                    performance = self.run_simulation(combination)
                    result = {"combination": combination, "performance": performance}
                    self.all_results.append(result)

            total_processed += len(batch)

            # 定期保存中间结果
            if batch_count % 10 == 0:
                self._save_intermediate_results(batch_count)

        print(f"\n分批处理完成，总计处理 {total_processed:,} 个有效组合")

        # 分析结果
        self._analyze_results()
        self._print_best_result()

    def _run_regular_mode(self):
        """常规处理模式（内存充足时使用）"""
        print("使用常规处理模式...")

        # Step 1: 生成所有组合
        all_combinations = self.generate_all_combinations()

        # Step 2: 过滤有效组合
        valid_combinations = self.filter_valid_combinations(all_combinations)

        if not valid_combinations:
            print("没有找到满足约束的有效组合！")
            return

        print(f"\n开始运行 {len(valid_combinations):,} 个有效组合的仿真...")

        # Step 3: 批量处理有效组合
        batch_size = max(1, len(valid_combinations) // N_JOBS)
        batches = [valid_combinations[i : i + batch_size] for i in range(0, len(valid_combinations), batch_size)]

        print(f"将分为 {len(batches)} 个批次并行处理")

        # Step 4: 运行仿真（智能并行处理）
        if N_JOBS > 1:
            parallel_success = False

            # 优先尝试ProcessPoolExecutor（参考traffic_sim_main.py）
            try:
                print(f"使用ProcessPoolExecutor并行处理 ({N_JOBS}核)")

                with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
                    # 提交所有任务
                    future_to_batch = {
                        executor.submit(run_batch_simulation_wrapper, batch, self.config_path, self.topo_type, self.traffic_files, self.traffic_weights, self.traffic_path): i
                        for i, batch in enumerate(batches)
                    }

                    total_combinations = sum(len(batch) for batch in batches)
                    avg_combinations_per_batch = total_combinations / len(batches) if batches else 0

                    print(f"已提交 {len(batches)} 个批次到进程池")
                    print(f"总组合数: {total_combinations:,}，平均每批次: {avg_combinations_per_batch:.0f} 个组合")
                    print("正在运行仿真... (多进程输出不显示，请等待)")
                    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

                    # 使用tqdm显示进度
                    start_time = time.time()

                    with tqdm(total=len(batches), desc="多进程仿真", unit="批次", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                        completed = 0
                        total_completed_combinations = 0

                        for future in as_completed(future_to_batch):
                            batch_idx = future_to_batch[future]
                            try:
                                batch_result = future.result()
                                self.all_results.extend(batch_result)
                                completed += 1
                                batch_size = len(batch_result)
                                total_completed_combinations += batch_size

                                # 更新进度条
                                pbar.update(1)

                                # 计算统计信息
                                elapsed_time = time.time() - start_time
                                avg_time_per_batch = elapsed_time / completed if completed > 0 else 0
                                avg_time_per_combination = elapsed_time / total_completed_combinations if total_completed_combinations > 0 else 0
                                remaining_batches = len(batches) - completed
                                remaining_combinations = total_combinations - total_completed_combinations
                                eta_seconds = avg_time_per_combination * remaining_combinations if avg_time_per_combination > 0 else 0

                                # 每完成一个批次都更新信息
                                pbar.set_postfix(
                                    {
                                        "已完成组合": f"{total_completed_combinations:,}/{total_combinations:,}",
                                        "批次耗时": f"{avg_time_per_batch:.1f}s",
                                        "组合耗时": f"{avg_time_per_combination:.2f}s",
                                        "ETA": f"{eta_seconds/60:.1f}min" if eta_seconds > 60 else f"{eta_seconds:.0f}s",
                                    }
                                )

                                # 每完成10个批次保存一次中间结果
                                if completed % 10 == 0 and completed > 0:
                                    self._save_intermediate_results(completed, len(batches))

                            except Exception as e:
                                print(f"\n批次 {batch_idx} 处理失败: {e}")
                                # 对失败的批次返回空性能
                                batch = batches[batch_idx]
                                for combination in batch:
                                    self.all_results.append({"combination": combination, "performance": 0})
                                completed += 1
                                pbar.update(1)

                parallel_success = True

            except Exception as e:
                print(f"ProcessPoolExecutor并行失败: {e}")
                print("尝试线程并行...")

            # 如果多进程失败或不可用，尝试线程并行
            if not parallel_success:
                try:
                    # 计算已完成的结果数量，避免重复执行
                    completed_combinations = len(self.all_results)
                    total_expected = sum(len(batch) for batch in batches)

                    if completed_combinations > 0:
                        print(f"ProcessPoolExecutor部分完成：已完成 {completed_combinations:,}/{total_expected:,} 组合")
                        print("使用线程池处理剩余批次...")

                        # 计算需要处理的剩余批次（简化版本：重新处理所有，但记录已有结果）
                        print("注意：由于批次边界问题，将重新处理所有批次，但会保留已有结果")
                    else:
                        print(f"使用线程池并行处理 ({N_JOBS}核)")

                    # 备份已完成的结果
                    backup_results = list(self.all_results)

                    # 使用线程池处理所有批次
                    with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
                        future_to_batch = {
                            executor.submit(run_batch_simulation_wrapper, batch, self.config_path, self.topo_type, self.traffic_files, self.traffic_weights, self.traffic_path): i
                            for i, batch in enumerate(batches)
                        }

                        # 重置结果列表，避免重复
                        self.all_results = []

                        # 使用tqdm显示进度
                        with tqdm(total=len(batches), desc="线程仿真") as pbar:
                            for future in as_completed(future_to_batch):
                                try:
                                    batch_result = future.result()
                                    self.all_results.extend(batch_result)
                                except Exception as e:
                                    print(f"批次处理失败: {e}")
                                    # 对失败的批次返回空性能
                                    batch_idx = future_to_batch[future]
                                    batch = batches[batch_idx]
                                    for combination in batch:
                                        self.all_results.append({"combination": combination, "performance": 0})
                                pbar.update(1)

                        # 合并备份结果（如果ThreadPoolExecutor成功但结果少于备份）
                        if len(backup_results) > len(self.all_results):
                            print(f"线程执行结果较少，使用ProcessPoolExecutor的部分结果")
                            self.all_results = backup_results

                    parallel_success = True

                except Exception as e:
                    print(f"线程并行也失败: {e}")

            # 如果所有并行方式都失败，使用单核处理
            if not parallel_success:
                print("所有并行方式失败，使用单核处理模式")
                for i, combination in enumerate(tqdm(valid_combinations, desc="单核仿真")):
                    performance = self.run_simulation(combination)
                    result = {"combination": combination, "performance": performance}
                    self.all_results.append(result)

                    # 每完成100个组合保存一次中间结果
                    if (i + 1) % 100 == 0:
                        self._save_intermediate_results_single(i + 1, len(valid_combinations))
        else:
            # 单核处理
            print("使用单核处理模式")
            for i, combination in enumerate(tqdm(valid_combinations, desc="仿真进度")):
                performance = self.run_simulation(combination)
                result = {"combination": combination, "performance": performance}
                self.all_results.append(result)

                # 每完成100个组合保存一次中间结果
                if (i + 1) % 100 == 0:
                    self._save_intermediate_results_single(i + 1, len(valid_combinations))

        # 分析结果
        self._analyze_results()
        self._print_best_result()

    def _save_intermediate_results(self, batch_count: int):
        """保存中间结果"""
        temp_path = os.path.join(self.result_dir, f"intermediate_results_batch_{batch_count}.json")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        print(f"  中间结果已保存: {len(self.all_results):,} 个结果 -> {temp_path}")

    def _print_best_result(self):
        """打印最佳结果"""
        if not self.all_results:
            print("没有有效的仿真结果")
            return

        best_result = max(self.all_results, key=lambda x: x["performance"])
        print(f"\n最佳配置性能: {best_result['performance']:.2f} GB/s")
        print("最佳参数组合:")
        for param, value in best_result["combination"].items():
            print(f"  {param}: {value}")

    def _analyze_results(self):
        """分析所有结果，为每个参数生成统计数据"""
        print("\n分析结果数据...")

        # 为每个参数收集数据
        for param_name in ALL_PARAMS.keys():
            param_data = []

            for result in self.all_results:
                combination = result["combination"]
                performance = result["performance"]
                param_value = combination[param_name]

                param_data.append({"param_name": param_name, "param_value": param_value, "performance": performance, "full_combination": combination})

            self.param_results[param_name] = param_data

            # 统计信息
            performances = [d["performance"] for d in param_data]
            unique_values = set(d["param_value"] for d in param_data)

            print(f"{param_name}:")
            print(f"  取值范围: {min(unique_values)} - {max(unique_values)}")
            print(f"  数据点数: {len(param_data)}")
            print(f"  性能范围: {min(performances):.2f} - {max(performances):.2f} GB/s")

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
    config_path = "../config/config2.json"
    topo_type = "5x4"
    traffic_files = ["LLama2_AllReduce.txt"]
    traffic_weights = [1.0]

    # 创建优化器
    optimizer = FIFOExhaustiveOptimizer(config_path=config_path, topo_type=topo_type, traffic_files=traffic_files, traffic_weights=traffic_weights)

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
