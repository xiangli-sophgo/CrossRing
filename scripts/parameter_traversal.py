#!/usr/bin/env python3
"""
通用参数遍历脚本
支持1-3个参数的遍历，可自定义遍历范围

使用方法:
1. 修改main()函数中的参数配置区域
2. 运行: python parameter_traversal.py

参数配置说明:
- param_configs: 参数配置列表，每个元素包含name和range
- range格式:
  * 连续范围: "0,10" → [0, 1, 2, ..., 10]
  * 步长范围: "0,20,5" → [0, 5, 10, 15, 20]
  * 离散值: "[1,3,5,7,9]" → [1, 3, 5, 7, 9]

可视化特性:
- 单参数: 生成性能曲线图
- 双参数: 生成热力图和3D表面图
- 三参数: 生成多个2D切片热力图

热力图风格参考RB_slot_curve_plot.py，包含:
- 精美的配色方案
- 最优点标注
- 性能统计信息
- 高清图片输出
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import time
import argparse
import json
import traceback
import pandas as pd
import gc
import logging
import threading
from datetime import datetime
from typing import List, Tuple, Dict, Any, Union
from itertools import product

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import CrossRingConfig
from src.core.base_model import BaseModel

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler("parameter_traversal.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# 设置matplotlib中文字体支持
try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # 测试中文显示
    import matplotlib.font_manager as fm

    available_fonts = [f.name for f in fm.fontManager.ttflist if "SimHei" in f.name or "YaHei" in f.name]
    if available_fonts:
        logger.info(f"可用中文字体: {available_fonts[:3]}")
    else:
        logger.warning("未找到SimHei或Microsoft YaHei字体，将使用默认字体")
except Exception as e:
    logger.warning(f"中文字体设置失败: {e}")
    plt.rcParams["font.family"] = "DejaVu Sans"


class SimulationTimeoutError(Exception):
    pass


class SimulationRunner:
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.result = None
        self.exception = None
        self.completed = False

    def run_with_timeout(self, target_func, *args, **kwargs):
        """使用线程实现跨平台超时控制"""

        def target():
            try:
                self.result = target_func(*args, **kwargs)
                self.completed = True
            except Exception as e:
                self.exception = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            raise SimulationTimeoutError(f"仿真超时 ({self.timeout_seconds}秒)")

        if self.exception:
            raise self.exception

        if not self.completed:
            raise SimulationTimeoutError("仿真异常终止")

        return self.result


def parse_range(range_str: str) -> List[Union[int, float]]:
    """
    解析范围字符串
    支持格式:
    - "start,end": 生成[start, start+1, ..., end]
    - "start,end,step": 生成[start, start+step, ..., end]
    - "[val1,val2,val3]": 直接使用提供的值列表
    """
    range_str = range_str.strip()

    # 检查是否是列表格式
    if range_str.startswith("[") and range_str.endswith("]"):
        # 解析列表
        values_str = range_str[1:-1]
        values = []
        for val in values_str.split(","):
            val = val.strip()
            try:
                # 尝试解析为整数
                values.append(int(val))
            except ValueError:
                try:
                    # 尝试解析为浮点数
                    values.append(float(val))
                except ValueError:
                    raise ValueError(f"无法解析值: {val}")
        return values
    else:
        # 解析范围格式
        parts = range_str.split(",")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            return list(range(start, end + 1))
        elif len(parts) == 3:
            start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
            return list(range(start, end + 1, step))
        else:
            raise ValueError(f"无效的范围格式: {range_str}")


def run_single_simulation(config_params: Dict[str, Any], traffic_file: str, base_config_path: str, topo_type, traffic_path: str, result_save_path: str, timeout: int) -> float:
    """运行单次仿真"""
    sim = None
    try:
        # 加载配置
        if topo_type in ["4x4", "8x8", "12x12"]:
            config = CrossRingConfig(base_config_path)
            config.BURST = 4
            config.NUM_COL = int(topo_type.split("x")[1])  # 确保转换为整数
            row = int(topo_type.split("x")[0])
            config.NUM_NODE = row * config.NUM_COL * 2
            config.NUM_ROW = row * 2
            config.NUM_IP = row * config.NUM_COL
            config.RN_R_TRACKER_OSTD = 64
            config.RN_W_TRACKER_OSTD = 32
            config.RN_RDB_SIZE = config.RN_R_TRACKER_OSTD * config.BURST
            config.RN_WDB_SIZE = config.RN_W_TRACKER_OSTD * config.BURST
            config.NETWORK_FREQUENCY = 2
            config.SN_DDR_R_TRACKER_OSTD = 96
            config.SN_DDR_W_TRACKER_OSTD = 48
            config.SN_L2M_R_TRACKER_OSTD = 96
            config.SN_L2M_W_TRACKER_OSTD = 48
            config.SN_DDR_WDB_SIZE = config.SN_DDR_W_TRACKER_OSTD * config.BURST
            config.SN_L2M_WDB_SIZE = config.SN_L2M_W_TRACKER_OSTD * config.BURST
            config.DDR_R_LATENCY_original = 100
            config.DDR_W_LATENCY_original = 40
            config.L2M_R_LATENCY_original = 12
            config.L2M_W_LATENCY_original = 16
            config.IQ_CH_FIFO_DEPTH = 10
            config.EQ_CH_FIFO_DEPTH = 10
            config.IQ_OUT_FIFO_DEPTH_HORIZONTAL = 8
            config.IQ_OUT_FIFO_DEPTH_VERTICAL = 8
            config.IQ_OUT_FIFO_DEPTH_EQ = 8
            config.RB_OUT_FIFO_DEPTH = 8
            config.SN_TRACKER_RELEASE_LATENCY = 40

            config.IP_H2L_H_FIFO_DEPTH = 4
            config.IP_H2L_L_FIFO_DEPTH = 4

            config.TL_Etag_T2_UE_MAX = 8 // 2
            config.TL_Etag_T1_UE_MAX = 15 // 2
            config.TR_Etag_T2_UE_MAX = 12 // 2
            config.RB_IN_FIFO_DEPTH = 16 // 2
            config.TU_Etag_T2_UE_MAX = 8 // 2
            config.TU_Etag_T1_UE_MAX = 15 // 2
            config.TD_Etag_T2_UE_MAX = 12 // 2
            config.EQ_IN_FIFO_DEPTH = 16 // 2

            config.ITag_TRIGGER_Th_H = config.ITag_TRIGGER_Th_V = 80
            config.ITag_MAX_NUM_H = config.ITag_MAX_NUM_V = 1
            config.ETag_BOTHSIDE_UPGRADE = 0
            config.SLICE_PER_LINK = 6

            config.GDMA_RW_GAP = np.inf
            config.SDMA_RW_GAP = np.inf
            config.CHANNEL_SPEC = {
                "gdma": 2,
                "sdma": 0,
                "cdma": 0,
                "ddr": 2,
                "l2m": 0,
            }
        elif topo_type == "5x4":
            config = CrossRingConfig(base_config_path)
            config.NUM_COL = 4
            config.NUM_NODE = 40
            config.NUM_ROW = 10
            config.BURST = 4
            config.NUM_IP = 32
            config.NUM_DDR = 32
            config.NUM_L2M = 32
            config.NUM_GDMA = 32
            config.NUM_SDMA = 32
            config.NUM_RN = 32
            config.NUM_SN = 32
            # config.RN_R_TRACKER_OSTD = 64
            # config.RN_W_TRACKER_OSTD = 64
            # config.SN_DDR_R_TRACKER_OSTD = 64
            # config.SN_DDR_W_TRACKER_OSTD = 64
            # config.SN_L2M_R_TRACKER_OSTD = 64
            # config.SN_L2M_W_TRACKER_OSTD = 64
            # config.RN_RDB_SIZE = config.RN_R_TRACKER_OSTD * config.BURST
            # config.RN_WDB_SIZE = config.RN_W_TRACKER_OSTD * config.BURST
            # config.SN_DDR_WDB_SIZE = config.SN_DDR_W_TRACKER_OSTD * config.BURST
            # config.SN_L2M_WDB_SIZE = config.SN_L2M_W_TRACKER_OSTD * config.BURST
            # config.DDR_R_LATENCY_original = 40
            # config.DDR_R_LATENCY_VAR_original = 0
            # config.DDR_W_LATENCY_original = 0
            # config.L2M_R_LATENCY_original = 12
            # config.L2M_W_LATENCY_original = 16
            # config.IQ_CH_FIFO_DEPTH = 4
            # config.EQ_CH_FIFO_DEPTH = 4
            # config.IQ_OUT_FIFO_DEPTH_HORIZONTAL = 8
            # config.IQ_OUT_FIFO_DEPTH_VERTICAL = 8
            # config.IQ_OUT_FIFO_DEPTH_EQ = 8
            # config.RB_OUT_FIFO_DEPTH = 8
            # config.SN_TRACKER_RELEASE_LATENCY = 40
            # # config.GDMA_BW_LIMIT = 16
            # # config.CDMA_BW_LIMIT = 16
            # # config.DDR_BW_LIMIT = 128
            # config.ENABLE_CROSSPOINT_CONFLICT_CHECK = 0
            # config.ENABLE_IN_ORDER_EJECTION = 0

            # config.TL_Etag_T2_UE_MAX = 8
            # config.TL_Etag_T1_UE_MAX = 15
            # config.TR_Etag_T2_UE_MAX = 12
            # config.RB_IN_FIFO_DEPTH = 16
            # config.TU_Etag_T2_UE_MAX = 8
            # config.TU_Etag_T1_UE_MAX = 15
            # config.TD_Etag_T2_UE_MAX = 12
            # config.EQ_IN_FIFO_DEPTH = 16

            # config.ITag_TRIGGER_Th_H = config.ITag_TRIGGER_Th_V = 80
            # config.ITag_MAX_NUM_H = config.ITag_MAX_NUM_V = 1
            # config.ETag_BOTHSIDE_UPGRADE = 0
            # config.SLICE_PER_LINK = 8

            # config.GDMA_RW_GAP = np.inf
            # config.SDMA_RW_GAP = np.inf
            # config.CHANNEL_SPEC = {
            #     "gdma": 2,
            #     "sdma": 2,
            #     "cdma": 1,
            #     "ddr": 2,
            #     "l2m": 2,
            # }
            config.RN_R_TRACKER_OSTD = 64
            config.RN_W_TRACKER_OSTD = 32
            config.RN_RDB_SIZE = config.RN_R_TRACKER_OSTD * config.BURST
            config.RN_WDB_SIZE = config.RN_W_TRACKER_OSTD * config.BURST
            config.NETWORK_FREQUENCY = 2
            config.SN_DDR_R_TRACKER_OSTD = 96
            config.SN_DDR_W_TRACKER_OSTD = 48
            config.SN_L2M_R_TRACKER_OSTD = 96
            config.SN_L2M_W_TRACKER_OSTD = 48
            config.SN_DDR_WDB_SIZE = config.SN_DDR_W_TRACKER_OSTD * config.BURST
            config.SN_L2M_WDB_SIZE = config.SN_L2M_W_TRACKER_OSTD * config.BURST
            config.DDR_R_LATENCY_original = 100
            config.DDR_W_LATENCY_original = 40
            config.L2M_R_LATENCY_original = 12
            config.L2M_W_LATENCY_original = 16
            config.IQ_CH_FIFO_DEPTH = 10
            config.EQ_CH_FIFO_DEPTH = 10
            config.IQ_OUT_FIFO_DEPTH_HORIZONTAL = 8
            config.IQ_OUT_FIFO_DEPTH_VERTICAL = 8
            config.IQ_OUT_FIFO_DEPTH_EQ = 8
            config.RB_OUT_FIFO_DEPTH = 8
            config.SN_TRACKER_RELEASE_LATENCY = 40
            # config.CDMA_BW_LIMIT = 8
            # config.DDR_BW_LIMIT = 102
            # config.GDMA_BW_LIMIT = 102
            config.ENABLE_CROSSPOINT_CONFLICT_CHECK = 0

            config.TL_Etag_T2_UE_MAX = 8
            config.TL_Etag_T1_UE_MAX = 15
            config.TR_Etag_T2_UE_MAX = 12
            config.RB_IN_FIFO_DEPTH = 16
            config.TU_Etag_T2_UE_MAX = 8
            config.TU_Etag_T1_UE_MAX = 15
            config.TD_Etag_T2_UE_MAX = 12
            config.EQ_IN_FIFO_DEPTH = 16

            config.ITag_TRIGGER_Th_H = config.ITag_TRIGGER_Th_V = 80
            config.ITag_MAX_NUM_H = config.ITag_MAX_NUM_V = 1
            config.ETag_BOTHSIDE_UPGRADE = 0
            config.SLICE_PER_LINK = 8

            config.GDMA_RW_GAP = np.inf
            config.SDMA_RW_GAP = np.inf
            config.CHANNEL_SPEC = {
                "gdma": 2,
                "sdma": 0,
                "cdma": 0,
                "ddr": 2,
                "l2m": 0,
            }

        # 更新参数
        for param_name, param_value in config_params.items():
            # 特殊处理 IN_FIFO_DEPTH 参数：同时设置 RB_IN_FIFO_DEPTH 和 EQ_IN_FIFO_DEPTH，并按比例调整相关参数
            if param_name == "IN_FIFO_DEPTH":
                if isinstance(param_value, (int, float)) and param_value > 0:
                    # 设置两个FIFO深度参数为相同值
                    config.RB_IN_FIFO_DEPTH = int(param_value)
                    config.EQ_IN_FIFO_DEPTH = int(param_value)

                    # 计算比例乘数（基准值为16）
                    ratio_multiplier = param_value / 16.0

                    # 按比例调整相关参数，确保最小值为1，并且T1比对应的T2至少大1
                    config.TL_Etag_T2_UE_MAX = max(1, int(8 * ratio_multiplier))
                    config.TL_Etag_T1_UE_MAX = max(config.TL_Etag_T2_UE_MAX + 1, int(15 * ratio_multiplier))
                    config.TR_Etag_T2_UE_MAX = max(1, int(12 * ratio_multiplier))
                    config.TU_Etag_T2_UE_MAX = max(1, int(8 * ratio_multiplier))
                    config.TU_Etag_T1_UE_MAX = max(config.TU_Etag_T2_UE_MAX + 1, int(15 * ratio_multiplier))
                    config.TD_Etag_T2_UE_MAX = max(1, int(12 * ratio_multiplier))

                    logger.info(f"IN_FIFO_DEPTH={param_value}, 比例乘数={ratio_multiplier:.3f}")
                    logger.info(
                        f"调整后的参数: TL_T2={config.TL_Etag_T2_UE_MAX}, TL_T1={config.TL_Etag_T1_UE_MAX}, TR_T2={config.TR_Etag_T2_UE_MAX}, TU_T2={config.TU_Etag_T2_UE_MAX}, TU_T1={config.TU_Etag_T1_UE_MAX}, TD_T2={config.TD_Etag_T2_UE_MAX}"
                    )
                else:
                    logger.warning(f"IN_FIFO_DEPTH 参数值无效: {param_value}")
            elif hasattr(config, param_name):
                # 普通参数的处理
                if isinstance(param_value, (int, float)):
                    setattr(config, param_name, param_value)
                else:
                    logger.warning(f"参数 {param_name} 的值 {param_value} 不是数值类型")
            else:
                logger.warning(f"配置中不存在参数: {param_name}")

        # 创建仿真实例
        sim = BaseModel(
            model_type="REQ_RSP",
            config=config,
            topo_type=config.TOPO_TYPE,
            traffic_file_path=traffic_path,
            traffic_config=traffic_file,
            result_save_path=result_save_path,
            verbose=1,
            print_trace=0,
            plot_link_state=0,
            plot_flow_fig=0,
            plot_RN_BW_fig=0,
        )

        # 运行仿真
        def run_simulation():
            sim.initial()
            sim.end_time = 6000
            sim.print_interval = 2000
            sim.run()
            return sim.get_results()  # 返回完整的结果字典

        runner = SimulationRunner(timeout)
        results = runner.run_with_timeout(run_simulation)

        return results

    except Exception as e:
        logger.error(f"仿真失败: {e}")
        logger.error(f"参数配置: {config_params}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        return {}  # 返回空字典而不是0
    finally:
        if sim is not None:
            del sim
        gc.collect()


def run_parameter_combination(
    params_dict: Dict[str, Any],
    traffic_files: List[str],
    traffic_weights: List[float],
    base_config_path: str,
    topo_type,
    traffic_path: str,
    result_save_path: str,
    repeats: int,
    timeout: int,
) -> Dict[str, Any]:
    """运行一个参数组合的所有仿真"""
    logger.info(f"测试参数组合: {params_dict}")

    all_results = params_dict.copy()
    all_bw_means = []
    all_sim_results = {}  # 存储所有仿真结果

    for traffic_file in traffic_files:
        traffic_name = traffic_file[:-4]
        traffic_results_list = []
        bw_list = []

        for rpt in range(repeats):
            sim_results = run_single_simulation(params_dict, traffic_file, base_config_path, topo_type, traffic_path, result_save_path, timeout)

            if isinstance(sim_results, dict) and sim_results:
                # 如果返回的是完整的结果字典
                traffic_results_list.append(sim_results)
                bw = sim_results.get("mixed_avg_weighted_bw", 0)
                bw_list.append(bw)
            else:
                # 如果返回的是单个带宽值（旧版本兼容）
                bw_list.append(sim_results if isinstance(sim_results, (int, float)) else 0)
                traffic_results_list.append({"mixed_avg_weighted_bw": sim_results if isinstance(sim_results, (int, float)) else 0})

        # 计算统计值
        bw_mean = float(np.mean(bw_list))
        bw_std = float(np.std(bw_list))
        all_bw_means.append(bw_mean)

        # 保存基本统计信息
        all_results[f"bw_mean_{traffic_name}"] = bw_mean
        all_results[f"bw_std_{traffic_name}"] = bw_std

        # 如果有完整的仿真结果，计算所有指标的统计值
        if traffic_results_list and isinstance(traffic_results_list[0], dict):
            # 获取所有可能的结果键
            all_keys = set()
            for result in traffic_results_list:
                if isinstance(result, dict):
                    all_keys.update(result.keys())

            # 为每个指标计算统计值
            for key in all_keys:
                if key != "mixed_avg_weighted_bw":  # 这个已经处理过了
                    values = []
                    for result in traffic_results_list:
                        if isinstance(result, dict) and key in result:
                            val = result[key]
                            # 特殊处理嵌套字典，如 circling_eject_stats
                            if key == "circling_eject_stats" and isinstance(val, dict):
                                # 展开 circling_eject_stats 字典
                                for sub_key, sub_val in val.items():
                                    if isinstance(sub_val, dict):
                                        for sub_sub_key, sub_sub_val in sub_val.items():
                                            full_key = f"{key}_{sub_key}_{sub_sub_key}"
                                            sub_values = []
                                            for res in traffic_results_list:
                                                if (
                                                    isinstance(res, dict)
                                                    and key in res
                                                    and isinstance(res[key], dict)
                                                    and sub_key in res[key]
                                                    and isinstance(res[key][sub_key], dict)
                                                    and sub_sub_key in res[key][sub_key]
                                                ):
                                                    sub_sub_val = res[key][sub_key][sub_sub_key]
                                                    if isinstance(sub_sub_val, (int, float)):
                                                        sub_values.append(sub_sub_val)
                                            if sub_values:
                                                all_results[f"{full_key}_mean_{traffic_name}"] = float(np.mean(sub_values))
                                                all_results[f"{full_key}_std_{traffic_name}"] = float(np.std(sub_values))
                                                all_results[f"{full_key}_max_{traffic_name}"] = float(np.max(sub_values))
                                                all_results[f"{full_key}_min_{traffic_name}"] = float(np.min(sub_values))
                            elif isinstance(val, (int, float)):
                                values.append(val)

                    if values:
                        all_results[f"{key}_mean_{traffic_name}"] = float(np.mean(values))
                        all_results[f"{key}_std_{traffic_name}"] = float(np.std(values))
                        all_results[f"{key}_max_{traffic_name}"] = float(np.max(values))
                        all_results[f"{key}_min_{traffic_name}"] = float(np.min(values))

        # 保存原始结果用于详细分析
        all_sim_results[traffic_name] = traffic_results_list

    # 计算综合指标
    weighted_bw = sum(bw * weight for bw, weight in zip(all_bw_means, traffic_weights))
    min_bw = min(all_bw_means) if all_bw_means else 0
    bw_variance = np.var(all_bw_means) if len(all_bw_means) > 1 else 0

    all_results.update(
        {
            "weighted_bw": weighted_bw,
            "min_bw": min_bw,
            "bw_variance": bw_variance,
        }
    )

    # 保存详细的仿真结果到单独的字段
    all_results["detailed_sim_results"] = all_sim_results

    logger.info(f"参数组合完成: {params_dict}, 加权带宽: {weighted_bw:.3f}")

    return all_results


def create_visualizations(results_df: pd.DataFrame, param_names: List[str], save_dir: str):
    """创建可视化图表"""
    os.makedirs(save_dir, exist_ok=True)

    logger.info("开始生成可视化图表...")

    # 检查数据是否为空
    if results_df.empty:
        logger.warning("结果数据为空，无法生成可视化图表")
        return

    try:
        if len(param_names) == 1:
            # 单参数折线图
            logger.info("生成单参数曲线图...")
            create_1d_plot(results_df, param_names[0], save_dir)

        elif len(param_names) == 2:
            # 双参数：热力图 + 两条曲线图
            logger.info("生成双参数热力图...")
            create_2d_heatmap(results_df, param_names[0], param_names[1], save_dir)

            logger.info("生成参数影响曲线图...")
            create_2d_curve_plots(results_df, param_names[0], param_names[1], save_dir)

            logger.info("生成3D表面图...")
            create_3d_surface_plot(results_df, param_names[0], param_names[1], save_dir)

        elif len(param_names) == 3:
            # 三参数多图展示
            logger.info("生成三参数可视化...")
            create_3d_visualizations(results_df, param_names, save_dir)

        logger.info("所有可视化图表生成完成")
    except Exception as e:
        logger.error(f"生成可视化图表时出错: {e}")
        logger.error(f"错误详情: {traceback.format_exc()}")


def create_1d_plot(df: pd.DataFrame, param_name: str, save_dir: str):
    """创建单参数折线图，参考RB_slot_curve_plot.py风格"""
    # 设置样式和字体
    sns.set_style("whitegrid")
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(14, 8))

    # 按参数值排序并计算统计数据
    param_grouped = df.groupby(param_name)["weighted_bw"].agg(["mean", "std", "max", "min", "count"])

    # 绘制主曲线 - 使用与RB_slot_curve_plot.py相同的风格
    plt.plot(param_grouped.index, param_grouped["mean"], marker="o", linewidth=3, markersize=8, color="#2E86AB", label="平均性能")

    # 如果有多个数据点，添加标准差区域
    if param_grouped["count"].max() > 1:
        plt.fill_between(param_grouped.index, param_grouped["mean"] - param_grouped["std"], param_grouped["mean"] + param_grouped["std"], alpha=0.3, color="#2E86AB", label="±1标准差")

    # 绘制最大值和最小值曲线
    if not param_grouped["max"].equals(param_grouped["mean"]):
        plt.plot(param_grouped.index, param_grouped["max"], "g--", linewidth=2, marker="s", markersize=6, label="最大值")
    if not param_grouped["min"].equals(param_grouped["mean"]):
        plt.plot(param_grouped.index, param_grouped["min"], "r--", linewidth=2, marker="^", markersize=6, label="最小值")

    # 设置标签和标题 - 与RB_slot_curve_plot.py风格一致
    plt.xlabel(f"{param_name}", fontsize=14, fontweight="bold")
    plt.ylabel("加权带宽 (GB/s)", fontsize=14, fontweight="bold")
    plt.title(f"{param_name} 性能关系曲线", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.7)
    plt.legend(fontsize=12)

    # 标注最优点
    if not param_grouped.empty and len(param_grouped) > 0:
        best_param_val = param_grouped["mean"].idxmax()  # idxmax直接返回索引值
        best_performance = param_grouped.loc[best_param_val, "mean"]

        plt.scatter(best_param_val, best_performance, c="red", s=300, marker="*", zorder=10, edgecolors="white", linewidth=2)

        # 添加最优点标注
        plt.annotate(
            f"最优配置\n{param_name}={best_param_val}\n性能: {best_performance:.3f} GB/s",
            xy=(best_param_val, best_performance),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8, edgecolor="red"),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
            fontsize=12,
            fontweight="bold",
            color="red",
        )

    # 添加性能统计信息
    stats_text = f"""性能统计:
最大值: {param_grouped['max'].max():.3f} GB/s
平均值: {param_grouped['mean'].mean():.3f} GB/s
最小值: {param_grouped['min'].min():.3f} GB/s
测试范围: {param_grouped.index.min()} - {param_grouped.index.max()}"""

    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.9, edgecolor="navy"),
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()

    # 保存图片
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plot_path = os.path.join(save_dir, f"{param_name}_curve_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    logger.info(f"参数曲线图已保存: {plot_path}")
    return plot_path


def create_2d_heatmap(df: pd.DataFrame, param1: str, param2: str, save_dir: str):
    """创建双参数热力图，参考RB_slot_curve_plot.py的风格"""
    import seaborn as sns

    # 设置seaborn样式和字体
    sns.set_style("whitegrid")
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建透视表
    pivot_table = df.pivot_table(values="weighted_bw", index=param2, columns=param1, aggfunc="mean")

    # 创建图表
    plt.figure(figsize=(16, 12))

    # 使用自定义配色方案，与RB_slot_curve_plot.py保持一致
    custom_cmap = sns.color_palette("RdYlBu_r", as_cmap=True)

    # 绘制热力图
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".3f",
        cmap=custom_cmap,
        center=None,
        square=False,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "加权带宽 (GB/s)", "shrink": 0.8, "aspect": 30},
        annot_kws={"size": 10, "weight": "bold"},
    )

    # 设置标题和标签，与RB_slot_curve_plot.py风格一致
    plt.title(f"{param1} vs {param2} 配置性能热力图", fontsize=20, fontweight="bold", pad=25)
    plt.xlabel(f"{param1}", fontsize=16, fontweight="bold", labelpad=15)
    plt.ylabel(f"{param2}", fontsize=16, fontweight="bold", labelpad=15)

    # 设置刻度标签样式
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # 找到并标注最优配置
    best_idx = df["weighted_bw"].idxmax()
    best_config = df.loc[best_idx]
    best_p1 = best_config[param1]
    best_p2 = best_config[param2]
    best_perf = best_config["weighted_bw"]

    # 在热力图上标记最优点
    try:
        p1_pos = list(pivot_table.columns).index(best_p1)
        p2_pos = list(pivot_table.index).index(best_p2)

        # 添加最优点矩形框
        ax.add_patch(plt.Rectangle((p1_pos, p2_pos), 1, 1, fill=False, edgecolor="red", linewidth=4, linestyle="--"))

        # 添加箭头和文本标注
        ax.annotate(
            f"最优配置\n{param1}={best_p1}, {param2}={best_p2}\n{best_perf:.3f} GB/s",
            xy=(p1_pos + 0.5, p2_pos + 0.5),
            xytext=(p1_pos + 2, p2_pos + 2),
            fontsize=12,
            fontweight="bold",
            color="red",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8, edgecolor="red"),
            arrowprops=dict(arrowstyle="->", color="red", lw=2, connectionstyle="arc3,rad=0.2"),
        )
    except (ValueError, IndexError) as e:
        logger.warning(f"无法标注最优点: {e}")

    # 添加统计信息框，与RB_slot_curve_plot.py风格一致
    stats_text = f"""性能统计:
最大值: {pivot_table.values.max():.3f} GB/s
平均值: {pivot_table.values.mean():.3f} GB/s  
最小值: {pivot_table.values.min():.3f} GB/s
标准差: {pivot_table.values.std():.3f} GB/s
变异系数: {(pivot_table.values.std()/pivot_table.values.mean()*100):.1f}%"""

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.9, edgecolor="navy"),
        fontsize=11,
        fontweight="bold",
    )

    # 调整布局
    plt.tight_layout()

    # 保存图片
    timestamp = datetime.now().strftime("%m%d_%H%M")
    heatmap_path = os.path.join(save_dir, f"{param1}_vs_{param2}_heatmap_{timestamp}.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    logger.info(f"热力图已保存: {heatmap_path}")
    return heatmap_path


def create_2d_curve_plots(df: pd.DataFrame, param1: str, param2: str, save_dir: str):
    """创建双参数曲线图，参考RB_slot_curve_plot.py风格"""
    logger.info(f"分析{param1}和{param2}的影响...")

    # 设置字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 找出全局最优配置
    best_idx = df["weighted_bw"].idxmax()
    best_config = df.loc[best_idx]
    best_p1 = best_config[param1]
    best_p2 = best_config[param2]

    # 分析param1的影响：对每个param1值，找出所有param2值中的最大值
    param1_stats = []
    param1_values = sorted(df[param1].unique())

    for p1_val in param1_values:
        p1_data = df[df[param1] == p1_val]
        if len(p1_data) > 0:
            max_bw = p1_data["weighted_bw"].max()
            best_p2_for_p1 = p1_data.loc[p1_data["weighted_bw"].idxmax(), param2]
            param1_stats.append({param1: p1_val, "max_weighted_bw": max_bw, f"best_{param2}": best_p2_for_p1})

    param1_df = pd.DataFrame(param1_stats)

    # 分析param2的影响：对每个param2值，找出所有param1值中的最大值
    param2_stats = []
    param2_values = sorted(df[param2].unique())

    for p2_val in param2_values:
        p2_data = df[df[param2] == p2_val]
        if len(p2_data) > 0:
            max_bw = p2_data["weighted_bw"].max()
            best_p1_for_p2 = p2_data.loc[p2_data["weighted_bw"].idxmax(), param1]
            param2_stats.append({param2: p2_val, "max_weighted_bw": max_bw, f"best_{param1}": best_p1_for_p2})

    param2_df = pd.DataFrame(param2_stats)

    # 设置绘图样式
    sns.set_style("whitegrid")

    # 创建组合图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 图1: param1对性能的影响（每个param1对应所有param2中的最大值）
    if not param1_df.empty:
        ax1.plot(param1_df[param1], param1_df["max_weighted_bw"], marker="o", linewidth=3, markersize=8, color="#2E86AB", label="最大性能")
        ax1.set_xlabel(f"{param1}", fontsize=14, fontweight="bold")
        ax1.set_ylabel("加权带宽 (GB/s)", fontsize=14, fontweight="bold")
        ax1.set_title(f"{param1}对性能的影响\n(每个{param1}对应所有{param2}中的最大值)", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.7)
        ax1.legend(fontsize=12)

        # 标注最优点
        best_p1_idx = param1_df["max_weighted_bw"].idxmax()
        best_p1_row = param1_df.loc[best_p1_idx]
        ax1.scatter(best_p1_row[param1], best_p1_row["max_weighted_bw"], c="red", s=300, marker="*", zorder=10, edgecolors="white", linewidth=2)
        ax1.annotate(
            f'最优: {best_p1_row[param1]}\n{best_p1_row["max_weighted_bw"]:.3f}',
            xy=(best_p1_row[param1], best_p1_row["max_weighted_bw"]),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
            fontsize=10,
            fontweight="bold",
        )

    # 图2: param2对性能的影响（每个param2对应所有param1中的最大值）
    if not param2_df.empty:
        ax2.plot(param2_df[param2], param2_df["max_weighted_bw"], marker="s", linewidth=3, markersize=8, color="#F18F01", label="最大性能")
        ax2.set_xlabel(f"{param2}", fontsize=14, fontweight="bold")
        ax2.set_ylabel("加权带宽 (GB/s)", fontsize=14, fontweight="bold")
        ax2.set_title(f"{param2}对性能的影响\n(每个{param2}对应所有{param1}中的最大值)", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.7)
        ax2.legend(fontsize=12)

        # 标注最优点
        best_p2_idx = param2_df["max_weighted_bw"].idxmax()
        best_p2_row = param2_df.loc[best_p2_idx]
        ax2.scatter(best_p2_row[param2], best_p2_row["max_weighted_bw"], c="red", s=300, marker="*", zorder=10, edgecolors="white", linewidth=2)
        ax2.annotate(
            f'最优: {best_p2_row[param2]}\n{best_p2_row["max_weighted_bw"]:.3f}',
            xy=(best_p2_row[param2], best_p2_row["max_weighted_bw"]),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
            fontsize=10,
            fontweight="bold",
        )

    # 图3: 显示达到最大值时的最佳配对参数
    if not param1_df.empty:
        ax3.scatter(param1_df[param1], param1_df[f"best_{param2}"], s=80, color="#2E86AB", alpha=0.7, label=f"最佳{param2}值")
        ax3.set_xlabel(f"{param1}", fontsize=14, fontweight="bold")
        ax3.set_ylabel(f"最佳{param2}", fontsize=14, fontweight="bold")
        ax3.set_title(f"达到最大性能时的最佳{param2}配置", fontsize=14, fontweight="bold")
        ax3.grid(True, alpha=0.7)
        ax3.legend(fontsize=12)

    # 图4: 显示达到最大值时的最佳配对参数
    if not param2_df.empty:
        ax4.scatter(param2_df[param2], param2_df[f"best_{param1}"], s=80, color="#F18F01", alpha=0.7, label=f"最佳{param1}值")
        ax4.set_xlabel(f"{param2}", fontsize=14, fontweight="bold")
        ax4.set_ylabel(f"最佳{param1}", fontsize=14, fontweight="bold")
        ax4.set_title(f"达到最大性能时的最佳{param1}配置", fontsize=14, fontweight="bold")
        ax4.grid(True, alpha=0.7)
        ax4.legend(fontsize=12)

    plt.tight_layout()

    # 保存图片
    timestamp = datetime.now().strftime("%m%d_%H%M")
    curve_path = os.path.join(save_dir, f"{param1}_vs_{param2}_curves_{timestamp}.png")
    plt.savefig(curve_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    logger.info(f"参数影响曲线图已保存: {curve_path}")

    # 输出分析摘要
    logger.info("参数影响分析摘要:")
    if not param1_df.empty:
        best_p1_config = param1_df.loc[param1_df["max_weighted_bw"].idxmax()]
        logger.info(f"  {param1}最优值: {best_p1_config[param1]} (性能: {best_p1_config['max_weighted_bw']:.3f}, 最佳{param2}: {best_p1_config[f'best_{param2}']})")

    if not param2_df.empty:
        best_p2_config = param2_df.loc[param2_df["max_weighted_bw"].idxmax()]
        logger.info(f"  {param2}最优值: {best_p2_config[param2]} (性能: {best_p2_config['max_weighted_bw']:.3f}, 最佳{param1}: {best_p2_config[f'best_{param1}']})")

    return curve_path


def create_3d_surface_plot(df: pd.DataFrame, param1: str, param2: str, save_dir: str):
    """创建3D表面图，参考RB_slot_curve_plot.py风格"""
    from mpl_toolkits.mplot3d import Axes3D

    # 设置字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建网格数据
    pivot_table = df.pivot_table(values="weighted_bw", index=param2, columns=param1, aggfunc="mean")

    X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    Z = pivot_table.values

    # 创建3D图
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制表面
    surf = ax.plot_surface(X, Y, Z, cmap="RdYlBu_r", alpha=0.9, linewidth=0.5, edgecolors="black")

    # 设置标签
    ax.set_xlabel(f"{param1}", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{param2}", fontsize=12, fontweight="bold")
    ax.set_zlabel("加权带宽 (GB/s)", fontsize=12, fontweight="bold")
    ax.set_title(f"{param1} vs {param2} 3D 性能表面", fontsize=16, fontweight="bold")

    # 添加颜色条
    fig.colorbar(surf, shrink=0.6, aspect=30, label="加权带宽 (GB/s)")

    # 标注最优点
    best_idx = df["weighted_bw"].idxmax()
    best_config = df.loc[best_idx]
    best_p1 = best_config[param1]
    best_p2 = best_config[param2]
    best_perf = best_config["weighted_bw"]

    ax.scatter([best_p1], [best_p2], [best_perf], color="red", s=200, marker="*", label=f"最优: ({best_p1},{best_p2}) = {best_perf:.3f}")
    ax.legend()

    # 保存图片
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plot_path = os.path.join(save_dir, f"{param1}_vs_{param2}_3d_surface_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    logger.info(f"3D表面图已保存: {plot_path}")
    return plot_path


def create_3d_visualizations(df: pd.DataFrame, param_names: List[str], save_dir: str):
    """创建三参数可视化（多个2D切片和3D图）"""
    param1, param2, param3 = param_names

    # 为第三个参数的每个值创建一个2D热力图
    param3_values = sorted(df[param3].unique())

    for p3_val in param3_values:
        df_slice = df[df[param3] == p3_val]
        if len(df_slice) > 0:
            slice_dir = os.path.join(save_dir, f"{param3}_{p3_val}")
            os.makedirs(slice_dir, exist_ok=True)
            create_2d_heatmap(df_slice, param1, param2, slice_dir)

    # 创建总体3D表面图（使用平均值）
    df_avg = df.groupby([param1, param2])["weighted_bw"].mean().reset_index()
    create_3d_surface_plot(df_avg, param1, param2, save_dir)


def main():
    """
    主函数 - 在这里设置要遍历的参数
    """
    # ==================== 参数配置区域 ====================

    # 参数设置：可以配置1-3个参数
    param_configs = [
        # 示例：使用 IN_FIFO_DEPTH 同时遍历 RB_IN_FIFO_DEPTH 和 EQ_IN_FIFO_DEPTH，并按比例调整相关参数
        # {"name": "IN_FIFO_DEPTH", "range": "8,32,4"},  # 从8到32，步长为4
        # {"name": "IN_FIFO_DEPTH", "range": "[28, 22, 16, 14, 12, 10, 8, 6, 4, 2]"},
        {"name": "SLICE_PER_LINK", "range": "5, 20"},
        # {"name": "SLICE_PER_LINK", "range": "[19, 20]"},
        # 其他参数配置示例：
        # {"name": "IQ_OUT_FIFO_DEPTH_VERTICAL", "range": "1,8"},
        # {"name": "SLICE_PER_LINK", "range": "17,20"},
    ]

    # 文件路径配置
    config_path = "../config/config2.json"
    traffic_path = "../traffic/traffic0730"
    # traffic_path = "../traffic/0617"
    # 仿真配置
    traffic_files = [
        "R_4x4.txt",
        # "W_5x4_CR_v1.0.2.txt",
        # "W_12x12.txt",
        # "LLama2_AllReduce.txt",
    ]
    topo_type = "4x4"
    # topo_type = "8x8"
    # topo_type = "12x12"
    # topo_type = "5x4"
    traffic_weights = [1]
    repeats = 1
    timeout = 300000

    # 输出配置
    custom_output_name = None  # 设置为None则自动生成，或指定名称如 "my_test"

    # ==================== 参数配置区域结束 ====================

    # 解析参数配置
    param_names = []
    param_ranges = []

    for param_config in param_configs:
        if param_config["name"] and param_config["range"]:
            param_names.append(param_config["name"])
            param_ranges.append(parse_range(param_config["range"]))

    if not param_names:
        logger.error("至少需要配置一个参数")
        return

    # 验证traffic配置
    if len(traffic_files) != len(traffic_weights):
        logger.error("Traffic文件数量和权重数量不匹配")
        return

    if abs(sum(traffic_weights) - 1.0) > 1e-6:
        logger.error("权重总和必须为1")
        return

    # 设置输出目录
    if custom_output_name:
        output_dir = f"../Result/ParameterTraversal/{custom_output_name}"
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        params_str = "_".join(param_names)
        output_dir = f"../Result/ParameterTraversal/{params_str}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # 记录配置
    config_info = {
        "parameters": param_names,
        "ranges": {name: range_vals for name, range_vals in zip(param_names, param_ranges)},
        "traffic_files": traffic_files,
        "traffic_weights": traffic_weights,
        "topo_type": topo_type,
        "repeats": repeats,
        "timeout": timeout,
        "config_path": config_path,
        "traffic_path": traffic_path,
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("参数遍历开始")
    logger.info("=" * 60)
    logger.info(f"参数: {param_names}")
    logger.info(f"范围: {param_ranges}")
    logger.info(f"总组合数: {np.prod([len(r) for r in param_ranges])}")
    logger.info(f"Traffic文件: {traffic_files}")
    logger.info(f"权重: {traffic_weights}")
    logger.info(f"重复次数: {repeats}")
    logger.info(f"超时时间: {timeout}秒")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)

    # 生成所有参数组合
    all_combinations = list(product(*param_ranges))
    total_combinations = len(all_combinations)

    # 运行所有组合
    results = []
    for i, combination in enumerate(all_combinations):
        params_dict = {name: value for name, value in zip(param_names, combination)}

        logger.info(f"进度: {i+1}/{total_combinations} - 测试参数: {params_dict}")

        result = run_parameter_combination(params_dict, traffic_files, traffic_weights, config_path, topo_type, traffic_path, output_dir, repeats, timeout)

        results.append(result)

        # 定期保存中间结果
        if (i + 1) % 10 == 0 or i == total_combinations - 1:
            # 保存主要结果（不包括详细仿真结果，避免CSV过于复杂）
            results_for_csv = []
            for r in results:
                r_copy = r.copy()
                if "detailed_sim_results" in r_copy:
                    del r_copy["detailed_sim_results"]  # 从CSV中移除详细结果
                results_for_csv.append(r_copy)

            df = pd.DataFrame(results_for_csv)
            df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
            logger.info(f"中间结果已保存 ({i+1}/{total_combinations})")

            # 保存详细的仿真结果为JSON格式
            detailed_results_path = os.path.join(output_dir, "detailed_results.json")
            with open(detailed_results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"详细结果已保存到JSON: {detailed_results_path}")

    # 最终结果分析 - 为CSV和可视化准备数据（移除详细结果）
    results_for_analysis = []
    for r in results:
        r_copy = r.copy()
        if "detailed_sim_results" in r_copy:
            del r_copy["detailed_sim_results"]  # 从分析中移除详细结果
        results_for_analysis.append(r_copy)

    results_df = pd.DataFrame(results_for_analysis)

    # 找出最优配置
    if not results_df.empty and "weighted_bw" in results_df.columns:
        best_idx = results_df["weighted_bw"].idxmax()
        best_config = results_df.loc[best_idx]

        logger.info("=" * 60)
        logger.info("最优配置:")
        for param in param_names:
            logger.info(f"  {param}: {best_config[param]}")
        logger.info(f"  加权带宽: {best_config['weighted_bw']:.3f}")

        # 显示更多详细信息
        logger.info("详细性能指标:")
        for col in results_df.columns:
            if col.endswith("_mean_R_4x4") or col.endswith("_mean_W_4x4"):
                if col in best_config:
                    logger.info(f"  {col}: {best_config[col]:.3f}")

        logger.info("=" * 60)

        # 创建可视化
        logger.info("生成可视化...")
        create_visualizations(results_df, param_names, os.path.join(output_dir, "visualizations"))

        # 生成报告
        report_path = os.path.join(output_dir, "report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("参数遍历报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"参数: {param_names}\n")
            f.write(f"总测试组合数: {total_combinations}\n")
            f.write("\n参数范围:\n")
            for i, param in enumerate(param_names):
                f.write(f"  {param}: {param_ranges[i]}\n")
            f.write("\n最优配置:\n")
            for param in param_names:
                f.write(f"  {param}: {best_config[param]}\n")
            f.write(f"  加权带宽: {best_config['weighted_bw']:.3f}\n")

            # 添加最优配置的详细指标
            f.write("\n最优配置详细性能指标:\n")
            for col in sorted(results_df.columns):
                if ("_mean_" in col or "_max_" in col or "_min_" in col) and col in best_config:
                    f.write(f"  {col}: {best_config[col]:.4f}\n")

            f.write("\nTop 10 配置:\n")
            top10 = results_df.nlargest(10, "weighted_bw")
            for i, (idx, row) in enumerate(top10.iterrows(), 1):
                f.write(f"{i}. ")
                for param in param_names:
                    f.write(f"{param}={row[param]} ")
                f.write(f"BW={row['weighted_bw']:.3f}\n")

        # 保存最终的完整结果
        final_csv_path = os.path.join(output_dir, "final_results.csv")
        results_df.to_csv(final_csv_path, index=False)

        final_json_path = os.path.join(output_dir, "final_detailed_results.json")
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"报告已生成: {report_path}")
    else:
        logger.error("无有效结果数据")

    logger.info("参数遍历完成!")


if __name__ == "__main__":
    main()
