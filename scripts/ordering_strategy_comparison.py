"""
保序策略对比脚本
对比不同保序模式、保序粒度和保序通道的性能影响

运行方式:
python ordering_strategy_comparison.py --traffic_path ../traffic/DeepSeek_0616/step6_ch_map/ --max_workers 8
"""

from src.traffic_process import step1_flatten, step2_hash_addr2node, step6_map_to_ch
from src.noc import *
import os
from config.config import CrossRingConfig
import csv
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
import logging
from datetime import datetime

logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


# 定义9种配置组合
STRATEGY_CONFIGS = [
    # Mode 0: 无保序 - 1种配置
    {"mode": 0, "granularity": 0, "channels": [], "name": "M0_NoOrder", "desc": "无保序（基准对照）"},
    # Mode 1: 单侧下环 (TL/TU固定) - 4种组合
    {"mode": 1, "granularity": 0, "channels": ["REQ"], "name": "M1_IP_REQ", "desc": "单侧下环+IP层级+仅REQ保序"},
    {"mode": 1, "granularity": 0, "channels": ["REQ", "RSP", "DATA"], "name": "M1_IP_ALL", "desc": "单侧下环+IP层级+全通道保序"},
    {"mode": 1, "granularity": 1, "channels": ["REQ"], "name": "M1_Node_REQ", "desc": "单侧下环+节点层级+仅REQ保序"},
    {"mode": 1, "granularity": 1, "channels": ["REQ", "RSP", "DATA"], "name": "M1_Node_ALL", "desc": "单侧下环+节点层级+全通道保序"},
    # Mode 2: 双侧下环 (白名单配置) - 4种组合
    {"mode": 2, "granularity": 0, "channels": ["REQ"], "name": "M2_IP_REQ", "desc": "双侧下环+IP层级+仅REQ保序"},
    {"mode": 2, "granularity": 0, "channels": ["REQ", "RSP", "DATA"], "name": "M2_IP_ALL", "desc": "双侧下环+IP层级+全通道保序"},
    {"mode": 2, "granularity": 1, "channels": ["REQ"], "name": "M2_Node_REQ", "desc": "双侧下环+节点层级+仅REQ保序"},
    {"mode": 2, "granularity": 1, "channels": ["REQ", "RSP", "DATA"], "name": "M2_Node_ALL", "desc": "双侧下环+节点层级+全通道保序"},
]


def run_single_simulation(sim_params):
    """运行单个仿真 - 针对特定流量文件和特定保序策略配置"""
    (config_path, traffic_path, model_type, file_name, strategy, result_save_base_path, output_csv) = sim_params

    try:
        print(f"[{strategy['name']}] 开始仿真: {file_name} (进程 {os.getpid()})")

        # 拓扑类型到配置文件的映射
        topo_config_map = {
            "3x3": r"../config/topologies/topo_3x3.yaml",
            "4x4": r"../config/topologies/topo_4x4.yaml",
            "5x2": r"../config/topologies/topo_5x2.yaml",
            "5x4": r"../config/topologies/topo_5x4.yaml",
            "6x5": r"../config/topologies/topo_6x5.yaml",
            "8x8": r"../config/topologies/topo_8x8.yaml",
        }

        # 默认拓扑类型
        topo_type = "5x4"

        # 根据拓扑类型选择配置文件
        actual_config_path = topo_config_map.get(topo_type, config_path)
        config = CrossRingConfig(actual_config_path)
        config.CROSSRING_VERSION = "V1"

        # 从配置文件获取拓扑类型
        topo_type = config.TOPO_TYPE if config.TOPO_TYPE else topo_type

        # ==================== 关键: 动态修改保序策略配置 ====================
        config.ORDERING_PRESERVATION_MODE = strategy["mode"]
        config.ORDERING_GRANULARITY = strategy["granularity"]
        config.IN_ORDER_PACKET_CATEGORIES = strategy["channels"]
        # ================================================================

        # 创建仿真实例
        sim: BaseModel = eval(f"{model_type}_model")(
            model_type=model_type,
            config=config,
            topo_type=topo_type,
            verbose=0,
        )

        # 配置流量调度器
        sim.setup_traffic_scheduler(
            traffic_file_path=traffic_path,
            traffic_chains=file_name,
        )

        # 配置结果分析 - 每个策略独立保存
        strategy_result_path = f"{result_save_base_path}/{strategy['name']}/{file_name[:-4]}/"

        sim.setup_result_analysis(
            result_save_path=strategy_result_path,
            plot_RN_BW_fig=1,
            flow_graph_interactive=1,
            fifo_utilization_heatmap=1,
            show_fig=0,
        )

        # 运行仿真
        sim.run_simulation(max_time=10000, print_interval=5000)

        # 获取结果
        results = sim.get_results()

        # 添加策略标识到结果中
        results["strategy"] = strategy["name"]
        results["mode"] = strategy["mode"]
        results["granularity"] = strategy["granularity"]
        results["channels"] = ",".join(strategy["channels"])
        results["strategy_desc"] = strategy["desc"]

        print(f"[{strategy['name']}] 完成仿真: {file_name} (进程 {os.getpid()})")
        return (file_name, strategy["name"], results, output_csv)

    except Exception:
        logging.exception(f"[{strategy['name']}] 仿真失败: {file_name}")
        return (file_name, strategy["name"], None, output_csv)


def save_results_to_csv(results_data):
    """保存仿真结果到CSV文件（线程安全）"""
    file_name, strategy_name, results, output_csv = results_data

    if results is None:
        print(f"[{strategy_name}] 跳过CSV写入: {file_name} (仿真错误)")
        return

    # 使用portalocker确保跨平台文件锁
    try:
        import portalocker

        use_portalocker = True
    except ImportError:
        print("警告: portalocker不可用，使用threading.Lock替代")
        use_portalocker = False

    csv_file_exists = os.path.isfile(output_csv)

    if use_portalocker:
        with open(output_csv, mode="a", newline="", encoding="utf-8-sig") as output_csv_file:
            # 锁定文件（跨平台）
            portalocker.lock(output_csv_file, portalocker.LOCK_EX)

            writer = csv.DictWriter(output_csv_file, fieldnames=results.keys())
            if not csv_file_exists:
                writer.writeheader()
            writer.writerow(results)

            # 文件关闭时自动解锁
    else:
        # 回退: 使用全局锁
        import threading

        if not hasattr(save_results_to_csv, "_lock"):
            save_results_to_csv._lock = threading.Lock()

        with save_results_to_csv._lock:
            with open(output_csv, mode="a", newline="", encoding="utf-8-sig") as output_csv_file:
                writer = csv.DictWriter(output_csv_file, fieldnames=results.keys())
                if not csv_file_exists:
                    writer.writeheader()
                writer.writerow(results)

    print(f"[{strategy_name}] CSV已保存: {file_name}")


def run_comparison_simulation(config_path, traffic_path, model_type, results_base_name, max_workers=None):
    """运行保序策略对比仿真"""

    # 获取所有流量文件
    all_files = os.listdir(traffic_path)
    file_names = [file for file in all_files if file.endswith(".txt")]

    if not file_names:
        print("未找到流量文件！")
        return

    print(f"找到 {len(file_names)} 个流量文件")
    print(f"将对每个文件运行 {len(STRATEGY_CONFIGS)} 种策略配置")
    print(f"总计 {len(file_names) * len(STRATEGY_CONFIGS)} 个仿真任务\n")

    # 显示策略配置
    print("=" * 80)
    print("保序策略配置:")
    print("=" * 80)
    for i, strategy in enumerate(STRATEGY_CONFIGS, 1):
        print(f"{i}. {strategy['name']}: {strategy['desc']}")
    print("=" * 80 + "\n")

    # 设置结果路径
    result_save_base_path = f"../Result/CrossRing/{model_type}/{results_base_name}/"

    # 为每个策略创建独立的CSV文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = os.path.join(r"../Result/Traffic_result_csv/", f"ordering_comparison_{timestamp}")
    os.makedirs(csv_dir, exist_ok=True)

    strategy_csv_map = {}
    for strategy in STRATEGY_CONFIGS:
        csv_path = os.path.join(csv_dir, f"{strategy['name']}.csv")
        strategy_csv_map[strategy["name"]] = csv_path

    os.makedirs(result_save_base_path, exist_ok=True)

    # 确定worker数量
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    print(f"使用 {max_workers} 个并行worker")
    print(f"结果将保存到目录: {csv_dir}")
    print(f"每个策略独立的CSV文件:")
    for strategy_name, csv_path in strategy_csv_map.items():
        print(f"  - {strategy_name}: {os.path.basename(csv_path)}")
    print()

    # 按策略分组准备仿真参数
    strategy_params_map = {}
    for strategy in STRATEGY_CONFIGS:
        strategy_params_map[strategy["name"]] = []
        output_csv = strategy_csv_map[strategy["name"]]
        for file_name in file_names:
            sim_params = (config_path, traffic_path, model_type, file_name, strategy, result_save_base_path, output_csv)
            strategy_params_map[strategy["name"]].append(sim_params)

    # 按策略串行执行,每个策略内部并行处理多个流量文件
    start_time = time.time()
    total_simulations = len(file_names) * len(STRATEGY_CONFIGS)
    global_completed = 0

    for strategy_idx, (strategy_name, sim_params_list) in enumerate(strategy_params_map.items(), 1):
        strategy_start_time = time.time()

        print("\n" + "=" * 80)
        print(f"策略 {strategy_idx}/{len(STRATEGY_CONFIGS)}: {strategy_name}")
        print(f"CSV文件: {os.path.basename(strategy_csv_map[strategy_name])}")
        print(f"流量文件数: {len(sim_params_list)}")
        print("=" * 80 + "\n")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交当前策略的所有仿真任务
            future_to_params = {executor.submit(run_single_simulation, params): params for params in sim_params_list}

            # 处理完成的仿真并保存结果
            strategy_completed = 0

            for future in future_to_params:
                params = future_to_params[future]
                file_name = params[3]

                try:
                    result_data = future.result()
                    save_results_to_csv(result_data)
                    strategy_completed += 1
                    global_completed += 1
                    print(f"[{strategy_name}] 进度: {strategy_completed}/{len(sim_params_list)} | 总进度: {global_completed}/{total_simulations} ({global_completed*100/total_simulations:.1f}%)")
                except Exception:
                    logging.exception(f"处理结果失败: {file_name} [{strategy_name}]")

        strategy_elapsed = time.time() - strategy_start_time
        print(f"\n[{strategy_name}] 完成! 耗时: {strategy_elapsed:.2f}秒 ({strategy_elapsed/60:.1f}分钟)")

    end_time = time.time()
    elapsed = end_time - start_time

    print("\n" + "=" * 80)
    print(f"所有仿真完成!")
    print(f"总耗时: {elapsed:.2f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"平均每个仿真: {elapsed/total_simulations:.2f} 秒")
    print(f"结果目录: {csv_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="保序策略对比仿真")
    parser.add_argument("--traffic_path", default="../traffic/DeepSeek_0616/step6_ch_map/", help="流量数据路径")
    parser.add_argument("--config", default="../config/topologies/topo_5x4.yaml", help="仿真配置文件路径")
    parser.add_argument("--model", default="REQ_RSP", choices=["Feature", "REQ_RSP", "Packet_Base"], help="仿真模型类型")
    parser.add_argument("--results_base_name", default="ordering_comparison_1113", help="结果文件基础名称")
    parser.add_argument("--max_workers", type=int, default=16, help="最大并行worker数量")

    args = parser.parse_args()
    np.random.seed(922)

    print("\n" + "=" * 80)
    print("保序策略对比仿真工具")
    print("=" * 80)
    print(f"流量路径: {args.traffic_path}")
    print(f"配置文件: {args.config}")
    print(f"模型类型: {args.model}")
    print(f"并行度: {args.max_workers}")
    print("=" * 80 + "\n")

    run_comparison_simulation(args.config, args.traffic_path, args.model, args.results_base_name, args.max_workers)


if __name__ == "__main__":
    main()
