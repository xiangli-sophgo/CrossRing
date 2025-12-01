"""
仲裁策略对比脚本
对比不同仲裁策略对NoC性能的影响

运行方式:
python arbitration_strategy_comparison.py --traffic_path ../traffic/DeepSeek_0616/step6_ch_map/ --max_workers 8
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

import portalocker

logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


# 定义仲裁策略配置组合（可通过注释/取消注释选择要运行的策略）
ARBITRATION_CONFIGS = [
    # ==================== 基础仲裁器 ====================
    {"name": "RoundRobin", "desc": "轮询仲裁（基准对照）", "config": {"type": "round_robin"}},
    # {"name": "Weighted_2211", "desc": "加权仲裁，权重[2,2,1,1]", "config": {"type": "weighted", "weights": [2, 2, 1, 1]}},
    # {"name": "Priority_0123", "desc": "优先级仲裁，优先级[0,1,2,3]", "config": {"type": "priority", "priorities": [0, 1, 2, 3]}},
    # ==================== iSLIP迭代次数对比 ====================
    {"name": "iSLIP_iter1", "desc": "iSLIP单次迭代", "config": {"type": "islip", "iterations": 1, "weight_strategy": "uniform"}},
    {"name": "iSLIP_iter2", "desc": "iSLIP两次迭代", "config": {"type": "islip", "iterations": 2, "weight_strategy": "uniform"}},
    {"name": "iSLIP_iter4", "desc": "iSLIP四次迭代", "config": {"type": "islip", "iterations": 4, "weight_strategy": "uniform"}},
    # ==================== iSLIP权重策略对比 ====================
    {"name": "iSLIP_QueueLen", "desc": "iSLIP+队列长度权重", "config": {"type": "islip", "iterations": 2, "weight_strategy": "queue_length"}},
    {"name": "iSLIP_WaitTime", "desc": "iSLIP+等待时间权重", "config": {"type": "islip", "iterations": 2, "weight_strategy": "wait_time"}},
    {"name": "iSLIP_Hybrid", "desc": "iSLIP+混合权重(0.7队列+0.3等待)", "config": {"type": "islip", "iterations": 2, "weight_strategy": "hybrid"}},
    # ==================== 其他高级匹配算法 ====================
    {"name": "LQF", "desc": "最长队列优先(Longest Queue First)", "config": {"type": "lqf", "weight_strategy": "queue_length"}},
    {"name": "OCF", "desc": "最老单元优先(Oldest Cell First)", "config": {"type": "ocf", "weight_strategy": "wait_time"}},
    {"name": "PIM_iter2", "desc": "并行迭代匹配(Parallel Iterative Matching)", "config": {"type": "pim", "iterations": 2, "weight_strategy": "uniform"}},
]


def run_single_simulation(sim_params):
    """运行单个仿真 - 针对特定流量文件和特定仲裁策略配置"""
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

        # ==================== 关键: 动态修改仲裁策略配置 ====================
        config.arbitration = {
            "default": strategy["config"],
            "iq": strategy["config"],
            "rb": strategy["config"],
            "eq": strategy["config"],
        }
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
            plot_RN_BW_fig=0,
            flow_graph_interactive=1,
            fifo_utilization_heatmap=1,
            show_result_analysis=0,
        )

        # 运行仿真
        sim.run_simulation(max_time=10000, print_interval=5000)

        # 处理综合结果（收集HTML和结果文件内容，不写本地文件）
        sim.process_comprehensive_results(save_to_db_only=True)

        # 获取结果
        results = sim.get_results()

        # 获取HTML报告内容和结果文件
        result_html = getattr(sim, "_result_html_content", None)
        result_file_contents = getattr(sim, "_result_file_contents", {})

        # 添加策略标识到结果中
        results["strategy"] = strategy["name"]
        results["strategy_desc"] = strategy["desc"]
        results["arbitration_type"] = strategy["config"]["type"]
        # 添加策略配置参数
        for key, value in strategy["config"].items():
            if key != "type":
                results[f"arb_{key}"] = value

        print(f"[{strategy['name']}] 完成仿真: {file_name} (进程 {os.getpid()})")
        return (file_name, strategy["name"], results, output_csv, result_html, result_file_contents)

    except Exception:
        logging.exception(f"[{strategy['name']}] 仿真失败: {file_name}")
        return (file_name, strategy["name"], None, output_csv, None, None)


def save_results_to_csv(results_data, db_manager=None, experiment_id=None, save_csv=True):
    """
    保存仿真结果到CSV文件和/或数据库（线程安全）

    Args:
        results_data: (file_name, strategy_name, results, output_csv, result_html, result_file_contents) 元组
        db_manager: ResultManager 实例（可选）
        experiment_id: 实验 ID（可选）
        save_csv: 是否保存到 CSV 文件
    """
    file_name, strategy_name, results, output_csv, result_html, result_file_contents = results_data

    if results is None:
        print(f"[{strategy_name}] 跳过保存: {file_name} (仿真错误)")
        return

    # CSV 写入（可选）
    if save_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        with open(output_csv, mode="a", newline="", encoding="utf-8-sig") as output_csv_file:
            portalocker.lock(output_csv_file, portalocker.LOCK_EX)

            # 检查文件是否为空
            output_csv_file.seek(0, 2)
            write_header = output_csv_file.tell() == 0

            writer = csv.DictWriter(output_csv_file, fieldnames=results.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(results)

        print(f"[{strategy_name}] CSV已保存: {file_name}")

    # 数据库写入
    if db_manager is not None and experiment_id is not None:
        try:
            # 使用带宽作为性能指标
            performance = results.get("带宽_DDR_混合", results.get("performance", 0))
            db_manager.add_result(
                experiment_id=experiment_id,
                config_params=results,
                performance=performance,
                result_details=results,
                result_html=result_html,
                result_file_contents=result_file_contents if result_file_contents else None,
            )
            if not save_csv:
                print(f"[{strategy_name}] 数据库已保存: {file_name}")
        except Exception as db_e:
            print(f"[{strategy_name}] 数据库写入失败: {db_e}")


def run_comparison_simulation(config_path, traffic_path, model_type, results_base_name, max_workers=None, save_to_db=False, experiment_name=None, save_csv=True):
    """
    运行仲裁策略对比仿真

    Args:
        config_path: 配置文件路径
        traffic_path: 流量文件目录
        model_type: 模型类型
        results_base_name: 结果文件夹名称
        max_workers: 并行进程数
        save_to_db: 是否保存到数据库
        experiment_name: 数据库中的实验名称
        save_csv: 是否生成 CSV 文件
    """

    # 获取所有流量文件
    all_files = os.listdir(traffic_path)
    file_names = [file for file in all_files if file.endswith(".txt")]

    if not file_names:
        print("未找到流量文件！")
        return

    print(f"找到 {len(file_names)} 个流量文件")
    print(f"将对每个文件运行 {len(ARBITRATION_CONFIGS)} 种仲裁策略配置")
    print(f"总计 {len(file_names) * len(ARBITRATION_CONFIGS)} 个仿真任务\n")

    # 显示策略配置
    print("=" * 80)
    print("仲裁策略配置:")
    print("=" * 80)
    for i, strategy in enumerate(ARBITRATION_CONFIGS, 1):
        config_str = ", ".join(f"{k}={v}" for k, v in strategy["config"].items())
        print(f"{i}. {strategy['name']}: {strategy['desc']}")
        print(f"   配置: {config_str}")
    print("=" * 80 + "\n")

    # 设置结果路径
    result_save_base_path = f"../Result/CrossRing/{model_type}/{results_base_name}/"

    # 为每个策略创建独立的CSV文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = os.path.join(r"../Result/Traffic_result_csv/", f"arbitration_comparison_{timestamp}")
    if save_csv:
        os.makedirs(csv_dir, exist_ok=True)

    strategy_csv_map = {}
    for strategy in ARBITRATION_CONFIGS:
        csv_path = os.path.join(csv_dir, f"{strategy['name']}.csv")
        strategy_csv_map[strategy["name"]] = csv_path

    os.makedirs(result_save_base_path, exist_ok=True)

    # 数据库初始化 - 为每个策略创建独立实验
    db_manager = None
    strategy_experiment_ids = {}
    if save_to_db:
        from src.database import ResultManager

        db_manager = ResultManager()

        for strategy in ARBITRATION_CONFIGS:
            exp_name = f"{experiment_name}_{strategy['name']}" if experiment_name else f"arbitration_{strategy['name']}_{results_base_name}"
            config_str = ", ".join(f"{k}={v}" for k, v in strategy["config"].items())
            exp_id = db_manager.create_experiment(
                name=exp_name,
                experiment_type="kcin",
                config_path=config_path,
                description=f"仲裁策略对比 - {strategy['name']}: {strategy['desc']} ({config_str})",
                total_combinations=len(file_names),
            )
            strategy_experiment_ids[strategy["name"]] = exp_id
            print(f"数据库实验已创建: {exp_name} (ID: {exp_id})")

    # 确定worker数量
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    print(f"\n使用 {max_workers} 个并行worker")
    if save_csv:
        print(f"结果将保存到目录: {csv_dir}")
        print(f"每个策略独立的CSV文件:")
        for strategy_name, csv_path in strategy_csv_map.items():
            print(f"  - {strategy_name}: {os.path.basename(csv_path)}")
    if save_to_db:
        print(f"结果将保存到数据库")
    print()

    # 按策略分组准备仿真参数
    strategy_params_map = {}
    for strategy in ARBITRATION_CONFIGS:
        strategy_params_map[strategy["name"]] = []
        output_csv = strategy_csv_map[strategy["name"]]
        for file_name in file_names:
            sim_params = (config_path, traffic_path, model_type, file_name, strategy, result_save_base_path, output_csv)
            strategy_params_map[strategy["name"]].append(sim_params)

    # 按策略串行执行,每个策略内部并行处理多个流量文件
    start_time = time.time()
    total_simulations = len(file_names) * len(ARBITRATION_CONFIGS)
    global_completed = 0

    for strategy_idx, (strategy_name, sim_params_list) in enumerate(strategy_params_map.items(), 1):
        strategy_start_time = time.time()

        print("\n" + "=" * 80)
        print(f"策略 {strategy_idx}/{len(ARBITRATION_CONFIGS)}: {strategy_name}")
        if save_csv:
            print(f"CSV文件: {os.path.basename(strategy_csv_map[strategy_name])}")
        print(f"流量文件数: {len(sim_params_list)}")
        print("=" * 80 + "\n")

        # 获取当前策略的实验ID
        current_experiment_id = strategy_experiment_ids.get(strategy_name) if save_to_db else None

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
                    save_results_to_csv(result_data, db_manager, current_experiment_id, save_csv)
                    strategy_completed += 1
                    global_completed += 1
                    print(f"[{strategy_name}] 进度: {strategy_completed}/{len(sim_params_list)} | 总进度: {global_completed}/{total_simulations} ({global_completed*100/total_simulations:.1f}%)")
                except Exception:
                    logging.exception(f"处理结果失败: {file_name} [{strategy_name}]")

        strategy_elapsed = time.time() - strategy_start_time
        print(f"\n[{strategy_name}] 完成! 耗时: {strategy_elapsed:.2f}秒 ({strategy_elapsed/60:.1f}分钟)")

        # 更新当前策略的实验状态
        if db_manager and current_experiment_id:
            db_manager.update_experiment_status(current_experiment_id, "completed")

    end_time = time.time()
    elapsed = end_time - start_time

    print("\n" + "=" * 80)
    print(f"所有仿真完成!")
    print(f"总耗时: {elapsed:.2f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"平均每个仿真: {elapsed/total_simulations:.2f} 秒")
    if save_csv:
        print(f"结果目录: {csv_dir}")
    if save_to_db:
        print(f"数据库实验ID列表:")
        for strategy_name, exp_id in strategy_experiment_ids.items():
            print(f"  - {strategy_name}: {exp_id}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="仲裁策略对比仿真")
    parser.add_argument("--traffic_path", default="../traffic/DeepSeek_0616/step6_ch_map/", help="流量数据路径")
    parser.add_argument("--config", default="../config/topologies/topo_5x4.yaml", help="仿真配置文件路径")
    parser.add_argument("--model", default="REQ_RSP", choices=["Feature", "REQ_RSP", "Packet_Base"], help="仿真模型类型")
    parser.add_argument("--results_base_name", default="arbitration_comparison", help="结果文件基础名称")
    parser.add_argument("--max_workers", type=int, default=8, help="最大并行worker数量")
    parser.add_argument("--save_to_db", type=int, default=1, help="是否保存到数据库 (0/1)")
    parser.add_argument("--experiment_name", default="arbitration_comparison", help="数据库中的实验名称前缀")
    parser.add_argument("--save_csv", type=int, default=0, help="是否生成CSV文件 (0/1)")

    args = parser.parse_args()
    np.random.seed(922)

    print("\n" + "=" * 80)
    print("仲裁策略对比仿真工具")
    print("=" * 80)
    print(f"流量路径: {args.traffic_path}")
    print(f"配置文件: {args.config}")
    print(f"模型类型: {args.model}")
    print(f"并行度: {args.max_workers}")
    print(f"保存到数据库: {'是' if args.save_to_db else '否'}")
    print(f"生成CSV文件: {'是' if args.save_csv else '否'}")
    if args.experiment_name:
        print(f"实验名称前缀: {args.experiment_name}")
    print("=" * 80 + "\n")

    run_comparison_simulation(
        args.config,
        args.traffic_path,
        args.model,
        args.results_base_name,
        args.max_workers,
        save_to_db=bool(args.save_to_db),
        experiment_name=args.experiment_name,
        save_csv=bool(args.save_csv),
    )


if __name__ == "__main__":
    main()
