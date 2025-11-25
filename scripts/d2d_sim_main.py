"""
D2D 批量模拟脚本
支持批量运行 D2D 仿真，结果保存到 CSV 文件
"""

import os
import sys
import csv
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import portalocker

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.d2d_config import D2DConfig
from src.d2d.d2d_model import D2D_Model


def collect_d2d_results(model: D2D_Model, traffic_file: str) -> dict:
    """
    从 D2D 模型收集结果统计（与 HTML 报告一致的汇总数据）

    Args:
        model: D2D_Model 实例
        traffic_file: 流量文件名

    Returns:
        包含所有统计数据的字典
    """
    from collections import defaultdict

    # 去掉 .txt 后缀
    traffic_name = traffic_file[:-4] if traffic_file.endswith(".txt") else traffic_file

    results = {
        "数据流名称": traffic_name,
        "Die数量": model.num_dies,
        "完成周期": model.current_cycle,
    }

    d2d_processor = getattr(model, "_cached_d2d_processor", None)
    if not d2d_processor:
        return results

    # ========== 请求统计 ==========
    d2d_requests = d2d_processor.d2d_requests
    results["读请求数"] = sum(1 for r in d2d_requests if r.req_type == "read")
    results["写请求数"] = sum(1 for r in d2d_requests if r.req_type == "write")
    results["总请求数"] = len(d2d_requests)

    # ========== 延迟统计 ==========
    latency_stats = d2d_processor._calculate_d2d_latency_stats()
    for cat, label in [("cmd", "CMD"), ("data", "Data"), ("trans", "Trans")]:
        if cat in latency_stats:
            rl = latency_stats[cat]
            for req_type in ["read", "write", "mixed"]:
                count = rl[req_type]["count"]
                avg = rl[req_type]["sum"] / count if count else 0.0
                results[f"{label}延迟_{req_type}_平均"] = round(avg, 2)
                results[f"{label}延迟_{req_type}_最大"] = rl[req_type]["max"]

    # ========== 绕环与Tag统计 ==========
    circuit_stats = d2d_processor._collect_d2d_circuit_stats(model.dies)
    if circuit_stats:
        summary = circuit_stats.get("summary", circuit_stats)

        # Retry统计
        results["读Retry次数"] = summary.get("read_retry_num", 0)
        results["写Retry次数"] = summary.get("write_retry_num", 0)

        # ITag统计
        results["ITag_横向"] = summary.get("ITag_h_num", 0)
        results["ITag_纵向"] = summary.get("ITag_v_num", 0)

        # ETag统计
        results["RB_ETag_T1"] = summary.get("RB_ETag_T1_num", 0)
        results["RB_ETag_T0"] = summary.get("RB_ETag_T0_num", 0)
        results["EQ_ETag_T1"] = summary.get("EQ_ETag_T1_num", 0)
        results["EQ_ETag_T0"] = summary.get("EQ_ETag_T0_num", 0)

        # 绕环次数统计
        results["绕环_请求_横向"] = summary.get("circuits_req_h", 0)
        results["绕环_请求_纵向"] = summary.get("circuits_req_v", 0)
        results["绕环_响应_横向"] = summary.get("circuits_rsp_h", 0)
        results["绕环_响应_纵向"] = summary.get("circuits_rsp_v", 0)
        results["绕环_数据_横向"] = summary.get("circuits_data_h", 0)
        results["绕环_数据_纵向"] = summary.get("circuits_data_v", 0)

        # 等待周期统计
        results["等待周期_请求_横向"] = summary.get("wait_cycle_req_h", 0)
        results["等待周期_请求_纵向"] = summary.get("wait_cycle_req_v", 0)
        results["等待周期_响应_横向"] = summary.get("wait_cycle_rsp_h", 0)
        results["等待周期_响应_纵向"] = summary.get("wait_cycle_rsp_v", 0)
        results["等待周期_数据_横向"] = summary.get("wait_cycle_data_h", 0)
        results["等待周期_数据_纵向"] = summary.get("wait_cycle_data_v", 0)

        # 绕环比例统计
        circling_ratio = summary.get("circling_ratio", {})
        for direction in ["horizontal", "vertical", "overall"]:
            ratio_data = circling_ratio.get(direction, {})
            dir_label = {"horizontal": "横向", "vertical": "纵向", "overall": "整体"}[direction]
            results[f"绕环比例_{dir_label}_总flit"] = ratio_data.get("total_flits", 0)
            results[f"绕环比例_{dir_label}_绕环flit"] = ratio_data.get("circling_flits", 0)
            results[f"绕环比例_{dir_label}_比例"] = round(ratio_data.get("circling_ratio", 0.0), 4)

    # ========== 端口带宽统计 ==========
    # 使用固定的 IP 类型列表，确保所有行字段一致
    FIXED_IP_TYPES = ["ddr", "gdma", "sdma", "cdma", "l2m", "d2d_rn", "d2d_sn"]

    die_ip_bandwidth_data = d2d_processor.die_ip_bandwidth_data
    all_dies_ip_type_groups = defaultdict(lambda: {"read": [], "write": [], "total": []})

    if die_ip_bandwidth_data:
        for die_id in die_ip_bandwidth_data.keys():
            die_data = die_ip_bandwidth_data[die_id]
            if not die_data:
                continue

            read_data = die_data.get("read", {})
            write_data = die_data.get("write", {})
            total_data = die_data.get("total", {})

            all_ip_instances = set(read_data.keys()) | set(write_data.keys()) | set(total_data.keys())

            for ip_instance in all_ip_instances:
                if ip_instance.lower().startswith("d2d"):
                    parts = ip_instance.lower().split("_")
                    ip_type = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
                else:
                    ip_type = ip_instance.split("_")[0]

                for mode, mode_data in [("read", read_data), ("write", write_data), ("total", total_data)]:
                    matrix = mode_data.get(ip_instance)
                    if matrix is not None:
                        non_zero_values = matrix[matrix > 0.001]
                        if len(non_zero_values) > 0:
                            avg = float(non_zero_values.mean())
                            all_dies_ip_type_groups[ip_type][mode].append(avg)

    # 使用固定顺序输出，没有数据的 IP 类型填 0
    for ip_type in FIXED_IP_TYPES:
        group = all_dies_ip_type_groups.get(ip_type, {"read": [], "write": [], "total": []})
        read_avg = sum(group["read"]) / len(group["read"]) if group["read"] else 0.0
        write_avg = sum(group["write"]) / len(group["write"]) if group["write"] else 0.0
        total_avg = sum(group["total"]) / len(group["total"]) if group["total"] else 0.0

        results[f"带宽_{ip_type.upper()}_读"] = round(read_avg, 3)
        results[f"带宽_{ip_type.upper()}_写"] = round(write_avg, 3)
        results[f"带宽_{ip_type.upper()}_混合"] = round(total_avg, 3)

    return results


def run_single_d2d_simulation(sim_params):
    """
    运行单个 D2D 仿真

    Args:
        sim_params: 仿真参数元组
            (d2d_config_path, traffic_path, traffic_file, result_save_path, output_csv, sim_config)

    Returns:
        (traffic_file, results, output_csv) 元组
    """
    (d2d_config_path, traffic_path, traffic_file, result_save_path, output_csv, sim_config) = sim_params

    try:
        # 1. 加载配置
        config = D2DConfig(d2d_config_file=d2d_config_path)

        # 2. 创建 D2D_Model 实例
        model = D2D_Model(
            config=config,
            result_save_path=result_save_path,
            results_fig_save_path=result_save_path,
            verbose=sim_config["verbose"],
        )

        # 3. 配置流量调度器
        model.setup_traffic_scheduler(traffic_file_path=traffic_path, traffic_chains=[[traffic_file]])

        # 4. 配置结果分析
        model.setup_result_analysis(
            export_d2d_requests_csv=sim_config.get("export_d2d_csv", True),
            export_ip_bandwidth_csv=sim_config.get("export_ip_bw_csv", True),
        )

        # 5. 运行仿真
        model.run_simulation(
            max_time=sim_config["max_time"],
            print_interval=sim_config["print_interval"],
            results_analysis=True,
            verbose=sim_config["verbose"],
        )

        # 6. 收集结果
        results = collect_d2d_results(model, traffic_file)

        return (traffic_file, results, output_csv)

    except Exception:
        logging.exception(f"D2D 仿真失败: {traffic_file}")
        return (traffic_file, None, output_csv)


def save_results_to_csv(results_data, db_manager=None, experiment_id=None, save_csv=True):
    """
    保存结果到 CSV 和/或数据库

    Args:
        results_data: (traffic_file, results, output_csv) 元组
        db_manager: ResultManager 实例（可选）
        experiment_id: 实验 ID（可选）
        save_csv: 是否保存到 CSV 文件
    """
    file_name, results, output_csv = results_data

    if results is None:
        print(f"跳过 {file_name} 的保存（仿真错误）")
        return

    # CSV 写入（可选）
    if save_csv:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        with open(output_csv, mode="a", newline="", encoding="utf-8-sig") as f:
            portalocker.lock(f, portalocker.LOCK_EX)

            # 检查文件是否为空
            f.seek(0, 2)
            write_header = f.tell() == 0

            writer = csv.DictWriter(f, fieldnames=results.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(results)

        print(f"{file_name} 已保存到 CSV")

    # 数据库写入
    if db_manager is not None and experiment_id is not None:
        try:
            performance = results.get("带宽_DDR_混合", 0)
            db_manager.add_result(
                experiment_id=experiment_id,
                config_params=results,
                performance=performance,
                result_details={},
            )
        except Exception as db_e:
            print(f"数据库写入失败: {db_e}")


def run_d2d_simulation(d2d_config_path, traffic_path, results_folder_name, max_workers, sim_config, save_to_db=False, experiment_name=None, save_csv=True):
    """
    批量运行 D2D 仿真

    Args:
        d2d_config_path: D2D 配置文件路径
        traffic_path: 流量文件目录
        results_folder_name: 结果保存文件夹名称
        max_workers: 并行工作进程数
        sim_config: 仿真配置字典
        save_to_db: 是否保存到数据库
        experiment_name: 数据库中的实验名称
        save_csv: 是否生成 CSV 文件
    """
    # 获取流量文件列表
    traffic_files = [f for f in os.listdir(traffic_path) if f.endswith(".txt")]

    if not traffic_files:
        print(f"未找到流量文件: {traffic_path}")
        return

    print(f"找到 {len(traffic_files)} 个流量文件")

    # 设置结果路径
    result_base_path = os.path.join("../Result/d2d/", results_folder_name)
    output_csv = os.path.join(result_base_path, "d2d_summary.csv")
    os.makedirs(result_base_path, exist_ok=True)

    # 如果 CSV 文件已存在，删除它（重新开始）
    if save_csv and os.path.exists(output_csv):
        os.remove(output_csv)

    # 数据库初始化
    db_manager = None
    experiment_id = None
    if save_to_db:
        from src.database import ResultManager
        db_manager = ResultManager()
        exp_name = experiment_name or f"d2d_{results_folder_name}"
        experiment_id = db_manager.create_experiment(
            name=exp_name,
            experiment_type="d2d",
            config_path=d2d_config_path,
            description=f"D2D 仿真 - {results_folder_name}",
            total_combinations=len(traffic_files),
        )
        print(f"数据库实验已创建: {exp_name} (ID: {experiment_id})")

    # 准备参数列表
    sim_params_list = []
    for traffic_file in traffic_files:
        file_result_path = os.path.join(result_base_path, traffic_file[:-4]) + "/"
        sim_params = (d2d_config_path, traffic_path, traffic_file, file_result_path, output_csv, sim_config)
        sim_params_list.append(sim_params)

    # 设置并行进程数
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    print(f"使用 {max_workers} 个并行进程")

    # 并行执行
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_d2d_simulation, params): params[2] for params in sim_params_list}

        completed = 0
        for future in as_completed(futures):
            file_name = futures[future]
            result = future.result()
            save_results_to_csv(result, db_manager, experiment_id, save_csv)
            completed += 1
            print(f"进度: {completed}/{len(traffic_files)} 完成 - {file_name}")

    # 更新实验状态
    if db_manager and experiment_id:
        db_manager.update_experiment_status(experiment_id, "completed")
        print(f"数据库实验状态已更新为 completed")

    if save_csv:
        print(f"\n仿真完成！结果保存到: {output_csv}")
    else:
        print(f"\n仿真完成！")


def main():
    # ========== D2D 配置 ==========
    d2d_config_path = "../config/topologies/d2d_4die_config.yaml"

    # ========== 流量配置 ==========
    traffic_path = "../traffic/2261/"

    # ========== 结果保存配置 ==========
    results_folder_name = "2261_1125"

    # ========== 并行配置 ==========
    max_workers = 2

    # ========== 仿真参数 ==========
    sim_config = {
        "max_time": 5800,  # 最大仿真时间 (ns)
        "print_interval": 2000,  # 打印间隔 (ns)
        "verbose": 0,  # 详细程度 (0=静默, 1=正常)
        "export_d2d_csv": True,  # 是否导出D2D请求CSV
        "export_ip_bw_csv": True,  # 是否导出IP带宽CSV
    }

    # ========== 数据库配置 ==========
    save_to_db = False  # 是否保存到数据库
    experiment_name = None  # 数据库中的实验名称，None 时自动生成

    # ========== CSV 配置 ==========
    save_csv = True  # 是否生成 CSV 文件

    # 运行仿真
    run_d2d_simulation(d2d_config_path, traffic_path, results_folder_name, max_workers, sim_config, save_to_db, experiment_name, save_csv)


if __name__ == "__main__":
    main()
