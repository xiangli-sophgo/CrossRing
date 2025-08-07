"""
批量仿真脚本 - 遍历仿真所有生成的数据流
"""

from src.core import *
import os
from src.utils.components import *
from config.config import CrossRingConfig
import matplotlib
import numpy as np
import sys
import itertools
import time
from datetime import datetime
import csv
import pandas as pd

if sys.platform == "darwin":  # macOS 的系统标识是 'darwin'
    matplotlib.use("macosx")  # 仅在 macOS 上使用该后端


def extract_bandwidth_data(result_save_path):
    """从仿真结果中提取带宽数据"""
    bandwidth_data = {"total_sum_bw": 0, "avg_port_bandwidth": 0, "port_count": 0}

    try:
        # 查找ports_bandwidth.csv文件
        ports_bw_file = None
        for root, dirs, files in os.walk(result_save_path):
            for file in files:
                if file == "ports_bandwidth.csv":
                    ports_bw_file = os.path.join(root, file)
                    break
            if ports_bw_file:
                break

        if ports_bw_file and os.path.exists(ports_bw_file):
            # 读取ports_bandwidth.csv文件
            df = pd.read_csv(ports_bw_file)

            # 提取端口带宽数据（使用mixed_weighted_bandwidth_gbps作为主要指标）
            if "mixed_weighted_bandwidth_gbps" in df.columns:
                # 过滤有效数据（非NaN且大于0）
                valid_bw = df["mixed_weighted_bandwidth_gbps"].dropna()
                valid_bw = valid_bw[valid_bw > 0]

                if len(valid_bw) > 0:
                    bandwidth_data["avg_port_bandwidth"] = valid_bw.mean()
                    bandwidth_data["port_count"] = len(valid_bw)
                    bandwidth_data["total_sum_bw"] = valid_bw.sum()

            print(f"✓ 成功提取带宽数据: 平均端口带宽 {bandwidth_data['avg_port_bandwidth']:.2f} GB/s")

        else:
            # 备选方案：查找其他结果文件
            result_files = []
            for root, dirs, files in os.walk(result_save_path):
                for file in files:
                    if file.startswith("Result_") and file.endswith(".txt"):
                        result_files.append(os.path.join(root, file))

            if result_files:
                print(f"⚠️ 未找到ports_bandwidth.csv，但找到 {len(result_files)} 个结果文件")
            else:
                print("⚠️ 未找到带宽相关的结果文件")

    except Exception as e:
        print(f"❌ 提取带宽数据时出错: {str(e)}")

    return bandwidth_data


def save_results_to_csv(results, csv_output_path):
    """将批量仿真结果保存到CSV文件"""
    try:
        # 定义CSV字段
        fieldnames = [
            "file_name",
            "topology",
            "c2c_type",
            "spare_core",
            "request_type",
            "simulation_time_seconds",
            "avg_port_bandwidth_gbps",
            "total_sum_bandwidth_gbps",
            "active_port_count",
            "timestamp",
        ]

        # 创建CSV文件
        with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 写入表头
            writer.writeheader()

            # 写入数据
            for result in results:
                cfg = result["config"]
                bw_data = result.get("bandwidth", {})

                row = {
                    "file_name": os.path.basename(result["file"]),
                    "topology": cfg["topo"],
                    "c2c_type": cfg["c2c_type"],
                    "spare_core": cfg["spare_core"],
                    "request_type": cfg["req_type"],
                    "simulation_time_seconds": round(result["time"], 2),
                    "avg_port_bandwidth_gbps": round(bw_data.get("avg_port_bandwidth", 0), 4),
                    "total_sum_bandwidth_gbps": round(bw_data.get("total_sum_bw", 0), 4),
                    "active_port_count": bw_data.get("port_count", 0),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                writer.writerow(row)

        print(f"✓ 成功保存 {len(results)} 条仿真结果到CSV文件")

        # 输出统计汇总
        if results:
            total_sims = len(results)
            avg_bw_list = [r.get("bandwidth", {}).get("avg_port_bandwidth", 0) for r in results]
            avg_bw_list = [bw for bw in avg_bw_list if bw > 0]  # 过滤无效值

            if avg_bw_list:
                overall_avg_bw = sum(avg_bw_list) / len(avg_bw_list)
                max_bw = max(avg_bw_list)
                min_bw = min(avg_bw_list)

                print(f"\n📊 带宽统计汇总:")
                print(f"   总仿真数: {total_sims}")
                print(f"   有效带宽数据: {len(avg_bw_list)}")
                print(f"   平均端口带宽: {overall_avg_bw:.4f} GB/s")
                print(f"   最大端口带宽: {max_bw:.4f} GB/s")
                print(f"   最小端口带宽: {min_bw:.4f} GB/s")

    except Exception as e:
        print(f"❌ 保存CSV文件时出错: {str(e)}")
        import traceback

        traceback.print_exc()


def run_single_simulation(traffic_file, topo_type, model_type, config_path, result_save_path):
    """运行单个仿真"""
    print(f"开始仿真: {traffic_file}")

    config = CrossRingConfig(config_path)
    config.CROSSRING_VERSION = "V1"
    config.TOPO_TYPE = topo_type

    # 根据拓扑类型配置参数
    if topo_type == "4x4":
        config.NUM_COL = 4
        config.NUM_NODE = 32
        config.NUM_ROW = 8
        config.BURST = 4
        config.NUM_IP = 16

    elif topo_type == "4x2":
        config.NUM_COL = 2
        config.NUM_NODE = 16
        config.NUM_ROW = 8

    # 通用配置
    config.RN_R_TRACKER_OSTD = 64
    config.RN_W_TRACKER_OSTD = 32
    config.SN_DDR_R_TRACKER_OSTD = 96
    config.SN_DDR_W_TRACKER_OSTD = 48
    config.RN_RDB_SIZE = config.RN_R_TRACKER_OSTD * config.BURST
    config.RN_WDB_SIZE = config.RN_W_TRACKER_OSTD * config.BURST
    config.SN_DDR_WDB_SIZE = config.SN_DDR_W_TRACKER_OSTD * config.BURST
    config.DDR_R_LATENCY_original = 100
    config.DDR_W_LATENCY_original = 40
    config.IQ_CH_FIFO_DEPTH = 10
    config.EQ_CH_FIFO_DEPTH = 10
    config.IQ_OUT_FIFO_DEPTH_HORIZONTAL = 8
    config.IQ_OUT_FIFO_DEPTH_VERTICAL = 8
    config.IQ_OUT_FIFO_DEPTH_EQ = 8
    config.RB_OUT_FIFO_DEPTH = 8
    config.SN_TRACKER_RELEASE_LATENCY = 40
    config.DDR_BW_LIMIT = 115.2

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
        "gdma": 3,
        "sdma": 2,
        "cdma": 4,
        "ddr": 2,
        "l2m": 2,
    }

    # 设置流量文件路径和配置
    traffic_file_path = os.path.dirname(traffic_file)
    traffic_file_name = os.path.basename(traffic_file)
    traffic_config = [[traffic_file_name]]

    # 创建仿真实例
    sim: BaseModel = eval(f"{model_type}_model")(
        model_type=model_type,
        config=config,
        topo_type=topo_type,
        traffic_file_path=traffic_file_path + "/",
        traffic_config=traffic_config,
        result_save_path=result_save_path,
        results_fig_save_path=result_save_path,
        plot_flow_fig=1,
        flow_fig_show_CDMA=1,
        plot_RN_BW_fig=1,
        plot_link_state=0,
        plot_start_time=1000,
        print_trace=0,
        show_trace_id=0,
        show_node_id=4,
        verbose=0,  # 减少输出
    )

    np.random.seed(801)

    # 运行仿真
    sim.initial()
    sim.end_time = 6000  # 与生成数据流的END_TIME一致
    sim.print_interval = 2000

    start_time = time.time()
    sim.run()
    end_time = time.time()

    print(f"完成仿真: {traffic_file}, 耗时: {end_time - start_time:.2f}秒")

    # 提取带宽信息
    bandwidth_data = extract_bandwidth_data(result_save_path)

    # 返回仿真结果
    return {
        "file": traffic_file,
        "time": end_time - start_time,
        "config": {"topo": topo_type, "c2c_type": "w" if "wc2c" in traffic_file else "wo", "spare_core": "w" if "wSPC" in traffic_file else "wo", "req_type": "R" if "_R.txt" in traffic_file else "W"},
        "bandwidth": bandwidth_data,
    }


def batch_simulate_all():
    """批量仿真所有生成的数据流"""
    print(f"批量仿真开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 参数设置
    model_type = "REQ_RSP"
    config_path = r"../config/config2.json"
    result_save_path = f"../Result/2262_0801_Sim/{model_type}/"
    traffic_base_path = "../test_data/"

    # 创建结果目录
    os.makedirs(result_save_path, exist_ok=True)

    # 定义所有参数组合
    TOPO_OPTIONS = ["4x2", "4x4"]
    REQ_TYPE_OPTIONS = ["R", "W"]
    C2C_TYPE_OPTIONS = ["wo", "w"]
    SPARE_CORE_OPTIONS = ["wo", "w"]

    results = []

    # 遍历所有组合
    for topo, req_type, c2c_type, spare_core in itertools.product(TOPO_OPTIONS, REQ_TYPE_OPTIONS, C2C_TYPE_OPTIONS, SPARE_CORE_OPTIONS):
        # 生成文件名
        file_name = f"2262_{topo}_{c2c_type}c2c_{spare_core}SPC_{req_type}.txt"
        traffic_file = os.path.join(traffic_base_path, file_name)

        # 检查文件是否存在
        if not os.path.exists(traffic_file):
            print(f"警告: 文件不存在 - {traffic_file}")
            continue

        # 创建特定的结果保存路径
        specific_result_path = os.path.join(result_save_path, f"{topo}_{c2c_type}c2c_{spare_core}SPC_{req_type}/")
        os.makedirs(specific_result_path, exist_ok=True)

        try:
            # 运行仿真
            result = run_single_simulation(traffic_file=traffic_file, topo_type=topo, model_type=model_type, config_path=config_path, result_save_path=specific_result_path)
            results.append(result)

        except Exception as e:
            print(f"错误: 仿真失败 - {traffic_file}")
            print(f"错误信息: {str(e)}")
            import traceback

            traceback.print_exc()

    # 打印汇总结果
    print("\n" + "=" * 80)
    print("批量仿真完成！")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"成功仿真: {len(results)} 个文件")
    print("\n仿真结果汇总:")
    print("-" * 80)
    print(f"{'文件名':<50} {'拓扑':<8} {'C2C':<6} {'备用核':<8} {'请求类型':<8} {'耗时(秒)':<10}")
    print("-" * 80)

    for result in results:
        cfg = result["config"]
        file_name = os.path.basename(result["file"])
        print(f"{file_name:<50} {cfg['topo']:<8} {cfg['c2c_type']:<6} {cfg['spare_core']:<8} {cfg['req_type']:<8} {result['time']:<10.2f}")

    # 统计信息
    total_time = sum(r["time"] for r in results)
    print("-" * 80)
    print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"平均每个仿真: {total_time/len(results):.2f}秒" if results else "")

    # 生成CSV汇总文件
    if results:
        csv_output_path = os.path.join(result_save_path, "batch_simulation_summary.csv")
        save_results_to_csv(results, csv_output_path)
        print(f"✓ CSV汇总文件已保存到: {csv_output_path}")


if __name__ == "__main__":
    batch_simulate_all()
