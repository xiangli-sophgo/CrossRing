import os
import numpy as np


# 从指定文件夹中读取所有包含 "Result_" 的文本文件
def read_data_from_folder(folder_path):
    data_stream = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") and "Result_" in filename:
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                data_stream.extend([line.strip() for line in file.readlines() if line.strip()])  # 去掉空行
    return data_stream


# 解析数据流
def parse_data(data):
    parsed_data = []
    for entry in data:
        if not entry[0].isdigit():
            continue
        fields = entry.split(",")
        tx_time = int(fields[0])
        src_id = int(fields[1])
        src_type = fields[2]
        des_id = int(fields[3])
        des_type = fields[4]
        rw = fields[5]
        burst_len = int(fields[6])
        rx_time = int(fields[7])
        # path = eval(fields[8])  # 将字符串转换为列表
        # circuits_completed_v = int(fields[9])
        # circuits_completed_h = int(fields[10])
        parsed_data.append(
            {
                "tx_time": tx_time,
                "src_id": src_id,
                "src_type": src_type,
                "des_id": des_id,
                "des_type": des_type,
                "rw": rw,
                "burst_len": burst_len,
                "rx_time": rx_time,
                # "path": path,
                # "circuits_completed_v": circuits_completed_v,
                # "circuits_completed_h": circuits_completed_h
            }
        )
    return parsed_data


# 计算带宽和延迟
def calculate_bandwidth_and_latency(parsed_data):
    # 用于存储不同类型的合并间隔和延迟
    flows = {
        "ddr_to_sdma": {"merged_intervals": [], "latencies": []},
        "sdma_to_l2m": {"merged_intervals": [], "latencies": []},
        "l2m_to_gdma": {"merged_intervals": [], "latencies": []},
    }

    for entry in parsed_data:
        tx_time = entry["tx_time"]
        rx_time = entry["rx_time"]
        burst_len = entry["burst_len"]

        # 计算延迟
        latency = rx_time - tx_time

        # 根据 src_type 和 des_type 更新不同的流
        if entry["src_type"] == "ddr" and entry["des_type"] == "sdma":
            flows["ddr_to_sdma"]["latencies"].append(latency)
            update_intervals(flows["ddr_to_sdma"]["merged_intervals"], tx_time, rx_time, burst_len)
        elif entry["src_type"] == "sdma" and entry["des_type"] == "l2m":
            flows["sdma_to_l2m"]["latencies"].append(latency)
            update_intervals(flows["sdma_to_l2m"]["merged_intervals"], tx_time, rx_time, burst_len)
        elif entry["src_type"] == "l2m" and entry["des_type"] == "gdma":
            flows["l2m_to_gdma"]["latencies"].append(latency)
            update_intervals(flows["l2m_to_gdma"]["merged_intervals"], tx_time, rx_time, burst_len)

    return flows


def update_intervals(merged_intervals, tx_time, rx_time, burst_len):
    if len(merged_intervals) == 0 or merged_intervals[-1][1] < tx_time:
        merged_intervals.append((tx_time, rx_time, burst_len))
    else:
        last_start, last_end, count = merged_intervals[-1]
        merged_intervals[-1] = (last_start, max(last_end, rx_time), count + burst_len)


# 输出结果
def output_results(flows):
    for flow_name, flow_data in flows.items():
        merged_intervals = flow_data["merged_intervals"]
        latencies = flow_data["latencies"]

        print(f"{flow_name} intervals and bandwidth results:")
        weighted_bandwidth_sum, total_count, total_bandwidth = 0, 0, [np.inf, -np.inf]

        for start, end, count in merged_intervals:
            if start == end:
                continue
            bandwidth = count * 128 / 8 / (end - start)  # 示例带宽计算
            total_bandwidth[0] = min(total_bandwidth[0], bandwidth)
            total_bandwidth[1] = max(total_bandwidth[1], bandwidth)
            weighted_bandwidth_sum += bandwidth * count
            total_count += count
            print(f"Interval: {start} to {end}, count: {count}, bandwidth: {bandwidth:.1f}")

        weighted_bandwidth = weighted_bandwidth_sum / total_count if total_count > 0 else 0
        latency_avg = np.average(latencies) if latencies else 0
        latency_max = max(latencies) if latencies else 0
        latency_min = min(latencies) if latencies else 0
        print(f"Weighted bandwidth: {weighted_bandwidth:.1f} AvgLatency: {latency_avg:.1f}, MaxLatency: {latency_max}, MinLatency: {latency_min}\n")


# 主程序
folder_path = r"..\Result\cross ring\v7-32\6x5\testcase-v1.1.1"  # 数据文件夹路径
data_stream = read_data_from_folder(folder_path)
parsed_data = parse_data(data_stream)
flows = calculate_bandwidth_and_latency(parsed_data)
output_results(flows)
