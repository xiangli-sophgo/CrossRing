import os
import math


def merge_intervals(intervals):
    # 如果区间列表为空，返回空列表
    if not intervals:
        return []

    # 按区间的起始值排序
    intervals.sort(key=lambda x: x[0])

    # 初始化合并后的区间列表，每个元素为区间和该区间包含的小区间数
    merged = [intervals[0]]

    for current in intervals:
        # 获取已合并区间列表中的最后一个区间及其包含的小区间数
        last_start, last_end, count = merged[-1]

        # 如果当前区间的开始小于或等于上一个区间的结束加20，则认为它们重叠或相邻，应合并
        if current[0] <= last_end:
            # 更新已合并区间的结束值，并增加小区间计数
            merged[-1] = (
                last_start,
                max(last_end, current[1]),
                count + current[2],
            )
        else:
            # 否则，添加新区间到列表
            merged.append((current[0], current[1], current[2]))

    # print(merged)
    return merged


def read_intervals(filename):
    with open(filename, "r") as file:
        return [
            (int(parts[0]), int(parts[7]), int(parts[6]))
            for line in file
            if (parts := line.strip().split(",")) and len(parts) >= 2 and parts[0].isdigit() and parts[7].isdigit() and parts[6].isdigit()
        ]


def src(filename):
    if "R.txt" in filename:
        return len(set(line.split(",")[1] for line in open(filename) if line.split(",")[1].isdigit()))
    elif "W.txt" in filename:
        return len(set(line.split(",")[3] for line in open(filename) if line.split(",")[3].isdigit()))
    return 1


def compute(filename, f):
    intervals = read_intervals(filename)
    merged_intervals = merge_intervals(intervals)
    num = src(filename)
    print(f"{filename[-5]}, {num}, {len(merged_intervals)}", file=f)
    total_bandwidth = [math.inf, -math.inf]
    weighted_bandwidth_sum = 0
    total_count = 0
    for start, end, count in merged_intervals:
        bandwidth = count * 128 / (end - start) / num
        total_bandwidth[0] = min(total_bandwidth[0], bandwidth)
        total_bandwidth[1] = max(total_bandwidth[1], bandwidth)
        weighted_bandwidth_sum += bandwidth * count
        total_count += count
        print(f"Interval: {start} to {end}, count: {count}, bandwidth: {bandwidth:.1f}", file=f)

    weighted_bandwidth = weighted_bandwidth_sum / total_count if total_count > 0 else 0
    print(f"Weighted bandwidth: {weighted_bandwidth:.1f}", file=f)

    latency = []
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split(",")
            if data[0].isdigit():
                t0 = int(data[0])
                t1 = int(data[7])
                for _ in range(int(data[6])):
                    latency.append(t1 - t0)
    if not latency:
        return (0, 0, 0, 0)

    print(f"There are {len(latency)} {filename[-5]} flits.", file=f)
    print(
        f"Latency(ns): Max={max(latency):.1f},Min={min(latency):.1f},Avg={sum(latency) / len(latency):.1f}\n",
        file=f,
    )
    return (
        weighted_bandwidth,
        max(latency),
        min(latency),
        sum(latency) / len(latency),
    )


if __name__ == "__main__":

    cwd = os.getcwd()
    result_dir = os.path.join(cwd, r"..\Result\mesh\v7-32\LLaMa2_Attention_QKV_Decode_Trace-2025-01-17 10-52-42")
    # result_dir = os.path.join(cwd, r"..\Result\cross ring\v7-32\5x4\demo2")
    path = os.path.join(result_dir, "total_result.txt")
    with open(path, "w", encoding="utf-8") as f:
        for item in os.listdir(result_dir):
            if item.startswith("Result"):
                file_path = os.path.join(result_dir, item)
                print(file_path, file=f)
                compute(file_path, f)
                print("", file=f)
    #     filename = os.path.join(cwd,item)
    #     main()
    # filename = 'Result_R_ins_SG2262_Ring_all_reduce_8cluster_all2all_0822_fix_Trace.txt'
    # main(filename)
    # filename = 'Result_W_ins_SG2262_Ring_all_reduce_8cluster_all2all_0822_fix_Trace.txt'
    # main(filename)
