import os


# 定义函数来处理请求数据并计算带宽
def process_requests(file_path):
    # 初始化数据结构
    read_intervals = []
    write_intervals = []
    current_read_interval = None
    current_write_interval = None

    # 从文件中读取数据
    with open(file_path, "r") as file:
        for line in file:
            # 解析每一行
            parts = line.strip().split(",")
            start_time = int(parts[0])
            req_type = parts[5]
            flit_count = int(parts[6])

            if req_type == "W":
                if current_write_interval is None:
                    # 开始一个新的写区间
                    current_write_interval = {"start_time": start_time, "end_time": start_time, "total_flits": flit_count}
                else:
                    # 检查是否可以扩展当前写区间
                    if start_time <= current_write_interval["end_time"] + 20:
                        # 扩展当前写区间
                        current_write_interval["end_time"] = start_time
                        current_write_interval["total_flits"] += flit_count
                    else:
                        # 关闭当前写区间并开始一个新区间
                        write_intervals.append(current_write_interval)
                        current_write_interval = {"start_time": start_time, "end_time": start_time, "total_flits": flit_count}
            else:  # 处理读请求
                if current_read_interval is None:
                    # 开始一个新的读区间
                    current_read_interval = {"start_time": start_time, "end_time": start_time, "total_flits": flit_count}
                else:
                    # 检查是否可以扩展当前读区间
                    if start_time <= current_read_interval["end_time"] + 20:
                        # 扩展当前读区间
                        current_read_interval["end_time"] = start_time
                        current_read_interval["total_flits"] += flit_count
                    else:
                        # 关闭当前读区间并开始一个新区间
                        read_intervals.append(current_read_interval)
                        current_read_interval = {"start_time": start_time, "end_time": start_time, "total_flits": flit_count}

    # 不要忘记添加最后一个区间
    if current_read_interval is not None:
        read_intervals.append(current_read_interval)
    if current_write_interval is not None:
        write_intervals.append(current_write_interval)

    # 初始化加权带宽计算所需的变量
    total_read_weighted_bandwidth = 0
    total_write_weighted_bandwidth = 0
    total_read_flits = 0
    total_write_flits = 0

    # 计算读区间的加权带宽
    for interval in read_intervals:
        start = interval["start_time"]
        end = interval["end_time"]
        total_flits_interval = interval["total_flits"]

        # 计算带宽
        duration = end - start
        if duration > 0:  # 确保时间间隔大于0
            bandwidth = (total_flits_interval * 128) / (32 * duration)
            total_read_weighted_bandwidth += bandwidth * total_flits_interval
            total_read_flits += total_flits_interval

    # 计算写区间的加权带宽
    for interval in write_intervals:
        start = interval["start_time"]
        end = interval["end_time"]
        total_flits_interval = interval["total_flits"]

        # 计算带宽
        duration = end - start
        if duration > 0:  # 确保时间间隔大于0
            bandwidth = (total_flits_interval * 128) / (32 * duration)
            total_write_weighted_bandwidth += bandwidth * total_flits_interval
            total_write_flits += total_flits_interval

    # 计算整体加权带宽
    overall_read_weighted_bandwidth = total_read_weighted_bandwidth / total_read_flits if total_read_flits > 0 else 0
    overall_write_weighted_bandwidth = total_write_weighted_bandwidth / total_write_flits if total_write_flits > 0 else 0

    # 输出整体加权带宽
    print(f"Overall Read Weighted Bandwidth: {overall_read_weighted_bandwidth:.1f} GB/s")
    print(f"Overall Write Weighted Bandwidth: {overall_write_weighted_bandwidth:.1f} GB/s", end="\n\n")


# 主函数,遍历文件夹中的所有文件
def main(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # 只处理以 .txt 结尾的文件
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            process_requests(file_path)


# 调用主函数并传入文件夹路径
if __name__ == "__main__":
    main(r"C:\Users\Administrator\Documents\NOC\Trace\output-v8-32\2M\step5_data_merge")
