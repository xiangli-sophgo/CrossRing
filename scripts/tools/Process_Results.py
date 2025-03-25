import os
import csv

# 定义 start_time 和 end_time
start_time = 5000
end_time = 25000

# 定义结果存储列表
results = []


# 递归遍历文件夹
def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".txt") and filename.startswith("Result_"):
                file_path = os.path.join(root, filename)
                process_file(file_path)


# 处理单个文件
def process_file(file_path):
    with open(file_path, "r") as file:
        data = [line.strip().split(",") for line in file if line.strip()]

        # 分别统计 R 和 W 的 burst_len
        if "Result_R" in file_path:
            r_burst_len = sum(int(row[6]) for row in data if row[5] == "R" and (start_time <= int(row[0]) <= end_time and start_time <= int(row[7]) <= end_time))
            w_burst_len = 0
        elif "Result_W" in file_path:
            r_burst_len = 0
            w_burst_len = sum(int(row[6]) for row in data if row[5] == "W" and (start_time <= int(row[0]) <= end_time and start_time <= int(row[7]) <= end_time))

        # 解析路径信息
        path_parts = file_path.split(os.sep)
        ro_tracker_ostd = int(path_parts[-4].split("_")[0])
        share_tracker_ostd = int(path_parts[-4].split("_")[1])
        topo_type = path_parts[-3]
        model_type = path_parts[4]

        # 将结果添加到结果列表
        results.append(
            {
                # "file_path": file_path,
                "ro_tracker_ostd": ro_tracker_ostd,
                "share_tracker_ostd": share_tracker_ostd,
                "topo_type": topo_type,
                "model_type": model_type,
                "ReadBandWidth": r_burst_len * 128 / (end_time - start_time) / 32,
                "WriteBandWidth": w_burst_len * 128 / (end_time - start_time) / 32,
                # "TotalBandWidth": (r_burst_len + w_burst_len) * 128 / (end_time - start_time) / 32,
            }
        )


# 主函数
def main():
    # 定义根文件夹路径
    root_folder = r"../../Result/CrossRing/REQ_RSP/FOP/SN_Tracker_OSTD_Results_459_fixed/"  # 替换为你的根文件夹路径

    # 处理文件夹
    process_folder(root_folder)

    # 将结果保存到 CSV 文件
    output_csv = r"../../Result/Params_csv/SN_Tracker_OSTD_Results_fixed_time_interval.csv"
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["ro_tracker_ostd", "share_tracker_ostd", "topo_type", "model_type", "ReadBandWidth", "WriteBandWidth"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"结果已保存到 {output_csv}")


if __name__ == "__main__":
    main()
