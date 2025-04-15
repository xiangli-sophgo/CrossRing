import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from collections import defaultdict
from matplotlib.font_manager import FontProperties
import pandas as pd
import numpy as np
import time
import csv


class AddressStat:
    def __init__(self, interval_num=20):
        self.interval_num = interval_num
        self.init_params()
        self.results = []  # To store results for each directory

    def init_params(self):
        self.shared_64_count = 0
        self.shared_8_count = 0
        self.private_count = 0
        self.read_flit_count = 0
        self.write_flit_count = 0
        self.total_flit_count = 0
        self.total_request_count = 0
        self.request_end_time = -1
        self.time_distribution = defaultdict(lambda: {"R": 0, "W": 0, "flit_num": 0})

    def classify_address(self, addr, flit_num):
        addr = int(addr, base=16)
        if 0x04_0000_0000 <= addr < 0x08_0000_0000:
            self.shared_64_count += flit_num
        elif 0x08_0000_0000 <= addr < 0x10_0000_0000:
            self.shared_8_count += flit_num
        elif 0x10_0000_0000 <= addr < 0x20_0000_0000:
            self.private_count += flit_num
        else:
            pass

    def process_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            data = line.strip().split(",")
            if len(data) > 1:
                time = int(data[0])
                addr = data[1]
                operation = data[2]
                flit_num = int(data[-1])

                try:
                    self.classify_address(addr, flit_num)
                except ValueError as e:
                    print(f"Error processing address in file {file_path}: {e}")

                if operation == "R":
                    self.read_flit_count += flit_num
                    self.time_distribution[time]["R"] += flit_num
                elif operation == "W":
                    self.write_flit_count += flit_num
                    self.time_distribution[time]["W"] += flit_num
                else:
                    print(f"Unknown operation '{operation}' in file {file_path}")

                self.total_flit_count += flit_num
                self.total_request_count += 1

        if lines:
            self.request_end_time = max(self.request_end_time, int(lines[-1].strip().split(",")[0]))

    def process_folder(self, input_folder, plot_data):
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) == 0:
                    continue
                self.process_file(file_path)

            if not files:
                continue

            # Calculate statistics
            dir_name = root.rsplit("/", 1)[-1]
            shared_64_percent = 100 * self.shared_64_count / self.total_flit_count if self.total_flit_count > 0 else 0
            shared_8_percent = 100 * self.shared_8_count / self.total_flit_count if self.total_flit_count > 0 else 0
            private_percent = 100 * self.private_count / self.total_flit_count if self.total_flit_count > 0 else 0
            read_percent = 100 * self.read_flit_count / self.total_flit_count if self.total_flit_count > 0 else 0
            write_percent = 100 * self.write_flit_count / self.total_flit_count if self.total_flit_count > 0 else 0
            read_flit_per_cycle = self.read_flit_count / (32 * self.request_end_time) if self.request_end_time > 0 else 0
            write_flit_per_cycle = self.write_flit_count / (32 * self.request_end_time) if self.request_end_time > 0 else 0
            total_bandwidth = self.total_flit_count * 128 / (1024 * self.request_end_time) if self.request_end_time > 0 else 0
            total_flit_per_cycle = self.total_flit_count / (32 * self.request_end_time) if self.request_end_time > 0 else 0

            # Print to console
            print(f"Directory: {dir_name}")
            print(f"64 shared flits: {self.shared_64_count}, {shared_64_percent:.1f} %")
            print(f"8 shared flits: {self.shared_8_count}, {shared_8_percent:.1f} %")
            print(f"Private flits: {self.private_count}, {private_percent:.1f} %")
            print(f"Read flit: {self.read_flit_count}, {read_percent:.1f}%, {read_flit_per_cycle:.2f} flit/Cycle/IP")
            print(f"Write flit: {self.write_flit_count}, {write_percent:.1f}%, {write_flit_per_cycle:.2f} flit/Cycle/IP")
            print(f"Total flit num: {self.total_flit_count}, {total_bandwidth:.2f} TB/s, {total_flit_per_cycle:.2f} flit/Cycle/IP")
            print(f"Total Request num: {self.total_request_count}")
            print(f"Request end time: {self.request_end_time} \n")

            # Store results for CSV
            self.results.append(
                {
                    "Traffic_name": dir_name,
                    # "64_shared_flits": self.shared_64_count,
                    "64_shared_percent": shared_64_percent / 100,
                    # "8_shared_flits": self.shared_8_count,
                    "8_shared_percent": shared_8_percent / 100,
                    # "private_flits": self.private_count,
                    "private_percent": private_percent / 100,
                    # "read_flits": self.read_flit_count,
                    "read_percent": read_percent / 100,
                    # "read_flit_per_cycle": read_flit_per_cycle,
                    # "write_flits": self.write_flit_count,
                    "write_percent": write_percent / 100,
                    # "write_flit_per_cycle": write_flit_per_cycle,
                    "total_flits": self.total_flit_count,
                    "total_bandwidth": total_bandwidth,
                    # "total_flit_per_cycle": total_flit_per_cycle,
                    "total_requests": self.total_request_count,
                    "end_time": self.request_end_time,
                }
            )
            if plot_data:
                self.plot_time_distribution(dir_name, self.request_end_time // self.interval_num)
            self.init_params()

    def aggregate_time_distribution(self, interval):
        # 将时间段内的 R 和 W 请求进行汇总
        aggregated_distribution = defaultdict(lambda: {"R": 0, "W": 0})
        for time, counts in self.time_distribution.items():
            interval_key = time // interval  # 计算时间段
            aggregated_distribution[interval_key]["R"] += counts["R"]
            aggregated_distribution[interval_key]["W"] += counts["W"]
        return aggregated_distribution

    def plot_time_distribution(self, file_name, interval):
        # 汇总时间分布
        aggregated_distribution = self.aggregate_time_distribution(interval)

        # 创建完整的时间段
        min_time = min(aggregated_distribution.keys())
        max_time = max(aggregated_distribution.keys())
        complete_intervals = range(min_time, max_time + 1)  # 生成完整的时间段

        # 确保每个时间段都有数据
        read_counts = []
        write_counts = []
        time_labels = []
        for i in complete_intervals:
            if i in aggregated_distribution:
                read_counts.append(aggregated_distribution[i]["R"] * 128 / (1024 * interval))
                write_counts.append(aggregated_distribution[i]["W"] * 128 / (1024 * interval))
            else:
                read_counts.append(0)  # 如果没有数据,设置为 0
                write_counts.append(0)

        time_labels = [f"{i * interval}" for i in complete_intervals]

        sns.set_theme(style="darkgrid")  # 设置 Seaborn 风格

        fig, ax = plt.subplots()

        # 绘制读请求
        ax.bar(time_labels, read_counts, label="Read")
        ax.bar(time_labels, write_counts, label="Write", bottom=read_counts)

        # ax.plot(time_labels, read_counts, label="Read Requests")
        # ax.plot(time_labels, write_counts, label="Write Requests")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))  # 或者使用 10000
        # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))  # 格式化为千分位
        max_value = max([read + write for read, write in zip(read_counts, write_counts)])  # 计算最大值
        ax.set_ylim(0, max_value + 0.4)  # 设置 y 轴的上限,留出一些空间
        # plt.ylim(0, max(read_counts + write_counts) + 0.4)  # 增加 y 轴的上限
        # plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.title(file_name)
        plt.xlabel("Time (ns)")
        plt.ylabel("Flit Input Bandwidth (TB/s)")
        plt.legend()
        plt.tight_layout()  # 自动调整布局以避免重叠

        # 保存或显示图形
        # plt.savefig("../data_stat/v7-32/" + file_name[10:] + ".png", dpi=300)  # 保存为图片
        plt.show()  # 显示图形

        # 重置时间分布
        self.time_distribution = defaultdict(lambda: {"R": 0, "W": 0, "flit_num": 0})

    def save_to_csv(self, output_file):
        if not self.results:
            print("No results to save.")
            return

        # Define the CSV fieldnames
        fieldnames = [
            "Traffic_name",
            "end_time",
            "total_requests",
            "total_flits",
            "total_bandwidth",
            # "64_shared_flits",
            # "8_shared_flits",
            # "private_flits",
            "64_shared_percent",
            "8_shared_percent",
            "private_percent",
            # "read_flits",
            # "read_flit_per_cycle",
            "read_percent",
            # "write_flits",
            "write_percent",
            # "write_flit_per_cycle",
            # "total_flit_per_cycle",
        ]

        # Write to CSV
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Results saved to {output_file}")

    def run(self, input_folder, output_csv=None, plot_data=False):
        self.process_folder(input_folder, plot_data)
        if output_csv:
            self.save_to_csv(output_csv)
        print("Processing has been completed.")


if __name__ == "__main__":
    stat = AddressStat(200)
    # Specify the output CSV file path
    output_csv = None
    output_csv = r"../../Result/Data_csv/DeepSeek_traffic_stats.csv"
    stat.run(r"../../traffic/output_DeepSeek/step1_flatten/", output_csv, plot_data=1)
