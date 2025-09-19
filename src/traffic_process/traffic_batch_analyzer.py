import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from collections import defaultdict
import csv


class AddressStat:
    def __init__(self, interval_num=20):
        self.interval_num = interval_num
        self.init_params()
        self.results = []  # To store results for each directory

    def init_params(self):
        self.shared_32ch_count = 0
        self.shared_16ch_count = 0
        self.shared_8ch_count = 0
        self.private_count = 0
        self.read_flit_count = 0
        self.write_flit_count = 0
        self.total_flit_count = 0
        self.total_request_count = 0
        self.request_end_time = -1
        self.time_distribution = defaultdict(lambda: {"R": 0, "W": 0, "flit_num": 0})

    def classify_address(self, addr, flit_num):
        addr = int(addr, base=16)
        if 0x80000000 <= addr < 0x100000000:
            # 32通道共享内存: 0x80000000 - 0xffffffff (2GB)
            self.shared_32ch_count += flit_num
        elif 0x100000000 <= addr < 0x500000000:
            # 16通道共享内存: 0x100000000 - 0x4ffffffff (16GB)
            self.shared_16ch_count += flit_num
        elif 0x500000000 <= addr < 0x700000000:
            # 8通道共享内存: 0x500000000 - 0x6ffffffff (8GB)
            self.shared_8ch_count += flit_num
        elif 0x700000000 <= addr < 0x1F00000000:
            # 私有内存: 0x700000000 - 0x1effffffff (96GB)
            self.private_count += flit_num
        else:
            # 未知地址范围，记录警告
            pass

    def process_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            data = line.strip().split(",")
            # 新格式：时间,源节点,源IP,目标地址,目标IP,请求类型,burst长度
            if len(data) >= 7:
                time = int(data[0])
                addr = data[3]  # 目标地址在第4列
                operation = data[5]  # 请求类型在第6列 (R/W)
                flit_num = int(data[6])  # burst长度在第7列

                try:
                    self.classify_address(addr, flit_num)
                except ValueError as e:
                    print(f"文件 {file_path} 中地址处理错误: {e}")

                if operation == "R":
                    self.read_flit_count += flit_num
                    self.time_distribution[time]["R"] += flit_num
                elif operation == "W":
                    self.write_flit_count += flit_num
                    self.time_distribution[time]["W"] += flit_num
                else:
                    print(f"文件 {file_path} 中未知操作类型 '{operation}'")

                self.total_flit_count += flit_num
                self.total_request_count += 1

        if lines:
            self.request_end_time = max(self.request_end_time, int(lines[-1].strip().split(",")[0]))

    def process_folder(self, input_folder, plot_data):
        # 直接处理指定文件夹中的txt文件，不递归子目录
        files = [f for f in os.listdir(input_folder) if f.endswith(".txt") and os.path.isfile(os.path.join(input_folder, f))]

        for file in files:
            file_path = os.path.join(input_folder, file)
            if os.path.getsize(file_path) == 0:
                print(f"跳过空文件: {file}")
                continue

            print(f"正在处理文件: {file}")
            # 重置统计数据，为每个文件单独统计
            self.init_params()
            self.process_file(file_path)

            if self.total_request_count == 0:
                print(f"文件 {file} 中没有有效数据")
                continue

            # 计算统计信息（使用文件名作为标识）
            file_name = os.path.splitext(file)[0]  # 去掉.txt扩展名
            shared_32ch_percent = 100 * self.shared_32ch_count / self.total_flit_count if self.total_flit_count > 0 else 0
            shared_16ch_percent = 100 * self.shared_16ch_count / self.total_flit_count if self.total_flit_count > 0 else 0
            shared_8ch_percent = 100 * self.shared_8ch_count / self.total_flit_count if self.total_flit_count > 0 else 0
            private_percent = 100 * self.private_count / self.total_flit_count if self.total_flit_count > 0 else 0
            read_percent = 100 * self.read_flit_count / self.total_flit_count if self.total_flit_count > 0 else 0
            write_percent = 100 * self.write_flit_count / self.total_flit_count if self.total_flit_count > 0 else 0
            read_flit_per_cycle = self.read_flit_count / (32 * self.request_end_time) if self.request_end_time > 0 else 0
            write_flit_per_cycle = self.write_flit_count / (32 * self.request_end_time) if self.request_end_time > 0 else 0
            total_bandwidth = self.total_flit_count * 128 / (1024 * self.request_end_time) if self.request_end_time > 0 else 0
            total_flit_per_cycle = self.total_flit_count / (32 * self.request_end_time) if self.request_end_time > 0 else 0

            # 控制台输出统计信息
            print(f"文件: {file_name}")
            print("=" * 50)
            print("内存访问分布:")
            print(f"  32通道共享内存: {self.shared_32ch_count:,} flits, {shared_32ch_percent:.1f}%")
            print(f"  16通道共享内存: {self.shared_16ch_count:,} flits, {shared_16ch_percent:.1f}%")
            print(f"  8通道共享内存:  {self.shared_8ch_count:,} flits, {shared_8ch_percent:.1f}%")
            print(f"  私有内存:       {self.private_count:,} flits, {private_percent:.1f}%")
            print("\n读写操作分布:")
            print(f"  读操作: {self.read_flit_count:,} flits, {read_percent:.1f}%, {read_flit_per_cycle:.2f} flit/周期/IP")
            print(f"  写操作: {self.write_flit_count:,} flits, {write_percent:.1f}%, {write_flit_per_cycle:.2f} flit/周期/IP")
            print(f"\n总体统计:")
            print(f"  总flit数量: {self.total_flit_count:,}, 总带宽: {total_bandwidth:.2f} TB/s")
            print(f"  总请求数量: {self.total_request_count:,}")
            print(f"  请求结束时间: {self.request_end_time:,} 周期")
            print("=" * 50 + "\n")

            # 存储CSV结果
            self.results.append(
                {
                    "流量名称": file_name,
                    "32通道共享内存占比": shared_32ch_percent / 100,
                    "16通道共享内存占比": shared_16ch_percent / 100,
                    "8通道共享内存占比": shared_8ch_percent / 100,
                    "私有内存占比": private_percent / 100,
                    "读操作占比": read_percent / 100,
                    "写操作占比": write_percent / 100,
                    "总flit数量": self.total_flit_count,
                    "总带宽TB_s": total_bandwidth,
                    "总请求数量": self.total_request_count,
                    "结束时间": self.request_end_time,
                    "32通道flit数": self.shared_32ch_count,
                    "16通道flit数": self.shared_16ch_count,
                    "8通道flit数": self.shared_8ch_count,
                    "私有内存flit数": self.private_count,
                    "读flit数": self.read_flit_count,
                    "写flit数": self.write_flit_count,
                }
            )
            # 暂时注释绘图功能
            # if plot_data:
            #     self.plot_time_distribution(dir_name, self.request_end_time // self.interval_num)
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
            print("没有结果可保存。")
            return

        # 定义CSV字段名
        fieldnames = [
            "流量名称",
            "结束时间",
            "总请求数量",
            "总flit数量",
            "总带宽TB_s",
            "32通道共享内存占比",
            "16通道共享内存占比",
            "8通道共享内存占比",
            "私有内存占比",
            "读操作占比",
            "写操作占比",
            "32通道flit数",
            "16通道flit数",
            "8通道flit数",
            "私有内存flit数",
            "读flit数",
            "写flit数",
        ]

        # 按流量名称排序结果
        sorted_results = sorted(self.results, key=lambda x: x["流量名称"])

        # 写入CSV文件，使用UTF-8 BOM确保Excel正确显示中文
        with open(output_file, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_results)

        print(f"结果已保存到: {output_file}")

    def run(self, input_folder, output_csv=None, plot_data=False):
        self.process_folder(input_folder, plot_data)
        if output_csv:
            self.save_to_csv(output_csv)
        print("流量分析处理完成。")


if __name__ == "__main__":
    stat = AddressStat(200)
    # Specify the output CSV file path
    output_csv = None
    output_csv = r"../../Result/Data_csv/DeepSeek_traffic_stats_0918.csv"
    stat.run(r"../../traffic/DeepSeek_0918/merged/", output_csv, plot_data=0)
