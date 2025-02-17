import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from collections import defaultdict
import math


class BandwidthAnalyzer:
    def __init__(self, interval_num=10):
        self.interval_num = interval_num
        self.init_params()

    def init_params(self):
        self.bandwidths_read = defaultdict(list)
        self.bandwidths_write = defaultdict(list)
        self.read_intervals = defaultdict(list)
        self.write_intervals = defaultdict(list)
        self.read_duration = [math.inf, -math.inf]
        self.write_duration = [math.inf, -math.inf]

        self.read_src_num = -1
        self.write_src_num = -1

    def read_data(self):
        if "R.txt" in self.filename:
            cdma_set = set()
            with open(self.filename, "r") as file:
                for line in file:
                    parts = line.strip().split(",")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[7].isdigit():
                        cdma_id = parts[1]
                        if cdma_id.isdigit():
                            cdma_set.add(cdma_id)
                        dest_id = int(parts[1])
                        start = int(parts[0])
                        end = int(parts[7])
                        flit_num = int(parts[6])
                        self.read_duration[0] = min(start, self.read_duration[0])
                        self.read_duration[1] = max(end, self.read_duration[1])
                        self.read_intervals[dest_id].append((start, end, flit_num))
                        self.read_intervals["all"].append((start, end, flit_num))
                self.read_src_num = len(cdma_set)
        elif "W.txt" in self.filename:
            ddr_set = set()
            with open(self.filename, "r") as file:
                for line in file:
                    parts = line.strip().split(",")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[7].isdigit():
                        ddr_id = parts[1]
                        if ddr_id.isdigit():
                            ddr_set.add(ddr_id)
                        dest_id = int(parts[1])
                        start = int(parts[0])
                        end = int(parts[7])
                        flit_num = int(parts[6])
                        self.write_duration[0] = min(start, self.write_duration[0])
                        self.write_duration[1] = max(end, self.write_duration[1])
                        self.write_intervals[dest_id].append((start, end, flit_num))
                        self.write_intervals["all"].append((start, end, flit_num))
                self.write_src_num = len(ddr_set)

    def compute_bandwidth_per_ip(self):
        # 计算读取和写入带宽
        min_time = min(min(start for start, _, _ in intervals) for intervals in self.read_intervals.values())
        max_time = max(max(end for _, end, _ in intervals) for intervals in self.read_intervals.values())

        interval_length = (max_time - min_time) // self.interval_num

        # 计算每个 dest_id 的读取带宽
        for dest_id, intervals in self.read_intervals.items():
            for i in range(self.interval_num):
                interval_start = min_time + i * interval_length
                interval_end = min_time + (i + 1) * interval_length

                active_request_flits = 0
                for start, end, flit_num in intervals:
                    overlap_start = max(start, interval_start)
                    overlap_end = min(end, interval_end)
                    if overlap_start < overlap_end:  # 有重叠
                        overlap_duration = overlap_end - overlap_start
                        total_duration = end - start
                        active_request_flits += flit_num * (overlap_duration / total_duration)

                if interval_end > interval_start:
                    bandwidth = (active_request_flits * 128) / (interval_length * self.read_src_num)
                else:
                    bandwidth = 0

                self.bandwidths_read[dest_id].append((interval_start, bandwidth))

        # 计算每个 dest_id 的写入带宽
        for dest_id, intervals in self.write_intervals.items():
            for i in range(self.interval_num):
                interval_start = min_time + i * interval_length
                interval_end = min_time + (i + 1) * interval_length

                active_request_flits = 0
                for start, end, flit_num in intervals:
                    overlap_start = max(start, interval_start)
                    overlap_end = min(end, interval_end)
                    if overlap_start < overlap_end:  # 有重叠
                        overlap_duration = overlap_end - overlap_start
                        total_duration = end - start
                        active_request_flits += flit_num * (overlap_duration / total_duration)

                if interval_end > interval_start:
                    bandwidth = (active_request_flits * 128) / (interval_length * self.write_src_num)
                else:
                    bandwidth = 0

                self.bandwidths_write[dest_id].append((interval_start, bandwidth))

    def compute_bandwidth_all(self):
        # 计算整体的最小和最大时间
        if self.write_intervals["all"]:
            min_time = min(self.read_intervals["all"][0][0], self.write_intervals["all"][0][0])
            max_time = max(self.read_intervals["all"][-1][1], self.write_intervals["all"][-1][1])
        else:
            min_time = self.read_intervals["all"][0][0]
            max_time = self.read_intervals["all"][-1][1]

        # 划分区间
        interval_length = (max_time - min_time) / self.interval_num

        # 计算每个区间的活动请求数和带宽
        for i in range(self.interval_num):
            interval_start = min_time + i * interval_length
            interval_end = min_time + (i + 1) * interval_length

            active_request_flits = 0
            for start, end, flit_num in self.read_intervals["all"]:
                # 计算重叠部分
                overlap_start = max(start, interval_start)
                overlap_end = min(end, interval_end)
                if overlap_start < overlap_end:  # 有重叠
                    overlap_duration = overlap_end - overlap_start
                    total_duration = end - start
                    active_request_flits += flit_num * (overlap_duration / total_duration)

            if interval_end > interval_start:
                bandwidth = (active_request_flits * 128) / (interval_length * self.read_src_num)
            else:
                bandwidth = 0

            self.bandwidths_read["all"].append((interval_start, bandwidth, active_request_flits))

        # total = 0
        # weighted_BW = 0
        # for _, bandwidth, active_request_flits in self.bandwidths_read["all"]:
        #     weighted_BW += bandwidth * active_request_flits
        #     total += active_request_flits
        # print(f"{weighted_BW / total:.2f}")

        for i in range(self.interval_num):
            interval_start = min_time + i * interval_length
            interval_end = min_time + (i + 1) * interval_length

            active_request_flits = 0
            for start, end, flit_num in self.write_intervals["all"]:
                # 计算重叠部分
                overlap_start = max(start, interval_start)
                overlap_end = min(end, interval_end)
                if overlap_start < overlap_end:  # 有重叠
                    overlap_duration = overlap_end - overlap_start
                    total_duration = end - start
                    active_request_flits += flit_num * (overlap_duration / total_duration)

            # 计算带宽
            if interval_end > interval_start and self.write_src_num > 0:
                bandwidth = (active_request_flits * 128) / (interval_length * self.write_src_num)
            else:
                bandwidth = 0

            self.bandwidths_write["all"].append((interval_start, bandwidth, active_request_flits))

    def print_bandwidth(self):
        for interval_start, bandwidth in self.bandwidths_read:
            print(f"读时间区间: {interval_start} ns, 带宽: {bandwidth:.1f} GB/s")
        for interval_start, bandwidth in self.bandwidths_write:
            print(f"写时间区间: {interval_start} ns, 带宽: {bandwidth:.1f} GB/s")
        print()

    def plot_bandwidth_all(self):
        # 提取时间区间和带宽值
        intervals = [interval_start for interval_start, _, _ in self.bandwidths_read["all"]]

        read_bandwidth_values = [bandwidth for _, bandwidth, _ in self.bandwidths_read["all"]]
        write_bandwidth_values = [bandwidth for _, bandwidth, _ in self.bandwidths_write["all"]]

        sns.set_theme(style="darkgrid")  # 设置 Seaborn 风格
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        # 绘制读带宽
        ax.bar(intervals, read_bandwidth_values, width=(intervals[1] - intervals[0]), label="Read")

        # 绘制写带宽，堆叠在读带宽上面
        ax.bar(intervals, write_bandwidth_values, width=(intervals[1] - intervals[0]), bottom=read_bandwidth_values, label="Write")

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Bandwidth (GB/s)")
        ax.set_title(self.itemname[7:-6] + " Read and Write Bandwidth")
        max_value = max([read + write for read, write in zip(read_bandwidth_values, write_bandwidth_values)])  # 计算最大值
        ax.set_ylim(0, max_value * 1.02)  # 设置 y 轴的上限，留出一些空间
        ax.legend()
        plt.tight_layout(pad=1.4)
        # plt.savefig("../Result/" + self.itemname[7:-6] + " Bandwidth.png", dpi=300)
        plt.show()

    def plot_bandwidth_per_ip(self):
        sns.set_theme(style="darkgrid")

        for dest_id, intervals in self.bandwidths_read.items():
            fig, ax = plt.subplots()
            read_bandwidth_values = [bandwidth for _, bandwidth in intervals]
            interval_starts = [interval_start for interval_start, _ in intervals]
            ax.bar(interval_starts, read_bandwidth_values, width=(interval_starts[1] - interval_starts[0]), label=f"Read {dest_id}")

            # for dest_id, intervals in self.bandwidths_write.items():
            intervals = self.bandwidths_write[dest_id]
            write_bandwidth_values = [bandwidth for _, bandwidth in intervals]
            interval_starts = [interval_start for interval_start, _ in intervals]
            ax.bar(
                interval_starts,
                write_bandwidth_values,
                width=(interval_starts[1] - interval_starts[0]),
                label=f"Write {dest_id}",
                bottom=read_bandwidth_values,
            )

            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Bandwidth (GB/s)")
            max_value = max([read + write for read, write in zip(read_bandwidth_values, write_bandwidth_values)])  # 计算最大值
            ax.set_ylim(0, max_value + 0.2)  # 设置 y 轴的上限，留出一些空间
            ax.set_title("Read and Write Bandwidth per Destination ID")
            ax.legend()
            plt.tight_layout()
            plt.show()

    def compute_weighted_overall_bandwidth(self):
        # 初始化差分数组
        read_time_flit = [0] * (self.read_duration[1] + 1)
        records = []

        total_read_flit = 0
        # 遍历读取请求的区间，进行区间更新
        for intervals in self.read_intervals.values():
            for start, end, flit_num in intervals:
                # records.append((start, end, flit_num, flit_num * 128 / (end - start)))
                read_time_flit[start] += flit_num * 128 / (end - start)
                read_time_flit[end] -= flit_num * 128 / (end - start)
                # total_read_flit += flit_num

        # # 计算加权读取带宽
        weighted_read_bandwidth = 0
        total_read_duration = 0

        # # 计算前缀和并计算加权带宽
        current_flit_num = 0
        for t in range(self.read_duration[0], self.read_duration[1] + 1):
            current_flit_num += read_time_flit[t]  # 更新当前时间点的 flit 数量
            if current_flit_num > 0:
                weighted_read_bandwidth += current_flit_num * current_flit_num  # 每个 flit 假设 128 bits
                total_read_duration += current_flit_num  # 假设每个时间单位的持续时间为 1

        # # 重复相同的步骤来计算写入请求的带宽
        # write_time_flit = [0] * (self.write_duration[1] + 1)  # 清空差分数组
        # total_write_flit = 0
        # for intervals in self.write_intervals.values():
        #     for start, end, flit_num in intervals:
        #         write_time_flit[start] += flit_num * 128 / (end - start)
        #         write_time_flit[end] -= flit_num * 128 / (end - start)
        #         total_write_flit += flit_num

        # # 计算加权写入带宽
        # weighted_write_bandwidth = 0
        # total_write_duration = 0

        # current_flit_num = 0
        # for t in range(self.write_duration[0], self.write_duration[1] + 1):
        #     current_flit_num += write_time_flit[t]  # 更新当前时间点的 flit 数量
        #     if current_flit_num > 0:
        #         weighted_write_bandwidth += current_flit_num * current_flit_num  # 每个 flit 假设 128 bits
        #         total_write_duration += current_flit_num  # 假设每个时间单位的持续时间为 1

        # 计算整体带宽
        # overall_read_bandwidth = sum(r[2] * r[3] for r in records) / sum(r[3] for r in records)

        overall_read_bandwidth = weighted_read_bandwidth / total_read_duration / self.read_src_num if total_read_duration > 0 else 0
        # overall_write_bandwidth = weighted_write_bandwidth / total_write_duration / self.write_src_num if total_write_duration > 0 else 0

        # 输出结果
        print(self.itemname[7:-6])
        print(f"整体读取带宽: {overall_read_bandwidth:.1f} GB/s")
        # print(f"整体写入带宽: {overall_write_bandwidth:.1f} GB/s\n")

    def run(self, input_folder):
        for root, dirs, files in os.walk(input_folder):
            if not files:
                continue
            for item in files:
                if item.startswith("Result"):
                    self.itemname = item
                    self.filename = os.path.join(root, item)
                    self.read_data()

            # self.print_bandwidth()  # 如果需要打印带宽结果，可以取消注释

            # 画出每一个IP的带宽随时间变化图
            # self.compute_bandwidth_per_dest()
            # self.plot_bandwidth_per_ip()

            # 画出整体NoC的带宽随时间变化图
            self.compute_bandwidth_all()
            self.plot_bandwidth_all()

            # self.compute_weighted_overall_bandwidth()

            # 在遍历所有文件后绘制带宽
            self.init_params()


if __name__ == "__main__":
    # 使用示例
    # directory = r"..\Result\test"
    directory = r"..\Result\cross ring\v7-32\4x9"
    analyzer = BandwidthAnalyzer(100)
    analyzer.run(directory)
