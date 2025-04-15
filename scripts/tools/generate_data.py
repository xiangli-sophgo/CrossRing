import random
from math import gcd
from collections import defaultdict


class TrafficGenerator:
    def __init__(self, read_duration=64, write_duration=64, total_bandwidth=128):
        """
        :param read_duration: 读操作时间窗口长度(ns)
        :param write_duration: 写操作时间窗口长度(ns)
        :param total_bandwidth: 总带宽单位(GB/s)
        """
        self.read_duration = read_duration
        self.write_duration = write_duration
        self.cycle_duration = read_duration + write_duration
        self.total_bandwidth = total_bandwidth

    def calculate_time_points(self, speed, burst, is_read=True):
        """计算读/写时间点序列"""
        window_duration = self.read_duration if is_read else self.write_duration
        total_transfers = speed * window_duration // (self.total_bandwidth * burst)

        if total_transfers == 0:
            return []

        # 计算均匀分布的时间间隔(保持整数)
        base_interval = window_duration // total_transfers
        remainder = window_duration % total_transfers
        intervals = [base_interval + (1 if i < remainder else 0) for i in range(total_transfers)]

        # 生成时间点序列
        time_points = []
        current_time = 0
        for interval in intervals:
            time_points.append(current_time)
            current_time += interval

        return time_points


def generate_data(topo, interval_count, file_name, sdma_pos, gdma_pos, ddr_pos, l2m_pos, speed, burst, flow_type=0, mix_ratios=None):
    """
    :param interval_count: 读写周期数量(每个周期=read_duration+write_duration)
    """
    # 初始化流量生成器(默认读128ns+写128ns)
    generator = TrafficGenerator(read_duration=128, write_duration=128)
    data_all = []

    def generate_entries(src_pos, src_type, dest_type, dest_pos, operation, burst, flow_type, speed, interval_count):
        """生成指定模式的流量条目"""
        is_read = operation == "R"
        time_pattern = generator.calculate_time_points(speed, burst, is_read)
        dest_len = len(dest_pos)
        entries = []

        for cycle in range(interval_count):
            base_time = cycle * generator.cycle_duration
            # 读操作需要加上0，写操作需要加上read_duration
            time_offset = 0 if is_read else generator.read_duration

            if flow_type == 1:  # 8-shared分组
                for src in src_pos:
                    group = src // 8
                    group_dests = dest_pos[group * 8 : (group + 1) * 8]
                    for t in time_pattern:
                        dest = random.choice(group_dests)
                        entries.append(f"{base_time + time_offset + t},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

            elif flow_type == 2:  # private模式
                for src in src_pos:
                    dest = src if src in dest_pos else dest_pos[src % dest_len]
                    for t in time_pattern:
                        entries.append(f"{base_time + time_offset + t},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

            else:  # 32-shared模式
                for src in src_pos:
                    for t in time_pattern:
                        dest = random.choice(dest_pos)
                        # if src == 3:
                        #     entries.append(f"{base_time + time_offset + t + 4},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")
                        # else:
                        #     entries.append(f"{base_time + time_offset + t},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")
                        entries.append(f"{base_time + time_offset + t},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

        return entries

    def generate_mixed_entries(src_pos, src_type, dest_type, dest_pos, operation, burst, ratios):
        """混合模式生成（保持原有逻辑，但区分读写时间）"""
        is_read = operation == "R"
        total_speed = speed[burst]
        mode_speeds = {k: int(total_speed * v) for k, v in ratios.items()}
        dest_len = len(dest_pos)
        entries = []

        for cycle in range(interval_count):
            base_time = cycle * generator.cycle_duration
            time_offset = 0 if is_read else generator.read_duration

            for mode, mode_speed in mode_speeds.items():
                if mode_speed <= 0:
                    continue

                time_pattern = generator.calculate_time_points(mode_speed, burst, is_read)

                for src in src_pos:
                    for t in time_pattern:
                        if mode == 0:  # 32-shared
                            dest = random.choice(dest_pos)
                        elif mode == 1:  # 8-shared
                            group = src // 8
                            group_dests = dest_pos[group * 8 : (group + 1) * 8]
                            dest = random.choice(group_dests)
                        else:  # private
                            dest = src if src in dest_pos else dest_pos[src % dest_len]
                        entries.append(f"{base_time + time_offset + t},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

        return entries

    # 生成数据逻辑
    if topo in ["4x9", "9x4", "4x5", "5x4"]:
        if flow_type == 3:
            mix_ratios = mix_ratios or {0: 0.4, 1: 0.4, 2: 0.2}
            data_all.extend(generate_mixed_entries(sdma_pos, "gdma", "ddr", ddr_pos, "R", burst, mix_ratios))
            data_all.extend(generate_mixed_entries(sdma_pos, "gdma", "ddr", l2m_pos, "W", burst, mix_ratios))
        else:
            # data_all.extend(generate_entries(gdma_pos, "gdma", "ddr", ddr_pos, "R", burst, flow_type, speed[burst], interval_count))
            data_all.extend(generate_entries(gdma_pos, "gdma", "ddr", l2m_pos, "W", burst, flow_type, speed[burst], interval_count))

    elif topo == "3x3":
        if flow_type == 3:
            mix_ratios = mix_ratios or {0: 0.4, 1: 0.4, 2: 0.2}
            data_all.extend(generate_mixed_entries(sdma_pos, "sdma", "ddr", ddr_pos, "R", burst, mix_ratios))
            data_all.extend(generate_mixed_entries(sdma_pos, "sdma", "l2m", l2m_pos, "W", burst, mix_ratios))
            data_all.extend(generate_mixed_entries(gdma_pos, "gdma", "l2m", l2m_pos, "R", burst, mix_ratios))
        else:
            data_all.extend(generate_entries(sdma_pos, "sdma", "ddr", ddr_pos, "R", burst, flow_type, speed[burst], interval_count))
            data_all.extend(generate_entries(sdma_pos, "sdma", "l2m", l2m_pos, "W", burst, flow_type, speed[burst], interval_count))
            data_all.extend(generate_entries(gdma_pos, "gdma", "l2m", l2m_pos, "R", burst, flow_type, speed[burst], interval_count))

    # 排序并写入文件
    with open(file_name, "w") as f:
        f.writelines(sorted(data_all, key=lambda x: int(x.split(",")[0])))


# 示例使用
if __name__ == "__main__":
    # 参数配置
    topo = "5x4"
    interval_count = 32
    file_name = "../../test_data/traffic_ITag_0414.txt"

    num_ip = 32
    sdma_pos = range(num_ip)
    # gdma_pos = range(1)
    gdma_pos = [0, 1, 2, 3]
    ddr_pos = range(15, 16)
    l2m_pos = range(15, 16)

    speed = {1: 128, 2: 68, 4: 128}  # 不同burst对应的带宽(GB/s)
    burst = 4

    # 生成数据(使用混合模式)
    generate_data(topo, interval_count, file_name, sdma_pos, gdma_pos, ddr_pos, l2m_pos, speed, burst, flow_type=0)
