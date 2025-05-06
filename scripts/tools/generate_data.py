import random
from math import gcd
from collections import defaultdict
import numpy as np
import itertools
import random


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


def group_numbers():
    """生成特殊分组的 4 个组（每组 8 个数字）"""
    points = list(range(32))

    # 第一步：分成两个大组（0-15和16-31）
    group_a = points[:16]
    group_b = points[16:]

    # 第二步：将每个大组重塑为4x4矩阵
    matrix_a = np.array(group_a).reshape(4, 4)
    matrix_b = np.array(group_b).reshape(4, 4)

    # 第三步：将每个4x4矩阵分成4个2x2象限
    def split_quadrants(matrix):
        return [matrix[:2, :2].flatten(), matrix[:2, 2:].flatten(), matrix[2:, :2].flatten(), matrix[2:, 2:].flatten()]  # 左上  # 右上  # 左下  # 右下

    quadrants_a = split_quadrants(matrix_a)
    quadrants_b = split_quadrants(matrix_b)

    # 第四步：将两个大组对应的象限合并
    final_groups = [np.concatenate((a, b)) for a, b in zip(quadrants_a, quadrants_b)]

    # 转换为列表形式返回
    return [group.tolist() for group in final_groups]


def generate_data(topo, read_duration, write_duration, interval_count, file_name, sdma_pos, gdma_pos, ddr_pos, l2m_pos, speed, burst, flow_type=0, mix_ratios=None):
    """
    :param interval_count: 读写周期数量(每个周期=read_duration+write_duration)
    """
    # 初始化流量生成器(默认读128ns+写128ns)
    generator = TrafficGenerator(read_duration=read_duration, write_duration=write_duration)
    data_all = []

    # def generate_entries(src_pos, src_type, dest_map, operation, burst, flow_type, speed, interval_count, dest_access_mode="random"):
    #     """
    #     生成指定模式的流量条目

    #     参数:
    #     src_pos: list of source positions
    #     src_type: str, source类型
    #     dest_map: dict, 键为dest_type(str)，值为对应的dest_pos(list)
    #     operation: 'R' 或 'W'
    #     burst: int, burst长度
    #     flow_type: int, 1=8-shared, 2=private, 其他=32-shared
    #     speed: dict or list, 计算时间点所需参数
    #     interval_count: int, 周期数
    #     dest_access_mode: 'round_robin' 或 'random', 控制目标选择方式
    #     """
    #     if dest_access_mode not in ("round_robin", "random"):
    #         raise ValueError("dest_access_mode 必须是 'round_robin' 或 'random'")

    #     is_read = operation == "R"
    #     time_pattern = generator.calculate_time_points(speed, burst, is_read)
    #     entries = []
    #     # 构造 (dest_type, pos) 列表
    #     dest_items = [(dtype, pos) for dtype, poses in dest_map.items() for pos in poses]

    #     # 准备循环器：仅 round_robin 模式
    #     if dest_access_mode == "round_robin":
    #         if flow_type == 1:
    #             groups = group_numbers()
    #             group_cycles = {}
    #             for gid, group in enumerate(groups):
    #                 items = [item for item in dest_items if item[1] in group]
    #                 if items:
    #                     group_cycles[gid] = itertools.cycle(items)
    #         elif flow_type != 2:
    #             dest_cycle = itertools.cycle(dest_items)

    #     # 生成条目
    #     for cycle in range(interval_count):
    #         base_time = cycle * generator.cycle_duration
    #         time_offset = 0 if is_read else generator.read_duration

    #         if flow_type == 1:
    #             groups = group_numbers()
    #             for t in time_pattern:
    #                 for src in src_pos:
    #                     gid = next((i for i, g in enumerate(groups) if src in g), -1)
    #                     if gid == -1:
    #                         continue
    #                     if dest_access_mode == "random":
    #                         group_items = [item for item in dest_items if item[1] in groups[gid]]
    #                         dest_type, dest = random.choice(group_items)
    #                     else:
    #                         dest_type, dest = next(group_cycles[gid])
    #                     entries.append(f"{base_time + time_offset + t},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

    #         elif flow_type == 2:
    #             total = len(dest_items)
    #             for t in time_pattern:
    #                 for src in src_pos:
    #                     # private: 优先匹配相同 src
    #                     match = next((item for item in dest_items if item[1] == src), None)
    #                     if match:
    #                         dest_type, dest = match
    #                     else:
    #                         if dest_access_mode == "random":
    #                             dest_type, dest = random.choice(dest_items)
    #                         else:
    #                             dest_type, dest = next(dest_cycle)
    #                     entries.append(f"{base_time + time_offset + t},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

    #         else:
    #             for t in time_pattern:
    #                 for src in src_pos:
    #                     if dest_access_mode == "random":
    #                         dest_type, dest = random.choice(dest_items)
    #                     else:
    #                         dest_type, dest = next(dest_cycle)
    #                     entries.append(f"{base_time + time_offset + t},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

    #     return entries

    def generate_entries(src_pos, src_type, dest_map, operation, burst, flow_type, speed, interval_count, dest_access_mode="random"):
        """
        生成指定模式的流量条目，确保同一时刻所有dest都有访问

        参数:
        src_pos: list of source positions
        src_type: str, source类型
        dest_map: dict, 键为dest_type(str)，值为对应的dest_pos(list)
        operation: 'R' 或 'W'
        burst: int, burst长度
        flow_type: int, 1=8-shared, 2=private, 其他=32-shared
        speed: dict or list, 计算时间点所需参数
        interval_count: int, 周期数
        dest_access_mode: 'round_robin' 或 'random', 控制目标选择方式
        """
        if dest_access_mode not in ("round_robin", "random"):
            raise ValueError("dest_access_mode 必须是 'round_robin' 或 'random'")

        is_read = operation == "R"
        time_pattern = generator.calculate_time_points(speed, burst, is_read)
        entries = []
        # 构造 (dest_type, pos) 列表
        dest_items = [(dtype, pos) for dtype, poses in dest_map.items() for pos in poses]

        # 准备循环器和随机访问列表
        if dest_access_mode == "round_robin":
            if flow_type == 1:
                groups = group_numbers()
                group_cycles = {}
                for gid, group in enumerate(groups):
                    items = [item for item in dest_items if item[1] in group]
                    if items:
                        group_cycles[gid] = itertools.cycle(items)
            elif flow_type != 2:
                dest_cycle = itertools.cycle(dest_items)
        else:  # random模式
            if flow_type == 1:
                groups = group_numbers()
                group_items = {}
                for gid, group in enumerate(groups):
                    group_items[gid] = [item for item in dest_items if item[1] in group]
            else:
                # 为random模式创建所有可能的dest排列组合
                all_dest_permutations = list(itertools.permutations(dest_items))
                random.shuffle(all_dest_permutations)

        # 生成条目
        for cycle in range(interval_count):
            base_time = cycle * generator.cycle_duration
            time_offset = 0 if is_read else generator.read_duration

            if flow_type == 1:  # 8-shared
                groups = group_numbers()
                for t in time_pattern:
                    for src in src_pos:
                        gid = next((i for i, g in enumerate(groups) if src in g), -1)
                        if gid == -1:
                            continue
                        if dest_access_mode == "random":
                            # 确保组内所有dest都被访问
                            if not hasattr(generate_entries, "group_access_idx"):
                                generate_entries.group_access_idx = {}
                            if gid not in generate_entries.group_access_idx:
                                generate_entries.group_access_idx[gid] = 0

                            idx = generate_entries.group_access_idx[gid] % len(group_items[gid])
                            dest_type, dest = group_items[gid][idx]
                            generate_entries.group_access_idx[gid] += 1
                        else:
                            dest_type, dest = next(group_cycles[gid])
                        entries.append(f"{base_time + time_offset + t},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

            elif flow_type == 2:  # private
                for t in time_pattern:
                    for src in src_pos:
                        # private: 优先匹配相同 src
                        match = next((item for item in dest_items if item[1] == src), None)
                        if match:
                            dest_type, dest = match
                        else:
                            if dest_access_mode == "random":
                                # 使用排列组合确保所有dest都被访问
                                if not hasattr(generate_entries, "perm_idx"):
                                    generate_entries.perm_idx = 0

                                perm = all_dest_permutations[generate_entries.perm_idx % len(all_dest_permutations)]
                                dest_type, dest = perm[src_pos.index(src) % len(perm)]
                                generate_entries.perm_idx += 1
                            else:
                                dest_type, dest = next(dest_cycle)
                        entries.append(f"{base_time + time_offset + t},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

            else:  # 32-shared或其他
                for t in time_pattern:
                    if dest_access_mode == "random":
                        # 确保每个src访问不同的dest
                        if not hasattr(generate_entries, "dest_assignments"):
                            generate_entries.dest_assignments = {}

                        if t not in generate_entries.dest_assignments:
                            # 为这个时间点创建新的dest分配
                            shuffled_dests = random.sample(dest_items, len(dest_items))
                            # 如果src比dest多，循环使用dest
                            generate_entries.dest_assignments[t] = itertools.cycle(shuffled_dests)

                        dest_cycle = generate_entries.dest_assignments[t]
                    else:
                        dest_cycle = itertools.cycle(dest_items)

                    for src in src_pos:
                        dest_type, dest = next(dest_cycle)
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

                if mode == 0:  # 32-shared
                    shuffled_dest_pos = list(dest_pos)
                    random.shuffle(shuffled_dest_pos)
                    dest_iter = iter(shuffled_dest_pos)  # 用迭代器逐个取
                elif mode == 1:  # 8-shared
                    shuffled_group_dests = {}
                    for group in range(len(dest_pos) // 8):
                        group_dests = dest_pos[group * 8 : (group + 1) * 8]
                        random.shuffle(group_dests)
                        shuffled_group_dests[group] = iter(group_dests)  # 每组一个迭代器

                for src in src_pos:
                    for t in time_pattern:
                        if mode == 0:  # 32-shared
                            try:
                                dest = next(dest_iter)
                            except StopIteration:  # 如果用完，重新打乱（可选）
                                random.shuffle(shuffled_dest_pos)
                                dest_iter = iter(shuffled_dest_pos)
                                dest = next(dest_iter)

                        elif mode == 1:  # 8-shared
                            group = src // 8
                            try:
                                dest = next(shuffled_group_dests[group])
                            except StopIteration:  # 如果当前组用完，重新打乱（可选）
                                group_dests = dest_pos[group * 8 : (group + 1) * 8]
                                random.shuffle(group_dests)
                                shuffled_group_dests[group] = iter(group_dests)
                                dest = next(shuffled_group_dests[group])

                        else:  # private
                            dest = src if src in dest_pos else dest_pos[src % len(dest_pos)]

                        entries.append(f"{base_time + time_offset + t},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

        return entries

    # 生成数据逻辑
    if topo in ["4x9", "9x4", "4x5", "5x4"]:
        if flow_type == 3:
            mix_ratios = mix_ratios or {0: 0.4, 1: 0.4, 2: 0.2}
            data_all.extend(generate_mixed_entries(sdma_pos, "gdma", "ddr", ddr_pos, "R", burst, mix_ratios))
            data_all.extend(generate_mixed_entries(sdma_pos, "gdma", "ddr", l2m_pos, "W", burst, mix_ratios))
        else:
            # data_all.extend(generate_entries(gdma_pos, "gdma", ddr_map, "R", burst, flow_type, speed[burst], interval_count))
            data_all.extend(generate_entries(gdma_pos, "gdma", ddr_map, "W", burst, flow_type, speed[burst], interval_count))
            # data_all.extend(generate_entries(gdma_pos, "gdma", l2m_map, "R", burst, flow_type, speed[burst], interval_count))
            # data_all.extend(generate_entries(sdma_pos, "sdma", ddr_map, "R", burst, flow_type, speed[burst], interval_count))
            # data_all.extend(generate_entries(sdma_pos, "sdma", l2m_map, "R", burst, flow_type, speed[burst], interval_count))
            # data_all.extend(generate_entries(gdma_pos, "gdma", "l2m", ddr_pos, "W", burst, flow_type, speed[burst], interval_count))

            # data_all.extend(generate_entries(gdma_pos, "gdma", "ddr", ddr_pos, "R", burst, flow_type, speed[burst], interval_count))
            # data_all.extend(generate_entries(gdma_pos, "gdma", "l2m", l2m_pos, "W", burst, flow_type, speed[burst], interval_count))

    elif topo == "3x3":
        if flow_type == 3:
            mix_ratios = mix_ratios or {0: 0.4, 1: 0.4, 2: 0.2}
            data_all.extend(generate_mixed_entries(sdma_pos, "sdma", "ddr", ddr_pos, "R", burst, mix_ratios))
            data_all.extend(generate_mixed_entries(sdma_pos, "sdma", "l2m", l2m_pos, "W", burst, mix_ratios))
            data_all.extend(generate_mixed_entries(gdma_pos, "gdma", "l2m", l2m_pos, "R", burst, mix_ratios))
        else:
            # data_all.extend(generate_entries(gdma_pos, "gdma", l2m_map, "R", burst, flow_type, speed[burst], interval_count))
            # data_all.extend(generate_entries(sdma_pos, "sdma", ddr_map, "R", burst, flow_type, speed[burst], interval_count))
            data_all.extend(generate_entries(sdma_pos, "sdma", l2m_map, "R", burst, flow_type, speed[burst], interval_count))
            # data_all.extend(generate_entries(gdma_pos, "gdma", "l2m", l2m_pos, "R", burst, flow_type, speed[burst], interval_count))
            # data_all.extend(generate_entries(gdma_pos, "gdma", "l2m", l2m_pos, "W", burst, flow_type, speed[burst], interval_count))

    # 排序并写入文件
    with open(file_name, "w") as f:
        f.writelines(sorted(data_all, key=lambda x: int(x.split(",")[0])))


# 示例使用
if __name__ == "__main__":
    # 参数配置
    topo = "3x3"
    interval_count = 32
    file_name = "../../test_data/traffic_2260E_SDMA_WO_l2m_0506.txt"
    np.random.seed(428)

    if topo == "5x4":
        num_ip = 32
        sdma_pos = range(num_ip)
        gdma_pos = range(num_ip)
        ddr_map = {"ddr_1": range(num_ip)}
        l2m_map = {"l2m_1": range(num_ip // 2)}

    # SG2260E
    elif topo == "3x3":
        sdma_pos = [0, 2, 6, 8]
        gdma_pos = [0, 2, 6, 8]
        # gdma_pos = [0, 2]

        # ddr_map = {
        #     "ddr_1": [3],
        #     "ddr_2": [3],
        # }
        ddr_map = {"ddr_1": [0, 2, 3, 5, 6, 8], "ddr_2": [3, 5]}
        l2m_map = {"l2m_1": [1, 7], "l2m_2": [1, 7]}

    speed = {1: 128, 2: 128, 4: 128}  # 不同burst对应的带宽(GB/s)
    burst = 2
    read_duration = 128
    write_duration = 128

    # 生成数据(使用混合模式)
    generate_data(topo, read_duration, write_duration, interval_count, file_name, sdma_pos, gdma_pos, ddr_map, l2m_map, speed, burst, flow_type=0)
