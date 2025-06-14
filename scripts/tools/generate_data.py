import random
from math import gcd
from collections import defaultdict
import numpy as np
import itertools
import random
from itertools import product


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
        """计算读/写时间点序列，使请求均匀分布"""
        window_duration = self.read_duration if is_read else self.write_duration
        # 计算请求数
        total_transfers = speed * window_duration // (self.total_bandwidth * burst)
        if total_transfers == 0:
            return []
        # 均匀分布
        time_points = [int(i * window_duration / total_transfers) for i in range(total_transfers)]
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


def generate_data(topo, interval_count, file_name, sdma_map, gdma_map, ddr_map, l2m_map, speed, burst, flow_type=0, mix_ratios=None, overlap=0):
    """
    :param interval_count: 读写周期数量(每个周期=read_duration+write_duration)
    """
    # 初始化流量生成器(默认读128ns+写128ns)
    # generator = TrafficGenerator(read_duration=read_duration, write_duration=write_duration)
    data_all = []

    def generate_entries(src_map, dest_map, operation, burst, flow_type, speed, interval_count, dest_access_mode="round_robin", overlap=True):
        """
        生成指定模式的流量条目，确保同一时刻所有dest都有访问

        参数:
        src_map: dict, 键为src_type(str)，值为对应的src_pos(list)
        dest_map: dict, 键为dest_type(str)，值为对应的dest_pos(list)
        operation: 'R' 或 'W'
        burst: int, burst长度
        flow_type: int, 1=8-shared, 2=private, 其他=32-shared
        speed: dict or list, 计算时间点所需参数
        interval_count: int, 周期数
        dest_access_mode: 'round_robin' 或 'random'
        overlap: bool, 是否让读写请求时间点重叠（True 时 time_offset 恒为 0）
        """
        if dest_access_mode not in ("round_robin", "random"):
            raise ValueError("dest_access_mode 必须是 'round_robin' 或 'random'")
        read_duration = write_duration = 0
        if operation == "R":
            read_duration = 128
        else:
            write_duration = 128
        generator = TrafficGenerator(read_duration=read_duration, write_duration=write_duration)

        is_read = operation == "R"
        time_pattern = generator.calculate_time_points(speed, burst, is_read)
        entries = []

        # 扁平化 src_map 和 dest_map
        src_items = [(stype, pos) for stype, poses in src_map.items() for pos in poses]
        dest_items = [(dtype, pos) for dtype, poses in dest_map.items() for pos in poses]

        # 为每个 src 单独创建轮询周期，顺序随机
        dest_cycles = {}
        for src_type, src in src_items:
            shuffled = dest_items.copy()
            random.shuffle(shuffled)
            dest_cycles[(src_type, src)] = itertools.cycle(shuffled)

        # 预处理：轮询器 / 随机排列
        if dest_access_mode == "round_robin":
            if flow_type == 1:
                groups = group_numbers()
                group_cycles = {gid: itertools.cycle([item for item in dest_items if item[1] in g]) for gid, g in enumerate(groups)}
            # elif flow_type != 2:
            #     dest_cycle = itertools.cycle(dest_items)
        else:  # random
            if flow_type == 1:
                groups = group_numbers()
                group_items = {gid: [item for item in dest_items if item[1] in g] for gid, g in enumerate(groups)}
                generate_entries.group_access_idx = getattr(generate_entries, "group_access_idx", {})
            else:
                all_dest_combinations = list(product(*dest_map.values()))
                random.shuffle(all_dest_combinations)
                generate_entries.perm_idx = getattr(generate_entries, "perm_idx", 0)

        # 生成条目
        for cycle in range(interval_count):
            base_time = cycle * generator.cycle_duration

            # 根据 overlap 参数决定 time_offset
            if overlap:
                time_offset = 0
            else:
                time_offset = 0 if is_read else generator.read_duration

            # 8-shared
            if flow_type == 1:
                groups = group_numbers()
                for t in time_pattern:
                    for src_type, src in src_items:
                        gid = next((i for i, g in enumerate(groups) if src in g), -1)
                        if gid < 0:
                            continue

                        if dest_access_mode == "random":
                            idx = generate_entries.group_access_idx.get(gid, 0) % len(group_items[gid])
                            dest_type, dest = group_items[gid][idx]
                            generate_entries.group_access_idx[gid] = idx + 1
                        else:
                            dest_type, dest = next(group_cycles[gid])

                        entries.append(f"{base_time + time_offset + t},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

            # private
            elif flow_type == 2:
                for t in time_pattern:
                    for src_type, src in src_items:
                        # 优先匹配同号 dest
                        match = next((item for item in dest_items if item[1] == src), None)
                        if match:
                            dest_type, dest = match
                        else:
                            if dest_access_mode == "random":
                                perm = all_dest_combinations[generate_entries.perm_idx % len(all_dest_combinations)]
                                dest_type, dest = perm[src_items.index((src_type, src)) % len(perm)]
                                generate_entries.perm_idx += 1
                            else:
                                dest_type, dest = next(dest_cycles[(src_type, src)])

                        entries.append(f"{base_time + time_offset + t},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

            # 32-shared 或其他
            else:
                for t in time_pattern:
                    if dest_access_mode == "random":
                        generate_entries.dest_assignments = {}
                        if t not in generate_entries.dest_assignments:
                            shuffled = random.sample(dest_items, len(dest_items))
                            generate_entries.dest_assignments[t] = itertools.cycle(shuffled)
                        dest_cycle_t = generate_entries.dest_assignments[t]
                    else:
                        # dest_cycle_t = itertools.cycle(dest_items)
                        dest_cycle_t_dict = dest_cycles
                    for src_type, src in src_items:
                        if dest_access_mode == "random":
                            dest_type, dest = next(dest_cycle_t)
                        else:
                            dest_type, dest = next(dest_cycles[(src_type, src)])
                        entries.append(f"{base_time + time_offset + t},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

        return entries

    def generate_simultaneous_entries(src_pos, src_type, dest_map, burst, flow_type, read_speed, write_speed, interval_count, dest_access_mode="random"):
        """
        同时生成读和写请求

        参数:
        src_pos: list of source positions
        src_type: str, source类型
        dest_map: dict, 键为dest_type(str)，值为对应的dest_pos(list)
        burst: int, burst长度
        flow_type: int, 流量类型
        read_speed: int, 读带宽
        write_speed: int, 写带宽
        interval_count: int, 周期数
        dest_access_mode: str, 目标选择模式
        """
        # 生成读请求
        read_entries = generate_entries(src_pos, src_type, dest_map, "R", burst, flow_type, read_speed, interval_count, dest_access_mode)
        # 生成写请求
        write_entries = generate_entries(src_pos, src_type, dest_map, "W", burst, flow_type, write_speed, interval_count, dest_access_mode)

        # 合并并排序
        all_entries = read_entries + write_entries
        return sorted(all_entries, key=lambda x: int(x.split(",")[0]))

    def generate_mixed_entries(src_pos, src_type, dest_type, dest_pos, operation, burst, ratios):
        """混合模式生成（保持原有逻辑，但区分读写时间）"""
        is_read = operation == "R"
        total_speed = speed[burst]
        mode_speeds = {k: int(total_speed * v) for k, v in ratios.items()}
        dest_len = len(dest_pos)
        entries = []
        read_duration = write_duration = 0
        if operation == "R":
            read_duration = 128
        else:
            write_duration = 128
        generator = TrafficGenerator(read_duration=read_duration, write_duration=write_duration)

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
            data_all.extend(generate_mixed_entries(sdma_map, "gdma", "ddr", ddr_map, "R", burst, mix_ratios))
            data_all.extend(generate_mixed_entries(sdma_map, "gdma", "ddr", l2m_map, "W", burst, mix_ratios))
        else:
            data_all.extend(generate_entries(gdma_map, ddr_map, "W", burst, flow_type, speed[burst], interval_count, overlap=overlap))

    elif topo == "3x3":
        if flow_type == 3:
            mix_ratios = mix_ratios or {0: 0.4, 1: 0.4, 2: 0.2}
            data_all.extend(generate_mixed_entries(sdma_map, "sdma", "ddr", ddr_map, "R", burst, mix_ratios))
            data_all.extend(generate_mixed_entries(sdma_map, "sdma", "l2m", l2m_map, "W", burst, mix_ratios))
            data_all.extend(generate_mixed_entries(gdma_map, "gdma", "l2m", l2m_map, "R", burst, mix_ratios))
        else:
            # data_all.extend(generate_entries(gdma_map, l2m_map, "R", burst, flow_type, speed[burst], interval_count, overlap=overlap))
            # data_all.extend(generate_entries(sdma_map, ddr_map, "R", burst, flow_type, speed[burst], interval_count, overlap=overlap))
            # data_all.extend(generate_entries(sdma_map, l2m_map, "W", burst, flow_type, speed[burst], interval_count, overlap=overlap))
            #
            # data_all.extend(generate_entries(gdma_map, l2m_map, "W", burst, flow_type, speed[burst], interval_count, overlap=overlap))
            # data_all.extend(generate_entries(sdma_map, ddr_map, "W", burst, flow_type, speed[burst], interval_count, overlap=overlap))
            # data_all.extend(generate_entries(sdma_map, l2m_map, "R", burst, flow_type, speed[burst], interval_count, overlap=overlap))

            # data_all.extend(generate_entries(gdma_map, ddr_map, "R", burst, flow_type, speed[burst], interval_count, overlap=overlap))
            data_all.extend(generate_entries(gdma_map, l2m_map, "R", burst, flow_type, speed[burst], interval_count, overlap=overlap))

    # 排序并写入文件
    with open(file_name, "w") as f:
        f.writelines(sorted(data_all, key=lambda x: int(x.split(",")[0])))


# 示例使用
if __name__ == "__main__":
    # 参数配置
    TOPO = "3x3"
    # TOPO = "5x4"
    INTERVAL_COUNT = 128
    # FILE_NAME = "../../test_data/traffic_2262_case1.txt"
    FILE_NAME = "../../test_data/traffic_2260E_case2.txt"
    # FILE_NAME = "../../test_data/traffic_2262_case1.txt"
    np.random.seed(520)

    if TOPO == "5x4":
        BURST = 4
        NUM_IP = 16
        SDMA_MAP = {
            "sdma_0": range(NUM_IP),
            "sdma_1": range(NUM_IP),
        }
        GDMA_MAP = {
            "gdma_0": range(NUM_IP),
            "gdma_1": range(NUM_IP),
        }
        DDR_MAP = {
            "ddr_0": range(NUM_IP),
            "ddr_1": range(NUM_IP),
        }
        L2M_MAP = {"l2m_0": range(NUM_IP)}

    # SG2260E
    elif TOPO == "3x3":
        BURST = 2
        SDMA_MAP = {
            "sdma_0": [
                0,
                2,
                6,
                8,
            ],
        }
        GDMA_MAP = {
            "gdma_0": [
                0,
                2,
                6,
                8,
            ],
        }
        DDR_MAP = {
            "ddr_0": [0, 2, 3, 5, 6, 8],
            "ddr_1": [0, 2, 3, 5, 6, 8],
            "ddr_2": [3, 5],
            "ddr_3": [3, 5],
            # "ddr_0": [3],
            # "ddr_1": [3],
            # "ddr_2": [3],
            # "ddr_3": [3],
        }
        L2M_MAP = {
            # "l2m_0": [3],
            # "l2m_1": [4],
            "l2m_0": [1, 7],
            "l2m_1": [1, 7],
        }

    SPEED = {1: 128, 2: 128, 4: 128}  # 不同burst对应的带宽(GB/s)
    # read_duration = 0
    # write_duration = 128
    overlap = 1

    # 生成数据(使用混合模式)
    generate_data(TOPO, INTERVAL_COUNT, FILE_NAME, SDMA_MAP, GDMA_MAP, DDR_MAP, L2M_MAP, SPEED, BURST, flow_type=0, overlap=overlap)
