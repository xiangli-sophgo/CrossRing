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
        # 计算请求数 - 使用浮点数计算然后四舍五入
        total_transfers_float = speed * window_duration / (self.total_bandwidth * burst)
        total_transfers = max(1, round(total_transfers_float))  # 至少1次传输，四舍五入

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


def generate_data(topo, end_time, file_name, sdma_map, gdma_map, cdma_map, ddr_map, l2m_map, speed, burst, flow_type=0, mix_ratios=None, overlap=0, custom_mapping=None, req_type="R"):
    """
    :param custom_mapping: 自定义映射配置，格式如下：
    {
        'read': {
            ('cdma', 12): [('ddr', 0), ('ddr', 1), ('ddr', 2), ('ddr', 3), ('ddr', 4), ('ddr', 5), ('ddr', 6), ('ddr', 7)],
            ('cdma', 13): [('ddr', 8), ('ddr', 9), ('ddr', 10), ('ddr', 11), ('ddr', 12), ('ddr', 13), ('ddr', 14), ('ddr', 15)],
            ('gdma', 0): [('l2m', 0), ('l2m', 1)]
        },
        'write': {
            ('cdma', 12): [('l2m', 0), ('l2m', 1)],
            ('cdma', 13): [('l2m', 2), ('l2m', 3)]
        }
    }
    """
    # 初始化流量生成器(默认读128ns+写128ns)
    data_all = []

    def generate_entries(src_map, dest_map, operation, burst, flow_type, speed, end_time, dest_access_mode="random", overlap=True, src_type=None):
        """
        生成指定模式的流量条目，确保同一时刻所有dest都有访问

        参数:
        src_map: dict, 键为src_type(str)，值为对应的src_pos(list)
        dest_map: dict, 键为dest_type(str)，值为对应的dest_pos(list)
        operation: 'R' 或 'W'
        burst: int, burst长度
        flow_type: int, 1=8-shared, 2=private, 4=custom_mapping, 其他=32-shared
        speed: dict, 支持IP类型特定带宽，格式: {burst: {ip_type: bandwidth}} 或 {burst: bandwidth}
        end_time: int, 结束时间(ns)
        dest_access_mode: 'round_robin' 或 'random'
        overlap: bool, 是否让读写请求时间点重叠（True 时 time_offset 恒为 0）
        src_type: str, 源IP类型（用于获取对应的带宽）
        """
        if dest_access_mode not in ("round_robin", "random"):
            raise ValueError("dest_access_mode 必须是 'round_robin' 或 'random'")
        read_duration = write_duration = 0
        if operation == "R":
            read_duration = 1280  # 增加到1280ns提高精确度
        else:
            write_duration = 1280  # 增加到1280ns提高精确度
        generator = TrafficGenerator(read_duration=read_duration, write_duration=write_duration)

        is_read = operation == "R"

        # 获取适用的带宽值
        effective_speed = speed
        if isinstance(speed, dict) and burst in speed:
            burst_speed = speed[burst]
            if isinstance(burst_speed, dict) and src_type and src_type in burst_speed:
                effective_speed = burst_speed[src_type]
            elif isinstance(burst_speed, dict) and "default" in burst_speed:
                effective_speed = burst_speed["default"]
            elif not isinstance(burst_speed, dict):
                effective_speed = burst_speed

        time_pattern = generator.calculate_time_points(effective_speed, burst, is_read)
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
        else:  # random
            if flow_type == 1:
                groups = group_numbers()
                group_items = {gid: [item for item in dest_items if item[1] in g] for gid, g in enumerate(groups)}
                generate_entries.group_access_idx = getattr(generate_entries, "group_access_idx", {})
            else:
                all_dest_combinations = list(product(*dest_map.values()))
                random.shuffle(all_dest_combinations)
                generate_entries.perm_idx = getattr(generate_entries, "perm_idx", 0)

        # 生成条目 - 改为使用结束时间限制
        cycle = 0
        while True:
            base_time = cycle * generator.cycle_duration

            # 根据 overlap 参数决定 time_offset
            if overlap:
                time_offset = 0
            else:
                time_offset = 0 if is_read else generator.read_duration

            # 检查是否超过结束时间 - 检查这个周期内所有可能的时间点
            max_time_in_cycle = base_time + time_offset + max(time_pattern) if time_pattern else base_time + time_offset
            if base_time + time_offset >= end_time:
                break

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

                        timestamp = base_time + time_offset + t
                        if timestamp < end_time:
                            entries.append(f"{timestamp},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

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

                        timestamp = base_time + time_offset + t
                        if timestamp < end_time:
                            entries.append(f"{timestamp},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

            # custom_mapping
            elif flow_type == 4:
                if custom_mapping and operation in custom_mapping:
                    op_key = operation
                    mapping = custom_mapping.get(op_key, {})

                    for t in time_pattern:
                        for src_type, src in src_items:
                            # 查找该源DMA的自定义目标列表
                            src_key = (src_type, src)
                            if src_key in mapping:
                                target_dests = mapping[src_key]
                                # 轮询访问目标列表中的所有目标
                                if not hasattr(generate_entries, "custom_cycles"):
                                    generate_entries.custom_cycles = {}
                                if src_key not in generate_entries.custom_cycles:
                                    generate_entries.custom_cycles[src_key] = itertools.cycle(target_dests)

                                dest_type, dest = next(generate_entries.custom_cycles[src_key])
                                timestamp = base_time + time_offset + t
                                if timestamp < end_time:
                                    entries.append(f"{timestamp},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")
                            else:
                                # 如果没有自定义映射，回退到默认行为
                                if dest_access_mode == "random":
                                    dest_type, dest = random.choice(dest_items)
                                else:
                                    dest_type, dest = next(dest_cycles[(src_type, src)])
                                timestamp = base_time + time_offset + t
                                if timestamp < end_time:
                                    entries.append(f"{timestamp},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")
                else:
                    # 没有自定义映射时，回退到默认32-shared行为
                    for t in time_pattern:
                        for src_type, src in src_items:
                            if dest_access_mode == "random":
                                dest_type, dest = random.choice(dest_items)
                            else:
                                dest_type, dest = next(dest_cycles[(src_type, src)])
                            timestamp = base_time + time_offset + t
                            if timestamp < end_time:
                                entries.append(f"{timestamp},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

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
                        dest_cycle_t_dict = dest_cycles
                    for src_type, src in src_items:
                        if dest_access_mode == "random":
                            dest_type, dest = next(dest_cycle_t)
                        else:
                            dest_type, dest = next(dest_cycles[(src_type, src)])
                        timestamp = base_time + time_offset + t
                        if timestamp < end_time:
                            entries.append(f"{timestamp},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

            cycle += 1

        return entries

    def generate_mixed_entries(src_pos, src_type, dest_type, dest_pos, operation, burst, ratios, end_time):
        """混合模式生成（保持原有逻辑，但区分读写时间）"""
        is_read = operation == "R"

        # 获取适用的带宽值
        effective_speed = speed[burst]
        if isinstance(speed[burst], dict) and src_type in speed[burst]:
            effective_speed = speed[burst][src_type]
        elif isinstance(speed[burst], dict) and "default" in speed[burst]:
            effective_speed = speed[burst]["default"]

        total_speed = effective_speed
        mode_speeds = {k: int(total_speed * v) for k, v in ratios.items()}
        dest_len = len(dest_pos)
        entries = []
        read_duration = write_duration = 0
        if operation == "R":
            read_duration = 1280  # 增加到1280ns提高精确度
        else:
            write_duration = 1280  # 增加到1280ns提高精确度
        generator = TrafficGenerator(read_duration=read_duration, write_duration=write_duration)

        cycle = 0
        while True:
            base_time = cycle * generator.cycle_duration
            time_offset = 0 if is_read else generator.read_duration

            # 检查是否超过结束时间
            if base_time + time_offset >= end_time:
                break

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

                        timestamp = base_time + time_offset + t
                        if timestamp < end_time:
                            entries.append(f"{timestamp},{src},{src_type}," f"{dest},{dest_type},{operation},{burst}\n")

            cycle += 1

        return entries

    if topo in ["4x4", "4x2"]:
        data_all.extend(generate_entries(gdma_map, ddr_map, req_type, burst, flow_type, speed, end_time, overlap=overlap, src_type="gdma"))
        # 添加CDMA相关的基础流量
        if cdma_map:
            data_all.extend(generate_entries(cdma_map, ddr_map, "R", burst, flow_type, speed, end_time, overlap=overlap, src_type="cdma"))
            data_all.extend(generate_entries(cdma_map, ddr_map, "W", burst, flow_type, speed, end_time, overlap=overlap, src_type="cdma"))
        if sdma_map:
            data_all.extend(generate_entries(sdma_map, ddr_map, "R", burst, flow_type, speed, end_time, overlap=overlap, src_type="sdma"))
            data_all.extend(generate_entries(sdma_map, ddr_map, "W", burst, flow_type, speed, end_time, overlap=overlap, src_type="sdma"))

    # 生成数据逻辑 - 增加CDMA相关的分支
    if topo in ["4x9", "9x4", "4x5", "5x4"]:
        if flow_type == 3:
            mix_ratios = mix_ratios or {0: 0.4, 1: 0.4, 2: 0.2}
            data_all.extend(generate_mixed_entries(sdma_map, "sdma", "ddr", ddr_map, "R", burst, mix_ratios))
            data_all.extend(generate_mixed_entries(sdma_map, "sdma", "l2m", l2m_map, "W", burst, mix_ratios))
            # 添加CDMA相关的混合流量
            if cdma_map:
                data_all.extend(generate_mixed_entries(cdma_map, "cdma", "ddr", ddr_map, "R", burst, mix_ratios))
                data_all.extend(generate_mixed_entries(cdma_map, "cdma", "l2m", l2m_map, "W", burst, mix_ratios))
        elif flow_type == 4:
            # 使用自定义映射
            if custom_mapping:
                # 处理读操作
                if "R" in custom_mapping:
                    data_all.extend(generate_entries(cdma_map, ddr_map, "R", burst, flow_type, speed, end_time, overlap=overlap, src_type="cdma"))

                # 处理写操作
                if "W" in custom_mapping:
                    data_all.extend(generate_entries(cdma_map, ddr_map, "W", burst, flow_type, speed, end_time, overlap=overlap, src_type="cdma"))
            else:
                # 没有自定义映射时，使用默认行为
                data_all.extend(generate_entries(gdma_map, ddr_map, "W", burst, flow_type, speed, end_time, overlap=overlap, src_type="gdma"))
                if cdma_map:
                    data_all.extend(generate_entries(cdma_map, ddr_map, "R", burst, flow_type, speed, end_time, overlap=overlap, src_type="cdma"))
        else:
            data_all.extend(generate_entries(gdma_map, ddr_map, "W", burst, flow_type, speed, end_time, overlap=overlap, src_type="gdma"))
            # 添加CDMA相关的基础流量
            if cdma_map:
                data_all.extend(generate_entries(cdma_map, ddr_map, "R", burst, flow_type, speed, end_time, overlap=overlap, src_type="cdma"))

    elif topo == "3x3":
        if flow_type == 5:
            mix_ratios = mix_ratios or {0: 0.4, 1: 0.4, 2: 0.2}
            data_all.extend(generate_mixed_entries(sdma_map, "sdma", "ddr", ddr_map, "R", burst, mix_ratios))
            data_all.extend(generate_mixed_entries(sdma_map, "sdma", "l2m", l2m_map, "W", burst, mix_ratios))
            data_all.extend(generate_mixed_entries(gdma_map, "gdma", "l2m", l2m_map, "R", burst, mix_ratios))
            # 添加CDMA相关的混合流量
            if cdma_map:
                data_all.extend(generate_mixed_entries(cdma_map, "cdma", "ddr", ddr_map, "R", burst, mix_ratios))
                data_all.extend(generate_mixed_entries(cdma_map, "cdma", "l2m", l2m_map, "W", burst, mix_ratios))
        elif flow_type == 4:
            # 使用自定义映射
            if custom_mapping:
                # 处理读操作
                if "R" in custom_mapping:
                    # data_all.extend(generate_entries(cdma_map, ddr_map, "R", burst, flow_type, speed[burst], end_time, overlap=overlap))
                    data_all.extend(generate_entries(gdma_map, ddr_map, "R", burst, flow_type, speed, end_time, overlap=overlap, src_type="gdma"))

                # 处理写操作
                if "W" in custom_mapping:
                    data_all.extend(generate_entries(cdma_map, l2m_map, "W", burst, flow_type, speed, end_time, overlap=overlap, src_type="cdma"))
        else:
            # 保持原有的逻辑
            # data_all.extend(generate_entries(gdma_map, l2m_map, "R", burst, flow_type, speed[burst], end_time, overlap=overlap))
            data_all.extend(generate_entries(gdma_map, ddr_map, "R", burst, flow_type, speed, end_time, overlap=overlap, src_type="gdma"))
            # data_all.extend(generate_entries(gdma_map, ddr_map, "W", burst, flow_type, speed[burst], end_time, overlap=overlap))
            # 添加CDMA相关的基础流量
            if cdma_map:
                data_all.extend(generate_entries(cdma_map, ddr_map, "R", burst, flow_type, speed, end_time, overlap=overlap, src_type="cdma"))
                data_all.extend(generate_entries(cdma_map, ddr_map, "W", burst, flow_type, speed, end_time, overlap=overlap, src_type="cdma"))
            # if sdma_map:
            #     data_all.extend(generate_entries(sdma_map, ddr_map, "R", burst, flow_type, speed, end_time, overlap=overlap, src_type="sdma"))
            #     data_all.extend(generate_entries(sdma_map, ddr_map, "W", burst, flow_type, speed, end_time, overlap=overlap, src_type="sdma"))

    # 过滤超出结束时间的数据并排序写入文件
    filtered_data = [line for line in data_all if int(line.split(",")[0]) < end_time]
    with open(file_name, "w") as f:
        f.writelines(sorted(filtered_data, key=lambda x: int(x.split(",")[0])))


# 示例使用
if __name__ == "__main__":
    # 参数配置
    END_TIME = 6300  # 结束时间50000ns
    TOPO = "4x2"
    TOPO = "4x4"
    REQ_TYEP = "R"
    REQ_TYEP = "W"
    C2C_TYPE = "wo"
    # C2C_TYPE = "w"
    SPARE_CORE = "wo"
    # SPARE_CORE = "w"
    FILE_NAME = f"../../test_data/2262_{TOPO}_{C2C_TYPE}c2c_{SPARE_CORE}SPC_{REQ_TYEP}.txt"
    np.random.seed(715)

    if TOPO == "5x4":
        BURST = 4
        NUM_IP = 16
        SDMA_MAP = {}
        GDMA_MAP = {}

        CDMA_MAP = {
            "cdma_0": [
                16,
                # 17,
                # 18,
                # 19,
            ],
        }

        DDR_MAP = {
            # "ddr_0": range(NUM_IP),
            # "ddr_1": range(NUM_IP),
        }
        L2M_MAP = {}

        # 自定义映射配置：每个CDMA发送到特定的8个DDR
        CUSTOM_MAPPING = {
            "R": {
                # ("cdma_0", 16): [
                #     ("ddr_0", 0),
                #     ("ddr_1", 0),
                #     ("ddr_0", 4),
                #     ("ddr_1", 4),
                #     ("ddr_0", 8),
                #     ("ddr_1", 8),
                #     ("ddr_0", 12),
                #     ("ddr_1", 12),
                # ],
                # ("cdma_0", 17): [
                #     ("ddr_0", 1),
                #     ("ddr_1", 1),
                #     ("ddr_0", 5),
                #     ("ddr_1", 5),
                #     ("ddr_0", 9),
                #     ("ddr_1", 9),
                #     ("ddr_0", 13),
                #     ("ddr_1", 13),
                # ],
                # ("cdma_0", 18): [
                #     ("ddr_0", 2),
                #     ("ddr_1", 2),
                #     ("ddr_0", 6),
                #     ("ddr_1", 6),
                #     ("ddr_0", 10),
                #     ("ddr_1", 10),
                #     ("ddr_0", 14),
                #     ("ddr_1", 14),
                # ],
                # ("cdma_0", 19): [
                #     ("ddr_0", 3),
                #     ("ddr_1", 3),
                #     ("ddr_0", 7),
                #     ("ddr_1", 7),
                #     ("ddr_0", 11),
                #     ("ddr_1", 11),
                #     ("ddr_0", 15),
                #     ("ddr_1", 15),
                # ],
            }
        }

    elif TOPO == "3x3":
        BURST = 4
        SDMA_MAP = {
            # "sdma_0": [
            #     # 0,
            #     # 2,
            #     # 6,
            #     # 8,
            # ],
        }
        GDMA_MAP = {
            "gdma_0": [
                3,
                # 2,
                # 6,
                # 8,
            ],
            # "gdma_1": list(range(10)),
        }
        CDMA_MAP = {
            # "cdma_0": [0, 2, 6, 8],
        }
        DDR_MAP = {
            "ddr_0": [
                4,
                # 2,
                # 3,
                # 1,
                # 6,
                # 8,
            ],
            # "ddr_1": list(range(10)),
            # "ddr_1": [0, 2, 3, 5, 6, 8],
            # "ddr_2": [3, 5],
            # "ddr_3": [3, 5],
        }
        L2M_MAP = {
            # "l2m_0": [1, 7],
            # "l2m_1": [1, 7],
        }

        CUSTOM_MAPPING = {}

    elif TOPO == "4x4":
        BURST = 4
        NUM_IP = 16
        SDMA_MAP = {
            "sdma_0": [
                15,
            ],
        }
        GDMA_MAP = {
            # "gdma_0": list(range((16))),
            # 冗余core
            "gdma_0": [
                0,
                1,
                2,
                # 3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
            ],
            "gdma_1": [13],
        }
        if SPARE_CORE == "wo":
            GDMA_MAP = {
                "gdma_0": list(range((16))),
            }
        CDMA_MAP = {
            "cdma_0": [
                14,
                15,
            ],
            "cdma_1": [
                14,
            ],
            "cdma_2": [
                14,
            ],
            "cdma_3": [
                14,
            ],
        }

        DDR_MAP = {
            "ddr_0": [
                0,
                3,
                4,
                7,
                8,
                11,
                12,
                15,
            ],
            "ddr_1": [
                0,
                3,
                4,
                7,
                8,
                11,
                12,
                15,
            ],
        }
        L2M_MAP = {}

        # 自定义映射配置：每个CDMA发送到特定的8个DDR
        CUSTOM_MAPPING = {
            "R": {
                # ("cdma_0", 16): [
                #     ("ddr_0", 0),
                #     ("ddr_1", 0),
                #     ("ddr_0", 4),
                #     ("ddr_1", 4),
                #     ("ddr_0", 8),
                #     ("ddr_1", 8),
                #     ("ddr_0", 12),
                #     ("ddr_1", 12),
                # ],
                # ("cdma_0", 17): [
                #     ("ddr_0", 1),
                #     ("ddr_1", 1),
                #     ("ddr_0", 5),
                #     ("ddr_1", 5),
                #     ("ddr_0", 9),
                #     ("ddr_1", 9),
                #     ("ddr_0", 13),
                #     ("ddr_1", 13),
                # ],
                # ("cdma_0", 18): [
                #     ("ddr_0", 2),
                #     ("ddr_1", 2),
                #     ("ddr_0", 6),
                #     ("ddr_1", 6),
                #     ("ddr_0", 10),
                #     ("ddr_1", 10),
                #     ("ddr_0", 14),
                #     ("ddr_1", 14),
                # ],
                # ("cdma_0", 19): [
                #     ("ddr_0", 3),
                #     ("ddr_1", 3),
                #     ("ddr_0", 7),
                #     ("ddr_1", 7),
                #     ("ddr_0", 11),
                #     ("ddr_1", 11),
                #     ("ddr_0", 15),
                #     ("ddr_1", 15),
                # ],
            }
        }

    elif TOPO == "4x2":
        BURST = 4
        NUM_IP = 16
        SDMA_MAP = {
            "sdma_0": [
                7,
            ],
        }
        GDMA_MAP = {
            "gdma_0": list(range((8))),
            # "gdma_1": list(range((8))),
            "gdma_1": [0, 2, 3, 4, 5, 6, 7],
            "gdma_2": [6],
        }
        if SPARE_CORE == "wo":
            GDMA_MAP = {
                "gdma_0": list(range((8))),
                "gdma_1": list(range((8))),
            }

        CDMA_MAP = {
            "cdma_0": [
                6,
                7,
            ],
            "cdma_1": [
                6,
            ],
            "cdma_2": [
                6,
            ],
            "cdma_3": [
                6,
            ],
        }

        DDR_MAP = {
            "ddr_0": list(range((8))),
            "ddr_1": list(range((8))),
        }
        L2M_MAP = {}
        CUSTOM_MAPPING = {}

    # 不同IP类型的带宽配置 (GB/s)
    SPEED = {
        1: {"gdma": 128, "sdma": 64, "cdma": 256},  # burst=1时各IP的带宽
        2: {"gdma": 128, "sdma": 64, "cdma": 256},  # burst=2时各IP的带宽
        4: {"gdma": 105.3, "sdma": 32, "cdma": 25.2},  # burst=4时各IP的带宽
        # 4: {"gdma": 115.2, "sdma": 0, "cdma": 0},  # burst=4时各IP的带宽
    }
    if C2C_TYPE == "wo":
        SPEED[4] = {"gdma": 115.2, "sdma": 0, "cdma": 0}
    overlap = 1

    # 生成数据，使用flow_type=4启用自定义映射
    generate_data(TOPO, END_TIME, FILE_NAME, SDMA_MAP, GDMA_MAP, CDMA_MAP, DDR_MAP, L2M_MAP, SPEED, BURST, flow_type=3, overlap=overlap, custom_mapping=CUSTOM_MAPPING)

    print(f"Traffic data generated successfully! {FILE_NAME}")


def batch_generate_all_combinations():
    """批量生成所有参数组合的数据流"""
    import itertools

    # 定义所有参数组合
    TOPO_OPTIONS = ["4x2", "4x4"]
    REQ_TYPE_OPTIONS = ["R", "W"]
    C2C_TYPE_OPTIONS = ["wo", "w"]
    SPARE_CORE_OPTIONS = ["wo", "w"]

    END_TIME = 6300
    np.random.seed(715)

    # 遍历所有组合
    for topo, req_type, c2c_type, spare_core in itertools.product(TOPO_OPTIONS, REQ_TYPE_OPTIONS, C2C_TYPE_OPTIONS, SPARE_CORE_OPTIONS):
        print(f"生成配置: TOPO={topo}, REQ_TYPE={req_type}, C2C_TYPE={c2c_type}, SPARE_CORE={spare_core}")

        # 生成文件名
        file_name = f"../../test_data/2262_{topo}_{c2c_type}c2c_{spare_core}SPC_{req_type}.txt"

        # 根据拓扑配置参数
        if topo == "4x4":
            BURST = 4
            NUM_IP = 16

            if spare_core == "wo":
                GDMA_MAP = {
                    "gdma_0": list(range(16)),
                }
            else:  # spare_core == "w"
                GDMA_MAP = {
                    "gdma_0": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    "gdma_1": [13],
                }
            if c2c_type == "w":
                SDMA_MAP = {
                    "sdma_0": [15],
                }
                CDMA_MAP = {
                    "cdma_0": [14, 15],
                    "cdma_1": [14],
                    "cdma_2": [14],
                    "cdma_3": [14],
                }
            else:
                CDMA_MAP = {}
                SDMA_MAP = {}

            DDR_MAP = {
                "ddr_0": [0, 3, 4, 7, 8, 11, 12, 15],
                "ddr_1": [0, 3, 4, 7, 8, 11, 12, 15],
            }
            L2M_MAP = {}

        elif topo == "4x2":
            BURST = 4
            NUM_IP = 8

            if spare_core == "wo":
                GDMA_MAP = {
                    "gdma_0": list(range(8)),
                    "gdma_1": list(range(8)),
                }
            else:  # spare_core == "w"
                GDMA_MAP = {
                    "gdma_0": list(range(8)),
                    "gdma_1": [0, 2, 3, 4, 5, 6, 7],
                    "gdma_2": [6],
                }
            if c2c_type == "w":
                SDMA_MAP = {
                    "sdma_0": [7],
                }
                CDMA_MAP = {
                    "cdma_0": [6, 7],
                    "cdma_1": [6],
                    "cdma_2": [6],
                    "cdma_3": [6],
                }
            else:
                CDMA_MAP = {}
                SDMA_MAP = {}

            DDR_MAP = {
                "ddr_0": list(range(8)),
                "ddr_1": list(range(8)),
            }
            L2M_MAP = {}

        # 配置速度
        SPEED = {
            1: {"gdma": 128, "sdma": 64, "cdma": 256},
            2: {"gdma": 128, "sdma": 64, "cdma": 256},
            4: {"gdma": 105.3, "sdma": 32, "cdma": 25.2},
        }

        if c2c_type == "wo":
            SPEED[4] = {"gdma": 115.2, "sdma": 0, "cdma": 0}

        CUSTOM_MAPPING = {}
        overlap = 1

        # 生成数据
        try:
            generate_data(topo, END_TIME, file_name, SDMA_MAP, GDMA_MAP, CDMA_MAP, DDR_MAP, L2M_MAP, SPEED, BURST, flow_type=3, overlap=overlap, custom_mapping=CUSTOM_MAPPING, req_type=req_type)
            print(f"✓ 成功生成: {file_name}")
        except Exception as e:
            print(f"✗ 生成失败: {file_name}, 错误: {e}")

    print("批量生成完成！")


# 如果要批量生成所有组合，取消下面的注释
batch_generate_all_combinations()
