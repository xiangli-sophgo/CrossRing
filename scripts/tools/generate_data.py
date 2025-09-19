import random
import numpy as np
import itertools
from itertools import product
from abc import ABC, abstractmethod


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


def _create_traffic_generator(operation):
    """创建流量生成器的工具函数"""
    read_duration = 1280 if operation == "R" else 0
    write_duration = 1280 if operation == "W" else 0
    return TrafficGenerator(read_duration=read_duration, write_duration=write_duration)


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


class TrafficStrategy(ABC):
    """流量生成策略抽象基类"""

    @abstractmethod
    def generate(self, ip_maps, speed, burst, end_time, **kwargs):
        """生成流量数据

        参数:
        - ip_maps: IP映射字典 {'gdma': {...}, 'ddr': {...}, ...}
        - speed: 带宽配置
        - burst: burst长度
        - end_time: 结束时间
        - kwargs: 其他参数（overlap, req_type等）
        """
        pass

    def _flatten_ip_items(self, ip_map):
        """扁平化IP映射为(类型, 位置)列表"""
        return [(ip_type, pos) for ip_type, poses in ip_map.items() for pos in poses]

    def _get_effective_speed(self, speed, burst, src_type):
        """获取有效带宽值"""
        effective_speed = speed
        if isinstance(speed, dict) and burst in speed:
            burst_speed = speed[burst]
            if isinstance(burst_speed, dict) and src_type and src_type in burst_speed:
                effective_speed = burst_speed[src_type]
            elif isinstance(burst_speed, dict) and "default" in burst_speed:
                effective_speed = burst_speed["default"]
            elif not isinstance(burst_speed, dict):
                effective_speed = burst_speed
        return effective_speed

    def _calculate_time_offset(self, overlap, is_read, generator):
        """计算时间偏移的通用方法"""
        if overlap:
            return 0
        else:
            return 0 if is_read else generator.read_duration

    def _get_source_types(self, src_type=None):
        """获取源类型列表的通用方法"""
        if src_type:
            return [src_type] if isinstance(src_type, str) else src_type
        return ["gdma", "sdma", "cdma"]

    def _get_operations_for_source(self, src_type, req_type=None):
        """根据源类型获取操作列表"""
        if src_type == "gdma":
            return [req_type] if req_type else ["R", "W"]
        else:
            return ["R", "W"]


class TrafficDataGenerator:
    """统一的流量数据生成器（不依赖拓扑）"""

    def __init__(self):
        """初始化生成器和各种策略"""
        self.strategies = {
            0: SharedTrafficStrategy(32),  # 32-shared
            1: SharedTrafficStrategy(8),  # 8-shared
            2: PrivateTrafficStrategy(),  # private
            3: MixedTrafficStrategy(),  # mixed
            4: CustomMappingStrategy(),  # custom mapping
            5: MixedTrafficStrategy(),  # mixed variant
        }

    def generate(self, end_time, file_name, ip_maps, speed, burst, flow_type=0, **kwargs):
        """主生成方法

        参数:
        - end_time: 结束时间
        - file_name: 输出文件名
        - ip_maps: IP映射字典 {'gdma': {...}, 'ddr': {...}, 'cdma': {...}, 'l2m': {...}, 'sdma': {...}}
        - speed: 带宽配置
        - burst: burst长度
        - flow_type: 流量类型
        - kwargs: 其他参数（mix_ratios, overlap, custom_mapping, req_type等）
        """
        # 根据flow_type选择策略
        strategy = self.strategies.get(flow_type, self.strategies[0])

        # 生成数据
        data = strategy.generate(ip_maps, speed, burst, end_time, **kwargs)

        # 保存到文件
        self._save_to_file(data, file_name, end_time)

    def _save_to_file(self, data, file_name, end_time):
        """保存数据到文件"""
        filtered_data = [line for line in data if int(line.split(",")[0]) < end_time]
        with open(file_name, "w") as f:
            f.writelines(sorted(filtered_data, key=lambda x: int(x.split(",")[0])))


class SharedTrafficStrategy(TrafficStrategy):
    """共享模式流量策略（支持8-shared和32-shared）"""

    def __init__(self, share_size):
        self.share_size = share_size

    def generate(self, ip_maps, speed, burst, end_time, **kwargs):
        """生成共享模式的流量数据"""
        data_all = []
        overlap = kwargs.get("overlap", 0)
        req_type = kwargs.get("req_type", "R")

        # 处理各种IP组合
        for src_type in self._get_source_types():
            if src_type in ip_maps and ip_maps[src_type]:
                ops = self._get_operations_for_source(src_type, req_type)

                for op in ops:
                    dest_type = "ddr"
                    if dest_type in ip_maps and ip_maps[dest_type]:
                        entries = self._generate_shared_entries(ip_maps[src_type], ip_maps[dest_type], op, burst, speed, end_time, overlap, src_type)
                        data_all.extend(entries)

        return data_all

    def _generate_shared_entries(self, src_map, dest_map, operation, burst, speed, end_time, overlap, src_type):
        """生成共享模式的具体条目"""
        generator = _create_traffic_generator(operation)
        is_read = operation == "R"

        # 获取有效带宽
        effective_speed = self._get_effective_speed(speed, burst, src_type)

        time_pattern = generator.calculate_time_points(effective_speed, burst, is_read)
        entries = []

        # 扁平化源和目标
        src_items = self._flatten_ip_items(src_map)
        dest_items = self._flatten_ip_items(dest_map)

        # 为8-shared使用分组逻辑
        if self.share_size == 8:
            groups = group_numbers()
            # 为每个组创建目标列表
            group_items = {gid: [item for item in dest_items if item[1] in g] for gid, g in enumerate(groups)}

        cycle = 0
        while True:
            base_time = cycle * generator.cycle_duration

            # 计算时间偏移
            time_offset = self._calculate_time_offset(overlap, is_read, generator)

            # 检查是否超过结束时间
            if base_time + time_offset >= end_time:
                break

            for t in time_pattern:
                if self.share_size == 8:  # 8-shared
                    for src_type, src in src_items:
                        # 确定源所属的组
                        gid = next((i for i, g in enumerate(groups) if src in g), -1)
                        if gid < 0:
                            continue

                        # 从对应组中随机选择目标
                        if group_items[gid]:
                            dest_type, dest = random.choice(group_items[gid])
                            timestamp = base_time + time_offset + t
                            if timestamp < end_time:
                                entries.append(f"{timestamp},{src},{src_type},{dest},{dest_type},{operation},{burst}\n")

                else:  # 32-shared
                    for src_type, src in src_items:
                        # 随机选择目标
                        dest_type, dest = random.choice(dest_items)
                        timestamp = base_time + time_offset + t
                        if timestamp < end_time:
                            entries.append(f"{timestamp},{src},{src_type},{dest},{dest_type},{operation},{burst}\n")

            cycle += 1

        return entries


class PrivateTrafficStrategy(TrafficStrategy):
    """私有模式流量策略"""

    def generate(self, ip_maps, speed, burst, end_time, **kwargs):
        """生成私有模式的流量数据"""
        data_all = []
        overlap = kwargs.get("overlap", 0)
        req_type = kwargs.get("req_type", "R")

        # 处理各种IP组合
        for src_type in self._get_source_types():
            if src_type in ip_maps and ip_maps[src_type]:
                ops = self._get_operations_for_source(src_type, req_type)

                for op in ops:
                    dest_type = "ddr"
                    if dest_type in ip_maps and ip_maps[dest_type]:
                        entries = self._generate_private_entries(ip_maps[src_type], ip_maps[dest_type], op, burst, speed, end_time, overlap, src_type)
                        data_all.extend(entries)

        return data_all

    def _generate_private_entries(self, src_map, dest_map, operation, burst, speed, end_time, overlap, src_type):
        """生成私有模式的具体条目"""
        generator = _create_traffic_generator(operation)
        is_read = operation == "R"

        # 获取有效带宽
        effective_speed = self._get_effective_speed(speed, burst, src_type)

        time_pattern = generator.calculate_time_points(effective_speed, burst, is_read)
        entries = []

        # 扁平化源和目标
        src_items = self._flatten_ip_items(src_map)
        dest_items = self._flatten_ip_items(dest_map)

        cycle = 0
        while True:
            base_time = cycle * generator.cycle_duration

            # 计算时间偏移
            time_offset = self._calculate_time_offset(overlap, is_read, generator)

            # 检查是否超过结束时间
            if base_time + time_offset >= end_time:
                break

            for t in time_pattern:
                for src_type, src in src_items:
                    # 私有模式：优先匹配同号目标
                    match = next((item for item in dest_items if item[1] == src), None)
                    if match:
                        dest_type, dest = match
                    else:
                        # 如果没有匹配的，使用模运算
                        dest_type, dest = dest_items[src % len(dest_items)]

                    timestamp = base_time + time_offset + t
                    if timestamp < end_time:
                        entries.append(f"{timestamp},{src},{src_type},{dest},{dest_type},{operation},{burst}\n")

            cycle += 1

        return entries


class MixedTrafficStrategy(TrafficStrategy):
    """混合模式流量策略"""

    def generate(self, ip_maps, speed, burst, end_time, **kwargs):
        """生成混合模式的流量数据"""
        mix_ratios = kwargs.get("mix_ratios", {0: 0.4, 1: 0.4, 2: 0.2})
        overlap = kwargs.get("overlap", 0)

        data_all = []

        # 处理各种IP组合的混合流量
        for src_type in self._get_source_types():
            if src_type in ip_maps and ip_maps[src_type]:
                # SDMA和CDMA: 读DDR + 写L2M
                if src_type in ["sdma", "cdma"]:
                    if "ddr" in ip_maps:
                        entries = self._generate_mixed_entries(ip_maps[src_type], src_type, "ddr", ip_maps["ddr"], "R", burst, mix_ratios, speed, end_time)
                        data_all.extend(entries)

                    if "l2m" in ip_maps:
                        entries = self._generate_mixed_entries(ip_maps[src_type], src_type, "l2m", ip_maps["l2m"], "W", burst, mix_ratios, speed, end_time)
                        data_all.extend(entries)

                # GDMA: 读L2M
                elif src_type == "gdma":
                    if "l2m" in ip_maps:
                        entries = self._generate_mixed_entries(ip_maps[src_type], src_type, "l2m", ip_maps["l2m"], "R", burst, mix_ratios, speed, end_time)
                        data_all.extend(entries)

        return data_all

    def _generate_mixed_entries(self, src_pos, src_type, dest_type, dest_pos, operation, burst, ratios, speed, end_time):
        """混合模式生成（基于原有逻辑）"""
        generator = _create_traffic_generator(operation)
        is_read = operation == "R"

        # 获取有效带宽
        effective_speed = self._get_effective_speed(speed, burst, src_type)

        # 计算各模式的带宽分配
        total_speed = effective_speed
        mode_speeds = {k: int(total_speed * v) for k, v in ratios.items()}
        dest_len = len(dest_pos)
        entries = []

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
                    dest_iter = iter(shuffled_dest_pos)
                elif mode == 1:  # 8-shared
                    shuffled_group_dests = {}
                    for group in range(len(dest_pos) // 8):
                        group_dests = dest_pos[group * 8 : (group + 1) * 8]
                        random.shuffle(group_dests)
                        shuffled_group_dests[group] = iter(group_dests)

                for src in src_pos:
                    for t in time_pattern:
                        if mode == 0:  # 32-shared
                            try:
                                dest = next(dest_iter)
                            except StopIteration:
                                random.shuffle(shuffled_dest_pos)
                                dest_iter = iter(shuffled_dest_pos)
                                dest = next(dest_iter)

                        elif mode == 1:  # 8-shared
                            group = src // 8
                            try:
                                dest = next(shuffled_group_dests[group])
                            except StopIteration:
                                group_dests = dest_pos[group * 8 : (group + 1) * 8]
                                random.shuffle(group_dests)
                                shuffled_group_dests[group] = iter(group_dests)
                                dest = next(shuffled_group_dests[group])

                        else:  # private
                            dest = src if src in dest_pos else dest_pos[src % len(dest_pos)]

                        timestamp = base_time + time_offset + t
                        if timestamp < end_time:
                            entries.append(f"{timestamp},{src},{src_type},{dest},{dest_type},{operation},{burst}\n")

            cycle += 1

        return entries


class CustomMappingStrategy(TrafficStrategy):
    """自定义映射流量策略"""

    def generate(self, ip_maps, speed, burst, end_time, **kwargs):
        """生成自定义映射的流量数据"""
        custom_mapping = kwargs.get("custom_mapping", {})
        overlap = kwargs.get("overlap", 0)

        data_all = []

        # 处理读操作
        if "R" in custom_mapping:
            for src_type in ["gdma", "cdma"]:
                if src_type in ip_maps and ip_maps[src_type]:
                    entries = self._generate_custom_entries(ip_maps[src_type], ip_maps.get("ddr", {}), "R", burst, speed, end_time, overlap, src_type, custom_mapping["R"])
                    data_all.extend(entries)

        # 处理写操作
        if "W" in custom_mapping:
            for src_type in ["gdma", "cdma"]:
                if src_type in ip_maps and ip_maps[src_type]:
                    dest_map = ip_maps.get("l2m", {}) if src_type == "cdma" else ip_maps.get("ddr", {})
                    entries = self._generate_custom_entries(ip_maps[src_type], dest_map, "W", burst, speed, end_time, overlap, src_type, custom_mapping["W"])
                    data_all.extend(entries)

        return data_all

    def _generate_custom_entries(self, src_map, dest_map, operation, burst, speed, end_time, overlap, src_type, mapping):
        """生成自定义映射的具体条目"""
        generator = _create_traffic_generator(operation)
        is_read = operation == "R"

        # 获取有效带宽
        effective_speed = self._get_effective_speed(speed, burst, src_type)

        time_pattern = generator.calculate_time_points(effective_speed, burst, is_read)
        entries = []

        # 扁平化源
        src_items = self._flatten_ip_items(src_map)

        cycle = 0
        while True:
            base_time = cycle * generator.cycle_duration

            # 计算时间偏移
            time_offset = self._calculate_time_offset(overlap, is_read, generator)

            # 检查是否超过结束时间
            if base_time + time_offset >= end_time:
                break

            for t in time_pattern:
                for src_type_key, src in src_items:
                    # 查找该源的自定义目标列表
                    src_key = (src_type_key, src)
                    if src_key in mapping:
                        target_dests = mapping[src_key]
                        # 轮询访问目标列表中的所有目标
                        if not hasattr(self, "custom_cycles"):
                            self.custom_cycles = {}
                        if src_key not in self.custom_cycles:
                            self.custom_cycles[src_key] = itertools.cycle(target_dests)

                        dest_type, dest = next(self.custom_cycles[src_key])
                        timestamp = base_time + time_offset + t
                        if timestamp < end_time:
                            entries.append(f"{timestamp},{src},{src_type_key},{dest},{dest_type},{operation},{burst}\n")

            cycle += 1

        return entries


def generate_data(topo, end_time, file_name, sdma_map, gdma_map, cdma_map, ddr_map, l2m_map, speed, burst, flow_type=0, mix_ratios=None, overlap=0, custom_mapping=None, req_type="R"):
    """
    流量数据生成函数 - 重构为使用新架构（向后兼容）

    :param topo: 拓扑参数（已废弃，保留为兼容性）
    :param end_time: 结束时间
    :param file_name: 输出文件名
    :param sdma_map: SDMA映射
    :param gdma_map: GDMA映射
    :param cdma_map: CDMA映射
    :param ddr_map: DDR映射
    :param l2m_map: L2M映射
    :param speed: 带宽配置
    :param burst: burst长度
    :param flow_type: 流量类型
    :param mix_ratios: 混合比例
    :param overlap: 重叠参数
    :param custom_mapping: 自定义映射配置
    :param req_type: 请求类型
    """
    # 转换参数格式以适配新架构
    ip_maps = {}

    # 只添加非空的映射
    if gdma_map:
        ip_maps["gdma"] = gdma_map
    if sdma_map:
        ip_maps["sdma"] = sdma_map
    if cdma_map:
        ip_maps["cdma"] = cdma_map
    if ddr_map:
        ip_maps["ddr"] = ddr_map
    if l2m_map:
        ip_maps["l2m"] = l2m_map

    # 创建新的生成器并生成数据
    generator = TrafficDataGenerator()
    generator.generate(
        end_time=end_time, file_name=file_name, ip_maps=ip_maps, speed=speed, burst=burst, flow_type=flow_type, mix_ratios=mix_ratios, overlap=overlap, custom_mapping=custom_mapping, req_type=req_type
    )


# 示例使用
if __name__ == "__main__":
    # 参数配置
    END_TIME = 6300  # 结束时间50000ns

    np.random.seed(919)

    TOPO = "3x3"
    REQ_TYPE = "R"
    BURST = 4
    NUM_IP = 64
    FILE_NAME = f"../../test_data/{TOPO}_{REQ_TYPE}.txt"

    SDMA_MAP = {}
    GDMA_MAP = {
        "gdma_0": list(range((32))),
        # "gdma_1": list(range((16))),
    }
    DDR_MAP = {
        "ddr_0": list(range((32))),
        # "ddr_1": list(range((16))),
    }
    CDMA_MAP = {}
    L2M_MAP = {}
    CUSTOM_MAPPING = {}

    # 不同IP类型的带宽配置 (GB/s)
    SPEED = {
        1: {"gdma": 128, "sdma": 64, "cdma": 256},  # burst=1时各IP的带宽
        2: {"gdma": 128, "sdma": 64, "cdma": 256},  # burst=2时各IP的带宽
        4: {"gdma": 128, "sdma": 32, "cdma": 25.2},  # burst=4时各IP的带宽
    }

    overlap = 1

    # 生成数据，使用flow_type=4启用自定义映射
    generate_data(TOPO, END_TIME, FILE_NAME, SDMA_MAP, GDMA_MAP, CDMA_MAP, DDR_MAP, L2M_MAP, SPEED, BURST, flow_type=3, overlap=overlap, custom_mapping=CUSTOM_MAPPING, req_type=REQ_TYPE)

    print(f"Traffic data generated successfully! {FILE_NAME}")
