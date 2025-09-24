import random
import numpy as np


class TrafficGenerator:
    """流量生成器，计算发送时间点"""

    def __init__(self, total_bandwidth=128):
        """
        :param total_bandwidth: 总带宽基准值(GB/s)
        """
        self.total_bandwidth = total_bandwidth

    def calculate_time_points(self, speed, burst, duration=1280):
        """
        计算发送时间点序列

        :param speed: 实际速度 (GB/s)
        :param burst: burst长度
        :param duration: 时间窗口持续时间 (ns)
        :return: 时间点列表
        """
        # 计算总传输次数
        total_transfers_float = speed * duration / (self.total_bandwidth * burst)
        total_transfers = max(1, round(total_transfers_float))

        if total_transfers == 0:
            return []

        # 均匀分布时间点
        time_points = [int(i * duration / total_transfers) for i in range(total_transfers)]
        return time_points


def generate_traffic_from_config(config, end_time):
    """
    根据单个配置生成流量数据

    :param config: 配置字典，包含：
        - src_map: 源IP映射 {"ip_type": [positions]}
        - dst_map: 目标IP映射 {"ip_type": [positions]}
        - speed: 带宽 (GB/s)
        - burst: burst长度
        - req_type: 请求类型 ("R" 或 "W")
    :param end_time: 结束时间
    :return: 流量数据列表
    """
    generator = TrafficGenerator()

    # 获取配置参数
    src_map = config["src_map"]
    dst_map = config["dst_map"]
    speed = config["speed"]
    burst = config["burst"]
    req_type = config["req_type"]

    # 扁平化源和目标映射
    src_items = [(ip_type, pos) for ip_type, positions in src_map.items() for pos in positions]
    dst_items = [(ip_type, pos) for ip_type, positions in dst_map.items() for pos in positions]

    if not src_items or not dst_items:
        return []

    # 计算时间点模式
    duration = 1280  # 读写时间窗口
    time_pattern = generator.calculate_time_points(speed, burst, duration)

    data = []
    cycle = 0

    while True:
        base_time = cycle * duration

        # 检查是否超过结束时间
        if base_time >= end_time:
            break

        # 为每个时间点生成流量
        for t in time_pattern:
            for src_type, src_pos in src_items:
                # 随机选择目标
                dst_type, dst_pos = random.choice(dst_items)

                timestamp = base_time + t
                if timestamp < end_time:
                    data.append((timestamp, src_pos, src_type, dst_pos, dst_type, req_type, burst))

        cycle += 1

    return data


def generate_traffic_from_configs(configs, end_time, output_file):
    """
    根据多个配置生成并合并流量数据

    :param configs: 配置列表
    :param end_time: 结束时间
    :param output_file: 输出文件路径
    """
    all_data = []

    # 为每个配置生成数据
    for config in configs:
        data = generate_traffic_from_config(config, end_time)
        all_data.extend(data)

    # 按时间和源位置排序
    all_data.sort(key=lambda x: (x[0], x[1]))

    # 写入文件
    with open(output_file, "w") as f:
        for timestamp, src_pos, src_type, dst_pos, dst_type, req_type, burst in all_data:
            f.write(f"{timestamp},{src_pos},{src_type},{dst_pos},{dst_type},{req_type},{burst}\n")


# 示例使用
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(919)
    random.seed(919)

    # 配置参数
    END_TIME = 6300
    burst = 4
    req_type = "R"
    OUTPUT_FILE = f"../../test_data/data_0924_{req_type}.txt"

    # 定义多个配置
    configs = [
        {
            "src_map": {
                "gdma_0": list(range(16)),
                "gdma_1": list(range(16)),
            },
            "dst_map": {
                "ddr_0": list(range(16)),
                "ddr_1": list(range(16)),
            },
            "speed": 128,
            "burst": burst,
            "req_type": req_type,
        },
        # {
        #     "src_map": {
        #         "gdma_0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19],
        #         "gdma_1": [3, 15, 19],
        #     },
        #     "dst_map": {
        #         "ddr_0": [3, 7, 11, 15],
        #         "ddr_1": [3, 7, 11, 15],
        #     },
        #     "speed": 51.2,
        #     "burst": burst,
        #     "req_type": req_type,
        # },
        # {
        #     "src_map": {
        #         "cdma_0": [18],
        #         "cdma_1": [18],
        #     },
        #     "dst_map": {
        #         "ddr_0": [3, 7, 11, 15],
        #         "ddr_1": [3, 7, 11, 15],
        #     },
        #     "speed": 56,
        #     "burst": burst,
        #     "req_type": "R",
        # },
        # {
        #     "src_map": {
        #         "cdma_0": [18],
        #         "cdma_1": [18],
        #     },
        #     "dst_map": {
        #         "ddr_0": [3, 7, 11, 15],
        #         "ddr_1": [3, 7, 11, 15],
        #     },
        #     "speed": 56,
        #     "burst": burst,
        #     "req_type": "W",
        # },
    ]

    # 生成数据
    generate_traffic_from_configs(configs, END_TIME, OUTPUT_FILE)
    print(f"流量数据生成成功！文件：{OUTPUT_FILE}")
