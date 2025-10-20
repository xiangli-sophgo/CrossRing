"""
D2D (Die-to-Die) 流量生成脚本
基于原版 generate_data.py 的简化版本，专门用于生成 D2D 流量模式。

D2D 流量格式：inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length

主要功能：
- 支持跨 die 和同 die 流量模式
- 可配置的带宽和突发长度
- IP 类型映射支持
- 时间均匀分布的请求生成
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import itertools


class D2DTrafficGenerator:
    """D2D 流量生成器 - 简化版"""

    def __init__(self, num_dies: int = 2, die_topo: str = "5x4", total_bandwidth: float = 128.0):
        """
        初始化 D2D 流量生成器

        Args:
            num_dies: Die 数量，默认 2
            die_topo: 每个 Die 的拓扑类型，默认 "5x4"
            total_bandwidth: 总带宽，单位 GB/s，默认 128
        """
        self.num_dies = num_dies
        self.die_topo = die_topo
        self.total_bandwidth = total_bandwidth

        # 根据拓扑类型设置参数
        if die_topo == "5x4":
            self.nodes_per_die = 40  # 5x4 拓扑有40个节点
        elif die_topo == "4x4":
            self.nodes_per_die = 16  # 4x4 拓扑有16个节点
        else:
            raise ValueError(f"不支持的拓扑类型: {die_topo}")

        # 默认IP映射配置 (根据5x4拓扑)
        self.default_ip_mappings = self._init_default_ip_mappings()

    def _init_default_ip_mappings(self) -> Dict[str, List[int]]:
        """初始化默认的IP映射配置"""
        if self.die_topo == "5x4":
            return {
                "gdma": [6, 7, 26, 27],  # GDMA IP 位置
                "ddr": [12, 13, 32, 33],  # DDR IP 位置
                "l2m": [18, 19, 38, 39],  # L2M IP 位置
                "sdma": [0, 1, 20, 21],  # SDMA IP 位置
                "cdma": [14, 15, 34],  # CDMA IP 位置
            }
        elif self.die_topo == "4x4":
            return {
                "gdma": [0, 1, 2, 3],  # GDMA IP 位置
                "ddr": [12, 13, 14, 15],  # DDR IP 位置
                "l2m": [8, 9, 10, 11],  # L2M IP 位置
                "sdma": [4, 5, 6, 7],  # SDMA IP 位置
                "cdma": [14, 15],  # CDMA IP 位置
            }
        else:
            return {}

    def calculate_request_times(self, bandwidth: float, burst_length: int, window_duration: int) -> List[int]:
        """
        计算请求时间点，使请求在时间窗口内均匀分布

        Args:
            bandwidth: 带宽，单位 GB/s
            burst_length: 突发长度
            window_duration: 时间窗口长度，单位 ns

        Returns:
            List[int]: 时间点列表
        """
        # 计算请求数量 - 基于带宽和突发长度
        total_requests_float = bandwidth * window_duration / (self.total_bandwidth * burst_length)
        total_requests = max(1, round(total_requests_float))

        if total_requests == 0:
            return []

        # 在时间窗口内均匀分布
        time_points = [int(i * window_duration / total_requests) for i in range(total_requests)]
        return time_points

    def generate_time_sequence(self, bandwidth: float, burst_length: int, end_time: int, cycle_duration: int = 1280) -> List[int]:
        """
        生成完整的时间序列，基于周期性重复

        Args:
            bandwidth: 带宽，单位 GB/s
            burst_length: 突发长度
            end_time: 结束时间，单位 ns
            cycle_duration: 周期持续时间，单位 ns，默认 1280ns

        Returns:
            List[int]: 完整的时间序列
        """
        time_sequence = []
        cycle = 0

        while True:
            base_time = cycle * cycle_duration
            if base_time >= end_time:
                break

            # 计算这个周期内的请求时间点
            cycle_times = self.calculate_request_times(bandwidth, burst_length, cycle_duration)

            # 添加到总序列，检查是否超过结束时间
            for t in cycle_times:
                timestamp = base_time + t
                if timestamp < end_time:
                    time_sequence.append(timestamp)
                else:
                    break

            cycle += 1

        return time_sequence

    def generate_cross_die_traffic(self, src_die: int, src_nodes: List[tuple], dst_die: int, dst_nodes: List[tuple], req_type: str, burst_length: int, bandwidth: float, end_time: int) -> List[str]:
        """
        生成跨 die 流量

        Args:
            src_die: 源 Die ID
            src_nodes: 源节点列表，格式为 [(node_id, ip_name), ...]
            dst_die: 目标 Die ID
            dst_nodes: 目标节点列表，格式为 [(node_id, ip_name), ...]
            req_type: 请求类型 ('R' 或 'W')
            burst_length: 突发长度
            bandwidth: 带宽，单位 GB/s
            end_time: 结束时间，单位 ns

        Returns:
            List[str]: D2D 流量条目列表
        """
        traffic_entries = []

        # 生成时间序列
        time_sequence = self.generate_time_sequence(bandwidth, burst_length, end_time)

        # 为每个时间点生成流量
        for timestamp in time_sequence:
            for src_node, src_ip in src_nodes:
                # 随机选择目标节点
                dst_node, dst_ip = random.choice(dst_nodes)

                # 格式：inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length
                entry = f"{timestamp}, {src_die}, {src_node}, {src_ip}, {dst_die}, {dst_node}, {dst_ip}, {req_type}, {burst_length}\n"
                traffic_entries.append(entry)

        return traffic_entries

    def generate_same_die_traffic(self, die_id: int, src_nodes: List[tuple], dst_nodes: List[tuple], req_type: str, burst_length: int, bandwidth: float, end_time: int) -> List[str]:
        """
        生成同 die 内流量

        Args:
            die_id: Die ID
            src_nodes: 源节点列表，格式为 [(node_id, ip_name), ...]
            dst_nodes: 目标节点列表，格式为 [(node_id, ip_name), ...]
            req_type: 请求类型 ('R' 或 'W')
            burst_length: 突发长度
            bandwidth: 带宽，单位 GB/s
            end_time: 结束时间，单位 ns

        Returns:
            List[str]: D2D 流量条目列表
        """
        # 同 die 流量实际上就是跨 die 流量，但源和目标在同一个 die
        return self.generate_cross_die_traffic(die_id, src_nodes, die_id, dst_nodes, req_type, burst_length, bandwidth, end_time)

    def generate_mixed_traffic(
        self,
        cross_die_ratio: float = 0.7,
        src_ip_config: Dict = None,
        dst_ip_config: Dict = None,
        req_type: str = "R",
        burst_length: int = 4,
        bandwidth: float = 64.0,
        end_time: int = 5000,
        src_die: int = 0,
        dst_die: int = 1,
    ) -> List[str]:
        """
        生成混合模式流量 (跨 die + 同 die)

        Args:
            cross_die_ratio: 跨 die 流量比例，默认 0.7 (70%)
            src_ip_config: 源 IP 配置，格式: {"ip_type": [node_list]}
            dst_ip_config: 目标 IP 配置
            req_type: 请求类型
            burst_length: 突发长度
            bandwidth: 带宽
            end_time: 结束时间
            src_die: 源 Die ID，默认 0
            dst_die: 目标 Die ID，默认 1

        Returns:
            List[str]: 混合流量条目列表
        """
        if src_ip_config is None:
            src_ip_config = {"gdma": self.default_ip_mappings["gdma"]}
        if dst_ip_config is None:
            dst_ip_config = {"ddr": self.default_ip_mappings["ddr"]}

        traffic_entries = []

        # 计算跨 die 和同 die 的带宽分配
        cross_die_bandwidth = bandwidth * cross_die_ratio
        same_die_bandwidth = bandwidth * (1 - cross_die_ratio)

        # 生成跨 die 流量 (src_die -> dst_die)
        for src_ip, src_nodes in src_ip_config.items():
            for dst_ip, dst_nodes in dst_ip_config.items():
                cross_entries = self.generate_cross_die_traffic(src_die, src_nodes, src_ip, dst_die, dst_nodes, dst_ip, req_type, burst_length, cross_die_bandwidth, end_time)
                traffic_entries.extend(cross_entries)

        # 生成同 die 流量 (src_die 内部)
        for src_ip, src_nodes in src_ip_config.items():
            for dst_ip, dst_nodes in dst_ip_config.items():
                same_entries = self.generate_same_die_traffic(src_die, src_nodes, src_ip, dst_nodes, dst_ip, req_type, burst_length, same_die_bandwidth, end_time)
                traffic_entries.extend(same_entries)

        return traffic_entries

    def get_ip_mapping(self, ip_type: str) -> List[int]:
        """
        获取 IP 映射

        Args:
            ip_type: IP 类型

        Returns:
            List[int]: 节点列表
        """
        if ip_type in self.default_ip_mappings:
            return self.default_ip_mappings[ip_type]
        else:
            raise ValueError(f"未找到 IP 类型 '{ip_type}' 的映射")

    @staticmethod
    def get_rotated_node_mapping(rows: int = 5, cols: int = 4, rotation: int = 0) -> Dict[int, int]:
        """
        计算Die旋转后每个原始节点对应的新节点ID

        Args:
            rows: 原始布局的行数
            cols: 原始布局的列数
            rotation: 旋转角度，可选值：0, 90, 180, 270（顺时针）

        Returns:
            Dict[int, int]: {原始节点ID: 旋转后节点ID} 的映射字典

        Example:
            # Die1顺时针旋转90°后，Die0的节点0对应Die1的哪个节点
            mapping = get_rotated_node_mapping(5, 4, 90)
            die1_node = mapping[0]  # Die0节点0在Die1中的对应节点
        """
        mapping = {}
        total_nodes = rows * cols

        for node_id in range(total_nodes):
            row = node_id // cols
            col = node_id % cols

            if rotation == 0:
                # 无旋转
                new_row, new_col = row, col
                new_cols = cols
            elif rotation == 90:
                # 顺时针90°：5行4列 → 4行5列
                new_row = col
                new_col = rows - 1 - row
                new_cols = rows
            elif rotation == 180:
                # 180°：5行4列 → 5行4列
                new_row = rows - 1 - row
                new_col = cols - 1 - col
                new_cols = cols
            elif rotation == 270:
                # 顺时针270°：5行4列 → 4行5列
                new_row = cols - 1 - col
                new_col = row
                new_cols = rows
            else:
                raise ValueError(f"不支持的旋转角度: {rotation}，只支持0/90/180/270")

            new_node_id = new_row * new_cols + new_col
            mapping[node_id] = new_node_id

        return mapping

    def validate_node_range(self, nodes: List[int]) -> bool:
        """
        验证节点范围是否有效

        Args:
            nodes: 节点列表

        Returns:
            bool: 是否有效
        """
        return all(0 <= node < self.nodes_per_die for node in nodes)

    def create_ip_config(self, ip_configs: Dict[str, List[int]] = None) -> Dict[str, List[int]]:
        """
        创建标准化的 IP 配置

        Args:
            ip_configs: 原始 IP 配置，可以是 {"ip_type": [nodes]} 或 {"ip_type": count}

        Returns:
            Dict[str, List[int]]: 标准化的 IP 配置
        """
        if ip_configs is None:
            return {"gdma": self.default_ip_mappings["gdma"]}

        result = {}
        for ip_type, config in ip_configs.items():
            if isinstance(config, list):
                # 直接指定节点列表
                if self.validate_node_range(config):
                    result[ip_type] = config
                else:
                    raise ValueError(f"IP 类型 '{ip_type}' 的节点范围超出有效范围")
            elif isinstance(config, int):
                # 指定数量，从默认映射中选择
                default_nodes = self.get_ip_mapping(ip_type)
                if config <= len(default_nodes):
                    result[ip_type] = default_nodes[:config]
                else:
                    raise ValueError(f"IP 类型 '{ip_type}' 请求的节点数量超过可用数量")
            else:
                raise ValueError(f"IP 类型 '{ip_type}' 的配置格式无效")

        return result

    def generate_traffic_file(
        self,
        filename: str,
        traffic_configs: List[Dict],
        traffic_mode: str = "cross_die",
        end_time: int = 5000,
    ):
        """
        生成 D2D 流量文件，支持多个 Die 对配置

        Args:
            filename: 输出文件名
            traffic_configs: 流量配置列表，每个配置包含:
                {
                    "src_die": int,
                    "dst_die": int,
                    "src_ip_config": Dict[str, List[int]],
                    "dst_ip_config": Dict[str, List[int]],
                    "req_type": str,
                    "burst_length": int,
                    "bandwidth": float,
                }
            traffic_mode: 流量模式 ("cross_die", "same_die", "mixed")
            end_time: 结束时间，单位 ns

        Example:
            traffic_configs = [
                {
                    "src_die": 0,
                    "dst_die": 2,
                    "src_ip_config": {"gdma_0": [6]},
                    "dst_ip_config": {"ddr_0": [12]},
                    "req_type": "R",
                    "burst_length": 4,
                    "bandwidth": 128.0,
                },
                {
                    "src_die": 1,
                    "dst_die": 3,
                    "src_ip_config": {"gdma_1": [7]},
                    "dst_ip_config": {"ddr_1": [13]},
                    "req_type": "W",
                    "burst_length": 8,
                    "bandwidth": 64.0,
                }
            ]
        """
        all_entries = []

        # 遍历每个流量配置，生成相应的流量数据
        for config in traffic_configs:
            # 提取配置参数
            src_die = config["src_die"]
            dst_die = config["dst_die"]
            src_ip_config = config["src_ip_config"]
            dst_ip_config = config["dst_ip_config"]
            req_type = config.get("req_type", "R")
            burst_length = config.get("burst_length", 4)
            bandwidth = config.get("bandwidth", 64.0)

            # 生成流量数据
            config_entries = []

            if traffic_mode == "cross_die":
                # 只生成跨 die 流量
                for src_ip, src_nodes in src_ip_config.items():
                    for dst_ip, dst_nodes in dst_ip_config.items():
                        # 转换为 [(node_id, ip_name), ...] 格式
                        src_nodes_with_ip = [(node, src_ip) for node in src_nodes]
                        dst_nodes_with_ip = [(node, dst_ip) for node in dst_nodes]
                        entries = self.generate_cross_die_traffic(src_die, src_nodes_with_ip, dst_die, dst_nodes_with_ip, req_type, burst_length, bandwidth, end_time)
                        config_entries.extend(entries)

            elif traffic_mode == "same_die":
                # 只生成同 die 流量
                for src_ip, src_nodes in src_ip_config.items():
                    for dst_ip, dst_nodes in dst_ip_config.items():
                        # 转换为 [(node_id, ip_name), ...] 格式
                        src_nodes_with_ip = [(node, src_ip) for node in src_nodes]
                        dst_nodes_with_ip = [(node, dst_ip) for node in dst_nodes]
                        entries = self.generate_same_die_traffic(src_die, src_nodes_with_ip, dst_nodes_with_ip, req_type, burst_length, bandwidth, end_time)
                        config_entries.extend(entries)

            else:
                raise ValueError(f"不支持的流量模式: {traffic_mode}")

            # 将当前配置的流量数据添加到总列表
            all_entries.extend(config_entries)
            print(f"Die {src_die} -> Die {dst_die}: 生成了 {len(config_entries)} 条流量记录")

        # 按时间排序并写入文件
        all_entries.sort(key=lambda x: int(x.split(",")[0]))

        with open(filename, "w") as f:
            # 写入头部注释
            f.write("# D2D Multi-Die-Pair Traffic\n")
            f.write("# Format: inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length\n")
            f.writelines(all_entries)

        print(f"多 Die 对 D2D 流量文件已生成: {filename}")
        print(f"总共生成了 {len(all_entries)} 条流量记录")

        # 打印配置总结
        print("\n流量配置总结:")
        for i, config in enumerate(traffic_configs, 1):
            print(
                f"  {i}. Die {config['src_die']} -> Die {config['dst_die']}: "
                f"{config.get('bandwidth', 64.0)} GB/s, "
                f"{config.get('req_type', 'R')} 请求, "
                f"burst_length={config.get('burst_length', 4)}"
            )


def generate_simple_example():
    """
    生成简单的2Die流量示例

    用于演示基本的D2D流量生成功能
    - Die0 -> Die2: 128 GB/s
    - Die2 -> Die0: 64 GB/s
    """
    generator = D2DTrafficGenerator(die_topo="5x4")

    traffic_configs = [
        {
            "src_die": 0,
            "dst_die": 2,
            "src_ip_config": {
                "gdma_0": [6],
                "gdma_1": [6],
            },
            "dst_ip_config": {
                "ddr_0": [12],
            },
            "req_type": "R",
            "burst_length": 4,
            "bandwidth": 128.0,
        },
        {
            "src_die": 2,
            "dst_die": 0,
            "src_ip_config": {
                "gdma_1": [7],
            },
            "dst_ip_config": {
                "ddr_1": [13],
            },
            "req_type": "R",
            "burst_length": 4,
            "bandwidth": 64.0,
        },
    ]

    generator.generate_traffic_file(
        filename="../../test_data/d2d_data_simple_example.txt",
        traffic_configs=traffic_configs,
        traffic_mode="cross_die",
        end_time=1000,
    )


def generate_4die_stress_test():
    """
    生成4Die压力测试场景

    包含4个测试场景：
    1. 全互联均匀流量 - 所有Die之间互相通信
    2. 环形单向流量 - Die0→Die1→Die2→Die3→Die0
    3. 对角双向流量 - Die0↔Die2, Die1↔Die3
    4. 热点集中流量 - 所有Die访问Die0

    Die配置：
    - 每个Die为5行4列（20个节点）
    - Die0: 0°旋转（基准）
    - Die1: 顺时针90°旋转
    - Die2: 180°旋转
    - Die3: 顺时针270°旋转
    """
    generator = D2DTrafficGenerator(die_topo="5x4")

    # ========== 第1步：定义Die0的基础配置（5行4列=20节点） ==========
    # Die0布局：
    #  0   1   2   3
    #  4   5   6   7
    #  8   9  10  11
    # 12  13  14  15
    # 16  17  18  19

    # Die0 GDMA配置：每个IP实例挂载的节点列表
    die0_gdma_base = {
        "gdma_0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19],  # 只有一个GDMA的节点
        "gdma_1": [3, 15, 19],  # 有两个GDMA的节点上的第二个实例
    }

    # Die0 DDR配置：每个IP实例挂载的节点列表
    die0_ddr_base = {
        "ddr_0": [3, 7, 11, 15],  # 每个DDR节点的第一个实例
        "ddr_1": [3, 7, 11, 15],  # 每个DDR节点的第二个实例
    }

    # ========== 第2步：计算Die1-3的旋转节点配置 ==========
    # rotations = {0: 0, 1: 90, 2: 180, 3: 270}
    rotations = {0: 0, 1: 180}
    die_configs = {}

    for die_id, rotation in rotations.items():
        mapping = D2DTrafficGenerator.get_rotated_node_mapping(rows=5, cols=4, rotation=rotation)

        # 应用旋转映射到每个IP实例的节点列表
        gdma_config = {}
        for ip_name, nodes in die0_gdma_base.items():
            gdma_config[ip_name] = [mapping[node] for node in nodes]

        ddr_config = {}
        for ip_name, nodes in die0_ddr_base.items():
            ddr_config[ip_name] = [mapping[node] for node in nodes]

        die_configs[die_id] = {"gdma": gdma_config, "ddr": ddr_config}

        # 打印排序后的配置
        gdma_config_sorted = {ip: sorted(nodes) for ip, nodes in gdma_config.items()}
        ddr_config_sorted = {ip: sorted(nodes) for ip, nodes in ddr_config.items()}

        print(f"Die{die_id} (旋转{rotation}°):")
        print(f"  GDMA配置: {gdma_config_sorted}")
        print(f"  DDR配置: {ddr_config_sorted}")
        print()

    print("=" * 60)
    print("生成场景：环形单向流量 (Die0→Die1→Die2→Die3→Die0)")
    print("=" * 60)

    traffic_configs_2 = []
    ring_pairs = [
        (0, 1),
        # (0, 2),
        # (0, 3),
        (1, 0),
        # (1, 2),
        # (1, 3),
        # (2, 0),
        # (2, 1),
        # (2, 3),
        # (3, 0),
        # (3, 1),
        # (3, 2),
        # (2, 3),
        # (3, 0),
    ]
    for src_die, dst_die in ring_pairs:
        traffic_configs_2.append(
            {
                "src_die": src_die,
                "dst_die": dst_die,
                "src_ip_config": die_configs[src_die]["gdma"],
                "dst_ip_config": die_configs[dst_die]["ddr"],
                "req_type": "W",
                "burst_length": 4,
                "bandwidth": 48.0,  # 每个方向256GB/s
            }
        )

    generator.generate_traffic_file(
        filename="../../test_data/d2d_4die_1016.txt",
        traffic_configs=traffic_configs_2,
        traffic_mode="cross_die",
        end_time=1000,
    )
    print()


if __name__ == "__main__":
    # 生成简单的2Die示例
    # generate_simple_example()

    # 生成4Die压力测试场景
    print("=" * 60)
    print("开始生成4Die压力测试场景...")
    print("=" * 60)
    generate_4die_stress_test()
