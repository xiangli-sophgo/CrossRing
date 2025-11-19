"""
Traffic IP Extractor - 从traffic文件中提取IP接口需求

该模块负责解析traffic文件,提取所有需要创建的IP接口信息,
支持单Die和多Die(D2D)两种traffic格式。
"""

import os
from typing import Set, Tuple, Dict, List


class TrafficIPExtractor:
    """从traffic文件中提取IP接口需求的工具类"""

    def __init__(self):
        self.required_ips: Set[Tuple[int, str]] = set()
        self.has_cross_die_traffic: bool = False
        self.traffic_format: str = "unknown"  # "single_die" or "d2d"

    def extract_from_file(self, traffic_file_path: str) -> Dict:
        """
        从单个traffic文件中提取IP需求

        Args:
            traffic_file_path: traffic文件路径

        Returns:
            包含以下信息的字典:
            - required_ips: Set[(node_id, ip_type)]
            - has_cross_die: 是否有跨Die请求
            - traffic_format: traffic文件格式
        """
        if not os.path.exists(traffic_file_path):
            raise FileNotFoundError(f"Traffic文件不存在: {traffic_file_path}")

        with open(traffic_file_path, 'r') as f:
            for line in f:
                line = line.strip()

                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue

                # 解析一行traffic数据
                self._parse_traffic_line(line)

        return {
            "required_ips": self.required_ips,
            "has_cross_die": self.has_cross_die_traffic,
            "traffic_format": self.traffic_format
        }

    def extract_from_multiple_files(self, traffic_file_paths: List[str]) -> Dict:
        """
        从多个traffic文件中提取IP需求(用于traffic chain场景)

        Args:
            traffic_file_paths: traffic文件路径列表

        Returns:
            合并后的IP需求字典
        """
        for file_path in traffic_file_paths:
            self.extract_from_file(file_path)

        return {
            "required_ips": self.required_ips,
            "has_cross_die": self.has_cross_die_traffic,
            "traffic_format": self.traffic_format
        }

    def _parse_traffic_line(self, line: str):
        """
        解析单行traffic数据并提取IP信息

        单Die格式(7字段):
        inject_time, src_node, src_ip, dst_node, dst_ip, req_type, burst_length

        D2D格式(9字段):
        inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length
        """
        parts = [p.strip() for p in line.split(',')]

        # 根据字段数量判断格式
        if len(parts) == 7:
            # 单Die格式
            self._parse_single_die_format(parts)
        elif len(parts) >= 9:
            # D2D格式
            self._parse_d2d_format(parts)
        else:
            # 格式不识别,跳过
            return

    def _parse_single_die_format(self, parts: List[str]):
        """
        解析单Die格式的traffic行
        格式: inject_time, src_node, src_ip, dst_node, dst_ip, req_type, burst_length
        """
        try:
            src_node = int(parts[1])
            src_ip = parts[2].strip()
            dst_node = int(parts[3])
            dst_ip = parts[4].strip()

            # 添加到需求集合
            self.required_ips.add((src_node, src_ip))
            self.required_ips.add((dst_node, dst_ip))

            # 标记格式
            if self.traffic_format == "unknown":
                self.traffic_format = "single_die"

        except (ValueError, IndexError):
            # 解析失败,跳过这行
            pass

    def _parse_d2d_format(self, parts: List[str]):
        """
        解析D2D格式的traffic行
        格式: inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length
        """
        try:
            src_die = int(parts[1])
            src_node = int(parts[2])
            src_ip = parts[3].strip()
            dst_die = int(parts[4])
            dst_node = int(parts[5])
            dst_ip = parts[6].strip()

            # 添加到需求集合
            self.required_ips.add((src_node, src_ip))
            self.required_ips.add((dst_node, dst_ip))

            # 检测跨Die流量
            if src_die != dst_die:
                self.has_cross_die_traffic = True

            # 标记格式
            if self.traffic_format == "unknown":
                self.traffic_format = "d2d"

        except (ValueError, IndexError):
            # 解析失败,跳过这行
            pass

    @staticmethod
    def get_unique_ip_types(required_ips: Set[Tuple[int, str]]) -> List[str]:
        """
        从(node_id, ip_type)集合中提取唯一的IP类型列表

        Args:
            required_ips: IP需求集合

        Returns:
            排序后的唯一IP类型列表,如["ddr_0", "gdma_0", "gdma_1"]
        """
        ip_types = set()
        for node_id, ip_type in required_ips:
            ip_types.add(ip_type)

        return sorted(list(ip_types))

    @staticmethod
    def infer_channel_spec(ip_types: List[str]) -> Dict[str, int]:
        """
        从IP类型列表反向推断CHANNEL_SPEC配置

        Args:
            ip_types: IP类型列表,如["gdma_0", "gdma_1", "ddr_0"]

        Returns:
            CHANNEL_SPEC字典,如{"gdma": 2, "ddr": 1}
        """
        from collections import defaultdict

        channel_counts = defaultdict(set)

        for ip_type in ip_types:
            # 分离基础类型和索引
            # 处理"gdma_0"和"d2d_rn_0"两种情况
            if '_' in ip_type:
                parts = ip_type.rsplit('_', 1)
                base_type = parts[0]
                idx_str = parts[1]

                try:
                    idx = int(idx_str)
                    channel_counts[base_type].add(idx)
                except ValueError:
                    # 无法解析为数字,可能是特殊IP类型
                    pass

        # 计算每种类型的数量(最大索引+1)
        channel_spec = {}
        for base_type, indices in channel_counts.items():
            if indices:
                channel_spec[base_type] = max(indices) + 1

        return channel_spec


def extract_ip_requirements(traffic_file_path: str) -> Dict:
    """
    便捷函数:从traffic文件中提取IP需求

    Args:
        traffic_file_path: traffic文件路径

    Returns:
        IP需求字典
    """
    extractor = TrafficIPExtractor()
    return extractor.extract_from_file(traffic_file_path)
