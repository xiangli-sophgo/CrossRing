"""
数据收集器模块 - 提供请求数据收集、延迟统计和绕环统计功能

包含:
1. RequestCollector - 请求数据收集类
2. LatencyStatsCollector - 延迟统计收集类
3. CircuitStatsCollector - 绕环统计收集类
"""

import os
import csv
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from collections import defaultdict
from .analyzers import RequestInfo, FLIT_SIZE_BYTES
from .d2d_analyzer import D2DRequestInfo
from .core_calculators import DataValidator
from src.utils.components import Flit
from src.utils.components.flit import get_original_source_type, get_original_destination_type


class RequestCollector:
    """请求数据收集器 - 从仿真模型或CSV文件收集请求数据"""

    def __init__(self, network_frequency: float = 2.0):
        """
        初始化请求收集器

        Args:
            network_frequency: 网络频率 (GHz)
        """
        self.network_frequency = network_frequency
        self.requests: List[RequestInfo] = []
        self.d2d_requests: List[D2DRequestInfo] = []

    def collect_requests_data(self, sim_model, simulation_end_cycle=None) -> List[RequestInfo]:
        """
        从仿真模型收集请求数据

        Args:
            sim_model: 仿真模型对象
            simulation_end_cycle: 仿真结束周期

        Returns:
            RequestInfo列表
        """
        self.requests.clear()

        for packet_id, flits in sim_model.data_network.arrive_flits.items():
            if not flits or len(flits) != flits[0].burst_length:
                continue

            # 根据flit_id找到真正的第一个和最后一个flit
            first_flit: Flit = min(flits, key=lambda f: f.flit_id)
            representative_flit: Flit = max(flits, key=lambda f: f.flit_id)

            # 计算不同角度的结束时间，并验证时间值有效性
            if not DataValidator.is_valid_number(representative_flit.data_received_complete_cycle):
                continue
            network_end_time = representative_flit.data_received_complete_cycle // self.network_frequency

            if representative_flit.req_type == "read":
                # 读请求：RN在收到数据时结束，SN在发出数据时结束
                rn_end_time = representative_flit.data_received_complete_cycle // self.network_frequency  # RN收到数据
                sn_end_time = first_flit.data_entry_noc_from_cake1_cycle // self.network_frequency  # SN发出数据

                # 读请求：flit的source是SN(DDR/L2M)，destination是RN(SDMA/GDMA/CDMA)
                actual_source_node = representative_flit.destination  # 实际发起请求的节点
                actual_dest_node = representative_flit.source  # 实际目标节点
                # 读请求中需要交换type获取逻辑，因为flit.source_type=DDR但actual_source_node=GDMA
                actual_source_type = get_original_destination_type(representative_flit)  # 实际发起请求的类型(GDMA)
                actual_dest_type = get_original_source_type(representative_flit)  # 实际目标类型(DDR)
            else:  # write
                # 写请求：RN在发出数据时结束，SN在收到数据时结束
                rn_end_time = first_flit.data_entry_noc_from_cake0_cycle // self.network_frequency  # RN发出数据
                sn_end_time = representative_flit.data_received_complete_cycle // self.network_frequency  # SN收到数据

                # 写请求：flit的source是RN(SDMA/GDMA/CDMA)，destination是SN(DDR/L2M)
                actual_source_node = representative_flit.source  # 实际发起请求的节点
                actual_dest_node = representative_flit.destination  # 实际目标节点
                actual_source_type = get_original_source_type(representative_flit)  # 实际发起请求的类型
                actual_dest_type = get_original_destination_type(representative_flit)  # 实际目标类型

            # 收集保序信息
            src_dest_order_id = getattr(representative_flit, "src_dest_order_id", -1)
            packet_category = getattr(representative_flit, "packet_category", "")

            # 收集每个数据flit的尝试下环次数
            data_eject_attempts_h_list = []
            data_eject_attempts_v_list = []
            data_ordering_blocked_h_list = []
            data_ordering_blocked_v_list = []
            for data_flit in flits:
                data_eject_attempts_h_list.append(data_flit.eject_attempts_h)
                data_eject_attempts_v_list.append(data_flit.eject_attempts_v)
                data_ordering_blocked_h_list.append(data_flit.ordering_blocked_eject_h)
                data_ordering_blocked_v_list.append(data_flit.ordering_blocked_eject_v)

            # 验证开始时间
            if not DataValidator.is_valid_number(representative_flit.cmd_entry_cake0_cycle):
                continue
            start_time = representative_flit.cmd_entry_cake0_cycle // self.network_frequency

            request_info = RequestInfo(
                packet_id=packet_id,
                start_time=start_time,
                end_time=network_end_time,  # 整体网络结束时间
                rn_end_time=rn_end_time,
                sn_end_time=sn_end_time,
                req_type=representative_flit.req_type,
                source_node=actual_source_node,  # 使用修正后的源节点
                dest_node=actual_dest_node,  # 使用修正后的目标节点
                source_type=actual_source_type,  # 使用修正后的源类型
                dest_type=actual_dest_type,  # 使用修正后的目标类型
                burst_length=representative_flit.burst_length,
                total_bytes=representative_flit.burst_length * 128,
                cmd_latency=representative_flit.cmd_latency // self.network_frequency,
                data_latency=representative_flit.data_latency // self.network_frequency,
                transaction_latency=representative_flit.transaction_latency // self.network_frequency,
                # 保序相关字段
                src_dest_order_id=src_dest_order_id,
                packet_category=packet_category,
                # 所有cycle数据字段
                cmd_entry_cake0_cycle=getattr(representative_flit, "cmd_entry_cake0_cycle", -1),
                cmd_entry_noc_from_cake0_cycle=getattr(representative_flit, "cmd_entry_noc_from_cake0_cycle", -1),
                cmd_entry_noc_from_cake1_cycle=getattr(representative_flit, "cmd_entry_noc_from_cake1_cycle", -1),
                cmd_received_by_cake0_cycle=getattr(representative_flit, "cmd_received_by_cake0_cycle", -1),
                cmd_received_by_cake1_cycle=getattr(representative_flit, "cmd_received_by_cake1_cycle", -1),
                data_entry_noc_from_cake0_cycle=getattr(first_flit, "data_entry_noc_from_cake0_cycle", -1),
                data_entry_noc_from_cake1_cycle=getattr(first_flit, "data_entry_noc_from_cake1_cycle", -1),
                data_received_complete_cycle=getattr(representative_flit, "data_received_complete_cycle", -1),
                rsp_entry_network_cycle=getattr(representative_flit, "rsp_entry_network_cycle", -1),
                # 数据flit的尝试下环次数列表
                data_eject_attempts_h_list=data_eject_attempts_h_list,
                data_eject_attempts_v_list=data_eject_attempts_v_list,
                # 数据flit因保序被阻止的下环次数列表
                data_ordering_blocked_h_list=data_ordering_blocked_h_list,
                data_ordering_blocked_v_list=data_ordering_blocked_v_list,
            )

            # 验证请求完整性
            if not DataValidator.validate_request(request_info):
                continue

            self.requests.append(request_info)

        # 按开始时间排序
        self.requests.sort(key=lambda x: x.start_time)

        return self.requests

    def load_requests_from_csv(self, csv_folder: str, config_dict: Dict = None) -> List[RequestInfo]:
        """
        从CSV文件重新加载请求数据

        Args:
            csv_folder: 包含read_requests.csv和write_requests.csv的文件夹
            config_dict: 配置字典，如果为None则尝试从保存的配置加载

        Returns:
            RequestInfo列表
        """
        self.requests.clear()

        # 加载配置
        if config_dict is None:
            config_file = os.path.join(csv_folder, "analysis_config.json")
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            else:
                config_data = {
                    "network_frequency": 1.0,
                    "burst": 4,
                }
        else:
            config_data = config_dict

        # 更新网络频率
        self.network_frequency = config_data.get("network_frequency", 1.0)

        # 读取CSV文件
        read_csv = os.path.join(csv_folder, "read_requests.csv")
        write_csv = os.path.join(csv_folder, "write_requests.csv")

        # 处理读写请求
        self._load_requests_from_single_csv(read_csv, "read")
        self._load_requests_from_single_csv(write_csv, "write")

        return self.requests

    def _load_requests_from_single_csv(self, csv_path: str, req_type: str):
        """
        从单个CSV文件加载请求数据

        Args:
            csv_path: CSV文件路径
            req_type: 请求类型 ("read" 或 "write")
        """
        if not os.path.exists(csv_path):
            return

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # 处理保序字段（向后兼容）
            src_dest_order_id = int(row.get("src_dest_order_id", -1))
            packet_category = str(row.get("packet_category", ""))

            # 验证并清理数据
            try:
                packet_id = int(row["packet_id"])
                start_time = DataValidator.sanitize_value(row["start_time_ns"], 0)
                end_time = DataValidator.sanitize_value(row["end_time_ns"], 0)
                rn_end_time = DataValidator.sanitize_value(row["rn_end_time_ns"], 0)
                sn_end_time = DataValidator.sanitize_value(row["sn_end_time_ns"], 0)
                total_bytes = DataValidator.sanitize_value(row["total_bytes"], 0)
                burst_length = DataValidator.sanitize_value(row["burst_length"], 0)

                # 检查基本有效性
                if start_time <= 0 or end_time <= 0 or total_bytes <= 0 or burst_length <= 0:
                    continue
                if start_time >= end_time:
                    continue

            except (ValueError, KeyError) as e:
                continue

            request_info = RequestInfo(
                packet_id=packet_id,
                start_time=start_time,
                end_time=end_time,
                rn_end_time=rn_end_time,
                sn_end_time=sn_end_time,
                req_type=req_type,
                source_node=int(row["source_node"]),
                dest_node=int(row["dest_node"]),
                source_type=str(row["source_type"]),
                dest_type=str(row["dest_type"]),
                burst_length=burst_length,
                total_bytes=total_bytes,
                cmd_latency=int(row["cmd_latency_ns"]),
                data_latency=int(row["data_latency_ns"]),
                transaction_latency=int(row["transaction_latency_ns"]),
                # 保序字段
                src_dest_order_id=src_dest_order_id,
                packet_category=packet_category,
                # cycle数据字段 (向后兼容)
                cmd_entry_cake0_cycle=int(row.get("cmd_entry_cake0_cycle", -1)),
                cmd_entry_noc_from_cake0_cycle=int(row.get("cmd_entry_noc_from_cake0_cycle", -1)),
                cmd_entry_noc_from_cake1_cycle=int(row.get("cmd_entry_noc_from_cake1_cycle", -1)),
                cmd_received_by_cake0_cycle=int(row.get("cmd_received_by_cake0_cycle", -1)),
                cmd_received_by_cake1_cycle=int(row.get("cmd_received_by_cake1_cycle", -1)),
                data_entry_noc_from_cake0_cycle=int(row.get("data_entry_noc_from_cake0_cycle", -1)),
                data_entry_noc_from_cake1_cycle=int(row.get("data_entry_noc_from_cake1_cycle", -1)),
                data_received_complete_cycle=int(row.get("data_received_complete_cycle", -1)),
                rsp_entry_network_cycle=int(row.get("rsp_entry_network_cycle", -1)),
            )

            # 验证请求完整性
            if not DataValidator.validate_request(request_info):
                continue

            self.requests.append(request_info)

    def collect_cross_die_requests(self, dies: Dict) -> List[D2DRequestInfo]:
        """
        从多个Die的网络中收集跨Die请求数据

        Args:
            dies: Dict[die_id, die_model] - Die模型字典

        Returns:
            D2DRequestInfo列表
        """
        self.d2d_requests.clear()

        for die_id, die_model in dies.items():
            # 检查数据网络中的arrive_flits（读数据返回）
            if hasattr(die_model, "data_network") and hasattr(die_model.data_network, "arrive_flits"):
                self._collect_requests_from_network(die_model.data_network, die_id)

            # 检查响应网络中的arrive_flits（写完成响应）
            if hasattr(die_model, "rsp_network") and hasattr(die_model.rsp_network, "arrive_flits"):
                self._collect_requests_from_network(die_model.rsp_network, die_id)

        return self.d2d_requests

    def _collect_requests_from_network(self, network, die_id: int):
        """从单个网络中收集跨Die请求"""
        for packet_id, flits in network.arrive_flits.items():
            if not flits:
                continue

            # 使用flit_id选择first和representative flit，与基类保持一致
            first_flit = min(flits, key=lambda f: f.flit_id)
            representative_flit = max(flits, key=lambda f: f.flit_id)

            # 改进数据验证
            # 对于特殊响应（write_complete, datasend, negative等），不需要检查burst_length
            rsp_type = getattr(first_flit, "rsp_type", None)
            is_special_response = rsp_type in ["write_complete", "datasend", "negative", "positive"]

            if not is_special_response:
                if not hasattr(first_flit, "burst_length") or len(flits) != first_flit.burst_length:
                    continue

            # 检查是否为D2D请求（包括Die内和跨Die）
            if not self._is_d2d_request(first_flit):
                continue

            # 只记录请求发起方Die的数据，避免重复记录
            if hasattr(first_flit, "d2d_origin_die") and first_flit.d2d_origin_die != die_id:
                continue

            # 提取D2D信息
            d2d_info = self._extract_d2d_info(first_flit, representative_flit, packet_id)
            if d2d_info:
                self.d2d_requests.append(d2d_info)

    def _is_d2d_request(self, flit) -> bool:
        """
        检查flit是否为D2D请求

        Args:
            flit: Flit对象

        Returns:
            bool: 是否为D2D请求
        """
        has_origin = hasattr(flit, "d2d_origin_die")
        has_target = hasattr(flit, "d2d_target_die")
        origin_not_none = has_origin and flit.d2d_origin_die is not None
        target_not_none = has_target and flit.d2d_target_die is not None

        result = has_origin and has_target and origin_not_none and target_not_none

        return result

    def _get_time_value(self, flit, attr_name: str, default: float = 0) -> float:
        """获取flit的时间值并转换为ns"""
        if attr_name and hasattr(flit, attr_name):
            value = getattr(flit, attr_name)
            if value < float("inf"):
                return value // self.network_frequency
        return default

    def _get_latency_value(self, flit, attr_name: str) -> int:
        """获取flit的延迟值并转换为ns"""
        if hasattr(flit, attr_name):
            value = getattr(flit, attr_name)
            if value < float("inf"):
                return int(value // self.network_frequency)
        return 0

    def _extract_d2d_info(self, first_flit, representative_flit, packet_id: int) -> Optional[D2DRequestInfo]:
        """
        从flit中提取D2D请求信息

        Args:
            first_flit: 第一个flit
            representative_flit: 代表性flit（通常是最后一个）
            packet_id: 包ID

        Returns:
            D2DRequestInfo对象或None
        """
        try:
            # 计算开始时间和结束时间
            start_time_ns = self._get_time_value(representative_flit, "cmd_entry_cake0_cycle", 0)

            req_type = getattr(representative_flit, "req_type", "unknown")
            end_time_field = {"read": "data_received_complete_cycle", "write": "write_complete_received_cycle"}.get(req_type)
            end_time_ns = self._get_time_value(representative_flit, end_time_field, start_time_ns) if end_time_field else start_time_ns

            # 从flit读取已计算的延迟值并转换为ns
            cmd_latency_ns = self._get_latency_value(first_flit, "cmd_latency")
            data_latency_ns = self._get_latency_value(first_flit, "data_latency")
            transaction_latency_ns = self._get_latency_value(first_flit, "transaction_latency")

            # 计算数据量
            burst_length = getattr(first_flit, "burst_length", 1)
            data_bytes = burst_length * FLIT_SIZE_BYTES

            d2d_sn_node = getattr(first_flit, "d2d_sn_node", None)
            d2d_rn_node = getattr(first_flit, "d2d_rn_node", None)

            return D2DRequestInfo(
                packet_id=packet_id,
                source_die=getattr(first_flit, "d2d_origin_die", 0),
                target_die=getattr(first_flit, "d2d_target_die", 1),
                source_node=getattr(first_flit, "d2d_origin_node", 0),
                target_node=getattr(first_flit, "d2d_target_node", 0),
                source_type=getattr(first_flit, "d2d_origin_type", ""),
                target_type=getattr(first_flit, "d2d_target_type", ""),
                req_type=getattr(first_flit, "req_type", "unknown"),
                burst_length=burst_length,
                data_bytes=data_bytes,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                cmd_latency_ns=cmd_latency_ns,
                data_latency_ns=data_latency_ns,
                transaction_latency_ns=transaction_latency_ns,
                d2d_sn_node=d2d_sn_node,
                d2d_rn_node=d2d_rn_node,
            )
        except (AttributeError, KeyError, ValueError) as e:
            return None
        except Exception as e:
            raise


class LatencyStatsCollector:
    """延迟统计收集器 - 计算CMD、Data、Transaction延迟统计"""

    @staticmethod
    def _init_latency_stats_structure() -> Dict:
        """初始化延迟统计数据结构"""
        return {
            "cmd": {"read": {"sum": 0, "max": 0, "count": 0, "values": []}, "write": {"sum": 0, "max": 0, "count": 0, "values": []}, "mixed": {"sum": 0, "max": 0, "count": 0, "values": []}},
            "data": {"read": {"sum": 0, "max": 0, "count": 0, "values": []}, "write": {"sum": 0, "max": 0, "count": 0, "values": []}, "mixed": {"sum": 0, "max": 0, "count": 0, "values": []}},
            "trans": {"read": {"sum": 0, "max": 0, "count": 0, "values": []}, "write": {"sum": 0, "max": 0, "count": 0, "values": []}, "mixed": {"sum": 0, "max": 0, "count": 0, "values": []}},
        }

    @staticmethod
    def _update_latency_stats(stats: Dict, category: str, req_type: str, latency_value: float):
        """更新延迟统计数据"""
        if not math.isfinite(latency_value):
            return

        group = stats[category][req_type]
        group["sum"] += latency_value
        group["count"] += 1
        group["max"] = max(group["max"], latency_value)
        group["values"].append(latency_value)

        mixed = stats[category]["mixed"]
        mixed["sum"] += latency_value
        mixed["count"] += 1
        mixed["max"] = max(mixed["max"], latency_value)
        mixed["values"].append(latency_value)

    @staticmethod
    def _finalize_latency_stats(stats: Dict) -> Dict:
        """计算百分位数并清理临时数据"""
        for category in ["cmd", "data", "trans"]:
            for req_type in ["read", "write", "mixed"]:
                values = stats[category][req_type]["values"]
                if len(values) > 0:
                    stats[category][req_type]["p95"] = np.percentile(values, 95)
                    stats[category][req_type]["p99"] = np.percentile(values, 99)
                else:
                    stats[category][req_type]["p95"] = 0.0
                    stats[category][req_type]["p99"] = 0.0
                del stats[category][req_type]["values"]
        return stats

    def calculate_latency_stats(self, requests: List[RequestInfo]) -> Dict:
        """
        计算并返回延迟统计数据字典，过滤掉无穷大

        Args:
            requests: RequestInfo列表

        Returns:
            延迟统计字典，包含read/write/mixed三种类型的cmd/data/trans延迟统计
        """
        stats = self._init_latency_stats_structure()

        for r in requests:
            self._update_latency_stats(stats, "cmd", r.req_type, r.cmd_latency)
            self._update_latency_stats(stats, "data", r.req_type, r.data_latency)
            self._update_latency_stats(stats, "trans", r.req_type, r.transaction_latency)

        return self._finalize_latency_stats(stats)

    def calculate_d2d_latency_stats(self, d2d_requests: List[D2DRequestInfo]) -> Dict:
        """
        计算D2D延迟统计数据

        Args:
            d2d_requests: D2DRequestInfo列表

        Returns:
            D2D延迟统计字典
        """
        stats = self._init_latency_stats_structure()

        # 定义延迟字段映射
        latency_fields = [("cmd", "cmd_latency_ns"), ("data", "data_latency_ns"), ("trans", "transaction_latency_ns")]

        for req in d2d_requests:
            for category, field_name in latency_fields:
                latency_ns = getattr(req, field_name, float("inf"))
                self._update_latency_stats(stats, category, req.req_type, latency_ns)

        return self._finalize_latency_stats(stats)


class CircuitStatsCollector:
    """绕环统计收集器 - 统计数据flit绕环次数和保序阻塞"""

    def calculate_circling_eject_stats(self, requests: List[RequestInfo]) -> Dict:
        """
        计算数据flit绕环下环统计

        Args:
            requests: RequestInfo列表

        Returns:
            绕环统计字典，包含横向、纵向和总体的绕环比例
        """
        # 初始化计数器
        total_data_flits_h = 0
        total_data_flits_v = 0
        circling_flits_h = 0  # 水平方向绕环 (attempt > 1)
        circling_flits_v = 0  # 垂直方向绕环 (attempt > 1)

        # 遍历所有请求，收集数据flit的尝试次数
        for req in requests:
            # 处理水平方向尝试次数
            for attempt in req.data_eject_attempts_h_list:
                total_data_flits_h += 1
                if attempt > 1:
                    circling_flits_h += 1

            # 处理垂直方向尝试次数
            for attempt in req.data_eject_attempts_v_list:
                total_data_flits_v += 1
                if attempt > 1:
                    circling_flits_v += 1

        # 计算比例
        circling_ratio_h = circling_flits_h / total_data_flits_h if total_data_flits_h > 0 else 0.0
        circling_ratio_v = circling_flits_v / total_data_flits_v if total_data_flits_v > 0 else 0.0

        # 准备结果字典
        results = {
            "horizontal": {
                "total_data_flits": total_data_flits_h,
                "circling_flits": circling_flits_h,  # 绕环次数大于1的flit数量
                "circling_ratio": circling_ratio_h,  # 绕环比例
            },
            "vertical": {
                "total_data_flits": total_data_flits_v,
                "circling_flits": circling_flits_v,  # 绕环次数大于1的flit数量
                "circling_ratio": circling_ratio_v,  # 绕环比例
            },
            # 整体统计
            "overall": {
                "total_data_flits": total_data_flits_h + total_data_flits_v,
                "circling_flits": circling_flits_h + circling_flits_v,
                "circling_ratio": (circling_flits_h + circling_flits_v) / (total_data_flits_h + total_data_flits_v) if (total_data_flits_h + total_data_flits_v) > 0 else 0.0,
            },
        }

        return results

    def calculate_ordering_blocked_stats(self, requests: List[RequestInfo]) -> Dict:
        """
        计算保序阻塞统计

        Args:
            requests: RequestInfo列表

        Returns:
            保序阻塞统计字典
        """
        # 初始化计数器
        total_data_flits_h = 0
        total_data_flits_v = 0
        ordering_blocked_flits_h = 0  # 水平方向因保序被阻止的flit数
        ordering_blocked_flits_v = 0  # 垂直方向因保序被阻止的flit数

        # 遍历所有请求，收集数据flit的保序阻止次数
        for req in requests:
            # 处理水平方向
            for i, blocked_count in enumerate(req.data_ordering_blocked_h_list):
                total_data_flits_h += 1
                if blocked_count > 0:
                    ordering_blocked_flits_h += 1

            # 处理垂直方向
            for i, blocked_count in enumerate(req.data_ordering_blocked_v_list):
                total_data_flits_v += 1
                if blocked_count > 0:
                    ordering_blocked_flits_v += 1

        # 计算比例
        ordering_blocked_ratio_h = ordering_blocked_flits_h / total_data_flits_h if total_data_flits_h > 0 else 0.0
        ordering_blocked_ratio_v = ordering_blocked_flits_v / total_data_flits_v if total_data_flits_v > 0 else 0.0

        # 准备结果字典
        results = {
            "horizontal": {
                "total_data_flits": total_data_flits_h,
                "ordering_blocked_flits": ordering_blocked_flits_h,
                "ordering_blocked_ratio": ordering_blocked_ratio_h,
            },
            "vertical": {
                "total_data_flits": total_data_flits_v,
                "ordering_blocked_flits": ordering_blocked_flits_v,
                "ordering_blocked_ratio": ordering_blocked_ratio_v,
            },
            # 整体统计
            "overall": {
                "total_data_flits": total_data_flits_h + total_data_flits_v,
                "ordering_blocked_flits": ordering_blocked_flits_h + ordering_blocked_flits_v,
                "ordering_blocked_ratio": (ordering_blocked_flits_h + ordering_blocked_flits_v) / (total_data_flits_h + total_data_flits_v) if (total_data_flits_h + total_data_flits_v) > 0 else 0.0,
            },
        }

        return results

    def process_fifo_usage_statistics(self, model) -> Dict:
        """
        处理FIFO使用率统计

        Args:
            model: 仿真模型对象

        Returns:
            FIFO统计字典
        """
        networks = {"req": model.req_network, "rsp": model.rsp_network, "data": model.data_network}

        total_cycles = model.cycle  # 使用总周期数
        results = {}

        # 获取配置对象
        config = model.config if hasattr(model, "config") else None
        if config is None:
            raise ValueError("模型对象缺少config属性")

        for net_name, network in networks.items():
            results[net_name] = {}

            # 获取FIFO容量配置
            capacities = {
                "IQ": {
                    "CH_buffer": config.IQ_CH_FIFO_DEPTH,
                    "TR": config.IQ_OUT_FIFO_DEPTH_HORIZONTAL,
                    "TL": config.IQ_OUT_FIFO_DEPTH_HORIZONTAL,
                    "TU": config.IQ_OUT_FIFO_DEPTH_VERTICAL,
                    "TD": config.IQ_OUT_FIFO_DEPTH_VERTICAL,
                    "EQ": config.IQ_OUT_FIFO_DEPTH_EQ,
                },
                "RB": {
                    "TR": config.RB_IN_FIFO_DEPTH,
                    "TL": config.RB_IN_FIFO_DEPTH,
                    "TU": config.RB_OUT_FIFO_DEPTH,
                    "TD": config.RB_OUT_FIFO_DEPTH,
                    "EQ": config.RB_OUT_FIFO_DEPTH,
                },
                "EQ": {"TU": config.EQ_IN_FIFO_DEPTH, "TD": config.EQ_IN_FIFO_DEPTH, "CH_buffer": config.EQ_CH_FIFO_DEPTH},
            }

            # 计算平均深度和使用率
            for category in network.fifo_depth_sum:
                results[net_name][category] = {}
                for fifo_type in network.fifo_depth_sum[category]:
                    results[net_name][category][fifo_type] = {}

                    if fifo_type == "CH_buffer":
                        # CH_buffer需要特殊处理，因为它按ip_type分组
                        for pos, ip_types_data in network.fifo_depth_sum[category][fifo_type].items():
                            if isinstance(ip_types_data, dict):
                                for ip_type, sum_depth in ip_types_data.items():
                                    max_depth = network.fifo_max_depth[category][fifo_type][pos][ip_type]
                                    capacity = capacities[category][fifo_type]
                                    flit_count = 0
                                    if pos in network.fifo_flit_count[category][fifo_type]:
                                        if ip_type in network.fifo_flit_count[category][fifo_type][pos]:
                                            flit_count = network.fifo_flit_count[category][fifo_type][pos][ip_type]

                                    key = f"{pos}_{ip_type}"
                                    results[net_name][category][fifo_type][key] = self._calculate_fifo_stats(sum_depth, max_depth, capacity, flit_count, total_cycles)
                    else:
                        # 其他FIFO类型
                        for pos, sum_depth in network.fifo_depth_sum[category][fifo_type].items():
                            max_depth = network.fifo_max_depth[category][fifo_type][pos]
                            capacity = capacities[category][fifo_type]
                            flit_count = 0
                            if pos in network.fifo_flit_count[category][fifo_type]:
                                flit_count = network.fifo_flit_count[category][fifo_type][pos]

                            results[net_name][category][fifo_type][pos] = self._calculate_fifo_stats(sum_depth, max_depth, capacity, flit_count, total_cycles)

        return results

    @staticmethod
    def _calculate_fifo_stats(sum_depth: float, max_depth: int, capacity: int, flit_count: int, total_cycles: int) -> Dict:
        """计算单个FIFO的统计数据"""
        avg_depth = sum_depth / total_cycles
        return {
            "avg_depth": avg_depth,
            "max_depth": max_depth,
            "avg_utilization": avg_depth / capacity * 100,
            "max_utilization": max_depth / capacity * 100,
            "flit_count": flit_count,
            "avg_throughput": flit_count / total_cycles if total_cycles > 0 else 0,
        }

    def generate_fifo_usage_csv(self, model, output_path: str = None):
        """生成FIFO使用率CSV文件（仅统计data通道）"""
        if output_path is None:
            # 使用模型的结果保存路径或当前目录
            if hasattr(model, "result_save_path") and model.result_save_path:
                output_dir = os.path.dirname(model.result_save_path)
                output_path = os.path.join(output_dir, "fifo_usage_statistics.csv")
            else:
                output_path = "fifo_usage_statistics.csv"

        # 获取FIFO使用率统计
        fifo_stats = self.process_fifo_usage_statistics(model)

        # 准备CSV数据
        rows = []
        headers = ["网络", "类别", "FIFO类型", "位置", "平均使用率(%)", "最大使用率(%)", "平均深度", "最大深度", "累计flit数", "平均吞吐量(flit/cycle)"]

        # 只处理data通道的数据
        if "data" in fifo_stats:
            net_data = fifo_stats["data"]
            for category, category_data in net_data.items():
                for fifo_type, fifo_data in category_data.items():
                    for pos, stats in fifo_data.items():
                        row = {
                            "网络": "data",
                            "类别": category,
                            "FIFO类型": fifo_type,
                            "位置": pos,
                            "平均使用率(%)": f"{stats['avg_utilization']:.2f}",
                            "最大使用率(%)": f"{stats['max_utilization']:.2f}",
                            "平均深度": f"{stats['avg_depth']:.2f}",
                            "最大深度": stats["max_depth"],
                            "累计flit数": stats.get("flit_count", 0),
                            "平均吞吐量(flit/cycle)": f"{stats.get('avg_throughput', 0):.4f}",
                        }
                        rows.append(row)

        # 写入CSV文件（使用UTF-8 with BOM编码，防止Excel打开乱码）
        with open(output_path, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
