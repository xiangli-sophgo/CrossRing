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
from .analyzers import RequestInfo, D2DRequestInfo, FLIT_SIZE_BYTES
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

        # 处理读请求
        if os.path.exists(read_csv):
            df_read = pd.read_csv(read_csv)
            for _, row in df_read.iterrows():
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
                    req_type="read",
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

        # 处理写请求
        if os.path.exists(write_csv):
            df_write = pd.read_csv(write_csv)
            for _, row in df_write.iterrows():
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
                    req_type="write",
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

        return self.requests

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
            # 对于write_complete响应，只有单个flit，不需要检查burst_length
            is_write_complete = hasattr(first_flit, "rsp_type") and first_flit.rsp_type == "write_complete"
            if not is_write_complete:
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
        return (
            hasattr(flit, "d2d_origin_die") and
            hasattr(flit, "d2d_target_die") and
            flit.d2d_origin_die is not None and
            flit.d2d_target_die is not None
        )

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
            # 计算开始时间 - 使用cmd_entry_cake0_cycle（tracker消耗开始）
            if hasattr(representative_flit, "cmd_entry_cake0_cycle") and representative_flit.cmd_entry_cake0_cycle < float("inf"):
                start_time_ns = representative_flit.cmd_entry_cake0_cycle // self.network_frequency
            else:
                start_time_ns = 0

            # 计算结束时间 - 根据请求类型选择合适的时间戳
            req_type = getattr(representative_flit, "req_type", "unknown")
            if req_type == "read":
                # 读请求：使用data_received_complete_cycle
                if hasattr(representative_flit, "data_received_complete_cycle") and representative_flit.data_received_complete_cycle < float("inf"):
                    end_time_ns = representative_flit.data_received_complete_cycle // self.network_frequency
                else:
                    end_time_ns = start_time_ns
            elif req_type == "write":
                # 写请求：使用write_complete_received_cycle
                if hasattr(representative_flit, "write_complete_received_cycle") and representative_flit.write_complete_received_cycle < float("inf"):
                    end_time_ns = representative_flit.write_complete_received_cycle // self.network_frequency
                else:
                    end_time_ns = start_time_ns
            else:
                end_time_ns = start_time_ns

            # 从flit读取已计算的延迟值并转换为ns
            cmd_latency_ns = 0
            data_latency_ns = 0
            transaction_latency_ns = 0

            if hasattr(first_flit, "cmd_latency") and first_flit.cmd_latency < float("inf"):
                cmd_latency_ns = int(first_flit.cmd_latency // self.network_frequency)

            if hasattr(first_flit, "data_latency") and first_flit.data_latency < float("inf"):
                data_latency_ns = int(first_flit.data_latency // self.network_frequency)

            if hasattr(first_flit, "transaction_latency") and first_flit.transaction_latency < float("inf"):
                transaction_latency_ns = int(first_flit.transaction_latency // self.network_frequency)

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

    def __init__(self):
        """初始化延迟统计收集器"""
        pass

    def calculate_latency_stats(self, requests: List[RequestInfo]) -> Dict:
        """
        计算并返回延迟统计数据字典，过滤掉无穷大

        Args:
            requests: RequestInfo列表

        Returns:
            延迟统计字典，包含read/write/mixed三种类型的cmd/data/trans延迟统计
        """
        stats = {
            "cmd": {
                "read": {"sum": 0, "max": 0, "count": 0, "values": []},
                "write": {"sum": 0, "max": 0, "count": 0, "values": []},
                "mixed": {"sum": 0, "max": 0, "count": 0, "values": []}
            },
            "data": {
                "read": {"sum": 0, "max": 0, "count": 0, "values": []},
                "write": {"sum": 0, "max": 0, "count": 0, "values": []},
                "mixed": {"sum": 0, "max": 0, "count": 0, "values": []}
            },
            "trans": {
                "read": {"sum": 0, "max": 0, "count": 0, "values": []},
                "write": {"sum": 0, "max": 0, "count": 0, "values": []},
                "mixed": {"sum": 0, "max": 0, "count": 0, "values": []}
            },
        }

        for r in requests:
            # CMD延迟统计
            if math.isfinite(r.cmd_latency):
                group = stats["cmd"][r.req_type]
                group["sum"] += r.cmd_latency
                group["count"] += 1
                group["max"] = max(group["max"], r.cmd_latency)
                group["values"].append(r.cmd_latency)

                mixed = stats["cmd"]["mixed"]
                mixed["sum"] += r.cmd_latency
                mixed["count"] += 1
                mixed["max"] = max(mixed["max"], r.cmd_latency)
                mixed["values"].append(r.cmd_latency)

            # Data延迟统计
            if math.isfinite(r.data_latency):
                group = stats["data"][r.req_type]
                group["sum"] += r.data_latency
                group["count"] += 1
                group["max"] = max(group["max"], r.data_latency)
                group["values"].append(r.data_latency)

                mixed = stats["data"]["mixed"]
                mixed["sum"] += r.data_latency
                mixed["count"] += 1
                mixed["max"] = max(mixed["max"], r.data_latency)
                mixed["values"].append(r.data_latency)

            # Transaction延迟统计
            if math.isfinite(r.transaction_latency):
                group = stats["trans"][r.req_type]
                group["sum"] += r.transaction_latency
                group["count"] += 1
                group["max"] = max(group["max"], r.transaction_latency)
                group["values"].append(r.transaction_latency)

                mixed = stats["trans"]["mixed"]
                mixed["sum"] += r.transaction_latency
                mixed["count"] += 1
                mixed["max"] = max(mixed["max"], r.transaction_latency)
                mixed["values"].append(r.transaction_latency)

        # 计算百分位数
        for category in ["cmd", "data", "trans"]:
            for req_type in ["read", "write", "mixed"]:
                values = stats[category][req_type]["values"]
                if len(values) > 0:
                    stats[category][req_type]["p95"] = np.percentile(values, 95)
                    stats[category][req_type]["p99"] = np.percentile(values, 99)
                else:
                    stats[category][req_type]["p95"] = 0.0
                    stats[category][req_type]["p99"] = 0.0
                # 删除values列表以节省内存
                del stats[category][req_type]["values"]

        return stats

    def calculate_d2d_latency_stats(self, d2d_requests: List[D2DRequestInfo]) -> Dict:
        """
        计算D2D延迟统计数据

        Args:
            d2d_requests: D2DRequestInfo列表

        Returns:
            D2D延迟统计字典
        """
        stats = {
            "cmd": {
                "read": {"sum": 0, "max": 0, "count": 0, "values": []},
                "write": {"sum": 0, "max": 0, "count": 0, "values": []},
                "mixed": {"sum": 0, "max": 0, "count": 0, "values": []}
            },
            "data": {
                "read": {"sum": 0, "max": 0, "count": 0, "values": []},
                "write": {"sum": 0, "max": 0, "count": 0, "values": []},
                "mixed": {"sum": 0, "max": 0, "count": 0, "values": []}
            },
            "trans": {
                "read": {"sum": 0, "max": 0, "count": 0, "values": []},
                "write": {"sum": 0, "max": 0, "count": 0, "values": []},
                "mixed": {"sum": 0, "max": 0, "count": 0, "values": []}
            },
        }

        # 定义延迟字段映射
        latency_fields = [("cmd", "cmd_latency_ns"), ("data", "data_latency_ns"), ("trans", "transaction_latency_ns")]

        for req in d2d_requests:
            for category, field_name in latency_fields:
                latency_ns = getattr(req, field_name, float("inf"))

                if math.isfinite(latency_ns):
                    req_type = req.req_type
                    group = stats[category][req_type]
                    group["sum"] += latency_ns
                    group["count"] += 1
                    group["max"] = max(group["max"], latency_ns)
                    group["values"].append(latency_ns)

                    mixed = stats[category]["mixed"]
                    mixed["sum"] += latency_ns
                    mixed["count"] += 1
                    mixed["max"] = max(mixed["max"], latency_ns)
                    mixed["values"].append(latency_ns)

        # 计算百分位数
        for category in ["cmd", "data", "trans"]:
            for req_type in ["read", "write", "mixed"]:
                values = stats[category][req_type]["values"]
                if len(values) > 0:
                    stats[category][req_type]["p95"] = np.percentile(values, 95)
                    stats[category][req_type]["p99"] = np.percentile(values, 99)
                else:
                    stats[category][req_type]["p95"] = 0.0
                    stats[category][req_type]["p99"] = 0.0
                # 删除values列表以节省内存
                del stats[category][req_type]["values"]

        return stats


class CircuitStatsCollector:
    """绕环统计收集器 - 统计数据flit绕环次数和保序阻塞"""

    def __init__(self):
        """初始化绕环统计收集器"""
        pass

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
                                    avg_depth = sum_depth / total_cycles
                                    max_depth = network.fifo_max_depth[category][fifo_type][pos][ip_type]
                                    capacity = capacities[category][fifo_type]

                                    # 获取flit_count
                                    flit_count = 0
                                    if pos in network.fifo_flit_count[category][fifo_type]:
                                        if ip_type in network.fifo_flit_count[category][fifo_type][pos]:
                                            flit_count = network.fifo_flit_count[category][fifo_type][pos][ip_type]

                                    key = f"{pos}_{ip_type}"
                                    results[net_name][category][fifo_type][key] = {
                                        "avg_depth": avg_depth,
                                        "max_depth": max_depth,
                                        "avg_utilization": avg_depth / capacity * 100,
                                        "max_utilization": max_depth / capacity * 100,
                                        "flit_count": flit_count,
                                        "avg_throughput": flit_count / total_cycles if total_cycles > 0 else 0,
                                    }
                    else:
                        # 其他FIFO类型
                        for pos, sum_depth in network.fifo_depth_sum[category][fifo_type].items():
                            avg_depth = sum_depth / total_cycles
                            max_depth = network.fifo_max_depth[category][fifo_type][pos]
                            capacity = capacities[category][fifo_type]

                            # 获取flit_count
                            flit_count = 0
                            if pos in network.fifo_flit_count[category][fifo_type]:
                                flit_count = network.fifo_flit_count[category][fifo_type][pos]

                            results[net_name][category][fifo_type][pos] = {
                                "avg_depth": avg_depth,
                                "max_depth": max_depth,
                                "avg_utilization": avg_depth / capacity * 100,
                                "max_utilization": max_depth / capacity * 100,
                                "flit_count": flit_count,
                                "avg_throughput": flit_count / total_cycles if total_cycles > 0 else 0,
                            }

        return results

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
