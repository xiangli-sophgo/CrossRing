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
from src.utils.flit import Flit, get_original_source_type, get_original_destination_type


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
        从RequestTracker收集请求数据（完全重构版本）

        Args:
            sim_model: 仿真模型对象
            simulation_end_cycle: 仿真结束周期

        Returns:
            RequestInfo列表（仅包含已完成的请求）
        """
        self.requests.clear()

        # 检查是否有request_tracker
        if not hasattr(sim_model, 'request_tracker') or not sim_model.request_tracker:
            raise RuntimeError("sim_model没有request_tracker属性，请确保BaseModel已正确初始化")

        request_tracker = sim_model.request_tracker

        # 仅处理已完成的请求
        completed_requests = request_tracker.get_completed_requests()

        for packet_id, lifecycle in completed_requests.items():
            # 从lifecycle获取所有时间戳
            timestamps = lifecycle.timestamps
            if not timestamps:
                # 如果没有预收集，现在收集
                timestamps = request_tracker.collect_timestamps_from_flits(packet_id)

            # 基本信息来自lifecycle
            req_type = lifecycle.op_type

            # 确定end_time（根据是否跨Die选择时间戳）
            if req_type == "read":
                end_time_cycle = timestamps.get('data_received_complete_cycle', float('inf'))
            elif req_type == "write":
                # 根据是否跨Die选择时间戳
                if lifecycle.origin_die != lifecycle.target_die:  # 跨Die（D2D请求）
                    end_time_cycle = timestamps.get('write_complete_received_cycle', float('inf'))
                else:  # 不跨Die（NoC请求）
                    end_time_cycle = timestamps.get('data_received_complete_cycle', float('inf'))
            else:
                continue

            # 验证必需时间戳
            start_time_cycle = timestamps.get('cmd_entry_cake0_cycle', float('inf'))
            if start_time_cycle >= float('inf') or end_time_cycle >= float('inf'):
                continue  # 跳过无效请求

            # 转换为ns
            start_time = start_time_cycle // self.network_frequency
            end_time = end_time_cycle // self.network_frequency

            # 源目标节点信息来自lifecycle
            actual_source_node = lifecycle.source
            actual_dest_node = lifecycle.destination
            actual_source_type = lifecycle.source_type
            actual_dest_type = lifecycle.dest_type

            # 从data_flits收集下环统计
            data_eject_attempts_h_list = [f.eject_attempts_h for f in lifecycle.data_flits]
            data_eject_attempts_v_list = [f.eject_attempts_v for f in lifecycle.data_flits]
            data_ordering_blocked_h_list = [f.ordering_blocked_eject_h for f in lifecycle.data_flits]
            data_ordering_blocked_v_list = [f.ordering_blocked_eject_v for f in lifecycle.data_flits]

            # 保序信息（从第一个flit获取）
            first_flit = lifecycle.data_flits[0] if lifecycle.data_flits else None
            src_dest_order_id = getattr(first_flit, "src_dest_order_id", -1) if first_flit else -1
            packet_category = getattr(first_flit, "packet_category", "") if first_flit else ""

            # 计算延迟
            cmd_latency, data_latency, transaction_latency = self._calculate_latencies(lifecycle, timestamps)

            request_info = RequestInfo(
                packet_id=packet_id,
                start_time=start_time,
                end_time=end_time,
                req_type=req_type,
                source_node=actual_source_node,
                dest_node=actual_dest_node,
                source_type=actual_source_type,
                dest_type=actual_dest_type,
                burst_length=lifecycle.burst_size,
                total_bytes=lifecycle.burst_size * 128,
                cmd_latency=cmd_latency,
                data_latency=data_latency,
                transaction_latency=transaction_latency,
                # 保序相关字段
                src_dest_order_id=src_dest_order_id,
                packet_category=packet_category,
                # 所有cycle数据字段（从timestamps获取）
                cmd_entry_cake0_cycle=timestamps.get('cmd_entry_cake0_cycle', -1),
                cmd_entry_noc_from_cake0_cycle=timestamps.get('cmd_entry_noc_from_cake0_cycle', -1),
                cmd_entry_noc_from_cake1_cycle=timestamps.get('cmd_entry_noc_from_cake1_cycle', -1),
                cmd_received_by_cake0_cycle=timestamps.get('cmd_received_by_cake0_cycle', -1),
                cmd_received_by_cake1_cycle=timestamps.get('cmd_received_by_cake1_cycle', -1),
                data_entry_noc_from_cake0_cycle=timestamps.get('data_entry_noc_from_cake0_cycle', -1),
                data_entry_noc_from_cake1_cycle=timestamps.get('data_entry_noc_from_cake1_cycle', -1),
                data_received_complete_cycle=timestamps.get('data_received_complete_cycle', -1),
                rsp_entry_network_cycle=timestamps.get('rsp_entry_network_cycle', -1),
                # 数据flit的尝试下环次数列表
                data_eject_attempts_h_list=data_eject_attempts_h_list,
                data_eject_attempts_v_list=data_eject_attempts_v_list,
                # 数据flit因保序被阻止的下环次数列表
                data_ordering_blocked_h_list=data_ordering_blocked_h_list,
                data_ordering_blocked_v_list=data_ordering_blocked_v_list,
            )

            self.requests.append(request_info)

        # 按开始时间排序
        self.requests.sort(key=lambda x: x.start_time)

        return self.requests

    def _calculate_latencies(self, lifecycle, timestamps):
        """计算延迟指标"""

        # 判断是否跨Die
        source_die = getattr(lifecycle, 'source_die', 0)
        target_die = getattr(lifecycle, 'target_die', 0)
        is_cross_die = (source_die != target_die)

        if lifecycle.op_type == "read":
            # 读请求延迟计算
            cmd_entry = timestamps.get('cmd_entry_noc_from_cake0_cycle', float('inf'))

            if is_cross_die:  # D2D读请求
                cmd_received = timestamps.get('cmd_received_by_cake1_cycle', float('inf'))
                data_entry = timestamps.get('data_entry_noc_from_cake1_cycle', float('inf'))
            else:  # NoC读请求
                cmd_received = timestamps.get('cmd_received_by_cake0_cycle', float('inf'))
                data_entry = timestamps.get('data_entry_noc_from_cake0_cycle', float('inf'))

            data_received = timestamps.get('data_received_complete_cycle', float('inf'))

            if cmd_entry < float('inf') and cmd_received < float('inf'):
                cmd_latency = int((cmd_received - cmd_entry) / self.network_frequency)
            else:
                cmd_latency = -1

            if data_entry < float('inf') and data_received < float('inf'):
                data_latency = int((data_received - data_entry) / self.network_frequency)
            else:
                data_latency = -1
        else:  # write
            # 写请求延迟计算
            cmd_entry = timestamps.get('cmd_entry_noc_from_cake0_cycle', float('inf'))
            cmd_received = timestamps.get('cmd_received_by_cake0_cycle', float('inf'))
            data_entry = timestamps.get('data_entry_noc_from_cake0_cycle', float('inf'))
            data_received = timestamps.get('data_received_complete_cycle', float('inf'))

            if cmd_entry < float('inf') and cmd_received < float('inf'):
                cmd_latency = int((cmd_received - cmd_entry) / self.network_frequency)
            else:
                cmd_latency = -1

            if data_entry < float('inf') and data_received < float('inf'):
                data_latency = int((data_received - data_entry) / self.network_frequency)
            else:
                data_latency = -1

        # 事务延迟
        start_cycle = timestamps.get('cmd_entry_cake0_cycle', float('inf'))
        end_cycle = lifecycle.completed_cycle
        if start_cycle < float('inf') and end_cycle < float('inf'):
            transaction_latency = int((end_cycle - start_cycle) / self.network_frequency)
        else:
            transaction_latency = -1

        return cmd_latency, data_latency, transaction_latency

    def _calculate_d2d_latencies(self, lifecycle, timestamps):
        """计算D2D请求的延迟指标"""

        if lifecycle.op_type == "read":
            # 读请求延迟计算
            cmd_entry = timestamps.get('cmd_entry_noc_from_cake0_cycle', float('inf'))
            cmd_received = timestamps.get('cmd_received_by_cake1_cycle', float('inf'))
            data_entry = timestamps.get('data_entry_noc_from_cake1_cycle', float('inf'))
            data_received = timestamps.get('data_received_complete_cycle', float('inf'))

            if cmd_entry < float('inf') and cmd_received < float('inf'):
                cmd_latency = int((cmd_received - cmd_entry) / self.network_frequency)
            else:
                cmd_latency = -1

            if data_entry < float('inf') and data_received < float('inf'):
                data_latency = int((data_received - data_entry) / self.network_frequency)
            else:
                data_latency = -1
        else:  # write
            # 写请求延迟计算
            cmd_entry = timestamps.get('cmd_entry_noc_from_cake0_cycle', float('inf'))
            cmd_received = timestamps.get('cmd_received_by_cake0_cycle', float('inf'))
            data_entry = timestamps.get('data_entry_noc_from_cake0_cycle', float('inf'))
            data_received = timestamps.get('data_received_complete_cycle', float('inf'))

            if cmd_entry < float('inf') and cmd_received < float('inf'):
                cmd_latency = int((cmd_received - cmd_entry) / self.network_frequency)
            else:
                cmd_latency = -1

            if data_entry < float('inf') and data_received < float('inf'):
                data_latency = int((data_received - data_entry) / self.network_frequency)
            else:
                data_latency = -1

        # 事务延迟
        start_cycle = timestamps.get('cmd_entry_cake0_cycle', float('inf'))
        end_cycle = lifecycle.completed_cycle

        if start_cycle < float('inf') and end_cycle < float('inf'):
            transaction_latency = int((end_cycle - start_cycle) / self.network_frequency)
        else:
            transaction_latency = -1

        return cmd_latency, data_latency, transaction_latency

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

        # 按开始时间排序
        self.requests.sort(key=lambda x: x.start_time)

        return self.requests

    def collect_cross_die_requests(self, dies: Dict) -> List[D2DRequestInfo]:
        """
        从多个Die的网络中收集跨Die请求数据（基于RequestTracker重构版本）

        Args:
            dies: Dict[die_id, die_model] - Die模型字典

        Returns:
            D2DRequestInfo列表
        """
        self.d2d_requests.clear()

        # 找到全局RequestTracker（应该在第一个Die中）
        request_tracker = None
        for die_id, die_model in dies.items():
            if hasattr(die_model, 'request_tracker') and die_model.request_tracker:
                request_tracker = die_model.request_tracker
                break

        if not request_tracker:
            print("[WARNING] collect_cross_die_requests: 未找到RequestTracker，使用旧方法从arrive_flits收集")
            # 回退到旧方法
            for die_id, die_model in dies.items():
                if hasattr(die_model, "data_network") and hasattr(die_model.data_network, "arrive_flits"):
                    self._collect_requests_from_network(die_model.data_network, die_id)
                if hasattr(die_model, "rsp_network") and hasattr(die_model.rsp_network, "arrive_flits"):
                    self._collect_requests_from_network(die_model.rsp_network, die_id)
            return self.d2d_requests

        # 从RequestTracker收集已完成的请求
        completed_requests = request_tracker.get_completed_requests()

        for packet_id, lifecycle in completed_requests.items():
            # 收集所有请求（包括本地请求和跨Die请求）

            # 收集时间戳
            timestamps = lifecycle.timestamps
            if not timestamps:
                timestamps = request_tracker.collect_timestamps_from_flits(packet_id)

            # 计算延迟
            cmd_latency, data_latency, transaction_latency = self._calculate_d2d_latencies(lifecycle, timestamps)

            # 计算时间
            start_time_ns = int(lifecycle.created_cycle / self.network_frequency)
            end_time_ns = int(lifecycle.completed_cycle / self.network_frequency)

            # 从flit获取D2D节点信息
            first_flit = None
            if lifecycle.request_flits:
                first_flit = lifecycle.request_flits[0]
            elif lifecycle.response_flits:
                first_flit = lifecycle.response_flits[0]
            elif lifecycle.data_flits:
                first_flit = lifecycle.data_flits[0]

            d2d_sn_node = getattr(first_flit, "d2d_sn_node", None) if first_flit else None
            d2d_rn_node = getattr(first_flit, "d2d_rn_node", None) if first_flit else None

            d2d_info = D2DRequestInfo(
                packet_id=packet_id,
                source_die=lifecycle.origin_die,
                target_die=lifecycle.target_die,
                source_node=lifecycle.source,
                target_node=lifecycle.destination,
                source_type=lifecycle.source_type,
                target_type=lifecycle.dest_type,
                req_type=lifecycle.op_type,
                burst_length=lifecycle.burst_size,
                data_bytes=lifecycle.burst_size * 128,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                cmd_latency_ns=cmd_latency,
                data_latency_ns=data_latency,
                transaction_latency_ns=transaction_latency,
                d2d_sn_node=d2d_sn_node,
                d2d_rn_node=d2d_rn_node,
            )
            self.d2d_requests.append(d2d_info)

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

    def _get_time_value(self, flit, attr_name: str, default: float = 0, allow_default: bool = True) -> float:
        """
        获取flit的时间值并转换为ns

        Args:
            flit: flit对象
            attr_name: 属性名称
            default: 默认值
            allow_default: 是否允许使用默认值，False时如果字段无效会报错
        """
        if attr_name and hasattr(flit, attr_name):
            value = getattr(flit, attr_name)
            if value < float("inf"):
                return value // self.network_frequency

        if not allow_default:
            raise ValueError(
                f"[ERROR] Required timestamp field '{attr_name}' is invalid or missing!\n"
                f"  Flit has attribute: {hasattr(flit, attr_name)}\n"
                f"  Value: {getattr(flit, attr_name, 'N/A')}\n"
                f"  Flit packet_id: {getattr(flit, 'packet_id', 'N/A')}\n"
                f"  Flit req_type: {getattr(flit, 'req_type', 'N/A')}"
            )
        return default

    def _get_latency_value(self, flit, attr_name: str) -> int:
        """获取flit的延迟值并转换为ns"""
        if hasattr(flit, attr_name):
            value = getattr(flit, attr_name)
            if value < float("inf"):
                return int(value / self.network_frequency)
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
            start_time_ns = self._get_time_value(representative_flit, "cmd_entry_cake0_cycle", 0, allow_default=False)

            req_type = getattr(representative_flit, "req_type", "unknown")
            end_time_field = {"read": "data_received_complete_cycle", "write": "write_complete_received_cycle"}.get(req_type)

            # 对于end_time字段，不允许使用默认值，必须有有效值
            if not end_time_field:
                raise ValueError(f"Unknown req_type: {req_type} for packet {packet_id}")

            end_time_ns = self._get_time_value(representative_flit, end_time_field, 0, allow_default=False)

            # 从flit读取已计算的延迟值并转换为ns
            cmd_latency_ns = self._get_latency_value(first_flit, "cmd_latency")
            data_latency_ns = self._get_latency_value(first_flit, "data_latency")
            transaction_latency_ns = self._get_latency_value(first_flit, "transaction_latency")

            # 计算数据量
            burst_length = getattr(first_flit, "burst_length", 1)
            data_bytes = burst_length * FLIT_SIZE_BYTES

            d2d_sn_node = getattr(first_flit, "d2d_sn_node", None)
            d2d_rn_node = getattr(first_flit, "d2d_rn_node", None)

            source_die = getattr(first_flit, "d2d_origin_die", 0)
            target_die = getattr(first_flit, "d2d_target_die", 1)

            return D2DRequestInfo(
                packet_id=packet_id,
                source_die=source_die,
                target_die=target_die,
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
    def _finalize_latency_stats(stats: Dict, keep_raw_values: bool = True) -> Dict:
        """计算百分位数并清理临时数据

        Args:
            stats: 延迟统计字典
            keep_raw_values: 是否保留原始延迟值列表,默认为True以支持分布图绘制
        """
        for category in ["cmd", "data", "trans"]:
            for req_type in ["read", "write", "mixed"]:
                values = stats[category][req_type]["values"]
                if len(values) > 0:
                    stats[category][req_type]["p95"] = np.percentile(values, 95)
                    stats[category][req_type]["p99"] = np.percentile(values, 99)
                else:
                    stats[category][req_type]["p95"] = 0.0
                    stats[category][req_type]["p99"] = 0.0
                if not keep_raw_values:
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
                                    base_stats = self._calculate_fifo_stats(sum_depth, max_depth, capacity, flit_count, total_cycles)
                                    # CH_buffer不统计Tag，补充空字段
                                    results[net_name][category][fifo_type][key] = {
                                        **base_stats,
                                        "itag_cumulative_count": 0,
                                        "itag_rate": 0.0,
                                        "etag_t0_cumulative": 0,
                                        "etag_t1_cumulative": 0,
                                        "etag_t2_cumulative": 0,
                                        "etag_t0_rate": 0.0,
                                        "etag_t1_rate": 0.0,
                                        "etag_t2_rate": 0.0,
                                    }
                    else:
                        # 其他FIFO类型
                        for pos, sum_depth in network.fifo_depth_sum[category][fifo_type].items():
                            max_depth = network.fifo_max_depth[category][fifo_type][pos]
                            capacity = capacities[category][fifo_type]
                            flit_count = 0
                            if pos in network.fifo_flit_count[category][fifo_type]:
                                flit_count = network.fifo_flit_count[category][fifo_type][pos]

                            # 基础统计
                            base_stats = self._calculate_fifo_stats(sum_depth, max_depth, capacity, flit_count, total_cycles)

                            # === ITag统计 ===
                            itag_cumulative = 0
                            itag_rate = 0.0
                            if category in network.fifo_itag_cumulative_count:
                                if fifo_type in network.fifo_itag_cumulative_count[category]:
                                    if pos in network.fifo_itag_cumulative_count[category][fifo_type]:
                                        itag_cumulative = network.fifo_itag_cumulative_count[category][fifo_type][pos]
                                        # ITag率 = ITag累计次数 / (总周期数 × 累计flit数) × 100%
                                        if flit_count > 0 and total_cycles > 0:
                                            itag_rate = itag_cumulative / (total_cycles * flit_count) * 100

                            # === ETag统计 ===
                            etag_t0_cumulative = 0
                            etag_t1_cumulative = 0
                            etag_t2_cumulative = 0
                            etag_t0_rate = 0.0
                            etag_t1_rate = 0.0
                            etag_t2_rate = 0.0

                            if category in network.fifo_etag_entry_count:
                                if fifo_type in network.fifo_etag_entry_count[category]:
                                    if pos in network.fifo_etag_entry_count[category][fifo_type]:
                                        etag_dist = network.fifo_etag_entry_count[category][fifo_type][pos]
                                        etag_t0_cumulative = etag_dist["T0"]
                                        etag_t1_cumulative = etag_dist["T1"]
                                        etag_t2_cumulative = etag_dist["T2"]

                                        total_etag = etag_t0_cumulative + etag_t1_cumulative + etag_t2_cumulative
                                        if total_etag > 0:
                                            etag_t0_rate = etag_t0_cumulative / total_etag * 100
                                            etag_t1_rate = etag_t1_cumulative / total_etag * 100
                                            etag_t2_rate = etag_t2_cumulative / total_etag * 100

                            # 合并所有统计
                            results[net_name][category][fifo_type][pos] = {
                                **base_stats,
                                "itag_cumulative_count": itag_cumulative,
                                "itag_rate": itag_rate,
                                "etag_t0_cumulative": etag_t0_cumulative,
                                "etag_t1_cumulative": etag_t1_cumulative,
                                "etag_t2_cumulative": etag_t2_cumulative,
                                "etag_t0_rate": etag_t0_rate,
                                "etag_t1_rate": etag_t1_rate,
                                "etag_t2_rate": etag_t2_rate,
                            }

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
        headers = [
            "网络", "类别", "FIFO类型", "位置",
            "平均使用率(%)", "最大使用率(%)", "平均深度", "最大深度",
            "累计flit数", "平均吞吐量(flit/cycle)",
            "ITag累计次数", "ITag率(%)",
            "ETag_T0累计次数", "ETag_T1累计次数", "ETag_T2累计次数",
            "ETag_T0率(%)", "ETag_T1率(%)", "ETag_T2率(%)"
        ]

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
                            "累计flit数": stats["flit_count"],
                            "平均吞吐量(flit/cycle)": f"{stats['avg_throughput']:.4f}",
                            # Tag字段
                            "ITag累计次数": stats["itag_cumulative_count"],
                            "ITag率(%)": f"{stats['itag_rate']:.2f}",
                            "ETag_T0累计次数": stats["etag_t0_cumulative"],
                            "ETag_T1累计次数": stats["etag_t1_cumulative"],
                            "ETag_T2累计次数": stats["etag_t2_cumulative"],
                            "ETag_T0率(%)": f"{stats['etag_t0_rate']:.2f}",
                            "ETag_T1率(%)": f"{stats['etag_t1_rate']:.2f}",
                            "ETag_T2率(%)": f"{stats['etag_t2_rate']:.2f}",
                        }
                        rows.append(row)

        # 写入CSV文件（使用UTF-8 with BOM编码，防止Excel打开乱码）
        with open(output_path, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        return output_path


class TrackerDataCollector:
    """Tracker使用情况数据收集器 - 从仿真模型收集IP的tracker使用数据"""

    def __init__(self):
        """初始化Tracker数据收集器"""
        self.tracker_data = {}

    def collect_tracker_data(self, sim_model) -> Dict:
        """
        从仿真模型收集所有IP的tracker使用数据

        Args:
            sim_model: 仿真模型对象

        Returns:
            tracker数据字典，格式：
            {
                die_id: {
                    ip_type: {
                        ip_pos: {
                            "usage_history": {...},
                            "block_events": [...],
                            "total_config": {...}
                        }
                    }
                }
            }
        """
        self.tracker_data.clear()

        # 获取die_id
        die_id = getattr(sim_model.config, "DIE_ID", 0)

        # 初始化die数据结构
        if die_id not in self.tracker_data:
            self.tracker_data[die_id] = {}

        # 遍历所有IP模块
        # ip_modules的格式是 {(ip_type, node_id): ip_module_instance}
        for (ip_type, ip_pos), ip_module in sim_model.ip_modules.items():
            if ip_type not in self.tracker_data[die_id]:
                self.tracker_data[die_id][ip_type] = {}

            # 获取IP的tracker使用数据
            tracker_usage_data = ip_module.get_tracker_usage_data()

            # 只保存有数据的IP
            if self._has_tracker_data(tracker_usage_data):
                self.tracker_data[die_id][ip_type][ip_pos] = tracker_usage_data

        return self.tracker_data

    def _has_tracker_data(self, tracker_usage_data: Dict) -> bool:
        """检查是否有有效的tracker数据"""
        events = tracker_usage_data.get("events", {})
        total_allocated = tracker_usage_data.get("total_allocated", {})
        block_events = tracker_usage_data.get("block_events", [])

        # 检查是否有任何事件
        for tracker_type, event_data in events.items():
            if len(event_data.get("allocations", [])) > 0:
                return True
            if len(event_data.get("releases", [])) > 0:
                return True

        # 检查总分配次数
        for count in total_allocated.values():
            if count > 0:
                return True

        # 检查是否有阻塞事件
        if len(block_events) > 0:
            return True

        return False

    def save_to_json(self, output_dir: str, filename: str = "tracker_data.json"):
        """
        将tracker数据保存为JSON文件

        Args:
            output_dir: 输出目录
            filename: 文件名

        Returns:
            保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        # 转换数据格式为可序列化的格式
        serializable_data = self._convert_to_serializable(self.tracker_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        return output_path

    def _convert_to_serializable(self, data: Dict) -> Dict:
        """将数据转换为JSON可序列化的格式（新格式：事件驱动）"""
        result = {}

        for die_id, die_data in data.items():
            result[str(die_id)] = {}
            for ip_type, ip_type_data in die_data.items():
                result[str(die_id)][ip_type] = {}
                for ip_pos, ip_data in ip_type_data.items():
                    # 新格式：直接使用events、total_allocated、block_events
                    result[str(die_id)][ip_type][str(ip_pos)] = {
                        "events": ip_data["events"],
                        "total_allocated": ip_data["total_allocated"],
                        "block_events": ip_data["block_events"],
                        "total_config": ip_data["total_config"]
                    }

        return result

    @staticmethod
    def load_from_json(file_path: str) -> Dict:
        """
        从JSON文件加载tracker数据

        Args:
            file_path: JSON文件路径

        Returns:
            tracker数据字典
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 转换回原始格式
        result = {}
        for die_id_str, die_data in data.items():
            die_id = int(die_id_str)
            result[die_id] = {}
            for ip_type, ip_type_data in die_data.items():
                result[die_id][ip_type] = {}
                for ip_pos_str, ip_data in ip_type_data.items():
                    ip_pos = int(ip_pos_str)

                    # 转换usage_history：从list of dicts转换为list of tuples
                    usage_history_converted = {}
                    for tracker_type, history in ip_data["usage_history"].items():
                        usage_history_converted[tracker_type] = [
                            (entry["cycle"], entry["available"], entry["total"])
                            for entry in history
                        ]

                    # 转换block_events：从list of dicts转换为list of tuples
                    block_events_converted = [
                        (event["cycle"], event["tracker_type"], event["reason"])
                        for event in ip_data["block_events"]
                    ]

                    result[die_id][ip_type][ip_pos] = {
                        "usage_history": usage_history_converted,
                        "block_events": block_events_converted,
                        "total_config": ip_data["total_config"]
                    }

        return result
