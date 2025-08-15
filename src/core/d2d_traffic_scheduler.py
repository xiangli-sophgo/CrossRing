"""
D2D Traffic Scheduler - 支持跨Die流量格式
扩展原有的traffic_scheduler.py，支持D2D格式的traffic文件。

D2D Traffic格式：inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length
"""

import os
import logging
from typing import List, Dict, Tuple, Optional, Iterator
from collections import defaultdict, deque
from .traffic_scheduler import TrafficFileReader, TrafficScheduler


class D2DTrafficFileReader(TrafficFileReader):
    """D2D专用的Traffic文件读取器，支持die信息"""

    def _calculate_file_stats(self):
        """预先扫描文件计算总请求数和flit数"""
        if self._stats_calculated:
            return

        with open(self.abs_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 9:  # D2D格式需要9个字段
                    continue

                # D2D格式：inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length
                req_type, burst = parts[7], int(parts[8])
                if req_type.upper() == "R":
                    self.read_req += 1
                    self.read_flit += burst
                else:
                    self.write_req += 1
                    self.write_flit += burst

        self.total_req = self.read_req + self.write_req
        self.total_flit = self.read_flit + self.write_flit
        self._stats_calculated = True

    def _fill_buffer(self):
        """填充缓冲区 - D2D格式解析"""
        if self._eof or not self._file_handle:
            return

        count = 0
        lines_batch = []

        # Read in larger batches to reduce I/O overhead
        while count < self.buffer_size:
            if not lines_batch:
                # Read multiple lines at once
                batch_size = min(1000, self.buffer_size - count)
                for _ in range(batch_size):
                    line = self._file_handle.readline()
                    if not line:
                        self._eof = True
                        break
                    lines_batch.append(line)
                if not lines_batch:
                    break

            line = lines_batch.pop(0).strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:  # D2D格式需要9个字段
                continue

            # D2D格式解析：inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length
            try:
                inject_time = int(parts[0])
                src_die = int(parts[1])
                src_node = int(parts[2])
                src_ip = parts[3]
                dst_die = int(parts[4])
                dst_node = int(parts[5])
                dst_ip = parts[6]
                req_type = parts[7].upper()
                burst_length = int(parts[8])

                # 转换时间为网络周期数
                t = (inject_time + self.time_offset) * self.config.NETWORK_FREQUENCY

                # 创建D2D请求元组 - 扩展格式包含die信息
                req_tuple = (t, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length, self.traffic_id)
                self._buffer.append(req_tuple)
                count += 1

            except (ValueError, IndexError) as e:
                # 跳过无效行
                continue


class D2DTrafficScheduler(TrafficScheduler):
    """D2D专用的Traffic调度器"""

    def __init__(self, traffic_config, traffic_file_path, config):
        """
        初始化D2D Traffic调度器

        Args:
            traffic_config: traffic配置列表
            traffic_file_path: traffic文件路径
            config: 仿真配置
        """
        # 调用父类初始化，但使用D2D专用的文件读取器
        self.traffic_file_path = traffic_file_path
        self.config = config
        self.readers = []
        self.pending_requests = defaultdict(list)
        self.received_flit = defaultdict(int)
        self._create_d2d_readers(traffic_config)

    def _create_d2d_readers(self, traffic_config):
        """创建D2D文件读取器"""
        for chain_id, chain_files in enumerate(traffic_config):
            time_offset = 0
            for file_idx, filename in enumerate(chain_files):
                traffic_id = f"chain_{chain_id}_file_{file_idx}"

                # 使用D2D专用的文件读取器
                reader = D2DTrafficFileReader(filename=filename, traffic_file_path=self.traffic_file_path, config=self.config, time_offset=time_offset, traffic_id=traffic_id)
                self.readers.append(reader)

    def get_pending_requests(self, max_cycle: int) -> List[Tuple]:
        """获取待处理的D2D请求"""
        new_requests = []

        for reader in self.readers:
            requests = reader.get_requests_until_cycle(max_cycle)
            new_requests.extend(requests)

        return new_requests

    def create_d2d_flit_from_request(self, req_data, base_model, die_id):
        """
        从D2D请求数据创建带有die信息的flit，实现三阶段路由的第一段

        Args:
            req_data: D2D请求元组 (t, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length, traffic_id)
            base_model: BaseModel实例，用于node_map映射
            die_id: 当前处理的Die ID

        Returns:
            带有die信息的flit对象
        """
        if len(req_data) < 10:
            raise ValueError(f"Invalid D2D request format: {req_data}")

        (inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length, traffic_id) = req_data

        # 使用node_map映射逻辑节点到物理节点
        source_physical = base_model.node_map(src_node, is_source=True)
        final_destination_physical = base_model.node_map(dst_node, is_source=False)

        # D2D三阶段路由：第一段路由（源节点 → 本Die的D2D_SN）
        if src_die != dst_die:
            # 跨Die请求：第一段路由到本Die的D2D_SN
            d2d_sn_position = base_model.config.D2D_SN_POSITION
            path = base_model.routes[source_physical][d2d_sn_position]
            intermediate_destination = d2d_sn_position
        else:
            # 本地请求：直接路由到目标
            path = base_model.routes[source_physical][final_destination_physical]
            intermediate_destination = final_destination_physical

        # 创建flit
        from src.utils.components.flit import Flit

        req = Flit.create_flit(source_physical, intermediate_destination, path)

        # 设置D2D专用信息
        req.source_die_id = src_die
        req.target_die_id = dst_die
        req.source_node_id = src_node
        req.target_node_id = dst_node
        req.source_type = src_ip
        req.destination_type = dst_ip
        req.req_type = req_type.lower()  # 'read' 或 'write'
        req.burst_length = burst_length
        req.traffic_id = traffic_id
        req.inject_time = inject_time

        # 保存最终目标信息（用于后续路由阶段）
        req.final_destination_physical = final_destination_physical
        req.final_destination_type = dst_ip

        # 设置其他必要属性 - 使用全局唯一packet_id
        from src.utils.components.node import Node

        req.packet_id = Node.get_next_packet_id()

        return req

    def is_all_completed(self) -> bool:
        """检查所有traffic是否已完成"""
        for reader in self.readers:
            if not reader._eof or reader._buffer:
                return False
        return True

    def get_total_stats(self) -> Dict[str, int]:
        """获取总体统计信息"""
        stats = {"total_req": 0, "total_flit": 0, "read_req": 0, "write_req": 0, "read_flit": 0, "write_flit": 0}

        for reader in self.readers:
            stats["total_req"] += reader.total_req
            stats["total_flit"] += reader.total_flit
            stats["read_req"] += reader.read_req
            stats["write_req"] += reader.write_req
            stats["read_flit"] += reader.read_flit
            stats["write_flit"] += reader.write_flit

        return stats
