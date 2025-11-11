"""
D2D Traffic Scheduler - 支持跨Die流量格式
扩展原有的traffic_scheduler.py，支持D2D格式的traffic文件。

D2D Traffic格式：inject_time, src_die, src_node, src_ip, dst_die, dst_node, dst_ip, req_type, burst_length
"""

import os
import logging
from typing import List, Dict, Tuple, Optional, Iterator
from collections import defaultdict, deque
from src.noc.traffic_scheduler import TrafficFileReader, TrafficScheduler, SerialChain


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

        # 初始化parallel_chains用于get_save_filename()
        self.parallel_chains = []
        self._create_parallel_chains_from_config(traffic_config)

        self._create_d2d_readers(traffic_config)

    def _create_parallel_chains_from_config(self, traffic_config):
        """根据traffic_config创建parallel_chains结构，用于get_save_filename()"""
        for i, traffic_files in enumerate(traffic_config):
            if not traffic_files:
                continue
            chain_id = f"chain_{i}"
            chain = SerialChain(chain_id, traffic_files)
            self.parallel_chains.append(chain)

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

    def _select_nearest_d2d_sn(self, source_position: int, base_model):
        """选择距离源节点最近的D2D_SN节点"""
        # 根据Die ID获取D2D位置
        die_id = getattr(base_model, "die_id", 0)  # 假设有die_id属性
        if die_id == 0:
            d2d_positions = getattr(base_model.config, "D2D_DIE0_POSITIONS", [])
        else:
            d2d_positions = getattr(base_model.config, "D2D_DIE1_POSITIONS", [])

        d2d_sn_positions = d2d_positions

        if not d2d_sn_positions:
            return None

        # 如果只有一个D2D_SN，直接返回
        if len(d2d_sn_positions) == 1:
            return d2d_sn_positions[0]

        # 计算距离并选择最近的
        min_distance = float("inf")
        nearest_sn = None

        for sn_pos in d2d_sn_positions:
            # 使用路由表计算距离（路径长度）
            if source_position in base_model.routes and sn_pos in base_model.routes[source_position]:
                path_length = len(base_model.routes[source_position][sn_pos])
                if path_length < min_distance:
                    min_distance = path_length
                    nearest_sn = sn_pos

        return nearest_sn if nearest_sn is not None else d2d_sn_positions[0]
