"""
导出器模块 - 提供CSV、JSON和报告导出功能

包含:
1. CSVExporter - CSV文件导出类
2. ReportGenerator - 文本报告生成类
3. JSONExporter - JSON文件导出类
"""

import os
import csv
import json
from collections import defaultdict
from typing import Dict, List, Optional, Any
from .analyzers import RequestInfo, D2DRequestInfo, PortBandwidthMetrics, BandwidthMetrics


class CSVExporter:
    """CSV导出器 - 导出请求数据、端口带宽和链路统计到CSV文件"""

    def generate_detailed_request_csv(
        self,
        requests: List[RequestInfo],
        output_path: str
    ) -> None:
        """
        生成详细的请求CSV文件（分离读写）

        Args:
            requests: RequestInfo列表
            output_path: 输出目录路径
        """
        # 分离读写请求
        read_requests = [req for req in requests if req.req_type == "read"]
        write_requests = [req for req in requests if req.req_type == "write"]

        # CSV文件头部
        csv_header = [
            "packet_id",
            "start_time_ns",
            "end_time_ns",
            "source_node",
            "source_type",
            "dest_node",
            "dest_type",
            "burst_length",
            "cmd_latency_ns",
            "data_latency_ns",
            "transaction_latency_ns",
            "src_dest_order_id",
            "packet_category",
            "cmd_entry_cake0_cycle",
            "cmd_entry_noc_from_cake0_cycle",
            "cmd_entry_noc_from_cake1_cycle",
            "cmd_received_by_cake0_cycle",
            "cmd_received_by_cake1_cycle",
            "data_entry_noc_from_cake0_cycle",
            "data_entry_noc_from_cake1_cycle",
            "data_received_complete_cycle",
            "rsp_entry_network_cycle",
            "data_eject_attempts_h_list",
            "data_eject_attempts_v_list",
        ]

        # 生成读写CSV文件
        read_csv_file = None
        write_csv_file = None

        if read_requests:
            read_csv_file = self._write_request_csv_file(
                read_requests, output_path, "read", csv_header
            )

        if write_requests:
            write_csv_file = self._write_request_csv_file(
                write_requests, output_path, "write", csv_header
            )

        # 打印统计信息
        if read_requests or write_requests:
            print("\n" + "=" * 60)
            print(f"详细请求记录统计:")
            if read_requests:
                print(f"  读请求CSV, {len(read_requests)} 条记录:  {read_csv_file}")
            if write_requests:
                print(f"  写请求CSV, {len(write_requests)} 条记录:  {write_csv_file}")

    def _write_request_csv_file(
        self,
        requests: List[RequestInfo],
        output_path: str,
        req_type: str,
        csv_header: List[str]
    ) -> str:
        """
        将请求列表写入CSV文件

        Args:
            requests: RequestInfo列表
            output_path: 输出目录路径
            req_type: 请求类型 ("read" 或 "write")
            csv_header: CSV文件头

        Returns:
            str: CSV文件路径
        """
        csv_file = os.path.join(output_path, f"{req_type}_requests.csv")
        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            f.write(",".join(csv_header) + "\n")
            for req in requests:
                row = [
                    req.packet_id,
                    req.start_time,
                    req.end_time,
                    req.source_node,
                    req.source_type,
                    req.dest_node,
                    req.dest_type,
                    req.burst_length,
                    req.cmd_latency,
                    req.data_latency,
                    req.transaction_latency,
                    req.src_dest_order_id,
                    req.packet_category,
                    req.cmd_entry_cake0_cycle,
                    req.cmd_entry_noc_from_cake0_cycle,
                    req.cmd_entry_noc_from_cake1_cycle,
                    req.cmd_received_by_cake0_cycle,
                    req.cmd_received_by_cake1_cycle,
                    req.data_entry_noc_from_cake0_cycle,
                    req.data_entry_noc_from_cake1_cycle,
                    req.data_received_complete_cycle,
                    req.rsp_entry_network_cycle,
                    ",".join(map(str, req.data_eject_attempts_h_list)),
                    ",".join(map(str, req.data_eject_attempts_v_list)),
                ]
                f.write(",".join(map(str, row)) + "\n")
        return csv_file

    def generate_ports_csv(
        self,
        rn_ports: Dict[str, PortBandwidthMetrics],
        output_path: str,
        sn_ports: Dict[str, PortBandwidthMetrics] = None,
        config: Any = None,
        topo_type: str = None
    ) -> None:
        """
        生成端口带宽CSV文件

        Args:
            rn_ports: RN端口带宽指标字典 {port_id: PortBandwidthMetrics}
            output_path: 输出目录路径
            sn_ports: SN端口带宽指标字典（可选）
            config: 配置对象（可选，用于坐标计算）
            topo_type: 拓扑类型（可选）
        """
        # 合并RN和SN端口数据
        all_ports = {**rn_ports}
        if sn_ports:
            all_ports.update(sn_ports)

        # 若无任何端口数据则跳过
        if not all_ports:
            return

        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)

        # CSV文件头
        csv_header = [
            "port_id",
            "coordinate",
            "read_unweighted_bandwidth_gbps",
            "read_weighted_bandwidth_gbps",
            "write_unweighted_bandwidth_gbps",
            "write_weighted_bandwidth_gbps",
            "mixed_unweighted_bandwidth_gbps",
            "mixed_weighted_bandwidth_gbps",
            "read_requests_count",
            "write_requests_count",
            "total_requests_count",
            "read_flits_count",
            "write_flits_count",
            "total_flits_count",
            "read_working_intervals_count",
            "write_working_intervals_count",
            "mixed_working_intervals_count",
            "read_total_working_time_ns",
            "write_total_working_time_ns",
            "mixed_total_working_time_ns",
            "read_network_start_time_ns",
            "read_network_end_time_ns",
            "write_network_start_time_ns",
            "write_network_end_time_ns",
            "mixed_network_start_time_ns",
            "mixed_network_end_time_ns",
        ]

        csv_file = os.path.join(output_path, "ports_bandwidth.csv")
        with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
            # 写入头部
            f.write(",".join(csv_header) + "\n")

            # 排序：先按端口类型字符串，再按节点编号大小
            sorted_ports = sorted(all_ports.items(), key=lambda x: (x[0].split("_")[0], int(x[0].rsplit("_", 1)[1])))

            for port_id, metrics in sorted_ports:
                # 提取节点索引
                idx = int(port_id.rsplit("_", 1)[1])

                # 计算coordinate
                if topo_type and topo_type.startswith("Ring"):
                    # Ring拓扑：直接使用节点编号
                    coordinate = str(idx)
                elif config:
                    # CrossRing拓扑：从左下角开始算x/y
                    cols = getattr(config, "NUM_COL", 4)
                    rows = getattr(config, "NUM_ROW", 5)
                    # 原编号从左上角行优先递增
                    row_from_top = idx // cols
                    col = idx % cols
                    row = rows - 1 - row_from_top
                    coordinate = f"x{col}_y{row}"
                else:
                    coordinate = str(idx)

                # 统计flit数量
                read_flits = sum(iv.flit_count for iv in metrics.read_metrics.working_intervals) if metrics.read_metrics.working_intervals else 0
                write_flits = sum(iv.flit_count for iv in metrics.write_metrics.working_intervals) if metrics.write_metrics.working_intervals else 0
                mixed_flits = sum(iv.flit_count for iv in metrics.mixed_metrics.working_intervals) if metrics.mixed_metrics.working_intervals else 0

                # 组装CSV行
                row_data = [
                    port_id,
                    coordinate,
                    metrics.read_metrics.unweighted_bandwidth,
                    metrics.read_metrics.weighted_bandwidth,
                    metrics.write_metrics.unweighted_bandwidth,
                    metrics.write_metrics.weighted_bandwidth,
                    metrics.mixed_metrics.unweighted_bandwidth,
                    metrics.mixed_metrics.weighted_bandwidth,
                    metrics.read_metrics.total_requests,
                    metrics.write_metrics.total_requests,
                    metrics.mixed_metrics.total_requests,
                    read_flits,
                    write_flits,
                    mixed_flits,
                    len(metrics.read_metrics.working_intervals),
                    len(metrics.write_metrics.working_intervals),
                    len(metrics.mixed_metrics.working_intervals),
                    metrics.read_metrics.total_working_time,
                    metrics.write_metrics.total_working_time,
                    metrics.mixed_metrics.total_working_time,
                    metrics.read_metrics.network_start_time,
                    metrics.read_metrics.network_end_time,
                    metrics.write_metrics.network_start_time,
                    metrics.write_metrics.network_end_time,
                    metrics.mixed_metrics.network_start_time,
                    metrics.mixed_metrics.network_end_time,
                ]
                f.write(",".join(map(str, row_data)) + "\n")

            # 添加端口带宽平均值统计
            self._write_port_bandwidth_averages(f, all_ports)

    def _write_port_bandwidth_averages(self, f, all_ports: Dict[str, PortBandwidthMetrics]):
        """
        写入端口带宽平均值统计行到CSV文件末尾

        Args:
            f: CSV文件句柄
            all_ports: 所有端口的带宽指标字典
        """
        # 按端口类型分组
        port_groups = defaultdict(list)
        for port_id, metrics in all_ports.items():
            port_type = port_id.split("_")[0]  # 提取端口类型
            port_groups[port_type].append(metrics)

        # 写入分隔行
        f.write("\n# Port Bandwidth Averages by Type\n")

        # 为每种端口类型计算并写入平均值
        for port_type, metrics_list in sorted(port_groups.items()):
            if not metrics_list:
                continue

            # 计算各类带宽的平均值
            read_unweighted_avg = sum(m.read_metrics.unweighted_bandwidth for m in metrics_list) / len(metrics_list)
            read_weighted_avg = sum(m.read_metrics.weighted_bandwidth for m in metrics_list) / len(metrics_list)
            write_unweighted_avg = sum(m.write_metrics.unweighted_bandwidth for m in metrics_list) / len(metrics_list)
            write_weighted_avg = sum(m.write_metrics.weighted_bandwidth for m in metrics_list) / len(metrics_list)
            mixed_unweighted_avg = sum(m.mixed_metrics.unweighted_bandwidth for m in metrics_list) / len(metrics_list)
            mixed_weighted_avg = sum(m.mixed_metrics.weighted_bandwidth for m in metrics_list) / len(metrics_list)

            # 写入平均值行
            avg_row = [
                f"{port_type}_average",
                "",
                f"{read_unweighted_avg:.6f}",
                f"{read_weighted_avg:.6f}",
                f"{write_unweighted_avg:.6f}",
                f"{write_weighted_avg:.6f}",
                f"{mixed_unweighted_avg:.6f}",
                f"{mixed_weighted_avg:.6f}",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
            f.write(",".join(avg_row) + "\n")

    def export_link_statistics_csv(
        self,
        network,
        csv_path: str
    ) -> None:
        """
        导出链路统计数据到CSV

        Args:
            network: Network对象（包含links_flow_stat）
            csv_path: CSV文件路径
        """
        if not hasattr(network, "get_links_utilization_stats") or not callable(network.get_links_utilization_stats):
            print(f"警告: 网络 {network.name} 不支持链路统计导出")
            return

        try:
            utilization_stats = network.get_links_utilization_stats()
            if not utilization_stats:
                print("警告: 没有链路统计数据可导出")
                return

            # 准备CSV数据
            csv_data = []
            for link_key, stats in utilization_stats.items():
                # 处理新架构：link可能是(i, j)或(i, j, 'h'/'v')
                if len(link_key) == 2:
                    src, dst = link_key
                    direction = ""
                elif len(link_key) == 3:
                    src, dst, direction = link_key
                else:
                    continue

                # 获取下环尝试次数统计数据
                eject_h = stats.get("eject_attempts_h", {"0": 0, "1": 0, "2": 0, ">2": 0})
                eject_v = stats.get("eject_attempts_v", {"0": 0, "1": 0, "2": 0, ">2": 0})
                eject_h_ratios = stats.get("eject_attempts_h_ratios", {"0": 0.0, "1": 0.0, "2": 0.0, ">2": 0.0})
                eject_v_ratios = stats.get("eject_attempts_v_ratios", {"0": 0.0, "1": 0.0, "2": 0.0, ">2": 0.0})

                row = {
                    "source_node": src,
                    "destination_node": dst,
                    "direction": direction,
                    "utilization": f"{stats.get('utilization', 0.0)*100:.2f}%",
                    "ITag_ratio": f"{stats.get('ITag_ratio', 0.0)*100:.2f}%",
                    "empty_ratio": f"{stats.get('empty_ratio', 0.0)*100:.2f}%",
                    "total_cycles": stats.get("total_cycles", 0),
                    "total_flit": stats.get("total_flit", 0),
                    # 横向环下环尝试次数统计
                    "eject_attempts_h_0": eject_h.get("0", 0),
                    "eject_attempts_h_1": eject_h.get("1", 0),
                    "eject_attempts_h_2": eject_h.get("2", 0),
                    "eject_attempts_h_>2": eject_h.get(">2", 0),
                    # 横向环下环尝试次数比例
                    "eject_attempts_h_0_ratio": f"{eject_h_ratios.get('0', 0.0)*100:.2f}%",
                    "eject_attempts_h_1_ratio": f"{eject_h_ratios.get('1', 0.0)*100:.2f}%",
                    "eject_attempts_h_2_ratio": f"{eject_h_ratios.get('2', 0.0)*100:.2f}%",
                    "eject_attempts_h_>2_ratio": f"{eject_h_ratios.get('>2', 0.0)*100:.2f}%",
                    # 纵向环下环尝试次数统计
                    "eject_attempts_v_0": eject_v.get("0", 0),
                    "eject_attempts_v_1": eject_v.get("1", 0),
                    "eject_attempts_v_2": eject_v.get("2", 0),
                    "eject_attempts_v_>2": eject_v.get(">2", 0),
                    # 纵向环下环尝试次数比例
                    "eject_attempts_v_0_ratio": f"{eject_v_ratios.get('0', 0.0)*100:.2f}%",
                    "eject_attempts_v_1_ratio": f"{eject_v_ratios.get('1', 0.0)*100:.2f}%",
                    "eject_attempts_v_2_ratio": f"{eject_v_ratios.get('2', 0.0)*100:.2f}%",
                    "eject_attempts_v_>2_ratio": f"{eject_v_ratios.get('>2', 0.0)*100:.2f}%",
                }
                csv_data.append(row)

            # 写入CSV文件
            if csv_data:
                fieldnames = [
                    "source_node",
                    "destination_node",
                    "direction",
                    # 下环尝试次数比例（按用户要求放在前面）
                    "eject_attempts_h_0_ratio",
                    "eject_attempts_h_1_ratio",
                    "eject_attempts_h_2_ratio",
                    "eject_attempts_h_>2_ratio",
                    "eject_attempts_v_0_ratio",
                    "eject_attempts_v_1_ratio",
                    "eject_attempts_v_2_ratio",
                    "eject_attempts_v_>2_ratio",
                    # empty和itag比例
                    "empty_ratio",
                    "ITag_ratio",
                    # 具体数量放在最后
                    "utilization",
                    "total_cycles",
                    "total_flit",
                    "eject_attempts_h_0",
                    "eject_attempts_h_1",
                    "eject_attempts_h_2",
                    "eject_attempts_h_>2",
                    "eject_attempts_v_0",
                    "eject_attempts_v_1",
                    "eject_attempts_v_2",
                    "eject_attempts_v_>2",
                ]

                with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_data)
            else:
                print("警告: 没有有效的链路统计数据")

        except Exception as e:
            print(f"导出链路统计数据时发生错误: {e}")

    def save_d2d_requests_csv(
        self,
        d2d_requests: List[D2DRequestInfo],
        output_path: str
    ) -> None:
        """
        保存D2D请求到CSV文件

        Args:
            d2d_requests: D2DRequestInfo列表
            output_path: 输出目录路径
        """
        os.makedirs(output_path, exist_ok=True)

        # 分别保存读请求和写请求
        read_requests = [req for req in d2d_requests if req.req_type == "read"]
        write_requests = [req for req in d2d_requests if req.req_type == "write"]

        # CSV文件头
        csv_header = [
            "packet_id",
            "source_die",
            "target_die",
            "source_node",
            "target_node",
            "source_type",
            "target_type",
            "burst_length",
            "start_time_ns",
            "end_time_ns",
            "cmd_latency_ns",
            "data_latency_ns",
            "transaction_latency_ns",
            "data_bytes",
            "d2d_sn_node",
            "d2d_rn_node",
        ]

        # 只有存在请求时才保存对应的CSV文件
        if read_requests:
            read_csv_path = os.path.join(output_path, "d2d_read_requests.csv")
            self._save_d2d_requests_to_csv(read_requests, read_csv_path, csv_header)

        if write_requests:
            write_csv_path = os.path.join(output_path, "d2d_write_requests.csv")
            self._save_d2d_requests_to_csv(write_requests, write_csv_path, csv_header)

    def _save_d2d_requests_to_csv(self, requests: List[D2DRequestInfo], file_path: str, header: List[str]):
        """保存D2D请求列表到CSV文件"""
        try:
            with open(file_path, "w", newline="", encoding="utf-8-sig") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)

                for req in requests:
                    writer.writerow(
                        [
                            req.packet_id,
                            req.source_die,
                            req.target_die,
                            req.source_node,
                            req.target_node,
                            req.source_type,
                            req.target_type,
                            req.burst_length,
                            req.start_time_ns,
                            req.end_time_ns,
                            req.cmd_latency_ns,
                            req.data_latency_ns,
                            req.transaction_latency_ns,
                            req.data_bytes,
                            req.d2d_sn_node if req.d2d_sn_node is not None else "",
                            req.d2d_rn_node if req.d2d_rn_node is not None else "",
                        ]
                    )
        except (IOError, OSError) as e:
            raise

    def save_ip_bandwidth_to_csv(
        self,
        die_ip_bandwidth_data: Dict,
        config,
        output_path: str
    ) -> None:
        """
        保存所有Die的IP带宽数据到单个CSV文件

        Args:
            die_ip_bandwidth_data: IP带宽数据字典
            config: 配置对象
            output_path: 输出目录路径
        """
        os.makedirs(output_path, exist_ok=True)

        # 检查数据是否存在
        if not die_ip_bandwidth_data:
            print("警告: 没有die_ip_bandwidth_data数据，跳过IP带宽CSV导出")
            return

        csv_path = os.path.join(output_path, "ip_bandwidth.csv")

        try:
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
                writer = csv.writer(csvfile)

                # 写入CSV头：简洁明了的格式
                writer.writerow(
                    [
                        "ip_instance",  # IP实例名 (如gdma_0, ddr_1)
                        "die_id",  # Die ID
                        "node_id",  # 节点ID
                        "ip_type",  # IP类型 (如gdma, ddr)
                        "read_bandwidth_gbps",  # 读带宽
                        "write_bandwidth_gbps",  # 写带宽
                        "total_bandwidth_gbps",  # 总带宽
                    ]
                )

                # 收集所有数据行
                all_rows = []

                for die_id, die_data in die_ip_bandwidth_data.items():
                    # 获取该Die的配置信息
                    num_col = getattr(config, "NUM_COL", 4)
                    num_row = getattr(config, "NUM_ROW", 5)

                    # 获取三种模式的数据
                    read_data = die_data.get("read", {})
                    write_data = die_data.get("write", {})
                    total_data = die_data.get("total", {})

                    # 收集所有IP实例
                    all_ip_instances = set(read_data.keys()) | set(write_data.keys()) | set(total_data.keys())

                    for ip_instance in all_ip_instances:
                        # 提取IP基本类型
                        if ip_instance.lower().startswith("d2d"):
                            parts = ip_instance.lower().split("_")
                            ip_type = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
                        else:
                            ip_type = ip_instance.split("_")[0]

                        # 获取矩阵
                        read_matrix = read_data.get(ip_instance)
                        write_matrix = write_data.get(ip_instance)
                        total_matrix = total_data.get(ip_instance)

                        # 确定矩阵形状
                        if read_matrix is not None:
                            rows, cols = read_matrix.shape
                        elif write_matrix is not None:
                            rows, cols = write_matrix.shape
                        elif total_matrix is not None:
                            rows, cols = total_matrix.shape
                        else:
                            continue

                        # 遍历矩阵中的所有位置
                        for matrix_row in range(rows):
                            for matrix_col in range(cols):
                                # 获取三种带宽值
                                read_bw = read_matrix[matrix_row, matrix_col] if read_matrix is not None else 0.0
                                write_bw = write_matrix[matrix_row, matrix_col] if write_matrix is not None else 0.0
                                total_bw = total_matrix[matrix_row, matrix_col] if total_matrix is not None else 0.0

                                # 只保存有带宽的数据（任一模式大于阈值）
                                if read_bw > 0.001 or write_bw > 0.001 or total_bw > 0.001:
                                    # 计算节点ID
                                    node_id = matrix_row * num_col + matrix_col

                                    all_rows.append([ip_instance, die_id, node_id, ip_type, f"{read_bw:.6f}", f"{write_bw:.6f}", f"{total_bw:.6f}"])

                # 排序：先按die_id，再按node_id，最后按ip_instance
                all_rows.sort(key=lambda x: (int(x[1]), int(x[2]), x[0]))

                # 写入所有数据行
                for row in all_rows:
                    writer.writerow(row)

                # 计算并添加平均带宽统计
                ip_type_groups = defaultdict(lambda: {"read": [], "write": [], "total": []})

                # 按IP类型分组（去掉实例编号）
                for row in all_rows:
                    ip_type = row[3]  # IP类型列
                    read_bw = float(row[4])  # 读带宽
                    write_bw = float(row[5])  # 写带宽
                    total_bw = float(row[6])  # 总带宽

                    ip_type_groups[ip_type]["read"].append(read_bw)
                    ip_type_groups[ip_type]["write"].append(write_bw)
                    ip_type_groups[ip_type]["total"].append(total_bw)

                # 添加空行分隔
                writer.writerow([])
                writer.writerow(["# 平均带宽统计（按IP类型）"])
                writer.writerow(["ip_type", "avg_read_bandwidth_gbps", "avg_write_bandwidth_gbps", "avg_total_bandwidth_gbps", "instance_count"])

                # 计算并写入平均值
                for ip_type in sorted(ip_type_groups.keys()):
                    group = ip_type_groups[ip_type]
                    count = len(group["read"])

                    avg_read = sum(group["read"]) / count if count > 0 else 0.0
                    avg_write = sum(group["write"]) / count if count > 0 else 0.0
                    avg_total = sum(group["total"]) / count if count > 0 else 0.0

                    writer.writerow([ip_type, f"{avg_read:.6f}", f"{avg_write:.6f}", f"{avg_total:.6f}", count])

        except (IOError, OSError) as e:
            print(f"警告: 保存IP带宽CSV失败 ({csv_path}): {e}")

    def save_d2d_axi_channel_statistics(
        self,
        output_path: str,
        d2d_bandwidth: Dict,
        dies: Dict = None,
        config: Any = None
    ) -> None:
        """
        保存所有AXI通道的带宽统计到文件

        Args:
            output_path: 输出目录路径
            d2d_bandwidth: D2D_Sys带宽统计字典 {die_id: {node_pos: {channel: bandwidth}}}
            dies: Die模型字典（可选，用于获取flit计数）
            config: 配置对象（可选，用于D2D连接信息）
        """
        # AXI通道描述
        AXI_CHANNEL_DESCRIPTIONS = {
            "AR": "读地址通道 (Address Read)",
            "R": "读数据通道 (Read Data)",
            "AW": "写地址通道 (Address Write)",
            "W": "写数据通道 (Write Data)",
            "B": "写响应通道 (Write Response)",
        }

        try:
            os.makedirs(output_path, exist_ok=True)

            # 1. 保存详细的AXI通道带宽统计到CSV
            csv_path = os.path.join(output_path, "d2d_axi_channel_bandwidth.csv")

            with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
                writer = csv.writer(csvfile)

                # CSV文件头
                writer.writerow(["Die_ID", "Channel", "Direction", "Bandwidth_GB/s", "Flit_Count", "Channel_Description"])

                # 写入各通道数据
                for die_id, node_data in d2d_bandwidth.items():
                    # 遍历该Die的每个D2D节点
                    for node_pos, channels in node_data.items():
                        # 从die模型获取该节点的原始flit计数
                        flit_counts = {channel: 0 for channel in ["AR", "R", "AW", "W", "B"]}

                        if dies:
                            die_model = dies.get(die_id)
                            if die_model and hasattr(die_model, "d2d_systems"):
                                d2d_sys = die_model.d2d_systems.get(node_pos)
                                if d2d_sys and hasattr(d2d_sys, "axi_channel_flit_count"):
                                    flit_counts = d2d_sys.axi_channel_flit_count.copy()

                        # 写入各通道数据
                        direction_mapping = {
                            "AR": f"Die{die_id}->Die{1-die_id}",
                            "R": f"Die{1-die_id}->Die{die_id}",
                            "AW": f"Die{die_id}->Die{1-die_id}",
                            "W": f"Die{die_id}->Die{1-die_id}",
                            "B": f"Die{1-die_id}->Die{die_id}",
                        }

                        for channel, bandwidth in channels.items():
                            flit_count = flit_counts.get(channel, 0)
                            direction = direction_mapping.get(channel, f"Die{die_id}")
                            description = AXI_CHANNEL_DESCRIPTIONS.get(channel, f"{channel} Channel")

                            # 添加节点位置信息到CSV
                            writer.writerow([f"Die{die_id}_Node{node_pos}", channel, direction, f"{bandwidth:.6f}", flit_count, description])

            # 2. 生成汇总报告
            summary_path = os.path.join(output_path, "d2d_axi_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("D2D AXI通道带宽统计汇总报告\n")
                f.write("=" * 60 + "\n\n")

                # 各Die的通道统计
                for die_id, node_data in d2d_bandwidth.items():
                    f.write(f"Die {die_id} AXI通道带宽:\n")
                    f.write("-" * 30 + "\n")

                    die_total_bandwidth = 0.0
                    die_data_bandwidth = 0.0  # 数据通道（R+W）

                    for node_pos, channels in node_data.items():
                        f.write(f"  节点{node_pos}:\n")
                        node_total = 0.0
                        node_data_bw = 0.0

                        for channel, bandwidth in channels.items():
                            if bandwidth > 0:
                                f.write(f"    {channel}通道: {bandwidth:.3f} GB/s\n")
                                node_total += bandwidth
                                die_total_bandwidth += bandwidth
                                if channel in ["R", "W"]:
                                    node_data_bw += bandwidth
                                    die_data_bandwidth += bandwidth

                        if node_total > 0:
                            f.write(f"    节点总带宽: {node_total:.3f} GB/s (数据: {node_data_bw:.3f} GB/s)\n")
                        f.write("\n")

                    f.write(f"  Die{die_id}总带宽: {die_total_bandwidth:.3f} GB/s\n")
                    f.write(f"  Die{die_id}数据带宽 (R+W): {die_data_bandwidth:.3f} GB/s\n\n")

                # 跨Die数据流汇总
                f.write("跨Die数据流汇总:\n")
                f.write("-" * 30 + "\n")

                if 0 in d2d_bandwidth and 1 in d2d_bandwidth:
                    # 计算各Die的总数据带宽
                    die0_total_w = sum(channels.get("W", 0.0) for channels in d2d_bandwidth[0].values())
                    die1_total_r = sum(channels.get("R", 0.0) for channels in d2d_bandwidth[1].values())

                    f.write(f"Die0 -> Die1 总写数据带宽: {die0_total_w:.3f} GB/s\n")
                    f.write(f"Die1 -> Die0 总读数据带宽: {die1_total_r:.3f} GB/s\n")
                    f.write(f"跨Die总数据带宽: {die0_total_w + die1_total_r:.3f} GB/s\n\n")

                    # 详细的节点对连接统计
                    if config:
                        f.write("详细节点对连接:\n")
                        d2d_pairs = getattr(config, "D2D_PAIRS", [])

                        if d2d_pairs:
                            for i, (die0_id, die0_node, die1_id, die1_node) in enumerate(d2d_pairs):
                                die0_w = d2d_bandwidth.get(die0_id, {}).get(die0_node, {}).get("W", 0.0)
                                die1_r = d2d_bandwidth.get(die1_id, {}).get(die1_node, {}).get("R", 0.0)
                                if die0_w > 0 or die1_r > 0:
                                    f.write(f"  连接{i}: Die{die0_id}节点{die0_node} <-> Die{die1_id}节点{die1_node}\n")
                                    f.write(f"    写数据: {die0_w:.3f} GB/s, 读数据: {die1_r:.3f} GB/s\n")
                        else:
                            # 向后兼容
                            die0_positions = getattr(config, "D2D_DIE0_POSITIONS", [])
                            die1_positions = getattr(config, "D2D_DIE1_POSITIONS", [])
                            if die0_positions and die1_positions:
                                for i, (die0_pos, die1_pos) in enumerate(zip(die0_positions, die1_positions)):
                                    die0_w = d2d_bandwidth.get(0, {}).get(die0_pos, {}).get("W", 0.0)
                                    die1_r = d2d_bandwidth.get(1, {}).get(die1_pos, {}).get("R", 0.0)
                                    if die0_w > 0 or die1_r > 0:
                                        f.write(f"  连接{i}: Die0节点{die0_pos} <-> Die1节点{die1_pos}\n")
                                        f.write(f"    写数据: {die0_w:.3f} GB/s, 读数据: {die1_r:.3f} GB/s\n")
                        f.write("\n")

                # 通道利用率分析
                f.write("AXI通道功能说明:\n")
                f.write("-" * 30 + "\n")
                f.write("AR (Address Read): 读地址通道 - 发送读请求地址\n")
                f.write("R  (Read Data):    读数据通道 - 返回读取的数据\n")
                f.write("AW (Address Write): 写地址通道 - 发送写请求地址\n")
                f.write("W  (Write Data):   写数据通道 - 发送写入的数据\n")
                f.write("B  (Write Response): 写响应通道 - 返回写操作完成确认\n\n")

                f.write("注意: 流量图中只显示数据通道(R+W)的带宽，\n")
                f.write("      完整的AXI通道统计请参考CSV文件。\n")

        except (IOError, PermissionError, OSError) as e:
            print(f"警告: 保存D2D AXI统计失败: {e}")
        except Exception as e:
            print(f"错误: 生成D2D AXI报告时发生未预期的错误: {e}")


class ReportGenerator:
    """报告生成器 - 生成文本格式的统计报告"""

    def generate_unified_report(
        self,
        results: Dict,
        output_path: str,
        num_ip: int = 1
    ) -> None:
        """
        生成统一的文本报告

        Args:
            results: 统计结果字典，包含:
                - network_overall: 网络整体带宽
                - rn_ports: RN端口带宽
                - summary: 汇总信息
                - latency_stats: 延迟统计（可选）
                - circuit_stats: 绕环统计（可选）
            output_path: 输出目录路径
            num_ip: IP数量（用于平均带宽计算）
        """
        os.makedirs(output_path, exist_ok=True)

        report_file = os.path.join(output_path, "bandwidth_analysis_report.txt")

        with open(report_file, "w", encoding="utf-8") as f:
            # 写入报告头部
            summary = results.get("summary", {})
            f.write("=" * 80 + "\n")
            f.write("网络带宽分析报告\n")
            f.write("=" * 80 + "\n\n")

            f.write("分析概览:\n")
            f.write(f"  总请求数: {summary.get('total_requests', 0)}\n")
            f.write(f"  读请求数: {summary.get('read_requests', 0)}\n")
            f.write(f"  写请求数: {summary.get('write_requests', 0)}\n")
            f.write(f"  读取总flit数: {summary.get('total_read_flits', 0)}\n")
            f.write(f"  写入总flit数: {summary.get('total_write_flits', 0)}\n")

            # 延迟统计
            latency_stats = summary.get("latency_stats", {})
            if latency_stats:
                f.write("\n延迟统计 (cycle):\n")
                for cat, label in [("cmd", "CMD"), ("data", "Data"), ("trans", "Trans")]:
                    if cat in latency_stats:
                        rl = latency_stats[cat]
                        read_avg = rl['read']['sum'] / rl['read']['count'] if rl['read']['count'] else 0.0
                        write_avg = rl['write']['sum'] / rl['write']['count'] if rl['write']['count'] else 0.0
                        mixed_avg = rl['mixed']['sum'] / rl['mixed']['count'] if rl['mixed']['count'] else 0.0
                        f.write(
                            f"  {label} 延迟 - "
                            f"读 avg {read_avg:.2f}, max {rl['read']['max']}; "
                            f"写 avg {write_avg:.2f}, max {rl['write']['max']}; "
                            f"混合 avg {mixed_avg:.2f}, max {rl['mixed']['max']}\n"
                        )

            # 网络带宽统计
            f.write("\n" + "=" * 50 + "\n")
            f.write("网络带宽统计\n")
            f.write("=" * 50 + "\n\n")

            network_overall = results.get("network_overall", {})

            if network_overall:
                read_metrics = network_overall.get("read")
                write_metrics = network_overall.get("write")
                mixed_metrics = network_overall.get("mixed")

                if read_metrics:
                    f.write(f"网络带宽:\n")
                    f.write(f"  读带宽:    {read_metrics.weighted_bandwidth:.3f} GB/s (平均: {read_metrics.weighted_bandwidth/num_ip:.3f} GB/s)\n")
                if write_metrics:
                    f.write(f"  写带宽:    {write_metrics.weighted_bandwidth:.3f} GB/s (平均: {write_metrics.weighted_bandwidth/num_ip:.3f} GB/s)\n")
                if mixed_metrics:
                    f.write(f"  混合带宽:  {mixed_metrics.weighted_bandwidth:.3f} GB/s (平均: {mixed_metrics.weighted_bandwidth/num_ip:.3f} GB/s)\n\n")

                # 详细信息
                for operation in ["read", "write", "mixed"]:
                    metrics = network_overall.get(operation)
                    if metrics:
                        f.write(f"{operation.upper()} 操作详细信息:\n")
                        f.write(f"  网络工作时间: {metrics.network_start_time} - {metrics.network_end_time} ns (总计 {metrics.network_end_time - metrics.network_start_time} ns)\n")
                        f.write(f"  实际工作时间: {metrics.total_working_time} ns\n")
                        f.write(f"  工作区间数: {len(metrics.working_intervals)}\n")
                        f.write(f"  请求总数: {metrics.total_requests}\n")
                        f.write("\n")

    def generate_d2d_bandwidth_report(
        self,
        output_path: str,
        d2d_stats: Any = None,
        d2d_requests: List = None,
        latency_stats: Dict = None,
        circuit_stats: Dict = None
    ) -> str:
        """
        生成D2D带宽报告

        Args:
            output_path: 输出目录路径
            d2d_stats: D2D带宽统计对象（包含pair_read_bw, pair_write_bw）
            d2d_requests: D2D请求列表（可选，用于额外统计）
            latency_stats: 延迟统计字典（可选）
            circuit_stats: 绕环统计字典（可选）

        Returns:
            报告文件路径
        """
        os.makedirs(output_path, exist_ok=True)
        report_file = os.path.join(output_path, "d2d_bandwidth_summary.txt")

        # 生成报告内容
        report_lines = [
            "=" * 50,
            "D2D带宽统计报告",
            "=" * 50,
            "",
        ]

        if d2d_stats:
            # 读带宽
            report_lines.append("按Die组合的读带宽 (GB/s):")
            total_unweighted = 0.0
            total_weighted = 0.0

            for (src, dst), (uw, wt) in sorted(d2d_stats.pair_read_bw.items()):
                report_lines.append(f"Die{src} → Die{dst} (Read):  {uw:.2f} (加权: {wt:.2f})")
                total_unweighted += uw
                total_weighted += wt

            report_lines.extend(["", "按Die组合的写带宽 (GB/s):"])

            # 写带宽
            for (src, dst), (uw, wt) in sorted(d2d_stats.pair_write_bw.items()):
                report_lines.append(f"Die{src} → Die{dst} (Write): {uw:.2f} (加权: {wt:.2f})")
                total_unweighted += uw
                total_weighted += wt

        # 延迟统计
        if latency_stats:
            def _avg(cat, op):
                s = latency_stats[cat][op]
                return s["sum"] / s["count"] if s["count"] else 0.0

            report_lines.extend(["", "延迟统计 (cycle):"])
            for cat, label in [("cmd", "CMD"), ("data", "Data"), ("trans", "Trans")]:
                if cat in latency_stats:
                    line = (
                        f"  {label} 延迟  - "
                        f"读: avg {_avg(cat,'read'):.2f}, max {int(latency_stats[cat]['read']['max'])}; "
                        f"写: avg {_avg(cat,'write'):.2f}, max {int(latency_stats[cat]['write']['max'])}; "
                        f"混合: avg {_avg(cat,'mixed'):.2f}, max {int(latency_stats[cat]['mixed']['max'])}"
                    )
                    report_lines.append(line)

        # 绕环统计
        if circuit_stats:
            summary = circuit_stats.get("summary", circuit_stats)
            report_lines.extend(["", "-" * 60, "绕环与Tag统计（汇总）", "-" * 60])
            report_lines.extend(self._format_circuit_stats(summary, prefix="  "))

        report_lines.append("-" * 60)

        # 打印到屏幕
        for line in report_lines:
            print(line)

        # 保存到文件
        with open(report_file, "w", encoding="utf-8") as f:
            for line in report_lines:
                f.write(line + "\n")

            # 添加每个Die的详细统计（如果有）
            if circuit_stats and "per_die" in circuit_stats:
                f.write("\n\n")
                f.write("=" * 60 + "\n")
                f.write("各Die详细统计\n")
                f.write("=" * 60 + "\n\n")

                for die_id in sorted(circuit_stats["per_die"].keys()):
                    die_stats = circuit_stats["per_die"][die_id]
                    f.write(f"Die {die_id}:\n")
                    f.write("-" * 30 + "\n")
                    for line in self._format_circuit_stats(die_stats, prefix="  "):
                        f.write(line + "\n")
                    f.write("\n")

        return report_file

    def _format_circuit_stats(self, stats: Dict, prefix: str = "") -> List[str]:
        """
        格式化绕环统计数据

        Args:
            stats: 绕环统计字典
            prefix: 行前缀

        Returns:
            格式化的文本行列表
        """
        lines = []

        # Circuits统计
        lines.append(f"{prefix}Circuits req  - h: {stats.get('req_circuits_h', 0)}, v: {stats.get('req_circuits_v', 0)}")
        lines.append(f"{prefix}Circuits rsp  - h: {stats.get('rsp_circuits_h', 0)}, v: {stats.get('rsp_circuits_v', 0)}")
        lines.append(f"{prefix}Circuits data - h: {stats.get('data_circuits_h', 0)}, v: {stats.get('data_circuits_v', 0)}")

        # Wait cycles统计
        lines.append(f"{prefix}Wait cycle req  - h: {stats.get('req_wait_cycles_h', 0)}, v: {stats.get('req_wait_cycles_v', 0)}")
        lines.append(f"{prefix}Wait cycle rsp  - h: {stats.get('rsp_wait_cycles_h', 0)}, v: {stats.get('rsp_wait_cycles_v', 0)}")
        lines.append(f"{prefix}Wait cycle data - h: {stats.get('data_wait_cycles_h', 0)}, v: {stats.get('data_wait_cycles_v', 0)}")

        # ETag统计
        lines.append(f"{prefix}RB ETag - T1: {stats.get('RB_ETag_T1_num', 0)}, T0: {stats.get('RB_ETag_T0_num', 0)}")
        lines.append(f"{prefix}EQ ETag - T1: {stats.get('EQ_ETag_T1_num', 0)}, T0: {stats.get('EQ_ETag_T0_num', 0)}")

        # ITag统计
        lines.append(f"{prefix}ITag - h: {stats.get('ITag_h_num', 0)}, v: {stats.get('ITag_v_num', 0)}")

        # Retry统计
        lines.append(f"{prefix}Retry - read: {stats.get('read_retry_num', 0)}, write: {stats.get('write_retry_num', 0)}")

        return lines


class JSONExporter:
    """JSON导出器 - 导出统计结果到JSON文件"""

    def generate_json_report(
        self,
        results: Dict,
        json_file: str
    ) -> None:
        """
        生成JSON格式的统计报告

        Args:
            results: 统计结果字典
            json_file: JSON文件路径
        """
        # 转换为可序列化的格式
        serializable_results = {}

        # 网络整体数据 - 包含混合带宽
        if "network_overall" in results:
            serializable_results["network_overall"] = {}
            for op_type, metrics in results["network_overall"].items():
                serializable_results["network_overall"][op_type] = {
                    "unweighted_bandwidth_gbps": metrics.unweighted_bandwidth,
                    "weighted_bandwidth_gbps": metrics.weighted_bandwidth,
                    "total_working_time_ns": metrics.total_working_time,
                    "network_start_time_ns": metrics.network_start_time,
                    "network_end_time_ns": metrics.network_end_time,
                    "total_bytes": metrics.total_bytes,
                    "total_requests": metrics.total_requests,
                    "working_intervals": [
                        {
                            "start_time_ns": interval.start_time,
                            "end_time_ns": interval.end_time,
                            "duration_ns": interval.duration,
                            "flit_count": interval.flit_count,
                            "total_bytes": interval.total_bytes,
                            "request_count": interval.request_count,
                            "bandwidth_bytes_per_ns": interval.bandwidth_bytes_per_ns,
                        }
                        for interval in metrics.working_intervals
                    ],
                }

        # RN端口数据 - 包含混合带宽
        if "rn_ports" in results:
            serializable_results["rn_ports"] = {}
            for port_id, port_metrics in results["rn_ports"].items():
                serializable_results["rn_ports"][port_id] = {
                    "read": {
                        "unweighted_bandwidth_gbps": port_metrics.read_metrics.unweighted_bandwidth,
                        "weighted_bandwidth_gbps": port_metrics.read_metrics.weighted_bandwidth,
                        "total_requests": port_metrics.read_metrics.total_requests,
                        "total_bytes": port_metrics.read_metrics.total_bytes,
                    },
                    "write": {
                        "unweighted_bandwidth_gbps": port_metrics.write_metrics.unweighted_bandwidth,
                        "weighted_bandwidth_gbps": port_metrics.write_metrics.weighted_bandwidth,
                        "total_requests": port_metrics.write_metrics.total_requests,
                        "total_bytes": port_metrics.write_metrics.total_bytes,
                    },
                    "mixed": {
                        "unweighted_bandwidth_gbps": port_metrics.mixed_metrics.unweighted_bandwidth,
                        "weighted_bandwidth_gbps": port_metrics.mixed_metrics.weighted_bandwidth,
                        "total_requests": port_metrics.mixed_metrics.total_requests,
                        "total_bytes": port_metrics.mixed_metrics.total_bytes,
                    },
                }

        # 汇总数据
        if "summary" in results:
            serializable_results["summary"] = results["summary"]

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    def save_config_json(
        self,
        config: Any,
        output_path: str,
        additional_data: Dict = None
    ) -> None:
        """
        保存配置和额外数据到JSON

        Args:
            config: 配置对象
            output_path: 输出目录路径
            additional_data: 额外要保存的数据字典
        """
        config_data = {
            "network_frequency": getattr(config, "NETWORK_FREQUENCY", 2.0),
            "burst": getattr(config, "BURST", 4),
            "topo_type": getattr(config, "TOPO_TYPE", "unknown"),
            "num_ip": getattr(config, "NUM_IP", 1),
            "num_col": getattr(config, "NUM_COL", 4),
            "num_row": getattr(config, "NUM_ROW", 5),
        }

        if additional_data:
            config_data.update(additional_data)

        json_path = os.path.join(output_path, "analysis_config.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
