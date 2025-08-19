"""
D2D系统专用结果处理器

用于处理跨Die通信的带宽统计和请求记录
"""

import os
import csv
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .result_processor import BandwidthAnalyzer, RequestInfo, BandwidthMetrics, WorkingInterval
from src.utils.components.flit import Flit


@dataclass
class D2DRequestInfo:
    """D2D请求信息数据结构"""
    packet_id: int
    source_die: int
    target_die: int
    source_node: int
    target_node: int
    source_type: str
    target_type: str
    req_type: str  # 'read' or 'write'
    burst_length: int
    data_bytes: int
    start_time_ns: int
    end_time_ns: int
    latency_ns: int


@dataclass
class D2DBandwidthStats:
    """D2D带宽统计数据结构"""
    die0_to_die1_read_bw: float = 0.0
    die0_to_die1_read_bw_weighted: float = 0.0
    die0_to_die1_write_bw: float = 0.0
    die0_to_die1_write_bw_weighted: float = 0.0
    
    die1_to_die0_read_bw: float = 0.0
    die1_to_die0_read_bw_weighted: float = 0.0
    die1_to_die0_write_bw: float = 0.0
    die1_to_die0_write_bw_weighted: float = 0.0
    
    total_read_requests: int = 0
    total_write_requests: int = 0
    total_bytes_transferred: int = 0


class D2DResultProcessor(BandwidthAnalyzer):
    """D2D系统专用的结果处理器，继承自BandwidthAnalyzer"""
    
    FLIT_SIZE_BYTES = 128  # 每个flit的字节数
    
    def __init__(self, config, min_gap_threshold: int = 50):
        super().__init__(config, min_gap_threshold)
        self.d2d_requests: List[D2DRequestInfo] = []
        self.d2d_stats = D2DBandwidthStats()
        # 修复网络频率属性问题
        self.network_frequency = getattr(config, 'NETWORK_FREQUENCY', 2)
        
    def collect_cross_die_requests(self, dies: Dict):
        """
        从两个Die的网络中收集跨Die请求数据
        
        Args:
            dies: Dict[die_id, die_model] - Die模型字典
        """
        self.d2d_requests.clear()
        
        for die_id, die_model in dies.items():
            # 检查数据网络中的arrive_flits
            if hasattr(die_model, 'data_network') and hasattr(die_model.data_network, 'arrive_flits'):
                self._collect_requests_from_network(die_model.data_network, die_id)
                
        print(f"[D2D结果处理] 收集到 {len(self.d2d_requests)} 个跨Die请求")
        
    def _collect_requests_from_network(self, network, die_id: int):
        """从单个网络中收集跨Die请求"""
        for packet_id, flits in network.arrive_flits.items():
            if not flits:
                continue
                
            first_flit = flits[0]
            # 改进数据验证
            if not hasattr(first_flit, 'burst_length') or len(flits) != first_flit.burst_length:
                continue
                
            last_flit = flits[-1]
            
            # 检查是否为跨Die请求
            if not self._is_cross_die_request(first_flit):
                continue
                
            # 只记录请求发起方Die的数据，避免重复记录
            if hasattr(first_flit, 'd2d_origin_die') and first_flit.d2d_origin_die != die_id:
                continue
                
            # 提取D2D信息
            d2d_info = self._extract_d2d_info(first_flit, last_flit, packet_id)
            if d2d_info:
                self.d2d_requests.append(d2d_info)
                
    def _is_cross_die_request(self, flit: Flit) -> bool:
        """检查flit是否为跨Die请求"""
        return (hasattr(flit, 'd2d_origin_die') and hasattr(flit, 'd2d_target_die') and
                flit.d2d_origin_die is not None and flit.d2d_target_die is not None and
                flit.d2d_origin_die != flit.d2d_target_die)
                
    def _extract_d2d_info(self, first_flit: Flit, last_flit: Flit, packet_id: int) -> Optional[D2DRequestInfo]:
        """从flit中提取D2D请求信息"""
        try:
            # 计算开始时间 - 优先使用req_start_cycle（tracker消耗开始）
            if hasattr(first_flit, 'req_start_cycle') and first_flit.req_start_cycle < float('inf'):
                start_time_ns = first_flit.req_start_cycle // self.network_frequency
            elif hasattr(first_flit, 'cmd_entry_noc_from_cake0_cycle') and first_flit.cmd_entry_noc_from_cake0_cycle < float('inf'):
                start_time_ns = first_flit.cmd_entry_noc_from_cake0_cycle // self.network_frequency
            else:
                start_time_ns = 0
                
            # 计算结束时间 - 根据请求类型选择合适的时间戳
            req_type = getattr(first_flit, 'req_type', 'unknown')
            if req_type == 'read':
                # 读请求：使用data_received_complete_cycle（读数据到达时tracker释放）
                if hasattr(last_flit, 'data_received_complete_cycle') and last_flit.data_received_complete_cycle < float('inf'):
                    end_time_ns = last_flit.data_received_complete_cycle // self.network_frequency
                else:
                    end_time_ns = start_time_ns
            elif req_type == 'write':
                # 写请求：使用write_complete_received_cycle（写完成响应到达时tracker释放）
                if hasattr(first_flit, 'write_complete_received_cycle') and first_flit.write_complete_received_cycle < float('inf'):
                    end_time_ns = first_flit.write_complete_received_cycle // self.network_frequency
                else:
                    end_time_ns = start_time_ns
            else:
                end_time_ns = start_time_ns
                
            latency_ns = end_time_ns - start_time_ns if end_time_ns > start_time_ns else 0
            
            # 计算数据量
            burst_length = getattr(first_flit, 'burst_length', 1)
            data_bytes = burst_length * self.FLIT_SIZE_BYTES
            
            return D2DRequestInfo(
                packet_id=packet_id,
                source_die=getattr(first_flit, 'd2d_origin_die', 0),
                target_die=getattr(first_flit, 'd2d_target_die', 1),
                source_node=getattr(first_flit, 'd2d_origin_node', 0),
                target_node=getattr(first_flit, 'd2d_target_node', 0),
                source_type=getattr(first_flit, 'd2d_origin_type', ''),
                target_type=getattr(first_flit, 'd2d_target_type', ''),
                req_type=getattr(first_flit, 'req_type', 'unknown'),
                burst_length=burst_length,
                data_bytes=data_bytes,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                latency_ns=latency_ns
            )
        except (AttributeError, KeyError, ValueError) as e:
            print(f"[D2D结果处理] 提取请求信息失败 packet_id={packet_id}: {e}")
            return None
        except Exception as e:
            print(f"[D2D结果处理] 未预期的错误 packet_id={packet_id}: {e}")
            raise
            
    def save_d2d_requests_csv(self, output_path: str):
        """
        保存D2D请求到CSV文件
        
        Args:
            output_path: 输出目录路径
        """
        os.makedirs(output_path, exist_ok=True)
        
        # 分别保存读请求和写请求
        read_requests = [req for req in self.d2d_requests if req.req_type == 'read']
        write_requests = [req for req in self.d2d_requests if req.req_type == 'write']
        
        # CSV文件头
        csv_header = [
            'packet_id', 'source_die', 'target_die', 'source_node', 'target_node',
            'source_type', 'target_type', 'burst_length', 'start_time_ns', 
            'end_time_ns', 'latency_ns', 'data_bytes'
        ]
        
        # 只有存在请求时才保存对应的CSV文件
        if read_requests:
            read_csv_path = os.path.join(output_path, 'd2d_read_requests.csv')
            self._save_requests_to_csv(read_requests, read_csv_path, csv_header)
            print(f"[D2D结果处理] 已保存 {len(read_requests)} 个读请求到 {read_csv_path}")
        else:
            print(f"[D2D结果处理] 无读请求数据，跳过读请求CSV文件生成")
        
        if write_requests:
            write_csv_path = os.path.join(output_path, 'd2d_write_requests.csv')
            self._save_requests_to_csv(write_requests, write_csv_path, csv_header)
            print(f"[D2D结果处理] 已保存 {len(write_requests)} 个写请求到 {write_csv_path}")
        else:
            print(f"[D2D结果处理] 无写请求数据，跳过写请求CSV文件生成")
        
    def _save_requests_to_csv(self, requests: List[D2DRequestInfo], file_path: str, header: List[str]):
        """保存请求列表到CSV文件"""
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                
                for req in requests:
                    writer.writerow([
                        req.packet_id, req.source_die, req.target_die, req.source_node, req.target_node,
                        req.source_type, req.target_type, req.burst_length, req.start_time_ns,
                        req.end_time_ns, req.latency_ns, req.data_bytes
                    ])
        except (IOError, OSError) as e:
            print(f"[D2D结果处理] 保存CSV文件失败 {file_path}: {e}")
            raise
                
    def calculate_d2d_bandwidth(self) -> D2DBandwidthStats:
        """计算D2D带宽统计"""
        stats = D2DBandwidthStats()
        
        # 按方向和类型分组请求
        groups = {
            ('0to1', 'read'): [],
            ('0to1', 'write'): [],
            ('1to0', 'read'): [],
            ('1to0', 'write'): []
        }
        
        for req in self.d2d_requests:
            direction = '0to1' if req.source_die == 0 else '1to0'
            key = (direction, req.req_type)
            if key in groups:
                groups[key].append(req)
        
        # 计算各组的带宽
        stats.die0_to_die1_read_bw, stats.die0_to_die1_read_bw_weighted = self._calculate_bandwidth_for_group(groups[('0to1', 'read')])
        stats.die0_to_die1_write_bw, stats.die0_to_die1_write_bw_weighted = self._calculate_bandwidth_for_group(groups[('0to1', 'write')])
        stats.die1_to_die0_read_bw, stats.die1_to_die0_read_bw_weighted = self._calculate_bandwidth_for_group(groups[('1to0', 'read')])
        stats.die1_to_die0_write_bw, stats.die1_to_die0_write_bw_weighted = self._calculate_bandwidth_for_group(groups[('1to0', 'write')])
        
        # 统计总数
        stats.total_read_requests = len(groups[('0to1', 'read')]) + len(groups[('1to0', 'read')])
        stats.total_write_requests = len(groups[('0to1', 'write')]) + len(groups[('1to0', 'write')])
        stats.total_bytes_transferred = sum(req.data_bytes for req in self.d2d_requests)
        
        self.d2d_stats = stats
        return stats
        
    def _calculate_bandwidth_for_group(self, requests: List[D2DRequestInfo]) -> Tuple[float, float]:
        """计算一组请求的带宽（非加权和加权）"""
        if not requests:
            return 0.0, 0.0
            
        # 计算总时间和总字节数
        if len(requests) == 1:
            # 单个请求，使用其延迟
            total_time_ns = max(requests[0].latency_ns, 1)  # 避免除零
        else:
            # 多个请求，计算整体时间跨度
            start_time = min(req.start_time_ns for req in requests)
            end_time = max(req.end_time_ns for req in requests)
            total_time_ns = max(end_time - start_time, 1)
            
        total_bytes = sum(req.data_bytes for req in requests)
        
        # 非加权带宽 (GB/s)
        unweighted_bw = (total_bytes / total_time_ns) if total_time_ns > 0 else 0.0
        
        # 加权带宽计算
        if len(requests) > 1:
            total_weighted_bw = 0.0
            total_weight = 0
            
            for req in requests:
                if req.latency_ns > 0:
                    weight = req.burst_length  # 使用burst_length作为权重
                    bandwidth = req.data_bytes / req.latency_ns if req.latency_ns > 0 else 0.0  # 修复零除错误
                    total_weighted_bw += bandwidth * weight
                    total_weight += weight
                    
            weighted_bw = (total_weighted_bw / total_weight) if total_weight > 0 else unweighted_bw
        else:
            weighted_bw = unweighted_bw
            
        return unweighted_bw, weighted_bw
        
    def generate_d2d_bandwidth_report(self, output_path: str):
        """生成D2D带宽报告，打印到屏幕并保存到txt文件"""
        stats = self.calculate_d2d_bandwidth()
        
        # 生成报告内容
        report_lines = [
            "=" * 50,
            "D2D带宽统计报告",
            "=" * 50,
            "",
            "Die0 → Die1:",
            f"  读带宽: {stats.die0_to_die1_read_bw:.2f} GB/s (加权: {stats.die0_to_die1_read_bw_weighted:.2f} GB/s)",
            f"  写带宽: {stats.die0_to_die1_write_bw:.2f} GB/s (加权: {stats.die0_to_die1_write_bw_weighted:.2f} GB/s)",
            "",
            "Die1 → Die0:",
            f"  读带宽: {stats.die1_to_die0_read_bw:.2f} GB/s (加权: {stats.die1_to_die0_read_bw_weighted:.2f} GB/s)",
            f"  写带宽: {stats.die1_to_die0_write_bw:.2f} GB/s (加权: {stats.die1_to_die0_write_bw_weighted:.2f} GB/s)",
            "",
            "总计:",
            f"  跨Die总带宽: {self._calculate_total_bandwidth(stats):.2f} GB/s",
            f"  跨Die加权总带宽: {self._calculate_total_weighted_bandwidth(stats):.2f} GB/s",
            f"  读请求数: {stats.total_read_requests}",
            f"  写请求数: {stats.total_write_requests}",
            f"  总传输字节数: {stats.total_bytes_transferred:,} bytes",
            "=" * 50
        ]
        
        # 打印到屏幕
        for line in report_lines:
            print(line)
            
        # 保存到文件
        os.makedirs(output_path, exist_ok=True)
        report_file = os.path.join(output_path, 'd2d_bandwidth_summary.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            for line in report_lines:
                f.write(line + '\n')
                
        print(f"\n[D2D结果处理] 带宽报告已保存到 {report_file}")
        
    def _calculate_total_bandwidth(self, stats: D2DBandwidthStats) -> float:
        """计算总带宽"""
        return (stats.die0_to_die1_read_bw + stats.die0_to_die1_write_bw +
                stats.die1_to_die0_read_bw + stats.die1_to_die0_write_bw)
                
    def _calculate_total_weighted_bandwidth(self, stats: D2DBandwidthStats) -> float:
        """计算加权总带宽"""
        return (stats.die0_to_die1_read_bw_weighted + stats.die0_to_die1_write_bw_weighted +
                stats.die1_to_die0_read_bw_weighted + stats.die1_to_die0_write_bw_weighted)
                
    def process_d2d_results(self, dies: Dict, output_path: str):
        """
        完整的D2D结果处理流程
        
        Args:
            dies: Die模型字典
            output_path: 输出目录路径
        """
        print("\n[D2D结果处理] 开始处理D2D系统结果...")
        
        # 1. 收集跨Die请求数据
        self.collect_cross_die_requests(dies)
        
        # 2. 保存请求到CSV文件
        self.save_d2d_requests_csv(output_path)
        
        # 3. 计算并输出带宽报告
        self.generate_d2d_bandwidth_report(output_path)
        
        print("[D2D结果处理] D2D结果处理完成!")