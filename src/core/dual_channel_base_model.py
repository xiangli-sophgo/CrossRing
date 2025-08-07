"""
DualChannelBaseModel class for NoC simulation.
Extends BaseModel class to support dual-channel data transmission with independent arbitration.
"""

from .base_model import BaseModel
from src.utils.components.dual_channel_network import DualChannelDataNetwork
from src.utils.components.dual_channel_ip_interface import DualChannelIPInterface
from src.utils.channel_selector import DefaultChannelSelector
from collections import deque, defaultdict
import logging


class DualChannelBaseModel(BaseModel):
    """双通道基础模型，支持数据双通道传输"""
    
    def __init__(self, *args, **kwargs):
        # 调用父类初始化
        super().__init__(*args, **kwargs)
        
        # 设置通道选择器（在父类初始化后，config已可用）
        self.channel_selector = self._create_channel_selector()
        
        logging.info("DualChannelBaseModel initialized with dual-channel data support")
    
    def _create_channel_selector(self):
        """创建通道选择器"""
        strategy = getattr(self.config, 'DATA_CHANNEL_SELECT_STRATEGY', 'hash_based')
        return DefaultChannelSelector(strategy)
    
    def _create_networks(self):
        """重写网络创建方法，使用双通道数据网络"""
        # 先创建标准的req和rsp网络
        super()._create_networks()
        
        # 替换data网络为双通道版本
        self.data_network = DualChannelDataNetwork(
            self.config, self.adjacency_matrix, "dual_channel_data"
        )
        
        # 更新通道选择器的网络引用
        self.channel_selector.network = self.data_network
    
    def _create_ip_interfaces(self):
        """重写IP接口创建，使用双通道接口"""
        self.ip_interfaces = {}
        
        # 获取IP位置信息
        ip_positions = self._get_ip_positions()
        
        for ip_type, positions in ip_positions.items():
            for pos in positions:
                self.ip_interfaces[(ip_type, pos)] = DualChannelIPInterface(
                    ip_type, pos, self.config,
                    self.req_network, self.rsp_network, self.data_network,
                    self.node, self.routes, self.channel_selector
                )
    
    def _get_ip_positions(self):
        """获取IP位置信息"""
        # 这个方法需要根据具体的拓扑结构实现
        # 这里提供一个基础实现
        ip_positions = defaultdict(list)
        
        # GDMA positions
        for i in range(getattr(self.config, 'NUM_GDMA', 0)):
            ip_positions['gdma'].append(i)
            
        # SDMA positions
        for i in range(getattr(self.config, 'NUM_SDMA', 0)):
            ip_positions['sdma'].append(i)
            
        # CDMA positions
        for i in range(getattr(self.config, 'NUM_CDMA', 0)):
            ip_positions['cdma'].append(i)
            
        # DDR positions
        for i in range(getattr(self.config, 'NUM_DDR', 0)):
            ip_positions['ddr'].append(i)
            
        # L2M positions
        for i in range(getattr(self.config, 'NUM_L2M', 0)):
            ip_positions['l2m'].append(i)
            
        return ip_positions
    
    def _inject_queue_arbitration(self, network=None, positions_list=None, network_type=None):
        """重写IQ仲裁，包含数据双通道仲裁"""
        # 如果有参数传入，说明是原有req和rsp网络仲裁
        if network is not None and positions_list is not None and network_type is not None:
            # 调用父类的原有仲裁方法
            super()._inject_queue_arbitration(network, positions_list, network_type)
        else:
            # 数据双通道仲裁
            self._inject_queue_arbitration_dual_channel()
    
    def _inject_queue_arbitration_dual_channel(self):
        """数据双通道IQ仲裁"""
        for channel_id in [0, 1]:
            iq_buffer = self.data_network.get_iq_channel_buffer(channel_id)
            inject_queues = self.data_network.get_inject_queues(channel_id)
            
            for ip_type in iq_buffer:
                for ip_pos in iq_buffer[ip_type]:
                    fifo = iq_buffer[ip_type][ip_pos]
                    if fifo:
                        flit = fifo.popleft()
                        
                        # 计算注入方向
                        direction = self._calculate_injection_direction(flit)
                        
                        # 获取目标节点ID
                        node_id = self._get_node_id_from_position(ip_pos, ip_type)
                        
                        # 确保目标队列存在
                        if node_id not in inject_queues[direction]:
                            inject_queues[direction][node_id] = deque()
                        
                        inject_queues[direction][node_id].append(flit)
                        flit.flit_position = f"inject_{direction}_ch{channel_id}"
    
    def _calculate_injection_direction(self, flit):
        """计算flit的注入方向"""
        try:
            source = flit.current_position
            if len(flit.path) > flit.path_index + 1:
                next_pos = flit.path[flit.path_index + 1]
                
                # 判断方向
                if hasattr(self.config, 'NUM_COL'):
                    col_diff = (next_pos % self.config.NUM_COL) - (source % self.config.NUM_COL)
                    row_diff = (next_pos // self.config.NUM_COL) - (source // self.config.NUM_COL)
                    
                    if col_diff > 0:
                        return "TR"  # Turn Right
                    elif col_diff < 0:
                        return "TL"  # Turn Left
                    elif row_diff > 0:
                        return "TD"  # Turn Down
                    elif row_diff < 0:
                        return "TU"  # Turn Up
                    else:
                        return "EQ"  # Eject
                else:
                    # 简单的方向判断逻辑
                    if next_pos > source:
                        return "TR"
                    else:
                        return "TL"
            else:
                return "EQ"  # 到达目的地，弹出
                
        except Exception as e:
            logging.warning(f"Error calculating injection direction for flit {flit}: {e}")
            return "EQ"
    
    def _get_node_id_from_position(self, ip_pos, ip_type):
        """从IP位置获取节点ID"""
        # 这个方法需要根据具体的拓扑映射实现
        # 这里提供一个基础实现
        try:
            if hasattr(self.config, 'NUM_COL'):
                return ip_pos
            else:
                return ip_pos
        except:
            return 0
    
    def Ring_Bridge_arbitration(self, network):
        """重写RB仲裁，包含数据双通道"""
        # 如果是双通道数据网络，使用双通道仲裁
        if hasattr(network, 'get_inject_queues'):
            # 这是双通道数据网络，进行双通道仲裁
            for channel_id in [0, 1]:
                for direction in ["TL", "TR", "TU", "TD", "EQ"]:
                    self._ring_bridge_arbitration_dual_channel(direction, channel_id)
        else:
            # 原有req和rsp网络RB仲裁
            super().Ring_Bridge_arbitration(network)
    
    def _ring_bridge_arbitration_dual_channel(self, direction, channel_id):
        """数据双通道RB仲裁"""
        try:
            inject_queues = self.data_network.get_inject_queues(channel_id)[direction]
            ring_bridge = self.data_network.get_ring_bridge(channel_id)[direction]
            
            # 执行round-robin仲裁
            self._execute_rb_arbitration_for_channel(inject_queues, ring_bridge, channel_id, direction)
        except Exception as e:
            logging.warning(f"RB arbitration failed for direction {direction}, channel {channel_id}: {e}")
    
    def _execute_rb_arbitration_for_channel(self, source_queues, target_queues, channel_id, direction):
        """执行指定通道的RB仲裁"""
        if not source_queues:
            return
            
        # 简单的round-robin实现
        for node_id, queue in source_queues.items():
            if queue:
                # 确保目标队列存在
                if node_id not in target_queues:
                    target_queues[node_id] = deque()
                
                # 移动flit
                flit = queue.popleft()
                target_queues[node_id].append(flit)
                flit.flit_position = f"RB_{direction}_ch{channel_id}"
    
    def Eject_Queue_arbitration(self, network, flit_type):
        """重写EQ仲裁，包含数据双通道"""
        # 如果是双通道数据网络，使用双通道仲裁
        if hasattr(network, 'get_eject_queues') and flit_type == "data":
            # 这是双通道数据网络，进行双通道仲裁
            for channel_id in [0, 1]:
                for direction in ["TU", "TD"]:
                    self._eject_queue_arbitration_dual_channel(direction, channel_id)
        else:
            # 原有req和rsp网络EQ仲裁
            super().Eject_Queue_arbitration(network, flit_type)
    
    def _eject_queue_arbitration_dual_channel(self, direction, channel_id):
        """数据双通道EQ仲裁"""
        try:
            eject_queues = self.data_network.get_eject_queues(channel_id)[direction]
            eq_buffer = self.data_network.get_eq_channel_buffer(channel_id)
            
            # 执行EQ仲裁
            self._execute_eq_arbitration_for_channel(eject_queues, eq_buffer, channel_id, direction)
        except Exception as e:
            logging.warning(f"EQ arbitration failed for direction {direction}, channel {channel_id}: {e}")
    
    def _execute_eq_arbitration_for_channel(self, source_queues, target_buffers, channel_id, direction):
        """执行指定通道的EQ仲裁"""
        if not source_queues:
            return
            
        for node_id, queue in source_queues.items():
            if queue:
                flit = queue.popleft()
                
                # 确定目标IP类型和位置
                ip_type, ip_pos = self._get_ip_info_from_flit(flit)
                
                # 确保目标缓冲区存在
                if ip_type not in target_buffers:
                    continue
                if ip_pos not in target_buffers[ip_type]:
                    continue
                    
                target_buffers[ip_type][ip_pos].append(flit)
                flit.flit_position = f"EQ_ch{channel_id}"
    
    def _get_ip_info_from_flit(self, flit):
        """从flit获取IP类型和位置信息"""
        try:
            # 从flit的目标信息获取IP信息
            ip_type = getattr(flit, 'destination_type', 'gdma')
            ip_pos = getattr(flit, 'destination', 0)
            
            # 清理IP类型名称
            if '_' in ip_type:
                ip_type = ip_type.split('_')[0]
                
            return ip_type, ip_pos
        except Exception as e:
            logging.warning(f"Error getting IP info from flit {flit}: {e}")
            return 'gdma', 0
    
    
    def _move_dual_channel_pre_to_fifo(self):
        """处理双通道pre缓冲区到正式fifo的移动"""
        for channel_id in [0, 1]:
            try:
                # IQ pre到fifo移动
                iq_buffer = self.data_network.get_iq_channel_buffer(channel_id)
                iq_buffer_pre = self.data_network.get_iq_channel_buffer_pre(channel_id)
                
                for ip_type in iq_buffer:
                    for ip_pos in iq_buffer[ip_type]:
                        if ip_pos in iq_buffer_pre[ip_type] and iq_buffer_pre[ip_type][ip_pos] is not None:
                            if len(iq_buffer[ip_type][ip_pos]) < getattr(self.config, "IQ_CH_FIFO_DEPTH", 8):
                                iq_buffer[ip_type][ip_pos].append(iq_buffer_pre[ip_type][ip_pos])
                                iq_buffer_pre[ip_type][ip_pos] = None
                
                # EQ pre到fifo移动
                eq_buffer = self.data_network.get_eq_channel_buffer(channel_id)
                eq_buffer_pre = self.data_network.get_eq_channel_buffer_pre(channel_id)
                
                for ip_type in eq_buffer:
                    for ip_pos in eq_buffer[ip_type]:
                        if ip_pos in eq_buffer_pre[ip_type] and eq_buffer_pre[ip_type][ip_pos] is not None:
                            if len(eq_buffer[ip_type][ip_pos]) < getattr(self.config, "EQ_CH_FIFO_DEPTH", 8):
                                eq_buffer[ip_type][ip_pos].append(eq_buffer_pre[ip_type][ip_pos])
                                eq_buffer_pre[ip_type][ip_pos] = None
                                
            except Exception as e:
                logging.warning(f"Error moving pre to fifo for channel {channel_id}: {e}")
    
    def get_dual_channel_statistics(self):
        """获取双通道统计信息"""
        try:
            if hasattr(self.data_network, 'get_channel_stats'):
                channel_stats = self.data_network.get_channel_stats()
            else:
                channel_stats = {"ch0": {"inject_count": 0, "eject_count": 0, "latency": []}, 
                               "ch1": {"inject_count": 0, "eject_count": 0, "latency": []}}
            
            if hasattr(self.data_network, 'dual_channel_stats'):
                dual_stats = self.data_network.dual_channel_stats
            else:
                dual_stats = {"ch0": {"inject_count": 0, "eject_count": 0, "latency": []}, 
                            "ch1": {"inject_count": 0, "eject_count": 0, "latency": []}}
            
            return {
                "channel_stats": channel_stats,
                "total_inject_ch0": dual_stats["ch0"]["inject_count"],
                "total_inject_ch1": dual_stats["ch1"]["inject_count"],
                "total_eject_ch0": dual_stats["ch0"]["eject_count"],
                "total_eject_ch1": dual_stats["ch1"]["eject_count"],
                "avg_latency_ch0": sum(dual_stats["ch0"]["latency"]) / len(dual_stats["ch0"]["latency"]) if dual_stats["ch0"]["latency"] else 0,
                "avg_latency_ch1": sum(dual_stats["ch1"]["latency"]) / len(dual_stats["ch1"]["latency"]) if dual_stats["ch1"]["latency"] else 0,
            }
        except Exception as e:
            # 返回默认统计信息
            return {
                "channel_stats": {"ch0": {}, "ch1": {}},
                "total_inject_ch0": 0,
                "total_inject_ch1": 0,
                "total_eject_ch0": 0,
                "total_eject_ch1": 0,
                "avg_latency_ch0": 0,
                "avg_latency_ch1": 0,
            }
    
    def print_dual_channel_summary(self):
        """打印双通道总结信息"""
        stats = self.get_dual_channel_statistics()
        
        print("\n=== Dual Channel Data Network Summary ===")
        print(f"Channel 0 - Injected: {stats['total_inject_ch0']}, Ejected: {stats['total_eject_ch0']}")
        print(f"Channel 1 - Injected: {stats['total_inject_ch1']}, Ejected: {stats['total_eject_ch1']}")
        print(f"Average Latency - CH0: {stats['avg_latency_ch0']:.2f}, CH1: {stats['avg_latency_ch1']:.2f}")
        
        total_inject = stats['total_inject_ch0'] + stats['total_inject_ch1']
        if total_inject > 0:
            ch0_ratio = stats['total_inject_ch0'] / total_inject * 100
            ch1_ratio = stats['total_inject_ch1'] / total_inject * 100
            print(f"Channel Distribution - CH0: {ch0_ratio:.1f}%, CH1: {ch1_ratio:.1f}%")
        
        print("==========================================\n")