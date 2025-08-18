"""
D2D_Sys class for managing Die-to-Die transmission arbitration and bandwidth control.
Implements 2:1 arbitration between RN and SN interfaces and token bucket rate limiting for data.
"""

from collections import deque
from .flit import TokenBucket, Flit
import logging


class D2D_Sys:
    """
    D2D传输系统 - 管理每个D2D节点的RN/SN接口仲裁和AXI通道传输
    
    主要功能：
    1. 2:1轮询仲裁（RN和SN共享物理通道）
    2. AXI通道延迟模拟（类似network.send_flits的机制）
    3. 各通道独立的带宽控制（使用令牌桶）
    4. 统计信息收集
    """
    
    def __init__(self, node_pos: int, die_id: int, config):
        """
        初始化D2D传输系统
        
        Args:
            node_pos: 节点位置
            die_id: Die ID
            config: 配置对象
        """
        self.position = node_pos
        self.die_id = die_id
        self.config = config
        self.current_cycle = 0
        
        # 获取对端Die的RN和SN位置（在初始化时确定）
        if die_id == 0:
            # 当前是Die0，对端是Die1
            self.target_die_id = 1
            die1_positions = getattr(config, 'D2D_DIE1_POSITIONS', [])
            if not die1_positions:
                raise ValueError("D2D_DIE1_POSITIONS未配置")
            self.target_die_rn_pos = die1_positions[0]  # Die1的D2D_RN位置
            self.target_die_sn_pos = die1_positions[0]  # Die1的D2D_SN位置（通常同一个位置）
        else:
            # 当前是Die1，对端是Die0
            self.target_die_id = 0
            die0_positions = getattr(config, 'D2D_DIE0_POSITIONS', [])
            if not die0_positions:
                raise ValueError("D2D_DIE0_POSITIONS未配置")
            self.target_die_rn_pos = die0_positions[0]  # Die0的D2D_RN位置
            self.target_die_sn_pos = die0_positions[0]  # Die0的D2D_SN位置（通常同一个位置）
        
        # RN和SN的待发送队列
        self.rn_pending_queue = deque()
        self.sn_pending_queue = deque()
        
        # 轮询仲裁计数器
        self.arbiter_counter = 0
        
        # AXI通道状态管理（类似network.send_flits）
        self.axi_channels = {
            'AR': {
                'send_flits': {},  # {packet_id: [flits_with_timing]}
                'latency': getattr(config, 'D2D_AR_LATENCY', 10),
                'bandwidth_limiter': TokenBucket(
                    rate=getattr(config, 'D2D_AR_BANDWIDTH', 64) / config.NETWORK_FREQUENCY / config.FLIT_SIZE,
                    bucket_size=getattr(config, 'D2D_AR_BANDWIDTH', 64)
                )
            },
            'R': {
                'send_flits': {},
                'latency': getattr(config, 'D2D_R_LATENCY', 8),
                'bandwidth_limiter': TokenBucket(
                    rate=getattr(config, 'D2D_R_BANDWIDTH', 128) / config.NETWORK_FREQUENCY / config.FLIT_SIZE,
                    bucket_size=getattr(config, 'D2D_R_BANDWIDTH', 128)
                )
            },
            'AW': {
                'send_flits': {},
                'latency': getattr(config, 'D2D_AW_LATENCY', 10),
                'bandwidth_limiter': TokenBucket(
                    rate=getattr(config, 'D2D_AW_BANDWIDTH', 64) / config.NETWORK_FREQUENCY / config.FLIT_SIZE,
                    bucket_size=getattr(config, 'D2D_AW_BANDWIDTH', 64)
                )
            },
            'W': {
                'send_flits': {},
                'latency': getattr(config, 'D2D_W_LATENCY', 2),
                'bandwidth_limiter': TokenBucket(
                    rate=getattr(config, 'D2D_W_BANDWIDTH', 128) / config.NETWORK_FREQUENCY / config.FLIT_SIZE,
                    bucket_size=getattr(config, 'D2D_W_BANDWIDTH', 128)
                )
            },
            'B': {
                'send_flits': {},
                'latency': getattr(config, 'D2D_B_LATENCY', 8),
                'bandwidth_limiter': TokenBucket(
                    rate=getattr(config, 'D2D_B_BANDWIDTH', 32) / config.NETWORK_FREQUENCY / config.FLIT_SIZE,
                    bucket_size=getattr(config, 'D2D_B_BANDWIDTH', 32)
                )
            }
        }
        
        # 保留原有的数据传输令牌桶（用于统一的数据流控制）
        d2d_data_bw_limit = getattr(config, 'D2D_DATA_BW_LIMIT', 64)
        self.data_token_bucket = TokenBucket(
            rate=d2d_data_bw_limit / config.NETWORK_FREQUENCY / config.FLIT_SIZE,
            bucket_size=d2d_data_bw_limit
        )
        
        # 目标Die的接口（由D2DModel设置）
        self.target_die_interfaces = {}  # {die_id: {'rn': rn_interface, 'sn': sn_interface}}
        
        # 统计信息
        self.rn_transmit_count = 0
        self.sn_transmit_count = 0
        self.data_throttled_count = 0
        self.total_transmit_count = 0
        
        # AXI通道统计
        self.axi_channel_stats = {
            'AR': {'injected': 0, 'ejected': 0, 'throttled': 0},
            'R': {'injected': 0, 'ejected': 0, 'throttled': 0},
            'AW': {'injected': 0, 'ejected': 0, 'throttled': 0},
            'W': {'injected': 0, 'ejected': 0, 'throttled': 0},
            'B': {'injected': 0, 'ejected': 0, 'throttled': 0}
        }
        
        # 关联的RN和SN接口（由D2DModel设置）
        self.rn_interface = None
        self.sn_interface = None
        
        # 日志
        self.logger = logging.getLogger(f"D2D_Sys_Die{die_id}_Node{node_pos}")
        
        # 类似Network的send_flits结构，用于debug和trace
        # {packet_id: [axi_flits]}
        self.send_flits = {}
    
    def enqueue_rn(self, flit: Flit, target_die_id: int, delay: int, channel: str = None):
        """
        RN接口请求加入发送队列
        
        Args:
            flit: 要发送的flit
            target_die_id: 目标Die ID
            delay: 延迟（保持接口一致性）
            channel: 可选的AXI通道类型，如果不指定则自动判断
        """
        # 如果指定了channel，使用指定的；否则根据flit类型选择AXI通道
        if channel:
            channel_type = channel
        else:
            channel_type = self._determine_axi_channel(flit, "rn")
        
        self.rn_pending_queue.append({
            'flit': flit,
            'target_die_id': target_die_id,
            'channel_type': channel_type,
            'enqueue_cycle': self.current_cycle
        })
    
    def enqueue_sn(self, flit: Flit, target_die_id: int, delay: int):
        """
        SN接口请求加入发送队列
        
        Args:
            flit: 要发送的flit
            target_die_id: 目标Die ID
        """
        # 根据flit类型选择AXI通道
        channel_type = self._determine_axi_channel(flit, "sn")
        
        self.sn_pending_queue.append({
            'flit': flit,
            'target_die_id': target_die_id,
            'channel_type': channel_type,
            'enqueue_cycle': self.current_cycle
        })
    
    def step(self, current_cycle: int):
        """
        执行一个周期的D2D传输处理
        
        Args:
            current_cycle: 当前周期
        """
        self.current_cycle = current_cycle
        
        # 更新所有令牌桶
        self.data_token_bucket.refill(current_cycle)
        for channel in self.axi_channels.values():
            channel['bandwidth_limiter'].refill(current_cycle)
        
        # 处理AXI通道传输
        self.step_axi_channels()
        
        # 2:1轮询仲裁
        selected_item = None
        source = None
        attempts = 0
        
        while selected_item is None and attempts < 2:
            if self.arbiter_counter % 2 == 0:
                # 选择RN队列
                if self.rn_pending_queue:
                    selected_item = self.rn_pending_queue[0]
                    source = "RN"
            else:
                # 选择SN队列
                if self.sn_pending_queue:
                    selected_item = self.sn_pending_queue[0]
                    source = "SN"
            
            if selected_item:
                # 尝试注入到AXI通道
                if self._inject_to_axi_channel(selected_item):
                    # 注入成功，从队列中移除
                    if source == "RN":
                        self.rn_pending_queue.popleft()
                        self.rn_transmit_count += 1
                    else:
                        self.sn_pending_queue.popleft()
                        self.sn_transmit_count += 1
                    
                    self.total_transmit_count += 1
                    break
                else:
                    # 注入失败（带宽不足），等待下个周期
                    return
            
            # 尝试另一个队列
            self.arbiter_counter = (self.arbiter_counter + 1) % 2
            attempts += 1
        
        # 更新仲裁计数器
        if selected_item:
            self.arbiter_counter = (self.arbiter_counter + 1) % 2
    
    def _determine_axi_channel(self, flit: Flit, interface_type: str) -> str:
        """
        根据flit类型和接口类型确定应该使用的AXI通道
        
        Args:
            flit: 待传输的flit
            interface_type: 接口类型 ("rn" 或 "sn")
            
        Returns:
            str: AXI通道类型 ("AR", "R", "AW", "W", "B")
        """
        
        # 添加调试信息
        flit_type = getattr(flit, 'flit_type', 'None')
        req_type = getattr(flit, 'req_type', 'None')
        has_write_data = hasattr(flit, 'write_data')
        packet_id = getattr(flit, 'packet_id', '?')
        
        print(f"[D2D_Sys] 通道判断: packet_id={packet_id}, flit_type={flit_type}, req_type={req_type}, has_write_data={has_write_data}")
        
        # 数据flit - 优先判断
        if hasattr(flit, 'flit_type') and flit.flit_type == 'data':
            # 根据是否为写数据判断
            if hasattr(flit, 'write_data') or (hasattr(flit, 'req_type') and flit.req_type == "write"):
                print(f"[D2D_Sys] → W通道 (写数据)")
                return "W"  # 写数据通道
            else:
                print(f"[D2D_Sys] → R通道 (读数据)")
                return "R"  # 读数据通道
        
        # 请求flit
        if hasattr(flit, 'req_type'):
            if flit.req_type == "read":
                print(f"[D2D_Sys] → AR通道 (读请求)")
                return "AR"  # 读地址通道
            elif flit.req_type == "write":
                print(f"[D2D_Sys] → AW通道 (写请求)")
                return "AW"  # 写地址通道
        
        # 响应flit
        if hasattr(flit, 'rsp_type'):
            if flit.rsp_type in ['datasend', 'read_data']:
                return "R"  # 读数据通道
            elif flit.rsp_type in ['write_ack', 'write_response']:
                return "B"  # 写响应通道
        
        # 数据flit
        if hasattr(flit, 'flit_type'):
            if flit.flit_type == 'data':
                # 根据是否为写数据判断
                if hasattr(flit, 'write_data') or (hasattr(flit, 'req_type') and flit.req_type == "write"):
                    return "W"  # 写数据通道
                else:
                    return "R"  # 读数据通道
        
        # 默认根据接口类型判断
        if interface_type == "rn":
            return "AR"  # RN发起的请求默认为读地址
        else:
            return "R"   # SN返回的默认为数据
    
    def _is_data_flit(self, flit: Flit) -> bool:
        """
        判断是否为数据flit（需要带宽限制）
        
        Args:
            flit: 待判断的flit
            
        Returns:
            bool: 是否为数据flit
        """
        # 数据flit的判断逻辑
        if hasattr(flit, 'flit_type') and flit.flit_type == 'data':
            return True
        
        # 响应中的数据部分
        if hasattr(flit, 'rsp_type') and flit.rsp_type in ['datasend', 'read_data']:
            return True
        
        return False
    
    def _inject_to_axi_channel(self, item: dict) -> bool:
        """
        将flit注入到指定的AXI通道
        
        Args:
            item: 包含flit和传输信息的字典
            
        Returns:
            bool: 是否成功注入
        """
        flit = item['flit']
        target_die_id = item['target_die_id']
        channel_type = item['channel_type']
        
        channel = self.axi_channels[channel_type]
        
        # 检查通道带宽限制
        if not channel['bandwidth_limiter'].consume(1):
            self.axi_channel_stats[channel_type]['throttled'] += 1
            return False
        
        # 创建AXI专用flit副本，避免修改原始flit
        from .flit import Flit
        axi_flit = Flit(
            source=flit.source,
            destination=flit.destination,
            path=getattr(flit, 'path', [])
        )
        
        
        # 复制关键属性到AXI flit（使用新的d2d属性）
        for attr in ['req_type', 'rsp_type', 'flit_type',
                     'd2d_origin_die', 'd2d_origin_node', 'd2d_origin_type',
                     'd2d_target_die', 'd2d_target_node', 'd2d_target_type',
                     'req_attr', 'packet_id', 'flit_id', 'burst_length']:
            if hasattr(flit, attr):
                setattr(axi_flit, attr, getattr(flit, attr))
        
        
        # AXI传输无需修改d2d属性，已在复制时继承
        # 通道类型将在后续路由时处理
        
        
        # 根据通道类型设置正确的阶段信息
        if channel_type in ['AR', 'AW', 'W']:
            # 请求/写通道：阶段2，从D2D_SN发送到D2D_RN
            axi_flit.source_type = "d2d_sn_0"  # D2D_SN类型
            axi_flit.destination_type = "d2d_rn_0"  # D2D_RN类型
            # 设置源和目标位置
            axi_flit.source = self.position  # 当前D2D_SN的位置
            axi_flit.destination = self.target_die_rn_pos  # 目标Die的D2D_RN位置
            
        elif channel_type in ['R', 'B']:
            # 响应通道：阶段5，从D2D_RN发送回D2D_SN
            axi_flit.source_type = "d2d_rn_0"  # D2D_RN类型
            axi_flit.destination_type = "d2d_sn_0"  # D2D_SN类型
            # 设置源和目标位置
            axi_flit.source = self.position  # 当前D2D_RN的位置
            axi_flit.destination = self.target_die_sn_pos  # 目标Die的D2D_SN位置
        
        # 设置AXI传输专用信息
        axi_flit.flit_position = f"AXI_{channel_type}"
        axi_flit.current_position = f"Die{self.die_id}_AXI"
        axi_flit.axi_end_cycle = self.current_cycle + channel['latency']
        axi_flit.axi_start_cycle = self.current_cycle
        
        # 加入通道的send_flits，包含完成时间信息
        packet_id = flit.packet_id
        if packet_id not in channel['send_flits']:
            channel['send_flits'][packet_id] = []
        
        # 创建传输条目，包含AXI flit和完成时间
        transmission_entry = {
            'flit': axi_flit,
            'end_cycle': self.current_cycle + channel['latency']
        }
        channel['send_flits'][packet_id].append(transmission_entry)
        
        # 同时更新D2D_Sys的send_flits结构用于debug
        if packet_id not in self.send_flits:
            self.send_flits[packet_id] = []
        self.send_flits[packet_id].append(axi_flit)
        
        self.axi_channel_stats[channel_type]['injected'] += 1
        return True
    
    def step_axi_channels(self):
        """
        更新所有AXI通道的传输状态
        """
        for channel_type, channel in self.axi_channels.items():
            self._step_single_axi_channel(channel_type, channel)
    
    def _step_single_axi_channel(self, channel_type: str, channel: dict):
        """
        更新单个AXI通道的传输状态
        
        Args:
            channel_type: 通道类型
            channel: 通道配置
        """
        arrived_flits = []
        
        for packet_id, transmission_entries in list(channel['send_flits'].items()):
            remaining_entries = []
            for entry in transmission_entries:
                if entry['end_cycle'] <= self.current_cycle:
                    # flit已到达
                    arrived_flits.append(entry['flit'])
                else:
                    # flit仍在传输中
                    remaining_entries.append(entry)
            
            if remaining_entries:
                channel['send_flits'][packet_id] = remaining_entries
            else:
                del channel['send_flits'][packet_id]
        
        # 处理到达的flit
        for flit in arrived_flits:
            self._deliver_arrived_flit(flit)
            self.axi_channel_stats[channel_type]['ejected'] += 1
            
            # 从send_flits中移除已完成的flit
            packet_id = flit.packet_id
            if packet_id in self.send_flits:
                try:
                    self.send_flits[packet_id].remove(flit)
                    if not self.send_flits[packet_id]:  # 如果列表为空，删除key
                        del self.send_flits[packet_id]
                except ValueError:
                    pass  # flit不在列表中，忽略
    
    def _deliver_arrived_flit(self, flit: Flit):
        """
        处理到达目标的flit
        
        Args:
            flit: 到达的flit
        """
        # 确定目标Die ID：
        # 需要根据flit的传输方向判断目标
        
        # 检查是否为AXI传输中的flit
        is_axi_flit = hasattr(flit, 'flit_position') and flit.flit_position and flit.flit_position.startswith('AXI_')
        
        if is_axi_flit:
            # AXI传输中的flit：
            # AXI_AR, AXI_AW: 请求类型，前往d2d_target_die
            # AXI_R, AXI_B: 响应类型，返回d2d_origin_die
            if flit.flit_position in ['AXI_AR', 'AXI_AW', 'AXI_W']:
                target_die_id = flit.d2d_target_die
            else:  # AXI_R, AXI_B
                target_die_id = flit.d2d_origin_die  # 响应返回原始请求者Die
        elif hasattr(flit, 'req_type') and flit.req_type:
            # 普通请求flit：前往目标Die
            target_die_id = flit.d2d_target_die
        else:
            # 普通响应/数据flit：返回原始源Die
            target_die_id = flit.d2d_origin_die
        
        # 获取目标接口
        if target_die_id is None or target_die_id not in self.target_die_interfaces:
            # 添加更详细的调试信息
            flit_info = f"flit_position={getattr(flit, 'flit_position', 'None')}, d2d_origin_die={getattr(flit, 'd2d_origin_die', 'None')}, d2d_target_die={getattr(flit, 'd2d_target_die', 'None')}, packet_id={getattr(flit, 'packet_id', 'None')}"
            self.logger.error(f"目标Die {target_die_id} 的接口未设置，flit信息: {flit_info}")
            return
        
        target_interfaces = self.target_die_interfaces[target_die_id]
        
        # 根据flit类型选择目标接口
        if is_axi_flit:
            # AXI传输：根据AXI通道类型判断
            if flit.flit_position in ['AXI_AR', 'AXI_AW', 'AXI_W']:
                # 请求/写通道 -> 发送到目标Die的RN（第二阶段：D2D_SN → D2D_RN）
                target_interface = target_interfaces.get('rn')
            else:  # AXI_R, AXI_B
                # 响应通道 -> 发送到目标Die的SN（第五阶段：D2D_RN → D2D_SN）
                target_interface = target_interfaces.get('sn')
        elif hasattr(flit, 'req_type'):
            # 普通请求flit -> 发送到目标Die的RN
            target_interface = target_interfaces.get('rn')
        else:
            # 普通响应/数据flit -> 发送到目标Die的SN
            target_interface = target_interfaces.get('sn')
        
        if target_interface:
            # 调度跨Die接收（使用当前周期，因为已经经过了AXI延迟）
            target_interface.schedule_cross_die_receive(flit, self.current_cycle)
            
            # 如果是数据传输完成，通知源接口释放tracker
            if self._is_data_flit(flit) and hasattr(flit, "d2d_origin_die"):
                # 这是跨Die数据返回，需要通知源D2D_RN
                source_die_id = self.die_id  # 当前Die是源Die
                source_interfaces = self.target_die_interfaces.get(source_die_id, {})
                source_rn = source_interfaces.get('rn')
                
                if source_rn and hasattr(source_rn, "notify_cross_die_transfer_complete"):
                    source_rn.notify_cross_die_transfer_complete(flit)
        else:
            self.logger.error(f"无法找到合适的目标接口")
    
    def _transmit(self, item: dict):
        """
        执行实际的跨Die传输（兼容旧接口，已被AXI通道机制替代）
        
        Args:
            item: 包含flit和传输信息的字典
        """
        # 这个方法保留用于兼容性，实际传输已经由AXI通道机制处理
        self.logger.warning("使用了已被AXI通道机制替代的_transmit方法")
    
    def get_statistics(self) -> dict:
        """
        获取D2D_Sys的统计信息
        
        Returns:
            dict: 统计信息字典
        """
        # 计算AXI通道中的传输中flit数量
        axi_in_transit = {}
        for channel_type, channel in self.axi_channels.items():
            total_flits = sum(len(flits) for flits in channel['send_flits'].values())
            axi_in_transit[channel_type] = total_flits
        
        stats = {
            'rn_transmit_count': self.rn_transmit_count,
            'sn_transmit_count': self.sn_transmit_count,
            'total_transmit_count': self.total_transmit_count,
            'data_throttled_count': self.data_throttled_count,
            'rn_queue_length': len(self.rn_pending_queue),
            'sn_queue_length': len(self.sn_pending_queue),
            'data_tokens_available': self.data_token_bucket.tokens,
            'arbiter_state': 'RN' if self.arbiter_counter % 2 == 0 else 'SN',
            'axi_channel_stats': self.axi_channel_stats.copy(),
            'axi_in_transit': axi_in_transit,
            'axi_bandwidth_tokens': {
                channel_type: channel['bandwidth_limiter'].tokens
                for channel_type, channel in self.axi_channels.items()
            }
        }
        return stats