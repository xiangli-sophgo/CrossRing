"""
IPInterface class for NoC simulation.
Handles IP interface functionality including frequency conversion, OSTD/Data-buffer behavior,
and network interaction through inject/eject FIFOs.
"""

from __future__ import annotations
import numpy as np
from collections import deque, defaultdict
from config.config import CrossRingConfig
from .flit import Flit, TokenBucket
from .node import Node
import logging


# IPInterface的Ring支持扩展
class RingIPInterface:
    """
    为IPInterface添加Ring支持的Mixin类
    """

    def __init__(self, ip_type, ip_pos, config, req_network, rsp_network, data_network, node, routes, ring_mode=False):
        # 调用原有的IPInterface初始化
        super().__init__(ip_type, ip_pos, config, req_network, rsp_network, data_network, node, routes)

        self.ring_mode = ring_mode
        self.ring_model = None  # 将由Ring模型设置
        if ring_mode:
            self._init_ring_specific_features()

    def _init_ring_specific_features(self):
        """初始化Ring特有的IPInterface功能"""
        # Ring模式下的特殊配置
        self.ring_config = {"adaptive_injection": True, "load_balancing": True, "etag_enabled": True, "itag_enabled": True}


class IPInterface:
    """
    模拟一个 IP 的频率转化和 OSTD/Data-buffer 行为:
      - inject_fifo: 1GHz 入队
      - l2h_fifo: 1→2GHz 上转换 FIFO（深度可调）
      - 和 network.IQ_channel_buffer 对接（2GHz）
      - network.EQ_channel_buffer → h2l_fifo → eject_fifo → IP（1GHz）
    """

    def __init__(
        self,
        ip_type: str,
        ip_pos: int,
        config: CrossRingConfig,
        req_network,
        rsp_network,
        data_network,
        node: Node,
        routes: dict,
        ip_id: int = None,
    ):
        self.current_cycle = 0
        self.ip_type = ip_type
        self.ip_pos = ip_pos
        self.ip_id = ip_id if ip_id is not None else 0  # IP在其类型中的唯一ID
        self.config = config
        self.req_network = req_network
        self.rsp_network = rsp_network
        self.data_network = data_network
        self.node = node
        self.routes = routes
        self.read_retry_num_stat = 0
        self.write_retry_num_stat = 0
        self.req_wait_cycles_h = 0
        self.req_wait_cycles_v = 0
        self.rsp_wait_cycles_h = 0
        self.rsp_wait_cycles_v = 0
        self.data_wait_cycles_h = 0
        self.data_wait_cycles_v = 0
        self.req_cir_h_num, self.req_cir_v_num = 0, 0
        self.rsp_cir_h_num, self.rsp_cir_v_num = 0, 0
        self.data_cir_h_num, self.data_cir_v_num = 0, 0

        # 验证FIFO深度配置
        l2h_depth = config.IP_L2H_FIFO_DEPTH
        h2l_h_depth = config.IP_H2L_H_FIFO_DEPTH
        h2l_l_depth = config.IP_H2L_L_FIFO_DEPTH

        self.networks = {
            "req": {
                "network": req_network,
                "send_flits": req_network.send_flits,
                "inject_fifo": deque(),  # 1GHz域 - IP核产生的请求
                "l2h_fifo_pre": None,  # 1GHz → 2GHz 预缓冲
                "h2l_fifo_h_pre": None,  # 2GHz → H2L_H 预缓冲
                "h2l_fifo_l_pre": None,  # H2L_H → H2L_L 预缓冲
                "l2h_fifo": deque(maxlen=l2h_depth),  # 1GHz → 2GHz 转换FIFO
                "h2l_fifo_h": deque(maxlen=h2l_h_depth),  # 2GHz域 高频FIFO
                "h2l_fifo_l": deque(maxlen=h2l_l_depth),  # 1GHz域 低频FIFO
            },
            "rsp": {
                "network": rsp_network,
                "send_flits": rsp_network.send_flits,
                "inject_fifo": deque(),
                "l2h_fifo_pre": None,
                "h2l_fifo_h_pre": None,
                "h2l_fifo_l_pre": None,
                "l2h_fifo": deque(maxlen=l2h_depth),
                "h2l_fifo_h": deque(maxlen=h2l_h_depth),
                "h2l_fifo_l": deque(maxlen=h2l_l_depth),
            },
            "data": {
                "network": data_network,
                "send_flits": data_network.send_flits,
                "inject_fifo": deque(),
                "l2h_fifo_pre": None,
                "h2l_fifo_h_pre": None,
                "h2l_fifo_l_pre": None,
                "l2h_fifo": deque(maxlen=l2h_depth),
                "h2l_fifo_h": deque(maxlen=h2l_h_depth),
                "h2l_fifo_l": deque(maxlen=h2l_l_depth),
            },
        }

        # 根据IP类型设置双向带宽限制令牌桶
        if ip_type.startswith("ddr"):
            # DDR通道限速
            tx_limit = rx_limit = self.config.DDR_BW_LIMIT
            self.tx_token_bucket = TokenBucket(
                rate=tx_limit / self.config.NETWORK_FREQUENCY / self.config.FLIT_SIZE,
                bucket_size=tx_limit,
            )
            self.rx_token_bucket = TokenBucket(
                rate=rx_limit / self.config.NETWORK_FREQUENCY / self.config.FLIT_SIZE,
                bucket_size=rx_limit,
            )
        elif ip_type.startswith("l2m"):
            # L2M通道限速
            tx_limit = rx_limit = self.config.L2M_BW_LIMIT
            self.tx_token_bucket = TokenBucket(
                rate=tx_limit / self.config.NETWORK_FREQUENCY / self.config.FLIT_SIZE,
                bucket_size=tx_limit,
            )
            self.rx_token_bucket = TokenBucket(
                rate=rx_limit / self.config.NETWORK_FREQUENCY / self.config.FLIT_SIZE,
                bucket_size=rx_limit,
            )
        elif ip_type[:4] in ("sdma", "gdma", "cdma"):
            # DMA通道（SDMA/GDMA/CDMA）限速
            ip_prefix = ip_type[:4].upper()
            tx_limit = rx_limit = getattr(self.config, f"{ip_prefix}_BW_LIMIT", 128)
            self.tx_token_bucket = TokenBucket(
                rate=tx_limit / self.config.FLIT_SIZE,
                bucket_size=tx_limit,
            )
            self.rx_token_bucket = TokenBucket(
                rate=rx_limit / self.config.FLIT_SIZE,
                bucket_size=rx_limit,
            )
        else:
            # 默认不限速
            self.tx_token_bucket = None
            self.rx_token_bucket = None

    def enqueue(self, flit: Flit, network_type: str, retry=False):
        """IP 核把flit丢进对应网络的 inject_fifo"""
        if retry:
            self.networks[network_type]["inject_fifo"].appendleft(flit)
        else:
            self.networks[network_type]["inject_fifo"].append(flit)
        flit.flit_position = "IP_inject"
        
        # 检查是否为新请求并记录统计
        is_new_request = (network_type == "req" and 
                         not self.networks[network_type]["send_flits"][flit.packet_id] and 
                         not retry)
        
        if is_new_request and hasattr(flit, "req_type"):
            # 判断是否为跨Die请求
            is_cross_die = (hasattr(flit, "d2d_target_die") and 
                           hasattr(flit, "d2d_origin_die") and 
                           flit.d2d_target_die != flit.d2d_origin_die)
            
            # 记录请求统计到D2D模型
            d2d_model = getattr(self.req_network, 'd2d_model', None)
            if d2d_model:
                die_id = getattr(self.config, "DIE_ID", 0)
                d2d_model.record_request_issued(flit.packet_id, die_id, flit.req_type, is_cross_die)
        
        if network_type == "req" and self.networks[network_type]["send_flits"][flit.packet_id]:
            return True
        self.networks[network_type]["send_flits"][flit.packet_id].append(flit)
        return True

    def inject_to_l2h_pre(self, network_type):
        """1GHz: inject_fifo → l2h_fifo_pre（分网络处理）"""
        net_info = self.networks[network_type]

        if not net_info["inject_fifo"] or len(net_info["l2h_fifo"]) >= net_info["l2h_fifo"].maxlen or net_info["l2h_fifo_pre"] is not None:
            return

        flit = net_info["inject_fifo"][0]

        # 根据网络类型进行不同的处理
        if network_type == "req":
            if flit.req_attr == "new" and not self._check_and_reserve_resources(flit):
                return  # 资源不足，保持在inject_fifo中
            flit.flit_position = "L2H"
            flit.start_inject = True
            net_info["l2h_fifo_pre"] = net_info["inject_fifo"].popleft()

        elif network_type == "rsp":
            # 响应网络：直接移动
            flit.flit_position = "L2H"
            flit.start_inject = True
            net_info["l2h_fifo_pre"] = net_info["inject_fifo"].popleft()

        elif network_type == "data":
            # 数据网络：检查departure_cycle
            current_cycle = getattr(self, "current_cycle", 0)
            if hasattr(flit, "departure_cycle") and flit.departure_cycle > current_cycle:
                return  # 还没到发送时间
            if self.tx_token_bucket:
                self.tx_token_bucket.refill(current_cycle)
                if not self.tx_token_bucket.consume():
                    return
            flit: Flit = net_info["inject_fifo"].popleft()
            flit.flit_position = "L2H"
            flit.start_inject = True
            net_info["l2h_fifo_pre"] = flit

    def l2h_to_IQ_channel_buffer(self, network_type):
        """2GHz: l2h_fifo → network.IQ_channel_buffer"""
        net_info = self.networks[network_type]
        network = net_info["network"]

        if not net_info["l2h_fifo"]:
            return

        # 检查目标缓冲区是否已满（只能在 *pre* 缓冲区为空且正式 FIFO 未满时移动）
        fifo = network.IQ_channel_buffer[self.ip_type][self.ip_pos]
        fifo_pre = network.IQ_channel_buffer_pre[self.ip_type][self.ip_pos]
        if len(fifo) >= getattr(self.config, "IQ_CH_FIFO_DEPTH", 8) or fifo_pre is not None:
            return  # 没空间，或 pre 槽已占用

        # 从 l2h_fifo 弹出一个 flit，先放到 *pre* 槽
        flit: Flit = net_info["l2h_fifo"].popleft()
        flit.flit_position = "IQ_CH"
        network.IQ_channel_buffer_pre[self.ip_type][self.ip_pos] = flit

        # 更新cycle统计
        if network_type == "req" and flit.req_attr == "new":
            flit.cmd_entry_noc_from_cake0_cycle = self.current_cycle
        elif network_type == "rsp":
            flit.cmd_entry_noc_from_cake1_cycle = self.current_cycle
        elif network_type == "data":
            # 只为第一个data flit设置entry时间戳
            if flit.req_type == "read" and flit.flit_id == 0:
                flit.data_entry_noc_from_cake1_cycle = self.current_cycle
            elif flit.req_type == "write" and flit.flit_id == 0:
                flit.data_entry_noc_from_cake0_cycle = self.current_cycle

    def _check_and_reserve_resources(self, req):
        """检查并预占新请求所需的资源"""
        ip_pos, ip_type = self.ip_pos, self.ip_type

        try:
            if req.req_type == "read":
                # 检查读请求资源
                rdb_available = self.node.rn_rdb_count[ip_type][ip_pos] >= req.burst_length
                tracker_available = self.node.rn_tracker_count["read"][ip_type][ip_pos] > 0
                reserve_ok = self.node.rn_rdb_count[ip_type][ip_pos] > self.node.rn_rdb_reserve[ip_type][ip_pos] * req.burst_length

                if not (rdb_available and tracker_available and reserve_ok):
                    return False

                # 预占资源
                self.node.rn_rdb_count[ip_type][ip_pos] -= req.burst_length
                self.node.rn_tracker_count["read"][ip_type][ip_pos] -= 1
                self.node.rn_rdb[ip_type][ip_pos][req.packet_id] = []
                req.cmd_entry_cake0_cycle = self.current_cycle  # 这里记录cycle
                self.node.rn_tracker["read"][ip_type][ip_pos].append(req)
                self.node.rn_tracker_pointer["read"][ip_type][ip_pos] += 1

            elif req.req_type == "write":
                # 检查写请求资源
                wdb_available = self.node.rn_wdb_count[ip_type][ip_pos] >= req.burst_length
                tracker_available = self.node.rn_tracker_count["write"][ip_type][ip_pos] > 0

                if not (wdb_available and tracker_available):
                    return False

                # 预占资源
                self.node.rn_wdb_count[ip_type][ip_pos] -= req.burst_length
                self.node.rn_tracker_count["write"][ip_type][ip_pos] -= 1
                self.node.rn_wdb[ip_type][ip_pos][req.packet_id] = []
                req.cmd_entry_cake0_cycle = self.current_cycle  # 这里记录cycle
                self.node.rn_tracker["write"][ip_type][ip_pos].append(req)
                self.node.rn_tracker_pointer["write"][ip_type][ip_pos] += 1

                # 写数据包将在收到 datasend 响应时创建

            return True

        except (KeyError, AttributeError) as e:
            logging.warning(f"Resource check failed for {req}: {e}")
            return False

    def EQ_channel_buffer_to_h2l_pre(self, network_type):
        """2GHz: network.EQ_channel_buffer → h2l_fifo_h_pre"""
        net_info = self.networks[network_type]
        network = net_info["network"]

        if net_info["h2l_fifo_h_pre"] is not None:
            return  # 预缓冲已占用

        try:
            # 接收，CrossRing使用 ip_pos - NUM_COL，Ring使用 ip_pos
            if hasattr(self.config, "RING_NUM_NODE") and self.config.RING_NUM_NODE > 0:
                # Ring topology: eject at same position as inject
                pos_index = self.ip_pos
            else:
                # CrossRing topology: eject at ip_pos - NUM_COL
                pos_index = self.ip_pos - self.config.NUM_COL

            eq_buf = network.EQ_channel_buffer[self.ip_type][pos_index]
            if not eq_buf:
                return

            # 检查h2l_fifo_h是否有空间
            if len(net_info["h2l_fifo_h"]) >= net_info["h2l_fifo_h"].maxlen:
                return

            flit = eq_buf.popleft()
            flit.is_arrive = True
            flit.flit_position = "H2L_H"
            net_info["h2l_fifo_h_pre"] = flit
            # arrive_flits添加移动到IP_eject阶段，确保只记录真正完成的flit

        except (KeyError, AttributeError) as e:
            logging.warning(f"EQ to h2l_h transfer failed for {network_type}: {e}")

    def h2l_h_to_h2l_l_pre(self, network_type):
        """网络频率: h2l_fifo_h → h2l_fifo_l_pre"""
        net_info = self.networks[network_type]

        if net_info["h2l_fifo_l_pre"] is not None:
            return  # L级预缓冲已占用

        if not net_info["h2l_fifo_h"]:
            return  # H级FIFO为空

        # 检查L级FIFO是否有空间
        if len(net_info["h2l_fifo_l"]) >= net_info["h2l_fifo_l"].maxlen:
            return

        # 从H级FIFO传输到L级预缓冲
        flit = net_info["h2l_fifo_h"].popleft()
        flit.flit_position = "H2L_L"
        net_info["h2l_fifo_l_pre"] = flit

    def _handle_received_request(self, req: Flit):
        """处理接收到的请求（SN端）"""
        # 只有SN-side IP类型可以处理接收到的请求
        if not (self.ip_type.startswith("ddr") or self.ip_type.startswith("l2m")):
            # RN-side IP类型不应该接收请求，直接忽略
            return


        req.cmd_received_by_cake1_cycle = getattr(self, "current_cycle", 0)
        self.req_wait_cycles_h += req.wait_cycle_h
        self.req_wait_cycles_v += req.wait_cycle_v
        self.req_cir_h_num += req.eject_attempts_h
        self.req_cir_v_num += req.eject_attempts_v
        req.cmd_received_by_cake1_cycle = self.current_cycle

        if req.req_type == "read":
            if req.req_attr == "new":
                if self.node.sn_tracker_count[self.ip_type]["ro"][self.ip_pos] > 0:
                    req.sn_tracker_type = "ro"
                    self.node.sn_tracker[self.ip_type][self.ip_pos].append(req)
                    self.node.sn_tracker_count[self.ip_type]["ro"][self.ip_pos] -= 1
                    self.create_read_packet(req)
                    self.release_completed_sn_tracker(req)
                else:
                    self.create_rsp(req, "negative")
                    self.node.sn_req_wait[req.req_type][self.ip_type][self.ip_pos].append(req)
            else:
                self.create_read_packet(req)
                self.release_completed_sn_tracker(req)

        elif req.req_type == "write":
            if req.req_attr == "new":
                if self.node.sn_tracker_count[self.ip_type]["share"][self.ip_pos] > 0 and self.node.sn_wdb_count[self.ip_type][self.ip_pos] >= req.burst_length:
                    req.sn_tracker_type = "share"
                    self.node.sn_tracker[self.ip_type][self.ip_pos].append(req)
                    self.node.sn_tracker_count[self.ip_type]["share"][self.ip_pos] -= 1
                    self.node.sn_wdb[self.ip_type][self.ip_pos][req.packet_id] = []
                    self.node.sn_wdb_count[self.ip_type][self.ip_pos] -= req.burst_length
                    self.create_rsp(req, "datasend")
                else:
                    self.create_rsp(req, "negative")
                    self.node.sn_req_wait[req.req_type][self.ip_type][self.ip_pos].append(req)
            else:
                self.create_rsp(req, "datasend")

    def _handle_received_response(self, rsp: Flit):
        """处理接收到的响应（RN端）"""
        rsp.cmd_received_by_cake1_cycle = getattr(self, "current_cycle", 0)
        self.rsp_wait_cycles_h += rsp.wait_cycle_h
        self.rsp_wait_cycles_v += rsp.wait_cycle_v
        self.rsp_cir_h_num += rsp.eject_attempts_h
        self.rsp_cir_v_num += rsp.eject_attempts_v
        rsp.cmd_received_by_cake0_cycle = self.current_cycle
        if rsp.req_type == "read" and rsp.rsp_type == "negative":
            self.read_retry_num_stat += 1
        elif rsp.req_type == "write" and rsp.rsp_type == "negative":
            self.write_retry_num_stat += 1

        req = next((req for req in self.node.rn_tracker[rsp.req_type][self.ip_type][self.ip_pos] if req.packet_id == rsp.packet_id), None)
        if not req:
            # For Ring topology, ignore spurious positive responses for read requests
            if hasattr(self.config, "RING_NUM_NODE") and self.config.RING_NUM_NODE > 0 and rsp.req_type == "read" and rsp.rsp_type == "positive":
                return  # Silently ignore - this is expected for Ring read completions
            logging.warning(f"RSP {rsp} do not have REQ")
            return

        req.sync_latency_record(rsp)

        if rsp.req_type == "read":
            if rsp.rsp_type == "negative":
                # 读请求retry逻辑
                if req.req_attr == "old":
                    # 该请求已经被重新注入网络，不需要再次修改。直接返回
                    return
                req.req_state = "invalid"
                req.is_injected = False
                req.path_index = 0
                req.req_attr = "old"
                self.node.rn_rdb_count[self.ip_type][self.ip_pos] += req.burst_length
                if req.packet_id in self.node.rn_rdb[self.ip_type][self.ip_pos]:
                    self.node.rn_rdb[self.ip_type][self.ip_pos].pop(req.packet_id)
                self.node.rn_rdb_reserve[self.ip_type][self.ip_pos] += 1

            elif rsp.rsp_type == "positive":
                # 处理读重试
                if req.req_attr == "new":
                    self.node.rn_rdb_count[self.ip_type][self.ip_pos] += req.burst_length
                    if req.packet_id in self.node.rn_rdb[self.ip_type][self.ip_pos]:
                        self.node.rn_rdb[self.ip_type][self.ip_pos].pop(req.packet_id)
                    self.node.rn_rdb_reserve[self.ip_type][self.ip_pos] += 1
                req.req_state = "valid"
                req.req_attr = "old"
                req.is_injected = False
                req.path_index = 0
                req.is_new_on_network = True
                req.is_arrive = False
                # 放入请求网络的inject_fifo
                self.enqueue(req, "req", retry=True)
                self.node.rn_rdb_reserve[self.ip_type][self.ip_pos] -= 1

        elif rsp.req_type == "write":
            if rsp.rsp_type == "negative":
                # 写请求retry逻辑
                if req.req_attr == "old":
                    # 该请求已经被重新注入网络，不需要再次修改。直接返回
                    return
                req.req_state = "invalid"
                req.req_attr = "old"
                req.is_injected = False
                req.path_index = 0

            elif rsp.rsp_type == "positive":
                # 处理写重试
                req.req_state = "valid"
                req.is_injected = False
                req.path_index = 0
                req.req_attr = "old"
                req.is_new_on_network = True
                req.is_arrive = False
                # 放入请求网络的inject_fifo
                self.enqueue(req, "req", retry=True)

            elif rsp.rsp_type == "datasend":
                # 收到写数据发送响应，现在创建并发送写数据包
                req: Flit = next((r for r in self.node.rn_tracker["write"][self.ip_type][self.ip_pos] if r.packet_id == rsp.packet_id), None)
                if req:
                    # 创建写数据包
                    self.create_write_packet(req)
                    # 将写数据包放入数据网络inject_fifo
                    for flit in self.node.rn_wdb[self.ip_type][self.ip_pos][rsp.packet_id]:
                        self.enqueue(flit, "data")
                    
                    # 检查是否为跨Die写请求
                    is_cross_die_write = self._is_cross_die_write_request(req)
                    
                    if is_cross_die_write:
                        # 跨Die写请求：发送数据但不释放tracker，等待B通道响应
                        pass
                    else:
                        # 普通Die内写请求：发送数据并立即释放tracker
                        self.node.rn_tracker["write"][self.ip_type][self.ip_pos].remove(req)
                        self.node.rn_wdb_count[self.ip_type][self.ip_pos] += req.burst_length
                        self.node.rn_tracker_count["write"][self.ip_type][self.ip_pos] += 1
                        self.node.rn_tracker_pointer["write"][self.ip_type][self.ip_pos] -= 1
                        
                # 同时清理写缓冲
                self.node.rn_wdb[self.ip_type][self.ip_pos].pop(rsp.packet_id, None)
                
            elif rsp.rsp_type == "write_complete":
                # 收到写完成响应，检查是否为跨Die写请求的最终响应
                req: Flit = next((r for r in self.node.rn_tracker["write"][self.ip_type][self.ip_pos] 
                                 if r.packet_id == rsp.packet_id and self._is_cross_die_write_request(r)), None)
                if req:
                    # 这是跨Die写请求的最终B通道响应，现在可以释放tracker
                    self.node.rn_tracker["write"][self.ip_type][self.ip_pos].remove(req)
                    self.node.rn_wdb_count[self.ip_type][self.ip_pos] += req.burst_length
                    self.node.rn_tracker_count["write"][self.ip_type][self.ip_pos] += 1
                    self.node.rn_tracker_pointer["write"][self.ip_type][self.ip_pos] -= 1
                    
                    # 更新D2D请求完成计数
                    if hasattr(req, 'd2d_origin_die') and hasattr(req, 'd2d_target_die'):
                        die_id = getattr(self.config, 'DIE_ID', None)
                        if die_id is not None and req.d2d_origin_die == die_id:
                            d2d_model = getattr(self.req_network, 'd2d_model', None)
                            if d2d_model:
                                d2d_model.d2d_requests_completed[die_id] += 1
                                # 记录write_complete响应的接收
                                d2d_model.record_write_complete(req.packet_id)

    def _is_cross_die_write_request(self, req: Flit) -> bool:
        """检查是否为跨Die写请求"""
        # 检查是否有D2D目标Die ID，且与当前Die ID不同
        if hasattr(req, 'd2d_target_die') and req.d2d_target_die is not None:
            current_die_id = getattr(self.config, 'DIE_ID', 0)
            return req.d2d_target_die != current_die_id
        return False

    def release_completed_sn_tracker(self, req: Flit):
        # —— 1) 移除已完成的 tracker ——
        self.node.sn_tracker[self.ip_type][self.ip_pos].remove(req)
        # 释放一个 tracker 槽
        self.node.sn_tracker_count[self.ip_type][req.sn_tracker_type][self.ip_pos] += 1

        # —— 2) 对于写请求，还要释放写缓冲额度 ——
        if req.req_type == "write":
            self.node.sn_wdb_count[self.ip_type][self.ip_pos] += req.burst_length

        # —— 3) 尝试给等待队列里的请求重新分配资源 ——
        wait_list = self.node.sn_req_wait[req.req_type][self.ip_type][self.ip_pos]
        if not wait_list:
            return

        if req.req_type == "write":
            # 写：既要有空 tracker，也要有足够 wdb_count
            if self.node.sn_tracker_count[self.ip_type][req.sn_tracker_type][self.ip_pos] > 0 and self.node.sn_wdb_count[self.ip_type][self.ip_pos] > 0:
                new_req = wait_list.pop(0)
                new_req.sn_tracker_type = req.sn_tracker_type

                # 分配 tracker + wdb
                self.node.sn_tracker[self.ip_type][self.ip_pos].append(new_req)
                self.node.sn_tracker_count[self.ip_type][new_req.sn_tracker_type][self.ip_pos] -= 1

                self.node.sn_wdb_count[self.ip_type][self.ip_pos] -= new_req.burst_length

                # 发送 positive 响应
                self.create_rsp(new_req, "positive")

        elif req.req_type == "read":
            # 读：只要有空 tracker 即可
            if self.node.sn_tracker_count[self.ip_type][req.sn_tracker_type][self.ip_pos] > 0:
                new_req = wait_list.pop(0)
                new_req.sn_tracker_type = req.sn_tracker_type

                # 分配 tracker
                self.node.sn_tracker[self.ip_type][self.ip_pos].append(new_req)
                self.node.sn_tracker_count[self.ip_type][new_req.sn_tracker_type][self.ip_pos] -= 1

                # 直接生成并发送读数据包
                self.create_read_packet(new_req)

    def _handle_received_data(self, flit: Flit):
        """处理接收到的数据"""
        flit.arrival_cycle = getattr(self, "current_cycle", 0)
        self.data_wait_cycles_h += flit.wait_cycle_h
        self.data_wait_cycles_v += flit.wait_cycle_v
        self.data_cir_h_num += flit.eject_attempts_h
        self.data_cir_v_num += flit.eject_attempts_v
        if flit.req_type == "read":
            # 检查是否为跨Die返回的数据，更新D2D统计
            if hasattr(flit, "d2d_origin_die") and hasattr(flit, "d2d_target_die"):
                if flit.d2d_origin_die != flit.d2d_target_die:
                    # 这是跨Die请求的返回数据，需要通过网络或config获取Die信息
                    die_id = getattr(self.config, 'DIE_ID', None)
                    if die_id is not None and flit.d2d_origin_die == die_id:
                        # 通过网络对象获取d2d_model引用
                        d2d_model = getattr(self.req_network, 'd2d_model', None)
                        if d2d_model:
                            d2d_model.d2d_received_flits[die_id] += 1
            
            # 读数据到达RN端，需要收集到data buffer中
            self.node.rn_rdb[self.ip_type][self.ip_pos][flit.packet_id].append(flit)
            
            # 如果是跨Die数据且完成了整个burst，更新请求完成计数
            if hasattr(flit, "d2d_origin_die") and hasattr(flit, "d2d_target_die"):
                if flit.d2d_origin_die != flit.d2d_target_die:
                    die_id = getattr(self.config, 'DIE_ID', None)
                    if die_id is not None and flit.d2d_origin_die == die_id:
                        if len(self.node.rn_rdb[self.ip_type][self.ip_pos][flit.packet_id]) == flit.burst_length:
                            d2d_model = getattr(self.req_network, 'd2d_model', None)
                            if d2d_model:
                                d2d_model.d2d_requests_completed[die_id] += 1
            # 检查是否收集完整个burst
            if len(self.node.rn_rdb[self.ip_type][self.ip_pos][flit.packet_id]) == flit.burst_length:
                req = next((req for req in self.node.rn_tracker["read"][self.ip_type][self.ip_pos] if req.packet_id == flit.packet_id), None)
                if req:
                    # 立即释放tracker和更新计数
                    self.node.rn_tracker["read"][self.ip_type][self.ip_pos].remove(req)
                    self.node.rn_tracker_count["read"][self.ip_type][self.ip_pos] += 1
                    self.node.rn_tracker_pointer["read"][self.ip_type][self.ip_pos] -= 1
                    self.node.rn_rdb_count[self.ip_type][self.ip_pos] += req.burst_length
                    # 为所有flit设置完成时间戳和计算延迟
                    first_flit = self.data_network.send_flits[flit.packet_id][0]
                    # 记录最后到达时间
                    complete_cycle = self.current_cycle

                    for f in self.data_network.send_flits[flit.packet_id]:
                        f.leave_db_cycle = self.current_cycle
                        f.sync_latency_record(req)
                        # 为所有flit设置receive时间戳，确保后续处理能获得正确值
                        f.data_received_complete_cycle = complete_cycle
                        # 计算延迟
                        f.cmd_latency = f.cmd_received_by_cake1_cycle - f.cmd_entry_noc_from_cake0_cycle
                        f.data_latency = complete_cycle - first_flit.data_entry_noc_from_cake1_cycle
                        f.transaction_latency = complete_cycle - f.cmd_entry_cake0_cycle

                    # 清理data buffer（数据已经收集完成）
                    self.node.rn_rdb[self.ip_type][self.ip_pos].pop(flit.packet_id)

                else:
                    print(f"Warning: No RN tracker found for packet_id {flit.packet_id}")

        elif flit.req_type == "write":
            self.node.sn_wdb[self.ip_type][self.ip_pos][flit.packet_id].append(flit)

            # 检查是否收集完整个burst
            if len(self.node.sn_wdb[self.ip_type][self.ip_pos][flit.packet_id]) == flit.burst_length:

                # 找到对应的请求
                req = next((req for req in self.node.sn_tracker[self.ip_type][self.ip_pos] if req.packet_id == flit.packet_id), None)
                if req:
                    # 设置tracker延迟释放时间
                    release_time = self.current_cycle + self.config.SN_TRACKER_RELEASE_LATENCY

                    # 初始化释放时间字典（如果不存在）
                    if not hasattr(self.node, "sn_tracker_release_time"):
                        self.node.sn_tracker_release_time = defaultdict(list)

                    # 更新flit时间戳
                    first_flit = next((flit for flit in self.data_network.send_flits[flit.packet_id] if flit.flit_id == 0), self.data_network.send_flits[flit.packet_id][0])
                    # 记录最后到达时间
                    complete_cycle = self.current_cycle

                    for f in self.data_network.send_flits[flit.packet_id]:
                        f.leave_db_cycle = release_time
                        f.sync_latency_record(req)
                        # 为所有flit设置receive时间戳，确保后续处理能获得正确值
                        f.data_received_complete_cycle = complete_cycle
                        # 计算延迟
                        f.cmd_latency = f.cmd_received_by_cake0_cycle - f.cmd_entry_noc_from_cake0_cycle
                        f.data_latency = complete_cycle - first_flit.data_entry_noc_from_cake0_cycle
                        f.transaction_latency = complete_cycle + self.config.SN_TRACKER_RELEASE_LATENCY - f.cmd_entry_cake0_cycle

                    # 清理data buffer（数据已经收集完成）
                    self.node.sn_wdb[self.ip_type][self.ip_pos].pop(flit.packet_id)

                    # 添加到延迟释放队列
                    self.node.sn_tracker_release_time[release_time].append((self.ip_type, self.ip_pos, req))
                else:
                    print(f"Warning: No SN tracker found for packet_id {flit.packet_id}")

    def h2l_l_to_eject_fifo(self, network_type):
        """1GHz: h2l_fifo_l → 处理完成"""
        net_info = self.networks[network_type]
        if not net_info["h2l_fifo_l"]:
            return

        current_cycle = getattr(self, "current_cycle", 0)
        if network_type == "data" and self.rx_token_bucket:
            self.rx_token_bucket.refill(current_cycle)
            if not self.rx_token_bucket.consume():
                return

        flit = net_info["h2l_fifo_l"].popleft()
        flit.flit_position = "IP_eject"
        flit.is_finish = True

        # 在IP_eject阶段添加到arrive_flits，确保只记录真正完成的flit
        if flit.packet_id not in net_info["network"].arrive_flits:
            net_info["network"].arrive_flits[flit.packet_id] = []
        net_info["network"].arrive_flits[flit.packet_id].append(flit)
        net_info["network"].recv_flits_num += 1

        # 根据网络类型进行特殊处理
        if network_type == "req":
            self._handle_received_request(flit)
        elif network_type == "rsp":
            self._handle_received_response(flit)
        elif network_type == "data":
            self._handle_received_data(flit)
        return flit
        # Ensure method always returns None if no flit ejected
        return None

    def inject_step(self, cycle):
        """根据周期和频率调用inject相应的方法"""
        self.current_cycle = cycle
        cycle_mod = cycle % self.config.NETWORK_FREQUENCY

        # 1GHz 操作（每个网络周期执行一次）
        if cycle_mod == 0:
            # 对三个网络分别执行inject_to_l2h_pre
            for net_type in ["req", "rsp", "data"]:
                self.inject_to_l2h_pre(net_type)

        # 2GHz 操作（每半个网络周期执行一次）
        for net_type in ["req", "rsp", "data"]:
            self.l2h_to_IQ_channel_buffer(net_type)

    def move_pre_to_fifo(self):
        # pre → fifo 的移动（每个周期都执行）
        for net_type, net_info in self.networks.items():
            net = net_info["network"]
            # l2h_fifo_pre → l2h_fifo
            if net_info["l2h_fifo_pre"] is not None and len(net_info["l2h_fifo"]) < net_info["l2h_fifo"].maxlen:
                net_info["l2h_fifo"].append(net_info["l2h_fifo_pre"])
                net_info["l2h_fifo_pre"] = None

            # h2l_fifo_h_pre → h2l_fifo_h
            if net_info["h2l_fifo_h_pre"] is not None and len(net_info["h2l_fifo_h"]) < net_info["h2l_fifo_h"].maxlen:
                net_info["h2l_fifo_h"].append(net_info["h2l_fifo_h_pre"])
                net_info["h2l_fifo_h_pre"] = None

            # h2l_fifo_l_pre → h2l_fifo_l
            if net_info["h2l_fifo_l_pre"] is not None and len(net_info["h2l_fifo_l"]) < net_info["h2l_fifo_l"].maxlen:
                net_info["h2l_fifo_l"].append(net_info["h2l_fifo_l_pre"])
                net_info["h2l_fifo_l_pre"] = None

    def eject_step(self, cycle):
        """根据周期和频率调用eject相应的方法"""
        self.current_cycle = cycle
        cycle_mod = cycle % self.config.NETWORK_FREQUENCY

        # Initialize list to collect ejected flits
        ejected_flits = []

        # 2GHz 操作（每半个网络周期执行一次）
        for net_type in ["req", "rsp", "data"]:
            self.EQ_channel_buffer_to_h2l_pre(net_type)
            self.h2l_h_to_h2l_l_pre(net_type)

        # 1GHz 操作（每个网络周期执行一次）
        if cycle_mod == 0:
            # 对三个网络分别执行h2l_to_eject_fifo，收集被eject的flit
            for net_type in ["req", "rsp", "data"]:
                flit = self.h2l_l_to_eject_fifo(net_type)
                if flit:
                    ejected_flits.append(flit)

        return ejected_flits

    def create_write_packet(self, req: Flit):
        """创建写数据包并放入数据网络inject_fifo"""
        cycle = getattr(self, "current_cycle", 0)
        for i in range(req.burst_length):
            source = req.source
            destination = req.destination
            path = self.routes[source][destination]
            flit = Flit(source, destination, path)
            flit.sync_latency_record(req)
            flit.source_original = req.source_original
            flit.destination_original = req.destination_original
            flit.flit_type = "data"
            # 设置保序信息
            flit.set_packet_category_and_order_id()
            flit.departure_cycle = (
                cycle + self.config.DDR_W_LATENCY + i * self.config.NETWORK_FREQUENCY
                if req.original_destination_type and req.original_destination_type.startswith("ddr")
                else cycle + self.config.L2M_W_LATENCY + i * self.config.NETWORK_FREQUENCY
            )
            flit.req_departure_cycle = req.departure_cycle
            flit.entry_db_cycle = req.entry_db_cycle
            flit.source_type = req.source_type
            flit.destination_type = req.destination_type
            flit.original_source_type = req.original_source_type
            flit.original_destination_type = req.original_destination_type
            flit.req_type = req.req_type
            flit.packet_id = req.packet_id
            flit.flit_id = i
            flit.burst_length = req.burst_length
            flit.traffic_id = req.traffic_id
            if i == req.burst_length - 1:
                flit.is_last_flit = True

            # 继承D2D属性
            if hasattr(req, "d2d_origin_die"):
                flit.d2d_origin_die = req.d2d_origin_die      # 发起Die ID
                flit.d2d_origin_node = req.d2d_origin_node    # 发起节点源映射位置
                flit.d2d_origin_type = req.d2d_origin_type    # 发起IP类型
                flit.d2d_target_die = req.d2d_target_die      # 目标Die ID
                flit.d2d_target_node = req.d2d_target_node    # 目标节点源映射位置
                flit.d2d_target_type = req.d2d_target_type    # 目标IP类型

            # 将写数据包放入wdb中
            self.node.rn_wdb[self.ip_type][self.ip_pos][flit.packet_id].append(flit)

    def create_read_packet(self, req: Flit):
        """创建读数据包并放入数据网络inject_fifo"""
        cycle = getattr(self, "current_cycle", 0)
        for i in range(req.burst_length):
            # Ring拓扑：数据包从当前SN位置返回到原始请求者位置
            if hasattr(self.config, "RING_NUM_NODE"):
                # Ring拓扑
                source = req.destination  # 当前SN位置
                destination = req.source  # 原始请求者位置
            else:
                # 原始网格拓扑逻辑
                source = req.destination + self.config.NUM_COL
                destination = req.source - self.config.NUM_COL

            # Ring拓扑：使用Ring路由策略动态计算数据包路径
            if hasattr(self.config, "RING_NUM_NODE") and hasattr(self, "ring_model") and self.ring_model:
                path = self.ring_model._get_ring_path(source, destination)
            else:
                path = self.routes[source][destination]
            flit = Flit(source, destination, path)
            flit.sync_latency_record(req)
            flit.source_original = req.destination_original
            flit.destination_original = req.source_original
            flit.req_type = req.req_type
            flit.flit_type = "data"
            # 设置保序信息
            flit.set_packet_category_and_order_id()
            if hasattr(req, "original_destination_type") and req.original_destination_type.startswith("ddr"):
                latency = np.random.uniform(low=self.config.DDR_R_LATENCY - self.config.DDR_R_LATENCY_VAR, high=self.config.DDR_R_LATENCY + self.config.DDR_R_LATENCY_VAR, size=None)
            elif hasattr(req, "destination_type") and req.destination_type and req.destination_type.startswith("ddr"):
                latency = np.random.uniform(low=self.config.DDR_R_LATENCY - self.config.DDR_R_LATENCY_VAR, high=self.config.DDR_R_LATENCY + self.config.DDR_R_LATENCY_VAR, size=None)
            else:
                latency = self.config.L2M_R_LATENCY
            flit.departure_cycle = cycle + latency + i * self.config.NETWORK_FREQUENCY
            flit.entry_db_cycle = cycle
            flit.req_departure_cycle = req.departure_cycle
            flit.source_type = getattr(req, "destination_type", None)
            flit.destination_type = getattr(req, "source_type", None)
            flit.original_source_type = getattr(req, "original_source_type", None)
            flit.original_destination_type = getattr(req, "original_destination_type", None)
            flit.packet_id = req.packet_id
            flit.flit_id = i
            flit.burst_length = req.burst_length
            flit.traffic_id = req.traffic_id
            if i == req.burst_length - 1:
                flit.is_last_flit = True
            
            # 继承D2D属性
            if hasattr(req, "d2d_origin_die"):
                flit.d2d_origin_die = req.d2d_origin_die      # 发起Die ID
                flit.d2d_origin_node = req.d2d_origin_node    # 发起节点源映射位置
                flit.d2d_origin_type = req.d2d_origin_type    # 发起IP类型
                flit.d2d_target_die = req.d2d_target_die      # 目标Die ID
                flit.d2d_target_node = req.d2d_target_node    # 目标节点源映射位置
                flit.d2d_target_type = req.d2d_target_type    # 目标IP类型

            # 将读数据包放入数据网络的inject_fifo
            self.enqueue(flit, "data")

    def create_rsp(self, req: Flit, rsp_type):
        """创建响应并放入响应网络inject_fifo"""
        cycle = getattr(self, "current_cycle", 0)

        # 检查是否为Ring拓扑
        if hasattr(self.config, "RING_NUM_NODE"):
            # Ring拓扑：响应直接从目标节点返回到源节点
            source = req.destination
            destination = req.source
        else:
            # 原始网格拓扑逻辑
            source = req.destination + self.config.NUM_COL
            destination = req.source - self.config.NUM_COL

        path = self.routes[source][destination]
        rsp = Flit(source, destination, path)
        rsp.flit_type = "rsp"
        rsp.rsp_type = rsp_type
        rsp.req_type = req.req_type
        # 设置保序信息
        rsp.set_packet_category_and_order_id()
        rsp.packet_id = req.packet_id
        rsp.departure_cycle = cycle + self.config.NETWORK_FREQUENCY
        rsp.req_departure_cycle = req.departure_cycle
        rsp.source_type = req.destination_type
        rsp.destination_type = req.source_type
        rsp.sync_latency_record(req)
        rsp.sn_rsp_generate_cycle = cycle

        # 将响应放入响应网络的inject_fifo
        self.enqueue(rsp, "rsp")


def create_ring_ip_interface(ip_type, ip_pos, config, req_network, rsp_network, data_network, node, routes):
    """创建支持Ring的IPInterface实例"""
    from .ip_interface import IPInterface

    # 创建混合类
    class RingIPInterfaceImpl(RingIPInterface, IPInterface):
        pass

    return RingIPInterfaceImpl(ip_type, ip_pos, config, req_network, rsp_network, data_network, node, routes, ring_mode=True)
