"""
Node class for NoC simulation.
Handles network node functionality including RN/SN trackers and databases.
Enhanced with route table support for flexible routing.
"""

from __future__ import annotations
from collections import defaultdict
from config.config import CrossRingConfig
from functools import lru_cache


class Node:
    global_packet_id = 0

    def __init__(self, config: CrossRingConfig, node_id: int = 0):
        self.config = config

        self.initialize_data_structures()
        self.initialize_rn()
        self.initialize_sn()

    @classmethod
    def get_next_packet_id(cls):
        cls.global_packet_id += 1
        return cls.global_packet_id

    @classmethod
    def clear_packet_id(cls):
        cls.global_packet_id = 0

    def initialize_data_structures(self):
        """Initialize the data structures for read and write databases."""

        # 包含D2D_RN接口
        rn_channels = ("sdma", "gdma", "cdma", "d2d_rn")
        self.rn_rdb = self.config._make_channels(rn_channels)
        self.rn_rdb_reserve = self.config._make_channels(rn_channels)
        self.rn_rdb_recv = self.config._make_channels(rn_channels)
        self.rn_rdb_count = self.config._make_channels(rn_channels)
        self.rn_wdb = self.config._make_channels(rn_channels)
        self.rn_wdb_reserve = self.config._make_channels(rn_channels)
        self.rn_wdb_count = self.config._make_channels(rn_channels)
        self.rn_wdb_send = self.config._make_channels(rn_channels)
        self.rn_tracker = {"read": self.config._make_channels(rn_channels), "write": self.config._make_channels(rn_channels)}
        self.rn_tracker_wait = {"read": self.config._make_channels(rn_channels), "write": self.config._make_channels(rn_channels)}
        self.rn_tracker_count = {"read": self.config._make_channels(rn_channels), "write": self.config._make_channels(rn_channels)}
        self.rn_tracker_pointer = {"read": self.config._make_channels(rn_channels), "write": self.config._make_channels(rn_channels)}
        # 包含D2D_SN接口
        sn_channels = ("ddr", "l2m", "d2d_sn")
        self.sn_rdb = self.config._make_channels(sn_channels)
        self.sn_rsp_queue = self.config._make_channels(sn_channels)
        self.sn_req_wait = {"read": self.config._make_channels(sn_channels), "write": self.config._make_channels(sn_channels)}
        self.sn_tracker = self.config._make_channels(sn_channels)
        self.sn_tracker_count = self.config._make_channels(sn_channels, value_factory={"ro": {}, "share": {}})
        self.sn_wdb = self.config._make_channels(sn_channels)
        self.sn_wdb_recv = self.config._make_channels(sn_channels)
        self.sn_wdb_count = self.config._make_channels(sn_channels)
        self.rn_wait_to_inject = []
        
        # 保序跟踪表: {(src, dest): {"REQ": last_ejected_id, "RSP": last_ejected_id, "DATA": last_ejected_id}}
        self.order_tracking_table = defaultdict(lambda: {"REQ": 0, "RSP": 0, "DATA": 0})
        
    # 全局顺序ID分配器: {(src, dest): {"REQ": next_id, "RSP": next_id, "DATA": next_id}}
    global_order_id_allocator = defaultdict(lambda: {"REQ": 1, "RSP": 1, "DATA": 1})
    
    @classmethod
    def get_next_order_id(cls, src, dest, packet_category):
        """获取下一个顺序ID"""
        current_id = cls.global_order_id_allocator[(src, dest)][packet_category]
        cls.global_order_id_allocator[(src, dest)][packet_category] += 1
        return current_id
        
    @classmethod
    def reset_order_ids(cls):
        """重置所有顺序ID"""
        cls.global_order_id_allocator.clear()

    def initialize_rn(self):
        """Initialize RN structures."""
        # 针对 Ring 拓扑，所有节点都有 RN tracker
        is_ring_topology = getattr(self.config, "TOPO_TYPE", "").startswith("Ring")
        if is_ring_topology:
            for ip_type in self.rn_rdb.keys():
                for ip_pos in range(self.config.NUM_NODE):
                    self.rn_rdb[ip_type][ip_pos] = defaultdict(list)
                    self.rn_wdb[ip_type][ip_pos] = defaultdict(list)
                    self.setup_rn_trackers(ip_type, ip_pos)
        else:
            # 原始 CrossRing 拓扑：只在特定位置创建 RN tracker
            for ip_type in self.rn_rdb.keys():
                positions = getattr(self.config, f"{ip_type[:-2].upper()}_SEND_POSITION_LIST")
                for ip_pos in positions:
                    self.rn_rdb[ip_type][ip_pos] = defaultdict(list)
                    self.rn_wdb[ip_type][ip_pos] = defaultdict(list)
                    self.setup_rn_trackers(ip_type, ip_pos)

    def setup_rn_trackers(self, ip_type, ip_pos):
        """Setup read and write trackers for RN."""
        self.rn_rdb_recv[ip_type][ip_pos] = []
        
        # 为D2D_RN使用专用配置
        if ip_type.startswith("d2d_rn"):
            self.rn_rdb_count[ip_type][ip_pos] = getattr(self.config, "D2D_RN_RDB_SIZE", self.config.RN_RDB_SIZE)
            self.rn_wdb_count[ip_type][ip_pos] = getattr(self.config, "D2D_RN_WDB_SIZE", self.config.RN_WDB_SIZE)
        else:
            self.rn_rdb_count[ip_type][ip_pos] = self.config.RN_RDB_SIZE
            self.rn_wdb_count[ip_type][ip_pos] = self.config.RN_WDB_SIZE
            
        self.rn_rdb_reserve[ip_type][ip_pos] = 0
        self.rn_wdb_send[ip_type][ip_pos] = []
        self.rn_wdb_reserve[ip_type][ip_pos] = 0

        for req_type in ["read", "write"]:
            self.rn_tracker[req_type][ip_type][ip_pos] = []
            self.rn_tracker_wait[req_type][ip_type][ip_pos] = []
            
            # 为D2D_RN使用专用配置
            if ip_type.startswith("d2d_rn"):
                if req_type == "read":
                    self.rn_tracker_count[req_type][ip_type][ip_pos] = getattr(self.config, "D2D_RN_R_TRACKER_OSTD", self.config.RN_R_TRACKER_OSTD)
                else:
                    self.rn_tracker_count[req_type][ip_type][ip_pos] = getattr(self.config, "D2D_RN_W_TRACKER_OSTD", self.config.RN_W_TRACKER_OSTD)
            else:
                self.rn_tracker_count[req_type][ip_type][ip_pos] = self.config.RN_R_TRACKER_OSTD if req_type == "read" else self.config.RN_W_TRACKER_OSTD
                
            self.rn_tracker_pointer[req_type][ip_type][ip_pos] = -1

    def initialize_sn(self):
        """Initialize SN structures."""
        self.sn_tracker_release_time = defaultdict(list)

        # 检查是否为Ring拓扑 - 通过TOPO_TYPE字段判断
        is_ring_topology = getattr(self.config, "TOPO_TYPE", "").startswith("Ring")

        if is_ring_topology:
            # Ring拓扑：每个节点都有所有IP类型的SN tracker
            for ip_pos in range(getattr(self.config, "RING_NUM_NODE", self.config.NUM_NODE)):
                for key in self.sn_tracker:
                    self.sn_rdb[key][ip_pos] = []
                    self.sn_wdb[key][ip_pos] = defaultdict(list)
                    self.setup_sn_trackers(key, ip_pos)
        else:
            # 原始CrossRing拓扑：只在特定位置创建SN tracker
            for ip_pos in set(self.config.DDR_SEND_POSITION_LIST + self.config.L2M_SEND_POSITION_LIST):
                for key in self.sn_tracker:
                    self.sn_rdb[key][ip_pos] = []
                    self.sn_wdb[key][ip_pos] = defaultdict(list)
                    self.setup_sn_trackers(key, ip_pos)

    def setup_sn_trackers(self, key, ip_pos):
        """Setup trackers for SN."""
        self.sn_rsp_queue[key][ip_pos] = []
        for req_type in ["read", "write"]:
            self.sn_req_wait[req_type][key][ip_pos] = []
        self.sn_wdb_recv[key][ip_pos] = []
        self.sn_tracker[key][ip_pos] = []
        # if self.config.topo_type != "3x3":
        if key.startswith("ddr"):
            self.sn_wdb_count[key][ip_pos] = self.config.SN_DDR_WDB_SIZE
            self.sn_tracker_count[key]["ro"][ip_pos] = self.config.SN_DDR_R_TRACKER_OSTD
            self.sn_tracker_count[key]["share"][ip_pos] = self.config.SN_DDR_W_TRACKER_OSTD
        elif key.startswith("l2m"):
            self.sn_wdb_count[key][ip_pos] = self.config.SN_L2M_WDB_SIZE
            self.sn_tracker_count[key]["ro"][ip_pos] = self.config.SN_L2M_R_TRACKER_OSTD
            self.sn_tracker_count[key]["share"][ip_pos] = self.config.SN_L2M_W_TRACKER_OSTD
        elif key.startswith("d2d_sn"):
            # 为D2D_SN使用专用配置
            self.sn_wdb_count[key][ip_pos] = getattr(self.config, "D2D_SN_WDB_SIZE", self.config.SN_DDR_WDB_SIZE)
            self.sn_tracker_count[key]["ro"][ip_pos] = getattr(self.config, "D2D_SN_R_TRACKER_OSTD", self.config.SN_DDR_R_TRACKER_OSTD)
            self.sn_tracker_count[key]["share"][ip_pos] = getattr(self.config, "D2D_SN_W_TRACKER_OSTD", self.config.SN_DDR_W_TRACKER_OSTD)


# 添加全局node_map函数，根据base_model的实现
@lru_cache(maxsize=1024)
def node_map(node, is_source=True, num_col=4):
    """
    Node mapping function based on base_model implementation
    Args:
        node: original node id
        is_source: True for source mapping, False for destination mapping  
        num_col: number of columns in topology (default 4)
    Returns:
        mapped node position
    """
    if is_source:
        return node % num_col + num_col + node // num_col * 2 * num_col
    else:
        return node % num_col + node // num_col * 2 * num_col
