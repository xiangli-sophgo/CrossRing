"""
Node class for NoC simulation.
Handles network node functionality including RN/SN trackers and databases.
Enhanced with route table support for flexible routing.
"""

from __future__ import annotations
from collections import defaultdict
from config.config import CrossRingConfig


class Node:
    global_packet_id = -1

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
        cls.global_packet_id = -1

    def initialize_data_structures(self):
        """Initialize the data structures for read and write databases."""

        self.rn_rdb = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_rdb_reserve = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_rdb_recv = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_rdb_count = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_wdb = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_wdb_reserve = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_wdb_count = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_wdb_send = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_tracker = {"read": self.config._make_channels(("sdma", "gdma", "cdma")), "write": self.config._make_channels(("sdma", "gdma", "cdma"))}
        self.rn_tracker_wait = {"read": self.config._make_channels(("sdma", "gdma", "cdma")), "write": self.config._make_channels(("sdma", "gdma", "cdma"))}
        self.rn_tracker_count = {"read": self.config._make_channels(("sdma", "gdma", "cdma")), "write": self.config._make_channels(("sdma", "gdma", "cdma"))}
        self.rn_tracker_pointer = {"read": self.config._make_channels(("sdma", "gdma", "cdma")), "write": self.config._make_channels(("sdma", "gdma", "cdma"))}
        self.sn_rdb = self.config._make_channels(("ddr", "l2m"))
        self.sn_rsp_queue = self.config._make_channels(("ddr", "l2m"))
        self.sn_req_wait = {"read": self.config._make_channels(("ddr", "l2m")), "write": self.config._make_channels(("ddr", "l2m"))}
        self.sn_tracker = self.config._make_channels(("ddr", "l2m"))
        self.sn_tracker_count = self.config._make_channels(("ddr", "l2m"), value_factory={"ro": {}, "share": {}})
        self.sn_wdb = self.config._make_channels(("ddr", "l2m"))
        self.sn_wdb_recv = self.config._make_channels(("ddr", "l2m"))
        self.sn_wdb_count = self.config._make_channels(("ddr", "l2m"))
        self.rn_wait_to_inject = []
        
        # 保序跟踪表: {(src, dest): {"REQ": last_ejected_id, "RSP": last_ejected_id, "DATA": last_ejected_id}}
        self.order_tracking_table = defaultdict(lambda: {"REQ": 0, "RSP": 0, "DATA": 0})
        
        # 动态IP挂载管理
        self.attached_ips = {}  # {node_pos: set(ip_types)}
        self.dynamic_ip_channels = {}  # 动态创建的IP通道
        
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
        """Initialize RN structures - 现在只初始化基础结构，具体IP位置通过动态挂载创建"""
        # 不再预创建任何位置的RN tracker，改为动态创建
        # 这些结构会在attach_ip时按需创建
        pass

    def setup_rn_trackers(self, ip_type, ip_pos):
        """Setup read and write trackers for RN."""
        self.rn_rdb_recv[ip_type][ip_pos] = []
        self.rn_rdb_count[ip_type][ip_pos] = self.config.RN_RDB_SIZE
        self.rn_rdb_reserve[ip_type][ip_pos] = 0
        self.rn_wdb_count[ip_type][ip_pos] = self.config.RN_WDB_SIZE
        self.rn_wdb_send[ip_type][ip_pos] = []
        self.rn_wdb_reserve[ip_type][ip_pos] = 0

        for req_type in ["read", "write"]:
            self.rn_tracker[req_type][ip_type][ip_pos] = []
            self.rn_tracker_wait[req_type][ip_type][ip_pos] = []
            self.rn_tracker_count[req_type][ip_type][ip_pos] = self.config.RN_R_TRACKER_OSTD if req_type == "read" else self.config.RN_W_TRACKER_OSTD
            self.rn_tracker_pointer[req_type][ip_type][ip_pos] = -1

    def initialize_sn(self):
        """Initialize SN structures - 现在只初始化基础结构，具体IP位置通过动态挂载创建"""
        self.sn_tracker_release_time = defaultdict(list)
        # 不再预创建任何位置的SN tracker，改为动态创建
        # 这些结构会在attach_ip时按需创建

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

    def attach_ip(self, ip_type: str, node_pos: int):
        """动态挂载IP到指定节点位置
        
        Args:
            ip_type: IP类型，如 "sdma_0", "ddr_1"
            node_pos: 物理节点位置
        """
        if node_pos not in self.attached_ips:
            self.attached_ips[node_pos] = set()
        
        if ip_type in self.attached_ips[node_pos]:
            return  # 已经挂载了
        
        # 解析IP类型前缀
        ip_prefix = ip_type.split('_')[0]
        
        try:
            # 动态创建所需的数据结构
            if ip_prefix in ("sdma", "gdma", "cdma"):
                self._create_rn_structures(ip_type, node_pos)
            elif ip_prefix in ("ddr", "l2m"):
                self._create_sn_structures(ip_type, node_pos)
            else:
                print(f"Warning: Unknown IP type prefix: {ip_prefix}")
                return
            
            self.attached_ips[node_pos].add(ip_type)
            print(f"Successfully attached {ip_type} to node {node_pos}")
            
        except Exception as e:
            print(f"Failed to attach {ip_type} to node {node_pos}: {e}")

    def _create_rn_structures(self, ip_type: str, node_pos: int):
        """为RN类型IP创建数据结构"""
        # 确保该IP类型的通道结构存在
        rn_structure_names = [
            'rn_rdb', 'rn_rdb_reserve', 'rn_rdb_recv', 'rn_rdb_count',
            'rn_wdb', 'rn_wdb_reserve', 'rn_wdb_count', 'rn_wdb_send'
        ]
        
        for structure_name in rn_structure_names:
            structure = getattr(self, structure_name)
            if ip_type not in structure:
                structure[ip_type] = {}
        
        # 确保tracker结构存在
        for req_type in ["read", "write"]:
            for tracker_name in ['rn_tracker', 'rn_tracker_wait', 'rn_tracker_count', 'rn_tracker_pointer']:
                tracker = getattr(self, tracker_name)
                if req_type not in tracker:
                    tracker[req_type] = {}
                if ip_type not in tracker[req_type]:
                    tracker[req_type][ip_type] = {}
            
        # 为该位置创建RN结构
        if node_pos not in self.rn_rdb[ip_type]:
            self.rn_rdb[ip_type][node_pos] = defaultdict(list)
            self.rn_wdb[ip_type][node_pos] = defaultdict(list)
            self.setup_rn_trackers(ip_type, node_pos)

    def _create_sn_structures(self, ip_type: str, node_pos: int):
        """为SN类型IP创建数据结构"""
        # 确保该IP类型的通道结构存在
        sn_structure_names = [
            'sn_rdb', 'sn_rsp_queue', 'sn_tracker', 'sn_wdb', 'sn_wdb_recv', 'sn_wdb_count'
        ]
        
        for structure_name in sn_structure_names:
            structure = getattr(self, structure_name)
            if ip_type not in structure:
                structure[ip_type] = {}
        
        # 特殊处理tracker_count结构
        if ip_type not in self.sn_tracker_count:
            self.sn_tracker_count[ip_type] = {"ro": {}, "share": {}}
        
        # 确保req_wait结构存在
        for req_type in ["read", "write"]:
            if req_type not in self.sn_req_wait:
                self.sn_req_wait[req_type] = {}
            if ip_type not in self.sn_req_wait[req_type]:
                self.sn_req_wait[req_type][ip_type] = {}
        
        # 为该位置创建SN结构
        if node_pos not in self.sn_rdb[ip_type]:
            self.sn_rdb[ip_type][node_pos] = []
            self.sn_wdb[ip_type][node_pos] = defaultdict(list)
            self.setup_sn_trackers(ip_type, node_pos)

    def get_attached_ips(self, node_pos: int = None):
        """获取挂载的IP信息
        
        Args:
            node_pos: 指定节点位置，如果为None则返回所有节点的信息
            
        Returns:
            如果指定了node_pos，返回该节点的IP集合；否则返回完整的attached_ips字典
        """
        if node_pos is not None:
            return self.attached_ips.get(node_pos, set())
        return self.attached_ips.copy()

    def is_ip_attached(self, ip_type: str, node_pos: int) -> bool:
        """检查指定IP是否已挂载到指定节点"""
        return ip_type in self.attached_ips.get(node_pos, set())
