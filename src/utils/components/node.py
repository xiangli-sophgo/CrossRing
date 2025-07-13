"""
Node class for NoC simulation.
Handles network node functionality including RN/SN trackers and databases.
Enhanced with route table support for flexible routing.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Optional, Dict, List, Any
from config.config import CrossRingConfig

# Import route table - handle import error gracefully
try:
    from .route_table import RouteTable, RouteEntry
except ImportError:
    # Fallback for cases where route_table is not available
    RouteTable = None
    RouteEntry = None


class Node:
    global_packet_id = -1

    def __init__(self, config: CrossRingConfig, node_id: int = 0):
        self.config = config
        self.node_id = node_id
        
        # Initialize route table if available
        self.route_table: Optional[RouteTable] = None
        if RouteTable is not None:
            topology_type = getattr(config, 'TOPO_TYPE', 'CrossRing')
            self.route_table = RouteTable(node_id, topology_type)
        
        # Route caching for performance
        self.route_cache: Dict[int, RouteEntry] = {}
        self.enable_route_caching = True
        
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

    # ====================== Route Table Methods ======================
    
    def lookup_route(self, destination: int, context: Dict[str, Any] = None) -> Optional[RouteEntry]:
        """查询到目标节点的路由"""
        if destination == self.node_id:
            # 本地节点，创建本地路由条目
            if RouteEntry is not None:
                return RouteEntry(
                    destination=self.node_id,
                    next_hop=self.node_id,
                    path=[self.node_id],
                    direction="LOCAL"
                )
            return None
        
        # 先检查缓存
        if self.enable_route_caching and destination in self.route_cache:
            return self.route_cache[destination]
        
        # 使用路由表查询
        if self.route_table is not None:
            route = self.route_table.lookup_route(destination, context)
            
            # 缓存结果
            if route and self.enable_route_caching:
                self.route_cache[destination] = route
            
            return route
        
        # 路由表不可用时的回退机制
        return self._fallback_routing(destination)
    
    def _fallback_routing(self, destination: int) -> Optional[RouteEntry]:
        """路由表不可用时的回退路由机制"""
        # 这里可以实现简单的路由逻辑，比如基于拓扑的直接路由
        # 为了兼容性，返回一个基本路由条目
        if RouteEntry is not None:
            return RouteEntry(
                destination=destination,
                next_hop=destination,  # 简化：直接跳转到目标
                path=[destination],
                direction="DIRECT"
            )
        return None
    
    def add_route(self, destination: int, next_hop: int, path: List[int] = None, 
                  metric: float = 1.0, priority: int = 1, direction: str = "") -> bool:
        """添加路由条目"""
        if self.route_table is not None:
            success = self.route_table.add_route(destination, next_hop, path, metric, priority, direction)
            
            # 清除相关缓存
            if success and destination in self.route_cache:
                del self.route_cache[destination]
            
            return success
        return False
    
    def remove_route(self, destination: int, remove_backups: bool = False) -> bool:
        """删除路由条目"""
        if self.route_table is not None:
            success = self.route_table.remove_route(destination, remove_backups)
            
            # 清除相关缓存
            if success and destination in self.route_cache:
                del self.route_cache[destination]
            
            return success
        return False
    
    def update_route_metric(self, destination: int, new_metric: float) -> bool:
        """更新路由度量值"""
        if self.route_table is not None:
            success = self.route_table.update_route_metric(destination, new_metric)
            
            # 清除相关缓存以强制重新查询
            if success and destination in self.route_cache:
                del self.route_cache[destination]
            
            return success
        return False
    
    def get_next_hop(self, destination: int, context: Dict[str, Any] = None) -> Optional[int]:
        """获取到目标节点的下一跳"""
        route = self.lookup_route(destination, context)
        return route.next_hop if route else None
    
    def get_route_direction(self, destination: int, context: Dict[str, Any] = None) -> Optional[str]:
        """获取路由方向（Ring拓扑专用）"""
        route = self.lookup_route(destination, context)
        return route.direction if route else None
    
    def get_full_path(self, destination: int, context: Dict[str, Any] = None) -> Optional[List[int]]:
        """获取到目标节点的完整路径"""
        route = self.lookup_route(destination, context)
        return route.path if route else None
    
    def clear_route_cache(self):
        """清空路由缓存"""
        self.route_cache.clear()
    
    def set_route_caching(self, enabled: bool):
        """启用或禁用路由缓存"""
        self.enable_route_caching = enabled
        if not enabled:
            self.clear_route_cache()
    
    def get_route_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        if self.route_table is not None:
            stats = self.route_table.get_statistics()
            stats['cache_size'] = len(self.route_cache)
            stats['cache_enabled'] = self.enable_route_caching
            return stats
        else:
            return {
                'route_table_available': False,
                'cache_size': len(self.route_cache),
                'cache_enabled': self.enable_route_caching
            }
    
    def export_routes(self) -> Optional[Dict]:
        """导出路由表"""
        if self.route_table is not None:
            return self.route_table.export_routes()
        return None
    
    def import_routes(self, route_data: Dict) -> bool:
        """导入路由表"""
        if self.route_table is not None:
            success = self.route_table.import_routes(route_data)
            if success:
                self.clear_route_cache()  # 清空缓存以使用新路由
            return success
        return False
    
    def has_route_to(self, destination: int) -> bool:
        """检查是否有到目标节点的路由"""
        if destination == self.node_id:
            return True
        
        if self.route_table is not None:
            return destination in self.route_table.get_all_destinations()
        
        return False  # 没有路由表时默认返回False
    
    def get_all_destinations(self) -> List[int]:
        """获取所有可达目标节点"""
        if self.route_table is not None:
            destinations = self.route_table.get_all_destinations()
            if self.node_id not in destinations:
                destinations.append(self.node_id)  # 确保包含本地节点
            return destinations
        return [self.node_id]  # 没有路由表时只能到达本地节点