"""
路由表模块 - 为CrossRing和Ring拓扑提供灵活的路由管理功能
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import copy

@dataclass
class RouteEntry:
    """路由表条目"""
    destination: int
    next_hop: int
    path: List[int]
    metric: float = 1.0  # 路由度量值（距离、延迟等）
    priority: int = 1    # 路由优先级（1=最高）
    direction: str = ""  # Ring拓扑专用：CW/CCW/LOCAL
    
    def __post_init__(self):
        """初始化后验证"""
        if not self.path or self.path[0] != self.destination:
            # 如果path为空或第一个节点不是destination，重新构建path
            if self.next_hop == self.destination:
                self.path = [self.destination]
            else:
                # 这里需要完整路径计算，暂时设为简单路径
                self.path = [self.next_hop, self.destination]


class RouteTable:
    """路由表类 - 支持静态和动态路由"""
    
    def __init__(self, node_id: int, topology_type: str = "CrossRing"):
        self.node_id = node_id
        self.topology_type = topology_type
        
        # 主路由表：destination -> RouteEntry
        self.routes: Dict[int, RouteEntry] = {}
        
        # 备用路由表：destination -> List[RouteEntry] (按优先级排序)
        self.backup_routes: Dict[int, List[RouteEntry]] = defaultdict(list)
        
        # 动态路由缓存：(destination, context_hash) -> RouteEntry
        self.dynamic_cache: Dict[Tuple[int, int], RouteEntry] = {}
        
        # 路由统计信息
        self.stats = {
            "route_lookups": 0,
            "cache_hits": 0,
            "primary_route_usage": 0,
            "backup_route_usage": 0,
            "failed_lookups": 0
        }
        
        # 配置参数
        self.enable_backup_routes = True
        self.enable_dynamic_routing = True
        self.cache_size_limit = 1000
        
        logging.info(f"RouteTable initialized for node {node_id}, topology: {topology_type}")
    
    def add_route_with_validation(self, destination: int, next_hop: int, path: List[int] = None, 
                                  metric: float = 1.0, priority: int = 1, direction: str = "") -> bool:
        """添加路由条目并进行完整验证和处理"""
        # 输入验证
        if destination < 0 or next_hop < 0:
            logging.error(f"Invalid destination {destination} or next_hop {next_hop}")
            return False
        
        if destination == self.node_id:
            logging.warning(f"Cannot add route to self node {self.node_id}")
            return False
        
        # 路径构建和验证
        if path is None:
            if next_hop == destination:
                path = [destination]
            else:
                path = [next_hop, destination]
        
        # 验证路径完整性
        if not path or path[-1] != destination:
            logging.error(f"Invalid path {path} for destination {destination}")
            return False
        
        # 创建路由条目
        route_entry = RouteEntry(
            destination=destination,
            next_hop=next_hop, 
            path=path,
            metric=metric,
            priority=priority,
            direction=direction
        )
        
        # 路由表更新逻辑
        if priority == 1 or destination not in self.routes:
            # 如果已有主路由且新路由优先级更高，将旧路由降级为备用路由
            if destination in self.routes and priority == 1:
                old_route = self.routes[destination]
                if self.enable_backup_routes:
                    old_route.priority = 2
                    self.backup_routes[destination].append(old_route)
                    self.backup_routes[destination].sort(key=lambda r: r.priority)
            
            self.routes[destination] = route_entry
        
        # 备用路由处理
        elif self.enable_backup_routes and priority > 1:
            self.backup_routes[destination].append(route_entry)
            self.backup_routes[destination].sort(key=lambda r: r.priority)
            
            # 限制备用路由数量，避免内存过度使用
            max_backup_routes = getattr(self, 'max_backup_routes_per_destination', 5)
            if len(self.backup_routes[destination]) > max_backup_routes:
                self.backup_routes[destination] = self.backup_routes[destination][:max_backup_routes]
        
        # 清理相关缓存
        cache_keys_to_remove = [key for key in self.dynamic_cache.keys() if key[0] == destination]
        for key in cache_keys_to_remove:
            del self.dynamic_cache[key]
            
        logging.debug(f"Successfully added route: {self.node_id}->{destination} via {next_hop} (priority {priority}, metric {metric})")
        return True
    
    def lookup_route_with_context_analysis(self, destination: int, context: Dict[str, Any] = None) -> Optional[RouteEntry]:
        """智能路由查询，结合上下文信息进行最佳路由选择"""
        self.stats["route_lookups"] += 1
        
        # 本地节点直接返回
        if destination == self.node_id:
            return RouteEntry(destination=self.node_id, next_hop=self.node_id, path=[self.node_id], direction="LOCAL")
        
        # 构建上下文信息用于缓存和路由决策
        context_info = context or {}
        congestion_threshold = context_info.get("congestion_threshold", 0.8)
        load_balancing_enabled = context_info.get("load_balancing", True)
        adaptive_routing_enabled = context_info.get("adaptive_routing", self.enable_dynamic_routing)
        
        # 动态路由缓存查询
        cache_key = None
        if adaptive_routing_enabled and context_info:
            context_hash = hash(frozenset(context_info.items()))
            cache_key = (destination, context_hash)
            if cache_key in self.dynamic_cache:
                cached_route = self.dynamic_cache[cache_key]
                if self._is_route_available(cached_route, context_info):
                    self.stats["cache_hits"] += 1
                    return cached_route
                else:
                    # 缓存的路由不可用，删除缓存条目
                    del self.dynamic_cache[cache_key]
        
        # 收集所有可用路由候选
        route_candidates = []
        
        # 添加主路由
        if destination in self.routes:
            primary_route = self.routes[destination]
            if self._is_route_available(primary_route, context_info):
                route_candidates.append((primary_route, "primary"))
        
        # 添加备用路由
        if self.enable_backup_routes and destination in self.backup_routes:
            for backup_route in self.backup_routes[destination]:
                if self._is_route_available(backup_route, context_info):
                    route_candidates.append((backup_route, "backup"))
        
        # 如果没有可用路由，返回失败
        if not route_candidates:
            self.stats["failed_lookups"] += 1
            logging.warning(f"No available route found from node {self.node_id} to {destination}")
            return None
        
        # 路由选择策略
        selected_route = None
        route_type = None
        
        if load_balancing_enabled and len(route_candidates) > 1:
            # 负载均衡模式：根据拥塞情况选择最佳路由
            best_route = None
            best_score = float('inf')
            
            for route, r_type in route_candidates:
                # 计算路由评分（越低越好）
                score = route.metric
                
                # 考虑拥塞情况
                if context_info:
                    congestion_key = f"link_congestion_{route.next_hop}"
                    congestion = context_info.get(congestion_key, 0.0)
                    score += congestion * 10  # 拥塞惩罚
                    
                    # Ring拓扑特殊处理：考虑方向均衡
                    if self.topology_type.startswith("Ring"):
                        direction_bias = context_info.get(f"direction_bias_{route.direction}", 0.0)
                        score += direction_bias * 5
                
                if score < best_score:
                    best_score = score
                    best_route = route
                    route_type = r_type
            
            selected_route = best_route
        else:
            # 简单模式：优先选择主路由
            for route, r_type in route_candidates:
                if r_type == "primary":
                    selected_route = route
                    route_type = r_type
                    break
            
            # 如果没有主路由，选择第一个备用路由
            if selected_route is None:
                selected_route, route_type = route_candidates[0]
        
        # 更新统计信息
        if route_type == "primary":
            self.stats["primary_route_usage"] += 1
        else:
            self.stats["backup_route_usage"] += 1
        
        # 缓存选择的路由
        if adaptive_routing_enabled and cache_key and selected_route:
            self._cache_dynamic_route_with_cleanup(cache_key, selected_route)
        
        return selected_route
    
    def update_and_manage_routes(self, destination: int, new_metric: float = None, 
                               remove_route: bool = False, remove_backups: bool = False) -> Dict[str, Any]:
        """综合路由管理：更新度量值、删除路由、缓存清理和统计信息"""
        result = {
            "success": False,
            "operation": "",
            "affected_routes": 0,
            "cache_entries_cleared": 0,
            "route_counts": {}
        }
        
        # 删除路由操作
        if remove_route:
            result["operation"] = "remove"
            removed_count = 0
            
            # 删除主路由
            if destination in self.routes:
                del self.routes[destination]
                removed_count += 1
                logging.info(f"Removed primary route to destination {destination}")
            
            # 删除备用路由
            if remove_backups and destination in self.backup_routes:
                backup_count = len(self.backup_routes[destination])
                del self.backup_routes[destination]
                removed_count += backup_count
                logging.info(f"Removed {backup_count} backup routes to destination {destination}")
            
            # 清理相关的动态缓存
            cache_keys_to_remove = [key for key in self.dynamic_cache.keys() if key[0] == destination]
            for key in cache_keys_to_remove:
                del self.dynamic_cache[key]
            
            result["affected_routes"] = removed_count
            result["cache_entries_cleared"] = len(cache_keys_to_remove)
            result["success"] = removed_count > 0
        
        # 更新度量值操作
        elif new_metric is not None:
            result["operation"] = "update_metric"
            
            # 更新主路由度量值
            if destination in self.routes:
                old_metric = self.routes[destination].metric
                self.routes[destination].metric = new_metric
                result["affected_routes"] += 1
                logging.info(f"Updated metric for destination {destination}: {old_metric} -> {new_metric}")
            
            # 更新备用路由度量值
            if destination in self.backup_routes:
                for backup_route in self.backup_routes[destination]:
                    backup_route.metric = new_metric
                    result["affected_routes"] += 1
                
                # 重新排序备用路由（按优先级和度量值）
                self.backup_routes[destination].sort(key=lambda r: (r.priority, r.metric))
            
            # 清理相关缓存以强制重新计算路由
            cache_keys_to_remove = [key for key in self.dynamic_cache.keys() if key[0] == destination]
            for key in cache_keys_to_remove:
                del self.dynamic_cache[key]
            
            result["cache_entries_cleared"] = len(cache_keys_to_remove)
            result["success"] = result["affected_routes"] > 0
        
        # 获取统计信息（总是执行）
        destinations = set(self.routes.keys())
        destinations.update(self.backup_routes.keys())
        
        result["route_counts"] = {
            "total_destinations": len(destinations),
            "primary_routes": len(self.routes),
            "backup_routes": sum(len(routes) for routes in self.backup_routes.values()),
            "cached_routes": len(self.dynamic_cache),
            "all_destinations": list(destinations)
        }
        
        return result
    
    def export_routes(self) -> Dict:
        """导出路由表为字典格式"""
        export_data = {
            "node_id": self.node_id,
            "topology_type": self.topology_type,
            "primary_routes": {},
            "backup_routes": {},
            "stats": self.stats.copy()
        }
        
        # 导出主路由
        for dest, route in self.routes.items():
            export_data["primary_routes"][dest] = {
                "destination": route.destination,
                "next_hop": route.next_hop,
                "path": route.path,
                "metric": route.metric,
                "priority": route.priority,
                "direction": route.direction
            }
        
        # 导出备用路由
        for dest, routes in self.backup_routes.items():
            export_data["backup_routes"][dest] = []
            for route in routes:
                export_data["backup_routes"][dest].append({
                    "destination": route.destination,
                    "next_hop": route.next_hop,
                    "path": route.path,
                    "metric": route.metric,
                    "priority": route.priority,
                    "direction": route.direction
                })
        
        return export_data
    
    def import_routes(self, route_data: Dict) -> bool:
        """从字典导入路由表"""
        try:
            self.node_id = route_data.get("node_id", self.node_id)
            self.topology_type = route_data.get("topology_type", self.topology_type)
            
            # 导入主路由
            primary_routes = route_data.get("primary_routes", {})
            for dest_str, route_info in primary_routes.items():
                dest = int(dest_str)
                self.add_route_with_validation(
                    destination=dest,
                    next_hop=route_info["next_hop"],
                    path=route_info["path"],
                    metric=route_info.get("metric", 1.0),
                    priority=route_info.get("priority", 1),
                    direction=route_info.get("direction", "")
                )
            
            # 导入备用路由
            backup_routes = route_data.get("backup_routes", {})
            for dest_str, routes_list in backup_routes.items():
                dest = int(dest_str)
                for route_info in routes_list:
                    self.add_route_with_validation(
                        destination=dest,
                        next_hop=route_info["next_hop"],
                        path=route_info["path"],
                        metric=route_info.get("metric", 1.0),
                        priority=route_info.get("priority", 2),
                        direction=route_info.get("direction", "")
                    )
            
            logging.info(f"Successfully imported routes for node {self.node_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to import routes: {e}")
            return False
    
    def save_to_file(self, file_path: str) -> bool:
        """保存路由表到文件"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.export_routes(), f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to save routes to {file_path}: {e}")
            return False
    
    def load_from_file(self, file_path: str) -> bool:
        """从文件加载路由表"""
        try:
            with open(file_path, 'r') as f:
                route_data = json.load(f)
            return self.import_routes(route_data)
        except Exception as e:
            logging.error(f"Failed to load routes from {file_path}: {e}")
            return False
    
    def clear_cache(self):
        """清空动态路由缓存"""
        self.dynamic_cache.clear()
        
    def get_statistics(self) -> Dict:
        """获取路由表统计信息"""
        stats = self.stats.copy()
        
        # 添加路由数量统计
        destinations = set(self.routes.keys())
        destinations.update(self.backup_routes.keys())
        
        route_counts = {
            "total_destinations": len(destinations),
            "primary_routes": len(self.routes),
            "backup_routes": sum(len(routes) for routes in self.backup_routes.values()),
            "cached_routes": len(self.dynamic_cache)
        }
        stats.update(route_counts)
        
        # 计算缓存命中率
        if stats["route_lookups"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["route_lookups"]
        else:
            stats["cache_hit_rate"] = 0.0
            
        return stats
    
    def _is_route_available(self, route: RouteEntry, context: Dict[str, Any] = None) -> bool:
        """检查路由是否可用（可扩展拥塞检测等）"""
        # 基本可用性检查
        if not route or not route.path:
            return False
            
        # 这里可以添加更复杂的可用性检查逻辑
        # 比如拥塞检测、链路状态检查等
        if context:
            # 示例：检查拥塞水平
            congestion_threshold = context.get("congestion_threshold", 0.8)
            route_congestion = context.get(f"link_congestion_{route.next_hop}", 0.0)
            if route_congestion > congestion_threshold:
                return False
        
        return True
    
    def _cache_dynamic_route_with_cleanup(self, cache_key: Tuple[int, int], route: RouteEntry):
        """缓存动态路由结果并进行清理管理"""
        # 缓存大小管理
        if len(self.dynamic_cache) >= self.cache_size_limit:
            # 删除最老的缓存条目（简单LRU）
            keys_to_remove = list(self.dynamic_cache.keys())[:self.cache_size_limit // 4]
            for old_key in keys_to_remove:
                del self.dynamic_cache[old_key]
        
        # 深拷贝路由条目以避免引用问题
        cached_route = copy.deepcopy(route)
        self.dynamic_cache[cache_key] = cached_route


class DistributedRouteManager:
    """分布式路由管理器 - 管理所有节点的路由表"""
    
    def __init__(self, topology_type: str = "CrossRing"):
        self.topology_type = topology_type
        self.node_route_tables: Dict[int, RouteTable] = {}
        self.global_topology: Dict[int, List[int]] = {}  # 节点邻接表
        
    def add_node(self, node_id: int) -> RouteTable:
        """添加节点并创建其路由表"""
        if node_id not in self.node_route_tables:
            self.node_route_tables[node_id] = RouteTable(node_id, self.topology_type)
        return self.node_route_tables[node_id]
    
    def get_route_table(self, node_id: int) -> Optional[RouteTable]:
        """获取指定节点的路由表"""
        return self.node_route_tables.get(node_id)
    
    def set_topology(self, adjacency_matrix: List[List[int]]):
        """设置网络拓扑"""
        self.global_topology.clear()
        for i, row in enumerate(adjacency_matrix):
            self.global_topology[i] = [j for j, connected in enumerate(row) if connected]
    
    def build_comprehensive_routing_tables(self, routing_strategy: str = "shortest_path", 
                                          enable_backup_routes: bool = True) -> Dict[str, Any]:
        """基于拓扑构建所有节点的完整路由表，支持多种策略和备用路由"""
        result = {
            "strategy": routing_strategy,
            "nodes_processed": 0,
            "total_routes": 0,
            "backup_routes": 0,
            "processing_time": 0,
            "success": False
        }
        
        start_time = time.time()
        
        try:
            if routing_strategy == "shortest_path":
                route_stats = self._build_shortest_path_routes_with_backups(enable_backup_routes)
            elif routing_strategy == "ring_balanced":
                route_stats = self._build_ring_balanced_routes_with_load_balancing(enable_backup_routes)
            elif routing_strategy == "adaptive":
                route_stats = self._build_adaptive_routes_with_congestion_awareness(enable_backup_routes)
            else:
                logging.error(f"Unknown routing strategy: {routing_strategy}")
                return result
            
            result.update(route_stats)
            result["success"] = True
            result["processing_time"] = time.time() - start_time
            
            logging.info(f"Successfully built routing tables using {routing_strategy} strategy: "
                        f"{result['nodes_processed']} nodes, {result['total_routes']} total routes")
            
        except Exception as e:
            logging.error(f"Failed to build routing tables: {e}")
            result["error"] = str(e)
            result["processing_time"] = time.time() - start_time
        
        return result
    
    def _build_shortest_path_routes_with_backups(self, enable_backup_routes: bool = True) -> Dict[str, int]:
        """构建最短路径路由"""
        # 使用Floyd-Warshall算法计算所有节点对之间的最短路径
        nodes = list(self.global_topology.keys())
        n = len(nodes)
        
        # 初始化距离矩阵和下一跳矩阵
        dist = [[float('inf')] * n for _ in range(n)]
        next_hop = [[None] * n for _ in range(n)]
        
        # 设置直接连接和自环
        for i in range(n):
            dist[i][i] = 0
            for neighbor in self.global_topology.get(nodes[i], []):
                j = nodes.index(neighbor)
                dist[i][j] = 1
                next_hop[i][j] = neighbor
        
        # Floyd-Warshall算法计算最短路径
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_hop[i][j] = next_hop[i][k]
        
        # 构建主路由和备用路由
        total_routes = 0
        backup_routes = 0
        
        for i, source in enumerate(nodes):
            route_table = self.add_node(source)
            route_table.enable_backup_routes = enable_backup_routes
            
            for j, destination in enumerate(nodes):
                if i != j and next_hop[i][j] is not None:
                    # 重构完整路径
                    path = self._reconstruct_shortest_path(source, destination, next_hop, nodes)
                    
                    # 添加主路由（最短路径）
                    route_table.add_route_with_validation(
                        destination=destination,
                        next_hop=next_hop[i][j],
                        path=path,
                        metric=dist[i][j],
                        priority=1,
                        direction="SHORTEST"
                    )
                    total_routes += 1
                    
                    # 添加备用路由（如果启用且存在多条路径）
                    if enable_backup_routes and dist[i][j] > 1:
                        backup_path = self._find_alternative_path(source, destination, nodes, next_hop, dist)
                        if backup_path and len(backup_path) > 0:
                            backup_next_hop = backup_path[0] if len(backup_path) > 1 else destination
                            route_table.add_route_with_validation(
                                destination=destination,
                                next_hop=backup_next_hop,
                                path=backup_path,
                                metric=len(backup_path) - 1,
                                priority=2,
                                direction="BACKUP"
                            )
                            backup_routes += 1
        
        return {
            "nodes_processed": len(nodes),
            "total_routes": total_routes,
            "backup_routes": backup_routes
        }
    
    def _build_ring_balanced_routes_with_load_balancing(self, enable_backup_routes: bool = True) -> Dict[str, int]:
        """为Ring拓扑构建负载均衡路由"""
        nodes = sorted(self.global_topology.keys())
        ring_size = len(nodes)
        total_routes = 0
        backup_routes = 0
        
        for source in nodes:
            route_table = self.add_node(source)
            route_table.enable_backup_routes = enable_backup_routes
            
            for destination in nodes:
                if source == destination:
                    continue
                
                # 计算顺时针和逆时针距离
                cw_distance = (destination - source + ring_size) % ring_size
                ccw_distance = (source - destination + ring_size) % ring_size
                
                # 构建顺时针路径
                cw_path = [source]
                current = source
                while current != destination:
                    current = (current + 1) % ring_size
                    cw_path.append(current)
                
                # 构建逆时针路径
                ccw_path = [source]
                current = source
                while current != destination:
                    current = (current - 1 + ring_size) % ring_size
                    ccw_path.append(current)
                
                # 选择主路由（最短路径）和备用路由
                if cw_distance <= ccw_distance:
                    # 顺时针为主路由
                    cw_next_hop = (source + 1) % ring_size
                    route_table.add_route_with_validation(
                        destination, cw_next_hop, cw_path, cw_distance, 1, "CW"
                    )
                    total_routes += 1
                    
                    # 逆时针为备用路由（如果启用且距离不太远）
                    if enable_backup_routes and ccw_distance < ring_size - 1:
                        ccw_next_hop = (source - 1 + ring_size) % ring_size
                        route_table.add_route_with_validation(
                            destination, ccw_next_hop, ccw_path, ccw_distance, 2, "CCW"
                        )
                        backup_routes += 1
                else:
                    # 逆时针为主路由
                    ccw_next_hop = (source - 1 + ring_size) % ring_size
                    route_table.add_route_with_validation(
                        destination, ccw_next_hop, ccw_path, ccw_distance, 1, "CCW"
                    )
                    total_routes += 1
                    
                    # 顺时针为备用路由
                    if enable_backup_routes and cw_distance < ring_size - 1:
                        cw_next_hop = (source + 1) % ring_size
                        route_table.add_route_with_validation(
                            destination, cw_next_hop, cw_path, cw_distance, 2, "CW"
                        )
                        backup_routes += 1
        
        return {
            "nodes_processed": len(nodes),
            "total_routes": total_routes,
            "backup_routes": backup_routes
        }
    
    def _build_adaptive_routes_with_congestion_awareness(self, enable_backup_routes: bool = True) -> Dict[str, int]:
        """构建自适应路由，支持拥塞感知"""
        # 先构建基础最短路径路由
        base_stats = self._build_shortest_path_routes_with_backups(enable_backup_routes)
        
        # 为每个节点添加自适应路由能力
        for node_id, route_table in self.node_route_tables.items():
            route_table.enable_dynamic_routing = True
            route_table.cache_size_limit = 2000  # 增大缓存以支持动态路由
        
        return base_stats
    
    def _reconstruct_shortest_path(self, source: int, destination: int, 
                                 next_hop: List[List[int]], nodes: List[int]) -> List[int]:
        """重构最短路径"""
        path = [source]
        current = source
        
        while current != destination:
            i = nodes.index(current)
            j = nodes.index(destination)
            next_node = next_hop[i][j]
            if next_node is None:
                break
            path.append(next_node)
            current = next_node
            
            # 防止无限循环
            if len(path) > len(nodes):
                break
        
        return path
    
    def _find_alternative_path(self, source: int, destination: int, nodes: List[int],
                             next_hop: List[List[int]], dist: List[List[float]]) -> Optional[List[int]]:
        """寻找备用路径（简化实现）"""
        # 这里可以实现更复杂的备用路径算法
        # 简化版本：如果存在间接路径，返回一个较长的路径
        source_idx = nodes.index(source)
        dest_idx = nodes.index(destination)
        
        # 寻找中间节点，构建间接路径
        for k, intermediate in enumerate(nodes):
            if k != source_idx and k != dest_idx:
                if (dist[source_idx][k] != float('inf') and 
                    dist[k][dest_idx] != float('inf') and
                    dist[source_idx][k] + dist[k][dest_idx] > dist[source_idx][dest_idx]):
                    
                    # 构建通过中间节点的路径
                    path_to_intermediate = self._reconstruct_shortest_path(source, intermediate, next_hop, nodes)
                    path_from_intermediate = self._reconstruct_shortest_path(intermediate, destination, next_hop, nodes)[1:]
                    
                    alternative_path = path_to_intermediate + path_from_intermediate
                    if len(alternative_path) > 2:  # 确保是真正的备用路径
                        return alternative_path
        
        return None
        
        # 初始化距离矩阵
        dist = [[float('inf')] * n for _ in range(n)]
        next_hop = [[None] * n for _ in range(n)]
        
        # 设置直接连接
        for i in range(n):
            dist[i][i] = 0
            for neighbor in self.global_topology.get(nodes[i], []):
                j = nodes.index(neighbor)
                dist[i][j] = 1
                next_hop[i][j] = neighbor
        
        # Floyd-Warshall算法
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_hop[i][j] = next_hop[i][k]
        
        # 构建路由表
        for i, source in enumerate(nodes):
            route_table = self.add_node(source)
            for j, destination in enumerate(nodes):
                if i != j and next_hop[i][j] is not None:
                    # 重构完整路径
                    path = self._reconstruct_path(source, destination, next_hop, nodes)
                    route_table.add_route_with_validation(
                        destination=destination,
                        next_hop=next_hop[i][j],
                        path=path,
                        metric=dist[i][j]
                    )
    
    def _build_ring_balanced_routes(self):
        """为Ring拓扑构建负载均衡路由"""
        nodes = sorted(self.global_topology.keys())
        ring_size = len(nodes)
        
        for source in nodes:
            route_table = self.add_node(source)
            
            for destination in nodes:
                if source == destination:
                    continue
                
                # 计算顺时针和逆时针距离
                cw_distance = (destination - source + ring_size) % ring_size
                ccw_distance = (source - destination + ring_size) % ring_size
                
                # 构建顺时针路径
                cw_path = []
                current = source
                while current != destination:
                    current = (current + 1) % ring_size
                    cw_path.append(current)
                cw_next_hop = (source + 1) % ring_size
                
                # 构建逆时针路径
                ccw_path = []
                current = source
                while current != destination:
                    current = (current - 1 + ring_size) % ring_size
                    ccw_path.append(current)
                ccw_next_hop = (source - 1 + ring_size) % ring_size
                
                # 添加主路由（最短路径）
                if cw_distance <= ccw_distance:
                    route_table.add_route_with_validation(destination, cw_next_hop, cw_path, cw_distance, 1, "CW")
                    # 添加备用路由
                    if ccw_distance < ring_size - 1:  # 避免添加过长的备用路由
                        route_table.add_route_with_validation(destination, ccw_next_hop, ccw_path, ccw_distance, 2, "CCW")
                else:
                    route_table.add_route_with_validation(destination, ccw_next_hop, ccw_path, ccw_distance, 1, "CCW")
                    # 添加备用路由
                    if cw_distance < ring_size - 1:
                        route_table.add_route_with_validation(destination, cw_next_hop, cw_path, cw_distance, 2, "CW")
    
    def _reconstruct_path(self, source: int, destination: int, next_hop: List[List[int]], nodes: List[int]) -> List[int]:
        """重构路径"""
        path = []
        current = source
        while current != destination:
            i = nodes.index(current)
            j = nodes.index(destination)
            current = next_hop[i][j]
            if current is None:
                break
            path.append(current)
        return path
    
    def export_all_routes(self) -> Dict:
        """导出所有节点的路由表"""
        export_data = {
            "topology_type": self.topology_type,
            "nodes": {}
        }
        
        for node_id, route_table in self.node_route_tables.items():
            export_data["nodes"][node_id] = route_table.export_routes()
        
        return export_data
    
    def import_all_routes(self, route_data: Dict) -> bool:
        """导入所有节点的路由表"""
        try:
            self.topology_type = route_data.get("topology_type", self.topology_type)
            
            nodes_data = route_data.get("nodes", {})
            for node_id_str, node_route_data in nodes_data.items():
                node_id = int(node_id_str)
                route_table = self.add_node(node_id)
                route_table.import_routes(node_route_data)
            
            return True
        except Exception as e:
            logging.error(f"Failed to import all routes: {e}")
            return False
    
    def get_global_statistics(self) -> Dict:
        """获取全局路由统计"""
        global_stats = {
            "total_nodes": len(self.node_route_tables),
            "total_primary_routes": 0,
            "total_backup_routes": 0,
            "total_route_lookups": 0,
            "total_cache_hits": 0,
            "nodes_stats": {}
        }
        
        for node_id, route_table in self.node_route_tables.items():
            node_stats = route_table.get_statistics()
            global_stats["nodes_stats"][node_id] = node_stats
            global_stats["total_primary_routes"] += node_stats["primary_routes"]
            global_stats["total_backup_routes"] += node_stats["backup_routes"]
            global_stats["total_route_lookups"] += node_stats["route_lookups"]
            global_stats["total_cache_hits"] += node_stats["cache_hits"]
        
        if global_stats["total_route_lookups"] > 0:
            global_stats["global_cache_hit_rate"] = global_stats["total_cache_hits"] / global_stats["total_route_lookups"]
        else:
            global_stats["global_cache_hit_rate"] = 0.0
        
        return global_stats