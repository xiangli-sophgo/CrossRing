#!/usr/bin/env python3
"""
路由表使用示例 - 演示如何在CrossRing项目中使用路由表功能

本示例展示：
1. 创建和配置路由表
2. 添加路由条目
3. 查询路由和路径
4. 使用分布式路由管理器
5. Ring拓扑的路由配置
6. 路由表的导入导出
7. 动态路由和负载均衡
"""

import sys
import os
import logging
import json
from typing import Dict, List

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.utils.components.route_table import RouteTable, DistributedRouteManager, RouteEntry
    from config.config import CrossRingConfig
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_basic_route_table():
    """演示基本路由表操作"""
    print("\n" + "="*50)
    print("演示1: 基本路由表操作")
    print("="*50)
    
    # 创建路由表
    node_id = 0
    route_table = RouteTable(node_id, "CrossRing_4x4")
    
    # 添加几条路由
    print("添加路由条目...")
    route_table.add_route_with_validation(
        destination=1, 
        next_hop=1, 
        path=[1], 
        metric=1.0, 
        priority=1, 
        direction="DIRECT"
    )
    
    route_table.add_route_with_validation(
        destination=2, 
        next_hop=1, 
        path=[1, 2], 
        metric=2.0, 
        priority=1, 
        direction="VIA_1"
    )
    
    route_table.add_route_with_validation(
        destination=3, 
        next_hop=1, 
        path=[1, 2, 3], 
        metric=3.0, 
        priority=1, 
        direction="VIA_1_2"
    )
    
    # 查询路由
    print("\n查询路由:")
    for dest in [1, 2, 3, 4]:
        route = route_table.lookup_route_with_context_analysis(dest)
        if route:
            print(f"到节点{dest}: 下一跳={route.next_hop}, 路径={route.path}, 方向={route.direction}")
        else:
            print(f"到节点{dest}: 无路由")
    
    # 显示统计信息
    print(f"\n路由统计: {route_table.get_statistics()}")
    
    return route_table

def demo_ring_topology_routing():
    """演示Ring拓扑路由"""
    print("\n" + "="*50)
    print("演示2: Ring拓扑路由")
    print("="*50)
    
    # 创建8节点Ring拓扑的分布式路由管理器
    route_manager = DistributedRouteManager("Ring_8")
    
    # 创建Ring拓扑邻接矩阵
    ring_size = 8
    adjacency_matrix = [[0] * ring_size for _ in range(ring_size)]
    
    for i in range(ring_size):
        next_node = (i + 1) % ring_size
        prev_node = (i - 1) % ring_size
        adjacency_matrix[i][next_node] = 1  # 顺时针连接
        adjacency_matrix[i][prev_node] = 1  # 逆时针连接
    
    # 设置拓扑并构建路由表
    route_manager.set_topology(adjacency_matrix)
    
    print("构建Ring负载均衡路由...")
    build_result = route_manager.build_comprehensive_routing_tables("ring_balanced", True)
    print(f"构建结果: {build_result}")
    
    # 演示路由查询
    print("\n演示路由查询:")
    source_node = 0
    route_table = route_manager.get_route_table(source_node)
    
    if route_table:
        for dest in [2, 4, 6]:
            print(f"\n从节点{source_node}到节点{dest}:")
            
            # 普通查询
            route = route_table.lookup_route_with_context_analysis(dest)
            if route:
                print(f"  主路由: 下一跳={route.next_hop}, 路径={route.path}, 方向={route.direction}")
            
            # 带拥塞上下文的查询
            context = {
                "load_balancing": True,
                "congestion_threshold": 0.6,
                f"link_congestion_{route.next_hop}": 0.8,  # 模拟拥塞
                "direction_bias_CW": 0.2,
                "direction_bias_CCW": 0.1
            }
            
            adaptive_route = route_table.lookup_route_with_context_analysis(dest, context)
            if adaptive_route:
                print(f"  自适应路由: 下一跳={adaptive_route.next_hop}, 路径={adaptive_route.path}, 方向={adaptive_route.direction}")
    
    return route_manager

def demo_crossring_topology_routing():
    """演示CrossRing拓扑路由"""
    print("\n" + "="*50)
    print("演示3: CrossRing拓扑路由")
    print("="*50)
    
    # 创建4x4 CrossRing拓扑
    route_manager = DistributedRouteManager("CrossRing_4x4")
    
    # 简化的4x4网格拓扑邻接矩阵
    grid_size = 4
    num_nodes = grid_size * grid_size
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    
    # 构建网格连接（横向和纵向）
    for row in range(grid_size):
        for col in range(grid_size):
            node_id = row * grid_size + col
            
            # 右邻居
            if col < grid_size - 1:
                right_neighbor = row * grid_size + (col + 1)
                adjacency_matrix[node_id][right_neighbor] = 1
                adjacency_matrix[right_neighbor][node_id] = 1
            
            # 下邻居
            if row < grid_size - 1:
                down_neighbor = (row + 1) * grid_size + col
                adjacency_matrix[node_id][down_neighbor] = 1
                adjacency_matrix[down_neighbor][node_id] = 1
    
    # 设置拓扑并构建最短路径路由
    route_manager.set_topology(adjacency_matrix)
    
    print("构建CrossRing最短路径路由...")
    build_result = route_manager.build_comprehensive_routing_tables("shortest_path", True)
    print(f"构建结果: {build_result}")
    
    # 演示多个节点的路由查询
    print("\n演示路由查询:")
    test_pairs = [(0, 15), (5, 10), (3, 12)]  # 几个测试节点对
    
    for source, dest in test_pairs:
        route_table = route_manager.get_route_table(source)
        if route_table:
            route = route_table.lookup_route_with_context_analysis(dest)
            if route:
                print(f"从节点{source}到节点{dest}: 路径={route.path}, 距离={route.metric}")
            else:
                print(f"从节点{source}到节点{dest}: 无路由")
    
    return route_manager

def demo_route_table_persistence():
    """演示路由表的保存和加载"""
    print("\n" + "="*50)
    print("演示4: 路由表持久化")
    print("="*50)
    
    # 创建路由表并添加路由
    route_table = RouteTable(0, "Test")
    for i in range(1, 5):
        route_table.add_route_with_validation(
            destination=i,
            next_hop=min(i, 2),
            path=list(range(min(i, 2), i+1)) if i > 2 else [i],
            metric=float(i),
            priority=1,
            direction=f"TO_{i}"
        )
    
    # 导出到字典
    print("导出路由表...")
    exported_data = route_table.export_routes()
    print(f"导出的路由数量: {len(exported_data.get('primary_routes', {}))}")
    
    # 保存到文件
    temp_file = "/tmp/route_table_demo.json"
    success = route_table.save_to_file(temp_file)
    print(f"保存到文件 {temp_file}: {'成功' if success else '失败'}")
    
    # 创建新的路由表并从文件加载
    new_route_table = RouteTable(0, "Test")
    load_success = new_route_table.load_from_file(temp_file)
    print(f"从文件加载: {'成功' if load_success else '失败'}")
    
    if load_success:
        print("验证加载的路由:")
        for dest in range(1, 5):
            route = new_route_table.lookup_route_with_context_analysis(dest)
            if route:
                print(f"  到节点{dest}: {route.path}")
    
    # 清理临时文件
    try:
        os.remove(temp_file)
    except:
        pass

def demo_dynamic_routing_with_congestion():
    """演示动态路由和拥塞感知"""
    print("\n" + "="*50)
    print("演示5: 动态路由和拥塞感知")
    print("="*50)
    
    # 创建路由表
    route_table = RouteTable(0, "Ring_6")
    route_table.enable_dynamic_routing = True
    
    # 添加到节点3的多条路由
    destination = 3
    
    # 主路由：直接路径
    route_table.add_route_with_validation(
        destination=destination,
        next_hop=1,
        path=[1, 2, 3],
        metric=3.0,
        priority=1,
        direction="CW"
    )
    
    # 备用路由：反向路径
    route_table.add_route_with_validation(
        destination=destination,
        next_hop=5,
        path=[5, 4, 3],
        metric=3.0,
        priority=2,
        direction="CCW"
    )
    
    print("测试不同拥塞条件下的路由选择:")
    
    # 场景1：无拥塞
    context1 = {"congestion_threshold": 0.8}
    route1 = route_table.lookup_route_with_context_analysis(destination, context1)
    print(f"无拥塞: 选择路径 {route1.path if route1 else 'None'}, 方向 {route1.direction if route1 else 'None'}")
    
    # 场景2：主路由拥塞
    context2 = {
        "congestion_threshold": 0.8,
        "link_congestion_1": 0.9,  # 主路由拥塞
        "load_balancing": True
    }
    route2 = route_table.lookup_route_with_context_analysis(destination, context2)
    print(f"主路由拥塞: 选择路径 {route2.path if route2 else 'None'}, 方向 {route2.direction if route2 else 'None'}")
    
    # 场景3：方向偏好
    context3 = {
        "load_balancing": True,
        "direction_bias_CW": 0.3,
        "direction_bias_CCW": 0.1
    }
    route3 = route_table.lookup_route_with_context_analysis(destination, context3)
    print(f"偏好CCW方向: 选择路径 {route3.path if route3 else 'None'}, 方向 {route3.direction if route3 else 'None'}")
    
    # 显示缓存和统计信息
    stats = route_table.get_statistics()
    print(f"\n路由统计: 查询次数={stats['route_lookups']}, 缓存命中={stats['cache_hits']}")

def demo_integration_with_crossring():
    """演示与CrossRing项目的集成"""
    print("\n" + "="*50)
    print("演示6: 与CrossRing项目集成")
    print("="*50)
    
    try:
        # 创建CrossRing配置
        config = CrossRingConfig()
        print(f"CrossRing配置加载成功: {config.TOPO_TYPE}")
        
        # 创建适配CrossRing的路由管理器
        route_manager = DistributedRouteManager(config.TOPO_TYPE)
        
        # 模拟创建网络节点
        num_nodes = getattr(config, 'NUM_NODE', 16)
        print(f"为{num_nodes}个节点创建路由表...")
        
        for node_id in range(num_nodes):
            route_table = route_manager.add_node(node_id)
            route_table.enable_backup_routes = True
            route_table.enable_dynamic_routing = True
            
            # 添加一些示例路由
            for dest in range(num_nodes):
                if dest != node_id:
                    # 简化的路由：假设通过相邻节点
                    next_hop = (node_id + 1) % num_nodes
                    route_table.add_route_with_validation(
                        destination=dest,
                        next_hop=next_hop,
                        path=[next_hop, dest] if dest != next_hop else [dest],
                        metric=abs(dest - node_id),
                        priority=1,
                        direction="FORWARD"
                    )
        
        # 获取全局统计信息
        global_stats = route_manager.get_global_statistics()
        print(f"全局路由统计: {global_stats}")
        
        # 演示节点间的路由查询
        print("\n节点间路由查询示例:")
        for source in [0, 4, 8]:
            route_table = route_manager.get_route_table(source)
            if route_table:
                dest = (source + num_nodes // 2) % num_nodes
                route = route_table.lookup_route_with_context_analysis(dest)
                if route:
                    print(f"节点{source} -> 节点{dest}: 下一跳={route.next_hop}")
    
    except Exception as e:
        print(f"集成演示出错: {e}")
        print("这可能是因为配置文件不存在或路径问题")

def main():
    """主演示函数"""
    print("CrossRing路由表功能演示")
    print("本演示展示路由表的各种使用场景")
    
    try:
        # 基本操作演示
        route_table = demo_basic_route_table()
        
        # Ring拓扑路由演示
        ring_manager = demo_ring_topology_routing()
        
        # CrossRing拓扑路由演示
        crossring_manager = demo_crossring_topology_routing()
        
        # 持久化演示
        demo_route_table_persistence()
        
        # 动态路由演示
        demo_dynamic_routing_with_congestion()
        
        # 项目集成演示
        demo_integration_with_crossring()
        
        print("\n" + "="*50)
        print("演示完成！")
        print("="*50)
        print("\n使用总结:")
        print("1. 创建RouteTable实例来管理单个节点的路由")
        print("2. 使用DistributedRouteManager管理整个网络的路由")
        print("3. 调用add_route_with_validation()添加路由条目")
        print("4. 调用lookup_route_with_context_analysis()查询最佳路由")
        print("5. 支持动态路由、负载均衡和拥塞感知")
        print("6. 支持路由表的导入导出和持久化")
        print("7. 适配Ring和CrossRing拓扑结构")
        
    except Exception as e:
        logger.error(f"演示过程中出错: {e}")
        raise

if __name__ == "__main__":
    main()