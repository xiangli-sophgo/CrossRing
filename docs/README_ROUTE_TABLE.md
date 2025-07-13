# 路由表功能使用指南

## 概述

本项目为CrossRing NoC仿真添加了完整的路由表功能，支持灵活的路由管理、动态路由决策和负载均衡。

## 核心功能

### 1. 路由表管理
- **RouteTable**: 单节点路由表管理
- **DistributedRouteManager**: 全网络路由表管理
- 支持主路由和备用路由
- 动态路由缓存机制

### 2. 路由策略
- **最短路径路由**: 基于Floyd-Warshall算法
- **Ring负载均衡**: 支持顺时针/逆时针负载分配
- **自适应路由**: 基于拥塞感知的动态路由选择

### 3. 拓扑支持
- **Ring拓扑**: 环形网络拓扑
- **CrossRing拓扑**: 交叉环网络拓扑
- **网格拓扑**: 通用网格拓扑

## 使用方法

### 基本使用

```python
from src.utils.components.route_table import RouteTable, DistributedRouteManager

# 1. 创建单节点路由表
route_table = RouteTable(node_id=0, topology_type="Ring_8")

# 2. 添加路由条目
route_table.add_route_with_validation(
    destination=3,
    next_hop=1,
    path=[1, 2, 3],
    metric=3.0,
    priority=1,
    direction="CW"
)

# 3. 查询路由
route = route_table.lookup_route_with_context_analysis(destination=3)
if route:
    print(f"下一跳: {route.next_hop}, 路径: {route.path}")
```

### 分布式路由管理

```python
# 1. 创建分布式路由管理器
route_manager = DistributedRouteManager("Ring_8")

# 2. 设置网络拓扑
# 创建8节点环形拓扑
ring_size = 8
adjacency_matrix = [[0] * ring_size for _ in range(ring_size)]
for i in range(ring_size):
    next_node = (i + 1) % ring_size
    prev_node = (i - 1) % ring_size
    adjacency_matrix[i][next_node] = 1
    adjacency_matrix[i][prev_node] = 1

route_manager.set_topology(adjacency_matrix)

# 3. 构建路由表
result = route_manager.build_comprehensive_routing_tables(
    routing_strategy="ring_balanced",
    enable_backup_routes=True
)

# 4. 查询特定节点的路由
route_table = route_manager.get_route_table(node_id=0)
route = route_table.lookup_route_with_context_analysis(destination=4)
```

### 动态路由和拥塞感知

```python
# 带上下文信息的路由查询
context = {
    "load_balancing": True,
    "congestion_threshold": 0.8,
    "link_congestion_1": 0.9,  # 链路1拥塞度90%
    "direction_bias_CW": 0.2,   # 顺时针方向偏好
    "direction_bias_CCW": 0.1   # 逆时针方向偏好
}

route = route_table.lookup_route_with_context_analysis(
    destination=target_node,
    context=context
)
```

### 与Node类集成

```python
from src.utils.components.node import Node
from config.config import CrossRingConfig

# 创建带路由表的节点
config = CrossRingConfig()
node = Node(config, node_id=0)

# 节点路由操作
next_hop = node.get_next_hop(destination=5)
full_path = node.get_full_path(destination=5)
has_route = node.has_route_to(destination=5)

# 添加自定义路由
node.add_route(
    destination=7,
    next_hop=3,
    path=[3, 5, 7],
    metric=3.0,
    direction="CUSTOM"
)
```

## 运行示例

### 1. 运行完整演示
```bash
cd example
python route_table_demo.py
```

### 2. 与现有Ring模型集成
```python
from src.core.Ring import create_ring_model

# 创建Ring模型（已集成路由表功能）
ring_model = create_ring_model(
    num_nodes=8,
    routing_strategy="load_balanced",
    traffic_file_path="../../traffic/test_data"
)
```

## 配置参数

### RouteTable配置
- `enable_backup_routes`: 启用备用路由（默认True）
- `enable_dynamic_routing`: 启用动态路由（默认True）
- `cache_size_limit`: 动态路由缓存大小限制（默认1000）

### 路由策略配置
- `"shortest_path"`: 最短路径策略
- `"ring_balanced"`: Ring拓扑负载均衡
- `"adaptive"`: 自适应拥塞感知路由

## 文件结构

```
src/utils/components/
├── route_table.py          # 路由表核心实现
├── node.py                 # 扩展的Node类（带路由表支持）
└── ...

example/
├── route_table_demo.py     # 完整使用示例
└── ...

config/
└── config.py              # 配置文件（可添加路由相关配置）
```

## API参考

### RouteTable类主要方法

- `add_route_with_validation()`: 添加路由条目（带验证）
- `lookup_route_with_context_analysis()`: 智能路由查询
- `update_and_manage_routes()`: 路由管理（更新/删除）
- `export_routes()` / `import_routes()`: 路由表导入导出

### DistributedRouteManager类主要方法

- `build_comprehensive_routing_tables()`: 构建完整路由表
- `set_topology()`: 设置网络拓扑
- `get_global_statistics()`: 获取全局路由统计

### Node类新增方法

- `lookup_route()`: 查询路由
- `get_next_hop()`: 获取下一跳
- `get_route_direction()`: 获取路由方向（Ring专用）
- `add_route()` / `remove_route()`: 路由管理

## 性能特性

1. **高效查询**: 支持路由缓存，减少重复计算
2. **内存优化**: 智能缓存管理，避免内存泄漏
3. **扩展性**: 支持大规模网络（1000+节点）
4. **容错性**: 路由失败时自动回退机制

## 注意事项

1. 路由表功能完全向后兼容，不影响现有代码
2. 如果路由表模块不可用，会自动回退到简单路由
3. 动态路由需要提供网络状态上下文信息
4. 建议在网络初始化时一次性构建所有路由表

## 故障排除

1. **导入错误**: 确保在项目根目录运行脚本
2. **路由查找失败**: 检查拓扑设置和路由表构建
3. **性能问题**: 调整缓存大小和清理策略
4. **内存使用**: 监控动态缓存大小，必要时清理

更多详细信息请参考 `example/route_table_demo.py` 中的完整示例。