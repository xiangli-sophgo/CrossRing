# 4-Die D2D（Die-to-Die）扩展设计文档

## 1. 概述

### 1.1 扩展背景
现有CrossRing D2D系统支持2个Die之间的通信，本文档描述如何将系统扩展至支持4个Die的2x2网格布局。这种扩展是为了满足更大规模芯片设计的需求，如多核心处理器和高性能计算芯片。

### 1.2 4-Die系统优势
- **更高并行度**: 4个Die可以同时处理不同的计算任务
- **更大存储容量**: 分布式存储架构提供更高的存储带宽
- **模块化设计**: 每个Die可以专注于特定功能（如计算、存储、I/O等）
- **可扩展性**: 为未来支持更多Die奠定基础

### 1.3 与现有2-Die设计的关系
- **兼容性**: 完全向后兼容现有2-Die配置
- **渐进式**: 在2-Die基础上扩展，最小化代码修改
- **模块化**: 复用现有D2D组件，扩展连接管理功能

## 2. 4-Die拓扑架构

### 2.1 2x2网格布局
```
┌─────────────────┬─────────────────┐
│      DIE 2      │      DIE 3      │
│    (左上)       │    (右上)       │
├─────────────────┼─────────────────┤
│      DIE 1      │      DIE 0      │
│    (左下)       │    (右下)       │
└─────────────────┴─────────────────┘
```

### 2.2 Die间连接关系
基于提供的4DIE互联图，各Die的连接关系如下：

| Die ID | 位置 | 连接的Die | 连接方向 |
|--------|------|-----------|----------|
| DIE 0  | 右下 | DIE 1 (左), DIE 2 (上) | 水平+垂直 |
| DIE 1  | 左下 | DIE 0 (右), DIE 3 (上) | 水平+垂直 |
| DIE 2  | 左上 | DIE 0 (下), DIE 3 (右) | 垂直+水平 |
| DIE 3  | 右上 | DIE 1 (下), DIE 2 (左) | 垂直+水平 |

### 2.3 连接类型定义
- **水平连接**: 左右相邻的Die，通过左右边缘的D2D接口
- **垂直连接**: 上下相邻的Die，通过上下边缘的D2D接口
- **对角连接**: 非直连Die，需要通过中间Die进行多跳路由

## 3. D2D接口配置

### 3.1 接口位置设计原则
每个Die需要配置两组D2D接口：
- **主D2D接口**: 处理一个方向的连接（如水平）
- **辅助D2D接口**: 处理另一个方向的连接（如垂直）

### 3.2 各Die的D2D节点位置计算

D2D节点位置由`D2DConfig`类根据拓扑和布局自动计算，不需要手动配置。基于GRID_2X2布局模式，各Die的D2D接口位置计算规则如下：

```python
# D2DConfig类中的位置计算逻辑（需要扩展）
def _calculate_grid_2x2_positions(self):
    """计算2x2网格布局的D2D节点位置"""
    # Die 0 (右下): 需要左边缘（连接Die 1）和上边缘（连接Die 2）的D2D接口
    # Die 1 (左下): 需要右边缘（连接Die 0）和上边缘（连接Die 3）的D2D接口  
    # Die 2 (左上): 需要下边缘（连接Die 0）和右边缘（连接Die 3）的D2D接口
    # Die 3 (右上): 需要下边缘（连接Die 1）和左边缘（连接Die 2）的D2D接口
    
    # 基于8x9拓扑的位置计算示例
    if self.NUM_ROW == 8 and self.NUM_COL == 9:
        # Die 0位置（右下角区域）
        self.D2D_DIE0_POSITIONS = [
            # 左边缘连接（连接Die 1）
            self._get_left_edge_positions(start_row=4, end_row=7),
            # 上边缘连接（连接Die 2）  
            self._get_top_edge_positions(start_col=5, end_col=8)
        ]
        
        # Die 1位置（左下角区域）
        self.D2D_DIE1_POSITIONS = [
            # 右边缘连接（连接Die 0）
            self._get_right_edge_positions(start_row=4, end_row=7),
            # 上边缘连接（连接Die 3）
            self._get_top_edge_positions(start_col=0, end_col=3)
        ]
        
        # 其他Die位置计算...
```

**位置计算原则**：
- **边缘对称**: 相邻Die的连接位置在对应边缘对称分布
- **避免冲突**: 确保D2D节点不与现有IP节点位置冲突
- **负载均衡**: 在可能的情况下均匀分布D2D接口

## 4. 连接映射设计

### 4.1 D2D连接映射数据结构
```python
# 4-Die连接映射表
D2D_CONNECTION_MAP = {
    # Die 0的连接
    0: {
        1: {"type": "horizontal", "direction": "left"},   # DIE 0 → DIE 1
        2: {"type": "vertical", "direction": "up"}        # DIE 0 → DIE 2
    },
    # Die 1的连接
    1: {
        0: {"type": "horizontal", "direction": "right"},  # DIE 1 → DIE 0
        3: {"type": "vertical", "direction": "up"}        # DIE 1 → DIE 3
    },
    # Die 2的连接
    2: {
        0: {"type": "vertical", "direction": "down"},     # DIE 2 → DIE 0
        3: {"type": "horizontal", "direction": "right"}   # DIE 2 → DIE 3
    },
    # Die 3的连接
    3: {
        1: {"type": "vertical", "direction": "down"},     # DIE 3 → DIE 1
        2: {"type": "horizontal", "direction": "left"}    # DIE 3 → DIE 2
    }
}
```

### 4.2 直连Die对（6对）
- DIE 0 ↔ DIE 1（水平连接）
- DIE 0 ↔ DIE 2（垂直连接）
- DIE 1 ↔ DIE 3（垂直连接）
- DIE 2 ↔ DIE 3（水平连接）

### 4.3 非直连Die对（2对，需多跳路由）
- DIE 0 ↔ DIE 3：可通过 DIE 1 或 DIE 2
- DIE 1 ↔ DIE 2：可通过 DIE 0 或 DIE 3

## 5. 配置系统扩展

### 5.1 新增配置参数
```python
# config/d2d_config.py中需要添加的参数
class D2DConfig(CrossRingConfig):
    def __init__(self, die_config_file=None, d2d_config_file=None):
        # 扩展Die数量支持
        self.NUM_DIES = 4  # 支持2-4个Die
        
        # 新增Die2和Die3的D2D位置
        self.D2D_DIE2_POSITIONS = []
        self.D2D_DIE3_POSITIONS = []
        
        # 连接映射配置
        self.D2D_CONNECTION_MAP = {}
        
        # 扩展的布局模式
        self.D2D_LAYOUT = "GRID_2X2"  # 新增网格布局模式
        
        # 多跳路由配置
        self.D2D_MULTI_HOP_ENABLED = True
        self.D2D_ROUTING_ALGORITHM = "shortest_path"  # shortest_path, load_balance
```

### 5.2 配置文件示例
**文件**: `config/topologies/d2d_4die_config.yaml`
```yaml
# D2D (Die-to-Die) 4-Die 配置文件

# D2D 特定配置
D2D_ENABLED: true
NUM_DIES: 4
D2D_LAYOUT: "GRID_2X2"  # 新增2x2网格布局模式

# 4-Die连接映射（新增）
D2D_CONNECTION_MAP:
  0: {1: "horizontal", 2: "vertical"}    # Die 0连接Die 1(水平)和Die 2(垂直)
  1: {0: "horizontal", 3: "vertical"}    # Die 1连接Die 0(水平)和Die 3(垂直)
  2: {0: "vertical", 3: "horizontal"}    # Die 2连接Die 0(垂直)和Die 3(水平)
  3: {1: "vertical", 2: "horizontal"}    # Die 3连接Die 1(垂直)和Die 2(水平)

# 多跳路由配置（新增）
D2D_MULTI_HOP_ENABLED: true
D2D_ROUTING_ALGORITHM: "shortest_path"  # shortest_path, load_balance

# D2D 延迟配置 (cycles) - 保持不变
D2D_AR_LATENCY: 10  # 地址读通道延迟
D2D_R_LATENCY: 10   # 读数据通道延迟
D2D_AW_LATENCY: 10  # 地址写通道延迟
D2D_W_LATENCY: 10   # 写数据通道延迟
D2D_B_LATENCY: 10   # 写响应通道延迟
D2D_DBID_LATENCY: 10 # DBID 信号延迟
D2D_MAX_OUTSTANDING: 16  # 最大未完成事务数

# D2D 带宽控制 - 保持不变
D2D_DATA_BW_LIMIT: 64   # D2D数据通道带宽限制
D2D_RN_BW_LIMIT: 128    # D2D RN 带宽限制 (GB/s)
D2D_SN_BW_LIMIT: 128    # D2D SN 带宽限制 (GB/s)

# D2D 资源管理配置 - 保持不变
# D2D_RN 跟踪器配置
D2D_RN_R_TRACKER_OSTD: 48   # D2D RN 读跟踪器 OSTD
D2D_RN_W_TRACKER_OSTD: 48   # D2D RN 写跟踪器 OSTD
D2D_RN_RDB_SIZE: 192        # D2D RN 读databuffer大小
D2D_RN_WDB_SIZE: 192        # D2D RN 写databuffer大小

# D2D_SN 跟踪器配置
D2D_SN_R_TRACKER_OSTD: 48   # D2D SN 读跟踪器 OSTD
D2D_SN_W_TRACKER_OSTD: 48   # D2D SN 写跟踪器 OSTD
D2D_SN_RDB_SIZE: 192        # D2D SN 读databuffer大小
D2D_SN_WDB_SIZE: 192        # D2D SN 写databuffer大小
```

### 5.3 向后兼容性
- **自动检测**: 根据NUM_DIES参数自动选择2-Die或4-Die模式
- **配置复用**: 2-Die配置在4-Die模式下仍然有效
- **渐进迁移**: 现有2-Die项目可以平滑升级到4-Die

## 6. 核心组件修改

### 6.1 D2D_Sys组件改造
**文件**: `src/utils/components/d2d_sys.py`

**主要修改点**:
```python
class D2D_Sys:
    def __init__(self, node_pos: int, die_id: int, config):
        # 支持多目标Die
        self.target_die_connections = {}  # 替代单一目标Die
        
        # 从配置中获取连接映射
        connection_map = getattr(config, 'D2D_CONNECTION_MAP', {})
        self.my_connections = connection_map.get(str(die_id), {})
        
        # 为每个连接的Die设置目标位置
        for target_die_id, connection_info in self.my_connections.items():
            self.setup_target_die_connection(int(target_die_id), connection_info)
    
    def route_to_target_die(self, flit: Flit):
        """智能路由到目标Die"""
        target_die = flit.d2d_target_die
        
        if target_die in self.my_connections:
            # 直连，直接发送
            return self.send_direct(flit, target_die)
        else:
            # 非直连，多跳路由
            return self.send_multi_hop(flit, target_die)
```

### 6.2 D2D_Model扩展
**文件**: `src/core/d2d_model.py`

**主要修改点**:
```python
class D2D_Model:
    def __init__(self, config: CrossRingConfig, traffic_config: list, **kwargs):
        # 支持动态Die数量
        self.num_dies = getattr(config, "NUM_DIES", 2)
        
        # 为每个Die创建多个D2D接口
        self._setup_multi_interface_dies()
    
    def _setup_multi_interface_dies(self):
        """为每个Die设置多个D2D接口"""
        for die_id in range(self.num_dies):
            die_config = self._create_die_config(die_id)
            die_model = BaseModel(...)
            
            # 为每个Die创建多个D2D_Sys实例
            self._create_multiple_d2d_systems(die_model, die_id)
    
    def _setup_cross_die_connections_4die(self):
        """建立4-Die的跨Die连接"""
        connection_map = getattr(self.config, "D2D_CONNECTION_MAP", {})
        
        for src_die_id, connections in connection_map.items():
            for dst_die_id, connection_type in connections.items():
                self._establish_die_connection(
                    int(src_die_id), int(dst_die_id), connection_type
                )
```

### 6.3 接口组件更新
**文件**: `src/utils/components/d2d_*_interface.py`

**主要修改点**:
- 支持多目标Die的接口管理
- 更新路由决策逻辑
- 处理中间Die的转发

## 7. 多跳路由机制

### 7.1 路由算法设计
```python
class MultiHopRouter:
    def __init__(self, connection_map):
        self.connection_map = connection_map
        self.routing_table = self._build_routing_table()
    
    def _build_routing_table(self):
        """构建路由表，使用最短路径算法"""
        routing_table = {}
        
        # 对于每个源Die，计算到其他Die的最短路径
        for src_die in range(4):
            routing_table[src_die] = {}
            for dst_die in range(4):
                if src_die != dst_die:
                    routing_table[src_die][dst_die] = self._find_shortest_path(src_die, dst_die)
        
        return routing_table
    
    def get_next_hop(self, src_die, dst_die):
        """获取从源Die到目标Die的下一跳"""
        if dst_die in self.connection_map.get(str(src_die), {}):
            # 直连
            return dst_die
        else:
            # 多跳路由
            path = self.routing_table[src_die][dst_die]
            return path[1] if len(path) > 1 else dst_die
```

### 7.2 路由表示例
```python
# 4-Die系统的路由表
ROUTING_TABLE = {
    0: {
        1: [0, 1],        # DIE 0 → DIE 1 (直连)
        2: [0, 2],        # DIE 0 → DIE 2 (直连)
        3: [0, 1, 3]      # DIE 0 → DIE 3 (经过DIE 1)
        # 或: [0, 2, 3]   # DIE 0 → DIE 3 (经过DIE 2)
    },
    1: {
        0: [1, 0],        # DIE 1 → DIE 0 (直连)
        2: [1, 0, 2],     # DIE 1 → DIE 2 (经过DIE 0)
        3: [1, 3]         # DIE 1 → DIE 3 (直连)
    },
    # ... 其他Die的路由表
}
```

### 7.3 中间Die转发逻辑
```python
def handle_forwarding_request(self, flit: Flit):
    """处理转发请求（作为中间Die）"""
    current_die = self.die_id
    target_die = flit.d2d_target_die
    
    # 检查是否为转发请求
    if target_die != current_die and not self.is_final_destination(flit):
        # 确定下一跳
        next_hop = self.router.get_next_hop(current_die, target_die)
        
        # 添加转发标记
        flit.forwarding_path.append(current_die)
        
        # 转发到下一跳
        self.forward_to_die(flit, next_hop)
    else:
        # 最终目的地，正常处理
        self.handle_local_request(flit)
```

### 7.4 死锁避免策略
- **路径固定**: 对于固定的源-目标Die对，始终使用相同的路径
- **优先级管理**: 转发请求具有更高优先级
- **缓冲区预留**: 为转发请求预留专用缓冲区

## 8. 实现计划

### 8.1 第一阶段：配置系统扩展
**目标**: 支持4-Die配置参数
**文件修改**:
- `config/d2d_config.py`: 添加4-Die支持
- `config/`: 创建4-Die配置文件示例

**验证标准**:
- [ ] 成功解析4-Die配置文件
- [ ] 正确生成4个Die的D2D节点位置
- [ ] 连接映射表构建正确

### 8.2 第二阶段：D2D_Sys组件改造
**目标**: 支持多目标Die连接
**文件修改**:
- `src/utils/components/d2d_sys.py`: 多目标Die支持
- `src/utils/components/d2d_*_interface.py`: 接口更新

**验证标准**:
- [ ] 每个Die正确识别其连接的Die
- [ ] 直连路由功能正常
- [ ] AXI通道延迟仿真正确

### 8.3 第三阶段：D2D_Model核心扩展
**目标**: 支持4个Die的协调管理
**文件修改**:
- `src/core/d2d_model.py`: 4-Die支持
- 相关工具文件更新

**验证标准**:
- [ ] 成功创建4个Die实例
- [ ] 跨Die连接建立正确
- [ ] 时钟同步机制正常

### 8.4 第四阶段：多跳路由实现
**目标**: 支持非直连Die的通信
**新增文件**:
- `src/utils/components/multi_hop_router.py`: 路由算法
**文件修改**:
- D2D相关组件添加路由支持

**验证标准**:
- [ ] 最短路径算法正确
- [ ] 中间Die转发功能正常
- [ ] 无死锁现象

### 8.5 第五阶段：测试验证
**目标**: 全面验证4-Die功能
**测试内容**:
- 直连Die通信测试
- 多跳路由通信测试
- 性能对比测试
- 稳定性测试

## 9. Traffic配置示例

### 9.1 4-Die Traffic文件格式
```python
# traffic/4die_example.csv
# cycle, src_die, src_node, dst_die, dst_node, req_type, burst_length
# 直连通信示例
100, 0, 5, 1, 10, read, 4      # DIE 0 → DIE 1 (直连)
200, 1, 6, 3, 12, write, 8     # DIE 1 → DIE 3 (直连)
300, 2, 8, 0, 15, read, 2      # DIE 2 → DIE 0 (直连)

# 多跳通信示例
400, 0, 20, 3, 25, read, 4     # DIE 0 → DIE 3 (通过DIE 1或DIE 2)
500, 1, 30, 2, 35, write, 8    # DIE 1 → DIE 2 (通过DIE 0或DIE 3)

# 并发通信测试
600, 0, 5, 1, 10, read, 4      # 多条并发请求
600, 0, 6, 2, 11, read, 4
600, 1, 7, 3, 12, write, 8
600, 2, 8, 0, 13, write, 8
```

### 9.2 通信模式示例
```python
# 1. 一对一通信（点对点）
generate_point_to_point_traffic(src_die=0, dst_die=3, request_count=100)

# 2. 一对多通信（广播）
generate_broadcast_traffic(src_die=0, dst_dies=[1, 2, 3], request_count=50)

# 3. 多对一通信（聚合）
generate_aggregate_traffic(src_dies=[1, 2, 3], dst_die=0, request_count=30)

# 4. 全连接通信（All-to-All）
generate_all_to_all_traffic(request_count=20)
```

## 10. 预期挑战和解决方案

### 10.1 性能挑战
**挑战**: 4-Die系统的复杂度显著增加，可能影响仿真性能
**解决方案**:
- 优化路由算法，预计算路由表
- 使用高效的数据结构存储连接信息
- 添加性能监控和分析工具

### 10.2 调试复杂度
**挑战**: 多Die、多跳路由增加调试难度
**解决方案**:
- 增强trace机制，记录完整的路径信息
- 添加可视化工具显示Die间通信状态
- 分阶段验证，确保每个阶段功能正确

### 10.3 死锁风险
**挑战**: 环形拓扑可能导致死锁
**解决方案**:
- 设计保守的路由算法，避免形成等待环
- 实现超时机制，检测和恢复死锁
- 添加死锁检测工具

## 11. 总结

4-Die D2D扩展是CrossRing系统的重要演进，它在保持现有2-Die设计兼容性的基础上，提供了更强大的多Die通信能力。通过合理的架构设计、渐进式实现策略和全面的测试验证，这个扩展将为用户提供更灵活、更强大的NoC仿真能力。

关键成功因素：
- **模块化设计**: 最小化对现有代码的影响
- **渐进实施**: 分阶段实现，确保每步都可验证
- **充分测试**: 覆盖各种通信场景和边界条件
- **性能优化**: 确保4-Die系统的仿真效率

这个设计为未来支持更多Die（如8-Die、16-Die）奠定了坚实的基础。