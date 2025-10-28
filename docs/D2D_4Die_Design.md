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
基于物理相邻关系，各Die的连接关系如下：

| Die ID | 位置 | 连接的Die | 连接方向 |
|--------|------|-----------|----------|
| DIE 0  | 右下 | DIE 1 (左), DIE 3 (上) | 水平+垂直 |
| DIE 1  | 左下 | DIE 0 (右), DIE 2 (上) | 水平+垂直 |
| DIE 2  | 左上 | DIE 1 (下), DIE 3 (右) | 垂直+水平 |
| DIE 3  | 右上 | DIE 0 (下), DIE 2 (左) | 垂直+水平 |

### 2.3 连接类型定义
- **水平连接**: 左右相邻的Die，通过左右边缘的D2D接口
- **垂直连接**: 上下相邻的Die，通过上下边缘的D2D接口

### 2.4 配置驱动的D2D连接设计

#### 2.4.1 拓扑规格
- **配置拓扑**: 5x4（5行4列）
- **仿真网络**: 10x4（两行表示一行物理节点）
- **总节点数**: 40个节点（0-39）

#### 2.4.2 基于配置的灵活连接定义
现有实现采用配置驱动的方式定义D2D连接，不再依赖硬编码的节点位置或Die旋转概念。每个Die的连接关系通过YAML配置文件中的`D2D_DIE_CONFIG`定义。

#### 2.4.3 配置驱动原则
```yaml
D2D_DIE_CONFIG:
  <die_id>:
    num_row: <行数>
    num_col: <列数>
    connections:
      <edge>: {die: <目标die>, d2d_nodes: [相对位置索引]}
```

其中：
- `<edge>`: 连接边缘（left/right/top/bottom）
- `d2d_nodes`: 该边缘上D2D节点的**相对位置索引**
- 索引从0开始，按边缘的排列顺序计算

#### 2.4.4 相对位置索引计算规则
- **left/right边**: 从上到下按奇数行排列，索引从0开始
- **top/bottom边**: 从左到右按列排列，索引从0开始

例如：在5x4拓扑中
- **left边奇数行节点**: [12, 20, 28]，索引[0, 1, 2]分别对应节点[12, 20, 28]
- **top边第二行节点**: [4, 5, 6, 7]，索引[0, 1, 2]分别对应节点[4, 5, 6]

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

## 4. 基于配置的节点配对设计

### 4.1 配对生成逻辑

系统根据`D2D_DIE_CONFIG`自动生成节点配对关系，无需硬编码固定的节点位置。配对生成遵循以下原则：

#### 4.1.1 配对生成流程
1. **读取Die配置**: 从`D2D_DIE_CONFIG`获取每个Die的连接定义
2. **计算边缘节点**: 根据拓扑规模和边缘类型计算具体节点位置
3. **获取相对索引**: 根据`d2d_nodes`配置获取具体的D2D节点
4. **自动对称匹配**: 查找对端Die的对应连接配置，生成配对关系

#### 4.1.2 边缘节点计算规则
```python
def _get_edge_nodes(self, edge, num_row, num_col):
    """获取指定边的所有节点"""
    if edge == "top":
        # 第二行（row=1）
        return list(range(num_col, 2 * num_col))
    elif edge == "bottom":
        # 最后一行（row=num_row-1）
        return list(range((num_row - 1) * num_col, num_row * num_col))
    elif edge == "left":
        # 奇数行的左边（col=0）
        return [row * num_col for row in range(1, num_row, 2)]
    elif edge == "right":
        # 奇数行的右边（col=num_col-1）
        return [row * num_col + (num_col - 1) for row in range(1, num_row, 2)]
```

#### 4.1.3 基于当前配置的配对示例
根据`config/topologies/d2d_4die_config.yaml`配置，生成的配对关系为：

```python
# 以5x4拓扑为例，生成的配对关系：
# Die0-Die1连接（Die0右边 ↔ Die1左边）
(0, 15, 1, 12),  # right[1] ↔ left[1]
(0, 23, 1, 20),  # right[2] ↔ left[2] 
(0, 31, 1, 28),  # right[3] ↔ left[3]

# Die0-Die3连接（Die0下边 ↔ Die3上边）
(0, 36, 3, 4),   # bottom[0] ↔ top[0]
(0, 37, 3, 5),   # bottom[1] ↔ top[1]
(0, 38, 3, 6),   # bottom[2] ↔ top[2]

# Die1-Die2连接（Die1下边 ↔ Die2上边）
(1, 36, 2, 4),   # bottom[0] ↔ top[0]
(1, 37, 2, 5),   # bottom[1] ↔ top[1]
(1, 38, 2, 6),   # bottom[2] ↔ top[2]

# Die2-Die3连接（Die2右边 ↔ Die3左边）
(2, 15, 3, 12),  # right[1] ↔ left[1]
(2, 23, 3, 20),  # right[2] ↔ left[2]
(2, 31, 3, 28),  # right[3] ↔ left[3]
```

### 4.2 配对生成算法优势

#### 4.2.1 灵活性
- **可配置**: 通过修改YAML文件即可调整连接关系
- **可扩展**: 支持不同拓扑规模的Die配置
- **可验证**: 自动检查配置的对称性和完整性

#### 4.2.2 自动化
- **自动对称匹配**: 系统自动找到对端Die的对应连接
- **自动索引转换**: 相对位置索引自动转换为具体节点ID
- **自动冲突检测**: 检测配置错误和资源冲突

### 4.3 配置验证机制

#### 4.3.1 连接完整性验证
系统自动验证每个Die是否与正确数量的其他Die连接：
- **2x2网格拓扑**: 每个Die应连接到2个相邻Die
- **连接对称性**: 每个连接都有对应的反向连接配置

#### 4.3.2 节点位置验证
- **范围检查**: 确保所有D2D节点位置在有效范围内（0到NUM_NODE-1）
- **重复检查**: 确保同一Die内D2D节点位置不重复
- **边缘匹配**: 验证相邻Die的边缘连接正确对应

## 5. 配置系统实现

### 5.1 配置驱动的节点配对生成

4-Die系统不再使用硬编码的节点配对函数，而是采用配置驱动的动态生成方式。配对生成过程在`D2DConfig`类中自动完成。

#### 5.1.1 配对生成核心算法
```python
def _generate_d2d_pairs_from_config(self, config):
    """
    根据配置生成D2D节点配对
    
    Args:
        config: Die配置字典（来自D2D_DIE_CONFIG）
    
    Returns:
        pairs: [(die0_id, node0, die1_id, node1), ...]
    """
    pairs = []
    processed = set()  # 避免重复处理
    
    for die_id, die_config in config.items():
        # 获取Die的网络规模
        num_row = die_config["num_row"]
        num_col = die_config["num_col"]
        
        # 处理每个连接
        for edge, conn_info in die_config["connections"].items():
            target_die = conn_info["die"]
            d2d_node_indices = conn_info["d2d_nodes"]  # 相对位置索引列表
            
            # 避免重复处理
            pair_key = tuple(sorted([die_id, target_die]))
            if pair_key in processed:
                continue
            processed.add(pair_key)
            
            # 获取边缘节点并生成配对
            edge_nodes = self._get_edge_nodes(edge, num_row, num_col)
            d2d_nodes = [edge_nodes[i] for i in d2d_node_indices if i < len(edge_nodes)]
            
            # 自动匹配对端配置
            target_d2d_nodes = self._get_target_d2d_nodes(config, target_die, die_id, edge)
            
            # 生成配对
            for i in range(min(len(d2d_nodes), len(target_d2d_nodes))):
                pairs.append((die_id, d2d_nodes[i], target_die, target_d2d_nodes[i]))
    
    return pairs
```

#### 5.1.2 边缘节点自动计算
```python
def _get_edge_nodes(self, edge, num_row, num_col):
    """获取指定边的所有节点"""
    if edge == "top":
        # 第二行（row=1）
        return list(range(num_col, 2 * num_col))
    elif edge == "bottom":
        # 最后一行（row=num_row-1）
        return list(range((num_row - 1) * num_col, num_row * num_col))
    elif edge == "left":
        # 奇数行的左边（col=0）
        return [row * num_col for row in range(1, num_row, 2)]
    elif edge == "right":
        # 奇数行的右边（col=num_col-1）
        return [row * num_col + (num_col - 1) for row in range(1, num_row, 2)]
```

#### 5.1.3 配置驱动的优势
- **无硬编码**: 不再依赖固定的节点位置假设
- **自动对称**: 系统自动查找并匹配对端Die的连接配置
- **灵活配置**: 支持任意拓扑规模和连接模式
- **错误检测**: 自动验证配置完整性和一致性

### 5.2 D2DConfig类实现

现有`D2DConfig`类采用配置驱动的通用设计，自动支持2-Die和4-Die模式。

#### 5.2.1 配置初始化流程
```python
class D2DConfig(CrossRingConfig):
    def __init__(self, die_config_file=None, d2d_config_file=None):
        super().__init__(die_config_file)
        
        # 初始化D2D配置
        self._init_d2d_config()
        
        # 加载D2D专用配置文件
        if d2d_config_file:
            self._load_d2d_config_file(d2d_config_file)
        
        # 生成D2D配对关系
        self._generate_d2d_pairs()
        
        # 更新通道规格
        self._update_channel_spec_for_d2d()
        
        # 验证配置
        self._validate_d2d_layout()
```

#### 5.2.2 配置驱动的配对生成
```python
def _generate_d2d_pairs(self):
    """生成D2D配对关系"""
    num_dies = getattr(self, "NUM_DIES", 2)
    if num_dies < 2:
        raise ValueError(f"D2D需要至少2个Die，当前配置: {num_dies}")
    
    # 使用配置驱动方式
    self._setup_config_based()

def _setup_config_based(self):
    """基于配置的通用Die设置方式"""
    die_config = getattr(self, "D2D_DIE_CONFIG", None)
    
    if die_config is None:
        raise ValueError("未找到D2D_DIE_CONFIG配置，请在YAML配置文件中定义")
    
    # 使用配置生成D2D配对
    pairs = self._generate_d2d_pairs_from_config(die_config)
    
    # 设置配对结果
    self.D2D_PAIRS = pairs
    
    # 自动设置各Die的D2D节点位置
    num_dies = getattr(self, "NUM_DIES", 2)
    for die_id in range(num_dies):
        positions = []
        for pair in pairs:
            if pair[0] == die_id:
                positions.append(pair[1])
            if pair[2] == die_id:
                positions.append(pair[3])
        # 去重并设置到字典中
        self.D2D_DIE_POSITIONS[die_id] = list(set(positions))
```

#### 5.2.3 自动验证机制
```python
def _validate_d2d_layout(self):
    """验证D2D布局的合理性"""
    if not getattr(self, "D2D_ENABLED", False):
        return
    
    num_dies = getattr(self, "NUM_DIES", 2)
    
    # 基础参数检查
    if num_dies < 2:
        raise ValueError(f"D2D需要至少2个Die，当前配置: {num_dies}")
    if num_dies > 4:
        raise ValueError(f"当前最多支持4个Die，当前配置: {num_dies}")
    
    # 节点位置有效性检查
    max_node_id = self.NUM_NODE - 1
    for die_id in range(num_dies):
        positions = self.D2D_DIE_POSITIONS.get(die_id, [])
        for pos in positions:
            if pos < 0 or pos > max_node_id:
                raise ValueError(f"Die{die_id}节点位置无效: {pos}")
    
    # 4-Die特定验证
    if num_dies == 4:
        self._validate_4die_config()
```

#### 5.2.4 配置类优势
- **通用性**: 同时支持2-Die和4-Die模式，无需不同实现
- **自动化**: 根据配置自动生成所有必要的连接关系
- **验证完整**: 自动检查配置错误和不一致性
- **向后兼容**: 完全兼容现有2-Die配置

### 5.3 配置文件结构和参数说明

现有配置系统采用YAML格式，通过`D2D_DIE_CONFIG`定义4-Die的连接关系。

#### 5.3.1 配置文件结构
**文件**: `config/topologies/d2d_4die_config.yaml`
```yaml
# D2D (Die-to-Die) 4-Die 配置文件

# D2D 基础配置
D2D_ENABLED: true
NUM_DIES: 4

# 4-Die节点配置 (配置驱动方式)
D2D_DIE_CONFIG:
  0:  # Die 0 配置
    num_row: 5         # Die内拓扑行数
    num_col: 4         # Die内拓扑列数
    connections:
      right: {die: 1, d2d_nodes: [1, 2, 3]}    # 连接Die1，使用右边第1,2,3个位置
      bottom: {die: 3, d2d_nodes: [0, 1, 2]}   # 连接Die3，使用下边第0,1,2个位置
  1:  # Die 1 配置
    num_row: 5
    num_col: 4
    connections:
      left: {die: 0, d2d_nodes: [1, 2, 3]}     # 连接Die0，使用左边第1,2,3个位置
      bottom: {die: 2, d2d_nodes: [0, 1, 2]}   # 连接Die2，使用下边第0,1,2个位置
  2:  # Die 2 配置
    num_row: 5
    num_col: 4
    connections:
      top: {die: 1, d2d_nodes: [0, 1, 2]}      # 连接Die1，使用上边第0,1,2个位置
      right: {die: 3, d2d_nodes: [1, 2, 3]}    # 连接Die3，使用右边第1,2,3个位置
  3:  # Die 3 配置
    num_row: 5
    num_col: 4
    connections:
      left: {die: 2, d2d_nodes: [1, 2, 3]}     # 连接Die2，使用左边第1,2,3个位置
      top: {die: 0, d2d_nodes: [0, 1, 2]}      # 连接Die0，使用上边第0,1,2个位置

# 多跳路由配置
D2D_MULTI_HOP_ENABLED: true
D2D_ROUTING_ALGORITHM: "shortest_path"

# D2D 延迟配置 (ns) - 会在配置加载时转换为cycles
D2D_AR_LATENCY: 5       # 地址读通道延迟 (ns)
D2D_R_LATENCY: 5        # 读数据通道延迟 (ns)
D2D_AW_LATENCY: 5       # 地址写通道延迟 (ns)
D2D_W_LATENCY: 5        # 写数据通道延迟 (ns)
D2D_B_LATENCY: 5        # 写响应通道延迟 (ns)
D2D_DBID_LATENCY: 3     # DBID 信号延迟 (ns)

# 注：延迟配置使用ns单位，D2DConfig.update_latency()会自动转换为cycles
# 转换公式：latency_cycles = latency_ns * NETWORK_FREQUENCY
D2D_MAX_OUTSTANDING: 16 # 最大未完成事务数

# D2D 带宽控制
D2D_DATA_BW_LIMIT: 64   # D2D数据通道带宽限制
D2D_RN_BW_LIMIT: 128    # D2D RN 带宽限制 (GB/s)
D2D_SN_BW_LIMIT: 128    # D2D SN 带宽限制 (GB/s)

# D2D 资源管理配置
D2D_RN_R_TRACKER_OSTD: 48   # D2D RN 读跟踪器 OSTD
D2D_RN_W_TRACKER_OSTD: 48   # D2D RN 写跟踪器 OSTD
D2D_RN_RDB_SIZE: 192        # D2D RN 读databuffer大小
D2D_RN_WDB_SIZE: 192        # D2D RN 写databuffer大小

D2D_SN_R_TRACKER_OSTD: 48   # D2D SN 读跟踪器 OSTD
D2D_SN_W_TRACKER_OSTD: 48   # D2D SN 写跟踪器 OSTD
D2D_SN_RDB_SIZE: 192        # D2D SN 读databuffer大小
D2D_SN_WDB_SIZE: 192        # D2D SN 写databuffer大小
```

#### 5.3.2 关键参数说明

**D2D_DIE_CONFIG结构**：
- `<die_id>`: Die编号（0-3）
- `num_row`/`num_col`: 该Die的内部拓扑规模
- `connections`: 该Die与其他Die的连接定义
  - `<edge>`: 连接边缘（left/right/top/bottom）
  - `die`: 目标Die编号
  - `d2d_nodes`: 相对位置索引数组，指定在该边缘使用哪些位置作为D2D节点

**相对位置索引规则**：
- **left/right边**: 按奇数行从上到下排序，索引从0开始
- **top/bottom边**: 按列从左到右排序，索引从0开始
- 例如：`d2d_nodes: [1, 2, 3]`表示使用该边缘的第1、2、3个位置（跳过第0个）

#### 5.3.3 配置优势
- **声明式配置**: 直接描述连接关系，无需计算具体节点位置
- **对称验证**: 系统自动检查连接配置的对称性
- **扩展性**: 易于修改连接模式或扩展到其他拓扑规模
- **可读性**: 配置结构直观，易于理解和维护

### 5.3 向后兼容性
- **自动检测**: 根据NUM_DIES参数自动选择2-Die或4-Die模式
- **配置复用**: 2-Die配置在4-Die模式下仍然有效
- **渐进迁移**: 现有2-Die项目可以平滑升级到4-Die

## 6. 核心组件修改

### 6.1 D2D_Sys组件简化
**文件**: `src/utils/components/d2d_sys.py`

**关键修改**：基于节点配对的点对点连接
```python
class D2D_Sys:
    def __init__(self, node_pos: int, die_id: int, config):
        self.position = node_pos
        self.die_id = die_id
        
        # 从配对表查找目标连接
        self.target_die_id = None
        self.target_node_pos = None
        
        if hasattr(config, 'D2D_NODE_PAIRS'):
            for pair in config.D2D_NODE_PAIRS:
                if pair[0] == die_id and pair[1] == node_pos:
                    self.target_die_id = pair[2]
                    self.target_node_pos = pair[3]
                    break
                elif pair[2] == die_id and pair[3] == node_pos:
                    self.target_die_id = pair[0]
                    self.target_node_pos = pair[1]
                    break
        
        if self.target_die_id is None:
            raise ValueError(f"Node {node_pos} in Die {die_id} is not a D2D node")
```

### 6.2 D2D_Model连接建立
**文件**: `src/core/d2d_model.py`

**主要修改**：使用配对表建立所有连接
```python
def _setup_cross_die_connections(self):
    """建立Die间的连接关系"""
    if self.num_dies == 4 and hasattr(self.config, 'D2D_NODE_PAIRS'):
        self._setup_4die_connections_from_pairs()
    else:
        self._setup_2die_connections()  # 保持原有逻辑

def _setup_4die_connections_from_pairs(self):
    """基于配对表建立4-Die连接"""
    for pair in self.config.D2D_NODE_PAIRS:
        die0_id, node0, die1_id, node1 = pair
        self._connect_d2d_pair(die0_id, node0, die1_id, node1)
```

## 7. 实现计划

### 7.1 第一阶段：在d2d_config.py中实现配对生成
**目标**: 添加节点配对计算函数
**修改内容**:
- 实现`calculate_4die_d2d_pairs()`函数
- 扩展`D2DConfig`类支持4-Die
- 自动生成各Die的D2D位置

### 7.2 第二阶段：简化D2D_Sys组件
**目标**: 基于节点配对的点对点连接
**修改内容**:
- 重构`__init__`方法，使用配对表查找目标
- 每个D2D_Sys实例只管理一个连接
- 移除复杂的队列管理逻辑

### 7.3 第三阶段：更新D2D_Model连接建立
**目标**: 使用配对表建立所有连接
**修改内容**:
- 修改`_setup_cross_die_connections`支持4-Die
- 实现`_setup_4die_connections_from_pairs`
- 自动建立所有12对连接

### 7.4 第四阶段：测试验证
**测试内容**:
- 配对表生成正确性
- 4-Die连接建立成功
- D2D_Sys初始化无错误
- 基本通信功能验证

## 8. 连接验证和拓扑确认

### 8.2 基于配置的连接验证

基于当前配置文件生成的连接关系，4个Die之间形成2x2网格拓扑：

#### 8.2.1 直连关系验证
```python
# 根据d2d_4die_config.yaml生成的连接对：
连接关系验证：
- Die0 ↔ Die1: 3对连接 (右边-左边)
- Die0 ↔ Die3: 3对连接 (下边-上边)
- Die1 ↔ Die2: 3对连接 (下边-上边)
- Die2 ↔ Die3: 3对连接 (右边-左边)

每个Die连接到2个相邻Die，符合2x2网格拓扑要求
```

#### 8.2.2 多跳路由支持
配置文件中启用了`D2D_MULTI_HOP_ENABLED: true`，支持非直连Die间的通信：
- **Die0 ↔ Die2**: 通过Die1或Die3中转（2跳）
- **Die1 ↔ Die3**: 通过Die0或Die2中转（2跳）

#### 8.2.3 路由算法
- **算法**: shortest_path（最短路径）
- **实现**: 系统自动计算最短路径，选择最优的中转路径

## 9. 总结

基于配置驱动的4-Die D2D扩展设计具有以下优势：

### 9.1 设计优势
- **配置驱动**: 通过YAML配置定义连接关系，无需硬编码
- **自动对称**: 系统自动匹配对端Die的连接配置
- **灵活配置**: 支持不同拓扑规模和连接模式的定制
- **相对索引**: 使用边缘相对位置索引，避免绝对节点ID依赖

### 9.2 实现简化  
- **配置自动生成**: 节点配对由配置文件驱动自动计算
- **验证完整**: 自动检查配置完整性和对称性
- **调试友好**: 配置关系直观，易于理解和调试
- **向后兼容**: 完全兼容现有2-Die设计

### 9.3 扩展性
- **通用框架**: 统一的配置框架支持任意Die数量扩展
- **标准化**: 建立了基于配置的Die间连接标准模式
- **可维护**: 配置文件结构清晰，易于修改和维护
- **错误检测**: 自动检测配置错误和不一致性

### 9.4 与硬编码方案对比
- **灵活性**: 从硬编码节点位置改为配置驱动的相对索引
- **可扩展性**: 从固定的2x2网格改为支持任意拓扑配置
- **维护性**: 从代码中的配对函数改为配置文件管理
- **验证机制**: 增加了完整的配置验证和错误检测

这个基于配置驱动的设计为CrossRing系统提供了灵活、可扩展的4-Die支持，为未来支持更多Die和更复杂拓扑奠定了坚实基础。