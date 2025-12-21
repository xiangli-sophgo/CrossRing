# CrossRing 请求通道保序功能设计文档

## 1. 概述

CrossRing NoC系统实现了针对请求（REQ）通道的包保序功能，确保在网络传输过程中来自同一源-目的对（source-destination pair）的请求包能够按照注入顺序到达目的地。该保序机制是CrossRing网络协议的核心特性之一，用于维护系统的一致性和正确性。

### 1.1 保序功能特点
- **三种保序模式**：支持不保序(Mode 0)、单侧下环(Mode 1)、双侧下环(Mode 2)三种模式
- **两种保序粒度**：支持IP层级保序和节点层级保序，灵活适配不同场景需求
- **可配置的包类型保序**：通过配置参数指定需要保序的包类型（REQ/RSP/DATA的任意组合）
- **IP层级保序**：每个IP对独立维护保序流，精细控制
- **节点层级保序**：以节点对为保序单元，同一节点内所有IP共享保序流
- **灵活的方向控制**：单侧模式固定TL/TU，双侧模式通过YAML配置指定各方向允许下环的源节点
- **全局顺序**：基于全局顺序ID分配器确保包的顺序
- **灵活配置**：可精确控制哪些包类型、哪些源-目的对需要保序，以及各方向的下环策略

## 2. 核心设计原理

### 2.1 设计思路

保序功能基于以下核心思想：
1. **节点级别保序**：以节点对为保序单元，同一源节点到同一目标节点的所有IP流量共享保序
2. **全局顺序ID分配**：每个源节点-目标节点对的每种包类型维护独立的顺序ID序列
3. **本地跟踪表**：每个节点维护保序跟踪表，记录已下环的最大顺序ID
4. **下环顺序检查**：包只能在其顺序ID等于期望的下一个ID时下环
5. **配置驱动的方向控制**：通过配置文件指定不同源节点允许的下环方向，灵活避免死锁

### 2.2 保序策略

CrossRing支持三种保序模式，通过配置参数`ORDERING_PRESERVATION_MODE`选择：

#### Mode 0: 不保序
- 禁用所有保序功能
- 所有flit可以从任意方向下环
- 性能最优，但不保证顺序

#### Mode 1: 单侧下环
- **固定方向限制**：只允许从TL（左）和TU（上）方向下环
- TR（右）和TD（下）方向强制绕环
- 简单可靠，避免死锁
- 性能中等

**单侧下环原理**：
- 所有flit统一只能从TL/TU下环
- 不依赖节点位置，配置简单
- 保证不会产生环路死锁

#### Mode 2: 双侧下环（方向配置）
- **灵活方向控制**：通过配置文件指定各方向允许下环的源节点
- 每个方向（TL/TR/TU/TD）维护一个允许下环的源节点列表
- flit在创建时根据其源节点确定允许的下环方向
- 性能最优，但需要正确配置以避免死锁

**典型配置示例**（5列×4行拓扑）：
- 左半部节点（列0,1）：允许从TR方向下环
- 右半部节点（列2,3）：允许从TL方向下环
- 上半部节点（行0-1）：允许从TD方向下环
- 下半部节点（行2-3）：允许从TU方向下环

这种配置方式可以灵活适配不同拓扑和性能需求。

### 2.3 保序粒度

CrossRing支持两种保序粒度，通过配置参数`ORDERING_GRANULARITY`选择：

#### Granularity 0: IP层级保序
- **独立保序流**：每个源IP到目标IP维护独立的顺序ID序列
- **精细控制**：不同IP对之间互不干扰
- **典型场景**：需要严格区分不同IP流量的保序需求

**示例**（节点5有2个GDMA，节点8有2个DDR）：
- `(node5, gdma_0) → (node8, ddr_0)` 独立保序流
- `(node5, gdma_0) → (node8, ddr_1)` 独立保序流
- `(node5, gdma_1) → (node8, ddr_0)` 独立保序流
- `(node5, gdma_1) → (node8, ddr_1)` 独立保序流

**order_id分配器key格式**：`(src_node_id, src_ip_type, dest_node_id, dest_ip_type)`
- 例如：`(5, "gdma_0", 8, "ddr_1")`

#### Granularity 1: 节点层级保序
- **共享保序流**：同一源节点到同一目标节点的所有IP共享一个顺序ID序列
- **简化管理**：减少保序流数量，降低tracking开销
- **典型场景**：节点间流量需要整体保序

**示例**（节点5所有IP到节点8所有IP）：
- 节点5的所有IP（gdma_0, gdma_1, sdma_0等）到节点8的所有IP共享一个保序流
- 无论源IP和目标IP是哪个，只要源节点=5，目标节点=8，就使用同一个order_id序列

**order_id分配器key格式**：`(src_node_id, dest_node_id)`
- 例如：`(5, 8)`

## 3. 关键数据结构

### 3.1 全局顺序ID分配器

```python
# 位置：src/utils/components/flit.py
_global_order_id_allocator = {}  # {key: {"REQ": next_id, "RSP": next_id, "DATA": next_id}}
```

**功能**：为每个保序流维护全局唯一的顺序ID序列，根据保序粒度使用不同的key结构。

**数据结构**：
- Key结构取决于 `ORDERING_GRANULARITY` 配置：
  - **IP层级（granularity=0）**：`(src_node_id, src_ip_type, dest_node_id, dest_ip_type)`
    - 例如：`(5, "gdma_0", 8, "ddr_1")`
    - 每个IP对独立维护顺序ID
  - **节点层级（granularity=1）**：`(src_node_id, dest_node_id)`
    - 例如：`(5, 8)`
    - 同一节点对的所有IP共享顺序ID
- Value: 包含三种包类型的字典，每种类型维护下一个待分配的ID
  - `{"REQ": 1, "RSP": 1, "DATA": 1}`

**Key构造说明**：
- 使用flit的 `source/destination` 属性（直接为节点ID，无需映射）
- 使用flit的 `source_type/destination_type` 属性（IP层级时使用）
- IP层级key更精细，节点层级key更简化

### 3.2 保序跟踪表

```python
# 位置：src/utils/components/network.py
# 每个network（REQ/RSP/DATA）独立维护自己的tracking_table
order_tracking_table = defaultdict(int)
```

**功能**：每个网络（REQ/RSP/DATA）独立维护的跟踪表，记录每个源-目的节点对在各下环方向上已下环的最大顺序ID。

**数据结构**：
- Key: `(src_node, dest_node, direction)` - 源节点、目标节点和下环方向的三元组
- Value: 该节点对在该方向上已下环的最大顺序ID
- **重要说明**：
  - **包含direction的原因**：CrossRing有水平环和垂直环，同一个flit可能需要先从水平环下环（如TL），再从垂直环下环（如TU）。如果key只用`(src, dest)`，第二次下环时会因为order_id已被记录而检查失败。
  - **network独立**：每个network（REQ/RSP/DATA）维护独立的tracking_table，不需要在value中区分packet_category。
  - **节点级别保序**：使用物理拓扑节点ID，同一源节点到同一目标节点的所有IP流量共享保序跟踪。

### 3.3 Flit保序属性

每个需要保序的Flit包含以下属性：
- `src_dest_order_id`：该flit的顺序ID
- `packet_category`：包类型（"REQ", "RSP", "DATA"）
- `source_original`：原始源IP位置（用于D2D场景）
- `destination_original`：原始目标IP位置（用于D2D场景）
- `allowed_eject_directions`：允许下环的方向列表（如["TR", "TU"]），在flit创建时根据源节点和配置确定

**节点ID转换说明**：
- flit的source/destination属性存储的是网络IP位置（经过node_map映射）
- 保序检查时需要将IP位置反向映射为物理拓扑节点ID
- 转换方式：根据source/destination mapping的逆向推导获得节点ID

## 4. 保序实现细节

### 4.1 初始化阶段

**网络初始化时的配置转换**：
1. **读取配置参数**：从YAML配置文件读取四个方向的源节点白名单
2. **节点ID到IP位置映射**：将配置中的物理节点ID转换为网络IP位置（source mapping）
3. **构建查询表**：为每个方向构建允许下环的源IP位置集合，用于快速查询

**转换原理**：
- 配置文件使用物理拓扑节点ID（0-19），便于理解和配置
- 网络运行时使用IP位置（经过node_map映射），用于实际路由
- 初始化时完成一次性转换，运行时直接查询

### 4.2 Flit创建阶段

**确定允许的下环方向**：
1. **获取源节点**：从flit的source属性获取源IP位置
2. **查询白名单**：遍历四个方向，检查源IP位置是否在该方向的白名单中
3. **设置属性**：将允许的方向列表赋值给flit.allowed_eject_directions

### 4.3 顺序ID分配流程

**分配时机**：在CrossPoint上环处理时分配order_id

1. **上环时刻**：在 `inject_queues_pre → inject_queues` 转移时（`_CP_process`函数）
2. **首次分配检查**：只在flit首次上环时分配（`flit.src_dest_order_id == -1`）
3. **获取源和目标信息**：
   - 节点ID：使用 `flit.source/destination` (已是节点ID)
   - IP类型：使用 `flit.source_type/destination_type`
   - 使用original属性处理D2D场景
4. **根据粒度构造key**：
   - IP层级：`(src_node, src_type, dest_node, dest_type)`
   - 节点层级：`(src_node, dest_node)`
5. **分配order_id**：调用 `Flit.get_next_order_id()` 获取唯一ID

**为何在上环时分配**：
- 确保只有真正上环的flit才分配order_id
- 避免在channel buffer阶段分配造成顺序混乱
- 与保序检查时机（下环时）形成对称

### 4.4 保序检查机制

#### 4.4.1 保序需求判断

检查是否需要进行保序：
1. 检查保序功能是否开启（ENABLE_IN_ORDER_EJECTION）
2. 检查包类型是否在需要保序的类型列表中（IN_ORDER_PACKET_CATEGORIES）
3. 检查源-目的对是否在配置的保序列表中（IN_ORDER_EJECTION_PAIRS）

#### 4.4.2 下环方向检查

检查是否允许从当前方向下环：
1. 获取当前尝试的下环方向（TL/TR/TU/TD）
2. 检查该方向是否在flit.allowed_eject_directions列表中
3. 根据结果决策：
   - 不需要保序：允许从任意方向下环
   - 需要保序但方向不允许：强制继续绕环
   - 需要保序且方向允许：进入顺序ID检查

#### 4.4.3 下环顺序检查

检查顺序ID是否匹配：
1. 从flit获取源和目标IP位置
2. 通过逆向映射将IP位置转换为物理拓扑节点ID
3. 使用节点ID和下环方向作为键`(src, dest, direction)`查询order_tracking_table
   - direction区分水平环和垂直环的下环
   - 同一flit在不同环上的下环使用不同的tracking entry
4. 检查flit.src_dest_order_id是否等于期望的下一个ID（当前记录值+1）
5. 匹配则允许下环，否则继续绕环

#### 4.4.4 跟踪表更新

成功下环后更新跟踪表：
1. 从flit获取源和目标IP位置并转换为节点ID
2. 使用节点ID和下环方向作为键`(src, dest, direction)`更新order_tracking_table
   - 记录该节点对在该下环方向上的最大顺序ID
   - 水平环和垂直环的下环分别跟踪
3. 后续同一节点对在同一方向的flit将期望更大的顺序ID

## 5. 基于配置的方向控制机制

### 5.1 方向控制总体设计

CrossRing的保序机制采用**配置驱动的方向控制策略**，通过配置文件灵活指定每个方向允许下环的源节点列表。这种设计允许系统根据不同的拓扑结构和性能需求，灵活配置保序下环行为，同时避免死锁。

**核心思想**：
- 每个方向（TL/TR/TU/TD）维护一个允许下环的源节点白名单
- Flit在创建时根据其源节点查询白名单，确定允许的下环方向
- Flit在尝试下环时，只有在允许的方向才进行保序下环检查
- 不在白名单中的源节点的flit，在该方向强制绕环

### 5.2 配置机制

**初始化阶段**：
1. 从配置文件读取四个方向的源节点白名单（物理拓扑节点ID）
2. 将物理节点ID转换为网络IP位置（source mapping）
3. 构建快速查询表供运行时使用

**运行时检查**：
1. Flit创建时，根据源节点查询四个方向的白名单
2. 将允许的方向列表存储在flit的`allowed_eject_directions`属性中
3. 下环时，检查当前方向是否在允许列表中
4. 若允许，进行保序顺序检查；若不允许，强制绕环

### 5.3 典型配置示例（5×4拓扑）

以5×4拓扑为例（5行4列），典型的配置策略如下：

**水平方向配置**：
- **左半部节点（列0,1）**：允许从TR方向下环
  - 原因：包向右传输，在右侧下环不会造成环路
- **右半部节点（列2,3）**：允许从TL方向下环
  - 原因：包向左传输，在左侧下环不会造成环路

**垂直方向配置**：
- **上半部节点（行0-2）**：允许从TD方向下环
  - 原因：包向下传输，在下方下环不会造成环路
- **下半部节点（行3-4）**：允许从TU方向下环
  - 原因：包向上传输，在上方下环不会造成环路

**配置效果**：
- 避免包在传输反方向下环造成的死锁
- 允许灵活适配不同拓扑结构
- 可根据性能分析调整配置

### 5.4 方向控制工作流程

**Flit下环尝试流程**：
1. **判断是否需要保序**：检查保序功能开关、包类型、源-目的对配置
2. **检查方向权限**：判断当前方向是否在flit的`allowed_eject_directions`中
3. **执行相应操作**：
   - 不需要保序：直接下环（不检查顺序ID）
   - 需要保序但方向不允许：强制绕环，可能进行ETag降级
   - 需要保序且方向允许：进行顺序ID检查
     - 顺序正确：下环成功，更新order_tracking_table
     - 顺序不正确：继续绕环，等待前序包下环

**关键特性**：
- 保序与方向控制分离：先检查方向权限，再检查顺序ID
- 灵活性：通过配置文件调整，无需修改代码
- 性能优化：不允许的方向直接绕环，减少不必要的顺序检查

## 6. 配置参数

### 6.1 主要配置参数

**位置**：配置文件（`config/topologies/*.yaml`）和 `config/config.py`

#### 6.1.1 ORDERING_PRESERVATION_MODE
- **类型**：int
- **默认值**：0
- **功能**：保序模式选择
- **可选值**：
  - 0: 不保序
  - 1: 单侧下环（固定TL/TU）
  - 2: 双侧下环（方向配置）

#### 6.1.2 ORDERING_GRANULARITY
- **类型**：int
- **默认值**：1
- **功能**：保序粒度选择
- **可选值**：
  - 0: IP层级保序（每个IP对独立保序）
  - 1: 节点层级保序（同节点内所有IP共享保序）

#### 6.1.3 IN_ORDER_PACKET_CATEGORIES
- **类型**：list
- **默认值**：`["REQ"]`
- **功能**：指定需要保序的包类型列表
- **可选值**：`["REQ", "RSP", "DATA"]` 的任意组合

#### 6.1.4 IN_ORDER_EJECTION_PAIRS
- **类型**：list of pairs
- **默认值**：`[]`（空列表表示全部源-目的对都保序）
- **功能**：指定需要保序的特定源-目的对
- **格式**：`[[src1, dest1], [src2, dest2], ...]`

#### 6.1.5 方向控制参数
- **参数名称**：`TL_ALLOWED_SOURCE_NODES`, `TR_ALLOWED_SOURCE_NODES`, `TU_ALLOWED_SOURCE_NODES`, `TD_ALLOWED_SOURCE_NODES`
- **类型**：list
- **默认值**：`[]`（空列表表示所有源节点都允许）
- **功能**：分别指定允许从TL（左）、TR（右）、TU（上）、TD（下）四个方向下环的源节点列表
- **格式**：物理拓扑节点ID列表（运行时自动转换为IP位置）

### 6.2 配置示例

#### 6.2.1 Mode 0: 不保序配置示例

```yaml
# config/topologies/kcin_5x4.yaml

# 禁用保序功能
ORDERING_PRESERVATION_MODE: 0  # 不保序
ORDERING_GRANULARITY: 1  # 0=IP层级, 1=节点层级 (Mode 0下不生效)

# 包类型配置（Mode 0下不生效）
IN_ORDER_PACKET_CATEGORIES:
  - "REQ"

# 源-目的对配置（Mode 0下不生效）
IN_ORDER_EJECTION_PAIRS: []
```

#### 6.2.2 Mode 1: 单侧下环配置示例

```yaml
# config/topologies/kcin_5x4.yaml

# 启用单侧下环
ORDERING_PRESERVATION_MODE: 1  # 单侧下环(TL/TU)
ORDERING_GRANULARITY: 1  # 0=IP层级, 1=节点层级

# 需要保序的包类型
IN_ORDER_PACKET_CATEGORIES:
  - "REQ"    # 只有REQ类型需要保序

# 需要保序的源-目的对 (空列表表示全部保序)
IN_ORDER_EJECTION_PAIRS: []

# 方向配置在Mode 1下不生效，固定为TL/TU
```

#### 6.2.3 Mode 2: 双侧下环配置示例（5×4拓扑）

```yaml
# config/topologies/kcin_5x4.yaml

# 启用双侧下环
ORDERING_PRESERVATION_MODE: 2  # 双侧下环(方向配置)
ORDERING_GRANULARITY: 1  # 0=IP层级, 1=节点层级

# 需要保序的包类型
IN_ORDER_PACKET_CATEGORIES:
  - "REQ"    # 只有REQ类型需要保序

# 需要保序的源-目的对 (空列表表示全部保序)
IN_ORDER_EJECTION_PAIRS: []

# 方向控制配置 - 5×4拓扑示例（节点ID: 0-19）
# TL方向（左方向）：右半部节点允许下环
TL_ALLOWED_SOURCE_NODES:
  - 2    # 列2
  - 3    # 列3
  - 6
  - 7
  - 10
  - 11
  - 14
  - 15
  - 18
  - 19

# TR方向（右方向）：左半部节点允许下环
TR_ALLOWED_SOURCE_NODES:
  - 0    # 列0
  - 1    # 列1
  - 4
  - 5
  - 8
  - 9
  - 12
  - 13
  - 16
  - 17

# TU方向（上方向）：下半部节点允许下环
TU_ALLOWED_SOURCE_NODES:
  - 8    # 行2
  - 9
  - 10
  - 11
  - 12   # 行3
  - 13
  - 14
  - 15
  - 16   # 行4
  - 17
  - 18
  - 19

# TD方向（下方向）：上半部节点允许下环
TD_ALLOWED_SOURCE_NODES:
  - 0    # 行0
  - 1
  - 2
  - 3
  - 4    # 行1
  - 5
  - 6
  - 7
```

#### 6.2.2 配置说明

**节点ID分布（5×4拓扑）**：
```
行0:  0   1   2   3
行1:  4   5   6   7
行2:  8   9  10  11
行3: 12  13  14  15
行4: 16  17  18  19
```

**配置策略解释**：
- **水平方向**：左半部（列0,1）从右侧下环，右半部（列2,3）从左侧下环
- **垂直方向**：上半部（行0-2）从下方下环，下半部（行3-4）从上方下环
- **目的**：避免包在传输反方向下环，防止死锁

## 7. 保序处理流程图

### 7.1 整体流程

```
开始
  ↓
检查ORDERING_PRESERVATION_MODE
  ├── Mode 0 → 直接下环，不保序
  ├── Mode 1 → 检查方向是否为TL或TU → 否 → 强制绕环
  │             ↓
  │            是 → 进入保序检查
  └── Mode 2 → 检查当前方向是否在allowed_eject_directions中
                ↓
               是 → 进入保序检查
                ↓
               否 → 强制绕环

保序检查流程：
  判断包类型是否在IN_ORDER_PACKET_CATEGORIES中
  ↓
  检查源-目的对是否需要保序
  ↓
  执行顺序ID检查 → 顺序正确？ → 是 → 下环成功，更新跟踪表
                   ↓
                  否 → 绕环等待
```

### 7.2 保序检查详细流程

```
保序检查开始
  ↓
获取flit的顺序ID和包类型
  ↓
从跟踪表获取期望的下一个顺序ID
  ↓
current_order_id == expected_order_id ?
  ├── 是 → 允许下环
  └── 否 → 拒绝下环，继续绕环
```

### 7.3 跟踪表更新流程

```
成功下环后
  ↓
更新本节点的order_tracking_table
  ↓
table[(src, dest)][packet_category] = flit.src_dest_order_id
  ↓
完成更新
```

## 8. 总结

CrossRing的保序功能通过全局顺序ID分配、本地跟踪表维护和灵活的模式选择策略，实现了高效可靠的包保序机制。系统支持三种保序模式，从不保序到单侧下环再到双侧下环，可根据不同场景需求选择最合适的方案。该设计在保证正确性的同时，通过灵活的配置参数，最小化了对系统性能的影响，并提供了良好的可扩展性。

关键特性总结：
- ✅ **三种保序模式**：不保序、单侧下环(TL/TU)、双侧下环(方向配置)
- ✅ **两种保序粒度**：IP层级（精细控制）、节点层级（简化管理）
- ✅ **可配置包类型**：支持REQ/RSP/DATA的任意组合保序
- ✅ **灵活方向控制**：Mode 1固定方向，Mode 2灵活配置
- ✅ **IP层级保序**：每个IP对独立维护保序流，精细控制
- ✅ **节点层级保序**：同一节点所有IP共享保序流，简化管理
- ✅ **全局一致性**：基于全局顺序ID分配器确保顺序
- ✅ **本地化检查**：每个节点独立维护跟踪表，无需全局同步
- ✅ **灵活扩展**：支持精确控制特定源-目的对的保序需求
- ✅ **上环时分配**：order_id在CP上环时分配，确保顺序正确

### 8.1 模式选择建议

- **Mode 0 (不保序)**：适用于对顺序无要求的场景，性能最优
- **Mode 1 (单侧下环)**：适用于需要简单可靠保序的场景，配置简单，避免死锁
- **Mode 2 (双侧下环)**：适用于对性能要求较高的场景，需要根据拓扑正确配置方向