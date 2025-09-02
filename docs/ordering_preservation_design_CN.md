# CrossRing 请求通道保序功能设计文档

## 1. 概述

CrossRing NoC系统实现了针对请求（REQ）通道的包保序功能，确保在网络传输过程中来自同一源-目的对（source-destination pair）的请求包能够按照注入顺序到达目的地。该保序机制是CrossRing网络协议的核心特性之一，用于维护系统的一致性和正确性。

### 1.1 保序功能特点
- **选择性保序**：仅对REQ通道进行保序，RSP（响应）和DATA（数据）通道不进行保序
- **方向限制**：保序请求只能从TL（左）和TU（上）方向下环，TR（右）和TD（下）方向禁止保序请求下环
- **全局顺序**：基于全局顺序ID分配器确保包的顺序
- **配置灵活**：通过配置参数控制哪些包类型需要保序

## 2. 核心设计原理

### 2.1 设计思路

保序功能基于以下核心思想：
1. **全局顺序ID分配**：每个源-目的对的每种包类型维护独立的顺序ID序列
2. **本地跟踪表**：每个节点维护保序跟踪表，记录已下环的最大顺序ID
3. **下环顺序检查**：包只能在其顺序ID等于期望的下一个ID时下环
4. **方向限制策略**：通过限制下环方向避免死锁和乱序

### 2.2 保序策略

```
保序策略矩阵：
┌──────────┬────────┬────────┬─────────┐
│   方向   │  TL    │  TR    │   TU    │ TD     │
├──────────┼────────┼────────┼─────────┤────────┤
│ REQ保序  │  允许  │  禁止  │  允许   │ 禁止   │
│ RSP/DATA │  正常  │  正常  │  正常   │ 正常   │
└──────────┴────────┴────────┴─────────┘────────┘
```

## 3. 关键数据结构

### 3.1 全局顺序ID分配器

```python
# 位置：src/utils/components/node.py
global_order_id_allocator = defaultdict(lambda: {"REQ": 1, "RSP": 1, "DATA": 1})
```

**功能**：为每个(源, 目的, 包类型)三元组维护全局唯一的顺序ID序列。

**数据结构**：
- Key: `(src, dest)` - 源节点和目的节点的元组
- Value: 包含三种包类型的字典，每种类型维护下一个待分配的ID

### 3.2 保序跟踪表

```python
# 位置：src/utils/components/node.py
order_tracking_table = defaultdict(lambda: {"REQ": 0, "RSP": 0, "DATA": 0})
```

**功能**：每个节点本地维护的跟踪表，记录每个源-目的对的各类型包已下环的最大顺序ID。

**数据结构**：
- Key: `(src, dest)` - 源节点和目的节点的元组  
- Value: 包含三种包类型的字典，记录已下环的最大顺序ID

### 3.3 Flit保序属性

每个需要保序的Flit包含以下属性：
- `src_dest_order_id`：该flit的顺序ID
- `packet_category`：包类型（"REQ", "RSP", "DATA"）
- `source_original`：原始源节点（用于D2D场景）
- `destination_original`：原始目的节点（用于D2D场景）

### 3.4 双ID系统设计

CrossRing系统中存在两套不同的ID标识系统，各有其特定的用途：

#### 3.4.1 packet_id - 全局包标识符

```python
# 位置：src/utils/components/node.py
class Node:
    global_packet_id = -1  # 全局共享计数器

    @classmethod
    def get_next_packet_id(cls):
        cls.global_packet_id += 1  # 简单自增
        return cls.global_packet_id
```

**特点**：
- **全局递增**：所有包（不管src-dest）共享一个全局计数器
- **用途**：包的唯一标识、调试追踪、统计分析
- **分配方式**：按包创建的时间顺序分配，不考虑src-dest关系
- **连续性**：同一src-dest对的packet_id **不连续**

#### 3.4.2 src_dest_order_id - 保序控制ID

```python
# 位置：src/utils/components/node.py
global_order_id_allocator = defaultdict(lambda: {"REQ": 1, "RSP": 1, "DATA": 1})

@classmethod
def get_next_order_id(cls, src, dest, packet_category):
    # 针对特定的(src, dest, packet_category)组合分配ID
    current_id = cls.global_order_id_allocator[(src, dest)][packet_category]
    cls.global_order_id_allocator[(src, dest)][packet_category] += 1
    return current_id
```

**特点**：
- **按源-目的对独立计数**：每个(src, dest, packet_category)组合有独立的计数序列
- **用途**：专门用于保序检查和控制
- **连续性**：同一src-dest对的同类型包，order_id **严格连续递增**
- **保序检查**：下环时使用此ID进行顺序验证

#### 3.4.3 双ID系统对比示例

假设有两个源节点S1、S2向目的节点D发送请求：

```
时间顺序    事件                     packet_id    src_dest_order_id
--------    --------------------     ---------    -----------------
T1          S1→D 创建REQ1           0            1 (S1→D的第1个REQ)
T2          S2→D 创建REQ1           1            1 (S2→D的第1个REQ)
T3          S1→D 创建REQ2           2            2 (S1→D的第2个REQ)
T4          S1→D 创建REQ3           3            3 (S1→D的第3个REQ)
T5          S2→D 创建REQ2           4            2 (S2→D的第2个REQ)
```

**观察结果**：
- `packet_id`：全局连续（0,1,2,3,4），但对于同一src-dest不连续
- `src_dest_order_id`：每个src-dest对独立连续
  - S1→D: 1,2,3（连续递增）
  - S2→D: 1,2（连续递增）

### 3.5 全局与局部的设计架构

#### 3.5.1 全局组件（类变量）

```python
class Node:
    # 全局包ID计数器
    global_packet_id = -1
    
    # 全局顺序ID分配器
    global_order_id_allocator = defaultdict(lambda: {"REQ": 1, "RSP": 1, "DATA": 1})
```

**特点**：
- **作用域**：所有Node实例共享（类级别变量）
- **访问方式**：通过@classmethod访问
- **用途**：确保ID分配的全局一致性和唯一性
- **必要性**：避免多节点间的ID冲突

#### 3.5.2 局部组件（实例变量）

```python
def initialize_data_structures(self):
    # 每个节点独立的保序跟踪表
    self.order_tracking_table = defaultdict(lambda: {"REQ": 0, "RSP": 0, "DATA": 0})
```

**特点**：
- **作用域**：每个Node实例独立维护
- **访问方式**：通过self访问
- **用途**：记录本节点已接收的最大顺序ID
- **优势**：分布式跟踪，无需全局同步

#### 3.5.3 设计优势

1. **分离关注点**：
   - 全局ID分配保证一致性
   - 本地跟踪表支持分布式检查

2. **性能优化**：
   - 避免全局同步开销
   - 本地检查决策快速

3. **扩展性**：
   - 新增节点无需修改全局状态
   - 支持动态拓扑变化

### 3.6 order_tracking_table 工作原理详解

#### 3.6.1 期望ID计算逻辑

```python
# network.py:1151
expected_order_id = self.node.order_tracking_table[(src, dest)][flit.packet_category] + 1
return flit.src_dest_order_id == expected_order_id
```

**核心思想**：
- `order_tracking_table` 记录已成功下环的**最大顺序ID**
- 下一个允许下环的ID = 当前记录值 + 1
- 只有ID连续的包才能下环，确保严格保序

#### 3.6.2 处理乱序包的详细示例

假设3个包乱序到达目的节点：

```
初始状态：order_tracking_table[(S,D)]["REQ"] = 0

时刻T1：包3(ID=3)尝试下环
        期望ID = 0 + 1 = 1
        3 ≠ 1 → 拒绝下环，继续绕环

时刻T2：包2(ID=2)尝试下环
        期望ID = 0 + 1 = 1  
        2 ≠ 1 → 拒绝下环，继续绕环
        包3仍在绕环

时刻T3：包1(ID=1)尝试下环
        期望ID = 0 + 1 = 1
        1 == 1 → 允许下环！
        更新：order_tracking_table[(S,D)]["REQ"] = 1

时刻T4：包2再次尝试下环
        期望ID = 1 + 1 = 2
        2 == 2 → 允许下环！
        更新：order_tracking_table[(S,D)]["REQ"] = 2

时刻T5：包3再次尝试下环
        期望ID = 2 + 1 = 3
        3 == 3 → 允许下环！
        更新：order_tracking_table[(S,D)]["REQ"] = 3
```

**关键要点**：
- 乱序到达的包必须等待前序包下环
- 跟踪表确保严格的连续性
- 系统自动处理乱序，无需额外排序逻辑

## 4. 保序实现细节

### 4.1 顺序ID分配流程

1. **flit注入时**：在flit注入到网络时分配顺序ID
2. **ID分配规则**：
   ```python
   @classmethod
   def get_next_order_id(cls, src, dest, packet_category):
       current_id = cls.global_order_id_allocator[(src, dest)][packet_category]
       cls.global_order_id_allocator[(src, dest)][packet_category] += 1
       return current_id
   ```
3. **包类型判断**：
   ```python
   def _get_flit_packet_category(self, flit: Flit):
       if flit.req_type is not None:
           return "REQ"
       elif flit.rsp_type is not None:
           return "RSP"
       elif hasattr(flit, "flit_type") and flit.flit_type == "data":
           return "DATA"
       else:
           return "REQ"  # 默认为REQ
   ```

### 4.2 保序检查机制

#### 4.2.1 保序需求判断

```python
def _need_in_order_check(self, flit: Flit):
    # 1. 检查保序功能是否开启
    if not self.config.ENABLE_IN_ORDER_EJECTION:
        return False
    
    # 2. 检查包类型是否需要保序
    packet_category = self._get_flit_packet_category(flit)
    if hasattr(self.config, "IN_ORDER_PACKET_CATEGORIES"):
        if packet_category not in self.config.IN_ORDER_PACKET_CATEGORIES:
            return False
    else:
        # 默认只有REQ类型需要保序（向后兼容）
        if flit.req_type is None:
            return False
    
    # 3. 检查源-目的对是否在配置的保序列表中
    src = flit.source_original if flit.source_original != -1 else flit.source
    dest = flit.destination_original if flit.destination_original != -1 else flit.destination
    
    if not hasattr(self.config, "IN_ORDER_EJECTION_PAIRS") or len(self.config.IN_ORDER_EJECTION_PAIRS) == 0:
        return True  # 空列表表示全部保序
        
    return [src, dest] in self.config.IN_ORDER_EJECTION_PAIRS
```

#### 4.2.2 下环顺序检查

```python
def _can_eject_in_order(self, flit: Flit, target_eject_node):
    # 1. 先判断是否需要保序
    if not self._need_in_order_check(flit):
        return True
    
    # 2. 获取保序信息
    src = flit.source_original if flit.source_original != -1 else flit.source
    dest = flit.destination_original if flit.destination_original != -1 else flit.destination
    
    # 3. 检查是否是期望的下一个顺序ID
    expected_order_id = self.node.order_tracking_table[(src, dest)][flit.packet_category] + 1
    
    return flit.src_dest_order_id == expected_order_id
```

#### 4.2.3 跟踪表更新

```python
def _update_order_tracking_table(self, flit: Flit):
    # 成功下环后更新跟踪表
    if not self._need_in_order_check(flit):
        return
        
    src = flit.source_original if flit.source_original != -1 else flit.source
    dest = flit.destination_original if flit.destination_original != -1 else flit.destination
    
    # 更新保序跟踪表
    if hasattr(self, "node") and self.node is not None:
        self.node.order_tracking_table[(src, dest)][flit.packet_category] = flit.src_dest_order_id
```

## 5. 各方向保序策略详解

### 5.1 TL方向（左方向）- 允许保序下环

**位置**：`src/utils/components/network.py:630-724`

**实现特点**：
- **完整保序支持**：支持T0, T1, T2所有优先级的保序检查
- **ETag升级机制**：T1可以升级到T0，T2可以升级到T1
- **下环条件**：
  ```python
  # 非T0情况
  if (len(link_station) < self.config.RB_IN_FIFO_DEPTH 
      and entry_available 
      and self._can_eject_in_order(flit, target_eject_node_id)):
      # 执行下环操作
      self._update_order_tracking_table(flit)
  
  # T0情况需要额外检查T0队列顺序
  if (self.T0_Etag_Order_FIFO[0] == (next_node, flit) 
      and can_use_T0 
      and self._can_eject_in_order(flit, target_eject_node_id)):
      # 执行T0下环操作
  ```

### 5.2 TR方向（右方向）- 禁止保序下环

**位置**：`src/utils/components/network.py:575-623`

**实现特点**：
- **保序REQ禁止**：需要保序的REQ包不能从TR方向下环
- **禁止逻辑**：
  ```python
  # TR方向尝试下环 - 需要保序的REQ禁止下环
  if self._need_in_order_check(flit):
      # 需要保序的REQ不能从TR下环，继续绕环
      link[flit.current_seat_index] = None
      next_pos = next_node + 1
      flit.current_link = (next_node, next_pos)
      flit.current_seat_index = 0
      # ETag可能降级
      if self.ETag_BOTHSIDE_UPGRADE and flit.ETag_priority == "T2":
          flit.ETag_priority = "T1"
  ```
- **非保序包正常**：RSP/DATA包或不需要保序的REQ包正常下环

### 5.3 TU方向（上方向）- 允许保序下环

**位置**：`src/utils/components/network.py:789-874`

**实现特点**：
- **完整保序支持**：与TL方向类似，支持完整的保序机制
- **ETag升级**：支持T2→T1→T0的升级路径
- **保序检查集成**：
  ```python
  # T1, T2情况
  if (len(link_eject) < self.config.EQ_IN_FIFO_DEPTH 
      and entry_available 
      and self._can_eject_in_order(flit, next_node)):
      # 执行下环并更新跟踪表
      self._update_order_tracking_table(flit)
  
  # T0情况
  if (len(link_eject) < self.config.EQ_IN_FIFO_DEPTH 
      and self._can_eject_in_order(flit, next_node)):
      # T0下环处理
  ```

### 5.4 TD方向（下方向）- 禁止保序下环

**位置**：`src/utils/components/network.py:732-774`

**实现特点**：
- **保序REQ禁止**：与TR方向类似，禁止保序REQ下环
- **强制绕环**：
  ```python
  # TD方向尝试下环 - 需要保序的REQ禁止下环
  if self._need_in_order_check(flit):
      # 需要保序的REQ不能从TD下环，继续绕环
      if self.ETag_BOTHSIDE_UPGRADE and flit.ETag_priority == "T2":
          flit.ETag_priority = "T1"
      link[flit.current_seat_index] = None
      next_pos = next_node + self.config.NUM_COL * 2
      flit.current_link = (next_node, next_pos)
      flit.current_seat_index = 0
  ```
- **无升级到T0**：TD方向的flit不能升级到T0优先级

## 6. 配置参数

### 6.1 主要配置参数

**位置**：配置文件（`config/topologies/*.yaml`）和 `config/config.py`

#### 6.1.1 ENABLE_IN_ORDER_EJECTION
- **类型**：boolean
- **默认值**：true/false（根据具体配置）
- **功能**：全局开关，控制是否启用保序功能

#### 6.1.2 IN_ORDER_PACKET_CATEGORIES
- **类型**：list
- **默认值**：`["REQ"]`
- **功能**：指定需要保序的包类型列表
- **可选值**：`["REQ", "RSP", "DATA"]` 的任意组合

#### 6.1.3 IN_ORDER_EJECTION_PAIRS
- **类型**：list of pairs
- **默认值**：`[]`（空列表表示全部源-目的对都保序）
- **功能**：指定需要保序的特定源-目的对
- **格式**：`[[src1, dest1], [src2, dest2], ...]`

### 6.2 配置示例

```yaml
# config/topologies/topo_12x12.yaml
ENABLE_IN_ORDER_EJECTION: 1

# 需要保序的包类型
IN_ORDER_PACKET_CATEGORIES:
  - "REQ"    # 只有REQ类型需要保序

# 需要保序的源-目的对 (空列表表示全部保序)
IN_ORDER_EJECTION_PAIRS: []
```

## 7. 保序处理流程图

### 7.1 整体流程

```
开始
  ↓
检查ENABLE_IN_ORDER_EJECTION
  ↓
判断包类型是否在IN_ORDER_PACKET_CATEGORIES中
  ↓
检查源-目的对是否需要保序
  ↓
判断当前下环方向
  ├── TL/TU → 执行保序检查 → 顺序正确？ → 是 → 下环成功，更新跟踪表
  │                        ↓
  │                       否 → 绕环等待
  └── TR/TD → 需要保序？ → 是 → 强制绕环
                        ↓
                       否 → 正常下环
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

## 8. 关键实现要点

### 8.1 避免死锁的设计

1. **方向限制**：通过限制保序REQ只能从TL/TU下环，避免了环路死锁
2. **ETag降级**：在TR/TD方向，无法下环的flit可以进行ETag降级，增加后续下环机会
3. **强制绕环**：保序REQ在TR/TD方向被强制绕环，确保最终到达TL/TU下环点

### 8.2 性能优化考虑

1. **选择性保序**：只对REQ包进行保序，减少性能开销
2. **本地跟踪表**：每个节点维护本地表，避免全局同步开销
3. **优先级机制**：保序与ETag优先级机制结合，保持原有性能特性

### 8.3 可扩展性设计

1. **配置驱动**：通过配置参数灵活控制保序行为
2. **包类型扩展**：支持对不同包类型的独立保序控制
3. **源-目的对选择**：支持对特定通信对的精确控制

## 9. 总结

CrossRing的请求通道保序功能通过全局顺序ID分配、本地跟踪表维护、以及方向限制策略的结合，实现了高效可靠的包保序机制。该设计在保证正确性的同时，通过选择性保序和配置驱动的方式，最小化了对系统性能的影响，并提供了良好的可扩展性和灵活性。

关键特性总结：
- ✅ **选择性保序**：仅对REQ通道保序
- ✅ **方向限制**：TL/TU允许，TR/TD禁止
- ✅ **全局一致**：基于全局顺序ID确保顺序
- ✅ **避免死锁**：通过设计避免环路死锁
- ✅ **配置灵活**：支持多种配置选项
- ✅ **性能优化**：最小化性能影响