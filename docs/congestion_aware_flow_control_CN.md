# 动态拥塞感知流控设计文档

本文档描述CrossRing NoC系统中的动态拥塞感知流控机制设计。

---

## 设计目标

- **作用范围**: 网络层（CrossPoint上环决策点）
- **监控粒度**: 方向级别（TL/TR/TU/TD独立监控）
- **流控方式**: 阈值式限流 - 感知下游拥塞后延迟上环
- **核心思想**: 在flit进入环之前，根据下游拥塞情况决定是否允许上环
- **硬件约束**: 设计需考虑实际硬件实现的可行性和成本

---

## 整体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                     CrossPoint._can_inject_to_link()                  │
│                                                                       │
│   IQ_OUT[TL/TR] ─┬─> [ITag检查] ─> [拥塞感知检查] ─> 横向Link        │
│                  │                      ↓                             │
│                  │              本节点Entry可用性                      │
│                  │                      ↓                             │
│                  │              [限流决策]                             │
│                  │              - ITag flit: 必须上环                  │
│                  │              - 普通flit: 阈值式限流                 │
│                  │                                                    │
│   RB_OUT[TU/TD] ─┴─> [ITag检查] ─> [拥塞感知检查] ─> 纵向Link        │
└──────────────────────────────────────────────────────────────────────┘
```

### 设计原理

与传统IP层流控（Token Bucket）不同，本方案在CrossPoint的上环决策点进行流控：

| 方面 | IP层流控 | CrossPoint层流控（本方案） |
|------|---------|---------------------------|
| 作用点 | IP的Token Bucket | CrossPoint的上环决策 |
| 方向感知 | 需要预测/统计 | 天然按方向区分（IQ_OUT/RB_OUT已分方向） |
| 流控粒度 | IP整体速率 | 单个flit是否上环 |
| 对ITag的处理 | 无 | ITag flit必须上环（保证公平性） |

---

## 拥塞指标设计

### 1. Link层面拥塞度（路径感知）- 仅用于仿真分析

计算flit要经过的所有Link的slot占用率，取最大值作为路径拥塞度：

```python
def _calculate_path_link_congestion(self, flit, direction) -> float:
    """计算flit路径上所有Link的拥塞度"""
    path = flit.path
    current_idx = path.index(flit.source) if flit.source in path else 0
    remaining_path = path[current_idx:]

    if len(remaining_path) <= 1:
        return 0.0

    congestion_values = []
    for i in range(len(remaining_path) - 1):
        link = (remaining_path[i], remaining_path[i + 1])
        if link in self.network.links:
            slots = self.network.links[link]
            occupied = sum(1 for s in slots if s is not None)
            congestion_values.append(occupied / len(slots))

    return max(congestion_values) if congestion_values else 0.0
```

**设计考量**：
- 使用最大值而非平均值，因为路径上任一Link拥塞都会造成阻塞
- 只计算剩余路径，已经过的Link不影响决策

#### 硬件实现分析 - 为什么不实现

| 问题 | 严重程度 | 详细说明 |
|------|---------|---------|
| **全局信息访问** | ⭐⭐⭐⭐⭐ | 需要访问路径上所有Link的状态，这些Link分布在多个不同节点上。在实际硬件中，每个节点只能直接访问本地状态，获取远端节点状态需要额外的通信机制。 |
| **动态路径长度** | ⭐⭐⭐⭐ | 不同flit的路径长度不同（1跳到N跳），硬件电路需要固定延迟，难以处理变长计算。 |
| **布线开销** | ⭐⭐⭐⭐⭐ | 若要实现全局Link状态汇聚，需要大量跨节点信号线。对于NxM的mesh，每个节点需要接收O(N+M)条Link的状态，布线复杂度极高。 |
| **时序收敛** | ⭐⭐⭐⭐ | 长距离信号传输导致关键路径过长。假设每跳传输需要1个周期，获取5跳外的Link状态就需要5个周期延迟，这对于上环决策来说太慢。 |
| **面积开销** | ⭐⭐⭐⭐ | 每个节点需要存储全局Link状态的副本，或者实现复杂的查询协议，面积开销巨大。 |

**结论**：Link路径拥塞度在**实际硬件中几乎不可实现**。在NoC设计中，节点通常只能感知本地或相邻1跳节点的状态。本仿真模型保留此功能用于**性能分析和对比研究**，但**不作为实际流控决策的依据**。

---

### 2. Entry层面拥塞度（本节点感知）- 硬件实现方案

计算本节点对应方向的Entry使用情况：

```python
def _calculate_local_entry_congestion(self, node_pos, direction) -> float:
    """计算本节点对应方向的Entry拥塞度"""
    if direction in ["TL", "TR"]:
        counters = self.RB_UE_Counters[direction][node_pos]
        capacity = self.RB_CAPACITY[direction][node_pos]
    else:  # TU, TD
        counters = self.EQ_UE_Counters[direction][node_pos]
        capacity = self.EQ_CAPACITY[direction][node_pos]

    total_used = sum(counters.values())
    total_capacity = sum(capacity.values())
    return total_used / total_capacity if total_capacity > 0 else 0.0
```

#### 硬件实现分析 - 完全可行

| 方面 | 开销 | 详细说明 |
|------|------|---------|
| **存储** | ~18位 | 3个计数器（T0/T1/T2），每个约4-6位，足以表示0-63的Entry数量 |
| **计算** | ~20门 | 简单的3输入加法器，计算 T0+T1+T2 |
| **比较** | ~10门 | 与容量阈值比较，判断拥塞等级 |
| **访问延迟** | 0周期 | 本地寄存器直接访问，纯组合逻辑 |
| **布线** | 无额外开销 | Entry计数器本身就在CrossPoint内部 |

**结论**：Entry拥塞度**完全可以在硬件中实现**，开销极小（约50门/方向）。

---

## 限流策略设计

### 概率限流函数 - 仅用于仿真分析

使用平方函数作为限流概率，实现渐进式限流：

```python
def _should_throttle_probabilistic(self, congestion: float) -> bool:
    """
    概率限流（仅用于仿真）

    限流概率 = congestion^2
    """
    if congestion < self.config.CONGESTION_THROTTLE_THRESHOLD:
        return False

    throttle_probability = congestion ** 2
    return random.random() < throttle_probability
```

**限流概率对照表**：

| 拥塞度 | 限流概率 |
|--------|---------|
| 0.5    | 25%     |
| 0.6    | 36%     |
| 0.7    | 49%     |
| 0.8    | 64%     |
| 0.9    | 81%     |
| 1.0    | 100%    |

#### 硬件实现分析 - 为什么不实现

| 问题 | 严重程度 | 详细说明 |
|------|---------|---------|
| **平方运算** | ⭐⭐⭐ | 需要乘法器，虽然可用查找表替代，但增加面积 |
| **随机数生成** | ⭐⭐⭐ | 需要LFSR（线性反馈移位寄存器），约16-32位，每个方向都需要独立的随机源 |
| **不确定性** | ⭐⭐⭐⭐ | 概率行为导致系统行为不可预测，难以调试和验证 |
| **时序一致性** | ⭐⭐⭐ | 多个节点的随机决策可能导致全局行为不一致 |

**结论**：概率限流在硬件中**可以实现但不推荐**。随机性增加了系统复杂度和不确定性，不利于硬件验证。

---

### 阈值式限流 - 硬件实现方案

使用简单的阈值比较和周期计数实现确定性限流：

```python
def _should_throttle(self, congestion: float, cycle: int) -> bool:
    """
    阈值式限流（硬件友好）

    根据拥塞等级决定限流强度：
    - 正常（<50%）: 不限流
    - 中度（50%-75%）: 每2个周期允许1个flit上环
    - 严重（>=75%）: 每4个周期允许1个flit上环
    """
    if congestion < self.config.CONGESTION_MODERATE_THRESHOLD:  # 0.5
        return False  # 不限流
    elif congestion < self.config.CONGESTION_SEVERE_THRESHOLD:  # 0.75
        # 中度拥塞：50%限流率
        return (cycle % 2) == 0
    else:
        # 严重拥塞：75%限流率
        return (cycle % 4) != 0
```

#### 硬件实现分析 - 完全可行

| 组件 | 门数 | 说明 |
|------|------|------|
| 阈值比较器 | ~20门 | 2个比较器，判断拥塞等级 |
| 周期计数器 | ~10门 | 2-bit计数器，用于周期模运算 |
| 限流逻辑 | ~15门 | MUX选择不同限流模式 |
| **总计** | **~45门/方向** | 极低的硬件开销 |

**时序特性**：
- 纯组合逻辑 + 1个2-bit寄存器
- 关键路径：比较器 → MUX → 输出，约2-3级逻辑
- 可以在1个周期内完成决策

**结论**：阈值式限流**非常适合硬件实现**，逻辑简单、确定性强、易于验证。

---

### ITag例外机制

**关键设计**：已获得ITag预约的flit不受限流影响，必须上环。

```python
has_itag = flit.itag_h if is_horizontal else flit.itag_v
if has_itag:
    return True  # ITag flit必须上环，跳过限流检查
```

这是为了：
1. **保证公平性**：长时间等待的flit通过ITag机制获得优先权
2. **避免饥饿**：防止某些flit因持续限流而无法上环
3. **与现有机制协调**：ITag是已有的公平性保证机制

#### 硬件实现 - 完全可行

- 只需检查flit中的1-bit ITag标志
- 开销：1个AND门
- 无额外延迟

---

## 最终实现方案

### 修改 _can_inject_to_link 方法

```python
def _can_inject_to_link(self, flit, link, direction, cycle) -> bool:
    # ... 现有变量定义 ...

    if not link_occupied:
        if not slot.itag_reserved:
            # 拥塞感知限流检查
            has_itag = flit.itag_h if is_horizontal else flit.itag_v

            if not has_itag and self.config.CONGESTION_CONTROL_ENABLED:
                # 计算本地Entry拥塞度（硬件可实现）
                entry_congestion = self._calculate_local_entry_congestion(
                    current_pos, direction
                )

                # 阈值式限流（硬件可实现）
                if self._should_throttle(entry_congestion, cycle):
                    return False  # 被限流，本周期不上环

            # ITag flit或未被限流的flit可以上环
            return True
        elif slot.check_itag_match(current_pos, direction):
            return True  # ITag匹配，必须上环
        else:
            return False  # ITag不匹配
    else:
        # Link被占用的逻辑保持不变
        # ...
```

---

## 硬件实现成本总结

### 每个方向的硬件开销

| 组件 | 门数 | 说明 |
|------|------|------|
| Entry计数器（已有） | 0 | 复用现有E-Tag计数器 |
| 加法器 | ~20 | T0+T1+T2 |
| 阈值比较器 | ~20 | 2个比较器 |
| 周期计数器 | ~10 | 2-bit |
| 限流逻辑 | ~15 | MUX + AND |
| ITag检查 | ~5 | 1-bit检查 |
| **总计** | **~70门** | |

### 每个CrossPoint的总开销

- 每个CrossPoint管理2个方向（horizontal管理TL/TR，vertical管理TU/TD）
- 总开销：~70门 × 2方向 = **~140门/CrossPoint**

### 全芯片开销估算

以5×5拓扑为例：
- 25个节点 × 140门 = **~3500门**
- 对比：一个简单的32位加法器约需~300门
- **结论**：整个拥塞感知流控的开销约等于12个32位加法器，完全可接受

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| CONGESTION_CONTROL_ENABLED | False | 是否启用拥塞感知流控 |
| CONGESTION_MODERATE_THRESHOLD | 0.5 | 中度拥塞阈值（开始50%限流） |
| CONGESTION_SEVERE_THRESHOLD | 0.75 | 严重拥塞阈值（开始75%限流） |

### 参数调优建议

| 场景 | MODERATE | SEVERE | 说明 |
|------|----------|--------|------|
| 高吞吐优先 | 0.6 | 0.85 | 容忍更高拥塞，减少限流 |
| 低延迟优先 | 0.4 | 0.6 | 更早开始限流，避免flit在环上等待 |
| 平衡模式 | 0.5 | 0.75 | 默认值 |

---

## 修改文件清单

### 1. `src/noc/components/cross_point.py` (主要修改)

新增方法：
- `_calculate_local_entry_congestion(node_pos, direction)` - 本地Entry拥塞度
- `_should_throttle(congestion, cycle)` - 阈值式限流决策
- `_calculate_path_link_congestion(flit, direction)` - 路径Link拥塞度（仅用于分析）

修改方法：
- `_can_inject_to_link()` - 添加拥塞感知限流检查

### 2. `config/config.py` (添加配置)

```python
self.CONGESTION_CONTROL_ENABLED = getattr(args, "CONGESTION_CONTROL_ENABLED", False)
self.CONGESTION_MODERATE_THRESHOLD = getattr(args, "CONGESTION_MODERATE_THRESHOLD", 0.5)
self.CONGESTION_SEVERE_THRESHOLD = getattr(args, "CONGESTION_SEVERE_THRESHOLD", 0.75)
```

### 3. `config/default.yaml` (添加默认值)

```yaml
CONGESTION_CONTROL_ENABLED: false
CONGESTION_MODERATE_THRESHOLD: 0.5
CONGESTION_SEVERE_THRESHOLD: 0.75
```

---

## 设计优势

1. **硬件友好**: 仅使用本地信息，无跨节点通信，开销约140门/节点
2. **精准定位**: 在上环决策点进行流控，方向感知天然准确
3. **保证公平性**: ITag flit必须上环，与现有公平性机制协调
4. **确定性行为**: 阈值式限流避免了概率随机性，便于调试和验证
5. **最小改动**: 主要修改集中在 `cross_point.py` 一个文件
6. **可配置**: 通过配置参数控制是否启用和限流强度

---

## 与其他流控机制的关系

本方案与现有流控机制的关系：

| 机制 | 层级 | 作用 | 与本方案关系 |
|------|------|------|-------------|
| Token Bucket | IP层 | 带宽限制 | 独立，可同时启用 |
| E-Tag | CrossPoint | 下环优先级 | **数据共享**，复用Entry计数器 |
| I-Tag | CrossPoint | 上环预约 | **协调**，ITag flit不受限流 |
| FIFO深度 | Network | 缓冲管理 | 独立，满时阻塞 |
| Tracker | IP层 | 事务并发控制 | 独立，控制Outstanding数 |

---

## 仿真专用功能

以下功能仅在仿真模型中实现，用于性能分析，不对应实际硬件：

### 1. 路径Link拥塞度统计

```python
def _calculate_path_link_congestion(self, flit, direction) -> float:
    """仅用于仿真分析，统计路径上的拥塞情况"""
    # ... 实现代码 ...
```

**用途**：
- 分析flit延迟与路径拥塞的相关性
- 评估本地Entry拥塞度作为近似指标的有效性
- 为未来可能的硬件改进提供数据支持

### 2. 概率限流模式

```python
def _should_throttle_probabilistic(self, congestion: float) -> bool:
    """仅用于仿真对比研究"""
    # ... 实现代码 ...
```

**用途**：
- 对比概率限流与阈值限流的性能差异
- 研究不同限流策略对吞吐量和延迟的影响

---

**文档版本**: v2.0
**最后更新**: 2025-11-27
**状态**: 设计完成，包含硬件可行性分析
