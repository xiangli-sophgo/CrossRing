# CrossRing仲裁系统使用指南

## 概述

CrossRing仲裁系统提供了多种仲裁算法，支持传统的单选仲裁和先进的最大权重匹配算法。本文档介绍如何配置和使用这些仲裁算法。

## 仲裁算法类型

### 1. 传统单选仲裁算法

#### 轮询仲裁 (Round Robin)
```yaml
arbitration:
  iq:
    type: "round_robin"
```

#### 加权轮询 (Weighted Round Robin)
```yaml
arbitration:
  eq:
    type: "weighted"
    weights: [3, 2, 2, 1]  # 对应TU, TD, IQ, RB的权重
```

#### 固定优先级 (Fixed Priority)
```yaml
arbitration:
  rb:
    type: "priority"
    priorities: [1, 2, 3, 4]  # 数字越小优先级越高
```

#### 动态优先级 (Dynamic Priority)
```yaml
arbitration:
  default:
    type: "dynamic"
    base_priorities: [1, 2, 3, 4]
    aging_factor: 1.5  # 老化因子
```

#### 随机仲裁 (Random)
```yaml
arbitration:
  default:
    type: "random"
    seed: 42  # 可选，用于结果重现
```

#### 令牌桶仲裁 (Token Bucket)
```yaml
arbitration:
  default:
    type: "token_bucket"
    bucket_size: 10
    refill_rate: 1.0
    port_rates: [1.0, 0.8, 0.6, 0.4]  # 各端口补充速率
```

### 2. 最大权重匹配算法

最大权重匹配算法专门用于多对多匹配场景，能够显著提升NoC的吞吐量。

#### 批量匹配适配器 (推荐)
```yaml
arbitration:
  iq:
    type: "batch_matching"
    batch_mode: "auto"          # auto/manual
    enable_fallback: true       # 启用降级机制
    batch_timeout: 0.001        # 批量超时时间(秒)
    matcher:
      algorithm: "islip"        # islip/lqf/ocf/pim
      iterations: 2             # 迭代次数
      weight_strategy: "queue_length"  # uniform/queue_length/wait_time/hybrid
```

#### 直接使用匹配算法
```yaml
arbitration:
  # 仅在特殊场景下使用，一般推荐batch_matching
  crossbar:
    type: "islip"
    iterations: 3
    weight_strategy: "hybrid"
```

## 匹配算法详解

### iSLIP算法
经典的三阶段迭代匹配算法：
- **请求阶段**：输入端口向所需输出端口发送请求
- **授权阶段**：输出端口从请求中选择一个授权
- **接受阶段**：输入端口从授权中选择一个接受

```yaml
matcher:
  algorithm: "islip"
  iterations: 3  # 多轮迭代提高匹配质量
```

### LQF (Longest Queue First)
基于队列长度的优先级匹配：
```yaml
matcher:
  algorithm: "lqf"
  weight_strategy: "queue_length"
```

### OCF (Oldest Cell First)
基于等待时间的优先级匹配：
```yaml
matcher:
  algorithm: "ocf"
  weight_strategy: "wait_time"
```

### PIM (Parallel Iterative Matching)
并行迭代匹配算法：
```yaml
matcher:
  algorithm: "pim"
  iterations: 2
```

## 权重策略

### 统一权重 (Uniform)
所有请求具有相同权重：
```yaml
weight_strategy: "uniform"
```

### 队列长度权重 (Queue Length)
根据队列深度分配权重，队列越深权重越高：
```yaml
weight_strategy: "queue_length"
```

### 等待时间权重 (Wait Time)
根据等待时间分配权重，等待越久权重越高：
```yaml
weight_strategy: "wait_time"
```

### 混合权重 (Hybrid)
结合队列长度和等待时间：
```yaml
weight_strategy: "hybrid"  # 70%队列长度 + 30%等待时间
```

## 应用场景建议

### IQ注入仲裁
推荐使用批量匹配适配器，能够同时处理多个IP类型向不同方向的注入：

```yaml
arbitration:
  iq:
    type: "batch_matching"
    matcher:
      algorithm: "islip"
      iterations: 2
      weight_strategy: "queue_length"
```

**优势**：
- 多个IP类型可以并行注入到不同方向
- 基于队列长度的公平调度
- 降级机制保证兼容性

### Ring Bridge仲裁
对于简单场景可以使用轮询，复杂场景推荐批量匹配：

```yaml
arbitration:
  rb:
    type: "round_robin"  # 简单场景
    # 或者
    type: "batch_matching"  # 复杂场景
    matcher:
      algorithm: "lqf"
      weight_strategy: "queue_length"
```

### EQ弹出仲裁
根据应用需求选择合适算法：

```yaml
arbitration:
  eq:
    # 公平调度
    type: "round_robin"
    # 或优先级调度
    type: "weighted"
    weights: [3, 2, 2, 1]
    # 或批量优化
    type: "batch_matching"
    matcher:
      algorithm: "islip"
```

## 性能调优

### 批量大小优化
```yaml
arbitration:
  iq:
    type: "batch_matching"
    batch_mode: "auto"  # 自动检测批量大小
    batch_timeout: 0.001  # 根据延迟要求调整
```

### 迭代次数调优
```yaml
matcher:
  iterations: 1  # 低延迟要求
  iterations: 3  # 高吞吐量要求
```

### 权重策略选择
- **uniform**：负载均匀时使用
- **queue_length**：缓解拥塞
- **wait_time**：保证公平性
- **hybrid**：平衡性能和公平性

## 监控和调试

### 获取统计信息
```python
# 获取仲裁器统计
stats = model.iq_arbiter.get_stats()
print(f"成功率: {stats['success_rate']}")
print(f"公平性指数: {stats['fairness_index']}")

# 获取批量匹配统计
if hasattr(model.iq_arbiter, 'get_batch_stats'):
    batch_stats = model.iq_arbiter.get_batch_stats()
    print(f"平均批量大小: {batch_stats['average_batch_size']}")
    print(f"降级次数: {batch_stats['fallback_count']}")
```

### 常见问题排查

1. **批量匹配不生效**
   - 检查配置文件格式
   - 确认enable_fallback=true
   - 查看fallback_count统计

2. **性能没有提升**
   - 增加iterations数量
   - 调整weight_strategy
   - 检查batch_timeout设置

3. **出现异常**
   - 启用enable_fallback降级机制
   - 检查queue_id格式
   - 查看日志文件

## 配置文件示例

完整的配置文件示例：

```yaml
# config/topologies/example.yaml
arbitration:
  # 全局默认配置
  default:
    type: "round_robin"

  # IQ注入仲裁 - 使用批量匹配优化
  iq:
    type: "batch_matching"
    batch_mode: "auto"
    enable_fallback: true
    batch_timeout: 0.001
    matcher:
      algorithm: "islip"
      iterations: 2
      weight_strategy: "queue_length"

  # Ring Bridge仲裁 - 使用加权轮询
  rb:
    type: "weighted"
    weights: [2, 2, 1, 1]  # TL, TR, TU, TD

  # EQ弹出仲裁 - 使用优先级
  eq:
    type: "priority"
    priorities: [1, 2, 3, 4]  # TU, TD, IQ, RB

  # D2D仲裁 - 使用轮询
  d2d:
    type: "round_robin"

  # 双通道仲裁 - 使用随机
  dual_channel:
    type: "random"
```

## 总结

CrossRing仲裁系统提供了从简单轮询到复杂匹配算法的完整解决方案。根据应用场景选择合适的算法：

- **简单场景**：使用轮询、优先级等传统算法
- **性能优化**：使用批量匹配适配器
- **特殊需求**：使用令牌桶、动态优先级等高级算法

通过合理配置，可以在保证兼容性的前提下显著提升NoC系统的性能。