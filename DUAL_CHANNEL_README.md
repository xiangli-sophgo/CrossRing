# CrossRing 双通道数据网络

这个文档描述了CrossRing系统中的双通道数据传输功能的实现和使用方法。

## 功能概述

双通道数据网络将原有的单一数据传输通道扩展为两个独立的并行通道，提供：

- **更高的数据传输带宽** - 理论上可提升80-100%的数据吞吐量
- **更好的负载分布** - 通过智能通道选择减少拥塞
- **灵活的QoS控制** - 支持不同优先级和类型的数据分离传输
- **完全向后兼容** - 不影响原有的请求(REQ)和响应(RSP)网络

## 架构设计

### 双通道架构
```
数据路径：
IP → inject_fifo[ch0/ch1] → L2H[ch0/ch1] → IQ_channel_buffer[ch0/ch1] 
    → IQ仲裁[ch0/ch1] → inject_queues[TL/TR/TU/TD][ch0/ch1] 
    → RB[ch0/ch1] → Network Links[ch0/ch1]
    → eject_queues[TU/TD][ch0/ch1] → EQ仲裁[ch0/ch1] 
    → EQ_channel_buffer[ch0/ch1] → H2L[ch0/ch1] → IP
```

### 关键特性
- **完全物理隔离**: 每个通道拥有独立的队列、缓冲区和仲裁逻辑
- **独立仲裁**: IQ、RB、EQ都有双通道独立仲裁
- **灵活策略**: 支持多种通道选择策略
- **统计监控**: 详细的双通道性能统计

## 快速开始

### 1. 基本使用

```python
from src.core.dual_channel_base_model import DualChannelBaseModel
from config.dual_channel_config import DualChannelConfig

# 创建双通道配置
config = DualChannelConfig()
config.TOPO_TYPE = "5x4"

# 创建双通道模型
sim = DualChannelBaseModel(
    model_type="DualChannel_REQ_RSP",
    config=config,
    topo_type="5x4",
    traffic_file_path="",
    traffic_config=[["test_data.txt"]],
    result_save_path="../Result/dual_channel/",
    verbose=1,
)

# 运行仿真
sim.initial()
sim.end_time = 1000
sim.run()

# 查看结果
sim.print_dual_channel_summary()
```

### 2. 运行示例

```bash
# 简单演示
cd example
python simple_dual_channel_demo.py

# 完整演示(包含多种策略对比)
python dual_channel_demo.py

# 功能测试
cd scripts
python dual_channel_test.py
```

## 通道选择策略

### 可用策略

1. **hash_based** (默认) - 基于源目的地址哈希分配
   ```python
   config.DATA_CHANNEL_SELECT_STRATEGY = "hash_based"
   ```

2. **size_based** - 基于包大小分配
   ```python
   config.DATA_CHANNEL_SELECT_STRATEGY = "size_based"
   # 小包(≤4 flits)走通道0，大包走通道1
   ```

3. **type_based** - 基于读写类型分配
   ```python
   config.DATA_CHANNEL_SELECT_STRATEGY = "type_based"
   # 读数据走通道0，写数据走通道1
   ```

4. **load_balanced** - 基于实时负载动态分配
   ```python
   config.DATA_CHANNEL_SELECT_STRATEGY = "load_balanced"
   ```

5. **round_robin** - 轮询分配
   ```python
   config.DATA_CHANNEL_SELECT_STRATEGY = "round_robin"
   ```

### 自定义通道选择器

```python
from src.utils.channel_selector import ChannelSelector

class CustomChannelSelector(ChannelSelector):
    def select_channel(self, flit):
        # 自定义选择逻辑
        if flit.ETag_priority == "T0":
            return 0  # 高优先级走通道0
        else:
            return 1  # 其他走通道1
```

## 配置选项

### 基本配置

```python
config = DualChannelConfig()

# 启用/禁用双通道
config.DATA_DUAL_CHANNEL_ENABLED = True

# 通道选择策略
config.DATA_CHANNEL_SELECT_STRATEGY = "hash_based"

# 通道带宽分配比例
config.DATA_CH0_BANDWIDTH_RATIO = 0.5
config.DATA_CH1_BANDWIDTH_RATIO = 0.5

# 通道优先级
config.DATA_CHANNEL_0_PRIORITY = "normal"  # normal, high, low
config.DATA_CHANNEL_1_PRIORITY = "normal"
```

### 缓冲区配置

```python
# 设置各通道的FIFO深度
config.set_channel_fifo_depths(
    ch0_iq_depth=8,
    ch1_iq_depth=8,
    ch0_eq_depth=8,
    ch1_eq_depth=8
)
```

### 预定义配置

```python
from config.dual_channel_config import (
    create_balanced_dual_channel_config,
    create_read_write_separated_config,
    create_size_based_dual_channel_config
)

# 均衡配置
config = create_balanced_dual_channel_config()

# 读写分离配置
config = create_read_write_separated_config()

# 基于包大小的配置
config = create_size_based_dual_channel_config()
```

## 主要文件结构

```
src/
├── core/
│   └── dual_channel_base_model.py      # 双通道基础模型
├── utils/
│   ├── components/
│   │   ├── dual_channel_network.py     # 双通道网络
│   │   ├── dual_channel_ip_interface.py# 双通道IP接口
│   │   └── flit.py                     # 修改：添加data_channel_id
│   └── channel_selector.py             # 通道选择策略
config/
└── dual_channel_config.py              # 双通道配置
example/
├── dual_channel_demo.py                # 完整演示
└── simple_dual_channel_demo.py         # 简单演示
scripts/
├── dual_channel_test.py                # 功能测试
└── dual_channel_example.py             # 使用示例
```

## 性能监控

### 获取统计信息

```python
# 运行仿真后
stats = sim.get_dual_channel_statistics()

print(f"通道0注入包数: {stats['total_inject_ch0']}")
print(f"通道1注入包数: {stats['total_inject_ch1']}")
print(f"通道0平均延迟: {stats['avg_latency_ch0']}")
print(f"通道1平均延迟: {stats['avg_latency_ch1']}")
```

### 通道分布分析

```python
# 分析通道选择的分布情况
selector = DefaultChannelSelector("hash_based")
channel_counts = {0: 0, 1: 0}

for i in range(1000):
    flit = create_test_flit(i)
    channel = selector.select_channel(flit)
    channel_counts[channel] += 1

print(f"通道分布: CH0={channel_counts[0]}, CH1={channel_counts[1]}")
```

## 高级功能

### 自适应通道选择

```python
from src.utils.channel_selector import AdaptiveChannelSelector

# 创建自适应选择器
selector = AdaptiveChannelSelector(network=data_network)

# 在DualChannelIPInterface中使用
ip_interface = DualChannelIPInterface(..., channel_selector=selector)
```

### 配置保存与加载

```python
# 保存配置
config.save_dual_channel_config("my_dual_channel_config.json")

# 加载配置
config.load_dual_channel_config("my_dual_channel_config.json")
```

## 调试和故障排除

### 常见问题

1. **通道分布不均**: 
   - 检查通道选择策略是否合适
   - 尝试使用"load_balanced"策略

2. **性能提升不明显**:
   - 确认流量模式适合双通道
   - 检查通道带宽配置

3. **仿真错误**:
   - 检查路径和依赖
   - 确认配置参数合理

### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 创建模型时设置verbose
sim = DualChannelBaseModel(..., verbose=2)
```

## 性能预期

基于测试结果，双通道数据网络预期可以提供：

- **吞吐量提升**: 80-100%
- **延迟改善**: 5-15% 
- **拥塞减少**: 显著改善高负载场景
- **负载均衡**: 更好的资源利用率

## 贡献和扩展

### 添加新的通道选择策略

1. 继承`ChannelSelector`基类
2. 实现`select_channel(flit)`方法
3. 在配置中注册新策略

### 扩展到更多通道

当前实现支持双通道(2个)，可以扩展支持更多通道：

1. 修改`DualChannelDataNetwork`中的通道数量
2. 更新配置选项
3. 扩展通道选择策略

## 许可证

遵循CrossRing项目的原有许可证。