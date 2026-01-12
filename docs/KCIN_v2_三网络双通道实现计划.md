# KCIN v2 三网络双通道实现计划

## 需求
在v2版本中实现三网络（req、rsp、data）的双通道支持。

## 现状分析

### v1双通道实现
- 只支持data网络双通道
- 使用两个独立的Network实例（`data_network_ch0`, `data_network_ch1`）
- 通道选择在IP接口层（`l2h_to_IQ_channel_buffer`时）
- 接收使用2to1轮询仲裁合并两个通道

### v2架构特点
- 使用RingStation架构（非IQ/RB/EQ）
- IP接口使用`tx_channel_buffer_pre`和`rx_channel_buffer`
- `networks`字典结构：每个网络类型有独立的缓冲区配置

## 设计方案

### 核心思路
采用**Network层双实例**方案：每种网络类型创建两个独立的Network实例。

```
三网络 × 双通道 = 6个Network实例
- req_network_ch0, req_network_ch1
- rsp_network_ch0, rsp_network_ch1
- data_network_ch0, data_network_ch1
```

### 数据流设计

**发送路径**：
```
inject_fifo → l2h_fifo → 通道选择 → tx_channel_buffer_pre[ch0/ch1] → RS[ch0/ch1].input_fifos
```

**接收路径**：
```
RS[ch0].output_fifos ──┐
                       ├→ 2to1仲裁 → rx_channel_buffer → h2l_fifo → eject
RS[ch1].output_fifos ──┘
```

### 通道选择策略
三网络使用**统一的通道选择策略**，通过配置参数选择：
- `ip_id_based`：基于IP的ID奇偶（默认）
- `target_node_based`：基于目标节点ID奇偶
- `packet_id_based`：基于packet_id奇偶（保证同一请求的flit在同一通道）

## 实现步骤

### 步骤1：扩展ChannelSelector
**文件**: `src/kcin/base/channel_selector.py`

修改内容：
- 添加`TriNetworkChannelSelector`类，支持三网络独立配置策略
- 添加`packet_id_based`策略

### 步骤2：创建DualChannelIPInterface
**文件**: `src/kcin/v2/components/dual_channel_ip_interface.py`（新建）

核心设计：
```python
class DualChannelIPInterface(IPInterface):
    def __init__(self, ...,
                 req_network_ch0, req_network_ch1,
                 rsp_network_ch0, rsp_network_ch1,
                 data_network_ch0, data_network_ch1, ...):
        # 双通道网络引用
        self.networks_ch0 = {"req": req_network_ch0, "rsp": rsp_network_ch0, "data": data_network_ch0}
        self.networks_ch1 = {"req": req_network_ch1, "rsp": rsp_network_ch1, "data": data_network_ch1}

        # 双通道缓冲结构（每个网络类型两个通道）
        self.tx_channel_buffer_pre = {
            "req": {"ch0": None, "ch1": None}, ...
        }
        self.rx_channel_buffer = {
            "req": {"ch0": deque(), "ch1": deque()}, ...
        }
```

关键方法重写：
- `l2h_to_tx_channel_buffer()`: 在此进行通道选择
- `rx_channel_buffer_to_h2l_pre()`: 实现2to1仲裁

### 步骤3：创建TriNetworkDualChannelModel
**文件**: `src/kcin/v2/tri_network_dual_channel_model.py`（新建）

核心设计：
```python
class TriNetworkDualChannelModel(BaseModel):
    def initial(self):
        # 创建6个网络实例
        self.req_network_ch0 = Network(...)
        self.req_network_ch1 = Network(...)
        self.rsp_network_ch0 = Network(...)
        self.rsp_network_ch1 = Network(...)
        self.data_network_ch0 = Network(...)
        self.data_network_ch1 = Network(...)

    def step(self):
        # 并行处理6个网络
        for ch in [0, 1]:
            self.move_flits_in_network(self.req_network_ch[ch], ...)
            self.move_flits_in_network(self.rsp_network_ch[ch], ...)
            self.move_flits_in_network(self.data_network_ch[ch], ...)
```

### 步骤4：修改DataflowMixin
**文件**: `src/kcin/v2/mixins/dataflow_mixin.py`

修改内容：
- 添加`_move_pre_to_queues_dual_channel()`方法
- 支持双通道的IP↔RS数据传输

### 步骤5：更新组件导出
**文件**: `src/kcin/v2/components/__init__.py`

添加导出：
- `DualChannelIPInterface`

### 步骤6：添加配置参数
**文件**: `src/kcin/v2/config.py`

添加参数：
```python
DUAL_CHANNEL_ENABLED = False  # 三网络双通道总开关
CHANNEL_SELECT_STRATEGY = "ip_id_based"  # 统一通道选择策略
```

## 关键文件列表

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/kcin/base/channel_selector.py` | 修改 | 添加TriNetworkChannelSelector |
| `src/kcin/v2/components/dual_channel_ip_interface.py` | 新建 | v2双通道IP接口 |
| `src/kcin/v2/tri_network_dual_channel_model.py` | 新建 | 三网络双通道模型 |
| `src/kcin/v2/mixins/dataflow_mixin.py` | 修改 | 添加双通道数据流方法 |
| `src/kcin/v2/components/__init__.py` | 修改 | 导出新组件 |
| `src/kcin/v2/config.py` | 修改 | 添加双通道配置 |

## 验证方法

1. 创建测试脚本验证双通道数据流
2. 对比单通道和双通道的仿真结果
3. 验证每个通道的统计数据独立收集
4. 检查结果合并逻辑正确性
