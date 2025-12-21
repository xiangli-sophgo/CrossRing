# D2D微架构设计文档

## 文档概述

本文档详细解析Die-to-Die (D2D) 互连的微架构设计，基于UCIe (Universal Chiplet Interconnect Express) 标准，并说明CrossRing项目中的具体实现方案。

**目标读者**: 芯片架构师、NoC设计工程师、性能建模研究人员

## 术语表

| 术语   | 全称                                   | 中文                 | 说明                   |
| ------ | -------------------------------------- | -------------------- | ---------------------- |
| UCIe   | Universal Chiplet Interconnect Express | 通用芯片互连快速标准 | Die间互连标准          |
| D2D    | Die-to-Die                             | 芯片间               | 多Die芯片的互连        |
| RN     | Request Node                           | 请求节点             | 发起事务的节点         |
| SN     | Slave Node                             | 从节点               | 接收事务的节点         |
| FAB    | Fabric                                 | 结构互连             | 聚合/分流逻辑          |
| ASYNC  | Asynchronous                           | 异步                 | 时钟域转换             |
| Cbuf   | Circular Buffer                        | 循环缓冲区           | 流控缓冲               |
| DMUX   | Demultiplexer                          | 解复用器             | 流量分离               |
| GT/s   | GigaTransfers per second               | 每秒千兆次传输       | 链路速率单位           |
| SerDes | Serializer/Deserializer                | 串行器/解串器        | 物理层组件             |
| STI    | Standard Tiled Interconnect            | 标准平铺互连         | Die内总线接口          |
| Lane   | Physical Signal Pair                   | 物理信号对           | UCIe的物理传输通道     |
| Flit   | Flow Control Unit                      | 流控单元             | 网络传输的基本数据单位 |

## 1. UCIe物理层微架构详解

### 1.1 整体架构图解析

基于提供的D2D微架构图，完整的Die间互连包含以下分层结构：

```
┌─────────────────────────────────────────────────────────┐
│                    UCIe Physical Layer                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │UCIe x16 │ │UCIe x16 │ │UCIe x16 │ │UCIe x16 │       │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
└───────┼──────────┼──────────┼──────────┼───────────────┘
        │          │          │          │
        └──────────┴──────────┴──────────┘
                    │
        ┌───────────▼───────────────────┐
        │   FAB (Fabric Interconnect)   │
        │   ┌─────────┬─────────┐       │
        │   │4to1 Mux │1to4 DMux│       │
        │   └────┬────┴────┬────┘       │
        └────────┼─────────┼─────────────┘
                 │         │
        ┌────────▼─────────▼────────┐
        │   RN / SN Protocol Layer   │
        │   (CHI or AXI)             │
        └────────┬─────────┬─────────┘
                 │         │
        ┌────────▼─────────▼────────┐
        │   ASYNC Clock Domain       │
        │   Crossing                 │
        └────────┬─────────┬─────────┘
                 │         │
        ┌────────▼─────────▼────────┐
        │   Cbuf (Circular Buffer)   │
        │   Flow Control             │
        └────────┬─────────┬─────────┘
                 │         │
        ┌────────▼─────────▼────────┐
        │   DMUX 1to2                │
        │   Traffic Demultiplexing   │
        └────────┬─────────┬─────────┘
                 │         │
        ┌────────▼─────────▼────────┐
        │   STI 1024bit Bus          │
        │   ┌──┬──┬──┬──┐            │
        │   │EQ│IQ│RB│P │            │
        │   └──┴──┴──┴──┘            │
        └────────┬──────────────────┘
                 │
        ┌────────▼─────────────────┐
        │   CrossRing Network       │
        └──────────────────────────┘
```

### 1.2 四条UCIe x16通道配置

#### 1.2.1 带宽聚合工作方式

**四条UCIe x16链路协同传输**：

四条物理链路并行工作，共同传输每一个Flit：

- **每条链路**：x16 = 16条Lane，每条Lane 1bit宽度 → 16 bits/cycle
- **四条并行**：4 × 16 = 64 bits/cycle
- **传输128B Flit**：(128 × 8 bits) ÷ 64 bits/cycle = 16 UCIe周期

**Flit分割示例**：

```
┌─────────────────────────────────────┐
│  128 Bytes Flit（1024 bits）        │
├─────────────────────────────────────┤
│  UCIe #0: Bytes 0-31    (256 bits)  │
│  UCIe #1: Bytes 32-63   (256 bits)  │
│  UCIe #2: Bytes 64-95   (256 bits)  │
│  UCIe #3: Bytes 96-127  (256 bits)  │
└─────────────────────────────────────┘
        ↓ 四条同时传输16周期
    接收端FAB 4to1重组
```

**关键特性**：

1. **通道无关**：所有AXI通道（AR/R/AW/W/B）的Flit都通过4条UCIe并行传输
2. **FAB分发**：FAB 1to4负责将Flit分割到4条物理链路
3. **FAB聚合**：FAB 4to1负责在接收端重组Flit
4. **带宽提升**：4倍于单条x16链路的带宽（252 GB/s vs 63 GB/s）

#### 1.2.2 带宽计算示例

假设UCIe规格：

- **速率**: 32 GT/s（每通道每Lane）
- **编码**: 128b/130b（UCIe标准编码）
- **Lane数**: 16条（每通道）

**单通道带宽计算**：

```
原始速率 = 32 GT/s × 16 lanes = 512 GT/s

考虑编码开销：
有效数据率 = 512 GT/s × (128/130) ≈ 504.62 Gbps

转换为字节：
单通道带宽 = 504.62 Gbps ÷ 8 ≈ 63.08 GB/s
```

**四通道聚合带宽**：

```
总带宽 = 63.08 GB/s × 4 ≈ 252.3 GB/s
```

**实际可用带宽**（考虑协议开销10%）：

```
实际带宽 ≈ 252.3 GB/s × 0.9 ≈ 227 GB/s
```

### 1.3 FAB层（Fabric Interconnect）

**核心作用**：将单一逻辑Flit与4条物理UCIe链路之间进行转换

**为什么需要FAB**：

- **Die内部**：使用完整的128B Flit进行逻辑传输
- **UCIe物理层**：需要分割成4份并行传输以提升带宽
- **FAB层**：负责分割（1to4）和重组（4to1）

```
逻辑层（单一Flit流）
       ↓
   FAB 1to4 分割
       ↓
┌──────┬──────┬──────┬──────┐
│ 1/4  │ 2/4  │ 3/4  │ 4/4  │  物理层（4条UCIe并行）
└──────┴──────┴──────┴──────┘
       ↓
   FAB 4to1 聚合
       ↓
逻辑层（重组后的Flit）
```

#### 1.3.1 FAB 4to1聚合器

**功能**: 从四条UCIe物理通道接收分片数据，重组为完整Flit

**工作流程**：

1. **并行接收**: 同时从4条UCIe链路接收数据分片
2. **缓冲对齐**: 等待同一Flit的4个分片全部到达
3. **重组**: 按照顺序拼接4个32B分片为完整的128B Flit
4. **输出**: 将重组后的Flit交付给RN/SN模块

**关键特性**：

- **同步机制**: 确保4个分片属于同一个Flit（通过序列号）
- **低延迟**: 最后一个分片到达后立即重组
- **背压处理**: 输出队列满时暂停接收新分片

#### 1.3.2 FAB 1to4扇出器

**功能**: 将完整Flit分割成4个分片，分发到四条UCIe物理通道

**工作流程**：

1. **接收Flit**: 从RN/SN模块接收完整的128B Flit
2. **分割**: 将Flit平均分成4个32B分片
   - 分片0: Bytes 0-31   → UCIe链路#0
   - 分片1: Bytes 32-63  → UCIe链路#1
   - 分片2: Bytes 64-95  → UCIe链路#2
   - 分片3: Bytes 96-127 → UCIe链路#3
3. **同步发送**: 同时将4个分片发送到对应的UCIe链路
4. **序列号标记**: 为每个分片添加序列号，便于接收端重组

**关键特性**：

1. **固定分割**: 总是按照固定位置分割，不区分流量类型
2. **并行发送**: 4个分片同时发送，最大化带宽利用
3. **原子性**: 确保同一Flit的4个分片同时传输

### 1.4 RN/SN协议层

#### 1.4.1 RN (Request Node) 模块

**职责**: 发起跨Die事务，处理协议转换

```
RN核心功能：
┌────────────────────────────────────┐
│  1. 接收Die内请求                   │
│  2. 协议转换（内部 → UCIe）         │
│  3. 事务跟踪（Tracker管理）         │
│  4. 流控管理（Credit based）        │
│  5. 发送到UCIe物理层                │
└────────────────────────────────────┘
```

**AXI协议映射**（以读事务为例）：

```
Die内请求 → RN处理 → UCIe传输

Step 1: 接收GDMA读请求
  - packet_id: 12345
  - address: 0x1000_0000
  - burst_length: 4

Step 2: 协议转换为AXI AR
  - ARID: 12345
  - ARADDR: 0x1000_0000
  - ARLEN: 3 (burst_length - 1)
  - ARSIZE: 6 (64 bytes)

Step 3: 发送到FAB → UCIe通道
  - 选择UCIe #0（读地址通道）
  - 添加AR通道延迟（5ns → 10 cycles）
```

#### 1.4.2 SN (Slave Node) 模块

**职责**: 接收跨Die事务，处理协议转换

```
SN核心功能：
┌────────────────────────────────────┐
│  1. 从UCIe物理层接收                │
│  2. 协议转换（UCIe → 内部）         │
│  3. 资源检查（Tracker/Buffer）      │
│  4. 转发到Die内目标                 │
│  5. 返回响应数据                    │
└────────────────────────────────────┘
```

**资源管理**：

- **Tracker**: 跟踪未完成的跨Die事务
  - Read Tracker: 48个
  - Write Tracker: 48个
- **Data Buffer**:
  - Read Data Buffer (RDB): 192 entries
  - Write Data Buffer (WDB): 192 entries

### 1.5 ASYNC时钟域转换

#### 1.5.1 为什么需要ASYNC？

**问题**: 两个Die可能运行在不同频率

- Die 0: 2.0 GHz
- Die 1: 2.5 GHz
- UCIe链路: 固定物理时钟

**ASYNC模块作用**：

```
┌─────────────┐  ASYNC   ┌─────────────┐
│ Die0 Clock  │ ──────→  │ UCIe Clock  │
│   2.0 GHz   │ ←──────  │   3.2 GHz   │
└─────────────┘          └─────────────┘
      │                        │
      ▼                        ▼
  异步FIFO写              异步FIFO读
  (Die0时钟域)           (UCIe时钟域)
```

#### 1.5.2 异步FIFO设计

**双时钟FIFO**：

```python
class AsyncFIFO:
    def __init__(self, depth=16):
        self.memory = [None] * depth
        self.wr_ptr = 0  # 写指针（Die0时钟域）
        self.rd_ptr = 0  # 读指针（UCIe时钟域）
        self.depth = depth

    def write(self, data, wr_clk):
        # 在Die0时钟上升沿写入
        if not self.is_full():
            self.memory[self.wr_ptr] = data
            self.wr_ptr = (self.wr_ptr + 1) % self.depth

    def read(self, rd_clk):
        # 在UCIe时钟上升沿读取
        if not self.is_empty():
            data = self.memory[self.rd_ptr]
            self.rd_ptr = (self.rd_ptr + 1) % self.depth
            return data
```

**亚稳态处理**：

- 使用格雷码传递指针跨时钟域
- 双级同步器消除亚稳态
- 保守的满/空判断逻辑

**延迟特性**：

- **最小延迟**: 2个时钟周期（写入→同步→读出）
- **最大延迟**: FIFO深度相关（拥塞时）
- **CrossRing建模**: 可配置固定延迟或动态延迟

### 1.6 Cbuf（循环缓冲区）

#### 1.6.1 流控缓冲作用

**位置**: ASYNC与DMUX之间，提供弹性缓冲

```
Cbuf功能：
1. 速率解耦：RN/SN处理速率 ≠ 网络传输速率
2. 突发吸收：应对瞬时流量峰值
3. Credit管理：实现反压流控
```

**Credit-based流控示例**：

```python
class Cbuf:
    def __init__(self, size=64):
        self.buffer = deque(maxlen=size)
        self.credits = size  # 初始credit等于buffer大小
        self.remote_credits = size  # 对方Cbuf的credit

    def enqueue(self, flit):
        if self.remote_credits > 0:
            self.buffer.append(flit)
            self.remote_credits -= 1
            return True
        return False  # 无credit，阻塞

    def dequeue(self):
        if self.buffer:
            flit = self.buffer.popleft()
            self.credits += 1
            self.send_credit_return()  # 通过侧边通道返回credit
            return flit

    def receive_credit_return(self, count=1):
        self.remote_credits += count
```

#### 1.6.2 与CrossRing现有设计的关系

CrossRing中的**IPInterface缓冲**可视为简化的Cbuf：

- `pre_buffer`: 接收缓冲（类似Cbuf入口）
- `post_buffer`: 发送缓冲（类似Cbuf出口）

**增强方向**：

- 添加显式的credit管理机制
- 实现credit返回通道（当前未建模）
- 支持可配置的缓冲深度

### 1.7 DMUX 1to2解复用器

#### 1.7.1 流量分离

**功能**: 将聚合的数据流分离为不同类型

```
DMUX分流策略：
┌──────────────┐
│  聚合流量     │
└───────┬──────┘
        │
   ┌────▼────┐
   │  DMUX   │
   └─┬─────┬─┘
     │     │
     ▼     ▼
  ┌───┐ ┌───┐
  │EQ │ │IQ │
  └───┘ └───┘
   出口   入口
```

**分流规则**：

1. **方向判断**: 基于flit的目标地址
   - 目标在本Die → IQ（入队列）
   - 目标在其他Die → EQ（出队列）
2. **优先级分离**:
   - 高优先级流量 → 快速通道
   - 低优先级流量 → 常规通道

### 1.8 STI 1024bit总线接口

#### 1.8.1 超宽总线设计

**STI (Standard Tiled Interconnect)** 规格：

- **位宽**: 1024 bits = 128 Bytes
- **频率**: 2 GHz（示例）
- **理论带宽**: 128 B × 2 GHz = 256 GB/s

**与CrossRing网络的接口**：

```
STI 1024bit总线 → CrossRing适配器
┌──────────────────┐
│  1024bit并行数据  │
└────────┬─────────┘
         │ 转换为Flit序列
         ▼
┌──────────────────┐
│  Flit流（64B/个） │  → CrossRing网络
└──────────────────┘
```

**转换逻辑**：

- 1024bit总线 = 16个64B Flit（并行传输）
- 或者1个128B Flit（根据Flit大小配置）

**工作流程示例**：

```
发送路径（跨Die）：
CrossRing → DMUX → Cbuf → ASYNC → RN/SN → FAB → UCIe

接收路径（跨Die）：
UCIe → FAB → RN/SN → ASYNC → Cbuf → DMUX → CrossRing
```

#### 1.8.2 并行到串行的转换过程

**完整数据路径解析**：

**发送端（Die内部 → UCIe）**：

```
Step 1: STI 1024bit并行总线
  - 1个CrossRing网络周期传输128B Flit
  - 频率：2 GHz
  - 时间：0.5 ns

Step 2: FAB 1to4分割
  - 将128B Flit分成4个32B分片
  - 每个分片：256 bits

Step 3: UCIe SerDes串行化
  - 每条x16链路：每周期传输16 bits
  - 传输256 bits需要：256 ÷ 16 = 16个UCIe周期
  - UCIe频率：32 GHz
  - 时间：16 × 0.03125 ns = 0.5 ns
```

**接收端（UCIe → Die内部）**：

```
Step 1: 4条UCIe链路接收
  - 每条接收32B（256 bits）
  - 16个UCIe周期完成

Step 2: UCIe SerDes解串行化
  - 将串行数据转换为32B并行分片

Step 3: FAB 4to1聚合
  - 等待4个分片全部到达
  - 重组为完整的128B Flit

Step 4: STI 1024bit并行输出
  - 1个CrossRing周期注入网络
```

**关键观察**：

- **Die内并行传输**：STI总线1个周期（2GHz）传输完整Flit
- **UCIe串行传输**：4条x16链路16个周期（32GHz）传输完整Flit
- **延迟匹配**：0.5 ns（2GHz 1周期）= 0.5 ns（32GHz 16周期）
- **带宽匹配**：256 GB/s（STI理论） ≈ 252 GB/s（4×UCIe实际）

通过更高的UCIe物理频率（32GHz vs 2GHz），串行化开销被完全补偿，实现了与Die内部并行总线相当的传输延迟。

## 2. 数据单位与带宽计算

### 2.1 基本单位关系

#### 2.1.1 bit与Byte

| 单位 | 符号      | 定义             | 关系      |
| ---- | --------- | ---------------- | --------- |
| bit  | b（小写） | 二进制位（0或1） | 最小单位  |
| Byte | B（大写） | 8个bit           | 1 B = 8 b |

**记忆口诀**：

- **小写b** = bit，用于**速率**（bps, Gbps）
- **大写B** = Byte，用于**容量**（KB, GB）

#### 2.1.2 存储容量单位（大写B）

**十进制标准（SI）**：硬盘厂商常用

```
1 KB = 1,000 Bytes = 10³ Bytes
1 MB = 1,000 KB = 10⁶ Bytes
1 GB = 1,000 MB = 10⁹ Bytes
1 TB = 1,000 GB = 10¹² Bytes
```

**二进制标准（IEC）**：操作系统常用

```
1 KiB = 1,024 Bytes = 2¹⁰ Bytes
1 MiB = 1,024 KiB = 2²⁰ Bytes
1 GiB = 1,024 MiB = 2³⁰ Bytes
1 TiB = 1,024 GiB = 2⁴⁰ Bytes
```

**差异示例**：

```
500 GB硬盘 = 500 × 10⁹ Bytes
           = 500 × 10⁹ / 2³⁰ GiB
           ≈ 465.66 GiB

所以买的500GB硬盘，系统显示约465GiB
```

#### 2.1.3 传输速率单位（小写b）

**网络/链路带宽**：

```
1 Kbps = 1,000 bits/s
1 Mbps = 1,000 Kbps = 10⁶ bits/s
1 Gbps = 1,000 Mbps = 10⁹ bits/s
1 Tbps = 1,000 Gbps = 10¹² bits/s
```

**GT/s（GigaTransfers per second）**：

- 用于PCIe、UCIe等串行链路
- **不等同于Gbps**，需考虑编码方案

**编码开销示例**：

```
PCIe Gen3: 8 GT/s，使用8b/10b编码
实际数据率 = 8 GT/s × (8/10) = 6.4 Gbps

UCIe: 32 GT/s，使用128b/130b编码
实际数据率 = 32 GT/s × (128/130) ≈ 31.38 Gbps
```

### 2.2 关键转换公式

#### 2.2.1 Gbps ↔ GB/s

```
公式：GB/s = Gbps ÷ 8

示例：
100 Gbps = 100 ÷ 8 = 12.5 GB/s
512 Gbps = 512 ÷ 8 = 64 GB/s

反向：
64 GB/s = 64 × 8 = 512 Gbps
```

#### 2.2.2 GT/s → Gbps（含编码）

```
公式：Gbps = GT/s × (有效位/总位)

UCIe（128b/130b编码）：
32 GT/s = 32 × (128/130) ≈ 31.38 Gbps

PCIe Gen4（128b/130b编码）：
16 GT/s = 16 × (128/130) ≈ 15.75 Gbps

PCIe Gen3（8b/10b编码）：
8 GT/s = 8 × (8/10) = 6.4 Gbps
```

#### 2.2.3 多Lane聚合带宽

```
公式：总带宽 = 单Lane速率 × Lane数量 × 编码效率 ÷ 8

UCIe x16链路（32 GT/s）：
总带宽 = 32 GT/s × 16 lanes × (128/130) ÷ 8
       = 512 GT/s × 0.9846 ÷ 8
       ≈ 63.08 GB/s
```

### 2.3 UCIe带宽计算实例

#### 2.3.1 单通道x16带宽

**参数**：

- Lane速率：32 GT/s
- Lane数量：16
- 编码：128b/130b

**计算步骤**：

```
Step 1: 原始传输速率
  32 GT/s × 16 lanes = 512 GT/s

Step 2: 考虑编码开销
  有效数据率 = 512 GT/s × (128/130)
              = 512 × 0.9846
              ≈ 504.62 Gbps

Step 3: 转换为GB/s
  带宽 = 504.62 Gbps ÷ 8
       ≈ 63.08 GB/s
```

#### 2.3.2 四通道聚合带宽

**总带宽**：

```
4 × 63.08 GB/s = 252.3 GB/s
```

**考虑协议开销（10%）**：

```
实际可用带宽 = 252.3 GB/s × 0.9
              ≈ 227 GB/s
```

**与DDR5对比**：

```
DDR5-6400（单通道）：
  6400 MT/s × 8 Bytes = 51.2 GB/s

UCIe 4×x16通道 ≈ 4.4倍DDR5单通道带宽
```

### 2.4 CrossRing中的带宽配置

#### 2.4.1 AXI通道带宽参数

在CrossRing配置中的对应关系：

```yaml
# config/topologies/d2d_config.yaml

# AXI通道带宽限制（单位：GB/s）
D2D_AR_BANDWIDTH: 128    # 读地址通道：128 GB/s → 1024 Gbps
D2D_R_BANDWIDTH: 128     # 读数据通道：128 GB/s → 1024 Gbps
D2D_AW_BANDWIDTH: 128    # 写地址通道：128 GB/s → 1024 Gbps
D2D_W_BANDWIDTH: 128     # 写数据通道：128 GB/s → 1024 Gbps
D2D_B_BANDWIDTH: 32      # 写响应通道：32 GB/s → 256 Gbps
```

**配置解读**：

- R/W数据通道配置128 GB/s，对应UCIe 2×x16链路
- B响应通道配置32 GB/s（响应包小，带宽需求低）
- 总配置 ≈ 544 GB/s，接近四通道理论带宽

#### 2.4.2 Flit大小与传输速率

**CrossRing Flit配置**：

```python
FLIT_SIZE = 64  # Bytes
NETWORK_FREQUENCY = 2  # GHz
```

**每周期最大Flit数**（单通道x16）：

```
单通道带宽 = 63.08 GB/s
每周期带宽 = 63.08 GB/s ÷ 2 GHz = 31.54 Bytes/cycle

每周期Flit数 = 31.54 ÷ 64 ≈ 0.49 Flit/cycle
→ 约每2周期1个Flit
```

**四通道聚合**：

```
总带宽 = 227 GB/s
每周期带宽 = 227 ÷ 2 = 113.5 Bytes/cycle
每周期Flit数 = 113.5 ÷ 64 ≈ 1.77 Flit/cycle
→ 约每周期1-2个Flit
```

### 2.5 快速换算表

| 从          | 到   | 公式       | 示例                    |
| ----------- | ---- | ---------- | ----------------------- |
| Gbps        | GB/s | ÷ 8       | 100 Gbps = 12.5 GB/s    |
| GB/s        | Gbps | × 8       | 64 GB/s = 512 Gbps      |
| GT/s (UCIe) | Gbps | × 0.9846  | 32 GT/s ≈ 31.5 Gbps    |
| Gbps (x16)  | GB/s | × 16 ÷ 8 | 32 Gbps × 16 = 64 GB/s |
| GB          | GiB  | ÷ 1.074   | 500 GB ≈ 465 GiB       |
| GiB         | GB   | × 1.074   | 256 GiB ≈ 275 GB       |

**常见错误**：

```
❌ "我的网速是100MB/s"
   （应该是100Mbps，即12.5MB/s）

❌ "1GB = 1024MB"
   （十进制中是1000MB，二进制中1GiB=1024MiB）

❌ "32GT/s = 32Gbps"
   （需考虑编码，32GT/s ≈ 31.38Gbps）
```

## 3. CrossRing中的D2D实现映射

### 3.1 架构分层对应

#### 3.1.1 CrossRing实现的层次

```
┌─────────────────────────────────────────────┐
│  CrossRing当前实现                           │
├─────────────────────────────────────────────┤
│  应用层    │ IPInterface (GDMA/DDR/L2M)     │ ✓ 已实现
├────────────┼────────────────────────────────┤
│  D2D协议层 │ D2D_RN / D2D_SN                │ ✓ 已实现
├────────────┼────────────────────────────────┤
│  AXI通道层 │ D2D_Sys (5通道延迟+带宽控制)   │ ✓ 已实现
├────────────┼────────────────────────────────┤
│  物理层    │ [简化] 固定延迟模拟             │ ⚠ 简化实现
└────────────┴────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  真实UCIe微架构（未完全建模）                 │
├─────────────────────────────────────────────┤
│  UCIe物理层 │ 4×x16通道 + FAB聚合           │ ✗ 未建模
├─────────────┼────────────────────────────────┤
│  ASYNC时钟域│ 异步FIFO + 双时钟              │ ✗ 未建模
├─────────────┼────────────────────────────────┤
│  Cbuf缓冲   │ Credit流控 + 循环缓冲          │ ✗ 未建模
├─────────────┼────────────────────────────────┤
│  DMUX分流   │ 1to2流量分离                   │ ✗ 未建模
├─────────────┼────────────────────────────────┤
│  重排序     │ RB (Reorder Buffer)            │ ✗ 未建模
└─────────────┴────────────────────────────────┘
```

#### 3.1.2 模块对应关系

| UCIe微架构模块 | CrossRing实现          | 实现状态 | 备注                              |
| -------------- | ---------------------- | -------- | --------------------------------- |
| UCIe x16 ×4   | D2D_Sys.axi_channels   | 部分     | 建模5个AXI通道，未显式建模4条UCIe |
| FAB 4to1/1to4  | -                      | 未实现   | 当前假设单一逻辑通道              |
| RN模块         | D2D_RN_Interface       | 已实现   | 继承IPInterface，处理跨Die请求    |
| SN模块         | D2D_SN_Interface       | 已实现   | 继承IPInterface，处理跨Die响应    |
| ASYNC          | -                      | 未实现   | 使用固定延迟模拟，未建模时钟域    |
| Cbuf           | IPInterface.pre_buffer | 简化     | 缺少Credit流控机制                |
| DMUX 1to2      | -                      | 未实现   | 无显式流量分离逻辑                |
| STI总线        | CrossRing Network      | 已实现   | Die内网络接口                     |

### 3.2 D2D_Sys：AXI通道建模

#### 3.2.1 五通道架构

**文件**: `src/utils/components/d2d_sys.py`

```python
class D2D_Sys:
    def _init_axi_channels(self, config):
        """初始化AXI通道配置"""
        channels = {}
        channel_configs = [
            ("AR", "D2D_AR_LATENCY", 10, "D2D_AR_BANDWIDTH", 128),
            ("R",  "D2D_R_LATENCY",   8, "D2D_R_BANDWIDTH",  128),
            ("AW", "D2D_AW_LATENCY", 10, "D2D_AW_BANDWIDTH", 128),
            ("W",  "D2D_W_LATENCY",   2, "D2D_W_BANDWIDTH",  128),
            ("B",  "D2D_B_LATENCY",   8, "D2D_B_BANDWIDTH",   32),
        ]

        for name, latency_key, default_lat, bw_key, default_bw in channel_configs:
            channels[name] = {
                "name": name,
                "latency": getattr(config, latency_key, default_lat),  # cycles
                "bandwidth": getattr(config, bw_key, default_bw),      # GB/s
                "pending_queue": deque(),  # 等待发送队列
                "in_flight_queue": [],     # 飞行中队列 [(arrival_cycle, flit)]
                "token_bucket": TokenBucket(...)  # 带宽限制
            }

        return channels
```

**对应关系**：

- **AR/R通道** → UCIe通道#0/1（读流量）
- **AW/W通道** → UCIe通道#0（写流量）
- **B通道** → UCIe通道#3（控制/响应）

**当前简化**：

- 5个AXI通道共享同一个物理D2D_Sys实例
- 未显式建模4条UCIe物理链路的并行传输
- 通过独立的 `in_flight_queue`模拟通道独立性

#### 3.2.2 通道延迟模拟

**发送流程**（类似Network.send_flits）：

```python
def send_to_axi_channel(self, flit: Flit, channel_name: str):
    """将flit发送到AXI通道"""
    channel = self.axi_channels[channel_name]

    # 1. 检查带宽限制
    if not channel["token_bucket"].consume(1):
        self.axi_channel_stats[channel_name]["throttled"] += 1
        return False

    # 2. 添加到飞行中队列
    arrival_cycle = self.current_cycle + channel["latency"]
    heapq.heappush(channel["in_flight_queue"], (arrival_cycle, flit))

    # 3. 统计
    self.axi_channel_stats[channel_name]["injected"] += 1

    return True
```

**接收流程**：

```python
def process_axi_channels(self):
    """处理所有AXI通道的到达flit"""
    for channel_name, channel in self.axi_channels.items():
        in_flight = channel["in_flight_queue"]

        while in_flight and in_flight[0][0] <= self.current_cycle:
            arrival_cycle, flit = heapq.heappop(in_flight)

            # 交付到目标Die的接口
            self.deliver_to_target_die(flit, channel_name)

            # 统计
            self.axi_channel_stats[channel_name]["ejected"] += 1
```

**与UCIe物理层的对应**：

- `latency`: 模拟UCIe链路传输延迟 + SerDes延迟
- `in_flight_queue`: 模拟正在UCIe物理链路上传输的数据
- `token_bucket`: 模拟UCIe链路带宽限制

### 3.3 D2D_RN_Interface：请求发起端

#### 3.3.1 核心职责

**文件**: `src/utils/components/d2d_rn_interface.py`

```python
class D2D_RN_Interface(IPInterface):
    """Die间请求节点"""

    def __init__(self, ip_type, ip_pos, config, ...):
        super().__init__(ip_type, ip_pos, config, ...)

        # D2D特有属性
        self.die_id = config.DIE_ID
        self.cross_die_receive_queue = []  # heapq管理
        self.target_die_interfaces = {}    # {die_id: d2d_sn}

        # D2D延迟配置（已转换为cycles）
        self.d2d_ar_latency = config.D2D_AR_LATENCY
        self.d2d_r_latency = config.D2D_R_LATENCY
        # ...
```

**对应UCIe架构中的RN模块**：

- 管理跨Die事务的发起
- 处理AXI协议转换
- 维护Tracker资源

#### 3.3.2 跨Die请求发送

**读请求发送流程**：

```python
def handle_cross_die_read_request(self, flit: Flit):
    """处理跨Die读请求"""
    target_die = flit.d2d_target_die
    target_node = flit.d2d_target_node

    # 1. 分配RN tracker资源
    if self.rn_tracker_count["read"]["count"] <= 0:
        # 资源不足，应该有背压机制（当前未完整实现）
        return False

    self.rn_tracker_count["read"]["count"] -= 1

    # 2. 通过D2D_Sys发送（添加AR通道延迟）
    if self.d2d_sys:
        success = self.d2d_sys.send_to_axi_channel(flit, "AR")
        if success:
            self.cross_die_requests_sent += 1
            return True

    # 3. 如果发送失败，释放tracker
    self.rn_tracker_count["read"]["count"] += 1
    return False
```

**对应UCIe流程**：

```
GDMA读请求 → D2D_RN检查资源 → 发送到AXI AR通道
→ D2D_Sys添加延迟 → UCIe传输 → 目标Die的D2D_SN接收
```

#### 3.3.3 跨Die响应接收

**接收调度**（由D2D_Sys调用）：

```python
def schedule_cross_die_receive(self, flit: Flit, arrival_cycle: int):
    """调度跨Die接收 - 由D2D_Sys调用"""
    heapq.heappush(self.cross_die_receive_queue, (arrival_cycle, flit))
    self.cross_die_requests_received += 1

def process_cross_die_receives(self):
    """处理到期的跨Die接收 - 每周期调用"""
    while (self.cross_die_receive_queue and
           self.cross_die_receive_queue[0][0] <= self.current_cycle):
        arrival_cycle, flit = heapq.heappop(self.cross_die_receive_queue)
        self.handle_received_cross_die_flit(flit)
```

**对应UCIe架构**：

- `cross_die_receive_queue`: 模拟RN模块的接收缓冲（类似Cbuf）
- `arrival_cycle`: 包含AXI R通道延迟 + UCIe传输延迟
- `handle_received_cross_die_flit`: 协议转换，交付给本地网络

### 3.4 D2D_SN_Interface：响应处理端

#### 3.4.1 核心职责

**文件**: `src/utils/components/d2d_sn_interface.py`

```python
class D2D_SN_Interface(IPInterface):
    """Die间响应节点"""

    def handle_local_cross_die_read_request(self, flit: Flit):
        """处理本地发起的跨Die读请求"""
        target_die = flit.d2d_target_die

        # 1. 检查D2D_SN的RO tracker资源
        has_tracker = self.sn_tracker_count["ro"]["count"] > 0

        if has_tracker:
            # 分配tracker
            self.sn_tracker_count["ro"]["count"] -= 1
            flit.sn_tracker_type = "ro"
            self.sn_tracker.append(flit)

            # 2. 转发到目标Die（通过D2D_Sys的AR通道）
            if self.d2d_sys:
                self.d2d_sys.send_to_axi_channel(flit, "AR")
        else:
            # 资源不足，返回negative响应
            negative_rsp = self.create_response_flit(flit, "negative")
            self.enqueue(negative_rsp, "rsp")

            # 加入等待队列（retry机制）
            self.sn_req_wait["read"].append(flit)
```

**对应UCIe架构中的SN模块**：

- 接收跨Die请求，进行资源检查
- 转发到本Die内的目标节点
- 收集响应数据，返回给源Die

#### 3.4.2 跨Die数据返回

**读数据返回流程**：

```python
def handle_cross_die_read_response(self, data_flit: Flit):
    """处理跨Die读响应数据"""
    # 1. 从本Die的DDR/L2M接收到读数据
    # 2. 需要返回到源Die（d2d_origin_die）

    origin_die = data_flit.d2d_origin_die
    origin_node = data_flit.d2d_origin_node

    # 3. 通过D2D_Sys发送（添加R通道延迟）
    if self.d2d_sys:
        self.d2d_sys.send_to_axi_channel(data_flit, "R")

    # 4. 释放tracker（读数据全部发送后）
    if self.is_last_data_flit(data_flit):
        self.release_completed_sn_tracker(data_flit)
```

**6阶段流程中的位置**：

- **阶段4→5**: D2D_SN接收本Die内返回的数据
- **阶段5**: 通过AXI R通道发送回源Die
- **阶段6**: 源Die的D2D_SN转发给原始请求者

### 3.5 当前实现的数据流路径

#### 3.5.1 读请求完整流程

```
Die 0 (源)                        Die 1 (目标)
─────────────────────────────────────────────────
GDMA (14.g0)
  │ enqueue(req)
  ▼
D2D_SN (33.ds)
  │ 资源检查 → 分配tracker
  │ handle_local_cross_die_read_request()
  ▼
D2D_Sys
  │ send_to_axi_channel(flit, "AR")
  │ 添加AR延迟（5ns → 10 cycles）
  │ in_flight_queue
  ▼
[UCIe传输] ─────────────────────────►
                                    D2D_RN (4.dr)
                                      │ schedule_cross_die_receive()
                                      │ process_cross_die_receives()
                                      ▼
                                    转发到内部网络
                                      │ destination = d2d_target_node
                                      ▼
                                    DDR (4.dr)
                                      │ 处理读请求
                                      │ 返回数据
                                      ▼
                                    D2D_RN (4.dr)
                                      │ 接收数据
D2D_SN (33.ds)           ◄──────────  │ send_to_axi_channel(data, "R")
  │ process_cross_die_receives()      │ 添加R延迟（4ns → 8 cycles）
  │ 转发给原始请求者                   ▼
  ▼                              [UCIe传输]
GDMA (14.g0)
  │ 接收数据
  │ 完成事务
  ▼
```

#### 3.5.2 写请求7阶段流程

```
阶段1: GDMA → D2D_SN（Die 0内部握手）
  - D2D_SN检查share tracker + WDB资源
  - 分配成功 → 返回datasend响应
  - GDMA发送写数据到D2D_SN

阶段2-3: 跨Die传输（AW + W通道）
  - D2D_Sys.send_to_axi_channel(req, "AW")   # 地址
  - D2D_Sys.send_to_axi_channel(data, "W")   # 数据
  - 添加AW延迟（5ns）和W延迟（1ns）

阶段4: Die 1内部处理
  - D2D_RN接收写请求和数据
  - 转发到DDR/L2M
  - DDR写入完成

阶段5: 跨Die响应返回（B通道）
  - D2D_RN发送write_complete响应
  - D2D_Sys.send_to_axi_channel(rsp, "B")
  - 添加B延迟（4ns）

阶段6-7: Die 0响应转发
  - D2D_SN接收B响应
  - 释放tracker + WDB
  - 转发给GDMA
```

### 3.6 缺失的微架构模块分析

#### 3.6.1 四通道并行传输

**真实UCIe**：

- 4条物理x16链路**协同传输**每一个Flit
- 通过FAB层将Flit分割成4个分片并行传输
- 所有AXI通道的Flit都使用相同的4条UCIe链路

**CrossRing当前**：

- 5个AXI通道通过同一个D2D_Sys实例
- 各通道有独立的 `in_flight_queue`建模AXI通道独立性
- 未显式建模4条UCIe物理链路的Flit分割和并行传输

**改进方向**：

1. **显式建模4条UCIe物理链路**

   - 每条链路独立的SerDes模块
   - 每条链路负责传输Flit的固定部分（1/4）
   - 4条链路的传输状态独立跟踪
2. **实现FAB分割/聚合逻辑**

   - FAB 1to4：将128B Flit分割成4个32B分片
   - 为每个分片添加序列号和校验
   - 同步将4个分片发送到对应UCIe链路
3. **带宽建模精细化**

   - 当前：总带宽限制在D2D_Sys层
   - 改进：每条UCIe链路独立的带宽限制（~63 GB/s）
   - 聚合：4条链路总带宽~252 GB/s

#### 3.6.2 FAB聚合/分流逻辑

**真实UCIe**：

- FAB 4to1：从4条UCIe接收分片，重组为完整Flit
- FAB 1to4：将完整Flit分割成4个分片，发送到4条UCIe

**CrossRing当前**：

- 未建模Flit分割/重组逻辑
- 假设Flit完整传输，未考虑物理层的分片

**改进方向**：

1. **FAB 1to4分割器**

   - 输入：128B完整Flit
   - 输出：4个32B分片（带序列号）
   - 同步：确保4个分片同时发送
2. **FAB 4to1聚合器**

   - 输入：从4条UCIe链路并行接收分片
   - 缓冲：等待同一Flit的4个分片全部到达
   - 重组：按序拼接成完整的128B Flit
   - 校验：验证分片完整性和顺序
3. **错误处理**

   - 分片丢失检测（超时机制）
   - 分片乱序重排（基于序列号）
   - 校验失败重传

#### 3.6.3 ASYNC时钟域转换

**真实UCIe**：

- Die 0时钟（2 GHz） ≠ Die 1时钟（2.5 GHz） ≠ UCIe时钟（3.2 GHz）
- 异步FIFO处理跨时钟域数据传递
- 引入额外延迟（2-8周期）

**CrossRing当前**：

- 假设所有Die使用相同时钟频率
- 未建模时钟域转换延迟

**改进方向**：

1. **异步FIFO实现**

   - 双时钟FIFO：独立的写时钟和读时钟
   - 格雷码指针同步：消除亚稳态
   - 深度配置：通常16-32 entries
2. **延迟建模**

   - 双级同步器延迟：2个目标时钟周期
   - 指针同步延迟：1-2个周期
   - 总延迟：3-4个目标时钟周期（可配置）
3. **时钟转换逻辑**

   - 源时钟域周期→目标时钟域周期的转换
   - 考虑频率比和相位关系
   - 支持可配置的Die时钟和UCIe时钟

#### 3.6.4 Cbuf + Credit流控

**真实UCIe**：

- 循环缓冲区提供弹性缓冲
- Credit-based流控实现背压
- 侧边通道传递Credit返回

**CrossRing当前**：

- 使用 `pre_buffer`/`post_buffer`简单队列
- 缺少显式Credit管理机制
- 无Credit返回通道建模

**改进方向**：

1. **Credit-based流控机制**

   - 发送端维护remote_credits计数器（对方缓冲可用空间）
   - 接收端每接收一个Flit消耗1个local_credit
   - 接收端通过Credit返回消息更新发送端的remote_credits
2. **Cbuf循环缓冲设计**

   - 固定深度的FIFO结构（通常64-128 entries）
   - 读写指针管理，支持环形访问
   - 满/空状态检测
3. **Credit返回通道**

   - 独立的小消息通道（不占用主数据带宽）
   - 周期性批量返回Credit（减少开销）
   - 低延迟传递（优先级高于数据）
4. **与现有设计集成**

   - 在D2D_RN_Interface和D2D_SN_Interface中添加Cbuf模块
   - 扩展AXI通道支持Credit返回消息
   - 配置Cbuf大小对应真实硬件缓冲深度

## 4. 实现状态与改进方向

### 4.1 已实现功能总结

#### 4.1.1 核心功能 ✓

| 功能模块    | 实现状态 | 文件位置                   | 说明                    |
| ----------- | -------- | -------------------------- | ----------------------- |
| D2D_RN接口  | ✓ 完整  | `d2d_rn_interface.py`    | 跨Die请求发起           |
| D2D_SN接口  | ✓ 完整  | `d2d_sn_interface.py`    | 跨Die请求接收           |
| AXI 5通道   | ✓ 完整  | `d2d_sys.py`             | AR/R/AW/W/B通道延迟模拟 |
| 带宽控制    | ✓ 完整  | `d2d_sys.py`             | 令牌桶限速              |
| Tracker管理 | ✓ 完整  | `d2d_rn/sn_interface.py` | 读/写Tracker资源管理    |
| Retry机制   | ✓ 完整  | `d2d_sn_interface.py`    | negative/positive响应   |
| 6阶段读流程 | ✓ 完整  | 跨多个文件                 | GDMA→DDR读事务         |
| 7阶段写流程 | ✓ 完整  | 跨多个文件                 | GDMA→DDR写事务         |
| 多Die支持   | ✓ 完整  | `d2d_model.py`           | 2-Die和4-Die配置        |
| 配置驱动    | ✓ 完整  | `config/d2d_config.py`   | YAML配置Die连接         |

#### 4.1.2 性能建模 ✓

| 性能指标    | 建模状态 | 精度评估             |
| ----------- | -------- | -------------------- |
| 端到端延迟  | ✓ 准确  | 包含所有AXI通道延迟  |
| 带宽利用率  | ✓ 准确  | 令牌桶限速 + 统计    |
| Tracker占用 | ✓ 准确  | 实时跟踪资源使用     |
| 通道拥塞    | ✓ 基本  | 带宽限制导致的阻塞   |
| 吞吐量      | ✓ 准确  | 每周期传输Flit数统计 |

### 4.2 部分实现功能

#### 4.2.1 物理层建模 ⚠

**当前状态**：

- ✓ AXI通道独立延迟
- ✓ 各通道独立带宽限制
- ✗ 未显式建模4条UCIe物理链路
- ✗ 未建模FAB聚合/分流

**精度影响**：

- **中等**: 对总体性能影响有限
- 当前假设AXI通道之间完全独立，实际上共享物理资源

**改进优先级**: 中（适用于精细化建模场景）

#### 4.2.2 时钟域转换 ⚠

**当前状态**：

- ✓ 固定延迟模拟
- ✗ 未建模异步FIFO
- ✗ 假设所有Die同频

**精度影响**：

- **低**: 大多数场景下Die确实同频
- 异构Die场景会有偏差

**改进优先级**: 低（除非研究异构Die）

### 4.3 未实现功能

#### 4.3.1 高级流控机制 ✗

**缺失功能**：

- Credit-based流控的显式建模
- Credit返回侧边通道
- 分层缓冲（Cbuf）

**当前替代方案**：

- 使用带宽限制 + 缓冲队列模拟流控效果
- 简化的背压机制（队列满则阻塞）

**精度影响**：

- **低-中**: 对稳态性能影响小
- 对瞬态行为（突发流量）建模不够精确

**改进优先级**: 中（提升瞬态建模精度）

### 4.4 改进路线图

#### 4.4.1 短期改进（3个月）

**目标**: 提升物理层建模精度

1. **显式建模4条UCIe链路**

   - 实现 `UCIe_Physical_Link`类
   - 支持多链路并行传输
   - AXI通道到UCIe链路的映射配置
2. **添加FAB聚合/分流逻辑**

   - 实现 `FAB_Aggregator`（4to1）
   - 实现 `FAB_Demux`（1to4）
   - 可配置的仲裁策略
3. **Credit流控机制**

   - 实现 `Cbuf_with_Credit`
   - 添加Credit返回侧边通道
   - 集成到D2D_RN/SN接口

**预期收益**：

- 物理层建模精度提升20%
- 支持更复杂的流量模式分析
- 与真实UCIe硬件行为更接近

#### 4.4.2 中期改进（6个月）

**目标**: 支持异构Die场景

1. **ASYNC时钟域转换**

   - 实现 `AsyncFIFO`类
   - 支持不同Die运行在不同频率
   - 动态计算跨时钟域延迟
2. **增强统计功能**

   - UCIe链路级别的利用率统计
   - 各通道的拥塞分析
   - Credit流控效率分析

**预期收益**：

- 支持异构多Die芯片建模
- 更精细的性能瓶颈分析

#### 4.4.3 长期改进（12个月）

**目标**: 完整UCIe微架构建模

1. **SerDes物理层**

   - 模拟串行化/解串行化延迟
   - 编码开销（128b/130b）建模
   - 信号完整性影响（可选）
2. **链路训练与状态管理**

   - L0/L1/L2功耗状态
   - 状态转换延迟
   - 功耗建模
3. **错误处理与重传**

   - CRC错误检测
   - Retry Buffer
   - ACK/NACK协议

**预期收益**：

- 接近RTL级别的建模精度
- 支持功耗分析
- 支持可靠性分析

### 4.5 配置示例：启用高级特性

#### 4.5.1 未来配置文件结构

```yaml
# config/topologies/d2d_advanced_config.yaml

# D2D基础配置
D2D_ENABLED: true
NUM_DIES: 2

# UCIe物理层配置
D2D_UCIE_PHY:
  enabled: true
  num_physical_links: 4
  link_configs:
    - link_id: 0
      lanes: 16
      speed: 32  # GT/s
      encoding: "128b130b"
      bandwidth: 63  # GB/s
    - link_id: 1
      lanes: 16
      speed: 32
      encoding: "128b130b"
      bandwidth: 63
    - link_id: 2
      lanes: 16
      speed: 32
      encoding: "128b130b"
      bandwidth: 63
    - link_id: 3
      lanes: 16
      speed: 16  # 侧边通道速率低
      encoding: "128b130b"
      bandwidth: 32

# FAB聚合/分流配置
D2D_FAB:
  enabled: true
  aggregator:
    type: "weighted_round_robin"
    weights: [1.0, 1.0, 1.0, 0.5]  # 侧边通道权重低
  demux:
    strategy: "traffic_type"  # "traffic_type" | "load_balance" | "hash"

# ASYNC时钟域配置
D2D_ASYNC:
  enabled: true
  die_clocks:
    0: 2.0  # GHz
    1: 2.5  # GHz
  ucie_clock: 3.2  # GHz
  fifo_depth: 16
  sync_delay: 4  # cycles

# Cbuf + Credit流控
D2D_CBUF:
  enabled: true
  size: 64
  credit_based: true
  credit_return_channel: 3  # 使用UCIe链路#3


# 统计与调试
D2D_STATISTICS:
  collect_link_utilization: true
  collect_channel_stats: true
  collect_credit_stats: true
```

#### 4.5.2 使用示例

```python
# scripts/d2d_advanced_simulation.py

from config.d2d_config import D2DConfig
from src.core.d2d_model import D2D_Model

# 加载高级配置
config = D2DConfig(
    die_config_file="config/topologies/kcin_5x4.yaml",
    d2d_config_file="config/topologies/d2d_advanced_config.yaml"
)

# 创建D2D模型（自动启用高级特性）
model = D2D_Model(config, traffic_config)

# 运行仿真
results = model.run_simulation()

# 分析UCIe链路级别统计
print("UCIe Link Utilization:")
for link_id, stats in results["ucie_link_stats"].items():
    print(f"  Link {link_id}: {stats['utilization']:.2%}")
    print(f"    Bandwidth: {stats['avg_bandwidth']:.2f} GB/s")
    print(f"    Congestion: {stats['congestion_cycles']} cycles")

# 分析Credit流控效率
print("\nCredit Flow Control:")
print(f"  Credit stalls: {results['credit_stats']['stall_cycles']} cycles")
print(f"  Credit efficiency: {results['credit_stats']['efficiency']:.2%}")
```

## 附录

### 参考文献

1. **UCIe规范**

   - UCIe Consortium, "UCIe Specification 1.0", 2022
   - https://www.uciexpress.org/
2. **AXI协议**

   - ARM, "AMBA AXI Protocol Specification", Version 5.0
3. **CrossRing项目文档**

   - `docs/D2D_Design.md` - D2D基础设计
   - `docs/D2D_4Die_Design.md` - 4-Die扩展设计
4. **相关论文**

   - "Chiplet-based Design for High-Performance Computing", ISSCC 2023
   - "UCIe: Enabling Ubiquitous Chiplet Interconnect", HotChips 2022

---

**文档版本**: 1.0
**最后更新**: 2025-10-29
**作者**: CrossRing开发团队
