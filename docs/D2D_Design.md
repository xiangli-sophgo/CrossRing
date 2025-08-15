# D2D（Die-to-Die）建模仿真设计文档

## 1. 系统架构概述

### 1.1 整体架构

D2D系统包含两个独立的Die，每个Die内部包含完整的CrossRing网络拓扑。两个Die之间通过AXI接口进行通信，实现跨Die的读写操作。

```
┌─────────────── Die 0 ────────────────┐    ┌───────────────  Die 1 ────────────────┐
│  ┌───┐  ┌───┐      ┌───┐  ┌───┐      │    │  ┌───┐  ┌───┐      ┌───┐  ┌───┐       │
│  │RN │  │SN │ .... │RN │  │SN │      │    │  │RN │  │SN │ .... │RN │  │SN │       │
│  └───┘  └───┘      └─┬─┘  └─┬─┘      │    │  └─┬─┘  └─┬─┘      └───┘  └───┘       │
│           │          │      │        │    │    │      │           │               │
│       CrossRing Network     │        │    │    │  CrossRing Network               │
│                         D2D_SN       │◄──AXI──►│ D2D_RN                           │
└──────────────────────────────┬───────┘    └────┬──────────────────────────────────┘
                            AXI Interface      AXI Interface
```

### 1.2 关键组件

1. **D2D_RN (Request Node)**: 专门处理跨Die请求的RN节点
2. **D2D_SN (Slave Node)**: 专门处理跨Die响应的SN节点  
3. **AXI Interface**: 模拟5个AXI通道的跨Die连接
4. **CrossRing Network**: 现有的片内网络架构（保持不变）

## 2. D2D读流程详细分析

### 2.1 读流程概述

基于提供的流程图，D2D读操作包含以下6个步骤：

1. **Die 0 RN → Die 0 SN**: 发送读请求 (Read Req)
2. **Die 0 SN → Die 1 RN**: 通过AXI AR通道转发读请求
3. **Die 1 RN → Die 1 SN**: 在Die 1内部转发读请求
4. **Die 1 SN → Die 1 RN**: Die 1内部返回读数据 (Read Data)
5. **Die 1 RN → Die 0 SN**: 通过AXI R通道返回读数据
6. **Die 0 SN → Die 0 RN**: 完成读数据传输

### 2.2 详细时序流程

#### 步骤1: Die 0内部读请求发起
```
时刻T0: RN[Die0] 生成读请求
  - packet_id: 全局唯一ID
  - source_die_id: 0
  - source_node_id: Die0_RN_pos (例如: 5)
  - target_die_id: 1
  - target_node_id: Die1_SN_pos (例如: 10)
  - destination: Die0_D2D_SN_pos (路由到D2D网关)
  - req_type: "read"
  - burst_length: 数据长度
```

#### 步骤2: 跨Die传输（AR通道延迟）
```
时刻T1: Die0_D2D_SN 接收读请求后
  - 检查target_die_id != 0，确认为跨Die请求
  - 直接传输flit到Die1，无需协议转换
  - 添加AR通道延迟: T1 + D2D_AR_LATENCY
  - flit在时刻(T1 + D2D_AR_LATENCY)到达Die1_D2D_RN
  - 保持所有原始信息（packet_id, source_die_id, target_node_id等）
```

#### 步骤3: Die 1内部请求转发
```
时刻T2: Die1_D2D_RN 接收AXI AR后
  - 解析AXI信号，重构读请求
  - source: Die1_D2D_RN_pos
  - destination: target_node_id (Die1内部目标SN节点ID)
  - source_die_id: 0 (保持原始信息)
  - source_node_id: 5 (保持原始信息)
  - target_die_id: 1 (当前Die)
  - target_node_id: 10 (目标SN节点)
  - 通过Die1内部网络转发到节点10
```

#### 步骤4: Die 1内部数据返回
```
时刻T3: Die1_SN 处理读请求
  - 访问本地存储/缓存
  - 生成读数据包
  - 返回到Die1_D2D_RN
```

#### 步骤5: 跨Die返回（R通道延迟）
```
时刻T4: Die1_D2D_RN 接收读数据后
  - 直接传输数据flit到Die0，无需协议转换
  - 添加R通道延迟: T4 + D2D_R_LATENCY
  - 数据在时刻(T4 + D2D_R_LATENCY)到达Die0_D2D_SN
  - 保持所有原始信息和数据内容
```

#### 步骤6: Die 0内部数据完成
```
时刻T5: Die0_D2D_SN 接收跨Die返回数据
  - 数据包保持原始packet_id和所有信息
  - destination: source_node_id (原始源节点5)
  - 转发到Die0内部节点5的RN
  - 完成读事务
```

## 3. D2D写流程详细分析

### 3.1 写流程概述

写流程更加复杂，包含3个AXI通道的协调：

1. **写地址阶段**: AW通道传输写地址
2. **写数据阶段**: W通道传输写数据
3. **写响应阶段**: B通道传输写完成响应

### 3.2 详细时序流程

#### 阶段1: 写请求发起
```
Die0_RN → Die0_D2D_SN: 
  - Write Req (包含地址和burst信息)
Die0_RN → Die0_D2D_SN: 
  - Write Data (实际数据负载)
```

#### 阶段2: 跨Die传输（AW + W通道延迟）
```
Die0_D2D_SN 发送写请求和数据:
  - Write Req: 添加AW通道延迟 (D2D_AW_LATENCY)
  - Write Data: 添加W通道延迟 (D2D_W_LATENCY)  
→ Die1_D2D_RN 按延迟时间接收并转发到Die1内部网络
```

#### 阶段3: 写响应处理（B通道延迟）
```
Die1_SN 完成写操作后:
  - 生成写完成响应
  - 经Die1_D2D_RN添加B通道延迟 (D2D_B_LATENCY)
  - Die0_D2D_SN按延迟时间接收后转发给Die0_RN
```

#### 特殊处理：DBIDValid信号（简化）
```
DBIDValid信号简化处理:
  - Die1处理写请求时生成DBIDValid
  - 添加固定延迟 (D2D_DBID_LATENCY)
  - 无需专用信号线，通过延迟队列模拟
```

## 4. 跨Die传输机制（延迟仿真）

### 4.1 传输通道定义

| 通道类型 | 方向 | 用途 | 延迟模拟 |
|---------|------|------|----------|
| AR | Die0→Die1 | 读请求传输 | D2D_AR_LATENCY cycles |
| R | Die1→Die0 | 读数据传输 | D2D_R_LATENCY cycles |
| AW | Die0→Die1 | 写地址传输 | D2D_AW_LATENCY cycles |
| W | Die0→Die1 | 写数据传输 | D2D_W_LATENCY cycles |
| B | Die1→Die0 | 写响应传输 | D2D_B_LATENCY cycles |

### 4.2 简化的Die识别机制

```python
# 简化的Die判断逻辑
def is_cross_die_request(flit, current_die_id):
    """检查是否为跨Die请求"""
    return flit.target_die_id != current_die_id

def get_target_die(flit):
    """获取目标Die ID"""
    return flit.target_die_id

# 示例请求结构
class D2D_Request:
    source_die_id: int      # 源Die ID (0 或 1)
    source_node_id: int     # 源Die内的节点ID
    target_die_id: int      # 目标Die ID (0 或 1)
    target_node_id: int     # 目标Die内的节点ID
    packet_id: int          # 事务ID
    req_type: str          # "read" 或 "write"
    burst_length: int      # 数据长度

# 简化的跨Die传输机制
class D2D_Interface:
    def __init__(self, die_id):
        self.die_id = die_id
        self.send_queue = deque()
        self.recv_queue = deque()  # (flit, arrival_time)
        
    def transfer_to_remote_die(self, flit, channel_type):
        """跨Die传输：仅添加延迟，不需要协议转换"""
        delay = self.get_channel_delay(channel_type)
        arrival_time = current_cycle + delay
        remote_die.recv_queue.append((flit, arrival_time))
        
    def get_channel_delay(self, channel_type):
        """获取通道延迟"""
        delays = {
            "AR": self.config.D2D_AR_LATENCY,
            "R": self.config.D2D_R_LATENCY,
            "AW": self.config.D2D_AW_LATENCY,
            "W": self.config.D2D_W_LATENCY,
            "B": self.config.D2D_B_LATENCY
        }
        return delays.get(channel_type, 0)
```

## 5. 性能参数定义

### 5.1 延迟参数

```json
{
  "D2D_AR_LATENCY": 10,     // AXI AR通道延迟（周期）
  "D2D_R_LATENCY": 8,       // AXI R通道延迟（周期）
  "D2D_AW_LATENCY": 10,     // AXI AW通道延迟（周期）
  "D2D_W_LATENCY": 2,       // AXI W通道延迟（周期）
  "D2D_B_LATENCY": 8,       // AXI B通道延迟（周期）
  "D2D_DBID_LATENCY": 5     // DBIDValid信号延迟（周期）
}
```

### 5.2 带宽限制

```json
{
  "D2D_AXI_DATA_WIDTH": 512, // AXI数据位宽
  "D2D_MAX_OUTSTANDING": 16, // 最大未完成事务数
  "D2D_AR_BANDWIDTH": 64,    // AR通道带宽限制（GB/s）
  "D2D_R_BANDWIDTH": 128,    // R通道带宽限制（GB/s）
  "D2D_AW_BANDWIDTH": 64,    // AW通道带宽限制（GB/s）
  "D2D_W_BANDWIDTH": 128,    // W通道带宽限制（GB/s）
  "D2D_B_BANDWIDTH": 32      // B通道带宽限制（GB/s）
}
```

### 5.3 缓冲区配置

```json
{
  "D2D_RN_BUFFER_SIZE": 64,    // D2D RN缓冲区大小
  "D2D_SN_BUFFER_SIZE": 64,    // D2D SN缓冲区大小
  "D2D_AXI_BUFFER_DEPTH": 16   // AXI接口缓冲区深度
}
```

## 6. 关键设计决策

### 6.1 组件复用策略
- **最大化复用**: Die内部网络完全复用现有CrossRing架构
- **最小化修改**: 仅新增D2D专用的RN和SN节点
- **模块化设计**: D2D功能作为独立模块，不影响现有功能

### 6.2 简化实现原则
- **基础功能优先**: 先实现基本的D2D读写流程
- **延迟模拟**: 使用简单的延迟队列模拟跨Die传输
- **最小复杂度**: 避免过度设计，保持实现简洁

## 7. 实现路线图

### 第一阶段：基础框架
1. 实现D2D_RN和D2D_SN基类
2. 创建跨Die延迟模拟器
3. 扩展配置系统支持D2D参数

### 第二阶段：读流程实现
1. 实现6步读流程的完整时序
2. 添加Die ID和节点ID路由逻辑
3. 集成基本延迟模拟

### 第三阶段：测试验证
1. 创建D2D专用测试用例
2. 基本功能验证和调试
3. 读流程性能测试

## 8. 使用示例

### 8.1 配置文件示例
```json
{
  "d2d_enabled": true,
  "num_dies": 2,
  "d2d_rn_positions": [35, 36],      // 每个Die的D2D RN节点位置
  "d2d_sn_positions": [37, 38],      // 每个Die的D2D SN节点位置  
  "d2d_latencies": {
    "ar": 10, "r": 8, "aw": 10, "w": 2, "b": 8
  },
  "d2d_node_mapping": {
    // 定义各Die内哪些节点可以作为D2D的源和目标
    "die_0": {
      "rn_nodes": [0, 1, 2, 5, 6, 7],      // Die0中的RN节点
      "sn_nodes": [10, 11, 12, 15, 16, 17]  // Die0中的SN节点
    },
    "die_1": {
      "rn_nodes": [0, 1, 2, 5, 6, 7],      // Die1中的RN节点
      "sn_nodes": [10, 11, 12, 15, 16, 17]  // Die1中的SN节点
    }
  }
}
```

### 8.2 Traffic文件格式
```
# D2D读写请求示例（完整格式）
# cycle, src_die, src_node, dst_die, dst_node, req_type, burst_length
# 说明: src_die.src_node → dst_die.dst_node
100, 0, 5, 1, 10, read, 4      # Die0节点5读取Die1节点10
200, 0, 6, 1, 12, write, 8     # Die0节点6写入Die1节点12  
300, 1, 3, 0, 7, read, 2       # Die1节点3读取Die0节点7
400, 1, 15, 0, 20, write, 4    # Die1节点15写入Die0节点20

# 节点编号说明：
# 每个Die内的节点按CrossRing拓扑编号（0 ~ NUM_NODE-1）
# 例如：8x9拓扑中，节点编号范围是0-71
```

## 9. 详细实现计划

### 9.1 架构理解要点

#### 核心概念修正
基于对现有系统的深入理解，D2D架构的关键要点：

1. **D2D_RN和D2D_SN是特殊IP类型**
   - 类似GDMA、DDR等IP，挂载在具体的节点位置上
   - 继承现有的IPInterface类，复用所有现有功能
   - 通过现有的CrossRing网络进行Die内通信

2. **每个Die是完整的BaseModel实例**
   - Die 0 = BaseModel实例，包含完整的CrossRing网络
   - Die 1 = BaseModel实例，包含完整的CrossRing网络
   - 每个Die独立运行，有自己的IP模块和网络状态

3. **D2D_Model作为多Die协调器**
   - 管理多个Die实例的生命周期
   - 协调跨Die的时钟同步
   - 不参与具体的数据传输，只做调度协调

4. **跨Die传输机制**
   - 在D2D_RN和D2D_SN内部实现延迟队列
   - 不需要全局的跨Die队列
   - 每个D2D组件管理自己的跨Die传输
