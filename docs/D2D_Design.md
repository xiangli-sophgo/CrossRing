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

写流程包含7个阶段，涉及3个AXI通道的协调和正确的tracker管理：

1. **Die0内部握手**: RN与D2D_SN之间的写请求/数据握手
2. **跨Die地址传输**: AW通道传输写地址
3. **跨Die数据传输**: W通道传输写数据
4. **Die1内部处理**: D2D_RN与SN之间的写请求/数据处理
5. **跨Die响应返回**: B通道传输写完成响应
6. **Die0响应转发**: D2D_SN转发响应给原始RN
7. **事务完成**: 释放所有资源

### 3.2 详细时序流程与Tracker管理

#### 阶段1: Die0内部写请求握手
```
时刻T0: Die0_RN 生成写请求
  - Die0_RN消耗tracker和databuffer资源
  - 发送写请求到Die0_D2D_SN

时刻T1: Die0_D2D_SN 处理写请求  
  - 检查D2D_SN自身资源（tracker/databuffer）
  - 如果有资源，消耗D2D_SN的tracker
  - 返回data_send响应给Die0_RN

时刻T2: Die0_RN 发送写数据
  - 收到data_send响应，开始发送写数据
  - 关键：保持tracker，等待最终B通道响应
  - Die0_D2D_SN接收并缓存写数据
```

#### 阶段2: 跨Die地址传输（AW通道）
```
时刻T3: Die0_D2D_SN → Die1_D2D_RN
  - AW通道传输写请求，延迟：D2D_AW_LATENCY (10周期)
  - 写请求在时刻(T3 + 10)到达Die1_D2D_RN
  - 保持所有原始信息（packet_id, source_die等）
```

#### 阶段3: 跨Die数据传输（W通道）
```
时刻T4: Die0_D2D_SN → Die1_D2D_RN  
  - W通道传输写数据，延迟：D2D_W_LATENCY (2周期)
  - 写数据在时刻(T4 + 2)到达Die1_D2D_RN
  - Die1_D2D_RN消耗自己的tracker资源接收数据
```

#### 阶段4: Die1内部写请求处理
```
时刻T5: Die1_D2D_RN → Die1_SN
  - 转发写请求到目标SN节点
  - Die1_SN检查资源并返回data_send响应

时刻T6: Die1_D2D_RN → Die1_SN  
  - 收到data_send响应后，从缓存发送写数据到Die1_SN
  - Die1_SN处理写操作（异步完成，不需要响应）

时刻T7: Die1_D2D_RN自行生成write_complete响应
  - D2D_RN在发送完写数据后立即生成write_complete响应
  - 释放D2D_RN自己的tracker
  - 注意：不等待Die1_SN的响应
```

#### 阶段5: 跨Die响应返回（B通道）
```
时刻T8: Die1_D2D_RN → Die0_D2D_SN
  - D2D_RN发送write_complete响应到AXI_B通道
  - B通道传输写完成响应，延迟：D2D_B_LATENCY (8周期)  
  - 响应在时刻(T8 + 8)到达Die0_D2D_SN
  - Die0_D2D_SN收到响应后释放自己的tracker
```

#### 阶段6-7: Die0响应转发与事务完成
```
时刻T9: Die0_D2D_SN → Die0_RN
  - 转发B通道写完成响应给原始RN
  - Die0_RN收到B通道响应后才释放tracker
  - 完成整个跨Die写事务
```

### 3.3 关键Tracker管理规则

| 组件 | Tracker消耗时机 | Tracker释放时机 |
|------|----------------|----------------|
| Die0_RN | 发送写请求时 | **收到B通道响应后**（跨Die特殊处理） |
| Die0_D2D_SN | 接收写请求时 | 收到B通道响应后 |
| Die1_D2D_RN | 接收W通道数据时 | 收到Die1_SN响应后 |
| Die1_SN | 接收写请求时 | 发送写完成响应后（标准流程） |

**重要说明**: Die0_RN的tracker管理与普通Die内写操作不同，必须等到B通道响应才能释放，而不是在data_send响应后释放。

## 4. D2D Tracker管理和Retry机制详细设计

### 4.1 读请求Tracker管理

#### D2D_SN读请求资源检查（阶段1）
**当前实现问题**: D2D_SN直接转发读请求，绕过了资源检查
**正确实现**:
```python
def handle_cross_die_read_request(self, flit: Flit):
    """处理跨Die读请求 - 必须进行资源检查"""
    # 检查D2D_SN的RO tracker资源
    has_tracker = self.node.sn_tracker_count[self.ip_type]["ro"][self.ip_pos] > 0
    
    if has_tracker:
        # 分配tracker
        self.node.sn_tracker_count[self.ip_type]["ro"][self.ip_pos] -= 1
        flit.sn_tracker_type = "ro"
        self.node.sn_tracker[self.ip_type][self.ip_pos].append(flit)
        
        # 转发请求到D2D_RN
        self._handle_cross_die_transfer(flit)
    else:
        # 资源不足，返回negative响应
        negative_rsp = self._create_response_flit(flit, "negative")
        self.enqueue(negative_rsp, "rsp")
        
        # 加入等待队列
        self.node.sn_req_wait[flit.req_type][self.ip_type][self.ip_pos].append(flit)
```

#### D2D_RN读请求资源检查（阶段3）
**当前实现问题**: 资源不足时直接丢弃请求（return False）
**正确设计**: 由于AXI不支持retry，D2D_RN不应该拒绝请求

#### Tracker释放时机
| 组件 | 分配时机 | 释放时机 | Tracker类型 |
|------|----------|----------|-------------|
| **D2D_SN** | 阶段1: 收到GDMA读请求 | 阶段6: 数据转发给GDMA后 | RO (Read Only) |
| **D2D_RN** | 阶段3: 收到跨Die读请求 | 阶段5: 数据发送到AXI R通道后 | Read |

### 4.2 写请求Tracker管理

#### D2D_SN写请求资源检查（阶段1）
**当前实现**: 已正确实现资源检查
```python
def handle_local_cross_die_write_request(self, flit: Flit):
    # 检查share tracker和WDB资源
    has_tracker = self.node.sn_tracker_count[self.ip_type]["share"][self.ip_pos] > 0
    has_databuffer = self.node.sn_wdb_count[self.ip_type][self.ip_pos] >= flit.burst_length
    
    if has_tracker and has_databuffer:
        # 分配资源，发送datasend响应
    else:
        # 返回negative响应，加入等待队列
```

#### Tracker释放时机
| 组件 | 分配时机 | 释放时机 | Tracker类型 |
|------|----------|----------|-------------|
| **D2D_SN** | 阶段1: 收到GDMA写请求 | 阶段6: 写完成响应转发后 | Share + WDB |
| **D2D_RN** | 阶段3: 收到跨Die写请求 | 阶段4: 本地写完成后 | Write + WDB |

### 4.3 Retry机制设计

#### GDMA Retry行为
```python
# ip_interface.py 中的retry逻辑
def _handle_received_response(self, rsp: Flit):
    if rsp.rsp_type == "negative":
        # 标记请求无效，等待positive响应
        req.req_state = "invalid"
        req.req_attr = "old"
        # 注意：不会自动retry
        
    elif rsp.rsp_type == "positive":
        # 重新激活请求
        req.req_state = "valid" 
        req.req_attr = "old"
        # 重新注入网络
        self.enqueue(req, "req", retry=True)
```

#### D2D_SN Retry通知机制
```python
def release_completed_sn_tracker(self, req: Flit):
    # 1. 释放tracker和databuffer资源
    self.node.sn_tracker[self.ip_type][self.ip_pos].remove(req)
    self.node.sn_tracker_count[self.ip_type][req.sn_tracker_type][self.ip_pos] += 1
    
    if req.req_type == "write":
        self.node.sn_wdb_count[self.ip_type][self.ip_pos] += req.burst_length
    
    # 2. 检查等待队列，处理等待的请求
    wait_list = self.node.sn_req_wait[req.req_type][self.ip_type][self.ip_pos]
    
    if wait_list and self.has_sufficient_resources():
        new_req = wait_list.pop(0)
        
        if req.req_type == "write":
            # 写请求：发送positive响应触发GDMA retry
            self.create_rsp(new_req, "positive")
            
        elif req.req_type == "read":
            # 读请求：分配资源并直接处理
            self.allocate_tracker_resources(new_req)
            self._handle_cross_die_transfer(new_req)
```

### 4.4 资源预留策略

#### 设计原则
1. **D2D_SN Gate-keeping**: 在源Die进行资源检查，确保有足够资源完成跨Die传输
2. **AXI Commitment**: 一旦进入AXI传输，必须保证能完成
3. **Early Allocation**: 在阶段1就分配所有必要资源

#### 资源配置
```yaml
# d2d_config.yaml
D2D_SN_R_TRACKER_OSTD: 48   # D2D SN 读跟踪器数量
D2D_SN_W_TRACKER_OSTD: 48   # D2D SN 写跟踪器数量  
D2D_SN_RDB_SIZE: 192        # D2D SN 读缓冲大小
D2D_SN_WDB_SIZE: 192        # D2D SN 写缓冲大小

D2D_RN_R_TRACKER_OSTD: 48   # D2D RN 读跟踪器数量
D2D_RN_W_TRACKER_OSTD: 48   # D2D RN 写跟踪器数量
D2D_RN_RDB_SIZE: 192        # D2D RN 读缓冲大小  
D2D_RN_WDB_SIZE: 192        # D2D RN 写缓冲大小
```

### 4.5 当前实现问题总结

#### 紧急修复项
1. **D2D_SN读请求绕过资源检查** - 导致无限制转发
2. **D2D_RN丢弃请求** - 违反AXI协议
3. **缺少retry通知机制** - 等待队列无法被唤醒

#### 修复优先级
1. **高**: 添加D2D_SN读请求资源检查
2. **高**: 实现正确的retry通知机制  
3. **中**: 优化D2D_RN资源管理
4. **低**: 添加详细的调试日志

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

### 9.2 具体实现步骤

#### 步骤1：创建D2D IP接口类

**文件：`src/utils/components/d2d_rn_interface.py`**
```python
class D2D_RN_Interface(IPInterface):
    """Die间请求节点 - 发起跨Die请求"""
    def __init__(self, ip_pos, config, req_network, rsp_network, data_network, node, routes):
        super().__init__("d2d_rn", ip_pos, config, req_network, rsp_network, data_network, node, routes)
        self.die_id = config.DIE_ID
        self.cross_die_delay_queue = []  # [(arrival_cycle, flit)]
        self.target_die_interfaces = {}  # 将由D2D_Model设置
    
    def handle_cross_die_request(self, flit):
        """处理跨Die请求 - 添加AR延迟"""
        if flit.target_die_id != self.die_id:
            delay = self.config.D2D_AR_LATENCY
            arrival_cycle = self.current_cycle + delay
            target_d2d_sn = self.target_die_interfaces[flit.target_die_id]
            target_d2d_sn.schedule_cross_die_receive(flit, arrival_cycle)
```

**文件：`src/utils/components/d2d_sn_interface.py`**
```python
class D2D_SN_Interface(IPInterface):
    """Die间响应节点 - 接收跨Die请求"""
    def __init__(self, ip_pos, config, req_network, rsp_network, data_network, node, routes):
        super().__init__("d2d_sn", ip_pos, config, req_network, rsp_network, data_network, node, routes)
        self.die_id = config.DIE_ID
        self.cross_die_receive_queue = []  # [(arrival_cycle, flit)]
        
    def schedule_cross_die_receive(self, flit, arrival_cycle):
        """调度跨Die接收"""
        heapq.heappush(self.cross_die_receive_queue, (arrival_cycle, flit))
    
    def process_cross_die_receives(self):
        """处理到期的跨Die接收"""
        while (self.cross_die_receive_queue and 
               self.cross_die_receive_queue[0][0] <= self.current_cycle):
            arrival_cycle, flit = heapq.heappop(self.cross_die_receive_queue)
            self.handle_received_cross_die_flit(flit)
```

#### 步骤2：创建D2D模型主类

**文件：`src/core/d2d_model.py`**
```python
class D2D_Model:
    """D2D仿真主类 - 管理多Die协调"""
    def __init__(self, config, traffic_config, **kwargs):
        self.config = config
        self.current_cycle = 0
        self.num_dies = config.NUM_DIES  # 默认2
        
        # 创建多个Die实例
        self.dies = {}
        for die_id in range(self.num_dies):
            die_config = self.create_die_config(die_id)
            self.dies[die_id] = BaseModel(
                die_config, traffic_config, **kwargs
            )
            self.dies[die_id].die_id = die_id
        
        # 设置跨Die连接
        self.setup_cross_die_connections()
    
    def create_die_config(self, die_id):
        """为每个Die创建配置"""
        die_config = copy.deepcopy(self.config)
        die_config.DIE_ID = die_id
        # 添加D2D节点到IP配置中
        die_config.D2D_RN_SEND_POSITION_LIST = [die_config.D2D_RN_POSITION]
        die_config.D2D_SN_SEND_POSITION_LIST = [die_config.D2D_SN_POSITION]
        return die_config
    
    def setup_cross_die_connections(self):
        """建立Die间连接"""
        for die_id, die in self.dies.items():
            d2d_rn = die.ip_modules.get(("d2d_rn", die.config.D2D_RN_POSITION))
            if d2d_rn:
                # 设置到其他Die的连接
                for other_die_id, other_die in self.dies.items():
                    if other_die_id != die_id:
                        d2d_sn = other_die.ip_modules.get(("d2d_sn", other_die.config.D2D_SN_POSITION))
                        d2d_rn.target_die_interfaces[other_die_id] = d2d_sn
```

#### 步骤3：扩展配置系统

**修改：`config/config.py`**
- 添加D2D相关参数解析
- 支持D2D节点位置配置
- 添加跨Die延迟参数

**新建：`config/d2d_config.json`**
```json
{
  "d2d_enabled": true,
  "num_dies": 2,
  "d2d_rn_position": 35,
  "d2d_sn_position": 36,
  "d2d_ar_latency": 10,
  "d2d_r_latency": 8,
  "d2d_aw_latency": 10,
  "d2d_w_latency": 2,
  "d2d_b_latency": 8
}
```

#### 步骤4：实现D2D读流程

**在D2D_RN中实现：**
1. 检测跨Die请求（target_die_id != current_die_id）
2. 添加AR延迟，发送到目标Die的D2D_SN
3. 接收跨Die返回的读数据（R延迟）

**在D2D_SN中实现：**
1. 接收跨Die请求，转发到Die内目标SN
2. 接收Die内返回数据，添加R延迟发回源Die

#### 步骤5：创建测试用例和Traffic文件

**Traffic文件格式：**
```
# cycle, src_die, src_node, dst_die, dst_node, req_type, burst_length
100, 0, 5, 1, 10, read, 4
200, 0, 6, 1, 12, write, 8
```

**测试脚本：**
- 基本D2D读功能测试
- 多Die并行测试
- 性能对比测试

### 9.3 关键设计决策

#### 不修改现有代码原则
- BaseModel保持完全不变
- 现有IP接口逻辑不修改
- 网络组件完全复用

#### 集成策略
- D2D组件作为新的IP类型
- 挂载在配置指定的节点位置
- 通过现有机制注册到ip_modules

#### 延迟处理方案
- 每个D2D组件内部管理延迟队列
- 在主循环中调用process_cross_die_*方法
- 使用heapq优化延迟队列性能

### 9.4 后续扩展规划

#### Phase 1: 基础读流程
- 实现上述步骤1-4
- 验证基本跨Die读功能

#### Phase 2: 写流程支持
- 扩展D2D组件支持写操作
- 实现AW/W/B三通道延迟
- 添加DBIDValid信号处理

#### Phase 3: 性能优化
- 支持多个outstanding事务
- 添加带宽限制模拟
- 优化延迟队列管理

#### Phase 4: 功能增强
- 支持超过2个Die
- 添加错误处理机制
- 集成可视化分析

这个实现计划确保了D2D功能与现有系统的完美集成，同时为未来扩展留出了空间。