# Tier6+ 多层级网络数学建模框架设计计划

## 目标

构建一个**纯数学建模**框架，支持从 Die 到 Pod 的多层级网络性能分析：
- Chip-to-Chip (C2C)
- Board-to-Board (B2B)
- Server-to-Server (S2S)
- Pod-to-Pod (P2P)

## 设计原则

1. **黑盒模型优先**：每层级作为独立单元，通过延迟和带宽参数建模
2. **预留扩展接口**：后续可细化为简化拓扑模型
3. **独立运行**：暂不与现有 NoC 仿真集成，但保留接口

## 架构设计

```
HierarchicalAnalyzer (顶层分析器)
    │
    ├── PodModel ─── ServerModel ─── BoardModel ─── ChipModel ─── DieModel
    │
    ├── QueuingModel (排队论: M/M/1, M/G/1)
    ├── CongestionModel (拥塞延迟模型)
    └── BandwidthModel (带宽瓶颈分析)
```

## 核心数学模型

### 1. 延迟模型
```
总延迟 = Σ(各层级延迟)
层级延迟 = 传播延迟 + 排队延迟 + 处理延迟

排队延迟 (M/M/1): W = 1/(μ-λ)
拥塞延迟: L = L_base / (1 - utilization)
```

### 2. 带宽模型
```
有效带宽 = min(各层级带宽限制)
瓶颈识别: 找出利用率最高的层级
```

### 3. 规模扩展分析
```
扩展因子 = f(节点数, 拓扑类型, 流量模式)
```

## 目录结构

```
src/hierarchical/
├── __init__.py
├── base.py              # HierarchicalModel 基类 + 数据类
├── math_models.py       # 排队论/拥塞/带宽数学模型
├── layers/
│   ├── __init__.py
│   ├── die_model.py     # Die层级
│   ├── chip_model.py    # Chip层级 (D2D)
│   ├── board_model.py   # Board层级 (C2C)
│   ├── server_model.py  # Server层级 (B2B)
│   └── pod_model.py     # Pod层级 (S2S)
├── analyzer.py          # HierarchicalAnalyzer 主分析器
└── config_loader.py     # 配置加载器

config/hierarchical/
├── default.yaml         # 默认参数配置
└── examples/            # 示例配置
```

## 实现步骤

### Phase 1: 核心框架
1. 创建 `src/hierarchical/base.py`
   - `HierarchicalModel` 抽象基类
   - `LatencyResult`, `BandwidthResult`, `TrafficFlow` 数据类
   - 层级组合接口

2. 创建 `src/hierarchical/math_models.py`
   - `QueuingModel`: M/M/1, M/G/1 延迟计算
   - `CongestionModel`: 利用率-延迟非线性模型
   - `BandwidthModel`: 带宽瓶颈计算

### Phase 2: 层级模型
3. 实现各层级模型 (按优先级):
   - `DieModel` - NoC 内部延迟估算
   - `ChipModel` - D2D 通信延迟
   - `BoardModel` - C2C (UCIe/PCIe)
   - `ServerModel` - 板间通信
   - `PodModel` - 网络交换

### Phase 3: 分析器
4. 创建 `src/hierarchical/analyzer.py`
   - 层级构建和递归分析
   - 延迟分解报告
   - 带宽瓶颈识别
   - 规模扩展预测

### Phase 4: 配置系统
5. 创建配置文件格式和加载器
   - YAML 配置支持
   - 层级参数继承

## 关键参数参考值

| 层级 | 典型延迟 | 典型带宽 |
|------|---------|---------|
| Die (NoC) | 5-50 ns | 128-512 GB/s |
| Chip (D2D) | 20-100 ns | 64-192 GB/s |
| Board (C2C) | 100-500 ns | 32-128 GB/s |
| Server (B2B) | 500-2000 ns | 16-64 GB/s |
| Pod (S2S) | 2-10 μs | 12.5-100 GB/s |

## 输出示例

```python
analyzer = HierarchicalAnalyzer(config)
analyzer.build_hierarchy(top_level='pod')

results = analyzer.analyze(traffic_flows)
# {
#   'total_latency_ns': 5234.5,
#   'latency_breakdown': {'die': 45, 'chip': 120, 'board': 350, ...},
#   'bottleneck': {'layer': 'board', 'utilization': 0.87},
#   'scaling_factor': 1.23
# }
```

---

## 可视化设计

### 跨层级连接类型

| 连接类型 | 层级 | 物理特性 | 典型带宽 | 延迟 |
|---------|------|----------|---------|------|
| D2D | Die 间 | UCIe/CoWoS | TB/s级 | <10ns |
| C2C | Chip 间 | UCIe/PCIe | 100GB/s级 | ~10ns |
| B2B | Board 间 | 背板/线缆 | 10-100GB/s | ~100ns |
| S2S | Server 间 | 网络交换 | 1-10GB/s | μs级 |
| P2P | Pod 间 | 数据中心网络 | 可变 | ms级 |

### 架构图可视化方案对比

#### 方案一：语义缩放嵌套容器 (React Flow)
```
缩放级别:
Level 1 (最远): Pod 视图 - Pod框 + P2P连接
Level 2: Server 视图 - Pod展开显示Server
Level 3: Board 视图 - Server展开显示Board
Level 4: Chip 视图 - Board展开显示Chip
Level 5: Die 视图 - Chip展开显示Die网格
Level 6 (最近): Node 视图 - 完整节点+IP+链路
```
- **优点**: 用户体验直观(类似地图)，性能优秀
- **缺点**: 需从Cytoscape迁移，难以同时看远景和近景

#### 方案二：3D层叠视图 (Three.js)
```
Z轴(层级)
    ↑
Pod ────── (z=500)
Server ─── (z=400)
Board ──── (z=300)
Chip ───── (z=200)
Die ────── (z=100)
Node ───── (z=0)
```
- **优点**: 视觉震撼，层级关系清晰，适合展示跨层连接
- **缺点**: 开发复杂，性能要求高，细节查看不便

#### 方案三：分离面板+联动高亮 (推荐)
```
┌────────────────────────┬─────────────────────────────┐
│                        │                             │
│   总览面板 (树状图)    │     主视图 (详细拓扑)       │
│   ┌─────────────┐      │   ┌─────────────────────┐   │
│   │ └Pod0       │      │   │                     │   │
│   │   ├Srv0    ●│←选中 │   │  当前层级详细视图   │   │
│   │   │├Brd0    │      │   │  (复用Cytoscape)    │   │
│   │   ││├Chip0  │      │   │                     │   │
│   │   │││└Die0  │      │   │   ┌──┬──┬──┐        │   │
│   │   └Srv1     │      │   │   │  │  │  │        │   │
│   └─────────────┘      │   │   └──┴──┴──┘        │   │
│                        │   └─────────────────────┘   │
├────────────────────────┴─────────────────────────────┤
│ 面包屑: Pod0 > Server0 > Board0 > Chip0 > Die0       │
└──────────────────────────────────────────────────────┘
```
- **优点**: 增量开发，复用现有代码，总览+细节分离
- **缺点**: 屏幕空间利用较低

#### 方案四：径向/圆环层级图 (D3.js)
- **优点**: 层级结构清晰，空间利用高
- **缺点**: 内层空间有限，不适合网格拓扑

### 推荐方案：方案三 + 方案一混合

**理由**：
1. 渐进式开发：先实现分离面板，复用现有Cytoscape代码
2. 稳定迭代：主视图后续可升级为React Flow支持语义缩放
3. 功能完整：总览+详情分离保证不丢失全局视野
4. 技术风险低：不依赖WebGL/3D

### 连接线视觉设计

```
连接类型 │ 线型     │ 粗细 │ 颜色   │ 动画
─────────┼──────────┼──────┼────────┼────────
D2D      │ 虚线     │ 3px  │ #13c2c2 青色  │ 无
C2C      │ 长虚线   │ 4px  │ #722ed1 紫色  │ 无
B2B      │ 点线     │ 5px  │ #fa8c16 橙色  │ 无
S2S      │ 实线     │ 6px  │ #52c41a 绿色  │ 流动
P2P      │ 粗实线   │ 8px  │ #1677ff 蓝色  │ 流动
```

### 外部连接端口表示

当进入某一层级内部视图时，跨层级连接显示为"端口"：
```
Die 0 内部视图:
┌────────────────────────────┐
│ [D2D→Die1] ←外部连接端口   │
│    ↓                       │
│  ┌──┬──┬──┐                │
│  │G0│D0│  │                │
│  ├──┼──┼──┤                │
│  │  │N1│  │                │
│  └──┴──┴──┘                │
│    ↓                       │
│ [D2D→Die2]                 │
└────────────────────────────┘
```

### 前端组件结构

```
unified_web/frontend/src/components/hierarchy/
├── HierarchyTree.tsx          # 左侧树状导航
├── HierarchyBreadcrumb.tsx    # 面包屑导航
├── HierarchyMainView.tsx      # 主视图容器
├── ConnectionPortal.tsx       # 外部连接端口组件
├── HierarchyLegend.tsx        # 连接类型图例
└── index.ts
```

### 统计图表

#### 延迟分解图 (ECharts堆叠条形图)
```
总延迟: 5234.5 ns
+=================================================================+
|  Die(45ns)  |  Chip(120ns)  |  Board(350ns)  | Server | Pod    |
|    0.9%     |     2.3%      |      6.7%      | 38.2%  | 51.9%  |
+=================================================================+
```

#### 规模扩展曲线 (ECharts对数坐标图)
```
延迟(us)
   10 |                    ....●
      |              .....
    5 |        .....
      |   .....
    1 |●●●
      +------------------------------>
       1   4   16   64  256  1024
              节点数
```

### 可视化实现路线

| Phase | 内容 | 预估 |
|-------|------|------|
| P1 | 数据模型扩展 + 层级数据结构 | 2周 |
| P2 | 总览面板(树状图+面包屑+联动) | 3周 |
| P3 | 主视图增强(扩展支持多层级) | 3周 |
| P4 | 交互优化(双击下钻+右键菜单) | 2周 |
| P5 | (可选) React Flow迁移实现语义缩放 | - |

---

## 后续扩展点

1. **细化建模**: 黑盒 → 简化拓扑 → 详细拓扑
2. **仿真集成**: 与 BaseModel/D2D_Model 结果对比校正
3. **动画效果**: 数据流动画展示跨层级通信过程
