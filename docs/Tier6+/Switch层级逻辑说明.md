# Rack层和Board层Switch逻辑说明

## 当前实现逻辑

### 后端 topology.py 中的层级关系

```
数据中心层 (Datacenter)
    ↓ 连接到 Pod层Switch 或 直接连接Pod
Pod层 (Pod)
    ↓ 连接到 Rack层Switch 或 直接连接Rack
Rack层 (Rack)
    ↓ 连接到 Board层Switch 或 直接连接Board
Board层 (Board)
    ↓ 连接到 Chip
```

### Board层Switch逻辑 (topology.py:291-314)

**触发条件：** `board_level.enabled=true` 且配置了 `layers`

**生成逻辑：**
1. 遍历每个Pod → 每个Rack → 每个Board
2. 获取该Board内所有Chip的ID
3. 生成Switch实例，`parent_id = board.id`, `hierarchy_level = 'board'`
4. 生成 Chip → Board层Switch 的连接

### Rack层Switch逻辑 (topology.py:317-379)

**触发条件：** `rack_level.enabled=true` 且配置了 `layers`

**生成逻辑：**
1. 遍历每个Pod → 每个Rack
2. **确定下层设备：**
   - 如果 `board_level.enabled=true` 且 `connect_to_upper_level=true`：
     - 连接到 **Board层顶层Switch**（不是直接连Board）
   - 否则：
     - 直接连接到 **Board**
3. 生成Switch实例，`parent_id = rack.id`, `hierarchy_level = 'rack'`
4. 生成连接（设备 → Rack层Switch）

### 前端显示逻辑 (TopologyGraph.tsx)

**Rack层视图 (674-729行)：**
- 显示节点：Board + Rack层Switch
- 显示连接：Board和Rack层Switch之间的连接

**问题：** 当Board层Switch启用时，后端实际生成的是 `Rack Switch → Board Switch` 的连接，但前端只筛选 Board 和 Rack Switch 的连接，导致连线不显示。

**Board层视图 (731-785行)：**
- 显示节点：Chip + Board层Switch
- 显示连接：Chip和Board层Switch之间的连接
- 工作正常

## 需要修复的问题

### Rack层连接转换缺失

参考Pod层的实现（643-672行），需要在Rack层添加类似的连接转换逻辑：

```typescript
// 构建Board层Switch到Board的映射
const boardSwitchToBoard: Record<string, string> = {}
;(topology.switches || [])
  .filter(s => s.hierarchy_level === 'board' && s.parent_id?.startsWith(currentRack.id))
  .forEach(s => { boardSwitchToBoard[s.id] = s.parent_id })

// 筛选和转换连接
edgeList = topology.connections
  .filter(c => {
    const sourceInRack = boardIds.has(c.source) || rackSwitchIds.has(c.source)
    const targetInRack = boardIds.has(c.target) || rackSwitchIds.has(c.target)
    if (sourceInRack && targetInRack) return true
    // 跨层连接（Rack Switch到Board Switch）
    if (rackSwitchIds.has(c.source) && boardSwitchToBoard[c.target]) return true
    if (rackSwitchIds.has(c.target) && boardSwitchToBoard[c.source]) return true
    return false
  })
  .map(c => {
    let source = c.source
    let target = c.target
    if (boardSwitchToBoard[c.source]) source = boardSwitchToBoard[c.source]
    if (boardSwitchToBoard[c.target]) target = boardSwitchToBoard[c.target]
    return { source, target, bandwidth: c.bandwidth, latency: c.latency }
  })
```

## 修改文件

- `Tier6+model/frontend/src/components/TopologyGraph.tsx`
  - 在Rack层视图部分（约711-729行）添加连接转换逻辑
