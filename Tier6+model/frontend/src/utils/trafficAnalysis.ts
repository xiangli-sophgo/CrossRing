/**
 * LLM 并行策略流量分析算法
 * 根据并行策略配置计算各 chip 间的理论通信流量
 */

import {
  ParallelismConfig,
  TrafficConfigItem,
  CommunicationGroup,
  LinkTraffic,
  NodeLoad,
  TrafficAnalysisResult,
  HierarchicalTopology,
  ChipConfig,
  ConnectionConfig,
  CollectiveType,
} from '../types';

// ============================================
// Chip Rank 计算
// ============================================

interface ChipRanks {
  dp: number;
  tp: number;
  pp: number;
  ep: number;
  sp: number;
}

/**
 * 根据 chip 索引计算其在各并行维度的 rank
 * 编号顺序: [dp, tp, pp, ep, sp] (外层到内层)
 */
export function getChipRanks(chipIndex: number, config: ParallelismConfig): ChipRanks {
  const { tp_size, pp_size, ep_size, sp_size } = config;
  const sp_rank = chipIndex % sp_size;
  const ep_rank = Math.floor(chipIndex / sp_size) % ep_size;
  const pp_rank = Math.floor(chipIndex / (sp_size * ep_size)) % pp_size;
  const tp_rank = Math.floor(chipIndex / (sp_size * ep_size * pp_size)) % tp_size;
  const dp_rank = Math.floor(chipIndex / (sp_size * ep_size * pp_size * tp_size));
  return { dp: dp_rank, tp: tp_rank, pp: pp_rank, ep: ep_rank, sp: sp_rank };
}

/**
 * 根据 ranks 计算 chip 索引
 */
export function getChipIndex(ranks: ChipRanks, config: ParallelismConfig): number {
  const { tp_size, pp_size, ep_size, sp_size } = config;
  return ranks.dp * (tp_size * pp_size * ep_size * sp_size) +
         ranks.tp * (pp_size * ep_size * sp_size) +
         ranks.pp * (ep_size * sp_size) +
         ranks.ep * sp_size +
         ranks.sp;
}

// ============================================
// 通信组生成
// ============================================

/**
 * 生成所有通信组
 */
export function generateCommunicationGroups(
  chips: ChipConfig[],
  config: ParallelismConfig
): CommunicationGroup[] {
  const groups: CommunicationGroup[] = [];
  const totalChips = chips.length;
  const expectedTotal = config.dp_size * config.tp_size * config.pp_size * config.ep_size * config.sp_size;

  if (expectedTotal > totalChips) {
    console.warn(`并行度配置需要 ${expectedTotal} 个 chip，但只有 ${totalChips} 个可用`);
    return groups;
  }

  // 生成 TP 通信组
  if (config.tp_size > 1) {
    groups.push(...generateTPGroups(chips, config));
  }

  // 生成 DP 通信组
  if (config.dp_size > 1) {
    groups.push(...generateDPGroups(chips, config));
  }

  // 生成 PP 通信组
  if (config.pp_size > 1) {
    groups.push(...generatePPGroups(chips, config));
  }

  // 生成 EP 通信组
  if (config.ep_size > 1) {
    groups.push(...generateEPGroups(chips, config));
  }

  // 生成 SP 通信组
  if (config.sp_size > 1) {
    groups.push(...generateSPGroups(chips, config));
  }

  return groups;
}

/**
 * 生成 TP 通信组 (相同 dp, pp, ep, sp rank 的 chips)
 */
function generateTPGroups(chips: ChipConfig[], config: ParallelismConfig): CommunicationGroup[] {
  const groups: CommunicationGroup[] = [];
  const { dp_size, tp_size, pp_size, ep_size, sp_size, message_size_mb, tp_collective } = config;

  for (let dp = 0; dp < dp_size; dp++) {
    for (let pp = 0; pp < pp_size; pp++) {
      for (let ep = 0; ep < ep_size; ep++) {
        for (let sp = 0; sp < sp_size; sp++) {
          const members: string[] = [];
          for (let tp = 0; tp < tp_size; tp++) {
            const idx = getChipIndex({ dp, tp, pp, ep, sp }, config);
            if (idx < chips.length) {
              members.push(chips[idx].id);
            }
          }
          if (members.length > 1) {
            groups.push({
              id: `tp_dp${dp}_pp${pp}_ep${ep}_sp${sp}`,
              type: 'TP',
              collective: tp_collective,
              members,
              data_volume_mb: calculateCollectiveVolume(tp_collective, members.length, message_size_mb),
            });
          }
        }
      }
    }
  }
  return groups;
}

/**
 * 生成 DP 通信组 (相同 tp, pp, ep, sp rank 的 chips)
 */
function generateDPGroups(chips: ChipConfig[], config: ParallelismConfig): CommunicationGroup[] {
  const groups: CommunicationGroup[] = [];
  const { dp_size, tp_size, pp_size, ep_size, sp_size, message_size_mb, dp_collective } = config;

  for (let tp = 0; tp < tp_size; tp++) {
    for (let pp = 0; pp < pp_size; pp++) {
      for (let ep = 0; ep < ep_size; ep++) {
        for (let sp = 0; sp < sp_size; sp++) {
          const members: string[] = [];
          for (let dp = 0; dp < dp_size; dp++) {
            const idx = getChipIndex({ dp, tp, pp, ep, sp }, config);
            if (idx < chips.length) {
              members.push(chips[idx].id);
            }
          }
          if (members.length > 1) {
            groups.push({
              id: `dp_tp${tp}_pp${pp}_ep${ep}_sp${sp}`,
              type: 'DP',
              collective: dp_collective,
              members,
              data_volume_mb: calculateCollectiveVolume(dp_collective, members.length, message_size_mb),
            });
          }
        }
      }
    }
  }
  return groups;
}

/**
 * 生成 PP 通信组 (相邻 stage 的 P2P 通信)
 */
function generatePPGroups(chips: ChipConfig[], config: ParallelismConfig): CommunicationGroup[] {
  const groups: CommunicationGroup[] = [];
  const { dp_size, tp_size, pp_size, ep_size, sp_size, message_size_mb, pp_collective } = config;

  for (let dp = 0; dp < dp_size; dp++) {
    for (let tp = 0; tp < tp_size; tp++) {
      for (let ep = 0; ep < ep_size; ep++) {
        for (let sp = 0; sp < sp_size; sp++) {
          // PP 默认是相邻 stage 之间的通信
          if (pp_collective === 'P2P') {
            // P2P: 相邻 stage 之间
            for (let pp = 0; pp < pp_size - 1; pp++) {
              const srcIdx = getChipIndex({ dp, tp, pp, ep, sp }, config);
              const dstIdx = getChipIndex({ dp, tp, pp: pp + 1, ep, sp }, config);
              if (srcIdx < chips.length && dstIdx < chips.length) {
                groups.push({
                  id: `pp_dp${dp}_tp${tp}_ep${ep}_sp${sp}_stage${pp}to${pp + 1}`,
                  type: 'PP',
                  collective: 'P2P',
                  members: [chips[srcIdx].id, chips[dstIdx].id],
                  data_volume_mb: message_size_mb,
                });
              }
            }
          } else {
            // 其他集合操作: 所有 PP stage 作为一个组
            const members: string[] = [];
            for (let pp = 0; pp < pp_size; pp++) {
              const idx = getChipIndex({ dp, tp, pp, ep, sp }, config);
              if (idx < chips.length) {
                members.push(chips[idx].id);
              }
            }
            if (members.length > 1) {
              groups.push({
                id: `pp_dp${dp}_tp${tp}_ep${ep}_sp${sp}`,
                type: 'PP',
                collective: pp_collective,
                members,
                data_volume_mb: calculateCollectiveVolume(pp_collective, members.length, message_size_mb),
              });
            }
          }
        }
      }
    }
  }
  return groups;
}

/**
 * 生成 EP 通信组 (相同 dp, tp, pp, sp rank 的 chips)
 */
function generateEPGroups(chips: ChipConfig[], config: ParallelismConfig): CommunicationGroup[] {
  const groups: CommunicationGroup[] = [];
  const { dp_size, tp_size, pp_size, ep_size, sp_size, message_size_mb, ep_collective } = config;

  for (let dp = 0; dp < dp_size; dp++) {
    for (let tp = 0; tp < tp_size; tp++) {
      for (let pp = 0; pp < pp_size; pp++) {
        for (let sp = 0; sp < sp_size; sp++) {
          const members: string[] = [];
          for (let ep = 0; ep < ep_size; ep++) {
            const idx = getChipIndex({ dp, tp, pp, ep, sp }, config);
            if (idx < chips.length) {
              members.push(chips[idx].id);
            }
          }
          if (members.length > 1) {
            groups.push({
              id: `ep_dp${dp}_tp${tp}_pp${pp}_sp${sp}`,
              type: 'EP',
              collective: ep_collective,
              members,
              data_volume_mb: calculateCollectiveVolume(ep_collective, members.length, message_size_mb),
            });
          }
        }
      }
    }
  }
  return groups;
}

/**
 * 生成 SP 通信组 (相同 dp, tp, pp, ep rank 的 chips)
 */
function generateSPGroups(chips: ChipConfig[], config: ParallelismConfig): CommunicationGroup[] {
  const groups: CommunicationGroup[] = [];
  const { dp_size, tp_size, pp_size, ep_size, sp_size, message_size_mb, sp_collective } = config;

  for (let dp = 0; dp < dp_size; dp++) {
    for (let tp = 0; tp < tp_size; tp++) {
      for (let pp = 0; pp < pp_size; pp++) {
        for (let ep = 0; ep < ep_size; ep++) {
          const members: string[] = [];
          for (let sp = 0; sp < sp_size; sp++) {
            const idx = getChipIndex({ dp, tp, pp, ep, sp }, config);
            if (idx < chips.length) {
              members.push(chips[idx].id);
            }
          }
          if (members.length > 1) {
            groups.push({
              id: `sp_dp${dp}_tp${tp}_pp${pp}_ep${ep}`,
              type: 'SP',
              collective: sp_collective,
              members,
              data_volume_mb: calculateCollectiveVolume(sp_collective, members.length, message_size_mb),
            });
          }
        }
      }
    }
  }
  return groups;
}

// ============================================
// 通信量计算公式
// ============================================

/**
 * Ring AllReduce 通信量
 * 每个节点发送和接收 2*(n-1)/n * data_size
 */
function calculateAllReduceVolume(groupSize: number, dataSize: number): number {
  if (groupSize <= 1) return 0;
  return 2 * (groupSize - 1) / groupSize * dataSize;
}

/**
 * AllGather 通信量
 * 每个节点接收 (n-1)/n * data_size
 */
function calculateAllGatherVolume(groupSize: number, dataSize: number): number {
  if (groupSize <= 1) return 0;
  return (groupSize - 1) / groupSize * dataSize;
}

/**
 * ReduceScatter 通信量
 * 每个节点发送 (n-1)/n * data_size
 */
function calculateReduceScatterVolume(groupSize: number, dataSize: number): number {
  if (groupSize <= 1) return 0;
  return (groupSize - 1) / groupSize * dataSize;
}

/**
 * AllToAll 通信量
 */
function calculateAllToAllVolume(groupSize: number, dataSize: number): number {
  if (groupSize <= 1) return 0;
  return (groupSize - 1) * dataSize / groupSize;
}

/**
 * 根据集合操作类型计算通信量
 */
function calculateCollectiveVolume(
  collective: CollectiveType,
  groupSize: number,
  dataSize: number
): number {
  switch (collective) {
    case 'AllReduce':
      return calculateAllReduceVolume(groupSize, dataSize);
    case 'AllGather':
      return calculateAllGatherVolume(groupSize, dataSize);
    case 'ReduceScatter':
      return calculateReduceScatterVolume(groupSize, dataSize);
    case 'AllToAll':
      return calculateAllToAllVolume(groupSize, dataSize);
    case 'P2P':
      return dataSize; // P2P 是单向传输
    default:
      return calculateAllReduceVolume(groupSize, dataSize);
  }
}

// ============================================
// 链路流量计算
// ============================================

/**
 * 从通信组计算每条链路的流量（基于实际拓扑路由）
 */
export function calculateLinkTraffic(
  groups: CommunicationGroup[],
  topology: HierarchicalTopology
): LinkTraffic[] {
  const linkTrafficMap = new Map<string, LinkTraffic>();

  // 构建拓扑图
  const graph = buildGraph(topology.connections);

  // 缓存路径（避免重复计算）
  const pathCache = new Map<string, string[] | null>();

  const getCachedPath = (src: string, dst: string): string[] | null => {
    const key = `${src}->${dst}`;
    if (pathCache.has(key)) return pathCache.get(key)!;
    const path = findShortestPath(graph, src, dst);
    pathCache.set(key, path);
    return path;
  };

  for (const group of groups) {
    const perLinkVolume = distributeGroupTrafficWithRouting(group, getCachedPath);

    for (const [linkKey, volume] of perLinkVolume.entries()) {
      const [source, target] = linkKey.split('->');
      const existing = linkTrafficMap.get(linkKey);

      if (existing) {
        existing.traffic_mb += volume;
        existing.contributions[group.type.toLowerCase() as keyof typeof existing.contributions] =
          (existing.contributions[group.type.toLowerCase() as keyof typeof existing.contributions] || 0) + volume;
      } else {
        linkTrafficMap.set(linkKey, {
          source,
          target,
          traffic_mb: volume,
          bandwidth_utilization: 0,
          contributions: {
            [group.type.toLowerCase()]: volume,
          },
        });
      }
    }
  }

  // 计算带宽利用率
  const linkTraffics = Array.from(linkTrafficMap.values());
  const maxTraffic = Math.max(...linkTraffics.map(lt => lt.traffic_mb), 1);

  for (const lt of linkTraffics) {
    lt.bandwidth_utilization = lt.traffic_mb / maxTraffic;
  }

  return linkTraffics;
}

/**
 * 将流量分配到路径上的所有边
 */
function addTrafficToPath(
  result: Map<string, number>,
  path: string[] | null,
  volume: number
): void {
  if (!path || path.length < 2) return;

  const edges = getPathEdges(path);
  for (const edge of edges) {
    // 统一使用 from->to 格式
    const linkKey = `${edge.from}->${edge.to}`;
    result.set(linkKey, (result.get(linkKey) || 0) + volume);
  }
}

/**
 * 将通信组的流量分配到各链路（使用实际路由）
 */
function distributeGroupTrafficWithRouting(
  group: CommunicationGroup,
  getPath: (src: string, dst: string) => string[] | null
): Map<string, number> {
  const result = new Map<string, number>();
  const { members, data_volume_mb, collective } = group;

  if (collective === 'P2P' && members.length === 2) {
    // P2P: 找到实际路径
    const path = getPath(members[0], members[1]);
    addTrafficToPath(result, path, data_volume_mb);
  } else if (collective === 'AllReduce' || collective === 'AllGather' || collective === 'ReduceScatter') {
    // Ring AllReduce: 每个节点向下一个节点发送数据，通过实际路径
    const perNodeVolume = data_volume_mb / members.length;
    for (let i = 0; i < members.length; i++) {
      const src = members[i];
      const dst = members[(i + 1) % members.length];
      const path = getPath(src, dst);
      addTrafficToPath(result, path, perNodeVolume);
    }
  } else if (collective === 'AllToAll') {
    // AllToAll: 每对节点之间都有通信，通过实际路径
    const perPairVolume = data_volume_mb / (members.length * (members.length - 1));
    for (const src of members) {
      for (const dst of members) {
        if (src !== dst) {
          const path = getPath(src, dst);
          addTrafficToPath(result, path, perPairVolume);
        }
      }
    }
  }

  return result;
}

// ============================================
// 节点负载计算
// ============================================

/**
 * 计算每个节点的流量负载
 */
export function calculateNodeLoads(linkTraffics: LinkTraffic[]): NodeLoad[] {
  const nodeLoadMap = new Map<string, { send: number; recv: number }>();

  for (const lt of linkTraffics) {
    // 发送端
    const sendLoad = nodeLoadMap.get(lt.source) || { send: 0, recv: 0 };
    sendLoad.send += lt.traffic_mb;
    nodeLoadMap.set(lt.source, sendLoad);

    // 接收端
    const recvLoad = nodeLoadMap.get(lt.target) || { send: 0, recv: 0 };
    recvLoad.recv += lt.traffic_mb;
    nodeLoadMap.set(lt.target, recvLoad);
  }

  return Array.from(nodeLoadMap.entries()).map(([id, load]) => ({
    id,
    send_mb: load.send,
    recv_mb: load.recv,
    total_mb: load.send + load.recv,
  }));
}

// ============================================
// 主分析函数
// ============================================

/**
 * 收集所有 chips
 */
export function collectAllChips(topology: HierarchicalTopology): ChipConfig[] {
  const chips: ChipConfig[] = [];
  for (const pod of topology.pods) {
    for (const rack of pod.racks) {
      for (const board of rack.boards) {
        chips.push(...board.chips);
      }
    }
  }
  return chips;
}

/**
 * 根据范围筛选 chips (支持多选，使用直接 ID 查找)
 */
export function collectChipsInScope(
  topology: HierarchicalTopology,
  config: ParallelismConfig
): ChipConfig[] {
  const { chip_scope, scope_pod_ids, scope_rack_ids, scope_board_ids, scope_chip_ids } = config;

  if (chip_scope === 'all') {
    return collectAllChips(topology);
  }

  const chips: ChipConfig[] = [];
  const addedChipIds = new Set<string>();

  // 辅助函数：添加 chip 并去重
  const addChip = (chip: ChipConfig) => {
    if (!addedChipIds.has(chip.id)) {
      addedChipIds.add(chip.id);
      chips.push(chip);
    }
  };

  // Pod 级别选择
  if (chip_scope === 'pod' && scope_pod_ids && scope_pod_ids.length > 0) {
    const podIdSet = new Set(scope_pod_ids);
    for (const pod of topology.pods) {
      if (podIdSet.has(pod.id)) {
        for (const rack of pod.racks) {
          for (const board of rack.boards) {
            board.chips.forEach(addChip);
          }
        }
      }
    }
    return chips;
  }

  // Rack 级别选择 (直接使用 rack.id)
  if (chip_scope === 'rack' && scope_rack_ids && scope_rack_ids.length > 0) {
    const rackIdSet = new Set(scope_rack_ids);
    for (const pod of topology.pods) {
      for (const rack of pod.racks) {
        if (rackIdSet.has(rack.id)) {
          for (const board of rack.boards) {
            board.chips.forEach(addChip);
          }
        }
      }
    }
    return chips;
  }

  // Board 级别选择 (直接使用 board.id)
  if (chip_scope === 'board' && scope_board_ids && scope_board_ids.length > 0) {
    const boardIdSet = new Set(scope_board_ids);
    for (const pod of topology.pods) {
      for (const rack of pod.racks) {
        for (const board of rack.boards) {
          if (boardIdSet.has(board.id)) {
            board.chips.forEach(addChip);
          }
        }
      }
    }
    return chips;
  }

  // Chip 级别选择 (直接使用 chip.id)
  if (chip_scope === 'chip' && scope_chip_ids && scope_chip_ids.length > 0) {
    const chipIdSet = new Set(scope_chip_ids);
    for (const pod of topology.pods) {
      for (const rack of pod.racks) {
        for (const board of rack.boards) {
          for (const chip of board.chips) {
            if (chipIdSet.has(chip.id)) {
              addChip(chip);
            }
          }
        }
      }
    }
    return chips;
  }

  return collectAllChips(topology);
}

/**
 * 根据 TrafficConfigItem 范围筛选 chips
 */
export function collectChipsFromConfigItem(
  topology: HierarchicalTopology,
  config: TrafficConfigItem
): ChipConfig[] {
  const { chip_scope, scope_pod_ids, scope_rack_ids, scope_board_ids, scope_chip_ids } = config;

  if (chip_scope === 'all') {
    return collectAllChips(topology);
  }

  const chips: ChipConfig[] = [];
  const addedChipIds = new Set<string>();

  const addChip = (chip: ChipConfig) => {
    if (!addedChipIds.has(chip.id)) {
      addedChipIds.add(chip.id);
      chips.push(chip);
    }
  };

  if (chip_scope === 'pod' && scope_pod_ids && scope_pod_ids.length > 0) {
    const podIdSet = new Set(scope_pod_ids);
    for (const pod of topology.pods) {
      if (podIdSet.has(pod.id)) {
        for (const rack of pod.racks) {
          for (const board of rack.boards) {
            board.chips.forEach(addChip);
          }
        }
      }
    }
    return chips;
  }

  if (chip_scope === 'rack' && scope_rack_ids && scope_rack_ids.length > 0) {
    const rackIdSet = new Set(scope_rack_ids);
    for (const pod of topology.pods) {
      for (const rack of pod.racks) {
        if (rackIdSet.has(rack.id)) {
          for (const board of rack.boards) {
            board.chips.forEach(addChip);
          }
        }
      }
    }
    return chips;
  }

  if (chip_scope === 'board' && scope_board_ids && scope_board_ids.length > 0) {
    const boardIdSet = new Set(scope_board_ids);
    for (const pod of topology.pods) {
      for (const rack of pod.racks) {
        for (const board of rack.boards) {
          if (boardIdSet.has(board.id)) {
            board.chips.forEach(addChip);
          }
        }
      }
    }
    return chips;
  }

  if (chip_scope === 'chip' && scope_chip_ids && scope_chip_ids.length > 0) {
    const chipIdSet = new Set(scope_chip_ids);
    for (const pod of topology.pods) {
      for (const rack of pod.racks) {
        for (const board of rack.boards) {
          for (const chip of board.chips) {
            if (chipIdSet.has(chip.id)) {
              addChip(chip);
            }
          }
        }
      }
    }
    return chips;
  }

  return collectAllChips(topology);
}

/**
 * 从单个 TrafficConfigItem 生成通信组
 */
export function generateGroupsFromConfigItem(
  chips: ChipConfig[],
  config: TrafficConfigItem
): CommunicationGroup[] {
  const groups: CommunicationGroup[] = [];
  const { parallelism, collective, size, message_size_mb } = config;

  if (size <= 1 || chips.length < size) {
    return groups;
  }

  // 只取前 size 个 chip 生成一个通信组
  const members = chips.slice(0, size).map(c => c.id);

  if (collective === 'P2P' && members.length >= 2) {
    // P2P: 相邻节点之间的通信
    for (let i = 0; i < members.length - 1; i++) {
      groups.push({
        id: `${config.id}_${parallelism}_p2p${i}`,
        type: parallelism,
        collective: 'P2P',
        members: [members[i], members[i + 1]],
        data_volume_mb: message_size_mb,
      });
    }
  } else {
    // 其他集合操作
    groups.push({
      id: `${config.id}_${parallelism}`,
      type: parallelism,
      collective,
      members,
      data_volume_mb: calculateCollectiveVolumeExported(collective, members.length, message_size_mb),
    });
  }

  return groups;
}

/**
 * 计算集合操作通信量 (导出版本)
 */
function calculateCollectiveVolumeExported(
  collective: CollectiveType,
  groupSize: number,
  dataSize: number
): number {
  if (groupSize <= 1) return 0;
  switch (collective) {
    case 'AllReduce':
      return 2 * (groupSize - 1) / groupSize * dataSize;
    case 'AllGather':
    case 'ReduceScatter':
      return (groupSize - 1) / groupSize * dataSize;
    case 'AllToAll':
      return (groupSize - 1) * dataSize / groupSize;
    case 'P2P':
      return dataSize;
    default:
      return 2 * (groupSize - 1) / groupSize * dataSize;
  }
}

// ============================================
// 图和路由算法
// ============================================

type Graph = Map<string, Set<string>>;

/**
 * 从拓扑连接构建无向图
 */
function buildGraph(connections: ConnectionConfig[]): Graph {
  const graph: Graph = new Map();

  for (const conn of connections) {
    // 添加 source -> target
    if (!graph.has(conn.source)) {
      graph.set(conn.source, new Set());
    }
    graph.get(conn.source)!.add(conn.target);

    // 添加 target -> source (无向图)
    if (!graph.has(conn.target)) {
      graph.set(conn.target, new Set());
    }
    graph.get(conn.target)!.add(conn.source);
  }

  return graph;
}

/**
 * BFS 最短路径算法
 * 返回从 source 到 target 的路径节点列表
 */
function findShortestPath(graph: Graph, source: string, target: string): string[] | null {
  if (source === target) return [source];
  if (!graph.has(source) || !graph.has(target)) return null;

  const visited = new Set<string>();
  const queue: { node: string; path: string[] }[] = [{ node: source, path: [source] }];
  visited.add(source);

  while (queue.length > 0) {
    const { node, path } = queue.shift()!;
    const neighbors = graph.get(node);

    if (!neighbors) continue;

    for (const neighbor of neighbors) {
      if (neighbor === target) {
        return [...path, neighbor];
      }
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push({ node: neighbor, path: [...path, neighbor] });
      }
    }
  }

  return null; // 无法到达
}

/**
 * 获取路径上的所有边 (有序)
 */
function getPathEdges(path: string[]): Array<{ from: string; to: string }> {
  const edges: Array<{ from: string; to: string }> = [];
  for (let i = 0; i < path.length - 1; i++) {
    edges.push({ from: path[i], to: path[i + 1] });
  }
  return edges;
}

/**
 * 运行流量分析 (兼容旧接口)
 */
export function runTrafficAnalysis(
  topology: HierarchicalTopology,
  config: ParallelismConfig
): TrafficAnalysisResult {
  // 根据范围筛选 chips
  const chips = collectChipsInScope(topology, config);

  // 生成通信组
  const groups = generateCommunicationGroups(chips, config);

  // 计算链路流量
  const linkTraffics = calculateLinkTraffic(groups, topology);

  // 计算节点负载
  const nodeLoads = calculateNodeLoads(linkTraffics);

  // 计算汇总信息
  const totalTraffic = linkTraffics.reduce((sum, lt) => sum + lt.traffic_mb, 0);
  const maxLinkTraffic = Math.max(...linkTraffics.map(lt => lt.traffic_mb), 0);
  const avgUtilization = linkTraffics.length > 0
    ? linkTraffics.reduce((sum, lt) => sum + lt.bandwidth_utilization, 0) / linkTraffics.length
    : 0;

  // 找出瓶颈链路 (利用率 > 80%)
  const bottleneckLinks = linkTraffics
    .filter(lt => lt.bandwidth_utilization > 0.8)
    .map(lt => `${lt.source}->${lt.target}`);

  return {
    configs: [],
    groups,
    link_traffic: linkTraffics,
    node_loads: nodeLoads,
    summary: {
      total_traffic_mb: totalTraffic,
      max_link_traffic_mb: maxLinkTraffic,
      avg_bandwidth_utilization: avgUtilization,
      bottleneck_links: bottleneckLinks,
      config_contributions: [],
    },
  };
}

/**
 * 运行多配置流量分析
 */
export function runMultiConfigAnalysis(
  topology: HierarchicalTopology,
  configs: TrafficConfigItem[]
): TrafficAnalysisResult {
  if (configs.length === 0) {
    return {
      configs: [],
      groups: [],
      link_traffic: [],
      node_loads: [],
      summary: {
        total_traffic_mb: 0,
        max_link_traffic_mb: 0,
        avg_bandwidth_utilization: 0,
        bottleneck_links: [],
        config_contributions: [],
      },
    };
  }

  // 收集所有配置的通信组
  const allGroups: CommunicationGroup[] = [];
  const configTrafficMap = new Map<string, number>(); // 记录每个配置的流量

  for (const config of configs) {
    // 根据配置范围收集 chips
    const chips = collectChipsFromConfigItem(topology, config);

    // 生成通信组
    const groups = generateGroupsFromConfigItem(chips, config);
    allGroups.push(...groups);

    // 计算该配置的总流量
    const configTraffic = groups.reduce((sum, g) => sum + g.data_volume_mb, 0);
    configTrafficMap.set(config.id, configTraffic);
  }

  // 计算链路流量（合并所有配置）
  const linkTraffics = calculateLinkTraffic(allGroups, topology);

  // 计算节点负载
  const nodeLoads = calculateNodeLoads(linkTraffics);

  // 计算汇总信息
  const totalTraffic = linkTraffics.reduce((sum, lt) => sum + lt.traffic_mb, 0);
  const maxLinkTraffic = Math.max(...linkTraffics.map(lt => lt.traffic_mb), 0);
  const avgUtilization = linkTraffics.length > 0
    ? linkTraffics.reduce((sum, lt) => sum + lt.bandwidth_utilization, 0) / linkTraffics.length
    : 0;

  // 找出瓶颈链路 (利用率 > 80%)
  const bottleneckLinks = linkTraffics
    .filter(lt => lt.bandwidth_utilization > 0.8)
    .map(lt => `${lt.source}->${lt.target}`);

  // 计算各配置的流量贡献
  const allConfigTraffic = Array.from(configTrafficMap.values()).reduce((a, b) => a + b, 0);
  const config_contributions = configs.map(c => ({
    config_id: c.id,
    config_name: c.name,
    traffic_mb: configTrafficMap.get(c.id) || 0,
    percentage: allConfigTraffic > 0 ? (configTrafficMap.get(c.id) || 0) / allConfigTraffic : 0,
  }));

  return {
    configs,
    groups: allGroups,
    link_traffic: linkTraffics,
    node_loads: nodeLoads,
    summary: {
      total_traffic_mb: totalTraffic,
      max_link_traffic_mb: maxLinkTraffic,
      avg_bandwidth_utilization: avgUtilization,
      bottleneck_links: bottleneckLinks,
      config_contributions,
    },
  };
}

// ============================================
// 辅助函数
// ============================================

/**
 * 颜色插值 (用于热力图)
 */
export function interpolateColor(color1: string, color2: string, t: number): string {
  const r1 = parseInt(color1.slice(1, 3), 16);
  const g1 = parseInt(color1.slice(3, 5), 16);
  const b1 = parseInt(color1.slice(5, 7), 16);
  const r2 = parseInt(color2.slice(1, 3), 16);
  const g2 = parseInt(color2.slice(3, 5), 16);
  const b2 = parseInt(color2.slice(5, 7), 16);

  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);

  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

/**
 * 根据带宽利用率获取热力图颜色
 * 绿色 (低) -> 黄色 (中) -> 红色 (高)
 */
export function getHeatmapColor(utilization: number): string {
  if (utilization < 0.5) {
    return interpolateColor('#52c41a', '#faad14', utilization * 2);
  } else {
    return interpolateColor('#faad14', '#f5222d', (utilization - 0.5) * 2);
  }
}

/**
 * 根据流量大小获取线宽 (2-8px)
 */
export function getTrafficLineWidth(traffic: number, maxTraffic: number): number {
  if (maxTraffic <= 0) return 2;
  return 2 + (traffic / maxTraffic) * 6;
}
