import { useState, useCallback, useMemo } from 'react'
import {
  ViewState,
  ViewLevel,
  BreadcrumbItem,
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
} from '../types'

// 根据路径深度获取层级
// depth 0 = 数据中心顶层(显示所有Pod)
// depth 1 = Pod内部(显示该Pod的所有Rack)
// depth 2 = Rack内部(显示该Rack的所有Board)
// depth 3+ = Board内部(显示该Board的所有Chip)
function getLevelFromDepth(depth: number): ViewLevel {
  if (depth === 0) return 'pod'      // 顶层
  if (depth === 1) return 'pod'      // Pod内部，仍是pod级别视图
  if (depth === 2) return 'rack'     // Rack内部
  return 'board'                      // Board内部
}

// 从拓扑数据中查找节点
function findNode(
  topology: HierarchicalTopology,
  path: string[]
): PodConfig | RackConfig | BoardConfig | null {
  if (path.length === 0) return null

  const podId = path[0]
  const pod = topology.pods.find(p => p.id === podId)
  if (!pod || path.length === 1) return pod || null

  const rackId = path[1]
  const rack = pod.racks.find(r => r.id === rackId)
  if (!rack || path.length === 2) return rack || null

  const boardId = path[2]
  const board = rack.boards.find(b => b.id === boardId)
  return board || null
}

// 获取节点标签
function getNodeLabel(
  topology: HierarchicalTopology,
  path: string[]
): string {
  const node = findNode(topology, path)
  if (!node) return path[path.length - 1]
  return node.label
}

export interface ViewNavigationReturn {
  viewState: ViewState
  breadcrumbs: BreadcrumbItem[]
  navigateTo: (nodeId: string) => void
  navigateToPod: (podId: string) => void
  navigateToRack: (podId: string, rackId: string) => void
  navigateBack: () => void
  navigateToBreadcrumb: (index: number) => void
  canGoBack: boolean
  currentPod: PodConfig | null
  currentRack: RackConfig | null
  currentBoard: BoardConfig | null
}

export function useViewNavigation(
  topology: HierarchicalTopology | null
): ViewNavigationReturn {
  const [viewState, setViewState] = useState<ViewState>({
    level: 'pod',
    path: [],
    selectedNode: undefined,
  })

  // 生成面包屑
  const breadcrumbs = useMemo(() => {
    const items: BreadcrumbItem[] = [
      { level: 'pod', id: 'root', label: '数据中心' }
    ]

    if (!topology || viewState.path.length === 0) return items

    // path[0] = podId, path[1] = rackId, path[2] = boardId
    const podId = viewState.path[0]
    const pod = topology.pods.find(p => p.id === podId)
    if (pod) {
      items.push({ level: 'pod', id: podId, label: pod.label })
    }

    if (viewState.path.length >= 2 && pod) {
      const rackId = viewState.path[1]
      const rack = pod.racks.find(r => r.id === rackId)
      if (rack) {
        items.push({ level: 'rack', id: rackId, label: rack.label })
      }
    }

    if (viewState.path.length >= 3 && pod) {
      const rackId = viewState.path[1]
      const rack = pod.racks.find(r => r.id === rackId)
      if (rack) {
        const boardId = viewState.path[2]
        const board = rack.boards.find(b => b.id === boardId)
        if (board) {
          items.push({ level: 'board', id: boardId, label: board.label })
        }
      }
    }

    return items
  }, [viewState.path, topology])

  // 导航到Pod内部
  const navigateToPod = useCallback((podId: string) => {
    setViewState({
      level: 'pod',
      path: [podId],
      selectedNode: undefined,
    })
  }, [])

  // 导航到子层级（通用）
  const navigateTo = useCallback((nodeId: string) => {
    setViewState(prev => {
      const newPath = [...prev.path, nodeId]
      const newLevel = getLevelFromDepth(newPath.length)
      return {
        level: newLevel,
        path: newPath,
        selectedNode: undefined,
      }
    })
  }, [])

  // 直接导航到Rack (从Pod视图)
  const navigateToRack = useCallback((podId: string, rackId: string) => {
    setViewState({
      level: 'rack',
      path: [podId, rackId],
      selectedNode: undefined,
    })
  }, [])

  // 返回上一级
  const navigateBack = useCallback(() => {
    setViewState(prev => {
      if (prev.path.length === 0) return prev
      const newPath = prev.path.slice(0, -1)
      const newLevel = getLevelFromDepth(newPath.length)
      return {
        level: newLevel,
        path: newPath,
        selectedNode: undefined,
      }
    })
  }, [])

  // 通过面包屑导航
  const navigateToBreadcrumb = useCallback((index: number) => {
    setViewState(prev => {
      if (index === 0) {
        return { level: 'pod', path: [], selectedNode: undefined }
      }
      const newPath = prev.path.slice(0, index)
      const newLevel = getLevelFromDepth(newPath.length)
      return {
        level: newLevel,
        path: newPath,
        selectedNode: undefined,
      }
    })
  }, [])

  // 获取当前Pod
  const currentPod = useMemo(() => {
    if (!topology || viewState.path.length === 0) return null
    return topology.pods.find(p => p.id === viewState.path[0]) || null
  }, [topology, viewState.path])

  // 获取当前Rack
  const currentRack = useMemo(() => {
    if (!currentPod || viewState.path.length < 2) return null
    const rackIdInPath = viewState.path[1]
    return currentPod.racks.find(r => r.id === rackIdInPath) || null
  }, [currentPod, viewState.path])

  // 获取当前Board
  const currentBoard = useMemo(() => {
    if (!currentRack || viewState.path.length < 3) return null
    return currentRack.boards.find(b => b.id === viewState.path[2]) || null
  }, [currentRack, viewState.path])

  const canGoBack = viewState.path.length > 0

  return {
    viewState,
    breadcrumbs,
    navigateTo,
    navigateToPod,
    navigateToRack,
    navigateBack,
    navigateToBreadcrumb,
    canGoBack,
    currentPod,
    currentRack,
    currentBoard,
  }
}
