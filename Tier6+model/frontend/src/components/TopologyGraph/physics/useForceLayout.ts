import { useRef, useCallback, useEffect, useState } from 'react'
import { Node, Edge } from '../shared'
import { ForceLayoutManager, ForceNode, ForceLayoutOptions, DEFAULT_FORCE_OPTIONS } from '../layouts'

export interface UseForceLayoutOptions extends Partial<ForceLayoutOptions> {
  enabled: boolean  // 是否启用力导向布局
  onNodePositionsChange?: (nodes: ForceNode[]) => void  // 位置变化回调
}

export interface UseForceLayoutResult {
  // 状态
  isSimulating: boolean  // 是否正在模拟
  nodes: ForceNode[]     // 当前节点位置

  // 方法
  initialize: (nodes: Node[], edges: Edge[]) => void  // 初始化模拟
  start: () => void         // 开始模拟
  stop: () => void          // 停止模拟
  reheat: () => void        // 重新激活模拟

  // 拖拽相关
  onDragStart: (nodeId: string, x: number, y: number) => void
  onDrag: (nodeId: string, x: number, y: number) => void
  onDragEnd: (nodeId: string) => void

  // 更新选项
  updateOptions: (options: Partial<ForceLayoutOptions>) => void
}

export function useForceLayout(options: UseForceLayoutOptions): UseForceLayoutResult {
  const { enabled, onNodePositionsChange, ...forceOptions } = options

  const managerRef = useRef<ForceLayoutManager | null>(null)
  const [isSimulating, setIsSimulating] = useState(false)
  const [nodes, setNodes] = useState<ForceNode[]>([])

  // 创建或获取 manager
  const getManager = useCallback(() => {
    if (!managerRef.current) {
      managerRef.current = new ForceLayoutManager({
        ...DEFAULT_FORCE_OPTIONS,
        ...forceOptions,
      })
    }
    return managerRef.current
  }, [])

  // 初始化模拟
  const initialize = useCallback((inputNodes: Node[], edges: Edge[]) => {
    if (!enabled) return

    const manager = getManager()

    // 更新宽高选项
    const opts: Partial<ForceLayoutOptions> = { ...forceOptions }

    const initialNodes = manager.initialize(inputNodes, edges, opts)

    // 设置 tick 回调
    manager.setOnTick((updatedNodes) => {
      setNodes([...updatedNodes])
      if (onNodePositionsChange) {
        onNodePositionsChange(updatedNodes)
      }
    })

    // 设置结束回调
    manager.setOnEnd(() => {
      setIsSimulating(false)
    })

    setNodes(initialNodes)
    setIsSimulating(true)

    // 自动开始模拟
    manager.start(1.0)
  }, [enabled, getManager, onNodePositionsChange, forceOptions])

  // 开始模拟
  const start = useCallback(() => {
    if (!enabled) return
    const manager = managerRef.current
    if (manager) {
      setIsSimulating(true)
      manager.start()
    }
  }, [enabled])

  // 停止模拟
  const stop = useCallback(() => {
    const manager = managerRef.current
    if (manager) {
      manager.stop()
      setIsSimulating(false)
    }
  }, [])

  // 重新加热
  const reheat = useCallback(() => {
    if (!enabled) return
    const manager = managerRef.current
    if (manager) {
      setIsSimulating(true)
      manager.reheat()
    }
  }, [enabled])

  // 拖拽开始
  const onDragStart = useCallback((nodeId: string, x: number, y: number) => {
    if (!enabled) return
    const manager = managerRef.current
    if (manager) {
      manager.fixNode(nodeId, x, y)
      setIsSimulating(true)
    }
  }, [enabled])

  // 拖拽中
  const onDrag = useCallback((nodeId: string, x: number, y: number) => {
    if (!enabled) return
    const manager = managerRef.current
    if (manager) {
      manager.dragNode(nodeId, x, y)
    }
  }, [enabled])

  // 拖拽结束
  const onDragEnd = useCallback((nodeId: string) => {
    if (!enabled) return
    const manager = managerRef.current
    if (manager) {
      manager.releaseNode(nodeId)
    }
  }, [enabled])

  // 更新选项
  const updateOptions = useCallback((newOptions: Partial<ForceLayoutOptions>) => {
    const manager = managerRef.current
    if (manager) {
      manager.updateOptions(newOptions)
    }
  }, [])

  // 禁用时停止模拟
  useEffect(() => {
    if (!enabled && managerRef.current) {
      managerRef.current.stop()
      setIsSimulating(false)
    }
  }, [enabled])

  // 清理
  useEffect(() => {
    return () => {
      if (managerRef.current) {
        managerRef.current.destroy()
        managerRef.current = null
      }
    }
  }, [])

  return {
    isSimulating,
    nodes,
    initialize,
    start,
    stop,
    reheat,
    onDragStart,
    onDrag,
    onDragEnd,
    updateOptions,
  }
}
