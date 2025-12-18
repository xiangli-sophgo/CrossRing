import { useEffect, useRef, useCallback } from 'react'

export interface GlobalTaskUpdate {
  type: 'task_update' | 'heartbeat'
  task_id: string
  status: string
  progress: number
  message: string
  experiment_name?: string
  current_file?: string
}

interface UseGlobalTaskWebSocketOptions {
  onTaskUpdate: (update: GlobalTaskUpdate) => void
  enabled?: boolean
}

/**
 * 全局任务 WebSocket Hook
 * 订阅所有任务的状态变化（不包含 sim_details 的高频更新）
 */
export const useGlobalTaskWebSocket = (options: UseGlobalTaskWebSocketOptions) => {
  const { onTaskUpdate, enabled = true } = options
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)
  const onTaskUpdateRef = useRef(onTaskUpdate)

  // 保持回调引用最新
  useEffect(() => {
    onTaskUpdateRef.current = onTaskUpdate
  }, [onTaskUpdate])

  const connect = useCallback(() => {
    if (!enabled) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    // 开发环境直接连接后端，生产环境使用相对路径
    const host = import.meta.env.DEV ? 'localhost:8000' : window.location.host
    const ws = new WebSocket(`${protocol}//${host}/api/simulation/ws/global`)

    ws.onopen = () => {
      console.log('Global WebSocket connected')
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as GlobalTaskUpdate
        if (data.type === 'heartbeat') return
        onTaskUpdateRef.current(data)
      } catch (e) {
        console.error('Failed to parse global WebSocket message:', e)
      }
    }

    ws.onerror = (error) => {
      console.error('Global WebSocket error:', error)
    }

    ws.onclose = (event) => {
      console.log(`Global WebSocket closed: code=${event.code}, reason=${event.reason}, wasClean=${event.wasClean}`)
      wsRef.current = null
      // 非正常关闭时自动重连（3秒后）
      if (event.code !== 1000 && enabled) {
        reconnectTimeoutRef.current = window.setTimeout(connect, 3000)
      }
    }

    wsRef.current = ws
  }, [enabled])

  useEffect(() => {
    connect()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close(1000) // 正常关闭
        wsRef.current = null
      }
    }
  }, [connect])

  return { ws: wsRef.current }
}
