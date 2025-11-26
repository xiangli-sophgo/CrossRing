// IP挂载相关类型定义

export interface IPMount {
  node_id: number
  ip_type: string
  topology: string
  position: {
    row: number
    col: number
  }
}

export interface IPMountRequest {
  node_ids: number[]
  ip_type: string
  topology: string
}

export interface BatchMountRequest {
  node_range: string
  ip_type_prefix: string
  topology: string
}

export interface IPMountResponse {
  success: boolean
  message: string
  mounted_ips: IPMount[]
}

export interface IPMountListResponse {
  topology: string
  mounts: IPMount[]
  total: number
}
