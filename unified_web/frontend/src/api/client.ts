/**
 * 统一的 API 客户端配置
 *
 * 所有 API 模块应该使用这个客户端，确保配置一致
 */
import axios, { type AxiosError, type AxiosResponse, type InternalAxiosRequestConfig } from 'axios'
import { message } from 'antd'

// API 基础 URL - 通过环境变量配置，默认使用相对路径
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

// 请求超时时间（毫秒）
const REQUEST_TIMEOUT = 60000

// 创建 axios 实例
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: REQUEST_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // 可以在这里添加认证token等
    return config
  },
  (error: AxiosError) => {
    console.error('请求配置错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response
  },
  (error: AxiosError) => {
    // 统一错误处理
    if (error.response) {
      const status = error.response.status
      const data = error.response.data as { detail?: string; message?: string }

      switch (status) {
        case 400:
          message.error(data.detail || '请求参数错误')
          break
        case 401:
          message.error('未授权，请重新登录')
          break
        case 403:
          message.error(data.detail || '无权访问该资源')
          break
        case 404:
          message.error(data.detail || '请求的资源不存在')
          break
        case 413:
          message.error(data.detail || '请求体过大')
          break
        case 500:
          message.error(data.detail || '服务器内部错误')
          break
        case 502:
          message.error('服务器暂时不可用')
          break
        case 503:
          message.error('服务正在维护中')
          break
        default:
          message.error(data.detail || `请求失败 (${status})`)
      }
    } else if (error.code === 'ECONNABORTED') {
      message.error('请求超时，请稍后重试')
    } else if (error.message === 'Network Error') {
      message.error('网络连接失败，请检查网络')
    } else {
      message.error('请求失败，请稍后重试')
    }

    return Promise.reject(error)
  }
)

export default apiClient

// 导出类型
export type { AxiosError, AxiosResponse }
