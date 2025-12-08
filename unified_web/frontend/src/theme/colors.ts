/**
 * 统一颜色系统
 */

// 主色调
export const primaryColor = '#1677ff'
export const primaryColorHover = '#4096ff'
export const primaryColorActive = '#0958d9'
export const primaryBg = '#e6f4ff'

// 状态颜色
export const successColor = '#52c41a'
export const warningColor = '#faad14'
export const errorColor = '#ff4d4f'
export const infoColor = '#1677ff'

// 中性色
export const textColor = 'rgba(0, 0, 0, 0.88)'
export const textColorSecondary = 'rgba(0, 0, 0, 0.65)'
export const textColorTertiary = 'rgba(0, 0, 0, 0.45)'
export const textColorQuaternary = 'rgba(0, 0, 0, 0.25)'

// 边框和分割线
export const borderColor = '#d9d9d9'
export const borderColorSplit = '#f0f0f0'

// 背景色
export const bgColorBase = '#ffffff'
export const bgColorLayout = '#f5f5f5'
export const bgColorContainer = '#ffffff'
export const bgColorElevated = '#ffffff'

// IP类型颜色 (用于拓扑图节点)
export const ipTypeColors: Record<string, { bg: string; border: string; text: string }> = {
  gdma: { bg: '#e6f7ff', border: '#1890ff', text: '#096dd9' },
  sdma: { bg: '#e6f7ff', border: '#1890ff', text: '#096dd9' },
  cdma: { bg: '#f6ffed', border: '#52c41a', text: '#389e0d' },
  npu: { bg: '#f9f0ff', border: '#722ed1', text: '#531dab' },
  ddr: { bg: '#fff1f0', border: '#ff4d4f', text: '#cf1322' },
  l2m: { bg: '#fff1f0', border: '#ff4d4f', text: '#cf1322' },
  pcie: { bg: '#fff7e6', border: '#fa8c16', text: '#d46b08' },
  eth: { bg: '#f6ffed', border: '#52c41a', text: '#389e0d' },
  dcin: { bg: '#e6fffb', border: '#13c2c2', text: '#08979c' },
  default: { bg: '#fafafa', border: '#d9d9d9', text: '#595959' },
}

// 状态标签颜色映射
export const statusColors: Record<string, { color: string; bg: string }> = {
  pending: { color: '#8c8c8c', bg: '#fafafa' },
  running: { color: '#1677ff', bg: '#e6f4ff' },
  completed: { color: '#52c41a', bg: '#f6ffed' },
  failed: { color: '#ff4d4f', bg: '#fff2f0' },
  cancelled: { color: '#faad14', bg: '#fffbe6' },
  interrupted: { color: '#fa8c16', bg: '#fff7e6' },
}

// 实验类型颜色
export const experimentTypeColors: Record<string, string> = {
  kcin: '#1677ff',
  dcin: '#722ed1',
}

// 图表颜色序列
export const chartColors = [
  '#1677ff',
  '#52c41a',
  '#722ed1',
  '#fa8c16',
  '#eb2f96',
  '#13c2c2',
  '#faad14',
  '#2f54eb',
]

// 获取IP类型颜色（带默认值）
export const getIPTypeColor = (ipType: string) => {
  const type = ipType.toLowerCase().replace(/_\d+$/, '')
  return ipTypeColors[type] || ipTypeColors.default
}
