/**
 * 统一颜色系统 - Linear风格浅色主题
 * 设计灵感：Linear App - 极简冷灰调，高对比度
 */

// ========== 核心品牌色 ==========
export const primaryColor = '#2563EB'        // 冷蓝色
export const primaryColorHover = '#1D4ED8'
export const primaryColorActive = '#1E40AF'
export const primaryBg = 'rgba(37, 99, 235, 0.08)'

// 强调色
export const accentColor = '#10b981'
export const accentColorHover = '#34d399'

// ========== 状态颜色 ==========
export const successColor = '#10b981'  // 翡翠绿
export const warningColor = '#f59e0b'  // 琥珀色
export const errorColor = '#ef4444'    // 珊瑚红
export const infoColor = '#2563EB'

// ========== 背景色层级 ==========
// 主背景 - Linear风格冷灰
export const bgLayout = '#F7F7F7'
// 侧边栏
export const bgSider = '#EFEFEF'
// 卡片/容器 - 纯白
export const bgContainer = '#FFFFFF'
// 悬浮层 - 纯白
export const bgElevated = '#FFFFFF'
// 悬停背景
export const bgHover = 'rgba(0, 0, 0, 0.03)'

// ========== 文字色 ==========
export const textColor = '#1A1A1A'              // 深色
export const textColorSecondary = '#666666'     // 中灰
export const textColorTertiary = '#999999'      // 浅灰
export const textColorQuaternary = '#CCCCCC'

// ========== 边框 ==========
export const borderColor = '#E5E5E5'
export const borderColorLight = '#EFEFEF'
export const borderColorDark = '#D4D4D4'

// ========== 兼容旧代码 ==========
export const borderColorSplit = borderColor
export const bgColorBase = bgContainer
export const bgColorLayout = bgLayout
export const bgColorContainer = bgContainer
export const bgColorElevated = bgElevated

// ========== IP类型颜色 (拓扑图节点) ==========
export const ipTypeColors: Record<string, { bg: string; border: string; text: string }> = {
  gdma: { bg: '#eef2ff', border: '#6366f1', text: '#4338ca' },
  sdma: { bg: '#eef2ff', border: '#6366f1', text: '#4338ca' },
  cdma: { bg: '#ecfdf5', border: '#10b981', text: '#047857' },
  npu: { bg: '#faf5ff', border: '#a855f7', text: '#7c3aed' },
  ddr: { bg: '#fef2f2', border: '#ef4444', text: '#dc2626' },
  l2m: { bg: '#fef2f2', border: '#ef4444', text: '#dc2626' },
  pcie: { bg: '#fffbeb', border: '#f59e0b', text: '#d97706' },
  eth: { bg: '#ecfdf5', border: '#10b981', text: '#047857' },
  dcin: { bg: '#ecfeff', border: '#06b6d4', text: '#0891b2' },
  default: { bg: '#f8fafc', border: '#94a3b8', text: '#64748b' },
}

// ========== 状态标签颜色映射 ==========
export const statusColors: Record<string, { color: string; bg: string }> = {
  pending: { color: '#666666', bg: '#EFEFEF' },
  running: { color: '#2563EB', bg: 'rgba(37, 99, 235, 0.08)' },
  completed: { color: '#10b981', bg: 'rgba(16, 185, 129, 0.08)' },
  failed: { color: '#ef4444', bg: 'rgba(239, 68, 68, 0.08)' },
  cancelled: { color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.08)' },
  interrupted: { color: '#f97316', bg: 'rgba(249, 115, 22, 0.08)' },
}

// ========== 实验类型颜色 ==========
export const experimentTypeColors: Record<string, string> = {
  kcin: '#2563EB',
  dcin: '#a855f7',
}

// ========== 图表颜色序列 ==========
export const chartColors = [
  '#2563EB',  // 冷蓝色
  '#10b981',  // 翡翠绿
  '#a855f7',  // 紫色
  '#f59e0b',  // 琥珀
  '#ec4899',  // 粉红
  '#06b6d4',  // 青色
  '#f97316',  // 橙色
  '#8b5cf6',  // 紫罗兰
]

// ========== 工具函数 ==========
export const getIPTypeColor = (ipType: string) => {
  const type = ipType.toLowerCase().replace(/_\d+$/, '')
  return ipTypeColors[type] || ipTypeColors.default
}
