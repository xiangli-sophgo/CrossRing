/**
 * 统一颜色系统 - 高级浅色主题 (Soft Blue-Gray)
 * 设计灵感：清晨薄雾、精致科技、北欧简约
 */

// ========== 核心品牌色 ==========
export const primaryColor = '#4f6ef7'        // 优雅蓝紫
export const primaryColorHover = '#6b85f9'
export const primaryColorActive = '#3d5bd9'
export const primaryBg = '#eef2ff'

// 强调色 - 青绿色点缀
export const accentColor = '#10b981'
export const accentColorHover = '#34d399'

// ========== 状态颜色 ==========
export const successColor = '#10b981'  // 翡翠绿
export const warningColor = '#f59e0b'  // 琥珀色
export const errorColor = '#ef4444'    // 珊瑚红
export const infoColor = '#4f6ef7'

// ========== 背景色层级 ==========
// 主背景 - 淡蓝灰，带一点温暖
export const bgLayout = '#f4f7fb'
// 侧边栏 - 稍深的蓝灰
export const bgSider = '#eaeff6'
// 卡片/容器 - 纯净白
export const bgContainer = '#ffffff'
// 悬浮层 - 纯白
export const bgElevated = '#ffffff'
// 悬停背景
export const bgHover = '#f0f4fa'

// ========== 文字色 ==========
export const textColor = '#1e293b'              // 深蓝灰
export const textColorSecondary = '#64748b'     // 中灰蓝
export const textColorTertiary = '#94a3b8'      // 浅灰蓝
export const textColorQuaternary = '#cbd5e1'

// ========== 边框 ==========
export const borderColor = '#e2e8f0'
export const borderColorLight = '#f1f5f9'
export const borderColorDark = '#cbd5e1'

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
  pending: { color: '#64748b', bg: '#f1f5f9' },
  running: { color: '#4f6ef7', bg: '#eef2ff' },
  completed: { color: '#10b981', bg: '#ecfdf5' },
  failed: { color: '#ef4444', bg: '#fef2f2' },
  cancelled: { color: '#f59e0b', bg: '#fffbeb' },
  interrupted: { color: '#f97316', bg: '#fff7ed' },
}

// ========== 实验类型颜色 ==========
export const experimentTypeColors: Record<string, string> = {
  kcin: '#4f6ef7',
  dcin: '#a855f7',
}

// ========== 图表颜色序列 ==========
export const chartColors = [
  '#4f6ef7',  // 主蓝紫
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
