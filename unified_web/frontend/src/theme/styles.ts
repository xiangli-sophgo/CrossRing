/**
 * 公共样式常量
 */
import type { CSSProperties } from 'react'

// 卡片样式
export const cardStyle: CSSProperties = {
  borderRadius: 12,
  boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.03), 0 1px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px 0 rgba(0, 0, 0, 0.02)',
}

export const cardHoverStyle: CSSProperties = {
  ...cardStyle,
  transition: 'box-shadow 0.3s ease',
}

// 统计卡片样式
export const statCardStyle: CSSProperties = {
  ...cardStyle,
  cursor: 'pointer',
}

// Flex布局
export const flexCenter: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}

export const flexBetween: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
}

export const flexStart: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'flex-start',
}

export const flexColumn: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
}

// 间距
export const spacing = {
  xs: 8,
  sm: 12,
  md: 16,
  lg: 24,
  xl: 32,
}

// 页面内容区样式
export const pageContainerStyle: CSSProperties = {
  padding: 24,
  minHeight: 'calc(100vh - 64px)',
}

// 页面头部样式
export const pageHeaderStyle: CSSProperties = {
  marginBottom: 24,
}

// 工具栏样式
export const toolbarStyle: CSSProperties = {
  ...flexBetween,
  marginBottom: 16,
  padding: '12px 0',
}

// 空白状态容器
export const emptyContainerStyle: CSSProperties = {
  ...flexCenter,
  minHeight: 300,
  background: '#fafafa',
  borderRadius: 8,
}

// 图标按钮背景样式
export const iconButtonBgStyle = (color: string): CSSProperties => ({
  width: 48,
  height: 48,
  borderRadius: 12,
  backgroundColor: color,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
})

// 渐变背景
export const gradientBg = {
  primary: 'linear-gradient(135deg, #1677ff 0%, #4096ff 100%)',
  success: 'linear-gradient(135deg, #52c41a 0%, #73d13d 100%)',
  warning: 'linear-gradient(135deg, #faad14 0%, #ffc53d 100%)',
  error: 'linear-gradient(135deg, #ff4d4f 0%, #ff7875 100%)',
  purple: 'linear-gradient(135deg, #722ed1 0%, #9254de 100%)',
}

// 响应式断点
export const breakpoints = {
  xs: 480,
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200,
  xxl: 1600,
}
