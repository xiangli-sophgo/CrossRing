/**
 * Ant Design 主题配置
 */
import type { ThemeConfig } from 'antd'
import { primaryColor } from './colors'

export const theme: ThemeConfig = {
  token: {
    // 品牌色
    colorPrimary: primaryColor,
    colorSuccess: '#52c41a',
    colorWarning: '#faad14',
    colorError: '#ff4d4f',
    colorInfo: primaryColor,

    // 圆角
    borderRadius: 8,
    borderRadiusLG: 12,
    borderRadiusSM: 6,

    // 字体
    fontFamily: `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif`,
    fontSize: 14,

    // 间距
    padding: 16,
    paddingLG: 24,
    paddingSM: 12,
    paddingXS: 8,

    // 阴影
    boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.03), 0 1px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px 0 rgba(0, 0, 0, 0.02)',
    boxShadowSecondary: '0 6px 16px 0 rgba(0, 0, 0, 0.08), 0 3px 6px -4px rgba(0, 0, 0, 0.12), 0 9px 28px 8px rgba(0, 0, 0, 0.05)',
  },
  components: {
    // 布局
    Layout: {
      headerBg: '#fff',
      headerHeight: 64,
      siderBg: '#fff',
      bodyBg: '#f5f5f5',
    },
    // 菜单
    Menu: {
      itemSelectedBg: '#e6f4ff',
      itemSelectedColor: primaryColor,
      itemHoverBg: '#f5f5f5',
      itemBorderRadius: 8,
      itemMarginInline: 8,
    },
    // 卡片
    Card: {
      borderRadiusLG: 12,
      paddingLG: 24,
    },
    // 按钮
    Button: {
      borderRadius: 6,
      controlHeight: 36,
      controlHeightLG: 44,
      controlHeightSM: 28,
    },
    // 表格
    Table: {
      borderRadius: 8,
      headerBg: '#fafafa',
      rowHoverBg: '#f5f5f5',
    },
    // 表单
    Form: {
      itemMarginBottom: 20,
    },
    // 输入框
    Input: {
      borderRadius: 6,
      controlHeight: 36,
    },
    // 选择器
    Select: {
      borderRadius: 6,
      controlHeight: 36,
    },
    // 标签
    Tag: {
      borderRadiusSM: 4,
    },
    // 进度条
    Progress: {
      circleTextFontSize: '16px',
    },
    // 面包屑
    Breadcrumb: {
      separatorMargin: 8,
    },
    // 单选按钮
    Radio: {
      buttonSolidCheckedBg: primaryColor,
      buttonSolidCheckedColor: '#fff',
      buttonSolidCheckedHoverBg: '#4096ff',
    },
  },
}

export * from './colors'
