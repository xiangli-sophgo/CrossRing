import React from 'react'
import {
  Collapse,
  Slider,
  Select,
  Switch,
  Button,
  Space,
  Typography,
  Divider,
  Tag,
} from 'antd'
import {
  ReloadOutlined,
  DownloadOutlined,
  EyeOutlined,
  EyeInvisibleOutlined,
} from '@ant-design/icons'
import { LevelConfig, LEVEL_NAMES, TOPOLOGY_TYPES, LEVEL_COLORS } from '../types'

const { Panel } = Collapse
const { Text, Title } = Typography

interface ConfigPanelProps {
  levels: LevelConfig[]
  showInterLevel: boolean
  onConfigChange: (levels: LevelConfig[]) => void
  onInterLevelChange: (show: boolean) => void
  onReset: () => void
}

export const ConfigPanel: React.FC<ConfigPanelProps> = ({
  levels,
  showInterLevel,
  onConfigChange,
  onInterLevelChange,
  onReset,
}) => {
  // 更新单个层级配置
  const updateLevel = (index: number, field: keyof LevelConfig, value: any) => {
    const newLevels = [...levels]
    newLevels[index] = { ...newLevels[index], [field]: value }
    onConfigChange(newLevels)
  }

  // 导出配置
  const handleExport = () => {
    const config = {
      levels,
      showInterLevel,
      exportTime: new Date().toISOString(),
    }
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'topology_config.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div>
      <Title level={5} style={{ marginBottom: 16 }}>层级配置</Title>

      <Collapse
        defaultActiveKey={levels.map((_, i) => i.toString())}
        style={{ marginBottom: 16 }}
      >
        {levels.map((level, index) => (
          <Panel
            key={index.toString()}
            header={
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Space>
                  <div style={{
                    width: 12,
                    height: 12,
                    borderRadius: 2,
                    background: LEVEL_COLORS[level.level],
                  }} />
                  <span>{LEVEL_NAMES[level.level] || level.level}</span>
                </Space>
                <Tag color={level.visible ? 'blue' : 'default'}>
                  {level.count} 节点
                </Tag>
              </div>
            }
            extra={
              <Switch
                size="small"
                checked={level.visible}
                onChange={(checked) => updateLevel(index, 'visible', checked)}
                checkedChildren={<EyeOutlined />}
                unCheckedChildren={<EyeInvisibleOutlined />}
                onClick={(_, e) => e.stopPropagation()}
              />
            }
          >
            <div style={{ padding: '8px 0' }}>
              <div style={{ marginBottom: 16 }}>
                <Text type="secondary" style={{ display: 'block', marginBottom: 8 }}>
                  节点数量
                </Text>
                <Slider
                  min={1}
                  max={16}
                  value={level.count}
                  onChange={(value) => updateLevel(index, 'count', value)}
                  marks={{
                    1: '1',
                    4: '4',
                    8: '8',
                    16: '16',
                  }}
                />
              </div>

              <div>
                <Text type="secondary" style={{ display: 'block', marginBottom: 8 }}>
                  拓扑类型
                </Text>
                <Select
                  style={{ width: '100%' }}
                  value={level.topology}
                  onChange={(value) => updateLevel(index, 'topology', value)}
                  options={TOPOLOGY_TYPES}
                />
              </div>
            </div>
          </Panel>
        ))}
      </Collapse>

      <Divider />

      <div style={{ marginBottom: 16 }}>
        <Text type="secondary" style={{ display: 'block', marginBottom: 8 }}>
          显示选项
        </Text>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Text>层间连接</Text>
          <Switch
            checked={showInterLevel}
            onChange={onInterLevelChange}
          />
        </div>
      </div>

      <Divider />

      <Space style={{ width: '100%' }} direction="vertical">
        <Button
          block
          icon={<DownloadOutlined />}
          onClick={handleExport}
        >
          导出配置
        </Button>
        <Button
          block
          icon={<ReloadOutlined />}
          onClick={onReset}
        >
          重置
        </Button>
      </Space>
    </div>
  )
}
