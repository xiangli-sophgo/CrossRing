/**
 * 参数遍历配置面板
 */
import React from 'react'
import {
  Row, Col, InputNumber, Button, Select, Input, Tag, Typography,
  Collapse, Alert, Empty, Space, Divider, Tooltip
} from 'antd'
import { ThunderboltOutlined, MinusSquareOutlined, SaveOutlined } from '@ant-design/icons'
import { CONFIG_TOOLTIPS, BIND_GROUP_COLORS, getNextBindGroupId } from '../helpers'
import type { SweepParam, SavedSweepConfig } from '../helpers'

const { Text } = Typography
const { Option } = Select

interface SweepConfigPanelProps {
  sweepParams: SweepParam[]
  configValues: Record<string, any>
  selectedFilesCount: number
  totalCombinations: number
  availableParams: string[]
  savedSweepConfigs: SavedSweepConfig[]
  sweepConfigName: string
  existingBindGroups: string[]        // 当前存在的绑定组
  bindingErrors: string[]             // 绑定验证错误
  onAddSweepParam: (key: string) => void
  onUpdateSweepParam: (index: number, field: 'start' | 'end' | 'step', value: number | null) => void
  onRemoveSweepParam: (index: number) => void
  onUpdateBindGroup: (index: number, groupId: string | undefined) => void
  onSaveSweepConfig: (name: string) => void
  onLoadSweepConfig: (name: string) => void
  onDeleteSweepConfig: (name: string) => void
  onSweepConfigNameChange: (name: string) => void
}

export const SweepConfigPanel: React.FC<SweepConfigPanelProps> = ({
  sweepParams,
  selectedFilesCount,
  totalCombinations,
  availableParams,
  savedSweepConfigs,
  sweepConfigName,
  existingBindGroups,
  bindingErrors,
  onAddSweepParam,
  onUpdateSweepParam,
  onRemoveSweepParam,
  onUpdateBindGroup,
  onSaveSweepConfig,
  onLoadSweepConfig,
  onDeleteSweepConfig,
  onSweepConfigNameChange,
  configValues,
}) => {
  return (
    <Collapse
      size="small"
      style={{ marginBottom: 16 }}
      defaultActiveKey={[]}
      items={[{
        key: 'sweep',
        label: (
          <Space>
            <span>参数遍历配置</span>
            {sweepParams.length > 0 && (
              <Text type="secondary">
                {sweepParams.map(p => `${p.key}[${p.values.length}]`).join(' × ')}
              </Text>
            )}
          </Space>
        ),
        children: (
          <div>
            {/* 表头说明 */}
            {sweepParams.length > 0 && (
              <Row gutter={16} style={{ marginBottom: 8 }}>
                <Col span={5}><Text type="secondary" style={{ fontSize: 12 }}>参数名</Text></Col>
                <Col span={4}><Text type="secondary" style={{ fontSize: 12 }}>起始值</Text></Col>
                <Col span={4}><Text type="secondary" style={{ fontSize: 12 }}>结束值</Text></Col>
                <Col span={3}><Text type="secondary" style={{ fontSize: 12 }}>步长</Text></Col>
                <Col span={2}><Text type="secondary" style={{ fontSize: 12 }}>数量</Text></Col>
                <Col span={4}><Text type="secondary" style={{ fontSize: 12 }}>绑定组</Text></Col>
                <Col span={2}></Col>
              </Row>
            )}

            {/* 已添加的遍历参数 */}
            {sweepParams.map((param, idx) => (
              <Row
                key={param.key}
                gutter={16}
                style={{
                  marginBottom: 12,
                  padding: '4px 8px',
                  borderRadius: 4,
                  backgroundColor: param.bindGroupId ? BIND_GROUP_COLORS[param.bindGroupId] || '#f5f5f5' : undefined,
                }}
                align="middle"
              >
                <Col span={5}>
                  <Tooltip title={CONFIG_TOOLTIPS[param.key] || param.key}>
                    <Text style={{ cursor: 'help', fontSize: 13 }}>{param.key}</Text>
                  </Tooltip>
                </Col>
                <Col span={4}>
                  <InputNumber
                    value={param.start}
                    onChange={(v) => onUpdateSweepParam(idx, 'start', v)}
                    placeholder="起始值"
                    style={{ width: '100%' }}
                  />
                </Col>
                <Col span={4}>
                  <InputNumber
                    value={param.end}
                    onChange={(v) => onUpdateSweepParam(idx, 'end', v)}
                    placeholder="结束值"
                    style={{ width: '100%' }}
                  />
                </Col>
                <Col span={3}>
                  <InputNumber
                    value={param.step}
                    onChange={(v) => onUpdateSweepParam(idx, 'step', v)}
                    placeholder="步长"
                    min={0.001}
                    style={{ width: '100%' }}
                  />
                </Col>
                <Col span={2}>
                  <Tag color="green">{param.values.length}个</Tag>
                </Col>
                <Col span={4}>
                  <Select
                    value={param.bindGroupId || ''}
                    onChange={(v) => onUpdateBindGroup(idx, v || undefined)}
                    style={{ width: '100%' }}
                    size="small"
                  >
                    <Option value="">无绑定</Option>
                    {existingBindGroups.map(g => (
                      <Option key={g} value={g}>
                        <span style={{ display: 'inline-block', width: 12, height: 12, backgroundColor: BIND_GROUP_COLORS[g], marginRight: 6, borderRadius: 2 }} />
                        组{g}
                      </Option>
                    ))}
                    <Option value={getNextBindGroupId(existingBindGroups)}>
                      + 新建组{getNextBindGroupId(existingBindGroups)}
                    </Option>
                  </Select>
                </Col>
                <Col span={2}>
                  <Button
                    type="text"
                    danger
                    icon={<MinusSquareOutlined />}
                    onClick={() => onRemoveSweepParam(idx)}
                  />
                </Col>
              </Row>
            ))}

            {/* 添加参数下拉框 */}
            <Select
              placeholder="+ 添加遍历参数"
              style={{ width: 350, marginTop: 8 }}
              onSelect={(key: string) => {
                onAddSweepParam(key)
              }}
              showSearch
              optionFilterProp="label"
              options={availableParams.map(key => ({
                value: key,
                label: `${key} (当前: ${configValues[key]})`,
              }))}
            />

            {/* 绑定验证错误提示 */}
            {bindingErrors.length > 0 && (
              <Alert
                type="error"
                style={{ marginTop: 16 }}
                message="绑定配置错误"
                description={
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {bindingErrors.map((err, i) => <li key={i}>{err}</li>)}
                  </ul>
                }
              />
            )}

            {/* 预览信息 */}
            {totalCombinations > 0 && bindingErrors.length === 0 && (
              <Alert
                type="info"
                style={{ marginTop: 16 }}
                message={
                  <Space direction="vertical" size={0}>
                    <Text>预计生成 <strong>{totalCombinations}</strong> 组参数组合</Text>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      每组执行 {selectedFilesCount} 个流量文件，共 {totalCombinations * selectedFilesCount} 个仿真任务
                    </Text>
                  </Space>
                }
              />
            )}

            {/* 保存/加载配置 */}
            <Divider style={{ margin: '16px 0 12px' }} />
            <Row gutter={8} align="middle">
              <Col flex="auto">
                <Space.Compact style={{ width: '100%' }}>
                  <Input
                    placeholder="输入配置名称"
                    value={sweepConfigName}
                    onChange={(e) => onSweepConfigNameChange(e.target.value)}
                    onPressEnter={() => onSaveSweepConfig(sweepConfigName)}
                    style={{ width: 200 }}
                  />
                  <Button
                    icon={<SaveOutlined />}
                    onClick={() => onSaveSweepConfig(sweepConfigName)}
                    disabled={sweepParams.length === 0}
                  >
                    保存
                  </Button>
                </Space.Compact>
              </Col>
              <Col>
                <Select
                  placeholder="加载已保存配置"
                  style={{ width: 200 }}
                  value={undefined}
                  onChange={onLoadSweepConfig}
                  dropdownRender={(menu) => (
                    <>
                      {menu}
                      {savedSweepConfigs.length === 0 && (
                        <div style={{ padding: 8, textAlign: 'center' }}>
                          <Text type="secondary">暂无保存的配置</Text>
                        </div>
                      )}
                    </>
                  )}
                >
                  {savedSweepConfigs.map(config => (
                    <Option key={config.name} value={config.name}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span>{config.name} ({config.params.length}个参数)</span>
                        <Button
                          type="text"
                          size="small"
                          danger
                          icon={<MinusSquareOutlined />}
                          onClick={(e) => {
                            e.stopPropagation()
                            onDeleteSweepConfig(config.name)
                          }}
                        />
                      </div>
                    </Option>
                  ))}
                </Select>
              </Col>
            </Row>

            {sweepParams.length === 0 && (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="请从上方下拉框选择要遍历的参数"
                style={{ marginTop: 16 }}
              />
            )}
          </div>
        )
      }]}
    />
  )
}
