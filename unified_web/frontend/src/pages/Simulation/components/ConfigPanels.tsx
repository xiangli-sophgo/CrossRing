/**
 * 配置面板组件集合 - ConfigLabel, KCINConfigPanel, DCINConfigPanel
 */
import React from 'react'
import { Row, Col, InputNumber, Switch, Select, Input, Button, Typography, Collapse, Tooltip } from 'antd'
import { CONFIG_TOOLTIPS } from '../helpers'

const { Text } = Typography
const { Option } = Select

// ============== ConfigLabel ==============

interface ConfigLabelProps {
  name: string
}

export const ConfigLabel: React.FC<ConfigLabelProps> = ({ name }) => (
  <Tooltip title={CONFIG_TOOLTIPS[name] || name}>
    <Text type="secondary" style={{ cursor: 'help' }}>{name}</Text>
  </Tooltip>
)

// ============== KCINConfigPanel ==============

interface KCINConfigPanelProps {
  configValues: Record<string, any>
  updateConfigValue: (key: string, value: any) => void
}

export const KCINConfigPanel: React.FC<KCINConfigPanelProps> = ({
  configValues,
  updateConfigValue,
}) => {
  return (
    <Collapse
      size="small"
      style={{ marginBottom: 16 }}
      items={[
        {
          key: 'basic',
          label: 'Basic Parameters',
          children: (
            <Row gutter={[16, 8]}>
              {configValues.FLIT_SIZE !== undefined && (
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="FLIT_SIZE" /></div>
                  <InputNumber value={configValues.FLIT_SIZE} onChange={(v) => updateConfigValue('FLIT_SIZE', v)} min={1} style={{ width: '100%' }} />
                </Col>
              )}
              {configValues.BURST !== undefined && (
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="BURST" /></div>
                  <InputNumber value={configValues.BURST} onChange={(v) => updateConfigValue('BURST', v)} min={1} style={{ width: '100%' }} />
                </Col>
              )}
              {configValues.NETWORK_FREQUENCY !== undefined && (
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="NETWORK_FREQUENCY" /></div>
                  <InputNumber value={configValues.NETWORK_FREQUENCY} onChange={(v) => updateConfigValue('NETWORK_FREQUENCY', v)} min={1} style={{ width: '100%' }} />
                </Col>
              )}
            </Row>
          ),
        },
        {
          key: 'buffer',
          label: 'Buffer Size',
          children: (
            <Row gutter={[16, 8]}>
              {configValues.RN_RDB_SIZE !== undefined && (
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="RN_RDB_SIZE" /></div>
                  <InputNumber value={configValues.RN_RDB_SIZE} onChange={(v) => updateConfigValue('RN_RDB_SIZE', v)} min={1} style={{ width: '100%' }} />
                </Col>
              )}
              {configValues.RN_WDB_SIZE !== undefined && (
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="RN_WDB_SIZE" /></div>
                  <InputNumber value={configValues.RN_WDB_SIZE} onChange={(v) => updateConfigValue('RN_WDB_SIZE', v)} min={1} style={{ width: '100%' }} />
                </Col>
              )}
              {configValues.SN_DDR_RDB_SIZE !== undefined && (
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="SN_DDR_RDB_SIZE" /></div>
                  <InputNumber value={configValues.SN_DDR_RDB_SIZE} onChange={(v) => updateConfigValue('SN_DDR_RDB_SIZE', v)} min={1} style={{ width: '100%' }} />
                </Col>
              )}
              {configValues.SN_DDR_WDB_SIZE !== undefined && (
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="SN_DDR_WDB_SIZE" /></div>
                  <InputNumber value={configValues.SN_DDR_WDB_SIZE} onChange={(v) => updateConfigValue('SN_DDR_WDB_SIZE', v)} min={1} style={{ width: '100%' }} />
                </Col>
              )}
              {configValues.SN_L2M_RDB_SIZE !== undefined && (
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="SN_L2M_RDB_SIZE" /></div>
                  <InputNumber value={configValues.SN_L2M_RDB_SIZE} onChange={(v) => updateConfigValue('SN_L2M_RDB_SIZE', v)} min={1} style={{ width: '100%' }} />
                </Col>
              )}
              {configValues.SN_L2M_WDB_SIZE !== undefined && (
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="SN_L2M_WDB_SIZE" /></div>
                  <InputNumber value={configValues.SN_L2M_WDB_SIZE} onChange={(v) => updateConfigValue('SN_L2M_WDB_SIZE', v)} min={1} style={{ width: '100%' }} />
                </Col>
              )}
            </Row>
          ),
        },
        {
          key: 'kcin',
          label: 'KCIN Config',
          children: (
            <div>
              <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 8 }}>Slice Per Link</Text>
              <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
                {configValues.SLICE_PER_LINK_HORIZONTAL !== undefined && (
                  <Col span={8}>
                    <div style={{ marginBottom: 4 }}><ConfigLabel name="SLICE_PER_LINK_HORIZONTAL" /></div>
                    <InputNumber value={configValues.SLICE_PER_LINK_HORIZONTAL} onChange={(v) => updateConfigValue('SLICE_PER_LINK_HORIZONTAL', v)} min={1} style={{ width: '100%' }} />
                  </Col>
                )}
                {configValues.SLICE_PER_LINK_VERTICAL !== undefined && (
                  <Col span={8}>
                    <div style={{ marginBottom: 4 }}><ConfigLabel name="SLICE_PER_LINK_VERTICAL" /></div>
                    <InputNumber value={configValues.SLICE_PER_LINK_VERTICAL} onChange={(v) => updateConfigValue('SLICE_PER_LINK_VERTICAL', v)} min={1} style={{ width: '100%' }} />
                  </Col>
                )}
              </Row>

              <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 8 }}>FIFO Depth</Text>
              <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
                {['IQ_CH_FIFO_DEPTH', 'EQ_CH_FIFO_DEPTH', 'IQ_OUT_FIFO_DEPTH_HORIZONTAL', 'IQ_OUT_FIFO_DEPTH_VERTICAL',
                  'IQ_OUT_FIFO_DEPTH_EQ', 'RB_OUT_FIFO_DEPTH', 'RB_IN_FIFO_DEPTH', 'EQ_IN_FIFO_DEPTH',
                  'IP_L2H_FIFO_DEPTH', 'IP_H2L_H_FIFO_DEPTH', 'IP_H2L_L_FIFO_DEPTH'].map(key =>
                  configValues[key] !== undefined && (
                    <Col span={8} key={key}>
                      <div style={{ marginBottom: 4 }}><ConfigLabel name={key} /></div>
                      <InputNumber value={configValues[key]} onChange={(v) => updateConfigValue(key, v)} min={1} style={{ width: '100%' }} />
                    </Col>
                  )
                )}
              </Row>

              <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 8 }}>Latency</Text>
              <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
                {['DDR_R_LATENCY', 'DDR_R_LATENCY_VAR', 'DDR_W_LATENCY', 'L2M_R_LATENCY', 'L2M_W_LATENCY',
                  'SN_TRACKER_RELEASE_LATENCY', 'SN_PROCESSING_LATENCY', 'RN_PROCESSING_LATENCY'].map(key =>
                  configValues[key] !== undefined && (
                    <Col span={8} key={key}>
                      <div style={{ marginBottom: 4 }}><ConfigLabel name={key} /></div>
                      <InputNumber value={configValues[key]} onChange={(v) => updateConfigValue(key, v)} min={0} style={{ width: '100%' }} />
                    </Col>
                  )
                )}
              </Row>

              <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 8 }}>Bandwidth Limit (GB/s)</Text>
              <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
                {['GDMA_BW_LIMIT', 'SDMA_BW_LIMIT', 'CDMA_BW_LIMIT', 'DDR_BW_LIMIT', 'L2M_BW_LIMIT'].map(key =>
                  configValues[key] !== undefined && (
                    <Col span={8} key={key}>
                      <div style={{ marginBottom: 4 }}><ConfigLabel name={key} /></div>
                      <InputNumber value={configValues[key]} onChange={(v) => updateConfigValue(key, v)} min={0} style={{ width: '100%' }} />
                    </Col>
                  )
                )}
              </Row>

              <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 8 }}>ETag Config</Text>
              <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
                {['TL_Etag_T2_UE_MAX', 'TL_Etag_T1_UE_MAX', 'TR_Etag_T2_UE_MAX', 'TU_Etag_T2_UE_MAX',
                  'TU_Etag_T1_UE_MAX', 'TD_Etag_T2_UE_MAX'].map(key =>
                  configValues[key] !== undefined && (
                    <Col span={8} key={key}>
                      <div style={{ marginBottom: 4 }}><ConfigLabel name={key} /></div>
                      <InputNumber value={configValues[key]} onChange={(v) => updateConfigValue(key, v)} min={0} style={{ width: '100%' }} />
                    </Col>
                  )
                )}
              </Row>

              <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 8 }}>ITag Config</Text>
              <Row gutter={[16, 8]}>
                {['ITag_TRIGGER_Th_H', 'ITag_TRIGGER_Th_V', 'ITag_MAX_Num_H', 'ITag_MAX_Num_V'].map(key =>
                  configValues[key] !== undefined && (
                    <Col span={8} key={key}>
                      <div style={{ marginBottom: 4 }}><ConfigLabel name={key} /></div>
                      <InputNumber value={configValues[key]} onChange={(v) => updateConfigValue(key, v)} min={0} style={{ width: '100%' }} />
                    </Col>
                  )
                )}
              </Row>
            </div>
          ),
        },
        {
          key: 'features',
          label: 'Feature Config',
          children: (
            <div>
              {configValues.UNIFIED_RW_TRACKER !== undefined && (
                <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                  <Col span={10}><ConfigLabel name="UNIFIED_RW_TRACKER" /></Col>
                  <Col span={14}>
                    <Switch checked={configValues.UNIFIED_RW_TRACKER} onChange={(v) => updateConfigValue('UNIFIED_RW_TRACKER', v)} />
                  </Col>
                </Row>
              )}
              {configValues.ETAG_T1_ENABLED !== undefined && (
                <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                  <Col span={10}><ConfigLabel name="ETAG_T1_ENABLED" /></Col>
                  <Col span={14}>
                    <Switch checked={!!configValues.ETAG_T1_ENABLED} onChange={(v) => updateConfigValue('ETAG_T1_ENABLED', v ? 1 : 0)} />
                  </Col>
                </Row>
              )}
              {configValues.ETag_BOTHSIDE_UPGRADE !== undefined && (
                <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                  <Col span={10}><ConfigLabel name="ETag_BOTHSIDE_UPGRADE" /></Col>
                  <Col span={14}>
                    <Switch checked={!!configValues.ETag_BOTHSIDE_UPGRADE} onChange={(v) => updateConfigValue('ETag_BOTHSIDE_UPGRADE', v ? 1 : 0)} />
                  </Col>
                </Row>
              )}
              {configValues.ORDERING_PRESERVATION_MODE !== undefined && (
                <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                  <Col span={10}><ConfigLabel name="ORDERING_PRESERVATION_MODE" /></Col>
                  <Col span={14}>
                    <Select value={configValues.ORDERING_PRESERVATION_MODE} onChange={(v) => updateConfigValue('ORDERING_PRESERVATION_MODE', v)} style={{ width: 160 }}>
                      <Option value={0}>0 - Disabled</Option>
                      <Option value={1}>1 - Single Side</Option>
                      <Option value={2}>2 - Both Sides</Option>
                      <Option value={3}>3 - Dynamic</Option>
                    </Select>
                  </Col>
                </Row>
              )}
              {configValues.ORDERING_ETAG_UPGRADE_MODE !== undefined && (
                <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                  <Col span={10}><ConfigLabel name="ORDERING_ETAG_UPGRADE_MODE" /></Col>
                  <Col span={14}>
                    <Select value={configValues.ORDERING_ETAG_UPGRADE_MODE} onChange={(v) => updateConfigValue('ORDERING_ETAG_UPGRADE_MODE', v)} style={{ width: 160 }}>
                      <Option value={0}>0 - Resource Only</Option>
                      <Option value={1}>1 - Include Ordering</Option>
                    </Select>
                  </Col>
                </Row>
              )}
              {configValues.ORDERING_GRANULARITY !== undefined && (
                <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                  <Col span={10}><ConfigLabel name="ORDERING_GRANULARITY" /></Col>
                  <Col span={14}>
                    <Select value={configValues.ORDERING_GRANULARITY} onChange={(v) => updateConfigValue('ORDERING_GRANULARITY', v)} style={{ width: 160 }}>
                      <Option value={0}>0 - IP Level</Option>
                      <Option value={1}>1 - Node Level</Option>
                    </Select>
                  </Col>
                </Row>
              )}
              {configValues.ORDERING_PRESERVATION_MODE === 2 && (
                <>
                  {['TL_ALLOWED_SOURCE_NODES', 'TR_ALLOWED_SOURCE_NODES', 'TU_ALLOWED_SOURCE_NODES', 'TD_ALLOWED_SOURCE_NODES'].map(key =>
                    configValues[key] !== undefined && (
                      <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }} key={key}>
                        <Col span={10}><ConfigLabel name={key} /></Col>
                        <Col span={14}>
                          <Input
                            value={Array.isArray(configValues[key]) ? configValues[key].join(', ') : configValues[key]}
                            onChange={(e) => updateConfigValue(key, e.target.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)))}
                            placeholder="e.g. 2,3,6,7"
                            style={{ width: '100%' }}
                          />
                        </Col>
                      </Row>
                    )
                  )}
                </>
              )}
              {configValues.REVERSE_DIRECTION_ENABLED !== undefined && (
                <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                  <Col span={10}><ConfigLabel name="REVERSE_DIRECTION_ENABLED" /></Col>
                  <Col span={14}>
                    <Switch checked={!!configValues.REVERSE_DIRECTION_ENABLED} onChange={(v) => updateConfigValue('REVERSE_DIRECTION_ENABLED', v ? 1 : 0)} />
                  </Col>
                </Row>
              )}
              {configValues.REVERSE_DIRECTION_THRESHOLD !== undefined && (
                <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                  <Col span={10}><ConfigLabel name="REVERSE_DIRECTION_THRESHOLD" /></Col>
                  <Col span={14}>
                    <InputNumber value={configValues.REVERSE_DIRECTION_THRESHOLD} onChange={(v) => updateConfigValue('REVERSE_DIRECTION_THRESHOLD', v)} min={0} max={1} step={0.05} style={{ width: 120 }} />
                  </Col>
                </Row>
              )}
              {configValues.arbitration?.default?.type !== undefined && (
                <>
                  <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                    <Col span={10}><ConfigLabel name="ARBITRATION_TYPE" /></Col>
                    <Col span={14}>
                      <Select
                        value={configValues.arbitration.default.type}
                        onChange={(v) => {
                          const newDefault: any = { type: v }
                          if (v === 'islip') {
                            newDefault.iterations = configValues.arbitration.default.iterations || 1
                            newDefault.weight_strategy = configValues.arbitration.default.weight_strategy || 'queue_length'
                          }
                          updateConfigValue('arbitration', { ...configValues.arbitration, default: newDefault })
                        }}
                        style={{ width: 160 }}
                      >
                        <Option value="round_robin">Round Robin</Option>
                        <Option value="islip">iSLIP</Option>
                        <Option value="weighted">Weighted</Option>
                        <Option value="priority">Priority</Option>
                        <Option value="dynamic">Dynamic</Option>
                        <Option value="random">Random</Option>
                      </Select>
                    </Col>
                  </Row>
                  {configValues.arbitration.default.type === 'islip' && (
                    <>
                      <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                        <Col span={10}><ConfigLabel name="ARBITRATION_ITERATIONS" /></Col>
                        <Col span={14}>
                          <InputNumber
                            value={configValues.arbitration.default.iterations || 1}
                            onChange={(v) => updateConfigValue('arbitration', { ...configValues.arbitration, default: { ...configValues.arbitration.default, iterations: v } })}
                            min={1}
                            max={10}
                            style={{ width: 120 }}
                          />
                        </Col>
                      </Row>
                      <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 8 }}>
                        <Col span={10}><ConfigLabel name="ARBITRATION_WEIGHT_STRATEGY" /></Col>
                        <Col span={14}>
                          <Select
                            value={configValues.arbitration.default.weight_strategy || 'queue_length'}
                            onChange={(v) => updateConfigValue('arbitration', { ...configValues.arbitration, default: { ...configValues.arbitration.default, weight_strategy: v } })}
                            style={{ width: 160 }}
                          >
                            <Option value="queue_length">Queue Length</Option>
                            <Option value="fixed">Fixed</Option>
                            <Option value="priority">Priority</Option>
                          </Select>
                        </Col>
                      </Row>
                    </>
                  )}
                </>
              )}
            </div>
          ),
        },
      ]}
    />
  )
}

// ============== DCINConfigPanel ==============

interface DCINConfigPanelProps {
  configValues: Record<string, any>
  updateConfigValue: (key: string, value: any) => void
}

export const DCINConfigPanel: React.FC<DCINConfigPanelProps> = ({
  configValues,
  updateConfigValue,
}) => {
  return (
    <Collapse
      size="small"
      style={{ marginBottom: 16 }}
      items={[
        {
          key: 'dcin_basic',
          label: 'Basic Parameters',
          children: (
            <Row gutter={[16, 8]}>
              {configValues.NUM_DIES !== undefined && (
                <Col span={8}>
                  <div style={{ marginBottom: 4 }}><ConfigLabel name="NUM_DIES" /></div>
                  <InputNumber value={configValues.NUM_DIES} onChange={(v) => updateConfigValue('NUM_DIES', v)} min={2} style={{ width: '100%' }} />
                </Col>
              )}
            </Row>
          ),
        },
        {
          key: 'dcin_latency',
          label: 'Latency (ns)',
          children: (
            <Row gutter={[16, 8]}>
              {['D2D_AR_LATENCY', 'D2D_R_LATENCY', 'D2D_AW_LATENCY', 'D2D_W_LATENCY', 'D2D_B_LATENCY'].map(key =>
                configValues[key] !== undefined && (
                  <Col span={8} key={key}>
                    <div style={{ marginBottom: 4 }}><ConfigLabel name={key} /></div>
                    <InputNumber value={configValues[key]} onChange={(v) => updateConfigValue(key, v)} min={0} style={{ width: '100%' }} />
                  </Col>
                )
              )}
            </Row>
          ),
        },
        {
          key: 'dcin_bandwidth',
          label: 'Bandwidth Limit (GB/s)',
          children: (
            <Row gutter={[16, 8]}>
              {['D2D_RN_BW_LIMIT', 'D2D_SN_BW_LIMIT', 'D2D_AXI_BANDWIDTH'].map(key =>
                configValues[key] !== undefined && (
                  <Col span={8} key={key}>
                    <div style={{ marginBottom: 4 }}><ConfigLabel name={key} /></div>
                    <InputNumber value={configValues[key]} onChange={(v) => updateConfigValue(key, v)} min={1} style={{ width: '100%' }} />
                  </Col>
                )
              )}
            </Row>
          ),
        },
        {
          key: 'dcin_buffer',
          label: 'Buffer Size',
          children: (
            <Row gutter={[16, 8]}>
              {['D2D_RN_RDB_SIZE', 'D2D_RN_WDB_SIZE', 'D2D_SN_RDB_SIZE', 'D2D_SN_WDB_SIZE'].map(key =>
                configValues[key] !== undefined && (
                  <Col span={8} key={key}>
                    <div style={{ marginBottom: 4 }}><ConfigLabel name={key} /></div>
                    <InputNumber value={configValues[key]} onChange={(v) => updateConfigValue(key, v)} min={1} style={{ width: '100%' }} />
                  </Col>
                )
              )}
            </Row>
          ),
        },
        {
          key: 'dcin_connections',
          label: 'D2D Connections',
          children: (
            <div>
              {configValues.D2D_CONNECTIONS !== undefined && Array.isArray(configValues.D2D_CONNECTIONS) && (
                <>
                  <div style={{ marginBottom: 8 }}>
                    <Text type="secondary">每行格式: [源Die, 源节点, 目标Die, 目标节点]</Text>
                  </div>
                  {(() => {
                    const sortedConns = configValues.D2D_CONNECTIONS
                      .map((conn: number[], originalIdx: number) => ({ conn, originalIdx }))
                      .sort((a: any, b: any) => {
                        for (let i = 0; i < 4; i++) {
                          if (a.conn[i] !== b.conn[i]) return a.conn[i] - b.conn[i]
                        }
                        return 0
                      })
                    return sortedConns.map(({ conn, originalIdx }: { conn: number[], originalIdx: number }, displayIdx: number) => (
                      <Row key={displayIdx} gutter={8} style={{ marginBottom: 8 }} align="middle">
                        {[0, 1, 2, 3].map(i => (
                          <Col span={5} key={i}>
                            <InputNumber
                              value={conn[i]}
                              onChange={(v) => {
                                const newConns = [...configValues.D2D_CONNECTIONS]
                                const newConn = [...conn]
                                newConn[i] = v ?? 0
                                newConns[originalIdx] = newConn
                                updateConfigValue('D2D_CONNECTIONS', newConns)
                              }}
                              min={0}
                              placeholder={['源Die', '源节点', '目标Die', '目标节点'][i]}
                              style={{ width: '100%' }}
                            />
                          </Col>
                        ))}
                        <Col span={4}>
                          <Button
                            type="text"
                            danger
                            size="small"
                            onClick={() => {
                              const newConns = configValues.D2D_CONNECTIONS.filter((_: any, i: number) => i !== originalIdx)
                              updateConfigValue('D2D_CONNECTIONS', newConns)
                            }}
                          >
                            删除
                          </Button>
                        </Col>
                      </Row>
                    ))
                  })()}
                  <Button
                    type="dashed"
                    size="small"
                    onClick={() => {
                      const newConns = [...configValues.D2D_CONNECTIONS, [0, 0, 0, 0]]
                      updateConfigValue('D2D_CONNECTIONS', newConns)
                    }}
                    style={{ width: '100%', marginTop: 8 }}
                  >
                    + 添加连接
                  </Button>
                </>
              )}
            </div>
          ),
        },
      ]}
    />
  )
}
