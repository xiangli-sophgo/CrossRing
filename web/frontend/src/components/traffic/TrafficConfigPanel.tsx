import { useState, useEffect } from 'react'
import {
  Card,
  Form,
  Input,
  InputNumber,
  Button,
  Table,
  Space,
  message,
  Popconfirm,
  Tag,
  Select,
  Row,
  Col,
  Divider,
  Modal,
  Radio,
  Tooltip
} from 'antd'
import {
  SettingOutlined,
  DeleteOutlined,
  ClearOutlined,
  ThunderboltOutlined,
  SaveOutlined,
  UploadOutlined,
  DownOutlined,
  UpOutlined,
  QuestionCircleOutlined
} from '@ant-design/icons'
import type { TrafficConfig } from '../../types/trafficConfig'
import type { IPMount } from '../../types/ipMount'
import {
  createTrafficConfig,
  createBatchTrafficConfig,
  getTrafficConfigs,
  deleteTrafficConfig,
  clearAllConfigs,
  generateTraffic,
  downloadTrafficFile,
  listConfigFiles,
  loadConfigsFromFile,
  saveConfigsToFile
} from '../../api/trafficConfig'
import { getMounts } from '../../api/ipMount'

const { Option } = Select

interface TrafficConfigPanelProps {
  topology: string
  mode: 'noc' | 'd2d'
  mountsVersion?: number
}

const TrafficConfigPanel: React.FC<TrafficConfigPanelProps> = ({ topology, mode, mountsVersion }) => {
  const [form] = Form.useForm()
  const [configs, setConfigs] = useState<TrafficConfig[]>([])
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [mounts, setMounts] = useState<IPMount[]>([])
  const [rnIPs, setRnIPs] = useState<string[]>([])
  const [snIPs, setSnIPs] = useState<string[]>([])
  const [selectedSourceIPs, setSelectedSourceIPs] = useState<string[]>([])
  const [selectedTargetIPs, setSelectedTargetIPs] = useState<string[]>([])
  const [saveModalVisible, setSaveModalVisible] = useState(false)
  const [saveFileName, setSaveFileName] = useState('')
  const [loadModalVisible, setLoadModalVisible] = useState(false)
  const [availableConfigFiles, setAvailableConfigFiles] = useState<any[]>([])
  const [selectedConfigFile, setSelectedConfigFile] = useState<string>('')
  const [loadMode, setLoadMode] = useState<'replace' | 'append'>('replace')
  const [loadingFiles, setLoadingFiles] = useState(false)
  const [trafficFileName, setTrafficFileName] = useState('')
  const [selectedDiePairs, setSelectedDiePairs] = useState<string[]>([])

  // 加载流量配置
  const loadConfigs = async () => {
    setRefreshing(true)
    try {
      const data = await getTrafficConfigs(topology, mode)
      setConfigs(data.configs)
    } catch (error) {
      message.error('加载流量配置失败')
      console.error(error)
    } finally {
      setRefreshing(false)
    }
  }

  // 加载IP挂载
  const loadMounts = async () => {
    try {
      const data = await getMounts(topology)
      setMounts(data.mounts)

      // 分类IP: RN和SN
      const rnTypes = ['gdma', 'sdma', 'cdma', 'npu', 'pcie', 'eth']
      const snTypes = ['ddr', 'l2m']

      const rns: string[] = []
      const sns: string[] = []

      // 统计各类型IP的数量
      const rnTypeCount: Record<string, number> = {}
      const snTypeCount: Record<string, number> = {}

      data.mounts.forEach((mount: IPMount) => {
        const type = mount.ip_type.split('_')[0].toLowerCase()
        const label = `节点${mount.node_id}-${mount.ip_type}`
        if (rnTypes.includes(type)) {
          rns.push(label)
          rnTypeCount[type] = (rnTypeCount[type] || 0) + 1
        } else if (snTypes.includes(type)) {
          sns.push(label)
          snTypeCount[type] = (snTypeCount[type] || 0) + 1
        }
      })

      // 添加"全部XXX"选项到最前面
      const rnAllOptions = Object.keys(rnTypeCount)
        .filter(type => rnTypeCount[type] > 0)
        .sort()
        .map(type => `全部${type}`)

      const snAllOptions = Object.keys(snTypeCount)
        .filter(type => snTypeCount[type] > 0)
        .sort()
        .map(type => `全部${type}`)

      const rnWithAll = [...rnAllOptions, ...rns.sort()]
      const snWithAll = [...snAllOptions, ...sns.sort()]

      setRnIPs(rnWithAll)
      setSnIPs(snWithAll)
    } catch (error) {
      console.error('加载IP挂载失败:', error)
    }
  }

  useEffect(() => {
    loadConfigs()
    loadMounts()
  }, [topology, mode, mountsVersion])

  // 展开"全部XXX"为具体IP列表
  const expandAllIPs = (selectedIPs: string[]): string[] => {
    const expanded: string[] = []
    selectedIPs.forEach(ip => {
      if (ip.startsWith('全部')) {
        // 提取类型，例如 "全部gdma" -> "gdma"
        const type = ip.replace('全部', '').toLowerCase()
        // 找到所有该类型的IP
        mounts.forEach(mount => {
          const mountType = mount.ip_type.split('_')[0].toLowerCase()
          if (mountType === type) {
            expanded.push(`节点${mount.node_id}-${mount.ip_type}`)
          }
        })
      } else {
        expanded.push(ip)
      }
    })
    return expanded
  }

  // 过滤可用选项：如果选择了"全部XXX"，则隐藏该类型的具体IP
  const getAvailableIPs = (allIPs: string[], selectedIPs: string[]): string[] => {
    // 找出所有已选的"全部XXX"类型
    const selectedAllTypes = selectedIPs
      .filter(ip => ip.startsWith('全部'))
      .map(ip => ip.replace('全部', '').toLowerCase())

    return allIPs.filter(ip => {
      // 如果是"全部XXX"选项，总是显示（除非已被选中）
      if (ip.startsWith('全部')) {
        return !selectedIPs.includes(ip)
      }
      // 如果是具体IP，检查其类型是否被"全部XXX"选中
      const ipType = ip.split('-')[1]?.split('_')[0]?.toLowerCase()
      if (ipType && selectedAllTypes.includes(ipType)) {
        return false // 隐藏该IP
      }
      return !selectedIPs.includes(ip)
    })
  }

  // 创建配置
  const handleCreate = async (values: any) => {
    setLoading(true)
    try {
      // 展开"全部XXX"选项
      const expandedSourceIPs = expandAllIPs(selectedSourceIPs)
      const expandedTargetIPs = expandAllIPs(selectedTargetIPs)

      // D2D模式：为每个DIE对创建配置
      if (mode === 'd2d') {
        // 验证DIE对选择
        if (selectedDiePairs.length === 0) {
          message.error('请选择DIE对')
          return
        }

        // 为每个DIE对创建配置
        const promises = []
        for (const diePair of selectedDiePairs) {
          // 解析DIE对，格式: "0->1"
          const [sourceDie, targetDie] = diePair.split('->').map(Number)

          const configData = {
            topology,
            mode,
            source_ips: expandedSourceIPs,
            target_ips: expandedTargetIPs,
            speed_gbps: values.speed_gbps,
            burst_length: values.burst_length,
            request_type: values.request_type,
            end_time_ns: values.end_time_ns,
            source_die: sourceDie,
            target_die: targetDie
          }
          promises.push(createBatchTrafficConfig(configData))
        }
        await Promise.all(promises)
        const totalConfigs = expandedSourceIPs.length * expandedTargetIPs.length * selectedDiePairs.length
        message.success(`成功创建 ${totalConfigs} 个流量配置`)
      } else {
        // NoC模式
        const configData = {
          topology,
          mode,
          source_ips: expandedSourceIPs,
          target_ips: expandedTargetIPs,
          speed_gbps: values.speed_gbps,
          burst_length: values.burst_length,
          request_type: values.request_type,
          end_time_ns: values.end_time_ns
        }
        await createBatchTrafficConfig(configData)
        message.success(`成功创建 ${expandedSourceIPs.length * expandedTargetIPs.length} 个流量配置`)
      }

      form.resetFields()
      setSelectedSourceIPs([])
      setSelectedTargetIPs([])
      setSelectedDiePairs([])
      loadConfigs()
    } catch (error: any) {
      message.error(error.response?.data?.detail || '创建失败')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  // 删除配置
  const handleDelete = async (configId: string) => {
    try {
      await deleteTrafficConfig(topology, mode, configId)
      message.success('删除成功')
      loadConfigs()
    } catch (error) {
      message.error('删除失败')
      console.error(error)
    }
  }

  // 清空所有配置
  const handleClearAll = async () => {
    try {
      await clearAllConfigs(topology, mode)
      message.success('已清空所有流量配置')
      loadConfigs()
    } catch (error) {
      message.error('清空失败')
      console.error(error)
    }
  }

  // 生成流量
  const handleGenerate = async () => {
    if (configs.length === 0) {
      message.warning('请先添加流量配置')
      return
    }

    if (!trafficFileName.trim()) {
      message.warning('请输入文件名')
      return
    }

    setGenerating(true)
    try {
      const result = await generateTraffic({
        topology,
        mode,
        split_by_source: false,
        random_seed: 42,
        filename: trafficFileName
      })

      message.success(
        `流量生成成功！共 ${result.total_lines} 行，耗时 ${result.generation_time_ms.toFixed(0)}ms`
      )
    } catch (error: any) {
      message.error(error.response?.data?.detail || '流量生成失败')
      console.error(error)
    } finally {
      setGenerating(false)
    }
  }

  // 打开保存对话框
  const handleOpenSaveModal = () => {
    if (configs.length === 0) {
      message.warning('当前没有数据流配置可保存')
      return
    }
    const defaultName = `traffic_${topology}_${mode}_${new Date().toISOString().slice(0,10)}`
    setSaveFileName(defaultName)
    setSaveModalVisible(true)
  }

  // 保存配置到文件
  const handleSaveToFile = async () => {
    if (!saveFileName.trim()) {
      message.error('请输入文件名')
      return
    }

    try {
      const response = await saveConfigsToFile(topology, mode, saveFileName)
      message.success(response.message)
      setSaveModalVisible(false)
    } catch (error: any) {
      message.error(error.response?.data?.detail || '保存失败')
      console.error(error)
    }
  }

  // 打开加载对话框
  const handleOpenLoadModal = async () => {
    setLoadingFiles(true)
    setLoadModalVisible(true)
    setLoadMode('replace')
    setSelectedConfigFile('')

    try {
      const response = await listConfigFiles()
      setAvailableConfigFiles(response.files)
    } catch (error: any) {
      message.error('获取文件列表失败')
      console.error(error)
      setAvailableConfigFiles([])
    } finally {
      setLoadingFiles(false)
    }
  }

  // 从文件加载配置
  const handleLoadFromFile = async () => {
    if (!selectedConfigFile) {
      message.error('请选择要加载的文件')
      return
    }

    try {
      const response = await loadConfigsFromFile(topology, mode, selectedConfigFile, loadMode)
      message.success(response.message)
      setLoadModalVisible(false)
      loadConfigs()
    } catch (error: any) {
      message.error(error.response?.data?.detail || '加载失败')
      console.error(error)
    }
  }

  // 将配置按IP类型分组（不区分编号）
  const groupedConfigs = configs.reduce((groups: any[], config: TrafficConfig) => {
    // 提取IP类型和编号
    const sourceFullType = config.source_ip.includes('-')
      ? config.source_ip.split('-')[1]
      : config.source_ip
    const targetFullType = config.target_ip.includes('-')
      ? config.target_ip.split('-')[1]
      : config.target_ip

    // 提取基础类型（去掉编号，如 gdma_0 -> gdma）
    const sourceBaseType = sourceFullType.split('_')[0]
    const targetBaseType = targetFullType.split('_')[0]

    // 查找是否已存在相同参数的组
    // D2D模式下还需要匹配源DIE
    const existingGroup = groups.find(g => {
      const baseMatch =
        g.sourceBaseType === sourceBaseType &&
        g.targetBaseType === targetBaseType &&
        g.speed_gbps === config.speed_gbps &&
        g.burst_length === config.burst_length &&
        g.request_type === config.request_type &&
        g.end_time_ns === config.end_time_ns

      // D2D模式下需要匹配源DIE
      if (mode === 'd2d' && config.source_die !== undefined) {
        return baseMatch && g.source_die === config.source_die
      }
      return baseMatch
    })

    const sourceNode = config.source_ip.includes('-')
      ? parseInt(config.source_ip.split('-')[0].replace('节点', ''))
      : null
    const targetNode = config.target_ip.includes('-')
      ? parseInt(config.target_ip.split('-')[0].replace('节点', ''))
      : null

    if (existingGroup) {
      // 添加到现有组的详细信息
      if (!existingGroup.details[sourceFullType]) {
        existingGroup.details[sourceFullType] = []
      }
      if (sourceNode !== null && !existingGroup.details[sourceFullType].includes(sourceNode)) {
        existingGroup.details[sourceFullType].push(sourceNode)
      }

      // D2D模式下按目标DIE分组目标IP
      if (mode === 'd2d' && config.target_die !== undefined) {
        if (!existingGroup.target_dies) {
          existingGroup.target_dies = new Set()
        }
        existingGroup.target_dies.add(config.target_die)

        if (!existingGroup.targetDetailsByDie) {
          existingGroup.targetDetailsByDie = {}
        }
        if (!existingGroup.targetDetailsByDie[config.target_die]) {
          existingGroup.targetDetailsByDie[config.target_die] = {}
        }
        if (!existingGroup.targetDetailsByDie[config.target_die][targetFullType]) {
          existingGroup.targetDetailsByDie[config.target_die][targetFullType] = []
        }
        if (targetNode !== null && !existingGroup.targetDetailsByDie[config.target_die][targetFullType].includes(targetNode)) {
          existingGroup.targetDetailsByDie[config.target_die][targetFullType].push(targetNode)
        }
      } else {
        // 非D2D模式使用原有逻辑
        if (!existingGroup.details[targetFullType]) {
          existingGroup.details[targetFullType] = []
        }
        if (targetNode !== null && !existingGroup.details[targetFullType].includes(targetNode)) {
          existingGroup.details[targetFullType].push(targetNode)
        }
      }

      existingGroup.configIds.push(config.id)
    } else {
      // 创建新组
      const details: Record<string, number[]> = {}
      if (sourceNode !== null) {
        details[sourceFullType] = [sourceNode]
      }
      if (targetNode !== null && mode !== 'd2d') {
        details[targetFullType] = [targetNode]
      }

      const groupData: any = {
        key: `${sourceBaseType}-${targetBaseType}-${config.speed_gbps}-${config.burst_length}-${config.request_type}`,
        sourceBaseType,
        targetBaseType,
        details,
        speed_gbps: config.speed_gbps,
        burst_length: config.burst_length,
        request_type: config.request_type,
        end_time_ns: config.end_time_ns,
        configIds: [config.id]
      }

      // D2D模式添加DIE信息和按DIE分组的目标IP
      if (config.source_die !== undefined) {
        groupData.source_die = config.source_die
      }
      if (config.target_die !== undefined) {
        groupData.target_die = config.target_die
        groupData.target_dies = new Set([config.target_die])
        groupData.targetDetailsByDie = {
          [config.target_die]: {
            [targetFullType]: targetNode !== null ? [targetNode] : []
          }
        }
      }

      groups.push(groupData)
    }
    return groups
  }, [])

  // 对每组的详细信息节点ID进行排序
  groupedConfigs.forEach(g => {
    Object.keys(g.details).forEach(key => {
      g.details[key].sort((a: number, b: number) => a - b)
    })

    // D2D模式下对按DIE分组的目标IP也进行排序
    if (g.targetDetailsByDie) {
      Object.keys(g.targetDetailsByDie).forEach(dieId => {
        Object.keys(g.targetDetailsByDie[dieId]).forEach(key => {
          g.targetDetailsByDie[dieId][key].sort((a: number, b: number) => a - b)
        })
      })
    }
  })

  const columns = [
    {
      title: '源IP',
      dataIndex: 'sourceBaseType',
      key: 'sourceBaseType',
      width: 150,
      render: (sourceBaseType: string) => (
        <Tag color="blue">{sourceBaseType}</Tag>
      )
    },
    {
      title: '目标IP',
      dataIndex: 'targetBaseType',
      key: 'targetBaseType',
      width: 150,
      render: (targetBaseType: string) => (
        <Tag color="green">{targetBaseType}</Tag>
      )
    },
    ...(mode === 'd2d' ? [{
      title: '源DIE',
      dataIndex: 'source_die',
      key: 'source_die',
      width: 100,
      render: (sourceDie: number, record: any) => {
        // 收集该组中所有目标DIE
        const targetDies = record.target_dies ? Array.from(record.target_dies).sort((a: number, b: number) => a - b) : []

        // 如果sourceDie未定义,返回空
        if (sourceDie === undefined || sourceDie === null) {
          return null
        }

        return (
          <div>
            <Tag color="purple">{sourceDie}</Tag>
            {targetDies.length > 0 && (
              <div style={{ marginTop: 4, fontSize: '12px', color: '#666' }}>
                → {targetDies.join(', ')}
              </div>
            )}
          </div>
        )
      }
    }] : []),
    {
      title: (
        <Space size={4}>
          带宽
          <Tooltip title="每个源IP的总带宽">
            <QuestionCircleOutlined style={{ color: '#999', fontSize: '12px' }} />
          </Tooltip>
        </Space>
      ),
      dataIndex: 'speed_gbps',
      key: 'speed_gbps',
      width: 80,
      render: (speed: number) => `${speed} GB/s`
    },
    {
      title: 'Burst',
      dataIndex: 'burst_length',
      key: 'burst_length',
      width: 80
    },
    {
      title: '类型',
      dataIndex: 'request_type',
      key: 'request_type',
      width: 80,
      render: (type: string) => (
        <Tag color={type === 'R' ? 'cyan' : 'orange'}>{type}</Tag>
      )
    },
    {
      title: '结束时间',
      dataIndex: 'end_time_ns',
      key: 'end_time_ns',
      width: 80,
      render: (time: number) => `${time} ns`
    },
    {
      title: '操作',
      key: 'action',
      width: 100,
      render: (record: any) => (
        <Popconfirm
          title={`确认删除此组配置(${record.configIds.length}个)？`}
          onConfirm={async () => {
            try {
              // 删除该组的所有配置
              await Promise.all(record.configIds.map((id: string) => deleteTrafficConfig(topology, mode, id)))
              message.success('删除成功')
              loadConfigs()
            } catch (error) {
              message.error('删除失败')
            }
          }}
          okText="确认"
          cancelText="取消"
        >
          <Button type="link" danger icon={<DeleteOutlined />} size="small">
            删除
          </Button>
        </Popconfirm>
      )
    }
  ]

  return (
    <Card
      title={
        <Space>
          <SettingOutlined />
          <span>流量配置</span>
          <Tag color="purple">{configs.length} 个配置</Tag>
          <Tag color={mode === 'noc' ? 'blue' : 'orange'}>{mode.toUpperCase()} 模式</Tag>
        </Space>
      }
      extra={
        <Space>
          <Button size="small" onClick={loadConfigs} loading={refreshing}>
            刷新
          </Button>
          <Button
            size="small"
            icon={<SaveOutlined />}
            onClick={handleOpenSaveModal}
            disabled={configs.length === 0}
          >
            保存
          </Button>
          <Button
            size="small"
            icon={<UploadOutlined />}
            onClick={handleOpenLoadModal}
          >
            加载
          </Button>
          <Popconfirm
            title="确认清空所有流量配置？"
            onConfirm={handleClearAll}
            okText="确认"
            cancelText="取消"
          >
            <Button size="small" danger icon={<ClearOutlined />}>
              清空
            </Button>
          </Popconfirm>
        </Space>
      }
    >
      <Card title="添加数据流配置" size="small" style={{ marginBottom: 16 }}>
        <Form form={form} onFinish={handleCreate} labelAlign="left">
          <Row gutter={16} style={{ width: '100%', marginBottom: 16 }}>
            <Col span={24}>
              <Form.Item
                name="source_ip"
                label="源IP"
                rules={[{ required: true, message: '请选择源IP' }]}
                labelCol={{ span: 3 }}
                wrapperCol={{ span: 21 }}
              >
                <Select
                  mode="multiple"
                  placeholder={mounts.length === 0 ? "请先在左侧挂载IP" : "选择源IP (RN, 可多选)"}
                  showSearch
                  allowClear
                  value={selectedSourceIPs}
                  onChange={setSelectedSourceIPs}
                  disabled={mounts.length === 0}
                  notFoundContent={mounts.length === 0 ? "暂无IP挂载" : "暂无RN类型IP"}
                  style={{ width: '100%' }}
                  listHeight={400}
                >
                  {getAvailableIPs(rnIPs, selectedSourceIPs).map(ip => (
                    <Option key={ip} value={ip}>{ip}</Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16} style={{ width: '100%', marginBottom: 16 }}>
            <Col span={24}>
              <Form.Item
                name="target_ip"
                label="目标IP"
                rules={[{ required: true, message: '请选择目标IP' }]}
                labelCol={{ span: 3 }}
                wrapperCol={{ span: 21 }}
              >
                <Select
                  mode="multiple"
                  placeholder={mounts.length === 0 ? "请先在左侧挂载IP" : "选择目标IP (SN, 可多选)"}
                  showSearch
                  allowClear
                  value={selectedTargetIPs}
                  onChange={setSelectedTargetIPs}
                  disabled={mounts.length === 0}
                  notFoundContent={mounts.length === 0 ? "暂无IP挂载" : "暂无SN类型IP"}
                  style={{ width: '100%' }}
                  listHeight={400}
                >
                  {getAvailableIPs(snIPs, selectedTargetIPs).map(ip => (
                    <Option key={ip} value={ip}>{ip}</Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
          </Row>
          {mode === 'd2d' && (
            <Row gutter={16} style={{ width: '100%', marginBottom: 16 }}>
              <Col span={24}>
                <Form.Item
                  name="die_pairs"
                  label="DIE对"
                  rules={[{ required: true, message: '请选择DIE对' }]}
                  labelCol={{ span: 3 }}
                  wrapperCol={{ span: 21 }}
                >
                  <Select
                    mode="multiple"
                    placeholder="选择DIE对 (可多选)"
                    value={selectedDiePairs}
                    onChange={setSelectedDiePairs}
                    showSearch
                    allowClear
                    style={{ width: '100%' }}
                    listHeight={400}
                  >
                    {[0, 1, 2, 3].flatMap(src =>
                      [0, 1, 2, 3]
                        .map(dst => `${src}->${dst}`)
                        .filter(pair => !selectedDiePairs.includes(pair))
                        .map(pair => {
                          const [src, dst] = pair.split('->')
                          return (
                            <Option key={pair} value={pair}>
                              DIE{src} → DIE{dst}
                            </Option>
                          )
                        })
                    )}
                  </Select>
                </Form.Item>
              </Col>
            </Row>
          )}
          <Row gutter={16} style={{ width: '100%', marginBottom: 16 }}>
            <Col span={12}>
              <Form.Item
                name="speed_gbps"
                label="带宽 (GB/s)"
                rules={[{ required: true, message: '请输入带宽' }]}
                initialValue={128}
                labelCol={{ span: 6 }}
                wrapperCol={{ span: 16 }}
              >
                <InputNumber
                  placeholder="带宽"
                  min={0.1}
                  max={128}
                  step={0.1}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="burst_length"
                label="Burst长度"
                rules={[{ required: true, message: '请输入Burst' }]}
                initialValue={4}
                labelCol={{ span: 8 }}
                wrapperCol={{ span: 16 }}
              >
                <InputNumber
                  placeholder="Burst"
                  min={1}
                  max={16}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16} style={{ width: '100%', marginBottom: 16 }}>
            <Col span={12}>
              <Form.Item
                name="request_type"
                label="请求类型"
                rules={[{ required: true, message: '请选择类型' }]}
                initialValue="R"
                labelCol={{ span: 6 }}
                wrapperCol={{ span: 16 }}
              >
                <Select placeholder="类型">
                  <Option value="R">读 (R)</Option>
                  <Option value="W">写 (W)</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="end_time_ns"
                label="结束时间 (ns)"
                rules={[{ required: true, message: '请输入结束时间' }]}
                initialValue={6000}
                labelCol={{ span: 8 }}
                wrapperCol={{ span: 16 }}
              >
                <InputNumber
                  placeholder="结束时间"
                  min={100}
                  max={100000}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16} style={{ width: '100%' }}>
            <Col span={24}>
              <Form.Item wrapperCol={{ offset: 2, span: 21 }}>
                <Button type="primary" htmlType="submit" loading={loading} block>
                  添加配置
                </Button>
              </Form.Item>
            </Col>
          </Row>
        </Form>
        <Space style={{ marginTop: 16, width: '100%', justifyContent: 'center' }}>
          <Input
            placeholder="输入文件名"
            value={trafficFileName}
            onChange={(e) => setTrafficFileName(e.target.value)}
            style={{ width: 600 }}
            suffix=".txt"
          />
          <Button
            type="primary"
            icon={<ThunderboltOutlined />}
            onClick={handleGenerate}
            loading={generating}
            disabled={configs.length === 0}
          >
            生成数据流
          </Button>
        </Space>
      </Card>

      <Table
        columns={columns}
        dataSource={groupedConfigs}
        rowKey="key"
        pagination={{ pageSize: 10 }}
        loading={refreshing}
        size="small"
        expandable={{
          expandedRowRender: (record) => (
            <div style={{ padding: '8px 16px', backgroundColor: '#f9f9f9' }}>
              <Space direction="vertical" size="small" style={{ width: '100%' }}>
                <div>
                  <strong>源IP详情：</strong>
                  {Object.keys(record.details)
                    .filter(key => key.startsWith(record.sourceBaseType))
                    .map(key => (
                      <div key={key} style={{ marginLeft: 16, marginTop: 4 }}>
                        <Tag color="blue">{key}</Tag>: [{record.details[key].join(', ')}]
                      </div>
                    ))}
                </div>
                {mode === 'd2d' && record.target_dies && record.target_dies.size > 0 ? (
                  <div>
                    <strong>目标DIE详情：</strong>
                    {Array.from(record.target_dies).sort((a, b) => a - b).map((targetDie: number) => (
                      <div key={targetDie} style={{ marginLeft: 16, marginTop: 8 }}>
                        <Tag color="purple">{targetDie}</Tag>
                        <div style={{ marginLeft: 32, marginTop: 4 }}>
                          {record.targetDetailsByDie && record.targetDetailsByDie[targetDie] ? (
                            Object.keys(record.targetDetailsByDie[targetDie]).map(key => (
                              <div key={key} style={{ marginTop: 2 }}>
                                <Tag color="green">{key}</Tag>: [{record.targetDetailsByDie[targetDie][key].join(', ')}]
                              </div>
                            ))
                          ) : null}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div>
                    <strong>目标IP详情：</strong>
                    {Object.keys(record.details)
                      .filter(key => key.startsWith(record.targetBaseType))
                      .map(key => (
                        <div key={key} style={{ marginLeft: 16, marginTop: 4 }}>
                          <Tag color="green">{key}</Tag>: [{record.details[key].join(', ')}]
                        </div>
                      ))}
                  </div>
                )}
              </Space>
            </div>
          ),
          expandIcon: ({ expanded, onExpand, record }) =>
            expanded ? (
              <UpOutlined onClick={e => onExpand(record, e)} style={{ cursor: 'pointer' }} />
            ) : (
              <DownOutlined onClick={e => onExpand(record, e)} style={{ cursor: 'pointer' }} />
            ),
          defaultExpandAllRows: false
        }}
      />

      {/* 保存配置对话框 */}
      <Modal
        title="保存数据流配置"
        open={saveModalVisible}
        onOk={handleSaveToFile}
        onCancel={() => setSaveModalVisible(false)}
        okText="保存"
        cancelText="取消"
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <label>文件名：</label>
            <Input
              value={saveFileName}
              onChange={(e) => setSaveFileName(e.target.value)}
              placeholder="请输入文件名"
              suffix=".json"
              style={{ marginTop: 8 }}
            />
          </div>
          <div style={{ color: '#8c8c8c', fontSize: 12 }}>
            文件将保存到: config/traffic_configs/{saveFileName || '...'}.json
          </div>
        </Space>
      </Modal>

      {/* 加载配置对话框 */}
      <Modal
        title="加载数据流配置"
        open={loadModalVisible}
        onOk={handleLoadFromFile}
        onCancel={() => setLoadModalVisible(false)}
        okText="加载"
        cancelText="取消"
        width={700}
        confirmLoading={loadingFiles}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>从 config/traffic_configs 目录选择文件:</div>
          <Select
            value={selectedConfigFile}
            onChange={setSelectedConfigFile}
            placeholder={loadingFiles ? "正在加载文件列表..." : "请选择配置文件"}
            style={{ width: '100%' }}
            showSearch
            loading={loadingFiles}
            disabled={loadingFiles}
          >
            {availableConfigFiles?.map(file => (
              <Option key={file.filename} value={file.filename}>
                {file.filename} - {file.topology}/{file.mode} ({new Date(file.modified * 1000).toLocaleString()})
              </Option>
            ))}
          </Select>
          {!loadingFiles && (!availableConfigFiles || availableConfigFiles.length === 0) && (
            <div style={{ color: '#8c8c8c', fontSize: 12 }}>
              没有找到可用的配置文件
            </div>
          )}
          {loadingFiles && (
            <div style={{ color: '#1890ff', fontSize: 12 }}>
              正在加载文件列表...
            </div>
          )}
          <div style={{ marginTop: 16 }}>
            <label>加载模式：</label>
            <Radio.Group value={loadMode} onChange={(e) => setLoadMode(e.target.value)} style={{ marginTop: 8 }}>
              <Radio value="replace">替换当前配置</Radio>
              <Radio value="append">添加到当前配置</Radio>
            </Radio.Group>
          </div>
        </Space>
      </Modal>
    </Card>
  )
}

export default TrafficConfigPanel
