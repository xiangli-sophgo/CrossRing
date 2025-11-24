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
  Tag,
  Select,
  Row,
  Col,
  Divider,
  Modal,
  Radio,
  Tooltip,
  Checkbox,
  Dropdown
} from 'antd'
import type { MenuProps } from 'antd'
import {
  SettingOutlined,
  DeleteOutlined,
  ClearOutlined,
  ThunderboltOutlined,
  SaveOutlined,
  UploadOutlined,
  DownOutlined,
  UpOutlined,
  QuestionCircleOutlined,
  CalculatorOutlined
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
  saveConfigsToFile,
  updateTrafficConfig
} from '../../api/trafficConfig'
import { getMounts } from '../../api/ipMount'
import { computeStaticBandwidth, getD2DConfigs } from '../../api/staticBandwidth'
import type { BandwidthComputeResponse } from '../../types/staticBandwidth'

const { Option } = Select

interface TrafficConfigPanelProps {
  topology: string
  mode: 'noc' | 'd2d'
  mountsVersion?: number
  onBandwidthComputed?: (bandwidthData: BandwidthComputeResponse) => void
}

const TrafficConfigPanel: React.FC<TrafficConfigPanelProps> = ({ topology, mode, mountsVersion, onBandwidthComputed }) => {
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
  const [routingType, setRoutingType] = useState<'XY' | 'YX'>('XY')
  const [computing, setComputing] = useState(false)
  const [useFixedSeed, setUseFixedSeed] = useState(false)
  const [randomSeed, setRandomSeed] = useState(42)
  const [d2dConfigs, setD2DConfigs] = useState<Array<{ filename: string; num_dies: number; connections: number }>>([])
  const [selectedD2DConfig, setSelectedD2DConfig] = useState<string>('')
  const [enableSplit, setEnableSplit] = useState(false)
  const [splitFiles, setSplitFiles] = useState<Array<{ filename: string; count: number; path: string }>>([])
  const [showSplitModal, setShowSplitModal] = useState(false)

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

  // 加载D2D配置文件列表
  const loadD2DConfigs = async () => {
    try {
      const data = await getD2DConfigs()
      setD2DConfigs(data.configs)
      if (data.configs.length > 0 && !selectedD2DConfig) {
        setSelectedD2DConfig(data.configs[0].filename)
      }
    } catch (error) {
      console.error('加载D2D配置失败:', error)
    }
  }

  useEffect(() => {
    loadConfigs()
    loadMounts()
  }, [topology, mode, mountsVersion])

  useEffect(() => {
    if (mode === 'd2d') {
      loadD2DConfigs()
    }
  }, [mode])

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

      // D2D模式：按源DIE分组创建配置
      if (mode === 'd2d') {
        // 验证DIE对选择
        if (selectedDiePairs.length === 0) {
          message.error('请选择DIE对')
          return
        }

        // 按源DIE分组DIE对
        const diePairsBySourceDie: Record<number, number[]> = {}
        selectedDiePairs.forEach(diePair => {
          const [sourceDie, targetDie] = diePair.split('->').map(Number)
          if (!diePairsBySourceDie[sourceDie]) {
            diePairsBySourceDie[sourceDie] = []
          }
          diePairsBySourceDie[sourceDie].push(targetDie)
        })

        // 为每个源DIE创建一个配置
        const promises = []
        for (const [sourceDie, targetDies] of Object.entries(diePairsBySourceDie)) {
          const sourceDieNum = Number(sourceDie)
          // 构建die_pairs: [[sourceDie, targetDie1], [sourceDie, targetDie2], ...]
          const diePairsArray = targetDies.map(targetDie => [sourceDieNum, targetDie])

          const configData = {
            topology,
            mode,
            source_ips: expandedSourceIPs,
            target_ips: expandedTargetIPs,
            speed_gbps: values.speed_gbps,
            burst_length: values.burst_length,
            request_type: values.request_type,
            end_time_ns: values.end_time_ns,
            die_pairs: diePairsArray  // 该源DIE的所有目标DIE
          }
          promises.push(createBatchTrafficConfig(configData))
        }

        await Promise.all(promises)
        const totalSourceDies = Object.keys(diePairsBySourceDie).length
        const totalConfigs = expandedSourceIPs.length * expandedTargetIPs.length * selectedDiePairs.length
        message.success(`成功创建 ${totalSourceDies} 个配置（共 ${totalConfigs} 个组合）`)
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
        const result = await createBatchTrafficConfig(configData)
        message.success(result.message)
      }

      // 不清空表单，保留已选配置方便继续添加
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

  // 切换请求类型
  const handleToggleRequestType = async (record: any) => {
    try {
      // 获取该记录的所有配置ID
      const configIds = record.configIds || []

      // 新的请求类型
      const newRequestType = record.request_type === 'R' ? 'W' : 'R'

      // 批量更新所有相关配置
      await Promise.all(
        configIds.map(async (configId: string) => {
          // 从原始configs中找到对应的配置
          const originalConfig = configs.find((c: TrafficConfig) => c.id === configId)
          if (!originalConfig) return

          // 更新配置
          await updateTrafficConfig(topology, mode, configId, {
            topology: originalConfig.topology,
            mode: originalConfig.mode,
            source_ip: originalConfig.source_ip,
            target_ip: originalConfig.target_ip,
            speed_gbps: originalConfig.speed_gbps,
            burst_length: originalConfig.burst_length,
            request_type: newRequestType,
            end_time_ns: originalConfig.end_time_ns,
            source_die: originalConfig.source_die,
            target_die: originalConfig.target_die,
            die_pairs: originalConfig.die_pairs
          })
        })
      )

      message.success(`已切换为 ${newRequestType} 类型`)
      loadConfigs()
    } catch (error) {
      message.error('切换失败')
      console.error(error)
    }
  }

  // 批量切换所有配置的类型
  const handleBatchToggleType = async (targetType: 'R' | 'W' | 'toggle') => {
    if (configs.length === 0) {
      message.warning('没有可切换的配置')
      return
    }

    Modal.confirm({
      title: '确认批量切换',
      content: `确定要将所有 ${configs.length} 个配置的类型${
        targetType === 'toggle' ? '进行切换' : `切换为 ${targetType}`
      }吗？`,
      onOk: async () => {
        try {
          // 批量更新所有配置
          await Promise.all(
            configs.map(async (config: TrafficConfig) => {
              let newRequestType: 'R' | 'W'
              if (targetType === 'toggle') {
                newRequestType = config.request_type === 'R' ? 'W' : 'R'
              } else {
                newRequestType = targetType
              }

              // 如果类型相同，跳过更新
              if (newRequestType === config.request_type) return

              await updateTrafficConfig(topology, mode, config.id, {
                topology: config.topology,
                mode: config.mode,
                source_ip: config.source_ip,
                target_ip: config.target_ip,
                speed_gbps: config.speed_gbps,
                burst_length: config.burst_length,
                request_type: newRequestType,
                end_time_ns: config.end_time_ns,
                source_die: config.source_die,
                target_die: config.target_die,
                die_pairs: config.die_pairs
              })
            })
          )

          message.success('批量切换成功')
          loadConfigs()
        } catch (error) {
          message.error('批量切换失败')
          console.error(error)
        }
      }
    })
  }

  // 计算静态链路带宽
  const handleComputeBandwidth = async () => {
    if (configs.length === 0) {
      message.warning('请先添加流量配置')
      return
    }

    setComputing(true)
    try {
      const result = await computeStaticBandwidth({
        topology,
        mode,
        routing_type: routingType,
        d2d_config_file: mode === 'd2d' ? selectedD2DConfig : undefined
      })

      if (result.success) {
        // 根据模式显示不同的统计信息
        if (result.mode === 'd2d') {
          // D2D模式：显示所有Die中的最大带宽
          const stats = result.statistics as Record<string, { max_bandwidth: number }>
          const maxBw = Math.max(...Object.values(stats).map(s => s.max_bandwidth))
          message.success(`${result.message}\n最大带宽: ${maxBw.toFixed(2)} GB/s`)
        } else {
          // NoC模式
          const stats = result.statistics as { max_bandwidth: number }
          message.success(`${result.message}\n最大带宽: ${stats.max_bandwidth.toFixed(2)} GB/s`)
        }
        // 通过回调函数将带宽数据传递给父组件
        if (onBandwidthComputed) {
          onBandwidthComputed(result)
        }
      } else {
        message.error('带宽计算失败')
      }
    } catch (error: any) {
      message.error(error.response?.data?.detail || '带宽计算失败')
      console.error(error)
    } finally {
      setComputing(false)
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
        split_by_source: enableSplit,
        random_seed: useFixedSeed ? randomSeed : (Date.now() % (2**32 - 1)),
        filename: trafficFileName
      })

      if (result.split_files && result.split_files.length > 0) {
        setSplitFiles(result.split_files)
        setShowSplitModal(true)
        message.success(result.message + `，耗时 ${result.generation_time_ms.toFixed(0)}ms`)
      } else {
        message.success(
          `流量生成成功！共 ${result.total_lines} 行，耗时 ${result.generation_time_ms.toFixed(0)}ms`
        )
      }
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

  // 将配置按IP类型分组（不再合并不同配置）
  const groupedConfigs = configs.filter((config: TrafficConfig) => {
    // 过滤掉无效配置
    return config.source_ip && config.target_ip
  }).map((config: TrafficConfig) => {
    // 处理IP（支持字符串或数组）
    const sourceIps = Array.isArray(config.source_ip) ? config.source_ip : [config.source_ip]
    const targetIps = Array.isArray(config.target_ip) ? config.target_ip : [config.target_ip]

    // 提取第一个IP的类型作为组的代表
    const firstSourceIp = sourceIps[0]
    const firstTargetIp = targetIps[0]

    if (!firstSourceIp || !firstTargetIp) {
      console.error('Invalid config:', config)
      return null
    }

    const sourceFullType = firstSourceIp.includes('-')
      ? firstSourceIp.split('-')[1]
      : firstSourceIp
    const targetFullType = firstTargetIp.includes('-')
      ? firstTargetIp.split('-')[1]
      : firstTargetIp

    // 提取基础类型（去掉编号，如 gdma_0 -> gdma）
    const sourceBaseType = sourceFullType.split('_')[0]
    const targetBaseType = targetFullType.split('_')[0]

    // 对于IP列表，提取所有节点信息
    const extractNodeFromIp = (ip: string) => {
      return ip.includes('-') ? parseInt(ip.split('-')[0].replace('节点', '')) : null
    }

    // 创建 IP类型 -> 节点列表 的映射（从IP挂载中）
    const ipTypeToNodes: Record<string, number[]> = {}
    mounts.forEach(mount => {
      if (!ipTypeToNodes[mount.ip_type]) {
        ipTypeToNodes[mount.ip_type] = []
      }
      ipTypeToNodes[mount.ip_type].push(mount.node_id)
    })

    // 创建配置对应的显示组
    const details: Record<string, number[]> = {}

    // 处理所有源IP
    sourceIps.forEach(srcIp => {
      const srcFullType = srcIp.includes('-') ? srcIp.split('-')[1] : srcIp
      const srcNode = extractNodeFromIp(srcIp)

      if (srcNode !== null) {
        // 有明确的节点ID，使用它
        if (!details[srcFullType]) {
          details[srcFullType] = []
        }
        details[srcFullType].push(srcNode)
      } else if (ipTypeToNodes[srcFullType]) {
        // 没有节点ID，从IP挂载中查找该类型的所有节点
        details[srcFullType] = ipTypeToNodes[srcFullType]
      }
    })

    // 处理所有目标IP (非D2D模式)
    if (mode !== 'd2d') {
      targetIps.forEach(tgtIp => {
        const tgtFullType = tgtIp.includes('-') ? tgtIp.split('-')[1] : tgtIp
        const tgtNode = extractNodeFromIp(tgtIp)

        if (tgtNode !== null) {
          // 有明确的节点ID，使用它
          if (!details[tgtFullType]) {
            details[tgtFullType] = []
          }
          details[tgtFullType].push(tgtNode)
        } else if (ipTypeToNodes[tgtFullType]) {
          // 没有节点ID，从IP挂载中查找该类型的所有节点
          details[tgtFullType] = ipTypeToNodes[tgtFullType]
        }
      })
    }

    const groupData: any = {
      key: config.id,
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
    if (config.die_pairs && config.die_pairs.length > 0) {
      // 使用新的die_pairs字段
      // 所有die_pairs应该有相同的源DIE（按源DIE分组的配置）
      const firstPair = config.die_pairs[0]
      groupData.source_die = firstPair[0]
      groupData.die_pairs = config.die_pairs
      groupData.target_dies = new Set(config.die_pairs.map((pair: any) => pair[1]))
      groupData.targetDetailsByDie = {}

      // 为每个目标DIE处理目标IP
      config.die_pairs.forEach((pair: any) => {
        const [, targetDie] = pair

        if (!groupData.targetDetailsByDie[targetDie]) {
          groupData.targetDetailsByDie[targetDie] = {}
        }

        // 处理所有目标IP（每个目标DIE都有完整的目标IP列表）
        targetIps.forEach(tgtIp => {
          const tgtFullType = tgtIp.includes('-') ? tgtIp.split('-')[1] : tgtIp
          const tgtNode = extractNodeFromIp(tgtIp)

          if (!groupData.targetDetailsByDie[targetDie][tgtFullType]) {
            groupData.targetDetailsByDie[targetDie][tgtFullType] = []
          }

          if (tgtNode !== null) {
            // 有明确的节点ID，使用它
            groupData.targetDetailsByDie[targetDie][tgtFullType].push(tgtNode)
          } else if (ipTypeToNodes[tgtFullType]) {
            // 没有节点ID，从IP挂载中查找该类型的所有节点
            groupData.targetDetailsByDie[targetDie][tgtFullType] = ipTypeToNodes[tgtFullType]
          }
        })
      })
    } else if (config.source_die !== undefined && config.target_die !== undefined) {
      // 向后兼容旧格式
      groupData.source_die = config.source_die
      groupData.target_die = config.target_die
      groupData.target_dies = new Set([config.target_die])
      groupData.targetDetailsByDie = {}
      groupData.targetDetailsByDie[config.target_die] = {}

      // 处理所有目标IP
      const targetDieKey = config.target_die as number
      targetIps.forEach(tgtIp => {
        const tgtFullType = tgtIp.includes('-') ? tgtIp.split('-')[1] : tgtIp
        const tgtNode = extractNodeFromIp(tgtIp)

        if (!groupData.targetDetailsByDie[targetDieKey][tgtFullType]) {
          groupData.targetDetailsByDie[targetDieKey][tgtFullType] = []
        }

        if (tgtNode !== null) {
          // 有明确的节点ID，使用它
          groupData.targetDetailsByDie[targetDieKey][tgtFullType].push(tgtNode)
        } else if (ipTypeToNodes[tgtFullType]) {
          // 没有节点ID，从IP挂载中查找该类型的所有节点
          groupData.targetDetailsByDie[targetDieKey][tgtFullType] = ipTypeToNodes[tgtFullType]
        }
      })
    }

    return groupData
  }).filter(g => g !== null)

  // 对每组的详细信息节点ID进行排序
  groupedConfigs.forEach(g => {
    if (!g || !g.details) return
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

  // 编辑字段的处理函数
  const handleFieldUpdate = async (record: any, field: string, value: number) => {
    try {
      await Promise.all(
        record.configIds.map(async (configId: string) => {
          const originalConfig = configs.find((c: TrafficConfig) => c.id === configId)
          if (!originalConfig) return

          await updateTrafficConfig(topology, mode, configId, {
            topology: originalConfig.topology,
            mode: originalConfig.mode,
            source_ip: originalConfig.source_ip,
            target_ip: originalConfig.target_ip,
            speed_gbps: field === 'speed_gbps' ? value : originalConfig.speed_gbps,
            burst_length: field === 'burst_length' ? value : originalConfig.burst_length,
            request_type: originalConfig.request_type,
            end_time_ns: field === 'end_time_ns' ? value : originalConfig.end_time_ns,
            source_die: originalConfig.source_die,
            target_die: originalConfig.target_die,
            die_pairs: originalConfig.die_pairs
          })
        })
      )
      message.success('更新成功')
      loadConfigs()
    } catch (error) {
      message.error('更新失败')
      console.error(error)
    }
  }

  // IP类型到颜色的映射
  const getIPColor = (ipType: string): string => {
    const type = ipType.toLowerCase()
    switch (type) {
      case 'gdma':
      case 'sdma':
        return '#5B8FF9'  // 蓝色
      case 'cdma':
        return '#5AD8A6'  // 绿色
      case 'ddr':
      case 'l2m':
        return '#E8684A'  // 红色
      case 'npu':
        return '#722ed1'  // 紫色
      case 'pcie':
        return '#13c2c2'  // 青色
      case 'eth':
        return '#eb2f96'  // 粉色
      default:
        return '#8c8c8c'  // 灰色
    }
  }

  const columns = [
    {
      title: '源IP',
      dataIndex: 'sourceBaseType',
      key: 'sourceBaseType',
      width: 100,
      align: 'center' as const,
      render: (sourceBaseType: string) => (
        <Tag color={getIPColor(sourceBaseType)}>{sourceBaseType}</Tag>
      )
    },
    {
      title: '目标IP',
      dataIndex: 'targetBaseType',
      key: 'targetBaseType',
      width: 100,
      align: 'center' as const,
      render: (targetBaseType: string) => (
        <Tag color={getIPColor(targetBaseType)}>{targetBaseType}</Tag>
      )
    },
    ...(mode === 'd2d' ? [{
      title: '源DIE',
      dataIndex: 'source_die',
      key: 'source_die',
      width: 100,
      align: 'center' as const,
      render: (sourceDie: number) => {
        if (sourceDie === undefined || sourceDie === null) {
          return null
        }
        return <Tag color="purple">{sourceDie}</Tag>
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
      width: 100,
      align: 'center' as const,
      render: (speed: number, record: any) => (
        <InputNumber
          size="small"
          defaultValue={speed}
          min={0.1}
          max={256}
          step={0.1}
          style={{ width: 80 }}
          onChange={(value) => {
            if (value !== null && value !== speed) {
              handleFieldUpdate(record, 'speed_gbps', value)
            }
          }}
        />
      )
    },
    {
      title: 'Burst',
      dataIndex: 'burst_length',
      key: 'burst_length',
      width: 100,
      align: 'center' as const,
      render: (burst: number, record: any) => (
        <InputNumber
          size="small"
          defaultValue={burst}
          min={1}
          max={16}
          style={{ width: 60 }}
          onChange={(value) => {
            if (value !== null && value !== burst) {
              handleFieldUpdate(record, 'burst_length', value)
            }
          }}
        />
      )
    },
    {
      title: () => {
        const menuItems: MenuProps['items'] = [
          {
            key: 'toggle',
            label: '切换全部',
            onClick: () => handleBatchToggleType('toggle')
          },
          {
            key: 'setR',
            label: '全部切换为 R',
            onClick: () => handleBatchToggleType('R')
          },
          {
            key: 'setW',
            label: '全部切换为 W',
            onClick: () => handleBatchToggleType('W')
          }
        ]

        return (
          <Dropdown menu={{ items: menuItems }} trigger={['click']}>
            <span style={{ cursor: 'pointer', userSelect: 'none' }}>
              类型 <DownOutlined style={{ fontSize: 10 }} />
            </span>
          </Dropdown>
        )
      },
      dataIndex: 'request_type',
      key: 'request_type',
      width: 100,
      align: 'center' as const,
      render: (type: string, record: any) => (
        <Tag
          color={type === 'R' ? 'cyan' : 'orange'}
          style={{ cursor: 'pointer' }}
          onClick={() => handleToggleRequestType(record)}
        >
          {type}
        </Tag>
      )
    },
    {
      title: '结束时间',
      dataIndex: 'end_time_ns',
      key: 'end_time_ns',
      width: 100,
      align: 'center' as const,
      render: (time: number, record: any) => (
        <InputNumber
          size="small"
          defaultValue={time}
          min={100}
          max={100000}
          style={{ width: 80 }}
          onChange={(value) => {
            if (value !== null && value !== time) {
              handleFieldUpdate(record, 'end_time_ns', value)
            }
          }}
        />
      )
    },
    {
      title: '操作',
      key: 'action',
      width: 100,
      align: 'center' as const,
      render: (record: any) => (
        <Button
          type="link"
          danger
          icon={<DeleteOutlined />}
          size="small"
          onClick={async () => {
            try {
              // 删除该组的所有配置
              await Promise.all(record.configIds.map((id: string) => deleteTrafficConfig(topology, mode, id)))
              message.success('删除成功')
              loadConfigs()
            } catch (error) {
              message.error('删除失败')
            }
          }}
        >
          删除
        </Button>
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
          <Button
            size="small"
            danger
            icon={<ClearOutlined />}
            onClick={handleClearAll}
          >
            清空
          </Button>
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
                    {Array.from(record.target_dies as Set<number>).sort((a: number, b: number) => a - b).map((targetDie: number) => (
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

      {/* 计算静态链路带宽 */}
      <div style={{ marginTop: 16, width: '100%', textAlign: 'center' }}>
        <Space>
          <span>路由算法:</span>
          <Select
            value={routingType}
            onChange={setRoutingType}
            style={{ width: 100 }}
          >
            <Option value="XY">XY路由</Option>
            <Option value="YX">YX路由</Option>
          </Select>
          {mode === 'd2d' && (
            <>
              <span>D2D配置:</span>
              <Select
                value={selectedD2DConfig}
                onChange={setSelectedD2DConfig}
                style={{ width: 180 }}
                placeholder="选择D2D配置"
              >
                {d2dConfigs.map(config => (
                  <Option key={config.filename} value={config.filename}>
                    {config.filename.replace('.yaml', '')} ({config.num_dies}Die)
                  </Option>
                ))}
              </Select>
            </>
          )}
          <Button
            type="default"
            icon={<CalculatorOutlined />}
            onClick={handleComputeBandwidth}
            loading={computing}
            disabled={configs.length === 0 || (mode === 'd2d' && !selectedD2DConfig)}
          >
            计算静态链路带宽
          </Button>
        </Space>
      </div>

      {/* 生成数据流 */}
      <Space direction="vertical" style={{ marginTop: 16, width: '100%', alignItems: 'center' }}>
        <Space>
          <Input
            placeholder="输入文件名"
            value={trafficFileName}
            onChange={(e) => setTrafficFileName(e.target.value)}
            style={{ width: 600 }}
            suffix=".txt"
          />
        </Space>
        <Space size="large" style={{ width: 600 }}>
          <Checkbox
            checked={useFixedSeed}
            onChange={(e) => setUseFixedSeed(e.target.checked)}
          >
            使用固定随机种子
          </Checkbox>
          <Checkbox
            checked={enableSplit}
            onChange={(e) => setEnableSplit(e.target.checked)}
          >
            按源IP拆分流量文件
            <Tooltip title="生成后自动将流量文件按源IP拆分成多个独立文件，存放在以文件名命名的文件夹中。文件名格式：NoC模式为 master_p{ip_index}_x{x}_y{y}.txt，D2D模式为 master_d{die_id}_p{ip_index}_x{x}_y{y}.txt">
              <QuestionCircleOutlined style={{ marginLeft: 4, color: '#999' }} />
            </Tooltip>
          </Checkbox>
        </Space>
        {useFixedSeed && (
          <InputNumber
            value={randomSeed}
            onChange={(value) => setRandomSeed(value || 42)}
            min={0}
            style={{ width: 600 }}
            placeholder="输入随机种子（默认42）"
          />
        )}
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

      {/* 拆分文件列表对话框 */}
      <Modal
        title="拆分文件列表"
        open={showSplitModal}
        onCancel={() => setShowSplitModal(false)}
        footer={[
          <Button key="close" onClick={() => setShowSplitModal(false)}>
            关闭
          </Button>
        ]}
        width={700}
      >
        <div style={{ marginBottom: 16 }}>
          共生成 {splitFiles.length} 个拆分文件
        </div>
        <Table
          dataSource={splitFiles.map((file, idx) => ({ ...file, key: idx }))}
          columns={[
            {
              title: '文件名',
              dataIndex: 'filename',
              key: 'filename',
              width: 300
            },
            {
              title: '行数',
              dataIndex: 'count',
              key: 'count',
              width: 100,
              align: 'center' as const
            },
            {
              title: '操作',
              key: 'action',
              width: 100,
              align: 'center' as const,
              render: (_, record) => (
                <Button
                  type="link"
                  size="small"
                  onClick={() => {
                    const folderName = trafficFileName
                    downloadTrafficFile(`${folderName}/${record.filename}`)
                  }}
                >
                  下载
                </Button>
              )
            }
          ]}
          size="small"
          pagination={false}
          scroll={{ y: 400 }}
        />
      </Modal>
    </Card>
  )
}

export default TrafficConfigPanel
