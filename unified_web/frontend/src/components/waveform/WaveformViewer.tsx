/**
 * 波形查看器主组件 - 统一的波形查看界面
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors, DragEndEvent } from '@dnd-kit/core';
import { arrayMove, SortableContext, sortableKeyboardCoordinates, horizontalListSortingStrategy, useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Card, Spin, Empty, Button, Space, Alert, Collapse, Row, Col, Tabs, Radio, Tag } from 'antd';
import { ReloadOutlined, LineChartOutlined, HeatMapOutlined, ClearOutlined, HolderOutlined } from '@ant-design/icons';

// 可拖拽的信号标签组件
interface SortableTagProps {
  id: string;
  label: string;
  color: string;
  onRemove: () => void;
}

function SortableTag({ id, label, color, onRemove }: SortableTagProps) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id });

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
    cursor: 'grab',
  };

  return (
    <Tag
      ref={setNodeRef}
      style={{ ...style, borderColor: color, color }}
      closable
      onClose={onRemove}
    >
      <HolderOutlined {...attributes} {...listeners} style={{ marginRight: 4, cursor: 'grab' }} />
      {label}
    </Tag>
  );
}
import WaveformChart from './WaveformChart';
import PacketSelector from './PacketSelector';
import FIFOWaveformChart from './FIFOWaveformChart';
import NodeArchitectureDiagram from './NodeArchitectureDiagram';
import SimpleTopologyGraph from './SimpleTopologyGraph';
import {
  getWaveformData,
  checkWaveformData,
  getTopologyData,
  getActiveIPs,
  type WaveformResponse,
  type WaveformCheckResponse,
  type TopologyData,
  type ActiveIPsResponse,
} from '@/api/waveform';
import { getFIFOWaveform, type FIFOWaveformResponse } from '@/api/fifoWaveform';

interface Props {
  experimentId: number;
  resultId: number;
}

// localStorage key 生成
const getStorageKey = (experimentId: number, resultId: number) =>
  `waveform-state-${experimentId}-${resultId}`;

// 状态持久化接口
interface PersistedState {
  activeTab: 'packet' | 'fifo';
  selectedPacketIds: number[];
  selectedNode: number | null;
  selectedChannel: string;
  selectedFifos: string[];
  fifoClickedPacketId: number | null;
  expandedRspSignals: string[];  // 已展开的 rsp 信号（如 ["IQ_TR.rsp"]）
  expandedReqSignals: string[];  // 已展开的 req 信号（如 ["IQ_TR.req"]）
  expandedDataSignals: string[]; // 已展开的 data 信号（如 ["IQ_TR.data"]）
}

// 从 localStorage 加载状态
const loadPersistedState = (experimentId: number, resultId: number): Partial<PersistedState> => {
  try {
    const key = getStorageKey(experimentId, resultId);
    const saved = localStorage.getItem(key);
    if (saved) {
      return JSON.parse(saved);
    }
  } catch (e) {
    console.error('加载波形状态失败', e);
  }
  return {};
};

// 保存状态到 localStorage
const savePersistedState = (experimentId: number, resultId: number, state: PersistedState) => {
  try {
    const key = getStorageKey(experimentId, resultId);
    localStorage.setItem(key, JSON.stringify(state));
  } catch (e) {
    console.error('保存波形状态失败', e);
  }
};

export default function WaveformViewer({ experimentId, resultId }: Props) {
  // 加载持久化状态
  const initialState = useMemo(() => loadPersistedState(experimentId, resultId), [experimentId, resultId]);

  const [loading, setLoading] = useState(true);
  const [checkResult, setCheckResult] = useState<WaveformCheckResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'packet' | 'fifo'>(initialState.activeTab || 'packet');

  // 请求波形相关状态
  const [waveformData, setWaveformData] = useState<WaveformResponse | null>(null);
  const [selectedPacketIds, setSelectedPacketIds] = useState<number[]>(initialState.selectedPacketIds || []);
  const [loadingWaveform, setLoadingWaveform] = useState(false);

  // FIFO波形相关状态
  const [selectedNode, setSelectedNode] = useState<number | null>(initialState.selectedNode ?? null);
  const [selectedChannel, setSelectedChannel] = useState<string>(initialState.selectedChannel || 'data');
  const [selectedFifos, setSelectedFifos] = useState<string[]>(initialState.selectedFifos || []);
  const [fifoWaveformData, setFifoWaveformData] = useState<FIFOWaveformResponse | null>(null);
  const [loadingFifoWaveform, setLoadingFifoWaveform] = useState(false);
  const [fifoError, setFifoError] = useState<string | null>(null);
  const [expandedRspSignals, setExpandedRspSignals] = useState<string[]>(initialState.expandedRspSignals || []);
  const [expandedReqSignals, setExpandedReqSignals] = useState<string[]>(initialState.expandedReqSignals || []);
  const [expandedDataSignals, setExpandedDataSignals] = useState<string[]>(initialState.expandedDataSignals || []);

  // 拓扑数据
  const [topoData, setTopoData] = useState<TopologyData | null>(null);
  const [loadingTopo, setLoadingTopo] = useState(false);

  // 活跃IP数据
  const [activeIPsData, setActiveIPsData] = useState<Record<number, string[]>>({});

  // FIFO 波形中点击的 packet（用于在下方显示请求波形）
  const [fifoClickedPacketId, setFifoClickedPacketId] = useState<number | null>(initialState.fifoClickedPacketId ?? null);
  const [fifoClickedWaveform, setFifoClickedWaveform] = useState<WaveformResponse | null>(null);
  const [loadingFifoClickedWaveform, setLoadingFifoClickedWaveform] = useState(false);

  // 当 experimentId 或 resultId 变化时，重置所有状态并重新加载
  useEffect(() => {
    // 重新加载持久化状态
    const newState = loadPersistedState(experimentId, resultId);

    // 重置所有状态
    setActiveTab(newState.activeTab || 'packet');
    setSelectedPacketIds(newState.selectedPacketIds || []);
    setSelectedNode(newState.selectedNode ?? null);
    setSelectedChannel(newState.selectedChannel || 'data');
    setSelectedFifos(newState.selectedFifos || []);
    setExpandedRspSignals(newState.expandedRspSignals || []);
    setExpandedReqSignals(newState.expandedReqSignals || []);
    setExpandedDataSignals(newState.expandedDataSignals || []);
    setFifoClickedPacketId(newState.fifoClickedPacketId ?? null);

    // 清空波形数据
    setWaveformData(null);
    setFifoWaveformData(null);
    setFifoClickedWaveform(null);
    setTopoData(null);
    setActiveIPsData({});
    setCheckResult(null);
    setFifoError(null);

    // 重新加载数据
    checkAvailability();
    loadTopology();
  }, [experimentId, resultId]);

  const checkAvailability = async () => {
    setLoading(true);
    try {
      const result = await checkWaveformData(experimentId, resultId);
      setCheckResult(result);
    } catch (error) {
      console.error('检查波形数据失败', error);
      setCheckResult({ available: false, message: '检查失败' });
    } finally {
      setLoading(false);
    }
  };

  // 加载拓扑数据和活跃IP
  const loadTopology = async () => {
    setLoadingTopo(true);
    try {
      // 并行加载拓扑和活跃IP
      const [topoResult, activeIPsResult] = await Promise.all([
        getTopologyData(experimentId, resultId),
        getActiveIPs(experimentId, resultId),
      ]);
      setTopoData(topoResult);
      setActiveIPsData(activeIPsResult.active_ips);
    } catch (error) {
      console.error('加载拓扑数据失败', error);
      setTopoData(null);
      setActiveIPsData({});
    } finally {
      setLoadingTopo(false);
    }
  };

  // 加载波形数据
  const loadWaveform = useCallback(async () => {
    if (selectedPacketIds.length === 0) {
      setWaveformData(null);
      return;
    }

    setLoadingWaveform(true);
    try {
      const data = await getWaveformData(experimentId, resultId, {
        packetIds: selectedPacketIds,
      });
      setWaveformData(data);
    } catch (error) {
      console.error('加载波形数据失败', error);
      setWaveformData(null);
    } finally {
      setLoadingWaveform(false);
    }
  }, [experimentId, resultId, selectedPacketIds]);

  // 当选择的packet改变时自动加载
  useEffect(() => {
    if (selectedPacketIds.length > 0) {
      loadWaveform();
    } else {
      setWaveformData(null);
    }
  }, [selectedPacketIds, loadWaveform]);

  // 节点点击处理 - 保留之前的端口选择
  const handleNodeClick = (nodeId: number) => {
    setSelectedNode(nodeId);
    // 不清空 selectedFifos，保留之前的端口选择
  };

  // FIFO点击处理（toggle选择）- 组合 nodeId.fifoId.channel
  const handleFifoSelect = (fifoId: string) => {
    if (selectedNode === null) return;
    const fullId = `${selectedNode}.${fifoId}.${selectedChannel}`;
    console.log('[DEBUG] handleFifoSelect:', fullId);
    // 使用函数式更新确保连续调用时状态正确
    setSelectedFifos(prev => {
      if (prev.includes(fullId)) {
        // 取消选择
        return prev.filter(f => f !== fullId);
      } else {
        // 添加选择
        return [...prev, fullId];
      }
    });
  };

  // 删除单个波形选择
  const handleRemoveFifo = (fullId: string) => {
    setSelectedFifos(selectedFifos.filter(f => f !== fullId));
  };

  // 清空所有波形选择
  const handleClearAllFifos = () => {
    setSelectedFifos([]);
  };

  // 格式化显示名称: "3.IQ_TD.data" -> "Node_3.IQ_TD.data"
  const formatFifoDisplayName = (fullId: string) => {
    const parts = fullId.split('.');
    if (parts.length === 3) {
      return `Node_${parts[0]}.${parts[1]}.${parts[2]}`;
    }
    return fullId;
  };

  // 获取通道颜色
  const getChannelColor = (channel: string) => {
    switch (channel) {
      case 'req': return '#ef4444';
      case 'rsp': return '#22c55e';
      case 'data': return '#3b82f6';
      default: return '#999';
    }
  };

  // 拖拽传感器配置
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );

  // 拖拽结束处理
  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      const oldIndex = selectedFifos.indexOf(active.id as string);
      const newIndex = selectedFifos.indexOf(over.id as string);
      setSelectedFifos(arrayMove(selectedFifos, oldIndex, newIndex));
    }
  };

  // 从 selectedFifos 中按节点分组提取 FIFO 和 flit 类型
  // 格式: nodeId.fifoId.channel -> { nodeId: { fifos: [], flitTypes: [] } }
  const getSelectedFifoInfoByNode = useCallback(() => {
    const result: Record<number, { fifos: Set<string>; flitTypes: Set<string>; fullIds: string[] }> = {};
    selectedFifos.forEach(s => {
      const parts = s.split('.');
      if (parts.length === 3) {
        const nodeId = parseInt(parts[0]);
        const fifoId = parts[1];
        const flitType = parts[2];
        if (!result[nodeId]) {
          result[nodeId] = { fifos: new Set(), flitTypes: new Set(), fullIds: [] };
        }
        result[nodeId].fifos.add(fifoId);
        result[nodeId].flitTypes.add(flitType);
        result[nodeId].fullIds.push(`${fifoId}.${flitType}`);
      }
    });
    return result;
  }, [selectedFifos]);

  // 加载FIFO波形数据 - 支持多节点
  const loadFifoWaveform = useCallback(async () => {
    if (selectedFifos.length === 0) {
      setFifoWaveformData(null);
      return;
    }

    const nodeInfoMap = getSelectedFifoInfoByNode();
    const nodeIds = Object.keys(nodeInfoMap).map(Number);
    if (nodeIds.length === 0) {
      setFifoWaveformData(null);
      return;
    }

    setLoadingFifoWaveform(true);
    setFifoError(null);
    try {
      // 并行请求所有节点的波形数据
      const promises = nodeIds.map(async (nodeId) => {
        const info = nodeInfoMap[nodeId];
        const data = await getFIFOWaveform(
          experimentId,
          resultId,
          nodeId,
          Array.from(info.fifos),
          Array.from(info.flitTypes),
          {
            expandRspSignals: expandedRspSignals,
            expandReqSignals: expandedReqSignals,
            expandDataSignals: expandedDataSignals
          }
        );
        // 过滤只保留用户选择的 fifo.flit_type 组合
        const filteredSignals = data.signals.filter(signal => {
          // signal.name 格式: "Node_X.FIFO.flit_type" 或 "Node_X.FIFO.flit_type.sub_type"
          const parts = signal.name.split('.');
          if (parts.length >= 3) {
            const fifoType = parts[1];
            const flitType = parts[2];
            // 基本匹配
            if (info.fullIds.includes(`${fifoType}.${flitType}`)) {
              return true;
            }
            // 展开的子类型（如 rsp.CompData, req.new, data.0）也需要通过
            if (parts.length >= 4) {
              return info.fullIds.includes(`${fifoType}.${flitType}`);
            }
          }
          return false;
        });
        return { ...data, signals: filteredSignals };
      });

      const results = await Promise.all(promises);

      // 合并所有节点的波形数据
      const mergedSignals = results.flatMap(r => r.signals);
      const timeRanges = results.map(r => r.time_range);
      const mergedTimeRange = {
        start_ns: Math.min(...timeRanges.map(t => t.start_ns)),
        end_ns: Math.max(...timeRanges.map(t => t.end_ns)),
      };
      const mergedAvailableFifos = results.flatMap(r => r.available_fifos);

      setFifoWaveformData({
        signals: mergedSignals,
        time_range: mergedTimeRange,
        available_fifos: mergedAvailableFifos,
      });
    } catch (err) {
      console.error('加载FIFO波形数据失败', err);
      setFifoError(err instanceof Error ? err.message : '加载失败');
      setFifoWaveformData(null);
    } finally {
      setLoadingFifoWaveform(false);
    }
  }, [experimentId, resultId, selectedFifos, getSelectedFifoInfoByNode, expandedRspSignals, expandedReqSignals, expandedDataSignals]);

  // 当选择的FIFO或展开状态改变时自动加载
  useEffect(() => {
    if (selectedFifos.length > 0) {
      loadFifoWaveform();
    } else {
      setFifoWaveformData(null);
    }
  }, [selectedFifos, loadFifoWaveform, expandedRspSignals, expandedReqSignals, expandedDataSignals]);

  // 保存状态到 localStorage
  useEffect(() => {
    savePersistedState(experimentId, resultId, {
      activeTab,
      selectedPacketIds,
      selectedNode,
      selectedChannel,
      selectedFifos,
      fifoClickedPacketId,
      expandedRspSignals,
      expandedReqSignals,
      expandedDataSignals,
    });
  }, [experimentId, resultId, activeTab, selectedPacketIds, selectedNode, selectedChannel, selectedFifos, fifoClickedPacketId, expandedRspSignals, expandedReqSignals, expandedDataSignals]);

  // 处理 flit 点击 - 在下方显示请求波形
  const handleFlitClick = useCallback((packetId: number) => {
    setFifoClickedPacketId(packetId);
  }, []);

  // 处理展开/折叠信号（支持 rsp, req, data）
  const handleToggleExpand = useCallback((signalKey: string) => {
    // signalKey 格式: "IQ_TR.rsp", "IQ_TR.req", "IQ_TR.data"
    if (signalKey.endsWith('.rsp')) {
      setExpandedRspSignals(prev => {
        if (prev.includes(signalKey)) {
          return prev.filter(s => s !== signalKey);
        } else {
          return [...prev, signalKey];
        }
      });
    } else if (signalKey.endsWith('.req')) {
      setExpandedReqSignals(prev => {
        if (prev.includes(signalKey)) {
          return prev.filter(s => s !== signalKey);
        } else {
          return [...prev, signalKey];
        }
      });
    } else if (signalKey.endsWith('.data')) {
      setExpandedDataSignals(prev => {
        if (prev.includes(signalKey)) {
          return prev.filter(s => s !== signalKey);
        } else {
          return [...prev, signalKey];
        }
      });
    }
  }, []);

  // 加载点击的 packet 的波形数据
  useEffect(() => {
    if (fifoClickedPacketId === null) {
      setFifoClickedWaveform(null);
      return;
    }

    const loadClickedWaveform = async () => {
      setLoadingFifoClickedWaveform(true);
      try {
        const data = await getWaveformData(experimentId, resultId, {
          packetIds: [fifoClickedPacketId],
        });
        setFifoClickedWaveform(data);
      } catch (error) {
        console.error('加载请求波形失败', error);
        setFifoClickedWaveform(null);
      } finally {
        setLoadingFifoClickedWaveform(false);
      }
    };

    loadClickedWaveform();
  }, [experimentId, resultId, fifoClickedPacketId]);

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 50 }}>
        <Spin tip="检查波形数据..." />
      </div>
    );
  }

  if (!checkResult?.available) {
    return (
      <Alert
        type="warning"
        message="波形数据不可用"
        description={checkResult?.message || '未找到波形数据文件，请确保仿真时已启用波形日志导出。'}
        showIcon
      />
    );
  }

  const stats = checkResult.stats;

  // Tabs 配置
  const tabItems = [
    {
      key: 'packet',
      label: (
        <span>
          <LineChartOutlined />
          <span style={{ marginLeft: 8 }}>请求波形</span>
        </span>
      ),
      children: (
        <div>
          {/* 请求选择器 */}
          <Collapse
            defaultActiveKey={['selector']}
            items={[
              {
                key: 'selector',
                label: '选择要查看的请求',
                children: (
                  <PacketSelector
                    experimentId={experimentId}
                    resultId={resultId}
                    selectedPacketIds={selectedPacketIds}
                    onSelectionChange={setSelectedPacketIds}
                    maxPackets={20}
                  />
                ),
              },
            ]}
            style={{ marginBottom: 16 }}
          />

          {/* 波形图 */}
          <Card
            title={
              <Space>
                <LineChartOutlined />
                <span>请求波形</span>
              </Space>
            }
            extra={
              <Button
                icon={<ReloadOutlined />}
                onClick={loadWaveform}
                disabled={selectedPacketIds.length === 0}
                loading={loadingWaveform}
              >
                刷新
              </Button>
            }
          >
            {loadingWaveform ? (
              <div style={{ textAlign: 'center', padding: 50 }}>
                <Spin tip="加载波形数据..." />
              </div>
            ) : !waveformData ? (
              <Empty description="请选择要查看的请求" />
            ) : waveformData.signals.length === 0 ? (
              <Empty description="所选请求无波形数据" />
            ) : (
              <WaveformChart
                signals={waveformData.signals}
                timeRange={waveformData.time_range}
                stages={waveformData.stages}
              />
            )}
          </Card>
        </div>
      ),
    },
    {
      key: 'fifo',
      label: (
        <span>
          <HeatMapOutlined />
          <span style={{ marginLeft: 8 }}>端口波形</span>
        </span>
      ),
      children: (
        <div>
          {/* 上半部分：拓扑图 + 微架构图 */}
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={12}>
              <Card title="拓扑图" bodyStyle={{ height: '500px', padding: 8, overflow: 'auto' }}>
                {loadingTopo ? (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                    <Spin tip="加载拓扑数据..." />
                  </div>
                ) : topoData ? (
                  <SimpleTopologyGraph
                    data={topoData}
                    activeIPs={activeIPsData}
                    selectedNode={selectedNode}
                    onNodeClick={handleNodeClick}
                  />
                ) : (
                  <Empty description="拓扑数据加载失败" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                )}
              </Card>
            </Col>
            <Col span={12}>
              <Card
                title={
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span>{selectedNode !== null ? `节点 ${selectedNode} 内部结构` : '节点内部结构'}</span>
                    <Radio.Group
                      value={selectedChannel}
                      onChange={(e) => setSelectedChannel(e.target.value)}
                      size="small"
                      optionType="button"
                      buttonStyle="solid"
                    >
                      <Radio.Button value="req" style={{ backgroundColor: selectedChannel === 'req' ? '#ef4444' : undefined, borderColor: '#ef4444', color: selectedChannel === 'req' ? '#fff' : '#ef4444' }}>req</Radio.Button>
                      <Radio.Button value="rsp" style={{ backgroundColor: selectedChannel === 'rsp' ? '#22c55e' : undefined, borderColor: '#22c55e', color: selectedChannel === 'rsp' ? '#fff' : '#22c55e' }}>rsp</Radio.Button>
                      <Radio.Button value="data" style={{ backgroundColor: selectedChannel === 'data' ? '#3b82f6' : undefined, borderColor: '#3b82f6', color: selectedChannel === 'data' ? '#fff' : '#3b82f6' }}>data</Radio.Button>
                    </Radio.Group>
                  </div>
                }
                bodyStyle={{ height: '500px', padding: 8, overflow: 'auto' }}
              >
                <NodeArchitectureDiagram
                  selectedNode={selectedNode}
                  selectedFifos={selectedFifos}
                  selectedChannel={selectedChannel}
                  onFifoSelect={handleFifoSelect}
                  activeIPs={selectedNode !== null ? (activeIPsData[selectedNode] || []) : []}
                />
              </Card>
            </Col>
          </Row>

          {/* 波形图 */}
          <Card
            title={
              <Space>
                <HeatMapOutlined />
                <span>端口波形</span>
                {selectedFifos.length > 0 && (
                  <span style={{ fontSize: 12, color: '#999' }}>
                    ({selectedFifos.length} 个信号)
                  </span>
                )}
              </Space>
            }
            extra={
              selectedFifos.length > 0 && (
                <Button
                  size="small"
                  icon={<ClearOutlined />}
                  onClick={handleClearAllFifos}
                  danger
                >
                  清空
                </Button>
              )
            }
          >
            {/* 已选波形列表 - 可拖拽排序 */}
            {selectedFifos.length > 0 && (
              <div style={{ marginBottom: 12, padding: '8px 12px', background: '#f5f5f5', borderRadius: 4 }}>
                <div style={{ marginBottom: 4, fontSize: 12, color: '#666' }}>已选信号（可拖拽排序）：</div>
                <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
                  <SortableContext items={selectedFifos} strategy={horizontalListSortingStrategy}>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                      {selectedFifos.map(fullId => {
                        const parts = fullId.split('.');
                        const channel = parts[2] || '';
                        return (
                          <SortableTag
                            key={fullId}
                            id={fullId}
                            label={formatFifoDisplayName(fullId)}
                            color={getChannelColor(channel)}
                            onRemove={() => handleRemoveFifo(fullId)}
                          />
                        );
                      })}
                    </div>
                  </SortableContext>
                </DndContext>
              </div>
            )}

            {fifoError ? (
              <Alert
                type="error"
                message="加载失败"
                description={fifoError}
                showIcon
                style={{ marginBottom: 16 }}
              />
            ) : null}

            {!fifoWaveformData || fifoWaveformData.signals.length === 0 ? (
              loadingFifoWaveform ? (
                <div style={{ textAlign: 'center', padding: 50 }}>
                  <Spin tip="加载端口波形数据..." />
                </div>
              ) : (
                <Empty
                  description={
                    selectedNode === null
                      ? '请在拓扑图中选择节点'
                      : selectedFifos.length === 0
                      ? '请在微架构图中选择端口'
                      : '所选端口无数据'
                  }
                />
              )
            ) : (
              <div style={{ minHeight: 400 }}>
                <Spin spinning={loadingFifoWaveform} tip="更新中...">
                  <FIFOWaveformChart
                    signals={fifoWaveformData.signals}
                    timeRange={fifoWaveformData.time_range}
                    onFlitClick={handleFlitClick}
                    expandedRspSignals={expandedRspSignals}
                    expandedReqSignals={expandedReqSignals}
                    expandedDataSignals={expandedDataSignals}
                    onToggleExpand={handleToggleExpand}
                  />
                </Spin>
              </div>
            )}
          </Card>

          {/* 点击 flit 后显示的请求波形 */}
          {fifoClickedPacketId !== null && (
            <Card
              title={
                <Space>
                  <LineChartOutlined />
                  <span>请求 #{fifoClickedPacketId} 波形</span>
                </Space>
              }
              extra={
                <Button
                  size="small"
                  onClick={() => setFifoClickedPacketId(null)}
                >
                  关闭
                </Button>
              }
              style={{ marginTop: 16 }}
            >
              {loadingFifoClickedWaveform ? (
                <div style={{ textAlign: 'center', padding: 30 }}>
                  <Spin tip="加载请求波形..." />
                </div>
              ) : !fifoClickedWaveform || fifoClickedWaveform.signals.length === 0 ? (
                <Empty description="该请求无波形数据" />
              ) : (
                <WaveformChart
                  signals={fifoClickedWaveform.signals}
                  timeRange={fifoClickedWaveform.time_range}
                  stages={fifoClickedWaveform.stages}
                />
              )}
            </Card>
          )}
        </div>
      ),
    },
  ];

  return (
    <div>
      {/* 波形类型切换 */}
      <Tabs
        activeKey={activeTab}
        onChange={(key) => setActiveTab(key as 'packet' | 'fifo')}
        items={tabItems}
      />
    </div>
  );
}
