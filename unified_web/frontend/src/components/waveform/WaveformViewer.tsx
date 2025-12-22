/**
 * 波形查看器主组件 - 统一的波形查看界面
 */

import { useState, useEffect, useCallback } from 'react';
import { Card, Spin, Empty, Button, Space, Alert, Collapse, Statistic, Row, Col, Tabs, Checkbox } from 'antd';
import { ReloadOutlined, LineChartOutlined, HeatMapOutlined } from '@ant-design/icons';
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

export default function WaveformViewer({ experimentId, resultId }: Props) {
  const [loading, setLoading] = useState(true);
  const [checkResult, setCheckResult] = useState<WaveformCheckResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'packet' | 'fifo'>('packet');

  // 请求波形相关状态
  const [waveformData, setWaveformData] = useState<WaveformResponse | null>(null);
  const [selectedPacketIds, setSelectedPacketIds] = useState<number[]>([]);
  const [loadingWaveform, setLoadingWaveform] = useState(false);

  // FIFO波形相关状态
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [selectedFifos, setSelectedFifos] = useState<string[]>([]);
  const [selectedFlitTypes, setSelectedFlitTypes] = useState<string[]>(['data']);
  const [fifoWaveformData, setFifoWaveformData] = useState<FIFOWaveformResponse | null>(null);
  const [loadingFifoWaveform, setLoadingFifoWaveform] = useState(false);
  const [fifoError, setFifoError] = useState<string | null>(null);

  // 拓扑数据
  const [topoData, setTopoData] = useState<TopologyData | null>(null);
  const [loadingTopo, setLoadingTopo] = useState(false);

  // 活跃IP数据
  const [activeIPsData, setActiveIPsData] = useState<Record<number, string[]>>({});

  // 检查波形数据可用性
  useEffect(() => {
    checkAvailability();
  }, [experimentId, resultId]);

  // 加载拓扑数据
  useEffect(() => {
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

  // 节点点击处理
  const handleNodeClick = (nodeId: number) => {
    setSelectedNode(nodeId);
    setSelectedFifos([]); // 切换节点时清空FIFO选择
  };

  // FIFO点击处理（toggle选择）
  const handleFifoSelect = (fifo: string) => {
    if (selectedFifos.includes(fifo)) {
      // 取消选择
      setSelectedFifos(selectedFifos.filter(f => f !== fifo));
    } else {
      // 添加选择
      setSelectedFifos([...selectedFifos, fifo]);
    }
  };

  // 加载FIFO波形数据
  const loadFifoWaveform = useCallback(async () => {
    if (selectedFifos.length === 0 || selectedNode === null) {
      setFifoWaveformData(null);
      return;
    }

    setLoadingFifoWaveform(true);
    setFifoError(null);
    try {
      const data = await getFIFOWaveform(
        experimentId,
        resultId,
        selectedNode,
        selectedFifos,
        selectedFlitTypes.length > 0 ? selectedFlitTypes : undefined
      );
      setFifoWaveformData(data);
    } catch (err) {
      console.error('加载FIFO波形数据失败', err);
      setFifoError(err instanceof Error ? err.message : '加载失败');
      setFifoWaveformData(null);
    } finally {
      setLoadingFifoWaveform(false);
    }
  }, [experimentId, resultId, selectedNode, selectedFifos, selectedFlitTypes]);

  // 当选择的节点、FIFO或Flit类型改变时自动加载
  useEffect(() => {
    if (selectedFifos.length > 0 && selectedNode !== null) {
      loadFifoWaveform();
    } else {
      setFifoWaveformData(null);
    }
  }, [selectedNode, selectedFifos, selectedFlitTypes, loadFifoWaveform]);

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
              <Card title={selectedNode !== null ? `节点 ${selectedNode} 内部结构` : '节点内部结构'} bodyStyle={{ height: '500px', padding: 8, overflow: 'auto' }}>
                <NodeArchitectureDiagram
                  selectedNode={selectedNode}
                  selectedFifos={selectedFifos}
                  onFifoSelect={handleFifoSelect}
                  activeIPs={selectedNode !== null ? (activeIPsData[selectedNode] || []) : []}
                />
              </Card>
            </Col>
          </Row>

          {/* Flit类型过滤 */}
          {selectedNode !== null && (
            <Card
              title="Flit类型过滤"
              size="small"
              style={{ marginBottom: 16 }}
              extra={
                <Space>
                  <Button
                    size="small"
                    onClick={() => setSelectedFlitTypes(['req', 'rsp', 'data'])}
                  >
                    全选
                  </Button>
                  <Button size="small" onClick={() => setSelectedFlitTypes(['data'])}>
                    重置
                  </Button>
                </Space>
              }
            >
              <Checkbox.Group
                value={selectedFlitTypes}
                onChange={(values) => setSelectedFlitTypes(values as string[])}
                style={{ width: '100%' }}
              >
                <Space>
                  <Checkbox value="req">请求 (req)</Checkbox>
                  <Checkbox value="rsp">响应 (rsp)</Checkbox>
                  <Checkbox value="data">数据 (data)</Checkbox>
                </Space>
              </Checkbox.Group>
              {selectedFlitTypes.length === 0 && (
                <div style={{ color: '#999', fontSize: 12, marginTop: 8 }}>
                  未选择时显示所有类型
                </div>
              )}
            </Card>
          )}

          {/* 波形图 */}
          <Card
            title={
              <Space>
                <HeatMapOutlined />
                <span>端口波形</span>
                {selectedFifos.length > 0 && (
                  <span style={{ fontSize: 12, color: '#999' }}>
                    ({selectedFifos.length} 个端口, {selectedFlitTypes.join('/')})
                  </span>
                )}
              </Space>
            }
          >
            {fifoError ? (
              <Alert
                type="error"
                message="加载失败"
                description={fifoError}
                showIcon
                style={{ marginBottom: 16 }}
              />
            ) : null}

            {loadingFifoWaveform ? (
              <div style={{ textAlign: 'center', padding: 50 }}>
                <Spin tip="加载端口波形数据..." />
              </div>
            ) : !fifoWaveformData || fifoWaveformData.signals.length === 0 ? (
              <Empty
                description={
                  selectedNode === null
                    ? '请在拓扑图中选择节点'
                    : selectedFifos.length === 0
                    ? '请在微架构图中选择端口'
                    : '所选端口无数据'
                }
              />
            ) : (
              <FIFOWaveformChart
                signals={fifoWaveformData.signals}
                timeRange={fifoWaveformData.time_range}
              />
            )}
          </Card>
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
