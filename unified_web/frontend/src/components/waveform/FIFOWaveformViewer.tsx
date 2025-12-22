/**
 * FIFO波形查看器主组件
 */

import { useState, useEffect } from 'react';
import { Card, Spin, Empty, Collapse, Alert, Space, Select } from 'antd';
import { LineChartOutlined } from '@ant-design/icons';
import FIFOWaveformChart from './FIFOWaveformChart';
import FIFOSelector from './FIFOSelector';
import { getFIFOWaveform, type FIFOWaveformResponse } from '@/api/fifoWaveform';

interface Props {
  experimentId: number;
  resultId: number;
  nodeId?: number;  // 可选，如果未提供则显示节点选择器
}

export default function FIFOWaveformViewer({ experimentId, resultId, nodeId: propNodeId }: Props) {
  const [selectedNode, setSelectedNode] = useState<number>(propNodeId || 0);
  const [selectedFifos, setSelectedFifos] = useState<string[]>([]);
  const [waveformData, setWaveformData] = useState<FIFOWaveformResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 使用 prop 提供的 nodeId 或用户选择的节点
  const currentNodeId = propNodeId !== undefined ? propNodeId : selectedNode;

  // 当节点或选择的FIFO改变时自动加载波形数据
  useEffect(() => {
    if (selectedFifos.length === 0 || currentNodeId === 0) {
      setWaveformData(null);
      return;
    }

    loadWaveform();
  }, [experimentId, resultId, currentNodeId, selectedFifos]);

  const loadWaveform = async () => {
    if (selectedFifos.length === 0 || currentNodeId === 0) return;

    setLoading(true);
    setError(null);
    try {
      const data = await getFIFOWaveform(
        experimentId,
        resultId,
        currentNodeId,
        selectedFifos
      );
      setWaveformData(data);
    } catch (err) {
      console.error('加载FIFO波形数据失败', err);
      setError(err instanceof Error ? err.message : '加载失败');
      setWaveformData(null);
    } finally {
      setLoading(false);
    }
  };

  // 生成节点选项（0-63 for 8x8 topology）
  const nodeOptions = Array.from({ length: 64 }, (_, i) => ({
    label: `节点 ${i}`,
    value: i,
  }));

  return (
    <div>
      {/* 节点选择器（如果未通过prop提供nodeId） */}
      {propNodeId === undefined && (
        <Card size="small" style={{ marginBottom: 16 }}>
          <Space>
            <strong>选择节点:</strong>
            <Select
              value={selectedNode}
              onChange={setSelectedNode}
              options={nodeOptions}
              style={{ width: 200 }}
              showSearch
              placeholder="选择要查看的节点"
            />
          </Space>
        </Card>
      )}

      {/* FIFO选择器 */}
      <Collapse
        defaultActiveKey={['selector']}
        items={[
          {
            key: 'selector',
            label: `选择要查看的FIFO (节点 ${currentNodeId})`,
            children: currentNodeId === 0 ? (
              <Empty description="请先选择节点" />
            ) : (
              <FIFOSelector
                experimentId={experimentId}
                resultId={resultId}
                nodeId={currentNodeId}
                selectedFifos={selectedFifos}
                onSelectionChange={setSelectedFifos}
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
            <span>FIFO波形图</span>
            {selectedFifos.length > 0 && (
              <span style={{ fontSize: 12, color: '#999' }}>
                ({selectedFifos.length} 个FIFO)
              </span>
            )}
          </Space>
        }
      >
        {error ? (
          <Alert
            type="error"
            message="加载失败"
            description={error}
            showIcon
            style={{ marginBottom: 16 }}
          />
        ) : null}

        {loading ? (
          <div style={{ textAlign: 'center', padding: 50 }}>
            <Spin tip="加载FIFO波形数据..." />
          </div>
        ) : !waveformData || waveformData.signals.length === 0 ? (
          <Empty
            description={
              selectedFifos.length === 0
                ? '请选择要查看的FIFO'
                : '所选FIFO无数据'
            }
          />
        ) : (
          <FIFOWaveformChart
            signals={waveformData.signals}
            timeRange={waveformData.time_range}
          />
        )}
      </Card>
    </div>
  );
}
