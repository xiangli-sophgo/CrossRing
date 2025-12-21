/**
 * 波形查看器主组件
 */

import { useState, useEffect, useCallback } from 'react';
import { Card, Spin, Empty, Button, Space, Alert, Collapse, Statistic, Row, Col } from 'antd';
import { ReloadOutlined, LineChartOutlined } from '@ant-design/icons';
import WaveformChart from './WaveformChart';
import PacketSelector from './PacketSelector';
import {
  getWaveformData,
  checkWaveformData,
  type WaveformResponse,
  type WaveformCheckResponse,
} from '@/api/waveform';

interface Props {
  experimentId: number;
  resultId: number;
}

export default function WaveformViewer({ experimentId, resultId }: Props) {
  const [loading, setLoading] = useState(true);
  const [checkResult, setCheckResult] = useState<WaveformCheckResponse | null>(null);
  const [waveformData, setWaveformData] = useState<WaveformResponse | null>(null);
  const [selectedPacketIds, setSelectedPacketIds] = useState<number[]>([]);
  const [loadingWaveform, setLoadingWaveform] = useState(false);

  // 检查波形数据可用性
  useEffect(() => {
    checkAvailability();
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

  return (
    <div>
      {/* 统计信息 */}
      {stats && (
        <Card size="small" style={{ marginBottom: 16 }}>
          <Row gutter={16}>
            <Col span={4}>
              <Statistic title="总请求数" value={stats.total_packets} />
            </Col>
            <Col span={4}>
              <Statistic title="读请求" value={stats.read_packets} />
            </Col>
            <Col span={4}>
              <Statistic title="写请求" value={stats.write_packets} />
            </Col>
            <Col span={4}>
              <Statistic title="总Flit数" value={stats.total_flits} />
            </Col>
            <Col span={8}>
              <Statistic
                title="时间范围 (ns)"
                value={`${stats.time_range_ns.start.toFixed(1)} - ${stats.time_range_ns.end.toFixed(1)}`}
              />
            </Col>
          </Row>
        </Card>
      )}

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
            <span>传输波形图</span>
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
        ) : !waveformData || waveformData.signals.length === 0 ? (
          <Empty description="请选择要查看的请求" />
        ) : (
          <WaveformChart
            signals={waveformData.signals}
            timeRange={waveformData.time_range}
            stages={waveformData.stages}
          />
        )}
      </Card>
    </div>
  );
}
