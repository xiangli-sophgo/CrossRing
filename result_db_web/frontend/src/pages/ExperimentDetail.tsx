/**
 * 实验详情页
 */

import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Card,
  Tabs,
  Descriptions,
  Tag,
  Button,
  Space,
  Spin,
  message,
  Typography,
  Row,
  Col,
} from 'antd';
import { ArrowLeftOutlined, ReloadOutlined } from '@ant-design/icons';
import { useExperimentStore } from '../stores/experimentStore';
import { getExperiment, getStatistics, getResults, getParamKeys, getTrafficStats, type TrafficStat } from '../api';
import ResultTable from '../components/ResultTable';
import type { ResultsPageResponse, ExperimentType } from '../types';
import { Table } from 'antd';

const experimentTypeColors: Record<ExperimentType, string> = {
  kcin: 'blue',
  dcin: 'green',
};

const experimentTypeText: Record<ExperimentType, string> = {
  kcin: 'KCIN',
  dcin: 'DCIN',
};

const { Title } = Typography;

const statusColors: Record<string, string> = {
  running: 'processing',
  completed: 'success',
  failed: 'error',
  interrupted: 'warning',
};

export default function ExperimentDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const {
    currentExperiment,
    setCurrentExperiment,
    setCurrentStatistics,
    filters,
  } = useExperimentStore();

  const [loading, setLoading] = useState(true);
  const [resultsData, setResultsData] = useState<ResultsPageResponse | null>(null);
  const [resultsLoading, setResultsLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(100);
  const [sortBy, setSortBy] = useState('id');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  const [paramKeys, setParamKeys] = useState<string[]>([]);
  const [trafficStats, setTrafficStats] = useState<TrafficStat[]>([]);
  const [activeTab, setActiveTab] = useState('results');

  const experimentId = parseInt(id || '0', 10);

  // 加载实验详情
  const loadExperiment = async () => {
    if (!experimentId) return;
    setLoading(true);
    try {
      const [exp, stats, keysData, trafficData] = await Promise.all([
        getExperiment(experimentId),
        getStatistics(experimentId),
        getParamKeys(experimentId),
        getTrafficStats(experimentId),
      ]);
      setCurrentExperiment(exp);
      setCurrentStatistics(stats);
      setParamKeys(keysData.param_keys);
      setTrafficStats(trafficData.traffic_stats);
    } catch (error) {
      message.error('加载实验详情失败');
    } finally {
      setLoading(false);
    }
  };

  // 加载结果数据
  const loadResults = async () => {
    if (!experimentId) return;
    setResultsLoading(true);
    try {
      const data = await getResults(
        experimentId,
        page,
        pageSize,
        sortBy,
        sortOrder,
        Object.keys(filters).length > 0 ? filters : undefined
      );
      setResultsData(data);
    } catch (error) {
      message.error('加载结果数据失败');
    } finally {
      setResultsLoading(false);
    }
  };

  useEffect(() => {
    loadExperiment();
  }, [experimentId]);

  useEffect(() => {
    loadResults();
  }, [experimentId, page, pageSize, sortBy, sortOrder, filters]);

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
      </div>
    );
  }

  if (!currentExperiment) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: 50 }}>
          <Typography.Text type="secondary">实验不存在</Typography.Text>
          <br />
          <Button type="link" onClick={() => navigate('/')}>
            返回列表
          </Button>
        </div>
      </Card>
    );
  }

  const tabItems = [
    {
      key: 'overview',
      label: '概览',
      children: (
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Card title="实验信息">
              <Descriptions column={2}>
                <Descriptions.Item label="实验名称">
                  {currentExperiment.name}
                </Descriptions.Item>
                <Descriptions.Item label="类型">
                  <Tag color={experimentTypeColors[currentExperiment.experiment_type]}>
                    {experimentTypeText[currentExperiment.experiment_type]}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="状态">
                  <Tag color={statusColors[currentExperiment.status || '']}>
                    {currentExperiment.status}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="拓扑类型">
                  {currentExperiment.topo_type || '-'}
                </Descriptions.Item>
                <Descriptions.Item label="创建时间">
                  {currentExperiment.created_at
                    ? new Date(currentExperiment.created_at).toLocaleString('zh-CN')
                    : '-'}
                </Descriptions.Item>
                <Descriptions.Item label="数据流文件">
                  {trafficStats.length > 0
                    ? trafficStats.map(s => s.traffic_name).join(', ')
                    : '-'}
                </Descriptions.Item>
                <Descriptions.Item label="描述" span={2}>
                  {currentExperiment.description || '-'}
                </Descriptions.Item>
              </Descriptions>
            </Card>
          </Col>
          <Col span={24}>
            <Card title="数据流统计">
              <Table
                dataSource={trafficStats}
                rowKey="traffic_name"
                pagination={false}
                size="small"
                columns={[
                  {
                    title: '数据流文件',
                    dataIndex: 'traffic_name',
                    key: 'traffic_name',
                    align: 'center' as const,
                  },
                  {
                    title: '结果数',
                    dataIndex: 'count',
                    key: 'count',
                    width: 200,
                    align: 'center' as const,
                  },
                  {
                    title: '平均性能 (GB/s)',
                    dataIndex: 'avg_performance',
                    key: 'avg_performance',
                    width: 250,
                    align: 'center' as const,
                    render: (val: number) => val.toFixed(2),
                  },
                  {
                    title: '最大性能 (GB/s)',
                    dataIndex: 'max_performance',
                    key: 'max_performance',
                    width: 250,
                    align: 'center' as const,
                    render: (val: number) => val.toFixed(2),
                  },
                  {
                    title: '最小性能 (GB/s)',
                    dataIndex: 'min_performance',
                    key: 'min_performance',
                    width: 250,
                    align: 'center' as const,
                    render: (val: number) => val.toFixed(2),
                  },
                ]}
              />
            </Card>
          </Col>
        </Row>
      ),
    },
    {
      key: 'results',
      label: '结果数据',
      children: (
        <Row gutter={16}>
          <Col span={24}>
            <Card title={`结果列表 (共 ${resultsData?.total || 0} 条)`}>
              <ResultTable
                data={resultsData}
                loading={resultsLoading}
                page={page}
                pageSize={pageSize}
                paramKeys={paramKeys}
                experimentId={experimentId}
                experimentType={currentExperiment?.experiment_type}
                onPageChange={(p, ps) => {
                  setPage(p);
                  setPageSize(ps);
                }}
                onSortChange={(field, order) => {
                  setSortBy(field);
                  setSortOrder(order);
                }}
                onDataChange={loadResults}
              />
            </Card>
          </Col>
        </Row>
      ),
    },
  ];

  return (
    <div>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
          <Space>
            <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/')}>
              返回
            </Button>
            <Title level={4} style={{ margin: 0 }}>
              {currentExperiment.name}
            </Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={loadExperiment}>
            刷新
          </Button>
        </div>

        <Tabs items={tabItems} activeKey={activeTab} onChange={setActiveTab} />
      </Card>
    </div>
  );
}
