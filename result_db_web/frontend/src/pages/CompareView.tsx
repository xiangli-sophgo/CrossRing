/**
 * 实验对比视图
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  Table,
  Button,
  Space,
  Spin,
  message,
  Typography,
  Row,
  Col,
  Empty,
} from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { useExperimentStore } from '../stores/experimentStore';
import { compareExperiments, getDistribution } from '../api';
import type { Experiment, PerformanceDistribution } from '../types';

const { Title } = Typography;

export default function CompareView() {
  const navigate = useNavigate();
  const { selectedExperimentIds, clearSelection } = useExperimentStore();

  const [loading, setLoading] = useState(true);
  const [compareData, setCompareData] = useState<{
    experiments: Experiment[];
    best_configs: Record<string, unknown>[];
  } | null>(null);
  const [distributions, setDistributions] = useState<
    Record<number, PerformanceDistribution>
  >({});

  // 加载对比数据
  const loadCompareData = async () => {
    if (selectedExperimentIds.length < 2) {
      message.warning('请至少选择2个实验进行对比');
      navigate('/');
      return;
    }

    setLoading(true);
    try {
      const data = await compareExperiments(selectedExperimentIds);
      setCompareData(data);

      // 加载各实验的性能分布
      const distResults: Record<number, PerformanceDistribution> = {};
      for (const expId of selectedExperimentIds) {
        try {
          const dist = await getDistribution(expId, 30);
          distResults[expId] = dist;
        } catch {
          // 忽略单个实验的加载失败
        }
      }
      setDistributions(distResults);
    } catch (error) {
      message.error('加载对比数据失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadCompareData();
  }, [selectedExperimentIds]);

  // 性能分布对比图配置
  const getDistributionChartOption = () => {
    if (!compareData) return {};

    const series = compareData.experiments.map((exp) => {
      const dist = distributions[exp.id];
      if (!dist || !dist.histogram) return null;

      return {
        name: exp.name,
        type: 'line',
        smooth: true,
        data: dist.histogram.map((h) => [(h[0] + h[1]) / 2, h[2]]),
      };
    }).filter(Boolean);

    return {
      tooltip: {
        trigger: 'axis',
      },
      legend: {
        data: compareData.experiments.map((e) => e.name),
      },
      xAxis: {
        type: 'value',
        name: '性能 (GB/s)',
      },
      yAxis: {
        type: 'value',
        name: '频次',
      },
      series,
    };
  };

  // 最佳性能对比图配置
  const getBestPerformanceChartOption = () => {
    if (!compareData) return {};

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      },
      xAxis: {
        type: 'category',
        data: compareData.experiments.map((e) => e.name),
        axisLabel: {
          rotate: 30,
        },
      },
      yAxis: {
        type: 'value',
        name: '性能 (GB/s)',
      },
      series: [
        {
          type: 'bar',
          data: compareData.experiments.map((e) => e.best_performance || 0),
          itemStyle: {
            color: '#1890ff',
          },
          label: {
            show: true,
            position: 'top',
            formatter: '{c} GB/s',
          },
        },
      ],
    };
  };

  // 最佳配置对比表格列
  const configColumns = [
    {
      title: '参数',
      dataIndex: 'param',
      key: 'param',
      fixed: 'left' as const,
      width: 200,
    },
    ...(compareData?.experiments.map((exp) => ({
      title: exp.name,
      dataIndex: `exp_${exp.id}`,
      key: `exp_${exp.id}`,
      width: 120,
    })) || []),
  ];

  // 从最佳配置中提取所有参数键
  const allParamKeys = compareData?.best_configs
    ? [...new Set(
        compareData.best_configs.flatMap((config) => {
          const params = config.config_params as Record<string, unknown> | undefined;
          return params ? Object.keys(params) : [];
        })
      )]
    : [];

  // 构建最佳配置对比数据
  const configTableData = allParamKeys.map((param) => {
    const row: Record<string, unknown> = { param, key: param };
    compareData?.best_configs.forEach((config, idx) => {
      const expId = compareData.experiments[idx].id;
      const params = config.config_params as Record<string, unknown> | undefined;
      row[`exp_${expId}`] = params?.[param] ?? '-';
    });
    return row;
  });

  // 添加性能行
  if (compareData) {
    configTableData.unshift({
      key: 'performance',
      param: '最佳性能 (GB/s)',
      ...Object.fromEntries(
        compareData.best_configs.map((config, idx) => [
          `exp_${compareData.experiments[idx].id}`,
          typeof config.performance === 'number'
            ? (config.performance as number).toFixed(2)
            : '-',
        ])
      ),
    });
  }

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
      </div>
    );
  }

  if (!compareData || compareData.experiments.length < 2) {
    return (
      <Card>
        <Empty description="请选择至少2个实验进行对比">
          <Button type="primary" onClick={() => navigate('/')}>
            返回列表
          </Button>
        </Empty>
      </Card>
    );
  }

  return (
    <div>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
          <Space>
            <Button
              icon={<ArrowLeftOutlined />}
              onClick={() => {
                clearSelection();
                navigate('/');
              }}
            >
              返回
            </Button>
            <Title level={4} style={{ margin: 0 }}>
              实验对比 ({compareData.experiments.length} 个实验)
            </Title>
          </Space>
        </div>

        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Card title="最佳性能对比" size="small">
              <ReactECharts option={getBestPerformanceChartOption()} style={{ height: 300 }} />
            </Card>
          </Col>
          <Col span={12}>
            <Card title="性能分布对比" size="small">
              <ReactECharts option={getDistributionChartOption()} style={{ height: 300 }} />
            </Card>
          </Col>
          <Col span={24}>
            <Card title="最佳配置对比" size="small">
              <Table
                columns={configColumns}
                dataSource={configTableData}
                pagination={false}
                size="small"
                scroll={{ x: 'max-content' }}
              />
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  );
}
