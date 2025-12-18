/**
 * 参数分析视图
 * 展示参数与性能的关系图表
 */

import { useState, useEffect, useMemo, useCallback } from 'react';
import { Card, Radio, TreeSelect, Row, Col, Spin, Empty, Typography } from 'antd';
import { LineChartOutlined, HeatMapOutlined, BarChartOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { getParameterSensitivity, getAllSensitivity, getParameterHeatmap } from '../api';
import { classifyParamKeysWithHierarchy, type HierarchicalClassification } from '../utils/paramClassifier';
import type { SensitivityDataPoint } from '../types';
import { primaryColor } from '@/theme/colors';

const { Text } = Typography;

interface ParameterAnalysisViewProps {
  experimentId: number;
  paramKeys: string[];
}

type ChartType = 'line' | 'heatmap' | 'sensitivity';

// TreeSelect 节点类型
interface TreeNode {
  title: string;
  value: string;
  key: string;
  selectable?: boolean;
  children?: TreeNode[];
}

// 将分类结果转换为 TreeSelect 的 treeData 格式
function buildTreeData(classification: HierarchicalClassification): TreeNode[] {
  const treeData: TreeNode[] = [];

  // 配置参数
  if (classification.config.length > 0) {
    treeData.push({
      title: `配置参数 (${classification.config.length})`,
      value: 'group_config',
      key: 'group_config',
      selectable: false,
      children: classification.config.map(key => ({
        title: key,
        value: key,
        key: key,
      })),
    });
  }

  // 重要统计
  if (classification.important.length > 0) {
    treeData.push({
      title: `重要统计 (${classification.important.length})`,
      value: 'group_important',
      key: 'group_important',
      selectable: false,
      children: classification.important.map(key => ({
        title: key,
        value: key,
        key: key,
      })),
    });
  }

  // Die 分组
  for (let i = 0; i <= 3; i++) {
    const dieKey = `die${i}` as keyof HierarchicalClassification;
    const dieData = classification[dieKey];
    if (dieData && typeof dieData === 'object' && 'config' in dieData) {
      const allKeys = [...dieData.config, ...dieData.important, ...dieData.result];
      if (allKeys.length > 0) {
        const dieChildren: TreeNode[] = [];

        if (dieData.config.length > 0) {
          dieChildren.push({
            title: `配置 (${dieData.config.length})`,
            value: `group_die${i}_config`,
            key: `group_die${i}_config`,
            selectable: false,
            children: dieData.config.map(key => ({ title: key, value: key, key })),
          });
        }
        if (dieData.important.length > 0) {
          dieChildren.push({
            title: `重要 (${dieData.important.length})`,
            value: `group_die${i}_important`,
            key: `group_die${i}_important`,
            selectable: false,
            children: dieData.important.map(key => ({ title: key, value: key, key })),
          });
        }
        if (dieData.result.length > 0) {
          dieChildren.push({
            title: `统计 (${dieData.result.length})`,
            value: `group_die${i}_result`,
            key: `group_die${i}_result`,
            selectable: false,
            children: dieData.result.map(key => ({ title: key, value: key, key })),
          });
        }

        treeData.push({
          title: `Die${i} (${allKeys.length})`,
          value: `group_die${i}`,
          key: `group_die${i}`,
          selectable: false,
          children: dieChildren,
        });
      }
    }
  }

  // 结果统计
  if (classification.result.length > 0) {
    treeData.push({
      title: `结果统计 (${classification.result.length})`,
      value: 'group_result',
      key: 'group_result',
      selectable: false,
      children: classification.result.map(key => ({
        title: key,
        value: key,
        key: key,
      })),
    });
  }

  return treeData;
}

// 单参数折线图
function SingleParamLineChart({
  experimentId,
  param,
  metric,
  metricName,
}: {
  experimentId: number;
  param: string;
  metric?: string;
  metricName: string;
}) {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<SensitivityDataPoint[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await getParameterSensitivity(experimentId, param, metric);
        setData(response.data);
      } catch (error) {
        console.error('获取敏感性数据失败:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [experimentId, param, metric]);

  if (loading) {
    return <div style={{ textAlign: 'center', padding: 40 }}><Spin /></div>;
  }

  if (data.length === 0) {
    return <Empty description="暂无数据" />;
  }

  const option = {
    tooltip: {
      trigger: 'axis',
      formatter: (params: { name: string; value: number; seriesName: string }[]) => {
        const p = params[0];
        const dataPoint = data.find(d => String(d.value) === p.name);
        if (!dataPoint) return '';
        return `
          <b>${param} = ${p.name}</b><br/>
          均值: ${dataPoint.mean_performance.toFixed(2)}<br/>
          最大: ${dataPoint.max_performance.toFixed(2)}<br/>
          最小: ${dataPoint.min_performance.toFixed(2)}<br/>
          样本数: ${dataPoint.count}
        `;
      },
    },
    xAxis: {
      type: 'category',
      data: data.map(d => String(d.value)),
      name: param,
      nameLocation: 'middle',
      nameGap: 30,
    },
    yAxis: {
      type: 'value',
      name: metricName,
    },
    series: [
      {
        name: '最大值',
        type: 'line',
        data: data.map(d => d.max_performance),
        lineStyle: { opacity: 0 },
        areaStyle: { opacity: 0 },
        stack: 'range',
        symbol: 'none',
      },
      {
        name: '范围',
        type: 'line',
        data: data.map(d => d.max_performance - d.min_performance),
        lineStyle: { opacity: 0 },
        areaStyle: { opacity: 0.2, color: primaryColor },
        stack: 'range',
        symbol: 'none',
      },
      {
        name: '均值',
        type: 'line',
        data: data.map(d => d.mean_performance),
        lineStyle: { color: primaryColor, width: 2 },
        itemStyle: { color: primaryColor },
        symbol: 'circle',
        symbolSize: 6,
      },
    ],
    grid: {
      left: 60,
      right: 20,
      top: 40,
      bottom: 50,
    },
  };

  return <ReactECharts option={option} style={{ height: 300 }} />;
}

// 双参数热力图
function DualParamHeatmap({
  experimentId,
  paramX,
  paramY,
  metric,
  metricName,
}: {
  experimentId: number;
  paramX: string;
  paramY: string;
  metric?: string;
  metricName: string;
}) {
  const [loading, setLoading] = useState(true);
  const [heatmapData, setHeatmapData] = useState<{
    x_values: number[];
    y_values: number[];
    data: { [key: string]: number | string }[];
  } | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await getParameterHeatmap(experimentId, paramX, paramY, metric);
        setHeatmapData(response);
      } catch (error) {
        console.error('获取热力图数据失败:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [experimentId, paramX, paramY, metric]);

  if (loading) {
    return <div style={{ textAlign: 'center', padding: 40 }}><Spin /></div>;
  }

  if (!heatmapData || heatmapData.data.length === 0) {
    return <Empty description="暂无数据" />;
  }

  // 转换为 ECharts 热力图数据格式 [x_index, y_index, value]
  const chartData = heatmapData.data.map(item => {
    const xVal = item[paramX];
    const yVal = item[paramY];
    const perf = item['mean_performance'] as number;
    const xIndex = heatmapData.x_values.indexOf(xVal as number);
    const yIndex = heatmapData.y_values.indexOf(yVal as number);
    return [xIndex, yIndex, perf];
  }).filter(item => item[0] >= 0 && item[1] >= 0);

  const values = chartData.map(d => d[2]);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);

  const option = {
    tooltip: {
      position: 'top',
      formatter: (params: { data: number[] }) => {
        const [xIdx, yIdx, val] = params.data;
        return `${paramX}: ${heatmapData.x_values[xIdx]}<br/>${paramY}: ${heatmapData.y_values[yIdx]}<br/>${metricName}: ${val.toFixed(2)}`;
      },
    },
    xAxis: {
      type: 'category',
      data: heatmapData.x_values.map(String),
      name: paramX,
      nameLocation: 'middle',
      nameGap: 30,
      splitArea: { show: true },
    },
    yAxis: {
      type: 'category',
      data: heatmapData.y_values.map(String),
      name: paramY,
      splitArea: { show: true },
    },
    visualMap: {
      min: minVal,
      max: maxVal,
      calculable: true,
      orient: 'vertical',
      right: 10,
      top: 'center',
      inRange: {
        color: ['#f5f5f5', '#91d5ff', '#1890ff', '#ff7875', '#ff4d4f'],
      },
    },
    series: [{
      type: 'heatmap',
      data: chartData,
      label: {
        show: chartData.length <= 100,
        formatter: (params: { data: number[] }) => params.data[2].toFixed(1),
        fontSize: 10,
      },
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowColor: 'rgba(0, 0, 0, 0.5)',
        },
      },
    }],
    grid: {
      left: 80,
      right: 100,
      top: 20,
      bottom: 50,
    },
  };

  return <ReactECharts option={option} style={{ height: 350 }} />;
}

// 敏感度排序图
function SensitivityRankingChart({
  experimentId,
  metric,
  metricName,
}: {
  experimentId: number;
  metric?: string;
  metricName: string;
}) {
  const [loading, setLoading] = useState(true);
  const [sensitivityData, setSensitivityData] = useState<{ param: string; sensitivity: number }[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await getAllSensitivity(experimentId, metric);
        const params = response.parameters;

        // 计算每个参数的敏感度（性能变化范围）
        const sensitivities = Object.entries(params)
          .map(([param, data]) => {
            const dataPoints = data.data;
            if (dataPoints.length === 0) return null;
            const maxPerf = Math.max(...dataPoints.map(d => d.max_performance));
            const minPerf = Math.min(...dataPoints.map(d => d.min_performance));
            return {
              param,
              sensitivity: maxPerf - minPerf,
            };
          })
          .filter((item): item is { param: string; sensitivity: number } => item !== null)
          .sort((a, b) => b.sensitivity - a.sensitivity)
          .slice(0, 20); // 只显示前20个

        setSensitivityData(sensitivities);
      } catch (error) {
        console.error('获取敏感性数据失败:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [experimentId, metric]);

  if (loading) {
    return <div style={{ textAlign: 'center', padding: 40 }}><Spin /></div>;
  }

  if (sensitivityData.length === 0) {
    return <Empty description="暂无数据" />;
  }

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params: { name: string; value: number }[]) => {
        const p = params[0];
        return `${p.name}<br/>敏感度: ${p.value.toFixed(2)}`;
      },
    },
    xAxis: {
      type: 'value',
      name: `敏感度 (${metricName})`,
    },
    yAxis: {
      type: 'category',
      data: sensitivityData.map(d => d.param).reverse(),
      axisLabel: {
        width: 150,
        overflow: 'truncate',
        ellipsis: '...',
      },
    },
    series: [{
      type: 'bar',
      data: sensitivityData.map(d => d.sensitivity).reverse(),
      itemStyle: {
        color: primaryColor,
      },
      barMaxWidth: 30,
    }],
    grid: {
      left: 160,
      right: 40,
      top: 20,
      bottom: 40,
    },
  };

  return <ReactECharts option={option} style={{ height: Math.max(400, sensitivityData.length * 25) }} />;
}

export default function ParameterAnalysisView({
  experimentId,
  paramKeys,
}: ParameterAnalysisViewProps) {
  const [chartType, setChartType] = useState<ChartType>('sensitivity');
  const [selectedParams, setSelectedParams] = useState<string[]>([]);
  const [selectedMetric, setSelectedMetric] = useState<string | undefined>('带宽_平均加权');

  // 构建树形数据
  const treeData = useMemo(() => {
    const classification = classifyParamKeysWithHierarchy(paramKeys);
    return buildTreeData(classification);
  }, [paramKeys]);

  // 指标显示名称
  const metricName = selectedMetric || 'performance';

  // 处理参数选择变化
  const handleParamChange = useCallback((values: string[]) => {
    // 过滤掉分组节点，只保留实际参数
    const actualParams = values.filter(v => !v.startsWith('group_'));
    setSelectedParams(actualParams);
  }, []);

  // 渲染图表内容
  const renderCharts = () => {
    if (chartType === 'sensitivity') {
      return (
        <Card title="参数敏感度排序" size="small">
          <Text type="secondary" style={{ display: 'block', marginBottom: 16 }}>
            敏感度 = 该参数下 {metricName} 的最大变化范围，值越大表示该参数对 {metricName} 影响越大
          </Text>
          <SensitivityRankingChart experimentId={experimentId} metric={selectedMetric} metricName={metricName} />
        </Card>
      );
    }

    if (selectedParams.length === 0) {
      return (
        <Empty
          description="请选择要分析的参数"
          style={{ padding: 60 }}
        />
      );
    }

    if (chartType === 'line') {
      // 单参数折线图：为每个选中的参数画一个图
      return (
        <Row gutter={[16, 16]}>
          {selectedParams.map(param => (
            <Col span={selectedParams.length > 2 ? 12 : 24} key={param}>
              <Card title={param} size="small">
                <SingleParamLineChart experimentId={experimentId} param={param} metric={selectedMetric} metricName={metricName} />
              </Card>
            </Col>
          ))}
        </Row>
      );
    }

    if (chartType === 'heatmap') {
      if (selectedParams.length < 2) {
        return (
          <Empty
            description="热力图需要选择至少2个参数"
            style={{ padding: 60 }}
          />
        );
      }

      // 双参数热力图：两两组合
      const pairs: [string, string][] = [];
      for (let i = 0; i < selectedParams.length; i++) {
        for (let j = i + 1; j < selectedParams.length; j++) {
          pairs.push([selectedParams[i], selectedParams[j]]);
        }
      }

      return (
        <Row gutter={[16, 16]}>
          {pairs.map(([paramX, paramY]) => (
            <Col span={pairs.length > 1 ? 12 : 24} key={`${paramX}-${paramY}`}>
              <Card title={`${paramX} × ${paramY}`} size="small">
                <DualParamHeatmap
                  experimentId={experimentId}
                  paramX={paramX}
                  paramY={paramY}
                  metric={selectedMetric}
                  metricName={metricName}
                />
              </Card>
            </Col>
          ))}
        </Row>
      );
    }

    return null;
  };

  return (
    <div>
      {/* 控制栏 */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col>
            <Text strong>图表类型：</Text>
          </Col>
          <Col>
            <Radio.Group
              value={chartType}
              onChange={e => setChartType(e.target.value)}
              optionType="button"
              buttonStyle="solid"
            >
              <Radio.Button value="sensitivity">
                <BarChartOutlined /> 敏感度排序
              </Radio.Button>
              <Radio.Button value="line">
                <LineChartOutlined /> 单参数曲线
              </Radio.Button>
              <Radio.Button value="heatmap">
                <HeatMapOutlined /> 双参数热力图
              </Radio.Button>
            </Radio.Group>
          </Col>
          <Col>
            <Text strong>性能指标：</Text>
          </Col>
          <Col>
            <TreeSelect
              treeData={treeData}
              value={selectedMetric}
              onChange={setSelectedMetric}
              showSearch
              allowClear
              placeholder="选择性能指标"
              style={{ width: 200 }}
              treeDefaultExpandAll={false}
              filterTreeNode={(input, node) =>
                (node.title as string)?.toLowerCase().includes(input.toLowerCase())
              }
            />
          </Col>
          {chartType !== 'sensitivity' && (
            <>
              <Col>
                <Text strong>选择参数：</Text>
              </Col>
              <Col flex="auto">
                <TreeSelect
                  treeData={treeData}
                  value={selectedParams}
                  onChange={handleParamChange}
                  treeCheckable
                  showSearch
                  placeholder="搜索并选择参数..."
                  style={{ width: '100%', maxWidth: 500 }}
                  maxTagCount={3}
                  maxTagPlaceholder={omittedValues => `+${omittedValues.length}...`}
                  treeDefaultExpandAll={false}
                  filterTreeNode={(input, node) =>
                    (node.title as string)?.toLowerCase().includes(input.toLowerCase())
                  }
                />
              </Col>
            </>
          )}
        </Row>
      </Card>

      {/* 图表区域 */}
      {renderCharts()}
    </div>
  );
}
