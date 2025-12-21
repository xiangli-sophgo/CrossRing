/**
 * 参数分析视图
 * 展示参数与性能的关系图表
 */

import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { Card, Radio, TreeSelect, Row, Col, Spin, Empty, Typography, Button, Modal, Input, message, Popconfirm, Space, Tooltip } from 'antd';
import { LineChartOutlined, HeatMapOutlined, BarChartOutlined, SaveOutlined, DeleteOutlined, DownloadOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { getParameterSensitivity, getParameterHeatmap, getAnalysisCharts, addAnalysisChart, deleteAnalysisChart, getParameterInfluence, type ChartConfig, type ParameterInfluence } from '../api';
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
  onExport,
}: {
  experimentId: number;
  param: string;
  metric?: string;
  metricName: string;
  onExport?: (chartInstance: ReactECharts) => void;
}) {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<SensitivityDataPoint[]>([]);
  const chartRef = useRef<ReactECharts>(null);

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
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderColor: '#e8e8e8',
      borderWidth: 1,
      textStyle: { color: '#333' },
      formatter: (params: { name: string; value: number; seriesName: string }[]) => {
        const p = params[0];
        const dataPoint = data.find(d => String(d.value) === p.name);
        if (!dataPoint) return '';
        return `
          <div style="font-weight:600;margin-bottom:4px">${param} = ${p.name}</div>
          <div>均值: <span style="color:${primaryColor};font-weight:500">${dataPoint.mean_performance.toFixed(2)}</span></div>
          <div>最大: ${dataPoint.max_performance.toFixed(2)}</div>
          <div>最小: ${dataPoint.min_performance.toFixed(2)}</div>
          <div style="color:#999;font-size:11px">样本数: ${dataPoint.count}</div>
        `;
      },
    },
    xAxis: {
      type: 'category',
      data: data.map(d => String(d.value)),
      name: param,
      nameLocation: 'middle',
      nameGap: 25,
      axisLine: { lineStyle: { color: '#d9d9d9' } },
      axisTick: { lineStyle: { color: '#d9d9d9' } },
      axisLabel: { color: '#666', fontSize: 11 },
      nameTextStyle: { color: '#333', fontSize: 12 },
    },
    yAxis: {
      type: 'value',
      name: metricName,
      scale: true,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: '#666', fontSize: 11 },
      nameTextStyle: { color: '#333', fontSize: 12 },
      splitLine: { lineStyle: { color: '#f0f0f0' } },
    },
    series: [
      {
        name: '最小值',
        type: 'line',
        data: data.map(d => d.min_performance),
        lineStyle: { opacity: 0 },
        areaStyle: { opacity: 0 },
        stack: 'range',
        symbol: 'none',
        smooth: true,
      },
      {
        name: '范围',
        type: 'line',
        data: data.map(d => d.max_performance - d.min_performance),
        lineStyle: { opacity: 0 },
        areaStyle: {
          opacity: 0.15,
          color: primaryColor,
        },
        stack: 'range',
        symbol: 'none',
        smooth: true,
      },
      {
        name: '均值',
        type: 'line',
        data: data.map(d => d.mean_performance),
        lineStyle: { color: primaryColor, width: 2 },
        itemStyle: { color: primaryColor },
        symbol: 'circle',
        symbolSize: 5,
        smooth: true,
      },
    ],
    grid: {
      left: 50,
      right: 15,
      top: 30,
      bottom: 40,
    },
  };

  const handleExport = () => {
    if (chartRef.current && onExport) {
      onExport(chartRef.current);
    }
  };

  return (
    <div>
      {onExport && (
        <div style={{ textAlign: 'right', marginBottom: 8 }}>
          <Button size="small" icon={<DownloadOutlined />} onClick={handleExport}>
            导出图片
          </Button>
        </div>
      )}
      <ReactECharts ref={chartRef} option={option} style={{ height: 400 }} />
    </div>
  );
}

// 双参数热力图
function DualParamHeatmap({
  experimentId,
  paramX,
  paramY,
  metric,
  metricName,
  onExport,
}: {
  experimentId: number;
  paramX: string;
  paramY: string;
  metric?: string;
  metricName: string;
  onExport?: (chartInstance: ReactECharts) => void;
}) {
  const [loading, setLoading] = useState(true);
  const [heatmapData, setHeatmapData] = useState<{
    x_values: number[];
    y_values: number[];
    data: { [key: string]: number | string }[];
  } | null>(null);
  const [baseValue, setBaseValue] = useState<number | null>(null);
  const chartRef = useRef<ReactECharts>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await getParameterHeatmap(experimentId, paramX, paramY, metric);
        setHeatmapData(response);
        setBaseValue(null); // 重置基准值
      } catch (error) {
        console.error('获取热力图数据失败:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [experimentId, paramX, paramY, metric]);

  // 图表绑定 visualMap 事件（必须在 early return 之前定义）
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const onChartReady = useCallback((chart: any) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleVisualMapChange = (params: any) => {
      console.log('visualMap event:', params);
      // visualMap 连续型的事件参数格式: { selected: [min, max] }
      if (params.selected !== undefined) {
        const [, selectedMax] = params.selected;
        if (typeof selectedMax === 'number') {
          setBaseValue(selectedMax);
        }
      }
    };

    // 先移除可能存在的旧监听器，再添加新的
    chart.off('datarangeselected');
    chart.on('datarangeselected', handleVisualMapChange);
  }, []);

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

  // 使用用户选择的基准值，如果没有则使用数据最大值
  const effectiveBaseValue = baseValue !== null ? baseValue : maxVal;

  // 计算单元格数量
  const xCount = heatmapData.x_values.length;
  const yCount = heatmapData.y_values.length;

  // 计算正方形单元格的尺寸和图表高度
  // 设置每个单元格大小为 60px，保证正方形
  const cellSize = 60;
  const gridLeft = 100;
  const gridRight = 120;
  const gridTop = 30;
  const gridBottom = 60;

  // 根据单元格数量计算图表宽度和高度
  const chartWidth = xCount * cellSize + gridLeft + gridRight;
  const chartHeight = yCount * cellSize + gridTop + gridBottom;

  // 根据数值范围决定小数位数
  const range = maxVal - minVal;
  const decimalPlaces = range > 100 ? 0 : range > 10 ? 1 : 2;

  // 根据单元格数量调整字体大小
  const fontSize = Math.max(8, Math.min(12, cellSize / 5));

  // 计算下降比例的格式化函数
  const formatDropPercent = (val: number) => {
    if (effectiveBaseValue === 0) return '0%';
    const dropPercent = ((effectiveBaseValue - val) / effectiveBaseValue) * 100;
    if (dropPercent <= 0) return '';  // 等于或超过基准值不显示下降比例
    return `-${dropPercent.toFixed(1)}%`;
  };

  const option = {
    tooltip: {
      position: 'top',
      formatter: (params: { data: number[] }) => {
        const [xIdx, yIdx, val] = params.data;
        const dropPercent = effectiveBaseValue > 0 ? ((effectiveBaseValue - val) / effectiveBaseValue) * 100 : 0;
        return `${paramX}: ${heatmapData.x_values[xIdx]}<br/>${paramY}: ${heatmapData.y_values[yIdx]}<br/>${metricName}: ${val.toFixed(2)}<br/>相对基准下降: ${dropPercent > 0 ? dropPercent.toFixed(1) : 0}%`;
      },
    },
    xAxis: {
      type: 'category',
      data: heatmapData.x_values.map(String),
      name: paramX,
      nameLocation: 'middle',
      nameGap: 40,
      splitArea: { show: true },
      axisLabel: {
        rotate: xCount > 8 ? 45 : 0,
        fontSize: 11,
      },
    },
    yAxis: {
      type: 'category',
      data: heatmapData.y_values.map(String),
      name: paramY,
      splitArea: { show: true },
      axisLabel: {
        fontSize: 11,
      },
    },
    visualMap: {
      type: 'continuous',
      min: minVal,
      max: maxVal,
      calculable: true,
      orient: 'vertical',
      right: 10,
      top: 'center',
      realtime: false,  // 拖动结束后触发事件
      inRange: {
        color: ['#f5f5f5', '#91d5ff', '#1890ff', '#ff7875', '#ff4d4f'],
      },
    },
    series: [{
      type: 'heatmap',
      data: chartData,
      label: {
        show: true,
        formatter: (params: { data: number[] }) => {
          const val = params.data[2];
          const valueStr = val.toFixed(decimalPlaces);
          const dropStr = formatDropPercent(val);
          return dropStr ? `${valueStr}\n{drop|${dropStr}}` : valueStr;
        },
        fontSize: fontSize,
        color: '#333',
        align: 'center',
        verticalAlign: 'middle',
        rich: {
          drop: {
            fontSize: fontSize - 1,
            color: '#666',
            lineHeight: fontSize + 2,
            align: 'center',
          },
        },
      },
      itemStyle: {
        borderColor: '#fff',
        borderWidth: 1,
      },
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowColor: 'rgba(0, 0, 0, 0.5)',
        },
      },
    }],
    grid: {
      left: gridLeft,
      right: gridRight,
      top: gridTop,
      bottom: gridBottom,
    },
  };

  const handleExport = () => {
    if (chartRef.current && onExport) {
      onExport(chartRef.current);
    }
  };

  return (
    <div>
      <div style={{ marginBottom: 8, fontSize: 12, color: '#666', display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
        <span>基准值: <span style={{ fontWeight: 'bold', color: '#1890ff' }}>{effectiveBaseValue.toFixed(decimalPlaces)}</span></span>
        {baseValue !== null && (
          <>
            <span style={{ color: '#faad14' }}>（已调整，原最高值: {maxVal.toFixed(decimalPlaces)}）</span>
            <Button size="small" type="link" onClick={() => setBaseValue(null)}>重置</Button>
          </>
        )}
        <span style={{ color: '#999' }}>| 拖动右侧色条可调整基准值</span>
        {onExport && (
          <Button size="small" icon={<DownloadOutlined />} onClick={handleExport} style={{ marginLeft: 'auto' }}>
            导出图片
          </Button>
        )}
      </div>
      <div style={{ overflowX: 'auto', overflowY: 'hidden', display: 'flex', justifyContent: 'center' }}>
        <ReactECharts
          ref={chartRef}
          option={option}
          onChartReady={onChartReady}
          style={{
            width: chartWidth,
            height: chartHeight,
            flexShrink: 0,
          }}
        />
      </div>
    </div>
  );
}

// 参数影响度排序图
function InfluenceRankingChart({
  experimentId,
  metric,
  onExport,
}: {
  experimentId: number;
  metric?: string;
  onExport?: (chartInstance: ReactECharts) => void;
}) {
  const [loading, setLoading] = useState(true);
  const [influenceData, setInfluenceData] = useState<ParameterInfluence[]>([]);
  const chartRef = useRef<ReactECharts>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await getParameterInfluence(experimentId, metric);
        setInfluenceData(response.parameters.slice(0, 20)); // 只显示前20个
      } catch (error) {
        console.error('获取影响度数据失败:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [experimentId, metric]);

  if (loading) {
    return <div style={{ textAlign: 'center', padding: 40 }}><Spin /></div>;
  }

  if (influenceData.length === 0) {
    return <Empty description="暂无数据" />;
  }

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderColor: '#e8e8e8',
      borderWidth: 1,
      textStyle: { color: '#333' },
      formatter: (params: { name: string; value: number; dataIndex: number }[]) => {
        const p = params[0];
        const data = influenceData[influenceData.length - 1 - p.dataIndex];
        return `
          <div style="font-weight:600;margin-bottom:4px">${p.name}</div>
          <div>影响度: <span style="color:${primaryColor};font-weight:500">${(data.influence * 100).toFixed(1)}%</span></div>
          <div>平均性能范围: ${data.mean_range.toFixed(2)}</div>
          <div style="color:#999;font-size:11px">取值数量: ${data.value_count}</div>
        `;
      },
    },
    xAxis: {
      type: 'value',
      name: '影响度',
      max: 1,
      axisLabel: {
        formatter: (value: number) => `${(value * 100).toFixed(0)}%`,
      },
      axisLine: { lineStyle: { color: '#d9d9d9' } },
      splitLine: { lineStyle: { color: '#f0f0f0' } },
    },
    yAxis: {
      type: 'category',
      data: influenceData.map(d => d.name).reverse(),
      axisLabel: {
        width: 150,
        overflow: 'truncate',
        ellipsis: '...',
        color: '#333',
      },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    series: [{
      type: 'bar',
      data: influenceData.map(d => d.influence).reverse(),
      itemStyle: {
        color: primaryColor,
      },
      barMaxWidth: 24,
      label: {
        show: true,
        position: 'right',
        formatter: (params: { value: number }) => `${(params.value * 100).toFixed(1)}%`,
        fontSize: 11,
        color: '#666',
      },
    }],
    grid: {
      left: 160,
      right: 60,
      top: 10,
      bottom: 30,
    },
  };

  const handleExport = () => {
    if (chartRef.current && onExport) {
      onExport(chartRef.current);
    }
  };

  return (
    <div>
      {onExport && (
        <div style={{ textAlign: 'right', marginBottom: 8 }}>
          <Button size="small" icon={<DownloadOutlined />} onClick={handleExport}>
            导出图片
          </Button>
        </div>
      )}
      <ReactECharts ref={chartRef} option={option} style={{ height: Math.max(300, influenceData.length * 28) }} />
    </div>
  );
}

export default function ParameterAnalysisView({
  experimentId,
  paramKeys,
}: ParameterAnalysisViewProps) {
  const [chartType, setChartType] = useState<ChartType>('sensitivity');
  const [selectedParams, setSelectedParams] = useState<string[]>([]);
  const [selectedMetric, setSelectedMetric] = useState<string | undefined>('带宽_平均加权');

  // 是否已开始分析（只有开始分析后才渲染图表）
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // 保存的图表列表
  const [savedCharts, setSavedCharts] = useState<ChartConfig[]>([]);
  const [loadingCharts, setLoadingCharts] = useState(false);

  // 保存图表对话框
  const [saveModalVisible, setSaveModalVisible] = useState(false);
  const [savingChart, setSavingChart] = useState(false);
  const [chartName, setChartName] = useState('');

  // 加载保存的图表
  const loadSavedCharts = useCallback(async () => {
    setLoadingCharts(true);
    try {
      const charts = await getAnalysisCharts(experimentId);
      setSavedCharts(charts);
    } catch (error) {
      console.error('加载保存的图表失败:', error);
    } finally {
      setLoadingCharts(false);
    }
  }, [experimentId]);

  useEffect(() => {
    loadSavedCharts();
  }, [loadSavedCharts]);

  // 保存当前图表配置
  const handleSaveChart = async () => {
    if (!chartName.trim()) {
      message.warning('请输入图表名称');
      return;
    }

    setSavingChart(true);
    try {
      await addAnalysisChart(experimentId, {
        name: chartName.trim(),
        chart_type: chartType,
        config: {
          params: selectedParams,
          metric: selectedMetric,
        },
      });
      message.success('图表配置已保存');
      setSaveModalVisible(false);
      setChartName('');
      loadSavedCharts();
    } catch (error) {
      console.error('保存图表失败:', error);
      message.error('保存图表失败');
    } finally {
      setSavingChart(false);
    }
  };

  // 删除保存的图表
  const handleDeleteChart = async (chartId: number) => {
    try {
      await deleteAnalysisChart(experimentId, chartId);
      message.success('已删除');
      loadSavedCharts();
    } catch (error) {
      console.error('删除图表失败:', error);
      message.error('删除失败');
    }
  };

  // 加载保存的图表配置
  const handleLoadChart = (chart: ChartConfig) => {
    setChartType(chart.chart_type);
    setSelectedParams(chart.config.params || []);
    setSelectedMetric(chart.config.metric);
    setIsAnalyzing(true);  // 加载配置后开始分析
  };

  // 开始分析
  const handleStartAnalysis = () => {
    setIsAnalyzing(true);
  };

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

  // 导出图表
  const handleChartExport = useCallback((chartInstance: ReactECharts, fileName: string) => {
    const echartsInstance = chartInstance.getEchartsInstance();
    const url = echartsInstance.getDataURL({
      type: 'png',
      pixelRatio: 2,
      backgroundColor: '#fff',
    });
    const link = document.createElement('a');
    link.href = url;
    link.download = `${fileName}.png`;
    link.click();
  }, []);

  // 渲染图表内容
  const renderCharts = () => {
    // 影响度排序不需要选择参数，直接显示
    if (chartType === 'sensitivity') {
      return (
        <Card title="参数影响度排序" size="small">
          <Text type="secondary" style={{ display: 'block', marginBottom: 16 }}>
            影响度 = 该参数对 {metricName} 方差的贡献比例，通过各取值平均性能的方差/总方差计算，值越大表示该参数对 {metricName} 的独立影响越大
          </Text>
          <InfluenceRankingChart
            experimentId={experimentId}
            metric={selectedMetric}
            onExport={(chart) => handleChartExport(chart, `influence_${metricName}`)}
          />
        </Card>
      );
    }

    // 其他图表类型需要点击"开始分析"
    if (!isAnalyzing) {
      return (
        <Card>
          <Empty
            description="选择参数后，点击上方「开始分析」按钮查看图表"
            style={{ padding: 60 }}
          />
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
      // 单参数折线图：为每个选中的参数画一个图，每行两个
      return (
        <Row gutter={[16, 16]}>
          {selectedParams.map(param => (
            <Col span={12} key={param}>
              <Card title={param} size="small">
                <SingleParamLineChart
                  experimentId={experimentId}
                  param={param}
                  metric={selectedMetric}
                  metricName={metricName}
                  onExport={(chart) => handleChartExport(chart, `line_${param}_${metricName}`)}
                />
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
                  onExport={(chart) => handleChartExport(chart, `heatmap_${paramX}_${paramY}_${metricName}`)}
                />
              </Card>
            </Col>
          ))}
        </Row>
      );
    }

    return null;
  };

  // 图表类型名称映射
  const chartTypeNames: Record<ChartType, string> = {
    sensitivity: '影响度排序',
    line: '单参数曲线',
    heatmap: '双参数热力图',
  };

  return (
    <div>
      {/* 已保存的图表列表 */}
      {savedCharts.length > 0 && (
        <Card
          title="已保存的图表配置"
          size="small"
          style={{ marginBottom: 16 }}
          loading={loadingCharts}
        >
          <Space wrap>
            {savedCharts.map((chart) => (
              <Button
                key={chart.id}
                onClick={() => handleLoadChart(chart)}
                style={{ display: 'flex', alignItems: 'center', gap: 8 }}
              >
                <span>{chart.name}</span>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  ({chartTypeNames[chart.chart_type]})
                </Text>
                <Popconfirm
                  title="确定删除这个图表配置?"
                  onConfirm={(e) => {
                    e?.stopPropagation();
                    handleDeleteChart(chart.id);
                  }}
                  onCancel={(e) => e?.stopPropagation()}
                  okText="删除"
                  cancelText="取消"
                >
                  <DeleteOutlined
                    style={{ color: '#ff4d4f', marginLeft: 4 }}
                    onClick={(e) => e.stopPropagation()}
                  />
                </Popconfirm>
              </Button>
            ))}
          </Space>
        </Card>
      )}

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
              <Radio.Button value="heatmap">
                <HeatMapOutlined /> 双参数热力图
              </Radio.Button>
              <Radio.Button value="line">
                <LineChartOutlined /> 单参数曲线
              </Radio.Button>
              <Radio.Button value="sensitivity">
                <BarChartOutlined /> 影响度排序
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
          <Col>
            {/* 影响度排序不需要"开始分析"按钮，其他图表类型需要 */}
            {chartType !== 'sensitivity' && !isAnalyzing ? (
              <Button
                type="primary"
                icon={<BarChartOutlined />}
                onClick={handleStartAnalysis}
              >
                开始分析
              </Button>
            ) : (
              <Tooltip title="保存当前图表配置">
                <Button
                  icon={<SaveOutlined />}
                  onClick={() => setSaveModalVisible(true)}
                >
                  保存配置
                </Button>
              </Tooltip>
            )}
          </Col>
        </Row>
      </Card>

      {/* 图表区域 */}
      {renderCharts()}

      {/* 保存图表对话框 */}
      <Modal
        title="保存图表配置"
        open={saveModalVisible}
        onOk={handleSaveChart}
        onCancel={() => {
          setSaveModalVisible(false);
          setChartName('');
        }}
        confirmLoading={savingChart}
        okText="保存"
        cancelText="取消"
      >
        <div style={{ marginBottom: 16 }}>
          <Text>当前配置：</Text>
          <ul style={{ margin: '8px 0', paddingLeft: 20 }}>
            <li>图表类型：{chartTypeNames[chartType]}</li>
            <li>性能指标：{selectedMetric || 'performance'}</li>
            {chartType !== 'sensitivity' && (
              <li>选中参数：{selectedParams.length > 0 ? selectedParams.join(', ') : '无'}</li>
            )}
          </ul>
        </div>
        <Input
          placeholder="请输入图表名称"
          value={chartName}
          onChange={(e) => setChartName(e.target.value)}
          onPressEnter={handleSaveChart}
        />
      </Modal>
    </div>
  );
}
