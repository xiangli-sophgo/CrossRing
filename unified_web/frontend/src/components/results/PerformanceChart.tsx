/**
 * 性能分布图组件
 */

import { useEffect, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { Spin, Empty } from 'antd';
import { getDistribution } from '../api';
import type { PerformanceDistribution } from '../types';

interface Props {
  experimentId: number;
  distribution?: PerformanceDistribution;
}

export default function PerformanceChart({ experimentId, distribution: initialDistribution }: Props) {
  const [loading, setLoading] = useState(!initialDistribution);
  const [distribution, setDistribution] = useState<PerformanceDistribution | null>(
    initialDistribution || null
  );

  useEffect(() => {
    if (!initialDistribution) {
      loadDistribution();
    }
  }, [experimentId]);

  const loadDistribution = async () => {
    setLoading(true);
    try {
      const data = await getDistribution(experimentId, 50);
      setDistribution(data);
    } catch (error) {
      console.error('加载性能分布失败', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 50 }}>
        <Spin />
      </div>
    );
  }

  if (!distribution || !distribution.histogram || distribution.histogram.length === 0) {
    return <Empty description="暂无数据" />;
  }

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
      formatter: (params: { data: number[] }[]) => {
        const data = params[0]?.data;
        if (!data) return '';
        return `性能范围: ${data[0].toFixed(2)} - ${data[1].toFixed(2)} GB/s<br/>数量: ${data[2]}`;
      },
    },
    xAxis: {
      type: 'value',
      name: '性能 (GB/s)',
      nameLocation: 'middle',
      nameGap: 30,
    },
    yAxis: {
      type: 'value',
      name: '数量',
    },
    series: [
      {
        type: 'bar',
        data: distribution.histogram.map((h) => [(h[0] + h[1]) / 2, h[2]]),
        barWidth: '90%',
        itemStyle: {
          color: '#1890ff',
        },
      },
    ],
    grid: {
      left: 60,
      right: 30,
      top: 30,
      bottom: 50,
    },
    dataZoom: [
      {
        type: 'inside',
        xAxisIndex: 0,
      },
      {
        type: 'slider',
        xAxisIndex: 0,
        height: 20,
        bottom: 5,
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: 350 }} />;
}
