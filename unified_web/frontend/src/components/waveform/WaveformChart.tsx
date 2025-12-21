/**
 * 波形图组件 - 使用ECharts的custom series渲染波形
 */

import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import type { WaveformSignal, WaveformEvent } from '@/api/waveform';
import { STAGE_COLORS, STAGE_NAMES } from '@/api/waveform';

interface Props {
  signals: WaveformSignal[];
  timeRange: { start_ns: number; end_ns: number };
  stages: string[];
  height?: number;
}

// 信号行高度
const ROW_HEIGHT = 24;
const HEADER_HEIGHT = 60;

export default function WaveformChart({ signals, timeRange, stages, height }: Props) {
  const chartHeight = height || Math.max(300, signals.length * ROW_HEIGHT + HEADER_HEIGHT + 80);

  const option = useMemo(() => {
    // 信号名列表（Y轴）
    const signalNames = signals.map(s => s.name);

    // 构建series数据
    const seriesData: Array<{
      value: [number, number, number, number, string];
      itemStyle: { color: string };
    }> = [];

    signals.forEach((signal, signalIndex) => {
      signal.events.forEach((event: WaveformEvent) => {
        seriesData.push({
          value: [
            event.start_ns,
            event.end_ns,
            signalIndex,
            signalIndex,
            event.stage,
          ],
          itemStyle: {
            color: STAGE_COLORS[event.stage] || '#999',
          },
        });
      });
    });

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: { value: [number, number, number, number, string] }) => {
          const [start, end, , , stage] = params.value;
          const duration = end - start;
          const stageName = STAGE_NAMES[stage] || stage;
          return `
            <div style="font-weight: bold">${stageName}</div>
            <div>开始: ${start.toFixed(2)} ns</div>
            <div>结束: ${end.toFixed(2)} ns</div>
            <div>持续: ${duration.toFixed(2)} ns</div>
          `;
        },
      },
      legend: {
        data: stages.map(s => STAGE_NAMES[s] || s),
        top: 0,
        itemWidth: 14,
        itemHeight: 14,
      },
      grid: {
        left: 120,
        right: 30,
        top: HEADER_HEIGHT,
        bottom: 60,
      },
      xAxis: {
        type: 'value',
        name: '时间 (ns)',
        nameLocation: 'middle',
        nameGap: 25,
        min: timeRange.start_ns,
        max: timeRange.end_ns,
        axisLabel: {
          formatter: (value: number) => value.toFixed(1),
        },
      },
      yAxis: {
        type: 'category',
        data: signalNames,
        inverse: true,
        axisLabel: {
          width: 100,
          overflow: 'truncate',
          ellipsis: '...',
        },
        axisTick: {
          show: false,
        },
      },
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: 0,
          filterMode: 'none',
        },
        {
          type: 'slider',
          xAxisIndex: 0,
          height: 20,
          bottom: 10,
          filterMode: 'none',
        },
      ],
      series: [
        // 为每个阶段创建一个series用于图例
        ...stages.map(stage => ({
          name: STAGE_NAMES[stage] || stage,
          type: 'custom',
          renderItem: () => null,
          data: [],
          itemStyle: {
            color: STAGE_COLORS[stage] || '#999',
          },
        })),
        // 实际数据series
        {
          type: 'custom',
          clip: true,  // 裁剪超出坐标轴范围的图形
          renderItem: (
            params: { coordSys: { x: number; y: number; width: number; height: number } },
            api: {
              value: (idx: number) => number | string;
              coord: (val: [number, number]) => [number, number];
              size: (val: [number, number]) => [number, number];
              style: () => Record<string, unknown>;
            }
          ) => {
            const start = api.value(0) as number;
            const end = api.value(1) as number;
            const signalIndex = api.value(2) as number;

            const startCoord = api.coord([start, signalIndex]);
            const endCoord = api.coord([end, signalIndex]);
            const rectHeight = api.size([0, 1])[1] * 0.7;

            return {
              type: 'rect',
              shape: {
                x: startCoord[0],
                y: startCoord[1] - rectHeight / 2,
                width: Math.max(endCoord[0] - startCoord[0], 2),
                height: rectHeight,
              },
              style: api.style(),
            };
          },
          data: seriesData,
          encode: {
            x: [0, 1],
            y: 2,
          },
        },
      ],
    };
  }, [signals, timeRange, stages]);

  return (
    <ReactECharts
      option={option}
      style={{ height: chartHeight }}
      opts={{ renderer: 'canvas' }}
    />
  );
}
