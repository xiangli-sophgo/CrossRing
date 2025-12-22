/**
 * FIFO波形图组件 - 显示flit在FIFO中的占用时段（阶梯波形）
 */

import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import type { FIFOSignal } from '@/api/fifoWaveform';
import { FIFO_COLORS } from '@/api/fifoWaveform';

interface Props {
  signals: FIFOSignal[];
  timeRange: { start_ns: number; end_ns: number };
  height?: number;
}

const ROW_HEIGHT = 50;
const HEADER_HEIGHT = 60;

interface WaveformPoint {
  time: number;
  level: number;  // 0: 低电平, 1+: 高电平（重叠的flit数量）
  flits: string[];  // 当前时间点占用的flit列表
}

interface EventWithFlit {
  enter_ns: number;
  leave_ns: number;
  flit_id: string;
}

/**
 * 将事件列表转换为阶梯波形数据点（带flit信息）
 */
function buildStepWaveform(
  events: EventWithFlit[],
  timeStart: number,
  timeEnd: number
): WaveformPoint[] {
  if (events.length === 0) {
    return [
      { time: timeStart, level: 0, flits: [] },
      { time: timeEnd, level: 0, flits: [] },
    ];
  }

  // 收集所有时间点及其变化
  const timeChanges: { time: number; delta: number; flit_id: string; isEnter: boolean }[] = [];
  events.forEach(event => {
    timeChanges.push({ time: event.enter_ns, delta: 1, flit_id: event.flit_id, isEnter: true });
    timeChanges.push({ time: event.leave_ns, delta: -1, flit_id: event.flit_id, isEnter: false });
  });

  // 按时间排序
  timeChanges.sort((a, b) => a.time - b.time);

  // 生成阶梯波形
  const points: WaveformPoint[] = [];
  let currentLevel = 0;
  let currentFlits: string[] = [];

  // 起始点
  if (timeChanges[0].time > timeStart) {
    points.push({ time: timeStart, level: 0, flits: [] });
  }

  // 处理每个时间点
  let i = 0;
  while (i < timeChanges.length) {
    const currentTime = timeChanges[i].time;

    // 在电平变化前，先画一条到当前时间的水平线
    if (points.length > 0 && points[points.length - 1].time < currentTime) {
      points.push({ time: currentTime, level: currentLevel, flits: [...currentFlits] });
    }

    // 累积同一时间点的所有变化
    while (i < timeChanges.length && timeChanges[i].time === currentTime) {
      const change = timeChanges[i];
      currentLevel += change.delta;
      if (change.isEnter) {
        currentFlits.push(change.flit_id);
      } else {
        currentFlits = currentFlits.filter(f => f !== change.flit_id);
      }
      i++;
    }

    // 添加新电平
    points.push({ time: currentTime, level: Math.max(0, currentLevel), flits: [...currentFlits] });
  }

  // 结束点
  if (points[points.length - 1].time < timeEnd) {
    points.push({ time: timeEnd, level: currentLevel, flits: [...currentFlits] });
  }

  return points;
}

export default function FIFOWaveformChart({ signals, timeRange, height }: Props) {
  const chartHeight = height || Math.max(400, signals.length * ROW_HEIGHT + HEADER_HEIGHT + 100);

  // 预处理波形数据，存储flit信息用于tooltip
  const waveformDataMap = useMemo(() => {
    const map: Map<number, WaveformPoint[]> = new Map();
    signals.forEach((signal, signalIndex) => {
      const waveformPoints = buildStepWaveform(
        signal.events as EventWithFlit[],
        timeRange.start_ns,
        timeRange.end_ns
      );
      map.set(signalIndex, waveformPoints);
    });
    return map;
  }, [signals, timeRange]);

  const option = useMemo(() => {
    // 为每个信号构建阶梯波形series
    const series = signals.map((signal, signalIndex) => {
      const waveformPoints = waveformDataMap.get(signalIndex) || [];

      // 转换为ECharts数据格式 [x, categoryIndex]
      // 使用category索引，高低电平通过自定义renderItem控制
      const data = waveformPoints.map(pt => [
        pt.time,
        signalIndex,
        pt.level,  // 保存电平信息用于绘制
      ]);

      return {
        name: signal.name,
        type: 'custom',
        renderItem: (params: any, api: any) => {
          const dataIndex = params.dataIndex;
          const nextDataIndex = dataIndex + 1;
          if (nextDataIndex >= data.length) return null;

          const startTime = api.value(0);
          const categoryIndex = api.value(1);
          const level = api.value(2);
          const nextTime = data[nextDataIndex][0];
          const nextLevel = data[nextDataIndex][2];

          // 获取坐标系边界用于裁剪
          const coordSys = params.coordSys;
          const clipLeft = coordSys.x;
          const clipRight = coordSys.x + coordSys.width;

          // 获取坐标
          const startCoord = api.coord([startTime, categoryIndex]);
          const endCoord = api.coord([nextTime, categoryIndex]);
          const bandHeight = api.size([0, 1])[1];

          // 高电平向上偏移，低电平在基准线
          const waveHeight = bandHeight * 0.35;
          const baseY = startCoord[1];
          const highY = baseY - waveHeight;
          const currentY = level > 0 ? highY : baseY;
          const nextY = nextLevel > 0 ? highY : baseY;

          const color = FIFO_COLORS[signal.fifo_type] || '#999';
          const children: any[] = [];

          // 同一时间点的电平变化：只画垂直线
          if (startTime === nextTime) {
            if (currentY !== nextY && startCoord[0] >= clipLeft && startCoord[0] <= clipRight) {
              children.push({
                type: 'line',
                shape: { x1: startCoord[0], y1: currentY, x2: startCoord[0], y2: nextY },
                style: { stroke: color, lineWidth: 2 },
              });
            }
            return children.length > 0 ? { type: 'group', children } : null;
          }

          // 裁剪到坐标系范围内
          let x1 = Math.max(startCoord[0], clipLeft);
          let x2 = Math.min(endCoord[0], clipRight);
          if (x1 >= x2) return null;

          // 绘制水平线段
          children.push({
            type: 'line',
            shape: { x1, y1: currentY, x2, y2: currentY },
            style: { stroke: color, lineWidth: 2 },
          });

          // 垂直跳变线（如果电平变化且终点在可见范围内）
          if (currentY !== nextY && endCoord[0] >= clipLeft && endCoord[0] <= clipRight) {
            children.push({
              type: 'line',
              shape: { x1: endCoord[0], y1: currentY, x2: endCoord[0], y2: nextY },
              style: { stroke: color, lineWidth: 2 },
            });
          }

          return { type: 'group', children };
        },
        data: data,
        clip: true,
        z: 10,
      };
    });

    // 查找某时间点对应的flit列表
    const findFlitsAtTime = (signalIndex: number, time: number): string[] => {
      const points = waveformDataMap.get(signalIndex);
      if (!points) return [];
      // 找到time之前最近的点
      for (let i = points.length - 1; i >= 0; i--) {
        if (points[i].time <= time) {
          return points[i].flits;
        }
      }
      return [];
    };

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'line',
        },
        formatter: (params: any) => {
          if (!params || params.length === 0) return '';
          // 从辅助series获取时间
          const time = params[0].value[0];
          // 2GHz频率，0.5ns精度
          const timeStr = Number.isInteger(time * 2) ? time.toFixed(1) : time.toFixed(2);
          let html = `<div><b>时间: ${timeStr} ns</b></div>`;
          // 按信号遍历，避免重复
          signals.forEach((signal, signalIdx) => {
            const flits = findFlitsAtTime(signalIdx, time);
            const isHigh = flits.length > 0;
            const status = isHigh ? '占用' : '空闲';
            const color = FIFO_COLORS[signal.fifo_type] || '#999';
            html += `<div style="color:${color}">${signal.name}: ${status}`;
            if (isHigh) {
              html += `<br/>&nbsp;&nbsp;Flits: ${flits.join(', ')}`;
            }
            html += `</div>`;
          });
          return html;
        },
      },
      grid: {
        left: 150,
        right: 30,
        top: 30,
        bottom: 80,
      },
      xAxis: {
        type: 'value',
        name: '时间 (ns)',
        nameLocation: 'middle',
        nameGap: 35,
        min: timeRange.start_ns,
        max: timeRange.end_ns,
        minInterval: 0.5,  // 2GHz频率，最小间隔0.5ns
        axisLabel: {
          formatter: (value: number) => {
            // 0.5ns精度显示
            return Number.isInteger(value * 2) ? value.toFixed(1) : value.toFixed(2);
          },
        },
      },
      yAxis: {
        type: 'category',
        data: signals.map(s => s.name),
        inverse: true,
        axisLabel: {
          width: 130,
          overflow: 'truncate',
          ellipsis: '...',
        },
        axisTick: {
          show: false,
        },
        splitLine: {
          show: true,
          lineStyle: {
            type: 'dashed',
            color: '#e8e8e8',
          },
        },
      },
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: 0,
          filterMode: 'none',
          zoomOnMouseWheel: 'ctrl',
          moveOnMouseWheel: 'shift',
          zoomOnMouseMove: false,
          moveOnMouseMove: false,
          preventDefaultMouseMove: false,
          startValue: timeRange.start_ns,
          endValue: timeRange.end_ns - timeRange.start_ns > 500
            ? timeRange.start_ns + 500
            : timeRange.end_ns,
        },
        {
          type: 'slider',
          xAxisIndex: 0,
          height: 20,
          bottom: 10,
          filterMode: 'none',
          startValue: timeRange.start_ns,
          endValue: timeRange.end_ns - timeRange.start_ns > 500
            ? timeRange.start_ns + 500
            : timeRange.end_ns,
        },
      ],
      series: [
        // 辅助series：用于触发tooltip的密集数据点（0.5ns间隔，最多2000点）
        {
          type: 'scatter',
          data: (() => {
            const points: [number, number][] = [];
            const range = timeRange.end_ns - timeRange.start_ns;
            const step = Math.max(0.5, range / 2000);
            for (let t = timeRange.start_ns; t <= timeRange.end_ns; t += step) {
              points.push([t, 0]);
            }
            return points;
          })(),
          symbol: 'none',
          silent: false,
          z: 1,
          xAxisIndex: 0,
          yAxisIndex: 0,
        },
        ...series,
      ],
    };
  }, [signals, timeRange, waveformDataMap]);

  return (
    <ReactECharts
      option={option}
      style={{ height: chartHeight }}
      opts={{ renderer: 'canvas' }}
    />
  );
}
