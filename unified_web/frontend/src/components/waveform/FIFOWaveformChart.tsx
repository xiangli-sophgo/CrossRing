/**
 * FIFO波形图组件 - 显示flit在FIFO中的占用时段（阶梯波形）
 */

import { useMemo, useRef, useState, useCallback, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import type { FIFOSignal } from '@/api/fifoWaveform';
import { getFIFOColor } from '@/api/fifoWaveform';

interface Props {
  signals: FIFOSignal[];
  timeRange: { start_ns: number; end_ns: number };
  height?: number;
}

// 时间标记类型
interface TimeMarkers {
  primary: number | null;    // 主标记（橙色）
  secondary: number | null;  // 次标记（蓝色）
}

const FIXED_CHART_HEIGHT = 400;  // 固定高度
const VISIBLE_SIGNALS = 20;  // 默认显示信号数（增大以减小间距）

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

  // 始终从 timeStart 开始，确保有前导低电平
  points.push({ time: timeStart, level: 0, flits: [] });

  // 处理每个时间点
  let i = 0;
  while (i < timeChanges.length) {
    const currentTime = timeChanges[i].time;

    // 在电平变化前，先画一条到当前时间的水平线（保持当前电平）
    if (points[points.length - 1].time < currentTime) {
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
  const chartHeight = height || FIXED_CHART_HEIGHT;
  const chartRef = useRef<ReactECharts>(null);

  // 时间标记状态
  const [markers, setMarkers] = useState<TimeMarkers>({ primary: null, secondary: null });

  // 格式化时间显示
  const formatTime = (time: number) => {
    return Number.isInteger(time * 2) ? time.toFixed(1) : time.toFixed(2);
  };

  // 计算时间差
  const timeDelta = useMemo(() => {
    if (markers.primary !== null && markers.secondary !== null) {
      return Math.abs(markers.secondary - markers.primary);
    }
    return null;
  }, [markers]);

  // 处理图表点击事件 - 使用zrender监听整个画布
  const bindChartClick = useCallback(() => {
    const chart = chartRef.current?.getEchartsInstance();
    if (!chart) return;

    const zr = chart.getZr();
    // 移除旧的监听器
    zr.off('click');
    // 添加新的监听器
    zr.on('click', (params: any) => {
      const pointInPixel = [params.offsetX, params.offsetY];
      // 检查是否在grid区域内
      if (!chart.containPixel('grid', pointInPixel)) return;

      const pointInGrid = chart.convertFromPixel('grid', pointInPixel);
      if (!pointInGrid) return;

      // 对齐到0.5ns粒度
      const rawTime = pointInGrid[0];
      const alignedTime = Math.round(rawTime * 2) / 2;

      if (alignedTime < timeRange.start_ns || alignedTime > timeRange.end_ns) return;

      // Shift+点击设置次标记，普通点击设置主标记
      if (params.event.shiftKey) {
        setMarkers(prev => ({ ...prev, secondary: alignedTime }));
      } else {
        setMarkers(prev => ({ ...prev, primary: alignedTime }));
      }
    });
  }, [timeRange]);

  // 清除标记
  const clearMarkers = useCallback(() => {
    setMarkers({ primary: null, secondary: null });
  }, []);

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

  // 使用 ref 存储最新数据，确保 formatter 能访问到
  const dataRef = useRef<{
    waveformDataMap: Map<number, WaveformPoint[]>;
    signals: FIFOSignal[];
  }>({ waveformDataMap: new Map(), signals: [] });
  dataRef.current = { waveformDataMap, signals };

  // 按节点分组信号，返回分组信息
  const groupedSignalInfo = useMemo(() => {
    // 提取节点号: "Node_3.IQ_TD.data" -> "Node_3"
    const getNodeKey = (name: string) => {
      const match = name.match(/^(Node_\d+)/);
      return match ? match[1] : 'Other';
    };

    // 按节点分组并记录每组的起始索引
    const groups: { nodeKey: string; startIndex: number; count: number }[] = [];
    let currentNode = '';
    signals.forEach((signal, index) => {
      const nodeKey = getNodeKey(signal.name);
      if (nodeKey !== currentNode) {
        groups.push({ nodeKey, startIndex: index, count: 1 });
        currentNode = nodeKey;
      } else {
        groups[groups.length - 1].count++;
      }
    });

    return groups;
  }, [signals]);

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

          // 高电平向上偏移，低电平在基准线（占用更多垂直空间）
          const waveHeight = bandHeight * 0.7;
          const baseY = startCoord[1] + bandHeight * 0.3;  // 基准线下移
          const highY = baseY - waveHeight;
          const currentY = level > 0 ? highY : baseY;
          const nextY = nextLevel > 0 ? highY : baseY;

          const color = getFIFOColor(signal.fifo_type);
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

    // 为不同节点组生成背景色 series（只有多个分组时才显示）
    const groupBackgroundSeries = groupedSignalInfo.length > 1
      ? groupedSignalInfo.map((group, idx) => ({
          type: 'custom',
          renderItem: (params: any, api: any) => {
            const coordSys = params.coordSys;
            const startY = api.coord([0, group.startIndex])[1];
            const endY = api.coord([0, group.startIndex + group.count - 1])[1];
            const bandHeight = api.size([0, 1])[1];

            return {
              type: 'rect',
              shape: {
                x: coordSys.x,
                y: startY - bandHeight / 2,
                width: coordSys.width,
                height: (endY - startY) + bandHeight,
              },
              style: {
                fill: idx % 2 === 0 ? 'rgba(59, 130, 246, 0.06)' : 'rgba(249, 115, 22, 0.06)',
              },
              z: 0,
            };
          },
          data: [[0, group.startIndex]],  // 只需要一个数据点触发渲染
          silent: true,
          z: 0,
        }))
      : [];

    return {
      // 图形元素：在左侧显示分组标签（只有多个分组时才显示）
      graphic: groupedSignalInfo.length > 1
        ? groupedSignalInfo.map((group, idx) => ({
            type: 'text',
            left: 5,
            top: `${30 + ((group.startIndex + group.count / 2) / signals.length) * (chartHeight - 110)}px`,
            style: {
              text: group.nodeKey.replace('Node_', 'N'),
              fontSize: 11,
              fontWeight: 'bold',
              fill: idx % 2 === 0 ? '#3b82f6' : '#f97316',
            },
            z: 100,
          }))
        : [],
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'line',
          axis: 'x',
          snap: false,
        },
        formatter: (params: any, ticket: any, callback: any) => {
          // 从 ref 获取最新数据
          const currentData = dataRef.current;
          const currentSignals = currentData.signals;
          const currentWaveformDataMap = currentData.waveformDataMap;

          // 查找某时间点对应的flit列表
          const findFlitsAtTime = (signalIndex: number, time: number): string[] => {
            const points = currentWaveformDataMap.get(signalIndex);
            if (!points) return [];
            // 找到time之前最近的点
            for (let i = points.length - 1; i >= 0; i--) {
              if (points[i].time <= time) {
                return points[i].flits;
              }
            }
            return [];
          };

          // 从 params 获取时间 - 检查多种格式
          let time: number | undefined;
          if (params && params.length > 0) {
            // 尝试从第一个参数获取时间
            const firstParam = params[0];
            if (firstParam.axisValue !== undefined) {
              time = firstParam.axisValue;
            } else if (firstParam.value && Array.isArray(firstParam.value)) {
              time = firstParam.value[0];
            } else if (typeof firstParam.value === 'number') {
              time = firstParam.value;
            }
          }

          if (time === undefined) return '';

          // 2GHz频率，0.5ns精度
          const timeStr = Number.isInteger(time * 2) ? time.toFixed(1) : time.toFixed(2);
          let html = `<div><b>时间: ${timeStr} ns</b></div>`;
          // 按信号遍历，避免重复
          currentSignals.forEach((signal, signalIdx) => {
            const flits = findFlitsAtTime(signalIdx, time!);
            const isHigh = flits.length > 0;
            const status = isHigh ? '占用' : '空闲';
            const color = getFIFOColor(signal.fifo_type);
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
          width: 140,
          overflow: 'truncate',
          ellipsis: '...',
          formatter: (value: string) => {
            // 简化显示: "Node_3.IQ_TD.data" -> "IQ_TD.data"
            // 因为分组已经显示了节点号
            const parts = value.split('.');
            if (parts.length >= 3) {
              return `${parts[1]}.${parts[2]}`;
            }
            return value;
          },
          rich: {
            group: {
              fontWeight: 'bold',
              color: '#1890ff',
            },
          },
        },
        axisTick: {
          show: false,
        },
        splitLine: {
          show: true,
          interval: (index: number) => {
            // 在分组边界处显示更明显的分隔线
            const isGroupStart = groupedSignalInfo.some(g => g.startIndex === index && index > 0);
            return !isGroupStart;  // 返回false表示显示分隔线
          },
          lineStyle: {
            type: 'dashed',
            color: '#e8e8e8',
          },
        },
      },
      dataZoom: [
        // X轴缩放（时间轴）
        {
          type: 'inside',
          xAxisIndex: 0,
          filterMode: 'none',
          zoomOnMouseWheel: 'ctrl',
          moveOnMouseWheel: true,
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
        // Y轴滚动（信号列表）
        {
          type: 'slider',
          yAxisIndex: 0,
          width: 20,
          right: 10,
          filterMode: 'none',
          start: 0,
          end: signals.length <= VISIBLE_SIGNALS ? 100 : (VISIBLE_SIGNALS / signals.length) * 100,
          showDetail: false,
          brushSelect: false,
        },
        {
          type: 'inside',
          yAxisIndex: 0,
          filterMode: 'none',
          zoomOnMouseWheel: false,
          moveOnMouseWheel: true,
          zoomOnMouseMove: false,
          moveOnMouseMove: false,
          start: 0,
          end: signals.length <= VISIBLE_SIGNALS ? 100 : (VISIBLE_SIGNALS / signals.length) * 100,
        },
      ],
      series: [
        // 辅助series：为每个category生成数据点，用于触发tooltip（0.5ns粒度，最多4000点/信号）
        // 第一个series添加markLine用于显示时间标记
        ...signals.map((_, signalIndex) => ({
          type: 'scatter',
          data: (() => {
            const points: [number, number][] = [];
            const range = timeRange.end_ns - timeRange.start_ns;
            const step = Math.max(0.5, range / 4000);  // 0.5ns粒度，但最多4000点
            for (let t = timeRange.start_ns; t <= timeRange.end_ns; t += step) {
              points.push([t, signalIndex]);
            }
            return points;
          })(),
          symbol: 'none',
          silent: false,
          z: 1,
          xAxisIndex: 0,
          yAxisIndex: 0,
          // 第一个series添加id用于单独更新markLine
          ...(signalIndex === 0 ? {
            id: 'marker-bindable-series',
          } : {}),
        })),
        ...groupBackgroundSeries,  // 分组背景色
        ...series,
      ],
    };
  }, [signals, timeRange, waveformDataMap, groupedSignalInfo, chartHeight]);

  // 单独更新 markLine，不触发整个 option 重建，避免 dataZoom 重置
  useEffect(() => {
    const chart = chartRef.current?.getEchartsInstance();
    if (!chart) return;

    const markLineData: any[] = [];
    if (markers.primary !== null) {
      markLineData.push({
        xAxis: markers.primary,
        lineStyle: { color: '#f97316', width: 2, type: 'solid' },
        label: { formatter: `A: ${formatTime(markers.primary)} ns`, position: 'start', color: '#f97316' },
      });
    }
    if (markers.secondary !== null) {
      markLineData.push({
        xAxis: markers.secondary,
        lineStyle: { color: '#3b82f6', width: 2, type: 'solid' },
        label: { formatter: `B: ${formatTime(markers.secondary)} ns`, position: 'start', color: '#3b82f6' },
      });
    }

    // 只更新指定 id 的 series 的 markLine
    chart.setOption({
      series: [{
        id: 'marker-bindable-series',
        markLine: {
          silent: true,
          symbol: 'none',
          data: markLineData,
          animation: false,
        },
      }],
    });
  }, [markers]);

  // 图表ready时绑定点击事件
  const onChartReady = useCallback(() => {
    bindChartClick();
  }, [bindChartClick]);

  // timeRange变化时重新绑定
  useEffect(() => {
    bindChartClick();
  }, [bindChartClick]);

  return (
    <div>
      {/* 标记信息栏 */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        padding: '8px 12px',
        marginBottom: 8,
        background: '#fafafa',
        borderRadius: 4,
        fontSize: 13,
      }}>
        <span style={{ color: '#666' }}>时间标记：</span>
        <span style={{ color: '#f97316' }}>
          A: {markers.primary !== null ? `${formatTime(markers.primary)} ns` : '--'}
        </span>
        <span style={{ color: '#3b82f6' }}>
          B: {markers.secondary !== null ? `${formatTime(markers.secondary)} ns` : '--'}
        </span>
        {timeDelta !== null && (
          <span style={{ color: '#10b981', fontWeight: 500 }}>
            Δt: {formatTime(timeDelta)} ns
          </span>
        )}
        <span style={{ color: '#999', marginLeft: 'auto', fontSize: 12 }}>
          点击设置A，Shift+点击设置B
        </span>
        {(markers.primary !== null || markers.secondary !== null) && (
          <button
            onClick={clearMarkers}
            style={{
              padding: '2px 8px',
              border: '1px solid #d9d9d9',
              borderRadius: 4,
              background: '#fff',
              cursor: 'pointer',
              fontSize: 12,
            }}
          >
            清除
          </button>
        )}
      </div>

      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: chartHeight }}
        opts={{ renderer: 'canvas' }}
        notMerge={false}
        lazyUpdate={true}
        onChartReady={onChartReady}
      />
    </div>
  );
}
