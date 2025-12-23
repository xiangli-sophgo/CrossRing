/**
 * FIFO波形图组件 - 显示flit在FIFO中的占用时段（阶梯波形）
 */

import { useMemo, useRef, useState, useCallback, useEffect, memo } from 'react';
import ReactECharts from 'echarts-for-react';
import type { FIFOSignal } from '@/api/fifoWaveform';
import { getFIFOColor, getSignalColor } from '@/api/fifoWaveform';

interface Props {
  signals: FIFOSignal[];
  timeRange: { start_ns: number; end_ns: number };
  height?: number;
  onFlitClick?: (packetId: number) => void;
  expandedRspSignals?: string[];  // 已展开的 rsp 信号（如 ["IQ_TR.rsp"]）
  expandedReqSignals?: string[];  // 已展开的 req 信号（如 ["IQ_TR.req"]）
  expandedDataSignals?: string[]; // 已展开的 data 信号（如 ["IQ_TR.data"]）
  onToggleExpand?: (signalKey: string) => void;  // 切换展开状态
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

// 从 flit_id 解析 packet_id
const parsePacketId = (flitId: string): number | null => {
  // flit_id 格式: "123.req.0" -> packet_id = 123
  const parts = flitId.split('.');
  if (parts.length >= 1) {
    const pktId = parseInt(parts[0], 10);
    return isNaN(pktId) ? null : pktId;
  }
  return null;
};

function FIFOWaveformChart({ signals, timeRange, height, onFlitClick, expandedRspSignals = [], expandedReqSignals = [], expandedDataSignals = [], onToggleExpand }: Props) {
  const chartHeight = height || FIXED_CHART_HEIGHT;
  const chartRef = useRef<ReactECharts>(null);

  // 时间标记状态
  const [markers, setMarkers] = useState<TimeMarkers>({ primary: null, secondary: null });
  // 用于强制清除残留
  const [chartKey, setChartKey] = useState(0);

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
    hoverSignalIndex: number;  // 当前悬停的信号索引
    hoverTime: number;  // 当前悬停的时间
  }>({ waveformDataMap: new Map(), signals: [], hoverSignalIndex: 0, hoverTime: 0 });
  dataRef.current.waveformDataMap = waveformDataMap;
  dataRef.current.signals = signals;

  // 查找某时间点对应的flit列表
  const findFlitsAtTime = useCallback((signalIndex: number, time: number): string[] => {
    const points = waveformDataMap.get(signalIndex);
    if (!points) return [];
    // 找到time之前最近的点
    for (let i = points.length - 1; i >= 0; i--) {
      if (points[i].time <= time) {
        return points[i].flits;
      }
    }
    return [];
  }, [waveformDataMap]);

  // 使用 ref 存储 onToggleExpand 以便在事件处理中使用
  const onToggleExpandRef = useRef(onToggleExpand);
  onToggleExpandRef.current = onToggleExpand;

  // 处理图表事件 - 使用zrender监听整个画布
  const bindChartClick = useCallback(() => {
    const chart = chartRef.current?.getEchartsInstance();
    if (!chart) return;

    const zr = chart.getZr();
    // 移除旧的监听器
    zr.off('click');
    zr.off('mousemove');
    chart.off('click');

    // 添加 mousemove 监听器，追踪当前悬停的信号索引和时间，并手动触发 tooltip
    zr.on('mousemove', (params: any) => {
      const pointInPixel = [params.offsetX, params.offsetY];
      if (!chart.containPixel('grid', pointInPixel)) {
        // 鼠标离开 grid 区域时隐藏 tooltip
        chart.dispatchAction({ type: 'hideTip' });
        return;
      }

      const pointInGrid = chart.convertFromPixel('grid', pointInPixel);
      if (!pointInGrid) return;

      const rawTime = pointInGrid[0];
      const signalIndex = Math.round(pointInGrid[1]);

      // 保存悬停位置信息
      if (signalIndex >= 0 && signalIndex < dataRef.current.signals.length) {
        dataRef.current.hoverSignalIndex = signalIndex;
      }
      // 对齐到 0.5ns 粒度
      dataRef.current.hoverTime = Math.round(rawTime * 2) / 2;

      // 手动触发 tooltip 显示
      chart.dispatchAction({
        type: 'showTip',
        x: params.offsetX,
        y: params.offsetY,
      });
    });

    // 添加 zrender click 监听器（用于 Y 轴标签和波形区域点击）
    zr.on('click', (params: any) => {
      const pointInPixel = [params.offsetX, params.offsetY];

      // === Y 轴标签区域点击检测 ===
      const leftMargin = 150;  // 与 grid.left 一致
      if (params.offsetX < leftMargin && params.offsetX > 5) {
        // 获取 grid 坐标信息（使用 any 绕过私有方法类型检查）
        const model = (chart as any).getModel();
        const gridComponent = model?.getComponent('grid', 0);
        if (gridComponent && gridComponent.coordinateSystem) {
          const gridRect = gridComponent.coordinateSystem.getRect();
          const currentSignals = dataRef.current.signals;

          if (currentSignals.length > 0) {
            // 考虑 Y 轴 dataZoom 的影响
            const yDataZoom = model.getComponent('dataZoom', 2);
            const startPercent = yDataZoom?.get('start') ?? 0;
            const endPercent = yDataZoom?.get('end') ?? 100;

            const visibleStart = Math.floor(currentSignals.length * startPercent / 100);
            const visibleEnd = Math.ceil(currentSignals.length * endPercent / 100);
            const visibleCount = Math.max(1, visibleEnd - visibleStart);

            const bandHeight = gridRect.height / visibleCount;
            const relativeY = params.offsetY - gridRect.y;
            const visibleIndex = Math.floor(relativeY / bandHeight);
            const signalIndex = visibleStart + visibleIndex;

            if (signalIndex >= 0 && signalIndex < currentSignals.length) {
              const signal = currentSignals[signalIndex];
              const parts = signal.name.split('.');

              // 支持 req/rsp/data 类型的展开/折叠
              // 3段格式（Node_X.FIFO.type）：可展开
              // 4段格式（Node_X.FIFO.type.subType）：展开后的子类型，可折叠
              if (parts.length >= 3) {
                const fifoType = parts[1];
                const flitType = parts[2];
                if (flitType === 'rsp' || flitType === 'req' || flitType === 'data') {
                  const signalKey = `${fifoType}.${flitType}`;
                  onToggleExpandRef.current?.(signalKey);
                  return;  // 阻止后续处理
                }
              }
            }
          }
        }
      }

      // === 波形区域点击 ===
      // 检查是否在grid区域内
      if (!chart.containPixel('grid', pointInPixel)) return;

      const pointInGrid = chart.convertFromPixel('grid', pointInPixel);
      if (!pointInGrid) return;

      const rawTime = pointInGrid[0];
      const signalIndex = Math.round(pointInGrid[1]);

      // 边界检查
      if (signalIndex < 0 || signalIndex >= dataRef.current.signals.length) return;
      if (rawTime < timeRange.start_ns || rawTime > timeRange.end_ns) return;

      // Ctrl+点击：跳转到请求波形
      if (params.event.ctrlKey && onFlitClick) {
        const flits = findFlitsAtTime(signalIndex, rawTime);
        if (flits.length > 0) {
          const packetIds = flits.map(parsePacketId).filter((id): id is number => id !== null);
          const uniquePacketIds = [...new Set(packetIds)];
          if (uniquePacketIds.length >= 1) {
            onFlitClick(uniquePacketIds[0]);
            return;
          }
        }
      }

      // 普通点击/Shift+点击：设置时间标记
      const alignedTime = Math.round(rawTime * 2) / 2;
      if (params.event.shiftKey) {
        setMarkers(prev => ({ ...prev, secondary: alignedTime }));
      } else {
        setMarkers(prev => ({ ...prev, primary: alignedTime }));
      }
    });
  }, [timeRange, onFlitClick, findFlitsAtTime]);

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

          const color = getSignalColor(signal.name);
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
        show: true,
        trigger: 'axis',
        triggerOn: 'mousemove',
        axisPointer: {
          type: 'none',
        },
        position: (point: number[]) => {
          // tooltip 跟随鼠标位置，稍微偏移避免遮挡
          return [point[0] + 10, point[1] + 10];
        },
        formatter: () => {
          // 从 ref 获取最新数据（包括 mousemove 事件中计算的时间和信号索引）
          const currentData = dataRef.current;
          const currentSignals = currentData.signals;
          const currentWaveformDataMap = currentData.waveformDataMap;
          const time = currentData.hoverTime;
          const signalIndex = currentData.hoverSignalIndex;

          // 查找某时间点对应的flit列表
          const findFlitsAtTime = (idx: number, t: number): string[] => {
            const points = currentWaveformDataMap.get(idx);
            if (!points) return [];
            // 找到time之前最近的点
            for (let i = points.length - 1; i >= 0; i--) {
              if (points[i].time <= t) {
                return points[i].flits;
              }
            }
            return [];
          };

          const signal = currentSignals[signalIndex];
          if (!signal) {
            return '';
          }

          const flits = findFlitsAtTime(signalIndex, time);
          const isOccupied = flits.length > 0;
          const color = getSignalColor(signal.name);
          const timeStr = time.toFixed(1);

          // 空闲状态：简化显示
          if (!isOccupied) {
            return `
              <div style="padding: 8px 12px;">
                <div style="font-weight: bold; margin-bottom: 6px;">时间: ${timeStr} ns</div>
                <div style="color: ${color};">${signal.name}: 空闲</div>
              </div>
            `;
          }

          // 占用状态：详细显示
          const flitList = flits.map(f => `<span style="color: #1890ff;">${f}</span>`).join(', ');
          return `
            <div style="padding: 8px 12px; min-width: 180px;">
              <div style="font-weight: bold; margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid #eee;">
                时间: ${timeStr} ns
              </div>
              <div style="color: ${color}; font-weight: 500; margin-bottom: 4px;">
                ${signal.name}
              </div>
              <div style="display: flex; align-items: center; margin-bottom: 6px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #52c41a; margin-right: 6px;"></span>
                <span>占用中</span>
              </div>
              <div style="background: #f5f5f5; padding: 6px 8px; border-radius: 4px; font-size: 12px;">
                <div style="color: #666; margin-bottom: 2px;">Flits:</div>
                <div>${flitList}</div>
              </div>
              <div style="font-size: 11px; color: #999; margin-top: 6px;">
                Ctrl+点击查看请求波形
              </div>
            </div>
          `;
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
        triggerEvent: true,  // 启用 Y 轴点击事件
        axisLabel: {
          width: 140,
          overflow: 'truncate',
          ellipsis: '...',
          triggerEvent: true,  // 启用点击事件
          formatter: (value: string) => {
            // 简化显示: "Node_3.IQ_TD.data" -> "IQ_TD.data"
            // 因为分组已经显示了节点号
            const parts = value.split('.');
            if (parts.length >= 3) {
              const fifoType = parts[1];
              const flitType = parts[2];

              // 3段格式：可展开的基础类型
              if (parts.length === 3) {
                const signalKey = `${fifoType}.${flitType}`;
                let isExpanded = false;
                if (flitType === 'rsp') {
                  isExpanded = expandedRspSignals.includes(signalKey);
                } else if (flitType === 'req') {
                  isExpanded = expandedReqSignals.includes(signalKey);
                } else if (flitType === 'data') {
                  isExpanded = expandedDataSignals.includes(signalKey);
                }
                const icon = isExpanded ? '▼' : '▶';
                return `${icon} ${fifoType}.${flitType}`;
              }

              // 4段格式：展开后的子类型
              if (parts.length >= 4) {
                const subType = parts[3];
                return `  ▲ ${fifoType}.${flitType}.${subType}`;  // 带折叠箭头
              }
            }
            return value;
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
        ...groupBackgroundSeries,  // 分组背景色
        ...series,
      ],
    };
  }, [signals, timeRange, waveformDataMap, groupedSignalInfo, chartHeight, expandedRspSignals, expandedReqSignals, expandedDataSignals]);

  // 暂时禁用 markLine 更新，测试 tooltip 问题
  // useEffect(() => {
  //   const chart = chartRef.current?.getEchartsInstance();
  //   if (!chart) return;
  //   // ... markLine 更新代码
  // }, [markers]);

  // 图表ready时绑定点击事件
  const onChartReady = useCallback(() => {
    bindChartClick();
  }, [bindChartClick]);

  // timeRange变化时重新绑定
  useEffect(() => {
    bindChartClick();
  }, [bindChartClick]);

  // 信号变化时强制清理图表，解决波形残留问题
  const prevSignalsRef = useRef<string>('');
  useEffect(() => {
    const signalKey = signals.map(s => s.name).join(',');
    if (prevSignalsRef.current && prevSignalsRef.current !== signalKey) {
      const chart = chartRef.current?.getEchartsInstance();
      if (chart) {
        // 清空图表并重新设置
        chart.clear();
        setChartKey(k => k + 1);
      }
    }
    prevSignalsRef.current = signalKey;
  }, [signals]);


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
        key={chartKey}
        ref={chartRef}
        option={option}
        style={{ height: chartHeight }}
        opts={{ renderer: 'canvas' }}
        notMerge={false}
        lazyUpdate={false}
        onChartReady={onChartReady}
      />
    </div>
  );
}

// 自定义比较函数，只有当关键 props 变化时才重新渲染
function propsAreEqual(prevProps: Props, nextProps: Props): boolean {
  // 比较 signals 数组长度和引用
  if (prevProps.signals !== nextProps.signals) {
    if (prevProps.signals.length !== nextProps.signals.length) return false;
    // 如果长度相同，检查第一个和最后一个元素的 name
    if (prevProps.signals.length > 0) {
      if (prevProps.signals[0].name !== nextProps.signals[0].name) return false;
      const lastIdx = prevProps.signals.length - 1;
      if (prevProps.signals[lastIdx].name !== nextProps.signals[lastIdx].name) return false;
    }
  }

  // 比较 timeRange
  if (prevProps.timeRange.start_ns !== nextProps.timeRange.start_ns) return false;
  if (prevProps.timeRange.end_ns !== nextProps.timeRange.end_ns) return false;

  // height 和 onFlitClick 不太可能变化，简单比较引用
  if (prevProps.height !== nextProps.height) return false;

  // 比较展开状态数组的辅助函数
  const compareArrays = (prev: string[] | undefined, next: string[] | undefined): boolean => {
    const prevArr = prev || [];
    const nextArr = next || [];
    if (prevArr.length !== nextArr.length) return false;
    for (let i = 0; i < prevArr.length; i++) {
      if (prevArr[i] !== nextArr[i]) return false;
    }
    return true;
  };

  // 比较所有展开状态
  if (!compareArrays(prevProps.expandedRspSignals, nextProps.expandedRspSignals)) return false;
  if (!compareArrays(prevProps.expandedReqSignals, nextProps.expandedReqSignals)) return false;
  if (!compareArrays(prevProps.expandedDataSignals, nextProps.expandedDataSignals)) return false;

  return true;
}

export default memo(FIFOWaveformChart, propsAreEqual);
