/**
 * NodeArchitectureDiagram - CrossRing节点内部微架构图
 * 显示单个节点的IQ/RB/EQ FIFO结构，支持点击选择
 * 布局：EQ在右上，RB在右下，IQ在左侧
 */

import React, { useMemo } from 'react';

interface NodeArchitectureDiagramProps {
  selectedNode: number | null;
  selectedFifos: string[];
  onFifoSelect: (fifo: string) => void;
  availableFifos?: string[];
  activeIPs?: string[];  // 当前节点实际挂载的IP类型列表
}

interface FIFORectProps {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  fifoId: string;
  isSelected: boolean;
  isAvailable: boolean;
  onClick: (fifoId: string) => void;
  textAngle?: number;
}

const FIFORect: React.FC<FIFORectProps> = ({
  x,
  y,
  width,
  height,
  label,
  fifoId,
  isSelected,
  isAvailable,
  onClick,
  textAngle = 0,
}) => {
  return (
    <g
      onClick={() => isAvailable && onClick(fifoId)}
      style={{ cursor: isAvailable ? 'pointer' : 'not-allowed' }}
    >
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        fill={isAvailable ? (isSelected ? '#3b82f6' : '#e5e7eb') : '#d1d5db'}
        stroke={isSelected ? '#1d4ed8' : '#9ca3af'}
        strokeWidth={isSelected ? 3 : 1}
        rx={4}
      />
      <text
        x={x + width / 2}
        y={y + height / 2}
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize={10}
        fontWeight={isSelected ? 'bold' : 'normal'}
        fill={isSelected ? '#fff' : '#374151'}
        transform={textAngle !== 0 ? `rotate(${textAngle}, ${x + width / 2}, ${y + height / 2})` : undefined}
      >
        {label}
      </text>
    </g>
  );
};

const ModuleBox: React.FC<{
  x: number;
  y: number;
  width: number;
  height: number;
  title: string;
  color: string;
}> = ({ x, y, width, height, title, color }) => {
  return (
    <>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        fill={color}
        fillOpacity={0.2}
        stroke="#000"
        strokeWidth={2}
        rx={8}
      />
      <text
        x={x + width / 2}
        y={y - 8}
        textAnchor="middle"
        fontSize={14}
        fontWeight="bold"
        fill="#000"
      >
        {title}
      </text>
    </>
  );
};

const NodeArchitectureDiagram: React.FC<NodeArchitectureDiagramProps> = ({
  selectedNode,
  selectedFifos,
  onFifoSelect,
  availableFifos,
  activeIPs,
}) => {
  // 根据activeIPs动态生成channels列表
  const channels = useMemo(() => {
    if (!activeIPs || activeIPs.length === 0) {
      // 默认空列表，不显示任何channel
      return [];
    }
    return activeIPs.map(ip => {
      const parts = ip.split('_');
      const typeChar = parts[0].charAt(0).toUpperCase();
      const num = parts[1] || '0';
      return `${typeChar}${num}`;  // gdma_0 -> G0
    });
  }, [activeIPs]);

  // 检查FIFO是否可用
  const isAvailable = (fifoId: string) => {
    if (!availableFifos) return true;
    return availableFifos.includes(fifoId);
  };

  // 检查FIFO是否被选中
  const isSelected = (fifoId: string) => {
    return selectedFifos.includes(fifoId);
  };

  if (selectedNode === null) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <span style={{ color: '#999' }}>请在拓扑图中选择节点</span>
      </div>
    );
  }

  // 当节点没有活跃IP时显示提示
  if (channels.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <span style={{ color: '#999' }}>节点 {selectedNode} 无活跃IP</span>
      </div>
    );
  }

  return (
    <svg
      viewBox="0 0 500 440"
      style={{ width: '100%', height: '100%' }}
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Eject Queue 模块（右上） */}
      <ModuleBox
        x={260}
        y={30}
        width={220}
        height={160}
        title="Eject Queue"
        color="#FFB6C1"
      />

      {/* EQ 通道缓冲（左侧纵向） */}
      {channels.map((ch, i) => (
        <FIFORect
          key={`EQ_CH_${ch}`}
          x={275}
          y={45 + i * 20}
          width={40}
          height={16}
          label={ch}
          fifoId={`EQ_CH_${ch}`}
          isSelected={isSelected(`EQ_CH_${ch}`)}
          isAvailable={isAvailable(`EQ_CH_${ch}`)}
          onClick={onFifoSelect}
        />
      ))}

      {/* EQ TU/TD（右侧纵向） */}
      <FIFORect
        x={415}
        y={45}
        width={50}
        height={35}
        label="TU"
        fifoId="EQ_TU"
        isSelected={isSelected('EQ_TU')}
        isAvailable={isAvailable('EQ_TU')}
        onClick={onFifoSelect}
      />
      <FIFORect
        x={415}
        y={85}
        width={50}
        height={35}
        label="TD"
        fifoId="EQ_TD"
        isSelected={isSelected('EQ_TD')}
        isAvailable={isAvailable('EQ_TD')}
        onClick={onFifoSelect}
      />

      {/* Ring Bridge 模块（右下） */}
      <ModuleBox
        x={260}
        y={220}
        width={220}
        height={180}
        title="Ring Bridge"
        color="#FFFFE0"
      />

      {/* RB EQ（左上角） */}
      <FIFORect
        x={275}
        y={235}
        width={50}
        height={35}
        label="EQ"
        fifoId="RB_EQ"
        isSelected={isSelected('RB_EQ')}
        isAvailable={isAvailable('RB_EQ')}
        onClick={onFifoSelect}
      />

      {/* RB TL/TR（底部水平排列） */}
      <FIFORect
        x={275}
        y={335}
        width={35}
        height={50}
        label="TL"
        fifoId="RB_TL"
        isSelected={isSelected('RB_TL')}
        isAvailable={isAvailable('RB_TL')}
        onClick={onFifoSelect}
      />
      <FIFORect
        x={320}
        y={335}
        width={35}
        height={50}
        label="TR"
        fifoId="RB_TR"
        isSelected={isSelected('RB_TR')}
        isAvailable={isAvailable('RB_TR')}
        onClick={onFifoSelect}
      />

      {/* RB TU/TD（右侧纵向） */}
      <FIFORect
        x={415}
        y={235}
        width={50}
        height={35}
        label="TU"
        fifoId="RB_TU"
        isSelected={isSelected('RB_TU')}
        isAvailable={isAvailable('RB_TU')}
        onClick={onFifoSelect}
      />
      <FIFORect
        x={415}
        y={275}
        width={50}
        height={35}
        label="TD"
        fifoId="RB_TD"
        isSelected={isSelected('RB_TD')}
        isAvailable={isAvailable('RB_TD')}
        onClick={onFifoSelect}
      />

      {/* Inject Queue 模块（左侧） */}
      <ModuleBox
        x={0}
        y={220}
        width={240}
        height={180}
        title="Inject Queue"
        color="#90EE90"
      />

      {/* IQ 通道缓冲（横向排列在顶部） */}
      {channels.map((ch, i) => (
        <FIFORect
          key={`IQ_CH_${ch}`}
          x={15 + i * 24}
          y={235}
          width={20}
          height={40}
          label={ch}
          fifoId={`IQ_CH_${ch}`}
          isSelected={isSelected(`IQ_CH_${ch}`)}
          isAvailable={isAvailable(`IQ_CH_${ch}`)}
          onClick={onFifoSelect}
          textAngle={90}
        />
      ))}

      {/* IQ EQ（右侧上） */}
      <FIFORect
        x={175}
        y={235}
        width={50}
        height={35}
        label="EQ"
        fifoId="IQ_EQ"
        isSelected={isSelected('IQ_EQ')}
        isAvailable={isAvailable('IQ_EQ')}
        onClick={onFifoSelect}
      />

      {/* IQ TU/TD（右侧） */}
      <FIFORect
        x={175}
        y={285}
        width={50}
        height={35}
        label="TU"
        fifoId="IQ_TU"
        isSelected={isSelected('IQ_TU')}
        isAvailable={isAvailable('IQ_TU')}
        onClick={onFifoSelect}
      />
      <FIFORect
        x={175}
        y={325}
        width={50}
        height={35}
        label="TD"
        fifoId="IQ_TD"
        isSelected={isSelected('IQ_TD')}
        isAvailable={isAvailable('IQ_TD')}
        onClick={onFifoSelect}
      />

      {/* IQ TL/TR（底部） */}
      <FIFORect
        x={15}
        y={335}
        width={35}
        height={50}
        label="TL"
        fifoId="IQ_TL"
        isSelected={isSelected('IQ_TL')}
        isAvailable={isAvailable('IQ_TL')}
        onClick={onFifoSelect}
      />
      <FIFORect
        x={60}
        y={335}
        width={35}
        height={50}
        label="TR"
        fifoId="IQ_TR"
        isSelected={isSelected('IQ_TR')}
        isAvailable={isAvailable('IQ_TR')}
        onClick={onFifoSelect}
      />
    </svg>
  );
};

export default NodeArchitectureDiagram;
