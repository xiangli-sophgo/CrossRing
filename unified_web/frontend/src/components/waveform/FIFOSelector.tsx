/**
 * FIFO选择器组件 - 支持多选FIFO类型
 */

import { useState, useEffect } from 'react';
import { Checkbox, Space, Button, Divider } from 'antd';
import type { CheckboxProps } from 'antd';
import { getAvailableFIFOs } from '@/api/fifoWaveform';

type CheckboxValueType = NonNullable<CheckboxProps['value']>;

interface Props {
  experimentId: number;
  resultId: number;
  nodeId: number;
  selectedFifos: string[];
  onSelectionChange: (fifos: string[]) => void;
}

// FIFO分组定义
const FIFO_GROUPS = {
  IQ: ['IQ_TR', 'IQ_TL', 'IQ_TU', 'IQ_TD', 'IQ_EQ', 'IQ_CH'],
  RB: ['RB_TR', 'RB_TL', 'RB_TU', 'RB_TD', 'RB_EQ'],
  EQ: ['EQ_TU', 'EQ_TD', 'EQ_CH'],
};

const FIFO_LABELS: Record<string, string> = {
  IQ_TR: '右侧',
  IQ_TL: '左侧',
  IQ_TU: '上方',
  IQ_TD: '下方',
  IQ_EQ: '弹出',
  IQ_CH: '通道',
  RB_TR: '右侧',
  RB_TL: '左侧',
  RB_TU: '上方',
  RB_TD: '下方',
  RB_EQ: '弹出',
  EQ_TU: '上方',
  EQ_TD: '下方',
  EQ_CH: '通道',
};

export default function FIFOSelector({
  experimentId,
  resultId,
  nodeId,
  selectedFifos,
  onSelectionChange,
}: Props) {
  const [availableFifos, setAvailableFifos] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadAvailableFifos();
  }, [experimentId, resultId, nodeId]);

  const loadAvailableFifos = async () => {
    setLoading(true);
    try {
      const fifos = await getAvailableFIFOs(experimentId, resultId, nodeId);
      setAvailableFifos(fifos);
    } catch (error) {
      console.error('加载可用FIFO列表失败', error);
      // 使用默认列表
      setAvailableFifos([...FIFO_GROUPS.IQ, ...FIFO_GROUPS.RB, ...FIFO_GROUPS.EQ]);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (checkedValues: CheckboxValueType[]) => {
    onSelectionChange(checkedValues as string[]);
  };

  const handleSelectAll = () => {
    onSelectionChange(availableFifos);
  };

  const handleClearAll = () => {
    onSelectionChange([]);
  };

  const handleSelectGroup = (group: keyof typeof FIFO_GROUPS) => {
    const groupFifos = FIFO_GROUPS[group].filter(f => availableFifos.includes(f));
    const newSelection = [...new Set([...selectedFifos, ...groupFifos])];
    onSelectionChange(newSelection);
  };

  const renderGroup = (groupName: keyof typeof FIFO_GROUPS, title: string) => {
    const groupFifos = FIFO_GROUPS[groupName].filter(f => availableFifos.includes(f));
    if (groupFifos.length === 0) return null;

    return (
      <div key={groupName} style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
          <strong style={{ marginRight: 8 }}>{title}</strong>
          <Button
            size="small"
            type="link"
            onClick={() => handleSelectGroup(groupName)}
            style={{ padding: 0, height: 'auto' }}
          >
            全选
          </Button>
        </div>
        <Checkbox.Group
          value={selectedFifos}
          onChange={handleChange}
          style={{ display: 'flex', flexDirection: 'column', gap: 4 }}
        >
          {groupFifos.map(fifo => (
            <Checkbox key={fifo} value={fifo}>
              {fifo} ({FIFO_LABELS[fifo]})
            </Checkbox>
          ))}
        </Checkbox.Group>
      </div>
    );
  };

  return (
    <div style={{ padding: '12px', backgroundColor: '#fafafa', borderRadius: 4 }}>
      <div style={{ marginBottom: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <strong>选择FIFO ({selectedFifos.length} / {availableFifos.length})</strong>
        <Space size="small">
          <Button size="small" onClick={handleSelectAll} disabled={loading}>
            全选
          </Button>
          <Button size="small" onClick={handleClearAll} disabled={loading}>
            清空
          </Button>
        </Space>
      </div>

      <Divider style={{ margin: '8px 0' }} />

      {loading ? (
        <div style={{ textAlign: 'center', padding: 20 }}>加载中...</div>
      ) : (
        <>
          {renderGroup('IQ', 'Input Queue (IQ)')}
          {renderGroup('RB', 'Ring Bridge (RB)')}
          {renderGroup('EQ', 'Eject Queue (EQ)')}
        </>
      )}
    </div>
  );
}
