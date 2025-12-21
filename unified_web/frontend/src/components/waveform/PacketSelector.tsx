/**
 * 请求选择器组件
 */

import { useState, useEffect } from 'react';
import { Table, Tag, Space, Radio, InputNumber, Button } from 'antd';
import type { ColumnsType, TableRowSelection } from 'antd/es/table/interface';
import { getPacketList, type PacketInfo } from '@/api/waveform';

interface Props {
  experimentId: number;
  resultId: number;
  selectedPacketIds: number[];
  onSelectionChange: (packetIds: number[]) => void;
  maxPackets?: number;
}

export default function PacketSelector({
  experimentId,
  resultId,
  selectedPacketIds,
  onSelectionChange,
  maxPackets = 20,
}: Props) {
  const [loading, setLoading] = useState(false);
  const [packets, setPackets] = useState<PacketInfo[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [reqTypeFilter, setReqTypeFilter] = useState<'all' | 'read' | 'write'>('all');
  const [quickSelectCount, setQuickSelectCount] = useState(10);

  useEffect(() => {
    loadPackets();
  }, [experimentId, resultId, page, pageSize, reqTypeFilter]);

  const loadPackets = async () => {
    setLoading(true);
    try {
      const result = await getPacketList(experimentId, resultId, {
        page,
        pageSize,
        reqType: reqTypeFilter === 'all' ? undefined : reqTypeFilter,
        sortBy: 'start_time_ns',
        order: 'asc',
      });
      setPackets(result.packets);
      setTotal(result.total);
    } catch (error) {
      console.error('加载请求列表失败', error);
    } finally {
      setLoading(false);
    }
  };

  const columns: ColumnsType<PacketInfo> = [
    {
      title: 'ID',
      dataIndex: 'packet_id',
      key: 'packet_id',
      width: 80,
    },
    {
      title: '类型',
      dataIndex: 'req_type',
      key: 'req_type',
      width: 80,
      render: (type: string) => (
        <Tag color={type === 'read' ? 'blue' : 'green'}>
          {type === 'read' ? '读' : '写'}
        </Tag>
      ),
    },
    {
      title: '源节点',
      key: 'source',
      width: 120,
      render: (_: unknown, record: PacketInfo) => (
        <span>{record.source_node} ({record.source_type})</span>
      ),
    },
    {
      title: '目标节点',
      key: 'dest',
      width: 120,
      render: (_: unknown, record: PacketInfo) => (
        <span>{record.dest_node} ({record.dest_type})</span>
      ),
    },
    {
      title: '延迟 (ns)',
      dataIndex: 'latency_ns',
      key: 'latency_ns',
      width: 100,
      render: (val: number) => val.toFixed(2),
    },
  ];

  const rowSelection: TableRowSelection<PacketInfo> = {
    selectedRowKeys: selectedPacketIds,
    onChange: (selectedRowKeys) => {
      const newSelection = selectedRowKeys as number[];
      if (newSelection.length > maxPackets) {
        // 超过最大限制，只保留最新选择的
        onSelectionChange(newSelection.slice(-maxPackets));
      } else {
        onSelectionChange(newSelection);
      }
    },
    getCheckboxProps: (record: PacketInfo) => ({
      disabled: !selectedPacketIds.includes(record.packet_id) && selectedPacketIds.length >= maxPackets,
    }),
  };

  const handleQuickSelect = () => {
    // 快速选择前N个请求
    const ids = packets.slice(0, quickSelectCount).map(p => p.packet_id);
    onSelectionChange(ids);
  };

  const handleClearSelection = () => {
    onSelectionChange([]);
  };

  return (
    <div>
      <Space style={{ marginBottom: 12 }} wrap>
        <span>类型过滤:</span>
        <Radio.Group
          value={reqTypeFilter}
          onChange={(e) => setReqTypeFilter(e.target.value)}
          size="small"
        >
          <Radio.Button value="all">全部</Radio.Button>
          <Radio.Button value="read">读</Radio.Button>
          <Radio.Button value="write">写</Radio.Button>
        </Radio.Group>

        <span style={{ marginLeft: 16 }}>快速选择前</span>
        <InputNumber
          value={quickSelectCount}
          onChange={(val) => setQuickSelectCount(val || 10)}
          min={1}
          max={maxPackets}
          size="small"
          style={{ width: 60 }}
        />
        <span>个</span>
        <Button size="small" onClick={handleQuickSelect}>
          选择
        </Button>
        <Button size="small" onClick={handleClearSelection}>
          清空
        </Button>

        <span style={{ marginLeft: 16, color: '#999' }}>
          已选: {selectedPacketIds.length} / {maxPackets}
        </span>
      </Space>

      <Table
        rowKey="packet_id"
        columns={columns}
        dataSource={packets}
        loading={loading}
        rowSelection={rowSelection}
        size="small"
        scroll={{ y: 300 }}
        pagination={{
          current: page,
          pageSize,
          total,
          showSizeChanger: true,
          pageSizeOptions: ['20', '50', '100'],
          showTotal: (t) => `共 ${t} 条`,
          onChange: (p, ps) => {
            setPage(p);
            setPageSize(ps);
          },
        }}
      />
    </div>
  );
}
