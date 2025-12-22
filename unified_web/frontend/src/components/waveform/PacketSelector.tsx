/**
 * 请求选择器组件
 */

import { useState, useEffect, useMemo } from 'react';
import { Table, Tag, Space, Select, Input, Button, message } from 'antd';
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
  const [packetIdInput, setPacketIdInput] = useState('');

  // 多选过滤器
  const [reqTypeFilter, setReqTypeFilter] = useState<string[]>([]);
  const [sourceIpFilter, setSourceIpFilter] = useState<string[]>([]);
  const [destIpFilter, setDestIpFilter] = useState<string[]>([]);

  // 排序
  const [sortField, setSortField] = useState<string>('start_time_ns');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');

  useEffect(() => {
    loadPackets();
  }, [experimentId, resultId, page, pageSize]);

  const loadPackets = async () => {
    setLoading(true);
    try {
      const result = await getPacketList(experimentId, resultId, {
        page,
        pageSize: 1000, // 加载更多数据用于前端过滤
        sortBy: sortField,
        order: sortOrder,
      });
      setPackets(result.packets);
      setTotal(result.total);
    } catch (error) {
      console.error('加载请求列表失败', error);
    } finally {
      setLoading(false);
    }
  };

  // 获取所有唯一的源IP和目标IP选项（节点.IP格式）
  const { sourceIpOptions, destIpOptions } = useMemo(() => {
    const sourceSet = new Set<string>();
    const destSet = new Set<string>();
    packets.forEach(p => {
      sourceSet.add(`${p.source_node}.${p.source_type}`);
      destSet.add(`${p.dest_node}.${p.dest_type}`);
    });
    return {
      sourceIpOptions: Array.from(sourceSet).sort(),
      destIpOptions: Array.from(destSet).sort(),
    };
  }, [packets]);

  // 前端过滤数据
  const filteredPackets = useMemo(() => {
    let filtered = packets;

    // 类型过滤
    if (reqTypeFilter.length > 0) {
      filtered = filtered.filter(p => reqTypeFilter.includes(p.req_type));
    }

    // 源IP过滤（匹配"节点.IP"格式）
    if (sourceIpFilter.length > 0) {
      filtered = filtered.filter(p =>
        sourceIpFilter.includes(`${p.source_node}.${p.source_type}`)
      );
    }

    // 目标IP过滤（匹配"节点.IP"格式）
    if (destIpFilter.length > 0) {
      filtered = filtered.filter(p =>
        destIpFilter.includes(`${p.dest_node}.${p.dest_type}`)
      );
    }

    return filtered;
  }, [packets, reqTypeFilter, sourceIpFilter, destIpFilter]);

  const columns: ColumnsType<PacketInfo> = [
    {
      title: 'ID',
      dataIndex: 'packet_id',
      key: 'packet_id',
      width: 80,
      align: 'center',
      sorter: (a, b) => a.packet_id - b.packet_id,
    },
    {
      title: '类型',
      dataIndex: 'req_type',
      key: 'req_type',
      width: 80,
      align: 'center',
      render: (type: string) => (
        <Tag color={type === 'read' ? 'blue' : 'green'}>
          {type === 'read' ? '读' : '写'}
        </Tag>
      ),
      sorter: (a, b) => a.req_type.localeCompare(b.req_type),
    },
    {
      title: '源IP',
      key: 'source',
      width: 150,
      align: 'center',
      render: (_: unknown, record: PacketInfo) => (
        <span>{record.source_node}.{record.source_type}</span>
      ),
      sorter: (a, b) => a.source_node - b.source_node || a.source_type.localeCompare(b.source_type),
    },
    {
      title: '目标IP',
      key: 'dest',
      width: 150,
      align: 'center',
      render: (_: unknown, record: PacketInfo) => (
        <span>{record.dest_node}.{record.dest_type}</span>
      ),
      sorter: (a, b) => a.dest_node - b.dest_node || a.dest_type.localeCompare(b.dest_type),
    },
    {
      title: '命令延迟 (ns)',
      dataIndex: 'cmd_latency_ns',
      key: 'cmd_latency_ns',
      width: 110,
      align: 'center',
      render: (val?: number | null) => val != null ? (Number.isInteger(val * 2) ? val.toFixed(1) : val.toFixed(2)) : '-',
      sorter: (a, b) => (a.cmd_latency_ns || 0) - (b.cmd_latency_ns || 0),
    },
    {
      title: '数据延迟 (ns)',
      dataIndex: 'data_latency_ns',
      key: 'data_latency_ns',
      width: 110,
      align: 'center',
      render: (val?: number | null) => val != null ? (Number.isInteger(val * 2) ? val.toFixed(1) : val.toFixed(2)) : '-',
      sorter: (a, b) => (a.data_latency_ns || 0) - (b.data_latency_ns || 0),
    },
    {
      title: '事务延迟 (ns)',
      dataIndex: 'transaction_latency_ns',
      key: 'transaction_latency_ns',
      width: 110,
      align: 'center',
      render: (val?: number | null) => val != null ? (Number.isInteger(val * 2) ? val.toFixed(1) : val.toFixed(2)) : '-',
      sorter: (a, b) => (a.transaction_latency_ns || 0) - (b.transaction_latency_ns || 0),
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

  const handleApplyPacketIds = () => {
    if (!packetIdInput.trim()) {
      message.warning('请输入请求ID');
      return;
    }

    try {
      const ids = packetIdInput
        .split(',')
        .map(s => s.trim())
        .filter(s => s !== '')
        .map(s => {
          const num = parseInt(s, 10);
          if (isNaN(num)) {
            throw new Error(`无效的ID: ${s}`);
          }
          return num;
        });

      if (ids.length === 0) {
        message.warning('请输入有效的请求ID');
        return;
      }

      if (ids.length > maxPackets) {
        message.warning(`最多只能选择 ${maxPackets} 个请求`);
        return;
      }

      onSelectionChange(ids);
      message.success(`已选择 ${ids.length} 个请求`);
    } catch (error) {
      message.error(error instanceof Error ? error.message : '输入格式错误');
    }
  };

  const handleClearSelection = () => {
    onSelectionChange([]);
    setPacketIdInput('');
  };

  return (
    <div>
      <Space style={{ marginBottom: 12 }} wrap>
        <span>指定请求ID:</span>
        <Input
          placeholder="输入请求ID，多个用逗号分隔，如: 1,5,10"
          value={packetIdInput}
          onChange={(e) => setPacketIdInput(e.target.value)}
          onPressEnter={handleApplyPacketIds}
          size="small"
          style={{ width: 300 }}
        />
        <Button size="small" type="primary" onClick={handleApplyPacketIds}>
          应用
        </Button>
        <Button size="small" onClick={handleClearSelection}>
          清空
        </Button>

        <span style={{ marginLeft: 16, color: '#999' }}>
          已选: {selectedPacketIds.length} / {maxPackets}
        </span>
      </Space>

      <Space style={{ marginBottom: 12,marginLeft: 50}} wrap>
        <span>类型:</span>
        <Select
          mode="multiple"
          placeholder="全部"
          value={reqTypeFilter}
          onChange={setReqTypeFilter}
          size="small"
          style={{ minWidth: 150 }}
          options={[
            { label: '读', value: 'read' },
            { label: '写', value: 'write' },
          ]}
        />

        <span style={{ marginLeft: 24 }}>源IP:</span>
        <Select
          mode="multiple"
          placeholder="全部"
          value={sourceIpFilter}
          onChange={setSourceIpFilter}
          size="small"
          style={{ minWidth: 200 }}
          options={sourceIpOptions.map(ip => ({ label: ip, value: ip }))}
        />

        <span style={{ marginLeft: 24 }}>目标IP:</span>
        <Select
          mode="multiple"
          placeholder="全部"
          value={destIpFilter}
          onChange={setDestIpFilter}
          size="small"
          style={{ minWidth: 200 }}
          options={destIpOptions.map(ip => ({ label: ip, value: ip }))}
        />
      </Space>

      <Table
        rowKey="packet_id"
        columns={columns}
        dataSource={filteredPackets}
        loading={loading}
        rowSelection={rowSelection}
        size="small"
        scroll={{ y: 300, x: 'max-content' }}
        pagination={{
          pageSize: 50,
          showSizeChanger: true,
          pageSizeOptions: ['20', '50', '100', '200'],
          showTotal: (t) => `共 ${t} 条`,
        }}
      />
    </div>
  );
}
