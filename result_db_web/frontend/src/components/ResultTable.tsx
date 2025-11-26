/**
 * 结果表格组件
 */

import { Table, Button, message } from 'antd';
import { DownloadOutlined } from '@ant-design/icons';
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table';
import type { SorterResult } from 'antd/es/table/interface';
import type { ResultsPageResponse, SimulationResult } from '../types';
import ResultDetailPanel from './ResultDetailPanel';

interface Props {
  data: ResultsPageResponse | null;
  loading: boolean;
  page: number;
  pageSize: number;
  paramKeys: string[];
  experimentId: number;
  onPageChange: (page: number, pageSize: number) => void;
  onSortChange: (field: string, order: 'asc' | 'desc') => void;
}

export default function ResultTable({
  data,
  loading,
  page,
  pageSize,
  paramKeys,
  experimentId,
  onPageChange,
  onSortChange,
}: Props) {
  // 动态生成列
  const columns: ColumnsType<SimulationResult> = [
    {
      title: '性能',
      dataIndex: 'performance',
      key: 'performance',
      width: 120,
      fixed: 'left',
      sorter: true,
      render: (value: number) => `${value.toFixed(2)} GB/s`,
    },
    ...paramKeys.map((param) => ({
      title: param.replace(/_/g, ' '),
      dataIndex: ['config_params', param],
      key: param,
      width: 120,
      sorter: true,
      render: (value: number | string | undefined) => value ?? '-',
    })),
  ];

  // 处理表格变化
  const handleTableChange = (
    pagination: TablePaginationConfig,
    _filters: Record<string, unknown>,
    sorter: SorterResult<SimulationResult> | SorterResult<SimulationResult>[]
  ) => {
    // 分页变化
    if (pagination.current && pagination.pageSize) {
      onPageChange(pagination.current, pagination.pageSize);
    }

    // 排序变化
    const singleSorter = Array.isArray(sorter) ? sorter[0] : sorter;
    if (singleSorter.field) {
      onSortChange(
        singleSorter.field as string,
        singleSorter.order === 'ascend' ? 'asc' : 'desc'
      );
    }
  };

  // 导出CSV
  const handleExport = () => {
    if (!data || !data.results.length) {
      message.warning('没有数据可导出');
      return;
    }

    // 构建CSV内容
    const headers = ['performance', ...paramKeys];
    const rows = data.results.map((row) =>
      headers.map((h) => {
        if (h === 'performance') return row.performance;
        return row.config_params?.[h] ?? '';
      }).join(',')
    );
    const csv = [headers.join(','), ...rows].join('\n');

    // 下载
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `results_page${page}.csv`;
    link.click();
    message.success('导出成功');
  };

  // 展开行渲染
  const expandedRowRender = (record: SimulationResult) => (
    <ResultDetailPanel result={record} experimentId={experimentId} />
  );

  return (
    <div>
      <div style={{ marginBottom: 16, textAlign: 'right' }}>
        <Button icon={<DownloadOutlined />} onClick={handleExport} disabled={!data?.results.length}>
          导出当前页
        </Button>
      </div>
      <Table
        columns={columns}
        dataSource={data?.results || []}
        rowKey="id"
        loading={loading}
        scroll={{ x: 'max-content', y: 500 }}
        size="small"
        expandable={{
          expandedRowRender,
          rowExpandable: () => true,
        }}
        pagination={{
          current: page,
          pageSize: pageSize,
          total: data?.total || 0,
          showSizeChanger: true,
          showQuickJumper: true,
          pageSizeOptions: ['50', '100', '200', '500'],
          showTotal: (total, range) =>
            `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
        }}
        onChange={handleTableChange}
      />
    </div>
  );
}
