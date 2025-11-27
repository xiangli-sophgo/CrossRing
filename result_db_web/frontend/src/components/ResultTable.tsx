/**
 * 结果表格组件 - 使用Handsontable实现Excel-like功能
 */

import { useState, useEffect, useMemo, useRef } from 'react';
import { Button, message, Space, Tree, Popover, Pagination, Divider, Checkbox, Popconfirm } from 'antd';
import { DownloadOutlined, SettingOutlined, DeleteOutlined } from '@ant-design/icons';
import type { DataNode } from 'antd/es/tree';
import { HotTable, HotTableClass } from '@handsontable/react';
import { registerAllModules } from 'handsontable/registry';
import 'handsontable/dist/handsontable.full.min.css';
import type { ResultsPageResponse, ExperimentType } from '../types';
import ResultDetailPanel from './ResultDetailPanel';
import { classifyParamKeys, PARAM_CATEGORIES } from '../utils/paramClassifier';
import { deleteResult } from '../api';

// 注册所有Handsontable模块
registerAllModules();

const STORAGE_KEY = 'result_table_visible_columns_v3';
const FIXED_COLUMNS_KEY = 'result_table_fixed_columns_v3';
const DEFAULT_CATEGORIES = ['basic', 'bw_result'];

interface Props {
  data: ResultsPageResponse | null;
  loading: boolean;
  page: number;
  pageSize: number;
  paramKeys: string[];
  experimentId: number;
  experimentType?: ExperimentType;
  onPageChange: (page: number, pageSize: number) => void;
  onSortChange: (field: string, order: 'asc' | 'desc') => void;
  onDataChange?: () => void;
}

export default function ResultTable({
  data,
  loading,
  page,
  pageSize,
  paramKeys,
  experimentId,
  experimentType = 'kcin',
  onPageChange,
  onSortChange,
  onDataChange,
}: Props) {
  // 可见列状态
  const [visibleColumns, setVisibleColumns] = useState<string[]>([]);

  // 固定列（具体列名）
  const [fixedColumns, setFixedColumns] = useState<string[]>(() => {
    const saved = localStorage.getItem(FIXED_COLUMNS_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) {
          return parsed;
        }
      } catch {
        // ignore
      }
    }
    return ['performance'];
  });

  // 选中的行索引（用于显示详情）
  const [selectedRowIndex, setSelectedRowIndex] = useState<number | null>(null);

  // 详情面板是否展开
  const [detailExpanded, setDetailExpanded] = useState(true);

  // Handsontable ref
  const hotTableRef = useRef<HotTableClass>(null);

  // 分类数据
  const classifiedParams = useMemo(() => classifyParamKeys(paramKeys), [paramKeys]);

  // 生成树形数据用于列选择器
  const treeData = useMemo((): DataNode[] => {
    const nodes: DataNode[] = [];
    for (const category of PARAM_CATEGORIES) {
      const columns = classifiedParams[category.key] || [];
      if (columns.length === 0) continue;
      nodes.push({
        title: `${category.label} (${columns.length})`,
        key: `category_${category.key}`,
        children: columns.map((col) => ({
          title: col.replace(/_/g, ' '),
          key: col,
        })),
      });
    }
    const uncategorized = classifiedParams['uncategorized'] || [];
    if (uncategorized.length > 0) {
      nodes.push({
        title: `未分类 (${uncategorized.length})`,
        key: 'category_uncategorized',
        children: uncategorized.map((col) => ({
          title: col.replace(/_/g, ' '),
          key: col,
        })),
      });
    }
    return nodes;
  }, [classifiedParams]);

  // 初始化可见列
  useEffect(() => {
    if (paramKeys.length === 0) return;
    const classified = classifyParamKeys(paramKeys);
    const defaultColumns: string[] = [];
    for (const cat of DEFAULT_CATEGORIES) {
      if (classified[cat]) {
        defaultColumns.push(...classified[cat]);
      }
    }
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          const validColumns = parsed.filter((col: string) => paramKeys.includes(col));
          if (validColumns.length > 0) {
            setVisibleColumns(validColumns);
            return;
          }
        }
      } catch {
        // ignore
      }
    }
    setVisibleColumns(defaultColumns);
  }, [paramKeys]);

  // 保存可见列
  useEffect(() => {
    if (visibleColumns.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(visibleColumns));
    }
  }, [visibleColumns]);

  // 保存固定列
  useEffect(() => {
    localStorage.setItem(FIXED_COLUMNS_KEY, JSON.stringify(fixedColumns));
  }, [fixedColumns]);

  // 切换固定列
  const toggleFixedColumn = (col: string) => {
    setFixedColumns((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  // Tree 展开状态
  const [expandedKeys, setExpandedKeys] = useState<React.Key[]>([]);
  const allCategoryKeys = useMemo(() => treeData.map((node) => node.key), [treeData]);
  const expandAll = () => setExpandedKeys(allCategoryKeys);
  const collapseAll = () => setExpandedKeys([]);

  // 处理列选择
  const handleColumnCheck = (checkedKeys: React.Key[]) => {
    const columnKeys = (checkedKeys as string[]).filter((key) => !key.startsWith('category_'));
    setVisibleColumns(columnKeys);
  };

  const checkedKeys = useMemo(() => visibleColumns, [visibleColumns]);

  // 生成列配置（固定列在前）
  const allColumns = useMemo(() => {
    const allCols = ['performance', ...visibleColumns];
    const fixedCols = allCols.filter((col) => fixedColumns.includes(col));
    const nonFixedCols = allCols.filter((col) => !fixedColumns.includes(col));
    return [...fixedCols, ...nonFixedCols];
  }, [visibleColumns, fixedColumns]);

  // 固定列数量
  const fixedColumnCount = useMemo(() => {
    return allColumns.filter((col) => fixedColumns.includes(col)).length;
  }, [allColumns, fixedColumns]);

  // 列头
  const colHeaders = useMemo(() => {
    return allColumns.map((col) => (col === 'performance' ? '性能 (GB/s)' : col.replace(/_/g, ' ')));
  }, [allColumns]);

  // 表格数据
  const tableData = useMemo(() => {
    if (!data?.results) return [];
    return data.results.map((row) => {
      return allColumns.map((col) => {
        if (col === 'performance') {
          return row.performance?.toFixed(2) ?? '-';
        }
        const value = row.config_params?.[col];
        if (value === undefined || value === null) return '-';
        if (typeof value === 'number') {
          return Number.isInteger(value) ? value : value.toFixed(2);
        }
        return value;
      });
    });
  }, [data?.results, allColumns]);

  // 数据变化时强制重新渲染 Handsontable
  useEffect(() => {
    if (hotTableRef.current?.hotInstance) {
      // 延迟执行以确保数据已更新
      setTimeout(() => {
        hotTableRef.current?.hotInstance?.render();
      }, 0);
    }
  }, [tableData, page, pageSize]);

  // 计算字符串显示宽度
  const calcStringWidth = (str: string) => {
    let width = 0;
    for (const char of String(str)) {
      width += /[\u4e00-\u9fa5]/.test(char) ? 14 : 9;
    }
    return width;
  };

  // 列宽计算（考虑表头和数据内容）
  const colWidths = useMemo(() => {
    return allColumns.map((col, colIndex) => {
      const title = col === 'performance' ? '性能 (GB/s)' : col.replace(/_/g, ' ');
      let maxWidth = calcStringWidth(title);

      // 检查数据中的最大宽度（只检查前20行以提高性能）
      const sampleRows = tableData.slice(0, 20);
      for (const row of sampleRows) {
        const cellValue = row[colIndex];
        if (cellValue !== undefined && cellValue !== null) {
          const cellWidth = calcStringWidth(String(cellValue));
          if (cellWidth > maxWidth) {
            maxWidth = cellWidth;
          }
        }
      }

      return Math.max(100, Math.min(300, maxWidth + 30));
    });
  }, [allColumns, tableData]);

  // 排序处理
  const handleAfterColumnSort = (currentSortConfig: { column: number; sortOrder: 'asc' | 'desc' }[]) => {
    if (currentSortConfig.length > 0) {
      const { column, sortOrder } = currentSortConfig[0];
      const field = allColumns[column];
      onSortChange(field, sortOrder);
    }
  };

  // 点击行选择（用于显示详情）
  const handleRowSelect = (row: number) => {
    if (row >= 0 && data?.results[row]) {
      setSelectedRowIndex(row);
      setDetailExpanded(true);
    }
  };

  // 获取选中的结果
  const selectedResult = useMemo(() => {
    if (selectedRowIndex !== null && data?.results[selectedRowIndex]) {
      return data.results[selectedRowIndex];
    }
    return null;
  }, [selectedRowIndex, data?.results]);

  // 删除结果
  const [deleting, setDeleting] = useState(false);
  const handleDeleteResult = async () => {
    if (!selectedResult) return;
    setDeleting(true);
    try {
      await deleteResult(selectedResult.id, experimentId);
      message.success('结果已删除');
      setSelectedRowIndex(null);
      onDataChange?.();
    } catch {
      message.error('删除失败');
    } finally {
      setDeleting(false);
    }
  };

  // 导出 CSV
  const handleExport = () => {
    if (!data || !data.results.length) {
      message.warning('没有数据可导出');
      return;
    }
    const headers = allColumns.map((col) => (col === 'performance' ? '性能' : col));
    const rows = data.results.map((row) =>
      allColumns.map((h) => (h === 'performance' ? row.performance : row.config_params?.[h] ?? '')).join(',')
    );
    const csv = [headers.join(','), ...rows].join('\n');
    const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `results_page${page}.csv`;
    link.click();
    message.success('导出成功');
  };

  // 列选择器
  const columnSelector = (
    <div style={{ maxHeight: 500, overflow: 'auto', minWidth: 300 }}>
      <div style={{ marginBottom: 8 }}>
        <strong>固定列（固定在左侧）</strong>
        <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
          <Checkbox
            checked={fixedColumns.includes('performance')}
            onChange={() => toggleFixedColumn('performance')}
          >
            性能 (GB/s)
          </Checkbox>
          {visibleColumns.slice(0, 10).map((col) => (
            <Checkbox
              key={col}
              checked={fixedColumns.includes(col)}
              onChange={() => toggleFixedColumn(col)}
            >
              {col.replace(/_/g, ' ')}
            </Checkbox>
          ))}
        </div>
      </div>
      <Divider style={{ margin: '12px 0' }} />
      <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <strong>显示列</strong>
        <Space size="small">
          <Button size="small" onClick={expandAll}>全部展开</Button>
          <Button size="small" onClick={collapseAll}>全部折叠</Button>
        </Space>
      </div>
      <Tree
        checkable
        selectable={false}
        treeData={treeData}
        checkedKeys={checkedKeys}
        expandedKeys={expandedKeys}
        onExpand={(keys) => setExpandedKeys(keys)}
        onCheck={(checked) => handleColumnCheck(checked as React.Key[])}
      />
    </div>
  );

  return (
    <div>
      <div
        style={{
          marginBottom: 16,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Space>
          <Popover content={columnSelector} title="列设置" trigger="click" placement="bottomLeft">
            <Button icon={<SettingOutlined />}>
              列设置 ({visibleColumns.length}/{paramKeys.length})
            </Button>
          </Popover>
          <span style={{ color: '#888', fontSize: 12 }}>
            提示：点击行查看详情
          </span>
        </Space>
        <Button icon={<DownloadOutlined />} onClick={handleExport} disabled={!data?.results.length}>
          导出当前页
        </Button>
      </div>

      <div className="result-table-container">
        {loading ? (
          <div style={{ textAlign: 'center', padding: 50, background: '#fafafa' }}>加载中...</div>
        ) : tableData.length === 0 ? (
          <div style={{ textAlign: 'center', padding: 50, background: '#fafafa', color: '#999' }}>暂无数据</div>
        ) : (
          <HotTable
            ref={hotTableRef}
            data={tableData}
            colHeaders={colHeaders}
            colWidths={colWidths}
            rowHeaders={true}
            width="100%"
            height="auto"
            fixedColumnsStart={fixedColumnCount}
            fixedRowsTop={0}
            manualColumnResize={true}
            columnSorting={true}
            licenseKey="non-commercial-and-evaluation"
            stretchH="all"
            selectionMode="multiple"
            outsideClickDeselects={false}
            afterColumnSort={handleAfterColumnSort}
            afterSelection={(row) => {
              handleRowSelect(row);
            }}
            cells={() => ({
              className: 'htCenter htMiddle',
            })}
            className="result-hot-table"
          />
        )}
      </div>

      <div style={{ marginTop: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ color: '#888', fontSize: 12 }}>
          {selectedRowIndex !== null ? `已选择第 ${selectedRowIndex + 1} 行` : '点击行查看详情'}
        </span>
        <Pagination
          current={page}
          pageSize={pageSize}
          total={data?.total || 0}
          showSizeChanger
          showQuickJumper
          pageSizeOptions={['50', '100', '200', '500']}
          showTotal={(total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`}
          onChange={(p, ps) => onPageChange(p, ps)}
        />
      </div>

      {/* 详情面板 */}
      {selectedResult && (
        <div style={{ marginTop: 16, border: '1px solid #d9d9d9', borderRadius: 4 }}>
          <div
            style={{
              padding: '8px 16px',
              background: '#fafafa',
              borderBottom: detailExpanded ? '1px solid #d9d9d9' : 'none',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <strong
              style={{ cursor: 'pointer', flex: 1 }}
              onClick={() => setDetailExpanded(!detailExpanded)}
            >
              实验详情 (第 {selectedRowIndex! + 1} 行, 性能: {selectedResult.performance?.toFixed(2)} GB/s)
            </strong>
            <Space>
              <Popconfirm
                title="确认删除"
                description="确定要删除这条结果吗？此操作不可恢复。"
                onConfirm={handleDeleteResult}
                okText="删除"
                cancelText="取消"
                okButtonProps={{ danger: true }}
              >
                <Button
                  size="small"
                  type="text"
                  danger
                  icon={<DeleteOutlined />}
                  loading={deleting}
                >
                  删除
                </Button>
              </Popconfirm>
              <Button size="small" type="text" onClick={() => setDetailExpanded(!detailExpanded)}>
                {detailExpanded ? '收起' : '展开'}
              </Button>
            </Space>
          </div>
          {detailExpanded && (
            <div style={{ padding: 16 }}>
              <ResultDetailPanel result={selectedResult} experimentId={experimentId} experimentType={experimentType} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
