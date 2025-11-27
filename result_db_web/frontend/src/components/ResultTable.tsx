/**
 * 结果表格组件 - 使用Handsontable实现Excel-like功能
 */

import { useState, useEffect, useMemo, useRef } from 'react';
import { Button, message, Space, Tree, Popover, Pagination, Divider, Checkbox, Popconfirm, Input } from 'antd';
import { DownloadOutlined, SettingOutlined, DeleteOutlined, SearchOutlined, HolderOutlined } from '@ant-design/icons';
import type { DataNode } from 'antd/es/tree';
import { HotTable, HotTableClass } from '@handsontable/react';
import { registerAllModules } from 'handsontable/registry';
import 'handsontable/dist/handsontable.full.min.css';
import type { ResultsPageResponse, ExperimentType } from '../types';
import ResultDetailPanel from './ResultDetailPanel';
import { classifyParamKeysWithHierarchy } from '../utils/paramClassifier';
import { deleteResult } from '../api';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

// 注册所有Handsontable模块
registerAllModules();

const STORAGE_KEY = 'result_table_visible_columns_v4';
const FIXED_COLUMNS_KEY = 'result_table_fixed_columns_v4';
const COLUMN_ORDER_KEY = 'result_table_column_order_v1';

// 可拖拽的列项组件
interface SortableColumnItemProps {
  id: string;
  isFixed: boolean;
  onToggleFixed: (col: string) => void;
}

function SortableColumnItem({ id, isFixed, onToggleFixed }: SortableColumnItemProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id });

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    display: 'flex',
    alignItems: 'center',
    padding: '4px 8px',
    marginBottom: 4,
    background: isDragging ? '#e6f7ff' : '#fafafa',
    border: '1px solid #d9d9d9',
    borderRadius: 4,
    cursor: 'grab',
    opacity: isDragging ? 0.8 : 1,
  };

  return (
    <div ref={setNodeRef} style={style} {...attributes}>
      <HolderOutlined {...listeners} style={{ marginRight: 8, color: '#999', cursor: 'grab' }} />
      <Checkbox
        checked={isFixed}
        onChange={() => onToggleFixed(id)}
        style={{ marginRight: 8 }}
      />
      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {id.replace(/_/g, ' ')}
      </span>
    </div>
  );
}

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

  // 列顺序（用户自定义排序）
  const [columnOrder, setColumnOrder] = useState<string[]>(() => {
    const saved = localStorage.getItem(COLUMN_ORDER_KEY);
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
    return [];
  });

  // 固定列（具体列名）
  const [fixedColumns, setFixedColumns] = useState<string[]>(() => {
    const saved = localStorage.getItem(FIXED_COLUMNS_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) {
          // 过滤掉已删除的performance列
          return parsed.filter((col: string) => col !== 'performance');
        }
      } catch {
        // ignore
      }
    }
    return [];
  });

  // 选中的行索引（用于显示详情）
  const [selectedRowIndex, setSelectedRowIndex] = useState<number | null>(null);

  // 详情面板是否展开
  const [detailExpanded, setDetailExpanded] = useState(true);

  // Handsontable ref
  const hotTableRef = useRef<HotTableClass>(null);

  // 分类数据（带层级结构）
  const classifiedParams = useMemo(() => classifyParamKeysWithHierarchy(paramKeys), [paramKeys]);

  // 生成树形数据用于列选择器（支持三层嵌套）
  const treeData = useMemo((): DataNode[] => {
    const nodes: DataNode[] = [];
    const classified = classifiedParams;

    // 重要统计（一级）
    if (classified.important.length > 0) {
      nodes.push({
        title: `重要统计 (${classified.important.length})`,
        key: 'category_important',
        children: classified.important.map((col) => ({
          title: col.replace(/_/g, ' '),
          key: col,
        })),
      });
    }

    // Die分组（二级嵌套）
    for (let i = 0; i <= 9; i++) {
      const dieData = classified[`die${i}`];
      if (!dieData) continue;
      const totalCount = dieData.important.length + dieData.result.length + dieData.config.length;
      if (totalCount === 0) continue;

      const dieChildren: DataNode[] = [];

      // Die内部的重要统计
      if (dieData.important.length > 0) {
        dieChildren.push({
          title: `重要统计 (${dieData.important.length})`,
          key: `category_die${i}_important`,
          children: dieData.important.map((col) => ({
            title: col.replace(/^Die\d+_/, '').replace(/_/g, ' '),
            key: col,
          })),
        });
      }

      // Die内部的结果统计
      if (dieData.result.length > 0) {
        dieChildren.push({
          title: `结果统计 (${dieData.result.length})`,
          key: `category_die${i}_result`,
          children: dieData.result.map((col) => ({
            title: col.replace(/^Die\d+_/, '').replace(/_/g, ' '),
            key: col,
          })),
        });
      }

      // Die内部的配置参数
      if (dieData.config.length > 0) {
        dieChildren.push({
          title: `配置参数 (${dieData.config.length})`,
          key: `category_die${i}_config`,
          children: dieData.config.map((col) => ({
            title: col.replace(/^Die\d+_/, '').replace(/_/g, ' '),
            key: col,
          })),
        });
      }

      nodes.push({
        title: `Die${i}统计 (${totalCount})`,
        key: `category_die${i}`,
        children: dieChildren,
      });
    }

    // 结果统计（一级）
    if (classified.result.length > 0) {
      nodes.push({
        title: `结果统计 (${classified.result.length})`,
        key: 'category_result',
        children: classified.result.map((col) => ({
          title: col.replace(/_/g, ' '),
          key: col,
        })),
      });
    }

    // 配置参数（一级）
    if (classified.config.length > 0) {
      nodes.push({
        title: `配置参数 (${classified.config.length})`,
        key: 'category_config',
        children: classified.config.map((col) => ({
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
    // 默认显示重要统计
    const defaultColumns = classifiedParams.important || [];
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
  }, [paramKeys, classifiedParams]);

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

  // 保存列顺序
  useEffect(() => {
    if (columnOrder.length > 0) {
      localStorage.setItem(COLUMN_ORDER_KEY, JSON.stringify(columnOrder));
    }
  }, [columnOrder]);

  // 切换固定列
  const toggleFixedColumn = (col: string) => {
    setFixedColumns((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  // 拖拽传感器
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 5,
      },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  // 拖拽结束处理
  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      const oldIndex = orderedVisibleColumns.indexOf(active.id as string);
      const newIndex = orderedVisibleColumns.indexOf(over.id as string);
      const newOrder = arrayMove(orderedVisibleColumns, oldIndex, newIndex);
      setColumnOrder(newOrder);
    }
  };

  // Tree 展开状态
  const [expandedKeys, setExpandedKeys] = useState<React.Key[]>([]);
  const allCategoryKeys = useMemo(() => treeData.map((node) => node.key), [treeData]);
  const expandAll = () => setExpandedKeys(allCategoryKeys);
  const collapseAll = () => setExpandedKeys([]);

  // 搜索状态
  const [searchValue, setSearchValue] = useState<string>('');

  // 过滤后的树数据（支持三层嵌套）
  const filteredTreeData = useMemo((): DataNode[] => {
    if (!searchValue.trim()) {
      return treeData;
    }
    const searchLower = searchValue.toLowerCase();

    // 递归过滤函数
    const filterNode = (node: DataNode): DataNode | null => {
      // 如果是叶子节点（没有children或children为空）
      if (!node.children || node.children.length === 0) {
        const matches =
          String(node.title).toLowerCase().includes(searchLower) ||
          String(node.key).toLowerCase().includes(searchLower);
        return matches ? node : null;
      }

      // 递归过滤子节点
      const filteredChildren: DataNode[] = [];
      for (const child of node.children) {
        const filtered = filterNode(child);
        if (filtered) {
          filteredChildren.push(filtered);
        }
      }

      if (filteredChildren.length > 0) {
        // 计算总数（叶子节点数量）
        const countLeaves = (nodes: DataNode[]): number => {
          return nodes.reduce((sum, n) => {
            if (!n.children || n.children.length === 0) return sum + 1;
            return sum + countLeaves(n.children);
          }, 0);
        };
        const leafCount = countLeaves(filteredChildren);
        return {
          ...node,
          title: `${String(node.title).split(' (')[0]} (${leafCount})`,
          children: filteredChildren,
        };
      }

      return null;
    };

    const filtered: DataNode[] = [];
    for (const category of treeData) {
      const result = filterNode(category);
      if (result) {
        filtered.push(result);
      }
    }
    return filtered;
  }, [treeData, searchValue]);

  // 处理列选择
  const handleColumnCheck = (checkedKeys: React.Key[]) => {
    const columnKeys = (checkedKeys as string[]).filter((key) => !key.startsWith('category_'));
    setVisibleColumns(columnKeys);
  };

  const checkedKeys = useMemo(() => visibleColumns, [visibleColumns]);

  // 根据用户自定义顺序排列可见列
  const orderedVisibleColumns = useMemo(() => {
    if (columnOrder.length === 0) {
      return visibleColumns;
    }
    // 按照columnOrder中的顺序排列，新增的列放在最后
    const ordered: string[] = [];
    for (const col of columnOrder) {
      if (visibleColumns.includes(col)) {
        ordered.push(col);
      }
    }
    // 添加不在columnOrder中的新列
    for (const col of visibleColumns) {
      if (!ordered.includes(col)) {
        ordered.push(col);
      }
    }
    return ordered;
  }, [visibleColumns, columnOrder]);

  // 生成列配置（固定列在前）
  const allColumns = useMemo(() => {
    const fixedCols = orderedVisibleColumns.filter((col) => fixedColumns.includes(col));
    const nonFixedCols = orderedVisibleColumns.filter((col) => !fixedColumns.includes(col));
    return [...fixedCols, ...nonFixedCols];
  }, [orderedVisibleColumns, fixedColumns]);

  // 固定列数量
  const fixedColumnCount = useMemo(() => {
    return allColumns.filter((col) => fixedColumns.includes(col)).length;
  }, [allColumns, fixedColumns]);

  // 列头
  const colHeaders = useMemo(() => {
    return allColumns.map((col) => col.replace(/_/g, ' '));
  }, [allColumns]);

  // 表格数据
  const tableData = useMemo(() => {
    if (!data?.results) return [];
    return data.results.map((row) => {
      return allColumns.map((col) => {
        const value = row.config_params?.[col];
        if (value === undefined || value === null) return '-';
        if (typeof value === 'number') {
          // 比例类数据显示为百分比
          if ((col.includes('比例') || col.toLowerCase().includes('ratio')) && value >= 0 && value <= 1) {
            return `${(value * 100).toFixed(2)}%`;
          }
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
      const title = col.replace(/_/g, ' ');
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

  // 双击行选择（用于显示详情）
  const handleRowDoubleClick = (row: number) => {
    if (row >= 0 && data?.results[row]) {
      setSelectedRowIndex(row);
      setDetailExpanded(true);
    }
  };

  // 容器ref用于检测点击
  const containerRef = useRef<HTMLDivElement>(null);

  // 点击表格空白区域隐藏详情（不包括详情面板）
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (!containerRef.current) return;
      const target = event.target as HTMLElement;
      // 如果点击的是详情面板内部，不隐藏
      const detailPanel = containerRef.current.querySelector('.detail-panel');
      if (detailPanel?.contains(target)) return;
      // 如果点击的是表格内部，不隐藏（由双击控制）
      const hotTable = containerRef.current.querySelector('.result-hot-table');
      if (hotTable?.contains(target)) return;
      // 如果点击的是分页器或列设置，不隐藏
      const pagination = containerRef.current.querySelector('.ant-pagination');
      if (pagination?.contains(target)) return;
      // 如果点击的是弹出层（Popover、Popconfirm等），不隐藏
      const popover = document.querySelector('.ant-popover');
      if (popover?.contains(target)) return;
      const popconfirm = document.querySelector('.ant-popconfirm');
      if (popconfirm?.contains(target)) return;
      // 点击其他空白区域，隐藏详情
      setSelectedRowIndex(null);
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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
    const headers = allColumns;
    const rows = data.results.map((row) =>
      allColumns.map((h) => row.config_params?.[h] ?? '').join(',')
    );
    const csv = [headers.join(','), ...rows].join('\n');
    const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `results_page${page}.csv`;
    link.click();
    message.success('导出成功');
  };

  // 列设置标签页状态
  const [columnSettingTab, setColumnSettingTab] = useState<'select' | 'order'>('select');

  // 列选择器
  const columnSelector = (
    <div style={{ maxHeight: 500, overflow: 'auto', minWidth: 360 }}>
      {/* 标签页切换 */}
      <div style={{ marginBottom: 12, display: 'flex', gap: 8 }}>
        <Button
          type={columnSettingTab === 'select' ? 'primary' : 'default'}
          size="small"
          onClick={() => setColumnSettingTab('select')}
        >
          选择列
        </Button>
        <Button
          type={columnSettingTab === 'order' ? 'primary' : 'default'}
          size="small"
          onClick={() => setColumnSettingTab('order')}
        >
          排序列 ({orderedVisibleColumns.length})
        </Button>
      </div>

      {columnSettingTab === 'select' ? (
        <>
          <div style={{ marginBottom: 8 }}>
            <Input
              placeholder="搜索列名..."
              prefix={<SearchOutlined />}
              value={searchValue}
              onChange={(e) => setSearchValue(e.target.value)}
              allowClear
              size="small"
            />
          </div>
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
            treeData={filteredTreeData}
            checkedKeys={checkedKeys}
            expandedKeys={searchValue ? filteredTreeData.map((n) => n.key) : expandedKeys}
            onExpand={(keys) => setExpandedKeys(keys)}
            onCheck={(checked) => handleColumnCheck(checked as React.Key[])}
          />
        </>
      ) : (
        <>
          <div style={{ marginBottom: 8, color: '#666', fontSize: 12 }}>
            拖拽调整列顺序，勾选复选框固定列到左侧
          </div>
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext
              items={orderedVisibleColumns}
              strategy={verticalListSortingStrategy}
            >
              <div style={{ maxHeight: 400, overflow: 'auto' }}>
                {orderedVisibleColumns.map((col) => (
                  <SortableColumnItem
                    key={col}
                    id={col}
                    isFixed={fixedColumns.includes(col)}
                    onToggleFixed={toggleFixedColumn}
                  />
                ))}
              </div>
            </SortableContext>
          </DndContext>
        </>
      )}
    </div>
  );

  return (
    <div ref={containerRef}>
      <div
        style={{
          marginBottom: 16,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Space>
          <Popover content={columnSelector} title="列显示设置" trigger="click" placement="bottomLeft">
            <Button icon={<SettingOutlined />}>
              列显示设置 ({visibleColumns.length}/{paramKeys.length})
            </Button>
          </Popover>
          <span style={{ color: '#888', fontSize: 12 }}>
            提示：双击行查看详情，点击空白隐藏
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
            afterOnCellMouseDown={(event, coords) => {
              // 双击事件通过afterOnCellMouseDown的detail判断
              if (event.detail === 2 && coords.row >= 0) {
                handleRowDoubleClick(coords.row);
              }
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
          {/* {selectedRowIndex !== null ? `已选择第 ${selectedRowIndex + 1} 行` : '双击行查看详情'} */}
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
        <div className="detail-panel" style={{ marginTop: 16, border: '1px solid #d9d9d9', borderRadius: 4 }}>
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
              实验详情 (第 {selectedRowIndex! + 1} 行, DDR带宽: {(selectedResult.config_params?.['带宽ddr_mixed'] as number)?.toFixed(2) ?? '-'} GB/s)
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
