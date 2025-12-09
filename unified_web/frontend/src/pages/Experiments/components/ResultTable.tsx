/**
 * 结果表格组件 - 使用Handsontable实现Excel-like功能
 */

import { useState, useEffect, useMemo, useRef } from 'react';
import { Button, message, Space, Tree, Popover, Pagination, Checkbox, Popconfirm, Input, Tooltip, Modal, Select, Divider } from 'antd';
import { DownloadOutlined, SettingOutlined, DeleteOutlined, SearchOutlined, HolderOutlined, EditOutlined, SwapOutlined, SaveOutlined, FolderOpenOutlined } from '@ant-design/icons';
import type { DataNode } from 'antd/es/tree';
import { HotTable, HotTableClass } from '@handsontable/react';
import { registerAllModules } from 'handsontable/registry';
import 'handsontable/dist/handsontable.full.min.css';
import type { ResultsPageResponse, ExperimentType } from '../types';
import ResultDetailPanel from './ResultDetailPanel';
import { classifyParamKeysWithHierarchy } from '../utils/paramClassifier';
import { deleteResult, deleteResultsBatch } from '../api';
import apiClient from '../../../api/client';
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

// 按实验类型分别保存列设置
const getStorageKey = (experimentType: ExperimentType) =>
  `result_table_visible_columns_${experimentType}`;
const getFixedColumnsKey = (experimentType: ExperimentType) =>
  `result_table_fixed_columns_${experimentType}`;
const getColumnOrderKey = (experimentType: ExperimentType) =>
  `result_table_column_order_${experimentType}`;
// 按实验ID保存行顺序
const getRowOrderKey = (experimentId: number) =>
  `result_table_row_order_${experimentId}`;

// 列配置方案类型
interface ColumnPreset {
  name: string;
  experimentType: ExperimentType;
  visibleColumns: string[];
  columnOrder: string[];
  fixedColumns: string[];
  createdAt: string;
}

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
  experimentName?: string;
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
  experimentName,
  experimentType = 'kcin',
  onPageChange,
  onSortChange,
  onDataChange,
}: Props) {
  // 可见列状态
  const [visibleColumns, setVisibleColumns] = useState<string[]>([]);

  // 列顺序（用户自定义排序）
  const [columnOrder, setColumnOrder] = useState<string[]>([]);

  // 固定列（具体列名）
  const [fixedColumns, setFixedColumns] = useState<string[]>([]);

  // 选中的行索引（用于显示详情）
  const [selectedRowIndex, setSelectedRowIndex] = useState<number | null>(null);

  // 批量选中的行索引集合
  const [selectedRowIndices, setSelectedRowIndices] = useState<Set<number>>(new Set());

  // 选择模式
  const [selectMode, setSelectMode] = useState(false);

  // 详情面板是否展开
  const [detailExpanded, setDetailExpanded] = useState(true);

  // 编辑模式
  const [editMode, setEditMode] = useState(false);

  // 转置模式
  const [transposed, setTransposed] = useState(false);

  // 行顺序（数据记录的拖拽顺序，存储 result.id）
  const [rowOrder, setRowOrder] = useState<number[]>([]);

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

  // 初始化可见列（按实验类型）
  useEffect(() => {
    if (paramKeys.length === 0) return;
    // 默认显示重要统计
    const defaultColumns = classifiedParams.important || [];
    const saved = localStorage.getItem(getStorageKey(experimentType));
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
  }, [paramKeys, classifiedParams, experimentType]);

  // 初始化列顺序和固定列（按实验类型）
  useEffect(() => {
    // 加载列顺序
    const savedOrder = localStorage.getItem(getColumnOrderKey(experimentType));
    if (savedOrder) {
      try {
        const parsed = JSON.parse(savedOrder);
        if (Array.isArray(parsed)) {
          setColumnOrder(parsed);
        }
      } catch {
        setColumnOrder([]);
      }
    } else {
      setColumnOrder([]);
    }

    // 加载固定列
    const savedFixed = localStorage.getItem(getFixedColumnsKey(experimentType));
    if (savedFixed) {
      try {
        const parsed = JSON.parse(savedFixed);
        if (Array.isArray(parsed)) {
          setFixedColumns(parsed.filter((col: string) => col !== 'performance'));
        }
      } catch {
        setFixedColumns([]);
      }
    } else {
      setFixedColumns([]);
    }
  }, [experimentType]);

  // 保存可见列（按实验类型）
  useEffect(() => {
    if (visibleColumns.length > 0) {
      localStorage.setItem(getStorageKey(experimentType), JSON.stringify(visibleColumns));
    }
  }, [visibleColumns, experimentType]);

  // 保存固定列（按实验类型）
  useEffect(() => {
    localStorage.setItem(getFixedColumnsKey(experimentType), JSON.stringify(fixedColumns));
  }, [fixedColumns, experimentType]);

  // 保存列顺序（按实验类型）
  useEffect(() => {
    if (columnOrder.length > 0) {
      localStorage.setItem(getColumnOrderKey(experimentType), JSON.stringify(columnOrder));
    }
  }, [columnOrder, experimentType]);

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
      // 默认按首字母排序
      return [...visibleColumns].sort((a, b) => a.localeCompare(b, 'zh-CN'));
    }

    // 获取手动排序的列（保持顺序）
    const manualCols = columnOrder.filter((col) => visibleColumns.includes(col));
    // 新列直接添加到末尾（不再按首字母自动插入）
    const newCols = visibleColumns.filter((col) => !columnOrder.includes(col));

    return [...manualCols, ...newCols];
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

  // 当数据/页面变化时清除选中状态
  useEffect(() => {
    setSelectedRowIndices(new Set());
  }, [page, pageSize, data?.results]);

  // 当数据变化时加载或初始化行顺序
  useEffect(() => {
    if (!data?.results || data.results.length === 0) {
      setRowOrder([]);
      return;
    }
    // 尝试从 localStorage 加载保存的顺序
    const saved = localStorage.getItem(getRowOrderKey(experimentId));
    if (saved) {
      try {
        const savedOrder: number[] = JSON.parse(saved);
        // 过滤出当前数据中存在的 id
        const currentIds = new Set(data.results.map((r) => r.id));
        const validOrder = savedOrder.filter((id) => currentIds.has(id));
        // 添加新数据的 id（不在保存顺序中的）
        const newIds = data.results.filter((r) => !savedOrder.includes(r.id)).map((r) => r.id);
        setRowOrder([...validOrder, ...newIds]);
        return;
      } catch {
        // ignore
      }
    }
    // 默认按原始顺序
    setRowOrder(data.results.map((r) => r.id));
  }, [data?.results, experimentId]);

  // 保存行顺序
  useEffect(() => {
    if (rowOrder.length > 0) {
      localStorage.setItem(getRowOrderKey(experimentId), JSON.stringify(rowOrder));
    }
  }, [rowOrder, experimentId]);

  // 根据行顺序排列的数据
  const sortedResults = useMemo(() => {
    if (!data?.results || data.results.length === 0) return [];
    if (rowOrder.length === 0) return data.results;
    // 创建 id -> result 的映射
    const resultMap = new Map(data.results.map((r) => [r.id, r]));
    // 按 rowOrder 排序，过滤掉不存在的
    const sorted = rowOrder.filter((id) => resultMap.has(id)).map((id) => resultMap.get(id)!);
    // 如果有数据不在 rowOrder 中，添加到末尾
    const orderedIds = new Set(rowOrder);
    const remaining = data.results.filter((r) => !orderedIds.has(r.id));
    return [...sorted, ...remaining];
  }, [data?.results, rowOrder]);

  // 表格数据（原始）
  const tableDataRaw = useMemo(() => {
    if (!sortedResults.length) return [];
    return sortedResults.map((row) => {
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
  }, [sortedResults, allColumns]);

  // 转置后的数据和列头
  const { tableData, displayColHeaders } = useMemo(() => {
    if (!transposed) {
      return { tableData: tableDataRaw, displayColHeaders: colHeaders };
    }
    // 转置：行变列，列变行
    // 转置后第一列是原来的列名，后续列是原来的每一行数据
    const transposedData: (string | number)[][] = [];
    for (let colIdx = 0; colIdx < allColumns.length; colIdx++) {
      const row: (string | number)[] = [colHeaders[colIdx]]; // 第一个元素是列名
      for (let rowIdx = 0; rowIdx < tableDataRaw.length; rowIdx++) {
        row.push(tableDataRaw[rowIdx][colIdx]);
      }
      transposedData.push(row);
    }
    // 转置后的列头：第一列是"参数名"，后续是行号
    const newHeaders = ['参数名', ...tableDataRaw.map((_, idx) => `结果${idx + 1}`)];
    return { tableData: transposedData, displayColHeaders: newHeaders };
  }, [transposed, tableDataRaw, colHeaders, allColumns]);

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

  // 列移动处理
  const handleAfterColumnMove = (movedColumns: number[], finalIndex: number) => {
    // 根据移动后的列顺序更新 columnOrder
    const newOrder = [...allColumns];
    // 获取被移动的列名
    const movedColNames = movedColumns.map((idx) => allColumns[idx]);
    // 从原位置移除
    const remaining = newOrder.filter((_, idx) => !movedColumns.includes(idx));
    // 插入到新位置
    remaining.splice(finalIndex, 0, ...movedColNames);
    setColumnOrder(remaining);
  };

  // 行移动处理（非转置模式下）
  const handleBeforeRowMove = (movedRows: number[], finalIndex: number) => {
    if (transposed) return true;
    // 获取当前显示顺序的 id 列表
    const currentIds = sortedResults.map((r) => r.id);
    // 获取被移动的行的 id
    const movedIds = movedRows.map((idx) => currentIds[idx]);
    // 从原位置移除
    const remaining = currentIds.filter((_, idx) => !movedRows.includes(idx));
    // 插入到新位置
    remaining.splice(finalIndex, 0, ...movedIds);
    setRowOrder(remaining);
    // 返回 false 阻止 Handsontable 内部移动，由 React 状态控制
    return false;
  };

  // 双击行选择（用于显示详情）
  const handleRowDoubleClick = (row: number) => {
    if (row >= 0 && sortedResults[row]) {
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
    if (selectedRowIndex !== null && sortedResults[selectedRowIndex]) {
      return sortedResults[selectedRowIndex];
    }
    return null;
  }, [selectedRowIndex, sortedResults]);

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

  // 批量删除结果
  const [batchDeleting, setBatchDeleting] = useState(false);
  const handleBatchDelete = async () => {
    if (selectedRowIndices.size === 0) return;
    setBatchDeleting(true);
    try {
      const resultIds = Array.from(selectedRowIndices).map((idx) => sortedResults[idx].id);
      const result = await deleteResultsBatch(experimentId, resultIds);
      message.success(result.message);
      setSelectedRowIndices(new Set());
      setSelectedRowIndex(null);
      onDataChange?.();
    } catch {
      message.error('批量删除失败');
    } finally {
      setBatchDeleting(false);
    }
  };

  // 切换行选中状态
  const toggleRowSelection = (rowIndex: number) => {
    setSelectedRowIndices((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(rowIndex)) {
        newSet.delete(rowIndex);
      } else {
        newSet.add(rowIndex);
      }
      return newSet;
    });
  };

  // 全选/取消全选当前页
  const toggleSelectAll = () => {
    if (selectedRowIndices.size === sortedResults.length) {
      setSelectedRowIndices(new Set());
    } else {
      setSelectedRowIndices(new Set(sortedResults.map((_, idx) => idx)));
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
  const [columnSettingTab, setColumnSettingTab] = useState<'select' | 'order' | 'preset'>('select');

  // 配置方案相关状态
  const [presets, setPresets] = useState<ColumnPreset[]>([]);
  const [savePresetModalVisible, setSavePresetModalVisible] = useState(false);
  const [newPresetName, setNewPresetName] = useState('');
  const [presetsLoading, setPresetsLoading] = useState(false);

  // 从后端加载配置方案
  const loadPresetsFromServer = async () => {
    try {
      const response = await apiClient.get('/api/column-presets/');
      if (response.data?.presets) {
        setPresets(response.data.presets);
      }
    } catch (error) {
      console.error('加载配置方案失败:', error);
    }
  };

  // 初始加载配置方案
  useEffect(() => {
    loadPresetsFromServer();
  }, []);

  // 保存配置方案到后端
  const savePresetsToServer = async (newPresets: ColumnPreset[]) => {
    try {
      await apiClient.post('/api/column-presets/', {
        version: 1,
        presets: newPresets,
      });
      setPresets(newPresets);
    } catch (error) {
      console.error('保存配置方案失败:', error);
      message.error('保存配置方案失败');
    }
  };

  // 保存当前配置为新方案
  const handleSavePreset = async () => {
    if (!newPresetName.trim()) {
      message.warning('请输入配置名称');
      return;
    }
    const preset: ColumnPreset = {
      name: newPresetName.trim(),
      experimentType,
      visibleColumns: [...visibleColumns],
      columnOrder: [...columnOrder],
      fixedColumns: [...fixedColumns],
      createdAt: new Date().toISOString(),
    };

    setPresetsLoading(true);
    try {
      const response = await apiClient.post('/api/column-presets/add', preset);
      message.success(response.data.message);
      await loadPresetsFromServer();
      setSavePresetModalVisible(false);
      setNewPresetName('');
    } catch (error) {
      message.error('保存配置方案失败');
    } finally {
      setPresetsLoading(false);
    }
  };

  // 加载配置方案
  const handleLoadPreset = (preset: ColumnPreset) => {
    // 过滤出当前 paramKeys 中存在的列
    const validVisible = preset.visibleColumns.filter(col => paramKeys.includes(col));
    const validOrder = preset.columnOrder.filter(col => paramKeys.includes(col));
    const validFixed = preset.fixedColumns.filter(col => paramKeys.includes(col));

    if (validVisible.length === 0) {
      message.warning('该配置方案中的列在当前数据中不存在');
      return;
    }

    setVisibleColumns(validVisible);
    setColumnOrder(validOrder);
    setFixedColumns(validFixed);
    message.success(`已加载配置方案「${preset.name}」`);
  };

  // 删除配置方案
  const handleDeletePreset = async (presetName: string) => {
    try {
      const response = await apiClient.delete(`/api/column-presets/${experimentType}/${encodeURIComponent(presetName)}`);
      message.success(response.data.message);
      await loadPresetsFromServer();
    } catch (error) {
      message.error('删除配置方案失败');
    }
  };

  // 当前实验类型的配置方案
  const currentPresets = useMemo(() => {
    return presets.filter(p => p.experimentType === experimentType);
  }, [presets, experimentType]);

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
        <Button
          type={columnSettingTab === 'preset' ? 'primary' : 'default'}
          size="small"
          onClick={() => setColumnSettingTab('preset')}
        >
          配置方案 ({currentPresets.length})
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
      ) : columnSettingTab === 'order' ? (
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
      ) : (
        <>
          <div style={{ marginBottom: 12 }}>
            <Button
              type="primary"
              icon={<SaveOutlined />}
              size="small"
              onClick={() => setSavePresetModalVisible(true)}
              block
              loading={presetsLoading}
            >
              保存当前配置为方案
            </Button>
          </div>
          <Divider style={{ margin: '12px 0' }}>已保存的方案 ({experimentType.toUpperCase()})</Divider>
          {currentPresets.length === 0 ? (
            <div style={{ color: '#999', textAlign: 'center', padding: '20px 0' }}>
              暂无保存的配置方案
            </div>
          ) : (
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              {currentPresets.map((preset) => (
                <div
                  key={preset.name}
                  style={{
                    padding: '8px 12px',
                    marginBottom: 8,
                    background: '#fafafa',
                    border: '1px solid #d9d9d9',
                    borderRadius: 4,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <div style={{ flex: 1, overflow: 'hidden' }}>
                    <div style={{ fontWeight: 500, marginBottom: 2 }}>{preset.name}</div>
                    <div style={{ fontSize: 11, color: '#888' }}>
                      {preset.visibleColumns.length} 列
                      {preset.fixedColumns.length > 0 && ` · ${preset.fixedColumns.length} 固定`}
                      {' · '}
                      {new Date(preset.createdAt).toLocaleDateString()}
                    </div>
                  </div>
                  <Space size="small">
                    <Tooltip title="加载此配置">
                      <Button
                        type="link"
                        size="small"
                        icon={<FolderOpenOutlined />}
                        onClick={() => handleLoadPreset(preset)}
                      />
                    </Tooltip>
                    <Popconfirm
                      title="确定删除此配置方案？"
                      onConfirm={() => handleDeletePreset(preset.name)}
                      okText="删除"
                      cancelText="取消"
                    >
                      <Tooltip title="删除">
                        <Button
                          type="link"
                          size="small"
                          danger
                          icon={<DeleteOutlined />}
                        />
                      </Tooltip>
                    </Popconfirm>
                  </Space>
                </div>
              ))}
            </div>
          )}
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
            {selectMode ? '选择模式：单击行选中，再次点击取消' : '双击行查看详情'}
          </span>
        </Space>
        <Space>
          <Tooltip title="行列转置，方便对比不同结果的同一参数">
            <Button
              icon={<SwapOutlined />}
              type={transposed ? 'primary' : 'default'}
              onClick={() => setTransposed(!transposed)}
            >
              {transposed ? '还原' : '转置'}
            </Button>
          </Tooltip>
          <Tooltip title="进入编辑模式后可以修改单元格内容">
            <Button
              icon={<EditOutlined />}
              type={editMode ? 'primary' : 'default'}
              onClick={() => setEditMode(!editMode)}
            >
              {editMode ? '退出编辑' : '编辑模式'}
            </Button>
          </Tooltip>
          <Tooltip title="进入选择模式后单击行可选中，用于批量删除">
            <Button
              icon={<DeleteOutlined />}
              type={selectMode ? 'primary' : 'default'}
              danger={selectMode}
              onClick={() => {
                setSelectMode(!selectMode);
                if (selectMode) {
                  setSelectedRowIndices(new Set());
                }
              }}
            >
              {selectMode ? '退出选择' : '选择模式'}
            </Button>
          </Tooltip>
          {selectMode && (
            <>
              <Tooltip title="选中/取消选中当前页所有行">
                <Checkbox
                  checked={selectedRowIndices.size === sortedResults.length && sortedResults.length > 0}
                  indeterminate={selectedRowIndices.size > 0 && selectedRowIndices.size < sortedResults.length}
                  onChange={toggleSelectAll}
                >
                  全选
                </Checkbox>
              </Tooltip>
              <Popconfirm
                title="批量删除"
                description={`确定要删除选中的 ${selectedRowIndices.size} 条结果吗？此操作不可恢复。`}
                onConfirm={handleBatchDelete}
                okText="删除"
                cancelText="取消"
                okButtonProps={{ danger: true }}
                disabled={selectedRowIndices.size === 0}
              >
                <Tooltip title="删除所有选中的结果">
                  <Button
                    danger
                    icon={<DeleteOutlined />}
                    loading={batchDeleting}
                    disabled={selectedRowIndices.size === 0}
                  >
                    删除 ({selectedRowIndices.size})
                  </Button>
                </Tooltip>
              </Popconfirm>
            </>
          )}
          <Tooltip title="将当前页数据导出为CSV文件">
            <Button icon={<DownloadOutlined />} onClick={handleExport} disabled={!data?.results.length}>
              导出当前页
            </Button>
          </Tooltip>
        </Space>
      </div>

      <div className="result-table-container">
        {loading ? (
          <div style={{ textAlign: 'center', padding: 50, background: '#fafafa' }}>加载中...</div>
        ) : tableData.length === 0 ? (
          <div style={{ textAlign: 'center', padding: 50, background: '#fafafa', color: '#999' }}>暂无数据</div>
        ) : (
          <HotTable
            key={`hot-table-${transposed ? 'transposed' : 'normal'}`}
            ref={hotTableRef}
            data={tableData}
            colHeaders={displayColHeaders}
            colWidths={transposed ? undefined : colWidths}
            rowHeaders={true}
            width="100%"
            height="auto"
            fixedColumnsStart={transposed ? 1 : fixedColumnCount}
            fixedRowsTop={transposed ? Math.min(fixedColumnCount, tableData.length) : 0}
            manualColumnResize={true}
            manualColumnMove={!transposed}
            manualRowMove={!transposed}
            columnSorting={transposed ? false : { headerAction: false, indicator: true }}
            contextMenu={transposed ? false : {
              items: {
                copy: { name: '复制' },
                cut: { name: '剪切' },
                sp1: { name: '---------' },
                row_above: { name: '上方插入行' },
                row_below: { name: '下方插入行' },
                col_left: { name: '左侧插入列' },
                col_right: { name: '右侧插入列' },
                sp2: { name: '---------' },
                remove_row: { name: '删除行' },
                remove_col: { name: '删除列' },
                sp3: { name: '---------' },
                alignment: { name: '对齐方式' },
              }
            }}
copyPaste={{
              copyColumnHeaders: true,
              copyColumnHeadersOnly: true,
            }}
            licenseKey="non-commercial-and-evaluation"
            stretchH="all"
            selectionMode="multiple"
            outsideClickDeselects={false}
            afterColumnSort={transposed ? undefined : handleAfterColumnSort}
            afterColumnMove={transposed ? undefined : handleAfterColumnMove}
            beforeRowMove={transposed ? undefined : handleBeforeRowMove}
            afterOnCellMouseDown={(event, coords) => {
              // 选择模式下单击行选中/取消选中
              if (selectMode && coords.row >= 0 && !transposed) {
                toggleRowSelection(coords.row);
                return;
              }
              // 双击事件通过afterOnCellMouseDown的detail判断
              if (event.detail === 2) {
                if (coords.row === -1 && coords.col >= 0 && !transposed) {
                  // 双击列头触发排序切换：升序 -> 降序 -> 取消
                  const hot = hotTableRef.current?.hotInstance;
                  if (hot) {
                    const columnSortingPlugin = hot.getPlugin('columnSorting');
                    const currentSort = columnSortingPlugin.getSortConfig();
                    const currentColSort = currentSort.find((s: { column: number }) => s.column === coords.col);

                    let newOrder: 'asc' | 'desc' | undefined;
                    if (!currentColSort) {
                      newOrder = 'asc';
                    } else if (currentColSort.sortOrder === 'asc') {
                      newOrder = 'desc';
                    } else {
                      newOrder = undefined; // 取消排序
                    }

                    if (newOrder) {
                      columnSortingPlugin.sort({ column: coords.col, sortOrder: newOrder });
                      onSortChange(allColumns[coords.col], newOrder);
                    } else {
                      columnSortingPlugin.clearSort();
                      onSortChange('id', 'asc'); // 恢复默认按id排序
                    }
                  }
                } else if (coords.row >= 0 && !editMode && !transposed) {
                  // 双击数据行显示详情
                  handleRowDoubleClick(coords.row);
                }
              }
            }}
            readOnly={!editMode}
            cells={(row) => ({
              className: `htCenter htMiddle${selectedRowIndices.has(row) ? ' selected-row' : ''}`,
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
              <ResultDetailPanel result={selectedResult} experimentId={experimentId} experimentName={experimentName} experimentType={experimentType} />
            </div>
          )}
        </div>
      )}

      {/* 保存配置方案弹窗 */}
      <Modal
        title="保存列配置方案"
        open={savePresetModalVisible}
        onOk={handleSavePreset}
        onCancel={() => {
          setSavePresetModalVisible(false);
          setNewPresetName('');
        }}
        okText="保存"
        cancelText="取消"
      >
        <div style={{ marginBottom: 16 }}>
          <div style={{ marginBottom: 8 }}>配置名称：</div>
          <Input
            placeholder="请输入配置方案名称"
            value={newPresetName}
            onChange={(e) => setNewPresetName(e.target.value)}
            onPressEnter={handleSavePreset}
          />
        </div>
        <div style={{ color: '#666', fontSize: 12 }}>
          <div>当前配置：</div>
          <div>• 显示列：{visibleColumns.length} 列</div>
          <div>• 固定列：{fixedColumns.length} 列</div>
          <div>• 实验类型：{experimentType.toUpperCase()}</div>
        </div>
        {currentPresets.some(p => p.name === newPresetName.trim()) && (
          <div style={{ color: '#faad14', fontSize: 12, marginTop: 8 }}>
            ⚠ 已存在同名配置，保存将覆盖原有配置
          </div>
        )}
      </Modal>
    </div>
  );
}
