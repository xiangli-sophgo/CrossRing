/**
 * 实验对比视图 - 按数据流对比多个实验的参数数据
 * 支持列选择，类似ResultTable的样式
 */

import { useEffect, useState, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  Button,
  Space,
  Spin,
  message,
  Typography,
  Empty,
  Tree,
  Popover,
  Input,
  Modal,
  Table,
  Tag,
} from 'antd';
import { ArrowLeftOutlined, DownloadOutlined, SettingOutlined, SearchOutlined, PlusOutlined, CloseOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type { DataNode } from 'antd/es/tree';
import { HotTable, HotTableClass } from '@handsontable/react';
import { registerAllModules } from 'handsontable/registry';
import 'handsontable/dist/handsontable.full.min.css';
import { useExperimentStore } from '../stores/experimentStore';
import { compareByTraffic, getExperiments } from '../api';
import type { TrafficCompareData, Experiment } from '../types';
import { classifyParamKeysWithHierarchy } from '../utils/paramClassifier';

// 注册所有Handsontable模块
registerAllModules();

const { Title } = Typography;

// localStorage key
const COMPARE_VISIBLE_COLUMNS_KEY = 'compare_view_visible_columns';

export default function CompareView() {
  const navigate = useNavigate();
  const { selectedExperimentIds, clearSelection, toggleExperimentSelection } = useExperimentStore();

  const [loading, setLoading] = useState(true);
  const [compareData, setCompareData] = useState<TrafficCompareData | null>(null);
  const [visibleColumns, setVisibleColumns] = useState<string[]>([]);
  const [searchValue, setSearchValue] = useState('');
  const [expandedKeys, setExpandedKeys] = useState<React.Key[]>([]);

  // 添加实验弹窗状态
  const [addModalVisible, setAddModalVisible] = useState(false);
  const [allExperiments, setAllExperiments] = useState<Experiment[]>([]);
  const [loadingExperiments, setLoadingExperiments] = useState(false);
  // 弹窗内的临时选中状态
  const [tempSelectedIds, setTempSelectedIds] = useState<number[]>([]);

  // 实验列排序状态：存储实验ID的排序顺序
  const [experimentOrder, setExperimentOrder] = useState<number[]>([]);
  // 当前排序的行索引和排序方向
  const [rowSortState, setRowSortState] = useState<{ rowIndex: number; order: 'asc' | 'desc' } | null>(null);

  const hotTableRef = useRef<HotTableClass>(null);

  // 加载对比数据
  const loadCompareData = async () => {
    if (selectedExperimentIds.length < 2) {
      message.warning('请至少选择2个实验进行对比');
      navigate('/');
      return;
    }

    setLoading(true);
    try {
      const data = await compareByTraffic(selectedExperimentIds);
      setCompareData(data);

      // 初始化实验顺序
      setExperimentOrder(data.experiments.map((exp) => exp.id));
      setRowSortState(null);

      // 初始化可见列
      const saved = localStorage.getItem(COMPARE_VISIBLE_COLUMNS_KEY);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          const validColumns = parsed.filter((col: string) => data.param_keys.includes(col));
          if (validColumns.length > 0) {
            setVisibleColumns(validColumns);
          } else {
            // 默认显示重要列
            setVisibleColumns(getDefaultColumns(data.param_keys));
          }
        } catch {
          setVisibleColumns(getDefaultColumns(data.param_keys));
        }
      } else {
        setVisibleColumns(getDefaultColumns(data.param_keys));
      }
    } catch {
      message.error('加载对比数据失败');
    } finally {
      setLoading(false);
    }
  };

  // 获取默认显示的列（重要统计列）
  const getDefaultColumns = (paramKeys: string[]): string[] => {
    const importantPatterns = ['带宽', 'bandwidth', 'bw', '延迟', 'latency', 'performance'];
    return paramKeys.filter((key) =>
      importantPatterns.some((pattern) => key.toLowerCase().includes(pattern.toLowerCase()))
    ).slice(0, 10);
  };

  useEffect(() => {
    loadCompareData();
  }, [selectedExperimentIds]);

  // 加载所有实验（用于添加实验弹窗）
  const loadAllExperiments = async () => {
    setLoadingExperiments(true);
    try {
      const data = await getExperiments();
      setAllExperiments(data);
    } catch {
      message.error('加载实验列表失败');
    } finally {
      setLoadingExperiments(false);
    }
  };

  // 打开添加实验弹窗
  const handleOpenAddModal = () => {
    setTempSelectedIds([...selectedExperimentIds]);
    setAddModalVisible(true);
    loadAllExperiments();
  };

  // 弹窗内：添加实验到临时选中
  const handleTempAddExperiment = (id: number) => {
    if (!tempSelectedIds.includes(id)) {
      setTempSelectedIds([...tempSelectedIds, id]);
    }
  };

  // 弹窗内：从临时选中移除实验
  const handleTempRemoveExperiment = (id: number) => {
    if (tempSelectedIds.length <= 2) {
      message.warning('至少需要保留2个实验进行对比');
      return;
    }
    setTempSelectedIds(tempSelectedIds.filter((i) => i !== id));
  };

  // 确认弹窗：应用更改
  const handleConfirmModal = () => {
    if (tempSelectedIds.length < 2) {
      message.warning('至少需要选择2个实验进行对比');
      return;
    }
    // 同步到全局 store
    clearSelection();
    tempSelectedIds.forEach((id) => toggleExperimentSelection(id));
    setAddModalVisible(false);
  };

  // 从标签移除实验（直接生效）
  const handleRemoveExperiment = (id: number) => {
    if (selectedExperimentIds.length <= 2) {
      message.warning('至少需要保留2个实验进行对比');
      return;
    }
    toggleExperimentSelection(id);
  };

  // 保存可见列设置
  useEffect(() => {
    if (visibleColumns.length > 0) {
      localStorage.setItem(COMPARE_VISIBLE_COLUMNS_KEY, JSON.stringify(visibleColumns));
    }
  }, [visibleColumns]);

  // 分类参数（复用ResultTable的分类逻辑）
  const classifiedParams = useMemo(() => {
    if (!compareData) return null;
    return classifyParamKeysWithHierarchy(compareData.param_keys);
  }, [compareData]);

  // 生成树形数据用于列选择器
  const treeData = useMemo((): DataNode[] => {
    if (!classifiedParams || !compareData) return [];

    const nodes: DataNode[] = [];

    // 重要统计
    if (classifiedParams.important.length > 0) {
      nodes.push({
        title: `重要统计 (${classifiedParams.important.length})`,
        key: 'category_important',
        children: classifiedParams.important.map((col) => ({
          title: col.replace(/_/g, ' '),
          key: col,
        })),
      });
    }

    // Die分组
    for (let i = 0; i <= 9; i++) {
      const dieData = classifiedParams[`die${i}`];
      if (!dieData) continue;
      const totalCount = dieData.important.length + dieData.result.length + dieData.config.length;
      if (totalCount === 0) continue;

      const dieChildren: DataNode[] = [];

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

    // 结果统计
    if (classifiedParams.result.length > 0) {
      nodes.push({
        title: `结果统计 (${classifiedParams.result.length})`,
        key: 'category_result',
        children: classifiedParams.result.map((col) => ({
          title: col.replace(/_/g, ' '),
          key: col,
        })),
      });
    }

    // 配置参数
    if (classifiedParams.config.length > 0) {
      nodes.push({
        title: `配置参数 (${classifiedParams.config.length})`,
        key: 'category_config',
        children: classifiedParams.config.map((col) => ({
          title: col.replace(/_/g, ' '),
          key: col,
        })),
      });
    }

    return nodes;
  }, [classifiedParams, compareData]);

  // 过滤后的树数据
  const filteredTreeData = useMemo((): DataNode[] => {
    if (!searchValue.trim()) return treeData;

    const searchLower = searchValue.toLowerCase();

    const filterNode = (node: DataNode): DataNode | null => {
      if (!node.children || node.children.length === 0) {
        const matches =
          String(node.title).toLowerCase().includes(searchLower) ||
          String(node.key).toLowerCase().includes(searchLower);
        return matches ? node : null;
      }

      const filteredChildren: DataNode[] = [];
      for (const child of node.children) {
        const filtered = filterNode(child);
        if (filtered) filteredChildren.push(filtered);
      }

      if (filteredChildren.length > 0) {
        return { ...node, children: filteredChildren };
      }
      return null;
    };

    return treeData.map(filterNode).filter(Boolean) as DataNode[];
  }, [treeData, searchValue]);

  // 处理列选择
  const handleColumnCheck = (checkedKeys: React.Key[]) => {
    const columnKeys = (checkedKeys as string[]).filter((key) => !key.startsWith('category_'));
    setVisibleColumns(columnKeys);
  };

  // 按排序顺序获取实验列表
  const sortedExperiments = useMemo(() => {
    if (!compareData) return [];
    if (experimentOrder.length === 0) return compareData.experiments;

    const expMap = new Map(compareData.experiments.map((exp) => [exp.id, exp]));
    return experimentOrder
      .filter((id) => expMap.has(id))
      .map((id) => expMap.get(id)!);
  }, [compareData, experimentOrder]);

  // 双击行头排序实验列
  const handleRowHeaderDoubleClick = (rowIndex: number) => {
    if (!compareData || rowIndex < 0) return;

    const row = compareData.data[rowIndex];
    if (!row) return;

    // 确定排序方向：如果当前已按此行排序，则切换方向；否则默认降序
    let newOrder: 'asc' | 'desc' = 'desc';
    if (rowSortState && rowSortState.rowIndex === rowIndex) {
      newOrder = rowSortState.order === 'desc' ? 'asc' : 'desc';
    }

    // 计算每个实验在该行的第一个参数值（用于排序）
    const expValues: { id: number; value: number }[] = [];
    for (const exp of compareData.experiments) {
      // 取第一个可见列的值作为排序依据
      const firstCol = visibleColumns[0];
      const key = `exp_${exp.id}_${firstCol}`;
      const value = row[key];
      expValues.push({
        id: exp.id,
        value: typeof value === 'number' ? value : 0,
      });
    }

    // 排序
    expValues.sort((a, b) => {
      return newOrder === 'desc' ? b.value - a.value : a.value - b.value;
    });

    setExperimentOrder(expValues.map((e) => e.id));
    setRowSortState({ rowIndex, order: newOrder });
  };

  // 构建嵌套列头（使用排序后的实验顺序）- 只保留实验名称行
  const nestedHeaders = useMemo(() => {
    if (!compareData || visibleColumns.length === 0) return [];

    // 只保留实验名称行（跨越多列）
    const firstRow: (string | { label: string; colspan: number })[] = [''];
    sortedExperiments.forEach((exp) => {
      firstRow.push({
        label: exp.name,
        colspan: visibleColumns.length,
      });
    });

    return [firstRow];
  }, [compareData, visibleColumns, sortedExperiments]);

  // 格式化单元格值
  const formatCellValue = (value: unknown, col: string): string | number | null => {
    if (value === undefined || value === null) return null;
    if (typeof value === 'number') {
      // 比例类数据显示为百分比
      if ((col.includes('比例') || col.toLowerCase().includes('ratio')) && value >= 0 && value <= 1) {
        return `${(value * 100).toFixed(2)}%`;
      }
      return Number.isInteger(value) ? value : Number(value.toFixed(4));
    }
    return value as string;
  };

  // 构建表格数据（使用排序后的实验顺序）- 第一行为参数列名（可选中）
  const tableData = useMemo(() => {
    if (!compareData || visibleColumns.length === 0) return [];

    // 第一行：参数列名（作为数据行，可选中复制）
    const headerRow: (string | number | null)[] = ['数据流'];
    sortedExperiments.forEach(() => {
      visibleColumns.forEach((col) => {
        headerRow.push(col.replace(/_/g, ' '));
      });
    });

    // 数据行
    const dataRows = compareData.data.map((row) => {
      const rowData: (string | number | null)[] = [row.traffic_file as string];

      sortedExperiments.forEach((exp) => {
        visibleColumns.forEach((col) => {
          const key = `exp_${exp.id}_${col}`;
          const value = row[key];
          rowData.push(formatCellValue(value, col));
        });
      });

      return rowData;
    });

    return [headerRow, ...dataRows];
  }, [compareData, visibleColumns, sortedExperiments]);

  // 列宽配置（使用排序后的实验顺序）
  const colWidths = useMemo(() => {
    if (!compareData || visibleColumns.length === 0) return [];

    const widths: number[] = [250]; // 数据流列，加宽显示完整名称

    sortedExperiments.forEach((exp) => {
      // 计算实验名称需要的最小宽度（中文14px，英文9px）
      let nameWidth = 0;
      for (const char of exp.name) {
        nameWidth += /[\u4e00-\u9fa5]/.test(char) ? 14 : 9;
      }
      nameWidth += 20; // padding

      // 每个参数列的最小宽度
      const minColWidth = 100;
      // 实验名称需要的总宽度 / 参数列数 = 每列最小宽度
      const widthPerCol = Math.max(minColWidth, Math.ceil(nameWidth / visibleColumns.length));

      visibleColumns.forEach(() => {
        widths.push(widthPerCol);
      });
    });
    return widths;
  }, [compareData, visibleColumns, sortedExperiments]);

  // 导出CSV
  const handleExport = () => {
    if (!compareData || !tableData.length) {
      message.warning('没有数据可导出');
      return;
    }

    const headers = ['数据流'];
    compareData.experiments.forEach((exp) => {
      visibleColumns.forEach((col) => {
        headers.push(`${exp.name}_${col}`);
      });
    });

    const csvRows = [headers.join(',')];
    tableData.forEach((row) => {
      csvRows.push(row.map((cell) => (cell === null ? '' : cell)).join(','));
    });

    const csv = csvRows.join('\n');
    const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `experiment_compare_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    message.success('导出成功');
  };

  // 添加实验弹窗的表格列
  const experimentColumns: ColumnsType<Experiment> = [
    {
      title: '实验名称',
      dataIndex: 'name',
      key: 'name',
      ellipsis: true,
    },
    {
      title: '结果数',
      key: 'count',
      width: 80,
      render: (_, record) => record.completed_combinations?.toLocaleString() || 0,
    },
    {
      title: '状态',
      key: 'status',
      width: 80,
      render: (_, record) => {
        const isSelected = tempSelectedIds.includes(record.id);
        return isSelected ? <Tag color="blue">已选</Tag> : null;
      },
    },
    {
      title: '操作',
      key: 'action',
      width: 80,
      render: (_, record) => {
        const isSelected = tempSelectedIds.includes(record.id);
        return isSelected ? (
          <Button
            type="link"
            size="small"
            danger
            onClick={() => handleTempRemoveExperiment(record.id)}
          >
            移除
          </Button>
        ) : (
          <Button
            type="link"
            size="small"
            onClick={() => handleTempAddExperiment(record.id)}
          >
            添加
          </Button>
        );
      },
    },
  ];

  // 列选择器
  const columnSelector = (
    <div style={{ maxHeight: 500, overflow: 'auto', minWidth: 360 }}>
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
        <strong>选择要对比的列</strong>
        <Space size="small">
          <Button size="small" onClick={() => setExpandedKeys(treeData.map((n) => n.key))}>
            全部展开
          </Button>
          <Button size="small" onClick={() => setExpandedKeys([])}>
            全部折叠
          </Button>
        </Space>
      </div>
      <Tree
        checkable
        selectable={false}
        treeData={filteredTreeData}
        checkedKeys={visibleColumns}
        expandedKeys={searchValue ? filteredTreeData.map((n) => n.key) : expandedKeys}
        onExpand={(keys) => setExpandedKeys(keys)}
        onCheck={(checked) => handleColumnCheck(checked as React.Key[])}
      />
    </div>
  );

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
      </div>
    );
  }

  if (!compareData || compareData.experiments.length < 2) {
    return (
      <Card>
        <Empty description="请选择至少2个实验进行对比">
          <Button type="primary" onClick={() => navigate('/')}>
            返回列表
          </Button>
        </Empty>
      </Card>
    );
  }

  return (
    <div>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
          <Space>
            <Button
              icon={<ArrowLeftOutlined />}
              onClick={() => {
                clearSelection();
                navigate('/');
              }}
            >
              返回
            </Button>
            <Title level={4} style={{ margin: 0 }}>
              数据流对比 ({compareData.experiments.length} 个实验, {compareData.traffic_files.length} 个数据流)
            </Title>
          </Space>
          <Space>
            <Button icon={<PlusOutlined />} onClick={handleOpenAddModal}>
              添加/移除实验
            </Button>
            <Popover content={columnSelector} title="列显示设置" trigger="click" placement="bottomRight">
              <Button icon={<SettingOutlined />}>
                列显示设置 ({visibleColumns.length}/{compareData.param_keys.length})
              </Button>
            </Popover>
            <Button icon={<DownloadOutlined />} onClick={handleExport}>
              导出CSV
            </Button>
          </Space>
        </div>

        {/* 当前对比的实验标签 */}
        <div style={{ marginBottom: 12 }}>
          <div style={{ marginBottom: 8, color: '#666' }}>当前对比:</div>
          <div>
            {compareData.experiments.map((exp) => (
              <Tag
                key={exp.id}
                closable={selectedExperimentIds.length > 2}
                onClose={() => handleRemoveExperiment(exp.id)}
                style={{ marginBottom: 4 }}
              >
                {exp.name}
              </Tag>
            ))}
          </div>
        </div>

        <div style={{ marginBottom: 8, color: '#888', fontSize: 12 }}>
          提示：拖拽行头可调整数据流顺序，双击列头按该列排序行，双击行头按该行数值排序实验列
          {rowSortState && (
            <span style={{ marginLeft: 8, color: '#1890ff' }}>
              (当前按第 {rowSortState.rowIndex + 1} 行 {rowSortState.order === 'desc' ? '降序' : '升序'} 排列实验)
            </span>
          )}
        </div>

        {visibleColumns.length === 0 ? (
          <Empty description="请在「列显示设置」中选择要对比的列" />
        ) : (
          <div className="result-table-container">
            <HotTable
              ref={hotTableRef}
              data={tableData}
              nestedHeaders={nestedHeaders}
              colWidths={colWidths}
              rowHeaders={true}
              width="100%"
              height="auto"
              fixedColumnsStart={1}
              fixedRowsTop={1}
              manualColumnResize={true}
              manualRowMove={true}
              columnSorting={{ headerAction: false, indicator: true }}
              copyPaste={{
                copyColumnHeaders: true,
                copyColumnHeadersOnly: true,
              }}
              contextMenu={{
                items: {
                  copy: { name: '复制' },
                  copy_with_column_headers: { name: '复制（含表头）' },
                  copy_column_headers_only: { name: '仅复制表头' },
                }
              }}
              licenseKey="non-commercial-and-evaluation"
              stretchH="all"
              readOnly={true}
              cells={(row, col) => ({
                className: 'htCenter htMiddle',
              })}
              afterOnCellMouseDown={(event, coords) => {
                if (event.detail === 2) {
                  // 双击列头：按该列排序行
                  if (coords.row === -1 && coords.col >= 0) {
                    const hot = hotTableRef.current?.hotInstance;
                    if (hot) {
                      const columnSortingPlugin = hot.getPlugin('columnSorting');
                      const currentSort = columnSortingPlugin.getSortConfig();
                      const currentColSort = currentSort.find(
                        (s: { column: number }) => s.column === coords.col
                      );

                      let newOrder: 'asc' | 'desc' | undefined;
                      if (!currentColSort) {
                        newOrder = 'desc';
                      } else if (currentColSort.sortOrder === 'desc') {
                        newOrder = 'asc';
                      } else {
                        newOrder = undefined;
                      }

                      if (newOrder) {
                        columnSortingPlugin.sort({ column: coords.col, sortOrder: newOrder });
                      } else {
                        columnSortingPlugin.clearSort();
                      }
                    }
                  }
                  // 双击行头：按该行数值排序实验列
                  else if (coords.col === -1 && coords.row >= 0) {
                    handleRowHeaderDoubleClick(coords.row);
                  }
                }
              }}
              className="result-hot-table"
            />
          </div>
        )}
      </Card>

      {/* 添加/移除实验弹窗 */}
      <Modal
        title="添加/移除对比实验"
        open={addModalVisible}
        onCancel={() => setAddModalVisible(false)}
        onOk={handleConfirmModal}
        okText="确认"
        cancelText="取消"
        width={600}
      >
        <div style={{ marginBottom: 12, color: '#666' }}>
          已选择 {tempSelectedIds.length} 个实验，点击「添加」或「移除」按钮管理对比实验
        </div>
        <Table
          columns={experimentColumns}
          dataSource={allExperiments}
          rowKey="id"
          loading={loadingExperiments}
          size="small"
          pagination={{ pageSize: 10, showSizeChanger: false }}
        />
      </Modal>
    </div>
  );
}
