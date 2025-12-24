/**
 * 实验列表页
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Input,
  message,
  Typography,
  Tooltip,
  Popconfirm,
  Radio,
  Empty,
} from 'antd';
import {
  UploadOutlined,
  ReloadOutlined,
  DeleteOutlined,
  BarChartOutlined,
  DownloadOutlined,
  EditOutlined,
  SwapOutlined,
  ExperimentOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  SyncOutlined,
  CloseCircleOutlined,
} from '@ant-design/icons';
import { primaryColor, successColor, warningColor, errorColor } from '@/theme/colors';
import type { ColumnsType } from 'antd/es/table';
import { useExperimentStore } from '../../stores/experimentStore';
import { getExperiments, deleteExperiment, updateExperiment, deleteExperimentsBatch, exportExperiment } from './api';
import type { Experiment, ExperimentType } from './types';
import ImportModal from './components/ImportModal';
import ExportExperimentModal from './components/ExportExperimentModal';

const { Title, Text } = Typography;

const statusConfig: Record<string, { color: string; text: string; icon: React.ReactNode }> = {
  running: { color: 'processing', text: '运行中', icon: <SyncOutlined spin /> },
  completed: { color: 'success', text: '已完成', icon: <CheckCircleOutlined /> },
  failed: { color: 'error', text: '失败', icon: <CloseCircleOutlined /> },
  interrupted: { color: 'warning', text: '已中断', icon: <ClockCircleOutlined /> },
  importing: { color: 'processing', text: '导入中', icon: <SyncOutlined spin /> },
};

// 统计卡片组件
interface StatCardProps {
  title: string;
  value: number;
  icon: React.ReactNode;
  color: string;
  bgColor: string;
}

const StatCard: React.FC<StatCardProps> = ({ title, value, icon, color, bgColor }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
    <div
      style={{
        width: 44,
        height: 44,
        borderRadius: 10,
        backgroundColor: bgColor,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: color,
        fontSize: 20,
      }}
    >
      {icon}
    </div>
    <div>
      <Text type="secondary" style={{ fontSize: 12 }}>{title}</Text>
      <div style={{ fontSize: 20, fontWeight: 600, color }}>{value}</div>
    </div>
  </div>
);


export default function ExperimentList() {
  const navigate = useNavigate();
  const {
    experiments,
    setExperiments,
    loading,
    setLoading,
    selectedExperimentIds,
    toggleExperimentSelection,
    clearSelection,
  } = useExperimentStore();
  const [importExpModalVisible, setImportExpModalVisible] = useState(false);
  const [exportExpModalVisible, setExportExpModalVisible] = useState(false);
  const [typeFilter, setTypeFilter] = useState<ExperimentType | 'all'>('all');
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editingField, setEditingField] = useState<'name' | 'description' | null>(null);
  const [editingValue, setEditingValue] = useState<string>('');

  // 加载实验列表
  const loadExperiments = async () => {
    setLoading(true);
    try {
      const data = await getExperiments(
        undefined,
        typeFilter === 'all' ? undefined : typeFilter
      );
      setExperiments(data);
    } catch (error) {
      message.error('加载实验列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadExperiments();
  }, [typeFilter]);

  // 处理删除
  const handleDelete = async (id: number) => {
    try {
      await deleteExperiment(id);
      message.success('删除成功');
      loadExperiments();
    } catch (error) {
      message.error('删除失败');
    }
  };

  // 批量删除
  const [batchDeleting, setBatchDeleting] = useState(false);
  const handleBatchDelete = async () => {
    if (selectedExperimentIds.length === 0) return;
    setBatchDeleting(true);
    try {
      const result = await deleteExperimentsBatch(selectedExperimentIds);
      message.success(result.message);
      clearSelection();
      loadExperiments();
    } catch (error) {
      message.error('批量删除失败');
    } finally {
      setBatchDeleting(false);
    }
  };

  // 处理字段编辑
  const handleStartEdit = (record: Experiment, field: 'name' | 'description') => {
    setEditingId(record.id);
    setEditingField(field);
    setEditingValue(field === 'name' ? record.name : (record.description || ''));
  };

  const handleSaveEdit = async () => {
    if (editingId === null || editingField === null) return;
    try {
      await updateExperiment(editingId, { [editingField]: editingValue });
      message.success(editingField === 'name' ? '名称修改成功' : '描述修改成功');
      handleCancelEdit();
      loadExperiments();
    } catch (error) {
      message.error(editingField === 'name' ? '名称修改失败' : '描述修改失败');
    }
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditingField(null);
    setEditingValue('');
  };

  // 表格列定义
  const columns: ColumnsType<Experiment> = [
    {
      title: '实验名称',
      dataIndex: 'name',
      key: 'name',
      width: 350,
      ellipsis: true,
      align: 'center',
      render: (text, record) => {
        if (editingId === record.id && editingField === 'name') {
          return (
            <Space>
              <Input
                size="small"
                value={editingValue}
                onChange={(e) => setEditingValue(e.target.value)}
                onPressEnter={handleSaveEdit}
                onKeyDown={(e) => e.key === 'Escape' && handleCancelEdit()}
                autoFocus
                style={{ width: 150 }}
              />
              <Button size="small" type="primary" onClick={handleSaveEdit}>
                保存
              </Button>
              <Button size="small" onClick={handleCancelEdit}>
                取消
              </Button>
            </Space>
          );
        }
        return (
          <Space>
            <Tooltip title="点击查看详情">
              <span
                style={{ color: primaryColor, cursor: 'pointer' }}
                onClick={() => navigate(`/experiments/${record.id}`)}
              >
                {text}
              </span>
            </Tooltip>
            <Tooltip title="编辑名称">
              <EditOutlined
                style={{ color: '#1890ff', cursor: 'pointer' }}
                onClick={(e) => {
                  e.stopPropagation();
                  handleStartEdit(record, 'name');
                }}
              />
            </Tooltip>
          </Space>
        );
      },
    },
    {
      title: '结果数',
      key: 'count',
      width: 100,
      align: 'center',
      render: (_, record) => (
        <span>{record.completed_combinations?.toLocaleString() || 0}</span>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      align: 'center',
      render: (text) => (text ? new Date(text).toLocaleString('zh-CN') : '-'),
    },
    {
      title: '状态',
      key: 'status',
      width: 120,
      align: 'center',
      render: (_, record) => {
        const config = statusConfig[record.status || ''] || { color: 'default', text: record.status, icon: null };
        return (
          <div>
            <Tag color={config.color} icon={config.icon}>
              {config.text}
            </Tag>
            {record.status === 'failed' && record.notes && (
              <Tooltip title={record.notes}>
                <span style={{ color: errorColor, fontSize: 12, marginLeft: 4, cursor: 'pointer' }}>
                  详情
                </span>
              </Tooltip>
            )}
          </div>
        );
      },
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      width: 300,
      align: 'center',
      render: (text, record) => {
        if (editingId === record.id && editingField === 'description') {
          return (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Input.TextArea
                size="small"
                value={editingValue}
                onChange={(e) => setEditingValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Escape') handleCancelEdit();
                  if (e.key === 'Enter' && e.ctrlKey) handleSaveEdit();
                }}
                autoFocus
                autoSize={{ minRows: 2, maxRows: 6 }}
                style={{ flex: 1, minWidth: 200 }}
              />
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <Button size="small" type="primary" onClick={handleSaveEdit}>
                  保存
                </Button>
                <Button size="small" onClick={handleCancelEdit}>
                  取消
                </Button>
              </div>
            </div>
          );
        }
        // 处理多行描述的折叠显示
        const lines = (text || '').split('\n').filter((line: string) => line.trim());
        const hasMultipleLines = lines.length > 2;
        const displayText = hasMultipleLines ? lines.slice(0, 2).join('\n') + '...' : text;
        return (
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
            <Tooltip
              title={text ? <pre style={{ margin: 0, whiteSpace: 'pre-wrap', maxWidth: 400 }}>{text}</pre> : undefined}
              overlayStyle={{ maxWidth: 450 }}
            >
              <span style={{
                flex: 1,
                overflow: 'hidden',
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
                whiteSpace: 'pre-wrap',
                fontSize: 12,
                lineHeight: '1.5',
              }}>
                {displayText || '-'}
              </span>
            </Tooltip>
            <Tooltip title="编辑描述">
              <EditOutlined
                style={{ color: '#1890ff', cursor: 'pointer', flexShrink: 0, marginTop: 2 }}
                onClick={() => handleStartEdit(record, 'description')}
              />
            </Tooltip>
          </div>
        );
      },
    },
    {
      title: '操作',
      key: 'action',
      width: 100,
      align: 'center',
      render: (_, record) => (
        <Space>
          <Tooltip title="查看详情">
            <Button
              type="link"
              size="small"
              icon={<BarChartOutlined />}
              onClick={() => navigate(`/experiments/${record.id}`)}
            />
          </Tooltip>
          <Popconfirm
            title="确定删除这个实验吗？"
            description="删除后将无法恢复"
            onConfirm={() => handleDelete(record.id)}
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  // 计算统计数据
  const stats = {
    total: experiments.length,
    completed: experiments.filter(e => e.status === 'completed').length,
    running: experiments.filter(e => e.status === 'running' || e.status === 'importing').length,
    totalResults: experiments.reduce((sum, e) => sum + (e.completed_combinations || 0), 0),
  };

  return (
    <div>
      
      {/* 主表格卡片 */}
      <Card
        title={
          <Space>
            <ExperimentOutlined style={{ color: primaryColor }} />
            <span>实验列表</span>
          </Space>
        }
        extra={
          <Space>
            <Radio.Group
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              optionType="button"
              buttonStyle="solid"
              size="small"
            >
              <Radio.Button value="all">全部</Radio.Button>
              <Radio.Button value="kcin">KCIN</Radio.Button>
              <Radio.Button value="dcin">DCIN</Radio.Button>
            </Radio.Group>
            <Tooltip title="刷新">
              <Button icon={<ReloadOutlined />} onClick={loadExperiments} size="small" />
            </Tooltip>
          </Space>
        }
      >
        {/* 工具栏 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
          <Space>
            {selectedExperimentIds.length > 0 ? (
              <>
                <Tag color="blue" style={{ fontSize: 13, padding: '4px 8px' }}>
                  已选择 {selectedExperimentIds.length} 个实验
                </Tag>
                <Tooltip title="对比选中的实验结果">
                  <Button
                    type="primary"
                    icon={<SwapOutlined />}
                    onClick={() => navigate('/compare')}
                    disabled={selectedExperimentIds.length < 2}
                    size="small"
                  >
                    对比
                  </Button>
                </Tooltip>
                <Popconfirm
                  title="批量删除"
                  description={`确定要删除选中的 ${selectedExperimentIds.length} 个实验吗？`}
                  onConfirm={handleBatchDelete}
                  okText="删除"
                  cancelText="取消"
                  okButtonProps={{ danger: true }}
                >
                  <Button
                    danger
                    icon={<DeleteOutlined />}
                    loading={batchDeleting}
                    size="small"
                  >
                    删除选中
                  </Button>
                </Popconfirm>
                <Button onClick={clearSelection} size="small">
                  取消选择
                </Button>
              </>
            ) : (
              <Text type="secondary">选择实验可进行批量操作</Text>
            )}
          </Space>
          <Space>
            <Tooltip title="导出实验到其他平台">
              <Button
                icon={<DownloadOutlined />}
                onClick={() => setExportExpModalVisible(true)}
                size="small"
              >
                导出实验
              </Button>
            </Tooltip>
            <Tooltip title="从其他平台导入实验包">
              <Button
                type="primary"
                icon={<UploadOutlined />}
                onClick={() => setImportExpModalVisible(true)}
                size="small"
              >
                导入实验
              </Button>
            </Tooltip>
          </Space>
        </div>

        <Table
          columns={columns}
          dataSource={experiments}
          rowKey="id"
          loading={loading}
          rowSelection={{
            selectedRowKeys: selectedExperimentIds,
            onChange: (selectedRowKeys) => {
              // 清除现有选择，设置新选择
              clearSelection();
              (selectedRowKeys as number[]).forEach((id) => toggleExperimentSelection(id));
            },
          }}
          pagination={{
            defaultPageSize: 20,
            pageSizeOptions: [10, 20, 50, 100],
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => <Text type="secondary">共 {total} 个实验</Text>,
          }}
          locale={{
            emptyText: (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="暂无实验数据"
              >
                <Button type="primary" icon={<UploadOutlined />} onClick={() => setImportExpModalVisible(true)}>
                  导入第一个实验
                </Button>
              </Empty>
            ),
          }}
        />
      </Card>

      {/* 导出实验弹窗 */}
      <ExportExperimentModal
        open={exportExpModalVisible}
        onClose={() => setExportExpModalVisible(false)}
        experiments={experiments}
      />

      {/* 导入实验弹窗 */}
      <ImportModal
        open={importExpModalVisible}
        onClose={() => setImportExpModalVisible(false)}
        onSuccess={() => {
          loadExperiments();
        }}
      />
    </div>
  );
}
