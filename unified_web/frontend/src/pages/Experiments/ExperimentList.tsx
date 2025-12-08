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
  Modal,
  Form,
  Input,
  Upload,
  message,
  Typography,
  Tooltip,
  Popconfirm,
  Select,
  Radio,
} from 'antd';
import {
  UploadOutlined,
  ReloadOutlined,
  DeleteOutlined,
  BarChartOutlined,
  DownloadOutlined,
  EditOutlined,
  SwapOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { useExperimentStore } from '../../stores/experimentStore';
import { getExperiments, importFromCSV, deleteExperiment, updateExperiment, deleteExperimentsBatch } from './api';
import type { Experiment, ExperimentType } from './types';
import ExportModal from './components/ExportModal';

const { Title } = Typography;

const statusColors: Record<string, string> = {
  running: 'processing',
  completed: 'success',
  failed: 'error',
  interrupted: 'warning',
  importing: 'processing',
};

const statusText: Record<string, string> = {
  running: '运行中',
  completed: '已完成',
  failed: '失败',
  interrupted: '已中断',
  importing: '导入中',
};


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
  const [importModalVisible, setImportModalVisible] = useState(false);
  const [exportModalVisible, setExportModalVisible] = useState(false);
  const [importForm] = Form.useForm();
  const [uploadFile, setUploadFile] = useState<File | null>(null);
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

  // 处理CSV导入
  const handleImport = async () => {
    try {
      const values = await importForm.validateFields();
      if (!uploadFile) {
        message.error('请选择CSV文件');
        return;
      }

      const result = await importFromCSV(
        uploadFile,
        values.name,
        values.experiment_type || 'kcin',
        values.description,
        values.topo_type
      );

      message.success(`导入成功！共导入 ${result.imported_count} 条记录`);
      if (result.errors.length > 0) {
        message.warning(`有 ${result.errors.length} 条记录导入失败`);
      }

      setImportModalVisible(false);
      importForm.resetFields();
      setUploadFile(null);
      loadExperiments();
    } catch (error) {
      message.error('导入失败');
    }
  };

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
            <span>{text}</span>
            <Tooltip title="编辑名称">
              <EditOutlined
                style={{ color: '#1890ff', cursor: 'pointer' }}
                onClick={() => handleStartEdit(record, 'name')}
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
      render: (_, record) => (
        <span>{record.completed_combinations?.toLocaleString() || 0}</span>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (text) => (text ? new Date(text).toLocaleString('zh-CN') : '-'),
    },
    {
      title: '状态',
      key: 'status',
      width: 150,
      render: (_, record) => (
        <div>
          <Tag color={statusColors[record.status || ''] || 'default'}>
            {statusText[record.status || ''] || record.status}
          </Tag>
          {record.status === 'failed' && record.notes && (
            <Tooltip title={record.notes}>
              <span style={{ color: '#ff4d4f', fontSize: 12, marginLeft: 4 }}>
                (查看错误)
              </span>
            </Tooltip>
          )}
        </div>
      ),
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
      render: (text, record) => {
        if (editingId === record.id && editingField === 'description') {
          return (
            <Space>
              <Input
                size="small"
                value={editingValue}
                onChange={(e) => setEditingValue(e.target.value)}
                onPressEnter={handleSaveEdit}
                onKeyDown={(e) => e.key === 'Escape' && handleCancelEdit()}
                autoFocus
                style={{ width: 200 }}
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
            <Tooltip title={text}>
              <span>{text || '-'}</span>
            </Tooltip>
            <Tooltip title="编辑描述">
              <EditOutlined
                style={{ color: '#1890ff', cursor: 'pointer' }}
                onClick={() => handleStartEdit(record, 'description')}
              />
            </Tooltip>
          </Space>
        );
      },
    },
    {
      title: '操作',
      key: 'action',
      width: 100,
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

  return (
    <div>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
          <Space>
            <Title level={4} style={{ margin: 0 }}>
              仿真结果数据库
            </Title>
            <Radio.Group
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              optionType="button"
              buttonStyle="solid"
            >
              <Tooltip title="显示所有类型的实验">
                <Radio.Button value="all">全部</Radio.Button>
              </Tooltip>
              <Tooltip title="仅显示KCIN类型实验">
                <Radio.Button value="kcin">KCIN</Radio.Button>
              </Tooltip>
              <Tooltip title="仅显示DCIN类型实验">
                <Radio.Button value="dcin">DCIN</Radio.Button>
              </Tooltip>
            </Radio.Group>
          </Space>
          <Space>
            {selectedExperimentIds.length > 0 && (
              <>
                <span style={{ color: '#1890ff' }}>
                  已选择 {selectedExperimentIds.length} 个实验
                </span>
                <Tooltip title="对比选中的实验结果">
                  <Button
                    type="primary"
                    icon={<SwapOutlined />}
                    onClick={() => navigate('/compare')}
                    disabled={selectedExperimentIds.length < 2}
                  >
                    对比
                  </Button>
                </Tooltip>
                <Popconfirm
                  title="批量删除"
                  description={`确定要删除选中的 ${selectedExperimentIds.length} 个实验吗？此操作不可恢复。`}
                  onConfirm={handleBatchDelete}
                  okText="删除"
                  cancelText="取消"
                  okButtonProps={{ danger: true }}
                >
                  <Tooltip title="删除选中的实验及其所有结果">
                    <Button
                      danger
                      icon={<DeleteOutlined />}
                      loading={batchDeleting}
                    >
                      删除选中
                    </Button>
                  </Tooltip>
                </Popconfirm>
                <Tooltip title="清除所有选择">
                  <Button onClick={clearSelection}>取消选择</Button>
                </Tooltip>
              </>
            )}
            <Tooltip title="从CSV文件导入实验数据">
              <Button
                icon={<UploadOutlined />}
                onClick={() => setImportModalVisible(true)}
              >
                导入CSV
              </Button>
            </Tooltip>
            <Tooltip title="将数据库和前端打包导出，可独立部署查看">
              <Button
                icon={<DownloadOutlined />}
                onClick={() => setExportModalVisible(true)}
              >
                导出打包
              </Button>
            </Tooltip>
            <Tooltip title="刷新实验列表">
              <Button icon={<ReloadOutlined />} onClick={loadExperiments}>
                刷新
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
            defaultPageSize: 50,
            pageSizeOptions: [10, 20, 50, 100],
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 个实验`,
          }}
        />
      </Card>

      {/* CSV导入弹窗 */}
      <Modal
        title="从CSV导入实验数据"
        open={importModalVisible}
        onOk={handleImport}
        onCancel={() => {
          setImportModalVisible(false);
          importForm.resetFields();
          setUploadFile(null);
        }}
        okText="导入"
        cancelText="取消"
      >
        <Form form={importForm} layout="vertical">
          <Form.Item
            name="name"
            label="实验名称"
            rules={[{ required: true, message: '请输入实验名称' }]}
          >
            <Input placeholder="输入唯一的实验名称" />
          </Form.Item>
          <Form.Item
            name="experiment_type"
            label="实验类型"
            initialValue="kcin"
            rules={[{ required: true, message: '请选择实验类型' }]}
          >
            <Select>
              <Select.Option value="kcin">KCIN</Select.Option>
              <Select.Option value="dcin">DCIN</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea rows={2} placeholder="可选的实验描述" />
          </Form.Item>
          <Form.Item name="topo_type" label="拓扑类型">
            <Input placeholder="如 5x4" />
          </Form.Item>
          <Form.Item label="CSV文件" required>
            <Upload
              beforeUpload={(file) => {
                setUploadFile(file);
                return false;
              }}
              onRemove={() => setUploadFile(null)}
              maxCount={1}
              accept=".csv"
            >
              <Button icon={<UploadOutlined />}>选择文件</Button>
            </Upload>
            {uploadFile && <span style={{ marginLeft: 8 }}>{uploadFile.name}</span>}
          </Form.Item>
        </Form>
      </Modal>

      {/* 导出打包弹窗 */}
      <ExportModal
        open={exportModalVisible}
        onClose={() => setExportModalVisible(false)}
        experiments={experiments}
      />
    </div>
  );
}
