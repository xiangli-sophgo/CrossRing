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
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { useExperimentStore } from '../stores/experimentStore';
import { getExperiments, importFromCSV, deleteExperiment } from '../api';
import type { Experiment, ExperimentType } from '../types';

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
  const { experiments, setExperiments, loading, setLoading } = useExperimentStore();
  const [importModalVisible, setImportModalVisible] = useState(false);
  const [importForm] = Form.useForm();
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [typeFilter, setTypeFilter] = useState<ExperimentType | 'all'>('all');

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
        values.experiment_type || 'noc',
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

  // 表格列定义
  const columns: ColumnsType<Experiment> = [
    {
      title: '实验名称',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <a onClick={() => navigate(`/experiments/${record.id}`)}>{text}</a>
      ),
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
      render: (text) => (
        <Tooltip title={text}>
          <span>{text || '-'}</span>
        </Tooltip>
      ),
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
              <Radio.Button value="all">全部</Radio.Button>
              <Radio.Button value="noc">NoC</Radio.Button>
              <Radio.Button value="d2d">D2D</Radio.Button>
            </Radio.Group>
          </Space>
          <Space>
            <Button
              icon={<UploadOutlined />}
              onClick={() => setImportModalVisible(true)}
            >
              导入CSV
            </Button>
            <Button icon={<ReloadOutlined />} onClick={loadExperiments}>
              刷新
            </Button>
          </Space>
        </div>

        <Table
          columns={columns}
          dataSource={experiments}
          rowKey="id"
          loading={loading}
          pagination={{
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
            initialValue="noc"
            rules={[{ required: true, message: '请选择实验类型' }]}
          >
            <Select>
              <Select.Option value="noc">NoC</Select.Option>
              <Select.Option value="d2d">D2D</Select.Option>
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
    </div>
  );
}
