/**
 * 数据库导出模态框
 */

import { useState, useEffect } from 'react';
import { Modal, Checkbox, Button, Space, Spin, message, Typography, Divider, Tag, Progress, Card, Empty, Row, Col } from 'antd';
import { RocketOutlined, DatabaseOutlined, ExperimentOutlined, FileZipOutlined } from '@ant-design/icons';
import type { Experiment } from '../types';
import { getExportInfo, buildExecutablePackage, type ExportInfo } from '../api';
import { primaryColor, successColor } from '@/theme/colors';

const { Text } = Typography;

interface ExportModalProps {
  open: boolean;
  onClose: () => void;
  experiments: Experiment[];
}

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export default function ExportModal({ open, onClose, experiments }: ExportModalProps) {
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [exportInfo, setExportInfo] = useState<ExportInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [building, setBuilding] = useState(false);
  const [buildProgress, setBuildProgress] = useState(0);

  // 获取导出信息
  useEffect(() => {
    if (open) {
      fetchExportInfo();
    }
  }, [open, selectedIds]);

  const fetchExportInfo = async () => {
    setLoading(true);
    try {
      const info = await getExportInfo(selectedIds.length > 0 ? selectedIds : undefined);
      setExportInfo(info);
    } catch (error) {
      console.error('获取导出信息失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleExperimentToggle = (id: number) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]
    );
  };

  const handleSelectAll = () => {
    if (selectedIds.length === experiments.length) {
      setSelectedIds([]);
    } else {
      setSelectedIds(experiments.map((e) => e.id));
    }
  };

  const handleBuildExecutable = async () => {
    setBuilding(true);
    setBuildProgress(0);

    // 模拟进度
    const progressInterval = setInterval(() => {
      setBuildProgress((prev) => {
        if (prev >= 90) return prev;
        return prev + Math.random() * 10;
      });
    }, 1000);

    try {
      message.loading({ content: '正在构建可执行包，请稍候...', key: 'build', duration: 0 });

      const url = buildExecutablePackage(selectedIds.length > 0 ? selectedIds : undefined);
      window.open(url, '_blank');

      clearInterval(progressInterval);
      setBuildProgress(100);
      message.success({ content: '构建完成，开始下载', key: 'build' });
    } catch (error) {
      clearInterval(progressInterval);
      message.error({ content: '构建失败，请检查后端日志', key: 'build' });
      console.error('构建失败:', error);
    } finally {
      setBuilding(false);
      setBuildProgress(0);
    }
  };

  return (
    <Modal
      title={
        <Space>
          <DatabaseOutlined style={{ color: primaryColor }} />
          <span>导出数据库</span>
        </Space>
      }
      open={open}
      onCancel={onClose}
      width={700}
      footer={null}
    >
      {/* 导出范围 */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <Space>
            <ExperimentOutlined style={{ color: primaryColor }} />
            <Text strong>选择导出内容</Text>
          </Space>
          <Checkbox
            indeterminate={selectedIds.length > 0 && selectedIds.length < experiments.length}
            checked={selectedIds.length === experiments.length}
            onChange={handleSelectAll}
          >
            {selectedIds.length === 0 ? '全选' : `已选 ${selectedIds.length}/${experiments.length}`}
          </Checkbox>
        </div>

        {/* 实验列表 */}
        <div
          style={{
            maxHeight: 220,
            overflowY: 'auto',
            padding: 12,
            border: '1px solid #f0f0f0',
            borderRadius: 8,
            background: '#fafafa',
          }}
        >
          {experiments.length === 0 ? (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无实验数据" />
          ) : (
            experiments.map((exp) => (
              <div
                key={exp.id}
                onClick={() => handleExperimentToggle(exp.id)}
                style={{
                  padding: '8px 12px',
                  marginBottom: 4,
                  background: selectedIds.includes(exp.id) ? '#e6f4ff' : '#fff',
                  borderRadius: 6,
                  border: `1px solid ${selectedIds.includes(exp.id) ? primaryColor : '#f0f0f0'}`,
                  transition: 'all 0.2s',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <Checkbox
                  checked={selectedIds.includes(exp.id)}
                  onClick={(e) => e.stopPropagation()}
                  onChange={() => handleExperimentToggle(exp.id)}
                />
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between', minWidth: 0 }}>
                  <Text ellipsis style={{ flex: 1, minWidth: 0 }}>{exp.name}</Text>
                  <Space size={4} style={{ flexShrink: 0, marginLeft: 8 }}>
                    <Tag color={exp.experiment_type === 'kcin' ? 'blue' : 'purple'}>
                      {exp.experiment_type.toUpperCase()}
                    </Tag>
                    {exp.topo_type && <Tag>{exp.topo_type}</Tag>}
                    {exp.completed_combinations && (
                      <Tag color="default">{exp.completed_combinations}条</Tag>
                    )}
                  </Space>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <Divider style={{ margin: '16px 0' }} />

      {/* 导出预览 */}
      <div style={{ marginBottom: 20 }}>
        <Space style={{ marginBottom: 12 }}>
          <FileZipOutlined style={{ color: successColor }} />
          <Text strong>导出预览</Text>
        </Space>
        {loading ? (
          <div style={{ textAlign: 'center', padding: 24 }}>
            <Spin />
          </div>
        ) : exportInfo ? (
          <Row gutter={16}>
            <Col span={8}>
              <Card size="small" style={{ textAlign: 'center' }}>
                <Text type="secondary" style={{ fontSize: 12 }}>实验数量</Text>
                <div style={{ fontSize: 24, fontWeight: 600, color: primaryColor }}>
                  {exportInfo.experiments_count}
                </div>
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small" style={{ textAlign: 'center' }}>
                <Text type="secondary" style={{ fontSize: 12 }}>结果数量</Text>
                <div style={{ fontSize: 24, fontWeight: 600, color: primaryColor }}>
                  {exportInfo.results_count.toLocaleString()}
                </div>
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small" style={{ textAlign: 'center' }}>
                <Text type="secondary" style={{ fontSize: 12 }}>预估大小</Text>
                <div style={{ fontSize: 20, fontWeight: 600, color: primaryColor }}>
                  {formatBytes(exportInfo.database_size)}
                </div>
              </Card>
            </Col>
          </Row>
        ) : null}
      </div>

      {/* 构建进度 */}
      {building && (
        <div style={{ marginBottom: 20 }}>
          <Text strong>构建进度</Text>
          <Progress
            percent={Math.round(buildProgress)}
            status="active"
            strokeColor={{ from: primaryColor, to: '#4096ff' }}
            style={{ marginTop: 8 }}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>
            正在构建前端和打包后端，这可能需要1-2分钟...
          </Text>
        </div>
      )}

      {/* 导出按钮 */}
      <Button
        block
        type="primary"
        size="large"
        icon={<RocketOutlined />}
        onClick={handleBuildExecutable}
        loading={building}
        style={{
          height: 48,
          background: `linear-gradient(135deg, ${primaryColor} 0%, #4096ff 100%)`,
        }}
      >
        生成可执行包
      </Button>
      <Text type="secondary" style={{ fontSize: 11, display: 'block', textAlign: 'center', marginTop: 4 }}>
        一键打包，双击即可运行查看
      </Text>
    </Modal>
  );
}
