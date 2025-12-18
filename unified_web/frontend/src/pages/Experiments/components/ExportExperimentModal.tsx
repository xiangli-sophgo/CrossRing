/**
 * 导出实验模态框
 */

import { useState, useEffect } from 'react';
import {
  Modal,
  Checkbox,
  Button,
  Space,
  Spin,
  Typography,
  Divider,
  Tag,
  Card,
  Empty,
  Row,
  Col,
} from 'antd';
import { DownloadOutlined, ExperimentOutlined, FileZipOutlined } from '@ant-design/icons';
import type { Experiment } from '../types';
import { getExperimentExportInfo, exportExperiment, type ExperimentExportInfo } from '../api';
import { primaryColor, successColor } from '@/theme/colors';

const { Text } = Typography;

interface ExportExperimentModalProps {
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

export default function ExportExperimentModal({
  open,
  onClose,
  experiments,
}: ExportExperimentModalProps) {
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [exportInfo, setExportInfo] = useState<ExperimentExportInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);

  // 获取导出信息
  useEffect(() => {
    if (open && selectedIds.length > 0) {
      fetchExportInfo();
    } else {
      setExportInfo(null);
    }
  }, [open, selectedIds]);

  const fetchExportInfo = async () => {
    if (selectedIds.length === 0) return;
    setLoading(true);
    try {
      const info = await getExperimentExportInfo(selectedIds);
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

  const handleExport = () => {
    if (selectedIds.length === 0) return;
    setExporting(true);
    const url = exportExperiment(selectedIds);
    window.open(url, '_blank');
    setTimeout(() => {
      setExporting(false);
    }, 1000);
  };

  const handleClose = () => {
    setSelectedIds([]);
    setExportInfo(null);
    onClose();
  };

  return (
    <Modal
      title={
        <Space>
          <DownloadOutlined style={{ color: primaryColor }} />
          <span>导出实验</span>
        </Space>
      }
      open={open}
      onCancel={handleClose}
      width={700}
      footer={null}
    >
      {/* 选择实验 */}
      <div style={{ marginBottom: 20 }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: 12,
          }}
        >
          <Space>
            <ExperimentOutlined style={{ color: primaryColor }} />
            <Text strong>选择要导出的实验</Text>
          </Space>
          <Checkbox
            indeterminate={selectedIds.length > 0 && selectedIds.length < experiments.length}
            checked={selectedIds.length === experiments.length}
            onChange={handleSelectAll}
          >
            {selectedIds.length === 0
              ? '全选'
              : `已选 ${selectedIds.length}/${experiments.length}`}
          </Checkbox>
        </div>

        {/* 实验列表 */}
        <div
          style={{
            maxHeight: 280,
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
                <div
                  style={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    minWidth: 0,
                  }}
                >
                  <Text ellipsis style={{ flex: 1, minWidth: 0 }}>
                    {exp.name}
                  </Text>
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
        {selectedIds.length === 0 ? (
          <div style={{ textAlign: 'center', padding: 24, color: '#999' }}>
            请先选择要导出的实验
          </div>
        ) : loading ? (
          <div style={{ textAlign: 'center', padding: 24 }}>
            <Spin />
          </div>
        ) : exportInfo ? (
          <Row gutter={16}>
            <Col span={8}>
              <Card size="small" style={{ textAlign: 'center' }}>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  实验数量
                </Text>
                <div style={{ fontSize: 24, fontWeight: 600, color: primaryColor }}>
                  {exportInfo.experiments_count}
                </div>
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small" style={{ textAlign: 'center' }}>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  结果数量
                </Text>
                <div style={{ fontSize: 24, fontWeight: 600, color: primaryColor }}>
                  {exportInfo.results_count.toLocaleString()}
                </div>
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small" style={{ textAlign: 'center' }}>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  预估大小
                </Text>
                <div style={{ fontSize: 20, fontWeight: 600, color: primaryColor }}>
                  {formatBytes(exportInfo.estimated_size)}
                </div>
              </Card>
            </Col>
          </Row>
        ) : null}
      </div>

      {/* 导出按钮 */}
      <Button
        block
        type="primary"
        size="large"
        icon={<DownloadOutlined />}
        onClick={handleExport}
        loading={exporting}
        disabled={selectedIds.length === 0}
        style={{
          height: 48,
          background:
            selectedIds.length > 0
              ? `linear-gradient(135deg, ${primaryColor} 0%, #4096ff 100%)`
              : undefined,
        }}
      >
        导出实验
      </Button>
      <Text
        type="secondary"
        style={{ fontSize: 11, display: 'block', textAlign: 'center', marginTop: 4 }}
      >
        导出为ZIP包，可在其他平台导入
      </Text>
    </Modal>
  );
}
