/**
 * 数据库导出模态框
 */

import { useState, useEffect } from 'react';
import { Modal, Checkbox, Button, Space, Spin, message, Typography, Divider, Tag, Progress } from 'antd';
import { DownloadOutlined, RocketOutlined } from '@ant-design/icons';
import type { Experiment } from '../types';
import { getExportInfo, buildExecutablePackage, type ExportInfo } from '../api';

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
          <DownloadOutlined />
          导出数据库
        </Space>
      }
      open={open}
      onCancel={onClose}
      width={600}
      footer={[
        <Button key="cancel" onClick={onClose}>
          取消
        </Button>,
        <Button
          key="build"
          type="primary"
          icon={<RocketOutlined />}
          onClick={handleBuildExecutable}
          loading={building}
        >
          生成可执行包
        </Button>,
      ]}
    >
      {/* 导出范围 */}
      <div style={{ marginBottom: 16 }}>
        <Text strong>导出范围</Text>
        <div style={{ marginTop: 8 }}>
          <Checkbox
            indeterminate={selectedIds.length > 0 && selectedIds.length < experiments.length}
            checked={selectedIds.length === experiments.length}
            onChange={handleSelectAll}
          >
            {selectedIds.length === 0 ? '全量导出（所有实验）' : `已选择 ${selectedIds.length} 个实验`}
          </Checkbox>
        </div>

        {/* 实验列表 */}
        <div
          style={{
            maxHeight: 200,
            overflowY: 'auto',
            marginTop: 8,
            padding: 8,
            border: '1px solid #d9d9d9',
            borderRadius: 4,
          }}
        >
          {experiments.map((exp) => (
            <div key={exp.id} style={{ padding: '4px 0' }}>
              <Checkbox
                checked={selectedIds.includes(exp.id)}
                onChange={() => handleExperimentToggle(exp.id)}
              >
                <Space>
                  <Text>{exp.name}</Text>
                  <Tag color={exp.experiment_type === 'kcin' ? 'blue' : 'green'}>
                    {exp.experiment_type.toUpperCase()}
                  </Tag>
                  {exp.topo_type && <Tag>{exp.topo_type}</Tag>}
                </Space>
              </Checkbox>
            </div>
          ))}
        </div>
      </div>

      <Divider />

      {/* 导出预览 */}
      <div>
        <Text strong>导出预览</Text>
        {loading ? (
          <div style={{ textAlign: 'center', padding: 16 }}>
            <Spin />
          </div>
        ) : exportInfo ? (
          <div
            style={{
              marginTop: 8,
              padding: 12,
              background: '#f5f5f5',
              borderRadius: 4,
            }}
          >
            <Space direction="vertical" size="small">
              <Text>
                实验数量: <Text strong>{exportInfo.experiments_count}</Text>
              </Text>
              <Text>
                结果数量: <Text strong>{exportInfo.results_count.toLocaleString()}</Text>
              </Text>
              <Text>
                数据库大小: <Text strong>{formatBytes(exportInfo.database_size)}</Text>
              </Text>
              <Text>
                导出类型:{' '}
                <Tag color={exportInfo.is_selective ? 'orange' : 'blue'}>
                  {exportInfo.is_selective ? '选择性导出' : '全量导出'}
                </Tag>
              </Text>
            </Space>
          </div>
        ) : null}
      </div>

      {/* 构建进度 */}
      {building && (
        <>
          <Divider />
          <div>
            <Text strong>构建进度</Text>
            <Progress percent={Math.round(buildProgress)} status="active" />
            <Text type="secondary">正在构建前端和打包后端，这可能需要1-2分钟...</Text>
          </div>
        </>
      )}
    </Modal>
  );
}
