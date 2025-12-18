/**
 * 实验导入模态框
 */

import { useState } from 'react';
import {
  Modal,
  Button,
  Space,
  message,
  Typography,
  Divider,
  Tag,
  Progress,
  Upload,
  Table,
  Radio,
  Input,
  Result,
  Alert,
} from 'antd';
import {
  ImportOutlined,
  InboxOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
} from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';
import {
  checkImportPackage,
  executeImport,
  type CheckImportResult,
  type ImportExperimentInfo,
  type ImportConfigItem,
  type ImportResult,
} from '../api';
import { primaryColor, successColor, warningColor, errorColor } from '@/theme/colors';

const { Text, Title } = Typography;
const { Dragger } = Upload;

interface ImportModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

type StepType = 'upload' | 'config' | 'importing' | 'result';

interface ConflictConfig {
  originalId: number;
  action: 'rename' | 'overwrite' | 'skip';
  newName: string;
}

export default function ImportModal({ open, onClose, onSuccess }: ImportModalProps) {
  const [step, setStep] = useState<StepType>('upload');
  const [uploading, setUploading] = useState(false);
  const [checkResult, setCheckResult] = useState<CheckImportResult | null>(null);
  const [conflictConfigs, setConflictConfigs] = useState<Map<number, ConflictConfig>>(new Map());
  const [importing, setImporting] = useState(false);
  const [importProgress, setImportProgress] = useState(0);
  const [importResult, setImportResult] = useState<ImportResult | null>(null);

  // 重置状态
  const resetState = () => {
    setStep('upload');
    setUploading(false);
    setCheckResult(null);
    setConflictConfigs(new Map());
    setImporting(false);
    setImportProgress(0);
    setImportResult(null);
  };

  // 关闭时重置
  const handleClose = () => {
    resetState();
    onClose();
  };

  // 处理文件上传
  const handleUpload = async (file: File) => {
    setUploading(true);
    try {
      const result = await checkImportPackage(file);
      setCheckResult(result);

      if (result.valid && result.experiments) {
        // 初始化冲突配置
        const configs = new Map<number, ConflictConfig>();
        result.experiments.forEach((exp) => {
          configs.set(exp.id, {
            originalId: exp.id,
            action: exp.conflict ? 'rename' : 'rename', // 默认直接导入
            newName: exp.conflict ? `${exp.name}_imported` : exp.name,
          });
        });
        setConflictConfigs(configs);
        setStep('config');
      }
    } catch (error) {
      message.error('上传失败，请检查文件格式');
      console.error('上传失败:', error);
    } finally {
      setUploading(false);
    }
    return false; // 阻止默认上传行为
  };

  // 更新冲突配置
  const updateConflictConfig = (id: number, field: keyof ConflictConfig, value: string) => {
    setConflictConfigs((prev) => {
      const newMap = new Map(prev);
      const config = newMap.get(id);
      if (config) {
        newMap.set(id, { ...config, [field]: value });
      }
      return newMap;
    });
  };

  // 执行导入
  const handleExecuteImport = async () => {
    if (!checkResult?.temp_file_id) return;

    setStep('importing');
    setImporting(true);
    setImportProgress(0);

    // 模拟进度
    const progressInterval = setInterval(() => {
      setImportProgress((prev) => (prev >= 90 ? prev : prev + Math.random() * 15));
    }, 500);

    try {
      const importConfig: ImportConfigItem[] = [];
      conflictConfigs.forEach((config) => {
        importConfig.push({
          original_id: config.originalId,
          action: config.action,
          new_name: config.action === 'rename' ? config.newName : undefined,
        });
      });

      const result = await executeImport(checkResult.temp_file_id, importConfig);
      clearInterval(progressInterval);
      setImportProgress(100);
      setImportResult(result);
      setStep('result');

      if (result.success && result.imported.length > 0) {
        onSuccess();
      }
    } catch (error) {
      clearInterval(progressInterval);
      message.error('导入失败');
      console.error('导入失败:', error);
      setStep('config');
    } finally {
      setImporting(false);
    }
  };

  // 渲染上传步骤
  const renderUploadStep = () => (
    <div>
      <Dragger
        accept=".zip"
        showUploadList={false}
        beforeUpload={handleUpload}
        disabled={uploading}
      >
        <p className="ant-upload-drag-icon">
          <InboxOutlined style={{ color: primaryColor, fontSize: 48 }} />
        </p>
        <p className="ant-upload-text">点击或拖拽ZIP文件到此处</p>
        <p className="ant-upload-hint">支持从其他平台导出的实验包</p>
      </Dragger>

      {uploading && (
        <div style={{ marginTop: 16, textAlign: 'center' }}>
          <Progress percent={50} status="active" showInfo={false} />
          <Text type="secondary">正在解析导入包...</Text>
        </div>
      )}

      {checkResult && !checkResult.valid && (
        <Alert
          type="error"
          message="导入包无效"
          description={checkResult.error}
          showIcon
          style={{ marginTop: 16 }}
        />
      )}
    </div>
  );

  // 渲染配置步骤
  const renderConfigStep = () => {
    if (!checkResult?.experiments) return null;

    const hasConflict = checkResult.experiments.some((e) => e.conflict);

    const columns = [
      {
        title: '实验名称',
        dataIndex: 'name',
        key: 'name',
        render: (name: string, record: ImportExperimentInfo) => (
          <Space>
            <Text>{name}</Text>
            {record.conflict && (
              <Tag color="warning" icon={<WarningOutlined />}>
                名称冲突
              </Tag>
            )}
          </Space>
        ),
      },
      {
        title: '类型',
        dataIndex: 'experiment_type',
        key: 'experiment_type',
        width: 80,
        render: (type: string) => (
          <Tag color={type === 'kcin' ? 'blue' : 'purple'}>{type.toUpperCase()}</Tag>
        ),
      },
      {
        title: '结果数',
        dataIndex: 'results_count',
        key: 'results_count',
        width: 80,
        render: (count: number) => count.toLocaleString(),
      },
      {
        title: '处理方式',
        key: 'action',
        width: 280,
        render: (_: unknown, record: ImportExperimentInfo) => {
          const config = conflictConfigs.get(record.id);
          if (!config) return null;

          return (
            <Space direction="vertical" size={4} style={{ width: '100%' }}>
              <Radio.Group
                value={config.action}
                onChange={(e) => updateConflictConfig(record.id, 'action', e.target.value)}
                size="small"
              >
                <Radio value="rename">
                  {record.conflict ? '重命名' : '导入'}
                </Radio>
                {record.conflict && <Radio value="overwrite">覆盖</Radio>}
                <Radio value="skip">跳过</Radio>
              </Radio.Group>
              {config.action === 'rename' && (
                <Input
                  size="small"
                  value={config.newName}
                  onChange={(e) => updateConflictConfig(record.id, 'newName', e.target.value)}
                  placeholder="输入新名称"
                  style={{ width: '100%' }}
                />
              )}
            </Space>
          );
        },
      },
    ];

    return (
      <div>
        {/* 包信息 */}
        {checkResult.package_info && (
          <Alert
            type="info"
            message={
              <Space>
                <Text>来源: {checkResult.package_info.source_platform}</Text>
                <Divider type="vertical" />
                <Text>导出时间: {new Date(checkResult.package_info.export_time).toLocaleString()}</Text>
              </Space>
            }
            style={{ marginBottom: 16 }}
          />
        )}

        {hasConflict && (
          <Alert
            type="warning"
            message="检测到名称冲突"
            description="部分实验与现有实验同名，请选择处理方式"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        {/* 实验列表 */}
        <Table
          dataSource={checkResult.experiments}
          columns={columns}
          rowKey="id"
          size="small"
          pagination={false}
          scroll={{ y: 300 }}
        />

        {/* 操作按钮 */}
        <div style={{ marginTop: 16, display: 'flex', justifyContent: 'space-between' }}>
          <Button onClick={() => setStep('upload')}>返回</Button>
          <Button type="primary" onClick={handleExecuteImport}>
            开始导入
          </Button>
        </div>
      </div>
    );
  };

  // 渲染导入中步骤
  const renderImportingStep = () => (
    <div style={{ textAlign: 'center', padding: '40px 0' }}>
      <Progress
        type="circle"
        percent={Math.round(importProgress)}
        status="active"
        strokeColor={{ '0%': primaryColor, '100%': successColor }}
      />
      <div style={{ marginTop: 16 }}>
        <Text strong>正在导入实验数据...</Text>
      </div>
      <Text type="secondary">请勿关闭此窗口</Text>
    </div>
  );

  // 渲染结果步骤
  const renderResultStep = () => {
    if (!importResult) return null;

    const successCount = importResult.imported.length;
    const skipCount = importResult.skipped.length;
    const errorCount = importResult.errors.length;

    return (
      <div>
        <Result
          status={importResult.success ? 'success' : 'warning'}
          title={importResult.success ? '导入完成' : '部分导入失败'}
          subTitle={
            <Space>
              <Tag color="success" icon={<CheckCircleOutlined />}>
                成功: {successCount}
              </Tag>
              {skipCount > 0 && <Tag color="default">跳过: {skipCount}</Tag>}
              {errorCount > 0 && (
                <Tag color="error" icon={<CloseCircleOutlined />}>
                  失败: {errorCount}
                </Tag>
              )}
            </Space>
          }
        />

        {/* 导入详情 */}
        {successCount > 0 && (
          <div style={{ marginBottom: 16 }}>
            <Text strong>成功导入的实验:</Text>
            <div
              style={{
                maxHeight: 150,
                overflowY: 'auto',
                padding: 12,
                background: '#f6ffed',
                borderRadius: 8,
                marginTop: 8,
              }}
            >
              {importResult.imported.map((item) => (
                <div key={item.new_id} style={{ padding: '4px 0' }}>
                  <CheckCircleOutlined style={{ color: successColor, marginRight: 8 }} />
                  <Text>{item.name}</Text>
                  <Text type="secondary"> ({item.results_count} 条结果)</Text>
                </div>
              ))}
            </div>
          </div>
        )}

        {errorCount > 0 && (
          <div style={{ marginBottom: 16 }}>
            <Text strong type="danger">
              导入失败:
            </Text>
            <div
              style={{
                maxHeight: 100,
                overflowY: 'auto',
                padding: 12,
                background: '#fff2f0',
                borderRadius: 8,
                marginTop: 8,
              }}
            >
              {importResult.errors.map((item) => (
                <div key={item.original_id} style={{ padding: '4px 0' }}>
                  <CloseCircleOutlined style={{ color: errorColor, marginRight: 8 }} />
                  <Text type="danger">{item.error}</Text>
                </div>
              ))}
            </div>
          </div>
        )}

        <Button type="primary" block onClick={handleClose}>
          完成
        </Button>
      </div>
    );
  };

  // 根据步骤渲染内容
  const renderContent = () => {
    switch (step) {
      case 'upload':
        return renderUploadStep();
      case 'config':
        return renderConfigStep();
      case 'importing':
        return renderImportingStep();
      case 'result':
        return renderResultStep();
      default:
        return null;
    }
  };

  return (
    <Modal
      title={
        <Space>
          <ImportOutlined style={{ color: primaryColor }} />
          <span>导入实验</span>
        </Space>
      }
      open={open}
      onCancel={handleClose}
      width={700}
      footer={null}
      destroyOnClose
    >
      {renderContent()}
    </Modal>
  );
}
