/**
 * 结果详情折叠面板组件
 * 展示单条结果的配置参数、结果统计和结果文件
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Collapse,
  Descriptions,
  Button,
  Space,
  List,
  Typography,
  Empty,
  Spin,
} from 'antd';
import {
  FileTextOutlined,
  EyeOutlined,
  DownloadOutlined,
  LineChartOutlined,
} from '@ant-design/icons';
import type { SimulationResult, ExperimentType } from '@/types';
import { classifyParams, formatParamValue } from '@/utils/paramClassifier';
import { getResultFiles, getFileDownloadUrl, type ResultFileInfo } from '@/api/experiments';

interface Props {
  result: SimulationResult;
  experimentId: number;
  experimentName?: string;
  experimentType?: ExperimentType;
  hideConfigParams?: boolean;
}

const { Text } = Typography;

export default function ResultDetailPanel({ result, experimentId, experimentName, experimentType = 'kcin', hideConfigParams = false }: Props) {
  const navigate = useNavigate();
  const [dbFiles, setDbFiles] = useState<ResultFileInfo[]>([]);
  const [loadingFiles, setLoadingFiles] = useState(false);

  // 分类参数
  const { configParams, resultStats } = classifyParams(result.config_params || {});

  // 加载数据库文件列表
  useEffect(() => {
    const loadDbFiles = async () => {
      setLoadingFiles(true);
      try {
        const files = await getResultFiles(result.id, experimentType);
        setDbFiles(files);
      } catch {
        // 忽略错误，可能没有数据库文件
        setDbFiles([]);
      } finally {
        setLoadingFiles(false);
      }
    };
    loadDbFiles();
  }, [result.id, experimentType]);

  // 获取结果标签名称
  const getResultLabel = () => {
    // 从config_params中获取数据流名称（支持多种字段名）
    const trafficName = result.config_params?.['数据流名称']
      || result.config_params?.file_name
      || result.config_params?.TRAFFIC_FILE
      || result.config_params?.traffic_file
      || '';
    // 如果是路径，提取文件名（去除路径和扩展名）
    const displayName = String(trafficName).includes('/') || String(trafficName).includes('\\')
      ? String(trafficName).split('/').pop()?.split('\\').pop()?.replace(/\.[^/.]+$/, '') || ''
      : String(trafficName);
    // 构建标签：实验名称 - 数据流名称
    const parts = [experimentName, displayName].filter(Boolean);
    return parts.length > 0 ? parts.join(' - ') : `结果 ${result.id}`;
  };

  // 在结果分析页面打开HTML报告
  const handleViewHtml = () => {
    const label = getResultLabel();
    navigate(`/analysis?resultId=${result.id}&experimentId=${experimentId}&label=${encodeURIComponent(label)}&type=html`);
  };

  // 在结果分析页面打开波形
  const handleViewWaveform = () => {
    const label = getResultLabel();
    navigate(`/analysis?resultId=${result.id}&experimentId=${experimentId}&label=${encodeURIComponent(label)}&type=waveform`);
  };

  // 下载文件
  const handleDownloadFile = (fileId: number, fileName: string) => {
    const url = getFileDownloadUrl(fileId);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    link.click();
  };

  // 格式化文件大小
  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const collapseItems = [
    ...(!hideConfigParams ? [{
      key: 'config',
      label: `配置参数 (${Object.keys(configParams).length})`,
      children: (
        <Descriptions column={3} size="small" bordered>
          {Object.entries(configParams).map(([key, value]) => (
            <Descriptions.Item key={key} label={key}>
              {formatParamValue(value, key)}
            </Descriptions.Item>
          ))}
        </Descriptions>
      ),
    }] : []),
    {
      key: 'stats',
      label: `结果统计 (${Object.keys(resultStats).length})`,
      children: (
        <Descriptions column={3} size="small" bordered>
          {Object.entries(resultStats).map(([key, value]) => (
            <Descriptions.Item key={key} label={key}>
              {formatParamValue(value, key)}
            </Descriptions.Item>
          ))}
        </Descriptions>
      ),
    },
    {
      key: 'files',
      label: '结果文件',
      children: (
        <Space direction="vertical" style={{ width: '100%' }}>
          {/* HTML报告 - 支持轻量模式（has_result_html）和完整模式（result_html） */}
          <Space>
            {(result.result_html || result.has_result_html) ? (
              <Button
                type="primary"
                icon={<EyeOutlined />}
                onClick={handleViewHtml}
              >
                查看HTML报告
              </Button>
            ) : (
              <Text type="secondary">无HTML报告</Text>
            )}
            <Button
              type="primary"
              icon={<LineChartOutlined />}
              onClick={handleViewWaveform}
            >
              查看波形
            </Button>
          </Space>

          {/* 数据库文件列表 */}
          {loadingFiles ? (
            <Spin tip="加载文件列表..." />
          ) : dbFiles.length > 0 ? (
            <List
              size="small"
              bordered
              dataSource={dbFiles}
              renderItem={(file) => (
                <List.Item
                  actions={[
                    <Button
                      key="download"
                      type="link"
                      icon={<DownloadOutlined />}
                      onClick={() => handleDownloadFile(file.id, file.file_name)}
                    >
                      下载
                    </Button>,
                  ]}
                >
                  <List.Item.Meta
                    avatar={<FileTextOutlined />}
                    title={file.file_name}
                    description={
                      <Text type="secondary">{formatFileSize(file.file_size)}</Text>
                    }
                  />
                </List.Item>
              )}
            />
          ) : (
            <Empty description="无结果文件" image={Empty.PRESENTED_IMAGE_SIMPLE} />
          )}
        </Space>
      ),
    },
  ];

  return (
    <Collapse
      defaultActiveKey={[]}
      items={collapseItems}
    />
  );
}
