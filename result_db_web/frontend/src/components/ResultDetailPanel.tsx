/**
 * 结果详情折叠面板组件
 * 展示单条结果的配置参数、结果统计和结果文件
 */

import { useState, useEffect } from 'react';
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
  FolderOpenOutlined,
} from '@ant-design/icons';
import type { SimulationResult, ExperimentType } from '../types';
import { classifyParams, formatParamValue } from '../utils/paramClassifier';
import { getResultHtmlUrl, getResultFiles, getFileDownloadUrl, openFileDirectory, type ResultFileInfo } from '../api';

interface Props {
  result: SimulationResult;
  experimentId: number;
  experimentType?: ExperimentType;
  hideConfigParams?: boolean;
}

const { Text } = Typography;

export default function ResultDetailPanel({ result, experimentId, experimentType = 'kcin', hideConfigParams = false }: Props) {
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

  // 打开HTML报告
  const handleViewHtml = () => {
    const url = getResultHtmlUrl(result.id, experimentId);
    window.open(url, '_blank');
  };

  // 打开文件所在目录
  const handleOpenDirectory = async (filePath: string) => {
    try {
      await openFileDirectory(filePath);
    } catch {
      // 忽略错误
    }
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

  // 从result_files中提取HTML路径
  const getHtmlPath = () => {
    if (result.result_files && result.result_files.length > 0) {
      const htmlFile = result.result_files.find(f => f.endsWith('.html'));
      if (htmlFile) return htmlFile;
    }
    return null;
  };

  const collapseItems = [
    ...(!hideConfigParams ? [{
      key: 'config',
      label: `配置参数 (${Object.keys(configParams).length})`,
      children: (
        <Descriptions column={3} size="small" bordered>
          {Object.entries(configParams).map(([key, value]) => (
            <Descriptions.Item key={key} label={key}>
              {formatParamValue(value)}
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
              {formatParamValue(value)}
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
          {/* HTML报告 */}
          {result.result_html ? (
            <Space>
              <Button
                type="primary"
                icon={<EyeOutlined />}
                onClick={handleViewHtml}
              >
                查看HTML报告
              </Button>
              {getHtmlPath() && (
                <Text type="secondary" style={{ marginLeft: 8 }}>
                  {getHtmlPath()}
                </Text>
              )}
            </Space>
          ) : (
            <Text type="secondary">无HTML报告</Text>
          )}

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
                      key="open-dir"
                      type="link"
                      icon={<FolderOpenOutlined />}
                      onClick={() => handleOpenDirectory(file.file_path)}
                    >
                      打开目录
                    </Button>,
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
                      <Space direction="vertical" size={0}>
                        <Text type="secondary" ellipsis style={{ maxWidth: 400 }}>{file.file_path}</Text>
                        <Text type="secondary">{formatFileSize(file.file_size)}</Text>
                      </Space>
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
