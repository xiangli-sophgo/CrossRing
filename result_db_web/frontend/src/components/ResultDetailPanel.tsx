/**
 * 结果详情折叠面板组件
 * 展示单条结果的配置参数、结果统计和结果文件
 */

import { useState } from 'react';
import {
  Collapse,
  Descriptions,
  Button,
  Space,
  List,
  message,
  Typography,
  Empty,
} from 'antd';
import {
  FileTextOutlined,
  FolderOpenOutlined,
  EyeOutlined,
} from '@ant-design/icons';
import type { SimulationResult } from '../types';
import { classifyParams, formatParamValue } from '../utils/paramClassifier';
import { getResultHtmlUrl, openLocalFile } from '../api';

interface Props {
  result: SimulationResult;
  experimentId: number;
  hideConfigParams?: boolean;
}

const { Text } = Typography;

export default function ResultDetailPanel({ result, experimentId, hideConfigParams = false }: Props) {
  const [openingFile, setOpeningFile] = useState<string | null>(null);

  // 分类参数
  const { configParams, resultStats } = classifyParams(result.config_params || {});

  // 打开HTML报告
  const handleViewHtml = () => {
    const url = getResultHtmlUrl(result.id, experimentId);
    window.open(url, '_blank');
  };

  // 打开本地文件
  const handleOpenFile = async (filePath: string) => {
    setOpeningFile(filePath);
    try {
      await openLocalFile(filePath);
      message.success('文件已打开');
    } catch {
      message.error('打开文件失败');
    } finally {
      setOpeningFile(null);
    }
  };

  // 获取文件名
  const getFileName = (filePath: string) => {
    const parts = filePath.replace(/\\/g, '/').split('/');
    return parts[parts.length - 1];
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
          {/* HTML报告按钮 */}
          {result.result_html ? (
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

          {/* 结果文件列表 */}
          {result.result_files && result.result_files.length > 0 ? (
            <List
              size="small"
              bordered
              dataSource={result.result_files}
              renderItem={(filePath) => (
                <List.Item
                  actions={[
                    <Button
                      key="open"
                      type="link"
                      icon={<FolderOpenOutlined />}
                      loading={openingFile === filePath}
                      onClick={() => handleOpenFile(filePath)}
                    >
                      打开
                    </Button>,
                  ]}
                >
                  <List.Item.Meta
                    avatar={<FileTextOutlined />}
                    title={getFileName(filePath)}
                    description={<Text type="secondary" ellipsis>{filePath}</Text>}
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
