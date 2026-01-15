/**
 * 流量文件选择树组件
 */
import React from 'react'
import { Card, Tree, Space, Button, Tag, Typography, Spin, Empty, Tooltip } from 'antd'
import {
  FileTextOutlined,
  FolderOpenOutlined,
  ReloadOutlined,
  MinusSquareOutlined,
  EyeOutlined,
} from '@ant-design/icons'
import { primaryColor } from '@/theme/colors'
import type { TrafficTreeNode } from '@/api/simulation'

const { Text } = Typography

interface TrafficFileTreeProps {
  trafficTree: TrafficTreeNode[]
  selectedFiles: string[]
  expandedKeys: string[]
  loading: boolean
  onSelect: (keys: string[]) => void
  onExpandedKeysChange: (keys: string[]) => void
  onRefresh: () => void
  onPreviewFile: (filePath: string) => void
}

export const TrafficFileTree: React.FC<TrafficFileTreeProps> = ({
  trafficTree,
  selectedFiles,
  expandedKeys,
  loading,
  onSelect,
  onExpandedKeysChange,
  onRefresh,
  onPreviewFile,
}) => {
  // 在树节点中查找节点
  const findNode = (nodes: TrafficTreeNode[], targetKey: string): TrafficTreeNode | null => {
    for (const node of nodes) {
      if (node.key === targetKey) return node
      if (node.children) {
        const found = findNode(node.children, targetKey)
        if (found) return found
      }
    }
    return null
  }

  return (
    <Card
      title={
        <Space>
          <FileTextOutlined style={{ color: primaryColor }} />
          <span>流量文件</span>
          <Tag color={selectedFiles.length > 0 ? 'blue' : 'default'}>{selectedFiles.length} 个已选</Tag>
        </Space>
      }
      extra={
        <Space>
          <Tooltip title="全部折叠">
            <Button icon={<MinusSquareOutlined />} onClick={() => onExpandedKeysChange([])} size="small" />
          </Tooltip>
          <Button icon={<ReloadOutlined />} onClick={onRefresh} size="small">
            刷新
          </Button>
        </Space>
      }
      style={{ marginBottom: 24 }}
      bodyStyle={{ maxHeight: 400, overflow: 'auto' }}
    >
      <Spin spinning={loading}>
        {trafficTree.length > 0 ? (
          <Tree
            checkable
            showIcon
            defaultExpandAll={false}
            expandedKeys={expandedKeys}
            onExpand={(keys) => onExpandedKeysChange(keys as string[])}
            checkedKeys={selectedFiles}
            onCheck={(checked) => {
              // 只选择文件（isLeaf=true），不选择目录
              const checkedKeys = Array.isArray(checked) ? checked : checked.checked
              const fileKeys = (checkedKeys as string[]).filter(key => {
                const node = findNode(trafficTree, key)
                return node?.isLeaf === true
              })
              onSelect(fileKeys)
            }}
            treeData={trafficTree}
            titleRender={(node: TrafficTreeNode) => (
              <span
                onDoubleClick={() => {
                  if (node.isLeaf && node.path) {
                    onPreviewFile(node.path)
                  }
                }}
                style={{ cursor: node.isLeaf ? 'pointer' : 'default' }}
              >
                {node.title}
                {node.isLeaf && node.size !== undefined && (
                  <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
                    ({(node.size / 1024).toFixed(1)} KB)
                  </Text>
                )}
                {node.isLeaf && (
                  <Tooltip title="预览">
                    <EyeOutlined
                      style={{ marginLeft: 8, color: primaryColor, fontSize: 12 }}
                      onClick={(e) => {
                        e.stopPropagation()
                        if (node.path) onPreviewFile(node.path)
                      }}
                    />
                  </Tooltip>
                )}
              </span>
            )}
            icon={(props: any) => props.data?.isLeaf ? <FileTextOutlined /> : <FolderOpenOutlined />}
          />
        ) : (
          <Empty description="无流量文件" image={Empty.PRESENTED_IMAGE_SIMPLE} />
        )}
      </Spin>
    </Card>
  )
}
