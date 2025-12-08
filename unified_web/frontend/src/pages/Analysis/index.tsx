/**
 * 结果分析页面 - 显示HTML报告
 * 支持多标签页查看不同结果的报告
 */

import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Tabs, Empty, Button, Space, Tooltip } from 'antd';
import {
  CloseOutlined,
  ExpandOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { getResultHtmlUrl } from '../Experiments/api';

interface TabItem {
  key: string;
  label: string;
  resultId: number;
  experimentId: number;
}

// 从 localStorage 加载标签页
const loadTabs = (): TabItem[] => {
  const saved = localStorage.getItem('analysis_tabs');
  if (saved) {
    try {
      return JSON.parse(saved);
    } catch {
      return [];
    }
  }
  return [];
};

// 保存标签页到 localStorage
const saveTabs = (tabs: TabItem[]) => {
  localStorage.setItem('analysis_tabs', JSON.stringify(tabs));
};

export default function Analysis() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const [tabs, setTabs] = useState<TabItem[]>(loadTabs);
  const [activeKey, setActiveKey] = useState<string>('');
  const [refreshKey, setRefreshKey] = useState(0);

  // 从 URL 参数获取要打开的结果
  useEffect(() => {
    const resultId = searchParams.get('resultId');
    const experimentId = searchParams.get('experimentId');
    const label = searchParams.get('label') || `结果 ${resultId}`;

    if (resultId && experimentId) {
      const key = `${experimentId}-${resultId}`;

      // 检查是否已存在该标签页
      const existingTab = tabs.find(t => t.key === key);
      if (!existingTab) {
        // 添加新标签页
        const newTab: TabItem = {
          key,
          label,
          resultId: parseInt(resultId),
          experimentId: parseInt(experimentId),
        };
        const newTabs = [...tabs, newTab];
        setTabs(newTabs);
        saveTabs(newTabs);
      }

      setActiveKey(key);
      // 清除 URL 参数
      setSearchParams({});
    }
  }, [searchParams]);

  // 初始化时设置激活的标签页
  useEffect(() => {
    if (!activeKey && tabs.length > 0) {
      setActiveKey(tabs[0].key);
    }
  }, [tabs]);

  // 关闭标签页
  const handleTabClose = (targetKey: string) => {
    const newTabs = tabs.filter(t => t.key !== targetKey);
    setTabs(newTabs);
    saveTabs(newTabs);

    // 如果关闭的是当前激活的标签页，切换到其他标签
    if (activeKey === targetKey && newTabs.length > 0) {
      setActiveKey(newTabs[newTabs.length - 1].key);
    }
  };

  // 标签页编辑（关闭）
  const handleTabEdit = (
    targetKey: React.MouseEvent | React.KeyboardEvent | string,
    action: 'add' | 'remove'
  ) => {
    if (action === 'remove' && typeof targetKey === 'string') {
      handleTabClose(targetKey);
    }
  };

  // 在新窗口打开
  const handleOpenInNewWindow = (tab: TabItem) => {
    const url = getResultHtmlUrl(tab.resultId, tab.experimentId);
    window.open(url, '_blank');
  };

  // 刷新当前标签页
  const handleRefresh = () => {
    setRefreshKey(prev => prev + 1);
  };

  // 获取当前激活的标签页
  const currentTab = tabs.find(t => t.key === activeKey);

  // 生成标签页项
  const tabItems = tabs.map(tab => ({
    key: tab.key,
    label: (
      <span style={{ userSelect: 'none' }}>
        {tab.label}
      </span>
    ),
    children: (
      <div style={{ height: 'calc(100vh - 220px)', position: 'relative' }}>
        <iframe
          key={`${tab.key}-${refreshKey}`}
          src={getResultHtmlUrl(tab.resultId, tab.experimentId)}
          style={{
            width: '100%',
            height: '100%',
            border: 'none',
            borderRadius: 4,
          }}
          title={tab.label}
        />
      </div>
    ),
  }));

  if (tabs.length === 0) {
    return (
      <div style={{
        height: 'calc(100vh - 200px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <Empty
          description={
            <span>
              暂无打开的报告
              <br />
              <span style={{ color: '#999', fontSize: 12 }}>
                在结果管理页面双击结果行，然后点击"查看HTML报告"
              </span>
            </span>
          }
        >
          <Button type="primary" onClick={() => navigate('/experiments')}>
            前往结果管理
          </Button>
        </Empty>
      </div>
    );
  }

  return (
    <div>
      {/* 工具栏 */}
      <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'flex-end' }}>
        <Space>
          <Tooltip title="刷新当前报告">
            <Button
              icon={<ReloadOutlined />}
              onClick={handleRefresh}
              disabled={!currentTab}
            />
          </Tooltip>
          <Tooltip title="在新窗口打开">
            <Button
              icon={<ExpandOutlined />}
              onClick={() => currentTab && handleOpenInNewWindow(currentTab)}
              disabled={!currentTab}
            />
          </Tooltip>
        </Space>
      </div>

      {/* 标签页 */}
      <Tabs
        type="editable-card"
        hideAdd
        activeKey={activeKey}
        onChange={setActiveKey}
        onEdit={handleTabEdit}
        items={tabItems}
        tabBarStyle={{ marginBottom: 0 }}
      />
    </div>
  );
}
