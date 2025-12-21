/**
 * 结果分析页面 - 显示HTML报告和波形
 * 支持多标签页查看不同结果的报告和波形
 */

import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Tabs, Empty, Button, Space, Tooltip } from 'antd';
import {
  ExpandOutlined,
  ReloadOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  FileTextOutlined,
  LineChartOutlined,
} from '@ant-design/icons';
import { getResultHtmlUrl } from '../Experiments/api';
import { WaveformViewer } from '@/components/waveform';

type ContentType = 'html' | 'waveform';

interface TabItem {
  key: string;
  label: string;
  resultId: number;
  experimentId: number;
  type: ContentType;  // 内容类型
}

// 从 localStorage 加载标签页
const loadTabs = (): TabItem[] => {
  const saved = localStorage.getItem('analysis_tabs');
  if (saved) {
    try {
      const tabs = JSON.parse(saved);
      // 迁移旧格式：为没有type字段的标签页添加默认值
      return tabs.map((tab: TabItem) => ({
        ...tab,
        type: tab.type || 'html',
        // 更新旧的key格式
        key: tab.key.includes('-html') || tab.key.includes('-waveform')
          ? tab.key
          : `${tab.key}-html`,
      }));
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

// 加载上次活动的标签页
const loadActiveKey = (): string => {
  return localStorage.getItem('analysis_active_key') || '';
};

// 保存当前活动标签页
const saveActiveKey = (key: string) => {
  localStorage.setItem('analysis_active_key', key);
};

export default function Analysis() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const [tabs, setTabs] = useState<TabItem[]>(loadTabs);
  const [activeKey, setActiveKey] = useState<string>(loadActiveKey);
  const [refreshKey, setRefreshKey] = useState(0);
  const [zoom, setZoom] = useState(0.8); // 默认80%缩放

  // 从 URL 参数获取要打开的结果
  useEffect(() => {
    const resultId = searchParams.get('resultId');
    const experimentId = searchParams.get('experimentId');
    const label = searchParams.get('label') || `结果 ${resultId}`;
    const type = (searchParams.get('type') || 'html') as ContentType;

    if (resultId && experimentId) {
      // key 包含类型，同一结果可以同时打开 html 和 waveform
      const key = `${experimentId}-${resultId}-${type}`;

      // 检查是否已存在该标签页
      const existingTab = tabs.find(t => t.key === key);
      if (!existingTab) {
        // 添加新标签页
        const newTab: TabItem = {
          key,
          label: type === 'waveform' ? `${label} [波形]` : label,
          resultId: parseInt(resultId),
          experimentId: parseInt(experimentId),
          type,
        };
        const newTabs = [...tabs, newTab];
        setTabs(newTabs);
        saveTabs(newTabs);
      }

      // 设置为当前活动标签页
      setActiveKey(key);
      saveActiveKey(key);
      // 清除 URL 参数
      setSearchParams({});
    } else if (!activeKey && tabs.length > 0) {
      // 没有URL参数时，如果activeKey为空，使用最后一个标签页
      const lastKey = tabs[tabs.length - 1].key;
      setActiveKey(lastKey);
      saveActiveKey(lastKey);
    }
  }, [searchParams, tabs.length]);

  // 切换标签页时保存activeKey
  const handleTabChange = (key: string) => {
    setActiveKey(key);
    saveActiveKey(key);
  };

  // 关闭标签页
  const handleTabClose = (targetKey: string) => {
    const newTabs = tabs.filter(t => t.key !== targetKey);
    setTabs(newTabs);
    saveTabs(newTabs);

    // 如果关闭的是当前激活的标签页，切换到其他标签
    if (activeKey === targetKey && newTabs.length > 0) {
      const newKey = newTabs[newTabs.length - 1].key;
      setActiveKey(newKey);
      saveActiveKey(newKey);
    } else if (newTabs.length === 0) {
      setActiveKey('');
      saveActiveKey('');
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

  // 渲染标签页内容
  const renderTabContent = (tab: TabItem) => {
    if (tab.type === 'waveform') {
      return (
        <div style={{ height: 'calc(100vh - 160px)', overflow: 'auto', padding: 16 }}>
          <WaveformViewer
            key={`${tab.key}-${refreshKey}`}
            experimentId={tab.experimentId}
            resultId={tab.resultId}
          />
        </div>
      );
    }
    // HTML 报告
    return (
      <div style={{ height: 'calc(100vh - 160px)', position: 'relative', overflow: 'auto' }}>
        <iframe
          key={`${tab.key}-${refreshKey}`}
          src={getResultHtmlUrl(tab.resultId, tab.experimentId)}
          style={{
            width: `${100 / zoom}%`,
            height: `${100 / zoom}%`,
            border: 'none',
            borderRadius: 4,
            transform: `scale(${zoom})`,
            transformOrigin: 'top left',
          }}
          title={tab.label}
        />
      </div>
    );
  };

  // 生成标签页项
  const tabItems = tabs.map(tab => ({
    key: tab.key,
    label: (
      <Space size={4} style={{ userSelect: 'none' }}>
        {tab.type === 'waveform' ? <LineChartOutlined /> : <FileTextOutlined />}
        <span>{tab.label}</span>
      </Space>
    ),
    children: renderTabContent(tab),
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
      <Tabs
        type="editable-card"
        hideAdd
        activeKey={activeKey}
        onChange={handleTabChange}
        onEdit={handleTabEdit}
        items={tabItems}
        tabBarStyle={{ marginBottom: 0 }}
        tabBarExtraContent={
          <Space>
            {/* 缩放控件只对HTML报告显示 */}
            {currentTab?.type !== 'waveform' && (
              <>
                <Tooltip title="缩小">
                  <Button
                    icon={<ZoomOutOutlined />}
                    onClick={() => setZoom(z => Math.max(0.5, z - 0.1))}
                    disabled={zoom <= 0.5}
                  />
                </Tooltip>
                <span style={{ minWidth: 45, textAlign: 'center' }}>{Math.round(zoom * 100)}%</span>
                <Tooltip title="放大">
                  <Button
                    icon={<ZoomInOutlined />}
                    onClick={() => setZoom(z => Math.min(1, z + 0.1))}
                    disabled={zoom >= 1}
                  />
                </Tooltip>
              </>
            )}
            <Tooltip title="刷新">
              <Button
                icon={<ReloadOutlined />}
                onClick={handleRefresh}
                disabled={!currentTab}
              />
            </Tooltip>
            {/* 新窗口打开只对HTML报告有效 */}
            {currentTab?.type !== 'waveform' && (
              <Tooltip title="在新窗口打开">
                <Button
                  icon={<ExpandOutlined />}
                  onClick={() => currentTab && handleOpenInNewWindow(currentTab)}
                  disabled={!currentTab}
                />
              </Tooltip>
            )}
          </Space>
        }
      />
    </div>
  );
}
