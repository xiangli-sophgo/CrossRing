/**
 * 参数筛选器组件
 */

import { Slider, Button, Space, Typography } from 'antd';
import { useExperimentStore } from '../stores/experimentStore';

const { Text } = Typography;

interface Props {
  paramKeys: string[];
  parameterRanges: Record<string, { min: number; max: number }>;
}

export default function ParameterFilter({ paramKeys, parameterRanges }: Props) {
  const { filters, updateFilter, clearFilters } = useExperimentStore();

  // 只显示有数据的参数
  const availableParams = paramKeys.filter((param) => parameterRanges[param]);

  if (availableParams.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: 20 }}>
        <Text type="secondary">无参数数据</Text>
      </div>
    );
  }

  return (
    <div>
      <Space direction="vertical" style={{ width: '100%' }}>
        {availableParams.map((param) => {
          const range = parameterRanges[param];
          if (!range) return null;

          const currentFilter = filters[param];
          const value = currentFilter || [range.min, range.max];

          return (
            <div key={param} style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text strong style={{ fontSize: 12 }}>{param}</Text>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {value[0]} - {value[1]}
                </Text>
              </div>
              <Slider
                range
                min={range.min}
                max={range.max}
                value={value as [number, number]}
                onChange={(val) => {
                  if (val[0] === range.min && val[1] === range.max) {
                    updateFilter(param, null);
                  } else {
                    updateFilter(param, val as [number, number]);
                  }
                }}
                marks={{
                  [range.min]: range.min,
                  [range.max]: range.max,
                }}
              />
            </div>
          );
        })}

        <Button
          block
          onClick={clearFilters}
          disabled={Object.keys(filters).length === 0}
        >
          清除筛选
        </Button>
      </Space>
    </div>
  );
}
