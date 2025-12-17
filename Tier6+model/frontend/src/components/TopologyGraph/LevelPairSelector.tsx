import React from 'react'
import { Segmented, Typography } from 'antd'
import { AdjacentLevelPair, LEVEL_PAIR_NAMES } from '../../types'

const { Text } = Typography

interface LevelPairSelectorProps {
  value: AdjacentLevelPair | null
  onChange: (pair: AdjacentLevelPair | null) => void
  disabled?: boolean
  currentLevel: string
  hasCurrentPod: boolean
  hasCurrentRack: boolean
  hasCurrentBoard: boolean
}

// 根据当前层级获取可用的层级组合选项
function getAvailableOptions(
  currentLevel: string,
  hasCurrentPod: boolean,
  hasCurrentRack: boolean,
  hasCurrentBoard: boolean
): { label: string; value: string; disabled?: boolean }[] {
  const options: { label: string; value: string; disabled?: boolean }[] = [
    { label: '单层级', value: 'single' },
  ]

  // 根据当前层级和上下文决定可用选项
  if (currentLevel === 'datacenter') {
    options.push({ label: LEVEL_PAIR_NAMES.datacenter_pod, value: 'datacenter_pod' })
  }

  if (currentLevel === 'pod' || (currentLevel === 'datacenter' && hasCurrentPod)) {
    options.push({
      label: LEVEL_PAIR_NAMES.pod_rack,
      value: 'pod_rack',
      disabled: currentLevel === 'datacenter' && !hasCurrentPod,
    })
  }

  if (currentLevel === 'rack' || (currentLevel === 'pod' && hasCurrentRack)) {
    options.push({
      label: LEVEL_PAIR_NAMES.rack_board,
      value: 'rack_board',
      disabled: currentLevel === 'pod' && !hasCurrentRack,
    })
  }

  if (currentLevel === 'board' || (currentLevel === 'rack' && hasCurrentBoard)) {
    options.push({
      label: LEVEL_PAIR_NAMES.board_chip,
      value: 'board_chip',
      disabled: currentLevel === 'rack' && !hasCurrentBoard,
    })
  }

  return options
}

export const LevelPairSelector: React.FC<LevelPairSelectorProps> = ({
  value,
  onChange,
  disabled = false,
  currentLevel,
  hasCurrentPod,
  hasCurrentRack,
  hasCurrentBoard,
}) => {
  const options = getAvailableOptions(currentLevel, hasCurrentPod, hasCurrentRack, hasCurrentBoard)

  const handleChange = (val: string | number) => {
    if (val === 'single') {
      onChange(null)
    } else {
      onChange(val as AdjacentLevelPair)
    }
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <Text style={{ fontSize: 12, color: '#666', whiteSpace: 'nowrap' }}>视图模式:</Text>
      <Segmented
        size="small"
        options={options}
        value={value || 'single'}
        onChange={handleChange}
        disabled={disabled}
      />
    </div>
  )
}
