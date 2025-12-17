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

  // board 层级（chip 视图）是最底层，多层级视图显示上一级的 rack_board
  // 因为 chip 没有下一层，所以 board_chip 组合只在 rack 层级时可用
  if (currentLevel === 'rack' && hasCurrentBoard) {
    options.push({
      label: LEVEL_PAIR_NAMES.board_chip,
      value: 'board_chip',
    })
  }

  // 当在 board 层级（chip 视图）时，多层级显示 rack_board（向上一级）
  if (currentLevel === 'board' && hasCurrentRack) {
    // 如果还没有 rack_board 选项，添加它
    if (!options.some(o => o.value === 'rack_board')) {
      options.push({
        label: LEVEL_PAIR_NAMES.rack_board,
        value: 'rack_board',
      })
    }
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
