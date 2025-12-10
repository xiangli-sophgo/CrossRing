import React, { useMemo, useRef, useState, useCallback } from 'react'
import { Canvas, useFrame, ThreeEvent, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Text, Line, Html } from '@react-three/drei'
import { Breadcrumb, Button, Tooltip } from 'antd'
import {
  ArrowLeftOutlined,
  CloudServerOutlined,
  ClusterOutlined,
  DatabaseOutlined,
  CreditCardOutlined,
  ApartmentOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import Icon from '@ant-design/icons'

// 自定义芯片图标 - UXWing专业图标
const ChipSvg = () => (
  <svg viewBox="0 0 122.88 122.88" width="1em" height="1em" fill="currentColor">
    <path d="M28.7,122.88h11.03v-13.4H28.7V122.88L28.7,122.88z M22.67,19.51h74.76c2.56,0,4.66,2.09,4.66,4.66v75.01 c0,2.56-2.1,4.66-4.66,4.66l-74.76,0c-2.56,0-4.66-2.1-4.66-4.66V24.16C18.01,21.6,20.1,19.51,22.67,19.51L22.67,19.51L22.67,19.51 z M42.35,41.29h35.38c1.55,0,2.81,1.27,2.81,2.81v35.12c0,1.55-1.27,2.81-2.81,2.81H42.35c-1.55,0-2.81-1.27-2.81-2.81V44.1 C39.54,42.56,40.8,41.29,42.35,41.29L42.35,41.29z M122.88,65.62v9.16h-13.4v-9.16H122.88L122.88,65.62z M122.88,48.1v9.16l-13.4,0 V48.1L122.88,48.1L122.88,48.1L122.88,48.1z M122.88,83.15v11.03h-13.4V83.15H122.88L122.88,83.15z M122.88,28.7v11.03h-13.4V28.7 H122.88L122.88,28.7z M0,65.62v9.16h13.4v-9.16H0L0,65.62z M0,48.1v9.16l13.4,0V48.1L0,48.1L0,48.1z M0,83.15v11.03h13.4V83.15H0 L0,83.15z M0,28.7v11.03h13.4V28.7H0L0,28.7z M65.62,0h9.16v13.4h-9.16V0L65.62,0L65.62,0z M48.1,0h9.16v13.4H48.1V0L48.1,0L48.1,0 z M83.15,0h11.03v13.4H83.15V0L83.15,0L83.15,0z M28.7,0h11.03v13.4H28.7V0L28.7,0L28.7,0z M65.62,122.88h9.16v-13.4h-9.16V122.88 L65.62,122.88z M48.1,122.88h9.16v-13.4H48.1V122.88L48.1,122.88z M83.15,122.88h11.03v-13.4H83.15V122.88L83.15,122.88z"/>
  </svg>
)
const ChipIcon = () => <Icon component={ChipSvg} />

// 自定义PCB板卡图标 - UXWing主板图标
const BoardSvg = () => (
  <svg viewBox="0 0 122.88 117.61" width="1em" height="1em" fill="currentColor">
    <path d="M71.39,103.48h3.64v-6.8h-3.64V103.48L71.39,103.48L71.39,103.48z M6.03,0h110.81c1.65,0,3.16,0.68,4.25,1.77 c1.1,1.1,1.78,2.61,1.78,4.26v105.54c0,1.66-0.68,3.17-1.77,4.27c-1.09,1.09-2.6,1.77-4.26,1.77H6.03c-1.66,0-3.17-0.68-4.26-1.77 S0,113.23,0,111.57V6.03c0-1.65,0.68-3.16,1.78-4.26C2.88,0.68,4.38,0,6.03,0L6.03,0z M115.35,7.53H7.53v46.73h10.68v-4.04 c-0.1-0.04-0.19-0.1-0.27-0.15c-0.16-0.1-0.31-0.22-0.44-0.34l-0.01-0.02c-0.23-0.22-0.41-0.5-0.54-0.8 c-0.12-0.3-0.19-0.63-0.19-0.97c0-0.35,0.07-0.67,0.19-0.97l0,0c0.13-0.31,0.32-0.59,0.55-0.84c0.23-0.23,0.51-0.42,0.82-0.55 c0.31-0.12,0.63-0.19,0.97-0.19c0.35,0,0.67,0.07,0.97,0.19c0.31,0.13,0.59,0.32,0.83,0.55c0.23,0.24,0.42,0.51,0.55,0.82 l0.01,0.02c0.12,0.29,0.19,0.62,0.19,0.95c0,0.34-0.07,0.67-0.19,0.97c-0.13,0.31-0.32,0.59-0.55,0.82l-0.02,0.02 c-0.12,0.11-0.24,0.22-0.38,0.31c-0.08,0.05-0.15,0.1-0.23,0.13v5.21c0,0.31-0.13,0.6-0.33,0.8c-0.21,0.2-0.49,0.33-0.8,0.33H7.53 v3.4h16.61l0,0c0.04-0.09,0.09-0.17,0.14-0.25l0.01-0.01c0.1-0.15,0.21-0.28,0.33-0.4c0.24-0.23,0.51-0.42,0.83-0.55l0.02-0.01 c0.3-0.12,0.62-0.19,0.95-0.19c0.34,0,0.67,0.07,0.97,0.19c0.31,0.13,0.6,0.32,0.82,0.55c0.23,0.23,0.42,0.51,0.55,0.82 c0.12,0.31,0.19,0.63,0.19,0.97c0,0.35-0.07,0.67-0.19,0.97c-0.13,0.31-0.32,0.59-0.55,0.82c-0.24,0.23-0.51,0.42-0.82,0.55 l-0.02,0.01c-0.3,0.12-0.62,0.19-0.95,0.19c-0.35,0-0.67-0.07-0.97-0.19c-0.31-0.13-0.59-0.32-0.83-0.55 c-0.12-0.12-0.24-0.26-0.33-0.42c-0.05-0.08-0.1-0.17-0.14-0.25H7.53v3.92h18.76l0,0c0.31,0,0.59,0.13,0.8,0.33 c0.21,0.2,0.33,0.48,0.33,0.8v8.39c0.1,0.04,0.21,0.1,0.31,0.16c0.17,0.11,0.34,0.23,0.48,0.38c0.23,0.24,0.42,0.51,0.55,0.82 l0.01,0.02c0.12,0.3,0.18,0.62,0.18,0.95c0,0.34-0.07,0.67-0.19,0.97c-0.13,0.31-0.32,0.59-0.55,0.84 c-0.23,0.23-0.51,0.42-0.82,0.55c-0.31,0.12-0.63,0.19-0.97,0.19c-0.34,0-0.67-0.07-0.97-0.19c-0.31-0.13-0.59-0.32-0.82-0.55 c-0.23-0.23-0.42-0.51-0.55-0.82c-0.12-0.3-0.19-0.63-0.19-0.97c0-0.35,0.07-0.67,0.19-0.97c0.13-0.31,0.32-0.6,0.55-0.83 c0.11-0.11,0.22-0.2,0.35-0.29c0.06-0.04,0.13-0.08,0.19-0.12v-7.38H7.53v3.2h9.89l0,0c0.31,0,0.59,0.13,0.79,0.33 c0.21,0.21,0.33,0.49,0.33,0.8v4.49c0.08,0.04,0.16,0.08,0.24,0.13c0.15,0.1,0.29,0.21,0.41,0.33l0.02,0.02 c0.22,0.23,0.4,0.51,0.53,0.81c0.12,0.3,0.19,0.63,0.19,0.97c0,0.35-0.07,0.67-0.19,0.97l-0.01,0.02 c-0.13,0.31-0.31,0.58-0.54,0.81l-0.02,0.01c-0.23,0.23-0.51,0.41-0.81,0.54C18.07,81.93,17.74,82,17.4,82 c-0.34,0-0.67-0.07-0.97-0.19c-0.31-0.13-0.6-0.32-0.82-0.55l-0.02-0.02c-0.22-0.23-0.4-0.51-0.53-0.81 c-0.12-0.31-0.19-0.63-0.19-0.97c0-0.34,0.07-0.66,0.19-0.96c0.13-0.31,0.32-0.6,0.55-0.82l0,0c0.13-0.14,0.27-0.25,0.43-0.35 c0.08-0.06,0.17-0.1,0.26-0.15v-3.34H7.53v36.24h107.82V7.53L115.35,7.53z"/>
  </svg>
)
const BoardIcon = () => <Icon component={BoardSvg} />
import { TopologyGraph } from './TopologyGraph'
import {
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  ChipConfig,
  ConnectionConfig,
  ViewState,
  BreadcrumbItem,
  ViewLevel,
  ChipType,
  RACK_DIMENSIONS,
  BOARD_DIMENSIONS,
  CHIP_DIMENSIONS,
  CHIP_TYPE_COLORS,
  CHIP_TYPE_NAMES,
  CAMERA_PRESETS,
  CAMERA_DISTANCE,
} from '../types'
import * as THREE from 'three'

// ============================================
// Props 接口定义
// ============================================

interface Scene3DProps {
  topology: HierarchicalTopology | null
  viewState: ViewState
  breadcrumbs: BreadcrumbItem[]
  currentPod: PodConfig | null
  currentRack: RackConfig | null
  currentBoard: BoardConfig | null
  onNavigate: (nodeId: string) => void
  onNavigateToPod: (podId: string) => void
  onNavigateToRack: (podId: string, rackId: string) => void
  onNavigateBack: () => void
  onBreadcrumbClick: (index: number) => void
  canGoBack: boolean
}

// ============================================
// 3D模型组件
// ============================================

// Chip模型 - 高端拟物风格，带有文字标识和精致外观
const ChipModel: React.FC<{
  chip: ChipConfig
  baseY?: number  // 底板高度
  totalChips?: number  // 总芯片数，用于计算居中
  onClick?: () => void
  onPointerOver?: () => void
  onPointerOut?: () => void
}> = ({ chip, baseY = 0, totalChips = 8, onClick, onPointerOver, onPointerOut }) => {
  const [hovered, setHovered] = useState(false)
  const baseDimensions = CHIP_DIMENSIONS[chip.type] || [0.06, 0.02, 0.06]
  // 芯片尺寸 - 稍微缩小
  const dimensions: [number, number, number] = [baseDimensions[0] * 0.9, baseDimensions[1] * 1.2, baseDimensions[2] * 0.9]

  // 智能计算网格大小（与后端保持一致）
  const cols = Math.ceil(Math.sqrt(totalChips))
  const rows = Math.ceil(totalChips / cols)

  // 当前行号和列号
  const row = chip.position[0]
  const col = chip.position[1]

  // 计算当前行有多少芯片（最后一行可能不满）
  const chipsInCurrentRow = row < rows - 1 ? cols : totalChips - (rows - 1) * cols

  // 计算居中偏移 - 缩小间距
  const spacing = 0.11
  const rowCenterOffset = (chipsInCurrentRow - 1) / 2
  const x = (col - rowCenterOffset) * spacing
  const z = (row - (rows - 1) / 2) * spacing
  const y = baseY + dimensions[1] / 2

  // 芯片类型对应的标签文字
  const chipLabel = chip.type === 'npu' ? 'NPU' : 'CPU'
  // 深色金属外壳颜色
  const shellColor = '#1a1a1a'
  const shellColorHover = '#2a2a2a'
  // 顶部标识颜色 - 根据类型
  const labelColor = chip.type === 'npu' ? '#4fc3f7' : '#81c784'

  return (
    <group position={[x, y, z]}>
      {/* 芯片主体 - 深色哑光外壳 */}
      <mesh
        onClick={onClick}
        onPointerOver={(e) => {
          e.stopPropagation()
          setHovered(true)
          onPointerOver?.()
        }}
        onPointerOut={(e) => {
          e.stopPropagation()
          setHovered(false)
          onPointerOut?.()
        }}
        castShadow
      >
        <boxGeometry args={dimensions} />
        <meshStandardMaterial
          color={hovered ? shellColorHover : shellColor}
          metalness={0.3}
          roughness={0.8}
        />
      </mesh>

      {/* 芯片顶部内嵌区域 */}
      <mesh position={[0, dimensions[1] / 2 - 0.001, 0]}>
        <boxGeometry args={[dimensions[0] * 0.92, 0.002, dimensions[2] * 0.92]} />
        <meshStandardMaterial
          color="#0d0d0d"
          metalness={0.2}
          roughness={0.9}
        />
      </mesh>

      {/* 芯片上的电路纹理装饰 - 细线条 */}
      {Array.from({ length: 3 }).map((_, i) => (
        <mesh key={`circuit-h-${i}`} position={[0, dimensions[1] / 2 + 0.001, (i - 1) * dimensions[2] * 0.25]}>
          <boxGeometry args={[dimensions[0] * 0.7, 0.001, 0.002]} />
          <meshStandardMaterial color="#333" metalness={0.2} roughness={0.8} />
        </mesh>
      ))}
      {Array.from({ length: 3 }).map((_, i) => (
        <mesh key={`circuit-v-${i}`} position={[(i - 1) * dimensions[0] * 0.25, dimensions[1] / 2 + 0.001, 0]}>
          <boxGeometry args={[0.002, 0.001, dimensions[2] * 0.7]} />
          <meshStandardMaterial color="#333" metalness={0.2} roughness={0.8} />
        </mesh>
      ))}

      {/* 芯片标识文字 */}
      <Text
        position={[0, dimensions[1] / 2 + 0.002, 0]}
        fontSize={dimensions[0] * 0.35}
        color={labelColor}
        anchorX="center"
        anchorY="middle"
        rotation={[-Math.PI / 2, 0, 0]}
        material-depthTest={false}
      >
        {chipLabel}
      </Text>

      {/* 边缘引脚装饰 - 四边 */}
      {/* 左右两侧引脚 */}
      {Array.from({ length: 6 }).map((_, i) => (
        <React.Fragment key={`pin-lr-${i}`}>
          <mesh position={[-dimensions[0] / 2 - 0.003, 0, (i - 2.5) * dimensions[2] / 6]}>
            <boxGeometry args={[0.006, dimensions[1] * 0.3, 0.004]} />
            <meshStandardMaterial color="#a0a0a0" metalness={0.4} roughness={0.6} />
          </mesh>
          <mesh position={[dimensions[0] / 2 + 0.003, 0, (i - 2.5) * dimensions[2] / 6]}>
            <boxGeometry args={[0.006, dimensions[1] * 0.3, 0.004]} />
            <meshStandardMaterial color="#a0a0a0" metalness={0.4} roughness={0.6} />
          </mesh>
        </React.Fragment>
      ))}
      {/* 前后两侧引脚 */}
      {Array.from({ length: 6 }).map((_, i) => (
        <React.Fragment key={`pin-fb-${i}`}>
          <mesh position={[(i - 2.5) * dimensions[0] / 6, 0, -dimensions[2] / 2 - 0.003]}>
            <boxGeometry args={[0.004, dimensions[1] * 0.3, 0.006]} />
            <meshStandardMaterial color="#a0a0a0" metalness={0.4} roughness={0.6} />
          </mesh>
          <mesh position={[(i - 2.5) * dimensions[0] / 6, 0, dimensions[2] / 2 + 0.003]}>
            <boxGeometry args={[0.004, dimensions[1] * 0.3, 0.006]} />
            <meshStandardMaterial color="#a0a0a0" metalness={0.4} roughness={0.6} />
          </mesh>
        </React.Fragment>
      ))}

      {/* 悬停时显示详细信息 */}
      {hovered && (
        <Html center position={[0, 0.06, 0]}>
          <div style={{
            background: 'rgba(0,0,0,0.9)',
            color: '#fff',
            padding: '6px 12px',
            borderRadius: 6,
            fontSize: 12,
            whiteSpace: 'nowrap',
            border: `1px solid ${labelColor}`,
          }}>
            {CHIP_TYPE_NAMES[chip.type]}
            {chip.label && ` - ${chip.label}`}
          </div>
        </Html>
      )}
    </group>
  )
}

// 不同U高度板卡的配色方案
const BOARD_U_COLORS: Record<number, { main: string; mainHover: string; front: string; accent: string }> = {
  1: { main: '#4a5568', mainHover: '#38b2ac', front: '#2d3748', accent: '#63b3ed' },  // 灰蓝色 - 1U交换机/轻量设备
  2: { main: '#2c5282', mainHover: '#38b2ac', front: '#1a365d', accent: '#90cdf4' },  // 深蓝色 - 2U标准服务器
  4: { main: '#553c9a', mainHover: '#38b2ac', front: '#322659', accent: '#b794f4' },  // 紫色 - 4U GPU服务器
}

// Board模型 - 服务器/板卡，根据U高度显示不同样式
const BoardModel: React.FC<{
  board: BoardConfig
  showChips?: boolean
  compact?: boolean  // 紧凑模式（在Rack外部视图使用）
  interactive?: boolean  // 是否可以交互（高亮和点击）
  onDoubleClick?: () => void
}> = ({ board, showChips = false, compact = false, interactive = true, onDoubleClick }) => {
  const [hovered, setHovered] = useState(false)
  const canHover = interactive && !compact  // 只有可交互且非紧凑模式才能高亮

  // 根据U高度获取颜色方案
  const uHeight = board.u_height
  const colorScheme = BOARD_U_COLORS[uHeight] || BOARD_U_COLORS[2]

  // 根据U高度计算实际3D尺寸
  const { uHeight: uSize } = RACK_DIMENSIONS
  const width = compact ? BOARD_DIMENSIONS.width * 0.85 : BOARD_DIMENSIONS.width
  const height = uHeight * uSize * (compact ? 0.85 : 0.9)  // 留一点间隙
  const depth = compact ? BOARD_DIMENSIONS.depth * 0.8 : BOARD_DIMENSIONS.depth

  // 颜色 - 只有canHover为true时才应用高亮效果
  const isHighlighted = canHover && hovered
  const mainColor = isHighlighted ? colorScheme.mainHover : colorScheme.main
  const frontColor = colorScheme.front

  // 边框厚度
  const wallThickness = 0.01

  return (
    <group>
      {showChips ? (
        // 显示芯片时：拟物化PCB板卡
        <>
          {/* PCB基板 - 多层结构 */}
          {/* 底层 - FR4基材 */}
          <mesh position={[0, -0.002, 0]} castShadow receiveShadow>
            <boxGeometry args={[width, 0.004, depth]} />
            <meshStandardMaterial color="#1a4d2e" metalness={0.1} roughness={0.9} />
          </mesh>
          {/* 中间层 - 主PCB */}
          <mesh position={[0, 0.001, 0]} castShadow receiveShadow>
            <boxGeometry args={[width - 0.002, 0.004, depth - 0.002]} />
            <meshStandardMaterial color="#0f3d1f" metalness={0.15} roughness={0.85} />
          </mesh>
          {/* 顶层 - 阻焊层(绿油) */}
          <mesh position={[0, 0.0035, 0]}>
            <boxGeometry args={[width - 0.004, 0.001, depth - 0.004]} />
            <meshStandardMaterial color="#0a2f18" metalness={0.1} roughness={0.7} />
          </mesh>

          {/* 铜走线 - 主要信号线（减少数量提升性能） */}
          {Array.from({ length: 6 }).map((_, i) => {
            const zPos = -depth / 2 + 0.05 + i * (depth / 7)
            const lineWidth = i % 2 === 0 ? 0.004 : 0.002
            return (
              <mesh key={`trace-h-${i}`} position={[0, 0.0042, zPos]}>
                <boxGeometry args={[width - 0.06, 0.0008, lineWidth]} />
                <meshStandardMaterial color="#c9a227" metalness={0.6} roughness={0.4} />
              </mesh>
            )
          })}
          {/* 纵向走线 */}
          {Array.from({ length: 5 }).map((_, i) => {
            const xPos = -width / 2 + 0.06 + i * (width / 6)
            const lineWidth = i % 2 === 0 ? 0.003 : 0.0015
            return (
              <mesh key={`trace-v-${i}`} position={[xPos, 0.0042, 0]}>
                <boxGeometry args={[lineWidth, 0.0008, depth - 0.06]} />
                <meshStandardMaterial color="#c9a227" metalness={0.6} roughness={0.4} />
              </mesh>
            )
          })}

          {/* 过孔(Via) - 减少数量 */}
          {Array.from({ length: 8 }).map((_, i) => {
            const viaX = (Math.sin(i * 2.5) * 0.35) * width / 2
            const viaZ = (Math.cos(i * 3.1) * 0.35) * depth / 2
            return (
              <mesh key={`via-${i}`} position={[viaX, 0.0043, viaZ]} rotation={[-Math.PI / 2, 0, 0]}>
                <cylinderGeometry args={[0.003, 0.003, 0.001, 6]} />
                <meshStandardMaterial color="#b8860b" metalness={0.7} roughness={0.3} />
              </mesh>
            )
          })}

          {/* 边缘金手指接口 */}
          <mesh position={[0, 0, -depth / 2 + 0.012]}>
            <boxGeometry args={[width * 0.6, 0.005, 0.018]} />
            <meshStandardMaterial color="#b8923a" metalness={0.3} roughness={0.7} />
          </mesh>

          {/* 安装孔 - 四角 */}
          {[[-1, -1], [-1, 1], [1, -1], [1, 1]].map(([dx, dz], i) => (
            <mesh key={`mount-${i}`} position={[dx * (width / 2 - 0.025), 0.0042, dz * (depth / 2 - 0.025)]} rotation={[-Math.PI / 2, 0, 0]}>
              <circleGeometry args={[0.01, 16]} />
              <meshStandardMaterial color="#8a7040" metalness={0.2} roughness={0.8} />
            </mesh>
          ))}

          {/* 渲染芯片 - 放在PCB上，居中排布 */}
          {board.chips.map(chip => (
            <ChipModel key={chip.id} chip={chip} baseY={0.004} totalChips={board.chips.length} />
          ))}

          {/* 板卡丝印标识 */}
          <Text
            position={[width / 2 - 0.06, 0.005, depth / 2 - 0.03]}
            fontSize={0.015}
            color="#c0c0c0"
            anchorX="center"
            anchorY="middle"
            rotation={[-Math.PI / 2, 0, 0]}
            material-depthTest={false}
          >
            {board.label}
          </Text>
          {/* 版本号丝印 */}
          <Text
            position={[-width / 2 + 0.05, 0.005, depth / 2 - 0.03]}
            fontSize={0.008}
            color="#888888"
            anchorX="center"
            anchorY="middle"
            rotation={[-Math.PI / 2, 0, 0]}
            material-depthTest={false}
          >
            REV 1.0
          </Text>
        </>
      ) : (
        // 不显示芯片时：封闭的服务器盒子
        <>
          {/* 服务器主体 - 金属外壳 */}
          <mesh
            onDoubleClick={canHover ? onDoubleClick : undefined}
            onPointerOver={canHover ? (e) => {
              e.stopPropagation()
              setHovered(true)
            } : undefined}
            onPointerOut={canHover ? (e) => {
              e.stopPropagation()
              setHovered(false)
            } : undefined}
            castShadow
            receiveShadow
          >
            <boxGeometry args={[width, height, depth]} />
            <meshStandardMaterial
              color={mainColor}
              emissive={isHighlighted ? '#38b2ac' : '#000000'}
              emissiveIntensity={isHighlighted ? 0.4 : 0}
              metalness={compact ? 0.5 : 0.7}
              roughness={compact ? 0.4 : 0.3}
            />
          </mesh>

          {/* 前面板 - 带有指示灯效果 */}
          <mesh position={[0, 0, depth / 2 + 0.001]}>
            <boxGeometry args={[width - 0.02, height - 0.005, 0.002]} />
            <meshStandardMaterial
              color={frontColor}
              metalness={0.5}
              roughness={0.5}
            />
          </mesh>

          {/* U高度标识条 - 左侧彩色条纹 */}
          <mesh position={[-width / 2 + 0.008, 0, depth / 2 + 0.002]}>
            <boxGeometry args={[0.012, height - 0.01, 0.001]} />
            <meshBasicMaterial color={colorScheme.accent} />
          </mesh>

          {/* LED指示灯 */}
          <mesh position={[-width / 2 + 0.03, height / 2 - 0.015, depth / 2 + 0.003]}>
            <circleGeometry args={[compact ? 0.004 : 0.006, 16]} />
            <meshBasicMaterial color="#52c41a" />
          </mesh>
          <mesh position={[-width / 2 + 0.045, height / 2 - 0.015, depth / 2 + 0.003]}>
            <circleGeometry args={[compact ? 0.004 : 0.006, 16]} />
            <meshBasicMaterial color={colorScheme.accent} />
          </mesh>

          {/* 板卡标签 - 调整位置确保不被遮挡 */}
          <Text
            position={[0, 0, depth / 2 + 0.015]}
            fontSize={compact ? 0.02 : 0.035}
            color="#ffffff"
            anchorX="center"
            anchorY="middle"
            outlineWidth={0.003}
            outlineColor="#000000"
            renderOrder={1}
            material-depthTest={false}
          >
            {board.label}
          </Text>
        </>
      )}
    </group>
  )
}

// Rack模型 - 拟物化42U机柜
const RackModel: React.FC<{
  rack: RackConfig
  showBoards?: boolean
  highlighted?: boolean  // 外部控制高亮（用于Pod高亮时联动）
  interactive?: boolean  // 是否允许Rack本身高亮
  simplified?: boolean   // 简化模式（远距离时减少细节提升性能）
  onClick?: () => void
  onDoubleClick?: () => void
  onBoardClick?: (boardId: string) => void
  onHoverChange?: (hovered: boolean) => void  // 悬停状态变化回调
}> = ({ rack, showBoards = false, highlighted = false, interactive = true, simplified = false, onClick, onDoubleClick, onBoardClick, onHoverChange }) => {
  const [hovered, setHovered] = useState(false)

  const { width, depth, uHeight, totalU } = RACK_DIMENSIONS
  const height = totalU * uHeight

  // 机柜框架粗细和颜色 - 只有interactive时才响应悬停
  const isHighlighted = (interactive && hovered) || highlighted
  const frameThickness = 0.02
  const frameColor = isHighlighted ? '#38b2ac' : '#333333'
  const topBottomColor = isHighlighted ? '#38b2ac' : '#1a1a1a'

  // 计算每个Board在机柜中的位置
  const getBoardPosition = (board: BoardConfig): [number, number, number] => {
    const y = (board.u_position - 1) * uHeight + (board.u_height * uHeight) / 2 - height / 2
    return [0, y, 0]
  }

  const handlePointerOver = () => {
    setHovered(true)
    onHoverChange?.(true)
  }

  const handlePointerOut = () => {
    setHovered(false)
    onHoverChange?.(false)
  }

  return (
    <group
      onClick={interactive ? onClick : undefined}
      onDoubleClick={onDoubleClick}
      onPointerOver={handlePointerOver}
      onPointerOut={handlePointerOut}
    >
      {/* 机柜底座 */}
      <mesh position={[0, -height / 2 - 0.02, 0]} receiveShadow>
        <boxGeometry args={[width + 0.04, 0.04, depth + 0.04]} />
        <meshStandardMaterial color={topBottomColor} metalness={0.3} roughness={0.7} />
      </mesh>

      {/* 机柜顶部 */}
      <mesh position={[0, height / 2 + 0.02, 0]} castShadow>
        <boxGeometry args={[width + 0.04, 0.04, depth + 0.04]} />
        <meshStandardMaterial color={topBottomColor} metalness={0.3} roughness={0.7} />
      </mesh>

      {/* 四个垂直立柱 */}
      {[
        [-width / 2, 0, -depth / 2],
        [width / 2, 0, -depth / 2],
        [-width / 2, 0, depth / 2],
        [width / 2, 0, depth / 2],
      ].map((pos, i) => (
        <mesh key={`pillar-${i}`} position={pos as [number, number, number]} castShadow>
          <boxGeometry args={[frameThickness, height, frameThickness]} />
          <meshStandardMaterial color={frameColor} metalness={0.3} roughness={0.7} />
        </mesh>
      ))}

      {/* U位刻度线 (每5U一条) - 仅在非简化模式下渲染 */}
      {!simplified && Array.from({ length: Math.floor(totalU / 5) + 1 }, (_, i) => i * 5).map(u => {
        const y = u * uHeight - height / 2
        return (
          <group key={`u-mark-${u}`}>
            <mesh position={[-width / 2 - 0.01, y, -depth / 2]}>
              <boxGeometry args={[0.01, 0.002, 0.02]} />
              <meshBasicMaterial color="#888888" />
            </mesh>
            <Text
              position={[-width / 2 - 0.03, y, -depth / 2]}
              fontSize={0.02}
              color="#888888"
              anchorX="right"
              anchorY="middle"
            >
              {u}U
            </Text>
          </group>
        )
      })}

      {/* 后面板 */}
      <mesh position={[0, 0, -depth / 2 + 0.005]} receiveShadow>
        <boxGeometry args={[width - frameThickness * 2, height, 0.01]} />
        <meshStandardMaterial
          color="#2a2a2a"
          metalness={0.3}
          roughness={0.7}
        />
      </mesh>

      {/* 前面板 - 半透明玻璃效果，根据是否显示Board详情调整透明度 */}
      <mesh position={[0, 0, depth / 2 - 0.005]}>
        <boxGeometry args={[width - frameThickness * 2, height, 0.008]} />
        <meshStandardMaterial
          color="#4a4a4a"
          transparent
          opacity={showBoards ? 0.05 : 0.1}
          metalness={0.9}
          roughness={0.1}
        />
      </mesh>

      {/* 机柜标签 - 文字 */}
      <Text
        position={[0, height / 2 + 0.12, 0.01]}
        fontSize={0.2}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        fontWeight="bold"
      >
        {rack.label}
      </Text>

      {/* 始终显示内部Boards */}
      {rack.boards.map(board => (
        <group key={board.id} position={getBoardPosition(board)}>
          <BoardModel
            board={board}
            compact={!showBoards}
            interactive={showBoards}
            onDoubleClick={() => onBoardClick?.(board.id)}
          />
        </group>
      ))}
    </group>
  )
}

// Pod标签组件 - 支持悬停高亮
const PodLabel: React.FC<{
  pod: PodConfig
  position: [number, number, number]
  onDoubleClick: () => void
  onHoverChange?: (hovered: boolean) => void
}> = ({ pod, position, onDoubleClick, onHoverChange }) => {
  const [hovered, setHovered] = useState(false)

  const handlePointerOver = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    setHovered(true)
    onHoverChange?.(true)
  }

  const handlePointerOut = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    setHovered(false)
    onHoverChange?.(false)
  }

  return (
    <group
      position={position}
      onDoubleClick={onDoubleClick}
      onPointerOver={handlePointerOver}
      onPointerOut={handlePointerOut}
    >
      {/* 背景板 */}
      <mesh>
        <planeGeometry args={[1.2, 0.4]} />
        <meshBasicMaterial
          color={hovered ? '#38b2ac' : '#1890ff'}
          transparent
          opacity={hovered ? 1 : 0.9}
        />
      </mesh>
      {/* 文字 */}
      <Text
        position={[0, 0, 0.01]}
        fontSize={0.2}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        fontWeight="bold"
      >
        {pod.label}
      </Text>
    </group>
  )
}

// ============================================
// 连接线组件
// ============================================

const ConnectionLine: React.FC<{
  connection: ConnectionConfig
  getNodePosition: (nodeId: string) => THREE.Vector3 | null
  elevate?: number  // 连线提升高度
}> = ({ connection, getNodePosition, elevate = 0 }) => {
  const sourcePos = getNodePosition(connection.source)
  const targetPos = getNodePosition(connection.target)

  // 必须有有效的位置
  if (!sourcePos || !targetPos) return null
  if (isNaN(sourcePos.x) || isNaN(targetPos.x)) return null

  const isInter = connection.type === 'inter'
  const color = isInter ? '#faad14' : '#52c41a'

  // 提升连线到顶部
  const elevatedSource = new THREE.Vector3(sourcePos.x, sourcePos.y + elevate, sourcePos.z)
  const elevatedTarget = new THREE.Vector3(targetPos.x, targetPos.y + elevate, targetPos.z)

  return (
    <Line
      points={[elevatedSource, elevatedTarget]}
      color={color}
      lineWidth={isInter ? 3 : 2}
      dashed={isInter}
      dashSize={0.1}
      gapSize={0.05}
    />
  )
}

// ============================================
// 视图容器组件
// ============================================

// Pod视图 - 显示所有Rack
const PodView: React.FC<{
  pods: PodConfig[]
  onPodClick: (podId: string) => void
  onRackDoubleClick: (podId: string, rackId: string) => void
  isDatacenterView?: boolean  // true: 整个Pod高亮; false: 单个Rack高亮
}> = ({ pods, onPodClick, onRackDoubleClick, isDatacenterView = true }) => {
  const [hoveredPodId, setHoveredPodId] = useState<string | null>(null)
  const [hoveredRackId, setHoveredRackId] = useState<string | null>(null)
  const rackSpacingX = 1.5
  const rackSpacingZ = 2

  // 计算Pod的尺寸和间距（根据Rack数量动态调整）
  const { podSpacingX, podSpacingZ, podCols } = useMemo(() => {
    // 获取第一个Pod的Rack布局来估算Pod尺寸
    const firstPod = pods[0]
    if (!firstPod) return { podSpacingX: 6, podSpacingZ: 4, podCols: 2 }

    // 计算Pod内Rack的列数和行数
    const rackCols = firstPod.grid_size[1]
    const rackRows = firstPod.grid_size[0]

    // Pod宽度 = Rack列数 * Rack间距X + 额外间隙
    const podWidth = rackCols * rackSpacingX + 2
    // Pod深度 = Rack行数 * Rack间距Z + 额外间隙（缩小前后距离）
    const podDepth = rackRows * rackSpacingZ + 1

    // 根据Pod数量选择列数
    const totalPods = pods.length
    let cols: number
    if (totalPods <= 2) cols = totalPods
    else if (totalPods <= 4) cols = 2
    else if (totalPods <= 6) cols = 3
    else if (totalPods <= 9) cols = 3
    else cols = 4

    return { podSpacingX: podWidth, podSpacingZ: podDepth, podCols: cols }
  }, [pods])

  // 计算Pod的网格布局
  const getPodGridPosition = (podIndex: number) => {
    const row = Math.floor(podIndex / podCols)
    const col = podIndex % podCols
    return { row, col }
  }

  // 计算Rack位置（居中）
  const rackPositions = useMemo(() => {
    const positions = new Map<string, THREE.Vector3>()

    // 首先计算所有Rack的原始位置，找出边界
    let minX = Infinity, maxX = -Infinity
    let minZ = Infinity, maxZ = -Infinity

    pods.forEach((pod, podIndex) => {
      const { row, col } = getPodGridPosition(podIndex)
      const podOffsetX = col * podSpacingX
      const podOffsetZ = row * podSpacingZ
      pod.racks.forEach(rack => {
        const x = podOffsetX + rack.position[1] * rackSpacingX
        const z = podOffsetZ + rack.position[0] * rackSpacingZ
        minX = Math.min(minX, x)
        maxX = Math.max(maxX, x)
        minZ = Math.min(minZ, z)
        maxZ = Math.max(maxZ, z)
      })
    })

    // 计算中心偏移
    const centerX = (minX + maxX) / 2
    const centerZ = (minZ + maxZ) / 2

    // 设置居中后的位置
    pods.forEach((pod, podIndex) => {
      const { row, col } = getPodGridPosition(podIndex)
      const podOffsetX = col * podSpacingX
      const podOffsetZ = row * podSpacingZ
      pod.racks.forEach(rack => {
        const x = podOffsetX + rack.position[1] * rackSpacingX - centerX
        const z = podOffsetZ + rack.position[0] * rackSpacingZ - centerZ
        positions.set(rack.id, new THREE.Vector3(x, 0, z))
      })
    })

    return positions
  }, [pods])

  // 计算每个Pod的中心位置
  const podCenters = useMemo(() => {
    const centers = new Map<string, THREE.Vector3>()
    pods.forEach((pod, podIndex) => {
      let sumX = 0, sumZ = 0, count = 0
      pod.racks.forEach(rack => {
        const pos = rackPositions.get(rack.id)
        if (pos) {
          sumX += pos.x
          sumZ += pos.z
          count++
        }
      })
      if (count > 0) {
        centers.set(pod.id, new THREE.Vector3(sumX / count, 0, sumZ / count))
      }
    })
    return centers
  }, [pods, rackPositions])

  return (
    <group>
      {/* 渲染所有Pod中的Rack */}
      {pods.map(pod => {
        const podCenter = podCenters.get(pod.id)
        const isPodHighlighted = hoveredPodId === pod.id
        return (
          <group key={pod.id}>
            {/* Pod标签 - 双击进入 */}
            {podCenter && (
              <PodLabel
                pod={pod}
                position={[podCenter.x, RACK_DIMENSIONS.totalU * RACK_DIMENSIONS.uHeight / 2 + 0.5, podCenter.z]}
                onDoubleClick={() => onPodClick(pod.id)}
                onHoverChange={(hovered) => setHoveredPodId(hovered ? pod.id : null)}
              />
            )}
            {pod.racks.map(rack => {
              const pos = rackPositions.get(rack.id)
              if (!pos) return null
              // 数据中心视图：整个Pod高亮; Pod视图：单个Rack高亮
              const isRackHighlighted = isDatacenterView
                ? isPodHighlighted
                : hoveredRackId === rack.id
              return (
                <group key={rack.id} position={[pos.x, pos.y, pos.z]}>
                  <RackModel
                    rack={rack}
                    highlighted={isRackHighlighted}
                    interactive={!isDatacenterView}
                    simplified={true}
                    onDoubleClick={() => onRackDoubleClick(pod.id, rack.id)}
                    onHoverChange={(hovered) => {
                      if (isDatacenterView) {
                        setHoveredPodId(hovered ? pod.id : null)
                      } else {
                        setHoveredRackId(hovered ? rack.id : null)
                      }
                    }}
                  />
                </group>
              )
            })}
          </group>
        )
      })}

      {/* 地面 */}
      <mesh
        position={[0, -RACK_DIMENSIONS.totalU * RACK_DIMENSIONS.uHeight / 2 - 0.06, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        receiveShadow
      >
        <planeGeometry args={[30, 30]} />
        <meshStandardMaterial color="#e8e8e8" />
      </mesh>
      {/* 地面网格线 */}
      <gridHelper
        args={[30, 30, '#bbb', '#ddd']}
        position={[0, -RACK_DIMENSIONS.totalU * RACK_DIMENSIONS.uHeight / 2 - 0.05, 0]}
      />
    </group>
  )
}

// Rack视图 - 显示单个Rack内部的Boards
const RackView: React.FC<{
  rack: RackConfig
  onBoardClick: (boardId: string) => void
}> = ({ rack, onBoardClick }) => {
  const { uHeight, totalU } = RACK_DIMENSIONS
  const height = totalU * uHeight

  return (
    <group>
      {/* 渲染Rack框架 - 只允许Board高亮，Rack不高亮 */}
      <RackModel
        rack={rack}
        showBoards={true}
        interactive={false}
        onBoardClick={onBoardClick}
      />

      {/* 地面 */}
      <mesh
        position={[0, -height / 2 - 0.06, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        receiveShadow
      >
        <planeGeometry args={[5, 5]} />
        <meshStandardMaterial color="#e8e8e8" />
      </mesh>
      <gridHelper
        args={[5, 10, '#bbb', '#ddd']}
        position={[0, -height / 2 - 0.05, 0]}
      />
    </group>
  )
}

// Board视图 - 显示单个Board上的Chips
const BoardView: React.FC<{
  board: BoardConfig
}> = ({ board }) => {
  return (
    <group>
      {/* Board基板 - 最底层，不需要交互 */}
      <BoardModel board={board} showChips={true} interactive={false} />
    </group>
  )
}

// ============================================
// 场景内容组件
// ============================================

const SceneContent: React.FC<{
  topology: HierarchicalTopology
  viewState: ViewState
  currentPod: PodConfig | null
  currentRack: RackConfig | null
  currentBoard: BoardConfig | null
  onNavigate: (nodeId: string) => void
  onNavigateToPod: (podId: string) => void
  onNavigateToRack: (podId: string, rackId: string) => void
}> = ({ topology, viewState, currentPod, currentRack, currentBoard, onNavigate, onNavigateToPod, onNavigateToRack }) => {
  // 根据当前视图层级渲染不同内容
  const renderView = () => {
    // 顶层视图：显示所有Pods（数据中心视图，整个Pod高亮）
    // 在顶层，双击Rack只进入对应的Pod（不直接跳到Rack内部）
    if (viewState.path.length === 0) {
      return (
        <PodView
          pods={topology.pods}
          onPodClick={onNavigateToPod}
          onRackDoubleClick={(podId) => onNavigateToPod(podId)}
          isDatacenterView={true}
        />
      )
    }

    // Pod内部视图：只显示该Pod的Racks（Pod视图，单个Rack高亮）
    if (viewState.path.length === 1 && currentPod) {
      return (
        <PodView
          pods={[currentPod]}
          onPodClick={() => {}}
          onRackDoubleClick={onNavigateToRack}
          isDatacenterView={false}
        />
      )
    }

    // Rack内部视图：显示Boards (path.length === 2)
    if (viewState.path.length === 2 && currentRack) {
      return (
        <RackView
          rack={currentRack}
          onBoardClick={onNavigate}
        />
      )
    }

    // Board视图：显示Chips (path.length >= 3)
    if (viewState.path.length >= 3 && currentBoard) {
      return (
        <BoardView
          board={currentBoard}
        />
      )
    }

    return null
  }

  return (
    <>
      {/* 灯光设置 */}
      <ambientLight intensity={0.4} />
      <directionalLight
        position={[10, 15, 10]}
        intensity={1}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      <directionalLight position={[-5, 10, -5]} intensity={0.3} />
      <pointLight position={[0, 5, 0]} intensity={0.5} />

      {/* 渲染当前视图 */}
      {renderView()}
    </>
  )
}

// ============================================
// 导航覆盖层组件
// ============================================

const NavigationOverlay: React.FC<{
  breadcrumbs: BreadcrumbItem[]
  onBreadcrumbClick: (index: number) => void
  onBack: () => void
  canGoBack: boolean
}> = ({ breadcrumbs, onBreadcrumbClick, onBack, canGoBack }) => {
  return (
    <div style={{
      position: 'absolute',
      top: 16,
      left: 16,
      zIndex: 100,
      display: 'flex',
      alignItems: 'center',
      gap: 12,
    }}>
      {/* 返回按钮 */}
      {canGoBack && (
        <Tooltip title="返回上级">
          <Button
            type="primary"
            icon={<ArrowLeftOutlined />}
            onClick={onBack}
          />
        </Tooltip>
      )}

      {/* 面包屑导航 */}
      <div style={{
        background: 'rgba(255, 255, 255, 0.95)',
        padding: '8px 16px',
        borderRadius: 8,
        boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
      }}>
        <Breadcrumb
          items={breadcrumbs.map((item, index) => {
            // 根据层级选择图标
            const getIcon = () => {
              if (index === 0) return <CloudServerOutlined />  // 数据中心
              switch (item.level) {
                case 'pod': return <ClusterOutlined />         // Pod - 集群图标
                case 'rack': return <DatabaseOutlined />       // Rack - 机柜图标
                case 'board': return <BoardIcon />             // Board - 板卡图标
                case 'chip': return <ChipIcon />               // Chip - 芯片图标
                default: return null
              }
            }
            return {
              title: (
                <a
                  onClick={(e) => {
                    e.preventDefault()
                    onBreadcrumbClick(index)
                  }}
                  style={{ cursor: 'pointer' }}
                >
                  {getIcon()}
                  {' '}{item.label}
                </a>
              ),
            }
          })}
        />
      </div>
    </div>
  )
}

// ============================================
// 主Scene3D组件
// ============================================

export const Scene3D: React.FC<Scene3DProps> = ({
  topology,
  viewState,
  breadcrumbs,
  currentPod,
  currentRack,
  currentBoard,
  onNavigate,
  onNavigateToPod,
  onNavigateToRack,
  onNavigateBack,
  onBreadcrumbClick,
  canGoBack,
}) => {
  const [showTopologyGraph, setShowTopologyGraph] = useState(false)
  const [cameraKey, setCameraKey] = useState(0)

  // 重置视图（相机位置）
  const handleResetView = useCallback(() => {
    setCameraKey(k => k + 1)
  }, [])

  // 获取当前视图的相机设置
  const cameraDistance = CAMERA_DISTANCE[viewState.level]

  // 根据Pod数量和Rack数量动态计算相机位置
  const cameraPreset = useMemo(() => {
    const basePreset = CAMERA_PRESETS[viewState.level]
    if (viewState.level !== 'pod' || !topology) return basePreset

    // 计算场景大小
    const podCount = topology.pods.length
    const racksPerPod = topology.pods[0]?.racks.length || 4

    // 根据Pod数量和Rack数量计算缩放因子
    const scaleFactor = Math.max(1, Math.sqrt(podCount * racksPerPod / 4))

    return [
      basePreset[0] * scaleFactor,
      basePreset[1] * scaleFactor,
      basePreset[2] * scaleFactor,
    ] as [number, number, number]
  }, [viewState.level, topology])

  // 动态调整最大缩放距离
  const dynamicCameraDistance = useMemo(() => {
    if (viewState.level !== 'pod' || !topology) return cameraDistance

    const podCount = topology.pods.length
    const racksPerPod = topology.pods[0]?.racks.length || 4
    const scaleFactor = Math.max(1, Math.sqrt(podCount * racksPerPod / 4))

    return {
      min: cameraDistance.min,
      max: cameraDistance.max * scaleFactor * 1.5,
    }
  }, [viewState.level, topology, cameraDistance])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* 3D Canvas */}
      <Canvas shadows>
        <PerspectiveCamera
          key={`camera-${cameraKey}`}
          makeDefault
          position={cameraPreset}
          fov={50}
        />
        <OrbitControls
          key={`controls-${cameraKey}`}
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={dynamicCameraDistance.min}
          maxDistance={dynamicCameraDistance.max}
          target={[0, 0, 0]}
        />

        <color attach="background" args={['#f0f2f5']} />

        {topology && (
          <SceneContent
            topology={topology}
            viewState={viewState}
            currentPod={currentPod}
            currentRack={currentRack}
            currentBoard={currentBoard}
            onNavigate={onNavigate}
            onNavigateToPod={onNavigateToPod}
            onNavigateToRack={onNavigateToRack}
          />
        )}
      </Canvas>

      {/* 导航覆盖层 */}
      <NavigationOverlay
        breadcrumbs={breadcrumbs}
        onBreadcrumbClick={onBreadcrumbClick}
        onBack={onNavigateBack}
        canGoBack={canGoBack}
      />

      {/* 右上角按钮组 */}
      <div style={{
        position: 'absolute',
        top: 16,
        right: 16,
        display: 'flex',
        gap: 8,
      }}>
        <Tooltip title="重置视图">
          <Button
            icon={<ReloadOutlined />}
            onClick={handleResetView}
          />
        </Tooltip>
        <Tooltip title="查看抽象拓扑图">
          <Button
            type="primary"
            icon={<ApartmentOutlined />}
            onClick={() => setShowTopologyGraph(true)}
          >
            拓扑图
          </Button>
        </Tooltip>
      </div>

      {/* 抽象拓扑图弹窗 */}
      <TopologyGraph
        visible={showTopologyGraph}
        onClose={() => setShowTopologyGraph(false)}
        topology={topology}
        currentLevel={viewState.path.length === 0 ? 'datacenter' : viewState.level}
        currentPod={currentPod}
        currentRack={currentRack}
        currentBoard={currentBoard}
        onNodeDoubleClick={(nodeId, nodeType) => {
          if (nodeType === 'pod') {
            onNavigateToPod(nodeId)
          } else if (nodeType === 'rack') {
            const podId = currentPod?.id
            if (podId) {
              onNavigateToRack(podId, nodeId)
            }
          } else if (nodeType === 'board') {
            onNavigate(nodeId)
          }
        }}
      />

      {/* 操作提示 */}
      <div style={{
        position: 'absolute',
        bottom: 16,
        left: 16,
        background: 'rgba(0, 0, 0, 0.7)',
        color: '#fff',
        padding: '8px 12px',
        borderRadius: 8,
        fontSize: 12,
      }}>
        {viewState.level === 'pod' && viewState.path.length === 0 && '双击Pod标签或机柜进入内部视图'}
        {viewState.level === 'pod' && viewState.path.length === 1 && '双击机柜进入内部视图'}
        {viewState.level === 'rack' && '双击板卡查看芯片布局'}
        {(viewState.level === 'board' || viewState.level === 'chip') && '使用导航返回上级'}
      </div>
    </div>
  )
}
