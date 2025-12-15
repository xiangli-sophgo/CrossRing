import React, { useMemo, useRef, useState, useCallback, useEffect } from 'react'
import { Canvas, useFrame, ThreeEvent, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Text, Html } from '@react-three/drei'
import { Breadcrumb, Button, Tooltip } from 'antd'
import { ReloadOutlined, QuestionCircleOutlined } from '@ant-design/icons'
import {
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  ChipConfig,
  SwitchInstance,
  ViewState,
  BreadcrumbItem,
  RACK_DIMENSIONS,
  BOARD_DIMENSIONS,
  CHIP_DIMENSIONS,
  CHIP_TYPE_NAMES,
  CAMERA_PRESETS,
  LODLevel,
  PIN_CONFIG,
  CIRCUIT_TRACE_CONFIG,
  KEYBOARD_SHORTCUTS,
} from '../types'
import * as THREE from 'three'

// ============================================
// 动画工具函数
// ============================================

// 缓动函数：先加速后减速
function easeInOutCubic(t: number): number {
  return t < 0.5
    ? 4 * t * t * t
    : 1 - Math.pow(-2 * t + 2, 3) / 2
}

// 线性插值
function lerp(start: number, end: number, t: number): number {
  return start + (end - start) * t
}

// ============================================
// InstancedMesh 组件 - 批量渲染引脚
// ============================================

interface ChipPinData {
  position: THREE.Vector3
  dimensions: [number, number, number]
}

const InstancedPins: React.FC<{
  chips: ChipPinData[]
  lodLevel: LODLevel
}> = ({ chips, lodLevel }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null)

  // 低细节模式不渲染引脚
  if (lodLevel === 'low' || chips.length === 0) return null

  // 中等细节模式减少引脚数
  const actualPinsPerSide = lodLevel === 'medium' ? 3 : PIN_CONFIG.pinsPerSide
  const actualPinsPerChip = actualPinsPerSide * 4
  const actualTotalPins = chips.length * actualPinsPerChip

  // 创建变换矩阵
  const matrices = useMemo(() => {
    const tempMatrix = new THREE.Matrix4()
    const result: THREE.Matrix4[] = []

    chips.forEach(({ position, dimensions }) => {

      // 左右两侧引脚
      for (let i = 0; i < actualPinsPerSide; i++) {
        const zOffset = (i - (actualPinsPerSide - 1) / 2) * (dimensions[2] / (actualPinsPerSide + 1))

        // 左侧
        tempMatrix.makeTranslation(
          position.x - dimensions[0] / 2 - PIN_CONFIG.pinWidth / 2,
          position.y,
          position.z + zOffset
        )
        result.push(tempMatrix.clone())

        // 右侧
        tempMatrix.makeTranslation(
          position.x + dimensions[0] / 2 + PIN_CONFIG.pinWidth / 2,
          position.y,
          position.z + zOffset
        )
        result.push(tempMatrix.clone())
      }

      // 前后两侧引脚
      for (let i = 0; i < actualPinsPerSide; i++) {
        const xOffset = (i - (actualPinsPerSide - 1) / 2) * (dimensions[0] / (actualPinsPerSide + 1))

        // 前侧
        tempMatrix.makeTranslation(
          position.x + xOffset,
          position.y,
          position.z - dimensions[2] / 2 - PIN_CONFIG.pinDepth / 2
        )
        result.push(tempMatrix.clone())

        // 后侧
        tempMatrix.makeTranslation(
          position.x + xOffset,
          position.y,
          position.z + dimensions[2] / 2 + PIN_CONFIG.pinDepth / 2
        )
        result.push(tempMatrix.clone())
      }
    })

    return result
  }, [chips, actualPinsPerSide])

  // 更新 InstancedMesh
  useEffect(() => {
    if (!meshRef.current) return
    matrices.forEach((matrix, i) => {
      meshRef.current!.setMatrixAt(i, matrix)
    })
    meshRef.current.instanceMatrix.needsUpdate = true
  }, [matrices])

  // 使用第一个芯片的尺寸作为引脚尺寸参考
  const pinDimensions = chips[0] ? [
    PIN_CONFIG.pinWidth,
    chips[0].dimensions[1] * PIN_CONFIG.pinHeightRatio,
    PIN_CONFIG.pinDepth
  ] as [number, number, number] : [0.006, 0.006, 0.004] as [number, number, number]

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, actualTotalPins]} castShadow>
      <boxGeometry args={pinDimensions} />
      <meshStandardMaterial
        color={PIN_CONFIG.pinColor}
        metalness={PIN_CONFIG.pinMetalness}
        roughness={PIN_CONFIG.pinRoughness}
      />
    </instancedMesh>
  )
}

// ============================================
// InstancedMesh 组件 - 批量渲染电路纹理
// ============================================

const InstancedCircuitTraces: React.FC<{
  chips: ChipPinData[]
  lodLevel: LODLevel
}> = ({ chips, lodLevel }) => {
  const hMeshRef = useRef<THREE.InstancedMesh>(null)
  const vMeshRef = useRef<THREE.InstancedMesh>(null)

  // 非高细节模式不渲染电路纹理
  if (lodLevel !== 'high' || chips.length === 0) return null

  const hCount = CIRCUIT_TRACE_CONFIG.horizontalCount
  const vCount = CIRCUIT_TRACE_CONFIG.verticalCount
  const totalHTraces = chips.length * hCount
  const totalVTraces = chips.length * vCount

  // 创建水平纹理变换矩阵
  const hMatrices = useMemo(() => {
    const tempMatrix = new THREE.Matrix4()
    const result: THREE.Matrix4[] = []

    chips.forEach(({ position, dimensions }) => {
      for (let i = 0; i < hCount; i++) {
        const zOffset = (i - (hCount - 1) / 2) * (dimensions[2] * 0.25)
        tempMatrix.makeTranslation(
          position.x,
          position.y + dimensions[1] / 2 + 0.001,
          position.z + zOffset
        )
        result.push(tempMatrix.clone())
      }
    })

    return result
  }, [chips])

  // 创建垂直纹理变换矩阵
  const vMatrices = useMemo(() => {
    const tempMatrix = new THREE.Matrix4()
    const result: THREE.Matrix4[] = []

    chips.forEach(({ position, dimensions }) => {
      for (let i = 0; i < vCount; i++) {
        const xOffset = (i - (vCount - 1) / 2) * (dimensions[0] * 0.25)
        tempMatrix.makeTranslation(
          position.x + xOffset,
          position.y + dimensions[1] / 2 + 0.001,
          position.z
        )
        result.push(tempMatrix.clone())
      }
    })

    return result
  }, [chips])

  // 更新 InstancedMesh
  useEffect(() => {
    if (hMeshRef.current) {
      hMatrices.forEach((matrix, i) => {
        hMeshRef.current!.setMatrixAt(i, matrix)
      })
      hMeshRef.current.instanceMatrix.needsUpdate = true
    }
    if (vMeshRef.current) {
      vMatrices.forEach((matrix, i) => {
        vMeshRef.current!.setMatrixAt(i, matrix)
      })
      vMeshRef.current.instanceMatrix.needsUpdate = true
    }
  }, [hMatrices, vMatrices])

  // 使用第一个芯片的尺寸
  const firstChip = chips[0]
  if (!firstChip) return null

  const hTraceWidth = firstChip.dimensions[0] * 0.7
  const vTraceDepth = firstChip.dimensions[2] * 0.7

  return (
    <>
      {/* 水平纹理 */}
      <instancedMesh ref={hMeshRef} args={[undefined, undefined, totalHTraces]}>
        <boxGeometry args={[hTraceWidth, CIRCUIT_TRACE_CONFIG.traceHeight, CIRCUIT_TRACE_CONFIG.traceWidth]} />
        <meshStandardMaterial color={CIRCUIT_TRACE_CONFIG.traceColor} metalness={0.2} roughness={0.8} />
      </instancedMesh>
      {/* 垂直纹理 */}
      <instancedMesh ref={vMeshRef} args={[undefined, undefined, totalVTraces]}>
        <boxGeometry args={[CIRCUIT_TRACE_CONFIG.traceWidth, CIRCUIT_TRACE_CONFIG.traceHeight, vTraceDepth]} />
        <meshStandardMaterial color={CIRCUIT_TRACE_CONFIG.traceColor} metalness={0.2} roughness={0.8} />
      </instancedMesh>
    </>
  )
}

// ============================================
// 相机动画控制器
// ============================================

interface CameraAnimationTarget {
  position: THREE.Vector3
  lookAt: THREE.Vector3
}

const CameraController: React.FC<{
  target: CameraAnimationTarget
  baseDuration?: number  // 基础动画时长
  onAnimationComplete?: () => void
  enabled?: boolean
  resetTrigger?: number  // 变化时强制重置，即使目标位置相同
}> = ({ target, baseDuration = 1.0, onAnimationComplete, enabled = true, resetTrigger = 0 }) => {
  const { camera } = useThree()
  const controlsRef = useRef<any>(null)

  // 动画状态
  const isAnimating = useRef(false)
  const startPosition = useRef(new THREE.Vector3())
  const startTarget = useRef(new THREE.Vector3())
  const progress = useRef(0)
  const actualDuration = useRef(baseDuration)  // 根据距离动态计算的实际时长
  const lastTarget = useRef<CameraAnimationTarget | null>(null)
  const lastResetTrigger = useRef(resetTrigger)
  const pendingCallback = useRef<(() => void) | null>(null)
  const isFirstRender = useRef(true)  // 首次渲染标记

  // 目标变化或 resetTrigger 变化时启动动画
  useEffect(() => {
    // 首次渲染时直接设置位置，不执行动画
    if (isFirstRender.current) {
      isFirstRender.current = false
      camera.position.copy(target.position)
      if (controlsRef.current) {
        controlsRef.current.target.copy(target.lookAt)
        controlsRef.current.update()
      }
      lastTarget.current = {
        position: target.position.clone(),
        lookAt: target.lookAt.clone()
      }
      return
    }

    // 检查是否是强制重置
    const isForceReset = resetTrigger !== lastResetTrigger.current
    lastResetTrigger.current = resetTrigger

    // 检查目标是否真的变化了（非强制重置时）
    if (!isForceReset && lastTarget.current &&
        lastTarget.current.position.equals(target.position) &&
        lastTarget.current.lookAt.equals(target.lookAt)) {
      return
    }

    // 记录起始位置（从当前相机实际位置开始）
    startPosition.current.copy(camera.position)
    if (controlsRef.current) {
      startTarget.current.copy(controlsRef.current.target)
    } else {
      startTarget.current.set(0, 0, 0)
    }

    // 使用固定动画时长
    actualDuration.current = baseDuration

    progress.current = 0
    isAnimating.current = true
    lastTarget.current = {
      position: target.position.clone(),
      lookAt: target.lookAt.clone()
    }
    pendingCallback.current = onAnimationComplete || null
  }, [target.position.x, target.position.y, target.position.z,
      target.lookAt.x, target.lookAt.y, target.lookAt.z, camera, onAnimationComplete, resetTrigger, baseDuration])

  // 每帧更新
  useFrame((_, delta) => {
    if (!isAnimating.current) return

    progress.current += delta / actualDuration.current
    const t = easeInOutCubic(Math.min(progress.current, 1))

    // 插值相机位置
    camera.position.lerpVectors(startPosition.current, target.position, t)

    // 插值观察目标
    if (controlsRef.current) {
      controlsRef.current.target.lerpVectors(startTarget.current, target.lookAt, t)
      controlsRef.current.update()
    }

    if (progress.current >= 1) {
      isAnimating.current = false
      if (pendingCallback.current) {
        pendingCallback.current()
        pendingCallback.current = null
      }
    }
  })

  return (
    <OrbitControls
      ref={controlsRef}
      enabled={enabled && !isAnimating.current}
      enablePan={true}
      enableZoom={true}
      enableRotate={true}
    />
  )
}

// ============================================
// Props 接口定义
// ============================================

// 3D视图节点详情
export interface Scene3DNodeDetail {
  id: string
  type: 'pod' | 'rack' | 'board' | 'chip' | 'switch'
  label: string
  subType?: string  // 如 chip 的 npu/cpu
  info: Record<string, string | number>  // 额外信息
  connections: { label: string; bandwidth?: number }[]
}

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
  // 历史导航
  onNavigateHistoryBack?: () => void
  onNavigateHistoryForward?: () => void
  canGoHistoryBack?: boolean
  canGoHistoryForward?: boolean
  // 节点选择（显示详情）
  onNodeSelect?: (nodeType: 'pod' | 'rack' | 'board' | 'chip' | 'switch', nodeId: string, label: string, info: Record<string, string | number>, subType?: string) => void
}

// ============================================
// 3D模型组件
// ============================================

// 计算芯片在 Board 上的位置（供 InstancedMesh 使用）
const getChipPosition = (
  chip: ChipConfig,
  totalChips: number,
  baseY: number
): { x: number; y: number; z: number; dimensions: [number, number, number] } => {
  const baseDimensions = CHIP_DIMENSIONS[chip.type] || [0.06, 0.02, 0.06]
  const dimensions: [number, number, number] = [baseDimensions[0] * 0.9, baseDimensions[1] * 1.2, baseDimensions[2] * 0.9]

  const cols = Math.ceil(Math.sqrt(totalChips))
  const rows = Math.ceil(totalChips / cols)
  const row = chip.position[0]
  const col = chip.position[1]
  const chipsInCurrentRow = row < rows - 1 ? cols : totalChips - (rows - 1) * cols

  const spacing = 0.11
  const rowCenterOffset = (chipsInCurrentRow - 1) / 2
  const x = (col - rowCenterOffset) * spacing
  const z = (row - (rows - 1) / 2) * spacing
  const y = baseY + dimensions[1] / 2

  return { x, y, z, dimensions }
}

// Chip模型 - 高端拟物风格，带有文字标识
// 引脚和电路纹理由 InstancedMesh 统一渲染以提升性能
const ChipModel: React.FC<{
  chip: ChipConfig
  baseY?: number  // 底板高度
  totalChips?: number  // 总芯片数，用于计算居中
  lodLevel?: LODLevel  // LOD 级别
  onClick?: () => void
  onPointerOver?: () => void
  onPointerOut?: () => void
}> = ({ chip, baseY = 0, totalChips = 8, lodLevel = 'high', onClick, onPointerOver, onPointerOut }) => {
  const [hovered, setHovered] = useState(false)
  const { x, y, z, dimensions } = getChipPosition(chip, totalChips, baseY)

  // 芯片标签文字 - 优先使用配置的label，否则使用类型名称
  const chipLabel = chip.label || (chip.type === 'npu' ? 'NPU' : 'CPU')
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

      {/* 芯片顶部内嵌区域 - 中等及以上细节 */}
      {lodLevel !== 'low' && (
        <mesh position={[0, dimensions[1] / 2 - 0.001, 0]}>
          <boxGeometry args={[dimensions[0] * 0.92, 0.002, dimensions[2] * 0.92]} />
          <meshStandardMaterial
            color="#0d0d0d"
            metalness={0.2}
            roughness={0.9}
          />
        </mesh>
      )}

      {/* 芯片标识文字 - 仅高细节显示 */}
      {lodLevel === 'high' && (
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
      )}

      {/* 悬停时显示详细信息 - 仅高细节显示 */}
      {lodLevel === 'high' && hovered && (
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
  interactive?: boolean  // 是否可以交互（高亮和点击）
  opacity?: number  // 透明度
  lodLevel?: LODLevel  // LOD 级别
  onDoubleClick?: () => void
  onClick?: () => void  // 单击显示详情
  onChipClick?: (chip: ChipConfig) => void  // 芯片点击
}> = ({ board, showChips = false, interactive = true, opacity = 1.0, lodLevel = 'high', onDoubleClick, onClick, onChipClick }) => {
  const groupRef = useRef<THREE.Group>(null)
  const hoveredRef = useRef(false)
  const [, forceRender] = useState(0)
  const canHover = interactive  // 可交互时才能高亮

  // 根据U高度获取颜色方案
  const uHeight = board.u_height
  const colorScheme = BOARD_U_COLORS[uHeight] || BOARD_U_COLORS[2]

  // 根据U高度计算实际3D尺寸 - 始终使用完整尺寸
  const { uHeight: uSize } = RACK_DIMENSIONS
  const width = BOARD_DIMENSIONS.width
  const height = uHeight * uSize * 0.9  // 留一点间隙
  const depth = BOARD_DIMENSIONS.depth

  // 高亮效果 - 使用ref实现即时响应
  const isHighlighted = canHover && hoveredRef.current

  // 高亮时整体提亮，使用accent颜色作为发光色，与板卡风格统一
  const highlightColor = isHighlighted ? colorScheme.accent : colorScheme.main
  const frontHighlightColor = isHighlighted ? colorScheme.accent : colorScheme.front
  const glowIntensity = isHighlighted ? 0.3 : 0
  const scale = isHighlighted ? 1.01 : 1.0

  // 计算所有芯片的位置数据（供 InstancedMesh 使用）
  const chipPinData = useMemo((): ChipPinData[] => {
    if (!showChips) return []
    return board.chips.map(chip => {
      const { x, y, z, dimensions } = getChipPosition(chip, board.chips.length, 0.004)
      return {
        position: new THREE.Vector3(x, y, z),
        dimensions
      }
    })
  }, [board.chips, showChips])

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

          {/* 铜走线 - 仅在高细节模式下显示 */}
          {lodLevel === 'high' && (
            <>
              {/* 横向走线 */}
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
            </>
          )}

          {/* 过孔(Via) - 仅在高细节模式下显示 */}
          {lodLevel === 'high' && Array.from({ length: 8 }).map((_, i) => {
            const viaX = (Math.sin(i * 2.5) * 0.35) * width / 2
            const viaZ = (Math.cos(i * 3.1) * 0.35) * depth / 2
            return (
              <mesh key={`via-${i}`} position={[viaX, 0.0043, viaZ]} rotation={[-Math.PI / 2, 0, 0]}>
                <cylinderGeometry args={[0.003, 0.003, 0.001, 6]} />
                <meshStandardMaterial color="#b8860b" metalness={0.7} roughness={0.3} />
              </mesh>
            )
          })}

          {/* 边缘金手指接口 - 中等及以上细节 */}
          {lodLevel !== 'low' && (
            <mesh position={[0, 0, -depth / 2 + 0.012]}>
              <boxGeometry args={[width * 0.6, 0.005, 0.018]} />
              <meshStandardMaterial color="#b8923a" metalness={0.3} roughness={0.7} />
            </mesh>
          )}

          {/* 安装孔 - 仅在高细节模式下显示 */}
          {lodLevel === 'high' && [[-1, -1], [-1, 1], [1, -1], [1, 1]].map(([dx, dz], i) => (
            <mesh key={`mount-${i}`} position={[dx * (width / 2 - 0.025), 0.0042, dz * (depth / 2 - 0.025)]} rotation={[-Math.PI / 2, 0, 0]}>
              <circleGeometry args={[0.01, 16]} />
              <meshStandardMaterial color="#8a7040" metalness={0.2} roughness={0.8} />
            </mesh>
          ))}

          {/* 使用 InstancedMesh 批量渲染引脚 */}
          <InstancedPins chips={chipPinData} lodLevel={lodLevel} />

          {/* 使用 InstancedMesh 批量渲染电路纹理 */}
          <InstancedCircuitTraces chips={chipPinData} lodLevel={lodLevel} />

          {/* 渲染芯片 - 放在PCB上，居中排布 */}
          {board.chips.map(chip => (
            <ChipModel
              key={chip.id}
              chip={chip}
              baseY={0.004}
              totalChips={board.chips.length}
              lodLevel={lodLevel}
              onClick={() => onChipClick?.(chip)}
            />
          ))}

          {/* 板卡丝印标识 - 高细节模式 */}
          {lodLevel === 'high' && (
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
          )}
          {/* 版本号丝印 - 高细节模式 */}
          {lodLevel === 'high' && (
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
          )}
        </>
      ) : (
        // 不显示芯片时：封闭的服务器盒子
        <>
          {/* 服务器主体 - 金属外壳 */}
          <group ref={groupRef} scale={scale}>
            <mesh
              onClick={canHover ? (e) => {
                e.stopPropagation()
                onClick?.()
              } : undefined}
              onDoubleClick={canHover ? onDoubleClick : undefined}
              onPointerOver={canHover ? (e) => {
                e.stopPropagation()
                hoveredRef.current = true
                forceRender(n => n + 1)
              } : undefined}
              onPointerOut={canHover ? (e) => {
                e.stopPropagation()
                hoveredRef.current = false
                forceRender(n => n + 1)
              } : undefined}
              castShadow={opacity > 0.5}
              receiveShadow={opacity > 0.5}
            >
              <boxGeometry args={[width, height, depth]} />
              <meshStandardMaterial
                color={highlightColor}
                emissive={colorScheme.accent}
                emissiveIntensity={glowIntensity}
                metalness={0.7}
                roughness={0.3}
                transparent={opacity < 1}
                opacity={opacity}
              />
            </mesh>

            {/* 前面板 - 带有指示灯效果 */}
            <mesh position={[0, 0, depth / 2 + 0.001]}>
              <boxGeometry args={[width - 0.02, height - 0.005, 0.002]} />
              <meshStandardMaterial
                color={frontHighlightColor}
                emissive={colorScheme.accent}
                emissiveIntensity={glowIntensity}
                metalness={0.5}
                roughness={0.5}
                transparent={opacity < 1}
                opacity={opacity}
              />
            </mesh>

            {/* U高度标识条 - 左侧彩色条纹，高亮时更亮 */}
            <mesh position={[-width / 2 + 0.008, 0, depth / 2 + 0.002]}>
              <boxGeometry args={[isHighlighted ? 0.016 : 0.012, height - 0.01, 0.001]} />
              <meshBasicMaterial
                color={isHighlighted ? '#ffffff' : colorScheme.accent}
                transparent={opacity < 1}
                opacity={opacity}
              />
            </mesh>

            {/* LED指示灯 - 高亮时更亮 */}
            <mesh position={[-width / 2 + 0.03, height / 2 - 0.015, depth / 2 + 0.003]}>
              <circleGeometry args={[isHighlighted ? 0.008 : 0.006, 16]} />
              <meshBasicMaterial
                color={isHighlighted ? '#7fff7f' : '#52c41a'}
                transparent={opacity < 1}
                opacity={opacity}
              />
            </mesh>
            <mesh position={[-width / 2 + 0.045, height / 2 - 0.015, depth / 2 + 0.003]}>
              <circleGeometry args={[isHighlighted ? 0.008 : 0.006, 16]} />
              <meshBasicMaterial
                color={isHighlighted ? '#ffffff' : colorScheme.accent}
                transparent={opacity < 1}
                opacity={opacity}
              />
            </mesh>

            {/* 板卡标签 - 调整位置确保不被遮挡 */}
            <Text
              position={[0, 0, depth / 2 + 0.015]}
              fontSize={0.035}
              color="#ffffff"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.003}
              outlineColor="#000000"
              renderOrder={1}
              material-depthTest={false}
              fillOpacity={opacity}
            >
              {board.label}
            </Text>
          </group>
        </>
      )}
    </group>
  )
}

// Switch模型 - 网络交换机，与服务器尺寸一致
const SwitchModel: React.FC<{
  switchData: SwitchInstance
  opacity?: number
  onClick?: () => void
}> = ({ switchData, opacity = 1.0, onClick }) => {
  const [hovered, setHovered] = useState(false)

  // 根据U高度计算实际3D尺寸 - 与BoardModel保持一致
  const { uHeight: uSize } = RACK_DIMENSIONS
  const uHeight = switchData.u_height || 1
  const width = BOARD_DIMENSIONS.width
  const height = uHeight * uSize * 0.9  // 与Board一致
  const depth = BOARD_DIMENSIONS.depth  // 与Board深度一致

  // Switch专用颜色 - 鲜明的深蓝色外壳，与Board明显区分
  const shellColor = '#0a2540'  // 深海蓝
  const shellColorHover = '#0d3a5c'
  const accentColor = '#00d4ff'  // 明亮的青蓝色
  const frontPanelColor = '#061a2e'

  const isHighlighted = hovered
  const glowIntensity = isHighlighted ? 0.4 : 0.1  // 始终有轻微发光

  // 根据U高度计算端口行数和每行端口数
  const portRows = uHeight  // 1U=1行, 2U=2行, 4U=4行
  const portsPerRow = 12  // 每行12个端口
  const portSpacing = (width - 0.12) / portsPerRow  // 端口间距
  const rowSpacing = height / (portRows + 1)  // 行间距

  return (
    <group>
      {/* Switch主体 - 深蓝色金属外壳 */}
      <mesh
        onClick={(e) => {
          e.stopPropagation()
          onClick?.()
        }}
        onPointerOver={(e) => {
          e.stopPropagation()
          setHovered(true)
        }}
        onPointerOut={(e) => {
          e.stopPropagation()
          setHovered(false)
        }}
        castShadow={opacity > 0.5}
        receiveShadow={opacity > 0.5}
      >
        <boxGeometry args={[width, height, depth]} />
        <meshStandardMaterial
          color={isHighlighted ? shellColorHover : shellColor}
          emissive={accentColor}
          emissiveIntensity={glowIntensity}
          metalness={0.8}
          roughness={0.2}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </mesh>

      {/* 前面板 - 更深的颜色 */}
      <mesh position={[0, 0, depth / 2 + 0.001]}>
        <boxGeometry args={[width - 0.02, height - 0.005, 0.002]} />
        <meshStandardMaterial
          color={frontPanelColor}
          emissive={accentColor}
          emissiveIntensity={glowIntensity * 0.3}
          metalness={0.6}
          roughness={0.4}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </mesh>

      {/* 左侧青蓝色标识条 - Switch特有标识，更宽更亮 */}
      <mesh position={[-width / 2 + 0.01, 0, depth / 2 + 0.002]}>
        <boxGeometry args={[isHighlighted ? 0.02 : 0.016, height - 0.008, 0.001]} />
        <meshBasicMaterial
          color={isHighlighted ? '#ffffff' : accentColor}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </mesh>

      {/* 网口区域 - 根据U高度显示多行端口 */}
      {Array.from({ length: portRows }).map((_, rowIdx) => {
        const rowY = -height / 2 + rowSpacing * (rowIdx + 1)
        return (
          <group key={`row-${rowIdx}`}>
            {Array.from({ length: portsPerRow }).map((_, portIdx) => {
              const portX = -width / 2 + 0.06 + portIdx * portSpacing
              const isActive = (rowIdx + portIdx) % 3 !== 0  // 部分端口激活
              return (
                <group key={`port-${rowIdx}-${portIdx}`}>
                  {/* 端口外框 */}
                  <mesh position={[portX, rowY, depth / 2 + 0.002]}>
                    <boxGeometry args={[0.022, 0.018, 0.001]} />
                    <meshBasicMaterial color="#1a1a1a" transparent={opacity < 1} opacity={opacity} />
                  </mesh>
                  {/* 端口内部 */}
                  <mesh position={[portX, rowY, depth / 2 + 0.0025]}>
                    <boxGeometry args={[0.018, 0.014, 0.001]} />
                    <meshBasicMaterial color="#0a0a0a" transparent={opacity < 1} opacity={opacity} />
                  </mesh>
                  {/* 端口LED - 在端口上方 */}
                  <mesh position={[portX, rowY + 0.012, depth / 2 + 0.003]}>
                    <circleGeometry args={[0.003, 8]} />
                    <meshBasicMaterial
                      color={isActive ? (isHighlighted ? '#7fff7f' : '#00ff88') : '#333'}
                      transparent={opacity < 1}
                      opacity={opacity}
                    />
                  </mesh>
                </group>
              )
            })}
          </group>
        )
      })}

      {/* 右侧状态LED区域 */}
      <group>
        {/* 电源LED */}
        <mesh position={[width / 2 - 0.025, height / 2 - 0.015, depth / 2 + 0.003]}>
          <circleGeometry args={[0.006, 12]} />
          <meshBasicMaterial
            color={isHighlighted ? '#7fff7f' : '#00ff88'}
            transparent={opacity < 1}
            opacity={opacity}
          />
        </mesh>
        {/* 状态LED */}
        <mesh position={[width / 2 - 0.04, height / 2 - 0.015, depth / 2 + 0.003]}>
          <circleGeometry args={[0.006, 12]} />
          <meshBasicMaterial
            color={isHighlighted ? '#ffffff' : accentColor}
            transparent={opacity < 1}
            opacity={opacity}
          />
        </mesh>
        {/* 活动LED */}
        <mesh position={[width / 2 - 0.055, height / 2 - 0.015, depth / 2 + 0.003]}>
          <circleGeometry args={[0.006, 12]} />
          <meshBasicMaterial
            color={isHighlighted ? '#ffff7f' : '#ffa500'}
            transparent={opacity < 1}
            opacity={opacity}
          />
        </mesh>
      </group>

      {/* 顶部散热孔装饰（2U以上显示） */}
      {uHeight >= 2 && (
        <group position={[0, height / 2 - 0.008, depth / 2 + 0.002]}>
          {Array.from({ length: 6 }).map((_, i) => (
            <mesh key={`vent-${i}`} position={[width / 2 - 0.08 - i * 0.012, 0, 0]}>
              <boxGeometry args={[0.008, 0.004, 0.001]} />
              <meshBasicMaterial color="#0a0a0a" transparent={opacity < 1} opacity={opacity} />
            </mesh>
          ))}
        </group>
      )}

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
          color={hovered ? '#7a9fd4' : '#1890ff'}
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

// ConnectionLine 组件预留，暂未使用


// ============================================
// 统一场景组件 - 一次性渲染所有层级内容
// ============================================

interface NodePositions {
  pods: Map<string, THREE.Vector3>      // Pod中心位置
  racks: Map<string, THREE.Vector3>     // Rack位置
  boards: Map<string, THREE.Vector3>    // Board世界坐标
}

// 透明度动画控制器组件 - 使用useFrame实现平滑过渡
const OpacityAnimator: React.FC<{
  targetOpacities: Map<string, number>
  currentOpacities: React.MutableRefObject<Map<string, number>>
  fadeInDuration?: number   // 淡入时长
  fadeOutDuration?: number  // 淡出时长
}> = ({ targetOpacities, currentOpacities, fadeInDuration = 0.8, fadeOutDuration = 0.3 }) => {
  useFrame((_, delta) => {
    targetOpacities.forEach((target, id) => {
      const current = currentOpacities.current.get(id) ?? target
      if (Math.abs(current - target) > 0.001) {
        // 淡出（目标为0）时速度快，淡入时速度慢
        const duration = target < current ? fadeOutDuration : fadeInDuration
        const speed = 1 / duration
        const newValue = lerp(current, target, Math.min(delta * speed * 2.5, 1))
        currentOpacities.current.set(id, newValue)
      } else {
        currentOpacities.current.set(id, target)
      }
    })
  })
  return null
}

// 触发React重渲染的组件 - 当透明度变化时触发更新
const OpacityUpdateTrigger: React.FC<{
  targetOpacities: Map<string, number>
  currentOpacities: React.MutableRefObject<Map<string, number>>
  onUpdate: () => void
}> = ({ targetOpacities, currentOpacities, onUpdate }) => {
  const lastUpdateRef = useRef(0)

  useFrame(() => {
    // 检查是否有任何透明度还在动画中
    let hasAnimation = false
    targetOpacities.forEach((target, id) => {
      const current = currentOpacities.current.get(id) ?? target
      if (Math.abs(current - target) > 0.001) {
        hasAnimation = true
      }
    })

    // 如果有动画，每隔一定时间触发一次重渲染
    if (hasAnimation) {
      const now = Date.now()
      if (now - lastUpdateRef.current > 16) { // 约60fps
        lastUpdateRef.current = now
        onUpdate()
      }
    }
  })
  return null
}

const UnifiedScene: React.FC<{
  topology: HierarchicalTopology
  focusPath: string[]  // 当前聚焦路径 ['pod_0', 'rack_1', 'board_2']
  onNavigateToPod: (podId: string) => void
  onNavigateToRack: (podId: string, rackId: string) => void
  onNavigateToBoard: (boardId: string) => void
  onNodeClick?: (nodeType: 'pod' | 'rack' | 'board' | 'chip' | 'switch', nodeId: string, label: string, info: Record<string, string | number>, subType?: string) => void
}> = ({ topology, focusPath, onNavigateToPod, onNavigateToRack, onNavigateToBoard, onNodeClick }) => {
  const [hoveredPodId, setHoveredPodId] = useState<string | null>(null)
  const [hoveredRackId, setHoveredRackId] = useState<string | null>(null)

  // 透明度动画状态 - 存储当前渲染的透明度值
  const currentOpacities = useRef<Map<string, number>>(new Map())
  // 用于触发重新渲染的状态
  const [, forceUpdate] = useState(0)

  const rackSpacingX = 1.5
  const rackSpacingZ = 2
  const { uHeight, totalU, width: rackWidth, depth: rackDepth } = RACK_DIMENSIONS
  const rackHeight = totalU * uHeight

  // 计算Pod布局参数
  const { podSpacingX, podSpacingZ, podCols } = useMemo(() => {
    const firstPod = topology.pods[0]
    if (!firstPod) return { podSpacingX: 6, podSpacingZ: 4, podCols: 2 }

    const rackCols = firstPod.grid_size[1]
    const rackRows = firstPod.grid_size[0]
    const podWidth = rackCols * rackSpacingX + 2
    const podDepth = rackRows * rackSpacingZ + 1

    const totalPods = topology.pods.length
    let cols: number
    if (totalPods <= 2) cols = totalPods
    else if (totalPods <= 4) cols = 2
    else if (totalPods <= 6) cols = 3
    else if (totalPods <= 9) cols = 3
    else cols = 4

    return { podSpacingX: podWidth, podSpacingZ: podDepth, podCols: cols }
  }, [topology.pods])

  // 计算所有节点的世界坐标
  const nodePositions = useMemo((): NodePositions => {
    const pods = new Map<string, THREE.Vector3>()
    const racks = new Map<string, THREE.Vector3>()
    const boards = new Map<string, THREE.Vector3>()

    // 计算Pod网格位置
    const getPodGridPosition = (podIndex: number) => {
      const row = Math.floor(podIndex / podCols)
      const col = podIndex % podCols
      return { row, col }
    }

    // 首先计算所有Rack位置以找出中心
    let minX = Infinity, maxX = -Infinity
    let minZ = Infinity, maxZ = -Infinity

    topology.pods.forEach((pod, podIndex) => {
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

    const centerX = (minX + maxX) / 2
    const centerZ = (minZ + maxZ) / 2

    // 设置所有节点位置
    topology.pods.forEach((pod, podIndex) => {
      const { row, col } = getPodGridPosition(podIndex)
      const podOffsetX = col * podSpacingX
      const podOffsetZ = row * podSpacingZ

      let podSumX = 0, podSumZ = 0, podCount = 0

      pod.racks.forEach(rack => {
        const rackX = podOffsetX + rack.position[1] * rackSpacingX - centerX
        const rackZ = podOffsetZ + rack.position[0] * rackSpacingZ - centerZ
        racks.set(rack.id, new THREE.Vector3(rackX, 0, rackZ))

        podSumX += rackX
        podSumZ += rackZ
        podCount++

        // 计算Board位置
        rack.boards.forEach(board => {
          const boardY = (board.u_position - 1) * uHeight + (board.u_height * uHeight) / 2 - rackHeight / 2
          boards.set(board.id, new THREE.Vector3(rackX, boardY, rackZ))
        })
      })

      // Pod中心
      if (podCount > 0) {
        pods.set(pod.id, new THREE.Vector3(podSumX / podCount, 0, podSumZ / podCount))
      }
    })

    return { pods, racks, boards }
  }, [topology, podSpacingX, podSpacingZ, podCols, uHeight, rackHeight])


  // 获取节点目标透明度 - 非聚焦内容完全隐藏
  const getTargetOpacity = useCallback((nodeId: string, nodeType: 'pod' | 'rack' | 'board'): number => {
    if (focusPath.length === 0) return 1.0 // 顶层全显示

    if (nodeType === 'rack') {
      if (focusPath.length === 1) {
        // 聚焦Pod，只显示该Pod下的Rack，其他Pod的Rack完全隐藏
        const pod = topology.pods.find(p => p.id === focusPath[0])
        const isInPod = pod?.racks.some(r => r.id === nodeId)
        return isInPod ? 1.0 : 0
      }
      if (focusPath.length === 2) {
        // 聚焦Rack，只显示聚焦的Rack
        return focusPath[1] === nodeId ? 1.0 : 0
      }
      if (focusPath.length >= 3) {
        // 聚焦Board，所有Rack都完全隐藏（只显示Board）
        return 0
      }
    }
    if (nodeType === 'board') {
      if (focusPath.length === 1) {
        // 聚焦Pod，只显示该Pod下的Board，其他Pod的Board隐藏
        const pod = topology.pods.find(p => p.id === focusPath[0])
        const isInPod = pod?.racks.some(r => r.boards.some(b => b.id === nodeId))
        return isInPod ? 1.0 : 0
      }
      if (focusPath.length === 2) {
        // 聚焦Rack，只显示该Rack下的Board
        const pod = topology.pods.find(p => p.id === focusPath[0])
        const rack = pod?.racks.find(r => r.id === focusPath[1])
        const isInRack = rack?.boards.some(b => b.id === nodeId)
        return isInRack ? 1.0 : 0
      }
      if (focusPath.length >= 3) {
        // 聚焦Board，只显示聚焦的Board
        return focusPath[2] === nodeId ? 1.0 : 0
      }
    }
    return 1.0
  }, [focusPath, topology])

  // 计算所有节点的目标透明度
  const targetOpacities = useMemo(() => {
    const opacities = new Map<string, number>()
    topology.pods.forEach(pod => {
      pod.racks.forEach(rack => {
        opacities.set(rack.id, getTargetOpacity(rack.id, 'rack'))
        rack.boards.forEach(board => {
          opacities.set(board.id, getTargetOpacity(board.id, 'board'))
        })
      })
    })
    return opacities
  }, [topology, getTargetOpacity])

  // 获取当前动画透明度（如果没有则使用目标值）
  const getAnimatedOpacity = useCallback((nodeId: string): number => {
    return currentOpacities.current.get(nodeId) ?? targetOpacities.get(nodeId) ?? 1.0
  }, [targetOpacities])

  // 当前聚焦层级
  const focusLevel = focusPath.length

  return (
    <group>
      {/* 透明度动画控制器 - 每帧更新透明度并触发重渲染 */}
      <OpacityAnimator
        targetOpacities={targetOpacities}
        currentOpacities={currentOpacities}
        fadeInDuration={1.2}
        fadeOutDuration={0.2}
      />
      {/* 每帧检查是否需要重渲染 */}
      <OpacityUpdateTrigger
        targetOpacities={targetOpacities}
        currentOpacities={currentOpacities}
        onUpdate={() => forceUpdate(n => n + 1)}
      />

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

      {/* 渲染所有Pod */}
      {topology.pods.map(pod => {
        const podCenter = nodePositions.pods.get(pod.id)
        if (!podCenter) return null

        const isPodHighlighted = hoveredPodId === pod.id

        return (
          <group key={pod.id}>
            {/* Pod标签 - 只在顶层或聚焦该Pod时显示 */}
            {(focusLevel === 0 || (focusLevel === 1 && focusPath[0] === pod.id)) && (
              <PodLabel
                pod={pod}
                position={[podCenter.x, rackHeight / 2 + 0.5, podCenter.z]}
                onDoubleClick={() => onNavigateToPod(pod.id)}
                onHoverChange={(hovered) => setHoveredPodId(hovered ? pod.id : null)}
              />
            )}

            {/* 渲染该Pod下的所有Rack */}
            {pod.racks.map(rack => {
              const rackPos = nodePositions.racks.get(rack.id)
              if (!rackPos) return null

              const rackOpacity = getAnimatedOpacity(rack.id)
              const isRackHighlighted = focusLevel === 0 ? isPodHighlighted : hoveredRackId === rack.id

              // 是否显示Board详情（聚焦到Rack级别或更深）
              const showBoardDetails = focusLevel >= 2 && focusPath[1] === rack.id

              // 透明度太低时不渲染（但要给动画留一点余地）
              if (rackOpacity < 0.01) return null

              return (
                <group key={rack.id} position={[rackPos.x, rackPos.y, rackPos.z]}>
                  {/* 机柜框架 */}
                  <group>
                    {/* 机柜底座 */}
                    <mesh position={[0, -rackHeight / 2 - 0.02, 0]} receiveShadow>
                      <boxGeometry args={[rackWidth + 0.04, 0.04, rackDepth + 0.04]} />
                      <meshStandardMaterial
                        color={isRackHighlighted ? '#7a9fd4' : '#1a1a1a'}
                        transparent
                        opacity={rackOpacity}
                        metalness={0.3}
                        roughness={0.7}
                      />
                    </mesh>

                    {/* 机柜顶部 */}
                    <mesh position={[0, rackHeight / 2 + 0.02, 0]} castShadow={rackOpacity > 0.5}>
                      <boxGeometry args={[rackWidth + 0.04, 0.04, rackDepth + 0.04]} />
                      <meshStandardMaterial
                        color={isRackHighlighted ? '#7a9fd4' : '#1a1a1a'}
                        transparent
                        opacity={rackOpacity}
                        metalness={0.3}
                        roughness={0.7}
                      />
                    </mesh>

                    {/* 四个垂直立柱 */}
                    {[
                      [-rackWidth / 2, 0, -rackDepth / 2],
                      [rackWidth / 2, 0, -rackDepth / 2],
                      [-rackWidth / 2, 0, rackDepth / 2],
                      [rackWidth / 2, 0, rackDepth / 2],
                    ].map((pos, i) => (
                      <mesh key={`pillar-${i}`} position={pos as [number, number, number]} castShadow={rackOpacity > 0.5}>
                        <boxGeometry args={[0.02, rackHeight, 0.02]} />
                        <meshStandardMaterial
                          color={isRackHighlighted ? '#7a9fd4' : '#333333'}
                          transparent
                          opacity={rackOpacity}
                          metalness={0.3}
                          roughness={0.7}
                        />
                      </mesh>
                    ))}

                    {/* 后面板 */}
                    <mesh position={[0, 0, -rackDepth / 2 + 0.005]} receiveShadow>
                      <boxGeometry args={[rackWidth - 0.04, rackHeight, 0.01]} />
                      <meshStandardMaterial
                        color="#2a2a2a"
                        transparent
                        opacity={rackOpacity * 0.8}
                        metalness={0.3}
                        roughness={0.7}
                      />
                    </mesh>

                    {/* 前面板 - 半透明 */}
                    <mesh position={[0, 0, rackDepth / 2 - 0.005]}>
                      <boxGeometry args={[rackWidth - 0.04, rackHeight, 0.008]} />
                      <meshStandardMaterial
                        color="#4a4a4a"
                        transparent
                        opacity={showBoardDetails ? 0.02 : 0.08 * rackOpacity}
                        metalness={0.9}
                        roughness={0.1}
                      />
                    </mesh>

                    {/* 机柜标签 */}
                    {rackOpacity > 0.3 && (
                      <Text
                        position={[0, rackHeight / 2 + 0.12, 0.01]}
                        fontSize={0.2}
                        color="#ffffff"
                        anchorX="center"
                        anchorY="middle"
                        fontWeight="bold"
                        fillOpacity={rackOpacity}
                      >
                        {rack.label}
                      </Text>
                    )}

                    {/* 交互层 - 用于点击和双击，只在顶层和Pod层级有效 */}
                    {focusLevel <= 1 && (
                      <mesh
                        visible={false}
                        onClick={(e) => {
                          e.stopPropagation()
                          onNodeClick?.('rack', rack.id, rack.label, {
                            '位置': `(${rack.position[0]}, ${rack.position[1]})`,
                            '总U数': rack.total_u,
                            '板卡数': rack.boards.length
                          })
                        }}
                        onDoubleClick={() => {
                          if (focusLevel === 0) {
                            onNavigateToPod(pod.id)
                          } else if (focusLevel === 1 && focusPath[0] === pod.id) {
                            onNavigateToRack(pod.id, rack.id)
                          }
                        }}
                        onPointerOver={() => {
                          if (focusLevel === 0) setHoveredPodId(pod.id)
                          else if (focusLevel === 1) setHoveredRackId(rack.id)
                        }}
                        onPointerOut={() => {
                          setHoveredPodId(null)
                          setHoveredRackId(null)
                        }}
                      >
                        <boxGeometry args={[rackWidth, rackHeight, rackDepth]} />
                        <meshBasicMaterial transparent opacity={0} />
                      </mesh>
                    )}
                  </group>

                </group>
              )
            })}
          </group>
        )
      })}

      {/* 独立渲染所有Board - 不受Rack透明度影响 */}
      {topology.pods.map(pod => (
        <group key={`boards-${pod.id}`}>
          {pod.racks.map(rack => {
            const rackPos = nodePositions.racks.get(rack.id)
            if (!rackPos) return null

            // 是否显示Board详情（聚焦到Rack级别或更深）
            const showBoardDetails = focusLevel >= 2 && focusPath[1] === rack.id

            return rack.boards.map(board => {
              const boardY = (board.u_position - 1) * uHeight + (board.u_height * uHeight) / 2 - rackHeight / 2
              const boardOpacity = getAnimatedOpacity(board.id)

              // 是否显示芯片（聚焦到Board级别）
              const showChips = focusLevel >= 3 && focusPath[2] === board.id

              // 透明度太低时不渲染（但要给动画留一点余地）
              if (boardOpacity < 0.01) return null

              return (
                <group key={board.id} position={[rackPos.x, rackPos.y + boardY, rackPos.z]}>
                  <BoardModel
                    board={board}
                    showChips={showChips}
                    interactive={showBoardDetails}
                    opacity={boardOpacity}
                    onDoubleClick={() => onNavigateToBoard(board.id)}
                    onClick={() => onNodeClick?.('board', board.id, board.label, {
                      'U位置': board.u_position,
                      'U高度': board.u_height,
                      '芯片数': board.chips.length
                    })}
                    onChipClick={(chip) => onNodeClick?.('chip', chip.id, chip.label || chip.type.toUpperCase(), {
                      '类型': chip.type.toUpperCase(),
                      '位置': `(${chip.position[0]}, ${chip.position[1]})`
                    }, chip.type)}
                  />
                </group>
              )
            })
          })}
        </group>
      ))}

      {/* 渲染Rack层级的Switch - 在Pod和Rack层级显示，Board层级隐藏 */}
      {topology.switch_config?.rack_level?.enabled && topology.pods.map(pod => (
        <group key={`switches-${pod.id}`}>
          {pod.racks.map(rack => {
            const rackPos = nodePositions.racks.get(rack.id)
            if (!rackPos) return null

            // 计算Switch透明度：
            // - 顶层(focusLevel=0): 所有Rack的Switch都显示
            // - Pod层级(focusLevel=1): 只显示该Pod下的Rack的Switch
            // - Rack层级(focusLevel=2): 只显示聚焦Rack的Switch
            // - Board层级(focusLevel>=3): 隐藏所有Switch
            let switchOpacity = 0
            if (focusLevel === 0) {
              switchOpacity = 1.0
            } else if (focusLevel === 1 && focusPath[0] === pod.id) {
              switchOpacity = 1.0
            } else if (focusLevel === 2 && focusPath[1] === rack.id) {
              switchOpacity = 1.0
            }
            if (switchOpacity === 0) return null

            // 获取该Rack下的所有Switch（使用后端计算的u_position）
            const rackSwitches = topology.switches?.filter(
              sw => sw.hierarchy_level === 'rack' && sw.parent_id === rack.id
            ) || []

            if (rackSwitches.length === 0) return null

            // 汇总显示：使用第一个Switch的u_position和配置的高度
            const firstSwitch = rackSwitches[0]
            const switchUPosition = firstSwitch.u_position || 1
            const switchUHeight = firstSwitch.u_height || 1  // 使用配置的高度，不累加
            const switchY = (switchUPosition - 1) * uHeight + (switchUHeight * uHeight) / 2 - rackHeight / 2

            // 创建汇总的Switch用于显示
            const summarySwitch: SwitchInstance = {
              id: `${rack.id}/switch_summary`,
              type_id: 'summary',
              layer: 'leaf',
              hierarchy_level: 'rack',
              parent_id: rack.id,
              label: `Switch ×${rackSwitches.length}`,
              uplink_ports_used: 0,
              downlink_ports_used: 0,
              inter_ports_used: 0,
              u_height: switchUHeight,
              u_position: switchUPosition
            }

            // 构建每个Switch的详细信息
            const switchInfoObj: Record<string, string | number> = {
              '所属Rack': rack.label,
              'U位置': switchUPosition,
              'U高度': switchUHeight,
            }
            // 添加每个Switch的详情
            rackSwitches.forEach((sw, idx) => {
              switchInfoObj[`[${idx + 1}] ${sw.label}`] = `上行:${sw.uplink_ports_used} 下行:${sw.downlink_ports_used} 互联:${sw.inter_ports_used}`
            })

            return (
              <group key={`${rack.id}/switch`} position={[rackPos.x, rackPos.y + switchY, rackPos.z]}>
                <SwitchModel
                  switchData={summarySwitch}
                  opacity={switchOpacity}
                  onClick={() => onNodeClick?.('switch', rackSwitches[0].id, `${rack.label} Switch ×${rackSwitches.length}`, switchInfoObj)}
                />
              </group>
            )
          })}
        </group>
      ))}

      {/* 地面 */}
      <mesh
        position={[0, -rackHeight / 2 - 0.06, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        receiveShadow
      >
        <planeGeometry args={[50, 50]} />
        <meshStandardMaterial color="#e8e8e8" />
      </mesh>
      {/* 地面网格线 */}
      <gridHelper
        args={[50, 50, '#bbb', '#ddd']}
        position={[0, -rackHeight / 2 - 0.05, 0]}
      />
    </group>
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
}> = ({ breadcrumbs, onBreadcrumbClick }) => {
  return (
    <div style={{
      position: 'absolute',
      top: 16,
      left: 16,
      zIndex: 100,
      background: 'rgba(255, 255, 255, 0.95)',
      padding: '8px 16px',
      borderRadius: 8,
      boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
    }}>
      <Breadcrumb
        items={breadcrumbs.map((item, index) => ({
          title: (
            <a
              onClick={(e) => {
                e.preventDefault()
                onBreadcrumbClick(index)
              }}
              style={{
                cursor: index < breadcrumbs.length - 1 ? 'pointer' : 'default',
                color: index < breadcrumbs.length - 1 ? '#1890ff' : 'rgba(0, 0, 0, 0.88)',
                fontWeight: index === breadcrumbs.length - 1 ? 500 : 400,
              }}
            >
              {item.label}
            </a>
          ),
        }))}
      />
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
  onNodeSelect,
}) => {
  // 用于强制重置相机位置的 key
  const [resetKey, setResetKey] = useState(0)
  // 是否显示快捷键帮助
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false)
  // 初始相机位置（只用于首次渲染）
  const initialCameraPositionRef = useRef<[number, number, number] | null>(null)

  // 重置视图（相机位置）
  const handleResetView = useCallback(() => {
    setResetKey(k => k + 1)
  }, [])

  // 键盘快捷键处理 - 仅处理3D视图特有的快捷键（R重置视角、?帮助）
  // ESC、Backspace、方向键已在App.tsx中全局处理
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // 如果正在输入框中则忽略
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      // R - 重置相机视角 (同时检查 key 和 code 以兼容输入法)
      if (KEYBOARD_SHORTCUTS.resetView.includes(e.code) || e.key === 'r' || e.key === 'R') {
        e.preventDefault()
        setResetKey(k => k + 1)
        return
      }

      // ? - 显示/隐藏快捷键帮助
      if (e.code === 'Slash' && e.shiftKey) {
        e.preventDefault()
        setShowKeyboardHelp(prev => !prev)
        return
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // 计算所有节点的世界坐标（与 UnifiedScene 保持一致）
  const nodePositions = useMemo(() => {
    if (!topology) return { pods: new Map(), racks: new Map(), boards: new Map() }

    const rackSpacingX = 1.5
    const rackSpacingZ = 2
    const { uHeight, totalU } = RACK_DIMENSIONS
    const rackHeight = totalU * uHeight

    const pods = new Map<string, THREE.Vector3>()
    const racks = new Map<string, THREE.Vector3>()
    const boards = new Map<string, THREE.Vector3>()

    // 计算Pod布局参数
    const firstPod = topology.pods[0]
    let podSpacingX = 6, podSpacingZ = 4, podCols = 2
    if (firstPod) {
      const rackCols = firstPod.grid_size[1]
      const rackRows = firstPod.grid_size[0]
      podSpacingX = rackCols * rackSpacingX + 2
      podSpacingZ = rackRows * rackSpacingZ + 1

      const totalPods = topology.pods.length
      if (totalPods <= 2) podCols = totalPods
      else if (totalPods <= 4) podCols = 2
      else if (totalPods <= 6) podCols = 3
      else if (totalPods <= 9) podCols = 3
      else podCols = 4
    }

    const getPodGridPosition = (podIndex: number) => {
      const row = Math.floor(podIndex / podCols)
      const col = podIndex % podCols
      return { row, col }
    }

    // 首先计算所有Rack位置以找出中心
    let minX = Infinity, maxX = -Infinity
    let minZ = Infinity, maxZ = -Infinity

    topology.pods.forEach((pod, podIndex) => {
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

    const centerX = (minX + maxX) / 2
    const centerZ = (minZ + maxZ) / 2

    // 设置所有节点位置
    topology.pods.forEach((pod, podIndex) => {
      const { row, col } = getPodGridPosition(podIndex)
      const podOffsetX = col * podSpacingX
      const podOffsetZ = row * podSpacingZ

      let podSumX = 0, podSumZ = 0, podCount = 0

      pod.racks.forEach(rack => {
        const rackX = podOffsetX + rack.position[1] * rackSpacingX - centerX
        const rackZ = podOffsetZ + rack.position[0] * rackSpacingZ - centerZ
        racks.set(rack.id, new THREE.Vector3(rackX, 0, rackZ))

        podSumX += rackX
        podSumZ += rackZ
        podCount++

        // 计算Board位置
        rack.boards.forEach(board => {
          const boardY = (board.u_position - 1) * uHeight + (board.u_height * uHeight) / 2 - rackHeight / 2
          boards.set(board.id, new THREE.Vector3(rackX, boardY, rackZ))
        })
      })

      // Pod中心
      if (podCount > 0) {
        pods.set(pod.id, new THREE.Vector3(podSumX / podCount, 0, podSumZ / podCount))
      }
    })

    return { pods, racks, boards }
  }, [topology])

  // 根据当前视图状态计算相机目标位置和观察点
  const cameraTarget = useMemo((): CameraAnimationTarget => {
    // 根据视图层级和路径计算相机位置
    if (viewState.path.length === 0) {
      // 数据中心顶层视图
      const basePreset = CAMERA_PRESETS['pod']
      const lookAt = new THREE.Vector3(0, 0, 0)
      if (topology) {
        const podCount = topology.pods.length
        const racksPerPod = topology.pods[0]?.racks.length || 4
        const scaleFactor = Math.max(1, Math.sqrt(podCount * racksPerPod / 4))
        return {
          position: new THREE.Vector3(
            basePreset[0] * scaleFactor,
            basePreset[1] * scaleFactor,
            basePreset[2] * scaleFactor
          ),
          lookAt
        }
      }
      return {
        position: new THREE.Vector3(basePreset[0], basePreset[1], basePreset[2]),
        lookAt
      }
    }

    if (viewState.path.length === 1 && currentPod && topology) {
      // Pod内部视图 - 相机飞向该Pod中心位置
      const podCenter = nodePositions.pods.get(currentPod.id)
      if (podCenter) {
        // 单Pod特殊处理：保持与数据中心层相同的视角
        if (topology.pods.length === 1) {
          const basePreset = CAMERA_PRESETS['pod']
          const racksPerPod = topology.pods[0]?.racks.length || 4
          const scaleFactor = Math.max(1, Math.sqrt(racksPerPod / 4))
          return {
            position: new THREE.Vector3(
              basePreset[0] * scaleFactor,
              basePreset[1] * scaleFactor,
              basePreset[2] * scaleFactor
            ),
            lookAt: podCenter.clone()
          }
        }

        // 多Pod情况：原有逻辑
        const racksCount = currentPod.racks.length
        const distance = 3 + racksCount * 0.5  // 根据Rack数量调整距离
        return {
          position: new THREE.Vector3(
            podCenter.x + distance,
            distance * 0.8,
            podCenter.z + distance
          ),
          lookAt: podCenter.clone()
        }
      }
    }

    if (viewState.path.length === 2 && currentRack) {
      // Rack内部视图 - 相机飞向该Rack前方
      const rackPos = nodePositions.racks.get(currentRack.id)
      if (rackPos) {
        return {
          position: new THREE.Vector3(
            rackPos.x + 0.8,
            rackPos.y + 0.5,
            rackPos.z + 2.5
          ),
          lookAt: new THREE.Vector3(rackPos.x, rackPos.y, rackPos.z)
        }
      }
    }

    if (viewState.path.length >= 3 && currentBoard) {
      // Board视图 - 相机飞向该Board上方，调整合适的观察距离
      const boardPos = nodePositions.boards.get(currentBoard.id)
      if (boardPos) {
        return {
          position: new THREE.Vector3(
            boardPos.x + 0.5,
            boardPos.y + 1.0,
            boardPos.z + 0.8
          ),
          lookAt: new THREE.Vector3(boardPos.x, boardPos.y, boardPos.z)
        }
      }
    }

    // 默认
    const basePreset = CAMERA_PRESETS[viewState.level]
    return {
      position: new THREE.Vector3(basePreset[0], basePreset[1], basePreset[2]),
      lookAt: new THREE.Vector3(0, 0, 0)
    }
  }, [viewState.path, viewState.level, topology, currentPod, currentRack, currentBoard, nodePositions, resetKey])

  // 记录初始相机位置（只在首次计算cameraTarget时设置）
  if (initialCameraPositionRef.current === null) {
    initialCameraPositionRef.current = [cameraTarget.position.x, cameraTarget.position.y, cameraTarget.position.z]
  }

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* 3D Canvas */}
      <Canvas shadows>
        {/* PerspectiveCamera只使用初始位置，后续由CameraController控制 */}
        <PerspectiveCamera
          makeDefault
          position={initialCameraPositionRef.current}
          fov={50}
        />

        {/* 使用 CameraController 实现平滑动画 - 由它控制相机位置 */}
        <CameraController
          target={cameraTarget}
          baseDuration={1.2}
          resetTrigger={resetKey}
        />

        <color attach="background" args={['#f0f2f5']} />

        {/* 使用统一场景渲染所有层级，通过相机移动和透明度控制实现层级切换 */}
        {topology && (
          <UnifiedScene
            topology={topology}
            focusPath={viewState.path}
            onNavigateToPod={onNavigateToPod}
            onNavigateToRack={onNavigateToRack}
            onNavigateToBoard={onNavigate}
            onNodeClick={onNodeSelect}
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
        <Tooltip title="快捷键帮助 (?)">
          <Button
            icon={<QuestionCircleOutlined />}
            onClick={() => setShowKeyboardHelp(prev => !prev)}
          />
        </Tooltip>
        <Tooltip title="重置视图 (R)">
          <Button
            icon={<ReloadOutlined />}
            onClick={handleResetView}
          />
        </Tooltip>
      </div>

      {/* 快捷键帮助面板 - 点击空白处关闭 */}
      {showKeyboardHelp && (
        <>
          {/* 透明遮罩层，点击关闭 */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 199,
            }}
            onClick={() => setShowKeyboardHelp(false)}
          />
          {/* 帮助面板 */}
          <div style={{
            position: 'absolute',
            top: 60,
            right: 16,
            background: 'rgba(255, 255, 255, 0.98)',
            padding: '16px 20px',
            borderRadius: 8,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            fontSize: 13,
            zIndex: 200,
            minWidth: 180,
          }}>
            <div style={{ fontWeight: 600, marginBottom: 12, color: '#1890ff' }}>键盘快捷键</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '8px 16px' }}>
              <kbd style={{ background: '#f0f0f0', padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace' }}>Esc</kbd>
              <span>返回上一级</span>
              <kbd style={{ background: '#f0f0f0', padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace' }}>←</kbd>
              <span>历史后退</span>
              <kbd style={{ background: '#f0f0f0', padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace' }}>→</kbd>
              <span>历史前进</span>
              <kbd style={{ background: '#f0f0f0', padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace' }}>R</kbd>
              <span>重置视图</span>
              <kbd style={{ background: '#f0f0f0', padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace' }}>?</kbd>
              <span>显示/隐藏帮助</span>
            </div>
            <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #eee', color: '#888', fontSize: 11 }}>
              鼠标操作: 左键旋转 / 右键平移 / 滚轮缩放
            </div>
          </div>
        </>
      )}

      {/* 左下角操作提示 */}
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
        {viewState.level === 'pod' && viewState.path.length === 0 && '单击查看详情 | 双击进入内部视图 | 按 ? 查看快捷键'}
        {viewState.level === 'pod' && viewState.path.length === 1 && '单击查看详情 | 双击机柜进入内部视图 | Esc返回'}
        {viewState.level === 'rack' && '单击查看详情 | 双击板卡查看芯片布局 | Esc返回'}
        {(viewState.level === 'board' || viewState.level === 'chip') && 'Esc返回上级 | R重置视图'}
      </div>
    </div>
  )
}
