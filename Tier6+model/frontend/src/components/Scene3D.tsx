import React, { useMemo, useRef, useState, useCallback, useEffect } from 'react'
import { Canvas, useFrame, ThreeEvent, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Text, Html } from '@react-three/drei'
import { Breadcrumb, Button, Tooltip } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'
import {
  HierarchicalTopology,
  PodConfig,
  RackConfig,
  BoardConfig,
  ChipConfig,
  ViewState,
  BreadcrumbItem,
  RACK_DIMENSIONS,
  BOARD_DIMENSIONS,
  CHIP_DIMENSIONS,
  CHIP_TYPE_NAMES,
  CAMERA_PRESETS,
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
// 相机动画控制器
// ============================================

interface CameraAnimationTarget {
  position: THREE.Vector3
  lookAt: THREE.Vector3
}

const CameraController: React.FC<{
  target: CameraAnimationTarget
  duration?: number
  onAnimationComplete?: () => void
  enabled?: boolean
}> = ({ target, duration = 1.0, onAnimationComplete, enabled = true }) => {
  const { camera } = useThree()
  const controlsRef = useRef<any>(null)

  // 动画状态
  const isAnimating = useRef(false)
  const startPosition = useRef(new THREE.Vector3())
  const startTarget = useRef(new THREE.Vector3())
  const progress = useRef(0)
  const lastTarget = useRef<CameraAnimationTarget | null>(null)
  const pendingCallback = useRef<(() => void) | null>(null)

  // 目标变化时启动动画
  useEffect(() => {
    // 检查目标是否真的变化了
    if (lastTarget.current &&
        lastTarget.current.position.equals(target.position) &&
        lastTarget.current.lookAt.equals(target.lookAt)) {
      return
    }

    // 记录起始位置
    startPosition.current.copy(camera.position)
    if (controlsRef.current) {
      startTarget.current.copy(controlsRef.current.target)
    } else {
      startTarget.current.set(0, 0, 0)
    }

    progress.current = 0
    isAnimating.current = true
    lastTarget.current = {
      position: target.position.clone(),
      lookAt: target.lookAt.clone()
    }
    pendingCallback.current = onAnimationComplete || null
  }, [target.position.x, target.position.y, target.position.z,
      target.lookAt.x, target.lookAt.y, target.lookAt.z, camera, onAnimationComplete])

  // 每帧更新
  useFrame((_, delta) => {
    if (!isAnimating.current) return

    progress.current += delta / duration
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
  interactive?: boolean  // 是否可以交互（高亮和点击）
  opacity?: number  // 透明度
  onDoubleClick?: () => void
}> = ({ board, showChips = false, interactive = true, opacity = 1.0, onDoubleClick }) => {
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
          <group ref={groupRef} scale={scale}>
            <mesh
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
}> = ({ topology, focusPath, onNavigateToPod, onNavigateToRack, onNavigateToBoard }) => {
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

                    {/* 交互层 - 用于双击进入 */}
                    <mesh
                      visible={false}
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
                  />
                </group>
              )
            })
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
}) => {
  // 用于强制重置相机位置的 key
  const [resetKey, setResetKey] = useState(0)

  // 重置视图（相机位置）
  const handleResetView = useCallback(() => {
    setResetKey(k => k + 1)
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

    if (viewState.path.length === 1 && currentPod) {
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


  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* 3D Canvas */}
      <Canvas shadows>
        <PerspectiveCamera
          makeDefault
          position={[5, 4, 5]}
          fov={50}
        />

        {/* 使用 CameraController 实现平滑动画 - 由它控制相机位置 */}
        <CameraController
          target={cameraTarget}
          duration={1.5}
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
      </div>

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
