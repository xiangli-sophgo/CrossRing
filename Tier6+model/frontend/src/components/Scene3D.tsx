import React, { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Text, Line } from '@react-three/drei'
import { Spin } from 'antd'
import { TopologyData, NodeData, EdgeData, LEVEL_COLORS } from '../types'
import * as THREE from 'three'

interface Scene3DProps {
  topologyData: TopologyData | null
  loading: boolean
}

// 节点形状组件
const NodeShape: React.FC<{
  node: NodeData
  onClick?: () => void
}> = ({ node, onClick }) => {
  const { level, position, color } = node

  // 根据层级返回不同形状
  const geometry = useMemo(() => {
    switch (level) {
      case 'die':
        return <boxGeometry args={[0.3, 0.3, 0.1]} />
      case 'chip':
        return <boxGeometry args={[0.5, 0.5, 0.15]} />
      case 'board':
        return <boxGeometry args={[0.8, 0.6, 0.05]} />
      case 'server':
        return <boxGeometry args={[1.0, 0.4, 0.15]} />
      case 'pod':
        return <boxGeometry args={[0.8, 0.8, 1.2]} />
      default:
        return <boxGeometry args={[0.4, 0.4, 0.4]} />
    }
  }, [level])

  return (
    <mesh
      position={position}
      onClick={onClick}
      castShadow
      receiveShadow
    >
      {geometry}
      <meshStandardMaterial
        color={color}
        metalness={0.3}
        roughness={0.4}
      />
    </mesh>
  )
}

// 连接线组件
const EdgeLine: React.FC<{
  edge: EdgeData
  nodes: Map<string, NodeData>
}> = ({ edge, nodes }) => {
  const sourceNode = nodes.get(edge.source)
  const targetNode = nodes.get(edge.target)

  if (!sourceNode || !targetNode) return null

  const points = [
    new THREE.Vector3(...sourceNode.position),
    new THREE.Vector3(...targetNode.position),
  ]

  const isInterLevel = edge.type === 'inter_level'

  return (
    <Line
      points={points}
      color={isInterLevel ? '#cccccc' : LEVEL_COLORS[sourceNode.level] || '#999999'}
      lineWidth={isInterLevel ? 1 : 2}
      dashed={isInterLevel}
      dashSize={0.2}
      gapSize={0.1}
    />
  )
}

// 3D场景内容
const SceneContent: React.FC<{
  topologyData: TopologyData
}> = ({ topologyData }) => {
  const { nodes, edges } = topologyData

  // 创建节点映射
  const nodeMap = useMemo(() => {
    const map = new Map<string, NodeData>()
    nodes.forEach(node => map.set(node.id, node))
    return map
  }, [nodes])

  return (
    <>
      {/* 灯光 */}
      <ambientLight intensity={0.5} />
      <directionalLight
        position={[10, 10, 5]}
        intensity={1}
        castShadow
      />
      <directionalLight
        position={[-10, -10, -5]}
        intensity={0.3}
      />

      {/* 网格辅助 */}
      <gridHelper
        args={[20, 20, '#dddddd', '#eeeeee']}
        position={[0, 0, -0.5]}
        rotation={[Math.PI / 2, 0, 0]}
      />

      {/* 渲染连接线 */}
      {edges.map((edge, index) => (
        <EdgeLine
          key={`edge-${index}`}
          edge={edge}
          nodes={nodeMap}
        />
      ))}

      {/* 渲染节点 */}
      {nodes.map(node => (
        <NodeShape
          key={node.id}
          node={node}
        />
      ))}
    </>
  )
}

export const Scene3D: React.FC<Scene3DProps> = ({
  topologyData,
  loading,
}) => {
  if (loading) {
    return (
      <div style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#f0f2f5',
      }}>
        <Spin size="large" tip="加载中..." />
      </div>
    )
  }

  return (
    <div style={{ width: '100%', height: '100%', background: '#f0f2f5' }}>
      <Canvas shadows>
        <PerspectiveCamera
          makeDefault
          position={[15, 15, 15]}
          fov={50}
        />
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={5}
          maxDistance={50}
        />

        {topologyData && <SceneContent topologyData={topologyData} />}
      </Canvas>

      {/* 图例 */}
      <div style={{
        position: 'absolute',
        bottom: 16,
        right: 16,
        background: 'rgba(255,255,255,0.9)',
        padding: 12,
        borderRadius: 8,
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: 8 }}>图例</div>
        {Object.entries(LEVEL_COLORS).map(([level, color]) => (
          <div key={level} style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
            <div style={{
              width: 16,
              height: 16,
              background: color,
              marginRight: 8,
              borderRadius: 2,
            }} />
            <span style={{ textTransform: 'capitalize' }}>{level}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
