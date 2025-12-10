// ============================================
// 层级类型定义
// ============================================

// 视图层级
export type ViewLevel = 'pod' | 'rack' | 'board' | 'chip';

// Chip类型
export type ChipType = 'npu' | 'cpu';

// ============================================
// 层级配置接口
// ============================================

// Chip配置
export interface ChipConfig {
  id: string;
  type: ChipType;
  position: [number, number];  // [行, 列]
  label?: string;
}

// Board配置
export interface BoardConfig {
  id: string;
  u_position: number;     // 起始U位 (1-42)
  u_height: number;       // 占用U数
  label: string;
  chips: ChipConfig[];
}

// Rack配置
export interface RackConfig {
  id: string;
  position: [number, number];  // 在Pod中的网格位置
  label: string;
  total_u: number;        // 总U数，默认42
  boards: BoardConfig[];
}

// Pod配置
export interface PodConfig {
  id: string;
  label: string;
  grid_size: [number, number];  // Rack排列网格 [行, 列]
  racks: RackConfig[];
}

// 连接配置
export interface ConnectionConfig {
  source: string;
  target: string;
  type: 'intra' | 'inter' | 'switch';
  bandwidth?: number;
  connection_role?: 'uplink' | 'downlink' | 'inter';  // Switch连接角色
}

// ============================================
// Switch配置接口
// ============================================

// Switch类型预定义
export interface SwitchTypeConfig {
  id: string;           // 类型标识，如 "leaf_72", "spine_512"
  name: string;         // 显示名称，如 "72端口Leaf交换机"
  port_count: number;   // 总端口数
}

// 单层Switch配置
export interface SwitchLayerConfig {
  layer_name: string;       // 层名称，如 "leaf", "spine"
  switch_type_id: string;   // 使用的Switch类型ID
  count: number;            // 该层Switch数量
  inter_connect: boolean;   // 同层Switch是否互联
}

// 直连拓扑类型
export type DirectTopologyType = 'none' | 'full_mesh' | 'hw_full_mesh' | 'ring' | 'torus_2d' | 'torus_3d';

// 层级Switch配置（支持多层Switch，如Leaf-Spine）
export interface HierarchyLevelSwitchConfig {
  enabled: boolean;                     // 是否启用该层级的Switch
  layers: SwitchLayerConfig[];          // Switch层列表（从下到上）
  downlink_redundancy: number;          // 下层设备连接几个Switch（冗余度）
  connect_to_upper_level: boolean;      // 是否连接到上层的Switch
  direct_topology?: DirectTopologyType; // 无Switch时的直连拓扑类型
}

// 全局Switch配置
export interface GlobalSwitchConfig {
  switch_types: SwitchTypeConfig[];                    // 预定义的Switch类型
  datacenter_level: HierarchyLevelSwitchConfig;        // Pod间Switch
  pod_level: HierarchyLevelSwitchConfig;               // Rack间Switch
  rack_level: HierarchyLevelSwitchConfig;              // Board间Switch
}

// Switch实例
export interface SwitchInstance {
  id: string;                                           // 唯一标识
  type_id: string;                                      // Switch类型ID
  layer: string;                                        // 所在层，如 "leaf", "spine"
  hierarchy_level: 'datacenter' | 'pod' | 'rack';       // 所属层级
  parent_id?: string;                                   // 父节点ID
  label: string;                                        // 显示标签
  uplink_ports_used: number;                            // 上行端口使用数
  downlink_ports_used: number;                          // 下行端口使用数
  inter_ports_used: number;                             // 同层互联端口使用数
}

// 完整拓扑数据
export interface HierarchicalTopology {
  pods: PodConfig[];
  connections: ConnectionConfig[];
  switches: SwitchInstance[];                           // Switch实例列表
  switch_config?: GlobalSwitchConfig;                   // Switch配置
}

// ============================================
// 视图状态
// ============================================

export interface ViewState {
  level: ViewLevel;
  path: string[];           // 当前路径 ['pod_0', 'rack_1', 'board_2']
  selectedNode?: string;
}

// 面包屑项
export interface BreadcrumbItem {
  level: ViewLevel;
  id: string;
  label: string;
}

// ============================================
// 常量定义
// ============================================

// 层级显示名称
export const LEVEL_NAMES: Record<ViewLevel, string> = {
  pod: 'Pod (机柜组)',
  rack: 'Rack (机柜)',
  board: 'Board (板卡)',
  chip: 'Chip (芯片)',
};

// Chip类型显示名称
export const CHIP_TYPE_NAMES: Record<ChipType, string> = {
  npu: 'NPU',
  cpu: 'CPU',
};

// 层级颜色
export const LEVEL_COLORS: Record<ViewLevel, string> = {
  pod: '#fa8c16',      // 橙色
  rack: '#1890ff',     // 蓝色
  board: '#52c41a',    // 绿色
  chip: '#722ed1',     // 紫色
};

// Chip类型颜色
export const CHIP_TYPE_COLORS: Record<ChipType, string> = {
  npu: '#eb2f96',      // 粉色
  cpu: '#1890ff',      // 蓝色
};

// Switch层级颜色
export const SWITCH_LAYER_COLORS: Record<string, string> = {
  leaf: '#13c2c2',     // 青色
  spine: '#faad14',    // 金色
  core: '#f5222d',     // 红色
};

// Switch层级显示名称
export const SWITCH_LAYER_NAMES: Record<string, string> = {
  leaf: 'Leaf交换机',
  spine: 'Spine交换机',
  core: '核心交换机',
};

// ============================================
// 物理尺寸常量 (3D世界单位)
// ============================================

// Rack尺寸
export const RACK_DIMENSIONS = {
  width: 0.6,           // 19英寸标准宽度
  depth: 1.0,           // 深度
  uHeight: 0.0445,      // 单U高度 (1U ≈ 4.45cm)
  totalU: 42,           // 标准42U
  get fullHeight() { return this.totalU * this.uHeight; },
};

// Board尺寸
export const BOARD_DIMENSIONS = {
  width: 0.5,
  depth: 0.8,
  height: 0.04,
};

// Chip尺寸 (根据类型不同) [width(x), height(y-厚度), depth(z)]
export const CHIP_DIMENSIONS: Record<ChipType, [number, number, number]> = {
  npu: [0.07, 0.02, 0.07],
  cpu: [0.06, 0.02, 0.06],
};

// 相机预设位置
export const CAMERA_PRESETS: Record<ViewLevel, [number, number, number]> = {
  pod: [5, 4, 5],
  rack: [2, 2, 3],
  board: [1, 0.8, 1],
  chip: [0.3, 0.25, 0.35],
};

// 相机距离限制
export const CAMERA_DISTANCE: Record<ViewLevel, { min: number; max: number }> = {
  pod: { min: 2, max: 30 },
  rack: { min: 0.5, max: 8 },
  board: { min: 0.3, max: 3 },
  chip: { min: 0.1, max: 1 },
};
