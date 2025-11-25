"""
Cytoscape.js静态带宽可视化器 - 完全匹配Web工具实现

复刻Web工具(MultiDieTopologyGraph.tsx)的精确视觉效果
"""

import json
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


# IP类型优先级排序（与Web工具一致）
IP_PRIORITY = ['gdma', 'ddr', 'sdma', 'cdma', 'npu', 'pcie', 'eth', 'l2m', 'd2d']


def get_ip_type_color(ip_type: str) -> Dict[str, str]:
    """
    获取IP类型的颜色配置（背景色+边框色）

    Args:
        ip_type: IP类型字符串（如"gdma_0"）

    Returns:
        {'bg': 背景色, 'border': 边框色}
    """
    type_prefix = ip_type.split('_')[0].lower()

    color_map = {
        'gdma': {'bg': '#5B8FF9', 'border': '#3A6FD9'},
        'sdma': {'bg': '#5B8FF9', 'border': '#3A6FD9'},
        'cdma': {'bg': '#5AD8A6', 'border': '#3AB886'},
        'npu': {'bg': '#722ed1', 'border': '#531dab'},
        'ddr': {'bg': '#E8684A', 'border': '#C8482A'},
        'l2m': {'bg': '#E8684A', 'border': '#C8482A'},
        'pcie': {'bg': '#fa8c16', 'border': '#d46b08'},
        'eth': {'bg': '#52c41a', 'border': '#389e0d'},
        'd2d': {'bg': '#13c2c2', 'border': '#08979c'},
    }

    return color_map.get(type_prefix, {'bg': '#eb2f96', 'border': '#c41d7f'})


def get_bandwidth_color(bandwidth_gbps: float) -> str:
    """
    获取带宽颜色（连续渐变：浅红→深红）

    Args:
        bandwidth_gbps: 带宽值(GB/s)

    Returns:
        RGB颜色字符串
    """
    if bandwidth_gbps == 0:
        return '#bfbfbf'  # 灰色：无流量

    # 0-256 GB/s 映射到 0-1
    ratio = min(bandwidth_gbps / 256, 1)

    # 从浅红(255,180,180)到深红(200,0,0)
    r = round(255 - ratio * 55)
    g = round(180 * (1 - ratio))
    b = round(180 * (1 - ratio))

    return f'rgb({r},{g},{b})'


def get_bandwidth_width(bandwidth_gbps: float) -> int:
    """
    获取带宽线宽

    Args:
        bandwidth_gbps: 带宽值(GB/s)

    Returns:
        线宽（像素）
    """
    if bandwidth_gbps == 0:
        return 2
    elif bandwidth_gbps < 50:
        return 2
    elif bandwidth_gbps < 100:
        return 3
    else:
        return 4


def get_ip_short_label(ip_type: str) -> str:
    """
    获取IP类型的简短标签（如"gdma_0" -> "G0"）

    Args:
        ip_type: IP类型字符串

    Returns:
        简短标签
    """
    parts = ip_type.split('_')
    if len(parts) >= 2:
        prefix = parts[0][0].upper()  # 取首字母大写
        suffix = parts[1]
        return f"{prefix}{suffix}"
    return ip_type[:2].upper()


class CytoscapeBandwidthVisualizer:
    """Cytoscape.js静态带宽可视化器（完全匹配Web工具）"""

    # 布局常量（与Web工具一致）
    NODE_SPACING = 150  # 节点间距
    NODE_SIZE = 70      # 节点尺寸
    DIE_GAP_BASE = 100  # Die间隙基础值

    def __init__(
        self,
        topo_type: str,
        mode: str,
        link_bandwidth: Dict,
        node_ips: Dict[int, List[str]],
        d2d_config: Optional[Dict] = None
    ):
        """
        初始化可视化器

        Args:
            topo_type: 拓扑类型(如"5x4")
            mode: 模式("noc"或"d2d")
            link_bandwidth: 链路带宽字典
            node_ips: 节点IP映射 {node_id: [ip_list]}
            d2d_config: D2D配置(包含num_dies和d2d_connections)
        """
        self.topo_type = topo_type
        self.mode = mode
        self.link_bandwidth = link_bandwidth
        self.node_ips = node_ips
        self.d2d_config = d2d_config or {}

        # 解析拓扑类型
        self.rows, self.cols = map(int, topo_type.split('x'))
        self.num_nodes = self.rows * self.cols

        # D2D相关
        self.num_dies = self.d2d_config.get('num_dies', 1)
        self.d2d_connections = self.d2d_config.get('d2d_connections', [])

    def _calculate_ip_layout(self, ips: List[str]) -> Dict[str, Any]:
        """
        计算IP块的网格布局

        Args:
            ips: IP列表

        Returns:
            布局信息字典
        """
        if not ips:
            return {'positions': [], 'ip_size': 30}

        # 按优先级排序
        def get_priority(ip: str) -> int:
            prefix = ip.split('_')[0].lower()
            try:
                return IP_PRIORITY.index(prefix)
            except ValueError:
                return len(IP_PRIORITY)

        sorted_ips = sorted(ips, key=get_priority)

        # 按类型分组计数
        type_groups = defaultdict(int)
        for ip in sorted_ips:
            type_prefix = ip.split('_')[0].lower()
            type_groups[type_prefix] += 1

        # 计算网格布局（尽量接近正方形）
        num_ips = len(sorted_ips)
        if num_ips == 1:
            num_rows = 1
            num_cols = 1
        elif num_ips == 2:
            num_rows = 1
            num_cols = 2
        elif num_ips <= 4:
            num_rows = 2
            num_cols = 2
        elif num_ips <= 6:
            num_rows = 2
            num_cols = 3
        else:
            num_cols = 3
            num_rows = (num_ips + num_cols - 1) // num_cols

        # 计算IP块大小（动态调整）
        base_size = 30
        max_size = 32
        min_size = 26

        if num_rows <= 1 and num_cols <= 1:
            ip_size = max_size
        elif num_rows >= 2 or num_cols >= 2:
            ip_size = min_size
        else:
            ip_size = base_size

        spacing = ip_size + 4

        # 计算每个IP的相对位置（相对于节点中心）
        positions = []
        total_height = num_rows * spacing
        total_width = num_cols * spacing
        start_y = -total_height / 2 + spacing / 2
        start_x = -total_width / 2 + spacing / 2

        for idx, ip in enumerate(sorted_ips):
            row = idx // num_cols
            col = idx % num_cols

            # 计算当前行的实际IP数量
            row_start = row * num_cols
            row_end = min(row_start + num_cols, num_ips)
            num_in_row = row_end - row_start

            # 当前行的宽度
            row_width = num_in_row * spacing
            row_start_x = -row_width / 2 + spacing / 2

            x = row_start_x + col * spacing
            y = start_y + row * spacing

            positions.append({
                'ip': ip,
                'x': x,
                'y': y,
                'label': get_ip_short_label(ip)
            })

        return {
            'positions': positions,
            'ip_size': ip_size
        }

    def _get_node_position_noc(self, node_id: int) -> Tuple[float, float]:
        """计算NoC单Die拓扑中节点的位置（0在左上角）"""
        row = node_id // self.cols
        col = node_id % self.cols

        x = col * self.NODE_SPACING
        y = row * self.NODE_SPACING  # 0在左上角

        return (x, y)

    def _get_node_position_d2d(self, die_id: int, node_id: int) -> Tuple[float, float]:
        """计算D2D多Die拓扑中节点的位置（0在左上角）"""
        row = node_id // self.cols
        col = node_id % self.cols

        base_x = col * self.NODE_SPACING
        base_y = row * self.NODE_SPACING  # 0在左上角

        # Die间距
        die_spacing = (self.cols + 2) * self.NODE_SPACING

        offset_x = die_id * die_spacing
        offset_y = 0

        return (base_x + offset_x, base_y + offset_y)

    def _get_all_adjacent_links(self) -> List[Tuple[int, int]]:
        """
        获取拓扑中所有相邻节点的链路对

        Returns:
            [(src_node, dst_node), ...] 列表
        """
        links = []
        for node_id in range(self.num_nodes):
            row = node_id // self.cols
            col = node_id % self.cols

            # 右侧邻居
            if col < self.cols - 1:
                neighbor = node_id + 1
                links.append((node_id, neighbor))

            # 下方邻居
            if row < self.rows - 1:
                neighbor = node_id + self.cols
                links.append((node_id, neighbor))

        return links

    def _create_edge_offset_data(self, src_pos: Tuple[float, float],
                                  dst_pos: Tuple[float, float],
                                  is_forward: bool) -> Dict[str, Any]:
        """
        计算边的偏移和方向数据（用于双向边平行显示）

        Args:
            src_pos: 源节点位置
            dst_pos: 目标节点位置
            is_forward: 是否是正向边

        Returns:
            包含direction和offset的字典
        """
        dx = abs(dst_pos[0] - src_pos[0])
        dy = abs(dst_pos[1] - src_pos[1])

        # 判断方向
        if dx > dy:
            direction = 'horizontal'
        elif dy > dx:
            direction = 'vertical'
        else:
            direction = 'diagonal'

        # 偏移量（正向+8，反向-8）
        offset = 8 if is_forward else -8

        return {
            'direction': direction,
            'offset': offset
        }

    def generate_cytoscape_data_noc(self) -> Dict[str, List]:
        """生成NoC模式的Cytoscape数据"""
        nodes = []
        edges = []

        # 生成节点（三层结构）
        for node_id in range(self.num_nodes):
            x, y = self._get_node_position_noc(node_id)
            node_ips = self.node_ips.get(node_id, [])

            # 层1：节点背景（灰色矩形）
            has_ip = len(node_ips) > 0
            nodes.append({
                'data': {
                    'id': f'node-{node_id}',
                    'node_id': node_id
                },
                'position': {'x': x, 'y': y},
                'classes': 'mounted-node' if has_ip else 'unmounted'
            })

            # 层2：IP块
            if node_ips:
                ip_layout = self._calculate_ip_layout(node_ips)
                for ip_info in ip_layout['positions']:
                    colors = get_ip_type_color(ip_info['ip'])
                    nodes.append({
                        'data': {
                            'id': f'ip-{node_id}-{ip_info["ip"]}',
                            'label': ip_info['label'],
                            'bgColor': colors['bg'],
                            'borderColor': colors['border'],
                            'ipSize': ip_layout['ip_size']
                        },
                        'position': {
                            'x': x + ip_info['x'],
                            'y': y + ip_info['y']
                        },
                        'classes': 'ip-block'
                    })

            # 层3：节点编号标签
            nodes.append({
                'data': {
                    'id': f'label-{node_id}',
                    'label': str(node_id)
                },
                'position': {'x': x, 'y': y},
                'classes': 'node-label'
            })

        # 生成边（显示所有链路，包括0带宽）
        edge_id = 0
        processed_pairs = {}
        all_links = self._get_all_adjacent_links()

        # 创建带宽查找字典
        bandwidth_dict = {}
        for link_key, bandwidth in self.link_bandwidth.items():
            if isinstance(link_key, tuple) and len(link_key) == 2:
                (col1, row1), (col2, row2) = link_key
                src_node = row1 * self.cols + col1
                dst_node = row2 * self.cols + col2
                pair_key = tuple(sorted([src_node, dst_node]))
                bandwidth_dict[pair_key] = bandwidth

        # 为每条链路生成双向边（两条箭头）
        for src_node, dst_node in all_links:
            # 获取位置
            src_pos = self._get_node_position_noc(src_node)
            dst_pos = self._get_node_position_noc(dst_node)

            # 查找带宽
            pair_key = tuple(sorted([src_node, dst_node]))
            bandwidth = bandwidth_dict.get(pair_key, 0)

            color = get_bandwidth_color(bandwidth)
            width = get_bandwidth_width(bandwidth)

            # 生成正向边（src -> dst，offset +8）
            offset_data_forward = self._create_edge_offset_data(src_pos, dst_pos, True)
            edges.append({
                'data': {
                    'id': f'edge-{edge_id}',
                    'source': f'node-{src_node}',
                    'target': f'node-{dst_node}',
                    'bandwidth': round(bandwidth, 2),
                    'label': f'{bandwidth:.1f}' if bandwidth > 0 else '',
                    'bandwidthColor': color,
                    'bandwidthWidth': width,
                    'direction': offset_data_forward['direction'],
                    'offset': offset_data_forward['offset']
                },
                'classes': 'internal-edge'
            })
            edge_id += 1

            # 生成反向边（dst -> src，offset -8）
            offset_data_backward = self._create_edge_offset_data(dst_pos, src_pos, False)
            edges.append({
                'data': {
                    'id': f'edge-{edge_id}',
                    'source': f'node-{dst_node}',
                    'target': f'node-{src_node}',
                    'bandwidth': round(bandwidth, 2),
                    'label': '',  # 反向边不显示标签，避免重复
                    'bandwidthColor': color,
                    'bandwidthWidth': width,
                    'direction': offset_data_backward['direction'],
                    'offset': offset_data_backward['offset']
                },
                'classes': 'internal-edge'
            })
            edge_id += 1

        return {'nodes': nodes, 'edges': edges}

    def generate_cytoscape_data_d2d(self) -> Dict[str, List]:
        """生成D2D模式的Cytoscape数据"""
        nodes = []
        edges = []
        edge_id = 0

        # 生成各Die的节点
        for die_id in range(self.num_dies):
            # Die标签
            die_width = (self.cols - 1) * self.NODE_SPACING
            die_height = (self.rows - 1) * self.NODE_SPACING
            die_center_x = die_id * (self.cols + 2) * self.NODE_SPACING + die_width / 2
            die_center_y = die_height / 2

            # 判断Die在左列还是右列
            is_right_col = die_id > 0 and die_id == self.num_dies - 1

            nodes.append({
                'data': {
                    'id': f'die-label-{die_id}',
                    'label': f'Die {die_id}',
                    'dieId': die_id
                },
                'position': {
                    'x': die_center_x + (self.NODE_SPACING / 2 if is_right_col else -self.NODE_SPACING / 2 - 100),
                    'y': die_center_y
                },
                'classes': 'die-label'
            })

            # Die内的节点
            for node_id in range(self.num_nodes):
                x, y = self._get_node_position_d2d(die_id, node_id)
                node_ips = self.node_ips.get(node_id, [])

                # 层1：节点背景
                has_ip = len(node_ips) > 0
                nodes.append({
                    'data': {
                        'id': f'die-{die_id}-node-{node_id}',
                        'die_id': die_id,
                        'node_id': node_id
                    },
                    'position': {'x': x, 'y': y},
                    'classes': 'mounted-node' if has_ip else 'unmounted'
                })

                # 层2：IP块
                if node_ips:
                    ip_layout = self._calculate_ip_layout(node_ips)
                    for ip_info in ip_layout['positions']:
                        colors = get_ip_type_color(ip_info['ip'])
                        nodes.append({
                            'data': {
                                'id': f'ip-{die_id}-{node_id}-{ip_info["ip"]}',
                                'label': ip_info['label'],
                                'bgColor': colors['bg'],
                                'borderColor': colors['border'],
                                'ipSize': ip_layout['ip_size']
                            },
                            'position': {
                                'x': x + ip_info['x'],
                                'y': y + ip_info['y']
                            },
                            'classes': 'ip-block'
                        })

                # 层3：节点编号标签
                nodes.append({
                    'data': {
                        'id': f'label-{die_id}-{node_id}',
                        'label': str(node_id)
                    },
                    'position': {'x': x, 'y': y},
                    'classes': 'node-label'
                })

        # 生成Die内链路（显示所有链路，包括0带宽）
        all_links = self._get_all_adjacent_links()

        for die_id in range(self.num_dies):
            processed_pairs = {}

            # 创建当前Die的带宽查找字典
            bandwidth_dict = {}
            if isinstance(self.link_bandwidth, dict):
                die_links = self.link_bandwidth.get(die_id, {})
                if isinstance(die_links, dict):
                    for link_key, bandwidth in die_links.items():
                        if isinstance(link_key, tuple) and len(link_key) == 2:
                            (col1, row1), (col2, row2) = link_key
                            src_node = row1 * self.cols + col1
                            dst_node = row2 * self.cols + col2
                            pair_key = tuple(sorted([src_node, dst_node]))
                            bandwidth_dict[pair_key] = bandwidth

            # 为当前Die的每条链路生成边
            for src_node, dst_node in all_links:
                src_pos = self._get_node_position_d2d(die_id, src_node)
                dst_pos = self._get_node_position_d2d(die_id, dst_node)

                # 查找带宽
                pair_key = tuple(sorted([src_node, dst_node]))
                bandwidth = bandwidth_dict.get(pair_key, 0)

                # 双向边处理
                if pair_key not in processed_pairs:
                    processed_pairs[pair_key] = []

                is_forward = len(processed_pairs[pair_key]) == 0
                processed_pairs[pair_key].append(edge_id)

                offset_data = self._create_edge_offset_data(src_pos, dst_pos, is_forward)

                color = get_bandwidth_color(bandwidth)
                width = get_bandwidth_width(bandwidth)

                edges.append({
                    'data': {
                        'id': f'edge-{edge_id}',
                        'source': f'die-{die_id}-node-{src_node}',
                        'target': f'die-{die_id}-node-{dst_node}',
                        'bandwidth': round(bandwidth, 2),
                        'label': f'{bandwidth:.1f}' if bandwidth > 0 else '',  # 0带宽不显示标签
                        'bandwidthColor': color,
                        'bandwidthWidth': width,
                        'direction': offset_data['direction'],
                        'offset': offset_data['offset']
                    },
                    'classes': 'internal-edge'
                })
                edge_id += 1

        # 生成D2D跨Die链路
        processed_d2d_pairs = {}

        for conn in self.d2d_connections:
            src_die, src_node, dst_die, dst_node = conn

            # 查找带宽
            bandwidth = 0
            for link_key, bw in self.link_bandwidth.items():
                if isinstance(link_key, str) and link_key == f'{src_die}-{src_node}-{dst_die}-{dst_node}':
                    bandwidth = bw
                    break

            if bandwidth <= 0:
                continue

            src_pos = self._get_node_position_d2d(src_die, src_node)
            dst_pos = self._get_node_position_d2d(dst_die, dst_node)

            # 双向边处理
            pair_key = (src_die, src_node, dst_die, dst_node)
            reverse_key = (dst_die, dst_node, src_die, src_node)

            is_forward = pair_key not in processed_d2d_pairs and reverse_key not in processed_d2d_pairs
            processed_d2d_pairs[pair_key] = edge_id

            offset_data = self._create_edge_offset_data(src_pos, dst_pos, is_forward)

            color = get_bandwidth_color(bandwidth)
            width = get_bandwidth_width(bandwidth)

            edges.append({
                'data': {
                    'id': f'd2d-edge-{edge_id}',
                    'source': f'die-{src_die}-node-{src_node}',
                    'target': f'die-{dst_die}-node-{dst_node}',
                    'bandwidth': round(bandwidth, 2),
                    'label': f'{bandwidth:.1f}',
                    'bandwidthColor': color,
                    'bandwidthWidth': width,
                    'direction': offset_data['direction'],
                    'offset': offset_data['offset']
                },
                'classes': 'd2d-edge'
            })
            edge_id += 1

        return {'nodes': nodes, 'edges': edges}

    def generate_cytoscape_data(self) -> Dict[str, List]:
        """生成Cytoscape数据(根据模式自动选择)"""
        if self.mode == 'd2d':
            return self.generate_cytoscape_data_d2d()
        else:
            return self.generate_cytoscape_data_noc()

    def generate_html_snippet(self) -> str:
        """生成完整的HTML代码片段"""
        cytoscape_data = self.generate_cytoscape_data()
        data_json = json.dumps(cytoscape_data, ensure_ascii=False)

        html = f"""
        <div id="cy-static-bandwidth" style="width: 100%; height: 700px; border: 1px solid #ccc; background: #f9f9f9;"></div>

        <script>
        // 先定义初始化函数
        function initCytoscapeBandwidth() {{
            if (typeof cytoscape === 'undefined') {{
                console.error('Cytoscape未加载');
                return;
            }}

            const cytoscapeData = {data_json};

            const cy = cytoscape({{
                container: document.getElementById('cy-static-bandwidth'),
                elements: cytoscapeData.nodes.concat(cytoscapeData.edges),

                style: [
                    // 节点背景层（灰色）
                    {{
                        selector: 'node.mounted-node, node.unmounted',
                        style: {{
                            'shape': 'rectangle',
                            'width': 70,
                            'height': 70,
                            'label': '',
                            'background-color': '#d9d9d9',
                            'border-width': 2,
                            'border-color': '#bfbfbf',
                            'z-index': 1
                        }}
                    }},

                    // 节点编号文本层（最上层）
                    {{
                        selector: '.node-label',
                        style: {{
                            'shape': 'rectangle',
                            'width': 1,
                            'height': 1,
                            'background-opacity': 0,
                            'border-width': 0,
                            'label': 'data(label)',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'color': '#000000',
                            'font-size': '23px',
                            'font-weight': 'bold',
                            'text-outline-width': 2,
                            'text-outline-color': '#ffffff',
                            'z-index': 100
                        }}
                    }},

                    // IP块层
                    {{
                        selector: '.ip-block',
                        style: {{
                            'shape': 'rectangle',
                            'width': 'data(ipSize)',
                            'height': 'data(ipSize)',
                            'label': 'data(label)',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'background-color': 'data(bgColor)',
                            'border-width': 1,
                            'border-color': 'data(borderColor)',
                            'color': '#fff',
                            'font-size': '10px',
                            'font-weight': 'bold',
                            'z-index': 15
                        }}
                    }},

                    // Die标签
                    {{
                        selector: '.die-label',
                        style: {{
                            'shape': 'rectangle',
                            'width': 1,
                            'height': 1,
                            'background-opacity': 0,
                            'border-width': 0,
                            'label': 'data(label)',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'font-size': '30px',
                            'font-weight': 'bold',
                            'color': '#1890ff',
                            'z-index': 100
                        }}
                    }},

                    // Die内部链路
                    {{
                        selector: '.internal-edge',
                        style: {{
                            'width': 'data(bandwidthWidth)',
                            'line-color': 'data(bandwidthColor)',
                            'target-arrow-color': 'data(bandwidthColor)',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'straight',
                            'source-distance-from-node': 36,
                            'target-distance-from-node': 36,
                            'arrow-scale': 1.0,
                            'opacity': 0.8,
                            'z-index': 5,
                            'label': 'data(label)',
                            'font-size': 16,
                            'color': '#d32029',
                            'text-background-color': '#fff',
                            'text-background-opacity': 0.8,
                            'text-background-padding': '2px',
                            'source-endpoint': function(ele) {{
                                const offset = ele.data('offset') || 0;
                                const direction = ele.data('direction');
                                if (direction === 'horizontal') {{
                                    return offset > 0 ? '0 8' : '0 -8';
                                }} else if (direction === 'vertical') {{
                                    return offset > 0 ? '-8 0' : '8 0';
                                }}
                                return '0 0';
                            }},
                            'target-endpoint': function(ele) {{
                                const offset = ele.data('offset') || 0;
                                const direction = ele.data('direction');
                                if (direction === 'horizontal') {{
                                    return offset > 0 ? '0 8' : '0 -8';
                                }} else if (direction === 'vertical') {{
                                    return offset > 0 ? '-8 0' : '8 0';
                                }}
                                return '0 0';
                            }},
                            'text-margin-x': function(ele) {{
                                const offset = ele.data('offset') || 0;
                                const direction = ele.data('direction');
                                if (direction === 'vertical') {{
                                    const label = ele.data('label') || '';
                                    const textWidth = label.length * 7.2;
                                    const baseOffset = 8;
                                    const dynamicOffset = baseOffset + textWidth / 2;
                                    return offset > 0 ? -dynamicOffset : dynamicOffset;
                                }}
                                return 0;
                            }},
                            'text-margin-y': function(ele) {{
                                const offset = ele.data('offset') || 0;
                                const direction = ele.data('direction');
                                if (direction === 'horizontal') {{
                                    return offset > 0 ? 12 : -12;
                                }}
                                return 0;
                            }}
                        }}
                    }},

                    // D2D跨Die链路
                    {{
                        selector: '.d2d-edge',
                        style: {{
                            'width': 'data(bandwidthWidth)',
                            'line-color': 'data(bandwidthColor)',
                            'target-arrow-color': 'data(bandwidthColor)',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'straight',
                            'line-style': 'dashed',
                            'line-dash-pattern': [10, 5],
                            'source-distance-from-node': 36,
                            'target-distance-from-node': 36,
                            'arrow-scale': 1.9,
                            'opacity': 0.9,
                            'z-index': 20,
                            'label': 'data(label)',
                            'font-size': 20,
                            'color': '#d32029',
                            'text-background-color': '#fff',
                            'text-background-opacity': 0.8,
                            'text-background-padding': '2px',
                            'text-rotation': 'autorotate',
                            'source-endpoint': function(ele) {{
                                const offset = ele.data('offset') || 0;
                                const direction = ele.data('direction');
                                if (direction === 'horizontal') {{
                                    return offset > 0 ? '0 8' : '0 -8';
                                }} else if (direction === 'vertical') {{
                                    return offset > 0 ? '-8 0' : '8 0';
                                }}
                                return '0 0';
                            }},
                            'target-endpoint': function(ele) {{
                                const offset = ele.data('offset') || 0;
                                const direction = ele.data('direction');
                                if (direction === 'horizontal') {{
                                    return offset > 0 ? '0 8' : '0 -8';
                                }} else if (direction === 'vertical') {{
                                    return offset > 0 ? '-8 0' : '8 0';
                                }}
                                return '0 0';
                            }}
                        }}
                    }},

                    // 选中状态
                    {{
                        selector: 'edge:selected',
                        style: {{
                            'line-color': '#0074D9',
                            'target-arrow-color': '#0074D9',
                            'z-index': 999
                        }}
                    }}
                ],

                layout: {{
                    name: 'preset'
                }},

                userZoomingEnabled: true,
                userPanningEnabled: false,
                autoungrabify: true,
                boxSelectionEnabled: false
            }});

            // 添加交互事件
            cy.on('tap', 'edge', function(evt) {{
                const edge = evt.target;
                const bandwidth = edge.data('bandwidth');
                console.log('点击链路，带宽:', bandwidth, 'GB/s');

                cy.elements().removeClass('highlighted');
                edge.addClass('highlighted');
            }});

            // 自适应视图
            cy.fit(cy.elements(), 50);
        }}
        </script>

        <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.0/dist/cytoscape.min.js"
                onload="initCytoscapeBandwidth()"></script>
        """

        return html
