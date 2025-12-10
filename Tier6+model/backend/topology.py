"""
层级拓扑生成器

生成数据中心层级拓扑: Pod -> Rack -> Board -> Chip
"""

import math
from typing import List, Optional, Tuple, Dict
from models import (
    HierarchicalTopology, PodConfig, RackConfig,
    BoardConfig, ChipConfig, ConnectionConfig, ChipType,
    GlobalSwitchConfig, SwitchInstance, SwitchLayerConfig,
    HierarchyLevelSwitchConfig, SwitchTypeConfig
)


class HierarchicalTopologyGenerator:
    """层级拓扑生成器"""

    def __init__(self):
        self._cached_topology: Optional[HierarchicalTopology] = None

    def generate_default(self) -> HierarchicalTopology:
        """生成默认的示例拓扑"""
        return self.generate(
            pod_count=1,
            racks_per_pod=4,
            board_counts={'u1': 0, 'u2': 8, 'u4': 0},
            chip_counts={'npu': 8, 'cpu': 0}
        )

    def generate(
        self,
        pod_count: int = 1,
        racks_per_pod: int = 4,
        board_counts: Dict[str, int] = None,  # 旧格式：各U高度板卡数量 {'u1': 0, 'u2': 8, 'u4': 0}
        chip_types: List[ChipType] = None,
        chip_counts: dict = None,  # 旧格式：每种芯片的数量 {'npu': 8, 'cpu': 0}
        board_configs: dict = None,  # 新格式：按U高度分类的完整配置
        switch_config: dict = None  # Switch配置
    ) -> HierarchicalTopology:
        """根据配置生成拓扑"""
        if chip_types is None:
            chip_types = ['npu', 'cpu']

        # 处理新旧格式兼容
        if board_configs is not None:
            # 新格式：从board_configs中提取信息
            board_config_map = {
                4: board_configs.get('u4', {'count': 0, 'chips': {'npu': 16, 'cpu': 2}}),
                2: board_configs.get('u2', {'count': 8, 'chips': {'npu': 8, 'cpu': 0}}),
                1: board_configs.get('u1', {'count': 0, 'chips': {'npu': 2, 'cpu': 0}}),
            }
        else:
            # 旧格式：使用board_counts和chip_counts
            if chip_counts is None:
                chip_counts = {'npu': 8, 'cpu': 0}
            if board_counts is None:
                board_counts = {'u1': 0, 'u2': 8, 'u4': 0}
            board_config_map = {
                4: {'count': board_counts.get('u4', 0), 'chips': chip_counts},
                2: {'count': board_counts.get('u2', 8), 'chips': chip_counts},
                1: {'count': board_counts.get('u1', 0), 'chips': chip_counts},
            }

        pods = []
        connections = []

        for pod_idx in range(pod_count):
            pod_id = f"pod_{pod_idx}"
            racks = []

            # 计算Rack网格布局（根据数量智能选择列数，使布局接近正方形）
            if racks_per_pod <= 2:
                grid_cols = racks_per_pod
            elif racks_per_pod <= 4:
                grid_cols = 2
            elif racks_per_pod <= 6:
                grid_cols = 3
            elif racks_per_pod <= 9:
                grid_cols = 3
            elif racks_per_pod <= 12:
                grid_cols = 4
            elif racks_per_pod <= 16:
                grid_cols = 4
            else:
                grid_cols = int(math.ceil(math.sqrt(racks_per_pod)))
            grid_rows = (racks_per_pod + grid_cols - 1) // grid_cols

            for rack_idx in range(racks_per_pod):
                rack_id = f"rack_{rack_idx}"
                rack_full_id = f"{pod_id}/{rack_id}"
                boards = []

                # 按U高度生成板卡，从底部(U1)开始堆叠
                current_u = 1
                board_idx = 0

                # 按顺序生成各种U高度的板卡: 4U -> 2U -> 1U (大的在下面)
                for u_height in [4, 2, 1]:
                    config = board_config_map[u_height]
                    count = config.get('count', 0)
                    board_chip_counts = config.get('chips', {'npu': 8, 'cpu': 0})

                    for _ in range(count):
                        if current_u + u_height - 1 > 42:
                            break  # 超出机柜容量

                        board_id = f"board_{board_idx}"
                        board_full_id = f"{rack_full_id}/{board_id}"

                        # 生成Board上的Chip（使用该U高度对应的chip配置）
                        chips = self._generate_board_chips(
                            board_full_id,
                            board_chip_counts
                        )

                        boards.append(BoardConfig(
                            id=board_full_id,
                            u_position=current_u,
                            u_height=u_height,
                            label=f"Board-{board_idx} ({u_height}U)",
                            chips=chips,
                        ))

                        # 生成Chip间连接
                        chip_connections = self._generate_chip_connections(chips)
                        connections.extend(chip_connections)

                        current_u += u_height
                        board_idx += 1

                racks.append(RackConfig(
                    id=rack_full_id,
                    position=(rack_idx // grid_cols, rack_idx % grid_cols),
                    label=f"Rack-{rack_idx}",
                    total_u=42,
                    boards=boards,
                ))

                # 生成Board间连接（仅在没有Rack层Switch时使用直连）
                if switch_config is None or not switch_config.get('rack_level', {}).get('enabled'):
                    board_connections = self._generate_board_connections(boards)
                    connections.extend(board_connections)

            pods.append(PodConfig(
                id=pod_id,
                label=f"Pod-{pod_idx}",
                grid_size=(grid_rows, grid_cols),
                racks=racks,
            ))

            # 生成Rack间连接（仅在没有Pod层Switch时使用直连）
            if switch_config is None or not switch_config.get('pod_level', {}).get('enabled'):
                rack_connections = self._generate_rack_connections(
                    [r.id for r in racks]
                )
                connections.extend(rack_connections)

        # 生成Pod间连接（仅在没有Switch配置时使用直连）
        if pod_count > 1 and (switch_config is None or not switch_config.get('datacenter_level', {}).get('enabled')):
            pod_connections = self._generate_pod_connections(
                [p.id for p in pods]
            )
            connections.extend(pod_connections)

        # ============================================
        # Switch生成
        # ============================================
        switches = []
        switch_types_list = []

        if switch_config:
            switch_types_list = [
                {'id': t['id'], 'name': t['name'], 'port_count': t['port_count']}
                for t in switch_config.get('switch_types', [])
            ]

            # 1. Rack层Switch（Board间）
            rack_level_config = switch_config.get('rack_level', {})
            if rack_level_config.get('enabled') and rack_level_config.get('layers'):
                for pod in pods:
                    for rack in pod.racks:
                        board_ids = [b.id for b in rack.boards]
                        rack_switches, rack_switch_conns = self._generate_switch_connections(
                            switch_layers=[
                                {'layer_name': l['layer_name'], 'switch_type_id': l['switch_type_id'],
                                 'count': l['count'], 'inter_connect': l.get('inter_connect', False)}
                                for l in rack_level_config['layers']
                            ],
                            switch_types=switch_types_list,
                            devices=board_ids,
                            redundancy=rack_level_config.get('downlink_redundancy', 1),
                            parent_id=rack.id,
                            hierarchy_level='rack'
                        )
                        switches.extend(rack_switches)
                        connections.extend(rack_switch_conns)

            # 2. Pod层Switch（Rack间）
            pod_level_config = switch_config.get('pod_level', {})
            if pod_level_config.get('enabled') and pod_level_config.get('layers'):
                for pod in pods:
                    # 确定Pod层的下层设备：Rack还是Rack层顶层Switch
                    if rack_level_config.get('enabled') and rack_level_config.get('connect_to_upper_level', True):
                        # 连接到Rack层顶层Switch
                        device_ids = [
                            s.id for s in switches
                            if s.hierarchy_level == 'rack' and
                               s.parent_id and s.parent_id.startswith(pod.id)
                        ]
                        # 只取顶层
                        top_rack_switches = self._get_top_layer_switches(switches, 'rack')
                        device_ids = [s.id for s in top_rack_switches if s.parent_id and s.parent_id.startswith(pod.id)]
                    else:
                        # 直接连接到Rack
                        device_ids = [r.id for r in pod.racks]

                    if device_ids:
                        pod_switches, pod_switch_conns = self._generate_switch_connections(
                            switch_layers=[
                                {'layer_name': l['layer_name'], 'switch_type_id': l['switch_type_id'],
                                 'count': l['count'], 'inter_connect': l.get('inter_connect', False)}
                                for l in pod_level_config['layers']
                            ],
                            switch_types=switch_types_list,
                            devices=device_ids,
                            redundancy=pod_level_config.get('downlink_redundancy', 1),
                            parent_id=pod.id,
                            hierarchy_level='pod'
                        )
                        switches.extend(pod_switches)
                        connections.extend(pod_switch_conns)

            # 3. 数据中心层Switch（Pod间）
            dc_level_config = switch_config.get('datacenter_level', {})
            if dc_level_config.get('enabled') and dc_level_config.get('layers'):
                # 确定数据中心层的下层设备：Pod还是Pod层顶层Switch
                if pod_level_config.get('enabled') and pod_level_config.get('connect_to_upper_level', True):
                    # 连接到Pod层顶层Switch
                    top_pod_switches = self._get_top_layer_switches(switches, 'pod')
                    device_ids = [s.id for s in top_pod_switches]
                else:
                    # 直接连接到Pod
                    device_ids = [p.id for p in pods]

                if device_ids:
                    dc_switches, dc_switch_conns = self._generate_switch_connections(
                        switch_layers=[
                            {'layer_name': l['layer_name'], 'switch_type_id': l['switch_type_id'],
                             'count': l['count'], 'inter_connect': l.get('inter_connect', False)}
                            for l in dc_level_config['layers']
                        ],
                        switch_types=switch_types_list,
                        devices=device_ids,
                        redundancy=dc_level_config.get('downlink_redundancy', 1),
                        parent_id='',
                        hierarchy_level='datacenter'
                    )
                    switches.extend(dc_switches)
                    connections.extend(dc_switch_conns)

        topology = HierarchicalTopology(
            pods=pods,
            connections=connections,
            switches=switches,
            switch_config=switch_config
        )
        self._cached_topology = topology
        return topology

    def _generate_board_chips(
        self,
        board_id: str,
        chip_counts: dict
    ) -> List[ChipConfig]:
        """生成板卡上的芯片，根据chip_counts配置数量，智能居中排布"""
        chips = []

        # 计算总芯片数
        total_chips = sum(chip_counts.values())
        if total_chips == 0:
            return chips

        # 智能计算网格大小：尽量接近正方形
        cols = int(math.ceil(math.sqrt(total_chips)))

        # 按类型生成芯片
        type_order = ['npu', 'cpu']
        type_labels = {
            'npu': 'NPU',
            'cpu': 'CPU',
        }

        # 先收集所有芯片
        chip_list = []
        for chip_type in type_order:
            count = chip_counts.get(chip_type, 0)
            for i in range(count):
                chip_list.append({
                    'type': chip_type,
                    'label': f"{type_labels.get(chip_type, chip_type.upper())}-{i}",
                })

        # 存储原始行列位置（整数），前端负责居中计算
        for idx, chip_info in enumerate(chip_list):
            row = idx // cols
            col = idx % cols

            chips.append(ChipConfig(
                id=f"{board_id}/chip_{idx}",
                type=chip_info['type'],
                position=(row, col),
                label=chip_info['label'],
            ))

        return chips

    def _generate_chip_connections(
        self,
        chips: List[ChipConfig]
    ) -> List[ConnectionConfig]:
        """生成芯片间的连接"""
        connections = []

        # 找出不同类型的芯片
        npus = [c for c in chips if c.type == 'npu']
        cpus = [c for c in chips if c.type == 'cpu']

        # NPU <-> CPU 连接
        for npu in npus:
            for cpu in cpus:
                connections.append(ConnectionConfig(
                    source=npu.id,
                    target=cpu.id,
                    type='intra',
                    bandwidth=64.0,  # PCIe连接
                ))

        return connections

    def _generate_board_connections(
        self,
        boards: List[BoardConfig]
    ) -> List[ConnectionConfig]:
        """生成Board间的连接 (通过背板)"""
        connections = []

        # 相邻Board连接
        for i in range(len(boards) - 1):
            connections.append(ConnectionConfig(
                source=boards[i].id,
                target=boards[i + 1].id,
                type='intra',
                bandwidth=100.0,  # 100Gbps背板
            ))

        return connections

    def _generate_rack_connections(
        self,
        rack_ids: List[str]
    ) -> List[ConnectionConfig]:
        """生成Rack间的连接 (通过ToR Switch)"""
        connections = []

        # 全连接拓扑
        for i in range(len(rack_ids)):
            for j in range(i + 1, len(rack_ids)):
                connections.append(ConnectionConfig(
                    source=rack_ids[i],
                    target=rack_ids[j],
                    type='intra',
                    bandwidth=400.0,  # 400Gbps
                ))

        return connections

    def _generate_pod_connections(
        self,
        pod_ids: List[str]
    ) -> List[ConnectionConfig]:
        """生成Pod间的连接"""
        connections = []

        for i in range(len(pod_ids)):
            for j in range(i + 1, len(pod_ids)):
                connections.append(ConnectionConfig(
                    source=pod_ids[i],
                    target=pod_ids[j],
                    type='inter',
                    bandwidth=1600.0,  # 1.6Tbps
                ))

        return connections

    def get_cached_topology(self) -> HierarchicalTopology:
        """获取缓存的拓扑数据"""
        if self._cached_topology is None:
            return self.generate_default()
        return self._cached_topology

    def get_pod(self, pod_id: str) -> Optional[PodConfig]:
        """获取指定Pod"""
        topology = self.get_cached_topology()
        for pod in topology.pods:
            if pod.id == pod_id:
                return pod
        return None

    def get_rack(self, rack_id: str) -> Optional[RackConfig]:
        """获取指定Rack"""
        topology = self.get_cached_topology()
        for pod in topology.pods:
            for rack in pod.racks:
                if rack.id == rack_id:
                    return rack
        return None

    def get_board(self, board_id: str) -> Optional[BoardConfig]:
        """获取指定Board"""
        topology = self.get_cached_topology()
        for pod in topology.pods:
            for rack in pod.racks:
                for board in rack.boards:
                    if board.id == board_id:
                        return board
        return None

    def get_connections_for_level(
        self,
        level: str,
        parent_id: Optional[str] = None
    ) -> List[ConnectionConfig]:
        """获取指定层级的连接"""
        topology = self.get_cached_topology()
        connections = []

        for conn in topology.connections:
            # 根据层级筛选
            if level == 'rack':
                if '/rack_' in conn.source and '/board_' not in conn.source:
                    if parent_id is None or parent_id in conn.source:
                        connections.append(conn)
            elif level == 'board':
                if '/board_' in conn.source and '/chip_' not in conn.source:
                    if parent_id is None or parent_id in conn.source:
                        connections.append(conn)
            elif level == 'chip':
                if '/chip_' in conn.source:
                    if parent_id is None or parent_id in conn.source:
                        connections.append(conn)

        return connections

    # ============================================
    # Switch生成相关方法
    # ============================================

    def _generate_switch_connections(
        self,
        switch_layers: List[dict],
        switch_types: List[dict],
        devices: List[str],
        redundancy: int,
        parent_id: str,
        hierarchy_level: str
    ) -> Tuple[List[SwitchInstance], List[ConnectionConfig]]:
        """
        通用Switch连接生成算法

        Args:
            switch_layers: Switch层配置列表（从下到上，如[leaf, spine]）
            switch_types: Switch类型定义列表
            devices: 下层设备ID列表
            redundancy: 冗余度（每个设备连接几个Switch）
            parent_id: 父节点ID前缀
            hierarchy_level: 层级名称 ('datacenter', 'pod', 'rack')

        Returns:
            (switches, connections) 元组
        """
        switches = []
        connections = []

        if not switch_layers or not devices:
            return switches, connections

        # 获取Switch类型映射
        type_map = {t['id']: t for t in switch_types}

        # 1. 创建Switch实例
        layer_switches: Dict[str, List[SwitchInstance]] = {}
        for layer_idx, layer_config in enumerate(switch_layers):
            layer_name = layer_config['layer_name']
            switch_type = type_map.get(layer_config['switch_type_id'])
            if not switch_type:
                raise ValueError(f"未找到Switch类型: {layer_config['switch_type_id']}")

            layer_switches[layer_name] = []

            for i in range(layer_config['count']):
                switch_id = f"{parent_id}/{layer_name}_{i}" if parent_id else f"{layer_name}_{i}"
                switch = SwitchInstance(
                    id=switch_id,
                    type_id=layer_config['switch_type_id'],
                    layer=layer_name,
                    hierarchy_level=hierarchy_level,
                    parent_id=parent_id if parent_id else None,
                    label=f"{switch_type['name']}-{i}",
                    uplink_ports_used=0,
                    downlink_ports_used=0,
                    inter_ports_used=0
                )
                switches.append(switch)
                layer_switches[layer_name].append(switch)

        # 2. 设备连接到最底层Switch（轮询+冗余）
        bottom_layer = switch_layers[0]['layer_name']
        bottom_switches = layer_switches[bottom_layer]

        for dev_idx, device_id in enumerate(devices):
            for r in range(min(redundancy, len(bottom_switches))):
                switch_idx = (dev_idx + r) % len(bottom_switches)
                switch = bottom_switches[switch_idx]
                connections.append(ConnectionConfig(
                    source=device_id,
                    target=switch.id,
                    type='switch',
                    connection_role='downlink'
                ))
                switch.downlink_ports_used += 1

        # 3. 相邻层Switch全连接（如leaf -> spine）
        for i in range(len(switch_layers) - 1):
            lower_layer = switch_layers[i]['layer_name']
            upper_layer = switch_layers[i + 1]['layer_name']

            for lower_sw in layer_switches[lower_layer]:
                for upper_sw in layer_switches[upper_layer]:
                    connections.append(ConnectionConfig(
                        source=lower_sw.id,
                        target=upper_sw.id,
                        type='switch',
                        connection_role='uplink'
                    ))
                    lower_sw.uplink_ports_used += 1
                    upper_sw.downlink_ports_used += 1

        # 4. 同层Switch互联
        for layer_config in switch_layers:
            if layer_config.get('inter_connect', False):
                layer_name = layer_config['layer_name']
                sw_list = layer_switches[layer_name]
                for i in range(len(sw_list)):
                    for j in range(i + 1, len(sw_list)):
                        connections.append(ConnectionConfig(
                            source=sw_list[i].id,
                            target=sw_list[j].id,
                            type='switch',
                            connection_role='inter'
                        ))
                        sw_list[i].inter_ports_used += 1
                        sw_list[j].inter_ports_used += 1

        return switches, connections

    def _validate_port_usage(
        self,
        switches: List[SwitchInstance],
        switch_types: List[dict]
    ) -> List[str]:
        """
        验证Switch端口是否足够

        Returns:
            错误信息列表（空列表表示通过）
        """
        type_map = {t['id']: t for t in switch_types}
        errors = []

        for switch in switches:
            switch_type = type_map.get(switch.type_id)
            if not switch_type:
                errors.append(f"Switch {switch.id} 使用了未定义的类型: {switch.type_id}")
                continue

            total_used = switch.uplink_ports_used + switch.downlink_ports_used + switch.inter_ports_used

            if total_used > switch_type['port_count']:
                errors.append(
                    f"Switch {switch.id} ({switch_type['name']}) 端口不足: "
                    f"需要 {total_used} 端口 (上行:{switch.uplink_ports_used}, "
                    f"下行:{switch.downlink_ports_used}, 互联:{switch.inter_ports_used}), "
                    f"但只有 {switch_type['port_count']} 端口"
                )

        return errors

    def _get_top_layer_switches(
        self,
        switches: List[SwitchInstance],
        hierarchy_level: str,
        parent_id: Optional[str] = None
    ) -> List[SwitchInstance]:
        """获取指定层级的顶层Switch（用于跨层连接）"""
        # 筛选该层级和父节点的Switch
        level_switches = [
            s for s in switches
            if s.hierarchy_level == hierarchy_level and
               (parent_id is None or s.parent_id == parent_id)
        ]

        if not level_switches:
            return []

        # 找出最高层（按layer名称排序，spine > leaf）
        layer_order = {'leaf': 0, 'spine': 1, 'core': 2}
        max_layer = max(level_switches, key=lambda s: layer_order.get(s.layer, 0)).layer

        return [s for s in level_switches if s.layer == max_layer]

    def _get_bottom_layer_switches(
        self,
        switches: List[SwitchInstance],
        hierarchy_level: str,
        parent_id: Optional[str] = None
    ) -> List[SwitchInstance]:
        """获取指定层级的底层Switch（用于跨层连接）"""
        level_switches = [
            s for s in switches
            if s.hierarchy_level == hierarchy_level and
               (parent_id is None or s.parent_id == parent_id)
        ]

        if not level_switches:
            return []

        layer_order = {'leaf': 0, 'spine': 1, 'core': 2}
        min_layer = min(level_switches, key=lambda s: layer_order.get(s.layer, 0)).layer

        return [s for s in level_switches if s.layer == min_layer]
