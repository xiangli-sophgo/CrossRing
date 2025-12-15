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
    HierarchyLevelSwitchConfig, SwitchTypeConfig,
    ManualConnectionConfig
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
        rack_config: dict = None,  # 最新格式：灵活Rack配置
        switch_config: dict = None,  # Switch配置
        manual_connections: dict = None  # 手动连接配置
    ) -> HierarchicalTopology:
        """根据配置生成拓扑"""
        if chip_types is None:
            chip_types = ['npu', 'cpu']

        # 判断使用哪种配置模式
        use_flex_rack_config = rack_config is not None and rack_config.get('boards')

        # 处理新旧格式兼容（仅在不使用flex模式时）
        if not use_flex_rack_config:
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

                # 获取Rack总U数
                rack_total_u = rack_config.get('total_u', 42) if rack_config else 42

                # 计算Switch预留空间（汇总显示，只占用配置的高度）
                switch_reserved_u = 0
                switch_position = 'top'  # 默认在顶部
                switch_u_height = 1
                if switch_config:
                    inter_board_cfg = switch_config.get('inter_board', {})
                    if inter_board_cfg.get('enabled'):
                        switch_position = inter_board_cfg.get('switch_position', 'top')
                        switch_u_height = inter_board_cfg.get('switch_u_height', 1)
                        # 汇总显示，只预留配置的高度
                        switch_reserved_u = switch_u_height

                # 根据Switch位置确定Board的起始U位置
                if switch_position == 'bottom':
                    # Switch在底部，Board从Switch上方开始
                    board_start_u = switch_reserved_u + 1
                else:
                    # Switch在顶部或中间，Board从U1开始（middle模式后续会重新调整）
                    board_start_u = 1

                if use_flex_rack_config:
                    # ===== 灵活Rack配置模式 =====
                    current_u = board_start_u
                    board_idx = 0
                    for flex_board in rack_config['boards']:
                        u_height = flex_board.get('u_height', 2)
                        board_name = flex_board.get('name', f'Board')
                        board_count = flex_board.get('count', 1)
                        flex_chips = flex_board.get('chips', [])

                        # 根据count生成多个相同配置的板卡
                        for _ in range(board_count):
                            if current_u + u_height - 1 > rack_total_u:
                                break  # 超出机柜容量

                            board_full_id = f"{rack_full_id}/board_{board_idx}"

                            # 使用灵活配置生成Chip
                            chips = self._generate_board_chips_flex(board_full_id, flex_chips)

                            boards.append(BoardConfig(
                                id=board_full_id,
                                u_position=current_u,
                                u_height=u_height,
                                label=f"{board_name}-{board_idx}",
                                chips=chips,
                            ))

                            # 生成Chip间连接（根据inter_chip配置）
                            inter_chip_cfg = switch_config.get('inter_chip', {}) if switch_config else {}
                            inter_chip_enabled = inter_chip_cfg.get('enabled', False)
                            inter_chip_topo = inter_chip_cfg.get('direct_topology', 'none')
                            keep_direct = inter_chip_cfg.get('keep_direct_topology', False)
                            # 未启用Switch时生成直连，或启用Switch但keep_direct_topology为True时也生成直连
                            if not inter_chip_enabled or keep_direct:
                                chip_connections = self._generate_chip_connections(chips, inter_chip_topo)
                                connections.extend(chip_connections)

                            current_u += u_height
                            board_idx += 1
                else:
                    # ===== 传统配置模式 =====
                    # 按U高度生成板卡，从board_start_u开始堆叠
                    current_u = board_start_u
                    board_idx = 0

                    # 按顺序生成各种U高度的板卡: 4U -> 2U -> 1U (大的在下面)
                    for u_height in [4, 2, 1]:
                        config = board_config_map[u_height]
                        count = config.get('count', 0)
                        board_chip_counts = config.get('chips', {'npu': 8, 'cpu': 0})

                        for _ in range(count):
                            if current_u + u_height - 1 > rack_total_u:
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
                                label=f"Board-{board_idx}",
                                chips=chips,
                            ))

                            # 生成Chip间连接（根据inter_chip配置）
                            inter_chip_cfg = switch_config.get('inter_chip', {}) if switch_config else {}
                            inter_chip_enabled = inter_chip_cfg.get('enabled', False)
                            inter_chip_topo = inter_chip_cfg.get('direct_topology', 'none')
                            keep_direct = inter_chip_cfg.get('keep_direct_topology', False)
                            # 未启用Switch时生成直连，或启用Switch但keep_direct_topology为True时也生成直连
                            if not inter_chip_enabled or keep_direct:
                                chip_connections = self._generate_chip_connections(chips, inter_chip_topo)
                                connections.extend(chip_connections)

                            current_u += u_height
                            board_idx += 1

                # middle模式：在Board中间插入Switch空间，调整后半部分Board的u_position
                if switch_position == 'middle' and switch_reserved_u > 0 and len(boards) > 0:
                    # 计算中间位置
                    half_count = len(boards) // 2
                    if half_count > 0:
                        # 找到中间分界点：前half_count个Board的最高位置
                        sorted_boards = sorted(boards, key=lambda b: b.u_position)
                        split_board = sorted_boards[half_count - 1]
                        split_u = split_board.u_position + split_board.u_height  # 分界U位置

                        # 后半部分Board的u_position向上移动switch_reserved_u
                        for board in boards:
                            if board.u_position >= split_u:
                                board.u_position += switch_reserved_u

                racks.append(RackConfig(
                    id=rack_full_id,
                    position=(rack_idx // grid_cols, rack_idx % grid_cols),
                    label=f"Rack-{rack_idx}",
                    total_u=rack_total_u,
                    boards=boards,
                ))

                # 生成Board间连接
                inter_board_cfg = switch_config.get('inter_board', {}) if switch_config else {}
                inter_board_enabled = inter_board_cfg.get('enabled', False)
                inter_board_topo = inter_board_cfg.get('direct_topology', 'full_mesh')
                keep_direct = inter_board_cfg.get('keep_direct_topology', False)
                # 未启用Switch时生成直连，或启用Switch但keep_direct_topology为True时也生成直连
                if not inter_board_enabled or keep_direct:
                    board_connections = self._generate_board_connections(boards, inter_board_topo)
                    connections.extend(board_connections)

            pods.append(PodConfig(
                id=pod_id,
                label=f"Pod-{pod_idx}",
                grid_size=(grid_rows, grid_cols),
                racks=racks,
            ))

            # 生成Rack间连接
            inter_rack_cfg = switch_config.get('inter_rack', {}) if switch_config else {}
            inter_rack_enabled = inter_rack_cfg.get('enabled', False)
            inter_rack_topo = inter_rack_cfg.get('direct_topology', 'full_mesh')
            keep_direct = inter_rack_cfg.get('keep_direct_topology', False)
            # 未启用Switch时生成直连，或启用Switch但keep_direct_topology为True时也生成直连
            if not inter_rack_enabled or keep_direct:
                rack_connections = self._generate_rack_connections(
                    [r.id for r in racks],
                    inter_rack_topo
                )
                connections.extend(rack_connections)

        # 生成Pod间连接
        if pod_count > 1:
            dc_level_cfg = switch_config.get('inter_pod', {}) if switch_config else {}
            dc_level_enabled = dc_level_cfg.get('enabled', False)
            dc_level_topo = dc_level_cfg.get('direct_topology', 'full_mesh')
            keep_direct = dc_level_cfg.get('keep_direct_topology', False)
            # 未启用Switch时生成直连，或启用Switch但keep_direct_topology为True时也生成直连
            if not dc_level_enabled or keep_direct:
                pod_connections = self._generate_pod_connections(
                    [p.id for p in pods],
                    dc_level_topo
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

            # 0. Board层Switch（Chip间）
            inter_chip_config = switch_config.get('inter_chip', {})
            if inter_chip_config.get('enabled') and inter_chip_config.get('layers'):
                for pod in pods:
                    for rack in pod.racks:
                        for board in rack.boards:
                            chip_ids = [c.id for c in board.chips]
                            if chip_ids:
                                board_switches, board_switch_conns = self._generate_switch_connections(
                                    switch_layers=[
                                        {'layer_name': l['layer_name'], 'switch_type_id': l['switch_type_id'],
                                         'count': l['count'], 'inter_connect': l.get('inter_connect', False)}
                                        for l in inter_chip_config['layers']
                                    ],
                                    switch_types=switch_types_list,
                                    devices=chip_ids,
                                    redundancy=inter_chip_config.get('downlink_redundancy', 1),
                                    parent_id=board.id,
                                    hierarchy_level='inter_chip',
                                    connection_mode=inter_chip_config.get('connection_mode', 'full_mesh'),
                                    group_config=inter_chip_config.get('group_config'),
                                    custom_connections=inter_chip_config.get('custom_connections')
                                )
                                switches.extend(board_switches)
                                connections.extend(board_switch_conns)

            # 1. Rack层Switch（Board间）
            inter_board_config = switch_config.get('inter_board', {})
            if inter_board_config.get('enabled') and inter_board_config.get('layers'):
                switch_position = inter_board_config.get('switch_position', 'top')
                switch_u_height = inter_board_config.get('switch_u_height', 1)

                for pod in pods:
                    for rack in pod.racks:
                        # 确定下层设备：Board还是Board层顶层Switch
                        if inter_chip_config.get('enabled') and inter_chip_config.get('connect_to_upper_level', True):
                            # 连接到Board层顶层Switch
                            top_board_switches = self._get_top_layer_switches(switches, 'inter_chip')
                            device_ids = [s.id for s in top_board_switches if s.parent_id and s.parent_id.startswith(rack.id)]
                        else:
                            # 直接连接到Board
                            device_ids = [b.id for b in rack.boards]

                        if not device_ids:
                            continue

                        rack_switches, rack_switch_conns = self._generate_switch_connections(
                            switch_layers=[
                                {'layer_name': l['layer_name'], 'switch_type_id': l['switch_type_id'],
                                 'count': l['count'], 'inter_connect': l.get('inter_connect', False)}
                                for l in inter_board_config['layers']
                            ],
                            switch_types=switch_types_list,
                            devices=device_ids,
                            redundancy=inter_board_config.get('downlink_redundancy', 1),
                            parent_id=rack.id,
                            hierarchy_level='inter_board',
                            connection_mode=inter_board_config.get('connection_mode', 'round_robin'),
                            group_config=inter_board_config.get('group_config'),
                            custom_connections=inter_board_config.get('custom_connections')
                        )

                        # 为rack层Switch分配u_position（所有Switch共用同一位置，汇总显示）
                        if switch_position == 'bottom':
                            # 底部：从U1开始
                            switch_u = 1
                        elif switch_position == 'middle':
                            # 中间：放在Board分界位置（前半部分Board之后）
                            if rack.boards:
                                half_count = len(rack.boards) // 2
                                if half_count > 0:
                                    sorted_boards = sorted(rack.boards, key=lambda b: b.u_position)
                                    split_board = sorted_boards[half_count - 1]
                                    switch_u = split_board.u_position + split_board.u_height
                                else:
                                    switch_u = 1
                            else:
                                switch_u = 1
                        else:
                            # 顶部：从最高Board之后开始
                            max_board_u = max((b.u_position + b.u_height - 1 for b in rack.boards), default=0)
                            switch_u = max_board_u + 1

                        # 所有Switch共用同一个u_position
                        for sw in rack_switches:
                            sw.u_position = switch_u
                            sw.u_height = switch_u_height

                        switches.extend(rack_switches)
                        connections.extend(rack_switch_conns)

            # 2. Pod层Switch（Rack间）
            inter_rack_config = switch_config.get('inter_rack', {})
            if inter_rack_config.get('enabled') and inter_rack_config.get('layers'):
                for pod in pods:
                    # 确定Pod层的下层设备：Rack还是Rack层顶层Switch
                    if inter_board_config.get('enabled') and inter_board_config.get('connect_to_upper_level', True):
                        # 连接到Rack层顶层Switch
                        device_ids = [
                            s.id for s in switches
                            if s.hierarchy_level == 'inter_board' and
                               s.parent_id and s.parent_id.startswith(pod.id)
                        ]
                        # 只取顶层
                        top_rack_switches = self._get_top_layer_switches(switches, 'inter_board')
                        device_ids = [s.id for s in top_rack_switches if s.parent_id and s.parent_id.startswith(pod.id)]
                    else:
                        # 直接连接到Rack
                        device_ids = [r.id for r in pod.racks]

                    if device_ids:
                        pod_switches, pod_switch_conns = self._generate_switch_connections(
                            switch_layers=[
                                {'layer_name': l['layer_name'], 'switch_type_id': l['switch_type_id'],
                                 'count': l['count'], 'inter_connect': l.get('inter_connect', False)}
                                for l in inter_rack_config['layers']
                            ],
                            switch_types=switch_types_list,
                            devices=device_ids,
                            redundancy=inter_rack_config.get('downlink_redundancy', 1),
                            parent_id=pod.id,
                            hierarchy_level='inter_rack',
                            connection_mode=inter_rack_config.get('connection_mode', 'round_robin'),
                            group_config=inter_rack_config.get('group_config'),
                            custom_connections=inter_rack_config.get('custom_connections')
                        )
                        switches.extend(pod_switches)
                        connections.extend(pod_switch_conns)

            # 3. 数据中心层Switch（Pod间）
            dc_level_config = switch_config.get('inter_pod', {})
            if dc_level_config.get('enabled') and dc_level_config.get('layers'):
                # 确定数据中心层的下层设备：Pod还是Pod层顶层Switch
                if inter_rack_config.get('enabled') and inter_rack_config.get('connect_to_upper_level', True):
                    # 连接到Pod层顶层Switch
                    top_pod_switches = self._get_top_layer_switches(switches, 'inter_rack')
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
                        hierarchy_level='inter_pod',
                        connection_mode=dc_level_config.get('connection_mode', 'round_robin'),
                        group_config=dc_level_config.get('group_config'),
                        custom_connections=dc_level_config.get('custom_connections')
                    )
                    switches.extend(dc_switches)
                    connections.extend(dc_switch_conns)

        # 处理手动连接
        manual_config = None
        if manual_connections and manual_connections.get('enabled'):
            manual_config = manual_connections
            mode = manual_connections.get('mode', 'append')
            manual_conn_list = manual_connections.get('connections', [])

            if mode == 'replace':
                # 替换模式：按层级移除自动生成的连接
                levels_with_manual = set(c.get('hierarchy_level') for c in manual_conn_list)
                connections = [c for c in connections if not self._is_connection_in_level(c, levels_with_manual, pods)]

            # 添加手动连接
            for mc in manual_conn_list:
                connections.append(ConnectionConfig(
                    source=mc['source'],
                    target=mc['target'],
                    type='manual',
                    bandwidth=mc.get('bandwidth'),
                    latency=mc.get('latency'),
                    is_manual=True
                ))

        topology = HierarchicalTopology(
            pods=pods,
            connections=connections,
            switches=switches,
            switch_config=switch_config,
            manual_connections=manual_config
        )
        self._cached_topology = topology
        return topology

    def _is_connection_in_level(
        self,
        connection: ConnectionConfig,
        levels: set,
        pods: List[PodConfig]
    ) -> bool:
        """判断连接是否属于指定层级"""
        source = connection.source
        target = connection.target

        # 根据节点ID格式判断层级
        # datacenter层：pod_X 之间的连接
        # pod层：pod_X/rack_Y 之间的连接
        # rack层：pod_X/rack_Y/board_Z 之间的连接
        # board层：chip 之间的连接

        source_parts = source.split('/')
        target_parts = target.split('/')

        # 判断层级
        if 'datacenter' in levels:
            # Pod间连接
            if len(source_parts) == 1 and source.startswith('pod_') and \
               len(target_parts) == 1 and target.startswith('pod_'):
                return True

        if 'pod' in levels:
            # Rack间连接（同一Pod内）
            if len(source_parts) == 2 and 'rack_' in source and \
               len(target_parts) == 2 and 'rack_' in target:
                return True

        if 'rack' in levels:
            # Board间连接（同一Rack内）
            if len(source_parts) == 3 and 'board_' in source and \
               len(target_parts) == 3 and 'board_' in target:
                return True

        if 'board' in levels:
            # Chip间连接
            if 'chip_' in source and 'chip_' in target:
                return True

        return False

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

    def _generate_board_chips_flex(
        self,
        board_id: str,
        flex_chips: List[dict]
    ) -> List[ChipConfig]:
        """使用灵活配置生成板卡上的芯片"""
        chips = []

        # 计算总芯片数
        total_chips = sum(fc.get('count', 1) for fc in flex_chips)
        if total_chips == 0:
            return chips

        # 智能计算网格大小：尽量接近正方形
        cols = int(math.ceil(math.sqrt(total_chips)))

        # 收集所有芯片
        chip_list = []
        for fc in flex_chips:
            chip_name = fc.get('name', 'CHIP')
            chip_count = fc.get('count', 1)
            # 将名称转换为小写作为类型（兼容现有逻辑）
            chip_type = chip_name.lower()
            # 只支持 npu 和 cpu 类型，其他类型默认为 npu
            if chip_type not in ['npu', 'cpu']:
                chip_type = 'npu'

            for i in range(chip_count):
                chip_list.append({
                    'type': chip_type,
                    'label': f"{chip_name}-{i}",
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
        chips: List[ChipConfig],
        topology_type: str = 'none'
    ) -> List[ConnectionConfig]:
        """生成芯片间的连接"""
        connections = []

        # 找出不同类型的芯片
        npus = [c for c in chips if c.type == 'npu']
        cpus = [c for c in chips if c.type == 'cpu']

        # NPU <-> CPU 连接（始终存在）
        for npu in npus:
            for cpu in cpus:
                connections.append(ConnectionConfig(
                    source=npu.id,
                    target=cpu.id,
                    type='intra',
                    bandwidth=64.0,  # PCIe连接
                    latency=50.0,  # PCIe延迟 (ns)
                ))

        # 根据拓扑类型生成NPU间连接
        if topology_type != 'none' and len(npus) > 1:
            npu_ids = [c.id for c in npus]
            npu_connections = self._generate_direct_connections(npu_ids, topology_type, 'intra', 400.0, latency=50.0)
            connections.extend(npu_connections)

        return connections

    def _generate_board_connections(
        self,
        boards: List[BoardConfig],
        topology_type: str = 'full_mesh'
    ) -> List[ConnectionConfig]:
        """生成Board间的连接"""
        board_ids = [b.id for b in boards]
        return self._generate_direct_connections(board_ids, topology_type, 'intra', 100.0, latency=100.0)

    def _generate_rack_connections(
        self,
        rack_ids: List[str],
        topology_type: str = 'full_mesh'
    ) -> List[ConnectionConfig]:
        """生成Rack间的连接"""
        return self._generate_direct_connections(rack_ids, topology_type, 'intra', 400.0, latency=200.0)

    def _generate_pod_connections(
        self,
        pod_ids: List[str],
        topology_type: str = 'full_mesh'
    ) -> List[ConnectionConfig]:
        """生成Pod间的连接"""
        return self._generate_direct_connections(pod_ids, topology_type, 'inter', 1600.0, latency=500.0)

    def _generate_direct_connections(
        self,
        node_ids: List[str],
        topology_type: str,
        conn_type: str,
        bandwidth: float,
        latency: float = 100.0
    ) -> List[ConnectionConfig]:
        """根据拓扑类型生成直连"""
        connections = []
        n = len(node_ids)

        if n < 2 or topology_type == 'none':
            return connections

        if topology_type == 'full_mesh':
            # 全连接：每个节点连接所有其他节点
            for i in range(n):
                for j in range(i + 1, n):
                    connections.append(ConnectionConfig(
                        source=node_ids[i],
                        target=node_ids[j],
                        type=conn_type,
                        bandwidth=bandwidth,
                        latency=latency,
                    ))

        elif topology_type == 'ring':
            # 环形：每个节点连接相邻节点
            for i in range(n):
                j = (i + 1) % n
                connections.append(ConnectionConfig(
                    source=node_ids[i],
                    target=node_ids[j],
                    type=conn_type,
                    bandwidth=bandwidth,
                    latency=latency,
                ))

        elif topology_type == 'torus_2d':
            # 2D Torus：二维环面，计算行列
            cols = int(math.ceil(math.sqrt(n)))
            rows = (n + cols - 1) // cols
            for i in range(n):
                row, col = i // cols, i % cols
                # 右邻居（环绕）
                right = row * cols + (col + 1) % cols
                if right < n and right != i:
                    connections.append(ConnectionConfig(
                        source=node_ids[i],
                        target=node_ids[right],
                        type=conn_type,
                        bandwidth=bandwidth,
                        latency=latency,
                    ))
                # 下邻居（环绕）
                down = ((row + 1) % rows) * cols + col
                if down < n and down != i:
                    connections.append(ConnectionConfig(
                        source=node_ids[i],
                        target=node_ids[down],
                        type=conn_type,
                        bandwidth=bandwidth,
                        latency=latency,
                    ))

        elif topology_type == 'torus_3d':
            # 3D Torus：三维环面
            dim = max(1, int(round(n ** (1/3))))
            for i in range(n):
                x = i % dim
                y = (i // dim) % dim
                z = i // (dim * dim)
                # X方向邻居（只在i < nx时生成，避免重复）
                nx = y * dim + ((x + 1) % dim) + z * dim * dim
                if nx < n and nx != i and i < nx:
                    connections.append(ConnectionConfig(
                        source=node_ids[i],
                        target=node_ids[nx],
                        type=conn_type,
                        bandwidth=bandwidth,
                        latency=latency,
                    ))
                # Y方向邻居（只在i < ny时生成，避免重复）
                ny = ((y + 1) % dim) * dim + x + z * dim * dim
                if ny < n and ny != i and i < ny:
                    connections.append(ConnectionConfig(
                        source=node_ids[i],
                        target=node_ids[ny],
                        type=conn_type,
                        bandwidth=bandwidth,
                        latency=latency,
                    ))
                # Z方向邻居（只在i < nz时生成，避免重复）
                nz = y * dim + x + ((z + 1) % dim) * dim * dim
                if nz < n and nz != i and i < nz:
                    connections.append(ConnectionConfig(
                        source=node_ids[i],
                        target=node_ids[nz],
                        type=conn_type,
                        bandwidth=bandwidth,
                        latency=latency,
                    ))

        elif topology_type == 'full_mesh_2d':
            # 2D FullMesh：行列全连接（同行全连接 + 同列全连接）
            cols = int(math.ceil(math.sqrt(n)))
            rows = (n + cols - 1) // cols
            # 同行全连接
            for row in range(rows):
                row_nodes = [i for i in range(row * cols, min((row + 1) * cols, n))]
                for i in range(len(row_nodes)):
                    for j in range(i + 1, len(row_nodes)):
                        connections.append(ConnectionConfig(
                            source=node_ids[row_nodes[i]],
                            target=node_ids[row_nodes[j]],
                            type=conn_type,
                            bandwidth=bandwidth,
                            latency=latency,
                        ))
            # 同列全连接
            for col in range(cols):
                col_nodes = [row * cols + col for row in range(rows) if row * cols + col < n]
                for i in range(len(col_nodes)):
                    for j in range(i + 1, len(col_nodes)):
                        connections.append(ConnectionConfig(
                            source=node_ids[col_nodes[i]],
                            target=node_ids[col_nodes[j]],
                            type=conn_type,
                            bandwidth=bandwidth,
                            latency=latency,
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
        hierarchy_level: str,
        connection_mode: str = 'round_robin',
        group_config: dict = None,
        custom_connections: List[dict] = None
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
            connection_mode: 连接模式 ('round_robin', 'full_mesh', 'group', 'custom')
            group_config: 分组配置（connection_mode为'group'时使用）
            custom_connections: 自定义连接列表（connection_mode为'custom'时使用）

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
                    inter_ports_used=0,
                    u_height=1 if hierarchy_level == 'inter_board' else None
                )
                switches.append(switch)
                layer_switches[layer_name].append(switch)

        # 2. 设备连接到最底层Switch（根据连接模式）
        bottom_layer = switch_layers[0]['layer_name']
        bottom_switches = layer_switches[bottom_layer]

        if connection_mode == 'custom':
            # 自定义连接：每个设备连接到指定数量的Switch（轮询分配）
            conn_count = min(redundancy, len(bottom_switches))
            if custom_connections:
                # 使用用户指定的连接关系
                for custom_conn in custom_connections:
                    device_id = custom_conn.get('device_id')
                    switch_indices = custom_conn.get('switch_indices', [])

                    if device_id in devices:
                        for switch_idx in switch_indices:
                            if switch_idx < len(bottom_switches):
                                switch = bottom_switches[switch_idx]
                                connections.append(ConnectionConfig(
                                    source=device_id,
                                    target=switch.id,
                                    type='switch',
                                    connection_role='downlink',
                                    latency=100.0,  # Switch连接延迟 (ns)
                                ))
                                switch.downlink_ports_used += 1
            else:
                # 没有自定义配置时，按每节点连接数轮询分配
                for dev_idx, device_id in enumerate(devices):
                    for r in range(conn_count):
                        switch_idx = (dev_idx + r) % len(bottom_switches)
                        switch = bottom_switches[switch_idx]
                        connections.append(ConnectionConfig(
                            source=device_id,
                            target=switch.id,
                            type='switch',
                            connection_role='downlink',
                            latency=100.0,  # Switch连接延迟 (ns)
                        ))
                        switch.downlink_ports_used += 1

        else:  # full_mesh (默认)
            # 全连接：每个设备连接到所有底层Switch
            for device_id in devices:
                for switch in bottom_switches:
                    connections.append(ConnectionConfig(
                        source=device_id,
                        target=switch.id,
                        type='switch',
                        connection_role='downlink',
                        latency=100.0,  # Switch连接延迟 (ns)
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
                        connection_role='uplink',
                        latency=100.0,  # Switch层间连接延迟 (ns)
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
                            connection_role='inter',
                            latency=100.0,  # Switch同层互联延迟 (ns)
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
