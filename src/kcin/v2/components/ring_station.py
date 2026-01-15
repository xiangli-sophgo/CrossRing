"""
RingStation 组件 - CrossRing 2.0 核心组件

统一 EQ（Ejection Queue）、IQ（Injection Queue）、RB（Ring Bridge）功能，
每个节点一个实例，实现统一仲裁和单周期维度转换。

输入端口（每个IP类型独立 + 4个环方向）：
- ddr_0, ddr_1, gdma_0, gdma_1, ...: 本地各 IP 类型注入
- TL, TR: 来自横向环（下环的 flit）
- TU, TD: 来自纵向环（下环的 flit）

输出端口（每个IP类型独立 + 4个环方向）：
- ddr_0, ddr_1, gdma_0, gdma_1, ...: 本地各 IP 类型弹出
- TL, TR: 到横向环（准备上环）
- TU, TD: 到纵向环（准备上环）

特性：
- 每个 IP 类型独立的 channel buffer（不共享）
- 支持一个周期内完成维度转换（TL/TR ↔ TU/TD）
- 统一的轮询仲裁
- 保留 E-Tag 和 I-Tag 机制（通过 CrossPoint 作为内部组件）
"""

from collections import deque
from typing import Optional, Dict, List, Tuple, Any
from src.kcin.base.config import KCINConfigBase
from src.utils.flit import Flit
from src.utils.statistical_fifo import StatisticalFIFO


class RingStation:
    """
    RingStation 2.0 - 统一的节点数据流交换组件
    """

    # 环方向常量
    RING_DIRECTIONS = ["TL", "TR", "TU", "TD"]
    # 注意: 实际端口列表是动态的 (ch_names + RING_DIRECTIONS)，需要通过实例获取

    def __init__(self, node_id: int, config: KCINConfigBase, network_ref=None):
        """
        初始化 RingStation

        Args:
            node_id: 节点 ID
            config: 配置对象
            network_ref: Network 对象引用
        """
        self.node_id = node_id
        self.config = config
        self.network = network_ref

        # 计算行列位置（用于FIFO命名）
        self.row = node_id // config.NUM_COL
        self.col = node_id % config.NUM_COL

        # 获取 IP 类型列表
        self.ch_names = config.CH_NAME_LIST if hasattr(config, 'CH_NAME_LIST') else []

        # 获取网络类型后缀（用于区分三网络的FIFO名称）
        # "Request Network" -> "req", "Response Network" -> "rsp", "Data Network" -> "dat"
        self._network_suffix = ""
        if network_ref and hasattr(network_ref, 'name'):
            self._network_suffix = "_" + network_ref.name.split()[0].lower()[:3]

        # ==================== 输入端 FIFO ====================
        # 每个 IP 类型有独立的 channel buffer
        self.input_fifos = {}
        for ch in self.ch_names:
            name = f"RS({self.row},{self.col})_IN_{ch}{self._network_suffix}"
            self.input_fifos[ch] = StatisticalFIFO(
                name=name, maxlen=config.RS_IN_CH_BUFFER,
                node_pos=node_id, category="RS_IN", fifo_type="CH", ip_type=ch
            )
        for direction in self.RING_DIRECTIONS:
            name = f"RS({self.row},{self.col})_IN_{direction}{self._network_suffix}"
            self.input_fifos[direction] = StatisticalFIFO(
                name=name, maxlen=config.RS_IN_FIFO_DEPTH,
                node_pos=node_id, category="RS_IN", fifo_type=direction
            )
        self.input_fifos_pre = {k: None for k in self.input_fifos.keys()}

        # ==================== 输出端 FIFO ====================
        # 每个 IP 类型有独立的 channel buffer
        self.output_fifos = {}
        for ch in self.ch_names:
            name = f"RS({self.row},{self.col})_OUT_{ch}{self._network_suffix}"
            self.output_fifos[ch] = StatisticalFIFO(
                name=name, maxlen=config.RS_OUT_CH_BUFFER,
                node_pos=node_id, category="RS_OUT", fifo_type="CH", ip_type=ch
            )
        for direction in self.RING_DIRECTIONS:
            name = f"RS({self.row},{self.col})_OUT_{direction}{self._network_suffix}"
            self.output_fifos[direction] = StatisticalFIFO(
                name=name, maxlen=config.RS_OUT_FIFO_DEPTH,
                node_pos=node_id, category="RS_OUT", fifo_type=direction
            )
        self.output_fifos_pre = {k: None for k in self.output_fifos.keys()}

        # ==================== 仲裁状态 ====================
        # 每个输出端口的轮询指针
        self.arb_pointers = {port: 0 for port in self.output_fifos.keys()}

        # 定义每个输出端口的候选输入端口
        self.output_candidates = {}
        # 本地弹出（每个IP类型）：来自四个环方向 + 其他 IP 类型（同节点传输）
        for ch in self.ch_names:
            # 环方向 + 其他 IP 类型
            other_ips = [other for other in self.ch_names if other != ch]
            self.output_candidates[ch] = ["TL", "TR", "TU", "TD"] + other_ips
        # 上环方向：来自所有IP类型 + 其他环方向
        other_dirs = {"TL": ["TR", "TU", "TD"], "TR": ["TL", "TU", "TD"],
                      "TU": ["TL", "TR", "TD"], "TD": ["TL", "TR", "TU"]}
        for direction in self.RING_DIRECTIONS:
            self.output_candidates[direction] = list(self.ch_names) + other_dirs[direction]

        # ==================== 统计 ====================
        self.stats = {
            "cross_dimension_transfers": 0,
            "local_ejects": 0,
            "local_injects": 0,
            "arbitration_conflicts": 0,
        }

    def register_ip_type(self, ip_type: str):
        """动态注册 IP 类型，创建对应的 channel buffer

        Args:
            ip_type: IP 类型名称（如 "ddr_0", "gdma_0"）
        """
        if ip_type in self.ch_names:
            return  # 已注册

        self.ch_names.append(ip_type)

        # 添加输入端 FIFO
        name_in = f"RS({self.row},{self.col})_IN_{ip_type}{self._network_suffix}"
        self.input_fifos[ip_type] = StatisticalFIFO(
            name=name_in, maxlen=self.config.RS_IN_CH_BUFFER,
            node_pos=self.node_id, category="RS_IN", fifo_type="CH", ip_type=ip_type
        )
        self.input_fifos_pre[ip_type] = None

        # 添加输出端 FIFO
        name_out = f"RS({self.row},{self.col})_OUT_{ip_type}{self._network_suffix}"
        self.output_fifos[ip_type] = StatisticalFIFO(
            name=name_out, maxlen=self.config.RS_OUT_CH_BUFFER,
            node_pos=self.node_id, category="RS_OUT", fifo_type="CH", ip_type=ip_type
        )
        self.output_fifos_pre[ip_type] = None

        # 添加仲裁指针
        self.arb_pointers[ip_type] = 0

        # 更新仲裁候选
        # 本地弹出：来自四个环方向 + 其他 IP 类型（同节点传输）
        self.output_candidates[ip_type] = ["TL", "TR", "TU", "TD"]
        # 添加已注册的其他 IP 类型作为候选输入（支持同节点传输）
        for other_ip in self.ch_names:
            if other_ip != ip_type:
                self.output_candidates[ip_type].append(other_ip)
                # 同时将新 IP 类型添加到其他 IP 的候选列表中
                if ip_type not in self.output_candidates[other_ip]:
                    self.output_candidates[other_ip].append(ip_type)
        # 更新环方向的候选（添加新的 IP 类型）
        for direction in self.RING_DIRECTIONS:
            if ip_type not in self.output_candidates[direction]:
                self.output_candidates[direction].append(ip_type)

    # ==================== 核心处理方法 ====================

    def move_pre_to_fifos(self, cycle: int = 0):
        """将 pre 缓冲中的 flit 移动到 FIFO（兼容接口）"""
        self.move_input_pre_to_fifos(cycle)
        self.move_output_pre_to_fifos(cycle)

    def move_input_pre_to_fifos(self, cycle: int = 0):
        """将输入端 pre 缓冲中的 flit 移动到 input_fifos"""
        for port, flit in self.input_fifos_pre.items():
            if flit is not None:
                if len(self.input_fifos[port]) < self.input_fifos[port].maxlen:
                    # 命名格式: RS_IN_<port>，例如 RS_IN_ddr_0, RS_IN_TL
                    flit.set_position(f"RS_IN_{port}", cycle)
                    self.input_fifos[port].append(flit)
                    self.input_fifos_pre[port] = None  # 只有成功添加后才清空

    def move_output_pre_to_fifos(self, cycle: int = 0):
        """将输出端 pre 缓冲中的 flit 移动到 output_fifos"""
        for port, flit in self.output_fifos_pre.items():
            if flit is not None:
                if len(self.output_fifos[port]) < self.output_fifos[port].maxlen:
                    # 命名格式: RS_OUT_<port>，例如 RS_OUT_ddr_0, RS_OUT_TL
                    flit.set_position(f"RS_OUT_{port}", cycle)
                    self.output_fifos[port].append(flit)
                    self.output_fifos_pre[port] = None  # 只有成功添加后才清空

    def process_cycle(self, cycle: int):
        """RingStation核心仲裁和路由逻辑

        每周期执行3个步骤完成数据转发：

        Step 1: 收集路由请求 (_collect_routing_requests)
        ----------------------------------------------
        遍历所有输入端口（IP类型 + TL/TR/TU/TD），对每个非空的input_fifo：
        - 查看队首flit（peek，不移除）
        - 调用_determine_output_port()确定目标输出端口
        - 记录到routing_requests字典: {output_port: [(input_port, flit), ...]}

        路由决策逻辑:
        a) 到达目的地节点 → 输出到destination_type对应的IP channel
        b) 根据flit.path → 确定下一跳方向（TL/TR/TU/TD）
        c) 无path时 → 使用XY路由算法计算方向

        Step 2: 仲裁 (_arbitrate)
        --------------------------
        对每个输出端口，从其候选输入中选择一个winner：
        - 检查输出端口可用性（output_fifos_pre为空 且 output_fifos未满）
        - 过滤已被其他输出端口选中的输入（避免一个输入同时发到多个输出）
        - 轮询仲裁（Round-Robin）：
          * 使用arb_pointers[output_port]记录上次选中位置
          * 从上次位置+1开始，按output_candidates[output_port]顺序查找
          * 找到第一个匹配的输入端口作为winner
        - 更新arb_pointers，确保下次仲裁时公平

        仲裁冲突场景:
        - 多个输入竞争同一输出 → Round-Robin选择一个，其余下周期重试
        - 输出端口不可用 → 所有候选延迟到下周期

        Step 3: 执行转移 (_execute_transfers)
        -------------------------------------
        对每个仲裁成功的(output_port, winner)：
        - 从input_fifos[input_port]移除队首flit（popleft）
        - 将flit放入output_fifos_pre[output_port]（准备下周期移入output_fifos）
        - 更新统计：跨维度转换、本地注入、本地弹出

        关键设计：
        ----------
        1. pre缓冲机制：
           仲裁结果先放入output_fifos_pre，下周期再移入output_fifos
           → 避免同一周期内output_fifos的读写冲突
           → 实现管道化，每周期可处理新的仲裁

        2. 轮询公平性：
           arb_pointers确保每个输入端口都有机会被选中
           → 避免饥饿，保证长期公平

        3. 单周期内部时序：
           peek → 仲裁决策 → popleft → 写pre缓冲
           → 所有操作在一个周期内完成，符合硬件时序

        时序依赖：
        - 调用前提：input_fifos已包含本周期可用数据（由_move_pre_to_queues Phase1保证）
        - 调用后续：output_fifos_pre将在_move_pre_to_queues Phase3中移入output_fifos
        """
        # Step 1: 收集所有输入端的 flit 和它们需要的输出端口
        routing_requests = self._collect_routing_requests(cycle)

        # Step 2: 仲裁 - 每个输出端口选择一个输入
        arbitration_results = self._arbitrate(routing_requests, cycle)

        # Step 3: 执行转移
        self._execute_transfers(arbitration_results, cycle)

    def _collect_routing_requests(self, cycle: int) -> Dict[str, List[Tuple[str, Flit]]]:
        """
        收集各输入端口的路由请求

        Returns:
            dict: {output_port: [(input_port, flit), ...]}
        """
        requests = {port: [] for port in self.output_fifos.keys()}

        for input_port, fifo in self.input_fifos.items():
            if not fifo:
                continue

            flit = fifo[0]  # peek
            output_port = self._determine_output_port(flit, input_port)

            if output_port:
                requests[output_port].append((input_port, flit))

        return requests

    def _determine_output_port(self, flit: Flit, input_port: str) -> Optional[str]:
        """
        决定 flit 的输出端口

        路由决策：
        1. 到达目的地节点 → 对应IP类型的 channel（本地弹出）
        2. 需要横向移动 → TL 或 TR
        3. 需要纵向移动 → TU 或 TD
        """
        # 获取最终目的地
        final_dest = flit.path[-1] if flit.path else flit.destination

        # Case 1: 已到达目的地 → 输出到对应IP类型的 channel
        if self.node_id == final_dest:
            dest_type = getattr(flit, 'destination_type', None)
            if dest_type and dest_type in self.output_fifos:
                return dest_type
            # 兼容：如果没有 destination_type，尝试第一个匹配的 channel
            return self.ch_names[0] if self.ch_names else None

        # Case 2: 根据 path 决定下一跳
        if flit.path:
            try:
                current_idx = flit.path.index(self.node_id)
                if current_idx + 1 < len(flit.path):
                    next_hop = flit.path[current_idx + 1]
                    return self._get_direction_to_neighbor(next_hop)
            except ValueError:
                pass

        # Case 3: 不在 path 中，根据位置计算（XY 路由）
        return self._calculate_direction_by_position(final_dest)

    def _get_direction_to_neighbor(self, next_hop: int) -> str:
        """根据下一跳节点计算方向"""
        diff = next_hop - self.node_id

        if diff == 1:
            return "TR"  # 右
        elif diff == -1:
            return "TL"  # 左
        elif diff == self.config.NUM_COL:
            return "TD"  # 下
        elif diff == -self.config.NUM_COL:
            return "TU"  # 上
        else:
            # 非相邻节点，需要先确定维度
            next_row = next_hop // self.config.NUM_COL
            next_col = next_hop % self.config.NUM_COL
            curr_row = self.node_id // self.config.NUM_COL
            curr_col = self.node_id % self.config.NUM_COL

            if next_col != curr_col:
                return "TR" if next_col > curr_col else "TL"
            else:
                return "TD" if next_row > curr_row else "TU"

    def _calculate_direction_by_position(self, dest: int) -> str:
        """根据目标位置计算方向（用于 XY 路由）"""
        dest_row = dest // self.config.NUM_COL
        dest_col = dest % self.config.NUM_COL
        curr_row = self.node_id // self.config.NUM_COL
        curr_col = self.node_id % self.config.NUM_COL

        # XY 路由：先 X（横向）后 Y（纵向）
        if dest_col != curr_col:
            return "TR" if dest_col > curr_col else "TL"
        elif dest_row != curr_row:
            return "TD" if dest_row > curr_row else "TU"
        else:
            return None  # 已到达，由调用方处理

    def _arbitrate(self, requests: Dict[str, List[Tuple[str, Flit]]], cycle: int) -> Dict[str, Optional[Tuple[str, Flit]]]:
        """
        轮询仲裁：每个输出端口从候选输入中选择一个

        Args:
            requests: {output_port: [(input_port, flit), ...]}

        Returns:
            dict: {output_port: (input_port, flit) or None}
        """
        results = {}
        used_inputs = set()  # 记录已被选中的输入端口

        for output_port, candidates in requests.items():
            if not candidates:
                results[output_port] = None
                continue

            # 检查输出端口是否可用
            if not self._output_port_available(output_port):
                results[output_port] = None
                continue

            # 过滤掉已被其他输出端口选中的输入
            available_candidates = [(inp, flit) for inp, flit in candidates if inp not in used_inputs]

            if not available_candidates:
                results[output_port] = None
                continue

            # 轮询仲裁
            winner = self._round_robin_select(output_port, available_candidates)
            if winner:
                results[output_port] = winner
                used_inputs.add(winner[0])

                if len(available_candidates) > 1:
                    self.stats["arbitration_conflicts"] += 1
            else:
                results[output_port] = None

        return results

    def _round_robin_select(self, output_port: str, candidates: List[Tuple[str, Flit]]) -> Optional[Tuple[str, Flit]]:
        """
        轮询选择

        Args:
            output_port: 输出端口名
            candidates: [(input_port, flit), ...]

        Returns:
            tuple: (input_port, flit) or None
        """
        candidate_order = self.output_candidates[output_port]
        pointer = self.arb_pointers[output_port]

        # 按轮询顺序查找第一个有效候选
        for i in range(len(candidate_order)):
            idx = (pointer + i) % len(candidate_order)
            candidate_port = candidate_order[idx]

            for input_port, flit in candidates:
                if input_port == candidate_port:
                    # 更新指针到下一个位置
                    self.arb_pointers[output_port] = (idx + 1) % len(candidate_order)
                    return (input_port, flit)

        return None

    def _output_port_available(self, output_port: str) -> bool:
        """
        检查输出端口是否可用

        对于上环端口（TL/TR/TU/TD）：检查 pre 缓冲和 FIFO 容量
        对于本地弹出（ch_buffer）：检查 FIFO 容量
        """
        # 检查 pre 缓冲是否被占用
        if self.output_fifos_pre[output_port] is not None:
            return False

        # 检查 FIFO 是否有空间
        if len(self.output_fifos[output_port]) >= self.output_fifos[output_port].maxlen:
            return False

        return True

    def _execute_transfers(self, results: Dict[str, Optional[Tuple[str, Flit]]], cycle: int):
        """执行仲裁结果的数据转移"""
        for output_port, selection in results.items():
            if selection is None:
                continue

            input_port, flit = selection

            # 从输入端口移除
            self.input_fifos[input_port].popleft()

            # Entry释放已在network._try_eject中处理（下环时直接添加到RS_pending_entry_release）

            # 更新 flit position（input → output_pre，等待下周期移入 output FIFO）
            flit.set_position(f"RS_OUT_{output_port}", cycle)

            # 添加到输出端口 pre 缓冲
            self.output_fifos_pre[output_port] = flit

            # 更新统计
            if self._is_cross_dimension(input_port, output_port):
                self.stats["cross_dimension_transfers"] += 1
            # 本地弹出：输出到 IP 类型（非环方向）
            if output_port in self.ch_names:
                self.stats["local_ejects"] += 1
            # 本地注入：来自 IP 类型（非环方向）
            if input_port in self.ch_names:
                self.stats["local_injects"] += 1

    def _is_cross_dimension(self, input_port: str, output_port: str) -> bool:
        """判断是否为跨维度转换"""
        horizontal = {"TL", "TR"}
        vertical = {"TU", "TD"}

        if input_port in horizontal and output_port in vertical:
            return True
        if input_port in vertical and output_port in horizontal:
            return True
        return False

    # ==================== 外部接口方法 ====================

    def enqueue_from_ring(self, flit: Flit, direction: str) -> bool:
        """
        从环上下环的 flit 进入 RS 输入端

        Args:
            flit: 下环的 flit
            direction: 来源方向 (TL/TR/TU/TD)

        Returns:
            bool: 是否成功入队
        """
        if direction not in self.RING_DIRECTIONS:
            return False

        if self.input_fifos_pre[direction] is not None:
            return False

        self.input_fifos_pre[direction] = flit
        return True

    def enqueue_from_local(self, flit: Flit, ip_type: str = None) -> bool:
        """
        从本地 IP 注入的 flit 进入 RS 输入端

        Args:
            flit: 本地注入的 flit
            ip_type: IP 类型（如 "ddr_0", "gdma_0" 等）

        Returns:
            bool: 是否成功入队
        """
        # 确定目标 channel
        if ip_type is None:
            ip_type = getattr(flit, 'source_type', None)
        if ip_type is None or ip_type not in self.input_fifos_pre:
            return False

        if self.input_fifos_pre[ip_type] is not None:
            return False

        self.input_fifos_pre[ip_type] = flit
        return True

    def dequeue_to_ring(self, direction: str) -> Optional[Flit]:
        """
        从 RS 输出端取出准备上环的 flit

        Args:
            direction: 目标方向 (TL/TR/TU/TD)

        Returns:
            Flit or None
        """
        if direction not in self.RING_DIRECTIONS:
            return None

        if self.output_fifos[direction]:
            return self.output_fifos[direction].popleft()
        return None

    def dequeue_to_local(self, ip_type: str = None) -> Optional[Flit]:
        """
        从 RS 输出端取出准备弹出到本地 IP 的 flit

        Args:
            ip_type: IP 类型（如 "ddr_0", "gdma_0" 等）

        Returns:
            Flit or None
        """
        if ip_type is None:
            # 遍历所有 IP 类型，返回第一个有数据的
            for ch in self.ch_names:
                if self.output_fifos[ch]:
                    return self.output_fifos[ch].popleft()
            return None

        if ip_type in self.output_fifos and self.output_fifos[ip_type]:
            return self.output_fifos[ip_type].popleft()
        return None

    def peek_output(self, port: str) -> Optional[Flit]:
        """查看输出端口队首的 flit（不移除）"""
        if port in self.output_fifos and self.output_fifos[port]:
            return self.output_fifos[port][0]
        return None

    def has_output(self, port: str) -> bool:
        return port in self.output_fifos and len(self.output_fifos[port]) > 0

    def can_accept_input(self, port: str) -> bool:
        if port not in self.input_fifos:
            return False
        return self.input_fifos_pre[port] is None

    # ==================== 调试和统计方法 ====================

    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "node_id": self.node_id,
            "input_fifos": {k: len(v) for k, v in self.input_fifos.items()},
            "output_fifos": {k: len(v) for k, v in self.output_fifos.items()},
            "input_pre": {k: v is not None for k, v in self.input_fifos_pre.items()},
            "output_pre": {k: v is not None for k, v in self.output_fifos_pre.items()},
            "stats": self.stats.copy(),
        }

    def reset_stats(self):
        """重置统计数据"""
        for key in self.stats:
            self.stats[key] = 0
