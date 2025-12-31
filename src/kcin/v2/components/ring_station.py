"""
RingStation 组件 - CrossRing 2.0 核心组件

统一 EQ（Ejection Queue）、IQ（Injection Queue）、RB（Ring Bridge）功能，
每个节点一个实例，实现统一仲裁和单周期维度转换。

输入端口（5个）：
- ch_buffer: 本地 IP 注入
- TL, TR: 来自横向环（下环的 flit）
- TU, TD: 来自纵向环（下环的 flit）

输出端口（5个）：
- ch_buffer: 本地 IP 弹出
- TL, TR: 到横向环（准备上环）
- TU, TD: 到纵向环（准备上环）

特性：
- 支持一个周期内完成维度转换（TL/TR ↔ TU/TD）
- 统一的轮询仲裁
- 保留 E-Tag 和 I-Tag 机制（通过 CrossPoint 作为内部组件）
"""

from collections import deque
from typing import Optional, Dict, List, Tuple, Any
from src.kcin.base.config import KCINConfigBase
from src.utils.flit import Flit


class RingStation:
    """
    RingStation 2.0 - 统一的节点数据流交换组件
    """

    # 所有方向常量
    RING_DIRECTIONS = ["TL", "TR", "TU", "TD"]
    ALL_PORTS = ["ch_buffer", "TL", "TR", "TU", "TD"]

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

        # ========== 输入端 FIFO ==========
        self.input_fifos = {
            "ch_buffer": deque(maxlen=config.RS_IN_CH_BUFFER),
            "TL": deque(maxlen=config.RS_IN_FIFO_DEPTH),
            "TR": deque(maxlen=config.RS_IN_FIFO_DEPTH),
            "TU": deque(maxlen=config.RS_IN_FIFO_DEPTH),
            "TD": deque(maxlen=config.RS_IN_FIFO_DEPTH),
        }
        self.input_fifos_pre = {k: None for k in self.input_fifos.keys()}

        # ========== 输出端 FIFO ==========
        self.output_fifos = {
            "ch_buffer": deque(maxlen=config.RS_OUT_CH_BUFFER),
            "TL": deque(maxlen=config.RS_OUT_FIFO_DEPTH),
            "TR": deque(maxlen=config.RS_OUT_FIFO_DEPTH),
            "TU": deque(maxlen=config.RS_OUT_FIFO_DEPTH),
            "TD": deque(maxlen=config.RS_OUT_FIFO_DEPTH),
        }
        self.output_fifos_pre = {k: None for k in self.output_fifos.keys()}

        # ========== 仲裁状态 ==========
        # 每个输出端口的轮询指针
        self.arb_pointers = {port: 0 for port in self.output_fifos.keys()}

        # 定义每个输出端口的候选输入端口
        self.output_candidates = {
            "ch_buffer": ["TL", "TR", "TU", "TD"],           # 本地弹出：来自四个环方向
            "TL": ["ch_buffer", "TR", "TU", "TD"],           # 向左上环
            "TR": ["ch_buffer", "TL", "TU", "TD"],           # 向右上环
            "TU": ["ch_buffer", "TL", "TR", "TD"],           # 向上上环
            "TD": ["ch_buffer", "TL", "TR", "TU"],           # 向下上环
        }

        # ========== 统计 ==========
        self.stats = {
            "cross_dimension_transfers": 0,
            "local_ejects": 0,
            "local_injects": 0,
            "arbitration_conflicts": 0,
        }

    # ------------------------------------------------------------------
    # 核心处理方法
    # ------------------------------------------------------------------

    def move_pre_to_fifos(self, cycle: int = 0):
        """将 pre 缓冲中的 flit 移动到 FIFO"""
        # 输入端 pre → FIFO
        for port, flit in self.input_fifos_pre.items():
            if flit is not None:
                if len(self.input_fifos[port]) < self.input_fifos[port].maxlen:
                    # 命名格式: RS_IN_CH, RS_IN_TL, RS_IN_TR, RS_IN_TU, RS_IN_TD
                    port_name = "CH" if port == "ch_buffer" else port
                    flit.set_position(f"RS_IN_{port_name}", cycle)
                    self.input_fifos[port].append(flit)
                self.input_fifos_pre[port] = None

        # 输出端 pre → FIFO
        for port, flit in self.output_fifos_pre.items():
            if flit is not None:
                if len(self.output_fifos[port]) < self.output_fifos[port].maxlen:
                    # 命名格式: RS_OUT_CH, RS_OUT_TL, RS_OUT_TR, RS_OUT_TU, RS_OUT_TD
                    port_name = "CH" if port == "ch_buffer" else port
                    flit.set_position(f"RS_OUT_{port_name}", cycle)
                    self.output_fifos[port].append(flit)
                self.output_fifos_pre[port] = None

    def process_cycle(self, cycle: int):
        """
        每周期处理：
        1. 收集路由请求
        2. 仲裁决定每个输出端口的来源
        3. 执行数据转移
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
        1. 到达目的地节点 → ch_buffer（本地弹出）
        2. 需要横向移动 → TL 或 TR
        3. 需要纵向移动 → TU 或 TD
        """
        # 获取最终目的地
        final_dest = flit.path[-1] if flit.path else flit.destination

        # Case 1: 已到达目的地
        if self.node_id == final_dest:
            return "ch_buffer"

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
            return "ch_buffer"  # 已到达

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

            # v2统一Entry释放：当flit从环方向输入端口转出时，释放对应的Entry
            if input_port in self.RING_DIRECTIONS and hasattr(flit, 'used_entry_level'):
                eject_dir = getattr(flit, 'eject_direction', None)
                if eject_dir in self.RING_DIRECTIONS and self.network:
                    self.network.RS_pending_entry_release[eject_dir][self.node_id].append(
                        (flit.used_entry_level, cycle + 1)
                    )

            # 更新 flit position（input → output_pre，等待下周期移入 output FIFO）
            port_name = "CH" if output_port == "ch_buffer" else output_port
            flit.set_position(f"RS_OUT_{port_name}", cycle)

            # 添加到输出端口 pre 缓冲
            self.output_fifos_pre[output_port] = flit

            # 更新统计
            if self._is_cross_dimension(input_port, output_port):
                self.stats["cross_dimension_transfers"] += 1
            if output_port == "ch_buffer":
                self.stats["local_ejects"] += 1
            if input_port == "ch_buffer":
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

    # ------------------------------------------------------------------
    # 外部接口方法
    # ------------------------------------------------------------------

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

    def enqueue_from_local(self, flit: Flit) -> bool:
        """
        从本地 IP 注入的 flit 进入 RS 输入端

        Args:
            flit: 本地注入的 flit

        Returns:
            bool: 是否成功入队
        """
        if self.input_fifos_pre["ch_buffer"] is not None:
            return False

        self.input_fifos_pre["ch_buffer"] = flit
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

    def dequeue_to_local(self) -> Optional[Flit]:
        """
        从 RS 输出端取出准备弹出到本地 IP 的 flit

        Returns:
            Flit or None
        """
        if self.output_fifos["ch_buffer"]:
            return self.output_fifos["ch_buffer"].popleft()
        return None

    def peek_output(self, port: str) -> Optional[Flit]:
        """查看输出端口队首的 flit（不移除）"""
        if port in self.output_fifos and self.output_fifos[port]:
            return self.output_fifos[port][0]
        return None

    def has_output(self, port: str) -> bool:
        """检查输出端口是否有 flit"""
        return port in self.output_fifos and len(self.output_fifos[port]) > 0

    def can_accept_input(self, port: str) -> bool:
        """检查输入端口是否可以接受新 flit"""
        if port not in self.input_fifos:
            return False
        return self.input_fifos_pre[port] is None

    # ------------------------------------------------------------------
    # 调试和统计方法
    # ------------------------------------------------------------------

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
