"""
CrossRing v2 仿真系统 - 数据流处理Mixin

本模块提供网络数据流处理相关功能，包括：
- IP ↔ RingStation 数据流管理
- 网络周期处理和flit移动
- pre缓冲与主FIFO之间的转移
- Entry释放和统计更新

主要类:
    DataflowMixin: 数据流处理功能混入类，提供网络数据流相关方法
"""

from src.kcin.v2.components import Network, IPInterface
from src.utils.flit import Flit


class DataflowMixin:
    """网络数据流处理功能Mixin"""

    def ip_inject_to_network(self):
        for node_id in range(self.config.NUM_NODE):
            for ip_type in self.config.CH_NAME_LIST:
                # 检查IP接口是否存在，避免KeyError
                ip_key = (ip_type, node_id)
                if ip_key in self.ip_modules:
                    ip_interface: IPInterface = self.ip_modules[ip_key]
                    ip_interface.inject_step(self.cycle)

    def network_to_ip_eject(self):
        """从网络到IP的eject步骤，并更新received_flit统计"""
        # 直接遍历实际创建的IP接口(动态挂载模式)
        for (ip_type, node_id), ip_interface in self.ip_modules.items():
            # 执行eject，获取已到达目的IP的flit列表
            ejected_flits = ip_interface.eject_step(self.cycle)
            # 更新TrafficScheduler中的received_flit统计
            if ejected_flits:
                for flit in ejected_flits:
                    self.update_traffic_completion_stats(flit)

    def release_completed_sn_tracker(self):
        """Check if any trackers can be released based on the current cycle."""
        # 遍历所有IP模块，检查各自的tracker释放队列
        for (ip_type, node_id), ip_interface in self.ip_modules.items():
            for release_time in sorted(ip_interface.sn_tracker_release_time.keys()):
                if release_time > self.cycle:
                    continue
                tracker_list = ip_interface.sn_tracker_release_time.pop(release_time)
                for req in tracker_list:
                    # 检查 tracker 是否还在列表中（避免重复释放）
                    if req in ip_interface.sn_tracker:
                        ip_interface.release_completed_sn_tracker(req)

    def _move_pre_to_queues(self, network: Network, node_id, network_type: str):
        """v2统一架构: 所有 pre → FIFO 的移动 + IP ↔ RingStation 数据流处理

        执行4个Phase的数据传输:
        Phase 1: RS输入端 pre → input_fifos
        Phase 2: IP发送缓冲 → RS输入端pre (IP → RS)
        Phase 3: RS输出端 pre → output_fifos
        Phase 4: RS输出端FIFO → IP接收缓冲 (RS → IP)

        为什么需要分4个Phase？
        ======================

        问题背景:
        在单个仿真周期内，同一个FIFO可能需要被多次读写。如果不分Phase，
        会出现"读后写"冲突：

        冲突场景示例（如果不分Phase）：
        1. RS.process_cycle()执行仲裁，决定将input_fifos[TL][0]转移到output_fifos_pre[ddr_0]
        2. 立即执行：output_fifos_pre[ddr_0] → output_fifos[ddr_0]
        3. 立即执行：output_fifos[ddr_0] → IP.rx_channel_buffer
        4. 问题：RS在同一周期内既读又写output_fifos[ddr_0]，违反硬件时序

        Phase分离解决方案:
        ------------------
        Phase 1: 将上周期产生的input_fifos_pre内容移入input_fifos
                 → 为本周期仲裁提供输入数据

        Phase 2: 将IP的发送缓冲移入RS的input_fifos_pre
                 → 准备下周期的输入（不会与Phase 1冲突）

        Phase 3: 将本周期仲裁产生的output_fifos_pre移入output_fifos
                 → 使本周期仲裁结果可见

        Phase 4: 从output_fifos取数据到IP接收缓冲
                 → 不会与Phase 3冲突（Phase 3已完成）

        关键设计:
        - pre缓冲作为"延迟一周期"的机制，实现管道化
        - 每个Phase严格顺序执行，避免同一FIFO的读写冲突
        - Phase 1和Phase 3确保仲裁逻辑看到正确的FIFO状态

        Args:
            network: 网络实例
            node_id: 节点ID
            network_type: 网络类型 ("req" / "rsp" / "data")
        """
        rs = network.ring_stations[node_id]

        # === Phase 1: RS 输入端 pre → input_fifos ===
        rs.move_input_pre_to_fifos(self.cycle)

        # === IP发送 → RS接收 ===
        # IP.tx_channel_buffer_pre → RS.input_fifos_pre[ip_type]
        for (ip_type, ip_pos), ip_interface in self.ip_modules.items():
            if ip_pos != node_id:
                continue
            net_info = ip_interface.networks.get(network_type)
            if net_info is None:
                continue

            tx_pre = net_info.get("tx_channel_buffer_pre")
            if tx_pre is not None and rs.can_accept_input(ip_type):
                flit = tx_pre
                flit.set_position(f"RS_IN_{ip_type}", self.cycle)
                rs.input_fifos_pre[ip_type] = flit
                net_info["tx_channel_buffer_pre"] = None
                network.increment_fifo_flit_count("IQ", "CH_buffer", node_id, ip_type)

        # === Phase 3: RS 输出端 pre → output_fifos ===
        rs.move_output_pre_to_fifos(self.cycle)

        # === RS输出 → IP接收 ===
        # RS.output_fifos[ip_type] → IP.rx_channel_buffer
        for (ip_type, ip_pos), ip_interface in self.ip_modules.items():
            if ip_pos != node_id:
                continue

            # 检查该IP类型对应的RS输出队列是否有数据
            if ip_type not in rs.output_fifos or not rs.output_fifos[ip_type]:
                continue

            net_info = ip_interface.networks.get(network_type)
            if net_info is None:
                continue

            rx_buf = net_info["rx_channel_buffer"]
            if len(rx_buf) < rx_buf.maxlen:
                flit = rs.output_fifos[ip_type].popleft()
                flit.set_position("IP_RX_CH", self.cycle)
                rx_buf.append(flit)
                network.increment_fifo_flit_count("EQ", "CH_buffer", node_id, ip_type)

        # 更新FIFO统计
        network.update_fifo_stats_after_move(node_id)

    def print_data_statistic(self):
        if self.verbose:
            print(f"Data statistic: Read: {self.read_req, self.read_flit}, " f"Write: {self.write_req, self.write_flit}, " f"Total: {self.read_req + self.write_req, self.read_flit + self.write_flit}")

    def log_summary(self):
        if self.verbose:
            print(
                f"T: {self.cycle // self.config.CYCLES_PER_NS}, Req_cnt: {self.req_count} In_Req: {self.req_num}, Rsp: {self.rsp_num},"
                f" R_fn: {self.send_read_flits_num_stat}, W_fn: {self.send_write_flits_num_stat}, "
                f"Trans_fn: {self.trans_flits_num}, Recv_fn: {self.data_network.recv_flits_num}"
            )

    def move_flits_in_network(self, network, flits, flit_type):
        """Process injection queues and move flits."""
        flits = self._network_cycle_process(network, flits, flit_type)
        return flits

    def move_pre_to_queues_all(self):
        #  所有 IPInterface 的 *_pre → FIFO
        # 直接遍历实际创建的IP接口(动态挂载模式)
        for (ip_type, node_id), ip_interface in self.ip_modules.items():
            ip_interface.move_pre_to_fifo()

        # 所有网络的 *_pre → FIFO
        for node_id in range(self.config.NUM_NODE):
            self._move_pre_to_queues(self.req_network, node_id, "req")
            self._move_pre_to_queues(self.rsp_network, node_id, "rsp")
            self._move_pre_to_queues(self.data_network, node_id, "data")

    def update_throughput_metrics(self, flits):
        """Update throughput metrics based on flit counts."""
        self.trans_flits_num = len(flits)

    def _get_next_hop_for_node(self, flit, current_node):
        """
        根据flit的路径与path_index定位当前节点的下一跳。

        Returns:
            int | None: 下一跳节点ID，若不存在或无法确定则返回None。
        """
        path = getattr(flit, "path", None)
        if not path:
            return None

        path_len = len(path)
        if path_len <= 1:
            return None

        path_index = getattr(flit, "path_index", None)
        candidate_idx = None

        if isinstance(path_index, int):
            for offset in (0, -1, 1):
                idx = path_index + offset
                if 0 <= idx < path_len and path[idx] == current_node:
                    candidate_idx = idx
                    break

        if candidate_idx is None:
            # Fallback: 从后向前搜索当前节点
            try:
                reverse_idx = path[::-1].index(current_node)
                candidate_idx = path_len - 1 - reverse_idx
            except ValueError:
                return None

        if candidate_idx + 1 < path_len:
            next_hop = path[candidate_idx + 1]
            if next_hop != current_node:
                return next_hop
        return None

    def _process_pending_entry_release(self, network):
        """处理延迟释放的Entry计数器

        在每个cycle末尾调用，检查并释放所有到期(cycle <= current_cycle)的Entry
        v2统一Entry管理：使用 RS_pending_entry_release 和 RS_UE_Counters
        """
        for direction in ["TL", "TR", "TU", "TD"]:
            for node_id, pending_list in network.RS_pending_entry_release[direction].items():
                to_remove = []
                for idx, (level, release_cycle) in enumerate(pending_list):
                    if release_cycle <= self.cycle:
                        network.RS_UE_Counters[direction][node_id][level] -= 1
                        to_remove.append(idx)
                # 从后往前删除，避免索引变化
                for idx in reversed(to_remove):
                    pending_list.pop(idx)

    # ------------------------------------------------------------------
    # v2 RingStation 架构相关方法
    # ------------------------------------------------------------------

    def process_inject_from_rs(self, network: Network, direction: str):
        """v2架构：从RingStation输出端口上环

        Args:
            network: 网络实例
            direction: 注入方向 (TL/TR/TU/TD)

        Returns:
            tuple: (注入数量, 注入的flit列表)
        """
        flit_num = 0
        flits = []

        is_horizontal = direction in ["TL", "TR"]
        cp_type = "horizontal" if is_horizontal else "vertical"
        wait_attr = "wait_cycle_h" if is_horizontal else "wait_cycle_v"
        threshold = self.config.ITag_TRIGGER_Th_H if is_horizontal else self.config.ITag_TRIGGER_Th_V

        for node_id in range(self.config.NUM_NODE):
            # 检查RS输出端口是否有flit
            if not network.rs_has_output(node_id, direction):
                continue

            # 获取CrossPoint并调用v2版本的注入方法
            crosspoint = network.crosspoints[node_id][cp_type]
            injected_flit = crosspoint.process_inject_from_rs(node_id, direction, self.cycle)

            if injected_flit:
                # 首次上环时分配order_id
                if injected_flit.src_dest_order_id == -1:
                    src_node = injected_flit.source
                    dest_node = injected_flit.destination
                    src_type = injected_flit.source_type
                    dest_type = injected_flit.destination_type
                    die_id = getattr(self.config, "DIE_ID", None)
                    injected_flit.src_dest_order_id = Flit.get_next_order_id(
                        src_node, src_type, dest_node, dest_type,
                        injected_flit.flit_type.upper(),
                        self.config.ORDERING_GRANULARITY, die_id
                    )

                # 横向注入更新统计
                if is_horizontal:
                    network.inject_num += 1
                    flit_num += 1

                # 纵向注入更新flit状态
                if not is_horizontal:
                    injected_flit.current_position = node_id
                    injected_flit.path_index += 1

                flits.append(injected_flit)

                # ITag释放处理
                if getattr(injected_flit, wait_attr, 0) >= threshold:
                    network.itag_req_counter[direction][node_id] -= 1
                    excess = network.tagged_counter[direction][node_id] - network.itag_req_counter[direction][node_id]
                    if excess > 0:
                        network.excess_ITag_to_remove[direction][node_id] += excess

        return flit_num, flits

    def _CP_process(self, network: Network, flits, flit_type: str):
        """CrossPoint处理（从RS输出上环）

        Args:
            network: 网络实例
            flits: 当前网络中的flit列表
            flit_type: flit类型 ("req" / "rsp" / "data")

        Returns:
            list: 更新后的flits列表
        """
        # 从RS输出端口上环（TL/TR/TU/TD四个方向）
        for direction in ["TL", "TR", "TU", "TD"]:
            num, injected_flits = self.process_inject_from_rs(network, direction)

            # 横向注入需要更新统计
            if direction in ["TL", "TR"] and num > 0:
                if flit_type == "req":
                    self.req_num += num
                elif flit_type == "rsp":
                    self.rsp_num += num
                elif flit_type == "data":
                    self.flit_num += num

            # 添加注入的flit到列表
            for flit in injected_flits:
                if flit not in flits:
                    flits.append(flit)

        # 更新ITag状态
        network.update_excess_ITag()

        return flits

    def _network_cycle_process(self, network: Network, flits, flit_type: str):
        """v2架构：网络周期处理（使用RingSlice环形链表）

        处理顺序：
        1. IP.tx_channel_buffer → RS.input_fifos[ch_buffer]（本地注入，由_move_pre_to_queues处理）
        2. RS处理（仲裁 + 内部路由）
        3. 环形链表处理（移动 + 下环 + 上环）
        4. RS.output_fifos[ch_buffer] → IP.rx_channel_buffer（本地弹出，由_move_pre_to_queues处理）

        Args:
            network: 网络实例
            flits: 当前网络中的flit列表（新系统中flit存储在RingSlice中）
            flit_type: flit类型 ("req" / "rsp" / "data")

        Returns:
            list: 更新后的flits列表
        """
        # 1. RS处理（内部仲裁和路由）
        network.process_ring_stations(self.cycle)

        # 2. 环形链表处理（移动 + 下环 + 上环）
        network.process_all_rings(self.cycle)

        # 3. 更新ITag状态
        network.update_excess_ITag()

        # IP↔RS 数据流在 _move_pre_to_queues 中统一处理
        return flits
