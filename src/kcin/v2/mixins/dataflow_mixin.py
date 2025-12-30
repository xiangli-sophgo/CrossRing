"""
网络数据流处理相关的Mixin类
包含IQ/Link/RB/EQ/CP处理、Flit移动、Tag处理等功能
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

    def _move_pre_to_queues(self, network: Network, node_id):
        """Move all items from pre-injection queues to injection queues for a given network."""
        # ===  注入队列 *_pre → *_FIFO ===

        # IQ_channel_buffer_pre → IQ_channel_buffer
        for ip_type in network.IQ_channel_buffer_pre.keys():
            queue_pre = network.IQ_channel_buffer_pre[ip_type]
            queue = network.IQ_channel_buffer[ip_type]
            if queue_pre[node_id] and len(queue[node_id]) < self.config.RS_IN_CH_BUFFER:
                flit = queue_pre[node_id]
                flit.set_position("RS_IN_CH", self.cycle)
                queue[node_id].append(flit)
                network.increment_fifo_flit_count("IQ", "CH_buffer", node_id, ip_type)
                queue_pre[node_id] = None

        # IQ_arbiter_input_fifo_pre → IQ_arbiter_input_fifo
        for ip_type in network.IQ_arbiter_input_fifo_pre.keys():
            queue_pre = network.IQ_arbiter_input_fifo_pre[ip_type]
            queue = network.IQ_arbiter_input_fifo[ip_type]
            if queue_pre[node_id] and len(queue[node_id]) < 2:
                flit = queue_pre[node_id]
                queue[node_id].append(flit)
                queue_pre[node_id] = None

        # IQ_pre → IQ_OUT
        for direction in self.IQ_directions:
            queue_pre = network.inject_queues_pre[direction]
            queue = network.inject_queues[direction]

            # 根据方向选择对应的FIFO深度
            if direction in ["TR", "TL"]:
                fifo_depth = self.config.RS_OUT_FIFO_DEPTH
            elif direction in ["TU", "TD"]:
                fifo_depth = self.config.RS_OUT_FIFO_DEPTH
            else:  # EQ
                fifo_depth = self.config.RS_OUT_CH_BUFFER

            if queue_pre[node_id] and len(queue[node_id]) < fifo_depth:
                flit = queue_pre[node_id]
                flit.departure_inject_cycle = self.cycle
                flit.set_position(f"RS_OUT_{direction}", self.cycle)
                queue[node_id].append(flit)
                network.increment_fifo_flit_count("IQ", direction, node_id)
                # 统计横向反方向上环
                if direction in ["TR", "TL"] and getattr(flit, "reverse_inject_h", 0) == 1:
                    network.increment_fifo_reverse_inject_count("IQ", direction, node_id)
                queue_pre[node_id] = None

        # RB_IN_PRE → RB_IN
        for direction in ["TL", "TR"]:
            queue_pre = network.ring_bridge_pre[direction]
            queue = network.ring_bridge[direction]
            if queue_pre[node_id] and len(queue[node_id]) < self.config.RS_IN_FIFO_DEPTH:
                flit = queue_pre[node_id]
                flit.set_position(f"RS_IN_{direction}", self.cycle)
                queue[node_id].append(flit)
                network.increment_fifo_flit_count("RB", direction, node_id)
                queue_pre[node_id] = None

        # RB_OUT_PRE → RB_OUT
        for fifo_pos in ("EQ", "TU", "TD"):
            queue_pre = network.ring_bridge_pre[fifo_pos]
            queue = network.ring_bridge[fifo_pos]
            if queue_pre[node_id] and len(queue[node_id]) < self.config.RS_OUT_FIFO_DEPTH:
                flit = queue_pre[node_id]
                flit.is_arrive = fifo_pos == "EQ"
                flit.set_position(f"RS_OUT_{fifo_pos}", self.cycle)
                queue[node_id].append(flit)
                network.increment_fifo_flit_count("RB", fifo_pos, node_id)
                # 统计纵向反方向上环
                if fifo_pos in ["TU", "TD"] and getattr(flit, "reverse_inject_v", 0) == 1:
                    network.increment_fifo_reverse_inject_count("RB", fifo_pos, node_id)
                queue_pre[node_id] = None

        # EQ_IN_PRE → EQ_IN
        for fifo_pos in ("TU", "TD"):
            queue_pre = network.eject_queues_in_pre[fifo_pos]
            queue = network.eject_queues[fifo_pos]
            if queue_pre[node_id] and len(queue[node_id]) < self.config.RS_IN_FIFO_DEPTH:
                flit = queue_pre[node_id]
                flit.is_arrive = fifo_pos == "EQ"
                flit.set_position(f"RS_IN_{fifo_pos}", self.cycle)
                queue[node_id].append(flit)
                network.increment_fifo_flit_count("EQ", fifo_pos, node_id)
                queue_pre[node_id] = None

        # EQ_arbiter_input_fifo_pre → EQ_arbiter_input_fifo
        for port_name in ["TU", "TD", "IQ", "RB"]:
            queue_pre = network.EQ_arbiter_input_fifo_pre[port_name]
            queue = network.EQ_arbiter_input_fifo[port_name]
            if queue_pre[node_id] and len(queue[node_id]) < 2:
                flit = queue_pre[node_id]
                queue[node_id].append(flit)
                queue_pre[node_id] = None

        # EQ_channel_buffer_pre → EQ_channel_buffer
        for ip_type in network.EQ_channel_buffer_pre.keys():
            queue_pre = network.EQ_channel_buffer_pre[ip_type]
            queue = network.EQ_channel_buffer[ip_type]
            if queue_pre[node_id] and len(queue[node_id]) < self.config.RS_OUT_CH_BUFFER:
                flit = queue_pre[node_id]
                flit.set_position("RS_OUT_CH", self.cycle)
                queue[node_id].append(flit)
                network.increment_fifo_flit_count("EQ", "CH_buffer", node_id, ip_type)
                queue_pre[node_id] = None

        # 更新FIFO统计
        network.update_fifo_stats_after_move(node_id)

    def print_data_statistic(self):
        if self.verbose:
            print(f"Data statistic: Read: {self.read_req, self.read_flit}, " f"Write: {self.write_req, self.write_flit}, " f"Total: {self.read_req + self.write_req, self.read_flit + self.write_flit}")

    def log_summary(self):
        if self.verbose:
            print(
                f"T: {self.cycle // self.config.NETWORK_FREQUENCY}, Req_cnt: {self.req_count} In_Req: {self.req_num}, Rsp: {self.rsp_num},"
                f" R_fn: {self.send_read_flits_num_stat}, W_fn: {self.send_write_flits_num_stat}, "
                f"Trans_fn: {self.trans_flits_num}, Recv_fn: {self.data_network.recv_flits_num}"
            )

    def move_flits_in_network(self, network, flits, flit_type):
        """Process injection queues and move flits."""
        flits = self._network_cycle_process(network, flits, flit_type)
        return flits

    def _try_inject_to_direction(self, req: Flit, ip_type, node_id, direction, counts):
        """检查tracker空间并尝试注入到指定direction的pre缓冲"""
        # 设置flit的允许下环方向（仅在第一次注入时设置）
        if not hasattr(req, "allowed_eject_directions") or req.allowed_eject_directions is None:
            req.allowed_eject_directions = self.req_network.determine_allowed_eject_directions(req)

        # 直接注入到指定direction的pre缓冲
        queue_pre = self.req_network.inject_queues_pre[direction]
        queue_pre[node_id] = req

        # 从仲裁输入FIFO移除
        self.req_network.IQ_arbiter_input_fifo[ip_type][node_id].popleft()

        # 更新计数和状态
        if req.req_attr == "new":  # 只有新请求才更新计数器和tracker
            if req.req_type == "read":
                counts["read"] += 1

            elif req.req_type == "write":
                counts["write"] += 1

        return True

    def _Link_process(self, network: Network, flits):
        """Link模块：Link传输处理

        处理Link上的flit移动

        Args:
            network: 网络实例
            flits: 当前网络中的flit列表

        Returns:
            list: 更新后的flits列表
        """
        # 第一步：对Link上的flit执行plan_move
        for flit in flits:
            if flit.flit_position == "Link":
                network.plan_move(flit, self.cycle)

        # 第二步：执行execute_moves并收集需要移除的flit
        executed_flits = set()
        for flit in flits:
            if network.execute_moves(flit, self.cycle):
                executed_flits.add(id(flit))

        # 第三步：一次过滤重建列表（O(n)，比多次remove快）
        if executed_flits:
            flits[:] = [flit for flit in flits if id(flit) not in executed_flits]

        return flits

    def _check_iq_injection_conditions(self, network, node_id, ip_type, direction, network_type, flit_cache):
        """
        检查是否可以从ip_type注入到direction

        Returns:
            bool: 是否可以注入
        """
        # 检查round_robin队列中是否有这个ip_type
        rr_queue = network.round_robin["IQ"][direction][node_id]
        if ip_type not in rr_queue:
            return False

        # 检查pre槽是否占用
        queue_pre = network.inject_queues_pre[direction]
        if queue_pre[node_id]:
            return False

        # 检查FIFO是否满
        queue = network.inject_queues[direction]
        if direction in ["TR", "TL"]:
            fifo_depth = self.config.RS_OUT_FIFO_DEPTH
        elif direction in ["TU", "TD"]:
            fifo_depth = self.config.RS_OUT_FIFO_DEPTH
        else:  # EQ
            fifo_depth = self.config.RS_OUT_CH_BUFFER

        if len(queue[node_id]) >= fifo_depth:
            return False

        # 网络特定 ip_type 过滤
        if network_type == "req" and not (ip_type.startswith("sdma") or ip_type.startswith("gdma") or ip_type.startswith("cdma") or ip_type.startswith("d2d_rn")):
            return False
        if network_type == "rsp" and not (ip_type.startswith("ddr") or ip_type.startswith("l2m") or ip_type.startswith("d2d_sn")):
            return False

        # 检查仲裁输入FIFO是否为空
        if not network.IQ_arbiter_input_fifo[ip_type][node_id]:
            return False

        flit = network.IQ_arbiter_input_fifo[ip_type][node_id][0]

        # 缓存flit供后续使用
        flit_cache[(ip_type, direction)] = flit

        # 反方向流控检查（横向TL/TR）
        if self.config.REVERSE_DIRECTION_ENABLED and direction in ["TL", "TR"]:
            normal_direction = None
            if len(flit.path) > 1:
                diff = flit.path[1] - flit.path[0]
                if diff == 1:
                    normal_direction = "TR"
                elif diff == -1:
                    normal_direction = "TL"

            # 如果当前检查的方向是反方向，进行流控判断
            if normal_direction and direction != normal_direction:
                reverse_direction = direction
                normal_depth = len(network.inject_queues[normal_direction][node_id])
                reverse_depth = len(network.inject_queues[reverse_direction][node_id])
                capacity = self.config.RS_OUT_FIFO_DEPTH

                # 只有当正常方向比反方向拥塞程度超过容量×阈值时，才允许走反方向
                if (normal_depth - reverse_depth) > capacity * self.config.REVERSE_DIRECTION_THRESHOLD:
                    return True  # 正常方向比反方向拥塞很多，允许走反方向
                else:
                    return False  # 差距不够大，不走反方向

        # 检查方向条件（正常逻辑）
        if not self.IQ_direction_conditions[direction](flit):
            return False

        return True

    def move_pre_to_queues_all(self):
        #  所有 IPInterface 的 *_pre → FIFO
        # 直接遍历实际创建的IP接口(动态挂载模式)
        for (ip_type, node_id), ip_interface in self.ip_modules.items():
            ip_interface.move_pre_to_fifo()

        # 所有网络的 *_pre → FIFO
        for node_id in range(self.config.NUM_NODE):
            self._move_pre_to_queues(self.req_network, node_id)
            self._move_pre_to_queues(self.rsp_network, node_id)
            self._move_pre_to_queues(self.data_network, node_id)

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

    def tag_move_all_networks(self):
        self._tag_move(self.req_network)
        self._tag_move(self.rsp_network)
        self._tag_move(self.data_network)

    def _tag_move(self, network: Network):
        # 第一部分：纵向环处理
        for col_start in range(self.config.NUM_COL):
            interval = self.config.NUM_COL  # 新架构: 直接使用NUM_COL
            col_end = col_start + interval * (self.config.NUM_ROW - 1)  # 最后一行节点

            # 保存起始位置的tag（使用垂直自环键）
            v_key_start = (col_start, col_start, "v")
            last_position = network.links_tag[v_key_start][0]

            # 前向传递：从起点到终点
            network.links_tag[v_key_start][0] = network.links_tag[(col_start + interval, col_start)][-1]

            for i in range(1, self.config.NUM_ROW):  # 新架构: 遍历所有行
                current_node = col_start + i * interval
                next_node = col_start + (i - 1) * interval

                for j in range(self.config.SLICE_PER_LINK_VERTICAL - 1, -1, -1):
                    if j == 0 and current_node == col_end:
                        v_key_current = (current_node, current_node, "v")
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[v_key_current][-1]
                    elif j == 0:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node + interval, current_node)][-1]
                    else:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]

            # 终点自环处理（使用垂直自环键）
            v_key_end = (col_end, col_end, "v")
            network.links_tag[v_key_end][-1] = network.links_tag[v_key_end][0]
            network.links_tag[v_key_end][0] = network.links_tag[(col_end - interval, col_end)][-1]

            # 回程传递：从终点回到起点
            # 修复：确保处理所有回程连接
            for i in range(1, self.config.NUM_ROW):  # 新架构: 遍历所有行
                current_node = col_end - i * interval
                next_node = col_end - (i - 1) * interval

                for j in range(self.config.SLICE_PER_LINK_VERTICAL - 1, -1, -1):
                    if j == 0 and current_node == col_start:
                        v_key_current = (current_node, current_node, "v")
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[v_key_current][-1]
                    elif j == 0:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node - interval, current_node)][-1]
                    else:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]

            # 起点自环处理（使用垂直自环键）
            network.links_tag[v_key_start][-1] = last_position

        # 第二部分：横向环处理
        # Skip horizontal tag movement if only one column or links_tag missing
        if self.config.NUM_COL <= 1:
            return
        # 新架构: 遍历所有行 (包括第一行row=0)
        for row_start in range(0, self.config.NUM_NODE, self.config.NUM_COL):
            row_end = row_start + self.config.NUM_COL - 1
            # 使用水平自环键
            h_key_start = (row_start, row_start, "h")
            if h_key_start not in network.links_tag:
                continue
            last_position = network.links_tag[h_key_start][0]
            if (row_start + 1, row_start) in network.links_tag:
                network.links_tag[h_key_start][0] = network.links_tag[(row_start + 1, row_start)][-1]
            else:
                network.links_tag[h_key_start][0] = last_position

            for i in range(1, self.config.NUM_COL):
                current_node, next_node = row_start + i, row_start + i - 1
                for j in range(self.config.SLICE_PER_LINK_HORIZONTAL - 1, -1, -1):
                    if j == 0 and current_node == row_end:
                        h_key_end = (current_node, current_node, "h")
                        if h_key_end in network.links_tag and (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[h_key_end][-1]
                    elif j == 0:
                        if (current_node + 1, current_node) in network.links_tag and (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node + 1, current_node)][-1]
                    else:
                        if (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]

            h_key_end = (row_end, row_end, "h")
            if h_key_end in network.links_tag:
                network.links_tag[h_key_end][-1] = network.links_tag[h_key_end][0]
                if (row_end - 1, row_end) in network.links_tag:
                    network.links_tag[h_key_end][0] = network.links_tag[(row_end - 1, row_end)][-1]
                else:
                    network.links_tag[h_key_end][0] = last_position

            for i in range(1, self.config.NUM_COL):
                current_node, next_node = row_end - i, row_end - i + 1
                for j in range(self.config.SLICE_PER_LINK_HORIZONTAL - 1, -1, -1):
                    if j == 0 and current_node == row_start:
                        h_key_current = (current_node, current_node, "h")
                        if h_key_current in network.links_tag and (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[h_key_current][-1]
                    elif j == 0:
                        if (current_node - 1, current_node) in network.links_tag and (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node - 1, current_node)][-1]
                    else:
                        if (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]

            if h_key_start in network.links_tag:
                network.links_tag[h_key_start][-1] = last_position

    def _process_pending_entry_release(self, network):
        """处理延迟释放的Entry计数器

        在每个cycle末尾调用，检查并释放所有到期(cycle <= current_cycle)的Entry
        """
        # 处理RB TL方向
        for node_id, pending_list in network.RB_pending_entry_release["TL"].items():
            to_remove = []
            for idx, (level, release_cycle) in enumerate(pending_list):
                if release_cycle <= self.cycle:
                    network.RB_UE_Counters["TL"][node_id][level] -= 1
                    to_remove.append(idx)
            # 从后往前删除，避免索引变化
            for idx in reversed(to_remove):
                pending_list.pop(idx)

        # 处理RB TR方向
        for node_id, pending_list in network.RB_pending_entry_release["TR"].items():
            to_remove = []
            for idx, (level, release_cycle) in enumerate(pending_list):
                if release_cycle <= self.cycle:
                    network.RB_UE_Counters["TR"][node_id][level] -= 1
                    to_remove.append(idx)
            for idx in reversed(to_remove):
                pending_list.pop(idx)

        # 处理EQ TU方向
        for node_id, pending_list in network.EQ_pending_entry_release["TU"].items():
            to_remove = []
            for idx, (level, release_cycle) in enumerate(pending_list):
                if release_cycle <= self.cycle:
                    network.EQ_UE_Counters["TU"][node_id][level] -= 1
                    to_remove.append(idx)
            for idx in reversed(to_remove):
                pending_list.pop(idx)

        # 处理EQ TD方向
        for node_id, pending_list in network.EQ_pending_entry_release["TD"].items():
            to_remove = []
            for idx, (level, release_cycle) in enumerate(pending_list):
                if release_cycle <= self.cycle:
                    network.EQ_UE_Counters["TD"][node_id][level] -= 1
                    to_remove.append(idx)
            for idx in reversed(to_remove):
                pending_list.pop(idx)

    def process_inject_queues(self, network: Network, inject_queues, direction):
        """统一的CrossPoint注入处理（支持TL/TR/TU/TD四个方向）

        Args:
            inject_queues: 对于TL/TR是network.inject_queues[direction]
                          对于TU/TD是network.ring_bridge[direction]
            direction: TL/TR/TU/TD
        """
        flit_num = 0
        flits = []

        # 判断是横向还是纵向
        is_horizontal = direction in ["TL", "TR"]
        cp_type = "horizontal" if is_horizontal else "vertical"
        wait_attr = "wait_cycle_h" if is_horizontal else "wait_cycle_v"
        itag_attr = "itag_h" if is_horizontal else "itag_v"
        threshold = self.config.ITag_TRIGGER_Th_H if is_horizontal else self.config.ITag_TRIGGER_Th_V

        for node_id, queue in inject_queues.items():
            if not queue or not queue[0]:
                continue

            # 1. 检查是否需要生成Buffer_Reach_Th信号
            flit = queue[0]
            if getattr(flit, wait_attr) == threshold:
                network.itag_req_counter[direction][node_id] += 1

            # 2. 获取CrossPoint并调用统一的注入方法
            crosspoint = network.crosspoints[node_id][cp_type]
            injected_flit = crosspoint.process_inject(node_id, queue, direction, self.cycle)

            if injected_flit:
                # 3. 首次上环时分配order_id
                # 设计说明：每个Die独立保序，使用flit的实际source/destination和die_id
                if injected_flit.src_dest_order_id == -1:
                    src_node = injected_flit.source
                    dest_node = injected_flit.destination
                    src_type = injected_flit.source_type
                    dest_type = injected_flit.destination_type
                    die_id = getattr(self.config, "DIE_ID", None)
                    injected_flit.src_dest_order_id = Flit.get_next_order_id(src_node, src_type, dest_node, dest_type, injected_flit.flit_type.upper(), self.config.ORDERING_GRANULARITY, die_id)

                # 4. 横向注入需要更新inject_num统计
                if is_horizontal:
                    network.inject_num += 1
                    flit_num += 1

                # 5. 纵向注入需要更新flit状态
                if not is_horizontal:
                    injected_flit.current_position = node_id
                    injected_flit.path_index += 1

                flits.append(injected_flit)

                # 7. ITag释放处理（统一逻辑）
                if getattr(injected_flit, wait_attr) >= threshold:
                    network.itag_req_counter[direction][node_id] -= 1
                    excess = network.tagged_counter[direction][node_id] - network.itag_req_counter[direction][node_id]
                    if excess > 0:
                        network.excess_ITag_to_remove[direction][node_id] += excess

            # 8. ITag统计
            if queue and queue[0] and getattr(queue[0], itag_attr, False):
                if is_horizontal:
                    self.ITag_h_num_stat += 1
                else:
                    self.ITag_v_num_stat += 1

        return flit_num, flits

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

        # 更新ITag和CrossPoint状态
        network.update_excess_ITag()
        network.update_cross_point()

        return flits

    def _move_IQ_to_RS(self, network: Network):
        """v2架构：将IQ_channel_buffer移动到RS的ch_buffer输入端口

        Args:
            network: 网络实例
        """
        for node_id in range(self.config.NUM_NODE):
            # 检查RS是否可以接受输入
            if not network.rs_can_accept_input(node_id, "ch_buffer"):
                continue

            # 遍历所有IP类型的channel buffer
            for ip_type in network.IQ_channel_buffer.keys():
                if not network.IQ_channel_buffer[ip_type][node_id]:
                    continue

                # peek flit
                flit = network.IQ_channel_buffer[ip_type][node_id][0]

                # 尝试发送到RS
                if network.rs_enqueue_from_local(node_id, flit):
                    # 成功后从IQ移除
                    network.IQ_channel_buffer[ip_type][node_id].popleft()
                    break  # 每个周期每个节点只能发送一个

    def _move_RS_to_EQ(self, network: Network):
        """v2架构：将RS的ch_buffer输出移动到EQ_channel_buffer

        Args:
            network: 网络实例
        """
        for node_id in range(self.config.NUM_NODE):
            # 从RS取出准备弹出到本地的flit
            flit = network.rs_dequeue_to_local(node_id)
            if flit is None:
                continue

            # 确定目标IP类型
            dest_type = flit.destination_type
            if dest_type not in network.EQ_channel_buffer:
                continue

            # 检查EQ_channel_buffer是否有空间
            if len(network.EQ_channel_buffer[dest_type][node_id]) < self.config.RS_OUT_CH_BUFFER:
                flit.is_arrive = True
                flit.set_position("RS_OUT_CH", self.cycle)
                network.EQ_channel_buffer[dest_type][node_id].append(flit)
                network.increment_fifo_flit_count("EQ", "CH_buffer", node_id, dest_type)

                # 释放 Entry（延迟释放，下一周期生效）
                if hasattr(flit, 'used_entry_level') and flit.used_entry_level in ("T0", "T1", "T2"):
                    # 根据 flit 的下环方向释放对应的 entry
                    eject_dir = getattr(flit, 'eject_direction', None)
                    if eject_dir == "TU":
                        network.EQ_pending_entry_release["TU"][node_id].append((flit.used_entry_level, self.cycle + 1))
                    elif eject_dir == "TD":
                        network.EQ_pending_entry_release["TD"][node_id].append((flit.used_entry_level, self.cycle + 1))

    def _network_cycle_process(self, network: Network, flits, flit_type: str):
        """v2架构：网络周期处理

        处理顺序：
        1. IQ_channel_buffer → RS.ch_buffer（本地注入）
        2. RS处理（仲裁 + 内部路由）
        3. Link传输
        4. CP下环处理（Link → RS）
        5. CP上环处理（RS → Link）
        6. RS.ch_buffer → EQ_channel_buffer（本地弹出）

        Args:
            network: 网络实例
            flits: 当前网络中的flit列表
            flit_type: flit类型 ("req" / "rsp" / "data")

        Returns:
            list: 更新后的flits列表
        """
        # 1. IQ_channel_buffer → RS.ch_buffer
        self._move_IQ_to_RS(network)

        # 2. RS处理
        network.process_ring_stations(self.cycle)

        # 3. Link传输
        flits = self._Link_process(network, flits)

        # 4. CP下环处理已在Link传输中完成（通过_handle_flit）

        # 5. CP上环处理（从RS输出）
        flits = self._CP_process(network, flits, flit_type)

        # 6. RS.ch_buffer → EQ_channel_buffer
        self._move_RS_to_EQ(network)

        return flits
