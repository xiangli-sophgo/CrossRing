"""
网络数据流处理相关的Mixin类
包含IQ/Link/RB/EQ/CP处理、Flit移动、Tag处理等功能
"""

from src.noc.components import Network, IPInterface
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
            if queue_pre[node_id] and len(queue[node_id]) < self.config.IQ_CH_FIFO_DEPTH:
                flit = queue_pre[node_id]
                flit.set_position("IQ_CH", self.cycle)
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
                fifo_depth = self.config.IQ_OUT_FIFO_DEPTH_HORIZONTAL
            elif direction in ["TU", "TD"]:
                fifo_depth = self.config.IQ_OUT_FIFO_DEPTH_VERTICAL
            else:  # EQ
                fifo_depth = self.config.IQ_OUT_FIFO_DEPTH_EQ

            if queue_pre[node_id] and len(queue[node_id]) < fifo_depth:
                flit = queue_pre[node_id]
                flit.departure_inject_cycle = self.cycle
                flit.set_position(f"IQ_{direction}", self.cycle)
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
            if queue_pre[node_id] and len(queue[node_id]) < self.config.RB_IN_FIFO_DEPTH:
                flit = queue_pre[node_id]
                flit.set_position(f"RB_{direction}", self.cycle)
                queue[node_id].append(flit)
                network.increment_fifo_flit_count("RB", direction, node_id)
                queue_pre[node_id] = None

        # RB_OUT_PRE → RB_OUT
        for fifo_pos in ("EQ", "TU", "TD"):
            queue_pre = network.ring_bridge_pre[fifo_pos]
            queue = network.ring_bridge[fifo_pos]
            if queue_pre[node_id] and len(queue[node_id]) < self.config.RB_OUT_FIFO_DEPTH:
                flit = queue_pre[node_id]
                flit.is_arrive = fifo_pos == "EQ"
                flit.set_position(f"RB_{fifo_pos}", self.cycle)
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
            if queue_pre[node_id] and len(queue[node_id]) < self.config.EQ_IN_FIFO_DEPTH:
                flit = queue_pre[node_id]
                flit.is_arrive = fifo_pos == "EQ"
                flit.set_position(f"EQ_{fifo_pos}", self.cycle)
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
            if queue_pre[node_id] and len(queue[node_id]) < self.config.EQ_CH_FIFO_DEPTH:
                flit = queue_pre[node_id]
                flit.set_position("EQ_CH", self.cycle)
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

    def _move_IQ_channel_buffer_to_arbiter_input(self, network: Network):
        """IQ模块：将IQ_channel_buffer移动到仲裁输入FIFO的pre缓冲

        从IQ_channel_buffer移动到IQ_arbiter_input_fifo_pre

        Args:
            network: 网络实例 (req_network / rsp_network / data_network)
        """
        for node_id in range(self.config.NUM_NODE):
            for ip_type in network.IQ_channel_buffer.keys():
                # 检查源FIFO是否有数据
                if not network.IQ_channel_buffer[ip_type][node_id]:
                    continue

                # 检查目标pre缓冲是否为空
                if network.IQ_arbiter_input_fifo_pre[ip_type][node_id] is not None:
                    continue

                # 检查目标FIFO是否有空间
                if len(network.IQ_arbiter_input_fifo[ip_type][node_id]) >= 2:
                    continue

                # 移动flit到pre缓冲
                flit = network.IQ_channel_buffer[ip_type][node_id].popleft()
                network.IQ_arbiter_input_fifo_pre[ip_type][node_id] = flit

    def _IQ_process(self, network: Network, flit_type: str):
        """IQ模块：IQ仲裁处理

        从IQ_arbiter_input_fifo移动到inject_queues_pre

        Args:
            network: 网络实例 (req_network / rsp_network / data_network)
            flit_type: flit类型 ("req" / "rsp" / "data")
        """
        # 所有节点都可以作为IP节点
        for node_id in range(self.config.NUM_NODE):
            # 1. 收集所有可能的 ip_types 和 directions
            all_ip_types = set()
            for direction in self.IQ_directions:
                rr_queue = network.round_robin["IQ"][direction][node_id]
                all_ip_types.update(rr_queue)

            if not all_ip_types:
                continue

            ip_types_list = sorted(list(all_ip_types))
            directions_list = list(self.IQ_directions)

            # 2. 构建请求矩阵和权重矩阵 (ip_types × directions)
            request_matrix = []
            weight_matrix = []
            ip_type_to_flit = {}  # 缓存每个ip_type的flit

            # 获取仲裁器的权重策略
            weight_strategy = getattr(self.iq_arbiter, "weight_strategy", "uniform")

            for ip_type in ip_types_list:
                req_row = []
                weight_row = []
                for direction in directions_list:
                    # 检查是否可以注入到这个方向
                    can_inject = self._check_iq_injection_conditions(network, node_id, ip_type, direction, flit_type, ip_type_to_flit)
                    req_row.append(can_inject)

                    # 计算权重
                    if can_inject and weight_strategy != "uniform":
                        flit = ip_type_to_flit.get((ip_type, direction))
                        if flit:
                            is_horizontal = direction in ["TL", "TR"]
                            wait_time = flit.wait_cycle_h if is_horizontal else flit.wait_cycle_v
                            queue_length = len(network.IQ_arbiter_input_fifo[ip_type][node_id])

                            if weight_strategy == "wait_time":
                                weight_row.append(float(wait_time))
                            elif weight_strategy == "queue_length":
                                weight_row.append(float(queue_length))
                            elif weight_strategy == "hybrid":
                                weight_row.append(queue_length * 0.7 + wait_time * 0.3)
                            else:
                                weight_row.append(1.0)
                        else:
                            weight_row.append(0.0)
                    else:
                        weight_row.append(1.0 if can_inject else 0.0)

                request_matrix.append(req_row)
                weight_matrix.append(weight_row)

            # 3. 执行匹配
            if not any(any(row) for row in request_matrix):
                continue  # 没有有效请求

            queue_id = f"IQ_pos{node_id}_{flit_type}"
            matches = self.iq_arbiter.match(request_matrix, weight_matrix=weight_matrix, queue_id=queue_id)

            # 4. 根据匹配结果处理注入
            for ip_idx, dir_idx in matches:
                ip_type = ip_types_list[ip_idx]
                direction = directions_list[dir_idx]

                flit = ip_type_to_flit.get((ip_type, direction))
                if not flit:
                    continue

                # 标记反方向上环（横向）
                if self.config.REVERSE_DIRECTION_ENABLED and direction in ["TL", "TR"]:
                    normal_direction = None
                    if len(flit.path) > 1:
                        diff = flit.path[1] - flit.path[0]
                        if diff == 1:
                            normal_direction = "TR"
                        elif diff == -1:
                            normal_direction = "TL"
                    if normal_direction and direction != normal_direction:
                        flit.reverse_inject_h = 1

                # 执行注入
                if flit_type == "req":
                    counts = None
                    if not ip_type.startswith("d2d_rn"):
                        counts = self.dma_rw_counts[ip_type][node_id]
                    else:
                        counts = self.dma_rw_counts.get(ip_type, {}).get(node_id, {"read": 0, "write": 0})

                    self._try_inject_to_direction(flit, ip_type, node_id, direction, counts)
                else:
                    # rsp / data 网络：直接移动到 pre‑缓冲
                    # 设置flit的允许下环方向（仅在第一次注入时设置）
                    if not hasattr(flit, "allowed_eject_directions") or flit.allowed_eject_directions is None:
                        flit.allowed_eject_directions = network.determine_allowed_eject_directions(flit)

                    network.IQ_arbiter_input_fifo[ip_type][node_id].popleft()
                    queue_pre = network.inject_queues_pre[direction]
                    queue_pre[node_id] = flit

                    if flit_type == "rsp":
                        flit.rsp_entry_network_cycle = self.cycle
                    elif flit_type == "data":
                        req = self.req_network.send_flits[flit.packet_id][0]
                        flit.sync_latency_record(req)
                        self.send_flits_num += 1
                        self.trans_flits_num += 1

                        if hasattr(req, "req_type"):
                            if req.req_type == "read":
                                self.send_read_flits_num_stat += 1
                            elif req.req_type == "write":
                                self.send_write_flits_num_stat += 1

                        if hasattr(flit, "traffic_id"):
                            self.traffic_scheduler.update_traffic_stats(flit.traffic_id, "sent_flit")

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

    def _RB_process(self, network: Network):
        """RB模块：Ring Bridge仲裁处理

        使用多对多匹配的Ring Bridge仲裁（全局最优）

        Args:
            network: 网络实例
        """
        # 遍历所有节点作为Ring Bridge
        for pos in range(self.config.NUM_NODE):
            # 新架构: Ring Bridge在本节点,键直接使用节点号
            next_pos = pos  # 保留用于兼容性（queue_id等使用）

            # 1. 获取各输入槽位的flit
            station_flits = [
                network.ring_bridge["TL"][pos][0] if network.ring_bridge["TL"][pos] else None,
                network.ring_bridge["TR"][pos][0] if network.ring_bridge["TR"][pos] else None,
                network.inject_queues["TU"][pos][0] if pos in network.inject_queues["TU"] and network.inject_queues["TU"][pos] else None,
                network.inject_queues["TD"][pos][0] if pos in network.inject_queues["TD"] and network.inject_queues["TD"][pos] else None,
            ]

            if not any(station_flits):
                continue

            # 2. 构建请求矩阵和权重矩阵 (input_slots × output_directions)
            # input_slots: 0=TL, 1=TR, 2=TU, 3=TD
            # output_directions: 0=EQ, 1=TU, 2=TD
            slot_names = ["TL", "TR", "TU", "TD"]
            output_dirs = ["EQ", "TU", "TD"]
            direction_conditions = {"EQ": lambda d, n: d == n, "TU": lambda d, n: d < n, "TD": lambda d, n: d > n}

            # 获取仲裁器的权重策略
            weight_strategy = getattr(self.rb_arbiter, "weight_strategy", "uniform")

            request_matrix = []
            weight_matrix = []
            for slot_idx, flit in enumerate(station_flits):
                req_row = []
                weight_row = []
                for out_dir in output_dirs:
                    # 检查是否可以从这个slot转发到这个输出方向
                    can_forward = self._check_rb_forward_conditions(network, flit, pos, next_pos, out_dir, direction_conditions[out_dir])

                    # 反方向流控检查（纵向TU/TD）
                    if self.config.REVERSE_DIRECTION_ENABLED and flit and out_dir in ["TU", "TD"]:
                        dest = flit.destination
                        # 获取正常方向
                        if dest < pos:
                            normal_direction = "TU"
                        elif dest > pos:
                            normal_direction = "TD"
                        else:
                            normal_direction = None  # 目的地就是当前节点，走EQ

                        # 如果当前检查的方向是反方向
                        if normal_direction and out_dir != normal_direction:
                            reverse_direction = out_dir
                            normal_depth = len(network.ring_bridge[normal_direction][pos])
                            reverse_depth = len(network.ring_bridge[reverse_direction][pos])
                            capacity = self.config.RB_OUT_FIFO_DEPTH

                            # 只有当正常方向比反方向拥塞程度超过容量×阈值时，才允许走反方向
                            if (normal_depth - reverse_depth) > capacity * self.config.REVERSE_DIRECTION_THRESHOLD:
                                can_forward = True  # 正常方向比反方向拥塞很多，允许走反方向
                            else:
                                can_forward = False  # 差距不够大，不走反方向

                    req_row.append(can_forward)

                    # 计算权重
                    if can_forward and weight_strategy != "uniform" and flit:
                        # RB的输入slot方向: TL/TR是横向，TU/TD是纵向
                        slot_name = slot_names[slot_idx]
                        is_horizontal = slot_name in ["TL", "TR"]
                        wait_time = flit.wait_cycle_h if is_horizontal else flit.wait_cycle_v

                        if weight_strategy == "wait_time":
                            weight_row.append(float(wait_time))
                        elif weight_strategy == "queue_length":
                            weight_row.append(1.0)  # 每个slot只有一个flit
                        elif weight_strategy == "hybrid":
                            weight_row.append(float(wait_time))  # hybrid时也使用等待时间
                        else:
                            weight_row.append(1.0)
                    else:
                        weight_row.append(1.0 if can_forward else 0.0)

                request_matrix.append(req_row)
                weight_matrix.append(weight_row)

            # 3. 执行匹配
            if not any(any(row) for row in request_matrix):
                continue

            queue_id = f"RB_pos{pos}_{next_pos}"
            matches = self.rb_arbiter.match(request_matrix, weight_matrix=weight_matrix, queue_id=queue_id)

            # 4. 根据匹配结果处理转发
            for slot_idx, out_dir_idx in matches:
                out_dir = output_dirs[out_dir_idx]
                flit = station_flits[slot_idx]

                if flit:
                    # 标记反方向上环（纵向）
                    if self.config.REVERSE_DIRECTION_ENABLED and out_dir in ["TU", "TD"]:
                        dest = flit.destination
                        if dest < pos:
                            normal_direction = "TU"
                        elif dest > pos:
                            normal_direction = "TD"
                        else:
                            normal_direction = None
                        if normal_direction and out_dir != normal_direction:
                            flit.reverse_inject_v = 1

                    # 新架构: ring_bridge_pre键直接使用节点号
                    network.ring_bridge_pre[out_dir][pos] = flit
                    station_flits[slot_idx] = None  # 标记为已使用
                    self._update_ring_bridge(network, pos, out_dir, slot_idx)

    def _EQ_process(self, network: Network, flit_type: str):
        """EQ模块：Eject Queue仲裁处理

        处理eject的仲裁逻辑，根据flit类型处理不同的eject队列

        Args:
            network: 网络实例
            flit_type: flit类型 ("req" / "rsp" / "data")
        """
        # 遍历所有节点处理eject_queues和ring_bridge
        for node_id in range(self.config.NUM_NODE):
            # 从仲裁输入FIFO构造eject_flits
            eject_flits = [
                network.EQ_arbiter_input_fifo["TU"][node_id][0] if network.EQ_arbiter_input_fifo["TU"][node_id] else None,
                network.EQ_arbiter_input_fifo["TD"][node_id][0] if network.EQ_arbiter_input_fifo["TD"][node_id] else None,
                network.EQ_arbiter_input_fifo["IQ"][node_id][0] if network.EQ_arbiter_input_fifo["IQ"][node_id] else None,
                network.EQ_arbiter_input_fifo["RB"][node_id][0] if network.EQ_arbiter_input_fifo["RB"][node_id] else None,
            ]
            if not any(eject_flits):
                continue
            self._move_to_eject_queues_pre(network, eject_flits, node_id)

    def _CP_process(self, network: Network, flits, flit_type: str):
        """CP模块：CrossPoint处理

        处理CrossPoint的上环逻辑

        Args:
            network: 网络实例
            flits: 当前网络中的flit列表
            flit_type: flit类型 ("req" / "rsp" / "data")

        Returns:
            list: 更新后的flits列表
        """
        # CrossPoint注入（TL/TR/TU/TD四个方向）
        for direction in ["TL", "TR", "TU", "TD"]:
            # 获取对应的队列（数据结构都是dict[node_pos] -> deque）
            if direction in ["TL", "TR"]:
                queues = network.inject_queues[direction]
            else:  # TU/TD
                queues = network.ring_bridge[direction]

            num, injected_flits = self.process_inject_queues(network, queues, direction)

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

        # 4. 更新ITag和CrossPoint状态
        network.update_excess_ITag()
        network.update_cross_point()

        return flits

    def _network_cycle_process(self, network: Network, flits, flit_type: str):
        """网络周期处理：协调各模块完成一个完整的网络周期

        处理顺序：
        1. IQ_channel_buffer → IQ仲裁输入FIFO
        2. IQ仲裁（从仲裁输入FIFO读取）
        3. Link传输
        4. RB仲裁
        5. EQ输入端口 → EQ仲裁输入FIFO
        6. EQ仲裁（从仲裁输入FIFO读取）
        7. CP处理

        Args:
            network: 网络实例
            flits: 当前网络中的flit列表
            flit_type: flit类型 ("req" / "rsp" / "data")

        Returns:
            list: 更新后的flits列表
        """
        # 1a. 移动IQ_channel_buffer到仲裁输入FIFO
        self._move_IQ_channel_buffer_to_arbiter_input(network)

        # 1b. IQ模块：IQ仲裁（从仲裁输入FIFO读取）
        self._IQ_process(network, flit_type)

        # 2. Link模块：Link传输
        flits = self._Link_process(network, flits)

        # 3. RB模块：Ring Bridge仲裁
        self._RB_process(network)

        # 4a. 移动EQ输入端口到仲裁输入FIFO
        self._move_eject_queues_to_arbiter_input(network)

        # 4b. EQ模块：Eject Queue仲裁（从仲裁输入FIFO读取）
        self._EQ_process(network, flit_type)

        # 5. CP模块：CrossPoint处理（包含上环和下环）
        flits = self._CP_process(network, flits, flit_type)

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
            fifo_depth = self.config.IQ_OUT_FIFO_DEPTH_HORIZONTAL
        elif direction in ["TU", "TD"]:
            fifo_depth = self.config.IQ_OUT_FIFO_DEPTH_VERTICAL
        else:  # EQ
            fifo_depth = self.config.IQ_OUT_FIFO_DEPTH_EQ

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
                capacity = self.config.IQ_OUT_FIFO_DEPTH_HORIZONTAL

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

    def _check_rb_forward_conditions(self, network, flit, pos, next_pos, out_dir, cmp_func):
        """
        检查是否可以从slot转发到输出方向

        Args:
            network: 网络实例
            flit: 待转发的flit
            pos: 当前位置
            next_pos: 下一个位置
            out_dir: 输出方向 ("EQ", "TU", "TD")
            cmp_func: 目的地比较函数

        Returns:
            bool: 是否可以转发
        """
        if not flit:
            return False

        # 检查输出FIFO是否满
        # 新架构: ring_bridge键直接使用节点号
        if len(network.ring_bridge[out_dir][pos]) >= self.config.RB_OUT_FIFO_DEPTH:
            return False

        # 基于路径的下一跳判断，避免在XY路由的横向阶段提前下竖向环
        next_hop = self._get_next_hop_for_node(flit, pos)

        if out_dir == "EQ":
            final_dest = flit.destination
            return final_dest == pos

        if next_hop is None:
            return False

        # 只允许在确实需要向上/向下移动时进入TU/TD
        diff = next_hop - pos
        if out_dir == "TU":
            return diff < 0 and diff % self.config.NUM_COL == 0
        if out_dir == "TD":
            return diff > 0 and diff % self.config.NUM_COL == 0

        return False

    def _update_ring_bridge(self, network: Network, pos, direction, index):
        """更新transfer stations

        新架构: ring_bridge键直接使用节点号pos，next_pos参数保留用于兼容性
        TU/TD方向由CrossPoint处理，flit作为参数传入，不再从队列pop
        """
        if index == 0:
            flit = network.ring_bridge["TL"][pos].popleft()
            # TL方向：延迟释放Entry，记录到pending列表
            if flit.used_entry_level in ("T0", "T1", "T2"):
                network.RB_pending_entry_release["TL"][pos].append((flit.used_entry_level, self.cycle + 1))
        elif index == 1:
            flit = network.ring_bridge["TR"][pos].popleft()
            # TR方向：延迟释放Entry，记录到pending列表
            if flit.used_entry_level in ("T1", "T2"):
                network.RB_pending_entry_release["TR"][pos].append((flit.used_entry_level, self.cycle + 1))
        elif index == 2:
            flit = network.inject_queues["TU"][pos].popleft()
        elif index == 3:
            flit = network.inject_queues["TD"][pos].popleft()

        # 获取通道类型
        channel_type = getattr(flit, "flit_type", "req")  # 默认为req

        # 更新RB总数据量统计（所有经过的flit，无论ETag等级）
        if direction != "EQ":
            if pos in self.RB_total_flits_per_node:
                self.RB_total_flits_per_node[pos][direction] += 1

            # 更新按通道分类的RB总数据量统计
            if pos in self.RB_total_flits_per_channel.get(channel_type, {}):
                self.RB_total_flits_per_channel[channel_type][pos][direction] += 1

            if flit.ETag_priority == "T1":
                self.RB_ETag_T1_num_stat += 1
                # Update per-node FIFO statistics
                if pos in self.RB_ETag_T1_per_node_fifo:
                    self.RB_ETag_T1_per_node_fifo[pos][direction] += 1

                # Update per-channel statistics
                if pos in self.RB_ETag_T1_per_channel.get(channel_type, {}):
                    self.RB_ETag_T1_per_channel[channel_type][pos][direction] += 1

            elif flit.ETag_priority == "T0":
                self.RB_ETag_T0_num_stat += 1
                # Update per-node FIFO statistics
                if pos in self.RB_ETag_T0_per_node_fifo:
                    self.RB_ETag_T0_per_node_fifo[pos][direction] += 1

                # Update per-channel statistics
                if pos in self.RB_ETag_T0_per_channel.get(channel_type, {}):
                    self.RB_ETag_T0_per_channel[channel_type][pos][direction] += 1

        flit.ETag_priority = "T2"

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

    def _move_eject_queues_to_arbiter_input(self, network: Network):
        """EQ模块：将4个输入端口的数据移动到仲裁输入FIFO的pre缓冲

        从[TU, TD, IQ, RB]输入端口移动到EQ_arbiter_input_fifo_pre
        同时从原始队列弹出，并更新相关计数器

        Args:
            network: 网络实例 (req_network / rsp_network / data_network)
        """
        for node_id in range(self.config.NUM_NODE):
            # 定义4个输入端口的来源
            input_ports_info = [
                ("TU", network.eject_queues["TU"][node_id]),
                ("TD", network.eject_queues["TD"][node_id]),
                ("IQ", network.inject_queues["EQ"][node_id]),
                ("RB", network.ring_bridge["EQ"][node_id]),
            ]

            for port_name, source_fifo in input_ports_info:
                # 检查源FIFO是否有数据
                if not source_fifo:
                    continue

                # 检查目标pre缓冲是否为空
                if network.EQ_arbiter_input_fifo_pre[port_name][node_id] is not None:
                    continue

                # 检查目标FIFO是否有空间
                if len(network.EQ_arbiter_input_fifo[port_name][node_id]) >= 2:
                    continue

                # 从源FIFO弹出flit并移动到pre缓冲
                flit = source_fifo.popleft()
                network.EQ_arbiter_input_fifo_pre[port_name][node_id] = flit

                # 对于TU/TD端口，延迟释放Entry
                if port_name == "TU":
                    if flit.used_entry_level in ("T0", "T1", "T2"):
                        network.EQ_pending_entry_release["TU"][node_id].append((flit.used_entry_level, self.cycle + 1))
                elif port_name == "TD":
                    if flit.used_entry_level in ("T1", "T2"):
                        network.EQ_pending_entry_release["TD"][node_id].append((flit.used_entry_level, self.cycle + 1))

    def _move_to_eject_queues_pre(self, network: Network, eject_flits, node_id):
        """
        使用多对多匹配的EQ仲裁（全局最优）

        构建 input_ports × ip_types 的请求矩阵，
        使用匹配算法进行全局优化，确保每个端口只弹出到一个IP类型。
        """
        # 1. 收集所有IP类型
        ip_types_list = list(network.EQ_channel_buffer.keys())
        if not ip_types_list:
            return

        # 2. 构建请求矩阵和权重矩阵 (input_ports × ip_types)
        # input_ports: 0=TU, 1=TD, 2=IQ, 3=RB
        port_names = ["TU", "TD", "IQ", "RB"]

        # 获取仲裁器的权重策略
        weight_strategy = getattr(self.eq_arbiter, "weight_strategy", "uniform")

        request_matrix = []
        weight_matrix = []

        for port_idx, flit in enumerate(eject_flits):
            req_row = []
            weight_row = []
            for ip_type in ip_types_list:
                # 检查是否可以从这个端口弹出到这个IP类型
                can_eject = self._check_eq_eject_conditions(network, flit, node_id, port_idx, ip_type)
                req_row.append(can_eject)

                # 计算权重
                if can_eject and weight_strategy != "uniform" and flit:
                    # EQ的输入端口方向: TU/TD是纵向，IQ/RB来源混合
                    port_name = port_names[port_idx]
                    is_horizontal = port_name in ["IQ", "RB"]  # IQ/RB端口视为横向
                    wait_time = flit.wait_cycle_h if is_horizontal else flit.wait_cycle_v
                    queue_length = len(network.EQ_arbiter_input_fifo[port_name][node_id])

                    if weight_strategy == "wait_time":
                        weight_row.append(float(wait_time))
                    elif weight_strategy == "queue_length":
                        weight_row.append(float(queue_length))
                    elif weight_strategy == "hybrid":
                        weight_row.append(queue_length * 0.7 + wait_time * 0.3)
                    else:
                        weight_row.append(1.0)
                else:
                    weight_row.append(1.0 if can_eject else 0.0)

            request_matrix.append(req_row)
            weight_matrix.append(weight_row)

        # 3. 执行匹配
        if not any(any(row) for row in request_matrix):
            return

        queue_id = f"EQ_pos{node_id}"
        matches = self.eq_arbiter.match(request_matrix, weight_matrix=weight_matrix, queue_id=queue_id)

        # 4. 根据匹配结果处理弹出
        for port_idx, ip_type_idx in matches:
            ip_type = ip_types_list[ip_type_idx]
            flit = eject_flits[port_idx]

            if flit:
                network.EQ_channel_buffer_pre[ip_type][node_id] = flit
                flit.is_arrive = True
                flit.arrival_eject_cycle = self.cycle
                eject_flits[port_idx] = None

                # 从仲裁输入FIFO中移除flit（UE计数器已在_move_eject_queues_to_arbiter_input中更新）
                port_names = ["TU", "TD", "IQ", "RB"]
                port_name = port_names[port_idx]
                removed_flit = network.EQ_arbiter_input_fifo[port_name][node_id].popleft()

                # 获取通道类型
                flit_channel_type = getattr(flit, "flit_type", "req")

                # 更新总数据量统计
                if node_id in self.EQ_total_flits_per_node:
                    if port_idx == 0:  # TU direction
                        self.EQ_total_flits_per_node[node_id]["TU"] += 1
                    elif port_idx == 1:  # TD direction
                        self.EQ_total_flits_per_node[node_id]["TD"] += 1

                # 更新按通道分类的总数据量统计
                if node_id in self.EQ_total_flits_per_channel.get(flit_channel_type, {}):
                    if port_idx == 0:  # TU direction
                        self.EQ_total_flits_per_channel[flit_channel_type][node_id]["TU"] += 1
                    elif port_idx == 1:  # TD direction
                        self.EQ_total_flits_per_channel[flit_channel_type][node_id]["TD"] += 1

                if flit.ETag_priority == "T1":
                    self.EQ_ETag_T1_num_stat += 1
                    # Update per-node FIFO statistics (only for TU and TD directions)
                    if node_id in self.EQ_ETag_T1_per_node_fifo:
                        if port_idx == 0:  # TU direction
                            self.EQ_ETag_T1_per_node_fifo[node_id]["TU"] += 1
                        elif port_idx == 1:  # TD direction
                            self.EQ_ETag_T1_per_node_fifo[node_id]["TD"] += 1

                    # Update per-channel statistics
                    if node_id in self.EQ_ETag_T1_per_channel.get(flit_channel_type, {}):
                        if port_idx == 0:  # TU direction
                            self.EQ_ETag_T1_per_channel[flit_channel_type][node_id]["TU"] += 1
                        elif port_idx == 1:  # TD direction
                            self.EQ_ETag_T1_per_channel[flit_channel_type][node_id]["TD"] += 1

                elif flit.ETag_priority == "T0":
                    self.EQ_ETag_T0_num_stat += 1
                    # Update per-node FIFO statistics (only for TU and TD directions)
                    if node_id in self.EQ_ETag_T0_per_node_fifo:
                        if port_idx == 0:  # TU direction
                            self.EQ_ETag_T0_per_node_fifo[node_id]["TU"] += 1
                        elif port_idx == 1:  # TD direction
                            self.EQ_ETag_T0_per_node_fifo[node_id]["TD"] += 1

                    # Update per-channel statistics
                    if node_id in self.EQ_ETag_T0_per_channel.get(flit_channel_type, {}):
                        if port_idx == 0:  # TU direction
                            self.EQ_ETag_T0_per_channel[flit_channel_type][node_id]["TU"] += 1
                        elif port_idx == 1:  # TD direction
                            self.EQ_ETag_T0_per_channel[flit_channel_type][node_id]["TD"] += 1

                flit.ETag_priority = "T2"

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

    def _check_eq_eject_conditions(self, network, flit, node_id, port_idx, ip_type):
        """
        检查是否可以从端口弹出到IP类型

        Args:
            network: 网络实例
            flit: 待弹出的flit
            node_id: 节点ID
            port_idx: 端口索引 (0=TU, 1=TD, 2=IQ, 3=RB)
            ip_type: IP类型

        Returns:
            bool: 是否可以弹出
        """
        if flit is None:
            return False

        # 检查目的地类型是否匹配
        if flit.destination_type != ip_type:
            return False

        # 检查EQ channel buffer是否满
        if len(network.EQ_channel_buffer[ip_type][node_id]) >= network.config.EQ_CH_FIFO_DEPTH:
            return False

        return True

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
