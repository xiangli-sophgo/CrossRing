"""
CrossPoint组件 - 管理NoC节点的上环/下环逻辑

负责：
- E-Tag机制：分层entry管理 + T0全局队列轮询
- I-Tag预约机制：slot预约 + 回收管理
- 下环决策：基于路由策略和entry可用性
- 上环决策：基于I-Tag预约
"""

from collections import deque, defaultdict
from typing import Dict, Optional, Any, Tuple
from config.config import CrossRingConfig
from src.utils.flit import Flit


class CrossPoint:
    """
    CrossPoint组件 - 管理单个节点的上环/下环逻辑

    每个节点有2个CrossPoint实例:
    - horizontal: 管理TL/TR方向
    - vertical: 管理TU/TD方向
    """

    def __init__(self, node_id: int, direction: str, config: CrossRingConfig, network_ref=None):  # "horizontal" or "vertical"  # Network对象引用，用于访问共享数据
        self.node_id = node_id
        self.direction = direction
        self.config = config
        self.network = network_ref

        # 管理的方向
        if direction == "horizontal":
            self.managed_directions = ["TL", "TR"]
        else:  # vertical
            self.managed_directions = ["TU", "TD"]

        # E-Tag: Entry管理 (共享Network的数据结构)
        if network_ref:
            self.RB_UE_Counters = network_ref.RB_UE_Counters
            self.EQ_UE_Counters = network_ref.EQ_UE_Counters
            self.RB_CAPACITY = network_ref.RB_CAPACITY
            self.EQ_CAPACITY = network_ref.EQ_CAPACITY
            # I-Tag: 预约管理 (共享Network的数据结构)
            self.remain_tag = network_ref.remain_tag
            self.tagged_counter = network_ref.tagged_counter
            self.itag_req_counter = network_ref.itag_req_counter
            self.excess_ITag_to_remove = network_ref.excess_ITag_to_remove
        else:
            # 如果没有network_ref，创建空的（用于测试）
            self.RB_UE_Counters = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}
            self.EQ_UE_Counters = {"TU": {}, "TD": {}}
            self.RB_CAPACITY = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}
            self.EQ_CAPACITY = {"TU": {}, "TD": {}}
            self.remain_tag = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}
            self.tagged_counter = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}
            self.itag_req_counter = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}
            self.excess_ITag_to_remove = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}

    def setup_itag(self, direction: str, pos: int, max_num: int):
        """
        设置ITag配置

        Args:
            direction: 方向 ("TL"/"TR"/"TU"/"TD")
            pos: 节点位置
            max_num: 最大ITag数量
        """
        if direction in self.managed_directions:
            self.remain_tag[direction][pos] = max_num
            self.tagged_counter[direction][pos] = 0
            self.itag_req_counter[direction][pos] = 0
            self.excess_ITag_to_remove[direction][pos] = 0

    # ------------------------------------------------------------------
    # E-Tag: Entry管理方法
    # ------------------------------------------------------------------

    def _record_etag_entry(self, flit: Flit, fifo_category: str, direction: str, node_pos: int):
        """记录flit进入FIFO时的ETag等级"""
        if self.network is None:
            return

        etag = flit.ETag_priority

        # 初始化统计字典
        if node_pos not in self.network.fifo_etag_entry_count[fifo_category][direction]:
            self.network.fifo_etag_entry_count[fifo_category][direction][node_pos] = {"T0": 0, "T1": 0, "T2": 0}

        # 累加ETag等级计数
        self.network.fifo_etag_entry_count[fifo_category][direction][node_pos][etag] += 1

    def _entry_available(self, dir_type: str, key: Any, level: str) -> bool:
        """
        检查指定entry是否可用

        Args:
            dir_type: 方向类型 ("TL"/"TR"/"TU"/"TD")
            key: entry的键
            level: entry等级 ("T0"/"T1"/"T2")

        Returns:
            bool: 是否可用
        """
        if dir_type in ("TL", "TR"):
            cap = self.RB_CAPACITY[dir_type][key][level]
            occ = self.RB_UE_Counters[dir_type][key][level]
        else:
            cap = self.EQ_CAPACITY[dir_type][key][level]
            occ = self.EQ_UE_Counters[dir_type][key][level]
        return occ < cap

    def _occupy_entry(self, dir_type: str, key: Any, level: str, flit: Flit):
        """
        占用entry并记录到flit

        Args:
            dir_type: 方向类型 ("TL"/"TR"/"TU"/"TD")
            key: entry的键 (RB用(cur,next), EQ用dest_pos)
            level: entry等级 ("T0"/"T1"/"T2")
            flit: 要记录的flit对象
        """
        if dir_type in ("TL", "TR"):
            self.RB_UE_Counters[dir_type][key][level] += 1
        else:
            self.EQ_UE_Counters[dir_type][key][level] += 1
        flit.used_entry_level = level

    # ------------------------------------------------------------------
    # E-Tag: T0轮询机制
    # ------------------------------------------------------------------

    def _is_T0_slot_winner(self, flit: Flit, direction: str) -> bool:
        """
        检查flit是否是T0仲裁的winner

        从当前指针位置开始，在T0_table中找到第一个slot作为winner

        Args:
            flit: 要检查的Flit对象
            direction: 下环方向 ("TL"/"TU")

        Returns:
            bool: 是否赢得仲裁
        """
        if self.network is None:
            return False

        # 只有TL/TU方向使用T0轮询
        if direction not in ["TL", "TU"]:
            return False

        # 获取flit当前slot_id
        slot = self.network.links_tag[flit.current_link][flit.current_seat_index]
        flit_slot_id = slot.slot_id

        # 获取对应的ring_slots、T0_table和指针
        if direction == "TL":
            ring_slots = self.network.horizontal_ring_slots[self.node_id]
            T0_table = self.network.T0_table_h[self.node_id]
            pointer = self.network.T0_arb_pointer_h[self.node_id]
        else:  # TU
            ring_slots = self.network.vertical_ring_slots[self.node_id]
            T0_table = self.network.T0_table_v[self.node_id]
            pointer = self.network.T0_arb_pointer_v[self.node_id]

        # 如果T0_table为空，没有winner
        if not T0_table:
            return False

        # 从指针位置开始，找到第一个在T0_table中的slot
        n = len(ring_slots)
        for i in range(n):
            idx = (pointer + i) % n
            candidate_slot_id = ring_slots[idx]
            if candidate_slot_id in T0_table:
                # 找到了winner
                return flit_slot_id == candidate_slot_id

        return False

    def T0_table_record(self, flit: Flit, direction: str):
        """
        将slot加入T0_table（flit升级到T0时调用）

        Args:
            flit: 升级到T0的flit
            direction: 下环方向 ("TL"/"TU")
        """
        if self.network is None:
            return

        slot = self.network.links_tag[flit.current_link][flit.current_seat_index]
        slot_id = slot.slot_id

        if direction == "TL":
            self.network.T0_table_h[self.node_id].add(slot_id)
        elif direction == "TU":
            self.network.T0_table_v[self.node_id].add(slot_id)

    def T0_remove_from_table(self, flit: Flit, direction: str):
        """
        将slot从T0_table移除（T0 flit下环时调用）

        Args:
            flit: 下环的T0 flit
            direction: 下环方向 ("TL"/"TU")
        """
        if self.network is None:
            return

        slot = self.network.links_tag[flit.current_link][flit.current_seat_index]
        slot_id = slot.slot_id

        if direction == "TL":
            self.network.T0_table_h[self.node_id].discard(slot_id)
        elif direction == "TU":
            self.network.T0_table_v[self.node_id].discard(slot_id)

    def _advance_T0_arb_pointer(self, direction: str):
        """
        推进仲裁指针到下一个位置（T0 flit成功使用T0 Entry下环后调用）

        Args:
            direction: 下环方向 ("TL"/"TU")
        """
        if self.network is None:
            return

        if direction == "TL":
            ring_slots = self.network.horizontal_ring_slots[self.node_id]
            pointer = self.network.T0_arb_pointer_h[self.node_id]
            self.network.T0_arb_pointer_h[self.node_id] = (pointer + 1) % len(ring_slots)
        elif direction == "TU":
            ring_slots = self.network.vertical_ring_slots[self.node_id]
            pointer = self.network.T0_arb_pointer_v[self.node_id]
            self.network.T0_arb_pointer_v[self.node_id] = (pointer + 1) % len(ring_slots)

    # ------------------------------------------------------------------
    # E-Tag: 升级机制
    # ------------------------------------------------------------------

    def _determine_etag_upgrade(self, flit: Flit, direction: str) -> Optional[str]:
        """
        判断下环失败后的ETag升级目标

        升级规则（ETAG_T1_ENABLED=true）：
        - T2->T1: 根据ETAG_BOTHSIDE_UPGRADE配置
        - T1->T0: 只有TL/TU能升级

        升级规则（ETAG_T1_ENABLED=false）：
        - T2->T0: 直接跳过T1

        Args:
            flit: 当前flit
            direction: 下环方向

        Returns:
            str: 升级目标ETag等级 ("T0"/"T1") 或 None（不升级）
        """
        if flit.ETag_priority == "T0":
            return None

        # 优先从network读取配置，否则使用全局config
        ETAG_BOTHSIDE_UPGRADE = getattr(self.network, "ETAG_BOTHSIDE_UPGRADE", getattr(self.config, "ETAG_BOTHSIDE_UPGRADE", False))
        t1_enabled = self.config.ETAG_T1_ENABLED

        if flit.ETag_priority == "T2":
            if not t1_enabled:
                # T1禁用时：T2直接升级到T0
                if direction == "TL":
                    return "T0"
                elif direction == "TU":
                    return "T0"
                elif direction in ["TR", "TD"]:
                    # TR/TD需要额外满足双侧下环保序条件
                    eject_count = flit.eject_attempts_h if direction == "TR" else flit.eject_attempts_v
                    if eject_count > 1 and self.config.ORDERING_PRESERVATION_MODE in [2, 3] and ETAG_BOTHSIDE_UPGRADE:
                        return "T0"
                return None
            else:
                # T2 -> T1 升级
                if direction in ["TL", "TU"]:
                    return "T1"
                elif direction in ["TR", "TD"]:
                    return "T1" if ETAG_BOTHSIDE_UPGRADE else None

        elif flit.ETag_priority == "T1":
            # T1 -> T0 升级
            if direction in ["TL", "TU"]:
                return "T0"
            elif direction in ["TR", "TD"]:
                # TR/TD只有在双侧下环保序(Mode 2/3)时才能升级到T0
                return "T0" if (self.config.ORDERING_PRESERVATION_MODE in [2, 3] and ETAG_BOTHSIDE_UPGRADE) else None

        return None

    # ------------------------------------------------------------------
    # I-Tag: 预约管理
    # ------------------------------------------------------------------

    def update_excess_ITag(self):
        """处理多余ITag释放"""
        if self.network is None:
            return

        for direction in self.managed_directions:
            for node_id in list(self.excess_ITag_to_remove[direction].keys()):
                if self.excess_ITag_to_remove[direction][node_id] > 0:
                    # 寻找该节点创建的ITag并释放
                    for link, seats in self.network.links_tag.items():
                        for slot in seats:
                            if slot.itag_reserved and slot.check_itag_match(node_id, direction) and link[0] == node_id:
                                # 释放多余ITag
                                slot.clear_itag()
                                self.tagged_counter[direction][node_id] -= 1
                                self.remain_tag[direction][node_id] += 1
                                self.excess_ITag_to_remove[direction][node_id] -= 1
                                break

    # ------------------------------------------------------------------
    # 下环逻辑
    # ------------------------------------------------------------------

    def _has_available_entry_for_flit(self, flit: Flit, direction: str, key: Any) -> bool:
        """
        检查flit是否有对应等级的可用entry

        根据flit的ETag等级判断能否使用对应的entry：
        - T0: 需要T0专用/T1/T2中至少有一个可用（T0需赢得轮询仲裁）
        - T1: 需要T1/T2中至少有一个可用（T1禁用时跳过T1）
        - T2: 只能使用T2

        Args:
            flit: 当前flit
            direction: 下环方向 (TL/TR/TU/TD)
            key: entry的键

        Returns:
            bool: 是否有可用的entry
        """
        # 检查各级entry可用性
        # T0 Entry检查：TL/TU总是可以；TR/TD只有在双侧下环保序(Mode 2/3)时
        ETAG_BOTHSIDE_UPGRADE = getattr(self.network, "ETAG_BOTHSIDE_UPGRADE", getattr(self.config, "ETAG_BOTHSIDE_UPGRADE", False))
        t1_enabled = self.config.ETAG_T1_ENABLED

        if direction in ["TL", "TU"]:
            can_use_T0 = self._entry_available(direction, key, "T0")
        elif direction in ["TR", "TD"]:
            can_use_T0 = self._entry_available(direction, key, "T0") if (self.config.ORDERING_PRESERVATION_MODE in [2, 3] and ETAG_BOTHSIDE_UPGRADE) else False
        else:
            can_use_T0 = False

        can_use_T1 = self._entry_available(direction, key, "T1") if t1_enabled else False
        can_use_T2 = self._entry_available(direction, key, "T2")

        if flit.ETag_priority == "T0":
            # T0优先级：需要T0专用/T1/T2中至少有一个可用，且T0需要赢得轮询仲裁
            return (self._is_T0_slot_winner(flit, direction) and can_use_T0) or can_use_T1 or can_use_T2
        elif flit.ETag_priority == "T1":
            # T1优先级：需要T1/T2中至少有一个可用
            return can_use_T1 or can_use_T2
        elif flit.ETag_priority == "T2":
            # T2优先级：只能使用T2
            return can_use_T2

        return False

    def _occupy_best_available_entry(self, flit: Flit, direction: str, key: Any, target_node: int, link: list) -> bool:
        """
        根据ETag优先级占用最佳可用Entry

        优先级策略（ETAG_T1_ENABLED=true）：
        - T0: T0专用 → T1 → T2
        - T1: T1 → T2
        - T2: T2

        优先级策略（ETAG_T1_ENABLED=false）：
        - T0: T0专用 → T2
        - T2: T2

        Args:
            flit: 当前flit
            direction: 下环方向 (TL/TR/TU/TD)
            key: entry的键
            target_node: 目标下环节点
            link: 当前链路

        Returns:
            bool: 是否成功占用Entry
        """
        t1_enabled = self.config.ETAG_T1_ENABLED

        # 检查各级entry可用性
        can_use_T0 = self._entry_available(direction, key, "T0") if direction in ["TL", "TU"] else False
        can_use_T1 = self._entry_available(direction, key, "T1") if t1_enabled else False
        can_use_T2 = self._entry_available(direction, key, "T2")

        entry_to_use = None

        if flit.ETag_priority == "T0":
            # T0优先级：尝试T0专用 → T1(若启用) → T2
            if self._is_T0_slot_winner(flit, direction) and can_use_T0:
                entry_to_use = "T0"
            elif can_use_T1:
                entry_to_use = "T1"
            elif can_use_T2:
                entry_to_use = "T2"
        elif flit.ETag_priority == "T1":
            # T1优先级：尝试T1 → T2
            if can_use_T1:
                entry_to_use = "T1"
            elif can_use_T2:
                entry_to_use = "T2"
        elif flit.ETag_priority == "T2":
            # T2优先级：只使用T2
            if can_use_T2:
                entry_to_use = "T2"

        if entry_to_use:
            self._complete_eject(flit, direction, target_node, link, key, entry_to_use)
            return True

        return False

    def _complete_eject(self, flit: Flit, direction: str, target_node: int, link: list, key: Any, entry_level: str):
        """
        完成下环操作：设置flit状态、占用entry、更新跟踪表

        Args:
            flit: 当前flit
            direction: 下环方向 (TL/TR/TU/TD)
            target_node: 目标下环节点
            link: 当前链路
            key: entry的键（ring_bridge用节点号，eject_queues用节点号）
            entry_level: 占用的entry等级 ("T0"/"T1"/"T2")
        """
        # 1. T0相关处理（必须在清空link位置之前，因为需要访问slot_id）
        if flit.ETag_priority == "T0" and direction in ["TL", "TU"]:
            # 无论使用什么Entry，都从T0_table移除
            self.T0_remove_from_table(flit, direction)
            # 只有使用T0 Entry时才推进指针
            if entry_level == "T0":
                self._advance_T0_arb_pointer(direction)

        # 2. 清空当前link位置
        link[flit.current_seat_index] = None

        # 3. 根据下环目标设置flit状态并添加到pre缓冲
        if direction in ["TL", "TR"]:
            # 横向环下环到ring_bridge
            flit.is_delay = False
            flit.current_link = None
            flit.current_seat_index = -1
            flit.set_position(f"RB_{direction}", self.network.cycle)

            # 添加到ring_bridge_pre缓冲位（带检查）
            if self.network:
                if self.network.ring_bridge_pre[direction][key] is not None:
                    raise RuntimeError(f"[Cycle {self.network.cycle}] ring_bridge_pre[{direction}][{key}] 已被占用！" f"当前flit: {self.network.ring_bridge_pre[direction][key]}, " f"尝试添加: {flit}")
                self.network.ring_bridge_pre[direction][key] = flit
                # 记录ETag入队统计
                self._record_etag_entry(flit, "RB", direction, key)
        else:  # TU, TD
            # 纵向环下环到eject_queues（最终目的地节点，但还未到IP）
            flit.is_delay = False
            flit.is_arrive = False  # 还未到IP模块，保持False
            flit.current_link = None
            flit.current_seat_index = 0
            flit.set_position(f"EQ_{direction}", self.network.cycle)

            # 添加到eject_queues_in_pre缓冲位（带检查）
            if self.network:
                if self.network.eject_queues_in_pre[direction][key] is not None:
                    raise RuntimeError(
                        f"[Cycle {self.network.cycle}] eject_queues_in_pre[{direction}][{key}] 已被占用！" f"当前flit: {self.network.eject_queues_in_pre[direction][key]}, " f"尝试添加: {flit}"
                    )
                self.network.eject_queues_in_pre[direction][key] = flit
                # 记录ETag入队统计
                self._record_etag_entry(flit, "EQ", direction, key)

        # 4. 占用Entry
        self._occupy_entry(direction, key, entry_level, flit)

        # 5. 更新保序跟踪表 - 调用network的方法
        if self.network and hasattr(self.network, "_update_order_tracking_table"):
            self.network._update_order_tracking_table(flit, target_node, direction)

    def _try_eject(self, flit: Flit, direction: str, target_node: int, link: list, ring_bridge: dict = None, eject_queues: dict = None) -> tuple:
        """
        尝试下环（适用于所有下环场景）

        检查顺序：
        1. 队列容量
        2. 对应等级的entry可用性
        3. 占用最佳Entry

        注意：保序检查（方向和order_id）由调用方在外层完成

        Args:
            flit: 当前flit
            direction: 下环方向 (TL/TR/TU/TD)
            target_node: 目标下环节点
            link: 当前链路
            ring_bridge: ring_bridge队列字典
            eject_queues: eject_queues队列字典

        Returns:
            tuple: (是否成功: bool, 失败原因: str)
                失败原因: "" (成功), "capacity" (容量), "entry" (Entry不足)
        """
        # 1. 获取队列和容量限制
        if direction in ["TL", "TR"]:
            # 横向环下环到ring_bridge
            if ring_bridge is None:
                return False, "capacity"
            current_node = flit.current_link[1]
            key = current_node  # 新架构: ring_bridge键直接使用节点号
            queue = ring_bridge[direction][key]
            capacity = self.config.RB_IN_FIFO_DEPTH
        else:  # TU, TD
            # 纵向环下环到eject_queues
            if eject_queues is None:
                return False, "capacity"
            key = flit.current_link[1]
            queue = eject_queues[direction][key]
            capacity = self.config.EQ_IN_FIFO_DEPTH

        # 2. 检查队列容量（第二优先级）
        if len(queue) >= capacity:
            return False, "capacity"

        # 3. 检查是否有对应等级的entry可用（第三优先级）
        if not self._has_available_entry_for_flit(flit, direction, key):
            return False, "entry"

        # 4. 尝试占用最佳Entry
        success = self._occupy_best_available_entry(flit, direction, key, target_node, link)
        return (True, "") if success else (False, "entry")

    # ------------------------------------------------------------------
    # 维度判断和下环决策方法（借鉴C2C设计）
    # ------------------------------------------------------------------

    def _needs_dimension_move(self, current_node: int, next_node: int, dimension: str) -> bool:
        """
        判断从当前节点到下一节点是否需要指定维度移动

        Args:
            current_node: 当前节点ID
            next_node: 下一节点ID
            dimension: 维度类型 ("vertical"或"horizontal")

        Returns:
            bool: 是否需要该维度的移动
        """
        num_col = self.config.NUM_COL

        if dimension == "vertical":
            # 垂直移动：行号不同
            return (current_node // num_col) != (next_node // num_col)
        else:  # horizontal
            # 水平移动：列号不同
            return (current_node % num_col) != (next_node % num_col)

    def _needs_vertical_move(self, current_node: int, next_node: int) -> bool:
        """判断从当前节点到下一节点是否需要垂直移动"""
        return self._needs_dimension_move(current_node, next_node, "vertical")

    def _needs_horizontal_move(self, current_node: int, next_node: int) -> bool:
        """判断从当前节点到下一节点是否需要水平移动"""
        return self._needs_dimension_move(current_node, next_node, "horizontal")

    def should_eject_flit(self, flit: Flit, current_node: int) -> tuple:
        """
        基于路径信息的下环决策逻辑（借鉴C2C设计）

        使用flit的path动态查找当前位置，判断是否需要下环

        Args:
            flit: 要判断的flit
            current_node: 当前所在节点

        Returns:
            tuple: (是否下环: bool, 下环目标: str, 下环方向: str)
                   下环目标: "RB"(Ring Bridge) / "EQ"(Eject Queue) / ""
                   下环方向: "TL"/"TR"/"TU"/"TD" / ""
        """
        if not flit.path or len(flit.path) == 0:
            return False, "", ""

        final_dest = flit.path[-1]

        # 根据当前crosspoint维度过滤不匹配的链路方向，避免错误下环
        if hasattr(flit, "current_link") and flit.current_link is not None and len(flit.current_link) >= 2:
            u, v = flit.current_link[:2]
            hop_diff = abs(u - v)
            is_self_loop = u == v  # 自环

            if self.direction == "horizontal":
                # 横向环：处理横向链路或横向自环
                if not is_self_loop and (hop_diff != 1 or (u // self.config.NUM_COL) != (v // self.config.NUM_COL)):
                    return False, "", ""
                # 如果是自环，检查是否为横向自环
                if is_self_loop and len(flit.current_link) == 3 and flit.current_link[2] != "h":
                    return False, "", ""
            elif self.direction == "vertical":
                # 纵向环：处理纵向链路或纵向自环
                if not is_self_loop and (hop_diff != self.config.NUM_COL or (u % self.config.NUM_COL) != (v % self.config.NUM_COL)):
                    return False, "", ""
                # 如果是自环，检查是否为纵向自环
                if is_self_loop and len(flit.current_link) == 3 and flit.current_link[2] != "v":
                    return False, "", ""

        # 1. 检查是否到达最终目标
        if current_node == final_dest:
            # 根据当前CrossPoint类型决定下环目标
            if self.direction == "horizontal":
                # 水平CrossPoint: 需要通过Ring Bridge转到垂直环
                return True, "RB", self._get_eject_direction_horizontal(flit, current_node, final_dest)
            else:  # vertical
                # 垂直CrossPoint: 直接下环到目的地
                return True, "EQ", self._get_eject_direction_vertical(flit, current_node, final_dest)

        # 2. 查找当前节点在路径中的位置
        try:
            path_index = flit.path.index(current_node)
            if path_index < len(flit.path) - 1:
                next_node = flit.path[path_index + 1]

                # 3. 判断下一跳是否需要维度转换
                if self.direction == "horizontal":
                    # 水平CrossPoint: 检查下一跳是否需要垂直移动
                    if self._needs_vertical_move(current_node, next_node):
                        # 需要从横向环转到纵向环,通过Ring Bridge下环
                        direction = self._get_eject_direction_horizontal(flit, current_node, next_node)
                        return True, "RB", direction
                elif self.direction == "vertical":
                    # 垂直CrossPoint: 检查下一跳是否需要水平移动
                    if self._needs_horizontal_move(current_node, next_node):
                        # 需要从纵向环转到横向环,通过Ring Bridge下环
                        direction = self._get_eject_direction_vertical(flit, current_node, next_node)
                        return True, "RB", direction
        except ValueError:
            # 当前节点不在路径中,可能是绕环情况
            # 检查是否绕环到达了目标节点
            if current_node == final_dest:
                if self.direction == "horizontal":
                    return True, "RB", self._get_eject_direction_horizontal(flit, current_node, final_dest)
                else:
                    return True, "EQ", self._get_eject_direction_vertical(flit, current_node, final_dest)

        # 4. 不需要下环,继续在当前环传输
        return False, "", ""

    def _get_eject_direction_horizontal(self, flit: Flit, current_node: int, target_node: int) -> str:
        """
        获取水平CrossPoint的下环方向

        根据flit的实际移动方向确定应该从哪个方向下环

        Args:
            flit: 当前flit对象
            current_node: 当前节点ID
            target_node: 目标节点ID

        Returns:
            str: "TL"或"TR"
        """
        prev_node = flit.current_link[0]
        curr_node = flit.current_link[1]

        # 自环：根据节点所在边缘判断流动方向
        if prev_node == curr_node:
            col = curr_node % self.config.NUM_COL
            # 左边缘 -> 向右流动 -> TR
            # 右边缘 -> 向左流动 -> TL
            return "TR" if col == 0 else "TL"

        # 非自环：比较列号判断移动方向
        prev_col = prev_node % self.config.NUM_COL
        curr_col = curr_node % self.config.NUM_COL

        if curr_col > prev_col or (prev_col == self.config.NUM_COL - 1 and curr_col == 0):
            # 向右移动 -> 从右侧下环
            return "TR"
        else:
            # 向左移动 -> 从左侧下环
            return "TL"

    def _get_eject_direction_vertical(self, flit: Flit, current_node: int, target_node: int) -> str:
        """
        获取垂直CrossPoint的下环方向

        根据flit的实际移动方向确定应该从哪个方向下环

        Args:
            flit: 当前flit对象
            current_node: 当前节点ID
            target_node: 目标节点ID

        Returns:
            str: "TU"或"TD"
        """
        prev_node = flit.current_link[0]
        curr_node = flit.current_link[1]

        # 自环：根据节点所在边缘判断流动方向
        if prev_node == curr_node:
            row = curr_node // self.config.NUM_COL
            num_rows = self.config.NUM_NODE // self.config.NUM_COL
            # 上边缘 -> 向下流动 -> TD
            # 下边缘 -> 向上流动 -> TU
            return "TD" if row == 0 else "TU"

        # 非自环：比较行号判断移动方向
        prev_row = prev_node // self.config.NUM_COL
        curr_row = curr_node // self.config.NUM_COL

        if curr_row > prev_row:
            # 向下移动 -> 从下侧下环
            return "TD"
        else:
            # 向上移动 -> 从上侧下环
            return "TU"

    # ------------------------------------------------------------------
    # 拥塞感知流控 - 智能综合策略 (v3.0)
    # ------------------------------------------------------------------

    def _get_entry_usage(self, node_pos: int, direction: str) -> float:
        """
        获取指定节点和方向的Entry使用率

        Args:
            node_pos: 节点位置
            direction: 方向 ("TL"/"TR"/"TU"/"TD")

        Returns:
            float: Entry使用率 (0.0 ~ 1.0)
        """
        total_used = 0
        total_capacity = 0

        if direction in ["TL", "TR"]:
            if node_pos in self.RB_UE_Counters[direction]:
                levels = self.RB_UE_Counters[direction][node_pos]
                capacity = self.RB_CAPACITY[direction][node_pos]
                for level in ["T0", "T1", "T2"]:
                    total_used += levels[level]
                    total_capacity += capacity[level]
        else:
            if node_pos in self.EQ_UE_Counters[direction]:
                levels = self.EQ_UE_Counters[direction][node_pos]
                capacity = self.EQ_CAPACITY[direction][node_pos]
                for level in ["T0", "T1", "T2"]:
                    total_used += levels[level]
                    total_capacity += capacity[level]

        return total_used / total_capacity if total_capacity > 0 else 0.0

    # ------------------------------------------------------------------
    # 上环逻辑 - 统一的注入管理
    # ------------------------------------------------------------------

    def _can_inject_to_link(self, flit: Flit, link: tuple, direction: str, cycle: int) -> bool:
        """
        检查flit是否可以上环到指定link（统一的I-Tag检查逻辑）

        检查流程：
        0. 检查CrossPoint冲突（根据ENABLE_CROSSPOINT_CONFLICT_CHECK配置）
        1. 检查link是否被占用
        2. 如果占用：
           - 检查是否已有I-Tag预约
           - 如果没有预约且等待时间达到阈值，尝试创建I-Tag
        3. 如果未占用：
           - 检查是否有I-Tag预约
           - 如果没有预约，可以直接注入
           - 如果有预约且是自己的预约，可以使用预约注入

        Args:
            flit: 要注入的flit
            link: 目标link (current, next_node)
            direction: 注入方向 ("TL"/"TR"/"TU"/"TD")
            cycle: 当前周期

        Returns:
            bool: 是否可以注入
        """
        if self.network is None:
            return False

        current_pos = link[0]

        # CrossPoint冲突检查
        dim = "horizontal" if direction in ["TL", "TR"] else "vertical"
        conflict_status = self.network.crosspoint_conflict[dim][current_pos][direction]
        if self.config.ENABLE_CROSSPOINT_CONFLICT_CHECK == 1:
            # 严格模式：当前或前一周期有冲突都不能上环
            if conflict_status[0] or conflict_status[1]:
                return False
        else:
            # 默认模式：当前周期有冲突不能上环
            if conflict_status[0]:
                return False

        link_occupied = self.network.links[link][0] is not None
        slot = self.network.links_tag[link][0]

        # 区分横向和纵向配置
        is_horizontal = direction in ["TL", "TR"]
        wait_cycle = flit.wait_cycle_h if is_horizontal else flit.wait_cycle_v
        trigger_threshold = self.config.ITag_TRIGGER_Th_H if is_horizontal else self.config.ITag_TRIGGER_Th_V
        max_itag = self.config.ITag_MAX_NUM_H if is_horizontal else self.config.ITag_MAX_NUM_V

        if link_occupied:
            # Link被占用，无论是否有ITag都不能注入
            # 但可以在没有预约时尝试创建ITag（为下次空闲时预约）
            if not slot.itag_reserved:
                # 没有预约，尝试创建I-Tag
                if (
                    wait_cycle > trigger_threshold
                    and self.tagged_counter[direction][current_pos] < max_itag
                    and self.remain_tag[direction][current_pos] > 0
                    and self.itag_req_counter[direction][current_pos] == 0
                ):
                    # 创建I-Tag预约
                    self.remain_tag[direction][current_pos] -= 1
                    self.tagged_counter[direction][current_pos] += 1
                    slot.reserve_itag(current_pos, direction)
                    if is_horizontal:
                        flit.itag_h = True
                    else:
                        flit.itag_v = True
            # 无论如何都不能注入到被占用的slice
            return False
        else:
            # Link未占用
            if not slot.itag_reserved:
                # 没有预约，可以上环
                return True
            elif slot.check_itag_match(current_pos, direction):
                # 有预约且是自己的预约，使用预约
                return True
            else:
                # 有预约但不是自己的
                return False

    def _inject_flit_to_link(self, flit: Flit, link: tuple, direction: str, cycle: int) -> bool:
        """
        统一的上环执行方法

        执行流程：
        1. 注入flit到link的第0个slice
        2. 释放I-Tag预约（如果使用了预约）
        3. 重置E-Tag显示优先级为T2
        4. 更新统计计数器

        Args:
            flit: 要注入的flit
            link: 目标link
            direction: 注入方向
            cycle: 当前周期

        Returns:
            bool: 是否注入成功
        """
        if self.network is None:
            return False

        current_pos = link[0]
        slot = self.network.links_tag[link][0]
        is_horizontal = direction in ["TL", "TR"]

        # 1. 注入到link（直接赋值）
        self.network.links[link][0] = flit
        flit.current_link = link
        flit.current_seat_index = 0
        flit.set_position("Link", cycle)

        # 2. 处理I-Tag释放
        if slot.itag_reserved and slot.check_itag_match(current_pos, direction):
            # 使用了预约，释放
            slot.clear_itag()
            self.remain_tag[direction][current_pos] += 1
            self.tagged_counter[direction][current_pos] -= 1

        # 清除flit上的I-Tag标记
        if is_horizontal and flit.itag_h:
            flit.itag_h = False
        elif not is_horizontal and flit.itag_v:
            flit.itag_v = False

        # 3. 重置E-Tag显示优先级为T2
        flit.ETag_priority = "T2"

        # 4. 更新统计（network层会处理）
        return True

    def _increment_wait_cycles(self, queue: "deque", direction: str):
        """
        统一的等待时间管理方法

        为队列中所有flit增加等待时间

        Args:
            queue: 注入队列（deque）
            direction: 方向 ("TL"/"TR"/"TU"/"TD")
        """
        is_horizontal = direction in ["TL", "TR"]

        for flit in queue:
            if flit:
                if is_horizontal:
                    flit.wait_cycle_h += 1
                else:
                    flit.wait_cycle_v += 1

    def process_inject(self, node_pos: int, queue: "deque", direction: str, cycle: int) -> Optional[Flit]:
        """
        统一的注入处理（支持TL/TR/TU/TD四个方向）

        Args:
            node_pos: 节点位置
            queue: inject_queues（TL/TR）或ring_bridge（TU/TD）队列
            direction: 注入方向 ("TL"/"TR"/"TU"/"TD")
            cycle: 当前周期

        Returns:
            成功注入的flit，失败返回None
        """
        if not queue or not queue[0]:
            return None

        if direction not in self.managed_directions:
            return None

        flit = queue[0]  # 先peek

        # 计算目标link（根据方向不同采用不同策略）
        if direction in ["TL", "TR"]:
            # 横向注入：根据direction计算横向邻居
            num_col = self.config.NUM_COL
            col = node_pos % num_col

            if direction == "TR":
                # TR方向：向右
                if col == num_col - 1:
                    # 右边界：自环
                    link = (node_pos, node_pos, "h")
                else:
                    link = (node_pos, node_pos + 1)
            else:  # TL
                # TL方向：向左
                if col == 0:
                    # 左边界：自环
                    link = (node_pos, node_pos, "h")
                else:
                    link = (node_pos, node_pos - 1)
        else:  # TU/TD
            # 纵向注入：根据方向计算垂直邻居
            num_col = self.config.NUM_COL
            num_row = self.config.NUM_ROW
            row = node_pos // num_col

            if direction == "TU":
                # TU方向：向上
                if row == 0:
                    # 上边界：自环
                    link = (node_pos, node_pos, "v")
                else:
                    link = (node_pos, node_pos - num_col)
            else:  # TD
                # TD方向：向下
                if row == num_row - 1:
                    # 下边界：自环
                    link = (node_pos, node_pos, "v")
                else:
                    link = (node_pos, node_pos + num_col)

        # 检查是否可以注入
        if self._can_inject_to_link(flit, link, direction, cycle):
            # 可以注入，从队列取出
            flit = queue.popleft()

            # 执行注入
            success = self._inject_flit_to_link(flit, link, direction, cycle)

            if success:
                return flit
            else:
                # 注入失败，放回队列
                queue.appendleft(flit)
                return None
        else:
            # 不能注入，更新等待时间
            self._increment_wait_cycles(queue, direction)
            return None
