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
from .flit import Flit


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
            self.T0_Etag_Order_FIFO = network_ref.T0_Etag_Order_FIFO
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
            self.T0_Etag_Order_FIFO = None
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

    def _register_T0_slot(self, flit: Flit) -> int:
        """
        为Flit注册T0 Slot到轮询队列

        当Flit从T1升级到T0时调用此方法

        Args:
            flit: 要注册的Flit对象

        Returns:
            int: 分配的slot_id
        """
        if self.T0_Etag_Order_FIFO is None:
            raise RuntimeError("T0_Etag_Order_FIFO未初始化")

        # 获取当前Slot的slot_id
        slot = self.network.links_tag[flit.current_link][flit.current_seat_index]
        slot_id = slot.slot_id

        # 注册到轮询队列
        self.T0_Etag_Order_FIFO.append(slot_id)

        # 记录在flit上
        flit.T0_slot_id = slot_id

        return slot_id

    def _unregister_T0_slot(self, flit: Flit):
        """
        从轮询队列中注销T0 Slot

        当Flit成功下环或被移除时调用

        Args:
            flit: 要注销的Flit对象
        """
        if self.T0_Etag_Order_FIFO is None:
            return

        if not hasattr(flit, "T0_slot_id") or flit.T0_slot_id is None:
            return

        # 从队列中移除
        try:
            self.T0_Etag_Order_FIFO.remove(flit.T0_slot_id)
        except ValueError:
            pass

        # 清除flit上的标记
        flit.T0_slot_id = None

    def _is_T0_slot_winner(self, flit: Flit) -> bool:
        """
        检查Flit是否赢得T0轮询仲裁

        Args:
            flit: 要检查的Flit对象

        Returns:
            bool: 是否赢得仲裁
        """
        if self.T0_Etag_Order_FIFO is None:
            return False

        if not hasattr(flit, "T0_slot_id") or flit.T0_slot_id is None:
            return False

        if not self.T0_Etag_Order_FIFO:
            return False

        return self.T0_Etag_Order_FIFO[0] == flit.T0_slot_id

    # ------------------------------------------------------------------
    # E-Tag: 升级机制
    # ------------------------------------------------------------------

    def _determine_etag_upgrade(self, flit: Flit, direction: str) -> Optional[str]:
        """
        判断下环失败后的ETag升级目标

        升级规则：
        - T2->T1: 根据ETAG_BOTHSIDE_UPGRADE配置
        - T1->T0: 只有TL/TU能升级

        Args:
            flit: 当前flit
            direction: 下环方向

        Returns:
            str: 升级目标ETag等级 ("T0"/"T1") 或 None（不升级）
        """
        if flit.ETag_priority == "T0":
            return None

        ETag_BOTHSIDE_UPGRADE = getattr(self.config, "ETag_BOTHSIDE_UPGRADE", False)

        if flit.ETag_priority == "T2":
            # T2 -> T1 升级
            if direction in ["TL", "TU"]:
                return "T1"
            elif direction in ["TR", "TD"]:
                return "T1" if ETag_BOTHSIDE_UPGRADE else None

        elif flit.ETag_priority == "T1":
            # T1 -> T0 升级（只有TL/TU能升级）
            if direction in ["TL", "TU"]:
                return "T0"

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
        - T1: 需要T1/T2中至少有一个可用
        - T2: 只能使用T2

        Args:
            flit: 当前flit
            direction: 下环方向 (TL/TR/TU/TD)
            key: entry的键

        Returns:
            bool: 是否有可用的entry
        """
        # 检查各级entry可用性
        can_use_T0 = self._entry_available(direction, key, "T0") if direction in ["TL", "TU"] else False
        can_use_T1 = self._entry_available(direction, key, "T1")
        can_use_T2 = self._entry_available(direction, key, "T2")

        if flit.ETag_priority == "T0":
            # T0优先级：需要T0专用/T1/T2中至少有一个可用，且T0需要赢得轮询仲裁
            return (self._is_T0_slot_winner(flit) and can_use_T0) or can_use_T1 or can_use_T2
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

        优先级策略：
        - T0: T0专用 → T1 → T2
        - T1: T1 → T2
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
        # 检查各级entry可用性
        can_use_T0 = self._entry_available(direction, key, "T0") if direction in ["TL", "TU"] else False
        can_use_T1 = self._entry_available(direction, key, "T1")
        can_use_T2 = self._entry_available(direction, key, "T2")

        entry_to_use = None

        if flit.ETag_priority == "T0":
            # T0优先级：尝试T0专用 → T1 → T2
            if self._is_T0_slot_winner(flit) and can_use_T0:
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
            key: entry的键（ring_bridge用tuple，eject_queues用int）
            entry_level: 占用的entry等级 ("T0"/"T1"/"T2")
        """
        # 1. 取消T0注册（如果需要）
        if flit.ETag_priority == "T0":
            self._unregister_T0_slot(flit)

        # 2. 设置flit状态
        flit.is_delay = False
        link[flit.current_seat_index] = None

        if direction in ["TL", "TR"]:
            # 横向环下环
            flit.current_link = (flit.current_link[1], target_node)
            flit.current_seat_index = 0 if direction == "TL" else 1
        else:  # TU, TD
            # 纵向环下环到最终目的地
            flit.is_arrive = True
            flit.current_seat_index = 0

        # 3. 占用Entry
        self._occupy_entry(direction, key, entry_level, flit)

        # 4. 更新保序跟踪表 - 调用network的方法
        if self.network and hasattr(self.network, "_update_order_tracking_table"):
            self.network._update_order_tracking_table(flit, target_node, direction)

    def _try_eject(self, flit: Flit, direction: str, target_node: int, link: list, ring_bridge: dict = None, eject_queues: dict = None, can_eject_in_order_func=None) -> bool:
        """
        尝试下环（适用于所有下环场景）

        检查顺序：
        1. 保序条件
        2. 队列容量
        3. 对应等级的entry可用性
        4. 占用最佳Entry

        Args:
            flit: 当前flit
            direction: 下环方向 (TL/TR/TU/TD)
            target_node: 目标下环节点
            link: 当前链路
            ring_bridge: ring_bridge队列字典
            eject_queues: eject_queues队列字典
            can_eject_in_order_func: 保序检查函数

        Returns:
            bool: 是否成功下环
        """
        # 1. 获取队列和容量限制
        if direction in ["TL", "TR"]:
            # 横向环下环到ring_bridge
            if ring_bridge is None:
                return False
            key = (flit.current_link[1], target_node)
            queue = ring_bridge[direction][key]
            capacity = self.config.RB_IN_FIFO_DEPTH
        else:  # TU, TD
            # 纵向环下环到eject_queues
            if eject_queues is None:
                return False
            key = flit.current_link[1]
            queue = eject_queues[direction][key]
            capacity = self.config.EQ_IN_FIFO_DEPTH

        # 2. 检查保序条件（第一优先级）
        if can_eject_in_order_func and not can_eject_in_order_func(flit, target_node, direction):
            # 保序检查失败
            if direction in ["TL", "TR"]:
                flit.ordering_blocked_eject_h += 1
            else:
                flit.ordering_blocked_eject_v += 1
            return False

        # 3. 检查队列容量（第二优先级）
        if len(queue) >= capacity:
            return False

        # 4. 检查是否有对应等级的entry可用（第三优先级）
        if not self._has_available_entry_for_flit(flit, direction, key):
            return False

        # 5. 尝试占用最佳Entry
        return self._occupy_best_available_entry(flit, direction, key, target_node, link)
