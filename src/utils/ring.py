"""
Ring类 - 环形链表的统一管理

用于管理环上的Slot移动（通过offset），以及ITag预约逻辑。

核心概念：
- Slice: 物理位置（固定座位），不移动
- Slot: 像车一样在环上循环移动，用offset表示旋转状态
- Flit: 乘客，跟随Slot移动
"""

from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.ring_slice import RingSlice

from src.utils.ring_slice import RingSlice


class Ring:
    """一个完整的环（横向或纵向）"""

    def __init__(self, slices: List["RingSlice"]):
        """
        初始化Ring

        Args:
            slices: 环上所有slice的列表（已按顺序排列）
        """
        self.slices = slices
        self.offset = 0  # 旋转偏移量，表示Slot的移动状态

        # ITag预约信息：slot_id -> {reserver_node, reserver_dir}
        self.itag: Dict[int, dict] = {}

        # 待释放的ITag：slot_id -> reserver_node
        # （预约者已用其他slot上环，等这个slot回来时释放）
        self.itag_pending_release: Dict[int, int] = {}

        # 设置每个slice的环索引
        for i, s in enumerate(slices):
            s.ring_index = i
            s.ring = self  # 反向引用

    def __len__(self) -> int:
        return len(self.slices)

    def get_slot_id_at(self, position: int) -> int:
        """
        获取某个位置当前的slot_id

        Args:
            position: slice在环上的位置索引（ring_index）

        Returns:
            当前在该位置的slot_id
        """
        return (position - self.offset) % len(self.slices)

    def get_slot_id_at_slice(self, slice: "RingSlice") -> int:
        """
        获取某个slice当前的slot_id

        Args:
            slice: RingSlice对象

        Returns:
            当前在该slice位置的slot_id
        """
        return self.get_slot_id_at(slice.ring_index)

    def advance(self):
        """
        推进一个cycle，所有Slot向前移动一格

        这是O(1)操作，只更新offset而不移动任何对象
        """
        self.offset = (self.offset + 1) % len(self.slices)

    # ------------------------------------------------------------------
    # ITag相关方法
    # ------------------------------------------------------------------

    def reserve_itag(self, slot_id: int, reserver_node: int, reserver_dir: str):
        """
        预约一个slot（标记ITag）

        Args:
            slot_id: 要预约的slot_id
            reserver_node: 预约者节点ID
            reserver_dir: 预约者方向
        """
        self.itag[slot_id] = {
            "reserver_node": reserver_node,
            "reserver_dir": reserver_dir,
        }

    def check_itag(self, slot_id: int, node_id: int, direction: str) -> bool:
        """
        检查slot是否被当前节点预约

        Args:
            slot_id: 要检查的slot_id
            node_id: 当前节点ID
            direction: 当前方向

        Returns:
            True如果是当前节点预约的，False否则
        """
        if slot_id not in self.itag:
            return False
        info = self.itag[slot_id]
        return info["reserver_node"] == node_id and info["reserver_dir"] == direction

    def is_itag_reserved_by_other(
        self, slot_id: int, node_id: int, direction: str
    ) -> bool:
        """
        检查slot是否被其他节点预约

        Args:
            slot_id: 要检查的slot_id
            node_id: 当前节点ID
            direction: 当前方向

        Returns:
            True如果被其他节点预约，False否则
        """
        if slot_id not in self.itag:
            return False
        info = self.itag[slot_id]
        return not (
            info["reserver_node"] == node_id and info["reserver_dir"] == direction
        )

    def release_itag(self, slot_id: int):
        """
        释放ITag预约

        Args:
            slot_id: 要释放的slot_id
        """
        if slot_id in self.itag:
            del self.itag[slot_id]
        if slot_id in self.itag_pending_release:
            del self.itag_pending_release[slot_id]

    def mark_itag_pending_release(self, slot_id: int, node_id: int):
        """
        标记ITag待释放（预约者用其他slot上环后调用）

        Args:
            slot_id: 预约的slot_id
            node_id: 预约者节点ID
        """
        self.itag_pending_release[slot_id] = node_id

    def check_and_release_pending_itag(self, slot_id: int, node_id: int) -> bool:
        """
        检查并释放待释放的ITag

        Args:
            slot_id: 当前位置的slot_id
            node_id: 当前节点ID

        Returns:
            True如果释放了ITag，False否则
        """
        if slot_id in self.itag_pending_release:
            if self.itag_pending_release[slot_id] == node_id:
                self.release_itag(slot_id)
                return True
        return False

    def get_all_slot_ids(self) -> List[int]:
        """
        获取环上所有slot_id的列表（按当前位置顺序）

        Returns:
            slot_id列表
        """
        return [self.get_slot_id_at(i) for i in range(len(self.slices))]

    # ------------------------------------------------------------------
    # 环构建静态方法
    # ------------------------------------------------------------------

    @staticmethod
    def build_horizontal_ring(
        config,
        row: int,
        cp_in_slices: Dict,
        cp_out_slices: Dict,
    ) -> "Ring":
        """
        构建一行的横向环

        Args:
            config: 配置对象，需要有NUM_COL, CP_SLICE_COUNT, SLICE_PER_LINK_HORIZONTAL
            row: 行号
            cp_in_slices: {(node_id, direction): RingSlice} 字典，会被更新
            cp_out_slices: {(node_id, direction): RingSlice} 字典，会被更新

        Returns:
            构建好的Ring对象
        """
        num_col = config.NUM_COL
        nodes = [row * num_col + col for col in range(num_col)]
        all_slices = []

        cp_slice_count = config.CP_SLICE_COUNT
        cp_out_is_link = (cp_slice_count == 1)
        cp_internal_count = max(0, cp_slice_count - 2)
        link_slice_count = config.SLICE_PER_LINK_HORIZONTAL

        # === TR方向：从左到右 ===
        for i, node in enumerate(nodes):
            # CP内部slice
            for _ in range(cp_internal_count):
                all_slices.append(RingSlice(RingSlice.CP_INTERNAL, node, "TR"))

            # CP_IN
            in_slice = RingSlice(RingSlice.CP_IN, node, "TR")
            all_slices.append(in_slice)
            cp_in_slices[(node, "TR")] = in_slice

            # 中间节点：有Link连接到下一节点
            if i < len(nodes) - 1:
                if cp_out_is_link:
                    # CP_SLICE_COUNT=1：Link[0]同时作为CP_OUT
                    out_slice = RingSlice(RingSlice.LINK, node, "TR")
                    out_slice.is_cp_out = True
                    out_slice.link_index = 0
                    all_slices.append(out_slice)
                    cp_out_slices[(node, "TR")] = out_slice
                    for idx in range(1, link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TR")
                        s.link_index = idx
                        all_slices.append(s)
                else:
                    # CP_SLICE_COUNT>=2：独立CP_OUT + Link
                    out_slice = RingSlice(RingSlice.CP_OUT, node, "TR")
                    all_slices.append(out_slice)
                    cp_out_slices[(node, "TR")] = out_slice
                    for idx in range(link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TR")
                        s.link_index = idx
                        all_slices.append(s)
            else:
                # 边缘节点：CP_OUT连接到另一方向的CP_IN
                if not cp_out_is_link:
                    out_slice = RingSlice(RingSlice.CP_OUT, node, "TR")
                    all_slices.append(out_slice)
                    cp_out_slices[(node, "TR")] = out_slice

        # 记录TL方向起始位置（用于边缘环回）
        tl_start_index = len(all_slices)

        # === TL方向：从右到左 ===
        for i, node in enumerate(reversed(nodes)):
            # CP内部slice
            for _ in range(cp_internal_count):
                all_slices.append(RingSlice(RingSlice.CP_INTERNAL, node, "TL"))

            # CP_IN
            in_slice = RingSlice(RingSlice.CP_IN, node, "TL")
            all_slices.append(in_slice)
            cp_in_slices[(node, "TL")] = in_slice

            # 中间节点：有Link连接到下一节点
            if i < len(nodes) - 1:
                if cp_out_is_link:
                    out_slice = RingSlice(RingSlice.LINK, node, "TL")
                    out_slice.is_cp_out = True
                    out_slice.link_index = 0
                    all_slices.append(out_slice)
                    cp_out_slices[(node, "TL")] = out_slice
                    for idx in range(1, link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TL")
                        s.link_index = idx
                        all_slices.append(s)
                else:
                    out_slice = RingSlice(RingSlice.CP_OUT, node, "TL")
                    all_slices.append(out_slice)
                    cp_out_slices[(node, "TL")] = out_slice
                    for idx in range(link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TL")
                        s.link_index = idx
                        all_slices.append(s)
            else:
                # 边缘节点
                if not cp_out_is_link:
                    out_slice = RingSlice(RingSlice.CP_OUT, node, "TL")
                    all_slices.append(out_slice)
                    cp_out_slices[(node, "TL")] = out_slice

        # 连接成环形链表
        for i in range(len(all_slices)):
            all_slices[i].next = all_slices[(i + 1) % len(all_slices)]

        # 设置边缘节点的CP_OUT引用（CP_SLICE_COUNT=1时）
        right_edge_node = nodes[-1]
        left_edge_node = nodes[0]
        if cp_out_is_link:
            cp_out_slices[(right_edge_node, "TR")] = all_slices[tl_start_index]
            cp_out_slices[(left_edge_node, "TL")] = all_slices[0]

        return Ring(all_slices)

    @staticmethod
    def build_vertical_ring(
        config,
        col: int,
        cp_in_slices: Dict,
        cp_out_slices: Dict,
    ) -> "Ring":
        """
        构建一列的纵向环

        Args:
            config: 配置对象，需要有NUM_COL, NUM_ROW, CP_SLICE_COUNT, SLICE_PER_LINK_VERTICAL
            col: 列号
            cp_in_slices: {(node_id, direction): RingSlice} 字典，会被更新
            cp_out_slices: {(node_id, direction): RingSlice} 字典，会被更新

        Returns:
            构建好的Ring对象
        """
        num_col = config.NUM_COL
        num_row = config.NUM_ROW
        nodes = [row * num_col + col for row in range(num_row)]
        all_slices = []

        cp_slice_count = config.CP_SLICE_COUNT
        cp_out_is_link = (cp_slice_count == 1)
        cp_internal_count = max(0, cp_slice_count - 2)
        link_slice_count = config.SLICE_PER_LINK_VERTICAL

        # === TD方向：从上到下 ===
        for i, node in enumerate(nodes):
            # CP内部slice
            for _ in range(cp_internal_count):
                all_slices.append(RingSlice(RingSlice.CP_INTERNAL, node, "TD"))

            # CP_IN
            in_slice = RingSlice(RingSlice.CP_IN, node, "TD")
            all_slices.append(in_slice)
            cp_in_slices[(node, "TD")] = in_slice

            # 中间节点
            if i < len(nodes) - 1:
                if cp_out_is_link:
                    out_slice = RingSlice(RingSlice.LINK, node, "TD")
                    out_slice.is_cp_out = True
                    out_slice.link_index = 0
                    all_slices.append(out_slice)
                    cp_out_slices[(node, "TD")] = out_slice
                    for idx in range(1, link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TD")
                        s.link_index = idx
                        all_slices.append(s)
                else:
                    out_slice = RingSlice(RingSlice.CP_OUT, node, "TD")
                    all_slices.append(out_slice)
                    cp_out_slices[(node, "TD")] = out_slice
                    for idx in range(link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TD")
                        s.link_index = idx
                        all_slices.append(s)
            else:
                # 边缘节点
                if not cp_out_is_link:
                    out_slice = RingSlice(RingSlice.CP_OUT, node, "TD")
                    all_slices.append(out_slice)
                    cp_out_slices[(node, "TD")] = out_slice

        # 记录TU方向起始位置
        tu_start_index = len(all_slices)

        # === TU方向：从下到上 ===
        for i, node in enumerate(reversed(nodes)):
            # CP内部slice
            for _ in range(cp_internal_count):
                all_slices.append(RingSlice(RingSlice.CP_INTERNAL, node, "TU"))

            # CP_IN
            in_slice = RingSlice(RingSlice.CP_IN, node, "TU")
            all_slices.append(in_slice)
            cp_in_slices[(node, "TU")] = in_slice

            # 中间节点
            if i < len(nodes) - 1:
                if cp_out_is_link:
                    out_slice = RingSlice(RingSlice.LINK, node, "TU")
                    out_slice.is_cp_out = True
                    out_slice.link_index = 0
                    all_slices.append(out_slice)
                    cp_out_slices[(node, "TU")] = out_slice
                    for idx in range(1, link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TU")
                        s.link_index = idx
                        all_slices.append(s)
                else:
                    out_slice = RingSlice(RingSlice.CP_OUT, node, "TU")
                    all_slices.append(out_slice)
                    cp_out_slices[(node, "TU")] = out_slice
                    for idx in range(link_slice_count):
                        s = RingSlice(RingSlice.LINK, node, "TU")
                        s.link_index = idx
                        all_slices.append(s)
            else:
                # 边缘节点
                if not cp_out_is_link:
                    out_slice = RingSlice(RingSlice.CP_OUT, node, "TU")
                    all_slices.append(out_slice)
                    cp_out_slices[(node, "TU")] = out_slice

        # 连接成环形链表
        for i in range(len(all_slices)):
            all_slices[i].next = all_slices[(i + 1) % len(all_slices)]

        # 设置边缘节点的CP_OUT引用（CP_SLICE_COUNT=1时）
        bottom_edge_node = nodes[-1]
        top_edge_node = nodes[0]
        if cp_out_is_link:
            cp_out_slices[(bottom_edge_node, "TD")] = all_slices[tu_start_index]
            cp_out_slices[(top_edge_node, "TU")] = all_slices[0]

        return Ring(all_slices)
