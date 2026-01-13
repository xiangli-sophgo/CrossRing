"""
环形链表Slice类

用于建模环上的slice节点，支持统一的flit移动逻辑。

核心概念：
- Slice: 物理位置（固定座位），不移动
- Slot: 像车一样在环上循环移动，用Ring.offset表示
- Flit: 乘客，跟随Slot移动
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.utils.ring import Ring


class RingSlice:
    """环上的一个slice节点（物理位置）"""

    # Slice类型常量
    LINK = "LINK"
    CP_INTERNAL = "CP_INTERNAL"
    CP_IN = "CP_IN"
    CP_OUT = "CP_OUT"

    def __init__(self, slice_type: str, node_id: int, direction: str):
        """
        初始化RingSlice

        Args:
            slice_type: slice类型 ("LINK", "CP_INTERNAL", "CP_IN", "CP_OUT")
            node_id: 所属节点ID
            direction: 方向 ("TL", "TR", "TU", "TD")
        """
        self.flit = None  # 当前slice中的flit
        self.next = None  # 下一个slice（环形链表）
        self.slice_type = slice_type
        self.node_id = node_id
        self.direction = direction
        self.is_cp_out = False  # 是否为上环注入点（用于模式2：Link[0]共享CP_OUT）
        self.link_index = 0  # Link内的位置索引（用于显示）

        # Ring相关属性（由Ring.__init__设置）
        self.ring_index: Optional[int] = None  # 在环上的位置索引
        self.ring: Optional["Ring"] = None  # 所属的Ring对象

    @property
    def slot_id(self) -> Optional[int]:
        """
        获取当前在此位置的slot_id（动态计算）

        Returns:
            当前slot_id，如果未关联Ring则返回None
        """
        if self.ring is None or self.ring_index is None:
            return None
        return self.ring.get_slot_id_at(self.ring_index)

    def is_inject_point(self) -> bool:
        """判断是否为上环注入点"""
        return self.slice_type == self.CP_OUT or self.is_cp_out

    def is_eject_point(self) -> bool:
        """判断是否为下环检查点"""
        return self.slice_type == self.CP_IN

    def get_position_str(self) -> str:
        """获取位置字符串（用于显示）"""
        type_map = {
            self.CP_INTERNAL: "CP_INT",
            self.CP_IN: "CP_IN",
            self.CP_OUT: "CP_OUT",
            self.LINK: "LINK",
        }
        type_str = type_map.get(self.slice_type, self.slice_type)
        return f"{self.node_id}:{type_str}_{self.direction}"

    def __repr__(self):
        return f"RingSlice({self.slice_type}, node={self.node_id}, dir={self.direction}, flit={self.flit is not None})"
