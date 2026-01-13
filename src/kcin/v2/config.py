"""
KCIN v2 配置类

继承 KCINConfigBase，添加 v2 特有参数：
- RS (RingStation) 配置
"""

from src.kcin.base.config import KCINConfigBase
from typing import Dict, Any


class V2Config(KCINConfigBase):
    """KCIN v2 配置类

    v2 架构使用统一的 RingStation 组件替代 IQ/RB/EQ。
    """

    def _apply_config(self, config: Dict[str, Any]):
        """应用配置（先调用基类，再添加 v2 特有参数）"""
        # 调用基类方法
        super()._apply_config(config)

        # 强制设置版本为 v2
        self.KCIN_VERSION = "v2"

        # ==================== RingStation 配置 ====================
        self.RS_IN_CH_BUFFER = config.get("RS_IN_CH_BUFFER")
        self.RS_IN_FIFO_DEPTH = config.get("RS_IN_FIFO_DEPTH")
        self.RS_OUT_CH_BUFFER = config.get("RS_OUT_CH_BUFFER")
        self.RS_OUT_FIFO_DEPTH = config.get("RS_OUT_FIFO_DEPTH")

        # ==================== Slice 配置 ====================
        # Link slice（节点间传输）
        self.SLICE_PER_LINK_HORIZONTAL = config.get("SLICE_PER_LINK_HORIZONTAL", 1)
        self.SLICE_PER_LINK_VERTICAL = config.get("SLICE_PER_LINK_VERTICAL", 1)

        # CP slice（CrossPoint内部，独立于Link）
        # 结构：[slice_0, slice_1, ..., in_slice, out_slice]
        # 最后两个固定是in和out，增加slice时在前面增加
        self.CP_SLICE_COUNT = config.get("CP_SLICE_COUNT", 2)  # 最小为2

        # ==================== Tag 配置（v2 简化版）====================
        # v2 仍使用 ETag/ITag 但通过 RingStation 统一管理
        self.ITag_TRIGGER_Th_H = config.get("ITag_TRIGGER_Th_H", 2)
        self.ITag_TRIGGER_Th_V = config.get("ITag_TRIGGER_Th_V", 2)
        self.ITag_MAX_NUM_H = config.get("ITag_MAX_Num_H", 4)
        self.ITag_MAX_NUM_V = config.get("ITag_MAX_Num_V", 4)

        # v2 ETag 使用 RS 缓冲区深度作为约束
        self.TL_Etag_T1_UE_MAX = config.get("TL_Etag_T1_UE_MAX", self.RS_IN_FIFO_DEPTH - 1)
        self.TL_Etag_T2_UE_MAX = config.get("TL_Etag_T2_UE_MAX", self.RS_IN_FIFO_DEPTH - 2)
        self.TR_Etag_T2_UE_MAX = config.get("TR_Etag_T2_UE_MAX", self.RS_IN_FIFO_DEPTH - 2)
        self.TU_Etag_T1_UE_MAX = config.get("TU_Etag_T1_UE_MAX", self.RS_IN_FIFO_DEPTH - 1)
        self.TU_Etag_T2_UE_MAX = config.get("TU_Etag_T2_UE_MAX", self.RS_IN_FIFO_DEPTH - 2)
        self.TD_Etag_T2_UE_MAX = config.get("TD_Etag_T2_UE_MAX", self.RS_IN_FIFO_DEPTH - 2)
        self.ETAG_BOTHSIDE_UPGRADE = config.get("ETAG_BOTHSIDE_UPGRADE", False)
        self.ETAG_T1_ENABLED = config.get("ETAG_T1_ENABLED", True)
        self.ENABLE_CROSSPOINT_CONFLICT_CHECK = config.get("ENABLE_CROSSPOINT_CONFLICT_CHECK", True)

