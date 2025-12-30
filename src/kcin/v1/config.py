"""
KCIN v1 配置类

继承 KCINConfigBase，添加 v1 特有参数：
- IQ (Injection Queue) 配置
- RB (Ring Bridge) 配置
- EQ (Ejection Queue) 配置
- ETag/ITag 机制参数
"""

from src.kcin.base.config import KCINConfigBase
from typing import Dict, Any


class V1Config(KCINConfigBase):
    """KCIN v1 配置类

    v1 架构使用分离的 IQ/RB/EQ 组件。
    """

    def _apply_config(self, config: Dict[str, Any]):
        """应用配置（先调用基类，再添加 v1 特有参数）"""
        # 调用基类方法
        super()._apply_config(config)

        # 强制设置版本为 v1
        self.KCIN_VERSION = "v1"

        # ==================== IQ 配置 ====================
        self.IQ_OUT_FIFO_DEPTH_HORIZONTAL = config.get("IQ_OUT_FIFO_DEPTH_HORIZONTAL", 4)
        self.IQ_OUT_FIFO_DEPTH_VERTICAL = config.get("IQ_OUT_FIFO_DEPTH_VERTICAL", 4)
        self.IQ_OUT_FIFO_DEPTH_EQ = config.get("IQ_OUT_FIFO_DEPTH_EQ", 4)
        self.IQ_CH_FIFO_DEPTH = config.get("IQ_CH_FIFO_DEPTH", 16)

        # ==================== RB 配置 ====================
        self.RB_IN_FIFO_DEPTH = config.get("RB_IN_FIFO_DEPTH", 4)
        self.RB_OUT_FIFO_DEPTH = config.get("RB_OUT_FIFO_DEPTH", 4)

        # ==================== EQ 配置 ====================
        self.EQ_IN_FIFO_DEPTH = config.get("EQ_IN_FIFO_DEPTH", 4)
        self.EQ_CH_FIFO_DEPTH = config.get("EQ_CH_FIFO_DEPTH", 16)

        # ==================== Slice 配置 ====================
        self.SLICE_PER_LINK_HORIZONTAL = config.get("SLICE_PER_LINK_HORIZONTAL", 1)
        self.SLICE_PER_LINK_VERTICAL = config.get("SLICE_PER_LINK_VERTICAL", 1)
        self.SLICE_PER_LINK_SELF = config.get("SLICE_PER_LINK_SELF", 1)

        # ==================== ITag 配置 ====================
        self.ITag_TRIGGER_Th_H = config.get("ITag_TRIGGER_Th_H", 2)
        self.ITag_TRIGGER_Th_V = config.get("ITag_TRIGGER_Th_V", 2)
        self.ITag_MAX_NUM_H = config.get("ITag_MAX_Num_H", 4)
        self.ITag_MAX_NUM_V = config.get("ITag_MAX_Num_V", 4)

        # ==================== ETag 配置 ====================
        self.TL_Etag_T1_UE_MAX = config.get("TL_Etag_T1_UE_MAX", 3)
        self.TL_Etag_T2_UE_MAX = config.get("TL_Etag_T2_UE_MAX", 2)
        self.TR_Etag_T2_UE_MAX = config.get("TR_Etag_T2_UE_MAX", 2)
        self.TU_Etag_T1_UE_MAX = config.get("TU_Etag_T1_UE_MAX", 3)
        self.TU_Etag_T2_UE_MAX = config.get("TU_Etag_T2_UE_MAX", 2)
        self.TD_Etag_T2_UE_MAX = config.get("TD_Etag_T2_UE_MAX", 2)
        self.ETAG_BOTHSIDE_UPGRADE = config.get("ETAG_BOTHSIDE_UPGRADE", False)
        self.ETAG_T1_ENABLED = config.get("ETAG_T1_ENABLED", True)
        self.ENABLE_CROSSPOINT_CONFLICT_CHECK = config.get("ENABLE_CROSSPOINT_CONFLICT_CHECK", True)

        # ==================== ETag 约束验证 ====================
        self._validate_etag_constraints()

    def _validate_etag_constraints(self):
        """验证 ETag 参数约束"""
        assert (
            self.TL_Etag_T2_UE_MAX <= self.TL_Etag_T1_UE_MAX <= self.RB_IN_FIFO_DEPTH
            and self.TL_Etag_T2_UE_MAX <= self.RB_IN_FIFO_DEPTH - 1
            and self.TR_Etag_T2_UE_MAX <= self.RB_IN_FIFO_DEPTH
            and self.TU_Etag_T2_UE_MAX <= self.TU_Etag_T1_UE_MAX <= self.EQ_IN_FIFO_DEPTH
            and self.TU_Etag_T2_UE_MAX <= self.EQ_IN_FIFO_DEPTH - 1
            and self.TD_Etag_T2_UE_MAX <= self.EQ_IN_FIFO_DEPTH
        ), "ETag 参数约束条件不满足"
