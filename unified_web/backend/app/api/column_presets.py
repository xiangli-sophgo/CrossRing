"""
列配置方案管理 API
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import json

from app.core.logger import get_logger

logger = get_logger("column_presets")

router = APIRouter(prefix="/api/column-presets", tags=["列配置"])

# 配置文件目录
CONFIG_DIR = Path(__file__).parent.parent.parent.parent.parent / "config" / "column_presets"


class ColumnPreset(BaseModel):
    name: str
    experimentType: str
    visibleColumns: List[str]
    columnOrder: List[str]
    fixedColumns: List[str]
    createdAt: str


class PresetsFile(BaseModel):
    version: int = 1
    presets: List[ColumnPreset]


def _ensure_dir():
    """确保配置目录存在"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _get_presets_file() -> Path:
    """获取配置文件路径"""
    return CONFIG_DIR / "presets.json"


@router.get("/", response_model=PresetsFile)
async def get_presets():
    """获取所有配置方案"""
    _ensure_dir()
    presets_file = _get_presets_file()

    if not presets_file.exists():
        return PresetsFile(version=1, presets=[])

    try:
        with open(presets_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PresetsFile(**data)
    except json.JSONDecodeError as e:
        logger.error(f"配置文件解析失败: {e}")
        raise HTTPException(status_code=500, detail="配置文件解析失败")
    except Exception as e:
        logger.error(f"读取配置文件失败: {e}")
        raise HTTPException(status_code=500, detail="读取配置文件失败")


@router.post("/")
async def save_presets(data: PresetsFile):
    """保存所有配置方案"""
    _ensure_dir()
    presets_file = _get_presets_file()

    try:
        with open(presets_file, "w", encoding="utf-8") as f:
            json.dump(data.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info(f"保存了 {len(data.presets)} 个配置方案")
        return {"success": True, "message": f"已保存 {len(data.presets)} 个配置方案"}
    except PermissionError:
        logger.error(f"无权限写入配置文件: {presets_file}")
        raise HTTPException(status_code=500, detail="无权限写入配置文件")
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        raise HTTPException(status_code=500, detail="保存配置文件失败")


@router.post("/add")
async def add_preset(preset: ColumnPreset):
    """添加或更新单个配置方案"""
    _ensure_dir()
    presets_file = _get_presets_file()

    # 读取现有配置
    presets: List[ColumnPreset] = []
    if presets_file.exists():
        try:
            with open(presets_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                presets = [ColumnPreset(**p) for p in data.get("presets", [])]
        except Exception:
            pass

    # 查找是否已存在同名配置
    existing_index = next(
        (i for i, p in enumerate(presets)
         if p.name == preset.name and p.experimentType == preset.experimentType),
        None
    )

    if existing_index is not None:
        presets[existing_index] = preset
        action = "更新"
    else:
        presets.append(preset)
        action = "保存"

    # 保存
    try:
        with open(presets_file, "w", encoding="utf-8") as f:
            json.dump({
                "version": 1,
                "presets": [p.model_dump() for p in presets]
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"{action}配置方案: {preset.name}")
        return {"success": True, "message": f"配置方案「{preset.name}」已{action}"}
    except Exception as e:
        logger.error(f"保存配置失败: {e}")
        raise HTTPException(status_code=500, detail="保存配置失败")


@router.delete("/{experiment_type}/{name}")
async def delete_preset(experiment_type: str, name: str):
    """删除配置方案"""
    _ensure_dir()
    presets_file = _get_presets_file()

    if not presets_file.exists():
        raise HTTPException(status_code=404, detail="配置文件不存在")

    try:
        with open(presets_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        presets = data.get("presets", [])
        new_presets = [
            p for p in presets
            if not (p["name"] == name and p["experimentType"] == experiment_type)
        ]

        if len(new_presets) == len(presets):
            raise HTTPException(status_code=404, detail="配置方案不存在")

        with open(presets_file, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "presets": new_presets}, f, ensure_ascii=False, indent=2)

        logger.info(f"删除配置方案: {name}")
        return {"success": True, "message": f"配置方案「{name}」已删除"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除配置失败: {e}")
        raise HTTPException(status_code=500, detail="删除配置失败")
