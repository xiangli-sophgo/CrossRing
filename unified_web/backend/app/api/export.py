"""
实验导出/导入API
"""

import json
import shutil
import sqlite3
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.database import ResultManager
from app.config import DATABASE_PATH

# 临时文件存储目录
TEMP_IMPORT_DIR = Path(tempfile.gettempdir()) / "crossring_import"
TEMP_IMPORT_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()

# 获取数据库管理器
db_manager = ResultManager()


@router.get("/export/experiment/info")
async def get_experiment_export_info(
    experiment_ids: str = Query(..., description="逗号分隔的实验ID列表")
):
    """
    获取导出实验的预估信息
    """
    ids = [int(id.strip()) for id in experiment_ids.split(",")]

    if not ids:
        return {"experiments_count": 0, "results_count": 0, "estimated_size": 0}

    # 统计实验和结果数量
    experiments = db_manager.list_experiments()
    selected_experiments = [e for e in experiments if e["id"] in ids]

    total_results = 0
    for exp in selected_experiments:
        results = db_manager.get_results(exp["id"])
        total_results += results["total"]

    # 估算大小：只计算 result_analysis.html 文件
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    placeholders = ",".join("?" * len(ids))
    cursor.execute(f"""
        SELECT COALESCE(SUM(LENGTH(file_content)), 0) FROM result_files
        WHERE file_name = 'result_analysis.html'
        AND result_id IN (
            SELECT id FROM kcin_results WHERE experiment_id IN ({placeholders})
            UNION
            SELECT id FROM dcin_results WHERE experiment_id IN ({placeholders})
        )
    """, ids + ids)

    html_size = cursor.fetchone()[0]
    conn.close()

    # ZIP压缩后大小估算（HTML压缩比约15%，SQLite结构数据约50%）
    estimated_size = int(html_size * 0.15 + (512 * 1024 + total_results * 1024) * 0.5)

    return {
        "experiments_count": len(selected_experiments),
        "results_count": total_results,
        "estimated_size": estimated_size,
    }


# ==================== 实验导出/导入功能 ====================

class ExperimentExportInfo(BaseModel):
    """导出实验信息"""
    id: int
    name: str
    experiment_type: str
    results_count: int
    created_at: Optional[str] = None


class ImportCheckResponse(BaseModel):
    """导入检查响应"""
    valid: bool
    error: Optional[str] = None
    package_info: Optional[Dict[str, Any]] = None
    experiments: Optional[List[Dict[str, Any]]] = None
    temp_file_id: Optional[str] = None


class ImportConfigItem(BaseModel):
    """导入配置项"""
    original_id: int
    action: str  # rename, overwrite, skip
    new_name: Optional[str] = None


class ImportExecuteRequest(BaseModel):
    """导入执行请求"""
    temp_file_id: str
    import_config: List[ImportConfigItem]


class ImportResult(BaseModel):
    """导入结果"""
    success: bool
    imported: List[Dict[str, Any]] = []
    skipped: List[int] = []
    errors: List[Dict[str, Any]] = []


def create_experiment_export_database(experiment_ids: List[int], output_path: Path) -> Dict[str, Any]:
    """
    创建只包含特定实验的新数据库（从空数据库开始，只导入需要的数据）

    返回实验信息列表用于metadata
    """
    # 连接源数据库
    src_conn = sqlite3.connect(DATABASE_PATH)
    src_conn.row_factory = sqlite3.Row
    src_cursor = src_conn.cursor()

    # 创建新的空数据库
    dst_conn = sqlite3.connect(output_path)
    dst_cursor = dst_conn.cursor()

    # 创建表结构（从源数据库复制schema）
    src_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='experiments'")
    dst_cursor.execute(src_cursor.fetchone()[0])

    src_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='kcin_results'")
    dst_cursor.execute(src_cursor.fetchone()[0])

    src_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='dcin_results'")
    dst_cursor.execute(src_cursor.fetchone()[0])

    src_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='result_files'")
    dst_cursor.execute(src_cursor.fetchone()[0])

    # 创建索引
    src_cursor.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")
    for row in src_cursor.fetchall():
        try:
            dst_cursor.execute(row[0])
        except sqlite3.OperationalError:
            pass  # 忽略已存在的索引

    dst_conn.commit()

    experiments_info = []
    placeholders = ",".join("?" * len(experiment_ids))

    # 1. 复制实验数据
    src_cursor.execute(f"SELECT * FROM experiments WHERE id IN ({placeholders})", experiment_ids)
    experiments = src_cursor.fetchall()

    for exp in experiments:
        exp_dict = dict(exp)
        columns = list(exp_dict.keys())
        values = list(exp_dict.values())
        placeholders_insert = ",".join(["?" for _ in columns])
        column_names = ",".join(columns)
        dst_cursor.execute(f"INSERT INTO experiments ({column_names}) VALUES ({placeholders_insert})", values)

        # 记录实验信息
        experiments_info.append({
            "id": exp_dict["id"],
            "name": exp_dict["name"],
            "experiment_type": exp_dict["experiment_type"],
            "created_at": exp_dict["created_at"],
            "results_count": 0  # 稍后更新
        })

    dst_conn.commit()

    # 2. 复制结果数据并收集result_id
    for exp_info in experiments_info:
        exp_id = exp_info["id"]
        exp_type = exp_info["experiment_type"]
        results_table = "kcin_results" if exp_type == "kcin" else "dcin_results"

        # 复制结果
        src_cursor.execute(f"SELECT * FROM {results_table} WHERE experiment_id = ?", (exp_id,))
        results = src_cursor.fetchall()

        result_ids = []
        for result in results:
            result_dict = dict(result)
            result_ids.append(result_dict["id"])
            columns = list(result_dict.keys())
            values = list(result_dict.values())
            placeholders_insert = ",".join(["?" for _ in columns])
            column_names = ",".join(columns)
            dst_cursor.execute(f"INSERT INTO {results_table} ({column_names}) VALUES ({placeholders_insert})", values)

        # 更新结果数量
        exp_info["results_count"] = len(results)

        # 3. 只复制 result_analysis.html 文件（每个result_id只保留一个）
        if result_ids:
            for rid in result_ids:
                src_cursor.execute("""
                    SELECT * FROM result_files
                    WHERE result_type = ? AND result_id = ?
                    AND file_name = 'result_analysis.html'
                    LIMIT 1
                """, [exp_type, rid])

                file = src_cursor.fetchone()
                if file:
                    file_dict = dict(file)
                    columns = list(file_dict.keys())
                    values = list(file_dict.values())
                    placeholders_insert = ",".join(["?" for _ in columns])
                    column_names = ",".join(columns)
                    dst_cursor.execute(f"INSERT INTO result_files ({column_names}) VALUES ({placeholders_insert})", values)

        dst_conn.commit()

    src_conn.close()
    dst_conn.close()

    return experiments_info


@router.get("/export/experiment")
async def export_experiment(
    experiment_ids: str = Query(..., description="逗号分隔的实验ID列表")
):
    """
    导出实验为ZIP包（用于在其他平台导入）

    - experiment_ids: 逗号分隔的实验ID
    """
    ids = [int(id.strip()) for id in experiment_ids.split(",")]

    if not ids:
        raise HTTPException(status_code=400, detail="请指定要导出的实验ID")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"experiment_export_{timestamp}.zip"
    zip_path = Path(temp_dir) / zip_name

    try:
        # 创建数据库副本
        db_path = Path(temp_dir) / "experiments.db"
        experiments_info = create_experiment_export_database(ids, db_path)

        if not experiments_info:
            raise HTTPException(status_code=404, detail="未找到指定的实验")

        # 创建metadata
        metadata = {
            "version": "1.0",
            "format": "crossring_experiment_export",
            "export_time": datetime.now().isoformat(),
            "source_platform": "CrossRing Simulation Platform",
            "experiments": experiments_info
        }

        metadata_path = Path(temp_dir) / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # 创建README
        exp_names = ", ".join([e["name"] for e in experiments_info])
        readme_content = f"""# CrossRing 实验导出包

导出时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
包含实验: {exp_names}
实验数量: {len(experiments_info)}

## 使用方法

在另一台电脑的 CrossRing 仿真平台中：
1. 打开实验列表页面
2. 点击"导入实验"按钮
3. 选择此ZIP文件
4. 按提示完成导入

## 包含内容

- metadata.json: 元数据信息
- data/experiments.db: SQLite数据库（含实验数据和结果文件）
"""

        # 打包
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(db_path, "data/experiments.db")
            zipf.write(metadata_path, "metadata.json")
            zipf.writestr("README.md", readme_content)

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=zip_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")


@router.post("/import/experiment/check")
async def check_import_package(
    file: UploadFile = File(..., description="导入的ZIP文件")
):
    """
    检查导入包，返回包信息和冲突检测结果
    """
    if not file.filename.endswith(".zip"):
        return ImportCheckResponse(valid=False, error="请上传ZIP格式的文件")

    # 生成临时文件ID
    temp_file_id = str(uuid.uuid4())
    temp_dir = TEMP_IMPORT_DIR / temp_file_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 保存上传的文件
        zip_path = temp_dir / "upload.zip"
        with open(zip_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 解压
        extract_dir = temp_dir / "extracted"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(extract_dir)

        # 验证结构
        metadata_path = extract_dir / "metadata.json"
        db_path = extract_dir / "data" / "experiments.db"

        if not metadata_path.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            return ImportCheckResponse(valid=False, error="无效的导入包：缺少metadata.json")

        if not db_path.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            return ImportCheckResponse(valid=False, error="无效的导入包：缺少数据库文件")

        # 读取metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # 验证格式
        if metadata.get("format") != "crossring_experiment_export":
            shutil.rmtree(temp_dir, ignore_errors=True)
            return ImportCheckResponse(valid=False, error="无效的导入包格式")

        # 获取现有实验名称
        existing_experiments = db_manager.list_experiments()
        existing_names = {e["name"]: e["id"] for e in existing_experiments}

        # 检查冲突
        experiments_with_conflict = []
        for exp in metadata.get("experiments", []):
            exp_info = {
                "id": exp["id"],
                "name": exp["name"],
                "experiment_type": exp.get("experiment_type", "kcin"),
                "results_count": exp.get("results_count", 0),
                "conflict": exp["name"] in existing_names,
                "existing_id": existing_names.get(exp["name"])
            }
            experiments_with_conflict.append(exp_info)

        return ImportCheckResponse(
            valid=True,
            package_info={
                "version": metadata.get("version"),
                "export_time": metadata.get("export_time"),
                "source_platform": metadata.get("source_platform")
            },
            experiments=experiments_with_conflict,
            temp_file_id=temp_file_id
        )

    except zipfile.BadZipFile:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return ImportCheckResponse(valid=False, error="无效的ZIP文件")
    except json.JSONDecodeError:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return ImportCheckResponse(valid=False, error="metadata.json格式错误")
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return ImportCheckResponse(valid=False, error=f"检查失败: {str(e)}")


@router.post("/import/experiment/execute")
async def execute_import(request: ImportExecuteRequest):
    """
    执行实验导入
    """
    temp_dir = TEMP_IMPORT_DIR / request.temp_file_id

    if not temp_dir.exists():
        raise HTTPException(status_code=404, detail="临时文件已过期，请重新上传")

    extract_dir = temp_dir / "extracted"
    db_path = extract_dir / "data" / "experiments.db"

    if not db_path.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=404, detail="数据库文件不存在")

    result = ImportResult(success=True)

    try:
        # 连接源数据库（只读）
        src_conn = sqlite3.connect(db_path)
        src_conn.row_factory = sqlite3.Row
        src_cursor = src_conn.cursor()

        # 连接目标数据库
        dst_conn = sqlite3.connect(DATABASE_PATH)
        dst_conn.row_factory = sqlite3.Row
        dst_cursor = dst_conn.cursor()

        # 处理每个实验
        for config in request.import_config:
            if config.action == "skip":
                result.skipped.append(config.original_id)
                continue

            try:
                # 获取源实验数据
                src_cursor.execute("SELECT * FROM experiments WHERE id = ?", (config.original_id,))
                src_exp = src_cursor.fetchone()

                if not src_exp:
                    result.errors.append({
                        "original_id": config.original_id,
                        "error": "实验不存在"
                    })
                    continue

                src_exp = dict(src_exp)
                exp_type = src_exp.get("experiment_type", "kcin")

                # 处理覆盖
                if config.action == "overwrite":
                    # 查找现有实验
                    dst_cursor.execute("SELECT id FROM experiments WHERE name = ?", (src_exp["name"],))
                    existing = dst_cursor.fetchone()
                    if existing:
                        existing_id = existing[0]
                        # 删除现有实验的所有数据
                        dst_cursor.execute("DELETE FROM result_files WHERE result_type = 'kcin' AND result_id IN (SELECT id FROM kcin_results WHERE experiment_id = ?)", (existing_id,))
                        dst_cursor.execute("DELETE FROM result_files WHERE result_type = 'dcin' AND result_id IN (SELECT id FROM dcin_results WHERE experiment_id = ?)", (existing_id,))
                        dst_cursor.execute("DELETE FROM kcin_results WHERE experiment_id = ?", (existing_id,))
                        dst_cursor.execute("DELETE FROM dcin_results WHERE experiment_id = ?", (existing_id,))
                        dst_cursor.execute("DELETE FROM experiments WHERE id = ?", (existing_id,))

                # 确定实验名称
                exp_name = config.new_name if config.action == "rename" and config.new_name else src_exp["name"]

                # 插入实验
                columns = [k for k in src_exp.keys() if k != "id"]
                values = [exp_name if k == "name" else src_exp[k] for k in columns]
                placeholders = ",".join(["?" for _ in columns])
                column_names = ",".join(columns)

                dst_cursor.execute(
                    f"INSERT INTO experiments ({column_names}) VALUES ({placeholders})",
                    values
                )
                new_exp_id = dst_cursor.lastrowid

                # 导入结果
                results_table = "kcin_results" if exp_type == "kcin" else "dcin_results"

                src_cursor.execute(f"SELECT * FROM {results_table} WHERE experiment_id = ?", (config.original_id,))
                src_results = src_cursor.fetchall()

                results_count = 0
                result_id_mapping = {}  # old_id -> new_id

                for src_result in src_results:
                    src_result = dict(src_result)
                    old_result_id = src_result["id"]

                    # 更新experiment_id
                    result_columns = [k for k in src_result.keys() if k != "id"]
                    result_values = [new_exp_id if k == "experiment_id" else src_result[k] for k in result_columns]
                    result_placeholders = ",".join(["?" for _ in result_columns])
                    result_column_names = ",".join(result_columns)

                    dst_cursor.execute(
                        f"INSERT INTO {results_table} ({result_column_names}) VALUES ({result_placeholders})",
                        result_values
                    )
                    new_result_id = dst_cursor.lastrowid
                    result_id_mapping[old_result_id] = new_result_id
                    results_count += 1

                # 导入结果文件
                for old_result_id, new_result_id in result_id_mapping.items():
                    src_cursor.execute(
                        "SELECT * FROM result_files WHERE result_type = ? AND result_id = ?",
                        (exp_type, old_result_id)
                    )
                    src_files = src_cursor.fetchall()

                    for src_file in src_files:
                        src_file = dict(src_file)
                        file_columns = [k for k in src_file.keys() if k != "id"]
                        file_values = [new_result_id if k == "result_id" else src_file[k] for k in file_columns]
                        file_placeholders = ",".join(["?" for _ in file_columns])
                        file_column_names = ",".join(file_columns)

                        dst_cursor.execute(
                            f"INSERT INTO result_files ({file_column_names}) VALUES ({file_placeholders})",
                            file_values
                        )

                dst_conn.commit()

                result.imported.append({
                    "original_id": config.original_id,
                    "new_id": new_exp_id,
                    "name": exp_name,
                    "results_count": results_count
                })

            except Exception as e:
                dst_conn.rollback()
                result.errors.append({
                    "original_id": config.original_id,
                    "error": str(e)
                })

        src_conn.close()
        dst_conn.close()

        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)

        if result.errors:
            result.success = False

        return result

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"导入失败: {str(e)}")
