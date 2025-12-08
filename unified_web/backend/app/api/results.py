"""
结果查询API
"""

import os
import re
import base64
import subprocess
import sys
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from src.database import ResultManager


def embed_images_in_html(html_content: str, base_path: str = None) -> str:
    """
    将HTML中的图片引用转换为base64嵌入

    Args:
        html_content: HTML内容
        base_path: HTML文件所在目录（用于解析相对路径）

    Returns:
        处理后的HTML内容
    """
    if not html_content:
        return html_content

    # 匹配img标签中的src属性
    img_pattern = r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*>'

    def replace_img(match):
        full_tag = match.group(0)
        src = match.group(1)

        # 跳过已经是base64或http(s)的图片
        if src.startswith('data:') or src.startswith('http://') or src.startswith('https://'):
            return full_tag

        # 尝试解析图片路径
        img_path = src
        if base_path and not os.path.isabs(src):
            img_path = os.path.join(base_path, src)

        # 如果文件存在，转换为base64
        if os.path.exists(img_path):
            try:
                with open(img_path, 'rb') as f:
                    img_data = f.read()

                # 根据扩展名确定MIME类型
                ext = os.path.splitext(img_path)[1].lower()
                mime_types = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.svg': 'image/svg+xml',
                    '.webp': 'image/webp',
                }
                mime_type = mime_types.get(ext, 'image/png')

                # 编码为base64
                b64_data = base64.b64encode(img_data).decode('utf-8')
                new_src = f'data:{mime_type};base64,{b64_data}'

                # 替换src属性
                return full_tag.replace(src, new_src)
            except Exception:
                pass

        return full_tag

    return re.sub(img_pattern, replace_img, html_content, flags=re.IGNORECASE)

router = APIRouter()

# 获取数据库管理器
db_manager = ResultManager()


# ==================== Pydantic模型 ====================

class ResultResponse(BaseModel):
    """单条结果响应"""
    id: int
    experiment_id: int
    created_at: Optional[str]
    performance: float
    config_params: Dict[str, Any]
    result_details: Dict[str, Any]
    result_html: Optional[str] = None
    result_files: Optional[List[str]] = None
    error: Optional[str] = None


class ResultsPageResponse(BaseModel):
    """分页结果响应"""
    results: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    total_pages: int


class StatisticsResponse(BaseModel):
    """统计信息响应"""
    experiment: Dict[str, Any]
    result_count: int
    performance_distribution: Dict[str, Any]
    param_keys: List[str]


class DistributionResponse(BaseModel):
    """性能分布响应"""
    min: float
    max: float
    mean: float
    count: int
    histogram: List[List[float]]


# ==================== API端点 ====================

@router.get("/experiments/{experiment_id}/results", response_model=ResultsPageResponse)
async def get_results(
    experiment_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    sort_by: str = Query("performance"),
    order: str = Query("desc", regex="^(asc|desc)$"),
):
    """
    分页获取实验结果

    - page: 页码（从1开始）
    - page_size: 每页数量（1-1000）
    - sort_by: 排序字段
    - order: 排序方向（asc/desc）
    """
    # 检查实验是否存在
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    # 获取结果
    result = db_manager.get_results(
        experiment_id=experiment_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        order=order,
    )

    return result


@router.get("/experiments/{experiment_id}/best")
async def get_best_results(
    experiment_id: int,
    limit: int = Query(10, ge=1, le=100),
):
    """获取最佳配置"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    results = db_manager.get_best_results(experiment_id, limit)
    return {"results": results, "count": len(results)}


@router.get("/experiments/{experiment_id}/stats", response_model=StatisticsResponse)
async def get_statistics(experiment_id: int):
    """获取实验统计信息"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    stats = db_manager.get_statistics(experiment_id)
    return stats


@router.get("/experiments/{experiment_id}/distribution")
async def get_distribution(
    experiment_id: int,
    bins: int = Query(50, ge=10, le=200),
):
    """获取性能分布数据"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    distribution = db_manager.db.get_performance_distribution(experiment_id, bins)
    return distribution


@router.get("/experiments/{experiment_id}/param-keys")
async def get_param_keys(experiment_id: int):
    """获取实验的配置参数键名列表（用于动态生成表格列）"""
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    param_keys = db_manager.db.get_param_keys(experiment_id)
    return {
        "param_keys": param_keys,
        "count": len(param_keys),
    }


@router.get("/experiments/{experiment_id}/traffic-stats")
async def get_traffic_stats(experiment_id: int):
    """
    按数据流文件分组统计实验结果

    返回每个数据流文件的结果数量和平均性能
    """
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    # 获取所有结果
    results, total = db_manager.db.get_results(experiment_id, page=1, page_size=100000)

    # 按数据流名称分组统计
    traffic_groups = {}
    for result in results:
        config_params = result.get("config_params", {})
        # 尝试获取数据流名称字段
        traffic_name = config_params.get("数据流名称") or config_params.get("file_name") or "未知"

        if traffic_name not in traffic_groups:
            traffic_groups[traffic_name] = {
                "count": 0,
                "performances": [],
            }

        traffic_groups[traffic_name]["count"] += 1
        performance = result.get("performance", 0)
        if performance is not None:
            traffic_groups[traffic_name]["performances"].append(performance)

    # 计算统计值
    stats = []
    for traffic_name, group in traffic_groups.items():
        performances = group["performances"]
        avg_performance = sum(performances) / len(performances) if performances else 0
        max_performance = max(performances) if performances else 0
        min_performance = min(performances) if performances else 0

        stats.append({
            "traffic_name": traffic_name,
            "count": group["count"],
            "avg_performance": avg_performance,
            "max_performance": max_performance,
            "min_performance": min_performance,
        })

    # 按结果数量降序排序
    stats.sort(key=lambda x: x["count"], reverse=True)

    return {
        "experiment_id": experiment_id,
        "total_results": total,
        "traffic_stats": stats,
    }


@router.get("/results/{result_id}/html", response_class=HTMLResponse)
async def get_result_html(result_id: int, experiment_id: int):
    """
    获取结果HTML报告内容

    - result_id: 结果ID
    - experiment_id: 实验ID（用于确定结果表）
    """
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    # 获取结果
    results_data = db_manager.get_results(
        experiment_id=experiment_id,
        page=1,
        page_size=100000,
    )

    # 查找指定result_id的结果
    target_result = None
    for result in results_data["results"]:
        if result["id"] == result_id:
            target_result = result
            break

    if not target_result:
        raise HTTPException(status_code=404, detail="结果不存在")

    result_html = target_result.get("result_html")
    if not result_html:
        raise HTTPException(status_code=404, detail="该结果没有HTML报告")

    # 尝试从result_files中获取HTML文件路径来确定base_path
    base_path = None
    result_files = target_result.get("result_files", [])
    if result_files:
        # 尝试从第一个文件推断目录
        for file_path in result_files:
            if file_path and os.path.exists(file_path):
                base_path = os.path.dirname(file_path)
                break

    # 如果还没找到base_path，尝试从HTML内容中的图片路径推断
    if not base_path:
        # 尝试找到img标签中的绝对路径
        img_match = re.search(r'<img[^>]*src=["\']([^"\']+)["\']', result_html, re.IGNORECASE)
        if img_match:
            img_src = img_match.group(1)
            if os.path.isabs(img_src) and os.path.exists(img_src):
                base_path = os.path.dirname(img_src)

    # 将图片嵌入HTML
    processed_html = embed_images_in_html(result_html, base_path)

    return HTMLResponse(content=processed_html)


class OpenFileRequest(BaseModel):
    """打开文件请求"""
    path: str


@router.post("/open-file")
async def open_local_file(request: OpenFileRequest):
    """
    打开本地文件（调用系统默认程序）

    - path: 文件路径
    """
    file_path = request.path

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")

    try:
        if sys.platform == "win32":
            os.startfile(file_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", file_path], check=True)
        else:
            subprocess.run(["xdg-open", file_path], check=True)

        return {"success": True, "message": f"已打开文件: {file_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"打开文件失败: {str(e)}")


@router.post("/open-directory")
async def open_file_directory(request: OpenFileRequest):
    """
    打开文件所在目录（在文件管理器中显示）

    - path: 文件路径
    """
    file_path = request.path
    dir_path = os.path.dirname(file_path)

    if not os.path.exists(dir_path):
        raise HTTPException(status_code=404, detail=f"目录不存在: {dir_path}")

    try:
        if sys.platform == "win32":
            # Windows: 使用explorer打开并选中文件
            if os.path.exists(file_path):
                subprocess.run(["explorer", "/select,", file_path], check=False)
            else:
                os.startfile(dir_path)
        elif sys.platform == "darwin":
            # macOS: 使用open -R显示文件
            if os.path.exists(file_path):
                subprocess.run(["open", "-R", file_path], check=True)
            else:
                subprocess.run(["open", dir_path], check=True)
        else:
            # Linux: 打开目录
            subprocess.run(["xdg-open", dir_path], check=True)

        return {"success": True, "message": f"已打开目录: {dir_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"打开目录失败: {str(e)}")


# ==================== 结果文件存储API ====================

@router.get("/results/{result_id}/files")
async def get_result_files_list(result_id: int, result_type: str = Query("kcin")):
    """
    获取结果的所有存储文件列表

    - result_id: 结果ID
    - result_type: 结果类型（kcin 或 dcin）
    """
    files = db_manager.db.get_result_files_list(result_id, result_type)
    return files


@router.delete("/results/{result_id}")
async def delete_result(result_id: int, experiment_id: int):
    """
    删除单个结果

    - result_id: 结果ID
    - experiment_id: 实验ID
    """
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    experiment_type = experiment.get("experiment_type", "kcin")
    success = db_manager.db.delete_result(result_id, experiment_type, experiment_id)

    if not success:
        raise HTTPException(status_code=404, detail="结果不存在")

    return {"success": True, "message": "结果已删除"}


class BatchDeleteRequest(BaseModel):
    """批量删除请求"""
    result_ids: List[int]


@router.post("/experiments/{experiment_id}/results/batch-delete")
async def delete_results_batch(experiment_id: int, request: BatchDeleteRequest):
    """
    批量删除结果

    - experiment_id: 实验ID
    - result_ids: 要删除的结果ID列表
    """
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    if not request.result_ids:
        raise HTTPException(status_code=400, detail="结果ID列表不能为空")

    experiment_type = experiment.get("experiment_type", "kcin")
    deleted_count = db_manager.db.delete_results_batch(
        request.result_ids, experiment_type, experiment_id
    )

    return {
        "success": True,
        "message": f"已删除 {deleted_count} 条结果",
        "deleted_count": deleted_count,
    }


@router.get("/files/{file_id}")
async def download_result_file(file_id: int):
    """
    下载存储在数据库中的结果文件

    - file_id: 文件ID
    """
    file_data = db_manager.db.get_result_file(file_id)
    if not file_data:
        raise HTTPException(status_code=404, detail="文件不存在")

    return Response(
        content=file_data["file_content"],
        media_type=file_data["mime_type"] or "application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{file_data["file_name"]}"'
        }
    )


@router.get("/files/{file_id}/view")
async def view_result_file(file_id: int):
    """
    在线查看存储在数据库中的结果文件

    - file_id: 文件ID
    """
    file_data = db_manager.db.get_result_file(file_id)
    if not file_data:
        raise HTTPException(status_code=404, detail="文件不存在")

    # 如果是HTML文件，处理图片嵌入
    mime_type = file_data["mime_type"] or "application/octet-stream"
    content = file_data["file_content"]

    if mime_type == "text/html":
        # HTML内容需要解码
        html_content = content.decode('utf-8') if isinstance(content, bytes) else content
        # 尝试嵌入图片（从同一结果的其他文件中查找）
        return HTMLResponse(content=html_content)

    return Response(
        content=content,
        media_type=mime_type,
    )


class StoreFilesRequest(BaseModel):
    """存储文件请求"""
    result_id: int
    experiment_id: int
    file_paths: List[str]


@router.post("/results/store-files")
async def store_result_files(request: StoreFilesRequest):
    """
    将本地文件存储到数据库（迁移用）

    - result_id: 结果ID
    - experiment_id: 实验ID
    - file_paths: 文件路径列表
    """
    experiment = db_manager.get_experiment(request.experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    result_type = experiment.get("experiment_type", "kcin")

    stored_count = 0
    errors = []

    for file_path in request.file_paths:
        try:
            db_manager.db.store_result_file(
                result_id=request.result_id,
                result_type=result_type,
                file_path=file_path,
            )
            stored_count += 1
        except Exception as e:
            errors.append(f"{file_path}: {str(e)}")

    return {
        "success": True,
        "stored_count": stored_count,
        "errors": errors,
    }


@router.post("/experiments/{experiment_id}/migrate-files")
async def migrate_experiment_files(experiment_id: int):
    """
    将实验的所有结果文件迁移到数据库

    遍历实验的所有结果，将 result_files 中的本地文件存储到数据库
    """
    experiment = db_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    result_type = experiment.get("experiment_type", "kcin")

    # 获取所有结果
    results_data = db_manager.get_results(
        experiment_id=experiment_id,
        page=1,
        page_size=100000,
    )

    total_files = 0
    stored_files = 0
    errors = []

    for result in results_data["results"]:
        result_files = result.get("result_files", [])
        if not result_files:
            continue

        for file_path in result_files:
            total_files += 1
            try:
                if os.path.exists(file_path):
                    db_manager.db.store_result_file(
                        result_id=result["id"],
                        result_type=result_type,
                        file_path=file_path,
                    )
                    stored_files += 1
                else:
                    errors.append(f"文件不存在: {file_path}")
            except Exception as e:
                errors.append(f"{file_path}: {str(e)}")

    return {
        "success": True,
        "experiment_id": experiment_id,
        "total_files": total_files,
        "stored_files": stored_files,
        "error_count": len(errors),
        "errors": errors[:20],  # 只返回前20个错误
    }
