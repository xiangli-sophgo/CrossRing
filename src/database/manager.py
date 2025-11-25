"""
仿真结果数据库 - 业务逻辑层

提供高级业务操作：CSV导入、统计分析、敏感性分析等
支持 NoC 和 D2D 两种仿真类型
"""

import csv
import json
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

from .database import DatabaseManager


class ResultManager:
    """仿真结果管理器 - 封装业务逻辑"""

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化仿真结果管理器

        Args:
            db_path: 数据库文件路径
        """
        self.db = DatabaseManager(db_path)
        self._write_lock = threading.Lock()

    # ==================== 实验管理 ====================

    def create_experiment(
        self,
        name: str,
        experiment_type: str = "noc",
        config_path: Optional[str] = None,
        topo_type: Optional[str] = None,
        traffic_files: Optional[List[str]] = None,
        traffic_weights: Optional[List[float]] = None,
        simulation_time: Optional[int] = None,
        n_repeats: int = 1,
        n_jobs: Optional[int] = None,
        total_combinations: Optional[int] = None,
        description: Optional[str] = None,
        git_commit: Optional[str] = None,
    ) -> int:
        """
        创建新实验

        Args:
            name: 实验名称（必须唯一）
            experiment_type: 实验类型 ("noc" 或 "d2d")
            config_path: 配置文件路径
            topo_type: 拓扑类型
            traffic_files: traffic文件列表
            traffic_weights: traffic权重列表
            simulation_time: 仿真时间
            n_repeats: 重复次数
            n_jobs: 并行作业数
            total_combinations: 总参数组合数
            description: 实验描述
            git_commit: Git commit hash

        Returns:
            实验ID
        """
        if experiment_type not in ("noc", "d2d"):
            raise ValueError(f"无效的实验类型: {experiment_type}，必须是 'noc' 或 'd2d'")

        # 检查名称是否已存在
        existing = self.db.get_experiment_by_name(name)
        if existing:
            raise ValueError(f"实验名称 '{name}' 已存在")

        experiment_id = self.db.create_experiment(
            name=name,
            experiment_type=experiment_type,
            config_path=config_path,
            topo_type=topo_type,
            traffic_files=json.dumps(traffic_files) if traffic_files else None,
            traffic_weights=json.dumps(traffic_weights) if traffic_weights else None,
            simulation_time=simulation_time,
            n_repeats=n_repeats,
            n_jobs=n_jobs,
            total_combinations=total_combinations,
            description=description,
            git_commit=git_commit,
            status="running",
        )
        return experiment_id

    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """获取实验详情"""
        return self.db.get_experiment(experiment_id)

    def get_experiment_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """根据名称获取实验"""
        return self.db.get_experiment_by_name(name)

    def list_experiments(
        self,
        status: Optional[str] = None,
        experiment_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取实验列表

        Args:
            status: 筛选状态
            experiment_type: 筛选类型 ("noc" 或 "d2d")
        """
        return self.db.get_all_experiments(status, experiment_type)

    def update_experiment(
        self,
        experiment_id: int,
        description: Optional[str] = None,
        status: Optional[str] = None,
        completed_combinations: Optional[int] = None,
        best_performance: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """更新实验信息"""
        kwargs = {}
        if description is not None:
            kwargs["description"] = description
        if status is not None:
            kwargs["status"] = status
        if completed_combinations is not None:
            kwargs["completed_combinations"] = completed_combinations
        if best_performance is not None:
            kwargs["best_performance"] = best_performance
        if notes is not None:
            kwargs["notes"] = notes

        if kwargs:
            return self.db.update_experiment(experiment_id, **kwargs)
        return False

    def update_experiment_status(self, experiment_id: int, status: str) -> bool:
        """更新实验状态"""
        return self.db.update_experiment(experiment_id, status=status)

    def delete_experiment(self, experiment_id: int) -> bool:
        """删除实验"""
        return self.db.delete_experiment(experiment_id)

    # ==================== 结果管理 ====================

    def add_result(
        self,
        experiment_id: int,
        config_params: Dict[str, Any],
        performance: float,
        result_details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> int:
        """
        添加单条仿真结果（线程安全）

        Args:
            experiment_id: 实验ID
            config_params: 配置参数字典
            performance: 主要性能指标
            result_details: 详细结果数据
            error: 错误信息

        Returns:
            结果ID
        """
        with self._write_lock:
            result = self.db.add_result(
                experiment_id=experiment_id,
                config_params=config_params,
                performance=performance,
                result_details=result_details,
                error=error,
            )

            # 更新实验统计
            self._update_experiment_stats(experiment_id, performance)

            return result.id

    def _update_experiment_stats(self, experiment_id: int, performance: float):
        """更新实验统计信息"""
        exp = self.db.get_experiment(experiment_id)
        if exp:
            completed = (exp.completed_combinations or 0) + 1
            best = exp.best_performance
            if best is None or performance > best:
                best = performance
            self.db.update_experiment(
                experiment_id,
                completed_combinations=completed,
                best_performance=best,
            )

    def get_results(
        self,
        experiment_id: int,
        page: int = 1,
        page_size: int = 100,
        sort_by: str = "performance",
        order: str = "desc",
    ) -> Dict[str, Any]:
        """
        分页获取结果

        Args:
            experiment_id: 实验ID
            page: 页码
            page_size: 每页数量
            sort_by: 排序字段
            order: 排序方向

        Returns:
            {results: [...], total: int, page: int, page_size: int}
        """
        results, total = self.db.get_results(
            experiment_id, page, page_size, sort_by, order
        )
        return {
            "results": results,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
        }

    def get_best_results(self, experiment_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最佳结果"""
        return self.db.get_best_results(experiment_id, limit)

    def get_statistics(self, experiment_id: int) -> Dict[str, Any]:
        """
        获取实验统计信息

        Returns:
            {
                experiment: {...},
                result_count: int,
                performance_distribution: {...},
                param_keys: [...]
            }
        """
        exp = self.db.get_experiment(experiment_id)
        if not exp:
            return {}

        distribution = self.db.get_performance_distribution(experiment_id)
        result_count = self.db.get_result_count(experiment_id)
        param_keys = self.db.get_param_keys(experiment_id)

        return {
            "experiment": exp,
            "result_count": result_count,
            "performance_distribution": distribution,
            "param_keys": param_keys,
        }

    # ==================== CSV导入 ====================

    def import_from_csv(
        self,
        csv_path: str,
        experiment_name: str,
        experiment_type: str = "noc",
        description: Optional[str] = None,
        topo_type: Optional[str] = None,
        batch_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        从CSV文件导入历史实验数据

        Args:
            csv_path: CSV文件路径
            experiment_name: 实验名称
            experiment_type: 实验类型 ("noc" 或 "d2d")
            description: 实验描述
            topo_type: 拓扑类型
            batch_size: 批量写入大小

        Returns:
            {experiment_id: int, imported_count: int, errors: [...]}
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

        # 创建实验
        experiment_id = self.create_experiment(
            name=experiment_name,
            experiment_type=experiment_type,
            description=description or f"从CSV导入: {csv_path.name}",
            topo_type=topo_type,
        )
        # 更新状态为导入中
        self.db.update_experiment(experiment_id, status="importing")

        imported_count = 0
        errors = []
        batch = []

        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)

                for row_num, row in enumerate(reader, start=2):
                    try:
                        result_dict = self._parse_csv_row(row)
                        batch.append(result_dict)

                        if len(batch) >= batch_size:
                            self._import_batch(experiment_id, batch)
                            imported_count += len(batch)
                            batch = []

                    except Exception as e:
                        errors.append(f"行 {row_num}: {str(e)}")

                # 处理剩余的批次
                if batch:
                    self._import_batch(experiment_id, batch)
                    imported_count += len(batch)

            # 更新实验状态和统计
            self.db.update_experiment(
                experiment_id,
                status="completed",
                total_combinations=imported_count,
                completed_combinations=imported_count,
            )

            # 更新最佳性能
            best_results = self.db.get_best_results(experiment_id, 1)
            if best_results:
                self.db.update_experiment(experiment_id, best_performance=best_results[0]["performance"])

        except Exception as e:
            self.db.update_experiment(experiment_id, status="failed", notes=str(e))
            raise

        return {
            "experiment_id": experiment_id,
            "imported_count": imported_count,
            "errors": errors[:100],  # 最多返回100条错误
        }

    def _parse_csv_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """解析CSV行数据"""
        # 支持多种性能字段名
        PERFORMANCE_FIELDS = [
            "performance",
            "optimization_performance",
            "带宽_DDR_混合",
            "mixed_avg_weighted_bw",
            "avg_ddr_bw",
        ]

        # 查找性能值
        performance = None
        performance_field = None
        for field in PERFORMANCE_FIELDS:
            if field in row and row[field]:
                try:
                    performance = float(row[field])
                    performance_field = field
                    break
                except (ValueError, TypeError):
                    continue

        if performance is None:
            raise ValueError(f"缺少性能字段，支持的字段: {PERFORMANCE_FIELDS}")

        # 配置参数（除了性能字段外的所有字段）
        config_params = {}
        for key, value in row.items():
            if key == performance_field:
                continue
            if value:
                try:
                    # 尝试转换为数值
                    if "." in value:
                        config_params[key] = float(value)
                    else:
                        config_params[key] = int(value)
                except ValueError:
                    # 保持字符串
                    config_params[key] = value

        return {
            "config_params": config_params,
            "performance": performance,
        }

    def _import_batch(self, experiment_id: int, batch: List[Dict]):
        """批量导入结果"""
        self.db.add_results_batch(experiment_id, batch)

    # ==================== 敏感性分析 ====================

    def get_parameter_sensitivity(
        self, experiment_id: int, parameter: str
    ) -> Dict[str, Any]:
        """
        获取单个参数的敏感性分析

        Args:
            experiment_id: 实验ID
            parameter: 参数名

        Returns:
            {
                parameter: str,
                data: [{value: any, mean_performance: float, count: int}, ...]
            }
        """
        # 获取所有结果
        results, total = self.db.get_results(experiment_id, page=1, page_size=100000)

        # 按参数值分组统计
        groups = {}
        for result in results:
            config_params = result.get("config_params")
            if config_params and parameter in config_params:
                value = config_params[parameter]
                if value not in groups:
                    groups[value] = {"performances": [], "count": 0}
                groups[value]["performances"].append(result["performance"])
                groups[value]["count"] += 1

        # 计算统计值
        data = []
        for value, stats in sorted(groups.items(), key=lambda x: x[0] if isinstance(x[0], (int, float)) else 0):
            performances = stats["performances"]
            data.append({
                "value": value,
                "mean_performance": sum(performances) / len(performances),
                "min_performance": min(performances),
                "max_performance": max(performances),
                "count": stats["count"],
            })

        return {"parameter": parameter, "data": data}

    def get_all_parameter_sensitivity(self, experiment_id: int) -> Dict[str, Any]:
        """获取所有参数的敏感性分析"""
        param_keys = self.db.get_param_keys(experiment_id)
        result = {}
        for param in param_keys:
            try:
                sensitivity = self.get_parameter_sensitivity(experiment_id, param)
                if sensitivity["data"]:
                    result[param] = sensitivity
            except Exception:
                pass
        return result

    # ==================== 实验对比 ====================

    def compare_experiments(self, experiment_ids: List[int]) -> Dict[str, Any]:
        """
        对比多个实验（仅限同类型实验）

        Args:
            experiment_ids: 实验ID列表

        Returns:
            {
                experiments: [{id, name, best_performance, ...}, ...],
                best_configs: [{experiment_id, params...}, ...]
            }
        """
        experiments = []
        best_configs = []
        experiment_types = set()

        for exp_id in experiment_ids:
            exp = self.db.get_experiment(exp_id)
            if exp:
                experiment_types.add(exp["experiment_type"])
                experiments.append(exp)

                # 获取最佳配置
                best = self.db.get_best_results(exp_id, 1)
                if best:
                    config = best[0].copy()
                    config["experiment_id"] = exp_id
                    config["experiment_name"] = exp["name"]
                    best_configs.append(config)

        # 检查是否为同类型实验
        if len(experiment_types) > 1:
            raise ValueError("只能对比同类型的实验（都是 NoC 或都是 D2D）")

        return {
            "experiments": experiments,
            "best_configs": best_configs,
        }

