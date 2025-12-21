"""
仿真结果数据库 - 业务逻辑层

提供高级业务操作：CSV导入、统计分析、敏感性分析等
支持 KCIN 和 DCIN 两种仿真类型
"""

import csv
import json
import re
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

from .database import DatabaseManager

# 配置参数的正则模式（仅用于影响度分析）
CONFIG_PARAM_PATTERNS = [
    re.compile(r"^TOPO_TYPE$"),
    re.compile(r"^FLIT_SIZE$"),
    re.compile(r"^BURST$"),
    re.compile(r"^NETWORK_FREQUENCY$"),
    re.compile(r"^SLICE_PER_LINK_"),
    re.compile(r"^RN_RDB_SIZE$"),
    re.compile(r"^RN_WDB_SIZE$"),
    re.compile(r"^SN_DDR_RDB_SIZE$"),
    re.compile(r"^SN_DDR_WDB_SIZE$"),
    re.compile(r"^SN_L2M_RDB_SIZE$"),
    re.compile(r"^SN_L2M_WDB_SIZE$"),
    re.compile(r"^UNIFIED_RW_TRACKER$"),
    re.compile(r"LATENCY_original$"),
    re.compile(r"FIFO_DEPTH$"),
    re.compile(r"Etag_T\d_UE_MAX$"),
    re.compile(r"^ETAG_BOTHSIDE_UPGRADE$"),
    re.compile(r"^ETAG_T1_ENABLED$"),
    re.compile(r"^ITag_TRIGGER_Th_"),
    re.compile(r"^ITag_MAX_NUM_"),
    re.compile(r"^ENABLE_CROSSPOINT_CONFLICT_CHECK$"),
    re.compile(r"^ORDERING_"),
    re.compile(r"BW_LIMIT$"),
    re.compile(r"^IN_ORDER_"),
]


def is_config_param(param_name: str) -> bool:
    """判断参数是否为配置参数"""
    return any(pattern.search(param_name) for pattern in CONFIG_PARAM_PATTERNS)


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
        experiment_type: str = "kcin",
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
            experiment_type: 实验类型 ("kcin" 或 "dcin")
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
        if experiment_type not in ("kcin", "dcin"):
            raise ValueError(f"无效的实验类型: {experiment_type}，必须是 'kcin' 或 'dcin'")

        # 检查同类型下名称是否已存在（同名不同类型可以共存）
        existing = self.db.get_experiment_by_name(name, experiment_type)
        if existing:
            raise ValueError(f"{experiment_type.upper()} 实验名称 '{name}' 已存在")

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

    def get_experiment_by_name(self, name: str, experiment_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """根据名称获取实验

        Args:
            name: 实验名称
            experiment_type: 实验类型，如果指定则同时按类型筛选
        """
        return self.db.get_experiment_by_name(name, experiment_type)

    def list_experiments(
        self,
        status: Optional[str] = None,
        experiment_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取实验列表

        Args:
            status: 筛选状态
            experiment_type: 筛选类型 ("kcin" 或 "dcin")
        """
        return self.db.get_all_experiments(status, experiment_type)

    def update_experiment(
        self,
        experiment_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        completed_combinations: Optional[int] = None,
        best_performance: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """更新实验信息"""
        kwargs = {}
        if name is not None:
            kwargs["name"] = name
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
        result_html: Optional[str] = None,
        result_files: Optional[List[str]] = None,
        result_file_contents: Optional[Dict[str, bytes]] = None,
        error: Optional[str] = None,
    ) -> int:
        """
        添加单条仿真结果（线程安全）

        Args:
            experiment_id: 实验ID
            config_params: 配置参数字典
            performance: 主要性能指标
            result_details: 详细结果数据
            result_html: HTML报告内容
            result_files: 结果文件路径列表（向后兼容，从文件读取内容）
            result_file_contents: 文件内容字典 {filename: bytes_content}，直接存储到数据库
            error: 错误信息

        Returns:
            结果ID
        """
        with self._write_lock:
            # 获取实验类型
            exp = self.db.get_experiment(experiment_id)
            experiment_type = exp["experiment_type"] if exp else "kcin"

            result_id = self.db.add_result(
                experiment_id=experiment_id,
                config_params=config_params,
                performance=performance,
                result_details=result_details,
                result_html=result_html,
                result_files=result_files,
                error=error,
            )

            # 优先使用直接传入的文件内容
            if result_file_contents:
                self.db.store_result_files_from_contents(result_id, experiment_type, result_file_contents)
            # 向后兼容：如果传入了文件路径列表，从文件读取内容
            elif result_files:
                self.db.store_result_files_batch(result_id, experiment_type, result_files)

            # 更新实验统计
            self._update_experiment_stats(experiment_id, performance)

            return result_id

    def _update_experiment_stats(self, experiment_id: int, performance: float):
        """更新实验统计信息"""
        exp = self.db.get_experiment(experiment_id)
        if exp:
            # 使用实时查询获取准确的结果数
            completed = self.db.get_result_count(experiment_id)
            best = exp.get("best_performance")
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
        lightweight: bool = True,
    ) -> Dict[str, Any]:
        """
        分页获取结果

        Args:
            experiment_id: 实验ID
            page: 页码
            page_size: 每页数量
            sort_by: 排序字段
            order: 排序方向
            lightweight: 是否轻量模式（不加载 result_html 和 result_files）

        Returns:
            {results: [...], total: int, page: int, page_size: int}
        """
        results, total = self.db.get_results(
            experiment_id, page, page_size, sort_by, order, lightweight=lightweight
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
        experiment_type: str = "kcin",
        description: Optional[str] = None,
        topo_type: Optional[str] = None,
        batch_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        从CSV文件导入历史实验数据

        Args:
            csv_path: CSV文件路径
            experiment_name: 实验名称
            experiment_type: 实验类型 ("kcin" 或 "dcin")
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

            # 更新实验状态和统计 - 使用实际结果数
            actual_count = self.db.get_result_count(experiment_id)
            self.db.update_experiment(
                experiment_id,
                status="completed",
                total_combinations=actual_count,
                completed_combinations=actual_count,
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
            "平均带宽_DDR_混合",
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
        self, experiment_id: int, parameter: str, metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取单个参数的敏感性分析

        Args:
            experiment_id: 实验ID
            parameter: 参数名
            metric: 性能指标名（默认使用 performance 字段，也可以指定 config_params 中的字段）

        Returns:
            {
                parameter: str,
                metric: str,
                data: [{value: any, mean_performance: float, count: int}, ...]
            }
        """
        # 轻量级查询，只获取 config_params 和 performance
        results = self.db.get_results_for_analysis(experiment_id)

        # 按参数值分组统计
        groups = {}
        for result in results:
            config_params = result.get("config_params") or {}
            if parameter in config_params:
                value = config_params[parameter]
                # 获取性能值：默认使用 performance，否则从 config_params 中获取
                if metric is None:
                    perf_value = result.get("performance")
                else:
                    perf_value = config_params.get(metric)

                if perf_value is not None and isinstance(perf_value, (int, float)):
                    if value not in groups:
                        groups[value] = {"performances": [], "count": 0}
                    groups[value]["performances"].append(perf_value)
                    groups[value]["count"] += 1

        # 计算统计值
        data = []
        for value, stats in sorted(groups.items(), key=lambda x: x[0] if isinstance(x[0], (int, float)) else 0):
            performances = stats["performances"]
            if performances:
                data.append({
                    "value": value,
                    "mean_performance": sum(performances) / len(performances),
                    "min_performance": min(performances),
                    "max_performance": max(performances),
                    "count": stats["count"],
                })

        return {"parameter": parameter, "metric": metric or "performance", "data": data}

    def get_all_parameter_sensitivity(
        self, experiment_id: int, metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取所有参数的敏感性分析

        Args:
            experiment_id: 实验ID
            metric: 性能指标名（默认使用 performance 字段）
        """
        param_keys = self.db.get_param_keys(experiment_id)
        result = {}
        for param in param_keys:
            # 跳过与 metric 相同的参数
            if metric and param == metric:
                continue
            try:
                sensitivity = self.get_parameter_sensitivity(experiment_id, param, metric)
                if sensitivity["data"]:
                    result[param] = sensitivity
            except Exception:
                pass
        return result

    def get_parameter_influence(
        self, experiment_id: int, metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        计算各参数对性能的影响度（基于方差贡献）

        使用简化的方差分解方法：
        影响度 = Var(各取值的平均性能) / Var(所有性能)

        这种方法通过取平均值来抵消其他参数变化带来的噪声，
        更准确地反映单个参数对性能的独立贡献。

        Args:
            experiment_id: 实验ID
            metric: 性能指标名（默认使用 performance 字段）

        Returns:
            {
                metric: str,
                total_variance: float,
                parameters: [
                    {
                        name: str,
                        influence: float,  # 影响度 (0-1)
                        between_variance: float,  # 组间方差
                        mean_range: float,  # 各取值平均性能的范围
                        value_count: int,  # 取值数量
                    },
                    ...
                ]
            }
        """
        # 获取所有结果
        results = self.db.get_results_for_analysis(experiment_id)
        if not results:
            return {"metric": metric or "performance", "total_variance": 0, "parameters": []}

        # 收集所有性能值
        all_performances = []
        for result in results:
            config_params = result.get("config_params") or {}
            if metric is None:
                perf_value = result.get("performance")
            else:
                perf_value = config_params.get(metric)
            if perf_value is not None and isinstance(perf_value, (int, float)):
                all_performances.append(perf_value)

        if len(all_performances) < 2:
            return {"metric": metric or "performance", "total_variance": 0, "parameters": []}

        # 计算总方差
        mean_all = sum(all_performances) / len(all_performances)
        total_variance = sum((p - mean_all) ** 2 for p in all_performances) / len(all_performances)

        if total_variance == 0:
            return {"metric": metric or "performance", "total_variance": 0, "parameters": []}

        # 获取所有参数，只保留配置参数
        param_keys = self.db.get_param_keys(experiment_id)
        config_params_keys = [p for p in param_keys if is_config_param(p)]
        parameter_influences = []

        for param in config_params_keys:
            # 跳过与 metric 相同的参数
            if metric and param == metric:
                continue

            # 按参数值分组，计算每组的平均性能
            groups = {}
            for result in results:
                config_params = result.get("config_params") or {}
                if param not in config_params:
                    continue
                value = config_params[param]
                # 转换为可哈希类型
                try:
                    if isinstance(value, list):
                        value = tuple(value)
                    elif isinstance(value, dict):
                        # 字典转为排序后的元组
                        value = tuple(sorted(value.items()))
                    # 测试是否可哈希
                    hash(value)
                except TypeError:
                    # 跳过不可哈希的值
                    continue
                if metric is None:
                    perf_value = result.get("performance")
                else:
                    perf_value = config_params.get(metric)
                if perf_value is not None and isinstance(perf_value, (int, float)):
                    if value not in groups:
                        groups[value] = []
                    groups[value].append(perf_value)

            if len(groups) < 2:
                continue

            # 计算每个取值的平均性能
            group_means = []
            for value, perfs in groups.items():
                group_means.append(sum(perfs) / len(perfs))

            # 计算组间方差（各取值平均性能的方差）
            mean_of_means = sum(group_means) / len(group_means)
            between_variance = sum((m - mean_of_means) ** 2 for m in group_means) / len(group_means)

            # 影响度 = 组间方差 / 总方差
            influence = between_variance / total_variance

            parameter_influences.append({
                "name": param,
                "influence": round(influence, 4),
                "between_variance": round(between_variance, 4),
                "mean_range": round(max(group_means) - min(group_means), 4),
                "value_count": len(groups),
            })

        # 按影响度排序
        parameter_influences.sort(key=lambda x: x["influence"], reverse=True)

        return {
            "metric": metric or "performance",
            "total_variance": round(total_variance, 4),
            "parameters": parameter_influences,
        }

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
            raise ValueError("只能对比同类型的实验（都是 KCIN 或都是 DCIN）")

        return {
            "experiments": experiments,
            "best_configs": best_configs,
        }

    def compare_by_traffic(self, experiment_ids: List[int]) -> Dict[str, Any]:
        """
        按数据流对比多个实验的完整参数数据

        Args:
            experiment_ids: 实验ID列表

        Returns:
            {
                traffic_files: ["flow1.json", ...],
                experiments: [{id, name}, ...],
                param_keys: ["带宽ddr_mixed", "平均延迟", ...],  # 所有实验的参数并集
                data: [
                    {
                        traffic_file: "flow1.json",
                        exp_1_带宽ddr_mixed: 10.5,
                        exp_1_平均延迟: 100,
                        exp_2_带宽ddr_mixed: 11.2,
                        ...
                    },
                    ...
                ]
            }
        """
        experiments = []
        experiment_types = set()
        all_param_keys = set()

        # 收集所有实验的数据流结果
        all_traffic_data = {}  # {exp_id: {traffic_name: {param: value, ...}}}

        for exp_id in experiment_ids:
            exp = self.db.get_experiment(exp_id)
            if not exp:
                continue

            experiment_types.add(exp["experiment_type"])
            experiments.append({"id": exp_id, "name": exp["name"]})

            # 获取该实验的所有结果
            results, _ = self.db.get_results(exp_id, page=1, page_size=100000)

            # 按数据流名称分组，取每个数据流的第一条结果（或可以取平均）
            traffic_results = {}
            for result in results:
                config_params = result.get("config_params", {})
                # 尝试获取数据流名称字段
                traffic_name = config_params.get("数据流名称") or config_params.get("file_name") or "未知"

                # 收集所有参数键
                all_param_keys.update(config_params.keys())

                # 每个数据流只保留第一条结果（通常每个数据流只有一条）
                if traffic_name not in traffic_results:
                    traffic_results[traffic_name] = config_params

            all_traffic_data[exp_id] = traffic_results

        # 检查是否为同类型实验
        if len(experiment_types) > 1:
            raise ValueError("只能对比同类型的实验（都是 KCIN 或都是 DCIN）")

        # 收集所有数据流名称
        all_traffic_files = set()
        for traffic_results in all_traffic_data.values():
            all_traffic_files.update(traffic_results.keys())
        traffic_files = sorted(all_traffic_files)

        # 移除数据流名称字段本身
        param_keys = sorted([k for k in all_param_keys if k not in ("数据流名称", "file_name")])

        # 构建对比数据矩阵
        data = []
        for traffic_file in traffic_files:
            row = {"traffic_file": traffic_file}
            for exp in experiments:
                exp_id = exp["id"]
                params = all_traffic_data.get(exp_id, {}).get(traffic_file, {})
                for key in param_keys:
                    row[f"exp_{exp_id}_{key}"] = params.get(key)
            data.append(row)

        return {
            "traffic_files": traffic_files,
            "experiments": experiments,
            "param_keys": param_keys,
            "data": data,
        }

