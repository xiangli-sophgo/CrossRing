"""
仿真结果数据库 - 数据库连接和CRUD操作

支持 KCIN 和 DCIN 两种仿真类型
"""

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session

from .models import Base, Experiment, KcinResult, DcinResult, ResultFile, AnalysisChart


# 默认数据库路径（保存到项目内 Result/database 目录）
DEFAULT_DB_DIR = Path(__file__).parent.parent.parent / "Result" / "database"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "simulation.db"


class DatabaseManager:
    """数据库管理器 - 处理连接和基础CRUD操作"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[str] = None):
        """单例模式，确保全局只有一个数据库连接"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化数据库管理器

        Args:
            db_path: 数据库文件路径，默认为 ../Result/Database/simulation.db
        """
        if self._initialized:
            # 检查数据库文件是否仍然存在，如果被删除则需要重新初始化
            if not Path(self.db_path).exists():
                self._reinitialize(db_path)
            return

        if db_path is None:
            db_path = str(DEFAULT_DB_PATH)

        self._do_init(db_path)

    def _do_init(self, db_path: str):
        """执行实际的初始化逻辑"""
        # 确保目录存在
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # 创建数据库引擎
        self.db_path = db_path
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},  # 允许多线程访问
        )

        # 创建会话工厂
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

        # 创建表
        self.create_tables()

        self._initialized = True

    def _reinitialize(self, db_path: Optional[str] = None):
        """重新初始化数据库连接（当数据库文件被删除后调用）"""
        if db_path is None:
            db_path = self.db_path if hasattr(self, 'db_path') else str(DEFAULT_DB_PATH)

        # 关闭旧的连接
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()

        self._do_init(db_path)

    def create_tables(self):
        """创建所有表结构"""
        Base.metadata.create_all(self.engine)

    def _ensure_db_exists(self):
        """确保数据库文件存在，如果被删除则重新初始化"""
        if not Path(self.db_path).exists():
            self._reinitialize()

    @contextmanager
    def get_session(self):
        """
        获取数据库会话（上下文管理器）

        使用方式:
            with db_manager.get_session() as session:
                session.query(...)
        """
        self._ensure_db_exists()
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session_direct(self) -> Session:
        """
        直接获取数据库会话（需要手动管理）

        Returns:
            Session: 数据库会话
        """
        self._ensure_db_exists()
        return self.SessionLocal()

    # ==================== 实验相关操作 ====================

    def create_experiment(self, **kwargs) -> int:
        """
        创建新实验

        Args:
            **kwargs: Experiment模型的字段，必须包含 experiment_type ("kcin" 或 "dcin")

        Returns:
            创建的实验ID
        """
        with self.get_session() as session:
            experiment = Experiment(**kwargs)
            session.add(experiment)
            session.flush()
            exp_id = experiment.id

            # 清理可能存在的孤儿记录（SQLite 会重用被删除的 ID）
            # 先获取可能的孤儿结果ID
            orphan_kcin_ids = [r.id for r in session.query(KcinResult).filter(
                KcinResult.experiment_id == exp_id
            ).all()]
            orphan_dcin_ids = [r.id for r in session.query(DcinResult).filter(
                DcinResult.experiment_id == exp_id
            ).all()]

            # 删除关联的 ResultFile
            if orphan_kcin_ids:
                session.query(ResultFile).filter(
                    ResultFile.result_type == "kcin",
                    ResultFile.result_id.in_(orphan_kcin_ids)
                ).delete(synchronize_session=False)
            if orphan_dcin_ids:
                session.query(ResultFile).filter(
                    ResultFile.result_type == "dcin",
                    ResultFile.result_id.in_(orphan_dcin_ids)
                ).delete(synchronize_session=False)

            # 删除孤儿结果和分析图表
            session.query(KcinResult).filter(
                KcinResult.experiment_id == exp_id
            ).delete(synchronize_session=False)
            session.query(DcinResult).filter(
                DcinResult.experiment_id == exp_id
            ).delete(synchronize_session=False)
            session.query(AnalysisChart).filter(
                AnalysisChart.experiment_id == exp_id
            ).delete(synchronize_session=False)

            return exp_id

    def _experiment_to_dict(self, exp: Experiment) -> dict:
        """将实验对象转为字典（在session内调用）"""
        import json

        # 解析 JSON 字符串字段
        traffic_files = exp.traffic_files
        if isinstance(traffic_files, str):
            traffic_files = json.loads(traffic_files)

        traffic_weights = exp.traffic_weights
        if isinstance(traffic_weights, str):
            traffic_weights = json.loads(traffic_weights)

        return {
            "id": exp.id,
            "name": exp.name,
            "experiment_type": exp.experiment_type,
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "description": exp.description,
            "config_path": exp.config_path,
            "topo_type": exp.topo_type,
            "traffic_files": traffic_files,
            "traffic_weights": traffic_weights,
            "simulation_time": exp.simulation_time,
            "n_repeats": exp.n_repeats,
            "n_jobs": exp.n_jobs,
            "status": exp.status,
            "total_combinations": exp.total_combinations,
            "completed_combinations": exp.completed_combinations,
            "best_performance": exp.best_performance,
            "git_commit": exp.git_commit,
            "notes": exp.notes,
        }

    def get_experiment(self, experiment_id: int) -> Optional[dict]:
        """获取单个实验"""
        with self.get_session() as session:
            exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if exp:
                return self._experiment_to_dict(exp)
            return None

    def get_experiment_by_name(self, name: str, experiment_type: Optional[str] = None) -> Optional[dict]:
        """根据名称获取实验

        Args:
            name: 实验名称
            experiment_type: 实验类型，如果指定则同时按类型筛选
        """
        with self.get_session() as session:
            query = session.query(Experiment).filter(Experiment.name == name)
            if experiment_type:
                query = query.filter(Experiment.experiment_type == experiment_type)
            exp = query.first()
            if exp:
                return self._experiment_to_dict(exp)
            return None

    def get_all_experiments(
        self,
        status: Optional[str] = None,
        experiment_type: Optional[str] = None,
    ) -> list:
        """
        获取所有实验

        Args:
            status: 筛选状态
            experiment_type: 筛选类型 ("kcin" 或 "dcin")
        """
        with self.get_session() as session:
            query = session.query(Experiment)
            if status:
                query = query.filter(Experiment.status == status)
            if experiment_type:
                query = query.filter(Experiment.experiment_type == experiment_type)
            experiments = query.order_by(Experiment.created_at.desc()).all()
            return [self._experiment_to_dict(exp) for exp in experiments]

    def update_experiment(self, experiment_id: int, **kwargs) -> bool:
        """更新实验信息"""
        with self.get_session() as session:
            result = session.query(Experiment).filter(Experiment.id == experiment_id).update(kwargs)
            return result > 0

    def delete_experiment(self, experiment_id: int) -> bool:
        """删除实验（级联删除所有结果和结果文件）"""
        with self.get_session() as session:
            experiment = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if experiment:
                # 先获取所有结果ID，用于删除result_files
                kcin_result_ids = [r.id for r in session.query(KcinResult).filter(
                    KcinResult.experiment_id == experiment_id
                ).all()]
                dcin_result_ids = [r.id for r in session.query(DcinResult).filter(
                    DcinResult.experiment_id == experiment_id
                ).all()]

                # 删除result_files表中的相关记录
                if kcin_result_ids:
                    session.query(ResultFile).filter(
                        ResultFile.result_type == "kcin",
                        ResultFile.result_id.in_(kcin_result_ids)
                    ).delete(synchronize_session=False)
                if dcin_result_ids:
                    session.query(ResultFile).filter(
                        ResultFile.result_type == "dcin",
                        ResultFile.result_id.in_(dcin_result_ids)
                    ).delete(synchronize_session=False)

                # 删除AnalysisChart（SQLite可能不启用外键CASCADE）
                session.query(AnalysisChart).filter(
                    AnalysisChart.experiment_id == experiment_id
                ).delete(synchronize_session=False)

                # 手动删除kcin_results和dcin_results（SQLite默认不启用外键CASCADE）
                session.query(KcinResult).filter(
                    KcinResult.experiment_id == experiment_id
                ).delete(synchronize_session=False)
                session.query(DcinResult).filter(
                    DcinResult.experiment_id == experiment_id
                ).delete(synchronize_session=False)

                # 删除实验
                session.delete(experiment)
                return True
            return False

    def delete_experiments_batch(self, experiment_ids: list) -> int:
        """批量删除实验（级联删除所有结果和结果文件）"""
        if not experiment_ids:
            return 0
        with self.get_session() as session:
            # 先获取所有结果ID
            kcin_result_ids = [r.id for r in session.query(KcinResult).filter(
                KcinResult.experiment_id.in_(experiment_ids)
            ).all()]
            dcin_result_ids = [r.id for r in session.query(DcinResult).filter(
                DcinResult.experiment_id.in_(experiment_ids)
            ).all()]

            # 删除result_files表中的相关记录
            if kcin_result_ids:
                session.query(ResultFile).filter(
                    ResultFile.result_type == "kcin",
                    ResultFile.result_id.in_(kcin_result_ids)
                ).delete(synchronize_session=False)
            if dcin_result_ids:
                session.query(ResultFile).filter(
                    ResultFile.result_type == "dcin",
                    ResultFile.result_id.in_(dcin_result_ids)
                ).delete(synchronize_session=False)

            # 删除AnalysisChart（SQLite可能不启用外键CASCADE）
            session.query(AnalysisChart).filter(
                AnalysisChart.experiment_id.in_(experiment_ids)
            ).delete(synchronize_session=False)

            # 手动删除kcin_results和dcin_results（SQLite默认不启用外键CASCADE）
            session.query(KcinResult).filter(
                KcinResult.experiment_id.in_(experiment_ids)
            ).delete(synchronize_session=False)
            session.query(DcinResult).filter(
                DcinResult.experiment_id.in_(experiment_ids)
            ).delete(synchronize_session=False)

            # 删除实验
            count = session.query(Experiment).filter(
                Experiment.id.in_(experiment_ids)
            ).delete(synchronize_session=False)
            return count

    # ==================== 结果相关操作 ====================

    def _get_result_model(self, experiment_type: str):
        """根据实验类型获取对应的结果模型"""
        if experiment_type == "dcin":
            return DcinResult
        return KcinResult

    def _result_to_dict(self, result, lightweight: bool = False) -> dict:
        """将结果对象转为字典（在session内调用）

        Args:
            result: 结果对象
            lightweight: 是否轻量模式（不加载 result_html 和 result_files）
        """
        import json

        if lightweight:
            # 轻量模式：不加载大字段，用于列表展示
            # 添加 has_result_html 标志，让前端知道是否有 HTML 报告
            return {
                "id": result.id,
                "experiment_id": result.experiment_id,
                "created_at": result.created_at.isoformat() if result.created_at else None,
                "performance": result.performance,
                "config_params": result.config_params,
                "result_details": result.result_details,
                "error": result.error,
                "has_result_html": bool(result.result_html),
            }

        # 完整模式：包含所有字段
        result_files = result.result_files
        if isinstance(result_files, str):
            result_files = json.loads(result_files)

        return {
            "id": result.id,
            "experiment_id": result.experiment_id,
            "created_at": result.created_at.isoformat() if result.created_at else None,
            "performance": result.performance,
            "config_params": result.config_params,
            "result_details": result.result_details,
            "result_html": result.result_html,
            "result_files": result_files,
            "error": result.error,
        }

    def _get_experiment_type(self, experiment_id: int) -> str:
        """获取实验的类型"""
        with self.get_session() as session:
            exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if exp is None:
                raise ValueError(f"实验 {experiment_id} 不存在")
            return exp.experiment_type

    def add_result(
        self,
        experiment_id: int,
        config_params: dict,
        performance: float,
        result_details: Optional[dict] = None,
        result_html: Optional[str] = None,
        result_files: Optional[list] = None,
        result_file_contents: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> int:
        """
        添加仿真结果

        Args:
            experiment_id: 实验ID
            config_params: 配置参数字典
            performance: 主要性能指标
            result_details: 详细结果数据
            result_html: HTML报告内容
            result_files: 结果文件路径列表
            result_file_contents: 文件内容字典 {filename: bytes_content}
            error: 错误信息

        Returns:
            创建的结果ID
        """
        import json

        experiment_type = self._get_experiment_type(experiment_id)
        ResultModel = self._get_result_model(experiment_type)

        with self.get_session() as session:
            result = ResultModel(
                experiment_id=experiment_id,
                config_params=config_params,
                performance=performance,
                result_details=result_details,
                result_html=result_html,
                result_files=json.dumps(result_files) if result_files else None,
                error=error,
            )
            session.add(result)
            session.flush()
            result_id = result.id

        # 如果有文件内容，存储到result_files表
        if result_file_contents:
            self.store_result_files_from_contents(result_id, experiment_type, result_file_contents)

        return result_id

    def add_kcin_result(
        self,
        experiment_id: int,
        config_params: dict,
        performance: float,
        result_details: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> KcinResult:
        """添加 KCIN 仿真结果"""
        with self.get_session() as session:
            result = KcinResult(
                experiment_id=experiment_id,
                config_params=config_params,
                performance=performance,
                result_details=result_details,
                error=error,
            )
            session.add(result)
            session.flush()
            result_id = result.id

        with self.get_session() as session:
            return session.query(KcinResult).filter(KcinResult.id == result_id).first()

    def add_dcin_result(
        self,
        experiment_id: int,
        config_params: dict,
        performance: float,
        result_details: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> DcinResult:
        """添加 DCIN 仿真结果"""
        with self.get_session() as session:
            result = DcinResult(
                experiment_id=experiment_id,
                config_params=config_params,
                performance=performance,
                result_details=result_details,
                error=error,
            )
            session.add(result)
            session.flush()
            result_id = result.id

        with self.get_session() as session:
            return session.query(DcinResult).filter(DcinResult.id == result_id).first()

    def add_results_batch(
        self,
        experiment_id: int,
        results: list,
    ) -> int:
        """
        批量添加仿真结果

        Args:
            experiment_id: 实验ID
            results: 结果字典列表，每个字典包含 config_params, performance, result_details

        Returns:
            添加的记录数
        """
        experiment_type = self._get_experiment_type(experiment_id)
        ResultModel = self._get_result_model(experiment_type)

        with self.get_session() as session:
            objects = [
                ResultModel(
                    experiment_id=experiment_id,
                    config_params=r.get("config_params"),
                    performance=r.get("performance"),
                    result_details=r.get("result_details"),
                    error=r.get("error"),
                )
                for r in results
            ]
            session.bulk_save_objects(objects)
            return len(objects)

    def get_results(
        self,
        experiment_id: int,
        page: int = 1,
        page_size: int = 100,
        sort_by: str = "performance",
        order: str = "desc",
        lightweight: bool = True,
    ) -> tuple:
        """
        分页获取结果

        Args:
            experiment_id: 实验ID
            page: 页码（从1开始）
            page_size: 每页数量
            sort_by: 排序字段
            order: 排序方向 (asc/desc)
            lightweight: 是否轻量模式（不加载 result_html 和 result_files）

        Returns:
            (结果字典列表, 总数)
        """
        experiment_type = self._get_experiment_type(experiment_id)
        ResultModel = self._get_result_model(experiment_type)

        with self.get_session() as session:
            # 获取总数
            total = session.query(func.count(ResultModel.id)).filter(
                ResultModel.experiment_id == experiment_id
            ).scalar()

            # 排序字段
            sort_column = getattr(ResultModel, sort_by, ResultModel.performance)

            if lightweight:
                # 轻量模式：只查询需要的字段，不加载 result_html 和 result_files 内容
                # 使用 func.length 检查 result_html 是否存在
                query = session.query(
                    ResultModel.id,
                    ResultModel.experiment_id,
                    ResultModel.created_at,
                    ResultModel.performance,
                    ResultModel.config_params,
                    ResultModel.result_details,
                    ResultModel.error,
                    func.length(ResultModel.result_html).label("html_length"),
                ).filter(ResultModel.experiment_id == experiment_id)

                if order == "desc":
                    query = query.order_by(sort_column.desc())
                else:
                    query = query.order_by(sort_column.asc())

                offset = (page - 1) * page_size
                results = query.offset(offset).limit(page_size).all()

                return [
                    {
                        "id": r.id,
                        "experiment_id": r.experiment_id,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                        "performance": r.performance,
                        "config_params": r.config_params,
                        "result_details": r.result_details,
                        "error": r.error,
                        "has_result_html": bool(r.html_length and r.html_length > 0),
                    }
                    for r in results
                ], total
            else:
                # 完整模式：加载完整对象
                query = session.query(ResultModel).filter(
                    ResultModel.experiment_id == experiment_id
                )

                if order == "desc":
                    query = query.order_by(sort_column.desc())
                else:
                    query = query.order_by(sort_column.asc())

                offset = (page - 1) * page_size
                results = query.offset(offset).limit(page_size).all()

                return [self._result_to_dict(r, lightweight=False) for r in results], total

    def get_kcin_results(
        self,
        experiment_id: int,
        page: int = 1,
        page_size: int = 100,
        sort_by: str = "performance",
        order: str = "desc",
    ) -> tuple:
        """分页获取 KCIN 结果"""
        with self.get_session() as session:
            query = session.query(KcinResult).filter(
                KcinResult.experiment_id == experiment_id
            )
            total = query.count()
            sort_column = getattr(KcinResult, sort_by, KcinResult.performance)
            if order == "desc":
                query = query.order_by(sort_column.desc())
            else:
                query = query.order_by(sort_column.asc())
            offset = (page - 1) * page_size
            results = query.offset(offset).limit(page_size).all()
            return results, total

    def get_dcin_results(
        self,
        experiment_id: int,
        page: int = 1,
        page_size: int = 100,
        sort_by: str = "performance",
        order: str = "desc",
    ) -> tuple:
        """分页获取 DCIN 结果"""
        with self.get_session() as session:
            query = session.query(DcinResult).filter(
                DcinResult.experiment_id == experiment_id
            )
            total = query.count()
            sort_column = getattr(DcinResult, sort_by, DcinResult.performance)
            if order == "desc":
                query = query.order_by(sort_column.desc())
            else:
                query = query.order_by(sort_column.asc())
            offset = (page - 1) * page_size
            results = query.offset(offset).limit(page_size).all()
            return results, total

    def get_best_results(self, experiment_id: int, limit: int = 10) -> list:
        """获取最佳结果"""
        experiment_type = self._get_experiment_type(experiment_id)
        ResultModel = self._get_result_model(experiment_type)

        with self.get_session() as session:
            results = (
                session.query(ResultModel)
                .filter(ResultModel.experiment_id == experiment_id)
                .order_by(ResultModel.performance.desc())
                .limit(limit)
                .all()
            )
            return [self._result_to_dict(r) for r in results]

    def get_result_count(self, experiment_id: int) -> int:
        """获取结果数量"""
        experiment_type = self._get_experiment_type(experiment_id)
        ResultModel = self._get_result_model(experiment_type)

        with self.get_session() as session:
            return (
                session.query(ResultModel)
                .filter(ResultModel.experiment_id == experiment_id)
                .count()
            )

    def get_performance_distribution(self, experiment_id: int, bins: int = 50) -> dict:
        """
        获取性能分布数据

        Args:
            experiment_id: 实验ID
            bins: 直方图分箱数

        Returns:
            {min, max, mean, count, histogram: [(bin_start, bin_end, count), ...]}
        """
        experiment_type = self._get_experiment_type(experiment_id)
        ResultModel = self._get_result_model(experiment_type)

        with self.get_session() as session:
            # 获取基本统计
            stats = (
                session.query(
                    func.min(ResultModel.performance),
                    func.max(ResultModel.performance),
                    func.avg(ResultModel.performance),
                    func.count(ResultModel.id),
                )
                .filter(ResultModel.experiment_id == experiment_id)
                .first()
            )

            if stats[0] is None:
                return {"min": 0, "max": 0, "mean": 0, "count": 0, "histogram": []}

            min_val, max_val, mean_val, count = stats

            # 获取所有性能值用于计算直方图
            performances = [
                r[0]
                for r in session.query(ResultModel.performance)
                .filter(ResultModel.experiment_id == experiment_id)
                .all()
            ]

            # 计算直方图
            bin_width = (max_val - min_val) / bins if max_val > min_val else 1
            histogram = []
            for i in range(bins):
                bin_start = min_val + i * bin_width
                bin_end = min_val + (i + 1) * bin_width
                bin_count = sum(1 for p in performances if bin_start <= p < bin_end)
                histogram.append((bin_start, bin_end, bin_count))

            # 最后一个bin包含max_val
            if histogram and performances:
                histogram[-1] = (histogram[-1][0], histogram[-1][1], histogram[-1][2] + sum(1 for p in performances if p == max_val))

            return {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "count": count,
                "histogram": histogram,
            }

    def get_param_keys(self, experiment_id: int) -> list:
        """
        获取实验结果中的所有配置参数键名（所有结果的并集）

        用于前端动态生成表格列
        """
        experiment_type = self._get_experiment_type(experiment_id)
        ResultModel = self._get_result_model(experiment_type)

        with self.get_session() as session:
            results = (
                session.query(ResultModel.config_params)
                .filter(ResultModel.experiment_id == experiment_id)
                .all()
            )
            # 收集所有结果的参数键的并集
            all_keys = set()
            for (config_params,) in results:
                if config_params:
                    all_keys.update(config_params.keys())
            return sorted(list(all_keys))

    def get_results_for_analysis(self, experiment_id: int) -> list:
        """
        轻量级查询，只返回分析需要的字段
        不加载 result_html 等大字段，用于热力图/敏感性分析等
        """
        experiment_type = self._get_experiment_type(experiment_id)
        ResultModel = self._get_result_model(experiment_type)

        with self.get_session() as session:
            results = session.query(
                ResultModel.id,
                ResultModel.config_params,
                ResultModel.performance
            ).filter(
                ResultModel.experiment_id == experiment_id
            ).all()

            return [
                {"id": r.id, "config_params": r.config_params, "performance": r.performance}
                for r in results
            ]

    @classmethod
    def reset_instance(cls):
        """重置单例实例（用于测试）"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance = None

    # ==================== 结果文件存储操作 ====================

    def store_result_file(
        self,
        result_id: int,
        result_type: str,
        file_path: str,
        file_content: bytes = None,
    ) -> int:
        """
        存储结果文件到数据库

        Args:
            result_id: 结果ID
            result_type: 结果类型 ("kcin" 或 "dcin")
            file_path: 文件路径
            file_content: 文件内容（如果为None则从file_path读取）

        Returns:
            存储的文件ID
        """
        import os
        import mimetypes

        # 如果没有提供内容，从文件读取
        if file_content is None:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            with open(file_path, 'rb') as f:
                file_content = f.read()

        file_name = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        file_size = len(file_content)

        with self.get_session() as session:
            result_file = ResultFile(
                result_id=result_id,
                result_type=result_type,
                file_name=file_name,
                file_path=file_path,
                mime_type=mime_type,
                file_size=file_size,
                file_content=file_content,
            )
            session.add(result_file)
            session.flush()
            return result_file.id

    def store_result_files_batch(
        self,
        result_id: int,
        result_type: str,
        file_paths: list,
    ) -> list:
        """
        批量存储结果文件

        Args:
            result_id: 结果ID
            result_type: 结果类型
            file_paths: 文件路径列表

        Returns:
            存储的文件ID列表
        """
        import os
        import mimetypes

        file_ids = []
        with self.get_session() as session:
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    continue

                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()

                    file_name = os.path.basename(file_path)
                    mime_type, _ = mimetypes.guess_type(file_path)
                    file_size = len(file_content)

                    result_file = ResultFile(
                        result_id=result_id,
                        result_type=result_type,
                        file_name=file_name,
                        file_path=file_path,
                        mime_type=mime_type,
                        file_size=file_size,
                        file_content=file_content,
                    )
                    session.add(result_file)
                    session.flush()
                    file_ids.append(result_file.id)
                except Exception:
                    continue

        return file_ids

    def store_result_files_from_contents(
        self,
        result_id: int,
        result_type: str,
        file_contents: dict,
    ) -> list:
        """
        从内存中的文件内容直接存储到数据库（不写本地文件）

        Args:
            result_id: 结果ID
            result_type: 结果类型
            file_contents: 文件内容字典 {filename: bytes_content}

        Returns:
            存储的文件ID列表
        """
        import mimetypes

        file_ids = []
        with self.get_session() as session:
            for file_name, file_content in file_contents.items():
                try:
                    # 确保内容是bytes类型
                    if isinstance(file_content, str):
                        file_content = file_content.encode('utf-8')

                    mime_type, _ = mimetypes.guess_type(file_name)
                    file_size = len(file_content)

                    result_file = ResultFile(
                        result_id=result_id,
                        result_type=result_type,
                        file_name=file_name,
                        file_path=f"db://{file_name}",  # 使用虚拟路径标识数据库存储
                        mime_type=mime_type,
                        file_size=file_size,
                        file_content=file_content,
                    )
                    session.add(result_file)
                    session.flush()
                    file_ids.append(result_file.id)
                except Exception:
                    continue

        return file_ids

    def get_result_file(self, file_id: int) -> Optional[dict]:
        """
        获取结果文件

        Args:
            file_id: 文件ID

        Returns:
            文件信息字典（包含内容）
        """
        with self.get_session() as session:
            result_file = session.query(ResultFile).filter(ResultFile.id == file_id).first()
            if result_file:
                return {
                    "id": result_file.id,
                    "result_id": result_file.result_id,
                    "result_type": result_file.result_type,
                    "file_name": result_file.file_name,
                    "file_path": result_file.file_path,
                    "mime_type": result_file.mime_type,
                    "file_size": result_file.file_size,
                    "file_content": result_file.file_content,
                }
            return None

    def get_result_files_list(self, result_id: int, result_type: str) -> list:
        """
        获取结果的所有文件列表（不包含内容）

        Args:
            result_id: 结果ID
            result_type: 结果类型

        Returns:
            文件信息列表
        """
        with self.get_session() as session:
            files = session.query(ResultFile).filter(
                ResultFile.result_id == result_id,
                ResultFile.result_type == result_type,
            ).all()
            return [
                {
                    "id": f.id,
                    "file_name": f.file_name,
                    "file_path": f.file_path,
                    "mime_type": f.mime_type,
                    "file_size": f.file_size,
                }
                for f in files
            ]

    def get_result_file_by_name(self, result_id: int, result_type: str, file_name: str):
        """
        根据文件名获取结果文件（包含内容）

        Args:
            result_id: 结果ID
            result_type: 结果类型
            file_name: 文件名

        Returns:
            包含file_content的对象；如果不存在返回None
        """
        with self.get_session() as session:
            result_file = session.query(ResultFile).filter(
                ResultFile.result_id == result_id,
                ResultFile.result_type == result_type,
                ResultFile.file_name == file_name,
            ).first()
            if result_file:
                # 在session关闭前访问file_content，创建一个简单对象返回
                class FileData:
                    pass
                data = FileData()
                data.id = result_file.id
                data.file_name = result_file.file_name
                data.file_content = result_file.file_content
                data.file_size = result_file.file_size
                data.mime_type = result_file.mime_type
                return data
            return None

    def delete_result_files(self, result_id: int, result_type: str) -> int:
        """
        删除结果的所有文件

        Args:
            result_id: 结果ID
            result_type: 结果类型

        Returns:
            删除的文件数量
        """
        with self.get_session() as session:
            count = session.query(ResultFile).filter(
                ResultFile.result_id == result_id,
                ResultFile.result_type == result_type,
            ).delete()
            return count

    def delete_result(self, result_id: int, experiment_type: str, experiment_id: int = None) -> bool:
        """
        删除单个结果

        Args:
            result_id: 结果ID
            experiment_type: 实验类型 ("kcin" 或 "dcin")
            experiment_id: 实验ID（用于更新统计）

        Returns:
            是否删除成功
        """
        ResultModel = self._get_result_model(experiment_type)
        with self.get_session() as session:
            # 获取结果的experiment_id（如果没有传入）
            if experiment_id is None:
                result = session.query(ResultModel).filter(ResultModel.id == result_id).first()
                if result:
                    experiment_id = result.experiment_id

            # 先删除关联的文件
            session.query(ResultFile).filter(
                ResultFile.result_id == result_id,
                ResultFile.result_type == experiment_type,
            ).delete()
            # 删除结果
            count = session.query(ResultModel).filter(
                ResultModel.id == result_id
            ).delete()

            # 更新实验统计
            if experiment_id and count > 0:
                new_count = session.query(ResultModel).filter(
                    ResultModel.experiment_id == experiment_id
                ).count()
                session.query(Experiment).filter(
                    Experiment.id == experiment_id
                ).update({"completed_combinations": new_count})

            return count > 0

    def delete_results_batch(self, result_ids: list, experiment_type: str, experiment_id: int) -> int:
        """
        批量删除结果

        Args:
            result_ids: 结果ID列表
            experiment_type: 实验类型 ("kcin" 或 "dcin")
            experiment_id: 实验ID

        Returns:
            删除的结果数量
        """
        if not result_ids:
            return 0

        ResultModel = self._get_result_model(experiment_type)
        with self.get_session() as session:
            # 先删除关联的文件
            session.query(ResultFile).filter(
                ResultFile.result_id.in_(result_ids),
                ResultFile.result_type == experiment_type,
            ).delete(synchronize_session=False)

            # 批量删除结果
            count = session.query(ResultModel).filter(
                ResultModel.id.in_(result_ids)
            ).delete(synchronize_session=False)

            # 更新实验统计
            if count > 0:
                new_count = session.query(ResultModel).filter(
                    ResultModel.experiment_id == experiment_id
                ).count()
                session.query(Experiment).filter(
                    Experiment.id == experiment_id
                ).update({"completed_combinations": new_count})

            return count

    # ==================== 分析图表配置管理 ====================

    def get_analysis_charts(self, experiment_id: int) -> list:
        """
        获取实验的所有分析图表配置

        Args:
            experiment_id: 实验ID

        Returns:
            图表配置列表
        """
        with self.get_session() as session:
            charts = session.query(AnalysisChart).filter(
                AnalysisChart.experiment_id == experiment_id
            ).order_by(AnalysisChart.sort_order, AnalysisChart.id).all()
            return [
                {
                    "id": c.id,
                    "name": c.name,
                    "chart_type": c.chart_type,
                    "config": c.config,
                    "sort_order": c.sort_order,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                }
                for c in charts
            ]

    def add_analysis_chart(
        self,
        experiment_id: int,
        name: str,
        chart_type: str,
        config: dict,
        sort_order: int = 0,
    ) -> int:
        """
        添加分析图表配置

        Args:
            experiment_id: 实验ID
            name: 图表名称
            chart_type: 图表类型
            config: 图表配置
            sort_order: 排序顺序

        Returns:
            图表ID
        """
        with self.get_session() as session:
            chart = AnalysisChart(
                experiment_id=experiment_id,
                name=name,
                chart_type=chart_type,
                config=config,
                sort_order=sort_order,
            )
            session.add(chart)
            session.flush()
            return chart.id

    def update_analysis_chart(
        self,
        chart_id: int,
        name: str = None,
        config: dict = None,
        sort_order: int = None,
    ) -> bool:
        """
        更新分析图表配置

        Args:
            chart_id: 图表ID
            name: 新名称
            config: 新配置
            sort_order: 新排序

        Returns:
            是否更新成功
        """
        with self.get_session() as session:
            chart = session.query(AnalysisChart).filter(
                AnalysisChart.id == chart_id
            ).first()
            if not chart:
                return False

            if name is not None:
                chart.name = name
            if config is not None:
                chart.config = config
            if sort_order is not None:
                chart.sort_order = sort_order

            return True

    def delete_analysis_chart(self, chart_id: int) -> bool:
        """
        删除分析图表配置

        Args:
            chart_id: 图表ID

        Returns:
            是否删除成功
        """
        with self.get_session() as session:
            count = session.query(AnalysisChart).filter(
                AnalysisChart.id == chart_id
            ).delete()
            return count > 0
