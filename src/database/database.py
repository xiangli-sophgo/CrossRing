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

from .models import Base, Experiment, KcinResult, DcinResult, ResultFile


# 默认数据库路径
DEFAULT_DB_DIR = Path(__file__).parent.parent.parent.parent / "Result" / "Database"
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
            return

        if db_path is None:
            db_path = str(DEFAULT_DB_PATH)

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

    def create_tables(self):
        """创建所有表结构"""
        Base.metadata.create_all(self.engine)

    @contextmanager
    def get_session(self):
        """
        获取数据库会话（上下文管理器）

        使用方式:
            with db_manager.get_session() as session:
                session.query(...)
        """
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

    def get_experiment_by_name(self, name: str) -> Optional[dict]:
        """根据名称获取实验"""
        with self.get_session() as session:
            exp = session.query(Experiment).filter(Experiment.name == name).first()
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
        """删除实验（级联删除所有结果）"""
        with self.get_session() as session:
            experiment = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if experiment:
                session.delete(experiment)
                return True
            return False

    # ==================== 结果相关操作 ====================

    def _get_result_model(self, experiment_type: str):
        """根据实验类型获取对应的结果模型"""
        if experiment_type == "dcin":
            return DcinResult
        return KcinResult

    def _result_to_dict(self, result) -> dict:
        """将结果对象转为字典（在session内调用）"""
        import json

        # 解析 result_files JSON 字符串
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
            return result.id

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
    ) -> tuple:
        """
        分页获取结果

        Args:
            experiment_id: 实验ID
            page: 页码（从1开始）
            page_size: 每页数量
            sort_by: 排序字段
            order: 排序方向 (asc/desc)

        Returns:
            (结果字典列表, 总数)
        """
        experiment_type = self._get_experiment_type(experiment_id)
        ResultModel = self._get_result_model(experiment_type)

        with self.get_session() as session:
            query = session.query(ResultModel).filter(
                ResultModel.experiment_id == experiment_id
            )

            # 获取总数
            total = query.count()

            # 排序
            sort_column = getattr(ResultModel, sort_by, ResultModel.performance)
            if order == "desc":
                query = query.order_by(sort_column.desc())
            else:
                query = query.order_by(sort_column.asc())

            # 分页
            offset = (page - 1) * page_size
            results = query.offset(offset).limit(page_size).all()

            return [self._result_to_dict(r) for r in results], total

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
        获取实验结果中的所有配置参数键名

        用于前端动态生成表格列
        """
        experiment_type = self._get_experiment_type(experiment_id)
        ResultModel = self._get_result_model(experiment_type)

        with self.get_session() as session:
            result = (
                session.query(ResultModel)
                .filter(ResultModel.experiment_id == experiment_id)
                .first()
            )
            if result and result.config_params:
                return list(result.config_params.keys())
            return []

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
