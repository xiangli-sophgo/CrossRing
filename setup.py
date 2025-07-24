# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os

os.environ["PYTHONUTF8"] = "1"
setup(
    name="CrossRing",  # 包名称
    version="1.0.0",  # 版本号
    author="xiang.li",  # 作者
    author_email="xiang.li@sophgo.com",  # 作者邮箱
    description="A model of CrossRing",  # 简短描述
    packages=find_packages(),  # 自动查找包
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # 许可证
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Python 版本要求
    install_requires=[
        # 在这里列出你的依赖库,例如：
        "numpy",
        "networkx",
        "matplotlib",
        "scipy",
        "seaborn",
        "joblib",
        "tqdm",
        "optuna",
    ],
)
