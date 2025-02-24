from setuptools import setup, find_packages

setup(
    name="CrossRing",  # 包名称
    version="0.1.0",  # 版本号
    author="xiang.li",  # 作者
    author_email="xiang.li@sophgo.com",  # 作者邮箱
    description="A model of CrossRing",  # 简短描述
    long_description=open("README.md").read(),  # 详细描述（从 README 文件读取）
    long_description_content_type="text/markdown",  # 描述内容类型
    url="https://github.com/xiangli-sophgo/NoC",  # 项目网址
    packages=find_packages(),  # 自动查找包
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # 许可证
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Python 版本要求
    install_requires=[
        # 在这里列出你的依赖库，例如：
        "numpy",
    ],
)
