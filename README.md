# CrossRing

CrossRing 是一个用于片上网络（NoC, Network-on-Chip）中 CrossRing 拓扑的建模与仿真的工具。支持对多种流量场景下的 CrossRing 拓扑进行建模实验和性能分析，适合芯片架构设计人员及研究人员进行 NoC 相关研究和验证。

## 主要功能

- 支持 CrossRing 拓扑结构的灵活建模与参数配置
- 支持多种流量文件输入与仿真场景配置
- 支持 REQ_RSP、Packet_Base、Feature 等多种模型
- 支持仿真结果的自动统计、可视化分析
- 提供多种辅助脚本实现流量预处理、参数寻优等功能

## 快速开始

1. **安装软件包和依赖**
   ```bash
   pip install -e .
   ```

2. **运行主仿真脚本**
   以 `scripts/model_main.py` 为例，配置好对应的流量文件路径和参数后，运行：
   ```bash
   python scripts/model_main.py
   ```
   你可以在 `scripts/model_main.py` 文件中修改 `traffic_file_path`、`file_name` 以及 `config_path` 等参数，以适配你的实验需求。

3. **结果分析与可视化**
   仿真结束后，结果会保存到 `../Result/` 目录下。可以使用仓库中的相关脚本（如 `result_statistic.py`、`FOP_realtime_visualize.py` 等）进行后处理和分析。


## 目录结构

```text
CrossRing/
├── src/                # 核心功能模块
│   ├── core/           # 拓扑与模型核心代码
│   ├── traffic_process/# 流量处理相关
│   └── utils/          # 工具类与基础组件
├── config/             # 配置文件
├── example/            # 示例代码
├── scripts/            # 各类仿真与分析脚本
├── requirements.txt    # 依赖说明
└── README.md           # 项目说明
```

## 贡献与反馈

欢迎提交 Issue 或 Pull Request 反馈问题或贡献代码！

---
如需了解更详细的功能、拓扑配置与高级用法，请参考各脚本代码和注释，或联系项目维护者。
