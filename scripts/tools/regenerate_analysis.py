#!/usr/bin/env python3
"""
重新生成优化结果分析的脚本
使用方法: python regenerate_analysis.py <result_folder_path>
"""

import os
import sys
import json
import joblib
import pandas as pd
from datetime import datetime
import optuna
from optuna.trial import TrialState

# 导入原始脚本中的可视化函数
# 注意：需要确保原始脚本中的这些函数可以被导入
from scripts.find_optimal_parameters import (
    enhanced_create_visualization_plots,
    create_summary_report,
    create_optimization_guidance_report,
    create_enhanced_optimization_insight,
    create_parameter_impact_plot,
)


def load_optimization_results(result_folder):
    """加载保存的优化结果"""

    # 1. 加载Study对象
    study_file = os.path.join(result_folder, "optuna_study.pkl")
    if os.path.exists(study_file):
        study = joblib.load(study_file)
        print(f"✓ 成功加载Study对象: {len(study.trials)} trials")
    else:
        print("❌ 未找到optuna_study.pkl文件")
        return None, None, None, None

    # 2. 加载配置文件
    config_file = os.path.join(result_folder, "optimization_config.json")
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        print("✓ 成功加载配置文件")
    else:
        print("❌ 未找到optimization_config.json文件")
        return None, None, None, None

    # 3. 加载CSV数据（备用验证）
    csv_files = [f for f in os.listdir(result_folder) if f.endswith(".csv")]
    if csv_files:
        csv_file = os.path.join(result_folder, csv_files[0])
        df = pd.read_csv(csv_file)
        print(f"✓ 找到CSV文件: {len(df)} 条记录")
    else:
        print("⚠️ 未找到CSV文件")
        df = None

    return study, config_data, df, result_folder


def regenerate_all_analysis(result_folder):
    """重新生成所有分析报告和可视化"""

    # 加载数据
    study, config_data, df, save_dir = load_optimization_results(result_folder)
    if study is None:
        return False

    # 提取配置信息
    traffic_files = config_data.get("traffic_files", [])
    traffic_weights = config_data.get("traffic_weights", [])

    print(f"\n开始重新生成分析报告...")
    print(f"Traffic文件: {traffic_files}")
    print(f"权重配置: {traffic_weights}")
    print(f"完成的trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")

    try:
        # 1. 生成增强版可视化图表
        print("\n1. 生成可视化图表...")
        enhanced_create_visualization_plots(study, traffic_files, traffic_weights, save_dir)

        # 2. 生成总结报告
        print("2. 生成总结报告...")
        create_summary_report(study, traffic_files, traffic_weights, save_dir)

        # 3. 生成优化指导报告
        print("3. 生成优化指导报告...")
        vis_dir = os.path.join(save_dir, "visualizations")
        create_optimization_guidance_report(study, traffic_files, vis_dir)

        # 4. 显示最佳结果
        print("\n" + "=" * 60)
        print("重新生成完成!")
        if study.best_trials:
            best_trial = study.best_trials[0]
            print(f"最佳试验: Trial {best_trial.number}")
            print("最佳指标:", best_trial.values)
            print("最佳参数:")
            for param, value in best_trial.params.items():
                print(f"  {param}: {int(value)}")

            print("\n详细性能:")
            for traffic_file in traffic_files:
                traffic_name = traffic_file[:-4]
                key = f"Total_sum_BW_mean_{traffic_name}"
                if key in best_trial.user_attrs:
                    print(f"  {traffic_name}: {best_trial.user_attrs[key]:.2f} GB/s")

            weighted_bw = best_trial.user_attrs.get("Total_sum_BW_weighted_mean", 0)
            print(f"  加权平均: {weighted_bw:.2f} GB/s")

        print("=" * 60)
        print(f"✓ 所有分析文件已重新生成到: {save_dir}")
        print(f"✓ 主报告: {save_dir}optimization_report.html")
        print(f"✓ 可视化: {save_dir}visualizations/")

        return True

    except Exception as e:
        print(f"❌ 重新生成失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python regenerate_analysis.py <result_folder_path>")
        print("示例: python regenerate_analysis.py ../Result/CrossRing/REQ_RSP/FOP/2260E_ETag_multi_traffic_1210_1435/")
        sys.exit(1)

    result_folder = sys.argv[1]

    if not os.path.exists(result_folder):
        print(f"❌ 结果文件夹不存在: {result_folder}")
        sys.exit(1)

    print(f"正在重新生成优化分析...")
    print(f"结果文件夹: {result_folder}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    success = regenerate_all_analysis(result_folder)

    if success:
        print("\n🎉 重新生成成功!")
    else:
        print("\n❌ 重新生成失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
