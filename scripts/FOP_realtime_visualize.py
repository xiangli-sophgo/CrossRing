import joblib
import optuna.visualization as vis
import plotly.io as pio

# 设置图形显示方式为浏览器
pio.renderers.default = "browser"

# 加载 study 快照（确保路径与 save_intermediate_result 保存路径一致）
study_path = "../Result/study_snapshot.pkl"

try:
    study = joblib.load(study_path)

    # 绘制优化历史
    fig1 = vis.plot_optimization_history(study)
    fig1.show()

    # 参数重要性分析
    fig2 = vis.plot_param_importances(study)
    fig2.show()

    # 并行坐标图查看参数交互影响
    fig3 = vis.plot_parallel_coordinate(study)
    fig3.show()

except FileNotFoundError:
    print(f"Study file not found at {study_path}. Please make sure the optimization process has saved it.")
except Exception as e:
    print(f"Failed to visualize study: {e}")
