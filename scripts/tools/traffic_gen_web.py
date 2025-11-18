"""
CrossRing 流量生成Web可视化工具

基于Streamlit的交互式流量生成工具,提供:
- 拓扑可视化与交互式节点选择
- 配置管理与参数验证
- 流量生成与结果分析
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.traffic_process.web_modules.topology_visualizer import (
    TopologyVisualizer,
    get_default_ip_mappings
)
from src.traffic_process.web_modules.config_manager import (
    ConfigManager,
    TrafficConfig
)
from src.traffic_process.web_modules.traffic_analyzer import TrafficAnalyzer
from src.traffic_process.web_modules.generation_engine import generate_traffic_from_configs


# ==================== 页面配置 ====================

st.set_page_config(
    page_title="CrossRing 流量生成器",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== 会话状态初始化 ====================

def init_session_state():
    """初始化会话状态"""
    if 'topo_type' not in st.session_state:
        st.session_state.topo_type = "5x4"

    if 'selected_src_nodes' not in st.session_state:
        st.session_state.selected_src_nodes = set()

    if 'selected_dst_nodes' not in st.session_state:
        st.session_state.selected_dst_nodes = set()

    if 'config_manager' not in st.session_state:
        rows, cols = map(int, st.session_state.topo_type.split('x'))
        num_nodes = rows * cols
        st.session_state.config_manager = ConfigManager(num_nodes)

    if 'generated_traffic' not in st.session_state:
        st.session_state.generated_traffic = None

    if 'last_file_path' not in st.session_state:
        st.session_state.last_file_path = None


# ==================== 辅助函数 ====================

def handle_node_click(node_id: int, select_mode: str):
    """
    处理节点点击事件

    :param node_id: 节点ID
    :param select_mode: 选择模式 ("source" 或 "destination")
    """
    if select_mode == "source":
        if node_id in st.session_state.selected_src_nodes:
            st.session_state.selected_src_nodes.remove(node_id)
        else:
            st.session_state.selected_src_nodes.add(node_id)
    else:
        if node_id in st.session_state.selected_dst_nodes:
            st.session_state.selected_dst_nodes.remove(node_id)
        else:
            st.session_state.selected_dst_nodes.add(node_id)


def create_config_from_selection(
    src_ip_type: str,
    dst_ip_type: str,
    speed: float,
    burst: int,
    req_type: str
) -> TrafficConfig:
    """
    根据当前选择创建配置

    :return: TrafficConfig对象
    """
    src_map = {src_ip_type: list(st.session_state.selected_src_nodes)}
    dst_map = {dst_ip_type: list(st.session_state.selected_dst_nodes)}

    return TrafficConfig(
        src_map=src_map,
        dst_map=dst_map,
        speed=speed,
        burst=burst,
        req_type=req_type
    )


# ==================== 主界面 ====================

def render_main_ui():
    """渲染主界面"""
    # 标题
    st.title("🚦 CrossRing 流量生成可视化工具")
    st.markdown("---")

    # 侧边栏 - 全局配置
    with st.sidebar:
        st.header("⚙️ 全局配置")

        # 拓扑类型选择
        topo_type = st.selectbox(
            "拓扑类型",
            ["5x4", "4x4"],
            index=0 if st.session_state.topo_type == "5x4" else 1
        )

        # 如果拓扑类型变化,重新初始化
        if topo_type != st.session_state.topo_type:
            st.session_state.topo_type = topo_type
            rows, cols = map(int, topo_type.split('x'))
            num_nodes = rows * cols
            st.session_state.config_manager = ConfigManager(num_nodes)
            st.session_state.selected_src_nodes = set()
            st.session_state.selected_dst_nodes = set()
            st.rerun()

        # 仿真时长
        end_time = st.number_input(
            "仿真时长 (ns)",
            min_value=100,
            max_value=100000,
            value=6000,
            step=100
        )

        st.markdown("---")
        st.markdown("### 📖 使用说明")
        st.markdown("""
        1. 在拓扑图上点击节点选择源/目标
        2. 配置流量参数
        3. 添加到配置列表
        4. 点击生成流量文件
        5. 查看结果分析
        """)

    # 主区域 - 分为左右两栏
    col_left, col_right = st.columns([1.2, 1])

    # 左栏 - 拓扑可视化
    with col_left:
        st.subheader("🗺️ 拓扑可视化")

        # 节点选择模式
        select_mode = st.radio(
            "点击模式",
            ["source", "destination"],
            format_func=lambda x: "选择源节点" if x == "source" else "选择目标节点",
            horizontal=True
        )

        # 绘制拓扑图
        ip_mappings = get_default_ip_mappings(st.session_state.topo_type)
        visualizer = TopologyVisualizer(st.session_state.topo_type, ip_mappings)

        fig = visualizer.draw_topology_grid(
            selected_src=st.session_state.selected_src_nodes,
            selected_dst=st.session_state.selected_dst_nodes
        )

        # 显示拓扑图并捕获点击事件
        click_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

        # 处理点击事件
        if click_data and 'selection' in click_data and 'points' in click_data['selection']:
            points = click_data['selection']['points']
            if points:
                node_id = points[0]['customdata']
                handle_node_click(node_id, select_mode)
                st.rerun()

        # 显示当前选择
        st.markdown("##### 当前选择:")
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            st.info(f"**源节点**: {sorted(st.session_state.selected_src_nodes) if st.session_state.selected_src_nodes else '未选择'}")
        with col_sel2:
            st.info(f"**目标节点**: {sorted(st.session_state.selected_dst_nodes) if st.session_state.selected_dst_nodes else '未选择'}")

        # 清空选择按钮
        if st.button("🗑️ 清空选择"):
            st.session_state.selected_src_nodes = set()
            st.session_state.selected_dst_nodes = set()
            st.rerun()

    # 右栏 - 配置管理
    with col_right:
        st.subheader("⚙️ 流量配置")

        # 配置表单
        with st.form("config_form"):
            src_ip_type = st.text_input("源IP类型", value="gdma_0")
            dst_ip_type = st.text_input("目标IP类型", value="ddr_0")

            speed = st.slider(
                "带宽 (GB/s)",
                min_value=0.1,
                max_value=128.0,
                value=46.08,
                step=0.01
            )

            burst = st.selectbox("Burst长度", [1, 2, 4, 8, 16], index=2)

            req_type = st.radio("请求类型", ["R", "W"], horizontal=True)

            submit_button = st.form_submit_button("➕ 添加配置", use_container_width=True)

            if submit_button:
                # 检查是否有选择节点
                if not st.session_state.selected_src_nodes or not st.session_state.selected_dst_nodes:
                    st.error("请先选择源节点和目标节点!")
                else:
                    # 创建配置
                    config = create_config_from_selection(
                        src_ip_type, dst_ip_type, speed, burst, req_type
                    )

                    # 添加到配置管理器
                    success, errors = st.session_state.config_manager.add_config(config)

                    if success:
                        st.success("✅ 配置添加成功!")
                        # 清空选择
                        st.session_state.selected_src_nodes = set()
                        st.session_state.selected_dst_nodes = set()
                        st.rerun()
                    else:
                        st.error("❌ 配置验证失败:\n" + "\n".join(errors))

    # 配置列表展示
    st.markdown("---")
    st.subheader("📋 配置列表")

    configs = st.session_state.config_manager.get_all_configs()

    if not configs:
        st.info("暂无配置,请添加流量配置")
    else:
        # 显示配置
        for i, config in enumerate(configs):
            with st.expander(f"配置 #{config.config_id}: {list(config.src_map.keys())[0]} → {list(config.dst_map.keys())[0]}", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.write(f"**源节点**: {config.get_source_nodes()}")
                    st.write(f"**目标节点**: {config.get_destination_nodes()}")

                with col2:
                    st.write(f"**带宽**: {config.speed} GB/s")
                    st.write(f"**Burst**: {config.burst}")
                    st.write(f"**类型**: {'读' if config.req_type == 'R' else '写'}")

                with col3:
                    if st.button("🗑️ 删除", key=f"del_{config.config_id}"):
                        st.session_state.config_manager.remove_config(config.config_id)
                        st.rerun()

        # 预估统计
        st.markdown("---")
        st.subheader("📊 预估统计")

        estimate = st.session_state.config_manager.estimate_traffic(end_time)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("预计总请求数", f"{estimate['total_requests']:,}")
        col2.metric("读请求", f"{estimate['read_requests']:,}")
        col3.metric("写请求", f"{estimate['write_requests']:,}")
        col4.metric("配置数", estimate['num_configs'])

    # 生成按钮
    st.markdown("---")
    st.subheader("🚀 生成流量文件")

    col_gen1, col_gen2 = st.columns([3, 1])

    with col_gen1:
        output_filename = st.text_input(
            "输出文件名",
            value=f"traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

    with col_gen2:
        st.write("")  # 占位
        st.write("")  # 占位

    if st.button("🚀 生成流量文件", type="primary", use_container_width=True):
        if not configs:
            st.error("❌ 请先添加至少一个配置!")
        else:
            # 生成流量
            with st.spinner("正在生成流量文件..."):
                # 输出路径
                output_dir = project_root / "traffic"
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / output_filename

                # 转换配置为字典格式
                config_dicts = [config.to_dict() for config in configs]

                # 生成流量
                file_path, df = generate_traffic_from_configs(
                    configs=config_dicts,
                    end_time=end_time,
                    output_file=str(output_file),
                    return_dataframe=True
                )

                st.session_state.generated_traffic = df
                st.session_state.last_file_path = file_path

            st.success(f"✅ 流量文件生成成功: {file_path}")

            # 提供下载按钮
            with open(file_path, 'r') as f:
                st.download_button(
                    label="📥 下载流量文件",
                    data=f.read(),
                    file_name=output_filename,
                    mime="text/plain"
                )

    # 结果分析
    if st.session_state.generated_traffic is not None:
        st.markdown("---")
        st.subheader("📈 结果分析")

        analyzer = TrafficAnalyzer()
        analyzer.load_dataframe(st.session_state.generated_traffic)

        # 显示统计表格
        st.markdown("##### 统计摘要")
        stats_df = analyzer.export_statistics_table()
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # 图表展示
        tab1, tab2, tab3, tab4 = st.tabs(["时间序列", "读写分布", "热力图", "数据预览"])

        with tab1:
            fig_time = analyzer.plot_time_series()
            st.plotly_chart(fig_time, use_container_width=True)

        with tab2:
            fig_req = analyzer.plot_req_type_distribution()
            st.plotly_chart(fig_req, use_container_width=True)

        with tab3:
            fig_heatmap = analyzer.plot_heatmap()
            st.plotly_chart(fig_heatmap, use_container_width=True)

        with tab4:
            st.markdown("##### 前100条数据预览")
            preview_df = analyzer.get_preview_dataframe(100)
            st.dataframe(preview_df, use_container_width=True)


# ==================== 主程序入口 ====================

def main():
    """主程序入口"""
    init_session_state()
    render_main_ui()


if __name__ == "__main__":
    main()
