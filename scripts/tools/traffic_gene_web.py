"""
CrossRing 流量生成Web可视化工具

基于Streamlit的交互式流量生成工具,提供:
- 拓扑可视化与交互式节点选择
- 配置管理与参数验证
- 流量生成与结果分析
"""

import streamlit as st
import sys
import json
from pathlib import Path
from datetime import datetime

# 配置页面，减少加载闪烁
st.set_page_config(page_title="数据流生成工具", page_icon="🗺️", layout="wide", initial_sidebar_state="expanded")  # 保持展开，避免来回跳动

# 添加项目路径到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.traffic_process.traffic_gene.topology_visualizer import TopologyVisualizer, get_default_ip_mappings
from src.traffic_process.traffic_gene.config_manager import ConfigManager, TrafficConfig
from src.traffic_process.traffic_gene.traffic_analyzer import TrafficAnalyzer
from src.traffic_process.traffic_gene.generation_engine import generate_traffic_from_configs, generate_d2d_traffic_from_configs, split_traffic_by_source


# ==================== 页面配置 ====================

st.set_page_config(page_title="数据流生成工具", layout="wide", initial_sidebar_state="expanded")


# ==================== 会话状态初始化 ====================


def init_session_state():
    """初始化会话状态"""
    if "topo_type" not in st.session_state:
        st.session_state.topo_type = "5x4"

    if "traffic_mode" not in st.session_state:
        st.session_state.traffic_mode = "单Die"

    # 当前要挂载的IP
    if "current_ip" not in st.session_state:
        st.session_state.current_ip = ""

    # 存储每个节点挂载的IP列表: {node_id: [ip_list]}
    if "node_ips" not in st.session_state:
        st.session_state.node_ips = {}

    if "config_manager" not in st.session_state:
        rows, cols = map(int, st.session_state.topo_type.split("x"))
        num_nodes = rows * cols
        st.session_state.config_manager = ConfigManager(num_nodes)

    if "generated_traffic" not in st.session_state:
        st.session_state.generated_traffic = None

    if "last_file_path" not in st.session_state:
        st.session_state.last_file_path = None

    if "split_result" not in st.session_state:
        st.session_state.split_result = None

    if "output_filename" not in st.session_state:
        st.session_state.output_filename = f"traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


# ==================== 辅助函数 ====================


def handle_node_click(node_id: int):
    """
    处理节点点击事件 - 挂载当前IP到节点

    :param node_id: 节点ID
    """
    current_ip = st.session_state.current_ip.strip()

    if not current_ip:
        return  # 不显示错误,只是不操作

    # 挂载IP到节点
    if node_id not in st.session_state.node_ips:
        st.session_state.node_ips[node_id] = []

    # 避免重复挂载
    if current_ip not in st.session_state.node_ips[node_id]:
        st.session_state.node_ips[node_id].append(current_ip)
        st.success(f"✅ {current_ip} 已挂载到节点 {node_id}", icon="✅")


# ==================== 主界面 ====================


def render_main_ui():
    """渲染主界面"""
    # 标题
    st.title("数据流生成可视化工具")
    st.markdown("---")

    # 侧边栏 - 全局配置
    with st.sidebar:
        st.header("⚙️ 全局配置")

        # 流量模式选择
        traffic_mode = st.selectbox("流量模式", ["单Die", "D2D"], index=0 if st.session_state.traffic_mode == "单Die" else 1, key="traffic_mode_select")

        if traffic_mode != st.session_state.traffic_mode:
            st.session_state.traffic_mode = traffic_mode

        # 拓扑类型选择
        topo_options = ["5x4", "自定义"]
        current_idx = 0
        if st.session_state.topo_type in topo_options:
            current_idx = topo_options.index(st.session_state.topo_type)
        else:
            current_idx = 1  # 自定义

        topo_mode = st.selectbox("拓扑类型", topo_options, index=current_idx, key="topo_type_select")

        # 如果选择自定义，显示行列输入框
        if topo_mode == "自定义":
            col1, col2 = st.columns(2)
            with col1:
                custom_rows = st.number_input("行数", min_value=2, max_value=10, value=5, step=1, key="custom_rows")
            with col2:
                custom_cols = st.number_input("列数", min_value=2, max_value=10, value=4, step=1, key="custom_cols")
            topo_type = f"{custom_rows}x{custom_cols}"
        else:
            topo_type = topo_mode

        # 如果拓扑类型变化,重新初始化
        if topo_type != st.session_state.topo_type:
            st.session_state.topo_type = topo_type
            rows, cols = map(int, topo_type.split("x"))
            num_nodes = rows * cols
            st.session_state.config_manager = ConfigManager(num_nodes)
            st.session_state.selected_src_nodes = set()
            st.session_state.selected_dst_nodes = set()

        # 仿真时长
        end_time = st.number_input("仿真时长 (ns)", min_value=100, max_value=100000, value=6000, step=100)

        st.markdown("---")
        st.markdown("### 📖 使用说明")
        st.markdown(
            """
        1. 选择流量模式(单Die/D2D)
        2. 输入节点ID或点击拓扑图选择
        3. 配置流量参数
        4. 添加到配置列表
        5. 生成流量文件(可选拆分)
        6. 查看结果分析
        """
        )

    # 主区域 - 分为左右两栏
    col_left, col_right = st.columns([1.2, 1])

    # 左栏 - 拓扑可视化
    with col_left:
        st.subheader("🗺️ IP挂载")

        # IP挂载区
        st.markdown("支持格式: 节点ID可以是单个`0`、多个`0,1,2`、范围`0-3`")
        col_ip, col_node, col_btn = st.columns([2, 2, 1])

        with col_ip:
            current_ip = st.text_input("IP名称", value=st.session_state.current_ip, placeholder="如: gdma_0", key="ip_input")
            st.session_state.current_ip = current_ip

        with col_node:
            target_node = st.text_input("节点ID", placeholder="如: 0 或 0,1,2", key="node_input")

        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)  # 垂直对齐
            if st.button("➕ 挂载", use_container_width=True):
                if current_ip.strip() and target_node.strip():
                    # 解析节点ID
                    visualizer = TopologyVisualizer(st.session_state.topo_type, ip_mappings={})
                    try:
                        node_ids = visualizer.parse_node_ids(target_node)
                        mount_count = 0
                        for node_id in node_ids:
                            if node_id not in st.session_state.node_ips:
                                st.session_state.node_ips[node_id] = []
                            if current_ip.strip() not in st.session_state.node_ips[node_id]:
                                st.session_state.node_ips[node_id].append(current_ip.strip())
                                mount_count += 1
                        if mount_count > 0:
                            st.toast(f"✅ {current_ip} 已挂载到 {mount_count} 个节点", icon="✅")
                        else:
                            st.toast(f"ℹ️ {current_ip} 已存在于选中节点", icon="ℹ️")
                    except ValueError as e:
                        st.toast(f"❌ {str(e)}", icon="❌")
                else:
                    st.toast("❌ 请输入IP名称和节点ID", icon="❌")

        # IP挂载管理
        col_save, col_load = st.columns(2)
        with col_save:
            if st.button("💾 保存挂载", use_container_width=True, disabled=not st.session_state.node_ips):
                st.session_state.show_save_dialog = True

        # 保存对话框（不需要rerun，自然刷新）
        if st.session_state.get("show_save_dialog", False):
            st.markdown("##### 💾 保存IP挂载配置")

            save_name = st.text_input("配置名称", placeholder="如: gdma_ddr_test", help="用于标识此配置的名称")

            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("✅ 确认保存", use_container_width=True):
                    if save_name.strip():
                        # 保存到JSON文件
                        save_dir = project_root / "config" / "ip_mounts"
                        save_dir.mkdir(parents=True, exist_ok=True)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # 文件名使用用户输入的名称
                        safe_name = save_name.strip().replace(" ", "_")
                        filename = f"{safe_name}_{st.session_state.topo_type}_{timestamp}.json"
                        save_path = save_dir / filename

                        save_data = {"name": save_name.strip(), "topo_type": st.session_state.topo_type, "node_ips": st.session_state.node_ips, "timestamp": timestamp}

                        with open(save_path, "w", encoding="utf-8") as f:
                            json.dump(save_data, f, indent=2, ensure_ascii=False)

                        st.session_state.show_save_dialog = False
                        st.toast(f"✅ 已保存为 {save_name}", icon="✅")
                        st.rerun()
                    else:
                        st.error("❌ 请输入配置名称")

            with col_cancel:
                if st.button("❌ 取消", use_container_width=True):
                    st.session_state.show_save_dialog = False
                    st.rerun()

            st.markdown("---")

        with col_load:
            # 查找可用的保存文件
            save_dir = project_root / "config" / "ip_mounts"
            if save_dir.exists():
                save_files = sorted(save_dir.glob("*.json"), reverse=True)
                if save_files:
                    if st.button("📂 加载挂载", use_container_width=True):
                        st.session_state.show_load_dialog = True

        # 加载对话框
        if st.session_state.get("show_load_dialog", False):
            save_dir = project_root / "config" / "ip_mounts"
            save_files = sorted(save_dir.glob("*.json"), reverse=True)

            if save_files:
                file_options = {}
                for f in save_files:
                    # 读取文件获取拓扑类型和名称
                    try:
                        with open(f, "r", encoding="utf-8") as fp:
                            data = json.load(fp)
                            name = data.get("name", f.stem)
                            topo = data.get("topo_type", "unknown")
                            timestamp = data.get("timestamp", "")
                            label = f"{name} ({topo}) - {timestamp}"
                            file_options[label] = f
                    except:
                        continue

                if file_options:
                    selected_file = st.selectbox("选择要加载的挂载配置", options=list(file_options.keys()), key="load_file_select")

                    col_confirm, col_delete, col_cancel = st.columns(3)
                    with col_confirm:
                        if st.button("✅ 加载", use_container_width=True):
                            try:
                                load_path = file_options[selected_file]
                                with open(load_path, "r", encoding="utf-8") as f:
                                    data = json.load(f)

                                # 检查拓扑类型是否匹配
                                if data["topo_type"] != st.session_state.topo_type:
                                    st.warning(f"⚠️ 加载的配置是 {data['topo_type']} 拓扑，当前是 {st.session_state.topo_type}")

                                # 获取当前拓扑的最大节点数
                                rows, cols = map(int, st.session_state.topo_type.split("x"))
                                max_node_id = rows * cols - 1

                                # 加载IP挂载数据并检查节点范围
                                node_ips_data = {int(k): v for k, v in data["node_ips"].items()}
                                invalid_nodes = [node_id for node_id in node_ips_data.keys() if node_id > max_node_id]

                                if invalid_nodes:
                                    st.error(f"❌ 加载失败: 节点 {invalid_nodes} 超过当前拓扑最大节点ID {max_node_id}")
                                else:
                                    st.session_state.node_ips = node_ips_data
                                    st.session_state.show_load_dialog = False
                                    st.toast(f"✅ 已加载配置", icon="✅")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ 加载失败: {str(e)}")

                    with col_delete:
                        if st.button("🗑️ 删除", use_container_width=True, type="secondary"):
                            try:
                                load_path = file_options[selected_file]
                                load_path.unlink()  # 删除文件
                                st.toast(f"✅ 已删除配置", icon="✅")
                                # 如果没有文件了，关闭对话框
                                remaining_files = list(save_dir.glob("*.json"))
                                if not remaining_files:
                                    st.session_state.show_load_dialog = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ 删除失败: {str(e)}")

                    with col_cancel:
                        if st.button("❌ 取消", use_container_width=True):
                            st.session_state.show_load_dialog = False
                            st.rerun()

        st.markdown("---")

        # 绘制拓扑图(仅用于显示,不捕获点击)
        visualizer = TopologyVisualizer(st.session_state.topo_type, ip_mappings={})

        fig = visualizer.draw_topology_grid(selected_src=set(), selected_dst=set(), node_ips=st.session_state.node_ips)

        # 显示拓扑图(不捕获点击事件)
        st.plotly_chart(fig, use_container_width=True, key="topology_display")

        # 节点IP管理面板(折叠显示)
        st.markdown("---")
        with st.expander("📋 已挂载IP列表", expanded=False):
            if st.session_state.node_ips:
                # 按IP类型分组显示
                ip_to_nodes = {}
                for node_id, ips in st.session_state.node_ips.items():
                    for ip in ips:
                        if ip not in ip_to_nodes:
                            ip_to_nodes[ip] = []
                        ip_to_nodes[ip].append(node_id)

                for ip in sorted(ip_to_nodes.keys()):
                    nodes = sorted(ip_to_nodes[ip])
                    node_str = ", ".join(map(str, nodes))

                    col_ip, col_del = st.columns([4, 1])
                    with col_ip:
                        st.markdown(f"**{ip}**: 节点 {node_str}")
                    with col_del:
                        if st.button("🗑️", key=f"del_ip_{ip}", use_container_width=True):
                            # 从所有节点中删除该IP
                            for node_id in nodes:
                                if node_id in st.session_state.node_ips:
                                    if ip in st.session_state.node_ips[node_id]:
                                        st.session_state.node_ips[node_id].remove(ip)
                                    # 如果节点没有IP了，删除该节点
                                    if not st.session_state.node_ips[node_id]:
                                        del st.session_state.node_ips[node_id]
                            st.rerun()

                st.markdown("---")
                if st.button("🗑️ 清空所有IP", use_container_width=True):
                    st.session_state.node_ips = {}
                    st.rerun()
            else:
                st.info("暂无挂载的IP")

    # 右栏 - 配置管理
    with col_right:
        st.subheader("⚙️ 数据流配置")

        # 获取已挂载IP的节点列表
        nodes_with_ips = sorted([node for node, ips in st.session_state.node_ips.items() if ips])

        if not nodes_with_ips:
            st.warning("⚠️ 请先在拓扑图中挂载IP到节点")
        else:
            # 配置模式选择(在表单外面,实现实时切换)
            config_mode = st.radio("配置模式", ["具体配置", "批量配置"], horizontal=True, help="具体配置: 精确指定某个节点的IP到另一个节点的IP; 批量配置: 按IP具体配置配置(如所有gdma到所有ddr)")

            # 配置表单
            with st.form("config_form"):

                # D2D模式显示DIE选择
                if st.session_state.traffic_mode == "D2D":
                    col_die1, col_die2 = st.columns(2)
                    with col_die1:
                        src_die = st.number_input("源Die编号", min_value=0, max_value=3, value=0, step=1)
                    with col_die2:
                        dst_die = st.number_input("目标Die编号", min_value=0, max_value=3, value=1, step=1)

                st.markdown("---")

                if config_mode == "具体配置":
                    # 模式1: 具体配置 - 直接选择"节点X的IP_Y"
                    # 构建选项: {显示文本: (node_id, ip)}
                    src_options = {}
                    for node_id in sorted(st.session_state.node_ips.keys()):
                        for ip in sorted(st.session_state.node_ips[node_id]):
                            label = f"节点{node_id} - {ip}"
                            src_options[label] = (node_id, ip)

                    st.write("**源IP (可多选):**")
                    selected_src_labels = st.multiselect("选择源IP", options=list(src_options.keys()), default=[], label_visibility="collapsed")

                    st.markdown("---")

                    st.write("**目标IP (可多选):**")
                    selected_dst_labels = st.multiselect("选择目标IP", options=list(src_options.keys()), default=[], label_visibility="collapsed")

                else:
                    # 模式2: 具体配置
                    # 提取所有IP类型(去掉下标)
                    all_ip_types = set()
                    for ips in st.session_state.node_ips.values():
                        for ip in ips:
                            ip_type = ip.split("_")[0] if "_" in ip else ip
                            all_ip_types.add(ip_type)

                    st.write("**源IP类型 (可多选):**")
                    src_ip_types = st.multiselect("选择源IP类型", options=sorted(all_ip_types), default=[], label_visibility="collapsed")

                    st.markdown("---")

                    st.write("**目标IP类型 (可多选):**")
                    dst_ip_types = st.multiselect("选择目标IP类型", options=sorted(all_ip_types), default=[], label_visibility="collapsed")

                st.markdown("---")

                speed = st.number_input("带宽 (GB/s)", min_value=0.1, max_value=128.0, value=46.08, step=0.01, format="%.2f")

                burst = st.selectbox("Burst长度", [1, 2, 4, 8, 16], index=2)

                req_type = st.radio("请求类型", ["R", "W"], horizontal=True)

                submit_button = st.form_submit_button("➕ 添加配置", use_container_width=True)

                if submit_button:
                    if config_mode == "具体配置":
                        # 模式1验证 - 具体配置
                        if not selected_src_labels or not selected_dst_labels:
                            st.error("请至少选择一个源IP和一个目标IP!")
                        else:
                            # 解析选择的IP和节点
                            src_map = {}
                            for label in selected_src_labels:
                                node_id, ip = src_options[label]
                                if ip not in src_map:
                                    src_map[ip] = []
                                src_map[ip].append(node_id)

                            dst_map = {}
                            for label in selected_dst_labels:
                                node_id, ip = src_options[label]
                                if ip not in dst_map:
                                    dst_map[ip] = []
                                dst_map[ip].append(node_id)

                            # 创建配置
                            config = TrafficConfig(src_map=src_map, dst_map=dst_map, speed=speed, burst=burst, req_type=req_type)

                            # D2D模式添加Die信息
                            if st.session_state.traffic_mode == "D2D":
                                config.src_die = src_die
                                config.dst_die = dst_die

                            # 添加到配置管理器
                            success, errors = st.session_state.config_manager.add_config(config)

                            if success:
                                st.success("✅ 配置添加成功!")
                                st.rerun()
                            else:
                                st.error("❌ 配置验证失败:\n" + "\n".join(errors))

                    else:
                        # 模式2验证和处理 - 批量配置
                        if not src_ip_types or not dst_ip_types:
                            st.error("请至少选择一个源IP类型和一个目标IP类型!")
                        else:
                            # D2D模式：需要区分Die
                            if st.session_state.traffic_mode == "D2D":
                                # 在D2D模式下，批量配置的含义是：
                                # 源Die的所有指定类型IP → 目标Die的所有指定类型IP
                                # 注意：由于当前所有IP都挂载在同一个Die的拓扑上，
                                # 这里的逻辑是收集所有匹配的IP，然后在生成流量时指定Die信息
                                st.info(f"💡 D2D批量模式: Die{src_die}的所有选中IP类型 → Die{dst_die}的所有选中IP类型")

                            # 收集所有匹配类型的IP和节点
                            src_map = {}
                            for node_id, ips in st.session_state.node_ips.items():
                                for ip in ips:
                                    ip_type = ip.split("_")[0] if "_" in ip else ip
                                    if ip_type in src_ip_types:
                                        if ip not in src_map:
                                            src_map[ip] = []
                                        src_map[ip].append(node_id)

                            dst_map = {}
                            for node_id, ips in st.session_state.node_ips.items():
                                for ip in ips:
                                    ip_type = ip.split("_")[0] if "_" in ip else ip
                                    if ip_type in dst_ip_types:
                                        if ip not in dst_map:
                                            dst_map[ip] = []
                                        dst_map[ip].append(node_id)

                            if not src_map or not dst_map:
                                st.error("未找到匹配的IP!")
                            else:
                                # 创建配置
                                config = TrafficConfig(src_map=src_map, dst_map=dst_map, speed=speed, burst=burst, req_type=req_type)

                                # D2D模式添加Die信息
                                if st.session_state.traffic_mode == "D2D":
                                    config.src_die = src_die
                                    config.dst_die = dst_die

                                # 添加到配置管理器
                                success, errors = st.session_state.config_manager.add_config(config)

                                if success:
                                    st.success("✅ 配置添加成功!")
                                    st.rerun()
                                else:
                                    st.error("❌ 配置验证失败:\n" + "\n".join(errors))

    # 配置列表展示
    st.markdown("---")

    col_title, col_save, col_load = st.columns([3, 1, 1])
    with col_title:
        st.subheader("📋 配置列表")
    with col_save:
        if st.button("💾 保存配置", use_container_width=True, disabled=not st.session_state.config_manager.get_all_configs()):
            st.session_state.show_save_config_dialog = True
            st.rerun()
    with col_load:
        if st.button("📂 加载配置", use_container_width=True):
            st.session_state.show_load_config_dialog = True
            st.rerun()

    # 保存配置对话框
    if st.session_state.get("show_save_config_dialog", False):
        st.markdown("##### 💾 保存流量配置")

        config_name = st.text_input("配置名称", placeholder="如: gdma_to_ddr_test", help="用于标识此配置集的名称")

        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✅ 确认保存", use_container_width=True, key="save_config_confirm"):
                if config_name.strip():
                    # 保存到JSON文件
                    save_dir = project_root / "config" / "traffic_configs"
                    save_dir.mkdir(parents=True, exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_name = config_name.strip().replace(" ", "_")
                    filename = f"{safe_name}_{st.session_state.topo_type}_{timestamp}.json"
                    save_path = save_dir / filename

                    # 导出配置数据
                    configs = st.session_state.config_manager.get_all_configs()
                    configs_data = []
                    for config in configs:
                        config_dict = {
                            "src_map": config.src_map,
                            "dst_map": config.dst_map,
                            "speed": config.speed,
                            "burst": config.burst,
                            "req_type": config.req_type,
                        }
                        if hasattr(config, "src_die"):
                            config_dict["src_die"] = config.src_die
                            config_dict["dst_die"] = config.dst_die
                        configs_data.append(config_dict)

                    save_data = {"name": config_name.strip(), "topo_type": st.session_state.topo_type, "traffic_mode": st.session_state.traffic_mode, "configs": configs_data, "timestamp": timestamp}

                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False)

                    st.session_state.show_save_config_dialog = False
                    st.toast(f"✅ 已保存为 {config_name}", icon="✅")
                    st.rerun()
                else:
                    st.error("❌ 请输入配置名称")

        with col_cancel:
            if st.button("❌ 取消", use_container_width=True, key="save_config_cancel"):
                st.session_state.show_save_config_dialog = False
                st.rerun()

        st.markdown("---")

    # 加载配置对话框
    if st.session_state.get("show_load_config_dialog", False):
        st.markdown("##### 📂 加载流量配置")

        save_dir = project_root / "config" / "traffic_configs"
        if save_dir.exists():
            save_files = sorted(save_dir.glob("*.json"), reverse=True)

            if save_files:
                file_options = {}
                for f in save_files:
                    try:
                        with open(f, "r", encoding="utf-8") as fp:
                            data = json.load(fp)
                            name = data.get("name", f.stem)
                            topo = data.get("topo_type", "unknown")
                            mode = data.get("traffic_mode", "unknown")
                            timestamp = data.get("timestamp", "")
                            label = f"{name} ({topo}, {mode}) - {timestamp}"
                            file_options[label] = f
                    except:
                        continue

                if file_options:
                    selected_file = st.selectbox("选择要加载的配置", options=list(file_options.keys()), key="load_config_select")

                    col_confirm, col_delete, col_cancel = st.columns(3)
                    with col_confirm:
                        if st.button("✅ 加载", use_container_width=True, key="load_config_confirm"):
                            try:
                                load_path = file_options[selected_file]
                                with open(load_path, "r", encoding="utf-8") as f:
                                    data = json.load(f)

                                # 检查拓扑类型
                                if data["topo_type"] != st.session_state.topo_type:
                                    st.warning(f"⚠️ 加载的配置是 {data['topo_type']} 拓扑，当前是 {st.session_state.topo_type}")

                                # 清空现有配置
                                st.session_state.config_manager = ConfigManager(st.session_state.config_manager.num_nodes)

                                # 加载配置
                                for config_dict in data["configs"]:
                                    config = TrafficConfig(
                                        src_map=config_dict["src_map"], dst_map=config_dict["dst_map"], speed=config_dict["speed"], burst=config_dict["burst"], req_type=config_dict["req_type"]
                                    )
                                    if "src_die" in config_dict:
                                        config.src_die = config_dict["src_die"]
                                        config.dst_die = config_dict["dst_die"]

                                    st.session_state.config_manager.add_config(config)

                                st.session_state.show_load_config_dialog = False
                                st.toast(f"✅ 已加载配置", icon="✅")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ 加载失败: {str(e)}")

                    with col_delete:
                        if st.button("🗑️ 删除", use_container_width=True, type="secondary", key="delete_config_btn"):
                            try:
                                load_path = file_options[selected_file]
                                load_path.unlink()
                                st.toast(f"✅ 已删除配置", icon="✅")
                                remaining_files = list(save_dir.glob("*.json"))
                                if not remaining_files:
                                    st.session_state.show_load_config_dialog = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ 删除失败: {str(e)}")

                    with col_cancel:
                        if st.button("❌ 取消", use_container_width=True, key="load_config_cancel"):
                            st.session_state.show_load_config_dialog = False
                            st.rerun()
                else:
                    st.info("暂无保存的配置")
            else:
                st.info("暂无保存的配置")
        else:
            st.info("暂无保存的配置")

        st.markdown("---")

    configs = st.session_state.config_manager.get_all_configs()

    if not configs:
        st.info("暂无配置,请添加流量配置")
    else:
        # 显示配置
        for i, config in enumerate(configs):
            # 智能生成标题
            src_ips = list(config.src_map.keys())
            dst_ips = list(config.dst_map.keys())

            # 检查是否为批量配置（多个IP且同类型）
            def get_ip_summary(ip_list):
                if len(ip_list) == 1:
                    return ip_list[0]
                # 提取IP类型
                ip_types = set()
                for ip in ip_list:
                    ip_type = ip.split("_")[0] if "_" in ip else ip
                    ip_types.add(ip_type)

                if len(ip_types) == 1:
                    # 同类型批量
                    ip_type = list(ip_types)[0]
                    return ip_type.upper()
                else:
                    # 多类型
                    return f"{len(ip_list)}个IP"

            src_summary = get_ip_summary(src_ips)
            dst_summary = get_ip_summary(dst_ips)

            # D2D模式在标题中显示Die信息
            if hasattr(config, "src_die"):
                title = f"配置 #{config.config_id}: Die{config.src_die}:{src_summary} → Die{config.dst_die}:{dst_summary}"
            else:
                title = f"配置 #{config.config_id}: {src_summary} → {dst_summary}"

            with st.expander(title, expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    # 详细显示源IP和节点
                    st.write("**源IP:**")
                    for ip, nodes in config.src_map.items():
                        st.write(f"  • {ip}: 节点 {nodes}")

                    st.write("**目标IP:**")
                    for ip, nodes in config.dst_map.items():
                        st.write(f"  • {ip}: 节点 {nodes}")

                with col2:
                    st.write(f"**带宽**: {config.speed} GB/s")
                    st.write(f"**Burst**: {config.burst}")
                    st.write(f"**类型**: {'读' if config.req_type == 'R' else '写'}")
                    if hasattr(config, "src_die"):
                        st.write(f"**Die**: {config.src_die} → {config.dst_die}")

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
        col4.metric("配置数", estimate["num_configs"])

    # 生成按钮
    st.markdown("---")
    st.subheader("🚀 生成流量文件")

    col_gen1, col_gen2 = st.columns([3, 1])

    with col_gen1:
        output_filename = st.text_input("输出文件名", value=st.session_state.output_filename, key="output_filename_input")
        st.session_state.output_filename = output_filename

    with col_gen2:
        st.write("")  # 占位
        st.write("")  # 占位

    # 流量拆分选项
    enable_split = st.checkbox("拆分流量文件(按源IP)", value=False, help="生成后自动按源IP拆分流量文件")

    if enable_split:
        split_output_dir = st.text_input("拆分输出目录", value="./split_output", help="相对于traffic目录的路径")

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

                # 根据模式生成流量
                if st.session_state.traffic_mode == "D2D":
                    file_path, df = generate_d2d_traffic_from_configs(configs=config_dicts, end_time=end_time, output_file=str(output_file), return_dataframe=True)
                else:
                    file_path, df = generate_traffic_from_configs(configs=config_dicts, end_time=end_time, output_file=str(output_file), return_dataframe=True)

                st.session_state.generated_traffic = df
                st.session_state.last_file_path = file_path

            st.success(f"✅ 流量文件生成成功: {file_path}")

            # 拆分流量文件
            if enable_split and st.session_state.traffic_mode == "单Die":
                with st.spinner("正在拆分流量文件..."):
                    try:
                        # 确定拆分输出目录
                        split_dir = output_dir / split_output_dir

                        # 获取拓扑参数
                        rows, cols = map(int, st.session_state.topo_type.split("x"))

                        # 执行拆分
                        split_result = split_traffic_by_source(input_file=file_path, output_dir=str(split_dir), num_col=cols, num_row=rows, verbose=False)

                        st.session_state.split_result = split_result
                        st.success(f"✅ 流量拆分完成! 输出目录: {split_result['output_dir']}")
                        st.info(f"共生成 {split_result['total_sources']} 个拆分文件")

                    except Exception as e:
                        st.error(f"❌ 拆分失败: {e}")

            # 提供下载按钮
            with open(file_path, "r") as f:
                st.download_button(label="📥 下载流量文件", data=f.read(), file_name=output_filename, mime="text/plain")

    # 显示拆分结果
    if st.session_state.split_result:
        st.markdown("---")
        st.subheader("📁 拆分文件列表")

        split_result = st.session_state.split_result
        st.write(f"**输出目录**: {split_result['output_dir']}")
        st.write(f"**总文件数**: {len(split_result['files'])}")

        # 显示文件列表
        for file_info in split_result["files"][:10]:  # 只显示前10个
            st.text(f"• {file_info['filename']}: {file_info['count']} 条请求")

        if len(split_result["files"]) > 10:
            st.text(f"... 还有 {len(split_result['files']) - 10} 个文件")

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
