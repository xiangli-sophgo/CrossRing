"""
CrossRing 数据流生成Web可视化工具

基于Streamlit的交互式数据流生成工具,提供:
- 拓扑可视化与交互式节点选择
- 配置管理与参数验证
- 数据流生成与结果分析
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
from src.traffic_process.traffic_gene.generation_engine import generate_traffic_from_configs, generate_d2d_traffic_from_configs, split_traffic_by_source, split_d2d_traffic_by_source


# ==================== 页面配置 ====================

st.set_page_config(page_title="数据流生成工具", layout="wide", initial_sidebar_state="expanded")


# ==================== 会话状态初始化 ====================


def init_session_state():
    """初始化会话状态"""
    if "topo_type" not in st.session_state:
        st.session_state.topo_type = "5x4"

    if "traffic_mode" not in st.session_state:
        st.session_state.traffic_mode = "NoC"

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


def parse_die_pair(die_pair_str):
    """
    解析Die对字符串
    :param die_pair_str: "Die0 → Die1"
    :return: (src_die, dst_die)
    """
    import re

    match = re.match(r"Die(\d+)\s*→\s*Die(\d+)", die_pair_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"无效的Die对格式: {die_pair_str}")


def generate_die_pair_options(max_die=4, include_self=True):
    """
    生成Die对选项列表
    :param max_die: 最大Die编号+1（默认4个Die: 0-3）
    :param include_self: 是否包含自环(Die到自己)
    :return: Die对字符串列表
    """
    options = []
    for src in range(max_die):
        for dst in range(max_die):
            if include_self or src != dst:
                options.append(f"Die{src} → Die{dst}")
    return options


def load_die_templates():
    """加载Die配置模板"""
    template_dir = project_root / "config" / "die_templates"
    if not template_dir.exists():
        return get_builtin_die_templates()

    templates = get_builtin_die_templates()

    # 加载用户自定义模板
    for template_file in template_dir.glob("*.json"):
        try:
            with open(template_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                name = data.get("name", template_file.stem)
                die_pairs = data.get("die_pairs", [])
                # 转换为字符串列表
                pair_strs = [f"Die{p['src_die']} → Die{p['dst_die']}" for p in die_pairs]
                templates[name] = pair_strs
        except Exception:
            continue

    return templates


def get_builtin_die_templates():
    """获取内置Die模板"""
    return {
        "Die0→其他": ["Die0 → Die1", "Die0 → Die2", "Die0 → Die3"],
        "Die0→All": ["Die0 → Die0", "Die0 → Die1", "Die0 → Die2", "Die0 → Die3"],
        "Die1→其他": ["Die1 → Die0", "Die1 → Die2", "Die1 → Die3"],
        "Die1→All": ["Die1 → Die0", "Die1 → Die1", "Die1 → Die2", "Die1 → Die3"],
        "Die2→其他": ["Die2 → Die0", "Die2 → Die1", "Die2 → Die3"],
        "Die2→All": ["Die2 → Die0", "Die2 → Die1", "Die2 → Die2", "Die2 → Die3"],
        "Die3→其他": ["Die3 → Die0", "Die3 → Die1", "Die3 → Die2"],
        "Die3→All": ["Die3 → Die0", "Die3 → Die1", "Die3 → Die2", "Die3 → Die3"],
        "全连接": generate_die_pair_options(4),
    }


def save_die_template(name, die_pairs):
    """保存Die配置模板"""
    template_dir = project_root / "config" / "die_templates"
    template_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.strip().replace(" ", "_")
    filename = f"{safe_name}_{timestamp}.json"
    save_path = template_dir / filename

    # 转换为字典格式
    die_pair_dicts = []
    for pair_str in die_pairs:
        src_die, dst_die = parse_die_pair(pair_str)
        die_pair_dicts.append({"src_die": src_die, "dst_die": dst_die})

    save_data = {"name": name, "die_pairs": die_pair_dicts, "timestamp": timestamp}

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    return save_path


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
    # 标题（带使用说明）
    col_title, col_help = st.columns([4, 1])
    with col_title:
        st.title("数据流生成可视化工具")
    with col_help:
        with st.expander("📖 使用说明"):
            st.markdown(
                """
            1. 选择拓扑类型和数据流模式
            2. 挂载IP到节点
            3. 配置数据流参数
            4. 添加到配置列表
            5. 生成数据流文件(可选拆分)
            6. 查看结果分析
            """
            )
    st.markdown("---")

    # 主区域 - 分为左右两栏
    col_left, col_right = st.columns([1, 1.5])

    # 左栏 - 拓扑可视化
    with col_left:
        st.subheader("🗺️ IP挂载")

        # 拓扑类型输入
        topo_input = st.text_input("拓扑类型", value=st.session_state.topo_type, placeholder="如: 5x4, 4X3, 4,3", help="支持格式: 5x4, 5X4, 5,4 等", key="topo_type_input")

        # 解析拓扑类型输入
        def parse_topology(input_str):
            """解析拓扑类型输入，支持多种格式"""
            import re

            # 移除空格
            input_str = input_str.strip().replace(" ", "")

            # 尝试匹配 数字x数字 或 数字,数字 格式（不区分大小写）
            match = re.match(r"^(\d+)[xX,](\d+)$", input_str)
            if match:
                rows = int(match.group(1))
                cols = int(match.group(2))
                if 2 <= rows <= 10 and 2 <= cols <= 10:
                    return f"{rows}x{cols}"
            return None

        # 验证并更新拓扑类型
        if topo_input:
            parsed_topo = parse_topology(topo_input)
            if parsed_topo:
                # 如果拓扑类型变化,重新初始化
                if parsed_topo != st.session_state.topo_type:
                    st.session_state.topo_type = parsed_topo
                    rows, cols = map(int, parsed_topo.split("x"))
                    num_nodes = rows * cols
                    st.session_state.config_manager = ConfigManager(num_nodes)
                    st.session_state.selected_src_nodes = set()
                    st.session_state.selected_dst_nodes = set()
                    st.rerun()
            else:
                st.error("❌ 拓扑格式错误，请使用如 5x4, 4X3, 4,3 等格式（行列范围: 2-10）")

        st.markdown("---")

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

        # 配置列表（紧凑显示在IP列表下方）
        st.markdown("---")

        configs = st.session_state.config_manager.get_all_configs()

        if configs:
            col_title, col_save, col_load = st.columns([2, 1, 1])
            with col_title:
                st.markdown("**📋 配置列表**")
            with col_save:
                if st.button("💾", key="save_cfg_left", use_container_width=True, help="保存配置"):
                    st.session_state.show_save_config_dialog = True
                    st.rerun()
            with col_load:
                if st.button("📂", key="load_cfg_left", use_container_width=True, help="加载配置"):
                    st.session_state.show_load_config_dialog = True
                    st.rerun()

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

            # D2D模式：按源Die分组，横向紧凑显示
            if st.session_state.traffic_mode == "D2D":
                # 按源Die分组
                src_die_groups = {}
                for config in configs:
                    if hasattr(config, "src_die"):
                        src_die_key = f"Die{config.src_die}"
                        if src_die_key not in src_die_groups:
                            src_die_groups[src_die_key] = []
                        src_die_groups[src_die_key].append(config)
                    else:
                        if "其他" not in src_die_groups:
                            src_die_groups["其他"] = []
                        src_die_groups["其他"].append(config)

                # 分组显示
                for src_die, group_configs in src_die_groups.items():
                    st.markdown(f"**{src_die}** ({len(group_configs)}个)")

                    # 横向显示该组所有配置 - 每行最多4个
                    for i in range(0, len(group_configs), 4):
                        cols = st.columns(4)
                        for j in range(4):
                            if i + j < len(group_configs):
                                config = group_configs[i + j]
                                src_summary = get_ip_summary(list(config.src_map.keys()))
                                dst_summary = get_ip_summary(list(config.dst_map.keys()))

                                with cols[j]:
                                    with st.container(border=True):
                                        st.markdown(f"**#{config.config_id}** Die{config.src_die}→Die{config.dst_die}")
                                        st.caption(f"{src_summary} → {dst_summary}")
                                        st.caption(f"时长: {config.end_time}ns")
                                        st.caption(f"带宽: {config.speed}GB/s | Burst: {config.burst}")
                                        st.caption(f"类型: {'读' if config.req_type == 'R' else '写'}")
                                        if st.button("删除", key=f"del_{config.config_id}", use_container_width=True):
                                            st.session_state.config_manager.remove_config(config.config_id)
                                            st.rerun()
            else:
                # NoC模式：横向紧凑显示，每行4个
                for i in range(0, len(configs), 4):
                    cols = st.columns(4)
                    for j in range(4):
                        if i + j < len(configs):
                            config = configs[i + j]
                            src_summary = get_ip_summary(list(config.src_map.keys()))
                            dst_summary = get_ip_summary(list(config.dst_map.keys()))

                            with cols[j]:
                                with st.container(border=True):
                                    st.markdown(f"**#{config.config_id}** {src_summary} → {dst_summary}")
                                    st.caption(f"时长: {config.end_time}ns")
                                    st.caption(f"带宽: {config.speed}GB/s | Burst: {config.burst}")
                                    st.caption(f"类型: {'读' if config.req_type == 'R' else '写'}")
                                    if st.button("删除", key=f"del_{config.config_id}", use_container_width=True):
                                        st.session_state.config_manager.remove_config(config.config_id)
                                        st.rerun()

    # 右栏 - 配置管理和配置列表
    with col_right:
        st.subheader("⚙️ 数据流配置")

        # 数据流模式选择
        traffic_mode = st.selectbox("数据流模式", ["NoC", "D2D"], index=0 if st.session_state.traffic_mode == "NoC" else 1, key="traffic_mode_select")
        if traffic_mode != st.session_state.traffic_mode:
            st.session_state.traffic_mode = traffic_mode

        st.markdown("---")

        # 获取已挂载IP的节点列表
        nodes_with_ips = sorted([node for node, ips in st.session_state.node_ips.items() if ips])

        if not nodes_with_ips:
            st.warning("⚠️ 请先在拓扑图中挂载IP到节点")
        else:
            # 配置模式选择(在表单外面,实现实时切换)
            config_mode = st.radio(
                "配置模式", ["具体配置", "批量配置"], horizontal=True, help="具体配置: 精确指定某个节点的IP到另一个节点的IP; 批量配置: 按IP具体配置配置(如所有gdma到所有ddr)"
            )

            # D2D模式的Die对选择(移到form外面,实现实时更新)
            if st.session_state.traffic_mode == "D2D":
                st.write("**Die对配置 (可多选):**")

                # 初始化session state
                if "last_selected_template" not in st.session_state:
                    st.session_state.last_selected_template = "自定义"
                if "selected_die_pairs" not in st.session_state:
                    st.session_state.selected_die_pairs = []

                # 模板快捷选择(在form外面)
                die_templates = load_die_templates()
                template_names = ["自定义"] + list(die_templates.keys())
                selected_template = st.selectbox("快速模板", options=template_names, key="die_template_select")

                # 如果模板变化，更新Die对列表并触发重新运行
                if selected_template != st.session_state.last_selected_template:
                    st.session_state.last_selected_template = selected_template
                    if selected_template != "自定义":
                        st.session_state.selected_die_pairs = die_templates[selected_template]
                    else:
                        # 切换到自定义时,清空选择
                        st.session_state.selected_die_pairs = []
                    st.rerun()

                # Die对多选
                die_pair_options = generate_die_pair_options(4)

                selected_die_pairs = st.multiselect(
                    "选择Die对", options=die_pair_options, default=st.session_state.selected_die_pairs, label_visibility="collapsed", key="die_pairs_multiselect"
                )

                # 更新session state
                st.session_state.selected_die_pairs = selected_die_pairs

                st.markdown("---")

            # 配置表单
            with st.form("config_form"):

                if config_mode == "具体配置":
                    # 模式1: 具体配置 - 直接选择"节点X的IP_Y"
                    # 定义源IP和目标IP类型判断函数
                    def is_src_type(ip_type):
                        # 包含dma或rn即为源IP
                        ip_lower = ip_type.lower()
                        return "dma" in ip_lower or "rn" in ip_lower

                    def is_dst_type(ip_type):
                        return ip_type.lower() in {"ddr", "l2m"}

                    # 构建源IP选项 - 包含dma或rn
                    src_options = {}
                    for node_id in sorted(st.session_state.node_ips.keys()):
                        for ip in sorted(st.session_state.node_ips[node_id]):
                            ip_type = ip.split("_")[0] if "_" in ip else ip
                            if is_src_type(ip_type):
                                label = f"节点{node_id} - {ip}"
                                src_options[label] = (node_id, ip)

                    # 构建目标IP选项 - 只包含目标IP类型
                    dst_options = {}
                    for node_id in sorted(st.session_state.node_ips.keys()):
                        for ip in sorted(st.session_state.node_ips[node_id]):
                            ip_type = ip.split("_")[0] if "_" in ip else ip
                            if is_dst_type(ip_type):
                                label = f"节点{node_id} - {ip}"
                                dst_options[label] = (node_id, ip)

                    st.write("**源IP (可多选):**")
                    selected_src_labels = st.multiselect("选择源IP", options=list(src_options.keys()), default=[], label_visibility="collapsed")

                    st.write("**目标IP (可多选):**")
                    selected_dst_labels = st.multiselect("选择目标IP", options=list(dst_options.keys()), default=[], label_visibility="collapsed")

                else:
                    # 模式2: 批量配置
                    # 提取所有IP类型(去掉下标)
                    all_ip_types = set()
                    for ips in st.session_state.node_ips.values():
                        for ip in ips:
                            ip_type = ip.split("_")[0] if "_" in ip else ip
                            all_ip_types.add(ip_type)

                    # 定义过滤函数 - 包含dma或rn即为源IP
                    def is_src_type(ip_type):
                        ip_lower = ip_type.lower()
                        return "dma" in ip_lower or "rn" in ip_lower

                    def is_dst_type(ip_type):
                        return ip_type.lower() in {"ddr", "l2m"}

                    # 过滤源IP类型 - 包含dma或rn的都算
                    src_ip_options = sorted([ip_type for ip_type in all_ip_types if is_src_type(ip_type)])
                    # 过滤目标IP类型
                    dst_ip_options = sorted([ip_type for ip_type in all_ip_types if is_dst_type(ip_type)])

                    st.write("**源IP类型 (可多选):**")
                    src_ip_types = st.multiselect("选择源IP类型", options=src_ip_options, default=[], label_visibility="collapsed")

                    st.write("**目标IP类型 (可多选):**")
                    dst_ip_types = st.multiselect("选择目标IP类型", options=dst_ip_options, default=[], label_visibility="collapsed")

                st.markdown("---")

                # 参数配置 - 第一行：仿真时长、带宽、Burst
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    end_time = st.number_input("仿真时长 (ns)", min_value=100, max_value=100000, value=6000, step=100)
                with col_p2:
                    speed = st.number_input("IP带宽 (GB/s)", min_value=0.1, max_value=128.0, value=128.0, step=0.01, format="%.2f")
                with col_p3:
                    burst = st.number_input("Burst长度", min_value=1, max_value=64, value=4, step=1)

                # 第二行：请求类型
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
                                node_id, ip = dst_options[label]
                                if ip not in dst_map:
                                    dst_map[ip] = []
                                dst_map[ip].append(node_id)

                            # D2D模式：批量创建多个Die对配置
                            if st.session_state.traffic_mode == "D2D":
                                if not selected_die_pairs:
                                    st.error("请至少选择一个Die对!")
                                else:
                                    success_count = 0
                                    error_messages = []

                                    for die_pair in selected_die_pairs:
                                        # 解析Die对
                                        src_die, dst_die = parse_die_pair(die_pair)

                                        # 创建配置
                                        config = TrafficConfig(src_map=src_map, dst_map=dst_map, speed=speed, burst=burst, req_type=req_type, end_time=end_time)
                                        config.src_die = src_die
                                        config.dst_die = dst_die

                                        # 添加到配置管理器
                                        success, errors = st.session_state.config_manager.add_config(config)

                                        if success:
                                            success_count += 1
                                        else:
                                            error_messages.extend([f"{die_pair}: {e}" for e in errors])

                                    if success_count > 0:
                                        st.success(f"✅ 成功添加 {success_count} 个配置!")
                                        if error_messages:
                                            st.warning("⚠️ 部分配置失败:\n" + "\n".join(error_messages))
                                        st.rerun()
                                    else:
                                        st.error("❌ 所有配置验证失败:\n" + "\n".join(error_messages))
                            else:
                                # NoC模式：单个配置
                                config = TrafficConfig(src_map=src_map, dst_map=dst_map, speed=speed, burst=burst, req_type=req_type, end_time=end_time)

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
                                # D2D模式：批量创建多个Die对配置
                                if st.session_state.traffic_mode == "D2D":
                                    if not selected_die_pairs:
                                        st.error("请至少选择一个Die对!")
                                    else:
                                        success_count = 0
                                        error_messages = []

                                        for die_pair in selected_die_pairs:
                                            # 解析Die对
                                            src_die, dst_die = parse_die_pair(die_pair)

                                            # 创建配置
                                            config = TrafficConfig(src_map=src_map, dst_map=dst_map, speed=speed, burst=burst, req_type=req_type, end_time=end_time)
                                            config.src_die = src_die
                                            config.dst_die = dst_die

                                            # 添加到配置管理器
                                            success, errors = st.session_state.config_manager.add_config(config)

                                            if success:
                                                success_count += 1
                                            else:
                                                error_messages.extend([f"{die_pair}: {e}" for e in errors])

                                        if success_count > 0:
                                            st.success(f"✅ 成功添加 {success_count} 个配置!")
                                            if error_messages:
                                                st.warning("⚠️ 部分配置失败:\n" + "\n".join(error_messages))
                                            st.rerun()
                                        else:
                                            st.error("❌ 所有配置验证失败:\n" + "\n".join(error_messages))
                                else:
                                    # NoC模式：单个配置
                                    config = TrafficConfig(src_map=src_map, dst_map=dst_map, speed=speed, burst=burst, req_type=req_type, end_time=end_time)

                                    # 添加到配置管理器
                                    success, errors = st.session_state.config_manager.add_config(config)

                                    if success:
                                        st.success("✅ 配置添加成功!")
                                        st.rerun()
                                    else:
                                        st.error("❌ 配置验证失败:\n" + "\n".join(errors))

        # 配置保存/加载对话框（在右栏内）
        st.markdown("---")

        # 保存配置对话框
        if st.session_state.get("show_save_config_dialog", False):
            st.markdown("##### 💾 保存数据流配置")

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
                                "end_time": config.end_time,
                            }
                            if hasattr(config, "src_die"):
                                config_dict["src_die"] = config.src_die
                                config_dict["dst_die"] = config.dst_die
                            configs_data.append(config_dict)

                        save_data = {
                            "name": config_name.strip(),
                            "topo_type": st.session_state.topo_type,
                            "traffic_mode": st.session_state.traffic_mode,
                            "configs": configs_data,
                            "timestamp": timestamp,
                        }

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
            st.markdown("##### 📂 加载数据流配置")

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
                                            src_map=config_dict["src_map"],
                                            dst_map=config_dict["dst_map"],
                                            speed=config_dict["speed"],
                                            burst=config_dict["burst"],
                                            req_type=config_dict["req_type"],
                                            end_time=config_dict.get("end_time", 6000),  # 兼容旧配置
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

        # 预估统计（移到右栏）
        configs = st.session_state.config_manager.get_all_configs()
        if configs:
            st.subheader("📊 预估统计")

            # 计算所有配置的总预估（每个配置使用自己的end_time）
            total_requests = 0
            read_requests = 0
            write_requests = 0

            for config in configs:
                config_end_time = config.end_time
                # 为每个配置单独估算
                single_estimate = st.session_state.config_manager.estimator.estimate_single_config(config, config_end_time)
                total_requests += single_estimate["total_requests"]
                if config.req_type == "R":
                    read_requests += single_estimate["total_requests"]
                else:
                    write_requests += single_estimate["total_requests"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("预计总请求数", f"{total_requests:,}")
            col2.metric("读请求", f"{read_requests:,}")
            col3.metric("写请求", f"{write_requests:,}")
            col4.metric("配置数", len(configs))

    # 生成按钮
    st.markdown("---")
    st.subheader("🚀 生成数据流文件")

    col_gen1, col_gen2 = st.columns([3, 1])

    with col_gen1:
        output_filename = st.text_input("输出文件名", value=st.session_state.output_filename, key="output_filename_input")
        st.session_state.output_filename = output_filename

    with col_gen2:
        st.write("")  # 占位
        st.write("")  # 占位

    # 数据流拆分选项
    enable_split = st.checkbox("拆分数据流文件(按源IP)", value=False, help="生成后自动按源IP拆分数据流文件,输出目录为输出文件名(去掉.txt)")

    if st.button("🚀 生成数据流文件", type="primary", use_container_width=True):
        if not configs:
            st.error("❌ 请先添加至少一个配置!")
        else:
            # 生成数据流
            with st.spinner("正在生成数据流文件..."):
                # 输出路径
                output_dir = project_root / "traffic"
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / output_filename

                # 转换配置为字典格式
                config_dicts = [config.to_dict() for config in configs]

                # 注意：现在使用每个配置自己的end_time
                # 我们传入一个虚拟的end_time（将被忽略），实际使用config.end_time
                # 生成引擎会使用配置中的end_time
                if st.session_state.traffic_mode == "D2D":
                    file_path, df = generate_d2d_traffic_from_configs(configs=config_dicts, end_time=None, output_file=str(output_file), return_dataframe=True)
                else:
                    file_path, df = generate_traffic_from_configs(configs=config_dicts, end_time=None, output_file=str(output_file), return_dataframe=True)

                st.session_state.generated_traffic = df
                st.session_state.last_file_path = file_path

            st.success(f"✅ 数据流文件生成成功: {file_path}")

            # 拆分数据流文件
            if enable_split:
                with st.spinner("正在拆分数据流文件..."):
                    try:
                        # 根据输出文件名生成拆分目录 - 去掉.txt后缀
                        base_name = output_filename.replace(".txt", "")
                        split_dir = output_dir / base_name

                        # 获取拓扑参数
                        rows, cols = map(int, st.session_state.topo_type.split("x"))

                        # 根据模式选择拆分函数
                        if st.session_state.traffic_mode == "D2D":
                            split_result = split_d2d_traffic_by_source(input_file=file_path, output_dir=str(split_dir), num_col=cols, num_row=rows, verbose=False)
                        else:
                            split_result = split_traffic_by_source(input_file=file_path, output_dir=str(split_dir), num_col=cols, num_row=rows, verbose=False)

                        st.session_state.split_result = split_result
                        st.success(f"✅ 数据流拆分完成! 输出目录: {split_result['output_dir']}")
                        st.info(f"共生成 {split_result['total_sources']} 个拆分文件")

                    except Exception as e:
                        st.error(f"❌ 拆分失败: {e}")

            # 提供下载按钮
            with open(file_path, "r") as f:
                st.download_button(label="📥 下载数据流文件", data=f.read(), file_name=output_filename, mime="text/plain")


# ==================== 主程序入口 ====================


def main():
    """主程序入口"""
    init_session_state()
    render_main_ui()


if __name__ == "__main__":
    main()
