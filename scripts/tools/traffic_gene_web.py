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

# 添加项目路径到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.traffic_process.traffic_gene.topology_visualizer import TopologyVisualizer, get_default_ip_mappings
from src.traffic_process.traffic_gene.config_manager import ConfigManager, TrafficConfig
from src.traffic_process.traffic_gene.traffic_analyzer import TrafficAnalyzer
from src.traffic_process.traffic_gene.generation_engine import (
    generate_traffic_from_configs,
    generate_d2d_traffic_from_configs,
    split_traffic_by_source,
    split_d2d_traffic_by_source,
)


# ==================== 页面配置 ====================

st.set_page_config(page_title="数据流生成工具", layout="wide", initial_sidebar_state="expanded")


# ==================== UI配置常量 ====================


class UIConfig:
    """UI配置常量 - Mac风格设计"""

    # 布局比例
    MAIN_COLS_RATIO = [1.2, 1.8]
    IP_MOUNT_COLS_RATIO = [2.5, 2.5, 1]
    PARAM_COLS_RATIO = [1, 1, 1]
    BTN_ROW_RATIO = [1.5, 1.5, 7]

    # 配置卡片
    CARDS_PER_ROW = 4

    # 拓扑范围
    TOPO_MIN = 2
    TOPO_MAX = 10

    # 默认参数值
    DEFAULT_END_TIME = 6000
    DEFAULT_SPEED = 128.0
    DEFAULT_BURST = 4

    # 参数范围
    END_TIME_RANGE = (100, 100000, 100)
    SPEED_RANGE = (0.1, 128.0, 0.01)
    BURST_RANGE = (1, 64, 1)


def load_custom_css():
    """加载Mac风格CSS样式"""
    st.markdown(
        """
        <style>
        /* Mac风格全局设置 */
        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');

        :root {
            --primary-color: #007AFF;
            --success-color: #34C759;
            --warning-color: #FF9500;
            --danger-color: #FF3B30;
            --gray-1: #F5F5F7;
            --gray-2: #E5E5EA;
            --gray-3: #D1D1D6;
            --gray-4: #8E8E93;
            --gray-5: #48484A;
            --text-primary: #1D1D1F;
            --text-secondary: #6E6E73;
            --border-radius: 12px;
            --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.04);
            --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.08);
            --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.12);
        }

        /* 主容器 */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }

        /* 优雅分隔线 */
        .section-divider {
            height: 1px;
            background: linear-gradient(to right, transparent, var(--gray-2) 20%, var(--gray-2) 80%, transparent);
            margin: 2.5rem 0;
        }

        .mini-divider {
            height: 1px;
            background: var(--gray-2);
            margin: 1.5rem 0;
        }

        /* 标题样式 */
        h1, h2, h3 {
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.02em;
        }

        h1 {
            font-size: 2.5rem !important;
            margin-bottom: 0.5rem !important;
        }

        h2 {
            font-size: 1.75rem !important;
            margin-top: 2rem !important;
        }

        h3 {
            font-size: 1.25rem !important;
            margin-top: 1.5rem !important;
        }

        /* 副标题 */
        .subtitle {
            font-size: 1.1rem;
            color: var(--text-secondary);
            font-weight: 400;
            margin-top: 0.5rem;
        }

        /* 配置卡片 - Mac风格 */
        .stContainer > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
            background: white;
            border: 1px solid var(--gray-2);
            border-radius: var(--border-radius);
            padding: 1.25rem;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: var(--shadow-sm);
        }

        .stContainer > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:hover {
            border-color: var(--primary-color);
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }

        /* 按钮样式 - Mac风格 */
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 500 !important;
            padding: 0.5rem 1rem !important;
            border: none !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            letter-spacing: -0.01em !important;
        }

        .stButton > button[kind="primary"] {
            background-color: var(--primary-color) !important;
            background: var(--primary-color) !important;
            color: white !important;
        }

        .stButton > button[kind="primary"]:hover {
            background-color: #0051D5 !important;
            background: #0051D5 !important;
            box-shadow: var(--shadow-md) !important;
        }

        /* 强制覆盖Streamlit默认主题色 */
        button[data-testid="baseButton-primary"] {
            background-color: #007AFF !important;
            background: #007AFF !important;
        }

        button[data-testid="baseButton-primary"]:hover {
            background-color: #0051D5 !important;
            background: #0051D5 !important;
        }

        .stButton > button[kind="secondary"] {
            background: var(--gray-1) !important;
            color: var(--text-primary) !important;
        }

        .stButton > button[kind="secondary"]:hover {
            background: var(--gray-2) !important;
        }

        /* 输入框样式 */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div {
            border-radius: 8px !important;
            border: 1px solid var(--gray-3) !important;
            padding: 0.5rem 0.75rem !important;
            font-size: 0.95rem !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1) !important;
        }

        /* 多选框样式 */
        .stMultiSelect [data-baseweb="tag"] {
            background-color: var(--primary-color) !important;
            border-radius: 6px !important;
        }

        /* Toast通知样式 */
        .stToast {
            border-radius: 12px !important;
            box-shadow: var(--shadow-lg) !important;
        }

        /* 对话框容器 */
        .dialog-container {
            background: var(--gray-1);
            border-left: 3px solid var(--primary-color);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            margin: 1.5rem 0;
            box-shadow: var(--shadow-sm);
        }

        /* 统计卡片 */
        .stat-card {
            background: linear-gradient(135deg, var(--primary-color) 0%, #5856D6 100%);
            color: white;
            padding: 1.25rem;
            border-radius: var(--border-radius);
            text-align: center;
            box-shadow: var(--shadow-md);
        }

        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 0.25rem;
        }

        .stat-label {
            font-size: 0.9rem;
            opacity: 0.9;
            font-weight: 400;
        }

        /* 帮助文本 */
        .help-text {
            color: var(--text-secondary);
            font-size: 0.9rem;
            line-height: 1.6;
        }

        /* 隐藏Streamlit默认元素 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Expander样式 */
        .streamlit-expanderHeader {
            border-radius: 8px !important;
            background-color: var(--gray-1) !important;
            font-weight: 500 !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


# ==================== 分隔符组件 ====================


def section_divider():
    """主分节分隔线"""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def mini_divider():
    """次要分隔线"""
    st.markdown('<div class="mini-divider"></div>', unsafe_allow_html=True)


# ==================== 反馈函数 ====================


def show_success(message: str):
    """显示成功提示"""
    st.toast(message, icon="✅")


def show_error(message: str):
    """显示错误提示"""
    st.toast(message, icon="❌")


def show_info(message: str):
    """显示信息提示"""
    st.toast(message, icon="ℹ️")


def show_warning(message: str):
    """显示警告提示"""
    st.toast(message, icon="⚠️")


# ==================== IP类型判断函数 ====================


def is_src_ip_type(ip_type: str) -> bool:
    """判断是否为源IP类型(包含dma或rn)"""
    ip_lower = ip_type.lower()
    return "dma" in ip_lower or "rn" in ip_lower


def is_dst_ip_type(ip_type: str) -> bool:
    """判断是否为目标IP类型(ddr或l2m)"""
    return ip_type.lower() in {"ddr", "l2m"}


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
        st.session_state.output_filename = f"traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 配置版本号（用于缓存失效）
    if "config_version" not in st.session_state:
        st.session_state.config_version = 0

    # 配置加载模式
    if "config_load_mode" not in st.session_state:
        st.session_state.config_load_mode = "replace"


# ==================== 缓存函数 ====================


@st.cache_data(ttl=1)
def get_cached_configs(_config_manager, version):
    """缓存配置列表获取，避免重复deepcopy"""
    import copy

    return copy.deepcopy(_config_manager.configs)


@st.cache_data(ttl=5)
def get_cached_traffic_estimate(_config_manager, version):
    """缓存流量预估统计"""
    configs = _config_manager.configs
    total_requests = 0
    read_requests = 0
    write_requests = 0

    for config in configs:
        single_estimate = _config_manager.estimator.estimate_single_config(config, config.end_time)
        total_requests += single_estimate["total_requests"]
        if config.req_type == "R":
            read_requests += single_estimate["total_requests"]
        else:
            write_requests += single_estimate["total_requests"]

    return {"total_requests": total_requests, "read_requests": read_requests, "write_requests": write_requests}


# ==================== 辅助函数 ====================


def delete_config_callback(config_id):
    """配置删除回调函数"""
    st.session_state.config_manager.remove_config(config_id)
    st.session_state.config_version += 1


@st.cache_resource
def get_topology_visualizer(topo_type):
    """缓存拓扑可视化器，避免重复创建"""
    return TopologyVisualizer(topo_type, ip_mappings={})


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
        show_success(f"{current_ip} 已挂载到节点 {node_id}")


# ==================== 主界面 ====================


def render_main_ui():
    """渲染主界面"""
    # 加载CSS样式
    load_custom_css()

    # 标题区域
    st.markdown("<h1>数据流生成工具</h1>", unsafe_allow_html=True)

    with st.expander("使用说明", expanded=False):
        st.markdown(
            """
        **操作流程:**
        1. 选择拓扑类型和数据流模式
        2. 挂载IP到节点
        3. 配置数据流参数
        4. 添加到配置列表
        5. 生成数据流文件
        6. 查看结果分析 (可选拆分)
        """,
            unsafe_allow_html=True,
        )

    section_divider()

    # 主区域 - 分为左右两栏
    col_left, col_right = st.columns(UIConfig.MAIN_COLS_RATIO)

    # 左栏 - 拓扑可视化（使用fragment实现局部刷新）
    with col_left:
        render_ip_mount_section()

    # 右栏 - 配置管理
    with col_right:
        render_config_section()

    # 配置列表（全宽显示）
    render_config_list()


@st.fragment
def render_ip_mount_section():
    """IP挂载区域（独立刷新）"""
    st.subheader("IP 挂载")

    # 拓扑类型输入
    topo_input = st.text_input("拓扑类型", value=st.session_state.topo_type, placeholder="例如: 5x4, 4X3, 4,3", help="支持格式: 行x列 (2-10)", key="topo_type_input")

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
            if UIConfig.TOPO_MIN <= rows <= UIConfig.TOPO_MAX and UIConfig.TOPO_MIN <= cols <= UIConfig.TOPO_MAX:
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
            st.error(f"拓扑格式错误,请使用如 5x4, 4X3, 4,3 等格式(行列范围: {UIConfig.TOPO_MIN}-{UIConfig.TOPO_MAX})")

    mini_divider()

    # IP挂载区
    st.caption("支持格式: 单节点 `0` | 多节点 `0,1,2` | 范围 `0-3`")

    col_ip, col_node = st.columns(2)
    with col_ip:
        current_ip = st.text_input("IP 名称", value=st.session_state.current_ip, placeholder="例如: gdma_0", key="ip_input")
        st.session_state.current_ip = current_ip

    with col_node:
        target_node = st.text_input("目标节点", placeholder="例如: 0 或 0,1,2", key="node_input")

    if st.button("挂载到节点", use_container_width=True, type="primary"):
        if current_ip.strip() and target_node.strip():
            # 解析节点ID
            visualizer = get_topology_visualizer(st.session_state.topo_type)
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
                    show_success(f"{current_ip} 已挂载到 {mount_count} 个节点")
                else:
                    show_info(f"{current_ip} 已存在于选中节点")
            except ValueError as e:
                show_error(str(e))
        else:
            show_error("请输入IP名称和节点ID")

    mini_divider()

    # IP挂载管理
    col_save, col_load = st.columns(2)
    with col_save:
        if st.button("保存挂载配置", use_container_width=True, disabled=not st.session_state.node_ips):
            st.session_state.show_save_dialog = True

    # 保存对话框
    if st.session_state.get("show_save_dialog", False):
        st.markdown('<div class="dialog-container">', unsafe_allow_html=True)
        st.markdown("**保存 IP 挂载配置**")

        save_name = st.text_input("配置名称", placeholder="例如: gdma_ddr_test", help="用于标识此配置", key="save_name_mount")

        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("确认保存", use_container_width=True, type="primary", key="save_confirm_mount"):
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
                    show_success(f"已保存为 {save_name}")
                else:
                    show_error("请输入配置名称")

        with col_cancel:
            if st.button("取消", use_container_width=True, key="save_cancel_mount"):
                st.session_state.show_save_dialog = False

        st.markdown("</div>", unsafe_allow_html=True)

    with col_load:
        # 始终显示加载按钮
        if st.button("加载挂载配置", use_container_width=True):
            st.session_state.show_load_dialog = True

    # 加载对话框
    if st.session_state.get("show_load_dialog", False):
        st.markdown("**加载 IP 挂载配置**")

        save_dir = project_root / "config" / "ip_mounts"

        if not save_dir.exists():
            st.info("暂无保存的配置文件")
            if st.button("关闭", use_container_width=True, key="close_load_empty"):
                st.session_state.show_load_dialog = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            save_files = sorted(save_dir.glob("*.json"), reverse=True)

            if not save_files:
                st.info("暂无保存的配置文件")
                if st.button("关闭", use_container_width=True, key="close_load_no_files"):
                    st.session_state.show_load_dialog = False
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                file_options = {}
                for f in save_files:
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

                if not file_options:
                    st.info("暂无有效的配置文件")
                    if st.button("关闭", use_container_width=True, key="close_load_invalid"):
                        st.session_state.show_load_dialog = False
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    selected_file = st.selectbox("选择配置", options=list(file_options.keys()), key="load_file_select")

                    col_confirm, col_delete, col_cancel = st.columns(3)
                    with col_confirm:
                        if st.button("加载", use_container_width=True, type="primary", key="load_confirm_mount"):
                            try:
                                load_path = file_options[selected_file]
                                with open(load_path, "r", encoding="utf-8") as f:
                                    data = json.load(f)

                                # 检查拓扑类型是否匹配
                                if data["topo_type"] != st.session_state.topo_type:
                                    show_warning(f"加载的配置是 {data['topo_type']} 拓扑,当前是 {st.session_state.topo_type}")

                                # 获取当前拓扑的最大节点数
                                rows, cols = map(int, st.session_state.topo_type.split("x"))
                                max_node_id = rows * cols - 1

                                # 加载IP挂载数据并检查节点范围
                                node_ips_data = {int(k): v for k, v in data["node_ips"].items()}
                                invalid_nodes = [node_id for node_id in node_ips_data.keys() if node_id > max_node_id]

                                if invalid_nodes:
                                    show_error(f"加载失败: 节点 {invalid_nodes} 超过当前拓扑最大节点ID {max_node_id}")
                                else:
                                    st.session_state.node_ips = node_ips_data
                                    st.session_state.show_load_dialog = False
                                    show_success("配置加载成功")
                            except Exception as e:
                                show_error(f"加载失败: {str(e)}")

                    with col_delete:
                        if st.button("删除", use_container_width=True, type="secondary", key="load_delete_mount"):
                            try:
                                load_path = file_options[selected_file]
                                load_path.unlink()
                                show_success("配置已删除")
                                remaining_files = list(save_dir.glob("*.json"))
                                if not remaining_files:
                                    st.session_state.show_load_dialog = False
                            except Exception as e:
                                show_error(f"删除失败: {str(e)}")

                    with col_cancel:
                        if st.button("取消", use_container_width=True, key="load_cancel_mount"):
                            st.session_state.show_load_dialog = False
                            st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)

    # 拓扑图标题
    st.markdown("---")
    st.subheader("拓扑图")

    # 获取已计算的链路带宽（从session_state缓存中读取）
    link_bandwidth = st.session_state.get("cached_link_bandwidth", None)

    # 绘制拓扑图
    visualizer = get_topology_visualizer(st.session_state.topo_type)
    fig = visualizer.draw_topology_grid(selected_src=set(), selected_dst=set(), node_ips=st.session_state.node_ips, link_bandwidth=link_bandwidth)
    st.plotly_chart(fig, use_container_width=True, key="topology_display")

    # 节点IP管理面板
    mini_divider()
    with st.expander("已挂载 IP 列表", expanded=False):
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
                    if st.button("删除", key=f"del_ip_{ip}", use_container_width=True):
                        for node_id in nodes:
                            if node_id in st.session_state.node_ips:
                                if ip in st.session_state.node_ips[node_id]:
                                    st.session_state.node_ips[node_id].remove(ip)
                                if not st.session_state.node_ips[node_id]:
                                    del st.session_state.node_ips[node_id]

            mini_divider()
            if st.button("清空所有 IP", use_container_width=True, type="secondary"):
                st.session_state.node_ips = {}
        else:
            st.info("暂无挂载的IP")


def render_config_section():
    """配置管理区域"""
    st.subheader("数据流配置")

    # 数据流模式选择
    traffic_mode = st.selectbox("数据流模式", ["NoC", "D2D"], index=0 if st.session_state.traffic_mode == "NoC" else 1, key="traffic_mode_select")
    if traffic_mode != st.session_state.traffic_mode:
        st.session_state.traffic_mode = traffic_mode

    mini_divider()

    # 获取已挂载IP的节点列表
    nodes_with_ips = sorted([node for node, ips in st.session_state.node_ips.items() if ips])

    if not nodes_with_ips:
        st.warning("请先在拓扑图中挂载IP到节点")
    else:
        # 配置模式选择
        config_mode = st.radio("配置模式", ["具体配置", "批量配置"], horizontal=True, help="具体配置: 精确指定某个节点的IP到另一个节点的IP; 批量配置: 按IP类型配置(如所有gdma到所有ddr)")

        # D2D模式的Die对选择
        if st.session_state.traffic_mode == "D2D":
            st.markdown("**Die 对配置 (可多选):**")

            # 初始化session state
            if "last_selected_template" not in st.session_state:
                st.session_state.last_selected_template = "自定义"
            if "selected_die_pairs" not in st.session_state:
                st.session_state.selected_die_pairs = []

            # 模板快捷选择
            die_templates = load_die_templates()
            template_names = ["自定义"] + list(die_templates.keys())
            selected_template = st.selectbox("快速模板", options=template_names, key="die_template_select")

            # Die对多选
            die_pair_options = generate_die_pair_options(4)

            # 如果模板变化,使用模板默认值
            default_pairs = []
            if selected_template != st.session_state.last_selected_template:
                st.session_state.last_selected_template = selected_template
                if selected_template != "自定义":
                    default_pairs = die_templates[selected_template]

            selected_die_pairs = st.multiselect("Die 对", options=die_pair_options, default=default_pairs, key="die_pairs_multiselect")

            mini_divider()

        # 配置表单（禁用回车提交）
        with st.form("config_form", enter_to_submit=False):

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

            mini_divider()

            # 参数配置
            st.markdown("**流量参数配置**")
            col_p1, col_p2, col_p3 = st.columns(UIConfig.PARAM_COLS_RATIO)
            with col_p1:
                end_time = st.number_input(
                    "仿真时长 (ns)",
                    min_value=UIConfig.END_TIME_RANGE[0],
                    max_value=UIConfig.END_TIME_RANGE[1],
                    value=UIConfig.DEFAULT_END_TIME,
                    step=UIConfig.END_TIME_RANGE[2],
                    help="数据流仿真的总时长",
                )
            with col_p2:
                speed = st.number_input(
                    "IP 带宽 (GB/s)",
                    min_value=UIConfig.SPEED_RANGE[0],
                    max_value=UIConfig.SPEED_RANGE[1],
                    value=UIConfig.DEFAULT_SPEED,
                    step=UIConfig.SPEED_RANGE[2],
                    format="%.2f",
                    help="IP接口的数据传输带宽",
                )
            with col_p3:
                burst = st.number_input(
                    "Burst 长度",
                    min_value=UIConfig.BURST_RANGE[0],
                    max_value=UIConfig.BURST_RANGE[1],
                    value=UIConfig.DEFAULT_BURST,
                    step=UIConfig.BURST_RANGE[2],
                    help="突发传输的数据包长度",
                )

            req_type = st.radio("请求类型", ["R", "W"], horizontal=True, help="R=读请求, W=写请求")

            submit_button = st.form_submit_button("添加配置", use_container_width=True, type="primary")

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
                            # 使用当前multiselect的选择（而不是session_state）
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
                                        st.session_state.config_version += 1
                                    else:
                                        error_messages.extend([f"{die_pair}: {e}" for e in errors])

                                if success_count > 0:
                                    st.success(f"成功添加 {success_count} 个配置!")
                                    if error_messages:
                                        st.warning("部分配置失败:\n" + "\n".join(error_messages))
                                    st.rerun()
                                else:
                                    st.error("所有配置验证失败:\n" + "\n".join(error_messages))
                        else:
                            # NoC模式：单个配置
                            config = TrafficConfig(src_map=src_map, dst_map=dst_map, speed=speed, burst=burst, req_type=req_type, end_time=end_time)

                            # 添加到配置管理器
                            success, errors = st.session_state.config_manager.add_config(config)

                            if success:
                                st.session_state.config_version += 1
                                st.success("配置添加成功!")
                                st.rerun()
                            else:
                                st.error("配置验证失败:\n" + "\n".join(errors))

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
                                            st.session_state.config_version += 1
                                        else:
                                            error_messages.extend([f"{die_pair}: {e}" for e in errors])

                                    if success_count > 0:
                                        st.success(f"成功添加 {success_count} 个配置!")
                                        if error_messages:
                                            st.warning("部分配置失败:\n" + "\n".join(error_messages))
                                        st.rerun()
                                    else:
                                        st.error("所有配置验证失败:\n" + "\n".join(error_messages))
                            else:
                                # NoC模式：单个配置
                                config = TrafficConfig(src_map=src_map, dst_map=dst_map, speed=speed, burst=burst, req_type=req_type, end_time=end_time)

                                # 添加到配置管理器
                                success, errors = st.session_state.config_manager.add_config(config)

                                if success:
                                    st.session_state.config_version += 1
                                    st.success("配置添加成功!")
                                    st.rerun()
                                else:
                                    st.error("配置验证失败:\n" + "\n".join(errors))

    # 带宽计算选项（放在添加配置下方）
    mini_divider()
    # 判断是否为D2D模式
    is_d2d_mode = any(
        getattr(cfg, 'src_die', 0) != getattr(cfg, 'dst_die', 0)
        for cfg in st.session_state.config_manager.configs
    )
    routing_type = st.selectbox("路由算法", ["XY", "YX"], index=0, key="bandwidth_routing_type", help="XY: 先水平后垂直; YX: 先垂直后水平")

    if st.button(
        "计算静态链路带宽",
        use_container_width=True,
        type="primary",
        disabled=is_d2d_mode,
        help="D2D模式暂不支持静态链路带宽计算" if is_d2d_mode else "基于当前配置计算静态链路带宽"
    ):
        # 执行带宽计算
        configs = get_cached_configs(st.session_state.config_manager, st.session_state.config_version)
        if configs and st.session_state.node_ips:
            try:
                from src.traffic_process.traffic_gene.static_bandwidth_analyzer import compute_link_bandwidth
                link_bandwidth = compute_link_bandwidth(
                    topo_type=st.session_state.topo_type,
                    node_ips=st.session_state.node_ips,
                    configs=configs,
                    routing_type=routing_type
                )
                st.session_state.cached_link_bandwidth = link_bandwidth
                st.success("静态链路带宽计算完成！")
                st.rerun()
            except Exception as e:
                st.error(f"带宽计算失败: {str(e)}")


def render_config_list():
    """配置列表区域（全宽显示）"""
    mini_divider()
    st.subheader("已配置列表")

    configs = get_cached_configs(st.session_state.config_manager, st.session_state.config_version)

    # 操作按钮行
    col_btn1, col_btn2, col_btn3, col_spacer = st.columns([1, 1, 1, 7])
    with col_btn1:
        if st.button("保存配置", use_container_width=True, disabled=not configs):
            st.session_state.show_save_config_dialog = True
    with col_btn2:
        if st.button("加载配置", use_container_width=True):
            st.session_state.show_load_config_dialog = True
    with col_btn3:
        if st.button("清空所有配置", use_container_width=True, disabled=not configs, type="secondary"):
            st.session_state.show_clear_all_dialog = True

    # 保存配置对话框
    if st.session_state.get("show_save_config_dialog", False):
        # st.markdown('<div class="dialog-container">', unsafe_allow_html=True)
        st.markdown("**保存数据流配置**")

        config_name = st.text_input("配置名称", placeholder="例如: gdma_to_ddr_test", help="用于标识此配置集的名称", key="save_config_name_bottom")

        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("确认保存", use_container_width=True, type="primary", key="save_config_confirm_bottom"):
                if config_name.strip():
                    # 保存到JSON文件
                    save_dir = project_root / "config" / "traffic_configs"
                    save_dir.mkdir(parents=True, exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_name = config_name.strip().replace(" ", "_")
                    filename = f"{safe_name}_{st.session_state.topo_type}_{timestamp}.json"
                    save_path = save_dir / filename

                    # 导出配置数据
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
                    show_success(f"已保存为 {config_name}")
                else:
                    show_error("请输入配置名称")

        with col_cancel:
            if st.button("取消", use_container_width=True, key="save_config_cancel_bottom"):
                st.session_state.show_save_config_dialog = False

        st.markdown("</div>", unsafe_allow_html=True)

    # 加载配置对话框
    if st.session_state.get("show_load_config_dialog", False):
        # st.markdown('<div class="dialog-container">', unsafe_allow_html=True)
        st.markdown("**加载数据流配置**")

        # 加载模式选择
        load_mode = st.radio(
            "加载模式",
            ["替换现有配置", "合并到现有配置"],
            horizontal=True,
            help="替换: 清空现有配置后加载 | 合并: 将新配置添加到现有配置",
            key="config_load_mode_radio",
        )

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
                    selected_file = st.selectbox("选择配置", options=list(file_options.keys()), key="load_config_select_bottom")

                    col_confirm, col_delete, col_cancel = st.columns(3)
                    with col_confirm:
                        if st.button("加载", use_container_width=True, type="primary", key="load_config_confirm_bottom"):
                            try:
                                load_path = file_options[selected_file]
                                with open(load_path, "r", encoding="utf-8") as f:
                                    data = json.load(f)

                                # 检查拓扑类型
                                if data["topo_type"] != st.session_state.topo_type:
                                    show_warning(f"加载的配置是 {data['topo_type']} 拓扑,当前是 {st.session_state.topo_type}")

                                # 根据选择的模式处理
                                rows, cols = map(int, st.session_state.topo_type.split("x"))
                                num_nodes = rows * cols

                                # 只在"替换模式"时重新初始化
                                if load_mode == "替换现有配置":
                                    st.session_state.config_manager = ConfigManager(num_nodes)

                                # 加载配置（两种模式都执行）
                                for config_dict in data["configs"]:
                                    config = TrafficConfig(
                                        src_map=config_dict["src_map"],
                                        dst_map=config_dict["dst_map"],
                                        speed=config_dict["speed"],
                                        burst=config_dict["burst"],
                                        req_type=config_dict["req_type"],
                                        end_time=config_dict.get("end_time", UIConfig.DEFAULT_END_TIME),
                                    )
                                    if "src_die" in config_dict:
                                        config.src_die = config_dict["src_die"]
                                        config.dst_die = config_dict["dst_die"]

                                    st.session_state.config_manager.add_config(config)

                                st.session_state.config_version += 1  # 更新版本号用于缓存失效
                                st.session_state.show_load_config_dialog = False
                                show_success("配置加载成功")
                            except Exception as e:
                                show_error(f"加载失败: {str(e)}")

                    with col_delete:
                        if st.button("删除", use_container_width=True, type="secondary", key="delete_config_btn_bottom"):
                            try:
                                load_path = file_options[selected_file]
                                load_path.unlink()
                                show_success("配置已删除")
                                remaining_files = list(save_dir.glob("*.json"))
                                if not remaining_files:
                                    st.session_state.show_load_config_dialog = False
                            except Exception as e:
                                show_error(f"删除失败: {str(e)}")

                    with col_cancel:
                        if st.button("取消", use_container_width=True, key="load_config_cancel_bottom"):
                            st.session_state.show_load_config_dialog = False
                else:
                    st.info("暂无保存的配置")
            else:
                st.info("暂无保存的配置")
        else:
            st.info("暂无保存的配置")

        st.markdown("</div>", unsafe_allow_html=True)

    # 清空所有配置对话框
    if st.session_state.get("show_clear_all_dialog", False):
        st.markdown("**清空所有配置**")
        st.warning(f"确定要清空所有 {len(configs)} 个配置吗？此操作不可恢复！")

        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("确认清空", use_container_width=True, type="primary", key="clear_all_confirm"):
                st.session_state.config_manager.configs.clear()
                st.session_state.config_version += 1
                st.session_state.show_clear_all_dialog = False
                # 清空缓存的带宽计算结果
                if "cached_link_bandwidth" in st.session_state:
                    del st.session_state.cached_link_bandwidth
                st.success("已清空所有配置")
                st.rerun()
        with col_cancel:
            if st.button("取消", use_container_width=True, key="clear_all_cancel"):
                st.session_state.show_clear_all_dialog = False
                st.rerun()

    # 配置列表显示
    if configs:
        # 简化的IP摘要函数
        def get_ip_summary(ip_list):
            if len(ip_list) == 1:
                return ip_list[0]
            ip_types = set()
            for ip in ip_list:
                ip_type = ip.split("_")[0] if "_" in ip else ip
                ip_types.add(ip_type)
            if len(ip_types) == 1:
                return list(ip_types)[0].upper()
            return f"{len(ip_list)}IP"

        # D2D模式：按源Die分组
        if st.session_state.traffic_mode == "D2D":
            src_die_groups = {}
            for config in configs:
                if hasattr(config, "src_die"):
                    src_die_key = f"Die{config.src_die}"
                    if src_die_key not in src_die_groups:
                        src_die_groups[src_die_key] = []
                    src_die_groups[src_die_key].append(config)

            # 分组显示 - 简化版配置卡片
            for src_die, group_configs in src_die_groups.items():
                st.markdown(f"**{src_die}** ({len(group_configs)}个)")

                for i in range(0, len(group_configs), 8):
                    cols = st.columns(8)
                    for j in range(8):
                        if i + j < len(group_configs):
                            config = group_configs[i + j]
                            src_summary = get_ip_summary(list(config.src_map.keys()))
                            dst_summary = get_ip_summary(list(config.dst_map.keys()))

                            with cols[j]:
                                with st.container(border=True):
                                    # 简化为单行显示
                                    st.markdown(
                                        f"**#{config.config_id}** D{config.src_die}→D{config.dst_die} | "
                                        f"{src_summary}→{dst_summary} | "
                                        f"{config.end_time}ns | {config.speed}GB/s | B{config.burst} | {config.req_type}"
                                    )
                                    st.button("删除", key=f"del_bottom_{config.config_id}", use_container_width=True, on_click=delete_config_callback, args=(config.config_id,))
        else:
            # NoC模式：简化版配置卡片
            for i in range(0, len(configs), 8):
                cols = st.columns(8)
                for j in range(8):
                    if i + j < len(configs):
                        config = configs[i + j]
                        src_summary = get_ip_summary(list(config.src_map.keys()))
                        dst_summary = get_ip_summary(list(config.dst_map.keys()))

                        with cols[j]:
                            with st.container(border=True):
                                # 简化为单行显示
                                st.markdown(f"**#{config.config_id}** {src_summary}→{dst_summary} | " f"{config.end_time}ns | {config.speed}GB/s | B{config.burst} | {config.req_type}")
                                st.button("删除", key=f"del_bottom_{config.config_id}", use_container_width=True, on_click=delete_config_callback, args=(config.config_id,))
    else:
        st.info("暂无配置，请添加配置或加载已保存的配置")

    # 生成按钮
    mini_divider()
    st.subheader("生成数据流文件")

    col_gen1, col_gen2 = st.columns([3, 1])

    with col_gen1:
        # 输入框中显示不带.txt的文件名
        display_filename = st.session_state.output_filename.replace(".txt", "")
        output_filename_input = st.text_input("输出文件名", value=display_filename, key="output_filename_input")
        # 更新session state，保存不带.txt的版本
        st.session_state.output_filename = output_filename_input

    with col_gen2:
        st.write("")  # 占位
        st.write("")  # 占位

    # 数据流拆分选项
    enable_split = st.checkbox("拆分数据流文件(按源IP)", value=False, help="生成后自动按源IP拆分数据流文件,输出目录为输出文件名(去掉.txt)")

    if st.button("生成数据流文件", type="primary", use_container_width=True):
        if not configs:
            st.error("请先添加至少一个配置!")
        else:
            # 生成数据流
            with st.spinner("正在生成数据流文件..."):
                # 输出路径
                output_dir = project_root / "traffic"
                output_dir.mkdir(exist_ok=True)
                # 自动添加.txt后缀
                final_filename = st.session_state.output_filename if st.session_state.output_filename.endswith(".txt") else st.session_state.output_filename + ".txt"
                output_file = output_dir / final_filename

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

            st.success(f"数据流文件生成成功: {file_path}")

            # 拆分数据流文件
            if enable_split:
                with st.spinner("正在拆分数据流文件..."):
                    try:
                        # 根据输出文件名生成拆分目录 - 去掉.txt后缀
                        base_name = final_filename.replace(".txt", "")
                        split_dir = output_dir / base_name

                        # 获取拓扑参数
                        rows, cols = map(int, st.session_state.topo_type.split("x"))

                        # 根据模式选择拆分函数
                        if st.session_state.traffic_mode == "D2D":
                            split_result = split_d2d_traffic_by_source(input_file=file_path, output_dir=str(split_dir), num_col=cols, num_row=rows, verbose=False)
                        else:
                            split_result = split_traffic_by_source(input_file=file_path, output_dir=str(split_dir), num_col=cols, num_row=rows, verbose=False)

                        st.session_state.split_result = split_result
                        st.success(f"数据流拆分完成! 输出目录: {split_result['output_dir']}")
                        st.info(f"共生成 {split_result['total_sources']} 个拆分文件")

                    except Exception as e:
                        st.error(f"拆分失败: {e}")

            # 提供下载按钮
            with open(file_path, "r") as f:
                st.download_button(label="下载数据流文件", data=f.read(), file_name=final_filename, mime="text/plain")


# ==================== 主程序入口 ====================


def main():
    """主程序入口"""
    init_session_state()
    render_main_ui()


if __name__ == "__main__":
    main()
