"""
FIFO热力图JavaScript生成模块

生成FIFO热力图的交互式JavaScript代码，包括：
- 热力图点击事件处理
- FIFO深度曲线图表
- 模式/网络/通道切换逻辑
"""

import json
from typing import Dict, List, Tuple


def generate_fifo_heatmap_javascript(fifo_data: Dict, fifo_options: List[Tuple[str, str, str, str, int]], num_dies: int, cycles_per_ns: int = 1) -> str:
    """
    生成FIFO热力图的自定义JavaScript代码

    Args:
        fifo_data: FIFO数据字典
        fifo_options: FIFO选项列表（5元组：name, category, fifo_type, network_type, ch_idx）
        num_dies: Die数量
        cycles_per_ns: 每纳秒的周期数，用于时间转换

    Returns:
        str: JavaScript代码字符串
    """
    # 创建FIFO选项映射 (用于JavaScript查找)
    # key: "category_fifo_type_network_type_ch_idx" -> index
    fifo_map = {}
    for idx, (name, category, fifo_type, network_type, ch_idx) in enumerate(fifo_options):
        key = f"{category}_{fifo_type}_{network_type}_{ch_idx}"
        fifo_map[key] = idx

    # 构建depth_events数据结构
    # key: "die_node_category_fifo_type_network_type_ch_idx" -> {"events": [...], "capacity": N}
    fifo_events_data = {}
    total_events_count = 0
    for die_id, networks_data in fifo_data.items():
        for network_type, channels_data in networks_data.items():
            for ch_idx, die_data in channels_data.items():
                for category in ["IQ", "RB", "EQ", "IQ_CH", "EQ_CH", "RS_IN", "RS_OUT"]:
                    category_data = die_data.get(category, {})
                    for node_pos, fifo_types in category_data.items():
                        if isinstance(fifo_types, dict):
                            for fifo_type, fifo_info in fifo_types.items():
                                if isinstance(fifo_info, dict) and "depth_events" in fifo_info:
                                    events = fifo_info.get("depth_events", [])
                                    capacity = fifo_info.get("capacity", 0)
                                    # 只保存有事件的FIFO
                                    if events:
                                        key = f"{die_id}_{node_pos}_{category}_{fifo_type}_{network_type}_{ch_idx}"
                                        fifo_events_data[key] = {"events": events, "capacity": capacity}
                                        total_events_count += len(events)

    # 检测每个网络类型有多少个通道
    channel_counts = {"req": set(), "rsp": set(), "data": set()}
    for _, _, _, network_type, ch_idx in fifo_options:
        channel_counts[network_type].add(ch_idx)

    # 转换为有序列表
    channel_info = {net_type: sorted(list(channels)) for net_type, channels in channel_counts.items()}
    max_channels = max(len(chs) for chs in channel_info.values()) if channel_info else 1

    # 计算trace索引
    num_heatmap_traces = len(fifo_options) * 3

    # 计算默认FIFO索引（network_type=="req" 且 ch_idx==0）
    default_fifo_index = 0
    for idx, opt in enumerate(fifo_options):
        if opt[3] == "req" and opt[4] == 0:
            default_fifo_index = idx
            break

    # 创建FIFO选项的详细信息（用于JavaScript）
    fifo_options_js = [[opt[0], opt[1], opt[2], opt[3], opt[4]] for opt in fifo_options]

    # 序列化fifo_events_data为JSON
    fifo_events_json = json.dumps(fifo_events_data)

    # 生成JavaScript代码
    js_code = f"""
<script>

    // FIFO选项映射（包含网络类型和通道索引）
    const fifoMap = {str(fifo_map).replace("'", '"')};
    const fifoOptionsData = {str(fifo_options_js).replace("'", '"')};
    const numFifoOptions = {len(fifo_options)};
    const numDies = {num_dies};
    const numHeatmapTraces = {num_heatmap_traces};
    const channelInfo = {str(channel_info).replace("'", '"')};  // {{req: [0,1], rsp: [0], data: [0,1,2]}}
    const maxChannels = {max_channels};


    // FIFO深度事件数据（用于曲线图）
    const fifoEventsData = {fifo_events_json};
    const cyclesPerNs = {cycles_per_ns};


    // 当前选中的状态（默认显示 data 网络 Ch0）
    const defaultFifoIndex = {default_fifo_index};
    let currentFifoIndex = defaultFifoIndex;
    let currentMode = 'avg';  // 默认平均模式
    let currentNetworkType = fifoOptionsData.length > defaultFifoIndex ? fifoOptionsData[defaultFifoIndex][3] : 'data';
    let currentCategory = fifoOptionsData.length > defaultFifoIndex ? fifoOptionsData[defaultFifoIndex][1] : null;
    let currentFifoType = fifoOptionsData.length > defaultFifoIndex ? fifoOptionsData[defaultFifoIndex][2] : null;
    let currentChannelIdx = fifoOptionsData.length > defaultFifoIndex ? fifoOptionsData[defaultFifoIndex][4] : 0;

    // 等待Plotly加载完成
    document.addEventListener('DOMContentLoaded', function() {{
        setTimeout(function() {{
            const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            if (!plotDiv) return;

            // 初始化图表面板（复用tracker-panel或创建新的）
            initChartPanel();

            // 初始化当前FIFO的category和type
            updateCurrentFifoInfo();

            // 监听热力图点击事件（显示曲线）
            plotDiv.on('plotly_click', function(data) {{
                const clickedPoint = data.points[0];
                const traceIndex = clickedPoint.curveNumber;

                // 点击热力图节点时显示曲线
                if (traceIndex < numHeatmapTraces) {{
                    const pointIndex = clickedPoint.pointIndex;
                    const hoverText = clickedPoint.hovertext || '';
                    // 解析hover文本获取节点信息
                    const dieMatch = hoverText.match(/Die (\\d+)/);
                    const nodeMatch = hoverText.match(/节点 (\\d+)/);
                    if (dieMatch && nodeMatch) {{
                        const dieId = parseInt(dieMatch[1]);
                        const nodePos = parseInt(nodeMatch[1]);
                        showFifoChart(dieId, nodePos, currentCategory, currentFifoType, currentNetworkType, currentChannelIdx);
                    }}
                    return;  // 不处理架构图切换
                }}

                if (traceIndex >= numHeatmapTraces) {{
                    const customdata = clickedPoint.customdata;
                    if (customdata && customdata.length >= 2) {{
                        const category = customdata[0];
                        const fifoType = customdata[1];

                        // 使用当前选中的网络类型和通道索引
                        const key = category + '_' + fifoType + '_' + currentNetworkType + '_' + currentChannelIdx;

                        let fifoIndex = fifoMap[key];
                        if (fifoIndex === undefined) {{
                            // 如果当前网络类型+通道不存在，尝试其他网络和通道
                            let found = false;
                            for (let net of ['data', 'req', 'rsp']) {{
                                const channels = channelInfo[net] || [0];
                                for (let ch of channels) {{
                                    const tryKey = category + '_' + fifoType + '_' + net + '_' + ch;
                                    if (fifoMap[tryKey] !== undefined) {{
                                        fifoIndex = fifoMap[tryKey];
                                        currentNetworkType = net;
                                        currentChannelIdx = ch;
                                        updateNetworkButtonHighlight();
                                        updateChannelButtonHighlight();
                                        found = true;
                                        break;
                                    }}
                                }}
                                if (found) break;
                            }}
                        }} else {{
                        }}
                        if (fifoIndex !== undefined) {{
                            currentFifoIndex = fifoIndex;
                            updateCurrentFifoInfo();
                            updateHeatmapVisibility();
                        }} else {{
                            console.warn('未找到FIFO:', category, fifoType);
                        }}
                    }}
                }}
            }});

            // 更新当前FIFO的信息
            function updateCurrentFifoInfo() {{
                if (currentFifoIndex >= 0 && currentFifoIndex < fifoOptionsData.length) {{
                    const option = fifoOptionsData[currentFifoIndex];
                    currentCategory = option[1];
                    currentFifoType = option[2];
                    currentNetworkType = option[3];
                    currentChannelIdx = option[4];
                }}
            }}

            // 更新热力图可见性和架构图高亮
            function updateHeatmapVisibility() {{
                const update = {{}};
                const visibility = [];

                // 计算哪些traces应该可见（每个FIFO选项+模式组合1个trace）
                for (let i = 0; i < numFifoOptions; i++) {{
                    for (let mode of ['avg', 'peak', 'flit_count']) {{
                        const shouldShow = (i === currentFifoIndex && mode === currentMode);
                        visibility.push(shouldShow);
                    }}
                }}

                // 架构图的traces保持可见
                for (let i = numHeatmapTraces; i < plotDiv.data.length; i++) {{
                    visibility.push(true);
                }}

                update.visible = visibility;
                Plotly.restyle(plotDiv, update);

                // 更新架构图高亮
                updateArchitectureHighlight();
            }}

            // 更新架构图高亮
            function updateArchitectureHighlight() {{
                const shapes = plotDiv.layout.shapes || [];
                const expectedName = currentCategory + '_' + currentFifoType;

                const newShapes = shapes.map((shape, idx) => {{
                    // 跳过没有name的shape（模块边框没有name属性）
                    if (!shape.name) {{
                        return shape;
                    }}

                    // 检查是否为当前选中的FIFO（只比较category和fifo_type）
                    const shapeName = shape.name;
                    const isSelected = (shapeName === expectedName);

                    // 返回更新后的shape
                    return {{
                        ...shape,
                        line: {{
                            ...shape.line,
                            color: isSelected ? 'red' : 'black',
                            width: isSelected ? 3 : 1
                        }}
                    }};
                }});

                // 更新layout
                Plotly.relayout(plotDiv, {{'shapes': newShapes}});
            }}

            // 等待按钮渲染完成后绑定事件
            function setupButtonListeners() {{
                const allButtons = plotDiv.querySelectorAll('.updatemenu-button');

                // 计算预期按钮总数：3(模式) + 3(网络) + maxChannels(通道，如果>1)
                const expectedButtons = maxChannels > 1 ? 6 + maxChannels : 6;
                if (allButtons.length < expectedButtons) {{
                    console.warn('按钮未完全渲染，重试...');
                    setTimeout(setupButtonListeners, 200);
                    return;
                }}

                // 第一组：模式按钮（平均/峰值/计数）- 前3个
                const modeButtons = Array.from(allButtons).slice(0, 3);
                // 第二组：网络类型按钮（REQ/RSP/DATA）- 接下来3个
                const networkButtons = Array.from(allButtons).slice(3, 6);
                // 第三组：通道按钮（Ch0/Ch1/...）- 剩余的（如果有多通道）
                const channelButtons = maxChannels > 1 ? Array.from(allButtons).slice(6, 6 + maxChannels) : [];

                // 监听平均/峰值/Flit计数按钮点击
                modeButtons.forEach((btn, idx) => {{
                    btn.addEventListener('click', function(e) {{
                        const modeNames = ['avg', 'peak', 'flit_count'];
                        setTimeout(() => {{
                            // 移除同组按钮的active类
                            modeButtons.forEach(b => b.classList.remove('active'));
                            // 添加到当前按钮
                            this.classList.add('active');

                            currentMode = modeNames[idx];
                            updateHeatmapVisibility();
                        }}, 10);
                    }});
                }});

                // 监听网络类型按钮点击
                networkButtons.forEach((btn, idx) => {{
                    btn.addEventListener('click', function(e) {{
                        const networks = ['req', 'rsp', 'data'];
                        setTimeout(() => {{
                            // 移除同组按钮的active类
                            networkButtons.forEach(b => b.classList.remove('active'));
                            // 添加到当前按钮
                            this.classList.add('active');

                            currentNetworkType = networks[idx];

                            // 更新通道按钮的可见性（根据当前网络类型的通道数）
                            updateChannelButtonsVisibility();

                            // 切换到当前FIFO在新网络中的对应项（保持当前通道，如果不存在则切换到第一个通道）
                            switchToNetwork(currentNetworkType);
                        }}, 10);
                    }});
                }});

                // 监听通道按钮点击（如果有多通道）
                if (channelButtons.length > 0) {{
                    channelButtons.forEach((btn, idx) => {{
                        btn.addEventListener('click', function(e) {{
                            setTimeout(() => {{
                                // 移除同组按钮的active类
                                channelButtons.forEach(b => b.classList.remove('active'));
                                // 添加到当前按钮
                                this.classList.add('active');

                                // 切换到新通道
                                switchToChannel(idx);
                            }}, 10);
                        }});
                    }});
                }}

                // 初始化按钮高亮状态
                if (modeButtons.length > 0) {{
                    modeButtons[0].classList.add('active');  // 平均
                }}
                if (networkButtons.length > 0) {{
                    // 根据第一个FIFO的网络类型初始化
                    const networks = ['req', 'rsp', 'data'];
                    const netIdx = networks.indexOf(currentNetworkType);
                    if (netIdx >= 0 && netIdx < networkButtons.length) {{
                        networkButtons[netIdx].classList.add('active');
                    }}
                }}
                if (channelButtons.length > 0) {{
                    // 根据第一个FIFO的通道索引初始化
                    if (currentChannelIdx < channelButtons.length) {{
                        channelButtons[currentChannelIdx].classList.add('active');
                    }}
                }}

                // 初始化通道按钮可见性
                updateChannelButtonsVisibility();
            }}

            // 启动按钮监听器设置
            setupButtonListeners();

            // 切换到指定网络类型（保持当前通道，如果不存在则切换到第一个通道）
            function switchToNetwork(networkType) {{
                if (!currentCategory || !currentFifoType) return;

                const channels = channelInfo[networkType] || [0];

                // 尝试保持当前通道索引
                let targetChannel = currentChannelIdx;
                if (!channels.includes(currentChannelIdx)) {{
                    // 如果当前通道在新网络中不存在，切换到第一个通道
                    targetChannel = channels[0];
                    currentChannelIdx = targetChannel;
                    updateChannelButtonHighlight();
                }}

                const key = currentCategory + '_' + currentFifoType + '_' + networkType + '_' + targetChannel;
                const fifoIndex = fifoMap[key];

                if (fifoIndex !== undefined) {{
                    currentFifoIndex = fifoIndex;
                    updateCurrentFifoInfo();
                    updateHeatmapVisibility();
                }} else {{
                    console.warn('FIFO not found for network:', networkType, 'channel:', targetChannel);
                }}
            }}

            // 切换到指定通道
            function switchToChannel(channelIdx) {{
                if (!currentCategory || !currentFifoType) return;

                const key = currentCategory + '_' + currentFifoType + '_' + currentNetworkType + '_' + channelIdx;
                const fifoIndex = fifoMap[key];

               

                if (fifoIndex !== undefined) {{
                    currentFifoIndex = fifoIndex;
                    currentChannelIdx = channelIdx;
                    updateCurrentFifoInfo();
                    updateHeatmapVisibility();
                }} else {{
                    console.warn('FIFO not found for channel:', channelIdx, 'key:', key);
                }}
            }}

            // 更新网络按钮高亮
            function updateNetworkButtonHighlight() {{
                const allButtons = plotDiv.querySelectorAll('.updatemenu-button');
                const networkButtons = Array.from(allButtons).slice(3, 6);  // 网络按钮是第4-6个(索引3-5)
                networkButtons.forEach(b => b.classList.remove('active'));
                const networks = ['req', 'rsp', 'data'];
                const netIdx = networks.indexOf(currentNetworkType);
                if (netIdx >= 0 && netIdx < networkButtons.length) {{
                    networkButtons[netIdx].classList.add('active');
                }}
            }}

            // 更新通道按钮高亮
            function updateChannelButtonHighlight() {{
                if (maxChannels <= 1) return;  // 单通道时无需更新

                const allButtons = plotDiv.querySelectorAll('.updatemenu-button');
                const channelButtons = Array.from(allButtons).slice(6, 6 + maxChannels);
                channelButtons.forEach(b => b.classList.remove('active'));

                if (currentChannelIdx < channelButtons.length) {{
                    channelButtons[currentChannelIdx].classList.add('active');
                }}
            }}

            // 更新通道按钮的可见性（根据当前网络类型的通道数）
            function updateChannelButtonsVisibility() {{
                if (maxChannels <= 1) return;  // 单通道时无需更新

                const allButtons = plotDiv.querySelectorAll('.updatemenu-button');
                const channelButtons = Array.from(allButtons).slice(6, 6 + maxChannels);
                const currentChannels = channelInfo[currentNetworkType] || [0];

                channelButtons.forEach((btn, idx) => {{
                    if (currentChannels.includes(idx)) {{
                        btn.style.display = '';  // 显示
                    }} else {{
                        btn.style.display = 'none';  // 隐藏
                    }}
                }});
            }}

            // 初始化图表面板（复用tracker-panel或创建新的）
            function initChartPanel() {{
                // 检查是否已存在tracker-panel（由tracker_html_injector创建）
                if (document.getElementById('tracker-panel')) {{
                    return;
                }}

                // 如果没有tracker-panel，创建一个新的chart-panel
                if (document.getElementById('chart-panel')) return;

                // 添加样式
                const style = document.createElement('style');
                style.textContent = `
                    .chart-section {{
                        position: fixed;
                        right: 10px;
                        top: 10px;
                        width: 920px;
                        max-width: 95vw;
                        max-height: 95vh;
                        background: white;
                        padding: 15px;
                        border-radius: 8px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        display: none;
                        overflow: auto;
                        z-index: 10000;
                        transition: width 0.3s ease;
                    }}
                    .chart-section.narrow {{ width: 480px; }}
                    .chart-section.active {{ display: block; }}
                    .chart-grid {{
                        display: grid;
                        grid-template-columns: repeat(2, 440px);
                        grid-auto-rows: 320px;
                        grid-auto-flow: row;
                        gap: 10px;
                        margin-top: 5px;
                        max-height: calc(95vh - 50px);
                        justify-content: start;
                    }}
                    .chart-grid[data-count="1"] {{ grid-template-columns: 440px; justify-content: center; }}
                    .chart-grid[data-count="2"] {{ grid-template-columns: repeat(2, 440px); justify-content: center; }}
                    .chart-item {{
                        position: relative;
                        background: #f5f5f5;
                        padding: 0;
                        border-radius: 6px;
                        border: 2px solid #ddd;
                        width: 440px;
                        height: 320px;
                        display: flex;
                        flex-direction: column;
                        overflow: hidden;
                    }}
                    .chart-item.tracker-type {{
                        border-color: #1976d2;
                    }}
                    .chart-item.fifo-type {{
                        border-color: #4caf50;
                    }}
                    .chart-item-header {{
                        padding: 6px 10px;
                        font-size: 12px;
                        font-weight: bold;
                        color: white;
                        text-align: center;
                    }}
                    .chart-item-header.tracker {{
                        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
                    }}
                    .chart-item-header.fifo {{
                        background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
                    }}
                    .chart-item-chart {{
                        width: 430px;
                        height: 285px;
                        flex-shrink: 0;
                        margin: 5px auto;
                    }}
                    .close-chart-btn {{
                        position: absolute;
                        top: 5px;
                        right: 5px;
                        cursor: pointer;
                        background: #ff9800;
                        color: white;
                        border: none;
                        width: 24px;
                        height: 24px;
                        border-radius: 50%;
                        font-size: 16px;
                        line-height: 1;
                        padding: 0;
                        z-index: 1;
                    }}
                    .close-chart-btn:hover {{ background: #f57c00; }}
                    .close-all-btn {{
                        position: absolute;
                        top: -14px;
                        right: -14px;
                        cursor: pointer;
                        background: #f44336;
                        color: white;
                        border: none;
                        width: 28px;
                        height: 28px;
                        border-radius: 50%;
                        font-size: 18px;
                        line-height: 1;
                        padding: 0;
                    }}
                    .close-all-btn:hover {{ background: #d32f2f; }}
                `;
                document.head.appendChild(style);

                // 创建面板（使用安全的DOM方法）
                const panel = document.createElement('div');
                panel.className = 'chart-section';
                panel.id = 'chart-panel';

                const closeAllBtn = document.createElement('button');
                closeAllBtn.className = 'close-all-btn';
                closeAllBtn.textContent = '×';
                closeAllBtn.title = '关闭全部';
                closeAllBtn.onclick = function() {{ window.closeChartPanel(); }};

                const container = document.createElement('div');
                container.className = 'chart-grid';
                container.id = 'chart-container';

                panel.appendChild(closeAllBtn);
                panel.appendChild(container);
                document.body.appendChild(panel);

                // 初始化全局队列
                if (!window.chartQueue) {{
                    window.chartQueue = [];
                }}
            }}

            // 更新所有图表（全局函数，供tracker也能调用）
            window.updateAllCharts = function() {{
                // 优先使用tracker-panel（如果存在）
                let panel = document.getElementById('tracker-panel');
                let container = document.getElementById('tracker-container');
                let queue = window.trackerQueue;
                let itemClass = 'tracker-item';
                let chartClass = 'tracker-chart';

                // 如果没有tracker-panel，使用chart-panel
                if (!panel) {{
                    panel = document.getElementById('chart-panel');
                    container = document.getElementById('chart-container');
                    queue = window.chartQueue;
                    itemClass = 'chart-item';
                    chartClass = 'chart-item-chart';
                }}

                if (!panel || !container || !queue) return;

                if (queue.length === 0) {{
                    panel.classList.remove('active');
                    return;
                }}

                panel.classList.add('active');
                if (queue.length === 1) {{
                    panel.classList.add('narrow');
                }} else {{
                    panel.classList.remove('narrow');
                }}

                // 清空容器
                while (container.firstChild) {{
                    container.removeChild(container.firstChild);
                }}
                container.setAttribute('data-count', queue.length);

                // 动态创建chart-item
                queue.forEach((item, index) => {{
                    const chartItem = document.createElement('div');
                    chartItem.className = itemClass;

                    // 根据类型添加样式类
                    if (item.type === 'tracker') {{
                        chartItem.classList.add('tracker-type');
                    }} else if (item.type === 'fifo') {{
                        chartItem.classList.add('fifo-type');
                    }}

                    const closeBtn = document.createElement('button');
                    closeBtn.className = panel.id === 'tracker-panel' ? 'close-item-btn' : 'close-chart-btn';
                    closeBtn.textContent = '×';
                    closeBtn.onclick = () => closeChartItem(index);

                    // 创建标题头部
                    const header = document.createElement('div');
                    header.className = 'chart-item-header';
                    if (item.type === 'tracker') {{
                        header.classList.add('tracker');
                        header.textContent = `Tracker使用: ${{item.ipType}} @ Pos ${{item.ipPos}}`;
                    }} else if (item.type === 'fifo') {{
                        header.classList.add('fifo');
                        // 构造FIFO标题：FIFO使用情况: [DieX] 节点Y category-type NETWORK [ChN]
                        const networkLabel = {{'req': 'REQ', 'rsp': 'RSP', 'data': 'DATA'}}[item.networkType] || item.networkType?.toUpperCase() || '';
                        const dieInfo = (item.dieId !== undefined && item.dieId !== 0) ? `Die${{item.dieId}} ` : '';
                        const chInfo = (item.chIdx !== undefined && item.chIdx !== 0) ? ` Ch${{item.chIdx}}` : '';
                        header.textContent = `FIFO使用情况: ${{dieInfo}}节点${{item.nodePos}} ${{item.category}}-${{item.fifoType}} ${{networkLabel}}${{chInfo}}`;
                    }}

                    const chartDiv = document.createElement('div');
                    chartDiv.id = `chart-item-${{index}}`;
                    chartDiv.className = chartClass;

                    chartItem.appendChild(closeBtn);
                    chartItem.appendChild(header);
                    chartItem.appendChild(chartDiv);
                    container.appendChild(chartItem);

                    setTimeout(() => {{
                        if (item.type === 'fifo') {{
                            renderFifoChart(item, `chart-item-${{index}}`);
                        }} else if (item.type === 'tracker' && window.createTrackerChart) {{
                            window.createTrackerChart(item.ipData, item.ipType, item.ipPos, `chart-item-${{index}}`, item.dieId);
                        }}
                    }}, 10);
                }});
            }}

            function closeChartItem(index) {{
                let queue = window.trackerQueue || window.chartQueue;
                if (queue && index < queue.length) {{
                    queue.splice(index, 1);
                    window.updateAllCharts();
                }}
            }}

            window.closeChartPanel = function() {{
                const panel = document.getElementById('tracker-panel') || document.getElementById('chart-panel');
                if (panel) panel.classList.remove('active');
                if (window.trackerQueue) window.trackerQueue.length = 0;
                if (window.chartQueue) window.chartQueue.length = 0;
            }};

            // 渲染FIFO深度曲线
            function renderFifoChart(item, targetDivId) {{
                const {{ times, depths, capacity, title }} = item;

                const trace = {{
                    x: times,
                    y: depths,
                    mode: 'lines',
                    line: {{ shape: 'hv', color: '#2196F3', width: 1.5 }},
                    name: '深度',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(33, 150, 243, 0.2)',
                    hovertemplate: '时间: %{{x:.2f}} ns<br>深度: %{{y}}<extra></extra>'
                }};

                const layout = {{
                    title: false,
                    xaxis: {{ title: '时间 (ns)', showgrid: true, gridcolor: '#eee' }},
                    yaxis: {{
                        title: 'FIFO深度',
                        range: [0, Math.max(capacity * 1.1, Math.max(...depths) * 1.1, 1)],
                        showgrid: true,
                        gridcolor: '#eee',
                        dtick: 1,
                        tick0: 0
                    }},
                    shapes: [{{
                        type: 'line',
                        x0: times[0],
                        x1: times[times.length - 1],
                        y0: capacity,
                        y1: capacity,
                        line: {{ dash: 'dash', color: 'red', width: 2 }}
                    }}],
                    annotations: [{{
                        x: times[times.length - 1],
                        y: capacity,
                        text: `容量: ${{capacity}}`,
                        showarrow: false,
                        xanchor: 'right',
                        yanchor: 'bottom',
                        font: {{ color: 'red', size: 12 }}
                    }}],
                    margin: {{ t: 10, b: 50, l: 60, r: 20 }},
                    showlegend: false,
                    hovermode: 'closest'
                }};

                Plotly.newPlot(targetDivId, [trace], layout, {{ responsive: true }});
            }}

            // 显示FIFO深度曲线（添加到共享面板）
            function showFifoChart(dieId, nodePos, category, fifoType, networkType, chIdx) {{
                const key = `${{dieId}}_${{nodePos}}_${{category}}_${{fifoType}}_${{networkType}}_${{chIdx}}`;

                const eventData = fifoEventsData[key];
                const capacity = eventData ? (eventData.capacity || 1) : 1;

                // 从事件构建时序数据
                let times = [0];
                let depths = [0];

                if (eventData && eventData.events && eventData.events.length > 0) {{
                    const events = eventData.events;
                    let currentDepth = 0;

                    for (const [cycle, delta] of events) {{
                        currentDepth += delta;
                        times.push(cycle / cyclesPerNs);  // 转换为ns
                        depths.push(currentDepth);
                    }}

                    // 添加最后一个点（保持最后深度）
                    if (times.length > 1) {{
                        const lastTime = times[times.length - 1];
                        times.push(lastTime + 1);
                        depths.push(currentDepth);
                    }}
                }} else {{
                    // 无事件数据时，画一条0横线
                    times = [0, 100];
                    depths = [0, 0];
                }}

                // 生成标题
                const networkLabel = {{'req': 'REQ', 'rsp': 'RSP', 'data': 'DATA'}}[networkType] || networkType;
                const title = `Die ${{dieId}} 节点${{nodePos}} - ${{category}}-${{fifoType}} (${{networkLabel}} Ch${{chIdx}})`;

                // 使用全局队列（优先使用trackerQueue，否则使用chartQueue）
                let queue = window.trackerQueue || window.chartQueue;
                if (!queue) {{
                    window.chartQueue = [];
                    queue = window.chartQueue;
                }}

                const MAX_CHARTS = 4;

                // 检查是否已存在相同的FIFO图表
                const existingIndex = queue.findIndex(item => item.type === 'fifo' && item.key === key);
                if (existingIndex >= 0) {{
                    // 已存在，移除旧的
                    queue.splice(existingIndex, 1);
                }}

                // 如果队列已满，移除最旧的
                if (queue.length >= MAX_CHARTS) {{
                    queue.shift();
                }}

                // 添加新的FIFO图表数据
                queue.push({{
                    type: 'fifo',
                    key: key,
                    times: times,
                    depths: depths,
                    capacity: capacity,
                    title: title,
                    dieId: dieId,
                    nodePos: nodePos,
                    category: category,
                    fifoType: fifoType,
                    networkType: networkType,
                    chIdx: chIdx
                }});

                // 更新显示
                window.updateAllCharts();
            }}
        }}, 500);
    }});
</script>
"""
    return js_code
