"""
Tracker HTML注入器 - 向现有的HTML文件注入tracker交互功能
"""

import os
import json
from typing import Dict


def inject_tracker_functionality(html_path: str, tracker_data_path: str) -> str:
    """
    向现有的HTML文件注入tracker交互功能

    Args:
        html_path: HTML文件路径
        tracker_data_path: tracker数据JSON文件路径

    Returns:
        修改后的HTML文件路径
    """
    # 读取HTML文件
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 读取tracker数据
    with open(tracker_data_path, "r", encoding="utf-8") as f:
        tracker_data = json.load(f)

    # 检查是否已经注入过
    if "tracker-panel" in html_content:
        return html_path

    # 生成tracker面板HTML（动态创建，竖向优先布局）
    tracker_panel_html = """
    <div class="tracker-section" id="tracker-panel">
        <button class="close-btn" onclick="closeTrackerPanel()" title="关闭全部">×</button>
        <div class="tracker-grid" id="tracker-container">
            <!-- tracker-item将由JavaScript动态创建 -->
        </div>
    </div>
    """

    # 生成CSS样式（响应式竖向优先布局，与FIFO热力图保持一致）
    tracker_css = """
    <style>
        .tracker-section {
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
        }
        /* 1-2个tracker时窄布局 */
        .tracker-section.narrow {
            width: 480px;
        }
        .tracker-section.active {
            display: block;
        }
        .close-btn {
            position: absolute;
            top: 8px;
            right: 8px;
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
            z-index: 1;
        }
        .close-btn:hover {
            background: #d32f2f;
        }
        .tracker-grid {
            display: grid;
            grid-template-columns: repeat(2, 440px);
            grid-auto-rows: 320px;
            grid-auto-flow: row;
            gap: 10px;
            margin-top: 5px;
            max-height: calc(95vh - 50px);
            justify-content: start;
        }
        /* 1个tracker时：单行单列 */
        .tracker-grid[data-count="1"] {
            grid-template-columns: 440px;
            justify-content: center;
        }
        /* 2个tracker时：单行2列 */
        .tracker-grid[data-count="2"] {
            grid-template-columns: repeat(2, 440px);
            justify-content: center;
        }
        /* 3-4个tracker时：2列，自动行 */
        .tracker-grid[data-count="3"],
        .tracker-grid[data-count="4"] {
            grid-template-columns: repeat(2, 440px);
        }
        .tracker-item {
            position: relative;
            background: #f5f5f5;
            padding: 5px;
            border-radius: 6px;
            border: 1px solid #ddd;
            width: 440px;
            height: 320px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .tracker-item-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #1976d2;
            font-size: 14px;
            flex-shrink: 0;
        }
        .tracker-chart {
            width: 430px;
            height: 310px;
            flex-shrink: 0;
        }
        .close-item-btn {
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
        }
        .close-item-btn:hover {
            background: #f57c00;
        }
    </style>
    """

    # 生成JavaScript代码（支持2x2网格和FIFO队列）
    tracker_js = f"""
    <script>
        // 嵌入tracker数据
        const trackerData = {json.dumps(tracker_data, ensure_ascii=False)};

        // IP显示队列（最多4个，FIFO）- 使用window全局变量以便与FIFO图表共享
        window.trackerQueue = [];
        const MAX_TRACKERS = 4;

        // 页面加载后初始化点击监听
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('[Tracker] DOMContentLoaded, waiting 1s to init...');
            setTimeout(initializeTrackerListener, 1000);
        }});

        function initializeTrackerListener() {{
            console.log('[Tracker] initializeTrackerListener called');
            const flowGraphDiv = document.getElementById('chart-0');
            if (!flowGraphDiv) {{
                console.log('[Tracker] ERROR: chart-0 not found!');
                // 尝试查找其他可能的图表容器
                const allDivs = document.querySelectorAll('[id^="chart-"]');
                console.log('[Tracker] Available chart divs:', Array.from(allDivs).map(d => d.id));
                return;
            }}
            console.log('[Tracker] Found chart-0 div');

            // 检查Plotly图表是否已初始化
            if (!flowGraphDiv.data || !flowGraphDiv.layout) {{
                console.log('[Tracker] Plotly not ready, retrying in 100ms...');
                setTimeout(initializeTrackerListener, 100);
                return;
            }}
            console.log('[Tracker] Plotly ready, bindng click event...');
            console.log('[Tracker] trackerData keys:', Object.keys(trackerData));

            // 使用Plotly原生事件绑定
            flowGraphDiv.on('plotly_click', function(data) {{
                console.log('[Tracker] plotly_click event fired!');
                console.log('[Tracker] Click data:', data);
                if (data.points && data.points.length > 0) {{
                    const point = data.points[0];
                    console.log('[Tracker] Clicked point:', point);
                    console.log('[Tracker] point.customdata:', point.customdata);
                    console.log('[Tracker] point.text:', point.text);
                    let dieId, ipType, ipPos;

                    if (point.customdata) {{
                        if (Array.isArray(point.customdata)) {{
                            dieId = point.customdata[0];
                            ipType = point.customdata[1];
                            ipPos = point.customdata[2];
                            console.log('[Tracker] Parsed from array customdata:', dieId, ipType, ipPos);
                        }} else if (typeof point.customdata === 'object') {{
                            dieId = point.customdata.die_id;
                            ipType = point.customdata.ip_type;
                            ipPos = point.customdata.ip_pos;
                            console.log('[Tracker] Parsed from object customdata:', dieId, ipType, ipPos);
                        }}
                    }}

                    // 从text中解析IP信息（备用方案）
                    if (!ipType && point.text) {{
                        console.log('[Tracker] Trying to parse from text...');
                        let match = point.text.match(/(\\w+_\\d+)\\s*@\\s*Pos\\s*(\\d+)/);
                        if (match) {{
                            ipType = match[1];
                            ipPos = parseInt(match[2]);
                            dieId = 0;
                            console.log('[Tracker] Parsed from text:', dieId, ipType, ipPos);
                        }} else {{
                            console.log('[Tracker] Text does not match IP pattern, ignoring click');
                            return;
                        }}
                    }}

                    if (ipType && ipPos !== undefined) {{
                        console.log('[Tracker] Calling addTrackerToQueue...');
                        addTrackerToQueue(dieId || 0, ipType, ipPos);
                    }} else {{
                        console.log('[Tracker] ipType or ipPos undefined, not adding to queue');
                    }}
                }} else {{
                    console.log('[Tracker] No points in click data');
                }}
            }});
            console.log('[Tracker] Click event bindng complete');
        }}

        function addTrackerToQueue(dieId, ipType, ipPos) {{
            console.log('[Tracker] addTrackerToQueue called:', dieId, ipType, ipPos);
            const dieData = trackerData[dieId.toString()];
            if (!dieData) {{
                console.log('[Tracker] dieData not found for die:', dieId);
                return;
            }}

            const ipTypeData = dieData[ipType];
            if (!ipTypeData) {{
                console.log('[Tracker] ipTypeData not found for:', ipType);
                return;
            }}

            const ipData = ipTypeData[ipPos.toString()];
            if (!ipData) {{
                console.log('[Tracker] ipData not found for pos:', ipPos);
                return;
            }}

            // 检查是否已经在队列中
            const ipKey = `${{dieId}}_${{ipType}}_${{ipPos}}`;
            const existingIndex = window.trackerQueue.findIndex(item => item.key === ipKey);
            if (existingIndex !== -1) {{
                console.log('[Tracker] Already in queue:', ipKey);
                return;
            }}

            // 添加到队列，如果满了则移除最旧的
            if (window.trackerQueue.length >= MAX_TRACKERS) {{
                window.trackerQueue.shift();
            }}

            // 添加到队列，包含type字段以便与FIFO图表兼容
            window.trackerQueue.push({{ type: 'tracker', key: ipKey, dieId, ipType, ipPos, ipData }});
            console.log('[Tracker] Added to queue, length:', window.trackerQueue.length);

            const panel = document.getElementById('tracker-panel');
            if (panel) panel.classList.add('active');

            // 优先使用统一的updateAllCharts（如果FIFO热力图也存在），否则用自己的
            if (typeof window.updateAllCharts === 'function') {{
                console.log('[Tracker] Using window.updateAllCharts');
                window.updateAllCharts();
            }} else {{
                console.log('[Tracker] Using updateAllTrackerCharts (fallback)');
                updateAllTrackerCharts();
            }}
        }}

        function updateAllTrackerCharts() {{
            const container = document.getElementById('tracker-container');
            const panel = document.getElementById('tracker-panel');

            if (window.trackerQueue.length === 0) {{
                panel.classList.remove('active');
                return;
            }}

            panel.classList.add('active');

            if (window.trackerQueue.length === 1) {{
                panel.classList.add('narrow');
            }} else {{
                panel.classList.remove('narrow');
            }}

            container.innerHTML = '';
            container.setAttribute('data-count', window.trackerQueue.length);

            // 动态创建tracker-item
            window.trackerQueue.forEach((item, index) => {{
                const trackerItem = document.createElement('div');
                trackerItem.className = 'tracker-item';
                trackerItem.innerHTML = `
                    <button class="close-item-btn" onclick="closeTrackerItem(${{index}})">×</button>
                    <div id="tracker-chart-${{index}}" class="tracker-chart"></div>
                `;
                container.appendChild(trackerItem);

                setTimeout(() => {{
                    window.createTrackerChart(item.ipData, item.ipType, item.ipPos, `tracker-chart-${{index}}`, item.dieId);
                }}, 10);
            }});
        }}

        function closeTrackerItem(index) {{
            if (index < window.trackerQueue.length) {{
                window.trackerQueue.splice(index, 1);
                updateAllTrackerCharts();
            }}
        }}

        function closeTrackerPanel() {{
            const panel = document.getElementById('tracker-panel');
            panel.classList.remove('active');
            window.trackerQueue.length = 0; // 清空队列
        }}

        window.createTrackerChart = function(ipData, ipType, ipPos, targetDivId, dieId) {{
            const trackerEvents = ipData.events;
            const totalAllocated = ipData.total_allocated;
            const totalConfig = ipData.total_config;

            // 准备数据
            const traces = [];
            const networkFrequency = 2.0; // GHz

            // Tracker类型映射
            const trackerNames = {{
                'rn_read': 'RN读',
                'rn_write': 'RN写',
                'sn_ro': 'SN读',
                'sn_share': 'SN写',
                'read_retry': 'SN读Retry',
                'write_retry': 'SN写Retry'
            }};

            // Tracker颜色映射：读=蓝色，写=橙色，retry用紫色和红色区分
            const trackerColors = {{
                'rn_read': '#1f77b4',      // 蓝色
                'rn_write': '#ff7f0e',     // 橙色
                'sn_ro': '#1f77b4',        // 蓝色
                'sn_share': '#ff7f0e',     // 橙色
                'read_retry': '#9467bd',   // 紫色
                'write_retry': '#d62728'   // 红色
            }};

            // 为每种tracker类型创建曲线
            for (const [trackerType, eventData] of Object.entries(trackerEvents)) {{
                // 跳过DB类型（rn_rdb, rn_wdb, sn_wdb）
                if (trackerType.includes('rdb') || trackerType.includes('wdb')) {{
                    console.log(`[Tracker] ${{trackerType}}: 跳过DB类型`);
                    continue;
                }}

                const allocations = eventData.allocations || [];
                const releases = eventData.releases || [];

                if (allocations.length === 0 && releases.length === 0) {{
                    console.log(`[Tracker] ${{trackerType}}: 无事件数据`);
                    continue;
                }}

                console.log(`[Tracker] ${{trackerType}}: ${{allocations.length}}次分配, ${{releases.length}}次释放`);

                // 构建事件流
                const events = [];
                allocations.forEach(cycle => events.push([cycle, +1]));
                releases.forEach(cycle => events.push([cycle, -1]));
                events.sort((a, b) => a[0] - b[0]);

                // 累加计算使用个数和累计分配次数
                const times = [0];
                const usageCounts = [0];
                const cumulativeAllocations = [0];  // 新增：累计分配次数
                let currentUsage = 0;
                let cumulativeCount = 0;

                for (const [cycle, delta] of events) {{
                    currentUsage += delta;
                    if (delta > 0) {{  // 只有分配时才增加累计
                        cumulativeCount += delta;
                    }}
                    times.push(cycle / networkFrequency);  // 转换为ns
                    usageCounts.push(currentUsage);
                    cumulativeAllocations.push(cumulativeCount);
                }}

                console.log(`[Tracker] ${{trackerType}} 时间范围: ${{Math.min(...times)}}-${{Math.max(...times)}} ns`);
                console.log(`[Tracker] ${{trackerType}} 使用个数范围: ${{Math.min(...usageCounts)}}-${{Math.max(...usageCounts)}}`);

                traces.push({{
                    x: times,
                    y: usageCounts,
                    mode: 'lines',
                    name: trackerNames[trackerType] || trackerType,
                    line: {{ width: 2.5, shape: 'hv', color: trackerColors[trackerType] || '#888888' }},  // 阶梯线，固定颜色
                    customdata: cumulativeAllocations,  // 传递累计分配数据
                    hovertemplate: '时间: %{{x:.1f}} ns' +
                        '<br>当前使用: %{{y}}' +
                        '<br>累计使用: %{{customdata}}' +
                        '<extra></extra>'  // <extra></extra>移除左上角的tracker名称
                }});
            }}

            if (traces.length === 0) {{
                console.warn(`[Tracker] ${{ipType}}@${{ipPos}} 无tracker数据`);
                return;
            }}

            console.log(`[Tracker] 总共创建了${{traces.length}}条曲线`);

            // 计算Y轴最大值
            let maxUsageCount = 0;
            traces.forEach(trace => {{
                maxUsageCount = Math.max(maxUsageCount, ...trace.y);
            }});

            // 布局（尺寸与容器匹配）
            const layout = {{
                title: false,
                xaxis: {{ title: '时间 (ns)' }},
                yaxis: {{
                    title: '使用个数',
                    range: [0, Math.max(Math.ceil(maxUsageCount * 1.1), 1)],
                    dtick: 1,
                    tick0: 0
                }},
                margin: {{ l: 50, r: 15, t: 10, b: 40 }},
                hovermode: 'closest',
                legend: {{ orientation: 'h', y: 1.02, xanchor: 'center', x: 0.5 }}
            }};

            // 渲染图表
            const targetDiv = document.getElementById(targetDivId);
            if (!targetDiv) {{
                console.error(`[Tracker] 找不到目标div: ${{targetDivId}}`);
                return;
            }}

            try {{
                Plotly.newPlot(targetDivId, traces, layout, {{displayModeBar: false, responsive: true}});
                console.log(`[Tracker] 图表渲染成功: ${{targetDivId}}`);
            }} catch (error) {{
                console.error(`[Tracker] 渲染失败:`, error);
            }}
        }}
    </script>
    """

    # 注入CSS（在</head>之前）
    html_content = html_content.replace("</head>", tracker_css + "</head>")

    # 注入面板HTML（在</body>之前）
    html_content = html_content.replace("</body>", tracker_panel_html + tracker_js + "</body>")

    # 保存修改后的HTML
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_path


def inject_tracker_functionality_to_content(html_content: str, tracker_json: str) -> str:
    """
    向HTML内容注入tracker交互功能（不写文件）

    Args:
        html_content: HTML内容字符串
        tracker_json: tracker数据JSON字符串

    Returns:
        修改后的HTML内容字符串
    """
    # 解析tracker数据
    tracker_data = json.loads(tracker_json) if isinstance(tracker_json, str) else tracker_json

    # 检查是否已经注入过
    if "tracker-panel" in html_content:
        return html_content

    # 生成tracker面板HTML（动态创建，竖向优先布局）
    tracker_panel_html = """
    <div class="tracker-section" id="tracker-panel">
        <button class="close-btn" onclick="closeTrackerPanel()" title="关闭全部">×</button>
        <div class="tracker-grid" id="tracker-container">
            <!-- tracker-item将由JavaScript动态创建 -->
        </div>
    </div>
    """

    # 生成CSS样式（与inject_tracker_functionality保持一致）
    tracker_css = """
    <style>
        .tracker-section {
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
        }
        /* 1-2个tracker时窄布局 */
        .tracker-section.narrow {
            width: 480px;
        }
        .tracker-section.active {
            display: block;
        }
        .close-btn {
            position: absolute;
            top: 8px;
            right: 8px;
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
            z-index: 1;
        }
        .close-btn:hover {
            background: #d32f2f;
        }
        .tracker-grid {
            display: grid;
            grid-template-columns: repeat(2, 440px);
            grid-auto-rows: 320px;
            grid-auto-flow: row;
            gap: 10px;
            margin-top: 5px;
            max-height: calc(95vh - 50px);
            justify-content: start;
        }
        /* 1个tracker时：单行单列 */
        .tracker-grid[data-count="1"] {
            grid-template-columns: 440px;
            justify-content: center;
        }
        /* 2个tracker时：单行2列 */
        .tracker-grid[data-count="2"] {
            grid-template-columns: repeat(2, 440px);
            justify-content: center;
        }
        /* 3-4个tracker时：2列，自动行 */
        .tracker-grid[data-count="3"],
        .tracker-grid[data-count="4"] {
            grid-template-columns: repeat(2, 440px);
        }
        .tracker-item {
            position: relative;
            background: #f5f5f5;
            padding: 5px;
            border-radius: 6px;
            border: 1px solid #ddd;
            width: 440px;
            height: 320px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .tracker-item-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #1976d2;
            font-size: 14px;
            flex-shrink: 0;
        }
        .tracker-chart {
            width: 430px;
            height: 310px;
            flex-shrink: 0;
        }
        .close-item-btn {
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
        }
        .close-item-btn:hover {
            background: #f57c00;
        }
    </style>
    """

    # 生成JavaScript代码（与inject_tracker_functionality保持一致）
    tracker_data_str = json.dumps(tracker_data, ensure_ascii=False)
    tracker_js = f"""
    <script>
        console.log('[Tracker] Script loaded! trackerData keys:', Object.keys({tracker_data_str}));

        // 嵌入tracker数据
        const trackerData = {tracker_data_str};

        // IP显示队列（最多4个，FIFO）- 使用window全局变量以便与FIFO图表共享
        window.trackerQueue = [];
        const MAX_TRACKERS = 4;

        // 页面加载后初始化点击监听
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('[Tracker] DOMContentLoaded, waiting 1s to init...');
            setTimeout(initializeTrackerListener, 1000);
        }});

        function initializeTrackerListener() {{
            console.log('[Tracker] initializeTrackerListener called');
            const flowGraphDiv = document.getElementById('chart-0');
            if (!flowGraphDiv) {{
                console.log('[Tracker] ERROR: chart-0 not found!');
                // 尝试查找其他可能的图表容器
                const allDivs = document.querySelectorAll('[id^="chart-"]');
                console.log('[Tracker] Available chart divs:', Array.from(allDivs).map(d => d.id));
                return;
            }}
            console.log('[Tracker] Found chart-0 div');

            // 检查Plotly图表是否已初始化
            if (!flowGraphDiv.data || !flowGraphDiv.layout) {{
                console.log('[Tracker] Plotly not ready, retrying in 100ms...');
                setTimeout(initializeTrackerListener, 100);
                return;
            }}
            console.log('[Tracker] Plotly ready, bindng click event...');
            console.log('[Tracker] trackerData keys:', Object.keys(trackerData));

            // 使用Plotly原生事件绑定
            flowGraphDiv.on('plotly_click', function(data) {{
                console.log('[Tracker] plotly_click event fired!');
                console.log('[Tracker] Click data:', data);
                if (data.points && data.points.length > 0) {{
                    const point = data.points[0];
                    console.log('[Tracker] Clicked point:', point);
                    console.log('[Tracker] point.customdata:', point.customdata);
                    console.log('[Tracker] point.text:', point.text);
                    let dieId, ipType, ipPos;

                    if (point.customdata) {{
                        if (Array.isArray(point.customdata)) {{
                            dieId = point.customdata[0];
                            ipType = point.customdata[1];
                            ipPos = point.customdata[2];
                            console.log('[Tracker] Parsed from array customdata:', dieId, ipType, ipPos);
                        }} else if (typeof point.customdata === 'object') {{
                            dieId = point.customdata.die_id;
                            ipType = point.customdata.ip_type;
                            ipPos = point.customdata.ip_pos;
                            console.log('[Tracker] Parsed from object customdata:', dieId, ipType, ipPos);
                        }}
                    }}

                    // 从text中解析IP信息（备用方案）
                    if (!ipType && point.text) {{
                        console.log('[Tracker] Trying to parse from text...');
                        let match = point.text.match(/(\\w+_\\d+)\\s*@\\s*Pos\\s*(\\d+)/);
                        if (match) {{
                            ipType = match[1];
                            ipPos = parseInt(match[2]);
                            dieId = 0;
                            console.log('[Tracker] Parsed from text:', dieId, ipType, ipPos);
                        }} else {{
                            console.log('[Tracker] Text does not match IP pattern, ignoring click');
                            return;
                        }}
                    }}

                    if (ipType && ipPos !== undefined) {{
                        console.log('[Tracker] Calling addTrackerToQueue...');
                        addTrackerToQueue(dieId || 0, ipType, ipPos);
                    }} else {{
                        console.log('[Tracker] ipType or ipPos undefined, not adding to queue');
                    }}
                }} else {{
                    console.log('[Tracker] No points in click data');
                }}
            }});
            console.log('[Tracker] Click event bindng complete');
        }}

        function addTrackerToQueue(dieId, ipType, ipPos) {{
            console.log('[Tracker] addTrackerToQueue called:', dieId, ipType, ipPos);
            const dieData = trackerData[dieId.toString()];
            if (!dieData) {{
                console.log('[Tracker] dieData not found for die:', dieId);
                return;
            }}

            const ipTypeData = dieData[ipType];
            if (!ipTypeData) {{
                console.log('[Tracker] ipTypeData not found for:', ipType);
                return;
            }}

            const ipData = ipTypeData[ipPos.toString()];
            if (!ipData) {{
                console.log('[Tracker] ipData not found for pos:', ipPos);
                return;
            }}

            // 检查是否已经在队列中
            const ipKey = `${{dieId}}_${{ipType}}_${{ipPos}}`;
            const existingIndex = window.trackerQueue.findIndex(item => item.key === ipKey);
            if (existingIndex !== -1) {{
                console.log('[Tracker] Already in queue:', ipKey);
                return;
            }}

            // 添加到队列，如果满了则移除最旧的
            if (window.trackerQueue.length >= MAX_TRACKERS) {{
                window.trackerQueue.shift();
            }}

            // 添加到队列，包含type字段以便与FIFO图表兼容
            window.trackerQueue.push({{ type: 'tracker', key: ipKey, dieId, ipType, ipPos, ipData }});
            console.log('[Tracker] Added to queue, length:', window.trackerQueue.length);

            const panel = document.getElementById('tracker-panel');
            if (panel) panel.classList.add('active');

            // 优先使用统一的updateAllCharts（如果FIFO热力图也存在），否则用自己的
            if (typeof window.updateAllCharts === 'function') {{
                console.log('[Tracker] Using window.updateAllCharts');
                window.updateAllCharts();
            }} else {{
                console.log('[Tracker] Using updateAllTrackerCharts (fallback)');
                updateAllTrackerCharts();
            }}
        }}

        function updateAllTrackerCharts() {{
            const container = document.getElementById('tracker-container');
            const panel = document.getElementById('tracker-panel');

            if (window.trackerQueue.length === 0) {{
                panel.classList.remove('active');
                return;
            }}

            panel.classList.add('active');

            if (window.trackerQueue.length === 1) {{
                panel.classList.add('narrow');
            }} else {{
                panel.classList.remove('narrow');
            }}

            container.innerHTML = '';
            container.setAttribute('data-count', window.trackerQueue.length);

            // 动态创建tracker-item
            window.trackerQueue.forEach((item, index) => {{
                const trackerItem = document.createElement('div');
                trackerItem.className = 'tracker-item';
                trackerItem.innerHTML = `
                    <button class="close-item-btn" onclick="closeTrackerItem(${{index}})">×</button>
                    <div id="tracker-chart-${{index}}" class="tracker-chart"></div>
                `;
                container.appendChild(trackerItem);

                setTimeout(() => {{
                    window.createTrackerChart(item.ipData, item.ipType, item.ipPos, `tracker-chart-${{index}}`, item.dieId);
                }}, 10);
            }});
        }}

        function closeTrackerItem(index) {{
            if (index < window.trackerQueue.length) {{
                window.trackerQueue.splice(index, 1);
                updateAllTrackerCharts();
            }}
        }}

        function closeTrackerPanel() {{
            const panel = document.getElementById('tracker-panel');
            panel.classList.remove('active');
            window.trackerQueue.length = 0; // 清空队列
        }}

        window.createTrackerChart = function(ipData, ipType, ipPos, targetDivId, dieId) {{
            const trackerEvents = ipData.events;
            const totalAllocated = ipData.total_allocated;
            const totalConfig = ipData.total_config;

            // 准备数据
            const traces = [];
            const networkFrequency = 2.0; // GHz

            // Tracker类型映射
            const trackerNames = {{
                'rn_read': 'RN读',
                'rn_write': 'RN写',
                'sn_ro': 'SN读',
                'sn_share': 'SN写',
                'read_retry': 'SN读Retry',
                'write_retry': 'SN写Retry'
            }};

            // Tracker颜色映射：读=蓝色，写=橙色，retry用紫色和红色区分
            const trackerColors = {{
                'rn_read': '#1f77b4',      // 蓝色
                'rn_write': '#ff7f0e',     // 橙色
                'sn_ro': '#1f77b4',        // 蓝色
                'sn_share': '#ff7f0e',     // 橙色
                'read_retry': '#9467bd',   // 紫色
                'write_retry': '#d62728'   // 红色
            }};

            // 为每种tracker类型创建曲线
            for (const [trackerType, eventData] of Object.entries(trackerEvents)) {{
                // 跳过DB类型（rn_rdb, rn_wdb, sn_wdb）
                if (trackerType.includes('rdb') || trackerType.includes('wdb')) {{
                    console.log(`[Tracker] ${{trackerType}}: 跳过DB类型`);
                    continue;
                }}

                const allocations = eventData.allocations || [];
                const releases = eventData.releases || [];

                if (allocations.length === 0 && releases.length === 0) {{
                    console.log(`[Tracker] ${{trackerType}}: 无事件数据`);
                    continue;
                }}

                console.log(`[Tracker] ${{trackerType}}: ${{allocations.length}}次分配, ${{releases.length}}次释放`);

                // 构建事件流
                const events = [];
                allocations.forEach(cycle => events.push([cycle, +1]));
                releases.forEach(cycle => events.push([cycle, -1]));
                events.sort((a, b) => a[0] - b[0]);

                // 累加计算使用个数和累计分配次数
                const times = [0];
                const usageCounts = [0];
                const cumulativeAllocations = [0];  // 新增：累计分配次数
                let currentUsage = 0;
                let cumulativeCount = 0;

                for (const [cycle, delta] of events) {{
                    currentUsage += delta;
                    if (delta > 0) {{  // 只有分配时才增加累计
                        cumulativeCount += delta;
                    }}
                    times.push(cycle / networkFrequency);  // 转换为ns
                    usageCounts.push(currentUsage);
                    cumulativeAllocations.push(cumulativeCount);
                }}

                console.log(`[Tracker] ${{trackerType}} 时间范围: ${{Math.min(...times)}}-${{Math.max(...times)}} ns`);
                console.log(`[Tracker] ${{trackerType}} 使用个数范围: ${{Math.min(...usageCounts)}}-${{Math.max(...usageCounts)}}`);

                traces.push({{
                    x: times,
                    y: usageCounts,
                    mode: 'lines',
                    name: trackerNames[trackerType] || trackerType,
                    line: {{ width: 2.5, shape: 'hv', color: trackerColors[trackerType] || '#888888' }},  // 阶梯线，固定颜色
                    customdata: cumulativeAllocations,  // 传递累计分配数据
                    hovertemplate: '时间: %{{x:.1f}} ns' +
                        '<br>当前使用: %{{y}}' +
                        '<br>累计使用: %{{customdata}}' +
                        '<extra></extra>'  // <extra></extra>移除左上角的tracker名称
                }});
            }}

            if (traces.length === 0) {{
                console.warn(`[Tracker] ${{ipType}}@${{ipPos}} 无tracker数据`);
                return;
            }}

            console.log(`[Tracker] 总共创建了${{traces.length}}条曲线`);

            // 计算Y轴最大值
            let maxUsageCount = 0;
            traces.forEach(trace => {{
                maxUsageCount = Math.max(maxUsageCount, ...trace.y);
            }});

            // 布局（尺寸与容器匹配）
            const layout = {{
                title: false,
                xaxis: {{ title: '时间 (ns)' }},
                yaxis: {{
                    title: '使用个数',
                    range: [0, Math.max(Math.ceil(maxUsageCount * 1.1), 1)],
                    dtick: 1,
                    tick0: 0
                }},
                margin: {{ l: 50, r: 15, t: 10, b: 40 }},
                hovermode: 'closest',
                legend: {{ orientation: 'h', y: 1.02, xanchor: 'center', x: 0.5 }}
            }};

            // 渲染图表
            const targetDiv = document.getElementById(targetDivId);
            if (!targetDiv) {{
                console.error(`[Tracker] 找不到目标div: ${{targetDivId}}`);
                return;
            }}

            try {{
                Plotly.newPlot(targetDivId, traces, layout, {{displayModeBar: false, responsive: true}});
                console.log(`[Tracker] 图表渲染成功: ${{targetDivId}}`);
            }} catch (error) {{
                console.error(`[Tracker] 渲染失败:`, error);
            }}
        }}
    </script>
    """

    # 注入CSS（在</head>之前）
    html_content = html_content.replace("</head>", tracker_css + "</head>")

    # 注入面板HTML和JS（在</body>之前）
    html_content = html_content.replace("</body>", tracker_panel_html + tracker_js + "</body>")

    return html_content
