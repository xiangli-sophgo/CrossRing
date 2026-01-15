"""
Outstanding可视化器 - 向现有的HTML文件注入outstanding事务可视化功能
"""

import os
import json
from typing import Dict


def inject_outstanding_functionality(html_path: str, outstanding_data_path: str) -> str:
    """
    向现有的HTML文件注入Outstanding可视化功能

    Args:
        html_path: HTML文件路径
        outstanding_data_path: outstanding数据JSON文件路径

    Returns:
        修改后的HTML文件路径
    """
    # 读取HTML文件
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 读取outstanding数据
    with open(outstanding_data_path, "r", encoding="utf-8") as f:
        outstanding_data = json.load(f)

    # 检查是否已经注入过（检查完整的div定义，而不是字符串引用）
    if '<div class="outstanding-section" id="outstanding-panel">' in html_content:
        return html_path

    # 生成outstanding面板HTML（动态创建，竖向优先布局）
    outstanding_panel_html = """
    <div class="outstanding-section" id="outstanding-panel">
        <button class="close-btn" onclick="closeOutstandingPanel()" title="关闭全部">×</button>
        <div class="outstanding-grid" id="outstanding-container">
            <!-- outstanding-item将由JavaScript动态创建 -->
        </div>
    </div>
    """

    # 生成CSS样式（响应式竖向优先布局，与FIFO热力图保持一致）
    outstanding_css = """
    <style>
        .outstanding-section {
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
        .outstanding-section.narrow {
            width: 480px;
        }
        .outstanding-section.active {
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
        .outstanding-grid {
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
        .outstanding-grid[data-count="1"] {
            grid-template-columns: 440px;
            justify-content: center;
        }
        /* 2个tracker时：单行2列 */
        .outstanding-grid[data-count="2"] {
            grid-template-columns: repeat(2, 440px);
            justify-content: center;
        }
        /* 3-4个tracker时：2列，自动行 */
        .outstanding-grid[data-count="3"],
        .outstanding-grid[data-count="4"] {
            grid-template-columns: repeat(2, 440px);
        }
        .outstanding-item {
            position: relative;
            background: #f5f5f5;
            padding: 0;
            border-radius: 6px;
            border: 2px solid #1976d2;
            width: 440px;
            height: 320px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .outstanding-item-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #1976d2;
            font-size: 14px;
            flex-shrink: 0;
        }
        .outstanding-chart {
            width: 430px;
            height: 285px;
            flex-shrink: 0;
            margin: 5px auto;
        }
        .chart-item-header {
            padding: 6px 10px;
            font-size: 12px;
            font-weight: bold;
            color: white;
            text-align: center;
        }
        .chart-item-header.outstanding {
            background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
        }
        .chart-item-header.fifo {
            background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
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
    outstanding_js = f"""
    <script>
        // 嵌入Outstanding数据
        const outstandingData = {json.dumps(outstanding_data, ensure_ascii=False)};

        // IP显示队列（最多4个，FIFO）- 使用window全局变量以便与FIFO图表共享
        window.outstandingQueue = [];
        const MAX_OUTSTANDING = 4;

        // 页面加载后初始化点击监听
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('[Outstanding] DOMContentLoaded, waiting 1s to init...');
            setTimeout(initializeOutstandingListener, 1000);
        }});

        function initializeOutstandingListener() {{
            console.log('[Outstanding] initializeOutstandingListener called');
            const flowGraphDiv = document.getElementById('chart-0');
            if (!flowGraphDiv) {{
                console.log('[Outstanding] ERROR: chart-0 not found!');
                // 尝试查找其他可能的图表容器
                const allDivs = document.querySelectorAll('[id^="chart-"]');
                console.log('[Outstanding] Available chart divs:', Array.from(allDivs).map(d => d.id));
                return;
            }}
            console.log('[Outstanding] Found chart-0 div');

            // 检查Plotly图表是否已初始化
            if (!flowGraphDiv.data || !flowGraphDiv.layout) {{
                console.log('[Outstanding] Plotly not ready, retrying in 100ms...');
                setTimeout(initializeOutstandingListener, 100);
                return;
            }}
            console.log('[Outstanding] Plotly ready, bindng click event...');
            console.log('[Outstanding] outstandingData keys:', Object.keys(outstandingData));

            // 使用Plotly原生事件绑定
            flowGraphDiv.on('plotly_click', function(data) {{
                console.log('[Outstanding] plotly_click event fired!');
                console.log('[Outstanding] Click data:', data);
                if (data.points && data.points.length > 0) {{
                    const point = data.points[0];
                    console.log('[Outstanding] Clicked point:', point);
                    console.log('[Outstanding] point.customdata:', point.customdata);
                    console.log('[Outstanding] point.text:', point.text);
                    let dieId, ipType, ipPos;

                    if (point.customdata) {{
                        if (Array.isArray(point.customdata)) {{
                            dieId = point.customdata[0];
                            ipType = point.customdata[1];
                            ipPos = point.customdata[2];
                            console.log('[Outstanding] Parsed from array customdata:', dieId, ipType, ipPos);
                        }} else if (typeof point.customdata === 'object') {{
                            dieId = point.customdata.die_id;
                            ipType = point.customdata.ip_type;
                            ipPos = point.customdata.ip_pos;
                            console.log('[Outstanding] Parsed from object customdata:', dieId, ipType, ipPos);
                        }}
                    }}

                    // 从text中解析IP信息（备用方案）
                    if (!ipType && point.text) {{
                        console.log('[Outstanding] Trying to parse from text...');
                        let match = point.text.match(/(\\w+_\\d+)\\s*@\\s*Pos\\s*(\\d+)/);
                        if (match) {{
                            ipType = match[1];
                            ipPos = parseInt(match[2]);
                            dieId = 0;
                            console.log('[Outstanding] Parsed from text:', dieId, ipType, ipPos);
                        }} else {{
                            console.log('[Outstanding] Text does not match IP pattern, ignoring click');
                            return;
                        }}
                    }}

                    if (ipType && ipPos !== undefined) {{
                        console.log('[Outstanding] Calling addOutstandingToQueue...');
                        addOutstandingToQueue(dieId || 0, ipType, ipPos);
                    }} else {{
                        console.log('[Outstanding] ipType or ipPos undefined, not adding to queue');
                    }}
                }} else {{
                    console.log('[Outstanding] No points in click data');
                }}
            }});
            console.log('[Outstanding] Click event bindng complete');
        }}

        function addOutstandingToQueue(dieId, ipType, ipPos) {{
            console.log('[Outstanding] addOutstandingToQueue called:', dieId, ipType, ipPos);
            const dieData = outstandingData[dieId.toString()];
            if (!dieData) {{
                console.log('[Outstanding] dieData not found for die:', dieId);
                return;
            }}

            const ipTypeData = dieData[ipType];
            if (!ipTypeData) {{
                console.log('[Outstanding] ipTypeData not found for:', ipType);
                return;
            }}

            const ipData = ipTypeData[ipPos.toString()];
            if (!ipData) {{
                console.log('[Outstanding] ipData not found for pos:', ipPos);
                return;
            }}

            // 检查是否已经在队列中
            const ipKey = `${{dieId}}_${{ipType}}_${{ipPos}}`;
            const existingIndex = window.outstandingQueue.findIndex(item => item.key === ipKey);
            if (existingIndex !== -1) {{
                console.log('[Outstanding] Already in queue:', ipKey);
                return;
            }}

            // 添加到队列，如果满了则移除最旧的
            if (window.outstandingQueue.length >= MAX_OUTSTANDING) {{
                window.outstandingQueue.shift();
            }}

            // 添加到队列，包含type字段以便与FIFO图表兼容
            window.outstandingQueue.push({{ type: 'outstanding', key: ipKey, dieId, ipType, ipPos, ipData }});
            console.log('[Outstanding] Added to queue, length:', window.outstandingQueue.length);

            const panel = document.getElementById('outstanding-panel');
            if (panel) panel.classList.add('active');

            // 优先使用统一的updateAllCharts（如果FIFO热力图也存在），否则用自己的
            if (typeof window.updateAllCharts === 'function') {{
                console.log('[Outstanding] Using window.updateAllCharts');
                window.updateAllCharts();
            }} else {{
                console.log('[Outstanding] Using updateAllOutstandingCharts (fallback)');
                updateAllOutstandingCharts();
            }}
        }}

        function updateAllOutstandingCharts() {{
            const container = document.getElementById('outstanding-container');
            const panel = document.getElementById('outstanding-panel');

            if (window.outstandingQueue.length === 0) {{
                panel.classList.remove('active');
                return;
            }}

            panel.classList.add('active');

            if (window.outstandingQueue.length === 1) {{
                panel.classList.add('narrow');
            }} else {{
                panel.classList.remove('narrow');
            }}

            container.innerHTML = '';
            container.setAttribute('data-count', window.outstandingQueue.length);

            // 动态创建outstanding-item
            window.outstandingQueue.forEach((item, index) => {{
                const outstandingItem = document.createElement('div');
                outstandingItem.className = 'outstanding-item';
                outstandingItem.innerHTML = `
                    <button class="close-item-btn" onclick="closeOutstandingItem(${{index}})">×</button>
                    <div id="outstanding-chart-${{index}}" class="outstanding-chart"></div>
                `;
                container.appendChild(outstandingItem);

                setTimeout(() => {{
                    window.createOutstandingChart(item.ipData, item.ipType, item.ipPos, `outstanding-chart-${{index}}`, item.dieId);
                }}, 10);
            }});
        }}

        function closeOutstandingItem(index) {{
            if (index < window.outstandingQueue.length) {{
                window.outstandingQueue.splice(index, 1);
                updateAllOutstandingCharts();
            }}
        }}

        function closeOutstandingPanel() {{
            const panel = document.getElementById('outstanding-panel');
            panel.classList.remove('active');
            window.outstandingQueue.length = 0; // 清空队列
        }}

        window.createOutstandingChart = function(ipData, ipType, ipPos, targetDivId, dieId) {{
            const outstandingEvents = ipData.events;
            const totalAllocated = ipData.total_allocated;
            const totalConfig = ipData.total_config;

            // 准备数据
            const traces = [];
            const networkFrequency = 2.0; // GHz

            // Outstanding类型映射
            const outstandingNames = {{
                'rn_read': 'RN读',
                'rn_write': 'RN写',
                'sn_ro': 'SN读',
                'sn_share': 'SN写',
                'read_retry': 'SN读Retry',
                'write_retry': 'SN写Retry'
            }};

            // Outstanding颜色映射：读=蓝色，写=橙色，retry用紫色和红色区分
            const outstandingColors = {{
                'rn_read': '#1f77b4',      // 蓝色
                'rn_write': '#ff7f0e',     // 橙色
                'sn_ro': '#1f77b4',        // 蓝色
                'sn_share': '#ff7f0e',     // 橙色
                'read_retry': '#9467bd',   // 紫色
                'write_retry': '#d62728'   // 红色
            }};

            // 为每种tracker类型创建曲线
            for (const [outstandingType, eventData] of Object.entries(outstandingEvents)) {{
                // 跳过DB类型（rn_rdb, rn_wdb, sn_wdb）
                if (outstandingType.includes('rdb') || outstandingType.includes('wdb')) {{
                    console.log(`[Outstanding] ${{outstandingType}}: 跳过DB类型`);
                    continue;
                }}

                const allocations = eventData.allocations || [];
                const releases = eventData.releases || [];

                if (allocations.length === 0 && releases.length === 0) {{
                    console.log(`[Outstanding] ${{outstandingType}}: 无事件数据`);
                    continue;
                }}

                console.log(`[Outstanding] ${{outstandingType}}: ${{allocations.length}}次分配, ${{releases.length}}次释放`);

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

                console.log(`[Outstanding] ${{outstandingType}} 时间范围: ${{Math.min(...times)}}-${{Math.max(...times)}} ns`);
                console.log(`[Outstanding] ${{outstandingType}} 使用个数范围: ${{Math.min(...usageCounts)}}-${{Math.max(...usageCounts)}}`);

                traces.push({{
                    x: times,
                    y: usageCounts,
                    mode: 'lines',
                    name: outstandingNames[outstandingType] || outstandingType,
                    line: {{ width: 2.5, shape: 'hv', color: outstandingColors[outstandingType] || '#888888' }},  // 阶梯线，固定颜色
                    customdata: cumulativeAllocations,  // 传递累计分配数据
                    hovertemplate: '时间: %{{x:.1f}} ns' +
                        '<br>当前使用: %{{y}}' +
                        '<br>累计使用: %{{customdata}}' +
                        '<extra></extra>'  // <extra></extra>移除左上角的tracker名称
                }});
            }}

            if (traces.length === 0) {{
                console.warn(`[Outstanding] ${{ipType}}@${{ipPos}} 无outstanding数据`);
                return;
            }}

            console.log(`[Outstanding] 总共创建了${{traces.length}}条曲线`);

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
                    dtick: maxUsageCount <= 10 ? 1 : (maxUsageCount <= 20 ? 2 : Math.ceil(maxUsageCount / 10)),
                    tick0: 0
                }},
                margin: {{ l: 50, r: 15, t: 10, b: 40 }},
                hovermode: 'closest',
                legend: {{
                    orientation: 'h',
                    y: 1.0,
                    yanchor: 'top',
                    xanchor: 'center',
                    x: 0.5
                }},
                showlegend: true
            }};

            // 渲染图表
            const targetDiv = document.getElementById(targetDivId);
            if (!targetDiv) {{
                console.error(`[Outstanding] 找不到目标div: ${{targetDivId}}`);
                return;
            }}

            try {{
                Plotly.newPlot(targetDivId, traces, layout, {{displayModeBar: false, responsive: true}});
                console.log(`[Outstanding] 图表渲染成功: ${{targetDivId}}`);
            }} catch (error) {{
                console.error(`[Outstanding] 渲染失败:`, error);
            }}
        }}
    </script>
    """

    # 注入CSS（在</head>之前）
    html_content = html_content.replace("</head>", outstanding_css + "</head>")

    # 注入面板HTML（在</body>之前）
    html_content = html_content.replace("</body>", outstanding_panel_html + outstanding_js + "</body>")

    # 保存修改后的HTML
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_path


def inject_outstanding_functionality_to_content(html_content: str, outstanding_json: str) -> str:
    """
    向HTML内容注入Outstanding可视化功能（不写文件）

    Args:
        html_content: HTML内容字符串
        outstanding_json: outstanding数据JSON字符串

    Returns:
        修改后的HTML内容字符串
    """
    # 解析outstanding数据
    outstanding_data = json.loads(outstanding_json) if isinstance(outstanding_json, str) else outstanding_json

    # 检查是否已经注入过（检查完整的div定义，而不是字符串引用）
    if '<div class="outstanding-section" id="outstanding-panel">' in html_content:
        return html_content

    # 生成outstanding面板HTML（动态创建，竖向优先布局）
    outstanding_panel_html = """
    <div class="outstanding-section" id="outstanding-panel">
        <button class="close-btn" onclick="closeOutstandingPanel()" title="关闭全部">×</button>
        <div class="outstanding-grid" id="outstanding-container">
            <!-- outstanding-item将由JavaScript动态创建 -->
        </div>
    </div>
    """

    # 生成CSS样式（与inject_outstanding_functionality保持一致）
    outstanding_css = """
    <style>
        .outstanding-section {
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
        .outstanding-section.narrow {
            width: 480px;
        }
        .outstanding-section.active {
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
        .outstanding-grid {
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
        .outstanding-grid[data-count="1"] {
            grid-template-columns: 440px;
            justify-content: center;
        }
        /* 2个tracker时：单行2列 */
        .outstanding-grid[data-count="2"] {
            grid-template-columns: repeat(2, 440px);
            justify-content: center;
        }
        /* 3-4个tracker时：2列，自动行 */
        .outstanding-grid[data-count="3"],
        .outstanding-grid[data-count="4"] {
            grid-template-columns: repeat(2, 440px);
        }
        .outstanding-item {
            position: relative;
            background: #f5f5f5;
            padding: 0;
            border-radius: 6px;
            border: 2px solid #1976d2;
            width: 440px;
            height: 320px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .outstanding-item-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #1976d2;
            font-size: 14px;
            flex-shrink: 0;
        }
        .outstanding-chart {
            width: 430px;
            height: 285px;
            flex-shrink: 0;
            margin: 5px auto;
        }
        .chart-item-header {
            padding: 6px 10px;
            font-size: 12px;
            font-weight: bold;
            color: white;
            text-align: center;
        }
        .chart-item-header.outstanding {
            background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
        }
        .chart-item-header.fifo {
            background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
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

    # 生成JavaScript代码（与inject_outstanding_functionality保持一致）
    outstanding_data_str = json.dumps(outstanding_data, ensure_ascii=False)
    outstanding_js = f"""
    <script>
        console.log('[Outstanding] Script loaded! outstandingData keys:', Object.keys({outstanding_data_str}));

        // 嵌入Outstanding数据
        const outstandingData = {outstanding_data_str};

        // IP显示队列（最多4个，FIFO）- 使用window全局变量以便与FIFO图表共享
        window.outstandingQueue = [];
        const MAX_OUTSTANDING = 4;

        // 页面加载后初始化点击监听
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('[Outstanding] DOMContentLoaded, waiting 1s to init...');
            setTimeout(initializeOutstandingListener, 1000);
        }});

        function initializeOutstandingListener() {{
            console.log('[Outstanding] initializeOutstandingListener called');
            const flowGraphDiv = document.getElementById('chart-0');
            if (!flowGraphDiv) {{
                console.log('[Outstanding] ERROR: chart-0 not found!');
                // 尝试查找其他可能的图表容器
                const allDivs = document.querySelectorAll('[id^="chart-"]');
                console.log('[Outstanding] Available chart divs:', Array.from(allDivs).map(d => d.id));
                return;
            }}
            console.log('[Outstanding] Found chart-0 div');

            // 检查Plotly图表是否已初始化
            if (!flowGraphDiv.data || !flowGraphDiv.layout) {{
                console.log('[Outstanding] Plotly not ready, retrying in 100ms...');
                setTimeout(initializeOutstandingListener, 100);
                return;
            }}
            console.log('[Outstanding] Plotly ready, bindng click event...');
            console.log('[Outstanding] outstandingData keys:', Object.keys(outstandingData));

            // 使用Plotly原生事件绑定
            flowGraphDiv.on('plotly_click', function(data) {{
                console.log('[Outstanding] plotly_click event fired!');
                console.log('[Outstanding] Click data:', data);
                if (data.points && data.points.length > 0) {{
                    const point = data.points[0];
                    console.log('[Outstanding] Clicked point:', point);
                    console.log('[Outstanding] point.customdata:', point.customdata);
                    console.log('[Outstanding] point.text:', point.text);
                    let dieId, ipType, ipPos;

                    if (point.customdata) {{
                        if (Array.isArray(point.customdata)) {{
                            dieId = point.customdata[0];
                            ipType = point.customdata[1];
                            ipPos = point.customdata[2];
                            console.log('[Outstanding] Parsed from array customdata:', dieId, ipType, ipPos);
                        }} else if (typeof point.customdata === 'object') {{
                            dieId = point.customdata.die_id;
                            ipType = point.customdata.ip_type;
                            ipPos = point.customdata.ip_pos;
                            console.log('[Outstanding] Parsed from object customdata:', dieId, ipType, ipPos);
                        }}
                    }}

                    // 从text中解析IP信息（备用方案）
                    if (!ipType && point.text) {{
                        console.log('[Outstanding] Trying to parse from text...');
                        let match = point.text.match(/(\\w+_\\d+)\\s*@\\s*Pos\\s*(\\d+)/);
                        if (match) {{
                            ipType = match[1];
                            ipPos = parseInt(match[2]);
                            dieId = 0;
                            console.log('[Outstanding] Parsed from text:', dieId, ipType, ipPos);
                        }} else {{
                            console.log('[Outstanding] Text does not match IP pattern, ignoring click');
                            return;
                        }}
                    }}

                    if (ipType && ipPos !== undefined) {{
                        console.log('[Outstanding] Calling addOutstandingToQueue...');
                        addOutstandingToQueue(dieId || 0, ipType, ipPos);
                    }} else {{
                        console.log('[Outstanding] ipType or ipPos undefined, not adding to queue');
                    }}
                }} else {{
                    console.log('[Outstanding] No points in click data');
                }}
            }});
            console.log('[Outstanding] Click event bindng complete');
        }}

        function addOutstandingToQueue(dieId, ipType, ipPos) {{
            console.log('[Outstanding] addOutstandingToQueue called:', dieId, ipType, ipPos);
            const dieData = outstandingData[dieId.toString()];
            if (!dieData) {{
                console.log('[Outstanding] dieData not found for die:', dieId);
                return;
            }}

            const ipTypeData = dieData[ipType];
            if (!ipTypeData) {{
                console.log('[Outstanding] ipTypeData not found for:', ipType);
                return;
            }}

            const ipData = ipTypeData[ipPos.toString()];
            if (!ipData) {{
                console.log('[Outstanding] ipData not found for pos:', ipPos);
                return;
            }}

            // 检查是否已经在队列中
            const ipKey = `${{dieId}}_${{ipType}}_${{ipPos}}`;
            const existingIndex = window.outstandingQueue.findIndex(item => item.key === ipKey);
            if (existingIndex !== -1) {{
                console.log('[Outstanding] Already in queue:', ipKey);
                return;
            }}

            // 添加到队列，如果满了则移除最旧的
            if (window.outstandingQueue.length >= MAX_OUTSTANDING) {{
                window.outstandingQueue.shift();
            }}

            // 添加到队列，包含type字段以便与FIFO图表兼容
            window.outstandingQueue.push({{ type: 'outstanding', key: ipKey, dieId, ipType, ipPos, ipData }});
            console.log('[Outstanding] Added to queue, length:', window.outstandingQueue.length);

            const panel = document.getElementById('outstanding-panel');
            if (panel) panel.classList.add('active');

            // 优先使用统一的updateAllCharts（如果FIFO热力图也存在），否则用自己的
            if (typeof window.updateAllCharts === 'function') {{
                console.log('[Outstanding] Using window.updateAllCharts');
                window.updateAllCharts();
            }} else {{
                console.log('[Outstanding] Using updateAllOutstandingCharts (fallback)');
                updateAllOutstandingCharts();
            }}
        }}

        function updateAllOutstandingCharts() {{
            const container = document.getElementById('outstanding-container');
            const panel = document.getElementById('outstanding-panel');

            if (window.outstandingQueue.length === 0) {{
                panel.classList.remove('active');
                return;
            }}

            panel.classList.add('active');

            if (window.outstandingQueue.length === 1) {{
                panel.classList.add('narrow');
            }} else {{
                panel.classList.remove('narrow');
            }}

            container.innerHTML = '';
            container.setAttribute('data-count', window.outstandingQueue.length);

            // 动态创建outstanding-item
            window.outstandingQueue.forEach((item, index) => {{
                const outstandingItem = document.createElement('div');
                outstandingItem.className = 'outstanding-item';
                outstandingItem.innerHTML = `
                    <button class="close-item-btn" onclick="closeOutstandingItem(${{index}})">×</button>
                    <div id="outstanding-chart-${{index}}" class="outstanding-chart"></div>
                `;
                container.appendChild(outstandingItem);

                setTimeout(() => {{
                    window.createOutstandingChart(item.ipData, item.ipType, item.ipPos, `outstanding-chart-${{index}}`, item.dieId);
                }}, 10);
            }});
        }}

        function closeOutstandingItem(index) {{
            if (index < window.outstandingQueue.length) {{
                window.outstandingQueue.splice(index, 1);
                updateAllOutstandingCharts();
            }}
        }}

        function closeOutstandingPanel() {{
            const panel = document.getElementById('outstanding-panel');
            panel.classList.remove('active');
            window.outstandingQueue.length = 0; // 清空队列
        }}

        window.createOutstandingChart = function(ipData, ipType, ipPos, targetDivId, dieId) {{
            const outstandingEvents = ipData.events;
            const totalAllocated = ipData.total_allocated;
            const totalConfig = ipData.total_config;

            // 准备数据
            const traces = [];
            const networkFrequency = 2.0; // GHz

            // Outstanding类型映射
            const outstandingNames = {{
                'rn_read': 'RN读',
                'rn_write': 'RN写',
                'sn_ro': 'SN读',
                'sn_share': 'SN写',
                'read_retry': 'SN读Retry',
                'write_retry': 'SN写Retry'
            }};

            // Outstanding颜色映射：读=蓝色，写=橙色，retry用紫色和红色区分
            const outstandingColors = {{
                'rn_read': '#1f77b4',      // 蓝色
                'rn_write': '#ff7f0e',     // 橙色
                'sn_ro': '#1f77b4',        // 蓝色
                'sn_share': '#ff7f0e',     // 橙色
                'read_retry': '#9467bd',   // 紫色
                'write_retry': '#d62728'   // 红色
            }};

            // 为每种tracker类型创建曲线
            for (const [outstandingType, eventData] of Object.entries(outstandingEvents)) {{
                // 跳过DB类型（rn_rdb, rn_wdb, sn_wdb）
                if (outstandingType.includes('rdb') || outstandingType.includes('wdb')) {{
                    console.log(`[Outstanding] ${{outstandingType}}: 跳过DB类型`);
                    continue;
                }}

                const allocations = eventData.allocations || [];
                const releases = eventData.releases || [];

                if (allocations.length === 0 && releases.length === 0) {{
                    console.log(`[Outstanding] ${{outstandingType}}: 无事件数据`);
                    continue;
                }}

                console.log(`[Outstanding] ${{outstandingType}}: ${{allocations.length}}次分配, ${{releases.length}}次释放`);

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

                console.log(`[Outstanding] ${{outstandingType}} 时间范围: ${{Math.min(...times)}}-${{Math.max(...times)}} ns`);
                console.log(`[Outstanding] ${{outstandingType}} 使用个数范围: ${{Math.min(...usageCounts)}}-${{Math.max(...usageCounts)}}`);

                traces.push({{
                    x: times,
                    y: usageCounts,
                    mode: 'lines',
                    name: outstandingNames[outstandingType] || outstandingType,
                    line: {{ width: 2.5, shape: 'hv', color: outstandingColors[outstandingType] || '#888888' }},  // 阶梯线，固定颜色
                    customdata: cumulativeAllocations,  // 传递累计分配数据
                    hovertemplate: '时间: %{{x:.1f}} ns' +
                        '<br>当前使用: %{{y}}' +
                        '<br>累计使用: %{{customdata}}' +
                        '<extra></extra>'  // <extra></extra>移除左上角的tracker名称
                }});
            }}

            if (traces.length === 0) {{
                console.warn(`[Outstanding] ${{ipType}}@${{ipPos}} 无outstanding数据`);
                return;
            }}

            console.log(`[Outstanding] 总共创建了${{traces.length}}条曲线`);

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
                    dtick: maxUsageCount <= 10 ? 1 : (maxUsageCount <= 20 ? 2 : Math.ceil(maxUsageCount / 10)),
                    tick0: 0
                }},
                margin: {{ l: 50, r: 15, t: 10, b: 40 }},
                hovermode: 'closest',
                legend: {{
                    orientation: 'h',
                    y: 1.0,
                    yanchor: 'top',
                    xanchor: 'center',
                    x: 0.5
                }},
                showlegend: true
            }};

            // 渲染图表
            const targetDiv = document.getElementById(targetDivId);
            if (!targetDiv) {{
                console.error(`[Outstanding] 找不到目标div: ${{targetDivId}}`);
                return;
            }}

            try {{
                Plotly.newPlot(targetDivId, traces, layout, {{displayModeBar: false, responsive: true}});
                console.log(`[Outstanding] 图表渲染成功: ${{targetDivId}}`);
            }} catch (error) {{
                console.error(`[Outstanding] 渲染失败:`, error);
            }}
        }}
    </script>
    """

    # 注入CSS（在</head>之前）
    html_content = html_content.replace("</head>", outstanding_css + "</head>")

    # 注入面板HTML和JS（在</body>之前）
    html_content = html_content.replace("</body>", outstanding_panel_html + outstanding_js + "</body>")

    return html_content
