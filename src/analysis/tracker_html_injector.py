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
        <button class="close-btn" onclick="closeTrackerPanel()">关闭全部</button>
        <h2 id="tracker-title">Tracker使用情况</h2>
        <div class="tracker-grid" id="tracker-container">
            <!-- tracker-item将由JavaScript动态创建 -->
        </div>
    </div>
    """

    # 生成CSS样式（响应式竖向优先布局）
    tracker_css = """
    <style>
        .tracker-section {
            position: fixed;
            right: 10px;
            top: 10px;
            width: 1300px;  /* 默认宽度，适合2列 */
            max-width: 95vw;
            max-height: 95vh;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: none;
            overflow-y: auto;
            z-index: 10000;
            transition: width 0.3s ease;
        }
        /* 1-2个tracker时窄布局 */
        .tracker-section.narrow {
            width: 670px;  /* 620 + padding */
        }
        .tracker-section.active {
            display: block;
        }
        .close-btn {
            float: right;
            cursor: pointer;
            background: #f44336;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .close-btn:hover {
            background: #d32f2f;
        }
        .tracker-grid {
            display: grid;
            grid-template-rows: repeat(2, 470px);  /* 固定2行 */
            grid-auto-columns: 620px;  /* 自动列宽 */
            grid-auto-flow: column;  /* 竖向优先填充 */
            gap: 15px;
            margin-top: 10px;
            max-height: calc(95vh - 60px);
            justify-content: start;
        }
        /* 1个tracker时：单列单行 */
        .tracker-grid[data-count="1"] {
            grid-template-rows: 470px;
            grid-auto-columns: 620px;
            justify-content: center;
        }
        /* 2个tracker时：单列2行 */
        .tracker-grid[data-count="2"] {
            grid-template-rows: repeat(2, 470px);
            grid-auto-columns: 620px;
            justify-content: center;
        }
        /* 3-4个tracker时：2行，自动列 */
        .tracker-grid[data-count="3"],
        .tracker-grid[data-count="4"] {
            grid-template-rows: repeat(2, 470px);
            grid-auto-columns: 620px;
        }
        .tracker-item {
            position: relative;
            background: #f5f5f5;
            padding: 5px;
            border-radius: 6px;
            border: 1px solid #ddd;
            width: 620px;  /* 固定宽度 */
            height: 470px;  /* 固定高度 */
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
            width: 600px;
            height: 450px;
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

        // IP显示队列（最多4个，FIFO）
        const trackerQueue = [];
        const MAX_TRACKERS = 4;

        // 页面加载后自动显示前两个IP的tracker曲线
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('[Tracker] 页面加载完成，准备显示tracker曲线');
            console.log('[Tracker] trackerData:', trackerData);

            // 延迟以确保DOM完全加载
            setTimeout(function() {{
                initializeTrackerListener();  // 启用点击监听
                console.log('[Tracker] 点击监听器已启用，等待用户点击IP');
            }}, 1000);
        }});

        function initializeTrackerListener() {{
            console.log('[Tracker] 开始初始化点击监听器');

            // 只绑定到流量图（第一个图表 chart-0）
            const flowGraphDiv = document.getElementById('chart-0');

            if (!flowGraphDiv) {{
                console.error('[Tracker] 找不到流量图 chart-0');
                return;
            }}

            console.log('[Tracker] 找到流量图chart-0，准备绑定点击事件');

            // 获取plotly元素
            const plotlyDiv = flowGraphDiv.querySelector('.plotly-graph-div') || flowGraphDiv;

            plotlyDiv.on('plotly_click', function(data) {{
                console.log('[Tracker] 流量图点击事件触发!');
                console.log('[Tracker] 点击数据:', data);

                if (data.points && data.points.length > 0) {{
                    const point = data.points[0];
                    console.log('[Tracker] 点击的点:', point);
                    console.log('[Tracker] customdata:', point.customdata);
                    console.log('[Tracker] text:', point.text);

                    // 尝试从不同的customdata格式获取IP信息
                    let dieId, ipType, ipPos;

                    if (point.customdata) {{
                        if (Array.isArray(point.customdata)) {{
                            // 格式: [die_id, ip_type, ip_pos]
                            dieId = point.customdata[0];
                            ipType = point.customdata[1];
                            ipPos = point.customdata[2];
                            console.log('[Tracker] 从数组customdata解析: dieId=', dieId, 'ipType=', ipType, 'ipPos=', ipPos);
                        }} else if (typeof point.customdata === 'object') {{
                            // 格式: {{die_id, ip_type, ip_pos}}
                            dieId = point.customdata.die_id;
                            ipType = point.customdata.ip_type;
                            ipPos = point.customdata.ip_pos;
                            console.log('[Tracker] 从对象customdata解析: dieId=', dieId, 'ipType=', ipType, 'ipPos=', ipPos);
                        }}
                    }}

                    // 从text中解析IP信息（备用方案）
                    if (!ipType && point.text) {{
                        console.log('[Tracker] 尝试从text解析IP信息:', point.text);

                        // 格式1: "ip_type @ Pos N" (scatter点的text)
                        let match = point.text.match(/(\\w+_\\d+)\\s*@\\s*Pos\\s*(\\d+)/);
                        if (match) {{
                            ipType = match[1];
                            ipPos = parseInt(match[2]);
                            dieId = 0;
                            console.log('[Tracker] 从scatter点text解析: ipType=', ipType, 'ipPos=', ipPos);
                        }} else {{
                            // 格式2: "<b>节点 N</b><br>GDMA_0: XX GB/s<br>" (节点shapes的text)
                            // 忽略节点整体点击，只允许点击具体的IP方块
                            console.log('[Tracker] 点击的是节点整体，忽略（请点击具体的IP方块）');
                            return;
                        }}
                    }}

                    if (ipType && ipPos !== undefined) {{
                        console.log('[Tracker] 准备添加IP到队列: dieId=', dieId, 'ipType=', ipType, 'ipPos=', ipPos);
                        addTrackerToQueue(dieId || 0, ipType, ipPos);
                    }} else {{
                        console.warn('[Tracker] 无法从点击数据中提取IP信息');
                    }}
                }} else {{
                    console.warn('[Tracker] 点击数据中没有points');
                }}
            }});

            console.log('[Tracker] 点击监听器绑定完成');
        }}

        function addTrackerToQueue(dieId, ipType, ipPos) {{
            console.log(`[Tracker] addTrackerToQueue调用: dieId=${{dieId}}, ipType=${{ipType}}, ipPos=${{ipPos}}`);

            // 检查tracker数据是否存在
            const dieData = trackerData[dieId.toString()];
            if (!dieData) {{
                console.error(`[Tracker] 没有找到Die ${{dieId}}的数据`);
                console.log('[Tracker] 可用的Die:', Object.keys(trackerData));
                alert(`没有找到Die ${{dieId}}的tracker数据`);
                return;
            }}

            const ipTypeData = dieData[ipType];
            if (!ipTypeData) {{
                console.error(`[Tracker] 没有找到${{ipType}}的数据`);
                console.log(`[Tracker] Die${{dieId}}可用的IP类型:`, Object.keys(dieData));
                alert(`没有找到${{ipType}}的tracker数据`);
                return;
            }}

            const ipData = ipTypeData[ipPos.toString()];
            if (!ipData) {{
                console.error(`[Tracker] 没有找到${{ipType}}@${{ipPos}}的数据`);
                console.log(`[Tracker] ${{ipType}}可用的位置:`, Object.keys(ipTypeData));
                alert(`没有找到${{ipType}}@${{ipPos}}的tracker数据`);
                return;
            }}

            console.log(`[Tracker] 找到数据:`, ipData);

            // 检查是否已经在队列中
            const ipKey = `${{dieId}}_${{ipType}}_${{ipPos}}`;
            const existingIndex = trackerQueue.findIndex(item => item.key === ipKey);
            if (existingIndex !== -1) {{
                // 已存在，不重复添加
                return;
            }}

            // 添加到队列，如果满了则移除最旧的
            if (trackerQueue.length >= MAX_TRACKERS) {{
                trackerQueue.shift(); // 移除最旧的
            }}

            trackerQueue.push({{ key: ipKey, dieId, ipType, ipPos, ipData }});
            console.log(`[Tracker] 队列更新，当前长度: ${{trackerQueue.length}}`);

            // 显示tracker面板
            const panel = document.getElementById('tracker-panel');
            if (panel) {{
                panel.classList.add('active');
                console.log('[Tracker] tracker-panel已激活显示');
            }} else {{
                console.error('[Tracker] 找不到tracker-panel元素');
            }}

            // 更新所有图表
            updateAllTrackerCharts();
        }}

        function updateAllTrackerCharts() {{
            console.log(`[Tracker] updateAllTrackerCharts调用，队列长度: ${{trackerQueue.length}}`);

            const container = document.getElementById('tracker-container');
            const panel = document.getElementById('tracker-panel');

            // 如果队列为空，隐藏整个面板
            if (trackerQueue.length === 0) {{
                panel.classList.remove('active');
                console.log('[Tracker] 队列为空，隐藏面板');
                return;
            }}

            // 显示面板
            panel.classList.add('active');

            // 根据tracker数量调整面板宽度
            if (trackerQueue.length <= 2) {{
                panel.classList.add('narrow');
                console.log('[Tracker] 1-2个tracker，使用窄布局');
            }} else {{
                panel.classList.remove('narrow');
                console.log('[Tracker] 3+个tracker，使用宽布局');
            }}

            // 清空容器
            container.innerHTML = '';

            // 设置grid的data-count属性用于CSS响应式
            container.setAttribute('data-count', trackerQueue.length);
            console.log(`[Tracker] 设置data-count=${{trackerQueue.length}}`);

            // 动态创建tracker-item
            trackerQueue.forEach((item, index) => {{
                const trackerItem = document.createElement('div');
                trackerItem.className = 'tracker-item';
                trackerItem.innerHTML = `
                    <button class="close-item-btn" onclick="closeTrackerItem(${{index}})">×</button>
                    <div id="tracker-chart-${{index}}" class="tracker-chart"></div>
                `;
                container.appendChild(trackerItem);

                console.log(`[Tracker] 创建tracker-item ${{index}}: ${{item.ipType}}@Pos${{item.ipPos}}`);

                // 延迟渲染图表，确保DOM已插入
                setTimeout(() => {{
                    createTrackerChart(item.ipData, item.ipType, item.ipPos, `tracker-chart-${{index}}`);
                }}, 10);
            }});
        }}

        function closeTrackerItem(index) {{
            console.log(`[Tracker] 关闭tracker: ${{index}}`);
            if (index < trackerQueue.length) {{
                trackerQueue.splice(index, 1);
                updateAllTrackerCharts();  // 重新渲染整个容器
            }}
        }}

        function closeTrackerPanel() {{
            const panel = document.getElementById('tracker-panel');
            panel.classList.remove('active');
            trackerQueue.length = 0; // 清空队列
        }}

        function createTrackerChart(ipData, ipType, ipPos, targetDivId) {{
            console.log(`[Tracker] createTrackerChart调用: ipType=${{ipType}}, ipPos=${{ipPos}}, targetDiv=${{targetDivId}}`);
            console.log('[Tracker] ipData:', ipData);

            const trackerEvents = ipData.events;
            const totalAllocated = ipData.total_allocated;
            const totalConfig = ipData.total_config;

            console.log('[Tracker] trackerEvents:', trackerEvents);

            // 准备数据
            const traces = [];
            const networkFrequency = 2.0; // GHz

            // Tracker类型映射
            const trackerNames = {{
                'rn_read': 'RN读Tracker',
                'rn_write': 'RN写Tracker',
                'sn_ro': 'SN只读Tracker(RO)',
                'sn_share': 'SN共享Tracker(Share)'
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
                    line: {{ width: 2.5, shape: 'hv' }},  // 阶梯线
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

            // 布局
            const layout = {{
                title: `${{ipType}}@${{ipPos}}`,
                xaxis: {{ title: '时间 (ns)' }},
                yaxis: {{
                    title: '使用个数',
                    range: [0, Math.max(Math.ceil(maxUsageCount * 1.1), 1)]
                }},
                height: 450,
                width: 600,
                margin: {{ l: 60, r: 20, t: 50, b: 50 }},
                hovermode: 'closest',  // 改为closest，只显示最近的曲线信息
                legend: {{ orientation: 'h', y: 1.12 }}
            }};

            // 渲染图表
            const targetDiv = document.getElementById(targetDivId);
            if (!targetDiv) {{
                console.error(`[Tracker] 找不到目标div: ${{targetDivId}}`);
                return;
            }}

            try {{
                Plotly.newPlot(targetDivId, traces, layout, {{displayModeBar: false}});
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
