"""
集成可视化HTML生成器

将多个Plotly图表合并到单一HTML文件中，支持垂直排列布局。
"""

from typing import Dict, Optional, List, Tuple
import plotly.graph_objects as go


class IntegratedVisualizer:
    """集成多个Plotly图表到单一HTML文件"""

    def __init__(self):
        self.charts = []  # [(title, fig, custom_html, custom_js), ...]

    def add_chart(self, title: str, fig: Optional[go.Figure], custom_html: Optional[str] = None, custom_js: Optional[str] = None):
        """
        添加图表到集成可视化

        Args:
            title: 图表章节标题
            fig: Plotly Figure对象,或None表示纯HTML内容
            custom_html: 可选的自定义HTML（如按钮控件，插入在图表前）
            custom_js: 可选的自定义JavaScript代码(fig=None时为HTML内容)
        """
        self.charts.append((title, fig, custom_html, custom_js))

    def generate_html(self, save_path: str = None, show_fig: bool = False, return_content: bool = False):
        """
        生成集成的HTML文件

        Args:
            save_path: HTML文件保存路径（如果return_content=True则忽略）
            show_fig: 是否在浏览器中打开
            return_content: 如果为True，返回HTML内容字符串而不是写文件

        Returns:
            如果return_content=True: str (HTML内容)
            否则: str (保存路径)
        """
        if not self.charts:
            raise ValueError("没有添加任何图表，无法生成HTML")

        # 生成HTML内容
        html_content = self._build_html()

        if return_content:
            return html_content

        # 保存文件
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # 显示
        if show_fig:
            import webbrowser
            import os

            webbrowser.open("file://" + os.path.abspath(save_path))

        return save_path

    def _build_html(self) -> str:
        """构建完整的HTML内容"""
        # HTML头部
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '    <meta charset="utf-8">',
            "    <title>仿真结果分析报告</title>",
            self._get_css_styles(),
            "</head>",
            "<body>",
            "    <h1>仿真结果分析报告</h1>",
        ]

        # 添加图表容器
        for idx, chart_item in enumerate(self.charts):
            # 兼容旧版3元组和新版4元组
            if len(chart_item) == 4:
                title, fig, custom_html, custom_js = chart_item
            else:
                title, fig, custom_js = chart_item
                custom_html = None

            section_id = f"section-{idx}"
            chart_id = f"chart-{idx}"

            html_parts.append(f'    <div class="section" id="{section_id}">')
            html_parts.append(f'        <div class="section-title">{title}</div>')

            # 插入自定义HTML（如按钮）
            if custom_html:
                html_parts.append(f'        <div class="controls-container">')
                for line in custom_html.split("\n"):
                    if line.strip():
                        html_parts.append(f"            {line}")
                html_parts.append(f"        </div>")

            # 如果fig为None，说明是纯HTML内容（如带宽分析报告）
            if fig is None:
                html_parts.append(f'        <div class="report-content">')
                if custom_js:
                    # 对每行内容添加适当缩进
                    for line in custom_js.split("\n"):
                        html_parts.append(f"            {line}")
                html_parts.append(f"        </div>")
            else:
                html_parts.append(f'        <div class="chart-container" id="{chart_id}"></div>')

            html_parts.append("    </div>")
            html_parts.append("")

        # 使用CDN加载Plotly库
        html_parts.append('<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>')
        html_parts.append("<script>")

        # 等待Plotly加载完成后初始化图表
        html_parts.append("    // 等待Plotly加载完成")
        html_parts.append("    function initializeCharts() {")
        html_parts.append("        console.log('initializeCharts called');")
        html_parts.append("        if (typeof Plotly === 'undefined') {")
        html_parts.append("            console.log('Plotly not ready, retrying in 100ms');")
        html_parts.append("            setTimeout(initializeCharts, 100);")
        html_parts.append("            return;")
        html_parts.append("        }")
        html_parts.append("        console.log('Plotly ready, initializing charts');")

        # 初始化每个图表（需要跟踪实际图表索引，跳过纯HTML内容）
        for idx, chart_item in enumerate(self.charts):
            # 兼容旧版3元组和新版4元组
            if len(chart_item) == 4:
                title, fig, custom_html, custom_js = chart_item
            else:
                title, fig, custom_js = chart_item
                custom_html = None

            # 跳过纯HTML内容（fig为None）
            if fig is None:
                continue

            # 使用原始的section索引来匹配HTML中创建的容器ID
            chart_id = f"chart-{idx}"
            # 获取图表的JSON配置
            fig_json = fig.to_json()
            html_parts.append(f"        // 初始化图表: {title}")
            html_parts.append(f"        (function() {{")
            html_parts.append(f"            try {{")
            html_parts.append(f"                var figData = {fig_json};")
            html_parts.append(f"                var data = figData.data;")
            html_parts.append(f"                var layout = figData.layout;")
            html_parts.append(f"                var config = {{displayModeBar: true, responsive: true}};")
            html_parts.append(f'                console.log("初始化图表 {chart_id}: " + data.length + " traces");')
            html_parts.append(f'                Plotly.newPlot("{chart_id}", data, layout, config);')
            html_parts.append(f'                console.log("图表 {chart_id} 初始化成功");')
            html_parts.append(f"            }} catch(e) {{")
            html_parts.append(f'                console.error("图表 {chart_id} 初始化失败:", e);')
            html_parts.append(f"            }}")
            html_parts.append(f"        }})();")
            html_parts.append("")

            # 添加自定义JavaScript（如果有）
            if custom_js:
                # 需要修改custom_js中的容器ID引用
                modified_js = self._adapt_custom_js(custom_js, idx)
                html_parts.append(f"        // 自定义交互逻辑: {title}")
                html_parts.append(f"        (function() {{")
                # 缩进（函数内部）
                indented_js = "\n".join(["            " + line if line.strip() else line for line in modified_js.split("\n")])
                html_parts.append(indented_js)
                html_parts.append(f"        }})();")
                html_parts.append("")

        html_parts.append("    }  // end initializeCharts function")
        html_parts.append("    // 使用多种方式确保脚本执行")
        html_parts.append("    console.log('Script loaded, document.readyState=' + document.readyState);")
        html_parts.append("    if (document.readyState === 'loading') {")
        html_parts.append("        console.log('Document loading, waiting for DOMContentLoaded');")
        html_parts.append("        document.addEventListener('DOMContentLoaded', function() {")
        html_parts.append("            console.log('DOMContentLoaded fired, calling initializeCharts');")
        html_parts.append("            initializeCharts();")
        html_parts.append("        });")
        html_parts.append("    } else if (document.readyState === 'interactive' || document.readyState === 'complete') {")
        html_parts.append("        console.log('Document already loaded, calling initializeCharts immediately');")
        html_parts.append("        if (typeof Plotly !== 'undefined') {")
        html_parts.append("            initializeCharts();")
        html_parts.append("        } else {")
        html_parts.append("            console.log('Plotly not ready yet, waiting for load event');")
        html_parts.append("            window.addEventListener('load', initializeCharts);")
        html_parts.append("        }")
        html_parts.append("    }")
        html_parts.append("</script>")
        html_parts.append("</body>")
        html_parts.append("</html>")

        return "\n".join(html_parts)

    def _get_css_styles(self) -> str:
        """获取CSS样式"""
        return """    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #1e40af;
            margin-bottom: 40px;
            font-size: 32px;
        }
        .section {
            margin-bottom: 40px;
            background: white;
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section-title {
            font-size: 24px;
            margin-bottom: 15px;
            color: #333;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 10px;
            font-weight: bold;
        }
        .chart-container {
            width: 100%;
            height: 700px;
            min-height: 500px;
            background: white;
        }
        /* FIFO热力图按钮样式 */
        .updatemenu-button.active {
            background-color: #3b82f6 !important;
            color: white !important;
            border: 2px solid #1d4ed8 !important;
            font-weight: bold !important;
        }
        .updatemenu-button {
            transition: all 0.2s ease !important;
        }
        /* 报告内容样式 */
        .report-content {
            padding: 20px;
        }
        .report-section {
            margin-bottom: 30px;
        }
        .report-section h3 {
            color: #1e40af;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        .report-section h4 {
            color: #333;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .report-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            border: 1px solid #d1d5db;
        }
        .report-table tbody tr.header-row {
            background-color: #3b82f6;
            color: white;
            font-weight: bold;
        }
        .report-table tbody tr.header-row td {
            padding: 12px;
            border: 1px solid #2563eb;
        }
        .report-table tbody tr:not(.header-row):nth-child(even) {
            background-color: #f9fafb;
        }
        .report-table tbody tr:not(.header-row):nth-child(odd) {
            background-color: white;
        }
        .report-table tbody tr:not(.header-row):hover {
            background-color: #e0f2fe;
        }
        .report-table {
            table-layout: auto;
        }
        .report-table td {
            padding: 10px 12px;
            border: 1px solid #e5e7eb;
            text-align: center;
        }
        .report-table td:first-child {
            font-weight: 600;
            color: #374151;
            width: 20%;
            text-align: left;
        }
    </style>"""

    def _get_plotly_script(self) -> str:
        """获取Plotly库（内嵌完整库）"""
        # 从第一个图表获取内嵌的Plotly库
        if self.charts:
            first_fig = self.charts[0][1]
            temp_html = first_fig.to_html(include_plotlyjs=True)

            # 查找第一个<script>标签（这是Plotly库）
            import re

            # 匹配第一个<script>标签（包含Plotly库的完整代码）
            match = re.search(r"<script[^>]*>.*?</script>", temp_html, re.DOTALL)
            if match:
                return match.group(0)

        # 如果提取失败，使用CDN
        return '<script src="https://cdn.plot.ly/plotly-2.20.0.min.js"></script>'

    def _adapt_custom_js(self, custom_js: str, chart_index: int) -> str:
        """
        适配自定义JavaScript代码，修改容器选择器

        Args:
            custom_js: 原始JavaScript代码
            chart_index: 图表索引

        Returns:
            str: 修改后的JavaScript代码
        """
        import re

        chart_id = f"chart-{chart_index}"
        js_code = custom_js

        # 去除<style>和<script>标签
        js_code = re.sub(r"<style>.*?</style>", "", js_code, flags=re.DOTALL)
        js_code = js_code.replace("<script>", "").replace("</script>", "")

        # 移除DOMContentLoaded事件监听器（因为已经在window.load中）
        # 匹配 document.addEventListener('DOMContentLoaded', function() { 和对应的 });
        js_code = re.sub(r"document\.addEventListener\('DOMContentLoaded',\s*function\(\)\s*\{", "", js_code)

        # 移除对应的结束括号和分号（在最后一个setTimeout的});之后）
        # 找到最后一个 }); 并移除它
        js_code = re.sub(r"\}\);\s*$", "", js_code.rstrip())

        # 替换所有对plotDiv容器的引用
        # 1. 替换 const plotDiv = ... 声明
        js_code = js_code.replace("const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];", f"const plotDiv = document.getElementById('{chart_id}');")

        # 2. 替换直接的 getElementsByClassName 调用（如果有的话）
        js_code = js_code.replace("document.getElementsByClassName('plotly-graph-div')[0]", f'document.getElementById("{chart_id}")')

        # 3. 替换 querySelectorAll 调用
        js_code = js_code.replace("plotDiv.querySelectorAll('.updatemenu-button')", f"document.getElementById('{chart_id}').querySelectorAll('.updatemenu-button')")

        # 缩进调整（因为包裹在IIFE中）
        lines = js_code.split("\n")
        indented_lines = ["        " + line if line.strip() else line for line in lines]

        return "\n".join(indented_lines)


def create_integrated_report(charts_config: List[Tuple[str, go.Figure, Optional[str]]], save_path: str = None, show_result_analysis: bool = False, return_content: bool = False):
    """
    便捷函数：创建集成报告

    Args:
        charts_config: 图表配置列表 [(title, fig, custom_js), ...]
        save_path: 保存路径（如果return_content=True则忽略）
        show_result_analysis: 是否显示结果分析
        return_content: 如果为True，返回HTML内容而不是写文件

    Returns:
        如果return_content=True: str (HTML内容)
        否则: str (保存路径)
    """
    visualizer = IntegratedVisualizer()

    for chart_item in charts_config:
        # 兼容旧版3元组和新版4元组
        if len(chart_item) == 4:
            title, fig, custom_html, custom_js = chart_item
            visualizer.add_chart(title, fig, custom_html, custom_js)
        else:
            title, fig, custom_js = chart_item
            visualizer.add_chart(title, fig, None, custom_js)

    if not visualizer.charts:
        return "" if return_content else None

    return visualizer.generate_html(save_path, show_result_analysis, return_content=return_content)
