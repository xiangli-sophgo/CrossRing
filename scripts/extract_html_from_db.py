#!/usr/bin/env python3
"""从数据库中提取最新的HTML报告"""

import sys
sys.path.insert(0, '..')

from src.database import ResultManager

db = ResultManager()

# 获取experiment_id=2的所有结果
results_dict = db.get_results(experiment_id=2)

print(f"返回类型: {type(results_dict)}")
if isinstance(results_dict, dict):
    print(f"Keys: {list(results_dict.keys())[:5]}")
    # 获取第一个result
    if results_dict:
        first_key = list(results_dict.keys())[0]
        result = results_dict[first_key]
    else:
        print("未找到结果")
        sys.exit(1)
else:
    print("返回格式不是字典")
    sys.exit(1)

if result and "result_html" in result:
    html_content = result["result_html"]

    # 保存到临时文件
    output_path = "/tmp/result_analysis_latest.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML已保存到: {output_path}")

    # 统计图表section数量
    section_count = html_content.count('<div class="section"')
    print(f"图表section数量: {section_count}")

    # 列出所有section标题
    import re
    titles = re.findall(r'<div class="section-title">([^<]+)</div>', html_content)
    print("\n图表列表:")
    for i, title in enumerate(titles, 1):
        print(f"  {i}. {title}")
else:
    print("未找到HTML内容")
