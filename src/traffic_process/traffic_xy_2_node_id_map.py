import os
import re
from pathlib import Path
import glob


def coordinate_to_node(x, y):
    """
    将坐标转换为节点编号
    网格为5行4列，从y=4开始编号
    x0y4=0, x1y4=1, x2y4=2, x3y4=3
    x0y3=4, x1y3=5, x2y3=6, x3y3=7
    以此类推
    """
    # y从4开始，向下递减
    # 节点编号 = (4-y) * 4 + x
    node_id = (4 - y) * 4 + x
    return node_id


def parse_filename(filename):
    """
    解析文件名获取gdma信息
    格式: master_p0_x1_y1.txt
    返回: gdma编号, gdma节点编号
    """
    # 使用正则表达式解析文件名
    pattern = r"master_(p\d+)_x(\d+)_y(\d+)\.txt"
    match = re.match(pattern, filename)

    if not match:
        return None, None

    gdma_raw = match.group(1)  # p0 或 p1
    x = int(match.group(2))
    y = int(match.group(3))

    # 转换gdma编号
    gdma_num = gdma_raw[1:]  # 去掉'p'
    gdma_id = f"gdma_{gdma_num}"

    # 转换坐标为节点编号
    gdma_node = coordinate_to_node(x, y)

    return gdma_id, gdma_node


def parse_line(line):
    """
    解析每一行数据
    格式: 时间,(p编号,x坐标,y坐标),请求类型,burst_length
    这里的p编号是ddr编号
    """
    line = line.strip()
    if not line:
        return None

    # 使用正则表达式解析
    # 匹配格式: 数字,(p数字,x数字,y数字),字母,数字
    pattern = r"(\d+),\((p\d+),x(\d+),y(\d+)\),([A-Z]),(\d+)"
    match = re.match(pattern, line)

    if not match:
        return None

    time = int(match.group(1))
    ddr_raw = match.group(2)  # p0 或 p1 (这是ddr编号)
    x = int(match.group(3))
    y = int(match.group(4))
    request_type = match.group(5)
    burst_length = int(match.group(6))

    # 转换ddr编号
    ddr_num = ddr_raw[1:]  # 去掉'p'
    ddr_id = f"ddr_{ddr_num}"

    # 转换坐标为ddr节点编号
    ddr_node = coordinate_to_node(x, y)

    return {"time": time, "ddr_id": ddr_id, "ddr_node": ddr_node, "request_type": request_type, "burst_length": burst_length}


def process_files(input_folder, output_file):
    """
    处理文件夹中的所有txt文件
    """
    input_path = Path(input_folder)
    all_data = []

    # 获取所有txt文件
    txt_files = list(input_path.glob("master_p*_x*_y*.txt"))

    if not txt_files:
        print(f"在文件夹 {input_folder} 中没有找到符合格式的txt文件")
        print("文件名格式应为: master_p0_x1_y1.txt")
        return

    print(f"找到 {len(txt_files)} 个txt文件")

    # 处理每个文件
    for txt_file in txt_files:
        print(f"正在处理: {txt_file.name}")

        # 解析文件名获取gdma信息
        gdma_id, gdma_node = parse_filename(txt_file.name)

        if gdma_id is None or gdma_node is None:
            print(f"警告: 无法解析文件名 {txt_file.name}")
            continue

        print(f"  GDMA信息: {gdma_id}, 节点编号: {gdma_node}")

        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                parsed_data = parse_line(line)
                if parsed_data:
                    # 添加gdma信息
                    parsed_data["gdma_id"] = gdma_id
                    parsed_data["gdma_node"] = gdma_node
                    all_data.append(parsed_data)
                elif line.strip():  # 如果不是空行但解析失败
                    print(f"警告: {txt_file.name} 第{line_num}行解析失败: {line.strip()}")

        except Exception as e:
            print(f"读取文件 {txt_file.name} 时出错: {e}")

    # 按时间排序
    all_data.sort(key=lambda x: x["time"])
    print(f"数据已按时间排序，时间范围: {all_data[0]['time']} - {all_data[-1]['time']}" if all_data else "没有数据")

    # 写入输出文件
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for data in all_data:
                # 格式: 时间,gdma节点编号,gdma编号,ddr节点编号,ddr编号,请求类型,burst_length
                line = f"{data['time']},{data['gdma_node']},{data['gdma_id']},{data['ddr_node']},{data['ddr_id']},{data['request_type']},{data['burst_length']}\n"
                f.write(line)

        print(f"成功处理 {len(all_data)} 条数据，输出到: {output_file}")

        # 显示一些示例数据
        if all_data:
            print("\n前5条数据示例:")
            print("时间,gdma节点编号,gdma编号,ddr节点编号,ddr编号,请求类型,burst_length")
            for i, data in enumerate(all_data[:5]):
                print(f"{data['time']},{data['gdma_node']},{data['gdma_id']},{data['ddr_node']},{data['ddr_id']},{data['request_type']},{data['burst_length']}")

    except Exception as e:
        print(f"写入输出文件时出错: {e}")


def main():
    """
    主函数：遍历输入根目录下的每个子文件夹，为每个子文件夹生成一个独立的 traffic 文件
    """
    # 设置输入根目录和输出文件夹
    input_root = r"../../traffic/original/xy_node/new_DeepSeek3-p-x-y"  # 根目录，内部含若干子文件夹
    output_folder = r"../../traffic/0617/DeepSeek/"  # 输出文件统一放在此目录

    # 检查输入根目录
    if not os.path.exists(input_root):
        print(f"输入根目录 {input_root} 不存在")
        return

    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    print("开始批量处理子文件夹...")
    # 遍历一级子目录
    for sub_name in sorted(os.listdir(input_root)):
        sub_path = os.path.join(input_root, sub_name)
        if not os.path.isdir(sub_path):
            continue

        # 为该子文件夹生成对应的输出文件
        output_file = os.path.join(output_folder, f"{sub_name}.txt")
        print(f"\n处理子目录: {sub_path}")
        print(f"  输出文件: {output_file}")
        process_files(sub_path, output_file)

    print("\n所有子文件夹处理完成！")


if __name__ == "__main__":
    main()
