import os
import shutil


def flatten_directory(directory):
    # 如果目录不存在则创建（包括父目录）
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)  # exist_ok避免竞态条件报错
        print(f"目录不存在,已自动创建: {directory}")
    elif not os.path.isdir(directory):
        raise NotADirectoryError(f"路径不是目录: {directory}")

    for dir_name in os.listdir(directory):
        path = os.path.join(directory, dir_name)
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path, topdown=False):
                if len(dirs) == 1 and not files:
                    single_dir = dirs[0]
                    single_dir_path = os.path.join(root, single_dir)

                    # 移动内容
                    for item in os.listdir(single_dir_path):
                        src = os.path.join(single_dir_path, item)
                        dst = os.path.join(root, item)
                        shutil.move(src, dst)

                    # 删除空目录
                    os.rmdir(single_dir_path)


def copy_files_with_suffix(input_path, output_path):
    # 确保输入路径存在
    if not os.path.exists(input_path):
        print("输入路径不存在")
        return

    # 确保输出路径存在,如果不存在则创建
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历输入路径
    for root, dirs, files in os.walk(input_path):
        for file in files:
            # 检查文件后缀
            if file.startswith("gmemTrace.CDMA") or file.endswith("tdma_instance.txt"):
                # 构造完整的文件路径
                full_file_path = os.path.join(root, file)
                # 构建输出目录结构
                relative_path = os.path.relpath(root, input_path)
                dest_dir = os.path.join(output_path, relative_path)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                # 构建目标文件路径
                dest_file_path = os.path.join(dest_dir, file)
                # 复制文件
                shutil.copy2(full_file_path, dest_file_path)
                # print(f"复制文件：{full_file_path} 到 {dest_file_path}")


# 使用示例
def main(input_path, output_path=None):
    output_path = output_path + r"/step1_flatten"
    copy_files_with_suffix(input_path, output_path)
    flatten_directory(output_path)
    print("step1_flatten has been completed.")


if __name__ == "__main__":
    input_path = "Results_LlaMa2-70B-Patterns-v5"
    main(input_path, output_path="../output")
