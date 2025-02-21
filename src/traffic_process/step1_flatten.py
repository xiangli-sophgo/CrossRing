import os
import shutil


def flatten_directory(directory):
    for dir in os.listdir(directory):
        path = os.path.join(directory, dir)
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path, topdown=False):
                # Check if there's only one directory in the current level
                if len(dirs) == 1 and not files:
                    single_dir = dirs[0]
                    single_dir_path = os.path.join(root, single_dir)

                    # Move all contents of the single directory to the current level
                    for item in os.listdir(single_dir_path):
                        s = os.path.join(single_dir_path, item)
                        d = os.path.join(root, item)
                        if os.path.isdir(s):
                            shutil.move(s, d)
                        else:
                            shutil.move(s, root)

                    # Remove the now-empty single directory
                    os.rmdir(single_dir_path)


def copy_files_with_suffix(input_path, output_path):
    # 确保输入路径存在
    if not os.path.exists(input_path):
        print("输入路径不存在")
        return

    # 确保输出路径存在，如果不存在则创建
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
