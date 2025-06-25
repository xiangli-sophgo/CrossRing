import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib

if sys.platform == "darwin":  # macOS 的系统标识是 'darwin'
    matplotlib.use("macosx")  # 仅在 macOS 上使用该后端

# 设置中文字体（如果标签需要显示中文）
plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows系统
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 读取数据流文件
# file_path = f"../../traffic/output_DeepSeek/step5_data_merge/MLP_MoE_Trace.txt"
# file_path = r"../../traffic/output_v8_32_2K/step5_data_merge/TPS009-Llama2-70B-S4K-O1-W8A8-B128-LMEM2M-AllReduce_Trace.txt"
# file_path = r"../../traffic/TPS009-Llama2-70B-S4K-O1-W8A8-B128-LMEM2M-AllReduce-2KB-new.txt"
# file_path = f"../../traffic/output_v8_32_0427/step5_data_merge/LLama2_Attention_FC_Trace.txt"
# file_path = f"../../traffic/output_v8_0427/step2_hash_addr2node/LLama2_Attention_FC/gmemTrace.TPU1.tdma_instance.txt"
# file_path = f"../../traffic/output_All_reduce_new_0427/step5_data_merge/TPS009-Llama2-70B-S4K-O1-W8A8-B128-LMEM2M-AllReduce_Trace.txt"
file_path = r"../../traffic/0617/R_5x2.txt"
# file_path = r"../../traffic/DeepSeek_0616/step6_ch_map/Add.txt"
data = pd.read_csv(file_path, header=None)

# file_path = r"../../Result/CrossRing/SCM/REQ_RSP/Spare_core_0403_to32_1_2_0/5x4/LLama2_Attention_FC_Trace/Result_ention_FC_R.txt"
# data = pd.read_csv(file_path, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8], skiprows=1)

# 用户输入要筛选的x值
x = int(input("请输入要筛选的src的值x: "))

# 筛选data[1]等于x的行
filtered_data = data[data[1] == x]

if filtered_data.empty:
    print(f"没有找到src等于{x}的记录")
else:
    # 分离读取(R)和写入(W)操作
    read_data = filtered_data[filtered_data[5] == "R"]
    write_data = filtered_data[filtered_data[5] == "W"]

    # 创建画布
    plt.figure(figsize=(14, 12))

    # 1. 读取操作的统计和图表
    if not read_data.empty:
        # 频率统计
        read_freq = read_data[3].value_counts().reset_index()
        read_freq.columns = ["Value", "Frequency"]
        read_freq["Value"] = read_freq["Value"].astype(int)
        read_freq = read_freq.sort_values(by="Value").reset_index(drop=True)

        # 时间统计
        read_time = read_data.groupby(3)[0].agg(["min", "max", "mean", "count"])
        read_time = read_time.reset_index()
        read_time.columns = ["Value", "First_Time", "Last_Time", "Avg_Time", "Count"]
        read_time["Value"] = read_time["Value"].astype(int)
        read_time = read_time.sort_values(by="Value").reset_index(drop=True)

        # 绘制读取频率分布
        plt.subplot(2, 2, 1)
        plt.bar(read_freq["Value"], read_freq["Frequency"], color="blue")
        plt.title(f"读取操作(R) - dst数值频率分布 (src={x})")
        plt.xlabel("数值")
        plt.ylabel("出现次数")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # 绘制读取时间分布
        plt.subplot(2, 2, 2)
        plt.plot(read_time["Value"], read_time["First_Time"], "bo-", label="首次时间")
        plt.plot(read_time["Value"], read_time["Last_Time"], "ro-", label="最后时间")
        plt.plot(read_time["Value"], read_time["Avg_Time"], "go-", label="平均时间")
        plt.title(f"读取操作(R) - dst数值时间分布 (src={x})")
        plt.xlabel("数值")
        plt.ylabel("时间")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.7)
    else:
        print(f"没有找到data[1]={x}的读取(R)操作记录")

    # 2. 写入操作的统计和图表
    if not write_data.empty:
        # 频率统计
        write_freq = write_data[3].value_counts().reset_index()
        write_freq.columns = ["Value", "Frequency"]
        write_freq["Value"] = write_freq["Value"].astype(int)
        write_freq = write_freq.sort_values(by="Value").reset_index(drop=True)

        # 时间统计
        write_time = write_data.groupby(3)[0].agg(["min", "max", "mean", "count"])
        write_time = write_time.reset_index()
        write_time.columns = ["Value", "First_Time", "Last_Time", "Avg_Time", "Count"]
        write_time["Value"] = write_time["Value"].astype(int)
        write_time = write_time.sort_values(by="Value").reset_index(drop=True)

        # 绘制写入频率分布
        plt.subplot(2, 2, 3)
        plt.bar(write_freq["Value"], write_freq["Frequency"], color="red")
        plt.title(f"写入操作(W) - dst数值频率分布 (src={x})")
        plt.xlabel("数值")
        plt.ylabel("出现次数")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # 绘制写入时间分布
        plt.subplot(2, 2, 4)
        plt.plot(write_time["Value"], write_time["First_Time"], "bo-", label="首次时间")
        plt.plot(write_time["Value"], write_time["Last_Time"], "ro-", label="最后时间")
        plt.plot(write_time["Value"], write_time["Avg_Time"], "go-", label="平均时间")
        plt.title(f"写入操作(W) - dst数值时间分布 (src={x})")
        plt.xlabel("数值")
        plt.ylabel("时间")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.7)
    else:
        print(f"没有找到src={x}的写入(W)操作记录")

    plt.tight_layout()
    plt.show()

    # 保存统计结果为CSV
    # if not read_data.empty:
    #     read_freq.to_csv(f'read_freq_x{x}.csv', index=False)
    #     read_time.to_csv(f'read_time_x{x}.csv', index=False)
    # if not write_data.empty:
    #     write_freq.to_csv(f'write_freq_x{x}.csv', index=False)
    #     write_time.to_csv(f'write_time_x{x}.csv', index=False)

    # 保存图表
    # plt.savefig(f'RW_stats_x{x}.png', dpi=300, bbox_inches='tight')
