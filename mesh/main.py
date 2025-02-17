from noc import NoC
from packet import Request
from datetime import datetime
from output import compute
import json
import os
import csv


# 读取json中的参数
with open("config.json", "r") as f3:
    config = json.load(f3)
end_cycle = float(config["end_cycle"])
# end_cycle = 40000
# freq = config["network_frequency"]
# itlv = config["interleave"]

# 读取输入文件
# file_name = 'Ins_SG2262_LLaMa_MLP_8cluster_64core_Trace.txt'
# file_name = 'Ins_SG2262_LLama2_Attention_FC_8cluster_64core_Trace.txt'
# file_name = 'Ins_SG2262_LLama2_Attention_QKV_8cluster_64core_Trace.txt'
# file_name = "Ins_SG2262_LLama2_MM_QKV_8cluster_64core_Trace.txt"
# file_name = "ins_SG2262_Ring_all_reduce_8cluster_all2all_Trace.txt"
# file_name = "demo1.txt"

# V7 trace
# trace_path = ""
trace_path = r"../traffic/output-v8-32/2M/step6_32core_map/"
# file_name = r"LLama2_Attention_FC_Trace.txt"
# file_name = r"LLaMa2_Attention_QKV_Decode_Trace.txt"
# file_name = r"LLaMa2_MLP_Trace.txt"
file_name = r"LLaMa2_MM_QKV_Trace.txt"
# file_name = "demo2.txt"

with open(trace_path + file_name, "r") as file:
    lines = file.readlines()
max_line = len(lines)
# max_line = 2326529

var1_start = 2
var2_start = 48
var3_start = 48
# lbn = 4
# freq = 4
print("file name: ", file_name)
for var1 in range(var1_start, var1_start + 1):
    for var2 in range(var2_start, var2_start + 1):
        for var3 in range(var3_start, var3_start + 1):
            # config["LBN"] = 4
            # config["LBN"] = var1
            freq = config["network_frequency"] = 2
            # config["fifo_depth"] = 2
            config["sn_r_tracker_outstanding"] = 48
            config["sn_w_tracker_outstanding"] = 48
            config["rn_r_tracker_outstanding"] = 48
            config["rn_w_tracker_outstanding"] = 48
            print(
                f"LBN: {config["LBN"]};",
                f"network_frequency: {config["network_frequency"]}",
                f"fifo_depth: {config["fifo_depth"]}",
                f"sn_r_tracker_outstanding: {config["sn_r_tracker_outstanding"]};",
                f"sn_w_tracker_outstanding: {
                    config["sn_w_tracker_outstanding"]}",
            )
            # 初始化网络
            noc = NoC(config=config)

            # for key in noc.rn:
            # assert key in noc.req_link

            cycle = int(lines[0].strip().split(",")[0])

            current_line = 0

            read_req_num = 0
            write_req_num = 0

            read_pac_num = 0
            write_pac_num = 0

            retry_count = [0, 0]

            read_flit_total_num = 0
            write_flit_total_num = 0

            read_req: list[Request] = []
            write_req: list[Request] = []
            while cycle < end_cycle:
                # print(cycle, noc.sn[(3, 7), 'ddr'].r_tracker_credit)

                # 1. 网络迭代(网络内部频率)
                # 2. RN/SN和上下游交互(1G)
                # 3. 网络输入(1G)

                # 1. 网络迭代(网络内部频率)
                for _ in range(freq):
                    noc.state_update()
                    noc.network_inside_walk()
                    noc.network_link2link()

                # 2. RN / SN和上下游交互(1G)
                noc.rn2noc()
                noc.noc2rn(cycle)
                read_pac_num += noc.rn2dma(read_req, cycle)

                noc.sn2noc(cycle)
                noc.noc2sn(cycle)
                write_pac_num += noc.sn2ddr(write_req, cycle)

                # 3. 网络输入(1G)
                while current_line < max_line:
                    data = lines[current_line].strip().split(",")
                    t = int(data[0])
                    if t > cycle:
                        break
                    else:
                        # assert t == cycle
                        src, dest = noc.id2coord(data[1]), noc.id2coord(data[3])
                        src_type, dest_type = data[2], data[4]
                        req_type = data[5]
                        flit_num = int(data[6])
                        assert src, src_type in noc.rn
                        if req_type == "R":
                            noc.rn[src, src_type].r_tracker.append(
                                Request(
                                    src=src,
                                    loc=src,
                                    dest=dest,
                                    src_type=src_type,
                                    dest_type=dest_type,
                                    cycle_start=cycle,
                                    req_type="R",
                                    flit_num=flit_num,
                                )
                            )
                            read_req_num += 1
                            read_flit_total_num += flit_num
                        else:
                            noc.rn[src, src_type].w_tracker.append(
                                Request(
                                    src=src,
                                    loc=src,
                                    dest=dest,
                                    src_type=src_type,
                                    dest_type=dest_type,
                                    cycle_start=cycle,
                                    req_type="W",
                                    flit_num=flit_num,
                                )
                            )
                            write_req_num += 1
                            write_flit_total_num += flit_num
                        current_line += 1

                if cycle % 1000 == 0:
                    print(
                        f"time: {cycle}ns, "
                        f"read: tx_req={read_req_num}, rx_dat={read_pac_num}, "
                        f"write: tx_req={write_req_num}, rx_dat={write_pac_num}"
                    )
                    # noc.display_all_links_table(noc.dat_link)
                    # noc.plot_links_with_buffers(noc.req_link)

                if read_flit_total_num == read_pac_num and write_flit_total_num == write_pac_num:
                    if current_line == max_line:
                        print(
                            f"Break! "
                            f"time={cycle}ns, "
                            f"read:tx_req={read_req_num}, rx_dat={read_pac_num}, "
                            f"write:tx_req={write_req_num}, rx_dat={write_pac_num}"
                        )
                        break
                    else:
                        cycle = int(lines[current_line].strip().split(",")[0])
                else:
                    cycle += 1

            output_path = (
                "../Result/mesh"
                # + str(config["LBN"])
                # + "-"
                # + str(config["sn_r_tracker_outstanding"])
                # + "-"
                # + str(config["sn_w_tracker_outstanding"])
                # + " "
                + "v8-32_new_mapping/"
                + file_name[:-4]
                + str("-" + datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            )
            if os.path.exists(output_path):
                if os.path.isfile(output_path):
                    os.remove(output_path)
                    os.makedirs(output_path)
            else:
                os.makedirs(output_path)

            # 保存json文件
            with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as json_file:
                json.dump(config, json_file, ensure_ascii=False, indent=4)  # 写入 JSON 数据

            read_retry_num = 0
            if read_req:
                r_file_name = "Result" + file_name[10:-9] + "R" + ".txt"

                f1 = open(os.path.join(output_path, r_file_name), "w", encoding="utf-8")
                print(
                    f"tx_time(ns), src_id, src_type, des_id, des_type, R/W, burst_len, rx_time(ns), retry",
                    file=f1,
                )

                for req in read_req:
                    if req.retry == "Complete":
                        retry = "retry"
                        read_retry_num += 1
                    else:
                        retry = "normal"
                        assert req.retry == "No", print(req.retry)
                    print(
                        f"{req.cycle_start},{noc.coord2id(req.src)},{req.src_type},{noc.coord2id(req.dest)},{req.dest_type},"
                        f"{req.req_type},{req.flit_num},{req.cycle_end},{retry}",
                        file=f1,
                    )
                f1.close()
                print(f"{len(read_req)} Read Request,{read_retry_num} retry!")

            write_retry_num = 0
            if write_req:
                w_file_name = "Result" + file_name[10:-9] + "W" + ".txt"
                f2 = open(os.path.join(output_path, w_file_name), "w", encoding="utf-8")
                print(
                    f"req_begin_time(ns), src_id, src_type, des_id, des_type, R/W, burst_length, flit_receive_time(ns), retry",
                    file=f2,
                )
                for req in write_req:
                    if req.retry == "Complete":
                        retry = "retry"
                        write_retry_num += 1
                    else:
                        retry = "normal"
                        assert req.retry == "No", print(req.retry)
                    print(
                        f"{req.cycle_start},{noc.coord2id(req.src)},{req.src_type},{noc.coord2id(req.dest)},{req.dest_type},"
                        f"{req.req_type},{req.flit_num},{req.cycle_end},{retry}",
                        file=f2,
                    )
                f2.close()
                print(f"{len(write_req)} Write Request,{write_retry_num} retry!")

            # 计算total_result
            output_csv = "../Result/all params output.csv"
            csv_file_exists = os.path.isfile(output_csv)
            with open(output_csv, mode="a", newline="") as output_csv_file:
                results = {
                    "LBN": -1,
                    "network_frequency": -1,
                    "fifo_depth": -1,
                    "sn_w_tracker_outstanding": -1,
                    "WriteRetryNum": -1,
                    "WriteBandWidth": -1,
                    "WriteMaxLatency": -1,
                    "WriteMinLatency": -1,
                    "WriteAvgLatency": -1,
                    "sn_r_tracker_outstanding": -1,
                    "ReadRetryNum": -1,
                    "ReadBandWidth": -1,
                    "ReadMaxLatency": -1,
                    "ReadMinLatency": -1,
                    "ReadAvgLatency": -1,
                    "BreakTime": -1,
                }
                writer = csv.DictWriter(output_csv_file, fieldnames=results.keys())
                if not csv_file_exists:
                    writer.writeheader()

                results["LBN"] = config["LBN"]
                results["sn_w_tracker_outstanding"] = config["sn_w_tracker_outstanding"]
                results["WriteRetryNum"] = write_retry_num
                results["sn_r_tracker_outstanding"] = config["sn_r_tracker_outstanding"]
                results["ReadRetryNum"] = read_retry_num
                results["BreakTime"] = cycle
                results["network_frequency"] = freq
                results["fifo_depth"] = config["fifo_depth"]

                total_result = os.path.join(output_path, "total_result.txt")
                f3 = open(total_result, "w", encoding="utf-8")
                for item in os.listdir(output_path):
                    if item.startswith("Result"):
                        item_path = os.path.join(output_path, item)
                        print(item_path, file=f3)
                        if item_path[-5] == "W":
                            (
                                results["WriteBandWidth"],
                                results["WriteMaxLatency"],
                                results["WriteMinLatency"],
                                results["WriteAvgLatency"],
                            ) = compute(item_path, f3)
                            print(
                                f"WriteBandWidth: {results["WriteBandWidth"]}, WriteAvgLatency: {results['WriteAvgLatency']}, WriteMaxLatency: {results['WriteMaxLatency']}"
                            )
                        elif item_path[-5] == "R":
                            (
                                results["ReadBandWidth"],
                                results["ReadMaxLatency"],
                                results["ReadMinLatency"],
                                results["ReadAvgLatency"],
                            ) = compute(item_path, f3)
                            print(
                                f"ReadBandWidth: {results["ReadBandWidth"]}, ReadAvgLatency: {results['ReadAvgLatency']}, ReadMaxLatency: {results['ReadMaxLatency']}"
                            )
                        print("", file=f3)

                writer.writerow(results)

                f3.close()
