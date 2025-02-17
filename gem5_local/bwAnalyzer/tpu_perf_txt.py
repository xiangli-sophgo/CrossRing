import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings("ignore")


START_ADDRS = {
    "ddr": 0x0,
    "l2m": 0x6980000000,
    "slc": 0x6984000000,
}

END_ADDRS = {
    "ddr": 0x1FFFFFFFFF,  # 128GB
    "l2m": 0x6983FFFFFF,  # 64MB
    "slc": 0x6985FFFFFF,  # 32MB
}

TRAFFIC_SRC = {
    0: "G",  # GDMA
    1: "S",  # SDMA
    2: "C",  # CDMA
}


def get_flits_data(txtpath):

    try:
        txt = pd.read_csv(txtpath, sep=",|\t", engine="python")
        selected_columns = ["Core", "TaskID", "TrafSrc", "Type", "Addr", "Size", "Vnet", "StartCycle", "EndCycle", "ReqCycle", "RespCycle"]
        txt = txt[selected_columns]
        txt["Addr"] = txt["Addr"].apply(lambda x: int(x, 16))
        txt_records = txt.to_dict(orient="records")
        read_responses = {f"{r['Core']}_{r['TaskID']}": r for r in txt_records if r["Type"] == "R" and int(r["Vnet"]) == 1}
        write_responses = {f"{w['Core']}_{w['TaskID']}": w for w in txt_records if w["Type"] == "W" and int(w["Vnet"]) == 1}
        flits = defaultdict(lambda: defaultdict(list))
        ex_n = 0
        for elem in txt_records:
            mems_key = None
            addr = elem["Addr"]
            if START_ADDRS["ddr"] <= addr <= END_ADDRS["ddr"]:
                mems_key = "ddr"
            elif START_ADDRS["l2m"] <= addr <= END_ADDRS["l2m"]:
                mems_key = "l2m"
            elif START_ADDRS["slc"] <= addr <= END_ADDRS["slc"]:
                mems_key = "slc"
            else:
                ex_n += 1
                continue
            core_task_key = f"{elem['Core']}_{elem['TaskID']}"
            core_key = f"CORE{elem['Core']}"
            if elem["Type"] == "R" and int(elem["Vnet"]) == 1:
                res = read_responses.get(core_task_key)
                if res:
                    elem["StartCycle"] = res["ReqCycle"]
                else:
                    print(f"Warning: R respond is not found for {core_task_key}!")
                flits[core_key][mems_key].append(elem)
            elif elem["Type"] == "W" and int(elem["Vnet"]) == 0:
                res = write_responses.get(core_task_key)
                if res:
                    elem["EndCycle"] = res["RespCycle"]
                else:
                    print(f"Warning: W respond is not found for {core_task_key}!")
                flits[core_key][mems_key].append(elem)
        if ex_n:
            print(f"Warning: {ex_n} flits are excluded!")
        return flits
    except Exception as e:
        print(f"data parsing error: {e}")
        sys.exit(1)


def check_group(flits, core_group=None, switch=False):

    core_group = core_group or [list(range(18))] if switch else [[i] for i in range(18)]
    core_gidx, core_gmems, core_gips, unknown_ip = [], [], [], False
    for group in core_group:
        group_idx = [f"CORE{core}" for core in group if flits.get(f"CORE{core}")]
        if not group_idx:
            # print(f"Warning: Group({group}) data is empty!")
            continue
        core_mems, core_ips = [], []
        for mem_type in ["ddr", "l2m", "slc"]:
            if any(flits.get(core, {}).get(mem_type) for core in group_idx):
                core_mems.append(mem_type)
                mem_rip, mem_wip = [], []
                for core in group_idx:
                    for flit in flits.get(core, {}).get(mem_type, []):
                        src = TRAFFIC_SRC.get(int(flit["TrafSrc"]), "U")
                        if src == "U":
                            unknown_ip = True
                        if flit["Type"] == "R" and src not in mem_rip:
                            mem_rip.append(src)
                        elif flit["Type"] == "W" and src not in mem_wip:
                            mem_wip.append(src)
                        mem_ip = [mem_rip, mem_wip]
                core_ips.append(mem_ip)
        core_gidx.append(group_idx)
        core_gmems.append(core_mems)
        core_gips.append(core_ips)
    if unknown_ip:
        print("Warning: Unknown IP detected.")
    if core_gidx:
        print(core_gidx)
    else:
        print("Warning: All group data is empty!")
        sys.exit(1)

    return core_gidx, core_gmems, core_gips


def get_bin_idx(flits, core_idx, bins, ips, mems, stack):

    if not stack:
        mems = [mems]
    mem_id = 0
    for mem in mems:
        start_tmin, end_tmax = 0, 0
        start_rtmin, end_rtmax = 0, 0
        start_wtmin, end_wtmax = 0, 0
        id, rid, wid = 0, 0, 0
        for core in core_idx:
            for flit in flits.get(core, {}).get(mem, []):
                id += 1
                if int(flit["StartCycle"]) < start_tmin or id == 1:
                    start_tmin = int(flit["StartCycle"])
                if int(flit["EndCycle"]) > end_tmax or id == 1:
                    end_tmax = int(flit["EndCycle"])
                if flit["Type"] == "R":
                    rid += 1
                    if int(flit["StartCycle"]) < start_rtmin or rid == 1:
                        start_rtmin = int(flit["StartCycle"])
                    if int(flit["EndCycle"]) > end_rtmax or rid == 1:
                        end_rtmax = int(flit["EndCycle"])
                elif flit["Type"] == "W":
                    wid += 1
                    if int(flit["StartCycle"]) < start_wtmin or wid == 1:
                        start_wtmin = int(flit["StartCycle"])
                    if int(flit["EndCycle"]) > end_wtmax or wid == 1:
                        end_wtmax = int(flit["EndCycle"])
        if stack:
            curr_t = [start_tmin, end_tmax]
            curr_rt = [start_rtmin, end_rtmax] if rid else [0, float("inf")]
            curr_wt = [start_wtmin, end_wtmax] if wid else [0, float("inf")]
            bins_ct = curr_t
            if ips[mem_id][0] and ips[mem_id][1]:
                if max(curr_rt[0], curr_wt[0]) < min(curr_rt[1], curr_wt[1]):
                    curr_t = [max(curr_rt[0], curr_wt[0]), min(curr_rt[1], curr_wt[1])]
            if mem_id == 0:
                bins_t, t, rt, wt = bins_ct, curr_t, curr_rt, curr_wt
            else:
                if bins_t[0] > bins_ct[0]:
                    bins_t[0] = bins_ct[0]
                if bins_t[1] < bins_ct[1]:
                    bins_t[1] = bins_ct[1]
                if max(t[0], curr_t[0]) < min(t[1], curr_t[1]):
                    t = [max(t[0], curr_t[0]), min(t[1], curr_t[1])]
                else:
                    t = [0, 0]
                    print("Warning: mem_time does not overlap!")
                if max(rt[0], curr_rt[0]) < min(rt[1], curr_rt[1]):
                    rt = [max(rt[0], curr_rt[0]), min(rt[1], curr_rt[1])]
                else:
                    rt = [0, 0]
                    print("Warning: r_time does not overlap!")
                if max(wt[0], curr_wt[0]) < min(wt[1], curr_wt[1]):
                    wt = [max(wt[0], curr_wt[0]), min(wt[1], curr_wt[1])]
                else:
                    wt = [0, 0]
                    print("Warning: w_time does not overlap!")
            mem_id += 1
        else:
            t, rt, wt = [start_tmin, end_tmax], [start_rtmin, end_rtmax], [start_wtmin, end_wtmax]
            bins_t = t
            if ips[0] and ips[1]:
                if max(rt[0], wt[0]) < min(rt[1], wt[1]):
                    t = [max(rt[0], wt[0]), min(rt[1], wt[1])]
                else:
                    t = [0, 0]
                    print(f"Warning: {mem} time does not overlap!")
    bins_idx = np.linspace(bins_t[0], bins_t[1], bins + 1)
    bins_wid = (bins_t[1] - bins_t[0]) / bins

    return bins_idx[:-1], bins_wid, t, rt, wt


def get_bandwidth_per_gidx(flits, core_gidx, core_gmems, core_gips, bins, stack):

    core_grbw, core_gwbw, core_gpbw, core_grpbw, core_gwpbw = [], [], [], [], []
    gid, bins_gidx, bins_gwid, core_gt, core_grt, core_gwt = 0, [], [], [], [], []
    for core_idx in core_gidx:
        ip_id, com_switch = 0, True
        core_rbw, core_wbw, core_pbw, core_rpbw, core_wpbw = [], [], [], [], []
        bins_midx, bins_mwid, core_t, core_rt, core_wt = [], [], [], [], []
        for mem in core_gmems[gid]:
            mem_pbw, mem_rpbw, mem_wpbw = 0, 0, 0
            mem_rip, mem_wip = core_gips[gid][ip_id][0], core_gips[gid][ip_id][1]
            mem_rbw = np.zeros((len(mem_rip), bins), dtype="float64") if len(mem_rip) else np.zeros(bins, dtype="float64")
            mem_wbw = np.zeros((len(mem_wip), bins), dtype="float64") if len(mem_wip) else np.zeros(bins, dtype="float64")
            if stack:
                if com_switch:
                    bins_idx, bins_wid, mem_t, mem_rt, mem_wt = get_bin_idx(flits, core_idx, bins, core_gips[gid], core_gmems[gid], stack)
            else:
                ip = core_gips[gid][ip_id]
                bins_idx, bins_wid, mem_t, mem_rt, mem_wt = get_bin_idx(flits, core_idx, bins, ip, mem, stack)
            for core in core_idx:
                for flit in flits.get(core, {}).get(mem, []):
                    flit_bw = int(flit["Size"])
                    flit_t = int(flit["EndCycle"]) - int(flit["StartCycle"])
                    # calc peak bandwidth
                    if int(flit["StartCycle"]) < mem_t[0]:
                        if int(flit["EndCycle"]) > mem_t[0] and int(flit["EndCycle"]) <= mem_t[1]:
                            peak_bw = flit_bw * (int(flit["EndCycle"]) - mem_t[0]) / flit_t
                            mem_pbw += peak_bw
                        elif int(flit["EndCycle"]) > mem_t[1]:
                            peak_bw = flit_bw * (mem_t[1] - mem_t[0]) / flit_t
                            mem_pbw += peak_bw
                    elif int(flit["StartCycle"]) >= mem_t[0] and int(flit["StartCycle"]) < mem_t[1]:
                        if int(flit["EndCycle"]) <= mem_t[1]:
                            mem_pbw += flit_bw
                        elif int(flit["EndCycle"]) > mem_t[1]:
                            peak_bw = flit_bw * (mem_t[1] - int(flit["StartCycle"])) / flit_t
                            mem_pbw += peak_bw
                    l = int((int(flit["StartCycle"]) - bins_idx[0]) / bins_wid)
                    m = int((int(flit["EndCycle"]) - bins_idx[0]) / bins_wid) + 1
                    m = len(bins_idx) if m == len(bins_idx) + 1 else m
                    if flit["Type"] == "R":
                        rip_ind = mem_rip.index(TRAFFIC_SRC[int(flit["TrafSrc"])])
                        if int(flit["EndCycle"]) <= bins_idx[l] + bins_wid:
                            mem_rbw[rip_ind, :][l] += flit_bw
                        else:
                            read_lbw = flit_bw * (bins_idx[l] + bins_wid - int(flit["StartCycle"])) / flit_t
                            read_mbw = flit_bw * bins_wid / flit_t
                            read_rbw = flit_bw * (int(flit["EndCycle"]) - bins_idx[m - 1]) / flit_t
                            mem_rbw[rip_ind, :][l] += read_lbw
                            mem_rbw[rip_ind, :][l + 1 : m - 1] += read_mbw
                            mem_rbw[rip_ind, :][m - 1] += read_rbw
                        # calc read peak bandwidth
                        if stack:
                            if int(flit["StartCycle"]) < mem_rt[0]:
                                if int(flit["EndCycle"]) > mem_rt[0] and int(flit["EndCycle"]) <= mem_rt[1]:
                                    peak_bw = flit_bw * (int(flit["EndCycle"]) - mem_rt[0]) / flit_t
                                    mem_rpbw += peak_bw
                                elif int(flit["EndCycle"]) > mem_rt[1]:
                                    peak_bw = flit_bw * (mem_rt[1] - mem_rt[0]) / flit_t
                                    mem_rpbw += peak_bw
                            elif int(flit["StartCycle"]) >= mem_rt[0] and int(flit["StartCycle"]) < mem_rt[1]:
                                if int(flit["EndCycle"]) <= mem_rt[1]:
                                    mem_rpbw += flit_bw
                                elif int(flit["EndCycle"]) > mem_rt[1]:
                                    peak_bw = flit_bw * (mem_rt[1] - int(flit["StartCycle"])) / flit_t
                                    mem_rpbw += peak_bw
                    elif flit["Type"] == "W":
                        wip_ind = mem_wip.index(TRAFFIC_SRC[int(flit["TrafSrc"])])
                        if int(flit["EndCycle"]) <= bins_idx[l] + bins_wid:
                            mem_wbw[wip_ind, :][l] += flit_bw
                        else:
                            write_lbw = flit_bw * (bins_idx[l] + bins_wid - int(flit["StartCycle"])) / flit_t
                            write_mbw = flit_bw * bins_wid / flit_t
                            write_rbw = flit_bw * (int(flit["EndCycle"]) - bins_idx[m - 1]) / flit_t
                            mem_wbw[wip_ind, :][l] += write_lbw
                            mem_wbw[wip_ind, :][l + 1 : m - 1] += write_mbw
                            mem_wbw[wip_ind, :][m - 1] += write_rbw
                        # calc write peak bandwidth
                        if stack:
                            if int(flit["StartCycle"]) < mem_wt[0]:
                                if int(flit["EndCycle"]) > mem_wt[0] and int(flit["EndCycle"]) <= mem_wt[1]:
                                    peak_bw = flit_bw * (int(flit["EndCycle"]) - mem_wt[0]) / flit_t
                                    mem_wpbw += peak_bw
                                elif int(flit["EndCycle"]) > mem_wt[1]:
                                    peak_bw = flit_bw * (mem_wt[1] - mem_wt[0]) / flit_t
                                    mem_wpbw += peak_bw
                            elif int(flit["StartCycle"]) >= mem_wt[0] and int(flit["StartCycle"]) < mem_wt[1]:
                                if int(flit["EndCycle"]) <= mem_wt[1]:
                                    mem_wpbw += flit_bw
                                elif int(flit["EndCycle"]) > mem_wt[1]:
                                    peak_bw = flit_bw * (mem_wt[1] - int(flit["StartCycle"])) / flit_t
                                    mem_wpbw += peak_bw
            core_rbw.append(mem_rbw)
            core_wbw.append(mem_wbw)
            core_pbw.append(mem_pbw)
            core_rpbw.append(mem_rpbw)
            core_wpbw.append(mem_wpbw)
            if com_switch:
                core_t.append(mem_t[1] - mem_t[0])
                core_rt.append(mem_rt[1] - mem_rt[0])
                core_wt.append(mem_wt[1] - mem_wt[0])
                bins_midx.append(bins_idx)
                bins_mwid.append(bins_wid)
            com_switch = False if stack else True
            ip_id += 1
        gid += 1
        core_grbw.append(core_rbw)
        core_gwbw.append(core_wbw)
        core_gpbw.append(core_pbw)
        core_grpbw.append(core_rpbw)
        core_gwpbw.append(core_wpbw)
        core_gt.append(core_t)
        core_grt.append(core_rt)
        core_gwt.append(core_wt)
        bins_gidx.append(bins_midx)
        bins_gwid.append(bins_mwid)
    bw_per_gidx = {
        "rbw": core_grbw,
        "wbw": core_gwbw,
        "pbw": core_gpbw,
        "rpbw": core_grpbw,
        "wpbw": core_gwpbw,
        "idx": bins_gidx,
        "wid": bins_gwid,
        "t": core_gt,
        "rt": core_grt,
        "wt": core_gwt,
    }

    return bw_per_gidx


def plot_bandwidth_per_gidx(core_gidx, core_gmems, core_gips, bw_per_gidx, stack, timescale, subplots, path):

    lw = 1.8
    sc = 1e6  # s -> us
    yscale = 1.1
    bwscale = 1e9
    default_color = "brown"
    mems = {"ddr": "red", "l2m": "green", "slc": "blue"}
    ips = {"G": "cyan", "S": "magenta", "C": "yellow"}
    group_num = len(core_gidx)
    fig_num = int((group_num + subplots - 1) / subplots)
    if len(core_gidx) == 1 and len(core_gidx[0]) in (8, 16, 18):
        plot_n, fig_n = ["CORE(U)"], ["CORE(U)"]
    else:
        plot_n, fig_n = [], []
        for gidx in core_gidx:
            plot_n.append("".join(["CORE("] + [",".join([elem.strip("CORE") for elem in gidx])] + [")"]))
        for i in range(fig_num):
            curr_num = subplots if (i + 1) * subplots <= group_num else group_num - i * subplots
            fig_n.append("".join(["CORE("] + [",".join([elem.strip("CORE()") for elem in plot_n[i * subplots : i * subplots + curr_num]])] + [")"]))
    for i in range(fig_num):
        plt.figure()
        curr_num = subplots if (i + 1) * subplots <= group_num else group_num - i * subplots
        mem_num = [len(core_gmems[i * subplots + k]) for k in range(curr_num)]
        for k in range(curr_num):
            if stack:
                wid_val = bw_per_gidx["wid"][i * subplots + k][0] / timescale * sc
                bw_xaxis = bw_per_gidx["idx"][i * subplots + k][0] / timescale * sc + (wid_val / 2)
                pbw_scale = sum(bw_per_gidx["pbw"][i * subplots + k]) / bwscale
                bw_time = bw_per_gidx["t"][i * subplots + k][0] / timescale
                rpbw_scale = sum(bw_per_gidx["rpbw"][i * subplots + k]) / bwscale
                rbw_time = bw_per_gidx["rt"][i * subplots + k][0] / timescale
                wpbw_scale = sum(bw_per_gidx["wpbw"][i * subplots + k]) / bwscale
                wbw_time = bw_per_gidx["wt"][i * subplots + k][0] / timescale

                bw_bott, rbw_bott, wbw_bott = 0, 0, 0
                for mem_id in range(len(core_gmems[i * subplots + k])):
                    rbw_yaxis = np.array(bw_per_gidx["rbw"][i * subplots + k][mem_id]) / bwscale / wid_val * sc
                    wbw_yaxis = np.array(bw_per_gidx["wbw"][i * subplots + k][mem_id]) / bwscale / wid_val * sc
                    mem_key = core_gmems[i * subplots + k][mem_id]
                    ip_key = core_gips[i * subplots + k][mem_id]
                    mc = mems.get(mem_key, default_color)
                    if ip_key[0]:
                        for rip_id in range(len(ip_key[0])):
                            rc = ips[ip_key[0][rip_id]]
                            plt.subplot(curr_num, 3, 3 * k + 1)
                            plt.bar(bw_xaxis, rbw_yaxis[rip_id, :], bottom=bw_bott, width=wid_val, linewidth=lw, edgecolor=mc, color=rc, alpha=1)
                            bw_bott += rbw_yaxis[rip_id, :]
                            plt.subplot(curr_num, 3, 3 * k + 2)
                            plt.bar(bw_xaxis, rbw_yaxis[rip_id, :], bottom=rbw_bott, width=wid_val, linewidth=lw, edgecolor=mc, color=rc, alpha=1)
                            rbw_bott += rbw_yaxis[rip_id, :]
                    else:
                        rc = default_color
                        plt.subplot(curr_num, 3, 3 * k + 2)
                        plt.bar(bw_xaxis, rbw_yaxis, bottom=rbw_bott, width=wid_val, linewidth=lw, edgecolor=mc, color=rc, alpha=1)
                    if ip_key[1]:
                        for wip_id in range(len(ip_key[1])):
                            wc = ips[ip_key[1][wip_id]]
                            plt.subplot(curr_num, 3, 3 * k + 1)
                            plt.bar(bw_xaxis, wbw_yaxis[wip_id, :], bottom=bw_bott, width=wid_val, linewidth=lw, edgecolor=mc, color=wc, alpha=1)
                            bw_bott += wbw_yaxis[wip_id, :]
                            plt.subplot(curr_num, 3, 3 * k + 3)
                            plt.bar(bw_xaxis, wbw_yaxis[wip_id, :], bottom=wbw_bott, width=wid_val, linewidth=lw, edgecolor=mc, color=wc, alpha=1)
                            wbw_bott += wbw_yaxis[wip_id, :]
                    else:
                        wc = default_color
                        plt.subplot(curr_num, 3, 3 * k + 3)
                        plt.bar(bw_xaxis, wbw_yaxis, bottom=wbw_bott, width=wid_val, linewidth=lw, edgecolor=mc, color=wc, alpha=1)

                bw_max = max(bw_bott) if isinstance(bw_bott, np.ndarray) else 0
                rbw_max = max(rbw_bott) if isinstance(rbw_bott, np.ndarray) else 0
                wbw_max = max(wbw_bott) if isinstance(wbw_bott, np.ndarray) else 0
                pbw_val = pbw_scale / bw_time if bw_time else bw_max
                rpbw_val = rpbw_scale / rbw_time if rbw_time else rbw_max
                wpbw_val = wpbw_scale / wbw_time if wbw_time else wbw_max
                bw_tle = f"{plot_n[i * subplots + k]}:{round(pbw_val, 2)}"
                rbw_tle = f"read:{round(rpbw_val, 2)}"
                wbw_tle = f"write:{round(wpbw_val, 2)}"
                plt.subplot(curr_num, 3, 3 * k + 1)
                plt.title(bw_tle)
                plt.ylabel("BW(GB/s)")
                plt.ylim(0, bw_max * yscale)
                plt.yticks([0, bw_max], [0, f"{bw_max:.2f}"])
                plt.axhline(bw_max, color=default_color, linestyle="--")
                plt.subplot(curr_num, 3, 3 * k + 2)
                plt.title(rbw_tle)
                plt.xlabel("Time(us)")
                plt.ylim(0, rbw_max * yscale)
                plt.yticks([0, rbw_max], [0, f"{rbw_max:.2f}"])
                plt.axhline(rbw_max, color=default_color, linestyle="--")
                plt.subplot(curr_num, 3, 3 * k + 3)
                plt.title(wbw_tle)
                plt.ylim(0, wbw_max * yscale)
                plt.yticks([0, wbw_max], [0, f"{wbw_max:.2f}"])
                plt.axhline(wbw_max, color=default_color, linestyle="--")
            else:
                curr_mnum = len(core_gmems[i * subplots + k])
                for mem_id in range(curr_mnum):
                    bw_bott, rbw_bott, wbw_bott = 0, 0, 0
                    wid_val = bw_per_gidx["wid"][i * subplots + k][mem_id] / timescale * sc
                    bw_xaxis = bw_per_gidx["idx"][i * subplots + k][mem_id] / timescale * sc + (wid_val / 2)
                    pbw_scale = bw_per_gidx["pbw"][i * subplots + k][mem_id] / bwscale
                    bw_time = bw_per_gidx["t"][i * subplots + k][mem_id] / timescale
                    rbw_scale = np.array(bw_per_gidx["rbw"][i * subplots + k][mem_id]) / bwscale
                    rbw_time = bw_per_gidx["rt"][i * subplots + k][mem_id] / timescale
                    wbw_scale = np.array(bw_per_gidx["wbw"][i * subplots + k][mem_id]) / bwscale
                    wbw_time = bw_per_gidx["wt"][i * subplots + k][mem_id] / timescale
                    rbw_yaxis = rbw_scale / wid_val * sc
                    wbw_yaxis = wbw_scale / wid_val * sc
                    mem_key = core_gmems[i * subplots + k][mem_id]
                    ip_key = core_gips[i * subplots + k][mem_id]
                    mc = mems.get(mem_key, default_color)

                    if ip_key[0]:
                        for rip_id in range(len(ip_key[0])):
                            rc = ips[ip_key[0][rip_id]]
                            plt.subplot(sum(mem_num), 3, (sum(mem_num[:k]) + mem_id) * 3 + 1)
                            plt.bar(bw_xaxis, rbw_yaxis[rip_id, :], bottom=bw_bott, width=wid_val, linewidth=lw, edgecolor=mc, color=rc, alpha=1)
                            bw_bott += rbw_yaxis[rip_id, :]
                            plt.subplot(sum(mem_num), 3, (sum(mem_num[:k]) + mem_id) * 3 + 2)
                            plt.bar(bw_xaxis, rbw_yaxis[rip_id, :], bottom=rbw_bott, width=wid_val, linewidth=lw, edgecolor=mc, color=rc, alpha=1)
                            rbw_bott += rbw_yaxis[rip_id, :]
                    else:
                        rc = default_color
                        plt.subplot(sum(mem_num), 3, (sum(mem_num[:k]) + mem_id) * 3 + 2)
                        plt.bar(bw_xaxis, rbw_yaxis, width=wid_val, linewidth=lw, edgecolor=mc, color=rc, alpha=1)
                    if ip_key[1]:
                        for wip_id in range(len(ip_key[1])):
                            wc = ips[ip_key[1][wip_id]]
                            plt.subplot(sum(mem_num), 3, (sum(mem_num[:k]) + mem_id) * 3 + 1)
                            plt.bar(bw_xaxis, wbw_yaxis[wip_id, :], bottom=bw_bott, width=wid_val, linewidth=lw, edgecolor=mc, color=wc, alpha=1)
                            bw_bott += wbw_yaxis[wip_id, :]
                            plt.subplot(sum(mem_num), 3, (sum(mem_num[:k]) + mem_id) * 3 + 3)
                            plt.bar(bw_xaxis, wbw_yaxis[wip_id, :], bottom=wbw_bott, width=wid_val, linewidth=lw, edgecolor=mc, color=wc, alpha=1)
                            wbw_bott += wbw_yaxis[wip_id, :]
                    else:
                        wc = default_color
                        plt.subplot(sum(mem_num), 3, (sum(mem_num[:k]) + mem_id) * 3 + 3)
                        plt.bar(bw_xaxis, wbw_yaxis, width=wid_val, linewidth=lw, edgecolor=mc, color=wc, alpha=1)

                    bw_max = max(bw_bott) if isinstance(bw_bott, np.ndarray) else 0
                    rbw_max = max(rbw_bott) if isinstance(rbw_bott, np.ndarray) else 0
                    wbw_max = max(wbw_bott) if isinstance(wbw_bott, np.ndarray) else 0
                    rbw_val = np.sum(rbw_scale) / rbw_time if rbw_time else 0
                    wbw_val = np.sum(wbw_scale) / wbw_time if wbw_time else 0
                    pbw_val = pbw_scale / bw_time if bw_time else max(rbw_val, wbw_val)
                    bw_tle = f"{plot_n[i * subplots + k]}_{mem_key.upper()}_:{round(pbw_val, 2)}"
                    rbw_tle = f"read:{round(rbw_val, 2)}"
                    wbw_tle = f"write:{round(wbw_val, 2)}"
                    plt.subplot(sum(mem_num), 3, (sum(mem_num[:k]) + mem_id) * 3 + 1)
                    plt.title(bw_tle)
                    plt.ylabel("BW(GB/s)")
                    plt.ylim(0, bw_max * yscale)
                    plt.yticks([0, bw_max], [0, f"{bw_max:.2f}"])
                    plt.axhline(bw_max, color=default_color, linestyle="--")
                    plt.subplot(sum(mem_num), 3, (sum(mem_num[:k]) + mem_id) * 3 + 2)
                    plt.title(rbw_tle)
                    plt.xlabel("Time(us)")
                    plt.ylim(0, rbw_max * yscale)
                    plt.yticks([0, rbw_max], [0, f"{rbw_max:.2f}"])
                    plt.axhline(rbw_max, color=default_color, linestyle="--")
                    plt.subplot(sum(mem_num), 3, (sum(mem_num[:k]) + mem_id) * 3 + 3)
                    plt.title(wbw_tle)
                    plt.ylim(0, wbw_max * yscale)
                    plt.yticks([0, wbw_max], [0, f"{wbw_max:.2f}"])
                    plt.axhline(wbw_max, color=default_color, linestyle="--")

        plt.tight_layout()
        if stack:
            plt.savefig(f"{path}/{fig_n[i]}_stack.png")
        else:
            plt.savefig(f"{path}/{fig_n[i]}.png")
        plt.show(block=False)
        plt.pause(1)


def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gem5_dir = os.path.dirname(script_dir)
    txt_path = os.path.join(gem5_dir, "data/Flit_Info.txt")
    results_path = os.path.join(gem5_dir, "data/results")
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", "-t", default=txt_path, type=str, help="txt file path")
    parser.add_argument("--dir", "-d", default=results_path, type=str, help="result path")
    parser.add_argument("--core_group", "-cg", type=str, help="group id for cores, e.g. [[1,2,3],[4,5,6]]")
    parser.add_argument("--bins", "-bn", type=int, default=20, help="num of bandwidth bins")
    parser.add_argument("--stack", "-st", action="store_true", help="Stack the bandwidth for Benchmark")
    parser.add_argument("--all", action="store_true", help="all cores as one group")
    parser.add_argument("--timescale", "-ts", type=int, default=2e9, help="scale for time unit conversion(s)")
    parser.add_argument("--subplots", "-sp", type=int, default=4, help="num of subplots")
    parser.add_argument("--num_cpus", "-n", type=int, default=16, help="num of cores")
    args = parser.parse_args()

    try:
        core_group = json.loads(args.core_group) if args.core_group else None
        if core_group and any(min(core) < 0 or max(core) > (args.num_cpus - 1) for core in core_group):
            sys.exit(1)
    except:
        print("Error: core_group is invalid!")
        sys.exit(1)
    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    flits = get_flits_data(args.txt)
    core_gidx, core_gmems, core_gips = check_group(flits, core_group, args.all)
    bw_per_gidx = get_bandwidth_per_gidx(flits, core_gidx, core_gmems, core_gips, args.bins, args.stack)
    plot_bandwidth_per_gidx(core_gidx, core_gmems, core_gips, bw_per_gidx, args.stack, args.timescale, args.subplots, args.dir)
    plt.show()


if __name__ == "__main__":
    main()
