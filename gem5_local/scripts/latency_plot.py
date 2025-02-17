#######################################################################
# Description: Extract latency statistics from csv file and plot bar  #
#              chart to compare among a series of traffic.            #
#                                                                     #
# Attention: need matplotlib environment. Should be used in local.    #
#                                                                     #
# Usage: python3.8 latency_plot.py stats.csv                          #
#######################################################################

import sys
import re
import numpy as np
import matplotlib.pyplot as plt

STATS_CSV = sys.argv[1]

router_latency = 2
link_latency = 6

avg_pkt_latency = []
avg_pkt_net_latency = []
avg_pkt_q_latency = []
# avg_pkt_vq_latency = []

avg_flit_latency = []
avg_flit_net_latency = []
avg_flit_q_latency = []
# avg_flit_vq_latency = []

avg_hops = []


if __name__ == "__main__":
    with open(STATS_CSV, 'r') as f:
        lines = f.readlines()
        table_head = re.split(r',|\n', lines[0])
        labels = table_head[1:-1]   # discard the last empty character
        for line in lines:
            if "average_packet_latency" in line:
                stats = re.split(r',|\n', line)
                for stat in stats[1:-1]:
                    avg_pkt_latency.append(float(stat))
            if "average_packet_network_latency" in line:
                stats = re.split(r',|\n', line)
                for stat in stats[1:-1]:
                    avg_pkt_net_latency.append(float(stat))
            if "average_packet_queueing_latency" in line:
                stats = re.split(r',|\n', line)
                for stat in stats[1:-1]:
                    avg_pkt_q_latency.append(float(stat))
            if "average_flit_latency" in line:
                stats = re.split(r',|\n', line)
                for stat in stats[1:-1]:
                    avg_flit_latency.append(float(stat))
            if "average_flit_network_latency" in line:
                stats = re.split(r',|\n', line)
                for stat in stats[1:-1]:
                    avg_flit_net_latency.append(float(stat))
            if "average_flit_queueing_latency" in line:
                stats = re.split(r',|\n', line)
                for stat in stats[1:-1]:
                    avg_flit_q_latency.append(float(stat))
            if "average_hops" in line:
                stats = re.split(r',|\n', line)
                for stat in stats[1:-1]:
                    avg_hops.append(float(stat))

    print(labels)

    theory_avg_pkt_net_lat = []
    for hop in avg_hops:
        theory_lat = hop * (router_latency + link_latency)
        theory_avg_pkt_net_lat.append(theory_lat)

    # Plot
    fig, ax1 = plt.subplots(figsize=(55,12))

    x = np.arange(len(labels))
    width = 0.3

    ax1.bar([i-width/2 for i in x], avg_pkt_net_latency, color='blueviolet', width=width, label="average_packet_network_latency")
    # ax1.bar([i-width/2 for i in x], theory_avg_pkt_net_lat, color='blueviolet', width=width, label="theory_avg_packet_network_latency")
    # plt.bar(x, avg_pkt_net_latency, width=width, label="average_packet_network_latency")
    # plt.bar([i+width for i in x], avg_pkt_q_latency, width=width, label="average_packet_queueing_latency")
    ax1.set_ylim(0, 50)
    for a, b in zip(x-width/2, avg_pkt_net_latency):
        ax1.text(a, b, '%.1f'%b, ha='center', va='bottom', fontsize=10)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_xlabel("traffics", fontsize=24)
    ax1.set_ylabel("Cycles", fontsize=24)
    ax1.grid(axis='y', color='gray', linestyle='--')
    plt.legend(loc='upper left', fontsize=20)

    ax2 = ax1.twinx()
    ax2.bar([i+width/2 for i in x], avg_hops, color='pink', width=width, label="average_hops")
    ax2.set_ylim(0, 5.0)
    for a, b in zip(x+width/2, avg_hops):
        ax2.text(a, b, '%.1f'%b, ha='center', va='bottom', fontsize=10)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.set_ylabel("Hops", fontsize=24)

    plt.title("Average Packet Latency & Average hops", fontsize=30)
    plt.xticks(x, labels)
    fig.autofmt_xdate(rotation=30)
    plt.legend(loc='upper right', fontsize=18)
    plt.savefig('avg_pkt_latency20231007.png')
    plt.show()

    # plt.savefig('avg_pkt_latency20231003.eps')
