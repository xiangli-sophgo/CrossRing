#######################################################################
# Description: extract buffer reads and writes from stats.txt and     #
#              generate a csv file. Parse the csv file and plot a     #
#              hotpoint map.                                          #
#                                                                     #
# Attention: visualize function need matplotlib environment, should   #
#            be used in local machine.                                #
#######################################################################

# import sys
import re
import numpy as np
import matplotlib.pyplot as plt

STATS_TXT = ""    # Use your own path
STATS_NAME = "resnet_1core"  # Use your own name

CSV_FILE = "resnet_1core"


class Statistics:
    """
    Extract buffer reads and writes from stats.txt and generate a csv file.

    :param file: name of the stats.txt file without extension, e.g. "resnet_1core".
    """
    def __init__(self, file):
        self.file = file
        self.buffer_read = {"routers00": 0, "routers01": 0, "routers02": 0, "routers03": 0, "routers04": 0,
                            "routers05": 0, "routers06": 0, "routers07": 0, "routers08": 0, "routers09": 0,
                            "routers10": 0, "routers11": 0, "routers12": 0, "routers13": 0, "routers14": 0,
                            "routers15": 0, "routers16": 0, "routers17": 0, "routers18": 0, "routers19": 0,
                            "routers20": 0, "routers21": 0, "routers22": 0, "routers23": 0, "routers24": 0,
                            "routers25": 0, "routers26": 0, "routers27": 0, "routers28": 0, "routers29": 0}
        self.buffer_write = {"routers00": 0, "routers01": 0, "routers02": 0, "routers03": 0, "routers04": 0,
                             "routers05": 0, "routers06": 0, "routers07": 0, "routers08": 0, "routers09": 0,
                             "routers10": 0, "routers11": 0, "routers12": 0, "routers13": 0, "routers14": 0,
                             "routers15": 0, "routers16": 0, "routers17": 0, "routers18": 0, "routers19": 0,
                             "routers20": 0, "routers21": 0, "routers22": 0, "routers23": 0, "routers24": 0,
                             "routers25": 0, "routers26": 0, "routers27": 0, "routers28": 0, "routers29": 0}
        self.buffer_read_list = []
        self.buffer_write_list = []

    def extract_stats(self):
        with open(STATS_TXT + self.file + ".txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "buffer_reads" in line:
                    stats = re.split(r"\W+", line)
                    routerid = stats[3]
                    buffer_reads = int(stats[5])
                    if routerid in self.buffer_read:
                        self.buffer_read[routerid] = buffer_reads
                if "buffer_writes" in line:
                    stats = re.split(r"\W+", line)
                    routerid = stats[3]
                    buffer_writes = int(stats[5])
                    if routerid in self.buffer_write:
                        self.buffer_write[routerid] = buffer_writes

    def get_buffer_read(self):
        return self.buffer_read

    def get_buffer_write(self):
        return self.buffer_write

    def generate_csv(self):
        with open(STATS_TXT + self.file + ".csv", 'w') as f:
            f.write("buffer_reads,buffer_writes,avg_buffer_traffic\n")
            for (r, w) in zip(self.buffer_read.values(), self.buffer_write.values()):
                avg_buffer_traffic = int((r + w) / 2)
                f.write(str(r) + ',' + str(w) + ',' + str(avg_buffer_traffic) + "\n")

class Hotpoints:
    """
    Extract buffer reads and writes from csv file and plot the hotpoint map.
    Need matplotlib environment. Use visualize function in local machine.

    :param file: name of the csv file without extension, e.g. "resnet_1core".
    """
    def __init__(self, file):
        self.file = file
        self.buffer_read = []
        self.buffer_write = []
        self.avg_buf_traffic = []

    def extract_csv(self):
        with open(self.file + ".csv", 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
            for line in lines:
                stats = re.split(r',', line)
                self.buffer_read.append(int(stats[0]))
                self.buffer_write.append(int(stats[1]))
                self.avg_buf_traffic.append(int(stats[2]))

    def get_buffer_read(self):
        return self.buffer_read

    def get_buffer_write(self):
        return self.buffer_write

    def get_avg_buf_traffic(self):
        return self.avg_buf_traffic

    def get_traffic_mesh(self):
        mesh = np.array(self.avg_buf_traffic)
        mesh = mesh.reshape(6, 5)
        mesh_rev = np.empty((6, 5), dtype=int)
        for i in range(6):
            mesh_rev[-(i+1),...] = mesh[i,...]
        return mesh, mesh_rev

    def visualize(self):
        mesh = self.get_traffic_mesh()[0]
        figure = plt.figure()
        axes = figure.add_subplot()
        caxes = axes.matshow(mesh, cmap=plt.cm.YlOrRd, origin='lower')
        xticks = axes.get_xticks()
        yticks = axes.get_yticks()
        xticks += 1
        yticks += 1
        axes.set_xticklabels(xticks)
        axes.set_yticklabels(yticks)
        figure.colorbar(caxes)
        plt.show()
        figure.savefig(self.file+'.png')



if __name__ == "__main__":
    stat = Statistics("resnet_1core")
    stat.extract_stats()
    stat.generate_csv()

    hotpoint1 = Hotpoints(CSV_FILE)
    hotpoint1.extract_csv()

    # This function should be used in local machine.
    hotpoint1.visualize()
