from link import Link
from rnsn import RN, SN
import matplotlib.pyplot as plt
import networkx as nx


class NoC:
    def __init__(
        self,
        config: dict = None,  # 表示配置文件
    ):
        if config is not None:
            self.config = config
        else:
            raise ValueError("Config file error!")
        self.parameter_compute()
        self.initialization()

    def parameter_compute(self):
        self.x_length = self.config["x_length"]
        self.y_length = self.config["y_length"]
        self.x_range = range(self.x_length)
        self.y_range = range(self.y_length)
        # 每个节点都挂了ddr
        ddr = [(x, y) for x in self.x_range for y in self.y_range]
        gdma = [(x, y) for x in self.x_range for y in self.y_range]
        cdma = [(x, y) for x in self.x_range for y in self.y_range]

        self.ip = {"ddr": ddr, "cdma": cdma, "gdma": gdma}
        # LBN为链路上的slice数
        LBN = self.config["LBN"]
        self.slice_x = [LBN for _ in self.x_range]
        self.slice_y = [LBN for _ in self.y_range]

        self.enter_mesh_slice = self.config["enter_mesh_slice"]
        self.leave_mesh_slice = self.config["leave_mesh_slice"]

    def id2coord(self, s) -> tuple[int, int]:
        """输入一个0到x_length * y_length - 1 的整数，输出一个二维坐标(x, y)"""
        if int(s) >= self.x_length * self.y_length:
            raise ValueError(f"Invalid id: {s} {self.x_length, self.y_length}")
        y_coord, x_coord = divmod(int(s), self.x_length)
        return x_coord, self.y_length - 1 - y_coord

    def coord2id(self, coord: tuple[int, int]) -> int:
        """输入一个二维坐标(x, y)，输出一个0到x_length * y_length - 1 的整数"""
        if coord[0] >= self.x_length or coord[1] >= self.y_length or coord[0] < 0 or coord[1] < 0:
            raise ValueError(f"Invalid coordinate: ({coord[0]}, {coord[1]})")
        s: int = coord[0] + (self.y_length - 1 - coord[1]) * self.x_length
        return s

    def initialization(self):
        """初始化NoC"""
        self.req_link = self.initialization_link()
        self.rsp_link = self.initialization_link()
        self.dat_link = self.initialization_link()

        self.pos = None

        self.rn = {}
        self.sn = {}
        for ip_name in self.ip:
            if ip_name.endswith("dma"):
                for coord in self.ip[ip_name]:
                    self.rn[coord, ip_name] = RN(coord=coord, name=ip_name, config=self.config)
                    # self.rn[coord, ip_name].initialization()
            else:
                assert ip_name == "ddr"
                for coord in self.ip[ip_name]:
                    self.sn[coord, ip_name] = SN(coord=coord, name=ip_name, config=self.config)
                    # self.sn[coord, ip_name].initialization()

    def state_update(self):
        """三种网络的所有链路状态更新,是否可输入和是否可输出"""
        self._update_link_states(self.req_link)
        self._update_link_states(self.rsp_link)
        self._update_link_states(self.dat_link)

    def network_inside_walk(self):
        """三种网络的所有链路内部的数据包往前走"""
        self._update_link_walks(self.req_link)
        self._update_link_walks(self.rsp_link)
        self._update_link_walks(self.dat_link)

    def _update_link_states(self, link_dict: dict):
        """更新链路状态，检查是否可输入和可输出"""
        for key in link_dict:
            link = link_dict[key]
            link.state_update()  # 调用链路的状态更新方法

    def _update_link_walks(self, link_dict: dict):
        """使链路内部的数据包往前走"""
        for key in link_dict:
            link = link_dict[key]
            link.inside_walk()  # 调用链路的内部走动方法

    def network_link2link(self):
        for key in self.req_link:
            self.req_link[key].link2link()
            self.rsp_link[key].link2link()
            self.dat_link[key].link2link()

    def rn2noc(self):
        for key in self.rn:
            self.rn[key].rn2noc(self.req_link[key[1], key[0]], self.dat_link[key[1], key[0]])

    def noc2rn(self, cycle):
        for key in self.rn:
            self.rn[key].noc2rn(self.rsp_link[key], self.dat_link[key], cycle)

    def rn2dma(self, read_req, cycle):
        i = 0
        for key in self.rn:
            if self.rn[key].rn2dma(read_req, cycle):
                i += 1
        # print(i)
        return i

    def sn2noc(self, cycle):
        for key in self.sn:
            self.sn[key].sn2noc(
                self.rsp_link[key[1], key[0]],
                self.dat_link[key[1], key[0]],
                cycle,
            )

    def noc2sn(self, cycle):
        for key in self.sn:
            self.sn[key].noc2sn(self.req_link[key], self.dat_link[key], cycle)

    def sn2ddr(self, write_req, cycle):
        i = 0
        for key in self.sn:
            if self.sn[key].sn2ddr(write_req, cycle):
                i += 1
        # print(i)
        return i

    def initialization_link(self):
        # 初始化REQ,RSP,DATA链路,使用全局参数
        link: dict[tuple:Link] = {}
        for x in self.x_range:
            for y in self.y_range:
                src = (x, y)
                # 初始化网络链路
                if x + 1 in self.x_range:
                    dest = (x + 1, y)
                    # 初始化往右的链路
                    self._create_link(link, src, dest, self.slice_x[x])
                    # 初始化往左的链路
                    self._create_link(link, dest, src, self.slice_x[x - 1])
                if y + 1 in self.y_range:
                    dest = (x, y + 1)
                    # 初始化往上的链路
                    self._create_link(link, src, dest, self.slice_y[y])
                    # 初始化往下的链路
                    self._create_link(link, dest, src, self.slice_y[y - 1])

        # 初始化上下网络链路,并和Router关联
        for ip_name in self.ip:
            for ip_coord in self.ip[ip_name]:
                # 初始化上网络链路
                # dest = ip_name + "_" + str(self.coord2id(ip_coord))
                self._create_link(link, ip_name, ip_coord, self.enter_mesh_slice)
                # 初始化下网络链路
                self._create_link(link, ip_coord, ip_name, self.leave_mesh_slice)

        # 给每条链路分配round_robin
        for output_link in link:
            sublink = link[output_link]
            src = sublink.src
            sublink.round_robin = []
            if type(src) is str:
                continue

            dest = output_link[1]
            mesh_dir = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            # 添加mesh的link到round robin
            for mesh_dir in mesh_dir:
                input_link_src = src[0] + mesh_dir[0], src[1] + mesh_dir[1]
                if dest == input_link_src:
                    continue
                if (input_link_src, src) not in link:
                    continue
                link[output_link].round_robin.append(link[(input_link_src, src)])
            # 添加上网络的link到round robin
            for ip_dir in self.ip:
                assert (ip_dir, src) in link
                if ip_dir == output_link[1]:
                    continue
                link[output_link].round_robin.append(link[(ip_dir, src)])

        return link

    def _create_link(self, link: dict, src: tuple, dest: tuple, slice_num: int):
        """创建链路并初始化缓冲区"""
        link[(src, dest)] = Link(src=src, dest=dest, slice_num=slice_num, buffer_depth=self.config["fifo_depth"])

    def display_all_links_table(self, link: dict[tuple[int, int], Link]):
        """
        以表格的形式展示 request_link 中所有链路的状态。

        参数:
            request_link: dict[tuple[int, int], Link]
                一个字典，键是 (start, dest) 的元组，值是 Link 对象。
        """
        # print("Displaying all links as a table:")
        # print("-" * 60)
        # print(f"{'Link (Start -> Dest)':<20} | {'Buffer State':<35}")
        # print("-" * 60)

        for (start, dest), link in link.items():
            # 构建每个 Link 的 buffer 状态
            buffer_state = " | ".join(f"[{', '.join(str(packet) for packet in slice_buffer)}]" for slice_buffer in link.buffer)
            print(f"{str(start) + ' -> ' + str(dest):<20} | {buffer_state:<35}")
        print("-" * 60)

    # def plot_links_with_buffers(self, link_dict: dict[tuple[int, int], Link]):
    #     """
    #     使用 matplotlib 和 networkx 绘制链路及其 buffer 状态。
    #     """
    #     G = nx.DiGraph()

    #     # 添加节点和边
    #     for (start, dest), link in link_dict.items():
    #         if dest in ["ddr", "cdma", "gdma"]:
    #             dest = dest + "_" + str(self.coord2id(start))
    #         if start in ["ddr", "cdma", "gdma"]:
    #             start = start + "_" + str(self.coord2id(dest))
    #         G.add_edge(start, dest, buffer=link.buffer)

    #     # 绘制图形
    #     if self.pos is None:
    #         self.pos = nx.spring_layout(G)  # 自动布局

    #     plt.clf()

    #     edge_labels = {}
    #     # 在边上显示 buffer 状态
    #     for (start, dest), link in link_dict.items():
    #         if dest in ["ddr", "cdma", "gdma"]:
    #             dest = dest + "_" + str(self.coord2id(start))
    #         if start in ["ddr", "cdma", "gdma"]:
    #             start = start + "_" + str(self.coord2id(dest))
    #         edge_labels[(start, dest)] = f"{[list(slice_buffer) for slice_buffer in link.buffer]}"

    #     nx.draw(G, self.pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10)
    #     nx.draw_networkx_edge_labels(G, self.pos, edge_labels=edge_labels, font_size=8)

    #     # plt.show()
    #     plt.draw()  # 更新图形
    #     plt.pause(2)  # 暂停以允许图形更新
