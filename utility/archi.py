import json
import numpy as np
import random
import os
from parameters import *


class ArchiGenerater:
    """
    turn architecture to the matric format
    """
    def __init__(self):
        # Basic node information
        self.node_num = None
        self.node_mat = None
        self.node_info = {}
        self.node_links = None
        # scope of the node buffer capacity
        self.rand_min = None
        self.rand_max = None

        # Basic flow information
        self.tt_num = None
        self.tt_flow = []
        self.tt_flow_cycle_option = args.tt_flow_cycles
        # scope of the flow characteristic
        self.delay_min = None
        self.delay_max = None
        self.pkt_min = None
        self.pkt_max = None

        # tsn topo and streams generated
        self.topo_set = None
        self.stream_set = None

    def node_mat_gene(self, node_links=((0, 1), (1, 2), (1, 3), (1, 5), (3, 4), (5, 4))):
        """
        generate adjecent matric of nodes
        """
        inf = float('inf')
        # Judge the number of nodes
        judge_node = set()
        judge = [set(link) for link in node_links]
        for i in judge:
            judge_node |= i
        if self.node_num != len(judge_node):
            self.node_num = len(judge_node)
            # print('node num changed to {}, existing node: {}'.format(len(judge_node), judge_node))
        self.node_links = node_links
        self.node_mat = np.full((self.node_num, self.node_num), inf)
        for link in self.node_links:
            self.node_mat[link[0] - 1, link[1] - 1] = 1
            self.node_mat[link[1] - 1, link[0] - 1] = 1
        return self.node_mat

    def node_info_gene(self, rand_min=30, rand_max=100):
        """
        rand_min: minimum buffer capacity
        rand_max: maximum buffer capacity
        generate result {node_idx : buff_size }
        """
        self.rand_min = rand_min
        self.rand_max = rand_max
        self.node_info = {}
        for i in range(self.node_num):
            self.node_info[i] = random.randint(rand_min, rand_max)

        return self.node_info

    def tt_flow_gene(self, tt_num=30, delay_min=2048, delay_max=4096, pkt_min=72, pkt_max=1526):
        """
        tt_num: the number of TT flow generated
        delay unit: ms
        pkt_len nit： byte
        return: tt_flow: List[Tuple[source node number, end node number, cycle, maximum delay, packet length]]
        """
        self.tt_num = tt_num
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.pkt_min = pkt_min
        self.pkt_max = pkt_max
        self.tt_flow = []
        for i in range(tt_num):
            source_node = random.randint(0, self.node_num - 1)
            end_node = random.randint(0, self.node_num - 1)
            while end_node == source_node:
                end_node = random.randint(0, self.node_num - 1)
            cycle = self.tt_flow_cycle_option[random.randint(0, len(self.tt_flow_cycle_option) - 1)]
            delay = random.randint(delay_min, delay_max)
            pkt_len = random.randint(pkt_min, pkt_max)
            self.tt_flow.append((source_node, end_node, cycle, delay, pkt_len))
        return self.tt_flow

    def gene_all(self, rand_min=30, rand_max=100,
                 tt_num=30, delay_min=2048, delay_max=4096, pkt_min=72, pkt_max=1526,
                 node_links=((0, 1), (1, 2), (1, 3), (1, 5), (3, 4), (5, 4))):
        self.node_mat_gene(node_links=node_links)
        self.node_info_gene(rand_min=rand_min, rand_max=rand_max)
        self.tt_flow_gene(tt_num=tt_num, delay_min=delay_min, delay_max=delay_max,
                          pkt_min=pkt_min, pkt_max=pkt_max)
        # print("function ArchiGenerator gene_all finish")
        return self.node_mat, self.node_info, self.tt_flow

    def write_to_file(self, filename=""):
        """
        specify the path to save the file
        """
        if not os.path.exists('../DataSaved/{}'.format(filename)):
            os.mkdir('../DataSaved/{}'.format(filename))
        if self.node_mat is not None:
            np.save('../DataSaved/{}/node_mat.npy'.format(filename), self.node_mat)
        if self.node_info:
            json.dump(self.node_info,
                      open('../DataSaved/{}/node_info.json'.format(filename), "w"), indent=4)
        if self.tt_flow:
            json.dump(self.tt_flow, open('../DataSaved/{}/tt_flow.json'.format(filename), "w"), indent=4)
        data = (self.node_num, self.node_links, self.rand_min, self.rand_max, self.tt_num, self.tt_flow_cycle_option,
                self.delay_min, self.delay_max, self.pkt_min, self.pkt_max)
        json.dump(data, open('../DataSaved/{}/basic_parameters.json'.format(filename), "w"), indent=4)

    def write_topo_to_file(self, filename=""):
        """
        specify the path to save the topo
        """
        if not os.path.exists('../DataSaved/{}'.format(filename)):
            os.mkdir('../DataSaved/{}'.format(filename))
        if self.topo_set:
            json.dump(self.topo_set, open('../DataSaved/{}/topo_set.json'.format(filename), "w"), indent=4)

    def write_stream_to_file(self, filename=""):
        """
        specify the path to save the topo
        """
        if not os.path.exists('../DataSaved/{}'.format(filename)):
            os.mkdir('../DataSaved/{}'.format(filename))
        if self.stream_set:
            json.dump(self.stream_set, open('../DataSaved/{}/stream_set.json'.format(filename), "w"), indent=4)

    def read_from_file(self, filename):
        """
        specify the path to read the file
        """
        self.node_mat = np.load('../DataSaved/{}/node_mat.npy'.format(filename))
        print('loading existing adjecent matric')
        self.node_info = json.load(open('../DataSaved/{}/node_info.json'.format(filename)))
        print('loading existing node imformation')
        self.tt_flow = json.load(open('../DataSaved/{}/tt_flow.json'.format(filename)))
        print('loading existing TT_flow imformation')
        data = json.load(open('../DataSaved/{}/basic_parameters.json'.format(filename)))
        print('loading existing basic parameters')
        self.node_num, self.node_links, self.rand_min, self.rand_max, self.tt_num, self.tt_flow_cycle_option,\
            self.delay_min, self.delay_max, self.pkt_min, self.pkt_max = data


def main():
    data_gene = ArchiGenerater()
    links = ((0, 1), (1, 2), (1, 3), (1, 5), (3, 4), (5, 4))
    data_gene.gene_all(rand_min=30, rand_max=100, tt_num=30, delay_min=2, delay_max=20,
                       pkt_min=1, pkt_max=5, node_links=links)
    data_gene.write_to_file(filename='Vehicle_NetWork')
    # data_gene.read_from_file(filename='Vehicle_NetWork')
    print(data_gene.node_mat)
    print(data_gene.node_info)
    print(data_gene.tt_flow)


if __name__ == '__main__':
    main()
