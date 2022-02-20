from node import Node
from edge import Edge
from parameters import *
import numpy as np
import json


class Graph:
    """
    includes all the nodes and edges
    make up the whole graphic
    """
    def __init__(self):
        # Basic information
        self.adjacent_node_matrix = []
        self.reachable_node_matrix = []
        self.distance_node_matrix = []
        # [source_node, destiny_node, cycle, max_delay, pkt_length]
        self.flow_dic = {}
        # uesed for static routing
        self.routes_dic = {}
        # edge adjacent should be used in dynamic routing
        self.adjacent_edge_matrix = []
        # Nodes and edges id start with 0
        self.nodes = {}
        self.edges = {}
        # (start node, end node) -> edge id
        self.node_to_edge = {}
        # Information initialization
        self.load_nodes()
        self.load_adjacent_matrix()
        self.load_flows()
        self.load_routes()
        # should be used in dynamic scheduling
        # self.init()

    def init(self):
        self.get_edges()
        self.get_reachable_edge_matrix()
        self.get_distance_edge_matrix()

    def load_nodes(self):
        if not self.nodes:
            # if nodes list is empty
            info = json.load(open('./DataSaved/{}/node_info.json'.format(args.data_path)))
            for node_id, buffer in info.items():
                self.nodes[int(node_id)] = Node(int(node_id), buffer)
        else:
            print('nodes list already existing')

    def load_adjacent_matrix(self):
        if not self.adjacent_node_matrix:
            self.adjacent_node_matrix = np.load('./DataSaved/{}/node_mat.npy'.format(args.data_path), allow_pickle=True)
        else:
            print('node adjacent matric already existing')

    def load_flows(self):
        if not self.flow_dic:
            self.flow_dic = json.load(open('./DataSaved/{}/tt_flow.json'.format(args.data_path)))
        else:
            print('flows info dictionary already existing')

    def load_routes(self):
        if not self.routes_dic:
            self.routes_dic = json.load(open('./DataSaved/{}/routes.json'.format(args.data_path)))
        else:
            print('routes dictionary already existing')

    def get_edges(self):
        """
        edges should be used in dynamic routing
        expansion needed
        """
        if not self.edges:
            edge_id = 0
            nodes_num = len(self.adjacent_node_matrix)
            self.node_to_edge = {}
            for i in range(nodes_num):
                for j in range(nodes_num):
                    if self.adjacent_node_matrix[i][j] == 0 or i == j:
                        continue
                    self.edges[edge_id] = Edge(edge_id, self.nodes[i], self.nodes[j])
                    self.node_to_edge[(i, j)] = edge_id
                    edge_id += 1
            print("edge number: {}; id start with 0".format(edge_id+1))
            # initialize edge adjacent matrix
            pass
        else:
            print('edge list already existing')

    def update_graph_depth(self):
        """
        Assuming an adjacent matric as G
        G^i represents the number of paths of length i between any two nodes
        """
        pass

    def get_reachable_edge_matrix(self):
        pass

    def get_distance_edge_matrix(self):
        pass

    def delete_edge(self, edge_id, tt_flow_to_edge):
        pass

    def reset(self):
        for node in self.nodes:
            node.reset()


def main():
    graph = Graph()
    for i in range(len(graph.nodes)):
        print(i, end=": ")
        for j in range(len(graph.nodes)):
            if graph.adjacent_node_matrix[i][j] == 1:
                print(j, end=' ')
    print()
    for key in graph.nodes:
        print(graph.nodes[key].id)


if __name__ == '__main__':
    main()
