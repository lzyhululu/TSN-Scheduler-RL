"""
including the calculate functions of the routing problem
might include dynamic routing algorithm in the future
"""
import json


def DFS(i, node_mat):
    """
    depth first traversal
    judge whether the nodes are reachable or not
    """
    node_visited = [False for _ in range(len(node_mat))]
    node_num = len(node_mat)
    j = 0
    print("node:{}".format(i), end=" ")
    node_visited[i] = True
    while j < node_num:
        if (node_mat[i][j] != 0) and (not node_visited[j]):
            print(node_mat[i][j], end=" ")
            DFS(j, node_mat)
        j += 1


def single_dijkstra(start, node_mat):
    """
    calculate the shortest path
    using dijkstra algorithm and asdjecent matric of nodes
    """
    node_num = len(node_mat)
    inf = float('inf')
    # Used to store the shortest path length from start to other nodes
    distance = [inf for _ in range(node_num)]
    # Used to store the shortest path from start to other nodes
    path = ['{},'.format(start) for _ in range(node_num)]
    # whether the shortest path is found from start to other nodes
    final = [False for _ in range(node_num)]
    for i in range(node_num):
        final[i] = False
        distance[i] = node_mat[start][i]
        if distance[i] < inf:
            # if start is directly connected to node i, the path will be directly changed to i
            path[i] = path[i] + str(i)
    distance[start] = 0
    final[start] = True
    # beginning
    for i in range(1, node_num):
        min_dis = inf
        v = start
        for k in range(node_num):
            if (not final[k]) and (distance[k] < min_dis):
                v = k
                min_dis = distance[k]
        # The nearest point is found
        final[v] = True
        for k in range(node_num):
            if (not final[k]) and (min_dis + node_mat[v][k] < distance[k]):
                # if the detour is closer
                distance[k] = min_dis + node_mat[v][k]
                path[k] = path[v] + ',' + str(k)
    return distance, path


def route_dijkstra(node_mat):
    """
    Calculate routes for all nodes
    """
    graph_route = {}
    node_num = len(node_mat)
    for i in range(node_num):
        _, single_route = single_dijkstra(i, node_mat)
        node_index = [j for j in range(node_num)]
        route_dict = dict(zip(node_index, single_route))
        del route_dict[i]
        graph_route[i] = route_dict
    print('all routes calculated')
    return graph_route


def main():
    from utility.archi import ArchiGenerater
    data_gene = ArchiGenerater()
    data_gene.read_from_file(filename='Vehicle_NetWork')
    print(data_gene.node_mat)
    routes_dic = route_dijkstra(data_gene.node_mat)
    json.dump(routes_dic, open('../DataSaved/Vehicle_NetWork/routes.json', "w"), indent=4)


if __name__ == '__main__':
    main()
