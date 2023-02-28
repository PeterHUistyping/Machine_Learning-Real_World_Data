import glob
import os
from typing import Dict, Set


def load_graph(filename: str) -> Dict[int, Set[int]]:
    """
    Load the graph file. Each line in the file corresponds to an edge; the first column is the source node and the
    second column is the target. As the graph is undirected, your program should add the source as a neighbour of the
    target as well as the target a neighbour of the source.

    @param filename: The path to the network specification
    @return: a dictionary mapping each node (represented by an integer ID) to a set containing all the nodes it is
        connected to (also represented by integer IDs)
    """
    data = dict()
    with open(filename, encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            edge = line.strip()
            ed = edge.split()
            source = int(ed[0])
            target = int(ed[1])
            if source in data:
                data[source].add(target)
            else:
                data[source] = set()
                data[source].add(target)
            if target in data:
                data[target].add(source)
            else:
                data[target] = set()
                data[target].add(source)
    return data


def get_node_degrees(graph: Dict[int, Set[int]]) -> Dict[int, int]:
    """
    Find the number of neighbours of each node in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to the degree of the node
    """
    graph_ = graph.copy()
    for key in graph:
        graph_[key] = len(graph[key])
    return graph_


def shortest_paths(g, s, t):
    queue = []
    paths = []
    distance = {s: 0}
    queue.append((s, [s]))
    while queue:
        node, path = queue.pop(0)
        if node == t:
            return path
            # paths.append(path)
            # continue
        if node in g:
            for neighbour in g[node]:
                if neighbour not in distance or distance[neighbour] == distance[node] + 1:
                    distance[neighbour] = distance[node] + 1
                    queue.append((neighbour, path + [neighbour]))
    return paths


def get_diameter(graph: Dict[int, Set[int]]) -> int:
    """
    Find the longest shortest path between any two nodes in the network using a breadth-first search.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: the length of the longest shortest path between any pair of nodes in the graph
    """
    diameter = 0
    key = list(graph)
    len_key = len(key)

    Shortest_paths = [-1] * (len_key+2)

    # for s in range(0, len_key):
    #     Flag = [False for i in range(len_key)]
    #     q = []
    #     Flag[s] = True
    #     q.append(s)
    #     Shortest_paths[s][s] = 0
    #     while q:
    #         v = q.pop(0)
    #         if v in graph:
    #             for neighbour in graph[v]:
    #                 if not Flag[neighbour]:
    #                     Flag[neighbour] = True
    #                     Shortest_paths[s][neighbour] = Shortest_paths[s][v] + 1
    #                     print(Shortest_paths[s][neighbour],s,neighbour)
    #                     diameter = max(diameter, Shortest_paths[s][neighbour])
    #                     q.append(neighbour)
    # return diameter
    for s in range(0, len_key):
        Flag = [False for i in range(len_key+2)]
        q = []
        Flag[s] = True
        q.append(s)
        Shortest_paths[s] = 0
        while q:
            v = q.pop(0)
            if v in graph:
                for neighbour in graph[v]:
                    # print(len(Flag),neighbour)
                    if not Flag[neighbour]:
                        Flag[neighbour] = True
                        Shortest_paths[neighbour] = Shortest_paths[v] + 1
                        # print(Shortest_paths[neighbour], s, neighbour)
                        diameter = max(diameter, Shortest_paths[neighbour])
                        q.append(neighbour)
    return diameter


def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'simple_network.edges'))
    degrees = get_node_degrees(graph)
    print(f"Node degrees: {degrees}")

    diameter = get_diameter(graph)
    print(f"Diameter: {diameter}")


if __name__ == '__main__':
    main()
