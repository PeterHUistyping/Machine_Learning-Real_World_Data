import os
from typing import Dict, Set
from exercises.tick10 import load_graph

from collections import deque


def get_node_betweenness(graph: Dict[int, Set[int]]) -> Dict[int, float]:
    """
    Use Brandes' algorithm to calculate the betweenness centrality for each node in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to that node's betweenness centrality
    """
    # using Brandes algorithm
    V = graph  # convert dict to list (list of dictionary key)
    C = dict((v, 0) for v in graph)
    for s in graph:
        print(s)
        S = []
        pred = dict((w, []) for w in graph)  # initiate
        sig = dict((t, 0) for t in graph)  # 0
        sig[s] = 1
        dist = dict((t, -1) for t in graph)  # infinity
        dist[s] = 0
        Queue = deque([])
        Queue.append(s)
        while Queue:
            v = Queue.popleft()
            S.append(v)
            for w in graph[v]:
                if dist[w] < 0:  # infinity
                    Queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sig[w] += sig[v]
                    pred[w].append(v)
        delta = dict((v, 0) for v in graph)
        while S:
            w = S.pop()
            for v in pred[w]:
                delta[v] += (sig[v] / sig[w]) * (1 + delta[w])
            if w != s:
                C[w] += delta[w]
    for key in C:
        C[key] /= 2
    # Duplicate edge since your program should add the source as a neighbour of the
    #     target as well as the target a neighbour of the source required in tick10
    return C


def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'simple_network.edges'))

    betweenness = get_node_betweenness(graph)
    print(f"Node betweenness values: {betweenness}")


if __name__ == '__main__':
    main()
