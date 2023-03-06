import os
from collections import deque

from typing import Set, Dict, List, Tuple
from exercises.tick10 import load_graph


def get_number_of_edges(graph: Dict[int, Set[int]]) -> int:
    """
    Find the number of edges in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: the number of edges
    """
    sum = 0
    for each_set in graph:
        sum += len(graph[each_set])
    return sum // 2
    # Duplicate edge since your program should add the source as a neighbour of the
    #     target as well as the target a neighbour of the source required in tick10


def get_components(graph: Dict[int, Set[int]]) -> List[Set[int]]:
    """
    Find the number of components in the graph using a DFS.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: list of components for the graph.
    """
    component = []
    visited = [False] * (max(list(graph)) + 2)
    connected_component = 0

    def dfs(u):
        for neighbour in graph[u]:
            if not visited[neighbour]:
                component_temp.add(neighbour)
                visited[neighbour] = True
                dfs(neighbour)

    for u in graph:
        if not visited[u]:
            visited[u] = True
            connected_component += 1
            component_temp = set()
            component_temp.add(u)
            dfs(u)
            component.append(component_temp)
    return component


def get_edge_betweenness(graph: Dict[int, Set[int]]) -> Dict[Tuple[int, int], float]:
    """
    Calculate the edge betweenness.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: betweenness for each pair of vertices in the graph connected to each other by an edge
    """
    # for i in  get_number_of_edges(graph):
    # C = dict((v, 0) )*
    C = dict()
    for s in graph:
        # print(s)
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
                c = (sig[v] / sig[w]) * (1 + delta[w])
                delta[v] += c
                if (v, w) in C:
                    C[(v, w)] += c
                else:
                    C[(v, w)] = c
            # if w != s:
            #     C[w] += delta[w]
    # for key in C:
    #     C[key] /= 2
    # Duplicate edge since your program should add the source as a neighbour of the
    #     target as well as the target a neighbour of the source required in tick10
    return C


def girvan_newman(graph: Dict[int, Set[int]], min_components: int) -> List[Set[int]]:
    """     * Find the number of edges in the graph.
     *
     * @param graph
     *        {@link Map}<{@link Integer}, {@link Set}<{@link Integer}>> The
     *        loaded graph
     * @return {@link Integer}> Number of edges.
    """
    g = graph.copy()
    component = get_components(g)
    while get_number_of_edges(g) > 0 and len(component)<min_components :
        max_bet = -1
        maxEdge = []
        betweenness = get_edge_betweenness(g)
        for key in betweenness:
            new = betweenness[key]
            if new >= max_bet - 1e-6:
                if abs(new - max_bet) < 1e-6:
                    maxEdge.append(key)
                else:
                    maxEdge = [key]
                max_bet = new
        for (v, w) in maxEdge:
            g[v].remove(w)
        component = get_components(g)
    return get_components(g)


def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'facebook_circle.edges'))

    num_edges = get_number_of_edges(graph)
    print(f"Number of edges: {num_edges}")

    components = get_components(graph)
    print(f"Number of components: {len(components)}")
    print(len(components))

    edge_betweenness = get_edge_betweenness(graph)
    print(f"Edge betweenness: {edge_betweenness}")

    clusters = girvan_newman(graph, min_components=20)
    print(f"Girvan-Newman for 20 clusters: {clusters}")
    print(len(clusters))

if __name__ == '__main__':
    main()
