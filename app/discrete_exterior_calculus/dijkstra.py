## starting from the vertex closest to (0,0), traverse the shortest edge and increment a running total of the edge lengths
import numpy as np
from heapdict import heapdict
import networkx as nx


def dijkstra(startpos: np.ndarray, graph: nx.Graph, source_node: int) -> np.ndarray:
    center_node = source_node

    to_visit = heapdict()
    to_visit[center_node] = 0

    visited = set()
    distances = [np.inf for _ in range(len(graph))]

    while len(to_visit) > 0:
        current, dist = to_visit.popitem()
        distances[current] = dist
        visited.add(current)

        nbs = list(graph.neighbors(current))
        for nb in nbs:
            a, b = (current, nb)
            if nb not in visited:
                old_dist = distances[nb]
                newdist = dist + graph[a][b]["weight"]
                if newdist < old_dist:
                    to_visit[nb] = newdist
                    distances[nb] = newdist

    # scale distances to [0, 1]
    distances = np.array(distances)
    return distances
