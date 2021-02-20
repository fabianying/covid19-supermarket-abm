import numpy as np
import networkx as nx
from typing import Dict, Tuple, List


def create_store_network(pos: Dict[int, Tuple[float, float]], edges: List[Tuple[int, int]], directed=False) -> nx.Graph:
    """
    Create a store network from positions dictionary and edges.
    :param pos: Dictionary of node positions
    :param edges: List of edges
    :param directed: Set to True to create directed network
    :return: G: Store network
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_edges_from(edges)
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

    # Add edge weight
    weighted_edges = [(u, v, dist(u, v, pos)) for (u, v) in G.edges()]
    G.add_weighted_edges_from(weighted_edges)
    return G


def dist(u, v, pos: dict) -> float:
    (x1, y1) = pos[u]
    (x2, y2) = pos[v]
    return np.linalg.norm([x1 - x2, y1 - y2])
