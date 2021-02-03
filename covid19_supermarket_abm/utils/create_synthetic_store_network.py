import logging

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def create_small_store():
    num_aisles_per_side = 1
    num_aisles = 5
    aisle_length = 3
    edge_length = 2
    G, pos, entrance_nodes, till_nodes, exit_nodes, item_nodes = create_synthetic_store_network(num_aisles_per_side,
                                                                                                num_aisles,
                                                                                                aisle_length,
                                                                                                edge_length)
    logging.info(f'Created small store. Floor area: {get_floor_area(pos, edge_length)} m^2')
    return G, pos, entrance_nodes, till_nodes, exit_nodes, item_nodes


def create_medium_store():
    num_aisles_per_side = 2
    num_aisles = 7
    aisle_length = 4
    edge_length = 2
    G, pos, entrance_nodes, till_nodes, exit_nodes, item_nodes = create_synthetic_store_network(num_aisles_per_side,
                                                                                                num_aisles,
                                                                                                aisle_length,
                                                                                                edge_length)
    logging.info(f'Created medium store. Floor area: {get_floor_area(pos, edge_length)} m^2')
    return G, pos, entrance_nodes, till_nodes, exit_nodes, item_nodes


def create_large_store():
    num_aisles_per_side = 2
    num_aisles = 15
    aisle_length = 6
    edge_length = 2
    G, pos, entrance_nodes, till_nodes, exit_nodes, item_nodes = create_synthetic_store_network(num_aisles_per_side,
                                                                                                num_aisles,
                                                                                                aisle_length,
                                                                                                edge_length)
    logging.info(f'Created large store. Floor area: {get_floor_area(pos, edge_length)} m^2')
    return G, pos, entrance_nodes, till_nodes, exit_nodes, item_nodes


def create_synthetic_store_network(num_aisles_per_side=2, num_aisles=7, aisle_length=3, edge_length=2):
    nrows = aisle_length * num_aisles_per_side + num_aisles_per_side + 2
    ncols = num_aisles * 2 + 1

    # Create initial grid graph
    G = nx.grid_2d_graph(nrows, ncols)

    # Remove a number of nodes
    col_idx_to_remove = [2 * k + 1 for k in range(int(ncols / 2))]
    row_idx_to_remove = []
    for i in range(num_aisles_per_side):
        for j in range(aisle_length):
            row_idx_to_remove.append(j + 2 + (i * (aisle_length + 1)))

    G.remove_nodes_from(product(row_idx_to_remove, col_idx_to_remove))

    # Specify the entrance, till, exit and item nodes
    entrance = [(0, 0)]
    tills = [(0, i) for i in col_idx_to_remove[1:]]
    exit = [(0, 1)]
    item_nodes = [(x,y) for (x,y) in G if (x >= 2)]

    # Relabel nodes from (0,0), (0, 1), ... to 0, 1, ...
    node_mapping = {node: idx for idx, node in enumerate(G)}
    pos = {idx: (edge_length * x, edge_length * y) for idx, (y, x) in enumerate(G)}
    G = nx.relabel_nodes(G, node_mapping)
    entrance_nodes = relabel_node_list(entrance, node_mapping)
    till_nodes = relabel_node_list(tills, node_mapping)
    exit_nodes  = relabel_node_list(exit, node_mapping)
    item_nodes = relabel_node_list(item_nodes, node_mapping)

    # Assign positions
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

        # Add edge weight
        weighted_edges = [(u, v, dist(u, v, pos)) for (u, v) in G.edges()]
        G.add_weighted_edges_from(weighted_edges)

    return G, pos, entrance_nodes, till_nodes, exit_nodes, item_nodes


def plot_graph(G, pos, entrance_nodes, till_nodes, exit_nodes, item_nodes, edge_length=2, with_labels=False, **kwargs):
    node_color = []
    for node in G:
        if node in entrance_nodes:
            color = 'yellow'
        elif node in till_nodes:
            color = 'orange'
        elif node in exit_nodes:
            color = 'r'
        elif node in item_nodes:
            color = 'lightblue'
        else:
            color = 'white'
        node_color.append(color)

    nrows, ncols = get_nrows_ncols_from_pos(pos, edge_length)
    fig, ax = plt.subplots(figsize=(ncols, nrows))
    nx.draw_networkx(G, pos=pos,
                     node_color=node_color,
                     with_labels=with_labels,
                     edgecolors='k',
                     node_size=600, ax=ax, **kwargs)
    ax.set_axis_off()
    return fig, ax


def get_nrows_ncols_from_pos(pos, edge_length):
    max_x, max_y = np.array(list(pos.values())).max(axis=0)
    nrows = (max_y + 1) / edge_length
    ncols = (max_x + 1) / edge_length
    return nrows, ncols


def get_floor_area(pos, edge_length):
    nrows, ncols = get_nrows_ncols_from_pos(pos, edge_length)
    return nrows * ncols * edge_length * edge_length


def relabel_node_list(node_list, node_mapping):
    return [node_mapping[node] for node in node_list]


def dist(u, v, pos: dict) -> float:
    (x1, y1) = pos[u]
    (x2, y2) = pos[v]
    return np.linalg.norm([x1 - x2, y1 - y2])
