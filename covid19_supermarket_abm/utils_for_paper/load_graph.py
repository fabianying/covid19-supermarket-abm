import json
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from covid19_supermarket_abm.utils.shelf_class import Shelf


def load_store_graph(store_id: int, graph_params: dict = {}, data_dir: str = '.') -> nx.Graph:
    # Load parameters
    suffix = graph_params.get('suffix', '')
    graph_suffix = graph_params.get('graph_suffix', '')
    directed = graph_params.get('directed', False)

    if directed:
        create_using = nx.DiGraph()
    else:
        create_using = nx.Graph()

    # Load zone
    df_zone = load_zones(store_id, suffix, data_dir)

    # Load graph
    edge_list_path = os.path.join(data_dir, f'{store_id}_edgelist{suffix}{graph_suffix}.tsv')
    G = nx.read_edgelist(edge_list_path, nodetype=int, create_using=create_using)
    pos = {node: (x_mean, y_mean) for node, x_mean, y_mean in df_zone.loc[:, ['zone', 'x_mean', 'y_mean']].values}

    # Assign positions
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

        # Add edge weight
        weighted_edges = [(u, v, dist(u, v, pos)) for (u, v) in G.edges()]
        G.add_weighted_edges_from(weighted_edges)
    G = G.copy()  # ensures that the nodes are from 0 to n-1
    return G


def load_zones(store_id: int, suffix='', data_dir='.') -> pd.DataFrame:
    # Loading zones
    df_zone = pd.read_csv(os.path.join(data_dir, f'{store_id}_zones_automated{suffix}.csv'))
    df_zone['x_mean'] = (df_zone.x0 + df_zone.x1) / 2
    df_zone['y_mean'] = (df_zone.y0 + df_zone.y1) / 2
    return df_zone


def dist(u, v, pos: dict) -> float:
    (x1, y1) = pos[u]
    (x2, y2) = pos[v]
    return np.linalg.norm([x1 - x2, y1 - y2])


def load_shelves(store_id: int, units='cm', suffix='', data_dir='.') -> List[Shelf]:
    """Import shelves from json file"""
    shelves_file_path = os.path.join(data_dir, f'{store_id}_shelves{suffix}.json')
    with open(shelves_file_path) as f:
        shelves_dict = json.load(f)

    # Convert shelf objects
    all_shelves = []
    for shelf in shelves_dict:
        x = shelf['x']
        y = shelf['y']
        width = shelf['fixtureWidth']
        depth = shelf['fixtureDepth']
        name = shelf['shelfCode']
        angle = shelf['fixtureAngle']
        equipment = shelf['fixtureType']
        if name is None:
            continue
        shelf_obj = Shelf(x, y, width, depth, name, angle, equipment)
        if units == 'cm':
            shelf_obj.convert_to_m()
        all_shelves.append(shelf_obj)
    return all_shelves


def plot_shelves(shelves, ax: Optional[plt.axes] = None, color: str = '#C0C0C0',
                 xdelta: float = 0, ydelta: float = 0, edgecolor="none", **kwargs) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    for shelf in shelves:
        corners = shelf.corners
        # Shift the coordinates
        corners[:, 0] += xdelta
        corners[:, 1] += ydelta
        if xmin > min(corners[:, 0]):
            xmin = min(corners[:, 0])
        if xmax < max(corners[:, 0]):
            xmax = max(corners[:, 0])
        if ymin > min(corners[:, 1]):
            ymin = min(corners[:, 1])
        if ymax < max(corners[:, 1]):
            ymax = max(corners[:, 1])
        ax.add_patch(Polygon(corners, edgecolor=edgecolor, closed=True,
                             fill=True, facecolor=color, hatch='', zorder=1, **kwargs))
    ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])
    plt.axis('equal')
    return ax


def plot_shelves_sainsbury(shelves, ax: Optional[plt.axes] = None, color: str = '#C0C0C0',
                           xdelta: float = 0, ydelta: float = 0, edgecolor="none",
                           with_label=False, plot_special_shelves=True, **kwargs) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    for shelf in shelves:
        corners = shelf.corners
        # midpoint_front = (corners[0] + corners[3])/2
        # Shift the coordinates
        corners[:, 0] += xdelta
        corners[:, 1] += ydelta
        if xmin > min(corners[:, 0]):
            xmin = min(corners[:, 0])
        if xmax < max(corners[:, 0]):
            xmax = max(corners[:, 0])
        if ymin > min(corners[:, 1]):
            ymin = min(corners[:, 1])
        if ymax < max(corners[:, 1]):
            ymax = max(corners[:, 1])
        if shelf.equipment == 'Entrance':
            if plot_special_shelves:
                ax.add_patch(Polygon(corners, edgecolor=edgecolor, closed=True,
                                     fill=True, facecolor='y', hatch='', zorder=0, **kwargs))
        elif shelf.equipment == 'Exit':
            if plot_special_shelves:
                ax.add_patch(Polygon(corners, edgecolor=edgecolor, closed=True,
                                     fill=True, facecolor='r', hatch='', zorder=0, **kwargs))
        elif shelf.equipment == 'Checkout':
            if plot_special_shelves:
                ax.add_patch(Polygon(corners, edgecolor=edgecolor, closed=True,
                                     fill=True, facecolor='orange', hatch='', zorder=0, **kwargs))
        else:
            ax.add_patch(Polygon(corners, edgecolor=edgecolor, closed=True,
                                 fill=True, facecolor=color, hatch='', zorder=0, **kwargs))
        if with_label:
            ax.annotate(shelf.name, shelf.center, ha='center')
        # ax.plot(*midpoint_front, 'x', color='C1')
    ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])
    plt.axis('equal')
    return ax


def get_floor_area(store_id: int, data_dir='.'):
    df = pd.read_csv(os.path.join(data_dir, f'{store_id}_floorarea.csv'))
    df['area'] = (df.x1 - df.x0) * (df.y1 - df.y0)
    return df.area.sum()
