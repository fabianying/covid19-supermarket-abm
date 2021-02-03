import random
from typing import List, Dict, Optional

import networkx as nx
import numpy as np


def paths_generator_from_actual_paths(all_paths):
    num_paths = len(all_paths)
    while True:
        i = np.random.randint(0, num_paths)
        yield all_paths[i]


def path_generator_from_transition_matrix(tmatrix: List[List[int]], shortest_path_dict: Dict):
    while True:
        yield zone_path_to_full_path(create_one_path(tmatrix), shortest_path_dict)


def get_transition_matrix(all_paths, num_states):
    n = num_states + 1  # number of states

    transition_matrix = [[0] * n for _ in range(n)]
    for path in all_paths:
        for (i, j) in zip(path, path[1:]):
            transition_matrix[i][j] += 1
        transition_matrix[path[-1]][n - 1] += 1  # ending

    # now convert to probabilities:
    for row in transition_matrix:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]
    return transition_matrix


def zone_path_to_full_path(zone_path, shortest_path_dict):
    full_path_dict = []
    for start, end in zip(zone_path[:-1], zone_path[1:]):
        full_path_dict += shortest_path_dict[start][end]
    return full_path_dict


def zone_path_to_full_path_multiple_paths(zone_path, shortest_path_dict):
    """Use this if there are shortest_path_dict gives multiple shortest_paths for each source and target"""
    L = []
    for start, end in zip(zone_path[:-1], zone_path[1:]):
        shortest_paths = shortest_path_dict[start][end]
        num_shortest_paths = len(shortest_paths)
        if num_shortest_paths > 0:
            i = np.random.randint(num_shortest_paths)
        else:
            i = 0
        L += shortest_paths[i]
    return L

def sample_num_products_in_basket_batch(mu, sigma, num_baskets):
    """
    Sample the number of items in basket using Mixed Poisson Log-Normal distribution.
    Reference: Sorensen H, Bogomolova S, Anderson K, Trinh G, Sharp A, Kennedy R, et al.
    Fundamental patterns of in-store shopper behavior. Journal of Retailing and Consumer Services. 2017;37:182–194.

    :param mu: Mean value of the underlying normal distribution
    :param sigma: Standard deviation of the underlying normal distribution
    :param num_baskets: number of baskets to sample
    :return: List of length num_baskets with the number of items in each basket
    """

    norm = np.random.lognormal(mean=mu, sigma=sigma, size=3 * num_baskets)
    num_items = np.random.poisson(norm)
    num_items = num_items[num_items > 0]
    assert len(num_items) >= num_baskets, \
        f"Somehow didn't get the enough non-zero baskets ({num_items} <= {num_baskets} (size))"
    return num_items[:num_baskets]


def create_random_item_paths(num_items, entrance_nodes, till_nodes, exit_nodes, item_nodes):
    """
    Create random item path based on the number of items in each basket and the shelves were items are located.
    We choose items uniformly at random from all item_nodes.
    We also choose a random entrance node, till node, and exit node (sampled uniformly at random from the
    corresponding nodes).
    """
    num_baskets = len(num_items)
    random_entrance_nodes = np.random.choice(entrance_nodes, size=num_baskets)
    random_till_nodes = np.random.choice(till_nodes, size=num_baskets)
    random_exit_nodes = np.random.choice(exit_nodes, size=num_baskets)
    concatenated_baskets = np.random.choice(item_nodes, size=np.sum(num_items))
    break_points = np.cumsum(num_items)
    item_paths = []
    start = 0
    i = 0
    for end in break_points:
        entrance = random_entrance_nodes[i]
        till = random_till_nodes[i]
        exit = random_exit_nodes[i]
        basket = [entrance] + list(concatenated_baskets[start:end]) + [till, exit]
        item_paths.append(basket)
        start = end
        i += 1
    return item_paths


def sythetic_paths_generator(mu, sigma, entrance_nodes, till_nodes, exit_nodes, item_nodes,
                                                shortest_path_dict, batch_size=1000):
    while True:
        num_items = sample_num_products_in_basket_batch(mu, sigma, batch_size)
        item_paths = create_random_item_paths(num_items, entrance_nodes, till_nodes, exit_nodes, item_nodes)
        for item_path in item_paths:
            full_path = zone_path_to_full_path_multiple_paths(item_path, shortest_path_dict)
            yield full_path



def get_next_term(num_states, trow):
    return random.choices(range(num_states), trow)[0]


def create_one_path(tmatrix: List[List[int]]):
    num_states = len(tmatrix)
    start_term = 0
    end = num_states - 1
    chain = [start_term]
    length = 1
    while True:
        current_position = get_next_term(num_states, tmatrix[chain[-1]])
        if current_position == end:
            break
        elif length > 100000:
            print('Generated is over 100000 stops long. Something must have gone wrong!')
            break
        chain.append(current_position)
        length += 1
    return chain


def replace_till_zone(path, till_zone, all_till_zones):
    assert path[-1] == till_zone, f'Final zone is not {till_zone}, but {path[-1]}'
    path[-1] = np.random.choice(all_till_zones)
    return path


def get_path_generator(G: nx.Graph, path_generation: str = 'empirical', zone_paths: List[List[int]]=None,
                       synthetic_path_generator_args: Optional[list] = None):
    """Create path generator functions and args from the list of zone paths.
    Each zone path is a sequence of zones that a customer purchased items from, so consecutive zones in the sequence
    may not be adjacent in the store graph. We map the zone path to the full shopping path by assuming that
    customers walk shortest paths between purchases."""

    # Decide how paths are generated

    if path_generation == 'synthetic':
        path_generator_function = sythetic_paths_generator
        path_generator_args = synthetic_path_generator_args  # [mu, sigma, entrance_nodes,
        # till_nodes, exit_nodes, item_nodes]
    elif path_generation == 'tmatrix':
        shortest_path_dict = dict(nx.all_pairs_dijkstra_path(G))
        shopping_paths = [zone_path_to_full_path(path, shortest_path_dict) for path in zone_paths]
        tmatrix = get_transition_matrix(shopping_paths, len(G))
        path_generator_function = path_generator_from_transition_matrix
        path_generator_args = [tmatrix, shortest_path_dict]
    elif path_generation == 'empirical':
        shortest_path_dict = dict(nx.all_pairs_dijkstra_path(G))
        shopping_paths = [zone_path_to_full_path(path, shortest_path_dict) for path in zone_paths]
        path_generator_function = paths_generator_from_actual_paths
        all_paths = [zone_path_to_full_path(path, shortest_path_dict) for path in shopping_paths]
        path_generator_args = [all_paths]
    else:
        raise ValueError(f'Unknown path_generation scheme == {path_generation}')
    return path_generator_function, path_generator_args
