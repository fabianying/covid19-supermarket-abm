import random
from typing import List, Dict

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


def get_path_generator(G: nx.Graph, zone_paths: List[List[int]], path_generation: str = 'empirical'):
    """Create path generator functions and args from the list of zone paths.
    Each zone path is a sequence of zones that a customer purchased items from, so consecutive zones in the sequence
    may not be adjacent in the store graph. We map the zone path to the full shopping path by assuming that
    customers walk shortest paths between purchases."""

    shortest_path_dict = dict(nx.all_pairs_dijkstra_path(G))
    shopping_paths = [zone_path_to_full_path(path, shortest_path_dict) for path in zone_paths]

    # Decide how paths are generated
    if path_generation == 'tmatrix':
        tmatrix = get_transition_matrix(shopping_paths, len(G))
        path_generator_function = path_generator_from_transition_matrix
        path_generator_args = [tmatrix, shortest_path_dict]
    elif path_generation == 'empirical':
        path_generator_function = paths_generator_from_actual_paths
        all_paths = [zone_path_to_full_path(path, shortest_path_dict) for path in shopping_paths]
        path_generator_args = [all_paths]
    else:
        raise ValueError(f'Unknown path_generation scheme == {path_generation}')
    return path_generator_function, path_generator_args
