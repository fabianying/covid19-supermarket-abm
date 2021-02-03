import numpy as np
from tqdm import tqdm
import networkx as nx

from covid19_supermarket_abm.path_generators import zone_path_to_full_path_multiple_paths


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


def get_all_shortest_path_dicts(G):
    shortest_path_dict = {}
    for source in tqdm(G):
        shortest_path_dict[source] = {}
        for target in G:
            shortest_path_dict[source][target] = list(nx.all_shortest_paths(G, source, target))
    return shortest_path_dict