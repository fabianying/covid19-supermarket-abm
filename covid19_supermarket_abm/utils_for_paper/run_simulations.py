import json
import os
import pickle
import logging

import networkx as nx
import pandas as pd

from covid19_supermarket_abm.path_generators import zone_path_to_full_path, get_path_generator
from covid19_supermarket_abm.simulator import simulate_several_days
from covid19_supermarket_abm.utils.create_synthetic_baskets import get_all_shortest_path_dicts
from covid19_supermarket_abm.utils.create_synthetic_store_network import create_small_store, create_medium_store, \
    create_large_store
from covid19_supermarket_abm.utils_for_paper.load_graph import load_store_graph

"""
Functions to run simulations and save results
"""


def load_data_for_sim(store_id, graph_params, data_dir):
    # load data
    suffix = graph_params.get('suffix', '')
    path_suffix = graph_params.get('path_suffix', '')
    graph_suffix = graph_params.get('graph_suffix', '')
    logging.info('Loading data')

    if store_id in [1, 2, 3]:  # synthetic stores
        if store_id == 1:
            G, pos, entrance_nodes, till_nodes, exit_nodes, item_nodes = create_small_store()
            store_size = 'small'
        elif store_id == 2:
            G, pos, entrance_nodes, till_nodes, exit_nodes, item_nodes = create_medium_store()
            store_size = 'medium'
        elif store_id == 3:
            G, pos, entrance_nodes, till_nodes, exit_nodes, item_nodes = create_large_store()
            store_size = 'large'
        else:
            raise RuntimeError('This should not happen.')  # superfluous, but here to help linting
        mu = graph_params.get('mu_basketsize', '')
        sigma = graph_params.get('sigma_basketsize', '')
        pickle_filename = os.path.join(data_dir, f'{store_id}_shortest_path_dict_{store_size}_store.pickle')
        if os.path.isfile(pickle_filename):
            with open(pickle_filename, 'rb') as f:
                shortest_path_dict = pickle.load(f)
            logging.info(f'Loaded shortest path dict: {pickle_filename}')
        else:
            G, pos, entrance_nodes, till_nodes, exit_nodes, item_nodes = create_medium_store()
            shortest_path_dict = get_all_shortest_path_dicts(G)
            logging.info('Calculated shortest path dict of graph')
        extra_outputs = [mu, sigma, entrance_nodes, till_nodes, exit_nodes, item_nodes, shortest_path_dict]
    else:
        paths_path = os.path.join(data_dir, f'{store_id}_zone_paths{suffix}{path_suffix}.parquet')
        logging.info('Loaded zone paths')
        df_paths = pd.read_parquet(paths_path)
        G = load_store_graph(store_id, graph_params, data_dir)
        logging.info('Loaded store graph')
        shortest_path_dict = dict(nx.all_pairs_dijkstra_path(G))
        use_TSP_path = graph_params.get('use_TSP_path', False)
        if use_TSP_path:
            path_col = 'TSP_path' + graph_suffix
        else:
            path_col = 'zone_path'
        df_paths['full_path'] = df_paths[path_col].apply(lambda x: zone_path_to_full_path(x, shortest_path_dict))
        all_zone_paths = list(df_paths[path_col])
        extra_outputs = all_zone_paths
    return G, extra_outputs


def run_several_simulations(config_name, num_iterations, multiplier_list, param='arrival_rate',
                            config_dir='.', data_dir='.', results_dir='.'):
    """Run several simulations where we vary over a specific parameter."""
    config_filename = os.path.join(config_dir, f"{config_name}.json")
    config_original = json.load(open(config_filename))
    logging.info(f'Loaded config file: {config_filename}')
    store_id = config_original['store_id']
    path_generation = config_original.get('path_generation', 'synthetic')

    # load data
    G, extra_outputs = load_data_for_sim(store_id, config_original, data_dir)

    # Do simulations
    for multiplier in multiplier_list:
        config = config_original.copy()
        config[param] *= multiplier
        path_generator_function, path_generator_args = get_path_generator(G, path_generation, zone_paths=extra_outputs,
                                                                          synthetic_path_generator_args=extra_outputs)
        df_cust, df_num_encounter_per_node_stats, df_exposure_time_per_node_stats = simulate_several_days(config, G, path_generator_function,
                                                                                     path_generator_args,
                                                                                     num_iterations=num_iterations)

        results_folder = os.path.join(results_dir, f'{param}')
        if not os.path.isdir(results_folder):
            logging.info(f'Created {results_folder}')
            os.mkdir(results_folder)

        # Save results
        filename1 = os.path.join(results_folder, f'{config_name}_{multiplier * 100:.0f}_{num_iterations}_1.parquet')
        df_cust.to_parquet(filename1)

        df_num_encounter_per_node_stats.columns = df_num_encounter_per_node_stats.columns.astype(str)
        filename2 = os.path.join(results_folder, f'{config_name}_{multiplier * 100:.0f}_{num_iterations}_2.parquet')
        df_num_encounter_per_node_stats.to_parquet(filename2)

        df_exposure_time_per_node_stats.columns = df_exposure_time_per_node_stats.columns.astype(str)
        filename3 = os.path.join(results_folder, f'{config_name}_{multiplier * 100:.0f}_{num_iterations}_3.parquet')
        df_exposure_time_per_node_stats.to_parquet(filename3)
        logging.info(f'Results saved in {filename1}, {filename2}, {filename3}.')


def run_one_simulation_and_record_stats(config_name, num_iterations, config_dir='.', data_dir='.', results_dir='.'):
    """Make one simulation with no multipliers."""
    print('Running simulation with no modified parameters')
    config_original = json.load(open(os.path.join(config_dir, f"{config_name}.json")))
    store_id = config_original['store_id']

    # Load data
    all_zone_paths, G = load_data_for_sim(store_id, graph_params=config_original, data_dir=data_dir)

    # Do simulations
    df_cust, df_encounter_stats, df_encounter_time_stats = simulate_several_days(config_original, all_zone_paths,
                                                                                 G,
                                                                                 num_iterations=num_iterations)
    results_folder = os.path.join(results_dir, 'results')
    if not os.path.isdir(results_folder):
        print(f'Created {results_folder}')
        os.mkdir(results_folder)

    # Save results
    filename1 = os.path.join(results_folder, f'{config_name}_{num_iterations}_1.parquet')
    df_cust.to_parquet(filename1)
    df_encounter_stats.columns = df_encounter_stats.columns.astype(str)
    filename2 = os.path.join(results_folder, f'{config_name}_{num_iterations}_2.parquet')
    df_encounter_stats.to_parquet(filename2)
    df_encounter_time_stats.columns = df_encounter_time_stats.columns.astype(str)
    filename3 = os.path.join(results_folder, f'{config_name}_{num_iterations}_3.parquet')
    df_encounter_time_stats.to_parquet(filename3)
    print(f'Results saved in {filename1}, {filename2}, {filename3}.')
