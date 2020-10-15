import json
import os

import networkx as nx
import pandas as pd

from covid19_supermarket_abm.path_generators import zone_path_to_full_path
from covid19_supermarket_abm.simulator import simulate_several_days
from covid19_supermarket_abm.utils_for_paper.load_graph import load_store_graph


def load_data_for_sim(store_id, graph_params, data_dir):
    # load data
    suffix = graph_params.get('suffix', '')
    path_suffix = graph_params.get('path_suffix', '')
    graph_suffix = graph_params.get('graph_suffix', '')

    paths_path = os.path.join(data_dir, f'{store_id}_zone_paths{suffix}{path_suffix}.parquet')
    df_paths = pd.read_parquet(paths_path)
    G = load_store_graph(store_id, graph_params, data_dir)
    shortest_path_dict = dict(nx.all_pairs_dijkstra_path(G))
    use_TSP_path = graph_params.get('use_TSP_path', False)
    if use_TSP_path:
        path_col = 'TSP_path' + graph_suffix
    else:
        path_col = 'zone_path'
    df_paths['full_path'] = df_paths[path_col].apply(lambda x: zone_path_to_full_path(x, shortest_path_dict))
    all_zone_paths = list(df_paths[path_col])
    return all_zone_paths, G


def run_several_simulations(config_name, num_iterations, multiplier_list, param='arrival_rate',
                            root_dir='.', data_dir='.'):
    """Run several simulations where we vary over a specific parameter."""
    config_original = json.load(open(os.path.join(root_dir, f"{config_name}.json")))
    store_id = config_original['store_id']

    # load data
    all_zone_paths, G = load_data_for_sim(store_id, config_original, data_dir)

    # Do simulations
    for multiplier in multiplier_list:
        config = config_original.copy()
        config[param] *= multiplier
        df_cust, df_encounter_stats, df_encounter_time_stats = simulate_several_days(config, all_zone_paths, G,
                                                                                     num_iterations=num_iterations)

        results_folder = os.path.join(root_dir, f'/results/{param}')
        if not os.path.isdir(results_folder):
            print(f'Created {results_folder}')
            os.mkdir(results_folder)

        # Save results
        filename1 = os.path.join(results_folder, f'{config_name}_{multiplier * 100:.0f}_{num_iterations}_1.parquet')
        df_cust.to_parquet(filename1)

        df_encounter_stats.columns = df_encounter_stats.columns.astype(str)
        filename2 = os.path.join(results_folder, f'{config_name}_{multiplier * 100:.0f}_{num_iterations}_2.parquet')
        df_encounter_stats.to_parquet(filename2)

        df_encounter_time_stats.columns = df_encounter_time_stats.columns.astype(str)
        filename3 = os.path.join(results_folder, f'{config_name}_{multiplier * 100:.0f}_{num_iterations}_3.parquet')
        df_encounter_time_stats.to_parquet(filename3)
        print(f'Results saved in {filename1}, {filename2}, {filename3}.')


def run_one_simulation_and_record_stats(config_name, num_iterations, root_dir='.', data_dir='.'):
    """Make one simulation with no multipliers."""
    print('Running simulation with no modified parameters')
    config_original = json.load(open(os.path.join(root_dir, f"{config_name}.json")))
    store_id = config_original['store_id']

    # Load data
    all_zone_paths, G = load_data_for_sim(store_id, graph_params=config_original, data_dir=data_dir)

    # Do simulations
    df_cust, df_encounter_stats, df_encounter_time_stats = simulate_several_days(config_original, all_zone_paths,
                                                                                 G,
                                                                                 num_iterations=num_iterations)
    results_folder = os.path.join(root_dir, 'results')
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
