import multiprocessing
from itertools import repeat

import networkx as nx
import numpy as np
import pandas as pd
import simpy
from tqdm import tqdm

from covid19_supermarket_abm.core import Store, _customer_arrivals, _stats_recorder, \
    _sanity_checks
from covid19_supermarket_abm.utils import istarmap  # enable progress bar with multiprocessing


def simulate_one_day(config: dict, G: nx.Graph, path_generator_function, path_generator_args: list):
    # Get parameters
    logging_enabled= config.get('logging_enabled', False)
    num_hours_open = config.get('num_hours_open', 12)
    with_node_capacity = config.get('with_node_capacity', False)
    max_customers_in_store_per_sqm = config.get('max_customers_in_store_per_sqm', None)
    floorarea = config.get('floorarea', None)
    if max_customers_in_store_per_sqm is None:
        max_customers_in_store = config.get('max_customers_in_store', None)
    else:
        if floorarea is not None:
            max_customers_in_store = int(max_customers_in_store_per_sqm * floorarea )
        else:
            raise ValueError('If you set the parameter "max_customers_in_store_per_sqm", '
                             'you need to specify the floor area via the "floorarea" parameter in the config.')

    env = simpy.Environment()
    store = Store(env, G, logging_enabled=logging_enabled, max_customers_in_store=max_customers_in_store)
    if with_node_capacity:
        node_capacity = config.get('node_capacity', 2)
        store.enable_node_capacity(node_capacity)

    path_generator = path_generator_function(*path_generator_args)
    env.process(_customer_arrivals(env, store, path_generator, config))
    env.process(_stats_recorder(store))
    env.run(until=num_hours_open * 60 * 4)

    # Record stats
    _sanity_checks(store)
    num_cust = len(store.customers)
    num_S = len(store.number_encounters_with_infected)
    shopping_times = list(store.shopping_times.values())
    waiting_times = np.array(list(store.waiting_times.values()))
    b = np.array(waiting_times) > 0
    num_waiting_people = sum(b)
    if num_waiting_people > 0:
        mean_waiting_time = np.mean(waiting_times[b])
    else:
        mean_waiting_time = 0

    num_contacts_per_cust = [contacts for contacts in store.number_encounters_with_infected.values() if contacts != 0]
    df_num_encounters = pd.DataFrame(store.number_encounters_per_node, index=[0])
    df_num_encounters = df_num_encounters[range(len(G))]
    df_time_with_infected = pd.DataFrame(store.time_with_infected_per_node, index=[0])
    df_time_with_infected = df_time_with_infected[range(len(G))]
    results = {'num_cust': num_cust,
               'num_S': num_S,
               'num_I': num_cust - num_S,
               'total_time_with_infected': sum(store.time_with_infected.values()),
               'num_contacts_per_cust': num_contacts_per_cust,
               'num_cust_w_contact': len(num_contacts_per_cust),
               'mean_num_cust_in_store': np.mean(list(store.stats['num_customers_in_store'].values())),
               'max_num_cust_in_store': max(list(store.stats['num_customers_in_store'].values())),
               'num_contacts': sum(num_contacts_per_cust),
               'shopping_times': shopping_times,
               'mean_shopping_time': np.mean(shopping_times),
               'num_waiting_people': num_waiting_people,
               'mean_waiting_time': mean_waiting_time,
               'store_open_length': max(list(store.stats['num_customers_in_store'].keys())),
               'df_num_encounters': df_num_encounters,
               'df_time_with_infected': df_time_with_infected,
               'total_time_crowded': store.total_time_crowded
               }

    if floorarea is not None:
        results['mean_num_cust_in_store_per_sqm'] = results['mean_num_cust_in_store'] / floorarea
        results['max_num_cust_in_store_per_sqm'] = results['max_num_cust_in_store'] / floorarea
    return results


def simulate_several_days(config: dict, G: nx.Graph, path_generator_function, path_generator_args: list,
                          num_iterations: int = 10000, use_parallel: bool = False):
    if use_parallel:
        args = [config, G, path_generator_function, path_generator_args]
        repeated_args = zip(*[repeat(item, num_iterations) for item in args])
        num_cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_cores) as p:
            with tqdm(total=num_iterations) as pbar:
                results = []
                for i, results_dict in enumerate(p.istarmap(simulate_one_day, repeated_args)):
                    results.append(results_dict)
                    pbar.update()
    else:
        results = []
        for _ in tqdm(range(num_iterations)):
            results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)
            results.append(results_dict)

    # Initialize
    df_num_encounters_list = []
    df_time_with_infected_list = []
    stats_dict = {}
    cols_to_record = [key for key, val in results_dict.items()
                      if isinstance(val, (int, np.integer)) or isinstance(val, (float, np.float))]
    cols_not_recording = [key for key in results_dict.keys() if key not in cols_to_record]
    print(f'Recording stats for {cols_to_record}.')
    print(f'We are not recording {cols_not_recording}.')
    for stat in cols_to_record:
        stats_dict[stat] = []

    # Record encounter stats as well
    for i in range(num_iterations):
        results_dict = results[i]
        df_num_encounters_list.append(results_dict['df_num_encounters'])
        df_time_with_infected_list.append(results_dict['df_time_with_infected'])
        for col in cols_to_record:
            stats_dict[col].append(results_dict[col])
    df_cust = pd.DataFrame(stats_dict)
    df_encounter_stats = pd.concat(df_num_encounters_list).reset_index(drop=True)
    df_encounter_time_stats = pd.concat(df_time_with_infected_list).reset_index(drop=True)
    return df_cust, df_encounter_stats, df_encounter_time_stats
