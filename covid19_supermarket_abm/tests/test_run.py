from random import seed

import numpy as np
import logging
import sys
root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

from covid19_supermarket_abm.path_generators import get_path_generator
from covid19_supermarket_abm.simulator import simulate_one_day, simulate_several_days
from covid19_supermarket_abm.utils.load_example_data import load_example_store_graph, load_example_paths

seed_val = 17
seed(seed_val)
np.random.seed(seed_val)

config = {'arrival_rate': 2.55,
           'traversal_time': 0.2,
           'num_hours_open': 14,
           'infection_proportion': 0.0011,
           'max_customers_in_store': None}

# load data
zone_paths = load_example_paths()
G = load_example_store_graph()

# Decide how paths are generated; by default we take the empirical paths
path_generator_function, path_generator_args = get_path_generator(G, zone_paths)
results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)
for key, val in results_dict.items():
    print(key)
    print(val)

print(list(results_dict.keys()))

# df_cust, df_encounter_stats, df_encounter_time_stats = simulate_several_days(config, G, path_generator_function,
#                                                                              path_generator_args, num_iterations=4,
#                                                                              use_parallel=True)
# print(df_cust)
# print(df_encounter_stats)
# print(df_encounter_time_stats)
