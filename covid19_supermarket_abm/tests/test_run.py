from random import seed

import numpy as np
import logging
import sys
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)
#
# file_handler = logging.FileHandler('simulation_logs.log')
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(formatter)
#
# logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

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
           'max_customers_in_store': None,
          'logging_enabled': True,
          'raise_test_error': False}

# load data
zone_paths = load_example_paths()
G = load_example_store_graph()

# Decide how paths are generated; by default we take the empirical paths
path_generator_function, path_generator_args = get_path_generator(zone_paths=zone_paths, G=G)
results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)
for key, val in results_dict.items():
    print(key)
    print(val)

print(list(results_dict.keys()))

df_stats, df_num_encounter_per_node_stats, df_encounter_time_per_node_stats = simulate_several_days(config, G, path_generator_function,
                                                                             path_generator_args, num_iterations=50,
                                                                             use_parallel=True)
print(df_stats)
print(df_num_encounter_per_node_stats)
print(df_encounter_time_per_node_stats)

df_stats.to_parquet('test.parquet')

