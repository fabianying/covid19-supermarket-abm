import unittest
import networkx as nx

from covid19_supermarket_abm.path_generators import get_path_generator
from covid19_supermarket_abm.simulator import simulate_one_day, simulate_several_days
from covid19_supermarket_abm.utils.create_synthetic_baskets import get_all_shortest_path_dicts
from covid19_supermarket_abm.utils.load_example_data import load_example_paths, load_example_store_graph


class TestSimulatorEmpiricalPathGeneration(unittest.TestCase):

    def setUp(self) -> None:
        zone_paths = load_example_paths()
        G = load_example_store_graph()
        self.G = G
        self.zone_paths = zone_paths

        self.config = {'arrival_rate': 2.55,
                  'traversal_time': 0.2,
                  'num_hours_open': 14,
                  'infection_proportion': 0.0011}


        path_generator_function, path_generator_args = get_path_generator(path_generation='empirical',
                                                                          zone_paths=zone_paths, G=G)
        self.path_generator_function = path_generator_function
        self.path_generator_args = path_generator_args

    def test_simulate_one_day_empirical(self):
        results_dict = simulate_one_day(self.config, self.G, self.path_generator_function, self.path_generator_args)

    def test_simulate_one_day_empirical_parallel(self):
        df_stats, df_num_encounter_per_node_stats, df_encounter_time_per_node_stats = simulate_several_days(self.config, self.G,
                                                                                                            self.path_generator_function,
                                                                                                            self.path_generator_args,
                                                                                                            num_iterations=5,
                                                                                                            use_parallel=True)
    def test_simulate_one_day_empirical_with_logs(self):
        config = self.config.copy()
        config['logging_enabled'] = True
        results_dict = simulate_one_day(config, self.G, self.path_generator_function, self.path_generator_args)
        self.assertGreater(len(results_dict['logs']), 0)
        print('\n'.join(results_dict['logs']))

class TestSimulatorSyntheticPathGeneration(unittest.TestCase):

    def setUp(self) -> None:
        zone_paths = load_example_paths()
        G = load_example_store_graph()
        self.G = G
        self.zone_paths = zone_paths

        self.config = {'arrival_rate': 2.55,
                       'traversal_time': 0.2,
                       'num_hours_open': 14,
                       'infection_proportion': 0.0011}

        entrance_nodes = [39, 40, 41]
        till_nodes = [33, 34, 35]
        exit_nodes = [42]
        item_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                      27, 28, 29, 30, 31, 32, 36, 37, 38]
        mu = 0.07
        sigma = 0.76
        shortest_path_dict = get_all_shortest_path_dicts(G)
        synthetic_path_generator_args = [mu, sigma, entrance_nodes, till_nodes, exit_nodes, item_nodes,
                                         shortest_path_dict]

        path_generator_function, path_generator_args = get_path_generator(path_generation='synthetic',
                                                                          synthetic_path_generator_args=synthetic_path_generator_args)
        self.path_generator_function = path_generator_function
        self.path_generator_args = path_generator_args

    def test_simulate_one_day_synthetic(self):
        results_dict = simulate_one_day(self.config, self.G, self.path_generator_function, self.path_generator_args)

    def test_simulate_one_day_synthetic_parallel(self):
        df_stats, df_num_encounter_per_node_stats, df_encounter_time_per_node_stats = simulate_several_days(self.config,
                                                                                                            self.G,
                                                                                                            self.path_generator_function,
                                                                                                            self.path_generator_args,
                                                                                                            num_iterations=5,
                                                                                                            use_parallel=True)
