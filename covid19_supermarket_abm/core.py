import logging
import random
from typing import List, Optional

import networkx as nx
import numpy as np
import simpy


class Store(object):
    """Store object that captures the state of the store"""

    def __init__(self, env: simpy.Environment, G: nx.Graph, max_customers_in_store: Optional[int] = None,
                 logging_enabled: bool = True, ):
        """

        :param env: Simpy environment on which the simulation runs
        :param G: Store graph
        :param logging_enabled: Toggle to True to log all simulation outputs
        :param max_customers_in_store: Maximum number of customers in the store
        """
        self.G = G.copy()
        self.customers_at_nodes = {node: [] for node in self.G}
        self.infected_customers_at_nodes = {node: [] for node in self.G}
        self.customers = []
        self.infected_customers = []
        self.env = env
        self.logging_enabled = logging_enabled
        self.number_encounters_with_infected = {}
        self.number_encounters_per_node = {node: 0 for node in self.G}
        self.arrival_times = {}
        self.exit_times = {}
        self.shopping_times = {}
        self.waiting_times = {}
        self.customers_next_zone = {}  # maps customer to the next zone that it wants to go
        self.is_open = True
        self.is_closed_event = self.env.event()
        self.time_with_infected = {}
        self.time_with_infected_per_node = {node: 0 for node in self.G}
        self.node_arrival_time_stamp = {}
        self.num_customers_waiting_outside = 0
        self.total_time_crowded = 0
        self.crowded_thres = 4
        self.node_is_crowded_since = {node: None for node in self.G}  # is None if not crowded, else it's the start time

        # Parameters
        self.node_capacity = np.inf
        self.with_node_capacity = False
        if max_customers_in_store is None:
            self.max_customers_in_store = np.inf
        else:
            self.max_customers_in_store = int(max_customers_in_store)
        self.counter = simpy.Resource(self.env, capacity=self.max_customers_in_store)

        # For profiling purposes
        self.profiling = {'move_customer': 0,
                          'infect_other_customers_at_node': 0,
                          'get_infected_by_other_customers_at_node': 0,
                          '_customer_wait': 0,
                          '_customer_arrival': 0,
                          '_customer_departure': 0
                          }

        # Stats recording
        self.stats = {}

    def open_store(self):
        assert len(self.customers) == 0, "Customers are already in the store before the store is open"
        self.is_open = True

    def close_store(self):
        self.log(f'Store is closing. There are {self.number_customers_in_store()} left in the store. ' +
                 f'({self.num_customers_waiting_outside} are waiting outisde')
        self.is_open = False
        self.is_closed_event.succeed()

    def enable_node_capacity(self, node_capacity: int = 2):
        self.with_node_capacity = True
        self.node_capacity = node_capacity

    def number_customers_in_store(self):
        return sum([len(cus) for cus in list(self.customers_at_nodes.values())])

    def move_customer(self, customer_id: int, infected: bool, start: int, end: int) -> bool:
        if self.check_valid_move(start, end):
            if start == end:  # start == end
                self._customer_wait(customer_id, start, infected)
                self.log(f'Customer {customer_id} stays at present location to buy something.')
                has_moved = True
            elif self.with_node_capacity and len(self.customers_at_nodes[end]) >= self.node_capacity \
                    and start not in [self.customers_next_zone[cust] for cust in self.customers_at_nodes[end]]:
                # Wait if next node is occupied and doesn't work.
                self.log(f'Customer {customer_id} is waiting at {start}, ' +
                         f'since the next node {end} is full. [{self.customers_at_nodes[end]}]')
                self._customer_wait(customer_id, start, infected)
                has_moved = False
            else:
                self.log(f'Customer {customer_id} is moving from {start} to {end}.')
                self._customer_departure(customer_id, start, infected)
                self._customer_arrival(customer_id, end, infected)
                has_moved = True
        else:
            raise ValueError(f'{start} -> {end} is not a valid transition in the graph!')
        return has_moved

    def check_valid_move(self, start: int, end: int):
        return self.G.has_edge(start, end) or start == end

    def add_customer(self, customer_id: int, start_node: int, infected: bool, wait: float):
        self.log(f'New customer {customer_id} arrives at the store. ' +
                 f'({infected * "infected"}{(not infected) * "susceptible"})')
        self.arrival_times[customer_id] = self.env.now
        self.waiting_times[customer_id] = wait
        self.customers.append(customer_id)
        if not infected:
            # Increase counter
            self.number_encounters_with_infected[customer_id] = 0
            self.time_with_infected[customer_id] = 0
        else:
            self.infected_customers.append(customer_id)
        self._customer_arrival(customer_id, start_node, infected)

    def infect_other_customers_at_node(self, customer_id: int, node: int):
        other_suspectible_customers = [other_customer for other_customer in self.customers_at_nodes[node] if
                                       other_customer not in self.infected_customers_at_nodes[node]]
        if len(other_suspectible_customers) > 0:
            self.log(
                f'Infected customer {customer_id} arrived in {node} and' +
                f' met {len(other_suspectible_customers)} customers')
        for other_customer in other_suspectible_customers:
            self.number_encounters_with_infected[other_customer] += 1
            self.number_encounters_per_node[node] += 1

    def get_infected_by_other_customers_at_node(self, customer_id: int, node: int):
        num_infected_here = len(self.infected_customers_at_nodes[node])

        # Track number of infected customers
        if num_infected_here > 0:
            self.log(
                f'Customer {customer_id} is in at zone {node} with {num_infected_here} infected people.' +
                f' ({self.infected_customers_at_nodes[node]})')
            self.number_encounters_with_infected[customer_id] += num_infected_here
            self.number_encounters_per_node[node] += num_infected_here

    def _customer_arrival(self, customer_id: int, node: int, infected: bool):
        """Process a customer arriving at a node."""
        self.customers_at_nodes[node].append(customer_id)
        self.node_arrival_time_stamp[customer_id] = self.env.now
        if infected:
            self.infected_customers_at_nodes[node].append(customer_id)
            self.infect_other_customers_at_node(customer_id, node)
        else:
            self.get_infected_by_other_customers_at_node(customer_id, node)
        num_cust_at_node = len(self.customers_at_nodes[node])
        if num_cust_at_node >= self.crowded_thres and self.node_is_crowded_since[node] is None:
            self.log(f'Node {node} has become crowded with {num_cust_at_node} customers here.')
            self.node_is_crowded_since[node] = self.env.now

    def _customer_wait(self, customer_id: int, node: int, infected: bool):
        if infected:
            self.infect_other_customers_at_node(customer_id, node)
        else:
            self.get_infected_by_other_customers_at_node(customer_id, node)

    def _customer_departure(self, customer_id: int, node: int, infected: bool):
        """Process a customer departing from a node."""
        self.customers_at_nodes[node].remove(customer_id)
        if infected:
            self.infected_customers_at_nodes[node].remove(customer_id)
            s_customers = self.get_susceptible_customers_at_node(node)
            for s_cust in s_customers:
                dt_with_infected = self.env.now - max(self.node_arrival_time_stamp[s_cust],
                                                      self.node_arrival_time_stamp[customer_id])
                self.time_with_infected[s_cust] += dt_with_infected
                self.time_with_infected_per_node[node] += dt_with_infected
        else:
            i_customers = self.infected_customers_at_nodes[node]
            for i_cust in i_customers:
                dt_with_infected = self.env.now - max(self.node_arrival_time_stamp[i_cust],
                                                      self.node_arrival_time_stamp[customer_id])
                self.time_with_infected[customer_id] += dt_with_infected
                self.time_with_infected_per_node[node] += dt_with_infected

        num_cust_at_node = len(self.customers_at_nodes[node])
        if self.node_is_crowded_since[node] is not None and num_cust_at_node < self.crowded_thres:
            # Node is no longer crowded
            total_time_crowded_at_node = self.env.now - self.node_is_crowded_since[node]
            self.total_time_crowded += total_time_crowded_at_node
            self.log(
                f'Node {node} is no longer crowded ({num_cust_at_node} customers here. ' +
                f'Total time crowded: {total_time_crowded_at_node:.2f}')
            self.node_is_crowded_since[node] = None

    def get_susceptible_customers_at_node(self, node):
        return [c for c in self.customers_at_nodes[node] if c not in self.infected_customers_at_nodes[node]]

    def remove_customer(self, customer_id: int, last_position: int, infected: bool):
        """Remove customer at exit."""
        self._customer_departure(customer_id, last_position, infected)
        self.exit_times[customer_id] = self.env.now
        self.node_arrival_time_stamp[customer_id] = self.env.now
        self.shopping_times[customer_id] = self.exit_times[customer_id] - self.arrival_times[customer_id]
        self.log(f'Customer {customer_id} left the store.')

    def now(self):
        return f'{self.env.now:.4f}'

    def log(self, string: str):
        if self.logging_enabled:
            logging.info(f'[Time: {self.now()}] ' + string)


def customer(env: simpy.Environment, customer_id: int, infected: bool, store: Store, path: List[int],
             traversal_time: float, thres: int = 50):
    """
    Simpy process simulating a single customer

    :param env: Simpy environment on which the simulation runs
    :param customer_id: ID of customer
    :param infected: True if infected
    :param store: Store object
    :param path: Assigned customer shopping path
    :param traversal_time: Mean time before moving to the next node in path (also called waiting time)
    :param thres: Threshold length of queue outside. If queue exceeds threshold, customer does not enter
    the queue and leaves.
    """

    arrive = env.now

    if store.num_customers_waiting_outside > thres:
        store.log(f'Customer {customer_id} does not queue up, since we have over {thres} customers waiting outside ' +
                  f'({store.num_customers_waiting_outside})')
        return
    else:
        store.num_customers_waiting_outside += 1

    with store.counter.request() as my_turn_to_enter:
        result = yield my_turn_to_enter | store.is_closed_event
        store.num_customers_waiting_outside -= 1
        wait = env.now - arrive

        if my_turn_to_enter not in result:
            store.log(f'Customer {customer_id} leaves the queue after waiting {wait:.2f} min, as shop is closed')
            return

        if my_turn_to_enter in result:
            store.log(f'Customer {customer_id} enters the shop after waiting {wait :.2f} min.')
            start_node = path[0]
            store.add_customer(customer_id, start_node, infected, wait)
            for start, end in zip(path[:-1], path[1:]):
                store.customers_next_zone[customer_id] = end
                has_moved = False
                while not has_moved:  # If it hasn't moved, wait a bit
                    yield env.timeout(random.expovariate(1 / traversal_time))
                    has_moved = store.move_customer(customer_id, infected, start, end)
            yield env.timeout(random.expovariate(1 / traversal_time))  # wait before leaving the store
            store.remove_customer(customer_id, path[-1], infected)


def _stats_recorder(store: Store):
    store.stats['num_customers_in_store'] = {}
    env = store.env
    while store.is_open or store.number_customers_in_store() > 0:
        store.stats['num_customers_in_store'][env.now] = store.number_customers_in_store()
        yield env.timeout(10)


def _customer_arrivals(env: simpy.Environment, store: Store, path_generator, config: dict):
    """Process that creates all customers."""
    arrival_rate = config['arrival_rate']
    num_hours_open = config['num_hours_open']
    infection_proportion = config['infection_proportion']
    traversal_time = config['traversal_time']
    customer_id = 0
    store.open_store()
    yield env.timeout(random.expovariate(arrival_rate))
    while env.now < num_hours_open * 60:
        infected = np.random.rand() < infection_proportion
        path = path_generator.__next__()
        env.process(customer(env, customer_id, infected, store, path, traversal_time))
        customer_id += 1
        yield env.timeout(random.expovariate(arrival_rate))
    store.close_store()


def _sanity_checks(store: Store, verbose: bool = False):
    infectious_contacts_list = [i for i in store.number_encounters_with_infected.values() if i != 0]
    num_susceptible = len(store.number_encounters_with_infected)
    num_infected = len(store.infected_customers)
    num_cust = len(store.customers)
    assert sum(infectious_contacts_list) == sum(store.number_encounters_per_node.values()), \
        "Number of infectious contacts doesn't add up"
    assert num_infected + num_susceptible == num_cust, \
        "Number of infected and susceptible customers doesn't add up to total number of customers"

    customers_at_nodes = [len(val) for val in store.infected_customers_at_nodes.values()]
    assert max(customers_at_nodes) == 0, \
        f"{sum(customers_at_nodes)} customers have not left the store. {store.infected_customers_at_nodes}"
    assert max([len(val) for val in store.customers_at_nodes.values()]) == 0
    assert set(store.waiting_times.keys()) == set(store.customers), \
        'Some customers are not recorded in waiting times (or vice versa)'
    assert all([val >= 0 for val in store.waiting_times.values()]), \
        'Some waiting times are negative!'
    actual_max_customer_in_store = max(store.stats['num_customers_in_store'].values())
    assert actual_max_customer_in_store <= store.max_customers_in_store, \
        f'Somehow more people were in the store than allowed ' + \
        f'(Allowed: {store.max_customers_in_store} | Actual: {actual_max_customer_in_store})'

    assert store.num_customers_waiting_outside == 0, \
        f"Somehow, there are still {store.num_customers_waiting_outside} people waiting outside"
    if verbose:
        print('Sanity checks passed!')
