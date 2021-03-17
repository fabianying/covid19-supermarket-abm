# Agent-based model for COVID-19 transmission in supermarkets. 
This code accompanies the paper ["Modelling COVID-19 transmission in supermarkets using an agent-based model"](https://arxiv.org/abs/2010.07868).

# Installation

Our package relies mainly on [SimPy](https://simpy.readthedocs.io/en/latest/), which requires Python >= 3.6.

```bash
> pip install covid19-supermarket-abm
```  

# Example

In the example below, we use the example data included in the package to simulate a day in the fictitious store
given the parameters below.

```python
from covid19_supermarket_abm.utils.load_example_data import load_example_store_graph, load_example_paths
from covid19_supermarket_abm.path_generators import get_path_generator
from covid19_supermarket_abm.simulator import simulate_one_day

# Set parameters
config = {'arrival_rate': 2.55,  # Poisson rate at which customers arrive
           'traversal_time': 0.2,  # mean wait time per node
           'num_hours_open': 14,  # store opening hours
           'infection_proportion': 0.0011,  # proportion of customers that are infectious
         }

# load synthetic data
zone_paths = load_example_paths()
G = load_example_store_graph()

# Create a path generator which feeds our model with customer paths
path_generator_function, path_generator_args = get_path_generator(zone_paths=zone_paths, G=G)

# Simulate a day and store results in results
results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)
```

The results from our simulations are stored in `results_dict`.

```python
print(list(results_dict.keys()))
```
Output:
```python
['num_cust', 'num_S', 'num_I', 'total_time_with_infected', 'num_contacts_per_cust', 'num_cust_w_contact', 'mean_num_cust_in_store', 'max_num_cust_in_store', 'num_contacts', 'shopping_times', 'mean_shopping_time', 'num_waiting_people', 'mean_waiting_time', 'store_open_length', 'df_num_encounters', 'df_time_with_infected', 'total_time_crowded', 'exposure_times']
```

See below for their description.

Key | Description
------------ | -------------
`num_cust `| Total number of customers
`num_S` | Number of susceptible customers
`num_I` | Number of infected customers
`total_exposure_time` | Total exposure time
`num_contacts_per_cust` | Number of contacts with infectious customers per susceptible customer
`num_cust_w_contact` | Number of susceptible customers which have at least one contact with an infectious customer
`mean_num_cust_in_store` | Mean number of customers in the store during the simulation
`max_num_cust_in_store` | Maximum number of customers in the store during the simulation
`num_contacts` | Total number of contacts between infectious customers and susceptible customers
`df_num_encounters_per_node` | Dataframe which contains the the number of encounters with infectious customers for each node
`shopping_times` | Array that contains the length of all customer shopping trips
`mean_shopping_time` | Mean of the shopping times
`num_waiting_people` | Number of people who are queueing outside at every minute of the simulation (when the number of customers in the store is restricted)
`mean_waiting_time` | Mean time that customers wait before being allowed to enter (when the number of customers in the store is restricted)
`store_open_length` | Length of the store's opening hours
`df_exposure_time_per_node` | Dataframe containing the exposure time per node
`total_time_crowded` | Total time that nodes were crowded (when there are more than `thres` number of customers in a node. Default value of `thres` is 3)
`exposure_times` | List of exposure times of customers (only recording positive exposure times)
 
 # Getting started
 
 As we can see from the above example, our model requires four inputs. 
  ```python
 results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)
```
 
 These inputs are:

 (1) Simulation configurations: `config`
 
 (2) A store network: `G`
 
 (3) A path generator: `path_generator_function`
 
 (4) Arguments for the path generator: `path_generator_args`
 
 We discuss each of these inputs in the following subsections.
 
 ## Simulation configurations
 
 We input the configuration using a dictionary.
 The following keys are accepted:

 
 ### Mandatory config keys
 
 Config key | Description
------------ | -------------
`arrival_rate`| Rate at which customers arrive to the store (in customers per minute)
`traversal_time`| Mean wait time at each node (in minutes)
`num_hours_open`| Number of hours that the store is open
`infection_proportion`| Proportion of customers that are infected
 
 ### Optional config keys
 
 Config key | Description
------------ | -------------
`max_customers_in_store`| Maximum number of customers allowed in store (Default: `None`, i.e., disabled)
 `with_node_capacity` | Set to `True` to limit the number of customers in each node. (Default: `False`). WARNING: This may cause simulations not to terminate due to gridlocks. 
 `node_capacity` | The number of customers allowed in each node, if  `with_node_capacity` is set to `True`. (Default: `2`)
 `logging_enabled` | Set to `True` to start logging simulations. (Default: `False`). The logs can be accessed in `results_dict['logs']`. Also if sanity checks fail, logs will be saved to file. 
 
 
 ## Store network
 
 We use the [NetworkX](https://networkx.org/documentation/stable/) package to create our store network.
 
 First, we need to specify the (x,y) coordinates of each node. 
 So in a very simple example, we have four nodes, arranged in a square at with coordinates (0,0), (0,1), (1,0), and (1,1).   
 
 ```python
pos = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
``` 
Next, we need to specify the edges in the network; in other words, which nodes are connected to each other.

 ```python
edges = [(0,1), (1,3), (0,2), (2,3)]
```
 
 We create the graph as follows.
 ```python
from covid19_supermarket_abm.utils.create_store_network import create_store_network
G = create_store_network(pos, edges)
```

 To visualize your network, you can use `nx.draw_networkx`:
 ```python
import networkx as nx
nx.draw_networkx(G, pos=pos, node_color='y')
```
 ## Path generator and arguments
 
The path generator is what its name suggests: 
It is a [generator](https://wiki.python.org/moin/Generators) that yields full customer paths.

There are two* path generators implemented in this package.

(1) Empirical path generator

(2) Synthetic path generator

You can also implement your own path generator and pass it.

To use one of the implemented path generators, 
it is often easiest to use the `get_path_generator` function from the `covid19_supermarket_abm.path_generators` module.

```python
from covid19_supermarket_abm.path_generators import get_path_generator
path_generator_function, path_generator_args = get_path_generator(path_generation, **args) 
```

\*There is a [third generator](https://github.com/fabianying/covid19-supermarket-abm/blob/12504eabfad03e2ffe0a6c9aac230d19e24c492a/covid19_supermarket_abm/path_generators.py#L196) implemented, but for most purposes, the first two are likely preferable.

### Empirical path generator 
The empirical path generator takes as input a list of full paths 
(which can be empirical paths or synthetically created paths) and yields random paths from that list.
Note that all paths must be valid paths in the store network or the simulation will fail at runtime.

To use it, simply 
```python
from covid19_supermarket_abm.path_generators import get_path_generator
full_paths = [[0, 1, 3], [0, 2, 3]]  # paths in the store network
path_generator_function, path_generator_args = get_path_generator(path_generation='empirical', full_paths=full_paths) 
```

Alternatively, you can input a list of what we call *zone paths* and the store network `G`.
A zone path is a sequence of nodes that a customer visits, but where consecutive nodes in the sequence need not be adjacent.
In the paper, this sequence represents the item locations of where a customer bought items along with the 
entrance, till and exit node that they visited.
The `get_path_generator` function automatically converts these zone paths to full paths by choosing shortest paths between
consecutive nodes in the zone path.

```python
from covid19_supermarket_abm.path_generators import get_path_generator
zone_paths = [[0, 3], [0, 2, 1], [0, 3, 2]]  # note that consecutive nodes need not be adjacent!
path_generator_function, path_generator_args = get_path_generator(path_generation='empirical', G=G, zone_paths=zone_paths)
```
 
 ### Synthetic path generator
 
 
 The synthetic path generator yields random paths as follows.
 
 (1) First, it samples the size K of the shopping basket using a [log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution)
  random variable with parameter `mu` and `sigma` 
 (the mean and standard deviation of the underlying normal distribution).
 (See [Sorensen et al, 2017](https://www.sciencedirect.com/science/article/abs/pii/S0969698916303186))
 
 (2) Second, it chooses a random entrance node as the first node $v_1$ in the path.
 
 (3) Third, it samples K random item nodes, chosen uniformly at random with replacement from item_nodes, which we denote by
 $v_2, ... v_{K+1}$.
 
 (4) Fourth, it samples a random till node and exit node, which we denote by $v_{K+2}$ and $v_{K+3}$.
 The sequence $v_1, ..., v_{K+3}$ is a node sequence where the customer bought items, along the the entrance, till and exit
 nodes that they visited.
 
 (5) Finally, we convert this sequence to a full path on the network using the shortest paths between consecutive nodes
 in the sequence.
 
 For more information, see the Data section in our [paper](https://arxiv.org/pdf/2010.07868.pdf).
 
 ```python
from covid19_supermarket_abm.path_generators import get_path_generator
from covid19_supermarket_abm.utils.create_synthetic_baskets import get_all_shortest_path_dicts
import networkx as nx
entrance_nodes = [0]
till_nodes = [2]
exit_nodes = [3]
item_nodes = [1]
mu = 0.07
sigma = 0.76
shortest_path_dict = get_all_shortest_path_dicts(G)
synthetic_path_generator_args = [mu, sigma, entrance_nodes, till_nodes, exit_nodes, item_nodes, shortest_path_dict]
path_generator_function, path_generator_args = get_path_generator(path_generation='synthetic',
                                                            synthetic_path_generator_args=synthetic_path_generator_args)
```

 Note that this path generator may be quite slow. In the paper, we first pre-generated paths 100,000 paths 
 and then used the Empirical path generator with the pre-generated paths.  
 
 # Questions?
 
 This is work in progress, but feel free to ask any questions by raising an issue or contacting me directly under 
 [fabian.m.ying@gmail.com](fabian.m.ying@gmail.com).
