# Agent-based model for COVID-19 transmission in supermarkets. 
 This code accompanies the paper "COVID-19 transmission in supermarkets using agent-based modelling".

# Install
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
config = {'arrival_rate': 2.55,  # Poisson rate at which customers arrival
           'traversal_time': 0.2,  # mean wait time per node
           'num_hours_open': 14,  # store opening hours
           'infection_proportion': 0.0011,  # proportion of customers that are infectious
         }

# load synthetic data
zone_paths = load_example_paths()
G = load_example_store_graph()

# Create a path generator function which feeds our model with customer paths
path_generator_function, path_generator_args = get_path_generator(G, zone_paths)

# Simulate a day and store results in results
results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)
```

We can examine the results that are stored in `results_dict`.

```python
> print(list(results_dict.keys()))

['num_cust', 'num_S', 'num_I', 'total_time_with_infected', 'num_contacts_per_cust', 'num_cust_w_contact', 'mean_num_cust_in_store', 'max_num_cust_in_store', 'num_contacts', 'shopping_times', 'mean_shopping_time', 'num_waiting_people', 'mean_waiting_time', 'store_open_length', 'df_num_encounters', 'df_time_with_infected', 'total_time_crowded']


```

See below for their description.

Key | Description
------------ | -------------
`num_cust `| Total number of customers
`num_S` | Number of susceptible customers
`num_I` | Number of infected customers
`total_time_with_infected` | Total exposure time
`num_contacts_per_cust` | Number of contacts with infectious customers per susceptible customer
`num_cust_w_contact` | Number of susceptible customers which have at least one contact with an infectious customer
`mean_num_cust_in_store` | Mean number of customers in the store during the simulation
`max_num_cust_in_store` | Maximum number of customers in the store during the simulation
`num_contacts` | Total number of contacts between infectious customers and susceptible customers
`df_num_encounters` | Dataframe which contains the the number of encounters with infectious customers for each node
`shopping_times` | Array that contains the length of all customer shopping trips
`mean_shopping_time` | Mean of the shopping times
`num_waiting_people` | Number of people who are queueing outside at every minute of the simulation (when the number of customers in the store is restricted)
`mean_waiting_time` | Mean time that customers wait before being allowed to enter (when the number of customers in the store is restricted)
`store_open_length` | Length of the store's opening hours
`df_time_with_infected` | Dataframe containing the exposure time per node
`total_time_crowded` | Total time that nodes were crowded (when there are more than `thres` number of customers in a node. Default value of `thres` is 3)
 
 # Questions?
 
 This is work in progress, but feel free to ask any questions by raising an issue or contacting me directly under 
 [fabian.m.ying@gmail.com](fabian.m.ying@gmail.com).