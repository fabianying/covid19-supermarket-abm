import os
import pathlib
import json
from typing import Optional, List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from covid19_supermarket_abm.utils_for_paper.load_graph import get_floor_area


def load_data(config_name, param, num_iterations, truncate=False, use_old_data=False):
    config = json.load(open(f'../configs/{config_name}.json'))

    df_custs = []
    multipliers = []
    directory = f'../results/{param}/'
    if use_old_data:
        directory = f'../results/old/{param}/'
    for p in pathlib.Path(directory).iterdir():
        if p.is_file() and p.name.endswith('parquet') and p.name.startswith(config_name + '_'):
            if len(p.name[len(config_name):].split('_')) != 4:
                continue
            num_iterations_file = int(p.name.split('_')[-2])
            multiplier = int(p.name.split('_')[-3]) / 100
            if num_iterations_file != num_iterations:
                continue
            if p.name.endswith('_1.parquet'):
                df_cust = pd.read_parquet(p)
                df_cust['multiplier'] = multiplier
                df_custs.append(df_cust)
                multipliers.append(multiplier)
            elif p.name.endswith('_2.parquet'):
                df_encounter_stats = pd.read_parquet(p)
    multipliers = sorted(multipliers)
    print(f'{config_name}: Found {len(df_custs)} files with multipliers={multipliers}')
    df_cust = pd.concat(df_custs)
    df_cust_grouped = df_cust.groupby('multiplier').mean().reset_index()
    df_cust_grouped_std = df_cust.groupby('multiplier').std().reset_index()

    df_cust_grouped[param] = df_cust_grouped.multiplier * config[param]
    df_cust_grouped_std[param] = df_cust_grouped_std.multiplier * config[param]

    if truncate:
        df_cust_grouped = df_cust_grouped.loc[df_cust_grouped.multiplier <= 1.0].copy()
        df_cust_grouped_std = df_cust_grouped_std.loc[df_cust_grouped_std.multiplier <= 1.0].copy()
    return df_cust_grouped, df_cust_grouped_std


def plot_graph_with_error_bars(config_names: List[str], labels: List[str], df_mean: pd.DataFrame, df_std: pd.DataFrame,
                               param: str, column: str, ax: Optional[plt.axes] = None, colors: Optional[List] = None,
                               num_iterations: Optional[int] = 1, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.figure
    for i, (config_name, label) in enumerate(zip(config_names, labels)):
        df_cust_grouped = df_mean.loc[df_mean['config_name'] == config_name]
        df_cust_grouped_std = df_std.loc[df_std['config_name'] == config_name]
        if colors is None:
            color = f'C{i%10}'
        else:
            color = colors[i]
        ax = df_cust_grouped.plot(param, column, label=label,
                                  ax=ax, color=color)
        x = df_cust_grouped[param]
        mean = df_cust_grouped[column]
        sdt = df_cust_grouped_std[column] / np.sqrt(num_iterations)
        ax.fill_between(x, mean - sdt, mean + sdt, alpha=0.3, facecolor=color)
    if param is 'max_customers_in_store':
        ax.set_xlabel('Max. number of customers allowed in store')
    elif param is 'arrival_rate':
        ax.set_xlabel('Rate of customer arrival (customer/min)')
    elif param is 'mean_num_cust_in_store':
        ax.set_xlabel('Mean number of customers in store')
    elif param is 'mean_num_cust_in_store_per_sqm':
        ax.set_xlabel('Mean number of customers in store per sqm')
    elif param is 'max_num_cust_in_store':
        ax.set_xlabel('Maximum number of customers in store')
    elif param is 'max_num_cust_in_store_per_sqm':
        ax.set_xlabel('Maximum number of customers in store per sqm')

    if column is 'num_infected':
        ax.set_ylabel('Number of infections per day')
    elif column is 'total_time_crowded':
        ax.set_ylabel('Total time (s) that zones are congested')
    else:
        ax.set_ylabel(column)
    ax.legend()
    return fig, ax


def load_all_data(config_names, param, num_iterations):
    results_summary_dir = '../results/summary/'
    if not os.path.isdir(results_summary_dir):
        os.mkdir(results_summary_dir)
    dfs_mean = []
    dfs_std = []
    for config_name in tqdm(config_names):
        df_cust_grouped, df_cust_grouped_std = load_data(config_name, param, num_iterations, truncate=True)
        df_cust_grouped.to_parquet(os.path.join(results_summary_dir, f'{config_name}_{num_iterations}.parquet'))
        df_cust_grouped_std.to_parquet(os.path.join(results_summary_dir, f'{config_name}_{num_iterations}_std.parquet'))
        df_cust_grouped['config_name'] = config_name
        df_cust_grouped_std['config_name'] = config_name
        df_cust_grouped['store_id'] = config_name.split('_')[0]
        df_cust_grouped_std['store_id'] = config_name.split('_')[0]
        dfs_mean.append(df_cust_grouped)
        dfs_std.append(df_cust_grouped_std)
    df_mean = pd.concat(dfs_mean)
    df_std = pd.concat(dfs_std)
    return df_mean, df_std


def add_additional_columns(df_mean, df_std, beta, use_time: bool = True, data_dir: str = '.'):
    if use_time:
        df_mean['num_infected'] = df_mean.total_time_with_infected * beta
        df_std['num_infected'] = df_std.total_time_with_infected * beta
    else:
        df_mean['num_infected'] = df_mean.num_contacts * beta
        df_std['num_infected'] = df_std.num_contacts * beta
    df_mean['num_infected_per_1000cust'] = df_mean.num_infected / df_mean.num_cust * 1000
    df_std['num_infected_per_1000cust'] = df_std.num_infected / df_mean.num_cust * 1000

    store_ids = df_mean.store_id.unique()
    df_mean['mean_num_cust_in_store_per_sqm'] = df_mean.mean_num_cust_in_store
    df_std['mean_num_cust_in_store_per_sqm'] = df_std.mean_num_cust_in_store
    for store_id in store_ids:
        floor_area = get_floor_area(store_id, data_dir)
        df_mean.loc[df_mean.store_id == store_id, 'mean_num_cust_in_store_per_sqm'] /= floor_area
        df_std.loc[df_std.store_id == store_id, 'mean_num_cust_in_store_per_sqm'] /= floor_area
    return df_mean, df_std
