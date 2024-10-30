#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 08:53:14 2024

@author: dpjohnso
"""

import random
import numpy as np
import geopandas as gpd
import networkx as nx
import multiprocessing as mp
from functools import partial
import pickle
import os

# Constants for z-score transformation
MEAN_PM = 16.08
STD_DEV_PM = 5.89

def load_data():
    streets = gpd.read_parquet('PRAS_streets.parquet')
    buildings = gpd.read_parquet('PRAS_buildings.parquet')
    buildings = buildings.to_crs(streets.crs)
    residential = buildings[buildings['SHAPEAREA'] < 6000]
    commercial = buildings[buildings['SHAPEAREA'] > 6000]
    residential_coords = [list(geom.exterior.coords) for geom in residential.geometry]
    commercial_coords = [list(geom.exterior.coords) for geom in commercial.geometry]
    air_quality_data = [gpd.read_parquet(f'PRAS_parquet/week_{i+1}.parquet') for i in range(50)]
    return streets, residential_coords, commercial_coords, air_quality_data

def create_graph(streets):
    G = nx.Graph()
    node_mapping = {}
    node_id = 0

    def process_line_string(line, G, node_mapping):
        nonlocal node_id
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            if coords[i] not in node_mapping:
                node_mapping[coords[i]] = node_id
                node_id += 1
            if coords[i + 1] not in node_mapping:
                node_mapping[coords[i + 1]] = node_id
                node_id += 1
            G.add_edge(node_mapping[coords[i]], node_mapping[coords[i + 1]])

    for _, row in streets.iterrows():
        geom = row.geometry
        if geom.geom_type == 'LineString':
            process_line_string(geom, G, node_mapping)
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                process_line_string(line, G, node_mapping)
    return G, {v: k for k, v in node_mapping.items()}

def closest_node(graph_layout, point):
    closest, min_dist = None, float('inf')
    for node, coord in graph_layout.items():
        dist = np.linalg.norm(np.array(coord) - np.array(point))
        if dist < min_dist:
            closest, min_dist = node, dist
    return closest

def initialize_agents(total_agents, residential_coords, commercial_coords):
    susceptibility_groups = ['very_low', 'low', 'medium', 'high']
    return [{
        'id': i + 1,
        'home': random.choice(residential_coords)[0],
        'work': random.choice(commercial_coords)[0],
        'path': [],
        'position': None,
        'history': [],
        'exposure': [],
        'susceptibility': random.choice(susceptibility_groups),
        'health_impact': 0,
        'current_step': 0  # Initialize current_step
    } for i in range(total_agents)]

def health_risk_calculation(agent, pm_value):
    base_risk = 0.01  # Base health risk without exposure
    susceptibility_factors = {
        'very_low': 0.01,  # Low risk increase per unit above baseline
        'low': 0.03,       # Slightly higher risk increase
        'medium': 0.05,    # Moderate risk increase
        'high': 0.1        # High risk increase for sensitive individuals
    }
    risk_factor = susceptibility_factors[agent['susceptibility']]
    
    # Define risk increases based on studies
    respiratory_mortality_risk = 0.0058  # 0.58% increase per 10 µg/m³ PM10
    hospitalization_rate_risk = 0.08     # 8% increase per 10 µg/m³ PM2.5
    respiratory_disease_prevalence_risk = 0.0207  # 2.07% increase per 10 µg/m³ PM2.5
    overall_mortality_risk = 0.04        # 4% increase per 10 µg/m³ PM2.5
    cardiopulmonary_mortality_risk = 0.06  # 6% increase per 10 µg/m³ PM2.5
    lung_cancer_mortality_risk = 0.08    # 8% increase per 10 µg/m³ PM2.5

    health_impact = base_risk + risk_factor * (
        respiratory_mortality_risk * pm_value +
        hospitalization_rate_risk * pm_value +
        respiratory_disease_prevalence_risk * pm_value +
        overall_mortality_risk * pm_value +
        cardiopulmonary_mortality_risk * pm_value +
        lung_cancer_mortality_risk * pm_value
    )
    
    return health_impact

def get_air_quality_value(air_quality_gdf, point):
    point_geom = gpd.points_from_xy([point[0]], [point[1]])
    for idx, row in air_quality_gdf.iterrows():
        if row.geometry.contains(point_geom[0]):
            z_scored_value = row['layer']
            original_value = (z_scored_value * STD_DEV_PM) + MEAN_PM
            return original_value
    return 0

def simulate_agent(agent, graph_layout, G, residential_coords, commercial_coords, air_quality_data, steps_per_week, steps):
    current_week = 0
    for step in range(steps):
        if agent['current_step'] < len(agent['path']) - 1:
            agent['current_step'] += 1
        else:
            start_node = closest_node(graph_layout, agent['home'])
            end_node = closest_node(graph_layout, agent['work'])
            try:
                agent['path'] = nx.shortest_path(G, source=start_node, target=end_node)
            except nx.NetworkXNoPath:
                print(f"No path between {start_node} and {end_node}. Skipping this agent.")
                return agent  # Skip the agent if no path is found
            agent['current_step'] = 0
        agent['position'] = agent['path'][agent['current_step']]
        agent['history'].append(agent['position'])
        
        pm_value = get_air_quality_value(air_quality_data[current_week], graph_layout[agent['position']])
        agent['exposure'].append(pm_value)
        agent['health_impact'] += health_risk_calculation(agent, pm_value)
        
        if (step + 1) % steps_per_week == 0:
            current_week = (current_week + 1) % 50
            
    return agent

def run_simulation():
    streets, residential_coords, commercial_coords, air_quality_data = load_data()
    G, graph_layout = create_graph(streets)
    total_agents = 5000
    batch_size = 250
    steps_per_week = 7
    steps = steps_per_week * 50
    all_agents = []

    for batch_start in range(0, total_agents, batch_size):
        batch_agents = initialize_agents(batch_size, residential_coords, commercial_coords)
        with mp.Pool(mp.cpu_count()) as pool:
            processed_agents = pool.map(partial(simulate_agent, graph_layout=graph_layout, G=G, residential_coords=residential_coords, commercial_coords=commercial_coords, air_quality_data=air_quality_data, steps_per_week=steps_per_week, steps=steps), batch_agents)
        all_agents.extend(processed_agents)
        with open(f'agents_data_batch_{batch_start//batch_size + 1}.pkl', 'wb') as f:
            pickle.dump(processed_agents, f)

    return all_agents, graph_layout, streets

if __name__ == "__main__":
    agents, graph_layout, streets = run_simulation()
    
    # Load all agents from pickles if necessary
    pickle_directory = "/geode2/home/u020/dpjohnso/BigRed200/PRAS/"  # Replace with your actual path
    all_agents = []
    for filename in os.listdir(pickle_directory):
        if filename.endswith(".pkl"):
            with open(os.path.join(pickle_directory, filename), 'rb') as f:
                agents_batch = pickle.load(f)
                all_agents.extend(agents_batch)

    print(f"Loaded {len(all_agents)} agents from pickled files.")

    # Find the agent with the highest health impact
    max_health_impact = -float('inf')
    max_health_impact_agent = None

    for agent in all_agents:
        if agent['health_impact'] > max_health_impact:
            max_health_impact = agent['health_impact']
            max_health_impact_agent = agent

    if max_health_impact_agent:
        print(f"Agent with highest health impact: ID {max_health_impact_agent['id']} with health impact {max_health_impact_agent['health_impact']}")
    else:
        print("No valid agent data to display.")
