import re
import csv
import sys
import argparse
import pandas as pd
import numpy as np
import networkx as nx
#from networkx.algorithms.clique import bron_kerbosch as find_cliques_recursive
#from networkx.algorithms.community.kclique import find_cliques_recursive
from networkx.algorithms.clique import enumerate_all_cliques
from networkx.utils import not_implemented_for
from itertools import combinations
import community
import random
import string


# Create argument parser
parser = argparse.ArgumentParser(description='Create a co-occurrence network of named entities mentioned in annotated texts.')

# Add arguments
parser.add_argument('conll', type=str, help='The path to the CoNLL-U formatted file containing the text data.')
parser.add_argument('node_attr', type=str, help='The path to the TSV file containing node attributes.')
parser.add_argument('edge_attr', type=str, help='The path to the TSV file containing edge attributes.')
parser.add_argument('NE_to_remove', type=str,  help='The path to the TXT file of named entities to remove from the graph')
args = parser.parse_args()

# Step 1: Read in the data and extract relevant information
text_dict = {}
#pos_list = ['PN', 'TN', 'DN']
pos_list = ['PN']

# Initialize graph
G = nx.Graph()

with open(args.conll, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        # Check if line indicates a new text
        if line.startswith('#new'):
            # Extract the text ID
            current_text = re.findall(r'P\d{6}', line)[0]
            # Add a new entry to the text dictionary
            text_dict[current_text] = set()
        else:
            # Extract the relevant fields from the CoNLL-U formatted line
            fields = line.split('\t')
            # Check if the field indicates a proper noun, temporal noun or divine name
            if len(fields) > 3 and fields[3] in pos_list:
                # Add the name to the set of names in the current text, along with its POS
                text_dict[current_text].add((fields[2], fields[3]))

                # Check if the name is already in the graph
                if fields[2] not in G.nodes():
                    # Add the name as a node to the graph and set its attributes
                    G.add_node(fields[2], POS=fields[3])

# Step 2: Create graph
for text in text_dict:
    name_list = [name for name, pos in text_dict[text]]
    for i, name1 in enumerate(name_list):
        for name2 in name_list[i + 1:]:
            if G.has_edge(name1, name2):
                # Update the label of the existing edge
                edge_attrs = G.get_edge_data(name1, name2)
                label_list = edge_attrs['label'].split(',')
                if text not in label_list:
                    label_list.append(text)
                label = ','.join(label_list)
                weight = len(label_list)
                G[name1][name2]['label'] = label
                G[name1][name2]['weight'] = weight
            else:
                # Create a new edge
                G.add_edge(name1, name2, weight=1, label=text)

# Step 3: Read in attributes and add to nodes
attributes_df = pd.read_csv(args.node_attr, delimiter='\t')
for index, row in attributes_df.iterrows():
    # Extract the node ID and attributes from the current row
    node_id = row['name']
    attributes_dict = dict(row.drop(labels='name'))
    # Replace NaN values with empty strings
    attributes_dict = {k: '' if pd.isna(v) else v for k, v in attributes_dict.items()}
    # Check if the node ID is in the graph
    if node_id in G.nodes():
        # Update the node attributes
        G.nodes[node_id].update(attributes_dict)


# Step 5: Read in edge attributes and add to edges
edge_attributes_df = pd.read_csv(args.edge_attr, delimiter='\t')

# Create a nested dictionary to store attribute values for each unique 'no_cdli' value
edge_attr_dict = {}
periods_set = set()
for _, row in edge_attributes_df.iterrows():
    no_cdli = row['no_cdli']
    period = row['period']
    periods_set.add(period)
    if no_cdli not in edge_attr_dict:
        edge_attr_dict[no_cdli] = {}
    edge_attr_dict[no_cdli][period] = '1'

periods_list = sorted(list(periods_set))

# Iterate over each edge in the graph
for name1, name2, edge_attrs in G.edges(data=True):
    # Initialize period attributes to 0
    for period in periods_list:
        edge_attrs[period] = '0'
    
    # Look up the corresponding attributes from the nested dictionary
    no_cdli = edge_attrs['label']
    if no_cdli in edge_attr_dict:
        for period, period_indicator in edge_attr_dict[no_cdli].items():
            edge_attrs[period] = period_indicator

# Define period ranges
early_range = range(0, 23)
middle_range = range(22, 26)
late_range = range(25, 34)

# Iterate over each edge in the graph
for name1, name2, edge_attrs in G.edges(data=True):
    # Initialize period attributes to 0
    edge_attrs['period:early'] = '0'
    edge_attrs['period:middle'] = '0'
    edge_attrs['period:late'] = '0'
    edge_attrs['period:Ur III'] = '0'
    
    # Look up the corresponding attributes from the nested dictionary
    no_cdli = edge_attrs['label']
    if no_cdli in edge_attr_dict:
        for period, period_indicator in edge_attr_dict[no_cdli].items():
            if period.startswith('40'):
                edge_attrs['period:Ur III'] = period_indicator
            else:
                try:
                    period_number = int(period.split()[0])
                    if period_number in early_range:
                        edge_attrs['period:early'] = period_indicator
                    elif period_number in middle_range:
                        edge_attrs['period:middle'] = period_indicator
                    elif period_number in late_range:
                        edge_attrs['period:late'] = period_indicator
                except ValueError:
                    pass


# Step 6: Remove names from the graph

with open(args.NE_to_remove, "r") as file:
    NEs = [line.strip() for line in file]

G.remove_nodes_from(NEs)

#Step 7: Do some pre-computations on the graph

# Calculate degree and weighted degree of nodes
degree_dict = dict(G.degree())
weighted_degree_dict = dict(G.degree(weight='weight'))

# Assign degree and weighted degree as node attributes
nx.set_node_attributes(G, degree_dict, 'degree')
nx.set_node_attributes(G, weighted_degree_dict, 'weighted_degree')

# Calculate modularity of the graph using the Louvain algorithm
from community import community_louvain
partition = community_louvain.best_partition(G)
modularity = community_louvain.modularity(partition, G)

# Compute k-plex communities
def kplex_communities(G, k):
    """Find all k-plex communities in a graph"""
    cliques = list(nx.find_cliques_recursive(G))
    kplexes = set()
    for clique in cliques:
        subgraph = G.subgraph(clique)
        degrees = dict(subgraph.degree(subgraph.nodes()))
        for subset in combinations(clique, k):
            if sum(degrees[node] for node in subset) >= 2*k-len(subset):
                kplexes.add(subset)
    kplexes = [kplex for kplex in kplexes if len(kplex) >= k-1]
    return kplexes

for k in range(1, 4):
    kplex_communities_list = kplex_communities(G, k)
    for i, community in enumerate(kplex_communities_list):
        nodes_in_community = list(community)
        for node in nodes_in_community:
            G.nodes[node][f'kplex_{k}'] = i+1



# Assign node size based on degree
# Get the degree values
degree_dict = dict(G.degree())

# Rescale the degree values to be between 0 and 20
max_size = 20
min_size = 0
max_degree = max(degree_dict.values())
min_degree = min(degree_dict.values())
rescaled_degrees = {}
for node, degree in degree_dict.items():
    rescaled_degree = ((degree - min_degree) / (max_degree - min_degree)) * (max_size - min_size) + min_size
    rescaled_degrees[node] = rescaled_degree

# Set the node sizes to be proportional to the rescaled degree values
node_size = list(rescaled_degrees.values())


# Step 8: Output graphs data to file
nx.write_graphml(G, 'output-graph.graphml')

# Create a list of the period attribute values
periods = ['period:Ur III', 'period:early', 'period:middle', 'period:late']


# Optimize graph creation and saving for each period attributes - (Start)

# Create a dictionary to store the graphs for each period attribute
graphs = {}

# Iterate over each period attribute
for period in periods:
    # Initialize a new graph for the current period
    G_period = nx.Graph()

    # Add nodes and edges with the given period attributes to the new graph
    for name, node_attrs in G.nodes(data=True):
        # Check if the current node has the desired period attributes
        period_attr = node_attrs.get(period)
        if period_attr == '1':
            G_period.add_node(name, **node_attrs)
        
    for name1, name2, edge_attrs in G.edges(data=True):
        # Check if the current edge has the desired period attribute
        period_attr = edge_attrs.get(period)
        if period_attr == '1':
            G_period.add_edge(name1, name2, **edge_attrs)

    # Store the graph for the current period in the dictionary
    graphs[period] = G_period

# Save each graph in the dictionary as a separate GraphML file
for period, graph in graphs.items():
    nx.write_graphml(graph, f'graph_{period}.graphml')

# Optimize graph creation and saving for each period attributes - (End)



# Step 9: Output unmatched ids and names to file
unmatched_ids = []
unmatched_names = []

for node in G.nodes():
    # Check if the node ID is in the attributes dataframe
    if node not in attributes_df['name'].tolist():
        unmatched_ids.append(node)
    # Check if the node name (lowercase) is in the attributes dataframe
    if node not in [x.lower() for x in attributes_df['name'].tolist()]:
        unmatched_names.append(node)

# Write the unmatched IDs and names to separate files
with open('unmatched_ids.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(unmatched_ids))

with open('unmatched_names.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(unmatched_names))
