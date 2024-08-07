import pandas as pd
import igraph as ig
import numpy as np
import gseno

def filter_cases(csv_file, n):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(
        csv_file,
        encoding='utf-8',
        sep=',',
        usecols=['ibgeID', 'newCases', 'totalCases', 'date'],
        dtype={'ibgeID': int}
    )

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    # Filter the DataFrame based on the given conditions
    filtered_df = df[(df['totalCases'] >= n) & (df['newCases'] >= 1) & (df['ibgeID'] > 1000)]
    
    # Remove duplicate rows based on 'ibgeID'
    filtered_df = filtered_df.drop_duplicates(subset='ibgeID', keep='first')
    return filtered_df

def filter_records(df1, df2):
    """
    Returns a DataFrame containing the ibgeIDs present in both input DataFrames.
    """
    # Extract the ibgeIDs from both DataFrames
    ibgeIDs_df1 = set(df1['ibgeID'])
    ibgeIDs_df2 = set(df2['ibgeID'])

    # Find the intersection of both sets of ibgeIDs
    common_ibgeIDs = ibgeIDs_df1.intersection(ibgeIDs_df2)
    
    # Filter df2 to keep only rows with common ibgeIDs
    filtered_df2 = df2[df2['ibgeID'].isin(common_ibgeIDs)]
 
    # Sort the filtered DataFrame by 'date' and remove duplicate rows based on 'ibgeID' and 'date'
    filtered_df2 = (
        filtered_df2
        [['ibgeID', 'date']]
        .sort_values(by='date', ascending=True)
        .drop_duplicates(subset=['ibgeID', 'date'])
    )

    # Return the filtered DataFrame
    return filtered_df2

# Load the mobility Brazil graph
g = ig.Graph.Read_GraphML('weighted_graph_all_modes.GraphML')

# Set vertex labels
g.vs['label'] = np.linspace(1, g.vcount(), g.vcount(), dtype=int).tolist()

# Inverse weights for betweenness calculation
g.es['w_inv'] = 1.0 / np.array(g.es['weight'])

# Calculate various graph metrics
g.vs['betweenness'] = g.betweenness(vertices=None, directed=False, cutoff=None, weights='w_inv')
g.vs['clustering'] = g.transitivity_local_undirected()
g.vs['strength'] = g.strength(weights="weight")
g.vs['closeness'] = g.closeness(vertices=None, mode='all', cutoff=None, weights='w_inv', normalized=True)
g.vs['eigenvector'] = g.evcent(directed=False, scale=True, weights='w_inv', return_eigenvalue=False)

# Create a DataFrame with graph metrics
graph_df = pd.DataFrame({
    'label': g.vs['label'],
    'ibgeID': g.vs["geocode"],
    'betweenness': g.vs['betweenness'],
    'clustering': g.vs['clustering'],
    'strength': g.vs['strength'],
    'closeness': g.vs['closeness'],
    'eigenvector': g.vs['eigenvector']
})

# Filter the cases DataFrame
filter_df = filter_cases("cases-brazil-cities-time_2020.csv", 5)

# Get the shared ibgeIDs between the graph DataFrame and filtered cases DataFrame
shared_cities = filter_records(graph_df, filter_df)
shared_cities = shared_cities.rename(columns={'ibgeID': 'label', 'date': 'rank'})
shared_cities['label'] = shared_cities['label'].astype(float)

shared_cities_set = set(shared_cities['label'])

# Filter vertices to keep only those in shared_cities
vertices_to_keep = []
for v in g.vs:
    geocode = v['geocode']
    if geocode in shared_cities_set:
        vertices_to_keep.append(v.index)
    else:
        print(f"Vertex {v.index} with geocode {geocode} is not in shared_cities")

g.vs['label'] = g.vs['geocode']
subgraph = g.subgraph(vertices_to_keep)

# First approach: Compute the intersections between sets of nodes ordered by a certain metric and its rank from the input DataFrame
results_df = gseno.csn(
    graph=subgraph, 
    sorted_nodes_df=shared_cities   
)

print('csn:')
print(f'{results_df}')

# Second approach: Accumulate the nodes' metrics from the sorted nodes by rank (from the input DataFrame)
# The output for each metric is normalized to be within [0,1]
results_df = gseno.anm(
    graph=subgraph, 
    sorted_nodes_df=shared_cities
)

print('anm:')
print(f'{results_df}')
