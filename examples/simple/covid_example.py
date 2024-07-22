import pandas as pd
import igraph as ig
import numpy as np
import gseno

def filter_cases(csv_file, n):
    df = pd.read_csv(
        csv_file,
        encoding='utf-8',
        sep=',',
        usecols=['ibgeID', 'newCases', 'totalCases', 'date'],
        dtype={'ibgeID': int}
    )
    # print(df.shape)
    filtered_df = df[(df['totalCases'] >= n) & (df['newCases'] >= 1) & (df['ibgeID'] > 1000)]
    # print(filtered_df.shape)
    filtered_df = filtered_df.drop_duplicates(subset='ibgeID',keep='first')
    return filtered_df

def filter_records(df1, df2):
    """
    Retorna uma lista contendo os ibgeIDs que estão presentes nos dois DataFrames.
    """
    # Extrair os ibgeIDs dos dois DataFrames
    ibgeIDs_df1 = set(df1['ibgeID'])
    ibgeIDs_df2 = set(df2['ibgeID'])
    # Encontrar a interseção dos dois conjuntos de ibgeIDs
    common_ibgeIDs = ibgeIDs_df1.intersection(ibgeIDs_df2)
    
    # Filtrar df2 para manter apenas as linhas com ibgeIDs comuns
    filtered_df2 = df2[df2['ibgeID'].isin(common_ibgeIDs)]
    
    # Ordenar df2 filtrado pela coluna de data
    filtered_df2 = filtered_df2.sort_values(by='date')
    # Retornar a lista de ibgeIDs comuns, preservando a ordem por data
    return filtered_df2['ibgeID'].unique().tolist()

# Use the mobility Brazi graph

g = ig.Graph.Read_GraphML('weighted_graph_all_modes.GraphML')
g.vs['label'] = np.linspace(1,g.vcount(),g.vcount(),dtype=int).tolist()
g.es['w_inv'] = 1.0 / np.array(g.es['weight'])
g.vs['betweenness'] = g.betweenness(vertices=None, directed=False, cutoff=None, weights='w_inv')
g.vs['clustering'] = g.transitivity_local_undirected()
g.vs['strength'] = g.strength(weights="weight")
g.vs['closeness'] = g.closeness(vertices=None, mode='all', cutoff=None, weights='w_inv', normalized=True)
g.vs['eignv'] = g.evcent(directed=False, scale=True, weights='w_inv', return_eigenvalue=False)


graph_df = pd.DataFrame({
    'label': g.vs['label'],
    'ibgeID': g.vs["geocode"],
    'betweenness': g.vs['betweenness'],
    'clustering': g.vs['clustering'],
    'strength': g.vs['strength'],
    'closeness': g.vs['closeness'],
    'eigenvector': g.vs['eignv']
})
#corrigir função
filter_df = filter_cases("cases-brazil-cities-time_2020.csv", 5)
#retorna uma lista com os ibgeIDs ordenados que estão presentes em ambos os dataFrames
shared_cities = filter_records(graph_df, filter_df)

shared_cities_set = set(shared_cities)

vertices_to_keep = []
for v in g.vs:
    geocode = v['geocode']
    if geocode in shared_cities_set:
        vertices_to_keep.append(v.index)
    else:
        print(f"Vértice {v.index} com geocode {geocode} não está em shared_cities")

# Subgrafo com os vértices filtrados
subgraph = g.subgraph(vertices_to_keep)

# DataFrame with sorted nodes. It must have two columns:
#   label: to name the node to be compared with the graph
#   rank: the rank of the node according to the data (e.g. sorted cities, sorted authors, sorted institutions, sorted concepts, etc.)
sorted_nodes_df = pd.DataFrame(
    data = {
        'rank': shared_cities,
        'label': np.linspace(1,subgraph.vcount(),subgraph.vcount(),dtype=int).tolist()
    }
)

# First approach: Compute the intersections between sets nodes nodes ordered by
#                 a certain metric and its rank from the input dataframe
results_df = gseno.csn(
    graph=subgraph, 
    sorted_nodes_df=sorted_nodes_df
)

print('csn:')
print(f'{results_df}')


# Second approach: Accumulate the nodes' metrics from the sorted nodes by rank (from the input dataframe)
#                  ps: the output for each metric is normalized to be within [0,1]
results_df = gseno.anm(
    graph=g, 
    sorted_nodes_df=sorted_nodes_df
)

print('anm:')
print(f'{results_df}')