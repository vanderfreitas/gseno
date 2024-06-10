import pandas as pd
import numpy as np
import igraph as ig
import gseno

# Use the Zachary Karate club graph
g = ig.Graph.Read_GraphML('zachary.graphml')

# Giving a label and weights (those are necessary for node comparisons and to calculating metrics)
g.vs['label'] = np.linspace(1,g.vcount(),g.vcount(),dtype=int).tolist()
g.es['weight'] = 1

# DataFrame with sorted nodes. It must have two columns:
#   label: to name the node to be compared with the graph
#   rank: the rank of the node according to the data (e.g. sorted cities, sorted authors, sorted institutions, sorted concepts, etc.)
sorted_nodes_df = pd.DataFrame(
    data = {
        'rank': [1,1,1,2,2,3,4,5,6,6,7,7,7,7,8,9,10,10,10,10,10,11,11,12,13,13,14,14,14,14,14,14,14,14], # this list was artificially generated
        'label': np.linspace(1,g.vcount(),g.vcount(),dtype=int).tolist()
    }
)

results_df = gseno.get_intersections_between_sets(
    graph=g, 
    sorted_nodes_df=sorted_nodes_df
)

print(f'{results_df}')