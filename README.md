
![GitHub](https://img.shields.io/github/license/vanderfreitas/gseno)
![GitHub last commit](https://img.shields.io/github/last-commit/vanderfreitas/gseno)
![GitHub stars](https://img.shields.io/github/stars/vanderfreitas/gseno?style=social)
![GitHub downloads](https://img.shields.io/github/downloads/vanderfreitas/gseno/total)


# Contents

- [gseno](#gseno)
- [Installation](#installation)
- [Running example](#running-example)


# gseno

gseno: a Python package with an implementation of two approaches to compare a graph structure with a given ranking of nodes. 

Author: Vander L. S. Freitas

# Installation

To install the gseno package in a Python environment:

```shell
pip install git+https://github.com/vanderfreitas/gseno@main
```

## Requirements

The required packages for gseno are:

+ numpy
+ pandas
+ python-igraph


# Running example

We provide a simple example to illustrate the usage of gseno with the Karate Club graph:

```python
import pandas as pd
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

# First approach: Compute the intersections between sets nodes nodes ordered by
#                 a certain metric and its rank from the input dataframe
results_df = gseno.csn(
    graph=g, 
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
```

To run the example, open the `examples/simple` directory:
```
cd examples/simple
```

and execute the code below in the terminal:

```
python simple_example.py
```

The example in examples/brazil correspond to the code used to generate the results of Rocha et al. (2024).
