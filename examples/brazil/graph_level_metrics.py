
from igraph import Graph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import rc

# Open the GraphML file and create a Graph object from it
graph_aerial = Graph.Read_GraphML("data/networks/weighted_graph_aerial.GraphML")
graph_fluvial = Graph.Read_GraphML("data/networks/weighted_graph_fluvial.GraphML")
graph_terrestrial = Graph.Read_GraphML("data/networks/weighted_graph_terrestrial.GraphML")
graph_all_modes = Graph.Read_GraphML("data/networks/weighted_graph_all_modes.GraphML")

def get_average_degree(G):
    degrees = G.degree()
    avg_degree = sum(degrees) / len(degrees)
    return avg_degree

def coefficient_of_heterogeneity(G):
  degrees = G.degree()
  acc = 0
  for k in degrees:
    acc = acc + k**2
  avg = acc/len(degrees)
  het = avg/(get_average_degree(G)**2)
  return het

metrics_graph_aerial = {
    'n_nodes': graph_aerial.vcount(),
    'n_edges': graph_aerial.ecount(),
    'avg_degree': np.mean(graph_aerial.degree()),
    'heterogeneity_coeff': coefficient_of_heterogeneity(graph_aerial),
    'density': graph_aerial.density()
}

metrics_graph_fluvial = {
    'n_nodes': graph_fluvial.vcount(),
    'n_edges': graph_fluvial.ecount(),
    'avg_degree': np.mean(graph_fluvial.degree()),
    'heterogeneity_coeff': coefficient_of_heterogeneity(graph_fluvial),
    'density': graph_fluvial.density()
}

metrics_graph_terrestrial = {
    'n_nodes': graph_terrestrial.vcount(),
    'n_edges': graph_terrestrial.ecount(),
    'avg_degree': np.mean(graph_terrestrial.degree()),
    'heterogeneity_coeff': coefficient_of_heterogeneity(graph_terrestrial),
    'density': graph_terrestrial.density()
}

metrics_graph_all_modes = {
    'n_nodes': graph_all_modes.vcount(),
    'n_edges': graph_all_modes.ecount(),
    'avg_degree': np.mean(graph_all_modes.degree()),
    'heterogeneity_coeff': coefficient_of_heterogeneity(graph_all_modes),
    'density': graph_all_modes.density()
}

