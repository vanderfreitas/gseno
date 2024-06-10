
from igraph import Graph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import rc
# Open the GraphML file and create a Graph object from it
graph2 = Graph.Read_GraphML("Datas/networks/grafo_Peso_aerio.GraphML")
graph3 = Graph.Read_GraphML("Datas/networks/grafo_Peso_hidro.GraphML")
graph4 = Graph.Read_GraphML("Datas/networks/grafo_Peso_rodo.GraphML")
graph = Graph.Read_GraphML("Datas/networks/grafo_Peso_Geral.GraphML")
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

metricas_graph2 = {
    'nós': graph2.vcount(),
    'arestas': graph2.ecount(),
    'grau_médio': np.mean(graph2.degree()),
    'coeficiente_de_heterogeneidade': coefficient_of_heterogeneity(graph2),
    'densidade': graph2.density()
}

metricas_graph3 = {
    'nós': graph3.vcount(),
    'arestas': graph3.ecount(),
    'grau_médio': np.mean(graph3.degree()),
    'coeficiente_de_heterogeneidade': coefficient_of_heterogeneity(graph3),
    'densidade': graph3.density()
}
metricas_graph4 = {
    'nós': graph4.vcount(),
    'arestas': graph4.ecount(),
    'grau_médio': np.mean(graph4.degree()),
    'coeficiente_de_heterogeneidade': coefficient_of_heterogeneity(graph4),
    'densidade': graph4.density()
}

metricas_graph = {
    'nós': graph.vcount(),
    'arestas': graph.ecount(),
    'grau_médio': np.mean(graph.degree()),
    'coeficiente_de_heterogeneidade': coefficient_of_heterogeneity(graph),
    'densidade': graph.density()
}

