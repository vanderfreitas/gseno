import os
import igraph as ig
import pandas as pd
from sklearn.preprocessing import LabelEncoder

mydir = os.getcwd()

# Armazena caminho do arquivo
path_xlsx = mydir + "/raw_data/Base_de_dados_ligacoes_rodoviarias_e_hidroviarias_2016.xlsx"

# Cria dataset dos dados originais
data = pd.read_excel(path_xlsx, header=[0], index_col=0)

# Codigo do estado (no Brasil no máximo é 53) 
data.COD_UF_A.unique()

# Foram encontrados maiores que 53 no B (80, 87, 92, 88, 82, 81, 91)
data.COD_UF_B.unique()

# Primeiros 5 elementos
data.head()

# Tipos dos dados
data.dtypes

# Retirando as linhas com codigo de municipio acima de 53
data = data[data.COD_UF_A <= 53]
data = data[data.COD_UF_B <= 53]

# Retirar colunas não importantens
data.drop(['COD_UF_A', 'UF_A', 'COD_UF_B', 'UF_B'], axis=1, inplace=True)

# Verificando se existem valores nulos
data.isnull().values.any()

# Quantos valores nulos
data.isnull().sum().sum()

# Onde estão os valores nulos e sua quantidade
data['VAR01'].isnull().values.any()
data['VAR01'].isnull().sum()

data['VAR02'].isnull().values.any()
data['VAR02'].isnull().sum()

# Preenchendo espaços nulos
data.VAR01.fillna('Missing', inplace=True)
data.VAR02.fillna('Missing', inplace=True)

# Verificando se ainda existem valores nulos
data.isnull().values.any()

# Transformando Sim/Não em S/N 
data.VAR13 = data.VAR13.astype(str).str[0]

# Transformando dados dos nomes dos municipios (categóricos) em numéricos
label_encoder_var01_var02 = LabelEncoder()
var01_var02 = pd.unique(data[['VAR01', 'VAR02']].values.ravel('K'))
label_encoder_var01_var02.fit(var01_var02)
data.VAR01 = label_encoder_var01_var02.transform(data.VAR01.values)
data.VAR02 = label_encoder_var01_var02.transform(data.VAR02.values)

# Primeiros 5 elementos
data.head()

# Salvando o novo dataset
data.to_excel(mydir + "/raw_data/dataset_transform.xlsx", encoding="utf8")

# Carregando o dataset
data = pd.read_excel(mydir + "/raw_data/dataset_transform.xlsx", header=0)


# Função existe nó
def has_node(graph, name):
    try:
        graph.vs.find(name=name)
    except:
        return False
    return True


# Função insere no
def add_vertex(graph, id, nome, hurb, long, lati):
    graph.add_vertex(id, geocode=id)
    graph.vs.find(geocode=id)['NOMEMUN'] = nome  # Nome do Municipio
    graph.vs.find(geocode=id)['HURB'] = hurb  # Tipo do local
    graph.vs.find(geocode=id)['LONG'] = long  # Longitude
    graph.vs.find(geocode=id)['LATI'] = lati  # Latitude


# Função insere aresta
def add_edge(graph, opt, a, b, id, cost, time, hidro, rodo, veic):
    graph.add_edge(a, b, id=id)  # Id do caminho
    graph.es.find(id=id)['MINCOST'] = cost  # Custo minimo (R$)
    graph.es.find(id=id)['MINTIME'] = time  # Tempo minimo (min)
    if opt == 1:
        graph.es.find(id=id)['FREQHIDRO'] = hidro  # Numero de viagens por meio fluvial
        graph.es.find(id=id)['FREQRODO'] = rodo  # Numero de viagens por meio rodoviário
        graph.es.find(id=id)['FREQVEIC'] = veic  # Numero de viagens por meio informal
        graph.es.find(id=id)['weight'] = hidro + rodo + veic  # Numero de viagens por todos os meios
    if opt == 2:
        graph.es.find(id=id)['weight'] = hidro  # Numero de viagens por meio fluvial
    if opt == 3:
        graph.es.find(id=id)['weight'] = rodo  # Numero de viagens por meio fluvial
    if opt == 4:
        graph.es.find(id=id)['weight'] = hidro + rodo + veic  # Numero de viagens por todos os meios


# Função contrua o grafo
def build_graph(data, directed=False):
    g1 = ig.Graph(directed=directed)  # Grafo completo
    g2 = ig.Graph(directed=directed)  # Grafo fluxo hidro
    g3 = ig.Graph(directed=directed)  # Grafo fluxo rodo
    g4 = ig.Graph(directed=directed)  # Grafo fluxos somados

    for index, row in data.iterrows():
        # Nó de partida
        id_1 = str(row["CODMUNDV_A"])  # Cod do municipio
        # Nó de chegada
        id_2 = str(row["CODMUNDV_B"])  # Cod do municipio

        if not has_node(g1, id_1):
            # Nó de partida g1
            add_vertex(g1, id_1, row['NOMEMUN_A'], row['VAR01'], row['VAR08'], row['VAR09'])
            # Nó de partida g4
            add_vertex(g4, id_1, row['NOMEMUN_A'], row['VAR01'], row['VAR08'], row['VAR09'])
        if not has_node(g1, id_2):
            # Nó de chegada g1
            add_vertex(g1, id_2, row['NOMEMUN_B'], row['VAR02'], row['VAR10'], row['VAR11'])
            # Nó de chegada g4
            add_vertex(g4, id_2, row['NOMEMUN_B'], row['VAR02'], row['VAR10'], row['VAR11'])

        # Aresta g1
        add_edge(g1, 1, id_1, id_2, row['ID'], row['VAR03'], row['VAR04'], row['VAR05'], row['VAR06'], row['VAR12'])
        # Aresta g4
        add_edge(g4, 4, id_1, id_2, row['ID'], row['VAR03'], row['VAR04'], row['VAR05'], row['VAR06'], row['VAR12'])

        # Grafo Hidro
        if row['VAR05'] > 0:
            if not has_node(g2, id_1):
                # Nó de partida g2
                add_vertex(g2, id_1, row['NOMEMUN_A'], row['VAR01'], row['VAR08'], row['VAR09'])
            if not has_node(g2, id_2):
                # Nó de chegada g2
                add_vertex(g2, id_2, row['NOMEMUN_B'], row['VAR02'], row['VAR10'], row['VAR11'])
            # Aresta g2
            add_edge(g2, 2, id_1, id_2, row['ID'], row['VAR03'], row['VAR04'], row['VAR05'], row['VAR06'], row['VAR12'])
        # Grafo Rodo
        if row['VAR06'] > 0:
            if not has_node(g3, id_1):
                # Nó de partida g3
                add_vertex(g3, id_1, row['NOMEMUN_A'], row['VAR01'], row['VAR08'], row['VAR09'])
            if not has_node(g3, id_2):
                # Nó de chegada g3
                add_vertex(g3, id_2, row['NOMEMUN_B'], row['VAR02'], row['VAR10'], row['VAR11'])
            # Aresta g3
            add_edge(g3, 3, id_1, id_2, row['ID'], row['VAR03'], row['VAR04'], row['VAR05'], row['VAR06'], row['VAR12'])

    # Salvar grafo
    g1.write_graphml(mydir + '/networks/grafo_completo.GraphML')
    g2.write_graphml(mydir + '/networks/grafo_peso_hidro.GraphML')
    g3.write_graphml(mydir + '/networks/grafo_peso_rodo.GraphML')
    g4.write_graphml(mydir + '/networks/grafo_peso_somado.GraphML')

    # return g1, g2, g3, g4ß


# Constroi o grafo a partir dos dados modificados
# g1, g2, g3, g4 =
build_graph(data, directed=True)
