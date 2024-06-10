import random
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from igraph import Graph
from scipy.optimize import curve_fit
from matplotlib import rc
#Gerar tabela para verios minimum_casess
def calculate_random_metric_averages(data_df):
    normalized_values = {}
    accumulated_sum_degree = 0
    accumulated_sum_clustering = 0
    accumulated_sum_strength = 0
    accumulated_sum_betweenness = 0
    accumulated_sum_closeness_w = 0
    accumulated_sum_eignv_w = 0
    dates = data_df['date'].unique()
    normalized_values['Date'] = dates.tolist()
    normalized_values['Degree'] = []
    normalized_values['Clustering'] = []
    normalized_values['Strength'] = []
    normalized_values['Weighted Betweenness'] = []
    normalized_values['Weighted Closeness'] = []
    normalized_values['Weighted Eignv'] = []
    geocodes = list(data_df['geocode'])
    soma_degree = data_df['degree'].sum()
    soma_strength = data_df['strength'].sum()
    soma_clustering = data_df['clustering'].sum()
    soma_betweenness = data_df['Weighted_betweenness'].sum()
    soma_closeness_w = data_df['Weighted_closeness'].sum()
    soma_eignv_w = data_df['Weighted_eignv'].sum()
    
    for date in dates:
        cities = data_df[data_df['date'] == date]['geocode'].unique()
        num_cities = len(cities)
        random_cities = []
        random_cities = random.sample(geocodes, num_cities)
        geocodes = list(set(geocodes) - set(random_cities))
        
        selected_data = data_df[(data_df['geocode'].isin(random_cities))]
        
        sum_degree = selected_data['degree'].sum()
        sum_clustering = selected_data['clustering'].sum()
        sum_strength = selected_data['strength'].sum()
        sum_betweenness = selected_data['Weighted_betweenness'].sum()
        sum_closeness = selected_data['Weighted_closeness'].sum()
        sum_eignv = selected_data['Weighted_eignv'].sum()
        
        accumulated_sum_degree += sum_degree
        accumulated_sum_clustering += sum_clustering
        accumulated_sum_strength += sum_strength
        accumulated_sum_betweenness += sum_betweenness
        accumulated_sum_closeness_w += sum_closeness
        accumulated_sum_eignv_w += sum_eignv
        
        normalized_values['Degree'].append(accumulated_sum_degree / soma_degree)
        normalized_values['Clustering'].append(accumulated_sum_clustering / soma_clustering)
        normalized_values['Strength'].append(accumulated_sum_strength / soma_strength)
        normalized_values['Weighted Betweenness'].append(accumulated_sum_betweenness / soma_betweenness)
        normalized_values['Weighted Closeness'].append(accumulated_sum_closeness_w / soma_closeness_w)
        normalized_values['Weighted Eignv'].append(accumulated_sum_eignv_w / soma_eignv_w)
    
    df_normalized = pd.DataFrame(normalized_values)
    return df_normalized

def filter_cases(csv_file, n):
    df = pd.read_csv(
        csv_file,
        encoding='utf-8',
        sep=',',
        usecols=['ibgeID', 'newCases', 'totalCases', 'date'],
        dtype={'ibgeID': int}
    ) 
    filtered_df = df[(df['totalCases'] >= n) & (df['newCases'] >= 1) & (df['ibgeID'] != 0) & (df['ibgeID'] > 1000)]
    filtered_df = filtered_df.drop_duplicates(subset='ibgeID')
    return filtered_df

def logistic_growth(x, a, b, k):
    return a / (1 + b * np.exp(-k * x))
#x = eixo x de datas 
#a =  b=  k =
rc('text', usetex=True)
font = {'family' : 'normal',
         'weight' : 'bold',
         'size'   : 12}

rc('font', **font)
params = {'legend.fontsize': 12}
plt.rcParams.update(params)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif":"Helvetica",
})
def plotGraph(leng, minimum_cases):
    upper_bound = mean_table + 2 * std_table
    lower_bound = mean_table - 2 * std_table

    point_indices = np.linspace(0, len(dates) - 1, num=20)
    date_indices = np.arange(len(dates))
    soma_degree = df['degree'].sum()
    soma_strength = df['strength'].sum()
    soma_clustering = df['clustering'].sum()
    soma_betweenness = df['Weighted_betweenness'].sum()
    soma_closeness = df['Weighted_closeness'].sum()
    soma_eignv = df['Weighted_eignv'].sum()
    degree_interp = np.interp(point_indices, date_indices, metrics_table['Degree Accumulated']) / soma_degree
    clustering_interp = np.interp(point_indices, date_indices, metrics_table['Clustering Accumulated']) / soma_clustering
    strength_interp = np.interp(point_indices, date_indices, metrics_table['Strength Accumulated']) / soma_strength
    betweenness_interp = np.interp(point_indices, date_indices, metrics_table['Betweenness Accumulated']) / soma_betweenness
    closeness_interp = np.interp(point_indices, date_indices, metrics_table['Closeness Accumulated']) / soma_closeness
    eignv_interp = np.interp(point_indices, date_indices, metrics_table['Eignv Accumulated']) / soma_eignv

    # Store the interpolation data in a pandas DataFrame
    interp_data_df = pd.DataFrame({
        'Degree': degree_interp,
        'Weighted Betweenness': betweenness_interp,
        'Clustering': clustering_interp,
        'Strength': strength_interp,
        'Weighted Closeness': closeness_interp,
        'Weighted Eigenvector': eignv_interp
    })

    # Calculate the area under each curve using numerical integration (trapezoidal rule)
    areas = interp_data_df.apply(lambda col: np.trapz(col, dx=1))

    popt_clustering, _ = curve_fit(logistic_growth, point_indices, clustering_interp)
    popt_betweenness, _ = curve_fit(logistic_growth, point_indices, betweenness_interp)

    # Curvas ajustadas usando os parâmetros estimados
    clustering_fit = logistic_growth(point_indices, *popt_clustering)
    betweenness_fit = logistic_growth(point_indices, *popt_betweenness)

    plt.figure(figsize=(12, 8))
    plt.plot(point_indices, degree_interp, 'ro-', label=f'Degree: {areas["Degree"]:.2f}', marker='$k$',color ='SlateBlue')
    plt.plot(point_indices, betweenness_interp, 'b^-', label=f'Weighted Betweenness: {areas["Weighted Betweenness"]:.2f}', marker='$s$',color ='DarkSlateBlue')
    plt.plot(point_indices, clustering_interp, 'gs-', label=f'Clustering: {areas["Clustering"]:.2f}', marker='$b$',color ='RebeccaPurple')
    plt.plot(point_indices, strength_interp, 'yd-', label=f'Strength: {areas["Strength"]:.2f}', marker='$b_w$',color ='MediumOrchid')
    plt.plot(point_indices, closeness_interp, 'mo-', label=f'Weighted Closeness: {areas["Weighted Closeness"]:.2f}', marker='$c$',color ='SteelBlue')
    plt.plot(point_indices, eignv_interp, 'c*-', label=f'Weighted Eigenvector: {areas["Weighted Eigenvector"]:.2f}', marker='$c_w$',color ='DarkTurquoise')

    plt.fill_between(mean_table.index, lower_bound['Degree'], upper_bound['Degree'], color='gray', alpha=0.3)
    plt.fill_between(mean_table.index, lower_bound['Clustering'], upper_bound['Clustering'], color='gray', alpha=0.3)
    plt.fill_between(mean_table.index, lower_bound['Strength'], upper_bound['Strength'], color='gray', alpha=0.3)
    plt.fill_between([], [], [], color='gray', alpha=0.3, label='Development with Random Cases')

    plt.plot(point_indices, clustering_fit, 'g--')
    plt.plot(point_indices, betweenness_fit, 'b--')

    plt.annotate(f'Logistic Fit (Clustering)\nA: {popt_clustering[0]:.2f}, B: {popt_clustering[1]:.2f}, K: {popt_clustering[2]:.2f}',
                 xy=(point_indices[5], clustering_fit[5]), xycoords='data',
                 xytext=(point_indices[8], clustering_fit[6]), textcoords='data',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.annotate(f'Logistic Fit (Weighted Betweenness)\nA: {popt_betweenness[0]:.2f}, B: {popt_betweenness[1]:.2f}, K: {popt_betweenness[2]:.2f}',
                 xy=(point_indices[4], betweenness_fit[4]), xycoords='data',
                 xytext=(point_indices[0], betweenness_fit[5]), textcoords='data',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.legend(fontsize='medium')
    plt.text(0.8, 0.90, f'Number of Cities: {leng}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    visible_indices = np.arange(0, len(metrics_table), step=10)
    visible_dates = metrics_table['date'].iloc[visible_indices]
    plt.xticks(visible_indices, visible_dates, rotation=45)

    plt.xlabel('Date')
    plt.ylabel(f'Normalized accumulated metric')
    plt.tight_layout()
    plt.xlim(0, len(metrics_table) - 1)
    plt.ylim(0, 1)
    plt.savefig(f'Datas/results/Accumulated_metricas_{minimum_cases}cases.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

def graph_plot_minimum_casesXintegral():    


    plt.figure(figsize=(10, 8))
    labels = ['Degrees', 'Weighted Betweenness', 'Clustering', 'Strength', 'Weighted Closeness ', 'Weighted Eigenvector ','Upper rand','Lower rand']
    data = [degrees_avg, betweenness_avg, clustering_avg, strength_avg, closeness_avg, eignv_avg,upper_random_avg,lower_random_avg]
    with open('table_Integral_ACC.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Minimum number of cases'] + labels)
        
        for min_case, values in zip(min_cases, zip(*data)):
            writer.writerow([min_case] + list(values))

import csv 

graph = Graph.Read_GraphML("Datas/networks/grafo_Peso_Geral.GraphML")
geocodes = list(map(int, graph.vs["geocode"]))
degrees = graph.degree()
clustering = graph.transitivity_local_undirected()
strength = graph.strength(weights="weight")
graph.es['w_inv'] = 1.0 / np.array(graph.es['weight'])
weighted_betweenness = graph.betweenness(vertices=None, directed=False, cutoff=None, weights='w_inv')
weighted_closeness = graph.closeness(vertices=None, mode='all', cutoff=None, weights='w_inv', normalized=True)
weighted_eignv = graph.evcent(directed=False, scale=True, weights='w_inv', return_eigenvalue=False)

metrics_df = pd.DataFrame({
    "geocode": geocodes,
    "degree": degrees,
    "clustering": clustering,
    "strength": strength,
    "Weighted_betweenness": weighted_betweenness,
    "Weighted_closeness": weighted_closeness,
    "Weighted_eignv": weighted_eignv
})
degrees_avg = []
betweenness_avg = []
clustering_avg = []
strength_avg = []
closeness_avg = []
eignv_avg = []
upper_random_avg = []
lower_random_avg = []
min_cases = list(range(1, 61))
leng = []

for minimum_cases in min_cases :   
    df = filter_cases("Datas/Pre-processed/cases-brazil-cities-time_2020.csv", minimum_cases)
    leng.append(len(df))
    df = df.merge(metrics_df, left_on='ibgeID', right_on='geocode')
    df_sum = df.groupby('date').sum().reset_index()

    soma_degree = df['degree'].sum()
    soma_strength = df['strength'].sum()
    soma_clustering = df['clustering'].sum()
    soma_betweenness = df['Weighted_betweenness'].sum()
    soma_closeness = df['Weighted_closeness'].sum()
    soma_eignv = df['Weighted_eignv'].sum()
    # Calculate the accumulated values for each metric
    df_sum['Degree Accumulated'] = df_sum['degree'].cumsum() / soma_degree
    df_sum['Clustering Accumulated'] = df_sum['clustering'].cumsum() / soma_clustering
    df_sum['Strength Accumulated'] = df_sum['strength'].cumsum() / soma_strength
    df_sum['Betweenness Accumulated'] = df_sum['Weighted_betweenness'].cumsum() / soma_betweenness
    df_sum['Closeness Accumulated'] = df_sum['Weighted_closeness'].cumsum() / soma_closeness
    df_sum['Eignv Accumulated'] = df_sum['Weighted_eignv'].cumsum() / soma_eignv
    
    df_sum['date'] = pd.to_datetime(df_sum['date'])

    
    df_sum = df_sum.sort_values(by='date')

    intervalos_tempo = df_sum['date'].diff().dt.days
    intervalos_tempo = intervalos_tempo[:len(df_sum['Degree Accumulated'])]
    intervalos_tempo = intervalos_tempo[1:]
    # print(df_sum['date'].head)
    # print("--------------------------------------------")
    # print(intervalos_tempo.head)
    metrics_table = df_sum[['date', 'Degree Accumulated', 'Clustering Accumulated', 'Strength Accumulated',
                    'Betweenness Accumulated', 'Closeness Accumulated', 'Eignv Accumulated']]
    metrics_table = metrics_table.reset_index(drop=True)
    result_list = [calculate_random_metric_averages(df) for _ in range(300)]
    combined_table = pd.concat(result_list)
    mean_table = combined_table.groupby('Date').mean()
    std_table = combined_table.groupby('Date').std()  
    upper_bound = mean_table + 2 * std_table
    lower_bound = mean_table - 2 * std_table
    lower_bound_mean = lower_bound.mean(axis=1)
    upper_bound_mean = upper_bound.mean(axis=1)
    # Calcular a área abaixo da curva do lower_bound_mean
    area_lower_curve_mean = np.trapz(lower_bound_mean.values, dx=intervalos_tempo)
    area_upper_curve_mean = np.trapz(upper_bound_mean.values, dx=intervalos_tempo)
    print(area_upper_curve_mean)
    # Calculate the accumulated area under each curve
    degrees_area = np.trapz(df_sum['Degree Accumulated'], dx=intervalos_tempo)
    betweenness_area = np.trapz(df_sum['Betweenness Accumulated'], dx=intervalos_tempo)
    clustering_area = np.trapz(df_sum['Clustering Accumulated'], dx=intervalos_tempo)
    strength_area = np.trapz(df_sum['Strength Accumulated'], dx=intervalos_tempo)
    closeness_w_area = np.trapz(df_sum['Closeness Accumulated'], dx=intervalos_tempo)
    eignv_w_area = np.trapz(df_sum['Eignv Accumulated'], dx=intervalos_tempo)
    # Append the calculated areas to the respective lists
    degrees_avg.append(degrees_area)
    betweenness_avg.append(betweenness_area)
    clustering_avg.append(clustering_area)
    strength_avg.append(strength_area)
    closeness_avg.append(closeness_w_area)
    eignv_avg.append(eignv_w_area)
    upper_random_avg.append(area_upper_curve_mean)
    lower_random_avg.append(area_lower_curve_mean)
graph_plot_minimum_casesXintegral()

#plotGraph(leng[0],min_cases[0])