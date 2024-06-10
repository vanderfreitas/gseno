import pandas as pd
import numpy as np

def compute_graph_metrics(graph):
    """
    Compute node-level graph metrics
 
    Args:
        graph (igraph): input graph
 
    Returns:
        pd.DataFrame: Metrics associated to each node
    """

    # Calculate the inverse of edge weights and store them in the 'w_inv' property
    graph.es['w_inv'] = 1.0 / np.array(graph.es['weight'])

    node_labels = graph.vs["label"]
    degrees = graph.degree()
    betweenness = graph.betweenness(vertices=None, directed=False, cutoff=None, weights='w_inv')
    clustering = graph.transitivity_local_undirected()
    strength = graph.strength(weights="weight")
    closeness_w = graph.closeness(vertices=None, mode='all', cutoff=None, weights='w_inv', normalized=True)
    eignv_w = graph.evcent(directed=False, scale=True, weights='w_inv', return_eigenvalue=False)

    graph_metrics_df = pd.DataFrame(
        data = {
            'label': node_labels,
            'degrees': degrees,
            'betweenness': betweenness,
            'clustering': clustering,
            'strength': strength,
            'closeness_w': closeness_w,
            'eignv_w': eignv_w
        }
    )

    # only valid values (NaN are replaced by zero)
    graph_metrics_df.fillna(value=0, inplace=True)

    return graph_metrics_df


def get_nodes_present_in_both_dfs(graph_metrics_df, sorted_nodes_df):
    """
    Keep only nodes present in both dataframes
 
    Args:
        graph_metrics_df (pd.DataFrame): node-level metrics
        sorted_nodes_df (pd.DataFrame): node labels and ranks
 
    Returns:
        graph_metrics_df (pd.DataFrame): node-level metrics for nodes present in both dataframes
        sorted_nodes_df (pd.DataFrame): node labels and ranks for nodes present in both dataframes
    """

    labels = set(pd.merge(graph_metrics_df, sorted_nodes_df, how='inner', on=['label'])['label'])
    graph_metrics_df = graph_metrics_df[graph_metrics_df['label'].isin(labels)]
    sorted_nodes_df = sorted_nodes_df[sorted_nodes_df['label'].isin(labels)]

    return graph_metrics_df, sorted_nodes_df


def get_intersections_between_sets(graph, sorted_nodes_df):
    """
    Get the intersections between nodes and graph metrics over different node ranks
 
    Args:
        graph (igraph): input graph to be compared with the sorted nodes
        sorted_nodes_df (pd.DataFrame): node labels and ranks
 
    Returns:
        pd.DataFrame: intersections between nodes and graph metrics over different node ranks
    """

    # Compute node-level metrics
    graph_metrics_df = compute_graph_metrics(graph)

    n_metrics = len(graph_metrics_df.columns)-1 # number of metrics
    len_results = sorted_nodes_df['rank'].nunique() # number of unique ranks

    # numpy array of the results
    results = np.zeros((len_results,n_metrics))
    results_i = 0
    results_j = 0

    # Select the columns from the matrices
    for m_col in graph_metrics_df.columns[1:]:

        # Sort the corresponding metric in descending order
        graph_metrics_df = graph_metrics_df.sort_values(by=[m_col], ascending=False) 
        metrics_index = 0

        # check every single unique rank and compute the intersections of its corresponding nodes
        # to nodes sorted by the graph metric, considering the same corresponding positions
        for rank in sorted_nodes_df['rank'].unique(): # unique returns the unique values in order of appearance.

            # find all nodes with the same rank        
            node_set_for_a_given_rank = set(sorted_nodes_df[sorted_nodes_df['rank']==rank]['label'])
            # Update the upper index for the nodes
            metrics_index += len(node_set_for_a_given_rank)

            # Update the sets, containing all nodes until the given metrics_index
            metrics_set = set(graph_metrics_df.iloc[:metrics_index]['label'])
            node_set = set(sorted_nodes_df.iloc[:metrics_index]['label'])

            # Compute and store the intersection of sets till metrics_index
            intersection = len(metrics_set.intersection(node_set)) / len(metrics_set)
            results[results_i,results_j] = intersection

            results_i += 1

        results_j += 1
        results_i = 0

    # Create a dataframe with the results and return it
    results_df = pd.DataFrame(
        data=results,
        columns=graph_metrics_df.columns[1:],
        index=sorted_nodes_df['rank'].unique()
    ) 
    return results_df


def get_accumulated_metrics_between_sets(graph, sorted_nodes_df):
    """
    Accumulate the metrics of nodes according to their rank
 
    Args:
        graph (igraph): input graph to be compared with the sorted nodes
        sorted_nodes_df (pd.DataFrame): node labels and ranks
 
    Returns:
        pd.DataFrame: Accumulated metrics of nodes according to their rank
    """

    # Compute node-level metrics
    graph_metrics_df = compute_graph_metrics(graph)

    n_metrics = len(graph_metrics_df.columns)-1 # number of metrics
    len_results = sorted_nodes_df['rank'].nunique() # number of unique ranks

    # numpy array of the results
    results = np.zeros((len_results,n_metrics))
    results_i = 0

    nodes_considered_so_far = set()

    # check every single unique rank and compute the intersections of its corresponding nodes
    # to nodes sorted by the graph metric, considering the same corresponding positions
    for rank in sorted_nodes_df['rank'].unique(): # unique returns the unique values in order of appearance.

        # find all nodes with the same rank        
        node_set_for_a_given_rank = set(sorted_nodes_df[sorted_nodes_df['rank']==rank]['label'])

        nodes_considered_so_far.update(node_set_for_a_given_rank)

        results_j = 0
        # Select the columns from the matrices
        for m_col in graph_metrics_df.columns[1:]:
            results[results_i,results_j] = graph_metrics_df[graph_metrics_df['label'].isin(nodes_considered_so_far)][m_col].sum()
            results_j += 1

        results_i += 1

    # Create a dataframe with the results and return it
    results_df = pd.DataFrame(
        data=results,
        columns=graph_metrics_df.columns[1:],
        index=sorted_nodes_df['rank'].unique()
    )

    # normalize the result to be within [0,1]
    for m_col in results_df.columns: 
        results_df[m_col] = results_df[m_col]  / results_df[m_col].abs().max() 
    
    return results_df