
# Required Libraries
import copy
import networkx as nx
import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree

############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    """
    Calculate the total distance of a tour.

    Parameters:
    - distance_matrix (numpy.array): Matrix of distances between cities.
    - city_tour (list): List containing the tour route and its distance.

    Returns:
    - distance (float): Total distance of the tour.
    """
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, time_limit=10, start_time=None, best_c=None,  verbose = True):
    """
    Perform a 2-opt local search to improve the tour.

    Parameters:
    - distance_matrix (numpy.array): Matrix of distances between cities.
    - city_tour (list): List containing the tour route and its distance.
    - recursive_seeding (int): Number of recursive iterations. Default is -1.
    - verbose (bool): Print iterations if True. Default is True.

    Returns:
    - city_list[0] (list): Improved tour route.
    - city_list[1] (float): Improved tour distance.
    """
    # Implementation of 2-opt local search remains unchanged.
    if (recursive_seeding < 0):
        count = -2
    else:
        count = 0
    city_list = copy.deepcopy(city_tour)
    distance  = city_list[1]*2
    iteration = 0
    while (count < recursive_seeding):
        if (verbose == True):
            print('Iteration = ', iteration, 'Distance = ', round(city_list[1], 2))  
        best_route = copy.deepcopy(city_list)
        seed       = copy.deepcopy(city_list)        
        for i in range(0, len(city_list[0]) - 2):
            for j in range(i+1, len(city_list[0]) - 1):
                best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
                best_route[0][-1]    = best_route[0][0]     
                best_route[1]        = distance_calc(distance_matrix, best_route)                    
                if (city_list[1] > best_route[1]):
                    city_list = copy.deepcopy(best_route)         
                best_route = copy.deepcopy(seed)
        count     = count + 1
        iteration = iteration + 1  
        if (distance > city_list[1] and recursive_seeding < 0):
             distance          = city_list[1]
             best_c.put(distance)
             time.sleep(0.1)
             count             = -2
             recursive_seeding = -1
        elif(city_list[1] >= distance and recursive_seeding < 0):
            count              = -1
            recursive_seeding  = -2
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break  
    return city_list[0], city_list[1]

############################################################################

# Function: Christofides Algorithm with improvements
def christofides_algorithm(distance_matrix, time_limit=10, best_c=None,  local_search = True, verbose = True):
    """
    Christofides Algorithm with improvements for the Traveling Salesman Problem (TSP).

    Improvement: Using a more efficient method to find the Minimum-Weight Perfect Matching (MWPM) in the graph.
    The implementation uses the `networkx` library to find the MWPM.

    Improvement: Using a more efficient method to find the Eulerian Path in the graph.
    The implementation uses the `networkx` library to find the Eulerian Path.

    Improvement: Applying a local search to improve the tour.
    The implementation uses the 2-opt local search algorithm to improve the tour.

    Parameters:
    - distance_matrix (numpy.array): Matrix of distances between cities.
    - local_search (bool): Apply local search if True. Default is True.
    - verbose (bool): Print iterations if True. Default is True.

    Returns:
    - route (list): Tour route.
    - distance (float): Tour distance.
    """
    start_time = time.time()
    # Minimum Spanning Tree T
    graph_T = minimum_spanning_tree(distance_matrix)
    graph_T = graph_T.toarray().astype(float)
    
    # Induced Subgraph G
    graph_O = np.array(graph_T, copy = True) 
    count_r = np.count_nonzero(graph_T  > 0, axis = 1)
    count_c = np.count_nonzero(graph_T  > 0, axis = 0)
    degree  = count_r + count_c
    graph_G = np.zeros((graph_O.shape))
    for i in range(0, degree.shape[0]):
        if (degree[i] % 2 != 0):
            graph_G[i,:] = 1
            graph_G[:,i] = 1  
    for i in range(0, degree.shape[0]):
        if (degree[i] % 2 == 0):
            graph_G[i,:] = 0
            graph_G[:,i] = 0
    np.fill_diagonal(graph_G, 0)
    for i in range(0, graph_G.shape[0]):
        for j in range(0, graph_G.shape[1]):
            if (graph_G[i, j] > 0):
                graph_G[i, j] = distance_matrix[i, j] 
    
    # Minimum-Weight Perfect Matching M
    """
    Improvement: Using a more efficient method to find the Minimum-Weight Perfect Matching (MWPM) in the graph.
    The implementation uses the `networkx` library to find the MWPM.
    """
    graph_G_inv = np.array(graph_G, copy = True) 
    graph_G_inv = -graph_G_inv
    G = nx.from_numpy_array(graph_G_inv)
    edges = list(G.edges)
    min_w_pm = nx.algorithms.matching.max_weight_matching(G, maxcardinality = True)
    graph_M = np.zeros((graph_G.shape)) 
    for item in min_w_pm:
        i, j = item
        graph_M[i, j] = distance_matrix[i, j] 
    
    # Eulerian Multigraph H
    graph_H = np.array(graph_T, copy = True) 
    for i in range(0, graph_H.shape[0]):
        for j in range(0, graph_H.shape[1]):
            if (graph_M[i, j] > 0 and graph_T[i, j] == 0):
                graph_H[i, j] = 1 
            elif (graph_M[i, j] > 0 and graph_T[i, j] > 0):
                graph_H[j, i] = 1 
    
    # Eulerian Path
    """
    Improvement: Using a more efficient method to find the Eulerian Path in the graph.
    The implementation uses the `networkx` library to find the Eulerian Path.
    """
    H = nx.from_numpy_array(graph_H)
    if (nx.is_eulerian(H)):
        euler = list(nx.eulerian_path(H))
    else:
        H     = nx.eulerize(H)
        euler = list(nx.eulerian_path(H))
    
    # Shortcutting
    route = []
    for path in euler:
        i, j = path
        if (i not in route):
            route.append(i)
        if (j not in route):
            route.append(j)
    route    = route + [route[0]]
    route    = [item + 1 for item in route]
    distance = distance_calc(distance_matrix, [route, 1])
    seed     = [route, distance]
    
    # Local Search
    """
    Improvement: Applying a local search to improve the tour.
    The implementation uses the 2-opt local search algorithm to improve the tour.
    """
    if (local_search == True):
        route, distance = local_search_2_opt(distance_matrix, seed, recursive_seeding = -1, time_limit=time_limit, start_time=start_time, best_c=best_c,  verbose = verbose)
        best_c.put(distance)
        time.sleep(0.1)
    return route, distance


################################################################################
# Unmodified functions from the original code:
# - distance_calc
# - local_search_2_opt
