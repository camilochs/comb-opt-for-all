
############################################################################

# Required Libraries
import copy
import networkx as nx
import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree

############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: Improved 2_opt with "Don't Look Bits"
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, time_limit=10, start_time=None, best_c=None,  verbose = True):
    """
    Implements the 2-opt local search optimization with the "Don't Look Bits" heuristic.
    
    Improvement Implemented:
    - The "Don't Look Bits" heuristic is integrated into the 2-opt local search algorithm.
    
    How It Enhances Performance:
    - "Don't Look Bits" keeps track of cities that haven't resulted in improvements.
    - Cities that do not lead to improvements are temporarily marked.
    - This reduces unnecessary comparisons, leading to faster convergence times.
    
    Technique Based On:
    - Based on the "Don't Look Bits" heuristic commonly used in state-of-the-art TSP solvers
      to accelerate local search algorithms like 2-opt.
    """
    city_list = copy.deepcopy(city_tour)
    n_cities = len(city_list[0]) - 1
    dont_look = [0] * n_cities  # Initialize "Don't Look Bits" for each city
    improved = True
    iteration = 0
    while improved:
        improved = False
        if (verbose == True):
            print('Iteration = ', iteration, 'Distance = ', round(city_list[1], 2))  
        for i in range(n_cities):
            if dont_look[i] == 0:
                for j in range(i+2, n_cities):
                    if j == n_cities - 1 and i == 0:
                        continue  # Skip if it's the start and end node
                    delta = (- distance_matrix[city_list[0][i]-1, city_list[0][i+1]-1]
                             - distance_matrix[city_list[0][j]-1, city_list[0][j+1]-1]
                             + distance_matrix[city_list[0][i]-1, city_list[0][j]-1]
                             + distance_matrix[city_list[0][i+1]-1, city_list[0][j+1]-1])
                    if delta < 0:
                        # Perform 2-opt swap
                        new_route = city_list[0][0:i+1] + city_list[0][j:i:-1] + city_list[0][j+1:]
                        city_list[0] = new_route
                        city_list[1] += delta
                        best_c.put(city_list[1])
                        time.sleep(0.1)
                        dont_look[i] = 0  # Reset don't look bit
                        dont_look[j] = 0  # Reset don't look bit
                        improved = True
                        break  # Improvement found, go to next i
                if not improved:
                    dont_look[i] = 1  # Mark city as not improved
        iteration +=1
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break  
    return city_list[0], city_list[1]

############################################################################

# Function: Christofides Algorithm with Improved Local Search
def christofides_algorithm(distance_matrix, time_limit=10, best_c=None, local_search = True, verbose = True):
    """
    Implements the Christofides Algorithm for the TSP with an improved local search.
    
    Improvement Implemented:
    - Uses an enhanced 2-opt local search with the "Don't Look Bits" heuristic.
    
    How It Enhances Performance:
    - The improved local search finds better quality solutions by efficiently exploring
      neighboring tours.
    - Faster convergence is achieved by reducing unnecessary route evaluations.
    
    Technique Based On:
    - Integrates the "Don't Look Bits" heuristic into the 2-opt algorithm, a state-of-the-art
      technique in TSP optimization to accelerate local search methods.
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
    graph_G_inv = np.array(graph_G, copy = True) 
    graph_G_inv = -graph_G_inv
    min_w_pm    = nx.algorithms.matching.max_weight_matching(nx.from_numpy_array(graph_G_inv), maxcardinality = True)
    graph_M     = np.zeros((graph_G.shape)) 
    for item in min_w_pm:
        i, j          = item
        graph_M[i, j] = distance_matrix[i, j] 
    # Eulerian Multigraph H
    graph_H = np.array(graph_T, copy = True) 
    for i in range(0, graph_H.shape[0]):
        for j in range(0, graph_H.shape[1]):
            if (graph_M[i, j] > 0 and graph_T[i, j] == 0):
                graph_H[i, j] = 1 #distance_matrix[i, j]  
            elif (graph_M[i, j] > 0 and graph_T[i, j] > 0):
                graph_H[j, i] = 1 #distance_matrix[i, j]    
    # Eulerian Path
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
    best_c.put(distance)
    time.sleep(0.1)
    seed     = [route, distance]
    if (local_search == True):
        route, distance = local_search_2_opt(distance_matrix, seed, recursive_seeding = -1, time_limit=time_limit, start_time=start_time, best_c=best_c,   verbose = verbose)
        best_c.put(distance)
        time.sleep(0.1)
    return route, distance

############################################################################
       
# Unmodified Functions from the Original Code:
# - distance_calc
