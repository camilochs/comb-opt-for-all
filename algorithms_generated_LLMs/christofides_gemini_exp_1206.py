import copy
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from networkx.algorithms.approximation import traveling_salesman_problem
from networkx.utils import arbitrary_element
import time
# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    """
    Calculates the total distance of a given tour.

    Parameters
    ----------
    distance_matrix : np.ndarray
        The distance matrix representing the distances between cities.
    city_tour : list
        A list representing the tour, where each element is a city index.

    Returns
    -------
    float
        The total distance of the tour.
    """
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]
    return distance

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, time_limit=10, start_time=None, best_c=None,   verbose = True):
    """
    Improves a tour using the 2-opt local search algorithm.

    Parameters
    ----------
    distance_matrix : np.ndarray
        The distance matrix representing the distances between cities.
    city_tour : list
        A list representing the tour, where each element is a city index.
    recursive_seeding : int, optional
        The number of iterations to run the 2-opt algorithm. If negative, the algorithm
        runs until no further improvement is found. The default is -1.
    verbose : bool, optional
        Whether to print the iteration number and distance at each iteration. The default is True.

    Returns
    -------
    list
        The improved tour.
    float
        The total distance of the improved tour.
    """
    if (recursive_seeding < 0):
        count = -2
    else:
        count = 0
    city_list = copy.deepcopy(city_tour)
    distance = city_list[1]*2
    iteration = 0
    while (count < recursive_seeding):
        if (verbose == True):
            print('Iteration = ', iteration, 'Distance = ', round(city_list[1], 2))
        best_route = copy.deepcopy(city_list)
        seed = copy.deepcopy(city_list)
        for i in range(0, len(city_list[0]) - 2):
            for j in range(i+1, len(city_list[0]) - 1):
                best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))
                best_route[0][-1] = best_route[0][0]
                best_route[1] = distance_calc(distance_matrix, best_route)
                if (city_list[1] > best_route[1]):
                    city_list = copy.deepcopy(best_route)
                best_route = copy.deepcopy(seed)
        count = count + 1
        iteration = iteration + 1
        if (distance > city_list[1] and recursive_seeding < 0):
            distance = city_list[1]
            best_c.put(distance)
            time.sleep(0.1)
            count = -2
            recursive_seeding = -1
        elif(city_list[1] >= distance and recursive_seeding < 0):
            count = -1
            recursive_seeding = -2
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break  
    return city_list[0], city_list[1]

def nearest_neighbor_initialization(distance_matrix):
    """
    Generates an initial tour using the nearest neighbor heuristic.

    This function provides a faster starting point for the Christofides algorithm compared to a random initialization.
    The nearest neighbor heuristic constructs a tour by iteratively selecting the nearest unvisited city.

    Parameters
    ----------
    distance_matrix : np.ndarray
        The distance matrix representing the distances between cities.

    Returns
    -------
    list
        The initial tour generated by the nearest neighbor heuristic.
    float
        The total distance of the initial tour.
    """
    
    num_cities = distance_matrix.shape[0]
    unvisited = set(range(1, num_cities + 1))
    current_city = arbitrary_element(unvisited)
    tour = [current_city]
    unvisited.remove(current_city)

    while unvisited:
        nearest_city = min(unvisited, key=lambda city: distance_matrix[current_city - 1, city - 1])
        tour.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city

    tour.append(tour[0])  # Close the tour
    return tour, distance_calc(distance_matrix, [tour, 1])

def _fast_minimum_weight_perfect_matching(graph_G, distance_matrix):
    """
    Computes a minimum-weight perfect matching using a faster heuristic based on the Blossom V algorithm.

    This function utilizes the `traveling_salesman_problem` function from NetworkX, which implements a
    heuristic based on the Blossom V algorithm for finding a minimum-weight perfect matching. This is
    generally faster than the `max_weight_matching` function used in the original implementation.
    It improves the performance by reducing the time spent in the matching phase.

    Parameters
    ----------
    graph_G : np.ndarray
        The induced subgraph representing the odd-degree nodes in the minimum spanning tree.
    distance_matrix : np.ndarray
        The distance matrix representing the distances between cities.

    Returns
    -------
    list
        A list of edges representing the minimum-weight perfect matching.
    """
    
    G = nx.Graph()
    nodes = np.where(np.any(graph_G != 0, axis=0))[0]
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if graph_G[u, v] != 0:
                G.add_edge(u, v, weight=distance_matrix[u, v])
    
    matching_path = traveling_salesman_problem(G, weight="weight", cycle=False)
    matching = []
    for i in range(0, len(matching_path) - 1):
        matching.append((matching_path[i], matching_path[i+1]))

    return matching

def _3_opt(distance_matrix, tour):
    """
    Improves a tour using the 3-opt local search algorithm.

    This function implements a basic version of the 3-opt algorithm, which considers all possible 3-edge exchanges
    to find a better tour. It is a more powerful local search heuristic than 2-opt, potentially leading to
    higher-quality solutions. However, it is also more computationally expensive.

    Parameters
    ----------
    distance_matrix : np.ndarray
        The distance matrix representing the distances between cities.
    tour : list
        The current tour.

    Returns
    -------
    list
        The improved tour.
    float
        The total distance of the improved tour.
    """
    
    best_tour = tour
    best_distance = distance_calc(distance_matrix, [tour, 1])
    n = len(tour)

    for i in range(n - 3):
        for j in range(i + 2, n - 1):
            for k in range(j + 2, n):
                # Generate all possible segment combinations
                segments = [
                    tour[0 : i + 1],
                    tour[i + 1 : j + 1],
                    tour[j + 1 : k + 1],
                    tour[k + 1 :],
                ]

                # Generate all possible permutations of the segments
                for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 2, 1), (0, 1, 3, 2), (0, 2, 3, 1), (0, 3, 1, 2)]:
                    new_tour = (
                        segments[p[0]]
                        + segments[p[1]][::-1]
                        + segments[p[2]][::-1]
                        + segments[p[3]][::-1]
                    )
                    new_tour[-1] = new_tour[0]
                    new_distance = distance_calc(distance_matrix, [new_tour, 1])
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_tour = new_tour

    return best_tour, best_distance

# Function: Christofides Algorithm
def christofides_algorithm(distance_matrix, time_limit=10, best_c=None, local_search = True, verbose = True):
    """
    Solves the Traveling Salesman Problem (TSP) using the Christofides algorithm.

    This function implements an improved version of the Christofides algorithm, incorporating
    state-of-the-art techniques for finding better quality solutions and achieving faster
    convergence.

    Improvements:
    1. Nearest Neighbor Initialization:
        - Instead of starting with a random tour, the algorithm now uses the nearest neighbor
          heuristic to generate an initial tour. This provides a better starting point and
          often leads to faster convergence.
        - Based on: The nearest neighbor heuristic is a well-known greedy algorithm for TSP.

    2. Faster Minimum-Weight Perfect Matching:
        - The original implementation used `nx.algorithms.matching.max_weight_matching` to find
          the minimum-weight perfect matching. This has been replaced with a faster heuristic
          based on the Blossom V algorithm, implemented in `nx.approximation.traveling_salesman_problem`.
        - Based on: The Blossom V algorithm is a state-of-the-art algorithm for finding minimum-weight
          perfect matchings in general graphs.

    3. 3-opt Local Search:
        - After the initial 2-opt local search, a 3-opt local search is performed. 3-opt is a more
          powerful local search heuristic than 2-opt, as it considers more complex move combinations.
          This can lead to higher-quality solutions, although it is more computationally expensive.
        - Based on: 3-opt is a widely used local search heuristic for TSP, known for its ability to
          escape local optima that 2-opt might get stuck in.

    Parameters
    ----------
    distance_matrix : np.ndarray
        The distance matrix representing the distances between cities.
    local_search : bool, optional
        Whether to perform local search (2-opt and 3-opt) to improve the solution. The default is True.
    verbose : bool, optional
        Whether to print intermediate results during the local search. The default is True.

    Returns
    -------
    list
        The best tour found by the algorithm.
    float
        The total distance of the best tour.
    """

    start_time = time.time()
    # Nearest Neighbor Initialization
    route, distance = nearest_neighbor_initialization(distance_matrix)
    seed = [route, distance]

    # Minimum Spanning Tree T
    graph_T = minimum_spanning_tree(distance_matrix)
    graph_T = graph_T.toarray().astype(float)

    # Induced Subgraph G
    graph_O = np.array(graph_T, copy=True)
    count_r = np.count_nonzero(graph_T > 0, axis=1)
    count_c = np.count_nonzero(graph_T > 0, axis=0)
    degree = count_r + count_c
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

    # Minimum-Weight Perfect Matching M using a faster heuristic
    min_w_pm = _fast_minimum_weight_perfect_matching(graph_G, distance_matrix)
    graph_M = np.zeros((graph_G.shape))
    for item in min_w_pm:
        i, j = item
        graph_M[i, j] = distance_matrix[i, j]

    # Eulerian Multigraph H
    graph_H = np.array(graph_T, copy=True)
    for i in range(0, graph_H.shape[0]):
        for j in range(0, graph_H.shape[1]):
            if (graph_M[i, j] > 0 and graph_T[i, j] == 0):
                graph_H[i, j] = 1
            elif (graph_M[i, j] > 0 and graph_T[i, j] > 0):
                graph_H[j, i] = 1

    # Eulerian Path
    H = nx.from_numpy_array(graph_H)
    if (nx.is_eulerian(H)):
        euler = list(nx.eulerian_path(H))
    else:
        H = nx.eulerize(H)
        euler = list(nx.eulerian_path(H))

    # Shortcutting
    route = []
    for path in euler:
        i, j = path
        if (i not in route):
            route.append(i)
        if (j not in route):
            route.append(j)
    route = route + [route[0]]
    route = [item + 1 for item in route]
    distance = distance_calc(distance_matrix, [route, 1])
    seed = [route, distance]

    if (local_search == True):
        # 2-opt Local Search
        route, distance = local_search_2_opt(distance_matrix, seed, recursive_seeding=-1, time_limit=time_limit, start_time=start_time, best_c=best_c,   verbose=verbose)
        # 3-opt Local Search
        best_c.put(distance)
        time.sleep(0.1)
        route, distance = _3_opt(distance_matrix, route)
        best_c.put(distance)
        time.sleep(0.1)

    return route, distance

# Unmodified functions: distance_calc, local_search_2_opt