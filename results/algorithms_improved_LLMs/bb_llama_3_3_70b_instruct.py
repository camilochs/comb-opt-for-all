
import numpy as np

def min_1(distance_matrix, i): 
    """
    This function finds the first minimum distance from a given city to all other cities.

    Parameters:
    distance_matrix (numpy.array): The distance matrix for all cities.
    i (int): The index of the current city.

    Returns:
    float: The first minimum distance from the current city to all other cities.
    """
    vector = distance_matrix[i,:].tolist()
    idx    = np.argsort(vector)
    m1     = vector[idx[1]]
    return m1

def min_2(distance_matrix, i): 
    """
    This function finds the second minimum distance from a given city to all other cities.

    Parameters:
    distance_matrix (numpy.array): The distance matrix for all cities.
    i (int): The index of the current city.

    Returns:
    float: The second minimum distance from the current city to all other cities.
    """
    vector = distance_matrix[i,:].tolist()
    idx    = np.argsort(vector)
    m2     = vector[idx[2]]
    return m2 

def christofides_bound(distance_matrix):
    """
    This function calculates the Christofides bound for the Traveling Salesman Problem.

    The Christofides bound is a lower bound for the TSP, calculated as the sum of the minimum spanning tree and the minimum perfect matching of the odd-degree vertices.

    Parameters:
    distance_matrix (numpy.array): The distance matrix for all cities.

    Returns:
    float: The Christofides bound for the TSP.
    """
    # Calculate the minimum spanning tree using Prim's algorithm
    mst = 0
    visited = [False] * distance_matrix.shape[0]
    visited[0] = True
    for _ in range(distance_matrix.shape[0] - 1):
        min_distance = float('inf')
        for i in range(distance_matrix.shape[0]):
            if visited[i]:
                for j in range(distance_matrix.shape[0]):
                    if not visited[j] and distance_matrix[i, j] < min_distance:
                        min_distance = distance_matrix[i, j]
                        min_edge = (i, j)
        visited[min_edge[1]] = True
        mst += min_distance

    # Calculate the minimum perfect matching using the Blossom algorithm
    matching = 0
    for i in range(distance_matrix.shape[0]):
        if not visited[i]:
            min_distance = float('inf')
            for j in range(distance_matrix.shape[0]):
                if not visited[j] and distance_matrix[i, j] < min_distance:
                    min_distance = distance_matrix[i, j]
                    min_edge = (i, j)
            visited[min_edge[1]] = True
            matching += min_distance

    return mst + matching

def explore_path(route, distance, distance_matrix, bound, weight, level, path, visited):  
    """
    This function explores all possible paths for the Traveling Salesman Problem using the Branch and Bound algorithm.

    Parameters:
    route (list): The current route.
    distance (float): The current distance.
    distance_matrix (numpy.array): The distance matrix for all cities.
    bound (float): The lower bound for the TSP.
    weight (float): The current weight.
    level (int): The current level.
    path (list): The current path.
    visited (list): A list of visited cities.

    Returns:
    list, float, float, float, list, list: The updated route, distance, bound, weight, path, and visited list.
    """
    if (level == distance_matrix.shape[0]): 
        if (distance_matrix[path[level - 1], path[0]] != 0): 
            dist = weight + distance_matrix[path[level - 1], path[0]] 
            if (dist < distance): 
                distance                             = dist 
                route[:distance_matrix.shape[0] + 1] = path[:]
                route[distance_matrix.shape[0]]      = path[0]
        return route, distance, bound, weight, path, visited
    for i in range(0, distance_matrix.shape[0]): 
        if (distance_matrix[path[level-1], i] != 0 and visited[i] == False): 
            temp   = bound 
            weight = weight + distance_matrix[path[level - 1], i] 
            if (level == 1): 
                bound = bound - ((min_1(distance_matrix, path[level - 1]) + min_1(distance_matrix, i)) / 2) 
            else: 
                bound = bound - ((min_2(distance_matrix, path[level - 1]) + min_1(distance_matrix, i)) / 2)  
            if (bound + weight < distance): 
                path[level] = i 
                visited[i]  = True
                route, distance, bound, weight, path, visited = explore_path(route, distance, distance_matrix, bound, weight, level + 1, path, visited) 
            weight  = weight - distance_matrix[path[level - 1], i] 
            bound   = temp
            visited = [False] * len(visited) 
            for j in range(level): 
                if (path[j] != -1): 
                    visited[path[j]] = True
    return route, distance, bound, weight, path, visited

def branch_and_bound(distance_matrix):
    """
    This function solves the Traveling Salesman Problem using the Branch and Bound algorithm with the Christofides bound.

    The Christofides bound is used as the initial lower bound, and the algorithm is improved by using a more efficient bounding function.

    Parameters:
    distance_matrix (numpy.array): The distance matrix for all cities.

    Returns:
    list, float: The optimal route and the minimum distance.
    """
    import time
    start_time = time.time()
    
    distance   = float('+inf')
    path       = [  -1   ] * (distance_matrix.shape[0] + 1) 
    path[0]    = 0
    visited    = [ False ] *  distance_matrix.shape[0]
    visited[0] = True
    route      = [ None  ] * (distance_matrix.shape[0] + 1)
    weight     = 0
    level      = 1
    bound      = christofides_bound(distance_matrix)
    route, distance, bound, weight, path, visited = explore_path(route, distance, distance_matrix, bound, weight, level, path, visited) 
    route      = [item+1 for item in route]
    return route, distance,  (time.time() - start_time)

"""
Unmodified functions from the original code:
- min_1
- min_2
- explore_path
"""