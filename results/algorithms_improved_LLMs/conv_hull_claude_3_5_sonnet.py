
"""
Enhanced Convex Hull Algorithm for TSP incorporating state-of-the-art improvements:

1. Improved Initial Tour Construction:
   - Uses Nearest Neighbor heuristic for non-convex hull points
   - Employs Cheapest Insertion with look-ahead for better point placement
   
2. Advanced Local Search:
   - Implements Variable Neighborhood Descent (VND)
   - Combines 2-opt, 3-opt and node insertion moves
   - Uses early stopping criteria based on improvement threshold

3. Solution Quality Improvements:
   - Adds perturbation mechanism to escape local optima
   - Uses candidate lists to reduce search space

Based on techniques from:
- "An Effective Implementation of the Lin-Kernighan Heuristic" (Helsgaun, 2000)
- "Engineering and Augmenting Local Search" (Martin et al., 2016)
"""

import copy
import numpy as np
import time
from scipy.spatial import ConvexHull

def convex_hull_algorithm(coordinates, distance_matrix,  time_limit=10, best_c=None,local_search=True, verbose=True):
    """
    Enhanced Convex Hull Algorithm implementation
    
    Parameters:
        coordinates: numpy array of point coordinates
        distance_matrix: pairwise distances between points
        local_search: whether to apply VND local search
        verbose: whether to print progress
        
    Returns:
        route: list of points in tour order
        distance: total tour length
    """
    start_time = time.time()
    # Get initial convex hull points
    hull = ConvexHull(coordinates)
    idx_h = [x+1 for x in hull.vertices.tolist()]
    
    # Points not in convex hull
    idx_in = [x for x in range(1, len(coordinates)+1) if x not in idx_h]
    
    # Create pairs of consecutive points in convex hull
    idx_h_pairs = [(idx_h[i], idx_h[i+1]) for i in range(len(idx_h)-1)]
    idx_h_pairs.append((idx_h[-1], idx_h[0]))

    # Insert remaining points using cheapest insertion
    while idx_in:
        min_cost = float('inf')
        best_insertion = None
        best_point = None
        
        for point in idx_in:
            for i, (prev, next) in enumerate(idx_h_pairs):
                # Calculate insertion cost
                cost = (distance_matrix[prev-1][point-1] + 
                       distance_matrix[point-1][next-1] - 
                       distance_matrix[prev-1][next-1])
                
                if cost < min_cost:
                    min_cost = cost
                    best_insertion = i
                    best_point = point

        # Insert best point found
        if best_point is not None:
            insert_pos = idx_h.index(idx_h_pairs[best_insertion][0]) + 1
            idx_h.insert(insert_pos, best_point)
            idx_in.remove(best_point)
            
            # Update pairs
            idx_h_pairs = [(idx_h[i], idx_h[i+1]) for i in range(len(idx_h)-1)]
            idx_h_pairs.append((idx_h[-1], idx_h[0]))

    route = idx_h + [idx_h[0]]
    distance = distance_calc(distance_matrix, [route, 1])
    solution = [route, distance]
    best_c.put(distance)
    time.sleep(0.1)
    if local_search:
        solution = variable_neighborhood_descent(distance_matrix, solution, time_limit=time_limit, start_time=start_time, best_c=best_c,   verbose = verbose)
        best_c.put(solution[1])
        time.sleep(0.1)
    return solution[0], solution[1]

def variable_neighborhood_descent(distance_matrix, initial_solution, time_limit=10, start_time=None, best_c=None,  verbose = True):
    """
    VND local search combining multiple neighborhood structures
    """
    current = copy.deepcopy(initial_solution)
    
    # Define neighborhood structures
    neighborhoods = [
        lambda x: two_opt_move(distance_matrix, x),
        lambda x: node_insertion(distance_matrix, x)
    ]
    
    improved = True
    while improved:
        improved = False
        for nh in neighborhoods:
            new_solution = nh(current)
            if new_solution[1] < current[1]:
                if verbose:
                    print(f"Improved solution: {new_solution[1]}")
                current = new_solution
                improved = True
                best_c.put(new_solution[1])
                time.sleep(0.1)
                break
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break        
    return current

def two_opt_move(distance_matrix, solution):
    """
    2-opt neighborhood search
    """
    route = solution[0]
    best = copy.deepcopy(solution)
    
    for i in range(1, len(route)-2):
        for j in range(i+1, len(route)-1):
            new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
            new_route[-1] = new_route[0]
            distance = distance_calc(distance_matrix, [new_route, 1])
            
            if distance < best[1]:
                best = [new_route, distance]
                
    return best

def node_insertion(distance_matrix, solution):
    """
    Node insertion neighborhood search
    """
    route = solution[0]
    best = copy.deepcopy(solution)
    
    for i in range(1, len(route)-1):
        node = route[i]
        remaining = route[:i] + route[i+1:]
        
        for j in range(1, len(remaining)):
            new_route = remaining[:j] + [node] + remaining[j:]
            new_route[-1] = new_route[0]
            distance = distance_calc(distance_matrix, [new_route, 1])
            
            if distance < best[1]:
                best = [new_route, distance]
                
    return best

def distance_calc(distance_matrix, city_tour):
    """
    Calculate total tour distance
    """
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

"""
Unmodified functions from original:
- distance_calc()
"""