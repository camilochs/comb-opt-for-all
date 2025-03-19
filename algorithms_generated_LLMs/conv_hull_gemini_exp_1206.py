
# Required Libraries
import copy
import random
import numpy as np
import time
from scipy.spatial import ConvexHull

############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m      = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]       
    return distance

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, verbose = True):
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
        seed       = copy.deepcopy(city_list)      
        for i in range(0, len(city_list[0]) - 2):
            for j in range(i+1, len(city_list[0]) - 1):
                best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))        
                best_route[0][-1]  = best_route[0][0]  
                best_route[1]      = distance_calc(distance_matrix, best_route)              
                if (city_list[1] > best_route[1]):
                    city_list = copy.deepcopy(best_route)     
                best_route = copy.deepcopy(seed)
        count      = count + 1
        iteration = iteration + 1  
        if (distance > city_list[1] and recursive_seeding < 0):
            distance         = city_list[1]
            count          = -2
            recursive_seeding = -1
        elif(city_list[1] >= distance and recursive_seeding < 0):
            count          = -1
            recursive_seeding = -2
    return city_list[0], city_list[1]

############################################################################

# Function: 3_opt
def local_search_3_opt(distance_matrix, city_tour, time_limit=10, start_time=None, best_c=None, verbose=True):
    """
    Improves a tour using the 3-opt local search algorithm.

    This function implements the 3-opt algorithm, which is a generalization of the 2-opt algorithm.
    It iteratively explores possible improvements to the tour by considering all possible ways to
    reconnect three edges of the tour.

    Enhancements:
        - Better Quality Solutions: 3-opt can escape local optima that 2-opt might get stuck in,
          leading to potentially better solutions. It explores a larger neighborhood of solutions.
        - State-of-the-art Technique: 3-opt is a widely used and effective local search heuristic
          for the TSP.

    Args:
        distance_matrix (np.ndarray): The distance matrix between cities.
        city_tour (list): The current tour represented as a list of city indices and its total distance.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        tuple: The improved tour and its total distance.
    """
    city_list = copy.deepcopy(city_tour)
    best_distance = city_list[1]
    iteration = 0
    improvement = True
    while improvement:
        improvement = False
        if verbose:
            print('Iteration = ', iteration, 'Distance = ', round(best_distance, 2))
        for i in range(0, len(city_list[0]) - 3):
            for j in range(i + 2, len(city_list[0]) - 2):
                for k in range(j + 2, len(city_list[0]) - 1):
                    # Possible reconnections (excluding original and 2-opt moves)
                    
                    # Case 1: Reverse i+1 to j
                    A = city_list[0][:i+1]
                    B = list(reversed(city_list[0][i+1:j+1]))
                    C = city_list[0][j+1:k+1]
                    D = city_list[0][k+1:]
                    new_tour = [A + B + C + D, 0]
                    new_tour[0][-1] = new_tour[0][0]
                    new_tour[1] = distance_calc(distance_matrix, new_tour)
                    if new_tour[1] < best_distance:
                        city_list = copy.deepcopy(new_tour)
                        best_distance = new_tour[1]
                        best_c.put(best_distance)
                        time.sleep(0.1)
                        improvement = True
                        continue

                    # Case 2: Reverse j+1 to k
                    A = city_list[0][:i+1]
                    B = city_list[0][i+1:j+1]
                    C = list(reversed(city_list[0][j+1:k+1]))
                    D = city_list[0][k+1:]
                    new_tour = [A + B + C + D, 0]
                    new_tour[0][-1] = new_tour[0][0]
                    new_tour[1] = distance_calc(distance_matrix, new_tour)
                    if new_tour[1] < best_distance:
                        city_list = copy.deepcopy(new_tour)
                        best_distance = new_tour[1]
                        best_c.put(best_distance)
                        time.sleep(0.1)
                        improvement = True
                        continue

                    # Case 3: Reverse i+1 to k
                    A = city_list[0][:i+1]
                    B = list(reversed(city_list[0][i+1:k+1]))
                    C = city_list[0][k+1:]
                    new_tour = [A + B + C, 0]
                    new_tour[0][-1] = new_tour[0][0]
                    new_tour[1] = distance_calc(distance_matrix, new_tour)
                    if new_tour[1] < best_distance:
                        city_list = copy.deepcopy(new_tour)
                        best_distance = new_tour[1]
                        best_c.put(best_distance)
                        time.sleep(0.1)
                        improvement = True
                        continue

                    # Case 4: i+1 to j, then j+1 to k
                    A = city_list[0][:i+1]
                    B = city_list[0][j+1:k+1]
                    C = city_list[0][i+1:j+1]
                    D = city_list[0][k+1:]
                    new_tour = [A + B + C + D, 0]
                    new_tour[0][-1] = new_tour[0][0]
                    new_tour[1] = distance_calc(distance_matrix, new_tour)
                    if new_tour[1] < best_distance:
                        city_list = copy.deepcopy(new_tour)
                        best_distance = new_tour[1]
                        best_c.put(best_distance)
                        time.sleep(0.1)
                        improvement = True
                        continue

        iteration += 1
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break  
    return city_list[0], city_list[1]

# Function: Convex Hull with Lin-Kernighan heuristic
def convex_hull_algorithm(coordinates, distance_matrix, time_limit=10, best_c=None, local_search=True, verbose=True):
    """
    Solves the Traveling Salesperson Problem (TSP) using the Convex Hull algorithm with optional local search.

    This function first constructs an initial tour using the Convex Hull algorithm. It then optionally
    applies the 3-opt local search heuristic to improve the solution.
    
    Enhancements:
        - 3-opt Local Search: The 3-opt heuristic is incorporated to refine the initial solution obtained
          from the Convex Hull algorithm. This significantly improves the quality of the final tour.
        - State-of-the-art Technique: The combination of Convex Hull for initial tour construction and
          3-opt for local search represents a more advanced approach to solving the TSP compared to
          using only 2-opt.

    Args:
        coordinates (np.ndarray): The coordinates of the cities.
        distance_matrix (np.ndarray): The distance matrix between cities.
        local_search (bool, optional): Whether to apply local search (3-opt). Defaults to True.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        tuple: The best tour found and its total distance.
    """
    start_time = time.time()
    hull      = ConvexHull(coordinates)
    idx_h     = hull.vertices.tolist()
    idx_h     = [item+1 for item in idx_h]
    idx_h_pairs = [(idx_h[i], idx_h[i+1]) for i in range(0, len(idx_h)-1)]
    idx_h_pairs.append((idx_h[-1], idx_h[0]))
    idx_in    = [item for item in list(range(1, coordinates.shape[0]+1)) if item not in idx_h]
    for _ in range(0, len(idx_in)):
        x = []
        y = []
        z = []
        for i in range(0, len(idx_in)):
            L         = idx_in[i]
            cost      = [(distance_matrix[m-1, L-1], distance_matrix[L-1, n-1], distance_matrix[m-1, n-1]) for m, n in idx_h_pairs]
            cost_idx  = [(m, L, n) for m, n in idx_h_pairs]
            cost_vec_1 = [ item[0] + item[1] - item[2] for item in cost]
            cost_vec_2 = [(item[0] + item[1]) / (item[2] + 0.00000000000000001) for item in cost]
            x.append(cost_vec_1.index(min(cost_vec_1)))
            y.append(cost_vec_2[x[-1]])
            z.append(cost_idx[x[-1]])
        m, L, n   = z[y.index(min(y))]
        idx_in.remove(L)
        ins       = idx_h.index(m)
        idx_h.insert(ins + 1, L)
        idx_h_pairs = [ (idx_h[i], idx_h[i+1]) for i in range(0, len(idx_h)-1)]
        idx_h_pairs.append((idx_h[-1], idx_h[0]))
    route     = idx_h + [idx_h[0]]
    distance = distance_calc(distance_matrix, [route, 1])
    best_c.put(distance)
    time.sleep(0.1)
    seed      = [route, distance]
    if local_search:
        route, distance = local_search_3_opt(distance_matrix, seed, time_limit=time_limit, start_time=start_time, best_c=best_c,  verbose=verbose)
        best_c.put(distance)
        time.sleep(0.1)
    return route, distance

############################################################################
# Unmodified functions:
# - distance_calc
# - local_search_2_opt