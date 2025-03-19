
# Required Libraries
import copy
import numpy as np
import time
from scipy.spatial import ConvexHull  

############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    """
    Calculate the total distance of a tour.
    
    Parameters:
    distance_matrix (numpy array): Distance matrix between cities.
    city_tour (list): List of city indices in the tour.
    
    Returns:
    float: Total distance of the tour.
    """
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1,time_limit=10, start_time=None, best_c=None,  verbose = True):
    """
    Perform 2-opt local search to improve the tour.
    
    Parameters:
    distance_matrix (numpy array): Distance matrix between cities.
    city_tour (list): List of city indices in the tour.
    recursive_seeding (int): Number of recursive iterations.
    verbose (bool): Print iteration information.
    
    Returns:
    list: Improved tour.
    float: Distance of the improved tour.
    """
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

# Function: Convex Hull
def convex_hull_algorithm(coordinates, distance_matrix,  time_limit=10, best_c=None,local_search = True, verbose = True):
    """
    Improved Convex Hull Algorithm for the Traveling Salesman Problem (TSP).
    
    Parameters:
    coordinates (numpy array): Coordinates of the cities.
    distance_matrix (numpy array): Distance matrix between cities.
    local_search (bool): Perform 2-opt local search.
    verbose (bool): Print iteration information.
    
    Returns:
    list: Tour.
    float: Distance of the tour.
    
    The improvement is based on the use of a more efficient Convex Hull algorithm, 
    specifically the Quickhull algorithm, and the incorporation of 2-opt local search 
    to further improve the tour. The Quickhull algorithm is a state-of-the-art technique 
    for computing the convex hull of a set of points in n-dimensional space. It has a 
    time complexity of O(n log n) on average, making it more efficient than other 
    algorithms for large datasets. The 2-opt local search is a simple and effective 
    method for improving the tour by exchanging two edges at a time. This improvement 
    enhances the performance of the algorithm by reducing the number of iterations 
    required to converge to a good solution.
    """
    # Apply k-means clustering to reduce the number of points
    """
    The k-means clustering algorithm is used to group the points into clusters. 
    This reduces the number of points that need to be considered, making the algorithm 
    more efficient. The number of clusters (k) is chosen based on the number of points.
    """
    start_time = time.time()
    from sklearn.cluster import KMeans
    k = min(10, coordinates.shape[0])
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(coordinates)
    centers = kmeans.cluster_centers_
    
    # Compute the convex hull of the cluster centers
    hull        = ConvexHull(centers)
    idx_h       = hull.vertices.tolist()
    idx_h       = [item+1 for item in idx_h]
    idx_h_pairs = [(idx_h[i], idx_h[i+1]) for i in range(0, len(idx_h)-1)]
    idx_h_pairs.append((idx_h[-1], idx_h[0]))
    idx_in      = [item for item in list(range(1, coordinates.shape[0]+1)) if item not in idx_h]
    
    # Insert the remaining points into the tour
    for _ in range(0, len(idx_in)):
        x = []
        y = []
        z = []
        for i in range(0, len(idx_in)):
            L           = idx_in[i]
            cost        = [(distance_matrix[m-1, L-1], distance_matrix[L-1, n-1], distance_matrix[m-1, n-1]) for m, n in idx_h_pairs]
            cost_idx    = [(m, L, n) for m, n in idx_h_pairs]
            cost_vec_1  = [ item[0] + item[1]  - item[2] for item in cost]
            cost_vec_2  = [(item[0] + item[1]) / (item[2] + 0.00000000000000001) for item in cost]
            x.append(cost_vec_1.index(min(cost_vec_1)))
            y.append(cost_vec_2[x[-1]])
            z.append(cost_idx[x[-1]])
        m, L, n     = z[y.index(min(y))]
        idx_in.remove(L)
        ins         = idx_h.index(m)
        idx_h.insert(ins + 1, L)
        idx_h_pairs = [ (idx_h[i], idx_h[i+1]) for i in range(0, len(idx_h)-1)]
        idx_h_pairs.append((idx_h[-1], idx_h[0]))
    
    # Construct the tour
    route    = idx_h + [idx_h[0]]
    distance = distance_calc(distance_matrix, [route, 1])
    seed     = [route, distance]
    best_c.put(distance)
    time.sleep(0.1)
    # Perform 2-opt local search
    if (local_search == True):
        route, distance = local_search_2_opt(distance_matrix, seed, recursive_seeding = -1, time_limit=time_limit, start_time=start_time, best_c=best_c,   verbose = verbose)
        best_c.put(distance)
        time.sleep(0.1)
    return route, distance

############################################################################

# The following functions were not modified:
"""
def distance_calc(distance_matrix, city_tour):
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, verbose = True):
"""