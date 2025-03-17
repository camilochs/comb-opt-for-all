# Required Libraries
import copy
import numpy as np

# Function: Calculate Path Distance
def calculate_distance(distance_matrix, city_list):
    path_distance = 0
    for i in range(0, len(city_list) - 1):
        path_distance = path_distance + distance_matrix[city_list[i]-1, city_list[i+1]-1]
    path_distance = path_distance + distance_matrix[city_list[-1]-1, city_list[0]-1]
    return path_distance

# Function: Perform Local Search
def local_search_2_opt(distance_matrix, city_tour):
    city_list, best_path_distance = city_tour[0], city_tour[1]
    improved                      = True
    while (improved == True):
        improved = False
        for i in range(1, len(city_list) - 2):
            for j in range(i + 1, len(city_list) - 1):
                new_city_list      = city_list[:]
                new_city_list[i:j] = city_list[i:j][::-1]
                new_distance       = calculate_distance(distance_matrix, new_city_list)
                if (new_distance < best_path_distance):
                    best_path_distance = new_distance
                    city_list          = new_city_list
                    improved           = True
    return city_list, best_path_distance

############################################################################

# Function: Calculate Attractiveness
def attractiveness(distance_matrix):
    h = 1 / (distance_matrix + 1e-10) 
    np.fill_diagonal(h, 0)
    return h

# Function: Update Pheromone Matrix
def update_thau(distance_matrix, thau, city_list, best_path_distance):
    """
    Updates the pheromone matrix based on the best ant's path.

    Implements Elitist Ant System (EAS) pheromone update. In EAS, the best-so-far ant deposits 
    extra pheromone on its path, reinforcing the best solution found. This helps to guide the search 
    towards promising regions of the search space.

    EAS is a state-of-the-art technique that improves convergence speed and solution quality by 
    emphasizing the best solution found so far.

    Parameters
    ----------
    distance_matrix : np.ndarray
        The distance matrix.
    thau : np.ndarray
        The pheromone matrix.
    city_list : list
        The best ant's city list.
    best_path_distance : float
        The distance of the best path found so far.

    Returns
    -------
    thau : np.ndarray
        The updated pheromone matrix.
    """
    path_distance = 0
    for i in range(len(city_list) - 1):
        path_distance = path_distance + distance_matrix[city_list[i]-1, city_list[i+1]-1]
    path_distance = path_distance + distance_matrix[city_list[-1]-1, city_list[0]-1]  
    
    # Elitist update: Add extra pheromone based on the best path
    for i in range(len(city_list) - 1):
        thau[city_list[ i ]-1, city_list[i+1]-1] = thau[city_list[ i ]-1, city_list[i+1]-1] + (1 / path_distance) + (1 / best_path_distance)
        thau[city_list[i+1]-1, city_list[ i ]-1] = thau[city_list[i+1]-1, city_list[ i ]-1] + (1 / path_distance) + (1 / best_path_distance)
    thau[city_list[-1]-1, city_list[ 0]-1] = thau[city_list[-1]-1, city_list[ 0]-1] + (1 / path_distance) + (1 / best_path_distance)
    thau[city_list[ 0]-1, city_list[-1]-1] = thau[city_list[ 0]-1, city_list[-1]-1] + (1 / path_distance) + (1 / best_path_distance)
    return thau

# Function: Generate Ant Paths
def ants_path(distance_matrix, h, thau, alpha, beta, full_list, ants, local_search, q0):
    """
    Generates paths for each ant using a probabilistic approach.

    Implements a modified ant path generation process with a pseudorandom proportional rule, 
    inspired by Ant Colony System (ACS). This rule introduces a parameter 'q0' that controls 
    the balance between exploration and exploitation. With probability 'q0', the ant chooses 
    the city with the highest pheromone and attractiveness (exploitation), and with probability 
    '1-q0', it uses the standard probabilistic selection (exploration).

    ACS is a state-of-the-art technique that improves convergence speed by introducing a more 
    aggressive selection rule and local pheromone updates.

    Parameters
    ----------
    distance_matrix : np.ndarray
        The distance matrix.
    h : np.ndarray
        The attractiveness matrix.
    thau : np.ndarray
        The pheromone matrix.
    alpha : float
        The pheromone importance parameter.
    beta : float
        The attractiveness importance parameter.
    full_list : list
        The list of all cities.
    ants : int
        The number of ants.
    local_search : bool
        Whether to perform local search.
    q0 : float
        The parameter for the pseudorandom proportional rule (0 <= q0 <= 1).

    Returns
    -------
    best_city_list : list
        The best city list found by the ants.
    best_path_distance : float
        The distance of the best path.
    thau : np.ndarray
        The updated pheromone matrix.
    """
    best_path_distance = float('inf')
    best_city_list     = None
    for _ in range(0, ants):
        city_list = [np.random.choice(full_list)]
        while (len(city_list) < len(full_list)):
            current_city  = city_list[-1]
            
            # Pseudorandom Proportional Rule (ACS)
            q = np.random.rand()
            if q <= q0:
                # Exploitation: Choose the best city based on pheromone and attractiveness
                max_prob = -1
                next_city = -1
                for city in full_list:
                    if city not in city_list:
                        prob = (thau[current_city-1, city-1] ** alpha) * (h[current_city-1, city-1] ** beta)
                        if prob > max_prob:
                            max_prob = prob
                            next_city = city
            else:
                # Exploration: Standard probabilistic selection
                probabilities = []
                for next_city in full_list:
                    if (next_city not in city_list):
                        p = (thau[current_city-1, next_city-1] ** alpha) * (h[current_city-1, next_city-1] ** beta)
                        probabilities.append(p)
                    else:
                        probabilities.append(0)
                probabilities = np.array(probabilities) / np.sum(probabilities)
                next_city     = np.random.choice(full_list, p = probabilities)
            
            city_list.append(next_city)
        path_distance = calculate_distance(distance_matrix, city_list)
        if (path_distance < best_path_distance):
            best_city_list     = copy.deepcopy(city_list)
            best_path_distance = path_distance
            
    if (local_search == True):
        best_city_list, best_path_distance = local_search_2_opt(distance_matrix, city_tour = [best_city_list, best_path_distance])
    thau = update_thau(distance_matrix, thau, city_list = best_city_list, best_path_distance = best_path_distance)
    return best_city_list, best_path_distance, thau

############################################################################

# ACO Function
def ant_colony_optimization(distance_matrix, ants = 5, iterations = 50, alpha = 1, beta = 2, decay = 0.05, time_limit=10, local_search = True, verbose = True):
    """
    Performs Ant Colony Optimization to solve the Traveling Salesman Problem.

    This implementation incorporates two state-of-the-art techniques:
    1. Elitist Ant System (EAS) for pheromone update.
    2. Pseudorandom proportional rule from Ant Colony System (ACS) for ant path generation.

    These enhancements improve both the convergence speed and the quality of the solutions found.

    Parameters
    ----------
    distance_matrix : np.ndarray
        The distance matrix representing the distances between cities.
    ants : int, optional
        The number of ants to use, by default 5.
    iterations : int, optional
        The number of iterations to run, by default 50.
    alpha : float, optional
        The pheromone importance parameter, by default 1.
    beta : float, optional
        The attractiveness importance parameter, by default 2.
    decay : float, optional
        The pheromone decay rate, by default 0.05.
    local_search : bool, optional
        Whether to perform local search (2-opt) after each iteration, by default True.
    verbose : bool, optional
        Whether to print progress information, by default True.

    Returns
    -------
    best_route : list
        The best route found, represented as a list of city indices.
    best_distance : float
        The total distance of the best route.
    """
    import time
    start_time = time.time()
    count      = 0
    best_route = []
    full_list  = list(range(1, distance_matrix.shape[0] + 1))
    distance   = np.sum(distance_matrix.sum())
    h          = attractiveness(distance_matrix)
    thau       = np.ones((distance_matrix.shape[0], distance_matrix.shape[0]))
    q0         = 0.9 # Parameter for the pseudorandom proportional rule (ACS)
    while (count <= iterations):
        if (verbose == True and count > 0):
            print(f'Iteration = {count}, Distance = {round(best_route[1], 2)}')
        city_list, path_distance, thau = ants_path(distance_matrix, h, thau, alpha, beta, full_list, ants, local_search, q0)
        thau                           = thau*(1 - decay)
        if (distance > path_distance) and validate_solution(distance_matrix, city_list, path_distance):
            best_route = copy.deepcopy([city_list])
            best_route.append(path_distance)
            distance   = best_route[1]
        count = count + 1
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break
    init_city = best_route[0][0]
    best_route[0].append(init_city)
    if not validate_solution(distance_matrix, best_route[0], best_route[1]):
        raise ValueError("La solución obtenida no es válida.")
    return best_route[0], best_route[1]

############################################################################

# Unmodified functions: calculate_distance, local_search_2_opt, attractiveness
def validate_solution(distance_matrix, city_list, path_distance):
    """Valida que la solución tenga todas las ciudades una sola vez y que la distancia calculada sea correcta."""
    n = distance_matrix.shape[0]
    expected_cities = set(range(1, n + 1))  # Conjunto de ciudades esperadas (1, ..., n)
    solution_cities = set(city_list)  # Conjunto de ciudades en la solución

    if solution_cities != expected_cities:
        missing = expected_cities - solution_cities
        extra = solution_cities - expected_cities
        print(f"Error: La solución no contiene todas las ciudades exactamente una vez.")
        print(f"  - Faltantes: {missing}")
        print(f"  - Extra: {extra}")
        print(f"  - Solución dada: {city_list}")
        return False

    # Verificar que la distancia calculada es correcta
    computed_distance = 0
    for i in range(len(city_list) - 1):
        computed_distance += distance_matrix[city_list[i] - 1, city_list[i + 1] - 1]  # Ajuste de índices

    # Cerrar el ciclo sumando la distancia de vuelta al inicio
    computed_distance += distance_matrix[city_list[-1] - 1, city_list[0] - 1]

    if not np.isclose(computed_distance, path_distance, atol=1e-5):
        print(f"Error: La distancia calculada no coincide con path_distance.")
        print(f"  - Distancia esperada: {path_distance}")
        print(f"  - Distancia calculada: {computed_distance}")
        return False

    return True
