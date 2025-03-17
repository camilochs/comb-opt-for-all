
import copy
import numpy as np

# Function: Calculate Path Distance
def calculate_distance(distance_matrix, city_list):
    """
    Calculate the total distance of a given path.
    """
    path_distance = 0
    for i in range(0, len(city_list) - 1):
        path_distance = path_distance + distance_matrix[city_list[i]-1, city_list[i+1]-1]
    path_distance = path_distance + distance_matrix[city_list[-1]-1, city_list[0]-1]
    return path_distance

# Function: Perform Local Search
def local_search_2_opt(distance_matrix, city_tour):
    """
    Perform a local search using the 2-opt algorithm to improve the solution.
    This is a state-of-the-art technique used in many TSP solvers.
    It enhances performance by exploring the neighborhood of the current solution.
    """
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

# Function: Calculate Attractiveness
def attractiveness(distance_matrix):
    """
    Calculate the attractiveness of each edge in the graph.
    This is done by inverting the distance matrix and adding a small value to avoid division by zero.
    """
    h = 1 / (distance_matrix + 1e-10) 
    np.fill_diagonal(h, 0)
    return h

# Function: Update Pheromone Matrix
def update_thau(distance_matrix, thau, city_list):
    """
    Update the pheromone matrix based on the path found by the ants.
    This is done by adding a small amount of pheromone to the edges that were visited.
    """
    path_distance = 0
    for i in range(len(city_list) - 1):
        path_distance = path_distance + distance_matrix[city_list[i]-1, city_list[i+1]-1]
    path_distance = path_distance + distance_matrix[city_list[-1]-1, city_list[0]-1]  
    for i in range(len(city_list) - 1):
        thau[city_list[ i ]-1, city_list[i+1]-1] = thau[city_list[ i ]-1, city_list[i+1]-1] + 1 / path_distance
        thau[city_list[i+1]-1, city_list[ i ]-1] = thau[city_list[i+1]-1, city_list[ i ]-1] + 1 / path_distance
    thau[city_list[-1]-1, city_list[ 0]-1] = thau[city_list[-1]-1, city_list[ 0]-1] + 1 / path_distance
    thau[city_list[ 0]-1, city_list[-1]-1] = thau[city_list[ 0]-1, city_list[-1]-1] + 1 / path_distance
    return thau

# Function: Generate Ant Paths
def ants_path(distance_matrix, h, thau, alpha, beta, full_list, ants, local_search):
    """
    Generate paths for each ant in the colony.
    This is done by choosing the next city based on the pheromone and attractiveness of each edge.
    The alpha and beta parameters control the importance of pheromone and attractiveness, respectively.
    """
    best_path_distance = float('inf')
    best_city_list     = None
    for _ in range(0, ants):
        city_list = [np.random.choice(full_list)]
        while (len(city_list) < len(full_list)):
            current_city  = city_list[-1]
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
    thau = update_thau(distance_matrix, thau, city_list = best_city_list)
    return best_city_list, best_path_distance, thau

# ACO Function with improvements
def ant_colony_optimization(distance_matrix, ants = 5, iterations = 50, alpha = 1, beta = 2, decay = 0.05, time_limit=10, local_search = True, verbose = True):
    """
    Ant Colony Optimization function with state-of-the-art techniques.
    This function uses the MAX-MIN Ant System (MMAS) to improve the convergence speed.
    It also uses the 2-opt local search to improve the solution quality.
    The alpha and beta parameters control the importance of pheromone and attractiveness, respectively.
    The decay parameter controls the rate at which the pheromone evaporates.
    """
    import time
    start_time = time.time()
    count      = 0
    best_route = []
    full_list  = list(range(1, distance_matrix.shape[0] + 1))
    distance   = np.sum(distance_matrix.sum())
    h          = attractiveness(distance_matrix)
    # Initialize the pheromone matrix with a small random value
    thau       = np.random.uniform(0.1, 0.5, (distance_matrix.shape[0], distance_matrix.shape[0]))
    np.fill_diagonal(thau, 0)
    while (count <= iterations):
        if (verbose == True and count > 0):
            print(f'Iteration = {count}, Distance = {round(best_route[1], 2)}')
        city_list, path_distance, thau = ants_path(distance_matrix, h, thau, alpha, beta, full_list, ants, local_search)
        # Apply the MAX-MIN Ant System (MMAS) to update the pheromone matrix
        thau = thau * (1 - decay)
        thau[thau < 0.1] = 0.1
        thau[thau > 0.5] = 0.5
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
# Unmodified functions from the original code:
# - calculate_distance
# - attractiveness
# - update_thau
############################################################################

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
