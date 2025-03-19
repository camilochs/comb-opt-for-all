
# Required Libraries
import copy
import numpy as np

def calculate_distance(distance_matrix, city_list):
    """
    Calculates the total distance of the path represented by city_list.

    Parameters:
    - distance_matrix: A 2D numpy array representing distances between cities.
    - city_list: A list of cities representing the tour.

    Returns:
    - path_distance: The total distance of the tour.
    """
    path_distance = 0
    for i in range(len(city_list) - 1):
        path_distance += distance_matrix[city_list[i]-1, city_list[i+1]-1]
    path_distance += distance_matrix[city_list[-1]-1, city_list[0]-1]
    return path_distance

def local_search_2_opt(distance_matrix, city_tour):
    """
    Performs the 2-opt local search algorithm to improve the given tour.

    Parameters:
    - distance_matrix: A 2D numpy array representing distances between cities.
    - city_tour: A list containing the tour and its total distance [city_list, path_distance].

    Returns:
    - city_list: The improved tour after 2-opt.
    - best_path_distance: The total distance of the improved tour.
    """
    city_list, best_path_distance = city_tour
    improved = True
    while improved:
        improved = False
        for i in range(1, len(city_list) - 2):
            for j in range(i + 1, len(city_list) - 1):
                new_city_list = city_list[:]
                new_city_list[i:j] = city_list[i:j][::-1]
                new_distance = calculate_distance(distance_matrix, new_city_list)
                if new_distance < best_path_distance:
                    best_path_distance = new_distance
                    city_list = new_city_list
                    improved = True
    return city_list, best_path_distance

def attractiveness(distance_matrix):
    """
    Calculates the heuristic matrix (attractiveness) from the distance matrix.

    Parameters:
    - distance_matrix: A 2D numpy array representing distances between cities.

    Returns:
    - h: The heuristic matrix where each element h_ij is the inverse of the distance between city i and j.
    """
    h = 1 / (distance_matrix + 1e-10) 
    np.fill_diagonal(h, 0)
    return h

def acs_ants_path(distance_matrix, h, thau, alpha, beta, full_list, ants, xi, q0, tau0, local_search):
    """
    Construct ant paths using the Ant Colony System (ACS) algorithm with pseudorandom proportional rule and local pheromone updating.

    Improvements Implemented:
    - Pseudorandom Proportional Rule: Each ant selects the next city using a probability q0 to balance exploitation of the best edge and exploration of available edges.
    - Local Pheromone Update: After each move, the pheromone level on the traversed edge is updated locally to encourage other ants to explore new paths.

    How It Enhances Performance:
    - The pseudorandom proportional rule guides ants towards promising solutions while maintaining diversity in the search space.
    - Local pheromone updating reduces the attractiveness of recently used edges, promoting exploration and preventing premature convergence.

    Based on State-of-the-Art Technique:
    - Ant Colony System (ACS) algorithm as introduced by Dorigo and Gambardella.

    Parameters:
    - distance_matrix: The distance matrix.
    - h: Heuristic matrix (attractiveness) calculated from the distance matrix.
    - thau: Pheromone matrix.
    - alpha: Exponent for pheromone importance.
    - beta: Exponent for heuristic importance.
    - full_list: List of all cities.
    - ants: Number of ants.
    - xi: Local pheromone decay coefficient.
    - q0: Threshold for pseudorandom proportional rule.
    - tau0: Initial pheromone value.
    - local_search: Whether to perform local search (2-opt).

    Returns:
    - best_city_list: The best tour found by the ants in this iteration.
    - best_path_distance: The distance of the best tour found.
    - thau: Updated pheromone matrix after local updates.
    """
    best_path_distance = float('inf')
    best_city_list     = None
    n = distance_matrix.shape[0]
    for _ in range(ants):
        city_list = [np.random.choice(full_list)]
        unvisited = set(full_list)
        unvisited.remove(city_list[0])
        while unvisited:
            current_city = city_list[-1]
            q = np.random.rand()
            if q <= q0:
                # Exploitation: select the best edge
                probabilities = {}
                for next_city in unvisited:
                    tau_eta = thau[current_city-1, next_city-1] ** alpha * h[current_city-1, next_city-1] ** beta
                    probabilities[next_city] = tau_eta
                next_city = max(probabilities, key=probabilities.get)
            else:
                # Exploration: select next city based on probabilities
                probabilities = []
                denom = 0.0
                for next_city in unvisited:
                    tau_eta = thau[current_city-1, next_city-1] ** alpha * h[current_city-1, next_city-1] ** beta
                    probabilities.append((next_city, tau_eta))
                    denom += tau_eta
                probs = [p[1]/denom for p in probabilities]
                choices = [p[0] for p in probabilities]
                next_city = np.random.choice(choices, p=probs)
            city_list.append(next_city)
            unvisited.remove(next_city)
            # Local pheromone update
            thau[current_city-1, next_city-1] = (1 - xi) * thau[current_city-1, next_city-1] + xi * tau0
            thau[next_city-1, current_city-1] = thau[current_city-1, next_city-1]  # Symmetric TSP
        # Compute path distance
        path_distance = calculate_distance(distance_matrix, city_list)
        if path_distance < best_path_distance:
            best_city_list = copy.deepcopy(city_list)
            best_path_distance = path_distance
    if local_search:
        best_city_list, best_path_distance = local_search_2_opt(distance_matrix, [best_city_list, best_path_distance])
    return best_city_list, best_path_distance, thau

def global_pheromone_update(thau, best_city_list, decay, best_path_distance):
    """
    Performs global pheromone updating on the pheromone matrix using the best-so-far path.

    Improvements Implemented:
    - Updates pheromone levels only on the edges of the best-so-far path to intensify search on promising solutions.

    How It Enhances Performance:
    - Focuses the search around the best solutions found, accelerating the convergence towards optimal solutions.

    Based on State-of-the-Art Technique:
    - Ant Colony System (ACS) algorithm's global pheromone update rule.

    Parameters:
    - thau: Pheromone matrix.
    - best_city_list: The best tour found so far.
    - decay: Global pheromone evaporation rate.
    - best_path_distance: The length of the best tour.

    Returns:
    - thau: Updated pheromone matrix after global update.
    """
    # Evaporation
    thau *= (1 - decay)
    # Deposit pheromone on best route
    delta_tau = 1.0 / best_path_distance
    for i in range(len(best_city_list)-1):
        from_city = best_city_list[i] - 1
        to_city = best_city_list[i+1] - 1
        thau[from_city, to_city] += decay * delta_tau
        thau[to_city, from_city] = thau[from_city, to_city]  # Symmetric TSP
    # Include edge returning to starting city
    from_city = best_city_list[-1] - 1
    to_city = best_city_list[0] -1
    thau[from_city, to_city] += decay * delta_tau
    thau[to_city, from_city] = thau[from_city, to_city]
    return thau

def ant_colony_optimization(distance_matrix, ants=5, iterations=50, alpha=1, beta=2, decay=0.05, time_limit=10, local_search=True, verbose=True):
    """
    Ant Colony Optimization function enhanced with Ant Colony System (ACS) features to improve solution quality and convergence speed.

    Improvements Implemented:
    - Pseudorandom Proportional Rule: Implements the pseudorandom proportional rule for next city selection, balancing exploration and exploitation.
    - Local Pheromone Update: Applies local pheromone updating during solution construction to encourage exploration of new paths.
    - Global Pheromone Update on Best-so-far Path: Updates pheromones globally only on the edges of the best-so-far path to intensify the search around the best solutions.

    How It Enhances Performance:
    - The pseudorandom proportional rule allows ants to make more informed choices, leading to better paths.
    - Local pheromone updating diversifies the search by decreasing pheromone levels on recently used edges.
    - Global updating on the best-so-far path focuses the search on promising areas, accelerating convergence.

    Based on State-of-the-Art Technique:
    - Ant Colony System (ACS) algorithm proposed by Dorigo and Gambardella, which improves upon the basic Ant System by incorporating the above enhancements.

    Parameters:
    - distance_matrix: A 2D numpy array representing distances between cities.
    - ants: Number of ants to use.
    - iterations: Number of iterations to perform.
    - alpha: Relative importance of pheromone.
    - beta: Relative importance of heuristic information.
    - decay: Pheromone evaporation rate.
    - local_search: Whether to perform local search (2-opt) on the constructed paths.
    - verbose: Whether to print information during iterations.

    Returns:
    - best_route: The best tour found.
    - best_distance: The distance of the best tour.
    """
    import time
    start_time = time.time()
    xi = 0.1  # Local pheromone decay coefficient
    q0 = 0.9  # Threshold for pseudorandom proportional rule
    n = distance_matrix.shape[0]
    average_distance = np.mean(distance_matrix[distance_matrix != 0])
    tau0 = 1 / (n * average_distance)
    count = 0
    full_list = list(range(1, n +1))
    h = attractiveness(distance_matrix)
    thau = np.full((n, n), tau0)
    best_route = None
    best_distance = float('inf')
    while count <= iterations:
        city_list, path_distance, thau = acs_ants_path(distance_matrix, h, thau, alpha, beta, full_list, ants, xi, q0, tau0, local_search)
        if path_distance < best_distance and validate_solution(distance_matrix, city_list, path_distance):
            best_route = copy.deepcopy(city_list)
            best_distance = path_distance
        thau = global_pheromone_update(thau, best_route, decay, best_distance)
        if verbose and count > 0:
            print(f'Iteration = {count}, Distance = {round(best_distance, 2)}')
        count +=1
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break
    best_route.append(best_route[0])  # Return to the starting city

    if not validate_solution(distance_matrix, best_route, best_distance):
        raise ValueError("La solución obtenida no es válida.")
    return best_route, best_distance

# Unmodified functions from the original code:
# - calculate_distance
# - local_search_2_opt
# - attractiveness

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
