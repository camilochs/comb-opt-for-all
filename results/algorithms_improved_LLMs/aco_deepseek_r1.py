
# Required Libraries
import copy
import numpy as np

############################################################################

# Function: Calculate Path Distance
def calculate_distance(distance_matrix, city_list):
    """Calculate the total distance of a given TSP tour.
    Includes return to the starting city for a closed loop.
    Ensures accurate tour length calculation for evaluation."""
    path_distance = 0
    for i in range(0, len(city_list) - 1):
        path_distance += distance_matrix[city_list[i]-1, city_list[i+1]-1]
    path_distance += distance_matrix[city_list[-1]-1, city_list[0]-1]
    return path_distance

def local_search_2_opt(distance_matrix, city_tour):
    """Mejora el recorrido utilizando la búsqueda local 2-opt.
    Invierte segmentos del recorrido para reducir la distancia total."""
    city_list, best_distance = city_tour[0], city_tour[1]
    improved = True
    while improved:
        improved = False
        for i in range(len(city_list) - 1):
            for j in range(i + 2, len(city_list)):
                if i == 0 and j == len(city_list) - 1:
                    continue  # Evita invertir el recorrido completo
                new_city_list = city_list[:i] + city_list[i:j][::-1] + city_list[j:]
                new_distance = calculate_distance(distance_matrix, new_city_list)
                if new_distance < best_distance:
                    city_list, best_distance = new_city_list, new_distance
                    improved = True
        if improved:
            break  # Si se encuentra una mejoría, reiniciar la búsqueda
    return city_list, best_distance

############################################################################

# Function: Calculate Attractiveness
def attractiveness(distance_matrix):
    """Compute heuristic attractiveness as inverse of distance matrix.
    Avoids division by zero via small epsilon. Critical for probabilistic path selection."""
    h = 1 / (distance_matrix + 1e-10)
    np.fill_diagonal(h, 0)
    return h

# Function: Update Pheromone Matrix with Bounds
def update_thau(distance_matrix, thau, city_list, tau_min=1e-5, tau_max=1e5):
    """Update pheromone matrix using global-best path with pheromone clamping.
    Prevents stagnation by limiting pheromone levels (MMAS technique).
    Deposition scale based on global-best path's quality."""
    path_distance = calculate_distance(distance_matrix, city_list)
    delta = 1.0 / path_distance
    for i in range(len(city_list)-1):
        thau[city_list[i]-1, city_list[i+1]-1] += delta
        thau[city_list[i+1]-1, city_list[i]-1] += delta
    thau[city_list[0]-1, city_list[-1]-1] += delta
    thau[city_list[-1]-1, city_list[0]-1] += delta
    thau = np.clip(thau, tau_min, tau_max)  # Pheromone clamping (MMAS)
    return thau

def ants_path(distance_matrix, h, thau, alpha, beta, full_list, ants, local_search, candidate_dict):
    """Genera recorridos de hormigas utilizando listas de candidatos y la regla de proporción seudoaleatoria.
    Se asegura de que cada hormiga visite todas las ciudades exactamente una vez."""
    best_path_distance = float('inf')
    best_city_list = None
    q0 = 0.9  # Probabilidad de explotación
    for _ in range(ants):
        city_list = [np.random.choice(full_list)]
        while len(city_list) < len(full_list):
            current_city = city_list[-1]
            candidates = candidate_dict[current_city]
            available = [c for c in candidates if c not in city_list]
            if not available:
                # Si todos los candidatos han sido visitados, considerar todas las ciudades no visitadas
                available = [c for c in full_list if c not in city_list]
            if not available:
                # Todas las ciudades han sido visitadas; romper el ciclo
                break
            q = np.random.random()
            if q <= q0:  # Paso de explotación
                max_p = -1
                next_city = None
                for c in available:
                    p_val = (thau[current_city-1, c-1] ** alpha) * (h[current_city-1, c-1] ** beta)
                    if p_val > max_p:
                        max_p = p_val
                        next_city = c
                if next_city is None:
                    # Si todas las probabilidades son cero, seleccionar aleatoriamente
                    next_city = np.random.choice(available)
                city_list.append(next_city)
            else:  # Paso de exploración con distribución de probabilidades
                p = np.array([(thau[current_city-1, c-1] ** alpha) * (h[current_city-1, c-1] ** beta) for c in available])
                p_total = p.sum()
                if p_total == 0:
                    p = np.ones(len(available))/len(available)
                else:
                    p = p / p_total
                next_city = np.random.choice(available, p=p)
                city_list.append(next_city)
        path_distance = calculate_distance(distance_matrix, city_list)
        if path_distance < best_path_distance:
            best_city_list, best_path_distance = copy.deepcopy(city_list), path_distance
    if local_search and best_city_list is not None:
        best_city_list, best_path_distance = local_search_2_opt(distance_matrix, (best_city_list, best_path_distance))
    return best_city_list, best_path_distance, thau

############################################################################

# ACO Function with Enhanced Features
def ant_colony_optimization(distance_matrix, ants=5, iterations=50, alpha=1, beta=2, decay=0.05, time_limit=10, local_search=True, verbose=True):
    """Optimized ACO with pheromone clamping, candidate lists, and ACS-inspired selection.
    Implements candidate lists for faster path construction and MMAS-based pheromone management.
    Evaporate pheromones before deposition (correct order) for proper exploration-exploitation balance."""
    import time
    start_time = time.time()
    count = 0
    best_route = []
    full_list = list(range(1, distance_matrix.shape[0]+1))
    h = attractiveness(distance_matrix)
    thau = np.ones((distance_matrix.shape[0], distance_matrix.shape[0]))
    candidate_dict = {}
    candidate_list_size = min(20, len(full_list)-1)
    for city in full_list:
        sorted_indices = np.argsort(distance_matrix[city-1])  # Closest cities first
        candidate_dict[city] = [idx+1 for idx in sorted_indices if idx+1 != city][:candidate_list_size]
    global_best_distance = np.inf
    while count <= iterations:
        thau *= (1 - decay)  # Evaporation before deposition (correct order)
        city_list, path_distance, thau = ants_path(distance_matrix, h, thau, alpha, beta, full_list, ants, local_search, candidate_dict)
        if verbose and count > 0:
            print(f'Iteration = {count}, Distance = {round(best_route[1], 2)}')
        if path_distance < global_best_distance and validate_solution(distance_matrix, city_list, path_distance):
            best_route = [copy.deepcopy(city_list), path_distance]
            global_best_distance = path_distance
            thau = update_thau(distance_matrix, thau, best_route[0])  # Elite global update

        count += 1
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break
    best_route[0].append(best_route[0][0])  # Complete the cycle
    if not validate_solution(distance_matrix, best_route[0], best_route[1]):
        raise ValueError("La solución obtenida no es válida.")
    return best_route[0], best_route[1]

# Unmodified from original code: calculate_distance, local_search_2_opt

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
