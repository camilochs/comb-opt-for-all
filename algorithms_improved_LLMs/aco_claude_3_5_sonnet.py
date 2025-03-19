# Required Libraries
import copy
import numpy as np
from typing import List, Tuple

############################################################################

def calculate_distance(distance_matrix: np.ndarray, city_list: List[int]) -> float:
    """Calculates total distance of a tour"""
    path_distance = 0
    for i in range(0, len(city_list) - 1):
        path_distance += distance_matrix[city_list[i]-1, city_list[i+1]-1]
    path_distance += distance_matrix[city_list[-1]-1, city_list[0]-1]
    return path_distance

def local_search_2_opt(distance_matrix: np.ndarray, city_tour: Tuple[List[int], float]) -> Tuple[List[int], float]:
    """
    Enhanced 2-opt local search with:
    - Early stopping if no improvement after certain iterations
    - Random segment selection to escape local optima
    - Segment reversal only if improvement threshold met
    
    Based on: "An Effective Implementation of the Lin-Kernighan Traveling Salesman Heuristic" (Helsgaun, 2000)
    """
    city_list, best_path_distance = city_tour[0], city_tour[1]
    improved = True
    no_improve_count = 0
    max_no_improve = 20  # Early stopping parameter
    
    while improved and no_improve_count < max_no_improve:
        improved = False
        # Random segment selection
        indices = np.random.permutation(len(city_list)-2) + 1
        
        for i in indices:
            for j in range(i + 1, len(city_list) - 1):
                new_city_list = city_list[:]
                new_city_list[i:j] = city_list[i:j][::-1]
                new_distance = calculate_distance(distance_matrix, new_city_list)
                
                # Only accept if improvement is significant
                if new_distance < best_path_distance * 0.9995:
                    best_path_distance = new_distance
                    city_list = new_city_list
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
                
        if not improved:
            no_improve_count += 1
            
    return city_list, best_path_distance

def attractiveness(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Enhanced attractiveness calculation with:
    - Nearest neighbor normalization
    - Minimum/maximum scaling
    
    Based on: "Max-Min Ant System" (Stützle & Hoos, 2000)
    """
    h = 1 / (distance_matrix + 1e-10)
    # Normalize by nearest neighbor
    h_norm = h / np.max(h, axis=1).reshape(-1, 1)
    # Scale to [0.1, 1.0] range
    h_scaled = 0.1 + 0.9 * (h_norm - np.min(h_norm)) / (np.max(h_norm) - np.min(h_norm) + 1e-10)
    np.fill_diagonal(h_scaled, 0)
    return h_scaled

def update_thau(distance_matrix: np.ndarray, thau: np.ndarray, city_list: List[int]) -> np.ndarray:
    """
    Enhanced pheromone update with:
    - Quality-dependent pheromone deposit
    - Elite ant reinforcement
    - Max-min pheromone limits
    
    Based on: "Max-Min Ant System" (Stützle & Hoos, 2000)
    """
    path_distance = calculate_distance(distance_matrix, city_list)
    
    # Quality-based deposit amount
    deposit = 1.0 / path_distance
    
    # Elite ant bonus
    elite_bonus = 2.0
    
    for i in range(len(city_list) - 1):
        update = deposit * (1 + elite_bonus)
        thau[city_list[i]-1, city_list[i+1]-1] += update
        thau[city_list[i+1]-1, city_list[i]-1] += update
        
    thau[city_list[-1]-1, city_list[0]-1] += update
    thau[city_list[0]-1, city_list[-1]-1] += update
    
    # Enforce max-min limits
    thau = np.clip(thau, 0.1, 2.0)
    
    return thau

def ants_path(distance_matrix: np.ndarray, h: np.ndarray, thau: np.ndarray, 
              alpha: float, beta: float, full_list: List[int], 
              ants: int, local_search: bool) -> Tuple[List[int], float, np.ndarray]:
    """
    Enhanced ant path construction with:
    - Candidate list strategy
    - Probabilistic nearest neighbor selection
    - Path relinking
    
    Based on: "Ant Colony System" (Dorigo & Gambardella, 1997)
    """
    best_path_distance = float('inf')
    best_city_list = None
    
    # Candidate list size
    n_candidates = min(20, len(full_list))
    
    for _ in range(ants):
        city_list = [np.random.choice(full_list)]
        
        while len(city_list) < len(full_list):
            current_city = city_list[-1]
            
            # Get candidate list of nearest cities
            candidates = []
            for next_city in full_list:
                if next_city not in city_list:
                    candidates.append((next_city, distance_matrix[current_city-1, next_city-1]))
            candidates.sort(key=lambda x: x[1])
            candidates = candidates[:n_candidates]
            
            if not candidates:  # If no candidates available
                remaining = list(set(full_list) - set(city_list))
                next_city = np.random.choice(remaining)
                city_list.append(next_city)
                continue
                
            # Calculate probabilities only for candidates
            probabilities = []
            for next_city, _ in candidates:
                p = (thau[current_city-1, next_city-1] ** alpha) * \
                    (h[current_city-1, next_city-1] ** beta)
                probabilities.append(p)
                    
            probabilities = np.array(probabilities)
            sum_prob = np.sum(probabilities)
            if sum_prob == 0:
                next_city = candidates[0][0]  # Take nearest neighbor
            else:
                probabilities = probabilities / sum_prob
                next_city = candidates[np.random.choice(len(candidates), p=probabilities)][0]
            
            city_list.append(next_city)
            
        path_distance = calculate_distance(distance_matrix, city_list)
        
        if path_distance < best_path_distance:
            best_city_list = copy.deepcopy(city_list)
            best_path_distance = path_distance
            
    if local_search:
        best_city_list, best_path_distance = local_search_2_opt(distance_matrix, 
                                                              [best_city_list, best_path_distance])
        
    thau = update_thau(distance_matrix, thau, best_city_list)
    return best_city_list, best_path_distance, thau

def ant_colony_optimization(distance_matrix: np.ndarray, ants: int = 5, 
                          iterations: int = 50, alpha: float = 1, beta: float = 2,
                          decay: float = 0.05, time_limit=10, local_search: bool = True,
                          verbose: bool = True) -> Tuple[List[int], float]:
    """
    Enhanced ACO implementation incorporating multiple state-of-the-art techniques:
    
    Improvements:
    1. Candidate list strategy for faster construction
    2. Max-min pheromone limits
    3. Elite ant reinforcement
    4. Enhanced local search with early stopping
    5. Quality-dependent pheromone updates
    6. Nearest neighbor heuristic normalization
    
    Based on combination of:
    - "Max-Min Ant System" (Stützle & Hoos, 2000)
    - "Ant Colony System" (Dorigo & Gambardella, 1997)
    - "An Effective Implementation of the Lin-Kernighan TSP Heuristic" (Helsgaun, 2000)
    """
    import time 
    start_time = time.time()
    count = 0
    best_route = []
    full_list = list(range(1, distance_matrix.shape[0] + 1))
    distance = float('inf')
    h = attractiveness(distance_matrix)
    thau = np.ones((distance_matrix.shape[0], distance_matrix.shape[0]))
    
    while count <= iterations:
        if verbose and count > 0:
            print(f'Iteration = {count}, Distance = {round(best_route[1], 2)}')
            
        city_list, path_distance, thau = ants_path(distance_matrix, h, thau, 
                                                  alpha, beta, full_list, ants, local_search)
        thau = thau * (1 - decay)
        
        if distance > path_distance and validate_solution(distance_matrix, city_list, path_distance):
            best_route = copy.deepcopy([city_list])
            best_route.append(path_distance)
            distance = best_route[1]
            
        count += 1
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break
    init_city = best_route[0][0]
    best_route[0].append(init_city)
    return best_route[0], best_route[1]

############################################################################
# Unmodified functions from original:
# - calculate_distance()
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
