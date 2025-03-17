# Required Libraries
import copy
import numpy as np
import random

from typing import List, Tuple
############################################################################

# Function: Euclidean Distance
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: Tour Distance
def distance_point(distance_matrix, city_tour):
    distance = 0
    for i in range(0, len(city_tour) - 1):
        distance = distance + distance_matrix[city_tour[i]][city_tour[i + 1]]
    distance = distance + distance_matrix[city_tour[-1]][city_tour[0]]
    return distance

############################################################################

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, verbose = True):
    if (recursive_seeding < 0):
        count = -2
    else:
        count = 0
    city_list = copy.deepcopy(city_tour)
    distance  = city_list[1]*2
    iteration = 0
    if (verbose == True):
        print('')
        print('Local Search')
        print('')
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
             count             = -2
             recursive_seeding = -1
        elif(city_list[1] >= distance and recursive_seeding < 0):
            count              = -1
            recursive_seeding  = -2
    return city_list[0], city_list[1]

############################################################################

# Function: Removal
def removal_operators(distance_matrix):
    """
    Defines the removal operators used in the ALNS algorithm.

    Returns:
        list: A list of removal operator functions.
    """
    def random_removal(city_tour, num_removals):
        """
        Randomly removes a specified number of cities from the tour.

        Args:
            city_tour (list): The current tour.
            num_removals (int): The number of cities to remove.

        Returns:
            list: A list of removed cities.
        """
        removed = set()
        while (len(removed) < num_removals):
            removed.add(random.choice(city_tour[1:]))
        return list(removed)

    def worst_distance_removal(city_tour, num_removals):
        """
        Removes cities that contribute the most to the tour distance.

        This is based on the idea that removing high-cost edges can lead to better solutions.

        Args:
            city_tour (list): The current tour.
            num_removals (int): The number of cities to remove.

        Returns:
            list: A list of removed cities.
        """
        
        worst_edges = []
        for i in range(1, len(city_tour)):
            cost = distance_matrix[city_tour[i-1]][city_tour[i]]
            worst_edges.append((cost, city_tour[i]))
        
        worst_edges.sort(reverse=True)
        removed = [city for cost, city in worst_edges[:num_removals]]
        
        return removed

    def shaw_removal(city_tour, num_removals):
        """
        Implements the Shaw removal operator, which removes cities that are related to each other.

        Relatedness is based on distance. Cities that are close to each other are considered more related.
        This is based on the Shaw removal operator proposed in:
        Shaw, P. (1998). Using constraint programming and local search methods to solve vehicle routing problems.

        Args:
            city_tour (list): The current tour.
            num_removals (int): The number of cities to remove.

        Returns:
            list: A list of removed cities.
        """
        
        if not city_tour:
            return []
        
        removed = []
        remaining = city_tour[1:].copy()
        
        r = random.choice(remaining)
        removed.append(r)
        remaining.remove(r)
        
        while len(removed) < num_removals and remaining:
            
            relatedness = [(distance_matrix[r][c], c) for c in remaining]
            relatedness.sort()
            
            p = random.random()
            
            # Deterministic component in Shaw removal
            d = 3
            
            index = min(int(abs(random.gauss(0,1))**d), len(relatedness) -1)
            
            r = relatedness[index][1]
            removed.append(r)
            remaining.remove(r)
        
        return removed
    
    return [random_removal, worst_distance_removal, shaw_removal]

# Function: Insertion
def insertion_operators(distance_matrix):
    """
    Defines the insertion operators used in the ALNS algorithm.

    Returns:
        list: A list of insertion operator functions.
    """
    def cheapest_insertion(removed_nodes, city_tour, distance_matrix):
        """
        Inserts each removed city into the tour at the position that results in the smallest increase in tour distance.

        Args:
            removed_nodes (list): The list of removed cities.
            city_tour (list): The current tour.
            distance_matrix (np.ndarray): The distance matrix.

        Returns:
            list: The updated tour.
        """
        for node in removed_nodes:
            best_insertion_cost  = float('inf')
            best_insertion_index = -1
            for i in range(1, len(city_tour) + 1):
                insertion_cost = (distance_matrix[city_tour[i - 1]][node] + distance_matrix[node][city_tour[i % len(city_tour)]] - distance_matrix[city_tour[i - 1]][city_tour[i % len(city_tour)]])
                if (insertion_cost < best_insertion_cost):
                    best_insertion_cost  = insertion_cost
                    best_insertion_index = i
            city_tour.insert(best_insertion_index, node)
        return city_tour

    def regret_insertion(removed_nodes, city_tour, distance_matrix):
        """
        Implements regret insertion, which considers the regret value of not inserting a city at its best position.

        The regret value is the difference between the second-best insertion cost and the best insertion cost.
        This encourages the algorithm to make decisions that minimize potential future regret.
        This is based on the regret heuristics described in:
        Potvin, J. Y., & Rousseau, J. M. (1993). A parallel route building algorithm for the vehicle routing and scheduling problem with time windows.

        Args:
            removed_nodes (list): The list of removed cities.
            city_tour (list): The current tour.
            distance_matrix (np.ndarray): The distance matrix.

        Returns:
            list: The updated tour.
        """
        
        for node in removed_nodes:
            
            best_insertion_cost = float('inf')
            second_best_insertion_cost = float('inf')
            best_insertion_index = -1
            
            for i in range(1, len(city_tour) + 1):
                insertion_cost = (distance_matrix[city_tour[i-1]][node] + distance_matrix[node][city_tour[i % len(city_tour)]] - distance_matrix[city_tour[i-1]][city_tour[i % len(city_tour)]])
                
                if insertion_cost < best_insertion_cost:
                    second_best_insertion_cost = best_insertion_cost
                    best_insertion_cost = insertion_cost
                    best_insertion_index = i
                elif insertion_cost < second_best_insertion_cost:
                    second_best_insertion_cost = insertion_cost
            
            
            regret_value = second_best_insertion_cost - best_insertion_cost
            
            if best_insertion_index != -1:
                city_tour.insert(best_insertion_index, node)
            else:
                city_tour.append(node)
        
        return city_tour

    return [cheapest_insertion, regret_insertion]

############################################################################

# Function: Adaptive Large Neighborhood Search
def adaptive_large_neighborhood_search(distance_matrix, iterations = 100, removal_fraction = 0.2, rho = 0.1, time_limit=10, best=None, local_search = True, verbose = True):
    """
    Implements the Adaptive Large Neighborhood Search (ALNS) algorithm for the Traveling Salesperson Problem (TSP).

    ALNS is a metaheuristic that uses multiple removal and insertion operators to explore the solution space.
    It adaptively adjusts the probability of selecting each operator based on its past performance.

    Args:
        distance_matrix (np.ndarray): A symmetric matrix representing the distances between cities.
        iterations (int, optional): The number of iterations to run the algorithm. Defaults to 100.
        removal_fraction (float, optional): The fraction of cities to remove in each iteration. Defaults to 0.2.
        rho (float, optional): The reaction factor, which controls how quickly the operator weights are updated. Defaults to 0.1.
        local_search (bool, optional): Whether to apply 2-opt local search at the end. Defaults to True.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        tuple: A tuple containing the best tour found (list) and its distance (float).
    """
    import time
    start_time = time.time()
    initial_tour       = list(range(0, distance_matrix.shape[0]))
    random.shuffle(initial_tour)
    route              = initial_tour.copy()
    distance           = distance_point(distance_matrix, route)
    best_route         = route.copy()
    best_distance      = distance
    removal_ops        = removal_operators(distance_matrix)
    insertion_ops      = insertion_operators(distance_matrix)
    weights_removal    = [1.0] * len(removal_ops)
    weights_insertion  = [1.0] * len(insertion_ops)
    
    # Segment length for updating operator scores
    segment_length = 50
    
    # Scores for each operator, reset every segment
    scores_removal = [0] * len(removal_ops)
    scores_insertion = [0] * len(insertion_ops)
    
    # Usage count for each operator, reset every segment
    usage_removal = [0] * len(removal_ops)
    usage_insertion = [0] * len(insertion_ops)
    
    invalid_solutions_count = 0
    repaired_solutions_count = 0
    count              = 0
    best.put(best_distance)
    while (count <= iterations):
        if (verbose == True and count > 0):
            print('Iteration = ', count, 'Distance = ', round(distance, 2), 'Best Distance = ', round(best_distance, 2))     
        city_tour     = route.copy()
        removal_op_index    = random.choices(range(len(removal_ops)),   weights = weights_removal)[0]
        insertion_op_index  = random.choices(range(len(insertion_ops)), weights = weights_insertion)[0]
        
        removal_op = removal_ops[removal_op_index]
        insertion_op = insertion_ops[insertion_op_index]
        
        num_removals  = int(removal_fraction * distance_matrix.shape[0])
        removed_nodes = removal_op(city_tour, num_removals)
        for node in removed_nodes:
            city_tour.remove(node)
        new_tour          = insertion_op(removed_nodes, city_tour, distance_matrix)
        
        if new_tour[0] != new_tour[-1]:
            new_tour = new_tour + [new_tour[0]]
            
        # Validar la nueva solución antes de aceptarla
        is_valid, validation_message = validate_solution(new_tour, distance_matrix)
        
        if not is_valid:
            invalid_solutions_count += 1
            new_tour, repair_success = repair_solution(new_tour, distance_matrix)
            
            if repair_success:
                repaired_solutions_count += 1
                is_valid = True
                if verbose:
                    print(f"Successfully repaired solution at iteration {count}")
            else:
                if verbose:
                    print(f"Could not repair solution at iteration {count}")
                continue

        new_tour_distance = distance_point(distance_matrix, new_tour)
        
        usage_removal[removal_op_index] += 1
        usage_insertion[insertion_op_index] += 1
        
        # Update operator scores based on solution improvement
        if (new_tour_distance < best_distance):
            best_route = new_tour.copy()
            best_distance = new_tour_distance
            best.put(best_distance)
            time.sleep(0.1)
            scores_removal[removal_op_index] += 3
            scores_insertion[insertion_op_index] += 3
            route = new_tour.copy()
            distance = new_tour_distance
        elif (new_tour_distance < distance):
            route = new_tour.copy()
            distance = new_tour_distance
            scores_removal[removal_op_index] += 2
            scores_insertion[insertion_op_index] += 2
        else:
            scores_removal[removal_op_index] += 1
            scores_insertion[insertion_op_index] += 1
        
        # Update operator weights every segment
        if (count % segment_length == 0 and count > 0):
            for i in range(len(removal_ops)):
                if usage_removal[i] > 0:
                    weights_removal[i] = weights_removal[i] * (1-rho) + rho * (scores_removal[i] / usage_removal[i])
                else:
                    weights_removal[i] *= (1-rho)
            for i in range(len(insertion_ops)):
                if usage_insertion[i] > 0:
                    weights_insertion[i] = weights_insertion[i] * (1-rho) + rho * (scores_insertion[i] / usage_insertion[i])
                else:
                    weights_insertion[i] *= (1-rho)
            
            # Normalize weights
            total_weight_removal = sum(weights_removal)
            total_weight_insertion = sum(weights_insertion)
            weights_removal = [w / total_weight_removal for w in weights_removal]
            weights_insertion = [w / total_weight_insertion for w in weights_insertion]
            
            # Reset scores and usage counts
            scores_removal = [0] * len(removal_ops)
            scores_insertion = [0] * len(insertion_ops)
            usage_removal = [0] * len(removal_ops)
            usage_insertion = [0] * len(insertion_ops)
            
        count = count + 1
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break  
        
    best_route = best_route + [best_route[0]]
    best_route = [item + 1 for item in best_route]
    if (local_search == True):
        best_route, best_distance = local_search_2_opt(distance_matrix, [best_route, best_distance], -1, verbose)
        best.put(best_distance)
        time.sleep(0.1)
    return best_route, best_distance

############################################################################

# Unmodified functions:
# - euclidean_distance
# - distance_calc
# - distance_point
# - local_search_2_opt
def validate_solution(tour, distance_matrix):
    """
    Validates if a tour is a valid solution for the TSP.
    
    Args:
        tour (list): The tour to validate
        distance_matrix (np.ndarray): The distance matrix
    
    Returns:
        tuple: (is_valid, message) where is_valid is a boolean and message describes any issues
    """
    # Verificar si el tour está vacío
    if not tour:
        return False, "Tour está vacío"
    
    # Verificar si el tour comienza y termina en la misma ciudad
    if tour[0] != tour[-1]:
        return False, "Tour no comienza y termina en la misma ciudad"
    
    # Verificar si todas las ciudades están presentes exactamente una vez (excepto la primera/última)
    n = len(distance_matrix)
    cities = set(tour[:-1])  # Excluimos la última ciudad ya que es igual a la primera
    
    if len(cities) != n:
        return False, f"No todas las ciudades están presentes exactamente una vez. Encontradas {len(cities)}, esperadas {n}"
    
    # Verificar si todas las ciudades están dentro del rango válido
    if any(city >= n or city < 0 for city in tour):
        return False, "Algunas ciudades están fuera del rango válido"
    
    # Verificar si hay conexiones válidas entre todas las ciudades
    for i in range(len(tour)-1):
        if distance_matrix[tour[i]][tour[i+1]] == float('inf'):
            return False, f"Conexión inválida entre ciudades {tour[i]} y {tour[i+1]}"
    
    return True, "Tour válido"

def repair_solution(tour, distance_matrix):
    """
    Intenta reparar una solución inválida del TSP.
    
    Args:
        tour (list): El tour a reparar
        distance_matrix (np.ndarray): La matriz de distancias
    
    Returns:
        tuple: (repaired_tour, success) donde success es un booleano que indica si la reparación fue exitosa
    """
    n = len(distance_matrix)
    repaired_tour = tour.copy()
    
    # Si el tour está vacío, crear uno nuevo
    if not repaired_tour:
        repaired_tour = list(range(n))
        random.shuffle(repaired_tour)
        repaired_tour.append(repaired_tour[0])
        return repaired_tour, True
    
    # Asegurarse de que todas las ciudades estén presentes
    cities_needed = set(range(n))
    cities_present = set(repaired_tour[:-1])  # Excluimos la última ciudad
    
    # Agregar ciudades faltantes
    missing_cities = cities_needed - cities_present
    for city in missing_cities:
        # Encontrar la mejor posición para insertar la ciudad faltante
        best_cost = float('inf')
        best_pos = 1  # Empezamos en 1 para mantener la primera ciudad fija
        
        for pos in range(1, len(repaired_tour)):
            # Calcular el costo de inserción
            prev_city = repaired_tour[pos-1]
            next_city = repaired_tour[pos]
            insertion_cost = (
                distance_matrix[prev_city][city] +
                distance_matrix[city][next_city] -
                distance_matrix[prev_city][next_city]
            )
            
            if insertion_cost < best_cost:
                best_cost = insertion_cost
                best_pos = pos
        
        repaired_tour.insert(best_pos, city)
    
    # Eliminar ciudades duplicadas (manteniendo la primera ocurrencia)
    seen = set()
    unique_tour = []
    for city in repaired_tour[:-1]:  # Excluimos la última ciudad
        if city not in seen:
            seen.add(city)
            unique_tour.append(city)
    
    # Asegurarse de que el tour termine donde comenzó
    unique_tour.append(unique_tour[0])
    
    # Verificar si la reparación fue exitosa
    is_valid, _ = validate_solution(unique_tour, distance_matrix)
    
    return unique_tour, is_valid
