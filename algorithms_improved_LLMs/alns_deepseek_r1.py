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
# Function: 2-opt (Corrected Input Handling)
def local_search_2_opt(distance_matrix, city_tour, max_attempts=20, verbose=True):
    """
    Fixed 2-opt implementation with:
    1. Proper input format validation
    2. Correct tour list handling
    3. Removed erroneous list comprehension
    4. Direct route processing
    Fixes 'int object is not iterable' error by correctly handling tour structure.
    """
    # Validate and parse input format
    if isinstance(city_tour, tuple) and len(city_tour) == 2:
        route, current_distance = city_tour
        # Ensure route is a proper list (not single integer)
        if not isinstance(route, list) or len(route) < 2:
            return [route] + [route[0]], current_distance  # Handle single-node edge case
    else:
        route = city_tour.copy()
        current_distance = distance_point(distance_matrix, route)
    
    # Remove duplicate end node if present
    if len(route) > 1 and route[-1] == route[0]:
        route = route[:-1]
    
    n = len(route)
    improved = True
    attempts = 0
    original_distance = current_distance
    
    # Main 2-opt loop remains unchanged
    while improved and attempts < max_attempts:
        improved = False
        best_delta = 0
        for i in range(n-1):
            for j in range(i+2, n):
                a, b, c, d = route[i], route[(i+1)%n], route[j], route[(j+1)%n]
                
                delta = (distance_matrix[a][c] + distance_matrix[b][d]) - (distance_matrix[a][b] + distance_matrix[c][d])
                
                if delta < -1e-9:
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    if len(new_route) != n:
                        continue
                    
                    new_distance = current_distance + delta
                    if new_distance < current_distance:
                        route = new_route
                        current_distance = new_distance
                        improved = True
                        attempts = 0
                        break
            if improved:
                break
        attempts += not improved

    # Final validation and return
    final_route = route + [route[0]]
    final_distance = distance_point(distance_matrix, final_route)
    return final_route, final_distance

# Function: Removal Operators
def removal_operators():
    """
    Enhanced removal operators including worst and related removal.
    Worst removal targets high-cost nodes using contribution calculations.
    Related removal clusters geographically close nodes for coherent destruction.
    Based on Ropke & Pisinger's adaptive large neighborhood search framework.
    """
    def random_removal(city_tour: List[int], num_removals: int, distance_matrix: np.ndarray) -> List[int]:
        """Random removal with guaranteed unique node selection"""
        return random.sample(city_tour, num_removals)

    def worst_removal(city_tour: List[int], num_removals: int, distance_matrix: np.ndarray) -> List[int]:
        """Removes nodes with highest individual contribution to tour cost"""
        contributions = []
        for i, node in enumerate(city_tour):
            prev = city_tour[i-1]
            next_node = city_tour[(i+1)%len(city_tour)]
            delta = distance_matrix[prev][node] + distance_matrix[node][next_node] - distance_matrix[prev][next_node]
            contributions.append((node, delta))
        contributions.sort(key=lambda x: -x[1])
        return [node for node, _ in contributions[:num_removals]]

    def related_removal(city_tour: List[int], num_removals: int, distance_matrix: np.ndarray) -> List[int]:
        """Removes geographically clustered nodes using adaptive neighborhood"""
        seed = random.choice(city_tour)
        related = sorted([(n, distance_matrix[seed][n]) for n in city_tour if n != seed], key=lambda x: x[1])
        return [seed] + [n for n, _ in related[:num_removals-1]]

    return [random_removal, worst_removal, related_removal]

# Function: Insertion Operators
def insertion_operators():
    """
    Advanced insertion operators including regret-2 heuristic.
    Regret insertion considers future insertion costs to avoid myopic decisions.
    Based on the state-of-the-art insertion strategies from ALNS literature.
    """
    def cheapest_insertion(removed_nodes: List[int], city_tour: List[int], distance_matrix: np.ndarray) -> List[int]:
        """Greedily inserts nodes at position with minimal immediate cost"""
        for node in removed_nodes:
            best_cost = float('inf')
            best_pos = 0
            for i in range(len(city_tour)+1):
                prev = city_tour[i-1] if i > 0 else city_tour[-1]
                next_node = city_tour[i%len(city_tour)]
                cost = distance_matrix[prev][node] + distance_matrix[node][next_node] - distance_matrix[prev][next_node]
                if cost < best_cost:
                    best_cost, best_pos = cost, i
            city_tour.insert(best_pos, node)
        return city_tour

    def regret_insertion(removed_nodes: List[int], city_tour: List[int], distance_matrix: np.ndarray) -> List[int]:
        """Regret-2 heuristic considering opportunity cost of delayed insertion"""
        while removed_nodes:
            regrets = []
            for node in removed_nodes:
                costs = []
                for i in range(len(city_tour)+1):
                    prev = city_tour[i-1] if i > 0 else city_tour[-1]
                    next_node = city_tour[i%len(city_tour)]
                    costs.append(distance_matrix[prev][node] + distance_matrix[node][next_node] - distance_matrix[prev][next_node])
                sorted_costs = sorted(costs)
                regret = sorted_costs[1] - sorted_costs[0] if len(sorted_costs) > 1 else 0
                regrets.append((node, regret, sorted_costs[0]))
            node_to_insert = max(regrets, key=lambda x: x[1])[0]
            best_pos = costs.index(min(costs))
            city_tour.insert(best_pos, node_to_insert)
            removed_nodes.remove(node_to_insert)
        return city_tour

    return [cheapest_insertion, regret_insertion]

############################################################################

# Function: Adaptive Large Neighborhood Search
def adaptive_large_neighborhood_search(distance_matrix, iterations=100, removal_fraction=0.2, rho=0.1, time_limit=10, best=None, local_search=True, verbose=True):
    """
    Enhanced ALNS with multiple innovations:
    1. Dynamic response factor adapting operator weights using exponential smoothing
    2. Strategic oscillation through adaptive removal fraction
    3. Greedy randomized initial solution using nearest neighbor
    4. Segmented roulette-wheel selection with operator scoring
    Integrates state-of-the-art components from leading ALNS research.
    """
    import time
    start_time = time.time()
    # Generate initial solution using nearest neighbor heuristic
    def nearest_neighbor_init():
        start = random.randrange(len(distance_matrix))
        tour = [start]
        unvisited = set(range(len(distance_matrix))) - {start}
        while unvisited:
            current = tour[-1]
            next_node = min(unvisited, key=lambda x: distance_matrix[current][x])
            tour.append(next_node)
            unvisited.remove(next_node)
        return tour

    # ALNS Core Implementation
    removal_ops = removal_operators()
    insertion_ops = insertion_operators()
    weights_removal = [1.0]*len(removal_ops)
    weights_insertion = [1.0]*len(insertion_ops)
    scores_removal = [0.0]*len(removal_ops)
    scores_insertion = [0.0]*len(insertion_ops)
    usage_counts = [1e-6]*len(removal_ops)  # Avoid division by zero
    
    current_route = nearest_neighbor_init()
    best_route = current_route.copy()
    current_cost = distance_point(distance_matrix, current_route)
    best_cost = current_cost
    best.put(best_cost)

    invalid_solutions_count = 0
    repaired_solutions_count = 0
    iteration = 0
    while True:
        iteration += 1
        # Add progress monitoring
        if verbose and (iteration % 500 == 0 or iteration == iterations-1):
            print(f'Iteration {iteration+1}/{iterations}: Best Cost = {best_cost:.2f}')
        
        # Add timeout check
        if iteration > 100 and (best_cost == current_cost):
            if all(w < 0.01 for w in weights_removal):
                print('Early termination: no improvement detected')
                break
        # Adaptive parameter adjustment
        dynamic_removal = removal_fraction * (1 + 0.5*np.sin(iteration/(iterations/2*np.pi)))
        num_removals = max(2, int(len(current_route)*dynamic_removal))
        
        # Operator selection with epsilon-greedy exploration
        if random.random() < 0.1:  # 10% exploration chance
            removal_op = random.choice(removal_ops)
            insertion_op = random.choice(insertion_ops)
        else:
            # Numerically stable selection
            try:
                removal_op = random.choices(removal_ops, weights=weights_removal, k=1)[0]
                insertion_op = random.choices(insertion_ops, weights=weights_insertion, k=1)[0]
            except:
                weights_removal = [1.0]*len(removal_ops)
                weights_insertion = [1.0]*len(insertion_ops)
                removal_op = random.choice(removal_ops)
                insertion_op = random.choice(insertion_ops)
        
        # Destroy and repair
        removed_nodes = removal_op(current_route, num_removals, distance_matrix)
        new_route = insertion_op(removed_nodes, 
                               [n for n in current_route if n not in removed_nodes], 
                               distance_matrix)
        # Asegurar que la ruta termine en el nodo inicial
        if new_route[0] != new_route[-1]:
            new_route = new_route + [new_route[0]]
            
        # Validar la nueva solución antes de aceptarla
        is_valid, validation_message = validate_solution(new_route, distance_matrix)
        
        if not is_valid:
            invalid_solutions_count += 1
            new_route, repair_success = repair_solution(new_route, distance_matrix)
            
            if repair_success:
                repaired_solutions_count += 1
                is_valid = True
                if verbose:
                    print(f"Successfully repaired solution at iteration {iteration}")
            else:
                if verbose:
                    print(f"Could not repair solution at iteration {iteration}")
                continue
        new_cost = distance_point(distance_matrix, new_route)
        
        # Adaptive weight update with cooling schedule
        ri = removal_ops.index(removal_op)
        ii = insertion_ops.index(insertion_op)
        usage_counts[ri] += 1
        
        if new_cost < best_cost:
            reward = 1.5  # Highest reward for new best solution
            best_route = new_route.copy()
            best_cost = new_cost
            best.put(best_cost)
            time.sleep(0.1)
        elif new_cost < current_cost:
            reward = 1.2  # Reward for improvement
        else:
            reward = 0.8  # Penalty for worse solution
            
        scores_removal[ri] += reward
        scores_insertion[ii] += reward
        
        # Update weights every 10 iterations using moving average
        if iteration % 10 == 0:
            # Numerically stable softmax
            def stable_softmax(weights):
                max_w = np.max(weights)
                e = np.exp(weights - max_w)
                return e / e.sum()
            
            for i in range(len(removal_ops)):
                weights_removal[i] = 0.8*weights_removal[i] + 0.2*(scores_removal[i]/usage_counts[i])
            weights_removal = stable_softmax(weights_removal)
            
            for i in range(len(insertion_ops)):
                weights_insertion[i] = 0.8*weights_insertion[i] + 0.2*(scores_insertion[i]/usage_counts[i])
            weights_insertion = stable_softmax(weights_insertion)
        
        current_route = new_route
        current_cost = new_cost
        
        if verbose and iteration % 10 == 0:
            print(f'Iteration {iteration}: Best Cost = {best_cost:.2f}')
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break  

    if local_search:
        # Ensure proper tour format
        validation_route = best_route.copy()
        if isinstance(validation_route, int):  # Handle single-node edge case
            validation_route = [validation_route]
        
        # Remove duplicate end node if present
        if len(validation_route) > 1 and validation_route[-1] == validation_route[0]:
            validation_route = validation_route[:-1]
        
        pre_ls_distance = distance_point(distance_matrix, validation_route)
        
        # Pass as tuple with proper types
        optimized_route, ls_distance = local_search_2_opt(
            distance_matrix,
            (validation_route, pre_ls_distance),
            max_attempts=50,
            verbose=verbose
        )
        
        # Maintain best solution with type check
        if isinstance(optimized_route, list) and (ls_distance < best_cost):
            best_route = optimized_route
            best_cost = ls_distance
            best.put(best_cost)
            time.sleep(0.1)

    # Final return with validation
    final_route = best_route + [best_route[0]] if len(best_route) > 0 else []
    final_distance = distance_point(distance_matrix, final_route)
    best.put(final_distance)
    return final_route, final_distance



############################################################################

# Unmodified functions from original code:
# - euclidean_distance()
# - distance_calc()
# - distance_point()

def repair_solution(route: List[int], distance_matrix: np.ndarray) -> Tuple[List[int], bool]:
    """
    Intenta reparar una solución inválida del TSP.
    
    Args:
        route: Lista de nodos que representa la ruta
        distance_matrix: Matriz de distancias
        
    Returns:
        Tuple[List[int], bool]: (ruta_reparada, éxito_reparación)
    """
    n = len(distance_matrix)
    try:
        # 1. Eliminar duplicados manteniendo el orden
        seen = set()
        repaired_route = []
        for node in route:
            if node not in seen:
                seen.add(node)
                repaired_route.append(node)
                
        # 2. Añadir nodos faltantes
        missing_nodes = set(range(n)) - set(repaired_route)
        if missing_nodes:
            # Insertar nodos faltantes en las mejores posiciones
            for node in missing_nodes:
                best_position = 0
                best_cost = float('inf')
                
                # Probar cada posición posible
                for i in range(len(repaired_route)):
                    # Insertar temporalmente y calcular costo
                    temp_route = repaired_route[:i] + [node] + repaired_route[i:]
                    cost = sum(distance_matrix[temp_route[j]][temp_route[j+1]] 
                             for j in range(len(temp_route)-1))
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_position = i
                
                # Insertar en la mejor posición encontrada
                repaired_route.insert(best_position, node)
        
        # 3. Asegurar que la ruta termine en el nodo inicial
        if repaired_route[0] != repaired_route[-1]:
            repaired_route.append(repaired_route[0])
            
        # 4. Validar la solución reparada
        is_valid, _ = validate_solution(repaired_route, distance_matrix)
        
        return repaired_route, is_valid
        
    except Exception as e:
        print(f"Error during repair: {str(e)}")
        return route, False
############################################################################
def validate_solution(route: List[int], distance_matrix: np.ndarray, original_distance: float = None) -> Tuple[bool, str]:
    """
    Validates a TSP solution checking multiple criteria.
    
    Args:
        route: List of nodes representing the tour
        distance_matrix: Distance matrix of the problem
        original_distance: Original tour distance for comparison (optional)
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        # 1. Check if route is empty or None
        if not route or route is None:
            return False, "Route is empty or None"
            
        # 2. Check if route contains integers
        if not all(isinstance(x, (int, np.integer)) for x in route):
            return False, "Route contains non-integer values"
            
        # 3. Check if route starts and ends with the same node
        if route[0] != route[-1]:
            return False, "Route does not start and end at the same node"
            
        # 4. Check if all nodes are within valid range
        n = len(distance_matrix)
        if not all(0 <= node < n for node in route):
            return False, f"Route contains invalid node indices. Valid range: 0 to {n-1}"
            
        # 5. Check for duplicates (excluding first/last node)
        route_without_last = route[:-1]
        if len(set(route_without_last)) != len(route_without_last):
            return False, "Route contains duplicate nodes"
            
        # 6. Check if all nodes are visited
        if len(route_without_last) != n:
            return False, f"Route does not visit all nodes. Expected {n} nodes, got {len(route_without_last)}"
            
        # 7. Verify distance calculation
        calculated_distance = distance_point(distance_matrix, route)
        if original_distance is not None:
            if abs(calculated_distance - original_distance) > 1e-6:
                return False, f"Distance mismatch. Calculated: {calculated_distance}, Reported: {original_distance}"
            
        # 8. Check for connectivity
        for i in range(len(route)-1):
            if distance_matrix[route[i]][route[i+1]] == float('inf'):
                return False, f"Invalid connection between nodes {route[i]} and {route[i+1]}"
                
        return True, "Solution is valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"
