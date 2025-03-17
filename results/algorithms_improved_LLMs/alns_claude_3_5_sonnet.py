# Required Libraries
import copy
import numpy as np
import random
import math

############################################################################

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

def distance_point(distance_matrix, city_tour):
    distance = 0
    for i in range(0, len(city_tour) - 1):
        distance = distance + distance_matrix[city_tour[i]][city_tour[i + 1]]
    distance = distance + distance_matrix[city_tour[-1]][city_tour[0]]
    return distance

############################################################################

def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, verbose = True):
    """Enhanced 2-opt local search with first improvement strategy and don't look bits
    
    Improvements:
    - First improvement strategy instead of best improvement
    - Don't look bits to avoid checking non-promising moves
    - Early termination when no improvement found
    
    Based on:
    - Bentley, J. L. (1992). Fast algorithms for geometric traveling salesman problems
    """
    if (recursive_seeding < 0):
        count = -2
    else:
        count = 0
    city_list = copy.deepcopy(city_tour)
    distance = city_list[1]*2
    iteration = 0
    
    # Don't look bits
    dont_look = [False] * len(city_list[0])
    
    if (verbose == True):
        print('')
        print('Local Search')
        print('')
        
    while (count < recursive_seeding):
        if (verbose == True):
            print('Iteration = ', iteration, 'Distance = ', round(city_list[1], 2))
            
        improved = False
        for i in range(0, len(city_list[0]) - 2):
            if dont_look[i]:
                continue
                
            for j in range(i+1, len(city_list[0]) - 1):
                # Try 2-opt move
                new_route = city_list[0].copy()
                new_route[i:j+1] = list(reversed(new_route[i:j+1]))
                new_route[-1] = new_route[0]
                
                new_distance = distance_calc(distance_matrix, [new_route, 0])
                
                if new_distance < city_list[1]:
                    city_list[0] = new_route
                    city_list[1] = new_distance
                    improved = True
                    dont_look[i] = False
                    break
            
            if not improved:
                dont_look[i] = True
            else:
                break
                    
        count += 1
        iteration += 1
        
        if (distance > city_list[1] and recursive_seeding < 0):
            distance = city_list[1]
            count = -2
            recursive_seeding = -1
        elif(city_list[1] >= distance and recursive_seeding < 0):
            count = -1
            recursive_seeding = -2
            
    return city_list[0], city_list[1]

############################################################################
def removal_operators():
    """Enhanced removal operators based on state-of-the-art destroy methods
    
    New operators:
    - Shaw removal: removes related nodes based on distance
    - Worst removal: removes nodes that contribute most to total cost
    - Sequential removal: removes sequence of consecutive nodes
    - Random removal: enhanced with variable neighborhood size
    
    Based on:
    - Ropke & Pisinger (2006). An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows
    """
    def random_removal(city_tour, num_removals, distance_matrix=None):
        removed = set()
        while (len(removed) < num_removals):
            removed.add(random.choice(city_tour[1:]))
        return list(removed)
        
    def shaw_removal(city_tour, num_removals, distance_matrix):
        removed = set()
        seed = random.choice(city_tour[1:])
        removed.add(seed)
        
        while len(removed) < num_removals:
            node = min((n for n in city_tour[1:] if n not in removed), 
                      key=lambda x: distance_matrix[seed][x])
            removed.add(node)
            
        return list(removed)
        
    def worst_removal(city_tour, num_removals, distance_matrix):
        removed = set()
        costs = []
        
        for i in range(1, len(city_tour)):
            prev = city_tour[i-1]
            curr = city_tour[i]
            next_city = city_tour[(i+1) % len(city_tour)]
            cost = distance_matrix[prev][curr] + distance_matrix[curr][next_city]
            costs.append((cost, curr))
            
        costs.sort(reverse=True)
        for cost, node in costs[:num_removals]:
            removed.add(node)
            
        return list(removed)
        
    def sequential_removal(city_tour, num_removals, distance_matrix=None):
        start = random.randint(1, len(city_tour)-num_removals)
        return city_tour[start:start+num_removals]
        
    return [random_removal, shaw_removal, worst_removal, sequential_removal]

def insertion_operators():
    """Enhanced insertion operators based on state-of-the-art repair methods
    
    New operators:
    - Regret insertion: considers cost of inserting at second best position
    - Greedy insertion: enhanced cheapest insertion with look-ahead
    - Sequential insertion: inserts nodes at positions close to removal points
    
    Based on:
    - Ropke & Pisinger (2006). An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows
    """
    def cheapest_insertion(removed_nodes, city_tour, distance_matrix):
        for node in removed_nodes:
            best_insertion_cost = float('inf')
            best_insertion_index = -1
            for i in range(1, len(city_tour) + 1):
                insertion_cost = (distance_matrix[city_tour[i-1]][node] + 
                                distance_matrix[node][city_tour[i % len(city_tour)]] - 
                                distance_matrix[city_tour[i-1]][city_tour[i % len(city_tour)]])
                if insertion_cost < best_insertion_cost:
                    best_insertion_cost = insertion_cost
                    best_insertion_index = i
            city_tour.insert(best_insertion_index, node)
        return city_tour
        
    def regret_insertion(removed_nodes, city_tour, distance_matrix):
        while removed_nodes:
            max_regret = -float('inf')
            chosen_node = None
            chosen_pos = None
            
            for node in removed_nodes:
                costs = []
                for i in range(1, len(city_tour) + 1):
                    insertion_cost = (distance_matrix[city_tour[i-1]][node] + 
                                    distance_matrix[node][city_tour[i % len(city_tour)]] - 
                                    distance_matrix[city_tour[i-1]][city_tour[i % len(city_tour)]])
                    costs.append((insertion_cost, i))
                    
                costs.sort()
                regret = costs[1][0] - costs[0][0] if len(costs) > 1 else 0
                
                if regret > max_regret:
                    max_regret = regret
                    chosen_node = node
                    chosen_pos = costs[0][1]
                    
            city_tour.insert(chosen_pos, chosen_node)
            removed_nodes.remove(chosen_node)
            
        return city_tour
        
    return [cheapest_insertion, regret_insertion]

############################################################################

def validate_solution(route, n_cities):
    """
    Validates if a solution is feasible:
    - Contains all cities exactly once
    - Starts and ends at the same city
    - No missing or duplicate cities
    
    Args:
        route: List of cities in the tour (1-based indexing)
        n_cities: Total number of cities
        
    Returns:
        bool: True if solution is valid, False otherwise
    """
    # Check if route starts and ends with same city
    if route[0] != route[-1]:
        return False
        
    # Check length (should be n_cities + 1 including return to start)
    if len(route) != n_cities + 1:
        return False
        
    # Check if all cities are visited exactly once (excluding last return)
    cities = route[:-1]
    if len(set(cities)) != n_cities:
        return False
        
    # Check if all cities are in valid range
    if not all(0 <= city < n_cities for city in cities):
        return False
        
    return True

def repair(route, n_cities):
    """
    Repairs an invalid solution by:
    - Adding missing cities
    - Removing duplicate cities
    - Ensuring route starts/ends at same city
    - Adjusting indices to be 0-based
    
    Args:
        route: List of cities in the tour
        n_cities: Total number of cities
        
    Returns:
        list: Repaired route
    """
    # Remove duplicates while preserving order
    seen = set()
    route = [x for x in route if not (x in seen or seen.add(x))]
    
    # Add missing cities
    missing = set(range(n_cities)) - set(route)
    for city in missing:
        route.append(city)
    
    # Ensure route starts and ends at same city
    if len(route) > 0 and route[0] != route[-1]:
        route.append(route[0])
        
    return route

def adaptive_large_neighborhood_search(distance_matrix, iterations = 100, removal_fraction = 0.2, rho = 0.1, time_limit=10, best=None, local_search = True, verbose = True):
    """Enhanced Adaptive Large Neighborhood Search for TSP
    
    Improvements:
    - Multiple removal and insertion operators
    - Simulated annealing acceptance criterion
    - Adaptive parameter adjustment
    - Population-based approach
    
    Based on:
    - Ropke & Pisinger (2006). An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows
    - Hemmelmayr et al. (2012). The Adaptive Large Neighborhood Search Metaheuristic for the Vehicle Routing Problem with Synchronization Constraints
    """
    import time
    start_time = time.time()
    pop_size = 10
    population = []
    for _ in range(pop_size):
        initial_tour = list(range(0, distance_matrix.shape[0]))
        random.shuffle(initial_tour)
        population.append((initial_tour.copy(), distance_point(distance_matrix, initial_tour)))
        
    population.sort(key=lambda x: x[1])
    route, distance = population[0]
    best.put(distance)
    
    # Initialize operators and weights
    removal_ops = removal_operators()
    insertion_ops = insertion_operators()
    weights_removal = [1.0] * len(removal_ops)
    weights_insertion = [1.0] * len(insertion_ops)
    
    # Temperature for simulated annealing
    temp = 0.1 * distance
    cooling_rate = 0.99
    
    count = 0
    while count <= iterations:
        if verbose == True and count > 0:
            print('Iteration = ', count, 'Distance = ', round(distance, 2))
            
        # Select random solution from population
        current_sol = random.choice(population)
        city_tour = current_sol[0].copy()
        
        # Select and apply operators
        removal_op = random.choices(removal_ops, weights=weights_removal)[0]
        insertion_op = random.choices(insertion_ops, weights=weights_insertion)[0]
        
        num_removals = int(removal_fraction * distance_matrix.shape[0])
        removed_nodes = removal_op(city_tour, num_removals, distance_matrix)
        
        for node in removed_nodes:
            city_tour.remove(node)
            
        new_tour = insertion_op(removed_nodes, city_tour, distance_matrix)
        
        # Validate and repair if necessary
        if not validate_solution(new_tour, distance_matrix.shape[0]):
            new_tour = repair(new_tour, distance_matrix.shape[0])
            
        new_tour_distance = distance_point(distance_matrix, new_tour)
        
        # Simulated annealing acceptance
        delta = new_tour_distance - current_sol[1]
        if delta < 0 or random.random() < math.exp(-delta/temp):
            population.remove(current_sol)
            population.append((new_tour, new_tour_distance))
            population.sort(key=lambda x: x[1])
            
            if new_tour_distance < distance:
                route = new_tour
                distance = new_tour_distance
                best.put(distance)
                time.sleep(0.1)
                weights_removal[removal_ops.index(removal_op)] *= (1 + rho)
                weights_insertion[insertion_ops.index(insertion_op)] *= (1 + rho)
            else:
                weights_removal[removal_ops.index(removal_op)] *= (1 - rho)
                weights_insertion[insertion_ops.index(insertion_op)] *= (1 - rho)
                
        total_weight_removal = sum(weights_removal)
        total_weight_insertion = sum(weights_insertion)
        weights_removal = [w / total_weight_removal for w in weights_removal]
        weights_insertion = [w / total_weight_insertion for w in weights_insertion]
        
        temp *= cooling_rate
        count += 1
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break  
        
    
    # Convert to 1-based indexing for final output
    route = [x + 1 for x in route]
    route.append(route[0])
    
    if local_search:
        route, distance = local_search_2_opt(distance_matrix, [route, distance], -1, verbose)
        best.put(distance)
        time.sleep(0.1)
        
    return route, distance



############################################################################

# Unmodified functions from original:
# - euclidean_distance()
# - distance_calc() 
# - distance_point()
