
# Required Libraries
import copy
import numpy as np
import random
import math

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

# Function: Removal Operators
def removal_operators():
    """
    The removal operators have been enhanced by adding Shaw Removal and Worst Removal, based on:
    - Shaw Removal: Removes related nodes to destroy specific parts of the solution, as per Shaw (1997).
    - Worst Removal: Removes nodes contributing most to the cost to intensify search, as per Ropke and Pisinger (2006).
    These improvements help in exploring the solution space more effectively, leading to better solutions and faster convergence.
    """

    # Random Removal Operator (Unmodified from the original code)
    def random_removal(city_tour, num_removals):
        removed = set()
        while (len(removed) < num_removals):
            removed.add(random.choice(city_tour[1:]))
        return list(removed)

    # Shaw Removal Operator
    def shaw_removal(city_tour, num_removals, distance_matrix):
        removed = set()
        seed_node = random.choice(city_tour[1:])
        removed.add(seed_node)
        relatedness = {}
        for node in city_tour[1:]:
            if node != seed_node:
                relatedness[node] = distance_matrix[seed_node][node]
        while len(removed) < num_removals:
            if len(relatedness) == 0:
                break
            sorted_nodes = sorted(relatedness.items(), key=lambda x: x[1])
            next_node = sorted_nodes[0][0]
            removed.add(next_node)
            del relatedness[next_node]
        return list(removed)

    # Worst Removal Operator
    def worst_removal(city_tour, num_removals, distance_matrix):
        contributions = {}
        for i in range(len(city_tour)):
            prev_node = city_tour[i - 1]
            node = city_tour[i]
            next_node = city_tour[(i + 1) % len(city_tour)]
            removal_cost = distance_matrix[prev_node][node] + distance_matrix[node][next_node] - distance_matrix[prev_node][next_node]
            contributions[node] = removal_cost
        sorted_nodes = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        removed_nodes = [node for node, cost in sorted_nodes[:num_removals]]
        return removed_nodes

    return [random_removal, shaw_removal, worst_removal]

# Function: Insertion Operators
def insertion_operators():
    """
    The insertion operators have been enhanced by adding Regret-2 Insertion, based on:
    - Regret-2 Insertion: Considers the opportunity cost of not selecting a node now, as per Potvin and Rousseau (1993).
    This allows for better decisions during reconstruction, leading to improved solution quality and convergence speed.
    """

    # Cheapest Insertion Operator (Modified to accept only necessary parameters)
    def cheapest_insertion(removed_nodes, city_tour, distance_matrix):
        for node in removed_nodes:
            best_insertion_cost  = float('inf')
            best_insertion_index = -1
            for i in range(len(city_tour)):
                prev_node = city_tour[i - 1]
                next_node = city_tour[i]
                insertion_cost = (distance_matrix[prev_node][node] + distance_matrix[node][next_node] - distance_matrix[prev_node][next_node])
                if (insertion_cost < best_insertion_cost):
                    best_insertion_cost  = insertion_cost
                    best_insertion_index = i
            city_tour.insert(best_insertion_index, node)
        return city_tour

    # Regret-2 Insertion Operator
    def regret2_insertion(removed_nodes, city_tour, distance_matrix):
        while removed_nodes:
            regret_values = []
            for node in removed_nodes:
                insertion_costs = []
                for i in range(len(city_tour)):
                    prev_node = city_tour[i - 1]
                    next_node = city_tour[i]
                    cost = (distance_matrix[prev_node][node] + distance_matrix[node][next_node] - distance_matrix[prev_node][next_node])
                    insertion_costs.append(cost)
                insertion_costs.sort()
                if len(insertion_costs) >= 2:
                    regret = insertion_costs[1] - insertion_costs[0]
                else:
                    regret = 0
                regret_values.append((regret, node, insertion_costs))
            regret_values.sort(reverse=True)
            _, node_to_insert, costs = regret_values[0]
            best_cost = min(costs)
            best_positions = [i for i, cost in enumerate(costs) if cost == best_cost]
            best_position = best_positions[0]
            city_tour.insert(best_position, node_to_insert)
            removed_nodes.remove(node_to_insert)
        return city_tour

    return [cheapest_insertion, regret2_insertion]

############################################################################

# Function: Adaptive Large Neighborhood Search
def adaptive_large_neighborhood_search(distance_matrix, iterations=100, removal_fraction=0.2, rho=0.1, time_limit=10, best=None,local_search=True, verbose=True):
    """
    This function has been improved by:
    1. Adding new removal operators (Shaw Removal, Worst Removal) and insertion operators (Regret-2 Insertion) to enhance diversification and intensification, based on Ropke and Pisinger (2006).
    2. Implementing an adaptive weight adjustment scheme with scores and weights updated according to operator performance, as per Ropke and Pisinger (2006).
    3. Including a simulated annealing acceptance criterion to accept worse solutions with a certain probability to escape local optima, based on Kirkpatrick et al. (1983).
    These improvements collectively enhance the search capabilities, leading to better quality solutions and faster convergence.
    """
    import time
    start_time = time.time()
    initial_tour = list(range(0, distance_matrix.shape[0]))
    random.shuffle(initial_tour)
    route = initial_tour.copy()
    best_route = route.copy()
    distance = distance_point(distance_matrix, route)
    best_distance = distance
    best.put(best_distance)
            
    removal_ops = removal_operators()
    insertion_ops = insertion_operators()
    # Initial weights and scores
    weights_removal = [1.0] * len(removal_ops)
    weights_insertion = [1.0] * len(insertion_ops)
    scores_removal = [0.0] * len(removal_ops)
    scores_insertion = [0.0] * len(insertion_ops)
    # Parameters for adaptive weight adjustment
    sigma1 = 33  # reward for best solution
    sigma2 = 9   # reward for better solution
    sigma3 = 3   # reward for accepted solution
    decay = 0.9  # decay factor for scores
    count = 0
    T_initial = distance * 0.01  # Initial temperature for simulated annealing
    T = T_initial
    alpha = 0.995  # Cooling rate
    while count <= iterations:
        if verbose and count > 0:
            print('Iteration = ', count, 'Distance = ', round(best_distance, 2))
        city_tour = route.copy()
        # Select removal and insertion operators based on weights
        removal_op = random.choices(removal_ops, weights=weights_removal)[0]
        insertion_op = random.choices(insertion_ops, weights=weights_insertion)[0]
        num_removals = max(1, int(removal_fraction * distance_matrix.shape[0]))
        # Remove nodes
        if removal_op.__name__ == 'shaw_removal' or removal_op.__name__ == 'worst_removal':
            removed_nodes = removal_op(city_tour, num_removals, distance_matrix)
        else:
            removed_nodes = removal_op(city_tour, num_removals)
        for node in removed_nodes:
            city_tour.remove(node)
        # Reinsert nodes
        new_tour = insertion_op(removed_nodes, city_tour, distance_matrix)
        new_tour_distance = distance_point(distance_matrix, new_tour)
        # Acceptance criterion (Simulated Annealing)
        delta = new_tour_distance - distance
        acceptance_probability = math.exp(-delta / T) if delta > 0 else 1
        if random.uniform(0, 1) < acceptance_probability:
            route = new_tour
            distance = new_tour_distance
            # Update scores
            idx_removal = removal_ops.index(removal_op)
            idx_insertion = insertion_ops.index(insertion_op)
            if new_tour_distance < best_distance:
                best_route = new_tour
                best_distance = new_tour_distance
                best.put(best_distance)
                time.sleep(0.1)
                scores_removal[idx_removal] += sigma1
                scores_insertion[idx_insertion] += sigma1
            else:
                scores_removal[idx_removal] += sigma2
                scores_insertion[idx_insertion] += sigma2
        else:
            # No improvement
            idx_removal = removal_ops.index(removal_op)
            idx_insertion = insertion_ops.index(insertion_op)
            scores_removal[idx_removal] += sigma3
            scores_insertion[idx_insertion] += sigma3
        # Update weights using adaptive weight adjustment
        for i in range(len(weights_removal)):
            weights_removal[i] = weights_removal[i] * decay + (1 - decay) * scores_removal[i]
        for i in range(len(weights_insertion)):
            weights_insertion[i] = weights_insertion[i] * decay + (1 - decay) * scores_insertion[i]
        # Normalize weights
        total_weight_removal = sum(weights_removal)
        total_weight_insertion = sum(weights_insertion)
        weights_removal = [w / total_weight_removal for w in weights_removal]
        weights_insertion = [w / total_weight_insertion for w in weights_insertion]
        # Decay scores
        scores_removal = [s * decay for s in scores_removal]
        scores_insertion = [s * decay for s in scores_insertion]
        # Update temperature
        T = T * alpha
        count += 1
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break 
    route = best_route + [best_route[0]]
    route = [item + 1 for item in route]
    if local_search:
        route, best_distance = local_search_2_opt(distance_matrix, [route, best_distance], -1, verbose)
        best.put(best_distance)
        time.sleep(0.1)
    return route, best_distance

############################################################################

# Unmodified functions from the original code:
# - euclidean_distance
# - distance_calc
# - distance_point
# - local_search_2_opt
