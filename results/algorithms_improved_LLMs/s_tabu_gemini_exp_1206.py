import random
import numpy as np
import copy

############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]
    return distance

############################################################################

# Function: Swap
def local_search_2_swap(distance_matrix, city_tour):
    best_route = copy.deepcopy(city_tour)
    i, j = random.sample(range(0, len(city_tour[0])-1), 2)
    if (i > j):
        i, j = j, i
    best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))
    best_route[0][-1] = best_route[0][0]
    best_route[1] = distance_calc(distance_matrix, best_route)
    return best_route

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour):
    city_list = copy.deepcopy(city_tour)
    best_route = copy.deepcopy(city_list)
    seed = copy.deepcopy(city_list)
    for i in range(0, len(city_list[0]) - 2):
        for j in range(i+1, len(city_list[0]) - 1):
            best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))
            best_route[0][-1] = best_route[0][0]
            best_route[1] = distance_calc(distance_matrix, best_route)
            if (best_route[1] < city_list[1]):
                city_list[1] = copy.deepcopy(best_route[1])
                for n in range(0, len(city_list[0])):
                    city_list[0][n] = best_route[0][n]
            best_route = copy.deepcopy(seed)
    return city_list

# Function: 4_opt
def local_search_4_opt_stochastic(distance_matrix, city_tour):
    count = 0
    city_list = [city_tour[0][:-1], city_tour[1]]
    best_route = copy.deepcopy(city_list)
    best_route_1 = [[], 1]
    seed = copy.deepcopy(city_list)
    i, j, k, L = random.sample(list(range(0, len(city_list[0]))), 4)
    idx = [i, j, k, L]
    idx.sort()
    i, j, k, L = idx
    A = best_route[0][:i+1] + best_route[0][i+1:j+1]
    B = best_route[0][j+1:k+1]
    b = list(reversed(B))
    C = best_route[0][k+1:L+1]
    c = list(reversed(C))
    D = best_route[0][L+1:]
    d = list(reversed(D))
    trial = [
        # 4-opt: Sequential
        [A + b + c + d], [A + C + B + d], [A + C + b + d], [A + c + B + d], [A + D + B + c],
        [A + D + b + C], [A + d + B + c], [A + d + b + C], [A + d + b + c], [A + b + D + C],
        [A + b + D + c], [A + b + d + C], [A + C + d + B], [A + C + d + b], [A + c + D + B],
        [A + c + D + b], [A + c + d + b], [A + D + C + b], [A + D + c + B], [A + d + C + B],

        # 4-opt: Non-Sequential
        [A + b + C + d], [A + D + b + c], [A + c + d + B], [A + D + C + B], [A + d + C + b]
    ]
    for item in trial:
        best_route_1[0] = item[0]
        best_route_1[1] = distance_calc(distance_matrix, [best_route_1[0] + [best_route_1[0][0]], 1])
        if (best_route_1[1] < best_route[1]):
            best_route = [best_route_1[0], best_route_1[1]]
        if (best_route[1] < city_list[1]):
            city_list = [best_route[0], best_route[1]]
        best_route = copy.deepcopy(seed)
    count = count + 1
    city_list = [city_list[0] + [city_list[0][0]], city_list[1]]
    return city_list

############################################################################

# Function: Build Recency Based Memory and Frequency Based Memory (STM and LTM)
def build_stm_and_ltm(distance_matrix):
    n = int((distance_matrix.shape[0]**2 - distance_matrix.shape[0])/2)
    stm_and_ltm = np.zeros((n, 5)) # ['City 1','City 2','Recency', 'Frequency', 'Distance']
    count = 0
    for i in range (0, int((distance_matrix.shape[0]**2))):
        city_1 = i // (distance_matrix.shape[1])
        city_2 = i % (distance_matrix.shape[1])
        if (city_1 < city_2):
            stm_and_ltm[count, 0] = city_1 + 1
            stm_and_ltm[count, 1] = city_2 + 1
            count = count + 1
    return stm_and_ltm

# Function: Diversification
def ltm_diversification (distance_matrix, stm_and_ltm, city_list):
    """
    Implements a diversification strategy based on Long-Term Memory (LTM).

    This function diversifies the search by performing 2-swaps on edges 
    that have been least frequently visited, as indicated by the LTM.
    The number of swaps is randomly chosen between 1 and n/3, where n is the number of cities.

    Enhances performance by:
    - Preventing the search from getting stuck in local optima.
    - Exploring new regions of the search space.

    Based on:
    - The concept of Long-Term Memory in Tabu Search, which tracks the frequency of visited edges.
    """
    stm_and_ltm = stm_and_ltm[stm_and_ltm[:,3].argsort()]
    stm_and_ltm = stm_and_ltm[stm_and_ltm[:,4].argsort()]
    lenght = random.sample((range(1, int(distance_matrix.shape[0]/3))), 1)[0]
    for i in range(0, lenght):
        city_list = local_search_2_swap(distance_matrix, city_list)
        stm_and_ltm[i, 3] = stm_and_ltm[i, 3] + 1
        stm_and_ltm[i, 2] = 1
    return stm_and_ltm, city_list

# Function: Tabu Update
def tabu_update(distance_matrix, stm_and_ltm, city_list, best_distance, tabu_list, tabu_tenure = 20, diversify = False):
    """
    Updates the Tabu list, Short-Term Memory (STM), and Long-Term Memory (LTM).

    This function performs the following steps:
    1. Intensification: Applies 2-opt to the current solution to potentially improve it.
    2. Evaluates candidate moves: Calculates the distance of potential 2-swaps.
    3. Aspiration criterion: If a tabu move results in a better solution than the current best, it's accepted.
    4. Tabu update: Updates the recency and frequency of the chosen move in STM and LTM.
    5. Tabu list management: Adds the chosen move to the Tabu list and removes old entries based on tenure.
    6. Diversification: If triggered, applies LTM-based diversification and stochastic 4-opt.

    Enhances performance by:
    - Intensification: Exploiting good solutions to find even better ones.
    - Aspiration criterion: Allowing good tabu moves to be accepted.
    - Tabu list: Preventing cycling and encouraging exploration.
    - Diversification: Escaping local optima and exploring new regions of the search space.

    Based on:
    - Core principles of Tabu Search, including intensification, diversification, aspiration criteria, and tabu list management.
    - Stochastic 4-opt as a diversification strategy.
    """
    m_list = []
    n_list = []
    city_list = local_search_2_opt(distance_matrix, city_list) # itensification
    for i in range(0, stm_and_ltm.shape[0]):
        stm_and_ltm[i, -1] = local_search_2_swap(distance_matrix, city_list)[1]
    stm_and_ltm = stm_and_ltm[stm_and_ltm[:,4].argsort()] # Distance
    m = int(stm_and_ltm[0,0]-1)
    n = int(stm_and_ltm[0,1]-1)
    recency = int(stm_and_ltm[0,2])
    distance = stm_and_ltm[0,-1]
    if (distance < best_distance): # Aspiration Criterion -> by Objective
        city_list = local_search_2_swap(distance_matrix, city_list)
        i = 0
        while (i < stm_and_ltm.shape[0]):
            if (stm_and_ltm[i, 0] == m + 1 and stm_and_ltm[i, 1] == n + 1):
                stm_and_ltm[i, 2] = 1
                stm_and_ltm[i, 3] = stm_and_ltm[i, 3] + 1
                stm_and_ltm[i, -1] = distance
                if (stm_and_ltm[i, 3] == 1):
                    m_list.append(m + 1)
                    n_list.append(n + 1)
                i = stm_and_ltm.shape[0]
            i = i + 1
    else:
        i = 0
        while (i < stm_and_ltm.shape[0]):
            recency = int(stm_and_ltm[i,2])
            distance = local_search_2_swap(distance_matrix, city_list)[1]
            if (distance < best_distance):
                city_list = local_search_2_swap(distance_matrix, city_list)
            if (recency == 0):
                city_list = local_search_2_swap(distance_matrix, city_list)
                stm_and_ltm[i, 2] = 1
                stm_and_ltm[i, 3] = stm_and_ltm[i, 3] + 1
                stm_and_ltm[i, -1] = distance
                if (stm_and_ltm[i, 3] == 1):
                    m_list.append(m + 1)
                    n_list.append(n + 1)
                i = stm_and_ltm.shape[0]
            i = i + 1
    if (len(m_list) > 0):
        tabu_list[0].append(m_list[0])
        tabu_list[1].append(n_list[0])
    if (len(tabu_list[0]) > tabu_tenure):
        i = 0
        while (i < stm_and_ltm.shape[0]):
            if (stm_and_ltm[i, 0] == tabu_list[0][0] and stm_and_ltm[i, 1] == tabu_list[1][0]):
                del tabu_list[0][0]
                del tabu_list[1][0]
                stm_and_ltm[i, 2] = 0
                i = stm_and_ltm.shape[0]
            i = i + 1
    if (diversify == True):
        stm_and_ltm, city_list = ltm_diversification(distance_matrix, stm_and_ltm, city_list) # diversification
        if (distance_matrix.shape[0] > 4):
            city_list = local_search_4_opt_stochastic(distance_matrix, city_list) # diversification
    return stm_and_ltm, city_list, tabu_list

############################################################################

# Function: Tabu Search
def tabu_search(distance_matrix, city_tour, iterations = 150, tabu_tenure = 20, time_limit=10, best=None, verbose = True):
    """
    Implements an enhanced Tabu Search algorithm for the Traveling Salesperson Problem (TSP).

    Improvements:
    1. Adaptive Tabu Tenure: The tabu tenure is dynamically adjusted based on the number of cities.
    2. Enhanced Diversification Trigger: Diversification is triggered based on a combination of lack of improvement and iteration count.
    3. Path Relinking (Elite Solutions): Periodically, path relinking is performed between the current best solution and a set of elite solutions.
        This helps to explore promising regions of the search space that lie between known good solutions.

    Enhances performance by:
    - Adaptive Tabu Tenure: Allows for more flexibility in exploring the search space, potentially leading to faster convergence.
    - Enhanced Diversification Trigger: More effectively balances exploration and exploitation, preventing premature convergence.
    - Path Relinking: Introduces a more systematic way of exploring the search space, potentially leading to higher quality solutions.

    Based on:
    - State-of-the-art Tabu Search techniques, including adaptive tabu tenure and path relinking.
    """
    import time
    start_time = time.time()
    count = 0
    best_solution = copy.deepcopy(city_tour)
    stm_and_ltm = build_stm_and_ltm(distance_matrix)
    tabu_list = [[],[]]
    diversify = False
    no_improvement = 0

    # Adaptive Tabu Tenure
    adaptive_tabu_tenure = max(tabu_tenure, int(distance_matrix.shape[0] / 4))

    # Elite Solutions for Path Relinking
    elite_solutions = []
    max_elite_size = 5

    best.put(best_solution[1])
    while (count < iterations):
        if (verbose == True):
            print('Iteration = ', count, 'Distance = ', round(best_solution[1], 2))
        stm_and_ltm, city_tour, tabu_list = tabu_update(distance_matrix, stm_and_ltm, city_tour, best_solution[1], tabu_list, adaptive_tabu_tenure, diversify)

        # Path Relinking
        if count % (iterations // 5) == 0 and len(elite_solutions) > 1:
            target_solution = random.choice(elite_solutions)
            city_tour = path_relinking(distance_matrix, city_tour, target_solution)

        if (city_tour[1] < best_solution[1]):
            best_solution = copy.deepcopy(city_tour)
            
            best.put(best_solution[1])
            time.sleep(0.1)
            
            no_improvement = 0
            diversify = False

            # Update Elite Solutions
            if len(elite_solutions) < max_elite_size:
                elite_solutions.append(copy.deepcopy(best_solution))
            else:
                elite_solutions.sort(key=lambda x: x[1])
                if best_solution[1] < elite_solutions[-1][1]:
                    elite_solutions[-1] = copy.deepcopy(best_solution)

        else:
            # Enhanced Diversification Trigger
            if (no_improvement > 0 and no_improvement % max(5, int(iterations/10)) == 0):
                diversify = True
                no_improvement = 0
            else:
                diversify = False
            no_improvement = no_improvement + 1
        count = count + 1
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break  
    route, distance = best_solution
    return route, distance

def path_relinking(distance_matrix, initial_solution, target_solution):
    """
    Implements path relinking between two solutions for the TSP.

    Path relinking is a strategy that explores trajectories connecting high-quality solutions.
    It generates new solutions by introducing attributes of the guiding solution (target) into the initiating solution (initial).

    Enhances performance by:
    - Exploring intermediate solutions between known good solutions.
    - Potentially finding better solutions in unexplored regions of the search space.

    Based on:
    - The concept of path relinking as an intensification and diversification strategy in metaheuristics.
    """
    initial_tour = initial_solution[0][:-1]
    target_tour = target_solution[0][:-1]
    best_tour = initial_tour[:]
    best_distance = initial_solution[1]

    common_edges = set()
    for i in range(len(initial_tour)):
        for j in range(i + 1, len(initial_tour)):
            if (initial_tour[i], initial_tour[j]) in [(target_tour[k], target_tour[l]) for k in range(len(target_tour)) for l in range(k + 1, len(target_tour))] or \
               (initial_tour[j], initial_tour[i]) in [(target_tour[k], target_tour[l]) for k in range(len(target_tour)) for l in range(k + 1, len(target_tour))]:
                common_edges.add((min(initial_tour[i], initial_tour[j]), max(initial_tour[i], initial_tour[j])))

    current_tour = initial_tour[:]
    while True:
        best_move = None
        best_move_distance = float('inf')

        for i in range(len(current_tour)):
            for j in range(i + 1, len(current_tour)):
                if (min(current_tour[i], current_tour[j]), max(current_tour[i], current_tour[j])) not in common_edges:
                    new_tour = current_tour[:]
                    new_tour[i:j+1] = reversed(new_tour[i:j+1])
                    new_distance = distance_calc(distance_matrix, [new_tour + [new_tour[0]], 1])

                    if new_distance < best_move_distance:
                        best_move_distance = new_distance
                        best_move = (i, j)

        if best_move:
            current_tour[best_move[0]:best_move[1]+1] = reversed(current_tour[best_move[0]:best_move[1]+1])
            if best_move_distance < best_distance:
                best_distance = best_move_distance
                best_tour = current_tour[:]
        else:
            break

    return [best_tour + [best_tour[0]], best_distance]

# Unmodified functions:
# - distance_calc
# - local_search_2_swap
# - local_search_2_opt
# - local_search_4_opt_stochastic
# - build_stm_and_ltm