# Required Libraries
import copy
import numpy as np
import os
import random

############################################################################

# Function: Initial Seed
def seed_function(distance_matrix):
    seed     = [[], float('inf')]
    sequence = random.sample(list(range(1, distance_matrix.shape[0]+1)), distance_matrix.shape[0])
    sequence.append(sequence[0])
    seed[0]  = sequence
    seed[1]  = distance_calc(distance_matrix, seed)
    return seed

############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, verbose = True):
    if (recursive_seeding < 0):
        count = -2
    else:
        count = 0
    city_list = copy.deepcopy(city_tour)
    distance  = city_list[1]*2
    iteration = 0
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

# Function: Update Solution with Adaptive Large Neighborhood Search (ALNS)
def update_solution_alns(distance_matrix, guess, operator_weights, operator_history, temperature):
    """
    Improves the solution update mechanism using Adaptive Large Neighborhood Search (ALNS).

    ALNS is a metaheuristic that adaptively selects from a set of neighborhood operators based on their past performance.
    This implementation uses 2-opt, 3-opt, and a modified 4-opt as operators.

    How it enhances performance:
        - Diversification: By using multiple operators, ALNS explores a wider range of the search space, potentially escaping local optima.
        - Intensification: The adaptive selection mechanism focuses on operators that have historically led to better solutions, intensifying the search in promising areas.
        - Adaptability: The operator weights are dynamically adjusted based on their performance, allowing the algorithm to adapt to the specific problem instance.

    State-of-the-art technique:
        Based on the Adaptive Large Neighborhood Search (ALNS) metaheuristic, which is a popular and effective technique for solving combinatorial optimization problems.
    """
    
    cl         = [guess[0][:-1], guess[1]]
    
    # Operator selection based on roulette wheel selection
    total_weight = sum(operator_weights.values())
    probabilities = {op: weight / total_weight for op, weight in operator_weights.items()}
    selected_operator = random.choices(list(probabilities.keys()), weights=list(probabilities.values()))[0]
    
    if selected_operator == '2-opt':
        # 2-opt operator
        i, j = random.sample(list(range(0, len(cl[0]))), 2)
        if i > j:
            i, j = j, i
        cl[0][i:j] = list(reversed(cl[0][i:j]))
    
    elif selected_operator == '3-opt':
        # 3-opt operator (simplified version)
        i, j, k = random.sample(list(range(0, len(cl[0]))), 3)
        idx = [i, j, k]
        idx.sort()
        i, j, k = idx
        
        A = cl[0][:i]
        B = cl[0][i:j]
        C = cl[0][j:k]
        D = cl[0][k:]
        
        # Choose one of the possible 3-opt moves
        options = [
            A + C + B + D,
            A + B[::-1] + C + D,
            A + B + C[::-1] + D
        ]
        cl[0] = random.choice(options)
        
    elif selected_operator == '4-opt':
        # Modified 4-opt operator (reduced set of moves for faster execution)
        i, j, k, L = random.sample(list(range(0, len(cl[0]))), 4)
        idx        = [i, j, k, L]
        idx.sort()
        i, j, k, L = idx
        A          = cl[0][:i+1]
        B          = cl[0][i+1:j+1]
        b          = list(reversed(B))
        C          = cl[0][j+1:k+1]
        c          = list(reversed(C))
        D          = cl[0][k+1:L+1]
        d          = list(reversed(D))
        E          = cl[0][L+1:]

        trial      = [          
                      [A + b + C + d + E], [A + D + b + c + E], [A + c + d + B + E], [A + D + C + B + E], [A + d + C + b + E]  
                     ]   
        item       = random.choice(trial)
        cl[0]      = item[0]
    
    cl[0]      = cl[0] + [cl[0][0]]
    cl[1]      = distance_calc(distance_matrix, cl)

    # Update operator history and weights
    operator_history[selected_operator].append(cl[1])
    
    # Update weights based on performance
    rho = 0.1  # Reaction factor
    
    if len(operator_history[selected_operator]) >= 50: # Only update weights after a certain number of uses
        avg_improvement = np.mean(operator_history[selected_operator][-50:])
        operator_weights[selected_operator] = operator_weights[selected_operator] * (1 - rho) + rho * avg_improvement / temperature

        # Normalize weights
        total_weight = sum(operator_weights.values())
        for op in operator_weights:
            operator_weights[op] /= total_weight

    return cl

############################################################################

# Function: Simulated Annealing
def simulated_annealing_tsp(distance_matrix, initial_temperature = 1.0, temperature_iterations = 10, final_temperature = 0.0001, alpha = 0.9,time_limit=10, best_c=None, verbose = True):
    """
    Implements the Simulated Annealing algorithm for the Traveling Salesman Problem (TSP).

    Improvements:
        - Adaptive Large Neighborhood Search (ALNS): Uses ALNS to dynamically select from a set of neighborhood operators (2-opt, 3-opt, and a modified 4-opt) based on their past performance.
        - Reheating: Implements a reheating mechanism to escape local optima. When the algorithm gets stuck, the temperature is increased to allow for more exploration.
        - Adaptive Cooling Schedule: Modifies the cooling schedule to be adaptive. The cooling rate (alpha) is adjusted based on the acceptance rate of new solutions. This allows for faster cooling when solutions are readily accepted and slower cooling when the algorithm is struggling to find improvements.

    How they enhance performance:
        - ALNS: Improves diversification and intensification by using multiple operators and adaptively selecting them based on their performance.
        - Reheating: Helps escape local optima by increasing the temperature and allowing for more exploration.
        - Adaptive Cooling Schedule: Optimizes the cooling process by adapting the cooling rate to the current state of the search. This can lead to faster convergence and better solutions.

    State-of-the-art techniques:
        - Adaptive Large Neighborhood Search (ALNS): A popular and effective metaheuristic for combinatorial optimization problems.
        - Reheating: A common technique in Simulated Annealing to escape local optima.
        - Adaptive Cooling Schedule: A more sophisticated cooling schedule that can improve the performance of Simulated Annealing.
    """
    import time
    start_time = time.time()

    guess       = seed_function(distance_matrix)
    best        = copy.deepcopy(guess)
    temperature = float(initial_temperature)
    fx_best     = float('+inf')
    
    # Initialize ALNS parameters
    operator_weights = {'2-opt': 1.0, '3-opt': 1.0, '4-opt': 1.0}
    operator_history = {'2-opt': [], '3-opt': [], '4-opt': []}
    
    # Reheating parameters
    reheating_interval = 50 # Number of temperature iterations before reheating
    reheating_factor = 2.0 # Factor by which to increase the temperature during reheating
    reheating_counter = 0
    
    # Adaptive cooling parameters
    acceptance_rate_target = 0.4 # Target acceptance rate
    alpha_min = 0.8 # Minimum cooling rate
    alpha_max = 0.99 # Maximum cooling rate
    alpha_update_factor = 0.05 # Factor by which to update alpha

    accepted_count = 0
    iteration_count = 0
    
    best_c.put(best[1])
    while (temperature > final_temperature): 
        for repeat in range(0, temperature_iterations):
            if (verbose == True):
                print('Temperature = ', round(temperature, 4), ' ; iteration = ', repeat, ' ; Distance = ', round(best[1], 2))
            fx_old    = guess[1]
            new_guess = update_solution_alns(distance_matrix, guess, operator_weights, operator_history, temperature)
            new_guess = local_search_2_opt(distance_matrix, new_guess, recursive_seeding = -1, verbose = False)
            fx_new    = new_guess[1] 
            delta     = (fx_new - fx_old)
            r         = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            p         = np.exp(-delta/temperature)
            if (delta < 0 or r <= p):
                guess = copy.deepcopy(new_guess)
                accepted_count += 1
            if (fx_new < fx_best):
                fx_best = fx_new
                best    = copy.deepcopy(guess)
                best_c.put(best[1])
                time.sleep(0.1)
        
        iteration_count += temperature_iterations
        acceptance_rate = accepted_count / iteration_count

        # Adaptive cooling schedule
        if acceptance_rate > acceptance_rate_target:
            alpha = min(alpha_max, alpha + alpha_update_factor)
        else:
            alpha = max(alpha_min, alpha - alpha_update_factor)
        
        # Reheating
        reheating_counter += 1
        if reheating_counter >= reheating_interval:
            temperature *= reheating_factor
            reheating_counter = 0

        temperature = alpha*temperature
        accepted_count = 0
        iteration_count = 0
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break 
        if temperature <= final_temperature:
            temperature = initial_temperature

    route, distance = guess
    return route, distance

############################################################################

# Unmodified functions:
# - seed_function
# - distance_calc
# - local_search_2_opt