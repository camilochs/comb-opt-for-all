
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

############################################################################

# Function: Generate 3-opt Moves
def generate_3opt_moves(route, i, j, k):
    """
    Generates all possible 3-opt moves for given indices i, j, k.

    Improvement:
    This helper function generates all possible permutations resulting from performing a 3-opt move on the route.
    It creates new routes by reversing and swapping segments between the indices.

    Enhancement:
    By systematically generating all possible 3-opt moves, the local search can explore a wider neighborhood,
    increasing the chances of finding better solutions.

    Technique Based On:
    - Lin, S., Kernighan, B. W. (1973). An effective heuristic algorithm for the traveling-salesman problem.
    """
    segments = [route[:i], route[i:j], route[j:k], route[k:]]
    moves = []

    # All possible combinations of 2 segments reversed or swapped
    moves.append(segments[0] + segments[1][::-1] + segments[2] + segments[3])
    moves.append(segments[0] + segments[1] + segments[2][::-1] + segments[3])
    moves.append(segments[0] + segments[2] + segments[1] + segments[3])
    moves.append(segments[0] + segments[2][::-1] + segments[1][::-1] + segments[3])
    moves.append(segments[0] + segments[1][::-1] + segments[2][::-1] + segments[3])
    moves.append(segments[0] + segments[2][::-1] + segments[1] + segments[3])
    moves.append(segments[0] + segments[2] + segments[1][::-1] + segments[3])
    return moves

############################################################################

# Function: 3-opt Local Search
def local_search_3_opt(distance_matrix, city_tour, recursive_seeding = -1, verbose = True):
    """
    Performs 3-opt local search on the TSP tour.

    Improvement:
    Implements a 3-opt local search, which considers swaps that remove three edges and reconnect the tour in a different way.
    This allows for more extensive exploration of the neighborhood compared to 2-opt, potentially escaping local minima.

    Enhancement:
    By considering more possible moves in each iteration, the algorithm can find better quality solutions.
    3-opt is known to produce shorter tours than 2-opt in most cases.

    Technique Based On:
    - Lin, S., Kernighan, B. W. (1973). An effective heuristic algorithm for the traveling-salesman problem.
    """
    if (recursive_seeding < 0):
        count = -2
    else:
        count = 0
    city_list = [city_tour[0][:-1], city_tour[1]]
    distance  = city_list[1]*2
    iteration = 0
    while (count < recursive_seeding):
        improvement = False
        if (verbose == True):
            print('Iteration = ', iteration, 'Distance = ', round(city_list[1], 2))
        for i in range(1, len(city_list[0]) - 2):
            for j in range(i+1, len(city_list[0]) - 1):
                for k in range(j+1, len(city_list[0])):
                    new_routes = generate_3opt_moves(city_list[0], i, j, k)
                    for route in new_routes:
                        trial_route = [route + [route[0]], 0]
                        trial_route[1] = distance_calc(distance_matrix, trial_route)
                        if trial_route[1] < city_list[1]:
                            city_list = [route, trial_route[1]]
                            improvement = True
        count     = count + 1
        iteration = iteration + 1  
        if (improvement == False and recursive_seeding < 0):
            break
    city_list[0] = city_list[0] + [city_list[0][0]]
    return city_list[0], city_list[1]

############################################################################

# Function: Update Solution with Double-Bridge Move
def update_solution(distance_matrix, guess):
    """
    Performs a double-bridge move on the tour to create a new solution.

    Improvement:
    Implements a double-bridge move, which is a large-scale perturbation that cuts the tour into four segments and reconnects them in a new order.
    This move helps the algorithm escape local minima that cannot be escaped using local searches like 2-opt or 3-opt.

    Enhancement:
    By significantly altering the tour structure, the double-bridge move allows the algorithm to explore distant regions of the solution space, improving the chances of finding a better global solution.

    Technique Based On:
    - Martin, O., Otto, S. W., Felten, E. W. (1991). Large-step Markov chains for the TSP incorporating local search heuristics. Operations Research Letters.
    """
    n = len(guess[0]) - 1  # number of cities excluding the return to the first
    pos = sorted(random.sample(range(1, n), 4))
    a, b, c, d = pos
    new_route = guess[0][0:a] + guess[0][c:d] + guess[0][b:c] + guess[0][a:b] + guess[0][d:]
    cl = [new_route + [new_route[0]], 0]
    cl[1] = distance_calc(distance_matrix, cl)
    return cl

############################################################################

# Function: Simulated Annealing with Improvements
def simulated_annealing_tsp(distance_matrix, initial_temperature = 1.0, temperature_iterations = 10, final_temperature = 0.0001, alpha = 0.9,  time_limit=10, best_c=None,verbose = True):
    """
    Solves the TSP using an improved Simulated Annealing algorithm.

    Improvements:
    1. Adaptive Cooling Schedule:
       Implements an adaptive cooling schedule where the temperature is adjusted based on the acceptance rate of new solutions.
       This allows the algorithm to dynamically adjust the exploration level, promoting better convergence.

    2. Double-Bridge Move:
       Uses a double-bridge move as the perturbation method, which makes large alterations to the tour structure.
       This helps the algorithm escape local minima that are difficult to escape with small perturbations.

    3. 3-opt Local Search:
       Applies a 3-opt local search instead of 2-opt, considering a broader neighborhood.
       This enhances the ability to find high-quality local improvements.

    Enhancement:
    These improvements collectively enhance both the solution quality and convergence speed by effectively balancing exploration and exploitation during the search process.

    Techniques Based On:
    - Adaptive Cooling Schedule:
      - Nourani, Y., Andresen, B. (1998). A comparison of simulated annealing cooling strategies. Journal of Physics A: Mathematical and General.
    - Double-Bridge Move:
      - Martin, O., Otto, S. W., Felten, E. W. (1991). Large-step Markov chains for the TSP incorporating local search heuristics.
    - 3-opt Local Search:
      - Lin, S., Kernighan, B. W. (1973). An effective heuristic algorithm for the traveling-salesman problem.

    """
    import time
    start_time = time.time()
    
    guess       = seed_function(distance_matrix)
    best        = copy.deepcopy(guess)
    temperature = float(initial_temperature)
    fx_best     = best[1]
    iteration = 0
    best_c.put(best[1])
    
    while (temperature > final_temperature):
        accepted = 0
        for repeat in range(0, temperature_iterations):
            iteration += 1
            if (verbose == True):
                print('Temperature = ', round(temperature, 4), ' ; Iteration = ', iteration, ' ; Distance = ', round(best[1], 2))
            fx_old    = guess[1]
            new_guess = update_solution(distance_matrix, guess)
            new_guess = local_search_3_opt(distance_matrix, new_guess, recursive_seeding = -1, verbose = False)
            fx_new    = new_guess[1]
            delta     = (fx_new - fx_old)
            r         = np.random.rand()
            p         = np.exp(-delta/temperature) if delta > 0 else 1.0
            if (delta < 0 or r <= p):
                guess = copy.deepcopy(new_guess)
                accepted += 1
            if (guess[1] < best[1]):
                best = copy.deepcopy(guess)
                best_c.put(best[1])
                time.sleep(0.1)
        acceptance_rate = accepted / temperature_iterations
        # Adaptive cooling based on acceptance rate
        if acceptance_rate > 0.96:
            temperature *= 0.5
        elif acceptance_rate > 0.8:
            temperature *= 0.9
        elif acceptance_rate > 0.15:
            temperature *= 0.95
        else:
            temperature *= 0.8
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break 
        if temperature <= final_temperature:
            temperature = initial_temperature
    route, distance = best
    return route, distance

############################################################################

# Unmodified functions from the original code:
# - seed_function
# - distance_calc

############################################################################
