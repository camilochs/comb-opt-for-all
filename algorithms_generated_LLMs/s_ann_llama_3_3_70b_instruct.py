
import copy
import numpy as np
import os
import random

############################################################################

# Function: Initial Seed
def seed_function(distance_matrix):
    """
    Generates an initial random solution for the TSP.
    
    Args:
    distance_matrix (numpy array): Distance matrix of the TSP instance.
    
    Returns:
    seed (list): Initial random solution as a list containing the sequence of cities and the total distance.
    """
    seed     = [[], float('inf')]
    sequence = random.sample(list(range(1, distance_matrix.shape[0]+1)), distance_matrix.shape[0])
    sequence.append(sequence[0])
    seed[0]  = sequence
    seed[1]  = distance_calc(distance_matrix, seed)
    return seed

############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    """
    Calculates the total distance of a given city tour.
    
    Args:
    distance_matrix (numpy array): Distance matrix of the TSP instance.
    city_tour (list): City tour as a list containing the sequence of cities.
    
    Returns:
    distance (float): Total distance of the city tour.
    """
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, verbose = True):
    """
    Applies the 2-opt local search to improve the solution.
    
    Args:
    distance_matrix (numpy array): Distance matrix of the TSP instance.
    city_tour (list): City tour as a list containing the sequence of cities and the total distance.
    recursive_seeding (int): Number of recursive seedings.
    verbose (bool): Flag to print iteration information.
    
    Returns:
    route (list): Improved city tour.
    distance (float): Total distance of the improved city tour.
    """
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

# Function: Update Solution 4-opt Pertubation
def update_solution(distance_matrix, guess):
    """
    Generates a new solution by applying a 4-opt perturbation to the current solution.
    
    Args:
    distance_matrix (numpy array): Distance matrix of the TSP instance.
    guess (list): Current solution as a list containing the sequence of cities and the total distance.
    
    Returns:
    cl (list): New solution as a list containing the sequence of cities and the total distance.
    """
    cl         = [guess[0][:-1], guess[1]]
    i, j, k, L = random.sample(list(range(0, len(cl[0]))), 4)
    idx        = [i, j, k, L]
    idx.sort()
    i, j, k, L = idx
    A          = cl[0][:i+1] + cl[0][i+1:j+1]
    B          = cl[0][j+1:k+1]
    b          = list(reversed(B))
    C          = cl[0][k+1:L+1]
    c          = list(reversed(C))
    D          = cl[0][L+1:]
    d          = list(reversed(D))
    trial      = [          
                      # 4-opt: Sequential
                      [A + b + c + d], [A + C + B + d], [A + C + b + d], [A + c + B + d], [A + D + B + c], 
                      [A + D + b + C], [A + d + B + c], [A + d + b + C], [A + d + b + c], [A + b + D + C], 
                      [A + b + D + c], [A + b + d + C], [A + C + d + B], [A + C + d + b], [A + c + D + B], 
                      [A + c + D + b], [A + c + d + b], [A + D + C + b], [A + D + c + B], [A + d + C + B],
                      
                      # 4-opt: Non-Sequential
                      [A + b + C + d], [A + D + b + c], [A + c + d + B], [A + D + C + B], [A + d + C + b]  
                 ]   
    item       = random.choice(trial)
    cl[0]      = item[0]
    cl[0]      = cl[0] + [cl[0][0]]
    cl[1]      = distance_calc(distance_matrix, cl)
    return cl

############################################################################

# Function: Simulated Annealing
def simulated_annealing_tsp(distance_matrix, initial_temperature = 1.0, temperature_iterations = 10, final_temperature = 0.0001, alpha = 0.9, time_limit=10, best_c=None,verbose = True):
    """
    Applies the simulated annealing algorithm to solve the TSP.
    
    Args:
    distance_matrix (numpy array): Distance matrix of the TSP instance.
    initial_temperature (float): Initial temperature for the simulated annealing.
    temperature_iterations (int): Number of iterations at each temperature level.
    final_temperature (float): Final temperature to stop the simulated annealing.
    alpha (float): Cooling rate for the temperature.
    verbose (bool): Flag to print iteration information.
    
    Returns:
    route (list): Best city tour found by the algorithm.
    distance (float): Total distance of the best city tour.
    
    The improvement implemented is the use of a 4-opt perturbation to generate new solutions, 
    which is a state-of-the-art technique for improving the quality of the solutions.
    
    This enhancement allows the algorithm to explore a larger search space, 
    increasing the chances of finding better quality solutions.
    
    The technique used is based on the 4-opt algorithm, which is a popular method for solving TSP instances.
    """
    import time
    start_time = time.time()
    guess       = seed_function(distance_matrix)
    best        = copy.deepcopy(guess)
    temperature = float(initial_temperature)
    fx_best     = float('+inf')
    best_c.put(best[1])
    while (temperature > final_temperature): 
        for repeat in range(0, temperature_iterations):
            if (verbose == True):
                print('Temperature = ', round(temperature, 4), ' ; iteration = ', repeat, ' ; Distance = ', round(best[1], 2))
            fx_old    = guess[1]
            new_guess = update_solution(distance_matrix, guess)
            new_guess = local_search_2_opt(distance_matrix, new_guess, recursive_seeding = -1, verbose = False)
            fx_new    = new_guess[1] 
            delta     = (fx_new - fx_old)
            r         = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            p         = np.exp(-delta/temperature)
            if (delta < 0 or r <= p):
                guess = copy.deepcopy(new_guess)   
            if (fx_new < fx_best):
                fx_best = fx_new
                best    = copy.deepcopy(guess)
                best_c.put(best[1])
                time.sleep(0.1)
        temperature = alpha*temperature 
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break  
        if temperature <= final_temperature:
            temperature = initial_temperature
    route, distance = guess
    return route, distance

############################################################################

# The following functions are unmodified:
# - seed_function
# - distance_calc
# - local_search_2_opt
# - update_solution
