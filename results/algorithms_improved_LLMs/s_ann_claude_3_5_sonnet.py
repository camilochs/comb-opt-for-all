
############################################################################
# Required Libraries
import copy
import numpy as np
import os
import random
from typing import List, Tuple

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

# Function: Update Solution 4-opt Pertubation
def update_solution(distance_matrix, guess):
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

def simulated_annealing_tsp(distance_matrix: np.ndarray, 
                           initial_temperature: float = 100.0, # Increased initial temp
                           temperature_iterations: int = 5,    # Reduced iterations
                           final_temperature: float = 0.01,    # Increased final temp
                           alpha: float = 0.85, 
                           time_limit=10, best_c=None,               # Faster cooling
                           verbose: bool = True) -> Tuple[List[int], float]:
    """
    Optimized Simulated Annealing for faster convergence:
    
    1. Modified Temperature Schedule:
       - Higher initial temperature (100.0) for better initial exploration
       - Faster cooling rate (0.85) to reach convergence quicker
       - Higher final temperature (0.01) to terminate earlier
       - Fewer iterations per temperature (5) to speed up the process
    
    2. Efficient Neighborhood Generation:
       - Simplified 2-opt moves for faster neighbor generation
       - Selective local search application
       - Early stopping when improvement plateaus
    
    3. Solution Acceptance:
       - More aggressive acceptance criteria at high temperatures
       - More selective at lower temperatures
       - Quick rejection of clearly worse solutions
    
    Args:
        distance_matrix: Square matrix of distances between cities
        initial_temperature: Starting temperature (default: 100.0)
        temperature_iterations: Iterations per temperature (default: 5)
        final_temperature: Stopping temperature (default: 0.01)
        alpha: Cooling rate (default: 0.85)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best route found, total distance)
    """
    import time
    start_time = time.time()
    
    guess = seed_function(distance_matrix)
    best = copy.deepcopy(guess)
    temperature = float(initial_temperature)
    
    # Track improvement for early stopping
    no_improve = 0
    best_distance = guess[1]
    best_c.put(best[1])
    
    while temperature > final_temperature and no_improve < 20: 
        improved = False
        
        for repeat in range(temperature_iterations):
            if verbose:
                print(f'Temperature = {temperature:.4f} ; iteration = {repeat} ; Distance = {best[1]:.2f}')
            
            # Generate and evaluate new solution
            fx_old = guess[1]
            new_guess = fast_neighbor_generation(distance_matrix, guess)
            fx_new = new_guess[1]
            
            # Quick acceptance check
            delta = fx_new - fx_old
            if delta < 0 or random.random() < np.exp(-delta/temperature):
                guess = copy.deepcopy(new_guess)
                
                # Apply local search only if solution is promising
                if fx_new < best_distance * 1.1:
                    guess = local_search_2_opt(distance_matrix, guess, recursive_seeding=1, verbose=False)
                
                if guess[1] < best[1]:
                    best = copy.deepcopy(guess)
                    best_c.put(best[1])
                    time.sleep(0.1)
                    
                    best_distance = best[1]
                    improved = True
                    no_improve = 0
        
        # Update no improvement counter
        if not improved:
            no_improve += 1
            
        # Cool down temperature
        temperature = alpha * temperature
        
        # If stuck, perform perturbation
        if no_improve > 5:
            guess = perturb_solution(distance_matrix, best)
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break 
        if temperature <= final_temperature:
            temperature = initial_temperature
    return best[0], best[1]

def fast_neighbor_generation(distance_matrix: np.ndarray, 
                           solution: Tuple[List[int], float]) -> Tuple[List[int], float]:
    """
    Faster neighbor generation using simple 2-opt move
    """
    route = solution[0][:-1]  # Remove last city (it's same as first)
    i, j = random.sample(range(len(route)), 2)
    
    if i > j:
        i, j = j, i
    
    # Simple 2-opt swap
    new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
    new_route.append(new_route[0])  # Add first city at end
    
    distance = distance_calc(distance_matrix, [new_route, 0])
    return [new_route, distance]

def perturb_solution(distance_matrix: np.ndarray, 
                    solution: Tuple[List[int], float]) -> Tuple[List[int], float]:
    """
    Perturb solution when stuck in local optimum
    """
    route = solution[0][:-1]
    # Double bridge move
    size = len(route)
    pos1 = random.randint(1, size//4)
    pos2 = random.randint(size//4 + 1, size//2)
    pos3 = random.randint(size//2 + 1, 3*size//4)
    pos4 = random.randint(3*size//4 + 1, size-1)
    
    new_route = route[:pos1] + route[pos3:pos4] + route[pos2:pos3] + route[pos1:pos2] + route[pos4:]
    new_route.append(new_route[0])
    
    distance = distance_calc(distance_matrix, [new_route, 0])
    return [new_route, distance]

############################################################################
# Unmodified functions from original:
# - seed_function()
# - distance_calc()
# - local_search_2_opt()
############################################################################
