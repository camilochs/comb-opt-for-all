
############################################################################

# Required Libraries
import copy
import numpy as np
import os
import random

############################################################################

# Function: Initial Seed
def seed_function(distance_matrix):
    """
    Generates initial solution using Nearest Neighbor heuristic for TSP. 
    Enhances quality of initial guess, leading to faster convergence. 
    Based on greedy constructive heuristic from TSP literature.
    """
    n = distance_matrix.shape[0]
    start = random.randint(0, n-1)
    current = start
    visited = [False]*n
    visited[current] = True
    tour = [current + 1]
    for _ in range(n-1):
        nearest = None
        min_dist = float('inf')
        for i in range(n):
            if not visited[i] and distance_matrix[current, i] < min_dist:
                min_dist = distance_matrix[current, i]
                nearest = i
        current = nearest
        visited[current] = True
        tour.append(current + 1)
    tour.append(tour[0])
    distance = distance_calc(distance_matrix, [tour, 0])
    return [tour, distance]

############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    """
    Unmodified original distance calculation function.
    """
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, verbose = True):
    """
    Performs 2-opt local search with first-improvement strategy to reduce computation time. 
    Immediate application of beneficial moves prevents unnecessary iterations. 
    Based on efficient TSP local search implementations.
    """
    city_list = copy.deepcopy(city_tour)
    improved = True
    iteration = 0
    while improved:
        improved = False
        seed_route = copy.deepcopy(city_list[0])
        seed_dist = city_list[1]
        for i in range(0, len(city_list[0]) - 2):
            for j in range(i+2, len(city_list[0]) - 1):
                new_route = seed_route[:i+1] + seed_route[i+1:j+1][::-1] + seed_route[j+1:]
                new_route[-1] = new_route[0]
                new_distance = distance_calc(distance_matrix, [new_route, 0])
                if new_distance < seed_dist:
                    city_list = [new_route, new_distance]
                    improved = True
                    break
            if improved:
                break
        iteration += 1
    return city_list[0], city_list[1]

############################################################################

# Function: Update Solution
def update_solution(distance_matrix, guess):
    """
    Applies double bridge 4-opt perturbation (a.k.a. quadruple move) to escape local optima. 
    Part of state-of-the-art Lin-Kernighan heuristic, providing more strategic exploration 
    than random moves while maintaining computational efficiency.
    """
    cl = [guess[0][:-1], guess[1]]
    if len(cl[0]) < 8:
        i = random.randint(0, len(cl[0])-2)
        j = random.randint(i+1, len(cl[0])-1)
        cl[0][i:j+1] = cl[0][i:j+1][::-1]
    else:
        points = sorted(random.sample(range(1, len(cl[0])-3), 4))
        a = cl[0][:points[0]]
        b = cl[0][points[0]:points[1]]
        c = cl[0][points[1]:points[2]]
        d = cl[0][points[2]:points[3]]
        rest = cl[0][points[3]:]
        cl[0] = a + d + c + b + rest
    cl[0].append(cl[0][0])
    cl[1] = distance_calc(distance_matrix, cl)
    return cl

############################################################################

# Function: Simulated Annealing
def simulated_annealing_tsp(distance_matrix, initial_temperature = 1.0, temperature_iterations = 10, final_temperature = 0.0001, alpha = 0.9, time_limit=10, best_c=None, verbose = True):
    """
    Implements enhanced Simulated Annealing with: 
    1) Lundy-Mees adaptive cooling schedule for better temperature control
    2) Hybrid Monte Carlo search combining global perturbation and local optimization
    3) Threshold accepting allowing temporary quality degradation for escape
    State-of-the-art components from TSP research significantly improve solution quality and convergence speed.
    """
    import time
    start_time = time.time()
    
    guess       = seed_function(distance_matrix)
    best        = copy.deepcopy(guess)
    temperature = float(initial_temperature)
    fx_best     = float('inf')
    
    # Lundy-Mees adaptive cooling parameter
    total_steps = int(np.log(final_temperature/initial_temperature)/np.log(alpha)) + 1
    beta = (initial_temperature - final_temperature)/(total_steps*initial_temperature*final_temperature)
    
    best_c.put(best[1])
    while temperature > final_temperature:
        for repeat in range(temperature_iterations):
            if verbose:
                print('Temperature =', round(temperature,4), 'Iteration =', repeat, 'Distance =', round(best[1],2))
            
            # Threshold accepting with adaptive threshold
            threshold = temperature * 0.5
            fx_old = guess[1]
            new_guess = update_solution(distance_matrix, guess)
            
            # Probabilistic local search application
            if random.random() < 0.7:
                new_guess = local_search_2_opt(distance_matrix, new_guess, -1, False)
                
            fx_new = new_guess[1]
            delta = fx_new - fx_old
            
            if delta < threshold or np.exp(-delta/temperature) > random.random():
                guess = copy.deepcopy(new_guess)
                if fx_new < fx_best:
                    fx_best = fx_new
                    best = copy.deepcopy(guess)
                    best_c.put(best[1])
                    time.sleep(0.1)
        
        # Lundy-Mees cooling schedule
        temperature = temperature/(1 + beta*temperature)
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break 
        if temperature <= final_temperature:
            temperature = initial_temperature
    return best[0], best[1]

############################################################################

# Comment block for unmodified functions
'''
Unmodified original functions:
- distance_calc
'''
