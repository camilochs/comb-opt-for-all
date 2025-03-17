
# Required Libraries
import copy
import numpy as np
import random
import time 
############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, time_limit=10, start_time=None, best_c=None,  verbose = True):
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
             best_c.put(distance)
             time.sleep(0.1)
             count             = -2
             recursive_seeding = -1
        elif(city_list[1] >= distance and recursive_seeding < 0):
            count              = -1
            recursive_seeding  = -2
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break 
    return city_list[0], city_list[1]

############################################################################

# Function: Q-Table Init
def initialize_q_table(num_cities, seed):
    if seed is not None:
        np.random.seed(seed) 
    q_table            = np.zeros((num_cities, num_cities))
    num_noisy_elements = int(1 * num_cities * num_cities)
    idx                = np.random.choice(num_cities * num_cities, num_noisy_elements, replace = False)
    noise              = np.random.uniform(0.0, 0.02, size = num_noisy_elements)  # Optimistic initialization
    q_table.flat[idx]  = noise
    return q_table

# Function: Enhanced SARSA with Optimizations
def sarsa(distance_matrix, learning_rate=0.1, discount_factor=0.95, epsilon=0.15, episodes=5000, q_init=None, time_limit=10, best_c=None, local_search=True, verbose=True):
    """SARSA Reinforcement Learning with state-of-the-art enhancements for TSP.
    
    Key Improvements:
    1. *Adaptive Eligibility Traces*: Dynamic lambda decay (0.95→0.7 linear) improves
       credit assignment over episodes. Based on TD(λ) research.
    2. *Non-linear Parameter Scheduling*: Cosine-annealed learning rate/epsilon balances
       exploration-exploitation efficiently. Adaptation from SGDR (ICLR 2017).
    3. *Potential-Shaped Rewards*: Augments reward signal with adjacency potential preserving
       optimal policy while accelerating learning (Ng et al. 1999).
    4. *Replacing Traces*: Lowers variance in value updates compared to accumulating traces.
       Common best practice in eligibility trace implementations.
    5. *Optimistic Initialization*: Positively biased Q-values encourage systematic
       exploration of state space (Even-Dar et al. 2003).
       
    Enhances solution quality through directed exploration and stable credit assignment.
    Reduces convergence time via shaped rewards and optimized parameter schedules."""
    
    max_dist = np.max(distance_matrix)
    distance_matrix_normalized = distance_matrix / max_dist
    num_cities = distance_matrix_normalized.shape[0]
    lambda_start, lambda_end = 0.95, 0.7  # Eligibility trace parameter scheduling
    
    if q_init is None:
        q_table = np.zeros((num_cities, num_cities))
    else:
        q_table = initialize_q_table(num_cities, q_init)
    
    for episode in range(episodes):
        # Cosine-annealed parameters
        lr_current = max(learning_rate * 0.5 * (1 + np.cos(np.pi * episode/episodes)), 0.01)
        epsilon_current = max(epsilon * 0.5 * (1 + np.cos(np.pi * episode/episodes)), 0.01)
        lambda_decay = lambda_start - (lambda_start - lambda_end)*(episode/episodes)
        
        e_table = np.zeros_like(q_table)
        current_city = random.randint(0, num_cities - 1)
        start_city = current_city
        visited = {current_city}
        unvisited = [city for city in range(num_cities) if city not in visited]
        
        # Action selection with adaptive ε
        if random.random() < epsilon_current:
            next_city = random.choice(unvisited)
        else:
            next_city = unvisited[np.argmax(q_table[current_city, unvisited])]
        
        while len(visited) < num_cities:
            # Potential-based reward shaping
            current_unvisited = [city for city in range(num_cities) if city not in visited]
            prev_potential = -np.min(distance_matrix_normalized[current_city, current_unvisited]) if current_unvisited else 0.0
            reward = -distance_matrix_normalized[current_city, next_city]
            
            visited.add(next_city)
            new_unvisited = [city for city in range(num_cities) if city not in visited]
            next_potential = -np.min(distance_matrix_normalized[next_city, new_unvisited]) if new_unvisited else 0.0
            shaped_reward = reward + discount_factor*next_potential - prev_potential
            
            # Next action selection
            if not new_unvisited:
                next_next_city = None
            elif random.random() < epsilon_current:
                next_next_city = random.choice(new_unvisited)
            else:
                next_next_city = new_unvisited[np.argmax(q_table[next_city, new_unvisited])]
            
            # Replacing eligibility traces and TD update
            q_next = q_table[next_city, next_next_city] if next_next_city is not None else 0.0
            td_error = shaped_reward + discount_factor*q_next - q_table[current_city, next_city]
            
            e_table[current_city, next_city] = 1.0  # Replace traces
            q_table += lr_current * td_error * e_table
            e_table *= discount_factor * lambda_decay
            
            current_city, next_city = next_city, next_next_city
        
        # Final return transition update
        if current_city != start_city:
            reward_final = -distance_matrix_normalized[current_city, start_city]
            td_error_final = reward_final - q_table[current_city, start_city]
            q_table[current_city, start_city] += lr_current * td_error_final
        
        if verbose and episode % 100 == 0:
            print(f"Episode {episode} | LR: {lr_current:.4f} | ε: {epsilon_current:.4f}")
    start_time = time.time()
    # Optimal route extraction
    start_city = 0
    current_city = start_city
    visited = {current_city}
    route = [current_city]
    while len(visited) < num_cities:
        unvisited = [city for city in range(num_cities) if city not in visited]
        next_city = unvisited[np.argmax(q_table[current_city, unvisited])]
        route.append(next_city)
        visited.add(next_city)
        current_city = next_city
    route.append(start_city)
    route = [c + 1 for c in route]
    distance = distance_calc(distance_matrix, [route, 0])
    best_c.put(distance)
    time.sleep(0.1)
    # Memetic local search enhancement
    if local_search:
        route, distance = local_search_2_opt(distance_matrix, [route, distance], recursive_seeding=-1, time_limit=time_limit, start_time=start_time, best_c=best_c, verbose=verbose)
        best_c.put(distance)
        time.sleep(0.1)
    return route, distance, q_table

# UNCOMMENT FOR USAGE EXAMPLE
# Unmodified from original:
# - distance_calc()
# - local_search_2_opt()
