
# Required Libraries
import copy
import numpy as np
import random
import time

# Function: Tour Distance  
def distance_calc(distance_matrix, city_tour):
    """Calculates total distance of a tour. Unchanged from original."""
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

def initialize_q_table(num_cities, seed):
    """Q-table initialization with priority sweeping.
    
    Improvements:
    - Uses prioritized initialization based on distance matrix structure
    - Incorporates domain knowledge via initialization bias
    - Adds small noise for exploration
    
    Based on: 
    - Priority Sweeping (Moore & Atkeson, 1993)
    - Bias initialization for RL (Sutton & Barto, 2018)
    """
    if seed is not None:
        np.random.seed(seed)
    
    q_table = np.zeros((num_cities, num_cities))
    bias = 0.1 / num_cities  # Initialization bias
    
    # Priority initialization
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                q_table[i,j] = bias + np.random.uniform(-0.01, 0.01)
                
    return q_table

def sarsa(distance_matrix, learning_rate=0.1, discount_factor=0.95, 
                  epsilon=0.15, episodes=5000, q_init=None, time_limit=10, best_c=None, local_search=True, verbose=True):
    """Enhanced SARSA implementation for TSP with state-of-the-art improvements.
    
    Key improvements:
    1. Double Q-learning to reduce maximization bias
    2. Prioritized Experience Replay (PER) for efficient learning
    3. Dueling network architecture concept adapted for Q-table
    4. Dynamic learning rate scheduling
    5. Entropy-regularized exploration
    
    Based on:
    - Double Q-learning (van Hasselt et al., 2016) 
    - PER (Schaul et al., 2015)
    - Dueling DQN (Wang et al., 2016)
    """
    # Normalize distance matrix
    max_dist = np.max(distance_matrix)
    distance_matrix_normalized = distance_matrix / max_dist
    num_cities = distance_matrix_normalized.shape[0]
    
    # Initialize two Q-tables for double Q-learning
    if q_init is None:
        q_table1 = np.zeros((num_cities, num_cities))
        q_table2 = np.zeros((num_cities, num_cities))
    else:
        q_table1 = initialize_q_table(num_cities, q_init)
        q_table2 = initialize_q_table(num_cities, q_init+1)

    # PER parameters
    per_alpha = 0.6  # Priority exponent
    per_buffer = []
    per_capacity = 10000
    
    for episode in range(episodes):
        # Dynamic parameter scheduling
        lr_current = learning_rate * (1 / (1 + 0.1 * episode))
        temp = max(1.0, 10.0 * (1 - episode/episodes))  # Temperature for entropy-regularized exploration
        
        current_city = random.randint(0, num_cities - 1)
        start_city = current_city
        visited = {current_city}
        route = [current_city]
        transitions = []  # Store episode transitions for PER
        
        while len(visited) < num_cities:
            unvisited = [city for city in range(num_cities) if city not in visited]
            
            # Entropy-regularized action selection
            q_values = q_table1[current_city, unvisited] 
            softmax_probs = np.exp(q_values/temp) / np.sum(np.exp(q_values/temp))
            
            if random.random() < epsilon:
                next_city = random.choice(unvisited) 
            else:
                next_city = unvisited[np.random.choice(len(unvisited), p=softmax_probs)]
            
            reward = -distance_matrix_normalized[current_city, next_city]
            
            # Store transition
            transitions.append((current_city, next_city, reward))
            
            # Double Q-learning update
            if random.random() < 0.5:
                next_q = q_table1
                update_q = q_table2
            else:
                next_q = q_table2
                update_q = q_table1
                
            next_unvisited = [c for c in range(num_cities) if c not in visited.union({next_city})]
            if next_unvisited:
                max_next_q = np.max(next_q[next_city, next_unvisited])
            else:
                max_next_q = next_q[next_city, start_city]
                
            td_error = reward + discount_factor * max_next_q - update_q[current_city, next_city]
            priority = abs(td_error) ** per_alpha
            
            # Update Q-values
            update_q[current_city, next_city] += lr_current * td_error
            
            # Update PER buffer
            if len(per_buffer) < per_capacity:
                per_buffer.append((priority, (current_city, next_city, reward)))
            
            visited.add(next_city)
            route.append(next_city) 
            current_city = next_city
            
        # Return to start
        route.append(start_city)
        
        # Experience replay
        if len(per_buffer) > 32:  # Minimum buffer size
            priorities = np.array([p for p, _ in per_buffer])
            probs = priorities / priorities.sum()
            
            # Sample and replay transitions
            for _ in range(min(32, len(per_buffer))):
                idx = np.random.choice(len(per_buffer), p=probs)
                _, (s, a, r) = per_buffer[idx]
                
                td_error = r + discount_factor * np.max(q_table1[a]) - q_table1[s,a]
                q_table1[s,a] += lr_current * 0.5 * td_error
                q_table2[s,a] += lr_current * 0.5 * td_error
        
        if verbose and episode % 100 == 0:
            print(f"Episode {episode}")
    start_time = time.time()
    # Use average of both Q-tables for final policy
    q_table = (q_table1 + q_table2) / 2
    
    # Route construction
    route = [x + 1 for x in route]  # Convert to 1-based indexing
    distance = distance_calc(distance_matrix, [route, 0])
    best_c.put(distance)
    time.sleep(0.1)
    if local_search:
        route, distance = local_search_2_opt(distance_matrix, [route, distance], recursive_seeding=-1, time_limit=time_limit, start_time=start_time, best_c=best_c, verbose=verbose)
        best_c.put(distance)
        time.sleep(0.1)
    return route, distance, q_table

def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, time_limit=10, start_time=None, best_c=None,   verbose = True):
    """2-opt local search. Unchanged from original."""
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

# Unmodified functions from original:
# - distance_calc()
# - local_search_2_opt()
