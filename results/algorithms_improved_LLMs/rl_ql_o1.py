
# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Lesson: Q-Learning
 
# GitHub Repository: <https://github.com/Valdecy>

############################################################################

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
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1,time_limit=10, start_time=None, best_c=None,  verbose = True):
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
    noise              = np.random.uniform(-0.01, 0.01, size = num_noisy_elements)
    q_table.flat[idx]  = noise
    return q_table

# Function:  Q-Learning
def q_learning(distance_matrix, learning_rate = 0.1, discount_factor = 0.95, epsilon = 0.15, episodes = 5000, q_init = None, time_limit=10, best_c=None, local_search = True, verbose = True):
    """
    This function implements Q-learning for the Traveling Salesman Problem (TSP) with Softmax action selection.

    **Improvement Implemented:**
    The standard epsilon-greedy action selection strategy is replaced with Softmax action selection. Instead of selecting 
    actions based on a fixed epsilon probability, actions are chosen probabilistically using a Softmax distribution over 
    Q-values.

    **How It Enhances Performance:**
    Softmax action selection allows for a smoother transition from exploration to exploitation compared to epsilon-greedy.
    It ensures that all actions have a chance to be selected, with higher probability given to actions with higher 
    estimated Q-values. This leads to better exploration of the action space and can prevent the algorithm from getting 
    stuck in suboptimal policies.

    **State-of-the-Art Technique It's Based On:**
    The Softmax (Boltzmann) action selection is a widely used strategy in reinforcement learning to balance exploration 
    and exploitation by probabilistically selecting actions according to their estimated value. It is considered a 
    state-of-the-art technique for action selection in RL algorithms.

    Parameters:
    - distance_matrix: A 2D numpy array representing the distances between cities.
    - learning_rate: The rate at which the algorithm updates its knowledge.
    - discount_factor: The factor by which future rewards are discounted.
    - epsilon: Initial epsilon value (unused in Softmax, kept for compatibility).
    - episodes: Number of episodes to run the Q-learning algorithm.
    - q_init: Seed for initializing the Q-table, if any.
    - local_search: Boolean indicating whether to apply local search (2-opt) after Q-learning.
    - verbose: Boolean indicating whether to print progress information.

    Returns:
    - route: The best route found as a list of city indices.
    - distance: The total distance of the best route.
    - q_table: The learned Q-table.
    """

    max_dist        = np.max(distance_matrix)
    distance_matrix = distance_matrix / max_dist
    num_cities      = distance_matrix.shape[0]
    if q_init is None:
        q_table = np.zeros((num_cities, num_cities))
    else:
        q_table = initialize_q_table(num_cities, q_init) 
    
    temperature = 1.0  # Temperature parameter for Softmax action selection
    
    for episode in range(episodes):
        current_city = random.randint(0, num_cities - 1)
        visited      = set([current_city])
        while len(visited) < num_cities:
            unvisited_cities = [city for city in range(num_cities) if city not in visited]
            preferences = q_table[current_city, unvisited_cities]
            # Numerical stability fix
            max_pref = np.max(preferences)
            preferences = preferences - max_pref
            exp_preferences = np.exp(preferences / temperature)
            probabilities = exp_preferences / np.sum(exp_preferences)
            next_city = np.random.choice(unvisited_cities, p=probabilities)
            reward = -distance_matrix[current_city, next_city]
            max_future_q = np.max(q_table[next_city, [city for city in range(num_cities) if city not in visited.union({next_city})]]) if len(visited) + 1 < num_cities else 0
            q_table[current_city, next_city] = q_table[current_city, next_city] + learning_rate * (reward + discount_factor * max_future_q - q_table[current_city, next_city])
            current_city = next_city
            visited.add(current_city)
        if verbose and episode % 100 == 0:
            print(f"Episode {episode}")
    
    start_time = time.time()
    distance_matrix = distance_matrix * max_dist
    start_city      = 0
    current_city    = start_city
    visited         = set([current_city])
    route           = [current_city]
    distance        = 0
    while len(visited) < num_cities:
        unvisited_cities = [city for city in range(num_cities) if city not in visited]
        next_city = unvisited_cities[np.argmax(q_table[current_city, unvisited_cities])]
        route.append(next_city)
        visited.add(next_city)
        distance     = distance + distance_matrix[current_city, next_city]
        best_c.put(distance)
        time.sleep(0.1)
        current_city = next_city
    route.append(start_city)
    distance        = distance + distance_matrix[current_city, start_city]
    best_c.put(distance)
    time.sleep(0.1)
    route           = [node + 1 for node in route]
    seed            = [route, distance]
    if local_search:
        route, distance = local_search_2_opt(distance_matrix, seed, recursive_seeding = -1, time_limit=time_limit, start_time=start_time, best_c=best_c, verbose = verbose)
        best_c.put(distance)
        time.sleep(0.1)
    return route, distance, q_table 

############################################################################

# Unmodified functions from the original code:
# - distance_calc
# - local_search_2_opt
# - initialize_q_table
