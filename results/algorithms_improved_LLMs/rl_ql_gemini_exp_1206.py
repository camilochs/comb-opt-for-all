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
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1,time_limit=10, start_time=None, best_c=None, verbose = True):
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
def q_learning(distance_matrix, learning_rate = 0.1, discount_factor = 0.95, epsilon = 0.15, episodes = 5000, q_init = None, time_limit=10, best_c=None,  local_search = True, verbose = True):
    """
    Implements the Q-learning algorithm for the Traveling Salesman Problem (TSP).

    Parameters:
        distance_matrix (numpy.ndarray): A 2D numpy array representing the distances between cities.
        learning_rate (float): The learning rate (alpha) for the Q-learning algorithm.
        discount_factor (float): The discount factor (gamma) for future rewards.
        epsilon (float): The exploration-exploitation trade-off parameter.
        episodes (int): The number of episodes to run the algorithm.
        q_init (int): Seed for Q-table initialization.
        local_search (bool): Whether to apply local search (2-opt) after Q-learning.
        verbose (bool): Whether to print progress information.

    Returns:
        tuple: A tuple containing the best route found, its distance, and the final Q-table.
    """
    max_dist        = np.max(distance_matrix)
    distance_matrix = distance_matrix/max_dist
    num_cities      = distance_matrix.shape[0]
    if (q_init == None):
        q_table = np.zeros((num_cities, num_cities))
    else:
        q_table = initialize_q_table(num_cities, q_init) 

    # Improvement 1: Epsilon Decay
    """
    Implements an epsilon decay strategy to balance exploration and exploitation.

    How it enhances performance:
        - Initially, higher epsilon encourages exploration of the state space.
        - As episodes progress, epsilon decreases, shifting the focus towards exploiting learned knowledge.
        - This dynamic adjustment helps in faster convergence and potentially finding better solutions.

    State-of-the-art technique:
        - This is a common practice in reinforcement learning to improve convergence. It's based on the idea that
          early exploration is crucial, but later exploitation of good actions is more beneficial.
    """
    initial_epsilon = epsilon
    epsilon_decay   = 0.995
    min_epsilon     = 0.01

    # Improvement 2: Double Q-Learning
    """
    Implements Double Q-learning to mitigate overestimation bias in Q-values.

    How it enhances performance:
        - Standard Q-learning can overestimate action values, leading to suboptimal policies.
        - Double Q-learning uses two Q-tables (q_table_a and q_table_b) to decouple action selection from evaluation.
        - One table is used to select the best action, and the other is used to evaluate its value.
        - This reduces overestimation and leads to more accurate value estimates and better policies.

    State-of-the-art technique:
        - Double Q-learning, proposed by Hado van Hasselt (2010), is a widely recognized technique for improving
          the stability and performance of Q-learning.
    """
    q_table_a = q_table.copy()
    q_table_b = q_table.copy()

    for episode in range(0, episodes):
        current_city = random.randint(0, num_cities - 1)
        visited      = set([current_city])
        
        # Epsilon Decay Update
        epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay ** episode))

        while (len(visited) < num_cities):
            unvisited_cities = [city for city in range(num_cities) if city not in visited]
            if (random.random() < epsilon):
                next_city = random.choice(unvisited_cities)
            else:
                # Use average of Q-tables for action selection during exploration
                avg_q_values = (q_table_a[current_city, unvisited_cities] + q_table_b[current_city, unvisited_cities]) / 2
                next_city = unvisited_cities[np.argmax(avg_q_values)]
            
            reward = -distance_matrix[current_city, next_city]

            # Double Q-Learning Update
            if random.random() < 0.5:
                best_next_action = np.argmax(q_table_a[next_city, unvisited_cities]) if unvisited_cities else 0
                q_table_a[current_city, next_city] = q_table_a[current_city, next_city] + learning_rate * (reward + discount_factor * q_table_b[next_city, unvisited_cities[best_next_action]] - q_table_a[current_city, next_city]) if unvisited_cities else q_table_a[current_city, next_city] + learning_rate * (reward - q_table_a[current_city, next_city])
            else:
                best_next_action = np.argmax(q_table_b[next_city, unvisited_cities]) if unvisited_cities else 0
                q_table_b[current_city, next_city] = q_table_b[current_city, next_city] + learning_rate * (reward + discount_factor * q_table_a[next_city, unvisited_cities[best_next_action]] - q_table_b[current_city, next_city]) if unvisited_cities else q_table_b[current_city, next_city] + learning_rate * (reward - q_table_b[current_city, next_city])

            current_city = next_city
            visited.add(current_city)
        if (verbose == True and episode % 100 == 0):
            print(f"Episode {episode}")
    
    start_time = time.time()
    # Use the average of the two Q-tables for final route construction
    q_table = (q_table_a + q_table_b) / 2

    distance_matrix = distance_matrix*max_dist
    start_city      = 0
    current_city    = start_city
    visited         = set([current_city])
    route           = [current_city]
    distance        = 0
    while (len(visited) < num_cities):
        next_city    = np.argmax([q_table[current_city, city] if city not in visited else -np.inf for city in range(0, num_cities)])
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
    if (local_search == True):
        route, distance = local_search_2_opt(distance_matrix, seed, recursive_seeding = -1, time_limit=time_limit, start_time=start_time, best_c=best_c,  verbose = verbose)
        best_c.put(distance)
        time.sleep(0.1)
    return route, distance, q_table

############################################################################

# Unmodified functions: distance_calc, local_search_2_opt, initialize_q_table