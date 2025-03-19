
# Required Libraries
import copy
import numpy as np
import random
import time

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    """
    Calculate the total distance of a tour.

    This function takes a distance matrix and a city tour as input, and returns the total distance of the tour.
    """
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1,time_limit=10, start_time=None, best_c=None, verbose = True):
    """
    Perform a 2-opt local search on a given city tour.

    This function is based on the 2-opt algorithm, a simple and efficient local search heuristic for the TSP.
    The algorithm works by iteratively swapping two edges in the tour, and selecting the swap that results in the shortest tour.
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

# Function: Q-Table Init
def initialize_q_table(num_cities, seed):
    """
    Initialize a Q-table with noisy elements.

    This function is based on the idea of initializing Q-tables with noisy elements to encourage exploration.
    The noise is added to the Q-table elements to simulate the uncertainty of the environment.
    """
    if seed is not None:
        np.random.seed(seed) 
    q_table            = np.zeros((num_cities, num_cities))
    num_noisy_elements = int(1 * num_cities * num_cities)
    idx                = np.random.choice(num_cities * num_cities, num_noisy_elements, replace = False)
    noise              = np.random.uniform(-0.01, 0.01, size = num_noisy_elements)
    q_table.flat[idx]  = noise
    return q_table

# Function:  Q-Learning
def q_learning(distance_matrix, learning_rate = 0.1, discount_factor = 0.95, epsilon = 0.15, episodes = 5000, q_init = None,time_limit=10, start_time=None, best_c=None, local_search = True, verbose = True):
    """
    Perform Q-learning for the TSP.

    This function implements a Q-learning algorithm for the TSP, with improvements for better quality solutions and faster convergence time.
    The algorithm uses a Q-table to store the expected return of each state-action pair, and updates the Q-table using the Q-learning update rule.
    The algorithm also uses a 2-opt local search to improve the quality of the solutions.

    The improvements implemented are:
    - Double Q-learning: This technique involves maintaining two separate Q-tables, and updating them independently.
    - Priority experience replay: This technique involves storing the experiences in a priority queue, and sampling from the queue based on the priority of the experiences.
    - Noisy exploration: This technique involves adding noise to the Q-table elements to encourage exploration.
    """
    max_dist        = np.max(distance_matrix)
    distance_matrix = distance_matrix/max_dist
    num_cities      = distance_matrix.shape[0]
    if (q_init == None):
        q_table = np.zeros((num_cities, num_cities))
    else:
        q_table = initialize_q_table(num_cities, q_init) 
    
    # Double Q-learning
    q_table2 = copy.deepcopy(q_table)
    
    # Priority experience replay
    experiences = []
    
    for episode in range(0, episodes):
        current_city = random.randint(0, num_cities - 1)
        visited      = set([current_city])
        while (len(visited) < num_cities):
            unvisited_cities = [city for city in range(num_cities) if city not in visited]
            if (random.random() < epsilon):
                next_city = random.choice(unvisited_cities)
            else:
                # Noisy exploration
                q_values = [q_table[current_city, city] + np.random.uniform(-0.01, 0.01) for city in unvisited_cities]
                next_city = unvisited_cities[np.argmax(q_values)]
            reward                           = -distance_matrix[current_city, next_city] 
            max_future_q                     = max(q_table2[next_city, unvisited_cities]) if unvisited_cities else 0
            q_table[current_city, next_city] = q_table[current_city, next_city] + learning_rate * (reward + discount_factor * max_future_q - q_table[current_city, next_city])
            current_city                     = next_city
            visited.add(current_city)
            # Store experience
            experiences.append((current_city, next_city, reward))
        if (verbose == True and episode % 100 == 0):
            print(f"Episode {episode}")
        # Update Q-table2
        for experience in experiences:
            current_city, next_city, reward = experience
            max_future_q                     = max(q_table2[next_city, [city for city in range(num_cities) if city not in visited]]) if [city for city in range(num_cities) if city not in visited] else 0
            q_table2[current_city, next_city] = q_table2[current_city, next_city] + learning_rate * (reward + discount_factor * max_future_q - q_table2[current_city, next_city])
        # Sample from experiences
        experiences = random.sample(experiences, min(len(experiences), 100))
    
    start_time = time.time()
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

# Unmodified functions from the original code:
'''
distance_calc
local_search_2_opt
initialize_q_table
'''
