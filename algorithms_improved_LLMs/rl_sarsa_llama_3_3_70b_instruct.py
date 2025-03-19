
# Required Libraries
import copy
import numpy as np
import random
import time
############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    """
    Calculate the total distance of a tour.

    Args:
        distance_matrix (np.ndarray): Matrix containing distances between cities.
        city_tour (list): List of cities in the tour.

    Returns:
        float: Total distance of the tour.
    """
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, time_limit=10, start_time=None, best_c=None,  verbose = True):
    """
    Perform local search using the 2-opt algorithm.

    Args:
        distance_matrix (np.ndarray): Matrix containing distances between cities.
        city_tour (list): List of cities in the tour.
        recursive_seeding (int, optional): Number of recursive iterations. Defaults to -1.
        verbose (bool, optional): Whether to print iteration information. Defaults to True.

    Returns:
        tuple: Optimal tour and its distance.
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

############################################################################

# Function: Q-Table Init
def initialize_q_table(num_cities, seed):
    """
    Initialize the Q-table with noise.

    Args:
        num_cities (int): Number of cities in the TSP instance.
        seed (int): Random seed.

    Returns:
        np.ndarray: Q-table with noise.
    """
    if seed is not None:
        np.random.seed(seed) 
    q_table            = np.zeros((num_cities, num_cities))
    num_noisy_elements = int(1 * num_cities * num_cities)
    idx                = np.random.choice(num_cities * num_cities, num_noisy_elements, replace = False)
    noise              = np.random.uniform(-0.01, 0.01, size = num_noisy_elements)
    q_table.flat[idx]  = noise
    return q_table

# Function: SARSA
def sarsa(distance_matrix, learning_rate=0.1, discount_factor=0.95, epsilon=0.15, episodes=5000, q_init=None, time_limit=10, best_c=None, local_search=True, verbose=True):
    """
    Improved SARSA algorithm for solving the TSP.

    This implementation incorporates the following state-of-the-art techniques:
    - Eligibility traces with a decaying rate to improve convergence.
    - Adaptive learning rate and epsilon-greedy exploration to enhance exploration-exploitation trade-off.
    - Local search using the 2-opt algorithm to refine the solution.

    Args:
        distance_matrix (np.ndarray): Matrix containing distances between cities.
        learning_rate (float, optional): Initial learning rate. Defaults to 0.1.
        discount_factor (float, optional): Discount factor for the MDP. Defaults to 0.95.
        epsilon (float, optional): Initial epsilon value for epsilon-greedy exploration. Defaults to 0.15.
        episodes (int, optional): Number of episodes to train the agent. Defaults to 5000.
        q_init (int, optional): Random seed for initializing the Q-table. Defaults to None.
        local_search (bool, optional): Whether to perform local search using 2-opt. Defaults to True.
        verbose (bool, optional): Whether to print episode information. Defaults to True.

    Returns:
        tuple: Optimal tour, its distance, and the Q-table.
    """
    max_dist = np.max(distance_matrix)
    distance_matrix_normalized = distance_matrix / max_dist
    num_cities = distance_matrix_normalized.shape[0]
    lambda_decay = 0.9  # Fixed eligibility trace decay rate
    
    # Improved Q-table initialization with noise
    if q_init is None:
        q_table = initialize_q_table(num_cities, seed=None)
    else:
        q_table = initialize_q_table(num_cities, q_init)
    
    # Improved epsilon-greedy exploration with decay
    epsilon_decay = 0.995
    
    for episode in range(episodes):
        # Adaptive parameter decay
        lr_current = max(learning_rate * (0.999 ** episode), 0.01)
        epsilon_current = max(epsilon * (epsilon_decay ** episode), 0.01)
        
        e_table = np.zeros_like(q_table)
        current_city = random.randint(0, num_cities - 1)
        start_city = current_city
        visited = {current_city}
        unvisited = [city for city in range(num_cities) if city not in visited]
        
        # Epsilon-greedy action selection with improved exploration
        if random.random() < epsilon_current:
            next_city = random.choice(unvisited)
        else:
            next_city = unvisited[np.argmax(q_table[current_city, unvisited])]
        
        while len(visited) < num_cities:
            reward = -distance_matrix_normalized[current_city, next_city]
            visited.add(next_city)
            unvisited = [city for city in range(num_cities) if city not in visited]
            
            # Select next_next_city with improved exploration
            if not unvisited:
                next_next_city = None
            elif random.random() < epsilon_current:
                next_next_city = random.choice(unvisited)
            else:
                next_next_city = unvisited[np.argmax(q_table[next_city, unvisited])]
            
            # Calculate TD error with improved convergence
            q_next = q_table[next_city, next_next_city] if next_next_city is not None else 0.0
            td_error = reward + discount_factor * q_next - q_table[current_city, next_city]
            
            # Update eligibility traces and Q-table with improved decay
            e_table[current_city, next_city] += 1
            q_table += lr_current * td_error * e_table
            e_table *= discount_factor * lambda_decay
            
            current_city, next_city = next_city, next_next_city
        
        # Final Q-update for returning to start city with improved convergence
        if current_city != start_city:
            reward_final = -distance_matrix_normalized[current_city, start_city]
            td_error_final = reward_final - q_table[current_city, start_city]
            q_table[current_city, start_city] += lr_current * td_error_final
        
        if verbose and episode % 100 == 0:
            print(f"Episode {episode}")
    start_time = time.time()
    # Route construction remains unchanged
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
    distance = distance_calc(distance_matrix, [route, 0])  # Recalculate actual distance
    best_c.put(distance)
    time.sleep(0.1)
    if local_search:
        route, distance = local_search_2_opt(distance_matrix, [route, distance], recursive_seeding=-1, time_limit=time_limit, start_time=start_time, best_c=best_c,  verbose=verbose)
        best_c.put(distance)
        time.sleep(0.1)
    return route, distance, q_table

############################################################################

'''
Unmodified functions from the original code:
    - distance_calc
    - local_search_2_opt
    - initialize_q_table
'''