
###############################################################################
import copy
import numpy as np
import random
import time


# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, time_limit=10, start_time=None, best_c=None, verbose = True):
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
# Function: SARSA with Expected SARSA and Boltzmann Exploration
def sarsa(distance_matrix, learning_rate=0.1, discount_factor=0.95, epsilon=0.15, episodes=5000, q_init=None, time_limit=10, best_c=None, local_search=True, verbose=True):
    """
    Implements Expected SARSA algorithm with Boltzmann exploration for the Traveling Salesman Problem (TSP).

    Improvements Implemented:
    - **Expected SARSA:** Instead of updating Q-values using the next action's Q-value, Expected SARSA uses the expected value over all possible next actions, weighted by their probabilities under the current policy.
    - **Boltzmann Exploration (Softmax):** Replaces epsilon-greedy action selection with Boltzmann exploration, where the probability of selecting an action is proportional to the exponential of its Q-value, scaled by a temperature parameter.

    How It Enhances Performance:
    - **Expected SARSA:** Provides a smoother update by considering all possible next actions, leading to more stable learning and better convergence properties compared to standard SARSA.
    - **Boltzmann Exploration:** Promotes exploration of actions proportionally to their estimated value, balancing exploration and exploitation more effectively than epsilon-greedy, which can improve solution quality and convergence speed.

    Based on State-of-the-Art Techniques:
    - Expected SARSA is a widely used algorithm in reinforcement learning that often outperforms standard SARSA by reducing variance in the update steps.
    - Boltzmann exploration is a common technique to improve the action selection mechanism, especially in problems where the action space is large or the estimates of the action values are uncertain.

    Parameters:
    - distance_matrix (numpy.ndarray): The distance matrix representing the TSP.
    - learning_rate (float): The learning rate (alpha) for Q-value updates.
    - discount_factor (float): The discount factor (gamma) for future rewards.
    - epsilon (float): Initial value for epsilon in epsilon-greedy (not used in Boltzmann exploration).
    - episodes (int): Number of episodes to run the algorithm.
    - q_init (int or None): Seed for Q-table initialization; if None, initializes Q-table with zeros.
    - local_search (bool): Whether to perform 2-opt local search on the final tour.
    - verbose (bool): Whether to print progress messages during training.

    Returns:
    - route (list): The best route found.
    - distance (float): The total distance of the best route.
    - q_table (numpy.ndarray): The learned Q-table.
    """
    max_dist = np.max(distance_matrix)
    distance_matrix_normalized = distance_matrix / max_dist
    num_cities = distance_matrix_normalized.shape[0]

    temperature = 1.0  # Initial temperature for Boltzmann exploration
    min_temperature = 0.01  # Minimum temperature
    temperature_decay = (temperature - min_temperature) / episodes

    if q_init is None:
        q_table = np.zeros((num_cities, num_cities))
    else:
        q_table = initialize_q_table(num_cities, q_init)

    for episode in range(episodes):
        current_city = random.randint(0, num_cities - 1)
        start_city = current_city
        visited = {current_city}
        route = [current_city]
        e_table = np.zeros_like(q_table)
        
        while len(visited) < num_cities:
            unvisited = [city for city in range(num_cities) if city not in visited]

            # Boltzmann exploration
            q_values = q_table[current_city, unvisited]
            exp_q = np.exp(q_values / temperature)
            probabilities = exp_q / np.sum(exp_q)
            next_city = np.random.choice(unvisited, p=probabilities)

            reward = -distance_matrix_normalized[current_city, next_city]
            visited.add(next_city)
            route.append(next_city)

            # Expected SARSA Update
            if len(visited) < num_cities:
                next_unvisited = [city for city in range(num_cities) if city not in visited]
                q_next_values = q_table[next_city, next_unvisited]
                exp_q_next = np.exp(q_next_values / temperature)
                probabilities_next = exp_q_next / np.sum(exp_q_next)
                expected_q_next = np.dot(probabilities_next, q_next_values)
            else:
                # If at the last city, expect to return to start city
                expected_q_next = q_table[next_city, start_city]

            td_target = reward + discount_factor * expected_q_next
            td_error = td_target - q_table[current_city, next_city]

            # Update eligibility traces and Q-table
            e_table[current_city, next_city] += 1
            q_table += learning_rate * td_error * e_table
            e_table *= discount_factor

            current_city = next_city

        # Final step: return to start city
        reward = -distance_matrix_normalized[current_city, start_city]
        td_error = reward - q_table[current_city, start_city]
        q_table[current_city, start_city] += learning_rate * td_error

        route.append(start_city)

        # Decay temperature
        temperature = max(temperature - temperature_decay, min_temperature)

        if verbose and episode % 100 == 0:
            print(f"Episode {episode}, Temperature: {temperature:.4f}")

    start_time = time.time()
    # Construct the best route based on learned Q-table
    start_city = 0
    current_city = start_city
    visited = {current_city}
    route = [current_city]
    while len(visited) < num_cities:
        unvisited = [city for city in range(num_cities) if city not in visited]
        q_values = q_table[current_city, unvisited]
        next_city = unvisited[np.argmax(q_values)]
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

###############################################################################

# Unmodified functions from the original code:
# - distance_calc
# - local_search_2_opt
# - initialize_q_table

###############################################################################
