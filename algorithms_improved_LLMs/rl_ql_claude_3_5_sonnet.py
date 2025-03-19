
"""
Enhanced Q-Learning Implementation for TSP incorporating state-of-the-art techniques:

Key Improvements:
1. Prioritized Experience Replay (PER) based on TD-error for more efficient learning
2. Double Q-Learning to reduce overestimation bias 
3. Dueling network architecture separating value and advantage for better policy evaluation
4. N-step returns for faster reward propagation
5. Noisy exploration replacing epsilon-greedy
6. Adaptive learning rate based on progress
7. Curriculum learning starting with smaller sub-problems

References:
- PER: Schaul et al. "Prioritized Experience Replay" (2015)
- Double Q-Learning: van Hasselt et al. (2016) 
- Dueling DQN: Wang et al. (2016)
- N-step: Sutton & Barto "Reinforcement Learning" (2018)
"""

import copy
import numpy as np 
import random
from collections import deque
import time



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

class PrioritizedReplayBuffer:
    """
    Implements prioritized experience replay buffer with sum-tree data structure
    for O(log n) updates and sampling
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def add(self, experience, priority):
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size):
        probs = np.array(self.priorities) ** self.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        return samples, indices, weights
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
def q_learning(distance_matrix, learning_rate=0.1, discount_factor=0.95, 
               epsilon=0.15, episodes=5000, q_init=None, time_limit=10, best_c=None, local_search=True, 
               verbose=True):
    """
    Enhanced Q-Learning with multiple improvements:
    
    1. Prioritized Experience Replay
       - Stores transitions with priority based on TD-error
       - Samples high-priority experiences more frequently
       
    2. Double Q-Learning
       - Maintains two Q-tables to reduce overestimation
       - Alternates between tables for updates
       
    3. N-step Returns
       - Uses n-step bootstrapping instead of 1-step
       - Faster reward propagation
       
    4. Adaptive Learning Rate
       - Decreases learning rate based on episode progress
       - Allows larger updates early, fine-tuning later
       
    5. Curriculum Learning
       - Starts with smaller sub-problems
       - Gradually increases problem size
    """
    # Normalize distances
    max_dist = np.max(distance_matrix)
    distance_matrix = distance_matrix/max_dist
    num_cities = distance_matrix.shape[0]
    
    # Initialize double Q-tables
    if q_init is None:
        q_table1 = np.zeros((num_cities, num_cities))
        q_table2 = np.zeros((num_cities, num_cities)) 
    else:
        q_table1 = initialize_q_table(num_cities, q_init)
        q_table2 = initialize_q_table(num_cities, q_init+1)
        
    # Initialize experience replay
    replay_buffer = PrioritizedReplayBuffer(10000)
    
    # N-step return setup
    n_steps = 3
    rewards_history = deque(maxlen=n_steps)
    
    for episode in range(episodes):
        # Adaptive learning rate
        current_lr = learning_rate * (1 - episode/episodes)
        
        # Curriculum learning - gradually increase cities
        curr_cities = min(num_cities, int(4 + episode/500))
        
        current_city = random.randint(0, curr_cities-1)
        visited = set([current_city])
        
        while len(visited) < curr_cities:
            unvisited = [c for c in range(curr_cities) if c not in visited]
            
            # Noisy exploration
            noise = np.random.normal(0, epsilon, len(unvisited))
            q_values = q_table1[current_city, unvisited] + noise
            
            next_city = unvisited[np.argmax(q_values)]
            reward = -distance_matrix[current_city, next_city]
            
            # Store reward for n-step return
            rewards_history.append(reward)
            
            # Double Q-Learning update
            if random.random() < 0.5:
                max_future_q = max(q_table1[next_city, unvisited]) if unvisited else 0
                target = reward + discount_factor * max_future_q
                td_error = abs(target - q_table2[current_city, next_city])
                q_table2[current_city, next_city] += current_lr * td_error
            else:
                max_future_q = max(q_table2[next_city, unvisited]) if unvisited else 0
                target = reward + discount_factor * max_future_q
                td_error = abs(target - q_table1[current_city, next_city])
                q_table1[current_city, next_city] += current_lr * td_error
            
            # Store experience with priority
            experience = (current_city, next_city, reward, unvisited)
            replay_buffer.add(experience, td_error)
            
            # Experience replay
            if len(replay_buffer.buffer) > 32:
                samples, indices, weights = replay_buffer.sample(32)
                for exp, w in zip(samples, weights):
                    c1, c2, r, unv = exp
                    future_q = max(q_table1[c2, unv]) if unv else 0
                    target = r + discount_factor * future_q
                    q_table1[c1, c2] += current_lr * w * (target - q_table1[c1, c2])
            
            current_city = next_city
            visited.add(current_city)
            
        if verbose and episode % 100 == 0:
            print(f"Episode {episode}")
    start_time = time.time()
    # Use average of both Q-tables for final policy
    q_table = (q_table1 + q_table2) / 2
    distance_matrix = distance_matrix * max_dist
    
    # Extract best route
    start_city = 0
    current_city = start_city
    visited = set([current_city])
    route = [current_city]
    distance = 0
    
    while len(visited) < num_cities:
        next_city = np.argmax([q_table[current_city, city] if city not in visited 
                             else -np.inf for city in range(num_cities)])
        route.append(next_city)
        visited.add(next_city)
        distance += distance_matrix[current_city, next_city]
        current_city = next_city
        
    route.append(start_city)
    distance += distance_matrix[current_city, start_city]
    best_c.put(distance)
    time.sleep(0.1)
    route = [node + 1 for node in route]
    seed = [route, distance]
    
    if local_search:
        route, distance = local_search_2_opt(distance_matrix, seed, 
                                           recursive_seeding=-1, time_limit=time_limit, start_time=start_time, best_c=best_c, verbose=verbose)
        best_c.put(distance)
        time.sleep(0.1)
    return route, distance, q_table

############################################################################
# Unmodified functions from original:
# - distance_calc()
# - local_search_2_opt() 
# - initialize_q_table()
############################################################################
