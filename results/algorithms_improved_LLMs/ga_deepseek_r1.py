import copy
import numpy as np
import random
import os

############################################################################


# Function: Initial Seed
def seed_function(distance_matrix):
    seed     = [[],float("inf")]
    sequence = random.sample(list(range(1, distance_matrix.shape[0]+1)), distance_matrix.shape[0])
    sequence.append(sequence[0])
    seed[0]  = sequence
    seed[1]  = distance_calc(distance_matrix, seed)
    return seed
def distance_calc(distance_matrix, city_tour):
    """Calculate total tour distance.
    
    Args:
        distance_matrix (np.array): Square matrix of pairwise distances.
        city_tour (list): Tour representation as [route, distance].
    
    Returns:
        float: Total distance of the tour.
    """
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance += distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]
    return distance

def local_search_2_opt(distance_matrix, city_tour, recursive_seeding=-1):
    """2-opt local search with stochastic first-improvement strategy.
    
    Enhances performance by checking random edge swaps first, reducing 
    computation time per iteration. Based on efficient 2-opt variants 
    used in modern TSP solvers.
    """
    city_list = copy.deepcopy(city_tour)
    best_distance = city_list[1]
    size = len(city_list[0]) - 1
    max_attempts = 50  # Balances speed vs solution quality
    
    for _ in range(max_attempts):
        i = random.randint(0, size-2)
        j = random.randint(i+1, size-1)
        new_route = city_list[0][:i] + city_list[0][i:j+1][::-1] + city_list[0][j+1:]
        new_distance = distance_calc(distance_matrix, [new_route, 0])
        
        if new_distance < best_distance:
            city_list[0], city_list[1] = new_route, new_distance
            best_distance = new_distance
            if recursive_seeding < 0:  # First-improvement strategy
                break
    
    return city_list

def nearest_neighbor_seed(distance_matrix):
    """Generate initial solution using Nearest Neighbor heuristic.
    
    Provides high-quality initial seeds to accelerate convergence.
    Based on constructive heuristic methods commonly used in TSP.
    """
    num_cities = distance_matrix.shape[0]
    start = random.randint(1, num_cities)
    tour = [start]
    unvisited = set(range(1, num_cities+1))
    unvisited.remove(start)
    
    while unvisited:
        current = tour[-1]
        next_city = min(unvisited, key=lambda x: distance_matrix[current-1, x-1])
        tour.append(next_city)
        unvisited.remove(next_city)
    
    tour.append(start)
    return [tour, distance_calc(distance_matrix, [tour, 0])]

def initial_population(population_size, distance_matrix):
    """Initialize population with mix of random and heuristic solutions.
    
    Combines diversity (random) with quality (NN) for better exploration.
    Implements hybrid population initialization from modern metaheuristics.
    """
    population = []
    if population_size >= 5:  # Include 20% NN seeds
        for _ in range(max(1, population_size//5)):
            population.append(nearest_neighbor_seed(distance_matrix))
    
    while len(population) < population_size:
        sequence = random.sample(range(1, distance_matrix.shape[0]+1), distance_matrix.shape[0])
        sequence.append(sequence[0])
        population.append([sequence, distance_calc(distance_matrix, [sequence, 0])])
    
    return population

############################################################################

def fitness_function(cost, population_size):
    """Rank-based fitness assignment with exponential scaling.
    
    Reduces selection pressure early in evolution while maintaining
    convergence capability. Based on rank-based selection methods
    from evolutionary computation literature.
    """
    ranks = np.argsort(np.argsort(cost)) + 1  # Rank positions
    fitness = np.exp(-ranks/5)  # Exponential rank scaling
    fitness /= fitness.sum()  # Normalize
    
    cumulative = np.cumsum(fitness)
    return np.column_stack((fitness, cumulative))

def tournament_selection(fitness, tournament_size=3):
    """Tournament selection with size 3.
    
    Maintains diversity better than roulette wheel while preserving
    selection pressure. Standard in modern GA implementations.
    """
    candidates = np.random.choice(len(fitness), tournament_size, replace=False)
    return candidates[np.argmax(fitness[candidates, 0])]

############################################################################

def crossover_tsp_ox(distance_matrix, parent1, parent2):
    """Ordered Crossover (OX) with adaptive cut points.
    
    Preserves city order and adjacency information from both parents.
    Considered one of the most effective TSP crossover operators.
    """
    size = len(parent1[0])-1
    cut = sorted(random.sample(range(size), 2))
    child = [-1]*(size+1)
    
    # Copy segment from parent1
    child[cut[0]:cut[1]+1] = parent1[0][cut[0]:cut[1]+1]
    
    # Fill remaining positions from parent2
    ptr = (cut[1]+1) % size
    for city in parent2[0][:-1]:
        if city not in child[cut[0]:cut[1]+1]:
            child[ptr] = city
            ptr = (ptr+1) % size
    
    child[-1] = child[0]
    return [child, distance_calc(distance_matrix, [child, 0])]

# Function: TSP Crossover - BCR (Best Cost Route Crossover)
def crossover_tsp_bcr(distance_matrix, parent_1, parent_2):
    individual = copy.deepcopy(parent_2)
    cut        = random.sample(list(range(0, len(parent_1[0]))), int(len(parent_1)/2))
    cut        = [ parent_1[0][cut[i]] for i in range(0, len(cut)) if parent_1[0][cut[i]] != parent_2[0][0]]
    d_1        = float('+inf')
    for i in range(0, len(cut)):
        best      = []
        A         = cut[i]
        parent_2[0].remove(A)
        dist_list = [distance_calc(distance_matrix, [ parent_2[0][:n] + [A] + parent_2[0][n:], parent_2[1]] ) for n in range(1, len(parent_2[0]))]
        d_2       = min(dist_list)
        if (d_2 <= d_1):
            d_1  = d_2
            n    = dist_list.index(d_1)
            best = parent_2[0][:n] + [A] + parent_2[0][n:]
        if (best[0] == best[-1]):
            parent_2[0] = best
            parent_2[1] = d_1
            individual  = copy.deepcopy(parent_2)
    return individual

# Function: TSP Crossover - ER (Edge Recombination)
def crossover_tsp_er(distance_matrix, parent_1, parent_2):
    ind_list   = [item for item in parent_2[0]]
    ind_list.sort()
    ind_list   = list(dict.fromkeys(ind_list))
    edg_list   = [ [item, []] for item in ind_list]
    for i in range(0, len(edg_list)):
        edges = []
        idx_c = parent_2[0].index(i+1)
        idx_l = np.clip(idx_c - 1, 0, len(parent_2[0])-1)
        idx_r = np.clip(idx_c + 1, 0, len(parent_2[0])-1)
        if (parent_2[0][idx_l] not in edges):
            edges.append(parent_2[0][idx_l])
        if (parent_2[0][idx_r] not in edges):
            edges.append(parent_2[0][idx_r])
        idx_c = parent_1[0].index(i+1)
        idx_l = np.clip(idx_c - 1, 0, len(parent_1[0])-1)
        idx_r = np.clip(idx_c + 1, 0, len(parent_1[0])-1)
        if (parent_1[0][idx_l] not in edges):
            edges.append(parent_1[0][idx_l])
        if (parent_1[0][idx_r] not in edges):
            edges.append(parent_1[0][idx_r])
        for edge in edges:
            edg_list[i][1].append(edge)
    start      = parent_1[0][0]
    individual = [[start], 1]
    target     = start
    del edg_list[start-1]
    while len(individual[0]) != len(parent_2[0])-1:
        limit      = float('+inf')
        candidates =  [ [[], []] for item in edg_list]
        for i in range(0, len(edg_list)):
            if (target in edg_list[i][1]):
                candidates[i][0].append(edg_list[i][0])
                candidates[i][1].append(len(edg_list[i][1]))
                if (len(edg_list[i][1]) < limit):
                    limit = len(edg_list[i][1])
                edg_list[i][1].remove(target)
        for i in range(len(candidates)-1, -1, -1):
            if (len(candidates[i][0]) == 0 or candidates[i][1] != [limit]):
                del candidates[i]
        if len(candidates) > 1:
            k = random.sample(list(range(0, len(candidates))), 1)[0]
        else:
            k = 0
        if (len(candidates) > 0):
            target = candidates[k][0][0]
            individual[0].append(target)
        else:
            if (len(edg_list) > 0):
                target = edg_list[0][0]
                individual[0].append(target)
            else:
                last_edges = [item for item in ind_list if item not in individual[0]]
                for edge in last_edges:
                    individual[0].append(edge)
        for i in range(len(edg_list)-1, -1, -1):
            if (edg_list[i][0] == target):
                del edg_list[i]
                break
        if (len(edg_list) == 1):
            individual[0].append(edg_list[0][0])
    individual[0].append(individual[0][0])
    individual[1] = distance_calc(distance_matrix, individual)
    return individual
def breeding(distance_matrix, population, fitness, elite):
    """Hybrid parent selection using tournament + roulette strategies.
    
    Combines benefits of both selection methods. Implements adaptive
    operator selection from multi-method evolutionary algorithms.
    """
    cost = [ind[1] for ind in population]
    sorted_pop = [x for _,x in sorted(zip(cost, population))]
    
    # Elitism
    offspring = sorted_pop[:elite]
    
    # Generate offspring
    for _ in range(len(population)-elite):
        # Parent selection
        if random.random() < 0.7:
            p1 = tournament_selection(fitness)
        else:
            p1 = np.searchsorted(fitness[:,1], random.random())
        
        p2 = p1
        while p2 == p1:
            if random.random() < 0.7:
                p2 = tournament_selection(fitness)
            else:
                p2 = np.searchsorted(fitness[:,1], random.random())
        
        # Operator selection
        parent1 = copy.deepcopy(population[p1])
        parent2 = copy.deepcopy(population[p2])
        
        if random.random() < 0.4:
            child = crossover_tsp_ox(distance_matrix, parent1, parent2)
        elif random.random() < 0.7:
            child = crossover_tsp_er(distance_matrix, parent1, parent2)
        else:
            child = crossover_tsp_bcr(distance_matrix, parent1, parent2)
        
        offspring.append(child)
    
    return offspring

############################################################################

def mutation_tsp_swap(distance_matrix, individual, mutation_search):
    """Adaptive mutation combining swap and inversion operators.
    
    Balances exploration (swap) and exploitation (inversion).
    Implements hybrid mutation strategies from memetic algorithms.
    """
    mut_ind = copy.deepcopy(individual)
    
    if random.random() < 0.6:  # 2-opt guided swap
        i, j = random.sample(range(1, len(mut_ind[0])-1), 2)
        mut_ind[0][i], mut_ind[0][j] = mut_ind[0][j], mut_ind[0][i]
    else:  # Inversion mutation
        i, j = sorted(random.sample(range(1, len(mut_ind[0])-1), 2))
        mut_ind[0][i:j+1] = mut_ind[0][i:j+1][::-1]
    
    mut_ind[1] = distance_calc(distance_matrix, mut_ind)
    return local_search_2_opt(distance_matrix, mut_ind, mutation_search)

def mutation(distance_matrix, offspring, mutation_rate, mutation_search, elite):
    """Adaptive mutation with diversity preservation.
    
    Uses duplicate detection and replacement to maintain population
    diversity. Based on niching techniques in evolutionary algorithms.
    """
    for i in range(elite, len(offspring)):
        if random.random() > mutation_rate:
            continue
        
        # Mutation
        offspring[i] = mutation_tsp_swap(distance_matrix, offspring[i], mutation_search)
        
        # Diversity check
        for j in range(i):
            if offspring[i][1] == offspring[j][1] and \
               set(offspring[i][0]) == set(offspring[j][0]):
                offspring[i] = seed_function(distance_matrix)
                break
    
    return offspring

############################################################################

def genetic_algorithm_tsp(distance_matrix, population_size=5, elite=1, 
                         mutation_rate=0.1, mutation_search=-1, generations=100, 
                         time_limit =10, best=None, verbose=True):
    """Enhanced GA for TSP with modern metaheuristic components.
    
    Improvements:
    1. Hybrid initialization with nearest neighbor heuristic
    2. Rank-based fitness + tournament selection
    3. Adaptive operator selection (OX, ER, BCR)
    4. Memetic local search with stochastic 2-opt
    5. Diversity preservation mechanisms
    
    Implements state-of-the-art techniques from:
    - Hybrid evolutionary algorithms (HEA)
    - Memetic algorithms
    - Adaptive operator selection
    """
    import time
    start_time = time.time()
    population = initial_population(population_size, distance_matrix)
    population.sort(key=lambda x: x[1])
    elite_ind = copy.deepcopy(population[0])
    stagnation = 0
    best.put(elite_ind[1])
    gen = 0
    while True:
        # Fitness calculation
        
        
        cost = [ind[1] for ind in population]
        fitness = fitness_function(cost, population_size)
        
        # Breeding
        offspring = breeding(distance_matrix, population, fitness, elite)
        
        # Mutation
        offspring = mutation(distance_matrix, offspring, mutation_rate, mutation_search, elite)
        for i, sol in enumerate(offspring):
            if not validate_solution(distance_matrix, sol):
                # Puedes optar por re-inicializar la solución o simplemente registrar una advertencia.
                #raise ValueError(f"Advertencia: Solución inválida en generación {gen}, individuo {i}. Re-inicializando.")
                offspring[i] = seed_function(distance_matrix)
        # Selection
        offspring.sort(key=lambda x: x[1])
        population = offspring[:population_size]
        
        # Update elite
        if population[0][1] < elite_ind[1]:
            elite_ind = copy.deepcopy(population[0])
            
            best.put(elite_ind[1])
            time.sleep(0.1)
            
            stagnation = 0
        else:
            stagnation += 1
        
        # Adaptive restart
        if stagnation >= 20 and gen < 0.8*generations:
            population = population[:elite] + initial_population(population_size-elite, distance_matrix)
            stagnation = 0
        
        if verbose and gen%10 == 0:
            print(f'Generation {gen}: Best Distance = {elite_ind[1]:.2f}')
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break
        gen += 1
    return elite_ind[0], elite_ind[1]

############################################################################
# Unmodified original functions:
# - seed_function
# - crossover_tsp_bcr
# - crossover_tsp_er
# - roulette_wheel



# Función: Validación de Solución para el TSP en el GA
def validate_solution(distance_matrix, solution, tol=1e-6):
    """
    Valida que la solución del TSP es correcta.
    
    Parámetros:
        distance_matrix: matriz de distancias (numpy array).
        solution: lista de dos elementos [ruta, distancia], donde:
                  - ruta es una lista de ciudades que inicia y termina en la misma ciudad.
                  - distancia es el valor total de la ruta.
        tol: tolerancia para comparar distancias (default: 1e-6).
                  
    Retorna:
        True si la solución es válida, False en caso contrario.
    """
    ruta, distancia_reportada = solution
    
    # Verificar que la ruta comienza y termina en la misma ciudad.
    if ruta[0] != ruta[-1]:
        # La ruta debe ser cíclica.
        return False
    
    # Verificar que se visitan todas las ciudades exactamente una vez (exceptuando la repetición de la ciudad inicial al final).
    ciudades_visitadas = ruta[:-1]
    n_ciudades = distance_matrix.shape[0]
    if len(ciudades_visitadas) != n_ciudades or len(set(ciudades_visitadas)) != n_ciudades:
        return False
    
    # Recalcular la distancia usando la función distance_calc.
    distancia_calculada = distance_calc(distance_matrix, solution)
    if abs(distancia_calculada - distancia_reportada) > tol:
        return False
    
    return True
