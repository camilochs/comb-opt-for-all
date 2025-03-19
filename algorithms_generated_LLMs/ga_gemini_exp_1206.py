# Required Libraries
import copy
import numpy as np
import random
import os

############################################################################


# Function: Breeding
def breeding(distance_matrix, population, fitness, elite, crossover_rate=0.7):
    """
    Breeds a new generation of individuals based on the current population and their fitness.
    
    This implementation introduces a crossover rate parameter to control the probability of crossover
    between two selected parents. It uses a combination of Elitism and Roulette Wheel Selection to 
    select parents for crossover.
    
    Improvements:
        - Crossover Rate: Introduces a crossover rate to control the exploration-exploitation balance.
        A higher rate encourages more exploration by creating more diverse offspring.
        - Elitism: Ensures that the best individuals are carried over to the next generation, preventing
        the loss of good solutions.
        - Roulette Wheel Selection: Provides a probabilistic selection mechanism based on fitness, allowing
        individuals with higher fitness to have a higher chance of being selected as parents.
        
    Techniques:
        - Crossover Rate: Based on the concept of controlling the rate of genetic operators to improve
        search efficiency.
        - Elitism: A common technique in genetic algorithms to preserve the best solutions found so far.
        - Roulette Wheel Selection: A widely used selection method that mimics the natural selection process.
        
    Args:
        distance_matrix (numpy.ndarray): The distance matrix representing the distances between cities.
        population (list): The current population of individuals.
        fitness (numpy.ndarray): The fitness values of the individuals in the population.
        elite (int): The number of elite individuals to be carried over to the next generation.
        crossover_rate (float): The probability of crossover between two selected parents.
        
    Returns:
        list: A new population of individuals (offspring).
    """
    cost = [item[1] for item in population]
    if elite > 0:
        cost, offspring = (list(t) for t in zip(*sorted(zip(cost, population))))
    else:
        offspring = copy.deepcopy(population)
    
    for i in range(elite, len(offspring)):
        idx1 = roulette_wheel(fitness)
        idx2 = roulette_wheel(fitness)
        while idx1 == idx2:
            idx2 = roulette_wheel(fitness)
        parent_1 = copy.deepcopy(population[idx1])
        parent_2 = copy.deepcopy(population[idx2])
        
        rand = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
        if rand < crossover_rate:
            # Alterna el orden de los padres
            rand_1 = int.from_bytes(os.urandom(8), byteorder='big') / ((1 << 64) - 1)
            if rand_1 > 0.5:
                offspring[i] = crossover_tsp_bcr(distance_matrix, parent_1, parent_2)
            else:
                offspring[i] = crossover_tsp_bcr(distance_matrix, parent_2, parent_1)
        else:
            offspring[i] = crossover_tsp_er(distance_matrix, parent_1, parent_2)
    return offspring

############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m     = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]       
    return distance

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1):
    if (recursive_seeding < 0):
        count = -2
    else:
        count = 0
    city_list = copy.deepcopy(city_tour)
    distance  = city_list[1]*2
    while (count < recursive_seeding):
        best_route = copy.deepcopy(city_list)
        seed      = copy.deepcopy(city_list)        
        for i in range(0, len(city_list[0]) - 2):
            for j in range(i+1, len(city_list[0]) - 1):
                best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))          
                best_route[0][-1]    = best_route[0][0] 
                best_route[1]       = distance_calc(distance_matrix, best_route)                  
                if (city_list[1] > best_route[1]):
                    city_list = copy.deepcopy(best_route)        
                best_route = copy.deepcopy(seed)
        count = count + 1
        if (distance > city_list[1] and recursive_seeding < 0):
            distance         = city_list[1]
            count            = -2
            recursive_seeding = -1
        elif(city_list[1] >= distance and recursive_seeding < 0):
            count            = -1
            recursive_seeding = -2
    return city_list

############################################################################

# Function: Initial Seed
def seed_function(distance_matrix):
    seed      = [[],float("inf")]
    sequence = random.sample(list(range(1, distance_matrix.shape[0]+1)), distance_matrix.shape[0])
    sequence.append(sequence[0])
    seed[0]   = sequence
    seed[1]   = distance_calc(distance_matrix, seed)
    return seed

# Function: Initial Population
def initial_population(population_size, distance_matrix):
    population = []
    for i in range(0, population_size):
        seed = seed_function(distance_matrix)
        population.append(seed)
    return population

############################################################################

# Function: Fitness
def fitness_function(cost, population_size):
    fitness = np.zeros((population_size, 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1 + cost[i] + abs(np.min(cost)))
    fit_sum      = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness):
    ix      = 0
    random = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
            ix = i
            break
    return ix

def crossover_tsp_bcr(distance_matrix, parent_1, parent_2):
    individual = copy.deepcopy(parent_2)
    cut      = random.sample(list(range(0, len(parent_1[0])-1)), int(len(parent_1[0])/2))
    cut      = [ parent_1[0][cut[i]] for i in range(0, len(cut)) if parent_1[0][cut[i]] != parent_2[0][0]]
    d_1      = float('+inf')
    best = []  # Inicializar best con una lista vacía
    for i in range(0, len(cut)):
        
        A        = cut[i]
        temp_parent_2 = copy.deepcopy(parent_2)
        temp_parent_2[0].remove(A)
        dist_list = [distance_calc(distance_matrix, [ temp_parent_2[0][:n] + [A] + temp_parent_2[0][n:], temp_parent_2[1]] ) for n in range(1, len(temp_parent_2[0]))]
        d_2      = min(dist_list)
        if (d_2 <= d_1):
            d_1 = d_2
            n   = dist_list.index(d_1)
            best = temp_parent_2[0][:n] + [A] + temp_parent_2[0][n:]
    if best: # Verificar que best no esté vacía
        if (best[0] == best[-1]):
            parent_2[0] = best
            parent_2[1] = d_1
            individual = copy.deepcopy(parent_2)
    return individual

# Function: TSP Crossover - ER (Edge Recombination)
def crossover_tsp_er(distance_matrix, parent_1, parent_2):
    n = len(parent_1[0]) - 1

    # Construir la tabla de vecinos (edge table)
    edge_table = {i: set() for i in range(1, n+1)}
    def add_edge(city, neighbor):
        if neighbor != city:
            edge_table[city].add(neighbor)
            
    for route in [parent_1[0][:-1], parent_2[0][:-1]]:
        for i in range(n):
            city = route[i]
            left  = route[i-1] if i > 0 else route[-1]
            right = route[i+1] if i < n-1 else route[0]
            add_edge(city, left)
            add_edge(city, right)
    
    # Iniciar la ruta con la primera ciudad del primer padre
    current = parent_1[0][0]
    tour = [current]
    
    while len(tour) < n:
        # Eliminar la ciudad actual de las listas de vecinos de todas las ciudades
        for neighbors in edge_table.values():
            neighbors.discard(current)
        # Si hay vecinos disponibles para la ciudad actual, se elige aquel con menor número de vecinos
        if edge_table[current]:
            next_city = min(edge_table[current], key=lambda city: len(edge_table[city]))
        else:
            # Si no hay vecinos, se elige aleatoriamente de entre las ciudades que aún no se han visitado
            remaining = list(set(range(1, n+1)) - set(tour))
            next_city = random.choice(remaining)
        tour.append(next_city)
        current = next_city

    # Reparar la ruta si faltan ciudades (por cualquier inconsistencia)
    missing = set(range(1, n+1)) - set(tour)
    if missing:
        # Insertar las ciudades faltantes antes de cerrar el ciclo
        tour = tour + list(missing)
    
    # Asegurar que la ruta sea cíclica
    if tour[0] != tour[-1]:
        tour.append(tour[0])
    
    distance = distance_calc(distance_matrix, [tour, None])
    return [tour, distance]

# Function: Mutation - Swap with 2-opt Local Search
def mutation_tsp_swap(distance_matrix, individual, mutation_search):
    k  = random.sample(list(range(1, len(individual[0])-1)), 2)
    k1 = k[0]
    k2 = k[1]  
    A  = individual[0][k1]
    B  = individual[0][k2]
    individual[0][k1] = B
    individual[0][k2] = A
    individual[1]     = distance_calc(distance_matrix, individual)
    individual       = local_search_2_opt(distance_matrix, individual, mutation_search)
    return individual

# Function: Mutation
def mutation(distance_matrix, offspring, mutation_rate, mutation_search, elite):
    for i in range(elite, len(offspring)):
        probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (probability <= mutation_rate):
            offspring[i] = mutation_tsp_swap(distance_matrix, offspring[i], mutation_search)
    return offspring

############################################################################

# Function: GA TSP
def genetic_algorithm_tsp(distance_matrix, population_size = 5, elite = 1, mutation_rate = 0.1, mutation_search = -1, generations = 100, time_limit=10, best=None, verbose = True):
    import time
    start_time = time.time()
    population     = initial_population(population_size, distance_matrix)
    cost         = [item[1] for item in population]
    cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
    elite_ind      = population[0] 
    fitness        = fitness_function(cost, population_size)
    count          = 0
    
    best.put(elite_ind[1])
    
    while (count <= generations): 
        if (verbose == True):
            print('Generation = ', count, 'Distance = ', round(elite_ind[1], 2))
        
        #if not validate_solution(distance_matrix, elite_ind):
        #    raise ValueError("Advertencia: La solución elite no es válida en la generación", count)
        
        offspring      = breeding(distance_matrix, population, fitness, elite)  
        offspring      = mutation(distance_matrix, offspring, mutation_rate, mutation_search, elite)
        cost         = [item[1] for item in offspring]
        cost, population = (list(t) for t in zip(*sorted(zip(cost, offspring ))))
        elite_child    = population[0]
        fitness        = fitness_function(cost, population_size)
        if(elite_ind[1] > elite_child[1]):
            elite_ind = elite_child 
            best.put(elite_ind[1])
            time.sleep(0.1)
        count = count + 1  
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break
    route, distance = elite_ind
    return route, distance



############################################################################

# Unmodified functions:
# - distance_calc
# - local_search_2_opt
# - seed_function
# - initial_population
# - roulette_wheel
# - crossover_tsp_bcr
# - crossover_tsp_er
# - mutation_tsp_swap


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
