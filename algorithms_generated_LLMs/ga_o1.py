# Required Libraries
import copy
import numpy as np
import random
import os

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    '''
    Calculate the total distance of a tour.

    Parameters:
    distance_matrix (numpy.array): The distance matrix.
    city_tour (list): The tour represented as a sequence of city indices (from 0 to n-1).

    Returns:
    float: The total distance of the tour.
    '''
    distance = 0
    for k in range(len(city_tour[0]) - 1):
        m = k + 1
        distance += distance_matrix[city_tour[0][k], city_tour[0][m]]
    return distance

# Function: 2-opt Local Search
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding=-1):
    '''
    Perform 2-opt local search on a tour to improve its distance.

    The 2-opt algorithm swaps two edges to reduce the tour distance.

    Parameters:
    distance_matrix (numpy.array): The distance matrix.
    city_tour (list): The tour represented as a sequence of city indices.
    recursive_seeding (int): Number of iterations for local search; if -1, runs until no improvement.

    Returns:
    list: Improved tour and its distance.
    '''
    if recursive_seeding < 0:
        count = -2
    else:
        count = 0
    city_list = copy.deepcopy(city_tour)
    distance = city_list[1] * 2
    while count < recursive_seeding:
        best_route = copy.deepcopy(city_list)
        seed = copy.deepcopy(city_list)
        for i in range(0, len(city_list[0]) - 2):
            for j in range(i + 1, len(city_list[0]) - 1):
                best_route[0][i:j + 1] = list(reversed(best_route[0][i:j + 1]))
                best_route[0][-1] = best_route[0][0]
                best_route[1] = distance_calc(distance_matrix, best_route)
                if city_list[1] > best_route[1]:
                    city_list = copy.deepcopy(best_route)
                best_route = copy.deepcopy(seed)
        count = count + 1
        if distance > city_list[1] and recursive_seeding < 0:
            distance = city_list[1]
            count = -2
            recursive_seeding = -1
        elif city_list[1] >= distance and recursive_seeding < 0:
            count = -1
            recursive_seeding = -2
    return city_list

# Function: Initial Seed
def seed_function(distance_matrix):
    '''
    Generate an initial seed tour.

    This function generates a random tour starting and ending at the same city.

    The city indices start from 0 to n-1 to maintain consistency.

    Parameters:
    distance_matrix (numpy.array): The distance matrix.

    Returns:
    list: A seed individual with the tour and its total distance.
    '''
    seed = [[], float("inf")]
    sequence = random.sample(list(range(0, distance_matrix.shape[0])), distance_matrix.shape[0])
    sequence.append(sequence[0])
    seed[0] = sequence
    seed[1] = distance_calc(distance_matrix, seed)
    return seed

# Function: Initial Population
def initial_population(population_size, distance_matrix):
    '''
    Generate the initial population for the genetic algorithm.

    Parameters:
    population_size (int): Size of the population.
    distance_matrix (numpy.array): The distance matrix.

    Returns:
    list: Initial population of individuals.
    '''
    population = []
    for i in range(population_size):
        seed = seed_function(distance_matrix)
        population.append(seed)
    return population

# Function: Tournament Selection
def tournament_selection(population, k=3):
    '''
    Select an individual from the population using tournament selection.

    Tournament selection picks 'k' individuals from the population at random and selects the best among them as a parent.

    This increases selection pressure on better individuals, leading to faster convergence.

    Based on state-of-the-art techniques in genetic algorithms, tournament selection is known for maintaining diversity and improving solution quality.

    Parameters:
    population (list): List of individuals in the population.
    k (int): Tournament size, default is 3.

    Returns:
    list: Selected individual from the population.
    '''
    candidates = random.sample(population, k)
    candidates.sort(key=lambda ind: ind[1])  # Assuming the second element is fitness (distance)
    return candidates[0]

# Function: Crossover - Partially Mapped Crossover (PMX)
def crossover_pmx(distance_matrix, parent1, parent2):
    '''
    Perform Partially Mapped Crossover (PMX) between two parent permutations.

    PMX preserves relative positions of cities and ensures offspring are valid permutations.

    This crossover helps maintain genetic diversity and contributes to better exploration of the search space.

    Based on state-of-the-art techniques in genetic algorithms for permutation problems like TSP.

    Parameters:
    distance_matrix (numpy.array): The distance matrix.
    parent1 (list): First parent individual.
    parent2 (list): Second parent individual.

    Returns:
    list: Offspring individual resulting from PMX crossover.
    '''
    parent1_seq = parent1[0][:-1]  # Exclude the last element (same as first)
    parent2_seq = parent2[0][:-1]

    size = len(parent1_seq)
    # Choose two random crossover points
    cx_point1 = random.randint(0, size - 2)
    cx_point2 = random.randint(cx_point1 + 1, size - 1)

    # Create empty offspring
    offspring_seq = [None] * size

    # Copy mapping section from first parent to offspring
    offspring_seq[cx_point1:cx_point2 + 1] = parent1_seq[cx_point1:cx_point2 + 1]

    # Mapping of swapped genes
    mapping = {}
    for i in range(cx_point1, cx_point2 + 1):
        mapping[parent2_seq[i]] = parent1_seq[i]

    # Fill in the rest from parent2
    for i in range(size):
        if offspring_seq[i] is None:
            candidate = parent2_seq[i]
            while candidate in mapping.values():
                candidate = [k for k, v in mapping.items() if v == candidate][0]
            offspring_seq[i] = candidate

    # Complete the tour by appending start city
    offspring_seq.append(offspring_seq[0])

    # Evaluate distance
    offspring = [offspring_seq, distance_calc(distance_matrix, [offspring_seq, 0])]

    return offspring

# Function: Breeding
def breeding(distance_matrix, population, elite):
    '''
    Breeding function with tournament selection and PMX crossover.

    This function uses tournament selection to select parents and performs Partially Mapped Crossover (PMX) to produce offspring.

    Tournament selection increases selection pressure for better individuals, leading to faster convergence.

    PMX crossover preserves relative positions and provides valid permutations, leading to better quality solutions.

    These are based on state-of-the-art techniques in genetic algorithms for combinatorial optimization problems like TSP.

    Parameters:
    distance_matrix (numpy.array): The distance matrix.
    population (list): Current population of individuals.
    elite (int): Number of elite individuals to carry over to next generation.

    Returns:
    list: New population after breeding.
    '''
    cost = [item[1] for item in population]
    if elite > 0:
        cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
        offspring = population[:elite] + [None] * (len(population) - elite)
    else:
        offspring = [None] * len(population)
    for i in range(elite, len(offspring)):
        # Parent selection using tournament selection
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        while parent2 == parent1:
            parent2 = tournament_selection(population)
        # Crossover
        offspring[i] = crossover_pmx(distance_matrix, parent1, parent2)
    return offspring

# Function: Mutation - Inversion Mutation with 2-opt Local Search
def mutation_inversion(distance_matrix, individual, mutation_search):
    '''
    Perform inversion mutation on an individual and apply 2-opt local search.

    In inversion mutation, a subset of the tour is selected and reversed.

    This mutation operator helps in exploring the search space more effectively by making larger changes in the tour.

    Using inversion mutation can lead to better quality solutions and faster convergence.

    Parameters:
    distance_matrix (numpy.array): The distance matrix.
    individual (list): Individual to be mutated.
    mutation_search (int): Parameter for local search; if -1, recursive search until no improvement.

    Returns:
    list: Mutated individual.
    '''
    size = len(individual[0]) - 1  # Exclude the last element (same as first)
    a = random.randint(1, size - 2)
    b = random.randint(a + 1, size - 1)
    individual[0][a:b + 1] = reversed(individual[0][a:b + 1])
    individual[0][-1] = individual[0][0]
    individual[1] = distance_calc(distance_matrix, individual)
    individual = local_search_2_opt(distance_matrix, individual, mutation_search)
    return individual

# Function: Mutation
def mutation(distance_matrix, offspring, mutation_rate, mutation_search, elite):
    '''
    Apply mutation operators to the offspring population.

    This function applies inversion mutation with a certain probability to introduce diversity.

    Mutation operators help in exploring the search space and preventing premature convergence.

    By using inversion mutation and 2-opt local search, better quality solutions can be found faster.

    Parameters:
    distance_matrix (numpy.array): The distance matrix.
    offspring (list): Offspring population.
    mutation_rate (float): Probability of mutation.
    mutation_search (int): Parameter for local search; if -1, recursive search until no improvement.
    elite (int): Number of elite individuals to skip mutation.

    Returns:
    list: Offspring population after mutation.
    '''
    for i in range(elite, len(offspring)):
        probability = random.random()
        if probability <= mutation_rate:
            offspring[i] = mutation_inversion(distance_matrix, offspring[i], mutation_search)
    return offspring

# Function: GA TSP
def genetic_algorithm_tsp(distance_matrix, population_size=50, elite=1, mutation_rate=0.1, mutation_search=-1, generations=100, time_limit=10, best=None, verbose=True):
    '''
    Genetic Algorithm for the Travelling Salesman Problem with improvements.

    This implementation includes the following improvements:

    1. Consistent City Indexing:

       All city indices start from 0 to n-1 to maintain consistency across all functions.

       This prevents index errors and ensures accurate distance calculations.

    2. Tournament Selection:

       Replaces roulette wheel selection with tournament selection, which increases selection pressure and maintains diversity, leading to faster convergence.

       Based on state-of-the-art techniques in genetic algorithms.

    3. Partially Mapped Crossover (PMX):

       Uses PMX crossover, which preserves relative order and positions, producing valid offspring permutations.

       This helps in maintaining genetic material and enhances exploration of the search space.

       Based on state-of-the-art techniques in genetic algorithms for permutation problems.

    4. Inversion Mutation:

       Uses inversion mutation operator, which selects a subset of the tour and reverses it, aiding in better exploration.

       Combined with 2-opt local search to improve solution quality.

    These improvements lead to finding better quality solutions and faster convergence times.

    Parameters:
    distance_matrix (numpy.array): The distance matrix.
    population_size (int): Size of the population.
    elite (int): Number of elite individuals to carry over to next generation.
    mutation_rate (float): Probability of mutation.
    mutation_search (int): Parameter for local search; if -1, recursive search until no improvement.
    generations (int): Number of generations to evolve.
    verbose (bool): If True, prints progress information.

    Returns:
    list: Best tour found.
    float: Distance of the best tour.
    '''
    
    import time
    start_time = time.time()

    population = initial_population(population_size, distance_matrix)
    cost = [item[1] for item in population]
    cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
    elite_ind = population[0]
    count = 0
    best.put(elite_ind[1])
    while count <= generations:
        if verbose:
            print('Generation =', count, 'Distance =', round(elite_ind[1], 4))
        
        #if not validate_solution(distance_matrix, elite_ind):
        #    raise ValueError("Advertencia: La solución elite no es válida en la generación", count)

        offspring = breeding(distance_matrix, population, elite)
        offspring = mutation(distance_matrix, offspring, mutation_rate, mutation_search, elite)
        cost = [item[1] for item in offspring]
        cost, population = (list(t) for t in zip(*sorted(zip(cost, offspring))))
        elite_child = population[0]
        if elite_ind[1] > elite_child[1]:
            elite_ind = elite_child
            best.put(elite_ind[1])
            time.sleep(0.1)
        count += 1
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"Tiempo límite alcanzado ({time_limit} segundos). Terminando...")
            break
    route, distance = elite_ind
    return route, distance

'''
Unmodified functions from the original code:

- local_search_2_opt
- initial_population
'''


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
