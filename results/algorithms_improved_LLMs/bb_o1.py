
# Required Libraries
import numpy as np

############################################################################

def nearest_neighbor(distance_matrix):
    """
    Nearest Neighbor Heuristic to obtain an initial feasible solution.
    
    This function computes an initial tour using the Nearest Neighbor heuristic,
    which starts at the first node and always moves to the nearest unvisited node.
    The total cost of this tour serves as an initial upper bound for the Branch and
    Bound algorithm.

    By providing a better initial upper bound, the algorithm can prune suboptimal paths
    more effectively, leading to faster convergence.

    This technique is based on classical heuristic methods for the TSP and improves
    the performance of Branch and Bound algorithms by reducing the search space.
    """
    n = distance_matrix.shape[0]
    start_node = 0
    unvisited = set(range(n))
    unvisited.remove(start_node)
    tour = [start_node]
    current_node = start_node
    total_distance = 0

    while unvisited:
        nearest_node = min(unvisited, key=lambda x: distance_matrix[current_node, x])
        total_distance += distance_matrix[current_node, nearest_node]
        current_node = nearest_node
        tour.append(current_node)
        unvisited.remove(current_node)
    total_distance += distance_matrix[current_node, start_node]
    tour.append(start_node)
    return tour, total_distance

############################################################################

def prim_mst_cost(distance_matrix, unvisited_nodes):
    """
    Computes the cost of a Minimum Spanning Tree (MST) over unvisited nodes using Prim's algorithm.
    
    The MST cost serves as a lower bound for the unexplored part of the tour. By adding this
    lower bound to the current path cost, we obtain a tighter overall lower bound, enabling
    more effective pruning in the Branch and Bound algorithm.

    This approach is based on state-of-the-art bounding techniques that use relaxations of the TSP
    (in this case, the MST) to obtain better lower bounds and improve performance.
    """
    if len(unvisited_nodes) == 0 or len(unvisited_nodes) == 1:
        return 0
    n = distance_matrix.shape[0]
    visited = {next(iter(unvisited_nodes))}
    mst_cost = 0
    while len(visited) < len(unvisited_nodes):
        crossing = set()
        for u in visited:
            for v in unvisited_nodes:
                if v not in visited:
                    crossing.add((u, v, distance_matrix[u, v]))
        if not crossing:
            break  # No crossing edges, graph might be disconnected
        edge = min(crossing, key=lambda e: e[2])
        mst_cost += edge[2]
        visited.add(edge[1])
    return mst_cost
    
############################################################################

def explore_path(route, distance, distance_matrix, weight, level, path, visited):  
    """
    Explores paths recursively, using an improved lower bound based on MST and initial upper bound.

    Improvements Implemented:
    - Uses the cost of the Minimum Spanning Tree (MST) over unvisited nodes to compute a tighter lower bound.
    - Utilizes an initial feasible solution obtained from the Nearest Neighbor heuristic as an upper bound.

    How it Enhances Performance:
    - A tighter lower bound allows the algorithm to prune suboptimal paths earlier, reducing the number of nodes explored.
    - A better initial upper bound (from Nearest Neighbor) helps in further pruning by limiting the maximum allowable tour cost.

    Based on State-of-the-Art Techniques:
    - Incorporates bounding strategies that use relaxations of the TSP (MST in this case) to compute lower bounds.
    - Uses heuristic methods to initialize upper bounds, a common practice in modern optimization algorithms.
    """
    n = distance_matrix.shape[0]
    if level == n: 
        if distance_matrix[path[level - 1], path[0]] != 0:
            current_distance = weight + distance_matrix[path[level - 1], path[0]]
            if current_distance < distance:
                distance                             = current_distance
                route[:n + 1] = path[:]
                route[n]      = path[0]
        return route, distance
    for i in range(n):
        if distance_matrix[path[level -1], i] != 0 and not visited[i]:
            temp_weight = weight + distance_matrix[path[level -1], i]
            path[level] = i
            temp_visited = visited.copy()
            temp_visited[i] = True
            # Compute lower bound
            unvisited_nodes = set(j for j in range(n) if not temp_visited[j])
            mst_cost = prim_mst_cost(distance_matrix, unvisited_nodes)
            if unvisited_nodes:
                min_edge_to_unvisited = min(distance_matrix[path[level], j] for j in unvisited_nodes)
                min_edge_from_unvisited = min(distance_matrix[j, path[0]] for j in unvisited_nodes)
            else:
                min_edge_to_unvisited = distance_matrix[path[level], path[0]]
                min_edge_from_unvisited = 0
            lower_bound = temp_weight + mst_cost + min_edge_to_unvisited + min_edge_from_unvisited
            if lower_bound < distance:
                route, distance = explore_path(route, distance, distance_matrix, temp_weight, level +1, path, temp_visited)
    return route, distance

############################################################################

def branch_and_bound(distance_matrix):
    """
    Branch and Bound algorithm for solving the Traveling Salesman Problem, enhanced with:
    - Initial upper bound using Nearest Neighbor heuristic.
    - Improved lower bound using Minimum Spanning Tree (MST) cost over unvisited nodes.

    Improvements Implemented:
    - The Nearest Neighbor heuristic provides a good initial feasible solution, which serves as a tight upper bound.
    - The MST-based lower bound offers a closer estimation of the minimal possible addition to the current path cost.

    How it Enhances Performance:
    - A better initial upper bound reduces the search space, as any partial paths exceeding this bound can be pruned.
    - A tighter lower bound allows earlier pruning of paths that cannot lead to a better solution than the current best.

    Based on State-of-the-Art Techniques:
    - Incorporation of heuristic solutions for initializing bounds.
    - Use of relaxation methods (like computing MST over unvisited nodes) to obtain effective lower bounds in combinatorial optimization.

    Parameters:
    - distance_matrix: 2D NumPy array representing the distances between nodes.

    Returns:
    - route: List representing the order in which nodes are visited in the best found tour.
    - distance: Total distance of the best found tour.
    """
    import time
    start_time = time.time()
    
    n = distance_matrix.shape[0]
    # Initial upper bound using Nearest Neighbor heuristic
    initial_route, initial_distance = nearest_neighbor(distance_matrix)
    distance = initial_distance
    # Initialization
    path = [-1] * (n + 1)
    path[0] = 0
    visited = [False] * n
    visited[0] = True
    route = [None] * (n + 1)
    weight = 0
    level = 1
    route, distance = explore_path(route, distance, distance_matrix, weight, level, path, visited)
    # Adjust route to start from node 1 instead of 0
    route = [node + 1 for node in route]
    return route, distance,  (time.time() - start_time)

############################################################################

# Unmodified Functions from the Original Code:
# - None, all functions have been modified or replaced with improved versions.
