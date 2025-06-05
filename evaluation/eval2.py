import os
import pandas as pd
from time import time
from memory_profiler import memory_usage
import random
import sys
from collections import defaultdict


# Original Graph class for backward compatibility
class OriginalGraph:
    def __init__(self, vertices=0):
        self.V = vertices
        self.graph = [[] for _ in range(vertices)]

    def add_vertex(self, v):
        while v >= self.V:
            self.graph.append([])
            self.V += 1

    def add_edge(self, u, v):
        self.add_vertex(u)
        self.add_vertex(v)

        if v not in self.graph[u]:
            self.graph[u].append(v)
        if u not in self.graph[v]:
            self.graph[v].append(u)

    def get_neighbors(self, v):
        return self.graph[v]


# New optimized Graph class
class Graph:
    def __init__(self):
        self.vertices = set()
        self.edges = {}
        self.num_vertices = 0
        self.num_edges = 0
        self._adj_lists = {}

    def add_vertex(self, v):
        if v not in self.vertices:
            self.vertices.add(v)
            self.edges[v] = set()
            self.num_vertices += 1

    def add_edge(self, v1, v2):
        self.add_vertex(v1)
        self.add_vertex(v2)
        if v2 not in self.edges[v1]:
            self.edges[v1].add(v2)
            self.edges[v2].add(v1)
            self.num_edges += 1

    def finalize(self):
        self._adj_lists = {v: list(neighbors) for v, neighbors in self.edges.items()}

    def get_neighbors(self, v):
        return self._adj_lists.get(v, [])

    def get_degree(self, v):
        return len(self._adj_lists.get(v, []))

    def get_density(self):
        if self.num_vertices <= 1:
            return 0
        return 2 * self.num_edges / (self.num_vertices * (self.num_vertices - 1))


# Original Greedy Coloring Algorithm
def original_greedy_coloring(graph):
    """Original greedy coloring algorithm using OriginalGraph"""
    result = [-1] * graph.V
    available = [False] * graph.V
    result[0] = 0

    for u in range(1, graph.V):
        for i in graph.graph[u]:
            if result[i] != -1:
                available[result[i]] = True

        color = next(c for c, is_used in enumerate(available) if not is_used)
        result[u] = color
        available = [False] * graph.V

    return result


# New optimized Greedy DSATUR Coloring
def greedy_dsatur_coloring(graph, randomize=False):
    """New optimized greedy DSATUR coloring algorithm"""
    max_degree = max(graph.get_degree(v) for v in graph.vertices) if graph.vertices else 1
    max_colors = max_degree + 1

    coloring = {}
    degrees = {v: graph.get_degree(v) for v in graph.vertices}
    colored_neighbors = {v: set() for v in graph.vertices}
    uncolored = set(graph.vertices)

    if not uncolored:
        return coloring

    if randomize and len(uncolored) > 1:
        top_candidates = sorted(uncolored, key=lambda v: degrees[v], reverse=True)[:5]
        start_vertex = random.choice(top_candidates)
    else:
        start_vertex = max(uncolored, key=lambda v: degrees[v])

    coloring[start_vertex] = 1
    uncolored.remove(start_vertex)

    for neighbor in graph.get_neighbors(start_vertex):
        if neighbor in uncolored:
            colored_neighbors[neighbor].add(1)

    while uncolored:
        candidates = list(uncolored)
        if randomize and len(candidates) > 3:
            candidates = sorted(candidates, key=lambda x: (len(colored_neighbors[x]), degrees[x], random.random()),
                                reverse=True)[:3]
            v = random.choice(candidates)
        else:
            v = max(uncolored, key=lambda x: (len(colored_neighbors[x]), degrees[x]))

        used_colors = colored_neighbors[v]
        color = 1
        while color <= max_colors and color in used_colors:
            color += 1

        if color > max_colors:
            color = random.randint(1, max_colors)

        coloring[v] = color
        uncolored.remove(v)

        for neighbor in graph.get_neighbors(v):
            if neighbor in uncolored:
                colored_neighbors[neighbor].add(color)

    return coloring


# Genetic Algorithm (original implementation)
def genetic_coloring(graph, population_size=100, generations=200):
    """Original genetic algorithm for graph coloring"""

    def fitness(solution):
        unique_colors = len(set(solution))
        penalty = 0
        for vertex in range(graph.V):
            for neighbor in graph.graph[vertex]:
                if solution[vertex] == solution[neighbor]:
                    penalty += 1
        return unique_colors + penalty * graph.V

    def is_valid_solution(solution):
        for vertex in range(graph.V):
            for neighbor in graph.graph[vertex]:
                if solution[vertex] == solution[neighbor]:
                    return False
        return True

    def mutate(solution):
        index = random.randint(0, graph.V - 1)
        solution[index] = random.randint(0, graph.V - 1)

    # Initialize the population with greedy solutions
    population = [original_greedy_coloring(graph) for _ in range(population_size)]

    # Evolve the population over generations
    for _ in range(generations):
        population.sort(key=fitness)
        next_gen = population[:population_size // 2]
        while len(next_gen) < population_size:
            parent1, parent2 = random.sample(next_gen, 2)
            crossover_point = random.randint(1, graph.V - 2)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            mutate(child)
            next_gen.append(child)
        population = next_gen

    # Return the best valid solution
    for solution in population:
        if is_valid_solution(solution):
            return solution

    return population[0]


# Original POP2 Algorithm helper functions (from evaluator.py)
def initial_coloring(graph, num_colors):
    coloring = {}
    for vertex in graph.vertices:
        coloring[vertex] = random.randint(1, num_colors)
    return coloring


def evaluate_conflicts_fast(graph, coloring):
    conflicts = 0
    for v in graph.vertices:
        v_color = coloring[v]
        for neighbor in graph.get_neighbors(v):
            if v_color == coloring[neighbor]:
                conflicts += 1
    return conflicts >> 1


def get_vertex_conflicts_fast(graph, coloring, vertex):
    vertex_color = coloring[vertex]
    conflicts = 0
    for neighbor in graph.get_neighbors(vertex):
        if vertex_color == coloring[neighbor]:
            conflicts += 1
    return conflicts


def get_move_delta_conflicts_fast(graph, coloring, vertex, new_color):
    old_color = coloring[vertex]
    if old_color == new_color:
        return 0
    delta = 0
    for neighbor in graph.get_neighbors(vertex):
        neighbor_color = coloring[neighbor]
        if neighbor_color == old_color:
            delta -= 1
        elif neighbor_color == new_color:
            delta += 1
    return delta


def greedy_dsatur_coloring_pop2(graph, max_colors):
    """DSATUR coloring for POP2 algorithm"""
    coloring = {}
    degrees = {v: len(graph.get_neighbors(v)) for v in graph.vertices}
    colored_neighbors = {v: set() for v in graph.vertices}
    uncolored = set(graph.vertices)

    if not uncolored:
        return coloring

    max_degree_vertex = max(graph.vertices, key=lambda v: degrees[v])
    coloring[max_degree_vertex] = 1
    uncolored.remove(max_degree_vertex)

    for neighbor in graph.get_neighbors(max_degree_vertex):
        if neighbor in uncolored:
            colored_neighbors[neighbor].add(1)

    while uncolored:
        v = max(uncolored, key=lambda x: (len(colored_neighbors[x]), degrees[x]))
        used_colors = colored_neighbors[v]
        color = 1
        while color <= max_colors:
            if color not in used_colors:
                break
            color += 1

        if color > max_colors:
            color = random.randint(1, max_colors)

        coloring[v] = color
        uncolored.remove(v)

        for neighbor in graph.get_neighbors(v):
            if neighbor in uncolored:
                colored_neighbors[neighbor].add(color)

    return coloring


# Original POP2 Algorithm (from evaluator.py)
def original_pop2_algorithm(graph, max_iterations=1000, max_time=60, tabu_tenure=10, verbose=False):
    """
    Original POP2 algorithm for graph coloring with tabu search and dynamic temperature
    Fixed to only return valid colorings with 0 conflicts
    """
    if graph.num_vertices == 0:
        return {}, 0

    density = 2 * graph.num_edges / (graph.num_vertices * (graph.num_vertices - 1))
    initial_colors = min(len(graph.vertices), max(3, int(1.5 * graph.num_vertices * density)))

    # Initialize with None to ensure we only return valid colorings
    best_num_colors = None
    best_coloring = None

    start_time = time()

    for num_colors in range(initial_colors, 0, -1):
        if time() - start_time > max_time:
            if verbose:
                print("Time limit reached.")
            break

        coloring = greedy_dsatur_coloring_pop2(graph, num_colors)
        conflict_count = {v: get_vertex_conflicts_fast(graph, coloring, v) for v in graph.vertices}
        current_conflicts = sum(conflict_count.values()) // 2

        if current_conflicts == 0:
            # Found a valid coloring with this number of colors
            best_coloring = coloring.copy()
            best_num_colors = num_colors
            continue

        local_best_coloring = coloring.copy()
        local_best_conflicts = current_conflicts

        tabu_list = {}
        current_tabu_tenure = tabu_tenure
        temperature = 1.0
        non_improving_iterations = 0
        plateau_iterations = 0

        iteration = 0
        while iteration < max_iterations:
            if time() - start_time > max_time:
                if verbose:
                    print("Time limit reached.")
                break  # Break inner loop, don't return here

            conflict_vertices = [(v, count) for v, count in conflict_count.items() if count > 0]

            if not conflict_vertices:
                # Found a valid coloring with 0 conflicts
                best_coloring = coloring.copy()
                best_num_colors = num_colors
                break

            if len(conflict_vertices) > 1:
                if random.random() >= temperature:
                    conflict_vertices.sort(key=lambda x: x[1], reverse=True)
                    v = conflict_vertices[0][0]
                else:
                    selection_range = min(len(conflict_vertices),
                                          max(1, int(temperature * len(conflict_vertices) // 2)))
                    top_conflicts = sorted(conflict_vertices, key=lambda x: x[1], reverse=True)[:selection_range]
                    v = random.choice(top_conflicts)[0]
            else:
                v = conflict_vertices[0][0]

            current_color = coloring[v]
            best_move_color = current_color
            best_move_delta = 0

            for color in range(1, num_colors + 1):
                if color == current_color:
                    continue

                is_tabu = (v, color) in tabu_list and tabu_list[(v, color)] > iteration
                delta = get_move_delta_conflicts_fast(graph, coloring, v, color)

                if (not is_tabu or current_conflicts + delta < local_best_conflicts) and delta < best_move_delta:
                    best_move_color = color
                    best_move_delta = delta

            if best_move_color != current_color:
                tabu_list[(v, current_color)] = iteration + current_tabu_tenure
                old_color = current_color
                new_color = best_move_color

                for neighbor in graph.get_neighbors(v):
                    if coloring[neighbor] == old_color:
                        conflict_count[neighbor] -= 1
                    if coloring[neighbor] == new_color:
                        conflict_count[neighbor] += 1

                coloring[v] = new_color
                current_conflicts += best_move_delta
                conflict_count[v] = get_vertex_conflicts_fast(graph, coloring, v)

                if current_conflicts < local_best_conflicts:
                    local_best_coloring = coloring.copy()
                    local_best_conflicts = current_conflicts
                    non_improving_iterations = 0
                    plateau_iterations = 0
                else:
                    non_improving_iterations += 1

                    if best_move_delta == 0:
                        plateau_iterations += 1
                    else:
                        plateau_iterations = 0
            else:
                non_improving_iterations += 1

            if non_improving_iterations > 100:
                temperature = min(1.0, temperature * 1.05)

                if non_improving_iterations > 300:
                    coloring = local_best_coloring.copy()
                    conflict_count = {v: get_vertex_conflicts_fast(graph, coloring, v) for v in graph.vertices}
                    current_conflicts = local_best_conflicts
                    non_improving_iterations = 0
            else:
                temperature = max(0.01, temperature * 0.995)

            if plateau_iterations > 30:
                current_tabu_tenure = min(40, current_tabu_tenure + 1)
            elif non_improving_iterations > 150:
                current_tabu_tenure = max(5, current_tabu_tenure - 1)

            iteration += 1

        # Stop trying fewer colors if we couldn't find a valid coloring
        if current_conflicts > 0 and local_best_conflicts > 0:
            break

    # Only return a valid coloring (0 conflicts) or fallback to a safe coloring
    if best_coloring is not None and best_num_colors is not None:
        return best_coloring, best_num_colors
    else:
        # Fallback: use a simple greedy coloring that guarantees 0 conflicts
        if verbose:
            print("No valid coloring found, using fallback greedy coloring")
        fallback_coloring = greedy_dsatur_coloring_pop2(graph, len(graph.vertices))
        # Verify it has 0 conflicts
        fallback_conflicts = evaluate_conflicts_fast(graph, fallback_coloring)
        if fallback_conflicts == 0:
            return fallback_coloring, len(set(fallback_coloring.values()))
        else:
            # Ultimate fallback: assign each vertex a unique color
            unique_coloring = {v: i + 1 for i, v in enumerate(graph.vertices)}
            return unique_coloring, len(graph.vertices)


def intensive_tabu_search_optimized(graph, coloring, num_colors, max_iterations=5000, max_time=60, verbose=False):
    """Optimized tabu search for graph coloring"""
    current_conflicts = evaluate_conflicts_fast(graph, coloring)
    best_coloring = coloring.copy()
    best_conflicts = current_conflicts
    base_tabu_tenure = max(5, min(15, graph.num_vertices // 50))
    tabu_list = {}
    conflict_count = {}

    for v in graph.vertices:
        conflict_count[v] = get_vertex_conflicts_fast(graph, coloring, v)

    start_time = time()
    non_improving = 0
    restart_limit = 500
    temperature = 0.7
    temp_decay = 0.998
    conflict_vertices = []

    for iteration in range(max_iterations):
        if time() - start_time > max_time:
            break

        conflict_vertices.clear()
        for v, count in conflict_count.items():
            if count > 0:
                conflict_vertices.append(v)

        if not conflict_vertices:
            break

        if non_improving < 50 or random.random() > temperature:
            vertex = max(conflict_vertices, key=lambda v: (conflict_count[v], graph.get_degree(v)))
        else:
            vertex = random.choice(conflict_vertices)

        current_color = coloring[vertex]
        best_color = current_color
        best_delta = float('inf')

        for color in range(1, num_colors + 1):
            if color == current_color:
                continue

            delta = get_move_delta_conflicts_fast(graph, coloring, vertex, color)
            tabu_key = (vertex, color)
            is_tabu = tabu_key in tabu_list and tabu_list[tabu_key] > iteration
            aspiration = current_conflicts + delta < best_conflicts

            if (not is_tabu or aspiration) and delta < best_delta:
                best_color = color
                best_delta = delta

        if best_color != current_color:
            tenure_variation = random.randint(-3, 3)
            tabu_list[(vertex, current_color)] = iteration + base_tabu_tenure + tenure_variation

            old_color = current_color
            coloring[vertex] = best_color
            neighbors = graph.get_neighbors(vertex)

            for neighbor in neighbors:
                neighbor_color = coloring[neighbor]
                if neighbor_color == old_color:
                    conflict_count[neighbor] -= 1
                elif neighbor_color == best_color:
                    conflict_count[neighbor] += 1

            current_conflicts += best_delta
            conflict_count[vertex] = get_vertex_conflicts_fast(graph, coloring, vertex)

            if current_conflicts < best_conflicts:
                best_coloring = coloring.copy()
                best_conflicts = current_conflicts
                non_improving = 0
            else:
                non_improving += 1
        else:
            non_improving += 1

        if non_improving > restart_limit:
            coloring = best_coloring.copy()
            for v in graph.vertices:
                conflict_count[v] = get_vertex_conflicts_fast(graph, coloring, v)
            current_conflicts = best_conflicts
            non_improving = 0
            temperature = 0.7
            tabu_list.clear()

        temperature *= temp_decay
        temperature = max(0.05, temperature)

    return best_coloring, best_conflicts


def aggressive_pop2_algorithm(graph, target_colors=None, max_iterations=5000, max_time=300, verbose=False):
    """New aggressive POP2 algorithm"""
    if graph.num_vertices == 0:
        return {}, 0

    density = graph.get_density()
    max_degree = max(graph.get_degree(v) for v in graph.vertices)

    if target_colors is None:
        start_colors = len(set(greedy_dsatur_coloring(graph, randomize=False).values()))
    else:
        start_colors = target_colors

    best_num_colors = start_colors
    best_coloring = None
    best_conflicts = float('inf')
    start_time = time()

    for num_colors in range(start_colors, 3, -1):
        if time() - start_time > max_time:
            break

        best_initial_conflicts = float('inf')
        best_initial_coloring = None

        for attempt in range(6):
            if attempt % 3 == 0:
                coloring = greedy_dsatur_coloring(graph, randomize=False)
            elif attempt % 3 == 1:
                coloring = greedy_dsatur_coloring(graph, randomize=True)
            else:
                coloring = initial_coloring(graph, num_colors)

            conflicts = evaluate_conflicts_fast(graph, coloring)
            if conflicts < best_initial_conflicts:
                best_initial_conflicts = conflicts
                best_initial_coloring = coloring.copy()

        if best_initial_conflicts == 0:
            best_coloring = best_initial_coloring
            best_num_colors = num_colors
            best_conflicts = 0
            continue

        improved_coloring, final_conflicts = intensive_tabu_search_optimized(
            graph, best_initial_coloring, num_colors,
            max_iterations=max_iterations,
            max_time=max_time - (time() - start_time),
            verbose=verbose and num_colors % 20 == 0
        )

        if final_conflicts < best_conflicts:
            best_coloring = improved_coloring
            best_num_colors = num_colors
            best_conflicts = final_conflicts

        if final_conflicts == 0:
            best_coloring = improved_coloring
            best_num_colors = num_colors
            best_conflicts = 0
            continue
        else:
            break

    if best_coloring is None:
        return initial_coloring(graph, start_colors), start_colors

    return best_coloring, best_num_colors


# Adapter functions to convert between graph formats
def convert_to_original_graph(new_graph):
    """Convert new Graph to OriginalGraph format"""
    max_vertex = max(new_graph.vertices) if new_graph.vertices else 0
    original_graph = OriginalGraph(max_vertex + 1)

    for v in new_graph.vertices:
        for neighbor in new_graph.get_neighbors(v):
            if v < neighbor:  # Avoid adding edges twice
                original_graph.add_edge(v, neighbor)

    return original_graph


def convert_new_coloring_to_list(coloring, num_vertices):
    """Convert new graph coloring dict to list format"""
    result = [0] * num_vertices
    for v, color in coloring.items():
        if v < num_vertices:
            result[v] = color - 1  # Convert to 0-indexed
    return result


def new_greedy_adapter(new_graph):
    """Adapter for new greedy algorithm"""
    coloring = greedy_dsatur_coloring(new_graph, randomize=False)
    max_vertex = max(new_graph.vertices) if new_graph.vertices else 0
    return convert_new_coloring_to_list(coloring, max_vertex + 1)


def original_pop2_adapter(new_graph):
    """Adapter for original POP2 algorithm"""
    coloring, num_colors = original_pop2_algorithm(
        new_graph,
        max_iterations=1000,
        max_time=60,
        tabu_tenure=10,
        verbose=False
    )
    max_vertex = max(new_graph.vertices) if new_graph.vertices else 0
    return convert_new_coloring_to_list(coloring, max_vertex + 1)


def new_pop2_adapter(new_graph):
    """Adapter for new POP2 algorithm"""
    coloring, num_colors = aggressive_pop2_algorithm(
        new_graph,
        max_iterations=1000,
        max_time=30,
        verbose=False
    )
    max_vertex = max(new_graph.vertices) if new_graph.vertices else 0
    return convert_new_coloring_to_list(coloring, max_vertex + 1)


def read_graph_from_file(filename):
    """Read graph from file and return both formats"""
    new_graph = Graph()

    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue
                if line.startswith('p'):
                    continue
                elif line.startswith('e'):
                    parts = line.split()
                    if len(parts) >= 3:
                        v1 = int(parts[1]) - 1  # Convert to 0-indexed
                        v2 = int(parts[2]) - 1
                        if v1 != v2:
                            new_graph.add_edge(v1, v2)

        new_graph.finalize()
        original_graph = convert_to_original_graph(new_graph)

    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

    return original_graph, new_graph


def analyze_performance(func, *args):
    """Analyze performance of a function"""
    mem_before = memory_usage()[0]
    start_time = time()
    result = func(*args)
    end_time = time()
    mem_after = memory_usage()[0]
    execution_time = end_time - start_time
    memory_usage_diff = mem_after - mem_before
    return result, execution_time, memory_usage_diff


if __name__ == "__main__":
    data_folder = "../data"
    results = []

    # Check if data folder exists
    if not os.path.exists(data_folder):
        print(f"Data folder '{data_folder}' not found. Creating it...")
        os.makedirs(data_folder)
        print(f"Please place .col graph files in the '{data_folder}' directory.")
        sys.exit(1)

    # Iterate through all files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".col") and filename.startswith("DSJC125.1"):
            filepath = os.path.join(data_folder, filename)
            print(f"Processing file: {filename}")

            # Read the graph from the file
            original_graph, new_graph = read_graph_from_file(filepath)
            if original_graph is None or new_graph is None:
                print(f"Failed to read {filename}")
                continue

            print(f"Graph loaded from {filename} with {original_graph.V} vertices and {new_graph.num_edges} edges")

            # Add a header for the file
            results.append({
                "Run": "Header",
                "Algorithm": f"File: {filename}",
                "Colors Used": "",
                "Execution Time (s)": "",
                "Memory Usage (MiB)": ""
            })

            # Determine number of runs based on graph size
            num_runs = 50 if original_graph.V < 500 else (20 if original_graph.V < 1000 else 5)

            # Run Original Greedy Coloring
            print("Running Original Greedy Coloring...")
            original_greedy_runs = []
            for run in range(1, num_runs + 1):
                result, exec_time, memory_use = analyze_performance(original_greedy_coloring, original_graph)
                colors = len(set(result))
                original_greedy_runs.append((colors, exec_time, memory_use))
                results.append({
                    "Run": run,
                    "Algorithm": "Original Greedy",
                    "Colors Used": colors,
                    "Execution Time (s)": exec_time,
                    "Memory Usage (MiB)": memory_use
                })

            results.append({
                "Run": "End",
                "Algorithm": "Original Greedy",
                "Colors Used": "---",
                "Execution Time (s)": "---",
                "Memory Usage (MiB)": "---"
            })

            # Run New Greedy Coloring
            print("Running New Greedy DSATUR Coloring...")
            new_greedy_runs = []
            for run in range(1, num_runs + 1):
                result, exec_time, memory_use = analyze_performance(new_greedy_adapter, new_graph)
                colors = len(set(result))
                new_greedy_runs.append((colors, exec_time, memory_use))
                results.append({
                    "Run": run,
                    "Algorithm": "New Greedy DSATUR",
                    "Colors Used": colors,
                    "Execution Time (s)": exec_time,
                    "Memory Usage (MiB)": memory_use
                })

            results.append({
                "Run": "End",
                "Algorithm": "New Greedy DSATUR",
                "Colors Used": "---",
                "Execution Time (s)": "---",
                "Memory Usage (MiB)": "---"
            })

            # Run Genetic Algorithm
            print("Running Genetic Algorithm...")
            genetic_runs = []
            for run in range(1, num_runs + 1):
                result, exec_time, memory_use = analyze_performance(genetic_coloring, original_graph)
                colors = len(set(result))
                genetic_runs.append((colors, exec_time, memory_use))
                results.append({
                    "Run": run,
                    "Algorithm": "Genetic Algorithm",
                    "Colors Used": colors,
                    "Execution Time (s)": exec_time,
                    "Memory Usage (MiB)": memory_use
                })

            results.append({
                "Run": "End",
                "Algorithm": "Genetic Algorithm",
                "Colors Used": "---",
                "Execution Time (s)": "---",
                "Memory Usage (MiB)": "---"
            })

            # Run Original POP2 Algorithm
            print("Running Original POP2 Algorithm...")
            original_pop2_runs = []
            for run in range(1, num_runs + 1):
                result, exec_time, memory_use = analyze_performance(original_pop2_adapter, new_graph)
                colors = len(set(result))
                original_pop2_runs.append((colors, exec_time, memory_use))
                results.append({
                    "Run": run,
                    "Algorithm": "Original POP2",
                    "Colors Used": colors,
                    "Execution Time (s)": exec_time,
                    "Memory Usage (MiB)": memory_use
                })

            results.append({
                "Run": "End",
                "Algorithm": "Original POP2",
                "Colors Used": "---",
                "Execution Time (s)": "---",
                "Memory Usage (MiB)": "---"
            })

            # Run New POP2 Algorithm
            print("Running New POP2 Algorithm...")
            new_pop2_runs = []
            for run in range(1, num_runs + 1):
                result, exec_time, memory_use = analyze_performance(new_pop2_adapter, new_graph)
                colors = len(set(result))
                new_pop2_runs.append((colors, exec_time, memory_use))
                results.append({
                    "Run": run,
                    "Algorithm": "New POP2",
                    "Colors Used": colors,
                    "Execution Time (s)": exec_time,
                    "Memory Usage (MiB)": memory_use
                })

            results.append({
                "Run": "End",
                "Algorithm": "New POP2",
                "Colors Used": "---",
                "Execution Time (s)": "---",
                "Memory Usage (MiB)": "---"
            })

            # Calculate and add best/worst runs for each algorithm
            for algorithm, runs in [("Original Greedy", original_greedy_runs),
                                    ("New Greedy DSATUR", new_greedy_runs),
                                    ("Genetic Algorithm", genetic_runs),
                                    ("Original POP2", original_pop2_runs),
                                    ("New POP2", new_pop2_runs)]:
                if runs:
                    best_run = min(runs, key=lambda x: x[0])
                    worst_run = max(runs, key=lambda x: x[0])
                    results.append({
                        "Run": "Best",
                        "Algorithm": algorithm,
                        "Colors Used": best_run[0],
                        "Execution Time (s)": best_run[1],
                        "Memory Usage (MiB)": best_run[2]
                    })
                    results.append({
                        "Run": "Worst",
                        "Algorithm": algorithm,
                        "Colors Used": worst_run[0],
                        "Execution Time (s)": worst_run[1],
                        "Memory Usage (MiB)": worst_run[2]
                    })
    # Create a DataFrame and save results to CSV
    df = pd.DataFrame(results)
    output_file = os.path.join(data_folder, "evaluation_results.csv")
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print("Evaluation completed.")
