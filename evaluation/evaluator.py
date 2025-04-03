import os
import pandas as pd
from time import time
from memory_profiler import memory_usage
import random
import sys


class Graph:
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



class POP2Graph:
    def __init__(self):
        self.vertices = set()
        self.edges = {}
        self.num_vertices = 0
        self.num_edges = 0

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

    def get_neighbors(self, v):
        return self.edges[v]


def greedy_coloring(graph):
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


def is_safe(graph, v, color, result):
    for neighbor in graph.graph[v]:
        if result[neighbor] == color:
            return False
    return True


def backtrack_util(graph, v, result, m):
    if v == graph.V:
        return True
    for color in range(m):
        if is_safe(graph, v, color, result):
            result[v] = color
            if backtrack_util(graph, v + 1, result, m):
                return True
            result[v] = -1
    return False


def backtracking_coloring(graph, m):
    result = [-1] * graph.V
    if not backtrack_util(graph, 0, result, m):
        return False
    return result


def dsatur_coloring(graph):
    result = [-1] * graph.V
    saturation = [0] * graph.V
    degrees = [len(graph.graph[i]) for i in range(graph.V)]
    vertex = max(range(graph.V), key=lambda x: degrees[x])
    result[vertex] = 0

    for _ in range(1, graph.V):
        for neighbor in graph.graph[vertex]:
            if result[neighbor] != -1:
                saturation[neighbor] += 1
        vertex = max(range(graph.V), key=lambda x: (saturation[x], degrees[x]) if result[x] == -1 else (-1, -1))
        used_colors = {result[n] for n in graph.graph[vertex] if result[n] != -1}
        result[vertex] = next(c for c in range(graph.V) if c not in used_colors)
    return result


def genetic_coloring(graph, population_size=100, generations=200):
    def fitness(solution):
        return len(set(solution))

    def mutate(solution):
        index = random.randint(0, graph.V - 1)
        solution[index] = random.randint(0, graph.V - 1)

    population = [greedy_coloring(graph) for _ in range(population_size)]
    for _ in range(generations):
        population.sort(key=fitness)
        if fitness(population[0]) <= 3:
            break
        next_gen = population[:population_size // 2]
        while len(next_gen) < population_size:
            parent1, parent2 = random.sample(next_gen, 2)
            crossover_point = random.randint(1, graph.V - 2)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            mutate(child)
            next_gen.append(child)
        population = next_gen
    return population[0]


def initial_coloring(graph, num_colors):
    coloring = {}
    for vertex in graph.vertices:
        coloring[vertex] = random.randint(1, num_colors)
    return coloring


def evaluate_conflicts(graph, coloring):
    conflicts = 0
    for v in graph.vertices:
        for neighbor in graph.get_neighbors(v):
            if coloring[v] == coloring[neighbor]:
                conflicts += 1
    return conflicts // 2


def get_vertex_conflicts(graph, coloring, vertex):
    """Count conflicts for a specific vertex"""
    conflicts = 0
    for neighbor in graph.get_neighbors(vertex):
        if coloring[vertex] == coloring[neighbor]:
            conflicts += 1
    return conflicts


def get_move_delta_conflicts(graph, coloring, vertex, new_color):
    """Calculate the change in conflicts if vertex changes to new_color"""
    old_color = coloring[vertex]
    if old_color == new_color:
        return 0

    delta = 0
    for neighbor in graph.get_neighbors(vertex):
        if coloring[neighbor] == old_color:
            delta -= 1
        if coloring[neighbor] == new_color:
            delta += 1

    return delta


def greedy_dsatur_coloring_pop2(graph, max_colors):
    coloring = {}

    degrees = {v: len(graph.get_neighbors(v)) for v in graph.vertices}

    colored_neighbors = {v: set() for v in graph.vertices}
    uncolored = set(graph.vertices)

    max_degree_vertex = max(graph.vertices, key=lambda v: degrees[v])
    coloring[max_degree_vertex] = 1
    uncolored.remove(max_degree_vertex)

    for neighbor in graph.get_neighbors(max_degree_vertex):
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


def pop2_algorithm(graph, max_iterations=1000, max_time=60, tabu_tenure=10, verbose=False):
    """
    POP2 algorithm for graph coloring with tabu search and dynamic temperature
    Optimized for performance
    """
    density = 2 * graph.num_edges / (graph.num_vertices * (graph.num_vertices - 1))
    initial_colors = min(len(graph.vertices), max(3, int(1.5 * graph.num_vertices * density)))
    best_num_colors = initial_colors
    best_coloring = initial_coloring(graph, best_num_colors)
    best_conflicts = evaluate_conflicts(graph, best_coloring)

    start_time = time()

    for num_colors in range(best_num_colors, 0, -1):
        if time() - start_time > max_time:
            if verbose:
                print("Time limit reached.")
            break

        coloring = greedy_dsatur_coloring_pop2(graph, num_colors)
        conflict_count = {v: get_vertex_conflicts(graph, coloring, v) for v in graph.vertices}
        current_conflicts = sum(conflict_count.values()) // 2

        if current_conflicts == 0:
            best_coloring = coloring.copy()
            best_num_colors = num_colors
            best_conflicts = 0
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
                return best_coloring, best_num_colors

            conflict_vertices = [(v, count) for v, count in conflict_count.items() if count > 0]

            if not conflict_vertices:
                best_coloring = coloring.copy()
                best_num_colors = num_colors
                best_conflicts = 0
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
                delta = get_move_delta_conflicts(graph, coloring, v, color)

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
                conflict_count[v] = get_vertex_conflicts(graph, coloring, v)

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
                    conflict_count = {v: get_vertex_conflicts(graph, coloring, v) for v in graph.vertices}
                    current_conflicts = local_best_conflicts
                    non_improving_iterations = 0
            else:
                temperature = max(0.01, temperature * 0.995)

            if plateau_iterations > 30:
                current_tabu_tenure = min(40, current_tabu_tenure + 1)
            elif non_improving_iterations > 150:
                current_tabu_tenure = max(5, current_tabu_tenure - 1)

            iteration += 1

        if current_conflicts > 0 and local_best_conflicts > 0:
            break

    return best_coloring, best_num_colors


def pop2_coloring_adapter(graph):
    pop2_graph = POP2Graph()
    for u in range(graph.V):
        for v in graph.graph[u]:
            if v > u:  # Avoid adding edges twice
                pop2_graph.add_edge(u, v)

    best_coloring, best_num_colors = pop2_algorithm(
        pop2_graph,
        max_iterations=500,
        max_time=30,
        tabu_tenure=10,
        verbose=False
    )

    result = [-1] * graph.V
    for v, color in best_coloring.items():
        if v < graph.V:
            result[v] = color - 1

    for i in range(graph.V):
        if result[i] == -1:
            result[i] = 0

    return result


def read_graph_from_file(filename):
    graph = Graph()

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p'):
                parts = line.split()
                if len(parts) >= 3:
                    num_vertices = int(parts[2])
                    graph = Graph(num_vertices)
                continue
            elif line.startswith('e'):
                parts = line.split()
                if len(parts) >= 3:
                    v1 = int(parts[1]) - 1
                    v2 = int(parts[2]) - 1
                    graph.add_edge(v1, v2)

    return graph


def analyze_performance(func, *args):
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

    if not os.path.exists(data_folder):
        print(f"Data folder '{data_folder}' not found. Creating it...")
        os.makedirs(data_folder)
        print(f"Please place .col graph files in the '{data_folder}' directory.")
        sys.exit(1)

    for filename in os.listdir(data_folder):
        if filename.endswith(".col"):
            filepath = os.path.join(data_folder, filename)
            print(f"Processing file: {filename}")

            graph = read_graph_from_file(filepath)
            print(f"Graph loaded from {filename} with {graph.V} vertices")

            results.append({
                "Run": "Header",
                "Algorithm": f"File: {filename}",
                "Colors Used": "",
                "Execution Time (s)": "",
                "Memory Usage (MiB)": ""
            })

            num_runs = 50 if graph.V < 500 else (20 if graph.V < 1000 else 5)

            print("Running Greedy Coloring...")
            greedy_runs = []
            for run in range(1, num_runs + 1):
                greedy_result, greedy_time, greedy_memory = analyze_performance(greedy_coloring, graph)
                greedy_colors = len(set(greedy_result))
                greedy_runs.append((greedy_colors, greedy_time, greedy_memory))
                results.append({
                    "Run": run,
                    "Algorithm": "Greedy Coloring",
                    "Colors Used": greedy_colors,
                    "Execution Time (s)": greedy_time,
                    "Memory Usage (MiB)": greedy_memory
                })

            results.append({
                "Run": "End",
                "Algorithm": "Greedy Coloring",
                "Colors Used": "---",
                "Execution Time (s)": "---",
                "Memory Usage (MiB)": "---"
            })

            print("Running DSATUR Coloring...")
            dsatur_runs = []
            for run in range(1, num_runs + 1):
                dsatur_result, dsatur_time, dsatur_memory = analyze_performance(dsatur_coloring, graph)
                dsatur_colors = len(set(dsatur_result))
                dsatur_runs.append((dsatur_colors, dsatur_time, dsatur_memory))
                results.append({
                    "Run": run,
                    "Algorithm": "DSATUR Coloring",
                    "Colors Used": dsatur_colors,
                    "Execution Time (s)": dsatur_time,
                    "Memory Usage (MiB)": dsatur_memory
                })

            results.append({
                "Run": "End",
                "Algorithm": "DSATUR Coloring",
                "Colors Used": "---",
                "Execution Time (s)": "---",
                "Memory Usage (MiB)": "---"
            })

            print("Running Genetic Algorithm...")
            genetic_runs = []
            for run in range(1, num_runs + 1):
                genetic_result, genetic_time, genetic_memory = analyze_performance(genetic_coloring, graph)
                genetic_colors = len(set(genetic_result))
                genetic_runs.append((genetic_colors, genetic_time, genetic_memory))
                results.append({
                    "Run": run,
                    "Algorithm": "Genetic Algorithm",
                    "Colors Used": genetic_colors,
                    "Execution Time (s)": genetic_time,
                    "Memory Usage (MiB)": genetic_memory
                })

            results.append({
                "Run": "End",
                "Algorithm": "Genetic Algorithm",
                "Colors Used": "---",
                "Execution Time (s)": "---",
                "Memory Usage (MiB)": "---"
            })

            print("Running POP2 Algorithm...")
            pop2_runs = []
            for run in range(1, num_runs + 1):
                pop2_result, pop2_time, pop2_memory = analyze_performance(pop2_coloring_adapter, graph)
                pop2_colors = len(set(pop2_result))
                pop2_runs.append((pop2_colors, pop2_time, pop2_memory))
                results.append({
                    "Run": run,
                    "Algorithm": "POP2 Algorithm",
                    "Colors Used": pop2_colors,
                    "Execution Time (s)": pop2_time,
                    "Memory Usage (MiB)": pop2_memory
                })

            results.append({
                "Run": "End",
                "Algorithm": "POP2 Algorithm",
                "Colors Used": "---",
                "Execution Time (s)": "---",
                "Memory Usage (MiB)": "---"
            })

            for algorithm, runs in [("Greedy Coloring", greedy_runs),
                                    ("DSATUR Coloring", dsatur_runs),
                                    ("Genetic Algorithm", genetic_runs),
                                    ("POP2 Algorithm", pop2_runs)]:
                best_run = min(runs, key=lambda x: (x[0], x[1]))
                worst_run = max(runs, key=lambda x: (x[0], x[1]))

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

            for algorithm, runs in [("Greedy Coloring", greedy_runs),
                                    ("DSATUR Coloring", dsatur_runs),
                                    ("Genetic Algorithm", genetic_runs),
                                    ("POP2 Algorithm", pop2_runs)]:
                avg_colors = sum(run[0] for run in runs) / len(runs)
                avg_time = sum(run[1] for run in runs) / len(runs)
                avg_memory = sum(run[2] for run in runs) / len(runs)

                results.append({
                    "Run": "Average",
                    "Algorithm": algorithm,
                    "Colors Used": round(avg_colors, 2),
                    "Execution Time (s)": round(avg_time, 4),
                    "Memory Usage (MiB)": round(avg_memory, 4)
                })

            results.append({
                "Run": "End",
                "Algorithm": f"File: {filename}",
                "Colors Used": "---",
                "Execution Time (s)": "---",
                "Memory Usage (MiB)": "---"
            })

    df = pd.DataFrame(results)
    df.to_csv("coloring_results_with_pop2.csv", index=False)
    print("Results saved to coloring_results_with_pop2.csv")