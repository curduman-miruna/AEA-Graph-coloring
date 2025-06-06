import os
import pandas as pd
from time import time
from memory_profiler import memory_usage
import random
import sys

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

def greedy_dsatur_coloring(graph, max_colors=None, randomize=False):
    """New optimized greedy DSATUR coloring algorithm"""
    if max_colors is None:
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

def genetic_coloring(graph, population_size=100, generations=200):
    """Original genetic algorithm adapted for new graph representation"""

    # convert vertex set to list for easier indexing
    vertex_list = list(graph.vertices)
    vertex_count = len(vertex_list)

    if vertex_count == 0:
        return {}

    def dict_to_list(coloring_dict):
        """Convert coloring dictionary to list format"""
        result = [0] * vertex_count
        for i, vertex in enumerate(vertex_list):
            result[i] = coloring_dict.get(vertex, 1) - 1  # Convert to 0-indexed
        return result

    def list_to_dict(coloring_list):
        """Convert coloring list to dictionary format"""
        result = {}
        for i, color in enumerate(coloring_list):
            result[vertex_list[i]] = color
        return result

    def fitness(solution):
        unique_colors = len(set(solution))
        penalty = 0
        solution_dict = list_to_dict(solution)

        for vertex in graph.vertices:
            for neighbor in graph.get_neighbors(vertex):
                if solution_dict[vertex] == solution_dict[neighbor]:
                    penalty += 1

        return unique_colors + penalty * vertex_count

    def is_valid_solution(solution):
        solution_dict = list_to_dict(solution)
        for vertex in graph.vertices:
            for neighbor in graph.get_neighbors(vertex):
                if solution_dict[vertex] == solution_dict[neighbor]:
                    return False
        return True

    def mutate(solution):
        index = random.randint(0, vertex_count - 1)
        solution[index] = random.randint(0, vertex_count - 1)

    population = []
    for _ in range(population_size):
        greedy_coloring = greedy_dsatur_coloring(graph, randomize=True)
        population.append(dict_to_list(greedy_coloring))

    # evolve the population over generations
    for _ in range(generations):
        population.sort(key=fitness)
        next_gen = population[:population_size // 2]
        while len(next_gen) < population_size:
            parent1, parent2 = random.sample(next_gen, 2)
            crossover_point = random.randint(1, vertex_count - 2)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            mutate(child)
            next_gen.append(child)
        population = next_gen

    # return the best valid solution
    for solution in population:
        if is_valid_solution(solution):
            return list_to_dict([color + 1 for color in solution])  # Convert back to 1-indexed

    # return best solution found (convert back to 1-indexed)
    best_solution = list_to_dict([color + 1 for color in population[0]])
    return best_solution


def read_graph_from_file(filename):
    """Read graph from file"""
    graph = Graph()

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
                            graph.add_edge(v1, v2)

        graph.finalize()

    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    return graph


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

    if not os.path.exists(data_folder):
        print(f"Data folder '{data_folder}' not found. Creating it...")
        os.makedirs(data_folder)
        print(f"Please place .col graph files in the '{data_folder}' directory.")
        sys.exit(1)

    for filename in os.listdir(data_folder):
        if filename.endswith(".col") and not filename.startswith('DSJC'):
            filepath = os.path.join(data_folder, filename)
            print(f"Processing file: {filename}")

            graph = read_graph_from_file(filepath)
            if graph is None:
                print(f"Failed to read {filename}")
                continue

            print(f"Graph loaded from {filename} with {graph.num_vertices} vertices and {graph.num_edges} edges")
            results.append({
                "Run": "Header",
                "Algorithm": f"File: {filename}",
                "Colors Used": "",
                "Execution Time (s)": "",
                "Memory Usage (MiB)": ""
            })

            # determine number of runs based on graph size
            num_runs = 100 if graph.num_vertices < 500 else (20 if graph.num_vertices < 1000 else 5)

            print("Running Genetic Algorithm...")
            genetic_runs = []
            for run in range(1, num_runs + 1):
                result, exec_time, memory_use = analyze_performance(genetic_coloring, graph)
                colors = len(set(result.values())) if result else 0
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

    # convert results to DataFrame and save to CSV
    df = pd.DataFrame(results)
    output_file = os.path.join(data_folder, "genetic_algorithm_results.csv")
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print("Evaluation completed successfully.")