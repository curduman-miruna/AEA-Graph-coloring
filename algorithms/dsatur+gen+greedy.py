from time import time
from memory_profiler import memory_usage
import random


class Graph:
    def __init__(self, vertices=0):
        self.V = vertices  # Number of vertices
        self.graph = [[] for _ in range(vertices)]  # Adjacency list representation

    def add_vertex(self, v):
        # Expand the graph if needed
        while v >= self.V:
            self.graph.append([])
            self.V += 1

    def add_edge(self, u, v):
        # Ensure vertices exist
        self.add_vertex(u)
        self.add_vertex(v)

        # Add the edge
        if v not in self.graph[u]:
            self.graph[u].append(v)
        if u not in self.graph[v]:
            self.graph[v].append(u)

    def get_neighbors(self, v):
        return self.graph[v]


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
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print(f"Memory Usage: {mem_after - mem_before:.2f} MiB")
    return result


if __name__ == "__main__":

    filename = "../data/DSJC250.1.col"
    print(f"Reading graph from file: {filename}")
    graph = read_graph_from_file(filename)
    print(f"Graph loaded with {graph.V} vertices")

    print("Greedy Coloring Solution:")
    greedy_result = analyze_performance(greedy_coloring, graph)
    print(f"Colors used: {len(set(greedy_result))}")

    print("DSATUR Coloring Solution:")
    dsatur_result = analyze_performance(dsatur_coloring, graph)
    print(f"Colors used: {len(set(dsatur_result))}")

    print("Genetic Algorithm Solution:")
    genetic_result = analyze_performance(genetic_coloring, graph)
    print(f"Colors used: {len(set(genetic_result))}")