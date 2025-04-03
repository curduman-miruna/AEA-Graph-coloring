import random
import time
import os

class Graph:
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


def read_graph_from_file(filename):
    graph = Graph()

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p'):
                continue
            elif line.startswith('e'):
                parts = line.split()
                v1 = int(parts[1])
                v2 = int(parts[2])
                graph.add_edge(v1, v2)

    return graph


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


def pop2_algorithm(graph, max_iterations=3000, max_time=6000, tabu_tenure=10, verbose=False):
    """
    POP2 algorithm for graph coloring with tabu search and dynamic temperature
    Optimized for performance
    """
    density = 2 * graph.num_edges / (graph.num_vertices * (graph.num_vertices - 1))
    initial_colors = min(len(graph.vertices), max(3, int(1.5 * graph.num_vertices * density)))
    best_num_colors = initial_colors
    best_coloring = initial_coloring(graph, best_num_colors)
    best_conflicts = evaluate_conflicts(graph, best_coloring)

    vertex_degrees = {v: len(graph.get_neighbors(v)) for v in graph.vertices}

    start_time = time.time()

    for num_colors in range(best_num_colors, 0, -1):

        coloring = greedy_dsatur_coloring(graph, num_colors)
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
            if time.time() - start_time > max_time:
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


def greedy_dsatur_coloring(graph, max_colors):
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


def main():
    script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    input_file = os.path.join(script_dir, "DSJC250.9.col")

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found!")
        return

    graph = read_graph_from_file(input_file)
    print(f"Graph loaded with {graph.num_vertices} vertices and {graph.num_edges} edges.")

    start_time = time.time()
    best_coloring, best_num_colors = pop2_algorithm(
        graph,
        max_iterations=2000,
        max_time=300,
        tabu_tenure=10,
        verbose=True
    )
    end_time = time.time()

    print(f"\nBest coloring found uses {best_num_colors} colors.")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    conflicts = evaluate_conflicts(graph, best_coloring)
    print(f"Number of conflicts: {conflicts}")

    output_file = os.path.join(script_dir, "coloring_result.txt")
    with open(output_file, 'w') as f:
        f.write(f"# Coloring with {best_num_colors} colors\n")
        for vertex in sorted(graph.vertices):
            f.write(f"{vertex} {best_coloring[vertex]}\n")

    print(f"Coloring saved to {output_file}")


if __name__ == "__main__":
    main()