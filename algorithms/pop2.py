import random
import time
import os
from collections import defaultdict


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


def read_graph_from_file(filename):
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
                        v1 = int(parts[1])
                        v2 = int(parts[2])
                        if v1 != v2:
                            graph.add_edge(v1, v2)
        graph.finalize()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    return graph


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


def greedy_dsatur_coloring(graph, max_colors, randomize=False):
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
            candidates = sorted(candidates, key=lambda x: (len(colored_neighbors[x]), degrees[x], random.random()), reverse=True)[:3]
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


def aggressive_pop2_algorithm(graph, target_colors=None, max_iterations=5000, max_time=300, verbose=False):
    if graph.num_vertices == 0:
        return {}, 0
    density = graph.get_density()
    max_degree = max(graph.get_degree(v) for v in graph.vertices)
    if target_colors is None:
        start_colors = len(set(greedy_dsatur_coloring(graph, max_degree + 1).values()))
    else:
        start_colors = target_colors
    if verbose:
        print(f"Starting search from {start_colors} colors (density: {density:.3f}, max degree: {max_degree})")
    best_num_colors = start_colors
    best_coloring = None
    best_conflicts = float('inf')
    start_time = time.time()
    for num_colors in range(start_colors, 3, -1):
        if time.time() - start_time > max_time:
            break
        if verbose and num_colors % 10 == 0:
            print(f"Trying {num_colors} colors... (best so far: {best_num_colors})")
        best_initial_conflicts = float('inf')
        best_initial_coloring = None
        for attempt in range(6):
            if attempt % 3 == 0:
                coloring = greedy_dsatur_coloring(graph, num_colors, randomize=False)
            elif attempt % 3 == 1:
                coloring = greedy_dsatur_coloring(graph, num_colors, randomize=True)
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
            if verbose:
                print(f"Perfect {num_colors}-coloring found immediately!")
            continue
        improved_coloring, final_conflicts = intensive_tabu_search_optimized(
            graph, best_initial_coloring, num_colors,
            max_iterations=max_iterations,
            max_time=max_time - (time.time() - start_time),
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
            if verbose:
                print(f"Perfect {num_colors}-coloring found! Continuing to search for better...")
            continue
        else:
            if verbose:
                print(f"Could not find valid {num_colors}-coloring (conflicts: {final_conflicts})")
            break
    if best_coloring is None:
        return initial_coloring(graph, start_colors), start_colors
    return best_coloring, best_num_colors


def intensive_tabu_search_optimized(graph, coloring, num_colors, max_iterations=5000, max_time=60, verbose=False):
    current_conflicts = evaluate_conflicts_fast(graph, coloring)
    best_coloring = coloring.copy()
    best_conflicts = current_conflicts
    base_tabu_tenure = max(5, min(15, graph.num_vertices // 50))
    tabu_list = {}
    conflict_count = {}
    for v in graph.vertices:
        conflict_count[v] = get_vertex_conflicts_fast(graph, coloring, v)
    start_time = time.time()
    non_improving = 0
    stagnation_limit = 200
    restart_limit = 500
    temperature = 0.7
    temp_decay = 0.998
    conflict_vertices = []
    for iteration in range(max_iterations):
        if time.time() - start_time > max_time:
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
                if verbose and current_conflicts == 0:
                    print(f"  Perfect coloring found at iteration {iteration}!")
            else:
                non_improving += 1
        else:
            non_improving += 1
        if non_improving > restart_limit:
            if verbose:
                print(f"  Restarting from best solution (conflicts: {best_conflicts})")
            coloring = best_coloring.copy()
            for v in graph.vertices:
                conflict_count[v] = get_vertex_conflicts_fast(graph, coloring, v)
            current_conflicts = best_conflicts
            non_improving = 0
            temperature = 0.7
            tabu_list.clear()
        temperature *= temp_decay
        temperature = max(0.05, temperature)
        if verbose and iteration % 2000 == 0 and iteration > 0:
            print(f"  Iteration {iteration}: current conflicts = {current_conflicts}, best = {best_conflicts}")
    return best_coloring, best_conflicts


def main():
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    input_file = os.path.join(script_dir, "homer.col")
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found!")
        return
    print("Loading graph...")
    graph = read_graph_from_file(input_file)
    if graph is None:
        print("Failed to load graph!")
        return
    print(f"Graph loaded successfully:")
    print(f"  Vertices: {graph.num_vertices}")
    print(f"  Edges: {graph.num_edges}")
    print(f"  Density: {graph.get_density():.4f}")
    print(f"  Max degree: {max(graph.get_degree(v) for v in graph.vertices)}")
    print("Running OPTIMIZED aggressive search...")
    # Greedy initial coloring number of colors
    initial_coloring = greedy_dsatur_coloring(graph, max(graph.get_degree(v) for v in graph.vertices) + 1)
    initial_num_colors = len(set(initial_coloring.values()))
    print(f"Greedy {initial_num_colors} colors.")
    start_time = time.time()
    best_coloring, best_num_colors = aggressive_pop2_algorithm(
        graph,
        max_iterations=10000,
        max_time=600,
        verbose=True
    )
    end_time = time.time()
    runtime = end_time - start_time
    conflicts = evaluate_conflicts_fast(graph, best_coloring)
    print(f"\n{'=' * 60}")
    print(f"FINAL RESULTS:")
    print(f"{'=' * 60}")
    print(f"Colors used: {best_num_colors}")
    print(f"Conflicts: {conflicts}")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Solution quality: {'VALID' if conflicts == 0 else 'INVALID'}")


if __name__ == "__main__":
    main()
