from ortools.linear_solver import pywraplp
import time
import sys


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
                parts = line.split()
                continue
            elif line.startswith('e'):
                parts = line.split()
                v1 = int(parts[1])
                v2 = int(parts[2])
                graph.add_edge(v1, v2)
    return graph


def greedy_coloring(graph):
    colors = {}
    for v in sorted(graph.vertices):
        used_colors = {colors.get(neighbor) for neighbor in graph.get_neighbors(v) if neighbor in colors}
        color = 0
        while color in used_colors:
            color += 1
        colors[v] = color
    return colors, max(colors.values()) + 1 if colors else 0


def color_graph_linear(graph, max_colors=None, time_limit=300):
    _, greedy_colors = greedy_coloring(graph)

    if max_colors is None:
        max_colors = greedy_colors
    else:
        max_colors = min(max_colors, greedy_colors)

    print(f"Încercăm să colorăm graful cu maxim {max_colors} culori (estimat greedy)")

    solver = pywraplp.Solver.CreateSolver('SCIP')

    if not solver:
        return None, 0, 0

    x = {}
    for v in graph.vertices:
        x[v] = {}
        for c in range(max_colors):
            x[v][c] = solver.IntVar(0, 1, f'x_{v}_{c}')

    y = {}
    for c in range(max_colors):
        y[c] = solver.IntVar(0, 1, f'y_{c}')

    for v in graph.vertices:
        solver.Add(sum(x[v][c] for c in range(max_colors)) == 1)

    for v in graph.vertices:
        for neighbor in graph.get_neighbors(v):
            if v < neighbor:
                for c in range(max_colors):
                    solver.Add(x[v][c] + x[neighbor][c] <= 1)

    for c in range(max_colors):
        for v in graph.vertices:
            solver.Add(x[v][c] <= y[c])

    for c in range(1, max_colors):
        solver.Add(y[c] <= y[c - 1])

    solver.Minimize(sum(y[c] for c in range(max_colors)))

    start_time = time.time()
    status = solver.Solve()
    solve_time = time.time() - start_time

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        solution = {}
        num_colors_used = sum(y[c].solution_value() for c in range(max_colors))

        for v in graph.vertices:
            for c in range(max_colors):
                if x[v][c].solution_value() > 0.5:
                    solution[v] = c
                    break

        return solution, int(num_colors_used), solve_time
    else:
        print(f"Status solver: {status}")
        return None, 0, solve_time


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../data/DSJC250.1.col"

    print(f"Citim graful din {filename}")
    graph = read_graph_from_file(filename)

    print(f"Graf încărcat: {graph.num_vertices} vârfuri, {graph.num_edges} muchii")

    greedy_solution, greedy_colors = greedy_coloring(graph)
    print(f"Colorare greedy: {greedy_colors} culori")

    time_limit = 60
    solution, num_colors, solve_time = color_graph_linear(graph, greedy_colors, time_limit)

    if solution:
        print(f"Soluție găsită utilizând {num_colors} culori în {solve_time:.2f} secunde")

        is_valid = True
        for v in graph.vertices:
            for neighbor in graph.get_neighbors(v):
                if solution[v] == solution[neighbor]:
                    print(f"Colorare invalidă: vârfurile {v} și {neighbor} au aceeași culoare")
                    is_valid = False
                    break
            if not is_valid:
                break

        if is_valid:
            print("Colorarea este validă")

            color_counts = {}
            for v in solution:
                color = solution[v]
                if color not in color_counts:
                    color_counts[color] = 0
                color_counts[color] += 1

            print("Distribuția culorilor:")
            for color in sorted(color_counts.keys()):
                print(f"  Culoarea {color}: {color_counts[color]} vârfuri")
    else:
        print(f"Nu s-a găsit nicio soluție optimală în {solve_time:.2f} secunde")
        print("Folosim rezultatul colorării greedy ca soluție de rezervă")

        is_valid = True
        for v in graph.vertices:
            for neighbor in graph.get_neighbors(v):
                if greedy_solution[v] == greedy_solution[neighbor]:
                    print(f"Colorare greedy invalidă: vârfurile {v} și {neighbor} au aceeași culoare")
                    is_valid = False
                    break
            if not is_valid:
                break

        if is_valid:
            print("Colorarea greedy este validă")


if __name__ == "__main__":
    main()
