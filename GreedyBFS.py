import sys
import heapq
import math

def read_file(filename):
    graph = {}
    coordinates = {}
    start = None
    goals = []

    with open(filename, 'r') as file:
        lines = file.readlines()

    section = None

    for line in lines:
        line = line.strip()

        if line == "Nodes:":
            section = "nodes"
            continue
        elif line == "Edges:":
            section = "edges"
            continue
        elif line == "Origin:":
            section = "origin"
            continue
        elif line == "Destinations:":
            section = "goals"
            continue

        if section == "nodes" and line:
            node_id, coords = line.split(":")
            x, y = map(int, coords.strip("() ").split(","))
            coordinates[int(node_id)] = (x, y)

        elif section == "edges" and line:
            edge, cost = line.split(":")
            a, b = map(int, edge.strip("() ").split(","))
            cost = int(cost)

            if a not in graph:
                graph[a] = []
            graph[a].append((b, cost))

        elif section == "origin" and line:
            start = int(line)

        elif section == "goals" and line:
            goals = list(map(int, line.split(";")))

    return graph, coordinates, start, goals

def calculate_heuristic(node, goals, coordinates):
    x1, y1 = coordinates[node]
    return min(math.sqrt((x1 - coordinates[goal][0])**2 + (y1 - coordinates[goal][1])**2) for goal in goals)

def greedy_bfs(graph, coordinates, start, goals):
    frontier = []
    heapq.heappush(frontier, (0, start, 0, [start]))  # (heuristic, node_id, creation_order, path)
    node_counter = 0
    num_nodes_created = 0

    while frontier:
        _, current_node, _, path = heapq.heappop(frontier)

        if current_node in goals:
            return path, num_nodes_created

        neighbors = graph.get(current_node, [])
        for neighbor, _ in neighbors:
            heuristic = calculate_heuristic(neighbor, goals, coordinates)
            node_counter += 1
            heapq.heappush(frontier, (heuristic, neighbor, node_counter, path + [neighbor]))
            num_nodes_created += 1

    return None, num_nodes_created

if __name__ == "__main__":

    filename = sys.argv[1]
    method = "GreedyBFS"


    graph, coordinates, start, goals = read_file(filename)
    result, num_nodes_created = greedy_bfs(graph, coordinates, start, goals)

    if result:
        print(f"{filename} {method} {result[-1]} {num_nodes_created} {' -> '.join(map(str, result))}")
    else:
        print("No path found.")