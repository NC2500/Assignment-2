import sys

def read_file(filename):
    graph = {}
    start = None
    goals = []

    with open(filename, 'r') as file:
        lines = file.readlines()

    section = None

    for line in lines:
        line = line.strip()

        if line == "Edges:":
            section = "edges"
            continue
        elif line == "Origin:":
            section = "origin"
            continue
        elif line == "Destinations:":
            section = "goals"
            continue

        if section == "edges" and line:
            edge, cost = line.split(":")
            a, b = edge.strip()[1:-1].split(",")
            a, b = int(a), int(b)
            cost = int(cost)

            if a not in graph:
                graph[a] = []
            graph[a].append((b, cost))

        elif section == "origin" and line:
            start = int(line)

        elif section == "goals" and line:
            goals = list(map(int, line.split(";")))

    return graph, start, goals

def get_heuristic(node, goals):
    return min(abs(node - goal) for goal in goals)

def greedy_bfs(graph, start, goals):
    visited = set()
    queue = [(start, [start])]

    while queue:
        current_node, path = queue.pop(0)

        if current_node in goals:
            return path

        if current_node not in visited:
            visited.add(current_node)

            neighbors = graph.get(current_node, [])
            neighbors.sort(key=lambda x: get_heuristic(x[0], goals))

            for neighbor, _ in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    return None

if __name__ == "__main__":
    filename = "PathFinder-test.txt"
    graph, start, goals = read_file(filename)
    result = greedy_bfs(graph, start, goals)

    if result:
        print(filename,"Greedy Best First Search", " -> ".join(map(str, result)))
    else:
        print("No path found.")