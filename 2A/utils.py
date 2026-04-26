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

        if not line:
            continue

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

        if section == "nodes":
            node_id, coords = line.split(":")
            x, y = map(int, coords.strip("() ").split(","))
            coordinates[int(node_id)] = (x, y)

        elif section == "edges":
            edge, cost = line.split(":")
            a, b = map(int, edge.strip("() ").split(","))
            cost = int(cost)

            if a not in graph:
                graph[a] = []
            graph[a].append((b, cost))

        elif section == "origin":
            start = int(line)

        elif section == "goals":
            goals.extend(int(x.strip()) for x in line.split(";") if x.strip())

    # ensure all nodes exist in graph dictionary
    for node_id in coordinates:
        if node_id not in graph:
            graph[node_id] = []

    # sort adjacency by ascending node number for stable tie-breaking
    for node_id in graph:
        graph[node_id].sort(key=lambda item: item[0])

    return graph, coordinates, start, goals



def calculate_heuristic(node, goals, coordinates):
    x1, y1 = coordinates[node]
    return min(
        math.sqrt((x1 - coordinates[goal][0]) ** 2 + (y1 - coordinates[goal][1]) ** 2)
        for goal in goals
    )



def format_output(filename, method, result):
    goal, number_of_nodes, path = result

    if goal is None:
        return f"{filename} {method}\nNo goal is reachable\n"

    path_text = " -> ".join(map(str, path))
    return f"{filename} {method}\n{goal} {number_of_nodes}\n{path_text}"
