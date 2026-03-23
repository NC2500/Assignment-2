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


# TEST
graph, start, goals = read_file("PathFinder-test-1.txt")

print("Graph:", graph)
print("Start:", start)
print("Goals:", goals)