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
graph, start, goals = read_file("PathFinder-test.txt")

print("Graph:", graph)
print("Start:", start)
print("Goals:", goals)

#DFS & BFS
from collections import deque

def read_graph(filename):
    nodes, edges, origin, destinations = {}, {}, None, []
    section = None
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line: continue
            if "Nodes:" in line: section = "nodes"
            elif "Edges:" in line: section = "edges"
            elif "Origin:" in line: section = "origin"
            elif "Destinations:" in line: section = "destinations"
            elif section == "nodes":
                parts = line.split(":")
                nodes[int(parts[0])] = parts[1].strip()
            elif section == "edges":
                # Format: (a,b): cost -> only take (a,b)
                edge_nodes = line.split(":")[0].strip("() ")
                a, b = map(int, edge_nodes.split(","))
                if a not in edges: edges[a] = []
                edges[a].append(b)
            elif section == "origin":
                origin = int(line)
            elif section == "destinations":
                # Handle semicolon-separated destinations across lines
                destinations.extend([int(d.strip()) for d in line.split(";") if d.strip()])
    return edges, origin, set(destinations)

def dfs(graph, start, goals):
    stack = [(start, [start])]
    visited = set()
    nodes_created = 0
    while stack:
        node, path = stack.pop()
        nodes_created += 1
        if node in goals: 
            return node, nodes_created, path
        if node not in visited:
            visited.add(node)
            # Sort reverse so that the smallest ID is popped first from the stack
            for neighbor in sorted(graph.get(node, []), reverse=True):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    return None, nodes_created, []

def bfs(graph, start, goals):
    queue = deque([(start, [start])])
    visited = {start}
    nodes_created = 0
    while queue:
        node, path = queue.popleft()
        nodes_created += 1
        if node in goals: 
            return node, nodes_created, path
        # Sort ascending to visit smaller IDs first
        for neighbor in sorted(graph.get(node, [])):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None, nodes_created, []

# --- Execution ---
filename = "PathFinder-test.txt"
graph, origin, goals = read_graph(filename)

# Run DFS
goal_dfs, created_dfs, path_dfs = dfs(graph, origin, goals)
print(f"DFS: Goal {goal_dfs}, Nodes Created {created_dfs}\nPath: {' '.join(map(str, path_dfs))}\n")

# Run BFS
goal_bfs, created_bfs, path_bfs = bfs(graph, origin, goals)
print(f"BFS: Goal {goal_bfs}, Nodes Created {created_bfs}\nPath: {' '.join(map(str, path_bfs))}")