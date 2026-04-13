from collections import deque
import os

# -------------------------------
# READ GRAPH
# -------------------------------
def read_graph(filename):
    nodes = {}
    edges = {}
    origin = None
    destinations = []
    section = None

    # Check if file exists before opening
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' was not found in {os.getcwd()}")

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # Identify Sections
            if line.startswith("Nodes:"):
                section = "nodes"
                continue
            elif line.startswith("Edges:"):
                section = "edges"
                continue
            elif line.startswith("Origin:"):
                section = "origin"
                continue
            elif line.startswith("Destinations:"):
                section = "destinations"
                continue

            # Parsing Logic
            if section == "nodes":
                node_id, coord = line.split(":")
                node_id = int(node_id.strip())
                coord = coord.strip().strip("() ")
                x, y = map(int, coord.split(","))
                nodes[node_id] = (x, y)

            elif section == "edges":
                # Matches format (2,1): 4 [cite: 1]
                edge_part, _ = line.split(":")
                edge_part = edge_part.strip().strip("() ")
                a, b = map(int, edge_part.split(","))
                if a not in edges:
                    edges[a] = []
                edges[a].append(b)

            elif section == "origin":
                origin = int(line)

            elif section == "destinations":
                # Robustly handles '5;' and '4' on separate lines 
                vals = [int(x.strip()) for x in line.replace(";", " ").split() if x.strip()]
                destinations.extend(vals)

    return edges, origin, destinations

# -------------------------------
# SEARCH ALGORITHMS
# -------------------------------
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
            # Reverse sort to explore smaller IDs first in a stack
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

        for neighbor in sorted(graph.get(node, [])):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None, nodes_created, []

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    # Ensure this string exactly matches your filename on your Desktop
    filename = "PathFinder-test.txt" 
    
    try:
        graph, origin, destinations = read_graph(filename)
        goals = set(destinations)

        print(f"File loaded successfully.")
        print(f"Origin: {origin} | Destinations: {destinations}")

        # Run DFS
        res_node, res_count, res_path = dfs(graph, origin, goals)
        print(f"\nDFS Result:\nGoal: {res_node} | Nodes Created: {res_count}")
        print(f"Path: {' '.join(map(str, res_path))}")

        # Run BFS
        res_node, res_count, res_path = bfs(graph, origin, goals)
        print(f"\nBFS Result:\nGoal: {res_node} | Nodes Created: {res_count}")
        print(f"Path: {' '.join(map(str, res_path))}")

    except Exception as e:
        print(f"Error: {e}")