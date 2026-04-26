import heapq
from utils import calculate_heuristic


def GBFS(graph, coordinates, start, goals):
    frontier = []
    creation_order = 0
    number_of_nodes = 1
    visited = set()

    heapq.heappush(frontier, (calculate_heuristic(start, goals, coordinates), start, creation_order, [start]))

    while frontier:
        _, current_node, _, path = heapq.heappop(frontier)

        if current_node in goals:
            return current_node, number_of_nodes, path

        if current_node in visited:
            continue

        visited.add(current_node)

        neighbors = graph.get(current_node, [])
        for neighbor, _ in neighbors:
            if neighbor not in visited:
                creation_order += 1
                heuristic = calculate_heuristic(neighbor, goals, coordinates)
                heapq.heappush(frontier, (heuristic, neighbor, creation_order, path + [neighbor]))
                number_of_nodes += 1

    return None, number_of_nodes, []
