import heapq
from utils import calculate_heuristic


def AStar(graph, coordinates, start, goals):
    frontier = []
    creation_order = 0
    number_of_nodes = 1

    start_h = calculate_heuristic(start, goals, coordinates)
    heapq.heappush(frontier, (start_h, start, creation_order, 0, [start]))

    best_cost = {start: 0}

    while frontier:
        _, current_node, _, current_cost, path = heapq.heappop(frontier)

        if current_node in goals:
            return current_node, number_of_nodes, path

        if current_cost > best_cost.get(current_node, float('inf')):
            continue

        neighbors = graph.get(current_node, [])
        for neighbor, edge_cost in neighbors:
            new_cost = current_cost + edge_cost

            if new_cost < best_cost.get(neighbor, float('inf')):
                best_cost[neighbor] = new_cost
                creation_order += 1
                heuristic = calculate_heuristic(neighbor, goals, coordinates)
                f_cost = new_cost + heuristic
                heapq.heappush(frontier, (f_cost, neighbor, creation_order, new_cost, path + [neighbor]))
                number_of_nodes += 1

    return None, number_of_nodes, []
