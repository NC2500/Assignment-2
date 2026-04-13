import heapq
from utils import calculate_heuristic


def CUS2(graph, coordinates, start, goals):
    # informed search to find shortest path with least moves
    frontier = []
    creation_order = 0
    number_of_nodes = 1

    start_h = calculate_heuristic(start, goals, coordinates)
    heapq.heappush(frontier, (start_h, 0, start, creation_order, [start]))

    best_moves = {start: 0}

    while frontier:
        _, moves, current_node, _, path = heapq.heappop(frontier)

        if current_node in goals:
            return current_node, number_of_nodes, path

        if moves > best_moves.get(current_node, float('inf')):
            continue

        neighbors = graph.get(current_node, [])
        for neighbor, _ in neighbors:
            new_moves = moves + 1

            if new_moves < best_moves.get(neighbor, float('inf')):
                best_moves[neighbor] = new_moves
                creation_order += 1
                heuristic = calculate_heuristic(neighbor, goals, coordinates)
                score = new_moves + heuristic
                heapq.heappush(frontier, (score, new_moves, neighbor, creation_order, path + [neighbor]))
                number_of_nodes += 1

    return None, number_of_nodes, []
