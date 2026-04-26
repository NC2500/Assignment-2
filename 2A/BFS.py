from collections import deque


def BFS(graph, start, goals):
    queue = deque([(start, [start])])
    visited = {start}
    number_of_nodes = 1

    while queue:
        current_node, path = queue.popleft()

        if current_node in goals:
            return current_node, number_of_nodes, path

        neighbors = graph.get(current_node, [])
        for neighbor, _ in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
                number_of_nodes += 1

    return None, number_of_nodes, []
