def DFS(graph, start, goals):
    stack = [(start, [start])]
    visited = set()
    number_of_nodes = 1

    while stack:
        current_node, path = stack.pop()

        if current_node in goals:
            return current_node, number_of_nodes, path

        if current_node not in visited:
            visited.add(current_node)

            neighbors = graph.get(current_node, [])
            # reverse so the smaller node is expanded first when popped from stack
            for neighbor, _ in reversed(neighbors):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
                    number_of_nodes += 1

    return None, number_of_nodes, []
