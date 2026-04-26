def _depth_limited_search(graph, current_node, goals, limit, path, visited, counter):
    if current_node in goals:
        return current_node, counter[0], path

    if limit == 0:
        return None, counter[0], []

    visited.add(current_node)

    neighbors = graph.get(current_node, [])
    for neighbor, _ in neighbors:
        if neighbor not in visited:
            counter[0] += 1
            result_goal, result_nodes, result_path = _depth_limited_search(
                graph,
                neighbor,
                goals,
                limit - 1,
                path + [neighbor],
                visited,
                counter,
            )
            if result_goal is not None:
                return result_goal, result_nodes, result_path

    visited.remove(current_node)
    return None, counter[0], []



def CUS1(graph, start, goals):
    # Iterative Deepening Search
    max_depth = len(graph) + 5
    total_nodes_created = 1

    if start in goals:
        return start, total_nodes_created, [start]

    for depth_limit in range(max_depth + 1):
        counter = [total_nodes_created]
        visited = set()
        goal, nodes, path = _depth_limited_search(
            graph, start, goals, depth_limit, [start], visited, counter
        )
        total_nodes_created = nodes
        if goal is not None:
            return goal, total_nodes_created, path

    return None, total_nodes_created, []
