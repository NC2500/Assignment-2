"""
Iterative Deepening Search and Beam Search Implementation
Using nodes from PathFinder-test-1.txt
"""

from collections import deque
import heapq


class Graph:
    """Graph representation for pathfinding algorithms"""
    
    def __init__(self):
        self.nodes = {}  # node_id: (x, y)
        self.edges = {}  # node_id: [(neighbor, weight)]
    
    def add_node(self, node_id, x, y):
        """Add a node with coordinates"""
        self.nodes[node_id] = (x, y)
        if node_id not in self.edges:
            self.edges[node_id] = []
    
    def add_edge(self, node1, node2, weight):
        """Add an undirected edge between two nodes"""
        if node1 not in self.edges:
            self.edges[node1] = []
        if node2 not in self.edges:
            self.edges[node2] = []
        self.edges[node1].append((node2, weight))
        self.edges[node2].append((node1, weight))
    
    def get_neighbors(self, node_id):
        """Get all neighbors of a node with their weights"""
        return self.edges.get(node_id, [])
    
    def euclidean_distance(self, node1, node2):
        """Calculate Euclidean distance between two nodes"""
        x1, y1 = self.nodes[node1]
        x2, y2 = self.nodes[node2]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def parse_pathfinder_file(filename):
    """
    Parse the PathFinder-test-1.txt file
    Returns: (graph, origin, destinations)
    """
    graph = Graph()
    origin = None
    destinations = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
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
        
        if section == "nodes":
            # Parse: "1: (4,1)"
            parts = line.split(":")
            node_id = int(parts[0].strip())
            coords = parts[1].strip().strip("()").split(",")
            x, y = int(coords[0]), int(coords[1])
            graph.add_node(node_id, x, y)
        
        elif section == "edges":
            # Parse: "(2,1): 4"
            parts = line.split(":")
            nodes = parts[0].strip().strip("()").split(",")
            node1, node2 = int(nodes[0]), int(nodes[1])
            weight = int(parts[1].strip())
            graph.add_edge(node1, node2, weight)
        
        elif section == "origin":
            origin = int(line.strip())
        
        elif section == "destinations":
            # Parse: "5; 4"
            dest_parts = line.split(";")
            for dest in dest_parts:
                dest = dest.strip()
                if dest:
                    destinations.append(int(dest))
    
    return graph, origin, destinations


def iterative_deepening_search(graph, start, goal, max_depth=50):
    """
    Iterative Deepening Search (IDS) algorithm
    
    Args:
        graph: The graph to search
        start: Starting node ID
        goal: Goal node ID
        max_depth: Maximum depth to search
    
    Returns:
        Tuple of (path, cost) if found, None otherwise
    """
    
    def depth_limited_search(current, goal, depth, path, visited, cost):
        """Depth-Limited Search helper function"""
        if current == goal:
            return path[:], cost
        
        if depth == 0:
            return None
        
        visited.add(current)
        
        for neighbor, weight in graph.get_neighbors(current):
            if neighbor not in visited:
                path.append(neighbor)
                result = depth_limited_search(neighbor, goal, depth - 1, path, visited, cost + weight)
                if result is not None:
                    return result
                path.pop()
        
        visited.remove(current)
        return None
    
    # Iteratively increase depth limit
    for depth in range(max_depth + 1):
        visited = set()
        result = depth_limited_search(start, goal, depth, [start], visited, 0)
        if result is not None:
            return result
    
    return None


def beam_search(graph, start, goal, beam_width=2):
    """
    Beam Search algorithm
    
    Args:
        graph: The graph to search
        start: Starting node ID
        goal: Goal node ID
        beam_width: Number of paths to keep at each level
    
    Returns:
        Tuple of (path, cost) if found, None otherwise
    """
    
    # Priority queue: (heuristic, cost, path)
    # Using heuristic (Euclidean distance to goal) as primary sort key
    initial_heuristic = graph.euclidean_distance(start, goal)
    beam = [(initial_heuristic, 0, [start])]
    
    visited = set()
    
    while beam:
        next_beam = []
        
        for _, cost, path in beam:
            current = path[-1]
            
            if current == goal:
                return path, cost
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Expand current node
            for neighbor, weight in graph.get_neighbors(current):
                if neighbor not in visited:
                    new_cost = cost + weight
                    new_path = path + [neighbor]
                    heuristic = graph.euclidean_distance(neighbor, goal)
                    heapq.heappush(next_beam, (heuristic, new_cost, new_path))
        
        # Keep only the best beam_width paths
        beam = []
        for _ in range(min(beam_width, len(next_beam))):
            if next_beam:
                beam.append(heapq.heappop(next_beam))
        
        if not beam:
            break
    
    return None


def print_graph_info(graph, origin, destinations):
    """Print graph information"""
    print("=" * 60)
    print("GRAPH INFORMATION")
    print("=" * 60)
    print("\nNodes ({}):".format(len(graph.nodes)))
    for node_id in sorted(graph.nodes.keys()):
        x, y = graph.nodes[node_id]
        print("  Node {}: ({}, {})".format(node_id, x, y))
    
    print("\nEdges:")
    for node_id in sorted(graph.edges.keys()):
        neighbors = graph.edges[node_id]
        for neighbor, weight in neighbors:
            if node_id < neighbor:  # Print each edge only once
                print("  ({}, {}): {}".format(node_id, neighbor, weight))
    
    print("\nOrigin: Node {}".format(origin))
    print("Destinations: {}".format(destinations))
    print("=" * 60)


def main():
    """Main function to test both search algorithms"""
    
    # Parse the PathFinder-test-1.txt file
    filename = "PathFinder-test-1.txt"
    graph, origin, destinations = parse_pathfinder_file(filename)
    
    # Print graph information
    print_graph_info(graph, origin, destinations)
    
    # Test Iterative Deepening Search
    print("\n" + "=" * 60)
    print("ITERATIVE DEEPENING SEARCH (IDS)")
    print("=" * 60)
    
    for destination in destinations:
        print("\nSearching path from Node {} to Node {}:".format(origin, destination))
        result = iterative_deepening_search(graph, origin, destination)
        
        if result:
            path, cost = result
            print("  Path found: {}".format(" -> ".join(map(str, path))))
            print("  Total cost: {}".format(cost))
            
            # Print detailed path with coordinates
            print("  Path with coordinates:")
            for i, node in enumerate(path):
                x, y = graph.nodes[node]
                print("      {}. Node {}: ({}, {})".format(i+1, node, x, y))
        else:
            print("  No path found from Node {} to Node {}".format(origin, destination))
    
    # Test Beam Search
    print("\n" + "=" * 60)
    print("BEAM SEARCH")
    print("=" * 60)
    
    # Test with different beam widths
    for beam_width in [1, 2, 3]:
        print("\n--- Beam Width: {} ---".format(beam_width))
        
        for destination in destinations:
            print("\nSearching path from Node {} to Node {}:".format(origin, destination))
            result = beam_search(graph, origin, destination, beam_width=beam_width)
            
            if result:
                path, cost = result
                print("  Path found: {}".format(" -> ".join(map(str, path))))
                print("  Total cost: {}".format(cost))
                
                # Print detailed path with coordinates
                print("  Path with coordinates:")
                for i, node in enumerate(path):
                    x, y = graph.nodes[node]
                    print("      {}. Node {}: ({}, {})".format(i+1, node, x, y))
            else:
                print("  No path found from Node {} to Node {}".format(origin, destination))
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    
    print("\nOrigin: Node {}".format(origin))
    print("Destinations: {}".format(destinations))
    
    for destination in destinations:
        print("\n--- Destination: Node {} ---".format(destination))
        
        # IDS result
        ids_result = iterative_deepening_search(graph, origin, destination)
        if ids_result:
            ids_path, ids_cost = ids_result
            print("IDS:  Path: {}, Cost: {}".format(" -> ".join(map(str, ids_path)), ids_cost))
        else:
            print("IDS:  No path found")
        
        # Beam Search results
        for beam_width in [1, 2, 3]:
            beam_result = beam_search(graph, origin, destination, beam_width=beam_width)
            if beam_result:
                beam_path, beam_cost = beam_result
                print("Beam (w={}): Path: {}, Cost: {}".format(beam_width, " -> ".join(map(str, beam_path)), beam_cost))
            else:
                print("Beam (w={}): No path found".format(beam_width))


if __name__ == "__main__":
    main()
