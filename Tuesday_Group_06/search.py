import sys
from utils import read_file, format_output
from DFS import DFS
from BFS import BFS
from GBFS import GBFS
from AStar import AStar
from CUS1 import CUS1
from CUS2 import CUS2


METHODS = {
    "DFS": DFS,
    "BFS": BFS,
    "GBFS": GBFS,
    "AS": AStar,
    "ASTAR": AStar,
    "CUS1": CUS1,
    "CUS2": CUS2,
}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python search.py <test_case> <method>")
        print("  test_case: TC02, TC03, ... TC15, or PathFinder-test")
        print("  method: DFS, BFS, GBFS, AS, CUS1, CUS2")
        sys.exit(1)

    filename = sys.argv[1]
    
    if not filename.startswith("Test_Cases") and not filename.startswith("."):
        if not filename.endswith(".txt"):
            filename = f"Test_Cases/{filename}.txt"
        else:
            filename = f"Test_Cases/{filename}"
    
    method = sys.argv[2].upper()

    if method not in METHODS:
        print("Invalid method. Use DFS, BFS, GBFS, AS/ASTAR, CUS1, or CUS2")
        sys.exit(1)

    graph, coordinates, start, goals = read_file(filename)

    if method in ["DFS", "BFS", "CUS1"]:
        result = METHODS[method](graph, start, goals)
    else:
        result = METHODS[method](graph, coordinates, start, goals)

    print(format_output(filename, method, result))
