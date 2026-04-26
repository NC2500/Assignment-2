import subprocess

methods = ["DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2"]

tests = [
    "PathFinder-test.txt", "TC02", "TC03", "TC04", "TC05",
    "TC06", "TC07", "TC08", "TC09", "TC10",
    "TC11", "TC12", "TC13", "TC14", "TC15"
]

for test in tests:
    print("\n" + "=" * 80)
    print(f"TEST CASE: {test}")
    print("=" * 80)
    print(f"{'Method':<10} {'Goal':<10} {'Nodes':<10} {'Path'}")
    print("-" * 80)

    for method in methods:
        try:
            result = subprocess.run(
                ["python", "search.py", test, method],
                capture_output=True,
                text=True
            )

            output = result.stdout.strip().splitlines()

            if "No goal is reachable" in result.stdout:
                print(f"{method:<10} {'-':<10} {'-':<10} No goal is reachable")

            elif len(output) >= 3:
                goal_nodes = output[1].split()
                goal = goal_nodes[0]
                nodes = goal_nodes[1]
                path = output[2]

                print(f"{method:<10} {goal:<10} {nodes:<10} {path}")

            else:
                print(f"{method:<10} ERROR")

        except Exception as e:
            print(f"{method:<10} ERROR: {e}")

print("\nAll tests completed.")