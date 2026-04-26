import py_compile, sys, os

files = [
    "data_processor.py",
    "traffic_converter.py",
    "graph_builder.py",
    "ml_models.py",
    "ml_sklearn.py",
    "traffic_predictor.py",
    "tbrgs.py",
    "config.py",
    "cli.py",
    "gui.py",
    "train_models.py",
    "search.py",
    "DFS.py",
    "BFS.py",
    "GBFS.py",
    "AStar.py",
    "CUS1.py",
    "CUS2.py",
    "utils.py"
]

errors = []
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"OK: {f}")
    except Exception as e:
        print(f"ERROR in {f}: {e}")
        errors.append((f, e))

if errors:
    print("\nCompilation errors found!")
    sys.exit(1)
else:
    print("\nAll files compiled successfully!")
