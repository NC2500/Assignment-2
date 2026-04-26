# TBRGS - Traffic-Based Route Guidance System

**COS30019 - Introduction to AI - Assignment 2 Part B**

A complete AI-powered traffic prediction and route guidance system for the Boroondara area. Uses deep learning (LSTM, GRU, CNN-LSTM, Transformer, MLP) to predict traffic flow and integrates with graph search algorithms (A*, BFS, DFS, GBFS, CUS1, CUS2) to find optimal routes based on predicted travel time.

## Features

- **Traffic Flow Prediction**: Multiple deep learning architectures:
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
  - CNN-LSTM hybrid
  - Transformer
  - MLP baseline

- **Travel Time Estimation**: Converts traffic flow to travel time using fundamental traffic flow diagrams (provided in assignment spec).

- **Graph-based Search**: Integration of Part A search algorithms with dynamic edge weights based on ML predictions.

- **User Interfaces**:
  - Command-line interface (`cli.py`)
  - Graphical user interface (`gui.py`)

- **Configuration**: Centralized config in `config.py`

## Project Structure

```
.
├── data_processor.py       # Load and process SCATS Excel data into time series
├── graph_builder.py        # Build Boroondara road network graph from SCATS metadata
├── traffic_converter.py    # Convert traffic flow to travel time (fundamental diagram)
├── ml_models.py            # PyTorch model definitions (LSTM, GRU, CNN-LSTM, Transformer, MLP)
├── traffic_predictor.py    # Model training, inference, and evaluation
├── tbrgs.py                # Main system: integrates prediction, conversion, and search
├── config.py               # Configuration dataclasses
├── cli.py                  # Command-line interface
├── gui.py                  # Tkinter GUI
├── utils_ml.py             # Utility functions (SCATS ID formatting, haversine)
├── search.py               # Part A search integration (uses DFS, BFS, GBFS, AStar, CUS1, CUS2 modules)
├── processed_data/         # Directory containing processed CSVs from data_processor
│   ├── scats_metadata.csv
│   └── scats_*_timeseries.csv
├── models/                 # Trained model weights (created after training)
└── results/                # Experiment results (metrics, plots)
```

## Prerequisites

- Python 3.9+
- Required packages:
  - numpy>=1.21
  - pandas>=1.3
  - torch>=2.0 (CPU or CUDA)
  - scikit-learn>=1.0
  - matplotlib>=3.4

Install dependencies:

```bash
pip install numpy pandas torch scikit-learn matplotlib
```

## Quick Start

### 1. Process raw SCATS data (if needed)

The `processed_data/` folder already contains CSVs generated from `data_processor.py`. If you need to re-process:

```bash
python data_processor.py
```

This reads `Scats Data October 2006.xls` and generates time series CSVs.

### 2. Command-line usage

List all available SCATS sites:

```bash
python cli.py list
```

Find routes between two sites:

```bash
python cli.py route 2000 3002 -k 5 -m lstm -a astar
```

Options:
- `-k`, `--k`: Number of routes to return (default 5)
- `-m`, `--model`: ML model for traffic prediction (`lstm`, `gru`, `cnnlstm`, `transformer`, `mlp`); default `lstm`
- `-a`, `--algorithm`: Search algorithm (`astar`, `bfs`, `dfs`, `gbfs`, `cus1`, `cus2`); default `astar`

Predict traffic for a specific site:

```bash
python cli.py predict 2000 -m gru -H 4
```

### 3. Graphical Interface

Launch the GUI:

```bash
python cli.py gui
# or
python gui.py
```

The GUI provides:
- Dropdowns to select origin/destination (populated automatically)
- Model type selection
- Algorithm selection
- Spinbox for number of routes
- List of routes with travel time and node count
- Detailed route breakdown and bar chart

### 4. Training models

Train a model using the full traffic dataset:

```bash
python cli.py train -m lstm -e 50 -b 32 -s 24 -o models/
```

Parameters:
- `-e`, `--epochs`: Training epochs (default 50)
- `-b`, `--batch-size`: Batch size (default 32)
- `-s`, `--seq-len`: Sequence length (past intervals; default 24 = 6 hours)
- `-l`, `--lr`: Learning rate (default 0.001)
- `-v`, `--val-split`: Validation split (default 0.2)
- `-o`, `--output-dir`: Directory to save model (default `models/`)

Trained models are saved as `<model_type>_best.pth` (or specified name).

### 5. Using the library programmatically

```python
from tbrgs import TBRGS

# Initialize system (loads graph and data, creates model)
tbrgs = TBRGS(model_type='lstm', model_path='models/lstm_best.pth')

# Find top-k routes
routes = tbrgs.find_top_k_paths(origin='2000', destination='3002', k=5, method='astar')

# Display results
for i, route in enumerate(routes):
    print(f"Route {i+1}: {route.travel_time_minutes:.1f} minutes")
    print(f"  Path: {' -> '.join(route.path)}")
```

## System Details

### Data Processing

The `data_processor.py` module reads the provided VicRoads SCATS Excel file and extracts 15-minute interval traffic counts for each SCATS site. It outputs per-site time series CSVs and a metadata file (`scats_metadata.csv`) containing location info.

### Graph Construction

`graph_builder.py` creates a road network graph from SCATS metadata. Nodes are intersections identified by SCATS IDs. Edges are inferred by:
- **Road name matching**: connecting sites sharing the same main road
- **Nearest-neighbor**: connecting each site to its k nearest neighbors within a distance threshold

The hybrid approach yields a well-connected graph of ~40 nodes and ~76 edges for the Boroondara area.

### Traffic Flow to Travel Time

The `TrafficFlowConverter` implements the fundamental diagram formula from the provided PDF:

```
speed = ? (solved from quadratic: -1.4648375*speed^2 + 93.75*speed = flow)
travel_time = distance / speed + intersection_delay
```

Assumptions:
- Speed limit: 60 km/h
- Intersection delay: 30 seconds per intersection

### Machine Learning Models

All models are implemented in PyTorch:

| Model | Description |
|--------|------------|
| LSTM | Classic recurrent network; captures long-term dependencies |
| GRU | Gated recurrent unit; similar to LSTM with fewer parameters |
| CNN-LSTM | 1D convolutional layers for feature extraction + LSTM |
| Transformer | Self-attention based model for sequence forecasting |
| MLP | Simple feedforward network (baseline) |

Sequence length: 24 (6 hours), Prediction horizon: 1 (next 15 minutes). Training uses MSE loss and Adam optimizer.

### Path Finding

The system converts the ML-predicted traffic flows into dynamic travel times and finds routes using the search algorithms from Assignment 2A:

- **DFS**: Depth-First Search (unweighted)
- **BFS**: Breadth-First Search (unweighted)
- **GBFS**: Greedy Best-First (heuristic only)
- **A***: Optimal cost+heuristic search (weighted, used for main routing)
- **CUS1**: Iterative Deepening Search (unweighted)
- **CUS2**: Uniform Cost Search with heuristic (moves-based)

The graph weights are recomputed for each query based on predicted traffic.

### Top-k Path Generation

The default shortest path is computed using the selected algorithm. Additional routes are generated by penalizing edges that appeared in earlier routes (cost multiplier). This encourages alternative paths while preserving optimality characteristics.

## Configuration

Key parameters in `config.py`:

```python
MLConfig.sequence_length = 24
MLConfig.hidden_size = 64
MLConfig.num_layers = 2
MLConfig.dropout = 0.2

TrafficConfig.speed_limit_kmh = 60.0
TrafficConfig.intersection_delay_seconds = 30.0

PathConfig.TOP_K_PATHS = 5
```

Modify these to experiment with model capacity and traffic parameters.

## Evaluation

The provided `train_models.py` script (not included by default; you can create) automates training and evaluation:

```bash
python train_models.py
```

It trains each model on the full dataset and computes metrics (MAE, RMSE, MAPE) on a held-out test set. Results are saved to `results/` directory.

Alternatively, use the CLI `train` command:

```bash
python cli.py train -m lstm -e 30 -b 64
```

### Metrics

- **MAE** (Mean Absolute Error): average absolute difference (vehicles/hour)
- **RMSE** (Root Mean Square Error): penalizes larger errors
- **MAPE** (Mean Absolute Percentage Error): percentage error

## Report Outline

The assignment report should include:

1. **Introduction**: Problem statement and algorithms used.
2. **Features/Bugs**: List of implemented features (LSTM, GRU, X, traffic converter, GUI, etc.).
3. **Testing**: Unit and integration tests; evaluation of ML models and route guidance.
4. **Insights**: Comparison of ML models (which performed better and why) and impact of traffic prediction on routing.
5. **Research** (optional): Any external resources or additional techniques tried.
6. **Conclusion**: Summary of findings, integration experiences, and potential improvements.
7. **References**: Cite textbooks, papers, online resources (PyTorch docs, etc.).

Include graphs of training loss, prediction vs actual, and example routes.

## Tips

- The first run may take a few seconds to build the graph and load data.
- Use the GUI for easy exploration of routes.
- For faster training, reduce epochs or use a subset of data.
- Ensure all SCATS IDs are zero-padded to 4 digits (e.g., `0970` not `970`) when referencing files.

## Troubleshooting

- **No module named 'torch'**: Install PyTorch (CPU version is fine): `pip install torch`
- **FileNotFoundError**: Make sure you are in the project root directory where `processed_data/` exists.
- **No route found**: Origin/destination pair may not be connected in the graph. Check with `python cli.py list` to confirm IDs.

## License

This project is developed for educational purposes as part of COS30019 (Introduction to AI) at [University Name].
