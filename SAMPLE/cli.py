"""
Command-Line Interface for TBRGS
Runs the route guidance system from terminal
"""

import argparse
import sys
import os
from typing import List

from tbrgs import TBRGS, quick_route_query
from graph_builder import BoroondaraGraphBuilder
from traffic_predictor import train_model, load_scats_data, TrafficPredictor
from ml_models import create_model
from data_processor import TrafficDataProcessor
import numpy as np


def cmd_find_route(args):
    """Find routes between two SCATS sites"""
    origin = args.origin
    destination = args.destination
    k = args.k
    model_type = args.model
    method = args.method

    print(f"\nTBRGS Route Finder")
    print(f"Origin: {origin}, Destination: {destination}")
    print(f"Model: {model_type}, Algorithm: {method}, K: {k}\n")

    routes = quick_route_query(origin, destination, model_type=model_type, k=k, method=method)

    if routes:
        print("\n" + "="*60)
        print("TOP ROUTES")
        print("="*60)

        for i, route in enumerate(routes):
            print(f"\nRoute {i+1}:")
            print(f"  Path: {' -> '.join(route.path)}")
            print(f"  Travel Time: {route.travel_time_minutes:.1f} minutes ({route.total_travel_time:.0f} seconds)")
            print(f"  Nodes: {len(route.path)}")

            # Show edge details
            if route.edge_times:
                print("  Edge details:")
                for j, (edge_time, (from_n, to_n)) in enumerate(
                    zip(route.edge_times, zip(route.path[:-1], route.path[1:]))
                ):
                    print(f"    {j+1}. {from_n} -> {to_n}: {edge_time/60:.1f} min")
    else:
        print("No routes found between these locations.")
        print("Please check that the SCATS IDs are valid and the network is connected.")


def cmd_list_sites(args):
    """List all available SCATS sites"""
    builder = BoroondaraGraphBuilder()
    builder.build_graph(method='hybrid')
    nodes = builder.get_nodes_dict()

    print("\nAvailable SCATS Sites in Boroondara Network:")
    print("="*70)
    print(f"{'ID':<10} {'Location':<45} {'Data Points'}")
    print("-"*70)

    for scats_id in sorted(nodes.keys()):
        node = nodes[scats_id]
        print(f"{scats_id:<10} {node.location:<45} {node.data_points}")

    print(f"\nTotal: {len(nodes)} sites")


def cmd_train(args):
    """Train ML models on the traffic dataset"""
    print(f"\nTraining {args.model.upper()} model...")
    print(f"Data directory: {args.data_dir}")
    print(f"Sequence length: {args.seq_len}, Epochs: {args.epochs}")
    print(f"Output directory: {args.output_dir}\n")

    # Load data
    os.makedirs(args.output_dir, exist_ok=True)

    # Load processed data
    data_dict = load_scats_data(args.data_dir)

    if not data_dict:
        print("Error: No training data found. Run data_processor.py first.")
        return

    # Prepare training data from all sites
    X_all, y_all = [], []

    for scats_id, values in data_dict.items():
        if len(values) < args.seq_len + 1:
            continue

        for i in range(len(values) - args.seq_len):
            X_all.append(values[i:i+args.seq_len])
            y_all.append(values[i+args.seq_len])  # Predict next step

    if not X_all:
        print("Error: Not enough data to create sequences")
        return

    X = np.array(X_all, dtype=np.float32).reshape(-1, args.seq_len, 1)
    y = np.array(y_all, dtype=np.float32).reshape(-1, 1)

    # Split train/val
    split_idx = int(len(X) * (1 - args.val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Train model
    save_path = os.path.join(args.output_dir, f"{args.model}_best.pth")

    model = train_model(
        model_type=args.model,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=save_path
    )

    print(f"\nModel trained and saved to {save_path}")

    # Quick evaluation
    from traffic_predictor import evaluate_model
    metrics = evaluate_model(model, X_val, y_val, args.model)
    print(f"\nValidation Metrics:")
    print(f"  MAE: {metrics['MAE']:.2f} vehicles/hour")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")


def cmd_predict(args):
    """Show traffic predictions for a specific site"""
    scats_id = args.site
    predictor = TrafficPredictor(model_type=args.model, sequence_length=args.seq_len)

    # Load data
    data_dict = load_scats_data(args.data_dir)

    if scats_id not in data_dict:
        print(f"Error: No data for SCATS site {scats_id}")
        print(f"Available sites: {list(data_dict.keys())[:10]}...")
        return

    data = data_dict[scats_id]
    print(f"\nTraffic Flow Prediction for SCATS {scats_id}")
    print(f"Historical data points: {len(data)}")
    print(f"Mean flow: {np.mean(data):.1f} veh/h, Std: {np.std(data):.1f}")

    # Make predictions for different time horizons
    print(f"\nPredictions (next {args.horizon} steps, 15-min each):")
    preds = predictor.predict(data, future_steps=args.horizon)

    for i, pred in enumerate(preds):
        print(f"  t+{i+1}: {pred:.1f} veh/h")


def cmd_gui(args):
    """Launch the graphical interface"""
    import tkinter as tk
    from gui import TBRGSGUI

    print("Starting TBRGS GUI...")
    root = tk.Tk()
    app = TBRGSGUI(root)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="TBRGS - Traffic-Based Route Guidance System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Route finding
    route_parser = subparsers.add_parser('route', help='Find routes between two points')
    route_parser.add_argument('origin', help='Origin SCATS site ID (e.g., 2000)')
    route_parser.add_argument('destination', help='Destination SCATS site ID (e.g., 3002)')
    route_parser.add_argument('-k', type=int, default=5, help='Number of routes to return')
    route_parser.add_argument('-m', '--model', default='lstm',
                              choices=['lstm', 'gru', 'randomforest', 'cnnlstm', 'transformer', 'mlp'],
                              help='ML model type')
    route_parser.add_argument('-a', '--algorithm', dest='method', default='astar',
                              choices=['astar', 'bfs', 'dfs', 'gbfs', 'cus1', 'cus2'],
                              help='Search algorithm')
    route_parser.set_defaults(func=cmd_find_route)

    # List sites
    list_parser = subparsers.add_parser('list', help='List available SCATS sites')
    list_parser.set_defaults(func=cmd_list_sites)

    # Train model
    train_parser = subparsers.add_parser('train', help='Train a traffic prediction model')
    train_parser.add_argument('-d', '--data-dir', default='processed_data',
                              help='Directory with processed SCATS data')
    train_parser.add_argument('-m', '--model', default='lstm',
                              choices=['lstm', 'gru', 'randomforest', 'cnnlstm', 'transformer', 'mlp', 'linear'],
                              help='Model architecture')
    train_parser.add_argument('-e', '--epochs', type=int, default=50,
                              help='Number of training epochs')
    train_parser.add_argument('-b', '--batch-size', type=int, default=32,
                              help='Batch size')
    train_parser.add_argument('-l', '--lr', type=float, default=0.001,
                              help='Learning rate')
    train_parser.add_argument('-s', '--seq-len', type=int, default=24,
                              help='Sequence length (past time steps)')
    train_parser.add_argument('-v', '--val-split', type=float, default=0.2,
                              help='Validation split fraction')
    train_parser.add_argument('-o', '--output-dir', default='models',
                              help='Directory to save trained models')
    train_parser.set_defaults(func=cmd_train)

    # Predict
    predict_parser = subparsers.add_parser('predict', help='Predict traffic flow for a site')
    predict_parser.add_argument('site', help='SCATS site ID')
    predict_parser.add_argument('-m', '--model', default='lstm',
                                choices=['lstm', 'gru', 'randomforest', 'cnnlstm', 'transformer', 'mlp', 'linear'],
                                help='Model to use')
    predict_parser.add_argument('-s', '--seq-len', type=int, default=24,
                                help='Sequence length')
    predict_parser.add_argument('-H', '--horizon', type=int, default=4,
                                help='Prediction horizon (steps ahead)')
    predict_parser.add_argument('-d', '--data-dir', default='processed_data',
                                help='Directory with processed SCATS data')
    predict_parser.set_defaults(func=cmd_predict)

    # GUI
    gui_parser = subparsers.add_parser('gui', help='Launch graphical user interface')
    gui_parser.set_defaults(func=cmd_gui)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
