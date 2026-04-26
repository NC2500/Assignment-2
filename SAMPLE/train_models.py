"""
Model Training and Evaluation Script
Trains LSTM, GRU, CNN-LSTM, Transformer, and MLP on SCATS traffic data
and evaluates their performance.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import json
import argparse

from traffic_predictor import load_scats_data, train_model, evaluate_model
from data_processor import TrafficDataProcessor
import torch

# Configuration
SEQUENCE_LENGTH = 24
PREDICTION_HORIZON = 1
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

# Required models: LSTM, GRU, Random Forest (plus optional extras)
MODELS = ['lstm', 'gru', 'randomforest']
# For extended comparison you can also train: 'cnnlstm', 'transformer', 'mlp', 'linear'
DATA_DIR = 'processed_data'
MODEL_DIR = 'models'
RESULT_DIR = 'results'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def load_data_sequences(data_dir: str, seq_len: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all SCATS time series and create training sequences

    Returns:
        X: (n_samples, seq_len, 1) - input sequences
        y: (n_samples, 1) - target next-step flow
    """
    print("Loading time series data...")
    data_dict = load_scats_data(data_dir)

    X_list = []
    y_list = []

    for scats_id, values in data_dict.items():
        values = np.array(values, dtype=np.float32)
        # Remove NaNs
        values = values[~np.isnan(values)]

        if len(values) < seq_len + 1:
            continue

        # Create sliding windows
        for i in range(len(values) - seq_len):
            X_list.append(values[i:i+seq_len])
            y_list.append(values[i+seq_len])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Reshape X to 3D for RNNs (n_samples, seq_len, 1)
    X = X.reshape(-1, seq_len, 1)
    y = y.reshape(-1, 1)

    print(f"Total sequences: {len(X)}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def split_data(X: np.ndarray, y: np.ndarray,
               test_size: float = 0.2, val_size: float = 0.2,
               random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """Split data into train/val/test sets"""
    from sklearn.model_selection import train_test_split

    # First split off test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Split remaining into train/val
    val_size_adj = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adj, random_state=random_state
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_evaluate_all(models: list = MODELS,
                          epochs: int = EPOCHS,
                          batch_size: int = BATCH_SIZE,
                          lr: float = LEARNING_RATE,
                          seq_len: int = SEQUENCE_LENGTH):
    """Train each model and evaluate on test set"""
    print("\n=== Loading Data ===")
    X, y = load_data_sequences(DATA_DIR, seq_len=seq_len)

    print("\n=== Splitting Data ===")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    results = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")

        # Train
        model_save_path = os.path.join(MODEL_DIR, f"{model_name}_best.pth")
        model = train_model(
            model_type=model_name,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            save_path=model_save_path
        )

        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results[model_name] = {
            'MAE': float(metrics['MAE']),
            'RMSE': float(metrics['RMSE']),
            'MAPE': float(metrics['MAPE'])
        }
        print(f"Test - MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.1f}%")

    # Save all results
    results_path = os.path.join(RESULT_DIR, 'model_comparison.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("\n=== Summary ===")
    print(f"{'Model':<12} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
    print("-" * 44)
    for model, m in results.items():
        print(f"{model.upper():<12} {m['MAE']:>10.2f} {m['RMSE']:>10.2f} {m['MAPE']:>9.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate traffic prediction models")
    parser.add_argument('-m', '--models', nargs='+', default=MODELS,
                        help='Models to train')
    parser.add_argument('-e', '--epochs', type=int, default=EPOCHS)
    parser.add_argument('-b', '--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('-l', '--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('-s', '--seq-len', type=int, default=SEQUENCE_LENGTH)
    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    train_and_evaluate_all(
        models=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len
    )


if __name__ == "__main__":
    main()
