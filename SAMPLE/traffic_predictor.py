"""
Traffic Flow Predictor
Uses trained ML models (PyTorch or scikit-learn) to predict future traffic flows
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os

from ml_models import create_model, LSTMModel, GRUModel
from ml_sklearn import create_sklearn_model, RandomForestPredictor, LinearRegressionPredictor
from data_processor import TrafficDataProcessor
from config import MLConfig, MODELS_DIR, DATA_DIR


class TrafficPredictor:
    """
    Predicts traffic flow for SCATS sites using trained models
    Supports both deep learning (PyTorch) and traditional ML (scikit-learn) models
    Supported types: 'lstm', 'gru', 'cnnlstm', 'transformer', 'mlp', 'randomforest', 'linear'
    """

    def __init__(self, model_type: str = 'lstm', model_path: Optional[str] = None,
                 sequence_length: int = 24, device: Optional[str] = None, **model_kwargs):
        """
        Initialize the traffic predictor

        Args:
            model_type: Type of model ('lstm', 'gru', 'cnnlstm', 'transformer', 'mlp', 'randomforest', 'linear')
            model_path: Path to saved model weights
            sequence_length: Number of past time steps to use
            device: 'cpu' or 'cuda' (for PyTorch models only)
            **model_kwargs: Additional model parameters (for sklearn models)
        """
        self.model_type = model_type.lower()
        self.sequence_length = sequence_length
        self.is_sklearn = self.model_type in ['randomforest', 'linear', 'ridge', 'lasso', 'svr']

        if self.is_sklearn:
            # Use sklearn implementation
            self.model = create_sklearn_model(self.model_type, sequence_length=sequence_length, **model_kwargs)
            self.device = None
        else:
            # PyTorch model
            self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = create_model(
                model_type=self.model_type,
                input_size=1,
                sequence_length=sequence_length,
                hidden_size=MLConfig.hidden_size,
                num_layers=MLConfig.num_layers,
                output_size=MLConfig.prediction_horizon,
                dropout=MLConfig.dropout
            ).to(self.device)

        # Load saved weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded {self.model_type.upper()} model from {model_path}")
        else:
            print(f"Initialized untrained {self.model_type.upper()} model")

        if not self.is_sklearn:
            self.model.eval()  # Set to evaluation mode (PyTorch only)

        # Cache for recent predictions
        self.prediction_cache: Dict[str, np.ndarray] = {}

    def load_model(self, path: str):
        """Load model weights from file"""
        if self.is_sklearn:
            self.model.load(path)
        else:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)

    def save_model(self, path: str):
        """Save model weights to file"""
        if self.is_sklearn:
            self.model.save(path)
        else:
            torch.save(self.model.state_dict(), path)

    def preprocess_sequence(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare raw traffic flow data into normalized sequences

        Args:
            data: 1D array of traffic flow values

        Returns:
            3D array (n_sequences, seq_len, 1)
        """
        # Handle NaNs
        data = np.nan_to_num(data, nan=0.0)

        # Normalize using robust statistics (clip outliers)
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            data_normalized = (data - mean) / std
        else:
            data_normalized = data - mean

        # Create sequences
        sequences = []
        for i in range(len(data_normalized) - self.sequence_length - MLConfig.prediction_horizon + 1):
            seq = data_normalized[i:i + self.sequence_length].reshape(-1, 1)
            sequences.append(seq)

        if not sequences:
            # Not enough data, pad with zeros
            pad = np.zeros((self.sequence_length, 1))
            sequences = [pad]

        return np.array(sequences, dtype=np.float32), mean, std

    def predict(self, data: np.ndarray, future_steps: int = 1) -> np.ndarray:
        """
        Predict future traffic flow given historical data

        Args:
            data: 1D array of historical traffic flow
            future_steps: Number of future time steps to predict

        Returns:
            Array of predicted traffic flow values
        """
        if len(data) < self.sequence_length:
            # Not enough data, use simple average as fallback
            return np.full(future_steps, np.nanmean(data) if len(data) > 0 else 0.0)

        if self.is_sklearn:
            # sklearn models expect 2D (n_samples, features)
            return self.model.predict(data, future_steps=future_steps)
        else:
            # Preprocess for PyTorch
            sequences, mean, std = self.preprocess_sequence(data)

            with torch.no_grad():
                # Use the last sequence for prediction
                x = torch.from_numpy(sequences[-1:]).to(self.device)  # Shape: (1, seq_len, 1)

                if self.model_type == 'mlp':
                    # MLP expects 2D input
                    x = x.squeeze(-1)  # Remove feature dimension

                predictions = []
                for _ in range(future_steps):
                    pred = self.model(x)  # Shape: (1, pred_horizon)
                    # Take the first prediction if prediction_horizon > 1
                    pred_val = pred.cpu().numpy()[0, 0] if pred.shape[1] >= 1 else pred.cpu().numpy()[0, -1]
                    predictions.append(pred_val)

                    # For multi-step prediction, recursively use prediction
                    if self.model_type != 'mlp':
                        # Shift sequence and append prediction
                        x = torch.cat([x[:, 1:, :],
                                       torch.from_numpy(np.array([[[pred_val]]]).astype(np.float32)).to(self.device)], dim=1)
                    else:
                        # MLP doesn't support recursive prediction
                        break

            # Denormalize predictions
            predictions = np.array(predictions)
            if std > 0:
                predictions = predictions * std + mean
            else:
                predictions = predictions + mean

            return predictions


class MultiModelEnsemble:
    """
    Ensemble predictor that combines predictions from multiple models
    """

    def __init__(self, model_configs: List[Dict], sequence_length: int = 24):
        """
        Args:
            model_configs: List of dicts with keys: 'type', 'path' (optional), 'weight' (optional)
            sequence_length: Sequence length for all models
        """
        self.predictors = []
        self.weights = []

        for config in model_configs:
            model_type = config['type']
            model_path = config.get('path')
            weight = config.get('weight', 1.0)

            predictor = TrafficPredictor(
                model_type=model_type,
                model_path=model_path,
                sequence_length=sequence_length
            )
            self.predictors.append(predictor)
            self.weights.append(weight)

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def predict(self, data: np.ndarray, future_steps: int = 1) -> np.ndarray:
        """Weighted average of all models' predictions"""
        predictions = []

        for predictor, weight in zip(self.predictors, self.weights):
            pred = predictor.predict(data, future_steps)
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)


def load_scats_data(data_dir: str = DATA_DIR) -> Dict[str, np.ndarray]:
    """
    Load processed time series data for all SCATS sites

    Returns:
        Dictionary mapping SCATS ID to traffic flow time series
    """
    data_dict = {}

    # Load metadata to get site IDs
    metadata_path = os.path.join(data_dir, 'scats_metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    metadata_df = pd.read_csv(metadata_path)

    for _, row in metadata_df.iterrows():
        scats_id = str(row['SCATS_ID'])
        file_path = os.path.join(data_dir, f'scats_{scats_id}_timeseries.csv')

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            values = df['traffic_flow'].values
            data_dict[scats_id] = values

    print(f"Loaded data for {len(data_dict)} SCATS sites")
    return data_dict


def train_model(model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001,
                save_path: Optional[str] = None, **kwargs) -> object:
    """
    Train a traffic prediction model (PyTorch or scikit-learn)

    Args:
        model_type: Type of model to train
        X_train: Training input sequences (n_samples, seq_len) for sklearn or (n_samples, seq_len, n_features) for PyTorch
        y_train: Training targets (n_samples,)
        X_val: Validation inputs
        y_val: Validation targets
        epochs: Number of training epochs (for PyTorch; ignored for sklearn)
        batch_size: Batch size (PyTorch only)
        learning_rate: Learning rate (PyTorch only)
        save_path: Path to save trained model
        **kwargs: Additional model-specific parameters

    Returns:
        Trained model object
    """
    model_type_lower = model_type.lower()

    # Check if it's a sklearn model
    sklearn_models = ['randomforest', 'linear', 'ridge', 'lasso', 'svr']
    is_sklearn = model_type_lower in sklearn_models

    if is_sklearn:
        # For sklearn, X_train should be 2D (n_samples, features)
        if len(X_train.shape) == 3:
            X_train_2d = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_2d = X_train

        if X_val is not None:
            if len(X_val.shape) == 3:
                X_val_2d = X_val.reshape(X_val.shape[0], -1)
            else:
                X_val_2d = X_val

        # Create and train sklearn model
        model = create_sklearn_model(model_type_lower, sequence_length=X_train_2d.shape[1])

        # Flatten y if needed
        if y_train.ndim > 1:
            y_train_flat = y_train.ravel()
        else:
            y_train_flat = y_train

        if X_val is not None and y_val is not None:
            if y_val.ndim > 1:
                y_val_flat = y_val.ravel()
            else:
                y_val_flat = y_val
            from ml_sklearn import train_sklearn_model
            model = train_sklearn_model(model, X_train_2d, y_train_flat, X_val_2d, y_val_flat)
        else:
            model.fit(X_train_2d, y_train_flat)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)

        print(f"Trained {model_type.upper()} (sklearn)")
        return model

    else:
        # PyTorch path
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input_size = X_train.shape[2] if len(X_train.shape) > 2 else 1

        model = create_model(
            model_type=model_type_lower,
            input_size=input_size,
            sequence_length=X_train.shape[1] if len(X_train.shape) > 2 else X_train.shape[1],
            hidden_size=MLConfig.hidden_size,
            num_layers=MLConfig.num_layers,
            output_size=y_train.shape[1] if len(y_train.shape) > 1 else 1,
            dropout=MLConfig.dropout,
            **kwargs
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Convert data to PyTorch tensors
        if len(X_train.shape) == 2:
            X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
        else:
            X_train_tensor = torch.FloatTensor(X_train)

        y_train_tensor = torch.FloatTensor(y_train) if not torch.is_tensor(y_train) else y_train

        if X_val is not None:
            if len(X_val.shape) == 2:
                X_val_tensor = torch.FloatTensor(X_val).unsqueeze(-1)
            else:
                X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val) if not torch.is_tensor(y_val) else y_val

        print(f"Training {model_type.upper()} on {device}...")
        print(f"Training samples: {len(X_train_tensor)}, Validation: {len(X_val_tensor) if X_val is not None else 0}")

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            n_batches = 0

            for i in range(0, len(X_train_tensor), batch_size):
                batch_x = X_train_tensor[i:i+batch_size].to(device)
                batch_y = y_train_tensor[i:i+batch_size].to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches

            if X_val is not None and epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val_tensor.to(device))
                    val_loss = criterion(val_preds, y_val_tensor.to(device)).item()
                print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        return model


def evaluate_model(model: object, X_test: np.ndarray, y_test: np.ndarray,
                   model_type: str, device: str = 'cpu') -> Dict[str, float]:
    """
    Evaluate model performance (supports both PyTorch and sklearn)

    Returns:
        Dictionary with MAE, RMSE, MAPE metrics
    """
    model_type_lower = model_type.lower()
    sklearn_models = ['randomforest', 'linear', 'ridge', 'lasso', 'svr']

    if model_type_lower in sklearn_models:
        # sklearn evaluation
        if len(X_test.shape) == 3:
            X_test_2d = X_test.reshape(X_test.shape[0], -1)
        else:
            X_test_2d = X_test

        X_test_scaled = model.scaler.transform(X_test_2d)
        predictions = model.model.predict(X_test_scaled)
    else:
        # PyTorch evaluation
        model.eval()

        if len(X_test.shape) == 2:
            X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1).to(device)
        else:
            X_test_tensor = torch.FloatTensor(X_test).to(device)

        with torch.no_grad():
            predictions = model(X_test_tensor).cpu().numpy()

    y_true = y_test.copy()
    if len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_true = y_true.ravel()

    # Calculate metrics
    mae = np.mean(np.abs(predictions - y_true))
    mse = np.mean((predictions - y_true) ** 2)
    rmse = np.sqrt(mse)

    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - predictions[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf

    return {
        'model': model_type,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'predictions': predictions,
        'actual': y_true
    }


if __name__ == "__main__":
    print("Traffic Predictor module loaded successfully")
    print("Use TrafficPredictor class to load models and make predictions")
