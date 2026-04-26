"""
Traditional Machine Learning Models for Traffic Flow Prediction
Includes Random Forest and other sklearn-based models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

from config import MLConfig


class RandomForestPredictor:
    """
    Random Forest Regressor for traffic flow prediction.
    Uses scikit-learn implementation with time series feature engineering.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 20,
                 min_samples_split: int = 5, random_state: int = 42,
                 sequence_length: int = 24):
        """
        Initialize Random Forest predictor

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            random_state: Random seed for reproducibility
            sequence_length: Number of past time steps to use as features
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.sequence_length = sequence_length

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )

        self.scaler = StandardScaler()
        self.is_trained = False

    def create_features(self, data: np.ndarray) -> np.ndarray:
        """
        Create feature matrix from time series using sliding window

        Args:
            data: 1D array of traffic flow values

        Returns:
            2D array: (n_samples, sequence_length) feature matrix
        """
        X = []
        y = []

        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None):
        """
        Train the Random Forest model

        Args:
            X: Training features (n_samples, seq_len)
            y: Training targets (n_samples,)
            validation_data: Optional (X_val, y_val) tuple
        """
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        self.is_trained = True

        train_pred = self.model.predict(X_scaled)
        train_mae = mean_absolute_error(y, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y, train_pred))

        print(f"Random Forest trained on {len(X)} samples")
        print(f"  Train MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}")

        if validation_data:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            print(f"  Val MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}")

        return self

    def predict(self, data: np.ndarray, future_steps: int = 1) -> np.ndarray:
        """
        Predict future traffic flow

        Args:
            data: Historical time series (1D array)
            future_steps: Number of steps ahead to predict

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            # Return naive prediction (mean) if not trained
            return np.full(future_steps, np.nanmean(data) if len(data) > 0 else 0.0)

        # Use last sequence as input
        if len(data) < self.sequence_length:
            # Pad with zeros or repeat last value
            pad_length = self.sequence_length - len(data)
            data = np.concatenate([np.zeros(pad_length), data])

        current_seq = data[-self.sequence_length:].reshape(1, -1)

        predictions = []
        for _ in range(future_steps):
            # Scale features
            X_scaled = self.scaler.transform(current_seq)

            # Predict
            pred = self.model.predict(X_scaled)[0]
            predictions.append(pred)

            # Update sequence for multi-step prediction
            current_seq = np.roll(current_seq, -1)
            current_seq[0, -1] = pred

        return np.array(predictions)

    def save(self, path: str):
        """Save model to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'sequence_length': self.sequence_length
            }, f)
        print(f"Random Forest model saved to {path}")

    def load(self, path: str):
        """Load model from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.sequence_length = data['sequence_length']
        self.is_trained = True
        print(f"Random Forest model loaded from {path}")


class LinearRegressionPredictor:
    """Simple linear regression baseline for traffic prediction"""

    def __init__(self, sequence_length: int = 24):
        self.sequence_length = sequence_length
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False

    def create_features(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        train_pred = self.model.predict(X_scaled)
        train_mae = mean_absolute_error(y, train_pred)
        print(f"Linear Regression trained. MAE: {train_mae:.2f}")
        return self

    def predict(self, data: np.ndarray, future_steps: int = 1) -> np.ndarray:
        if not self.is_trained:
            return np.full(future_steps, np.nanmean(data) if len(data) > 0 else 0.0)

        if len(data) < self.sequence_length:
            data = np.concatenate([np.zeros(self.sequence_length - len(data)), data])

        current_seq = data[-self.sequence_length:].reshape(1, -1)
        X_scaled = self.scaler.transform(current_seq)
        pred = self.model.predict(X_scaled)[0]

        return np.array([pred])


# Factory function to create sklearn models
def create_sklearn_model(model_type: str, sequence_length: int = 24, **kwargs) -> object:
    """
    Create a scikit-learn based predictor

    Args:
        model_type: 'randomforest', 'linear', 'ridge', 'lasso', 'svr'
        sequence_length: Length of input sequence
        **kwargs: Additional model parameters

    Returns:
        Initialized model object
    """
    if model_type.lower() == 'randomforest':
        return RandomForestPredictor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 20),
            min_samples_split=kwargs.get('min_samples_split', 5),
            sequence_length=sequence_length
        )
    elif model_type.lower() == 'linear':
        return LinearRegressionPredictor(sequence_length=sequence_length)
    else:
        raise ValueError(f"Unknown sklearn model type: {model_type}")


def train_sklearn_model(model, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
    """
    Train sklearn model with optional validation

    Args:
        model: sklearn predictor object
        X_train: Training features (n_samples, seq_len)
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets

    Returns:
        Trained model
    """
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, validation_data=(X_val, y_val))
    else:
        model.fit(X_train, y_train)

    return model


def evaluate_sklearn_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate sklearn model

    Returns:
        Dictionary with MAE, RMSE, MAPE
    """
    X_scaled = model.scaler.transform(X_test)
    predictions = model.model.predict(X_scaled)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    mask = y_test != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100
    else:
        mape = np.inf

    return {
        'model': model.__class__.__name__,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'predictions': predictions,
        'actual': y_test
    }


if __name__ == "__main__":
    print("Traditional ML models module loaded.")
    print("Use RandomForestPredictor, LinearRegressionPredictor, or create_sklearn_model()")
