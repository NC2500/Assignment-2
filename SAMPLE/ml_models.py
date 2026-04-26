"""
Deep Learning Models for Traffic Flow Prediction
Implements LSTM, GRU, and CNN-based models for time series forecasting
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class LSTMModel(nn.Module):
    """Long Short-Term Memory (LSTM) model for traffic flow prediction"""

    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2,
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Use only the last time step output
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


class GRUModel(nn.Module):
    """Gated Recurrent Unit (GRU) model for traffic flow prediction"""

    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2,
                 output_size: int = 1, dropout: float = 0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        gru_out, _ = self.gru(x)
        # Use only the last time step output
        last_hidden = gru_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid model: CNN extracts features, LSTM captures temporal dependencies"""

    def __init__(self, input_size: int = 1, seq_len: int = 24, cnn_filters: int = 32,
                 lstm_hidden: int = 64, lstm_layers: int = 2, output_size: int = 1,
                 dropout: float = 0.2):
        super(CNNLSTMModel, self).__init__()
        self.seq_len = seq_len

        # 1D CNN for feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.bn2 = nn.BatchNorm1d(cnn_filters)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Calculate the output length after CNN layers
        # After conv1 with padding: seq_len
        # After pool: seq_len // 2
        cnn_output_len = seq_len // 2

        self.lstm_input_size = cnn_filters * cnn_output_len

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)

        # Permute to (batch_size, input_size, sequence_length) for Conv1d
        x = x.permute(0, 2, 1)

        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        # After second conv, shape: (batch_size, cnn_filters, seq_len//2)

        # Reshape for LSTM: flatten the CNN features
        x = x.permute(0, 2, 1)  # (batch_size, seq_len//2, cnn_filters)
        x = x.reshape(batch_size, -1)  # Flatten: (batch_size, seq_len//2 * cnn_filters)
        x = x.unsqueeze(1).repeat(1, self.seq_len // 2, 1)  # Repeat to create sequence

        # LSTM
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        # Output
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer model"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    """Transformer-based model for traffic flow prediction"""

    def __init__(self, input_size: int = 1, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super(TransformerModel, self).__init__()
        self.d_model = d_model

        # Input embedding layer
        self.embedding = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        # Use the output corresponding to the last position
        x = x[:, -1, :]  # (batch_size, d_model)
        x = self.dropout(x)
        out = self.fc(x)
        return out


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron as baseline model"""

    def __init__(self, input_size: int = 24, hidden_sizes: list = [128, 64],
                 output_size: int = 1, dropout: float = 0.2):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length) - flattened
        return self.network(x)


def create_model(model_type: str, input_size: int = 1, sequence_length: int = 24,
                 hidden_size: int = 64, num_layers: int = 2, output_size: int = 1,
                 dropout: float = 0.2) -> nn.Module:
    """
    Factory function to create a model by type

    Args:
        model_type: 'lstm', 'gru', 'cnnlstm', 'transformer', or 'mlp'
        input_size: Number of input features
        sequence_length: Length of input sequence
        hidden_size: Hidden dimension for RNNs/Transformer
        num_layers: Number of layers
        output_size: Number of output values
        dropout: Dropout rate

    Returns:
        PyTorch model
    """
    model_type = model_type.lower()

    if model_type == 'lstm':
        return LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_type == 'gru':
        return GRUModel(input_size, hidden_size, num_layers, output_size, dropout)
    elif model_type == 'cnnlstm':
        return CNNLSTMModel(input_size, sequence_length, cnn_filters=hidden_size//2,
                           lstm_hidden=hidden_size, lstm_layers=num_layers,
                           output_size=output_size, dropout=dropout)
    elif model_type == 'transformer':
        return TransformerModel(input_size, d_model=hidden_size, nhead=4,
                               num_layers=num_layers, output_size=output_size, dropout=dropout)
    elif model_type == 'mlp':
        # MLP expects flattened sequence
        return SimpleMLP(input_size=sequence_length, hidden_sizes=[hidden_size*2, hidden_size],
                        output_size=output_size, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Quick test
    print("Testing model creation...")

    batch_size = 10
    seq_len = 24
    input_size = 1

    models_to_test = ['lstm', 'gru', 'cnnlstm', 'transformer', 'mlp']

    for model_name in models_to_test:
        try:
            if model_name == 'mlp':
                x = torch.randn(batch_size, seq_len)
            else:
                x = torch.randn(batch_size, seq_len, input_size)

            model = create_model(model_name, input_size=input_size, sequence_length=seq_len)
            out = model(x)
            print(f"{model_name.upper()}: Input {x.shape} -> Output {out.shape}")
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
