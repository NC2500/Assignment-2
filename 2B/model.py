"""
Definition of NN models for Traffic Flow Prediction.
Models: LSTM, GRU, SAEs, CNN-LSTM
"""

from keras.layers import (
    Dense, Dropout, Activation,
    LSTM, GRU,
    Conv1D, MaxPooling1D, Flatten, Reshape
)
from keras.models import Sequential


def get_lstm(units):
    """LSTM (Long Short-Term Memory)
    Build a stacked LSTM model.

    # Arguments
        units: List(int), [input_steps, hidden1, hidden2, output]
                e.g. [12, 64, 64, 1]
    # Returns
        model: Sequential Keras model
    """
    model = Sequential([
        LSTM(units[1], input_shape=(units[0], 1), return_sequences=True),
        LSTM(units[2]),
        Dropout(0.2),
        Dense(units[3], activation='sigmoid')
    ])
    return model


def get_gru(units):
    """GRU (Gated Recurrent Unit)
    Build a stacked GRU model.

    # Arguments
        units: List(int), [input_steps, hidden1, hidden2, output]
                e.g. [12, 64, 64, 1]
    # Returns
        model: Sequential Keras model
    """
    model = Sequential([
        GRU(units[1], input_shape=(units[0], 1), return_sequences=True),
        GRU(units[2]),
        Dropout(0.2),
        Dense(units[3], activation='sigmoid')
    ])
    return model


def get_cnn_lstm(units):
    """CNN-LSTM (Convolutional + Long Short-Term Memory)
    CNN extracts local temporal patterns first, then LSTM
    captures longer-range dependencies.


    # Arguments
        units: List(int), [input_steps, filters, lstm_units, output]
                e.g. [12, 64, 64, 1]
    # Returns
        model: Sequential Keras model
    """
    model = Sequential([
        # CNN block: extract short-term local patterns
        Conv1D(filters=units[1], kernel_size=3, activation='relu',
               input_shape=(units[0], 1), padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # LSTM block: capture temporal dependencies
        LSTM(units[2], return_sequences=False),
        Dropout(0.2),

        Dense(units[3], activation='sigmoid')
    ])
    return model
