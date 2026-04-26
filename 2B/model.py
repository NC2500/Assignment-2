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


def _get_sae(inputs, hidden, output):
    """Single Stacked Auto-Encoder layer.

    # Arguments
        inputs: Integer, number of input units
        hidden: Integer, number of hidden units
        output: Integer, number of output units
    # Returns
        model: Sequential Keras model
    """
    model = Sequential([
        Dense(hidden, input_dim=inputs, name='hidden'),
        Activation('sigmoid'),
        Dropout(0.2),
        Dense(output, activation='sigmoid')
    ])
    return model


def get_saes(layers):
    """SAEs (Stacked Auto-Encoders)
    Builds 3 individual SAE sub-models and one full stacked SAE.

    # Arguments
        layers: List(int), [input, hidden1, hidden2, hidden3, output]
                e.g. [12, 400, 400, 400, 1]
    # Returns
        models: List of [sae1, sae2, sae3, full_saes]
    """
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential([
        Dense(layers[1], input_dim=layers[0], name='hidden1'),
        Activation('sigmoid'),
        Dense(layers[2], name='hidden2'),
        Activation('sigmoid'),
        Dense(layers[3], name='hidden3'),
        Activation('sigmoid'),
        Dropout(0.2),
        Dense(layers[4], activation='sigmoid')
    ])

    return [sae1, sae2, sae3, saes]