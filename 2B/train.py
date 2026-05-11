"""
Train, evaluate, and save all traffic flow prediction models.
Outputs: saved model files + metrics comparison printed to console.
"""

import os
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import get_lstm, get_gru, get_cnn_lstm, get_saes
from data import process_data


# --- Config ---
DATA_FILE  = 'dataset/Scats Data October 2006.xls'
LAGS       = 12
BATCH      = 256
EPOCHS     = 1
UNITS      = [12, 64, 64, 1]       # LSTM / GRU / CNN-LSTM
SAE_UNITS  = [12, 400, 400, 400, 1]
MODEL_DIR  = 'model'

os.makedirs(MODEL_DIR, exist_ok=True)


# --- Metrics ---
def evaluate(y_true, y_pred, scaler):
    """Inverse-transform predictions and compute all metrics."""
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2   = r2_score(y_true, y_pred)

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


def print_metrics(name, metrics):
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


# --- Load Data ---
X_train, y_train, X_test, y_test, scaler = process_data(DATA_FILE, lags=LAGS)

# 3D shape for LSTM / GRU / CNN-LSTM: (samples, lags, 1)
X_train_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_3d  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

# 2D shape for SAE: (samples, lags) - already correct from process_data


# --- Helper: train one model ---
def train_model(model, name, X_tr, X_te):
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(f'{MODEL_DIR}/{name}.h5', save_best_only=True)
    ]

    model.fit(
        X_tr, y_train,
        batch_size=BATCH,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    y_pred = model.predict(X_te).flatten()
    metrics = evaluate(y_test, y_pred, scaler)
    print_metrics(name, metrics)
    return metrics


# --- Train LSTM ---
lstm = get_lstm(UNITS)
lstm_metrics = train_model(lstm, 'lstm', X_train_3d, X_test_3d)

# --- Train GRU ---
gru = get_gru(UNITS)
gru_metrics = train_model(gru, 'gru', X_train_3d, X_test_3d)

# --- Train CNN-LSTM ---
cnn = get_cnn_lstm(UNITS)
cnn_metrics = train_model(cnn, 'cnn_lstm', X_train_3d, X_test_3d)

# --- Train SAEs ---
# SAEs are pretrained layer by layer, then the full model is fine-tuned
saes = get_saes(SAE_UNITS)
sae1, sae2, sae3, saes_model = saes

for i, (sae, (in_dim, out_dim)) in enumerate(zip(
    [sae1, sae2, sae3],
    [(SAE_UNITS[0], SAE_UNITS[1]),
     (SAE_UNITS[1], SAE_UNITS[2]),
     (SAE_UNITS[2], SAE_UNITS[3])]
), start=1):
    sae.compile(loss='mse', optimizer='adam')
    sae.fit(X_train, y_train, batch_size=BATCH, epochs=EPOCHS,
            validation_split=0.1, verbose=0)

# Fine-tune full SAE
saes_model.compile(loss='mse', optimizer='adam', metrics=['mae'])
callbacks_sae = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(f'{MODEL_DIR}/saes.h5', save_best_only=True)
]
saes_model.fit(
    X_train, y_train,
    batch_size=BATCH, epochs=EPOCHS,
    validation_split=0.1, callbacks=callbacks_sae, verbose=1
)
y_pred_sae = saes_model.predict(X_test).flatten()
sae_metrics = evaluate(y_test, y_pred_sae, scaler)
print_metrics('SAEs', sae_metrics)


# --- Final Comparison ---
print("\n====== Model Comparison ======")
print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'R2':>8}")
print("-" * 44)
for name, m in [('LSTM', lstm_metrics), ('GRU', gru_metrics),
                ('CNN-LSTM', cnn_metrics), ('SAEs', sae_metrics)]:
    print(f"{name:<12} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['MAPE']:>7.2f}% {m['R2']:>8.4f}")