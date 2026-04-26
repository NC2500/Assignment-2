from train_models import load_data_sequences, split_data
from traffic_predictor import train_model, evaluate_model
import numpy as np

print("=== Loading Data ===")
X, y = load_data_sequences('processed_data', seq_len=12)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.2, val_size=0.2)

# Use tiny subset for quick test
X_train_s = X_train[:2000]
y_train_s = y_train[:2000]
X_val_s = X_val[:400]
y_val_s = y_val[:400]
X_test_s = X_test[:400]
y_test_s = y_test[:400]

models = ['lstm', 'gru', 'randomforest']
results = {}

for m in models:
    print(f"\n{'='*50}")
    print(f"Training {m.upper()}")
    print('='*50)
    model = train_model(m, X_train_s, y_train_s, X_val_s, y_val_s, epochs=2, batch_size=32, save_path=None)
    metrics = evaluate_model(model, X_test_s, y_test_s, m)
    print(f"Test - MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.1f}%")
    results[m] = metrics

print("\n\n=== SUMMARY ===")
print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
for m, r in results.items():
    print(f"{m.upper():<12} {r['MAE']:>8.2f} {r['RMSE']:>8.2f} {r['MAPE']:>7.1f}%")
