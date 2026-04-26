from train_models import load_data_sequences, split_data
from traffic_predictor import train_model, evaluate_model
import numpy as np

print('Loading data...')
X, y = load_data_sequences('processed_data', seq_len=12)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

X_train_s = X_train[:2000]; y_train_s = y_train[:2000]
X_test_s = X_test[:400]; y_test_s = y_test[:400]

model = train_model('randomforest', X_train_s, y_train_s, X_val[:400], y_val[:400], save_path=None)
metrics = evaluate_model(model, X_test_s, y_test_s, 'randomforest')
print('Test MAE:', metrics['MAE'], 'RMSE:', metrics['RMSE'])
