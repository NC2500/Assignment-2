# train and save all three models.
#Feed the sequences 
#from step 2 into each model. 
# Save the trained weights as .h5 files. 
# Log MAE, MSE, RMSE, MAPE, and R2 for each model 
# so you have comparison data for the report.


from model import get_lstm, get_gru, get_saes, get_cnn_lstm
from data import process_data

# All 40 sites combined
X_train, y_train, X_test, y_test, scaler = process_data(
    'dataset/Scats Data October 2006.xls', lags=12)

# Reshape for LSTM/GRU/CNN-LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# X_train_flat stays as-is (shape: samples, 12) for SAE

# All models take [input_steps, hidden1, hidden2, output]
lstm_model  = get_lstm([12, 64, 64, 1])
gru_model   = get_gru([12, 64, 64, 1])
cnn_model   = get_cnn_lstm([12, 64, 64, 1])

# SAEs takes [input, h1, h2, h3, output]
sae_models  = get_saes([12, 400, 400, 400, 1])

# Compile the same way for all three main models
lstm_model.compile(loss='mse', optimizer='adam', metrics=['mape'])
gru_model.compile(loss='mse', optimizer='adam', metrics=['mape'])
cnn_model.compile(loss='mse', optimizer='adam', metrics=['mape'])

lstm_model.fit(X_train, y_train, batch_size=256, epochs=50,
          validation_split=0.1)
gru_model.fit(X_train, y_train, batch_size=256, epochs=50,
            validation_split=0.1)
cnn_model.fit(X_train, y_train, batch_size=256, epochs=50,
            validation_split=0.1)