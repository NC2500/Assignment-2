


from model import get_lstm, get_gru, get_saes, get_cnn_lstm



# All models take [input_steps, hidden1, hidden2, output]
lstm_model  = get_lstm([12, 64, 64, 1])
gru_model   = get_gru([12, 64, 64, 1])
cnn_model   = get_cnn_lstm([12, 64, 64, 1])

# SAEs takes [input, h1, h2, h3, output]
sae_models  = get_saes([12, 400, 400, 400, 1])

# Compile the same way for all three main models
model.compile(loss='mse', optimizer='adam', metrics=['mape'])
model.fit(X_train, y_train, batch_size=256, epochs=50,
          validation_split=0.1)