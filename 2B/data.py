# to process the data and prepare it for training the model
# it reads the XLS, extract the 15-minute interval flow counts per SCATS site,
# normalize the values with MinMaxScaler,
# and build sequences (X, y) for model training.
# this is worth 9 marks

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import xlrd


# V00 to V95: 96 x 15-minute intervals per day
V_COLS = 96


def load_flow_series(filepath, scats_num=None):
    wb = xlrd.open_workbook(filepath)
    sh = wb.sheet_by_name('Data')

    rows = []
    for r in range(2, sh.nrows):
        site = str(sh.cell_value(r, 0)).strip()
        if scats_num and site != str(scats_num):
            continue
        # Columns 10 to 105 are V00-V95
        day_vals = [sh.cell_value(r, 10 + i) for i in range(V_COLS)]
        rows.append(day_vals)

    # Flatten all days into one time series, preserving order
    flow = np.array(rows, dtype=np.float32).flatten()
    flow = flow[~np.isnan(flow)]
    return flow


def process_data(filepath, lags, scats_num=None, test_ratio=0.2):
    flow = load_flow_series(filepath, scats_num)

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(flow.reshape(-1, 1))
    flow_norm = scaler.transform(flow.reshape(-1, 1)).reshape(-1)

    # Build sliding window sequences
    samples = []
    for i in range(lags, len(flow_norm)):
        samples.append(flow_norm[i - lags: i + 1])
    samples = np.array(samples)

    # Split train/test (no shuffle - preserve time order)
    split = int(len(samples) * (1 - test_ratio))
    train, test = samples[:split], samples[split:]

    X_train, y_train = train[:, :-1], train[:, -1]
    X_test,  y_test  = test[:, :-1],  test[:, -1]

    return X_train, y_train, X_test, y_test, scaler