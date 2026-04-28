import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(file_path):

    # Read Excel file
    data = pd.read_excel(file_path,engine='xlrd').fillna(0)

    scaler = MinMaxScaler(feature_range=(0, 1))

    data = []
    for i in range(len(data)):
        row = scaler.fit_transform(data.iloc[i].values.reshape(-1, 1)).flatten()
        data.append(row)

    return data

def main():
    data = load_data('dataset/Scats Data October 2006.xls')
    print(data)

if __name__ == "__main__":
    main()