import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score as R2

def read_data(file_name: str) -> pd.DataFrame:
    df = pd.read_csv("data/{}.csv".format(file_name))
    return df

def encode_data(data: pd.DataFrame) -> np.ndarray:
    enc = OrdinalEncoder(encoded_missing_value=-1)
    vals = data.loc[:, 'parentspecies'].values
    sp_values = enc.fit_transform(vals.reshape(-1, 1))
    
    return sp_values

def process_data(data: pd.DataFrame, sp_values: np.ndarray, istest) -> np.ndarray:
    X = data.loc[1:, data.columns != "pSat_Pa"].values
    if not istest:
        y = data.loc[1:, "pSat_Pa"].values
    rows = data.shape[0] - 1

    str_clm_index = 0
    for i, column in enumerate(X[0,:]):
        if isinstance(column, str):
            str_clm_index = i
    
    for row in range(rows):
        X[row, str_clm_index] = sp_values[row]

    if istest:
        return X    
    else:
        return X, y


data_train = read_data("train")

sp_encode = encode_data(data_train)

X, y = process_data(data_train, sp_encode, istest=False)

model = LinearRegression().fit(X, y)

y_pred_train = model.predict(X)

print("R-squared for training data: ", R2(y, y_pred_train))

data_test = read_data("test")
X_test = process_data(data_test, sp_encode, istest=True)

y_test = model.predict(X_test)

print(y_test)