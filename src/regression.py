from preprocess import *
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score as R2

df_train = read_data("train")
df_test = read_data("test")

edf_train, enc = generate_encoding(df_train) 
edf_test = apply_encoding(df_test, enc)

X_train = edf_train.loc[:,edf_train.columns != 'pSat_Pa'].values
y_train = edf_train.loc[:, 'pSat_Pa'].values

model = LinearRegression().fit(X_train, y_train)

print("Model score on training data: ", model.score(X_train, y_train))