from preprocess import read_data
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score as R2

X, y = read_data("train")

print(X.shape)
print(y.shape)

model = LinearRegression().fit(X, y)