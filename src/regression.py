from preprocess import *
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score as R2

df_train = read_data("train")
df_test = read_data("test")

edf_train, enc = generate_encoding(df_train) 
edf_test = apply_encoding(df_test, enc)

X_train = edf_train.loc[:,edf_train.columns != 'pSat_Pa'].values
y_train = edf_train.loc[:, 'pSat_Pa'].values

lr_model = LinearRegression()
dr_model = DummyRegressor(strategy='median')

models = [lr_model, dr_model]
cv_scores = []

for model in models:
    """ Perform 5-fold CV for all models in models list. """
    score = cross_validate(model, 
                           X_train, 
                           y_train)
    cv_scores.append(score)

for i, model in enumerate(models):
    print("Model ", i)
    print(cv_scores[i]['test_score'])