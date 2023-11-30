from preprocess import *
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score as R2
import matplotlib.pyplot as plt

df_train = read_data("train")
df_test = read_data("test")

df_train = df_train.drop('Id',axis=1)
df_test = df_test.drop('Id', axis=1)

edf_train, enc = generate_encoding(df_train) 
edf_test = apply_encoding(df_test, enc)

print(edf_train)

X_train = edf_train.loc[:,edf_train.columns != 'pSat_Pa'].values
y_train = edf_train.loc[:, 'pSat_Pa'].values

lr_model = LinearRegression()
dr_model = DummyRegressor(strategy='median')
rf_model = RandomForestRegressor(n_jobs=-1)

models = [lr_model, dr_model, rf_model]
m_keys = ['LR', 'Dummy', 'RF']
cv_scores = []

for model in models:
    """ Perform 5-fold CV for all models in models list. """
    score = cross_validate(model, 
                           X_train, 
                           y_train,
                           scoring=('r2', 'neg_mean_squared_error'))
    cv_scores.append(score)

# print models and corresponding r2 scores.
for i, model in enumerate(models):
    print("Model: ", m_keys[i])
    print("R2 scores: ", cv_scores[i]["test_r2"])