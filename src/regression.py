from preprocess import *
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score as R2
import matplotlib.pyplot as plt

def do_cv(models, X_train, y_train, m_keys):
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
        print("Average: ", np.mean(cv_scores[i]["test_r2"]))

df_train = read_data("train")
df_test = read_data("test")

df_train = df_train.drop('Id',axis=1)

edf_train, enc = generate_encoding(df_train) 
edf_test = apply_encoding(df_test.drop('Id', axis=1), enc)

X_train = edf_train.loc[:,edf_train.columns != 'pSat_Pa'].values
y_train = np.log10(edf_train.loc[:, 'pSat_Pa'].values)
y_test = edf_test.values

#lr_model = LinearRegression()
#dr_model = DummyRegressor(strategy='median')
rf_model = RandomForestRegressor(n_jobs=-1, n_estimators=250, random_state=2)


#models = [rf_model]
#m_keys = ['RF']
#do_cv(models, X_train, y_train, m_keys)

predictor = rf_model.fit(X_train, y_train)
y_pred = predictor.predict(y_test)

f = open("predictions.csv", 'w')
f.write("Id,target\n")
for i in range(y_test.shape[0]):
    f.write("{},{}\n".format(df_test.loc[i,"Id"], y_pred[i]))

f.close()