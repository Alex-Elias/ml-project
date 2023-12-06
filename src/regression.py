import numpy as np
import matplotlib.pyplot as plt
from preprocess import *

# IMPORTS FOR REGRESSION RELATED OPERATIONS
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score as R2
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE

def do_cv(model, X_train, y_train):
    """ Do 5-fold CV on the given model. Returns the estimator which
    performed the best. """

    print("Performing cross validation...")

    score = cross_validate(model, 
                            X_train, 
                            y_train,
                            n_jobs=-1,
                            return_estimator=True,
                            scoring=('r2'))
    
    best_index = np.argmax(score['test_score'])
    print("Best score = ", score['test_score'][best_index])
    return score['estimator'][best_index]

def normalize_dataset(dataset):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(dataset)
    normalized_df = pd.DataFrame(normalized_data, columns=dataset.columns)
    return normalized_df

def write_to_file(y_pred, file_name):
    f = open(file_name, 'w')
    f.write("Id,target\n")
    for i in range(y_pred.shape[0]):
        f.write("{},{}\n".format(df_test.loc[i,"Id"], y_pred[i]))
    f.close()
    print("Write to file succesful.")



def sv_regression(x_train, y_train, x_test):
    regr = svm.SVR(kernel='rbf')
    estimator = regr.fit(x_train, y_train)
    fse = SelectFromModel(estimator=estimator,
                          threshold=1e-4,
                          prefit=True)
    fse = fse.fit(x_train, y_train)
    y_pred = fse.estimator_.predict(x_test)
    return y_pred

def rf_regression(x_train, y_train, x_test):
    regr = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=33)
    estimator = regr.fit(x_train, y_train)
    print("RFE importances: ", estimator.feature_importances_)
    fse = SelectFromModel(estimator, threshold=1e-4, prefit=True)
    xn_train = fse.transform(x_train)
    fse = fse.fit(xn_train, y_train)
    y_pred = fse.estimator_.predict(x_test)
    return y_pred

df_train = read_data("train")
df_test = read_data("test")

df_train = df_train.drop('Id',axis=1)   # remove ID column.

edf_train, enc = generate_encoding(df_train)    # encode the training data.
edf_test = apply_encoding(df_test.drop('Id', axis=1), enc)  # encode test data with the same encoder.

X_train = normalize_dataset(edf_train.loc[:,edf_train.columns != 'pSat_Pa'])
y_train = np.log10(edf_train.loc[:, 'pSat_Pa'])
X_test = normalize_dataset(edf_test)


y_pred = sv_regression(X_train, y_train, X_test)
#y_pred = rf_regression(X_train.values, y_train.values, X_test.values)


write_to_file(y_pred, "predictions.csv")