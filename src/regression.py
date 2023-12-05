from preprocess import *
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score as R2
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def do_cv(model, X_train, y_train):
    """ Do 5-fold CV on the given model. Returns the estimator which
    performed the best. """

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

def write_to_file(y_test, y_pred, file_name):
    f = open(file_name, 'w')
    f.write("Id,target\n")
    for i in range(y_test.shape[0]):
        f.write("{},{}\n".format(df_test.loc[i,"Id"], y_pred[i]))
    f.close()

df_train = read_data("train")
df_test = read_data("test")

df_train = df_train.drop('Id',axis=1)   # remove ID column.

edf_train, enc = generate_encoding(df_train)    # encode the training data.
edf_test = apply_encoding(df_test.drop('Id', axis=1), enc)  # encode test data with the same encoder.

X_train = normalize_dataset(edf_train.loc[:,edf_train.columns != 'pSat_Pa'])
y_train = np.log10(edf_train.loc[:, 'pSat_Pa'])
x_test = normalize_dataset(edf_test)

#lr_model = LinearRegression()
#dr_model = DummyRegressor(strategy='median')
rf_model = RandomForestRegressor(n_jobs=-1, n_estimators=1000, random_state=2)


best_model = do_cv(rf_model, X_train, y_train)

predictor = best_model.fit(X_train, y_train)
y_pred = predictor.predict(x_test)

write_to_file(df_test, y_pred, "predictions.csv")