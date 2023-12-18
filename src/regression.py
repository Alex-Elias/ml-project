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
    
    print("SCORE = ", np.mean(score['test_score']))
    

def normalize_dataset(dataset):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(dataset)
    normalized_df = pd.DataFrame(normalized_data, columns=dataset.columns)
    return normalized_df

def normalize_dataset_2(dataset, columns_to_normalize):
    # Create a copy of the original dataset to avoid modifying the input dataset
    normalized_df = dataset.copy()
    # Select only the specified columns for normalization
    selected_data = dataset[columns_to_normalize]
    # Apply StandardScaler to the selected columns
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(selected_data)
    # Replace the original values with the normalized values in the copy
    normalized_df[columns_to_normalize] = normalized_data

    return normalized_df

def write_to_file(y_pred, file_name):
    df_test = pd.read_csv("data/test.csv")
    print("DF_TEST SHAPE: ", df_test.shape)
    print("Y_PRED SHAPE: ", y_pred.shape)
    f = open(file_name, 'w')
    f.write("Id,target\n")
    for i in range(y_pred.shape[0]):
        f.write("{},{}\n".format(df_test.loc[i,"Id"], y_pred[i]))
    f.close()
    print("Write to file succesful.")

def take_outliers(x_train, y_train, y_pred):
    abs_errors = calc_abs_error(y_pred, y_train)
    x = range(len(abs_errors))
    #plt.plot(x, abs_errors)
    #plt.show()

    outliers = find_outliers_after_regr(abs_errors, 1.4)
    print("Taking out ",len(outliers),"outliers...")
    
    x_train_we = drop_rows_by_indexes(x_train, outliers)
    y_train_we = drop_rows_by_indexes(y_train, outliers)

    return x_train_we, y_train_we

def take_outliers_priori(edf_train, y_train, y_pred):
    abs_errors = calc_abs_error(y_pred, y_train)
    x = range(len(abs_errors))
    #plt.plot(x, abs_errors)
    #plt.show()

    outliers = find_outliers_after_regr(abs_errors, 8)
    print("Taking out ",len(outliers),"outliers...")
    
    edf_train_we = drop_rows_by_indexes(edf_train, outliers)
    y_train_we = drop_rows_by_indexes(y_train, outliers)

    return edf_train_we, y_train_we

def take_outliers_z(df, feature, threshold, edf_train, y_train):
    # Calculate z-scores for the specified feature
    z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
    # Find indexes where z-score exceeds the threshold
    outliers = np.where(z_scores > threshold)[0]
    print("Taking out ",len(outliers),"outliers...")
    
    edf_train_we = drop_rows_by_indexes(edf_train, outliers)
    y_train_we = drop_rows_by_indexes(y_train, outliers)

    return edf_train_we, y_train_we

def sv_regression(x_train, y_train, x_test):
    regr = svm.SVR(C=4.194044473072977, kernel='rbf',gamma = "scale", coef0 = 0.14897658506713812, tol = 0.001143941929309215, epsilon = 0.2498414560389054)
    estimator = regr.fit(x_train, y_train)
    fse = SelectFromModel(estimator=estimator,
                          threshold=1e-6,
                          prefit=True)
    fse = fse.fit(x_train, y_train)
    
    do_cv(fse.estimator_, x_train, y_train)

    y_pred = fse.estimator_.predict(x_test)
    return y_pred

def calc_abs_error(y_pred,y_actual):
    return np.abs(y_pred-y_actual)

def drop_rows_by_indexes(dataframe, indexes):
    """
    Drops rows from the DataFrame based on the given list of indexes.
    """
    new_dataframe = dataframe.drop(index=indexes)
    return new_dataframe

def find_outliers_after_regr(errors,limit):
    indexes = [index for index, value in enumerate(errors) if value > limit]
    return indexes

def plot_histograms(df, features_to_plot):
    # Select only the specified features
    selected_features = df[features_to_plot]

    # Plot histograms for each selected feature
    for feature in selected_features.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(selected_features[feature], bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Histogram for {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

#df_train = pd.read_csv("data/train.csv")
#df_test = pd.read_csv("data/test.csv")

#df_train = df_train.drop('Id',axis=1)   # remove ID column.

# Multiplication of columns
#MW * NumOfO * NUMOFN
#df_train["MW_NumOfO_NumOfN"] = df_train["MW"] * df_train["NumOfO"]  * df_train["NumOfN"]
#df_test["MW_NumOfO_NumOfN"] = df_test["MW"] * df_test["NumOfO"] * df_test["NumOfN"]
#MW * NumOfO
#df_train["MW_NumOfO"] = df_train["MW"] * df_train["NumOfO"] 
#df_test["MW_NumOfO"] = df_test["MW"] * df_test["NumOfO"]
#MW * NumOfN
#df_train["MW_NumOfN"] = df_train["MW"] * df_train["NumOfN"] 
#df_test["MW_NumOfN"] = df_test["MW"] * df_test["NumOfN"]
#MW * NumOfAtoms
#df_train["MW_NumOfAtoms"] = df_train["MW"] * df_train["NumOfAtoms"]
#df_test["MW_NumOfAtoms"] = df_test["MW"] * df_test["NumOfAtoms"]
#MW * NumOfO * NumOfN * NumofAtoms
#df_train["MW_NumOfO_NumOfN_NumOfAtoms"] = df_train["MW"] * df_train["NumOfO"]  * df_train["NumOfN"] * df_train["NumOfAtoms"]
#df_test["MW_NumOfO_NumOfN_NumOfAtoms"] = df_test["MW"] * df_test["NumOfO"] * df_test["NumOfN"] * df_test["NumOfAtoms"]
#NumOfAtoms * NumofC
#df_train["NumOfAtoms_NumOfC"] = df_train["NumOfAtoms"] * df_train["NumOfC"] 
#df_test["NumOfAtoms_NumOfC"] = df_test["NumOfAtoms"] * df_test["NumOfC"]

#edf_train, enc = generate_encoding(df_train)    # encode the training data.
#edf_test = apply_encoding(df_test.drop('Id', axis=1), enc)  # encode test data with the same encoder.

#X_train = normalize_dataset_2(edf_train.loc[:,edf_train.columns != 'pSat_Pa'],["MW","MW_NumOfO_NumOfN","MW_NumOfO","MW_NumOfN","NumOfConf","NumOfConfUsed","NumOfAtoms","NumOfC","NumOfO","NumHBondDonors","carbonylperoxyacid","NumOfN","ketone","ester","C.C..non.aromatic."])
#y_train = np.log10(edf_train.loc[:, 'pSat_Pa'])
#X_test = normalize_dataset_2(edf_test,["MW","MW_NumOfO_NumOfN","MW_NumOfO","MW_NumOfN","NumOfConf","NumOfConfUsed","NumOfAtoms","NumOfC","NumOfO","NumHBondDonors","carbonylperoxyacid","NumOfN","ketone","ester","C.C..non.aromatic."])

#edf_train_we , y_train_we = take_outliers_priori(edf_train,y_train,sv_regression(X_train,y_train,X_train))
#X_train_we = normalize_dataset_2(edf_train_we.loc[:,edf_train_we.columns != 'pSat_Pa'],["MW","MW_NumOfO_NumOfN","MW_NumOfO","MW_NumOfN","NumOfConf","NumOfConfUsed","NumOfAtoms","NumOfC","NumOfO","NumHBondDonors","carbonylperoxyacid","NumOfN"])
#y_pred = sv_regression(X_train_we, y_train_we, X_test)

#X_train_we,y_train_we = take_outliers_z(df_train,'pSat_Pa',3,X_train,y_train)
#y_pred = sv_regression(X_train_we, y_train_we, X_test)

#y_pred = sv_regression(X_train, y_train, X_test)
#write_to_file(y_pred, "predictions.csv")

#plot_histograms(df_train,df_train.columns)