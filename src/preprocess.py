from sklearn.preprocessing import OrdinalEncoder
import pandas as pd 
import numpy as np

def encode_data(data: pd.DataFrame):
    enc = OrdinalEncoder(encoded_missing_value=-1)
    vals = data.loc[:, 'parentspecies'].values
    sp_values = enc.fit_transform(vals.reshape(-1, 1))
    
    return sp_values

def process_data(data: pd.DataFrame, sp_values = None, istest = True):
    X = data.loc[1:, data.columns != "pSat_Pa"].values
    if not istest:
        y = data.loc[1:, "pSat_Pa"].values
    else:
        y = None
    rows = data.shape[0] - 1

    str_clm_index = 0
    for i, column in enumerate(X[0,:]):
        if isinstance(column, str):
            str_clm_index = i
    
    if not istest:
        for row in range(rows):
            X[row, str_clm_index] = sp_values[row]

    return X, y
    
def read_data(file_name: str, istest=False):
    '''
        Read data given the filename file_name. If data is testing
        data, then it has no pSat_Pa column and encoding it unnecessary.
        For training data, column "parentspecies" is encoded with missing values as
        -1. In case of testing data, the function returns only one array of data. 
        In case of training data, two arrays are returned: X, y.
    '''
    dataframe = pd.read_csv("data/{}.csv".format(file_name))
    
    if not istest:
        encode = encode_data(dataframe)
        return process_data(dataframe, encode, istest)
    else:
        return process_data(dataframe, None, istest)

