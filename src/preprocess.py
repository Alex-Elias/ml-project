from sklearn.preprocessing import OrdinalEncoder
import pandas as pd 
import numpy as np

def generate_encoding(data: pd.DataFrame):
    """
        Generates ordinal encoding from given dataframe object.
        Object must contain parentspecies column. This is applied
        to the training data at the very beginning of the 
        preprocessing. 
        
        Returns copy of the training data as pd.Dataframe and
        the encoder which can be then used to apply encoding to
        the testing data.
    """
    enc = OrdinalEncoder(encoded_missing_value=-1)
    vals = data.loc[:, 'parentspecies'].values
    sp_values = enc.fit_transform(vals.reshape(-1, 1))
    
    res = data.copy(deep=True)
    res.loc[:, 'parentspecies'] = sp_values

    return res, enc

def apply_encoding(data: pd.DataFrame, encoder: OrdinalEncoder) -> pd.DataFrame:
    """
        Applies the given OrdinalEncoder to the data. After
        the encoding has been generated, this function can be used
        to apply the same encoding to the testing data.

        Returns a copy of the input data with parentspecies column
        altered using the given encoding.
    """
    res = data.copy(deep=True)
    res.loc[:, 'parentspecies'] = encoder.transform(data
                                                    .loc[:, 'parentspecies']
                                                    .values.reshape(-1, 1))
    return res
    
def read_data(file_name: str) -> pd.DataFrame:
    '''
        Read datafile with the given file_name.
        Returns pd.Dataframe object.
    '''
    dataframe = pd.read_csv("data/{}.csv".format(file_name))
    return dataframe

