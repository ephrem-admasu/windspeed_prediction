import numpy as np
import pandas as pd
# import FIF
import warnings

warnings.filterwarnings('ignore')


def impute_missing_values(series, k=3):
    """Impute missing values in a series using neighboring average.

    Args:
        series (pd.Series): The series with missing values.
        k (int): The number of neighboring values to use for imputation.

    Returns:
        pd.Series: The series with imputed values.
    """

    # Find the indices of the missing values.
    missing_indices = series[series.isna()].index

    # For each missing value, impute it with the average of its k nearest neighbors.
    for index in missing_indices:
        neighbors = None
        if k >= index:
            neighbors = series[0:index + k].dropna()
        else:
            neighbors = series[index-k:index + k].dropna()
        series.loc[index] = neighbors.mean()

    return series


def create_features(sequence, n_steps):
    """ Create a dataframe from a series with n_steps lags """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
		# check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def split_data(series, tr_len):
    # series = np.array(series.tolist())
    val_len = (len(series) - tr_len) // 2
    tr_series = series[:tr_len]
    
    val_series = series[tr_len:(tr_len+val_len)]
    test_series = series[(tr_len+val_len):]
    
    return tr_series, val_series, test_series


'''
def fif_analysis(y, create_df = False, df_name = None):
    fif=FIF.FIF()
    fif.run(y)
    if create_df:
        imfs = []
        for i in range(fif.data['IMC'].shape[0]):
            imfs.append(fif.data['IMC'][i, :])
        
        df = pd.DataFrame()
        for i in range(len(imfs)):
            df['IMC-' + str(i+1)] = fif.data['IMC'][i, :] 
        name = df_name  if df_name else 'decomposed_clean'
        df.to_csv(name + '.csv', index=False)   

    return fif
'''

from dtwi_analyzer import find_gaps, edtwbi
def impute_dtwi(arr):
    arr = pd.Series(arr).fillna(value=0).values

    filled_arr = arr.copy()
    
    gaps = find_gaps(arr)
    
    for gap in gaps:
        if gap[1] - gap[0] < 0.3*len(arr):
            filled_arr[gap[0]:gap[1]] = edtwbi(arr, gap[0], gap[1])
    
    return filled_arr

from statsmodels.tsa.stattools import adfuller

def check_stationarity(series):
    # Copied from https://machinelearningmastery.com/time-series-data-stationary-python/
    
    result = adfuller(series.values)
    
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")