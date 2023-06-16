import numpy as np
import pandas as pd
import FIF
import warnings

warnings.filterwarnings('ignore')


def create_features(series, m, hmin = 10):
    h = hmin//10
    dataframe = pd.DataFrame()

    for i in range(m+h, h, -1):
        # if i > h:
        dataframe['t-'+str(i-h)] = series.shift(i).values
    dataframe['t'] = series.values
    dataframe = dataframe[(m+h):]
    return dataframe

def split_data(arr, tr_len):
    total_len = arr.shape[0]
    val_len = (total_len - tr_len) // 2
    tr_arr = arr[:tr_len]
    val_arr = arr[tr_len:tr_len+val_len]
    tes_arr = arr[tr_len+val_len:]
    return tr_arr, val_arr, tes_arr


def fif_analysis(y, create_df = False):
    fif=FIF.FIF()
    fif.run(y)
    if create_df:
        imfs = []
        for i in range(fif.data['IMC'].shape[0]):
            imfs.append(fif.data['IMC'][i, :])
        
        df = pd.DataFrame()
        for i in range(imfs):
            df['IMC-' + str(i+1)] = fif.data['IMC'][i, :] 

        df.to_csv('decomposed_clean.csv', index=False)   

    return fif
