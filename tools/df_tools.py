
import pandas as pd
import numpy as np


def lookup_fn(df, key_row, key_col):

    return df.iloc[key_row][key_col]


def remove_row(df,row):
    df.drop([row], axis=0, inplace=True)
    return df

# Create a data folder in your current dir.
def SaveData(df, filename):

    df.to_csv("./data/yfinance_data/" + filename + ".csv")

def new_df_movement_list():

    df = pd.DataFrame({"ticker": [],
                       "nb_days": [],
                       "data_size": [],
                       "delta_%_h_l_5d": [],
                       "delta_%_o_c_5d": [],
                       "delta_%_o_c_1d": [],
                       "DTR_pred": [],
                       "DTR_best": [],
                       "DTR_dt_s": [],
                       "SVM_pred": [],
                       "SVM_best": [],
                       "SVM_dt_s": [],
                       "RF_pred": [],
                       "RF_best": [],
                       "RF_dt_s": [],
                       "ADABST_pred": [],
                       "ADABST_best": [],
                       "ADABST_dt_s": [],
                       "GRBST_pred": [],
                       "GRBST_best": [],
                       "GRBST_dt_s": [],
                       "GNaiveB_pred": [],
                       "GNaiveB_best": [],
                       "GNaiveB_dt_s": [],
                       "KN_pred": [],
                       "KN_best": [],
                       "KN_dt_s": [],
                       "XGBST_pred": [],
                       "XGBST_best": [],
                       "XGBST_dt_s": [],
                       "lGBM_pred": [],
                       "lGBM_best": [],
                       "lGBM_dt_s": [],
                       "LSTM": [],
                       })

    return df


def empty_df_movement_list(stock):

    df = pd.DataFrame({"ticker": [stock],
                       "nb_days": [0],
                       "data_size": [0.0],
                       "delta_%_h_l_5d": [0.0],
                       "delta_%_o_c_5d": [0.0],
                       "delta_%_o_c_1d": [0.0],
                       "DTR_pred": [0.0],
                       "DTR_best": [0.0],
                       "DTR_dt_s": [0.0],
                       "SVM_pred": [0.0],
                       "SVM_best": [0.0],
                       "SVM_dt_s": [0.0],
                       "RF_pred": [0.0],
                       "RF_best": [0.0],
                       "RF_dt_s": [0.0],
                       "ADABST_pred": [0.0],
                       "ADABST_best": [0.0],
                       "ADABST_dt_s": [0.0],
                       "GRBST_pred": [0.0],
                       "GRBST_best": [0.0],
                       "GRBST_dt_s": [0.0],
                       "GNaiveB_pred": [0.0],
                       "GNaiveB_best": [0.0],
                       "GNaiveB_dt_s": [0.0],
                       "KN_pred": [0.0],
                       "KN_best": [0.0],
                       "KN_dt_s": [0.0],
                       "XGBST_pred": [0.0],
                       "XGBST_best": [0.0],
                       "XGBST_dt_s": [0.0],
                       "lGBM_pred": [0.0],
                       "lGBM_best": [0.0],
                       "lGBM_dt_s": [0.0],
                       "LSTM": [0.0]
                       })

    return df