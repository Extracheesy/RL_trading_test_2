
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
                       "DTR": [],
                       "XGBOOST": [],
                       "lightGBM": [],
                       "LSTM": [],
                       })

    return df


def empty_df_movement_list(stock):

    df = pd.DataFrame({"ticker": [stock],
                       "nb_days": [0],
                       "data_size": [0],
                       "delta_%_h_l_5d": [0],
                       "delta_%_o_c_5d": [0],
                       "delta_%_o_c_1d": [0],
                       "DTR": [0],
                       "XGBOOST": ["0"],
                       "lightGBM": [],
                       "LSTM": [0]
                       })

    return df