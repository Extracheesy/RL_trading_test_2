import sys
import os, fnmatch
import shutil

# imports
import yfinance as yf
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np

from main_DT import *
from train_predict import *
from predictor_predict import *
import datetime
import config
from datetime import date, timedelta
from pre_process_indicator import pre_process_indictor_data

from yscraping import get_YAHOO_ticker_list
from df_tools import *
from load_data import *


def get_DT_prediction(stock):

    df_stock_data = getData_5years(stock)
    stock_DT_predicct = process_decision_tree(df_stock_data,stock)

    return stock_DT_predicct

def get_data_finance(filter):

    if config.DATA_SOURCE == "READ_CSV_FROM_DATABASE":
        df_ticker_list = pd.read_csv("./database/tickerlist_2021-03-21.csv")
    elif config.DATA_SOURCE == "SCRAPING":
        df_ticker_list = get_YAHOO_ticker_list()

    if filter != "ALL":
        df_ticker_list = df_ticker_list[df_ticker_list['Type'] == filter]

    SaveData(df_ticker_list, "tickerlist.csv")
    df_ticker_list = df_ticker_list.drop_duplicates(subset=['Symbol'])

    # Calling DataFrame constructor
    df_movement_list = new_df_movement_list()

    global_start_stock = datetime.datetime.now()

    cpt = 0
    for stock in df_ticker_list['Symbol']:
        if(cpt < 0):
            cpt = cpt + 1
        else:
            if(config.DEBUG_FORCE_STOCK == True):
                stock = "AAPL"

            start_stock = datetime.datetime.now()
            print("start", stock," time: ",start_stock)

            df_movement_stock_list = get_movment_list_5D(stock)
            df_full_data_stock = get_Data_5years(stock)

            # if not enough data available in regard with requested Stock
            if(len(df_full_data_stock) < 100):
                print("Not enough data available in regard with requested Stock: ",stock)
            else:

                df_movement_stock_list["data_size"] = round(len(df_full_data_stock) / 253, 2)

                df_full_data_stock = pre_process_indictor_data(stock, df_full_data_stock)

                if (config.COMPUTE_DT == True):
                    df_movement_stock_list = run_DT_prediction(df_full_data_stock, df_movement_stock_list)

                # Compute Model
                if (config.COMPUTE_PREDICT_MODEL == True):
                    if (len(df_full_data_stock) > 402):
                        rmse, mape, lstm_model, lstm_scaler = train_model(stock, df_full_data_stock)

                        # Insert new row in dataframe
                        TA = pred_predictor(stock, df_full_data_stock, lstm_model, lstm_scaler)
                    else:
                        TA = 0
                else:
                    TA = 0
                df_movement_stock_list["LSTM"] = round(TA, 2)

            df_movement_list = df_movement_list.append(df_movement_stock_list)

            end_stock = datetime.datetime.now()
            delta = end_stock - start_stock
            print("stock:",stock ," time: ",delta)

            cpt = cpt + 1
            if ((cpt % 5) == 0):
                SaveData(df_movement_list, "movmentlist_tmp_" + str(cpt) + ".csv")

    today = date.today()
    #SaveData(df_movement_list, filter + "_movmentlist_final_" + str(today) + ".csv")
    SaveData(df_movement_list, filter + "_movmentlist_final_" + str(today))
    print("file to copy: ", "./data/yfinance_data/" + filter + "_movmentlist_final_" + str(today) + ".csv" )

    if (config.COLAB == True):
        shutil.copy( "./data/yfinance_data/" + filter + "_movmentlist_final_" + str(today) + ".csv" , "../drive/MyDrive/colab_results")

    #files.download("./" + filter + "_movmentlist_final_" + str(today) + ".csv")

    global_end_stock = datetime.datetime.now()
    delta = global_end_stock - global_start_stock
    print("consumed time: ", delta)

    return df_ticker_list
