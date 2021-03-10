import sys
import os, fnmatch
import shutil

# imports
import yfinance as yf
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np

#from google.colab import files

from build_DT import *
from train_predict import *
from predictor_predict import *
import datetime
from datetime import date, timedelta

from yscraping import get_YAHOO_ticker_list

def lookup_fn(df, key_row, key_col):

    return df.iloc[key_row][key_col]

    #except IndexError:
    #return 0

def remove_row(df,row):
    df.drop([row], axis=0, inplace=True)
    return df

# Create a data folder in your current dir.
def SaveData(df, filename):

    df.to_csv("./data/yfinance_data/" + filename + ".csv")

def get_Data_5days(ticker):

    # We can get data by our choice by giving days bracket
    #start_date= str("2017") + "-" + str("01") + "-" + str("01")

    nb_df_row = 0
    nb_days = 9
    today = date.today()

    start_date = today - timedelta(days=nb_days)

    nb_try = 0
    while True:
        try:
            data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
            break
        except KeyError:
            print("I got a KeyError for: ", ticker)
            nb_try = nb_try + 1
            if (nb_try > 5):
                df_empty = pd.DataFrame({"ticker": [ticker],
                                         "nb_days": [0],
                                         "delta_%_h_l_5d": [0],
                                         "delta_%_o_c_5d": [0],
                                         "delta_%_o_c_1d": [0],
                                         "DT_results": [0],
                                         "RMSE": [0],
                                         "MAPE": [0],
                                         "Trend_Accuracy": [0],
                                         "data_size": [0]})
                return df_empty

    data['Date'] = data.index

    data = data.reset_index(drop=True)

    cols = ['Date'] + [col for col in data if col != 'Date']
    data = data[cols]
    data['Date'] = pd.to_datetime(data['Date'])

    # sort by datetime
    data.sort_values(by = 'Date', inplace = True, ascending = False)

    data = data.reset_index(drop=True)

    nb_df_row = len(data)
    #print("len rows: ", nb_df_row)

    if nb_df_row == 8:
        data = remove_row(data,5)
        data = remove_row(data,6)
        data = remove_row(data,7)
    if nb_df_row == 7:
        data = remove_row(data,5)
        data = remove_row(data,6)
    if nb_df_row == 6:
        data = remove_row(data,nb_df_row - 1)

    #print("len rows: ", nb_df_row)

    files = []
    dataname= ticker + "_" + str(today)
    files.append(dataname)
    SaveData(data, dataname)
    return data

def get_NASDAQ_ticker_list():

    # list all NASDAQ stocks
    url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    df = pd.read_csv(url, sep="|")

    # print("nb stocks",len(df['Symbol']))

    return df


def get_movment_list(stock):

    df_stock_data = get_Data_5days(stock)
    #print("stock: ",stock)

    if len(df_stock_data) == 1:
        df_stock_empty = pd.DataFrame({"ticker": [stock],
                                       "nb_days": [0],
                                       "delta_%_h_l_5d": [0],
                                       "delta_%_o_c_5d": [0],
                                       "delta_%_o_c_1d": [0],
                                       "DT_results": [0],
                                       "RMSE": [0],
                                       "MAPE": [0],
                                       "Trend_Accuracy": [0],
                                       "data_size": [0]})
        return df_stock_empty

    low = float(10000)
    high = float(0)

    for low_value in df_stock_data["Low"]:
        if low_value < low:
            low = low_value

    for high_value in df_stock_data["High"]:
        if high_value > high:
            high = high_value

    if low == 0:
        deltapercent = 0
    else:
        deltapercent = 100 * (high - low) / low

    len_df = len(df_stock_data)

    Open1d = lookup_fn(df_stock_data, 0, "Open")
    Open = lookup_fn(df_stock_data, len_df - 1, "Open")
    Close = lookup_fn(df_stock_data, 0, "Close")

    if Open == 0:
        deltaprice = 0
    else:
        deltaprice = 100 * (Close - Open) / Open

    if Open1d == 0:
        deltaprice1d = 0
    else:
        deltaprice1d = 100 * (Close - Open1d) / Open1d

    #print(stock + " " + str(deltapercent) + " " + str(deltaprice))

    df_movementlist = pd.DataFrame({"ticker": [stock],
                                    "nb_days": [len_df],
                                    "delta_%_h_l_5d": [round(deltapercent,2)],
                                    "delta_%_o_c_5d": [round(deltaprice,2)],
                                    "delta_%_o_c_1d": [round(deltaprice1d,2)],
                                    "DT_results": [0],
                                    "RMSE": [0],
                                    "MAPE": [0],
                                    "Trend_Accuracy": [0],
                                    "data_size": [0]})

    # pair = [stock, deltapercent, deltaprice]

    return df_movementlist




def get_DT_prediction(stock):

    df_stock_data = getData_5years(stock)
    stock_DT_predicct = process_decision_tree(df_stock_data,stock)

    return stock_DT_predicct



def get_data_finance(letter_filter):

    COMPUTE_MODEL = "COMPUTE_MODEL"
    #DATA_SOURCE = "NASDAQ"
    DATA_SOURCE = "SCRAPING"

    if DATA_SOURCE == "NASDAQ":
        df_ticker_list = get_NASDAQ_ticker_list()
        df_ticker_list.drop([len(df_ticker_list) - 1], axis=0, inplace=True)
    elif DATA_SOURCE == "SCRAPING":
        df_ticker_list = get_YAHOO_ticker_list("MIXED_DATA")
        #df_ticker_list = get_YAHOO_ticker_list("GAINER")
        #df_ticker_list = get_YAHOO_ticker_list("LOOSERS")
        #df_ticker_list = get_YAHOO_ticker_list("TRENDING")
        #df_ticker_list = get_YAHOO_ticker_list("ACTIVES")

    SaveData(df_ticker_list, "tickerlist.csv")

    # Calling DataFrame constructor
    df_movementlist = pd.DataFrame({"ticker": [],
                                    "nb_days": [],
                                    "delta_%_h_l_5d": [],
                                    "delta_%_o_c_5d": [],
                                    "delta_%_o_c_1d": [],
                                    "DT_results": [],
                                    "RMSE": [],
                                    "MAPE": [],
                                    "Trend_Accuracy": [],
                                    "data_size": []})

    if DATA_SOURCE == "NASDAQ":
        df_filtered_ticker_list = df_ticker_list[ (df_ticker_list['Test Issue'] == "N")]
    else:
        df_filtered_ticker_list = df_ticker_list

    global_start_stock = datetime.datetime.now()

    cpt = 0
    for stock in df_filtered_ticker_list['Symbol']:
        if(cpt < 0):
            cpt = cpt + 1
        else:
            if( stock.startswith(letter_filter) ):
            #if (stock.startswith("A") or stock.startswith("B") or stock.startswith("C")):
            #if( stock == "AACG" ): AACQ
            #if (stock == "BHFAN"):

                start_stock = datetime.datetime.now()
                print("start", stock)

                df_movementstocklist = get_movment_list(stock)

                DT_result , data_len, df_data_yf = process_decision_tree(stock)

                if (data_len < 2):
                    df_movementstocklist["DT_results"] = DT_result
                    df_movementstocklist["data_size"] = round(data_len / 253, 2)
                    df_movementstocklist["RMSE"] = 0
                    df_movementstocklist["MAPE"] = 0
                    df_movementstocklist["Trend_Accuracy"] = 0
                else:
                    df_movementstocklist["DT_results"] = DT_result
                    df_movementstocklist["data_size"] = round(data_len / 253, 2)

                    # Compute Model
                    if (COMPUTE_MODEL == "COMPUTE_MODEL"):
                        # MODEL selection not implemented
                        COMPUTE_MODEL = "COMPUTE_MODEL"

                    if (len(df_data_yf) > 402):
                        rmse, mape = train_model(stock, df_data_yf)

                        # Insert new row in dataframe
                        df_movementstocklist["RMSE"] = round(rmse, 2)
                        df_movementstocklist["MAPE"] = round(mape, 2)
                        TA = pred_predictor(stock, df_data_yf)
                    else:
                        TA = 0
                    df_movementstocklist["Trend_Accuracy"] = round(TA, 2)

                df_movementlist = df_movementlist.append(df_movementstocklist)

                end_stock = datetime.datetime.now()
                delta = end_stock - start_stock
                print("stock:",stock ," time: ",delta)

                cpt = cpt + 1
                if ((cpt % 5) == 0):
                    SaveData(df_movementlist, "movmentlist_tmp_" + str(cpt) + ".csv")

    today = date.today()
    #SaveData(df_movementlist, letter_filter + "_movmentlist_final_" + str(today) + ".csv")
    SaveData(df_movementlist, letter_filter + "_movmentlist_final_" + str(today))
    print("file to copy: ", "./data/yfinance_data/" + letter_filter + "_movmentlist_final_" + str(today) + ".csv" )
    shutil.copy( "./data/yfinance_data/" + letter_filter + "_movmentlist_final_" + str(today) + ".csv" , "../drive/MyDrive/colab_results")

    #files.download("./" + letter_filter + "_movmentlist_final_" + str(today) + ".csv")

    global_end_stock = datetime.datetime.now()
    delta = global_end_stock - global_start_stock
    print("consumed time: ", delta)

    return df_ticker_list
