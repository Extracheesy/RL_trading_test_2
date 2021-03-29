from datetime import date
from datetime import timedelta
import config
import pandas as pd
import numpy as np
from df_tools import new_df_movement_list
from df_tools import empty_df_movement_list
from pandas_datareader import data as pdr
from df_tools import remove_row
from df_tools import lookup_fn

def get_NASDAQ_ticker_list():

    # list all NASDAQ stocks
    url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    df = pd.read_csv(url, sep="|")

    return df

def get_Data_5days(ticker):

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
                df_empty = pd.DataFrame()
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

    if nb_df_row == 8:
        data = remove_row(data,5)
        data = remove_row(data,6)
        data = remove_row(data,7)
    if nb_df_row == 7:
        data = remove_row(data,5)
        data = remove_row(data,6)
    if nb_df_row == 6:
        data = remove_row(data,nb_df_row - 1)

    files = []
    dataname= ticker + "_" + str(today)
    files.append(dataname)

    if(config.SAVE_5DAY_DATA == True):
        SaveData(data, dataname)

    return data


def get_Data_5years(ticker):

    # We can get data by our choice by giving days bracket
    #start_date= str("2017") + "-" + str("01") + "-" + str("01")

    nb_years = config.LOAD_YEARS
    today = date.today()

    start_date = today - timedelta(weeks=nb_years)

    nb_try = 0
    while True:
        try:
            data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
            break
        except KeyError:
            print("I got a KeyError for: ", ticker)
            nb_try = nb_try + 1
            if (nb_try > 5):
                df_empty = pd.DataFrame()
                return df_empty

    return data


def get_movment_list_5D(stock):

    df_stock_data = get_Data_5days(stock)

    if len(df_stock_data) == 0:
        df_stock_empty = empty_df_movement_list(stock)
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

    #df_movementlist = new_df_movement_list()
    df_movementlist = empty_df_movement_list(stock)
    df_movementlist["ticker"][0] = stock
    df_movementlist["nb_days"][0] = len_df
    df_movementlist["delta_%_h_l_5d"][0] = round(deltapercent,2)
    df_movementlist["delta_%_o_c_5d"][0] = round(deltaprice,2)
    df_movementlist["delta_%_o_c_1d"][0] = round(deltaprice1d,2)

    return df_movementlist