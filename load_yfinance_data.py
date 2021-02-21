import sys

# imports
import yfinance as yf
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np

from build_DT import *
import datetime
from datetime import date, timedelta

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
    nb_days = 8
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
                                         "deltapercent": [0],
                                         "deltaprice": [0],
                                         "DT_results": [0],
                                         "data_size": [0]})
                return df_empty

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
        data = remove_row(data,5)

    #print("len rows: ", nb_df_row)

    files = []
    dataname= ticker + "_" + str(today)
    files.append(dataname)
    SaveData(data, dataname)
    return data

"""
# High risers:
def lookup_stockinfo(thestock):
  try
    return thestock.info
  except IndexError:
    return 0

cutoff=float(80)
for entry in movementlist:
  if entry[2]>cutoff:
    print("\n"+ str(entry))
    thestock = yf.Ticker(str(entry[0]))
    if entry[0]=='HJLIW':
      print("no info")
    else:
      a = lookup_stockinfo(thestock)
if a == 0:
        print("no info")
      else:
        if a is None:
          print("no info")
        else:
          if a == "":
            print("no")
          else:
            print(a)
            print('Up '+ str(entry[2]) + "%")
            print(str(a['sector']))
            print(str(a['longBusinessSummary']))
            print("year high "+ str(a['fiftyTwoWeekHigh']))

"""


def get_NASDAQ_ticker_list():

    # list all NASDAQ stocks
    url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    df = pd.read_csv(url, sep="|")
    #print(df.head())
    #print(df['Symbol'].head())
    print("nb stocks",len(df['Symbol']))

    return df


def get_movment_list(stock):

    df_stock_data = get_Data_5days(stock)
    #print("stock: ",stock)

    if len(df_stock_data) == 1:
        df_stock_empty = pd.DataFrame({"ticker": [stock],
                                       "nb_days": [0],
                                       "deltapercent": [0],
                                       "deltaprice": [0]})
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

    Open = lookup_fn(df_stock_data, 0, "Open")
    Close = lookup_fn(df_stock_data, len_df - 1, "Close")

    if Open == 0:
        deltaprice = 0
    else:
        deltaprice = 100 * (Close - Open) / Open

    #print(stock + " " + str(deltapercent) + " " + str(deltaprice))

    df_movementlist = pd.DataFrame({"ticker": [stock],
                                    "nb_days": [len_df],
                                    "deltapercent": [deltapercent],
                                    "deltaprice": [deltaprice],
                                    "DT_results": [0],
                                    "data_size": [0]})

    # pair = [stock, deltapercent, deltaprice]

    return df_movementlist




def get_DT_prediction(stock):

    df_stock_data = getData_5years(stock)
    stock_DT_predicct = process_decision_tree(df_stock_data,stock)

    return stock_DT_predicct



def get_data_finance():

    df_ticker_list = get_NASDAQ_ticker_list()
    df_ticker_list.drop([len(df_ticker_list) - 1], axis=0, inplace=True)

    SaveData(df_ticker_list, "tickerlist.csv")

    # Calling DataFrame constructor
    df_movementlist = pd.DataFrame({"ticker": [],
                                    "nb_days": [],
                                    "deltapercent": [],
                                    "deltaprice": [],
                                    "DT_results": [],
                                    "data_size": []})

    df_filtered_ticker_list = df_ticker_list[ (df_ticker_list['Test Issue'] == "N")]

    for stock in df_filtered_ticker_list['Symbol']:

            start_stock = datetime.datetime.now()

            df_movementstocklist = get_movment_list(stock)

            DT_result , data_len = process_decision_tree(stock)

            df_movementstocklist["DT_results"] = DT_result
            df_movementstocklist["data_size"] = round(data_len / 253, 2)

            df_movementlist = df_movementlist.append(df_movementstocklist)

            end_stock = datetime.datetime.now()
            delta = end_stock - start_stock
            print("stock:",stock ," time: ",delta)

    SaveData(df_movementlist, "movmentlist.csv")

    return df_ticker_list
