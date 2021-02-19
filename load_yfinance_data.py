import sys

# imports
import yfinance as yf
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
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

def getData_5day(ticker):

    # We can get data by our choice by giving days bracket
    #start_date= str("2017") + "-" + str("01") + "-" + str("01")

    nb_df_row = 0
    nb_days = 7
    today = date.today()

    start_date = today - timedelta(days=nb_days)
    data = pdr.get_data_yahoo(ticker, start=start_date, end=today)

    nb_df_row = len(data)
    print("len rows: ", nb_df_row)

    if nb_df_row == 7:
        data = remove_row(data,5)
        data = remove_row(data,6)
    if nb_df_row == 6:
        data = remove_row(data,5)

    print("len rows: ", nb_df_row)

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
    print(df.head())
    print(df['Symbol'].head())
    print(len(df['Symbol']))

    return df


def get_movment_list(df):

    movementlist = []

    for stock in df['Symbol']:

        df_stock_data = getData_5day(stock)
        print("stock: ",stock)

        low = float(10000)
        high = float(0)

        for low_value in df_stock_data["Low"]:
            if low_value < low:
                low = low_value

        for high_value in df_stock_data["High"]:
            if high_value > high:
                high = high_value

        deltapercent = 100 * (high - low) / low

        len_df = len(df_stock_data)

        Open = lookup_fn(df_stock_data, 0, "Open")
        Close = lookup_fn(df_stock_data, len_df - 1, "Close")

        deltaprice = 100 * (Close - Open) / Open

        print(stock + " " + str(deltapercent) + " " + str(deltaprice))
        pair = [stock, deltapercent, deltaprice]
        movementlist.append(pair)

    return movementlist


def get_data_finance():

    df_ticker_list = get_NASDAQ_ticker_list()

    movementlist = get_movment_list(df_ticker_list)

    return df_ticker_list
