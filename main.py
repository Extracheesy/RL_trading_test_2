# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import os, fnmatch
import pandas as pd
import numpy as np



#from stockstats import StockDataFrame as Sdf
from load_yfinance_data import *


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def clear_data_directory():

    listOfFilesToRemove = os.listdir('./data/yfinance_data/')
    pattern = "*.csv"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern):
            print("remove : ",entry)
            os.remove("./data/yfinance_data/" + entry)

    listOfFilesToRemove = os.listdir('./data/yfinance_data_DT/')
    pattern = "*.csv"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern):
            print("remove : ", entry)
            os.remove("./data/yfinance_data_DT/" + entry)

    listOfFilesToRemove = os.listdir('./data/yfinance_data_predict/')
    pattern = "*.csv"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern):
            print("remove : ", entry)
            os.remove("./data/yfinance_data_predict/" + entry)

    listOfFilesToRemove = os.listdir('./data/yfinance_output/')
    pattern = "*.csv"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern):
            print("remove : ", entry)
            os.remove("./data/yfinance_output/" + entry)

def clear_model_directory():

    listOfFilesToRemove = os.listdir('./data/yfinance_model/')
    pattern_pkl = "*.pkl"
    pattern_gz = "*.gz"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern_pkl):
            print("remove : ",entry)
            os.remove("./data/yfinance_model/" + entry)
        else:
            if fnmatch.fnmatch(entry, pattern_gz):
                print("remove : ", entry)
                os.remove("./data/yfinance_model/" + entry)

# Press the green button in the gutter to run the script.

def mk_directories():

    if not os.path.exists("./data"):
        os.makedirs("./data")

    if not os.path.exists("./data/yfinance_data/"):
        os.makedirs("./data/yfinance_data/")

    if not os.path.exists("./data/yfinance_data_DT/"):
        os.makedirs("./data/yfinance_data_DT/")

    if not os.path.exists("./data/yfinance_data_model/"):
        os.makedirs("./data/yfinance_data_model/")

    if not os.path.exists("./data/yfinance_data_predict/"):
        os.makedirs("./data/yfinance_data_predict/")

    if not os.path.exists("./data/yfinance_output/"):
        os.makedirs("./data/yfinance_output/")


if __name__ == '__main__':

    COMPUTE_MODEL = "COMPUTE_MODEL"

    if (COMPUTE_MODEL == "COMPUTE_MODEL"):
        mk_directories()
        clear_model_directory()
        clear_data_directory()

    df_data_stock_list = get_data_finance()

    """
    fill_and_select_stock()

    creat_env()

    build_and_save_model()

    split_data()

    run_trading_strategy()

    strategy_measure_evaluation()
    """
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
