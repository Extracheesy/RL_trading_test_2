# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import pandas as pd
import numpy as np

#from stockstats import StockDataFrame as Sdf
from load_yfinance_data import *


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

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
