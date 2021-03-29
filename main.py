# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys

sys.path.append("./load_yfinance/")
sys.path.append("./predict/")
sys.path.append("./scraping/")
sys.path.append("./init/")
sys.path.append("./DT/")
sys.path.append("./tools/")

from os_tools import *
from load_yfinance_data import *

import config

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

if __name__ == '__main__':

    print("This is the name of the script: ", sys.argv[0])
    print("Number of arguments: ", len(sys.argv))
    print("The arguments are: ", str(sys.argv))

    mk_directories()

    for filter in config.SELCTION:
        df_data_stock_list = get_data_finance(filter)

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
