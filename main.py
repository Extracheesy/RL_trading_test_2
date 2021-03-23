# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys

sys.path.append("./load_yfinance/")
sys.path.append("./predict/")
sys.path.append("./scraping/")
sys.path.append("./init/")
sys.path.append("./DT/")

from load_yfinance_data import *
import config

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
    else:
        shutil.rmtree("./data")
        print("remove : ./data/*")
        os.makedirs("./data")

    if not os.path.exists("./data/yfinance_data/"):
        os.makedirs("./data/yfinance_data/")

    if not os.path.exists("./data/yfinance_data_DT/"):
        os.makedirs("./data/yfinance_data_DT/")

    if not os.path.exists("./data/yfinance_model/"):
        os.makedirs("./data/yfinance_model/")

    if not os.path.exists("./data/yfinance_data_predict/"):
        os.makedirs("./data/yfinance_data_predict/")

    if not os.path.exists("./data/yfinance_output/"):
        os.makedirs("./data/yfinance_output/")


if __name__ == '__main__':

    mk_directories()
    #clear_model_directory()
    #clear_data_directory()

    section_1 = ["GAINERS"]
    section_2 = ["TRENDING"]
    section_3 = ["CAC40"]
    section_4 = ["DJI"]
    section_5 = ["NASDAQ"]
    section_6 = ["ACTIVES"]
    section_7 = ["LOOSERS"]
    section_8 = ["DAX"]
    section_9 = ["SP500"]
    section_all = ["ALL"]

    print("This is the name of the script: ", sys.argv[0])
    print("Number of arguments: ", len(sys.argv))
    print("The arguments are: ", str(sys.argv))

    if (str(sys.argv[1]) == "section_1"):
        for filter in section_1:
            df_data_stock_list = get_data_finance(filter)
    if (str(sys.argv[1]) == "section_2"):
        for filter in section_2:
            df_data_stock_list = get_data_finance(filter)
    if (str(sys.argv[1]) == "section_3"):
        for filter in section_3:
            df_data_stock_list = get_data_finance(filter)
    if (str(sys.argv[1]) == "section_4"):
        for filter in section_4:
            df_data_stock_list = get_data_finance(filter)
    if (str(sys.argv[1]) == "section_5"):
        for filter in section_5:
            df_data_stock_list = get_data_finance(filter)
    if (str(sys.argv[1]) == "section_6"):
        for filter in section_6:
            df_data_stock_list = get_data_finance(filter)
    if (str(sys.argv[1]) == "section_7"):
        for filter in section_7:
            df_data_stock_list = get_data_finance(filter)
    if (str(sys.argv[1]) == "section_8"):
        for filter in section_8:
            df_data_stock_list = get_data_finance(filter)
    if (str(sys.argv[1]) == "section_9"):
        for filter in section_9:
            df_data_stock_list = get_data_finance(filter)
    if (str(sys.argv[1]) == "section_all"):
        for filter in section_all:
            df_data_stock_list = get_data_finance(filter)

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
