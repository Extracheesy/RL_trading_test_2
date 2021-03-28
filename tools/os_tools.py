import fnmatch
import sys
import os
import config
import shutil


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

def mk_directories():

    if not os.path.exists("./data"):
        os.makedirs("./data")
    else:
        clear_data_directory()
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