import os
import re
from selenium import webdriver
import pandas as pd

def get_list_gainers(driver):

    driver.get('https://finance.yahoo.com/gainers?offset=0&count=100')
    html_src_1 = driver.page_source
    driver.get('https://finance.yahoo.com/gainers?count=100&offset=100')
    html_src_2 = driver.page_source

    html_src_str_1 = str(html_src_1)
    html_src_str_1 = html_src_str_1.replace("{",'\n')
    html_src_str_1 = html_src_str_1.replace("}",'\n')

    html_src_str_2 = str(html_src_2)
    html_src_str_2 = html_src_str_2.replace("{",'\n')
    html_src_str_2 = html_src_str_2.replace("}",'\n')

    match_1 = re.findall(r'"symbol":".*","shortName":', html_src_str_1)
    match_2 = re.findall(r'"symbol":".*","shortName":', html_src_str_2)

    list_gainers = []

    for i in range(0,len(match_1),1):
        tmp_string = match_1[i][10:]
        size = len(tmp_string)
        string = tmp_string[: size - 14]
        list_gainers.append(string)

    for i in range(0,len(match_2),1):
        tmp_string = match_2[i][10:]
        size = len(tmp_string)
        string = tmp_string[: size - 14]
        list_gainers.append(string)

    return list_gainers

def get_list_losers(driver):

    driver.get('https://finance.yahoo.com/losers?offset=0&count=100')
    html_src_1 = driver.page_source
    driver.get('https://finance.yahoo.com/losers?count=100&offset=100')
    html_src_2 = driver.page_source

    html_src_str_1 = str(html_src_1)
    html_src_str_1 = html_src_str_1.replace("{",'\n')
    html_src_str_1 = html_src_str_1.replace("}",'\n')

    html_src_str_2 = str(html_src_2)
    html_src_str_2 = html_src_str_2.replace("{",'\n')
    html_src_str_2 = html_src_str_2.replace("}",'\n')

    match_1 = re.findall(r'"symbol":".*","shortName":', html_src_str_1)
    match_2 = re.findall(r'"symbol":".*","shortName":', html_src_str_2)

    list_losers = []

    for i in range(0, len(match_1), 1):
        tmp_string = match_1[i][10:]
        size = len(tmp_string)
        string = tmp_string[: size - 14]
        list_losers.append(string)

    for i in range(0, len(match_2), 1):
        tmp_string = match_2[i][10:]
        size = len(tmp_string)
        string = tmp_string[: size - 14]
        list_losers.append(string)

    return list_losers

def get_list_trending_tickers(driver):

    driver.get('https://finance.yahoo.com/trending-tickers')
    html_src_1 = driver.page_source

    html_src_str_1 = str(html_src_1)
    html_src_str_1 = html_src_str_1.replace("{",'\n')
    html_src_str_1 = html_src_str_1.replace("}",'\n')

    match_1 = re.findall(r'"YFINANCE:.*","fallbackCategory":', html_src_str_1)
    tmp_string = match_1[0][10:]
    size = len(tmp_string)
    string = tmp_string[: size - 21]
    list_trending_tickers = string.split(",")

    return list_trending_tickers

def get_list_most_actives(driver):

    driver.get('https://finance.yahoo.com/most-active?offset=0&count=100')
    html_src_1 = driver.page_source
    driver.get('https://finance.yahoo.com/most-active?count=100&offset=100')
    html_src_2 = driver.page_source

    html_src_str_1 = str(html_src_1)
    html_src_str_1 = html_src_str_1.replace("{", '\n')
    html_src_str_1 = html_src_str_1.replace("}", '\n')

    html_src_str_2 = str(html_src_2)
    html_src_str_2 = html_src_str_2.replace("{", '\n')
    html_src_str_2 = html_src_str_2.replace("}", '\n')

    match_1 = re.findall(r'"symbol":".*","shortName":', html_src_str_1)
    match_2 = re.findall(r'"symbol":".*","shortName":', html_src_str_2)

    list_most_actives = []

    for i in range(0, len(match_1), 1):
        tmp_string = match_1[i][10:]
        size = len(tmp_string)
        string = tmp_string[: size - 14]
        list_most_actives.append(string)

    for i in range(0, len(match_2), 1):
        tmp_string = match_2[i][10:]
        size = len(tmp_string)
        string = tmp_string[: size - 14]
        list_most_actives.append(string)

    return list_most_actives

def get_YAHOO_ticker_list(data_type):

    DRIVER_PATH = "C:/Users/despo/chromedriver_win32/chromedriver.exe"
    driver = webdriver.Chrome(executable_path=DRIVER_PATH)
    driver.get('https://finance.yahoo.com/gainers')
    driver.find_element_by_name("agree").click()

    list_gainers = get_list_gainers(driver)
    list_losers = get_list_losers(driver)
    list_trending_tickers = get_list_trending_tickers(driver)
    list_most_actives = get_list_most_actives(driver)

    ticker_list = list_gainers + list_losers + list_trending_tickers + list_most_actives

    if data_type == "MIXED_DATA":
        df = pd.DataFrame({'Symbol': ticker_list})
    elif data_type == "GAINER":
        df = pd.DataFrame({'Symbol': list_gainers})
    elif data_type == "LOOSERS":
        df = pd.DataFrame({'Symbol': list_losers})
    elif data_type == "TRENDING":
        df = pd.DataFrame({'Symbol': list_trending_tickers})
    elif data_type == "ACTIVES":
        df = pd.DataFrame({'Symbol': list_most_actives})

    df = df.drop_duplicates()
    #df = df.sort_values(['Symbol'], ignore_index=True)
    df.sort_values(by='Symbol', inplace=True, ascending=True)
    df = df.reset_index()
    df = df.drop(['index'], axis=1)
    #df.to_csv("ticker_list_no_duplicate.csv", index=False)

    return df
