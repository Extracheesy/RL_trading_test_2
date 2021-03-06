from stockstats import StockDataFrame as Sdf
from load_yfinance_data import *
from predict_DT import *
from datetime import date, timedelta
from df_tools import SaveData

import config

def calcualte_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    data['adjcp'] = data['prccd'] / data['ajexdi']
    data['open'] = data['prcod'] / data['ajexdi']
    data['high'] = data['prchd'] / data['ajexdi']
    data['low'] = data['prcld'] / data['ajexdi']
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data


def add_trend(df_stock_price):

    close = df_stock_price['Close']

    # Get the difference in price from previous step
    delta = close.diff()

    #df_stock_price['trend'] = delta


    df_stock_price = df_stock_price.replace([np.inf, -np.inf], np.nan).dropna()
    df_stock_price = df_stock_price.reset_index(drop=True)
    # df = df.reset_index(drop=True)

    len_df = len(df_stock_price)

    #for i in range(0,len_df,1):
    #    df_stock_price.loc[i,"trend"] = df_stock_price.loc[i,"trend"] * (-1)
    #    if(df_stock_price.loc[i,"trend"] <= 0):
    #        df_stock_price.loc[i, "trend"] = 0 # 0 - trend is going down
    #    else:
    #        df_stock_price.loc[i, "trend"] = 1 # 1 - trend is going up

    df_stock_price.insert(len(df_stock_price.columns), "trend", 0)
    df_stock_price.insert(len(df_stock_price.columns), "trend_d+1", 0)

    for i in range(0, len_df, 1):
        df_stock_price.loc[i, "trend"] = df_stock_price.loc[i, "Close"] - df_stock_price.loc[i, "Open"]
        if (df_stock_price.loc[i, "trend"] <= 0):
            df_stock_price.loc[i, "trend"] = int(0)  # 0 - trend is going down
        else:
            df_stock_price.loc[i, "trend"] = int(1)  # 1 - trend is going up

        if i>0:
            df_stock_price.loc[i - 1, "trend_d+1"] = df_stock_price.loc[i, "trend"]


    target_raw = (df_stock_price['Adj Close'].shift(-1) / df_stock_price['Adj Close']) - 1

    target_raw[target_raw > 0] = 1
    target_raw[target_raw <= 0] = 0

    df_stock_price.insert(len(df_stock_price.columns), "target", target_raw)


    df_stock_price.drop([0,1,2,3,4,5,6,7,8,9,10], axis=0, inplace=True)
    df_stock_price.drop([len_df - 1], axis=0, inplace=True)

    # reset index
    df_stock_price.reset_index(drop=True, inplace=True)

    df_stock_price["trend"] = df_stock_price["trend_d+1"]

    df_stock_price.pop('trend_d+1')

    return df_stock_price



def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())

    temp_macd = stock['macd']
    temp_macds = stock['macds']
    temp_macdh = stock['macdh']
    macd = pd.DataFrame(temp_macd)
    macds = pd.DataFrame(temp_macds)
    macdh = pd.DataFrame(temp_macdh)

    temp_rsi = stock['rsi_6']
    rsi = pd.DataFrame(temp_rsi)

    temp_cci = stock['cci']
    cci = pd.DataFrame(temp_cci)

    temp_adx = stock['adx']
    adx = pd.DataFrame(temp_adx)

    temp_pdi = stock['pdi']
    temp_mdi = stock['mdi']
    pdi = pd.DataFrame(temp_pdi)
    mdi = pd.DataFrame(temp_mdi)

    df.insert(len(df.columns), "macd",0)
    df.insert(len(df.columns), "macd_signal_line",0)
    df.insert(len(df.columns), "macd_hist",0)

    df.insert(len(df.columns), "rsi",0)

    df.insert(len(df.columns), "cci",0)

    df.insert(len(df.columns), "adx",0)

    df.insert(len(df.columns), "+DI",0)
    df.insert(len(df.columns), "-DI",0)

    len_df = len(df)
    for i in range(0,len_df,1):
        df.loc[i,"macd"] = macd.iloc[i][0]
        df.loc[i,"macd_signal_line"] = macds.iloc[i][0]
        df.loc[i,"macd_hist"] = macdh.iloc[i][0]

        df.loc[i,"rsi"] = rsi.iloc[i][0]

        df.loc[i,"cci"] = cci.iloc[i][0]

        df.loc[i,"adx"] = adx.iloc[i][0]

        df.loc[i,"+DI"] = pdi.iloc[i][0]
        df.loc[i,"-DI"] = mdi.iloc[i][0]

    #df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df

#def ADX(df_stock_price):
    # ADX(Average directional movement)

def CCI(df_stock_price):
    # Define function for Commodity Channel Index (CCI).

    close = df_stock_price["Close"]
    high  = df_stock_price["High"]
    low   = df_stock_price["Low"]
    ndays = 10
    constant = 0.015

    TP = (high + low + close) / 3

    CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.015 * TP.rolling(ndays).std()), name='CCI')

    df_stock_price['CCI'] = CCI

    return df_stock_price



def RSI(df_stock_price):
    # Window length for moving average
    window_length = 14

    close = df_stock_price['Close']
    # Get the difference in price from previous step
    delta = close.diff()
    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    delta = delta[1:]

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up1 = up.ewm(span=window_length).mean()
    roll_down1 = down.abs().ewm(span=window_length).mean()

    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))

    # Calculate the SMA
    roll_up2 = up.rolling(window_length).mean()
    roll_down2 = down.abs().rolling(window_length).mean()

    # Calculate the RSI based on SMA
    RS2 = roll_up2 / roll_down2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))

    df_stock_price['RSI EWMA'] = RSI1
    df_stock_price['RSI SMA'] = RSI2

    return df_stock_price



# Calculate the MACD and Signal Line indicators.
def MACD(df_stock_price):

    # Calculate the MACD and Signal Line indicators
    # Calculate the Short Term Exponential Moving Average
    ShortEMA = df_stock_price.Close.ewm(span=12, adjust=False).mean()  # AKA Fast moving average
    # Calculate the Long Term Exponential Moving Average
    LongEMA = df_stock_price.Close.ewm(span=26, adjust=False).mean()  # AKA Slow moving average
    # Calculate the Moving Average Convergence/Divergence (MACD)
    MACD = ShortEMA - LongEMA
    # Calcualte the signal line
    signal = MACD.ewm(span=9, adjust=False).mean()

    df_stock_price['My MACD'] = MACD
    df_stock_price['Signal Line'] = signal

    return (df_stock_price)



def SMA(df_stock_price):
    # Simple Moving Average (SMA)

    close = df_stock_price['Close']

    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()

    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()

    df_stock_price['SMA 20'] = sma_20
    df_stock_price['SMA 50'] = sma_50

    df_stock_price['EMA 20'] = ema_20
    df_stock_price['EMA 50'] = ema_50

    return df_stock_price


def DT_load_process_data(df):

    df['Date'] = df.index

    df = df.reset_index(drop=True)

    cols = ['Date'] + [col for col in df if col != 'Date']
    df = df[cols]

    # drop duplicates
    df = df.drop_duplicates()

    # convert Date column to datetime
    # df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')
    df['Date'] = pd.to_datetime(df['Date'])

    # sort by datetime
    df.sort_values(by = 'Date', inplace = True, ascending = True)

    return df


def DT_process_trend(df):

    # Old fashion calc technical incicators
    # df = SMA(df)
    # df = MACD(df)
    # df = RSI(df)
    # df = CCI(df)

    df = add_technical_indicator(df)
    df = add_trend(df)

    return df

def SaveDataDT(df, filename):

    df.to_csv("./data/yfinance_data_DT/" + filename + ".csv")

def SaveDataPredict(df, filename):

    df.to_csv("./data/yfinance_data_predict/" + filename + ".csv")

def pre_process_indictor_data(stock, df):

    df = DT_load_process_data(df)

    today = date.today()

    df = DT_process_trend(df)

    if(config.SAVE_PRE_PROCESSED_DATA == True):
        SaveData(df, stock + "_" + str(today))

    return df





