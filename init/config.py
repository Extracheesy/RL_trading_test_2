LOAD_YEARS = 10 * 52 # 5 * 12 months for 5 years

#DEBUG_FORCE_STOCK = True
DEBUG_FORCE_STOCK = False

#COMPUTE_PREDICT_MODEL = True
COMPUTE_PREDICT_MODEL = False

#SAVE_MODEL = "SAVE_LSTM_MODEL"
SAVE_MODEL = "NO_SAVE_LSTM_MODEL"

COMPUTE_DT = True
#COMPUTE_DT = False

XGBOOST = True
#XGBOOST = False
COMPUTE_SVM = True
#COMPUTE_SVM = False
COMPUTE_KN = True
#COMPUTE_KN = False
COMPUTE_RF = True
#COMPUTE_RF = False
COMPUTE_ADABOOST = True
#COMPUTE_ADABOOST = False
COMPUTE_GRBOOST = True
#COMPUTE_GRBOOST = False
COMPUTE_GNaiveB = True
#COMPUTE_GNaiveB = False
LIGHTGBM = True
#LIGHTGBM = False


COLAB = False
#COLAB = True

DATA_SOURCE = "READ_CSV_FROM_DATABASE"
# DATA_SOURCE = "SCRAPING"

NASDAQ_100 = True
SCRAPING_NASDAQ = False
WIKI = True

DEBUG = True


#SELCTION = ["GAINERS"]
#SELCTION = ["TRENDING"]
#SELCTION = ["CAC40"]
SELCTION = ["DJI"]
#SELCTION = ["NASDAQ"]
#SELCTION = ["ACTIVES"]
#SELCTION = ["LOOSERS"]
#SELCTION = ["DAX"]
#SELCTION = ["SP500"]
#SELCTION = ["ALL"]

SAVE_5DAY_DATA = False
#SAVE_5DAY_DATA = True

SAVE_PRE_PROCESSED_DATA = False
#SAVE_PRE_PROCESSED_DATA = True

USE_TREND = True
#USE_TREND = False # Use target