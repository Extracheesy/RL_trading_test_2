import pandas as pd
import config
from predict_DT import get_DTR_prediction
from predict_DT import get_XGBOOST_prediction
from predict_DT import get_KNeighbors_prediction
from predict_DT import get_ADABOOST_prediction
from predict_DT import get_RF_prediction
from predict_DT import get_SVM_prediction
from predict_DT import get_GRBOOST_prediction
from predict_DT import get_GNaiveB_prediction
from predict_lightgbm import get_lightGBM_prediction


from sklearn.model_selection import train_test_split


def run_DT_prediction(df, df_movement_stock_list):

    if(config.USE_TREND == True):
        y = df.pop('trend')
        df.pop('target')
    else:
        y = df.pop('target')
        df.pop('trend')

    df.pop('Date')
    df.pop('Adj Close')
    df.pop('Open')
    df.pop('Close')
    df.pop('High')
    df.pop('Low')
    df.pop('Volume')

    X = df

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=5)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    expected_y = pd.DataFrame(Y_test)
    expected_y.reset_index(drop=True, inplace=True)

    if(config.USE_TREND == True):
        Y_test = expected_y["trend"].to_list()
    else:
        Y_test = expected_y["target"].to_list()

    if(config.LIGHTGBM == True):
        accuracy_lightGBM = get_lightGBM_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["lightGBM"] = accuracy_lightGBM

    if(config.COMPUTE_GRBOOST == True):
        accuracy_GRBOOST = get_GRBOOST_prediction(X_train, X_test, Y_train, Y_test, 100)
        df_movement_stock_list["GRBOOST"] = accuracy_GRBOOST

    if(config.COMPUTE_GNaiveB == True):
        accuracy_GNaiveB = get_GNaiveB_prediction(X_train, X_test, Y_train, Y_test, 100)
        df_movement_stock_list["GNaiveB"] = accuracy_GNaiveB

    if(config.COMPUTE_ADABOOST == True):
        accuracy_ADABOOST = get_ADABOOST_prediction(X_train, X_test, Y_train, Y_test, 100)
        df_movement_stock_list["ADABOOST"] = accuracy_ADABOOST

    if(config.COMPUTE_DT == True):
        accuracy_DTR = get_DTR_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["DTR"] = accuracy_DTR

    if(config.COMPUTE_RF == True):
        accuracy_RF = get_RF_prediction(X_train, X_test, Y_train, Y_test, 10)
        df_movement_stock_list["RF"] = accuracy_RF

    if(config.COMPUTE_KN == True):
        accuracy_KN = get_KNeighbors_prediction(X_train, X_test, Y_train, Y_test, 3)
        df_movement_stock_list["KNeighbors"] = accuracy_KN

    if(config.COMPUTE_SVM == True):
        accuracy_SVM = get_SVM_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["SVM"] = accuracy_SVM

    if(config.XGBOOST == True):
        accuracy_XGBOOST, result = get_XGBOOST_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["XGBOOST"] = accuracy_XGBOOST
        print("XGBOOST PARAMETER: ",result)

    return df_movement_stock_list