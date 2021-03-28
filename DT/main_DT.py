import pandas as pd

from predict_DT import get_DTR_prediction
from predict_DT import get_XGBOOST_prediction
from predict_lightgbm import predict_lightGBM

from sklearn.model_selection import train_test_split


def run_DT_prediction(df, df_movement_stock_list):

    y = df.pop('trend')
    df.pop('Date')
    X = df

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=5)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    expected_y = pd.DataFrame(Y_test)
    expected_y.reset_index(drop=True, inplace=True)
    Y_test = expected_y["trend"].to_list()

    if(config.COMPUTE_DT == True):
        accuracy_DTR = get_DTR_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["DTR"] = accuracy_DTR

    if(config.XGBOOST == True):
        accuracy_XGBOOST = get_XGBOOST_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["XGBOOST"] = accuracy_XGBOOST

    if(config.LIGHTGBM == True):
        accuracy_lightGBM = predict_lightGBM(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["lightGBM"] = accuracy_lightGBM

    return df_movement_stock_list