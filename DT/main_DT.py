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
#from predict_DT import model_SVM_preprocessing
#from predict_DT import model_ADABoost_preprocessing

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
    #df.pop('Volume')

    X = df

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    expected_y = pd.DataFrame(Y_test)
    expected_y.reset_index(drop=True, inplace=True)

    if(config.USE_TREND == True):
        Y_test = expected_y["trend"].to_list()
    else:
        Y_test = expected_y["target"].to_list()

    print("model prediction >>>>")

    # Model hyperparameters Tuning
    #model_SVM_preprocessing(X_train, X_test, Y_train, Y_test)
    #model_ADABoost_preprocessing(X_train, X_test, Y_train, Y_test)

    if(config.COMPUTE_DT == True):
        print("model DT >>>>")
        accuracy_DTR, bestscore_DTR, len_data_DTR, result = get_DTR_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["DTR_pred"] = accuracy_DTR
        df_movement_stock_list["DTR_best"] = bestscore_DTR
        df_movement_stock_list["DTR_dt_s"] = len_data_DTR
        print("DTR PARAMETER: ", result)

    if(config.XGBOOST == True):
        print("model XGBOOST >>>>")
        accuracy_XGBOOST, bestscore_XGBOOST, len_data_XGBOOST, result = get_XGBOOST_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["XGBST_pred"] = accuracy_XGBOOST
        df_movement_stock_list["XGBST_best"] = bestscore_XGBOOST
        df_movement_stock_list["XGBST_dt_s"] = len_data_XGBOOST
        print("XGBOOST PARAMETER: ",result)

    if(config.COMPUTE_GNaiveB == True):
        print("model GNaiveB >>>>")
        accuracy_GNaiveB, bestscore_GNaiveB, len_data_GNaiveB, result = get_GNaiveB_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["GNaiveB_pred"] = accuracy_GNaiveB
        df_movement_stock_list["GNaiveB_best"] = bestscore_GNaiveB
        df_movement_stock_list["GNaiveB_dt_s"] = len_data_GNaiveB
        print("GNaiveB PARAMETER: ", result)

    if(config.COMPUTE_RF == True):
        print("model RF >>>>")
        accuracy_RF, bestscore_RF, len_data_RF, result = get_RF_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["RF_pred"] = accuracy_RF
        df_movement_stock_list["RF_best"] = bestscore_RF
        df_movement_stock_list["RF_dt_s"] = len_data_RF
        print("RF PARAMETER: ", result)

    if(config.LIGHTGBM == True):
        print("model LightGBM >>>>")
        accuracy_lightGBM, bestscore_lightGBM, len_data_lightGBM, result = get_lightGBM_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["lGBM_pred"] = accuracy_lightGBM
        df_movement_stock_list["lGBM_best"] = bestscore_lightGBM
        df_movement_stock_list["lGBM_dt_s"] = len_data_lightGBM
        print("LightGBM PARAMETER: ", result)

    if(config.COMPUTE_GRBOOST == True):
        print("model GRBOOST >>>>")
        accuracy_GRBOOST, bestscore_GRBOOST, len_data_GRBOOST, result = get_GRBOOST_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["GRBST_pred"] = accuracy_GRBOOST
        df_movement_stock_list["GRBST_best"] = bestscore_GRBOOST
        df_movement_stock_list["GRBST_dt_s"] = len_data_GRBOOST
        print("GRBOOST PARAMETER: ", result)

    if(config.COMPUTE_KN == True):
        print("model KN >>>>")
        accuracy_KN, bestscore_KN, len_data_KN, result = get_KNeighbors_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["KN_pred"] = accuracy_KN
        df_movement_stock_list["KN_best"] = bestscore_KN
        df_movement_stock_list["KN_dt_s"] = len_data_KN
        print("KN PARAMETER: ", result)

    if(config.COMPUTE_SVM == True):
        print("model SVM >>>>")
        accuracy_SVM, bestscore_SVM, len_data_SVM, result = get_SVM_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["SVM_pred"] = accuracy_SVM
        df_movement_stock_list["SVM_best"] = bestscore_SVM
        df_movement_stock_list["SVM_dt_s"] = len_data_SVM
        print("SVM PARAMETER: ", result)

    if(config.COMPUTE_ADABOOST == True):
        print("model ADABOOST >>>>")
        accuracy_ADABOOST, bestscore_ADABOOST, len_data_ADABOOST, result = get_ADABOOST_prediction(X_train, X_test, Y_train, Y_test)
        df_movement_stock_list["ADABST_pred"] = accuracy_ADABOOST
        df_movement_stock_list["ADABST_best"] = bestscore_ADABOOST
        df_movement_stock_list["ADABST_dt_s"] = len_data_ADABOOST
        print("ADABOOST PARAMETER: ", result)

    return df_movement_stock_list