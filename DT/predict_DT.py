import sys
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from train_predict import *


def main_DT_train(df):

    n_estimators = [150, 200, 250, 450, 500, 550, 1000]
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    best_depth_XGB = 0
    best_estimator_XGB = 0
    max_score_XGB = 0

    y = df.pop('trend')
    remove_date = df.pop('Date')
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5  )
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    expected_y = pd.DataFrame(y_test)
    expected_y.reset_index(drop=True, inplace=True)
    len_y_test = len(y_test)
    y_test = expected_y["trend"].to_list()

    ################### DTR ###################
    Classifier_DTR = DecisionTreeRegressor()
    Classifier_DTR.fit(X_train, y_train)

    Result_predicted_DTR = Classifier_DTR.predict(X_test)
    Result_predicted_DTR = Result_predicted_DTR.reshape(-1, 1)

    result_DTR = pd.DataFrame(Result_predicted_DTR)
    result_DTR.reset_index(drop=True, inplace=True)

    predictions = result_DTR[0].to_list()
    accuracy_DTR = accuracy_score(y_test, predictions, normalize=False)

    for n in n_estimators:
        for md in max_depth:

            ################### XGBOOST ###################
            Classifier_XGB = XGBClassifier(n_estimators=n, max_depth=md)
            Classifier_XGB.fit(X_train, y_train)

            Result_predicted_XGB = Classifier_XGB.predict(X_test)
            Result_predicted_XGB = Result_predicted_XGB.reshape(-1,1)

            result_XGB = pd.DataFrame(Result_predicted_XGB)
            result_XGB.reset_index(drop=True, inplace=True)

            predictions = result_XGB[0].to_list()
            accuracy_XGB = accuracy_score(y_test, predictions, normalize = False)

            if accuracy_XGB > max_score_XGB:
                best_depth_XGB = md
                best_estimator_XGB = n
                max_score_XGB = accuracy_XGB

    if accuracy_DTR > max_score_XGB:
        accuracy = accuracy_DTR
        result = "DTR"
    else:
        accuracy = max_score_XGB
        result = "XGB_depth_" + str(md) + "_est_" + str(n)

    return round(accuracy / len_y_test * 100), result
