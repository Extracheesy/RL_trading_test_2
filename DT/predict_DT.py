import sys
import pandas as pd
import numpy as np

import config

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from train_predict import *
from sklearn import svm

def main_DT_train(df):

    n_estimators = [150, 200, 250, 450, 500, 550, 1000]
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    best_depth_XGB = 0
    best_estimator_XGB = 0
    max_score_XGB = 0

    y = df.pop('trend')
    df.pop('Date')
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
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

    if (config.XGBOOST == True):
        for n in n_estimators:
            for md in max_depth:
                ################### XGBOOST ###################
                Classifier_XGB = XGBClassifier(n_estimators=n, max_depth=md)
                #Classifier_XGB = XGBRegressor(objective='reg:squarederror')
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
        result = "XGB_depth_" + str(best_depth_XGB) + "_est_" + str(best_estimator_XGB)



    return round(accuracy / len_y_test * 100), result



def get_DTR_prediction(X_train, X_test, Y_train, Y_test):

    len_y_test = len(Y_test)

    ################### DTR ###################
    Classifier = DecisionTreeRegressor()
    Classifier.fit(X_train, Y_train)

    Result_predicted = Classifier.predict(X_test)
    Result_predicted = Result_predicted.reshape(-1, 1)

    result = pd.DataFrame(Result_predicted)
    result.reset_index(drop=True, inplace=True)

    predictions = result[0].to_list()
    accuracy = accuracy_score(Y_test, predictions, normalize=False)

    return round(accuracy / len_y_test * 100)



def get_XGBOOST_prediction(X_train, X_test, Y_train, Y_test):

    n_estimators = [150, 200, 250, 450, 500, 550, 1000]
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    best_depth_XGB = 0
    best_estimator_XGB = 0
    max_score_XGB = 0

    len_y_test = len(Y_test)

    for n in n_estimators:
        for md in max_depth:
            ################### XGBOOST ###################
            Classifier_XGB = XGBClassifier(n_estimators=n, max_depth=md)
            # Classifier_XGB = XGBRegressor(objective='reg:squarederror')
            Classifier_XGB.fit(X_train, Y_train)

            Result_predicted_XGB = Classifier_XGB.predict(X_test)
            Result_predicted_XGB = Result_predicted_XGB.reshape(-1, 1)

            result_XGB = pd.DataFrame(Result_predicted_XGB)
            result_XGB.reset_index(drop=True, inplace=True)

            predictions = result_XGB[0].to_list()
            accuracy_XGB = accuracy_score(Y_test, predictions, normalize=False)

            if accuracy_XGB > max_score_XGB:
                best_depth_XGB = md
                best_estimator_XGB = n
                max_score_XGB = accuracy_XGB

    accuracy = max_score_XGB
    result = "XGB_depth_" + str(best_depth_XGB) + "_est_" + str(best_estimator_XGB)

    return round(accuracy / len_y_test * 100), result



def get_SVM_prediction(X_train, X_test, Y_train, Y_test):

    len_y_test = len(Y_test)

    accuracy = 0
    ################### SVM ###################
    for kernel in ['linear','rbf']:
        Classifier_SVM = svm.SVC(kernel = kernel)
        Classifier_SVM.fit(X_train, Y_train)

        Result_predicted_SVM = Classifier_SVM.predict(X_test)
        Result_predicted_SVM = Result_predicted_SVM.reshape(-1, 1)

        result_SVM = pd.DataFrame(Result_predicted_SVM)
        result_SVM.reset_index(drop=True, inplace=True)

        predictions = result_SVM[0].to_list()
        accuracy_SVM = accuracy_score(Y_test, predictions, normalize=False)

        if accuracy_SVM > accuracy:
            accuracy = accuracy_SVM

    return round(accuracy / len_y_test * 100)

def get_KNeighbors_prediction(X_train, X_test, Y_train, Y_test, n_neighbors):

    len_y_test = len(Y_test)

    ################### KNeighbors ###################
    Classifier = KNeighborsClassifier(n_neighbors = n_neighbors)
    Classifier.fit(X_train, Y_train)

    Result_predicted = Classifier.predict(X_test)
    Result_predicted = Result_predicted.reshape(-1, 1)

    result = pd.DataFrame(Result_predicted)
    result.reset_index(drop=True, inplace=True)

    predictions = result[0].to_list()
    accuracy = accuracy_score(Y_test, predictions, normalize=False)

    return round(accuracy / len_y_test * 100)


def get_RF_prediction(X_train, X_test, Y_train, Y_test, n_estimators):

    len_y_test = len(Y_test)

    ################### Random Forest ###################
    Classifier = RandomForestClassifier(n_estimators = n_estimators)
    Classifier.fit(X_train, Y_train)

    Result_predicted = Classifier.predict(X_test)
    Result_predicted = Result_predicted.reshape(-1, 1)

    result = pd.DataFrame(Result_predicted)
    result.reset_index(drop=True, inplace=True)

    predictions = result[0].to_list()
    accuracy = accuracy_score(Y_test, predictions, normalize=False)

    return round(accuracy / len_y_test * 100)

def get_ADABOOST_prediction(X_train, X_test, Y_train, Y_test, n_estimators):

    len_y_test = len(Y_test)

    ################### ADABOOST ###################
    Classifier = AdaBoostClassifier(n_estimators = n_estimators)
    Classifier.fit(X_train, Y_train)

    Result_predicted = Classifier.predict(X_test)
    Result_predicted = Result_predicted.reshape(-1, 1)

    result = pd.DataFrame(Result_predicted)
    result.reset_index(drop=True, inplace=True)

    predictions = result[0].to_list()
    accuracy = accuracy_score(Y_test, predictions, normalize=False)

    return round(accuracy / len_y_test * 100)

def get_GRBOOST_prediction(X_train, X_test, Y_train, Y_test, n_estimators):

    len_y_test = len(Y_test)

    ################### GradientBoostingClassifier ###################
    Classifier = GradientBoostingClassifier(n_estimators = n_estimators)
    Classifier.fit(X_train, Y_train)

    Result_predicted = Classifier.predict(X_test)
    Result_predicted = Result_predicted.reshape(-1, 1)

    result = pd.DataFrame(Result_predicted)
    result.reset_index(drop=True, inplace=True)

    predictions = result[0].to_list()
    accuracy = accuracy_score(Y_test, predictions, normalize=False)

    return round(accuracy / len_y_test * 100)

def get_GNaiveB_prediction(X_train, X_test, Y_train, Y_test, n_estimators):

    len_y_test = len(Y_test)

    ################### AdaBoost Classifier with NaiveBayes base ###################
    Classifier = AdaBoostClassifier(n_estimators=n_estimators, base_estimator=GaussianNB())
    Classifier.fit(X_train, Y_train)

    Result_predicted = Classifier.predict(X_test)
    Result_predicted = Result_predicted.reshape(-1, 1)

    result = pd.DataFrame(Result_predicted)
    result.reset_index(drop=True, inplace=True)

    predictions = result[0].to_list()
    accuracy = accuracy_score(Y_test, predictions, normalize=False)

    return round(accuracy / len_y_test * 100)