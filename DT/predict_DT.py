import sys
import pandas as pd
import numpy as np

import config

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier

#from train_predict import *
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


def get_DTR_prediction(X_train, X_test, Y_train, Y_test):
    len_y_test = len(Y_test)
    best_score = 0

    for i in range(3):
        ################### DTC ###################
        params = {
            'polynomialfeatures__degree': [2, 3],
            'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
            #'decisiontreeclassifier__criterion' : ['gini','entropy']
            }
        model = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=8),
                              DecisionTreeClassifier(random_state=0))
        Classifier_grid = GridSearchCV(model, param_grid=params, cv=4)


        Classifier_grid.fit(X_train, Y_train)

        if (Classifier_grid.best_score_ > best_score):
            best_score = Classifier_grid.best_score_
            Classifier = Classifier_grid
            best_len_data = len(X_train)

        X_train, x_dump, Y_train, y_dump = train_test_split(X_train, Y_train, test_size=0.5, random_state=0)


    if (config.PRINT_MODEL == True):
        print("model :", Classifier)
        print("best_param: ", Classifier.best_params_)
        print("best_score: ", Classifier.best_score_)

    Result_predicted = Classifier.predict(X_test)
    Result_predicted = Result_predicted.reshape(-1, 1)

    result = pd.DataFrame(Result_predicted)
    result.reset_index(drop=True, inplace=True)

    predictions = result[0].to_list()
    #accuracy = accuracy_score(Y_test, predictions, normalize=False)
    accuracy = round(accuracy_score(Y_test, predictions, normalize=False) / len_y_test * 100,2)
    best_result = "accuracy: " + str(accuracy) + " best_score: " + str(Classifier.best_score_) + " best_param: " + str(Classifier.best_params_) + " df_len: " + str(best_len_data)

    return accuracy, round(Classifier.best_score_*100,2), best_len_data, best_result



def get_XGBOOST_prediction(X_train, X_test, Y_train, Y_test):

    n_estimators = [150, 200, 250, 450, 500, 550, 1000]
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    len_y_test = len(Y_test)
    best_score = 0
    len_data = len(X_train) + len(X_test)

    for i in range(3):
        params = {
            'polynomialfeatures__degree': [2, 3],
            'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
            #'xgbclassifier__n_estimators' : [150, 200, 250, 450, 500, 550, 1000],
            #'xgbclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        }
        model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                              SelectKBest(f_classif, k=8),
                              XGBClassifier(random_state=0))
                              #XGBClassifier(n_estimators=n, max_depth=md))

        Classifier_XGB = GridSearchCV(model, param_grid=params, cv=4)
        #Classifier_XGB = XGBClassifier(n_estimators=n, max_depth=md)
        # Classifier_XGB = XGBRegressor(objective='reg:squarederror')
        Classifier_XGB.fit(X_train, Y_train)

        if (Classifier_XGB.best_score_ > best_score):
            best_score = Classifier_XGB.best_score_
            Classifier = Classifier_XGB
            best_len_data = len(X_train)

        X_train, x_dump, Y_train, y_dump = train_test_split(X_train, Y_train, test_size=0.5, random_state=0)

    if (config.PRINT_MODEL == True):
        print("model :", Classifier)
        print("best_param: ", Classifier.best_params_)
        print("best_score: ", Classifier.best_score_)

    Result_predicted_XGB = Classifier.predict(X_test)
    Result_predicted_XGB = Result_predicted_XGB.reshape(-1, 1)

    result_XGB = pd.DataFrame(Result_predicted_XGB)
    result_XGB.reset_index(drop=True, inplace=True)

    predictions = result_XGB[0].to_list()
    accuracy = round(accuracy_score(Y_test, predictions, normalize=True), 4)
    #best_result = "XGB_depth_" + str(best_depth_XGB) + "_est_" + str(best_estimator_XGB)
    best_result = "accuracy: " + str(accuracy * 100) + " best praram: " + str(Classifier.best_params_) + " df_len: " + str(best_len_data)

    return accuracy, round(Classifier.best_score_*100,2), best_len_data, best_result



def get_SVM_prediction(X_train, X_test, Y_train, Y_test):
    len_y_test = len(Y_test)
    best_score = 0

    for i in range(3):
        ################### SVM ###################
        for kernel in ['linear','rbf', 'poly']:

            params = {
                'polynomialfeatures__degree': [2, 3],
                'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
            }
            model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                                  SelectKBest(f_classif, k=8),
                                  StandardScaler(),
                                  SVC(random_state=0, kernel=kernel))

            Classifier_SVM = GridSearchCV(model, param_grid=params, cv=4)
            #Classifier_SVM = SVC(kernel = kernel)

            Classifier_SVM.fit(X_train, Y_train)

            if(Classifier_SVM.best_score_ > best_score):
                best_score = Classifier_SVM.best_score_
                Classifier = Classifier_SVM
                best_len_data = len(X_train)

        X_train, x_dump, Y_train, y_dump = train_test_split(X_train, Y_train, test_size=0.5, random_state=0)


    if (config.PRINT_MODEL == True):
        print("model :", Classifier)
        print("best_param: ", Classifier.best_params_)
        print("best_score: ", Classifier.best_score_)

    Result_predicted_SVM = Classifier.predict(X_test)
    Result_predicted_SVM = Result_predicted_SVM.reshape(-1, 1)

    result_SVM = pd.DataFrame(Result_predicted_SVM)
    result_SVM.reset_index(drop=True, inplace=True)

    predictions = result_SVM[0].to_list()
    #accuracy = accuracy_score(Y_test, predictions, normalize=False)
    accuracy = round(accuracy_score(Y_test, predictions, normalize=False) / len_y_test * 100,2)
    best_result = "accuracy: " + str(accuracy) + " best_score: " + str(Classifier.best_score_) + " best_param: " + str(Classifier.best_params_) + " kernel: " + kernel + " df_len: " + str(best_len_data)

    return accuracy, round(Classifier.best_score_*100,2), best_len_data, best_result

def get_KNeighbors_prediction(X_train, X_test, Y_train, Y_test):
    len_y_test = len(Y_test)
    best_score = 0
    weights = ['uniform', 'distance']
    n_neighbors = [3, 5, 7, 10, 25, 50, 100]

    for i in range(3):
        for w in weights:
            for n in n_neighbors:
                if((len(X_train) > n) and (len(Y_train) > n)):
                    ################### KNeighbors ###################

                    params = {
                        'polynomialfeatures__degree': [2, 3],
                        'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
                    }
                    model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                                          SelectKBest(f_classif, k=8),
                                          StandardScaler(),
                                          KNeighborsClassifier(weights = w, n_neighbors = n))

                    GridClassifier = GridSearchCV(model, param_grid=params, cv=4)
                    #GridClassifier = KNeighborsClassifier(weights = w, n_neighbors = n)

                    GridClassifier.fit(X_train, Y_train)

                    if (GridClassifier.best_score_ > best_score):
                        best_score = GridClassifier.best_score_
                        Classifier = GridClassifier
                        best_len_data = len(X_train)
        X_train, x_dump, Y_train, y_dump = train_test_split(X_train, Y_train, test_size=0.5, random_state=0)

    if (config.PRINT_MODEL == True):
        print("model :", Classifier)
        print("best_param: ", Classifier.best_params_)
        print("best_score: ", Classifier.best_score_)

    Result_predicted = Classifier.predict(X_test)
    Result_predicted = Result_predicted.reshape(-1, 1)

    result = pd.DataFrame(Result_predicted)
    result.reset_index(drop=True, inplace=True)

    predictions = result[0].to_list()
    #accuracy = accuracy_score(Y_test, predictions, normalize=False)
    accuracy = round(accuracy_score(Y_test, predictions, normalize=False) / len_y_test * 100,2)
    best_result = "accuracy: " + str(accuracy) + " best_score: " + str(Classifier.best_score_) + " best_param: " + str(Classifier.best_params_) + " weights: " + str(w) + " n_neighbors: " + str(n) + " df_len: " + str(best_len_data)

    return accuracy, round(Classifier.best_score_*100,2), best_len_data, best_result


def get_RF_prediction(X_train, X_test, Y_train, Y_test):
    len_y_test = len(Y_test)
    best_score = 0
    n_estimators = [5, 10, 20]
    criterion = ['gini','entropy']

    for i in range(3):
        for n in n_estimators:
            for c in criterion:
                ################### Random Forest ###################
                params = {
                    'polynomialfeatures__degree': [2, 3],
                    'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
                }
                model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                                      SelectKBest(f_classif, k=8),
                                      RandomForestClassifier(random_state=0, criterion=c, n_estimators=n))

                GridClassifier = GridSearchCV(model, param_grid=params, cv=4)
                #Classifier = RandomForestClassifier(n_estimators = n, criterion = c)
                GridClassifier.fit(X_train, Y_train)

                if (GridClassifier.best_score_ > best_score):
                    best_score = GridClassifier.best_score_
                    Classifier = GridClassifier
                    best_n = n
                    best_c = c
                    best_len_data = len(X_train)
        X_train, x_dump, Y_train, y_dump = train_test_split(X_train, Y_train, test_size=0.5, random_state=0)

    if(config.PRINT_MODEL == True):
        print("model :", Classifier)
        print("best_param: ", Classifier.best_params_)
        print("best_score: ", Classifier.best_score_)

    Result_predicted = Classifier.predict(X_test)
    Result_predicted = Result_predicted.reshape(-1, 1)

    result = pd.DataFrame(Result_predicted)
    result.reset_index(drop=True, inplace=True)

    predictions = result[0].to_list()
    #accuracy = accuracy_score(Y_test, predictions, normalize=False)
    accuracy = round(accuracy_score(Y_test, predictions, normalize=False) / len_y_test * 100,2)
    best_result = "accuracy: " + str(accuracy) + " best_score: " + str(Classifier.best_score_) + " best_param: " + str(Classifier.best_params_) + " criterion: " + str(best_c) + " n_estimators: " + str(best_n) + " df_len: " + str(best_len_data)

    return accuracy, round(Classifier.best_score_*100,2), best_len_data, best_result


def get_ADABOOST_prediction(X_train, X_test, Y_train, Y_test):
    len_y_test = len(Y_test)
    best_score = 0
    n_estimators = [3, 5, 7, 10, 25, 50, 75, 100]
    learning_rate = [0.01, 0.05, 0.1, 0.5, 1]
    base_estimator = [True, False]

    for i in range(3):
        for be in base_estimator:
            #for n in n_estimators:
                #for lr in learning_rate:
            ################### ADABOOST ###################
            params = {
                'polynomialfeatures__degree': [2, 3],
                'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
            }
            if be == True:
                model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                                      SelectKBest(f_classif, k=8),
                                      AdaBoostClassifier(n_estimators = n, learning_rate=lr))
                GridClassifier = GridSearchCV(model, param_grid=params, cv=4)
                #GridClassifier = AdaBoostClassifier(n_estimators = n, learning_rate=lr)
            else:
                model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                                      SelectKBest(f_classif, k=8),
                                      AdaBoostClassifier(base_estimator = DecisionTreeClassifier() ,n_estimators=n, learning_rate=lr))
                GridClassifier = GridSearchCV(model, param_grid=params, cv=4)
                #GridClassifier = AdaBoostClassifier(base_estimator = DecisionTreeClassifier() ,n_estimators=n, learning_rate=lr)

            GridClassifier.fit(X_train, Y_train)

            if (GridClassifier.best_score_ > best_score):
                Classifier = GridClassifier
                best_lr = lr
                best_n = n
                best_score = GridClassifier.best_score_
                if be == True:
                    best_be = "base_estimator: DecisionTreeClassifier"
                else:
                    best_be = "base_estimator: none"
                best_len_data = len(X_train)
        X_train, x_dump, Y_train, y_dump = train_test_split(X_train, Y_train, test_size=0.5, random_state=0)

    if (config.PRINT_MODEL == True):
        print("model :", Classifier)
        print("best_param: ", Classifier.best_params_)
        print("best_score: ", Classifier.best_score_)

    Result_predicted = Classifier.predict(X_test)
    Result_predicted = Result_predicted.reshape(-1, 1)

    result = pd.DataFrame(Result_predicted)
    result.reset_index(drop=True, inplace=True)

    predictions = result[0].to_list()
    #accuracy = accuracy_score(Y_test, predictions, normalize=False)
    accuracy = round(accuracy_score(Y_test, predictions, normalize=False) / len_y_test * 100,2)
    best_result = "accuracy: " + str(accuracy) + " best_score: " + str(Classifier.best_score_) + " best_param: " + str(Classifier.best_params_) + best_be + " learning_rate: " + str(best_lr) + " n_estimators: " + str(best_n) + " df_len: " + str(best_len_data)

    return accuracy, round(Classifier.best_score_*100,2), best_len_data, best_result

def get_GRBOOST_prediction(X_train, X_test, Y_train, Y_test):
    len_data = len(X_train) + len(X_test)
    len_y_test = len(Y_test)
    best_score = 0
    n_estimators = [3, 5, 7, 10, 25, 50, 75, 100]

    for i in range(3):
        for n in n_estimators:
            ################### GradientBoostingClassifier ###################
            params = {
                'polynomialfeatures__degree': [2, 3],
                'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
            }

            model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                                  SelectKBest(f_classif, k=8),
                                  GradientBoostingClassifier(n_estimators = n))
            GridClassifier = GridSearchCV(model, param_grid=params, cv=4)
            #Classifier = GradientBoostingClassifier(n_estimators = n)

            GridClassifier.fit(X_train, Y_train)

            if (GridClassifier.best_score_ > best_score):
                Classifier = GridClassifier
                best_n = n
                best_score = GridClassifier.best_score_
                best_len_data = len(X_train)
        X_train, x_dump, Y_train, y_dump = train_test_split(X_train, Y_train, test_size=0.5, random_state=0)

    if (config.PRINT_MODEL == True):
        print("model :", Classifier)
        print("best_param: ", Classifier.best_params_)
        print("best_score: ", Classifier.best_score_)

    Result_predicted = Classifier.predict(X_test)
    Result_predicted = Result_predicted.reshape(-1, 1)

    result = pd.DataFrame(Result_predicted)
    result.reset_index(drop=True, inplace=True)

    predictions = result[0].to_list()
    #accuracy = accuracy_score(Y_test, predictions, normalize=False)
    accuracy = round(accuracy_score(Y_test, predictions, normalize=False) / len_y_test * 100,2)
    best_result = "accuracy: " + str(accuracy) + " best_score: " + str(Classifier.best_score_) + " best_param: " + str(Classifier.best_params_) + " n_estimators: " + str(best_n) + " df_len: " + str(best_len_data)

    return accuracy, round(Classifier.best_score_*100,2), best_len_data, best_result

def get_GNaiveB_prediction(X_train, X_test, Y_train, Y_test):
    len_data = len(X_train) + len(X_test)
    len_y_test = len(Y_test)
    best_score = 0
    n_estimators = [3, 5, 7, 10, 25, 50, 75, 100]
    learning_rate = [0.01, 0.05, 0.1, 0.5, 1]

    for i in range(3):
        #for n in n_estimators:
        #    for lr in learning_rate:
        ################### AdaBoost Classifier with NaiveBayes base ###################
        params = {
            'polynomialfeatures__degree': [2, 3],
            'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
        }

        model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                              SelectKBest(f_classif, k=8),
                              AdaBoostClassifier(base_estimator=GaussianNB()))
                              #AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators = n, learning_rate=lr))
        GridClassifier = GridSearchCV(model, param_grid=params, cv=4)
        #Classifier = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators = n, learning_rate=lr)
        GridClassifier.fit(X_train, Y_train)

        if (GridClassifier.best_score_ > best_score):
            Classifier = GridClassifier
            #best_n = n
            #best_lr = lr
            best_score = GridClassifier.best_score_
            best_len_data = len(X_train)
        X_train, x_dump, Y_train, y_dump = train_test_split(X_train, Y_train, test_size=0.5, random_state=0)


    if(config.PRINT_MODEL == True):
        print("model :", Classifier)
        print("best_param: ", Classifier.best_params_)
        print("best_score: ", Classifier.best_score_)

    Result_predicted = Classifier.predict(X_test)
    Result_predicted = Result_predicted.reshape(-1, 1)

    result = pd.DataFrame(Result_predicted)
    result.reset_index(drop=True, inplace=True)

    predictions = result[0].to_list()
    #accuracy = accuracy_score(Y_test, predictions, normalize=False)
    accuracy = round(accuracy_score(Y_test, predictions, normalize=False) / len_y_test * 100,2)
    #best_result = "accuracy: " + str(accuracy) + " best_score: " + str(Classifier.best_score_) + " best_param: " + str(Classifier.best_params_) + " learning_rate: " + str(best_lr) + " n_estimators: " + str(best_n)
    best_result = "accuracy: " + str(accuracy) + " best_score: " + str(Classifier.best_score_) + " best_param: " + str(Classifier.best_params_) + " df_len: " + str(best_len_data)

    return accuracy, round(Classifier.best_score_*100,2), best_len_data, best_result