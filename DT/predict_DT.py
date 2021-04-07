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

    regressor = [True, False]

    best_accuracy = 0

    print("grid search DTR...")

    for r in regressor:
        ################### DTC ###################
        if r == True:
            params = {
                'polynomialfeatures__degree': [2, 3, 4],
                'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
            }
            model = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=8),
                                  DecisionTreeRegressor(random_state=0))
            Classifier_grid = GridSearchCV(model, param_grid=params, cv=4)

        else:

            params = {
                'polynomialfeatures__degree': [2, 3, 4],
                'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
                 #'decisiontreeclassifier__criterion' : ['gini','entropy']
            }
            model = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=8),
                                  DecisionTreeClassifier(random_state=0))
            Classifier_grid = GridSearchCV(model, param_grid=params, cv=4)


        Classifier_grid.fit(X_train, Y_train)

        if (config.PRINT_MODEL == True):
            print("model :", Classifier_grid)
            print("best_param: ", Classifier_grid.best_params_)
            print("best_score: ", Classifier_grid.best_score_)

        Result_predicted = Classifier_grid.predict(X_test)
        Result_predicted = Result_predicted.reshape(-1, 1)

        result = pd.DataFrame(Result_predicted)
        result.reset_index(drop=True, inplace=True)

        predictions = result[0].to_list()
        accuracy = accuracy_score(Y_test, predictions, normalize=False)

        if (accuracy > best_accuracy):
            best_accuracy = accuracy

    return round(best_accuracy / len_y_test * 100,2)



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
            params = {
                'polynomialfeatures__degree': [2, 3, 4],
                'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
            }
            model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                                  SelectKBest(f_classif, k=8),
                                  XGBClassifier(n_estimators=n, max_depth=md))

            Classifier_XGB = GridSearchCV(model, param_grid=params, cv=4)
            #Classifier_XGB = XGBClassifier(n_estimators=n, max_depth=md)
            # Classifier_XGB = XGBRegressor(objective='reg:squarederror')
            Classifier_XGB.fit(X_train, Y_train)

            if (config.PRINT_MODEL == True):
                print("model :", Classifier_XGB)
                print("best_param: ", Classifier_XGB.best_params_)
                print("best_score: ", Classifier_XGB.best_score_)

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
    for kernel in ['linear','rbf', 'poly']:

        params = {
            'polynomialfeatures__degree': [2, 3, 4],
            'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
        }
        model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                              SelectKBest(f_classif, k=8),
                              StandardScaler(),
                              SVC(random_state=0, kernel=kernel))

        Classifier_SVM = GridSearchCV(model, param_grid=params, cv=4)
        #Classifier_SVM = SVC(kernel = kernel)

        Classifier_SVM.fit(X_train, Y_train)

        if (config.PRINT_MODEL == True):
            print("model :", Classifier_SVM)
            print("best_param: ", Classifier_SVM.best_params_)
            print("best_score: ", Classifier_SVM.best_score_)

        Result_predicted_SVM = Classifier_SVM.predict(X_test)
        Result_predicted_SVM = Result_predicted_SVM.reshape(-1, 1)

        result_SVM = pd.DataFrame(Result_predicted_SVM)
        result_SVM.reset_index(drop=True, inplace=True)

        predictions = result_SVM[0].to_list()
        accuracy_SVM = accuracy_score(Y_test, predictions, normalize=False)

        if accuracy_SVM > accuracy:
            accuracy = accuracy_SVM

    return round(accuracy / len_y_test * 100,2)

def get_KNeighbors_prediction(X_train, X_test, Y_train, Y_test):

    len_y_test = len(Y_test)

    weights = ['uniform', 'distance']
    n_neighbors = [3, 5, 7, 10, 25, 50, 100]

    best_accuracy = 0

    for w in weights:
        for n in n_neighbors:
            if((len(X_train) > n) and (len(Y_train) > n)):
                ################### KNeighbors ###################

                params = {
                    'polynomialfeatures__degree': [2, 3, 4],
                    'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
                }
                model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                                      SelectKBest(f_classif, k=8),
                                      StandardScaler(),
                                      KNeighborsClassifier(weights = w, n_neighbors = n))

                Classifier = GridSearchCV(model, param_grid=params, cv=4)
                #Classifier = KNeighborsClassifier(weights = w, n_neighbors = n)

                Classifier.fit(X_train, Y_train)

                if (config.PRINT_MODEL == True):
                    print("model :", Classifier)
                    print("best_param: ", Classifier.best_params_)
                    print("best_score: ", Classifier.best_score_)

                Result_predicted = Classifier.predict(X_test)
                Result_predicted = Result_predicted.reshape(-1, 1)

                result = pd.DataFrame(Result_predicted)
                result.reset_index(drop=True, inplace=True)

                predictions = result[0].to_list()
                accuracy = accuracy_score(Y_test, predictions, normalize=False)

                if(accuracy > best_accuracy):
                    best_accuracy = accuracy

    return round(best_accuracy / len_y_test * 100,2)


def get_RF_prediction(X_train, X_test, Y_train, Y_test):

    len_y_test = len(Y_test)

    n_estimators = [5, 10, 20]
    criterion = ['gini','entropy']

    best_accuracy = 0

    for n in n_estimators:
        for c in criterion:
            ################### Random Forest ###################
            params = {
                'polynomialfeatures__degree': [2, 3, 4],
                'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
            }
            model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                                  SelectKBest(f_classif, k=8),
                                  RandomForestClassifier(random_state=0, criterion=c, n_estimators=n))

            Classifier = GridSearchCV(model, param_grid=params, cv=4)
            #Classifier = RandomForestClassifier(n_estimators = n, criterion = c)
            Classifier.fit(X_train, Y_train)

            if(config.PRINT_MODEL == True):
                print("model :", Classifier)
                print("best_param: ", Classifier.best_params_)
                print("best_score: ", Classifier.best_score_)

            Result_predicted = Classifier.predict(X_test)
            Result_predicted = Result_predicted.reshape(-1, 1)

            result = pd.DataFrame(Result_predicted)
            result.reset_index(drop=True, inplace=True)

            predictions = result[0].to_list()
            accuracy = accuracy_score(Y_test, predictions, normalize=False)

            if(accuracy > best_accuracy):
                best_accuracy = accuracy

    return round(best_accuracy / len_y_test * 100, 2)

def get_ADABOOST_prediction(X_train, X_test, Y_train, Y_test):

    len_y_test = len(Y_test)

    n_estimators = [3, 5, 7, 10, 25, 50, 75, 100]
    learning_rate = [0.01, 0.05, 0.1, 0.5, 1]
    base_estimator = [True, False]

    best_accuracy = 0

    for be in base_estimator:
        for n in n_estimators:
            for lr in learning_rate:
                ################### ADABOOST ###################
                params = {
                    'polynomialfeatures__degree': [2, 3, 4],
                    'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
                }
                if be == True:
                    model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                                          SelectKBest(f_classif, k=8),
                                          AdaBoostClassifier(n_estimators = n, learning_rate=lr))
                    Classifier = GridSearchCV(model, param_grid=params, cv=4)
                    #Classifier = AdaBoostClassifier(n_estimators = n, learning_rate=lr)
                else:
                    model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                                          SelectKBest(f_classif, k=8),
                                          AdaBoostClassifier(base_estimator = DecisionTreeClassifier() ,n_estimators=n, learning_rate=lr))
                    Classifier = GridSearchCV(model, param_grid=params, cv=4)
                    #Classifier = AdaBoostClassifier(base_estimator = DecisionTreeClassifier() ,n_estimators=n, learning_rate=lr)
                Classifier.fit(X_train, Y_train)

                if (config.PRINT_MODEL == True):
                    print("model :", Classifier)
                    print("best_param: ", Classifier.best_params_)
                    print("best_score: ", Classifier.best_score_)

                Result_predicted = Classifier.predict(X_test)
                Result_predicted = Result_predicted.reshape(-1, 1)

                result = pd.DataFrame(Result_predicted)
                result.reset_index(drop=True, inplace=True)

                predictions = result[0].to_list()
                accuracy = accuracy_score(Y_test, predictions, normalize=False)

                if (accuracy > best_accuracy):
                    best_accuracy = accuracy

    return round(best_accuracy / len_y_test * 100,2)

def get_GRBOOST_prediction(X_train, X_test, Y_train, Y_test):

    len_y_test = len(Y_test)

    n_estimators = [3, 5, 7, 10, 25, 50, 75, 100]

    best_accuracy = 0

    for n in n_estimators:
        ################### GradientBoostingClassifier ###################
        params = {
            'polynomialfeatures__degree': [2, 3, 4],
            'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
        }

        model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                              SelectKBest(f_classif, k=8),
                              GradientBoostingClassifier(n_estimators = n))
        Classifier = GridSearchCV(model, param_grid=params, cv=4)
        #Classifier = GradientBoostingClassifier(n_estimators = n)

        Classifier.fit(X_train, Y_train)

        if (config.PRINT_MODEL == True):
            print("model :", Classifier)
            print("best_param: ", Classifier.best_params_)
            print("best_score: ", Classifier.best_score_)

        Result_predicted = Classifier.predict(X_test)
        Result_predicted = Result_predicted.reshape(-1, 1)

        result = pd.DataFrame(Result_predicted)
        result.reset_index(drop=True, inplace=True)

        predictions = result[0].to_list()
        accuracy = accuracy_score(Y_test, predictions, normalize=False)

        if (accuracy > best_accuracy):
            best_accuracy = accuracy

    return round(best_accuracy / len_y_test * 100,2)

def get_GNaiveB_prediction(X_train, X_test, Y_train, Y_test):

    len_y_test = len(Y_test)

    n_estimators = [3, 5, 7, 10, 25, 50, 75, 100]
    learning_rate = [0.01, 0.05, 0.1, 0.5, 1]

    best_accuracy = 0

    for n in n_estimators:
        for lr in learning_rate:
            ################### AdaBoost Classifier with NaiveBayes base ###################
            params = {
                'polynomialfeatures__degree': [2, 3, 4],
                'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
            }

            model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                                  SelectKBest(f_classif, k=8),
                                  AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators = n, learning_rate=lr))
            Classifier = GridSearchCV(model, param_grid=params, cv=4)
            #Classifier = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators = n, learning_rate=lr)
            Classifier.fit(X_train, Y_train)

            if(config.PRINT_MODEL == True):
                print("model :", Classifier)
                print("best_param: ", Classifier.best_params_)
                print("best_score: ", Classifier.best_score_)

            Result_predicted = Classifier.predict(X_test)
            Result_predicted = Result_predicted.reshape(-1, 1)

            result = pd.DataFrame(Result_predicted)
            result.reset_index(drop=True, inplace=True)

            predictions = result[0].to_list()
            accuracy = accuracy_score(Y_test, predictions, normalize=False)

            if (accuracy > best_accuracy):
                best_accuracy = accuracy

    return round(best_accuracy / len_y_test * 100,2)