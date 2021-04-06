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

from sklearn.neural_network import MLPClassifier

#from train_predict import *
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV





def model_final(model, X ,threshold=0):
    return model.decision_function(X) > threshold


def evaluation(model, X_train, X_test, Y_train, Y_test):

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    print("Y_test: ", Y_test)
    print("prediction: ",Y_pred)
    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))



    N, train_score, val_score = learning_curve(model, X_train, Y_train,
                                               cv = 4, scoring = 'f1',
                                               train_sizes = np.linspace(0.1, 1, 10))



    #pd.DataFrame(model.feature_importances_, index=X_train.columns).plot
    #pd.DataFrame(model.feature_importances_ , index=X_train.columns).plot.bar()

    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='val score')
    plt.legend()

    plt.show()


def model_SVM_preprocessing(X_train, X_test, Y_train, Y_test):

    MULTI_MODEL = False
    GRID_SEARCH = True
    GRID_SEARCH_RANDOM = False

    if MULTI_MODEL == True:

        preprocessor = make_pipeline(PolynomialFeatures(3, include_bias=False), SelectKBest(f_classif, k=4))

        RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state = 0))
        AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
        SVM = make_pipeline(preprocessor, StandardScaler() ,SVC(random_state=0))
        KNN= make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())

        dict_of_models = {'RandomForest' : RandomForest,
                          'AdaBoost' : AdaBoost,
                          'SVM' : SVM,
                          'KNN' : KNN
                          }

        for name , model in dict_of_models.items():
            print("model: ", name)
            evaluation(model, X_train, X_test, Y_train, Y_test)


    else:
        if GRID_SEARCH == True:

            params = {'polynomialfeatures__degree': [2, 3],
                      'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
                      'svc__kernel': ['linear', 'rbf', 'poly'],
                      'svc__C': [0.1, 1, 10, 100, 500, 1000],
                      'svc__degree': [200, 400, 600, 800],
                      'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001]
                       #'adaboostclassifier__learning_rate': [0.01, 0.05, 0.1, 1],
                       #"adaboostclassifier__n_estimators": [10, 20, 50, 80, 100, 120, 150, 180, 200]
                      }

            #params = {'polynomialfeatures__degree': [3],
            #          'selectkbest__k': [4]
            #          }

            model = make_pipeline(PolynomialFeatures(3, include_bias=False), SelectKBest(f_classif, k=4),
                                  StandardScaler(),
                                  SVC(random_state=0))

            print("grid search...")

            if(GRID_SEARCH_RANDOM == True):
                grid = RandomizedSearchCV(model, params, cv=4, n_iter=40)
            else:
                grid = GridSearchCV(model, param_grid=params, cv=4)
                #grid = GridSearchCV(model, param_grid=params, scoring='f1', cv=4)
                #grid = GridSearchCV(model, param_grid=params, scoring='recall', cv=4)

            grid.fit(X_train, Y_train)

            precision, recall, threshold = precision_recall_curve(Y_test, grid.best_estimator_.decision_function(X_test))

            print("model :", model)
            print("best_param: ", grid.best_params_)
            print("best_score: ", grid.best_score_)

            plt.plot(threshold, precision[:-1], label='precision')
            plt.plot(threshold, recall[:-1], label='recall')
            plt.legend()
            plt.show()


            #Y_pred = model_final(grid.best_estimator_, X_test, threshold= -0.00056)

            Y_pred = grid.predict(X_test)

            f1score = f1_score(Y_test, Y_pred)
            recallscore = recall_score(Y_test, Y_pred)
            accuracy = accuracy_score(Y_test, Y_pred, normalize=False)
            accuracy_percent = accuracy_score(Y_test, Y_pred)

            print("f1_score: ", f1score)
            print("recall_score: ", recallscore)
            print("accuracy: ", accuracy, " lengh: ", len(Y_test))
            print("accuracy %: ", accuracy_percent)

            print(confusion_matrix(Y_test, Y_pred))
            print(classification_report(Y_test, Y_pred))

            # evaluation(grid, X_train, X_test, Y_train, Y_test)
        else:

            # accuracy    0.55
            model_test = make_pipeline(PolynomialFeatures(3), SelectKBest(f_classif, k=4),
                                       DecisionTreeClassifier(random_state=0))
            print("model evaluation: ")
            evaluation(model_test, X_train, X_test, Y_train, Y_test)


def model_ADABoost_preprocessing(X_train, X_test, Y_train, Y_test):

    MULTI_MODEL = False
    GRID_SEARCH = True
    GRID_SEARCH_RANDOM = False

    if MULTI_MODEL == True:

        preprocessor = make_pipeline(PolynomialFeatures(3, include_bias=False), SelectKBest(f_classif, k=4))

        RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state = 0))
        AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
        SVM = make_pipeline(preprocessor, StandardScaler() ,SVC(random_state=0))
        KNN= make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())

        dict_of_models = {'RandomForest' : RandomForest,
                          'AdaBoost' : AdaBoost,
                          'SVM' : SVM,
                          'KNN' : KNN
                          }

        for name , model in dict_of_models.items():
            print("model: ", name)
            evaluation(model, X_train, X_test, Y_train, Y_test)


    else:
        if GRID_SEARCH == True:

            params = {'polynomialfeatures__degree': [2, 3, 4],
                      'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
                      'adaboostclassifier__learning_rate': [0.01, 0.05, 0.1, 1],
                      "adaboostclassifier__n_estimators": [10, 20, 50, 80, 100, 120, 150, 180, 200]
                      }

            params = {'polynomialfeatures__degree': [3],
                      'selectkbest__k': [4]
                      }

            model = make_pipeline(PolynomialFeatures(3, include_bias=False), SelectKBest(f_classif, k=4),
                                  AdaBoostClassifier(random_state=0))

            print("grid search...")

            if(GRID_SEARCH_RANDOM == True):
                grid = RandomizedSearchCV(model, params, cv=4, n_iter=40)
            else:
                grid = GridSearchCV(model, param_grid=params, scoring='f1', cv=4)

            grid.fit(X_train, Y_train)

            precision, recall, threshold = precision_recall_curve(Y_test, grid.best_estimator_.decision_function(X_test))

            print("model :", model)
            print("best_param: ", grid.best_params_)
            print("best_score: ", grid.best_score_)

            plt.plot(threshold, precision[:-1], label='precision')
            plt.plot(threshold, recall[:-1], label='recall')
            plt.legend()
            plt.show()


            Y_pred = model_final(grid.best_estimator_, X_test, threshold= -0.00056)

            f1score = f1_score(Y_test, Y_pred)
            recallscore = recall_score(Y_test, Y_pred)
            accuracy = accuracy_score(Y_test, Y_pred, normalize=False)
            accuracy_percent = accuracy_score(Y_test, Y_pred)

            print("f1_score: ", f1score)
            print("recall_score: ", recallscore)
            print("accuracy: ", accuracy, " lengh: ", len(Y_test))
            print("accuracy %: ", accuracy_percent)

            print(confusion_matrix(Y_test, Y_pred))
            print(classification_report(Y_test, Y_pred))

            # evaluation(grid, X_train, X_test, Y_train, Y_test)
        else:

            # accuracy    0.55
            model_test = make_pipeline(PolynomialFeatures(3), SelectKBest(f_classif, k=4),
                                       DecisionTreeClassifier(random_state=0))
            print("model evaluation: ")
            evaluation(model_test, X_train, X_test, Y_train, Y_test)






def get_MLP_prediction(X_train, X_test, Y_train, Y_test):

    FEATURE_OPTIMIZATION = True

    print("MLPClassifier...")

    if(FEATURE_OPTIMIZATION == True):
        preprocessor = make_pipeline(PolynomialFeatures(3, include_bias=False), SelectKBest(f_classif, k=10))
        #mlp_model = make_pipeline(preprocessor, StandardScaler(), MLPClassifier(max_iter=100))
        #mlp_model = make_pipeline(preprocessor, MLPClassifier(max_iter=100)

        mlp_model = MLPClassifier(max_iter=100)


        evaluation(mlp_model, X_train, X_test, Y_train, Y_test)

        predictions = mlp_model.predic(X_train)

    else:

        parameter_space = {
            'polynomialfeatures__degree': [2, 3, 4],
            'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
            'mlpclassifier__hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
            'mlpclassifier__activation': ['tanh', 'relu'],
            'mlpclassifier__solver': ['sgd', 'adam'],
            'mlpclassifier__alpha': [0.0001, 0.05],
            'mlpclassifier__learning_rate': ['constant', 'adaptive'],
        }

        mlp_model = make_pipeline(PolynomialFeatures(3, include_bias=False), SelectKBest(f_classif, k=4),
                                  MLPClassifier(random_state=0, max_iter=100))

        #mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
        #mlp_model = MLPClassifier(max_iter=100)

        clf = GridSearchCV(mlp_model, parameter_space, n_jobs=-1, cv=3)
        clf.fit(X_train, Y_train)

        # Best paramete set
        print('Best parameters found:\n', clf.best_params_)

        # All results
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        predictions = clf.predict(X_test)

    print("Y_test: ", Y_test)
    print("prediction: ", predictions)

    accuracy = accuracy_score(Y_test, predictions)

    print("MLP accuracy: ",accuracy)

    print(confusion_matrix(Y_test, predictions))
    print(classification_report(Y_test, predictions))
    print("=======================================")





def get_DTR_prediction(X_train, X_test, Y_train, Y_test):

    len_y_test = len(Y_test)

    criterion =  ['gini','entropy']
    regressor = [True, False]

    best_accuracy = 0

    for r in regressor:
        for c in criterion:
            ################### DTC ###################
            if r == True:
                Classifier = DecisionTreeRegressor()
            else:
                Classifier = DecisionTreeClassifier(criterion = c)

            Classifier.fit(X_train, Y_train)

            Result_predicted = Classifier.predict(X_test)
            Result_predicted = Result_predicted.reshape(-1, 1)

            result = pd.DataFrame(Result_predicted)
            result.reset_index(drop=True, inplace=True)

            predictions = result[0].to_list()
            accuracy = accuracy_score(Y_test, predictions, normalize=False)

            if (accuracy > best_accuracy):
                best_accuracy = accuracy

    return round(best_accuracy / len_y_test * 100)



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
    for kernel in ['linear','rbf', 'poly']:
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

def get_KNeighbors_prediction(X_train, X_test, Y_train, Y_test):

    len_y_test = len(Y_test)

    weights = ['uniform', 'distance']
    n_neighbors = [3, 5, 7, 10, 25, 50, 100]

    best_accuracy = 0

    for w in weights:
        for n in n_neighbors:
            if((len(X_train) > n) and (len(Y_train) > n)):
                ################### KNeighbors ###################
                Classifier = KNeighborsClassifier(weights = w, n_neighbors = n)
                Classifier.fit(X_train, Y_train)

                Result_predicted = Classifier.predict(X_test)
                Result_predicted = Result_predicted.reshape(-1, 1)

                result = pd.DataFrame(Result_predicted)
                result.reset_index(drop=True, inplace=True)

                predictions = result[0].to_list()
                accuracy = accuracy_score(Y_test, predictions, normalize=False)

                if(accuracy > best_accuracy):
                    best_accuracy = accuracy

    return round(best_accuracy / len_y_test * 100)


def get_RF_prediction(X_train, X_test, Y_train, Y_test):

    len_y_test = len(Y_test)

    n_estimators = [5, 10, 20]
    criterion = ['gini','entropy']

    best_accuracy = 0

    for n in n_estimators:
        for c in criterion:

            ################### Random Forest ###################
            Classifier = RandomForestClassifier(n_estimators = n, criterion = c)
            Classifier.fit(X_train, Y_train)

            Result_predicted = Classifier.predict(X_test)
            Result_predicted = Result_predicted.reshape(-1, 1)

            result = pd.DataFrame(Result_predicted)
            result.reset_index(drop=True, inplace=True)

            predictions = result[0].to_list()
            accuracy = accuracy_score(Y_test, predictions, normalize=False)

            if(accuracy > best_accuracy):
                best_accuracy = accuracy

    return round(best_accuracy / len_y_test * 100)

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
                if be == True:
                    Classifier = AdaBoostClassifier(n_estimators = n, learning_rate=lr)
                else:
                    Classifier = AdaBoostClassifier(base_estimator = DecisionTreeClassifier() ,n_estimators=n, learning_rate=lr)
                Classifier.fit(X_train, Y_train)

                Result_predicted = Classifier.predict(X_test)
                Result_predicted = Result_predicted.reshape(-1, 1)

                result = pd.DataFrame(Result_predicted)
                result.reset_index(drop=True, inplace=True)

                predictions = result[0].to_list()
                accuracy = accuracy_score(Y_test, predictions, normalize=False)

                if (accuracy > best_accuracy):
                    best_accuracy = accuracy

    return round(best_accuracy / len_y_test * 100)

def get_GRBOOST_prediction(X_train, X_test, Y_train, Y_test):

    len_y_test = len(Y_test)

    n_estimators = [3, 5, 7, 10, 25, 50, 75, 100]

    best_accuracy = 0

    for n in n_estimators:

        ################### GradientBoostingClassifier ###################
        Classifier = GradientBoostingClassifier(n_estimators = n)
        Classifier.fit(X_train, Y_train)

        Result_predicted = Classifier.predict(X_test)
        Result_predicted = Result_predicted.reshape(-1, 1)

        result = pd.DataFrame(Result_predicted)
        result.reset_index(drop=True, inplace=True)

        predictions = result[0].to_list()
        accuracy = accuracy_score(Y_test, predictions, normalize=False)

        if (accuracy > best_accuracy):
            best_accuracy = accuracy

    return round(best_accuracy / len_y_test * 100)

def get_GNaiveB_prediction(X_train, X_test, Y_train, Y_test):

    len_y_test = len(Y_test)

    n_estimators = [3, 5, 7, 10, 25, 50, 75, 100]
    learning_rate = [0.01, 0.05, 0.1, 0.5, 1]

    best_accuracy = 0

    for n in n_estimators:
        for lr in learning_rate:
            ################### AdaBoost Classifier with NaiveBayes base ###################
            Classifier = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators = n, learning_rate=lr)
            Classifier.fit(X_train, Y_train)

            Result_predicted = Classifier.predict(X_test)
            Result_predicted = Result_predicted.reshape(-1, 1)

            result = pd.DataFrame(Result_predicted)
            result.reset_index(drop=True, inplace=True)

            predictions = result[0].to_list()
            accuracy = accuracy_score(Y_test, predictions, normalize=False)

            if (accuracy > best_accuracy):
                best_accuracy = accuracy

    return round(best_accuracy / len_y_test * 100)