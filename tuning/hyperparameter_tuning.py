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

            params = {
                'polynomialfeatures__degree': [2, 3, 4],
                'selectkbest__k': [4, 5, 6, 7, 8, 9, 10],
                #'svc__kernel': ['linear', 'rbf', 'poly'],
                #'svc__C': [0.1, 1, 10, 100, 500, 1000],
                #'svc__degree': [10, 20, 50, 100, 200, 400, 600, 800],
                #'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001]
                      }

            #params = {'polynomialfeatures__degree': [3],
            #          'selectkbest__k': [4]
            #          }

            model = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=8),
                                  StandardScaler(),
                                  SVC(random_state=1, kernel='rbf', C=1, gamma= 0.01))

            print("grid search...")

            if(GRID_SEARCH_RANDOM == True):
                grid = RandomizedSearchCV(model, params, cv=4, n_iter=100)
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

            print("Test: ", Y_test)
            print("Pred: ",Y_pred)

            print("##########################################################")

            Y_pred = model_final(grid.best_estimator_, X_test, threshold= 0.6011)

            prediction = []
            for i in range(len(Y_pred)):
                if Y_pred[i] == True:
                    prediction.append(1)
                else:
                    prediction.append(0)


            f1score = f1_score(Y_test, prediction)
            recallscore = recall_score(Y_test, prediction)
            accuracy = accuracy_score(Y_test, prediction, normalize=False)
            accuracy_percent = accuracy_score(Y_test, prediction)

            print("f1_score: ", f1score)
            print("recall_score: ", recallscore)
            print("accuracy: ", accuracy, " lengh: ", len(Y_test))
            print("accuracy %: ", accuracy_percent)

            print(confusion_matrix(Y_test, prediction))
            print(classification_report(Y_test, prediction))

            print("Test: ", Y_test)
            print("Pred: ", prediction)

            print("##########################################################")







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
