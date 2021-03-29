import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



def get_lightGBM_prediction(x_train, x_test, y_train, y_test):

    #y_train['0'] = y_train['0'].apply(np.int64)

    #y_train['0'] = y_train['0'].apply(lambda x: int(x))
    y_train = y_train.astype(int)

    for i in range(len(y_test)):
        y_test[i] = int(y_test[i])

    # Importing the dataset
    #dataset = pd.read_csv('./database/Social_Network_Ads.csv')
    #X = dataset.iloc[:, [2, 3]].values
    #y = dataset.iloc[:, 4].values

    # Splitting the dataset into the Training set and Test set
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Feature Scaling
    # sc = StandardScaler()
    # x_train = sc.fit_transform(x_train)
    # x_test = sc.transform(x_test)

    #x_train = x_train.drop(columns=["High","Low","Open","Close","Volume","Adj Close","macd","macd_signal_line","macd_hist","rsi","cci","adx","+DI","-DI"])
    #x_test = x_test.drop(columns=["High","Low","Open","Close","Volume","Adj Close","macd","macd_signal_line","macd_hist","rsi","cci","adx","+DI","-DI"])

    #x_train = x_train.drop(columns=["High","Low","Adj Close","macd_hist"])
    #x_test = x_test.drop(columns=["High","Low","Adj Close","macd_hist"])

    #x_train = x_train.drop(columns=["High","Low","Open","Close","Volume","Adj Close","macd_signal_line","macd_hist"])
    #x_test = x_test.drop(columns=["High","Low","Open","Close","Volume","Adj Close","macd_signal_line","macd_hist"])

    #x_train = x_train.drop(columns=["High","Low","Open","Close","Adj Close","macd_signal_line","macd_hist","+DI","-DI"])
    #x_test = x_test.drop(columns=["High","Low","Open","Close","Adj Close","macd_signal_line","macd_hist","+DI","-DI"])

    #lgb_train = lgb.Dataset(x_train, label=y_train)
    #lgb_test = lgb.Dataset(x_test, label=y_test)
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_test = lgb.Dataset(x_test, y_test)

    params_1 = {}
    params_1['learning_rate'] = 0.003
    params_1['boosting_type'] = 'gbdt'
    params_1['objective'] = 'binary'
    params_1['metric'] = 'binary_logloss'
    params_1['sub_feature'] = 0.5
    params_1['num_leaves'] = 10
    params_1['min_data'] = 50
    params_1['max_depth'] = 10

    params_2 = {}
    params_2['class_weight'] = 'balanced'
    params_2['drop_rate'] =0.9
    params_2['min_data_in_leaf'] =100
    params_2['max_bin'] =255
    params_2['n_estimators'] =500
    params_2['min_sum_hessian_in_leaf'] =1
    params_2['importance_type'] ='gain'
    params_2['learning_rate'] =0.1
    params_2['bagging_fraction'] =0.85
    params_2['colsample_bytree'] =1.0
    params_2['feature_fraction'] =0.1
    params_2['lambda_l1'] =5.0
    params_2['lambda_l2'] =3.0
    params_2['max_depth'] =9
    params_2['min_child_samples'] =55
    params_2['min_child_weight'] =5.0
    params_2['min_split_gain'] =0.1
    params_2['num_leaves'] =45
    params_2['subsample'] =0.75

    params_3 = {}
    params_3['application'] = 'binary'
    params_3['objective'] = 'binary'
    params_3['metric'] = 'auc'
    params_3['is_unbalance'] = 'true'
    params_3['boosting'] = 'gbdt'
    params_3['num_leaves'] = 31
    params_3['feature_fraction'] = 0.5
    params_3['bagging_fraction'] = 0.5
    params_3['bagging_freq'] = 20
    params_3['learning_rate'] = 0.05
    params_3['verbose'] = 0

    params_4 = {}
#    params_4['learning_rate'] = 0.003
#    params_4['boosting_type'] = 'gbdt'
    params_4['objective'] = 'binary'
    params_4['metric'] = 'binary_logloss'
    params_4['learning_rate'] = 0.003

    params_5 = {}
    params_5['boosting_type'] = 'gbdt'
    params_5['objective'] = 'binary'
    params_5['metric'] = 'binary_logloss'
    params_5['min_data_in_leaf'] = 100
    params_5['feature_fraction'] = 0.8

    params_6 = {}
    params_6['boosting_type'] = 'gbdt'
    params_6['objective'] = 'binary'
    params_6['metric'] = 'binary_logloss'
    params_6['min_data_in_leaf'] = 100
    params_6['feature_fraction'] = 0.8
    params_6['learning_rate'] = 0.4
    params_6['max_depth'] = 15
    params_6['num_leaves'] = 32
    params_6['feature_fraction'] = 0.8
    params_6['subsample'] = 0.2
    params_6['objective'] = 'binary'
    params_6['metric'] = 'auc'
    params_6['is_unbalance'] = True
    params_6['bagging_freq'] = 5
    #params_6['boosting'] = 'dart'
    params_6['num_boost_round'] = 300
    #params_6['early_stopping_rounds'] = 30


    #params_5['objective'] = 'multiclass'
    #params_5['metric'] = 'multi_logloss'
    #params_5['num_class'] = 1



    #model_params = {'boosting_type': 'gbdt', 'colsample_bytree': 0.9500000000000001, 'learning_rate': 0.1,
    #                'max_depth': -1, 'min_child_samples': 89, 'min_child_weight': 0.001, 'min_split_gain': 0.0,
    #                'n_jobs': -1, 'num_leaves': 127, 'objective': 'binary', 'random_state': 0, 'seed': 0,
    #                'reg_alpha': 0.030559559479313415,
    #                'reg_lambda': 1.0949294490500017e-06, 'subsample': 0.8, 'subsample_for_bin': 200000,
    #                'subsample_freq': 4, 'verbose': -1, 'metric': 'binary_logloss'}

    #clf = lgb.train(params_3, lgb_train, 10000)
    clf = lgb.train(params_6, lgb_train, num_boost_round=500, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=10)


    #Prediction
    y_pred=clf.predict(x_test)
    #convert into binary values
    for i in range(0,len(y_pred)):
        if y_pred[i]>=.5:       # setting threshold to .5
           y_pred[i]=int(1)
        else:
           y_pred[i]=int(0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Accuracy
    accuracy = accuracy_score(y_pred, y_test, normalize=False)

    print(cm)


    return round(accuracy / len(y_test) * 100)
