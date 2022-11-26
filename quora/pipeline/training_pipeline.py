import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import log_loss
import pickle
import os.path as path
import sys

from quora.logger import logging 
from quora.constants.file_paths import Final_feature_path
from quora.constants.file_paths import model_path
from quora.exception import QuoraException



class ModelTraining:
    def __init__(self):
       pass


    def model_training():
        try:
            logging.info("Model Training start")
            data = pd.read_csv(Final_feature_path)

            y_true = data['is_duplicate']
            data.drop([ 'id','is_duplicate'], axis=1, inplace=True)

            cols = list(data.columns)
            for i in cols:
                data[i] = data[i].apply(pd.to_numeric)

            y_true = list(map(int, y_true.values))

            X_train,X_test, y_train, y_test = train_test_split(data, y_true, stratify=y_true, test_size=0.3)
            labels = np.array([0,1])
            
        
            params = {}
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
            params['eta'] = 0.02
            params['max_depth'] = 4

            d_train = xgb.DMatrix(X_train, label = y_train)
            d_test = xgb.DMatrix(X_test, label = y_test)

            watchlist = [(d_train, 'train'), (d_test, 'valid')]

            bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=20, verbose_eval=10)

            d_test1 = xgb.DMatrix(X_test)
            predict_y = bst.predict(d_test1)
            logging.info("The test log loss is: {} ".format(log_loss(y_test, predict_y, labels=labels, eps=1e-15)))
            print("The test log loss is:",log_loss(y_test, predict_y, labels=labels, eps=1e-15))

            print("Model training done and model saved")
            
            feature_path = path.abspath(path.join(model_path))
            return pickle.dump(bst, open(feature_path, 'wb'))

        except  Exception as e:
                raise  QuoraException(e,sys)

