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
from quora.constants.data_constants import Number_of_rows
from quora.exception import QuoraException



class ModelTraining:
    def __init__(self):
       self.Final_feature_path = Final_feature_path
       self.Number_of_rows = Number_of_rows
       self.model_path = model_path


    def model_training(self):
        try:
            # load final features data
            data = pd.read_csv(self.Final_feature_path)

            # create dependant and independant variables
            y_true = data['is_duplicate']
            data.drop([ 'id','is_duplicate'], axis=1, inplace=True)

            # convert all the features into numaric before we apply any model
            cols = list(data.columns)
            for i in cols:
                data[i] = data[i].apply(pd.to_numeric)

            # convert-all-strings-in-a-list-to-int
            y_true = list(map(int, y_true.values))

            # split data into train and test set (70:30)
            X_train,X_test, y_train, y_test = train_test_split(data, y_true, stratify=y_true, test_size=0.3)
            labels = np.array([0,1])
            
            # Training
            # set parameters
            params = {}
            # The objective is set to 'binary:logistic' which means that the model will be used for binary classification with a logistic loss function
            params['objective'] = 'binary:logistic'
            # The eval_metric is set to 'logloss' which means that the model's performance will be evaluated using the logarithmic loss metric.
            params['eval_metric'] = 'logloss'
            params['eta'] = 0.02 # learning rate
            params['max_depth'] = 4 # maximum depth of the  decision tree.

            # DMatrix objects are created for training and testing data respectively.
            # The DMatrix object is a internal data structure that used by XGBoost which is optimized for both memory efficiency and training speed.
            d_train = xgb.DMatrix(X_train, label = y_train)
            d_test = xgb.DMatrix(X_test, label = y_test)

            # watchlist is created which contains the training and validation data.
            # This will allow XGBoost to perform early stopping if the performance on the validation set does not improve after a certain number of rounds. 
            watchlist = [(d_train, 'train'), (d_test, 'valid')]

            # tarin model
            bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=20, verbose_eval=10)

            # prediction on test data.
            d_test1 = xgb.DMatrix(X_test)
            predict_y = bst.predict(d_test1)
            logging.info("The test log loss is: {} ".format(log_loss(y_test, predict_y, labels=labels, eps=1e-15)))
            
            logging.info("Model tained on {} data points".format(self.Number_of_rows))
            logging.info("Model training done and model saved")
            
            # save model
            feature_path = path.abspath(path.join(self.model_path))
            return pickle.dump(bst, open(feature_path, 'wb'))

        except  Exception as e:
                raise  QuoraException(e,sys)

