# Import required libraries
import pandas as pd 
import os.path as path
import sys
from quora.logger import logging
from quora.constants.file_paths import Data
from quora.constants.file_paths import Basic_Feature_Path
from quora.constants.data_constants import Number_of_rows
from quora.exception import QuoraException




class Basic_Features:
    def __init__(self):
       pass

    def basic_feature_extraction():
        '''Basic feature extraction'''
        try:
            logging.info("{} data points are selected".format(Number_of_rows))
            # load data
            df = pd.read_csv(Data, nrows = Number_of_rows)

            # extract new features from available data
            df.fillna("0",inplace = True)
            df['q1len'] = df['question1'].str.len() # length of question1
            df['q2len'] = df['question2'].str.len() # length of question2
            df['q1+q2_len'] = df['q1len'] + df['q2len'] # total length of question1 and question2.
            df['q1-q2_len'] = abs(df['q1len'] - df['q2len']) # abs lenght difference between question1 and question2.
            df['q1_words'] = df['question1'].str.split().str.len() # Total number of words in question1.
            df['q2_words'] = df['question2'].str.split().str.len() # Total number of words in question2.
            df['total_words'] = df['q1_words'] + df['q2_words'] # Total number of words in question1 and question2.
            df['words_difference'] = abs(df.q1_words - df.q2_words) # number of words difference between question1 and question2.
            df['simillar_words'] = df.apply(lambda x: set(x['question1'].split()) & set(x['question2'].split()),axis=1) # Similar words between question1 and questin2.
            df['simillar_words_count'] = df['simillar_words'].str.len() # Number of similar words between question1 and questin2.
            df['word_share'] = df['simillar_words_count'] / df['total_words'] # ratio of similar_word_count and total_words
            
            # function to check, is first word of both question1 and question2 is same or not.
            def first_word_same(question1, question2):
                first_words_1 = question1.apply(lambda x: x.split()[0].lower())
                first_words_2 = question2.apply(lambda x: x.split()[0].lower())
                lst = []
                for q1,q2 in zip(first_words_1,first_words_2):
                    if q1==q2:
                        q1 = 1
                    else:
                        q1 = 0
                    lst.append(q1)
                return lst
            df['first_word_same'] = first_word_same(df['question1'],df['question2']) # applying above function
            df = df.drop(['qid1','qid2','question1','question2','simillar_words'], axis = 1) # keeping only extracted features
            # save extracted features to csv
            feature_path = path.abspath(path.join(Basic_Feature_Path))
            logging.info("Basic features extraction done")
            return df.to_csv(feature_path,index=False)
            
        except  Exception as e:
                raise  QuoraException(e,sys)
