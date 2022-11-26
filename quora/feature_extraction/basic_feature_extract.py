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
        try:
            logging.info("Extracting basic features")
            df = pd.read_csv(Data, nrows = Number_of_rows)

            # extract new features
            df.fillna("0",inplace = True)
            df['q1len'] = df['question1'].str.len() 
            df['q2len'] = df['question2'].str.len()
            df['q1+q2_len'] = df['q1len'] + df['q2len']
            df['q1-q2_len'] = abs(df['q1len'] - df['q2len'])
            df['q1_words'] = df['question1'].str.split().str.len()
            df['q2_words'] = df['question2'].str.split().str.len()
            df['total_words'] = df['q1_words'] + df['q2_words']
            df['words_difference'] = abs(df.q1_words - df.q2_words)
            df['simillar_words'] = df.apply(lambda x: set(x['question1'].split()) & set(x['question2'].split()),axis=1)
            df['simillar_words_count'] = df['simillar_words'].str.len()
            df['word_share'] = df['simillar_words_count'] / df['total_words']

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
            df['first_word_same'] = first_word_same(df['question1'],df['question2'])
            df = df.drop(['qid1','qid1','question1','question2','simillar_words'], axis = 1)
            # save all features to csv
            feature_path = path.abspath(path.join(Basic_Feature_Path))
            print("Basic features extraction done")
            return df.to_csv(feature_path,index=False)
            
        except  Exception as e:
                raise  QuoraException(e,sys)

Basic_Features.basic_feature_extraction()