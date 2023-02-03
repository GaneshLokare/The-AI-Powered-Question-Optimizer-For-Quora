# Import required libraries
import pandas as pd 
import os.path as path
import sys
from quora.logger import logging
from quora.constants.file_paths import Data
from quora.constants.file_paths import Basic_Feature_Path
from quora.constants.data_constants import Number_of_rows
from quora.exception import QuoraException




class BasicFeatures:
    def __init__(self):
       self.df = pd.read_csv(Data, nrows = Number_of_rows)

    def basic_feature_extraction(self):
        '''Basic feature extraction'''
        try:
            logging.info("{} data points are selected".format(Number_of_rows))
            
            # extract new features from available data
            self.df.fillna("0",inplace = True)
            self.df['q1len'] = self.df['question1'].str.len() # length of question1
            self.df['q2len'] = self.df['question2'].str.len() # length of question2
            self.df['q1+q2_len'] = self.df['q1len'] + self.df['q2len'] # total length of question1 and question2.
            self.df['q1-q2_len'] = abs(self.df['q1len'] - self.df['q2len']) # abs lenght difference between question1 and question2.
            self.df['q1_words'] = self.df['question1'].str.split().str.len() # Total number of words in question1.
            self.df['q2_words'] = self.df['question2'].str.split().str.len() # Total number of words in question2.
            self.df['total_words'] = self.df['q1_words'] + self.df['q2_words'] # Total number of words in question1 and question2.
            self.df['words_difference'] = abs(self.df.q1_words - self.df.q2_words) # number of words difference between question1 and question2.
            self.df['simillar_words'] = self.df.apply(lambda x: set(x['question1'].split()) & set(x['question2'].split()),axis=1) # Similar words between question1 and questin2.
            self.df['simillar_words_count'] = self.df['simillar_words'].str.len() # Number of similar words between question1 and questin2.
            self.df['word_share'] = self.df['simillar_words_count'] / self.df['total_words'] # ratio of similar_word_count and total_words
            
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
            self.df['first_word_same'] = first_word_same(self.df['question1'],self.df['question2']) # applying above function
            self.df = self.df.drop(['qid1','qid2','question1','question2','simillar_words'], axis = 1) # keeping only extracted features
            # save extracted features to csv
            feature_path = path.abspath(path.join(Basic_Feature_Path))
            logging.info("Basic features extraction done")
            return self.df.to_csv(feature_path,index=False)
            
            
            
        except  Exception as e:
                raise  QuoraException(e,sys)



