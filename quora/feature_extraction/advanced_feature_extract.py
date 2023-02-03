# Import required libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import distance
import os.path as path
import sys

from quora.logger import logging
from quora.constants.file_paths import Data
from quora.constants.file_paths import Adv_features_path
from quora.constants.data_constants import Number_of_rows
from quora.exception import QuoraException


class AdvancedFeatures:
    def __init__(self):
       self.df = pd.read_csv(Data, nrows = Number_of_rows)

    def adv_features_extraction(self):
        '''extract advanced features'''
        try:
            
            # To get the results in 4 decemal points
            SAFE_DIV = 0.0001 
            # save all stopwords in stop_words variable
            STOP_WORDS = stopwords.words("english")

            # function to Removing html tags, Removing Punctuations, Performing stemming and Removing Stopwords.
            def preprocess(x):
                x = str(x).lower()
                x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                                    .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                                    .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                                    .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                                    .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                                    .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                                    .replace("€", " euro ").replace("'ll", " will")
                x = re.sub(r"([0-9]+)000000", r"\1m", x)
                x = re.sub(r"([0-9]+)000", r"\1k", x)
                
                
                porter = PorterStemmer()
                pattern = re.compile('\W')
                
                if type(x) == type(''):
                    x = re.sub(pattern, ' ', x)
                
                
                if type(x) == type(''):
                    x = porter.stem(x)
                    example1 = BeautifulSoup(x)
                    x = example1.get_text()
                        
                
                return x

            # function to extract advanced features.
            def get_token_features(q1, q2):
                token_features = [0.0]*10
                
                # Converting the Sentence into Tokens: 
                q1_tokens = q1.split()
                q2_tokens = q2.split()

                if len(q1_tokens) == 0 or len(q2_tokens) == 0:
                    return token_features
                # Get the non-stopwords in Questions
                q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
                q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
                
                #Get the stopwords in Questions
                q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
                q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
                
                # Get the common non-stopwords from Question pair
                common_word_count = len(q1_words.intersection(q2_words))
                
                # Get the common stopwords from Question pair
                common_stop_count = len(q1_stops.intersection(q2_stops))
                
                # Get the common Tokens from Question pair
                common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
                
                
                token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
                token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
                token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
                token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
                token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
                token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
                
                # Last word of both question is same or not
                token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
                
                # First word of both question is same or not
                token_features[7] = int(q1_tokens[0] == q2_tokens[0])
                
                token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
                
                #Average Token Length of both Questions
                token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
                return token_features

            # get the Longest Common sub string

            def get_longest_substr_ratio(a, b):
                strs = list(distance.lcsubstrings(a, b))
                if len(strs) == 0:
                    return 0
                else:
                    return len(strs[0]) / (min(len(a), len(b)) + 1)

            def extract_features():
                # preprocessing each question
                self.df["question1"] = self.df["question1"].fillna("").apply(preprocess)
                self.df["question2"] = self.df["question2"].fillna("").apply(preprocess)

                
                
                # Merging Features with dataset
                
                token_features = self.df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
                
                self.df["cwc_min"]       = list(map(lambda x: x[0], token_features))
                self.df["cwc_max"]       = list(map(lambda x: x[1], token_features))
                self.df["csc_min"]       = list(map(lambda x: x[2], token_features))
                self.df["csc_max"]       = list(map(lambda x: x[3], token_features))
                self.df["ctc_min"]       = list(map(lambda x: x[4], token_features))
                self.df["ctc_max"]       = list(map(lambda x: x[5], token_features))
                self.df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
                self.df["first_word_eq"] = list(map(lambda x: x[7], token_features))
                self.df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
                self.df["mean_len"]      = list(map(lambda x: x[9], token_features))
            
                #Computing Fuzzy Features and Merging with Dataset
                

                self.df["token_set_ratio"]       = self.df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
                # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and 
                # then joining them back into a string We then compare the transformed strings with a simple ratio().
                self.df["token_sort_ratio"]      = self.df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
                self.df["fuzz_ratio"]            = self.df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
                self.df["fuzz_partial_ratio"]    = self.df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
                self.df["longest_substr_ratio"]  = self.df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
                self.df = self.df.drop(['qid1','qid2','question1','question2','is_duplicate'], axis = 1) # keeping only extracted features
                return self.df
            final = extract_features()
            logging.info("advanced features extraction done")
            # save extracted features to csv
            feature_path = path.abspath(path.join(Adv_features_path))
            return final.to_csv(feature_path,index=False)
            
            
            
        except  Exception as e:
                raise  QuoraException(e,sys)


