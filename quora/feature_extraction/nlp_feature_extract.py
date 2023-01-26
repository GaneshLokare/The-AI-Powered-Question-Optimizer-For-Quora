# import required libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from quora.logger import logging
from quora.constants.file_paths import Data
from quora.constants.file_paths import q1_Feature_Path
from quora.constants.file_paths import q2_Feature_Path
from quora.constants.data_constants import Number_of_rows
from quora.exception import QuoraException
import sys

import os.path as path


# load spacy liabrary
nlp = spacy.load("venv\en_core_web_lg\en_core_web_lg-3.4.1")

class NLP_Features:
    def __init__(self):
       pass


    def npl_feature_extraction():
        '''extract nlp features'''
        try:
            # load data
            df = pd.read_csv(Data, nrows = Number_of_rows)
            
            # convert all questions into string
            df['question1'] = df['question1'].apply(lambda x: str(x))
            df['question2'] = df['question2'].apply(lambda x: str(x))
            questions = list(df['question1']) + list(df['question2'])

            # get tf-idf for all words
            # merge texts
            tfidf = TfidfVectorizer(lowercase=False, )
            tfidf.fit_transform(questions)

            # dict key:word and value:tf-idf score
            word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

            vecs1 = []
        # https://github.com/noamraph/tqdm
        # tqdm is used to print the progrss bar
            for qu1 in (list(df['question1'])):
                doc1 = nlp(qu1) 
            # 300 is the number of dimensions of vectors 
                mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)])
                for word1 in doc1:
            # word2vec
                    vec1 = word1.vector
                # fetch df score
                    try:
                        idf = word2tfidf[str(word1)]
                    except:
                        idf = 0
                # compute final vec
                    mean_vec1 += vec1 * idf
                mean_vec1 = mean_vec1.mean(axis=0)
                vecs1.append(mean_vec1)
            df['q1_feats_m'] = list(vecs1)

            vecs2 = []
            for qu2 in (list(df['question2'])):
                doc2 = nlp(qu2) 
                mean_vec2 = np.zeros([len(doc2), len(doc2[0].vector)])
                for word2 in doc2:
                    # word2vec
                    vec2 = word2.vector
                    # fetch df score
                    try:
                        idf = word2tfidf[str(word2)]
                    except:
                        #print word
                        idf = 0
                    # compute final vec
                    mean_vec2 += vec2 * idf
                mean_vec2 = mean_vec2.mean(axis=0)
                vecs2.append(mean_vec2)
            df['q2_feats_m'] = list(vecs2)

            # convert spacy vectors into dataframe
            df3 = df.drop(['qid1','qid2','question1','question2','is_duplicate'],axis=1)
            df3_q1 = pd.DataFrame(df3.q1_feats_m.values.tolist(), index= df3.index)
            df3_q2 = pd.DataFrame(df3.q2_feats_m.values.tolist(), index= df3.index)

            q1_path = path.abspath(path.join(q1_Feature_Path))
            q2_path = path.abspath(path.join(q2_Feature_Path))
            # save extracted features
            q1_save = df3_q1.to_csv(q1_path,index=False) 
            q2_save = df3_q2.to_csv(q2_path,index=False)

            logging.info("NPL features extraction done")

            return q1_save, q2_save
        except  Exception as e:
                raise  QuoraException(e,sys)

