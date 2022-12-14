import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import re

import distance
import spacy
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import os.path as path

from quora.constants.file_paths import Data, model_path, q1_Feature_Path, new_questions_path
from quora.constants.data_constants import Number_of_rows

from flask import Flask,request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')



loaded_model = pickle.load(open(model_path, 'rb'))


@app.route('/check_simillar_question',methods=['POST'])
def check_simillar_question():

    q = request.form.to_dict()
    que = q['Question']

    data = pd.read_csv(Data, nrows=Number_of_rows)
    df = data.drop(['qid1','qid2'],axis = 1)
    df['question2'] = que
    # drop null values
    df.fillna("0",inplace = True)

    def new_features(df):
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
        return df
    df = new_features(df)  

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

    # To get the results in 4 decemal points
    SAFE_DIV = 0.0001 

    STOP_WORDS = stopwords.words("english")


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

    def extract_features(df):
        # preprocessing each question
        df["question1"] = df["question1"].fillna("").apply(preprocess)
        df["question2"] = df["question2"].fillna("").apply(preprocess)

        
        
        # Merging Features with dataset
        
        token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
        
        df["cwc_min"]       = list(map(lambda x: x[0], token_features))
        df["cwc_max"]       = list(map(lambda x: x[1], token_features))
        df["csc_min"]       = list(map(lambda x: x[2], token_features))
        df["csc_max"]       = list(map(lambda x: x[3], token_features))
        df["ctc_min"]       = list(map(lambda x: x[4], token_features))
        df["ctc_max"]       = list(map(lambda x: x[5], token_features))
        df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
        df["first_word_eq"] = list(map(lambda x: x[7], token_features))
        df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
        df["mean_len"]      = list(map(lambda x: x[9], token_features))
    
        #Computing Fuzzy Features and Merging with Dataset
        
        # do read this blog: http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
        # https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings
        # https://github.com/seatgeek/fuzzywuzzy
        

        df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
        # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and 
        # then joining them back into a string We then compare the transformed strings with a simple ratio().
        df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
        df["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
        df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
        df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
        return df

    df = extract_features(df)

    df['question2'] = df['question2'].apply(lambda x: str(x))

    # get tf-idf for all words
    # merge texts
    questions = list(df['question2'])

    tfidf = TfidfVectorizer(lowercase=False, )
    tfidf.fit_transform(questions)

    # dict key:word and value:tf-idf score
    word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

    # convert each question to a weighted average of word2vec vectors
    nlp = spacy.load("venv\en_core_web_lg\en_core_web_lg-3.4.1")
    vecs2 = []
    qu2  = df['question2'][0]
    doc2 = nlp(qu2) 
    mean_vec2 = np.zeros([len(doc2), len(doc2[0].vector)])
    for word2 in doc2:
            # word2vec
        vec2 = word2.vector
            # fetch df score
        try:
            idf = word2tfidf[str(word2)]
        except:
            idf = 0
            # compute final vec
        mean_vec2 += vec2 * idf
    mean_vec2 = mean_vec2.mean(axis=0)
    vecs2.append(mean_vec2)
    lst = list(vecs2)
    df3_q2 = pd.DataFrame(lst)
    for i in range(len(df)):
        df3_q2.loc[i] = df3_q2.loc[0]
    lst = []
    for i in range(len(df)):
        lst.append(i)
    df3_q2['id'] = lst

    df3_q1 = pd.read_csv(q1_Feature_Path, nrows = Number_of_rows)
    df3_q1.columns = df3_q1.columns.values + '_x'

    q2_cols = []
    for i in df3_q2.columns:
        q2_cols.append(str(i))
    df3_q2.columns = q2_cols
    df3_q2.columns = df3_q2.columns.values + '_y'

    df3 = df.drop(['question1','question2','is_duplicate','simillar_words'],axis=1)

    df3_q1['id']=df['id']
    df3_q2['id']=df['id']

    df2  = df3_q1.merge(df3_q2, on='id',how='left')
    result  = df3.merge(df2, on='id',how='left')

    result.drop([ 'id','id_y'], axis = 1, inplace = True)

    # after we read from sql table each entry was read it as a string
    # we convert all the features into numaric before we apply any model
    cols = list(result.columns)
    for i in cols:
        result[i] = result[i].apply(pd.to_numeric)

    result = xgb.DMatrix(result)

    pred_y = loaded_model.predict(result)
    

    similar_questions = []
    probability = []
    for i in range(len(pred_y)):
        if pred_y[i] >0.5:
            res = (data.iloc[i]['question1'])
            similar_questions.append(res)
            prob = round(pred_y[i] * 100,2)
            probability.append(prob)
            
    if len(similar_questions) == 0:
    # Open File
        file_path = path.abspath(path.join(new_questions_path))
        with open(file_path, 'a') as f_object:

# Write data to file
            f_object.write(que + "\n")
            f_object.close()   
    
    if len(similar_questions) == 0:
        return render_template('output1.html')
    else:
        return render_template('output.html',Similar_questions = similar_questions )
        

    

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)


        


