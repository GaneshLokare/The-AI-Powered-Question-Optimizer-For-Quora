from quora.feature_extraction.basic_feature_extract import Basic_Features
from quora.feature_extraction.advanced_feature_extract import Advanced_Features
from quora.feature_extraction.nlp_feature_extract import NLP_Features
from quora.feature_extraction.final_features import Final_Features
from quora.exception import QuoraException
import sys



class FeatureExtraction:
    def __init__(self):
       pass

    def extract_all_features():
        '''extract and merge all features'''
        try:

            def basic_features():
                return Basic_Features.basic_feature_extraction()

            def advanced_features():
                return Advanced_Features.adv_features_extraction()

            def nlp_features():
                return NLP_Features.npl_feature_extraction()

            def final_features():
                return Final_Features.final_features()
            
            return basic_features(),advanced_features(),nlp_features(),final_features()
        except  Exception as e:
                raise  QuoraException(e,sys)
