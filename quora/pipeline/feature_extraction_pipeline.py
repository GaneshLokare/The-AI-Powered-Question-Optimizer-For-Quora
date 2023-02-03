from quora.feature_extraction.basic_feature_extract import BasicFeatures
from quora.feature_extraction.advanced_feature_extract import AdvancedFeatures
from quora.feature_extraction.nlp_feature_extract import NLPFeatures
from quora.feature_extraction.final_features import FinalFeatures
from quora.exception import QuoraException
import sys

class FeatureExtraction:
    def __init__(self):
        pass

    def extract_all_features(self):
        try:
            basic_features = BasicFeatures().basic_feature_extraction()
            advanced_features = AdvancedFeatures().adv_features_extraction()
            nlp_features = NLPFeatures().npl_feature_extraction()
            final_features = FinalFeatures().merge_features()
            
            return basic_features, advanced_features, nlp_features, final_features
        except Exception as e:
            raise QuoraException(e, sys)


