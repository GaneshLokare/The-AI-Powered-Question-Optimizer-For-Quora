from quora.pipeline.feature_extraction_pipeline import FeatureExtraction
from quora.pipeline.training_pipeline import ModelTraining
#from quora.pipeline.prediction_pipeline import Prediction

#que = "How can I see all my Youtube comments?"
FeatureExtraction.extract_all_features()
ModelTraining.model_training()
#Prediction.check_simillar_question(que)