from quora.pipeline.feature_extraction_pipeline import FeatureExtraction
from quora.pipeline.training_pipeline import ModelTraining
from quora.pipeline.prediction_pipeline import Prediction

#que = "How can I see all my Youtube comments?"

#features = FeatureExtraction()
#features.extract_all_features()

#training = ModelTraining()
#training.model_training()

pred = Prediction()
pred.check_simillar_question("How can I see all my Youtube comments?")