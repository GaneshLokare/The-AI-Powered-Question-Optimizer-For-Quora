import pandas as pd
import os.path as path
from quora.constants.file_paths import Final_feature_path
from quora.constants.file_paths import Basic_Feature_Path
from quora.constants.file_paths import Adv_features_path
from quora.constants.file_paths import q1_Feature_Path
from quora.constants.file_paths import q2_Feature_Path
from quora.exception import QuoraException
import sys





class Final_Features:
    def __init__(self):
       pass

    def final_features():
        try:
            basic_df = pd.read_csv(Basic_Feature_Path)
            adv_df = pd.read_csv(Adv_features_path)
            q1_featurs = pd.read_csv(q1_Feature_Path)
            q2_featurs = pd.read_csv(q2_Feature_Path)

            q1_featurs['id']=basic_df['id']
            q2_featurs['id']=basic_df['id']

            df1 = basic_df.merge(adv_df, on='id',how='left')
            df2  = q1_featurs.merge(q2_featurs, on='id',how='left')
            result  = df1.merge(df2, on='id',how='left')

            print("All features merged and saved to csv")

            feature_path = path.abspath(path.join(Final_feature_path))
            return result.to_csv(feature_path,index=False)
        except  Exception as e:
                raise  QuoraException(e,sys)
