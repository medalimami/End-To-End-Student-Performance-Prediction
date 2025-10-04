import sys
import pandas as pd
from send2trash.util import preprocess_paths

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocess_path = 'artifacts/data_transformer.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocess_path)
            data_scaled=preprocessor.transform(features)
            data_predicted=model.predict(data_scaled)
            return data_predicted
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,
                 lunch,test_preparation_course,reading_score,writing_score,average,total_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        self.average = average,
        self.total_score = total_score

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score],
                'Average': [self.average],
                'Total_Score': [self.total_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)