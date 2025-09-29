import sys,os
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.components.model_trainer import ModelTrainerPath

os.chdir("D:/PythonProject")
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

from src.utils import save_object
from src.components.data_ingestion import DataIngestor

from src.components.model_trainer import ModelTrainerPath
from src.components.model_trainer import ModelTrainer

@dataclass
class DataTransformerConfig:
    data_transformer_path= os.path.join("artifacts","data_transformer.pkl")

class DataTransformer:
    def __init__(self):
        self.DataTransformerConfig_obj = DataTransformerConfig()

    def create_transformer(self):
        try:
            num_features = ["writing_score", "reading_score"]
            cat_features = ["gender", "race_ethnicity", "parental_level_of_education",
                            "lunch", "test_preparation_course"]

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant')),
                ('ohe', OneHotEncoder())
            ])

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant')),
                ('scaler', StandardScaler())
            ])

            transformer = ColumnTransformer([
                ('train', cat_pipeline, cat_features),
                ('test', num_pipeline, num_features)
            ])

            save_object(
                self.DataTransformerConfig_obj.data_transformer_path,
                transformer
            )

            return transformer

        except Exception as e:
            raise CustomException(e, sys)

    def apply_transformer(self,train_data_path,test_data_path):
        try:
            transformer = self.create_transformer()

            target_column = "math_score"

            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            X_train = train_data.drop(columns=[target_column])
            X_test = test_data.drop(columns=[target_column])
            y_train = train_data[target_column]
            y_test = test_data[target_column]

            transformed_train_data = transformer.fit_transform(train_data)
            transformed_test_data = transformer.transform(test_data)

            train_array = np.c_[transformed_train_data, np.array(y_train)]
            test_array = np.c_[transformed_test_data, np.array(y_test)]


            return train_array, test_array


        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataIngestor_obj=DataIngestor()
    train_data_path, test_data_path = DataIngestor_obj.initiate_data_ingestion()

    DataTransformer_obj=DataTransformer()
    train_arr,test_arr= DataTransformer_obj.apply_transformer(train_data_path, test_data_path)

    modeltrainer = ModelTrainer()
    print(modeltrainer.create_model(train_arr, test_arr))