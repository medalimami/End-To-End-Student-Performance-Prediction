import sys,os
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
os.chdir("D:/PythonProject")
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

from src.utils import save_object
from src.components.data_ingestion import DataIngestor


#class DataTransformerPath:
#    def __init__(self):
#        self.preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

@dataclass(frozen=True,order=True)
class DataTransformerPath:
       preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformer:
    '''
        This function creates the data transformer object
    '''
    def __init__(self):
        self.datatransformer_path_obj = DataTransformerPath()

    def create_transformer(self):
        try:
            num_features=["writing_score","reading_score"]
            cat_features=["gender","race_ethnicity","parental_level_of_education",
                          "lunch","test_preparation_course"]
            num_transform=Pipeline(steps=[
                ('impute',SimpleImputer(strategy="mean")),
                ('sc',StandardScaler()),
            ])

            cat_transform=Pipeline(steps=[
                ('impute', SimpleImputer(strategy="most_frequent")),
                ('ohe', OneHotEncoder())
            ])

            data_transformer = ColumnTransformer([
                ("num_transform",num_transform,num_features),
                ("cat_transform",cat_transform,cat_features)
            ])

            logging.info("Data transformer object created")

            return data_transformer

        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)


    def apply_transformer(self, train_data_path, test_data_path):

        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info("train and test data loaded")

            preprocessing_obj = self.create_transformer()


            logging.info("preprocessing object loaded")

            target_column ="math_score"

            X_train = train_df.drop(columns=[target_column],axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column],axis=1)
            y_test = test_df[target_column]

            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            logging.info("saved preprocessing object")

            save_object(
                file_path=self.datatransformer_path_obj.preprocessor_path,
                obj=preprocessing_obj
            )

            return(train_arr, test_arr,self.datatransformer_path_obj.preprocessor_path)

        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

if __name__ == "__main__":
    DataIngestor_obj=DataIngestor()
    train_data_path, test_data_path = DataIngestor_obj.initiate_data_ingestion()
    DataTransformer_obj=DataTransformer()
    DataTransformer_obj.apply_transformer(train_data_path, test_data_path)