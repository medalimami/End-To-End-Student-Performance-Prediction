from dataclasses import dataclass
import sys,os
from sklearn.linear_model import LinearRegression


from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models,save_object
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerPath:
    model_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.modelTrainerPath_object = ModelTrainerPath()

    def create_model(self,train_array, test_array):
        try:
          X_test,y_test=(test_array[:, :-1],test_array[:,-1])
          logging.info("Creating model...")

          models = {
              "RandomForest":RandomForestRegressor(),
              "DecisionTree":DecisionTreeRegressor(),
              "GradientBoostingRegressor":GradientBoostingRegressor(),
              "KNeighborsRegressor":KNeighborsRegressor(),
              "LinearRegression":LinearRegression(),
              "XGBRegressor":XGBRegressor(verbosity=0),
              "CatBoostRegressor":CatBoostRegressor(verbose=False),
              "AdaBoostRegressor":AdaBoostRegressor()
              }

          model_report = evaluate_models(train_array,test_array,models)

          best_model_name = max(model_report.items(), key=lambda x: x[1])[0]
          best_model_score = max(model_report.items(), key=lambda x: x[1])[1]

          print(f"best model is {best_model_name} with an r2 score of {best_model_score}")

          best_model = models[best_model_name]

          save_object(
              file_path=self.modelTrainerPath_object.model_path,
              obj=best_model
          )

          return best_model_score


        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

