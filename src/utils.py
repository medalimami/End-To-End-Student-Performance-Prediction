import os,sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)

import os,sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(train_array, test_array, models):
    """
    Train each model on (X_train, y_train), evaluate on (X_test, y_test)
    and return a dict: {model_name: test_r2}.
    """
    try:
        report = {}

        X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1]
        )

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            report[name] = test_r2

        return report

    except Exception as e:
        raise CustomException(e,sys)



