import os,sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

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


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(train_array, test_array, models, params):
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
            print(f"\n🔍 Training {name}...")

            # Get hyperparameter grid if available (otherwise empty dict)
            param = params.get(name, {})

            if param:  # If there are parameters to search
                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param,
                    n_iter=5,  # number of random combinations to try
                    scoring='r2',
                    cv=3,  # 3-fold cross validation
                    verbose=1,
                    random_state=42,
                    n_jobs=-1  # use all CPU cores
                )
                random_search.fit(X_train, y_train)
                best_model = random_search.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_pred = best_model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            report[name] = test_r2

            print(f"{name} R2 score: {test_r2:.4f}")

        return report

    except Exception as e:
        raise CustomException(e,sys)



