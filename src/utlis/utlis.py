from src.exceptions.expection import CustomException
from src.logger.custom_logging import logger
import sys,os
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb')as f:
            pickle.dump(obj,f)

    except Exception as e:
        logger.info(f'Error ocuured in {e}')
        raise CustomException(e,sys)
    
def model_evaluate(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for model_name, model in models.items():
            param = params.get(model_name, {})

            # Perform Randomized Search for hyperparameter tuning
            gs = RandomizedSearchCV(model, param, cv=5, n_iter=10, scoring='accuracy', verbose=1, random_state=42)
            gs.fit(X_train, y_train)

            # Set the best parameters and fit the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            test_model_accuracy = accuracy_score(y_test, y_pred)

            # Store the result in the report dictionary
            report[model_name] = test_model_accuracy

        return report

    except Exception as e:
        raise CustomException(e, sys)   
    
def load_obj(file_path):
    try:
        with open(file_path,'rb')as f:
            return pickle.load(f)
    except Exception as e:
        logger.info(f'Error ocuured in {e}')
        raise CustomException(e,sys)