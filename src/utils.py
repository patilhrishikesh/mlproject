import os, sys
import pandas as pd
import numpy as np
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import Custom_exception

from src.logger import logging

def save_obj (file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise Custom_exception(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for model_name, model in models.items():
            current_params = param.get(model_name, {})
            
            if current_params:
                gs = GridSearchCV(model, current_params, cv=3, scoring='r2', n_jobs=-1)
                gs.fit(X_train, y_train)
                
                test_score = r2_score(y_test, gs.best_estimator_.predict(X_test))
                report[model_name] = {
                    'test_score': test_score,
                    'best_params': gs.best_params_,
                    'best_estimator': gs.best_estimator_  # Store the best fitted model
                }
            else:
                model.fit(X_train, y_train)
                test_score = r2_score(y_test, model.predict(X_test))
                report[model_name] = {
                    'test_score': test_score,
                    'best_params': None,
                    'best_estimator': model  # Store the fitted model
                }
                
            logging.info(f"{model_name} - Test R2: {test_score:.4f}")
            
        return report
          
    except Exception as e:
        raise Custom_exception(e, sys)