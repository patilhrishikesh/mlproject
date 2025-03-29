import os, sys
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from src.exception import Custom_exception

def save_obj (file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise Custom_exception(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():  # Proper way to iterate
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score
            
            # Optional: Print or log scores for debugging
            print(f"{model_name}: Train Score: {train_model_score:.4f}, Test Score: {test_model_score:.4f}")
            
        return report
       
            
    except Exception as e:
        raise Custom_exception(e, sys)