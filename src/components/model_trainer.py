import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from dataclasses import dataclass
# Modelling
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.exception import Custom_exception
from src.logger import logging
from src.utils import save_obj, evaluate_models

@dataclass()
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and test input data")
            X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1]
        )
            models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "KNN": KNeighborsRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(),
            "CatBoost": CatBoostRegressor(verbose=0),
        }
            params={
            "Decision Tree": {
                'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            },
            "Random Forest":{
                'n_estimators': [8,16,32,64,128,256]
            },
            "Gradient Boosting":{
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Linear Regression":{},
            "XGBoost":{  # Fixed key to match models dictionary
                'learning_rate':[.1,.01,.05,.001],
                'n_estimators': [8,16,32,64,128,256]
            },
            "CatBoost":{  # Fixed key to match models dictionary
                'depth': [6,8,10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            },
            "AdaBoost":{  # Fixed key to match models dictionary
                'learning_rate':[.1,.01,0.5,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
        }
        
            model_report = evaluate_models(X_train=X_train, y_train=y_train, 
                                     X_test=X_test, y_test=y_test, 
                                     models=models, param=params)
        
        # Get best model info
            best_model_name = max(model_report.items(), 
                            key=lambda x: x[1]['test_score'])[0]
            best_model_info = model_report[best_model_name]
            best_model_score = best_model_info['test_score']
        
        # Get the model instance and set best parameters if they exist
            best_model = models[best_model_name]
            if best_model_info['best_params']:
                best_model.set_params(**best_model_info['best_params'])
        
        # Refit the model with best parameters
            best_model.fit(X_train, y_train)
        
            if best_model_score < 0.6:
                raise Custom_exception("Best model score is less than 0.6", sys)
            logging.info(f"Best model: {best_model_name} with score: {best_model_score:.4f}")
        
            save_obj(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )
        
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
        
            return r2
        
        except Exception as e:
            raise Custom_exception(e, sys)
        
        
