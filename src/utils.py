import os, sys
import pandas as pd
import numpy as np
import dill
from src.exception import Custom_exception

def save_obj (file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise Custom_exception(e, sys)