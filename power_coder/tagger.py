import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import pickle

class DataStructureTagger:

    def __init__(self, model="rfc"):

        self.model = pickle.load(file="https://github.com/SkAndMl/question_tagging/raw/main/models/gb_max_depth_26_n_estimators_1019_lr_0.82.sav")
    
    # def predict(self, ):