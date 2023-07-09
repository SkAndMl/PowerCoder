import pandas as pd
import numpy as np
from typing import Union
import pickle
import os
import sys
from .tokenizer import TfIdfTokenizer


class DSTagger:

    CLASSES = {0: "graph", 1: "array", 2: "string"}

    def __init__(self,
                 model: str='stacked') -> None:
        
        self.model_name = model
        
        if model+".sav" not in os.listdir("models/"):
            raise ValueError(f"The passed model name: {self.model_name} is not available in the list of models")

        self.model = pickle.load(open(f"models/{self.model_name}.sav", "rb"))
        self.tokenizer = TfIdfTokenizer()
    
    def __repr__(self) -> str:
        return f"DSTagger(model={self.model_name})"
    
    def __str__(self) -> str:
        return f"DSTagger(model={self.model_name})"
    
    def tag(self,
            questions: Union[np.ndarray, pd.Series, pd.DataFrame],
            col_name: str=None) -> np.ndarray:
        
        if not (isinstance(questions, np.ndarray) or isinstance(questions, pd.Series) or isinstance(questions, pd.DataFrame)):
            raise TypeError(f"'questions' must be of type numpy.ndarray or pandas.Series or pandas.DataFrame. But it is of type {type(questions)}")

        if isinstance(questions, np.ndarray):
            if len(questions.shape) > 1:
                raise ValueError(f"Expected one dimensional numpy array, but got array of shape: {questions.shape}")
            else:
                self.df = pd.DataFrame(data=questions, columns=["questions"])
        elif isinstance(questions, pd.Series):
            self.df = pd.DataFrame(questions, columns=["questions"])
        else:
            if col_name is None:
                raise ValueError(f"col_name must consist of the name of the column in the dataframe cosisting the questions, should not be None")
            else:
                self.df = pd.DataFrame(questions[col_name], columns=["questions"])
        
        self._features = self.tokenizer.process(questions)
        self._make_predictions()
        return self.pred_proba


    def int_to_label(self,
                     class_idx: int) -> str:
        return self.CLASSES[class_idx]

    def _make_predictions(self) -> None:
        self.pred_proba = self.model.predict_proba(self._features)
        for (k, v) in self.CLASSES.items():
            self.df[v] = self.pred_proba[:, k]
        pred_classes = np.argmax(self.pred_proba, axis=-1)
        self.df["ds_tag"] = pred_classes
        self.df["ds_tag"] = self.df["ds_tag"].apply(lambda class_idx: self.int_to_label(class_idx))


if __name__ == "__main__":
    tagger = DSTagger()
    tagged = tagger.tag(np.array(["Solve this", "Solve that given this",
                                "Given an array of size n, find sum"]))
    print(tagged)