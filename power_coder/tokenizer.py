import nltk
import numpy as np
import pandas as pd
from typing import Union
import pickle
import sys
sys.path.insert(0, "POWERCODER")

class TfIdfProcessor:

    STOPWORDS = nltk.corpus.stopwords.words("english")
    WN_LEMMATIZER = nltk.stem.WordNetLemmatizer()

    def __init__(self,
                 casefold: bool=True,
                 remove_stop_words: bool=True,
                 lemmatize: bool=True) -> None:
        
        self.casefold = casefold
        self.remove_stop_words = remove_stop_words
        self.lemmatize = lemmatize
        self.vectorizer = pickle.load(open("tokenizer/tfidf.sav", "rb"))
    
    def __repr__(self) -> str:
        return f"TfIdfProcesser(casefold={self.casefold}" + \
               f",remove_stop_words={self.remove_stop_words}," + \
               f"lemmatize={self.lemmatize})"
    
    def __str__(self) -> str:
        return f"TfIdfProcesser(casefold={self.casefold}" + \
               f",remove_stop_words={self.remove_stop_words}," + \
               f"lemmatize={self.lemmatize})"

    def process(self,
                questions: Union[np.ndarray, pd.Series, pd.DataFrame],
                col_name: str=None) -> np.ndarray:
        
        if not (isinstance(questions, np.ndarray), isinstance(questions, pd.Series), pd.DataFrame):
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
        
        if self.casefold:
            self.df["questions"] = self.df["questions"].apply(lambda qn: self._casefold(qn))
        if self.remove_stop_words:
            self.df["questions"] = self.df["questions"].apply(lambda qn: self._remove_stopwords(qn))
        if self.lemmatize:
            self.df["questions"] = self.df["questions"].apply(lambda qn: self._lemmatize(qn))
        
        vector_qns = self.vectorizer.transform(self.df["questions"])
        return vector_qns


    def _remove_stopwords(self,
                          text: str) -> str:
        new_text = [word for word in text.split(' ') if word not in self.STOPWORDS]
        return ' '.join(new_text)
    
    def _lemmatize(self,
                   text: str) -> str:
        new_text = [self.WN_LEMMATIZER.lemmatize(word) for word in text.split(' ')]
        return ' '.join(new_text)
    
    def _casefold(self,
                  text: str) -> str:
        return text.lower()
    

if __name__ == "__main__":
    tf = TfIdfProcessor()
    vect = tf.process(np.array(["Solve this", "Solve that given this",  
                            "Given an array of size n, find sum"]))
    print(vect.toarray().shape)