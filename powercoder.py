import pandas as pd
import string
import nltk
import pickle
import numpy as np


# nltk.download("stopwords")
# nltk.download('omw-1.4')
# nltk.download('wordnet')

PUNCTUATION = string.punctuation
idx_open = PUNCTUATION.index("[")
PUNCTUATION = PUNCTUATION[:idx_open] + PUNCTUATION[idx_open + 1:]
idx_closed = PUNCTUATION.index("]")
PUNCTUATION = PUNCTUATION[:idx_closed] + PUNCTUATION[idx_closed + 1:]
STOPWORDS = nltk.corpus.stopwords.words("english")
LEMMATIZER = nltk.stem.WordNetLemmatizer()
DATASTRUCTURE_CLASSES = {"graph": 0, "array": 1, "string": 2}


class PowerCoder:

    def __init__(self):
        """
        present for namesake for now
        """
        self.vectorizer = pickle.load(open("models/vectorizer_model.sav", "rb"))
        self.model = pickle.load(open("models/rfc_model.sav", "rb"))

    #    ------  DATA PREPARATION STARTS    ------

    # noinspection PyMethodMayBeStatic
    def _remove_punctuation(self, text):
        """
           This function takes in a string and removes the punctuations except '[' and ']' as they might represent
           the array indexing and return a string
           :param text: question string
           :return: punctuation removed string
            """
        new_text = "".join([t for t in text if t not in PUNCTUATION])
        return new_text

    # noinspection PyMethodMayBeStatic
    def _remove_stopwords(self, text):
        """
            This function takes in a string and returns a stop word removed string
           :param text: question string
           :return: stop word removed string
           """
        new_text = [word for word in text.split(' ') if word not in STOPWORDS]
        return " ".join(new_text)

    # noinspection PyMethodMayBeStatic
    def _lemmatization(self, text):
        new_text = [LEMMATIZER.lemmatize(word) for word in text.split(" ")]
        return ' '.join(new_text)

    def _prepare_data(self):
        self.questions["question"] = self.questions["question"].apply(lambda qn: self._remove_punctuation(qn))
        self.questions["question"] = self.questions["question"].str.lower()
        self.questions["question"] = self.questions["question"].apply(lambda qn: self._remove_stopwords(qn))
        self.questions["question"] = self.questions["question"].apply(lambda qn: self._lemmatization(qn))

    # ------ DATA PREPARATION ENDS ------

    # ------ PREPROCESSING STARTS ------

    def _vectorize(self):
        self.X = self.vectorizer.transform(self.questions["question"])

    # ------ PREPROCESSING ENDS ------

    def predict(self, questions):

        if not isinstance(questions, pd.DataFrame) or not isinstance(questions, pd.Series):
            if not isinstance(questions, list) or not isinstance(questions, tuple):
                questions = (questions,)
            questions = pd.DataFrame(data=questions, columns=["question"])

        self.questions = questions
        self._prepare_data()
        self._vectorize()

        return self.model.predict_proba(self.X)

    def __call__(self, questions):
        return self.predict(questions)



