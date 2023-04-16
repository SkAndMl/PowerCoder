import pandas as pd
import string
import nltk
import pickle
import numpy as np


# nltk.download("stopwords")
# nltk.download('omw-1.4')
# nltk.download('wordnet')

IDX_TO_CLASS = {0: "graph", 1: "array", 2: "string"}
PUNCTUATION = string.punctuation
idx_open = PUNCTUATION.index("[")
PUNCTUATION = PUNCTUATION[:idx_open] + PUNCTUATION[idx_open + 1:]
idx_closed = PUNCTUATION.index("]")
PUNCTUATION = PUNCTUATION[:idx_closed] + PUNCTUATION[idx_closed + 1:]
STOPWORDS = nltk.corpus.stopwords.words("english")
LEMMATIZER = nltk.stem.WordNetLemmatizer()
DATASTRUCTURE_CLASSES = {"graph": 0, "array": 1, "string": 2}


class PowerTagger:

    def __init__(self, vocab_size=1500, model="xgb", n_estimator=200):
        """
        Initializes the parameters of PowerTagger. The default values used, are based on the best results obtained
        during pretraining. You can play around with them depending on the questions that are being tagged.
        :param vocab_size: vocab size of the vectorizer to be loaded. Supported sizes: [1500]
        :param model: The model string to be loaded. Supported models: ["rfc", "xgb"]
        """
        self.vocab_size = vocab_size
        try:
            self.model_pipeline = pickle.load(open(f"/Users/sathyakrishnansuresh/Desktop/PowerCoder-main/models/{model}_{n_estimator}_vect_{vocab_size}.sav", "rb"))
        except Exception:
            raise ValueError(f"At present, PowerTagger does not model pipeline with {n_estimator} estimators and "
                             f"{vocab_size} max features")

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
        new_text = [word for word in text.split() if word not in STOPWORDS]
        return " ".join(new_text)

    # noinspection PyMethodMayBeStatic
    def _lemmatization(self, text):
        """
        Uses WordnetLemmatizer to lemmatize the given question
        :param text: un-lemmatized input question
        :return: lemmatized text
        """
        new_text = [LEMMATIZER.lemmatize(word) for word in text.split(" ")]
        return ' '.join(new_text)

    def _prepare_data(self):
        self.questions["question"] = self.questions["question"].apply(lambda qn: self._remove_punctuation(qn))
        self.questions["question"] = self.questions["question"].str.lower()
        self.questions["question"] = self.questions["question"].apply(lambda qn: self._remove_stopwords(qn))
        self.questions["question"] = self.questions["question"].apply(lambda qn: self._lemmatization(qn))

    # ------ DATA PREPARATION ENDS ------

    # ------ PREPROCESSING STARTS ------

    # ------ PREPROCESSING ENDS ------

    def predict(self, question: str):
        """
        :param questions: Pandas dataframe or series containing the questions
        :return: TagPrediction object with the predicted probabilities
        """
    
        self.questions = pd.DataFrame(data=[question], columns=["question"])
        # print("From pc,", self.questions)
        self._prepare_data()

        return TagPrediction(self.model_pipeline.predict_proba(self.questions["question"]))

    def __call__(self, questions):
        return self.predict(questions)


# 0 - graph
# 1 - array
# 2 - string

class TagPrediction:

    def __init__(self, prediction_probs):
        self.class_probs = {IDX_TO_CLASS[i] : [prediction_probs[0][i]] \
                            for i in range(len(prediction_probs[0]))}
        
    
    
    
        
        
