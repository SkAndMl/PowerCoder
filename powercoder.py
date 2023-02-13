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


class PowerTagger:

    def __init__(self, vocab_size=1500, model="xgb"):
        """
        Initializes the parameters of PowerTagger. The default values used, are based on the best results obtained
        during pretraining. You can play around with them depending on the questions that are being tagged.
        :param vocab_size: vocab size of the vectorizer to be loaded. Supported sizes: [1500]
        :param model: The model string to be loaded. Supported models: ["rfc", "xgb"]
        """
        self.vocab_size = vocab_size
        try:
            self.vectorizer = pickle.load(open(f"models/vectorizer_model_{self.vocab_size}.sav", "rb"))
        except Exception:
            raise ValueError(f"At present, PowerTagger does not support vocab size of {self.vocab_size}")

        try:
            self.model = pickle.load(open(f"models/{model}_model.sav", "rb"))
        except Exception:
            raise ValueError(f"At present, there are no models with the passed string: {model}")

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

    def _vectorize(self):
        """
        Uses TfIdfVectorizer to generate a vector for each input
        :return: returns a sparse array of dimensions [len(self.questions), VOCAB_SIZE]
        """
        self.X = self.vectorizer.transform(self.questions["question"])

    # ------ PREPROCESSING ENDS ------

    def predict(self, questions):
        """
        :param questions: Pandas dataframe or series containing the questions
        :return: TagPrediction object with the predicted probabilities
        """
        if not isinstance(questions, pd.DataFrame) or not isinstance(questions, pd.Series):
            if not isinstance(questions, list) or not isinstance(questions, tuple):
                questions = (questions,)
            questions = pd.DataFrame(data=questions, columns=["question"])

        if "question" not in questions.columns:
            raise ValueError("PowerTagger expects column named 'question' to contain the questions")

        self.questions = questions
        self._prepare_data()
        self._vectorize()

        return TagPrediction(self.model.predict_proba(self.X))

    def __call__(self, questions):
        return self.predict(questions)


class TagPrediction:

    def __init__(self, prediction_probs):
        self.prediction_probs = prediction_probs
        self.classes = np.argmax(prediction_probs, axis=-1)
        self.max_prob = np.max(prediction_probs, axis=-1)

    def __str__(self):
        return pd.DataFrame(data={"Predicted class": self.classes, "Probability": self.max_prob})
