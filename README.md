# PowerCoder
PowerCoder is being built as a Large Language Model (LLM) based python library  <br>

The project was started and is maintained by [Sathya Krishnan Suresh](https://www.linkedin.com/in/sathya-krishnan-suresh-914763217/). It was started as a research project and the preliminary research can be seen here - [Coding Problem Tagging](https://github.com/SkAndMl/question_tagging/)
## Dependencies
PowerCoder requires: 
* Scikit-learn
* Numpy
* Pandas
* NLTK
* Spacy

## Example
```python
from power_coder.tokenizer import TfIdfProcessor
processor = TfIdfProcessor(casefold=True, lemmatizer=True, remove_stop_words=True)
qns = ["Given a sorted integer array nums and an integer n, add/patch elements to the array such that any number in the range [1, n] inclusive can be formed by the sum of some elements in the array. Return the minimum number of patches required",
       "Given a binary array nums, return the maximum length of a contiguous subarray with an equal number of 0 and 1."]

vectorized_qns = processor.process(np.array(qns))
print(vectorized_qns.toarray().shape)
```

## Running it 
You can also tinker around PowerCoder through the following steps.
* Download the repository
* Run the following command in the terminal 
```bash
pip install -r requirements.txt
```

## Features
Currently PowerCoder is in it's nascent stage, with just TfIdfProcessor being made available.

