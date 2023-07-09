from power_coder.tagger import DSTagger
# from power_coder.tokenizer import TfIdfTokenizer
import numpy as np

tagger = DSTagger()
tagged = tagger.tag(np.array(["Solve this", "Solve that given this",
                            "Given an array of size n, find sum"]))
print(tagged)