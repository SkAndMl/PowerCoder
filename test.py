from powercoder import PowerTagger
import pandas as pd

pt = PowerTagger(vocab_size=800, model='rfc', n_estimator=400)

qns = ["Given a sorted integer array nums and an integer n, add/patch elements to the array such that any number in the range [1, n] inclusive can be formed by the sum of some elements in the array. Return the minimum number of patches required",
       "Given a binary array nums, return the maximum length of a contiguous subarray with an equal number of 0 and 1."]

tp = pt.predict(qns)
print(tp.results)
