# PowerCoder
PowerCoder is a python module built on top of [huggingface](https://huggingface.co/) and [scikit-learn](https://scikit-learn.org/stable/index.html) and is 
distributed under the MIT License <br>

The project was started and is maintained by Sathya Krishnan Suresh. It was started as a research project.

## Dependiencies
PowerCoder requires: 
* Scikit-learn (>=1.2.0)
* HuggingFace transformers
* HuggingFace datasets

## Features
Currently PowerCoder is in it's nascent stage, with just PowerTagger being deployed.
### PowerTagger
The aim of PowerTagger is to tag any coding question based on the data structure used to solve the question and the algorithms that can be used to solve that question.<br>

At present, it can only tag data structure ("array", "graph" and "string") questions, but constant work is going on to add more such structure. <br>

PowerTagger uses a RandomForestClassifier model for now but very soon a Distil-Bert model will be added to PowerTagger and then more transformer based models will be added.
Based on the computation budget, the user can go for any kind of available model. <br>

PowerTagger uses a TfIdfVectorizer for now, with a vocab size of 1500. TfIdfVectorizer with different vocabulary sizes, and other vectorization models will be added soon.

### PowerGenerator
The aim of PowerGenerator is to generate questions based on prompts, data structures and algorithms.<br>Deployment of PowerGenerator will be around June, 2023
