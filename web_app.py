import streamlit as st
import pandas as pd
import numpy as np
from powercoder import PowerTagger, TagPrediction

st.title("PowerCoder")
st.write("One stop solution for all coding problems")

classifier_name = st.sidebar.selectbox("model", options=["Random Forest", "XGBoost"])
n_estimators = st.sidebar.slider("n_estimators", min_value=100, max_value=400, step=50)
vocab_size = st.sidebar.slider("vocab_size", min_value=200, max_value=800, step=100)

if classifier_name == "Random Forest":
    classifier_name = "rfc"
else:
    classifier_name = "xgb"


question = st.text_input(label="question",
                         key="question",
                         value="null")

tag_btn = st.button("Tag it")

if tag_btn:
    pt = PowerTagger(vocab_size=vocab_size,
                     model=classifier_name,
                     n_estimator=n_estimators)
    # pt = PowerTagger(vocab_size=200, model='rfc', n_estimator=400)
    tp = pt.predict(question)
    st.dataframe(pd.DataFrame(data=tp.class_probs))