import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import streamlit as st
import re
import fitz  # reads the pdf

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#from sklearn.multiclass import OneVsRestClassifier

df = pd.read_csv("UpdatedResumeDataSet.csv")


def cleanResume(txt):
    txt = re.sub(r'http\S+|www\.\S+', ' ', txt)
    txt = re.sub(r'\S+@gmail\.com', ' ', txt)
    txt = re.sub(r'\S+@\S+', ' ', txt)
    txt = re.sub(r'\+?\d[\d -]{8,}\d', ' ', txt)
    txt = re.sub(r'<.*?>', ' ', txt)
    txt = re.sub(r'\bRT\b|\bCC\b', ' ', txt)
    txt = re.sub(r'[^A-Za-z\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt


df["Cleaned_Resume"] = df["Resume"].apply(lambda x: cleanResume(x))


le = LabelEncoder()
df["Category_Encoded"] = le.fit_transform(df["Category"])


tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df["Cleaned_Resume"])
y = df["Category_Encoded"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#clf = KNeighborsClassifier()
#clf= OneVsRestClassifier(KNeighborsClassifier())
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)

clf.fit(X_train, y_train)


st.title("Resume Analyzer: Find Out What Employers See!")


uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
input_text = st.text_area("Or paste your Resume text here:")

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

if st.button("Analyze"):
    if uploaded_file is not None:
      
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ PDF uploaded and text extracted!")
    elif input_text.strip() != "":
        resume_text = input_text
        st.success("✅ Text received!")
    else:
        st.error("⚠️ Please upload a PDF or paste your resume text.")
        st.stop()

 
    cleaned_resume = cleanResume(resume_text)

    vectorized_resume = tfidf.transform([cleaned_resume])

    prediction = clf.predict(vectorized_resume)
    predicted_category = le.inverse_transform(prediction)[0]

    st.subheader("🎯 Predicted Job Category:")
    st.success(f"**{predicted_category}**")

