import streamlit as st
from transformers import pipeline
import pandas as pd
import os
import matplotlib.pyplot as plt


sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


# Function to convert Hugging Face score (0-1) to 0-10 scale
def scale_score(score, label):
    if label == "NEGATIVE":
        return round((1 - score) * 10, 2)
    else:  # POSITIVE
        return round(score * 10, 2)

# CSV file to store results
CSV_FILE = "sentiment_results.csv"

# Ensure CSV file exists
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["sentence", "label", "score", "scaled_score"]).to_csv(CSV_FILE, index=False)

# Streamlit UI
st.title("Sentiment Score Analyzer")
sentence = st.text_area("Enter a sentence:")

if st.button("Analyze Sentiment"):
    if sentence.strip():
        result = sentiment_pipeline(sentence)[0]
        label = result["label"]
        raw_score = result["score"]
        scaled = scale_score(raw_score, label)

        st.markdown(f"**Sentiment:** {label}")
        st.markdown(f"**Confidence Score:** {round(raw_score, 3)}")
        st.markdown(f"**Scaled Score (out of 10):** {scaled}/10")

        # Save to CSV
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, pd.DataFrame([{
            "sentence": sentence,
            "label": label,
            "score": round(raw_score, 3),
            "scaled_score": scaled
        }])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)

        st.success("Result saved to `sentiment_results.csv`!")
    else:
        st.warning("Please enter a sentence.")
