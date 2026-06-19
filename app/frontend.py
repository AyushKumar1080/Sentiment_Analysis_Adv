import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("Aspect Based Sentiment Analysis")

sentence = st.text_area(
    "Enter Review"
)

aspect = st.text_input(
    "Enter Aspect"
)

if st.button("Analyze"):

    payload = {
        "sentence": sentence,
        "aspect": aspect
    }

    response = requests.post(
        API_URL,
        json=payload
    )

    result = response.json()

    st.success(
        f"Sentiment: {result['sentiment']}"
    )

    st.write(
        f"Confidence: {result['confidence']:.4f}"
    )