import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Aspect-Based Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Aspect-Based Sentiment Analysis")
st.write("Enter a product review below. The system will automatically identify aspects and predict the sentiment for each one.")

sentence = st.text_area(
    "Enter Review",
    height=150
)

if st.button("Analyze"):

    if not sentence.strip():
        st.warning("Please enter a review.")
        st.stop()

    payload = {
        "sentence": sentence
    }

    try:
        response = requests.post(
            API_URL,
            json=payload
        )

        response.raise_for_status()

        result = response.json()

        st.subheader("Results")

        df = pd.DataFrame(result["results"])

        st.dataframe(
            df,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error: {e}")