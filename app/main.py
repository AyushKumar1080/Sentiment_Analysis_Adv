import os
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(title="Aspect-Based Sentiment Analysis")

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

# Groq client
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None

# Load saved models
tokenizer = joblib.load(MODEL_DIR / "tokenizer.pkl")
le = joblib.load(MODEL_DIR / "label_encoder.pkl")
model = load_model(MODEL_DIR / "absa_bilstm_attention.keras")

MAX_LEN = 61


# Input received from JavaScript
class ReviewInput(BaseModel):
    review: str


# Home page
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "index.html")


# API status
@app.get("/health")
def health():
    return {"status": "ok"}


# Extract aspects using Groq
def extract_aspects(review: str):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing from your .env file."
        )

    prompt = f"""
Extract all product aspects from the review.

Return ONLY a JSON array of aspect names.

Example:
Input: The battery is amazing but the screen is dull.
Output: ["battery", "screen"]

Review:
{review}
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        # Handle JSON returned by the LLM
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        aspects = json.loads(content)

        if not isinstance(aspects, list):
            raise ValueError("Expected a JSON list of aspects.")

        # Keep only non-empty strings
        aspects = [
            aspect.strip()
            for aspect in aspects
            if isinstance(aspect, str) and aspect.strip()
        ]

        # Remove duplicate aspects
        return list(dict.fromkeys(aspects))

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Aspect extraction failed")
        raise HTTPException(
            status_code=502,
            detail="Failed to extract aspects. Check your Groq API key and connection."
        ) from exc


# Predict sentiment
@app.post("/predict")
def predict(data: ReviewInput):
    review = data.review.strip()

    if not review:
        raise HTTPException(
            status_code=400,
            detail="Please enter a product review."
        )

    aspects = extract_aspects(review)

    if not aspects:
        return {"review": review, "results": []}

    results = []

    try:
        for aspect in aspects:
            text = f"{review} [SEP] {aspect}"

            sequence = tokenizer.texts_to_sequences([text])

            padded_sequence = pad_sequences(
                sequence,
                maxlen=MAX_LEN,
                padding="post",
                truncating="post"
            )

            prediction = model.predict(padded_sequence, verbose=0)

            index = int(np.argmax(prediction[0]))
            sentiment = str(le.inverse_transform([index])[0])
            confidence = float(np.max(prediction[0]))

            results.append({
                "aspect": aspect,
                "sentiment": sentiment,
                "confidence": round(confidence, 4)
            })

        return {
            "review": review,
            "results": results
        }

    except Exception as exc:
        logger.exception("Sentiment prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to predict sentiment."
        ) from exc