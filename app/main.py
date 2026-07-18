from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import numpy as np
import json
import os

from groq import Groq
from dotenv import load_dotenv

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MAX_LEN = 61

app = FastAPI()

tokenizer = joblib.load("models/tokenizer.pkl")
le = joblib.load("models/label_encoder.pkl")
model = load_model("models/absa_bilstm_attention.keras")

class ReviewInput(BaseModel):
    sentence: str

@app.get("/")
def home():
    return {"message": "ABSA API Running"}

def extract_aspects(review: str):

    prompt = f"""
Extract all product aspects from the review.

Return ONLY a JSON array.

Example:

Input:
The battery is amazing but the screen is dull.

Output:
["battery","screen"]

Review:
{review}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user","content": prompt}],
        temperature=0
    )
    content = response.choices[0].message.content.strip()
    try:
        aspects = json.loads(content)

    except Exception:
        aspects = [
            x.strip().replace('"', '')
            for x in content.replace("[", "")
                            .replace("]", "")
                            .split(",")
        ]

    return aspects

@app.post("/predict")
def predict(data: ReviewInput):

    review = data.sentence

    aspects = extract_aspects(review)

    results = []

    for aspect in aspects:

        text = f"{review} [SEP] {aspect}"

        seq = tokenizer.texts_to_sequences([text])

        pad = pad_sequences(seq,maxlen=MAX_LEN,padding="post",truncating="post")

        pred = model.predict(pad, verbose=0)

        idx = np.argmax(pred)

        sentiment = le.inverse_transform([idx])[0]

        confidence = float(np.max(pred))

        results.append({"aspect": aspect,"sentiment": sentiment,"confidence": round(confidence, 4)})

    return {"review": review,"results": results}