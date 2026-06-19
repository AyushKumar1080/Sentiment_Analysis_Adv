from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 61

app = FastAPI()

tokenizer = joblib.load(r"models/tokenizer.pkl")
le = joblib.load(r"models/label_encoder.pkl")
model = load_model(r"models/absa_bilstm_attention.keras")

class ReviewInput(BaseModel):
    sentence: str
    aspect: str


@app.get("/")
def home():
    return {"message": "ABSA API Running"}


@app.post("/predict")
def predict(data: ReviewInput):

    text = f"{data.sentence} [SEP] {data.aspect}"

    seq = tokenizer.texts_to_sequences([text])

    pad = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    pred = model.predict(pad, verbose=0)

    idx = np.argmax(pred)

    sentiment = le.inverse_transform([idx])[0]

    confidence = float(np.max(pred))

    return {
        "sentiment": sentiment,
        "confidence": round(confidence,4)
    }