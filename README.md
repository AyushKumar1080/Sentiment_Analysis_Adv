# Aspect-Based Sentiment Analysis (ABSA)

## Overview

This project implements an **Aspect-Based Sentiment Analysis (ABSA)** system using a **BiLSTM with Attention** deep learning model. Unlike traditional sentiment analysis, which predicts sentiment for an entire review, ABSA determines sentiment towards a specific aspect mentioned in the review.

For example:

**Review:** "The laptop performance is excellent but the battery life is disappointing."

| Aspect       | Sentiment |
| ------------ | --------- |
| Performance  | Positive  |
| Battery Life | Negative  |

The model is deployed through a **FastAPI REST API** and accessed via a **Streamlit web application**.

---

## Features

* Aspect-Based Sentiment Classification
* Deep Learning using BiLSTM + Attention
* FastAPI Backend
* Streamlit Frontend
* Confidence Score Prediction
* Easy-to-use REST API

---

## Tech Stack

### Machine Learning & Deep Learning

* TensorFlow
* NumPy
* Pandas
* Scikit-learn

### Backend

* FastAPI
* Pydantic

### Frontend

* Streamlit

### Utilities

* Joblib

---

## Dataset

The model is trained on a labeled Aspect-Based Sentiment Analysis dataset containing:

* Review Text
* Aspect
* Sentiment Label

Supported sentiment classes:

* Positive
* Negative
* Neutral

---

## Example

Input:

```text
Review:
The laptop performance is amazing but battery life is poor.

Aspect:
battery
```

Output:

```text
Sentiment: Negative
Confidence: 0.94
```

---

## Future Improvements

* Transformer-based Models (BERT, RoBERTa)
* Multi-Aspect Extraction
* Explainable AI Visualizations
* Docker Deployment
* Cloud Deployment (AWS/Azure/GCP)

---

## Author

Ayush Kumar

B.Tech, Computer Science & Engineering

National Institute of Technology (NIT) Durgapur
