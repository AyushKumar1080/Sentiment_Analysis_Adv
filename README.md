# Aspect-Based Sentiment Analysis (ABSA)

An Aspect-Based Sentiment Analysis (ABSA) application built using **Bidirectional LSTM (BiLSTM)** and **TensorFlow**. The project predicts sentiment for a user-specified aspect within a review and provides real-time inference through a **FastAPI** backend and an interactive **Streamlit** web application.

---

## Overview

Traditional sentiment analysis predicts the overall sentiment of a review. In contrast, **Aspect-Based Sentiment Analysis (ABSA)** identifies the sentiment associated with a specific aspect mentioned in the text.

### Example

**Review**

> The laptop performance is excellent, but the battery life is disappointing.

| Aspect | Predicted Sentiment |
|---------|---------------------|
| Performance | Positive |
| Battery Life | Negative |

This enables fine-grained sentiment analysis, making the model suitable for product review analysis, customer feedback mining, and opinion analysis.

---

## Features

- Aspect-Based Sentiment Analysis
- Deep Learning using **Bidirectional LSTM (BiLSTM)**
- FastAPI REST API for model inference
- Interactive Streamlit web interface
- Real-time sentiment prediction
- Confidence score for each prediction
- Clean and responsive user interface

---

## Tech Stack

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Bidirectional LSTM (BiLSTM)

### Data Processing

- Pandas
- NumPy
- Scikit-learn

### Backend

- FastAPI
- Pydantic

### Frontend

- Streamlit

### Utilities

- Joblib

---

## Project Architecture

```
                User Input
                     │
                     ▼
             Streamlit Frontend
                     │
                     ▼
               FastAPI Backend
                     │
                     ▼
      Text Preprocessing Pipeline
                     │
                     ▼
      Bidirectional LSTM Model
                     │
                     ▼
      Sentiment Prediction
                     │
                     ▼
     Confidence Score + Result
```

---

## Dataset

The model is trained on a labeled Aspect-Based Sentiment Analysis dataset containing:

- Review Text
- Aspect
- Sentiment Label

### Supported Sentiment Classes

- Positive
- Negative
- Neutral

---

## Example Prediction

### Input

```text
Review:
The laptop performance is amazing but the battery life is poor.

Aspect:
battery life
```

### Output

```text
Predicted Sentiment: Negative
Confidence Score: 94%
```

---

## Future Improvements

- Integrate Transformer-based models (BERT, RoBERTa)
- Support multiple aspect prediction
- Explainable AI (Attention Visualization / SHAP)
- Docker containerization
- Cloud deployment (AWS, Azure, or GCP)
- Batch prediction API

---

## Author

**Ayush Kumar**

B.Tech, Computer Science & Engineering

National Institute of Technology (NIT) Durgapur

GitHub: https://github.com/AyushKumar1080

LinkedIn: https://www.linkedin.com/in/ayush-kumar-421b65284/