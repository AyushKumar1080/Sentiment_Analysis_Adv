# Aspect-Based Sentiment Analysis using Groq LLM & BiLSTM

An intelligent **Aspect-Based Sentiment Analysis (ABSA)** system that combines the power of a **Large Language Model (Groq Llama 3.3)** for automatic aspect extraction with a **Bidirectional LSTM (BiLSTM)** model for aspect-wise sentiment classification.

The application provides an interactive **Streamlit** interface and a **FastAPI** backend for real-time sentiment analysis of product reviews.

---

# Overview

Traditional sentiment analysis predicts the overall sentiment of an entire review.

Aspect-Based Sentiment Analysis (ABSA) identifies the sentiment associated with individual aspects mentioned in a review.

Unlike conventional ABSA applications where users manually provide the aspect, this project automatically extracts product aspects using **Groq Llama 3.3** before classifying their sentiments using a trained **BiLSTM** model.

---

# Features

- Automatic Aspect Extraction using Groq LLM
- Aspect-wise Sentiment Classification using BiLSTM
- Real-time inference using FastAPI
- Interactive Streamlit Web Application
- Confidence score for every prediction
- Handles multiple aspects from a single review
- Clean and responsive UI

---

# Example

### User Input

```text
The battery is excellent, but the display is dull and the speakers are average.
```

### Automatically Extracted Aspects

```text
Battery
Display
Speakers
```

### Output

| Aspect | Sentiment | Confidence |
|---------|-----------|------------|
| Battery | Positive | 98.7% |
| Display | Negative | 96.3% |
| Speakers | Neutral | 91.5% |

---

# Project Workflow

```text
                 Product Review
                        │
                        ▼
           Groq Llama 3.3 (LLM)
        Automatic Aspect Extraction
                        │
                        ▼
          battery, display, speakers
                        │
                        ▼
        Sentence + [SEP] + Aspect
                        │
                        ▼
      Bidirectional LSTM (TensorFlow)
                        │
                        ▼
      Aspect-wise Sentiment Prediction
                        │
                        ▼
            Streamlit Dashboard
```

---

# Tech Stack

## Programming Language

- Python

## Large Language Model

- Groq API
- Llama 3.3 70B Versatile

## Deep Learning

- TensorFlow
- Bidirectional LSTM (BiLSTM)

## Backend

- FastAPI
- Pydantic

## Frontend

- Streamlit

## Data Processing

- Pandas
- NumPy
- Scikit-learn

## Utilities

- Joblib
- python-dotenv

---

# Dataset

The model is trained on an Aspect-Based Sentiment Analysis dataset containing

- Review Sentence
- Aspect Term
- Sentiment Label

### Supported Sentiments

- Positive
- Negative
- Neutral

---

# Model Architecture

```
Input Review
      │
      ▼
Tokenizer
      │
      ▼
Embedding Layer
      │
      ▼
Bidirectional LSTM
      │
      ▼
Dense Layer
      │
      ▼
Softmax
```

The BiLSTM model receives inputs in the following format:

```
Review [SEP] Aspect
```

Example:

```
The battery is excellent but the display is poor [SEP] battery
```

---

# Running the Project

## Clone Repository

```bash
git clone https://github.com/AyushKumar1080/Sentiment_Analysis_Adv.git

cd Sentiment_Analysis_Adv
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Create a `.env` file

```text
GROQ_API_KEY=your_groq_api_key
```

## Start FastAPI

```bash
uvicorn app.main:app --reload
```

API:

```
http://127.0.0.1:8000
```

## Start Streamlit

```bash
streamlit run app/frontend.py
```


---

# Future Improvements

- Replace BiLSTM with BERT/RoBERTa
- Docker deployment
- AWS deployment
- Batch prediction
- Multi-language sentiment analysis

---

# Author

**Ayush Kumar**

B.Tech – Computer Science & Engineering

National Institute of Technology (NIT) Durgapur

GitHub: https://github.com/AyushKumar1080

LinkedIn: https://www.linkedin.com/in/ayush-kumar-421b65284/

---
