# Aspect-Based Sentiment Analysis (ABSA)

An AI-powered web application that analyzes a product review, identifies the product aspects mentioned in it, and predicts the sentiment associated with each aspect. Results include the aspect name, predicted sentiment, and model confidence.

The application combines a Groq-hosted language model for aspect extraction with a trained BiLSTM + Attention neural network for aspect-level sentiment classification. A FastAPI backend serves the prediction API and the HTML/CSS/JavaScript interface.

## Features

- Accepts natural-language product reviews.
- Extracts product aspects from a review using the Groq API.
- Predicts sentiment separately for each extracted aspect.
- Displays the aspect, sentiment label, and confidence score.
- Uses a trained TensorFlow/Keras BiLSTM + Attention model.
- Uses a saved tokenizer and label encoder for inference.
- Provides a responsive frontend with an animated gradient background and results table.
- Exposes a health-check endpoint for the API.

## Example

**Input review**

> The battery is excellent, but the display is dull.

The application may identify aspects such as `battery` and `display`, then return a sentiment prediction and confidence score for each aspect.

Predictions depend on the trained model and the review provided; this example illustrates the workflow, not guaranteed model output.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10.11 | Runtime specified in `runtime.txt` |
| FastAPI | Backend API and frontend hosting |
| Uvicorn | ASGI server |
| TensorFlow / Keras | Loads and runs the trained sentiment model |
| BiLSTM + Attention | Aspect-level sentiment classification model |
| Groq API | Extracts aspects from a product review |
| `openai/gpt-oss-20b` | Model configured for aspect extraction |
| NumPy | Prediction processing |
| Joblib | Loads the tokenizer and label encoder |
| Pydantic | Validates API input |
| HTML, CSS, JavaScript | Web interface |
| Streamlit, Pandas, Requests | Also present in the repository's requirements and `frontend.py` |

## How It Works

1. **Enter a review:** The user submits a product review through the web interface.
2. **Extract aspects:** The FastAPI backend sends the review to the configured Groq model, which is prompted to return a JSON array of aspect names.
3. **Prepare each aspect:** For every extracted aspect, the backend combines the review and aspect in the format `review [SEP] aspect`.
4. **Tokenize and pad:** The saved tokenizer converts the text into sequences. Sequences are padded or truncated to a maximum length of `61`.
5. **Predict sentiment:** The trained Keras model produces class scores for each review-aspect pair.
6. **Decode the prediction:** The saved label encoder maps the highest-scoring class to a sentiment label. The maximum class score is returned as confidence.
7. **Display results:** The frontend shows the aspect, sentiment, and confidence in a table.

## Project Structure

```text
project-root/
├── app/
│   ├── main.py                 # FastAPI backend and prediction endpoint
│   └── frontend.py             # Streamlit frontend (alternative interface)
├── data/
│   └── Laptop_Train_v2.csv     # Training dataset
├── models/
│   ├── absa_bilstm_attention.keras  # Trained sentiment model
│   ├── label_encoder.pkl            # Saved sentiment-label encoder
│   └── tokenizer.pkl                # Saved tokenizer
├── static/
│   ├── script.js               # Frontend API request and results rendering
│   └── style.css               # Frontend styling
├── index.html                  # HTML frontend
├── model.ipynb                 # Notebook for model experimentation/training
├── demo.py                     # Demo script
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python runtime declaration
├── .env                        # Local environment variables (do not commit)
└── .gitignore
```

The tree reflects the folder structure and files shared for this project. Adjust filenames if your local repository differs.

## Requirements

- Python **3.10.11** (as specified in `runtime.txt`)
- A Groq API key
- The saved model artifacts in the `models/` directory
- The dataset and notebook if you want to explore or retrain the model

## Installation

### 1. Clone the repository

Replace the URL below with the repository URL if you have published the project on GitHub.

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own API key. Keep `.env` private and never commit it to version control. The backend loads this file from the project root.

## Run the Application

From the project root, start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Then open:

- **Web application:** `http://127.0.0.1:8000/`
- **Health check:** `http://127.0.0.1:8000/health`
- **Interactive API documentation:** `http://127.0.0.1:8000/docs`

The command above assumes the backend file is located at `app/main.py`, as shown in the shared folder structure.

### Streamlit frontend (optional)

The repository also contains `app/frontend.py`, which sends requests to `http://127.0.0.1:8000/predict`. If you choose to use this interface, run it separately:

```bash
streamlit run app/frontend.py
```

Note that the supplied Streamlit code sends the input under the JSON key `sentence`, while the FastAPI endpoint expects `review`. Update one side to use the same key before relying on this optional interface.

## API Endpoints

### `GET /`

Serves `index.html`.

### `GET /health`

Returns the API health status:

```json
{
  "status": "ok"
}
```

### `POST /predict`

Accepts a JSON request containing a product review.

**Request**

```json
{
  "review": "The battery is excellent, but the display is dull."
}
```

**Response shape**

```json
{
  "review": "The battery is excellent, but the display is dull.",
  "results": [
    {
      "aspect": "battery",
      "sentiment": "positive",
      "confidence": 0.95
    },
    {
      "aspect": "display",
      "sentiment": "negative",
      "confidence": 0.91
    }
  ]
}
```

The values above illustrate the response format only; they are not measured outputs from the project.

If no aspects are extracted, the endpoint returns the review with an empty `results` list. Empty reviews are rejected with a `400` response. Aspect-extraction failures can return a `502` response, and prediction failures can return a `500` response.

## Model and Inference Details

- **Aspect extraction:** Groq chat completion using `openai/gpt-oss-20b`, with temperature set to `0`.
- **Input format for classification:** `review [SEP] aspect`.
- **Maximum sequence length:** `61`.
- **Padding:** Post-padding and post-truncation.
- **Prediction selection:** The class with the highest predicted score is selected.
- **Confidence:** The highest class score is returned rounded to four decimal places.

The classification model, tokenizer, and label encoder are loaded from the `models/` directory at application startup.

## Troubleshooting

- **Missing API key:** Confirm that `GROQ_API_KEY` is set in the root `.env` file.
- **Model file not found:** Verify that all three artifacts exist in `models/` and that the filenames match those loaded by `app/main.py`.
- **Import or dependency errors:** Activate the virtual environment and install the dependencies from `requirements.txt`.
- **Groq aspect extraction fails:** Check network connectivity, API-key validity, account access, and the backend logs.
- **Frontend returns an API error:** Inspect the response in the browser developer tools and check the FastAPI terminal logs.
- **Streamlit frontend does not work:** Ensure its request payload uses `review`, because the API's Pydantic request model expects that field.

## Potential Improvements

- Add automated tests for aspect extraction, model inference, and API validation.
- Add robust handling for malformed or unexpected LLM responses.
- Evaluate the model with appropriate aspect-level sentiment metrics and a held-out test set.
- Add support for batch review analysis and CSV export.
- Improve model and dataset documentation with training configuration and evaluation results.
- Remove unused dependencies or clearly separate dependencies for the FastAPI and Streamlit interfaces.

## Author

Add your name, GitHub profile, and project repository link here.

## License

No license information was included in the supplied files. Add a `LICENSE` file if you intend to specify terms for reuse.

