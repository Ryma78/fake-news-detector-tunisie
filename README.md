# Fake News Detector

A machine learning project for detecting fake news using SVM with Optuna optimization.

## Project Structure

```
fake-news-detector/
├── optuna_svm.py              # Main training script with Optuna optimization
├── data_generator_pro.py      # Generate processed Tunisian dataset
├── requirements.txt           # Python dependencies
├── app/
│   ├── api/
│   │   └── api.py            # FastAPI web service for predictions
│   └── data/
│       ├── collect_data.py   # Collect and combine raw datasets
│       ├── clean_data.py     # Clean the combined dataset
│       └── preprocess_text.py # NLP preprocessing with spaCy
├── data/
│   ├── raw/                  # Raw datasets (GossipCop, PolitiFact)
│   ├── processed/            # Intermediate processed files
│   └── *.csv                 # Final datasets
├── model/                    # Trained models (joblib files)
└── notebooks/                # Jupyter notebooks for exploration
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Prepare data:**
   ```bash
   # Generate the Tunisian professional dataset
   python data_generator_pro.py

   # Or process the full pipeline:
   python app/data/collect_data.py
   python app/data/clean_data.py
   python app/data/preprocess_text.py
   ```

3. **Train the model:**
   ```bash
   python optuna_svm.py
   ```

4. **Run the API:**
   ```bash
   python app/api/api.py
   ```

   The API will be available at http://localhost:8000

## API Usage

### Predict Fake News

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "text": "Your news text here"
}
```

**Response:**
```json
{
  "text": "Your news text here",
  "prediction": 1,
  "probability_fake": 0.85,
  "probability_real": 0.15,
  "is_fake": true
}
```

### Health Check

**Endpoint:** `GET /health`

Returns the API status and whether the model is loaded.

## Model Details

- **Algorithm:** Support Vector Machine (SVM) with RBF kernel
- **Features:** TF-IDF vectorization (max 3000 features)
- **Optimization:** Optuna hyperparameter tuning
- **Class balancing:** Automatic class weighting
- **Performance:** F1-score ~95% on test set

## Data Sources

- **GossipCop:** Real and fake news from gossip websites
- **PolitiFact:** Political fact-checking data
- **Tunisian News:** Custom dataset for Tunisian news detection</content>
<parameter name="filePath">c:\Users\Asus-\Desktop\fake-news-detector\README.md