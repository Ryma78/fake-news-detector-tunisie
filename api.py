from fastapi import FastAPI
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = FastAPI(title="🚨 Fake-News Detector Tunisie F1=92%")

# CHARGE modèles
model = joblib.load("model/svm_optuna_F1_92pct.joblib")
vectorizer = joblib.load("model/tfidf_vectorizer.joblib")

@app.post("/predict")
async def predict(text: str):
    text_clean = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text.lower())
    X = vectorizer.transform([text_clean])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].max()
    return {"fake": bool(pred), "confidence": float(proba), "f1_score": 0.9234}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)