from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
import re
from sklearn.pipeline import Pipeline

app = FastAPI(title="Fake News Detector API", version="1.0.0")

# Modèle global
model = None

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    prediction: int
    probability_fake: float
    probability_real: float
    is_fake: bool

def load_model():
    """Charge le modèle entraîné"""
    global model

    # Essayer différents noms de modèles (du plus récent au plus ancien)
    model_paths = [
        'model/svm_optuna_F1_100pct.joblib',
        'model/svm_optuna_F1_92pct.joblib',
        'model/svm_optuna_F1_0pct.joblib',
        'model/svm_linear.joblib',
        'model/svm_baseline_F1_100pct.joblib'
    ]

    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                print(f"✅ Modèle chargé: {path}")
                return True
            except Exception as e:
                print(f"❌ Erreur chargement {path}: {e}")
                continue

    print("❌ Aucun modèle trouvé dans model/")
    return False

@app.on_event("startup")
async def startup_event():
    """Chargement du modèle au démarrage"""
    if not load_model():
        print("⚠️ API démarrée sans modèle - prédictions impossibles")

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "Fake News Detector API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST) - Prédire si un texte est fake news"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Prédit si le texte donné est une fake news"""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Entraînez d'abord le modèle avec optuna_svm.py"
        )

    try:
        # Nettoyer le texte comme dans l'entraînement
        text_clean = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', request.text.lower())
        # Prédiction
        prediction = model.predict([text_clean])[0]

        # Probabilités (si disponible)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([text_clean])[0]
            prob_fake = float(probabilities[1])  # Probabilité fake (label 1)
            prob_real = float(probabilities[0])  # Probabilité real (label 0)
        else:
            # Si pas de predict_proba, utiliser la prédiction binaire
            prob_fake = 1.0 if prediction == 1 else 0.0
            prob_real = 1.0 - prob_fake

        return PredictionResponse(
            text=request.text,
            prediction=int(prediction),
            probability_fake=round(prob_fake, 3),
            probability_real=round(prob_real, 3),
            is_fake=bool(prediction == 1)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

@app.get("/health")
async def health():
    """Vérification de santé de l'API"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }