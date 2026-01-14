# 📰 Détecteur de Fake News Tunisie

Un système d'intelligence artificielle pour détecter les fake news en Tunisie utilisant le machine learning.

## ✨ Fonctionnalités

- 🤖 Modèle ML entraîné sur des données tunisiennes
- 🚀 API FastAPI pour les prédictions
- 🎨 Interface Streamlit moderne et intuitive
- 🐳 Déploiement Docker complet
- 📊 Suivi MLflow des expériences
- 🔧 Pipeline de preprocessing automatique

## 🚀 Démarrage rapide

### Avec Docker (recommandé)
```bash
# Construire et lancer tous les services
docker-compose up --build

# Accéder à l'application
# - Frontend Streamlit: http://localhost:8501
# - API FastAPI: http://localhost:8000
# - MLflow UI: http://localhost:5000
```

### Installation locale
```bash
# Installer les dépendances
pip install -r requirements.txt

# Entraîner le modèle
python optuna_svm.py

# Lancer l'API
uvicorn predict_api:app --host 0.0.0.0 --port 8000

# Lancer le frontend (dans un autre terminal)
streamlit run streamlit_frontend.py
```

## 📊 Performance du modèle

- **Algorithme**: Régression Logistique + TF-IDF
- **Précision**: F1-Score de 100% sur les données d'entraînement
- **Données**: 21,353 textes tunisiens
- **Fake News**: 5,227 (24.5%)
- **News Réelles**: 16,126 (75.5%)

## 🛠️ Architecture du projet

```
fake-news-detector/
├── optuna_svm.py              # Entraînement avec optimisation Optuna
├── predict_api.py             # API FastAPI pour les prédictions
├── streamlit_frontend.py      # Interface utilisateur moderne
├── data_preprocessing.py      # Préparation des données
├── data_generator_pro.py      # Génération du dataset tunisien
├── requirements.txt           # Dépendances Python
├── Dockerfile                 # Image Docker
├── docker-compose.yml         # Orchestration des services
├── data/
│   ├── raw/                  # Données brutes (GossipCop, PolitiFact)
│   ├── processed/            # Données traitées
│   └── *.csv                 # Datasets finaux
├── model/                    # Modèles entraînés (.joblib)
└── mlruns/                   # Suivi des expériences MLflow
```

## 📡 API Documentation

### Prédire une fake news

**Endpoint:** `POST /predict`

**Requête:**
```json
{
  "text": "Le président tunisien annonce de nouvelles réformes économiques."
}
```

**Réponse:**
```json
{
  "text": "Le président tunisien annonce de nouvelles réformes économiques.",
  "prediction": 0,
  "probability_fake": 0.15,
  "probability_real": 0.85,
  "is_fake": false
}
```

### Vérification santé

**Endpoint:** `GET /health`

Retourne le statut de l'API et si le modèle est chargé.

## 🔧 Technologies utilisées

- **Python 3.11**
- **Scikit-learn** - Machine Learning
- **FastAPI** - API web
- **Streamlit** - Interface utilisateur
- **Optuna** - Optimisation hyperparamètres
- **MLflow** - Suivi des expériences
- **Docker** - Conteneurisation
- **Pandas** - Manipulation des données

## 📚 Sources de données

- **GossipCop**: Actualités réelles et fake depuis des sites de ragots
- **PolitiFact**: Données de vérification factuelle politique
- **News Tunisiennes**: Dataset personnalisé pour les actualités tunisiennes

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👨‍💻 Auteur

**Ryma78** - *Développement initial*

---

⭐ Si ce projet vous plaît, n'oubliez pas de mettre une étoile !
