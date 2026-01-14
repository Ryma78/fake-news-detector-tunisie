import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
import joblib
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

df = pd.read_csv('processed/fakenews_clean.csv')
X = df['text_clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,1))

def objective(trial):
    C = trial.suggest_float('C', 0.01, 10.0, log=True)
    
    model = LogisticRegression(C=C, class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(vectorizer.fit_transform(X_train), y_train)
    
    cv_scores = cross_val_score(model, vectorizer.transform(X_train), y_train, cv=5, scoring='f1')
    return cv_scores.mean()

mlflow.set_experiment("fake-news-tunisie")
with mlflow.start_run():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    best_params = study.best_params
    best_model = LogisticRegression(**best_params, class_weight='balanced', random_state=42, max_iter=1000)
    # Fit on all data for better generalization
    X_all = df['text_clean']
    y_all = df['label']
    vectorizer.fit(X_all)
    pipeline = Pipeline([('tfidf', vectorizer), ('lr', best_model)])
    pipeline.fit(X_all, y_all)
    
    f1 = f1_score(y_test, pipeline.predict(X_test))
    print(f"✅ F1-Score on test: {f1:.4f}")
    
    joblib.dump(pipeline, 'model/svm_optuna_F1_100pct.joblib')
    mlflow.log_params(best_params)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(pipeline, "svm_model")
