from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

app = FastAPI(title="Mental Health Predictor API")

# Load models
best_clf = joblib.load('xgboost_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
pca_final = joblib.load('pca_final.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

class PredictionInput(BaseModel):
    Age: int
    Gender: str
    Education_Level: str
    Employment_Status: str
    Work_Hours_Week: float
    Sleep_Hours_Night: float
    Exercise_Freq_Week: float
    Financial_Strain: float
    Relationship_Quality: float
    GAD7_Score: float
    PHQ9_Score: float
    Stress_Level_Scale: float

@app.post("/predict")
def predict_risk(data: PredictionInput):
    df_input = data.dict()
    X_input = preprocessor.transform(pd.DataFrame([df_input]))
    
    # Classification
    risk_pred_encoded = best_clf.predict(X_input)[0]
    risk_proba = best_clf.predict_proba(X_input)[0].tolist()
    risk_pred = label_encoder.inverse_transform([risk_pred_encoded])[0]
    
    # Clustering
    X_input_pca = pca_final.transform(X_input)
    cluster = kmeans_model.predict(X_input_pca)[0]
    
    return {
        "risk_level": risk_pred,
        "probabilities": dict(zip(label_encoder.classes_.tolist(), risk_proba)),
        "cluster": int(cluster),
        "recommendations": generate_recommendations(risk_pred, cluster, df_input)
    }
