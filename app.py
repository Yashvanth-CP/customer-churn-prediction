from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



pipeline = joblib.load("churn_pipeline.pkl")
raw_cols = joblib.load("raw_cols.pkl")
THRESHOLD = 0.12

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict-churn")
def predict(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])[raw_cols]
    proba = float(pipeline.predict_proba(row)[0, 1])
    return {
        "churn_probability": round(proba, 3),
        "will_churn": proba >= THRESHOLD,
    }