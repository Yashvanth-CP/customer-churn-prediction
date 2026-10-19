# Customer Churn Prediction (Telecom)

A machine learning project that predicts which telecom customers are likely to leave (churn), with a cost-based decision threshold and a FastAPI prediction service.

## Problem

Telecom companies lose revenue when customers leave. If we can predict who is likely to leave, the company can offer a retention deal before they go. A missed churner costs far more than an unnecessary discount, so the model is tuned for **high recall**.

## Dataset

- Telco Customer Churn (Kaggle, IBM sample data)
- 7,043 rows originally, **7,032 after cleaning** (11 rows with blank `TotalCharges` removed; they were brand-new customers with tenure 0)
- 19 input features: demographics, services, contract, billing
- Target: `Churn` (Yes/No). About **26.6%** of customers churned (imbalanced data)

## Approach

1. **Cleaning:** converted `TotalCharges` to numeric, dropped `customerID`, encoded the target as 0/1.
2. **Exploration:** churn is much higher for month-to-month contracts (42.7%) than one-year (11.3%) or two-year (2.8%). Churners have a lower average tenure (about 18 vs 38 months).
3. **Modeling:** stratified 80/20 train-test split. Compared Logistic Regression and Random Forest using 5-fold cross-validation.
4. **Evaluation:** used ROC-AUC, precision, recall and F1 instead of accuracy, because the classes are imbalanced.
5. **Pipeline:** scaling and one-hot encoding are inside a scikit-learn `Pipeline`, so training and prediction use identical steps and the API accepts raw values.

## Results
![Churn predictor frontend](images/frontend.png)
![Churn predictor frontend](images/frontend2.png)
| Model | CV ROC-AUC (5-fold) |
|---|---|
| Logistic Regression | **0.845** |
| Random Forest | 0.828 |

- Test ROC-AUC of the final pipeline: **0.836**
- Hyperparameter tuning (C) and extra engineered features (`num_addons`, `tenure_group`, `avg_monthly_spend`) gave **no meaningful improvement** (changes of about 0.001 AUC). The simple baseline was already close to the best a linear model can do on this data.

### Cost-based threshold

The default threshold of 0.5 misses too many churners. I chose the threshold by minimizing total business cost on cross-validated training predictions:

- Assumed cost of a missed churner: **₹500**
- Assumed cost of a false alarm (unneeded retention offer): **₹50**
- Best threshold: **0.12**

On the held-out test set at 0.12:

| Metric (churn class) | Value |
|---|---|
| Recall | 0.93 |
| Precision | 0.41 |

The model catches about 93% of churners, at the price of many false alarms, which is acceptable when a false alarm is 10x cheaper than a missed customer. The cost numbers are **assumptions for illustration**; a real company would use its own.

### Key drivers (Logistic Regression coefficients)

- Higher churn: fiber optic internet, streaming services, multiple lines
- Lower churn: longer tenure, one- and two-year contracts

These are patterns in the data, not proof of cause.

## API

A FastAPI service loads the saved pipeline and returns a churn probability.

**Run:**

```
pip install fastapi uvicorn scikit-learn pandas joblib
python -m uvicorn app:app --reload
```

**Endpoints:**

- `GET /api/health` returns `{"status": "ok"}`
- `POST /api/predict-churn` takes the 19 customer fields and returns:

```json
{"churn_probability": 0.758, "will_churn": true}
```

Example request body:

```json
{
  "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
  "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
  "StreamingMovies": "Yes", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.0, "TotalCharges": 190.0
}
```

Interactive docs are available at `/docs` when the server is running.

## Frontend

Open `index.html` in a browser while the API is running. Fill in the customer form and click **Predict**.

## Project structure

```
churn-project/
├── data/                  # dataset CSV (not committed; download from Kaggle)
├── 01_Preprocessing.py    # cleaning, EDA, modeling, threshold, saves the pipeline
├── app.py                 # FastAPI service
├── index.html             # simple frontend
├── churn_pipeline.pkl     # trained pipeline (scaler + encoder + model)
├── raw_cols.pkl           # raw input column order
├── test_api.py            # sends sample requests to the API
└── README.md
```

## Limitations and next steps

- Single dataset of about 7,000 customers; results may not transfer to other companies.
- Cost values are assumptions.
- Probabilities were not explicitly calibrated.
- Next: try gradient boosting (XGBoost) and a neural network, add SHAP explanations, and deploy the API.

## Tech stack

Python, pandas, scikit-learn, FastAPI, joblib, matplotlib, seaborn