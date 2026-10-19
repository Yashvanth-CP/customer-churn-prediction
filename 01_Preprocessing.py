import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
import joblib




df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.shape)
print(df.head())
print(df["Churn"].value_counts)
print(df["Partner"].values)

# cleaning and exploring the data 

df.info()
print(df.isnull().sum())
print(df.duplicated().sum())

# converting Totalcharges to numbers

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df["TotalCharges"].isnull().sum())

# customer having tenure 0: new customer drop them 
df = df.dropna(subset=["TotalCharges"])

# customerID is just label, not usnful for learning
df = df.drop("customerID", axis=1)
df["Churn"]  = df["Churn"].map({"Yes": 1, "No": 0})


# Churn Rate 

print(df["Churn"].mean())
print(df.groupby("Contract")["Churn"].mean())
print(df.groupby("Churn")[["tenure", "MonthlyCharges"]].mean())
print(df["tenure"].mean())


df_encoded = pd.get_dummies(df, drop_first=True)
print(df_encoded.shape)
print(df_encoded.head())



X = df_encoded.drop("Churn", axis=1) # input (features)
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, X_test.shape)


model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train) # learning happend here

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["stay", "Churn"]).plot()
plt.show()

y_proba = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

threshold = 0.4
y_pred_new = (y_proba >= threshold).astype(int)
print(classification_report(y_test, y_pred_new))



rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
print(classification_report(y_test, rf.predict(X_test)))
print("RF ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))



models = {"Logistic Regression": model, "RandomForest": rf}

for name, m in models.items():
    proba = m.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.3).astype(int)
    print(name,
          "| P:", round(precision_score(y_test, pred), 2),
          "| R:", round(recall_score(y_test, pred),2),
          "| F1:", round(f1_score(y_test, pred), 2),
          "| AUC:", round(roc_auc_score(y_test, proba), 3))



cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, m in {"Logistic Regression": model, "Random Forest": rf}.items():
    scores = cross_val_score(m, X, y, cv = cv, scoring="roc_auc")
    print(name, "| mean AUC:", round(scores.mean(), 3), "| std:", round(scores.std(), 3))

importance = pd.Series(rf.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).head(10).plot(kind="barh")
plt.title("Top 10 features for churn")
plt.gca().invert_yaxis()
plt.show()




pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

params = {"clf__C": [0.01, 0.1, 1, 100]}

grid = GridSearchCV(pipe, params, cv=cv, scoring="roc_auc")
grid.fit(X_train, y_train)

print("Best C:", grid.best_params_)
print("Best CV AUC:", round(grid.best_score_,3))
print("Test AUC:", round(roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1]), 3))


scaler = grid.best_estimator_.named_steps["scale"]
clf = grid.best_estimator_.named_steps["clf"]

coefs = pd.Series(clf.coef_[0], index=X.columns).sort_values()
print(coefs.head(5))   # strongest push toward STAY
print(coefs.tail(5))   # strongest push toward CHURN

df_fe = df.copy()

addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
          "StreamingTV", "StreamingMovies"]

df_fe["num_addons"] = (df_fe[addons] == "Yes").sum(axis=1)
df_fe["tenure_group"] = pd.cut(df_fe["tenure"],
                        bins=[-1, 12, 24, 48, 72],
                        labels=["0-12", "13-24", "25-48", "49-72"])

df_fe["avg_monthly_spend"] = df_fe["TotalCharges"]/ df_fe["tenure"]

df_fe = pd.get_dummies(df_fe, drop_first=True)


X2 = df_fe.drop("Churn", axis=1)
y2 = df_fe["Churn"]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)

grid2 = GridSearchCV(pipe, params, cv=cv, scoring="roc_auc")
grid2.fit(X2_train, y2_train)

print("Best C:", grid2.best_params_)
print("CV AUC:", round(grid2.best_score_, 3))
print("Test AUC:", round(roc_auc_score(y2_test, grid2.predict_proba(X2_test)[:, 1]), 3))


best_model = grid.best_estimator_

# probability from train Data

proba_cv = cross_val_predict(best_model, X_train, y_train,
                             cv=cv, method="predict_proba")[:, 1]

COST_FN = 500 # Missed churner
COST_FP = 50 # false alarm

y_arr = np.array(y_train)
thresholds = np.arange(0.02, 0.52, 0.02)
costs = []
for t in thresholds:
    pred = (proba_cv >= t).astype(int)
    fn = ((pred == 0) & (y_arr == 1)).sum()
    fp = ((pred == 1) & (y_arr == 0)).sum()
    costs.append(fn * COST_FN + fp * COST_FP)

best_t = thresholds[int(np.argmin(costs))]
print("Best threshold:", round(best_t, 2), "| cost:", min(costs))

# Final test, once
pred_test = (best_model.predict_proba(X_test)[:, 1] >= best_t).astype(int)
print(classification_report(y_test, pred_test))
print(proba_cv.shape, y_train.shape)   # both should be (5625,)

joblib.dump(best_model, "churn_model.pkl")

joblib.dump(list(X.columns), "feature_cols.pkl")


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

X_raw = df.drop("Churn", axis=1)
y_raw = df["Churn"]

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in X_raw.columns if c not in num_cols]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

final_pipe = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(C=1, max_iter=2000)),
])

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

final_pipe.fit(Xr_train, yr_train)
print("Test AUC:", round(roc_auc_score(yr_test, final_pipe.predict_proba(Xr_test)[:, 1]), 3))

joblib.dump(final_pipe, "churn_pipeline.pkl")
joblib.dump(list(X_raw.columns), "raw_cols.pkl")