"""
Trains Logistic Regression + Random Forest on data/breast_cancer_prediction.csv.

Fully driven by whatever CSV is in data/ — column names, categories, and value ranges
are all read from the file, not hardcoded. Drop in the real Kaggle CSV (same schema)
and re-run this script; nothing else in the app needs to change.

Outputs (all in models/):
  scaler.pkl, log_reg.pkl, random_forest.pkl, feature_columns.json,
  form_schema.json (drives the web form), metrics.json (drives the dashboard),
  importances.json
"""
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE

DATA_PATH = "data/breast_cancer_prediction.csv"
MODELS_DIR = "models"

TARGET = "Biopsy_Result"
DROP_ALWAYS = ["Patient_ID", "Cancer", "Cancer_Stage", "Mammogram_Result"]  # leakage / IDs

print("Loading", DATA_PATH)
df = pd.read_csv(DATA_PATH)

for col in DROP_ALWAYS:
    if col in df.columns:
        df = df.drop(columns=col)

# fill missing
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include=["object", "category"]).columns.drop(TARGET, errors="ignore")

for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

df[TARGET] = df[TARGET].map({"Malignant": 1, "Benign": 0})

# ---- capture form schema BEFORE dummy-encoding, so the web form matches the raw data ----
form_schema = {"numerical": [], "categorical": []}
for c in num_cols:
    form_schema["numerical"].append({
        "name": c,
        "min": float(df[c].min()),
        "max": float(df[c].max()),
        "default": float(df[c].median()),
        "step": 1 if pd.api.types.is_integer_dtype(df[c]) else 0.1
    })
for c in cat_cols:
    form_schema["categorical"].append({
        "name": c,
        "options": sorted(df[c].dropna().unique().tolist())
    })

with open(f"{MODELS_DIR}/form_schema.json", "w") as f:
    json.dump(form_schema, f, indent=2)

# ---- encode ----
df_encoded = pd.get_dummies(df, columns=list(cat_cols), drop_first=True)

X = df_encoded.drop(columns=[TARGET])
y = df_encoded[TARGET]

feature_columns = X.columns.tolist()
with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# drop near-duplicate correlated features (fit on train only)
corr = X_train.corr().abs()
mask = np.triu(np.ones_like(corr, dtype=bool))
tri = corr.mask(mask)
to_drop = [c for c in tri.columns if any(tri[c] > 0.92)]
X_train = X_train.drop(columns=to_drop, errors="ignore")
X_test = X_test.drop(columns=to_drop, errors="ignore")
feature_columns = X_train.columns.tolist()
with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

with open(f"{MODELS_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_s, y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Tuning Logistic Regression...")
lr_grid = GridSearchCV(
    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]},
    cv=cv, scoring="recall", n_jobs=-1
)
lr_grid.fit(X_train_s, y_train)
log_reg = lr_grid.best_estimator_
print("Best LR params:", lr_grid.best_params_)

print("Tuning Random Forest...")
rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    {"n_estimators": [200, 300], "max_depth": [8, 14, None], "min_samples_leaf": [1, 3]},
    cv=cv, scoring="recall", n_jobs=-1
)
rf_grid.fit(X_train_s, y_train)
random_forest = rf_grid.best_estimator_
print("Best RF params:", rf_grid.best_params_)

with open(f"{MODELS_DIR}/log_reg.pkl", "wb") as f:
    pickle.dump(log_reg, f)
with open(f"{MODELS_DIR}/random_forest.pkl", "wb") as f:
    pickle.dump(random_forest, f)

def evaluate(name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    cm = confusion_matrix(y_te, y_pred).tolist()
    return {
        "name": name,
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred), 4),
        "recall": round(recall_score(y_te, y_pred), 4),
        "f1": round(f1_score(y_te, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_te, y_prob), 4),
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr.tolist()[::max(1, len(fpr)//50)], "tpr": tpr.tolist()[::max(1, len(tpr)//50)]}
    }

metrics = {
    "logistic_regression": evaluate("Logistic Regression", log_reg, X_test_s, y_test),
    "random_forest": evaluate("Random Forest", random_forest, X_test_s, y_test),
    "class_balance": {
        "benign": int((y == 0).sum()),
        "malignant": int((y == 1).sum())
    },
    "n_samples": int(len(df)),
    "n_features": int(len(feature_columns))
}

with open(f"{MODELS_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

lr_importance = sorted(
    zip(feature_columns, np.abs(log_reg.coef_[0]).tolist()),
    key=lambda x: -x[1]
)[:10]
rf_importance = sorted(
    zip(feature_columns, random_forest.feature_importances_.tolist()),
    key=lambda x: -x[1]
)[:10]

with open(f"{MODELS_DIR}/importances.json", "w") as f:
    json.dump({
        "logistic_regression": [{"feature": f, "value": round(v, 4)} for f, v in lr_importance],
        "random_forest": [{"feature": f, "value": round(v, 4)} for f, v in rf_importance]
    }, f, indent=2)

print("\nDone. Logistic Regression:", metrics["logistic_regression"]["accuracy"],
      "acc /", metrics["logistic_regression"]["recall"], "recall")
print("Random Forest:          ", metrics["random_forest"]["accuracy"],
      "acc /", metrics["random_forest"]["recall"], "recall")
