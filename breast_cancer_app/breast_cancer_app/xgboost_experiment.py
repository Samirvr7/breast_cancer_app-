"""
Standalone experiment: does XGBoost beat Random Forest on precision (while keeping
recall reasonably high)? Uses the exact same preprocessing/split as train_model.py
so the comparison is apples-to-apples.

This does NOT touch your mandatory LR + RF pipeline or the web app — it's a
side-by-side check. Run it any time with your real CSV in data/.
"""
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

DATA_PATH = "data/breast_cancer_prediction.csv"
TARGET = "Biopsy_Result"
DROP_ALWAYS = ["Patient_ID", "Cancer", "Cancer_Stage", "Mammogram_Result"]

df = pd.read_csv(DATA_PATH)
for col in DROP_ALWAYS:
    if col in df.columns:
        df = df.drop(columns=col)

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include=["object", "category"]).columns.drop(TARGET, errors="ignore")
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

df[TARGET] = df[TARGET].map({"Malignant": 1, "Benign": 0})
df_encoded = pd.get_dummies(df, columns=list(cat_cols), drop_first=True)

X = df_encoded.drop(columns=[TARGET])
y = df_encoded[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

corr = X_train.corr().abs()
mask = np.triu(np.ones_like(corr, dtype=bool))
tri = corr.mask(mask)
to_drop = [c for c in tri.columns if any(tri[c] > 0.92)]
X_train = X_train.drop(columns=to_drop, errors="ignore")
X_test = X_test.drop(columns=to_drop, errors="ignore")

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# scale_pos_weight = negative/positive ratio -- XGBoost's equivalent of class_weight='balanced'
pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = neg / pos

print("Tuning XGBoost (scored on F1, to balance precision AND recall)...")
xgb_grid = GridSearchCV(
    XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    ),
    {
        "n_estimators": [200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
    },
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1",
    n_jobs=-1,
)
xgb_grid.fit(X_train_s, y_train)
xgb_model = xgb_grid.best_estimator_
print("Best XGBoost params:", xgb_grid.best_params_)

def evaluate(name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    cm = confusion_matrix(y_te, y_pred)
    return {
        "name": name,
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred), 4),
        "recall": round(recall_score(y_te, y_pred), 4),
        "f1": round(f1_score(y_te, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_te, y_prob), 4),
        "confusion_matrix": cm.tolist(),
    }

result = evaluate("XGBoost", xgb_model, X_test_s, y_test)

print("\n" + "="*55)
print("XGBoost results (F1-optimized, same split as LR/RF):")
print("="*55)
for k, v in result.items():
    print(f"{k:>16}: {v}")

# compare against existing RF/LR if metrics.json exists
try:
    with open("models/metrics.json") as f:
        existing = json.load(f)
    print("\n" + "="*55)
    print("Comparison vs your existing models:")
    print("="*55)
    print(f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'ROC-AUC':>9}")
    for key in ["logistic_regression", "random_forest"]:
        m = existing[key]
        print(f"{m['name']:<22}{m['accuracy']:>10}{m['precision']:>11}{m['recall']:>9}{m['f1']:>8}{m['roc_auc']:>9}")
    print(f"{'XGBoost':<22}{result['accuracy']:>10}{result['precision']:>11}{result['recall']:>9}{result['f1']:>8}{result['roc_auc']:>9}")
except FileNotFoundError:
    pass
