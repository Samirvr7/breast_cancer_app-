# NiT Diagnostic Console — Breast Cancer Risk Assessment

Full Flask web app for the Cognizant Nurture Hackathon use case (#4, Breast Cancer Prediction).
Two pages:

- **Assess Patient (`/`)** — intake form, model selector (Logistic Regression / Random Forest),
  live risk gauge + top contributing factors.
- **Model Report (`/dashboard`)** — accuracy/precision/recall/F1/ROC-AUC for both models,
  confusion matrices, ROC curve, dataset balance, feature importance charts.

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 2. Use the real dataset (important)

The repo ships with a **synthetic sample CSV** (`data/breast_cancer_prediction.csv`) so the
app runs immediately out of the box. Swap in the real data before your demo:

1. Download the CSV from Kaggle:
   https://www.kaggle.com/datasets/mobeenfatimah/breast-cancer-prediction-dataset-10000-patient
2. Replace `data/breast_cancer_prediction.csv` with it (same filename, same column names).
3. Re-run training:
   ```bash
   python3 train_model.py
   ```

That's it — no other code changes needed. `train_model.py` reads the column names,
category values, and numeric ranges directly from whatever CSV is in `data/`, and
regenerates the web form, model files, and dashboard metrics to match.

## 3. Run the app

```bash
python3 app.py
```

Open http://127.0.0.1:5000

## How it addresses the use case brief

| Use case requirement | Where it's handled |
|---|---|
| EDA on the dataset | `breastcancerprediction_improved.ipynb` (the notebook) |
| Preprocess (missing values, encoding, scaling) | `train_model.py` |
| Address class imbalance (SMOTE / class-weighting) | `train_model.py` — SMOTE oversampling + `class_weight='balanced'`, GridSearchCV scored on **recall** |
| Train & compare LR and Random Forest | Both trained, tuned, and shown side-by-side on `/dashboard` |
| Evaluate via precision/recall/F1/ROC-AUC, prioritizing recall | `/dashboard` — recall is the highlighted metric |
| Explainability (feature importance) | Coefficient/importance charts on `/dashboard` + top factors on each prediction |
| Deploy via API/dashboard for clinician use | This app — `/api/predict`, `/api/metrics`, and the two pages |

## Project structure

```
app.py                  Flask app (routes, prediction API)
train_model.py          Training pipeline (reads CSV, tunes LR+RF, saves everything to models/)
generate_sample_data.py Creates the placeholder dataset (only needed if you want to regenerate it)
data/                   Dataset CSV goes here
models/                 Generated: scaler.pkl, log_reg.pkl, random_forest.pkl,
                         feature_columns.json, form_schema.json, metrics.json, importances.json
templates/               index.html (predict page), dashboard.html (report page), base.html
static/css/style.css    Visual design
static/js/main.js       Prediction form + gauge
static/js/dashboard.js  Chart.js dashboard visuals
```

## Notes

- The synthetic sample dataset gives the app realistic-but-fake numbers so you can demo the
  full flow today. Present the retrained-on-real-data numbers for the actual hackathon judging.
- `Mammogram_Result` and `Cancer_Stage` are excluded from the model on purpose — they're
  post-diagnosis fields in the original dataset and would leak the answer into the model.
