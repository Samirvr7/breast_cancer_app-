"""
Generates a schema-accurate SAMPLE dataset so the app runs end-to-end out of the box.

This is a stand-in only. To use real data: download
https://www.kaggle.com/datasets/mobeenfatimah/breast-cancer-prediction-dataset-10000-patient
and drop the CSV into data/breast_cancer_prediction.csv (same column names), then re-run
train_model.py. The app does not need any code changes when you swap the file — the
training script reads categories and ranges directly from whatever CSV is present.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 4000

age = rng.integers(25, 85, N)
bmi = np.round(rng.normal(27, 5, N).clip(15, 48), 1)
tumor_size = np.round(rng.exponential(2.2, N).clip(0.1, 12), 2)
bp = rng.integers(90, 180, N)
cholesterol = rng.integers(140, 300, N)
exercise_days = rng.integers(0, 7, N)
income = rng.integers(15000, 180000, N)

gender = rng.choice(["Female", "Male"], N, p=[0.99, 0.01])
family_history = rng.choice(["Yes", "No"], N, p=[0.25, 0.75])
smoking = rng.choice(["Yes", "No"], N, p=[0.2, 0.8])
alcohol = rng.choice(["None", "Moderate", "Heavy"], N, p=[0.5, 0.4, 0.1])
phys_activity = rng.choice(["Low", "Moderate", "High"], N, p=[0.35, 0.45, 0.2])
hormone_therapy = rng.choice(["Yes", "No"], N, p=[0.15, 0.85])
menopause = rng.choice(["Pre-menopausal", "Peri-menopausal", "Post-menopausal"], N, p=[0.35, 0.15, 0.5])
genetic_mutation = rng.choice(["Yes", "No"], N, p=[0.1, 0.9])
lymph_node = rng.choice(["Yes", "No"], N, p=[0.18, 0.82])
mammogram = rng.choice(["Normal", "Suspicious", "Abnormal"], N, p=[0.6, 0.25, 0.15])
diabetes = rng.choice(["Yes", "No"], N, p=[0.18, 0.82])
breastfeeding = rng.choice(["Yes", "No"], N, p=[0.6, 0.4])
cancer_stage = rng.choice(["Stage I", "Stage II", "Stage III", "Stage IV", "N/A"], N,
                           p=[0.08, 0.07, 0.03, 0.02, 0.8])

# latent malignancy score - gives the model genuine, learnable signal
score = (
    0.03 * (age - 50)
    + 0.9 * tumor_size
    + (family_history == "Yes") * 1.1
    + (genetic_mutation == "Yes") * 1.6
    + (lymph_node == "Yes") * 1.4
    + (menopause == "Post-menopausal") * 0.4
    + (hormone_therapy == "Yes") * 0.3
    + 0.02 * (bmi - 27)
    + rng.normal(0, 1.6, N)
)
prob = 1 / (1 + np.exp(-(score - 6.5)))
biopsy_result = np.where(rng.random(N) < prob, "Malignant", "Benign")

df = pd.DataFrame({
    "Patient_ID": [f"P{100000+i}" for i in range(N)],
    "Age": age,
    "Gender": gender,
    "BMI": bmi,
    "Tumor_Size_cm": tumor_size,
    "Blood_Pressure": bp,
    "Cholesterol": cholesterol,
    "Exercise_Days_Per_Week": exercise_days,
    "Annual_Income_USD": income,
    "Family_History": family_history,
    "Smoking": smoking,
    "Alcohol_Consumption": alcohol,
    "Physical_Activity": phys_activity,
    "Hormone_Therapy": hormone_therapy,
    "Menopause_Status": menopause,
    "Genetic_Mutation": genetic_mutation,
    "Lymph_Node_Involvement": lymph_node,
    "Mammogram_Result": mammogram,
    "Diabetes": diabetes,
    "Breastfeeding_History": breastfeeding,
    "Cancer_Stage": cancer_stage,
    "Cancer": np.where(biopsy_result == "Malignant", "Yes", "No"),
    "Biopsy_Result": biopsy_result,
})

df.to_csv("data/breast_cancer_prediction.csv", index=False)
print("Wrote data/breast_cancer_prediction.csv:", df.shape)
print(df["Biopsy_Result"].value_counts(normalize=True))
