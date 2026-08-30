import csv
import io
import json
import pickle
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from flask import (
    Flask, render_template, request, jsonify, session,
    redirect, url_for, flash, Response, abort, send_file
)
from werkzeug.security import generate_password_hash, check_password_hash

import db 
import report_pdf

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"

app = Flask(__name__)
app.config["SECRET_KEY"] = "curequest-dev-secret"  # move to an env var before real deployment

db.init_db()

# ---------------------------------------------------------------- load once
with open(MODELS_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(MODELS_DIR / "log_reg.pkl", "rb") as f:
    log_reg = pickle.load(f)
with open(MODELS_DIR / "random_forest.pkl", "rb") as f:
    random_forest = pickle.load(f)
with open(MODELS_DIR / "feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)
with open(MODELS_DIR / "form_schema.json") as f:
    FORM_SCHEMA = json.load(f)
with open(MODELS_DIR / "metrics.json") as f:
    METRICS = json.load(f)
with open(MODELS_DIR / "importances.json") as f:
    IMPORTANCES = json.load(f)

MODELS = {
    "logistic_regression": {"model": log_reg, "label": "CureQuest LR"},
    "random_forest": {"model": random_forest, "label": "CureQuest RF"},
}

# Real per-prediction explainability:
# - Random Forest: exact SHAP values via TreeExplainer (fast, <5ms per row)
# - Logistic Regression: exact local contribution = coefficient x scaled value
#   (this IS the model's actual math for a linear model -- not an approximation)
RF_EXPLAINER = shap.TreeExplainer(random_forest)


def build_feature_row(payload: dict):
    row = {}
    for f in FORM_SCHEMA["numerical"]:
        row[f["name"]] = float(payload.get(f["name"], f["default"]))
    for f in FORM_SCHEMA["categorical"]:
        row[f["name"]] = payload.get(f["name"], f["options"][0])

    df = pd.DataFrame([row])
    df_encoded = pd.get_dummies(df)
    for col in FEATURE_COLUMNS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[FEATURE_COLUMNS]
    return df_encoded, row


def explain_prediction(model_key, row_scaled_df):
    """Returns top contributing factors for THIS specific patient, not global importance."""
    if model_key == "random_forest":
        sv = RF_EXPLAINER.shap_values(row_scaled_df)
        arr = np.array(sv)
        # shape is (1, n_features, n_classes) in recent shap versions
        if arr.ndim == 3:
            contribs = arr[0, :, 1]  # contribution toward the Malignant class
        else:
            contribs = arr[1][0] if isinstance(sv, list) else arr[0]
        pairs = list(zip(FEATURE_COLUMNS, contribs))
    else:
        coefs = log_reg.coef_[0]
        values = row_scaled_df.iloc[0].values
        contribs = coefs * values  # exact contribution to the log-odds for THIS patient
        pairs = list(zip(FEATURE_COLUMNS, contribs))

    pairs.sort(key=lambda x: -abs(x[1]))
    top = pairs[:6]
    return [
        {"feature": f, "value": round(float(v), 4), "direction": "up" if v > 0 else "down"}
        for f, v in top
    ]


# ---------------------------------------------------------------- auth helpers
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def doctor_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
        if session.get("role") != "doctor":
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def patient_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
        if session.get("role") != "patient":
            abort(403)
        return view(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_user():
    user = None
    if session.get("user_id"):
        row = db.get_user_by_id(session["user_id"])
        if row:
            user = {"id": row["id"], "full_name": row["full_name"],
                    "email": row["email"], "role": row["role"]}
    return {"current_user": user}


@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", code=403,
                            message="You don't have access to this. Doctors can only view patients who've explicitly shared access with them."), 403


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="That page doesn't exist."), 404


# ---------------------------------------------------------------- auth routes
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "patient")
        if role not in ("patient", "doctor"):
            role = "patient"

        if not full_name or not email or not password:
            flash("All fields are required.")
            return render_template("signup.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters.")
            return render_template("signup.html")
        if db.get_user_by_email(email):
            flash("An account with that email already exists.")
            return render_template("signup.html")

        db.create_user(full_name, email, generate_password_hash(password), role)
        user = db.get_user_by_email(email)
        session["user_id"] = user["id"]
        session["role"] = user["role"]
        return redirect(url_for("doctor_patients") if role == "doctor" else url_for("index"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = db.get_user_by_email(email)

        if not user or not check_password_hash(user["password_hash"], password):
            flash("Incorrect email or password.")
            return render_template("login.html")

        session["user_id"] = user["id"]
        session["role"] = user["role"]
        next_url = request.args.get("next")
        if next_url:
            return redirect(next_url)
        return redirect(url_for("doctor_patients") if user["role"] == "doctor" else url_for("index"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------- app routes
@app.route("/")
@login_required
def index():
    return render_template(
        "index.html", schema=FORM_SCHEMA, models=MODELS, active="predict",
    )


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template(
        "dashboard.html", metrics=METRICS, importances=IMPORTANCES,
        metrics_json=json.dumps(METRICS), importances_json=json.dumps(IMPORTANCES),
        active="dashboard",
    )


@app.route("/history")
@login_required
def history():
    rows = db.get_history(session["user_id"])
    records = [{
        "id": r["id"], "model_used": r["model_used"], "inputs": json.loads(r["inputs_json"]),
        "prediction": r["prediction"], "probability_malignant": r["probability_malignant"],
        "risk_tier": r["risk_tier"], "created_at": r["created_at"],
    } for r in rows]
    return render_template("history.html", records=records, active="history")


@app.route("/report/pdf")
@patient_required
def download_own_report():
    user = db.get_user_by_id(session["user_id"])
    rows = db.get_history(session["user_id"], limit=1000)
    records = [{
        "model_used": r["model_used"], "inputs": json.loads(r["inputs_json"]),
        "prediction": r["prediction"], "probability_malignant": r["probability_malignant"],
        "risk_tier": r["risk_tier"], "created_at": r["created_at"],
    } for r in rows]
    pdf_buf = report_pdf.generate_report_pdf(
        patient_name=user["full_name"], patient_email=user["email"],
        records=records, viewer_label="Patient self-report",
    )
    filename = f"{user['full_name'].replace(' ', '_')}_CureQuest_Report.pdf"
    return send_file(pdf_buf, mimetype="application/pdf", as_attachment=True, download_name=filename)


@app.route("/profile")
@login_required
def profile():
    user = db.get_user_by_id(session["user_id"])
    recent, shared_doctors = [], []
    if user["role"] == "patient":
        rows = db.get_history(session["user_id"], limit=5)
        recent = [{
            "prediction": r["prediction"], "risk_tier": r["risk_tier"],
            "probability_malignant": r["probability_malignant"], "created_at": r["created_at"],
        } for r in rows]
        shared_doctors = db.get_shared_doctors(session["user_id"])
    return render_template("profile.html", user=user, recent=recent, shared_doctors=shared_doctors, active="profile")


@app.route("/profile/share", methods=["POST"])
@patient_required
def profile_share():
    doctor_email = request.form.get("doctor_email", "").strip().lower()
    result = db.share_with_doctor(session["user_id"], doctor_email)
    messages = {
        "ok": f"Access granted to {doctor_email}.",
        "not_found": f"No account found for {doctor_email}.",
        "not_doctor": f"{doctor_email} is not registered as a doctor account.",
        "already": f"{doctor_email} already has access.",
    }
    flash(messages.get(result, "Something went wrong."))
    return redirect(url_for("profile"))


@app.route("/profile/revoke/<int:doctor_id>", methods=["POST"])
@patient_required
def profile_revoke(doctor_id):
    db.revoke_share(session["user_id"], doctor_id)
    flash("Access revoked.")
    return redirect(url_for("profile"))


# ---------------------------------------------------------------- doctor-only routes (consent-gated)
@app.route("/doctor/patients")
@doctor_required
def doctor_patients():
    patients = db.get_patients_shared_with_doctor(session["user_id"])
    return render_template("doctor_patients.html", patients=patients, active="patients")


@app.route("/doctor/patients/<int:patient_id>")
@doctor_required
def doctor_patient_detail(patient_id):
    if not db.is_shared_with(patient_id, session["user_id"]):
        abort(403)
    patient = db.get_user_by_id(patient_id)
    if not patient or patient["role"] != "patient":
        abort(404)
    rows = db.get_history(patient_id, limit=200)
    records = [{
        "model_used": r["model_used"], "inputs": json.loads(r["inputs_json"]),
        "prediction": r["prediction"], "probability_malignant": r["probability_malignant"],
        "risk_tier": r["risk_tier"], "created_at": r["created_at"],
    } for r in rows]
    return render_template("doctor_patient_detail.html", patient=patient, records=records, active="patients")


@app.route("/doctor/patients/<int:patient_id>/download")
@doctor_required
def doctor_patient_download(patient_id):
    if not db.is_shared_with(patient_id, session["user_id"]):
        abort(403)
    patient = db.get_user_by_id(patient_id)
    if not patient or patient["role"] != "patient":
        abort(404)
    rows = db.get_history(patient_id, limit=1000)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date", "Model", "Prediction", "Risk Tier", "Probability Malignant", "Inputs"])
    for r in rows:
        writer.writerow([
            r["created_at"], r["model_used"], r["prediction"], r["risk_tier"],
            r["probability_malignant"], json.loads(r["inputs_json"]),
        ])

    filename = f"{patient['full_name'].replace(' ', '_')}_report.csv"
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/doctor/patients/<int:patient_id>/download/pdf")
@doctor_required
def doctor_patient_download_pdf(patient_id):
    if not db.is_shared_with(patient_id, session["user_id"]):
        abort(403)
    patient = db.get_user_by_id(patient_id)
    if not patient or patient["role"] != "patient":
        abort(404)
    doctor = db.get_user_by_id(session["user_id"])
    rows = db.get_history(patient_id, limit=1000)
    records = [{
        "model_used": r["model_used"], "inputs": json.loads(r["inputs_json"]),
        "prediction": r["prediction"], "probability_malignant": r["probability_malignant"],
        "risk_tier": r["risk_tier"], "created_at": r["created_at"],
    } for r in rows]
    pdf_buf = report_pdf.generate_report_pdf(
        patient_name=patient["full_name"], patient_email=patient["email"],
        records=records, viewer_label=f"Dr. {doctor['full_name']}",
    )
    filename = f"{patient['full_name'].replace(' ', '_')}_CureQuest_Report.pdf"
    return send_file(pdf_buf, mimetype="application/pdf", as_attachment=True, download_name=filename)


# ---------------------------------------------------------------- prediction API
@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    payload = request.get_json(force=True)
    model_key = payload.get("model", "random_forest")
    if model_key not in MODELS:
        return jsonify({"error": "unknown model"}), 400

    try:
        row, raw_inputs = build_feature_row(payload)
    except Exception as e:
        return jsonify({"error": f"invalid input: {e}"}), 400

    row_scaled = pd.DataFrame(scaler.transform(row), columns=row.columns)
    model = MODELS[model_key]["model"]
    prob_malignant = float(model.predict_proba(row_scaled)[0][1])
    prediction = "Malignant" if prob_malignant >= 0.5 else "Benign"

    if prob_malignant < 0.3:
        tier = "low"
    elif prob_malignant < 0.6:
        tier = "moderate"
    else:
        tier = "high"

    top_factors = explain_prediction(model_key, row_scaled)

    db.save_assessment(
        user_id=session["user_id"], model_used=MODELS[model_key]["label"],
        inputs=raw_inputs, prediction=prediction,
        probability_malignant=round(prob_malignant, 4), risk_tier=tier,
    )

    return jsonify({
        "model_used": MODELS[model_key]["label"], "model_key": model_key,
        "prediction": prediction,
        "probability_malignant": round(prob_malignant, 4),
        "probability_benign": round(1 - prob_malignant, 4), "risk_tier": tier,
        "top_factors": top_factors,
    })


@app.route("/api/metrics")
@login_required
def api_metrics():
    return jsonify(METRICS)


@app.route("/api/importances")
@login_required
def api_importances():
    return jsonify(IMPORTANCES)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
