import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "instance" / "clinicians.db"


def get_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'patient',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            model_used TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            prediction TEXT NOT NULL,
            probability_malignant REAL NOT NULL,
            risk_tier TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shares (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(patient_id, doctor_id)
        )
    """)
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "role" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'patient'")
    conn.commit()
    conn.close()


def create_user(full_name, email, password_hash, role="patient"):
    conn = get_db()
    conn.execute(
        "INSERT INTO users (full_name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
        (full_name, email, password_hash, role, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_user_by_email(email):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return row


def get_user_by_id(user_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row


def save_assessment(user_id, model_used, inputs, prediction, probability_malignant, risk_tier):
    conn = get_db()
    conn.execute(
        """INSERT INTO assessments
           (user_id, model_used, inputs_json, prediction, probability_malignant, risk_tier, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (user_id, model_used, json.dumps(inputs), prediction,
         probability_malignant, risk_tier, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(user_id, limit=50):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM assessments WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------- sharing (consent) system
def share_with_doctor(patient_id, doctor_email):
    """Patient explicitly grants a specific doctor access. Returns 'ok' | 'not_found' | 'not_doctor' | 'already'."""
    conn = get_db()
    doctor = conn.execute("SELECT * FROM users WHERE email = ?", (doctor_email,)).fetchone()
    if not doctor:
        conn.close()
        return "not_found"
    if doctor["role"] != "doctor":
        conn.close()
        return "not_doctor"
    existing = conn.execute(
        "SELECT id FROM shares WHERE patient_id = ? AND doctor_id = ?", (patient_id, doctor["id"])
    ).fetchone()
    if existing:
        conn.close()
        return "already"
    conn.execute(
        "INSERT INTO shares (patient_id, doctor_id, created_at) VALUES (?, ?, ?)",
        (patient_id, doctor["id"], datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return "ok"


def revoke_share(patient_id, doctor_id):
    conn = get_db()
    conn.execute("DELETE FROM shares WHERE patient_id = ? AND doctor_id = ?", (patient_id, doctor_id))
    conn.commit()
    conn.close()


def get_shared_doctors(patient_id):
    conn = get_db()
    rows = conn.execute("""
        SELECT u.id, u.full_name, u.email, s.created_at AS shared_at
        FROM shares s JOIN users u ON u.id = s.doctor_id
        WHERE s.patient_id = ? ORDER BY s.created_at DESC
    """, (patient_id,)).fetchall()
    conn.close()
    return rows


def is_shared_with(patient_id, doctor_id):
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM shares WHERE patient_id = ? AND doctor_id = ?", (patient_id, doctor_id)
    ).fetchone()
    conn.close()
    return row is not None


def get_patients_shared_with_doctor(doctor_id):
    """Only patients who explicitly shared access with THIS doctor -- not every patient."""
    conn = get_db()
    rows = conn.execute("""
        SELECT
            u.id, u.full_name, u.email, u.created_at,
            COUNT(a.id) AS assessment_count,
            MAX(a.created_at) AS last_assessment_at,
            (SELECT risk_tier FROM assessments WHERE user_id = u.id ORDER BY created_at DESC LIMIT 1) AS latest_risk_tier
        FROM shares s
        JOIN users u ON u.id = s.patient_id
        LEFT JOIN assessments a ON a.user_id = u.id
        WHERE s.doctor_id = ?
        GROUP BY u.id
        ORDER BY last_assessment_at DESC NULLS LAST, u.created_at DESC
    """, (doctor_id,)).fetchall()
    conn.close()
    return rows
