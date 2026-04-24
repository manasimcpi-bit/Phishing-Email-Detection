from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import os
import sqlite3
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.metrics import confusion_matrix

from utils import (
    prepare_uploaded_dataframe,
    get_risk_level,
    save_bar_chart,
    save_probability_histogram,
    save_risk_pie_chart,
    save_confusion_heatmap
)

app = Flask(__name__)
app.secret_key = "change_this_to_a_secure_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
DB_PATH = os.path.join(BASE_DIR, "users.db")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Database functions
# -----------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # default admin
    existing_admin = conn.execute("SELECT * FROM users WHERE username = ?", ("admin",)).fetchone()
    if existing_admin is None:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", generate_password_hash("Admin@123"), "admin")
        )
        conn.commit()

    conn.close()

def get_current_user():
    if "user_id" not in session:
        return None
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    conn.close()
    return user

# -----------------------------
# Decorators
# -----------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if user is None or user["role"] != "admin":
            flash("Admin access required.")
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated_function

# -----------------------------
# Load models
# -----------------------------
lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_model.pkl"))
rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
bert_available = False
bert_predictor = None

try:
    from bert_utils import BertPredictor
    bert_path = os.path.join(MODELS_DIR, "bert_model")

    if os.path.exists(bert_path):
        bert_predictor = BertPredictor(bert_path)
        bert_available = True
        print("BERT loaded successfully.")
    else:
        print("BERT folder not found:", bert_path)

except Exception as e:
    print("BERT loading failed:", e)
    bert_available = False

# -----------------------------
# Auth routes
# -----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            flash("Login successful.")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for("login"))

# -----------------------------
# Main app routes
# -----------------------------
@app.route("/")
@login_required
def home():
    current_user = get_current_user()

    performance = {
        "Logistic Regression": {"accuracy": 0.98, "precision": 0.98, "recall": 0.98},
        "Random Forest": {"accuracy": 0.99, "precision": 0.99, "recall": 0.99},
        "SVM": {"accuracy": 0.99, "precision": 0.99, "recall": 0.99},
        "BERT": {"accuracy": 0.95, "precision": 0.95, "recall": 0.95} if bert_available else None
    }

    model_status = {
        "Logistic Regression": "Ready",
        "Random Forest": "Ready",
        "SVM": "Ready",
        "BERT": "Ready" if bert_available else "Not Loaded"
    }

    return render_template(
        "index.html",
        current_user=current_user,
        performance=performance,
        model_status=model_status,
        bert_available=bert_available
    )

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        flash("No file uploaded.")
        return redirect(url_for("home"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("home"))

    df = pd.read_csv(file)
    df, text_col = prepare_uploaded_dataframe(df)

    X = df["clean_text"]
    X_tfidf = tfidf.transform(X)

    # classical models
    lr_preds = lr_model.predict(X_tfidf)
    rf_preds = rf_model.predict(X_tfidf)
    svm_preds = svm_model.predict(X_tfidf)

    lr_probs = lr_model.predict_proba(X_tfidf)[:, 1] if hasattr(lr_model, "predict_proba") else [0.5] * len(df)
    rf_probs = rf_model.predict_proba(X_tfidf)[:, 1] if hasattr(rf_model, "predict_proba") else [0.5] * len(df)

    # BERT
    bert_preds = None
    bert_probs = None
    runtime_bert_available = False

    if bert_available:
        try:
            bert_preds, bert_probs = bert_predictor.predict_batch(X.tolist())
            if len(bert_preds) == len(df):
                runtime_bert_available = True
        except Exception:
            runtime_bert_available = False

    # ensemble + risk
    ensemble_preds = []
    risk_levels = []

    for i in range(len(df)):
        preds_list = [int(lr_preds[i]), int(rf_preds[i]), int(svm_preds[i])]

        if runtime_bert_available:
            preds_list.append(int(bert_preds[i]))

        attack_votes = sum(preds_list)
        vote_threshold = 3 if runtime_bert_available else 2
        ensemble_pred = 1 if attack_votes >= vote_threshold else 0
        ensemble_preds.append(ensemble_pred)

        avg_probs = [float(lr_probs[i]), float(rf_probs[i])]
        if runtime_bert_available:
            avg_probs.append(float(bert_probs[i]))

        avg_prob = sum(avg_probs) / len(avg_probs)
        risk_levels.append(get_risk_level(avg_prob))

    # dataframe output
    df["LR Prediction"] = ["Attack" if x == 1 else "Normal" for x in lr_preds]
    df["LR Probability"] = [round(float(x), 3) for x in lr_probs]
    df["RF Prediction"] = ["Attack" if x == 1 else "Normal" for x in rf_preds]
    df["RF Probability"] = [round(float(x), 3) for x in rf_probs]
    df["SVM Prediction"] = ["Attack" if x == 1 else "Normal" for x in svm_preds]

    if runtime_bert_available:
        df["BERT Prediction"] = ["Attack" if x == 1 else "Normal" for x in bert_preds]
        df["BERT Probability"] = [round(float(x), 3) for x in bert_probs]

    df["Ensemble"] = ["Attack" if x == 1 else "Normal" for x in ensemble_preds]
    df["Risk Level"] = risk_levels

    # summary counts
    lr_attacks = int(sum(lr_preds))
    rf_attacks = int(sum(rf_preds))
    svm_attacks = int(sum(svm_preds))
    bert_attacks = int(sum(bert_preds)) if runtime_bert_available else None
    ensemble_attacks = int(sum(ensemble_preds))
    high_risk = int(sum(1 for x in risk_levels if x == "HIGH"))

    # charts
    model_counts = {
        "Logistic Regression": lr_attacks,
        "Random Forest": rf_attacks,
        "SVM": svm_attacks,
        "Ensemble": ensemble_attacks
    }

    if runtime_bert_available:
        model_counts["BERT"] = bert_attacks

    save_bar_chart(
        model_counts,
        os.path.join(OUTPUT_DIR, "attacks_by_model.png")
    )

    save_probability_histogram(
        lr_probs,
        rf_probs,
        os.path.join(OUTPUT_DIR, "probability_distribution.png")
    )

    save_risk_pie_chart(
        risk_levels,
        os.path.join(OUTPUT_DIR, "risk_distribution.png")
    )

    # confusion matrices if true labels exist
    possible_label_cols = ["label", "target", "Label", "Attack", "class", "label_encoded"]
    true_label_col = None
    for col in possible_label_cols:
        if col in df.columns:
            true_label_col = col
            break

    show_confusion = False
    if true_label_col:
        try:
            y_true = df[true_label_col].astype(int)

            lr_cm = confusion_matrix(y_true, lr_preds)
            rf_cm = confusion_matrix(y_true, rf_preds)

            save_confusion_heatmap(
                lr_cm,
                "Logistic Regression Confusion Matrix",
                os.path.join(OUTPUT_DIR, "lr_confusion.png")
            )

            save_confusion_heatmap(
                rf_cm,
                "Random Forest Confusion Matrix",
                os.path.join(OUTPUT_DIR, "rf_confusion.png")
            )

            show_confusion = True
        except Exception:
            show_confusion = False

    preview_df = df.head(50).copy()
    preview_rows = preview_df.to_dict(orient="records")

    return render_template(
        "result.html",
        total_rows=len(df),
        lr_attacks=lr_attacks,
        rf_attacks=rf_attacks,
        svm_attacks=svm_attacks,
        bert_attacks=bert_attacks,
        ensemble_attacks=ensemble_attacks,
        high_risk=high_risk,
        preview_rows=preview_rows,
        show_confusion=show_confusion,
        bert_available=runtime_bert_available,
        current_user=get_current_user()
    )

# -----------------------------
# Admin routes
# -----------------------------
@app.route("/admin/users", methods=["GET", "POST"])
@login_required
@admin_required
def admin_users():
    current_user = get_current_user()

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        role = request.form.get("role", "user").strip()

        if not username or not password or role not in ["admin", "user"]:
            flash("Please provide valid user details.")
            return redirect(url_for("admin_users"))

        conn = get_db_connection()
        existing = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

        if existing:
            conn.close()
            flash("Username already exists.")
            return redirect(url_for("admin_users"))

        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), role)
        )
        conn.commit()
        conn.close()

        flash("User created successfully.")
        return redirect(url_for("admin_users"))

    conn = get_db_connection()
    users = conn.execute("SELECT id, username, role, created_at FROM users ORDER BY id ASC").fetchall()
    conn.close()

    return render_template("admin_users.html", users=users, current_user=current_user)

@app.route("/admin/delete-user/<int:user_id>", methods=["POST"])
@login_required
@admin_required
def delete_user(user_id):
    current_user = get_current_user()

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    if user:
        if user["username"] == "admin":
            flash("Default admin cannot be deleted.")
        elif user["id"] == current_user["id"]:
            flash("You cannot delete your own account while logged in.")
        else:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            flash("User deleted successfully.")

    conn.close()
    return redirect(url_for("admin_users"))

# -----------------------------
# Start app
# -----------------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=8888, use_reloader=False)