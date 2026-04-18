from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from utils import (
    prepare_uploaded_dataframe,
    get_risk_level,
    save_bar_chart,
    save_probability_histogram,
    save_risk_pie_chart,
    save_confusion_heatmap
)
from sklearn.metrics import confusion_matrix

app = Flask(__name__)

# folders
OUTPUT_DIR = os.path.join("app", "static", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load models
lr_model = joblib.load("app/models/logistic_model.pkl")
rf_model = joblib.load("app/models/random_forest.pkl")
svm_model = joblib.load("app/models/svm_model.pkl")
tfidf = joblib.load("app/models/tfidf_vectorizer.pkl")

# optional BERT
bert_available = False
bert_predictor = None
try:
    from bert_utils import BertPredictor
    bert_predictor = BertPredictor("app/models/bert_model")
    bert_available = True
except Exception:
    bert_available = False

@app.route("/")
def home():
    model_status = {
        "Logistic Regression": "Ready",
        "Random Forest": "Ready",
        "SVM": "Ready",
        "BERT": "Ready" if bert_available else "Optional / Not Loaded"
    }

    performance = {
        "Logistic Regression": {"accuracy": 0.98, "precision": 0.98, "recall": 0.98},
        "Random Forest": {"accuracy": 0.99, "precision": 0.99, "recall": 0.99},
        "SVM": {"accuracy": 0.99, "precision": 0.99, "recall": 0.99},
        "BERT": {"accuracy": 0.95, "precision": 0.95, "recall": 0.95} if bert_available else None
    }

    return render_template(
        "index.html",
        model_status=model_status,
        performance=performance,
        bert_available=bert_available
    )

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    df = pd.read_csv(file)
    df, text_col = prepare_uploaded_dataframe(df)

    X = df["clean_text"]
    X_tfidf = tfidf.transform(X)

    # predictions
    lr_preds = lr_model.predict(X_tfidf)
    rf_preds = rf_model.predict(X_tfidf)
    svm_preds = svm_model.predict(X_tfidf)

    lr_probs = lr_model.predict_proba(X_tfidf)[:, 1] if hasattr(lr_model, "predict_proba") else [0.5] * len(df)
    rf_probs = rf_model.predict_proba(X_tfidf)[:, 1] if hasattr(rf_model, "predict_proba") else [0.5] * len(df)

    # optional bert
    bert_preds = None
    bert_probs = None
    if bert_available:
        try:
            bert_preds, bert_probs = bert_predictor.predict_batch(X.tolist()[:300])
        except Exception:
            bert_preds, bert_probs = None, None

    # ensemble logic
    ensemble_preds = []
    ensemble_probs = []
    risk_levels = []

    for i in range(len(df)):
        preds_list = [int(lr_preds[i]), int(rf_preds[i]), int(svm_preds[i])]
        attack_votes = sum(preds_list)

        ensemble_pred = 1 if attack_votes >= 2 else 0
        avg_prob = (float(lr_probs[i]) + float(rf_probs[i])) / 2

        ensemble_preds.append(ensemble_pred)
        ensemble_probs.append(avg_prob)
        risk_levels.append(get_risk_level(avg_prob))

    df["LR Prediction"] = ["Attack" if x == 1 else "Normal" for x in lr_preds]
    df["LR Probability"] = [round(float(x), 3) for x in lr_probs]
    df["RF Prediction"] = ["Attack" if x == 1 else "Normal" for x in rf_preds]
    df["RF Probability"] = [round(float(x), 3) for x in rf_probs]
    df["SVM Prediction"] = ["Attack" if x == 1 else "Normal" for x in svm_preds]
    df["Ensemble"] = ["Attack" if x == 1 else "Normal" for x in ensemble_preds]
    df["Risk Level"] = risk_levels

    # counts
    lr_attacks = int(sum(lr_preds))
    rf_attacks = int(sum(rf_preds))
    svm_attacks = int(sum(svm_preds))
    ensemble_attacks = int(sum(ensemble_preds))
    high_risk = int(sum(1 for x in risk_levels if x == "HIGH"))

    # save charts
    save_bar_chart(
        {
            "Logistic Regression": lr_attacks,
            "Random Forest": rf_attacks,
            "SVM": svm_attacks,
            "Ensemble": ensemble_attacks
        },
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

    # optional confusion matrices if label column exists
    possible_label_cols = ["label", "target", "Label", "Attack", "class", "label_encoded"]
    true_label_col = None
    for col in possible_label_cols:
        if col in df.columns:
            true_label_col = col
            break

    if true_label_col:
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

    # first 50 rows for table
    preview_df = df.head(50).copy()
    preview_rows = preview_df.reset_index().rename(columns={"index": "Connection #"}).to_dict(orient="records")

    return render_template(
        "result.html",
        total_rows=len(df),
        lr_attacks=lr_attacks,
        rf_attacks=rf_attacks,
        svm_attacks=svm_attacks,
        ensemble_attacks=ensemble_attacks,
        high_risk=high_risk,
        preview_rows=preview_rows,
        show_confusion=(true_label_col is not None),
        bert_available=bert_available
    )

if __name__ == "__main__":
    app.run(debug=True, port=8888)