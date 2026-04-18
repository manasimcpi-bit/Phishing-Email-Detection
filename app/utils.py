import re
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_uploaded_dataframe(df):
    df = df.copy()

    possible_text_cols = ["text", "email", "email_text", "message", "Email Text", "text_combined"]
    text_col = None

    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        # if only one column, use first column
        text_col = df.columns[0]

    df["clean_text"] = df[text_col].astype(str).apply(clean_text)
    return df, text_col

def get_risk_level(prob):
    if prob >= 0.80:
        return "HIGH"
    elif prob >= 0.50:
        return "MEDIUM"
    elif prob >= 0.20:
        return "LOW"
    return "NORMAL"

def save_bar_chart(model_counts, output_path):
    plt.figure(figsize=(8, 5))
    plt.bar(model_counts.keys(), model_counts.values())
    plt.title("Attack Detections by Model")
    plt.ylabel("Number of Attack Detections")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_probability_histogram(lr_probs, rf_probs, output_path):
    plt.figure(figsize=(8, 5))
    plt.hist(lr_probs, bins=20, alpha=0.6, label="Logistic Regression")
    plt.hist(rf_probs, bins=20, alpha=0.6, label="Random Forest")
    plt.title("Attack Probability Distribution")
    plt.xlabel("Attack Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_risk_pie_chart(risk_levels, output_path):
    counts = pd.Series(risk_levels).value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    plt.title("Risk Category Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_confusion_heatmap(cm, title, output_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    labels = np.unique([0, 1])
    plt.xticks(labels, ["Normal", "Attack"])
    plt.yticks(labels, ["Normal", "Attack"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()