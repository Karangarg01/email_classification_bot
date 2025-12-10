import os
from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

EMAILS_DIR = "emails_raw"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "email_classifier.joblib")


def infer_label_from_filename(filename: str) -> str:
    """
    Convert 'login_1.txt' -> 'login_issue'
            'payment_2.txt' -> 'payment_issue'
            'order_1.txt' -> 'order_issue'
            'account_1.txt' -> 'account_update'
            'general_1.txt' -> 'general_query'
    """
    name, _ = os.path.splitext(filename)
    if name.startswith("login"):
        return "login_issue"
    if name.startswith("payment"):
        return "payment_issue"
    if name.startswith("order"):
        return "order_issue"
    if name.startswith("account"):
        return "account_update"
    if name.startswith("general"):
        return "general_query"
    return "unknown"


def load_email_dataset() -> pd.DataFrame:
    """
    Reads all .txt files from emails_raw/ and returns
    a DataFrame with columns: ['filename', 'text', 'label']
    """
    records: List[Tuple[str, str, str]] = []

    if not os.path.exists(EMAILS_DIR):
        raise FileNotFoundError(f"Folder '{EMAILS_DIR}' not found.")

    for fname in os.listdir(EMAILS_DIR):
        if not fname.lower().endswith(".txt"):
            continue

        path = os.path.join(EMAILS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        label = infer_label_from_filename(fname)
        records.append((fname, text, label))

    df = pd.DataFrame(records, columns=["filename", "text", "label"])
    return df


def train_and_save_model(test_size: float = 0.5, random_state: int = 42):
    """
    Train a simple text classifier (TF-IDF + Logistic Regression)
    and save the trained pipeline to models/email_classifier.joblib
    """
    df = load_email_dataset()
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Build pipeline: TF-IDF -> Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    print("🔹 Training classifier...")
    pipeline.fit(X_train, y_train)

    print("\n🔹 Evaluating on test set:")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the pipeline
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")


def load_model():
    """
    Load the trained pipeline from disk.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train the model first."
        )
    return joblib.load(MODEL_PATH)


def predict_category(text: str) -> str:
    """
    Predict the category/label for a single email text.
    """
    model = load_model()
    return model.predict([text])[0]