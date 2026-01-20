from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import joblib

from .config import DATA_PATH, MODEL_DIR, MODEL_PATH, TARGET_COL, FEATURE_ORDER

METRICS_PATH = MODEL_PATH.with_suffix(".metrics.txt")

def main():
    df = pd.read_csv(DATA_PATH)

    # Zorg dat kolommen kloppen en in juiste volgorde staan
    missing = [c for c in FEATURE_ORDER + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[FEATURE_ORDER].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)

    print(classification_report(y_test, pred, digits=4))
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    print(
        "Accuracy: "
        f"{accuracy:.4f} | Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | F1: {f1:.4f}"
    )
    print("ROC-AUC:", roc_auc)

    timestamp = datetime.now().isoformat(timespec="seconds")
    metrics_line = (
        f"{timestamp} | samples={len(y_test)} | "
        f"accuracy={accuracy:.4f} | precision={precision:.4f} | "
        f"recall={recall:.4f} | f1={f1:.4f} | roc_auc={roc_auc:.4f}"
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(metrics_line + "\n")
    print(f"Metrics logged -> {METRICS_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

if __name__ == "__main__":
    main()
