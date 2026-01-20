import joblib
import pandas as pd

from .config import MODEL_PATH, FEATURE_ORDER

def load_model():
    return joblib.load(MODEL_PATH)

def predict_from_features(model, features: dict, threshold: float = 0.5) -> dict:
    # DataFrame met juiste kolomvolgorde
    X = pd.DataFrame([[features[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

    proba_fake = float(model.predict_proba(X)[0][1])
    pred_fake = int(proba_fake >= threshold)

    return {
        "prediction": "FAKE" if pred_fake else "REAL",
        "fake_probability": proba_fake,
        "threshold": float(threshold),
    }
