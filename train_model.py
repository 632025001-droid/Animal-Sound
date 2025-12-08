# train_model.py
import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from db_utils import get_session, AudioSample, BASE_DIR

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset_from_db():
    session = get_session()
    samples = session.query(AudioSample).all()
    X = []
    y = []
    missing = []
    for s in samples:
        if s.mfcc_path and Path(s.mfcc_path).exists():
            vec = np.load(s.mfcc_path)
            X.append(vec)
            y.append(s.label)
        else:
            missing.append(s.filename)
    session.close()
    print(f"[INFO] Loaded {len(X)} feature vectors, missing {len(missing)} files.")
    if missing:
        print("[WARN] Missing files (not in DB features):")
        for m in missing[:10]:
            print(" -", m)
    return np.array(X), np.array(y)

def train_and_save(model_name="rf_model.joblib"):
    X, y = load_dataset_from_db()
    if len(X) == 0:
        raise RuntimeError("No features found. Run feature_extraction.py first.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("[RESULTS] Classification report:")
    print(classification_report(y_test, preds))
    print("[RESULTS] Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    model_path = MODEL_DIR / model_name
    joblib.dump(clf, str(model_path))
    print(f"[OK] Model saved to {model_path}")
    return model_path

if __name__ == "__main__":
    train_and_save()
