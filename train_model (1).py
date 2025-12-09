# train_model.py
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from db_utils import get_session, AudioSample

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_dataset_from_db():
    session = get_session()
    samples = session.query(AudioSample).all()
    X = []
    y = []
    missing = 0
    for s in samples:
        if s.mfcc_path and os.path.exists(s.mfcc_path):
            vec = np.load(s.mfcc_path)
            X.append(vec)
            y.append(s.label)
        else:
            missing += 1
    session.close()
    print(f"Loaded {len(X)} samples, missing {missing}")
    return np.array(X), np.array(y)

def train_and_save():
    X, y = load_dataset_from_db()
    if len(X) == 0:
        raise RuntimeError("No features found. Run feature_extraction.py first.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Classification report:\n", classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    model_path = os.path.join(MODEL_DIR, "rf_model.joblib")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save()
